#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/original_kkt_gmres.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
void dgesv_(const int*, const int*, double*, const int*, int*, double*,
            const int*, int*);
}

namespace lazyipdca::qcqp {
namespace {

void add_row_t(const SparseRow& row, double c, Vec& y) {
  for (const auto& [j,a] : row.entries) y[j] += c*a;
}

double row_dot(const SparseRow& row, const Vec& x) {
  double s=0.0; for (const auto& [j,a] : row.entries) s += a*x[j]; return s;
}

bool finite_vec(const Vec& x) {
  for (double v : x) if (!std::isfinite(v)) return false;
  return true;
}

double max_abs(const Vec& x) {
  double m=0.0; for (double v : x) m=std::max(m,std::abs(v)); return m;
}

bool fit_free_equality_duals_at_state(const ConvexSubproblem& p,
                                      const Vec& stationarity_without_lambda,
                                      double regularization,
                                      Vec& lambda,Vec& residual) {
  const int m=static_cast<int>(p.equalities.rows.size());
  const int n=p.nvar;
  if(static_cast<int>(stationarity_without_lambda.size())!=n) return false;
  if(m==0) {
    lambda.clear(); residual=stationarity_without_lambda; return true;
  }
  std::vector<double> scale(static_cast<std::size_t>(m),1.0);
  for(int i=0;i<m;++i) {
    double n2=0.0;
    for(const auto& [j,a]:p.equalities.rows[static_cast<std::size_t>(i)].entries) {
      if(j<0 || j>=n || !std::isfinite(a)) return false;
      n2+=a*a;
    }
    if(!(n2>0.0) || !std::isfinite(n2)) return false;
    scale[static_cast<std::size_t>(i)]=1.0/std::sqrt(n2);
  }
  std::vector<double> gram(static_cast<std::size_t>(m)*m,0.0);
  Vec rhs(static_cast<std::size_t>(m),0.0);
  for(int i=0;i<m;++i) {
    const auto& ri=p.equalities.rows[static_cast<std::size_t>(i)];
    rhs[static_cast<std::size_t>(i)]=
        -scale[static_cast<std::size_t>(i)]*row_dot(
            ri,stationarity_without_lambda);
    for(int k=0;k<=i;++k) {
      const auto& rk=p.equalities.rows[static_cast<std::size_t>(k)];
      std::size_t a=0,b=0;
      double value=0.0;
      while(a<ri.entries.size() && b<rk.entries.size()) {
        if(ri.entries[a].first<rk.entries[b].first) { ++a; continue; }
        if(rk.entries[b].first<ri.entries[a].first) { ++b; continue; }
        value+=ri.entries[a].second*rk.entries[b].second;
        ++a; ++b;
      }
      value*=scale[static_cast<std::size_t>(i)]*
             scale[static_cast<std::size_t>(k)];
      gram[static_cast<std::size_t>(i+k*m)]=value;
      gram[static_cast<std::size_t>(k+i*m)]=value;
    }
    gram[static_cast<std::size_t>(i+i*m)]+=
        std::max(1e-14,regularization);
  }
  std::vector<int> pivots(static_cast<std::size_t>(m),0);
  const int nrhs=1;
  int info=0;
  dgesv_(&m,&nrhs,gram.data(),&m,pivots.data(),rhs.data(),&m,&info);
  if(info!=0 || !finite_vec(rhs)) return false;
  lambda.resize(static_cast<std::size_t>(m));
  residual=stationarity_without_lambda;
  for(int i=0;i<m;++i) {
    lambda[static_cast<std::size_t>(i)]=
        scale[static_cast<std::size_t>(i)]*rhs[static_cast<std::size_t>(i)];
    add_row_t(p.equalities.rows[static_cast<std::size_t>(i)],
              lambda[static_cast<std::size_t>(i)],residual);
  }
  return finite_vec(lambda) && finite_vec(residual);
}

double minimize_direction_inf_norm(const Vec& residual,const Vec& direction) {
  if(residual.size()!=direction.size() || residual.empty()) return 0.0;
  auto value=[&](double alpha) {
    double norm=0.0;
    for(std::size_t i=0;i<residual.size();++i)
      norm=std::max(norm,std::abs(residual[i]+alpha*direction[i]));
    return norm;
  };
  double lo=0.0,hi=1.0;
  for(int iteration=0;iteration<64;++iteration) {
    const double left=(2.0*lo+hi)/3.0;
    const double right=(lo+2.0*hi)/3.0;
    if(value(left)<=value(right)) hi=right;
    else lo=left;
  }
  const double candidates[]={0.0,lo,0.5*(lo+hi),hi,1.0};
  double best_alpha=0.0;
  double best_value=value(0.0);
  for(double alpha:candidates) {
    const double candidate=value(alpha);
    if(candidate<best_value) {
      best_value=candidate;
      best_alpha=alpha;
    }
  }
  return best_alpha;
}

double max_step_positive(const Vec& x,const Vec& dx,double fraction) {
  double a=1.0;
  for (std::size_t i=0;i<x.size();++i)
    if (dx[i] < 0.0) a=std::min(a,-fraction*x[i]/dx[i]);
  return std::max(0.0,std::min(1.0,a));
}

struct StateEval {
  Vec rdual, req, phi, rpri;
  struct SparseJacobian {
    std::vector<std::size_t> row_offsets;
    std::vector<int> columns;
    Vec values;
  } J;
  double dual_inf=0.0, eq_inf=0.0, pri_inf=0.0, ineq_inf=0.0, mu=0.0;
};

// Sorted structural support of an inequality model.  Models carrying a
// prefix shift or a low-rank term remain on the generic path because those
// terms are dense by construction.  Eligibility is independently checked
// against the subproblem dimension so malformed models can never turn a
// bounds-checked generic failure into an out-of-bounds fast-path access.
struct ModelSupport {
  std::vector<int> index;
  bool eligible=false;
};

ModelSupport build_model_support(const ConvexQuadraticModel& model,
                                 int expected_dimension) {
  ModelSupport support;
  if(expected_dimension<0 ||
     static_cast<int>(model.linear.size())!=expected_dimension ||
     model.base_hessian.n!=expected_dimension ||
     !model.low_rank.empty() ||
     (model.shift_dimension>0 && model.scalar_shift!=0.0))
    return support;

  std::vector<unsigned char> seen(
      static_cast<std::size_t>(expected_dimension),0u);
  for(int j=0;j<expected_dimension;++j) {
    const double value=model.linear[static_cast<std::size_t>(j)];
    if(!std::isfinite(value)) return ModelSupport{};
    if(value!=0.0) seen[static_cast<std::size_t>(j)]=1u;
  }
  for(const auto& entry:model.base_hessian.entries) {
    if(entry.i<0 || entry.j<0 || entry.i>=expected_dimension ||
       entry.j>=expected_dimension || entry.i>entry.j ||
       !std::isfinite(entry.v))
      return ModelSupport{};
    if(entry.v!=0.0) {
      seen[static_cast<std::size_t>(entry.i)]=1u;
      seen[static_cast<std::size_t>(entry.j)]=1u;
    }
  }
  support.index.reserve(static_cast<std::size_t>(expected_dimension));
  for(int j=0;j<expected_dimension;++j)
    if(seen[static_cast<std::size_t>(j)]) support.index.push_back(j);
  support.eligible=true;
  return support;
}

std::vector<ModelSupport> build_model_supports(
    const std::vector<ConvexQuadraticModel>& models,int dimension) {
  std::vector<ModelSupport> supports;
  supports.reserve(models.size());
  for(const auto& model:models)
    supports.push_back(build_model_support(model,dimension));
  return supports;
}

double model_value_support(const ConvexQuadraticModel& model,const Vec& y,
                           const ModelSupport& support,
                           InnerResult* diagnostics) {
  if(static_cast<int>(y.size())!=static_cast<int>(model.linear.size()) ||
     model.base_hessian.n!=static_cast<int>(y.size()))
    throw std::invalid_argument("model_value_support: dimension mismatch");
  if(diagnostics) ++diagnostics->lazyqp_model_support_value_evaluations;
  double linear_sum=0.0;
  for(int j:support.index)
    linear_sum+=model.linear[static_cast<std::size_t>(j)]*
                y[static_cast<std::size_t>(j)];
  return model.constant+linear_sum+0.5*model.base_hessian.quad(y);
}

// Components outside support are intentionally left untouched.  Every
// consumer of this workspace is restricted to exactly the same support.
void model_gradient_support_into(const ConvexQuadraticModel& model,const Vec& y,
                                 Vec& gradient,const ModelSupport& support,
                                 InnerResult* diagnostics) {
  const int n=static_cast<int>(model.linear.size());
  if(static_cast<int>(y.size())!=n || model.base_hessian.n!=n)
    throw std::invalid_argument(
        "model_gradient_support_into: dimension mismatch");
  if(diagnostics) ++diagnostics->lazyqp_model_support_gradient_evaluations;
  gradient.resize(static_cast<std::size_t>(n));
  for(int j:support.index) gradient[static_cast<std::size_t>(j)]=0.0;
  for(const auto& entry:model.base_hessian.entries) {
    gradient[static_cast<std::size_t>(entry.i)]+=
        entry.v*y[static_cast<std::size_t>(entry.j)];
    if(entry.i!=entry.j)
      gradient[static_cast<std::size_t>(entry.j)]+=
          entry.v*y[static_cast<std::size_t>(entry.i)];
  }
  for(int j:support.index)
    gradient[static_cast<std::size_t>(j)]+=
        model.linear[static_cast<std::size_t>(j)];
}

// Allocation-free counterpart of ConvexQuadraticModel::gradient.  Keep the
// operation order identical to the dense public helper: sparse base Hessian,
// dense linear term, prefix shift, then the low-rank terms in storage order.
// One buffer is shared by every inequality row and every state evaluation in
// a LazyQP solve; only the nonzeros are retained in StateEval::J.
void model_gradient_into(const ConvexQuadraticModel& model, const Vec& y,
                         Vec& gradient) {
  const int n=static_cast<int>(model.linear.size());
  if(static_cast<int>(y.size())!=n || model.base_hessian.n!=n)
    throw std::invalid_argument("model_gradient_into: dimension mismatch");
  gradient.resize(static_cast<std::size_t>(n));
  std::fill(gradient.begin(),gradient.end(),0.0);
  for(const auto& entry:model.base_hessian.entries) {
    gradient[static_cast<std::size_t>(entry.i)] +=
        entry.v*y[static_cast<std::size_t>(entry.j)];
    if(entry.i!=entry.j)
      gradient[static_cast<std::size_t>(entry.j)] +=
          entry.v*y[static_cast<std::size_t>(entry.i)];
  }
  for(int j=0;j<n;++j)
    gradient[static_cast<std::size_t>(j)] +=
        model.linear[static_cast<std::size_t>(j)];
  for(int j=0;j<model.shift_dimension;++j)
    gradient[static_cast<std::size_t>(j)] +=
        model.scalar_shift*y[static_cast<std::size_t>(j)];
  for(const auto& term:model.low_rank) {
    const double coefficient=term.beta*dot(term.direction,y);
    for(int j=0;j<n;++j)
      gradient[static_cast<std::size_t>(j)] +=
          coefficient*term.direction[static_cast<std::size_t>(j)];
  }
}

template<class Function>
void for_each_jacobian_entry(const StateEval::SparseJacobian& J, int row,
                             Function&& function) {
  const std::size_t begin=J.row_offsets[static_cast<std::size_t>(row)];
  const std::size_t end=J.row_offsets[static_cast<std::size_t>(row+1)];
  for(std::size_t k=begin;k<end;++k)
    function(J.columns[k],J.values[k]);
}

double jacobian_row_dot(const StateEval::SparseJacobian& J, int row,
                        const Vec& x) {
  double sum=0.0;
  for_each_jacobian_entry(J,row,[&](int column,double value) {
    sum += value*x[static_cast<std::size_t>(column)];
  });
  return sum;
}

StateEval evaluate_state(const ConvexSubproblem& p,const Vec& y,const Vec& lambda,
                         const Vec& z,const Vec& s,
                         Vec& inequality_gradient_work,
                         const Vec* precomputed_phi=nullptr,
                         const std::vector<ModelSupport>* supports=nullptr,
                         InnerResult* diagnostics=nullptr) {
  const int me=(int)p.equalities.rows.size(), mi=(int)p.inequalities.size();
  StateEval e;
  e.rdual=p.objective.gradient(y);
  e.req.assign(me,0.0); e.phi.assign(mi,0.0); e.rpri.assign(mi,0.0);
  e.J.row_offsets.resize(static_cast<std::size_t>(mi+1));
  const std::size_t expected_jacobian_nnz=
      static_cast<std::size_t>(mi)*std::min<std::size_t>(y.size(),16u);
  e.J.columns.reserve(expected_jacobian_nnz);
  e.J.values.reserve(expected_jacobian_nnz);
  e.J.row_offsets[0]=0;
  for (int r=0;r<me;++r) {
    e.req[r]=row_dot(p.equalities.rows[r],y)-p.equalities.rhs[r];
    add_row_t(p.equalities.rows[r],lambda[r],e.rdual);
  }
  const bool use_precomputed=precomputed_phi &&
      static_cast<int>(precomputed_phi->size())==mi;
  for (int i=0;i<mi;++i) {
    const ModelSupport* support=
        supports && supports->size()==p.inequalities.size() &&
        (*supports)[static_cast<std::size_t>(i)].eligible
            ? &(*supports)[static_cast<std::size_t>(i)] : nullptr;
    e.phi[i]=use_precomputed ? (*precomputed_phi)[i]
        : support ? model_value_support(
              p.inequalities[static_cast<std::size_t>(i)],y,*support,
              diagnostics)
                  : p.inequalities[i].value(y);
    if(support)
      model_gradient_support_into(
          p.inequalities[static_cast<std::size_t>(i)],y,
          inequality_gradient_work,*support,diagnostics);
    else
      model_gradient_into(p.inequalities[static_cast<std::size_t>(i)],y,
                          inequality_gradient_work);
    const auto add_component=[&](int j) {
      const double value=inequality_gradient_work[static_cast<std::size_t>(j)];
      if(value!=0.0) {
        e.J.columns.push_back(j);
        e.J.values.push_back(value);
        e.rdual[static_cast<std::size_t>(j)] += z[i]*value;
      }
    };
    if(support) {
      for(int j:support->index) add_component(j);
    } else {
      for(int j=0;j<static_cast<int>(inequality_gradient_work.size());++j)
        add_component(j);
    }
    e.J.row_offsets[static_cast<std::size_t>(i+1)]=e.J.columns.size();
    e.rpri[i]=e.phi[i]+s[i];
    e.ineq_inf=std::max(e.ineq_inf,std::max(0.0,e.phi[i]));
  }
  e.dual_inf=max_abs(e.rdual); e.eq_inf=max_abs(e.req); e.pri_inf=max_abs(e.rpri);
  e.mu=mi ? dot(s,z)/(double)mi : 0.0;
  return e;
}

double complementarity_inf(const Vec& slacks,const Vec& duals) {
  double value=0.0;
  for(std::size_t i=0;i<slacks.size() && i<duals.size();++i)
    value=std::max(value,std::abs(slacks[i]*duals[i]));
  return value;
}

double forcing_residual_inf(const StateEval& state,const Vec& slacks,
                            const Vec& duals) {
  return std::max({state.dual_inf,state.eq_inf,state.pri_inf,state.ineq_inf,
                   complementarity_inf(slacks,duals)});
}

// Independent centered reference at the model origin.  This is used only to
// recertify a reduced forcing result after exact affine expansion.  Ignoring a
// possibly transported warm start makes the reference deterministic; the
// kernel itself additionally contracts from the actual accepted initial
// state, so warm starts cannot weaken the test.
double centered_origin_forcing_residual(const ConvexSubproblem& p,
                                        const Settings& settings) {
  const int n=p.nvar;
  const int me=static_cast<int>(p.equalities.rows.size());
  const int mi=static_cast<int>(p.inequalities.size());
  Vec y(static_cast<std::size_t>(n),0.0);
  Vec lambda(static_cast<std::size_t>(me),0.0);
  Vec z(static_cast<std::size_t>(mi),1.0);
  Vec s(static_cast<std::size_t>(mi),1.0);
  Vec gradient_work(static_cast<std::size_t>(n),0.0);
  const double gscale=std::max(
      1e-2,std::min(1.0,max_abs(p.objective.gradient(y))));
  const double target_mu=std::max(
      {settings.lazyqp_barrier_mu0,gscale,
       100.0*settings.inner_tolerance});
  const double slack_push=std::max(1e-4,10.0*settings.inner_tolerance);
  for(int i=0;i<mi;++i) {
    const double phi=p.inequalities[static_cast<std::size_t>(i)].value(y);
    const double raw=std::max(0.0,-phi);
    s[static_cast<std::size_t>(i)]=std::max(raw,slack_push);
    z[static_cast<std::size_t>(i)]=std::clamp(
        target_mu/std::max(s[static_cast<std::size_t>(i)],target_mu),
        settings.minimum_positive,10.0);
  }
  const StateEval state=evaluate_state(
      p,y,lambda,z,s,gradient_work);
  return forcing_residual_inf(state,s,z);
}

struct NewtonStep { Vec dy,dlambda,dz,ds; };

struct TrialState {
  bool ok=false;
  Vec y,lambda,z,s;
  StateEval eval;
  double alpha_pri=0.0, alpha_lambda=0.0, alpha_dual=0.0;
  int backtracks=0;
  double nonlinear_defect_inf=0.0;
  double slack_correction_min=1.0;
};

struct FactorCounterSnapshot {
  int analyses=0;
  int reuses=0;
  int invalidations=0;
  int scatter_builds=0;
  int scatter_reuses=0;
  int scatter_invalidations=0;
};

FactorCounterSnapshot factor_counters(
    const std::unique_ptr<SymmetricFactorization>& factor) {
  if(!factor) return {};
  return {factor->symbolic_analyses(),factor->symbolic_reuses(),
          factor->symbolic_invalidations(),factor->numeric_scatter_builds(),
          factor->numeric_scatter_reuses(),
          factor->numeric_scatter_invalidations()};
}

struct FactorStatisticsDelta {
  InnerResult& out;
  const std::unique_ptr<SymmetricFactorization>& full;
  const std::unique_ptr<SymmetricFactorization>& reduced;
  const std::unique_ptr<SymmetricFactorization>& hybrid;
  FactorCounterSnapshot full_before;
  FactorCounterSnapshot reduced_before;
  FactorCounterSnapshot hybrid_before;
  bool applied=false;

  FactorStatisticsDelta(
      InnerResult& result,
      const std::unique_ptr<SymmetricFactorization>& full_factor,
      const std::unique_ptr<SymmetricFactorization>& reduced_factor,
      const std::unique_ptr<SymmetricFactorization>& hybrid_factor)
      : out(result),full(full_factor),reduced(reduced_factor),hybrid(hybrid_factor),
        full_before(factor_counters(full_factor)),
        reduced_before(factor_counters(reduced_factor)),
        hybrid_before(factor_counters(hybrid_factor)) {}

  void add_delta(const std::unique_ptr<SymmetricFactorization>& factor,
                 const FactorCounterSnapshot& before) {
    if(!factor) return;
    out.symbolic_analyses +=
        std::max(0,factor->symbolic_analyses()-before.analyses);
    out.symbolic_reuses +=
        std::max(0,factor->symbolic_reuses()-before.reuses);
    out.symbolic_invalidations +=
        std::max(0,factor->symbolic_invalidations()-before.invalidations);
    out.numeric_scatter_builds += std::max(
        0,factor->numeric_scatter_builds()-before.scatter_builds);
    out.numeric_scatter_reuses += std::max(
        0,factor->numeric_scatter_reuses()-before.scatter_reuses);
    out.numeric_scatter_invalidations += std::max(
        0,factor->numeric_scatter_invalidations()-
              before.scatter_invalidations);
  }

  void apply() {
    if(applied) return;
    add_delta(full,full_before);
    add_delta(reduced,reduced_before);
    add_delta(hybrid,hybrid_before);
    applied=true;
  }

  ~FactorStatisticsDelta() { apply(); }
};

// Nonlinear primal-dual trial for an already-convex QCQP.
//
// The slack is an independent primal variable.  This is essential: forcing
// s^+ = -g(y^+) at every trial turns the IPM into a strictly-feasible method
// and can drive alpha to zero on an active, strongly curved quadratic row.
// We therefore take the Newton slack step first and apply only as much of the
// exact quadratic second-order correction as positivity permits.  The
// unabsorbed quadratic remainder is left in r_pri and is corrected by the
// next Newton step.  Exact nonlinear residuals are always evaluated before a
// trial is accepted.
TrialState make_trial(const ConvexSubproblem& p,const Vec& y,const Vec& lambda,
                      const Vec& z,const Vec& s,const StateEval& e,
                      const NewtonStep& d,double ap,double al,double ad,
                      double min_positive,double fraction_to_boundary,
                      Vec& inequality_gradient_work,
                      const std::vector<ModelSupport>* supports=nullptr,
                      InnerResult* diagnostics=nullptr) {
  const int mi=(int)p.inequalities.size();
  TrialState t; t.alpha_pri=ap; t.alpha_lambda=al; t.alpha_dual=ad;
  t.y=y; t.lambda=lambda; t.z=z; t.s=s;
  axpy(ap,d.dy,t.y); axpy(al,d.dlambda,t.lambda);
  for (int i=0;i<mi;++i) {
    t.z[i] += ad*d.dz[i];
    if(!(t.z[i]>0.0) || !std::isfinite(t.z[i])) return t;
    // minimum_positive is a numerical floor used by the KKT algebra, not an
    // extra optimization constraint.  Inactive multipliers are allowed to
    // approach zero and are clipped only after positivity has been checked.
    t.z[i]=std::max(min_positive,t.z[i]);
  }
  if(!finite_vec(t.y)||!finite_vec(t.lambda)||!finite_vec(t.z)) return t;

  const double frac=std::clamp(fraction_to_boundary,0.0,0.999999999999);
  Vec phi_trial(mi,0.0);
  for (int i=0;i<mi;++i) {
    double s_linear=s[i]+ap*d.ds[i];
    if(!(s_linear>0.0) || !std::isfinite(s_linear)) return t;
    s_linear=std::max(min_positive,s_linear);

    const ModelSupport* support=
        supports && supports->size()==p.inequalities.size() &&
        (*supports)[static_cast<std::size_t>(i)].eligible
            ? &(*supports)[static_cast<std::size_t>(i)] : nullptr;
    phi_trial[i]=support
        ? model_value_support(p.inequalities[static_cast<std::size_t>(i)],t.y,
                              *support,diagnostics)
        : p.inequalities[i].value(t.y);
    const double jd=jacobian_row_dot(e.J,i,d.dy);
    const double q=phi_trial[i]-e.phi[i]-ap*jd;
    if(!std::isfinite(phi_trial[i]) || !std::isfinite(q)) return t;

    // Use the exact second-order slack correction whenever it remains safely
    // inside s>0.  Otherwise damp only the correction, not the primal Newton
    // step.  This preserves the IPM's ability to use infeasible nonlinear
    // iterates and avoids the v0.4.4 curvature wall.
    double gamma=1.0;
    if(q>0.0) {
      const double floor=std::max(min_positive,(1.0-frac)*s_linear);
      if(s_linear-q<=floor) {
        gamma=(s_linear>floor) ? (s_linear-floor)/q : 0.0;
        gamma=std::clamp(gamma,0.0,1.0);
      }
    }
    t.s[i]=s_linear-gamma*q;
    if(!(t.s[i]>0.0) || !std::isfinite(t.s[i])) return t;
    t.s[i]=std::max(min_positive,t.s[i]);
    t.nonlinear_defect_inf=std::max(t.nonlinear_defect_inf,std::abs((1.0-gamma)*q));
    t.slack_correction_min=std::min(t.slack_correction_min,gamma);
  }
  t.eval=evaluate_state(p,t.y,t.lambda,t.z,t.s,inequality_gradient_work,
                        &phi_trial,supports,diagnostics);
  t.ok=finite_vec(t.eval.rdual) && finite_vec(t.eval.req) && finite_vec(t.eval.rpri);
  return t;
}

bool valid_quadratic_model(const ConvexQuadraticModel& model, int n) {
  if (static_cast<int>(model.linear.size()) != n ||
      model.base_hessian.n != n || model.shift_dimension < 0 ||
      model.shift_dimension > n || !std::isfinite(model.constant) ||
      !std::isfinite(model.scalar_shift))
    return false;
  try {
    model.base_hessian.validate();
  } catch (const std::exception&) {
    return false;
  }
  if (!finite_vec(model.linear)) return false;
  for (const auto& term : model.low_rank)
    if (!std::isfinite(term.beta) ||
        static_cast<int>(term.direction.size()) != n ||
        !finite_vec(term.direction))
      return false;
  return true;
}

Vec transform_transpose_product(const std::vector<SparseRow>& transform,
                                int reduced_n, const Vec& full) {
  Vec reduced(static_cast<std::size_t>(reduced_n), 0.0);
  for (int j = 0; j < static_cast<int>(transform.size()); ++j)
    for (const auto& [k, value] :
         transform[static_cast<std::size_t>(j)].entries)
      reduced[static_cast<std::size_t>(k)] +=
          value * full[static_cast<std::size_t>(j)];
  return reduced;
}

// Sparse arithmetic for affine composition/audits. Supports are rebuilt from
// the current immutable model, never reused across a changed majorant.
Vec supported_transpose_gradient(const ConvexQuadraticModel& model,
    const Vec& x, const std::vector<SparseRow>& transform, int reduced_n,
    const ModelSupport& support) {
  if(!support.eligible)
    return transform_transpose_product(transform,reduced_n,model.gradient(x));
  Vec work(model.linear.size(),0.0), result(reduced_n,0.0);
  model_gradient_support_into(model,x,work,support,nullptr);
  for(int j:support.index)
    for(const auto& [k,a]:transform[static_cast<std::size_t>(j)].entries)
      result[static_cast<std::size_t>(k)]+=a*work[static_cast<std::size_t>(j)];
  return result;
}

void supported_gradient_axpy(const ConvexQuadraticModel& model,
    const Vec& x, double multiplier, Vec& destination,
    const ModelSupport& support, Vec& workspace) {
  if(!support.eligible) {
    axpy(multiplier,model.gradient(x),destination);
    return;
  }
  model_gradient_support_into(model,x,workspace,support,nullptr);
  for(int j:support.index)
    destination[static_cast<std::size_t>(j)]+=multiplier*workspace[static_cast<std::size_t>(j)];
}

double supported_value(const ConvexQuadraticModel& model,const Vec& x,
                       const ModelSupport& support) {
  return support.eligible ? model_value_support(model,x,support,nullptr) : model.value(x);
}

void add_polynomial_quadratic(
    std::map<std::pair<int, int>, double>& upper, int i, int j,
    double polynomial_coefficient) {
  if (polynomial_coefficient == 0.0) return;
  if (i == j) {
    upper[{i, i}] += 2.0 * polynomial_coefficient;
  } else {
    if (i > j) std::swap(i, j);
    // In 0.5*z^T*H*z an upper off-diagonal entry H_ij contributes
    // H_ij*z_i*z_j.  Ordered products that map to the same monomial are
    // intentionally accumulated here.
    upper[{i, j}] += polynomial_coefficient;
  }
}

bool compose_quadratic_model(
    const ConvexQuadraticModel& full, const Vec& offset,
    const std::vector<SparseRow>& transform, int reduced_n,
    ConvexQuadraticModel& reduced, std::string& message, bool sparse_checks) {
  const int full_n = static_cast<int>(transform.size());
  if (!valid_quadratic_model(full, full_n) ||
      static_cast<int>(offset.size()) != full_n) {
    message = "invalid full quadratic model in affine reduction";
    return false;
  }

  try {
    const auto support=sparse_checks ? build_model_support(full,full_n) : ModelSupport{};
    reduced.constant = supported_value(full,offset,support);
    reduced.linear = supported_transpose_gradient(full,offset,transform,reduced_n,support);
  } catch (const std::exception& e) {
    message = std::string("affine quadratic composition failed: ") + e.what();
    return false;
  }
  reduced.base_hessian.n = reduced_n;
  reduced.base_hessian.entries.clear();
  reduced.shift_dimension = 0;
  reduced.scalar_shift = 0.0;
  reduced.low_rank.clear();

  std::map<std::pair<int, int>, double> upper;
  for (const auto& entry : full.base_hessian.entries) {
    const auto& left = transform[static_cast<std::size_t>(entry.i)].entries;
    const auto& right = transform[static_cast<std::size_t>(entry.j)].entries;
    // A stored diagonal represents 0.5*H_ii*y_i^2; a stored off-diagonal
    // represents H_ij*y_i*y_j in the quadratic value.
    const double polynomial_scale =
        entry.i == entry.j ? 0.5 * entry.v : entry.v;
    for (const auto& [i, ai] : left)
      for (const auto& [j, aj] : right)
        add_polynomial_quadratic(upper, i, j,
                                 polynomial_scale * ai * aj);
  }
  // A prefix scalar shift is also composed, rather than retained as a scalar
  // identity in the new coordinates: eliminated lift rows make T^T*T
  // non-diagonal.
  if (full.scalar_shift != 0.0) {
    for (int original = 0; original < full.shift_dimension; ++original) {
      const auto& row = transform[static_cast<std::size_t>(original)].entries;
      for (const auto& [i, ai] : row)
        for (const auto& [j, aj] : row)
          add_polynomial_quadratic(
              upper, i, j, 0.5 * full.scalar_shift * ai * aj);
    }
  }
  reduced.base_hessian.entries.reserve(upper.size());
  for (const auto& [coordinate, value] : upper)
    if (value != 0.0)
      reduced.base_hessian.entries.push_back(
          {coordinate.first, coordinate.second, value});

  // Low-rank curvature remains low-rank under congruence.  Its affine offset
  // is already present in reduced.constant and reduced.linear, evaluated at
  // the common offset above.
  for (const auto& term : full.low_rank) {
    LowRankTerm mapped;
    mapped.beta = term.beta;
    mapped.direction = transform_transpose_product(
        transform, reduced_n, term.direction);
    if (!finite_vec(mapped.direction)) {
      message = "nonfinite mapped low-rank direction";
      return false;
    }
    if (max_abs(mapped.direction) != 0.0)
      reduced.low_rank.push_back(std::move(mapped));
  }

  if (!valid_quadratic_model(reduced, reduced_n)) {
    message = "nonfinite reduced quadratic model after affine composition";
    return false;
  }
  return true;
}

bool close_scalar(double a, double b, double multiplier = 32768.0) {
  const double scale = std::max({1.0, std::abs(a), std::abs(b)});
  return std::isfinite(a) && std::isfinite(b) &&
      std::abs(a - b) <=
          multiplier * std::numeric_limits<double>::epsilon() * scale;
}

bool validate_affine_reduction(
    const ConvexSubproblem& full,
    const ExactAffineSubproblemReduction& reduction,
    std::string& message, bool sparse_checks) {
  const int reduced_n = reduction.reduced.nvar;
  std::vector<Vec> probes;
  probes.emplace_back(static_cast<std::size_t>(reduced_n), 0.0);
  Vec deterministic(static_cast<std::size_t>(reduced_n), 0.0);
  for (int j = 0; j < reduced_n; ++j)
    deterministic[static_cast<std::size_t>(j)] =
        0.03125 * std::sin(0.5 + static_cast<double>(j));
  probes.push_back(std::move(deterministic));

  for (const Vec& z : probes) {
    const Vec y = reduction.expand_primal(z);
    if (static_cast<int>(y.size()) != full.nvar || !finite_vec(y)) {
      message = "affine primal expansion failed validation";
      return false;
    }
    auto check_model = [&](const ConvexQuadraticModel& original,
                           const ConvexQuadraticModel& mapped,
                           const char* label) {
      const auto original_support=sparse_checks ? build_model_support(original,full.nvar) : ModelSupport{};
      const auto mapped_support=sparse_checks ? build_model_support(mapped,reduced_n) : ModelSupport{};
      const double original_value = supported_value(original,y,original_support);
      const double mapped_value = supported_value(mapped,z,mapped_support);
      if (!close_scalar(original_value, mapped_value)) {
        std::ostringstream os;
        os << label << " affine value mismatch: " << original_value
           << " versus " << mapped_value;
        message = os.str();
        return false;
      }
      const Vec expected = supported_transpose_gradient(original,y,
          reduction.transform_rows,reduced_n,original_support);
      Vec actual;
      if(mapped_support.eligible) {
        actual.assign(reduced_n,0.0);
        model_gradient_support_into(mapped,z,actual,mapped_support,nullptr);
      } else actual=mapped.gradient(z);
      if (expected.size() != actual.size()) {
        message = std::string(label) + " affine gradient dimension mismatch";
        return false;
      }
      for (std::size_t k = 0; k < expected.size(); ++k) {
        if (!close_scalar(expected[k], actual[k])) {
          std::ostringstream os;
          os << label << " affine gradient mismatch at " << k << ": "
             << expected[k] << " versus " << actual[k];
          message = os.str();
          return false;
        }
      }
      return true;
    };

    if (!check_model(full.objective, reduction.reduced.objective,
                     "objective"))
      return false;
    if (full.inequalities.size() != reduction.reduced.inequalities.size()) {
      message = "affine inequality count mismatch";
      return false;
    }
    for (std::size_t i = 0; i < full.inequalities.size(); ++i)
      if (!check_model(full.inequalities[i],
                       reduction.reduced.inequalities[i], "inequality"))
        return false;

    for (int r = 0; r < static_cast<int>(full.equalities.rows.size()); ++r) {
      const double full_residual =
          row_dot(full.equalities.rows[static_cast<std::size_t>(r)], y) -
          full.equalities.rhs[static_cast<std::size_t>(r)];
      const int mapped = reduction.full_to_reduced_equality[
          static_cast<std::size_t>(r)];
      if (mapped < 0) {
        const double row_scale = std::max(
            1.0, std::abs(full.equalities.rhs[static_cast<std::size_t>(r)]));
        if (!std::isfinite(full_residual) || std::abs(full_residual) >
            32768.0 * std::numeric_limits<double>::epsilon() * row_scale) {
          message = "eliminated affine equality is not exact on expansion";
          return false;
        }
      } else {
        const double reduced_residual = row_dot(
            reduction.reduced.equalities.rows[
                static_cast<std::size_t>(mapped)], z) -
            reduction.reduced.equalities.rhs[
                static_cast<std::size_t>(mapped)];
        if (!close_scalar(full_residual, reduced_residual)) {
          message = "retained affine equality residual mismatch";
          return false;
        }
      }
    }
  }
  message.clear();
  return true;
}

} // namespace

Vec ExactAffineSubproblemReduction::expand_primal(
    const Vec& reduced_primal) const {
  if (static_cast<int>(reduced_primal.size()) != reduced.nvar ||
      static_cast<int>(offset.size()) != full_variable_dimension ||
      static_cast<int>(transform_rows.size()) != full_variable_dimension)
    return {};
  Vec full = offset;
  for (int j = 0; j < full_variable_dimension; ++j)
    for (const auto& [k, value] :
         transform_rows[static_cast<std::size_t>(j)].entries)
      full[static_cast<std::size_t>(j)] +=
          value * reduced_primal[static_cast<std::size_t>(k)];
  return full;
}

bool build_exact_affine_subproblem_reduction(
    const ConvexSubproblem& p, ExactAffineSubproblemReduction& reduction,
    std::string& message, bool sparse_checks) {
  reduction = {};
  bool committed = false;
  struct ResetPartialReduction {
    ExactAffineSubproblemReduction& value;
    bool& committed;
    ~ResetPartialReduction() {
      if (!committed) value = {};
    }
  } reset_partial{reduction, committed};
  const int n = p.nvar;
  const int me = static_cast<int>(p.equalities.rows.size());
  if (n <= 0 || p.exact_affine_eliminations.empty()) {
    message = "no exact affine eliminations requested";
    return false;
  }
  if (!p.prefer_feasible_centered_start || p.original_dimension != n ||
      p.equalities.nvar != n ||
      static_cast<int>(p.equalities.rhs.size()) != me ||
      !valid_quadratic_model(p.objective, n)) {
    message = "affine elimination is restricted to a valid nonelastic LazyQP majorant";
    return false;
  }
  for (const auto& model : p.inequalities)
    if (!valid_quadratic_model(model, n)) {
      message = "invalid inequality model in affine elimination";
      return false;
    }
  for (int r = 0; r < me; ++r) {
    if (!std::isfinite(p.equalities.rhs[static_cast<std::size_t>(r)])) {
      message = "nonfinite equality rhs in affine elimination";
      return false;
    }
    for (const auto& [j, value] :
         p.equalities.rows[static_cast<std::size_t>(r)].entries) {
      if (j < 0 || j >= n || !std::isfinite(value)) {
        message = "invalid sparse equality row in affine elimination";
        return false;
      }
    }
  }

  std::vector<int> variable_to_elimination(static_cast<std::size_t>(n), -1);
  std::vector<int> equality_to_elimination(static_cast<std::size_t>(me), -1);
  std::vector<double> pivots(p.exact_affine_eliminations.size(), 0.0);
  for (int k = 0;
       k < static_cast<int>(p.exact_affine_eliminations.size()); ++k) {
    const auto& elimination =
        p.exact_affine_eliminations[static_cast<std::size_t>(k)];
    if (elimination.variable < 0 || elimination.variable >= n ||
        elimination.equality_row < 0 || elimination.equality_row >= me ||
        variable_to_elimination[static_cast<std::size_t>(
            elimination.variable)] >= 0 ||
        equality_to_elimination[static_cast<std::size_t>(
            elimination.equality_row)] >= 0) {
      message = "duplicate or out-of-range affine elimination hint";
      return false;
    }
    double pivot = 0.0;
    int pivot_entries = 0;
    for (const auto& [j, value] : p.equalities.rows[
             static_cast<std::size_t>(elimination.equality_row)].entries) {
      if (j < 0 || j >= n || !std::isfinite(value)) {
        message = "invalid sparse equality row in affine elimination";
        return false;
      }
      if (j == elimination.variable) {
        pivot += value;
        ++pivot_entries;
      }
    }
    if (pivot_entries != 1 || (pivot != 1.0 && pivot != -1.0)) {
      message = "affine elimination pivot is not the unique exact +/-1 entry";
      return false;
    }
    pivots[static_cast<std::size_t>(k)] = pivot;
    variable_to_elimination[static_cast<std::size_t>(
        elimination.variable)] = k;
    equality_to_elimination[static_cast<std::size_t>(
        elimination.equality_row)] = k;
  }

  // Each pivot column must occur only in its own row, and no pivot row may
  // depend on a second eliminated variable.  These make the substitution
  // diagonal and the dual reconstruction unambiguous.
  for (int r = 0; r < me; ++r) {
    for (const auto& [j, value] :
         p.equalities.rows[static_cast<std::size_t>(r)].entries) {
      if (value == 0.0) continue;
      const int k = variable_to_elimination[static_cast<std::size_t>(j)];
      if (k < 0) continue;
      const auto& elimination =
          p.exact_affine_eliminations[static_cast<std::size_t>(k)];
      if (r != elimination.equality_row) {
        message = "affine pivot variable occurs in another equality";
        return false;
      }
    }
  }
  for (const auto& elimination : p.exact_affine_eliminations) {
    for (const auto& [j, value] : p.equalities.rows[
             static_cast<std::size_t>(elimination.equality_row)].entries) {
      if (value != 0.0 && j != elimination.variable &&
          variable_to_elimination[static_cast<std::size_t>(j)] >= 0) {
        message = "coupled affine pivot rows are not eligible";
        return false;
      }
    }
  }

  reduction.full_variable_dimension = n;
  reduction.full_equality_dimension = me;
  reduction.eliminations = p.exact_affine_eliminations;
  reduction.full_to_reduced_variable.assign(static_cast<std::size_t>(n), -1);
  for (int j = 0; j < n; ++j) {
    if (variable_to_elimination[static_cast<std::size_t>(j)] >= 0) continue;
    reduction.full_to_reduced_variable[static_cast<std::size_t>(j)] =
        static_cast<int>(reduction.retained_variables.size());
    reduction.retained_variables.push_back(j);
  }
  const int reduced_n = static_cast<int>(reduction.retained_variables.size());
  if (reduced_n <= 0) {
    message = "affine elimination removed every variable";
    return false;
  }

  reduction.offset.assign(static_cast<std::size_t>(n), 0.0);
  reduction.transform_rows.resize(static_cast<std::size_t>(n));
  for (int k = 0; k < reduced_n; ++k) {
    const int original =
        reduction.retained_variables[static_cast<std::size_t>(k)];
    reduction.transform_rows[static_cast<std::size_t>(original)]
        .entries.push_back({k, 1.0});
  }
  for (int k = 0; k < static_cast<int>(reduction.eliminations.size()); ++k) {
    const auto& elimination = reduction.eliminations[static_cast<std::size_t>(k)];
    const double pivot = pivots[static_cast<std::size_t>(k)];
    reduction.offset[static_cast<std::size_t>(elimination.variable)] =
        p.equalities.rhs[static_cast<std::size_t>(
            elimination.equality_row)] / pivot;
    auto& transform = reduction.transform_rows[
        static_cast<std::size_t>(elimination.variable)].entries;
    for (const auto& [j, value] : p.equalities.rows[
             static_cast<std::size_t>(elimination.equality_row)].entries) {
      if (j == elimination.variable || value == 0.0) continue;
      const int reduced = reduction.full_to_reduced_variable[
          static_cast<std::size_t>(j)];
      if (reduced < 0) {
        message = "affine pivot row references an eliminated variable";
        return false;
      }
      transform.push_back({reduced, -value / pivot});
    }
    std::sort(transform.begin(), transform.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
  }

  reduction.reduced.nvar = reduced_n;
  reduction.reduced.original_dimension = reduced_n;
  reduction.reduced.allow_dca_single_loop = p.allow_dca_single_loop;
  reduction.reduced.allow_dca_single_loop_outer_filter_progress =
      p.allow_dca_single_loop_outer_filter_progress;
  reduction.reduced.allow_dca_single_loop_dual_bootstrap =
      p.allow_dca_single_loop_dual_bootstrap;
  reduction.reduced.prefer_feasible_centered_start =
      p.prefer_feasible_centered_start;
  if (!compose_quadratic_model(p.objective, reduction.offset,
                               reduction.transform_rows, reduced_n,
                               reduction.reduced.objective, message, sparse_checks))
    return false;
  reduction.reduced.inequalities.resize(p.inequalities.size());
  for (std::size_t i = 0; i < p.inequalities.size(); ++i)
    if (!compose_quadratic_model(
            p.inequalities[i], reduction.offset, reduction.transform_rows,
            reduced_n, reduction.reduced.inequalities[i], message, sparse_checks))
      return false;

  reduction.reduced.equalities.nvar = reduced_n;
  reduction.full_to_reduced_equality.assign(static_cast<std::size_t>(me), -1);
  for (int r = 0; r < me; ++r) {
    if (equality_to_elimination[static_cast<std::size_t>(r)] >= 0) continue;
    const int mapped = static_cast<int>(reduction.retained_equalities.size());
    reduction.full_to_reduced_equality[static_cast<std::size_t>(r)] = mapped;
    reduction.retained_equalities.push_back(r);
    Vec dense(static_cast<std::size_t>(reduced_n), 0.0);
    double rhs = p.equalities.rhs[static_cast<std::size_t>(r)];
    for (const auto& [j, value] :
         p.equalities.rows[static_cast<std::size_t>(r)].entries) {
      rhs -= value * reduction.offset[static_cast<std::size_t>(j)];
      for (const auto& [k, coefficient] :
           reduction.transform_rows[static_cast<std::size_t>(j)].entries)
        dense[static_cast<std::size_t>(k)] += value * coefficient;
    }
    SparseRow row;
    for (int k = 0; k < reduced_n; ++k)
      if (dense[static_cast<std::size_t>(k)] != 0.0)
        row.entries.push_back({k, dense[static_cast<std::size_t>(k)]});
    reduction.reduced.equalities.rows.push_back(std::move(row));
    reduction.reduced.equalities.rhs.push_back(rhs);
  }

  if (p.has_primal_warm_start &&
      static_cast<int>(p.initial_primal.size()) == n &&
      finite_vec(p.initial_primal)) {
    reduction.reduced.has_primal_warm_start = true;
    reduction.reduced.initial_primal.resize(static_cast<std::size_t>(reduced_n));
    for (int k = 0; k < reduced_n; ++k)
      reduction.reduced.initial_primal[static_cast<std::size_t>(k)] =
          p.initial_primal[static_cast<std::size_t>(
              reduction.retained_variables[static_cast<std::size_t>(k)])];
  }
  if (p.has_dual_warm_start &&
      static_cast<int>(p.initial_equality_duals.size()) == me &&
      p.initial_inequality_duals.size() == p.inequalities.size() &&
      finite_vec(p.initial_equality_duals) &&
      finite_vec(p.initial_inequality_duals)) {
    reduction.reduced.has_dual_warm_start = true;
    reduction.reduced.dual_warm_start_is_mapped =
        p.dual_warm_start_is_mapped;
    reduction.reduced.initial_equality_duals.resize(
        reduction.retained_equalities.size());
    for (int k = 0;
         k < static_cast<int>(reduction.retained_equalities.size()); ++k)
      reduction.reduced.initial_equality_duals[static_cast<std::size_t>(k)] =
          p.initial_equality_duals[static_cast<std::size_t>(
              reduction.retained_equalities[static_cast<std::size_t>(k)])];
    reduction.reduced.initial_inequality_duals =
        p.initial_inequality_duals;
  }
  if (p.has_inequality_slack_warm_start &&
      p.initial_inequality_slacks.size() == p.inequalities.size() &&
      finite_vec(p.initial_inequality_slacks)) {
    bool remapped = true;
    Vec reduced_slacks = p.initial_inequality_slacks;
    // Exact pivot elimination may move the warm primal point: T*y_r+c obeys
    // the eliminated affine rows, whereas the full-space origin can retain a
    // small linearization defect.  Copying the independent slacks verbatim
    // would then change r_pri=g(y)+s and can reject an otherwise valid direct
    // continuation.  Preserve every primal-slack residual exactly across the
    // coordinate change.  If positivity cannot be preserved, omit the slack
    // warm start transactionally and let the validated cold route take over.
    if (p.has_primal_warm_start &&
        static_cast<int>(p.initial_primal.size()) == n &&
        reduction.reduced.has_primal_warm_start) {
      for (std::size_t i = 0; i < p.inequalities.size(); ++i) {
        const double old_value = p.inequalities[i].value(p.initial_primal);
        const double reduced_value = reduction.reduced.inequalities[i].value(
            reduction.reduced.initial_primal);
        const double slack = p.initial_inequality_slacks[i] + old_value -
            reduced_value;
        if (!(slack > 0.0) || !std::isfinite(slack)) {
          remapped = false;
          break;
        }
        reduced_slacks[i] = slack;
      }
    }
    if (remapped) {
      reduction.reduced.has_inequality_slack_warm_start = true;
      reduction.reduced.initial_inequality_slacks =
          std::move(reduced_slacks);
    }
  }
  reduction.reduced.exact_affine_eliminations.clear();

  if (!validate_affine_reduction(p, reduction, message, sparse_checks)) {
    return false;
  }
  committed = true;
  message.clear();
  return true;
}

namespace {

double equality_residual_inf(const ConvexSubproblem& p, const Vec& y) {
  double residual = 0.0;
  for (int r = 0; r < static_cast<int>(p.equalities.rows.size()); ++r) {
    const double current =
        row_dot(p.equalities.rows[static_cast<std::size_t>(r)], y) -
        p.equalities.rhs[static_cast<std::size_t>(r)];
    if (!std::isfinite(current))
      return std::numeric_limits<double>::infinity();
    residual = std::max(
        residual, std::abs(current));
  }
  return residual;
}

struct FullAffineStateAudit {
  bool ok=false;
  Vec primal;
  Vec equality_duals;
  Vec inequality_duals;
  Vec inequality_slacks;
  double dual_inf=std::numeric_limits<double>::infinity();
  double equality_inf=std::numeric_limits<double>::infinity();
  double inequality_inf=std::numeric_limits<double>::infinity();
  double primal_slack_inf=std::numeric_limits<double>::infinity();
  double complementarity_inf=std::numeric_limits<double>::infinity();
  double composite_residual=std::numeric_limits<double>::infinity();
  double model=std::numeric_limits<double>::infinity();
};

FullAffineStateAudit expand_affine_state_for_audit(
    const ConvexSubproblem& full,
    const ExactAffineSubproblemReduction& reduction,
    const Vec& reduced_primal,const Vec& reduced_equality_duals,
    const Vec& inequality_duals,const Vec& inequality_slacks,
    const std::vector<ModelSupport>& supports) {
  Vec gradient_workspace;
  FullAffineStateAudit audit;
  if(static_cast<int>(reduced_primal.size())!=reduction.reduced.nvar ||
     reduced_equality_duals.size()!=reduction.retained_equalities.size() ||
     inequality_duals.size()!=full.inequalities.size() ||
     inequality_slacks.size()!=full.inequalities.size() ||
     !finite_vec(reduced_primal)||!finite_vec(reduced_equality_duals)||
     !finite_vec(inequality_duals)||!finite_vec(inequality_slacks))
    return audit;
  audit.primal=reduction.expand_primal(reduced_primal);
  if(static_cast<int>(audit.primal.size())!=full.nvar ||
     !finite_vec(audit.primal))
    return audit;
  audit.inequality_duals=inequality_duals;
  audit.inequality_slacks=inequality_slacks;
  audit.equality_duals.assign(full.equalities.rows.size(),0.0);
  for(int k=0;k<(int)reduction.retained_equalities.size();++k)
    audit.equality_duals[static_cast<std::size_t>(
        reduction.retained_equalities[static_cast<std::size_t>(k)])]=
        reduced_equality_duals[static_cast<std::size_t>(k)];

  Vec no_eliminated=full.objective.gradient(audit.primal);
  for(int k=0;k<(int)reduction.retained_equalities.size();++k)
    add_row_t(full.equalities.rows[static_cast<std::size_t>(
                  reduction.retained_equalities[static_cast<std::size_t>(k)])],
              reduced_equality_duals[static_cast<std::size_t>(k)],
              no_eliminated);
  for(int i=0;i<(int)full.inequalities.size();++i)
    if(inequality_duals[static_cast<std::size_t>(i)]!=0.0)
      supported_gradient_axpy(full.inequalities[i],audit.primal,
          inequality_duals[i],no_eliminated,supports[i],gradient_workspace);
  for(const auto& elimination:reduction.eliminations) {
    double pivot=0.0;
    for(const auto& [j,value]:full.equalities.rows[
            static_cast<std::size_t>(elimination.equality_row)].entries)
      if(j==elimination.variable) pivot+=value;
    if(pivot!=1.0 && pivot!=-1.0) return audit;
    audit.equality_duals[static_cast<std::size_t>(
        elimination.equality_row)]=
        -no_eliminated[static_cast<std::size_t>(elimination.variable)]/pivot;
  }
  Vec dual=full.objective.gradient(audit.primal);
  for(int r=0;r<(int)full.equalities.rows.size();++r)
    add_row_t(full.equalities.rows[static_cast<std::size_t>(r)],
              audit.equality_duals[static_cast<std::size_t>(r)],dual);
  for(int i=0;i<(int)full.inequalities.size();++i)
    if(inequality_duals[static_cast<std::size_t>(i)]!=0.0)
      supported_gradient_axpy(full.inequalities[i],audit.primal,
          inequality_duals[i],dual,supports[i],gradient_workspace);
  if(!finite_vec(dual)||!finite_vec(audit.equality_duals)) return audit;

  audit.dual_inf=max_abs(dual);
  audit.equality_inf=equality_residual_inf(full,audit.primal);
  audit.inequality_inf=0.0;
  audit.primal_slack_inf=0.0;
  audit.complementarity_inf=0.0;
  bool interior=true;
  for(int i=0;i<(int)full.inequalities.size();++i) {
    const double zi=inequality_duals[static_cast<std::size_t>(i)];
    const double si=inequality_slacks[static_cast<std::size_t>(i)];
    const double phi=supported_value(full.inequalities[i],audit.primal,supports[i]);
    audit.inequality_inf=std::max(audit.inequality_inf,phi);
    interior=interior&&zi>0.0&&si>0.0&&std::isfinite(phi);
    audit.primal_slack_inf=std::max(
        audit.primal_slack_inf,std::abs(phi+si));
    audit.complementarity_inf=std::max(
        audit.complementarity_inf,std::abs(si*zi));
  }
  audit.model=full.objective.value(audit.primal);
  audit.composite_residual=std::max(
      {audit.dual_inf,audit.equality_inf,audit.inequality_inf,
       audit.primal_slack_inf,audit.complementarity_inf});
  audit.ok=interior&&std::isfinite(audit.model)&&
      std::isfinite(audit.composite_residual);
  return audit;
}

bool expand_affine_inner_result(
    const ConvexSubproblem& full,
    const ExactAffineSubproblemReduction& reduction,
    const InnerResult& reduced_result, const Settings& settings,
    InnerResult& expanded,
    std::string& message) {
  expanded = reduced_result;
  expanded.y = reduction.expand_primal(reduced_result.y);
  if (static_cast<int>(expanded.y.size()) != full.nvar ||
      !finite_vec(expanded.y) ||
      reduced_result.equality_duals.size() !=
          reduction.retained_equalities.size() ||
      reduced_result.inequality_duals.size() != full.inequalities.size() ||
      reduced_result.inequality_slacks.size() != full.inequalities.size()) {
    message = "reduced LazyQP result has incompatible dimensions";
    return false;
  }

  expanded.equality_duals.assign(full.equalities.rows.size(), 0.0);
  for (int k = 0;
       k < static_cast<int>(reduction.retained_equalities.size()); ++k)
    expanded.equality_duals[static_cast<std::size_t>(
        reduction.retained_equalities[static_cast<std::size_t>(k)])] =
        reduced_result.equality_duals[static_cast<std::size_t>(k)];

  const auto supports=settings.lazyqp_affine_support_fastpath
      ? build_model_supports(full.inequalities,full.nvar)
      : std::vector<ModelSupport>(full.inequalities.size());
  Vec gradient_workspace;
  // Stationarity without the eliminated equality multipliers.  The strict
  // structural contract makes each pivot column unique, so its missing free
  // multiplier is recovered independently from that coordinate.
  Vec no_eliminated = full.objective.gradient(expanded.y);
  for (int k = 0;
       k < static_cast<int>(reduction.retained_equalities.size()); ++k)
    add_row_t(
        full.equalities.rows[static_cast<std::size_t>(
            reduction.retained_equalities[static_cast<std::size_t>(k)])],
        reduced_result.equality_duals[static_cast<std::size_t>(k)],
        no_eliminated);
  for (int i = 0; i < static_cast<int>(full.inequalities.size()); ++i) {
    const double multiplier =
        expanded.inequality_duals[static_cast<std::size_t>(i)];
    if (multiplier != 0.0)
      supported_gradient_axpy(full.inequalities[i],expanded.y,multiplier,
          no_eliminated,supports[i],gradient_workspace);
  }

  for (const auto& elimination : reduction.eliminations) {
    double pivot = 0.0;
    for (const auto& [j, value] : full.equalities.rows[
             static_cast<std::size_t>(elimination.equality_row)].entries)
      if (j == elimination.variable) pivot += value;
    if (pivot != 1.0 && pivot != -1.0) {
      message = "affine dual reconstruction lost its exact pivot";
      return false;
    }
    expanded.equality_duals[static_cast<std::size_t>(
        elimination.equality_row)] =
        -no_eliminated[static_cast<std::size_t>(elimination.variable)] /
        pivot;
  }

  Vec full_dual = full.objective.gradient(expanded.y);
  for (int r = 0; r < static_cast<int>(full.equalities.rows.size()); ++r)
    add_row_t(full.equalities.rows[static_cast<std::size_t>(r)],
              expanded.equality_duals[static_cast<std::size_t>(r)],
              full_dual);
  for (int i = 0; i < static_cast<int>(full.inequalities.size()); ++i) {
    const double multiplier =
        expanded.inequality_duals[static_cast<std::size_t>(i)];
    if (multiplier != 0.0)
      supported_gradient_axpy(full.inequalities[i],expanded.y,multiplier,
          full_dual,supports[i],gradient_workspace);
  }
  if (!finite_vec(expanded.equality_duals) || !finite_vec(full_dual)) {
    message = "affine dual reconstruction produced a nonfinite result";
    return false;
  }

  expanded.dual_inf = max_abs(full_dual);
  expanded.equality_inf = equality_residual_inf(full, expanded.y);
  expanded.inequality_inf = 0.0;
  expanded.primal_residual_inf = 0.0;
  double complementarity_sum = 0.0;
  double full_complementarity_inf = 0.0;
  for (int i = 0; i < static_cast<int>(full.inequalities.size()); ++i) {
    const double slack = expanded.inequality_slacks[static_cast<std::size_t>(i)];
    const double multiplier =
        expanded.inequality_duals[static_cast<std::size_t>(i)];
    const double value = supported_value(full.inequalities[i],expanded.y,supports[i]);
    expanded.inequality_inf=std::max(expanded.inequality_inf,value);
    if (!(slack > 0.0) || !(multiplier > 0.0) ||
        !std::isfinite(value)) {
      message = "expanded affine state has an invalid inequality slack or dual";
      return false;
    }
    expanded.primal_residual_inf = std::max(
        expanded.primal_residual_inf, std::abs(value + slack));
    complementarity_sum += slack * multiplier;
    full_complementarity_inf = std::max(
        full_complementarity_inf,std::abs(slack*multiplier));
  }
  expanded.complementarity = full.inequalities.empty()
      ? 0.0
      : complementarity_sum / static_cast<double>(full.inequalities.size());

  // Preserve the rolled-back origin of every single-loop transaction,
  // including a retryable guard rejection.  The outer solver needs this full
  // coordinate lambda/z/s snapshot to rebuild a different proximal majorant
  // without silently reverting to a cold or strict solve.
  const FullAffineStateAudit single_loop_initial=expand_affine_state_for_audit(
      full,reduction,reduced_result.dca_single_loop_initial_primal,
      reduced_result.dca_single_loop_initial_equality_duals,
      reduced_result.dca_single_loop_initial_inequality_duals,
      reduced_result.dca_single_loop_initial_inequality_slacks,supports);
  if(single_loop_initial.ok) {
    expanded.dca_single_loop_initial_primal=single_loop_initial.primal;
    expanded.dca_single_loop_initial_equality_duals=
        single_loop_initial.equality_duals;
    expanded.dca_single_loop_initial_inequality_duals=
        single_loop_initial.inequality_duals;
    expanded.dca_single_loop_initial_inequality_slacks=
        single_loop_initial.inequality_slacks;
    expanded.dca_single_loop_initial_residual=
        single_loop_initial.composite_residual;
  }

  // A reduced success is accepted only when it is also the same KKT point of
  // the untouched full model.  Test the historical LazyQP success contract on
  // the recomputed full residuals, then require full/reduced metrics to agree
  // one decade more tightly than inner_tolerance.  Comparing two infinity
  // norms at a fixed O(eps) absolute threshold is not sound here: different
  // coordinates can tie for the maximum after cancellation of O(1e3) terms,
  // even though their full residual vectors agree to working precision.
  // Algebraic/indexing mistakes remain fail-closed through both checks.
  if (reduced_result.success || reduced_result.dca_single_loop_accepted) {
    const double model_tolerance =
        std::max(settings.inner_tolerance, settings.qp_model_solved_tol);
    const double full_residual = std::max(
        {expanded.dual_inf, expanded.equality_inf,
         expanded.primal_residual_inf});
    const bool full_strict = full_residual <= model_tolerance &&
        expanded.complementarity <= model_tolerance;
    const bool full_guarded =
        full_residual <= 100.0 * settings.inner_tolerance &&
        expanded.complementarity <=
            settings.inner_iteration_limit_mu_fallback;
    bool full_forcing=false;
    if(reduced_result.lazyqp_inexact_forcing_accepted &&
       reduced_result.lazyqp_inexact_forcing_target>0.0 &&
       reduced_result.lazyqp_inexact_forcing_primal_target>0.0) {
      const double forcing_residual=std::max(
          {expanded.dual_inf,expanded.equality_inf,
           expanded.inequality_inf,expanded.primal_residual_inf,
           full_complementarity_inf});
      const double forcing_primal=std::max(
          {expanded.equality_inf,expanded.inequality_inf,
           expanded.primal_residual_inf});
      Vec zero(static_cast<std::size_t>(full.nvar),0.0);
      const double model_start=full.objective.value(zero);
      const double model_final=full.objective.value(expanded.y);
      const double model_noise=128.0*std::numeric_limits<double>::epsilon()*
          std::max({1.0,std::abs(model_start),std::abs(model_final)});
      const double full_initial_residual=
          centered_origin_forcing_residual(full,settings);
      expanded.lazyqp_inexact_forcing_model_evaluations+=3;
      const double full_outer_target=
          reduced_result.lazyqp_inexact_forcing_outer_target>0.0
              ? reduced_result.lazyqp_inexact_forcing_outer_target
              : reduced_result.lazyqp_inexact_forcing_target;
      const double full_forcing_target=std::min(
          full_outer_target,
          settings.lazyqp_forcing_eta*full_initial_residual);
      full_forcing=std::isfinite(forcing_residual) &&
          std::isfinite(full_initial_residual) &&
          forcing_residual<=full_forcing_target &&
          forcing_primal<=
              reduced_result.lazyqp_inexact_forcing_primal_target &&
          std::isfinite(model_final) &&
          model_start-model_final>model_noise;
      if(!full_forcing) {
        std::ostringstream os;
        os << "expanded state fails full-model forcing contract: residual "
           << forcing_residual << "/"
           << full_forcing_target
           << ", primal " << forcing_primal << "/"
           << reduced_result.lazyqp_inexact_forcing_primal_target
           << ", model decrease " << model_start-model_final
           << ", noise " << model_noise;
        message=os.str();
        return false;
      }
      expanded.lazyqp_inexact_forcing_initial_residual=
          full_initial_residual;
      expanded.lazyqp_inexact_forcing_outer_target=full_outer_target;
      expanded.lazyqp_inexact_forcing_target=full_forcing_target;
    }
    bool full_single_loop=false;
    if(reduced_result.dca_single_loop_accepted) {
      const double full_composite_residual=std::max(
          {expanded.dual_inf,expanded.equality_inf,
           expanded.inequality_inf,expanded.primal_residual_inf,
           full_complementarity_inf});
      const double full_final_primal=std::max(
          {expanded.equality_inf,expanded.inequality_inf,
           expanded.primal_residual_inf});
      full_single_loop=single_loop_initial.ok &&
          std::isfinite(full_composite_residual) &&
          std::isfinite(full_final_primal);
      if(full_single_loop) {
        expanded.dca_single_loop_final_residual=full_composite_residual;
      }
    }
    const bool historical_contract=reduced_result.success &&
        (full_strict||full_guarded||full_forcing);
    const bool provisional_contract=
        reduced_result.dca_single_loop_accepted&&full_single_loop;
    if (!historical_contract && !provisional_contract) {
      std::ostringstream os;
      os << "expanded state fails full-model LazyQP KKT contract: residual "
         << full_residual << ", complementarity "
         << expanded.complementarity;
      message = os.str();
      return false;
    }
    const double metric_match_tolerance = std::max(
        1048576.0 * std::numeric_limits<double>::epsilon(),
        0.1 * settings.inner_tolerance);
    auto same_metric = [metric_match_tolerance](double original,
                                                double mapped) {
      const double scale = std::max({1.0, std::abs(original),
                                     std::abs(mapped)});
      return std::isfinite(original) && std::isfinite(mapped) &&
          std::abs(original - mapped) <= metric_match_tolerance * scale;
    };
    if (!same_metric(expanded.dual_inf, reduced_result.dual_inf) ||
        !same_metric(expanded.equality_inf,
                     reduced_result.equality_inf) ||
        !same_metric(expanded.inequality_inf,
                     reduced_result.inequality_inf) ||
        !same_metric(expanded.primal_residual_inf,
                     reduced_result.primal_residual_inf) ||
        !same_metric(expanded.complementarity,
                     reduced_result.complementarity)) {
      std::ostringstream os;
      os << "expanded full-model KKT mismatch: dual "
         << expanded.dual_inf << "/" << reduced_result.dual_inf
         << ", equality " << expanded.equality_inf << "/"
         << reduced_result.equality_inf << ", inequality "
         << expanded.inequality_inf << "/"
         << reduced_result.inequality_inf << ", primal-slack "
         << expanded.primal_residual_inf << "/"
         << reduced_result.primal_residual_inf << ", complementarity "
         << expanded.complementarity << "/"
         << reduced_result.complementarity;
      message = os.str();
      return false;
    }
  }
  message.clear();
  return true;
}

void accumulate_failed_reduction_work(const InnerResult& attempted,
                                      InnerResult& fallback) {
  fallback.lazyqp_inexact_forcing_attempted =
      fallback.lazyqp_inexact_forcing_attempted ||
      attempted.lazyqp_inexact_forcing_attempted;
  fallback.lazyqp_inexact_forcing_strict_fallback =
      fallback.lazyqp_inexact_forcing_strict_fallback ||
      attempted.lazyqp_inexact_forcing_attempted ||
      attempted.lazyqp_inexact_forcing_strict_fallback;
  fallback.lazyqp_inexact_forcing_target=std::max(
      fallback.lazyqp_inexact_forcing_target,
      attempted.lazyqp_inexact_forcing_target);
  fallback.lazyqp_inexact_forcing_initial_residual=std::max(
      fallback.lazyqp_inexact_forcing_initial_residual,
      attempted.lazyqp_inexact_forcing_initial_residual);
  fallback.lazyqp_inexact_forcing_outer_target=std::max(
      fallback.lazyqp_inexact_forcing_outer_target,
      attempted.lazyqp_inexact_forcing_outer_target);
  fallback.lazyqp_inexact_forcing_primal_target=std::max(
      fallback.lazyqp_inexact_forcing_primal_target,
      attempted.lazyqp_inexact_forcing_primal_target);
  fallback.dca_single_loop_attempted =
      fallback.dca_single_loop_attempted ||
      attempted.dca_single_loop_attempted;
  fallback.dca_single_loop_strict_fallback =
      fallback.dca_single_loop_strict_fallback ||
      attempted.dca_single_loop_attempted ||
      attempted.dca_single_loop_strict_fallback;
  fallback.dca_single_loop_state_transport_used =
      fallback.dca_single_loop_state_transport_used ||
      attempted.dca_single_loop_state_transport_used;
  fallback.dca_single_loop_state_transport_rejected =
      fallback.dca_single_loop_state_transport_rejected ||
      attempted.dca_single_loop_state_transport_rejected;
  fallback.dca_single_loop_steps += attempted.dca_single_loop_steps;
  fallback.dca_single_loop_factorizations +=
      attempted.dca_single_loop_factorizations;
  fallback.dca_single_loop_initial_residual = std::max(
      fallback.dca_single_loop_initial_residual,
      attempted.dca_single_loop_initial_residual);
  fallback.dca_single_loop_final_residual = std::max(
      fallback.dca_single_loop_final_residual,
      attempted.dca_single_loop_final_residual);
  fallback.lazyqp_inexact_forcing_model_evaluations +=
      attempted.lazyqp_inexact_forcing_model_evaluations;
  fallback.lazyqp_model_support_fastpath_requests +=
      attempted.lazyqp_model_support_fastpath_requests;
  fallback.lazyqp_model_support_fastpath_effective_subproblems +=
      attempted.lazyqp_model_support_fastpath_effective_subproblems;
  fallback.lazyqp_model_support_builds +=
      attempted.lazyqp_model_support_builds;
  fallback.lazyqp_model_support_eligible_models +=
      attempted.lazyqp_model_support_eligible_models;
  fallback.lazyqp_model_support_generic_fallback_models +=
      attempted.lazyqp_model_support_generic_fallback_models;
  fallback.lazyqp_model_support_value_evaluations +=
      attempted.lazyqp_model_support_value_evaluations;
  fallback.lazyqp_model_support_gradient_evaluations +=
      attempted.lazyqp_model_support_gradient_evaluations;
  fallback.iterations += attempted.iterations;
  fallback.factorizations += attempted.factorizations;
  fallback.symbolic_analyses += attempted.symbolic_analyses;
  fallback.symbolic_reuses += attempted.symbolic_reuses;
  fallback.symbolic_invalidations += attempted.symbolic_invalidations;
  fallback.numeric_scatter_builds += attempted.numeric_scatter_builds;
  fallback.numeric_scatter_reuses += attempted.numeric_scatter_reuses;
  fallback.numeric_scatter_invalidations +=
      attempted.numeric_scatter_invalidations;
  fallback.kkt_regularization_retries += attempted.kkt_regularization_retries;
  fallback.kkt_refinement_steps += attempted.kkt_refinement_steps;
  fallback.lazyqp_scaled_split_steps += attempted.lazyqp_scaled_split_steps;
  fallback.lazyqp_max_primal_dual_step_ratio = std::max(
      fallback.lazyqp_max_primal_dual_step_ratio,
      attempted.lazyqp_max_primal_dual_step_ratio);
  fallback.lazyqp_max_dual_primal_step_ratio = std::max(
      fallback.lazyqp_max_dual_primal_step_ratio,
      attempted.lazyqp_max_dual_primal_step_ratio);
  fallback.lazyqp_min_globalization_tau = std::min(
      fallback.lazyqp_min_globalization_tau,
      attempted.lazyqp_min_globalization_tau);
  fallback.initialization_bound_pushes +=
      attempted.initialization_bound_pushes;
  fallback.lazyqp_reduced_kkt_factorizations +=
      attempted.lazyqp_reduced_kkt_factorizations;
  fallback.lazyqp_full_kkt_factorizations +=
      attempted.lazyqp_full_kkt_factorizations;
  fallback.lazyqp_reduced_kkt_fallbacks +=
      attempted.lazyqp_reduced_kkt_fallbacks;
  fallback.lazyqp_hybrid_kkt_attempts +=
      attempted.lazyqp_hybrid_kkt_attempts;
  fallback.lazyqp_hybrid_kkt_factorizations +=
      attempted.lazyqp_hybrid_kkt_factorizations;
  fallback.lazyqp_hybrid_kkt_fallbacks +=
      attempted.lazyqp_hybrid_kkt_fallbacks;
  fallback.lazyqp_hybrid_kkt_retained_rows_total +=
      attempted.lazyqp_hybrid_kkt_retained_rows_total;
  fallback.lazyqp_hybrid_kkt_retained_rows_max = std::max(
      fallback.lazyqp_hybrid_kkt_retained_rows_max,
      attempted.lazyqp_hybrid_kkt_retained_rows_max);
  fallback.lazyqp_hybrid_kkt_outer_product_triplets_avoided +=
      attempted.lazyqp_hybrid_kkt_outer_product_triplets_avoided;
  fallback.lazyqp_hybrid_kkt_border_triplets_emitted +=
      attempted.lazyqp_hybrid_kkt_border_triplets_emitted;
  fallback.lazyqp_kkt_triplet_layout_builds +=
      attempted.lazyqp_kkt_triplet_layout_builds;
  fallback.lazyqp_kkt_triplet_layout_reuses +=
      attempted.lazyqp_kkt_triplet_layout_reuses;
  fallback.lazyqp_kkt_triplet_layout_invalidations +=
      attempted.lazyqp_kkt_triplet_layout_invalidations;
  fallback.lazyqp_numeric_kkt_reuse_attempts +=
      attempted.lazyqp_numeric_kkt_reuse_attempts;
  fallback.lazyqp_numeric_kkt_reuse_acceptances +=
      attempted.lazyqp_numeric_kkt_reuse_acceptances;
  fallback.lazyqp_numeric_kkt_reuse_fallbacks +=
      attempted.lazyqp_numeric_kkt_reuse_fallbacks;
  fallback.lazyqp_numeric_kkt_reuse_iterations +=
      attempted.lazyqp_numeric_kkt_reuse_iterations;
  fallback.lazyqp_zero_dual_continuation_attempts +=
      attempted.lazyqp_zero_dual_continuation_attempts;
  fallback.lazyqp_zero_dual_continuation_mapped_attempts +=
      attempted.lazyqp_zero_dual_continuation_mapped_attempts;
  fallback.lazyqp_zero_dual_continuation_direct_attempts +=
      attempted.lazyqp_zero_dual_continuation_direct_attempts;
  fallback.lazyqp_zero_dual_continuation_acceptances +=
      attempted.lazyqp_zero_dual_continuation_acceptances;
  fallback.lazyqp_zero_dual_continuation_fallbacks +=
      attempted.lazyqp_zero_dual_continuation_fallbacks;
  fallback.lazyqp_zero_dual_continuation_repairs +=
      attempted.lazyqp_zero_dual_continuation_repairs;
  fallback.lazyqp_kkt_dimension = std::max(
      fallback.lazyqp_kkt_dimension, attempted.lazyqp_kkt_dimension);
  fallback.lazyqp_full_kkt_dimension = std::max(
      fallback.lazyqp_full_kkt_dimension,
      attempted.lazyqp_full_kkt_dimension);
  fallback.max_kkt_rho_x =
      std::max(fallback.max_kkt_rho_x, attempted.max_kkt_rho_x);
  fallback.max_kkt_delta_d =
      std::max(fallback.max_kkt_delta_d, attempted.max_kkt_delta_d);
  fallback.max_kkt_backward_error = std::max(
      fallback.max_kkt_backward_error,
      attempted.max_kkt_backward_error);
  fallback.max_kkt_residual_inf = std::max(
      fallback.max_kkt_residual_inf, attempted.max_kkt_residual_inf);
}

} // namespace

bool certify_affine_tangent_inertia(const ConvexSubproblem& model,
    SymmetricFactorization& factor, std::string& message) {
  if(!model.inequalities.empty() || !valid_quadratic_model(model.objective,model.nvar)) {
    message="invalid tangent-only model"; return false;
  }
  ExactAffineSubproblemReduction reduction;
  const ConvexSubproblem* p=&model;
  if(!model.exact_affine_eliminations.empty()) {
    if(!build_exact_affine_subproblem_reduction(model,reduction,message,true)) return false;
    p=&reduction.reduced;
  }
  const int n=p->nvar, m=static_cast<int>(p->equalities.rows.size());
  if(p->equalities.nvar!=n || p->equalities.rhs.size()!=p->equalities.rows.size()) {
    message="tangent Jacobian dimension mismatch"; return false;
  }
  std::vector<SymmetricTriplet> K;
  K.reserve(p->objective.base_hessian.entries.size()+n+m);
  for(const auto& e:p->objective.base_hessian.entries) K.push_back({e.i,e.j,e.v});
  for(int j=0;j<n;++j)
    K.push_back({j,j,j<p->objective.shift_dimension ? p->objective.scalar_shift : 0.0});
  for(int a=0;a<m;++a) {
    K.push_back({n+a,n+a,0.0}); // no dual perturbation in a curvature certificate
    for(const auto& [j,v]:p->equalities.rows[a].entries) {
      if(j<0 || j>=n || !std::isfinite(v)) { message="invalid tangent Jacobian"; return false; }
      K.push_back({j,n+a,v});
    }
  }
  int rank=0;
  for(const auto& t:p->objective.low_rank) {
    if(t.beta<0.0) { message="negative tangent correction"; return false; }
    if(t.beta==0.0) continue;
    const int border=n+m+rank++;
    K.push_back({border,border,-1.0});
    for(int j=0;j<n;++j) if(t.direction[j]!=0.0)
      K.push_back({j,border,std::sqrt(t.beta)*t.direction[j]});
  }
  // The backend must be configured without implicit diagonal shifts. The
  // caller owns that Settings lifetime; this is the same contract as the
  // condensed tangent certificate.
  return factor.factor(n+m+rank,n,K,message);
}

void ConvexQCQPLazyQPKernel::clear_factorization_cache() const {
  full_factor_.reset();
  reduced_factor_.reset();
  hybrid_factor_.reset();
  full_triplets_.clear();
  reduced_triplets_.clear();
  hybrid_triplets_.clear();
  factor_backend_.clear();
  factor_metis_=false;
  factor_seed_=0;
}

SymmetricFactorization* ConvexQCQPLazyQPKernel::acquire_factorization(
    bool reduced,bool hybrid,std::string& message) const {
  auto& slot=!reduced ? full_factor_ :
      (hybrid ? hybrid_factor_ : reduced_factor_);
  if(!slot) {
    slot=make_symmetric_factorization(st_,message);
    if(!slot) return nullptr;
    factor_backend_=st_.linear_backend;
    factor_metis_=st_.linear_metis;
    factor_seed_=st_.random_seed;
  }
  return slot.get();
}

InnerResult ConvexQCQPLazyQPKernel::solve(const ConvexSubproblem& p) const {
  const bool single_loop_requested=
      st_.dca_single_loop_primal_dual && p.allow_dca_single_loop;
  if(single_loop_requested && p.inequalities.empty()) {
    InnerResult failed;
    failed.message=
        "single-loop primal-dual majorant requires at least one inequality row";
    return failed;
  }
  if(single_loop_requested && st_.lazyqp_numeric_kkt_reuse) {
    InnerResult failed;
    failed.message="invalid single-loop primal-dual configuration";
    return failed;
  }
  // The outer solver deliberately retains its historical full-dimensional
  // majorant.  Only this private LazyQP call is reduced, and every result is
  // expanded before it crosses the kernel boundary.
  if (st_.lazyqp_flow_lift_elimination &&
      !p.exact_affine_eliminations.empty()) {
    const bool single_loop_transaction =
        st_.dca_single_loop_primal_dual && p.allow_dca_single_loop;
    ExactAffineSubproblemReduction reduction;
    std::string reduction_message;
    const bool reduction_ok = build_exact_affine_subproblem_reduction(
        p, reduction, reduction_message, st_.lazyqp_affine_support_fastpath);
    InnerResult attempted;
    InnerResult expanded;
    bool expanded_ok = false;
    if (reduction_ok) {
      attempted = solve(reduction.reduced);
      expanded_ok = expand_affine_inner_result(
          p, reduction, attempted, st_, expanded, reduction_message);
      if(attempted.dca_single_loop_retryable && expanded_ok) {
        ++expanded.lazyqp_affine_elimination_attempts;
        return expanded;
      }
      if ((attempted.success || attempted.dca_single_loop_accepted) &&
          expanded_ok) {
        ++expanded.lazyqp_affine_elimination_attempts;
        ++expanded.lazyqp_affine_elimination_acceptances;
        expanded.lazyqp_affine_eliminated_variables =
            static_cast<int>(reduction.eliminations.size());
        expanded.lazyqp_affine_eliminated_equalities =
            static_cast<int>(reduction.eliminations.size());
        expanded.lazyqp_affine_reduced_variables = reduction.reduced.nvar;
        expanded.lazyqp_affine_reduced_equalities =
            static_cast<int>(reduction.reduced.equalities.rows.size());
        if (st_.verbose)
          std::cout << "[LAZYQP-AFFINE-ELIM] status=accepted"
                    << " variables=" << p.nvar << "->"
                    << reduction.reduced.nvar
                    << " equalities=" << p.equalities.rows.size() << "->"
                    << reduction.reduced.equalities.rows.size()
                    << " pivots=" << reduction.eliminations.size() << "\n";
        return expanded;
      }
      attempted.lazyqp_inexact_forcing_model_evaluations=std::max(
          attempted.lazyqp_inexact_forcing_model_evaluations,
          expanded.lazyqp_inexact_forcing_model_evaluations);
    }

    if(single_loop_transaction) {
      // This wrapper is part of the same provisional transaction.  Once the
      // single-loop route is selected, a bad hint, a failed reduced solve, or
      // an invalid expansion must never escape through the historical strict
      // full-dimensional solve below.  A numerical transaction which reached
      // the predictor-corrector is retryable by a new outer proximal model;
      // structural/configuration failures remain ordinary fail-closed errors.
      InnerResult failed = expanded_ok ? expanded : attempted;
      failed.success=false;
      failed.dca_single_loop_accepted=false;
      failed.dca_single_loop_strict_fallback=false;
      failed.dca_single_loop_retryable=
          attempted.dca_single_loop_retryable ||
          attempted.dca_single_loop_attempted;
      ++failed.lazyqp_affine_elimination_attempts;
      if(reduction_ok) {
        failed.lazyqp_affine_eliminated_variables =
            static_cast<int>(reduction.eliminations.size());
        failed.lazyqp_affine_eliminated_equalities =
            static_cast<int>(reduction.eliminations.size());
        failed.lazyqp_affine_reduced_variables = reduction.reduced.nvar;
        failed.lazyqp_affine_reduced_equalities =
            static_cast<int>(reduction.reduced.equalities.rows.size());
      }
      failed.message=failed.dca_single_loop_retryable
          ? "retryable single-loop affine transaction failed closed: "+
                reduction_message
          : "single-loop affine transaction failed closed: "+
                reduction_message;
      return failed;
    }

    // Fail closed.  Clear only the untrusted hint and run the untouched model
    // through the exact historical kernel.  A failed reduced attempt is
    // included in work counters so timing/factorization diagnostics remain
    // honest.
    ConvexSubproblem full = p;
    full.exact_affine_eliminations.clear();
    // A reduction/expansion rejection invalidates the provisional coordinate
    // transaction.  The untouched full model is therefore solved strictly;
    // it must not immediately attempt a second provisional path.
    full.allow_dca_single_loop=false;
    full.allow_dca_single_loop_outer_filter_progress=false;
    InnerResult fallback = solve(full);
    if (reduction_ok) accumulate_failed_reduction_work(attempted, fallback);
    ++fallback.lazyqp_affine_elimination_attempts;
    ++fallback.lazyqp_affine_elimination_fallbacks;
    if (reduction_ok) {
      fallback.lazyqp_affine_eliminated_variables =
          static_cast<int>(reduction.eliminations.size());
      fallback.lazyqp_affine_eliminated_equalities =
          static_cast<int>(reduction.eliminations.size());
      fallback.lazyqp_affine_reduced_variables = reduction.reduced.nvar;
      fallback.lazyqp_affine_reduced_equalities =
          static_cast<int>(reduction.reduced.equalities.rows.size());
    }
    if (st_.verbose)
      std::cout << "[LAZYQP-AFFINE-ELIM] status=fallback"
                << " reason=" << reduction_message << "\n";
    return fallback;
  }

  InnerResult out;
  const int n=p.nvar, me=(int)p.equalities.rows.size(), mi=(int)p.inequalities.size();
  out.lazyqp_model_support_fastpath_requests=
      st_.lazyqp_model_support_fastpath ? 1 : 0;
  if(n<=0 || (int)p.objective.linear.size()!=n ||
     (int)p.equalities.rhs.size()!=me ||
     !valid_quadratic_model(p.objective,n)) {
    out.message="invalid convex subproblem dimensions"; return out;
  }
  for(const auto& model:p.inequalities) {
    if(!valid_quadratic_model(model,n)) {
      out.message="invalid inequality model dimensions";
      return out;
    }
  }
  for(int r=0;r<me;++r) {
    if(!std::isfinite(p.equalities.rhs[static_cast<std::size_t>(r)])) {
      out.message="invalid equality model dimensions";
      return out;
    }
    for(const auto& [j,value]:
        p.equalities.rows[static_cast<std::size_t>(r)].entries) {
      if(j<0 || j>=n || !std::isfinite(value)) {
        out.message="invalid equality model dimensions";
        return out;
      }
    }
  }

  // The nonlinear-residual LazyQP extension is reserved for explicit
  // Lazy-DCA majorants.  Restoration and SQP auxiliary models retain the
  // historical IPM.  In v0.4.7, enabling an inequality warm start accidentally
  // made `centered` false and routed a Lazy-DCA majorant back to that legacy
  // solver.  Keep solver selection independent of initialization state.
  const bool dedicated=p.prefer_feasible_centered_start;
  if(!dedicated) {
    ConvexQCQPInteriorPoint legacy(st_);
    InnerResult legacy_result=legacy.solve(p);
    legacy_result.lazyqp_model_support_fastpath_requests=
        out.lazyqp_model_support_fastpath_requests;
    if(st_.lazyqp_model_support_fastpath)
      legacy_result.lazyqp_model_support_generic_fallback_models=mi;
    return legacy_result;
  }

  const bool primal_warm_dimensions=p.has_primal_warm_start &&
      static_cast<int>(p.initial_primal.size())==n && finite_vec(p.initial_primal);
  Vec y=primal_warm_dimensions ? p.initial_primal : Vec(n,0.0);
  out.primal_warm_start_used=primal_warm_dimensions;
  Vec cold_lambda(me,0.0), cold_z(mi,1.0), cold_s(mi,1.0);
  Vec inequality_gradient_work(static_cast<std::size_t>(n),0.0);
  const std::vector<ModelSupport> model_supports=
      st_.lazyqp_model_support_fastpath
          ? build_model_supports(p.inequalities,n)
          : std::vector<ModelSupport>();
  const std::vector<ModelSupport>* supports=
      st_.lazyqp_model_support_fastpath ? &model_supports : nullptr;
  if(supports) {
    out.lazyqp_model_support_builds=1;
    out.lazyqp_model_support_eligible_models=static_cast<int>(std::count_if(
        supports->begin(),supports->end(),
        [](const ModelSupport& support){ return support.eligible; }));
    out.lazyqp_model_support_generic_fallback_models=
        mi-out.lazyqp_model_support_eligible_models;
    out.lazyqp_model_support_fastpath_effective_subproblems=
        out.lazyqp_model_support_eligible_models>0 ? 1 : 0;
  }
  int cold_pushes=0;

  // Validated centered cold start.  It remains the reference against which a
  // continuation state is guarded, so enabling warm starts cannot silently
  // degrade a previously robust first-subproblem initialization.
  const double gscale=std::max(1e-2,std::min(1.0,max_abs(p.objective.gradient(y))));
  const double target_mu=std::max({st_.lazyqp_barrier_mu0,gscale,100.0*st_.inner_tolerance});
  const double slack_push=std::max(1e-4,10.0*st_.inner_tolerance);
  for(int i=0;i<mi;++i) {
    const ModelSupport* support=
        supports && (*supports)[static_cast<std::size_t>(i)].eligible
            ? &(*supports)[static_cast<std::size_t>(i)] : nullptr;
    const double phi=support
        ? model_value_support(p.inequalities[static_cast<std::size_t>(i)],y,
                              *support,&out)
        : p.inequalities[i].value(y);
    const double raw=std::max(0.0,-phi);
    cold_s[i]=std::max(raw,slack_push);
    if(cold_s[i]>raw) ++cold_pushes;
    cold_z[i]=std::clamp(target_mu/std::max(cold_s[i],target_mu),
                         st_.minimum_positive,10.0);
  }
  const StateEval cold_eval=evaluate_state(
      p,y,cold_lambda,cold_z,cold_s,inequality_gradient_work,nullptr,
      supports,&out);

  Vec lambda=cold_lambda, z=cold_z, s=cold_s;
  StateEval e=cold_eval;
  int pushes=cold_pushes;

  const bool warm_dimensions=st_.inner_dual_warm_start && p.has_dual_warm_start &&
      (int)p.initial_equality_duals.size()==me &&
      (int)p.initial_inequality_duals.size()==mi;
  const bool full_warm_requested=warm_dimensions && st_.inner_inequality_dual_warm_start;
  const bool slack_warm_dimensions=full_warm_requested &&
      p.has_inequality_slack_warm_start &&
      static_cast<int>(p.initial_inequality_slacks.size())==mi &&
      finite_vec(p.initial_inequality_slacks) &&
      std::all_of(p.initial_inequality_slacks.begin(),
                  p.initial_inequality_slacks.end(),
                  [](double value){ return value>0.0; });
  bool warm_accepted=false;
  bool zero_dual_continuation_attempted=false;
  bool zero_dual_continuation_constructed=true;
  int zero_dual_continuation_repairs=0;
  double single_loop_bootstrap_reference_residual=0.0;
  StateEval warm_eval;

  if(warm_dimensions) {
    Vec warm_lambda=p.initial_equality_duals;
    Vec warm_z=cold_z, warm_s=cold_s;
    int warm_pushes=cold_pushes;
    if(full_warm_requested) {
      warm_pushes=0;
      for(int i=0;i<mi;++i) {
        const ModelSupport* support=
            supports && (*supports)[static_cast<std::size_t>(i)].eligible
                ? &(*supports)[static_cast<std::size_t>(i)] : nullptr;
        const double phi=support
            ? model_value_support(
                  p.inequalities[static_cast<std::size_t>(i)],y,*support,&out)
            : p.inequalities[i].value(y);
        const double raw=std::max(0.0,-phi);
        const double push=std::max(st_.minimum_positive,
            std::min(st_.qp_bound_push*std::max(1.0,std::abs(phi)),
                     st_.inner_bound_frac*std::max(st_.inner_bound_push_reference,raw)));
        // A direct single-loop continuation carries an independent slack.
        // The outer loop has already remapped it to the new majorant origin;
        // retain that strictly-positive value here.  Mapped multiplier-only
        // starts keep the historical centered construction unchanged.
        warm_s[i]=slack_warm_dimensions
            ? std::max(st_.minimum_positive,
                       p.initial_inequality_slacks[i])
            : std::max(raw,push);
        if(warm_s[i]>raw) ++warm_pushes;
        const double zi=p.initial_inequality_duals[i];
        // A direct continuation produced by LazyQP is already strictly
        // positive.  Mapped states can contain exact zeros; for those entries,
        // r3p23 retained the safe centered value.  The opt-in continuation
        // barrier below instead keeps the entry strictly positive while
        // bounding its componentwise complementarity.  It leaves every
        // transported positive multiplier untouched.
        if(std::isfinite(zi) && zi>st_.minimum_positive) {
          warm_z[i]=zi;
        } else if(st_.lazyqp_zero_dual_continuation &&
                  std::isfinite(zi) && zi>=0.0 &&
                  zi<=st_.minimum_positive) {
          zero_dual_continuation_attempted=true;
          ++zero_dual_continuation_repairs;
          // Direct LazyQP continuation may carry the numerical floor rather
          // than an exact zero.  Treat every non-strict entry as a numerical
          // zero without touching any multiplier strictly above it.
          // max(s,mu_cap) makes z<=1 near an active constraint while preserving
          // s*z<=mu_cap.
          const double denominator=std::max(
              warm_s[i],st_.lazyqp_zero_dual_mu_cap);
          const double barrier_value=st_.lazyqp_zero_dual_mu_cap/denominator;
          const double candidate=std::clamp(
              barrier_value,st_.minimum_positive,10.0);
          const double product=warm_s[i]*candidate;
          const double product_guard=st_.lazyqp_zero_dual_mu_cap*
              (1.0+64.0*std::numeric_limits<double>::epsilon());
          if(std::isfinite(candidate) && candidate>0.0 &&
             std::isfinite(product) && product<=product_guard) {
            warm_z[i]=candidate;
          } else {
            // Fail closed transactionally: the ordinary warm-state guard is
            // forced to reject the complete continuation below, leaving
            // lambda/z/s/e at their untouched centered cold values.
            zero_dual_continuation_constructed=false;
            warm_z[i]=cold_z[i];
          }
        } else {
          // A negative inequality multiplier is not a numerical zero.  When
          // the opt-in repair policy is active, reject the complete warm
          // state instead of silently turning invalid dual data into a valid
          // continuation.
          if(st_.lazyqp_zero_dual_continuation &&
             std::isfinite(zi) && zi<0.0)
            zero_dual_continuation_constructed=false;
          warm_z[i]=cold_z[i];
        }
      }
    }

    warm_eval=evaluate_state(
        p,y,warm_lambda,warm_z,warm_s,inequality_gradient_work,nullptr,
        supports,&out);
    if(st_.dca_single_loop_primal_dual && p.allow_dca_single_loop &&
       p.allow_dca_single_loop_dual_bootstrap &&
       full_warm_requested && zero_dual_continuation_constructed) {
      out.dca_single_loop_reduced_bootstrap_attempted=true;
      Vec stationarity_without_lambda=warm_eval.rdual;
      for(int r=0;r<me;++r)
        add_row_t(p.equalities.rows[static_cast<std::size_t>(r)],
                  -warm_lambda[static_cast<std::size_t>(r)],
                  stationarity_without_lambda);
      Vec fitted_lambda,fitted_residual;
      if(fit_free_equality_duals_at_state(
             p,stationarity_without_lambda,st_.multiplier_regularization,
             fitted_lambda,fitted_residual) &&
         inf_norm(fitted_residual)<warm_eval.dual_inf) {
        const double before=warm_eval.dual_inf;
        warm_lambda=std::move(fitted_lambda);
        warm_eval.rdual=std::move(fitted_residual);
        warm_eval.dual_inf=inf_norm(warm_eval.rdual);
        out.dca_single_loop_reduced_bootstrap_accepted=true;
        single_loop_bootstrap_reference_residual=
            forcing_residual_inf(cold_eval,cold_s,cold_z);
        if(st_.verbose)
          std::cout << "[DCA-SINGLE-LOOP-REDUCED-BOOTSTRAP] accepted=1"
                    << " source=mapped-initialization dual="<<std::scientific
                    << before<<"->"<<warm_eval.dual_inf
                    << " equalities="<<me<<"\n"<<std::defaultfloat;
      }
    }
    const double guard=std::max(1.0,st_.lazyqp_warm_start_guard_factor);
    const double dual_ref=std::max(cold_eval.dual_inf,100.0*st_.inner_tolerance);
    const double pri_ref=std::max(cold_eval.pri_inf,100.0*st_.inner_tolerance);
    const double mu_ref=std::max(cold_eval.mu,st_.inner_iteration_limit_mu_fallback);
    const double cold_complementarity_inf=
        complementarity_inf(cold_s,cold_z);
    const double warm_complementarity_inf=
        complementarity_inf(warm_s,warm_z);
    const double complementarity_ref=std::max(
        cold_complementarity_inf,st_.inner_iteration_limit_mu_fallback);
    const bool complete_single_loop_transport=
        !st_.dca_single_loop_primal_dual ||
        !p.allow_dca_single_loop ||
        !p.has_inequality_slack_warm_start ||
        slack_warm_dimensions;
    const bool finite=zero_dual_continuation_constructed &&
        complete_single_loop_transport &&
        finite_vec(warm_lambda)&&finite_vec(warm_z)&&finite_vec(warm_s)&&
        finite_vec(warm_eval.rdual)&&finite_vec(warm_eval.req)&&finite_vec(warm_eval.rpri);
    const bool direct_single_loop_transport=
        st_.dca_single_loop_primal_dual && p.allow_dca_single_loop &&
        full_warm_requested && slack_warm_dimensions &&
        !p.dual_warm_start_is_mapped;
    warm_accepted=finite && (direct_single_loop_transport ||
        (warm_eval.dual_inf<=guard*dual_ref &&
         warm_eval.pri_inf<=guard*pri_ref && warm_eval.mu<=guard*mu_ref &&
         warm_complementarity_inf<=guard*complementarity_ref));
    if(warm_accepted) {
      lambda=std::move(warm_lambda);
      z=std::move(warm_z);
      s=std::move(warm_s);
      e=std::move(warm_eval);
      pushes=warm_pushes;
    }
    if(st_.verbose) {
      std::cout << "[LAZYQP-WARM] requested="<<(full_warm_requested?"full":"equality")
                << " accepted="<<(warm_accepted?1:0)
                << " cold_dual="<<std::scientific<<cold_eval.dual_inf
                << " warm_dual="<<warm_eval.dual_inf
                << " cold_pri="<<cold_eval.pri_inf
                << " warm_pri="<<warm_eval.pri_inf
                << " cold_mu="<<cold_eval.mu
                << " warm_mu="<<warm_eval.mu
                << " cold_comp_inf="<<cold_complementarity_inf
                << " warm_comp_inf="<<warm_complementarity_inf
                << " guard="<<guard<<"\n"<<std::defaultfloat;
    }
  }

  if(st_.dca_single_loop_primal_dual && p.allow_dca_single_loop &&
     p.allow_dca_single_loop_dual_bootstrap && !warm_accepted) {
    out.dca_single_loop_reduced_bootstrap_attempted=true;
    Vec bootstrap_lambda,bootstrap_residual;
    if(fit_free_equality_duals_at_state(
           p,e.rdual,st_.multiplier_regularization,
           bootstrap_lambda,bootstrap_residual) &&
       inf_norm(bootstrap_residual)<e.dual_inf) {
      lambda=std::move(bootstrap_lambda);
      e.rdual=std::move(bootstrap_residual);
      e.dual_inf=inf_norm(e.rdual);
      out.dca_single_loop_reduced_bootstrap_accepted=true;
      single_loop_bootstrap_reference_residual=
          forcing_residual_inf(cold_eval,cold_s,cold_z);
      if(st_.verbose)
        std::cout << "[DCA-SINGLE-LOOP-REDUCED-BOOTSTRAP] accepted=1"
                  << " dual="<<std::scientific<<cold_eval.dual_inf
                  << "->"<<e.dual_inf<<" equalities="<<me
                  << "\n"<<std::defaultfloat;
    } else if(st_.verbose) {
      std::cout << "[DCA-SINGLE-LOOP-REDUCED-BOOTSTRAP] accepted=0\n";
    }
  }

  out.dual_warm_start_used=warm_accepted;
  out.full_dual_warm_start_used=warm_accepted && full_warm_requested;
  out.dual_warm_start_rejected=warm_dimensions && !warm_accepted;
  if(st_.dca_single_loop_primal_dual &&
     p.has_inequality_slack_warm_start) {
    out.dca_single_loop_state_transport_used=
        slack_warm_dimensions && warm_accepted;
    out.dca_single_loop_state_transport_rejected=
        !out.dca_single_loop_state_transport_used;
  }
  if(zero_dual_continuation_attempted) {
    out.lazyqp_zero_dual_continuation_attempts=1;
    if(p.dual_warm_start_is_mapped)
      out.lazyqp_zero_dual_continuation_mapped_attempts=1;
    else
      out.lazyqp_zero_dual_continuation_direct_attempts=1;
    out.lazyqp_zero_dual_continuation_repairs=
        zero_dual_continuation_repairs;
    if(warm_accepted)
      out.lazyqp_zero_dual_continuation_acceptances=1;
    else
      out.lazyqp_zero_dual_continuation_fallbacks=1;
  }
  out.initialization_bound_pushes=pushes;

  out.initial_primal_residual_inf=std::max(e.eq_inf,e.pri_inf);
  out.initial_complementarity=e.mu;
  const double initial_dual_inf=e.dual_inf;
  // Compute the complete forcing contract only when it can be used.  In
  // particular, --no-lazyqp-inexact-forcing must not trigger the independent
  // centered full-model state evaluation below on every majorant.  The two
  // inexact contracts are mutually exclusive: single-loop contracts from its
  // transported state and must not revive forcing-only work.
  const bool forcing_enabled=st_.lazyqp_inexact_forcing &&
      !st_.dca_single_loop_primal_dual &&
      std::isfinite(st_.lazyqp_forcing_eta) &&
      st_.lazyqp_forcing_eta>0.0 && st_.lazyqp_forcing_eta<1.0 &&
      std::isfinite(st_.lazyqp_forcing_target_tolerance) &&
      std::isfinite(st_.lazyqp_forcing_primal_tolerance) &&
      st_.lazyqp_forcing_target_tolerance>st_.inner_tolerance &&
      st_.lazyqp_forcing_primal_tolerance>0.0 &&
      st_.lazyqp_forcing_fast_iterations>=st_.lazyqp_forcing_min_iterations &&
      st_.lazyqp_forcing_min_iterations>=1;
  double initial_forcing_residual=0.0;
  if(forcing_enabled) {
    const double actual_initial_forcing_residual=
        forcing_residual_inf(e,s,z);
    const double centered_initial_forcing_residual=
        centered_origin_forcing_residual(p,st_);
    ++out.lazyqp_inexact_forcing_model_evaluations;
    initial_forcing_residual=std::min(
        actual_initial_forcing_residual,centered_initial_forcing_residual);
  }
  if(st_.verbose && primal_warm_dimensions) {
    std::cout << "[LAZYQP-PRIMAL-TRANSPORT] used=1"
              << " pri=" << std::scientific << out.initial_primal_residual_inf
              << " dual=" << initial_dual_inf
              << " mu=" << out.initial_complementarity
              << "\n" << std::defaultfloat;
  }

  // The non-dedicated auxiliary path returned to the historical IPM above.
  // From this point on the model is necessarily a dedicated Lazy-DCA majorant;
  // failures must be reported directly rather than advertised as an unreachable
  // legacy-IPM fallback.
  auto fail_dedicated = [&](const std::string& reason) -> InnerResult {
    InnerResult fail=out;
    fail.message=reason;
    fail.y=y;
    fail.equality_duals=lambda;
    fail.inequality_duals=z;
    fail.inequality_slacks=s;
    return fail;
  };

  if(st_.verbose) {
    std::cout << "[LAZYQP-INIT] warm="<<(warm_accepted?1:0)
              << " fullwarm="<<(out.full_dual_warm_start_used?1:0)
              << " warmreject="<<(out.dual_warm_start_rejected?1:0)
              << " centered=1"
              << " mode=primal-dual-nonlinear-residual"
              << " pushes="<<pushes
              << " dual="<<std::scientific<<e.dual_inf
              << " pri="<<e.pri_inf
              << " mu="<<e.mu<<"\n"<<std::defaultfloat;
  }

  std::string msg;
  const bool have_factor=static_cast<bool>(full_factor_) ||
                         static_cast<bool>(reduced_factor_) ||
                         static_cast<bool>(hybrid_factor_);
  const bool compatible=!have_factor ||
      (factor_backend_==st_.linear_backend &&
       factor_metis_==st_.linear_metis && factor_seed_==st_.random_seed);
  if(!st_.kkt_persistent_symbolic_cache || !compatible)
    clear_factorization_cache();
  FactorStatisticsDelta factor_statistics(
      out,full_factor_,reduced_factor_,hybrid_factor_);

  // Select the rows retained by the hybrid inequality Schur complement once
  // from the initial Jacobian of this convex subproblem.  The selection then
  // remains frozen for every Newton step, so numerical zeros cannot churn the
  // KKT graph.  Retaining a row replaces k(k+1)/2 outer-product triplets by
  // k Jacobian entries and one diagonal entry.
  std::vector<int> hybrid_retained_rows;
  std::vector<int> hybrid_retained_slot(static_cast<std::size_t>(mi),-1);
  if(st_.lazyqp_kkt_form=="hybrid" &&
     st_.lazyqp_hybrid_kkt_max_retained_inequalities>0) {
    struct Candidate {
      long long savings=0;
      int row=0;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(mi));
    for(int i=0;i<mi;++i) {
      const std::size_t begin=e.J.row_offsets[static_cast<std::size_t>(i)];
      const std::size_t end=e.J.row_offsets[static_cast<std::size_t>(i+1)];
      const long long nnz=static_cast<long long>(end-begin);
      if(nnz<st_.lazyqp_hybrid_kkt_min_row_nnz) continue;
      const long long condensed=nnz*(nnz+1)/2;
      const long long augmented=nnz+1;
      if(condensed>augmented)
        candidates.push_back({condensed-augmented,i});
    }
    std::sort(candidates.begin(),candidates.end(),
              [](const Candidate& lhs,const Candidate& rhs) {
                if(lhs.savings!=rhs.savings) return lhs.savings>rhs.savings;
                return lhs.row<rhs.row;
              });
    const int keep=std::min(
        static_cast<int>(candidates.size()),
        st_.lazyqp_hybrid_kkt_max_retained_inequalities);
    hybrid_retained_rows.reserve(static_cast<std::size_t>(keep));
    for(int k=0;k<keep;++k)
      hybrid_retained_rows.push_back(candidates[static_cast<std::size_t>(k)].row);
    std::sort(hybrid_retained_rows.begin(),hybrid_retained_rows.end());
    for(int k=0;k<static_cast<int>(hybrid_retained_rows.size());++k)
      hybrid_retained_slot[static_cast<std::size_t>(
          hybrid_retained_rows[static_cast<std::size_t>(k)])]=k;
    if(hybrid_retained_rows.empty())
      ++out.lazyqp_hybrid_kkt_fallbacks;
  }

  const bool adaptive=st_.kkt_adaptive_regularization &&
      (st_.linear_backend=="native"||st_.linear_backend=="ldl"||
       st_.linear_backend=="qdldl"||st_.linear_backend=="cholesk"||
       st_.linear_backend=="mumps");
  double rho_x=adaptive?std::max(st_.kkt_rho_x_floor,st_.kkt_primal_regularization):st_.kkt_primal_regularization;
  double delta_boost=1.0;
  auto grow_regularization = [&](const char* reason,int iteration,double alpha) {
    if(!adaptive || rho_x>=st_.kkt_rho_x_max) return false;
    const double old=rho_x;
    const double growth=std::max(1.1,st_.kkt_rho_x_growth);
    rho_x=std::min(st_.kkt_rho_x_max,
        std::max(st_.kkt_rho_x_floor,std::max(old*growth,st_.kkt_rho_x_floor*growth)));
    if(!(rho_x>old)) return false;
    delta_boost=std::min(1e12,delta_boost*growth);
    ++out.kkt_regularization_retries;
    if(st_.verbose)
      std::cout << "[LAZYQP-REG] it="<<iteration
                << " reason="<<reason
                << " alpha="<<std::scientific<<alpha
                << " rho="<<old<<"->"<<rho_x
                << " delta_boost="<<delta_boost<<"\n"<<std::defaultfloat;
    return true;
  };

  // This validity is deliberately local to one convex majorant.  The factor
  // objects remain persistent across outer iterations for their symbolic
  // cache, but a numerical factorization is never trusted across two different
  // subproblems.
  bool numeric_factor_valid=false;
  bool numeric_factor_reduced=false;
  int numeric_factor_dimension=0;
  int numeric_factor_age=0;

  // The forcing target is deliberately separate from inner_tolerance: it may
  // permit an inexact majorant direction, but it must not weaken any of the
  // historical strict/fallback tests below.  Invalid or terminal-sized
  // targets disable the fast path and therefore fail closed.
  const double forcing_relative_target=forcing_enabled
      ? st_.lazyqp_forcing_eta*std::max(initial_forcing_residual,
                                        st_.inner_tolerance)
      : 0.0;
  const double forcing_target=forcing_enabled
      ? std::min(st_.lazyqp_forcing_target_tolerance,
                 forcing_relative_target)
      : 0.0;
  double forcing_model_start=0.0;
  if(forcing_enabled) {
    forcing_model_start=p.objective.value(
        Vec(static_cast<std::size_t>(n),0.0));
    ++out.lazyqp_inexact_forcing_model_evaluations;
  }
  bool forcing_fallback_reported=false;
  out.lazyqp_inexact_forcing_attempted=forcing_enabled;
  if(forcing_enabled) {
    out.lazyqp_inexact_forcing_initial_residual=initial_forcing_residual;
    out.lazyqp_inexact_forcing_outer_target=
        st_.lazyqp_forcing_target_tolerance;
    out.lazyqp_inexact_forcing_target=forcing_target;
    out.lazyqp_inexact_forcing_primal_target=
        st_.lazyqp_forcing_primal_tolerance;
  }

  const bool single_loop_configuration_valid=
      st_.dca_single_loop_inner_steps==1 &&
      std::isfinite(st_.dca_single_loop_forcing_eta) &&
      st_.dca_single_loop_forcing_eta>0.0 &&
      st_.dca_single_loop_forcing_eta<1.0 && st_.qp_mehrotra;
  if(st_.dca_single_loop_primal_dual && p.allow_dca_single_loop &&
     !single_loop_configuration_valid)
    return fail_dedicated(
        "invalid single-loop primal-dual configuration");
  // A cold first majorant is a centered single-loop transaction.  A later
  // majorant transports (y,lambda,z,s) when its complete warm-state guard
  // passes and otherwise restarts from the same safe centered construction.
  const bool single_loop_route=st_.dca_single_loop_primal_dual &&
      p.allow_dca_single_loop;
  // A rejected/missing warm state falls back to the safe centered state, but
  // still receives exactly one transactional predictor-corrector step.  It no
  // longer triggers an almost-complete bootstrap solve.
  bool single_loop_active=single_loop_route;
  // API callers do not pass through the CLI validation above.  Keep the
  // execution bound independent of mutable settings after validating that
  // the public value is exactly one.  A zero-step return remains legitimate
  // only when the transported state already satisfies the strict contract.
  constexpr int single_loop_step_limit=1;
  int single_loop_steps_taken=0;
  Vec single_loop_start_y;
  Vec single_loop_start_lambda;
  Vec single_loop_start_z;
  Vec single_loop_start_s;
  StateEval single_loop_start_eval;
  double single_loop_start_residual=0.0;
  double single_loop_start_primal=0.0;
  double single_loop_start_model=0.0;
  if(single_loop_route) {
    single_loop_start_y=y;
    single_loop_start_lambda=lambda;
    single_loop_start_z=z;
    single_loop_start_s=s;
    single_loop_start_eval=e;
    const double state_start_residual=forcing_residual_inf(e,s,z);
    // The reduced dual bootstrap and this PC step form one initialization
    // transaction.  Contract against the deterministic centered state that
    // entered the transaction, while rollback still retains the fitted state
    // so a proximal retry never repeats the fit.
    single_loop_start_residual=single_loop_bootstrap_reference_residual>0.0
        ? std::max(state_start_residual,
                   single_loop_bootstrap_reference_residual)
        : state_start_residual;
    single_loop_start_primal=std::max({e.eq_inf,e.pri_inf,e.ineq_inf});
    single_loop_start_model=p.objective.value(y);
  }
  const int single_loop_factorizations_begin=out.factorizations;
  out.dca_single_loop_initial_residual=single_loop_route
      ? single_loop_start_residual : 0.0;
  if(single_loop_route) {
    out.dca_single_loop_initial_primal=single_loop_start_y;
    out.dca_single_loop_initial_equality_duals=single_loop_start_lambda;
    out.dca_single_loop_initial_inequality_duals=single_loop_start_z;
    out.dca_single_loop_initial_inequality_slacks=single_loop_start_s;
  }

  auto rollback_single_loop = [&](const char* reason) {
    const double rejected_residual=forcing_residual_inf(e,s,z);
    y=single_loop_start_y;
    lambda=single_loop_start_lambda;
    z=single_loop_start_z;
    s=single_loop_start_s;
    e=single_loop_start_eval;
    out.dca_single_loop_retryable=true;
    out.dca_single_loop_steps=single_loop_steps_taken;
    out.dca_single_loop_factorizations=std::max(
        0,out.factorizations-single_loop_factorizations_begin);
    out.dca_single_loop_final_residual=rejected_residual;
    out.y=y;
    out.equality_duals=lambda;
    out.inequality_duals=z;
    out.inequality_slacks=s;
    out.dual_inf=e.dual_inf;
    out.equality_inf=e.eq_inf;
    out.inequality_inf=e.ineq_inf;
    out.primal_residual_inf=e.pri_inf;
    out.complementarity=e.mu;
    out.message=std::string("retryable single-loop rollback: ")+reason;
    if(st_.verbose)
      std::cout << "[DCA-SINGLE-LOOP] accepted=0 reason="<<reason
                << " action=rollback-outer-proximal-retry"
                << " steps="<<single_loop_steps_taken
                << " fact="<<out.dca_single_loop_factorizations
                << "\n";
  };

  int total_iterations=0;
  // The single-loop route owns one predictor-corrector iteration.  Historical
  // strict solves retain their ordinary inner budget when the route is off.
  const int total_iteration_budget=st_.max_inner_iterations+
      (single_loop_active?single_loop_step_limit:0);
  for(int iter=0;iter<total_iteration_budget;++iter) {
    // e is the exact state computed either by initialization or by the accepted
    // preceding trial.  Do not evaluate the same point a second time.
    out.iterations=++total_iterations; out.dual_inf=e.dual_inf; out.equality_inf=e.eq_inf;
    out.inequality_inf=e.ineq_inf; out.primal_residual_inf=e.pri_inf; out.complementarity=e.mu;
    const double residual_inf=std::max({e.dual_inf,e.eq_inf,e.pri_inf});
    const double model_tol=std::max(st_.inner_tolerance,st_.qp_model_solved_tol);
    const bool strict=residual_inf<=model_tol && e.mu<=model_tol;
    const double guarded_residual_tol=100.0*st_.inner_tolerance;
    // Inexact inner contract: once the convex QCQP residual is within two decades of the
    // requested inner tolerance and complementarity is below the explicit
    // fallback tolerance, returning the point is preferable to numerical
    // micro-steps.  This is an inexact subproblem solve, not an outer KKT
    // certificate.
    const bool guarded=residual_inf<=guarded_residual_tol &&
        e.mu<=st_.inner_iteration_limit_mu_fallback;
    if(strict||guarded) {
      // Keep the public forcing counters a partition of eligible
      // subproblems: a model that reaches the historical contract without an
      // inexact acceptance is a strict fallback, even when it converges
      // before the nominal fast window closes.
      if(forcing_enabled)
        out.lazyqp_inexact_forcing_strict_fallback=true;
      out.success=true; out.used_iteration_limit_fallback=false;
      out.message=guarded&&!strict
        ? "convex QCQP accepted by inexact nonlinear-residual LazyQP contract"
        : "convex QCQP solved by LazyIPDCA-QP nonlinear-residual primal-dual kernel";
      out.y=y; out.equality_duals=lambda; out.inequality_duals=z;
      out.inequality_slacks=s;
      factor_statistics.apply();
      return out;
    }

    if(single_loop_active && single_loop_steps_taken==0)
      out.dca_single_loop_attempted=true;

    if(forcing_enabled && iter>=st_.lazyqp_forcing_min_iterations &&
       iter<=st_.lazyqp_forcing_fast_iterations) {
      const double forcing_residual=forcing_residual_inf(e,s,z);
      const double forcing_primal=std::max(
          {e.eq_inf,e.pri_inf,e.ineq_inf});
      const double forcing_model_final=p.objective.value(y);
      ++out.lazyqp_inexact_forcing_model_evaluations;
      const double model_noise=128.0*std::numeric_limits<double>::epsilon()*
          std::max({1.0,std::abs(forcing_model_start),
                    std::abs(forcing_model_final)});
      // This is an exact convex-model decrease test, not an optimality-gap
      // estimate.  Direction quality is controlled separately by contraction
      // of the complete model KKT residual and by the original-QCQP
      // globalization in Solver.
      const bool residual_ok=forcing_residual<=forcing_target;
      const bool primal_ok=forcing_primal<=
          st_.lazyqp_forcing_primal_tolerance;
      const bool model_ok=std::isfinite(forcing_model_final) &&
          forcing_model_start-forcing_model_final>
              model_noise;
      const bool finite=finite_vec(y)&&finite_vec(lambda)&&finite_vec(z)&&
          finite_vec(s)&&std::isfinite(forcing_residual);
      const bool interior=std::all_of(
          s.begin(),s.end(),[&](double value) {
            return value>st_.minimum_positive;
          }) && std::all_of(z.begin(),z.end(),[&](double value) {
            return value>st_.minimum_positive;
          });
      if(finite && interior && residual_ok && primal_ok && model_ok) {
        out.success=true;
        out.used_iteration_limit_fallback=false;
        out.lazyqp_inexact_forcing_accepted=true;
        out.lazyqp_inexact_forcing_target=forcing_target;
        out.lazyqp_inexact_forcing_primal_target=
            st_.lazyqp_forcing_primal_tolerance;
        out.message=
            "convex QCQP accepted by guarded outer-residual forcing sequence";
        out.y=y;
        out.equality_duals=lambda;
        out.inequality_duals=z;
        out.inequality_slacks=s;
        if(st_.verbose)
          std::cout << "[LAZYQP-FORCING] accepted=1 it="<<iter
                    << " residual="<<std::scientific<<forcing_residual
                    << "/"<<forcing_target
                    << " primal="<<forcing_primal<<"/"
                    << st_.lazyqp_forcing_primal_tolerance
                    << " model_decrease="
                    << forcing_model_start-forcing_model_final
                    << " noise="<<model_noise
                    << "\n"<<std::defaultfloat;
        factor_statistics.apply();
        return out;
      }
    }
    if(forcing_enabled && !forcing_fallback_reported &&
       iter>st_.lazyqp_forcing_fast_iterations) {
      forcing_fallback_reported=true;
      out.lazyqp_inexact_forcing_strict_fallback=true;
      if(st_.verbose)
        std::cout << "[LAZYQP-FORCING] accepted=0 action=strict-fallback"
                  << " fast_iters="<<st_.lazyqp_forcing_fast_iterations
                  << " target="<<std::scientific<<forcing_target
                  << "\n"<<std::defaultfloat;
    }

    struct Border { double gamma; const Vec* u; };
    std::vector<Border> borders;
    for(const auto& t:p.objective.low_rank) if(t.beta>1e-18) borders.push_back({t.beta,&t.direction});
    for(int i=0;i<mi;++i) if(z[i]>1e-18)
      for(const auto& t:p.inequalities[i].low_rank) {
        const double g=z[i]*t.beta; if(g>1e-18) borders.push_back({g,&t.direction});
      }

    const int raw_nb=(int)borders.size();
    // KKT representation is an algebraic/sparsity decision, independent of the
    // curvature oracle's exact-eigensolve threshold.  The default auto rule
    // preserves the historical n<=256 behavior, while explicit condensed and
    // bordered modes permit a clean LDL+METIS A/B test.
    const bool condense_low_rank = st_.lazyqp_low_rank_form=="condensed" ||
        (st_.lazyqp_low_rank_form=="auto" &&
         n<=st_.lazyqp_low_rank_condense_threshold);
    const int nb=condense_low_rank?0:raw_nb;
    const int full_eq0=n, full_in0=n+me, full_b0=n+me+mi;
    const int Nfull=n+me+mi+nb;
    // Inequality elimination is independent of the representation selected
    // for the positive-semidefinite low-rank terms.  When those terms remain
    // bordered, retain their exact saddle-point block in the reduced system:
    //
    //   [ H + J' D J   A'   U ] [dy]       [ -r_d + J' D q ]
    //   [ A             -dI   0 ] [dlambda] = [ -r_eq          ]
    //   [ U'             0  -G^-1] [dw]      [  0              ].
    //
    // Only the inequality multiplier block is removed.
    const int retained_count=static_cast<int>(hybrid_retained_rows.size());
    const int reduced_eq0=n;
    const int reduced_retained0=n+me;
    const int reduced_b0=n+me+retained_count;
    const int Nreduced=n+me+retained_count+nb;
    out.lazyqp_full_kkt_dimension=std::max(out.lazyqp_full_kkt_dimension,Nfull);

    // Condensed low-rank values are numerical data for this Newton state.
    // Compute them once, then emit them into the persistent KKT input layout
    // along with every other numerical coefficient below.
    std::vector<double> dense_condensed_low_rank;
    if(condense_low_rank && raw_nb>0) {
      dense_condensed_low_rank.assign((std::size_t)n*n,0.0);
      for(const auto& b:borders) {
        const Vec& u=*b.u;
        for(int j=0;j<n;++j) if(u[j]!=0.0)
          for(int k=j;k<n;++k) if(u[k]!=0.0)
            dense_condensed_low_rank[j+(std::size_t)k*n]+=
                b.gamma*u[j]*u[k];
      }
    }

    auto emit_primal_hessian = [&](auto&& emit) {
      for(const auto& entry:p.objective.base_hessian.entries)
        emit(entry.i,entry.j,entry.v);
      for(int j=0;j<p.objective.shift_dimension;++j)
        emit(j,j,p.objective.scalar_shift);
      for(int i=0;i<mi;++i) if(z[i]>1e-18) {
        for(const auto& entry:p.inequalities[i].base_hessian.entries)
          emit(entry.i,entry.j,z[i]*entry.v);
        for(int j=0;j<p.inequalities[i].shift_dimension;++j)
          emit(j,j,z[i]*p.inequalities[i].scalar_shift);
      }
      if(condense_low_rank && raw_nb>0) {
        for(int k=0;k<n;++k)
          for(int j=0;j<=k;++j) {
            const double value=
                dense_condensed_low_rank[j+(std::size_t)k*n];
            if(std::abs(value)>1e-30) emit(j,k,value);
          }
      }
    };

    const bool reduced_eligible=mi>0;
    const bool reduced_auto=st_.lazyqp_kkt_form=="auto" &&
        st_.linear_backend=="native" && reduced_eligible &&
        mi>=std::max(1,st_.lazyqp_reduced_kkt_min_inequalities);
    const bool reduced_forced=
      (st_.lazyqp_kkt_form=="reduced" || st_.lazyqp_kkt_form=="hybrid") &&
      reduced_eligible;
    bool use_reduced=reduced_auto||reduced_forced;

    double dual_scale=std::max(max_abs(lambda),max_abs(z));
    double delta_d=st_.kkt_dual_regularization;
    Vec schur_diag;
    bool factor_ok=false;
    SymmetricFactorization* factor=nullptr;
    bool reduced_fallback_used=false;
    bool reused_numeric_factor=false;
    std::string factor_failure;
    int attempt=0;
    std::vector<SymmetricTriplet> current_kkt;
    int current_kkt_dimension=0;

    auto current_delta = [&]() {
      double d=st_.kkt_dual_regularization;
      if(adaptive) {
        d=st_.kkt_delta_dual_inverse
            ? st_.kkt_delta_dual_base/(1.0+dual_scale)
            : st_.kkt_delta_dual_base;
        d=std::clamp(d,st_.kkt_delta_dual_floor,st_.kkt_delta_dual_max);
        d=std::min(st_.kkt_delta_dual_max,d*delta_boost);
      }
      return d;
    };

    while(true) {
      delta_d=current_delta();
      const std::vector<SymmetricTriplet>* kkt_entries=nullptr;
      int Nactive=Nfull;
      const bool using_hybrid=use_reduced && retained_count>0;
      if(use_reduced) {
        Nactive=Nreduced;
        schur_diag.assign(mi,0.0);
        bool valid_elimination=true;
        for(int i=0;i<mi;++i) {
          if(hybrid_retained_slot[static_cast<std::size_t>(i)]>=0)
            continue;
          const double zi=std::max(z[i],st_.minimum_positive);
          const double denom=s[i]/zi+delta_d;
          if(!(denom>0.0) || !std::isfinite(denom)) {
            factor_failure="non-positive inequality elimination denominator";
            valid_elimination=false;
            break;
          }
          const double di=1.0/denom;
          schur_diag[i]=di;
        }
        if(!valid_elimination) {
          // Restart this factorization attempt with the full KKT.  The v0.4.9
          // loop-local continue stayed inside the inequality loop, leaving a
          // reduced matrix in the full-factor slot and eventually causing an
          // RHS-dimension mismatch.
          if(using_hybrid) ++out.lazyqp_hybrid_kkt_fallbacks;
          use_reduced=false;
          reduced_fallback_used=true;
          ++out.lazyqp_reduced_kkt_fallbacks;
          attempt=0;
          continue;
        }

      }

      // Emit the exact historical raw triplet order for every KKT form.
      // Duplicate entries remain intentional: all backends aggregate them in
      // this deterministic order.  The cache owns that order, so a matching
      // iteration performs value assignments only and the SuiteSparse LDL
      // backend can reuse its positional triplet-to-CSC scatter unchanged.
      auto emit_kkt = [&](auto&& emit) {
        emit_primal_hessian(emit);
        if(use_reduced) {
          for(int i=0;i<mi;++i) {
            if(hybrid_retained_slot[static_cast<std::size_t>(i)]>=0)
              continue;
            const double di=schur_diag[i];
            const std::size_t row_begin=
                e.J.row_offsets[static_cast<std::size_t>(i)];
            const std::size_t row_end=
                e.J.row_offsets[static_cast<std::size_t>(i+1)];
            for(std::size_t a=row_begin;a<row_end;++a)
              for(std::size_t b=a;b<row_end;++b)
                emit(e.J.columns[a],e.J.columns[b],
                     di*e.J.values[a]*e.J.values[b]);
          }
          for(int r=0;r<me;++r)
            for(const auto& [j,a]:p.equalities.rows[r].entries)
              if(a!=0.0) emit(j,reduced_eq0+r,a);
          for(int i:hybrid_retained_rows) {
            const int slot=hybrid_retained_slot[static_cast<std::size_t>(i)];
            const int column=reduced_retained0+slot;
            for_each_jacobian_entry(e.J,i,[&](int j,double value) {
              emit(j,column,value);
            });
            const double zi=std::max(z[i],st_.minimum_positive);
            emit(column,column,-s[i]/zi);
          }
          if(!condense_low_rank) {
            for(int b=0;b<nb;++b) {
              const Vec& u=*borders[b].u;
              for(int j=0;j<n;++j) if(u[j]!=0.0)
                emit(j,reduced_b0+b,u[j]);
              emit(reduced_b0+b,reduced_b0+b,-1.0/borders[b].gamma);
            }
          }
          if(adaptive) {
            for(int j=0;j<n;++j) emit(j,j,rho_x);
            for(int r=0;r<me;++r)
              emit(reduced_eq0+r,reduced_eq0+r,-delta_d);
            for(int k=0;k<retained_count;++k)
              emit(reduced_retained0+k,reduced_retained0+k,-delta_d);
          }
        } else {
          for(int r=0;r<me;++r)
            for(const auto& [j,a]:p.equalities.rows[r].entries)
              if(a!=0.0) emit(j,full_eq0+r,a);
          for(int i=0;i<mi;++i) {
            for_each_jacobian_entry(e.J,i,[&](int j,double value) {
              emit(j,full_in0+i,value);
            });
            const double zi=std::max(z[i],st_.minimum_positive);
            emit(full_in0+i,full_in0+i,-s[i]/zi);
          }
          if(!condense_low_rank) {
            for(int b=0;b<nb;++b) {
              const Vec& u=*borders[b].u;
              for(int j=0;j<n;++j) if(u[j]!=0.0)
                emit(j,full_b0+b,u[j]);
              emit(full_b0+b,full_b0+b,-1.0/borders[b].gamma);
            }
          }
          if(adaptive) {
            for(int j=0;j<n;++j) emit(j,j,rho_x);
            for(int r=0;r<me;++r)
              emit(full_eq0+r,full_eq0+r,-delta_d);
            for(int i=0;i<mi;++i)
              emit(full_in0+i,full_in0+i,-delta_d);
          }
        }
      };

      auto& triplet_cache=!use_reduced ? full_triplets_ :
          (using_hybrid ? hybrid_triplets_ : reduced_triplets_);
      bool values_valid=true;
      bool layout_matches=triplet_cache.valid &&
          triplet_cache.dimension==Nactive;
      std::size_t position=0;
      emit_kkt([&](int i,int j,double value) {
        if(i<0 || i>=Nactive || j<0 || j>=Nactive || i>j ||
           (use_reduced && !std::isfinite(value))) {
          values_valid=false;
          return;
        }
        if(layout_matches) {
          if(position>=triplet_cache.entries.size() ||
             triplet_cache.entries[position].i!=i ||
             triplet_cache.entries[position].j!=j) {
            layout_matches=false;
          } else {
            triplet_cache.entries[position].value=value;
          }
        }
        ++position;
      });
      if(layout_matches && position!=triplet_cache.entries.size())
        layout_matches=false;

      if(!values_valid) {
        if(triplet_cache.valid)
          ++out.lazyqp_kkt_triplet_layout_invalidations;
        triplet_cache.clear();
        factor_failure=std::string("invalid ")+
            (use_reduced ? "reduced" : "full")+
            " KKT triplet coordinate or value";
        if(use_reduced) {
          if(using_hybrid) ++out.lazyqp_hybrid_kkt_fallbacks;
          use_reduced=false;
          reduced_fallback_used=true;
          ++out.lazyqp_reduced_kkt_fallbacks;
          attempt=0;
          continue;
        }
        break;
      }

      if(layout_matches) {
        ++out.lazyqp_kkt_triplet_layout_reuses;
      } else {
        if(triplet_cache.valid)
          ++out.lazyqp_kkt_triplet_layout_invalidations;
        std::vector<SymmetricTriplet> rebuilt;
        rebuilt.reserve(position);
        bool rebuild_valid=true;
        emit_kkt([&](int i,int j,double value) {
          if(i<0 || i>=Nactive || j<0 || j>=Nactive || i>j ||
             (use_reduced && !std::isfinite(value))) {
            rebuild_valid=false;
            return;
          }
          rebuilt.push_back({i,j,value});
        });
        triplet_cache.clear();
        if(!rebuild_valid || rebuilt.size()!=position) {
          factor_failure=std::string(use_reduced ? "reduced" : "full")+
              " KKT triplet layout rebuild failed validation";
          if(use_reduced) {
            if(using_hybrid) ++out.lazyqp_hybrid_kkt_fallbacks;
            use_reduced=false;
            reduced_fallback_used=true;
            ++out.lazyqp_reduced_kkt_fallbacks;
            attempt=0;
            continue;
          }
          break;
        }
        triplet_cache.dimension=Nactive;
        triplet_cache.entries=std::move(rebuilt);
        triplet_cache.valid=true;
        ++out.lazyqp_kkt_triplet_layout_builds;
      }
      kkt_entries=&triplet_cache.entries;

      if(st_.lazyqp_numeric_kkt_reuse) {
        current_kkt=*kkt_entries;
        current_kkt_dimension=Nactive;
      }
      msg.clear();
      factor=acquire_factorization(use_reduced,using_hybrid,msg);
      if(!factor) {
        factor_failure="linear factorization backend unavailable ["+
            st_.linear_backend+"]: "+msg;
        break;
      }
      // Defense in depth: public and Solver-level validation already reject
      // numerical-factor reuse for single-loop transactions.  Keep the
      // factorization site closed as well so a future call path cannot turn a
      // one-factor-per-model contract into a zero-factor Krylov reuse.
      const bool may_reuse_numeric=!single_loop_active &&
          st_.lazyqp_numeric_kkt_reuse &&
          numeric_factor_valid && numeric_factor_reduced==use_reduced &&
          numeric_factor_dimension==Nactive &&
          numeric_factor_age<=std::max(0,st_.lazyqp_numeric_kkt_reuse_max_age) &&
          st_.lazyqp_numeric_kkt_reuse_max_iterations>0;
      if(may_reuse_numeric) {
        // The old factors are not used as the Newton matrix.  solve_step below
        // applies the current KKT operator explicitly and uses these factors
        // only as a right preconditioner for a residual-certified FGMRES solve.
        factor_ok=true;
        reused_numeric_factor=true;
        out.lazyqp_kkt_dimension=std::max(out.lazyqp_kkt_dimension,Nactive);
        break;
      }
      ++out.factorizations;
      if(use_reduced) ++out.lazyqp_reduced_kkt_factorizations;
      else ++out.lazyqp_full_kkt_factorizations;
      if(using_hybrid) {
        ++out.lazyqp_hybrid_kkt_attempts;
        out.lazyqp_hybrid_kkt_retained_rows_total+=retained_count;
        out.lazyqp_hybrid_kkt_retained_rows_max=std::max(
            out.lazyqp_hybrid_kkt_retained_rows_max,retained_count);
        for(int i:hybrid_retained_rows) {
          const long long nnz=static_cast<long long>(
              e.J.row_offsets[static_cast<std::size_t>(i+1)]-
              e.J.row_offsets[static_cast<std::size_t>(i)]);
          out.lazyqp_hybrid_kkt_outer_product_triplets_avoided+=
              nnz*(nnz+1)/2;
          out.lazyqp_hybrid_kkt_border_triplets_emitted+=nnz+1;
        }
      }
      if(factor->factor(Nactive,n,*kkt_entries,msg)) {
        factor_ok=true;
        if(using_hybrid) ++out.lazyqp_hybrid_kkt_factorizations;
        numeric_factor_valid=true;
        numeric_factor_reduced=use_reduced;
        numeric_factor_dimension=Nactive;
        numeric_factor_age=0;
        out.lazyqp_kkt_dimension=std::max(out.lazyqp_kkt_dimension,Nactive);
        break;
      }
      factor_failure=msg;

      // A single-loop trial owns at most one numerical factorization.  Any
      // factor failure rejects the complete provisional transaction; the
      // strict fallback below is then free to use its ordinary regularization
      // and reduced/full retry ladder from the untouched starting state.
      if(single_loop_active) break;
      if(adaptive && attempt<std::max(0,st_.kkt_regularization_retry_max)) {
        ++attempt;
        ++out.kkt_regularization_retries;
        rho_x=std::min(st_.kkt_rho_x_max,std::max(st_.kkt_rho_x_floor,
            rho_x>0.0?rho_x*st_.kkt_rho_x_growth:st_.kkt_rho_x_floor));
        delta_boost=std::min(1e12,delta_boost*st_.kkt_rho_x_growth);
        continue;
      }
      if(use_reduced) {
        // The Schur form is algebraically exact, but a numerically awkward
        // reduced inertia should not make the solver less robust than v0.4.7.
        // Re-factor the historical full KKT as a guarded fallback.
        if(using_hybrid) ++out.lazyqp_hybrid_kkt_fallbacks;
        use_reduced=false;
        reduced_fallback_used=true;
        ++out.lazyqp_reduced_kkt_fallbacks;
        attempt=0;
        continue;
      }
      break;
    }

    out.final_kkt_rho_x=rho_x;
    out.max_kkt_rho_x=std::max(out.max_kkt_rho_x,rho_x);
    out.final_kkt_delta_d=delta_d;
    out.max_kkt_delta_d=std::max(out.max_kkt_delta_d,delta_d);
    if(!factor_ok) {
      if(single_loop_active) {
        rollback_single_loop("factorization");
        numeric_factor_valid=false;
        factor_statistics.apply();
        return out;
      }
      factor_statistics.apply();
      return fail_dedicated("LazyQP nonlinear-residual KKT factorization failed: "+factor_failure);
    }

    if(st_.verbose && iter==0) {
      std::cout << "[LAZYQP-KKT] n="<<n<<" eq="<<me<<" ineq="<<mi
                << " lowrank="<<raw_nb
                << " representation="<<(condense_low_rank?"condensed":"bordered")
                << " kkt="<<(use_reduced
                       ? (retained_count>0 ? "ineq-hybrid" : "ineq-reduced")
                       : "full")
                << " retained="<<(use_reduced?retained_count:0)
                << " dimension="<<(use_reduced?Nreduced:Nfull)
                << " full_dimension="<<Nfull
                << " fallback="<<(reduced_fallback_used?1:0)
                << " symcache="<<((factor->symbolic_analyses()>0 ||
                                    factor->has_symbolic_cache())
                                   ? (factor->last_symbolic_reused()?"hit":"miss")
                                   : "n/a")
                << " symbolic="<<((factor->symbolic_analyses()>0 ||
                                     factor->has_symbolic_cache())
                                    ? (factor->last_symbolic_reused()?"reused":"analyzed")
                                    : "n/a")
                << "\n";
    }

    bool numeric_reuse_failed=false;
    auto solve_linear_system=[&](Vec& rhs)->bool {
      if(!reused_numeric_factor) {
        msg.clear();
        return factor->solve(rhs,msg);
      }
      ++out.lazyqp_numeric_kkt_reuse_attempts;
      const Vec exact_rhs=rhs;
      auto apply_current_kkt=[&](const Vec& input,Vec& output) {
        output.assign(static_cast<std::size_t>(current_kkt_dimension),0.0);
        for(const auto& entry:current_kkt) {
          output[static_cast<std::size_t>(entry.i)] +=
              entry.value*input[static_cast<std::size_t>(entry.j)];
          if(entry.i!=entry.j)
            output[static_cast<std::size_t>(entry.j)] +=
                entry.value*input[static_cast<std::size_t>(entry.i)];
        }
      };
      auto apply_preconditioner=[&](const Vec& input,Vec& output,
                                    std::string& failure) {
        output=input;
        return factor->solve(output,failure);
      };
      double rhs_norm2=0.0;
      for(double value:exact_rhs) rhs_norm2+=value*value;
      rhs_norm2=std::sqrt(std::max(0.0,rhs_norm2));
      // Reused factors must reproduce a freshly factorized Newton direction to
      // the solver's "good" backward-error scale, not merely to the looser
      // rejection threshold.  Otherwise several individually acceptable
      // Krylov solves can accumulate enough directional error for the guarded
      // inexact stopping test to return a measurably different minimizer.
      const double relative_tolerance=std::min(
          1e-12,std::max(1e-14,st_.kkt_backward_error_good));
      const double absolute_tolerance=std::max(
          1e-14,relative_tolerance*std::max(rhs_norm2,1e-4));
      Vec correction;
      const auto krylov=original_kkt_right_preconditioned_gmres(
          apply_current_kkt,apply_preconditioner,exact_rhs,correction,
          st_.lazyqp_numeric_kkt_reuse_max_iterations,absolute_tolerance);
      out.lazyqp_numeric_kkt_reuse_iterations+=krylov.iterations;
      if(!krylov.converged || correction.size()!=rhs.size() ||
         !finite_vec(correction)) {
        ++out.lazyqp_numeric_kkt_reuse_fallbacks;
        numeric_reuse_failed=true;
        msg=krylov.message;
        return false;
      }
      ++out.lazyqp_numeric_kkt_reuse_acceptances;
      rhs=std::move(correction);
      return true;
    };

    auto solve_step=[&](const Vec& rcent)->std::pair<bool,NewtonStep> {
      NewtonStep d;
      if(use_reduced) {
        Vec rhs(Nreduced,0.0);
        for(int j=0;j<n;++j) rhs[j]=-e.rdual[j];
        for(int i=0;i<mi;++i) {
          const double zi=std::max(z[i],st_.minimum_positive);
          const double qi=rcent[i]/zi-e.rpri[i];
          const int retained=
              hybrid_retained_slot[static_cast<std::size_t>(i)];
          if(retained>=0) {
            rhs[static_cast<std::size_t>(reduced_retained0+retained)]=qi;
            continue;
          }
          const double w=schur_diag[i]*qi;
          for_each_jacobian_entry(e.J,i,[&](int j,double value) {
            rhs[static_cast<std::size_t>(j)]+=value*w;
          });
        }
        for(int r=0;r<me;++r) rhs[n+r]=-e.req[r];
        if(!solve_linear_system(rhs)) return {false,{}};
        d.dy.assign(rhs.begin(),rhs.begin()+n);
        d.dlambda.assign(rhs.begin()+reduced_eq0,
                         rhs.begin()+reduced_eq0+me);
        d.dz.assign(mi,0.0);
        d.ds.assign(mi,0.0);
        for(int i=0;i<mi;++i) {
          const double zi=std::max(z[i],st_.minimum_positive);
          const double qi=rcent[i]/zi-e.rpri[i];
          const double jacobian_step=jacobian_row_dot(e.J,i,d.dy);
          const int retained=
              hybrid_retained_slot[static_cast<std::size_t>(i)];
          d.dz[i]=retained>=0
              ? rhs[static_cast<std::size_t>(reduced_retained0+retained)]
              : schur_diag[i]*(jacobian_step-qi);
          d.ds[i]=-e.rpri[i]-jacobian_step;
        }
      } else {
        Vec rhs(Nfull,0.0);
        for(int j=0;j<n;++j) rhs[j]=-e.rdual[j];
        for(int r=0;r<me;++r) rhs[full_eq0+r]=-e.req[r];
        for(int i=0;i<mi;++i) {
          const double zi=std::max(z[i],st_.minimum_positive);
          rhs[full_in0+i]=rcent[i]/zi-e.rpri[i];
        }
        if(!solve_linear_system(rhs)) return {false,{}};
        d.dy.assign(rhs.begin(),rhs.begin()+n);
        d.dlambda.assign(rhs.begin()+full_eq0,rhs.begin()+full_eq0+me);
        d.dz.assign(rhs.begin()+full_in0,rhs.begin()+full_in0+mi);
        d.ds.assign(mi,0.0);
        for(int i=0;i<mi;++i)
          d.ds[i]=-e.rpri[i]-jacobian_row_dot(e.J,i,d.dy);
      }
      out.max_kkt_backward_error=std::max(out.max_kkt_backward_error,factor->last_backward_error());
      out.max_kkt_residual_inf=std::max(out.max_kkt_residual_inf,factor->last_residual_inf());
      out.kkt_refinement_steps+=factor->last_refinement_steps();
      const bool ok=finite_vec(d.dy)&&finite_vec(d.dlambda)&&
          finite_vec(d.dz)&&finite_vec(d.ds);
      return {ok,std::move(d)};
    };

    // Primal-normal component from the already factorized KKT matrix.
    // This is a backsolve only (no symbolic or numerical factorization).  It
    // lets the one-step path globalize the tangent primal-dual correction
    // without scaling away the affine equality correction itself.
    auto solve_primal_normal=[&]()->std::pair<bool,NewtonStep> {
      NewtonStep normal;
      if(use_reduced) {
        Vec rhs(Nreduced,0.0);
        // Zero dual/centering residual, full linearized feasibility RHS.
        // In the reduced system q_i=-r_pri,i.
        for(int i=0;i<mi;++i) {
          const int retained=
              hybrid_retained_slot[static_cast<std::size_t>(i)];
          if(retained>=0) {
            rhs[static_cast<std::size_t>(reduced_retained0+retained)]=
                -e.rpri[i];
            continue;
          }
          const double w=-schur_diag[i]*e.rpri[i];
          for_each_jacobian_entry(e.J,i,[&](int j,double value) {
            rhs[static_cast<std::size_t>(j)]+=value*w;
          });
        }
        for(int r=0;r<me;++r) rhs[n+r]=-e.req[r];
        if(!solve_linear_system(rhs)) return {false,{}};
        normal.dy.assign(rhs.begin(),rhs.begin()+n);
        normal.dlambda.assign(rhs.begin()+reduced_eq0,
                              rhs.begin()+reduced_eq0+me);
        normal.dz.assign(mi,0.0);
        normal.ds.assign(mi,0.0);
        for(int i=0;i<mi;++i) {
          const double jacobian_step=jacobian_row_dot(e.J,i,normal.dy);
          const int retained=
              hybrid_retained_slot[static_cast<std::size_t>(i)];
          normal.dz[i]=retained>=0
              ? rhs[static_cast<std::size_t>(reduced_retained0+retained)]
              : schur_diag[i]*(jacobian_step+e.rpri[i]);
          normal.ds[i]=-e.rpri[i]-jacobian_step;
        }
      } else {
        Vec rhs(Nfull,0.0);
        for(int r=0;r<me;++r) rhs[full_eq0+r]=-e.req[r];
        for(int i=0;i<mi;++i)
          rhs[full_in0+i]=-e.rpri[i];
        if(!solve_linear_system(rhs)) return {false,{}};
        normal.dy.assign(rhs.begin(),rhs.begin()+n);
        normal.dlambda.assign(rhs.begin()+full_eq0,
                              rhs.begin()+full_eq0+me);
        normal.dz.assign(rhs.begin()+full_in0,rhs.begin()+full_in0+mi);
        normal.ds.assign(mi,0.0);
        for(int i=0;i<mi;++i)
          normal.ds[i]=-e.rpri[i]-jacobian_row_dot(e.J,i,normal.dy);
      }
      const bool ok=finite_vec(normal.dy)&&finite_vec(normal.dlambda)&&
          finite_vec(normal.dz)&&finite_vec(normal.ds);
      return {ok,std::move(normal)};
    };

    Vec rcent_aff(mi,0.0); for(int i=0;i<mi;++i) rcent_aff[i]=s[i]*z[i];
    auto [ok_aff,aff]=solve_step(rcent_aff);
    if(!ok_aff) {
      if(numeric_reuse_failed) {
        if(single_loop_active) {
          rollback_single_loop("affine-predictor-reuse");
          numeric_factor_valid=false;
          factor_statistics.apply();
          return out;
        }
        // Repeat this exact state with a fresh factorization.  No iterate,
        // multiplier, barrier value or regularization has been changed.
        numeric_factor_valid=false;
        --iter;
        continue;
      }
      if(single_loop_active) {
        rollback_single_loop("affine-predictor");
        numeric_factor_valid=false;
        factor_statistics.apply();
        return out;
      }
      if(grow_regularization("affine-solve",iter+1,0.0)) continue;
      factor_statistics.apply();
      return fail_dedicated("LazyQP affine predictor solve failed");
    }
    const double aap=mi?max_step_positive(s,aff.ds,1.0):1.0;
    const double aad=mi?max_step_positive(z,aff.dz,1.0):1.0;
    double mu_aff=0.0;
    if(mi) { for(int i=0;i<mi;++i) mu_aff+=(s[i]+aap*aff.ds[i])*(z[i]+aad*aff.dz[i]); mu_aff/=mi; }
    double sigma=0.0;
    NewtonStep d;
    if(st_.qp_mehrotra) {
      if(mi && e.mu>0.0) {
        const double ratio=std::clamp(mu_aff/e.mu,0.0,1.0);
        sigma=ratio*ratio*ratio;
        // LazyIPDCA-QP continuation contract: never let the centering parameter
        // cancel barrier reduction completely on an already-convex subproblem.
        sigma=std::min(sigma,std::clamp(st_.lazyqp_barrier_r,0.0,0.95));
        sigma=std::min(sigma,st_.qp_sigma_max);
      }
      Vec rcent(mi,0.0);
      // In a one-step continuation the affine predictor can be very long
      // while positivity admits only a small fraction.  Injecting the
      // *unscaled* second-order product then makes the corrector 10--100x
      // larger than the predictor and collapses the admissible step.  Use the
      // product at the actually admissible affine endpoint.  Ordinary
      // multi-iteration LazyQP retains the historical Mehrotra formula.
      const double affine_product_scale=single_loop_active ? aap*aad : 1.0;
      for(int i=0;i<mi;++i)
        rcent[i]=s[i]*z[i]+affine_product_scale*aff.ds[i]*aff.dz[i]
            -sigma*e.mu;
      auto corrected=solve_step(rcent);
      if(!corrected.first) {
        if(numeric_reuse_failed) {
          if(single_loop_active) {
            rollback_single_loop("mehrotra-corrector-reuse");
            numeric_factor_valid=false;
            factor_statistics.apply();
            return out;
          }
          numeric_factor_valid=false;
          --iter;
          continue;
        }
        if(single_loop_active) {
          rollback_single_loop("mehrotra-corrector");
          numeric_factor_valid=false;
          factor_statistics.apply();
          return out;
        }
        if(grow_regularization("corrector-solve",iter+1,0.0)) continue;
        factor_statistics.apply();
        return fail_dedicated("LazyQP corrector solve failed");
      }
      d=std::move(corrected.second);
    } else {
      // Affine-scaling diagnostic mode exposed by --no-qp-mehrotra.
      d=std::move(aff);
    }

    NewtonStep equality_normal;
    NewtonStep tangent=d;
    Vec normal_slack=s;
    bool equality_normal_used=false;
    if(single_loop_active &&
       (e.eq_inf>st_.inner_tolerance || e.pri_inf>st_.inner_tolerance)) {
      auto normal_result=solve_primal_normal();
      if(normal_result.first) {
        equality_normal=std::move(normal_result.second);
        axpy(1.0,equality_normal.ds,normal_slack);
        const bool normal_interior=finite_vec(normal_slack) &&
            std::all_of(normal_slack.begin(),normal_slack.end(),
                        [](double value){ return value>0.0; });
        if(normal_interior) {
          axpy(-1.0,equality_normal.dy,tangent.dy);
          axpy(-1.0,equality_normal.ds,tangent.ds);
          equality_normal_used=true;
        }
      }
    }
    const double ap_max=mi
        ? max_step_positive(equality_normal_used?normal_slack:s,
                            equality_normal_used?tangent.ds:d.ds,
                            st_.fraction_to_boundary)
        : 1.0;
    const double ad_max=mi?max_step_positive(z,d.dz,st_.fraction_to_boundary):1.0;
    if(st_.verbose && single_loop_active)
      std::cout << "[DCA-SINGLE-LOOP-DIRECTION] sigma="<<std::scientific
                << sigma<<" affine-ap/ad="<<aap<<"/"<<aad
                <<" corrected-ap/ad="<<ap_max<<"/"<<ad_max
                <<" dy="<<inf_norm(d.dy)
                <<" dlambda="<<inf_norm(d.dlambda)
                <<" dz="<<inf_norm(d.dz)
                <<" ds="<<inf_norm(d.ds)
                <<" normal="<<(equality_normal_used?inf_norm(equality_normal.dy):0.0)
                <<"\n"<<std::defaultfloat;
    // Primal and dual positivity generally impose very different maximal
    // steps.  In particular, a nearly active slack may give ap_max=O(1e-3)
    // while the multiplier step remains essentially full.  Forcing the common
    // step min(ap_max,ad_max), as v0.4.6 did, throttles stationarity by the most
    // restrictive slack and produces the long linear tail observed on A4X12.
    //
    // Keep the Newton pair globally coupled through ONE backtracking scalar
    // tau, but preserve the two fraction-to-boundary scales:
    //
    //       alpha_primal = tau * ap_max,
    //       alpha_dual   = tau * ad_max.
    //
    // This is different from v0.4.5, where primal and dual backtracking could
    // evolve independently after the trial tests.  A rejected nonlinear trial
    // now damps both components by the same tau, so an underconstrained primal
    // direction cannot be paired with an unrelated large dual update.
    double tau=1.0;

    const double beta=std::clamp(st_.lazyqp_line_search_beta,0.05,0.95);
    const double primal_now=std::max(e.eq_inf,e.pri_inf);
    const double primal_limit=std::max({100.0*st_.inner_tolerance,
                                       10.0*std::max(primal_now,st_.inner_tolerance),
                                       0.10*e.mu});
    const double dual_limit=std::max({100.0*st_.inner_tolerance,
                                     10.0*std::max(e.dual_inf,st_.inner_tolerance),
                                     0.10*e.mu});
    const double residual_now=std::max({e.dual_inf,e.eq_inf,e.pri_inf});
    const bool centrality_phase=residual_now<=std::max(1e-5,100.0*st_.inner_tolerance);
    const double mu_limit=centrality_phase
        ? std::max(st_.inner_iteration_limit_mu_fallback,0.95*e.mu)
        : std::max(100.0*st_.inner_tolerance,10.0*std::max(e.mu,st_.inner_tolerance));
    TrialState trial;
    int bt=0;
    double alpha_pri=ap_max;
    double alpha_lambda=ad_max;
    double alpha_dual=ad_max;
    Vec lambda_direction_gradient;
    double lambda_direction_norm_sq=0.0;
    if(single_loop_active) {
      lambda_direction_gradient.assign(static_cast<std::size_t>(n),0.0);
      for(int r=0;r<me;++r)
        add_row_t(p.equalities.rows[static_cast<std::size_t>(r)],
                  d.dlambda[static_cast<std::size_t>(r)],
                  lambda_direction_gradient);
      lambda_direction_norm_sq=dot(lambda_direction_gradient,
                                   lambda_direction_gradient);
    }
    for(;bt<=st_.lazyqp_line_search_max;++bt) {
      alpha_pri=tau*ap_max;
      alpha_dual=tau*ad_max;
      alpha_lambda=single_loop_active ? 0.0 : alpha_dual;
      NewtonStep trial_direction=d;
      if(equality_normal_used && alpha_pri>0.0) {
        trial_direction.dy=tangent.dy;
        trial_direction.ds=tangent.ds;
        axpy(1.0/alpha_pri,equality_normal.dy,trial_direction.dy);
        axpy(1.0/alpha_pri,equality_normal.ds,trial_direction.ds);
      }
      trial=make_trial(p,y,lambda,z,s,e,trial_direction,
                       alpha_pri,alpha_lambda,alpha_dual,
                       st_.minimum_positive,st_.fraction_to_boundary,
                       inequality_gradient_work,supports,&out);
      if(trial.ok) {
        // Equality multipliers are sign-free.  In single-loop mode the
        // fraction-to-boundary limit for z must not choke their update.  With
        // y/z/s fixed at the globally admissible trial, choose the scalar
        // along the already-factorized Newton lambda direction that minimizes
        // the dual residual in the Euclidean norm.  This is a one-dimensional
        // residual projection, not a multiplier reconstruction or KKT solve.
        if(single_loop_active && lambda_direction_norm_sq>0.0) {
          alpha_lambda=minimize_direction_inf_norm(
              trial.eval.rdual,lambda_direction_gradient);
          if(alpha_lambda>0.0) {
            axpy(alpha_lambda,d.dlambda,trial.lambda);
            axpy(alpha_lambda,lambda_direction_gradient,trial.eval.rdual);
            trial.eval.dual_inf=inf_norm(trial.eval.rdual);
          }
          trial.alpha_lambda=alpha_lambda;
        }
        const double primal_trial=std::max(trial.eval.eq_inf,trial.eval.pri_inf);
        const bool primal_bad=primal_trial>primal_limit;
        const bool dual_bad=trial.eval.dual_inf>dual_limit;
        const bool mu_bad=trial.eval.mu>mu_limit;
        if(st_.verbose && iter<5 && (primal_bad||dual_bad||mu_bad)) {
          std::cout << "[LAZYQP-LS] it="<<iter+1<<" bt="<<bt
                    << " tau="<<std::scientific<<tau
                    << " ap="<<alpha_pri<<" ad="<<alpha_dual
                    << " pri="<<primal_trial<<"/"<<primal_limit
                    << " dual="<<trial.eval.dual_inf<<"/"<<dual_limit
                    << " mu="<<trial.eval.mu<<"/"<<mu_limit
                    << " bad="<<(primal_bad?"P":"-")<<(dual_bad?"D":"-")
                    <<(mu_bad?"M":"-")
                    << "\n"<<std::defaultfloat;
        }
        if(!primal_bad && !dual_bad && !mu_bad) break;
      }
      tau*=beta;
      if(tau<1e-14) break;
    }
    const bool globalization_ok=trial.ok &&
        std::max(trial.eval.eq_inf,trial.eval.pri_inf)<=primal_limit &&
        trial.eval.dual_inf<=dual_limit && trial.eval.mu<=mu_limit;
    if(!globalization_ok) {
      if(single_loop_active) {
        rollback_single_loop("inner-globalization");
        numeric_factor_valid=false;
        factor_statistics.apply();
        return out;
      }
      if(grow_regularization("globalization",iter+1,tau)) continue;
      factor_statistics.apply();
      return fail_dedicated("LazyQP scaled split-step nonlinear-residual globalization failed");
    }
    if(adaptive && tau<1e-8) {
      if(single_loop_active) {
        rollback_single_loop("tiny-global-step");
        numeric_factor_valid=false;
        factor_statistics.apply();
        return out;
      }
      if(grow_regularization("tiny-global-step",iter+1,tau)) continue;
    }

    out.lazyqp_min_globalization_tau=std::min(out.lazyqp_min_globalization_tau,tau);
    const double primal_dual_ratio=alpha_pri/std::max(alpha_dual,1e-300);
    const double dual_primal_ratio=alpha_dual/std::max(alpha_pri,1e-300);
    out.lazyqp_max_primal_dual_step_ratio=std::max(
        out.lazyqp_max_primal_dual_step_ratio,primal_dual_ratio);
    out.lazyqp_max_dual_primal_step_ratio=std::max(
        out.lazyqp_max_dual_primal_step_ratio,dual_primal_ratio);
    if(std::max(primal_dual_ratio,dual_primal_ratio)>10.0)
      ++out.lazyqp_scaled_split_steps;

    y=std::move(trial.y);
    lambda=std::move(trial.lambda);
    z=std::move(trial.z);
    s=std::move(trial.s);
    e=std::move(trial.eval);
    ++numeric_factor_age;

    if(single_loop_active) {
      ++single_loop_steps_taken;
      out.dca_single_loop_steps=single_loop_steps_taken;
      out.dca_single_loop_factorizations=std::max(
          0,out.factorizations-single_loop_factorizations_begin);
      const double final_residual=forcing_residual_inf(e,s,z);
      out.dca_single_loop_final_residual=final_residual;
      if(single_loop_steps_taken>=single_loop_step_limit) {
        const double final_primal=std::max(
            {e.eq_inf,e.pri_inf,e.ineq_inf});
        const double final_model=p.objective.value(y);
        const double model_noise=128.0*
            std::numeric_limits<double>::epsilon()*
            std::max({1.0,std::abs(single_loop_start_model),
                      std::abs(final_model)});
        // A fraction-to-boundary/globalization step is only a fraction of the
        // full Newton correction.  Scale the requested full-step contraction
        // by the fraction actually applied to both primal and positive-dual
        // variables; demanding eta*R0 after a 1e-3 step is impossible even
        // for the exact linear Newton equation.  The factor remains strictly
        // below one for every accepted interior step and equals eta at a full
        // step.
        const double forcing_step=std::clamp(
            std::min(alpha_pri,alpha_dual),0.0,1.0);
        const double residual_factor=1.0-
            (1.0-st_.dca_single_loop_forcing_eta)*forcing_step;
        const double residual_target=residual_factor*
            std::max(single_loop_start_residual,
                     std::numeric_limits<double>::min());
        // One predictor-corrector step is an inexact model step.  Requiring
        // its nonlinear slack residual to be nonincreasing duplicates (and is
        // stricter than) the globalization gate that selected the trial just
        // above.  Retain that scale-aware gate here; the outer filter checks
        // every original QCQP constraint before committing the endpoint.
        const double primal_guard=primal_limit;
        const bool finite=finite_vec(y)&&finite_vec(lambda)&&
            finite_vec(z)&&finite_vec(s)&&
            std::isfinite(final_residual)&&
            std::isfinite(final_primal)&&std::isfinite(final_model);
        const bool interior=std::all_of(
            s.begin(),s.end(),[](double value){ return value>0.0; }) &&
            std::all_of(
                z.begin(),z.end(),[](double value){ return value>0.0; });
        // Direct kernel users retain the transactional contraction contract.
        // Only the primary Lazy-DCA route may delegate this globalization to
        // the original-QCQP outer filter when forcing was explicitly disabled.
        const bool outer_filter_progress=
            p.allow_dca_single_loop_outer_filter_progress &&
            !st_.lazyqp_inexact_forcing;
        const bool contracted=outer_filter_progress ||
            final_residual<=residual_target;
        const bool primal_ok=final_primal<=primal_guard;
        const bool model_ok=final_model<single_loop_start_model-model_noise;
        const bool feasibility_ok=final_primal<
            (1.0-st_.dca_feasibility_step_fraction)*
                std::max(single_loop_start_primal,st_.inner_tolerance);
        const double model_tol=std::max(
            st_.inner_tolerance,st_.qp_model_solved_tol);
        const double strict_residual=std::max(
            {e.dual_inf,e.eq_inf,e.pri_inf});
        const bool strict=strict_residual<=model_tol && e.mu<=model_tol;
        const bool guarded=strict_residual<=100.0*st_.inner_tolerance &&
            e.mu<=st_.inner_iteration_limit_mu_fallback;
        if(finite && interior && (strict||guarded)) {
          out.success=true;
          out.dca_single_loop_accepted=false;
          out.used_iteration_limit_fallback=false;
          out.message=guarded&&!strict
              ? "single-loop step reached the inexact nonlinear-residual LazyQP contract"
              : "single-loop step solved the convex QCQP strictly";
          out.y=y;
          out.equality_duals=lambda;
          out.inequality_duals=z;
          out.inequality_slacks=s;
          out.dual_inf=e.dual_inf;
          out.equality_inf=e.eq_inf;
          out.inequality_inf=e.ineq_inf;
          out.primal_residual_inf=e.pri_inf;
          out.complementarity=e.mu;
          factor_statistics.apply();
          return out;
        }
        // The Mehrotra step has already passed the nonlinear residual
        // globalization above.  This second guard is the outer-continuation
        // contract: it authorizes only a provisional direction/state and does
        // not claim that the convex majorant was solved.
        if(finite && interior && contracted && primal_ok &&
           (model_ok||(outer_filter_progress&&feasibility_ok))) {
          out.success=false;
          out.dca_single_loop_accepted=true;
          out.used_iteration_limit_fallback=false;
          out.message=
              "guarded provisional single-loop primal-dual state";
          out.y=y;
          out.equality_duals=lambda;
          out.inequality_duals=z;
          out.inequality_slacks=s;
          out.dual_inf=e.dual_inf;
          out.equality_inf=e.eq_inf;
          out.inequality_inf=e.ineq_inf;
          out.primal_residual_inf=e.pri_inf;
          out.complementarity=e.mu;
          if(st_.verbose)
            std::cout << "[DCA-SINGLE-LOOP] accepted=1"
                      << " steps="<<single_loop_steps_taken
                      << " fact="<<out.dca_single_loop_factorizations
                      << " residual="<<std::scientific<<final_residual
                      << "/"<<residual_target
                      << " forcing-step="<<forcing_step
                      << " primal="<<final_primal<<"/"<<primal_guard
                      << " model_decrease="
                      << single_loop_start_model-final_model
                      << "\n"<<std::defaultfloat;
          factor_statistics.apply();
          return out;
        }
        if(st_.verbose)
          std::cout << "[DCA-SINGLE-LOOP-GUARD] finite="<<(finite?1:0)
                    << " interior="<<(interior?1:0)
                    << " contracted="<<(contracted?1:0)
                    << " primal="<<(primal_ok?1:0)
                    << " decrease="<<(model_ok?1:0)
                    << " feasibility="<<(feasibility_ok?1:0)
                    << " residual="<<std::scientific<<final_residual
                    << "/"<<residual_target
                    << " forcing-step="<<forcing_step
                    << " start-dual/eq/pri/comp="
                    << single_loop_start_eval.dual_inf<<"/"
                    << single_loop_start_eval.eq_inf<<"/"
                    << single_loop_start_eval.pri_inf<<"/"
                    << complementarity_inf(single_loop_start_s,
                                            single_loop_start_z)
                    << " final-dual/eq/pri/comp="
                    << e.dual_inf<<"/"<<e.eq_inf<<"/"<<e.pri_inf<<"/"
                    << complementarity_inf(s,z)
                    << " tau/ap/al/ad="<<tau<<"/"<<alpha_pri<<"/"
                    << alpha_lambda<<"/"<<alpha_dual
                    << " model="<<single_loop_start_model<<"->"
                    << final_model
                    << "\n"<<std::defaultfloat;
        rollback_single_loop("residual-or-model-guard");
        numeric_factor_valid=false;
        factor_statistics.apply();
        return out;
      }
    }

    if(adaptive && tau>=0.5 && std::max(alpha_pri,alpha_dual)>=0.5 &&
       rho_x>st_.kkt_rho_x_floor)
      rho_x=std::max(st_.kkt_rho_x_floor,st_.kkt_rho_x_decay*rho_x);

    if(st_.verbose && (iter<5 || (iter+1)%20==0)) {
      std::cout << "[LAZYQP] it="<<iter+1
                << " mu="<<std::scientific<<e.mu
                << " dual="<<e.dual_inf
                << " eq="<<e.eq_inf
                << " pri="<<e.pri_inf
                << " ineq="<<e.ineq_inf
                << " sigma="<<sigma
                << " tau="<<tau
                << " ap="<<alpha_pri<<" ad="<<alpha_dual
                << " apmax="<<ap_max<<" admax="<<ad_max
                << " bt="<<bt
                << " qdef="<<trial.nonlinear_defect_inf
                << " scorr="<<trial.slack_correction_min
                << "\n"<<std::defaultfloat;
    }
  }

  out.y=y; out.equality_duals=lambda; out.inequality_duals=z;
  out.inequality_slacks=s;
  out.dual_inf=e.dual_inf; out.equality_inf=e.eq_inf; out.inequality_inf=e.ineq_inf;
  out.primal_residual_inf=e.pri_inf; out.complementarity=e.mu;
  factor_statistics.apply();

  // Inexact majorant contract at the iteration limit.  Restarting a useful
  // nonlinear-residual trajectory from y=0 discards all primal progress; on
  // HS108 that happened four times in succession.  Return the last point only
  // when it is already primal feasible and complementary, has reduced the
  // initial dual residual by at least a factor of ten, and decreases the
  // exact convex model.  The outer original-QCQP globalization and final KKT
  // certificate remain unchanged, so this is never a convergence shortcut.
  const double fallback_primal_tol=std::max(
      100.0*st_.inner_tolerance,st_.inner_iteration_limit_mu_fallback);
  const double fallback_dual_tol=std::max(
      100.0*st_.inner_tolerance,0.10*initial_dual_inf);
  Vec zero(static_cast<std::size_t>(n),0.0);
  const double model_start=p.objective.value(zero);
  const double model_final=p.objective.value(y);
  const double model_noise=100.0*st_.inner_tolerance*
      std::max(1.0,std::abs(model_start));
  const bool useful_limit_point=finite_vec(y) &&
      std::max(e.eq_inf,e.pri_inf)<=fallback_primal_tol &&
      e.mu<=st_.inner_iteration_limit_mu_fallback &&
      e.dual_inf<=fallback_dual_tol &&
      std::isfinite(model_final) && model_final<=model_start+model_noise;
  if(st_.verbose) {
    std::cout << "[LAZYQP-LIMIT] accept=" << (useful_limit_point?1:0)
              << " pri=" << std::scientific << std::max(e.eq_inf,e.pri_inf)
              << "/" << fallback_primal_tol
              << " dual=" << e.dual_inf << "/" << fallback_dual_tol
              << " mu=" << e.mu << "/" << st_.inner_iteration_limit_mu_fallback
              << " model_delta=" << (model_final-model_start)
              << " noise=" << model_noise
              << "\n" << std::defaultfloat;
  }
  if(useful_limit_point) {
    out.success=true;
    out.used_iteration_limit_fallback=true;
    out.message="convex QCQP accepted at the iteration limit by the guarded inexact-majorant contract";
    return out;
  }
  return fail_dedicated("LazyIPDCA-QP nonlinear-residual kernel iteration limit reached");
}

} // namespace lazyipdca::qcqp
