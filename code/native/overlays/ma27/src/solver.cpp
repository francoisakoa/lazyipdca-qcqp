#include "lazyipdca_qcqp/native_original_anchor_finish.hpp"
#include "closed_zero_face.hpp"
#include "exact_redundant_equalities.hpp"
#include "affine_face_quadratic_rewrite.hpp"
#include "positive_square_epigraph_restoration.hpp"
#include "positive_square_epigraph_substitution.hpp"
#include "common_factor_substitution.hpp"
#include "affine_cycle_certificate.hpp"
#include "convex_aggregate_certificate.hpp"
#include "coefficient_graph_certificate.hpp"
#include "exact_affine_face_presolve.hpp"
#include "affine_face_discovery.hpp"
#include <cstdlib>
#include "power2_scaling.hpp"
#include "product_definition_presolve.hpp"
#include "native_augmented_repair.hpp"
#include "native_al_fallback_engine.hpp"
#include "lazyipdca_qcqp/majorant_equality_stabilization.hpp"
#include "lazyipdca_qcqp/standard_auto.hpp"
#include "native_signed_square_reduction.hpp"
#include "standard_exact_presolve.hpp"
#include "adaptive_equality_probe.hpp"
#include "shared_restoration_budget.hpp"
#include "bounded_origin_certificate.hpp"
#include "feasible_boundary_certificate.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/free_epigraph_presolve.hpp"
#include "lazyipdca_qcqp/lazy_newton.hpp"
#include "lazyipdca_qcqp/dca_filter.hpp"
#include "lazyipdca_qcqp/inner_sparse_single_loop.hpp"
#include "lazyipdca_qcqp/condensed_retraction.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/original_kkt_gmres.hpp"
#include "lazyipdca_qcqp/inner_ipopt.hpp"
#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/inner_socp.hpp"
#include "lazyipdca_qcqp/inner_sqp.hpp"
#include "lazyipdca_qcqp/qcqp_ipopt_termination.hpp"
#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/single_loop_policy.hpp"
#include "lazyipdca_qcqp/single_loop_direct_start.hpp"
#include "lazyipdca_qcqp/single_loop_slack_transport.hpp"
#include "majorant_barrier_retry.hpp"
#include "rank_retry_statistics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

extern "C" {
void dgesv_(const int*, const int*, double*, const int*, int*, double*, const int*, int*);
void dsyev_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*);
}

namespace lazyipdca::qcqp {
namespace {
struct AccumulateSeconds {
  double& total;
  std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
  ~AccumulateSeconds() {
    total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  }
};


ConvexQuadraticModel linear_inequality(int nvar, int idx, double coeff, double constant) {
  ConvexQuadraticModel m;
  m.constant = constant;
  m.linear.assign(nvar, 0.0);
  m.linear[idx] = coeff;
  m.base_hessian.n = nvar;
  return m;
}

SparseRow dense_gradient_row(const Vec& g, double zero_tol = 0.0) {
  SparseRow r;
  for (int j = 0; j < static_cast<int>(g.size()); ++j)
    if (std::abs(g[j]) > zero_tol) r.entries.push_back({j, g[j]});
  return r;
}

// Structural gradient supports are immutable during one Solver::solve call.
// Compile them once instead of rescanning every dense `linear` vector at every
// multiplier refit / stationarity check (case118 has 1016 equality rows whose
// actual supports are tiny in a 1088-variable ambient space).
struct QuadraticGradientSupport {
  std::vector<int> linear;
  std::vector<int> gradient;
};

class ProblemGradientSparsity {
 public:
  explicit ProblemGradientSparsity(const QCQPProblem& problem) {
    supports_.reserve(1 + problem.equalities.size() +
                      problem.inequalities.size());
    add(problem.objective);
    for (const auto& q : problem.equalities) add(q);
    for (const auto& q : problem.inequalities) add(q);
  }

  const QuadraticGradientSupport* find(const QuadraticFunction& q) const {
    const auto it = supports_.find(&q);
    return it == supports_.end() ? nullptr : &it->second;
  }

 private:
  void add(const QuadraticFunction& q) {
    QuadraticGradientSupport support;
    support.linear.reserve(q.linear.size());
    support.gradient.reserve(q.hessian.entries.size() * 2);
    for (const auto& e : q.hessian.entries) {
      support.gradient.push_back(e.i);
      if (e.i != e.j) support.gradient.push_back(e.j);
    }
    for (int j = 0; j < static_cast<int>(q.linear.size()); ++j) {
      if (q.linear[static_cast<std::size_t>(j)] == 0.0) continue;
      support.linear.push_back(j);
      support.gradient.push_back(j);
    }
    std::sort(support.gradient.begin(), support.gradient.end());
    support.gradient.erase(
        std::unique(support.gradient.begin(), support.gradient.end()),
        support.gradient.end());
    supports_.emplace(&q, std::move(support));
  }

  std::unordered_map<const QuadraticFunction*, QuadraticGradientSupport>
      supports_;
};

thread_local const ProblemGradientSparsity* active_gradient_sparsity = nullptr;

class ScopedProblemGradientSparsity {
 public:
  explicit ScopedProblemGradientSparsity(
      const ProblemGradientSparsity& sparsity)
      : previous_(active_gradient_sparsity) {
    active_gradient_sparsity = &sparsity;
  }
  ~ScopedProblemGradientSparsity() {
    active_gradient_sparsity = previous_;
  }
  ScopedProblemGradientSparsity(const ScopedProblemGradientSparsity&) = delete;
  ScopedProblemGradientSparsity& operator=(
      const ScopedProblemGradientSparsity&) = delete;

 private:
  const ProblemGradientSparsity* previous_;
};

const QuadraticGradientSupport* active_gradient_support(
    const QuadraticFunction& q) {
  return active_gradient_sparsity ? active_gradient_sparsity->find(q) : nullptr;
}

double structural_quadratic_value(const QuadraticFunction& q, const Vec& x) {
  if(x.size()!=q.linear.size()) return q.value(x);
  const auto* support=active_gradient_support(q);
  if(!support) return q.value(x);
  double linear=0.0;
  // Ascending nonzero coordinates preserve the effective order of dot();
  // omitted products are exact zeros and cannot change a finite partial sum.
  for(int j:support->linear)
    linear+=q.linear[static_cast<std::size_t>(j)]*
        x[static_cast<std::size_t>(j)];
  return q.constant+linear+0.5*q.hessian.quad(x);
}

Feasibility structural_feasibility(const QCQPProblem& p,const Vec& x) {
  Feasibility f;
  for(const auto& h:p.equalities)
    f.equality_inf=std::max(
        f.equality_inf,std::abs(structural_quadratic_value(h,x)));
  for(const auto& g:p.inequalities)
    f.inequality_inf=std::max(
        f.inequality_inf,
        std::max(0.0,structural_quadratic_value(g,x)));
  for(int j=0;j<p.n;++j) {
    if(std::isfinite(p.lb[static_cast<std::size_t>(j)]))
      f.bound_inf=std::max(
          f.bound_inf,
          std::max(0.0,p.lb[static_cast<std::size_t>(j)]-
                           x[static_cast<std::size_t>(j)]));
    if(std::isfinite(p.ub[static_cast<std::size_t>(j)]))
      f.bound_inf=std::max(
          f.bound_inf,
          std::max(0.0,x[static_cast<std::size_t>(j)]-
                           p.ub[static_cast<std::size_t>(j)]));
  }
  f.total_inf=std::max({f.equality_inf,f.inequality_inf,f.bound_inf});
  return f;
}

// Reusable exact-order workspace for structurally sparse gradients.  The
// dense values are touched only on one quadratic's support and cleared before
// the next row.  Within each coordinate accumulation exactly follows
// SymMatrix::apply's stored-triplet order followed by the linear term; the
// consumer is then visited in ascending coordinate order, like dense axpy.
struct SparseGradientWorkspace {
  Vec values;
  std::vector<unsigned char> touched_mask;
  std::vector<int> touched;

  explicit SparseGradientWorkspace(int n)
      : values(static_cast<std::size_t>(n), 0.0),
        touched_mask(static_cast<std::size_t>(n), 0) {
    touched.reserve(32);
  }
};

template <class Consumer>
void consume_sparse_quadratic_gradient(
    const QuadraticFunction& q, const Vec& x,
    SparseGradientWorkspace& workspace, Consumer&& consume) {
  const int n = static_cast<int>(x.size());
  if (static_cast<int>(q.linear.size()) != n || q.hessian.n != n ||
      static_cast<int>(workspace.values.size()) != n ||
      static_cast<int>(workspace.touched_mask.size()) != n)
    throw std::invalid_argument("sparse gradient dimension mismatch");

  if (const auto* support = active_gradient_support(q)) {
    // The support is already sorted/unique, so neither a dense linear scan nor
    // a per-gradient touched-coordinate sort is needed on the hot path.
    for (const auto& e : q.hessian.entries) {
      workspace.values[static_cast<std::size_t>(e.i)] +=
          e.v * x[static_cast<std::size_t>(e.j)];
      if (e.i != e.j)
        workspace.values[static_cast<std::size_t>(e.j)] +=
            e.v * x[static_cast<std::size_t>(e.i)];
    }
    for (int j : support->linear)
      workspace.values[static_cast<std::size_t>(j)] +=
          q.linear[static_cast<std::size_t>(j)];
    for (int j : support->gradient) {
      consume(j, workspace.values[static_cast<std::size_t>(j)]);
      workspace.values[static_cast<std::size_t>(j)] = 0.0;
    }
    return;
  }

  // Compatibility fallback for internal callers outside Solver::solve.
  auto touch = [&](int j) {
    if (workspace.touched_mask[static_cast<std::size_t>(j)]) return;
    workspace.touched_mask[static_cast<std::size_t>(j)] = 1;
    workspace.touched.push_back(j);
  };
  for (const auto& e : q.hessian.entries) {
    touch(e.i);
    workspace.values[static_cast<std::size_t>(e.i)] +=
        e.v * x[static_cast<std::size_t>(e.j)];
    if (e.i != e.j) {
      touch(e.j);
      workspace.values[static_cast<std::size_t>(e.j)] +=
          e.v * x[static_cast<std::size_t>(e.i)];
    }
  }
  for (int j = 0; j < n; ++j) {
    const double a = q.linear[static_cast<std::size_t>(j)];
    if (workspace.touched_mask[static_cast<std::size_t>(j)]) {
      workspace.values[static_cast<std::size_t>(j)] += a;
    } else if (a != 0.0) {
      touch(j);
      workspace.values[static_cast<std::size_t>(j)] += a;
    }
  }
  std::sort(workspace.touched.begin(), workspace.touched.end());
  for (int j : workspace.touched) {
    consume(j, workspace.values[static_cast<std::size_t>(j)]);
    workspace.values[static_cast<std::size_t>(j)] = 0.0;
    workspace.touched_mask[static_cast<std::size_t>(j)] = 0;
  }
  workspace.touched.clear();
}

SparseRow multiplier_sparse_gradient(
    const QuadraticFunction& q, const Vec& x,
    const std::vector<unsigned char>& fixed,
    SparseGradientWorkspace& workspace) {
  const int n = static_cast<int>(x.size());
  if (static_cast<int>(fixed.size()) != n)
    throw std::invalid_argument("multiplier sparse gradient dimension mismatch");
  SparseRow row;
  if (const auto* support = active_gradient_support(q))
    row.entries.reserve(support->gradient.size());
  consume_sparse_quadratic_gradient(q, x, workspace,
                                    [&](int j, double value) {
    if (!fixed[static_cast<std::size_t>(j)] && value != 0.0)
      row.entries.push_back({j, value});
  });
  return row;
}

double multiplier_sparse_dot(const SparseRow& row, const Vec& dense) {
  double value = 0.0;
  for (const auto& [j, a] : row.entries)
    value += a * dense[static_cast<std::size_t>(j)];
  return value;
}

void multiplier_sparse_axpy(double alpha, const SparseRow& row, Vec& dense) {
  if (alpha == 0.0) return;
  for (const auto& [j, a] : row.entries)
    dense[static_cast<std::size_t>(j)] += alpha * a;
}

struct StructuralQCQPEvaluation {
  SparseGradientWorkspace workspace;
  explicit StructuralQCQPEvaluation(int n):workspace(n) {}
  double value(const QuadraticFunction& q,const Vec& x) { return structural_quadratic_value(q,x); }
  double violation(const QCQPProblem& p,const Vec& x) { return structural_feasibility(p,x).total_inf; }
  void gradient_axpy(const QuadraticFunction& q,const Vec& x,double scale,Vec& y) {
    consume_sparse_quadratic_gradient(q,x,workspace,[&](int j,double v) {
      y[static_cast<std::size_t>(j)]+=scale*v;
    });
  }
};

void add_scaled_hessian(SymMatrix& dst, const SymMatrix& src, double alpha) {
  if (alpha == 0.0) return;
  for (const auto& e : src.entries) dst.entries.push_back({e.i, e.j, alpha*e.v});
}

using MultiplierEstimate = QCQPTerminationMultipliers;

bool solve_dense(std::vector<double> A, Vec rhs, Vec& sol) {
  const int n = static_cast<int>(rhs.size());
  if (n == 0) { sol.clear(); return true; }
  std::vector<int> piv(n, 0);
  int nrhs = 1, info = 0;
  dgesv_(&n, &nrhs, A.data(), &n, piv.data(), rhs.data(), &n, &info);
  if (info != 0) return false;
  for (double v : rhs) if (!std::isfinite(v)) return false;
  sol = std::move(rhs);
  return true;
}


struct ConstrainedMultiplierLS {
  Vec free_duals;
  Vec nonnegative_duals;
  double stationarity = std::numeric_limits<double>::infinity();
  int sweeps = 0;
  double projected_gradient_inf = std::numeric_limits<double>::infinity();
  bool converged = false;
};


// Sparse row-space projector for free equality multipliers.
//
// Given row gradients F_i and a vector v, compute
//
//   lambda = argmin ||v + F^T lambda||_2^2 + reg ||S^{-1}lambda||_2^2,
//
// through the scaled row-space system
//
//   (S F F^T S + reg I) nu = -S F v,  lambda = S nu.
//
// The Gram matrix is assembled from column incidence lists, not from all
// O(m^2 n) dense row dot products.  This is decisive on CATENA/CATENARY:
// about 1000 equality rows have short supports in 3003 variables.  The factor
// object persists, so LDL+METIS reuses its symbolic envelope as the Jacobian
// values evolve between outer iterations.
struct FreeMultiplierProjection {
  Vec multipliers;
  Vec residual;
  double stationarity = std::numeric_limits<double>::infinity();
  bool success = false;
  std::string message;
};

class FreeMultiplierProjector {
 public:
  explicit FreeMultiplierProjector(const Settings& st) : factor_settings_(st) {
    // The row-space regularization is assembled explicitly below.  Prevent the
    // backend from silently adding another primal/dual KKT shift.
    factor_settings_.kkt_adaptive_regularization=true;
    std::string msg;
    factor_=make_symmetric_factorization(factor_settings_,msg);
    if(!factor_) last_message_=msg;
  }

  bool factor_rows(const std::vector<SparseRow>& rows,int n,
                   double regularization) {
    const int m=static_cast<int>(rows.size());
    if(m==0) {
      row_sparse_.clear(); row_scale_.clear(); dimension_=0;
      variable_dimension_=n;
      factored_=true; return true;
    }
    if(n<0) {
      last_message_="free-multiplier negative row dimension";
      factored_=false; ++failures_; return false;
    }
    if(!factor_) {
      last_message_="free-multiplier factor backend unavailable";
      factored_=false; ++failures_; return false;
    }

    row_sparse_.assign(m,{});
    row_scale_.assign(m,1.0);
    std::vector<std::vector<std::pair<int,double>>> columns(n);
    std::size_t structural_nnz=0;
    for(int i=0;i<m;++i) {
      double n2=0.0;
      int previous=-1;
      for(const auto& [j,a]:rows[static_cast<std::size_t>(i)].entries) {
        if(j<0 || j>=n || j<=previous) {
          last_message_="free-multiplier sparse row is not strictly ordered";
          factored_=false; ++failures_; return false;
        }
        previous=j;
        n2+=a*a;
      }
      const double norm=std::sqrt(n2);
      if(!std::isfinite(norm)) {
        last_message_="free-multiplier fit contains a nonfinite row";
        factored_=false; ++failures_; return false;
      }
      // A locally zero equality normal carries no stationarity information.
      // Keep a regularized diagonal slot and return its multiplier as zero
      // instead of aborting the whole original-Jacobian projection.
      const double scale=(norm>0.0)?1.0/norm:1.0;
      row_scale_[i]=scale;
      auto& sparse=row_sparse_[i];
      sparse.reserve(rows[static_cast<std::size_t>(i)].entries.size());
      for(const auto& [j,a]:rows[static_cast<std::size_t>(i)].entries) {
        const double as=scale*a;
        sparse.push_back({j,as});
        columns[j].push_back({i,as});
        ++structural_nnz;
      }
    }

    std::vector<SymmetricTriplet> gram;
    std::size_t pair_count=static_cast<std::size_t>(m);
    for(const auto& col:columns)
      pair_count+=col.size()*(col.size()+1)/2;
    gram.reserve(pair_count);
    for(int i=0;i<m;++i) {
      // Row equilibration uses A = D F and lambda = D nu.  Preserve the
      // original Tikhonov term reg*||lambda||^2 by adding reg*D^2 in the
      // equilibrated normal equations.
      const double diagonal_regularization=std::max(
          1e-18,regularization*row_scale_[i]*row_scale_[i]);
      gram.push_back({i,i,diagonal_regularization});
    }
    for(const auto& col:columns) {
      for(std::size_t a=0;a<col.size();++a) {
        const int i=col[a].first;
        const double vi=col[a].second;
        for(std::size_t b=a;b<col.size();++b) {
          const int k=col[b].first;
          const double vk=col[b].second;
          gram.push_back({std::min(i,k),std::max(i,k),vi*vk});
        }
      }
    }

    std::string msg;
    ++factorizations_;
    const int symbolic_before=factor_->symbolic_analyses();
    const int reuse_before=factor_->symbolic_reuses();
    if(!factor_->factor(m,m,gram,msg)) {
      last_message_="free-multiplier row-space factorization failed: "+msg;
      factored_=false; ++failures_; return false;
    }
    symbolic_analyses_+=factor_->symbolic_analyses()-symbolic_before;
    symbolic_reuses_+=factor_->symbolic_reuses()-reuse_before;
    dimension_=m;
    variable_dimension_=n;
    regularization_=regularization;
    structural_nnz_=structural_nnz;
    factored_=true;
    last_message_.clear();
    return true;
  }

  // Compatibility path for the certificate-only fit.  The hot path in
  // estimate_multipliers_legacy supplies structural rows directly and never
  // materializes its 1016-by-1088 dense gradient array.
  bool factor_rows(const std::vector<Vec>& rows,double regularization) {
    if(rows.empty())
      return factor_rows(std::vector<SparseRow>{},0,regularization);
    const int n=static_cast<int>(rows.front().size());
    std::vector<SparseRow> sparse_rows;
    sparse_rows.reserve(rows.size());
    for(const auto& row:rows) {
      if(static_cast<int>(row.size())!=n) {
        last_message_="free-multiplier row dimension mismatch";
        factored_=false; ++failures_; return false;
      }
      // dense_gradient_row deliberately omits numerical zeros.  Check
      // finiteness first so NaN is not silently dropped by its threshold
      // comparison; the historical dense norm path failed closed here.
      for(double value:row) if(!std::isfinite(value)) {
        last_message_="free-multiplier fit contains a nonfinite row";
        factored_=false; ++failures_; return false;
      }
      sparse_rows.push_back(dense_gradient_row(row));
    }
    return factor_rows(sparse_rows,n,regularization);
  }

  FreeMultiplierProjection project(const Vec& v) {
    FreeMultiplierProjection out;
    if(!factored_) { out.message=last_message_; return out; }
    if(dimension_==0) {
      out.residual=v; out.stationarity=inf_norm(v); out.success=true; return out;
    }
    if(static_cast<int>(v.size())!=variable_dimension_) {
      out.message="free-multiplier projection vector dimension mismatch";
      ++failures_; return out;
    }
    Vec rhs(dimension_,0.0);
    for(int i=0;i<dimension_;++i) {
      double fv=0.0;
      for(const auto& [j,a]:row_sparse_[i]) fv+=a*v[j];
      rhs[i]=-fv;
    }
    std::string msg;
    ++solves_;
    if(!factor_->solve(rhs,msg)) {
      out.message="free-multiplier row-space solve failed: "+msg;
      last_message_=out.message; ++failures_; return out;
    }
    out.multipliers.resize(dimension_);
    for(int i=0;i<dimension_;++i) out.multipliers[i]=row_scale_[i]*rhs[i];
    out.residual=v;
    for(int i=0;i<dimension_;++i) {
      const double lambda=out.multipliers[i];
      if(lambda==0.0) continue;
      const double unscale=1.0/row_scale_[i];
      for(const auto& [j,a]:row_sparse_[i])
        out.residual[j]+=lambda*(a*unscale);
    }
    out.stationarity=inf_norm(out.residual);
    out.success=std::isfinite(out.stationarity);
    if(!out.success) { out.message="free-multiplier projection produced nonfinite residual"; ++failures_; }
    return out;
  }

  FreeMultiplierProjection project(const SparseRow& v,int n) {
    FreeMultiplierProjection out;
    if(!factored_) { out.message=last_message_; return out; }
    if(n!=variable_dimension_) {
      out.message="free-multiplier sparse projection vector dimension mismatch";
      ++failures_; return out;
    }
    int previous=-1;
    for(const auto& [j,a]:v.entries) {
      if(j<0 || j>=n || j<=previous || !std::isfinite(a)) {
        out.message=
            "free-multiplier sparse projection vector is invalid, nonfinite, "
            "or not strictly ordered";
        ++failures_; return out;
      }
      previous=j;
    }
    if(dimension_==0) {
      out.residual.assign(static_cast<std::size_t>(n),0.0);
      for(const auto& [j,a]:v.entries)
        out.residual[static_cast<std::size_t>(j)]=a;
      out.stationarity=inf_norm(out.residual);
      out.success=true;
      return out;
    }

    Vec rhs(dimension_,0.0);
    for(int i=0;i<dimension_;++i) {
      double fv=0.0;
      std::size_t k=0;
      for(const auto& [j,a]:row_sparse_[static_cast<std::size_t>(i)]) {
        while(k<v.entries.size() && v.entries[k].first<j) ++k;
        const double value=(k<v.entries.size() && v.entries[k].first==j)
            ? v.entries[k].second : 0.0;
        fv+=a*value;
      }
      rhs[static_cast<std::size_t>(i)]=-fv;
    }
    std::string msg;
    ++solves_;
    if(!factor_->solve(rhs,msg)) {
      out.message="free-multiplier row-space solve failed: "+msg;
      last_message_=out.message; ++failures_; return out;
    }
    out.multipliers.resize(static_cast<std::size_t>(dimension_));
    for(int i=0;i<dimension_;++i)
      out.multipliers[static_cast<std::size_t>(i)]=
          row_scale_[static_cast<std::size_t>(i)]*rhs[static_cast<std::size_t>(i)];
    out.residual.assign(static_cast<std::size_t>(n),0.0);
    for(const auto& [j,a]:v.entries)
      out.residual[static_cast<std::size_t>(j)]=a;
    for(int i=0;i<dimension_;++i) {
      const double lambda=out.multipliers[static_cast<std::size_t>(i)];
      if(lambda==0.0) continue;
      const double unscale=1.0/row_scale_[static_cast<std::size_t>(i)];
      for(const auto& [j,a]:row_sparse_[static_cast<std::size_t>(i)])
        out.residual[static_cast<std::size_t>(j)]+=lambda*(a*unscale);
    }
    out.stationarity=inf_norm(out.residual);
    out.success=std::isfinite(out.stationarity);
    if(!out.success) {
      out.message="free-multiplier projection produced nonfinite residual";
      ++failures_;
    }
    return out;
  }

  int factorizations() const { return factorizations_; }
  int solves() const { return solves_; }
  int symbolic_analyses() const { return symbolic_analyses_; }
  int symbolic_reuses() const { return symbolic_reuses_; }
  int failures() const { return failures_; }
  int dimension() const { return dimension_; }
  std::size_t structural_nnz() const { return structural_nnz_; }
  const std::string& last_message() const { return last_message_; }

 private:
  Settings factor_settings_;
  std::unique_ptr<SymmetricFactorization> factor_;
  std::vector<std::vector<std::pair<int,double>>> row_sparse_;
  Vec row_scale_;
  int dimension_=0;
  int variable_dimension_=0;
  double regularization_=0.0;
  std::size_t structural_nnz_=0;
  bool factored_=false;
  int factorizations_=0;
  int solves_=0;
  int symbolic_analyses_=0;
  int symbolic_reuses_=0;
  int failures_=0;
  std::string last_message_;
};

// Solve
//   min_{lambda free, mu >= 0}
//       1/2 || g + F^T lambda + A^T mu ||_2^2
//       + reg/2 (||lambda||_2^2 + ||mu||_2^2).
//
// The equalities/free multipliers are eliminated through the small system
// F F^T + reg I.  The remaining bound-constrained convex quadratic in mu is
// solved by exact cyclic coordinate minimization.  This avoids the singular
// m-by-m normal equations that occur when the active set is larger than n
// (A4X12: roughly 287 candidate inequalities for n=130), and enforces the KKT
// sign convention mu >= 0 by construction.
ConstrainedMultiplierLS constrained_multiplier_ls(
    const Vec& g,
    const std::vector<Vec>& free_grad,
    const std::vector<Vec>& nonneg_grad,
    const Settings& st,
    const Vec* warm_nonnegative=nullptr,
    int max_sweeps_override=0,
    FreeMultiplierProjector* persistent_free_projector=nullptr) {
  ConstrainedMultiplierLS out;
  const int nf = static_cast<int>(free_grad.size());
  const int q = static_cast<int>(nonneg_grad.size());
  out.free_duals.assign(nf, 0.0);
  out.nonnegative_duals.assign(q, 0.0);

  const double reg = std::max(1e-14, st.multiplier_regularization);
  FreeMultiplierProjector local_projector(st);
  FreeMultiplierProjector* projector = persistent_free_projector
      ? persistent_free_projector : &local_projector;
  if(nf>0 && !projector->factor_rows(free_grad,reg)) {
    out.projected_gradient_inf=std::numeric_limits<double>::infinity();
    out.stationarity=std::numeric_limits<double>::infinity();
    out.converged=false;
    return out;
  }

  bool free_projection_ok=true;
  auto eliminate_free = [&](const Vec& v, Vec* lambda_out)->Vec {
    if(nf==0) {
      if(lambda_out) lambda_out->clear();
      return v;
    }
    const auto projected=projector->project(v);
    if(!projected.success) {
      free_projection_ok=false;
      if(lambda_out) lambda_out->assign(nf,0.0);
      return v;
    }
    if(lambda_out) *lambda_out=projected.multipliers;
    return projected.residual;
  };

  if(q==0) {
    Vec lam;
    const Vec r=eliminate_free(g,&lam);
    out.free_duals=std::move(lam);
    out.stationarity=inf_norm(r);
    out.projected_gradient_inf=0.0;
    // Numerical success means that the least-squares projector was actually
    // solved.  The previous code unconditionally returned converged=true even
    // when dgesv failed and zero multipliers were silently substituted.
    out.converged=free_projection_ok && std::isfinite(out.stationarity);
    return out;
  }
  const Vec pg=eliminate_free(g,nullptr);
  if(!free_projection_ok) return out;

  std::vector<Vec> PA(q);
  for(int i=0;i<q;++i) {
    PA[i]=eliminate_free(nonneg_grad[i],nullptr);
    if(!free_projection_ok) return out;
  }

  std::vector<double> H(static_cast<std::size_t>(q)*q,0.0);
  Vec grad(q,0.0);
  for(int i=0;i<q;++i) {
    grad[i]=dot(nonneg_grad[i],pg);
    for(int j=0;j<=i;++j) {
      double v=dot(nonneg_grad[i],PA[j]);
      if(i==j) v+=reg;
      H[j+i*q]=v; H[i+j*q]=v;
    }
  }

  Vec mu(q,0.0);
  if(warm_nonnegative && static_cast<int>(warm_nonnegative->size())==q)
    for(int i=0;i<q;++i) mu[i]=std::max(0.0,(*warm_nonnegative)[i]);
  if(warm_nonnegative)
    for(int i=0;i<q;++i) if(mu[i]!=0.0)
      for(int j=0;j<q;++j) grad[j]+=H[j+i*q]*mu[i];
  // This cyclic NNLS fit is a path-model construction, not the final KKT
  // certificate.  Its projected gradient has the scale of the objective
  // gradient, so an absolute O(1e-8) threshold spuriously reports failure on
  // otherwise identical large-scale objectives (case118: ||g||_inf=O(1e3)).
  // Use a relative stopping test for this internal fit.  The independent
  // certificate-only active-set solve below deliberately keeps its historical
  // absolute tolerance, as do all original-QCQP KKT termination tests.
  // Preserve the validated inequality-only trajectory exactly; the scaling
  // issue arises in the joint fit after eliminating a large equality row
  // space.
  const double path_gradient_scale=nf>0 ? std::max(1.0,inf_norm(g)) : 1.0;
  const double pg_tol=std::max(1e-11,0.01*st.stationarity_tolerance)*
      path_gradient_scale;
  const int max_sweeps=max_sweeps_override>0 ? max_sweeps_override
      : std::max(80,std::min(800,4*q+40));
  for(int sweep=0;sweep<max_sweeps;++sweep) {
    for(int i=0;i<q;++i) {
      const double hii=std::max(H[i+i*q],1e-18);
      const double old=mu[i];
      const double neu=std::max(0.0,old-grad[i]/hii);
      const double delta=neu-old;
      if(delta==0.0) continue;
      mu[i]=neu;
      for(int j=0;j<q;++j) grad[j]+=delta*H[j+i*q];
    }
    double pginf=0.0;
    for(int i=0;i<q;++i) {
      const double pg_i=(mu[i]>1e-14)?grad[i]:std::min(0.0,grad[i]);
      pginf=std::max(pginf,std::abs(pg_i));
    }
    out.sweeps=sweep+1;
    out.projected_gradient_inf=pginf;
    if(pginf<=pg_tol) { out.converged=true; break; }
  }

  // A failed path fit is discarded by every caller.  Do not pay for one more
  // row-space solve merely to populate diagnostics/duals that cannot be used.
  if(!out.converged) {
    out.nonnegative_duals=std::move(mu);
    out.stationarity=std::numeric_limits<double>::infinity();
    return out;
  }

  Vec v=g;
  for(int i=0;i<q;++i) if(mu[i]!=0.0) axpy(mu[i],nonneg_grad[i],v);
  Vec lam;
  Vec r=eliminate_free(v,&lam);
  if(!free_projection_ok) {
    out.converged=false;
    out.stationarity=std::numeric_limits<double>::infinity();
    return out;
  }
  out.free_duals=std::move(lam);
  out.nonnegative_duals=std::move(mu);
  out.stationarity=inf_norm(r);
  return out;
}

// Structurally sparse counterpart used by the original-Jacobian path fit.
// Only the storage and zero-skipping change: row order, coordinate order,
// equilibration, Gram triplet order, NNLS sweeps and stopping tests are the
// same as in the dense compatibility routine above.
ConstrainedMultiplierLS constrained_multiplier_ls(
    const Vec& g,
    const std::vector<SparseRow>& free_grad,
    const std::vector<SparseRow>& nonneg_grad,
    int variable_dimension,
    const Settings& st,
    const Vec* warm_nonnegative=nullptr,
    int max_sweeps_override=0,
    FreeMultiplierProjector* persistent_free_projector=nullptr) {
  ConstrainedMultiplierLS out;
  const int nf=static_cast<int>(free_grad.size());
  const int q=static_cast<int>(nonneg_grad.size());
  out.free_duals.assign(static_cast<std::size_t>(nf),0.0);
  out.nonnegative_duals.assign(static_cast<std::size_t>(q),0.0);
  if(static_cast<int>(g.size())!=variable_dimension) return out;

  const double reg=std::max(1e-14,st.multiplier_regularization);
  FreeMultiplierProjector local_projector(st);
  FreeMultiplierProjector* projector=persistent_free_projector
      ? persistent_free_projector : &local_projector;
  if(nf>0 && !projector->factor_rows(free_grad,variable_dimension,reg)) {
    out.projected_gradient_inf=std::numeric_limits<double>::infinity();
    out.stationarity=std::numeric_limits<double>::infinity();
    out.converged=false;
    return out;
  }

  bool free_projection_ok=true;
  auto eliminate_free_dense=[&](const Vec& v,Vec* lambda_out)->Vec {
    if(nf==0) {
      if(lambda_out) lambda_out->clear();
      return v;
    }
    const auto projected=projector->project(v);
    if(!projected.success) {
      free_projection_ok=false;
      if(lambda_out) lambda_out->assign(static_cast<std::size_t>(nf),0.0);
      return v;
    }
    if(lambda_out) *lambda_out=projected.multipliers;
    return projected.residual;
  };
  auto eliminate_free_sparse=[&](const SparseRow& v)->Vec {
    if(nf==0) {
      Vec dense(static_cast<std::size_t>(variable_dimension),0.0);
      for(const auto& [j,a]:v.entries)
        dense[static_cast<std::size_t>(j)]=a;
      return dense;
    }
    const auto projected=projector->project(v,variable_dimension);
    if(!projected.success) {
      free_projection_ok=false;
      Vec dense(static_cast<std::size_t>(variable_dimension),0.0);
      for(const auto& [j,a]:v.entries)
        dense[static_cast<std::size_t>(j)]=a;
      return dense;
    }
    return projected.residual;
  };

  if(q==0) {
    Vec lam;
    const Vec r=eliminate_free_dense(g,&lam);
    out.free_duals=std::move(lam);
    out.stationarity=inf_norm(r);
    out.projected_gradient_inf=0.0;
    out.converged=free_projection_ok && std::isfinite(out.stationarity);
    return out;
  }
  const Vec pg=eliminate_free_dense(g,nullptr);
  if(!free_projection_ok) return out;

  std::vector<Vec> PA(static_cast<std::size_t>(q));
  for(int i=0;i<q;++i) {
    PA[static_cast<std::size_t>(i)]=
        eliminate_free_sparse(nonneg_grad[static_cast<std::size_t>(i)]);
    if(!free_projection_ok) return out;
  }

  std::vector<double> H(static_cast<std::size_t>(q)*q,0.0);
  Vec grad(static_cast<std::size_t>(q),0.0);
  for(int i=0;i<q;++i) {
    grad[static_cast<std::size_t>(i)]=
        multiplier_sparse_dot(nonneg_grad[static_cast<std::size_t>(i)],pg);
    for(int j=0;j<=i;++j) {
      double v=multiplier_sparse_dot(
          nonneg_grad[static_cast<std::size_t>(i)],
          PA[static_cast<std::size_t>(j)]);
      if(i==j) v+=reg;
      H[static_cast<std::size_t>(j+i*q)]=v;
      H[static_cast<std::size_t>(i+j*q)]=v;
    }
  }

  Vec mu(static_cast<std::size_t>(q),0.0);
  if(warm_nonnegative && static_cast<int>(warm_nonnegative->size())==q)
    for(int i=0;i<q;++i)
      mu[static_cast<std::size_t>(i)]=
          std::max(0.0,(*warm_nonnegative)[static_cast<std::size_t>(i)]);
  if(warm_nonnegative)
    for(int i=0;i<q;++i) if(mu[static_cast<std::size_t>(i)]!=0.0)
      for(int j=0;j<q;++j)
        grad[static_cast<std::size_t>(j)]+=
            H[static_cast<std::size_t>(j+i*q)]*
            mu[static_cast<std::size_t>(i)];
  const double path_gradient_scale=nf>0 ? std::max(1.0,inf_norm(g)) : 1.0;
  const double pg_tol=std::max(1e-11,0.01*st.stationarity_tolerance)*
      path_gradient_scale;
  const int max_sweeps=max_sweeps_override>0 ? max_sweeps_override
      : std::max(80,std::min(800,4*q+40));
  for(int sweep=0;sweep<max_sweeps;++sweep) {
    for(int i=0;i<q;++i) {
      const double hii=std::max(H[static_cast<std::size_t>(i+i*q)],1e-18);
      const double old=mu[static_cast<std::size_t>(i)];
      const double neu=std::max(
          0.0,old-grad[static_cast<std::size_t>(i)]/hii);
      const double delta=neu-old;
      if(delta==0.0) continue;
      mu[static_cast<std::size_t>(i)]=neu;
      for(int j=0;j<q;++j)
        grad[static_cast<std::size_t>(j)]+=
            delta*H[static_cast<std::size_t>(j+i*q)];
    }
    double pginf=0.0;
    for(int i=0;i<q;++i) {
      const double pg_i=(mu[static_cast<std::size_t>(i)]>1e-14)
          ? grad[static_cast<std::size_t>(i)]
          : std::min(0.0,grad[static_cast<std::size_t>(i)]);
      pginf=std::max(pginf,std::abs(pg_i));
    }
    out.sweeps=sweep+1;
    out.projected_gradient_inf=pginf;
    if(pginf<=pg_tol) { out.converged=true; break; }
  }

  if(!out.converged) {
    out.nonnegative_duals=std::move(mu);
    out.stationarity=std::numeric_limits<double>::infinity();
    return out;
  }

  Vec v=g;
  for(int i=0;i<q;++i)
    if(mu[static_cast<std::size_t>(i)]!=0.0)
      multiplier_sparse_axpy(mu[static_cast<std::size_t>(i)],
                             nonneg_grad[static_cast<std::size_t>(i)],v);
  Vec lam;
  Vec r=eliminate_free_dense(v,&lam);
  if(!free_projection_ok) {
    out.converged=false;
    out.stationarity=std::numeric_limits<double>::infinity();
    return out;
  }
  out.free_duals=std::move(lam);
  out.nonnegative_duals=std::move(mu);
  out.stationarity=inf_norm(r);
  return out;
}

MultiplierEstimate estimate_multipliers_legacy(
    const QCQPProblem& p,
    const Vec& x,
    const std::set<int>& working_set,
    const Settings& st,
    FreeMultiplierProjector* persistent_free_projector=nullptr,
    ConstrainedMultiplierLS* fit_out=nullptr,
    bool include_active_bounds=false) {
  const int me=static_cast<int>(p.equalities.size());
  MultiplierEstimate out;
  out.equality.assign(me,0.0);
  out.inequality.assign(p.inequalities.size(),0.0);
  out.lower_bound.assign(p.n,0.0);
  out.upper_bound.assign(p.n,0.0);

  // Fixed variables are parameters in the original-QCQP certificate.  Their
  // stationarity components are balanced by an unconstrained bound-difference
  // multiplier and must not distort the equality multiplier fit or the
  // Lagrangian Hessian used by the local corrector.
  std::vector<unsigned char> fixed(static_cast<std::size_t>(p.n),0);
  for(int j=0;j<p.n;++j)
    fixed[j]=std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
        std::abs(p.ub[j]-p.lb[j])<=st.fixed_bound_tolerance;

  Vec objg=p.objective.gradient(x);
  for(int j=0;j<p.n;++j) if(fixed[j]) objg[j]=0.0;

  SparseGradientWorkspace gradient_workspace(p.n);
  std::vector<SparseRow> free_grad;
  free_grad.reserve(me);
  for(const auto& h:p.equalities)
    free_grad.push_back(
        multiplier_sparse_gradient(h,x,fixed,gradient_workspace));

  enum class NonnegativeNormalKind { Inequality, LowerBound, UpperBound };
  struct NonnegativeNormalLocation {
    NonnegativeNormalKind kind;
    int index;
  };
  std::vector<NonnegativeNormalLocation> nonneg_map;
  std::vector<SparseRow> nonneg_grad;
  nonneg_map.reserve(working_set.size()+2*static_cast<std::size_t>(p.n));
  nonneg_grad.reserve(nonneg_map.capacity());
  for(int i:working_set) {
    if(structural_quadratic_value(p.inequalities[i],x)<
       -10.0*st.working_set_activation) continue;
    nonneg_map.push_back({NonnegativeNormalKind::Inequality,i});
    nonneg_grad.push_back(multiplier_sparse_gradient(
        p.inequalities[static_cast<std::size_t>(i)],x,fixed,
        gradient_workspace));
  }

  if(include_active_bounds) {
    const double btol=std::max(1e-8,10.0*st.feasibility_tolerance);
    for(int j=0;j<p.n;++j) {
      if(fixed[j]) continue;
      if(std::isfinite(p.lb[j]) && x[j]<=p.lb[j]+btol) {
        SparseRow g; g.entries.push_back({j,-1.0});
        nonneg_map.push_back({NonnegativeNormalKind::LowerBound,j});
        nonneg_grad.push_back(std::move(g));
      }
      if(std::isfinite(p.ub[j]) && x[j]>=p.ub[j]-btol) {
        SparseRow g; g.entries.push_back({j,+1.0});
        nonneg_map.push_back({NonnegativeNormalKind::UpperBound,j});
        nonneg_grad.push_back(std::move(g));
      }
    }
  }

  const auto ls=constrained_multiplier_ls(
      objg,free_grad,nonneg_grad,p.n,st,nullptr,0,
      persistent_free_projector);
  if(fit_out) *fit_out=ls;
  if(!ls.converged) {
    out.valid=false;
    if(st.verbose) {
      std::cout << "[MULTIPLIER-LS-SKIP] numerical multiplier projection failed"
                << " sweeps=" << ls.sweeps
                << " pg=" << std::scientific << ls.projected_gradient_inf
                << " stationarity=" << ls.stationarity
                << " candidates=" << nonneg_grad.size()
                << "\n" << std::defaultfloat;
    }
    return out;
  }

  for(int j=0;j<me && j<static_cast<int>(ls.free_duals.size());++j)
    out.equality[j]=ls.free_duals[j];
  for(int k=0;k<static_cast<int>(nonneg_map.size()) &&
                  k<static_cast<int>(ls.nonnegative_duals.size());++k) {
    const double mu=std::max(0.0,ls.nonnegative_duals[k]);
    const auto& c=nonneg_map[k];
    if(c.kind==NonnegativeNormalKind::Inequality) out.inequality[c.index]=mu;
    else if(c.kind==NonnegativeNormalKind::LowerBound) out.lower_bound[c.index]=mu;
    else if(c.kind==NonnegativeNormalKind::UpperBound) out.upper_bound[c.index]=mu;
  }
  out.valid=true;
  return out;
}


SymMatrix lagrangian_hessian(const QCQPProblem& p, const MultiplierEstimate& mult) {
  SymMatrix B;
  B.n = p.n;
  add_scaled_hessian(B, p.objective.hessian, 1.0);
  for (int j = 0; j < static_cast<int>(p.equalities.size()); ++j)
    add_scaled_hessian(B, p.equalities[j].hessian, mult.equality[j]);
  for (int i = 0; i < static_cast<int>(p.inequalities.size()); ++i)
    add_scaled_hessian(B, p.inequalities[i].hessian, mult.inequality[i]);
  return B;
}

// Equality-manifold Hessian used by the condensed-retraction route.
// Inequality curvature remains in the independently convexified constraint
// models, while the signed equality Hessians enter the objective through the
// current free multipliers exactly as in an SQP/Riemannian second-order model.
SymMatrix equality_lagrangian_hessian(const QCQPProblem& p,
                                      const MultiplierEstimate& mult) {
  SymMatrix B;
  B.n = p.n;
  add_scaled_hessian(B, p.objective.hessian, 1.0);
  for (int j = 0; j < static_cast<int>(p.equalities.size()); ++j) {
    const double lambda = j < static_cast<int>(mult.equality.size())
        ? mult.equality[static_cast<std::size_t>(j)] : 0.0;
    add_scaled_hessian(B, p.equalities[static_cast<std::size_t>(j)].hessian,
                       lambda);
  }
  return B;
}

double projected_stationarity(const QCQPProblem& p, const Vec& x,
                              const MultiplierEstimate& mult, double bound_tol) {
  Vec r = p.objective.gradient(x);
  SparseGradientWorkspace gradient_workspace(p.n);
  for (int j = 0; j < static_cast<int>(p.equalities.size()); ++j) {
    const double multiplier = mult.equality[j];
    consume_sparse_quadratic_gradient(
        p.equalities[j], x, gradient_workspace,
        [&](int coordinate, double value) {
          r[static_cast<std::size_t>(coordinate)] += multiplier * value;
        });
  }
  for (int i = 0; i < static_cast<int>(p.inequalities.size()); ++i)
    if (mult.inequality[i] != 0.0)
      consume_sparse_quadratic_gradient(
          p.inequalities[i], x, gradient_workspace,
          [&](int coordinate, double value) {
            r[static_cast<std::size_t>(coordinate)] +=
                mult.inequality[i] * value;
          });

  double s = 0.0;
  for (int j = 0; j < p.n; ++j) {
    const bool at_l = std::isfinite(p.lb[j]) && x[j] <= p.lb[j] + bound_tol;
    const bool at_u = std::isfinite(p.ub[j]) && x[j] >= p.ub[j] - bound_tol;
    double v = r[j];
    if (at_l && at_u) v = 0.0;
    else if (at_l) v = std::min(0.0, v);
    else if (at_u) v = std::max(0.0, v);
    s = std::max(s, std::abs(v));
  }
  return s;
}

ConvexQuadraticModel inequality_model(const QuadraticFunction& q, const Vec& x,
                                      const Convexification& cvx) {
  ConvexQuadraticModel m;
  m.constant = structural_quadratic_value(q,x);
  m.linear = q.gradient(x);
  m.base_hessian = q.hessian;
  m.shift_dimension = q.hessian.n;
  m.scalar_shift = cvx.scalar_shift;
  m.low_rank = cvx.terms;
  return m;
}

ConvexQuadraticModel linearized_inequality_model(const QuadraticFunction& q,
                                                  const Vec& x) {
  ConvexQuadraticModel m;
  m.constant = structural_quadratic_value(q,x);
  m.linear = q.gradient(x);
  m.base_hessian.n = q.hessian.n;
  m.shift_dimension = q.hessian.n;
  return m;
}

ConvexQuadraticModel extend_model_dimension(ConvexQuadraticModel m,int nvar) {
  const int old=static_cast<int>(m.linear.size());
  if(nvar<old) throw std::invalid_argument("extend_model_dimension: shrinking model");
  m.linear.resize(nvar,0.0);
  m.base_hessian.n=nvar;
  for(auto& t:m.low_rank) t.direction.resize(nvar,0.0);
  return m;
}

ConvexQuadraticModel elastic_equality_majorant(const QuadraticFunction& h,
                                                const Vec& x,double sign,
                                                double eta0,int elastic_index) {
  const int n=static_cast<int>(x.size());
  ConvexQuadraticModel m;
  m.constant=sign*structural_quadratic_value(h,x)-eta0;
  m.linear=h.gradient(x);
  for(double& v:m.linear) v*=sign;
  m.linear.resize(n+1,0.0);
  m.linear[elastic_index]=-1.0;
  m.base_hessian.n=n+1;
  SymMatrix signed_h; signed_h.n=n;
  signed_h.entries.reserve(h.hessian.entries.size());
  for(const auto& e:h.hessian.entries) {
    signed_h.entries.push_back({e.i,e.j,sign*e.v});
    m.base_hessian.entries.push_back({e.i,e.j,sign*e.v});
  }
  const double lb=signed_h.gershgorin_lower_bound();
  const double shift=std::max(0.0,-lb+64.0*std::numeric_limits<double>::epsilon()*
                                      std::max(1.0,std::abs(lb)));
  if(shift>0.0)
    for(int j=0;j<n;++j) m.base_hessian.entries.push_back({j,j,shift});
  m.shift_dimension=0;
  return m;
}

struct BuiltSubproblem {
  ConvexSubproblem p;
  std::vector<int> active_original_ineq;
  // Stable semantic identity for every inequality row.  Numeric row
  // positions change when the lazy working set adds or drops an original
  // inequality; direct primal-dual continuation must never confuse that churn
  // with a different original row or a bound.
  std::vector<std::uint64_t> inequality_continuation_keys;
  int modeled_side_recovery_margin_rows=0;
  bool elastic_equalities=false;
  int elastic_index=-1;
  double elastic_base=0.0;
  double elastic_cap=0.0;
  std::vector<std::pair<int,int>> elastic_dual_positions;
  int original_ineq_offset=0;
};

enum class DcaInequalityRowKind : std::uint32_t {
  ElasticEqualityPlus=1,
  ElasticEqualityMinus=2,
  ElasticEtaLower=3,
  ElasticEtaUpper=4,
  OriginalInequality=5,
  UpperBound=6,
  LowerBound=7,
  TrustRegion=8
};

std::uint64_t inequality_row_key(DcaInequalityRowKind kind,int index) {
  return (static_cast<std::uint64_t>(kind)<<32) |
      static_cast<std::uint32_t>(index);
}

BuiltSubproblem build_subproblem(const QCQPProblem& prob, const Vec& x,
                                 const SymMatrix& B,
                                 const Convexification& lag_cvx,
                                 const std::vector<Convexification>& ineq_cvx,
                                 const std::set<int>& working_set,
                                 const Settings& st,
                                 double proximal,
                                 bool linearize_inequalities,
                                 double trust_radius = std::numeric_limits<double>::infinity()) {
  BuiltSubproblem out;
  out.p.nvar = prob.n;
  out.p.original_dimension = prob.n;
  out.p.objective.constant = structural_quadratic_value(prob.objective,x);
  out.p.objective.linear = prob.objective.gradient(x);
  out.p.objective.base_hessian = B;
  out.p.objective.shift_dimension = prob.n;
  out.p.objective.scalar_shift = lag_cvx.scalar_shift + proximal;
  out.p.objective.low_rank = lag_cvx.terms;
  out.p.equalities.nvar = prob.n;

  for (const auto& h : prob.equalities) {
    out.p.equalities.rows.push_back(dense_gradient_row(h.gradient(x)));
    out.p.equalities.rhs.push_back(-structural_quadratic_value(h,x));
  }

  for (int j = 0; j < prob.n; ++j) {
    if (std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
        std::abs(prob.ub[j]-prob.lb[j]) <= st.fixed_bound_tolerance) {
      SparseRow row;
      row.entries.push_back({j,1.0});
      out.p.equalities.rows.push_back(std::move(row));
      out.p.equalities.rhs.push_back(prob.lb[j]-x[j]);
    }
  }

  for (int idx : working_set) {
    out.active_original_ineq.push_back(idx);
    out.p.inequalities.push_back(linearize_inequalities
        ? linearized_inequality_model(prob.inequalities[idx], x)
        : inequality_model(prob.inequalities[idx], x, ineq_cvx[idx]));
    out.inequality_continuation_keys.push_back(
        inequality_row_key(DcaInequalityRowKind::OriginalInequality,idx));
  }

  for (int j = 0; j < prob.n; ++j) {
    const bool fixed = std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
                       std::abs(prob.ub[j]-prob.lb[j]) <= st.fixed_bound_tolerance;
    if (fixed) continue;
    if (std::isfinite(prob.ub[j])) {
      out.p.inequalities.push_back(linear_inequality(prob.n, j, +1.0, x[j]-prob.ub[j]));
      out.inequality_continuation_keys.push_back(
          inequality_row_key(DcaInequalityRowKind::UpperBound,j));
    }
    if (std::isfinite(prob.lb[j])) {
      out.p.inequalities.push_back(linear_inequality(prob.n, j, -1.0, prob.lb[j]-x[j]));
      out.inequality_continuation_keys.push_back(
          inequality_row_key(DcaInequalityRowKind::LowerBound,j));
    }
  }

  // r5 Euclidean trust region on the displacement variable itself:
  //                  1/2 ||d||_2^2 - 1/2 Delta^2 <= 0.
  // It is a single convex quadratic inequality, so the trust region does not
  // add 2*n barrier rows to the inner IPM.  Shrinking Delta changes the
  // optimizer/direction instead of merely scaling one stale direction.
  if (std::isfinite(trust_radius)) {
    const double delta=std::max(0.0,trust_radius);
    ConvexQuadraticModel tr;
    tr.constant=-0.5*delta*delta;
    tr.linear.assign(prob.n,0.0);
    tr.base_hessian.n=prob.n;
    for(int j=0;j<prob.n;++j) tr.base_hessian.entries.push_back({j,j,1.0});
    out.p.inequalities.push_back(std::move(tr));
    out.inequality_continuation_keys.push_back(
        inequality_row_key(DcaInequalityRowKind::TrustRegion,0));
  }
  return out;
}

// v0.4.0 core model.  For every quadratic F_i, including i=0, the current
// convex DCA majorant in displacement coordinates d is
//
//   F_{i,k}(x_k+d) = F_i(x_k+d) + 1/2 d^T M_{i,k} d,
//
// where M_{i,k} is the persistent lazy PSD correction.  Consequently the
// Hessian is Q_i+M_{i,k}, F_{i,k}(x_k)=F_i(x_k), and the gradients agree at
// x_k.  Objective and inequalities therefore use exactly the same DC rule.
std::vector<ExactAffineElimination> detect_pglib_apparent_flow_lifts(
    const QCQPProblem& prob) {
  const int n = prob.n;
  if (n <= 0 || prob.equalities.empty() || prob.inequalities.empty()) return {};

  std::vector<int> equality_linear_incidence(static_cast<std::size_t>(n), 0);
  std::vector<int> equality_hessian_incidence(static_cast<std::size_t>(n), 0);
  std::vector<int> inequality_incidence(static_cast<std::size_t>(n), 0);
  std::vector<int> last_inequality(static_cast<std::size_t>(n), -1);
  std::vector<unsigned char> objective_incidence(static_cast<std::size_t>(n), 0);

  for (int j = 0; j < n; ++j)
    if (prob.objective.linear[static_cast<std::size_t>(j)] != 0.0)
      objective_incidence[static_cast<std::size_t>(j)] = 1;
  for (const auto& entry : prob.objective.hessian.entries) {
    objective_incidence[static_cast<std::size_t>(entry.i)] = 1;
    objective_incidence[static_cast<std::size_t>(entry.j)] = 1;
  }
  for (const auto& equality : prob.equalities) {
    for (int j = 0; j < n; ++j)
      if (equality.linear[static_cast<std::size_t>(j)] != 0.0)
        ++equality_linear_incidence[static_cast<std::size_t>(j)];
    for (const auto& entry : equality.hessian.entries) {
      ++equality_hessian_incidence[static_cast<std::size_t>(entry.i)];
      if (entry.j != entry.i)
        ++equality_hessian_incidence[static_cast<std::size_t>(entry.j)];
    }
  }
  for (int i = 0; i < static_cast<int>(prob.inequalities.size()); ++i) {
    const auto& inequality = prob.inequalities[static_cast<std::size_t>(i)];
    for (int j = 0; j < n; ++j) {
      if (inequality.linear[static_cast<std::size_t>(j)] != 0.0 &&
          last_inequality[static_cast<std::size_t>(j)] != i) {
        ++inequality_incidence[static_cast<std::size_t>(j)];
        last_inequality[static_cast<std::size_t>(j)] = i;
      }
    }
    for (const auto& entry : inequality.hessian.entries) {
      for (int j : {entry.i, entry.j}) {
        if (last_inequality[static_cast<std::size_t>(j)] == i) continue;
        ++inequality_incidence[static_cast<std::size_t>(j)];
        last_inequality[static_cast<std::size_t>(j)] = i;
      }
    }
  }

  std::unordered_map<std::string, int> slim_by_suffix;
  for (int i = 0; i < static_cast<int>(prob.inequalities.size()); ++i) {
    const std::string& name = prob.inequalities[static_cast<std::size_t>(i)].name;
    if (name.rfind("slim_", 0) != 0) continue;
    const std::string suffix = name.substr(5);
    if (suffix.empty() || !slim_by_suffix.emplace(suffix, i).second) return {};
  }

  struct Pair {
    int flow_p_row = -1;
    int flow_q_row = -1;
    int flow_p_variable = -1;
    int flow_q_variable = -1;
    int slim = -1;
  };
  std::unordered_map<std::string, Pair> pairs;
  std::vector<int> row_variable(prob.equalities.size(), -1);
  std::vector<unsigned char> used_variable(static_cast<std::size_t>(n), 0);
  int named_flow_rows = 0;

  for (int i = 0; i < static_cast<int>(prob.equalities.size()); ++i) {
    const auto& equality = prob.equalities[static_cast<std::size_t>(i)];
    const bool is_p = equality.name.rfind("flowP_", 0) == 0;
    const bool is_q = equality.name.rfind("flowQ_", 0) == 0;
    if (!is_p && !is_q) continue;
    ++named_flow_rows;
    const std::string suffix = equality.name.substr(6);
    const auto slim_it = slim_by_suffix.find(suffix);
    if (suffix.empty() || slim_it == slim_by_suffix.end()) return {};

    int candidate = -1;
    for (int j = 0; j < n; ++j) {
      const double pivot = equality.linear[static_cast<std::size_t>(j)];
      if (pivot != 1.0 && pivot != -1.0) continue;
      if (equality_linear_incidence[static_cast<std::size_t>(j)] != 1 ||
          equality_hessian_incidence[static_cast<std::size_t>(j)] != 0 ||
          objective_incidence[static_cast<std::size_t>(j)] != 0 ||
          inequality_incidence[static_cast<std::size_t>(j)] != 1 ||
          last_inequality[static_cast<std::size_t>(j)] != slim_it->second ||
          std::isfinite(prob.lb[static_cast<std::size_t>(j)]) ||
          std::isfinite(prob.ub[static_cast<std::size_t>(j)]))
        continue;
      if (candidate >= 0) return {};
      candidate = j;
    }
    if (candidate < 0 || used_variable[static_cast<std::size_t>(candidate)])
      return {};
    used_variable[static_cast<std::size_t>(candidate)] = 1;
    row_variable[static_cast<std::size_t>(i)] = candidate;

    Pair& pair = pairs[suffix];
    pair.slim = slim_it->second;
    int& row = is_p ? pair.flow_p_row : pair.flow_q_row;
    int& variable = is_p ? pair.flow_p_variable : pair.flow_q_variable;
    if (row >= 0) return {};
    row = i;
    variable = candidate;
  }

  if (named_flow_rows == 0 || pairs.empty() ||
      pairs.size() != slim_by_suffix.size())
    return {};
  for (const auto& item : pairs) {
    const Pair& pair = item.second;
    if (pair.flow_p_row < 0 || pair.flow_q_row < 0 ||
        pair.flow_p_variable < 0 || pair.flow_q_variable < 0 ||
        pair.flow_p_variable == pair.flow_q_variable || pair.slim < 0)
      return {};
    const auto& slim = prob.inequalities[static_cast<std::size_t>(pair.slim)];
    for (double linear : slim.linear)
      if (linear != 0.0) return {};
    if (slim.hessian.entries.size() != 2) return {};
    double p_diagonal = 0.0;
    double q_diagonal = 0.0;
    bool have_p = false;
    bool have_q = false;
    for (const auto& entry : slim.hessian.entries) {
      if (entry.i != entry.j || !(entry.v > 0.0)) return {};
      if (entry.i == pair.flow_p_variable && !have_p) {
        p_diagonal = entry.v;
        have_p = true;
      } else if (entry.i == pair.flow_q_variable && !have_q) {
        q_diagonal = entry.v;
        have_q = true;
      } else {
        return {};
      }
    }
    if (!have_p || !have_q || p_diagonal != q_diagonal) return {};
  }

  std::vector<ExactAffineElimination> plan;
  plan.reserve(static_cast<std::size_t>(named_flow_rows));
  for (int i = 0; i < static_cast<int>(row_variable.size()); ++i)
    if (row_variable[static_cast<std::size_t>(i)] >= 0)
      plan.push_back({i, row_variable[static_cast<std::size_t>(i)]});
  if (static_cast<int>(plan.size()) != named_flow_rows) return {};
  return plan;
}

BuiltSubproblem build_dca_subproblem(const QCQPProblem& prob, const Vec& x,
                                     const Convexification& objective_cvx,
                                     const std::vector<Convexification>& ineq_cvx,
                                     const std::set<int>& working_set,
                                     const std::set<int>&
                                         modeled_side_recovery_rows,
                                     const Settings& st,
                                     const std::vector<ExactAffineElimination>&
                                         exact_affine_eliminations,
                                     const SymMatrix* objective_hessian_override = nullptr) {
  BuiltSubproblem out;
  const bool elastic = st.dca_equality_model=="elastic-split" && !prob.equalities.empty();
  const int nvar=prob.n+(elastic?1:0);
  const int eta_idx=elastic?prob.n:-1;
  out.elastic_equalities=elastic;
  out.elastic_index=eta_idx;
  out.p.nvar=nvar;
  out.p.original_dimension=prob.n;
  out.p.prefer_feasible_centered_start = true;
  out.p.objective.constant=structural_quadratic_value(prob.objective,x);
  out.p.objective.linear=prob.objective.gradient(x);
  out.p.objective.base_hessian=objective_hessian_override
      ? *objective_hessian_override : prob.objective.hessian;
  out.p.objective.shift_dimension=prob.n;
  out.p.objective.scalar_shift=objective_cvx.scalar_shift;
  out.p.objective.low_rank=objective_cvx.terms;
  out.p.objective=extend_model_dimension(std::move(out.p.objective),nvar);
  out.p.objective.scalar_shift += st.dca_extra_proximal_regularization;
  out.p.equalities.nvar=nvar;
  if (!elastic)
    out.p.exact_affine_eliminations = exact_affine_eliminations;

  if(elastic) {
    double eta0=0.0;
    for(const auto& h:prob.equalities)
      eta0=std::max(
          eta0,std::abs(structural_quadratic_value(h,x)));
    const double cap=std::max(st.feasibility_tolerance,
        st.dca_equality_elastic_cap_factor*
            std::max(st.feasibility_tolerance,eta0));
    out.elastic_base=eta0;
    out.elastic_cap=cap;
    out.p.objective.linear[eta_idx] += st.dca_equality_elastic_penalty +
        st.dca_equality_elastic_quadratic*eta0;
    if(st.dca_equality_elastic_quadratic>0.0)
      out.p.objective.base_hessian.entries.push_back(
          {eta_idx,eta_idx,st.dca_equality_elastic_quadratic});
    for(int equality_index=0;
        equality_index<static_cast<int>(prob.equalities.size());
        ++equality_index) {
      const auto& h=prob.equalities[static_cast<std::size_t>(equality_index)];
      const int plus=static_cast<int>(out.p.inequalities.size());
      out.p.inequalities.push_back(elastic_equality_majorant(h,x,+1.0,eta0,eta_idx));
      out.inequality_continuation_keys.push_back(inequality_row_key(
          DcaInequalityRowKind::ElasticEqualityPlus,equality_index));
      const int minus=static_cast<int>(out.p.inequalities.size());
      out.p.inequalities.push_back(elastic_equality_majorant(h,x,-1.0,eta0,eta_idx));
      out.inequality_continuation_keys.push_back(inequality_row_key(
          DcaInequalityRowKind::ElasticEqualityMinus,equality_index));
      out.elastic_dual_positions.push_back({plus,minus});
    }
    // eta = eta0 + e >= 0  -> -e - eta0 <= 0
    out.p.inequalities.push_back(linear_inequality(nvar,eta_idx,-1.0,-eta0));
    out.inequality_continuation_keys.push_back(
        inequality_row_key(DcaInequalityRowKind::ElasticEtaLower,0));
    // eta <= cap -> e + eta0 - cap <= 0
    out.p.inequalities.push_back(linear_inequality(nvar,eta_idx,+1.0,eta0-cap));
    out.inequality_continuation_keys.push_back(
        inequality_row_key(DcaInequalityRowKind::ElasticEtaUpper,0));
    out.original_ineq_offset=static_cast<int>(out.p.inequalities.size());
  } else {
    // Classical SQP linearization of original equalities.
    SparseGradientWorkspace equality_workspace(prob.n);
    for(const auto& h:prob.equalities) {
      if(st.lazyqp_affine_support_fastpath) {
        SparseRow row;
        consume_sparse_quadratic_gradient(h,x,equality_workspace,[&](int j,double v) {
          if(v!=0.0) row.entries.push_back({j,v});
        });
        out.p.equalities.rows.push_back(std::move(row));
      } else out.p.equalities.rows.push_back(dense_gradient_row(h.gradient(x)));
      out.p.equalities.rhs.push_back(-structural_quadratic_value(h,x));
    }
    out.original_ineq_offset=0;
  }

  // Fixed variables remain hard affine equalities in both modes.
  for(int j=0;j<prob.n;++j) {
    if(std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
       std::abs(prob.ub[j]-prob.lb[j])<=st.fixed_bound_tolerance) {
      SparseRow row; row.entries.push_back({j,1.0});
      out.p.equalities.rows.push_back(std::move(row));
      out.p.equalities.rhs.push_back(prob.lb[j]-x[j]);
    }
  }

  for(int idx:working_set) {
    out.active_original_ineq.push_back(idx);
    ConvexQuadraticModel model=
        inequality_model(prob.inequalities[idx],x,ineq_cvx[idx]);
    if(st.dca_condensed_modeled_side_filter &&
       st.dca_equality_model=="condensed-retraction" &&
       modeled_side_recovery_rows.count(idx)>0 &&
       model.constant>st.feasibility_tolerance) {
      // A previously accepted modeled-only Maratos excursion must be driven
      // back inside.  Tightening the existing convex majorant has the safe
      // sign: model(d)+margin<=0 implies the original untightened majorant is
      // at most -margin.  This changes no solve count and is disabled outside
      // the explicit transient-side policy.
      model.constant+=st.dca_condensed_modeled_side_recovery_margin*
          std::max(st.feasibility_tolerance,model.constant);
      ++out.modeled_side_recovery_margin_rows;
    }
    out.p.inequalities.push_back(
        extend_model_dimension(std::move(model),nvar));
    out.inequality_continuation_keys.push_back(
        inequality_row_key(DcaInequalityRowKind::OriginalInequality,idx));
  }
  for(int j=0;j<prob.n;++j) {
    const bool fixed=std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
        std::abs(prob.ub[j]-prob.lb[j])<=st.fixed_bound_tolerance;
    if(fixed) continue;
    if(std::isfinite(prob.ub[j])) {
      out.p.inequalities.push_back(linear_inequality(nvar,j,+1.0,x[j]-prob.ub[j]));
      out.inequality_continuation_keys.push_back(
          inequality_row_key(DcaInequalityRowKind::UpperBound,j));
    }
    if(std::isfinite(prob.lb[j])) {
      out.p.inequalities.push_back(linear_inequality(nvar,j,-1.0,prob.lb[j]-x[j]));
      out.inequality_continuation_keys.push_back(
          inequality_row_key(DcaInequalityRowKind::LowerBound,j));
    }
  }
  return out;
}

void seed_inner_dual_warm_start(BuiltSubproblem& built,
                                const QCQPProblem& prob,
                                const MultiplierEstimate& mult,
                                bool valid, const Settings& st) {
  if (!st.inner_dual_warm_start || !valid) return;
  const int me = static_cast<int>(built.p.equalities.rows.size());
  const int mi = static_cast<int>(built.p.inequalities.size());
  Vec proposed_equality_duals(static_cast<std::size_t>(me),0.0);
  Vec proposed_inequality_duals(static_cast<std::size_t>(mi),0.0);
  if(!built.elastic_equalities) {
    const int me_orig=static_cast<int>(prob.equalities.size());
    for(int j=0;j<std::min(me_orig,me) && j<(int)mult.equality.size();++j)
      proposed_equality_duals[static_cast<std::size_t>(j)]=mult.equality[j];
  } else {
    for(int j=0;j<(int)built.elastic_dual_positions.size() &&
                    j<(int)mult.equality.size();++j) {
      const auto [ip,im]=built.elastic_dual_positions[j];
      const double lam=mult.equality[j];
      if(ip>=0 && ip<mi)
        proposed_inequality_duals[static_cast<std::size_t>(ip)]=
            std::max(0.0,lam);
      if(im>=0 && im<mi)
        proposed_inequality_duals[static_cast<std::size_t>(im)]=
            std::max(0.0,-lam);
    }
  }
  if(st.inner_inequality_dual_warm_start) {
    int off=built.original_ineq_offset;
    for(int k=0;k<(int)built.active_original_ineq.size() && off<mi;++k,++off){
      const int idx=built.active_original_ineq[k];
      if(idx>=0 && idx<(int)mult.inequality.size())
        proposed_inequality_duals[static_cast<std::size_t>(off)]=
            std::max(0.0,mult.inequality[idx]);
    }
    for(int j=0;j<prob.n && off<mi;++j){
      const bool fixed=std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
          std::abs(prob.ub[j]-prob.lb[j])<=st.fixed_bound_tolerance;
      if(fixed) continue;
      if(std::isfinite(prob.ub[j]) && off<mi) {
        if(j<(int)mult.upper_bound.size())
          proposed_inequality_duals[static_cast<std::size_t>(off)]=
              std::max(0.0,mult.upper_bound[j]);
        ++off;
      }
      if(std::isfinite(prob.lb[j]) && off<mi) {
        if(j<(int)mult.lower_bound.size())
          proposed_inequality_duals[static_cast<std::size_t>(off)]=
              std::max(0.0,mult.lower_bound[j]);
        ++off;
      }
    }
  }

  Vec proposed_slacks;
  const bool mapped_single_loop_state=st.dca_single_loop_primal_dual &&
      st.inner_inequality_dual_warm_start;
  if(mapped_single_loop_state) {
    const auto finite=[](const Vec& values) {
      return std::all_of(values.begin(),values.end(),
                         [](double value){ return std::isfinite(value); });
    };
    if(!finite(proposed_equality_duals) ||
       !finite(proposed_inequality_duals) ||
       !std::all_of(proposed_inequality_duals.begin(),
                    proposed_inequality_duals.end(),
                    [](double value){ return value>=0.0; }))
      return;

    const Vec zero(static_cast<std::size_t>(built.p.nvar),0.0);
    const double gscale=std::max(
        1e-2,std::min(1.0,inf_norm(built.p.objective.gradient(zero))));
    const double target_mu=std::max(
        {st.lazyqp_barrier_mu0,gscale,100.0*st.inner_tolerance});
    const double slack_push=std::max(
        st.minimum_positive,
        std::min(st.feasibility_tolerance,10.0*st.inner_tolerance));
    proposed_slacks.resize(static_cast<std::size_t>(mi),1.0);
    for(int i=0;i<mi;++i) {
      const double phi=
          built.p.inequalities[static_cast<std::size_t>(i)].value(zero);
      if(!std::isfinite(phi)) return;
      const double slack=std::max(std::max(0.0,-phi),slack_push);
      double& dual=proposed_inequality_duals[static_cast<std::size_t>(i)];
      const double product=slack*dual;
      if(!std::isfinite(product)) return;
      if(product>target_mu)
        dual=std::min(dual,target_mu/std::max(slack,st.minimum_positive));
      proposed_slacks[static_cast<std::size_t>(i)]=slack;
    }
    if(!finite(proposed_inequality_duals) || !finite(proposed_slacks) ||
       !std::all_of(proposed_slacks.begin(),proposed_slacks.end(),
                    [](double value){ return value>0.0; }))
      return;
  }

  // Commit only after the complete mapped state (including the independently
  // remapped slacks required after a retraction/SOC invalidates direct
  // continuation) has passed its dimensional, finite and interior checks.
  built.p.initial_equality_duals=std::move(proposed_equality_duals);
  built.p.initial_inequality_duals=std::move(proposed_inequality_duals);
  built.p.has_dual_warm_start=true;
  built.p.dual_warm_start_is_mapped=true;
  if(mapped_single_loop_state) {
    built.p.has_primal_warm_start=true;
    built.p.initial_primal.assign(static_cast<std::size_t>(built.p.nvar),0.0);
    built.p.has_inequality_slack_warm_start=true;
    built.p.initial_inequality_slacks=std::move(proposed_slacks);
  }
}

struct DirectContinuationSeedResult {
  bool attempted=false;
  bool seeded=false;
  int matched_rows=0;
  int new_rows=0;
  int dropped_rows=0;
  int centrality_capped_rows=0;
  const char* reason="not-attempted";
};

DirectContinuationSeedResult seed_direct_inner_dual_continuation(
    BuiltSubproblem& built,const Vec& equality_duals,
    const Vec& inequality_duals,const Vec& inequality_slacks,
    const Vec& inequality_primal_residuals,
    const std::vector<std::uint64_t>& inequality_keys,
    bool valid,const Settings& st) {
  DirectContinuationSeedResult result;
  if((!st.inner_dual_warm_start && !st.dca_single_loop_primal_dual) ||
     !valid)
    return result;
  result.attempted=true;
  if((int)equality_duals.size()!=(int)built.p.equalities.rows.size()) {
    result.reason="equality-dimension";
    return result;
  }
  const auto finite=[](const Vec& values) {
    return std::all_of(values.begin(),values.end(),
                       [](double value){ return std::isfinite(value); });
  };
  if(!finite(equality_duals) || !finite(inequality_duals)) {
    result.reason="nonfinite-dual";
    return result;
  }
  if(inequality_keys.size()!=inequality_duals.size() ||
     built.inequality_continuation_keys.size()!=
         built.p.inequalities.size()) {
    result.reason="row-key-dimension";
    return result;
  }
  const bool transport_slacks=st.dca_single_loop_primal_dual;
  if(transport_slacks &&
     (inequality_slacks.size()!=inequality_duals.size() ||
      inequality_primal_residuals.size()!=inequality_duals.size())) {
    result.reason="slack-dimension";
    return result;
  }

  std::unordered_map<std::uint64_t,std::size_t> old_rows;
  old_rows.reserve(inequality_keys.size());
  for(std::size_t i=0;i<inequality_keys.size();++i) {
    if(!old_rows.emplace(inequality_keys[i],i).second) {
      result.reason="duplicate-old-row-key";
      return result;
    }
    if(inequality_duals[i]<0.0 ||
       (transport_slacks &&
        (!std::isfinite(inequality_slacks[i]) ||
         !(inequality_slacks[i]>0.0) ||
         !std::isfinite(inequality_primal_residuals[i])))) {
      result.reason="invalid-old-row-state";
      return result;
    }
  }

  Vec proposed_duals(built.p.inequalities.size(),1.0);
  Vec proposed_slacks;
  if(transport_slacks)
    proposed_slacks.resize(built.p.inequalities.size(),1.0);
  std::set<std::uint64_t> new_row_keys;
  Vec zero(static_cast<std::size_t>(built.p.nvar),0.0);
  const double gscale=std::max(
      1e-2,std::min(1.0,inf_norm(built.p.objective.gradient(zero))));
  const double target_mu=std::max(
      {st.lazyqp_barrier_mu0,gscale,100.0*st.inner_tolerance});
  const double slack_push=std::max(1e-4,10.0*st.inner_tolerance);

  for(int i=0;i<(int)built.p.inequalities.size();++i) {
    const std::uint64_t key=
        built.inequality_continuation_keys[static_cast<std::size_t>(i)];
    if(!new_row_keys.insert(key).second) {
      result.reason="duplicate-new-row-key";
      return result;
    }
    const auto previous=old_rows.find(key);
    const bool matched=previous!=old_rows.end();
    const double phi=built.p.inequalities[static_cast<std::size_t>(i)].value(zero);
    if(!std::isfinite(phi)) {
      result.reason="nonfinite-new-row";
      return result;
    }
    const double raw=std::max(0.0,-phi);
    if(matched) {
      ++result.matched_rows;
      const std::size_t old=previous->second;
      double remapped_dual=inequality_duals[old];
      if(transport_slacks) {
        const SingleLoopSlackRemap remap=remap_single_loop_slack_residual(
            inequality_primal_residuals[old],phi,slack_push);
        if(!remap.valid) {
          result.reason="nonfinite-remapped-slack";
          return result;
        }
        const double remapped_slack=remap.slack;
        // A semantic row may survive while its new majorant value at y=0
        // changes substantially.  Keeping z verbatim in that case can turn a
        // previously central pair into an arbitrarily large s*z pair and make
        // the guarded warm start reject the whole state.  Preserve z whenever
        // possible, but cap componentwise complementarity at the same target
        // used by the validated cold start.  In particular, exact/numerical
        // ZERO_DUAL entries remain zero and are repaired transactionally by
        // LazyQP's existing continuation barrier.
        const double product=remapped_slack*remapped_dual;
        if(!std::isfinite(product)) {
          result.reason="nonfinite-remapped-complementarity";
          return result;
        }
        if(product>target_mu) {
          remapped_dual=std::min(
              remapped_dual,target_mu/std::max(remapped_slack,
                                               st.minimum_positive));
          ++result.centrality_capped_rows;
        }
        proposed_slacks[static_cast<std::size_t>(i)]=remapped_slack;
      }
      proposed_duals[static_cast<std::size_t>(i)]=remapped_dual;
    } else {
      ++result.new_rows;
      const double centered_slack=std::max(raw,slack_push);
      proposed_duals[static_cast<std::size_t>(i)]=std::clamp(
          target_mu/std::max(centered_slack,target_mu),
          st.minimum_positive,10.0);
      if(transport_slacks)
        proposed_slacks[static_cast<std::size_t>(i)]=centered_slack;
    }
  }
  result.dropped_rows=static_cast<int>(inequality_keys.size())-
      result.matched_rows;
  if(!finite(proposed_duals) ||
     !std::all_of(proposed_duals.begin(),proposed_duals.end(),
                  [](double value){ return value>=0.0; }) ||
     (transport_slacks &&
      (!finite(proposed_slacks) ||
       !std::all_of(proposed_slacks.begin(),proposed_slacks.end(),
                    [](double value){ return value>0.0; })))) {
    result.reason="invalid-proposed-row-state";
    return result;
  }

  // The accepted outer point is the new majorant origin, hence its primal
  // continuation coordinate is y=0.  Old independent slacks cannot be copied
  // verbatim because every majorant has new functions phi_i.  Rows matched by
  // semantic key preserve r_pri=phi_old(y_old)+s_old exactly whenever the
  // implied new slack is positive; genuinely new rows get an independent
  // centered positive (z,s) pair.
  // Commit only after every dimension, old slack and new model value has
  // passed.  A failed proposal leaves `built` byte-for-byte on its previous
  // initialization route instead of leaking a partial warm state.
  built.p.initial_equality_duals=equality_duals;
  built.p.initial_inequality_duals=std::move(proposed_duals);
  built.p.has_dual_warm_start=true;
  built.p.dual_warm_start_is_mapped=false;
  if(transport_slacks) {
    built.p.has_primal_warm_start=true;
    built.p.initial_primal.assign(
        static_cast<std::size_t>(built.p.nvar),0.0);
    built.p.has_inequality_slack_warm_start=true;
    built.p.initial_inequality_slacks=std::move(proposed_slacks);
  }
  result.seeded=true;
  result.reason="accepted";
  return result;
}

MultiplierEstimate multipliers_from_inner(const QCQPProblem& prob,
                                           const BuiltSubproblem& built,
                                           const InnerResult& inner,
                                           const Settings& st) {
  MultiplierEstimate out;
  out.equality.assign(prob.equalities.size(), 0.0);
  out.inequality.assign(prob.inequalities.size(), 0.0);
  out.lower_bound.assign(prob.n, 0.0);
  out.upper_bound.assign(prob.n, 0.0);
  if(!built.elastic_equalities) {
    const int me = static_cast<int>(prob.equalities.size());
    for (int j = 0; j < me && j < static_cast<int>(inner.equality_duals.size()); ++j)
      out.equality[j] = inner.equality_duals[j];
  } else {
    for(int j=0;j<(int)built.elastic_dual_positions.size() &&
                    j<(int)out.equality.size();++j) {
      const auto [ip,im]=built.elastic_dual_positions[j];
      const double mup=(ip>=0 && ip<(int)inner.inequality_duals.size())
          ? std::max(0.0,inner.inequality_duals[ip]) : 0.0;
      const double mum=(im>=0 && im<(int)inner.inequality_duals.size())
          ? std::max(0.0,inner.inequality_duals[im]) : 0.0;
      out.equality[j]=mup-mum;
    }
  }

  int off=built.original_ineq_offset;
  for (int k=0; k < static_cast<int>(built.active_original_ineq.size()) &&
         off < static_cast<int>(inner.inequality_duals.size()); ++k,++off) {
    const int idx = built.active_original_ineq[k];
    if (idx >= 0 && idx < static_cast<int>(out.inequality.size()))
      out.inequality[idx] = std::max(0.0, inner.inequality_duals[off]);
  }
  for (int j = 0; j < prob.n; ++j) {
    const bool fixed = std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
        std::abs(prob.ub[j]-prob.lb[j]) <= st.fixed_bound_tolerance;
    if (fixed) continue;
    if (std::isfinite(prob.ub[j]) && off < static_cast<int>(inner.inequality_duals.size()))
      out.upper_bound[j] = std::max(0.0, inner.inequality_duals[off++]);
    if (std::isfinite(prob.lb[j]) && off < static_cast<int>(inner.inequality_duals.size()))
      out.lower_bound[j] = std::max(0.0, inner.inequality_duals[off++]);
  }
  out.valid = inner.success || dca_single_loop_provisional_state(st, inner);
  return out;
}

MultiplierEstimate blend_multipliers(const MultiplierEstimate& oldm,
                                      const MultiplierEstimate& newm,
                                      double alpha) {
  alpha = std::max(0.0, std::min(1.0, alpha));
  MultiplierEstimate out = newm;
  out.valid = oldm.valid && newm.valid;
  if (oldm.equality.size() == newm.equality.size())
    for (std::size_t j = 0; j < out.equality.size(); ++j)
      out.equality[j] = (1.0-alpha)*oldm.equality[j] + alpha*newm.equality[j];
  if (oldm.inequality.size() == newm.inequality.size())
    for (std::size_t i = 0; i < out.inequality.size(); ++i)
      out.inequality[i] = std::max(0.0,(1.0-alpha)*oldm.inequality[i] + alpha*newm.inequality[i]);
  if (oldm.lower_bound.size() == newm.lower_bound.size())
    for (std::size_t i = 0; i < out.lower_bound.size(); ++i)
      out.lower_bound[i] = std::max(0.0,(1.0-alpha)*oldm.lower_bound[i] + alpha*newm.lower_bound[i]);
  if (oldm.upper_bound.size() == newm.upper_bound.size())
    for (std::size_t i = 0; i < out.upper_bound.size(); ++i)
      out.upper_bound[i] = std::max(0.0,(1.0-alpha)*oldm.upper_bound[i] + alpha*newm.upper_bound[i]);
  return out;
}

BuiltSubproblem build_tangential_fallback(const QCQPProblem& prob, const Vec& x,
                                            const std::set<int>& working_set,
                                            const Settings& st) {
  BuiltSubproblem out;
  out.p.nvar = prob.n;
  out.p.original_dimension = prob.n;
  out.p.objective.constant = 0.0;
  out.p.objective.linear = prob.objective.gradient(x);
  out.p.objective.base_hessian.n = prob.n;
  for (int j = 0; j < prob.n; ++j) out.p.objective.base_hessian.entries.push_back({j,j,1.0});
  out.p.equalities.nvar = prob.n;
  for (const auto& h : prob.equalities) {
    out.p.equalities.rows.push_back(dense_gradient_row(h.gradient(x)));
    out.p.equalities.rhs.push_back(-structural_quadratic_value(h,x));
  }
  for (int j = 0; j < prob.n; ++j) {
    if (std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
        std::abs(prob.ub[j]-prob.lb[j]) <= st.fixed_bound_tolerance) {
      SparseRow row; row.entries.push_back({j,1.0});
      out.p.equalities.rows.push_back(std::move(row));
      out.p.equalities.rhs.push_back(prob.lb[j]-x[j]);
    }
  }
  // The fallback is first-order and used only locally near feasibility.  Keep
  // the currently relevant inequalities as linear tangent half-spaces.
  for (int i : working_set) {
    ConvexQuadraticModel q;
    q.constant = structural_quadratic_value(prob.inequalities[i],x);
    q.linear = prob.inequalities[i].gradient(x);
    q.base_hessian.n = prob.n;
    out.p.inequalities.push_back(std::move(q));
  }
  for (int j = 0; j < prob.n; ++j) {
    const bool fixed = std::isfinite(prob.lb[j]) && std::isfinite(prob.ub[j]) &&
                       std::abs(prob.ub[j]-prob.lb[j]) <= st.fixed_bound_tolerance;
    if (fixed) continue;
    if (std::isfinite(prob.ub[j])) out.p.inequalities.push_back(linear_inequality(prob.n,j,+1.0,x[j]-prob.ub[j]));
    if (std::isfinite(prob.lb[j])) out.p.inequalities.push_back(linear_inequality(prob.n,j,-1.0,prob.lb[j]-x[j]));
  }
  return out;
}

InnerResult solve_equality_qp(const ConvexSubproblem& p, const Settings& st,
                                    SymmetricFactorization* persistent_factor=nullptr) {
  InnerResult out;
  const int n = p.nvar;
  const int me = static_cast<int>(p.equalities.rows.size());

  struct Border {
    double beta = 0.0;
    const Vec* direction = nullptr;
  };
  std::vector<Border> borders;
  borders.reserve(p.objective.low_rank.size());
  for (const auto& term : p.objective.low_rank) {
    if (term.beta == 0.0) continue;
    if (!(term.beta > 0.0) || static_cast<int>(term.direction.size()) != n) {
      out.message = "equality-QP received an invalid PSD low-rank correction";
      return out;
    }
    borders.push_back({term.beta, &term.direction});
  }

  // Do not materialize beta*u*u^T.  CATENA/CATENARY have equality-only
  // subproblems, and projected tangent corrections are generally dense even
  // though the native QCQP Hessians and Jacobian are sparse.  The bordered
  // representation
  //
  //   [ H  A^T  U ] [d]   [-g]
  //   [ A   0    0 ] [l] = [ b]
  //   [U^T  0  -D^-1] [w] [ 0]
  //
  // has Schur complement H+U D U^T and costs only O(n*r) stored coefficients.
  const int nb = static_cast<int>(borders.size());
  const int equality0 = n;
  const int border0 = n + me;
  const int N = n + me + nb;
  std::vector<SymmetricTriplet> K;
  std::size_t reserve = p.objective.base_hessian.entries.size() +
      static_cast<std::size_t>(p.objective.shift_dimension) +
      static_cast<std::size_t>(me + nb);
  for (const auto& row : p.equalities.rows) reserve += row.entries.size();
  for (const auto& border : borders) {
    for (double value : *border.direction)
      if (value != 0.0) ++reserve;
  }
  K.reserve(reserve);
  for (const auto& e : p.objective.base_hessian.entries)
    K.push_back({e.i,e.j,e.v});
  for (int j=0;j<p.objective.shift_dimension;++j)
    K.push_back({j,j,p.objective.scalar_shift});
  for (int r=0;r<me;++r)
    for (const auto& [j,a] : p.equalities.rows[r].entries)
      if (a != 0.0) K.push_back({j,equality0+r,a});
  for (int b=0;b<nb;++b) {
    const auto& border=borders[static_cast<std::size_t>(b)];
    for (int j=0;j<n;++j) {
      const double value=(*border.direction)[static_cast<std::size_t>(j)];
      if (value != 0.0) K.push_back({j,border0+b,value});
    }
    K.push_back({border0+b,border0+b,-1.0/border.beta});
  }

  Vec rhs(static_cast<std::size_t>(N),0.0);
  for (int j=0;j<n;++j) rhs[static_cast<std::size_t>(j)] =
      -p.objective.linear[static_cast<std::size_t>(j)];
  for (int r=0;r<me;++r) rhs[static_cast<std::size_t>(equality0+r)] =
      p.equalities.rhs[static_cast<std::size_t>(r)];

  const bool adaptive_kkt=st.kkt_adaptive_regularization &&
      (st.linear_backend=="native" || st.linear_backend=="ldl" ||
       st.linear_backend=="qdldl" || st.linear_backend=="cholesk" ||
       st.linear_backend=="mumps");
  double rho_x=adaptive_kkt ? std::max(st.kkt_rho_x_floor,st.kkt_primal_regularization)
                            : st.kkt_primal_regularization;
  // Equality-only QPs do not need a barrier-scale dual shift. Starting at
  // delta_dual_base=1e-8 perturbs A y=b into A y-delta*lambda=b and caused
  // exact affine-equality subproblems to fail their original residual check.
  // Start at the true numerical floor and grow only after a factorization
  // failure.
  double delta_d=adaptive_kkt
      ? std::clamp(std::max(st.kkt_delta_dual_floor,st.kkt_dual_regularization),
                   st.kkt_delta_dual_floor,st.kkt_delta_dual_max)
      : st.kkt_dual_regularization;

  // The backend must not apply its generic "all dual variables" shift to the
  // low-rank border coordinates, because that would alter beta.  Both adaptive
  // and fixed regularizations are therefore assembled explicitly below.
  Settings local_factor_settings=st;
  local_factor_settings.kkt_adaptive_regularization=true;
  std::string msg;
  std::unique_ptr<SymmetricFactorization> owned_factor;
  SymmetricFactorization* factor=persistent_factor;
  if(!factor) {
    owned_factor=make_symmetric_factorization(local_factor_settings,msg);
    factor=owned_factor.get();
  }
  const int symbolic_before=factor ? factor->symbolic_analyses() : 0;
  const int reuse_before=factor ? factor->symbolic_reuses() : 0;
  const int invalidation_before=factor ? factor->symbolic_invalidations() : 0;
  bool factored=false;
  if(factor){
    for(int attempt=0;attempt<=std::max(0,st.kkt_regularization_retry_max);++attempt){
      auto Kreg=K;
      if(rho_x!=0.0)
        for(int j=0;j<n;++j) Kreg.push_back({j,j,rho_x});
      if(delta_d!=0.0)
        for(int r=0;r<me;++r)
          Kreg.push_back({equality0+r,equality0+r,-delta_d});
      ++out.factorizations;
      ++out.lazyqp_full_kkt_factorizations;
      if(factor->factor(N,n,Kreg,msg)){ factored=true; break; }
      if(!adaptive_kkt || attempt==st.kkt_regularization_retry_max) break;
      ++out.kkt_regularization_retries;
      rho_x=std::min(st.kkt_rho_x_max,std::max(st.kkt_rho_x_floor,
          rho_x>0.0 ? rho_x*st.kkt_rho_x_growth : st.kkt_rho_x_floor));
      delta_d=std::min(st.kkt_delta_dual_max,
                       std::max(st.kkt_delta_dual_floor,delta_d*st.kkt_rho_x_growth));
    }
  }
  out.symbolic_analyses=factor
      ? std::max(0,factor->symbolic_analyses()-symbolic_before) : 0;
  out.symbolic_reuses=factor
      ? std::max(0,factor->symbolic_reuses()-reuse_before) : 0;
  out.symbolic_invalidations=factor
      ? std::max(0,factor->symbolic_invalidations()-invalidation_before) : 0;
  out.lazyqp_kkt_dimension=N;
  out.lazyqp_full_kkt_dimension=N;
  out.final_kkt_rho_x=rho_x; out.max_kkt_rho_x=rho_x;
  out.final_kkt_delta_d=delta_d; out.max_kkt_delta_d=delta_d;
  if (!factor || !factored || !factor->solve(rhs,msg)) {
    if(factor){
      out.max_kkt_backward_error=factor->last_backward_error();
      out.max_kkt_residual_inf=factor->last_residual_inf();
      out.kkt_refinement_steps=factor->last_refinement_steps();
    }
    out.message = "equality-QP bordered KKT solve failed [" +
        st.linear_backend + "]: " + msg;
    return out;
  }
  out.max_kkt_backward_error=factor->last_backward_error();
  out.max_kkt_residual_inf=factor->last_residual_inf();
  out.kkt_refinement_steps=factor->last_refinement_steps();
  out.y.assign(rhs.begin(),rhs.begin()+n);
  out.equality_duals.assign(rhs.begin()+equality0,rhs.begin()+border0);
  Vec border_coordinates(rhs.begin()+border0,rhs.end());
  out.iterations=1;

  struct EqualityQPKktMetrics {
    Vec residual;
    double dual_inf = 0.0;
    double equality_inf = 0.0;
    double dual_scale = 1.0;
    double equality_scale = 1.0;
    double scaled_inf = 0.0;
    double strict_dual_tolerance = 0.0;
    double strict_equality_tolerance = 0.0;
    double dual_tolerance = 0.0;
    double equality_tolerance = 0.0;
    bool strictly_converged = false;
    bool acceptable = false;
  };

  auto original_kkt_metrics = [&]() {
    EqualityQPKktMetrics m;
    m.residual.assign(static_cast<std::size_t>(n+me),0.0);

    const Vec objective_gradient=p.objective.gradient(out.y);
    Vec equality_dual_gradient(static_cast<std::size_t>(n),0.0);
    Vec full_gradient=objective_gradient;
    for (int r=0;r<me;++r) {
      for (const auto& [j,a] : p.equalities.rows[r].entries) {
        equality_dual_gradient[static_cast<std::size_t>(j)]+=
            out.equality_duals[static_cast<std::size_t>(r)]*a;
        full_gradient[static_cast<std::size_t>(j)]+=
            out.equality_duals[static_cast<std::size_t>(r)]*a;
      }
      double lhs=0.0;
      for (const auto& [j,a] : p.equalities.rows[r].entries)
        lhs+=a*out.y[static_cast<std::size_t>(j)];
      const double rr=lhs-p.equalities.rhs[static_cast<std::size_t>(r)];
      m.residual[static_cast<std::size_t>(n+r)]=rr;
      m.equality_inf=std::max(m.equality_inf,std::abs(rr));
      m.equality_scale=std::max({m.equality_scale,std::abs(lhs),
                                 std::abs(p.equalities.rhs[static_cast<std::size_t>(r)])});
    }
    for(int j=0;j<n;++j)
      m.residual[static_cast<std::size_t>(j)]=full_gradient[static_cast<std::size_t>(j)];
    m.dual_inf=inf_norm(full_gradient);
    m.dual_scale=std::max({1.0,inf_norm(objective_gradient),
                           inf_norm(equality_dual_gradient)});
    m.scaled_inf=std::max(m.dual_inf/m.dual_scale,
                          m.equality_inf/m.equality_scale);

    const double eps=std::numeric_limits<double>::epsilon();
    // First try to meet the requested absolute inner tolerance.  Add only the
    // amount of floating-point uncertainty needed for a genuine boundary
    // comparison (BYRDSPHR), and cap it so a large row/RHS scale cannot weaken
    // the mathematical contract.
    const double dual_roundoff=std::min(
        0.1*st.inner_tolerance,100.0*eps*m.dual_scale);
    const double equality_roundoff=std::min(
        0.1*st.inner_tolerance,100.0*eps*m.equality_scale);
    m.strict_dual_tolerance=st.inner_tolerance+dual_roundoff;
    m.strict_equality_tolerance=st.inner_tolerance+equality_roundoff;
    m.strictly_converged=m.dual_inf<=m.strict_dual_tolerance &&
                         m.equality_inf<=m.strict_equality_tolerance;

    // If defect correction stagnates because the original KKT equations are
    // ill-conditioned, permit the usual mixed absolute/relative inexact-QP
    // criterion, but cap it at one tenth of the requested outer certificate.
    // Thus CATENA/CATENARY may accept a 4.9e-8 stationarity residual when it is
    // a 1e-12 relative cancellation, while the final QCQP target remains 1e-6.
    const double dual_outer_budget=0.1*std::max(st.inner_tolerance,
        st.dual_infeasibility_tolerance);
    const double equality_outer_budget=0.1*std::max(st.inner_tolerance,
        st.feasibility_tolerance);
    const double dual_scaled_allowance=std::min(
        dual_outer_budget,st.qp_model_solved_tol*m.dual_scale);
    const double equality_scaled_allowance=std::min(
        equality_outer_budget,st.qp_model_solved_tol*m.equality_scale);
    m.dual_tolerance=std::max(m.strict_dual_tolerance,
                              dual_scaled_allowance);
    m.equality_tolerance=std::max(m.strict_equality_tolerance,
                                  equality_scaled_allowance);
    m.acceptable=m.dual_inf<=m.dual_tolerance &&
                 m.equality_inf<=m.equality_tolerance;
    return m;
  };

  // Residual of the unregularized bordered equations.  Maintaining the border
  // coordinates during refinement is essential: a zero RHS in those rows is
  // only valid when U^T d-D^-1 w is already zero to working precision.
  auto bordered_kkt_residual = [&]() {
    Vec residual(static_cast<std::size_t>(N),0.0);
    Vec primal=p.objective.base_hessian.apply(out.y);
    for(int j=0;j<n;++j) {
      primal[static_cast<std::size_t>(j)]+=
          p.objective.linear[static_cast<std::size_t>(j)];
      if(j<p.objective.shift_dimension)
        primal[static_cast<std::size_t>(j)]+=
            p.objective.scalar_shift*out.y[static_cast<std::size_t>(j)];
    }
    for(int r=0;r<me;++r) {
      const double lambda=out.equality_duals[static_cast<std::size_t>(r)];
      double lhs=0.0;
      for(const auto& [j,a]:p.equalities.rows[static_cast<std::size_t>(r)].entries) {
        primal[static_cast<std::size_t>(j)]+=lambda*a;
        lhs+=a*out.y[static_cast<std::size_t>(j)];
      }
      residual[static_cast<std::size_t>(equality0+r)]=
          lhs-p.equalities.rhs[static_cast<std::size_t>(r)];
    }
    for(int b=0;b<nb;++b) {
      const auto& border=borders[static_cast<std::size_t>(b)];
      const double w=border_coordinates[static_cast<std::size_t>(b)];
      axpy(w,*border.direction,primal);
      residual[static_cast<std::size_t>(border0+b)]=
          dot(*border.direction,out.y)-w/border.beta;
    }
    std::copy(primal.begin(),primal.end(),residual.begin());
    return residual;
  };

  // The factorization is of the numerically regularized bordered KKT matrix,
  // but the subproblem contract is the ORIGINAL equality-QP.  Refine against
  // the unregularized bordered equations and assess convergence against the
  // original condensed objective gradient.
  const int original_refine_max=12;
  double previous_measure=std::numeric_limits<double>::infinity();
  int stagnant_refinements=0;
  EqualityQPKktMetrics metrics=original_kkt_metrics();
  for(int refine=0;refine<original_refine_max && !metrics.strictly_converged;++refine) {
    Vec correction_rhs=bordered_kkt_residual();
    for(double& v:correction_rhs) v=-v;
    if(!factor->solve(correction_rhs,msg)) break;
    for(int j=0;j<n;++j)
      out.y[static_cast<std::size_t>(j)]+=correction_rhs[static_cast<std::size_t>(j)];
    for(int r=0;r<me;++r)
      out.equality_duals[static_cast<std::size_t>(r)]+=
          correction_rhs[static_cast<std::size_t>(equality0+r)];
    for(int b=0;b<nb;++b)
      border_coordinates[static_cast<std::size_t>(b)]+=
          correction_rhs[static_cast<std::size_t>(border0+b)];
    out.kkt_refinement_steps += 1;
    out.max_kkt_backward_error=std::max(out.max_kkt_backward_error,
                                        factor->last_backward_error());
    out.max_kkt_residual_inf=std::max(out.max_kkt_residual_inf,
                                     factor->last_residual_inf());

    metrics=original_kkt_metrics();
    const double measure=std::max(
        metrics.dual_inf/std::max(metrics.dual_tolerance,
                                  std::numeric_limits<double>::min()),
        metrics.equality_inf/std::max(metrics.equality_tolerance,
                                      std::numeric_limits<double>::min()));
    if(measure>=0.95*previous_measure) ++stagnant_refinements;
    else stagnant_refinements=0;
    previous_measure=measure;
    if(stagnant_refinements>=3) break;
  }

  // Large no-pivot regularization can make the historical Richardson defect
  // correction stagnate even though the regularized factorization itself is
  // accurate.  Preserve that established path exactly and enter this fallback
  // only after it fails the ORIGINAL equality-QP residual contract.
  if(st.equality_qp_original_kkt_gmres && !metrics.acceptable) {
    ++out.equality_qp_original_kkt_gmres_attempts;
    const Vec residual_before=bordered_kkt_residual();
    const double bordered_before=inf_norm(residual_before);
    const double dual_before=metrics.dual_inf;
    const double equality_before=metrics.equality_inf;
    const double metric_before=std::max(
        metrics.dual_inf/std::max(metrics.dual_tolerance,
                                  std::numeric_limits<double>::min()),
        metrics.equality_inf/std::max(metrics.equality_tolerance,
                                      std::numeric_limits<double>::min()));

    Vec target=residual_before;
    for(double& value:target) value=-value;
    auto apply_original_bordered_kkt=[&](const Vec& x,Vec& y) {
      y.assign(static_cast<std::size_t>(N),0.0);
      for(const auto& entry:K) {
        y[static_cast<std::size_t>(entry.i)]+=
            entry.value*x[static_cast<std::size_t>(entry.j)];
        if(entry.i!=entry.j)
          y[static_cast<std::size_t>(entry.j)]+=
              entry.value*x[static_cast<std::size_t>(entry.i)];
      }
    };
    auto apply_regularized_inverse=[&](const Vec& rhs_value,Vec& solution,
                                        std::string& why) {
      solution=rhs_value;
      const bool ok=factor->solve(solution,why);
      if(ok) {
        out.max_kkt_backward_error=std::max(out.max_kkt_backward_error,
                                            factor->last_backward_error());
        out.max_kkt_residual_inf=std::max(out.max_kkt_residual_inf,
                                         factor->last_residual_inf());
      }
      return ok;
    };
    const double gmres_tolerance=std::max(
        std::numeric_limits<double>::min(),
        0.5*std::min(metrics.dual_tolerance,metrics.equality_tolerance));

    Vec correction;
    const auto gmres=original_kkt_right_preconditioned_gmres(
        apply_original_bordered_kkt,apply_regularized_inverse,target,correction,
        st.equality_qp_original_kkt_gmres_max_iterations,gmres_tolerance);
    out.equality_qp_original_kkt_gmres_iterations+=gmres.iterations;

    const Vec y_before=out.y;
    const Vec lambda_before=out.equality_duals;
    const Vec coordinates_before=border_coordinates;
    if(correction.size()==static_cast<std::size_t>(N)) {
      for(int j=0;j<n;++j)
        out.y[static_cast<std::size_t>(j)]+=
            correction[static_cast<std::size_t>(j)];
      for(int r=0;r<me;++r)
        out.equality_duals[static_cast<std::size_t>(r)]+=
            correction[static_cast<std::size_t>(equality0+r)];
      for(int b=0;b<nb;++b)
        border_coordinates[static_cast<std::size_t>(b)]+=
            correction[static_cast<std::size_t>(border0+b)];
    }

    const EqualityQPKktMetrics metrics_after=original_kkt_metrics();
    const double metric_after=std::max(
        metrics_after.dual_inf/std::max(metrics_after.dual_tolerance,
                                        std::numeric_limits<double>::min()),
        metrics_after.equality_inf/std::max(metrics_after.equality_tolerance,
                                            std::numeric_limits<double>::min()));
    const double bordered_after=inf_norm(bordered_kkt_residual());
    const double bordered_roundoff_guard=bordered_before+
        100.0*std::numeric_limits<double>::epsilon()*
        std::max(1.0,bordered_before);

    // This is a rescue, not a new inexact-QP path: commit only if the existing
    // original-QP certificate is actually reached and both normalized and
    // bordered residuals improve.  Otherwise restore every optimization
    // variable and multiplier exactly as they were before GMRES.
    const bool accepted=correction.size()==static_cast<std::size_t>(N) &&
        metrics_after.acceptable && std::isfinite(metric_after) &&
        std::isfinite(bordered_after) && metric_after<metric_before &&
        bordered_after<=bordered_roundoff_guard;
    if(accepted) {
      metrics=metrics_after;
      ++out.equality_qp_original_kkt_gmres_acceptances;
    } else {
      out.y=y_before;
      out.equality_duals=lambda_before;
      border_coordinates=coordinates_before;
    }

    if(st.verbose) {
      std::cout << "[EQUALITY-QP-ORIGINAL-KKT-GMRES] rho_x="
                << std::scientific << rho_x << " delta_d=" << delta_d
                << " dual_before=" << dual_before
                << " equality_before=" << equality_before
                << " bordered_before=" << bordered_before
                << " metric_before=" << metric_before
                << " metric_after=" << (accepted?metric_after:metric_before)
                << " bordered_after="
                << (accepted?bordered_after:bordered_before)
                << " explicit_residual2=" << gmres.residual_2
                << " iter=" << gmres.iterations
                << " accepted=" << (accepted?1:0)
                << " krylov_converged=" << (gmres.converged?1:0)
                << "\n" << std::defaultfloat;
    }
  }

  metrics=original_kkt_metrics();
  out.dual_inf=metrics.dual_inf;
  out.equality_inf=metrics.equality_inf;
  out.primal_residual_inf=out.equality_inf;
  out.complementarity=0.0;
  out.scaled_model_residual_inf=metrics.scaled_inf;
  out.success=metrics.acceptable;
  if(out.success) out.message=nb>0
      ? "equality-QP solved with sparse bordered low-rank corrections"
      : "equality-QP solved";
  else {
    std::ostringstream os;
    os << std::setprecision(17)
       << "equality-QP residual check failed: dual=" << std::scientific
       << out.dual_inf << "/" << metrics.dual_tolerance << " equality="
       << out.equality_inf << "/" << metrics.equality_tolerance
       << " scaled=" << metrics.scaled_inf << "/"
       << st.qp_model_solved_tol;
    out.message=os.str();
  }
  return out;
}
InnerResult solve_inner(const ConvexSubproblem& p, const Settings& st,
                        SymmetricFactorization* persistent_equality_factor=nullptr) {
  // `allow_dca_single_loop` is the per-model authority: standalone and
  // auxiliary equality QPs retain their strict historical contract.  The
  // Lazy-DCA primary loop separately rejects an empty-barrier majorant before
  // it can reach this dispatcher, even when that authority is unavailable.
  if (st.dca_single_loop_primal_dual && p.allow_dca_single_loop &&
      p.inequalities.empty()) {
    InnerResult failed;
    failed.message =
        "single-loop primal-dual majorant requires at least one inequality row";
    return failed;
  }
  // The SOCP objective epigraph remains a genuine rotated cone when there
  // are no inequality rows.  Dispatch it before the historical direct
  // equality-QP shortcut so the requested backend is never silently changed.
  if (st.inner_backend == "socp") {
    ConvexQCQPSOCPKernel socp(st);
    return socp.solve(p);
  }
  if (p.inequalities.empty())
    return solve_equality_qp(p, st, persistent_equality_factor);
  if (st.inner_backend == "ipopt") {
    ConvexQCQPIpopt ipopt(st);
    return ipopt.solve(p);
  }
  if (st.inner_backend == "filter-sqp") {
    ConvexQCQPFilterSQP sqp(st);
    return sqp.solve(p);
  }
  if (st.inner_backend == "tr-sqp") {
    ConvexQCQPTrustRegionSQP sqp(st);
    return sqp.solve(p);
  }
  if (st.inner_backend == "legacy-ipm") {
    ConvexQCQPInteriorPoint ipm(st);
    return ipm.solve(p);
  }
  if (st.inner_backend == "lazyqp-sparse" && st.dca_single_loop_primal_dual &&
      p.allow_dca_single_loop) {
    // v0.4.28r3p28 graft: lean sparse single-loop transaction for the
    // primary majorant; every other subproblem stays on LazyQP below.
    SparseSingleLoopKernel sparse(st);
    return sparse.solve(p);
  }
  // Default: direct extension of the LazyIPDCA-QP numerical kernel.
  // "ipm" and "native" remain user-facing aliases for this kernel.
  ConvexQCQPLazyQPKernel qp(st);
  return qp.solve(p);
}

inline double final_feasibility_tube_limit(const Settings& st) {
  // Internal globalization guard only.  The final KKT certificate continues
  // to use st.feasibility_tolerance exactly.  The additive term prevents a
  // line search from failing because evaluation noise is a few ulps above the
  // hard tube while the current point is strictly feasible.
  const double roundoff = 8192.0*std::numeric_limits<double>::epsilon();
  return st.feasibility_tolerance + roundoff;
}

inline bool within_final_feasibility_tube(double theta, const Settings& st) {
  return theta <= final_feasibility_tube_limit(st);
}

inline double inequality_bound_violation(const Feasibility& f) {
  return std::max(f.inequality_inf,f.bound_inf);
}

inline double bound_violation(const QCQPProblem& problem, const Vec& x) {
  double violation=0.0;
  for(int j=0;j<problem.n;++j) {
    if(std::isfinite(problem.lb[static_cast<std::size_t>(j)]))
      violation=std::max(
          violation,problem.lb[static_cast<std::size_t>(j)]-
                        x[static_cast<std::size_t>(j)]);
    if(std::isfinite(problem.ub[static_cast<std::size_t>(j)]))
      violation=std::max(
          violation,x[static_cast<std::size_t>(j)]-
                        problem.ub[static_cast<std::size_t>(j)]);
  }
  return std::max(0.0,violation);
}

struct EqualityRetractionTrial {
  bool attempted=false;
  bool success=false;
  Vec x;
  Feasibility feasibility;
  double correction_inf=0.0;
};

EqualityRetractionTrial try_dca_equality_retraction(
    const QCQPProblem& problem, const Vec& base, const Vec& raw_trial,
    const Vec& scaled_step, double equality_funnel, const Settings& st) {
  EqualityRetractionTrial out;
  out.x=raw_trial;
  out.feasibility=feasibility(problem,raw_trial);
  if(!st.dca_equality_retraction || problem.equalities.empty()) return out;
  out.attempted=true;

  Vec polished=retract_equalities(problem,raw_trial,
      std::max(1,st.equality_retraction_iterations),
      st.equality_retraction_tolerance,
      st.equality_retraction_regularization,
      st.fraction_to_boundary);
  for(double v:polished) if(!std::isfinite(v)) return out;
  Vec correction=polished;
  axpy(-1.0,raw_trial,correction);
  out.correction_inf=inf_norm(correction);
  const double step_scale=std::max(inf_norm(scaled_step),
      st.step_tolerance*(1.0+inf_norm(base)));
  const double max_correction=std::max(0.0,
      st.dca_equality_retraction_max_ratio)*step_scale;
  const Feasibility fp=feasibility(problem,polished);
  const double noise=8192.0*std::numeric_limits<double>::epsilon()*
      std::max({1.0,out.feasibility.total_inf,fp.total_inf,equality_funnel});
  const bool correction_ok=std::isfinite(out.correction_inf) &&
      out.correction_inf<=max_correction+noise;
  const bool equality_ok=fp.equality_inf<=equality_funnel+noise;
  const bool side_ok=inequality_bound_violation(fp)<=
      final_feasibility_tube_limit(st);
  const bool improves=fp.equality_inf<=out.feasibility.equality_inf+noise;
  if(correction_ok && equality_ok && side_ok && improves) {
    out.x=std::move(polished);
    out.feasibility=fp;
    out.success=true;
  }
  return out;
}

struct SecondOrderCorrectionResult {
  bool attempted=false;
  bool success=false;
  Vec x;
  Feasibility feasibility;
  int inner_iterations=0;
  int factorizations=0;
  int symbolic_analyses=0;
  double correction_inf=0.0;
  double equality_target_inf=0.0;
  double equality_defect_inf=0.0;
  double inequality_target_violation_inf=0.0;
  double inequality_defect_violation_inf=0.0;
  double inequality_correction_rhs_inf=0.0;
  double normal_correction_inf=0.0;
  double qp_correction_inf=0.0;
  bool iterated_attempted=false;
  bool iterated_success=false;
};

// One-shot Maratos second-order correction for a Phase-II step generated from
// tangent half-spaces.  Historical SQP calls target zero residuals.  A scaled
// Lazy-DCA line-search call instead preserves the affine model at z=x+alpha*d:
//
//   target_h = h(x) + grad h(x)^T (z-x),
//   target_g = g(x) + grad g(x)^T (z-x).
//
// The minimum-norm QP corrects only the nonlinear defects.  For inequalities,
// a positive target already admitted by the outer feasibility tube is retained,
// while negative model slack may be spent:
//
//   h(z) + grad h(x)^T p = target_h,
//   g(z) + grad g(x)^T p <= max(0,target_g).
//
// Thus the correction tends to zero with alpha and does not consume the current
// feasibility residual a second time.
SecondOrderCorrectionResult second_order_correction(
    const QCQPProblem& prob, const Vec& x, const Vec& z,
    const std::set<int>& working_set, const Settings& st,
    bool nonlinear_equality_retraction=true,
    bool iterated_dca_soc=false,
    bool preserve_scaled_constraint_model=false) {
  SecondOrderCorrectionResult out;
  out.attempted=true;
  out.x=z;
  out.feasibility=feasibility(prob,z);
  if (!st.second_order_correction || prob.n<=0) return out;

  ConvexSubproblem q;
  q.nvar=prob.n;
  q.original_dimension=prob.n;
  q.objective.constant=0.0;
  q.objective.linear.assign(prob.n,0.0);
  q.objective.base_hessian.n=prob.n;
  for(int j=0;j<prob.n;++j) q.objective.base_hessian.entries.push_back({j,j,1.0});
  q.equalities.nvar=prob.n;

  // A classical SOC at an infeasible-but-accepted SQP point must remove only
  // the nonlinear defect relative to the SCALED linearized model.  For
  // z=x+s, that model predicts
  //
  //   target_j = h_j(x) + grad h_j(x)^T s.
  //
  // Solving Jp=-h(z) instead (the v0.4.16 behavior) also removes the residual
  // already present at x.  When |h(x)| is near the final feasibility tolerance,
  // that produces a non-vanishing correction as ||s|| -> 0 and destroys line-
  // search continuity.  The historical SQP paths retain zero targets; only the
  // single-route Lazy-DCA calls request defect-only correction.
  Vec equality_targets(prob.equalities.size(),0.0);
  Vec scaled_step=z;
  axpy(-1.0,x,scaled_step);
  for(int j=0;j<(int)prob.equalities.size();++j){
    const auto& h=prob.equalities[j];
    Vec gx=h.gradient(x);
    const double target=preserve_scaled_constraint_model
        ? structural_quadratic_value(h,x)+dot(gx,scaled_step) : 0.0;
    equality_targets[j]=target;
    const double defect=h.value(z)-target;
    out.equality_target_inf=std::max(out.equality_target_inf,std::abs(target));
    out.equality_defect_inf=std::max(out.equality_defect_inf,std::abs(defect));
    q.equalities.rows.push_back(dense_gradient_row(std::move(gx)));
    q.equalities.rhs.push_back(-defect);
  }

  const double base_inf=inf_norm(scaled_step);
  const double ratio_limit=iterated_dca_soc
      ? std::max(st.second_order_correction_max_ratio,
                 st.dca_iterated_soc_max_ratio)
      : st.second_order_correction_max_ratio;
  const double max_corr=ratio_limit*
      std::max(base_inf,st.step_tolerance*(1.0+inf_norm(x)));

  // For a defect-only Lazy-DCA SOC, use the complete lifted equality
  // Jacobian and do not project t=|q| before forming the correction.  The
  // reduced projector used by Phase I is excellent for finding a feasible
  // point, but at an already nearly feasible CORE iterate it can introduce a
  // fixed O(1e-4) correction when tiny lift roundoff is amplified by strongly
  // scaled coupling equations.  A full-space target correction is homogeneous
  // in h(z)-target, so its norm must vanish with alpha.
  if(iterated_dca_soc && preserve_scaled_constraint_model &&
     out.feasibility.equality_inf>st.equality_retraction_tolerance) {
    out.iterated_attempted=true;
    Vec polished=retract_equalities_to_targets_full_space(
        prob,z,equality_targets,
        std::max(1,st.dca_iterated_soc_max_iterations),
        st.equality_retraction_tolerance,
        st.equality_retraction_regularization,
        st.fraction_to_boundary);
    Vec total_correction=polished;
    axpy(-1.0,z,total_correction);
    const double total_inf=inf_norm(total_correction);
    out.normal_correction_inf=total_inf;
    const Feasibility fp=feasibility(prob,polished);
    const double improve_noise=8192.0*std::numeric_limits<double>::epsilon()*
        std::max({1.0,std::abs(fp.total_inf),std::abs(out.feasibility.total_inf)});
    if(std::isfinite(total_inf) && total_inf<=max_corr &&
       fp.total_inf<=out.feasibility.total_inf+improve_noise &&
       within_final_feasibility_tube(fp.total_inf,st)) {
      out.x=std::move(polished);
      out.feasibility=fp;
      out.correction_inf=total_inf;
      out.iterated_success=true;
      out.success=true;
      return out;
    }
  }

  const double near_active=std::max(st.working_set_activation,10.0*st.feasibility_tolerance);
  for(int i=0;i<(int)prob.inequalities.size();++i){
    const auto& g=prob.inequalities[i];
    const double gz=g.value(z);
    Vec gx=g.gradient(x);
    const double target=preserve_scaled_constraint_model
        ? structural_quadratic_value(g,x)+dot(gx,scaled_step) : 0.0;
    const double defect=preserve_scaled_constraint_model ? gz-target : gz;
    // The correction is allowed to retain a positive residual already
    // predicted by the scaled linearized model (up to the outer feasibility
    // tube).  Negative model slack need not be preserved: ordinary
    // inequality feasibility is sufficient.  Therefore enforce
    //
    //   g(z) + grad g(x)^T p <= max(0,target),
    //
    // or, equivalently,
    //
    //   [g(z)-max(0,target)] + grad g(x)^T p <= 0.
    //
    // This is the least restrictive homogeneous correction: at alpha=0 the
    // right-hand side is inactive for a feasible inequality and exactly zero
    // for a tolerance-level positive residual.  Using `defect` alone would
    // unnecessarily preserve all negative target slack and regressed DISCS;
    // using `g(z)` would erase a pre-existing positive residual and recreate
    // the CORE2 correction floor.
    const double correction_constant=preserve_scaled_constraint_model
        ? gz-std::max(0.0,target) : gz;
    out.inequality_target_violation_inf=std::max(
        out.inequality_target_violation_inf,std::max(0.0,target));
    out.inequality_defect_violation_inf=std::max(
        out.inequality_defect_violation_inf,std::max(0.0,defect));
    out.inequality_correction_rhs_inf=std::max(
        out.inequality_correction_rhs_inf,std::max(0.0,correction_constant));

    if(!working_set.count(i) && correction_constant<=0.0) continue;
    if(working_set.count(i) && gz < -near_active &&
       target < -near_active && correction_constant<=0.0) continue;
    ConvexQuadraticModel c;
    c.constant=correction_constant;
    c.linear=std::move(gx);
    c.base_hessian.n=prob.n;
    q.inequalities.push_back(std::move(c));
  }
  for(int j=0;j<prob.n;++j){
    if(std::isfinite(prob.ub[j]))
      q.inequalities.push_back(linear_inequality(prob.n,j,+1.0,z[j]-prob.ub[j]));
    if(std::isfinite(prob.lb[j]))
      q.inequalities.push_back(linear_inequality(prob.n,j,-1.0,prob.lb[j]-z[j]));
  }

  InnerResult ir=solve_inner(q,st);
  out.inner_iterations=ir.iterations;
  out.factorizations=ir.factorizations;
  out.symbolic_analyses=ir.symbolic_analyses;
  if(!ir.success || (int)ir.y.size()<prob.n) return out;
  Vec pcor(ir.y.begin(),ir.y.begin()+prob.n);
  out.qp_correction_inf=inf_norm(pcor);
  out.correction_inf=out.qp_correction_inf;
  if(!std::isfinite(out.correction_inf) || out.correction_inf>max_corr) return out;

  Vec cand=z; axpy(1.0,pcor,cand);
  Feasibility fc=feasibility(prob,cand);

  // The primary equality model remains the classical SQP linearization.  This
  // optional refinement is an iterated normal/SOC correction applied only
  // after that step, then guarded by its total size and by the original-QCQP
  // line search.  It does not replace h(x_k)+grad h(x_k)^T d=0 and cannot
  // switch the outer algorithm.
  if(iterated_dca_soc && !within_final_feasibility_tube(fc.total_inf,st)) {
    out.iterated_attempted=true;
    Vec polished=retract_equalities_to_targets_full_space(
        prob,cand,equality_targets,
        std::max(1,st.dca_iterated_soc_max_iterations),
        st.equality_retraction_tolerance,
        st.equality_retraction_regularization,
        st.fraction_to_boundary);
    Vec total_correction=polished;
    axpy(-1.0,z,total_correction);
    const double total_inf=inf_norm(total_correction);
    out.normal_correction_inf=std::max(out.normal_correction_inf,total_inf);
    const Feasibility fp=feasibility(prob,polished);
    const double improve_noise=8192.0*std::numeric_limits<double>::epsilon()*
        std::max({1.0,std::abs(fp.total_inf),std::abs(fc.total_inf)});
    if(std::isfinite(total_inf) && total_inf<=max_corr &&
       fp.total_inf<=fc.total_inf+improve_noise) {
      cand=std::move(polished);
      fc=fp;
      out.correction_inf=total_inf;
      out.iterated_success=within_final_feasibility_tube(fc.total_inf,st);
    }
  } else if(nonlinear_equality_retraction &&
            fc.total_inf>st.feasibility_tolerance) {
    cand=retract_equalities(prob,cand,std::min(2,st.equality_retraction_iterations),
                            st.equality_retraction_tolerance,
                            st.equality_retraction_regularization,
                            st.fraction_to_boundary);
    fc=feasibility(prob,cand);
  }
  if(fc.total_inf >= out.feasibility.total_inf) return out;
  out.x=std::move(cand);
  out.feasibility=fc;
  // v0.2.13 treated any residual improvement as a successful SOC.  On A4X12
  // this produced theta~1e-4 at alpha=1, still far outside the 1e-7 feasible
  // tube, and then disabled the proven feasibility repair.  A SOC is useful to
  // the feasible-axis line search only when it actually restores feasibility.
  out.success = within_final_feasibility_tube(fc.total_inf,st);
  return out;
}

inline double objective_roundoff_noise(double f) {
  return 64.0*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(f));
}

inline bool armijo_objective_ok(double fk, double ft, double alpha,
                                double predicted_decrease, const Settings& st) {
  if (!(predicted_decrease>0.0) || !std::isfinite(predicted_decrease)) return false;
  return ft <= fk - st.line_search_armijo*alpha*predicted_decrease
               + objective_roundoff_noise(fk);
}


enum class ActiveKind { Equality, Inequality, LowerBound, UpperBound };
struct ActiveConstraint {
  ActiveKind kind;
  int index;
  bool nonnegative_multiplier;
};

std::vector<ActiveConstraint> active_constraints(const QCQPProblem& p, const Vec& x,
                                                 const Settings& st) {
  std::vector<ActiveConstraint> a;
  for (int j=0;j<(int)p.equalities.size();++j)
    a.push_back({ActiveKind::Equality,j,false});
  const double atol=std::max(1e-5,100.0*st.feasibility_tolerance);
  for (int i=0;i<(int)p.inequalities.size();++i)
    if (structural_quadratic_value(p.inequalities[i],x)>=-atol)
      a.push_back({ActiveKind::Inequality,i,true});
  const double btol=std::max(1e-7,100.0*st.feasibility_tolerance);
  for(int j=0;j<p.n;++j){
    if(std::isfinite(p.lb[j]) && x[j]<=p.lb[j]+btol)
      a.push_back({ActiveKind::LowerBound,j,true});
    if(std::isfinite(p.ub[j]) && x[j]>=p.ub[j]-btol)
      a.push_back({ActiveKind::UpperBound,j,true});
  }
  return a;
}

Vec active_gradient(const QCQPProblem& p,const Vec& x,const ActiveConstraint& c){
  if(c.kind==ActiveKind::Equality) return p.equalities[c.index].gradient(x);
  if(c.kind==ActiveKind::Inequality) return p.inequalities[c.index].gradient(x);
  Vec g(p.n,0.0);
  g[c.index]=(c.kind==ActiveKind::LowerBound)?-1.0:+1.0;
  return g;
}

double active_value(const QCQPProblem& p,const Vec& x,const ActiveConstraint& c){
  if(c.kind==ActiveKind::Equality)
    return structural_quadratic_value(p.equalities[c.index],x);
  if(c.kind==ActiveKind::Inequality)
    return structural_quadratic_value(p.inequalities[c.index],x);
  if(c.kind==ActiveKind::LowerBound) return p.lb[c.index]-x[c.index];
  return x[c.index]-p.ub[c.index];
}


// Certificate-only active-set NNLS.  This deliberately does not replace the
// historical cyclic multiplier fit used by the v0.2.14 SQP trajectory.
// It solves the same reduced convex least-squares problem with a finite
// Lawson-Hanson-style passive-set method so nearly dependent active normals do
// not require thousands of coordinate sweeps merely to decide termination.
ConstrainedMultiplierLS termination_multiplier_ls_active_set(
    const Vec& g,
    const std::vector<Vec>& free_grad,
    const std::vector<Vec>& nonneg_grad,
    const Settings& st,
    double regularization,
    FreeMultiplierProjector* persistent_free_projector=nullptr) {
  ConstrainedMultiplierLS out;
  const int nf=static_cast<int>(free_grad.size());
  const int q=static_cast<int>(nonneg_grad.size());
  out.free_duals.assign(nf,0.0);
  out.nonnegative_duals.assign(q,0.0);

  const double reg=std::max(1e-14,regularization);
  const double pg_tol=std::max(1e-11,0.01*st.stationarity_tolerance);
  FreeMultiplierProjector local_projector(st);
  FreeMultiplierProjector* projector=persistent_free_projector
      ? persistent_free_projector : &local_projector;
  if(nf>0 && !projector->factor_rows(free_grad,reg)) {
    out.converged=false;
    return out;
  }

  bool free_projection_ok=true;
  auto eliminate_free=[&](const Vec& v,Vec* lambda_out)->Vec {
    if(nf==0) {
      if(lambda_out) lambda_out->clear();
      return v;
    }
    const auto projected=projector->project(v);
    if(!projected.success) {
      free_projection_ok=false;
      if(lambda_out) lambda_out->assign(nf,0.0);
      return v;
    }
    if(lambda_out) *lambda_out=projected.multipliers;
    return projected.residual;
  };

  if(q==0) {
    Vec lam;
    const Vec r=eliminate_free(g,&lam);
    out.free_duals=std::move(lam);
    out.stationarity=inf_norm(r);
    out.projected_gradient_inf=0.0;
    out.converged=free_projection_ok && std::isfinite(out.stationarity);
    return out;
  }
  const Vec pg=eliminate_free(g,nullptr);
  if(!free_projection_ok) return out;

  std::vector<Vec> PA(q);
  for(int i=0;i<q;++i) {
    PA[i]=eliminate_free(nonneg_grad[i],nullptr);
    if(!free_projection_ok) return out;
  }
  std::vector<double> H(static_cast<std::size_t>(q)*q,0.0);
  Vec c(q,0.0);
  for(int i=0;i<q;++i) {
    c[i]=dot(nonneg_grad[i],pg);
    for(int j=0;j<=i;++j) {
      double v=dot(nonneg_grad[i],PA[j]);
      if(i==j) v+=reg;
      H[j+i*q]=v; H[i+j*q]=v;
    }
  }

  Vec mu(q,0.0),grad=c;
  std::vector<unsigned char> passive(q,0);
  const int max_outer=std::max(100,20*q+100);
  const int max_inner=std::max(100,20*q+100);
  int iterations=0;

  auto recompute_grad=[&]() {
    grad=c;
    for(int i=0;i<q;++i) if(mu[i]!=0.0)
      for(int j=0;j<q;++j) grad[j]+=H[j+i*q]*mu[i];
  };
  auto projected_gradient_inf=[&]() {
    double v=0.0;
    for(int i=0;i<q;++i) {
      const double pgi=(mu[i]>1e-14)?grad[i]:std::min(0.0,grad[i]);
      v=std::max(v,std::abs(pgi));
    }
    return v;
  };

  for(int outer=0;outer<max_outer;++outer) {
    recompute_grad();
    const double pginf=projected_gradient_inf();
    out.projected_gradient_inf=pginf;
    if(pginf<=pg_tol) { out.converged=true; break; }

    int enter=-1;
    double most_negative=-pg_tol;
    for(int i=0;i<q;++i) {
      if(passive[i]) continue;
      if(grad[i]<most_negative) { most_negative=grad[i]; enter=i; }
    }
    if(enter<0) break;
    passive[enter]=1;

    for(int inner=0;inner<max_inner;++inner) {
      ++iterations;
      std::vector<int> P;
      for(int i=0;i<q;++i) if(passive[i]) P.push_back(i);
      if(P.empty()) break;

      const int np=static_cast<int>(P.size());
      std::vector<double> Hp(static_cast<std::size_t>(np)*np,0.0);
      Vec rhs(np,0.0),zp;
      for(int a=0;a<np;++a) {
        rhs[a]=-c[P[a]];
        for(int b=0;b<np;++b) Hp[b+a*np]=H[P[b]+P[a]*q];
      }
      if(!solve_dense(Hp,rhs,zp)) {
        out.sweeps=iterations;
        out.projected_gradient_inf=projected_gradient_inf();
        out.converged=false;
        return out;
      }

      Vec z(q,0.0);
      bool all_positive=true;
      for(int a=0;a<np;++a) {
        z[P[a]]=zp[a];
        if(zp[a]<=1e-14) all_positive=false;
      }
      if(all_positive) { mu=std::move(z); break; }

      double alpha=1.0;
      bool have_blocker=false;
      for(int a=0;a<np;++a) {
        const int i=P[a];
        if(z[i]<=1e-14) {
          const double denom=mu[i]-z[i];
          if(denom>0.0) {
            alpha=std::min(alpha,mu[i]/denom);
            have_blocker=true;
          } else if(mu[i]<=1e-14) {
            alpha=0.0;
            have_blocker=true;
          }
        }
      }
      if(!have_blocker) alpha=1.0;
      for(int i=0;i<q;++i) mu[i]+=alpha*(z[i]-mu[i]);
      for(int i=0;i<q;++i) {
        if(passive[i] && mu[i]<=1e-12) {
          mu[i]=0.0;
          passive[i]=0;
        }
      }
    }
  }

  recompute_grad();
  out.projected_gradient_inf=projected_gradient_inf();
  if(out.projected_gradient_inf<=pg_tol) out.converged=true;
  out.sweeps=iterations;

  Vec v=g;
  for(int i=0;i<q;++i) if(mu[i]!=0.0) axpy(mu[i],nonneg_grad[i],v);
  Vec lam;
  Vec r=eliminate_free(v,&lam);
  if(!free_projection_ok) {
    out.converged=false;
    out.stationarity=std::numeric_limits<double>::infinity();
    return out;
  }
  out.free_duals=std::move(lam);
  out.nonnegative_duals=std::move(mu);
  out.stationarity=inf_norm(r);
  return out;
}

struct ActiveMultiplierEstimate {
  std::vector<ActiveConstraint> active;
  Vec y;
  double stationarity=std::numeric_limits<double>::infinity();
  int multiplier_sweeps=0;
  double multiplier_projected_gradient_inf=std::numeric_limits<double>::infinity();
  bool multiplier_solve_converged=false;
};

ActiveMultiplierEstimate estimate_active_multipliers_legacy(const QCQPProblem& p,const Vec& x,
                                                      const Settings& st){
  ActiveMultiplierEstimate out;
  out.active=active_constraints(p,x,st);
  std::vector<Vec> free_grad, nonneg_grad;
  std::vector<int> free_pos, nonneg_pos;
  for(int i=0;i<(int)out.active.size();++i){
    Vec g=active_gradient(p,x,out.active[i]);
    if(out.active[i].nonnegative_multiplier){nonneg_pos.push_back(i);nonneg_grad.push_back(std::move(g));}
    else {free_pos.push_back(i);free_grad.push_back(std::move(g));}
  }
  const auto ls=constrained_multiplier_ls(p.objective.gradient(x),free_grad,nonneg_grad,st);
  out.y.assign(out.active.size(),0.0);
  for(int k=0;k<(int)free_pos.size() && k<(int)ls.free_duals.size();++k) out.y[free_pos[k]]=ls.free_duals[k];
  for(int k=0;k<(int)nonneg_pos.size() && k<(int)ls.nonnegative_duals.size();++k) out.y[nonneg_pos[k]]=ls.nonnegative_duals[k];
  out.multiplier_sweeps=ls.sweeps;
  out.multiplier_projected_gradient_inf=ls.projected_gradient_inf;
  out.multiplier_solve_converged=ls.converged;
  // A nonconverged multiplier fit is diagnostic only; never expose it as a
  // KKT convergence certificate.
  out.stationarity=ls.converged ? ls.stationarity : std::numeric_limits<double>::infinity();
  return out;
}


MultiplierEstimate multiplier_estimate_for_termination(
    const QCQPProblem& p,const Vec& x,
    const Settings& st,double regularization,
    ActiveMultiplierEstimate* fit_out=nullptr,
    FreeMultiplierProjector* persistent_free_projector=nullptr){
  // Certificate-only multiplier refit with exact Ipopt make_parameter geometry:
  // fixed primal coordinates are removed from the stationarity least-squares
  // problem, and their bound multipliers do not exist in the certificate.
  std::vector<bool> fixed(p.n,false);
  for(int j=0;j<p.n;++j)
    fixed[j]=std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
             std::abs(p.ub[j]-p.lb[j])<=st.fixed_bound_tolerance;

  std::vector<ActiveConstraint> active;
  for(int j=0;j<(int)p.equalities.size();++j)
    active.push_back({ActiveKind::Equality,j,false});
  const double atol=std::max(1e-5,100.0*st.feasibility_tolerance);
  for(int i=0;i<(int)p.inequalities.size();++i)
    if(structural_quadratic_value(p.inequalities[i],x)>=-atol)
      active.push_back({ActiveKind::Inequality,i,true});
  const double btol=std::max(1e-7,100.0*st.feasibility_tolerance);
  for(int j=0;j<p.n;++j){
    if(fixed[j]) continue;
    if(std::isfinite(p.lb[j]) && x[j]<=p.lb[j]+btol)
      active.push_back({ActiveKind::LowerBound,j,true});
    if(std::isfinite(p.ub[j]) && x[j]>=p.ub[j]-btol)
      active.push_back({ActiveKind::UpperBound,j,true});
  }

  Vec objg=p.objective.gradient(x);
  for(int j=0;j<p.n;++j) if(fixed[j]) objg[j]=0.0;
  std::vector<Vec> free_grad,nonneg_grad;
  std::vector<int> free_pos,nonneg_pos;
  for(int k=0;k<(int)active.size();++k){
    Vec g=active_gradient(p,x,active[k]);
    for(int j=0;j<p.n;++j) if(fixed[j]) g[j]=0.0;
    if(active[k].nonnegative_multiplier){
      nonneg_pos.push_back(k); nonneg_grad.push_back(std::move(g));
    } else {
      free_pos.push_back(k); free_grad.push_back(std::move(g));
    }
  }
  const auto ls=termination_multiplier_ls_active_set(
      objg,free_grad,nonneg_grad,st,regularization,
      persistent_free_projector);

  if(fit_out){
    fit_out->active=active;
    fit_out->y.assign(active.size(),0.0);
    for(int k=0;k<(int)free_pos.size() && k<(int)ls.free_duals.size();++k)
      fit_out->y[free_pos[k]]=ls.free_duals[k];
    for(int k=0;k<(int)nonneg_pos.size() && k<(int)ls.nonnegative_duals.size();++k)
      fit_out->y[nonneg_pos[k]]=ls.nonnegative_duals[k];
    fit_out->stationarity=ls.converged?ls.stationarity:std::numeric_limits<double>::infinity();
    fit_out->multiplier_sweeps=ls.sweeps;
    fit_out->multiplier_projected_gradient_inf=ls.projected_gradient_inf;
    fit_out->multiplier_solve_converged=ls.converged;
  }

  MultiplierEstimate out;
  out.equality.assign(p.equalities.size(),0.0);
  out.inequality.assign(p.inequalities.size(),0.0);
  out.lower_bound.assign(p.n,0.0);
  out.upper_bound.assign(p.n,0.0);
  out.valid=ls.converged;
  if(!out.valid) return out;
  Vec y(active.size(),0.0);
  for(int k=0;k<(int)free_pos.size() && k<(int)ls.free_duals.size();++k) y[free_pos[k]]=ls.free_duals[k];
  for(int k=0;k<(int)nonneg_pos.size() && k<(int)ls.nonnegative_duals.size();++k) y[nonneg_pos[k]]=ls.nonnegative_duals[k];
  for(int k=0;k<(int)active.size();++k){
    const auto& c=active[k]; const double v=y[k];
    if(c.kind==ActiveKind::Equality) out.equality[c.index]=v;
    else if(c.kind==ActiveKind::Inequality) out.inequality[c.index]=std::max(0.0,v);
    else if(c.kind==ActiveKind::LowerBound) out.lower_bound[c.index]=std::max(0.0,v);
    else if(c.kind==ActiveKind::UpperBound) out.upper_bound[c.index]=std::max(0.0,v);
  }
  return out;
}

void store_termination_metrics(const QCQPIpoptTermination& t, Result& result) {
  result.dual_infeasibility_inf=t.stop.dual_infeasibility;
  result.complementarity_inf=t.stop.complementarity;
  result.scaled_nlp_error=t.stop.overall_nlp_error;
  result.dual_error_scaling=t.stop.dual_scaling;
  result.complementarity_error_scaling=t.stop.complementarity_scaling;
  result.constraint_multiplier_l1=t.stop.y_l1;
  result.bound_slack_multiplier_l1=t.stop.z_l1;
  result.ipopt_dual_multiplier_dimension=t.stop.dual_multiplier_dimension;
  result.ipopt_complementarity_multiplier_dimension=t.stop.complementarity_multiplier_dimension;
  result.ipopt_y_c_dimension=t.y_c_dimension;
  result.ipopt_y_d_dimension=t.y_d_dimension;
  result.ipopt_z_L_dimension=t.z_L_dimension;
  result.ipopt_z_U_dimension=t.z_U_dimension;
  result.ipopt_v_L_dimension=t.v_L_dimension;
  result.ipopt_v_U_dimension=t.v_U_dimension;
  result.equality_multiplier_inf=t.equality_multiplier_inf;
  result.inequality_multiplier_inf=t.inequality_multiplier_inf;
  result.bound_multiplier_inf=t.bound_multiplier_inf;
}

QCQPIpoptTermination fresh_termination_certificate(
    const QCQPProblem& p,const Vec& x,const Settings& st,Result& result,
    FreeMultiplierProjector* persistent_free_projector=nullptr){
  // Certificate-only regularization ladder.  The first candidate preserves
  // the historical fit; lower regularizations are tried only when the actual
  // original-QCQP certificate does not pass.  The candidate with the smallest
  // Ipopt-style NLP error is retained.  No stopping tolerance is relaxed and
  // the SQP trajectory is unchanged.
  const double reg0=std::max(1e-14,st.multiplier_regularization);
  std::vector<double> regs;
  auto add_reg=[&](double r){
    r=std::max(1e-14,r);
    for(double q:regs)
      if(std::abs(q-r)<=1e-12*std::max(std::abs(q),std::abs(r))) return;
    regs.push_back(r);
  };
  add_reg(reg0);
  add_reg(1e-2*reg0);
  add_reg(1e-4*reg0);

  ActiveMultiplierEstimate best_fit;
  QCQPIpoptTermination best_t;
  bool have_best=false;
  double best_reg=std::numeric_limits<double>::infinity();
  int attempts=0;

  for(double reg:regs){
    ActiveMultiplierEstimate fit;
    const auto mult=multiplier_estimate_for_termination(
        p,x,st,reg,&fit,persistent_free_projector);
    ++attempts;
    if(!fit.multiplier_solve_converged || !mult.valid) continue;
    const auto t=qcqp_ipopt_termination(p,x,mult,st);
    const bool better=!have_best ||
        (t.stop.converged && !best_t.stop.converged) ||
        (t.stop.converged==best_t.stop.converged &&
         t.stop.overall_nlp_error<best_t.stop.overall_nlp_error);
    if(better){
      best_fit=fit;
      best_t=t;
      result.certificate_equality_multipliers=mult.equality;
      result.certificate_inequality_multipliers=mult.inequality;
      result.certificate_lower_multipliers=mult.lower_bound;
      result.certificate_upper_multipliers=mult.upper_bound;
      best_reg=reg;
      have_best=true;
    }
    if(t.stop.converged) break;
  }

  result.termination_multiplier_refinement_attempts=attempts;
  result.termination_multiplier_regularization=best_reg;
  result.termination_multiplier_source=(attempts>1)?"fit-refined":"fit";
  result.termination_multiplier_fit_converged=have_best && best_fit.multiplier_solve_converged;
  result.termination_multiplier_fit_sweeps=have_best?best_fit.multiplier_sweeps:0;
  result.termination_multiplier_fit_projected_gradient_inf=have_best
      ?best_fit.multiplier_projected_gradient_inf
      :std::numeric_limits<double>::infinity();
  result.termination_multiplier_fit_stationarity_inf=have_best
      ?best_fit.stationarity
      :std::numeric_limits<double>::infinity();

  if(!have_best){
    result.ipopt_desired_convergence=false;
    return QCQPIpoptTermination{};
  }
  store_termination_metrics(best_t,result);
  result.ipopt_desired_convergence=best_t.stop.converged;
  return best_t;
}

QCQPIpoptTermination mapped_termination_certificate(const QCQPProblem& p,const Vec& x,
                                                     const MultiplierEstimate& mult,
                                                     const Settings& st,
                                                     Result& result){
  result.termination_multiplier_source="inner-mapped";
  result.termination_multiplier_fit_converged=false;
  result.termination_multiplier_fit_sweeps=0;
  result.termination_multiplier_fit_projected_gradient_inf=0.0;
  result.termination_multiplier_fit_stationarity_inf=0.0;
  result.termination_multiplier_refinement_attempts=0;
  result.termination_multiplier_regularization=std::numeric_limits<double>::infinity();
  const auto t=qcqp_ipopt_termination(p,x,mult,st);
  store_termination_metrics(t,result);
  result.certificate_equality_multipliers=mult.equality;
  result.certificate_inequality_multipliers=mult.inequality;
  result.certificate_lower_multipliers=mult.lower_bound;
  result.certificate_upper_multipliers=mult.upper_bound;
  result.ipopt_desired_convergence=mult.valid && t.stop.converged;
  return t;
}

bool mapped_certificate_passes(const QCQPProblem& p,const Vec& x,
                               const MultiplierEstimate& mult,const Settings& st,
                               Result& result){
  const auto t=mapped_termination_certificate(p,x,mult,st,result);
  if(st.verbose){
    std::cout << "[KKT-STOP-MAPPED] dual=" << std::scientific << t.stop.dual_infeasibility
              << " comp=" << t.stop.complementarity
              << " nlp=" << t.stop.overall_nlp_error
              << " sd=" << t.stop.dual_scaling
              << " sc=" << t.stop.complementarity_scaling
              << " pass=" << (result.ipopt_desired_convergence?1:0)
              << "\n" << std::defaultfloat;
  }
  return result.ipopt_desired_convergence;
}

bool fresh_certificate_passes(
    const QCQPProblem& p,const Vec& x,const Settings& st,Result& result,
    FreeMultiplierProjector* persistent_free_projector=nullptr){
  const auto t=fresh_termination_certificate(
      p,x,st,result,persistent_free_projector);
  if(st.verbose){
    std::cout << "[KKT-STOP-CHECK] fit="
              << (result.termination_multiplier_fit_converged?"ok":"fail")
              << " dual=" << std::scientific << t.stop.dual_infeasibility
              << " comp=" << t.stop.complementarity
              << " nlp=" << t.stop.overall_nlp_error
              << " sd=" << t.stop.dual_scaling
              << " sc=" << t.stop.complementarity_scaling
              << " reg=" << result.termination_multiplier_regularization
              << " attempts=" << result.termination_multiplier_refinement_attempts
              << " pass=" << (result.ipopt_desired_convergence?1:0)
              << "\n" << std::defaultfloat;
  }
  return result.ipopt_desired_convergence;
}



struct TangentDiagnostics {
  int jacobian_rank = 0;
  int tangent_dimension = 0;
  int reduced_negative_eigenvalues = 0;
  double reduced_min_eigenvalue = 0.0;
};

// Modified Gram-Schmidt rank and explicit orthonormal null-space basis.  This
// is used only by the local terminal phase/diagnostics.  For large problems
// the eigenspectrum diagnostic is skipped, but the KKT step itself remains
// sparse/backend-driven.
std::vector<Vec> orthonormal_row_basis(const std::vector<Vec>& rows, int n) {
  std::vector<Vec> basis;
  basis.reserve(std::min<int>(n, rows.size()));
  for (const Vec& gin : rows) {
    Vec v = gin;
    const double n0 = norm2(v);
    if (n0 <= 1e-18) continue;
    for (const auto& q : basis) axpy(-dot(q,v),q,v);
    for (const auto& q : basis) axpy(-dot(q,v),q,v);
    const double nr = norm2(v);
    if (nr <= 1e-9*std::max(1.0,n0)) continue;
    for (double& z : v) z /= nr;
    basis.push_back(std::move(v));
    if ((int)basis.size() == n) break;
  }
  return basis;
}

std::vector<Vec> null_basis_from_rows(const std::vector<Vec>& row_basis, int n) {
  std::vector<Vec> Z;
  Z.reserve(std::max(0,n-(int)row_basis.size()));
  for (int j=0;j<n;++j) {
    Vec v(n,0.0); v[j]=1.0;
    for (const auto& q:row_basis) axpy(-dot(q,v),q,v);
    for (const auto& z:Z) axpy(-dot(z,v),z,v);
    // A second pass is important when the active Jacobian is nearly rank
    // deficient, exactly the regime seen on A4X12.
    for (const auto& q:row_basis) axpy(-dot(q,v),q,v);
    for (const auto& z:Z) axpy(-dot(z,v),z,v);
    const double nr=norm2(v);
    if (nr<=1e-9) continue;
    for(double& a:v) a/=nr;
    Z.push_back(std::move(v));
  }
  return Z;
}

TangentDiagnostics tangent_diagnostics(const SymMatrix& B,
                                       const std::vector<Vec>& active_rows,
                                       const Settings& st) {
  TangentDiagnostics out;
  const auto R=orthonormal_row_basis(active_rows,B.n);
  out.jacobian_rank=(int)R.size();
  out.tangent_dimension=std::max(0,B.n-out.jacobian_rank);
  if (out.tangent_dimension==0) return out;
  // Avoid a cubic dense diagnostic on large problems.  The terminal KKT step
  // still works there; only the reduced eigenvalue counters are omitted.
  if (B.n > std::max(512,st.exact_eigen_threshold)) return out;
  const auto Z=null_basis_from_rows(R,B.n);
  const int t=(int)Z.size();
  out.tangent_dimension=t;
  if(t==0) return out;
  std::vector<double>A((std::size_t)t*t,0.0);
  for(int j=0;j<t;++j){
    const Vec bz=B.apply(Z[j]);
    for(int i=0;i<=j;++i){
      const double v=dot(Z[i],bz);
      A[i+j*t]=v; A[j+i*t]=v;
    }
  }
  Vec w(t,0.0); const char jobz='N',uplo='U'; int info=0,lwork=-1; double wk=0.0;
  dsyev_(&jobz,&uplo,&t,A.data(),&t,w.data(),&wk,&lwork,&info);
  if(info!=0) return out;
  lwork=std::max(1,(int)wk); Vec work(lwork,0.0);
  dsyev_(&jobz,&uplo,&t,A.data(),&t,w.data(),work.data(),&lwork,&info);
  if(info!=0) return out;
  out.reduced_min_eigenvalue=w.front();
  const double tol=std::max(st.curvature_tolerance,1e-10);
  for(double e:w) if(e < -tol) ++out.reduced_negative_eigenvalues;
  return out;
}

struct QPDualActiveSet {
  std::vector<ActiveConstraint> active;
  Vec y;
  std::vector<Vec> rows;
  TangentDiagnostics tangent;
  int fixed_variables = 0;
  int free_variables = 0;
  double multiplier_cutoff = 0.0;
  double multiplier_max = 0.0;
  std::vector<long long> signature;
};

long long active_signature_key(const ActiveConstraint& c) {
  return (static_cast<long long>(static_cast<int>(c.kind)) << 32) |
         static_cast<unsigned int>(c.index);
}

QPDualActiveSet qpdual_active_set(const QCQPProblem& p, const Vec& x,
                                  const MultiplierEstimate& mult,
                                  const Settings& st) {
  QPDualActiveSet out;
  std::vector<unsigned char> fixed(static_cast<std::size_t>(p.n), 0);
  for (int j = 0; j < p.n; ++j) {
    fixed[static_cast<std::size_t>(j)] =
        std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
        std::abs(p.ub[j] - p.lb[j]) <= st.fixed_bound_tolerance;
    if (fixed[static_cast<std::size_t>(j)]) ++out.fixed_variables;
  }
  out.free_variables = p.n - out.fixed_variables;
  std::vector<Vec> basis;
  basis.reserve(static_cast<std::size_t>(out.free_variables));

  auto independent_add = [&](const ActiveConstraint& c, double y, const Vec& gin)->bool {
    Vec v=gin;
    // Fixed primal coordinates are parameters in the original-QCQP
    // certificate.  Test independence in exactly that reduced space; keeping
    // their components here can select a row which becomes dependent after
    // the terminal Newton system eliminates the fixed variables.
    for (int j = 0; j < p.n; ++j)
      if (fixed[static_cast<std::size_t>(j)]) v[static_cast<std::size_t>(j)] = 0.0;
    const double n0=norm2(v);
    if(n0<=1e-18) return false;
    for(const auto& q:basis) axpy(-dot(q,v),q,v);
    for(const auto& q:basis) axpy(-dot(q,v),q,v);
    const double nr=norm2(v);
    if(nr<=1e-9*std::max(1.0,n0)) return false;
    for(double& z:v) z/=nr;
    basis.push_back(std::move(v));
    out.active.push_back(c);
    out.y.push_back(y);
    out.rows.push_back(gin);
    out.signature.push_back(active_signature_key(c));
    return true;
  };

  // Original equalities always define the local manifold.  Redundant normals
  // are omitted from the reduced Newton equations but their duals remain in
  // the full Lagrangian gradient/Hessian used below.
  for(int j=0;j<(int)p.equalities.size();++j){
    const double y=j<(int)mult.equality.size()?mult.equality[j]:0.0;
    independent_add({ActiveKind::Equality,j,false},y,p.equalities[j].gradient(x));
  }

  // Barrier-aware active-set filter.  The native Mehrotra IPM returns tiny
  // strictly-positive duals until mu is essentially zero.  Treating every one
  // of those as an active manifold constraint was the v0.2.10 A4X12 failure:
  // jacrank jumped from ~83 to 118--121 and the dual Newton pivot exploded.
  double mumax=0.0;
  for(double mu:mult.inequality) mumax=std::max(mumax,std::max(0.0,mu));
  out.multiplier_max=mumax;
  const double mutol=std::max(
      std::max(st.terminal_kkt_multiplier_tolerance,1e-14),
      st.terminal_kkt_multiplier_relative_tolerance*std::max(1.0,mumax));
  out.multiplier_cutoff=mutol;

  std::vector<std::pair<double,int>> icand;
  for(int i=0;i<(int)p.inequalities.size();++i){
    const double mu=i<(int)mult.inequality.size()?std::max(0.0,mult.inequality[i]):0.0;
    if(mu<=mutol) continue;
    if(structural_quadratic_value(p.inequalities[i],x)<
       -st.terminal_kkt_active_slack) continue;
    icand.push_back({mu,i});
  }
  std::sort(icand.begin(),icand.end(),[](const auto&a,const auto&b){return a.first>b.first;});
  for(const auto& [mu,i]:icand){
    if((int)basis.size()>=p.n) break;
    independent_add({ActiveKind::Inequality,i,true},mu,p.inequalities[i].gradient(x));
  }

  // Bound multipliers are inferred from the residual after original QP duals.
  Vec r=p.objective.gradient(x);
  for(int j=0;j<(int)p.equalities.size() && j<(int)mult.equality.size();++j)
    axpy(mult.equality[j],p.equalities[j].gradient(x),r);
  for(int i=0;i<(int)p.inequalities.size() && i<(int)mult.inequality.size();++i)
    if(mult.inequality[i]!=0.0) axpy(mult.inequality[i],p.inequalities[i].gradient(x),r);
  const double btol=std::max(1e-8,10.0*st.feasibility_tolerance);
  std::vector<std::pair<double,ActiveConstraint>> bcand;
  for(int j=0;j<p.n;++j){
    // Ipopt's make_parameter treatment, used by the final certificate,
    // removes fixed variables and their bound multipliers.  The terminal
    // system must not reintroduce them as sign-filtered active bounds.
    if (fixed[static_cast<std::size_t>(j)]) continue;
    if(std::isfinite(p.lb[j]) && x[j]<=p.lb[j]+btol){
      const double mu=std::max(0.0,r[j]);
      if(mu>mutol) bcand.push_back({mu,{ActiveKind::LowerBound,j,true}});
    }
    if(std::isfinite(p.ub[j]) && x[j]>=p.ub[j]-btol){
      const double mu=std::max(0.0,-r[j]);
      if(mu>mutol) bcand.push_back({mu,{ActiveKind::UpperBound,j,true}});
    }
  }
  std::sort(bcand.begin(),bcand.end(),[](const auto&a,const auto&b){return a.first>b.first;});
  for(const auto& [mu,c]:bcand){
    if((int)basis.size()>=p.n) break;
    Vec g(p.n,0.0); g[c.index]=(c.kind==ActiveKind::LowerBound)?-1.0:+1.0;
    independent_add(c,mu,g);
  }

  std::sort(out.signature.begin(),out.signature.end());
  const SymMatrix B=lagrangian_hessian(p,mult);
  // Tangent diagnostics still account for fixed coordinates, but those rows
  // are diagnostic constraints only and never become Newton dual unknowns.
  std::vector<Vec> diagnostic_rows = out.rows;
  diagnostic_rows.reserve(out.rows.size() +
                          static_cast<std::size_t>(out.fixed_variables));
  for (int j = 0; j < p.n; ++j) {
    if (!fixed[static_cast<std::size_t>(j)]) continue;
    Vec g(static_cast<std::size_t>(p.n), 0.0);
    g[static_cast<std::size_t>(j)] = 1.0;
    diagnostic_rows.push_back(std::move(g));
  }
  out.tangent=tangent_diagnostics(B,diagnostic_rows,st);
  return out;
}

Vec qpdual_lagrangian_gradient(const QCQPProblem& p, const Vec& x,
                               const MultiplierEstimate& mult) {
  Vec g=p.objective.gradient(x);
  for(int j=0;j<(int)p.equalities.size() && j<(int)mult.equality.size();++j)
    if(mult.equality[j]!=0.0) axpy(mult.equality[j],p.equalities[j].gradient(x),g);
  for(int i=0;i<(int)p.inequalities.size() && i<(int)mult.inequality.size();++i)
    if(mult.inequality[i]!=0.0) axpy(mult.inequality[i],p.inequalities[i].gradient(x),g);
  return g;
}

// Materialize canonical bound multipliers from the signed residual of the
// non-bound Lagrangian terms.  Projected stationarity alone can be tiny at an
// active bound while stale or missing z_L/z_U values make the full Ipopt-style
// dual residual large.  The terminal KKT certificate must carry both pieces.
void complete_active_bound_multipliers(const QCQPProblem& p, const Vec& x,
                                       MultiplierEstimate& mult,
                                       const Settings& st) {
  if(mult.lower_bound.size()!=static_cast<std::size_t>(p.n))
    mult.lower_bound.assign(p.n,0.0);
  if(mult.upper_bound.size()!=static_cast<std::size_t>(p.n))
    mult.upper_bound.assign(p.n,0.0);
  const Vec r=qpdual_lagrangian_gradient(p,x,mult);
  const double btol=std::max(1e-8,10.0*st.feasibility_tolerance);
  for(int j=0;j<p.n;++j){
    mult.lower_bound[j]=0.0;
    mult.upper_bound[j]=0.0;
    const bool fixed=std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
      std::abs(p.ub[j]-p.lb[j])<=st.fixed_bound_tolerance;
    if(fixed) continue;
    const bool at_l=std::isfinite(p.lb[j]) && x[j]<=p.lb[j]+btol;
    const bool at_u=std::isfinite(p.ub[j]) && x[j]>=p.ub[j]-btol;
    if(at_l && (!at_u || r[j]>=0.0))
      mult.lower_bound[j]=std::max(0.0,r[j]);
    else if(at_u)
      mult.upper_bound[j]=std::max(0.0,-r[j]);
  }
}

struct QPDualKKTResult {
  bool success=false;
  bool improved=false;
  bool attempted=false;
  Vec x;
  MultiplierEstimate multipliers;
  int iterations=0;
  int accepted_steps=0;
  int active_set_drops=0; // v0.2.11+ invariant: never pivot/drop from a Newton dy
  double stationarity=std::numeric_limits<double>::infinity();
  TangentDiagnostics tangent;
  double multiplier_cutoff=0.0;
};

// Guarded primal-dual Newton on the barrier-aware QP-dual active manifold.
//
// The crucial v0.2.11 rules are:
//   * active inequalities are filtered by an absolute + relative QP-dual cut;
//   * NO constraint is dropped because Newton predicts a negative multiplier;
//   * on n>64 with inequalities, at most one Newton attempt is allowed per
//     fresh QP-dual state; equality-only problems use an autonomous loop;
//   * a candidate with an invalid multiplier sign is backtracked/rejected;
//   * rejection is harmless: the global convexified SQP continues unchanged.
//
// This preserves the fast DISCS behavior of the exact KKT phase while ruling
// out the A4X12 v0.2.10 cascade of 1e5--1e7 predicted negative multipliers.
QPDualKKTResult kkt_polish_qpdual(
    const QCQPProblem& p, Vec x, MultiplierEstimate mult, const Settings& st,
    bool autonomous_large_inequality_takeover = false) {
  QPDualKKTResult out; out.x=x; out.multipliers=mult;
  if(st.terminal_kkt_max_iterations<=0) return out;
  // QDLDL and cholesk are quasi-definite and are not general-indefinite
  // terminal Newton backends.
  if(st.linear_backend=="qdldl" || st.linear_backend=="cholesk") return out;

  std::vector<int> free_variables;
  free_variables.reserve(static_cast<std::size_t>(p.n));
  for (int j = 0; j < p.n; ++j) {
    const bool fixed = std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
        std::abs(p.ub[j] - p.lb[j]) <= st.fixed_bound_tolerance;
    if (fixed) x[static_cast<std::size_t>(j)] = p.lb[j];
    else free_variables.push_back(j);
  }
  out.x=x;

  // A large equality-only QCQP has no combinatorial inequality working set.
  // Once the mapped original stationarity is small, let primal-dual Newton
  // iterate autonomously instead of interleaving one Newton attempt with one
  // more DCA/retraction step.  General large inequality problems retain the
  // historical one-fresh-dual-attempt safeguard.
  const bool autonomous_large_equality_takeover =
      p.n > 64 && p.inequalities.empty();
  const bool autonomous_large_takeover =
      autonomous_large_equality_takeover ||
      (p.n > 64 && !p.inequalities.empty() &&
       autonomous_large_inequality_takeover);
  const int maxit = (p.n > 64 && !autonomous_large_takeover)
      ? 1 : std::max(1,st.terminal_kkt_max_iterations);
  const double alpha_floor = (p.n > 64 && !autonomous_large_takeover)
      ? 1e-4 : st.minimum_step_length;
  std::vector<long long> takeover_active_signature;
  auto normalized_certificate_residual = [&](const QCQPIpoptTermination& c) {
    return std::max({
        c.stop.overall_nlp_error/st.stationarity_tolerance,
        c.stop.dual_infeasibility/st.dual_infeasibility_tolerance,
        c.stop.constraint_violation/st.feasibility_tolerance,
        c.stop.complementarity/st.complementarity_tolerance});
  };
  auto finalize = [&]() -> QPDualKKTResult {
    complete_active_bound_multipliers(p,x,mult,st);
    out.x=x;
    out.multipliers=mult;
    out.stationarity=projected_stationarity(
        p,x,mult,std::max(1e-10,10.0*st.feasibility_tolerance));
    auto aset=qpdual_active_set(p,x,mult,st);
    out.tangent=aset.tangent;
    out.multiplier_cutoff=aset.multiplier_cutoff;
    // `success` is the common Ipopt-style certificate, not merely projected
    // stationarity.  In particular, complementarity and the full signed bound
    // residual must pass before the caller may exempt the endpoint from its
    // monotone-objective rollback rule.
    const auto certificate=qcqp_ipopt_termination(p,x,mult,st);
    out.success=certificate.stop.converged;
    return out;
  };

  for(int it=0;it<maxit;++it){
    out.attempted=true;
    out.iterations=it+1;
    complete_active_bound_multipliers(p,x,mult,st);
    const Feasibility fx=feasibility(p,x);
    const double stat0=projected_stationarity(
        p,x,mult,std::max(1e-10,10.0*st.feasibility_tolerance));
    auto aset=qpdual_active_set(p,x,mult,st);
    out.tangent=aset.tangent;
    out.stationarity=stat0;
    out.multiplier_cutoff=aset.multiplier_cutoff;
    const auto certificate0=qcqp_ipopt_termination(p,x,mult,st);

    // A large inequality-constrained takeover is autonomous only while the
    // active manifold which was certified stable by the outer loop remains
    // unchanged.  Any internal signature change returns the last accepted
    // primal-dual state; the caller can then resume ordinary Lazy-DCA and
    // obtain fresh convex-subproblem multipliers.
    if(autonomous_large_inequality_takeover) {
      if(it==0) takeover_active_signature=aset.signature;
      else if(aset.signature!=takeover_active_signature) {
        if(st.verbose)
          std::cout << "[KKT-QPDUAL-EXIT] reason=active-signature-change"
                    << " old=" << takeover_active_signature.size()
                    << " new=" << aset.signature.size() << "\n";
        return finalize();
      }
    }

    if(st.verbose){
      std::cout << "[KKT-QPDUAL] it="<<it
                << " theta="<<std::scientific<<fx.total_inf
                << " stat="<<stat0
                << " active="<<aset.active.size()
                << " fixed="<<aset.fixed_variables
                << " free="<<aset.free_variables
                << " jacrank="<<aset.tangent.jacobian_rank
                << " tangent="<<aset.tangent.tangent_dimension
                << " redneg="<<aset.tangent.reduced_negative_eigenvalues
                << " redmin="<<aset.tangent.reduced_min_eigenvalue
                << " mucut="<<aset.multiplier_cutoff
                << " mode=guarded-primal-dual"
                << "\n"<<std::defaultfloat;
    }

    if(p.n <= std::max(512,st.exact_eigen_threshold) &&
       aset.tangent.reduced_negative_eigenvalues > 0) return finalize();
    if(certificate0.stop.converged){
      return finalize();
    }

    const int m=(int)aset.active.size();
    const int nf=static_cast<int>(free_variables.size());
    const int N=nf+m;
    if(m==0 || nf==0) {
      if(st.verbose)
        std::cout << "[KKT-QPDUAL-EXIT] reason=empty-reduced-system"
                  << " active=" << m << " free=" << nf << "\n";
      return finalize();
    }

    // Exact stationarity uses ALL original QP duals.  The selected active set
    // only defines the local manifold and the multiplier increments solved by
    // this Newton system.
    Vec stat=qpdual_lagrangian_gradient(p,x,mult);
    // Bound multipliers are inferred in qpdual_active_set and have zero Hessian.
    for(int a=0;a<m;++a)
      if(aset.active[a].kind==ActiveKind::LowerBound ||
         aset.active[a].kind==ActiveKind::UpperBound)
        axpy(aset.y[a],active_gradient(p,x,aset.active[a]),stat);

    SymMatrix B=lagrangian_hessian(p,mult);
    std::vector<int> reduced_index(static_cast<std::size_t>(p.n), -1);
    for (int r = 0; r < nf; ++r)
      reduced_index[static_cast<std::size_t>(free_variables[static_cast<std::size_t>(r)])] = r;
    std::vector<SymmetricTriplet>K;
    K.reserve(B.entries.size()+static_cast<std::size_t>(m)*std::min(nf,16)+nf+m);
    for(const auto&e:B.entries) {
      const int ri=reduced_index[static_cast<std::size_t>(e.i)];
      const int rj=reduced_index[static_cast<std::size_t>(e.j)];
      if(ri>=0 && rj>=0) K.push_back({ri,rj,e.v});
    }
    for(int j=0;j<nf;++j) K.push_back({j,j,st.terminal_kkt_primal_regularization});
    for(int a=0;a<m;++a){
      K.push_back({nf+a,nf+a,-st.terminal_kkt_dual_regularization});
      const Vec g=active_gradient(p,x,aset.active[a]);
      for(int r=0;r<nf;++r) {
        const int j=free_variables[static_cast<std::size_t>(r)];
        if(g[static_cast<std::size_t>(j)]!=0.0)
          K.push_back({r,nf+a,g[static_cast<std::size_t>(j)]});
      }
    }

    Vec rhs(N,0.0);
    for(int r=0;r<nf;++r)
      rhs[static_cast<std::size_t>(r)]=
          -stat[static_cast<std::size_t>(free_variables[static_cast<std::size_t>(r)])];
    for(int a=0;a<m;++a) rhs[static_cast<std::size_t>(nf+a)]=
        -active_value(p,x,aset.active[a]);
    std::string msg; auto fac=make_symmetric_factorization(st,msg);
    if(!fac) {
      if(st.verbose)
        std::cout << "[KKT-QPDUAL-EXIT] reason=backend-create message=" << msg << "\n";
      return finalize();
    }
    if(!fac->factor(N,nf,K,msg)) {
      if(st.verbose)
        std::cout << "[KKT-QPDUAL-EXIT] reason=factorization"
                  << " dim=" << N << " primal=" << nf
                  << " message=" << msg << "\n";
      return finalize();
    }
    // Every compiled symmetric backend validates the regularized KKT inertia
    // in factor(): exactly nf positive and m negative pivots are required.
    // This is a fail-closed solvability/direction gate, not an SOSC
    // certificate for the unregularized original Hessian.  Final acceptance
    // is therefore still decided from exact original-QCQP residuals below.
    if(!fac->solve(rhs,msg)) {
      if(st.verbose)
        std::cout << "[KKT-QPDUAL-EXIT] reason=solve"
                  << " dim=" << N << " primal=" << nf
                  << " message=" << msg << "\n";
      return finalize();
    }
    if(!std::isfinite(fac->last_backward_error()) ||
       fac->last_backward_error()>st.kkt_backward_error_reject) {
      if(st.verbose)
        std::cout << "[KKT-QPDUAL-EXIT] reason=linear-residual"
                  << " berr=" << std::scientific
                  << fac->last_backward_error() << "\n" << std::defaultfloat;
      return finalize();
    }
    Vec dx(static_cast<std::size_t>(p.n),0.0);
    for(int r=0;r<nf;++r)
      dx[static_cast<std::size_t>(free_variables[static_cast<std::size_t>(r)])]=
          rhs[static_cast<std::size_t>(r)];
    Vec dy(rhs.begin()+nf,rhs.end());

    const double d_inf=inf_norm(dx);
    const double dy_inf=inf_norm(dy);
    double multiplier_scale=0.0;
    for(double y:mult.equality) multiplier_scale=std::max(multiplier_scale,std::abs(y));
    for(double y:mult.inequality) multiplier_scale=std::max(multiplier_scale,std::abs(y));
    for(double y:mult.lower_bound) multiplier_scale=std::max(multiplier_scale,std::abs(y));
    for(double y:mult.upper_bound) multiplier_scale=std::max(multiplier_scale,std::abs(y));
    // The outer iterate tolerance is far too coarse here: once x is almost
    // stationary, a Newton step can be tiny in x and still carry the final
    // multiplier correction needed by the original KKT certificate.  Only a
    // roundoff-sized full primal-dual step is terminally indistinguishable
    // from zero; the guarded residual-decrease test below decides all other
    // cases.
    const double terminal_roundoff=64.0*std::numeric_limits<double>::epsilon();
    const double terminal_primal_step_floor=terminal_roundoff*(1.0+inf_norm(x));
    const double terminal_dual_step_floor=terminal_roundoff*(1.0+multiplier_scale);
    if(!std::isfinite(d_inf) || !std::isfinite(dy_inf)) {
      if(st.verbose)
        std::cout << "[KKT-QPDUAL-EXIT] reason=nonfinite-step\n";
      return finalize();
    }
    // A nearly converged primal point can still need a material multiplier
    // correction.  Stopping on ||dx|| alone handed an inconsistent
    // primal-dual state back to DCA and made the result backend-dependent.
    // Treat the Newton step as small only when both blocks are small in their
    // own natural scale.
    if(d_inf<=terminal_primal_step_floor &&
       dy_inf<=terminal_dual_step_floor) {
      if(st.verbose)
        std::cout << "[KKT-QPDUAL-EXIT] reason=small-step"
                  << " step=" << std::scientific << d_inf
                  << " primal_floor=" << terminal_primal_step_floor
                  << " dual_step=" << dy_inf
                  << " dual_floor=" << terminal_dual_step_floor
                  << " stat=" << stat0 << "\n" << std::defaultfloat;
      return finalize();
    }

    const double oldres=autonomous_large_inequality_takeover
        ? normalized_certificate_residual(certificate0)
        : std::max(fx.total_inf,stat0);
    double alpha=1.0;
    bool accepted=false;
    for(int ls=0;ls<st.line_search_max_steps;++ls){
      // Never pivot on a negative predicted multiplier.  Either a sufficiently
      // large step preserves dual feasibility, or this local attempt is not
      // trusted and the global SQP gets control back.
      bool dual_ok=true;
      for(int a=0;a<m;++a){
        if(!aset.active[a].nonnegative_multiplier) continue;
        const double yn=aset.y[a]+alpha*dy[a];
        if(yn < -std::max(aset.multiplier_cutoff,st.terminal_kkt_multiplier_tolerance)){
          dual_ok=false; break;
        }
      }
      if(!dual_ok){
        alpha*=st.line_search_contraction;
        if(alpha<alpha_floor) break;
        continue;
      }

      Vec z=x; Vec ad=dx; for(double&v:ad)v*=alpha; axpy(1.0,ad,z);
      bool bounds_ok=true;
      for(int j=0;j<p.n;++j){
        if(std::isfinite(p.lb[j])&&z[j]<p.lb[j]-10.0*st.feasibility_tolerance) bounds_ok=false;
        if(std::isfinite(p.ub[j])&&z[j]>p.ub[j]+10.0*st.feasibility_tolerance) bounds_ok=false;
      }
      if(!bounds_ok){
        alpha*=st.line_search_contraction;
        if(alpha<alpha_floor) break;
        continue;
      }
      z=retract_equalities(p,z,st.equality_retraction_iterations,
                           st.equality_retraction_tolerance,
                           st.equality_retraction_regularization,
                           st.fraction_to_boundary);
      // The retraction already treats fixed coordinates as parameters.  Snap
      // once more here to make that contract explicit for alternate
      // restoration implementations and for the subsequent bound test.
      for (int j = 0; j < p.n; ++j) {
        if (reduced_index[static_cast<std::size_t>(j)] < 0)
          z[static_cast<std::size_t>(j)] = p.lb[j];
      }

      MultiplierEstimate mt=mult;
      bool mt_ok=true;
      for(int a=0;a<m;++a){
        const auto& c=aset.active[a];
        const double yn=aset.y[a]+alpha*dy[a];
        if(c.kind==ActiveKind::Equality && c.index<(int)mt.equality.size()) {
          mt.equality[c.index]=yn;
        } else if(c.kind==ActiveKind::Inequality && c.index<(int)mt.inequality.size()) {
          if(yn < -std::max(aset.multiplier_cutoff,st.terminal_kkt_multiplier_tolerance)) {
            mt_ok=false; break;
          }
          mt.inequality[c.index]=std::max(0.0,yn);
        } else if(c.kind==ActiveKind::LowerBound && c.index<(int)mt.lower_bound.size()) {
          if(yn < -std::max(aset.multiplier_cutoff,st.terminal_kkt_multiplier_tolerance)) {
            mt_ok=false; break;
          }
          mt.lower_bound[c.index]=std::max(0.0,yn);
        } else if(c.kind==ActiveKind::UpperBound && c.index<(int)mt.upper_bound.size()) {
          if(yn < -std::max(aset.multiplier_cutoff,st.terminal_kkt_multiplier_tolerance)) {
            mt_ok=false; break;
          }
          mt.upper_bound[c.index]=std::max(0.0,yn);
        }
      }
      if(!mt_ok){
        alpha*=st.line_search_contraction;
        if(alpha<alpha_floor) break;
        continue;
      }
      // Remove only already-tiny duals of constraints that are materially slack
      // at the CANDIDATE point.  This is ordinary complementarity cleanup, not
      // a Newton active-set pivot.
      for(int i=0;i<(int)mt.inequality.size();++i)
        if(p.inequalities[i].value(z)<-st.working_set_drop_slack &&
           mt.inequality[i]<=10.0*st.working_set_drop_multiplier)
          mt.inequality[i]=0.0;

      // Canonical bound duals make the takeover merit identical to the final
      // original-QCQP certificate rather than a projected-stationarity proxy.
      if(autonomous_large_inequality_takeover)
        complete_active_bound_multipliers(p,z,mt,st);

      const Feasibility fz=feasibility(p,z);
      const double sz=projected_stationarity(
          p,z,mt,std::max(1e-10,10.0*st.feasibility_tolerance));
      const auto certificate_z=autonomous_large_inequality_takeover
          ? qcqp_ipopt_termination(p,z,mt,st)
          : QCQPIpoptTermination{};
      const double nr=autonomous_large_inequality_takeover
          ? normalized_certificate_residual(certificate_z)
          : std::max(fz.total_inf,sz);
      const bool preserve_feas = autonomous_large_inequality_takeover
          ? fz.total_inf<=st.feasibility_tolerance
          : fz.total_inf<=std::max(10.0*st.feasibility_tolerance,
                                  2.0*fx.total_inf+st.feasibility_tolerance);
      const bool stationarity_improves =
          !autonomous_large_inequality_takeover ||
          sz < (1.0-1e-6*alpha)*stat0;
      const bool improve = std::isfinite(oldres) && std::isfinite(nr) &&
          nr < (1.0-1e-4*alpha)*oldres && stationarity_improves;
      if(preserve_feas && improve){
        x=std::move(z); mult=std::move(mt); accepted=true; out.improved=true;
        ++out.accepted_steps; out.stationarity=sz;
        if(st.verbose){
          std::cout << "[KKT-QPDUAL-ACCEPT] alpha="<<std::scientific<<alpha
                    << " step="<<alpha*d_inf
                    << " stat="<<sz
                    << " theta="<<fz.total_inf
                    << "\n"<<std::defaultfloat;
        }
        break;
      }
      alpha*=st.line_search_contraction;
      if(alpha<alpha_floor) break;
    }

    if(!accepted){
      if(st.verbose) std::cout << "[KKT-QPDUAL-REJECT] no safe guarded KKT step\n";
      break;
    }
  }

  return finalize();
}



bool finite_state(const Vec& x) {
  for (double v : x) if (!std::isfinite(v)) return false;
  return true;
}

struct FilterEntry { double theta; double f; };

struct DirectStartEvaluation {
  bool finite = false;
  double objective = std::numeric_limits<double>::infinity();
  Feasibility feasibility;
  std::vector<int> violated_inequalities;
};

// Evaluate the unmodified ORIGINAL QCQP once before deciding whether Phase I
// may be bypassed.  Do not use max() to infer finiteness: a NaN row can be
// hidden by std::max's ordering semantics.  The direct-start gate must fail
// closed on every non-finite primal coordinate, objective, or constraint row.
DirectStartEvaluation evaluate_direct_start(
    const QCQPProblem& problem, const Vec& x) {
  DirectStartEvaluation out;
  if(static_cast<int>(x.size())!=problem.n || !finite_state(x)) return out;

  out.objective=problem.objective.value(x);
  if(!std::isfinite(out.objective)) return out;

  for(const auto& h:problem.equalities) {
    const double value=h.value(x);
    if(!std::isfinite(value)) return out;
    out.feasibility.equality_inf=std::max(
        out.feasibility.equality_inf,std::abs(value));
  }
  out.violated_inequalities.reserve(problem.inequalities.size());
  for(int i=0;i<static_cast<int>(problem.inequalities.size());++i) {
    const double value=problem.inequalities[static_cast<std::size_t>(i)].value(x);
    if(!std::isfinite(value)) return out;
    out.feasibility.inequality_inf=std::max(
        out.feasibility.inequality_inf,std::max(0.0,value));
    if(value>0.0) out.violated_inequalities.push_back(i);
  }
  for(int j=0;j<problem.n;++j) {
    const double value=x[static_cast<std::size_t>(j)];
    if(std::isfinite(problem.lb[static_cast<std::size_t>(j)]))
      out.feasibility.bound_inf=std::max(
          out.feasibility.bound_inf,
          std::max(0.0,problem.lb[static_cast<std::size_t>(j)]-value));
    if(std::isfinite(problem.ub[static_cast<std::size_t>(j)]))
      out.feasibility.bound_inf=std::max(
          out.feasibility.bound_inf,
          std::max(0.0,value-problem.ub[static_cast<std::size_t>(j)]));
  }
  out.feasibility.total_inf=std::max({
      out.feasibility.equality_inf,
      out.feasibility.inequality_inf,
      out.feasibility.bound_inf});
  out.finite=std::isfinite(out.feasibility.total_inf);
  return out;
}

bool filter_acceptable(const std::vector<FilterEntry>& filter,
                       double theta, double f, const Settings& st) {
  for (const auto& e : filter) {
    const bool theta_good = theta < (1.0-st.filter_gamma_theta)*e.theta;
    // Fletcher-Leyffer style filter margin: objective margin scales with
    // the stored infeasibility and therefore vanishes on the feasible axis.
    // The previous max(1,theta) created an artificial absolute 1e-8 floor.
    const bool f_good = f < e.f-st.filter_gamma_f*std::max(0.0,e.theta);
    if (!theta_good && !f_good) return false;
  }
  return true;
}

void filter_insert(std::vector<FilterEntry>& filter, double theta, double f) {
  filter.erase(std::remove_if(filter.begin(), filter.end(), [&](const FilterEntry& e) {
    return theta <= e.theta && f <= e.f;
  }), filter.end());
  filter.push_back({theta,f});
  if (filter.size() > 64) filter.erase(filter.begin());
}

} // namespace

Solver::Solver(Settings settings) : st_(std::move(settings)) {}

Result Solver::solve(const QCQPProblem& problem) {
  if(!st_.native_al_fallback_resolved)
    return detail::solve_native_al_fallback(problem,st_);
  if(!st_.standard_auto_resolved) {
    problem.validate();
    if(st_.gr_proximal_policy!="historical"&&st_.gr_proximal_policy!="frozen")
      throw std::invalid_argument("gr_proximal_policy expects historical|frozen");
    if(st_.gr_fixed_centering_steps<0||st_.gr_fixed_centering_steps>8)
      throw std::invalid_argument("gr_fixed_centering_steps expects 0..8");
    using AdaptiveClock=std::chrono::steady_clock;
    const auto adaptive_began=AdaptiveClock::now();
    const double certificate_deadline=std::chrono::duration<double>(adaptive_began.time_since_epoch()).count()+st_.time_limit_seconds;
    if(st_.standard_r10_policy=="auto"&&st_.convexification_study=="off"&&
        st_.study_controller=="r3"&&!st_.rank_presolve_context&&!st_.original_private_phase&&
        !st_.product_definition_presolve&&!st_.power2_scaling&&
        (!st_.accepted_checkpoint_callback||st_.accepted_checkpoint_builtin)){
      const auto convex=detail::convex_aggregate_certificate(problem,st_.feasibility_tolerance,certificate_deadline);
      if(convex.verified){
        Result r;r.status=Status::Infeasible;r.x=problem.x0;
        r.message="exact convex combination of original balances, bound and diagonal quadratics proves infeasibility";
        r.objective=problem.objective.value(r.x);r.feasibility=feasibility(problem,r.x);
        r.certificate_equality_multipliers.assign(problem.equalities.size(),0);
        r.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
        r.certificate_lower_multipliers.assign(problem.n,0);r.certificate_upper_multipliers.assign(problem.n,0);
        r.convex_infeasibility_certificate=true;r.convex_infeasibility_lower_bound=convex.primal_lower_bound;
        r.convex_infeasibility_minimum=convex.minimum;r.convex_infeasibility_weight_norm=convex.weight_norm;
        r.convex_infeasibility_equality_sign=convex.equality_sign;r.convex_infeasibility_bound_variable=convex.bound_variable;
        r.convex_infeasibility_lower_side=convex.lower;r.convex_infeasibility_bound_weight=convex.bound_weight;
        r.convex_infeasibility_rows=convex.inequality_rows;r.convex_infeasibility_weights=convex.inequality_weights;
        r.elapsed_seconds=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
        r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;r.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]=r.elapsed_seconds;
        std::cout<<"[EXACT-CONVEX-INFEASIBILITY] verified=1 original_primal_lower_bound="<<convex.primal_lower_bound
            <<" strict_KKT_success=0 all_original_coefficients_retained=1 external_optimizer=0"<<std::endl;
        return r;
      }
      const auto contradiction=detail::affine_cycle_certificate(problem,certificate_deadline);
      // Exact infeasibility can be smaller than the requested residual
      // tolerance. Only stop when the certified lower bound excludes a
      // tolerance-valid original primal point as well.
      if(contradiction.verified&&boost::multiprecision::cpp_rational(contradiction.primal_lower_bound)>
          boost::multiprecision::cpp_rational(st_.feasibility_tolerance)){
        Result r;r.status=Status::Infeasible;r.x=problem.x0;
        r.message="exact original affine equality cycle proves infeasibility";
        r.objective=problem.objective.value(r.x);r.feasibility=feasibility(problem,r.x);
        r.certificate_equality_multipliers.assign(problem.equalities.size(),0);
        r.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
        r.certificate_lower_multipliers.assign(problem.n,0);r.certificate_upper_multipliers.assign(problem.n,0);
        r.affine_infeasibility_certificate=true;r.affine_infeasibility_rows=contradiction.rows;
        r.affine_infeasibility_weights=contradiction.weights;r.affine_infeasibility_constant=contradiction.constant;
        r.affine_infeasibility_primal_lower_bound=contradiction.primal_lower_bound;
        r.elapsed_seconds=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
        r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
        r.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]=r.elapsed_seconds;
        std::cout<<"[EXACT-AFFINE-INFEASIBILITY] verified=1 rows="<<contradiction.rows.size()
            <<" constant="<<contradiction.constant<<" original_primal_lower_bound="<<contradiction.primal_lower_bound
            <<" strict_KKT_success=0 external_optimizer=0 frozen_input_unchanged=1"<<std::endl;
        return r;
      }
    }
    // Independent exact row proof, optionally after a declared native
    // projection onto a necessary zero face. With no face projection, every
    // supplied x0 coordinate is bit-preserved and no iteration is charged.
    // All exact proof time is charged; lifts precede the full original audit.
    if(st_.standard_r10_policy=="auto"&&st_.convexification_study=="off"&&
        st_.study_controller=="r3"&&st_.dca_engine=="majorants"&&st_.outer_algorithm=="lazy-dca"&&
        !st_.rank_presolve_context&&!st_.original_private_phase&&!st_.accepted_checkpoint_callback&&
        !st_.product_definition_presolve&&!st_.power2_scaling&&st_.restoration_shared_iteration_budget) {
      detail::ClosedZeroFace face(problem,certificate_deadline);
      // Reuse the general source/time caps checked by the face inspector;
      // an absent face is not a reason to skip independent EQ identities.
      if(face.reason!="source-budget"&&face.reason!="signed-square-model-refused"&&face.reason!="deadline") {
        const bool projected=face.applied();
        const int projection_iterations=projected?1:0;
        const QCQPProblem& native_base=projected?face.model:problem;
        detail::ExactRedundantEqualities equality_rows(native_base,st_.feasibility_tolerance,certificate_deadline);
      if(projected||equality_rows.changed()) {
        // Constant-coefficient/product-simplex identities are verified in
        // exact rationals, retaining all coordinates and original row bytes.
        // Their full-row feasibility amplification tightens only the private
        // solve; the final ORIGINAL test below remains 1e-6 (user tolerance).
        const double preparation=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
        Settings inner=st_;
        if(equality_rows.changed())inner.feasibility_tolerance=std::min(inner.feasibility_tolerance,equality_rows.inner_feasibility_tolerance);
        inner.time_limit_seconds=std::max(0.,st_.time_limit_seconds-preparation);
        inner.max_outer_iterations=std::max(0,st_.max_outer_iterations-projection_iterations);
        inner.restoration_max_iterations=std::max(0,st_.restoration_max_iterations-projection_iterations);
        inner.restoration_shared_iteration_budget=true;
        const bool may_continue=inner.time_limit_seconds>0&&inner.max_outer_iterations>=1&&st_.restoration_max_iterations>=projection_iterations;
        std::cout<<"[NATIVE-EXACT-EQ-OPTIONAL-FACE] proofs="<<face.proofs.size()<<" forced_coordinates="<<face.introduced.size()
            <<" EQ="<<problem.equalities.size()<<"->"<<native_base.equalities.size()
            <<" IQ="<<problem.inequalities.size()<<"->"<<native_base.inequalities.size()
            <<" exact_redundant_EQ="<<equality_rows.witnesses.size()
            <<" retained_EQ="<<equality_rows.reduced.equalities.size()
            <<" private_feasibility_tolerance="<<inner.feasibility_tolerance
            <<" projection_inf="<<face.projection_inf<<" projection_l1="<<face.projection_l1
            <<" original_frozen_input_unchanged=1 supplied_raw_x0_retained=1"
            <<" private_native_start_changed="<<(may_continue&&projected&&face.projection_inf>0)<<" phase_outer="<<(may_continue?projection_iterations:0)
            <<" phase_restoration="<<(may_continue?projection_iterations:0)
            <<" native_projection_applied="<<(may_continue&&projected)
            <<" no_projection_raw_start_bit_preserved="<<!projected
            <<" projection_is_not_KKT=1 external_endpoint=0 original_audit_required=1"
            <<" proof_reason="<<equality_rows.reason
            <<" original_global_budget_seconds="<<st_.time_limit_seconds
            <<" remaining_child_budget_seconds="<<inner.time_limit_seconds
            <<" preparation_seconds="<<preparation<<std::endl;
        Result transformed;
        if(may_continue) {
          const QCQPProblem& child_model=equality_rows.changed()?equality_rows.reduced:native_base;
          // The finisher audits a disposable lifted COPY. Ordinary return
          // below retains ownership of the single actual EQ/face lift.
          detail::NativeOriginalView original_view(child_model,problem,st_,[&](Result& copy) {
            try {if(equality_rows.changed())equality_rows.lift(copy);}
            catch(const std::exception&){return false;}
            return !projected||face.lift(copy);
          },std::chrono::duration<double>(adaptive_began.time_since_epoch()).count()+st_.time_limit_seconds);
          detail::NativeOriginalViewScope original_scope(original_view);
          transformed=Solver(inner).solve(child_model);
        }
        else {
          transformed.status=Status::TimeLimit;transformed.x=problem.x0;
          transformed.message="exact-EQ/optional-face proof/projection budget unavailable; no original certificate";
        }
        // Only an actual native projection consumes outer/restoration
        // iterations. Exact row proof alone consumes time/work, never a
        // solver iteration or factor/linear solve.
        if(may_continue&&projected){transformed.outer_iterations+=projection_iterations;transformed.restoration_iterations+=projection_iterations;transformed.restoration_total_seconds+=preparation;}
        const auto certificate_began=AdaptiveClock::now();
        bool row_lifted=may_continue;
        if(row_lifted&&equality_rows.changed())try{equality_rows.lift(transformed);}catch(const std::exception&){row_lifted=false;}
        const bool lifted=row_lifted&&(!projected||face.lift(transformed));
        if(!lifted) {
          if(transformed.x.size()!=problem.x0.size())transformed.x=problem.x0;
          transformed.certificate_equality_multipliers.assign(problem.equalities.size(),0);
          transformed.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
          transformed.certificate_lower_multipliers.assign(problem.n,0);
          transformed.certificate_upper_multipliers.assign(problem.n,0);
          transformed.termination_audit_valid=false;transformed.termination_scaling_available=false;
          transformed.dual_infeasibility_inf=INFINITY;transformed.complementarity_inf=INFINITY;
          transformed.raw_kkt_error=INFINITY;transformed.acceptable_convergence=false;
          transformed.acceptable_point=false;transformed.acceptable_on_failure=false;
          transformed.acceptable_streak=0;transformed.acceptable_trigger.clear();
          transformed.objective=problem.objective.value(transformed.x);transformed.feasibility=feasibility(problem,transformed.x);
        }
        Settings original=st_;original.rank_presolve_context=nullptr;original.termination_test="project";
        original.acceptable_iterations=0;original.acceptable_on_failure=false;
        const bool strict=lifted&&detail::audit_original_termination(problem,original,transformed)&&
            detail::adaptive_original_strict(problem,original,transformed);
        const double certificate=std::chrono::duration<double>(AdaptiveClock::now()-certificate_began).count();
        const double total=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
        const bool within=total<=st_.time_limit_seconds&&transformed.outer_iterations<=st_.max_outer_iterations&&
            transformed.restoration_iterations<=st_.restoration_max_iterations;
        if(!within){transformed.status=Status::TimeLimit;transformed.message="exact-EQ/optional-face phase: cumulative original task budget exceeded; endpoint unqualified";}
        else if(strict){transformed.status=Status::Solved;transformed.message="native exact equality reduction with optional zero-face phase and lifted full original QCQP KKT";}
        else if(transformed.status==Status::Solved||transformed.status==Status::SolvedAcceptable||transformed.status==Status::SolvedAcceptableOnFailure){transformed.status=Status::NumericalError;transformed.message="exact-EQ/optional-face phase: original certificate rejected";}
        transformed.elapsed_seconds=total;transformed.lazy_newton.solver_wall_seconds=total;
        // Outer native phase owns the global envelope. Nested adaptive fields
        // describe child work, so aggregate the declared projection/proof and
        // never relabel a private-model strict flag as an ORIGINAL certificate.
        auto& adaptive=transformed.adaptive_equality_probe;
        if(adaptive.eligible||adaptive.attempted) {
          adaptive.global_seconds=st_.time_limit_seconds;
          adaptive.global_outer_budget=st_.max_outer_iterations;
          adaptive.global_restoration_budget=std::max(0,std::min(st_.max_outer_iterations,st_.restoration_max_iterations));
          adaptive.total_seconds=total;adaptive.within_global_budget=within;
          if(may_continue) {
            adaptive.probe_outer+=projection_iterations;adaptive.probe_restoration+=projection_iterations;
            adaptive.probe_outer_budget+=projection_iterations;
            adaptive.probe_seconds+=preparation;
            adaptive.probe_cost_seconds[int(NewtonCost::Preparation)]+=preparation;
          }
          adaptive.selection_audit_seconds+=certificate;
          adaptive.probe_strict=false;adaptive.stabilized_strict=false;
          adaptive.probe_raw_residual=INFINITY;adaptive.stabilized_raw_residual=INFINITY;
          if(adaptive.selected_phase=="historical") {
            adaptive.probe_strict=strict&&within;
            adaptive.probe_raw_residual=lifted?transformed.raw_kkt_error:INFINITY;
          } else if(adaptive.selected_phase=="stabilized") {
            adaptive.stabilized_strict=strict&&within;
            adaptive.stabilized_raw_residual=lifted?transformed.raw_kkt_error:INFINITY;
          }
          adaptive.reason="native-exact-EQ-proof-and-optional-face-projection-charged-to-first-phase;only-selected-final-lifted-original-audited:"+adaptive.reason;
        }
        transformed.standard_auto_enabled=true;
        transformed.standard_auto_reason="native-exact-redundant-EQ+optional-closed-zero-face:"+transformed.standard_auto_reason;
        transformed.standard_auto_exact_global_faces=projected;
        transformed.standard_auto_exact_global_faces_applied=may_continue&&projected;
        transformed.standard_auto_exact_global_faces_reason=projected?"exact-positive-affine-IQ-and-original-lower-zero-bounds;declared-native-projection":"no-face-projection;independent-exact-equality-row-proof-only";
        transformed.standard_auto_exact_face_start_projection_inf=may_continue&&projected?face.projection_inf:0.;
        transformed.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=preparation;
        transformed.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=certificate;
        transformed.lazy_newton.certificate_seconds+=certificate;
        transformed.termination_multiplier_source=projected?"exact-positive-IQ-zero-face-upper-to-original-IQ-dual-lift":"exact-redundant-equality-zero-lift-full-original-QCQP";
        std::cout<<"[NATIVE-EXACT-EQ-OPTIONAL-FACE-POSTSOLVE] strict="<<strict<<" budget="<<within
            <<" objective_original="<<transformed.objective<<" primal="<<transformed.feasibility.total_inf
            <<" dual="<<transformed.dual_infeasibility_inf<<" compl="<<transformed.complementarity_inf
            <<" outer="<<transformed.outer_iterations<<" restoration="<<transformed.restoration_iterations
            <<" total_seconds="<<total<<std::endl;
        return transformed;
      }
      }
    }
    if(!st_.affine_face_quadratic_rewrite_consumed&&st_.standard_r10_policy=="auto"&&
        st_.convexification_study=="off"&&st_.study_controller=="r3"&&
        st_.dca_engine=="majorants"&&st_.outer_algorithm=="lazy-dca"&&
        !st_.rank_presolve_context&&!st_.original_private_phase&&
        !st_.accepted_checkpoint_callback&&!st_.product_definition_presolve&&!st_.power2_scaling) {
      detail::AffineFaceQuadraticRewrite rewrite(problem,certificate_deadline);
      if(rewrite.applied()) {
        const double preparation=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
        Settings inner=st_;inner.affine_face_quadratic_rewrite_consumed=true;
        inner.restoration_shared_iteration_budget=true;
        inner.time_limit_seconds=std::max(0.,st_.time_limit_seconds-preparation);
        std::cout<<"[AFFINE-FACE-QUADRATIC-REWRITE] rows="<<rewrite.links.size()
            <<" exact_polynomial_identity=1 retained_equalities=1 raw_x0_unchanged=1"
            <<" native_R3=1 original_audit_required=1 preparation_seconds="<<preparation<<std::endl;
        for(const auto& link:rewrite.links)std::cout<<"[AFFINE-FACE-QUADRATIC-LINK] inequality="<<link.inequality
            <<" equality="<<link.equality<<" t="<<link.t<<" u="<<link.u<<" scale="<<std::setprecision(17)<<link.scale
            <<" q_original=q_reduced+scale*(t-u)*h"<<std::endl;
        Result transformed;
        if(inner.time_limit_seconds>0)transformed=Solver(inner).solve(rewrite.model);
        else {transformed.status=Status::TimeLimit;transformed.x=problem.x0;}
        const auto certificate_began=AdaptiveClock::now();
        const bool lifted=rewrite.lift(transformed);
        if(!lifted) {
          transformed.x=problem.x0;
          transformed.certificate_equality_multipliers.assign(problem.equalities.size(),0);
          transformed.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
          transformed.certificate_lower_multipliers.assign(problem.n,0);
          transformed.certificate_upper_multipliers.assign(problem.n,0);
          transformed.termination_audit_valid=false;
          transformed.termination_scaling_available=false;
          transformed.dual_infeasibility_inf=INFINITY;
          transformed.complementarity_inf=INFINITY;transformed.raw_kkt_error=INFINITY;
          transformed.acceptable_convergence=false;transformed.acceptable_point=false;
          transformed.acceptable_on_failure=false;transformed.acceptable_streak=0;
          transformed.acceptable_trigger.clear();
          transformed.objective=problem.objective.value(transformed.x);
          transformed.feasibility=feasibility(problem,transformed.x);
        }
        Settings original=st_;original.rank_presolve_context=nullptr;original.termination_test="project";
        original.acceptable_iterations=0;original.acceptable_on_failure=false;
        const bool strict=lifted&&detail::audit_original_termination(problem,original,transformed)&&
            detail::adaptive_original_strict(problem,original,transformed);
        const double certificate=std::chrono::duration<double>(AdaptiveClock::now()-certificate_began).count();
        const double total=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
        const bool within=total<=st_.time_limit_seconds&&transformed.outer_iterations<=st_.max_outer_iterations&&
            transformed.restoration_iterations<=st_.restoration_max_iterations;
        if(strict&&within){transformed.status=Status::Solved;transformed.message="exact affine-face quadratic rewrite and lifted original QCQP KKT certificate";}
        else if(transformed.status==Status::Solved||transformed.status==Status::SolvedAcceptable||transformed.status==Status::SolvedAcceptableOnFailure){transformed.status=within?Status::NumericalError:Status::TimeLimit;transformed.message="affine-face quadratic rewrite original certificate or shared budget rejected";}
        transformed.elapsed_seconds=total;transformed.lazy_newton.solver_wall_seconds=total;
        transformed.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=preparation;
        transformed.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=certificate;
        transformed.lazy_newton.certificate_seconds+=certificate;
        transformed.termination_multiplier_source="exact-affine-face-quadratic-lift-full-original-audit";
        std::cout<<"[AFFINE-FACE-QUADRATIC-POSTSOLVE] strict="<<strict<<" budgets="<<within
            <<" original_primal="<<transformed.feasibility.total_inf<<" original_dual="<<transformed.dual_infeasibility_inf
            <<" original_compl="<<transformed.complementarity_inf<<" total_seconds="<<total<<std::endl;
        return transformed;
      }
    }
    detail::CommonFactorSubstitution common_factor(problem,st_,certificate_deadline);
    if(common_factor.applied) {
      const double preprocessing=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
      Settings inner=st_;inner.time_limit_seconds=std::max(0.,st_.time_limit_seconds-preprocessing);
      inner.dca_rank_presolve=false;
      // The inverse Jacobian multiplies the u-stationarity residual by t.
      // Request tighter internal residuals; acceptance still uses the user's
      // unchanged tolerances on the complete original model.
      inner.dual_infeasibility_tolerance=std::min(inner.dual_infeasibility_tolerance,1e-10);
      inner.complementarity_tolerance=std::min(inner.complementarity_tolerance,1e-10);
      std::cout<<"[COMMON-FACTOR-SUBSTITUTION] variable="<<common_factor.variable
          <<" exact_polynomial_identity=1 box_perspective=1 frozen_input_unchanged=1"
          <<" mapped_start_from_frozen_input=1 external_optimizer=0 configuration=R3"<<std::endl;
      Result transformed=detail::solve_with_power2_scaling(common_factor.model,inner);
      const bool lifted=common_factor.lift(problem,transformed);
      Settings original=st_;original.rank_presolve_context=nullptr;
      original.acceptable_iterations=0;original.acceptable_on_failure=false;original.termination_test="project";
      bool strict=lifted&&detail::audit_original_termination(problem,original,transformed)&&
          detail::adaptive_original_strict(problem,original,transformed);
      const double total=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
      if(strict&&total<st_.time_limit_seconds){
        if(transformed.status!=Status::Solved){
          transformed.original_failure_status=status_string(transformed.status);
          transformed.original_failure_message=transformed.message;
          transformed.termination_certificate_rescued=true;
        }
        transformed.status=Status::Solved;
        transformed.message="native common-factor affine reduction satisfies audited original QCQP KKT";
      } else if(transformed.status==Status::Solved){
        transformed.status=total>=st_.time_limit_seconds?Status::TimeLimit:Status::NumericalError;
        transformed.message="common-factor original postsolve certificate unavailable or rejected";
      }
      if(!lifted){transformed.x=problem.x0;
        transformed.certificate_equality_multipliers.assign(problem.equalities.size(),0);
        transformed.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
        transformed.certificate_lower_multipliers.assign(problem.n,0);
        transformed.certificate_upper_multipliers.assign(problem.n,0);
        transformed.termination_audit_valid=false;
      }
      transformed.objective=problem.objective.value(transformed.x);
      transformed.feasibility=feasibility(problem,transformed.x);
      transformed.common_factor_substitution=true;transformed.common_factor_variable=common_factor.variable;
      transformed.termination_multiplier_source="common-factor-affine-substitution-lift-full-original-KKT-audit";
      transformed.elapsed_seconds=total;transformed.lazy_newton.solver_wall_seconds=total;
      transformed.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=preprocessing;
      std::cout<<"[COMMON-FACTOR-POSTSOLVE] strict="<<strict<<" positive_inverse="<<lifted
          <<" original_primal="<<transformed.feasibility.total_inf
          <<" original_dual="<<transformed.dual_infeasibility_inf
          <<" original_compl="<<transformed.complementarity_inf<<" seconds="<<total<<std::endl;
      return transformed;
    }
    detail::PositiveSquareEpigraphSubstitution epigraph(problem,st_,certificate_deadline);
    if(epigraph.applied) {
      const auto restored=detail::restoration_step_available(st_.restoration_shared_iteration_budget,
          st_.restoration_max_iterations,0)?
          detail::restore_positive_square_epigraph(problem,st_,certificate_deadline):
          detail::PositiveSquareEpigraphRestoration{};
      if(restored.accepted){
        epigraph.model.x0=restored.x;
        epigraph.model.x0[epigraph.variable]=restored.x[epigraph.variable]*restored.x[epigraph.variable];
        std::cout<<"[POSITIVE-SQUARE-EPIGRAPH-RESTORATION] stage=before-exact-substitution"
            <<" accepted=1 original_primal="<<restored.before<<"->"<<restored.after
            <<" seconds="<<restored.seconds<<" frozen_input_unchanged=1 internal_feasibility_phase=1"<<std::endl;
      }
      const double preprocessing=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
      Settings inner=st_;inner.time_limit_seconds=std::max(0.,st_.time_limit_seconds-preprocessing);
      if(st_.restoration_shared_iteration_budget&&restored.attempted)
        inner.restoration_max_iterations=detail::remaining_shared_restoration(
            st_.restoration_max_iterations,1);
      std::cout<<"[POSITIVE-SQUARE-EPIGRAPH-SUBSTITUTION] exact=1 variable="<<epigraph.variable
          <<" u_equals_t_squared=1 t_nonnegative=1 objective_monotone=1"
          <<" all_original_constants_retained=1 original_frozen_start_mapped_exactly=1"
          <<" external_optimizer=0 configuration=R3"<<std::endl;
      // The detected epigraph can be orders of magnitude larger after u=t^2.
      // Existing exact dyadic scaling keeps every binary64 coefficient and
      // performs its own unscaled transformed-QCQP audit before the t lift.
      inner.dca_rank_presolve=false;
      Result transformed=detail::solve_with_power2_scaling(epigraph.model,inner);
      if(restored.attempted){++transformed.restoration_iterations;transformed.restoration_total_seconds+=restored.seconds;}
      transformed.restoration_linear_solves+=restored.solves;
      transformed.lazy_newton.factor_attempts+=restored.factors;
      transformed.lazy_newton.linear_solves+=restored.solves;
      transformed.kkt_factorizations+=restored.factors;
      const bool lifted=epigraph.lift(transformed);
      Settings original=st_;original.rank_presolve_context=nullptr;
      original.acceptable_iterations=0;original.acceptable_on_failure=false;original.termination_test="project";
      bool strict=false;
      if(lifted)strict=detail::audit_original_termination(problem,original,transformed)&&
          detail::adaptive_original_strict(problem,original,transformed);
      const double total=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
      if(strict&&total<st_.time_limit_seconds){
        transformed.status=Status::Solved;
        transformed.message="native positive-square substitution and multiplier lift satisfy audited original QCQP KKT";
      } else if(transformed.status==Status::Solved){
        transformed.status=total>=st_.time_limit_seconds?Status::TimeLimit:Status::NumericalError;
        transformed.message="positive-square original postsolve certificate unavailable or rejected";
      }
      if(!lifted){
        transformed.x=problem.x0;
        transformed.certificate_equality_multipliers.assign(problem.equalities.size(),0);
        transformed.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
        transformed.certificate_lower_multipliers.assign(problem.n,0);
        transformed.certificate_upper_multipliers.assign(problem.n,0);
        transformed.termination_audit_valid=false;
      }
      transformed.objective=problem.objective.value(transformed.x);
      transformed.feasibility=feasibility(problem,transformed.x);
      transformed.positive_square_epigraph_substitution=true;
      transformed.positive_square_epigraph_variable=epigraph.variable;
      transformed.termination_multiplier_source="positive-square-exact-substitution-lift-full-original-KKT-audit";
      transformed.elapsed_seconds=total;transformed.lazy_newton.solver_wall_seconds=total;
      transformed.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=preprocessing;
      std::cout<<"[POSITIVE-SQUARE-EPIGRAPH-POSTSOLVE] strict="<<strict<<" positive_inverse="<<lifted
          <<" original_primal="<<transformed.feasibility.total_inf
          <<" original_dual="<<transformed.dual_infeasibility_inf
          <<" original_compl="<<transformed.complementarity_inf<<" seconds="<<total<<std::endl;
      return transformed;
    }
    const auto graph=detail::restoration_step_available(st_.restoration_shared_iteration_budget,
        st_.restoration_max_iterations,0)?
        detail::coefficient_graph_certificate(problem,st_,certificate_deadline):
        detail::CoefficientGraphCertificate{};
    if(graph.constructed) {
      Result certificate;certificate.x=graph.x;
      certificate.certificate_equality_multipliers=graph.multipliers;
      certificate.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
      certificate.certificate_lower_multipliers.assign(problem.n,0);
      certificate.certificate_upper_multipliers.assign(problem.n,0);
      Settings original=st_;original.rank_presolve_context=nullptr;
      original.acceptable_iterations=0;original.acceptable_on_failure=false;
      original.termination_test="project";
      const bool audited=detail::audit_original_termination(problem,original,certificate)&&
          detail::adaptive_original_strict(problem,original,certificate);
      const double total=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
      if(audited&&total<st_.time_limit_seconds) {
        certificate.status=Status::Solved;
        certificate.message="exact coefficient graph point and multipliers satisfy audited original QCQP KKT";
        certificate.objective=problem.objective.value(certificate.x);
        certificate.feasibility=feasibility(problem,certificate.x);
        certificate.termination_multiplier_source="coefficient-graph-exact-domain-full-original-KKT-audit";
        certificate.coefficient_graph_certificate=true;
        certificate.coefficient_graph_vertices=graph.vertices;
        certificate.coefficient_graph_edges=graph.edges;
        certificate.coefficient_graph_components=graph.components;
        certificate.coefficient_graph_independent=graph.independent;
        certificate.standard_auto_mode="native-coefficient-graph-certificate";
        certificate.standard_auto_reason=graph.reason;
        certificate.standard_auto_configuration="R3";
        certificate.standard_auto_requested_engine=st_.dca_engine;
        certificate.standard_auto_effective_engine=st_.dca_engine;
        certificate.effective_outer_algorithm="coefficient-graph-certificate";
        certificate.elapsed_seconds=total;certificate.lazy_newton.solver_wall_seconds=total;
        certificate.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]=total;
        certificate.restoration_iterations=1;certificate.restoration_total_seconds=total;
        std::cout<<"[COEFFICIENT-GRAPH-CERTIFICATE] accepted=1 vertices="<<graph.vertices
            <<" edges="<<graph.edges<<" coordinate_blocks="<<graph.components
            <<" independent_vertices="<<graph.independent
            <<" original_primal="<<certificate.feasibility.total_inf
            <<" original_dual="<<certificate.dual_infeasibility_inf
            <<" original_compl="<<certificate.complementarity_inf
            <<" original_input_unchanged=1 internal_point_constructed=1"
            <<" original_coefficients_verified=1 external_optimizer=0 global_optimum_claimed=0"
            <<" seconds="<<total<<std::endl;
        return certificate;
      }
    }
    auto selected=detail::resolve_standard_with_exact_presolve(problem,st_);
    const double policy_setup_seconds=std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();
    selected.settings.time_limit_seconds=std::max(0.,st_.time_limit_seconds-policy_setup_seconds);
    const double adaptive_absolute=std::chrono::duration<double>(adaptive_began.time_since_epoch()).count();
    auto adaptive_elapsed=[&](){return std::chrono::duration<double>(AdaptiveClock::now()-adaptive_began).count();};
    const bool adaptive=detail::adaptive_equality_probe_eligible(problem,st_,selected);
    AdaptiveEqualityProbeStatistics adaptive_stats;
    Settings historical_settings=st_;
    std::unique_ptr<Result> historical;
    auto audit_raw=[&](Result& point) {
      if(detail::adaptive_original_strict(problem,st_,point))return true;
      const auto audit_began=AdaptiveClock::now();
      Settings audit_settings=st_;audit_settings.rank_presolve_context=nullptr;
      audit_settings.accepted_checkpoint_callback={};audit_settings.acceptable_iterations=0;
      audit_settings.acceptable_on_failure=false;audit_settings.termination_test="project";
      audit_settings.feasibility_tolerance=std::min(1e-6,st_.feasibility_tolerance);
      audit_settings.dual_infeasibility_tolerance=std::min(1e-6,st_.dual_infeasibility_tolerance);
      audit_settings.complementarity_tolerance=std::min(1e-6,st_.complementarity_tolerance);
      point.termination_audit_valid=false;
      try {detail::audit_original_termination(problem,audit_settings,point);}
      catch(const std::exception&) {point.termination_audit_valid=false;}
      const double seconds=std::chrono::duration<double>(AdaptiveClock::now()-audit_began).count();
      adaptive_stats.selection_audit_seconds+=seconds;
      point.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=seconds;
      point.lazy_newton.certificate_seconds+=seconds;
      return detail::adaptive_original_strict(problem,st_,point);
    };
    auto select_historical_settings=[&](const char* reason) {
      selected.settings=historical_settings;selected.enabled=false;
      selected.mode="adaptive-historical-primary";selected.reason=reason;
    };
    if(adaptive) {
      adaptive_stats.eligible=adaptive_stats.attempted=true;
      adaptive_stats.probe_outer_budget=detail::adaptive_equality_probe_outer_budget(st_);
      adaptive_stats.global_outer_budget=st_.max_outer_iterations;
      adaptive_stats.global_restoration_budget=std::max(0,std::min(st_.max_outer_iterations,st_.restoration_max_iterations));
      adaptive_stats.global_seconds=st_.time_limit_seconds;
      historical_settings.standard_auto_resolved=true;
      historical_settings.standard_equality_watch=true;
      historical_settings.dca_engine="majorants";historical_settings.outer_algorithm="lazy-dca";
      historical_settings.max_outer_iterations=adaptive_stats.probe_outer_budget;
      historical_settings.restoration_max_iterations=std::min(adaptive_stats.probe_outer_budget,
          adaptive_stats.global_restoration_budget);
      historical_settings.restoration_shared_iteration_budget=true;
      historical_settings.time_limit_seconds=std::max(0.,st_.time_limit_seconds-adaptive_elapsed());
      std::cout<<"[ADAPTIVE-EQUALITY] event=historical-observation configuration="<<st_.convexification_study
          <<" outer_budget="<<historical_settings.max_outer_iterations
          <<" global_outer="<<st_.max_outer_iterations<<" global_seconds="<<st_.time_limit_seconds
          <<" trigger="<<(detail::original_diagonal_convex_equality_domain(problem)?
              "diagonal-convex-objective-equality-domain":"nonconstant-affine-objective")
          <<" source_QE=1 source_IQ_affine=1"
          <<" switch=failed-solve-or-measured-short-step-stagnation fixed_prefix=0"
          <<" original_frozen_start=1 same_configuration=1 AL_stage=0"<<std::endl;
      const auto phase_began=AdaptiveClock::now();
      detail::NativeAnchorStore native_anchors(problem,st_,
          adaptive_absolute+st_.time_limit_seconds,st_.convexification_study=="off");
      detail::NativeAnchorCaptureScope native_capture(native_anchors);
      historical=std::make_unique<Result>(Solver(historical_settings).solve(problem));
      adaptive_stats.probe_strict=audit_raw(*historical);
      detail::NativeAnchorFinishOutcome native_finish;
      if(!adaptive_stats.probe_strict) {
        const auto finish_began=AdaptiveClock::now();
        native_anchors.observe(problem,nullptr,historical->x,historical->feasibility.total_inf,
            false,detail::NativeAnchorOrigin::HistoricalReturnedState);
        native_finish=detail::native_original_anchor_finish(native_anchors,*historical,
            std::min(5.,std::max(0.,st_.time_limit_seconds-adaptive_elapsed())),
            std::max(0,st_.max_outer_iterations-historical->outer_iterations),1);
        const double finish_seconds=std::chrono::duration<double>(AdaptiveClock::now()-finish_began).count();
        // One outer attempt, no primal restoration iteration. Failed fits
        // retain all consumed time/sweeps and the unmodified prior endpoint.
        historical->outer_iterations+=native_finish.outer_charge;
        historical->multiplier_refit_attempts+=native_finish.attempted;
        historical->multiplier_refit_successes+=native_finish.accepted;
        historical->termination_multiplier_fit_sweeps+=native_finish.work.sweeps;
        historical->lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=finish_seconds;
        historical->lazy_newton.certificate_seconds+=finish_seconds;
        if(native_finish.accepted) {
          adaptive_stats.probe_strict=true;
          historical->termination_certificate_rescued=true;
        }
      }
      if(native_anchors.observations)std::cout<<"[NATIVE-ORIGINAL-ANCHOR-CAPTURE] observations="
          <<native_anchors.observations<<" full_original_primal_audits="<<native_anchors.primal_audits
          <<" ownership_refusals="<<native_anchors.ownership_refusals<<" stored="<<native_anchors.stored
          <<" observation_seconds="<<native_anchors.observation_seconds
          <<" same_execution=1 external_endpoint=0"<<std::endl;
      adaptive_stats.probe_seconds=std::chrono::duration<double>(AdaptiveClock::now()-phase_began).count();
      adaptive_stats.probe_outer=historical->outer_iterations;
      adaptive_stats.probe_restoration=historical->restoration_iterations;
      adaptive_stats.probe_factorizations=historical->kkt_factorizations;
      adaptive_stats.probe_status=status_string(historical->status);
      adaptive_stats.probe_raw_residual=native_finish.accepted?
          std::max({native_finish.original_primal,native_finish.original_dual,native_finish.original_complementarity}):
          detail::adaptive_original_raw_score(problem,*historical);
      adaptive_stats.probe_cost_seconds=historical->lazy_newton.cost_seconds;
      const bool probe_budget_ok=historical->outer_iterations<=historical_settings.max_outer_iterations&&
          historical->restoration_iterations<=historical_settings.restoration_max_iterations;
      if(!probe_budget_ok)adaptive_stats.probe_strict=false;
      if(adaptive_stats.probe_strict&&adaptive_elapsed()<=st_.time_limit_seconds) {
        historical->status=Status::Solved;
        if(historical->message.empty())historical->message="raw original QCQP KKT after observed historical primary";
        adaptive_stats.selected_phase="historical";
        adaptive_stats.reason="historical-original-KKT-no-switch";
        select_historical_settings("historical-original-KKT-no-switch");
      } else {
        auto& continuation=selected.settings;
        continuation.standard_equality_watch=false;
        continuation.max_outer_iterations=detail::adaptive_remaining_outer(st_,*historical);
        continuation.restoration_max_iterations=detail::adaptive_remaining_restoration(st_,*historical);
        continuation.restoration_shared_iteration_budget=true;
        continuation.time_limit_seconds=std::max(0.,st_.time_limit_seconds-adaptive_elapsed());
        if(historical->restoration_iterations>0||historical->restoration_total_seconds>0)
          continuation.restoration_deadline_seconds=std::min(continuation.restoration_deadline_seconds,adaptive_absolute+60.);
        adaptive_stats.stabilized_outer_budget=continuation.max_outer_iterations;
        adaptive_stats.stabilized_restoration_budget=continuation.restoration_max_iterations;
        adaptive_stats.transitioned=probe_budget_ok&&continuation.max_outer_iterations>0&&
            continuation.time_limit_seconds>1e-6;
        if(adaptive_stats.transitioned) {
          selected.mode="adaptive-stabilized-primary";
          selected.reason="historical-failure-or-observed-stagnation-GR-restart";
          std::cout<<"[ADAPTIVE-EQUALITY] event=stabilized-continuation probe_status="
              <<adaptive_stats.probe_status<<" consumed_outer="<<adaptive_stats.probe_outer
              <<" remaining_outer="<<continuation.max_outer_iterations
              <<" remaining_restoration="<<continuation.restoration_max_iterations
              <<" remaining_seconds="<<continuation.time_limit_seconds
              <<" original_frozen_start=1 same_configuration=1 AL_stage=0"<<std::endl;
        } else {
          adaptive_stats.selected_phase="historical";
          adaptive_stats.reason=probe_budget_ok?"global-budget-exhausted-after-prefix":"historical-prefix-work-limit-violated";
          if(adaptive_elapsed()>=st_.time_limit_seconds)historical->status=Status::TimeLimit;
          select_historical_settings("bounded-historical-prefix-global-budget-exhausted");
        }
      }
    }
    const auto& effective=selected.settings;
    std::cout<<"[STANDARD-AUTO] policy="<<st_.standard_r10_policy
        <<" mode="<<selected.mode<<" reason="<<selected.reason
        <<" requested_engine="<<st_.dca_engine<<" effective_engine="<<effective.dca_engine
        <<" configuration="<<(st_.convexification_study=="off"?"R3":st_.convexification_study)
        <<" stabilization="<<effective.majorant_equality_stabilization
        <<" GR="<<detail::original_stabilized_merit_requested(effective)
        <<" source_barrier="<<detail::original_boolean_mode("LAZY_ORIGINAL_SOURCE_PRIMAL_BARRIER",
            effective.original_source_primal_barrier||detail::original_stabilized_merit_requested(effective))
        <<" short_prox="<<detail::original_boolean_mode("LAZY_ORIGINAL_GR_SHORT_STEP_PROX",effective.original_gr_short_step_prox)
        <<" positive_bound_product="<<detail::original_boolean_mode("LAZY_ORIGINAL_BOUND_PRODUCT_PATH",effective.original_bound_product_path)
        <<" consistent_bound_repair="<<detail::original_boolean_mode("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_SLACKS",effective.original_gr_consistent_bound_slacks)
        <<" consistent_bound_trigger="<<effective.original_gr_consistent_bound_trigger
        <<" exact_global_faces="<<effective.original_exact_affine_faces
        <<" atomic_regularization="<<effective.lazy_newton_atomic_regularization
        <<" initial_mu="<<effective.lazy_newton_initial_mu<<" startup="<<effective.majorant_startup_policy
        <<" requested_startup="<<st_.majorant_startup_policy
        <<" rank_presolve="<<effective.dca_rank_presolve<<" requested_rank_presolve="<<st_.dca_rank_presolve
        <<" free_epigraph_presolve="<<effective.free_epigraph_presolve
        <<" requested_free_epigraph_presolve="<<st_.free_epigraph_presolve
        <<" checkpoint_requested="<<bool(st_.accepted_checkpoint_callback)
        <<" checkpoint_effective="<<bool(effective.accepted_checkpoint_callback)
        <<" builtin_checkpoint_skipped="<<(selected.enabled&&st_.accepted_checkpoint_builtin&&bool(st_.accepted_checkpoint_callback)&&!effective.accepted_checkpoint_callback)
        <<" full_source_rows="<<selected.enabled
        <<" input_x0_unchanged=1 warm_initializer=0"<<std::endl;
    bool gr_adjustments_active=detail::original_stabilized_merit_requested(effective)&&
        effective.majorant_equality_stabilization;
    std::cout<<"[GR-ADJUSTMENTS] active="<<gr_adjustments_active
        <<" constraint_increments="<<(effective.gr_coefficient_constraint_increments?"coefficients":"rounded")
        <<" proximal_policy="<<effective.gr_proximal_policy
        <<" fixed_centering_steps="<<effective.gr_fixed_centering_steps
        <<" residual_components="<<effective.gr_residual_components
        <<" equality_Hessian=exact Armijo_unchanged=1 original_KKT_unchanged=1"<<std::endl;
    Result result;
    if(!adaptive)result=Solver(effective).solve(problem);
    else if(!adaptive_stats.transitioned)result=std::move(*historical);
    else {
      const auto phase_began=AdaptiveClock::now();
      Result stabilized=Solver(effective).solve(problem);
      adaptive_stats.stabilized_strict=audit_raw(stabilized);
      adaptive_stats.stabilized_seconds=std::chrono::duration<double>(AdaptiveClock::now()-phase_began).count();
      adaptive_stats.stabilized_outer=stabilized.outer_iterations;
      adaptive_stats.stabilized_restoration=stabilized.restoration_iterations;
      adaptive_stats.stabilized_factorizations=stabilized.kkt_factorizations;
      adaptive_stats.stabilized_status=status_string(stabilized.status);
      adaptive_stats.stabilized_raw_residual=detail::adaptive_original_raw_score(problem,stabilized);
      adaptive_stats.stabilized_cost_seconds=stabilized.lazy_newton.cost_seconds;
      const bool within=stabilized.outer_iterations<=adaptive_stats.stabilized_outer_budget&&
          stabilized.restoration_iterations<=adaptive_stats.stabilized_restoration_budget&&
          adaptive_elapsed()<=st_.time_limit_seconds;
      adaptive_stats.stabilized_strict=adaptive_stats.stabilized_strict&&within;
      const bool choose_stabilized=within&&(adaptive_stats.stabilized_strict||
          adaptive_stats.stabilized_raw_residual<adaptive_stats.probe_raw_residual);
      if(choose_stabilized) {
        result=std::move(stabilized);
        detail::merge_rank_retry_work(result,*historical,false,true);
        adaptive_stats.selected_phase="stabilized";
        adaptive_stats.reason=adaptive_stats.stabilized_strict?"stabilized-raw-original-KKT":"best-audited-raw-residual-stabilized";
        if(adaptive_stats.stabilized_strict)result.status=Status::Solved;
      } else {
        result=std::move(*historical);
        detail::merge_rank_retry_work(result,stabilized,false,false);
        adaptive_stats.selected_phase="historical";
        adaptive_stats.reason=within?"best-audited-raw-residual-historical":"global-budget-exhausted-historical-retained";
        select_historical_settings("bounded-historical-prefix-endpoint-retained");
      }
      result.message="observed historical primary then stabilized primary: "+result.message;
    }
    if(adaptive) {
      adaptive_stats.total_seconds=adaptive_elapsed();
      adaptive_stats.within_global_budget=adaptive_stats.total_seconds<=st_.time_limit_seconds&&
          result.outer_iterations<=st_.max_outer_iterations&&
          result.restoration_iterations<=adaptive_stats.global_restoration_budget;
      if(!adaptive_stats.within_global_budget&&result.status==Status::Solved) {
        result.status=Status::TimeLimit;result.message="global primary budget exhausted in adaptive equality treatment";
      }
      result.adaptive_equality_probe=adaptive_stats;
      result.elapsed_seconds=adaptive_stats.total_seconds;
      result.lazy_newton.solver_wall_seconds=result.elapsed_seconds;
      gr_adjustments_active=detail::original_stabilized_merit_requested(effective)&&effective.majorant_equality_stabilization;
      std::cout<<"[ADAPTIVE-EQUALITY] event=end selected="<<adaptive_stats.selected_phase
          <<" probe_strict="<<adaptive_stats.probe_strict<<" stabilized_strict="<<adaptive_stats.stabilized_strict
          <<" cumulative_outer="<<result.outer_iterations<<" global_outer="<<st_.max_outer_iterations
          <<" cumulative_restoration="<<result.restoration_iterations
          <<" global_restoration="<<adaptive_stats.global_restoration_budget
          <<" total_seconds="<<result.elapsed_seconds
          <<" within_global_budget="<<adaptive_stats.within_global_budget<<" AL_stage=0"<<std::endl;
    }
    result.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=policy_setup_seconds;
    if(!adaptive){result.elapsed_seconds+=policy_setup_seconds;result.lazy_newton.solver_wall_seconds=result.elapsed_seconds;}
    result.gr_adjustments_active=gr_adjustments_active;
    result.gr_coefficient_constraint_increments=effective.gr_coefficient_constraint_increments;
    result.gr_proximal_policy=effective.gr_proximal_policy;
    result.gr_fixed_centering_steps=effective.gr_fixed_centering_steps;
    result.gr_residual_components=effective.gr_residual_components;
    result.standard_auto_mode=selected.mode;result.standard_auto_reason=selected.reason;
    result.standard_auto_requested_engine=st_.dca_engine;
    result.standard_auto_effective_engine=effective.dca_engine;
    result.standard_auto_configuration=st_.convexification_study=="off"?"R3":st_.convexification_study;
    result.standard_auto_enabled=selected.enabled;
    result.standard_auto_exact_global_faces=detail::original_boolean_mode("LAZY_EXACT_AFFINE_FACE_RESEARCH",effective.original_exact_affine_faces);
    result.standard_auto_rank_presolve_requested=st_.dca_rank_presolve;
    result.standard_auto_rank_presolve_effective=effective.dca_rank_presolve;
    result.standard_auto_free_epigraph_requested=st_.free_epigraph_presolve;
    result.standard_auto_free_epigraph_effective=effective.free_epigraph_presolve;
    result.standard_auto_rank_presolve_skipped=selected.enabled&&st_.dca_rank_presolve&&!effective.dca_rank_presolve;
    result.standard_auto_free_epigraph_presolve_skipped=selected.enabled&&st_.free_epigraph_presolve&&!effective.free_epigraph_presolve;
    result.standard_auto_checkpoint_requested=bool(st_.accepted_checkpoint_callback);
    result.standard_auto_checkpoint_effective=bool(effective.accepted_checkpoint_callback);
    result.standard_auto_builtin_checkpoint_skipped=selected.enabled&&st_.accepted_checkpoint_builtin&&
        bool(st_.accepted_checkpoint_callback)&&!effective.accepted_checkpoint_callback;
    result.standard_auto_requested_startup=st_.majorant_startup_policy;
    result.standard_auto_effective_startup=effective.majorant_startup_policy;
    result.standard_auto_configured_initial_mu=effective.lazy_newton_initial_mu;
    return result;
  }
  detail::validate_majorant_equality_stabilization_problem(problem,st_);
  const bool gr_merit_requested=detail::original_stabilized_merit_requested(st_);
  if(gr_merit_requested&&!st_.majorant_equality_stabilization)
    throw std::invalid_argument("GR merit requires explicit all-source constraint stabilization");
  // Research-only GLOBAL faces of the original feasible set, proved with
  // exact original coefficients. No tangent-model face or product branch is
  // fixed here. The nested solve is the same parameterized Lazy solver.
  const bool exact_faces=detail::original_boolean_mode("LAZY_EXACT_AFFINE_FACE_RESEARCH",st_.original_exact_affine_faces);
  if(!st_.original_exact_affine_faces_consumed&&exact_faces) {
    const auto phase_began=std::chrono::steady_clock::now();
    auto phase_elapsed=[&](){return std::chrono::duration<double>(
        std::chrono::steady_clock::now()-phase_began).count();};
    if(st_.native_augmented_repair||st_.product_definition_presolve||
        st_.outer_algorithm!="lazy-dca"||st_.dca_engine!="majorants"||
        st_.accepted_checkpoint_callback||st_.rank_presolve_context)
      throw std::invalid_argument("exact global face research requires original majorants, checkpoints off, global product substitution off and AL-repair off");
    auto discovery=detail::discover_homogeneous_affine_faces(problem,16384,
        [&](){return phase_elapsed()>=st_.time_limit_seconds;});
    std::vector<detail::ExactAffineFacePresolve::Certificate> certificates;
    for(const auto& c:discovery.certificates) {
      detail::ExactAffineFacePresolve::Certificate next;
      for(std::size_t i=0;i<c.rows.size();++i)next.emplace_back(c.rows[i],c.weights[i]);
      certificates.push_back(std::move(next));
    }
    std::unique_ptr<detail::ExactAffineFacePresolve> face_storage;
    try {
      face_storage=std::make_unique<detail::ExactAffineFacePresolve>(problem,certificates);
    } catch(const std::runtime_error& error) {
      // Automatic preprocessing is optional. An unsupported exact reduction
      // (including a zero-dimensional private model) keeps the original
      // problem and spends only the remaining common solve budget.
      if(!st_.original_exact_affine_faces)throw;
      const double prep=phase_elapsed();
      std::cout<<"[EXACT-AFFINE-FACE] supported=0 fallback=original-unreduced-majorants reason="
          <<std::quoted(error.what())<<" prep_seconds="<<prep
          <<" input_x0_unchanged=1 warm_initializer=0"<<std::endl;
      Settings inner=st_;inner.original_exact_affine_faces_consumed=true;
      inner.time_limit_seconds=std::max(0.,st_.time_limit_seconds-prep);
      Result r;
      if(inner.time_limit_seconds>0)r=Solver(inner).solve(problem);
      else {r.status=Status::TimeLimit;r.x=problem.x0;r.message="time limit during optional exact face preprocessing";}
      r.standard_auto_exact_global_faces_reason="unsupported-reduction-kept-original-model";
      r.elapsed_seconds=phase_elapsed();r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
      if(r.elapsed_seconds>=st_.time_limit_seconds)r.status=Status::TimeLimit;
      return r;
    }
    auto& face=*face_storage;
    std::cout<<"[EXACT-AFFINE-FACE] n="<<problem.n
        <<" reduced_n="<<(face.changed()?face.reduced.n:problem.n)
        <<" certificates="<<certificates.size()<<" initial_forced="<<face.initial_forced.size()
        <<" closure="<<face.zero_propagations.size()<<" removed_ineq="<<face.removed_inequalities
        <<" removed_eq="<<face.removed_equalities<<" projection_inf="<<face.projection_inf
        <<" prep_seconds="<<phase_elapsed()
        <<" exact=1 global_faces_only=1 optimizer=original-Lazy-majorants AL_repair=0"
        <<" supplied_start_preserved_before_presolve=1 projection_is_exact_presolve=1 warm_initializer=0"<<std::endl;
    Settings inner=st_;
    inner.original_exact_affine_faces_consumed=true;
    inner.time_limit_seconds=std::max(0.,st_.time_limit_seconds-phase_elapsed());
    Result r;
    const QCQPProblem& reduced=face.changed()?face.reduced:problem;
    if(inner.time_limit_seconds>0)r=Solver(inner).solve(reduced);
    else {
      r.status=Status::TimeLimit;r.x=reduced.x0;
      r.certificate_equality_multipliers.assign(reduced.equalities.size(),0);
      r.certificate_inequality_multipliers.assign(reduced.inequalities.size(),0);
      r.certificate_lower_multipliers.assign(r.x.size(),0);
      r.certificate_upper_multipliers.assign(r.x.size(),0);
    }
    if(face.changed())face.lift(r);
    r.standard_auto_exact_global_faces_applied=face.changed();
    r.standard_auto_exact_global_faces_reason=face.changed()?"certified-exact-original-face":"no-certified-reduction";
    r.standard_auto_exact_face_start_projection_inf=face.projection_inf;
    Settings audit_settings=st_;audit_settings.rank_presolve_context=nullptr;
    const bool audit_ok=detail::audit_original_termination(problem,audit_settings,r);
    const bool strict=audit_ok&&r.termination_audit_valid&&
        r.feasibility.total_inf<=st_.feasibility_tolerance&&
        r.dual_infeasibility_inf<=st_.dual_infeasibility_tolerance&&
        r.complementarity_inf<=st_.complementarity_tolerance;
    if(r.status==Status::Solved&&!strict) {
      r.status=Status::NumericalError;r.message="exact face original raw KKT audit failed";
    }
    r.message="exact global affine face + original Lazy majorants: "+r.message;
    r.elapsed_seconds=phase_elapsed();
    if(r.elapsed_seconds>=st_.time_limit_seconds)r.status=Status::TimeLimit;
    std::cout<<"[EXACT-AFFINE-FACE-AUDIT] strict="<<strict
        <<" primal="<<r.feasibility.total_inf<<" dual="<<r.dual_infeasibility_inf
        <<" comp="<<r.complementarity_inf<<" total_seconds="<<r.elapsed_seconds<<std::endl;
    return r;
  }
  if(st_.native_augmented_repair) {
    if(st_.rank_presolve_context||st_.accepted_checkpoint_callback||st_.termination_test!="project" ||
       st_.linear_backend!="mumps"||!st_.linear_metis)
      throw std::invalid_argument("native augmented repair requires top-level original project termination and MUMPS+METIS with checkpoints off");
    return detail::solve_native_augmented_repair(problem,st_);
  }
  if(st_.product_definition_presolve) {
    const bool majorants=st_.outer_algorithm=="lazy-dca"&&st_.dca_engine=="majorants";
    const bool atomic=st_.outer_algorithm=="lazy-newton"&&st_.ln_engine=="atomic"&&
        !st_.dca_rank_presolve&&!st_.accepted_checkpoint_callback;
    if(st_.rank_presolve_context || st_.termination_test!="project" || (!majorants&&!atomic))
      throw std::invalid_argument("experimental product-definition presolve requires top-level project majorants or atomic Newton with rank/checkpoint off");
    return detail::solve_with_product_definition_presolve(problem,st_);
  }
  if(st_.power2_scaling) {
    if(st_.rank_presolve_context||st_.dca_rank_presolve||st_.accepted_checkpoint_callback||st_.termination_test!="project")
      throw std::invalid_argument("experimental power2 scaling requires project termination with rank/checkpoint off");
    return detail::solve_with_power2_scaling(problem,st_);
  }
  if(st_.free_epigraph_presolve && st_.majorant_adaptive_globalization &&
     st_.convexification_study!="off" && st_.outer_algorithm=="lazy-dca" &&
     st_.dca_engine=="majorants" && !st_.rank_presolve_context && !problem.has_signed_square())
    return solve_with_free_epigraph_presolve(problem,st_);
  // The first GR diagnostic has no old-policy whole-solve cold retry.
  if(gr_merit_requested||st_.majorant_barrier_retry_consumed)return solve_once(problem);
  using Clock=std::chrono::steady_clock;
  const auto began=Clock::now();
  const double started_absolute=std::chrono::duration<double>(began.time_since_epoch()).count();
  Settings primary_st=st_;primary_st.majorant_barrier_retry_consumed=true;
  Result primary=Solver(primary_st).solve_once(problem);
  const auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-began).count();};
  const auto budget=detail::majorant_barrier_retry_budget(problem,st_,primary,elapsed(),started_absolute);
  if(!budget.allowed)return primary;

  Settings retry_st=st_;
  retry_st.majorant_barrier_retry_consumed=true;
  retry_st.majorant_complete_barrier_gate=true;
  retry_st.time_limit_seconds=budget.seconds;
  retry_st.max_outer_iterations=budget.outer;
  retry_st.restoration_max_iterations=budget.restoration;
  retry_st.restoration_shared_iteration_budget=true;
  retry_st.restoration_deadline_seconds=budget.restoration_deadline;
  if(st_.live_display_level)std::cout<<"[NATIVE-BARRIER-RETRY] event=begin primary_status="
      <<status_string(primary.status)<<" primary_primal="<<primary.feasibility.total_inf
      <<" primary_dual="<<primary.dual_infeasibility_inf
      <<" primary_outer="<<primary.outer_iterations<<" remaining_outer="<<budget.outer
      <<" remaining_restoration="<<budget.restoration<<" remaining_seconds="<<budget.seconds
      <<" configuration="<<st_.convexification_study
      <<" original_frozen_start=1 same_configuration=1 hidden_reference_fallback=0"
      <<" recursion_guard=1 complete_barrier_gate=retry-only original_filter_required=1"<<std::endl;
  const auto retry_began=Clock::now();
  Result retry=Solver(retry_st).solve_once(problem);
  // A child postsolve has already reconstructed every original multiplier.
  // Audit that final ORIGINAL point again before it can replace the primary.
  if(retry.status==Status::Solved) {
    const auto audit_began=Clock::now();
    Settings original_st=st_;original_st.rank_presolve_context=nullptr;
    original_st.majorant_barrier_retry_consumed=true;
    original_st.termination_test="project";
    retry.termination_scaling_available=false;
    detail::audit_original_termination(problem,original_st,retry);
    const double audit_seconds=std::chrono::duration<double>(Clock::now()-audit_began).count();
    retry.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=audit_seconds;
    retry.lazy_newton.certificate_seconds+=audit_seconds;
  }
  retry.elapsed_seconds=std::chrono::duration<double>(Clock::now()-retry_began).count();
  retry.lazy_newton.solver_wall_seconds=retry.elapsed_seconds;
  const bool accepted=detail::majorant_barrier_retry_strict(st_,retry,budget,elapsed());
  Result final=accepted?retry:primary;
  detail::set_native_barrier_retry_statistics(final,primary,retry,accepted);
  detail::merge_rank_retry_work(final,accepted?primary:retry,false,accepted);
  final.elapsed_seconds=elapsed();final.lazy_newton.solver_wall_seconds=final.elapsed_seconds;
  if(st_.live_display_level)std::cout<<"[NATIVE-BARRIER-RETRY] event=end accepted="<<accepted
      <<" primary_status="<<status_string(primary.status)<<" retry_status="<<status_string(retry.status)
      <<" final_status="<<status_string(final.status)<<" primary_endpoint_preserved="<<(!accepted)
      <<" cumulative_outer="<<final.outer_iterations<<" outer_limit="<<st_.max_outer_iterations
      <<" cumulative_restoration="<<final.restoration_iterations
      <<" restoration_limit="<<st_.restoration_max_iterations
      <<" cumulative_seconds="<<final.elapsed_seconds<<" time_limit="<<st_.time_limit_seconds
      <<" original_filter_required=1 original_KKT_required=1"<<std::endl;
  return final;
}

Result Solver::solve_once(const QCQPProblem& problem) {
  problem.validate();
  // The legacy signed-square elimination is nonlinear and creates piecewise
  // functions. A study of globally convexified quadratic functions must keep
  // the supplied exact QCQP lift; otherwise preprocessing itself manufactures
  // a model subsequently rejected by the study. The reference-off path keeps
  // its historical elimination and multiplier lift unchanged.
  if(!st_.majorant_equality_stabilization && !problem.has_signed_square() && st_.outer_algorithm=="lazy-dca" && st_.dca_engine=="majorants" &&
      st_.convexification_study=="off") {
    const auto begin=std::chrono::steady_clock::now();
    NativeSignedSquareReduction native(problem);
    if(native.changed()) {
      Settings inner=st_;inner.dca_rank_presolve=false;inner.rank_presolve_context=nullptr;
      const double preprocessing=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
      if(std::isfinite(inner.time_limit_seconds))inner.time_limit_seconds=std::max(0.,inner.time_limit_seconds-preprocessing);
      if(st_.live_display_level)std::cout<<"[NATIVE-SIGNED-SQUARE] links="<<native.links.size()<<" variables="<<problem.n<<"->"<<native.reduced.n<<" exact_initial_point=1"<<std::endl;
      Result r;
      if(inner.time_limit_seconds<=0) {
        r.status=Status::TimeLimit;r.message="time limit during native signed-square preprocessing";
        r.x=native.reduced.x0;
        r.certificate_equality_multipliers.assign(native.reduced.equalities.size(),0);
        r.certificate_inequality_multipliers.assign(native.reduced.inequalities.size(),0);
        r.certificate_lower_multipliers.assign(native.reduced.n,0);
        r.certificate_upper_multipliers.assign(native.reduced.n,0);
      } else r=Solver(inner).solve(native.reduced);
      const auto postsolve=std::chrono::steady_clock::now();
      bool lifted=true;
      try {native.lift(r);}
      catch(const std::runtime_error&) {
        lifted=false;
        Vec x=problem.x0;
        if(r.x.size()==native.variables.size()) {
          for(std::size_t j=0;j<native.variables.size();++j)x[native.variables[j]]=r.x[j];
          for(auto link:native.links)x[link.t]=std::abs(x[link.x]);
        }
        r.x=std::move(x);r.certificate_equality_multipliers.assign(problem.equalities.size(),0);
        r.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
        r.certificate_lower_multipliers.assign(problem.n,0);r.certificate_upper_multipliers.assign(problem.n,0);
        r.termination_scaling_available=false;r.termination_audit_valid=false;
        if(r.status==Status::Solved){r.status=Status::NumericalError;r.message="native signed-square postsolve certificate unavailable";}
      }
      r.objective=problem.objective.value(r.x);r.feasibility=feasibility(problem,r.x);
      const bool strict=detail::audit_original_termination(problem,st_,r)&&lifted;
      if(r.status==Status::Solved&&!strict){r.status=Status::NumericalError;r.message="native signed-square original lifted certificate failed";}
      else if(r.status==Status::Solved)r.message="original QCQP KKT certificate after native signed-square elimination and exact multiplier lift";
      r.termination_multiplier_source=lifted?"native-signed-square-exact-lift-original-audit":"native-signed-square-multipliers-unavailable";
      r.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=preprocessing;
      r.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-postsolve).count();
      r.elapsed_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
      r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
      if(st_.live_display_level)std::cout<<"[NATIVE-SIGNED-SQUARE-POSTSOLVE] strict="<<strict<<" primal="<<r.feasibility.total_inf<<" dual="<<r.dual_infeasibility_inf<<" compl="<<r.complementarity_inf<<std::endl;
      return r;
    }
  }
  if(problem.has_signed_square()&&(st_.outer_algorithm!="lazy-dca"||st_.dca_engine!="majorants"))throw std::invalid_argument("native signed-square model currently requires majorants engine");
  if(st_.dca_rank_presolve && !st_.rank_presolve_context && !problem.has_signed_square())
    return solve_with_rank_presolve(problem,st_);
  if(st_.outer_algorithm!="lazy-newton"&&st_.dca_engine!="filter"&&st_.dca_engine!="majorants") {
    bool compact=problem.objective.has_compact_linear();
    for(const auto& q:problem.equalities)compact=compact||q.has_compact_linear();
    for(const auto& q:problem.inequalities)compact=compact||q.has_compact_linear();
    if(compact){QCQPProblem dense=problem;dense.objective.materialize_linear();
      for(auto& q:dense.equalities)q.materialize_linear();
      for(auto& q:dense.inequalities)q.materialize_linear();
      return solve(dense);}
  }

  if(!std::isfinite(st_.dca_tangent_inertia_trigger) || st_.dca_tangent_inertia_trigger<=0.0)
    throw std::runtime_error("tangent inertia trigger must be positive and finite");
  if(st_.dca_equality_model=="tangent-inertia" &&
     (!st_.dca_single_loop_primal_dual ||
      !st_.dca_condensed_tangent_curvature ||
      !st_.dca_condensed_block_exact_curvature))
    throw std::runtime_error(
        "tangent-inertia requires single-loop, condensed curvature and block-exact fallback");
  if(st_.dca_single_loop_primal_dual) {
    if(st_.outer_algorithm!="lazy-dca")
      throw std::runtime_error(
          "single-loop primal-dual mode requires outer_algorithm=lazy-dca");
    if(st_.inner_backend!="lazyqp" && st_.inner_backend!="socp" &&
       st_.inner_backend!="lazyqp-sparse")
      throw std::runtime_error(
          "single-loop primal-dual mode requires inner_backend=lazyqp, lazyqp-sparse, or socp");
    if(!st_.qp_mehrotra || st_.dca_single_loop_inner_steps!=1)
      throw std::runtime_error(
          "single-loop primal-dual mode requires Mehrotra and exactly one inner step");
    if(st_.dca_single_loop_retraction_fallback_only)
      throw std::runtime_error(
          "single-loop primal-dual mode forbids auxiliary retraction fallback");
    if(st_.lazyqp_numeric_kkt_reuse)
      throw std::runtime_error(
          "single-loop primal-dual mode forbids numerical KKT reuse");
    bool has_nonfixed_finite_bound=false;
    for(int j=0;j<problem.n;++j) {
      const bool finite_lower=std::isfinite(problem.lb[j]);
      const bool finite_upper=std::isfinite(problem.ub[j]);
      const bool fixed=finite_lower && finite_upper &&
          std::abs(problem.ub[j]-problem.lb[j])<=st_.fixed_bound_tolerance;
      if(!fixed && (finite_lower || finite_upper)) {
        has_nonfixed_finite_bound=true;
        break;
      }
    }
    const bool has_inequality_route=!problem.inequalities.empty() ||
        has_nonfixed_finite_bound;
    if(!has_inequality_route)
      throw std::runtime_error(
          "single-loop primal-dual mode requires an inequality or nonfixed finite bound");
  }
  if (st_.inner_backend != "ipopt" && !linear_backend_compiled(st_.linear_backend))
    throw std::runtime_error("linear backend not compiled: " + st_.linear_backend);

  if ((st_.outer_algorithm=="lazy-dca"||st_.outer_algorithm=="auto")&&
      (st_.dca_engine=="filter"||st_.dca_engine=="majorants")) {
    // The historical terminal bounded-origin certificate trial is charged
    // one restoration unit below. Preserve that opportunity when the SAME
    // candidate is already original-feasible, instead of letting an
    // unsuccessful primary Phase I consume its entire shared allowance.
    const auto reservation_begin=std::chrono::steady_clock::now();
    const bool reserve_terminal=st_.dca_engine=="majorants"&&st_.restoration_enabled&&
        st_.restoration_shared_iteration_budget&&st_.restoration_max_iterations>0&&
        !detail::original_stabilized_merit_requested(st_)&&
        detail::bounded_origin_reservation_candidate(problem,st_.feasibility_tolerance);
    const double reservation_seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-reservation_begin).count();
    Settings primary_settings=st_;
    primary_settings.restoration_max_iterations=detail::primary_restoration_allowance(
        st_.restoration_shared_iteration_budget,st_.restoration_max_iterations,reserve_terminal);
    primary_settings.time_limit_seconds=std::max(0.,st_.time_limit_seconds-reservation_seconds);
    if(reserve_terminal&&st_.live_display_level!=0)
      std::cout<<"[RESTORATION-BUDGET] phase=bounded-origin action=reserve"
          <<" primary_limit="<<primary_settings.restoration_max_iterations
          <<" terminal_limit=1 shared_limit="<<st_.restoration_max_iterations
          <<" original_primal_admission_only=1 supplied_start_unchanged=1"<<std::endl;
    Result out;
    if(primary_settings.time_limit_seconds>0)out=solve_filter_dca(problem,primary_settings);
    else {out.status=Status::TimeLimit;out.x=problem.x0;
      out.message="time limit during terminal restoration budget admission";}
    out.elapsed_seconds+=reservation_seconds;
    out.lazy_newton.solver_wall_seconds=out.elapsed_seconds;
    out.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=reservation_seconds;
    // GR diagnostic keeps its accepted primal trajectory. The engine has
    // already audited that endpoint; terminal candidates that replace the
    // primal point are not part of this merit-globalized variant.
    if(detail::original_stabilized_merit_requested(st_))return out;
    const bool recoverable=out.status==Status::InnerFailure||
        out.status==Status::FactorizationFailure||out.status==Status::LineSearchFailure;
    const bool terminal_restoration_requested=st_.dca_engine=="majorants"&&
        st_.restoration_enabled&&recoverable&&out.elapsed_seconds<st_.time_limit_seconds;
    const bool terminal_restoration_available=detail::restoration_step_available(
        st_.restoration_shared_iteration_budget,st_.restoration_max_iterations,
        out.restoration_iterations);
    if(terminal_restoration_requested&&!terminal_restoration_available&&st_.live_display_level!=0)
      std::cout<<"[RESTORATION-BUDGET] phase=bounded-origin action=skip"
          <<" consumed="<<out.restoration_iterations<<" limit="<<st_.restoration_max_iterations
          <<" endpoint_preserved=1 work_counter_unchanged=1"<<std::endl;
    if(terminal_restoration_requested&&terminal_restoration_available) {
      // A terminal restoration candidate, accepted only by an independent
      // original KKT certificate. Successful filtered trajectories are intact.
      // Its adjoint factorization uses the requested backend and no fallback.
      const auto recovery_start=std::chrono::steady_clock::now();
      const double remaining=st_.time_limit_seconds-out.elapsed_seconds;
      const auto recovery=detail::bounded_origin_certificate(problem,st_,out,remaining);
      const double recovery_seconds=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-recovery_start).count();
      out.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=recovery_seconds;
      if(recovery.attempted) {
        if(!out.restoration_iterations)out.restoration_initial_feasibility=recovery.initial_primal;
        ++out.restoration_iterations;
        out.restoration_linear_solves+=recovery.solves;
        out.lazy_newton.linear_solves+=recovery.solves;
        out.lazy_newton.factor_attempts+=recovery.factorizations;
        out.kkt_factorizations+=recovery.factorizations;
        if(recovery.accepted)out.restoration_final_feasibility=recovery.primal;
        if(!recovery.accepted&&recovery_seconds>=remaining) {
          out.original_failure_status=status_string(out.status);
          out.original_failure_message=out.message;
          out.status=Status::TimeLimit;
          out.message="time limit during bounded-origin certificate recovery";
        }
        if(st_.live_display_level!=0)std::cout<<"[BOUNDED-ORIGIN-CERTIFICATE] accepted="
            <<recovery.accepted<<" backend="<<recovery.backend
            <<" supported_columns="<<recovery.supported_columns<<" equalities="<<recovery.equalities
            <<" factors="<<recovery.factorizations<<" solves="<<recovery.solves
            <<" adjoint_residual="<<recovery.expanded_relative_residual
            <<" primal="<<recovery.primal<<" dual="<<recovery.dual
            <<" compl="<<recovery.complementarity<<" seconds="<<recovery_seconds
            <<" message="<<recovery.message<<std::endl;
      }
      const double total_recovery=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-recovery_start).count();
      out.lazy_newton.cost_seconds[int(NewtonCost::Diagnostics)]+=total_recovery-recovery_seconds;
      out.elapsed_seconds+=total_recovery;
      out.lazy_newton.solver_wall_seconds=out.elapsed_seconds;
    }
    // Terminal rescue for the precise feasible-boundary barrier failure only.
    // It is after all historical recovery, so successful trajectories and the
    // old successful bounded-origin certificates do not enter this branch.
    if(st_.dca_engine=="majorants" && st_.restoration_enabled &&
       ((out.status==Status::InnerFailure &&
         (out.dca_filter.majorant_reject_precision>0 || out.message.find("non-finite")!=std::string::npos)) ||
        (out.status==Status::LineSearchFailure && out.dca_filter.majorant_last_rejection=="barrier-filter" &&
         std::isfinite(out.complementarity_inf) && out.complementarity_inf>st_.complementarity_tolerance)) &&
       out.elapsed_seconds<st_.time_limit_seconds) {
      const auto began=std::chrono::steady_clock::now();
      const double remaining=st_.time_limit_seconds-out.elapsed_seconds;
      const auto fit=detail::feasible_boundary_certificate(problem,st_,out,remaining);
      const double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
      out.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=spent;
      out.elapsed_seconds+=spent;out.lazy_newton.solver_wall_seconds=out.elapsed_seconds;
      if(fit.attempted) {
        ++out.multiplier_refit_attempts;
        if(fit.accepted)++out.multiplier_refit_successes;
        out.lazy_newton.linear_solves+=fit.solves;
        out.lazy_newton.factor_attempts+=fit.factorizations;out.kkt_factorizations+=fit.factorizations;
        if(st_.live_display_level!=0)std::cout<<"[FEASIBLE-BOUNDARY-CERTIFICATE] attempted=1 accepted="
          <<fit.accepted<<" fixed_primal=1 backend="<<fit.backend
          <<" supported_columns="<<fit.supported_columns<<" equalities="<<fit.equalities
          <<" factors="<<fit.factorizations<<" solves="<<fit.solves
          <<" adjoint_residual="<<fit.expanded_relative_residual
          <<" primal="<<fit.primal<<" dual="<<fit.dual<<" compl="<<fit.complementarity
          <<" seconds="<<spent<<" message="<<fit.message<<std::endl;
      }
    }
    return out;
  }

  if (st_.outer_algorithm == "lazy-newton") {
    // v0.4.28r3p28: lazy primal-dual Newton route (curvature-aware equalities).
    if(st_.ln_engine=="atomic") { auto out=solve_atomic_newton(problem,st_); out.atomic_engine_used=true; return out; }
    return solve_lazy_newton(problem, st_);
  }
  if (st_.outer_algorithm == "auto") {
    // v0.4.14: one scientific route.  "auto" is a compatibility alias and
    // must never switch the outer algorithm based on problem structure.
    Settings routed = st_;
    routed.outer_algorithm = "lazy-dca";
    if (st_.verbose) {
      std::cout << "[ALGORITHM-ROUTE] requested=auto selected=lazy-dca"
                << " reason=single-route-linearized-equalities"
                << " n=" << problem.n << "\n";
    }
    Result routed_result = Solver(routed).solve(problem);
    routed_result.effective_outer_algorithm = "lazy-dca";
    return routed_result;
  }

  Result result;
  result.effective_outer_algorithm = st_.outer_algorithm;
  if(st_.outer_algorithm=="lazy-dca") result.lazy_dca_equality_model=st_.dca_equality_model;
  const std::string linear_label =
      (st_.linear_backend == "ldl" ?
          (st_.kkt_persistent_symbolic_cache
              ? "suitesparse_ldl_metis_persistent_symbolic"
              : "suitesparse_ldl_metis") :
      (st_.linear_backend == "qdldl" ? "qdldl_metis" :
       (st_.linear_backend == "cholesk" ? "cholesk_qdldlt_metis" :
       (st_.linear_backend == "mumps" ? "mumps_sym_indef_metis" : (st_.linear_backend == "ma27" ? (st_.linear_metis ? "ma27_sym_indef_metis" : "ma27_sym_indef_internal") : "native_dense_lapack")))));

  result.linear_solver_backend = (st_.inner_backend == "ipopt") ? "ipopt" :
      (st_.inner_backend + "/" + linear_label);
  result.x = problem.x0;
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  };

  try {
    const ProblemGradientSparsity gradient_sparsity(problem);
    const ScopedProblemGradientSparsity gradient_sparsity_scope(
        gradient_sparsity);
    Vec x = problem.x0;

    // ----------------------------- Phase I -----------------------------
    bool direct_start_accepted=false;
    DirectStartEvaluation direct_start;
    const bool direct_start_requested=st_.restoration_enabled &&
        st_.outer_algorithm=="lazy-dca" &&
        st_.dca_single_loop_primal_dual &&
        st_.dca_single_loop_direct_start;
    if(direct_start_requested) {
      ++result.dca_single_loop_direct_start_attempts;
      direct_start=evaluate_direct_start(problem,x);
      if(direct_start.finite) {
        result.dca_single_loop_direct_start_initial_theta=
            direct_start.feasibility.total_inf;
        result.dca_single_loop_direct_start_violated_inequalities=
            static_cast<int>(direct_start.violated_inequalities.size());
      }
      const bool bounds_eligible=direct_start.finite &&
          direct_start.feasibility.bound_inf<=st_.feasibility_tolerance;
      // This bypass is a side-constraint recovery route, not a replacement for
      // nonlinear-equality restoration.  An equality-infeasible start must
      // retain Phase I even when its total residual happens to lie below the
      // side theta cap.
      const bool equalities_eligible=direct_start.finite &&
          direct_start.feasibility.equality_inf<=st_.feasibility_tolerance;
      const bool theta_eligible=direct_start.finite &&
          direct_start.feasibility.total_inf>st_.feasibility_tolerance &&
          direct_start.feasibility.total_inf<=
              st_.dca_single_loop_direct_start_theta_cap;
      direct_start_accepted=
          bounds_eligible && equalities_eligible && theta_eligible;
      if(direct_start_accepted) {
        ++result.dca_single_loop_direct_start_acceptances;
        result.dca_single_loop_direct_start_recovery_max_theta=
            direct_start.feasibility.total_inf;
        if(st_.verbose)
          std::cout << "[PHASE-I-DIRECT-START] accepted=1 theta="
                    << std::scientific
                    << direct_start.feasibility.total_inf
                    << " bound=" << direct_start.feasibility.bound_inf
                    << " cap=" << st_.dca_single_loop_direct_start_theta_cap
                    << " violated-inequalities="
                    << direct_start.violated_inequalities.size()
                    << "\n" << std::defaultfloat;
      } else {
        if(!direct_start.finite)
          ++result.dca_single_loop_direct_start_rejections_nonfinite;
        else if(!bounds_eligible)
          ++result.dca_single_loop_direct_start_rejections_bounds;
        else if(!equalities_eligible)
          ++result.dca_single_loop_direct_start_rejections_equalities;
        else
          ++result.dca_single_loop_direct_start_rejections_theta;
        if(st_.verbose)
          std::cout << "[PHASE-I-DIRECT-START] accepted=0 reason="
                    << (!direct_start.finite ? "nonfinite"
                        : (!bounds_eligible ? "bounds"
                           : (!equalities_eligible ? "equalities" : "theta")))
                    << " theta=" << std::scientific
                    << direct_start.feasibility.total_inf
                    << " bound=" << direct_start.feasibility.bound_inf
                    << " cap=" << st_.dca_single_loop_direct_start_theta_cap
                    << "\n" << std::defaultfloat;
      }
    }
    if (st_.restoration_enabled && !direct_start_accepted &&
        structural_feasibility(problem,x).total_inf>
            st_.feasibility_tolerance) {
      Settings restoration_settings=st_;
      if(st_.dca_equality_model=="elastic-split")
        restoration_settings.restoration_tangent_activation=false;
      FeasibilityRestoration restoration(restoration_settings);
      RestorationResult rr = restoration.solve(problem,x);
      result.restoration_iterations = rr.iterations;
      result.restoration_restarts = rr.restarts;
      result.restoration_linear_solves = rr.linear_solves;
      result.restoration_exact_hessian_solves = rr.exact_hessian_solves;
      result.restoration_matrix_free_solves = rr.matrix_free_solves;
      result.restoration_row_space_solves = rr.row_space_solves;
      result.restoration_harmonic_attempts = rr.harmonic_attempts;
      result.restoration_harmonic_acceptances = rr.harmonic_acceptances;
      result.restoration_harmonic_factorizations = rr.harmonic_factorizations;
      result.restoration_tangent_activation_attempts = rr.tangent_activation_attempts;
      result.restoration_tangent_activation_acceptances = rr.tangent_activation_acceptances;
      result.restoration_tangent_activation_components = rr.tangent_activation_components;
      result.restoration_cg_iterations = rr.cg_iterations;
      result.restoration_exact_workspace_growths = rr.exact_workspace_growths;
      result.restoration_exact_matvecs = rr.exact_matvecs;
      result.restoration_exact_parallel_row_dot_batches =
          rr.exact_parallel_row_dot_batches;
      result.restoration_exact_parallel_teams = rr.exact_parallel_teams;
      result.restoration_exact_sparse_row_dot_batches =
          rr.exact_sparse_row_dot_batches;
      result.restoration_exact_sparse_row_dot_verifications =
          rr.exact_sparse_row_dot_verifications;
      result.restoration_exact_sparse_row_dot_fallbacks =
          rr.exact_sparse_row_dot_fallbacks;
      result.restoration_line_search_backtracks = rr.line_search_backtracks;
      result.restoration_stagnation_events = rr.stagnation_events;
      result.restoration_minimax_qp_solves = rr.minimax_qp_solves;
      result.restoration_minimax_inner_iterations = rr.minimax_inner_iterations;
      result.restoration_minimax_factorizations = rr.minimax_factorizations;
      result.restoration_minimax_accepted_steps = rr.minimax_accepted_steps;
      result.restoration_minimax_line_search_backtracks = rr.minimax_line_search_backtracks;
      result.restoration_structured_abs_links = rr.structured_abs_links;
      result.restoration_effective_dimension = rr.effective_dimension;
      result.restoration_initial_feasibility = rr.initial_feasibility;
      result.restoration_harmonic_feasibility = rr.harmonic_feasibility;
      result.restoration_tangent_activation_feasibility = rr.tangent_activation_feasibility;
      result.restoration_tangent_activation_step_inf = rr.tangent_activation_step_inf;
      result.restoration_harmonic_seconds = rr.harmonic_seconds;
      result.restoration_tangent_activation_seconds = rr.tangent_activation_seconds;
      result.restoration_legacy_gate_seconds = rr.legacy_gate_seconds;
      result.restoration_exact_seconds = rr.exact_restoration_seconds;
      result.restoration_minimax_seconds = rr.minimax_seconds;
      result.restoration_restart_seconds = rr.restart_seconds;
      result.restoration_total_seconds = rr.total_seconds;
      result.restoration_tangent_activation_gradient_rows = rr.tangent_activation_gradient_rows;
      result.restoration_tangent_activation_line_quadratics = rr.tangent_activation_line_quadratics;
      result.restoration_tangent_activation_candidates = rr.tangent_activation_candidates;
      result.restoration_harmonic_unknowns = rr.harmonic_unknowns;
      result.restoration_harmonic_graph_edges = rr.harmonic_graph_edges;
      result.restoration_final_feasibility = rr.final_feasibility;
      if (st_.verbose) {
        std::cout << "[PHASE-I] theta=" << std::scientific << rr.initial_feasibility
                  << " -> " << rr.final_feasibility
                  << " iterations=" << std::defaultfloat << rr.iterations
                  << " restarts=" << rr.restarts << "\n";
      }
      x = rr.x;
      if (!rr.success) {
        result.status = Status::RestorationFailure;
        result.message = rr.message;
        result.x = x;
        result.objective = structural_quadratic_value(problem.objective,x);
        result.feasibility = structural_feasibility(problem,x);
        result.elapsed_seconds = elapsed();
        return result;
      }
    }

    bool has_quadratic_equality=false;
    for(const auto& h:problem.equalities) {
      for(const auto& e:h.hessian.entries) {
        if(std::abs(e.v)>0.0) {
          has_quadratic_equality=true;
          break;
        }
      }
      if(has_quadratic_equality) break;
    }
    const bool requested_condensed_equality_model=
        st_.outer_algorithm=="lazy-dca" && has_quadratic_equality &&
        (st_.dca_equality_model=="condensed-retraction" ||
         st_.dca_equality_model=="tangent-inertia");

    CurvatureOracle curvature(st_);
    // Isolate the large condensed-model oracle from the objective/inequality
    // oracle so its randomized Lanczos sequence cannot perturb their legacy
    // trajectory.  Reuse the condensed route's explicitly configured Krylov
    // budget rather than the generic curvature defaults.
    Settings condensed_curvature_settings=st_;
    condensed_curvature_settings.lanczos_steps=st_.dca_condensed_lanczos_steps;
    condensed_curvature_settings.lanczos_starts=st_.dca_condensed_lanczos_starts;
    CurvatureOracle condensed_curvature(condensed_curvature_settings);
    Convexification objective_cvx;
    // The ordinary Lazy-DCA routes need an ambient-space convexification of
    // Q0.  The condensed-retraction route deliberately skips it: there the
    // signed equality-Lagrangian Hessian is handled by its dedicated
    // tangent/ambient-large oracle rather than by convexifying Q0 alone.
    if(st_.outer_algorithm=="lazy-dca" && !requested_condensed_equality_model)
      curvature.ensure_convex(problem.objective.hessian,objective_cvx,0.0,
                              st_.max_objective_rank);
    std::vector<Convexification> ineq_cvx(problem.inequalities.size());
    std::set<int> working_set;
    std::vector<int> working_set_drop_streak(problem.inequalities.size(),0);
    bool objective_is_affine=true;
    for(const auto& e:problem.objective.hessian.entries)
      if(std::abs(e.v)>st_.curvature_tolerance) { objective_is_affine=false; break; }
    // The v0.4.6 full-majorant guard is essential when an omitted inequality
    // is the only object that bounds an affine descent direction (A4X12-like
    // models).  It is needlessly expensive for OPF models whose nonzero cost
    // coefficients belong to box-bounded generation variables.  This exact
    // sign test is a sufficient certificate that min c^T x is bounded by the
    // simple box alone; zero-cost, unbounded lift variables are harmless.
    bool affine_descent_box_bounded=objective_is_affine;
    if(affine_descent_box_bounded) {
      for(int j=0;j<problem.n;++j) {
        const double c=problem.objective.linear[static_cast<std::size_t>(j)];
        if((c>0.0 && !std::isfinite(problem.lb[static_cast<std::size_t>(j)])) ||
           (c<0.0 && !std::isfinite(problem.ub[static_cast<std::size_t>(j)]))) {
          affine_descent_box_bounded=false;
          break;
        }
      }
    }
    // The certificate is valid for the production condensed and linearized
    // routes.  Keep the experimental elastic-split route on its historical
    // full-majorant policy until it has an equivalent regression contract.
    const bool affine_box_lazy_route = st_.dca_affine_box_bounded_lazy &&
        affine_descent_box_bounded && has_quadratic_equality &&
        (st_.dca_equality_model=="condensed-retraction" ||
         st_.dca_equality_model=="tangent-inertia" ||
         st_.dca_equality_model=="linearized");
    const bool full_affine_majorant = st_.outer_algorithm=="lazy-dca" &&
        st_.dca_full_constraints_for_affine_objective && objective_is_affine &&
        !affine_box_lazy_route &&
        st_.dca_affine_full_constraint_limit>=0 &&
        static_cast<int>(problem.inequalities.size())<=st_.dca_affine_full_constraint_limit;
    for (int i = 0; i < static_cast<int>(problem.inequalities.size()); ++i) {
      if (full_affine_majorant ||
          structural_quadratic_value(problem.inequalities[i],x) >=
              -st_.working_set_activation) {
        working_set.insert(i);
        curvature.ensure_convex(problem.inequalities[i].hessian, ineq_cvx[i], 0.0,
                                st_.max_rank_per_quadratic);
      }
    }

    // The direct start is intentionally allowed to be slightly infeasible.
    // Seed every originally violated side row before the first majorant so no
    // violation can hide behind the ordinary near-active heuristic.  This is
    // redundant for the usual positive activation radius but is an explicit
    // invariant of the Phase-I bypass, independent of that tuning knob.
    if(direct_start_accepted) {
      for(int i:direct_start.violated_inequalities) {
        working_set.insert(i);
        curvature.ensure_convex(
            problem.inequalities[static_cast<std::size_t>(i)].hessian,
            ineq_cvx[static_cast<std::size_t>(i)],0.0,
            st_.max_rank_per_quadratic);
      }
      result.dca_single_loop_direct_start_working_set_rows=
          static_cast<int>(direct_start.violated_inequalities.size());
    }

    std::vector<FilterEntry> filter;
    if(direct_start_accepted) {
      filter_insert(filter,direct_start.feasibility.total_inf,
                    direct_start.objective);
      ++result.dca_single_loop_direct_start_filter_initializations;
    }
    double proximal = st_.proximal_regularization;
    const bool adaptive_kkt = st_.kkt_adaptive_regularization &&
        (st_.linear_backend=="native" || st_.linear_backend=="ldl" ||
         st_.linear_backend=="qdldl" || st_.linear_backend=="cholesk" ||
         st_.linear_backend=="mumps");
    double numerical_rho_x = adaptive_kkt
        ? std::max(st_.kkt_rho_x_floor,st_.kkt_primal_regularization)
        : st_.kkt_primal_regularization;
    int numerical_clean_outer=0;
    double current_lazyqp_forcing_target=0.0;
    double current_lazyqp_forcing_primal_target=0.0;
    auto inner_settings = [&](bool allow_forcing=false) {
      Settings si=st_;
      if(adaptive_kkt) si.kkt_primal_regularization=numerical_rho_x;
      // A forcing target is meaningful only for the primary Lazy-DCA
      // majorant at the current original-QCQP point.  Restoration, SOC and
      // local corrector models call this lambda without allow_forcing and
      // therefore retain the historical strict inner contract.
      si.lazyqp_forcing_target_tolerance=allow_forcing
          ? current_lazyqp_forcing_target : 0.0;
      si.lazyqp_forcing_primal_tolerance=allow_forcing
          ? current_lazyqp_forcing_primal_target : 0.0;
      return si;
    };
    auto absorb_kkt_diagnostics = [&](const InnerResult& ir) {
      result.equality_qp_original_kkt_gmres_attempts +=
          ir.equality_qp_original_kkt_gmres_attempts;
      result.equality_qp_original_kkt_gmres_acceptances +=
          ir.equality_qp_original_kkt_gmres_acceptances;
      result.equality_qp_original_kkt_gmres_iterations +=
          ir.equality_qp_original_kkt_gmres_iterations;
      if(!adaptive_kkt) return;
      numerical_rho_x=std::max(numerical_rho_x,ir.final_kkt_rho_x);
      result.final_kkt_rho_x=numerical_rho_x;
      result.max_kkt_rho_x=std::max(result.max_kkt_rho_x,ir.max_kkt_rho_x);
      result.final_kkt_delta_d=ir.final_kkt_delta_d;
      result.max_kkt_delta_d=std::max(result.max_kkt_delta_d,ir.max_kkt_delta_d);
      result.max_kkt_backward_error=std::max(result.max_kkt_backward_error,ir.max_kkt_backward_error);
      result.max_kkt_residual_inf=std::max(result.max_kkt_residual_inf,ir.max_kkt_residual_inf);
      result.kkt_regularization_retries += ir.kkt_regularization_retries;
      result.kkt_refinement_steps += ir.kkt_refinement_steps;
    };
    auto absorb_lazyqp_efficiency = [&](const InnerResult& ir) {
      if(ir.dual_warm_start_used) ++result.inner_dual_warm_start_acceptances;
      if(ir.full_dual_warm_start_used) ++result.inner_full_dual_warm_start_acceptances;
      if(ir.dual_warm_start_rejected) ++result.inner_dual_warm_start_rejections;
      result.lazyqp_zero_dual_continuation_attempts +=
          ir.lazyqp_zero_dual_continuation_attempts;
      result.lazyqp_zero_dual_continuation_mapped_attempts +=
          ir.lazyqp_zero_dual_continuation_mapped_attempts;
      result.lazyqp_zero_dual_continuation_direct_attempts +=
          ir.lazyqp_zero_dual_continuation_direct_attempts;
      result.lazyqp_zero_dual_continuation_acceptances +=
          ir.lazyqp_zero_dual_continuation_acceptances;
      result.lazyqp_zero_dual_continuation_fallbacks +=
          ir.lazyqp_zero_dual_continuation_fallbacks;
      result.lazyqp_zero_dual_continuation_repairs +=
          ir.lazyqp_zero_dual_continuation_repairs;
      if(ir.primal_warm_start_used) ++result.working_set_primal_transports;
      result.lazyqp_reduced_kkt_factorizations += ir.lazyqp_reduced_kkt_factorizations;
      result.lazyqp_full_kkt_factorizations += ir.lazyqp_full_kkt_factorizations;
      result.lazyqp_reduced_kkt_fallbacks += ir.lazyqp_reduced_kkt_fallbacks;
      result.lazyqp_hybrid_kkt_attempts += ir.lazyqp_hybrid_kkt_attempts;
      result.lazyqp_hybrid_kkt_factorizations +=
          ir.lazyqp_hybrid_kkt_factorizations;
      result.lazyqp_hybrid_kkt_fallbacks += ir.lazyqp_hybrid_kkt_fallbacks;
      result.lazyqp_hybrid_kkt_retained_rows_total +=
          ir.lazyqp_hybrid_kkt_retained_rows_total;
      result.lazyqp_hybrid_kkt_retained_rows_max = std::max(
          result.lazyqp_hybrid_kkt_retained_rows_max,
          ir.lazyqp_hybrid_kkt_retained_rows_max);
      result.lazyqp_hybrid_kkt_outer_product_triplets_avoided +=
          ir.lazyqp_hybrid_kkt_outer_product_triplets_avoided;
      result.lazyqp_hybrid_kkt_border_triplets_emitted +=
          ir.lazyqp_hybrid_kkt_border_triplets_emitted;
      result.lazyqp_numeric_kkt_reuse_attempts +=
          ir.lazyqp_numeric_kkt_reuse_attempts;
      result.lazyqp_numeric_kkt_reuse_acceptances +=
          ir.lazyqp_numeric_kkt_reuse_acceptances;
      result.lazyqp_numeric_kkt_reuse_fallbacks +=
          ir.lazyqp_numeric_kkt_reuse_fallbacks;
      result.lazyqp_numeric_kkt_reuse_iterations +=
          ir.lazyqp_numeric_kkt_reuse_iterations;
      result.lazyqp_max_kkt_dimension=std::max(result.lazyqp_max_kkt_dimension,
                                              ir.lazyqp_kkt_dimension);
      result.lazyqp_max_full_kkt_dimension=std::max(result.lazyqp_max_full_kkt_dimension,
                                                   ir.lazyqp_full_kkt_dimension);
      result.lazyqp_affine_elimination_attempts +=
          ir.lazyqp_affine_elimination_attempts;
      result.lazyqp_affine_elimination_acceptances +=
          ir.lazyqp_affine_elimination_acceptances;
      result.lazyqp_affine_elimination_fallbacks +=
          ir.lazyqp_affine_elimination_fallbacks;
      result.lazyqp_affine_eliminated_variables = std::max(
          result.lazyqp_affine_eliminated_variables,
          ir.lazyqp_affine_eliminated_variables);
      result.lazyqp_affine_eliminated_equalities = std::max(
          result.lazyqp_affine_eliminated_equalities,
          ir.lazyqp_affine_eliminated_equalities);
      result.lazyqp_affine_reduced_variables = std::max(
          result.lazyqp_affine_reduced_variables,
          ir.lazyqp_affine_reduced_variables);
      result.lazyqp_affine_reduced_equalities = std::max(
          result.lazyqp_affine_reduced_equalities,
          ir.lazyqp_affine_reduced_equalities);
      result.socp_formulation_attempts += ir.socp_formulation_attempts;
      result.socp_formulation_builds += ir.socp_formulation_builds;
      result.socp_formulation_rejections += ir.socp_formulation_rejections;
      result.socp_formulation_audit_failures +=
          ir.socp_formulation_audit_failures;
      result.socp_objective_rotated_cones +=
          ir.socp_objective_rotated_cones;
      result.socp_inequality_rotated_cones +=
          ir.socp_inequality_rotated_cones;
      result.socp_linear_nonnegative_cones +=
          ir.socp_linear_nonnegative_cones;
      result.socp_max_affine_equalities = std::max(
          result.socp_max_affine_equalities, ir.socp_affine_equalities);
      result.socp_max_explicit_lift_cone_dimension = std::max(
          result.socp_max_explicit_lift_cone_dimension,
          ir.socp_explicit_lift_cone_dimension);
      result.socp_condensed_newton_iterations +=
          ir.socp_condensed_newton_iterations;
      result.socp_max_rotated_cone_violation = std::max(
          result.socp_max_rotated_cone_violation,
          ir.socp_max_rotated_cone_violation);
      result.kkt_symbolic_reuses += ir.symbolic_reuses;
      result.kkt_symbolic_invalidations += ir.symbolic_invalidations;
    };
    auto record_inner = [&](const InnerResult& ir) {
      result.inner_iterations += ir.iterations;
      result.kkt_factorizations += ir.factorizations;
      result.kkt_symbolic_analyses += ir.symbolic_analyses;
      result.kkt_numeric_scatter_builds += ir.numeric_scatter_builds;
      result.kkt_numeric_scatter_reuses += ir.numeric_scatter_reuses;
      result.kkt_numeric_scatter_invalidations +=
          ir.numeric_scatter_invalidations;
      result.lazyqp_inexact_forcing_model_evaluations +=
          ir.lazyqp_inexact_forcing_model_evaluations;
      result.lazyqp_model_support_fastpath_requests +=
          ir.lazyqp_model_support_fastpath_requests;
      result.lazyqp_model_support_fastpath_effective_subproblems +=
          ir.lazyqp_model_support_fastpath_effective_subproblems;
      result.lazyqp_model_support_builds +=
          ir.lazyqp_model_support_builds;
      result.lazyqp_model_support_eligible_models +=
          ir.lazyqp_model_support_eligible_models;
      result.lazyqp_model_support_generic_fallback_models +=
          ir.lazyqp_model_support_generic_fallback_models;
      result.lazyqp_model_support_value_evaluations +=
          ir.lazyqp_model_support_value_evaluations;
      result.lazyqp_model_support_gradient_evaluations +=
          ir.lazyqp_model_support_gradient_evaluations;
      result.lazyqp_kkt_triplet_layout_builds +=
          ir.lazyqp_kkt_triplet_layout_builds;
      result.lazyqp_kkt_triplet_layout_reuses +=
          ir.lazyqp_kkt_triplet_layout_reuses;
      result.lazyqp_kkt_triplet_layout_invalidations +=
          ir.lazyqp_kkt_triplet_layout_invalidations;
      result.last_inner_dual_inf=ir.dual_inf;
      result.last_inner_equality_inf=ir.equality_inf;
      result.last_inner_inequality_inf=ir.inequality_inf;
      result.last_inner_primal_residual_inf=ir.primal_residual_inf;
      result.last_inner_complementarity=ir.complementarity;
      result.last_inner_scaled_model_residual_inf=ir.scaled_model_residual_inf;
      result.last_inner_used_iteration_limit_fallback=ir.used_iteration_limit_fallback;
      if(ir.used_iteration_limit_fallback)
        ++result.inner_iteration_limit_fallback_acceptances;
      if(ir.lazyqp_inexact_forcing_attempted)
        ++result.lazyqp_inexact_forcing_attempts;
      if(ir.lazyqp_inexact_forcing_accepted)
        ++result.lazyqp_inexact_forcing_acceptances;
      if(ir.lazyqp_inexact_forcing_strict_fallback)
        ++result.lazyqp_inexact_forcing_strict_fallbacks;
      result.lazyqp_inexact_forcing_max_target=std::max(
          result.lazyqp_inexact_forcing_max_target,
          ir.lazyqp_inexact_forcing_target);
      if(ir.dca_single_loop_attempted)
        ++result.dca_single_loop_attempts;
      if(ir.dca_single_loop_accepted)
        ++result.dca_single_loop_acceptances;
      if(ir.dca_single_loop_strict_fallback)
        ++result.dca_single_loop_strict_fallbacks;
      if(ir.dca_single_loop_state_transport_used)
        ++result.dca_single_loop_state_transport_acceptances;
      if(ir.dca_single_loop_state_transport_rejected)
        ++result.dca_single_loop_state_transport_rejections;
      if(ir.dca_single_loop_reduced_bootstrap_attempted)
        ++result.dca_single_loop_reduced_bootstrap_attempts;
      if(ir.dca_single_loop_reduced_bootstrap_accepted)
        ++result.dca_single_loop_reduced_bootstrap_acceptances;
      result.dca_single_loop_inner_steps+=ir.dca_single_loop_steps;
      result.dca_single_loop_factorizations+=
          ir.dca_single_loop_factorizations;
      result.dca_single_loop_max_initial_residual=std::max(
          result.dca_single_loop_max_initial_residual,
          ir.dca_single_loop_initial_residual);
      result.dca_single_loop_max_final_residual=std::max(
          result.dca_single_loop_max_final_residual,
          ir.dca_single_loop_final_residual);
      result.final_kkt_rho_x=ir.final_kkt_rho_x;
      result.max_kkt_rho_x=std::max(result.max_kkt_rho_x,ir.max_kkt_rho_x);
      result.final_kkt_delta_d=ir.final_kkt_delta_d;
      result.max_kkt_delta_d=std::max(result.max_kkt_delta_d,ir.max_kkt_delta_d);
      result.max_kkt_backward_error=std::max(result.max_kkt_backward_error,ir.max_kkt_backward_error);
      result.max_kkt_residual_inf=std::max(result.max_kkt_residual_inf,ir.max_kkt_residual_inf);
      result.kkt_regularization_retries += ir.kkt_regularization_retries;
      result.kkt_refinement_steps += ir.kkt_refinement_steps;
      absorb_kkt_diagnostics(ir);
      absorb_lazyqp_efficiency(ir);
    };
    // rho_x is changed only inside the inner KKT solver, where a concrete
    // factorization/backward-error diagnostic exists.  Global SQP events do
    // not modify numerical regularization.

    // ------------------------------------------------------------------
    // v0.4.0 Lazy-DCA outer loop.
    // ------------------------------------------------------------------
    // In the historical/elastic routes Q0 is convexified independently of all
    // constraint multipliers, as are the active inequality matrices.  The
    // condensed-retraction route is deliberately different: it uses the
    // signed equality-Lagrangian Hessian and convexifies only its restriction
    // to the affine equality tangent space.  Active inequalities retain their
    // independent convex majorants in every route.
    if(st_.outer_algorithm=="lazy-dca") {
      // The sparse factor object must outlive one convex majorant.  Keeping it
      // here allows SuiteSparse LDL/METIS (and the other sparse backends) to
      // retain their ordering, CSC envelope, elimination tree and workspaces
      // across the entire Lazy-DCA phase.  The Settings object is persistent as
      // well; only its numerical rho_x seed is refreshed between subproblems.
      Settings persistent_lazyqp_settings=inner_settings();
      ConvexQCQPLazyQPKernel persistent_lazyqp(persistent_lazyqp_settings);
      ConvexQCQPSOCPKernel persistent_socp(persistent_lazyqp_settings);
      SparseSingleLoopKernel persistent_sparse(persistent_lazyqp_settings);
      // Equality-only convex majorants use a sparse bordered KKT representation
      // for every tangent low-rank correction.  Keep that factor object alive
      // across outer iterations as well, so LDL/METIS can reuse its symbolic
      // envelope whenever the bordered dimension is unchanged.  Regularization
      // is inserted explicitly by solve_equality_qp; the backend must not shift
      // all non-primal coordinates and thereby alter the low-rank coefficients.
      Settings persistent_equality_factor_settings=inner_settings();
      persistent_equality_factor_settings.kkt_adaptive_regularization=true;
      std::string persistent_equality_factor_message;
      auto persistent_equality_factor=make_symmetric_factorization(
          persistent_equality_factor_settings,persistent_equality_factor_message);
      // A missing persistent factor is nonfatal here: solve_equality_qp will
      // attempt to construct a local factor and report the concrete backend
      // failure if an equality-only subproblem is actually encountered.
      // Multiplier projections have the same sparse Jacobian graph from one
      // accepted point to the next.  Keep separate path/certificate factors so
      // the regularization ladder used only near termination cannot evict the
      // symbolic state used by the per-iteration Lagrangian model.
      FreeMultiplierProjector path_multiplier_projector(st_);
      FreeMultiplierProjector certificate_multiplier_projector(st_);
      // Keep the tangent-curvature projector and the bound-face retraction on
      // separate factor objects.  The former sees the full equality tangent
      // space; the latter may lock active simple-bound coordinates without
      // changing the convexity certificate used by the majorant.
      CondensedQuadraticRetraction condensed_geometry(
          st_,st_.dca_condensed_flow_lift_schur);
      Settings tangent_factor_settings=st_;
      tangent_factor_settings.kkt_adaptive_regularization=true;
      tangent_factor_settings.linear_metis=false;
      std::string tangent_factor_message;
      auto tangent_factor=st_.dca_equality_model=="tangent-inertia"
          ? make_symmetric_factorization(tangent_factor_settings,tangent_factor_message)
          : std::unique_ptr<SymmetricFactorization>{};
      CondensedQuadraticRetraction condensed_retraction(
          st_,st_.dca_condensed_flow_lift_schur);
      const bool uses_lazyqp_kernel=st_.inner_backend=="lazyqp" ||
          st_.inner_backend=="lazyqp-sparse" ||
          st_.inner_backend=="socp" || st_.inner_backend=="ipm" ||
          st_.inner_backend=="native";
      const std::vector<ExactAffineElimination> lazyqp_flow_lift_plan =
          st_.lazyqp_flow_lift_elimination && uses_lazyqp_kernel &&
                  st_.dca_equality_model != "elastic-split"
              ? detect_pglib_apparent_flow_lifts(problem)
              : std::vector<ExactAffineElimination>{};
      MultiplierEstimate warm_mult;
      bool warm_mult_valid=false;
      Vec mapped_stationarity_cache_point;
      double mapped_stationarity_cache_value=
          std::numeric_limits<double>::infinity();
      bool mapped_stationarity_cache_valid=false;
      auto invalidate_mapped_stationarity_cache = [&]() {
        mapped_stationarity_cache_valid=false;
      };
      auto cached_projected_stationarity = [&](const Vec& point)->double {
        if(mapped_stationarity_cache_valid &&
           mapped_stationarity_cache_point==point)
          return mapped_stationarity_cache_value;
        mapped_stationarity_cache_value=projected_stationarity(
            problem,point,warm_mult,st_.working_set_activation);
        mapped_stationarity_cache_point=point;
        mapped_stationarity_cache_valid=true;
        return mapped_stationarity_cache_value;
      };
      // A transported/mapped multiplier may remain perfectly usable for the
      // next tangent model after an optional refit fails.  Keep that model
      // validity separate from certificate eligibility: a model-only state
      // must be refreshed (or independently refined) before it can terminate
      // the original QCQP.
      bool warm_mult_model_only=false;
      bool single_loop_final_certificate_attempted=false;
      if(st_.dca_single_loop_primal_dual) {
        // The first majorant has no predecessor from which to transport a
        // primal-dual state. Start from a finite model-only zero state; the
        // reduced LazyQP bootstrap initializes free equality multipliers in
        // the actual affine-reduced coordinates. No original-Jacobian
        // multiplier reconstruction is permitted before the final
        // certificate.
        warm_mult.equality.assign(problem.equalities.size(),0.0);
        warm_mult.inequality.assign(problem.inequalities.size(),0.0);
        warm_mult.lower_bound.assign(static_cast<std::size_t>(problem.n),0.0);
        warm_mult.upper_bound.assign(static_cast<std::size_t>(problem.n),0.0);
        warm_mult.valid=true;
        warm_mult_valid=true;
        warm_mult_model_only=true;
        result.model_stationarity_inf=cached_projected_stationarity(x);
        if(st_.verbose)
          std::cout << "[DCA-SINGLE-LOOP-BOOTSTRAP] accepted=1"
                    << " source=model-zero full-fit=0 stationarity="
                    << std::scientific
                    << result.model_stationarity_inf
                    << std::defaultfloat
                    << "\n";
      }
      // Large inequality-constrained original-KKT Newton is allowed to take
      // over only after the same multiplier-filtered active manifold has been
      // observed at several consecutive feasible accepted points.  This state
      // is local to one solve and is cleared after every rejected or partial
      // terminal sequence.
      std::vector<long long> lazy_terminal_active_signature;
      int lazy_terminal_active_stability=0;
      Vec continuation_equality_duals;
      Vec continuation_inequality_duals;
      Vec continuation_inequality_slacks;
      Vec continuation_inequality_primal_residuals;
      std::vector<std::uint64_t> continuation_inequality_keys;
      bool continuation_valid=false;
      // Only exact side constraints that were violated by an actually
      // accepted transient filter step may tighten the next majorant.  This
      // state starts empty, so merely enabling the opt-in filter cannot alter
      // restoration of an arbitrary infeasible initial point.
      std::set<int> modeled_side_recovery_rows;
      // A direct start with an initially violated side constraint begins in a
      // bounded feasibility-recovery phase.  The first accepted point inside
      // the strict side tube latches this flag permanently; the relaxed
      // h-only path can never be re-entered later in the solve.
      bool direct_start_side_strict_latched=!direct_start_accepted ||
          direct_start.feasibility.inequality_inf<=
              st_.feasibility_tolerance;
      int small_step_streak=0;
      // Safeguarded condensed line-search history.  It is deliberately local
      // to one solve and is invalidated by a working-set, bound-face,
      // direction-scale, or proximal-model discontinuity.
      bool condensed_alpha_history_valid=false;
      double condensed_last_accepted_alpha=1.0;
      double condensed_last_direction_inf=0.0;
      double condensed_last_model_prox=0.0;
      int condensed_last_bound_face_coordinates=-1;
      int condensed_iterations_since_full_probe=0;
      int condensed_alpha_one_streak=0;
      std::vector<int> condensed_last_working_set;
      // The first single-loop model deliberately has no reconstructed
      // original-QCQP multipliers, so its multiplier-weighted equality
      // curvature is absent. Keep a small deterministic convex baseline on
      // every continuation model; it is not a retry penalty and does not add
      // any factorization to the one-PC transaction.
      const double single_loop_baseline_prox=
          st_.dca_single_loop_primal_dual && has_quadratic_equality
              ? st_.dca_single_loop_baseline_proximal_regularization : 0.0;
      double dca_model_prox=0.0;
      result.dca_single_loop_baseline_proximal_regularization=
          single_loop_baseline_prox;
      result.lazy_dca_max_model_proximal=single_loop_baseline_prox;
      int dca_model_clean_steps=0;
      int dca_warm_failure_streak=0;
      int dca_warm_cooldown_remaining=0;
      const bool elastic_equality_mode=has_quadratic_equality &&
          st_.dca_equality_model=="elastic-split";
      const bool condensed_equality_mode=has_quadratic_equality &&
          (st_.dca_equality_model=="condensed-retraction" ||
           st_.dca_equality_model=="tangent-inertia");
      double equality_funnel=st_.feasibility_tolerance;
      if(has_quadratic_equality) {
        const Feasibility initial_funnel_feas=
            structural_feasibility(problem,x);
        if(st_.dca_equality_funnel_initial>0.0) {
          equality_funnel=std::max(st_.feasibility_tolerance,
              std::max(st_.dca_equality_funnel_initial,
                       st_.dca_equality_funnel_growth*
                           initial_funnel_feas.equality_inf));
        } else if(st_.dca_equality_funnel_auto) {
          const double automatic=std::sqrt(std::max(
              st_.feasibility_tolerance,
              64.0*std::numeric_limits<double>::epsilon()));
          equality_funnel=std::max(st_.feasibility_tolerance,
              std::min(std::max(st_.feasibility_tolerance,
                                st_.dca_equality_funnel_auto_ceiling),
                       std::max(automatic,st_.dca_equality_funnel_growth*
                                            initial_funnel_feas.equality_inf)));
        }
      }
      result.lazy_dca_final_equality_funnel=equality_funnel;

      auto refit_path_multipliers = [&](const Vec& point,int outer,
                                        const char* reason)->bool {
        // A single-loop trajectory is never allowed to fall back to the full
        // original-Jacobian path fit.  Call sites must either keep a guarded
        // model-only state or fail transactionally.  The sole full fit is the
        // independent terminal certificate and does not use this helper.
        if(st_.dca_single_loop_primal_dual) {
          if(st_.verbose)
            std::cout << "[MULTIPLIER-REFIT] k="<<outer
                      << " reason="<<reason
                      << " accepted=0 action=single-loop-model-only\n";
          return false;
        }
        ++result.multiplier_refit_attempts;
        ConstrainedMultiplierLS fit;
        MultiplierEstimate refit=estimate_multipliers_legacy(
            problem,point,working_set,st_,&path_multiplier_projector,&fit,true);
        if(!refit.valid) {
          if(warm_mult_valid) warm_mult_model_only=true;
          if(st_.verbose)
            std::cout << "[MULTIPLIER-REFIT] k="<<outer
                      << " reason="<<reason
                      << " accepted=0 message="
                      << path_multiplier_projector.last_message()
                      << "\n";
          return false;
        }
        warm_mult=std::move(refit);
        invalidate_mapped_stationarity_cache();
        warm_mult_valid=true;
        warm_mult_model_only=false;
        result.last_multiplier_refit_stationarity_inf=
            cached_projected_stationarity(point);
        result.model_stationarity_inf=result.last_multiplier_refit_stationarity_inf;
        ++result.multiplier_refit_successes;
        if(st_.verbose)
          std::cout << "[MULTIPLIER-REFIT] k="<<outer
                    << " reason="<<reason
                    << " accepted=1 stationarity="<<std::scientific
                    << result.last_multiplier_refit_stationarity_inf
                    << " free="<<problem.equalities.size()
                    << " nonnegative="<<fit.nonnegative_duals.size()
                    << "\n"<<std::defaultfloat;
        return true;
      };
      // This function is called only after the primal endpoint has passed the
      // existing outer globalization on the ORIGINAL QCQP.  The proposed
      // multiplier is nevertheless evaluated privately before commit.  Any
      // malformed/interior-inconsistent endpoint or non-finite original KKT
      // state leaves the live multiplier untouched and rejects the provisional
      // transaction; it never authorizes an intermediate full refit.
      auto commit_guarded_single_loop_transport = [&]
          (const Vec& point,const MultiplierEstimate& proposed,
           bool primary_single_loop_state,bool endpoint_matches,
           int outer,const char* reason)->bool {
        if(!st_.dca_single_loop_primal_dual ||
           !st_.dca_single_loop_multiplier_transport ||
           !primary_single_loop_state)
          return false;
        if(!endpoint_matches) {
          ++result.dca_single_loop_refit_fallbacks;
          if(st_.verbose)
            std::cout << "[SINGLE-LOOP-MULTIPLIER-TRANSPORT] k="<<outer
                      << " reason="<<reason
                      << " accepted=0 cause=endpoint-state"
                      << " action=reject-transaction\n";
          return false;
        }
        const SingleLoopMultiplierGuard guard=
            st_.lazyqp_affine_support_fastpath
                ? guard_single_loop_multiplier_transport(problem,point,proposed,st_,
                    StructuralQCQPEvaluation(problem.n))
                : guard_single_loop_multiplier_transport(problem,point,proposed,st_);
        if(!guard.accepted) {
          ++result.dca_single_loop_refit_fallbacks;
          if(st_.verbose)
            std::cout << "[SINGLE-LOOP-MULTIPLIER-TRANSPORT] k="<<outer
                      << " reason="<<reason
                      << " accepted=0 cause=original-kkt"
                      << " action=reject-transaction\n";
          return false;
        }
        warm_mult=proposed;
        invalidate_mapped_stationarity_cache();
        warm_mult_valid=true;
        warm_mult_model_only=true;
        result.model_stationarity_inf=cached_projected_stationarity(point);
        ++result.dca_single_loop_refits_skipped;
        if(st_.verbose)
          std::cout << "[SINGLE-LOOP-MULTIPLIER-TRANSPORT] k="<<outer
                    << " reason="<<reason
                    << " accepted=1 certificate=0 original-dual="
                    << std::scientific
                    << guard.original_dual_infeasibility
                    << " original-complementarity="
                    << guard.original_complementarity
                    << "\n"<<std::defaultfloat;
        return true;
      };
      const double fresh_certificate_gate=std::max(
          1e-4,100.0*std::max({st_.stationarity_tolerance,
                               st_.dual_infeasibility_tolerance,
                               st_.complementarity_tolerance,
                               st_.feasibility_tolerance}));
      auto mapped_near_termination = [&]()->bool {
        return std::isfinite(result.scaled_nlp_error) &&
               result.scaled_nlp_error<=fresh_certificate_gate;
      };
      // Intermediate certificate refits are expensive on large equality
      // systems.  More importantly, `warm_mult_model_only` describes the
      // provenance of a multiplier; it is not evidence that the current point
      // is close to stationary.  Admit a fresh path/certificate fit only when
      // a finite raw original-model stationarity residual is below the
      // documented fresh-certificate gate above.  The final
      // exact certificate after the outer loop intentionally remains governed
      // by the historical policy above.
      const double fresh_path_stationarity_gate=fresh_certificate_gate;
      auto raw_model_near_termination = [&](const Vec& point)->bool {
        if(!warm_mult_valid || !warm_mult.valid) return false;
        const double raw=cached_projected_stationarity(point);
        return std::isfinite(raw) && raw<=fresh_path_stationarity_gate;
      };
      // A single-loop trajectory is allowed only one full multiplier fit.  A
      // looser historical 1e-4 path-refit gate can therefore spend that sole
      // fit prematurely and strand an otherwise convergent trajectory.  Use
      // the requested first-order scale for the cheap transported-state
      // neighborhood test; unlike the old mapped KKT gate this still does not
      // require the transported dual/slack state itself to be certifiable.
      const double single_loop_final_stationarity_gate=std::min(
          {st_.stationarity_tolerance,
           st_.dual_infeasibility_tolerance,
           st_.complementarity_tolerance});
      auto raw_single_loop_final_near_termination =
          [&](const Vec& point)->bool {
        if(!warm_mult_valid || !warm_mult.valid) return false;
        const double raw=cached_projected_stationarity(point);
        return std::isfinite(raw) &&
            raw<=fresh_certificate_gate &&
            std::isfinite(result.step_inf) &&
            result.step_inf<=single_loop_final_stationarity_gate &&
            result.last_inner_dual_inf<=single_loop_final_stationarity_gate &&
            result.last_inner_complementarity<=
                0.1*single_loop_final_stationarity_gate;
      };
      auto run_single_loop_final_certificate = [&]
          (const Vec& point,int outer,const char* reason)->bool {
        if(!st_.dca_single_loop_primal_dual ||
           single_loop_final_certificate_attempted)
          return false;
        // `multiplier_refit_attempts` counts the path/full fits above.  It is
        // a runtime audit of the literal single-loop contract: none may have
        // happened before entering this one terminal certificate phase.
        if(result.multiplier_refit_attempts!=0) {
          result.status=Status::NumericalError;
          result.message=
              "single-loop terminal certificate reached after an "
              "intermediate multiplier refit";
          return false;
        }
        single_loop_final_certificate_attempted=true;
        ++result.dca_single_loop_full_refit_calls;
        if(st_.verbose)
          std::cout << "[DCA-SINGLE-LOOP-FINAL-CERTIFICATE] k="<<outer
                    << " reason="<<reason
                    << " call="<<result.dca_single_loop_full_refit_calls
                    << "\n";
        return fresh_certificate_passes(
            problem,point,st_,result,&certificate_multiplier_projector);
      };

      // Ipopt-style live view of the ORIGINAL QCQP, augmented with the Lazy
      // model state.  Build each line in a private stream and flush it in one
      // write so redirected campaign logs never contain half a row.
      int live_rows=0;
      int live_bound_face_coordinates=0;
      const char* live_equality_model = !has_quadratic_equality ? "AFF" :
          (condensed_equality_mode ? "COND" :
           (elastic_equality_mode ? "ELAS" : "LIN"));
      auto live_real = [](double value,int precision=2)->std::string {
        if(!std::isfinite(value)) return "-";
        std::ostringstream os;
        os<<std::scientific<<std::setprecision(precision)<<value;
        return os.str();
      };
      auto live_inequality_rank = [&]()->int {
        int rank=0;
        for(int i:working_set)
          if(i>=0 && i<(int)ineq_cvx.size()) rank+=ineq_cvx[i].rank();
        return rank;
      };
      auto print_live_row = [&](int iteration,const Vec& point,
                                const Feasibility& feas,double dual_inf,
                                double step_inf,double alpha,int objective_rank,
                                int line_searches,const char* tag,bool force=false) {
        if(!st_.live_iterations) return;
        if(!force && iteration%st_.live_iteration_every!=0) return;
        std::ostringstream os;
        if(live_rows%st_.live_iteration_header_every==0) {
          os<<"[ITER] "<<std::setw(4)<<"iter"
            <<" "<<std::setw(13)<<"objective"
            <<" "<<std::setw(10)<<"inf_eq"
            <<" "<<std::setw(10)<<"inf_ineq"
            <<" "<<std::setw(10)<<"inf_bnd"
            <<" "<<std::setw(10)<<"inf_du"
            <<" "<<std::setw(9)<<"lg(mu)"
            <<" "<<std::setw(10)<<"||d||"
            <<" "<<std::setw(9)<<"alpha"
            <<" "<<std::setw(3)<<"r0"
            <<" "<<std::setw(3)<<"rI"
            <<" "<<std::setw(9)<<"ws"
            <<" "<<std::setw(4)<<"bF"
            <<" "<<std::setw(3)<<"ls"
            <<" "<<std::setw(3)<<"tag"
            <<" "<<std::setw(4)<<"eqM"<<"\n";
        }
        const double log_mu=result.last_inner_complementarity>0.0
            ? std::log10(result.last_inner_complementarity)
            : std::numeric_limits<double>::quiet_NaN();
        std::ostringstream ws;
        ws<<working_set.size()<<"/"<<problem.inequalities.size();
        os<<"[ITER] "<<std::setw(4)<<iteration
          <<" "<<std::setw(13)<<live_real(problem.objective.value(point),5)
          <<" "<<std::setw(10)<<live_real(feas.equality_inf)
          <<" "<<std::setw(10)<<live_real(feas.inequality_inf)
          <<" "<<std::setw(10)<<live_real(feas.bound_inf)
          <<" "<<std::setw(10)<<live_real(dual_inf)
          <<" "<<std::setw(9)<<live_real(log_mu,1)
          <<" "<<std::setw(10)<<live_real(step_inf)
          <<" "<<std::setw(9)<<live_real(alpha,1)
          <<" "<<std::setw(3)<<objective_rank
          <<" "<<std::setw(3)<<live_inequality_rank()
          <<" "<<std::setw(9)<<ws.str()
          <<" "<<std::setw(4)<<live_bound_face_coordinates
          <<" "<<std::setw(3)<<line_searches
          <<" "<<std::setw(3)<<tag
          <<" "<<std::setw(4)<<live_equality_model<<"\n";
        std::cout<<os.str()<<std::flush;
        ++live_rows;
      };

      if(st_.verbose) {
        const char* constraint_set=full_affine_majorant
            ? "full-affine-objective"
            : (objective_is_affine && affine_box_lazy_route
                ? "lazy-affine-box-bounded" : "lazy");
        std::cout << "[LAZY-DCA] objective rank="<<objective_cvx.rank()
                  <<" shift="<<std::scientific<<objective_cvx.scalar_shift
                  <<" inner="<<st_.inner_backend
                  <<" nonlinear_eq="<<(has_quadratic_equality?1:0)
                  <<" equality_model="<<st_.dca_equality_model
                  <<" constraint_set="<<constraint_set
                  <<" ws="<<working_set.size()
                  <<"\n"<<std::defaultfloat;
      }

      const Feasibility live_initial_feas=structural_feasibility(problem,x);
      print_live_row(0,x,live_initial_feas,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),objective_cvx.rank(),0,"I",true);

      for(int outer=0;outer<st_.max_outer_iterations;++outer) {
        live_bound_face_coordinates=0;
        const int live_ls_before=result.line_search_backtracks;
        const int live_soc_before=result.soc_accepted_steps;
        const int live_retraction_before=result.lazy_dca_equality_retraction_acceptances;
        const int live_corrector_before=result.lazy_dca_corrector_acceptances;
        const int live_terminal_before=result.terminal_kkt_accepted_steps;
        result.outer_iterations=outer;
        if(elapsed()>=st_.time_limit_seconds) {
          result.status=Status::TimeLimit;
          result.message="wall-clock limit reached in Lazy-DCA";
          break;
        }
        const bool direct_start_recovery_active=direct_start_accepted &&
            !direct_start_side_strict_latched;
        const DirectStartEvaluation direct_start_recovery_current=
            direct_start_recovery_active
                ? evaluate_direct_start(problem,x)
                : DirectStartEvaluation{};
        if(direct_start_recovery_active &&
           !direct_start_recovery_current.finite) {
          result.status=Status::NumericalError;
          result.message=
              "single-loop direct-start current iterate is non-finite in the original QCQP";
          break;
        }
        const Feasibility fcur=direct_start_recovery_active
            ? direct_start_recovery_current.feasibility
            : (condensed_equality_mode && condensed_retraction.ready())
                ? condensed_retraction.evaluate_feasibility(x)
            : (condensed_equality_mode && condensed_geometry.ready())
                ? condensed_geometry.evaluate_feasibility(x)
                : structural_feasibility(problem,x);
        const double objcur=direct_start_recovery_active
            ? direct_start_recovery_current.objective
            : structural_quadratic_value(problem.objective,x);
        const Vec outer_origin=x;
        // The accepted point was already certified at the end of the previous
        // outer iteration.  Repeating a fresh multiplier least-squares fit at
        // the top of every loop doubled the dominant CATENA/CATENARY cost.
        // Only the restored initial point needs a pre-step certificate on the
        // historical route.  Single-loop deliberately enters its first model
        // with the zero/model-only state initialized above.
        if(!st_.dca_single_loop_primal_dual && outer==0 &&
           fcur.total_inf<=st_.feasibility_tolerance) {
          if((!warm_mult_valid || warm_mult_model_only) &&
             has_quadratic_equality)
            refit_path_multipliers(x,outer,"initial");
          bool kkt_ok=false;
          if(warm_mult_valid && !warm_mult_model_only)
            kkt_ok=mapped_certificate_passes(problem,x,warm_mult,st_,result);
          if(!kkt_ok && raw_model_near_termination(x))
            kkt_ok=fresh_certificate_passes(
                problem,x,st_,result,&certificate_multiplier_projector);
          if(kkt_ok) {
            result.status=Status::Solved;
            result.message="composite original-QCQP KKT test satisfied by Lazy-DCA";
            break;
          }
        }

        current_lazyqp_forcing_target=0.0;
        current_lazyqp_forcing_primal_target=0.0;
        if(st_.lazyqp_inexact_forcing && warm_mult_valid) {
          const double outer_stationarity=cached_projected_stationarity(x);
          const double outer_residual=std::max(fcur.total_inf,
                                               outer_stationarity);
          if(std::isfinite(outer_residual)) {
            current_lazyqp_forcing_target=std::clamp(
                st_.lazyqp_forcing_eta*outer_residual,
                st_.inner_tolerance,st_.lazyqp_forcing_max_tolerance);
            // An inexact dual/model residual may be useful far from the
            // solution, but the tangent equalities and modeled inequalities
            // must remain inside the original globalization tube.
            current_lazyqp_forcing_primal_target=std::min(
                current_lazyqp_forcing_target,
                std::max(st_.feasibility_tolerance,
                         10.0*st_.inner_tolerance));
          }
          if(st_.verbose)
            std::cout << "[LAZY-DCA-FORCING] k="<<outer
                      << " original_residual="<<std::scientific
                      << outer_residual
                      << " target="<<current_lazyqp_forcing_target
                      << " primal_target="
                      << current_lazyqp_forcing_primal_target
                      << " fast_iters="<<st_.lazyqp_forcing_fast_iterations
                      << "\n"<<std::defaultfloat;
        }

        // True two-way lazy working set.  A previously modeled inequality can
        // leave once it is safely slack in ORIGINAL coordinates and its mapped
        // convex-subproblem multiplier is negligible.  Any dropped constraint
        // is still checked globally on the candidate and is re-added before
        // acceptance if it becomes violated.
        if(!full_affine_majorant && st_.working_set_dropping && warm_mult_valid) {
          std::vector<int> drop;
          const int drop_patience=std::max(1,st_.working_set_drop_patience);
          for(int i:working_set) {
            const double mu=(i<(int)warm_mult.inequality.size())
                ? std::max(0.0,warm_mult.inequality[i]) : 0.0;
            if(structural_quadratic_value(problem.inequalities[i],x)<
                   -st_.working_set_drop_slack &&
               mu<=st_.working_set_drop_multiplier) {
              int& streak=working_set_drop_streak[static_cast<std::size_t>(i)];
              streak=std::min(drop_patience,streak+1);
              if(streak>=drop_patience) drop.push_back(i);
            } else {
              working_set_drop_streak[static_cast<std::size_t>(i)]=0;
            }
          }
          for(int i:drop) {
            working_set.erase(i);
            working_set_drop_streak[static_cast<std::size_t>(i)]=0;
            if(i<(int)warm_mult.inequality.size()) {
              warm_mult.inequality[i]=0.0;
              invalidate_mapped_stationarity_cache();
            }
            ++result.working_set_drops;
          }
          if(st_.verbose && !drop.empty())
            std::cout << "[LAZY-DCA-DROP] k="<<outer<<" dropped="<<drop.size()
                      << " ws="<<working_set.size()<<"\n";
        } else if(!full_affine_majorant && st_.working_set_dropping) {
          // Patience counts accepted points for which both the slack and
          // multiplier tests are available.  Missing multipliers break that
          // consecutive-point contract.
          std::fill(working_set_drop_streak.begin(),
                    working_set_drop_streak.end(),0);
        }

        // Activate near-active inequalities.  Omitted constraints remain
        // globally checked on every candidate before acceptance.
        if(!full_affine_majorant) {
          for(int i=0;i<(int)problem.inequalities.size();++i) {
            if(working_set.count(i)) continue;
            if(structural_quadratic_value(problem.inequalities[i],x)>=
               -st_.working_set_activation) {
              working_set.insert(i);
              working_set_drop_streak[static_cast<std::size_t>(i)]=0;
              curvature.ensure_convex(problem.inequalities[i].hessian,ineq_cvx[i],0.0,
                                      st_.max_rank_per_quadratic);
              ++result.working_set_additions;
            }
          }
        }

        SymMatrix condensed_objective_hessian;
        Convexification condensed_objective_cvx;
        const SymMatrix* objective_hessian_override=nullptr;
        const Convexification* active_objective_cvx=&objective_cvx;
        // r3p20 can certify the ambient signed Hessian exactly from its
        // independent sparse blocks.  In that case no tangent projection is
        // performed and the later bound-aware retraction factor is the sole
        // normal map built for this outer iteration.
        bool condensed_single_normal_map=false;
        if(condensed_equality_mode) {
          AccumulateSeconds timer{result.equality_model_seconds};
          // Equality curvature must be available from the first manifold step.
          // A working-set edit changes only inequality membership.  The signed
          // equality multiplier transported by the accepted convex-QP step is
          // still a valid provisional Lagrangian model and must not trigger a
          // full joint equality/inequality/bound refit.  Such transported
          // multipliers remain model-only and can never certify termination.
          bool condensed_multiplier_valid=warm_mult_valid && warm_mult.valid;
          if(condensed_multiplier_valid && warm_mult_model_only) {
            ++result.multiplier_refit_fallback_reuses;
            if(st_.verbose)
              std::cout << "[CONDENSED-MULTIPLIER-TRANSPORT] k="<<outer
                        << " source=accepted-model"
                        << " ws="<<working_set.size()
                        << " certificate=0\n";
          }
          if(!condensed_multiplier_valid &&
             !st_.dca_single_loop_primal_dual) {
            condensed_multiplier_valid=refit_path_multipliers(
                x,outer,"condensed-model-initial");
          }
          if(!condensed_multiplier_valid) {
            result.status=Status::NumericalError;
            result.message=
                "condensed quadratic equality model requires a valid signed "
                "equality-multiplier fit";
            break;
          }
          condensed_objective_hessian=
              equality_lagrangian_hessian(problem,warm_mult);
          // Start from no ambient-space correction.  Only negative curvature
          // visible on the equality tangent space is needed by the model.
          condensed_objective_cvx=Convexification{};
          if(st_.dca_equality_model=="tangent-inertia" && warm_mult_valid &&
             cached_projected_stationarity(x)<=st_.dca_tangent_inertia_trigger) {
            ++result.equality_tangent_inertia_attempts;
            std::string message;
            ConvexSubproblem tangent;
            tangent.nvar=problem.n;
            tangent.original_dimension=problem.n;
            tangent.prefer_feasible_centered_start=true;
            tangent.objective.base_hessian=condensed_objective_hessian;
            tangent.objective.linear.assign(problem.n,0.0);
            tangent.objective.shift_dimension=problem.n;
            tangent.objective.scalar_shift=single_loop_baseline_prox-st_.curvature_margin;
            tangent.equalities.nvar=problem.n;
            tangent.exact_affine_eliminations=lazyqp_flow_lift_plan;
            SparseGradientWorkspace workspace(problem.n);
            for(const auto& h:problem.equalities) {
              SparseRow row;
              consume_sparse_quadratic_gradient(h,x,workspace,[&](int j,double v) {
                if(v!=0.0) row.entries.push_back({j,v});
              });
              tangent.equalities.rows.push_back(std::move(row));
              tangent.equalities.rhs.push_back(0.0);
            }
            for(int j=0;j<problem.n;++j)
              if(std::isfinite(problem.lb[j]) && std::isfinite(problem.ub[j]) &&
                 std::abs(problem.ub[j]-problem.lb[j])<=st_.fixed_bound_tolerance) {
                SparseRow row; row.entries.push_back({j,1.0});
                tangent.equalities.rows.push_back(std::move(row));
                tangent.equalities.rhs.push_back(0.0);
              }
            const bool certified=tangent_factor &&
                certify_affine_tangent_inertia(tangent,*tangent_factor,message);
            if(certified) {
              condensed_single_normal_map=true;
              ++result.equality_tangent_inertia_acceptances;
            }
            if(st_.verbose)
              std::cout << "[TANGENT-INERTIA] k=" << outer
                        << " accepted=" << (certified?1:0)
                        << " fallback=" << (certified?0:1)
                        << " message=" << message << "\n";
          }
          if(!condensed_single_normal_map && st_.dca_condensed_tangent_curvature &&
             st_.dca_condensed_block_exact_curvature) {
            ++result.lazy_dca_condensed_block_exact_attempts;
            const int calls_before=condensed_curvature.calls();
            BlockExactConvexificationResult block_exact;
            std::string block_failure;
            try {
              block_exact=condensed_curvature.ensure_convex_block_exact(
                  condensed_objective_hessian,condensed_objective_cvx,
                  st_.curvature_margin,st_.dca_condensed_max_tangent_rank);
            } catch(const std::exception& e) {
              block_failure=e.what();
              condensed_objective_cvx=Convexification{};
            }
            result.lazy_dca_condensed_block_exact_blocks_max=std::max(
                result.lazy_dca_condensed_block_exact_blocks_max,
                block_exact.block_count);
            result.lazy_dca_condensed_block_exact_active_dimension_max=std::max(
                result.lazy_dca_condensed_block_exact_active_dimension_max,
                block_exact.active_dimension);
            result.lazy_dca_condensed_block_exact_max_block_dimension=std::max(
                result.lazy_dca_condensed_block_exact_max_block_dimension,
                block_exact.max_block_dimension);
            // Count every partial-spectrum kernel invocation returned by the
            // fail-closed block oracle, including work performed by a call
            // that ultimately reports unavailable.  If the oracle throws,
            // only the default-zero data in the caller-owned result are
            // available and no unobservable work is inferred.
            result.lazy_dca_condensed_block_partial_spectrum_attempts+=
                block_exact.partial_spectrum_attempts;
            result.lazy_dca_condensed_block_partial_spectrum_acceptances+=
                block_exact.partial_spectrum_acceptances;
            result.lazy_dca_condensed_block_partial_spectrum_fallbacks+=
                block_exact.partial_spectrum_fallbacks;
            if(block_exact.available) {
              condensed_single_normal_map=true;
              ++result.lazy_dca_condensed_block_exact_acceptances;
              result.lazy_dca_condensed_tangent_curvature_calls+=std::max(
                  0,condensed_curvature.calls()-calls_before);
              result.lazy_dca_condensed_tangent_rank_max=std::max(
                  result.lazy_dca_condensed_tangent_rank_max,
                  condensed_objective_cvx.rank());
              result.lazy_dca_condensed_scalar_safeguard_updates_max=std::max(
                  result.lazy_dca_condensed_scalar_safeguard_updates_max,
                  block_exact.scalar_shift_added>0.0 ? 1 : 0);
              result.lazy_dca_condensed_tangent_shift_max=std::max(
                  result.lazy_dca_condensed_tangent_shift_max,
                  condensed_objective_cvx.scalar_shift);
              result.lazy_dca_condensed_initial_tangent_min_eigenvalue=
                  block_exact.initial_min_eigenvalue;
              result.lazy_dca_condensed_final_tangent_min_eigenvalue=
                  block_exact.final_min_eigenvalue;
              if(st_.verbose)
                std::cout << "[CONDENSED-CURVATURE] k="<<outer
                          << " mode=ambient-block-exact-single-normal"
                          << " lambda="<<std::scientific
                          << block_exact.initial_min_eigenvalue<<"->"
                          << block_exact.final_min_eigenvalue
                          << " rank="<<condensed_objective_cvx.rank()
                          << " shift="<<condensed_objective_cvx.scalar_shift
                          << " blocks="<<block_exact.block_count
                          << " active_dim="<<block_exact.active_dimension
                          << " max_block="<<block_exact.max_block_dimension
                          << "\n"<<std::defaultfloat;
            } else if(st_.verbose) {
              std::cout << "[CONDENSED-CURVATURE-FALLBACK] k="<<outer
                        << " block_exact_message="
                        << (block_failure.empty()?"structural-guard":block_failure)
                        << " blocks="<<block_exact.block_count
                        << " active_dim="<<block_exact.active_dimension
                        << " max_block="<<block_exact.max_block_dimension
                        << "\n";
            }
          }
          if(!condensed_single_normal_map) {
            std::string condensed_message;
            if(!condensed_geometry.factor(problem,x,condensed_message)) {
              result.status=Status::NumericalError;
              result.message="condensed quadratic equality normal operator failed: "+
                  condensed_message;
              break;
            }
            result.lazy_dca_condensed_normal_regularization=
                condensed_geometry.regularization();
          }
          if(st_.dca_condensed_tangent_curvature &&
             !condensed_single_normal_map) {
            bool multi_ritz_ready=false;
            if(st_.dca_condensed_projected_multi_ritz_inertia) {
              const int inertia_before=
                  condensed_geometry.tangent_inertia_factorizations();
              const auto multi=condensed_geometry.ensure_tangent_convex_multi_ritz(
                  condensed_objective_hessian,condensed_objective_cvx,
                  st_.curvature_margin,st_.dca_condensed_max_tangent_rank);
              result.lazy_dca_condensed_tangent_curvature_calls+=
                  multi.oracle_calls;
              result.lazy_dca_condensed_projected_multi_ritz_batches+=
                  multi.oracle_calls;
              const int inertia_attempts=std::max(
                  0,condensed_geometry.tangent_inertia_factorizations()-
                      inertia_before);
              result.lazy_dca_condensed_tangent_inertia_attempts+=
                  inertia_attempts;
              if(multi.success)
                ++result.lazy_dca_condensed_tangent_inertia_acceptances;
              result.lazy_dca_condensed_tangent_inertia_shift_updates+=
                  multi.scalar_safeguard_updates;
              result.lazy_dca_condensed_tangent_inertia_shift_max=std::max(
                  result.lazy_dca_condensed_tangent_inertia_shift_max,
                  multi.scalar_shift_added);
              multi_ritz_ready=multi.success;
              if(multi_ritz_ready) {
                result.lazy_dca_condensed_tangent_rank_max=std::max(
                    result.lazy_dca_condensed_tangent_rank_max,
                    condensed_objective_cvx.rank());
                result.lazy_dca_condensed_scalar_safeguard_updates_max=std::max(
                    result.lazy_dca_condensed_scalar_safeguard_updates_max,
                    multi.scalar_safeguard_updates);
                result.lazy_dca_condensed_tangent_shift_max=std::max(
                    result.lazy_dca_condensed_tangent_shift_max,
                    condensed_objective_cvx.scalar_shift);
                result.lazy_dca_condensed_initial_tangent_min_eigenvalue=
                    multi.initial_min_eigenvalue;
                result.lazy_dca_condensed_final_tangent_min_eigenvalue=
                    multi.final_min_eigenvalue;
              }
              if(st_.verbose)
                std::cout << "[CONDENSED-CURVATURE] k="<<outer
                          << " mode=projected-multi-ritz-inertia"
                          << " success="<<(multi.success?1:0)
                          << " lambda="<<std::scientific
                          << multi.initial_min_eigenvalue<<"->"
                          << multi.final_min_eigenvalue
                          << " rank="<<condensed_objective_cvx.rank()
                          << " shift="<<condensed_objective_cvx.scalar_shift
                          << " batches="<<multi.oracle_calls
                          << " inertia="<<inertia_attempts
                          << "\n"<<std::defaultfloat;
              if(!multi_ritz_ready) {
                // Fail closed: discard every partial PSD proposal before the
                // unchanged r3p18 ambient/projected route below.
                condensed_objective_cvx=Convexification{};
                ++result.lazy_dca_condensed_projected_multi_ritz_fallbacks;
                if(st_.verbose)
                  std::cout << "[CONDENSED-CURVATURE-FALLBACK] k="<<outer
                            << " multi_ritz_message="<<multi.message<<"\n";
              }
            }
            if(!multi_ritz_ready) {
            // For a large ambient dimension, repeatedly projecting every
            // Lanczos vector through J^T(JJ^T)^{-1}J dominates the solve (on
            // case118 it accounted for more than 90% of all normal solves).
            // A convexification that is PSD in the ambient space is also PSD
            // on ker(J), so the ordinary sparse CurvatureOracle is a stronger
            // but much cheaper certificate here.  Small problems retain the
            // sharper projected oracle; a numerical ambient-oracle failure
            // also falls back to that historical route.
            const bool prefer_ambient_oracle=
                problem.n>std::max(512,st_.exact_eigen_threshold);
            bool ambient_ready=false;
            std::string ambient_failure;
            double ambient_initial=std::numeric_limits<double>::quiet_NaN();
            double ambient_final=std::numeric_limits<double>::quiet_NaN();
            if(prefer_ambient_oracle) {
              const int calls_before=condensed_curvature.calls();
              try {
                // A sparse Gershgorin pass provides a finite certified initial
                // lower bound without spending a diagnostic-only Lanczos run.
                ambient_initial=
                    condensed_objective_hessian.gershgorin_lower_bound();
                condensed_curvature.ensure_convex(
                    condensed_objective_hessian,condensed_objective_cvx,
                    st_.curvature_margin,st_.dca_condensed_max_tangent_rank);
                ambient_ready=std::isfinite(condensed_objective_cvx.scalar_shift) &&
                    condensed_objective_cvx.scalar_shift>=0.0;
                for(const auto& term:condensed_objective_cvx.terms) {
                  ambient_ready=ambient_ready && std::isfinite(term.beta) &&
                      term.beta>=0.0 &&
                      static_cast<int>(term.direction.size())==problem.n;
                  if(!ambient_ready) break;
                  double direction_norm2=0.0;
                  for(double value:term.direction) {
                    if(!std::isfinite(value)) { ambient_ready=false; break; }
                    direction_norm2+=value*value;
                  }
                  ambient_ready=ambient_ready &&
                      std::isfinite(direction_norm2) && direction_norm2>0.0;
                  if(!ambient_ready) break;
                }
                if(ambient_ready) {
                  if(st_.strict_scalar_certificate) {
                    // The strict oracle only adds a scalar shift, hence its
                    // Gershgorin certificate translates exactly.
                    ambient_final=ambient_initial+
                        condensed_objective_cvx.scalar_shift;
                  } else {
                    // Verify with an independent Ritz search.  Failure of this
                    // stronger ambient route is recoverable via the projected
                    // tangent oracle below.
                    ambient_final=condensed_curvature.smallest_eigenpair(
                        condensed_objective_hessian,
                        condensed_objective_cvx).lambda_min;
                  }
                  const double acceptance_floor=st_.curvature_margin-
                      10.0*st_.curvature_tolerance;
                  ambient_ready=std::isfinite(ambient_initial) &&
                      std::isfinite(ambient_final) &&
                      ambient_final>=acceptance_floor;
                  if(!ambient_ready) ambient_failure="submargin-global-ritz";
                }
              } catch(const std::exception& e) {
                ambient_failure=e.what();
                ambient_ready=false;
              }
              if(ambient_ready) {
                const int oracle_calls=std::max(
                    0,condensed_curvature.calls()-calls_before);
                result.lazy_dca_condensed_tangent_curvature_calls+=oracle_calls;
                result.lazy_dca_condensed_tangent_rank_max=std::max(
                    result.lazy_dca_condensed_tangent_rank_max,
                    condensed_objective_cvx.rank());
                result.lazy_dca_condensed_scalar_safeguard_updates_max=std::max(
                    result.lazy_dca_condensed_scalar_safeguard_updates_max,
                    condensed_objective_cvx.scalar_shift>0.0 ? 1 : 0);
                result.lazy_dca_condensed_tangent_shift_max=std::max(
                    result.lazy_dca_condensed_tangent_shift_max,
                    condensed_objective_cvx.scalar_shift);
                result.lazy_dca_condensed_initial_tangent_min_eigenvalue=
                    ambient_initial;
                result.lazy_dca_condensed_final_tangent_min_eigenvalue=
                    ambient_final;
                if(st_.verbose)
                  std::cout << "[CONDENSED-CURVATURE] k="<<outer
                            << " mode=ambient-global"
                            << " lambda="<<std::scientific
                            << ambient_initial<<"->"<<ambient_final
                            << " rank="<<condensed_objective_cvx.rank()
                            << " shift="<<condensed_objective_cvx.scalar_shift
                            << " oracle_calls="<<oracle_calls
                            << " normal_reg="<<condensed_geometry.regularization()
                            << " multiplier=transport-or-fit"
                            << "\n"<<std::defaultfloat;
              } else {
                // Discard any partial ambient correction before invoking the
                // independently verified projected fallback.
                condensed_objective_cvx=Convexification{};
                if(st_.verbose)
                  std::cout << "[CONDENSED-CURVATURE-FALLBACK] k="<<outer
                            << " ambient_message="
                            << (ambient_failure.empty()?"nonfinite-model":ambient_failure)
                            << "\n";
              }
            }
            if(!ambient_ready) {
              const auto tangent=condensed_geometry.ensure_tangent_convex(
                  condensed_objective_hessian,condensed_objective_cvx,
                  st_.curvature_margin,st_.dca_condensed_max_tangent_rank);
              result.lazy_dca_condensed_tangent_curvature_calls+=tangent.oracle_calls;
              result.lazy_dca_condensed_tangent_rank_max=std::max(
                  result.lazy_dca_condensed_tangent_rank_max,tangent.rank_added);
              result.lazy_dca_condensed_scalar_safeguard_updates_max=std::max(
                  result.lazy_dca_condensed_scalar_safeguard_updates_max,
                  tangent.scalar_safeguard_updates);
              result.lazy_dca_condensed_tangent_shift_max=std::max(
                  result.lazy_dca_condensed_tangent_shift_max,
                  tangent.scalar_shift_added);
              result.lazy_dca_condensed_initial_tangent_min_eigenvalue=
                  tangent.initial_min_eigenvalue;
              result.lazy_dca_condensed_final_tangent_min_eigenvalue=
                  tangent.final_min_eigenvalue;
              if(!tangent.success) {
                result.status=Status::NumericalError;
                result.message="condensed tangent curvature model failed: "+
                    tangent.message;
                break;
              }
              if(st_.verbose)
                std::cout << "[CONDENSED-CURVATURE] k="<<outer
                          << " mode=projected"
                          << " lambda="<<std::scientific
                          << tangent.initial_min_eigenvalue<<"->"
                          << tangent.final_min_eigenvalue
                          << " rank="<<tangent.rank_added
                          << " shift="<<tangent.scalar_shift_added
                          << " scalar_updates="<<tangent.scalar_safeguard_updates
                          << " normal_reg="<<condensed_geometry.regularization()
                          << " multiplier=transport-or-fit"
                          << "\n"<<std::defaultfloat;
            }
            }
          }
          objective_hessian_override=&condensed_objective_hessian;
          active_objective_cvx=&condensed_objective_cvx;
        }

        const bool suppress_warm_this_outer=dca_warm_cooldown_remaining>0;
        if(suppress_warm_this_outer) {
          --dca_warm_cooldown_remaining;
          ++result.lazy_dca_warm_start_skips;
        }
        const double dca_state_scale=std::max(1.0,inf_norm(x));
        const double dca_step_target=std::max(
            st_.step_tolerance,
            std::min(st_.dca_step_target_maximum,
                     st_.dca_step_target_absolute+
                         st_.dca_step_target_relative*dca_state_scale));
        result.lazy_dca_last_step_target_inf=dca_step_target;
        // If the restored point still has a large tangential gradient, do not
        // first solve an almost unregularized equality QP merely to discover a
        // million-scale direction.  In a flat/null-space direction the
        // proximal model predicts ||d|| approximately ||grad_T||/tau.  Use
        // the fresh original-QCQP multiplier refit to initialize tau before
        // the first factorization; the ordinary solved-step controller remains
        // the final guard and can increase it further.
        if(!st_.dca_single_loop_primal_dual && st_.dca_step_control &&
           has_quadratic_equality && warm_mult_valid &&
           std::isfinite(result.model_stationarity_inf) &&
           result.model_stationarity_inf>dca_step_target) {
          const double predicted=std::max(
              st_.dca_step_control_min_prox,
              1.10*result.model_stationarity_inf/
                  std::max(st_.step_tolerance,dca_step_target));
          if(predicted>dca_model_prox*(1.0+1e-12)) {
            const double previous=dca_model_prox;
            dca_model_prox=std::min(st_.dca_inner_retry_max_prox,predicted);
            ++result.lazy_dca_step_control_predictions;
            result.lazy_dca_max_model_proximal=std::max(
                result.lazy_dca_max_model_proximal,
                single_loop_baseline_prox+dca_model_prox);
            if(st_.verbose)
              std::cout << "[LAZY-DCA-STEP-PREDICTOR] k="<<outer
                        << " stationarity="<<std::scientific
                        << result.model_stationarity_inf
                        << " target="<<dca_step_target
                        << " prox="<<previous<<"->"<<dca_model_prox
                        << "\n"<<std::defaultfloat;
          }
        }
        double accepted_raw_direction_inf=0.0;

        // Both the ordinary full-endpoint working-set check below and the
        // post-retraction check in the condensed globalization may request a
        // new majorant at the same outer point.  Keep the successful primal
        // and multiplier state across either kind of guarded expansion.
        Vec working_set_primal_seed;
        MultiplierEstimate working_set_multiplier_seed;
        bool working_set_transport_valid=false;
        int expansions=0;
        int post_retraction_expansions=0;
        bool extrapolation_attempted_this_outer=false;
        bool accepted_on_outer_extrapolation=false;
        int single_loop_model_retries=0;
        bool single_loop_retry_pending=false;
        const bool outer_started_in_side_recovery=
            !modeled_side_recovery_rows.empty();
        bool force_strict_inner_this_outer=false;
        if(outer_started_in_side_recovery) {
          // The tightened recovery majorant is a different model.  Its first
          // globalization must probe alpha=1 instead of inheriting a small
          // accepted alpha from the preceding relaxed model.
          condensed_alpha_history_valid=false;
          condensed_iterations_since_full_probe=0;
          condensed_alpha_one_streak=0;
        }

retry_majorant_after_condensed_side_expansion:
        ;
        InnerResult inner;
        BuiltSubproblem accepted_built;
        bool have_candidate=false;
        Vec candidate;
        Vec accepted_direction;
        double accepted_alpha=1.0;
        bool accepted_soc=false;
        bool accepted_retraction=false;
        bool accepted_single_loop_fallback_retraction=false;
        const auto prepare_single_loop_retry_snapshot=
            [&](const InnerResult& source,
                const BuiltSubproblem& source_built,
                Vec& equality_duals,
                Vec& inequality_duals,
                Vec& inequality_slacks,
                Vec& inequality_primal_residuals,
                std::vector<std::uint64_t>& inequality_keys) {
          const int me=static_cast<int>(source_built.p.equalities.rows.size());
          const int mi=static_cast<int>(source_built.p.inequalities.size());
          const auto finite=[](const Vec& values) {
            return std::all_of(values.begin(),values.end(),
                               [](double value) {
                                 return std::isfinite(value);
                               });
          };
          const bool source_is_single_loop_model=
              st_.dca_single_loop_primal_dual &&
              source_built.p.allow_dca_single_loop &&
              (source.success || source.dca_single_loop_accepted ||
               source.dca_single_loop_retryable);
          if(!source_is_single_loop_model ||
             static_cast<int>(source.dca_single_loop_initial_primal.size())!=
                 source_built.p.nvar ||
             static_cast<int>(
                 source.dca_single_loop_initial_equality_duals.size())!=me ||
             static_cast<int>(
                 source.dca_single_loop_initial_inequality_duals.size())!=mi ||
             static_cast<int>(
                 source.dca_single_loop_initial_inequality_slacks.size())!=mi ||
             static_cast<int>(
                 source_built.inequality_continuation_keys.size())!=mi ||
             !finite(source.dca_single_loop_initial_primal) ||
             !finite(source.dca_single_loop_initial_equality_duals) ||
             !finite(source.dca_single_loop_initial_inequality_duals) ||
             !finite(source.dca_single_loop_initial_inequality_slacks) ||
             !std::all_of(
                          source.dca_single_loop_initial_inequality_duals.begin(),
                          source.dca_single_loop_initial_inequality_duals.end(),
                          [](double value) { return value>=0.0; }) ||
             !std::all_of(
                          source.dca_single_loop_initial_inequality_slacks.begin(),
                          source.dca_single_loop_initial_inequality_slacks.end(),
                          [](double value) { return value>0.0; }))
            return false;

          Vec proposed_primal_residuals(static_cast<std::size_t>(mi),0.0);
          for(int i=0;i<mi;++i) {
            const double phi=source_built.p.inequalities[
                static_cast<std::size_t>(i)].value(
                    source.dca_single_loop_initial_primal);
            const double residual=phi+
                source.dca_single_loop_initial_inequality_slacks[
                    static_cast<std::size_t>(i)];
            if(!std::isfinite(phi) || !std::isfinite(residual))
              return false;
            proposed_primal_residuals[static_cast<std::size_t>(i)]=residual;
          }

          // The expanded primal is a provenance/finiteness check only; the
          // retry seeds lambda/z/s and the associated primal-slack residuals
          // at the unchanged outer point, not this vector.  Exact affine
          // elimination can legitimately map reduced
          // y=0 to a nonzero particular solution, so its magnitude is not an
          // equality-funnel violation and must not exhaust proximal retries.
          // Roll back to the state at the origin of the rejected one-step
          // transaction.  Its final lambda/z/s belongs to x+d and must not seed
          // a rebuilt majorant at the unchanged outer point x.
          equality_duals=source.dca_single_loop_initial_equality_duals;
          inequality_duals=source.dca_single_loop_initial_inequality_duals;
          inequality_slacks=source.dca_single_loop_initial_inequality_slacks;
          inequality_primal_residuals=std::move(proposed_primal_residuals);
          inequality_keys=source_built.inequality_continuation_keys;
          return true;
        };
        const auto schedule_single_loop_model_retry=
            [&](const InnerResult& rejected,
                const BuiltSubproblem& rejected_built,
                const char* reason) {
          const bool rejected_single_loop_model=
              st_.dca_single_loop_primal_dual &&
              rejected_built.p.allow_dca_single_loop &&
              (rejected.success || rejected.dca_single_loop_accepted);
          if(!rejected_single_loop_model ||
             single_loop_model_retries>=st_.dca_inner_retry_max)
            return false;
          const double first=std::max(
              st_.dca_inner_retry_initial_prox,
              st_.dca_step_control_min_prox);
          const double proposed_next=dca_model_prox<=0.0
              ? first
              : std::max(first,st_.dca_inner_retry_growth*dca_model_prox);
          const double next=std::min(
              st_.dca_inner_retry_max_prox,proposed_next);
          if(!std::isfinite(next) ||
             !(next>dca_model_prox*(1.0+1e-12)))
            return false;
          Vec retry_equality_duals;
          Vec retry_inequality_duals;
          Vec retry_inequality_slacks;
          Vec retry_inequality_primal_residuals;
          std::vector<std::uint64_t> retry_inequality_keys;
          if(!prepare_single_loop_retry_snapshot(
                 rejected,rejected_built,retry_equality_duals,
                 retry_inequality_duals,retry_inequality_slacks,
                 retry_inequality_primal_residuals,
                 retry_inequality_keys))
            return false;

          // Commit proximal weight and rollback state as one transaction only
          // after the entire proposal has been validated.
          const double previous=dca_model_prox;
          dca_model_prox=next;
          dca_model_clean_steps=0;
          ++single_loop_model_retries;
          ++result.dca_single_loop_outer_retries;
          ++result.lazy_dca_inner_retries;
          ++result.lazy_dca_proximal_retries;
          result.lazy_dca_max_model_proximal=std::max(
              result.lazy_dca_max_model_proximal,
              single_loop_baseline_prox+dca_model_prox);
          continuation_equality_duals=std::move(retry_equality_duals);
          continuation_inequality_duals=std::move(retry_inequality_duals);
          continuation_inequality_slacks=std::move(retry_inequality_slacks);
          continuation_inequality_primal_residuals=
              std::move(retry_inequality_primal_residuals);
          continuation_inequality_keys=
              std::move(retry_inequality_keys);
          continuation_valid=true;
          working_set_transport_valid=false;
          force_strict_inner_this_outer=false;
          single_loop_retry_pending=true;
          if(st_.verbose)
            std::cout << "[DCA-SINGLE-LOOP-MODEL-RETRY] k="<<outer
                      << " reason="<<reason
                      << " prox="<<std::scientific<<previous
                      << "->"<<dca_model_prox
                      << " retry="<<single_loop_model_retries
                      << "/"<<st_.dca_inner_retry_max
                      << " action=one-predictor-corrector\n"
                      << std::defaultfloat;
          return true;
        };
        while(expansions<=st_.max_working_set_expansions) {
          BuiltSubproblem built;
          bool inner_ready=false;
          bool force_cold=
              suppress_warm_this_outer && !single_loop_retry_pending;
          int proximal_retries=0;
          int step_control_retries=0;
          double trial_model_prox=dca_model_prox;

          while(true) {
            Settings build_st=st_;
            build_st.dca_extra_proximal_regularization +=
                single_loop_baseline_prox+trial_model_prox;
            {
            AccumulateSeconds timer{result.primary_model_build_seconds};
            built=build_dca_subproblem(problem,x,*active_objective_cvx,ineq_cvx,
                                       working_set,modeled_side_recovery_rows,
                                       build_st,
                                       lazyqp_flow_lift_plan,
                                       objective_hessian_override);
            }
            result.lazy_dca_condensed_modeled_side_filter_recovery_margin_rows+=
                built.modeled_side_recovery_margin_rows;
            if(built.elastic_equalities) ++result.lazy_dca_elastic_subproblems;
            const bool use_working_set_transport=working_set_transport_valid &&
                static_cast<int>(working_set_primal_seed.size())==built.p.nvar &&
                std::all_of(working_set_primal_seed.begin(),
                            working_set_primal_seed.end(),
                            [](double v){ return std::isfinite(v); });
            bool direct_continuation=false;
            if(!force_cold) {
              if(use_working_set_transport) {
                seed_inner_dual_warm_start(built,problem,
                                           working_set_multiplier_seed,true,st_);
              } else {
                const DirectContinuationSeedResult direct_seed=
                    seed_direct_inner_dual_continuation(
                    built,continuation_equality_duals,continuation_inequality_duals,
                    continuation_inequality_slacks,
                    continuation_inequality_primal_residuals,
                    continuation_inequality_keys,continuation_valid,st_);
                direct_continuation=direct_seed.seeded;
                if(st_.dca_single_loop_primal_dual && direct_seed.attempted) {
                  ++result.dca_single_loop_keyed_transport_attempts;
                  if(direct_seed.seeded) {
                    ++result.dca_single_loop_keyed_transport_acceptances;
                    result.dca_single_loop_keyed_transport_matched_rows+=
                        direct_seed.matched_rows;
                    result.dca_single_loop_keyed_transport_new_rows+=
                        direct_seed.new_rows;
                    result.dca_single_loop_keyed_transport_dropped_rows+=
                        direct_seed.dropped_rows;
                    result.dca_single_loop_keyed_transport_centrality_caps+=
                        direct_seed.centrality_capped_rows;
                    if(direct_seed.new_rows>0 || direct_seed.dropped_rows>0)
                      ++result.dca_single_loop_keyed_transport_churn_acceptances;
                  } else {
                    ++result.dca_single_loop_keyed_transport_rejections;
                  }
                  if(st_.verbose)
                    std::cout << "[DCA-SINGLE-LOOP-ROW-TRANSPORT] k="
                              << outer
                              << " accepted="<<(direct_seed.seeded?1:0)
                              << " matched="<<direct_seed.matched_rows
                              << " new="<<direct_seed.new_rows
                              << " dropped="<<direct_seed.dropped_rows
                              << " capped="<<direct_seed.centrality_capped_rows
                              << " reason="<<direct_seed.reason<<"\n";
                }
                if(!direct_continuation)
                  seed_inner_dual_warm_start(built,problem,warm_mult,warm_mult_valid,st_);
              }
            } else {
              built.p.has_dual_warm_start=false;
              built.p.initial_equality_duals.clear();
              built.p.initial_inequality_duals.clear();
            }
            if(use_working_set_transport) {
              built.p.has_primal_warm_start=true;
              built.p.initial_primal=working_set_primal_seed;
            }

            // Only the primary majorant at the accepted outer origin may
            // expose an inexact single-loop state.  Working-set expansion
            // starts from a nonzero restricted-model coordinate and retains
            // the strict contract until that transaction is complete.
            built.p.allow_dca_single_loop=
                st_.dca_single_loop_primal_dual &&
                !force_strict_inner_this_outer &&
                !use_working_set_transport &&
                (!single_loop_retry_pending || direct_continuation);
            // The primary outer loop checks the true QCQP filter immediately
            // before commit.  Let it own provisional endpoint globalization
            // when the user disabled the inner forcing policy; auxiliary and
            // standalone kernel calls retain transactional contraction.
            built.p.allow_dca_single_loop_outer_filter_progress=
                built.p.allow_dca_single_loop &&
                !st_.lazyqp_inexact_forcing;
            built.p.allow_dca_single_loop_dual_bootstrap=
                built.p.allow_dca_single_loop && outer==0 &&
                !single_loop_retry_pending;

            if(single_loop_retry_pending && !direct_continuation) {
              inner.message=
                  "single-loop model retry lost its direct lambda/z/s state";
              break;
            }
            if(st_.dca_single_loop_primal_dual &&
               built.p.inequalities.empty()) {
              // An empty barrier system would otherwise escape through the
              // historical equality-QP direct solve below.  Single-loop is a
              // primal-dual inequality continuation protocol, so this model
              // fails before any numerical factorization.
              inner.message=
                  "single-loop primal-dual majorant requires at least one inequality row";
              break;
            }
            const bool attempted_single_loop_direct_step=
                built.p.allow_dca_single_loop && direct_continuation;
            single_loop_retry_pending=false;

            persistent_lazyqp_settings=inner_settings(
                !force_strict_inner_this_outer);
            persistent_lazyqp_settings.dca_single_loop_primal_dual=
                st_.dca_single_loop_primal_dual &&
                !force_strict_inner_this_outer;
            if(st_.dca_single_loop_primal_dual) {
              persistent_lazyqp_settings.inner_dual_warm_start=true;
              persistent_lazyqp_settings.inner_inequality_dual_warm_start=true;
            }
            {
            AccumulateSeconds timer{result.primary_inner_seconds};
            if(uses_lazyqp_kernel && !built.p.inequalities.empty()) {
              inner=st_.inner_backend=="socp"
                  ? persistent_socp.solve(built.p)
                  : (st_.inner_backend=="lazyqp-sparse" &&
                     st_.dca_single_loop_primal_dual
                         ? persistent_sparse.solve(built.p)
                         : persistent_lazyqp.solve(built.p));
            } else
              inner=solve_inner(built.p,persistent_lazyqp_settings,
                                  persistent_equality_factor.get());
            }
            const bool attempted_warm=built.p.has_dual_warm_start && !force_cold;
            record_inner(inner);
            const bool usable_inner=inner.success ||
                (built.p.allow_dca_single_loop &&
                 inner.dca_single_loop_accepted);
            if(usable_inner && (int)inner.y.size()>=problem.n) {
              double raw_direction_inf=0.0;
              for(int j=0;j<problem.n;++j)
                raw_direction_inf=std::max(raw_direction_inf,
                    std::abs(inner.y[static_cast<std::size_t>(j)]));
              result.lazy_dca_max_raw_direction_inf=std::max(
                  result.lazy_dca_max_raw_direction_inf,raw_direction_inf);
              if(st_.dca_step_control && has_quadratic_equality &&
                 raw_direction_inf>dca_step_target*(1.0+1e-10) &&
                 step_control_retries<st_.dca_step_control_max_retries &&
                 trial_model_prox<st_.dca_inner_retry_max_prox) {
                const double ratio=raw_direction_inf/
                    std::max(st_.step_tolerance,dca_step_target);
                const double gradient_scale=inf_norm(problem.objective.gradient(x));
                const double scale_suggestion=gradient_scale/
                    std::max(st_.step_tolerance,dca_step_target);
                double next=0.0;
                if(trial_model_prox<=0.0) {
                  next=std::max({st_.dca_step_control_min_prox,
                      scale_suggestion,
                      st_.dca_step_control_min_prox*std::min(1e8,ratio)});
                } else {
                  const double multiplier=std::max(
                      st_.dca_step_control_growth,
                      std::min(100.0,std::sqrt(std::max(1.0,ratio))));
                  next=std::max(scale_suggestion,trial_model_prox*multiplier);
                }
                next=std::min(st_.dca_inner_retry_max_prox,
                    std::max(st_.dca_step_control_min_prox,next));
                if(next>trial_model_prox*(1.0+1e-12)) {
                  const bool provisional_retry=
                      st_.dca_single_loop_primal_dual &&
                      built.p.allow_dca_single_loop &&
                      (inner.success || inner.dca_single_loop_accepted) &&
                      !force_strict_inner_this_outer;
                  Vec retry_equality_duals;
                  Vec retry_inequality_duals;
                  Vec retry_inequality_slacks;
                  Vec retry_inequality_primal_residuals;
                  std::vector<std::uint64_t> retry_inequality_keys;
                  if(provisional_retry) {
                    // Rebuild the more proximal majorant at y=0 while carrying
                    // only a complete lambda/z/s snapshot.  The next kernel
                    // call remains a single predictor-corrector step; a
                    // malformed snapshot terminates this bounded retry path
                    // instead of silently bootstrapping a full inner solve.
                    if(!prepare_single_loop_retry_snapshot(
                           inner,built,retry_equality_duals,
                           retry_inequality_duals,retry_inequality_slacks,
                           retry_inequality_primal_residuals,
                           retry_inequality_keys)) {
                      inner.message=
                          "single-loop step-control retry state is incomplete";
                      break;
                    }
                  }
                  const double previous=trial_model_prox;
                  trial_model_prox=next;
                  if(provisional_retry) {
                    continuation_equality_duals=
                        std::move(retry_equality_duals);
                    continuation_inequality_duals=
                        std::move(retry_inequality_duals);
                    continuation_inequality_slacks=
                        std::move(retry_inequality_slacks);
                    continuation_inequality_primal_residuals=
                        std::move(retry_inequality_primal_residuals);
                    continuation_inequality_keys=
                        std::move(retry_inequality_keys);
                    continuation_valid=true;
                    single_loop_retry_pending=true;
                    ++result.dca_single_loop_outer_retries;
                  }
                  ++step_control_retries;
                  ++result.lazy_dca_step_control_retries;
                  ++result.lazy_dca_proximal_retries;
                  result.lazy_dca_max_model_proximal=std::max(
                      result.lazy_dca_max_model_proximal,
                      single_loop_baseline_prox+trial_model_prox);
                  force_cold=!provisional_retry;
                  if(!provisional_retry) {
                    continuation_valid=false;
                    continuation_equality_duals.clear();
                    continuation_inequality_duals.clear();
                    continuation_inequality_slacks.clear();
                    continuation_inequality_primal_residuals.clear();
                    continuation_inequality_keys.clear();
                  }
                  if(st_.verbose)
                    std::cout << "[LAZY-DCA-STEP-CONTROL] k="<<outer
                              << " raw="<<std::scientific<<raw_direction_inf
                              << " target="<<dca_step_target
                              << " prox="<<previous<<"->"<<trial_model_prox
                              << " retry="<<step_control_retries
                              << "\n"<<std::defaultfloat;
                  continue;
                }
              }
              if(attempted_warm) dca_warm_failure_streak=0;
              accepted_raw_direction_inf=raw_direction_inf;
              inner_ready=true;
              break;
            }

            // A provisional predictor-corrector transaction can fail its
            // local guard without corrupting the transported state.  In
            // single-loop mode, rebuild a genuinely different (more
            // proximal) majorant and take one new PC step.  Never consume a
            // near-complete strict inner solve here; only the final
            // certificate is allowed to perform a full adjustment.
            if(inner.dca_single_loop_retryable) {
              const double first=std::max(
                  st_.dca_inner_retry_initial_prox,
                  st_.dca_step_control_min_prox);
              const double next=trial_model_prox<=0.0 ? first : std::min(
                  st_.dca_inner_retry_max_prox,
                  std::max(first,
                      st_.dca_inner_retry_growth*trial_model_prox));
              if(proximal_retries>=st_.dca_inner_retry_max ||
                 !(next>trial_model_prox*(1.0+1e-12)))
                break;
              Vec retry_equality_duals;
              Vec retry_inequality_duals;
              Vec retry_inequality_slacks;
              Vec retry_inequality_primal_residuals;
              std::vector<std::uint64_t> retry_inequality_keys;
              if(!prepare_single_loop_retry_snapshot(
                     inner,built,retry_equality_duals,
                     retry_inequality_duals,retry_inequality_slacks,
                     retry_inequality_primal_residuals,
                     retry_inequality_keys)) {
                inner.message=
                    "retryable single-loop state lost its initial "
                    "lambda/z/s snapshot";
                break;
              }
              trial_model_prox=next;
              ++proximal_retries;
              ++result.lazy_dca_inner_retries;
              ++result.lazy_dca_proximal_retries;
              ++result.dca_single_loop_outer_retries;
              result.lazy_dca_max_model_proximal=std::max(
                  result.lazy_dca_max_model_proximal,
                  single_loop_baseline_prox+trial_model_prox);
              // Seed the rebuilt model from the origin of the rejected
              // transaction.  This is essential for a cold first model (and
              // for affine-reduced models), where no previous outer
              // continuation snapshot exists yet.
              continuation_equality_duals=std::move(retry_equality_duals);
              continuation_inequality_duals=
                  std::move(retry_inequality_duals);
              continuation_inequality_slacks=
                  std::move(retry_inequality_slacks);
              continuation_inequality_primal_residuals=
                  std::move(retry_inequality_primal_residuals);
              continuation_inequality_keys=
                  std::move(retry_inequality_keys);
              continuation_valid=true;
              single_loop_retry_pending=true;
              force_cold=false;
              if(st_.verbose)
                std::cout << "[DCA-SINGLE-LOOP-RETRY] k="<<outer
                          << " reason=inner-guard mode=model-prox"
                          << " retry="<<proximal_retries
                          << " prox="<<std::scientific<<trial_model_prox
                          << " message="<<inner.message
                          << "\n"<<std::defaultfloat;
              continue;
            }

            continuation_valid=false;
            continuation_equality_duals.clear();
            continuation_inequality_duals.clear();
            continuation_inequality_slacks.clear();
            continuation_inequality_primal_residuals.clear();
            continuation_inequality_keys.clear();

            if(attempted_single_loop_direct_step) {
              // A rejected one-step transaction is an outer-level failure (or
              // a new guarded proximal majorant), never authority to continue
              // into the historical cold/full inner solve here.
              break;
            }

            if(!force_cold && built.p.has_dual_warm_start) {
              force_cold=true;
              ++result.lazy_dca_inner_retries;
              ++result.lazy_dca_cold_retries;
              ++dca_warm_failure_streak;
              if(st_.dca_warm_failure_limit>0 &&
                 dca_warm_failure_streak>=st_.dca_warm_failure_limit) {
                dca_warm_failure_streak=0;
                dca_warm_cooldown_remaining=std::max(
                    dca_warm_cooldown_remaining,st_.dca_warm_cooldown);
                ++result.lazy_dca_warm_start_suppressions;
                if(st_.verbose)
                  std::cout << "[LAZY-DCA-WARM-GUARD] k="<<outer
                            << " action=suppress cooldown="
                            << dca_warm_cooldown_remaining << "\n";
              }
              if(st_.verbose)
                std::cout << "[LAZY-DCA-INNER-RETRY] k="<<outer
                          << " reason=warm-start-failure mode=cold"
                          << " message="<<inner.message<<"\n";
              continue;
            }

            if(proximal_retries<st_.dca_inner_retry_max) {
              const double first=std::max(0.0,st_.dca_inner_retry_initial_prox);
              if(trial_model_prox<=0.0) trial_model_prox=first;
              else trial_model_prox=std::min(st_.dca_inner_retry_max_prox,
                    std::max(first,st_.dca_inner_retry_growth*trial_model_prox));
              ++proximal_retries;
              ++result.lazy_dca_inner_retries;
              ++result.lazy_dca_proximal_retries;
              result.lazy_dca_max_model_proximal=std::max(
                  result.lazy_dca_max_model_proximal,
                  single_loop_baseline_prox+trial_model_prox);
              force_cold=true;
              if(st_.verbose)
                std::cout << "[LAZY-DCA-INNER-RETRY] k="<<outer
                          << " reason=inner-failure mode=model-prox"
                          << " retry="<<proximal_retries
                          << " prox="<<std::scientific<<trial_model_prox
                          << " message="<<inner.message
                          << "\n"<<std::defaultfloat;
              continue;
            }
            break;
          }

          if(!inner_ready) {
            result.status=Status::InnerFailure;
            result.message="convex Lazy-DCA subproblem failed after cold/proximal retries with "+
                st_.inner_backend+": "+inner.message;
            break;
          }
          if(trial_model_prox>dca_model_prox) {
            dca_model_prox=trial_model_prox;
            dca_model_clean_steps=0;
          }

          candidate=x;
          Vec d(problem.n,0.0);
          std::copy(inner.y.begin(),inner.y.begin()+problem.n,d.begin());
          axpy(1.0,d,candidate);
          // Quadratic equalities are NOT projected or convexified here.  They
          // already enter build_dca_subproblem() through the classical SQP
          // equation h(x_k)+grad h(x_k)^T d=0.  Globalization of that linearized
          // step is performed below on the original QCQP functions.

          std::vector<int> violated_omitted;
          bool active_violation=false;
          const double violation_threshold=direct_start_recovery_active
              ? 0.0 : st_.feasibility_tolerance;
          for(int i=0;i<(int)problem.inequalities.size();++i) {
            const double gi=structural_quadratic_value(
                problem.inequalities[i],candidate);
            if(gi>violation_threshold) {
              if(!working_set.count(i)) violated_omitted.push_back(i);
              else active_violation=true;
            }
          }
          if(!violated_omitted.empty()) {
            const bool single_loop_expansion=
                st_.dca_single_loop_primal_dual &&
                built.p.allow_dca_single_loop;
            if(single_loop_expansion) {
              // An enlarged working set is a new majorant at the unchanged
              // outer origin.  Roll back to the original y=0 lambda/z/s
              // snapshot and remap it by semantic row key; carrying the
              // rejected full-step primal seed would disable the one-step route
              // and silently launch a historical strict solve.
              Vec retry_equality_duals;
              Vec retry_inequality_duals;
              Vec retry_inequality_slacks;
              Vec retry_inequality_primal_residuals;
              std::vector<std::uint64_t> retry_inequality_keys;
              if(!prepare_single_loop_retry_snapshot(
                     inner,built,retry_equality_duals,
                     retry_inequality_duals,retry_inequality_slacks,
                     retry_inequality_primal_residuals,
                     retry_inequality_keys)) {
                result.status=Status::InnerFailure;
                result.message=
                    "single-loop working-set expansion lost its initial "
                    "lambda/z/s snapshot";
                break;
              }
              continuation_equality_duals=std::move(retry_equality_duals);
              continuation_inequality_duals=std::move(retry_inequality_duals);
              continuation_inequality_slacks=std::move(retry_inequality_slacks);
              continuation_inequality_primal_residuals=
                  std::move(retry_inequality_primal_residuals);
              continuation_inequality_keys=std::move(retry_inequality_keys);
              continuation_valid=true;
              single_loop_retry_pending=true;
              working_set_transport_valid=false;
              ++result.dca_single_loop_outer_retries;
            } else {
              // Historical strict mode preserves the successful restricted
              // primal endpoint as an infeasible-start seed for the enlarged
              // convex subproblem.
              working_set_primal_seed=inner.y;
              working_set_multiplier_seed=multipliers_from_inner(
                  problem,built,inner,st_);
              working_set_transport_valid=
                  static_cast<int>(working_set_primal_seed.size())==
                      built.p.nvar &&
                  std::all_of(working_set_primal_seed.begin(),
                              working_set_primal_seed.end(),
                              [](double v){ return std::isfinite(v); }) &&
                  working_set_multiplier_seed.valid;
            }
            for(int i:violated_omitted) {
              working_set.insert(i);
              working_set_drop_streak[static_cast<std::size_t>(i)]=0;
              curvature.ensure_convex(problem.inequalities[i].hessian,ineq_cvx[i],0.0,
                                      st_.max_rank_per_quadratic);
              ++result.working_set_additions;
              if(direct_start_recovery_active)
                ++result.dca_single_loop_direct_start_recovery_working_set_additions;
            }
            ++expansions;
            if(st_.verbose)
              std::cout << "[LAZY-DCA-EXPAND] k="<<outer<<" added="<<violated_omitted.size()
                        <<" ws="<<working_set.size()
                        <<" mode="<<(single_loop_expansion
                                       ? "single-loop-row-transport"
                                       : "historical-primal-transport")
                        <<"\n";
            continue;
          }
          if(active_violation && !inner.dca_single_loop_accepted) {
            if(!st_.dca_single_loop_primal_dual &&
               inner.lazyqp_inexact_forcing_accepted &&
               !inner.dca_single_loop_accepted &&
               !force_strict_inner_this_outer) {
              // The original modeled constraint check is stronger than the
              // forcing contract.  Re-solve the identical majorant under the
              // historical strict inner contract before reporting failure.
              force_strict_inner_this_outer=true;
              if(inner.lazyqp_inexact_forcing_accepted)
                ++result.lazyqp_inexact_forcing_outer_retries;
              if(inner.dca_single_loop_accepted)
                ++result.dca_single_loop_outer_retries;
              working_set_transport_valid=false;
              continuation_valid=false;
              continuation_equality_duals.clear();
              continuation_inequality_duals.clear();
              continuation_inequality_slacks.clear();
              continuation_inequality_primal_residuals.clear();
              continuation_inequality_keys.clear();
              if(st_.verbose)
                std::cout << (inner.dca_single_loop_accepted
                                  ? "[DCA-SINGLE-LOOP-FALLBACK] k="
                                  : "[LAZY-DCA-FORCING-FALLBACK] k=")<<outer
                          << " reason=modeled-side action=strict-resolve\n";
              goto retry_majorant_after_condensed_side_expansion;
            }
            result.status=Status::InnerFailure;
            result.message="convex subproblem accuracy insufficient: modeled majorant constraint violated in original QCQP";
            break;
          }
          accepted_built=std::move(built);
          accepted_direction=std::move(d);
          have_candidate=true;
          break;
        }
        if(result.status==Status::InnerFailure ||
           result.status==Status::LineSearchFailure) break;
        if(!have_candidate) {
          result.status=Status::InnerFailure;
          result.message="Lazy-DCA working-set expansion limit reached";
          break;
        }

        // This marks every result produced by the primary single-loop model,
        // including the rare case where its sole PC step already satisfies the
        // strict inner tolerance (or the transported state is a zero-step
        // strict solution).  Classification follows the model route, not the
        // provisional return marker: otherwise a successful one-step result
        // could accidentally re-enable historical auxiliary solves.
        const bool single_loop_primary_route=
            st_.dca_single_loop_primal_dual &&
            accepted_built.p.allow_dca_single_loop;
        const bool skip_single_loop_auxiliaries=single_loop_primary_route;

        Feasibility fcand=structural_feasibility(problem,candidate);
        double objcand=structural_quadratic_value(problem.objective,candidate);
        const double monotone_tol=100.0*st_.inner_tolerance*
            std::max(1.0,std::abs(objcur));

        if(direct_start_recovery_active) {
          // Phase I was bypassed only inside a small, fixed original-QCQP
          // tube.  Until the side constraints first enter their strict tube,
          // globalization is an h-only filter: objective decrease is neither
          // required nor allowed to compensate for increased infeasibility.
          bool h_accepted=false;
          double alpha=1.0;
          for(int ls=0;ls<st_.line_search_max_steps;++ls) {
            Vec trial=x;
            Vec scaled=accepted_direction;
            for(double& value:scaled) value*=alpha;
            axpy(1.0,scaled,trial);

            DirectStartEvaluation trial_evaluation;
            if(finite_state(trial))
              trial_evaluation=evaluate_direct_start(problem,trial);
            bool all_violated_modeled=trial_evaluation.finite;
            if(trial_evaluation.finite) {
              for(int idx:trial_evaluation.violated_inequalities) {
                if(!working_set.count(idx)) {
                  all_violated_modeled=false;
                  break;
                }
              }
            }
            const Feasibility trial_feasibility=
                trial_evaluation.feasibility;
            const auto h_check=check_direct_start_recovery_step(
                fcur,trial_feasibility,alpha,st_.feasibility_tolerance,
                st_.dca_single_loop_direct_start_theta_cap,
                st_.dca_single_loop_direct_start_gamma_h,
                trial_evaluation.finite,
                all_violated_modeled);

            ++result.dca_single_loop_direct_start_recovery_attempts;
            if(trial_evaluation.finite)
              result.dca_single_loop_direct_start_recovery_max_theta=std::max(
                  result.dca_single_loop_direct_start_recovery_max_theta,
                  trial_feasibility.total_inf);
            if(h_check.accepted()) {
              ++result.dca_single_loop_direct_start_recovery_acceptances;
            } else {
              if(!h_check.finite_original_evaluation)
                ++result.dca_single_loop_direct_start_recovery_rejections_nonfinite;
              if(!h_check.bounds_strict)
                ++result.dca_single_loop_direct_start_recovery_rejections_bounds;
              if(!h_check.below_theta_cap)
                ++result.dca_single_loop_direct_start_recovery_rejections_cap;
              if(!h_check.contracts_theta)
                ++result.dca_single_loop_direct_start_recovery_rejections_contraction;
              if(!h_check.all_violated_inequalities_modeled)
                ++result.dca_single_loop_direct_start_recovery_rejections_unmodeled;
            }

            if(st_.verbose)
              std::cout << "[DCA-SINGLE-LOOP-DIRECT-H] k=" << outer
                        << " alpha=" << std::scientific << alpha
                        << " theta=" << fcur.total_inf << "->"
                        << trial_feasibility.total_inf
                        << " target=" << h_check.theta_target
                        << " bound=" << trial_feasibility.bound_inf
                        << " cap="
                        << st_.dca_single_loop_direct_start_theta_cap
                        << " modeled=" << (all_violated_modeled?1:0)
                        << " objective=" << objcur << "->"
                        << trial_evaluation.objective
                        << " accepted=" << (h_check.accepted()?1:0)
                        << "\n" << std::defaultfloat;

            if(h_check.accepted()) {
              filter_insert(filter,fcur.total_inf,objcur);
              candidate=std::move(trial);
              fcand=trial_feasibility;
              objcand=trial_evaluation.objective;
              accepted_alpha=alpha;
              h_accepted=true;
              if(condensed_equality_mode)
                ++result.lazy_dca_condensed_accepted_steps;
              else if(has_quadratic_equality)
                ++result.lazy_dca_linearized_equality_steps;
              ++result.lazy_dca_equality_feasibility_steps;
              result.lazy_dca_last_equality_step_alpha=alpha;
              if(has_quadratic_equality) {
                const double contracted=
                    st_.dca_equality_funnel_contraction*equality_funnel;
                const double trial_guard=st_.dca_equality_funnel_growth*
                    fcand.equality_inf;
                equality_funnel=std::max(st_.feasibility_tolerance,
                    std::min(equality_funnel,
                             std::max(contracted,trial_guard)));
                result.lazy_dca_final_equality_funnel=equality_funnel;
              }
              if(fcand.inequality_inf<=final_feasibility_tube_limit(st_)) {
                direct_start_side_strict_latched=true;
                ++result.dca_single_loop_direct_start_strict_latches;
                if(st_.verbose)
                  std::cout << "[DCA-SINGLE-LOOP-DIRECT-LATCH] k="
                            << outer << " inequality=" << std::scientific
                            << fcand.inequality_inf
                            << " strict-limit="
                            << final_feasibility_tube_limit(st_)
                            << "\n" << std::defaultfloat;
              }
              break;
            }
            ++result.line_search_backtracks;
            alpha*=st_.line_search_contraction;
            if(alpha<st_.minimum_step_length) break;
          }
          if(!h_accepted) {
            result.status=Status::LineSearchFailure;
            result.message=
                "single-loop direct-start h-step failed its exact original-QCQP recovery guard";
            break;
          }
        } else if(elastic_equality_mode) {
          // Alternative equality globalization: nonlinear equalities are not
          // imposed as affine equations in the convex subproblem.  Instead,
          // two convex DC majorants share one elastic eta variable.  The line
          // search uses the ORIGINAL QCQP merit and deliberately avoids the
          // systematic equality SOC/retraction stack.
          Vec zero(accepted_built.p.nvar,0.0);
          const double model0=accepted_built.p.objective.value(zero);
          const double model_full=accepted_built.p.objective.value(inner.y);
          const double predicted_decrease=std::max(0.0,model0-model_full);
          const double phi0=objcur+st_.dca_equality_elastic_penalty*fcur.equality_inf+
              0.5*st_.dca_equality_elastic_quadratic*fcur.equality_inf*fcur.equality_inf;
          bool accepted=false;
          double alpha=1.0;
          Vec accepted_trial;
          Feasibility accepted_feas;
          double accepted_obj=objcur;
          // Literal single-loop models never schedule an auxiliary
          // retraction.  A rejected endpoint may only rebuild a proximal
          // model and execute another one-PC transaction.
          constexpr bool elastic_single_loop_retraction_fallback=false;
          const int elastic_schedules=
              elastic_single_loop_retraction_fallback?2:1;
          for(int schedule=0;schedule<elastic_schedules && !accepted;
              ++schedule) {
          const bool retraction_fallback_schedule=
              elastic_single_loop_retraction_fallback && schedule==1;
          if(retraction_fallback_schedule) {
            ++result.dca_single_loop_retraction_fallbacks;
            alpha=1.0;
          }
          const int elastic_line_search_steps=skip_single_loop_auxiliaries
              ? 1 : st_.line_search_max_steps;
          for(int ls=0;ls<elastic_line_search_steps;++ls) {
            Vec trial=x;
            Vec scaled=accepted_direction;
            for(double& v:scaled) v*=alpha;
            axpy(1.0,scaled,trial);
            if(!finite_state(trial)) {
              ++result.line_search_backtracks;
              alpha*=st_.line_search_contraction;
              if(alpha<st_.minimum_step_length) break;
              continue;
            }
            Feasibility tf=structural_feasibility(problem,trial);
            bool retraction_used_here=false;
            if((!skip_single_loop_auxiliaries ||
                retraction_fallback_schedule) &&
               st_.dca_equality_elastic_post_retraction &&
               tf.equality_inf>st_.feasibility_tolerance) {
              ++result.lazy_dca_equality_retraction_attempts;
              const auto retract=try_dca_equality_retraction(
                  problem,x,trial,scaled,accepted_built.elastic_cap,st_);
              result.lazy_dca_max_equality_retraction_inf=std::max(
                  result.lazy_dca_max_equality_retraction_inf,retract.correction_inf);
              if(retract.success) {
                trial=retract.x;
                tf=retract.feasibility;
                retraction_used_here=true;
                ++result.lazy_dca_equality_retraction_acceptances;
              } else {
                ++result.lazy_dca_equality_retraction_rejections;
              }
            }
            const double ft=structural_quadratic_value(problem.objective,trial);
            const double phit=ft+st_.dca_equality_elastic_penalty*tf.equality_inf+
                0.5*st_.dca_equality_elastic_quadratic*tf.equality_inf*tf.equality_inf;
            const double cap_noise=8192.0*std::numeric_limits<double>::epsilon()*
                std::max(1.0,accepted_built.elastic_cap);
            const bool within_cap=tf.equality_inf<=accepted_built.elastic_cap+cap_noise;
            const bool side_feasible=inequality_bound_violation(tf)<=
                final_feasibility_tube_limit(st_);
            const bool merit_ok=predicted_decrease>0.0
                ? phit<=phi0-st_.line_search_armijo*alpha*predicted_decrease
                : phit<=phi0+monotone_tol;
            if(side_feasible && within_cap && merit_ok) {
              accepted=true;
              accepted_trial=std::move(trial);
              accepted_feas=tf;
              accepted_obj=ft;
              accepted_alpha=alpha;
              accepted_retraction=retraction_used_here;
              accepted_single_loop_fallback_retraction=
                  retraction_fallback_schedule && retraction_used_here;
              ++result.lazy_dca_elastic_accepted_steps;
              if(accepted_built.elastic_index>=0 && accepted_built.elastic_index<(int)inner.y.size())
                result.lazy_dca_elastic_last_eta=std::max(0.0,accepted_built.elastic_base+inner.y[accepted_built.elastic_index]);
              if(st_.verbose) {
                const double e=(accepted_built.elastic_index>=0 &&
                    accepted_built.elastic_index<(int)inner.y.size())
                    ? inner.y[accepted_built.elastic_index] : 0.0;
                const double eta=std::max(0.0,accepted_built.elastic_base+e);
                std::cout << "[LAZY-DCA-ELASTIC] k="<<outer
                          << " alpha="<<std::scientific<<alpha
                          << " eta="<<eta
                          << " equality="<<tf.equality_inf
                          << " cap="<<accepted_built.elastic_cap
                          << " merit="<<phit
                          << " pred="<<predicted_decrease
                          << "\n"<<std::defaultfloat;
              }
              break;
            }
            ++result.line_search_backtracks;
            alpha*=st_.line_search_contraction;
            if(alpha<st_.minimum_step_length) break;
          }
          }
          if(!accepted) {
            if(single_loop_primary_route &&
               !force_strict_inner_this_outer &&
               schedule_single_loop_model_retry(
                   inner,accepted_built,"elastic-globalization")) {
              goto retry_majorant_after_condensed_side_expansion;
            }
            if(!st_.dca_single_loop_primal_dual &&
               inner.lazyqp_inexact_forcing_accepted &&
               !inner.dca_single_loop_accepted &&
               !force_strict_inner_this_outer) {
              force_strict_inner_this_outer=true;
              ++result.lazyqp_inexact_forcing_outer_retries;
              working_set_transport_valid=false;
              continuation_valid=false;
              continuation_equality_duals.clear();
              continuation_inequality_duals.clear();
              continuation_inequality_slacks.clear();
              continuation_inequality_primal_residuals.clear();
              continuation_inequality_keys.clear();
              if(st_.verbose)
                std::cout << "[LAZY-DCA-FORCING-FALLBACK] k="<<outer
                          << " reason=elastic-globalization"
                          << " action=strict-resolve\n";
              goto retry_majorant_after_condensed_side_expansion;
            }
            result.status=Status::LineSearchFailure;
            result.message="Lazy-DCA elastic-split equality merit line search failed";
            break;
          }
          candidate=std::move(accepted_trial);
          fcand=accepted_feas;
          objcand=accepted_obj;
        } else if(has_quadratic_equality) {
          // The large convex majorant keeps affine tangent equalities.  In the
          // ordinary route, globalization may use the generic SOC/retraction.
          // In condensed-retraction mode, every trial instead follows the fixed
          // normal map and solves the original quadratic equalities exactly in
          // equality space; the signed equality-Lagrangian curvature remains in
          // the tangent model and no legacy SOC is stacked on top of it.
          CondensedQuadraticRetraction* active_condensed_retraction=
              &condensed_geometry;
          bool condensed_retraction_prepared=false;
          std::string condensed_retraction_prepare_failure;
          const auto prepare_condensed_retraction=[&]() {
            if(!condensed_equality_mode || condensed_retraction_prepared)
              return true;
            const int base_active_bound_count=
                condensed_geometry.active_bound_coordinates(problem,x);
            const int predicted_active_bound_count=
                condensed_geometry.active_bound_coordinates(
                    problem,x,accepted_direction);
            int active_bound_count=predicted_active_bound_count;
            bool bound_lookahead_fallback=false;
            live_bound_face_coordinates=active_bound_count;
            if(active_bound_count>0 || condensed_single_normal_map) {
              std::string bound_map_message;
              if(!condensed_retraction.factor_bound_aware(
                     problem,x,accepted_direction,bound_map_message)) {
                // A predicted face can remove the final usable equality
                // normal.  Fall back to the historical base-only map; the
                // original bound and inequality checks below remain exact.
                bound_lookahead_fallback=true;
                active_bound_count=base_active_bound_count;
                live_bound_face_coordinates=active_bound_count;
                std::string base_map_message;
                if(!condensed_retraction.factor_bound_aware(
                       problem,x,base_map_message)) {
                  condensed_retraction_prepare_failure=
                      "bound-aware condensed normal operator failed after "
                      "look-ahead fallback: "+base_map_message+
                      " (look-ahead: "+bound_map_message+")";
                  return false;
                }
                active_condensed_retraction=&condensed_retraction;
              } else {
                active_condensed_retraction=&condensed_retraction;
              }
              if(active_condensed_retraction==&condensed_retraction) {
                if(active_bound_count>0)
                  ++result.lazy_dca_condensed_bound_face_factorizations;
                result.lazy_dca_condensed_max_active_bound_coordinates=std::max(
                    result.lazy_dca_condensed_max_active_bound_coordinates,
                    active_bound_count);
                result.lazy_dca_condensed_normal_regularization=std::max(
                    result.lazy_dca_condensed_normal_regularization,
                    condensed_retraction.regularization());
              }
            }
            if(st_.verbose)
              std::cout << "[CONDENSED-BOUND-MAP] k="<<outer
                        << " active="<<base_active_bound_count
                        << " lookahead="<<predicted_active_bound_count
                        << " fallback="<<(bound_lookahead_fallback?1:0)
                        << " operator="<<(active_bound_count>0?"face":
                            (condensed_single_normal_map?"single-normal":"geometry"))
                        << " locked="<<active_condensed_retraction->locked_coordinates()
                        << " normal_reg="<<std::scientific
                        << active_condensed_retraction->regularization()
                        << "\n"<<std::defaultfloat;
            condensed_retraction_prepared=true;
            return true;
          };
          if(condensed_equality_mode && !skip_single_loop_auxiliaries &&
             !prepare_condensed_retraction()) {
            result.status=Status::NumericalError;
            result.message=condensed_retraction_prepare_failure;
            break;
          }
          Vec zero(problem.n,0.0);
          const double model0=accepted_built.p.objective.value(zero);
          const double model_full=accepted_built.p.objective.value(accepted_direction);
          const double predicted_decrease=std::max(0.0,model0-model_full);
          const double predicted_floor=100.0*std::numeric_limits<double>::epsilon()*
              std::max(1.0,std::abs(model0));
          bool accepted=false;
          bool accepted_objective_step=false;
          bool accepted_feasibility_step=false;
          double alpha=1.0;
          Vec accepted_trial;
          Feasibility accepted_feas;
          double accepted_obj=objcur;
          const double raw_direction_inf=inf_norm(accepted_direction);
          const std::vector<int> current_working_set(
              working_set.begin(),working_set.end());
          const bool predictor_structure_same=condensed_alpha_history_valid &&
              current_working_set==condensed_last_working_set &&
              live_bound_face_coordinates==condensed_last_bound_face_coordinates;
          const double direction_ratio=condensed_last_direction_inf>0.0
              ? raw_direction_inf/condensed_last_direction_inf
              : std::numeric_limits<double>::infinity();
          const double prox_scale=std::max(
              {1.0,std::abs(dca_model_prox),
               std::abs(condensed_last_model_prox)});
          const bool predictor_model_same=predictor_structure_same &&
              std::isfinite(direction_ratio) && direction_ratio>=0.25 &&
              direction_ratio<=4.0 &&
              std::abs(dca_model_prox-condensed_last_model_prox)<=
                  64.0*std::numeric_limits<double>::epsilon()*prox_scale;
          // Scheduling an external probe does not require an identical
          // proximal weight: every endpoint is certified on the original
          // QCQP.  It still requires the same working-set/bound structure and
          // a comparable direction, preventing extrapolation across an active
          // model discontinuity.
          const bool extrapolation_structure_same=predictor_structure_same &&
              std::isfinite(direction_ratio) && direction_ratio>=0.25 &&
              direction_ratio<=4.0;
          const bool periodic_full_probe=condensed_alpha_history_valid &&
              condensed_iterations_since_full_probe+1>=
                  st_.dca_condensed_alpha_full_step_frequency;
          bool predictor_seeded=false;
          bool accepted_on_seeded_schedule=false;
          int predictor_leading_levels=0;
          double alpha_start=1.0;
          if(condensed_equality_mode && !skip_single_loop_auxiliaries &&
             st_.dca_condensed_alpha_predictor &&
             condensed_alpha_history_valid && predictor_model_same &&
             !periodic_full_probe) {
            alpha_start=std::min(
                1.0,st_.dca_condensed_alpha_growth*
                        condensed_last_accepted_alpha);
            alpha_start=std::max(alpha_start,st_.minimum_step_length);
            predictor_seeded=alpha_start<1.0-
                16.0*std::numeric_limits<double>::epsilon();
            if(predictor_seeded) {
              ++result.lazy_dca_condensed_alpha_predictor_uses;
              double skipped_alpha=1.0;
              while(skipped_alpha>alpha_start*(1.0+16.0*
                        std::numeric_limits<double>::epsilon()) &&
                    predictor_leading_levels<st_.line_search_max_steps) {
                skipped_alpha*=st_.line_search_contraction;
                ++predictor_leading_levels;
              }
              if(st_.verbose)
                std::cout << "[CONDENSED-ALPHA-PREDICTOR] k="<<outer
                          << " previous="<<std::scientific
                          << condensed_last_accepted_alpha
                          << " start="<<alpha_start
                          << " skipped="<<predictor_leading_levels
                          << "\n"<<std::defaultfloat;
            }
          } else if(condensed_equality_mode &&
                    !skip_single_loop_auxiliaries &&
                    st_.dca_condensed_alpha_predictor &&
                    condensed_alpha_history_valid) {
            ++result.lazy_dca_condensed_alpha_predictor_full_step_resets;
          }

          // A stable sequence of accepted full steps identifies the locally
          // contractive fixed-point regime in which scalar over-relaxation can
          // be useful.  This is only a probe: it gets one exact original-QCQP
          // globalization check and, on any rejection, cannot enter the
          // contracted alpha grid.  The next schedule starts at the historical
          // alpha=1 endpoint.
          const bool outer_extrapolation_probe=condensed_equality_mode &&
              !skip_single_loop_auxiliaries &&
              st_.dca_condensed_outer_extrapolation &&
              !extrapolation_attempted_this_outer &&
              condensed_alpha_history_valid && extrapolation_structure_same &&
              !predictor_seeded && !outer_started_in_side_recovery &&
              modeled_side_recovery_rows.empty() &&
              within_final_feasibility_tube(fcur.total_inf,st_) &&
              condensed_alpha_one_streak>=
                  st_.dca_condensed_outer_extrapolation_min_full_steps;

          const double normal_strict_gate=
              st_.dca_condensed_normal_strict_stationarity_factor*
              std::max(st_.stationarity_tolerance,
                       st_.dual_infeasibility_tolerance);
          const bool allow_adaptive_normal=condensed_equality_mode &&
              st_.dca_condensed_adaptive_normal_refinement &&
              warm_mult_valid &&
              std::isfinite(result.model_stationarity_inf) &&
              result.model_stationarity_inf>normal_strict_gate;

          const int base_line_search_schedules=predictor_seeded?2:1;
          // A one-PC result gets original-QCQP globalization only.  It may
          // request a bounded proximal model retry, but never a historical
          // equality retraction or strict resolution of the same model.
          constexpr bool single_loop_retraction_fallback_schedule=false;
          const int line_search_schedules=base_line_search_schedules+
              (outer_extrapolation_probe?1:0)+
              (single_loop_retraction_fallback_schedule?1:0);
          for(int schedule=0;
              schedule<line_search_schedules && !accepted;++schedule) {
            const bool retraction_fallback_schedule=
                single_loop_retraction_fallback_schedule &&
                schedule==line_search_schedules-1;
            const bool extrapolation_schedule=
                !retraction_fallback_schedule &&
                outer_extrapolation_probe && schedule==0;
            const int base_schedule=schedule-(outer_extrapolation_probe?1:0);
            alpha=retraction_fallback_schedule
                ? 1.0
                : extrapolation_schedule
                ? st_.dca_condensed_outer_extrapolation_alpha
                : (base_schedule==0?alpha_start:1.0);
            if(retraction_fallback_schedule) {
              ++result.dca_single_loop_retraction_fallbacks;
              if(condensed_equality_mode &&
                 !prepare_condensed_retraction()) {
                if(st_.verbose)
                  std::cout << "[DCA-SINGLE-LOOP-RETRACTION-FALLBACK] k="
                            <<outer
                            << " prepared=0 action=model-retry message="
                            <<condensed_retraction_prepare_failure<<"\n";
                continue;
              }
              if(st_.verbose)
                std::cout << "[DCA-SINGLE-LOOP-RETRACTION-FALLBACK] k="
                          <<outer
                          << " alpha=1 action=try-historical-retraction\n";
            } else if(extrapolation_schedule) {
              extrapolation_attempted_this_outer=true;
              ++result.lazy_dca_condensed_outer_extrapolation_attempts;
              if(st_.verbose)
                std::cout << "[CONDENSED-OUTER-EXTRAPOLATION] k="<<outer
                          << " probe-alpha="<<std::scientific<<alpha
                          << " full-step-streak="<<condensed_alpha_one_streak
                          << " action=probe-original-qcqp\n"
                          << std::defaultfloat;
            } else if(base_schedule>0) {
              ++result.lazy_dca_condensed_alpha_predictor_historical_retries;
              if(st_.verbose)
                std::cout << "[CONDENSED-ALPHA-PREDICTOR] k="<<outer
                          << " retry=historical-alpha-one\n";
            }
          const int schedule_line_search_steps=
              (extrapolation_schedule || retraction_fallback_schedule)
                  ? 1 : st_.line_search_max_steps;
          for(int ls=0;ls<schedule_line_search_steps;++ls) {
            Vec trial=x;
            Vec scaled=accepted_direction;
            for(double& v:scaled) v*=alpha;
            const double trial_predicted_decrease=condensed_equality_mode
                ? std::max(0.0,model0-accepted_built.p.objective.value(scaled))
                : predicted_decrease;
            axpy(1.0,scaled,trial);
            if(!finite_state(trial)) {
              // The over-relaxed probe is not part of the historical
              // contracted grid.  A non-finite endpoint rejects that one
              // probe and the next schedule restarts exactly at alpha=1.
              if(extrapolation_schedule) break;
              ++result.line_search_backtracks;
              alpha*=st_.line_search_contraction;
              if(alpha<st_.minimum_step_length) break;
              continue;
            }
            const double raw_bound_inf=bound_violation(problem,trial);
            const double raw_bound_scale=std::max(
                {1.0,inf_norm(x),inf_norm(trial)});

            Feasibility tf{};
            if(!condensed_equality_mode ||
               (skip_single_loop_auxiliaries &&
                !retraction_fallback_schedule))
              tf=structural_feasibility(problem,trial);
            bool retraction_used_here=false;
            if(condensed_equality_mode &&
               (!skip_single_loop_auxiliaries ||
                retraction_fallback_schedule)) {
              ++result.lazy_dca_condensed_retraction_attempts;
              ++result.lazy_dca_equality_retraction_attempts;
              const int retraction_refinements_before=
                  active_condensed_retraction->normal_refinement_steps();
              const auto retract=active_condensed_retraction->retract(
                  scaled,{},allow_adaptive_normal);
              result.lazy_dca_condensed_retraction_normal_solves+=
                  retract.normal_solves;
              result.lazy_dca_condensed_retraction_normal_refinement_steps+=
                  active_condensed_retraction->normal_refinement_steps()-
                  retraction_refinements_before;
              result.lazy_dca_condensed_continuation_attempts+=
                  retract.continuation_attempts;
              result.lazy_dca_condensed_continuation_acceptances+=
                  retract.continuation_acceptances;
              result.lazy_dca_condensed_newton_iterations+=
                  retract.newton_iterations;
              result.lazy_dca_condensed_krylov_iterations+=
                  retract.krylov_iterations;
              result.lazy_dca_condensed_backtracks+=
                  retract.line_search_backtracks;
              result.lazy_dca_condensed_affine_normal_map_reuses+=
                  retract.affine_normal_map_reuses;
              result.lazy_dca_condensed_newton_endpoint_reuses+=
                  retract.newton_endpoint_reuses;
              result.lazy_dca_condensed_quadratic_linesearch_reuses+=
                  retract.quadratic_linesearch_reuses;
              result.lazy_dca_condensed_quadratic_linesearch_predictor_builds+=
                  retract.quadratic_linesearch_predictor_builds;
              result.lazy_dca_condensed_quadratic_linesearch_exact_checks+=
                  retract.quadratic_linesearch_exact_checks;
              result.lazy_dca_condensed_quadratic_linesearch_early_rejections+=
                  retract.quadratic_linesearch_early_rejections;
              result.lazy_dca_condensed_quadratic_linesearch_rows_skipped+=
                  retract.quadratic_linesearch_rows_skipped;
              result.lazy_dca_condensed_last_residual_inf=
                  retract.equality_residual_inf;
              result.lazy_dca_condensed_max_correction_inf=std::max(
                  result.lazy_dca_condensed_max_correction_inf,
                  retract.correction_inf);
              result.lazy_dca_condensed_max_correction_ratio=std::max(
                  result.lazy_dca_condensed_max_correction_ratio,
                  retract.correction_ratio);
              result.lazy_dca_max_equality_retraction_inf=std::max(
                  result.lazy_dca_max_equality_retraction_inf,
                  retract.correction_inf);
              if(retract.success) {
                trial=retract.x;
                tf=retract.feasibility;
                retraction_used_here=true;
                ++result.lazy_dca_condensed_retraction_acceptances;
                ++result.lazy_dca_equality_retraction_acceptances;
                if(st_.verbose)
                  std::cout << "[CONDENSED-RETRACTION] k="<<outer
                            << " alpha="<<std::scientific<<alpha
                            << " accepted=1 corr="<<retract.correction_inf
                            << " basecorr="<<retract.baseline_correction_inf
                            << " incorr="<<retract.incremental_correction_inf
                            << " ratio="<<retract.correction_ratio
                            << " residual="<<retract.equality_residual_inf
                            << " continuation="<<retract.continuation_acceptances
                            << "/"<<retract.continuation_attempts
                            << " newton="<<retract.newton_iterations
                            << " krylov="<<retract.krylov_iterations
                            << " normal_solves="<<retract.normal_solves
                            << " normal_reg="<<active_condensed_retraction->regularization()
                            << " side="<<inequality_bound_violation(tf)
                            << "\n"<<std::defaultfloat;
              } else {
                ++result.lazy_dca_condensed_retraction_rejections;
                ++result.lazy_dca_equality_retraction_rejections;
                if(st_.verbose)
                  std::cout << "[CONDENSED-RETRACTION] k="<<outer
                            << " alpha="<<std::scientific<<alpha
                            << " accepted=0 corr="<<retract.correction_inf
                            << " basecorr="<<retract.baseline_correction_inf
                            << " incorr="<<retract.incremental_correction_inf
                            << " ratio="<<retract.correction_ratio
                            << " residual="<<retract.equality_residual_inf
                            << " message="<<retract.message
                            << "\n"<<std::defaultfloat;
                if(extrapolation_schedule) break;
                ++result.line_search_backtracks;
                alpha*=st_.line_search_contraction;
                if(alpha<st_.minimum_step_length) break;
                continue;
              }
            } else if((!skip_single_loop_auxiliaries ||
                       retraction_fallback_schedule) &&
                      st_.dca_equality_retraction) {
              ++result.lazy_dca_equality_retraction_attempts;
              const auto retract=try_dca_equality_retraction(
                  problem,x,trial,scaled,equality_funnel,st_);
              result.lazy_dca_max_equality_retraction_inf=std::max(
                  result.lazy_dca_max_equality_retraction_inf,
                  retract.correction_inf);
              if(retract.success) {
                trial=retract.x;
                tf=retract.feasibility;
                retraction_used_here=true;
                ++result.lazy_dca_equality_retraction_acceptances;
                if(st_.verbose)
                  std::cout << "[LAZY-DCA-RETRACTION] k="<<outer
                            << " alpha="<<std::scientific<<alpha
                            << " accepted=1 corr="<<retract.correction_inf
                            << " eq="<<tf.equality_inf
                            << " side="<<inequality_bound_violation(tf)
                            << " funnel="<<equality_funnel
                            << "\n"<<std::defaultfloat;
              } else {
                ++result.lazy_dca_equality_retraction_rejections;
              }
            }
            bool soc_attempted_here=false;
            bool soc_used_here=false;
            const double soc_trigger_ratio=st_.dca_iterated_soc
                ? std::max(st_.second_order_correction_trigger_ratio,
                           st_.dca_iterated_soc_trigger_ratio)
                : st_.second_order_correction_trigger_ratio;
            const double soc_trigger=soc_trigger_ratio*
                std::max(st_.feasibility_tolerance,fcur.total_inf);
            if(!skip_single_loop_auxiliaries &&
               !condensed_equality_mode && !retraction_used_here &&
               tf.equality_inf>st_.feasibility_tolerance &&
               st_.second_order_correction && tf.total_inf<=soc_trigger) {
              ++result.soc_attempts;
              soc_attempted_here=true;
              auto soc=second_order_correction(problem,x,trial,working_set,st_,
                                                  false,st_.dca_iterated_soc,true);
              if(soc.iterated_attempted) {
                ++result.lazy_dca_iterated_soc_attempts;
                if(soc.iterated_success) ++result.lazy_dca_iterated_soc_acceptances;
                else ++result.lazy_dca_iterated_soc_rejections;
                if(st_.verbose)
                  std::cout << "[LAZY-DCA-ITERATED-SOC] k="<<outer
                            << " alpha="<<std::scientific<<alpha
                            << " restored="<<(soc.iterated_success?1:0)
                            << " space=full-lifted"
                            << " model=all-constraint-defect"
                            << " theta="<<soc.feasibility.total_inf
                            << " corr="<<soc.correction_inf
                            << " ncorr="<<soc.normal_correction_inf
                            << " qpcorr="<<soc.qp_correction_inf
                            << " eqtarget="<<soc.equality_target_inf
                            << " eqdefect="<<soc.equality_defect_inf
                            << " ineqtarget="<<soc.inequality_target_violation_inf
                            << " ineqdefect="<<soc.inequality_defect_violation_inf
                            << " ineqrhs="<<soc.inequality_correction_rhs_inf
                            << "\n"<<std::defaultfloat;
              }
              result.soc_inner_iterations+=soc.inner_iterations;
              result.soc_factorizations+=soc.factorizations;
              result.inner_iterations+=soc.inner_iterations;
              result.kkt_factorizations+=soc.factorizations;
              result.kkt_symbolic_analyses+=soc.symbolic_analyses;
              if(soc.success) {
                trial=std::move(soc.x);
                tf=soc.feasibility;
                soc_used_here=true;
                if(st_.verbose) {
                  std::cout << "[LAZY-DCA-SOC] k="<<outer
                            << " alpha="<<std::scientific<<alpha
                            << " corr="<<soc.correction_inf
                            << " theta="<<tf.total_inf
                            << "\n"<<std::defaultfloat;
                }
              }
            }

            const double ft=structural_quadratic_value(problem.objective,trial);
            const double funnel_noise=8192.0*
                std::numeric_limits<double>::epsilon()*
                std::max(1.0,equality_funnel);
            const double effective_equality_funnel=
                skip_single_loop_auxiliaries
                    ? std::max(equality_funnel,
                               st_.dca_equality_funnel_auto_ceiling)
                    : equality_funnel;
            const bool within_funnel=
                tf.equality_inf<=effective_equality_funnel+funnel_noise;
            const double bound_noise=8192.0*
                std::numeric_limits<double>::epsilon()*raw_bound_scale;
            // The equality correction may not consume the feasibility tube by
            // leaking through a simple bound.  This prevents the accepted
            // O(alpha^2) drift that drove case118 to alpha=1/2048.
            const bool bound_not_worsened=!condensed_equality_mode ||
                tf.bound_inf<=raw_bound_inf+bound_noise;
            const double strict_side_limit=final_feasibility_tube_limit(st_);
            const bool strict_side_feasible=bound_not_worsened &&
                inequality_bound_violation(tf)<=strict_side_limit;
            bool modeled_side_filter_candidate=false;

            // Single-loop SQP/IPM iterates are intentionally infeasible: one
            // predictor-corrector step leaves an O(||d||^2) residual on a
            // curved active majorant.  Let the existing objective/violation
            // filter globalize that residual when every offending original
            // row was present in the solved model.  Bounds and omitted rows
            // retain their strict historical guard.
            if(skip_single_loop_auxiliaries && bound_not_worsened &&
               !strict_side_feasible && tf.bound_inf<=strict_side_limit) {
              bool modeled_offender=false;
              bool omitted_offender=false;
              for(int i=0;i<(int)problem.inequalities.size();++i) {
                if(structural_quadratic_value(
                       problem.inequalities[static_cast<std::size_t>(i)],
                       trial)<=strict_side_limit)
                  continue;
                if(working_set.count(i)>0) modeled_offender=true;
                else omitted_offender=true;
              }
              modeled_side_filter_candidate=modeled_offender &&
                  !omitted_offender && tf.inequality_inf<=std::max(
                      strict_side_limit,
                      st_.dca_condensed_modeled_side_filter_limit);
            }

            // The affine endpoint was checked against every omitted
            // inequality before entering globalization.  A condensed normal
            // correction can nevertheless move that endpoint across a side
            // constraint (the case118 flow-limit pattern).  Identify only
            // violations created by that correction: an offender must be
            // feasible at the raw tangent trial and outside the unchanged
            // final feasibility tube at the retracted point.
            if(condensed_equality_mode && retraction_used_here &&
               bound_not_worsened && !strict_side_feasible) {
              const double side_limit=strict_side_limit;
              Vec raw_trial=x;
              axpy(1.0,scaled,raw_trial);
              std::vector<int> omitted_post_retraction_offenders;
              int modeled_post_retraction_offenders=0;
              bool every_offender_created_by_retraction=true;
              for(int i=0;i<(int)problem.inequalities.size();++i) {
                const double corrected_value=structural_quadratic_value(
                    problem.inequalities[i],trial);
                if(!(corrected_value>side_limit)) continue;
                const double raw_value=structural_quadratic_value(
                    problem.inequalities[i],raw_trial);
                if(raw_value>side_limit)
                  every_offender_created_by_retraction=false;
                if(working_set.count(i)) {
                  ++modeled_post_retraction_offenders;
                } else {
                  omitted_post_retraction_offenders.push_back(i);
                }
              }
              result.lazy_dca_condensed_post_retraction_modeled_offenders+=
                  modeled_post_retraction_offenders;
              result.lazy_dca_condensed_post_retraction_omitted_offenders+=
                  static_cast<int>(omitted_post_retraction_offenders.size());

              // A filter method need not force every intermediate iterate
              // through the *final* feasibility tube.  Admit only the
              // classical condensed Maratos pattern: all exact offenders were
              // created by the equality retraction, every one is already in
              // the solved convex majorant, no bound is relaxed, and the
              // violation stays below an explicit cap.  The exact objective
              // and Fletcher--Leyffer filter tests below still decide whether
              // the trial is accepted.  Final KKT termination continues to
              // use the strict side_limit.
              const bool modeled_only=
                  modeled_post_retraction_offenders>0 &&
                  omitted_post_retraction_offenders.empty() &&
                  (every_offender_created_by_retraction ||
                   outer_started_in_side_recovery) &&
                  tf.bound_inf<=side_limit;
              if(st_.dca_condensed_modeled_side_filter && modeled_only &&
                 !direct_start_accepted) {
                ++result.lazy_dca_condensed_modeled_side_filter_attempts;
                const double configured_side_limit=std::max(
                    side_limit,
                    std::min(st_.dca_condensed_modeled_side_filter_limit,
                             std::max(side_limit,st_.working_set_activation)));
                // Once a relaxed trial has been accepted, every subsequent
                // relaxed step must recover a fixed fraction of its current
                // side violation.  This prevents a low-objective sequence
                // from circulating inside the transient tube.  A strict
                // candidate is, as always, accepted by the historical path.
                const double recovery_limit=
                    fcur.inequality_inf<=side_limit
                        ? configured_side_limit
                        : side_limit+0.80*std::max(
                              0.0,fcur.inequality_inf-side_limit);
                modeled_side_filter_candidate=
                    tf.inequality_inf<=configured_side_limit &&
                    tf.inequality_inf<=recovery_limit;
                if(!modeled_side_filter_candidate)
                  ++result.lazy_dca_condensed_modeled_side_filter_fallbacks;
              }

              if(!extrapolation_schedule &&
                 !retraction_fallback_schedule &&
                 st_.dca_condensed_post_retraction_working_set_expansion &&
                 !omitted_post_retraction_offenders.empty()) {
                if(expansions<st_.max_working_set_expansions &&
                   post_retraction_expansions<
                       st_.dca_condensed_post_retraction_working_set_max_expansions) {
                  // Preserve the solved restricted majorant exactly as for an
                  // ordinary full-endpoint expansion.  The enlarged model is
                  // re-solved at the same x_k; no rejected retracted point is
                  // ever accepted or used as a primal seed.
                  working_set_primal_seed=inner.y;
                  working_set_multiplier_seed=multipliers_from_inner(
                      problem,accepted_built,inner,st_);
                  working_set_transport_valid=
                      static_cast<int>(working_set_primal_seed.size())==
                          accepted_built.p.nvar &&
                      std::all_of(working_set_primal_seed.begin(),
                                  working_set_primal_seed.end(),
                                  [](double v){ return std::isfinite(v); }) &&
                      working_set_multiplier_seed.valid;
                  for(int i:omitted_post_retraction_offenders) {
                    if(!working_set.insert(i).second) continue;
                    working_set_drop_streak[static_cast<std::size_t>(i)]=0;
                    curvature.ensure_convex(problem.inequalities[i].hessian,
                                            ineq_cvx[i],0.0,
                                            st_.max_rank_per_quadratic);
                    ++result.working_set_additions;
                  }
                  ++expansions;
                  ++post_retraction_expansions;
                  ++result.lazy_dca_condensed_post_retraction_working_set_expansions;
                  ++result.lazy_dca_condensed_outer_reject_side;
                  if(st_.verbose)
                    std::cout << "[CONDENSED-POST-RETRACTION-EXPAND] k="
                              << outer
                              << " alpha=" << std::scientific << alpha
                              << " added="
                              << omitted_post_retraction_offenders.size()
                              << " modeled="
                              << modeled_post_retraction_offenders
                              << " expansion=" << post_retraction_expansions
                              << "/"
                              << st_.dca_condensed_post_retraction_working_set_max_expansions
                              << " total_expansion=" << expansions << "/"
                              << st_.max_working_set_expansions
                              << " ws=" << working_set.size()
                              << " action=resolve-same-outer\n"
                              << std::defaultfloat;
                  goto retry_majorant_after_condensed_side_expansion;
                }
                // The expansion cap is a robustness guard, not a convergence
                // shortcut.  Once exhausted, execute the exact historical
                // alpha contraction below.
                ++result.lazy_dca_condensed_post_retraction_expansion_fallbacks;
                if(st_.verbose)
                  std::cout << "[CONDENSED-POST-RETRACTION-EXPAND] k="
                            << outer
                            << " alpha=" << std::scientific << alpha
                            << " omitted="
                            << omitted_post_retraction_offenders.size()
                            << " expansion=" << post_retraction_expansions
                            << "/"
                            << st_.dca_condensed_post_retraction_working_set_max_expansions
                            << " total_expansion=" << expansions << "/"
                            << st_.max_working_set_expansions
                            << " action=historical-backtrack\n"
                            << std::defaultfloat;
              }
            }
            const bool side_acceptable=
                strict_side_feasible || modeled_side_filter_candidate;
            const bool nondominated=filter_acceptable(filter,tf.total_inf,ft,st_);
            const bool objective_ok=(trial_predicted_decrease>predicted_floor)
                ? (condensed_equality_mode
                    ? ft<=objcur-st_.line_search_armijo*trial_predicted_decrease+
                        128.0*std::numeric_limits<double>::epsilon()*
                            std::max(1.0,std::abs(objcur))
                    : armijo_objective_ok(objcur,ft,alpha,
                                          trial_predicted_decrease,st_))
                : ft<=objcur+monotone_tol;
            const double equality_target=std::max(final_feasibility_tube_limit(st_),
                (1.0-st_.dca_feasibility_step_fraction*alpha)*
                    fcur.equality_inf);
            const bool feasibility_step=strict_side_feasible &&
                tf.equality_inf<fcur.equality_inf &&
                tf.equality_inf<=equality_target;
            const bool objective_step=side_acceptable && within_funnel && objective_ok;
            const bool side_recovery_step=modeled_side_filter_candidate &&
                fcur.inequality_inf>strict_side_limit && within_funnel &&
                tf.inequality_inf<fcur.inequality_inf;

            // Extrapolation intentionally steps beyond the minimizer of the
            // convex majorant, so that majorant cannot predict its gain.  The
            // probe is instead admitted only on a strict decrease of the
            // ORIGINAL objective at an ORIGINAL-QCQP feasible endpoint.  The
            // historical funnel, filter, bound, side and objective tests stay
            // active as additional gates.
            const double extrapolation_objective_noise=128.0*
                std::numeric_limits<double>::epsilon()*
                std::max(1.0,std::abs(objcur));
            const bool extrapolation_gate=!extrapolation_schedule ||
                (ft<objcur-extrapolation_objective_noise &&
                 within_final_feasibility_tube(tf.total_inf,st_) &&
                 strict_side_feasible && within_funnel && objective_ok &&
                 objective_step);
            if(extrapolation_gate && nondominated &&
               (objective_step || feasibility_step || side_recovery_step)) {
              accepted=true;
              accepted_on_seeded_schedule=predictor_seeded && schedule==0;
              accepted_on_outer_extrapolation=extrapolation_schedule;
              accepted_objective_step=objective_step;
              accepted_feasibility_step=!objective_step &&
                  (feasibility_step || side_recovery_step);
              accepted_trial=std::move(trial);
              accepted_feas=tf;
              accepted_obj=ft;
              accepted_alpha=alpha;
              accepted_soc=soc_used_here;
              accepted_retraction=retraction_used_here;
              accepted_single_loop_fallback_retraction=
                  retraction_fallback_schedule && retraction_used_here;
              if(modeled_side_filter_candidate)
                ++result.lazy_dca_condensed_modeled_side_filter_acceptances;
              if(modeled_side_filter_candidate) {
                result.lazy_dca_condensed_modeled_side_filter_max_violation=
                    std::max(
                        result.lazy_dca_condensed_modeled_side_filter_max_violation,
                        tf.inequality_inf);
                modeled_side_recovery_rows.clear();
                for(int idx:working_set)
                  if(structural_quadratic_value(
                         problem.inequalities[static_cast<std::size_t>(idx)],
                         accepted_trial)>strict_side_limit)
                    modeled_side_recovery_rows.insert(idx);
              } else if(strict_side_feasible) {
                modeled_side_recovery_rows.clear();
              }
              if(side_recovery_step)
                ++result.lazy_dca_condensed_modeled_side_filter_recovery_acceptances;
              if(soc_used_here) ++result.soc_accepted_steps;
              break;
            }
            if(soc_attempted_here) ++result.soc_rejected_steps;
            if(within_funnel && nondominated && !objective_ok) ++result.armijo_rejections;
            if(condensed_equality_mode) {
              if(!bound_not_worsened)
                ++result.lazy_dca_condensed_outer_reject_bound;
              if(bound_not_worsened && !side_acceptable)
                ++result.lazy_dca_condensed_outer_reject_side;
              if(!within_funnel)
                ++result.lazy_dca_condensed_outer_reject_funnel;
              if(!nondominated)
                ++result.lazy_dca_condensed_outer_reject_filter;
              if(!objective_ok)
                ++result.lazy_dca_condensed_outer_reject_objective;
            }
            if(extrapolation_schedule) break;
            ++result.line_search_backtracks;
            alpha*=st_.line_search_contraction;
            if(alpha<st_.minimum_step_length) break;
          }
          if(extrapolation_schedule && !accepted) {
            ++result.lazy_dca_condensed_outer_extrapolation_historical_fallbacks;
            if(st_.verbose)
              std::cout << "[CONDENSED-OUTER-EXTRAPOLATION] k="<<outer
                        << " rejected-alpha="<<std::scientific<<alpha
                        << " action=historical-alpha-one\n"
                        << std::defaultfloat;
          }
          }
          if(accepted && accepted_on_seeded_schedule)
            result.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped+=
                predictor_leading_levels;
          if(accepted && condensed_equality_mode &&
             (st_.dca_condensed_alpha_predictor ||
              st_.dca_condensed_outer_extrapolation)) {
            if(outer_started_in_side_recovery ||
               !modeled_side_recovery_rows.empty()) {
              // Entering or leaving the recovery model changes at least one
              // majorant constant, so its alpha history is not transferable.
              condensed_alpha_history_valid=false;
              condensed_iterations_since_full_probe=0;
              condensed_alpha_one_streak=0;
            } else {
              condensed_alpha_history_valid=true;
              condensed_last_accepted_alpha=accepted_alpha;
              condensed_last_direction_inf=raw_direction_inf;
              condensed_last_model_prox=dca_model_prox;
              condensed_last_bound_face_coordinates=live_bound_face_coordinates;
              condensed_last_working_set=current_working_set;
              if(!predictor_seeded || !accepted_on_seeded_schedule)
                condensed_iterations_since_full_probe=0;
              else
                ++condensed_iterations_since_full_probe;
            }
          }
          if(accepted && condensed_equality_mode &&
             st_.dca_condensed_outer_extrapolation) {
            const double alpha_one_noise=64.0*
                std::numeric_limits<double>::epsilon();
            if(outer_started_in_side_recovery ||
               !modeled_side_recovery_rows.empty()) {
              condensed_alpha_one_streak=0;
            } else if(accepted_on_outer_extrapolation) {
              ++result.lazy_dca_condensed_outer_extrapolation_acceptances;
              result.lazy_dca_condensed_outer_extrapolation_alpha_max=std::max(
                  result.lazy_dca_condensed_outer_extrapolation_alpha_max,
                  accepted_alpha);
              condensed_alpha_one_streak=0;
            } else if(std::abs(accepted_alpha-1.0)<=alpha_one_noise) {
              // A failed probe falls through to the exact alpha=1 schedule.
              // Count that accepted fallback as the first member of a fresh
              // streak, preventing an immediate probe on the next outer
              // iteration. A changed working-set/bound-face/direction
              // structure likewise starts a new streak at one; the configured
              // minimum can therefore never be inherited across a model
              // discontinuity.
              condensed_alpha_one_streak=
                  (!extrapolation_structure_same ||
                   extrapolation_attempted_this_outer)
                      ? 1 : condensed_alpha_one_streak+1;
              result.lazy_dca_condensed_outer_extrapolation_full_step_streak_max=
                  std::max(
                      result.lazy_dca_condensed_outer_extrapolation_full_step_streak_max,
                      condensed_alpha_one_streak);
            } else {
              condensed_alpha_one_streak=0;
            }
          }

          if(!accepted) {
            if(single_loop_primary_route &&
               !force_strict_inner_this_outer &&
               schedule_single_loop_model_retry(
                   inner,accepted_built,"original-globalization")) {
              goto retry_majorant_after_condensed_side_expansion;
            }
            if(!st_.dca_single_loop_primal_dual &&
               inner.lazyqp_inexact_forcing_accepted &&
               !inner.dca_single_loop_accepted &&
               !force_strict_inner_this_outer) {
              force_strict_inner_this_outer=true;
              ++result.lazyqp_inexact_forcing_outer_retries;
              working_set_transport_valid=false;
              continuation_valid=false;
              continuation_equality_duals.clear();
              continuation_inequality_duals.clear();
              continuation_inequality_slacks.clear();
              continuation_inequality_primal_residuals.clear();
              continuation_inequality_keys.clear();
              if(st_.verbose)
                std::cout << "[LAZY-DCA-FORCING-FALLBACK] k="<<outer
                          << " reason=original-globalization"
                          << " action=strict-resolve\n";
              goto retry_majorant_after_condensed_side_expansion;
            }
            // Equality-only problems have a fixed active manifold.  If the
            // globalized DCA step has exhausted its line search near a
            // feasible stationary point, do not report a globalization
            // failure before trying the guarded primal-dual Newton system.
            // This is deliberately a last-resort takeover: all ordinary DCA
            // decisions and the final original-QCQP certificate are kept.
            bool terminal_resumed=false;
            if(!st_.dca_single_loop_primal_dual &&
               st_.terminal_kkt_enabled && problem.inequalities.empty() &&
               warm_mult_valid &&
               fcur.total_inf<=st_.feasibility_tolerance) {
              const double rescue_stationarity=
                  cached_projected_stationarity(x);
              const double rescue_gate=std::max(
                  1.0,100.0*st_.terminal_kkt_trigger_stationarity);
              if(rescue_stationarity<=rescue_gate) {
                const double f_before=
                    structural_quadratic_value(problem.objective,x);
                const Vec rescue_origin=x;
                const auto tk=kkt_polish_qpdual(problem,x,warm_mult,st_);
                result.terminal_kkt_attempts += tk.attempted?1:0;
                result.terminal_kkt_iterations += tk.iterations;
                const double f_after=problem.objective.value(tk.x);
                const double f_noise=128.0*
                    std::numeric_limits<double>::epsilon()*
                    std::max(1.0,std::abs(f_before));
                const bool commit=(tk.improved||tk.success) &&
                    (tk.success||f_after<=f_before+f_noise);
                if(commit) {
                  x=tk.x;
                  // The terminal corrector changed the primal point outside
                  // the condensed DCA step whose alpha was memorized.  Never
                  // seed a later line search from stale geometry.
                  condensed_alpha_history_valid=false;
                  condensed_alpha_one_streak=0;
                  warm_mult=tk.multipliers;
                  invalidate_mapped_stationarity_cache();
                  warm_mult_valid=warm_mult.valid;
                  warm_mult_model_only=!tk.success;
                  result.terminal_kkt_accepted_steps+=tk.accepted_steps;
                  result.outer_iterations=outer+1;
                  continuation_valid=false;
                  continuation_equality_duals.clear();
                  continuation_inequality_duals.clear();
                  continuation_inequality_slacks.clear();
                  continuation_inequality_primal_residuals.clear();
                  continuation_inequality_keys.clear();
                  if(st_.verbose)
                    std::cout << "[LAZY-DCA-KKT-RESCUE] k="<<outer
                              << " accepted="<<tk.accepted_steps
                              << " stationarity="<<std::scientific
                              << tk.stationarity
                              << " objective_delta="<<f_after-f_before
                              << "\n"<<std::defaultfloat;
                  Vec rescue_step=x;
                  axpy(-1.0,rescue_origin,rescue_step);
                  print_live_row(outer+1,x,structural_feasibility(problem,x),
                      tk.stationarity,
                      inf_norm(rescue_step),1.0,active_objective_cvx->rank(),
                      result.line_search_backtracks-live_ls_before,"N",true);
                  if(tk.success && warm_mult_valid &&
                     mapped_certificate_passes(
                         problem,x,warm_mult,st_,result)) {
                    result.status=Status::Solved;
                    result.message=
                        "guarded equality-manifold Newton rescued the exhausted Lazy-DCA line search";
                    break;
                  }
                  terminal_resumed=true;
                } else if(tk.attempted) {
                  ++result.terminal_kkt_rejected_steps;
                }
              }
            }
            if(terminal_resumed) continue;
            result.status=Status::LineSearchFailure;
            result.message=single_loop_primary_route
                ? "single-loop original globalization failed after bounded proximal retries"
                : condensed_equality_mode
                ? "Lazy-DCA line search with condensed quadratic equality retraction failed"
                : "Lazy-DCA line search on linearized quadratic equalities failed";
            break;
          }
          filter_insert(filter,fcur.total_inf,objcur);
          if(condensed_equality_mode)
            ++result.lazy_dca_condensed_accepted_steps;
          else
            ++result.lazy_dca_linearized_equality_steps;
          if(accepted_objective_step) ++result.lazy_dca_equality_objective_steps;
          if(accepted_feasibility_step) ++result.lazy_dca_equality_feasibility_steps;
          result.lazy_dca_last_equality_step_alpha=accepted_alpha;
          candidate=std::move(accepted_trial);
          fcand=accepted_feas;
          objcand=accepted_obj;
          // Objective steps are already globalized by the exact objective,
          // side-feasibility, filter and current funnel tests. Tightening the
          // funnel after every such step manufactures a sequence of small
          // steps although no h-step was accepted. Only a genuine accepted
          // feasibility step earns a funnel contraction.
          if(accepted_feasibility_step) {
            const double contracted=
                st_.dca_equality_funnel_contraction*equality_funnel;
            const double trial_guard=st_.dca_equality_funnel_growth*
                fcand.equality_inf;
            equality_funnel=std::max(st_.feasibility_tolerance,
                std::min(equality_funnel,std::max(contracted,trial_guard)));
          }
          result.lazy_dca_final_equality_funnel=equality_funnel;
        } else if(within_final_feasibility_tube(fcur.total_inf,st_) &&
                  within_final_feasibility_tube(fcand.total_inf,st_) &&
                  objcand>objcur+monotone_tol) {
          if(single_loop_primary_route &&
             !force_strict_inner_this_outer &&
             schedule_single_loop_model_retry(
                 inner,accepted_built,"majorization-invariant")) {
            goto retry_majorant_after_condensed_side_expansion;
          }
          if(!st_.dca_single_loop_primal_dual &&
             inner.lazyqp_inexact_forcing_accepted &&
             !inner.dca_single_loop_accepted &&
             !force_strict_inner_this_outer) {
            force_strict_inner_this_outer=true;
            ++result.lazyqp_inexact_forcing_outer_retries;
            working_set_transport_valid=false;
            continuation_valid=false;
            continuation_equality_duals.clear();
            continuation_inequality_duals.clear();
            continuation_inequality_slacks.clear();
            continuation_inequality_primal_residuals.clear();
            continuation_inequality_keys.clear();
            if(st_.verbose)
              std::cout << "[LAZY-DCA-FORCING-FALLBACK] k="<<outer
                        << " reason=majorization-invariant"
                        << " action=strict-resolve\n";
            goto retry_majorant_after_condensed_side_expansion;
          }
          // With affine equalities, the exact convex majorant step is an MM/DCA
          // step and should not increase the original objective.
          result.status=Status::NumericalError;
          result.message="Lazy-DCA majorization invariant violated: feasible convex-subproblem solution increases original objective";
          break;
        }

        // Once a direct-start trajectory has first reached the strict side
        // tube, it may never leave it.  The nonlinear-equality branches above
        // already impose this check; keep the invariant explicit here as a
        // final fail-closed guard for affine-equality/no-equality routes too.
        if(direct_start_accepted && direct_start_side_strict_latched &&
           inequality_bound_violation(fcand)>
               final_feasibility_tube_limit(st_)) {
          result.status=Status::LineSearchFailure;
          result.message=
              "single-loop direct-start strict side latch rejected an infeasible trial";
          break;
        }

        // Reaching this point means that the uncorrected provisional endpoint
        // passed every original-QCQP globalization and majorization guard.
        // Count the omitted retraction only now; rejected private trials are
        // accounted as strict transactional fallbacks above.
        if(has_quadratic_equality && skip_single_loop_auxiliaries &&
           !accepted_single_loop_fallback_retraction)
          ++result.dca_single_loop_retractions_skipped;

        Vec step=candidate; axpy(-1.0,x,step);
        double step_inf=inf_norm(step);
        result.step_inf=step_inf;
        result.lazy_dca_last_raw_direction_inf=accepted_raw_direction_inf;

        const bool complete_inner_state=
            (int)inner.y.size()==accepted_built.p.nvar &&
            (int)inner.equality_duals.size()==(int)accepted_built.p.equalities.rows.size() &&
            (int)inner.inequality_duals.size()==(int)accepted_built.p.inequalities.size() &&
            (int)inner.inequality_slacks.size()==(int)accepted_built.p.inequalities.size() &&
            finite_state(inner.y) && finite_state(inner.equality_duals) &&
            finite_state(inner.inequality_duals) &&
            finite_state(inner.inequality_slacks) &&
            std::all_of(inner.inequality_duals.begin(),
                        inner.inequality_duals.end(),
                        [](double value){ return value>0.0; }) &&
            std::all_of(inner.inequality_slacks.begin(),
                        inner.inequality_slacks.end(),
                        [](double value){ return value>0.0; });
        const bool complete_initial_inner_state=
            (int)inner.dca_single_loop_initial_primal.size()==
                accepted_built.p.nvar &&
            (int)inner.dca_single_loop_initial_equality_duals.size()==
                (int)accepted_built.p.equalities.rows.size() &&
            (int)inner.dca_single_loop_initial_inequality_duals.size()==
                (int)accepted_built.p.inequalities.size() &&
            (int)inner.dca_single_loop_initial_inequality_slacks.size()==
                (int)accepted_built.p.inequalities.size() &&
            finite_state(inner.dca_single_loop_initial_primal) &&
            finite_state(inner.dca_single_loop_initial_equality_duals) &&
            finite_state(inner.dca_single_loop_initial_inequality_duals) &&
            finite_state(inner.dca_single_loop_initial_inequality_slacks) &&
            std::all_of(
                inner.dca_single_loop_initial_inequality_duals.begin(),
                inner.dca_single_loop_initial_inequality_duals.end(),
                [](double value){ return value>0.0; }) &&
            std::all_of(
                inner.dca_single_loop_initial_inequality_slacks.begin(),
                inner.dca_single_loop_initial_inequality_slacks.end(),
                [](double value){ return value>0.0; });
        const bool exact_endpoint=accepted_alpha==1.0 &&
            !accepted_soc && !accepted_retraction;
        const bool alpha_interpolation_valid=
            accepted_alpha>=0.0 && accepted_alpha<=1.0 &&
            !accepted_soc &&
            (!accepted_retraction ||
             accepted_single_loop_fallback_retraction);
        bool single_loop_continuation=
            single_loop_primary_route && complete_inner_state &&
            complete_initial_inner_state && alpha_interpolation_valid;
        const bool historical_continuation=
            !single_loop_primary_route && st_.inner_dual_warm_start &&
            complete_inner_state && exact_endpoint;

        // Materialize the primal-dual state at the endpoint actually accepted
        // by the outer line search.  This is especially important after
        // alpha<1: mapping the full-step duals would mix states from different
        // points.  Positivity is preserved by convex interpolation, and the
        // next majorant independently remaps the transported primal-slack
        // residual.
        InnerResult endpoint_inner=inner;
        if(single_loop_continuation) {
          const double one_minus_alpha=1.0-accepted_alpha;
          endpoint_inner.y.resize(inner.y.size());
          for(std::size_t i=0;i<endpoint_inner.y.size();++i)
            endpoint_inner.y[i]=one_minus_alpha*
                inner.dca_single_loop_initial_primal[i]+
                accepted_alpha*inner.y[i];
          endpoint_inner.equality_duals.resize(inner.equality_duals.size());
          for(std::size_t i=0;i<endpoint_inner.equality_duals.size();++i)
            endpoint_inner.equality_duals[i]=one_minus_alpha*
                inner.dca_single_loop_initial_equality_duals[i]+
                accepted_alpha*inner.equality_duals[i];
          endpoint_inner.inequality_duals.resize(inner.inequality_duals.size());
          endpoint_inner.inequality_slacks.resize(inner.inequality_slacks.size());
          for(std::size_t i=0;i<endpoint_inner.inequality_duals.size();++i) {
            endpoint_inner.inequality_duals[i]=one_minus_alpha*
                inner.dca_single_loop_initial_inequality_duals[i]+
                accepted_alpha*inner.inequality_duals[i];
            endpoint_inner.inequality_slacks[i]=one_minus_alpha*
                inner.dca_single_loop_initial_inequality_slacks[i]+
                accepted_alpha*inner.inequality_slacks[i];
          }
          single_loop_continuation=finite_state(endpoint_inner.y) &&
              finite_state(endpoint_inner.equality_duals) &&
              finite_state(endpoint_inner.inequality_duals) &&
              finite_state(endpoint_inner.inequality_slacks) &&
              std::all_of(endpoint_inner.inequality_duals.begin(),
                          endpoint_inner.inequality_duals.end(),
                          [](double value){ return value>0.0; }) &&
              std::all_of(endpoint_inner.inequality_slacks.begin(),
                          endpoint_inner.inequality_slacks.end(),
                          [](double value){ return value>0.0; });
        }

        bool endpoint_continuation_valid=
            single_loop_continuation || historical_continuation;
        Vec endpoint_continuation_equality_duals;
        Vec endpoint_continuation_inequality_duals;
        Vec endpoint_continuation_inequality_slacks;
        Vec endpoint_continuation_inequality_primal_residuals;
        std::vector<std::uint64_t> endpoint_continuation_inequality_keys;
        if(single_loop_continuation) {
          endpoint_continuation_equality_duals=endpoint_inner.equality_duals;
          endpoint_continuation_inequality_duals=endpoint_inner.inequality_duals;
          endpoint_continuation_inequality_slacks=endpoint_inner.inequality_slacks;
          endpoint_continuation_inequality_keys=
              accepted_built.inequality_continuation_keys;
        } else if(historical_continuation) {
          endpoint_continuation_equality_duals=inner.equality_duals;
          endpoint_continuation_inequality_duals=inner.inequality_duals;
          endpoint_continuation_inequality_slacks=inner.inequality_slacks;
          endpoint_continuation_inequality_keys=
              accepted_built.inequality_continuation_keys;
        }
        if(endpoint_continuation_valid) {
          const Vec& endpoint_y=single_loop_continuation
              ? endpoint_inner.y : inner.y;
          const int mi=static_cast<int>(accepted_built.p.inequalities.size());
          endpoint_continuation_valid=
              static_cast<int>(endpoint_y.size())==accepted_built.p.nvar &&
              static_cast<int>(
                  endpoint_continuation_inequality_slacks.size())==mi &&
              static_cast<int>(
                  endpoint_continuation_inequality_keys.size())==mi;
          if(endpoint_continuation_valid) {
            endpoint_continuation_inequality_primal_residuals.resize(
                static_cast<std::size_t>(mi),0.0);
            for(int i=0;i<mi;++i) {
              const double phi=accepted_built.p.inequalities[
                  static_cast<std::size_t>(i)].value(endpoint_y);
              const double residual=phi+
                  endpoint_continuation_inequality_slacks[
                      static_cast<std::size_t>(i)];
              if(!std::isfinite(phi) || !std::isfinite(residual)) {
                endpoint_continuation_valid=false;
                endpoint_continuation_inequality_primal_residuals.clear();
                break;
              }
              endpoint_continuation_inequality_primal_residuals[
                  static_cast<std::size_t>(i)]=residual;
            }
          }
        }

        const bool single_loop_endpoint_matches=
            single_loop_primary_route && single_loop_continuation;
        const InnerResult& multiplier_source=single_loop_endpoint_matches
            ? endpoint_inner : inner;
        const MultiplierEstimate inner_mult=multipliers_from_inner(
            problem,accepted_built,multiplier_source,st_);
        // Historical strict paths retain their alpha blend and exact refit.
        // A marked provisional path instead maps the interpolated endpoint
        // state above, then commits it only through the original-QCQP guard.
        const MultiplierEstimate proposed_mult=
            single_loop_endpoint_matches &&
                    st_.dca_single_loop_multiplier_transport
                ? inner_mult
                : (has_quadratic_equality && warm_mult_valid
                    ? blend_multipliers(warm_mult,inner_mult,accepted_alpha)
                    : inner_mult);
        bool primary_refit=false;
        const bool transport_requested=has_quadratic_equality &&
            single_loop_primary_route &&
            st_.dca_single_loop_multiplier_transport;
        const bool transported=has_quadratic_equality &&
            commit_guarded_single_loop_transport(
                candidate,proposed_mult,single_loop_primary_route,
                single_loop_endpoint_matches,
                outer,"accepted-primary");
        if(transport_requested && !transported) {
          // Never turn a rejected transport guard into an intermediate full
          // multiplier fit.  Globalization bookkeeping has already selected
          // this endpoint, so terminate cleanly instead of attempting a fragile
          // partial rollback of filter/funnel/history state.
          result.status=Status::InnerFailure;
          result.message=
              "single-loop multiplier transport guard rejected the accepted "
              "primal endpoint";
          break;
        }
        if(!transported) {
          // The default-off path retains its historical full refit.  Under
          // the single-loop master switch, the mapped inner state remains
          // explicitly model-only and is never promoted to a certificate.
          warm_mult=proposed_mult;
          invalidate_mapped_stationarity_cache();
          warm_mult_valid=warm_mult.valid;
          if(has_quadratic_equality) warm_mult_model_only=true;
          if(has_quadratic_equality &&
             !st_.dca_single_loop_primal_dual)
            primary_refit=refit_path_multipliers(
                candidate,outer,"accepted-primary");
        }
        if(!primary_refit && warm_mult_valid)
          result.model_stationarity_inf=
              cached_projected_stationarity(candidate);

        // Publish direct continuation only after the associated multiplier
        // transaction has either passed its original-QCQP guard or explicitly
        // selected the historical (transport-disabled/bootstrap) policy.
        continuation_valid=endpoint_continuation_valid;
        if(continuation_valid) {
          continuation_equality_duals=
              std::move(endpoint_continuation_equality_duals);
          continuation_inequality_duals=
              std::move(endpoint_continuation_inequality_duals);
          continuation_inequality_slacks=
              std::move(endpoint_continuation_inequality_slacks);
          continuation_inequality_primal_residuals=
              std::move(endpoint_continuation_inequality_primal_residuals);
          continuation_inequality_keys=
              std::move(endpoint_continuation_inequality_keys);
        } else {
          continuation_equality_duals.clear();
          continuation_inequality_duals.clear();
          continuation_inequality_slacks.clear();
          continuation_inequality_primal_residuals.clear();
          continuation_inequality_keys.clear();
        }
        x=candidate;

        // The exact DCA majorant above is always the primary step.  This
        // optional multiplier-aware corrector is local to the SAME Lazy-DCA
        // iteration: it linearizes the original constraints, uses a convexified
        // Lagrangian metric, and is accepted only by the original-QCQP
        // globalization.  It never invokes the historical SQP outer loop.
        const bool corrector_eligible=has_quadratic_equality || problem.n<=64;
        const bool corrector_due=!skip_single_loop_auxiliaries &&
            st_.dca_local_corrector && !elastic_equality_mode &&
            !condensed_equality_mode && corrector_eligible &&
            outer>=st_.dca_local_corrector_start_outer &&
            st_.dca_local_corrector_frequency>0 &&
            ((outer-st_.dca_local_corrector_start_outer)%
                 st_.dca_local_corrector_frequency==0);
        if(corrector_due) {
          ++result.lazy_dca_corrector_attempts;
          MultiplierEstimate corr_mult;
          if(has_quadratic_equality) {
            if(!warm_mult_valid &&
               !st_.dca_single_loop_primal_dual)
              refit_path_multipliers(x,outer,"pre-corrector");
            corr_mult=warm_mult;
          } else {
            // Preserve the validated inequality-only corrector trajectory.
            corr_mult=estimate_multipliers_legacy(
                problem,x,working_set,st_);
            if(!corr_mult.valid && warm_mult_valid) corr_mult=warm_mult;
          }
          bool corrector_accepted=false;
          bool corrector_accepted_feasibility_step=false;
          if(corr_mult.valid) {
            const SymMatrix Bc=lagrangian_hessian(problem,corr_mult);
            Convexification corr_cvx;
            curvature.ensure_convex(Bc,corr_cvx,0.0,st_.max_lagrangian_rank);
            auto cb=build_subproblem(problem,x,Bc,corr_cvx,ineq_cvx,working_set,st_,
                                     std::max(st_.proximal_regularization,dca_model_prox),
                                     true);
            cb.p.prefer_feasible_centered_start=true;
            seed_inner_dual_warm_start(cb,problem,corr_mult,true,st_);
            persistent_lazyqp_settings=inner_settings();
            auto solve_corrector = [&]() {
              if(uses_lazyqp_kernel && !cb.p.inequalities.empty())
                return st_.inner_backend=="socp"
                    ? persistent_socp.solve(cb.p)
                    : persistent_lazyqp.solve(cb.p);
              return solve_inner(cb.p,persistent_lazyqp_settings,
                                 persistent_equality_factor.get());
            };
            InnerResult ci=solve_corrector();
            record_inner(ci);
            result.lazy_dca_corrector_inner_iterations+=ci.iterations;
            if((!ci.success || (int)ci.y.size()<problem.n) &&
               cb.p.has_dual_warm_start) {
              cb.p.has_dual_warm_start=false;
              cb.p.initial_equality_duals.clear();
              cb.p.initial_inequality_duals.clear();
              ++result.lazy_dca_corrector_cold_retries;
              if(st_.verbose)
                std::cout << "[LAZY-DCA-CORRECTOR-RETRY] k="<<outer
                          << " mode=cold message="<<ci.message<<"\n";
              ci=solve_corrector();
              record_inner(ci);
              result.lazy_dca_corrector_inner_iterations+=ci.iterations;
            }

            if(ci.success && (int)ci.y.size()>=problem.n) {
              Vec cd(ci.y.begin(),ci.y.begin()+problem.n);
              const double predicted_decrease=std::max(0.0,
                  -dot(problem.objective.gradient(x),cd));
              const double predicted_floor=100.0*std::numeric_limits<double>::epsilon()*
                  std::max(1.0,std::abs(
                      structural_quadratic_value(problem.objective,x)));
              const Feasibility corr_fcur=structural_feasibility(problem,x);
              const double corr_fbase=
                  structural_quadratic_value(problem.objective,x);
              double alpha=1.0;
              for(int ls=0;ls<st_.line_search_max_steps;++ls) {
                Vec trial=x;
                Vec scaled=cd; for(double&v:scaled) v*=alpha;
                axpy(1.0,scaled,trial);
                if(!finite_state(trial)) {
                  alpha*=st_.line_search_contraction;
                  continue;
                }
                Feasibility tf=structural_feasibility(problem,trial);
                bool retraction_used=false;
                if(has_quadratic_equality && st_.dca_equality_retraction) {
                  ++result.lazy_dca_equality_retraction_attempts;
                  const auto retract=try_dca_equality_retraction(
                      problem,x,trial,scaled,equality_funnel,st_);
                  result.lazy_dca_max_equality_retraction_inf=std::max(
                      result.lazy_dca_max_equality_retraction_inf,
                      retract.correction_inf);
                  if(retract.success) {
                    trial=retract.x;
                    tf=retract.feasibility;
                    retraction_used=true;
                    ++result.lazy_dca_equality_retraction_acceptances;
                    if(st_.verbose)
                      std::cout << "[LAZY-DCA-RETRACTION] k="<<outer
                                << " phase=corrector alpha="<<std::scientific<<alpha
                                << " accepted=1 corr="<<retract.correction_inf
                                << " eq="<<tf.equality_inf
                                << " side="<<inequality_bound_violation(tf)
                                << " funnel="<<equality_funnel
                                << "\n"<<std::defaultfloat;
                  } else {
                    ++result.lazy_dca_equality_retraction_rejections;
                  }
                }
                bool soc_used=false;
                const double soc_trigger_ratio=st_.dca_iterated_soc
                    ? std::max(st_.second_order_correction_trigger_ratio,
                               st_.dca_iterated_soc_trigger_ratio)
                    : st_.second_order_correction_trigger_ratio;
                const double soc_trigger=soc_trigger_ratio*
                    std::max(st_.feasibility_tolerance,corr_fcur.total_inf);
                if(has_quadratic_equality && !retraction_used &&
                   st_.second_order_correction &&
                   tf.equality_inf>st_.feasibility_tolerance &&
                   tf.total_inf<=soc_trigger) {
                  ++result.soc_attempts;
                  auto soc=second_order_correction(problem,x,trial,working_set,st_,
                                                      false,st_.dca_iterated_soc,true);
                  if(soc.iterated_attempted) {
                    ++result.lazy_dca_iterated_soc_attempts;
                    if(soc.iterated_success) ++result.lazy_dca_iterated_soc_acceptances;
                    else ++result.lazy_dca_iterated_soc_rejections;
                    if(st_.verbose)
                      std::cout << "[LAZY-DCA-ITERATED-SOC] k="<<outer
                                << " phase=corrector alpha="<<std::scientific<<alpha
                                << " restored="<<(soc.iterated_success?1:0)
                                << " space=full-lifted"
                                << " model=all-constraint-defect"
                                << " theta="<<soc.feasibility.total_inf
                                << " corr="<<soc.correction_inf
                                << " ncorr="<<soc.normal_correction_inf
                                << " qpcorr="<<soc.qp_correction_inf
                                << " eqtarget="<<soc.equality_target_inf
                                << " eqdefect="<<soc.equality_defect_inf
                                << " ineqtarget="<<soc.inequality_target_violation_inf
                                << " ineqdefect="<<soc.inequality_defect_violation_inf
                                << " ineqrhs="<<soc.inequality_correction_rhs_inf
                                << "\n"<<std::defaultfloat;
                  }
                  result.soc_inner_iterations+=soc.inner_iterations;
                  result.soc_factorizations+=soc.factorizations;
                  result.inner_iterations+=soc.inner_iterations;
                  result.kkt_factorizations+=soc.factorizations;
                  result.kkt_symbolic_analyses+=soc.symbolic_analyses;
                  if(soc.success) {
                    trial=std::move(soc.x);
                    tf=soc.feasibility;
                    soc_used=true;
                  }
                }
                const double ft=structural_quadratic_value(
                    problem.objective,trial);
                const bool objective_ok=(predicted_decrease>predicted_floor)
                    ? armijo_objective_ok(corr_fbase,ft,alpha,predicted_decrease,st_)
                    : ft<=corr_fbase+100.0*st_.inner_tolerance*
                                      std::max(1.0,std::abs(corr_fbase));
                bool accept=false;
                bool corrector_objective_step=false;
                bool corrector_hstep=false;
                if(has_quadratic_equality) {
                  const double funnel_noise=8192.0*
                      std::numeric_limits<double>::epsilon()*
                      std::max(1.0,equality_funnel);
                  const bool side_feasible=inequality_bound_violation(tf)<=
                      final_feasibility_tube_limit(st_);
                  const bool within_funnel=tf.equality_inf<=
                      equality_funnel+funnel_noise;
                  const bool nondominated=filter_acceptable(
                      filter,tf.total_inf,ft,st_);
                  corrector_objective_step=
                      side_feasible && within_funnel && objective_ok;
                  const double equality_target=std::max(
                      final_feasibility_tube_limit(st_),
                      (1.0-st_.dca_feasibility_step_fraction*alpha)*
                          corr_fcur.equality_inf);
                  corrector_hstep=side_feasible &&
                      tf.equality_inf<corr_fcur.equality_inf &&
                      tf.equality_inf<=equality_target;
                  accept=nondominated &&
                      (corrector_objective_step || corrector_hstep);
                } else {
                  accept=within_final_feasibility_tube(tf.total_inf,st_) &&
                         objective_ok;
                }
                if(accept) {
                  const MultiplierEstimate mapped=multipliers_from_inner(
                      problem,cb,ci,st_);
                  x=std::move(trial);
                  fcand=tf;
                  objcand=ft;
                  warm_mult=blend_multipliers(corr_mult,mapped,alpha);
                  invalidate_mapped_stationarity_cache();
                  warm_mult_valid=warm_mult.valid;
                  if(has_quadratic_equality) warm_mult_model_only=true;
                  if(has_quadratic_equality &&
                     !st_.dca_single_loop_primal_dual)
                    refit_path_multipliers(
                        x,outer,"accepted-corrector");
                  continuation_valid=false;
                  continuation_equality_duals.clear();
                  continuation_inequality_duals.clear();
                  continuation_inequality_slacks.clear();
                  continuation_inequality_primal_residuals.clear();
                  continuation_inequality_keys.clear();
                  corrector_accepted=true;
                  corrector_accepted_feasibility_step=
                      !corrector_objective_step && corrector_hstep;
                  ++result.lazy_dca_corrector_acceptances;
                  if(soc_used) ++result.soc_accepted_steps;
                  if(has_quadratic_equality &&
                     corrector_accepted_feasibility_step) {
                    const double contracted=st_.dca_equality_funnel_contraction*
                        equality_funnel;
                    const double trial_guard=st_.dca_equality_funnel_growth*
                        tf.equality_inf;
                    equality_funnel=std::max(st_.feasibility_tolerance,
                        std::min(equality_funnel,std::max(contracted,trial_guard)));
                    result.lazy_dca_final_equality_funnel=equality_funnel;
                  }
                  if(st_.verbose)
                    std::cout << "[LAZY-DCA-CORRECTOR] k="<<outer
                              << " accepted=1 alpha="<<std::scientific<<alpha
                              << " theta="<<tf.total_inf
                              << " f="<<ft<<"\n"<<std::defaultfloat;
                  break;
                }
                alpha*=st_.line_search_contraction;
                if(alpha<st_.minimum_step_length) break;
              }
            }
          }
          if(!corrector_accepted) {
            ++result.lazy_dca_corrector_rejections;
            if(st_.verbose)
              std::cout << "[LAZY-DCA-CORRECTOR] k="<<outer
                        << " accepted=0\n";
          }
        }

        const double current_mapped_stationarity=warm_mult_valid
            ? cached_projected_stationarity(x)
            : std::numeric_limits<double>::infinity();
        const Feasibility terminal_feasibility=
            structural_feasibility(problem,x);
        const bool terminal_local_neighborhood=warm_mult_valid &&
            terminal_feasibility.total_inf<=st_.feasibility_tolerance &&
            current_mapped_stationarity<=
                st_.terminal_kkt_trigger_stationarity;
        const bool auto_large_equality_kkt=
            has_quadratic_equality && problem.inequalities.empty() &&
            terminal_local_neighborhood;
        // Large inequality-constrained takeover remains explicit opt-in.  A
        // mapped multiplier which failed its original-Jacobian refit is valid
        // for the next convex model but not for active-manifold
        // identification, so fail closed on `warm_mult_model_only` here.
        const bool opt_in_large_inequality_kkt=
            problem.n>64 && condensed_equality_mode &&
            !problem.inequalities.empty() &&
            st_.terminal_kkt_large_scale_enabled &&
            !warm_mult_model_only && terminal_local_neighborhood;
        const bool lazy_terminal_kkt_allowed=
            !st_.dca_single_loop_primal_dual &&
            !skip_single_loop_auxiliaries &&
            (problem.n<=64 || auto_large_equality_kkt ||
             opt_in_large_inequality_kkt);
        QPDualActiveSet lazy_terminal_aset;
        bool lazy_terminal_aset_available=false;
        if(st_.terminal_kkt_enabled && lazy_terminal_kkt_allowed &&
           terminal_local_neighborhood && st_.linear_backend!="qdldl" &&
           st_.linear_backend!="cholesk") {
          lazy_terminal_aset=qpdual_active_set(problem,x,warm_mult,st_);
          lazy_terminal_aset_available=true;
          result.terminal_kkt_multiplier_cutoff=
              lazy_terminal_aset.multiplier_cutoff;
          if(lazy_terminal_aset.signature==lazy_terminal_active_signature) {
            ++lazy_terminal_active_stability;
          } else {
            lazy_terminal_active_signature=lazy_terminal_aset.signature;
            lazy_terminal_active_stability=1;
          }
          result.terminal_kkt_active_stability=
              lazy_terminal_active_stability;
        } else {
          lazy_terminal_active_signature.clear();
          lazy_terminal_active_stability=0;
        }
        const int lazy_terminal_stability_required=problem.n<=64
            ? 1
            : std::max(1,st_.terminal_kkt_active_stability_iterations);
        const bool lazy_terminal_stable=lazy_terminal_aset_available &&
            lazy_terminal_active_stability>=
                lazy_terminal_stability_required;
        const bool lazy_terminal_iteration_allowed=
            opt_in_large_inequality_kkt ||
            outer>=st_.terminal_kkt_start_outer;
        if(st_.terminal_kkt_enabled && lazy_terminal_kkt_allowed &&
           terminal_local_neighborhood && lazy_terminal_iteration_allowed &&
           lazy_terminal_stable && st_.linear_backend!="qdldl" &&
           st_.linear_backend!="cholesk") {
          const double terminal_objective_before=
              structural_quadratic_value(problem.objective,x);
          const auto tk=kkt_polish_qpdual(
              problem,x,warm_mult,st_,opt_in_large_inequality_kkt);
          result.terminal_kkt_attempts += tk.attempted?1:0;
          result.terminal_kkt_iterations += tk.iterations;
          if(tk.improved || tk.success) {
            const double terminal_objective_after=problem.objective.value(tk.x);
            const double terminal_objective_noise=128.0*
                std::numeric_limits<double>::epsilon()*
                std::max(1.0,std::abs(terminal_objective_before));
            // A fully certified KKT point may terminate immediately.  A
            // partial local sequence, however, must preserve the monotone
            // objective invariant of the surrounding Lazy-DCA iteration.
            // Otherwise a backend-dependent partial Newton exit can leave a
            // point dominated by the existing filter and make the next DCA
            // line search fail.
            // KKT is a first-order certificate; it is not by itself a license
            // to violate the monotone feasible-axis globalization used by the
            // surrounding Lazy-DCA route.  Even a fully certified endpoint is
            // committed only when it does not increase the original
            // objective (up to roundoff).  Otherwise the whole autonomous
            // sequence is rolled back.
            const bool terminal_commit_safe=terminal_objective_after<=
                terminal_objective_before+terminal_objective_noise;
            if(terminal_commit_safe) {
              x=tk.x;
              // A post-step Newton correction invalidates the alpha history:
              // its endpoint no longer belongs to the memorized DCA path.
              condensed_alpha_history_valid=false;
              condensed_alpha_one_streak=0;
              warm_mult=tk.multipliers;
              invalidate_mapped_stationarity_cache();
              warm_mult_valid=warm_mult.valid;
              warm_mult_model_only=!tk.success;
              result.terminal_kkt_accepted_steps += tk.accepted_steps;
              continuation_valid=false;
              continuation_equality_duals.clear();
              continuation_inequality_duals.clear();
              continuation_inequality_slacks.clear();
              continuation_inequality_primal_residuals.clear();
              continuation_inequality_keys.clear();
              // A partial terminal sequence may legitimately end when an
              // inequality enters or leaves the active manifold.  Require a
              // fresh stability window before attempting another takeover.
              lazy_terminal_active_signature.clear();
              lazy_terminal_active_stability=0;
              if(st_.verbose)
                std::cout << "[LAZY-DCA-KKT-CORRECTOR] k="<<outer
                          << " accepted="<<tk.accepted_steps
                          << " stationarity="<<std::scientific<<tk.stationarity
                          << " objective_delta="
                          << terminal_objective_after-terminal_objective_before
                          << "\n"<<std::defaultfloat;
            } else {
              ++result.terminal_kkt_rejected_steps;
              lazy_terminal_active_signature.clear();
              lazy_terminal_active_stability=0;
              if(st_.verbose)
                std::cout << "[LAZY-DCA-KKT-CORRECTOR] k="<<outer
                          << " accepted=0 reason=partial-objective-increase"
                          << " objective_delta="<<std::scientific
                          << terminal_objective_after-terminal_objective_before
                          << " internal_steps="<<tk.accepted_steps
                          << "\n"<<std::defaultfloat;
            }
          } else if(tk.attempted) {
            ++result.terminal_kkt_rejected_steps;
            lazy_terminal_active_signature.clear();
            lazy_terminal_active_stability=0;
          }
        }

        candidate=x;
        fcand=structural_feasibility(problem,x);
        objcand=structural_quadratic_value(problem.objective,x);
        if(warm_mult_valid)
          result.model_stationarity_inf=cached_projected_stationarity(x);
        step=x; axpy(-1.0,outer_origin,step);
        step_inf=inf_norm(step);
        result.step_inf=step_inf;

        if(dca_model_prox>0.0) {
          const bool clean_for_decay=accepted_alpha>=0.5 &&
              accepted_raw_direction_inf<=0.75*dca_step_target;
          if(clean_for_decay) ++dca_model_clean_steps;
          else dca_model_clean_steps=0;
          if(dca_model_clean_steps>=3) {
            dca_model_prox*=0.5;
            if(dca_model_prox<st_.dca_step_control_min_prox) dca_model_prox=0.0;
            dca_model_clean_steps=0;
          }
        }

        result.outer_iterations=outer+1;
        const char* live_tag="D";
        if(result.terminal_kkt_accepted_steps>live_terminal_before) live_tag="N";
        else if(result.lazy_dca_corrector_acceptances>live_corrector_before) live_tag="C";
        else if(result.soc_accepted_steps>live_soc_before) live_tag="S";
        else if(result.lazy_dca_equality_retraction_acceptances>
                live_retraction_before) live_tag="R";
        const double live_dual=st_.live_iterations && warm_mult_valid
            ? cached_projected_stationarity(x)
            : std::numeric_limits<double>::quiet_NaN();
        print_live_row(outer+1,x,fcand,live_dual,step_inf,accepted_alpha,
            active_objective_cvx->rank(),
            result.line_search_backtracks-live_ls_before,live_tag);
        if(st_.verbose) {
          std::cout << "[LAZY-DCA] k="<<outer
                    <<" f="<<std::setprecision(12)
                    <<structural_quadratic_value(problem.objective,x)
                    <<" theta="<<std::scientific<<fcand.total_inf
                    <<" step="<<step_inf
                    <<" alpha="<<accepted_alpha
                    <<" eqmodel="<<(has_quadratic_equality
                        ? st_.dca_equality_model : "affine-exact")
                    <<" soc="<<(accepted_soc?1:0)
                    <<" ws="<<working_set.size()
                    <<" r0="<<active_objective_cvx->rank()
                    <<" inner="<<st_.inner_backend
                    <<"\n"<<std::defaultfloat;
        }

        if(fcand.total_inf<=st_.feasibility_tolerance) {
          // A mapped/model multiplier is refreshed for certification only in
          // the local stationarity neighborhood.  Previously
          // `warm_mult_model_only` alone caused one joint refit and often one
          // more certificate fit after every accepted outer step.
          const bool raw_near_termination=st_.dca_single_loop_primal_dual
              ? raw_single_loop_final_near_termination(x)
              : raw_model_near_termination(x);
          bool kkt_ok=false;
          if(st_.dca_single_loop_primal_dual) {
            // Model-only multipliers may select the next majorant but never
            // certify the original problem by declaration.  The transported
            // state is only a cheap neighborhood detector; requiring that same
            // stale/model-only multiplier to pass the complete original KKT
            // test would make the independent final fit unreachable precisely
            // when it is needed.  Once the primal point is feasible and the
            // raw transported stationarity is locally small, spend the sole
            // certificate-only full adjustment.
            if(raw_near_termination) {
              kkt_ok=run_single_loop_final_certificate(
                  x,outer,"terminal-neighborhood");
              if(!kkt_ok) {
                result.status=Status::NumericalError;
                if(result.message.empty())
                  result.message=
                      "single-loop final multiplier certificate failed";
                break;
              }
            }
          } else {
            if(raw_near_termination && warm_mult_model_only &&
               has_quadratic_equality)
              refit_path_multipliers(x,outer,"certificate");
            kkt_ok=raw_near_termination && warm_mult_valid &&
                !warm_mult_model_only &&
                mapped_certificate_passes(problem,x,warm_mult,st_,result);
            // The path multipliers are already an original-Jacobian projection.
            // A second active-set fit is useful only in the terminal neighborhood;
            // far from stationarity it merely repeats a large sparse/dense solve.
            if(!kkt_ok && raw_near_termination)
              kkt_ok=fresh_certificate_passes(
                  problem,x,st_,result,&certificate_multiplier_projector);
          }
          if(kkt_ok) {
            result.status=Status::Solved;
            result.message="original-QCQP KKT test satisfied by single-route Lazy-DCA";
            break;
          }
        }
        // A small DCA displacement is not itself a KKT test.  v0.4.0 used
        // 0.1*stationarity_tolerance, which stopped A4X12 around a 1e-6 step
        // while the mapped model stationarity was still O(1e-4).  Only declare
        // a numerical fixed-point stall at the explicit step tolerance.
        const double numerical_step_floor=100.0*std::numeric_limits<double>::epsilon()*
            std::max(1.0,inf_norm(x));
        const double dca_step_gate=std::max(st_.step_tolerance,numerical_step_floor);
        if(step_inf<=dca_step_gate) ++small_step_streak; else small_step_streak=0;
        if(small_step_streak>=5) {
          // A transported primal-dual state is only a path state. Near a
          // primal fixed point it may remain just outside the cheap dual gate
          // even though the point admits a valid original-QCQP multiplier.
          // This is precisely the terminal event for which the single-loop
          // contract permits its sole complete multiplier adjustment.
          const bool terminal_primal_feasible=
              fcand.total_inf<=st_.feasibility_tolerance;
          const bool terminal_certified=
              st_.dca_single_loop_primal_dual && terminal_primal_feasible &&
              run_single_loop_final_certificate(
                  x,outer,"primal-fixed-point");
          if(terminal_certified) {
            result.status=Status::Solved;
            result.message=
                "original-QCQP KKT test satisfied at single-loop primal fixed point";
          } else {
            result.status=Status::MaxOuterIterations;
            result.message=st_.dca_single_loop_primal_dual &&
                    terminal_primal_feasible
                ? "single-loop primal fixed point failed the final original-QCQP KKT certificate"
                : "Lazy-DCA numerical fixed-point stall before mapped original-QCQP KKT tolerance";
          }
          break;
        }
      }

      // A rejected model cannot be committed. At a numerically stationary
      // accepted primal point, however, the terminal event can precede the
      // five accepted-small-step counter: a new line search has no resolvable
      // objective decrease left. Spend the SAME sole independent final fit on
      // that accepted point. No model multiplier or small step certifies KKT.
      if(st_.dca_equality_model=="tangent-inertia" &&
         result.status==Status::LineSearchFailure &&
         !single_loop_final_certificate_attempted && result.outer_iterations>0 &&
         std::isfinite(result.step_inf) &&
         result.step_inf<=std::max(st_.step_tolerance,
             100.0*std::numeric_limits<double>::epsilon()*std::max(1.0,inf_norm(x))) &&
         structural_feasibility(problem,x).total_inf<=st_.feasibility_tolerance) {
        ++result.equality_tangent_stall_certificates;
        if(run_single_loop_final_certificate(x,result.outer_iterations,"tangent-primal-stall")) {
          result.status=Status::Solved;
          result.message="original-QCQP KKT certificate passed at tangent-model primal stall";
        }
      }
      if(result.status==Status::NumericalError && result.message.empty()) {
        result.status=Status::MaxOuterIterations;
        result.message="Lazy-DCA outer iteration limit reached";
      }
      if(result.outer_iterations>=st_.max_outer_iterations && result.status==Status::NumericalError) {
        result.status=Status::MaxOuterIterations;
        result.message="Lazy-DCA outer iteration limit reached";
      }
      result.x=x;
      result.objective=structural_quadratic_value(problem.objective,x);
      result.feasibility=
          (condensed_equality_mode && condensed_retraction.ready())
              ? condensed_retraction.evaluate_feasibility(x)
          : (condensed_equality_mode && condensed_geometry.ready())
              ? condensed_geometry.evaluate_feasibility(x)
              : structural_feasibility(problem,x);
      // A successful fresh certificate at the accepted final point is already
      // exact.  Do not overwrite its metrics with the less accurate mapped
      // path multiplier while packaging the result.  When no exact certificate
      // has passed, retain the historical final refit/certificate sequence and
      // all of its absolute tolerances unchanged.
      if(result.feasibility.total_inf<=st_.feasibility_tolerance &&
         !result.ipopt_desired_convergence) {
        if(st_.dca_single_loop_primal_dual) {
          // Packaging must not turn every merely feasible failure/timeout into
          // an expensive multiplier reconstruction.  The same cheap
          // transported-stationarity neighborhood used in-loop is required;
          // the independent fit below is then the sole logical full refit and
          // its outcome explicitly controls the returned status.
          ++result.dca_single_loop_refits_skipped;
          if(!single_loop_final_certificate_attempted &&
             raw_single_loop_final_near_termination(x)) {
            if(run_single_loop_final_certificate(
                   x,result.outer_iterations,"final-packaging")) {
              result.status=Status::Solved;
              result.message=
                  "original-QCQP KKT test satisfied by single-route Lazy-DCA";
            } else {
              result.status=Status::NumericalError;
              if(result.message.empty())
                result.message=
                    "single-loop final multiplier certificate failed";
            }
          }
        } else {
          if((!warm_mult_valid || warm_mult_model_only) &&
             has_quadratic_equality)
            refit_path_multipliers(x,result.outer_iterations,"final");
          if(warm_mult_valid && !warm_mult_model_only)
            mapped_termination_certificate(problem,x,warm_mult,st_,result);
          if(!result.ipopt_desired_convergence &&
             (!warm_mult_valid || warm_mult_model_only ||
              mapped_near_termination()))
            fresh_termination_certificate(
                problem,x,st_,result,&certificate_multiplier_projector);
        }
      }
      result.multiplier_row_space_factorizations=
          path_multiplier_projector.factorizations()+
          certificate_multiplier_projector.factorizations();
      result.multiplier_row_space_solves=
          path_multiplier_projector.solves()+certificate_multiplier_projector.solves();
      result.multiplier_row_space_symbolic_analyses=
          path_multiplier_projector.symbolic_analyses()+
          certificate_multiplier_projector.symbolic_analyses();
      result.multiplier_row_space_symbolic_reuses=
          path_multiplier_projector.symbolic_reuses()+
          certificate_multiplier_projector.symbolic_reuses();
      result.multiplier_row_space_failures=
          path_multiplier_projector.failures()+certificate_multiplier_projector.failures();
      result.lazy_dca_condensed_factorizations=
          condensed_geometry.factorization_count()+
          condensed_retraction.factorization_count();
      result.lazy_dca_condensed_normal_solves=
          condensed_geometry.solve_count()+condensed_retraction.solve_count();
      result.lazy_dca_condensed_geometry_normal_solves=
          std::max(0,result.lazy_dca_condensed_normal_solves-
                     result.lazy_dca_condensed_retraction_normal_solves);
      result.lazy_dca_condensed_symbolic_analyses=
          condensed_geometry.symbolic_analyses()+
          condensed_retraction.symbolic_analyses();
      result.lazy_dca_condensed_symbolic_reuses=
          condensed_geometry.symbolic_reuses()+condensed_retraction.symbolic_reuses();
      result.lazy_dca_condensed_symbolic_invalidations=
          condensed_geometry.symbolic_invalidations()+
          condensed_retraction.symbolic_invalidations();
      result.lazy_dca_condensed_cached_regularization_attempts=
          condensed_geometry.cached_regularization_attempts()+
          condensed_retraction.cached_regularization_attempts();
      result.lazy_dca_condensed_cached_regularization_hits=
          condensed_geometry.cached_regularization_hits()+
          condensed_retraction.cached_regularization_hits();
      result.lazy_dca_condensed_sparse_equality_evaluations=
          condensed_geometry.sparse_equality_evaluations()+
          condensed_retraction.sparse_equality_evaluations();
      result.lazy_dca_condensed_parallel_equality_batches=
          condensed_geometry.parallel_equality_batches()+
          condensed_retraction.parallel_equality_batches();
      result.lazy_dca_condensed_normal_refinement_steps=
          condensed_geometry.normal_refinement_steps()+
          condensed_retraction.normal_refinement_steps();
      result.lazy_dca_condensed_geometry_normal_refinement_steps=
          std::max(0,result.lazy_dca_condensed_normal_refinement_steps-
                     result.lazy_dca_condensed_retraction_normal_refinement_steps);
      result.lazy_dca_condensed_normal_refinement_zero_solves=
          condensed_geometry.normal_refinement_zero_solves()+
          condensed_retraction.normal_refinement_zero_solves();
      result.lazy_dca_condensed_normal_refinement_one_solves=
          condensed_geometry.normal_refinement_one_solves()+
          condensed_retraction.normal_refinement_one_solves();
      result.lazy_dca_condensed_normal_refinement_two_solves=
          condensed_geometry.normal_refinement_two_solves()+
          condensed_retraction.normal_refinement_two_solves();
      result.lazy_dca_condensed_normal_refinement_three_plus_solves=
          condensed_geometry.normal_refinement_three_plus_solves()+
          condensed_retraction.normal_refinement_three_plus_solves();
      result.lazy_dca_condensed_relaxed_normal_solves=
          condensed_geometry.relaxed_normal_solves()+
          condensed_retraction.relaxed_normal_solves();
      result.lazy_dca_condensed_strict_normal_solves=
          condensed_geometry.strict_normal_solves()+
          condensed_retraction.strict_normal_solves();
      result.lazy_dca_condensed_relaxed_accepts_above_strict_tolerance=
          condensed_geometry.relaxed_accepts_above_strict_tolerance()+
          condensed_retraction.relaxed_accepts_above_strict_tolerance();
      result.lazy_dca_condensed_strict_retry_retractions=
          condensed_geometry.strict_retry_retractions()+
          condensed_retraction.strict_retry_retractions();
      result.lazy_dca_condensed_max_normal_initial_backward_error=
          std::max(condensed_geometry.max_normal_initial_backward_error(),
                   condensed_retraction.max_normal_initial_backward_error());
      const int condensed_total_solves=condensed_geometry.solve_count()+
          condensed_retraction.solve_count();
      if (condensed_total_solves > 0) {
        result.lazy_dca_condensed_mean_normal_initial_backward_error=
            (condensed_geometry.sum_normal_initial_backward_error()+
             condensed_retraction.sum_normal_initial_backward_error())/
            static_cast<double>(condensed_total_solves);
        result.lazy_dca_condensed_mean_normal_final_backward_error=
            (condensed_geometry.sum_normal_final_backward_error()+
             condensed_retraction.sum_normal_final_backward_error())/
            static_cast<double>(condensed_total_solves);
      }
      result.lazy_dca_condensed_max_normal_backward_error=
          std::max(condensed_geometry.max_normal_backward_error(),
                   condensed_retraction.max_normal_backward_error());
      result.lazy_dca_condensed_gmres_workspace_growths=
          condensed_geometry.gmres_workspace_growths()+
          condensed_retraction.gmres_workspace_growths();
      result.lazy_dca_condensed_zero_rhs_solve_skips=
          condensed_geometry.zero_rhs_solve_skips()+
          condensed_retraction.zero_rhs_solve_skips();
      result.lazy_dca_condensed_flow_lift_schur_detected_variables=std::max(
          condensed_geometry.flow_lift_schur_detected_variables(),
          condensed_retraction.flow_lift_schur_detected_variables());
      result.lazy_dca_condensed_flow_lift_schur_factor_dimension=std::max(
          condensed_geometry.normal_factor_dimension(),
          condensed_retraction.normal_factor_dimension());
      result.lazy_dca_condensed_flow_lift_schur_factorizations=
          condensed_geometry.flow_lift_schur_factorizations()+
          condensed_retraction.flow_lift_schur_factorizations();
      result.lazy_dca_condensed_flow_lift_schur_fallbacks=
          condensed_geometry.flow_lift_schur_fallbacks()+
          condensed_retraction.flow_lift_schur_fallbacks();
      result.curvature_calls=curvature.calls()+condensed_curvature.calls()+
          condensed_geometry.tangent_curvature_calls();
      result.objective_rank=objective_cvx.rank()+
          result.lazy_dca_condensed_tangent_rank_max;
      result.inequality_total_rank=0;
      for(const auto& c:ineq_cvx) result.inequality_total_rank+=c.rank();
      result.active_inequalities=(int)working_set.size();
      // Preserve the historical meaning of this public field: the fixed
      // user-requested extra prox only.  Single-loop baseline and adaptive
      // retry maxima have dedicated diagnostics.
      result.final_proximal_regularization=st_.dca_extra_proximal_regularization;
      result.constraint_curvature_seconds=curvature.seconds();
      result.elapsed_seconds=elapsed();
      return result;
    }

    auto trust_radius = std::min(st_.trust_region_max_radius,
        std::max(st_.trust_region_min_radius,st_.trust_region_initial_radius));
    result.final_trust_region_radius = trust_radius;
    double last_step = std::numeric_limits<double>::infinity();
    double last_stationarity = std::numeric_limits<double>::infinity();
    int last_lagrangian_rank = 0;
    TangentDiagnostics last_tangent;
    MultiplierEstimate qp_multiplier_state;
    bool qp_multiplier_state_valid = false;
    MultiplierEstimate final_mapped_certificate_multipliers;
    bool final_mapped_certificate_valid = false;
    std::vector<long long> terminal_active_signature;
    int terminal_active_stability = 0;
    bool terminal_requires_qp_refresh = false;
    bool terminal_policy_reported = false;

    // ----------------------------- Phase II ----------------------------
    for (int outer = 0; outer < st_.max_outer_iterations; ++outer) {
      result.outer_iterations = outer;
      if (elapsed() >= st_.time_limit_seconds) {
        result.status = Status::TimeLimit;
        result.message = "wall-clock limit reached";
        break;
      }

      const Feasibility fk_feas = feasibility(problem,x);
      const double fk = problem.objective.value(x);

      // Dynamic working set with hysteresis.  Additions remain violation-driven
      // as before; dropping is allowed only for a clearly slack constraint
      // whose QP multiplier is essentially zero.
      if (st_.working_set_dropping && qp_multiplier_state_valid) {
        std::vector<int> drop;
        for (int i : working_set) {
          const double mu = (i < (int)qp_multiplier_state.inequality.size())
              ? std::max(0.0,qp_multiplier_state.inequality[i]) : 0.0;
          if (problem.inequalities[i].value(x) < -st_.working_set_drop_slack &&
              mu <= st_.working_set_drop_multiplier) drop.push_back(i);
        }
        for (int i : drop) {
          working_set.erase(i);
          if(i<(int)qp_multiplier_state.inequality.size())
            qp_multiplier_state.inequality[i]=0.0;
          ++result.working_set_drops;
        }
      }

      // Preserve the v0.2.6 multiplier path during the global SQP phase.
      // In the terminal phase use the primal-dual multipliers returned by the
      // previously solved convex SQP subproblem.  These are the natural SQP
      // multiplier update; unlike the v0.2.7 NNLS fit they are consistent with
      // the model that actually generated the accepted step.
      const bool terminal_multiplier_mode =
          fk_feas.total_inf <= 10.0*st_.feasibility_tolerance &&
          qp_multiplier_state_valid;
      MultiplierEstimate mult = terminal_multiplier_mode
          ? qp_multiplier_state
          : estimate_multipliers_legacy(problem,x,working_set,st_);
      if (fk_feas.total_inf <= 10.0*st_.feasibility_tolerance) {
        last_stationarity = terminal_multiplier_mode
            ? projected_stationarity(problem,x,mult,std::max(1e-10,10.0*st_.feasibility_tolerance))
            : estimate_active_multipliers_legacy(problem,x,st_).stationarity;
      } else {
        last_stationarity = mult.valid
            ? projected_stationarity(problem,x,mult,
                std::max(1e-10,10.0*st_.feasibility_tolerance))
            : std::numeric_limits<double>::infinity();
      }
      if (fk_feas.total_inf <= st_.feasibility_tolerance &&
          last_stationarity <= st_.stationarity_tolerance) {
        if (fresh_certificate_passes(problem,x,st_,result)) {
          result.status = Status::Solved;
          result.message = "fresh original-QCQP primal-dual KKT certificate satisfied";
          break;
        }
      }

      const bool terminal_size_allowed =
          (problem.n <= 64 || st_.terminal_kkt_large_scale_enabled);
      const bool terminal_backend_allowed =
          st_.linear_backend != "qdldl" && st_.linear_backend != "cholesk";
      const bool terminal_kkt_allowed = st_.terminal_kkt_enabled &&
          terminal_size_allowed && terminal_backend_allowed;
      if (st_.terminal_kkt_enabled && !terminal_kkt_allowed &&
          !terminal_policy_reported && st_.verbose) {
        std::cout << "[KKT-QPDUAL-SKIP] disabled_by_policy reason="
                  << (!terminal_size_allowed ? "large_scale" : "ldl_backend")
                  << " n=" << problem.n
                  << " backend=" << st_.linear_backend << "\n";
        terminal_policy_reported=true;
      }

      // Barrier-aware reduced-space diagnostics and guarded primal Newton.
      // The active signature must remain unchanged for a few consecutive
      // outer iterates before the local accelerator is allowed to fire.  This
      // prevents tiny Mehrotra barrier duals from changing the manifold on
      // every call.  A terminal step never edits QP multipliers; after one
      // accepted reduced step, at least one fresh convex-QP solve is required.
      QPDualActiveSet terminal_aset;
      bool terminal_aset_available=false;
      if (terminal_kkt_allowed && terminal_multiplier_mode &&
          fk_feas.total_inf <= st_.feasibility_tolerance &&
          last_stationarity <= 10.0*st_.terminal_kkt_trigger_stationarity) {
        terminal_aset=qpdual_active_set(problem,x,qp_multiplier_state,st_);
        terminal_aset_available=true;
        last_tangent=terminal_aset.tangent;
        result.terminal_kkt_multiplier_cutoff=terminal_aset.multiplier_cutoff;
        if(terminal_aset.signature==terminal_active_signature) {
          ++terminal_active_stability;
        } else {
          terminal_active_signature=terminal_aset.signature;
          terminal_active_stability=1;
        }
        result.terminal_kkt_active_stability=terminal_active_stability;
      }

      const int terminal_stability_required = (problem.n <= 64)
          ? 1 : std::max(1,st_.terminal_kkt_active_stability_iterations);
      const bool terminal_stable = terminal_aset_available &&
          terminal_active_stability >= terminal_stability_required;
      const bool terminal_ready = terminal_kkt_allowed && terminal_multiplier_mode &&
          !terminal_requires_qp_refresh && outer >= st_.terminal_kkt_start_outer &&
          fk_feas.total_inf <= st_.feasibility_tolerance &&
          last_stationarity <= st_.terminal_kkt_trigger_stationarity &&
          terminal_aset_available && terminal_aset.tangent.reduced_negative_eigenvalues == 0;

      if (terminal_ready && !terminal_stable) {
        ++result.terminal_kkt_skipped_unstable;
        if(st_.verbose) {
          std::cout << "[KKT-QPDUAL-SKIP] unstable_active_set stability="
                    << terminal_active_stability << "/"
                    << terminal_stability_required
                    << " active=" << terminal_aset.active.size()
                    << " mucut=" << std::scientific << terminal_aset.multiplier_cutoff
                    << "\n" << std::defaultfloat;
        }
      }

      if (terminal_ready && terminal_stable) {
        ++result.terminal_kkt_attempts;
        const Vec x_before_terminal=x;
        const auto tk = kkt_polish_qpdual(problem,x,qp_multiplier_state,st_);
        result.terminal_kkt_iterations += tk.iterations;
        result.terminal_kkt_accepted_steps += tk.accepted_steps;
        result.terminal_kkt_active_set_drops += tk.active_set_drops;
        result.terminal_kkt_multiplier_cutoff=tk.multiplier_cutoff;
        last_tangent = tk.tangent;
        if (tk.improved || tk.success) {
          x = tk.x;
          // QP duals are intentionally frozen by the reduced primal step.
          // Force one ordinary convex-SQP solve before another local step.
          terminal_requires_qp_refresh=true;
          last_stationarity = tk.stationarity;
          Vec tstep=x; axpy(-1.0,x_before_terminal,tstep); last_step=inf_norm(tstep);
          result.outer_iterations = outer+1;
          if (tk.success) {
            qp_multiplier_state=tk.multipliers;
            qp_multiplier_state_valid=tk.multipliers.valid;
          }
          if (tk.success && qp_multiplier_state_valid &&
              mapped_certificate_passes(problem,x,qp_multiplier_state,st_,result)) {
            final_mapped_certificate_multipliers=qp_multiplier_state;
            final_mapped_certificate_valid=true;
            result.status = Status::Solved;
            result.message = "guarded QP-dual primal-dual Newton plus mapped KKT certificate converged";
            break;
          }
          continue;
        }
        ++result.terminal_kkt_rejected_steps;
      }

      // If a phase-II trial has drifted materially away from feasibility,
      // restore on the ORIGINAL QCQP before constructing another optimality model.
      if (st_.restoration_enabled && fk_feas.total_inf > 100.0*st_.feasibility_tolerance) {
        FeasibilityRestoration restoration(st_);
        RestorationResult rr = restoration.solve(problem,x);
        result.restoration_iterations += rr.iterations;
        result.restoration_restarts += rr.restarts;
        result.restoration_linear_solves += rr.linear_solves;
        result.restoration_exact_hessian_solves += rr.exact_hessian_solves;
        result.restoration_matrix_free_solves += rr.matrix_free_solves;
        result.restoration_row_space_solves += rr.row_space_solves;
        result.restoration_harmonic_attempts += rr.harmonic_attempts;
        result.restoration_harmonic_acceptances += rr.harmonic_acceptances;
        result.restoration_harmonic_factorizations += rr.harmonic_factorizations;
        result.restoration_tangent_activation_attempts += rr.tangent_activation_attempts;
        result.restoration_tangent_activation_acceptances += rr.tangent_activation_acceptances;
        result.restoration_tangent_activation_components += rr.tangent_activation_components;
        result.restoration_tangent_activation_feasibility = rr.tangent_activation_feasibility;
        result.restoration_tangent_activation_step_inf = rr.tangent_activation_step_inf;
        result.restoration_cg_iterations += rr.cg_iterations;
        result.restoration_exact_workspace_growths += rr.exact_workspace_growths;
        result.restoration_exact_matvecs += rr.exact_matvecs;
        result.restoration_exact_parallel_row_dot_batches +=
            rr.exact_parallel_row_dot_batches;
        result.restoration_exact_parallel_teams += rr.exact_parallel_teams;
        result.restoration_exact_sparse_row_dot_batches +=
            rr.exact_sparse_row_dot_batches;
        result.restoration_exact_sparse_row_dot_verifications +=
            rr.exact_sparse_row_dot_verifications;
        result.restoration_exact_sparse_row_dot_fallbacks +=
            rr.exact_sparse_row_dot_fallbacks;
        result.restoration_line_search_backtracks += rr.line_search_backtracks;
        result.restoration_stagnation_events += rr.stagnation_events;
        result.restoration_minimax_qp_solves += rr.minimax_qp_solves;
        result.restoration_minimax_inner_iterations += rr.minimax_inner_iterations;
        result.restoration_minimax_factorizations += rr.minimax_factorizations;
        result.restoration_minimax_accepted_steps += rr.minimax_accepted_steps;
        result.restoration_minimax_line_search_backtracks += rr.minimax_line_search_backtracks;
        result.restoration_structured_abs_links = std::max(result.restoration_structured_abs_links, rr.structured_abs_links);
        if (rr.effective_dimension > 0) result.restoration_effective_dimension = rr.effective_dimension;
        if (!rr.success) {
          result.status = Status::RestorationFailure;
          result.message = "phase-II feasibility recovery failed: " + rr.message;
          x = rr.x;
          break;
        }
        x = rr.x;
        continue;
      }

      const SymMatrix B = lagrangian_hessian(problem,mult);
      Convexification lag_cvx;
      curvature.ensure_convex(B, lag_cvx, 0.0, st_.max_lagrangian_rank);
      last_lagrangian_rank = lag_cvx.rank();

      bool accepted_outer = false;
      if (st_.trust_region_enabled) {
        // r6: exact-penalty trust-region SQP globalization.  Every rejected
        // trial shrinks ||d||_2 <= Delta and RE-SOLVES the convex QCQP.
        // Acceptance is based on Phi_sigma=f+sigma*theta, with sigma chosen
        // from the current original-QCQP multiplier l1 norm.  This replaces
        // the r5 hard 1e-10 feasibility funnel that collapsed Delta.
        const int max_tr_rejects=std::max(1,st_.trust_region_max_rejections);
        for(int tr=0;tr<max_tr_rejects && !accepted_outer;++tr){
          if(elapsed()>=st_.time_limit_seconds){
            result.status=Status::TimeLimit;
            result.message="wall-clock limit reached inside trust-region globalization";
            break;
          }

          InnerResult inner;
          Vec d(problem.n,0.0);
          bool inner_ok=false;
          BuiltSubproblem accepted_built;
          bool accepted_built_valid=false;
          int expansions=0;

          while(expansions<=st_.max_working_set_expansions){
            const bool linearize_phase2_ineq =
                st_.phase2_linearized_inequalities && qp_multiplier_state_valid &&
                fk_feas.total_inf <= st_.feasibility_tolerance;
            auto built=build_subproblem(problem,x,B,lag_cvx,ineq_cvx,working_set,st_,
                                        proximal,linearize_phase2_ineq,trust_radius);
            seed_inner_dual_warm_start(built,problem,qp_multiplier_state,qp_multiplier_state_valid,st_);
            ++result.trust_region_subproblems;
            { auto si=inner_settings(); inner=solve_inner(built.p,si); }
            result.inner_iterations += inner.iterations;
            result.kkt_factorizations += inner.factorizations;
            result.kkt_symbolic_analyses += inner.symbolic_analyses;
            result.last_inner_dual_inf = inner.dual_inf;
            result.last_inner_equality_inf = inner.equality_inf;
            result.last_inner_inequality_inf = inner.inequality_inf;
            result.last_inner_primal_residual_inf = inner.primal_residual_inf;
            result.last_inner_complementarity = inner.complementarity;
            result.last_inner_scaled_model_residual_inf = inner.scaled_model_residual_inf;
            result.last_inner_used_iteration_limit_fallback = inner.used_iteration_limit_fallback;
            absorb_kkt_diagnostics(inner);
            if(inner.used_iteration_limit_fallback)
              ++result.inner_iteration_limit_fallback_acceptances;
            if(!inner.success || static_cast<int>(inner.y.size())<problem.n){
              inner_ok=false;
              break;
            }
            std::copy(inner.y.begin(),inner.y.begin()+problem.n,d.begin());

            // Full trust-region step.  New violated original inequalities are
            // inserted and the bounded subproblem is re-solved.
            Vec xfull=x; axpy(1.0,d,xfull);
            xfull=retract_equalities(problem,xfull,st_.equality_retraction_iterations,
                                     st_.equality_retraction_tolerance,
                                     st_.equality_retraction_regularization,
                                     st_.fraction_to_boundary);
            std::vector<int> add;
            for(int i=0;i<static_cast<int>(problem.inequalities.size());++i){
              if(working_set.count(i)) continue;
              const double add_tol=std::max(st_.working_set_add_tolerance,
                                            st_.feasibility_tolerance);
              if(problem.inequalities[i].value(xfull)>add_tol) add.push_back(i);
            }
            if(add.empty()){
              inner_ok=true;
              accepted_built=std::move(built);
              accepted_built_valid=true;
              break;
            }
            for(int i:add){
              working_set.insert(i);
              if(!linearize_phase2_ineq)
                curvature.ensure_convex(problem.inequalities[i].hessian,ineq_cvx[i],0.0,
                                        st_.max_rank_per_quadratic);
              ++result.working_set_additions;
            }
            ++expansions;
          }

          if(!inner_ok){
            // Numerical rescue only: modestly strengthen the convex inner
            // model and shrink Delta.  This is not an acceptance mechanism.
            proximal=std::min(st_.proximal_regularization_max,
                              std::max(10.0*proximal,1e-8));
            const double old_delta=trust_radius;
            trust_radius=std::max(st_.trust_region_min_radius,
                                  st_.trust_region_shrink_factor*trust_radius);
            if(trust_radius<old_delta) ++result.trust_region_contractions;
            ++result.trust_region_rejected_steps;
            result.final_trust_region_radius=trust_radius;
            if(st_.verbose){
              std::cout << "[TR-REJECT] k="<<outer
                        << " reason=inner delta="<<std::scientific<<old_delta
                        << " -> "<<trust_radius
                        << " prox="<<proximal<<"\n"<<std::defaultfloat;
            }
            if(trust_radius<=st_.trust_region_min_radius*(1.0+1e-12)) break;
            continue;
          }

          Vec trial=x; axpy(1.0,d,trial);
          if(!finite_state(trial)){
            const double old_delta=trust_radius;
            trust_radius=std::max(st_.trust_region_min_radius,
                                  st_.trust_region_shrink_factor*trust_radius);
            if(trust_radius<old_delta) ++result.trust_region_contractions;
            ++result.trust_region_rejected_steps;
            result.final_trust_region_radius=trust_radius;
            continue;
          }
          trial=retract_equalities(problem,trial,st_.equality_retraction_iterations,
                                   st_.equality_retraction_tolerance,
                                   st_.equality_retraction_regularization,
                                   st_.fraction_to_boundary);
          Feasibility tf=feasibility(problem,trial);
          bool soc_candidate_used=false;
          const bool local_linearized_model =
              st_.phase2_linearized_inequalities && qp_multiplier_state_valid &&
              fk_feas.total_inf <= st_.feasibility_tolerance;

          // At most one SOC for this newly solved trust-region candidate.  A
          // contraction gives a different d and may therefore justify a new SOC.
          if(tf.total_inf>st_.feasibility_tolerance && local_linearized_model &&
             st_.second_order_correction &&
             tf.total_inf<=st_.second_order_correction_trigger_ratio*
                               st_.feasibility_tolerance){
            ++result.soc_attempts;
            auto soc=second_order_correction(problem,x,trial,working_set,st_);
            result.soc_inner_iterations += soc.inner_iterations;
            result.soc_factorizations += soc.factorizations;
            result.inner_iterations += soc.inner_iterations;
            result.kkt_factorizations += soc.factorizations;
            result.kkt_symbolic_analyses += soc.symbolic_analyses;
            if(soc.success){
              trial=std::move(soc.x); tf=soc.feasibility; soc_candidate_used=true;
            }else{
              ++result.soc_rejected_steps;
            }
          }

          Vec actual_step=trial; axpy(-1.0,x,actual_step);
          const double actual_step_inf=inf_norm(actual_step);
          const double actual_step_norm=std::sqrt(std::max(0.0,dot(actual_step,actual_step)));
          const double d_norm=std::sqrt(std::max(0.0,dot(d,d)));
          Vec nonlinear_correction=actual_step;
          axpy(-1.0,d,nonlinear_correction);
          const double correction_norm=std::sqrt(std::max(0.0,dot(nonlinear_correction,nonlinear_correction)));
          const double ft=problem.objective.value(trial);

          Vec zero(problem.n,0.0);
          const double model0=accepted_built.p.objective.value(zero);
          // Evaluate the model on the point ACTUALLY tested after equality
          // retraction/SOC, not on the pre-correction d.
          const double modelt=accepted_built.p.objective.value(actual_step);
          const double pred=model0-modelt;
          const double ared=fk-ft;
          const double pred_floor=128.0*std::numeric_limits<double>::epsilon()*
                                  std::max(1.0,std::abs(model0));
          const double rho=(pred>pred_floor && std::isfinite(pred))
              ? ared/pred : -std::numeric_limits<double>::infinity();

          result.last_trust_region_predicted_reduction=pred;
          result.last_trust_region_actual_reduction=ared;

          // Exact-penalty merit ratio.  Since the convex SQP subproblem models
          // the active constraints as feasible, its predicted feasibility
          // violation is zero.  For theta=max violation, ||multipliers||_1 is
          // the natural exact-penalty scale because |lambda^T c| <=
          // ||lambda||_1*theta.
          double multiplier_l1=0.0;
          for(double v:mult.equality) multiplier_l1 += std::abs(v);
          for(double v:mult.inequality) multiplier_l1 += std::max(0.0,v);
          const double penalty=std::max(st_.trust_region_penalty_min,
              st_.trust_region_penalty_safety*(1.0+multiplier_l1));

          // r7: model-consistent exact-penalty reduction with a central
          // feasibility deadband.  r6 incorrectly assumed that every convex
          // SQP subproblem predicts theta_trial=0, so merit_pred contained the
          // O(sigma*theta_k) constant even when Delta -> 0.  On A4X12 this made
          // merit_pred ~ 6e-5 while the objective model reduction was ~ 1e-9,
          // forcing rho < 0 for all smaller radii.
          //
          // The QCQP constraints are cheap and available exactly, so use the
          // SAME observed feasibility change in the predicted and actual merit
          // reductions; only the objective part distinguishes model from truth.
          // A central deadband (10% of feas_tol, never below the configured
          // floor) prevents multiplier magnitudes O(1e5) from amplifying
          // O(1e-12) retraction/SOC noise into a false merit rejection.
          // r9: once Phase II is already feasible, the merit deadband must
          // coincide with the same feasible envelope used by the hard guard.
          // Otherwise a point with theta in (0.1*feas_tol, feas_tol] is legal
          // for preserve_feas but is penalized by sigma=O(||lambda||_1); on
          // A4X12 sigma~9e4 turned harmless O(1e-8) feasibility motion into
          // O(1e-3) merit losses and rejected excellent objective steps.
          const double theta_deadband=(fk_feas.total_inf<=st_.feasibility_tolerance)
              ? std::max(st_.trust_region_feasibility_floor,st_.feasibility_tolerance)
              : std::max(st_.trust_region_feasibility_floor,
                         0.1*st_.feasibility_tolerance);
          const double theta_eff_k=std::max(0.0,fk_feas.total_inf-theta_deadband);
          const double theta_eff_t=std::max(0.0,tf.total_inf-theta_deadband);
          const double merit_feas_change=penalty*(theta_eff_k-theta_eff_t);
          const double merit_pred=pred + merit_feas_change;
          const double merit_ared=ared + merit_feas_change;
          const double merit_floor=pred_floor +
              128.0*std::numeric_limits<double>::epsilon()*
              std::max(1.0,penalty*std::max(theta_eff_k,theta_eff_t));
          const double merit_rho=(merit_pred>merit_floor && std::isfinite(merit_pred))
              ? merit_ared/merit_pred : -std::numeric_limits<double>::infinity();
          result.last_trust_region_ratio=merit_rho;
          result.last_trust_region_penalty=penalty;
          result.last_trust_region_merit_predicted_reduction=merit_pred;
          result.last_trust_region_merit_actual_reduction=merit_ared;

          // Hard safety envelope only.  A feasible Phase-II iterate may use the
          // configured 1e-7 tolerance, but feasibility loss is charged in the
          // merit ratio above; this prevents the r5 Delta collapse at theta=0.
          const double theta_cap=(fk_feas.total_inf<=st_.feasibility_tolerance)
              ? st_.feasibility_tolerance
              : std::max(st_.feasibility_tolerance,fk_feas.total_inf);
          const bool preserve_feas=tf.total_inf<=theta_cap;
          const bool nondominated=filter_acceptable(filter,tf.total_inf,ft,st_);
          const bool objective_step=merit_pred>merit_floor && merit_ared>0.0 &&
                                    merit_rho>=st_.trust_region_eta_accept;
          // r10: the merit deadband and the feasibility-progress liveness
          // threshold serve different purposes and must not be coupled.  In
          // r9, theta_deadband=feas_tol for an already-feasible Phase-II
          // iterate also disabled feasibility_progress whenever
          // theta_k<=feas_tol.  A4X12 k=156 then rejected a model-nondescent
          // step that reduced BOTH the original objective and theta by more
          // than 50%, and the subsequent contractions drove the inner solve
          // into numerical failure.  Keep the merit deadband at feas_tol, but
          // regard any residual above the configured numerical floor as live.
          const double feasibility_progress_floor=
              std::max(st_.trust_region_feasibility_floor,
                       128.0*std::numeric_limits<double>::epsilon());
          const double feasibility_objective_allowance=
              (fk_feas.total_inf<=st_.feasibility_tolerance)
              ? std::max(1e-12,st_.filter_gamma_f*std::max(1.0,std::abs(fk)))
              : std::max(1e-12,0.1*penalty*fk_feas.total_inf);
          const bool feasibility_step=
              fk_feas.total_inf>feasibility_progress_floor &&
              tf.total_inf <= (1.0-st_.trust_region_feasibility_reduction)*
                                  fk_feas.total_inf &&
              ft <= fk + feasibility_objective_allowance;
          // r8: the trust region applies to the point that is actually tested.
          // Equality retraction/SOC may change the raw inner step d, so enforce
          // ||trial-x|| <= Delta up to only solver-scale numerical tolerance.
          const double x_norm=std::sqrt(std::max(0.0,dot(x,x)));
          // r9: r8 used a 1e-6 relative boundary tolerance.  That was tighter
          // than the finite-precision accuracy of the native QCQP solve: A4X12
          // produced model-consistent steps with rho~1 and only 3.3e-5 relative
          // boundary excess, which were repeatedly rejected until Delta hit its
          // floor.  Keep the check strict (default 1e-4 = 0.01%) but align it
          // with the numerical accuracy of the inner trust-region constraint.
          const double trust_radius_tol=std::max(
              st_.trust_region_boundary_absolute_tolerance,
              st_.trust_region_boundary_relative_tolerance*
                  std::max(trust_radius,st_.trust_region_min_radius));
          const double trust_boundary_excess=std::max(0.0,actual_step_norm-trust_radius);
          const bool trust_consistent=trust_boundary_excess <= trust_radius_tol;
          const bool accepted=nondominated && preserve_feas && trust_consistent &&
                              (objective_step || feasibility_step);

          if(st_.verbose){
            std::cout << "[TR] k="<<outer
                      << " try="<<tr
                      << " delta="<<std::scientific<<trust_radius
                      << " step_inf="<<actual_step_inf
                      << " step_2="<<actual_step_norm
                      << " raw_step_2="<<d_norm
                      << " corr_2="<<correction_norm
                      << " pred="<<pred
                      << " ared="<<ared
                      << " rho_obj="<<rho
                      << " sigma="<<penalty
                      << " deadband="<<theta_deadband
                      << " mpred="<<merit_pred
                      << " mared="<<merit_ared
                      << " rho="<<merit_rho
                      << " theta="<<tf.total_inf
                      << " cap="<<theta_cap
                      << " feas_live_floor="<<feasibility_progress_floor
                      << " feas_obj_allow="<<feasibility_objective_allowance
                      << " tr_over="<<trust_boundary_excess
                      << " tr_tol="<<trust_radius_tol
                      << " tr_ok="<<(trust_consistent?1:0)
                      << " type="<<(objective_step?"f":(feasibility_step?"theta":"reject"))
                      << " accept="<<(accepted?1:0)
                      << "\n"<<std::defaultfloat;
          }

          if(!accepted){
            ++result.trust_region_rejected_steps;
            const double old_delta=trust_radius;
            trust_radius=std::max(st_.trust_region_min_radius,
                                  st_.trust_region_shrink_factor*trust_radius);
            if(trust_radius<old_delta) ++result.trust_region_contractions;
            result.final_trust_region_radius=trust_radius;
            if(trust_radius<=st_.trust_region_min_radius*(1.0+1e-12)) break;
            continue;
          }

          filter_insert(filter,fk_feas.total_inf,fk);
          x=std::move(trial);
          last_step=actual_step_inf;
          ++result.trust_region_accepted_steps;
          if(feasibility_step && !objective_step) ++result.trust_region_feasibility_steps;
          if(soc_candidate_used) ++result.soc_accepted_steps;

          // r8 multiplier provenance must match the accepted primal point.
          // Inner QP duals belong to the raw bounded step d.  If equality
          // retraction or SOC materially changes that point, refit multipliers
          // on the ORIGINAL QCQP at the actual accepted trial instead of
          // propagating stale raw-step duals into the next Lagrangian Hessian.
          const double correction_tol=std::max(
              1e-10*(1.0+actual_step_norm),
              10.0*st_.step_tolerance*(1.0+x_norm));
          const bool corrected_primal = soc_candidate_used || correction_norm>correction_tol;
          if(corrected_primal){
            MultiplierEstimate refreshed=estimate_multipliers_legacy(problem,x,working_set,st_);
            if(refreshed.valid){
              qp_multiplier_state=std::move(refreshed);
              qp_multiplier_state_valid=true;
              terminal_requires_qp_refresh=false;
            }else{
              qp_multiplier_state_valid=false;
              terminal_requires_qp_refresh=true;
            }
          }else if(accepted_built_valid &&
             (!inner.equality_duals.empty() || !inner.inequality_duals.empty())){
            const MultiplierEstimate qpm=multipliers_from_inner(
                problem,accepted_built,inner,st_);
            qp_multiplier_state=qpm;
            for(int i=0;i<(int)qp_multiplier_state.inequality.size();++i)
              if(!working_set.count(i)) qp_multiplier_state.inequality[i]=0.0;
            qp_multiplier_state_valid=true;
            terminal_requires_qp_refresh=false;
          }

          // Radius update from the quality of the accepted model prediction.
          if(objective_step && merit_rho<st_.trust_region_eta_shrink){
            const double old_delta=trust_radius;
            trust_radius=std::max(st_.trust_region_min_radius,
                                  st_.trust_region_shrink_factor*trust_radius);
            if(trust_radius<old_delta) ++result.trust_region_contractions;
          }else if(objective_step && merit_rho>=st_.trust_region_eta_expand &&
                   actual_step_norm>=st_.trust_region_boundary_fraction*trust_radius){
            const double old_delta=trust_radius;
            trust_radius=std::min(st_.trust_region_max_radius,
                                  st_.trust_region_expand_factor*trust_radius);
            if(trust_radius>old_delta) ++result.trust_region_expansions;
          }
          result.final_trust_region_radius=trust_radius;
          proximal=std::max(st_.proximal_regularization,0.3*proximal);
          accepted_outer=true;
          result.outer_iterations=outer+1;

          if(st_.verbose && (outer%std::max(1,st_.print_every)==0)){
            std::cout << "[TR-SQP] k="<<outer
                      << " f="<<std::setprecision(12)<<ft
                      << " theta="<<std::scientific<<tf.total_inf
                      << " step="<<last_step
                      << " delta="<<trust_radius
                      << " rho="<<merit_rho
                      << " prox="<<proximal
                      << " ws="<<working_set.size()
                      << " lagrank="<<last_lagrangian_rank
                      << " mult="<<(terminal_multiplier_mode?"qpdual":"legacy")
                      << "\n"<<std::defaultfloat;
          }
        }

        if(!accepted_outer && result.status!=Status::TimeLimit){
          // One final exact certificate before declaring a trust-region stall.
          if(fk_feas.total_inf<=st_.feasibility_tolerance &&
             fresh_certificate_passes(problem,x,st_,result)){
            result.status=Status::Solved;
            result.message="trust-region stall at a fresh original-QCQP KKT point";
          }else{
            result.status=Status::TrustRegionFailure;
            result.message="trust-region SQP failed to obtain an acceptable model-consistent step";
          }
        }
      } else {
      const int max_globalization_retries = 6; // numerical KKT retries happen inside solve_inner; globalization remains algorithmic
      for (int preg = 0; preg < max_globalization_retries && !accepted_outer; ++preg) {
        InnerResult inner;
        Vec d(problem.n,0.0);
        bool inner_ok = false;
        BuiltSubproblem accepted_built;
        bool accepted_built_valid = false;
        int expansions = 0;

        while (expansions <= st_.max_working_set_expansions) {
          const bool linearize_phase2_ineq =
              st_.phase2_linearized_inequalities &&
              qp_multiplier_state_valid &&
              fk_feas.total_inf <= st_.feasibility_tolerance;
          auto built = build_subproblem(problem,x,B,lag_cvx,ineq_cvx,working_set,st_,proximal,
                                        linearize_phase2_ineq);
          seed_inner_dual_warm_start(built,problem,qp_multiplier_state,qp_multiplier_state_valid,st_);
          { auto si=inner_settings(); inner = solve_inner(built.p,si); }
          result.inner_iterations += inner.iterations;
          result.kkt_factorizations += inner.factorizations;
          result.kkt_symbolic_analyses += inner.symbolic_analyses;
          result.last_inner_dual_inf = inner.dual_inf;
          result.last_inner_equality_inf = inner.equality_inf;
          result.last_inner_inequality_inf = inner.inequality_inf;
          result.last_inner_primal_residual_inf = inner.primal_residual_inf;
          result.last_inner_complementarity = inner.complementarity;
          result.last_inner_scaled_model_residual_inf = inner.scaled_model_residual_inf;
          result.last_inner_used_iteration_limit_fallback = inner.used_iteration_limit_fallback;
          absorb_kkt_diagnostics(inner);
          if (inner.used_iteration_limit_fallback) ++result.inner_iteration_limit_fallback_acceptances;
          if (!inner.success || static_cast<int>(inner.y.size()) < problem.n) {
            inner_ok = false;
            break;
          }
          std::copy(inner.y.begin(),inner.y.begin()+problem.n,d.begin());

          Vec xfull = x;
          axpy(1.0,d,xfull);
          xfull = retract_equalities(problem,xfull,st_.equality_retraction_iterations,
                                     st_.equality_retraction_tolerance,
                                     st_.equality_retraction_regularization,
                                     st_.fraction_to_boundary);
          std::vector<int> add;
          for (int i = 0; i < static_cast<int>(problem.inequalities.size()); ++i) {
            if (working_set.count(i)) continue;
            const double add_tol=std::max(st_.working_set_add_tolerance,st_.feasibility_tolerance);
            if (problem.inequalities[i].value(xfull) > add_tol) add.push_back(i);
          }
          if (add.empty()) {
            inner_ok = true;
            accepted_built = std::move(built);
            accepted_built_valid = true;
            break;
          }
          for (int i : add) {
            working_set.insert(i);
            if (!linearize_phase2_ineq)
              curvature.ensure_convex(problem.inequalities[i].hessian,ineq_cvx[i],0.0,st_.max_rank_per_quadratic);
            ++result.working_set_additions;
          }
          ++expansions;
        }

        if (!inner_ok) {
          // rho_x/delta_d are NUMERICAL KKT stabilizers.  solve_inner already
          // retries them on factorization/backward-error failures.  A generic
          // inner failure is not evidence that the Newton matrix needs more
          // numerical regularization, so do not change rho_x here.
          proximal = std::min(st_.proximal_regularization_max,std::max(10.0*proximal,1e-8));
          if (preg == max_globalization_retries-1) {
            result.status = Status::InnerFailure;
            result.message = inner.message.empty() ? "convex QCQP subproblem failed" : inner.message;
          }
          continue;
        }

        const Vec grad_fk = problem.objective.gradient(x);
        const double predicted_decrease = std::max(0.0,-dot(grad_fk,d));
        const bool local_linearized_model =
            st_.phase2_linearized_inequalities && qp_multiplier_state_valid &&
            fk_feas.total_inf <= st_.feasibility_tolerance;
        double alpha = 1.0;
        Vec accepted_x;
        Feasibility accepted_feas;
        double accepted_f = fk;
        bool accepted = false;

        for (int ls = 0; ls < st_.line_search_max_steps; ++ls) {
          Vec trial = x;
          Vec ad = d;
          for (double& v : ad) v *= alpha;
          axpy(1.0,ad,trial);
          if (!finite_state(trial)) {
            ++result.line_search_backtracks;
            alpha *= st_.line_search_contraction;
            continue;
          }

          trial = retract_equalities(problem,trial,st_.equality_retraction_iterations,
                                     st_.equality_retraction_tolerance,
                                     st_.equality_retraction_regularization,
                                     st_.fraction_to_boundary);
          Feasibility tf = feasibility(problem,trial);
          bool soc_attempted_here=false;
          bool soc_candidate_used=false;

          // SOC is a LOCAL correction.  v0.2.13 attempted it only at alpha=1
          // and then never retried it after backtracking; A4X12 consequently
          // produced 999 rejected SOCs and alpha~2^-11 for most outer steps.
          // Try SOC at the current line-search scale once the nonlinear defect
          // is within a controlled multiple of the requested feasible tube.
          if (tf.total_inf > st_.feasibility_tolerance &&
              local_linearized_model && st_.second_order_correction &&
              tf.total_inf <= st_.second_order_correction_trigger_ratio*
                                  st_.feasibility_tolerance) {
            ++result.soc_attempts;
            soc_attempted_here=true;
            auto soc=second_order_correction(problem,x,trial,working_set,st_);
            result.soc_inner_iterations += soc.inner_iterations;
            result.soc_factorizations += soc.factorizations;
            result.inner_iterations += soc.inner_iterations;
            result.kkt_factorizations += soc.factorizations;
            result.kkt_symbolic_analyses += soc.symbolic_analyses;
            if (soc.success) {
              trial=std::move(soc.x);
              tf=soc.feasibility;
              soc_candidate_used=true;
              if (st_.verbose) {
                std::cout << "[SOC] alpha=" << std::scientific << alpha
                          << " corr=" << soc.correction_inf
                          << " theta=" << tf.total_inf << "\n" << std::defaultfloat;
              }
            }
          }

          // Safety net retained from v0.2.12.  The feasibility repair is
          // objective-blind, but it can no longer silently determine
          // acceptance: the repaired point must still pass the ORIGINAL
          // objective filter and Armijo test below.  This prevents a failed SOC
          // from forcing tiny alpha while preserving the robust A4X12 path.
          if (tf.total_inf > st_.feasibility_tolerance) {
            FeasibilityRestoration polisher(st_);
            RestorationResult pr = polisher.polish(problem,trial,24);
            result.restoration_iterations += pr.iterations;
            result.restoration_linear_solves += pr.linear_solves;
            result.restoration_exact_hessian_solves += pr.exact_hessian_solves;
            result.restoration_matrix_free_solves += pr.matrix_free_solves;
            result.restoration_row_space_solves += pr.row_space_solves;
            result.restoration_harmonic_attempts += pr.harmonic_attempts;
            result.restoration_harmonic_acceptances += pr.harmonic_acceptances;
            result.restoration_harmonic_factorizations += pr.harmonic_factorizations;
            result.restoration_tangent_activation_attempts += pr.tangent_activation_attempts;
            result.restoration_tangent_activation_acceptances += pr.tangent_activation_acceptances;
            result.restoration_tangent_activation_components += pr.tangent_activation_components;
            result.restoration_tangent_activation_feasibility = pr.tangent_activation_feasibility;
            result.restoration_tangent_activation_step_inf = pr.tangent_activation_step_inf;
            result.restoration_cg_iterations += pr.cg_iterations;
            result.restoration_exact_workspace_growths += pr.exact_workspace_growths;
            result.restoration_exact_matvecs += pr.exact_matvecs;
            result.restoration_exact_parallel_row_dot_batches +=
                pr.exact_parallel_row_dot_batches;
            result.restoration_exact_parallel_teams += pr.exact_parallel_teams;
            result.restoration_exact_sparse_row_dot_batches +=
                pr.exact_sparse_row_dot_batches;
            result.restoration_exact_sparse_row_dot_verifications +=
                pr.exact_sparse_row_dot_verifications;
            result.restoration_exact_sparse_row_dot_fallbacks +=
                pr.exact_sparse_row_dot_fallbacks;
            result.restoration_line_search_backtracks += pr.line_search_backtracks;
            result.restoration_stagnation_events += pr.stagnation_events;
            result.restoration_minimax_qp_solves += pr.minimax_qp_solves;
            result.restoration_minimax_inner_iterations += pr.minimax_inner_iterations;
            result.restoration_minimax_factorizations += pr.minimax_factorizations;
            result.restoration_minimax_accepted_steps += pr.minimax_accepted_steps;
            result.restoration_minimax_line_search_backtracks += pr.minimax_line_search_backtracks;
            if (pr.final_feasibility < tf.total_inf) {
              trial = std::move(pr.x);
              tf = feasibility(problem,trial);
            }
          }
          const double ft = problem.objective.value(trial);
          const bool nondominated = filter_acceptable(filter,tf.total_inf,ft,st_);
          const bool preserve_feas = tf.total_inf <= st_.feasibility_tolerance;
          const bool filter_objective_progress =
              ft < fk - st_.filter_gamma_f*alpha*std::max(0.0,fk_feas.total_inf);
          const bool armijo = armijo_objective_ok(
              fk,ft,alpha,predicted_decrease,st_);
          const bool objective_progress = filter_objective_progress && armijo;
          const bool feasibility_progress =
              fk_feas.total_inf > st_.feasibility_tolerance &&
              tf.total_inf < (1.0-st_.filter_gamma_theta*alpha)*fk_feas.total_inf;

          if (nondominated && preserve_feas && filter_objective_progress && !armijo)
            ++result.armijo_rejections;

          if (nondominated && preserve_feas && (objective_progress || feasibility_progress)) {
            accepted = true;
            accepted_x = std::move(trial);
            accepted_feas = tf;
            accepted_f = ft;
            if (soc_candidate_used) ++result.soc_accepted_steps;
            break;
          }
          if (soc_attempted_here) ++result.soc_rejected_steps;
          ++result.line_search_backtracks;
          alpha *= st_.line_search_contraction;
          if (alpha < st_.minimum_step_length) break;
        }

        if (!accepted) {
          // A filter/Armijo rejection is a GLOBALIZATION event, not a linear
          // algebra failure.  Never increase numerical rho_x merely because
          // the trial step is rejected; doing so distorted A4X12 despite
          // backward errors near machine precision.
          proximal = std::min(st_.proximal_regularization_max,std::max(10.0*proximal,1e-8));
          if (preg == max_globalization_retries-1) {
            result.status = Status::LineSearchFailure;
            result.message = "filter/Armijo line search failed after numerical KKT and proximal retries";
          }
          continue;
        }

        filter_insert(filter,fk_feas.total_inf,fk);
        Vec actual_step=accepted_x; axpy(-1.0,x,actual_step);
        x = std::move(accepted_x);
        last_step = inf_norm(actual_step);

        // Standard line-search SQP multiplier update.  Original equalities and
        // currently modeled original inequalities occupy the leading dual
        // slots of the convex subproblem; bound duals have zero Hessian and are
        // handled by projected stationarity.
        if (accepted_built_valid && (!inner.equality_duals.empty() || !inner.inequality_duals.empty())) {
          const MultiplierEstimate qpm = multipliers_from_inner(
              problem,accepted_built,inner,st_);
          qp_multiplier_state = qp_multiplier_state_valid
              ? blend_multipliers(qp_multiplier_state,qpm,alpha)
              : qpm;
          // A constraint that is no longer modeled must not survive forever
          // through (1-alpha)*old when the line search takes tiny steps.
          // This was the source of qpdual active sets much larger than the
          // actual working set.
          for(int i=0;i<(int)qp_multiplier_state.inequality.size();++i)
            if(!working_set.count(i)) qp_multiplier_state.inequality[i]=0.0;
          qp_multiplier_state_valid = true;
          terminal_requires_qp_refresh = false;
        }

        if(adaptive_kkt){
          const bool clean=(inner.kkt_regularization_retries==0 &&
                            inner.max_kkt_backward_error<=st_.kkt_backward_error_good);
          if(clean) ++numerical_clean_outer; else numerical_clean_outer=0;
          if(numerical_clean_outer>=std::max(1,st_.kkt_rho_x_clean_outer) &&
             numerical_rho_x>st_.kkt_rho_x_floor){
            const double old=numerical_rho_x;
            numerical_rho_x=std::max(st_.kkt_rho_x_floor,
                                     st_.kkt_rho_x_decay*numerical_rho_x);
            result.final_kkt_rho_x=numerical_rho_x;
            numerical_clean_outer=0;
            if(st_.verbose) std::cout << "[KKT-RHOX] reason=clean_decay rho_x="
                <<std::scientific<<old<<"->"<<numerical_rho_x<<"\n"<<std::defaultfloat;
          }
        }
        proximal = std::max(st_.proximal_regularization,0.3*proximal);
        accepted_outer = true;
        result.outer_iterations = outer+1;

        if (st_.verbose && (outer % std::max(1,st_.print_every) == 0)) {
          std::cout << "[QCQP-SQP] k=" << outer
                    << " f=" << std::setprecision(12) << accepted_f
                    << " theta=" << std::scientific << accepted_feas.total_inf
                    << " eq=" << accepted_feas.equality_inf
                    << " ineq=" << accepted_feas.inequality_inf
                    << " step=" << last_step
                    << " alpha=" << alpha
                    << " prox=" << proximal
                    << " rhox=" << numerical_rho_x
                    << " ws=" << working_set.size()
                    << " lagrank=" << last_lagrangian_rank
                    << " jacrank=" << last_tangent.jacobian_rank
                    << " tangent=" << last_tangent.tangent_dimension
                    << " mult=" << (terminal_multiplier_mode ? "qpdual" : "legacy")
                    << "\n" << std::defaultfloat;
        }
      }

      }

      // Trust-region mode deliberately has no tangential objective fallback.
      // If the bounded SQP cannot make a model-consistent step, return the
      // explicit trust-region status rather than walking along a non-KKT tube.
      if(st_.trust_region_enabled && !accepted_outer) break;

      if (!accepted_outer && fk_feas.total_inf <= st_.feasibility_tolerance &&
          qp_multiplier_state_valid && problem.n <= 64 && st_.terminal_kkt_enabled &&
          st_.linear_backend != "qdldl" && st_.linear_backend != "cholesk") {
        // Small-system emergency postsolve: use the SAME QP-dual KKT equations
        // as the SQP multiplier path.  v0.2.14 retains removal of the old
        // NNLS/legacy reconstruction, whose stationarity measure was not a
        // validated certificate.
        ++result.terminal_kkt_attempts;
        const Vec x_before_terminal=x;
        const auto tk=kkt_polish_qpdual(problem,x,qp_multiplier_state,st_);
        result.terminal_kkt_iterations += tk.iterations;
        result.terminal_kkt_accepted_steps += tk.accepted_steps;
        result.terminal_kkt_active_set_drops += tk.active_set_drops;
        result.terminal_kkt_multiplier_cutoff=tk.multiplier_cutoff;
        last_tangent=tk.tangent;
        if(tk.improved || tk.success){
          x=tk.x; last_stationarity=tk.stationarity;
          Vec tstep=x; axpy(-1.0,x_before_terminal,tstep); last_step=inf_norm(tstep);
          result.outer_iterations=outer+1;
          if(tk.success){
            qp_multiplier_state=tk.multipliers;
            qp_multiplier_state_valid=tk.multipliers.valid;
          }
          if(tk.success && qp_multiplier_state_valid &&
             mapped_certificate_passes(problem,x,qp_multiplier_state,st_,result)){
            final_mapped_certificate_multipliers=qp_multiplier_state;
            final_mapped_certificate_valid=true;
            result.status=Status::Solved;
            result.message="QP-dual emergency postsolve plus mapped KKT certificate converged";
            break;
          }
          terminal_requires_qp_refresh=true;
          continue;
        }
        ++result.terminal_kkt_rejected_steps;
      }

      if (!accepted_outer && !st_.tangential_globalization_enabled) {
        result.status = Status::LineSearchFailure;
        result.message = "SQP/local-KKT step rejected; tangential globalization disabled by --no-tangential";
        break;
      }

      if (!accepted_outer) {
        // Local tangential safeguard: when the convexified QCQP is too
        // conservative to pass the filter, solve the Euclidean projected
        // gradient QP on the same linearized equalities/active inequalities.
        // This does not replace the lazy QCQP step; it is only a globalization
        // fallback and is accepted against the ORIGINAL QCQP.
        auto fb = build_tangential_fallback(problem,x,working_set,st_);
        InnerResult ti = solve_inner(fb.p,st_);
        result.inner_iterations += ti.iterations;
        result.kkt_factorizations += ti.factorizations;
        result.kkt_symbolic_analyses += ti.symbolic_analyses;
        if (ti.success && static_cast<int>(ti.y.size()) >= problem.n) {
          Vec td(ti.y.begin(),ti.y.begin()+problem.n);
          const Vec grad_fk_t = problem.objective.gradient(x);
          const double predicted_decrease_t = std::max(0.0,-dot(grad_fk_t,td));
          double alpha = 1.0;
          for (int ls = 0; ls < st_.line_search_max_steps; ++ls) {
            Vec trial = x;
            Vec ad = td; for (double& v : ad) v *= alpha;
            axpy(1.0,ad,trial);
            trial = retract_equalities(problem,trial,st_.equality_retraction_iterations,
                                       st_.equality_retraction_tolerance,
                                       st_.equality_retraction_regularization,
                                       st_.fraction_to_boundary);
            Feasibility tf = feasibility(problem,trial);
            bool soc_attempted_here=false;
            bool soc_candidate_used=false;
            if(tf.total_inf>st_.feasibility_tolerance && st_.second_order_correction &&
               tf.total_inf <= st_.second_order_correction_trigger_ratio*
                                   st_.feasibility_tolerance){
              ++result.soc_attempts;
              soc_attempted_here=true;
              auto soc=second_order_correction(problem,x,trial,working_set,st_);
              result.soc_inner_iterations += soc.inner_iterations;
              result.soc_factorizations += soc.factorizations;
              result.inner_iterations += soc.inner_iterations;
              result.kkt_factorizations += soc.factorizations;
              result.kkt_symbolic_analyses += soc.symbolic_analyses;
              if(soc.success){
                trial=std::move(soc.x); tf=soc.feasibility; soc_candidate_used=true;
                if(st_.verbose){
                  std::cout << "[SOC-TANGENTIAL] alpha=" << std::scientific << alpha
                            << " corr=" << soc.correction_inf
                            << " theta=" << tf.total_inf << "\n" << std::defaultfloat;
                }
              }
            }
            if (tf.total_inf > st_.feasibility_tolerance) {
              FeasibilityRestoration polisher(st_);
              RestorationResult pr = polisher.polish(problem,trial,32);
              result.restoration_iterations += pr.iterations;
              result.restoration_linear_solves += pr.linear_solves;
              result.restoration_exact_hessian_solves += pr.exact_hessian_solves;
              result.restoration_matrix_free_solves += pr.matrix_free_solves;
              result.restoration_row_space_solves += pr.row_space_solves;
              result.restoration_harmonic_attempts += pr.harmonic_attempts;
              result.restoration_harmonic_acceptances += pr.harmonic_acceptances;
              result.restoration_harmonic_factorizations += pr.harmonic_factorizations;
              result.restoration_tangent_activation_attempts += pr.tangent_activation_attempts;
              result.restoration_tangent_activation_acceptances += pr.tangent_activation_acceptances;
              result.restoration_tangent_activation_components += pr.tangent_activation_components;
              result.restoration_tangent_activation_feasibility = pr.tangent_activation_feasibility;
              result.restoration_tangent_activation_step_inf = pr.tangent_activation_step_inf;
              result.restoration_cg_iterations += pr.cg_iterations;
              result.restoration_exact_workspace_growths += pr.exact_workspace_growths;
              result.restoration_exact_matvecs += pr.exact_matvecs;
              result.restoration_exact_parallel_row_dot_batches +=
                  pr.exact_parallel_row_dot_batches;
              result.restoration_exact_parallel_teams += pr.exact_parallel_teams;
              result.restoration_exact_sparse_row_dot_batches +=
                  pr.exact_sparse_row_dot_batches;
              result.restoration_exact_sparse_row_dot_verifications +=
                  pr.exact_sparse_row_dot_verifications;
              result.restoration_exact_sparse_row_dot_fallbacks +=
                  pr.exact_sparse_row_dot_fallbacks;
              result.restoration_line_search_backtracks += pr.line_search_backtracks;
              result.restoration_stagnation_events += pr.stagnation_events;
              result.restoration_minimax_qp_solves += pr.minimax_qp_solves;
              result.restoration_minimax_inner_iterations += pr.minimax_inner_iterations;
              result.restoration_minimax_factorizations += pr.minimax_factorizations;
              result.restoration_minimax_accepted_steps += pr.minimax_accepted_steps;
              result.restoration_minimax_line_search_backtracks += pr.minimax_line_search_backtracks;
              if (pr.final_feasibility < tf.total_inf) {
                trial=std::move(pr.x); tf=feasibility(problem,trial);
              }
            }
            const double ft = problem.objective.value(trial);
            const bool nondominated=filter_acceptable(filter,tf.total_inf,ft,st_);
            const bool filter_progress=
                ft < fk - st_.filter_gamma_f*alpha*std::max(0.0,fk_feas.total_inf);
            const bool armijo=armijo_objective_ok(
                fk,ft,alpha,predicted_decrease_t,st_);
            if(nondominated && tf.total_inf<=st_.feasibility_tolerance && filter_progress && !armijo)
              ++result.armijo_rejections;
            if (nondominated && tf.total_inf <= st_.feasibility_tolerance &&
                filter_progress && armijo) {
              filter_insert(filter,fk_feas.total_inf,fk);
              Vec actual_step=trial; axpy(-1.0,x,actual_step);
              x = std::move(trial);
              last_step = inf_norm(actual_step);
              last_stationarity = ti.dual_inf;
              result.last_inner_dual_inf = ti.dual_inf;
              result.last_inner_equality_inf = ti.equality_inf;
              result.last_inner_inequality_inf = ti.inequality_inf;
              result.last_inner_primal_residual_inf = ti.primal_residual_inf;
              result.last_inner_complementarity = ti.complementarity;
              proximal = st_.proximal_regularization;
              accepted_outer = true;
              result.outer_iterations = outer+1;
              if(soc_candidate_used) ++result.soc_accepted_steps;
              if (st_.verbose) {
                std::cout << "[TANGENTIAL] k=" << outer << " f=" << std::setprecision(12) << ft
                          << " theta=" << std::scientific << tf.total_inf
                          << " step=" << last_step << " alpha=" << alpha << "\n" << std::defaultfloat;
              }
              break;
            }
            if(soc_attempted_here) ++result.soc_rejected_steps;
            ++result.line_search_backtracks;
            alpha *= st_.line_search_contraction;
            if (alpha < st_.minimum_step_length) break;
          }
        }

        if (!accepted_outer) {
          // If the tangential QP itself is stationary and the original point
          // is feasible, use its KKT residual as the stationarity certificate.
          if (fk_feas.total_inf <= st_.feasibility_tolerance && ti.success &&
              inf_norm(ti.y) <= st_.step_tolerance*(1.0+inf_norm(x)) &&
              ti.dual_inf <= st_.stationarity_tolerance) {
            last_stationarity = ti.dual_inf;
            if (fresh_certificate_passes(problem,x,st_,result)) {
              result.status = Status::Solved;
              result.message = "tangential stationary trigger plus fresh original-QCQP KKT certificate converged";
            }
          }
          break;
        }
      }
      if (outer+1 == st_.max_outer_iterations) {
        result.status = Status::MaxOuterIterations;
        result.message = "outer SQP iteration limit reached";
      }
    }

    if (result.message.empty()) {
      const Feasibility ff = feasibility(problem,x);
      if(ff.total_inf<=10.0*st_.feasibility_tolerance && qp_multiplier_state_valid) {
        last_stationarity=projected_stationarity(problem,x,qp_multiplier_state,
            std::max(1e-10,10.0*st_.feasibility_tolerance));
      } else if(ff.total_inf<=10.0*st_.feasibility_tolerance) {
        last_stationarity=estimate_active_multipliers_legacy(problem,x,st_).stationarity;
      } else {
        const auto mm = estimate_multipliers_legacy(problem,x,working_set,st_);
        last_stationarity = mm.valid
            ? projected_stationarity(problem,x,mm,
                std::max(1e-10,10.0*st_.feasibility_tolerance))
            : std::numeric_limits<double>::infinity();
      }
      if (ff.total_inf <= st_.feasibility_tolerance &&
          fresh_certificate_passes(problem,x,st_,result)) {
        result.status = Status::Solved;
        result.message = "fresh original-QCQP primal-dual KKT certificate satisfied at termination";
      } else {
        result.status = Status::MaxOuterIterations;
        result.message = "terminated without fresh original-QCQP primal-dual KKT certificate";
      }
    }

    // Preserve an exact original-QCQP certificate already validated with the
    // primal-dual multipliers returned by the terminal KKT solve.  Refitting a
    // different multiplier vector here can destroy a valid certificate on a
    // degenerate active set.  Otherwise retain the historical fresh-fit path.
    const Feasibility final_feasibility=feasibility(problem,x);
    if(final_mapped_certificate_valid)
      mapped_termination_certificate(problem,x,final_mapped_certificate_multipliers,st_,result);
    else
      fresh_termination_certificate(problem,x,st_,result);

    // Globalization may stop after the primal point has already converged.
    // Promote the result only when a fresh original-QCQP certificate satisfies
    // the requested tolerances.  This is a certificate rescue, not a tolerance
    // relaxation and not an acceptance of a rejected primal step.
    if(result.status!=Status::Solved &&
       final_feasibility.total_inf<=st_.feasibility_tolerance &&
       result.ipopt_desired_convergence){
      result.status=Status::Solved;
      result.message="terminal high-accuracy multiplier refinement satisfied original-QCQP KKT certificate";
      result.termination_certificate_rescued=true;
    }
    result.x = x;
    result.objective = problem.objective.value(x);
    result.feasibility = final_feasibility;
    result.model_stationarity_inf = last_stationarity;
    result.step_inf = last_step;
    result.curvature_calls = curvature.calls();
    result.constraint_curvature_seconds=curvature.seconds();
    result.objective_rank = last_lagrangian_rank;
    result.active_inequalities = static_cast<int>(working_set.size());
    result.active_jacobian_rank = last_tangent.jacobian_rank;
    result.tangent_dimension = last_tangent.tangent_dimension;
    result.reduced_negative_eigenvalues = last_tangent.reduced_negative_eigenvalues;
    result.reduced_min_eigenvalue = last_tangent.reduced_min_eigenvalue;
    int total_rank = 0;
    for (const auto& c : ineq_cvx) total_rank += c.rank();
    result.inequality_total_rank = total_rank;
    result.final_proximal_regularization = proximal;
    if(adaptive_kkt) result.final_kkt_rho_x=numerical_rho_x;
    result.elapsed_seconds = elapsed();
    return result;
  } catch (const std::exception& e) {
    result.status = Status::NumericalError;
    result.message = e.what();
    result.objective = problem.objective.value(result.x);
    result.feasibility = feasibility(problem,result.x);
    result.elapsed_seconds = elapsed();
    return result;
  }
}

} // namespace lazyipdca::qcqp
