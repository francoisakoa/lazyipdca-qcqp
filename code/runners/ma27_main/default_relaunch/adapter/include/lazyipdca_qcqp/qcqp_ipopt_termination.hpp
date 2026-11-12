#pragma once

#include "lazyipdca_qcqp/ipopt_termination.hpp"
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace lazyipdca::qcqp {

// Multiplier state in the solver's original-QCQP sign convention:
//   h(x)=0                 : equality is free
//   g(x)<=0                : inequality >= 0
//   lb<=x                  : lower_bound >= 0
//   x<=ub                  : upper_bound >= 0
struct QCQPTerminationMultipliers {
  Vec equality;
  Vec inequality;
  Vec lower_bound;
  Vec upper_bound;
  bool valid = true;
};

struct QCQPIpoptTermination {
  IpoptTerminationMetrics stop;
  double equality_multiplier_inf = 0.0;
  double inequality_multiplier_inf = 0.0;
  double bound_multiplier_inf = 0.0;

  // Effective Ipopt-internal multiplier-vector dimensions under the default
  // fixed_variable_treatment=make_parameter and our g(x)<=0 representation.
  int y_c_dimension = 0;
  int y_d_dimension = 0;
  int z_L_dimension = 0;
  int z_U_dimension = 0;
  int v_L_dimension = 0;
  int v_U_dimension = 0;
};

struct DefaultQCQPEvaluation {
  double value(const QuadraticFunction& q,const Vec& x) { return q.value(x); }
  void gradient_axpy(const QuadraticFunction& q,const Vec& x,double scale,Vec& y) {
    axpy(scale,q.gradient(x),y);
  }
  double violation(const QCQPProblem& p,const Vec& x) { return feasibility(p,x).total_inf; }
};

// Sparse original-coefficient audit. Hessian entries accumulate in their
// original order; only touched coordinates are reset, with one workspace.
struct SparseQCQPEvaluation : DefaultQCQPEvaluation {
  Vec work; std::vector<unsigned char> seen; std::vector<int> touched;
  void gradient_axpy(const QuadraticFunction& q,const Vec& x,double scale,Vec& y) {
    if(work.size()!=x.size()){work.assign(x.size(),0.0);seen.assign(x.size(),0);}
    auto add=[&](int j,double v){if(!seen[j]){seen[j]=1;touched.push_back(j);}work[j]+=v;};
    for(const auto& e:q.hessian.entries){add(e.i,e.v*x[e.j]);if(e.i!=e.j)add(e.j,e.v*x[e.i]);}
    for(auto e:q.signed_square)add(e.first,2*e.second*std::abs(x[e.first]));
    q.for_each_linear_nonzero([&](int j,double v){add(j,v);});
    for(int j:touched){y[j]+=scale*work[j];work[j]=0.0;seen[j]=0;}
    touched.clear();
  }
};

template<class Evaluation>
inline QCQPIpoptTermination qcqp_ipopt_termination_evaluated(
    const QCQPProblem& p,
    const Vec& x,
    const QCQPTerminationMultipliers& mult,
    const Settings& st, Evaluation& evaluation) {
  QCQPIpoptTermination out;
  if (!mult.valid || static_cast<int>(x.size()) != p.n ||
      mult.equality.size() != p.equalities.size() ||
      mult.inequality.size() != p.inequalities.size()) {
    return out;
  }

  // Determine which original variable-bound multiplier vectors actually exist
  // in Ipopt after make_parameter removes fixed variables.  Each finite bound
  // side is a separate vector entry; a two-sided variable contributes two.
  int finite_lower_nonfixed = 0;
  int finite_upper_nonfixed = 0;
  for (int j = 0; j < p.n; ++j) {
    const bool fixed = std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
                       std::abs(p.ub[j] - p.lb[j]) <= st.fixed_bound_tolerance;
    if (fixed) continue;
    if (std::isfinite(p.lb[j])) ++finite_lower_nonfixed;
    if (std::isfinite(p.ub[j])) ++finite_upper_nonfixed;
  }

  if ((finite_lower_nonfixed > 0 &&
       mult.lower_bound.size() != static_cast<std::size_t>(p.n)) ||
      (finite_upper_nonfixed > 0 &&
       mult.upper_bound.size() != static_cast<std::size_t>(p.n))) {
    return out;
  }

  out.y_c_dimension = static_cast<int>(p.equalities.size());
  out.y_d_dimension = static_cast<int>(p.inequalities.size());
  out.z_L_dimension = finite_lower_nonfixed;
  out.z_U_dimension = finite_upper_nonfixed;
  out.v_L_dimension = 0; // every native QCQP inequality is -inf < g(x) <= 0
  out.v_U_dimension = static_cast<int>(p.inequalities.size());

  Vec r = p.objective.gradient(x);
  double y_l1 = 0.0;
  double z_l1 = 0.0;

  for (int j = 0; j < static_cast<int>(p.equalities.size()); ++j) {
    const double y = mult.equality[j];
    if (!std::isfinite(y)) return QCQPIpoptTermination{};
    y_l1 += std::abs(y);
    out.equality_multiplier_inf =
        std::max(out.equality_multiplier_inf, std::abs(y));
    if (y != 0.0) evaluation.gradient_axpy(p.equalities[j],x,y,r);
  }

  double comp_inf = 0.0;
  for (int i = 0; i < static_cast<int>(p.inequalities.size()); ++i) {
    if (!std::isfinite(mult.inequality[i])) return QCQPIpoptTermination{};
    const double mu = std::max(0.0, mult.inequality[i]);
    y_l1 += mu; // y_d
    z_l1 += mu; // v_U; grad_L(s)=v_U-y_d=0 in the canonical slack form
    out.inequality_multiplier_inf =
        std::max(out.inequality_multiplier_inf, mu);
    if (mu != 0.0) evaluation.gradient_axpy(p.inequalities[i],x,mu,r);
    comp_inf =
        std::max(comp_inf, std::abs(mu * evaluation.value(p.inequalities[i],x)));
  }

  const bool have_lb =
      mult.lower_bound.size() == static_cast<std::size_t>(p.n);
  const bool have_ub =
      mult.upper_bound.size() == static_cast<std::size_t>(p.n);

  for (int j = 0; j < p.n; ++j) {
    const bool fixed = std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
                       std::abs(p.ub[j] - p.lb[j]) <= st.fixed_bound_tolerance;
    if (fixed) {
      // Default Ipopt behavior: make_parameter removes this primal component
      // and its internal bound-multiplier entries from the convergence test.
      r[j] = 0.0;
      continue;
    }

    const double zl = have_lb && std::isfinite(p.lb[j])
                          ? std::max(0.0, mult.lower_bound[j])
                          : 0.0;
    const double zu = have_ub && std::isfinite(p.ub[j])
                          ? std::max(0.0, mult.upper_bound[j])
                          : 0.0;
    if (!std::isfinite(zl) || !std::isfinite(zu))
      return QCQPIpoptTermination{};

    z_l1 += zl + zu;
    out.bound_multiplier_inf =
        std::max({out.bound_multiplier_inf, zl, zu});
    r[j] -= zl;
    r[j] += zu;

    if (std::isfinite(p.lb[j]))
      comp_inf = std::max(comp_inf, std::abs(zl * (x[j] - p.lb[j])));
    if (std::isfinite(p.ub[j]))
      comp_inf = std::max(comp_inf, std::abs(zu * (p.ub[j] - x[j])));
  }

  const int comp_dim = out.z_L_dimension + out.z_U_dimension +
                       out.v_L_dimension + out.v_U_dimension;
  const int dual_dim = out.y_c_dimension + out.y_d_dimension + comp_dim;
  const double dual = inf_norm(r);
  const double constr = evaluation.violation(p,x);
  out.stop = ipopt_desired_convergence_metrics(
      dual, constr, comp_inf, y_l1, z_l1, dual_dim, comp_dim, st);
  return out;
}

inline QCQPIpoptTermination qcqp_ipopt_termination(
    const QCQPProblem& p,const Vec& x,const QCQPTerminationMultipliers& mult,
    const Settings& st) {
  DefaultQCQPEvaluation evaluation;
  return qcqp_ipopt_termination_evaluated(p,x,mult,st,evaluation);
}

} // namespace lazyipdca::qcqp
