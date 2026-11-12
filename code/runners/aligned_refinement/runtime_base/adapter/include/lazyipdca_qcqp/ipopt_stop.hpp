#pragma once
#include "ipopt_termination.hpp"

namespace lazyipdca::qcqp {
// Separate the internally scaled residuals from the unscaled absolute guards.
// This implements Ipopt 3.14.11 OptimalityErrorConvergenceCheck and curr_nlp_error.
struct IpoptStopSnapshot {
  double internal_dual = INFINITY, internal_primal = INFINITY, internal_compl = INFINITY;
  double unscaled_dual = INFINITY, unscaled_primal = INFINITY, unscaled_compl = INFINITY;
  double y_l1 = 0, z_l1 = 0;
  int dual_dimension = 0, compl_dimension = 0, free_variables = 0, equality_dimension = 0;
  double objective_change = INFINITY;
};
struct IpoptStopDecision {
  IpoptTerminationMetrics metrics;
  bool square = false, desired = false, acceptable = false;
};
inline IpoptStopDecision ipopt_stop_decision(const IpoptStopSnapshot& v, const Settings& s) {
  IpoptStopDecision out;
  out.metrics = ipopt_desired_convergence_metrics(v.internal_dual, v.internal_primal,
      v.internal_compl, v.y_l1, v.z_l1, v.dual_dimension, v.compl_dimension, s);
  out.square = v.free_variables == v.equality_dimension;
  const bool finite = out.metrics.finite &&
      std::isfinite(out.metrics.dual_scaling) && std::isfinite(out.metrics.complementarity_scaling) &&
      std::isfinite(v.unscaled_dual) && v.unscaled_dual >= 0 &&
      std::isfinite(v.unscaled_primal) && v.unscaled_primal >= 0 &&
      std::isfinite(v.unscaled_compl) && v.unscaled_compl >= 0 &&
      v.internal_dual >= 0 && v.internal_primal >= 0 && v.internal_compl >= 0 &&
      v.dual_dimension >= 0 && v.compl_dimension >= 0 && v.free_variables >= 0 && v.equality_dimension >= 0;
  if (out.square) out.metrics.overall_nlp_error = v.internal_primal;
  const double dual_cap = out.square ? 1e300 : s.dual_infeasibility_tolerance;
  const double comp_cap = out.square ? 1e300 : s.complementarity_tolerance;
  out.desired = finite &&
      ((out.metrics.overall_nlp_error <= s.stationarity_tolerance &&
        v.unscaled_dual <= dual_cap && v.unscaled_primal <= s.feasibility_tolerance &&
        v.unscaled_compl <= comp_cap) ||
       (v.free_variables == 0 && v.unscaled_primal <= s.feasibility_tolerance));
  out.acceptable = finite &&
      out.metrics.overall_nlp_error <= s.acceptable_tolerance &&
      v.unscaled_dual <= (out.square ? 1e300 : s.acceptable_dual_tolerance) &&
      v.unscaled_primal <= s.acceptable_constraint_tolerance &&
      v.unscaled_compl <= (out.square ? 1e300 : s.acceptable_complementarity_tolerance) &&
      std::isfinite(v.objective_change) &&
      v.objective_change <= s.acceptable_objective_change_tolerance;
  out.metrics.converged = out.desired;
  out.metrics.finite = finite;
  return out;
}
} // namespace lazyipdca::qcqp
