#pragma once

#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/qcqp_ipopt_termination.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp {

// A global single-loop flag is not sufficient authority to skip a safeguard.
// The inner kernel must explicitly identify the returned state as its one-step
// provisional result.  The outer solver separately checks and accepts the
// corresponding primal point on the original QCQP before committing it.
inline bool dca_single_loop_provisional_state(
    const Settings& settings, const InnerResult& inner) {
  return settings.dca_single_loop_primal_dual &&
         inner.dca_single_loop_accepted && !inner.success;
}

inline bool dca_single_loop_transport_requested(
    const Settings& settings, const InnerResult& inner) {
  return dca_single_loop_provisional_state(settings, inner) &&
         settings.dca_single_loop_multiplier_transport;
}

inline bool dca_single_loop_raw_retraction_schedule_requested(
    const Settings&, const InnerResult&) {
  // Retained as a source-compatible query for r3p24 callers.  r3p25's
  // literal single-loop contract never authorizes an auxiliary retraction;
  // Solver/CLI validation also reject the legacy switch under the master.
  return false;
}

struct SingleLoopMultiplierGuard {
  bool accepted = false;
  double original_dual_infeasibility =
      std::numeric_limits<double>::infinity();
  double original_complementarity =
      std::numeric_limits<double>::infinity();
};

// Validate a proposed transported multiplier without modifying the live state.
// This deliberately evaluates the original QCQP KKT equations, not the affine
// majorant equations.  It is a finiteness/provenance guard, not a convergence
// certificate: the final independent multiplier fit remains mandatory.
template<class Evaluation=DefaultQCQPEvaluation>
inline SingleLoopMultiplierGuard guard_single_loop_multiplier_transport(
    const QCQPProblem& problem, const Vec& point,
    const QCQPTerminationMultipliers& multipliers,
    const Settings& settings, Evaluation evaluation=Evaluation{}) {
  SingleLoopMultiplierGuard guard;
  if (!multipliers.valid || point.size() != static_cast<std::size_t>(problem.n) ||
      multipliers.equality.size() != problem.equalities.size() ||
      multipliers.inequality.size() != problem.inequalities.size() ||
      multipliers.lower_bound.size() != static_cast<std::size_t>(problem.n) ||
      multipliers.upper_bound.size() != static_cast<std::size_t>(problem.n))
    return guard;

  const auto finite = [](const Vec& values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  const auto finite_nonnegative = [](const Vec& values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
      return std::isfinite(value) && value >= 0.0;
    });
  };
  if (!finite(point) || !finite(multipliers.equality) ||
      !finite_nonnegative(multipliers.inequality) ||
      !finite_nonnegative(multipliers.lower_bound) ||
      !finite_nonnegative(multipliers.upper_bound))
    return guard;

  const QCQPIpoptTermination original = qcqp_ipopt_termination_evaluated(
      problem, point, multipliers, settings, evaluation);
  guard.original_dual_infeasibility = original.stop.dual_infeasibility;
  guard.original_complementarity = original.stop.complementarity;
  guard.accepted = original.stop.finite;
  return guard;
}

}  // namespace lazyipdca::qcqp
