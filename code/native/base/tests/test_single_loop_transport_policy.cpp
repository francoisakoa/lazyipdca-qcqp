#include "lazyipdca_qcqp/single_loop_policy.hpp"

#include <cmath>
#include <iostream>
#include <limits>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_problem() {
  QCQPProblem problem;
  problem.name = "single_loop_transport_guard";
  problem.n = 1;
  problem.x0 = {0.0};
  problem.lb = {-2.0};
  problem.ub = {2.0};
  problem.objective.linear = {0.0};
  problem.objective.hessian.n = 1;

  QuadraticFunction equality;
  equality.name = "x_equals_zero";
  equality.linear = {1.0};
  equality.hessian.n = 1;
  problem.equalities.push_back(equality);

  QuadraticFunction inequality;
  inequality.name = "x_le_one";
  inequality.constant = -1.0;
  inequality.linear = {1.0};
  inequality.hessian.n = 1;
  problem.inequalities.push_back(inequality);
  problem.validate();
  return problem;
}

}  // namespace

int main() {
  Settings settings;
  InnerResult inner;
  inner.dca_single_loop_accepted = true;

  if (dca_single_loop_transport_requested(settings, inner) ||
      dca_single_loop_raw_retraction_schedule_requested(settings, inner)) {
    std::cerr << "[FAIL] subordinate policy escaped the default-off master\n";
    return 1;
  }

  settings.dca_single_loop_primal_dual = true;
  inner.success = true;
  if (dca_single_loop_transport_requested(settings, inner) ||
      dca_single_loop_raw_retraction_schedule_requested(settings, inner)) {
    std::cerr << "[FAIL] successful strict result with a stale marker became provisional\n";
    return 2;
  }
  inner.dca_single_loop_accepted = false;
  inner.success = true;  // strict bootstrap: usable, but not provisional
  if (dca_single_loop_transport_requested(settings, inner) ||
      dca_single_loop_raw_retraction_schedule_requested(settings, inner)) {
    std::cerr << "[FAIL] global flag authorized a strict bootstrap result\n";
    return 3;
  }

  inner.success = false;
  inner.dca_single_loop_accepted = true;
  if (!dca_single_loop_transport_requested(settings, inner) ||
      dca_single_loop_raw_retraction_schedule_requested(settings, inner)) {
    std::cerr << "[FAIL] provisional state activated forbidden retraction\n";
    return 4;
  }
  settings.dca_single_loop_multiplier_transport = false;
  if (dca_single_loop_transport_requested(settings, inner) ||
      dca_single_loop_raw_retraction_schedule_requested(settings, inner)) {
    std::cerr << "[FAIL] multiplier sub-switch affected wrong policy\n";
    return 5;
  }
  settings.dca_single_loop_multiplier_transport = true;
  settings.dca_single_loop_retraction_fallback_only = true;
  if (!dca_single_loop_transport_requested(settings, inner) ||
      dca_single_loop_raw_retraction_schedule_requested(settings, inner)) {
    std::cerr << "[FAIL] legacy switch authorized a forbidden retraction\n";
    return 6;
  }
  settings.dca_single_loop_retraction_fallback_only = false;

  const QCQPProblem problem = make_problem();
  QCQPTerminationMultipliers multipliers;
  multipliers.equality = {0.0};
  multipliers.inequality = {0.0};
  multipliers.lower_bound = {0.0};
  multipliers.upper_bound = {0.0};
  const SingleLoopMultiplierGuard valid =
      guard_single_loop_multiplier_transport(
          problem, problem.x0, multipliers, settings);
  if (!valid.accepted || !std::isfinite(valid.original_dual_infeasibility) ||
      !std::isfinite(valid.original_complementarity)) {
    std::cerr << "[FAIL] finite original-QCQP state failed transport guard\n";
    return 7;
  }

  multipliers.inequality[0] = -1.0;
  if (guard_single_loop_multiplier_transport(
          problem, problem.x0, multipliers, settings).accepted) {
    std::cerr << "[FAIL] negative inequality multiplier passed guard\n";
    return 8;
  }
  multipliers.inequality[0] = 0.0;
  multipliers.equality[0] = std::numeric_limits<double>::quiet_NaN();
  if (guard_single_loop_multiplier_transport(
          problem, problem.x0, multipliers, settings).accepted) {
    std::cerr << "[FAIL] non-finite equality multiplier passed guard\n";
    return 9;
  }
  multipliers.equality[0] = 0.0;
  multipliers.upper_bound.clear();
  if (guard_single_loop_multiplier_transport(
          problem, problem.x0, multipliers, settings).accepted) {
    std::cerr << "[FAIL] incomplete bound state passed guard\n";
    return 10;
  }

  std::cout << "[PASS] single-loop skip policy and transactional original-QCQP "
               "multiplier guard\n";
  return 0;
}
