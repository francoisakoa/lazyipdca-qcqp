#include "lazyipdca_qcqp/condensed_retraction.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

QuadraticFunction circle(int i, int j, const std::string& name) {
  QuadraticFunction h;
  h.name = name;
  h.constant = -1.0;
  h.linear.assign(4, 0.0);
  h.hessian.n = 4;
  h.hessian.entries = {{i, i, 2.0}, {j, j, 2.0}};
  return h;
}

QCQPProblem product_of_circles() {
  QCQPProblem p;
  p.name = "adaptive_normal_product_of_circles";
  p.n = 4;
  p.x0 = {1.0, 0.0, 1.0, 0.0};
  p.lb.assign(4, -kInf);
  p.ub.assign(4, kInf);
  p.objective.name = "zero";
  p.objective.linear.assign(4, 0.0);
  p.objective.hessian.n = 4;
  p.equalities.push_back(circle(0, 1, "circle_1"));
  p.equalities.push_back(circle(2, 3, "circle_2"));
  return p;
}

Settings settings() {
  Settings st;
  st.linear_backend = linear_backend_compiled("ldl") ? "ldl" : "native";
  st.verbose = false;
  st.kkt_backward_error_tolerance = 1e-12;
  st.kkt_backward_error_reject = 1e-8;
  st.kkt_iterative_refinement_steps = 3;
  st.minimum_step_length = 1e-13;
  st.dca_condensed_retraction_tolerance = 1e-12;
  st.dca_condensed_retraction_max_ratio = 4.0;
  st.dca_condensed_newton_iterations = 12;
  st.dca_condensed_krylov_iterations = 12;
  st.dca_condensed_krylov_relative_tolerance = 1e-10;
  st.dca_condensed_continuation_attempts = 24;
  st.dca_condensed_adaptive_normal_refinement = true;
  st.dca_condensed_normal_intermediate_backward_error_tolerance = 1e-6;
  st.dca_condensed_normal_intermediate_backward_error_reject = 1e-4;
  return st;
}

bool exact_endpoint(const QCQPProblem& p,
                    const CondensedRetractionResult& result) {
  if (!result.success || result.equality_residual_inf > 5e-11) return false;
  for (const auto& h : p.equalities)
    if (std::abs(h.value(result.x)) > 5e-11) return false;
  return true;
}

}  // namespace

int main() {
  const QCQPProblem p = product_of_circles();
  const Vec step = {0.0, 0.3, 0.0, 0.4};
  std::string message;

  Settings adaptive_settings = settings();
  CondensedQuadraticRetraction adaptive(adaptive_settings);
  if (!adaptive.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] adaptive normal factorization: " << message << "\n";
    return 1;
  }
  const auto adaptive_result = adaptive.retract(step, {}, true);
  if (!exact_endpoint(p, adaptive_result)) {
    std::cerr << "[FAIL] adaptive normal retraction lost the exact equality "
                 "contract: "
              << adaptive_result.message << " residual="
              << adaptive_result.equality_residual_inf << "\n";
    return 2;
  }
  if (adaptive.relaxed_normal_solves() <= 0 ||
      adaptive.strict_normal_solves() <= 0) {
    std::cerr << "[FAIL] adaptive path did not separate intermediate and final "
                 "normal solves: relaxed="
              << adaptive.relaxed_normal_solves() << " strict="
              << adaptive.strict_normal_solves() << "\n";
    return 3;
  }

  // The public right inverse is part of the tangent/certificate geometry and
  // must remain strict even after an adaptive retraction.
  const int relaxed_before = adaptive.relaxed_normal_solves();
  const int strict_before = adaptive.strict_normal_solves();
  Vec correction;
  if (!adaptive.apply_right_inverse({0.25, -0.5}, correction, message)) {
    std::cerr << "[FAIL] public strict right inverse: " << message << "\n";
    return 4;
  }
  if (adaptive.relaxed_normal_solves() != relaxed_before ||
      adaptive.strict_normal_solves() != strict_before + 1) {
    std::cerr << "[FAIL] public right inverse used intermediate accuracy\n";
    return 5;
  }

  // Direct Settings construction bypasses CLI validation intentionally here:
  // a negative intermediate reject threshold forces the accelerated attempt
  // to fail for every finite backward error.  The full continuation must then
  // restart from z=0 under the strict historical policy and still converge.
  Settings retry_settings = settings();
  retry_settings.dca_condensed_normal_intermediate_backward_error_reject = -1.0;
  CondensedQuadraticRetraction retry(retry_settings);
  if (!retry.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] retry normal factorization: " << message << "\n";
    return 6;
  }
  const auto retry_result = retry.retract(step, {}, true);
  if (!exact_endpoint(p, retry_result)) {
    std::cerr << "[FAIL] strict retry did not recover the exact retraction: "
              << retry_result.message << " residual="
              << retry_result.equality_residual_inf << "\n";
    return 7;
  }
  if (retry.strict_retry_retractions() != 1 ||
      retry.relaxed_normal_solves() <= 0 ||
      retry.strict_normal_solves() <= 0) {
    std::cerr << "[FAIL] strict retry counters are inconsistent: retries="
              << retry.strict_retry_retractions() << " relaxed="
              << retry.relaxed_normal_solves() << " strict="
              << retry.strict_normal_solves() << "\n";
    return 8;
  }

  std::cout << "[PASS] adaptive normal refinement relaxed="
            << adaptive.relaxed_normal_solves() << " strict="
            << adaptive.strict_normal_solves() << " above_strict="
            << adaptive.relaxed_accepts_above_strict_tolerance()
            << " retry=" << retry.strict_retry_retractions() << "\n";
  return 0;
}
