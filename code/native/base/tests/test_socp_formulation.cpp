#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/inner_socp.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

ConvexSubproblem problem() {
  ConvexSubproblem p;
  p.nvar = 1;
  p.original_dimension = 1;
  p.equalities.nvar = 1;
  p.objective.constant = 0.5;
  p.objective.linear = {-1.0};
  p.objective.base_hessian.n = 1;
  p.objective.base_hessian.entries = {{0, 0, 1.0}};

  ConvexQuadraticModel ball;
  ball.constant = -1.0;
  ball.linear = {0.0};
  ball.base_hessian.n = 1;
  ball.base_hessian.entries = {{0, 0, 2.0}};
  p.inequalities.push_back(ball);
  p.prefer_feasible_centered_start = true;
  return p;
}

} // namespace

int main() {
  const ConvexSubproblem p = problem();
  CondensedSOCPFormulation formulation;
  std::string message;
  if (!build_condensed_socp_formulation(p, formulation, message)) {
    std::cerr << "[FAIL] conic formulation: " << message << "\n";
    return 1;
  }
  if (formulation.objective_rotated_cones != 1 ||
      formulation.inequality_rotated_cones != 1 ||
      formulation.linear_nonnegative_cones != 0 ||
      formulation.explicit_lift_cone_dimension != 6) {
    std::cerr << "[FAIL] wrong cone inventory\n";
    return 2;
  }

  const Vec y = {0.5};
  const double inequality_det = condensed_socp_rotated_determinant(
      p.inequalities.front(), y, 0.0, false);
  const double objective_det = condensed_socp_rotated_determinant(
      p.objective, y, 0.0, true);
  if (std::abs(inequality_det - 1.5) > 1e-14 ||
      std::abs(objective_det + 0.25) > 1e-14) {
    std::cerr << "[FAIL] rotated-cone determinant identity: inequality="
              << inequality_det << " objective=" << objective_det << "\n";
    return 3;
  }

  Settings settings;
  settings.verbose = false;
  settings.inner_tolerance = 1e-9;
  settings.max_inner_iterations = 200;
  ConvexQCQPSOCPKernel socp(settings);
  const InnerResult result = socp.solve(p);
  if (!result.success || result.y.size() != 1 ||
      std::abs(result.y.front() - 1.0) > 5e-3) {
    std::cerr << "[FAIL] condensed SOCP solve: " << result.message << "\n";
    return 4;
  }
  if (result.socp_formulation_attempts != 1 ||
      result.socp_formulation_builds != 1 ||
      result.socp_formulation_rejections != 0 ||
      result.socp_formulation_audit_failures != 0 ||
      result.socp_inequality_rotated_cones != 1 ||
      result.socp_max_rotated_cone_violation > 1e-6 ||
      result.message.find("exact Schur-condensed SOCP") == std::string::npos) {
    std::cerr << "[FAIL] SOCP audit counters/message\n";
    return 5;
  }

  ConvexSubproblem invalid = p;
  invalid.inequalities.front().low_rank.push_back({-1.0, {1.0}});
  CondensedSOCPFormulation rejected;
  if (build_condensed_socp_formulation(invalid, rejected, message)) {
    std::cerr << "[FAIL] negative atom accepted as a cone Gram term\n";
    return 6;
  }

  std::cout << "[PASS] exact rotated-SOCP identity and condensed solve"
            << " y=" << result.y.front()
            << " cone_violation=" << result.socp_max_rotated_cone_violation
            << "\n";
  return 0;
}
