#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>
#include <limits>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem circle_problem() {
  QCQPProblem p;
  p.name = "socp_equality_routes_circle";
  p.n = 2;
  p.x0 = {1.0, 0.0};
  p.lb = {-2.0, -2.0};
  p.ub = {2.0, 2.0};

  p.objective.name = "tangent_target";
  p.objective.constant = 1.0;
  p.objective.linear = {0.0, -2.0};
  p.objective.hessian.n = 2;
  p.objective.hessian.entries = {{1, 1, 2.0}};

  QuadraticFunction equality;
  equality.name = "unit_circle";
  equality.constant = -1.0;
  equality.linear = {0.0, 0.0};
  equality.hessian.n = 2;
  equality.hessian.entries = {{0, 0, 2.0}, {1, 1, 2.0}};
  p.equalities.push_back(equality);

  // Active convex quadratic row.  Its presence guarantees that the conic
  // dispatcher is exercised even for this small equality-route test while
  // retaining a nonzero feasible tangent arc 0 <= x1 <= 0.1.
  QuadraticFunction inequality;
  inequality.name = "short_tangent_arc";
  inequality.constant = 0.0;
  inequality.linear = {0.0, -0.1};
  inequality.hessian.n = 2;
  inequality.hessian.entries = {{1, 1, 2.0}};
  p.inequalities.push_back(inequality);
  return p;
}

Settings settings(const std::string& equality_model) {
  Settings st;
  st.outer_algorithm = "lazy-dca";
  st.inner_backend = "socp";
  st.linear_backend = "native";
  st.qp_backend = "native";
  st.dca_equality_model = equality_model;
  st.verbose = false;
  st.restoration_enabled = false;
  st.second_order_correction = false;
  st.dca_local_corrector = false;
  st.max_outer_iterations = 1;
  st.max_inner_iterations = 300;
  st.feasibility_tolerance = 1e-6;
  st.stationarity_tolerance = 1e-8;
  st.dual_infeasibility_tolerance = 1e-8;
  st.complementarity_tolerance = 1e-8;
  st.inner_tolerance = 1e-10;
  st.line_search_max_steps = 40;
  st.minimum_step_length = 1e-12;
  if (equality_model == "linearized") {
    st.dca_equality_retraction = false;
    st.dca_equality_funnel_auto = false;
  }
  return st;
}

int audit_common(const Result& result, const std::string& route) {
  if (result.effective_outer_algorithm != "lazy-dca" ||
      result.lazy_dca_equality_model != route) {
    std::cerr << "[FAIL] route identity for " << route << "\n";
    return 1;
  }
  if (result.socp_formulation_builds < 1 ||
      result.socp_inequality_rotated_cones < 1 ||
      result.socp_max_affine_equalities < 1 ||
      result.socp_formulation_audit_failures != 0 ||
      result.linear_solver_backend.find("socp/") != 0) {
    std::cerr << "[FAIL] SOCP backend not exercised for " << route << "\n";
    return 2;
  }
  return 0;
}

} // namespace

int main() {
  const QCQPProblem problem = circle_problem();
  const Result linearized = Solver(settings("linearized")).solve(problem);
  if (const int rc = audit_common(linearized, "linearized")) return rc;
  if (linearized.lazy_dca_linearized_equality_steps != 1 ||
      !(linearized.lazy_dca_last_equality_step_alpha > 0.0 &&
        linearized.lazy_dca_last_equality_step_alpha < 1.0) ||
      linearized.lazy_dca_condensed_retraction_attempts != 0) {
    std::cerr << "[FAIL] LIN did not retain its line-search-only semantics\n";
    return 4;
  }

  const Result condensed =
      Solver(settings("condensed-retraction")).solve(problem);
  if (const int rc = audit_common(condensed, "condensed-retraction"))
    return 10 + rc;
  if (condensed.lazy_dca_condensed_retraction_attempts < 1 ||
      condensed.lazy_dca_condensed_retraction_acceptances < 1 ||
      condensed.feasibility.equality_inf > 1e-6) {
    std::cerr << "[FAIL] COND exact quadratic retraction was not retained"
              << " attempts="
              << condensed.lazy_dca_condensed_retraction_attempts
              << " acceptances="
              << condensed.lazy_dca_condensed_retraction_acceptances
              << " equality=" << condensed.feasibility.equality_inf << "\n";
    return 14;
  }

  QCQPProblem equality_only = problem;
  equality_only.name = "socp_objective_cone_with_equality_only";
  equality_only.inequalities.clear();
  equality_only.lb.assign(2, -std::numeric_limits<double>::infinity());
  equality_only.ub.assign(2, std::numeric_limits<double>::infinity());
  const Result objective_cone =
      Solver(settings("condensed-retraction")).solve(equality_only);
  if (objective_cone.socp_formulation_builds < 1 ||
      objective_cone.socp_objective_rotated_cones < 1 ||
      objective_cone.socp_formulation_audit_failures != 0) {
    std::cerr << "[FAIL] equality-only model bypassed the SOCP objective cone\n";
    return 15;
  }

  std::cout << "[PASS] SOCP keeps LIN and COND equality semantics"
            << " lin_alpha=" << linearized.lazy_dca_last_equality_step_alpha
            << " cond_retractions="
            << condensed.lazy_dca_condensed_retraction_acceptances << "\n";
  return 0;
}
