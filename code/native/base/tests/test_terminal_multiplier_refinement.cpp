#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>
#include <limits>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p;
  p.name = "terminal_multiplier_regularization_bias";
  p.n = 1;
  p.x0 = {0.0};
  p.lb = {-std::numeric_limits<double>::infinity()};
  p.ub = { std::numeric_limits<double>::infinity()};

  // Exact feasible KKT point at x=0:
  //   grad f + lambda grad h = 1 + lambda*eps = 0.
  // The historical certificate fit adds reg=1e-10 to eps^2 and therefore
  // leaves reg/(eps^2+reg)=1.5e-6, just above the requested 1e-6 tolerance.
  constexpr double eps = 0.00816496580927726;
  constexpr double reg = 1e-10;
  const double biased_residual = reg / (eps*eps + reg);
  if (!(biased_residual > 1e-6 && biased_residual < 2e-6)) {
    std::cerr << "[FAIL] regression geometry does not reproduce bias: "
              << biased_residual << "\n";
    return 1;
  }

  p.objective.name = "objective";
  p.objective.linear = {1.0};
  p.objective.hessian.n = 1;

  QuadraticFunction h;
  h.name = "scaled_equality";
  h.linear = {eps};
  h.hessian.n = 1;
  p.equalities.push_back(h);

  Settings st;
  st.outer_algorithm = "legacy-sqp";
  st.restoration_enabled = false;
  st.verbose = false;
  st.max_outer_iterations = 1;
  st.line_search_max_steps = 0;  // deterministically force the pre-rescue line-search failure
  st.tangential_globalization_enabled = false;
  st.feasibility_tolerance = 1e-6;
  st.stationarity_tolerance = 1e-6;
  st.dual_infeasibility_tolerance = 1e-6;
  st.complementarity_tolerance = 1e-6;
  st.inner_tolerance = 1e-8;
  st.multiplier_regularization = reg;

  const Result r = Solver(st).solve(p);
  if (r.status != Status::Solved || !r.ipopt_desired_convergence) {
    std::cerr << "[FAIL] refined terminal certificate not recovered: status="
              << status_string(r.status) << " dual=" << r.dual_infeasibility_inf
              << " nlp=" << r.scaled_nlp_error << " message=" << r.message << "\n";
    return 2;
  }
  if (r.dual_infeasibility_inf > st.dual_infeasibility_tolerance ||
      r.scaled_nlp_error > st.stationarity_tolerance) {
    std::cerr << "[FAIL] refined certificate misses tolerance: dual="
              << r.dual_infeasibility_inf << " nlp=" << r.scaled_nlp_error << "\n";
    return 3;
  }
  if (r.termination_multiplier_refinement_attempts < 2 ||
      !(r.termination_multiplier_regularization < reg) ||
      !r.termination_certificate_rescued) {
    std::cerr << "[FAIL] refinement/rescue path not exercised: attempts="
              << r.termination_multiplier_refinement_attempts
              << " reg=" << r.termination_multiplier_regularization
              << " rescued=" << r.termination_certificate_rescued << "\n";
    return 4;
  }

  std::cout << "[PASS] biased=" << biased_residual
            << " refined_dual=" << r.dual_infeasibility_inf
            << " reg=" << r.termination_multiplier_regularization
            << " attempts=" << r.termination_multiplier_refinement_attempts
            << "\n";
  return 0;
}
