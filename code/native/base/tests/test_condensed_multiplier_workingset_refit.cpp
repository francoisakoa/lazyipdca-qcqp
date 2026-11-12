#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_problem(double delta = 5e-4) {
  constexpr double eps = 1e-3;
  const double inf = std::numeric_limits<double>::infinity();

  QCQPProblem p;
  p.name = "condensed_multiplier_workingset_refit";
  p.n = 3;
  // The nonlinear equality is feasible and leaves (x,z) as its tangent
  // coordinates at the initial point.
  p.x0 = {0.0, 1.0, 0.0};
  p.lb = {-10.0, -inf, -2.0};
  p.ub = { 10.0,  inf,  2.0};
  p.objective.name = "correlated_active_normals";
  p.objective.linear = {-2.0, 0.0, -eps};
  p.objective.hessian.n = p.n;

  QuadraticFunction circle;
  circle.name = "unit_circle";
  circle.constant = -1.0;
  circle.linear.assign(p.n, 0.0);
  circle.hessian.n = p.n;
  circle.hessian.entries = {{1, 1, 2.0}, {2, 2, 2.0}};
  p.equalities.push_back(std::move(circle));

  QuadraticFunction first;
  first.name = "first_correlated_normal";
  first.linear = {1.0, 0.0, 0.0};
  first.hessian.n = p.n;
  p.inequalities.push_back(std::move(first));

  QuadraticFunction delayed;
  delayed.name = "delayed_correlated_normal";
  delayed.constant = -delta;
  delayed.linear = {1.0, 0.0, eps};
  delayed.hessian.n = p.n;
  p.inequalities.push_back(std::move(delayed));
  p.validate();
  return p;
}

Settings settings() {
  Settings st;
  st.outer_algorithm = "lazy-dca";
  st.inner_backend = "lazyqp";
  st.linear_backend = "native";
  st.dca_equality_model = "condensed-retraction";
  st.verbose = std::getenv("LAZYIPDCA_TEST_VERBOSE") != nullptr;
  st.restoration_enabled = false;
  st.dca_local_corrector = false;
  st.dca_iterated_soc = false;
  st.second_order_correction = false;
  st.dca_step_control = true;
  st.dca_step_target_absolute = 0.025;
  st.dca_step_target_relative = 0.0;
  st.dca_step_target_maximum = 0.025;
  // The delayed row enters at k=21 on this deterministic trajectory.  A few
  // subsequent iterations are enough to prove that the last valid signed
  // equality multiplier remains usable after the diagnostic NNLS failure.
  st.max_outer_iterations = 24;
  st.max_inner_iterations = 500;
  st.feasibility_tolerance = 1e-7;
  st.stationarity_tolerance = 1e-6;
  st.dual_infeasibility_tolerance = 1e-6;
  st.complementarity_tolerance = 1e-6;
  st.inner_tolerance = 1e-10;
  st.working_set_activation = 1e-4;
  st.working_set_dropping = false;
  st.dca_affine_box_bounded_lazy = true;
  return st;
}

} // namespace

int main() {
  const Settings st = settings();
  const Result r = Solver(st).solve(make_problem());
  if (r.working_set_additions < 1 || r.outer_iterations < 22) {
    std::cerr << "[FAIL] delayed correlated inequality did not exercise the "
                 "post-step working-set change: outer=" << r.outer_iterations
              << " additions=" << r.working_set_additions << "\n";
    return 1;
  }
  if (r.multiplier_refit_attempts <= r.multiplier_refit_successes ||
      r.multiplier_refit_fallback_reuses < 1) {
    std::cerr << "[FAIL] transient nonnegative multiplier-fit failure was not "
                 "continued from the last valid signed equality fit: refit="
              << r.multiplier_refit_attempts << "/"
              << r.multiplier_refit_successes << " fallback="
              << r.multiplier_refit_fallback_reuses << "\n";
    return 2;
  }
  if (r.status != Status::Solved && r.status != Status::MaxOuterIterations) {
    std::cerr << "[FAIL] condensed model aborted after a recoverable multiplier "
                 "refit failure: status=" << status_string(r.status)
              << " message=" << r.message << "\n";
    return 3;
  }
  if (r.status == Status::Solved && !r.ipopt_desired_convergence) {
    std::cerr << "[FAIL] model-only multiplier continuation produced an "
                 "uncertified solved status\n";
    return 4;
  }
  if (r.message.find("requires a valid signed equality-multiplier fit") !=
      std::string::npos) {
    std::cerr << "[FAIL] historical fatal multiplier-fit path remains reachable\n";
    return 5;
  }

  // The continuation is intentionally conditional on a previously certified
  // fit.  Move the delayed correlated row inside the activation threshold so
  // both ill-conditioned normals are present at k=0: no last-valid signed
  // equality multiplier exists, hence the model must still fail closed.
  const Result no_prior = Solver(st).solve(make_problem(5e-5));
  if (no_prior.status != Status::NumericalError ||
      no_prior.message.find("requires a valid signed equality-multiplier fit") ==
          std::string::npos ||
      no_prior.multiplier_refit_successes != 0 ||
      no_prior.multiplier_refit_fallback_reuses != 0) {
    std::cerr << "[FAIL] condensed model did not fail closed without a prior "
                 "valid signed equality fit: status="
              << status_string(no_prior.status) << " message=" << no_prior.message
              << " refit=" << no_prior.multiplier_refit_attempts << "/"
              << no_prior.multiplier_refit_successes << " fallback="
              << no_prior.multiplier_refit_fallback_reuses << "\n";
    return 6;
  }
  std::cout << "[PASS] condensed model continues after transient working-set "
               "multiplier refit failure"
            << " outer=" << r.outer_iterations
            << " additions=" << r.working_set_additions
            << " refit=" << r.multiplier_refit_attempts << "/"
            << r.multiplier_refit_successes
            << " fallback=" << r.multiplier_refit_fallback_reuses
            << " no-prior=fail-closed\n";
  return 0;
}
