#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem aljazzaf() {
  constexpr int n = 1000;
  constexpr int n1 = 500;
  constexpr double biga = 100.0;
  const double f = (biga*biga - 1.0) / (n - 1.0);
  const double f2 = f / biga;

  QCQPProblem p;
  p.name = "ALJAZZAF_CONDENSED_REGRESSION";
  p.n = n;
  p.x0.assign(n, 0.0);
  p.lb.assign(n, -INFINITY);
  p.ub.assign(n, INFINITY);

  p.objective.name = "objective";
  p.objective.linear.assign(n, 0.0);
  p.objective.hessian.n = n;
  for (int j = 0; j < n; ++j) {
    const double A = biga - j*f2;
    const double shift = j == 0 ? 0.5 : (j < n1 ? -1.0 : 1.0);
    p.objective.constant += A*shift*shift;
    p.objective.linear[j] = -2.0*A*shift;
    if (A != 0.0)
      p.objective.hessian.entries.push_back({j, j, 2.0*A});
  }

  QuadraticFunction h;
  h.name = "H";
  h.constant = 1.0;
  h.linear.assign(n, 0.0);
  h.linear[0] = -1.0;
  h.hessian.n = n;
  for (int j = 1; j < n; ++j) {
    const double B = 1.0 + j*f;
    h.hessian.entries.push_back({j, j, 2.0*B});
    if (j >= n1) {
      h.linear[j] = -2.0*B;
      h.constant += B;
    }
  }
  p.equalities.push_back(std::move(h));
  p.validate();
  return p;
}

}  // namespace

int main() {
  try {
    Settings st;
    st.verbose = std::getenv("LAZYIPDCA_TEST_VERBOSE") != nullptr;
    st.outer_algorithm = "lazy-dca";
    st.inner_backend = "lazyqp";
    st.linear_backend = "native";
    st.max_outer_iterations = 1000;
    st.max_inner_iterations = 1000;
    st.time_limit_seconds = 60.0;
    st.feasibility_tolerance = 1e-6;
    st.stationarity_tolerance = 1e-6;
    st.dual_infeasibility_tolerance = 1e-6;
    st.complementarity_tolerance = 1e-6;
    st.inner_tolerance = 1e-8;
    st.restoration_max_iterations = 600;
    st.restoration_max_restarts = 16;
    st.dca_equality_model = "condensed-retraction";
    st.dca_condensed_tangent_curvature = true;
    st.lazyqp_low_rank_form = "bordered";
    st.dca_condensed_max_tangent_rank = 8;
    st.dca_condensed_scalar_safeguard_updates = 8;
    st.dca_condensed_newton_iterations = 12;
    st.dca_condensed_krylov_iterations = 32;
    st.dca_condensed_krylov_relative_tolerance = 1e-6;
    st.dca_condensed_continuation_attempts = 32;
    // Internal globalization target.  This is deliberately 100x stricter
    // than the final 1e-6 feasibility certificate, but remains resolvable on
    // ALJAZZAF's large-scale quadratic equality with the LDL arithmetic path.
    st.dca_condensed_retraction_tolerance = 1e-8;
    st.dca_local_corrector = true;
    st.dca_iterated_soc = true;
    st.dca_equality_retraction = true;
    st.dca_step_control = true;
    st.inner_dual_warm_start = true;
    st.inner_inequality_dual_warm_start = true;
    st.kkt_adaptive_regularization = true;

    const Result r = Solver(st).solve(aljazzaf());
    if (r.status != Status::Solved)
      throw std::runtime_error(std::string("condensed ALJAZZAF did not converge: ") +
          status_string(r.status) + " / " + r.message);
    if (r.feasibility.total_inf > 1e-6 ||
        r.dual_infeasibility_inf > 1e-6 ||
        !std::isfinite(r.objective) || r.objective > 1e5)
      throw std::runtime_error("condensed ALJAZZAF final certificate or objective regressed");
    if (r.lazy_dca_condensed_accepted_steps < 1)
      throw std::runtime_error("condensed ALJAZZAF did not exercise the condensed route");
    if (r.terminal_kkt_accepted_steps < 1)
      throw std::runtime_error("condensed ALJAZZAF terminal KKT refinement was not exercised");

    std::cout << "[PASS] ALJAZZAF condensed scale-guard regression f=" << r.objective
              << " feas=" << r.feasibility.total_inf
              << " dual=" << r.dual_infeasibility_inf
              << " outer=" << r.outer_iterations
              << " inner=" << r.inner_iterations << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
