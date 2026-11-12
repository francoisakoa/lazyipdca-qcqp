#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

using namespace lazyipdca::qcqp;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class T, class = void>
struct has_single_loop_dual_bootstrap_authority : std::false_type {};

template <class T>
struct has_single_loop_dual_bootstrap_authority<
    T, std::void_t<decltype(
           std::declval<T&>().allow_dca_single_loop_dual_bootstrap)>>
    : std::true_type {};

template <class T>
void authorize_single_loop_dual_bootstrap(T& p) {
  if constexpr (has_single_loop_dual_bootstrap_authority<T>::value) {
    p.allow_dca_single_loop_dual_bootstrap = true;
  }
}

// Minimal, standalone first-majorant bootstrap with three ingredients:
//
//   * the primal point is feasible and the transported free multipliers are 0;
//   * a mapped zero inequality multiplier is repaired by ZERO_DUAL;
//   * one exact affine variable/equality pair is privately eliminated.
//
// The objective gradient in x0 is corrected only by the free equality
// multiplier.  The ZERO_DUAL row is weakly active at the initial point.  In
// r3p25 the Mehrotra corrector creates a large trial complementarity, so the
// common backtracking scalar is about 2^-12 and also throttles the unconstrained
// equality multiplier.  The final dual residual is consequently almost the
// same as the initial residual and the 0.9 contraction guard rejects every
// proximal retry.
ConvexSubproblem cold_zero_dual_affine_majorant() {
  ConvexSubproblem p;
  p.nvar = 3;
  p.original_dimension = 3;
  p.objective.linear = {10000.0, 1000.0, 0.0};
  p.objective.base_hessian.n = 3;
  p.objective.base_hessian.entries = {
      {0, 0, 1.0}, {1, 1, 1.0}, {2, 2, 1.0}};

  p.equalities.nvar = 3;
  // Exact lift row, eliminated by the private LazyQP wrapper.
  p.equalities.rows.push_back({{{2, 1.0}}});
  p.equalities.rhs.push_back(0.0);
  // Retained equality.  Its free multiplier must remove grad_x0 f = 10000.
  p.equalities.rows.push_back({{{0, 1.0}}});
  p.equalities.rhs.push_back(0.0);
  p.exact_affine_eliminations = {{0, 2}};

  ConvexQuadraticModel side;
  side.constant = -1.0e-2;
  side.linear = {0.0, 1.0, 0.0};
  side.base_hessian.n = 3;
  p.inequalities.push_back(std::move(side));

  p.prefer_feasible_centered_start = true;
  p.allow_dca_single_loop = true;
  // On corrected builds, grant the one-shot authority normally supplied by
  // the outer solver.  The detection wrapper deliberately keeps this fixture
  // source-compatible with released r3p25, where the field does not exist, so
  // the old implementation fails by behavior rather than by compilation.
  authorize_single_loop_dual_bootstrap(p);
  p.has_primal_warm_start = true;
  p.initial_primal = {0.0, 0.0, 0.0};
  p.has_dual_warm_start = true;
  p.dual_warm_start_is_mapped = true;
  p.initial_equality_duals = {0.0, 0.0};
  p.initial_inequality_duals = {0.0};
  p.has_inequality_slack_warm_start = true;
  p.initial_inequality_slacks = {1.0e-2};
  return p;
}

Settings settings() {
  Settings st;
  st.linear_backend = "native";
  st.max_inner_iterations = 100;
  st.inner_tolerance = 1.0e-12;
  st.qp_model_solved_tol = 1.0e-14;
  st.inner_iteration_limit_mu_fallback = 1.0e-14;
  st.inner_dual_warm_start = true;
  st.inner_inequality_dual_warm_start = true;
  st.lazyqp_warm_start_guard_factor = 2.0;
  st.lazyqp_zero_dual_continuation = true;
  st.lazyqp_zero_dual_mu_cap = 1.0e-4;
  st.lazyqp_flow_lift_elimination = true;
  st.lazyqp_inexact_forcing = false;
  st.dca_single_loop_primal_dual = true;
  st.dca_single_loop_inner_steps = 1;
  st.dca_single_loop_forcing_eta = 0.9;
  st.qp_mehrotra = true;
  st.verbose = false;
  return st;
}

} // namespace

int main() {
  try {
    const Settings st = settings();
    const ConvexSubproblem p = cold_zero_dual_affine_majorant();
    const InnerResult result = ConvexQCQPLazyQPKernel(st).solve(p);

    require(result.dca_single_loop_attempted,
            "fixture did not reach the single-loop predictor-corrector");
    require(result.dca_single_loop_steps == 1 &&
                result.dca_single_loop_factorizations == 1 &&
                result.factorizations == 1,
            "fixture no longer performs exactly one factorization");
    require(result.lazyqp_zero_dual_continuation_attempts == 1 &&
                result.lazyqp_zero_dual_continuation_mapped_attempts == 1 &&
                result.lazyqp_zero_dual_continuation_acceptances == 1 &&
                result.lazyqp_zero_dual_continuation_fallbacks == 0 &&
                result.lazyqp_zero_dual_continuation_repairs == 1,
            "fixture did not exercise the mapped ZERO_DUAL bootstrap");
    require(result.lazyqp_affine_elimination_attempts == 1 &&
                result.lazyqp_full_kkt_dimension == 4 &&
                result.y.size() == 3 && result.equality_duals.size() == 2,
            "fixture did not exercise the exact affine wrapper");
    require(result.lazyqp_inexact_forcing_model_evaluations == 0,
            "disabled forcing evaluated a model in the bootstrap fixture");

    // The corrected route deliberately keeps the *cold* composite residual as
    // the forcing reference while returning the fitted state as the
    // transactional snapshot.  Reconstruct the cold stationarity norm from
    // this fixture instead of mistaking the post-fit snapshot telemetry for
    // the forcing denominator.
    const Vec cold_gradient = p.objective.gradient(p.initial_primal);
    double cold_reference = 0.0;
    for (double value : cold_gradient)
      cold_reference = std::max(cold_reference, std::abs(value));
    const double initial = result.dca_single_loop_initial_residual;
    const double final = result.dca_single_loop_final_residual;
    const double target = st.dca_single_loop_forcing_eta * cold_reference;
    require(std::isfinite(initial) && std::isfinite(final) &&
                std::isfinite(cold_reference) && cold_reference > 0.0,
            "single-loop residual telemetry is invalid");

    // This is the regression requirement.  Released r3p25 fails here with
    // initial=10000, final indistinguishable from 10000 at the printed
    // precision and target=9000: one expensive KKT factorization makes no
    // useful change to the composite/dual residual.
    require(final <= target,
            "cold/ZERO_DUAL/affine bootstrap did not contract: initial=" +
                std::to_string(initial) + " final=" +
                std::to_string(final) + " target=" +
                std::to_string(target));
    require(result.dca_single_loop_accepted && !result.success &&
                !result.dca_single_loop_retryable &&
                result.lazyqp_affine_elimination_acceptances == 1,
            "contracting bootstrap state was not returned provisionally");

    std::cout << "[PASS] cold/ZERO_DUAL/affine single-loop contraction"
              << " residual=" << cold_reference << "->" << final
              << " returned-origin=" << initial
              << " target=" << target << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
