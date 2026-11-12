#include "lazyipdca_qcqp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

// The affine tangent model of y-x^2=0 at the origin permits a long step in x,
// while the exact condensed retraction exposes y=x^2.  Consequently the side
// constraint y<=0.09 forces several outer backtracks.  This is the small,
// deterministic analogue of the apparent-flow limits responsible for the
// rejected (but internally successful) case118 retractions.
QCQPProblem make_side_limited_parabola() {
  const double inf = std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name = "condensed_alpha_predictor_side_limited_parabola";
  p.n = 2;
  p.x0 = {0.0, 0.0};
  p.lb = {-1.0, -inf};
  p.ub = {1.0, inf};

  p.objective.name = "minus_x";
  p.objective.linear = {-1.0, 0.0};
  p.objective.hessian.n = 2;

  QuadraticFunction h;
  h.name = "parabola";
  h.linear = {0.0, 1.0};
  h.hessian.n = 2;
  h.hessian.entries = {{0, 0, -2.0}};  // y-x^2=0
  p.equalities.push_back(std::move(h));

  QuadraticFunction g;
  g.name = "side_cap";
  g.constant = -0.09;
  g.linear = {0.0, 1.0};               // y-0.09<=0
  g.hessian.n = 2;
  p.inequalities.push_back(std::move(g));

  p.validate();
  return p;
}

Settings settings(bool predictor) {
  Settings st;
  st.outer_algorithm = "lazy-dca";
  st.inner_backend = "lazyqp";
  st.linear_backend = "native";
  st.dca_equality_model = "condensed-retraction";
  st.verbose = false;

  st.restoration_enabled = false;
  st.restoration_tangent_activation = false;
  st.dca_step_control = false;
  st.dca_local_corrector = false;
  st.dca_iterated_soc = false;
  st.second_order_correction = false;
  st.terminal_kkt_enabled = false;

  st.max_outer_iterations = 200;
  st.max_inner_iterations = 1000;
  st.line_search_max_steps = 40;
  st.feasibility_tolerance = 1e-7;
  st.stationarity_tolerance = 1e-6;
  st.dual_infeasibility_tolerance = 1e-6;
  st.complementarity_tolerance = 1e-6;
  st.inner_tolerance = 1e-10;
  st.dca_condensed_retraction_tolerance = 1e-11;
  st.dca_condensed_retraction_max_ratio = 4.0;
  st.dca_condensed_newton_iterations = 12;
  st.dca_condensed_krylov_iterations = 12;
  st.dca_condensed_krylov_relative_tolerance = 1e-10;
  st.dca_condensed_continuation_attempts = 24;

  st.dca_condensed_alpha_predictor = predictor;
  // Stay infinitesimally above the validated lower bound.  The historical
  // accepted alpha on the stable active set is 1/2, so this setting exercises
  // a genuine leading skip without replacing the test by an aggressive and
  // trajectory-sensitive step-size policy.
  st.dca_condensed_alpha_growth = 1.000001;
  // The instance is tiny and deterministic.  Keep periodic historical probes
  // far enough apart that the test exercises the remembered-alpha path, while
  // leaving the production default unchanged.
  st.dca_condensed_alpha_full_step_frequency = 64;
  return st;
}

void require_certificate(const Result& r, const Settings& st,
                         const std::string& route) {
  if (r.status != Status::Solved)
    throw std::runtime_error(route + " route did not solve: " + r.message);
  if (r.feasibility.equality_inf > st.feasibility_tolerance ||
      r.feasibility.inequality_inf > st.feasibility_tolerance ||
      r.feasibility.bound_inf > st.feasibility_tolerance)
    throw std::runtime_error(route + " route violated an original constraint");
  if (!std::isfinite(r.dual_infeasibility_inf) ||
      r.dual_infeasibility_inf > st.dual_infeasibility_tolerance ||
      !std::isfinite(r.complementarity_inf) ||
      r.complementarity_inf > st.complementarity_tolerance)
    throw std::runtime_error(route + " route did not return the requested KKT certificate");
}

}  // namespace

int main() {
  try {
    const QCQPProblem p = make_side_limited_parabola();
    const Settings historical_settings = settings(false);
    const Settings predicted_settings = settings(true);
    Settings full_probe_settings = settings(true);
    full_probe_settings.dca_condensed_alpha_full_step_frequency = 1;

    const Result historical = Solver(historical_settings).solve(p);
    const Result predicted = Solver(predicted_settings).solve(p);
    const Result full_probe = Solver(full_probe_settings).solve(p);
    require_certificate(historical, historical_settings, "historical");
    require_certificate(predicted, predicted_settings, "predicted");
    require_certificate(full_probe, full_probe_settings, "periodic-full-probe");

    if (historical.lazy_dca_condensed_alpha_predictor_uses != 0 ||
        historical.lazy_dca_condensed_alpha_predictor_full_step_resets != 0 ||
        historical.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped != 0 ||
        historical.lazy_dca_condensed_alpha_predictor_historical_retries != 0)
      throw std::runtime_error("disabled predictor changed its diagnostic counters");

    if (predicted.lazy_dca_condensed_alpha_predictor_uses <= 0 ||
        predicted.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped <= 0)
      throw std::runtime_error(
          "enabled predictor did not exercise the remembered-alpha path: uses=" +
          std::to_string(predicted.lazy_dca_condensed_alpha_predictor_uses) +
          " skipped=" +
          std::to_string(
              predicted.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped) +
          " retractions=" +
          std::to_string(predicted.lazy_dca_condensed_retraction_attempts));

    // Frequency one forces the historical alpha=1 schedule after every
    // accepted step.  It is a deterministic test of the predictor's periodic
    // safety fallback and must reproduce the predictor-off trajectory.
    if (full_probe.lazy_dca_condensed_alpha_predictor_uses != 0 ||
        full_probe.lazy_dca_condensed_alpha_predictor_full_step_resets <= 0 ||
        full_probe.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped != 0 ||
        full_probe.lazy_dca_condensed_retraction_attempts !=
            historical.lazy_dca_condensed_retraction_attempts ||
        std::abs(full_probe.objective - historical.objective) > 2e-12 ||
        std::abs(full_probe.x[0] - historical.x[0]) > 2e-12 ||
        std::abs(full_probe.x[1] - historical.x[1]) > 2e-12)
      throw std::runtime_error(
          "periodic full-step safeguard did not reproduce the historical schedule");

    // Both routes solve the same original QCQP.  Do not require bitwise path
    // identity: the predictor deliberately changes the order in which trial
    // alphas are examined.  Require the same active solution and certificate.
    const double objective_scale = std::max({1.0, std::abs(historical.objective),
                                             std::abs(predicted.objective)});
    if (std::abs(historical.objective - predicted.objective) >
            5e-6 * objective_scale ||
        std::abs(historical.x[0] - predicted.x[0]) > 5e-6 ||
        std::abs(historical.x[1] - predicted.x[1]) > 5e-6)
      throw std::runtime_error("predictor changed the certified solution materially");

    // The predictor may deliberately fall back to the complete historical
    // grid when the seeded schedule is exhausted.  That safety path can make
    // a tiny synthetic perform one or two extra trials, so attempt counts are
    // diagnostic rather than a brittle unit-test requirement.  The PGLib A/B
    // benchmark is responsible for establishing the performance gain.

    std::cout << "[PASS] safeguarded condensed alpha predictor"
              << " objective=" << predicted.objective
              << " equality=" << predicted.feasibility.equality_inf
              << " inequality=" << predicted.feasibility.inequality_inf
              << " dual=" << predicted.dual_infeasibility_inf
              << " retractions="
              << historical.lazy_dca_condensed_retraction_attempts << "->"
              << predicted.lazy_dca_condensed_retraction_attempts
              << " uses=" << predicted.lazy_dca_condensed_alpha_predictor_uses
              << " skipped="
              << predicted.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped
              << " resets="
              << predicted.lazy_dca_condensed_alpha_predictor_full_step_resets
              << " retries="
              << predicted.lazy_dca_condensed_alpha_predictor_historical_retries
              << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
