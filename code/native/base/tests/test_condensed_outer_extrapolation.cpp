#include "lazyipdca_qcqp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_problem() {
  constexpr double eps = 1e-3;
  const double inf = std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name = "condensed_outer_extrapolation_guard";
  p.n = 3;
  p.x0 = {0.0, 1.0, 0.0};
  p.lb = {-10.0, -inf, -2.0};
  // The z cap makes an over-relaxed endpoint eventually cross an original
  // bound while the historical alpha=1 endpoint remains admissible.  This
  // deterministically exercises the fail-closed fallback.
  p.ub = {10.0, inf, 0.2};

  p.objective.name = "correlated_active_normals";
  p.objective.linear = {-2.0, 0.0, -eps};
  p.objective.hessian.n = p.n;

  QuadraticFunction h;
  h.name = "unit_circle";
  h.constant = -1.0;
  h.linear.assign(p.n, 0.0);
  h.hessian.n = p.n;
  h.hessian.entries = {{1, 1, 2.0}, {2, 2, 2.0}};
  p.equalities.push_back(std::move(h));

  QuadraticFunction g;
  g.name = "first_correlated_normal";
  g.linear = {1.0, 0.0, 0.0};
  g.hessian.n = p.n;
  p.inequalities.push_back(std::move(g));

  QuadraticFunction delayed;
  delayed.name = "delayed_correlated_normal";
  delayed.constant = -5e-4;
  delayed.linear = {1.0, 0.0, eps};
  delayed.hessian.n = p.n;
  p.inequalities.push_back(std::move(delayed));
  p.validate();
  return p;
}

Settings settings(bool extrapolation) {
  Settings st;
  st.outer_algorithm = "lazy-dca";
  st.inner_backend = "lazyqp";
  st.linear_backend = "native";
  st.dca_equality_model = "condensed-retraction";
  st.verbose = false;
  st.restoration_enabled = false;
  st.restoration_tangent_activation = false;
  st.dca_step_control = true;
  st.dca_step_target_absolute = 0.025;
  st.dca_step_target_relative = 0.0;
  st.dca_step_target_maximum = 0.025;
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
  st.working_set_activation = 1e-4;
  st.working_set_dropping = false;
  st.dca_affine_box_bounded_lazy = true;
  st.dca_condensed_retraction_tolerance = 1e-11;
  st.dca_condensed_retraction_max_ratio = 4.0;
  st.dca_condensed_newton_iterations = 12;
  st.dca_condensed_krylov_iterations = 12;
  st.dca_condensed_krylov_relative_tolerance = 1e-10;
  st.dca_condensed_continuation_attempts = 24;
  st.dca_condensed_alpha_predictor = false;
  st.dca_condensed_outer_extrapolation = extrapolation;
  st.dca_condensed_outer_extrapolation_alpha = 2.0;
  st.dca_condensed_outer_extrapolation_min_full_steps = 1;
  return st;
}

void require_certificate(const Result& r, const Settings& st,
                         const std::string& route) {
  if (r.status != Status::Solved)
    throw std::runtime_error(route + " did not solve: " + r.message);
  if (r.feasibility.total_inf > st.feasibility_tolerance ||
      !std::isfinite(r.dual_infeasibility_inf) ||
      r.dual_infeasibility_inf > st.dual_infeasibility_tolerance ||
      !std::isfinite(r.complementarity_inf) ||
      r.complementarity_inf > st.complementarity_tolerance)
    throw std::runtime_error(route + " did not certify the original QCQP");
}

}  // namespace

int main() {
  try {
    const QCQPProblem p = make_problem();
    const Settings historical_settings = settings(false);
    const Settings guarded_settings = settings(true);
    Settings stable_streak_settings = settings(true);
    // A separate run requires a real stable streak. The problem changes its
    // active working set during the solve, so a probe must not inherit history
    // across that change.
    stable_streak_settings.dca_condensed_outer_extrapolation_min_full_steps = 3;
    const Result historical = Solver(historical_settings).solve(p);
    const Result guarded = Solver(guarded_settings).solve(p);
    const Result stable_streak = Solver(stable_streak_settings).solve(p);
    require_certificate(historical, historical_settings, "historical route");
    require_certificate(guarded, guarded_settings, "guarded route");
    require_certificate(stable_streak, stable_streak_settings,
                        "stable-streak route");

    if (historical.lazy_dca_condensed_outer_extrapolation_attempts != 0 ||
        historical.lazy_dca_condensed_outer_extrapolation_acceptances != 0 ||
        historical.lazy_dca_condensed_outer_extrapolation_historical_fallbacks != 0)
      throw std::runtime_error("default-off route changed extrapolation counters");
    if (guarded.lazy_dca_condensed_outer_extrapolation_attempts <= 0)
      throw std::runtime_error("enabled route never exercised a guarded probe");
    if (stable_streak.lazy_dca_condensed_outer_extrapolation_attempts <= 0 ||
        stable_streak.lazy_dca_condensed_outer_extrapolation_full_step_streak_max < 3)
      throw std::runtime_error(
          "probe ran without the configured stable full-step streak");
    if (guarded.lazy_dca_condensed_outer_extrapolation_historical_fallbacks <= 0)
      throw std::runtime_error("enabled route never exercised the historical fallback");
    if (guarded.lazy_dca_condensed_outer_extrapolation_attempts !=
        guarded.lazy_dca_condensed_outer_extrapolation_acceptances +
        guarded.lazy_dca_condensed_outer_extrapolation_historical_fallbacks)
      throw std::runtime_error("a probe neither accepted nor used the historical fallback");

    const double scale = std::max({1.0, std::abs(historical.objective),
                                   std::abs(guarded.objective)});
    if (std::abs(historical.objective - guarded.objective) > 5e-6 * scale)
      throw std::runtime_error("guarded probe changed the certified solution materially");

    std::cout << "[PASS] guarded condensed outer extrapolation"
              << " try/accept/fallback="
              << guarded.lazy_dca_condensed_outer_extrapolation_attempts << "/"
              << guarded.lazy_dca_condensed_outer_extrapolation_acceptances << "/"
              << guarded.lazy_dca_condensed_outer_extrapolation_historical_fallbacks
              << " objective=" << guarded.objective
              << " dual=" << guarded.dual_infeasibility_inf << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
