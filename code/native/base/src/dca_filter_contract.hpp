#pragma once
#include "dca_terminal_policy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace lazyipdca::qcqp::detail {
struct OriginalFilterPoint {
  double objective = 0, theta = 0, kkt = 0;
  double lagrangian = std::numeric_limits<double>::quiet_NaN();
  // NaN preserves the standalone legacy filter API. Engine points always
  // supply the original infinity residual and multiplier-scaled NLP error.
  double primal_inf = std::numeric_limits<double>::quiet_NaN();
  double normalized_kkt = std::numeric_limits<double>::quiet_NaN();
  bool acceptable_certified = false;
  bool within_acceptable_caps = false;
  bool distinct_from_anchor = false;
};
// A diagnostic of one decision, without inserting or removing filter entries.
// Limits use exactly the same frozen scaling as OriginalFilterPoint.
struct OriginalFilterDecision {
  bool accepted = false, finite = false;
  bool feasibility = false, objective = false, terminal = false;
  bool stationary_phase = false, acceptable_continuation = false;
  bool curvature_descent = false;
  bool history_checked = false, history_bypassed = false;
  const char *reason = "nonfinite";
  int history_index = -1;
  double noise = std::numeric_limits<double>::quiet_NaN();
  double feasibility_theta_limit = std::numeric_limits<double>::quiet_NaN();
  double objective_limit = std::numeric_limits<double>::quiet_NaN();
  double theta_cap = std::numeric_limits<double>::quiet_NaN();
  double terminal_kkt_limit = std::numeric_limits<double>::quiet_NaN();
  double history_theta_limit = std::numeric_limits<double>::quiet_NaN();
  double history_objective_limit = std::numeric_limits<double>::quiet_NaN();
};
class OriginalDcaFilter {
  std::vector<OriginalFilterPoint> entries_;
  double theta_cap_;
  double optimality_tolerance_;

public:
  explicit OriginalDcaFilter(double theta_cap = 1.0, double optimality_tolerance = 1e-6)
      : theta_cap_(theta_cap), optimality_tolerance_(optimality_tolerance) {}
  // All quantities are evaluated on original functions with frozen scaling.
  // The local progress test is additional to nondomination by filter history.
  OriginalFilterDecision diagnose(const OriginalFilterPoint &from, const OriginalFilterPoint &to,
                                  double predicted, double feasibility_tol, double terminal_fraction=1.0, bool curvature_descent=false) const {
    OriginalFilterDecision decision;
    decision.finite =
        std::isfinite(to.objective) && std::isfinite(to.theta) && std::isfinite(to.kkt);
    decision.theta_cap = theta_cap_;
    if (!decision.finite)
      return decision;
    const double noise =
        32 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(from.objective));
    const bool explicit_primal = !std::isnan(from.primal_inf) && !std::isnan(to.primal_inf);
    const double primal0 = explicit_primal ? from.primal_inf : from.theta;
    const double primal = explicit_primal ? to.primal_inf : to.theta;
    const double normalized = std::isnan(from.normalized_kkt) ? from.kkt : from.normalized_kkt;
    const bool stationary = !curvature_descent && explicit_primal && dca_stationary_phase(
        primal0, normalized, feasibility_tol, optimality_tolerance_);
    decision.stationary_phase = stationary;
    decision.curvature_descent = curvature_descent;
    const bool feasibility = !curvature_descent && !stationary && from.theta > 1e-12 &&
                             to.theta <= (1 - 1e-3) * from.theta &&
                             (primal0 > feasibility_tol || to.kkt <= 0.99 * from.kkt);
    const bool objective = !stationary && predicted > noise &&
                           to.objective <= from.objective - 1e-4 * predicted &&
                           to.theta <= theta_cap_;
    // At a stationary feasible point, objective cancellation cannot measure
    // progress. Require ORIGINAL unperturbed KKT contraction and no increase.
    const bool terminal_merit =
        to.objective <= from.objective + noise ||
        (from.kkt <= 1.0 && std::isfinite(from.lagrangian) && std::isfinite(to.lagrangian) &&
         to.lagrangian <= from.lagrangian + noise);
    const bool continuation = stationary && from.acceptable_certified &&
        to.within_acceptable_caps && to.distinct_from_anchor;
    const bool terminal = !curvature_descent && primal0 <= feasibility_tol && primal <= feasibility_tol &&
        terminal_merit && (dca_terminal_contracts(from.kkt, to.kkt, terminal_fraction) || continuation);
    decision.acceptable_continuation = terminal && continuation;
    decision.noise = noise;
    decision.feasibility_theta_limit = (1 - 1e-3) * from.theta;
    decision.objective_limit = from.objective - 1e-4 * predicted;
    decision.terminal_kkt_limit = dca_terminal_contraction_limit(from.kkt,terminal_fraction);
    decision.feasibility = feasibility;
    decision.objective = objective;
    decision.terminal = terminal;
    if (!(feasibility || objective || terminal)) {
      decision.reason = "local-progress";
      return decision;
    }
    if (terminal) {
      // Preserve the G10 terminal exception, and explicitly report it. This is
      // not a claim that nondomination by historical entries was checked.
      decision.accepted = true;
      decision.history_bypassed = true;
      decision.reason = "accepted";
      return decision;
    }
    decision.history_checked = true;
    for (std::size_t i = 0; i < entries_.size(); ++i) {
      const auto &p = entries_[i];
      if (!(to.theta <= (1 - 1e-4) * p.theta || to.objective <= p.objective - 1e-4 * p.theta)) {
        decision.reason = "filter-history";
        decision.history_index = static_cast<int>(i);
        decision.history_theta_limit = (1 - 1e-4) * p.theta;
        decision.history_objective_limit = p.objective - 1e-4 * p.theta;
        return decision;
      }
    }
    decision.accepted = true;
    decision.reason = "accepted";
    return decision;
  }
  bool acceptable(const OriginalFilterPoint &from, const OriginalFilterPoint &to, double predicted,
                  double feasibility_tol, double terminal_fraction=1.0, bool curvature_descent=false) const {
    return diagnose(from, to, predicted, feasibility_tol,terminal_fraction,curvature_descent).accepted;
  }
  void insert(OriginalFilterPoint p) {
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [&](const auto &a) {
                                    return p.theta <= a.theta && p.objective <= a.objective;
                                  }),
                   entries_.end());
    entries_.push_back(p);
  }
  std::size_t size() const { return entries_.size(); }
};
} // namespace lazyipdca::qcqp::detail
