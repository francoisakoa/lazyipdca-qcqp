#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp::detail {
// R2 observes a stalled numerical trajectory. No instance or suite identifier
// is accepted by this class. This is a bounded retry, NOT a convergence theorem.
class AutoStartupRetry {
 public:
  static constexpr int required_steps = 64;
  bool used() const noexcept { return used_; }
  int streak() const noexcept { return streak_; }
  void mark_used() noexcept { used_ = true; clear_window(); }
  void clear_window() noexcept {
    streak_ = 0; first_primal_ = first_dual_ = first_mu_ = 0.0;
  }
  bool observe(bool eligible, double alpha, double primal, double dual,
               double comp, double mu, double initial_mu, double feas_tol,
               double dual_tol, int remaining_iterations, double remaining_seconds) {
    const double roundoff = 64.0 * std::numeric_limits<double>::epsilon();
    if (used_ || !eligible || !std::isfinite(alpha) || !std::isfinite(primal) ||
        !std::isfinite(dual) || !std::isfinite(comp) || !std::isfinite(mu) ||
        !std::isfinite(initial_mu) || !std::isfinite(feas_tol) ||
        !std::isfinite(dual_tol) || !std::isfinite(remaining_seconds) ||
        primal < 0 || dual < 0 || comp < 0 || feas_tol <= 0 || dual_tol <= 0 ||
        !(alpha > 0 && alpha <= 1e-4) || !(mu > 0 && mu <= 1e-4*(1.0+roundoff)) ||
        std::abs(mu-initial_mu) > roundoff*std::max(mu,initial_mu) || primal <= 100.0 * feas_tol ||
        dual <= 10.0 * std::max({primal, comp, dual_tol}) ||
        remaining_iterations < required_steps || remaining_seconds <= 0) {
      clear_window(); return false;
    }
    if (streak_ == 0) {
      first_primal_ = primal; first_dual_ = dual; first_mu_ = mu; streak_ = 1;
      return false;
    }
    // A 10% improvement of either residual, or a barrier update, interrupts
    // the window. The next observation can start a new independent window.
    if (mu != first_mu_ || primal < 0.9 * first_primal_ || dual < 0.9 * first_dual_) {
      clear_window(); return false;
    }
    return ++streak_ >= required_steps;
  }
 private:
  bool used_ = false;
  int streak_ = 0;
  double first_primal_ = 0, first_dual_ = 0, first_mu_ = 0;
};
}  // namespace lazyipdca::qcqp::detail
