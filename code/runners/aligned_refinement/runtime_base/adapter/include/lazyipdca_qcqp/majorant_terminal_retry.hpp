#pragma once
#include <cmath>
namespace lazyipdca::qcqp::detail {
// Only consulted at the old terminal line-search failure, after all existing
// recovery paths. Share AutoStartupRetry's one-shot budget with the R2 retry.
inline bool terminal_central_retry_needed(bool auto_mode, bool central_first,
    bool retry_used, bool mixed, bool signed_squares, bool objective_lazy,
    double initial_mu, double primal, double dual, double comp,
    double feas_tol, double dual_tol, double comp_tol,
    int remaining_iterations, double remaining_seconds) noexcept {
  return auto_mode && !central_first && !retry_used && mixed &&
      !signed_squares && !objective_lazy &&
      std::isfinite(initial_mu) && initial_mu>0.0 && initial_mu<1.0 &&
      std::isfinite(primal) && std::isfinite(dual) && std::isfinite(comp) &&
      std::isfinite(feas_tol) && feas_tol>0 &&
      std::isfinite(dual_tol) && dual_tol>0 &&
      std::isfinite(comp_tol) && comp_tol>0 && dual>=0 && comp>=0 &&
      primal>100.0*feas_tol && (dual>dual_tol || comp>comp_tol) &&
      remaining_iterations>0 && std::isfinite(remaining_seconds) && remaining_seconds>0;
}
} // namespace lazyipdca::qcqp::detail
