#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp::detail {
// Aggregate probes retain the full-step 0.1% contraction boundary. A certified
// original root direction uses the ACTUAL damped fraction in its Armijo test:
// requiring 0.1% independently of alpha blocks legitimate active-bound transitions.
inline constexpr double dca_terminal_contraction_factor = 1.0 - 1e-3;
inline double dca_terminal_contraction_limit(double current,double step=1.0) {
  return (1.0-(1.0-dca_terminal_contraction_factor)*step)*current;
}
inline bool dca_terminal_contracts(double current,double candidate,double step=1.0) {
  return std::isfinite(current) && current>0 && std::isfinite(candidate) && candidate>=0 &&
         std::isfinite(step) && step>0 && step<=1 && candidate<current &&
         candidate<=dca_terminal_contraction_limit(current,step);
}
inline double dca_terminal_radius(double optimality_tolerance) {
  return 10.0 * std::sqrt(std::max(0.0, optimality_tolerance));
}
inline bool dca_stationary_phase(double primal, double normalized_kkt,
                                 double feasibility_tolerance,
                                 double optimality_tolerance) {
  return std::isfinite(primal) && std::isfinite(normalized_kkt) &&
         primal <= feasibility_tolerance &&
         normalized_kkt <= dca_terminal_radius(optimality_tolerance);
}
inline double dca_equality_step(double primal, double feasibility_tolerance,
                                double primal_step, double historical_step) {
  // Equality multipliers have no positivity boundary. Near feasibility they
  // follow x; the independent positive multiplier fraction still applies to z.
  return std::isfinite(primal) && primal <= feasibility_tolerance
             ? primal_step : historical_step;
}
struct DcaStationarityEvidence {
  bool zero_shift_uncertified = false;
  bool regularized_noncontracting = false;
  double minimum_certified_rho = std::numeric_limits<double>::infinity();
  void certificate(double rho, bool certified) {
    if (rho == 0.0 && !certified) zero_shift_uncertified = true;
    if (certified && std::isfinite(rho) && rho >= 0.0)
      minimum_certified_rho = std::min(minimum_certified_rho, rho);
  }
  void candidate(double rho, double current, double trial) {
    if (rho > 0.0 && std::isfinite(rho) && std::isfinite(trial) &&
        !dca_terminal_contracts(current, trial))
      regularized_noncontracting = true;
  }
  bool needs_rescue() const {
    return zero_shift_uncertified && regularized_noncontracting;
  }
};
} // namespace lazyipdca::qcqp::detail
