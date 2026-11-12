#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp::detail {
struct RhoDecayFeasibilityGate {
  bool allowed=false;
  double limit=0;
  const char* reason="invalid-physical-primal";
};
// Admission of a REDUCTION in the next auxiliary Newton proximal memory only.
// Physical original residuals exclude barrier slacks and include original bounds.
// No direction, point, multiplier, model, filter or final tolerance is changed.
inline RhoDecayFeasibilityGate rho_decay_feasibility_gate(
    double previous_primal,double current_primal,double feasibility_tolerance) {
  RhoDecayFeasibilityGate out;
  if(!std::isfinite(previous_primal)||previous_primal<0||
      !std::isfinite(current_primal)||current_primal<0||
      !std::isfinite(feasibility_tolerance)||!(feasibility_tolerance>0))return out;
  const long double reference=std::max(previous_primal,feasibility_tolerance);
  const long double scale=std::max({1.0,previous_primal,current_primal,feasibility_tolerance});
  const long double limit=reference+64*std::numeric_limits<double>::epsilon()*scale;
  out.limit=static_cast<double>(limit);
  out.allowed=static_cast<long double>(current_primal)<=limit;
  out.reason=out.allowed?"physical-primal-preserved":"physical-primal-worsened";
  return out;
}
}
