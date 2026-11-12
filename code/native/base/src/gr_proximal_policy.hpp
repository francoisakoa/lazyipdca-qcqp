#pragma once
#include <cmath>
#include <stdexcept>
#include <string>

namespace lazyipdca::qcqp::detail {

// An ablation of GLOBALIZATION-triggered proximal growth, not a numerical
// regularization policy. Call this only at the two historical growth sites;
// required inertia/pivot recovery and certified floors never pass through it.
struct GrProximalGrowthDecision {
  double effective=0;
  bool blocked=false;
  const char* reason="historical-policy";
};

inline void validate_gr_proximal_policy(const std::string& policy) {
  if(policy!="historical"&&policy!="frozen")
    throw std::invalid_argument("gr-proximal-policy expects historical or frozen");
}

inline GrProximalGrowthDecision gr_globalization_proximal_growth(
    bool gr_globalization,const std::string& policy,
    double remembered,double historical_proposal) {
  validate_gr_proximal_policy(policy);
  GrProximalGrowthDecision out;
  out.effective=historical_proposal;
  out.reason=gr_globalization?"historical-policy":"non-GR-policy-preserved";
  if(gr_globalization&&policy=="frozen") {
    // Preserve invalid/no-growth proposals exactly as before: their caller's
    // finite/increase tests reject them. A new nonfinite value cannot become
    // an apparently admissible proposal through this additional policy.
    if(std::isfinite(remembered)&&remembered>=0&&
       std::isfinite(historical_proposal)&&historical_proposal>remembered) {
      out.effective=remembered;out.blocked=true;
      out.reason="globalization-growth-frozen";
    } else out.reason="frozen-no-admissible-growth";
  }
  return out;
}

} // namespace lazyipdca::qcqp::detail
