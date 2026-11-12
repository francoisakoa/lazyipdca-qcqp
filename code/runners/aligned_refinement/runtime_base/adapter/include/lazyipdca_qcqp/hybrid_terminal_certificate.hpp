#pragma once
#include "lazyipdca_qcqp/acceptable.hpp"

namespace lazyipdca::qcqp::detail {
struct HybridTerminalAudit {
  bool attempted=false,strict=false,expired=false;
  double elapsed_before=0,elapsed_after=0,seconds=0;
  double primal=INFINITY,dual=INFINITY,complementarity=INFINITY;
};
inline bool hybrid_terminal_commit_open(const HybridTerminalAudit& audit,double elapsed,double limit) {
  return audit.strict&&std::isfinite(elapsed)&&elapsed>=audit.elapsed_after&&elapsed<limit;
}
// This certifies termination, never an auxiliary step or a filter acceptance.
// The source coefficient stream and original-coordinate multipliers are the
// same as the final audit. Raw project tolerances remain mandatory even if
// the caller selected an additional normalized termination convention.
template<class Elapsed>
HybridTerminalAudit audit_hybrid_terminal_original(const QCQPProblem& original,
    const Settings& requested,Result& candidate,Elapsed elapsed,
    const std::vector<std::vector<int>>* support=nullptr) {
  HybridTerminalAudit out;out.elapsed_before=out.elapsed_after=elapsed();
  if(!std::isfinite(out.elapsed_before)||out.elapsed_before<0||
      !(out.elapsed_before<requested.time_limit_seconds)) {out.expired=true;return out;}
  Settings strict=requested;strict.termination_test="project";
  out.attempted=true;
  const bool passed=audit_original_termination(original,strict,candidate,support);
  out.elapsed_after=elapsed();
  out.seconds=std::max(0.0,out.elapsed_after-out.elapsed_before);
  out.expired=!std::isfinite(out.elapsed_after)||out.elapsed_after<out.elapsed_before||
      !(out.elapsed_after<requested.time_limit_seconds);
  out.primal=candidate.feasibility.total_inf;
  out.dual=candidate.dual_infeasibility_inf;
  out.complementarity=candidate.complementarity_inf;
  out.strict=passed&&candidate.termination_audit_valid&&!out.expired&&
      std::isfinite(out.primal)&&std::isfinite(out.dual)&&std::isfinite(out.complementarity)&&
      out.primal<=requested.feasibility_tolerance&&
      out.dual<=std::min(requested.stationarity_tolerance,requested.dual_infeasibility_tolerance)&&
      out.complementarity<=requested.complementarity_tolerance;
  return out;
}
} // namespace lazyipdca::qcqp::detail
