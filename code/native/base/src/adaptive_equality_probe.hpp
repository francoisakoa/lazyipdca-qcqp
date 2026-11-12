#pragma once
#include "lazyipdca_qcqp/standard_auto.hpp"
#include "lazyipdca_qcqp/native_al_fallback_policy.hpp"
#include "rank_retry_statistics.hpp"
#include <chrono>

namespace lazyipdca::qcqp::detail {
// A healthy historical trajectory keeps the entire requested allowance.
// A failed solve or an observed short-step stagnation may release its unused
// work to the stabilized route. This is not a fixed iteration prefix.
inline int adaptive_equality_probe_outer_budget(const Settings& st) {
  return st.max_outer_iterations>=8?st.max_outer_iterations:0;
}
inline bool adaptive_equality_probe_eligible(const QCQPProblem& p,const Settings& st,
    const StandardAutoSelection& selected) {
  if(!selected.enabled||st.standard_r10_policy!="auto"||
      adaptive_equality_probe_outer_budget(st)==0||
      st.majorant_equality_stabilization_explicit||st.majorant_equality_stabilization||
      st.original_stabilized_merit||st.original_source_primal_barrier||
      st.original_private_phase||st.majorant_call_depth||
      std::getenv("LAZY_ORIGINAL_STABILIZED_MERIT")||
      std::getenv("LAZY_ORIGINAL_SOURCE_PRIMAL_BARRIER"))return false;
  bool nonconstant=false;
  p.objective.for_each_linear_nonzero([&](int,double a){nonconstant=nonconstant||a!=0.;});
  const bool affine=p.objective.signed_square.empty()&&
      std::all_of(p.objective.hessian.entries.begin(),p.objective.hessian.entries.end(),
          [](const auto& h){return h.v==0.;});
  // The automatic stabilized route was extended to diagonal convex
  // objectives. Preserve the same observed historical-primary policy for
  // that extension: successful trajectories keep their original operator,
  // while an actual failure/stagnation can release unused work to R10.
  return (affine&&nonconstant)||original_diagonal_convex_equality_domain(p);
}
inline bool adaptive_original_dimensions(const QCQPProblem& p,const Result& r) {
  auto finite=[](const Vec& a){return std::all_of(a.begin(),a.end(),[](double v){return std::isfinite(v);});};
  return r.termination_audit_valid&&r.x.size()==std::size_t(p.n)&&finite(r.x)&&
      r.certificate_equality_multipliers.size()==p.equalities.size()&&
      r.certificate_inequality_multipliers.size()==p.inequalities.size()&&
      r.certificate_lower_multipliers.size()==r.x.size()&&
      r.certificate_upper_multipliers.size()==r.x.size()&&
      finite(r.certificate_equality_multipliers)&&finite(r.certificate_inequality_multipliers)&&
      finite(r.certificate_lower_multipliers)&&finite(r.certificate_upper_multipliers);
}
inline double adaptive_original_raw_score(const QCQPProblem& p,const Result& r) {
  if(!adaptive_original_dimensions(p,r)||!std::isfinite(r.feasibility.total_inf)||
      !std::isfinite(r.dual_infeasibility_inf)||!std::isfinite(r.complementarity_inf))return INFINITY;
  double score=std::max({r.feasibility.total_inf,r.dual_infeasibility_inf,r.complementarity_inf});
  for(double z:r.certificate_inequality_multipliers)score=std::max(score,-z);
  for(int j=0;j<p.n;++j) {
    const double lo=r.certificate_lower_multipliers[j],up=r.certificate_upper_multipliers[j];
    score=std::max({score,-lo,-up});
    if(!std::isfinite(p.lb[j]))score=std::max(score,std::abs(lo));
    if(!std::isfinite(p.ub[j]))score=std::max(score,std::abs(up));
  }
  return score;
}
inline bool adaptive_original_strict(const QCQPProblem& p,const Settings& st,const Result& r) {
  return native_al_fallback_strict_original(p,st,r)&&adaptive_original_raw_score(p,r)<=
      std::min({1e-6,st.stationarity_tolerance,st.feasibility_tolerance,
        st.dual_infeasibility_tolerance,st.complementarity_tolerance});
}
inline int adaptive_remaining_outer(const Settings& st,const Result& probe) {
  return std::max(0,st.max_outer_iterations-std::max(0,probe.outer_iterations));
}
inline int adaptive_remaining_restoration(const Settings& st,const Result& probe) {
  return std::max(0,std::min(st.max_outer_iterations,st.restoration_max_iterations)-
      std::max(0,probe.restoration_iterations));
}
} // namespace lazyipdca::qcqp::detail
