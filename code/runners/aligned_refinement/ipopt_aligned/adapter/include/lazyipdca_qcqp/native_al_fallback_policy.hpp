#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {

struct NativeALFallbackEligibility {
  bool coefficient_eligible=false,eligible=false;
  std::string reason;
};

inline NativeALFallbackEligibility native_al_fallback_eligibility(
    const QCQPProblem& p,const Settings& st) {
  NativeALFallbackEligibility out;
  auto stop=[&](const char* reason){out.reason=reason;return out;};
  if(st.native_al_fallback!="auto"&&st.native_al_fallback!="off")
    throw std::invalid_argument("native-al-fallback expects auto or off");
  if(st.native_al_fallback=="off")return stop("explicit-policy-off");
  if(p.has_signed_square())return stop("signed-square-structure-unsupported");
  auto affine=[](const QuadraticFunction& q) {
    return q.signed_square.empty()&&std::all_of(q.hessian.entries.begin(),q.hessian.entries.end(),
        [](const auto& h){return h.v==0.;});
  };
  if(!affine(p.objective))return stop("objective-not-affine");
  if(!std::all_of(p.inequalities.begin(),p.inequalities.end(),affine))
    return stop("source-inequality-not-affine");
  if(std::none_of(p.equalities.begin(),p.equalities.end(),[&](const auto& q){return !affine(q);}))
    return stop("no-quadratic-source-equality");
  out.coefficient_eligible=true;
  if(st.linear_backend!="mumps"||!st.linear_metis||st.inner_backend=="ipopt")
    return stop("native-MUMPS-METIS-required");
  if(st.termination_test!="project")return stop("original-project-termination-required");
  if(st.outer_algorithm!="lazy-dca"&&st.outer_algorithm!="auto")
    return stop("explicit-outer-algorithm-preserved");
  if(st.native_augmented_repair||st.product_definition_presolve||st.power2_scaling)
    return stop("explicit-alternate-presolve-or-repair-preserved");
  if(st.rank_presolve_context)return stop("existing-rank-reduction-context-preserved");
  if(st.original_private_phase||st.majorant_call_depth)
    return stop("private-nested-model-unsupported");
  if(st.accepted_checkpoint_callback&&!st.accepted_checkpoint_builtin)
    return stop("custom-observer-preserved");
  if(st.dca_aggregate_only||st.dca_aggregate||st.dca_krylov||st.dca_local_active||
      st.dca_filter_anderson||st.dca_pair_elimination)
    return stop("explicit-incompatible-acceleration-preserved");
  out.eligible=true;return stop("exact-coefficient-domain-eligible");
}

inline bool native_al_fallback_strict_original(const QCQPProblem& p,const Settings& st,
    const Result& r) {
  const double primal=std::min({1e-6,st.feasibility_tolerance,st.stationarity_tolerance});
  const double dual=std::min({1e-6,st.dual_infeasibility_tolerance,st.stationarity_tolerance});
  const double comp=std::min({1e-6,st.complementarity_tolerance,st.stationarity_tolerance});
  auto finite=[](const Vec& values) {
    return std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);});
  };
  auto nonnegative=[](const Vec& values) {
    return std::all_of(values.begin(),values.end(),[](double value){return value>=0.;});
  };
  return r.termination_audit_valid&&r.x.size()==static_cast<std::size_t>(p.n)&&
    r.certificate_equality_multipliers.size()==p.equalities.size()&&
    r.certificate_inequality_multipliers.size()==p.inequalities.size()&&
    r.certificate_lower_multipliers.size()==r.x.size()&&
    r.certificate_upper_multipliers.size()==r.x.size()&&finite(r.x)&&
    finite(r.certificate_equality_multipliers)&&finite(r.certificate_inequality_multipliers)&&
    finite(r.certificate_lower_multipliers)&&finite(r.certificate_upper_multipliers)&&
    nonnegative(r.certificate_inequality_multipliers)&&nonnegative(r.certificate_lower_multipliers)&&
    nonnegative(r.certificate_upper_multipliers)&&
    primal>0&&dual>0&&comp>0&&std::isfinite(r.feasibility.total_inf)&&
    std::isfinite(r.dual_infeasibility_inf)&&std::isfinite(r.complementarity_inf)&&
    r.feasibility.total_inf<=primal&&r.dual_infeasibility_inf<=dual&&r.complementarity_inf<=comp;
}

inline bool native_al_fallback_dense_admitted(int private_n) {
  return private_n>0&&24.L*private_n*private_n<=128.L*1024*1024;
}

// All stages use the same wall deadline. No iteration-unit conversion is
// implied: Lazy outer iterations and AL accepted Newton steps stay separate.
inline bool native_al_fallback_choose_candidate(const QCQPProblem& p,const Settings& st,
    const Result& candidate,double total_seconds) {
  return candidate.status==Status::Solved&&std::isfinite(total_seconds)&&
      total_seconds<st.time_limit_seconds&&native_al_fallback_strict_original(p,st,candidate);
}

} // namespace lazyipdca::qcqp::detail
