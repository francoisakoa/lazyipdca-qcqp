#pragma once
#include "lazyipdca_qcqp/majorant_equality_stabilization.hpp"

namespace lazyipdca::qcqp::detail {

struct StandardAutoSelection {
  Settings settings;
  bool enabled=false;
  std::string mode="historical",reason;
};

// Exact coefficient inspection only: neither labels nor starting points
// participate. Keep the six supplied curvature configurations distinct.
inline StandardAutoSelection resolve_standard_auto(const QCQPProblem& p,const Settings& requested) {
  StandardAutoSelection out;out.settings=requested;
  auto& st=out.settings;st.standard_auto_resolved=true;
  if(st.standard_r10_policy!="auto"&&st.standard_r10_policy!="off")
    throw std::invalid_argument("standard-r10-policy expects auto or off");
  auto affine=[](const QuadraticFunction& q) {
    return q.signed_square.empty()&&std::all_of(q.hessian.entries.begin(),q.hessian.entries.end(),
        [](const auto& h){return h.v==0.0;});
  };
  const bool auto_route=st.dca_engine=="auto";
  auto fallback=[&](const char* reason) {
    out.reason=reason;
    if(auto_route)st.dca_engine=st.convexification_study=="off"?"legacy":"majorants";
    return out;
  };
  if(st.standard_r10_policy=="off")return fallback("explicit-standard-policy-off");
  if(!auto_route&&st.dca_engine!="majorants")return fallback("explicit-engine-preserved");
  if(st.outer_algorithm!="lazy-dca"&&st.outer_algorithm!="auto")return fallback("outer-algorithm-unsupported");
  if(st.study_controller!="r3")return fallback("controller-unsupported");
  if(st.convexification_study=="SS_legacy")return fallback("SS_legacy-incompatible-preserved");
  if(st.convexification_study!="off"&&st.convexification_study!="HH"&&st.convexification_study!="HS"&&
      st.convexification_study!="SH"&&st.convexification_study!="SS")return fallback("curvature-configuration-unsupported");
  if(st.convexification_schedule!="lazy")return fallback("explicit-nonlazy-schedule-preserved");
  if(p.has_signed_square())return fallback("signed-square-structure-unsupported");
  if(!affine(p.objective)&&st.convexification_study!="off"){
    bool finite=false;for(int j=0;j<p.n;++j)finite=finite||std::isfinite(p.lb[j])||std::isfinite(p.ub[j]);
    if(finite)return fallback("fixed-bound-PSD-extension-reference-R3-only");
  }
  if(!affine(p.objective)&&!original_quadratic_equality_domain(p))return fallback("objective-not-affine");
  if(!std::all_of(p.inequalities.begin(),p.inequalities.end(),affine))return fallback("source-inequality-not-affine");
  if(std::none_of(p.equalities.begin(),p.equalities.end(),[&](const auto& q){return !affine(q);}))
    return fallback("no-quadratic-source-equality");
  if((st.linear_backend!="mumps"&&st.linear_backend!="ma27")||!st.linear_metis||st.inner_backend=="ipopt")return fallback("native-MUMPS-or-MA27-METIS-required");
  if(st.native_augmented_repair||st.product_definition_presolve||st.power2_scaling)
    return fallback("explicit-alternate-presolve-or-repair-preserved");
  if(st.rank_presolve_context)return fallback("existing-rank-reduction-context-unsupported");
  if(st.accepted_checkpoint_callback&&!st.accepted_checkpoint_builtin)
    return fallback("custom-observer-preserved-R10-unsupported");
  if(st.dca_aggregate_only||st.dca_aggregate||st.dca_krylov||st.dca_local_active||st.dca_filter_anderson||
      st.dca_pair_elimination)return fallback("explicit-incompatible-acceleration-preserved");
  if(st.majorant_equality_stabilization_explicit&&!st.majorant_equality_stabilization)
    return fallback("explicit-source-stabilization-off");
  if(!original_boolean_mode("LAZY_ORIGINAL_STABILIZED_MERIT",true))
    return fallback("explicit-GR-merit-off");
  st.dca_engine="majorants";st.outer_algorithm="lazy-dca";st.majorant_equality_stabilization=true;
  st.dca_dual_step="common";st.dca_strict_finish=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;
  // R10 retains every source row. The historical rank-start/auto-start
  // profile delegates this choice to STANDARD; explicit policy off replays it.
  st.dca_rank_presolve=false;st.free_epigraph_presolve=false;st.majorant_startup_policy="legacy";
  if(st.accepted_checkpoint_builtin)st.accepted_checkpoint_callback={};
  if(!st.lazy_newton_initial_mu_explicit&&
      st.lazy_newton_initial_mu==Settings{}.lazy_newton_initial_mu)st.lazy_newton_initial_mu=.1;
  st.original_stabilized_merit=true;
  st.original_source_primal_barrier=original_boolean_mode("LAZY_ORIGINAL_SOURCE_PRIMAL_BARRIER",true);
  st.original_bound_product_path=original_boolean_mode("LAZY_ORIGINAL_BOUND_PRODUCT_PATH",st.original_source_primal_barrier);
  st.original_gr_short_step_prox=original_boolean_mode("LAZY_ORIGINAL_GR_SHORT_STEP_PROX",st.original_source_primal_barrier);
  const bool analytic_soc=original_boolean_mode("LAZY_ORIGINAL_GR_SOC",false);
  st.original_gr_consistent_bound_slacks=original_boolean_mode("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_SLACKS",
      st.original_source_primal_barrier&&st.original_bound_product_path&&!analytic_soc);
  const char* trigger=std::getenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER");
  st.original_gr_consistent_bound_trigger=trigger?trigger:"on_failure";
  if(st.original_gr_consistent_bound_trigger!="always"&&st.original_gr_consistent_bound_trigger!="on_failure")
    throw std::invalid_argument("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER expects always or on_failure");
  st.original_exact_affine_faces=original_boolean_mode("LAZY_EXACT_AFFINE_FACE_RESEARCH",
      !st.accepted_checkpoint_callback&&!st.rank_presolve_context&&!st.dca_rank_presolve);
  out.enabled=true;out.mode="native-R10-standard";out.reason=original_diagonal_convex_equality_domain(p)?
      "exact-diagonal-convex-objective-equality-domain-eligible":(original_quadratic_equality_domain(p)?
      "exact-signed-quadratic-objective-equality-domain-eligible":"exact-coefficient-domain-eligible");
  return out;
}

} // namespace lazyipdca::qcqp::detail
