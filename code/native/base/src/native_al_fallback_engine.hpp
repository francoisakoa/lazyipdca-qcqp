#pragma once
#include "lazyipdca_qcqp/native_al_fallback_policy.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include "native_augmented_repair.hpp"
#include <chrono>
#include <iostream>
#include <utility>

namespace lazyipdca::qcqp::detail {

inline void native_al_fallback_capture_primary(const Result& r,NativeALFallbackStats& s) {
  s.primary_status=status_string(r.status);s.primary_message=r.message;
  s.primary_outer_iterations=r.outer_iterations;s.primary_inner_iterations=r.inner_iterations;
  s.primary_factorizations=r.kkt_factorizations;s.total_factorizations=r.kkt_factorizations;
  s.primary_primal=r.feasibility.total_inf;s.primary_dual=r.dual_infeasibility_inf;
  s.primary_complementarity=r.complementarity_inf;
  s.primary_effective_outer_algorithm=r.effective_outer_algorithm;
  s.primary_standard_auto_enabled=r.standard_auto_enabled;
  s.primary_standard_auto_exact_global_faces=r.standard_auto_exact_global_faces;
  s.primary_standard_auto_exact_global_faces_applied=r.standard_auto_exact_global_faces_applied;
  s.primary_standard_auto_checkpoint_requested=r.standard_auto_checkpoint_requested;
  s.primary_standard_auto_checkpoint_effective=r.standard_auto_checkpoint_effective;
  s.primary_standard_auto_builtin_checkpoint_skipped=r.standard_auto_builtin_checkpoint_skipped;
  s.primary_standard_auto_rank_presolve_requested=r.standard_auto_rank_presolve_requested;
  s.primary_standard_auto_rank_presolve_effective=r.standard_auto_rank_presolve_effective;
  s.primary_standard_auto_free_epigraph_requested=r.standard_auto_free_epigraph_requested;
  s.primary_standard_auto_free_epigraph_effective=r.standard_auto_free_epigraph_effective;
  s.primary_standard_auto_mode=r.standard_auto_mode;s.primary_standard_auto_reason=r.standard_auto_reason;
  s.primary_standard_auto_requested_engine=r.standard_auto_requested_engine;
  s.primary_standard_auto_effective_engine=r.standard_auto_effective_engine;
  s.primary_standard_auto_configuration=r.standard_auto_configuration;
  s.primary_standard_auto_requested_startup=r.standard_auto_requested_startup;
  s.primary_standard_auto_effective_startup=r.standard_auto_effective_startup;
  s.primary_standard_auto_configured_initial_mu=r.standard_auto_configured_initial_mu;
  s.primary_gr_adjustments_active=r.gr_adjustments_active;
  s.primary_gr_coefficient_constraint_increments=r.gr_coefficient_constraint_increments;
  s.primary_gr_proximal_policy=r.gr_proximal_policy;s.primary_gr_fixed_centering_steps=r.gr_fixed_centering_steps;
  s.primary_gr_residual_components=r.gr_residual_components;
}

inline Result solve_native_al_fallback(const QCQPProblem& original,const Settings& requested) {
  using Clock=std::chrono::steady_clock;
  const auto began=Clock::now();
  auto elapsed=[&](){return std::chrono::duration<double>(Clock::now()-began).count();};
  if(requested.native_al_fallback!="auto"&&requested.native_al_fallback!="off")
    throw std::invalid_argument("native-al-fallback expects auto or off");
  NativeALFallbackStats stats;
  stats.policy=requested.native_al_fallback;stats.enabled=requested.native_al_fallback=="auto";
  stats.original_variables=original.n;stats.global_budget_seconds=requested.time_limit_seconds;
  stats.primary_iteration_budget=requested.max_outer_iterations;
  stats.fallback_iteration_budget=requested.max_outer_iterations;
  Settings primary_settings=requested;primary_settings.native_al_fallback_resolved=true;
  std::cout<<"[SOLVER-ROUTE] stage=primary optimizer="
      <<(requested.native_augmented_repair?"native-al-explicit":"Lazy")
      <<" fallback_policy="<<stats.policy<<" full_primary_budget=1 global_seconds="
      <<requested.time_limit_seconds<<" input_x0_unchanged=1"<<std::endl;
  // Deliberately no time reserve, warm start, changed primary settings or
  // preflight before this call. The original Lazy trajectory stays unchanged.
  Result primary=Solver(primary_settings).solve(original);
  stats.primary_seconds=elapsed();native_al_fallback_capture_primary(primary,stats);
  stats.primary_strict=native_al_fallback_strict_original(original,requested,primary);
  if(!stats.primary_strict&&stats.enabled&&requested.termination_test=="project"&&
      requested.time_limit_seconds-elapsed()>0) {
    const auto audit_began=Clock::now();
    // Stale/missing audit flags cannot cause a strict primary point to enter
    // AL. The audit may reconstruct fixed-bound multipliers, so audit a copy.
    Result primary_audit=primary;
    Settings audit_settings=requested;audit_settings.rank_presolve_context=nullptr;
    audit_settings.accepted_checkpoint_callback={};audit_settings.acceptable_iterations=0;
    audit_settings.acceptable_on_failure=false;
    primary_audit.termination_audit_valid=false;
    try {
      audit_original_termination(original,audit_settings,primary_audit);
      stats.primary_strict=native_al_fallback_strict_original(original,requested,primary_audit);
    } catch(const std::exception&) {stats.primary_strict=false;}
    stats.primary_audit_seconds=std::chrono::duration<double>(Clock::now()-audit_began).count();
    stats.primary_seconds=elapsed();
  }
  if(primary.native_augmented.enabled)stats.endpoint_origin="native-al";
  auto finish=[&](Result chosen) {
    stats.total_seconds=elapsed();stats.within_global_budget=stats.total_seconds<requested.time_limit_seconds;
    if(stats.accepted&&!stats.within_global_budget) {
      // A candidate is committed only while the same common wall deadline is
      // still live; audit/selection overhead receives no extra time grant.
      stats.accepted=false;stats.endpoint_origin="lazy";
      stats.reason="global-wall-budget-exhausted-primary-retained";
      chosen=std::move(primary);
    }
    chosen.native_al_fallback=stats;
    chosen.elapsed_seconds=stats.total_seconds;
    chosen.lazy_newton.solver_wall_seconds=stats.total_seconds;
    std::cout<<"[NATIVE-AL-FALLBACK] attempted="<<stats.attempted<<" accepted="<<stats.accepted
        <<" reason="<<std::quoted(stats.reason)<<" primary_status="<<stats.primary_status
        <<" fallback_status="<<stats.fallback_status<<" primary_seconds="<<stats.primary_seconds
        <<" preflight_seconds="<<stats.preflight_seconds<<" AL_seconds="<<stats.fallback_seconds
        <<" final_audit_seconds="<<stats.final_audit_seconds<<" total_seconds="<<stats.total_seconds
        <<" primary_outer_iterations="<<stats.primary_outer_iterations
        <<" AL_accepted_Newton_steps="<<stats.fallback_accepted_newton_steps
        <<" total_factorizations="<<stats.total_factorizations<<std::endl;
    std::cout<<"[SOLVER-ROUTE] endpoint="<<stats.endpoint_origin<<" fallback_accepted="<<stats.accepted
        <<" original_QCQP_KKT="<<(stats.accepted?stats.fallback_strict:stats.primary_strict)
        <<" primary_retained="<<!stats.accepted<<std::endl;
    return chosen;
  };
  if(stats.primary_strict){stats.reason="primary-original-KKT-no-fallback";return finish(std::move(primary));}
  const auto eligibility=native_al_fallback_eligibility(original,requested);
  stats.coefficient_eligible=eligibility.coefficient_eligible;
  if(!eligibility.eligible){stats.reason=eligibility.reason;return finish(std::move(primary));}
  if(!(requested.time_limit_seconds-elapsed()>0)) {
    stats.reason="no-wall-budget-after-primary";return finish(std::move(primary));
  }
  // This sparse/rational preflight cannot allocate the AL dense Hessian. Its
  // dimension is conservative when the frozen x0 prevents a product rewrite;
  // the unchanged native kernel repeats admission after its own initialization.
  const auto preflight_began=Clock::now();
  try {
    ProductDefinitionPresolve reduction(original);
    stats.preflight_private_variables=reduction.changed()?reduction.reduced.n:original.n;
    stats.preflight_reason=reduction.reason;
    stats.dense_workspace_estimate_bytes=static_cast<double>(
        24.L*stats.preflight_private_variables*stats.preflight_private_variables);
  } catch(const std::exception& error) {
    stats.preflight_reason=error.what();stats.reason="preflight-refused-original-primary-retained";
    stats.preflight_seconds=std::chrono::duration<double>(Clock::now()-preflight_began).count();
    return finish(std::move(primary));
  }
  stats.preflight_seconds=std::chrono::duration<double>(Clock::now()-preflight_began).count();
  if(!native_al_fallback_dense_admitted(stats.preflight_private_variables)) {
    stats.reason="dense-private-Hessian-estimate-exceeds-128-MiB";return finish(std::move(primary));
  }
  stats.fallback_budget_seconds=std::max(0.,requested.time_limit_seconds-elapsed());
  if(!(stats.fallback_budget_seconds>0)) {
    stats.reason="no-wall-budget-after-preflight";return finish(std::move(primary));
  }
  Settings fallback_settings=requested;
  fallback_settings.native_al_fallback_resolved=true;fallback_settings.native_al_fallback="off";
  fallback_settings.time_limit_seconds=stats.fallback_budget_seconds;
  // Built-in checkpoint logging belongs to the unchanged primary stage. The
  // native kernel does not support observer callbacks; custom ones were refused.
  fallback_settings.accepted_checkpoint_callback={};
  fallback_settings.rank_presolve_context=nullptr;
  stats.attempted=true;stats.internal_product_elimination_authorized=true;
  stats.internal_dyadic_scaling_authorized=true;stats.internal_ray_initialization_authorized=true;
  std::cout<<"[NATIVE-AL-FALLBACK] stage=start optimizer=native-PHR-AL-Newton external_optimizer=0"
      <<" original_n="<<original.n<<" preflight_private_n="<<stats.preflight_private_variables
      <<" dense_Hessian_assembly_estimate_bytes="<<stats.dense_workspace_estimate_bytes
      <<" estimate_excludes_MUMPS_and_sparse_copies=1 remaining_seconds="<<stats.fallback_budget_seconds
      <<" AL_accepted_step_budget="<<requested.max_outer_iterations
      <<" original_input_unchanged=1 frozen_x0_unchanged=1 internal_ray_initialization=authorized"
      <<" internal_product_elimination=authorized internal_dyadic_scaling=authorized"
      <<" builtin_observer_skipped="<<(requested.accepted_checkpoint_builtin&&bool(requested.accepted_checkpoint_callback))
      <<std::endl;
  const auto fallback_began=Clock::now();
  Result candidate;
  try {
    candidate=solve_native_augmented_repair(original,fallback_settings);
  } catch(const std::exception& error) {
    stats.fallback_seconds=std::chrono::duration<double>(Clock::now()-fallback_began).count();
    stats.fallback_status="exception";stats.fallback_message=error.what();
    stats.reason="native-AL-exception-primary-retained";return finish(std::move(primary));
  }
  stats.fallback_seconds=std::chrono::duration<double>(Clock::now()-fallback_began).count();
  const auto audit_began=Clock::now();
  if(candidate.status==Status::Solved&&requested.time_limit_seconds-elapsed()>0) {
    Settings audit_settings=requested;audit_settings.rank_presolve_context=nullptr;
    audit_settings.accepted_checkpoint_callback={};audit_settings.acceptable_iterations=0;
    audit_settings.acceptable_on_failure=false;
    candidate.termination_audit_valid=false;
    try {audit_original_termination(original,audit_settings,candidate);}
    catch(const std::exception& error) {candidate.termination_audit_valid=false;stats.fallback_message=error.what();}
  }
  stats.final_audit_seconds=std::chrono::duration<double>(Clock::now()-audit_began).count();
  const auto& a=candidate.native_augmented;
  stats.fallback_status=status_string(candidate.status);
  if(stats.fallback_message.empty())stats.fallback_message=candidate.message;
  stats.fallback_primal=candidate.feasibility.total_inf;stats.fallback_dual=candidate.dual_infeasibility_inf;
  stats.fallback_complementarity=candidate.complementarity_inf;
  stats.fallback_strict=native_al_fallback_strict_original(original,requested,candidate);
  stats.fallback_outer_stages=a.stages;stats.fallback_accepted_newton_steps=a.accepted_steps;
  stats.fallback_factorizations=a.factorizations;stats.fallback_linear_solves=a.linear_solves;
  stats.total_factorizations+=a.factorizations;stats.fallback_admitted=a.admitted;
  stats.fallback_reduced_variables=a.reduced_variables;stats.fallback_initial_projection=a.initial_projection;
  stats.fallback_initial_projection_inf=a.initial_projection_inf;
  stats.fallback_preparation_seconds=a.preparation_seconds;stats.fallback_core_seconds=a.core_seconds;
  stats.fallback_certificate_seconds=a.certificate_seconds;stats.fallback_reason=a.reason;
  stats.fallback_seed_policy=a.seed_policy;
  stats.accepted=native_al_fallback_choose_candidate(original,requested,candidate,elapsed());
  if(stats.accepted) {
    stats.reason="native-AL-original-KKT-selected";stats.endpoint_origin="native-al";
    candidate.message="native AL-Newton fallback after Lazy "+stats.primary_status+": "+candidate.message;
    return finish(std::move(candidate));
  }
  stats.reason=elapsed()>=requested.time_limit_seconds?"global-wall-budget-exhausted-primary-retained":
      "native-AL-not-strictly-solved-primary-retained";
  return finish(std::move(primary));
}

} // namespace lazyipdca::qcqp::detail
