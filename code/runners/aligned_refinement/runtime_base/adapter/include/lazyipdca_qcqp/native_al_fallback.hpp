#pragma once
#include <limits>
#include <string>

namespace lazyipdca::qcqp {

// Route provenance is separate from the selected endpoint. In particular a
// rejected AL attempt must never overwrite a Lazy endpoint or its certificate.
struct NativeALFallbackStats {
  std::string policy="auto",reason="unresolved",endpoint_origin="lazy";
  bool enabled=false,coefficient_eligible=false,attempted=false,accepted=false;
  bool primary_strict=false,fallback_strict=false,within_global_budget=true;
  bool original_input_preserved=true,original_start_preserved=true;
  bool internal_product_elimination_authorized=false,internal_dyadic_scaling_authorized=false;
  bool internal_ray_initialization_authorized=false;
  int original_variables=0,preflight_private_variables=0;
  double dense_workspace_estimate_bytes=0,dense_workspace_limit_bytes=128.*1024.*1024.;
  double global_budget_seconds=0,primary_seconds=0,primary_audit_seconds=0,preflight_seconds=0;
  double fallback_budget_seconds=0,fallback_seconds=0,final_audit_seconds=0,total_seconds=0;
  int primary_outer_iterations=0,primary_inner_iterations=0,primary_factorizations=0;
  int fallback_outer_stages=0,fallback_accepted_newton_steps=0,fallback_factorizations=0;
  int fallback_linear_solves=0,total_factorizations=0;
  int primary_iteration_budget=0,fallback_iteration_budget=0;
  std::string primary_iteration_unit="Lazy outer iteration";
  std::string fallback_iteration_unit="AL accepted Newton step";
  std::string primary_status,fallback_status="not-run",primary_message,fallback_message;
  double primary_primal=std::numeric_limits<double>::infinity();
  double primary_dual=std::numeric_limits<double>::infinity();
  double primary_complementarity=std::numeric_limits<double>::infinity();
  double fallback_primal=std::numeric_limits<double>::infinity();
  double fallback_dual=std::numeric_limits<double>::infinity();
  double fallback_complementarity=std::numeric_limits<double>::infinity();
  bool fallback_admitted=false,fallback_initial_projection=false;
  int fallback_reduced_variables=0;
  double fallback_initial_projection_inf=0,fallback_preparation_seconds=0;
  double fallback_core_seconds=0,fallback_certificate_seconds=0;
  std::string fallback_reason,fallback_seed_policy,preflight_reason;
  // Metadata describes the Lazy stage even when AL supplies the final point.
  bool primary_standard_auto_enabled=false,primary_standard_auto_exact_global_faces=false;
  bool primary_standard_auto_exact_global_faces_applied=false;
  bool primary_standard_auto_checkpoint_requested=false,primary_standard_auto_checkpoint_effective=false;
  bool primary_standard_auto_builtin_checkpoint_skipped=false;
  bool primary_standard_auto_rank_presolve_requested=false,primary_standard_auto_rank_presolve_effective=false;
  bool primary_standard_auto_free_epigraph_requested=false,primary_standard_auto_free_epigraph_effective=false;
  std::string primary_standard_auto_mode,primary_standard_auto_reason;
  std::string primary_standard_auto_requested_engine,primary_standard_auto_effective_engine;
  std::string primary_standard_auto_configuration,primary_effective_outer_algorithm;
  std::string primary_standard_auto_requested_startup,primary_standard_auto_effective_startup;
  double primary_standard_auto_configured_initial_mu=0;
  bool primary_gr_adjustments_active=false,primary_gr_coefficient_constraint_increments=false;
  bool primary_gr_residual_components=false;
  std::string primary_gr_proximal_policy;
  int primary_gr_fixed_centering_steps=0;
};

} // namespace lazyipdca::qcqp
