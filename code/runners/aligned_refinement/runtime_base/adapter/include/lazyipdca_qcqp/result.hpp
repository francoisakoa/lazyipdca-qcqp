#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/native_al_fallback.hpp"
#include <limits>
#include <array>
#include <cstdint>
#include <string>

namespace lazyipdca::qcqp {

enum class Status {
  Solved,
  MaxOuterIterations,
  TimeLimit,
  RestorationFailure,
  InnerFailure,
  FactorizationFailure,
  LineSearchFailure,
  TrustRegionFailure,
  NumericalError,
  Infeasible,
  SolvedAcceptable,
  SolvedAcceptableOnFailure,
  // Control transfer from the observed historical phase, not a numerical
  // failure. Append so every previously exported status code stays stable.
  HistoricalStagnation
};

inline const char* status_string(Status s) {
  switch (s) {
    case Status::SolvedAcceptable: return "solved_acceptable";
    case Status::SolvedAcceptableOnFailure: return "solved_acceptable_on_failure";
    case Status::Solved: return "solved";
    case Status::MaxOuterIterations: return "max_outer_iterations";
    case Status::TimeLimit: return "time_limit";
    case Status::RestorationFailure: return "restoration_failure";
    case Status::InnerFailure: return "inner_failure";
    case Status::FactorizationFailure: return "factorization_failure";
    case Status::LineSearchFailure: return "line_search_failure";
    case Status::TrustRegionFailure: return "trust_region_failure";
    case Status::NumericalError: return "numerical_error";
    case Status::Infeasible: return "infeasible";
    case Status::HistoricalStagnation: return "historical_stagnation";
  }
  return "unknown";
}

// One attempt at x_k, before applying its direction. Terminal records describe
// the final point. Curvatures are diagnostic Rayleigh quotients of the actual
// full primal direction and scaled Newton blocks, NOT eigenvalue estimates.
struct LazyNewtonIteration {
  int iteration=0, attempt=0;
  std::string event="INIT", factor_backend="none";
  bool has_direction=false, curvature_available=false, inertia_satisfied=false;
  bool regularization_reused=false;
  int active_inequalities=0, total_inequalities=0, active_bounds=0;
  int active_quadratic_inequalities=0, weighted_quadratic_inequalities=0;
  int weighted_quadratic_equalities=0, active_dense_constraints=0;
  int objective_atoms=0, objective_correction_rank=0;
  int inequality_atoms=0, inequality_correction_rank=0;
  int equality_atoms=0,equality_correction_rank=0,completion_atoms=0;
  int atomic_active=0;bool atom_ranks_exact=true;
  double atomic_trace=0.0;
  int regularization_rank=0, normal_augmentation_rank=0;
  int factor_attempts=0, linear_solves=0, backtracks=0, activations=0;
  int regularization_increases=0, native_attempts=0, recovery_attempts=0;
  int symbolic_analyses=0, symbolic_reuses=0;
  int condensation_recoveries=0, promoted_inequalities=0, promoted_bounds=0, kkt_dimension=0;
  double objective=0, inf_eq=0, inf_ineq=0, inf_bnd=0, inf_du=0, comp=0, mu=0;
  double direction_inf=0, step_inf=0, alpha_primal=0, alpha_dual=0;
  double regularization=0, normal_augmentation_shift=0, direction_residual=0;
  double objective_curvature=std::numeric_limits<double>::quiet_NaN();
  double inequality_curvature=std::numeric_limits<double>::quiet_NaN();
  double equality_curvature=std::numeric_limits<double>::quiet_NaN();
  double inequality_barrier_curvature=std::numeric_limits<double>::quiet_NaN();
  double bound_barrier_curvature=std::numeric_limits<double>::quiet_NaN();
  double regularization_curvature=std::numeric_limits<double>::quiet_NaN();
  double lagrangian_curvature=std::numeric_limits<double>::quiet_NaN();
  double barrier_newton_curvature=std::numeric_limits<double>::quiet_NaN();
  double factor_seconds=0, solve_seconds=0, elapsed_seconds=0;
};

enum class NewtonCost { Preparation, Evaluation, Assembly, Factorization, LinearBackend,
  Direction, DirectionAudit, Globalization, Atoms, CoreOther, Certificate, Diagnostics, Cleanup, Count };
inline constexpr std::array<const char*,13> newton_cost_names={"preparation","evaluation","assembly","factorization","linear_backend","direction","direction_audit","globalization","atoms","core_other","certificate","diagnostics","cleanup"};
struct NewtonCalls {
  std::uint64_t model_values=0,model_gradients=0;
  std::uint64_t values=0,gradients=0,cached_gradient_accumulations=0,jacobian_products=0,hessian_assemblies=0,hessian_products=0;
};
// Phases: preparation, core, diagnostics, original certificate.
// Families: objective, equalities, original inequalities, generated bound rows.
struct LazyNewtonStatistics {
  int majorant_geometry_parallel_batches=0,majorant_geometry_threads=1;
  int predictor_corrector_fallbacks=0;
  bool packed_gradients_enabled=false;
  bool predictor_corrector_enabled=false;
  bool predictor_corrector_requested=false;
  bool predictor_corrector_initialization_guard=false;
  int display_level=0;
  std::array<double,13> cost_seconds{};
  std::array<std::array<NewtonCalls,4>,4> calls{};
  std::uint64_t lagrangian_hessian_products=0;
  double core_solve_seconds=0,solver_wall_seconds=0;

  bool atomic_regularization_enabled=false, atom_ranks_exact=true,sparse_certificate_enabled=false;
  bool atom_energy_sparsify_enabled=false,atom_rayleigh_gate_enabled=false,atom_transport_enabled=false,atom_step_decay_enabled=false,atom_guided_enabled=false;
  int atom_guided_batches=0,atom_guided_after=2,atom_plateau_events=0;
  bool atom_adaptation_activated=false;
  int atom_multiplier_transports=0;
  int atom_projection_builds=0,atom_projection_updates=0,atom_projection_requests=0;
  double atom_projection_seconds=0.0,atom_assembly_seconds=0.0,atom_product_seconds=0.0;
  int direction_corrections=0,direction_residual_evaluations=0;
  int precision_condensation_recoveries=0,precision_condensation_rows=0;
  int atom_oracle_calls=0,atom_oracle_solves=0,atom_oracle_failures=0,atom_tangent_witnesses=0;
  double atom_oracle_seconds=0.0,atom_maximum_tangent_error=0.0,atom_last_witness_coverage=0.0;
  int equality_atoms=0,equality_correction_rank=0,completion_atoms=0;
  int atom_sparsifications=0,atom_dropped_coordinates=0,atom_sparse_sign_rejections=0;
  int atom_rayleigh_probes=0,atom_covered_rejections=0,atom_coverage_observations=0;
  double atom_last_probe_coverage=0.0;
  int atom_source_analyses=0,atom_spectral_blocks=0,atom_large_blocks=0;
  int atom_cached_candidates=0,atom_creations=0,atom_reactivations=0;
  int atom_activation_batches=0,atom_completion_batches=0,atom_layout_updates=0;
  int atom_peak_active=0,atom_active=0,atom_final_rank=0;
  double atom_analysis_seconds=0.0,atom_update_seconds=0.0,atom_decay=0.0;
  double atom_maximum_trace=0.0,atom_final_trace=0.0;
  // Cumulative selected source atoms; ranks refer to the final active set.
  int objective_atoms=0, objective_correction_rank=0;
  int inequality_atoms=0, inequality_correction_rank=0;
  int objective_hessian_nnz=0, inequality_hessian_nnz=0, equality_hessian_nnz=0;
  int quadratic_inequalities=0, quadratic_equalities=0;
  int kkt_dimension=0, kkt_structural_nnz=0, dense_constraint_borders=0;
  double diagnostic_seconds=0.0;
  bool live_iterations_enabled=false, live_curvature_enabled=false;
  std::vector<LazyNewtonIteration> iteration_history;

  int condensation_recoveries=0, promoted_inequalities=0, promoted_bounds=0;
  double maximum_condensation_ratio=0.0;
  int dual_only_acceptances = 0;
  int structure_builds = 0;
  int reduced_variables = 0;
  int reduced_equalities = 0;
  int eliminated_variables = 0;
  int fixed_variables = 0;
  int numeric_updates = 0;
  int factor_attempts = 0;
  int inertia_rejections = 0;
  int regularization_increases = 0;
  int regularization_decreases = 0;
  int backend_regularization_attempts = 0;
  int backend_factor_failures = 0;
  int backend_regularized_factor_failures = 0;
  int dual_regularized_factors = 0;
  int curvature_reuses = 0;
  int unshifted_factors = 0;
  int unshifted_accepted_steps = 0;
  int pivoted_fallbacks = 0;
  int native_factor_successes = 0;
  int native_linear_solves = 0;
  int sparse_numeric_failures = 0;
  int normal_recovery_attempts = 0;
  int normal_recovery_successes = 0;
  int normal_recovery_linear_solves = 0;
  int normal_recovery_symbolic_analyses = 0;
  int normal_recovery_symbolic_reuses = 0;
  int first_sparse_failure_iteration = -1;
  int last_sparse_failure_coordinate = -1;
  int last_native_iteration = -1;
  int last_normal_recovery_equation = -1;
  double native_factor_seconds = 0.0;
  double native_solve_seconds = 0.0;
  double normal_recovery_factor_seconds = 0.0;
  double normal_recovery_solve_seconds = 0.0;
  double normal_recovery_maximum_shift = 0.0;
  int symbolic_analyses = 0;
  int symbolic_reuses = 0;
  int scatter_builds = 0;
  int scatter_reuses = 0;
  int initial_active_constraints = 0;
  int final_active_constraints = 0;
  int constraint_activations = 0;
  int working_set_retries = 0;
  int line_search_backtracks = 0;
  int accepted_steps = 0;
  int full_steps = 0;
  int linear_solves = 0;
  int original_certificates = 0;
  double objective_scale = 0.0;
  double initial_scaled_violation = 0.0;
  double initial_barrier = 0.0;
  double maximum_regularization = 0.0;
  double final_regularization = 0.0;
  double maximum_backend_primal_regularization = 0.0;
  double maximum_backend_dual_regularization = 0.0;
  double maximum_failed_backend_primal_regularization = 0.0;
  double maximum_failed_backend_dual_regularization = 0.0;
  double final_backend_primal_regularization = 0.0;
  double final_backend_dual_regularization = 0.0;
  double minimum_step = 0.0;
  double maximum_direction_residual = 0.0;
  double setup_seconds = 0.0;
  double evaluation_seconds = 0.0;
  double assembly_seconds = 0.0;
  double factor_seconds = 0.0;
  double solve_seconds = 0.0;
  double globalization_seconds = 0.0;
  double certificate_seconds = 0.0;
};

struct DcaFilterStatistics {
  // Conditional quadratic homotopy recovery; t<1 is never original success.
  bool square_continuation_attempted=false,square_continuation_accepted=false;
  bool square_cold_restoration_attempted=false,square_cold_restoration_accepted=false;
  int square_cold_restoration_iterations=0;
  int square_continuation_stages=0,square_continuation_trials=0;
  int square_continuation_iterations=0,square_continuation_factors=0,square_continuation_solves=0;
  int square_continuation_evaluations=0,square_continuation_gradient_evaluations=0;
  int square_continuation_precision_rejections=0,square_continuation_krylov_iterations=0,square_continuation_reused_directions=0;
  double square_continuation_parameter=0,square_continuation_seconds=0;
  double square_continuation_linear_error=0,square_continuation_attempted_linear_error=0;


  // Inequality majorants keep the original objective and equalities.
  bool inequality_majorants_enabled=false;
  bool majorant_filter_reanchor_enabled=false;
  bool majorant_outer_only_enabled=false;
  int majorant_original_cache_evaluations=0,majorant_mu_updates=0;
  bool majorant_barrier_filter_enabled=false;
  bool majorant_original_filter_enabled=false;
  bool majorant_slack_transport_enabled=false;
  int majorant_slack_transports=0;
  bool majorant_outer_predictor_corrector_enabled=false;
  int majorant_predictor_directions=0,majorant_corrector_directions=0;
  double majorant_affine_mu=0,majorant_centering_sigma=0;
  bool majorant_startup_predictor_corrector_enabled=false;
  bool majorant_terminal_single_newton=false;
  int majorant_startup_predictor_corrector_steps=0;
  int majorant_terminal_single_newton_steps=0;
  int majorant_direction_precision_retries=0;
  double majorant_startup_predictor_corrector_seconds=0;
  int majorant_barrier_phases=0,majorant_barrier_filter_resets=0;
  int majorant_barrier_line_trials=0,majorant_barrier_line_acceptances=0;
  int majorant_barrier_line_rejections=0;
  double majorant_barrier_mu=0,majorant_barrier_residual=0;
  int majorant_filter_path_builds=0,majorant_filter_path_evaluations=0;
  int majorant_early_reanchors=0;
  int majorant_curvature_threads=1;
  int majorant_curvature_certificates=0,majorant_curvature_reuses=0;
  int majorant_curvature_small_spectra=0,majorant_curvature_gershgorin=0;
  int majorant_curvature_analytic_spectra=0;
  double majorant_curvature_seconds=0;
  int majorant_outer_iterations=0,majorant_inner_convergences=0,majorant_reanchors=0;
  int majorant_reject_factorization=0,majorant_reject_precision=0,majorant_reject_linear_backend=0;
  int majorant_reject_globalization=0,majorant_reject_domain=0;
  double majorant_inner_kkt=0,majorant_inner_primal=0,majorant_inner_dual=0;
  double majorant_inner_complementarity=0,majorant_gap_max=0;
  double majorant_stationarity_gap=0,majorant_complementarity_gap=0;
  int majorant_lambda_attempts=0,majorant_lambda_acceptances=0,majorant_lambda_iterations=0;
  double majorant_lambda_seconds=0,majorant_lambda_last_alpha=1;
  int majorant_lambda_interpolations=0,majorant_lambda_rho_probes=0;
  int majorant_curvature_attempts=0,majorant_curvature_solves=0,majorant_curvature_acceptances=0;
  double majorant_curvature_correction_seconds=0;
  double majorant_slack_transport_error=0,majorant_regularization=0;
  std::string majorant_last_rejection;
  double kkt_growth_max=0;
  int progress_window=0,progress_events=0,aggregate_progress_stalls=0,dual_progress_stalls=0;
  int progress_rescue_trials=0,progress_rescue_acceptances=0;
  int terminal_rescue_triggers=0,terminal_rescue_trials=0,terminal_rescue_acceptances=0;
  int aggregate_root_trials=0,aggregate_root_acceptances=0;
  int aggregate_curvature_descents=0,aggregate_curvature_shortcuts=0;
  std::string progress_last_event;
  bool aggregate_enabled=false;
  std::string aggregate_kkt_layout,aggregate_rejection_policy,aggregate_last_rejection_kind;
  std::string aggregate_last_backend_message;
  bool aggregate_fast_assembly=false,aggregate_rho_memory=false,aggregate_lean=false;
  int aggregate_kkt_layout_rebuilds=0,aggregate_kkt_border_capacity=0;
  int aggregate_kkt_border_assigned=0,aggregate_kkt_dimension=0;
  int aggregate_reject_inertia=0,aggregate_reject_backend=0,aggregate_reject_numerical_direction=0;
  int aggregate_reject_domain=0,aggregate_reject_filter=0,aggregate_reject_kkt=0,aggregate_reject_globalization=0;
  int aggregate_rho_memory_uses=0,aggregate_step_regularizations=0;
  double aggregate_last_positive_rho=0,aggregate_border_reserve=0;
  double aggregate_backend_assembly_seconds=0,aggregate_backend_symbolic_seconds=0;
  double aggregate_backend_numeric_seconds=0,aggregate_backend_solve_seconds=0;
  int aggregate_scatter_builds=0,aggregate_scatter_reuses=0,aggregate_scatter_invalidations=0;
  std::uint64_t model_layout_builds=0,original_point_calls=0,original_point_value_reuses=0;
  double model_layout_seconds=0,original_point_seconds=0;
  bool aggregate_only_enabled=false,aggregate_affine_retry_enabled=false;
  int aggregate_only_retry_budget=0,aggregate_only_attempts=0,aggregate_only_exhaustions=0;
  int aggregate_only_inertia_rejections=0,aggregate_only_step_rejections=0;
  double aggregate_only_rho_initial=0,aggregate_only_rho_growth=0,aggregate_only_rho_decay=0,aggregate_only_rho_cap=0;
  bool rho_complete_search_enabled=false,strict_finish_enabled=false;
  int rho_grid_completions=0;
  int strict_finish_budget=0,strict_finish_refinement_budget=0;
  double strict_finish_eta=0,strict_finish_mu_fraction=0,strict_finish_mu_target=0;
  int strict_finish_attempts=0,strict_finish_acceptances=0,strict_finish_rejections=0;
  int strict_finish_refinements=0,strict_finish_certificates=0,strict_finish_solved=0;
  double strict_finish_seconds=0,strict_finish_linear_residual=0;
  double strict_finish_error_before=0,strict_finish_error_after=0;
  std::string strict_finish_last_rejection;
  bool aggregate_preparation_cache_enabled=false,aggregate_strict_probe_stop_enabled=false;
  int aggregate_preparation_requests=0,aggregate_preparation_builds=0,aggregate_preparation_hits=0;
  int aggregate_preparation_invalidations=0;
  double aggregate_preparation_seconds=0,aggregate_preparation_key_seconds=0;
  double aggregate_preparation_geometry_seconds=0,aggregate_preparation_operator_seconds=0;
  std::uint64_t aggregate_preparation_peak_bytes=0;
  int aggregate_strict_probe_checks=0,aggregate_strict_probe_successes=0;
  int aggregate_strict_probe_failures=0,aggregate_strict_probe_skipped=0;
  double aggregate_strict_probe_seconds=0;
  int aggregate_trials=0,aggregate_acceptances=0,aggregate_certificates=0,aggregate_pairs=0,aggregate_fill=0,aggregate_soc=0;
  double aggregate_seconds=0,aggregate_residual=0,aggregate_factor_seconds=0;
  double aggregate_max_accepted_residual=0,aggregate_last_rho=0,aggregate_max_accepted_rho=0;
  int aggregate_linear_solves=0,aggregate_symbolic_analyses=0,aggregate_symbolic_reuses=0,aggregate_rejections=0;
  std::string aggregate_backend,aggregate_last_rejection;
  bool krylov_enabled=false;
  int krylov_budget=0,krylov_cooldown=0;
  bool krylov_predictor_corrector=false;
  int krylov_trials=0,krylov_acceptances=0,krylov_iterations=0,krylov_solves=0;
  int krylov_linear_rejections=0,krylov_filter_rejections=0,krylov_skips=0;
  int krylov_operator_products=0,krylov_preconditioner_solves=0;
  double krylov_seconds=0,krylov_max_accepted_residual=0;
  double krylov_first_accept_seconds=-1;
  int krylov_first_accept_iteration=-1;
  std::string krylov_last_rejection;
  int krylov_last_dimension=0,krylov_last_lifted_rows=0,krylov_last_iterations=0;
  double krylov_last_original_residual=INFINITY,krylov_last_reduced_residual=INFINITY;
  double krylov_last_dual_residual=INFINITY,krylov_last_equality_residual=INFINITY;
  double krylov_last_primal_residual=INFINITY,krylov_last_complementarity_residual=INFINITY;
  std::string krylov_last_linear_message;
  int local_trials=0,local_certificates=0,local_acceptances=0,local_pairs=0,local_fill=0,local_strong=0,local_soc=0;
  double local_seconds=0,local_residual=0;
  bool used=false;
  bool tangent_certificate=false;
  bool active_evaluation=false,packed_atoms=false,lazy_merit=false,block_ritz=false,anderson=false;
  bool certificate_transport=false,sparse_fallback=false,equilibration=false;
  std::string curvature_backend;
  std::string pivoted_backend;
  int anderson_trials=0,anderson_acceptances=0;
  int certificate_transport_trials=0,certificate_transports=0;
  bool initial_restoration_requested=false,normal_atom_weighting=false;
  int rho_refinement_budget=0,rho_refinement_trials=0,rho_refinement_acceptances=0;
  int model_reuses=0,maximum_model_steps=0,current_model_steps=0;
  int basis_compressions=0,retired_directions=0,cumulative_atoms_lagrangian=0,peak_stored_atoms=0;
  // R7D13: effective atomic search-space controls of the run.
  int atom_basis_keep=32,atom_basis_tail=8,atom_probe_depth=18;bool atom_probe_count_projection=true;
  int extrapolation_trials=0,extrapolation_acceptances=0;
  double extrapolation_seconds=0;
  int curvature_factor_reuses=0,subspace_spectra=0,dependent_witnesses=0;
  int curvature_ritz_shift_proposals=0,curvature_gershgorin_fallbacks=0;
  int corrector_steps=0,centering_trials=0,original_line_trials=0,original_line_acceptances=0;
  int original_line_rejections=0,private_line_trials=0,private_line_rejections=0;
  double curvature_ensure_seconds=0,curvature_factor_seconds=0,curvature_inverse_seconds=0,subspace_seconds=0;
  double last_penalty=0,last_sigma=0;
  int models=0,hessian_updates=0,inner_steps=0,filter_trials=0,filter_rejections=0;
  int private_continuations=0,model_retries=0,soc_attempts=0,soc_acceptances=0;
  int restorations=0,atom_events=0,atoms_lagrangian=0,atoms_inequalities=0;
  int curvature_factors=0,curvature_reuses=0,first_step_acceptances=0;
  int spectral_inverse_solves=0,small_block_spectra=0,atoms_active_lagrangian=0,atoms_active_inequalities=0;
  double normal_completion_weight=0,eliminated_normal_completion_weight=0;
  double rho_lagrangian=0,rho_inequalities_max=0,model_build_seconds=0;
  double original_filter_seconds=0,curvature_seconds=0,soc_seconds=0;
};
struct RankPresolveStatistics {
  // Populated only for the explicit same-configuration exact-fixed retry.
  // Work/cost counters are cumulative; geometric/controller end state is
  // that of the final attempt. Keep primary costs available independently.
  bool exact_fixed_retry_attempted=false;
  std::string exact_fixed_primary_status,exact_fixed_retry_status;
  double exact_fixed_primary_seconds=0,exact_fixed_retry_seconds=0;
  double exact_fixed_primary_core_seconds=0,exact_fixed_primary_restoration_seconds=0;
  int exact_fixed_primary_outer_iterations=0,exact_fixed_retry_outer_iterations=0;
  int exact_fixed_primary_restoration_iterations=0,exact_fixed_retry_restoration_iterations=0;
  int exact_fixed_primary_factor_attempts=0,exact_fixed_retry_factor_attempts=0;
  int exact_fixed_primary_linear_solves=0,exact_fixed_retry_linear_solves=0;
  std::size_t exact_fixed_primary_history_records=0;
  std::array<double,13> exact_fixed_primary_cost_seconds{};
  bool enabled=false,applied=false,initial_point_preserved=true,postsolve_valid=false;
  int original_variables=0,reduced_variables=0,original_equalities=0,reduced_equalities=0;
  int original_inequalities=0,reduced_inequalities=0;
  int leaf_candidates=0,leaf_pairs=0,nonzero_rejected=0,start_rejected=0;
  int arithmetic_rejected=0,duplicate_equalities=0,zero_equalities=0;
  int duplicate_inequalities=0,constant_inequalities=0,duplicate_bounds=0,implied_bound_sides=0;
  double preparation_seconds=0,postsolve_seconds=0,initial_lift_error=0;
  double lifted_dual_identity_error=0;
  std::string reason="disabled";
  std::string audit_mode="none";
  int audit_shared_values=0,audit_recipe_exceptions=0,audit_duplicate_rows_covered=0,audit_unique_terms=0;
  std::uint64_t audit_ordered_sums=0,audit_term_evaluations=0,audit_term_reuses=0;
  std::uint64_t audit_calls=0,audit_mapped_calls=0,audit_full_calls=0,audit_check_calls=0;
  std::uint64_t audit_value_evaluations=0,audit_value_reuses=0,audit_gradient_evaluations=0;
  std::uint64_t audit_result_copies=0,audit_lift_calls=0,audit_final_full_calls=0;
  std::uint64_t audit_coordinate_expansions=0;
  double audit_seconds=0;
};

// A bounded native Lazy retry is a separate numerical phase, not an exact
// coefficient-reduction retry. Endpoint fields always describe the retained
// phase; costs and designated work counters include both phases.
struct NativeBarrierRetryStatistics {
  bool attempted=false,accepted=false;
  std::string primary_status,retry_status;
  double primary_seconds=0,retry_seconds=0;
  int primary_outer_iterations=0,retry_outer_iterations=0;
  int primary_restoration_iterations=0,retry_restoration_iterations=0;
  int primary_factor_attempts=0,retry_factor_attempts=0;
  int primary_linear_solves=0,retry_linear_solves=0;
  std::array<double,13> primary_cost_seconds{},retry_cost_seconds{};
};

struct FreeEpigraphStatistics {
  bool enabled=false,applied=false,initial_point_preserved=true,postsolve_valid=false;
  int original_variables=0,reduced_variables=0,original_inequalities=0,reduced_inequalities=0;
  int candidates=0,selected=0,bounds_rejected=0,start_rejected=0,ambiguous_rejected=0;
  int checkpoint_lifts=0;
  double preparation_seconds=0,postsolve_seconds=0,checkpoint_seconds=0;
  std::string reason="disabled";
};

struct NativeAugmentedStatistics {
  bool enabled=false,admitted=false,initial_projection=false;
  int original_variables=0,reduced_variables=0,stages=0,accepted_steps=0;
  int factorizations=0,linear_solves=0;
  double preparation_seconds=0,core_seconds=0,certificate_seconds=0;
  double initial_projection_inf=0;
  std::string reason,seed_policy;
};

struct AdaptiveEqualityProbeStatistics {
  bool eligible=false,attempted=false,transitioned=false;
  bool probe_strict=false,stabilized_strict=false,within_global_budget=true;
  int probe_outer_budget=0,stabilized_outer_budget=0,global_outer_budget=0;
  int probe_outer=0,stabilized_outer=0,probe_restoration=0,stabilized_restoration=0;
  int global_restoration_budget=0,stabilized_restoration_budget=0;
  int probe_factorizations=0,stabilized_factorizations=0;
  double global_seconds=0,probe_seconds=0,stabilized_seconds=0,total_seconds=0;
  double selection_audit_seconds=0,probe_raw_residual=INFINITY,stabilized_raw_residual=INFINITY;
  std::string probe_status="not-run",stabilized_status="not-run",selected_phase="not-run";
  std::string reason="ineligible-or-not-requested";
  std::array<double,13> probe_cost_seconds{},stabilized_cost_seconds{};
};

struct Result {
  AdaptiveEqualityProbeStatistics adaptive_equality_probe;
  bool positive_square_epigraph_substitution=false;
  bool common_factor_substitution=false;
  bool affine_infeasibility_certificate=false;
  bool convex_infeasibility_certificate=false,convex_infeasibility_lower_side=false;
  int convex_infeasibility_equality_sign=0,convex_infeasibility_bound_variable=-1;
  std::string convex_infeasibility_minimum,convex_infeasibility_weight_norm,convex_infeasibility_lower_bound,convex_infeasibility_bound_weight;
  std::vector<int> convex_infeasibility_rows;
  std::vector<std::string> convex_infeasibility_weights;
  std::vector<int> affine_infeasibility_rows;
  std::vector<std::string> affine_infeasibility_weights;
  std::string affine_infeasibility_constant,affine_infeasibility_primal_lower_bound;
  int common_factor_variable=-1;
  int positive_square_epigraph_variable=-1;
  bool coefficient_graph_certificate=false;
  int coefficient_graph_vertices=0,coefficient_graph_edges=0;
  int coefficient_graph_components=0,coefficient_graph_independent=0;
  bool gr_adjustments_active=false;
  bool gr_coefficient_constraint_increments=false,gr_residual_components=false;
  std::string gr_proximal_policy="historical";
  int gr_fixed_centering_steps=0;
  bool standard_auto_enabled=false,standard_auto_exact_global_faces=false;
  bool standard_auto_rank_presolve_requested=false,standard_auto_rank_presolve_effective=false;
  bool standard_auto_free_epigraph_requested=false,standard_auto_free_epigraph_effective=false;
  bool standard_auto_rank_presolve_skipped=false,standard_auto_free_epigraph_presolve_skipped=false;
  bool standard_auto_checkpoint_requested=false,standard_auto_checkpoint_effective=false;
  bool standard_auto_builtin_checkpoint_skipped=false;
  double standard_auto_configured_initial_mu=0;
  std::string standard_auto_requested_startup,standard_auto_effective_startup;
  bool standard_auto_exact_global_faces_applied=false;
  double standard_auto_exact_face_start_projection_inf=0;
  std::string standard_auto_exact_global_faces_reason="disabled";
  std::string standard_auto_mode="unresolved",standard_auto_reason;
  std::string standard_auto_requested_engine,standard_auto_effective_engine,standard_auto_configuration;
  NativeALFallbackStats native_al_fallback;
  NativeAugmentedStatistics native_augmented;
  FreeEpigraphStatistics free_epigraph;
  NativeBarrierRetryStatistics native_barrier_retry;
  RankPresolveStatistics rank_presolve;
  std::string termination_test = "project";
  bool termination_scaling_available = false;
  double termination_objective_scale = 1;
  Vec termination_equality_scales, termination_inequality_scales;
  // Positive upper-slack distances in ORIGINAL constraint units, transported
  // from a live DC model back to the original inequality, not replaced by -g.
  Vec termination_inequality_slacks;
  double termination_internal_dual_inf = INFINITY;
  double termination_internal_primal_inf = INFINITY;
  double termination_internal_compl_inf = INFINITY;
  double termination_dual_inf = INFINITY, termination_compl_inf = INFINITY;
  double original_normalized_nlp_error = INFINITY;
  double termination_previous_objective = -1e50;
  double termination_current_objective = -1e50;
  double termination_objective_change = INFINITY;
  int termination_objective_iteration = -1;
  bool termination_square_problem = false;

  bool acceptable_convergence=false,acceptable_point=false,termination_audit_valid=false;
  bool acceptable_on_failure=false;
  int acceptable_streak=0,acceptable_required=0,fixed_multipliers_reconstructed=0;
  std::string acceptable_trigger,original_failure_status,original_failure_message;
  double raw_kkt_error=INFINITY,acceptable_error=INFINITY,termination_sd=1,termination_sc=1;
  DcaFilterStatistics dca_filter;

  LazyNewtonStatistics lazy_newton;
  bool atomic_engine_used=false;
  double cli_read_seconds=0.0,cli_solver_call_seconds=0.0,cli_arguments_seconds=0.0;
  Status status = Status::NumericalError;
  std::string message;
  std::string linear_solver_backend = "native_dense_lapack";
  std::string effective_outer_algorithm = "unresolved";
  std::string dca_dual_step = "not-applicable";
  Vec x;
  Vec certificate_equality_multipliers;
  Vec certificate_inequality_multipliers;
  Vec certificate_lower_multipliers;
  Vec certificate_upper_multipliers;
  double objective = 0.0;
  Feasibility feasibility;
  // Historical projected-stationarity trajectory diagnostic.
  double model_stationarity_inf = std::numeric_limits<double>::infinity();

  // Fresh original-QCQP primal-dual termination certificate.
  // A failed/infeasible run may never construct a valid original-QCQP
  // multiplier certificate. Nonfinite defaults serialize as JSON null rather
  // than masquerading as a perfect zero residual.
  double dual_infeasibility_inf = std::numeric_limits<double>::infinity();
  double complementarity_inf = std::numeric_limits<double>::infinity();
  double scaled_nlp_error = std::numeric_limits<double>::infinity();
  double dual_error_scaling = 1.0;
  double complementarity_error_scaling = 1.0;
  double constraint_multiplier_l1 = 0.0;
  double bound_slack_multiplier_l1 = 0.0;
  int ipopt_dual_multiplier_dimension = 0;
  int ipopt_complementarity_multiplier_dimension = 0;
  int ipopt_y_c_dimension = 0;
  int ipopt_y_d_dimension = 0;
  int ipopt_z_L_dimension = 0;
  int ipopt_z_U_dimension = 0;
  int ipopt_v_L_dimension = 0;
  int ipopt_v_U_dimension = 0;
  double equality_multiplier_inf = 0.0;
  double inequality_multiplier_inf = 0.0;
  double bound_multiplier_inf = 0.0;
  bool ipopt_desired_convergence = false;
  std::string termination_multiplier_source = "unavailable";
  bool termination_multiplier_fit_converged = false;
  int termination_multiplier_fit_sweeps = 0;
  double termination_multiplier_fit_projected_gradient_inf = std::numeric_limits<double>::infinity();
  double termination_multiplier_fit_stationarity_inf = std::numeric_limits<double>::infinity();
  int termination_multiplier_refinement_attempts = 0;
  double termination_multiplier_regularization = std::numeric_limits<double>::infinity();
  bool termination_certificate_rescued = false;

  // Sparse original-QCQP multiplier projection/transport diagnostics.
  int multiplier_refit_attempts = 0;
  int multiplier_refit_successes = 0;
  int multiplier_refit_fallback_reuses = 0;
  int multiplier_row_space_factorizations = 0;
  int multiplier_row_space_solves = 0;
  int multiplier_row_space_symbolic_analyses = 0;
  int multiplier_row_space_symbolic_reuses = 0;
  int multiplier_row_space_failures = 0;
  double last_multiplier_refit_stationarity_inf =
      std::numeric_limits<double>::infinity();
  int dca_single_loop_refits_skipped = 0;
  int dca_single_loop_refit_fallbacks = 0;
  // Forbidden original-Jacobian full fits used to initialize the first
  // majorant. A conforming single-loop trajectory keeps this at zero and uses
  // the reduced first-model bootstrap reported below instead.
  int dca_single_loop_bootstrap_full_fits = 0;
  // Logical full multiplier fits performed by the single-loop route.  The
  // value must remain zero throughout the trajectory and may become one only
  // for the independent terminal certificate (whose internal regularization
  // ladder is counted separately above).
  int dca_single_loop_full_refit_calls = 0;
  int dca_single_loop_retractions_skipped = 0;
  int dca_single_loop_retraction_fallbacks = 0;

  double step_inf = 0.0;

  double last_inner_dual_inf = 0.0;
  double last_inner_equality_inf = 0.0;
  double last_inner_inequality_inf = 0.0;
  double last_inner_primal_residual_inf = 0.0;
  double last_inner_complementarity = 0.0;
  double last_inner_scaled_model_residual_inf = 0.0;
  bool last_inner_used_iteration_limit_fallback = false;
  double final_kkt_rho_x = 0.0;
  double max_kkt_rho_x = 0.0;
  double final_kkt_delta_d = 0.0;
  double max_kkt_delta_d = 0.0;
  double max_kkt_backward_error = 0.0;
  double max_kkt_residual_inf = 0.0;
  int kkt_regularization_retries = 0;
  int kkt_refinement_steps = 0;
  int equality_qp_original_kkt_gmres_attempts = 0;
  int equality_qp_original_kkt_gmres_acceptances = 0;
  int equality_qp_original_kkt_gmres_iterations = 0;
  int inner_dual_warm_start_acceptances = 0;
  int inner_full_dual_warm_start_acceptances = 0;
  int inner_dual_warm_start_rejections = 0;
  int lazyqp_zero_dual_continuation_attempts = 0;
  int lazyqp_zero_dual_continuation_mapped_attempts = 0;
  int lazyqp_zero_dual_continuation_direct_attempts = 0;
  int lazyqp_zero_dual_continuation_acceptances = 0;
  int lazyqp_zero_dual_continuation_fallbacks = 0;
  int lazyqp_zero_dual_continuation_repairs = 0;
  int working_set_primal_transports = 0;
  int lazyqp_reduced_kkt_factorizations = 0;
  int lazyqp_full_kkt_factorizations = 0;
  int lazyqp_reduced_kkt_fallbacks = 0;
  int lazyqp_hybrid_kkt_attempts = 0;
  int lazyqp_hybrid_kkt_factorizations = 0;
  int lazyqp_hybrid_kkt_fallbacks = 0;
  int lazyqp_hybrid_kkt_retained_rows_total = 0;
  int lazyqp_hybrid_kkt_retained_rows_max = 0;
  long long lazyqp_hybrid_kkt_outer_product_triplets_avoided = 0;
  long long lazyqp_hybrid_kkt_border_triplets_emitted = 0;
  int lazyqp_kkt_triplet_layout_builds = 0;
  int lazyqp_kkt_triplet_layout_reuses = 0;
  int lazyqp_kkt_triplet_layout_invalidations = 0;
  int lazyqp_numeric_kkt_reuse_attempts = 0;
  int lazyqp_numeric_kkt_reuse_acceptances = 0;
  int lazyqp_numeric_kkt_reuse_fallbacks = 0;
  int lazyqp_numeric_kkt_reuse_iterations = 0;
  int lazyqp_max_kkt_dimension = 0;
  int lazyqp_max_full_kkt_dimension = 0;
  int lazyqp_affine_elimination_attempts = 0;
  int lazyqp_affine_elimination_acceptances = 0;
  int lazyqp_affine_elimination_fallbacks = 0;
  int lazyqp_affine_eliminated_variables = 0;
  int lazyqp_affine_eliminated_equalities = 0;
  int lazyqp_affine_reduced_variables = 0;
  int lazyqp_affine_reduced_equalities = 0;
  int socp_formulation_attempts = 0;
  int socp_formulation_builds = 0;
  int socp_formulation_rejections = 0;
  int socp_formulation_audit_failures = 0;
  int socp_objective_rotated_cones = 0;
  int socp_inequality_rotated_cones = 0;
  int socp_linear_nonnegative_cones = 0;
  int socp_max_affine_equalities = 0;
  long long socp_max_explicit_lift_cone_dimension = 0;
  int socp_condensed_newton_iterations = 0;
  double socp_max_rotated_cone_violation = 0.0;

  int restoration_iterations = 0;
  int restoration_restarts = 0;
  int restoration_linear_solves = 0;
  int restoration_exact_hessian_solves = 0;
  int restoration_matrix_free_solves = 0;
  int restoration_row_space_solves = 0;
  int restoration_harmonic_attempts = 0;
  int restoration_harmonic_acceptances = 0;
  int restoration_harmonic_factorizations = 0;
  int restoration_tangent_activation_attempts = 0;
  int restoration_tangent_activation_acceptances = 0;
  int restoration_tangent_activation_components = 0;
  int restoration_cg_iterations = 0;
  int restoration_exact_workspace_growths = 0;
  int restoration_exact_matvecs = 0;
  int restoration_exact_parallel_row_dot_batches = 0;
  int restoration_exact_parallel_teams = 0;
  int restoration_exact_sparse_row_dot_batches = 0;
  int restoration_exact_sparse_row_dot_verifications = 0;
  int restoration_exact_sparse_row_dot_fallbacks = 0;
  int restoration_line_search_backtracks = 0;
  int restoration_stagnation_events = 0;
  int restoration_minimax_qp_solves = 0;
  int restoration_minimax_inner_iterations = 0;
  int restoration_minimax_factorizations = 0;
  int restoration_minimax_accepted_steps = 0;
  int restoration_minimax_line_search_backtracks = 0;
  int restoration_structured_abs_links = 0;
  int restoration_effective_dimension = 0;
  double restoration_initial_feasibility = 0.0;
  double restoration_harmonic_feasibility = 0.0;
  double restoration_tangent_activation_feasibility = 0.0;
  double restoration_tangent_activation_step_inf = 0.0;
  double restoration_harmonic_seconds = 0.0;
  double restoration_tangent_activation_seconds = 0.0;
  double restoration_legacy_gate_seconds = 0.0;
  double restoration_exact_seconds = 0.0;
  double restoration_minimax_seconds = 0.0;
  double restoration_restart_seconds = 0.0;
  double restoration_total_seconds = 0.0;
  int restoration_tangent_activation_gradient_rows = 0;
  int restoration_tangent_activation_line_quadratics = 0;
  int restoration_tangent_activation_candidates = 0;
  int restoration_harmonic_unknowns = 0;
  int restoration_harmonic_graph_edges = 0;
  double restoration_final_feasibility = 0.0;

  double final_proximal_regularization = 0.0;
  int outer_iterations = 0;
  int inner_iterations = 0;
  int kkt_factorizations = 0;
  int kkt_symbolic_analyses = 0;
  int kkt_symbolic_reuses = 0;
  int kkt_symbolic_invalidations = 0;
  int kkt_numeric_scatter_builds = 0;
  int kkt_numeric_scatter_reuses = 0;
  int kkt_numeric_scatter_invalidations = 0;
  int inner_iteration_limit_fallback_acceptances = 0;
  int lazyqp_inexact_forcing_attempts = 0;
  int lazyqp_inexact_forcing_acceptances = 0;
  int lazyqp_inexact_forcing_strict_fallbacks = 0;
  int lazyqp_inexact_forcing_outer_retries = 0;
  int lazyqp_inexact_forcing_model_evaluations = 0;
  double lazyqp_inexact_forcing_max_target = 0.0;
  int lazyqp_model_support_fastpath_requests = 0;
  int lazyqp_model_support_fastpath_effective_subproblems = 0;
  int lazyqp_model_support_builds = 0;
  int lazyqp_model_support_eligible_models = 0;
  int lazyqp_model_support_generic_fallback_models = 0;
  int lazyqp_model_support_value_evaluations = 0;
  int lazyqp_model_support_gradient_evaluations = 0;
  int dca_single_loop_attempts = 0;
  int dca_single_loop_acceptances = 0;
  int dca_single_loop_strict_fallbacks = 0;
  int dca_single_loop_outer_retries = 0;
  int dca_single_loop_state_transport_acceptances = 0;
  int dca_single_loop_state_transport_rejections = 0;
  int dca_single_loop_reduced_bootstrap_attempts = 0;
  int dca_single_loop_reduced_bootstrap_acceptances = 0;
  int dca_single_loop_keyed_transport_attempts = 0;
  int dca_single_loop_keyed_transport_acceptances = 0;
  int dca_single_loop_keyed_transport_rejections = 0;
  int dca_single_loop_keyed_transport_churn_acceptances = 0;
  int dca_single_loop_keyed_transport_matched_rows = 0;
  int dca_single_loop_keyed_transport_new_rows = 0;
  int dca_single_loop_keyed_transport_dropped_rows = 0;
  int dca_single_loop_keyed_transport_centrality_caps = 0;
  int dca_single_loop_inner_steps = 0;
  int dca_single_loop_factorizations = 0;
  double dca_single_loop_max_initial_residual = 0.0;
  double dca_single_loop_max_final_residual = 0.0;
  double dca_single_loop_baseline_proximal_regularization = 0.0;
  int dca_single_loop_direct_start_attempts = 0;
  int dca_single_loop_direct_start_acceptances = 0;
  int dca_single_loop_direct_start_rejections_nonfinite = 0;
  int dca_single_loop_direct_start_rejections_bounds = 0;
  int dca_single_loop_direct_start_rejections_equalities = 0;
  int dca_single_loop_direct_start_rejections_theta = 0;
  int dca_single_loop_direct_start_violated_inequalities = 0;
  int dca_single_loop_direct_start_working_set_rows = 0;
  int dca_single_loop_direct_start_filter_initializations = 0;
  double dca_single_loop_direct_start_initial_theta =
      std::numeric_limits<double>::infinity();
  int dca_single_loop_direct_start_recovery_attempts = 0;
  int dca_single_loop_direct_start_recovery_acceptances = 0;
  int dca_single_loop_direct_start_recovery_rejections_nonfinite = 0;
  int dca_single_loop_direct_start_recovery_rejections_bounds = 0;
  int dca_single_loop_direct_start_recovery_rejections_cap = 0;
  int dca_single_loop_direct_start_recovery_rejections_contraction = 0;
  int dca_single_loop_direct_start_recovery_rejections_unmodeled = 0;
  int dca_single_loop_direct_start_recovery_working_set_additions = 0;
  int dca_single_loop_direct_start_strict_latches = 0;
  double dca_single_loop_direct_start_recovery_max_theta = 0.0;
  int line_search_backtracks = 0;
  int armijo_rejections = 0;
  // Equality-model diagnostics.
  std::string lazy_dca_equality_model = "linearized";
  int lazy_dca_elastic_subproblems = 0;
  int lazy_dca_elastic_accepted_steps = 0;
  double lazy_dca_elastic_last_eta = 0.0;

  // Single-route Lazy-DCA diagnostics for SQP-linearized nonlinear equalities.
  int lazy_dca_linearized_equality_steps = 0;
  double lazy_dca_last_equality_step_alpha = 1.0;
  int lazy_dca_equality_objective_steps = 0;
  int lazy_dca_equality_feasibility_steps = 0;
  double lazy_dca_final_equality_funnel = 0.0;
  int lazy_dca_equality_retraction_attempts = 0;
  int lazy_dca_equality_retraction_acceptances = 0;
  int lazy_dca_equality_retraction_rejections = 0;
  double lazy_dca_max_equality_retraction_inf = 0.0;

  // Curvature-preserving condensed quadratic equality route.
  int lazy_dca_condensed_factorizations = 0;
  int lazy_dca_condensed_bound_face_factorizations = 0;
  int lazy_dca_condensed_max_active_bound_coordinates = 0;
  int lazy_dca_condensed_normal_solves = 0;
  int lazy_dca_condensed_geometry_normal_solves = 0;
  int lazy_dca_condensed_retraction_normal_solves = 0;
  int lazy_dca_condensed_symbolic_analyses = 0;
  int lazy_dca_condensed_symbolic_reuses = 0;
  int lazy_dca_condensed_symbolic_invalidations = 0;
  int lazy_dca_condensed_retraction_attempts = 0;
  int lazy_dca_condensed_retraction_acceptances = 0;
  // Trials for which the condensed equation was solved are not necessarily
  // accepted by the outer objective/filter globalization.  Count those final
  // accepted manifold steps separately.
  int lazy_dca_condensed_accepted_steps = 0;
  int lazy_dca_condensed_retraction_rejections = 0;
  int lazy_dca_condensed_continuation_attempts = 0;
  int lazy_dca_condensed_continuation_acceptances = 0;
  int lazy_dca_condensed_newton_iterations = 0;
  int lazy_dca_condensed_krylov_iterations = 0;
  int lazy_dca_condensed_backtracks = 0;
  int lazy_dca_condensed_affine_normal_map_reuses = 0;
  int lazy_dca_condensed_newton_endpoint_reuses = 0;
  int lazy_dca_condensed_quadratic_linesearch_reuses = 0;
  int lazy_dca_condensed_quadratic_linesearch_predictor_builds = 0;
  int lazy_dca_condensed_quadratic_linesearch_exact_checks = 0;
  int lazy_dca_condensed_quadratic_linesearch_early_rejections = 0;
  long long lazy_dca_condensed_quadratic_linesearch_rows_skipped = 0;
  int lazy_dca_condensed_cached_regularization_attempts = 0;
  int lazy_dca_condensed_cached_regularization_hits = 0;
  int lazy_dca_condensed_sparse_equality_evaluations = 0;
  int lazy_dca_condensed_parallel_equality_batches = 0;
  int lazy_dca_condensed_normal_refinement_steps = 0;
  int lazy_dca_condensed_geometry_normal_refinement_steps = 0;
  int lazy_dca_condensed_retraction_normal_refinement_steps = 0;
  int lazy_dca_condensed_normal_refinement_zero_solves = 0;
  int lazy_dca_condensed_normal_refinement_one_solves = 0;
  int lazy_dca_condensed_normal_refinement_two_solves = 0;
  int lazy_dca_condensed_normal_refinement_three_plus_solves = 0;
  int lazy_dca_condensed_relaxed_normal_solves = 0;
  int lazy_dca_condensed_strict_normal_solves = 0;
  int lazy_dca_condensed_relaxed_accepts_above_strict_tolerance = 0;
  int lazy_dca_condensed_strict_retry_retractions = 0;
  double lazy_dca_condensed_max_normal_initial_backward_error = 0.0;
  double lazy_dca_condensed_mean_normal_initial_backward_error = 0.0;
  double lazy_dca_condensed_mean_normal_final_backward_error = 0.0;
  double lazy_dca_condensed_max_normal_backward_error = 0.0;
  int lazy_dca_condensed_gmres_workspace_growths = 0;
  int lazy_dca_condensed_zero_rhs_solve_skips = 0;
  int lazy_dca_condensed_tangent_curvature_calls = 0;
  int lazy_dca_condensed_block_exact_attempts = 0;
  int lazy_dca_condensed_block_exact_acceptances = 0;
  int lazy_dca_condensed_block_exact_blocks_max = 0;
  int lazy_dca_condensed_block_exact_active_dimension_max = 0;
  int lazy_dca_condensed_block_exact_max_block_dimension = 0;
  int lazy_dca_condensed_block_partial_spectrum_attempts = 0;
  int lazy_dca_condensed_block_partial_spectrum_acceptances = 0;
  int lazy_dca_condensed_block_partial_spectrum_fallbacks = 0;
  int lazy_dca_condensed_projected_multi_ritz_batches = 0;
  int lazy_dca_condensed_projected_multi_ritz_fallbacks = 0;
  int lazy_dca_condensed_tangent_inertia_attempts = 0;
  int lazy_dca_condensed_tangent_inertia_acceptances = 0;
  int lazy_dca_condensed_tangent_inertia_shift_updates = 0;
  double lazy_dca_condensed_tangent_inertia_shift_max = 0.0;
  int lazy_dca_condensed_tangent_rank_max = 0;
  int lazy_dca_condensed_scalar_safeguard_updates_max = 0;
  double lazy_dca_condensed_tangent_shift_max = 0.0;
  double lazy_dca_condensed_initial_tangent_min_eigenvalue = 0.0;
  double lazy_dca_condensed_final_tangent_min_eigenvalue = 0.0;
  double lazy_dca_condensed_normal_regularization = 0.0;
  double lazy_dca_condensed_last_residual_inf = 0.0;
  double lazy_dca_condensed_max_correction_inf = 0.0;
  double lazy_dca_condensed_max_correction_ratio = 0.0;
  int lazy_dca_condensed_alpha_predictor_uses = 0;
  int lazy_dca_condensed_alpha_predictor_full_step_resets = 0;
  int lazy_dca_condensed_alpha_predictor_leading_retractions_skipped = 0;
  int lazy_dca_condensed_alpha_predictor_historical_retries = 0;
  int lazy_dca_condensed_outer_extrapolation_attempts = 0;
  int lazy_dca_condensed_outer_extrapolation_acceptances = 0;
  int lazy_dca_condensed_outer_extrapolation_historical_fallbacks = 0;
  int lazy_dca_condensed_outer_extrapolation_full_step_streak_max = 0;
  double lazy_dca_condensed_outer_extrapolation_alpha_max = 0.0;
  int lazy_dca_condensed_outer_reject_bound = 0;
  int lazy_dca_condensed_outer_reject_side = 0;
  int lazy_dca_condensed_outer_reject_funnel = 0;
  int lazy_dca_condensed_outer_reject_filter = 0;
  int lazy_dca_condensed_outer_reject_objective = 0;
  int lazy_dca_condensed_post_retraction_modeled_offenders = 0;
  int lazy_dca_condensed_post_retraction_omitted_offenders = 0;
  int lazy_dca_condensed_post_retraction_working_set_expansions = 0;
  int lazy_dca_condensed_post_retraction_expansion_fallbacks = 0;
  int lazy_dca_condensed_modeled_side_filter_attempts = 0;
  int lazy_dca_condensed_modeled_side_filter_acceptances = 0;
  int lazy_dca_condensed_modeled_side_filter_recovery_acceptances = 0;
  int lazy_dca_condensed_modeled_side_filter_fallbacks = 0;
  int lazy_dca_condensed_modeled_side_filter_recovery_margin_rows = 0;
  double lazy_dca_condensed_modeled_side_filter_max_violation = 0.0;
  int lazy_dca_condensed_flow_lift_schur_detected_variables = 0;
  int lazy_dca_condensed_flow_lift_schur_factor_dimension = 0;
  int lazy_dca_condensed_flow_lift_schur_factorizations = 0;
  int lazy_dca_condensed_flow_lift_schur_fallbacks = 0;

  int lazy_dca_corrector_attempts = 0;
  int lazy_dca_corrector_acceptances = 0;
  int lazy_dca_corrector_rejections = 0;
  int lazy_dca_corrector_inner_iterations = 0;
  int lazy_dca_corrector_cold_retries = 0;
  int lazy_dca_iterated_soc_attempts = 0;
  int lazy_dca_iterated_soc_acceptances = 0;
  int lazy_dca_iterated_soc_rejections = 0;
  int lazy_dca_inner_retries = 0;
  int lazy_dca_cold_retries = 0;
  int lazy_dca_proximal_retries = 0;
  int lazy_dca_step_control_predictions = 0;
  int lazy_dca_step_control_retries = 0;
  int lazy_dca_warm_start_suppressions = 0;
  int lazy_dca_warm_start_skips = 0;
  double lazy_dca_max_raw_direction_inf = 0.0;
  double lazy_dca_last_raw_direction_inf = 0.0;
  double lazy_dca_last_step_target_inf = 0.0;
  double lazy_dca_max_model_proximal = 0.0;
  int trust_region_subproblems = 0;
  int trust_region_accepted_steps = 0;
  int trust_region_rejected_steps = 0;
  int trust_region_contractions = 0;
  int trust_region_expansions = 0;
  int trust_region_feasibility_steps = 0;
  double final_trust_region_radius = 0.0;
  double last_trust_region_ratio = 0.0;
  double last_trust_region_predicted_reduction = 0.0;
  double last_trust_region_actual_reduction = 0.0;
  double last_trust_region_penalty = 0.0;
  double last_trust_region_merit_predicted_reduction = 0.0;
  double last_trust_region_merit_actual_reduction = 0.0;
  int soc_attempts = 0;
  int soc_accepted_steps = 0;
  int soc_rejected_steps = 0;
  int soc_inner_iterations = 0;
  int soc_factorizations = 0;
  int curvature_calls = 0;
  int objective_rank = 0; // v0.4.0: rank of Q0 lazy correction (legacy-sqp: Lagrangian correction)
  int inequality_total_rank = 0;
  int active_inequalities = 0;
  int working_set_additions = 0;
  int working_set_drops = 0;
  int active_jacobian_rank = 0;
  int tangent_dimension = 0;
  int reduced_negative_eigenvalues = 0;
  double reduced_min_eigenvalue = 0.0;
  int terminal_kkt_iterations = 0;
  int terminal_kkt_accepted_steps = 0;
  int terminal_kkt_active_set_drops = 0; // legacy v0.2.10 counter; always zero since v0.2.11
  int terminal_kkt_attempts = 0;
  int terminal_kkt_rejected_steps = 0;
  int terminal_kkt_skipped_unstable = 0;
  int terminal_kkt_active_stability = 0;
  double terminal_kkt_multiplier_cutoff = 0.0;
  double elapsed_seconds = 0.0;
  double constraint_curvature_seconds = 0.0;
  double equality_model_seconds = 0.0;
  double primary_model_build_seconds = 0.0;
  double primary_inner_seconds = 0.0;
  int equality_tangent_inertia_attempts = 0;
  int equality_tangent_inertia_acceptances = 0;
  int equality_tangent_stall_certificates = 0;
  // v0.4.28r3p28 lazy-newton counters (zero for the historical routes).
  int lazy_newton_iterations = 0;
  int lazy_newton_factorizations = 0;
  int lazy_newton_symbolic_analyses = 0;
  int lazy_newton_inertia_corrections = 0;
  int lazy_newton_refinement_sweeps = 0;
  int lazy_newton_quadratic_corrections = 0;
  int lazy_newton_quadratic_correction_acceptances = 0;
  int lazy_newton_lazy_cuts = 0;
  int lazy_newton_condensed_rows_max = 0;
  int lazy_newton_augmented_rows_max = 0;
  int lazy_newton_proximal_shifts = 0;
  int lazy_newton_exact_curvature_steps = 0;
  int lazy_newton_majorant_curvature_steps = 0;
  int lazy_newton_curvature_fallbacks = 0;
  int lazy_newton_dual_regularizations = 0;
  int lazy_newton_working_set_final = 0;
  int lazy_newton_working_set_max = 0;
  int lazy_newton_kkt_dimension = 0;
  int lazy_newton_kkt_nnz = 0;
  int lazy_newton_factor_nnz = 0;
  int lazy_newton_barrier_updates = 0;
  int lazy_newton_line_search_backtracks = 0;
  int lazy_newton_certificate_evaluations = 0;
  double lazy_newton_final_mu = 0.0;
  double lazy_newton_objective_scale = 1.0;
  double lazy_newton_max_delta_x = 0.0;
  double lazy_newton_setup_seconds = 0.0;
  double lazy_newton_factorization_seconds = 0.0;
  double lazy_newton_assembly_seconds = 0.0;
  double lazy_newton_structure_seconds = 0.0;
  double lazy_newton_evaluation_seconds = 0.0;
  double lazy_newton_line_search_seconds = 0.0;
  double lazy_newton_certificate_seconds = 0.0;
};

} // namespace lazyipdca::qcqp
