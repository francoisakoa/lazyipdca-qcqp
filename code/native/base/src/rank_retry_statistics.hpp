#pragma once
#include "lazyipdca_qcqp/result.hpp"
#include <algorithm>
#include <utility>

namespace lazyipdca::qcqp::detail {

// Called ONLY after a real second native solve. The default annotations are
// reserved for the historical exact-fixed retry in the SAME configuration.
// Native barrier recovery passes annotate_rank_retry=false and records its
// explicit separate provenance through set_native_barrier_retry_statistics.
// Final point/certificate, dimensions, ranks, parameters and controller state
// remain those of the retry. Actual costs, function calls and the designated
// cumulative work counters below include BOTH attempts. Primary phase costs
// are independently serialized so the recovery cannot hide its first solve.
inline void set_native_barrier_retry_statistics(Result& final,
    const Result& primary,const Result& retry,bool accepted) {
  auto& report=final.native_barrier_retry;
  report.attempted=true;report.accepted=accepted;
  report.primary_status=status_string(primary.status);
  report.retry_status=status_string(retry.status);
  report.primary_seconds=primary.elapsed_seconds;report.retry_seconds=retry.elapsed_seconds;
  report.primary_outer_iterations=primary.outer_iterations;
  report.retry_outer_iterations=retry.outer_iterations;
  report.primary_restoration_iterations=primary.restoration_iterations;
  report.retry_restoration_iterations=retry.restoration_iterations;
  report.primary_factor_attempts=primary.lazy_newton.factor_attempts;
  report.retry_factor_attempts=retry.lazy_newton.factor_attempts;
  report.primary_linear_solves=primary.lazy_newton.linear_solves;
  report.retry_linear_solves=retry.lazy_newton.linear_solves;
  report.primary_cost_seconds=primary.lazy_newton.cost_seconds;
  report.retry_cost_seconds=retry.lazy_newton.cost_seconds;
}

inline void merge_rank_retry_work(Result& final,const Result& primary,
    bool annotate_rank_retry=true,bool other_precedes_final=true) {
  auto& report=final.rank_presolve;
  if(annotate_rank_retry) {
  report.exact_fixed_retry_attempted=true;
  report.exact_fixed_primary_status=status_string(primary.status);
  report.exact_fixed_retry_status=status_string(final.status);
  report.exact_fixed_primary_seconds=primary.elapsed_seconds;
  report.exact_fixed_retry_seconds=final.elapsed_seconds;
  report.exact_fixed_primary_core_seconds=primary.lazy_newton.core_solve_seconds;
  report.exact_fixed_primary_restoration_seconds=primary.restoration_total_seconds;
  report.exact_fixed_primary_outer_iterations=primary.outer_iterations;
  report.exact_fixed_retry_outer_iterations=final.outer_iterations;
  report.exact_fixed_primary_restoration_iterations=primary.restoration_iterations;
  report.exact_fixed_retry_restoration_iterations=final.restoration_iterations;
  report.exact_fixed_primary_factor_attempts=primary.lazy_newton.factor_attempts;
  report.exact_fixed_retry_factor_attempts=final.lazy_newton.factor_attempts;
  report.exact_fixed_primary_linear_solves=primary.lazy_newton.linear_solves;
  report.exact_fixed_retry_linear_solves=final.lazy_newton.linear_solves;
  report.exact_fixed_primary_history_records=primary.lazy_newton.iteration_history.size();
  report.exact_fixed_primary_cost_seconds=primary.lazy_newton.cost_seconds;
  }

  auto& a=final.lazy_newton;const auto& b=primary.lazy_newton;
  for(std::size_t k=0;k<a.cost_seconds.size();++k)a.cost_seconds[k]+=b.cost_seconds[k];
  for(std::size_t p=0;p<a.calls.size();++p)for(std::size_t f=0;f<a.calls[p].size();++f) {
    auto& x=a.calls[p][f];const auto& y=b.calls[p][f];
#define CALL_ADD(field) x.field+=y.field
    CALL_ADD(model_values);CALL_ADD(model_gradients);CALL_ADD(values);CALL_ADD(gradients);
    CALL_ADD(cached_gradient_accumulations);CALL_ADD(jacobian_products);
    CALL_ADD(hessian_assemblies);CALL_ADD(hessian_products);
#undef CALL_ADD
  }
#define NEWTON_ADD(field) a.field+=b.field
  NEWTON_ADD(majorant_geometry_parallel_batches);NEWTON_ADD(predictor_corrector_fallbacks);
  NEWTON_ADD(lagrangian_hessian_products);NEWTON_ADD(core_solve_seconds);NEWTON_ADD(solver_wall_seconds);
  NEWTON_ADD(atom_guided_batches);NEWTON_ADD(atom_plateau_events);NEWTON_ADD(atom_multiplier_transports);
  NEWTON_ADD(atom_projection_builds);NEWTON_ADD(atom_projection_updates);NEWTON_ADD(atom_projection_requests);
  NEWTON_ADD(atom_projection_seconds);NEWTON_ADD(atom_assembly_seconds);NEWTON_ADD(atom_product_seconds);
  NEWTON_ADD(direction_corrections);NEWTON_ADD(direction_residual_evaluations);
  NEWTON_ADD(precision_condensation_recoveries);NEWTON_ADD(precision_condensation_rows);
  NEWTON_ADD(atom_oracle_calls);NEWTON_ADD(atom_oracle_solves);NEWTON_ADD(atom_oracle_failures);
  NEWTON_ADD(atom_tangent_witnesses);NEWTON_ADD(atom_oracle_seconds);
  NEWTON_ADD(atom_sparsifications);NEWTON_ADD(atom_dropped_coordinates);NEWTON_ADD(atom_sparse_sign_rejections);
  NEWTON_ADD(atom_rayleigh_probes);NEWTON_ADD(atom_covered_rejections);NEWTON_ADD(atom_coverage_observations);
  NEWTON_ADD(atom_source_analyses);NEWTON_ADD(atom_spectral_blocks);NEWTON_ADD(atom_large_blocks);
  NEWTON_ADD(atom_cached_candidates);NEWTON_ADD(atom_creations);NEWTON_ADD(atom_reactivations);
  NEWTON_ADD(atom_activation_batches);NEWTON_ADD(atom_completion_batches);NEWTON_ADD(atom_layout_updates);
  NEWTON_ADD(atom_analysis_seconds);NEWTON_ADD(atom_update_seconds);
  NEWTON_ADD(objective_atoms);NEWTON_ADD(inequality_atoms);NEWTON_ADD(diagnostic_seconds);
  NEWTON_ADD(condensation_recoveries);NEWTON_ADD(promoted_inequalities);NEWTON_ADD(promoted_bounds);
  NEWTON_ADD(dual_only_acceptances);NEWTON_ADD(structure_builds);NEWTON_ADD(numeric_updates);
  NEWTON_ADD(factor_attempts);NEWTON_ADD(inertia_rejections);
  NEWTON_ADD(regularization_increases);NEWTON_ADD(regularization_decreases);
  NEWTON_ADD(backend_regularization_attempts);NEWTON_ADD(backend_factor_failures);
  NEWTON_ADD(backend_regularized_factor_failures);NEWTON_ADD(dual_regularized_factors);
  NEWTON_ADD(curvature_reuses);NEWTON_ADD(unshifted_factors);NEWTON_ADD(unshifted_accepted_steps);
  NEWTON_ADD(pivoted_fallbacks);NEWTON_ADD(native_factor_successes);NEWTON_ADD(native_linear_solves);
  NEWTON_ADD(sparse_numeric_failures);NEWTON_ADD(normal_recovery_attempts);
  NEWTON_ADD(normal_recovery_successes);NEWTON_ADD(normal_recovery_linear_solves);
  NEWTON_ADD(normal_recovery_symbolic_analyses);NEWTON_ADD(normal_recovery_symbolic_reuses);
  NEWTON_ADD(native_factor_seconds);NEWTON_ADD(native_solve_seconds);
  NEWTON_ADD(normal_recovery_factor_seconds);NEWTON_ADD(normal_recovery_solve_seconds);
  NEWTON_ADD(symbolic_analyses);NEWTON_ADD(symbolic_reuses);NEWTON_ADD(scatter_builds);NEWTON_ADD(scatter_reuses);
  NEWTON_ADD(constraint_activations);NEWTON_ADD(working_set_retries);NEWTON_ADD(line_search_backtracks);
  NEWTON_ADD(accepted_steps);NEWTON_ADD(full_steps);NEWTON_ADD(linear_solves);NEWTON_ADD(original_certificates);
  NEWTON_ADD(setup_seconds);NEWTON_ADD(evaluation_seconds);NEWTON_ADD(assembly_seconds);
  NEWTON_ADD(factor_seconds);NEWTON_ADD(solve_seconds);NEWTON_ADD(globalization_seconds);NEWTON_ADD(certificate_seconds);
#undef NEWTON_ADD
#define NEWTON_MAX(field) a.field=std::max(a.field,b.field)
  NEWTON_MAX(atom_peak_active);NEWTON_MAX(atom_maximum_trace);NEWTON_MAX(atom_maximum_tangent_error);
  NEWTON_MAX(maximum_condensation_ratio);NEWTON_MAX(normal_recovery_maximum_shift);
  NEWTON_MAX(maximum_regularization);NEWTON_MAX(maximum_backend_primal_regularization);
  NEWTON_MAX(maximum_backend_dual_regularization);NEWTON_MAX(maximum_failed_backend_primal_regularization);
  NEWTON_MAX(maximum_failed_backend_dual_regularization);NEWTON_MAX(maximum_direction_residual);
#undef NEWTON_MAX
  if(b.minimum_step>0&&(a.minimum_step<=0||b.minimum_step<a.minimum_step))a.minimum_step=b.minimum_step;
  const int preceding_outer=other_precedes_final?primary.outer_iterations:final.outer_iterations;
  const double preceding_seconds=other_precedes_final?primary.elapsed_seconds:final.elapsed_seconds;
  const int earlier_failure=other_precedes_final?b.first_sparse_failure_iteration:a.first_sparse_failure_iteration;
  const int later_failure=other_precedes_final?a.first_sparse_failure_iteration:b.first_sparse_failure_iteration;
  a.first_sparse_failure_iteration=earlier_failure>=0?earlier_failure:
      later_failure>=0?later_failure+preceding_outer:-1;
  auto history=other_precedes_final?b.iteration_history:a.iteration_history;
  const auto& later_history=other_precedes_final?a.iteration_history:b.iteration_history;
  history.reserve(history.size()+later_history.size());
  for(auto item:later_history) {
    item.iteration+=preceding_outer;item.elapsed_seconds+=preceding_seconds;
    history.push_back(std::move(item));
  }
  a.iteration_history=std::move(history);

#define RESULT_ADD(field) final.field+=primary.field
  RESULT_ADD(outer_iterations);RESULT_ADD(inner_iterations);RESULT_ADD(elapsed_seconds);
  RESULT_ADD(kkt_factorizations);RESULT_ADD(kkt_symbolic_analyses);RESULT_ADD(kkt_symbolic_reuses);
  RESULT_ADD(kkt_symbolic_invalidations);RESULT_ADD(kkt_numeric_scatter_builds);
  RESULT_ADD(kkt_numeric_scatter_reuses);RESULT_ADD(kkt_numeric_scatter_invalidations);
  RESULT_ADD(restoration_iterations);RESULT_ADD(restoration_restarts);RESULT_ADD(restoration_linear_solves);
  RESULT_ADD(restoration_exact_hessian_solves);RESULT_ADD(restoration_matrix_free_solves);RESULT_ADD(restoration_row_space_solves);
  RESULT_ADD(restoration_harmonic_attempts);RESULT_ADD(restoration_harmonic_acceptances);RESULT_ADD(restoration_harmonic_factorizations);
  RESULT_ADD(restoration_tangent_activation_attempts);RESULT_ADD(restoration_tangent_activation_acceptances);
  RESULT_ADD(restoration_tangent_activation_components);RESULT_ADD(restoration_cg_iterations);
  RESULT_ADD(restoration_exact_workspace_growths);RESULT_ADD(restoration_exact_matvecs);
  RESULT_ADD(restoration_exact_parallel_row_dot_batches);RESULT_ADD(restoration_exact_parallel_teams);
  RESULT_ADD(restoration_exact_sparse_row_dot_batches);RESULT_ADD(restoration_exact_sparse_row_dot_verifications);
  RESULT_ADD(restoration_exact_sparse_row_dot_fallbacks);RESULT_ADD(restoration_line_search_backtracks);
  RESULT_ADD(restoration_stagnation_events);RESULT_ADD(restoration_minimax_qp_solves);
  RESULT_ADD(restoration_minimax_inner_iterations);RESULT_ADD(restoration_minimax_factorizations);
  RESULT_ADD(restoration_minimax_accepted_steps);RESULT_ADD(restoration_minimax_line_search_backtracks);
  RESULT_ADD(restoration_harmonic_seconds);RESULT_ADD(restoration_tangent_activation_seconds);
  RESULT_ADD(restoration_legacy_gate_seconds);RESULT_ADD(restoration_exact_seconds);
  RESULT_ADD(restoration_minimax_seconds);RESULT_ADD(restoration_restart_seconds);RESULT_ADD(restoration_total_seconds);
  RESULT_ADD(restoration_tangent_activation_gradient_rows);RESULT_ADD(restoration_tangent_activation_line_quadratics);
  RESULT_ADD(restoration_tangent_activation_candidates);
  RESULT_ADD(lazy_newton_iterations);RESULT_ADD(lazy_newton_factorizations);RESULT_ADD(lazy_newton_symbolic_analyses);
  RESULT_ADD(lazy_newton_inertia_corrections);RESULT_ADD(lazy_newton_refinement_sweeps);
  RESULT_ADD(lazy_newton_quadratic_corrections);RESULT_ADD(lazy_newton_quadratic_correction_acceptances);
  RESULT_ADD(lazy_newton_lazy_cuts);RESULT_ADD(lazy_newton_proximal_shifts);RESULT_ADD(lazy_newton_exact_curvature_steps);
  RESULT_ADD(lazy_newton_majorant_curvature_steps);RESULT_ADD(lazy_newton_curvature_fallbacks);
  RESULT_ADD(lazy_newton_dual_regularizations);RESULT_ADD(lazy_newton_barrier_updates);
  RESULT_ADD(lazy_newton_line_search_backtracks);RESULT_ADD(lazy_newton_certificate_evaluations);
  RESULT_ADD(lazy_newton_setup_seconds);RESULT_ADD(lazy_newton_factorization_seconds);
  RESULT_ADD(lazy_newton_assembly_seconds);RESULT_ADD(lazy_newton_structure_seconds);
  RESULT_ADD(lazy_newton_evaluation_seconds);RESULT_ADD(lazy_newton_line_search_seconds);RESULT_ADD(lazy_newton_certificate_seconds);
#undef RESULT_ADD
  auto& rank=final.rank_presolve;const auto& pr=primary.rank_presolve;
#define RANK_ADD(field) rank.field+=pr.field
  RANK_ADD(preparation_seconds);RANK_ADD(postsolve_seconds);RANK_ADD(audit_seconds);
  RANK_ADD(audit_ordered_sums);RANK_ADD(audit_term_evaluations);RANK_ADD(audit_term_reuses);
  RANK_ADD(audit_calls);RANK_ADD(audit_mapped_calls);RANK_ADD(audit_full_calls);RANK_ADD(audit_check_calls);
  RANK_ADD(audit_value_evaluations);RANK_ADD(audit_value_reuses);RANK_ADD(audit_gradient_evaluations);
  RANK_ADD(audit_result_copies);RANK_ADD(audit_lift_calls);RANK_ADD(audit_final_full_calls);RANK_ADD(audit_coordinate_expansions);
#undef RANK_ADD

  if(!annotate_rank_retry) {
    // The historical exact-fixed retry retains its established serialization.
    // A native barrier retry must also expose all work of a rejected second
    // phase; endpoint residuals, current ranks, parameters and flags stay those
    // of the retained phase. Only count/time fields are additive here.
    auto& d=final.dca_filter;const auto& e=primary.dca_filter;
#define DCA_ADD(field) d.field+=e.field
    DCA_ADD(majorant_original_cache_evaluations);DCA_ADD(majorant_mu_updates);
    DCA_ADD(majorant_slack_transports);DCA_ADD(majorant_predictor_directions);DCA_ADD(majorant_corrector_directions);
    DCA_ADD(majorant_startup_predictor_corrector_steps);DCA_ADD(majorant_terminal_single_newton_steps);
    DCA_ADD(majorant_direction_precision_retries);DCA_ADD(majorant_startup_predictor_corrector_seconds);
    DCA_ADD(majorant_barrier_phases);DCA_ADD(majorant_barrier_filter_resets);
    DCA_ADD(majorant_barrier_line_trials);DCA_ADD(majorant_barrier_line_acceptances);DCA_ADD(majorant_barrier_line_rejections);
    DCA_ADD(majorant_filter_path_builds);DCA_ADD(majorant_filter_path_evaluations);DCA_ADD(majorant_early_reanchors);
    DCA_ADD(majorant_curvature_certificates);DCA_ADD(majorant_curvature_reuses);DCA_ADD(majorant_curvature_small_spectra);
    DCA_ADD(majorant_curvature_gershgorin);DCA_ADD(majorant_curvature_analytic_spectra);DCA_ADD(majorant_curvature_seconds);
    DCA_ADD(majorant_outer_iterations);DCA_ADD(majorant_inner_convergences);DCA_ADD(majorant_reanchors);
    DCA_ADD(majorant_reject_factorization);DCA_ADD(majorant_reject_precision);DCA_ADD(majorant_reject_linear_backend);
    DCA_ADD(majorant_reject_globalization);DCA_ADD(majorant_reject_domain);
    DCA_ADD(majorant_lambda_attempts);DCA_ADD(majorant_lambda_acceptances);DCA_ADD(majorant_lambda_iterations);
    DCA_ADD(majorant_lambda_seconds);DCA_ADD(majorant_lambda_interpolations);DCA_ADD(majorant_lambda_rho_probes);
    DCA_ADD(majorant_curvature_attempts);DCA_ADD(majorant_curvature_solves);DCA_ADD(majorant_curvature_acceptances);
    DCA_ADD(majorant_curvature_correction_seconds);DCA_ADD(progress_events);DCA_ADD(aggregate_progress_stalls);DCA_ADD(dual_progress_stalls);
    DCA_ADD(progress_rescue_trials);DCA_ADD(progress_rescue_acceptances);DCA_ADD(terminal_rescue_triggers);
    DCA_ADD(terminal_rescue_trials);DCA_ADD(terminal_rescue_acceptances);DCA_ADD(aggregate_root_trials);DCA_ADD(aggregate_root_acceptances);
    DCA_ADD(aggregate_curvature_descents);DCA_ADD(aggregate_curvature_shortcuts);
    DCA_ADD(aggregate_kkt_layout_rebuilds);DCA_ADD(aggregate_reject_inertia);DCA_ADD(aggregate_reject_backend);
    DCA_ADD(aggregate_reject_numerical_direction);DCA_ADD(aggregate_reject_domain);DCA_ADD(aggregate_reject_filter);
    DCA_ADD(aggregate_reject_kkt);DCA_ADD(aggregate_reject_globalization);DCA_ADD(aggregate_rho_memory_uses);
    DCA_ADD(aggregate_step_regularizations);DCA_ADD(aggregate_backend_assembly_seconds);DCA_ADD(aggregate_backend_symbolic_seconds);
    DCA_ADD(aggregate_backend_numeric_seconds);DCA_ADD(aggregate_backend_solve_seconds);DCA_ADD(aggregate_scatter_builds);
    DCA_ADD(aggregate_scatter_reuses);DCA_ADD(aggregate_scatter_invalidations);DCA_ADD(model_layout_builds);
    DCA_ADD(original_point_calls);DCA_ADD(original_point_value_reuses);DCA_ADD(model_layout_seconds);DCA_ADD(original_point_seconds);
    DCA_ADD(aggregate_only_attempts);DCA_ADD(aggregate_only_exhaustions);DCA_ADD(aggregate_only_inertia_rejections);
    DCA_ADD(aggregate_only_step_rejections);DCA_ADD(rho_grid_completions);DCA_ADD(strict_finish_attempts);
    DCA_ADD(strict_finish_acceptances);DCA_ADD(strict_finish_rejections);DCA_ADD(strict_finish_refinements);
    DCA_ADD(strict_finish_certificates);DCA_ADD(strict_finish_solved);DCA_ADD(strict_finish_seconds);
    DCA_ADD(aggregate_preparation_requests);DCA_ADD(aggregate_preparation_builds);DCA_ADD(aggregate_preparation_hits);
    DCA_ADD(aggregate_preparation_invalidations);DCA_ADD(aggregate_preparation_seconds);DCA_ADD(aggregate_preparation_key_seconds);
    DCA_ADD(aggregate_preparation_geometry_seconds);DCA_ADD(aggregate_preparation_operator_seconds);
    DCA_ADD(aggregate_strict_probe_checks);DCA_ADD(aggregate_strict_probe_successes);DCA_ADD(aggregate_strict_probe_failures);
    DCA_ADD(aggregate_strict_probe_skipped);DCA_ADD(aggregate_strict_probe_seconds);
    DCA_ADD(aggregate_trials);DCA_ADD(aggregate_acceptances);DCA_ADD(aggregate_certificates);DCA_ADD(aggregate_seconds);
    DCA_ADD(aggregate_factor_seconds);DCA_ADD(aggregate_linear_solves);DCA_ADD(aggregate_symbolic_analyses);
    DCA_ADD(aggregate_symbolic_reuses);DCA_ADD(aggregate_rejections);DCA_ADD(krylov_trials);DCA_ADD(krylov_acceptances);
    DCA_ADD(krylov_iterations);DCA_ADD(krylov_solves);DCA_ADD(krylov_linear_rejections);DCA_ADD(krylov_filter_rejections);
    DCA_ADD(krylov_skips);DCA_ADD(krylov_operator_products);DCA_ADD(krylov_preconditioner_solves);DCA_ADD(krylov_seconds);
    DCA_ADD(local_trials);DCA_ADD(local_certificates);DCA_ADD(local_acceptances);DCA_ADD(local_seconds);
    DCA_ADD(anderson_trials);DCA_ADD(anderson_acceptances);DCA_ADD(certificate_transport_trials);DCA_ADD(certificate_transports);
    DCA_ADD(rho_refinement_trials);DCA_ADD(rho_refinement_acceptances);DCA_ADD(model_reuses);DCA_ADD(basis_compressions);
    DCA_ADD(retired_directions);DCA_ADD(cumulative_atoms_lagrangian);DCA_ADD(extrapolation_trials);DCA_ADD(extrapolation_acceptances);
    DCA_ADD(extrapolation_seconds);DCA_ADD(curvature_factor_reuses);DCA_ADD(subspace_spectra);DCA_ADD(dependent_witnesses);
    DCA_ADD(curvature_ritz_shift_proposals);DCA_ADD(curvature_gershgorin_fallbacks);DCA_ADD(corrector_steps);
    DCA_ADD(centering_trials);DCA_ADD(original_line_trials);DCA_ADD(original_line_acceptances);DCA_ADD(original_line_rejections);
    DCA_ADD(private_line_trials);DCA_ADD(private_line_rejections);DCA_ADD(curvature_ensure_seconds);DCA_ADD(curvature_factor_seconds);
    DCA_ADD(curvature_inverse_seconds);DCA_ADD(subspace_seconds);DCA_ADD(models);DCA_ADD(hessian_updates);DCA_ADD(inner_steps);
    DCA_ADD(filter_trials);DCA_ADD(filter_rejections);DCA_ADD(private_continuations);DCA_ADD(model_retries);
    DCA_ADD(soc_attempts);DCA_ADD(soc_acceptances);DCA_ADD(restorations);DCA_ADD(atom_events);
    DCA_ADD(curvature_factors);DCA_ADD(curvature_reuses);DCA_ADD(first_step_acceptances);DCA_ADD(spectral_inverse_solves);
    DCA_ADD(small_block_spectra);DCA_ADD(model_build_seconds);DCA_ADD(original_filter_seconds);DCA_ADD(curvature_seconds);DCA_ADD(soc_seconds);
#undef DCA_ADD
#define FIT_ADD(field) final.field+=primary.field
    FIT_ADD(multiplier_refit_attempts);FIT_ADD(multiplier_refit_successes);FIT_ADD(multiplier_refit_fallback_reuses);
    FIT_ADD(multiplier_row_space_factorizations);FIT_ADD(multiplier_row_space_solves);FIT_ADD(multiplier_row_space_symbolic_analyses);
    FIT_ADD(multiplier_row_space_symbolic_reuses);FIT_ADD(multiplier_row_space_failures);
    FIT_ADD(termination_multiplier_refinement_attempts);FIT_ADD(termination_multiplier_fit_sweeps);
#undef FIT_ADD
  }
}
} // namespace lazyipdca::qcqp::detail
