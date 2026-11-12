#pragma once
#include <cmath>
#include <limits>
#include <string>
#include <functional>
#include <memory>
#include <cstdint>

namespace lazyipdca::qcqp {

class ConstraintPresolve;
struct QCQPProblem;
struct Result;
struct FactorialTraceState {
  std::uint64_t accepted_events=0;
  bool success=true;
  std::string error;
};
struct Settings {
  // Independent research factors. Empty strings preserve every historical path.
  std::string factorial_inequalities, factorial_globalization, factorial_trace;
  bool factorial_validation_legacy_objective=false;
  std::shared_ptr<FactorialTraceState> factorial_trace_state;
  bool factorial_active() const {return !factorial_inequalities.empty();}
  bool factorial_exact_objective() const {return factorial_active()&&!factorial_validation_legacy_objective;}
  bool factorial_r3_controls() const {return factorial_active()?factorial_globalization=="r3":convexification_study=="off";}

  // CLI-only optional diagnostic observer; absent by default, not a restart.
  std::function<void(const QCQPProblem&,Result&,const ConstraintPresolve*,int,double)> accepted_checkpoint_callback;
  bool accepted_checkpoint_builtin = false; // CLI diagnostic observer; custom API observers remain authoritative
  std::string dca_rank_presolve_report_file;
  bool dca_rank_presolve=false;
  bool free_epigraph_presolve=true; // coefficient-certified; adaptive study only
  std::string dca_rank_audit_mode="mapped"; // full, mapped, or check (dual audit; diagnostics only)
  bool dca_rank_implied_bounds=true; // explicit opt-in; the rank-start runner enables it
  const ConstraintPresolve* rank_presolve_context=nullptr; // borrowed during solve

  // Select the termination semantics, independently of the direction backend.
  // project preserves the legacy original-units certificate. ipopt reproduces
  // the bundled Ipopt 3.14.11 test (mu_target=0, no variable scaling/relaxation).
  std::string termination_test = "project";
  // Native stopping tolerances MUST NOT retune the existing DCA controller.
  // The R7D10 project runner used 1e-6 for all four numerical control tolerances.
  // This explicit value controls the filter/model/barrier, not certification.
  double dca_control_tolerance = 1e-6;
  std::string dca_endpoint_start_file; // explicit warm start, not a trajectory checkpoint
  double acceptable_objective_change_tolerance = 1e20;

  int acceptable_iterations = 15; // zero disables the acceptable stopping rule
  // Ipopt-style recovery of one individually acceptable point after a
  // terminal globalization/restoration/numerical failure. It is deliberately
  // distinct from the consecutive acceptable-iterate stopping rule.
  bool acceptable_on_failure = false;
  // Validated Lazy-DCA acceptable profile for the frozen case9241 instance.
  // Strict convergence remains governed independently by the 1e-6 KKT limits.
  double acceptable_tolerance = 1e-5;
  double acceptable_constraint_tolerance = 1e-6;
  double acceptable_dual_tolerance = 1e-5;
  double acceptable_complementarity_tolerance = 1e-6;
  // Runtime MA27 adapter; the HSL solver library is supplied separately.
  std::string hsl_library;
  double hsl_pivot_tolerance = 1e-8;
  double hsl_memory_factor = 1.5;
  // Explicit comparative research route; off preserves historical R3.
  std::string convexification_study = "off";
  std::string convexification_schedule = "lazy";
  // H3: function-model metrics; independent of numerical KKT regularization.
  int study_hybrid_max_atoms = 8;
  int study_hybrid_total_atoms = 32;
  int study_hybrid_event_interval = 5;
  int study_hybrid_downweight_patience = 3;
  int study_hybrid_atom_support = 32;
  double study_hybrid_downweight_factor = 0.5;
  double study_hybrid_dominance = 2.0;
  double study_hybrid_scalar_decay = 0.5;
  double study_hybrid_margin = 0.0;
  double study_inner_relative = 0.2;
  double study_inner_absolute = 1e-10;
  int study_inner_max = 80;
  int study_inner_min = 1;
  int study_hybrid_candidates = 8;
  int study_hybrid_lanczos_steps = 24;
  int study_hybrid_lanczos_starts = 2;
  int study_hybrid_additions_per_event = 2;
  double study_hybrid_support_energy = 0.999999;
  double study_hybrid_strength = 1.0;
  double study_hybrid_beta_offset = 1e-6;
  double study_hybrid_negative_tolerance = 1e-8;
  double study_hybrid_scalar_safety = 1.10;
  double study_hybrid_scalar_floor = 0.10;
  double study_hybrid_score_activity = 0.50;
  double study_hybrid_score_scalar = 0.50;
  double study_hybrid_score_support = 0.10;
  double study_hybrid_score_density = 0.10;
  double study_hybrid_score_weight = 0.05;
  double study_hybrid_min_lift = 1e-4;
  double study_hybrid_min_activity = 1e-2;
  double study_hybrid_min_score = 1e-4;
  double study_hybrid_objective_floor = 1e-12;
  // R3 controller changes only f/g curvature; certified retains the earlier experimental inner solve.
  std::string study_controller = "r3";
  std::string study_inner_rule = "model-relative";
  double study_inner_kappa_fraction = 0.25;
  double study_inner_margin = 1e-3;
  double study_inner_activity = 1e-8;
  double study_inner_feasibility = 1e-10;
  double study_inner_complementarity = 1e-10;
  bool study_inner_separate_steps = true; // recovery after common PC and centering fail
  bool study_inner_merit_recovery = false; // experimental guarded Armijo fallback
  double study_inner_barrier_decay = 0.2;
  bool study_inner_predictor_corrector = true;

  bool objective_lazy_convexification = false;
  int objective_max_atoms = 32, objective_initial_atoms = 2, objective_atom_batch = 1;
  double objective_rho_decay = 0.5;
  double objective_rho_excess = 0.1; // initial scalar fraction of the uncorrected deficit
  std::string dca_engine = "auto"; // standard structural selection, or explicit legacy/filter/majorants
  // Standard uses only the frozen native R10 mechanisms on their verified
  // structural domain. "off" preserves the requested historical policy.
  std::string standard_r10_policy = "auto";
  // Native AL is a guarded second stage after an unchanged unsuccessful Lazy run.
  std::string native_al_fallback = "auto";
  bool native_al_fallback_resolved = false; // internal recursion guard
  bool standard_auto_resolved = false; // internal: propagated through native nested solves
  bool standard_equality_watch = false; // internal: return unused work on measured historical stagnation
  bool majorant_equality_stabilization_explicit = false; // preserves explicit CLI 0
  bool original_stabilized_merit = false;
  bool original_source_primal_barrier = false;
  bool original_bound_product_path = false;
  bool original_gr_short_step_prox = false;
  bool original_gr_kkt_soc = false; // explicit frozen-factor SOC, original GR acceptance
  bool original_gr_consistent_bound_slacks = false;
  std::string original_gr_consistent_bound_trigger = "always";
  bool original_exact_affine_faces = false;
  bool original_exact_affine_faces_consumed = false;
  bool original_private_phase = false;
  // Independent R12 controlled adjustments; no change to original KKT or Armijo.
  bool gr_coefficient_constraint_increments = true;
  std::string gr_proximal_policy = "historical";
  int gr_fixed_centering_steps = 0;
  bool gr_residual_components = true;
  int majorant_call_depth = 0;
  bool lazy_newton_initial_mu_explicit = false;
  int dca_filter_inner_max = 8;
  bool dca_filter_diagnostics = false;
  // Read-only decomposition of original dual-residual growth.  This is kept
  // separate from the verbose filter trace because it evaluates several
  // counterfactual original-QCQP states and is intended for targeted runs.
  bool dca_dual_decomposition = false;
  double dca_dual_decomposition_growth = 1.25;
  // Numerical recovery does not grant acceptance of a DCA direction.
  bool mumps_memory_recovery = true;
  // Internal numerical recovery policy; the requested METIS ordering remains
  // in force on MUMPS's weighted-matching compressed symmetric graph.
  bool mumps_symmetric_matching = false;
  bool dca_restoration_compare = true;
  bool dca_aggregate_adaptive_pivot = false;
  bool dca_aggregate_full_refinement = false;
  bool dca_aggregate_adaptive_dual = false;
  // Ipopt-compatible globalization keeps positive inequality/bound
  // multipliers at their independent fraction-to-boundary step while only
  // the primal step is backtracked. Equality multipliers follow the primal
  // step. It remains experimental until the full six-case campaign validates
  // its centrality safeguard; "coupled" is the reproducible R5 default.
  std::string dca_dual_step = "coupled";
  // Near feasibility, compare an already filter-acceptable split step with a
  // common feasible primal/dual fraction only when the split step grows the
  // original dual residual.  This preserves the established far-from-feasible
  // trajectory and repairs the measured loss of Newton cancellation.
  double dca_common_step_primal_trigger = 1e-1;
  double dca_common_step_dual_growth = 1.25;
  int dca_filter_rho_refinements = 3;
  bool dca_filter_initial_restoration = false;
  bool dca_filter_normal_atom_weighting = false;
  // G6 research controls are opt-in: the reference DCA behavior remains the
  // default. These select independently measurable paths, not looser audits.
  bool dca_filter_active_evaluation = false;
  bool dca_filter_packed_atoms = false;
  bool dca_filter_lazy_merit = false;
  bool dca_filter_block_ritz = false;
  bool dca_filter_anderson = false;
  bool dca_filter_tangent_curvature = false;
  // G9: optional original primal-dual Newton-Krylov acceleration. A failed
  // transaction leaves the general DCA state and factorization intact.
  bool dca_aggregate = false;
  // Research route: one ORIGINAL aggregate transaction per accepted outer.
  // No DcaModels::build, private inner loop, or majorant/restoration fallback.
  // A failed bounded search returns an explicit non-success, never a false certificate.
  bool dca_aggregate_only = false;
  int dca_aggregate_retry_max = 24;
  double dca_aggregate_rho_initial = 1e-6;
  double dca_aggregate_rho_growth = 10.0;
  double dca_aggregate_rho_decay = 0.2;
  bool dca_aggregate_affine_retry = true;
  // R3 controls: independently reversible without weakening certification.
  std::string dca_aggregate_kkt_layout = "persistent";
  double dca_aggregate_border_reserve = 1.25;
  bool dca_aggregate_fast_assembly = true;
  bool dca_aggregate_rho_memory = true;
  std::string dca_aggregate_rejection_policy = "separated";
  bool dca_aggregate_lean = true;
  // Exact, transaction-local reuse between rho probes; neither inertia nor
  // candidate audits are cached. Independent switches enable A/B validation.
  bool dca_aggregate_preparation_cache = true;
  bool dca_aggregate_strict_probe_stop = true;
  // Complete the bounded inertia grid, without increasing the rho cap.
  bool dca_rho_complete_search = true;
  // Bounded, exact-original residual finish of an acceptable non-strict state.
  bool dca_strict_finish = true;
  int dca_strict_finish_max = 6;
  int dca_strict_finish_refinements = 8;
  double dca_strict_finish_eta = 0.1;
  double dca_strict_finish_mu_fraction = 0.01;
  double dca_aggregate_rho_max = 100;
  std::string dca_aggregate_backend = "auto";
  double dca_aggregate_primal_trigger = std::numeric_limits<double>::infinity();
  // Preserve the historical aggregate trajectory: the former filter admitted
  // candidates up to 10x the current original KKT. The experimental two-point
  // stall response is opt-in because it interrupts that trajectory on case9241.
  double dca_kkt_growth_max = 10.0;
  int dca_aggregate_progress_window = 0;
  bool dca_krylov = false;
  int dca_krylov_max_iterations = 24;
  int dca_krylov_failure_cooldown = 2;
  bool dca_krylov_predictor_corrector = true;
  bool dca_local_active = false;
  bool dca_pair_elimination = false;
  double dca_local_trigger = 1.0;
  double dca_local_primal_trigger = 1e-3;
  std::string dca_local_linear_backend = "same";
  bool dca_local_weak_affine = false;
  bool dca_filter_certificate_transport = false;
  bool dca_filter_sparse_fallback = false;
  std::string dca_filter_curvature_backend = "same";
  bool dca_filter_model_reuse = true;
  bool dca_filter_extrapolation = true;
  int dca_filter_model_retries = 3;
  int dca_filter_max_atoms = 500;
  int dca_filter_atom_batch = 2;
  // R7D13: atomic search-space controls, previously hard-coded in
  // dca_curvature.hpp. Defaults reproduce the historical behavior exactly.
  int dca_atom_basis_keep = 32;          // compaction threshold of the stored basis
  int dca_atom_basis_tail = 8;           // most recent basis vectors retained by a compaction
  int dca_atom_probe_depth = 18;         // inverse Krylov/Ritz probe depth
  bool dca_atom_probe_count_projection = true; // the ker(N) projection solve consumes one unit of depth
  int dca_filter_atom_every = 1;
  bool dca_filter_atom_events = true;
  double dca_filter_atom_gain = 1e-6;

  double native_inertia_relative_zero_tolerance = 0.0;
  std::string ln_engine = "legacy"; // legacy filter/DC graft | atomic reduced Newton
  bool lazy_newton_reduction = true;
  bool lazy_newton_lazy_curvature = true;
  bool lazy_newton_lazy_constraints = true;
  bool lazy_newton_predictor_corrector = false;
  bool lazy_newton_packed_gradients = true; // persistent contiguous Jacobian storage
  bool lazy_newton_adaptive_barrier = true;
  bool lazy_newton_interleaved_ordering = true;
  bool lazy_newton_sparse_recovery = true;
  double lazy_newton_initial_mu = 1e-4;
  double lazy_newton_activation_margin = 1.0;
  double lazy_newton_fraction_to_boundary = 0.995;
  double lazy_newton_objective_gradient_target = 100.0;
  double lazy_newton_constraint_gradient_target = 100.0;
  double lazy_newton_regularization_initial = 1e-6;
  double lazy_newton_regularization_decay = 0.2;
  int lazy_newton_dense_threshold = 32;
  bool kkt_primal_first_ordering = false;
  bool kkt_interleaved_equality_ordering = false;
  bool kkt_preserve_structural_zeros = false;
  int live_display_level = -1; // -1 resolves legacy flags; 0..4 explicit progressive display
  bool live_curvature = false;
  bool lazy_newton_sparse_certificate = true; // original coefficients, indexed fused audit
  bool lazy_newton_atom_guided = true; // tangent source ranking after an inertia-repair plateau
  int lazy_newton_atom_guided_after = 2;
  bool lazy_newton_atom_step_decay = true; // decay per accepted step length
  bool lazy_newton_atom_transport = true; // retire with decreasing source multipliers
  bool lazy_newton_atom_energy_sparsify = false;
  bool lazy_newton_atom_rayleigh_gate = false; // QP-inspired post-projection admission
  double lazy_newton_atom_decay = 0.04; // faster retirement of localized corrections
  bool lazy_newton_atomic_regularization = true; // r3p29r4: lazy sparse rank-one corrections
  bool lazy_newton_condensation_guard = true;
  double lazy_newton_condensation_ratio = 1.0/std::sqrt(std::numeric_limits<double>::epsilon());

  // Convex-subproblem backend. "lazyqp" is the coupled nonlinear-residual
  // extension of the LazyIPDCA-QP primal-dual kernel to already-convex QCQPs;
  // "socp" builds the exact rotated-cone epigraph and analytically eliminates
  // its cone auxiliaries back to the sparse QCQP primal-dual equations;
  // "legacy-ipm" preserves the older experimental QCQP IPM and "ipopt" is an
  // independent reference backend.
  std::string inner_backend = "lazyqp";
  // Linear KKT factorization used by the native convex-subproblem solver.
  // native = dense LAPACK symmetric-indefinite, ldl = historical SuiteSparse LDL + METIS,
  // qdldl = raw QDLDL + METIS, cholesk = the sign-aware quasi-definite
  // QDLDL adaptation with separate primal/dual regularization, and mumps =
  // symmetric-indefinite MUMPS + METIS.
  std::string linear_backend = "native";
  bool linear_metis = true;
  // Keep the sparse factor object alive across successive Lazy-DCA convex
  // majorants.  Its METIS permutation, CSC structure, elimination tree and
  // workspaces are reused while only the numerical values are refreshed.
  // Pattern growth invalidates the cache safely and triggers a new analysis.
  bool kkt_persistent_symbolic_cache = true;
  // Numerical KKT regularization.  These are distinct from the SQP model
  // proximal term below: rho_x damps the Newton/KKT solve only.
  double kkt_primal_regularization = 1e-12; // rho_x initial/floor for compatibility
  double kkt_dual_regularization = 1e-12;   // fixed delta_d when adaptive mode is off
  // Internal boundary: negative auxiliary atom rows after this coordinate
  // retain their exact mathematical pivots; -1 preserves historical behavior.
  int kkt_dual_regularization_end = -1;
  bool kkt_adaptive_regularization = false;
  bool kkt_equilibration = false; // symmetric Ruiz scaling in raw LDL/QDLDL kernels
  double kkt_rho_x_floor = 1e-12;
  double kkt_rho_x_growth = 10.0;
  double kkt_rho_x_decay = 0.5;
  double kkt_rho_x_max = 1e-2;
  int kkt_rho_x_clean_outer = 3;
  int kkt_regularization_retry_max = 8;
  double kkt_delta_dual_base = 1e-8;
  double kkt_delta_dual_floor = 1e-12;
  double kkt_delta_dual_growth = 10.0;
  double kkt_delta_dual_decay = 0.5;
  double kkt_delta_dual_max = 1e-4;
  bool kkt_delta_dual_inverse = true;
  double kkt_backward_error_tolerance = 1e-9;
  double kkt_backward_error_good = 1e-12;
  double kkt_backward_error_reject = 1e-7;
  int kkt_iterative_refinement_steps = 3;
  // Adaptation of the supplied legacy `cholesk` LDL^T solver.  When the
  // caller already assembles adaptive KKT shifts (`kkt_adaptive_regularization`),
  // cholesk factorizes that exact matrix and never adds a hidden shift.  In
  // fixed mode it first attempts the exact caller matrix, then repairs a
  // zero/wrong-sign/tiny pivot by growing the primal and dual KKT
  // regularizations independently, using the common floors, caps, growth and
  // retry budget above. Thus the floors are safeguards, not unconditional
  // perturbations of every Newton system.
  bool cholesk_adaptive_regularization = true;
  bool cholesk_regularization_warm_start = true;
  bool cholesk_equilibration = true;
  double cholesk_relative_pivot_tolerance = 1e-14;
  // Equality-only convex QPs retain the historical original-KKT Richardson
  // refinement first.  If it still misses the original unregularized QP
  // contract, use the regularized factor only as a right preconditioner for
  // flexible GMRES on the exact bordered KKT operator.
  bool equality_qp_original_kkt_gmres = true;
  int equality_qp_original_kkt_gmres_max_iterations = 128;
  // Outer algorithm. LazyIPDCA-QCQP has one scientific route: Lazy-DCA.
  // Every quadratic equality is represented in each convex subproblem by the
  // classical SQP linearization h(x_k)+grad h(x_k)^T d=0.  "auto" is retained
  // only as a backward-compatible alias for "lazy-dca"; it never selects SQP.
  // "legacy-sqp" remains an explicit comparison method and is never entered
  // implicitly. auto_sqp_max_dimension is retained only for source/CLI
  // compatibility and is ignored by the solver.
  std::string outer_algorithm = "lazy-dca";
  int auto_sqp_max_dimension = 0;
  int max_outer_iterations = 300;
  int max_inner_iterations = 500;
  double time_limit_seconds = 3600.0;

  // Final status certificate only.  These mirror Ipopt desired-convergence
  // guards and do not alter SQP/Phase-I/line-search steps.
  double feasibility_tolerance = 1e-6;       // constr_viol_tol analogue
  double stationarity_tolerance = 1e-5;      // tol analogue
  double dual_infeasibility_tolerance = 1e-5; // dual_inf_tol default
  double complementarity_tolerance = 1e-5;   // compl_inf_tol default
  double nlp_error_scaling_max = 100.0;       // s_max default
  double step_tolerance = 1e-9;
  double inner_tolerance = 1e-7;
  double inner_iteration_limit_mu_fallback = 1e-6;

  // Direct LazyIPDCA-QP kernel for already-convex QCQP subproblems.
  // There is no DC/curvature logic at this level: only barrier continuation
  // and Newton solves of the exact convex QCQP KKT equations.
  double lazyqp_barrier_mu0 = 1e-4;
  double lazyqp_barrier_r = 0.1;
  double lazyqp_barrier_theta = 0.1;
  int lazyqp_mu_inner_max = 0; // 0 = relative residual gate only
  double lazyqp_armijo = 1e-4;
  double lazyqp_line_search_beta = 0.5;
  int lazyqp_line_search_max = 40;
  // Reuse the most recent numerical KKT factorization only as a right
  // preconditioner for the *current* KKT system.  Flexible GMRES must satisfy
  // the current-system residual tolerance; otherwise LazyQP refactorizes the
  // current matrix before taking a step.  Thus this option skips numerical
  // factorizations without freezing the Newton equations or weakening any
  // inner/outer acceptance test.
  bool lazyqp_numeric_kkt_reuse = false;
  int lazyqp_numeric_kkt_reuse_max_age = 3;
  int lazyqp_numeric_kkt_reuse_max_iterations = 4;

  // Optional inexact-majorant forcing sequence.  The outer Lazy-DCA loop
  // publishes the absolute target
  //
  //   tau_k = clamp(eta * R_k, inner_tolerance, max_tolerance),
  //
  // where R_k is the current ORIGINAL-QCQP feasibility/stationarity residual.
  // LazyQP may return during the first fast_iterations only after satisfying
  // both this absolute target and an eta reduction from its own initial KKT
  // residual, with a strictly safeguarded model decrease.  If that contract
  // is missed, the ordinary strict inner solve continues unchanged.  The
  // transient target fields are populated only by Solver and are zero for
  // standalone LazyQP calls unless a targeted test sets them explicitly.
  bool lazyqp_inexact_forcing = false;
  double lazyqp_forcing_eta = 0.1;
  double lazyqp_forcing_max_tolerance = 1e-2;
  int lazyqp_forcing_fast_iterations = 4;
  int lazyqp_forcing_min_iterations = 1;
  double lazyqp_forcing_target_tolerance = 0.0;
  double lazyqp_forcing_primal_tolerance = 0.0;

  // Optional one-step primal-dual continuation for successive Lazy-DCA
  // majorants.  The accepted outer point is the next primal origin while the
  // equality multipliers, inequality multipliers and independent slacks are
  // transported transactionally.  Each eligible, nonterminal majorant
  // performs exactly one shared-factor predictor-corrector step before
  // returning an explicitly inexact direction.  A state which already meets
  // the strict inner contract may return without a step.  Failure of the
  // finite/interior/residual/decrease guard rolls the complete trial state
  // back and delegates the retry to the outer model globalization.
  // Disabled by default so the historical r3p24 trajectory is unchanged.
  bool dca_single_loop_primal_dual = false;
  // Multiplier transport is subordinate to the master route.  The historical
  // retraction-fallback switch remains in the API for source compatibility,
  // but defaults off and is rejected when the literal single-loop contract is
  // active: a rejected raw endpoint may request only a new proximal model.
  bool dca_single_loop_multiplier_transport = true;
  bool dca_single_loop_retraction_fallback_only = false;
  int dca_single_loop_inner_steps = 1;
  double dca_single_loop_forcing_eta = 0.9;
  // Deterministic convex curvature used by the literal single-loop route
  // before transported equality multipliers provide Lagrangian curvature.
  // It is part of every single-loop majorant, not a failed-step retry penalty.
  double dca_single_loop_baseline_proximal_regularization = 5e-2;
  // Fail-closed Phase-I bypass for a finite, nearly feasible ORIGINAL-QCQP
  // start.  This is a separate opt-in and is honored only by the Lazy-DCA
  // single-loop route.  Bounds must already satisfy feasibility_tolerance and
  // the full original violation must lie in (feasibility_tolerance, cap].
  // Rejected starts retain the historical Phase-I path unchanged.
  bool dca_single_loop_direct_start = false;
  double dca_single_loop_direct_start_theta_cap = 1e-2;
  double dca_single_loop_direct_start_gamma_h = 5e-2;

  // Public LazyIPDCA-QP compatibility surface for the convex LazyQP backend.
  // The qp_ prefix prevents these controls from being confused with the outer
  // QCQP algorithm.  Profile C with AMRS is the CLI/runner default requested
  // for reproducibility with the production QP solver.  Because every LazyQP
  // subproblem is already convex, atom discovery, curvature trials and the
  // nonconvex DCA stopping controller are recorded but deliberately bypassed;
  // they become active only in a future nonconvex-QP backend.  The numerical
  // IPM controls (backend, initialization metadata, Mehrotra, sigma cap,
  // bound-push policy and model tolerance) are live in the current kernel.
  std::string qp_option = "C";
  bool qp_amrs = true;
  int qp_init = 2;
  std::string qp_backend = "ldl";
  int qp_rho_trial_every = 2;
  int qp_max_atoms = 500;
  double qp_seed_coverage = 0.25;
  double qp_seed_stall_tol = 0.0;
  int qp_seed_patience = 999;
  bool qp_atom_events = true;
  int qp_atom_event_every = 50;
  double qp_bound_push = 0.0;
  double qp_bp_push_kappa = 1e-4;
  double qp_bp_margin_tol = 1e-4;
  int qp_escapes = 3;
  bool qp_mehrotra = true;
  double qp_sigma_max = 1e14;
  double qp_model_solved_tol = 1e-12;
  bool qp_dca_stop = true;
  double qp_dca_stall_tol_rel = 1e-10;
  int qp_dca_stall_iters = 5;
  double qp_dca_monotone_tol_abs = 1e-6;
  bool qp_bp_adaptive = true;

  // KKT representation used by the dedicated LazyQP kernel.
  // auto    : eliminate inequality duals for dense native solves when this
  //           reduces the system materially and all low-rank corrections are
  //           already condensed in the primal Hessian;
  // full    : factor the historical (y,lambda,z[,border]) KKT system;
  // reduced : request exact inequality elimination whenever eligible, with a
  //           guarded fallback to the full system if factorization fails;
  // hybrid  : eliminate sparse inequality rows exactly but retain a bounded
  //           number of dense rows in the KKT border, avoiding their dense
  //           J_i^T D_i J_i outer products.  This is algebraically identical
  //           to reduced and keeps the same guarded full-system fallback.
  std::string lazyqp_kkt_form = "auto";
  int lazyqp_reduced_kkt_min_inequalities = 32;
  int lazyqp_hybrid_kkt_max_retained_inequalities = 2;
  int lazyqp_hybrid_kkt_min_row_nnz = 64;
  // Publish exact affine PGLib apparent-flow pivots to the LazyQP kernel.
  // Detection is deliberately opt-in and fail-closed.
  bool lazyqp_flow_lift_elimination = false;
  // Restrict inequality value/gradient kernels to their validated structural
  // supports.  The generic dense traversal remains the default and is always
  // used for models with a dense shift or a low-rank term.
  bool lazyqp_model_support_fastpath = false;
  bool lazyqp_affine_support_fastpath = false;

  // Representation of retained rank-one curvature corrections in LazyQP.
  // This is deliberately independent of exact_eigen_threshold, which belongs
  // only to the curvature oracle.
  // auto      : condense when n <= lazyqp_low_rank_condense_threshold;
  // condensed : add U diag(beta) U^T directly to the primal Hessian;
  // bordered  : preserve the sparse primal block and carry corrections as KKT
  //             border variables.
  std::string lazyqp_low_rank_form = "auto";
  int lazyqp_low_rank_condense_threshold = 256;

  // Native-IPM initialization.  The SQP multiplier state is propagated to
  // the next convex subproblem, while slacks and nonnegative duals are pushed
  // strictly inside their lower bound at zero.  bound_push/bound_frac mirror
  // the historical LazyIPDCA/Ipopt-style initialization controls.
  bool inner_dual_warm_start = false;
  // Warm start is opt-in. Equality-only propagation must not alter the inequality
  // barrier state (s,z). Full inequality warm start enables bound-push on s,z.
  // This preserves the historical cold-start trajectory by default.
  bool inner_inequality_dual_warm_start = false;
  // A continuation state is accepted only if each initial residual is within
  // this dimensionless factor of the validated centered cold start.  The guard
  // prevents a stale/degenerate multiplier vector from selecting a destructive
  // primal-dual branch while allowing nearby Lazy-DCA majorants to start from
  // their preceding multipliers.
  double lazyqp_warm_start_guard_factor = 2.0;
  // Optional continuation barrier for exact-zero inequality multipliers
  // produced by the outer multiplier mapping and numerical-floor entries in
  // direct same-working-set continuation.  For every such entry the
  // trial warm start uses a strictly positive z whose componentwise s*z is
  // capped by lazyqp_zero_dual_mu_cap.  The complete warm state still has to
  // pass the ordinary residual guard; otherwise LazyQP falls back to the
  // untouched cold state.  Disabled by default to preserve the r3p23 trajectory.
  bool lazyqp_zero_dual_continuation = false;
  double lazyqp_zero_dual_mu_cap = 1e-4;
  double inner_bound_push = 1e-2;
  double inner_bound_frac = 1e-2;
  double inner_bound_push_reference = 1e-6;
  double inner_bound_mult_init_val = 1.0;

  // Lazy curvature discovery.
  double curvature_margin = 0.0;
  double curvature_tolerance = 1e-9;
  double curvature_safety = 0.10;
  // Keep constraint convexification and Lagrangian convexification budgets
  // separate.  A4X12 needs about 44 Lagrangian directions while individual
  // quadratic constraints remain well served by the historical cap 24.
  int max_rank_per_quadratic = 24;
  int max_objective_rank = 64;
  int max_lagrangian_rank = 64; // legacy-SQP only
  double dca_extra_proximal_regularization = 0.0; // optional extra term; default exact DC majorant
  // A restricted working-set master can be unbounded when the original
  // objective is affine: omitted convex constraints may be the only objects
  // that bound the descent direction.  For moderate affine-objective QCQPs,
  // solve the complete constraint majorant.  Large models
  // retain the historical lazy working set through the explicit size limit.
  bool dca_full_constraints_for_affine_objective = true;
  int dca_affine_full_constraint_limit = 4096;
  // An affine objective does not need the complete inequality majorant when
  // every objective-improving coordinate direction is already stopped by a
  // finite simple bound.  This is the common OPF case: generation variables
  // carry the cost and have PMIN/PMAX bounds, while zero-cost flow lifts may
  // remain unbounded.  Omitted inequalities are still checked globally and
  // activated before an outer step is accepted.
  bool dca_affine_box_bounded_lazy = true;

  // Single-route globalization/acceleration.  The exact DCA majorant remains
  // the primary subproblem at every outer iteration.  The optional corrector
  // is a multiplier-aware local model, solved only after the DCA step and
  // accepted exclusively on the original QCQP; it never changes the reported
  // outer route and never invokes the historical SQP loop.
  bool dca_local_corrector = true;
  int dca_local_corrector_start_outer = 2;
  int dca_local_corrector_frequency = 1;

  // Classical SQP still supplies the equality model
  // h(x_k)+grad h(x_k)^T d=0.  When its one-shot Maratos correction does not
  // return to the final QCQP tube, an iterated normal correction may refine
  // that SOC.  This is a globalization safeguard, not a nonlinear equality
  // model and not a second outer route.
  bool dca_iterated_soc = true;
  int dca_iterated_soc_max_iterations = 12;
  double dca_iterated_soc_max_ratio = 2.0;
  double dca_iterated_soc_trigger_ratio = 1e6;

  // A continuation that repeatedly fails is temporarily suppressed instead of
  // being retried and discarded at every outer iteration.  A later probe is
  // automatic after the cooldown.
  int dca_warm_failure_limit = 3;
  int dca_warm_cooldown = 32;

  // A failed convex majorant is retried cold and then with an increased model
  // proximal term.  Numerical KKT regularization remains governed separately
  // by kkt_* above.
  int dca_inner_retry_max = 3;
  double dca_inner_retry_initial_prox = 1e-8;
  double dca_inner_retry_growth = 10.0;
  double dca_inner_retry_max_prox = 1e2;

  // A successfully solved convex majorant may still be nearly flat in a
  // large tangent/null space.  In that case a tiny numerical KKT shift must
  // not be allowed to define the mathematical DCA step.  Re-solve the SAME
  // majorant with an explicit proximal term until its infinity-norm lies in a
  // scale-aware trust envelope.  This remains a proximal Lazy-DCA step; it is
  // not a route switch and it does not alter the original constraints.
  bool dca_step_control = true;
  double dca_step_target_absolute = 1.0;
  double dca_step_target_relative = 1e-2;
  double dca_step_target_maximum = 10.0;
  double dca_step_control_min_prox = 1e-8;
  double dca_step_control_growth = 4.0;
  int dca_step_control_max_retries = 8;

  // Funnel globalization for SQP-linearized quadratic equalities.  The
  // primary Lazy-DCA subproblem always uses h(x_k)+grad h(x_k)^T d=0.  The
  // funnel controls only acceptance on the ORIGINAL nonlinear equalities.
  // When no explicit initial value is supplied, the default is automatically
  // initialized at sqrt(feas_tol), capped by dca_equality_funnel_auto_ceiling.
  // Original inequalities and bounds remain inside the final feasibility tube.
  bool dca_equality_funnel_auto = true;
  double dca_equality_funnel_initial = 0.0;
  double dca_equality_funnel_auto_ceiling = 1e-2;
  double dca_equality_funnel_contraction = 0.9;
  double dca_equality_funnel_growth = 1.25;
  double dca_feasibility_step_fraction = 1e-2;

  // Explicit SQP-style nonlinear equality retraction applied to a scaled
  // Lazy-DCA trial before the more expensive defect-SOC QP.  The equalities
  // remain linearized in the convex subproblem; this correction is a guarded
  // globalization step, evaluated on the original QCQP.  For CORE lifts it
  // uses the validated reduced t=|q| manifold.  It is accepted only when the
  // correction is small relative to the scaled step and does not violate an
  // original inequality or bound beyond the final tube.
  bool dca_equality_retraction = true;
  double dca_equality_retraction_max_ratio = 2.0;

  // Nonlinear quadratic-equality model.
  //
  // linearized
  //   Historical SQP equation h(x_k)+J_k d=0 followed by the generic
  //   retraction/SOC globalization stack.
  //
  // condensed-retraction
  //   Curvature-preserving large-scale route.  The large convex subproblem
  //   keeps the affine tangent equation, but its objective uses the signed
  //   equality-Lagrangian Hessian.  Negative curvature is detected only on
  //   ker(J_k), and the accepted point is mapped back to the exact quadratic
  //   manifold through x_k+d+Y_k z.  The fixed normal map Y_k is applied by a
  //   reusable sparse bordered KKT factorization; Newton--Krylov and branch
  //   continuation operate only in the equality space R^m.
  //
  // elastic-split
  //   Experimental two-majorant elastic alternative retained for comparison.
  std::string dca_equality_model = "linearized";
  double dca_tangent_inertia_trigger = 1.0;

  int dca_condensed_newton_iterations = 8;
  int dca_condensed_krylov_iterations = 16;
  double dca_condensed_krylov_relative_tolerance = 1e-4;
  int dca_condensed_continuation_attempts = 16;
  double dca_condensed_retraction_tolerance = 1e-10;
  double dca_condensed_regularization = 1e-12;
  double dca_condensed_regularization_growth = 10.0;
  double dca_condensed_regularization_max = 1e-4;
  double dca_condensed_retraction_max_ratio = 2.0;
  // Keep the fixed-normal correction off coordinates that lie on a simple
  // bound.  The affine DCA step may still move such a coordinate into the
  // interior; only the nonlinear normal correction is restricted.  This
  // prevents O(alpha^2) equality corrections from crossing an active OPF
  // generator/voltage bound and forcing the outer line search to alpha~0.
  bool dca_condensed_bound_aware_normal = true;
  double dca_condensed_bound_activity_tolerance = 1e-10;
  bool dca_condensed_tangent_curvature = true;
  // Opt-in batched tangent Ritz proposal with exact bordered-inertia gate.
  // OFF preserves the historical ambient/projected route bit for bit.
  bool dca_condensed_projected_multi_ritz_inertia = false;
  // Opt-in exact ambient certificate for structurally block-sparse signed
  // equality-Lagrangian Hessians.  Each connected block is diagonalized in
  // isolation and its modes are embedded back into the ambient space.  When
  // the certificate is available, the tangent projector is unnecessary and
  // the bound-aware retraction map is the only normal factorization of the
  // outer iteration.  Generic default OFF preserves the r3p19 route.
  bool dca_condensed_block_exact_curvature = false;
  int dca_condensed_lanczos_steps = 24;
  int dca_condensed_lanczos_starts = 2;
  int dca_condensed_max_tangent_rank = 64;
  // Once the low-rank budget is exhausted, a randomized projected-Lanczos
  // check can reveal a more negative direction after the first scalar shift.
  // Permit verified, monotone scalar updates without growing the bordered
  // low-rank block.  A value of one recovers the historical one-shot policy.
  int dca_condensed_scalar_safeguard_updates = 8;

  // v0.4.28r1: algebraic accelerations only. These options preserve the
  // v0.4.28 retraction branch, globalization, and KKT termination contract.
  // Y is linear in the equality coordinate, so Y(z+alpha*dz) can be formed
  // without a fresh bordered solve.
  bool dca_condensed_affine_normal_reuse = true;
  // For quadratic equalities, residuals along a Newton line are exact
  // quadratics in alpha. Rejected trial alphas are screened algebraically;
  // any predicted accepted endpoint is still checked by equality_values().
  bool dca_condensed_quadratic_linesearch_reuse = true;

  // v0.4.28r2: implementation-only acceleration controls.  They do not
  // change the equality model or the historical terminal KKT corrector.
  // Sparse equality evaluation preserves the original per-equality arithmetic
  // order (linear sum, quadratic sum, final combination).
  bool dca_condensed_sparse_equality_evaluation = true;
  // Reuse a repeatedly successful positive normal regularization after two
  // consecutive identical outcomes.  Disable for a strict historical A/B.
  bool dca_condensed_cached_regularization = true;
  // Parallelize only across independent equality rows.  The arithmetic inside
  // each row remains serial, so the row values are reproducible.  Runtime OFF
  // by default; use the dedicated CATENARY runner switch to benchmark it.
  bool dca_condensed_parallel_equalities = false;
  int dca_condensed_parallel_min_equalities = 128;

  // Optional r3p18 accuracy schedule for the *retraction* normal solves.
  // Tangent projections, curvature certification, the final retraction map,
  // and the original-QCQP KKT certificate keep the strict global KKT
  // tolerance.  Only intermediate Newton--GMRES maps may use the looser
  // policy, and a failed relaxed retraction is repeated strictly.
  bool dca_condensed_adaptive_normal_refinement = false;
  double dca_condensed_normal_intermediate_backward_error_tolerance = 1e-6;
  double dca_condensed_normal_intermediate_backward_error_reject = 1e-4;
  double dca_condensed_normal_strict_stationarity_factor = 1e3;

  // Optional r3p18 line-search seed.  A successful condensed step seeds the
  // next search one contraction level above its accepted alpha.  Exact
  // equality, inequality, bound, filter, and objective tests are unchanged;
  // if the seeded grid is exhausted, the historical alpha=1 grid is retried.
  // Generic default OFF preserves historical trajectories.  The dedicated
  // PGLib runner enables it explicitly.
  bool dca_condensed_alpha_predictor = false;
  double dca_condensed_alpha_growth = 2.0;
  int dca_condensed_alpha_full_step_frequency = 8;

  // Optional guarded over-relaxation of a stable condensed DCA fixed-point
  // iteration.  After several consecutive alpha=1 steps, probe exactly one
  // alpha>1 endpoint.  The endpoint is retracted and
  // checked against the original QCQP by the ordinary bound, side, funnel,
  // filter, and objective tests.  A rejected probe is never contracted: the
  // complete historical alpha=1 schedule is retried immediately.  Generic
  // default OFF preserves the historical trajectory bit-for-bit.
  bool dca_condensed_outer_extrapolation = false;
  double dca_condensed_outer_extrapolation_alpha = 1.5;
  int dca_condensed_outer_extrapolation_min_full_steps = 3;

  // A normal equality correction can expose a side-constraint violation that
  // is invisible at the affine tangent endpoint.  Optionally add only such an
  // omitted original inequality to the current majorant and re-solve at the
  // same outer iteration.  Modeled offenders and an exhausted ordinary
  // working-set expansion budget retain the historical alpha backtracking.
  // Generic default OFF preserves r3p18 trajectories; application runners may
  // enable the guarded policy explicitly.
  bool dca_condensed_post_retraction_working_set_expansion = false;
  int dca_condensed_post_retraction_working_set_max_expansions = 8;

  // A condensed equality retraction can create the classical Maratos side
  // effect: the raw convex-majorant endpoint satisfies every modeled
  // inequality, while its O(||d||^2) normal correction lies just outside the
  // final feasibility tube.  Optionally let the existing objective/filter
  // globalization accept that *modeled-only* transient violation inside a
  // separate bounded tube.  Omitted inequalities and simple bounds remain on
  // the strict historical path, and final termination always uses
  // feasibility_tolerance exactly.  Generic default OFF preserves historical
  // trajectories; application runners may opt in.
  bool dca_condensed_modeled_side_filter = false;
  double dca_condensed_modeled_side_filter_limit = 1e-3;
  double dca_condensed_modeled_side_recovery_margin = 1.0;

  // Exact Schur elimination for strictly recognized PGLib apparent-flow
  // auxiliaries.  Generic default OFF preserves historical arithmetic; the
  // dedicated PGLib runner enables it explicitly.  The detector fails closed
  // and the bordered normal operator remains the factorization fallback.
  bool dca_condensed_flow_lift_schur = false;

  double dca_equality_elastic_penalty = 10.0;
  double dca_equality_elastic_quadratic = 1e-8;
  double dca_equality_elastic_cap_factor = 1.0;
  bool dca_equality_elastic_post_retraction = false;
  int lanczos_steps = 40;
  int lanczos_starts = 2;
  int exact_eigen_threshold = 256;
  bool strict_scalar_certificate = false;

  // Convex-subproblem SQP backends used only when inner_backend is
  // filter-sqp or tr-sqp.  Their convex QP/QCQP Newton substeps use the
  // LazyIPDCA-QP-derived kernel with the selected linear KKT backend.
  double inner_sqp_filter_gamma_theta = 1e-4;
  double inner_sqp_filter_gamma_f = 1e-8;
  double inner_sqp_line_search_contraction = 0.5;
  int inner_sqp_line_search_max_steps = 24;
  double inner_sqp_tr_initial_radius = 1.0;
  double inner_sqp_tr_min_radius = 1e-10;
  double inner_sqp_tr_max_radius = 1e4;
  double inner_sqp_tr_eta_accept = 0.1;
  double inner_sqp_tr_eta_expand = 0.75;
  double inner_sqp_tr_shrink = 0.25;
  double inner_sqp_tr_expand = 2.0;

  // Phase I: bound-aware exact-QCQP Newton restoration on the ORIGINAL
  // QCQP residuals.  Equalities and violated inequalities are treated
  // together; bounds are enforced by an active-set step.
  // Optional equality-multiplier projection, only after dual stagnation.
  bool majorant_recalc_lambda = false;
  bool majorant_dual_diagnostics = false;
  int majorant_curvature_corrections = 0;
  bool majorant_lambda_interpolation = false;
  bool majorant_lambda_rho_retry = false;
  std::string majorant_lambda_trigger = "consecutive";
  int majorant_recalc_lambda_max_iterations = 40;
  bool majorant_initial_restoration = false;
  // Opt-in, one centrality reset only after terminal filter exhaustion.
  bool majorant_filter_recenter = false;
  // Opt-in paper section 2.3 history augmentation using switching AND Armijo.
  bool majorant_filter_history_fix = false;
  // Opt-in common policy; legacy callers retain their exact startup.
  std::string majorant_startup_policy = "legacy";
  // Internal native failure-recovery context. The historical primary path
  // retains its barrier policy; one charged cold retry may require complete
  // primal-dual barrier progress. Propagation through nested presolve calls
  // prevents another top-level retry from acquiring fresh budgets.
  bool majorant_barrier_retry_consumed = false;
  bool affine_face_quadratic_rewrite_consumed = false;
  bool majorant_complete_barrier_gate = false;
  // Candidate-only complete barrier gate and measured nonlinear SOC window.
  bool majorant_adaptive_globalization = false;
  // The adaptive controller may admit its existing bounded recovery for
  // any still-infeasible anchor, including the historical (tol,100*tol] gap.
  // The switch permits exact legacy replay; it never relaxes acceptance.
  bool majorant_recovery_gap_fix = true;
  bool majorant_restoration_bound_stall_fix = true;
  bool majorant_initial_dual_balance = false;
  // Explicit V2 research policy: dual stabilization of h and source g+s
  // in original R3. Physical bound rows have rho=0. Separate from backend shifts.
  // Defaults off; full sparse equality rows are retained when enabled.
  bool majorant_equality_stabilization = false;
  // Optional SOC proposal for a filter-accepted nonlinear equality defect.
  bool majorant_nonlinear_trial_soc = false;
  bool majorant_rho_feasibility_guard = false;
  int majorant_trial_soc_max_corrections = 1;
  bool majorant_trial_soc_full_primal = false;
  // Optional transactional arc on the first accepted step after the existing
  // barrier gate decreases mu, when a measured SOC fails its size/domain guard.
  // False retains the complete R9 source fallback.
  bool majorant_trial_soc_arc = false;
  // Optional coefficient-verified source level-set repair after a selected
  // SOC remains insufficient. Availability starts after a barrier decrease
  // and ends on anchor-replacing recovery; unsupported geometry keeps R10.
  // False preserves the complete R10 globalization and fallback behavior.
  bool majorant_soc_defect_control = false;
  bool majorant_conditional_retraction = false;
  int majorant_soc_small_step_patience = 0;
  double majorant_soc_small_step_threshold = 0.125;
  int majorant_restoration_iterations = 20;
  bool restoration_enabled = true;
  // Internal recovery context: entered only after a rejected majorant KKT
  // direction/factor at a materially infeasible nonlinear equality point.
  bool restoration_failed_linearization_context = false;
  // Internal precision continuation only: restoration target is strictly
  // below the unchanged final feasibility tolerance after filter rejection.
  // Coarse restoration and historical callers retain absolute-slope gating.
  bool restoration_private_precision_context = false;
  // Internal hybrid Phase-I continuation: legacy and exact/restart phases
  // share restoration_max_iterations. False preserves historical callers.
  bool restoration_shared_iteration_budget = false;
  // Optional absolute steady-clock deadline; infinity preserves legacy callers.
  double restoration_deadline_seconds = std::numeric_limits<double>::infinity();
  int restoration_max_iterations = 600;
  int restoration_restart_iterations = 400;
  int restoration_max_restarts = 16;
  int restoration_bound_active_passes = 12;
  double restoration_lm_initial_damping = 1e-3;
  double restoration_lm_max_damping = 1e24;
  double restoration_bound_activity_tolerance = 1e-12;
  double restoration_step_tolerance = 1e-13;

  // Objective-driven activation of one-anchor quadratic components whose
  // equality Jacobian is initially blind to a nonzero objective gradient.
  // A graph-distance mode is scaled by an exact one-dimensional minimax
  // search on the quadratic equalities before Phase I.  This prevents a flat
  // squared-difference chain from preserving an unrepresentable tangential
  // gradient (CATENARY: 4.905) through many outer iterations.
  bool restoration_tangent_activation = true;

  // v0.2.3 exact-QCQP restoration globalization.
  int restoration_exact_hessian_threshold = 256;
  int restoration_cg_max_iterations = 500;
  double restoration_cg_tolerance = 1e-10;
  // Phase-I exact-CG implementation controls. Workspace reuse does not alter
  // arithmetic. Optional row-dot parallelism keeps one OpenMP team alive for
  // a complete PCG solve.  The sparse-row-dot fast path visits the nonzeros in
  // their original coordinate order and is enabled only after its first
  // product agrees byte-for-byte with the dense historical product.  Any
  // mismatch automatically keeps the dense path for that PCG solve.  J^T(Jv)
  // remains accumulated serially in the historical row order.
  bool restoration_exact_workspace_reuse = true;
  bool restoration_exact_parallel_row_dots = false;
  bool restoration_exact_sparse_row_dots = true;
  int restoration_exact_parallel_min_rows = 128;
  int restoration_line_search_max_steps = 28;
  double restoration_line_search_contraction = 0.5;
  double restoration_armijo = 1e-4;
  double restoration_theta_fraction = 1e-4;
  double restoration_theta_merit_growth = 1.10;
  double restoration_min_theta_improvement = 1e-7;
  int restoration_stagnation_iterations = 80;
  double restoration_stagnation_relative_tolerance = 1e-8;

  // Terminal L-infinity epigraph restoration, used only if the L2 Newton
  // Phase-I remains infeasible.
  bool restoration_minimax_enabled = true;
  int restoration_minimax_max_iterations = 80;
  double restoration_minimax_proximal = 1e-6;
  double restoration_minimax_theta_fraction = 1e-3;

  // SQP second-order model.  This is deliberately separate from the lazy
  // corrections so diagnostics never count numerical stabilization as
  // discovered negative curvature.
  double proximal_regularization = 1e-6;
  double proximal_regularization_max = 1e4;
  double multiplier_regularization = 1e-10;
  int equality_retraction_iterations = 12;
  double equality_retraction_tolerance = 1e-10;
  double equality_retraction_regularization = 1e-12;

  // Phase-II inequality model.  Once Phase I has delivered a feasible point
  // and a fresh QP-dual multiplier state exists, standard SQP uses tangent
  // half-spaces for nonlinear inequalities.  Keeping a fixed convex quadratic
  // majorant asymptotically adds an O(||d||^2) inner restriction and can turn
  // the local method into an MM/CCP iteration.  The first outer model remains
  // conservative; callers can force the historical majorants for A/B tests.
  bool phase2_linearized_inequalities = true;

  double working_set_activation = 1e-4;
  double working_set_add_tolerance = 1e-7;
  int max_working_set_expansions = 20;
  // Hysteretic working-set dropping: a modeled inequality may leave the set
  // only if it has material slack AND an essentially zero SQP multiplier.
  bool working_set_dropping = true;
  double working_set_drop_slack = 2e-4;
  double working_set_drop_multiplier = 1e-9;
  // Require the drop test at several consecutive accepted points.  A
  // constraint introduced by a full-step look-ahead is otherwise often
  // removed immediately after line-search contraction, then re-added by the
  // next full trial.
  int working_set_drop_patience = 3;

  // Optional guarded local KKT acceleration.  It remains enabled for small
  // regression-scale systems, where it is effective, but is OFF by default for
  // n>64.  A4X12 established that the global QP-dual SQP is more reliable than
  // the experimental large-scale terminal accelerator.
  bool terminal_kkt_enabled = true;
  bool terminal_kkt_large_scale_enabled = false;
  int terminal_kkt_start_outer = 20;
  // Explicit large-scale opt-in uses this autonomous iteration limit.  Every
  // accepted step still passes fresh-dual, inertia/backward-error, original-
  // QCQP residual and objective-monotonicity guards; rejection rolls back.
  int terminal_kkt_max_iterations = 16;
  double terminal_kkt_trigger_stationarity = 1e-2;
  double terminal_kkt_active_slack = 1e-5;
  double terminal_kkt_multiplier_tolerance = 1e-9;
  double terminal_kkt_multiplier_relative_tolerance = 1e-8;
  int terminal_kkt_active_stability_iterations = 3;
  double terminal_kkt_primal_regularization = 1e-10;
  double terminal_kkt_dual_regularization = 1e-10; // normal-step Gram regularization

  double fixed_bound_tolerance = 1e-14;
  double fraction_to_boundary = 0.995;
  double minimum_positive = 1e-12;

  double filter_gamma_theta = 1e-4;
  double filter_gamma_f = 1e-8;
  // Feasible-axis globalization: the filter margin may vanish with theta, but
  // accepted objective steps must still satisfy Armijo decrease against the
  // original objective directional derivative.
  double line_search_armijo = 1e-4;
  // Guarded second-order correction (SOC) for the linearized Phase-II model.
  // SOC is attempted at the CURRENT backtracking scale, not only at alpha=1.
  // A correction is considered successful only if it restores original-QCQP
  // feasibility; otherwise the proven v0.2.12 local feasibility repair remains
  // available as a fallback and Armijo still decides acceptance.
  bool second_order_correction = true;
  double second_order_correction_max_ratio = 0.5;
  double second_order_correction_trigger_ratio = 1000.0;
  double line_search_contraction = 0.5;
  int line_search_max_steps = 28;
  double minimum_step_length = 1e-10;
  // v0.30 contract: allow disabling the tangential globalization fallback.
  bool tangential_globalization_enabled = true;

  // Phase-II trust-region globalization (r5).  The trust region is imposed
  // directly on the SQP displacement d through ||d||_2 <= Delta, so a
  // rejected trial changes the next subproblem and direction rather than
  // merely scaling the same step.  Line-search/tangential globalization is
  // bypassed when this mode is enabled.
  bool trust_region_enabled = false;
  double trust_region_initial_radius = 10.0;
  double trust_region_min_radius = 1e-10;
  double trust_region_max_radius = 10.0;
  double trust_region_eta_accept = 0.10;
  double trust_region_eta_shrink = 0.25;
  double trust_region_eta_expand = 0.75;
  double trust_region_shrink_factor = 0.25;
  double trust_region_expand_factor = 2.0;
  double trust_region_boundary_fraction = 0.80;
  // Numerical admissibility of the solved L2 trust-region boundary.  The
  // native inner IPM solves 1/2||d||^2 <= 1/2 Delta^2 only to finite precision;
  // this tolerance is deliberately tiny and only prevents false outer rejects.
  double trust_region_boundary_relative_tolerance = 1e-4;
  double trust_region_boundary_absolute_tolerance = 1e-9;
  int trust_region_max_rejections = 18;
  // r6 exact-penalty trust-region acceptance.  The hard feasibility cap is
  // the configured original-QCQP feasibility tolerance; the merit ratio uses
  // Phi_sigma(x)=f(x)+sigma*theta(x), with sigma safely above the current
  // original-QCQP multiplier l1 norm.  This avoids the r5 1e-10 funnel that
  // collapsed Delta on feasible iterates while still pricing feasibility loss.
  double trust_region_feasibility_floor = 1e-12;
  double trust_region_feasibility_growth = 2.0; // retained for CLI compatibility
  double trust_region_feasibility_reduction = 0.10;
  double trust_region_penalty_min = 10.0;
  double trust_region_penalty_safety = 1.25;


  // ---------------------------------------------------------------------
  // v0.4.28r3p28: Lazy primal-dual Newton route (--algorithm lazy-newton).
  // All defaults are inert for the historical routes.
  double ln_mu_init = 0.1;             // initial barrier parameter
  double ln_mu_kappa = 0.2;            // linear barrier decrease factor
  double ln_mu_theta = 1.5;            // superlinear barrier decrease exponent
  double ln_kappa_epsilon = 10.0;      // barrier subproblem accuracy factor
  double ln_activation = 0.05;         // lazy working-set activation (scaled distance)
  double ln_envelope = 0.2;            // structural reservation distance at the start
  int ln_drop_patience = 8;            // iterations before dropping an inactive row
  double ln_drop_distance = 1.0;       // scaled distance beyond which a row may be dropped
  double ln_delta_c = 1e-8;            // dual (quasi-definite) regularization
  double ln_delta_x_init = 1e-4;       // first inertia shift trial
  double ln_delta_x_max = 1e20;        // inertia shift cap before failure
  double ln_delta_x_min = 1e-10;       // floor of the warm-started inertia shift
  int ln_kkt_refine = 3;               // iterative refinement sweeps
  double ln_kkt_berr_tol = 1e-10;      // refinement stop (relative backward error)
  bool ln_quadratic_correction = true; // exact second-order equality correction
  int ln_max_soc = 4;                  // correction sweeps per line search
  double ln_bound_push = 1e-2;         // initial bound push (relative)
  double ln_tau_min = 0.99;            // fraction-to-boundary floor
  int ln_max_line_search = 30;         // Armijo backtracks
  bool ln_flow_elimination = false;    // reserved: analytic lift elimination
  int ln_certificate_every = 1;        // 0 = only on the final internal test
  double ln_prox_alpha_min = 0.0;  // 0 disables the guard
  double ln_mult_init_max = 1e3;       // discard the least-squares multiplier start above this
  double ln_scaling_max_gradient = 100.0; // gradient-based row/objective scaling target (0 = off)
  std::string ln_inequality_curvature = "adaptive"; // adaptive | majorant (PSD DC part) | exact
  double ln_adaptive_curvature_delta = 1.0; // shift above which adaptive reverts to the majorant
  std::string ln_globalization = "filter"; // filter (Fletcher-Leyffer) | merit (l1 exact penalty)
  int ln_condense_max_support = 16;
  double ln_transport_complementarity_floor = 1e3; // lazyqp-sparse: keep s*z >= mu/kappa on return (0 = off)   // working-set rows with at most this many free variables are condensed
  std::string ln_ordering = "amd";     // fill-reducing ordering: amd (if compiled) | metis     // blocked-step threshold for the adaptive proximal shift

  bool verbose = true;
  int print_every = 1;
  // Compact, flushed outer-iteration table.  This is intentionally
  // independent of verbose so production runs can use --quiet together with
  // --live-iterations and avoid the detailed inner-kernel trace.
  bool live_iterations = false;
  int live_iteration_every = 1;
  int live_iteration_header_every = 20;
  unsigned random_seed = 7;
  bool power2_scaling = false; // experimental exact dyadic coordinate and row units
  bool product_definition_presolve = false; // experimental exact unit-product elimination
  // Native PHR recovery; all optimization remains in this library. Explicitly
  // opt in while the historical campaign profiles retain their exact route.
  bool native_augmented_repair = false;
};

} // namespace lazyipdca::qcqp
