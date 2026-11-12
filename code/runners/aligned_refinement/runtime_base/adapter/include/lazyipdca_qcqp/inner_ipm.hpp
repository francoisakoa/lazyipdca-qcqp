#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <string>

namespace lazyipdca::qcqp {

struct SparseRow {
  std::vector<std::pair<int,double>> entries;
};

struct LinearEqualitySystem {
  int nvar = 0;
  std::vector<SparseRow> rows;
  Vec rhs; // A y = rhs
};

struct ConvexQuadraticModel {
  double constant = 0.0;
  Vec linear;
  SymMatrix base_hessian;
  int shift_dimension = 0;
  double scalar_shift = 0.0;
  std::vector<LowRankTerm> low_rank;

  double value(const Vec& y) const;
  Vec gradient(const Vec& y) const;
};

struct ExactAffineElimination {
  int equality_row = -1;
  int variable = -1;
};

struct ConvexSubproblem {
  int nvar = 0;
  int original_dimension = 0;
  // Optional stricter mean s*z target. NaN preserves historical tolerances.
  double complementarity_tolerance = std::numeric_limits<double>::quiet_NaN();
  ConvexQuadraticModel objective;
  LinearEqualitySystem equalities;
  std::vector<ConvexQuadraticModel> inequalities; // <= 0
  std::vector<ExactAffineElimination> exact_affine_eliminations;

  // The global Settings switch is necessary but not sufficient: only the
  // primary Lazy-DCA majorant may return a provisional single-loop state.
  // Restoration, SOC, local correctors and standalone subproblems retain the
  // strict historical contract unless their caller marks them explicitly.
  bool allow_dca_single_loop = false;

  // The outer Lazy-DCA filter may own globalization of a provisional
  // predictor-corrector endpoint.  Standalone kernel callers leave this
  // false and retain the strict residual-contraction transaction.
  bool allow_dca_single_loop_outer_filter_progress = false;

  // One-shot authority for initializing the first single-loop majorant in
  // its final (possibly affine-reduced) coordinates.  It is never propagated
  // to later outer models or proximal retries.
  bool allow_dca_single_loop_dual_bootstrap = false;

  // Optional SQP-to-IPM dual warm start. Equality entries are free;
  // inequality entries use the canonical g<=0, mu>=0 convention.
  bool has_dual_warm_start = false;
  // True only for a multiplier vector reconstructed by the outer QCQP
  // mapping.  This provenance marker is diagnostic: the guarded numerical-
  // zero policy may repair either mapped or direct transported multipliers,
  // while retaining distinct counters for the two paths.
  bool dual_warm_start_is_mapped = false;
  Vec initial_equality_duals;
  Vec initial_inequality_duals;

  // Optional primal continuation.  The Lazy-DCA working-set loop uses this
  // only when an accepted restricted-model direction reveals omitted
  // inequalities: the enriched model starts from that direction instead of
  // discarding it and restarting at y=0.
  bool has_primal_warm_start = false;
  Vec initial_primal;

  // Optional independent-slack continuation.  This is kept separate from
  // the inequality multiplier warm start so older callers preserve their
  // centered slack construction exactly.  The dedicated single-loop route
  // accepts the complete (y,lambda,z,s) state only through the ordinary warm
  // residual guard; otherwise every component is rolled back to the centered
  // cold state.
  bool has_inequality_slack_warm_start = false;
  Vec initial_inequality_slacks;

  // v0.4.0-r2: the Lazy-DCA outer loop starts each convex majorant at y=0,
  // i.e. at the current outer iterate.  After Phase I this point is already
  // feasible (up to the outer feasibility tolerance), so the native IPM may
  // initialize directly near the central path instead of forcing the legacy
  // s>=1,z=1 cold start used by restoration/SQP auxiliary problems.
  bool prefer_feasible_centered_start = false;
};

struct InnerResult {
  bool success = false;
  bool used_iteration_limit_fallback = false;
  // Set only when the opt-in forcing contract, rather than the historical
  // strict inner contract, authorized this majorant direction.
  bool lazyqp_inexact_forcing_accepted = false;
  bool lazyqp_inexact_forcing_attempted = false;
  bool lazyqp_inexact_forcing_strict_fallback = false;
  double lazyqp_inexact_forcing_initial_residual = 0.0;
  double lazyqp_inexact_forcing_outer_target = 0.0;
  double lazyqp_inexact_forcing_target = 0.0;
  double lazyqp_inexact_forcing_primal_target = 0.0;
  bool dca_single_loop_attempted = false;
  bool dca_single_loop_accepted = false;
  // Transaction rolled back cleanly after one predictor-corrector step.  The
  // outer loop may rebuild a more proximal majorant and try one new step; this
  // is neither a solved subproblem nor a certificate.
  bool dca_single_loop_retryable = false;
  bool dca_single_loop_strict_fallback = false;
  bool dca_single_loop_state_transport_used = false;
  bool dca_single_loop_state_transport_rejected = false;
  bool dca_single_loop_reduced_bootstrap_attempted = false;
  bool dca_single_loop_reduced_bootstrap_accepted = false;
  int dca_single_loop_steps = 0;
  int dca_single_loop_factorizations = 0;
  double dca_single_loop_initial_residual = 0.0;
  double dca_single_loop_final_residual = 0.0;
  // Private continuation audit state.  Exact affine elimination expands this
  // snapshot independently so contraction is certified in the untouched
  // full model, not by comparing residuals from different coordinate spaces.
  Vec dca_single_loop_initial_primal;
  Vec dca_single_loop_initial_equality_duals;
  Vec dca_single_loop_initial_inequality_duals;
  Vec dca_single_loop_initial_inequality_slacks;
  // Full model/state passes performed solely for the optional forcing guard.
  // This must remain zero when lazyqp_inexact_forcing is disabled.
  int lazyqp_inexact_forcing_model_evaluations = 0;
  // Audit counters for the opt-in structural-support kernels.  They allow
  // differential tests and release runners to prove that parity is not
  // vacuous and that unsupported dense models fail over to the generic path.
  int lazyqp_model_support_fastpath_requests = 0;
  int lazyqp_model_support_fastpath_effective_subproblems = 0;
  int lazyqp_model_support_builds = 0;
  int lazyqp_model_support_eligible_models = 0;
  int lazyqp_model_support_generic_fallback_models = 0;
  int lazyqp_model_support_value_evaluations = 0;
  int lazyqp_model_support_gradient_evaluations = 0;
  std::string message;
  Vec y;
  Vec equality_duals;
  Vec inequality_duals;
  // Native LazyQP's independent positive inequality slacks.  Keeping these
  // in the private inner result lets exact affine reductions validate the
  // complete full-model primal-dual state before it is returned.  Other
  // inner backends may leave this empty.
  Vec inequality_slacks;
  int iterations = 0;
  int factorizations = 0;
  int symbolic_analyses = 0;
  int symbolic_reuses = 0;
  int symbolic_invalidations = 0;
  // Numerical assembly work reported by sparse factorization backends.  The
  // current producer is SuiteSparse LDL's positional triplet-to-CSC scatter.
  int numeric_scatter_builds = 0;
  int numeric_scatter_reuses = 0;
  int numeric_scatter_invalidations = 0;
  double dual_inf = 0.0;
  double equality_inf = 0.0;
  double inequality_inf = 0.0;
  double primal_residual_inf = 0.0;
  double complementarity = 0.0;
  double scaled_model_residual_inf = 0.0;
  // Numerical KKT diagnostics (do not enter the optimization model).
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
  // LazyQP split-step diagnostics. The primal and dual fraction-to-boundary
  // maxima are scaled by one common globalization factor tau.
  int lazyqp_scaled_split_steps = 0;
  double lazyqp_max_primal_dual_step_ratio = 1.0;
  double lazyqp_max_dual_primal_step_ratio = 1.0;
  double lazyqp_min_globalization_tau = 1.0;
  bool dual_warm_start_used = false;
  bool full_dual_warm_start_used = false;
  bool dual_warm_start_rejected = false;
  int lazyqp_zero_dual_continuation_attempts = 0;
  int lazyqp_zero_dual_continuation_mapped_attempts = 0;
  int lazyqp_zero_dual_continuation_direct_attempts = 0;
  int lazyqp_zero_dual_continuation_acceptances = 0;
  int lazyqp_zero_dual_continuation_fallbacks = 0;
  int lazyqp_zero_dual_continuation_repairs = 0;
  bool primal_warm_start_used = false;
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
  // Persistent reduced-KKT input layout diagnostics.  A build records an
  // atomically validated coordinate layout, while a reuse updates only its
  // numerical values.  Any coordinate/count/dimension mismatch invalidates
  // the old layout before a replacement can be committed.
  int lazyqp_kkt_triplet_layout_builds = 0;
  int lazyqp_kkt_triplet_layout_reuses = 0;
  int lazyqp_kkt_triplet_layout_invalidations = 0;
  int lazyqp_numeric_kkt_reuse_attempts = 0;
  int lazyqp_numeric_kkt_reuse_acceptances = 0;
  int lazyqp_numeric_kkt_reuse_fallbacks = 0;
  int lazyqp_numeric_kkt_reuse_iterations = 0;
  int lazyqp_kkt_dimension = 0;
  int lazyqp_full_kkt_dimension = 0;
  int lazyqp_affine_elimination_attempts = 0;
  int lazyqp_affine_elimination_acceptances = 0;
  int lazyqp_affine_elimination_fallbacks = 0;
  int lazyqp_affine_eliminated_variables = 0;
  int lazyqp_affine_eliminated_equalities = 0;
  int lazyqp_affine_reduced_variables = 0;
  int lazyqp_affine_reduced_equalities = 0;
  // Exact rotated-cone formulation diagnostics.  The SOCP backend eliminates
  // cone auxiliaries analytically before entering the sparse Newton solve;
  // these counters prove that the conic dispatcher and its structural audit
  // were exercised rather than silently aliasing the ordinary backend.
  int socp_formulation_attempts = 0;
  int socp_formulation_builds = 0;
  int socp_formulation_rejections = 0;
  int socp_formulation_audit_failures = 0;
  int socp_objective_rotated_cones = 0;
  int socp_inequality_rotated_cones = 0;
  int socp_linear_nonnegative_cones = 0;
  int socp_affine_equalities = 0;
  long long socp_explicit_lift_cone_dimension = 0;
  int socp_condensed_newton_iterations = 0;
  double socp_max_rotated_cone_violation = 0.0;
  int initialization_bound_pushes = 0;
  double initial_primal_residual_inf = 0.0;
  double initial_complementarity = 0.0;
};

class ConvexQCQPInteriorPoint {
 public:
  explicit ConvexQCQPInteriorPoint(const Settings& settings) : st_(settings) {}
  InnerResult solve(const ConvexSubproblem& p) const;

 private:
  const Settings& st_;
};

} // namespace lazyipdca::qcqp
