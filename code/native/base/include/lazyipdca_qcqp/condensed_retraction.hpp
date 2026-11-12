#pragma once

#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace lazyipdca::qcqp {

// Result of the fixed-normal condensed quadratic retraction
//
//   x_trial = x_base + step + Y z,
//   h(x_trial) = target,
//   Y = J_base^T (J_base J_base^T)^{-1}
//
// (with the selected sparse symmetric-indefinite backend used through the
// equivalent bordered KKT system).  The nonlinear solve is performed only in
// z \in R^m, never in the full R^n space.
struct CondensedRetractionResult {
  bool success = false;
  std::string message;
  Vec x;
  Feasibility feasibility;
  // Exact original inequality values at x.  Retraction already evaluates
  // them for its feasibility certificate; retaining them lets the caller
  // inspect that same arithmetic result without another O(m_i n) pass.
  Vec inequality_values;
  double equality_residual_inf = 0.0;
  double correction_inf = 0.0;
  // When the accepted base is feasible only to the numerical retraction
  // tolerance, the total normal correction contains a base-residual term
  // independent of the trial step.  The ratio guard uses the incremental
  // correction after subtracting its fixed-normal linear prediction.
  double baseline_correction_inf = 0.0;
  double incremental_correction_inf = 0.0;
  double normal_coordinate_inf = 0.0;
  double correction_ratio = 0.0;
  int continuation_attempts = 0;
  int continuation_acceptances = 0;
  int newton_iterations = 0;
  int krylov_iterations = 0;
  int line_search_backtracks = 0;
  int normal_solves = 0;
  // Algebraic acceleration counters. They do not alter the accepted endpoint:
  // every candidate predicted to pass the quadratic screen is still checked
  // with the original equality evaluator.
  int affine_normal_map_reuses = 0;
  // Number of exact accepted endpoints reused as the subsequent Newton state.
  int newton_endpoint_reuses = 0;
  // `quadratic_linesearch_reuses` counts candidates handled by the enabled
  // quadratic-reuse policy, including the exact-first alpha=1 fast path.
  int quadratic_linesearch_reuses = 0;
  // Actual slope/curvature constructions, deferred until alpha=1 is rejected.
  int quadratic_linesearch_predictor_builds = 0;
  int quadratic_linesearch_exact_checks = 0;
  int quadratic_linesearch_early_rejections = 0;
  std::int64_t quadratic_linesearch_rows_skipped = 0;
};

struct TangentConvexificationResult {
  bool success = false;
  std::string message;
  int oracle_calls = 0;
  int rank_added = 0;
  int scalar_safeguard_updates = 0;
  double scalar_shift_added = 0.0;
  double initial_min_eigenvalue = 0.0;
  double final_min_eigenvalue = 0.0;
};

// Fixed-base normal operator and tangent curvature oracle for quadratic
// equalities.  A single sparse bordered factorization is reused for:
//   * Y r normal corrections;
//   * tangent projection P v = v - Y J v;
//   * every Newton--Krylov matvec of the condensed retraction;
//   * projected Lanczos curvature tests.
class CondensedQuadraticRetraction {
 public:
  explicit CondensedQuadraticRetraction(
      const Settings& settings, bool allow_flow_lift_schur = false);

  bool factor(const QCQPProblem& problem, const Vec& base, std::string& message);
  // Retraction-only normal map.  In addition to genuinely fixed variables,
  // coordinates on a simple-bound face at the base are excluded from Y.  Use
  // a separate object from the tangent-curvature
  // projector so its symmetric equality-manifold geometry is unchanged.
  bool factor_bound_aware(const QCQPProblem& problem, const Vec& base,
                          std::string& message);
  // Also protect a bound face reached by the raw tangent endpoint base+step.
  // The historical overload above remains the base-only compatibility path.
  bool factor_bound_aware(const QCQPProblem& problem, const Vec& base,
                          const Vec& predicted_step, std::string& message);
  int active_bound_coordinates(const QCQPProblem& problem,
                               const Vec& base) const;
  int active_bound_coordinates(const QCQPProblem& problem, const Vec& base,
                               const Vec& predicted_step) const;
  bool ready() const { return factored_; }

  CondensedRetractionResult retract(
      const Vec& step, const Vec& final_targets = {},
      bool allow_adaptive_normal = false) const;

  TangentConvexificationResult ensure_tangent_convex(
      const SymMatrix& hessian, Convexification& convexification,
      double target_margin, int max_rank);
  TangentConvexificationResult ensure_tangent_convex_multi_ritz(
      const SymMatrix& hessian, Convexification& convexification,
      double target_margin, int max_rank);
  bool certify_tangent_convex_by_inertia(
      const SymMatrix& hessian, const Convexification& convexification,
      double floor, std::string& message);

  bool apply_right_inverse(const Vec& rhs, Vec& correction,
                           std::string& message) const;
  bool project_tangent(const Vec& v, Vec& projected,
                       std::string& message) const;
  Feasibility evaluate_feasibility(const Vec& x) const;

  int factorization_count() const { return factorization_count_; }
  int solve_count() const { return solve_count_; }
  int symbolic_analyses() const { return symbolic_analyses_; }
  int symbolic_reuses() const { return symbolic_reuses_; }
  int symbolic_invalidations() const { return symbolic_invalidations_; }
  int cached_regularization_attempts() const { return cached_regularization_attempts_; }
  int cached_regularization_hits() const { return cached_regularization_hits_; }
  int sparse_equality_evaluations() const { return sparse_equality_evaluations_; }
  int parallel_equality_batches() const { return parallel_equality_batches_; }
  int normal_refinement_steps() const { return normal_refinement_steps_; }
  int normal_refinement_zero_solves() const { return normal_refinement_zero_solves_; }
  int normal_refinement_one_solves() const { return normal_refinement_one_solves_; }
  int normal_refinement_two_solves() const { return normal_refinement_two_solves_; }
  int normal_refinement_three_plus_solves() const { return normal_refinement_three_plus_solves_; }
  double max_normal_initial_backward_error() const { return max_normal_initial_backward_error_; }
  double sum_normal_initial_backward_error() const { return sum_normal_initial_backward_error_; }
  double sum_normal_final_backward_error() const { return sum_normal_final_backward_error_; }
  double max_normal_backward_error() const { return max_normal_backward_error_; }
  int gmres_workspace_growths() const { return gmres_workspace_growths_; }
  int zero_rhs_solve_skips() const { return zero_rhs_solve_skips_; }
  int relaxed_normal_solves() const { return relaxed_normal_solves_; }
  int strict_normal_solves() const { return strict_normal_solves_; }
  int relaxed_accepts_above_strict_tolerance() const {
    return relaxed_accepts_above_strict_tolerance_;
  }
  int strict_retry_retractions() const { return strict_retry_retractions_; }
  double regularization() const { return regularization_; }
  bool exact_right_inverse() const { return regularization_ == 0.0; }
  int equality_dimension() const { return me_; }
  int tangent_curvature_calls() const { return tangent_curvature_calls_; }
  int tangent_inertia_factorizations() const {
    return tangent_inertia_factorizations_;
  }
  int tangent_inertia_symbolic_analyses() const {
    return tangent_inertia_symbolic_analyses_;
  }
  int tangent_inertia_symbolic_reuses() const {
    return tangent_inertia_symbolic_reuses_;
  }
  int tangent_inertia_symbolic_invalidations() const {
    return tangent_inertia_symbolic_invalidations_;
  }
  int locked_coordinates() const { return locked_coordinates_; }
  // Exact block elimination for PGLib apparent-flow auxiliaries.  The route
  // is enabled only after a strict structural check; otherwise the historical
  // (n+m)-dimensional bordered normal system is used unchanged.
  bool flow_lift_schur_active() const { return flow_lift_schur_active_; }
  int flow_lift_schur_detected_variables() const {
    return static_cast<int>(flow_lifts_.size());
  }
  int normal_factor_dimension() const { return normal_factor_dimension_; }
  int flow_lift_schur_factorizations() const {
    return flow_lift_schur_factorizations_;
  }
  int flow_lift_schur_fallbacks() const {
    return flow_lift_schur_fallbacks_;
  }

 private:
  struct SparseGradientRow {
    std::vector<std::pair<int, double>> entries;
  };
  struct SparseGradientComponent {
    int index = 0;
    double linear = 0.0;
    std::vector<std::pair<int, double>> hessian_terms;
  };
  struct SparseEqualityData {
    std::vector<std::pair<int, double>> linear_entries;
    std::vector<SparseGradientComponent> gradient_components;
  };
  struct SparseValueData {
    // Entries retain the dense vector's increasing-coordinate order.  Zero
    // coefficients are the only omitted operations.
    std::vector<std::pair<int, double>> linear_entries;
  };
  struct FlowLiftEntry {
    int equality = -1;
    int variable = -1;
    double sign = 1.0;
  };

  const Settings& st_;
  bool allow_flow_lift_schur_ = false;
  // Factor implementations retain a reference to this private copy.  Only
  // retraction-internal solves may temporarily replace its refinement
  // thresholds; the public right inverse, tangent projector and final
  // retraction reconstruction always use the original strict thresholds.
  mutable Settings factor_settings_;
  std::unique_ptr<SymmetricFactorization> factor_;
  mutable Settings tangent_certificate_settings_;
  std::unique_ptr<SymmetricFactorization> tangent_certificate_factor_;
  const QCQPProblem* problem_ = nullptr;
  Vec base_;
  Vec base_values_;
  std::vector<SparseGradientRow> base_gradients_;
  std::vector<double> row_scale_;
  std::vector<unsigned char> fixed_;
  std::vector<SparseEqualityData> equality_sparse_data_;
  const QCQPProblem* equality_sparse_problem_ = nullptr;
  std::vector<SparseValueData> inequality_sparse_data_;
  const QCQPProblem* inequality_sparse_problem_ = nullptr;
  mutable Vec inequality_values_work_;
  mutable Vec normal_work_;
  // Persistent Krylov/Newton workspaces. They remove repeated heap allocation
  // from the condensed Newton loop without changing any arithmetic decision.
  mutable std::vector<Vec> gmres_v_;
  mutable std::vector<Vec> gmres_mapped_basis_;
  mutable Vec gmres_h_;
  mutable Vec gmres_cs_;
  mutable Vec gmres_sn_;
  mutable Vec gmres_g_;
  mutable Vec gmres_point_delta_;
  mutable Vec gmres_normal_;
  mutable Vec gmres_w_;
  mutable Vec gmres_y_;
  // Persistent continuation/Newton workspaces.  The retraction object is
  // already stateful (factor workspace and diagnostics), so retaining these
  // buffers does not reduce its thread-safety contract.
  mutable Vec continuation_targets_;
  mutable Vec continuation_z_;
  mutable Vec continuation_trial_z_;
  mutable Vec continuation_stage_targets_;
  mutable Vec newton_scaled_step_;
  mutable Vec newton_correction_;
  mutable Vec newton_point_;
  mutable Vec newton_residual_;
  mutable Vec newton_rhs_;
  mutable Vec newton_dz_;
  mutable Vec newton_mapped_dz_;
  mutable Vec newton_slope_;
  mutable Vec newton_curvature_;
  mutable Vec newton_ztrial_;
  mutable Vec newton_ptrial_;
  mutable Vec newton_ctrial_;
  mutable Vec newton_rtrial_;
  mutable int gmres_workspace_growths_ = 0;
  mutable int zero_rhs_solve_skips_ = 0;
  mutable int relaxed_normal_solves_ = 0;
  mutable int strict_normal_solves_ = 0;
  mutable int relaxed_accepts_above_strict_tolerance_ = 0;
  mutable int strict_retry_retractions_ = 0;
  mutable bool relaxed_normal_solve_failed_ = false;
  double cached_positive_regularization_ = 0.0;
  int cached_positive_regularization_streak_ = 0;
  int n_ = 0;
  int me_ = 0;
  bool factored_ = false;
  double regularization_ = 0.0;
  mutable int factorization_count_ = 0;
  mutable int solve_count_ = 0;
  mutable int symbolic_analyses_ = 0;
  mutable int symbolic_reuses_ = 0;
  mutable int symbolic_invalidations_ = 0;
  mutable int cached_regularization_attempts_ = 0;
  mutable int cached_regularization_hits_ = 0;
  mutable int sparse_equality_evaluations_ = 0;
  mutable int parallel_equality_batches_ = 0;
  mutable int normal_refinement_steps_ = 0;
  mutable int normal_refinement_zero_solves_ = 0;
  mutable int normal_refinement_one_solves_ = 0;
  mutable int normal_refinement_two_solves_ = 0;
  mutable int normal_refinement_three_plus_solves_ = 0;
  mutable double max_normal_initial_backward_error_ = 0.0;
  mutable double sum_normal_initial_backward_error_ = 0.0;
  mutable double sum_normal_final_backward_error_ = 0.0;
  mutable double max_normal_backward_error_ = 0.0;
  int tangent_curvature_calls_ = 0;
  int tangent_inertia_factorizations_ = 0;
  int tangent_inertia_symbolic_analyses_ = 0;
  int tangent_inertia_symbolic_reuses_ = 0;
  int tangent_inertia_symbolic_invalidations_ = 0;
  int locked_coordinates_ = 0;
  std::vector<FlowLiftEntry> flow_lifts_;
  std::vector<int> flow_core_equalities_;
  std::vector<int> flow_base_variables_;
  std::vector<int> flow_full_to_base_;
  std::vector<int> flow_equality_to_lift_;
  const QCQPProblem* flow_lift_structure_problem_ = nullptr;
  bool flow_lift_schur_candidate_ = false;
  bool flow_lift_schur_active_ = false;
  int normal_factor_dimension_ = 0;
  mutable int flow_lift_schur_factorizations_ = 0;
  mutable int flow_lift_schur_fallbacks_ = 0;
  std::mt19937 rng_;

  bool factor_impl(const QCQPProblem& problem, const Vec& base,
                   bool bound_aware, const Vec* predicted_step,
                   std::string& message);
  bool coordinate_on_active_bound(const QCQPProblem& problem,
                                  const Vec& base, int j,
                                  const Vec* predicted_step = nullptr) const;

  void build_sparse_equality_data(const QCQPProblem& problem);
  void build_sparse_inequality_data(const QCQPProblem& problem);
  bool detect_flow_lift_schur_structure(const QCQPProblem& problem);
  bool parallel_equalities_enabled() const;
  double sparse_equality_value(int equality_index, const Vec& x) const;
  double sparse_inequality_value(int inequality_index, const Vec& x) const;
  void inequality_values_into(const Vec& x, Vec& values) const;
  Feasibility feasibility_from_equality_values(const Vec& x,
                                                const Vec& equality_values,
                                                Vec* inequality_values = nullptr) const;
  Vec equality_values(const Vec& x) const;
  void equality_values_into(const Vec& x, Vec& values) const;
  SparseGradientRow sparse_gradient(int equality_index, const Vec& x) const;
  double sparse_row_dot(const SparseGradientRow& row, const Vec& v) const;
  double symmetric_bilinear(const SymMatrix& matrix, const Vec& x,
                            const Vec& y) const;
  Vec base_jacobian_product(const Vec& v) const;
  Vec jacobian_product_from_delta(const Vec& delta, const Vec& v) const;
  void jacobian_product_from_delta_into(const Vec& delta, const Vec& v, Vec& out) const;
  Vec jacobian_product_at(const Vec& point, const Vec& v) const;
  void jacobian_product_at_into(const Vec& point, const Vec& v, Vec& out) const;
  Vec apply_model(const SymMatrix& hessian,
                  const Convexification& convexification,
                  const Vec& v) const;
  bool apply_right_inverse_impl(const Vec& rhs, Vec& correction,
                                std::string& message,
                                bool use_intermediate_accuracy) const;
  bool solve_condensed_newton(const Vec& step, double tau,
                              const Vec& targets, Vec& z,
                              CondensedRetractionResult& diagnostics,
                              std::string& message,
                              bool use_intermediate_accuracy) const;
  bool solve_scalar_quadratic(const Vec& step, double tau,
                              double target, double previous_z,
                              double& z, std::string& message) const;
  bool gmres(const Vec& current_point, const Vec& rhs,
             Vec& solution, Vec* mapped_solution, int& iterations,
             std::string& message,
             bool use_intermediate_accuracy) const;
  CondensedRetractionResult retract_impl(
      const Vec& step, const Vec& final_targets,
      bool use_intermediate_accuracy) const;
  bool smallest_projected_eigenpair(const SymMatrix& hessian,
                                    const Convexification& convexification,
                                    double& lambda, Vec& vector,
                                    std::string& message);
  bool projected_ritz_batch(
      const SymMatrix& hessian, const Convexification& convexification,
      double target_margin, int max_vectors, double& lambda,
      std::vector<std::pair<double, Vec>>& candidates,
      std::string& message);
};

} // namespace lazyipdca::qcqp
