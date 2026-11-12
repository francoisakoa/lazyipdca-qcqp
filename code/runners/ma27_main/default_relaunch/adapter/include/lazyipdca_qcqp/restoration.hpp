#pragma once

#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <string>
#include <cstddef>

namespace lazyipdca::qcqp {

// Returns false for compiler modes that explicitly advertise globally unsafe
// floating-point semantics.  A true result means only that sparse Jv is a
// candidate: every PCG still performs an exact runtime probe, and one mismatch
// latches the complete Phase I onto the historical dense reduction.
bool restoration_exact_sparse_row_dots_build_compatible();

struct SparseRestorationBudget {
  bool allowed=false;
  std::size_t jacobian_entries=0,normal_triplets=0;
  long double estimated_auxiliary_bytes=0;
  int maximum_factor_workspace_mb=128;
};
// Bounds sparse construction and explicitly caps MUMPS numerical workspace.
// This is not a process-RSS or symbolic-fill estimate.
SparseRestorationBudget sparse_restoration_budget(const QCQPProblem& p,const Settings& settings);

struct RestorationResult {
  bool success = false;
  std::string message;
  Vec x;
  int iterations = 0;
  int restarts = 0;
  int linear_solves = 0;
  int exact_hessian_solves = 0;
  int matrix_free_solves = 0;
  int row_space_solves = 0;
  int harmonic_attempts = 0;
  int harmonic_acceptances = 0;
  int harmonic_factorizations = 0;
  int tangent_activation_attempts = 0;
  int tangent_activation_acceptances = 0;
  int tangent_activation_components = 0;
  int cg_iterations = 0;
  int exact_workspace_growths = 0;
  int exact_matvecs = 0;
  int exact_parallel_row_dot_batches = 0;
  int exact_parallel_teams = 0;
  int exact_sparse_row_dot_batches = 0;
  int exact_sparse_row_dot_verifications = 0;
  int exact_sparse_row_dot_fallbacks = 0;
  int line_search_backtracks = 0;
  int stagnation_events = 0;
  int minimax_qp_solves = 0;
  int minimax_inner_iterations = 0;
  int minimax_factorizations = 0;
  int minimax_accepted_steps = 0;
  int minimax_line_search_backtracks = 0;
  int structured_abs_links = 0;
  int effective_dimension = 0;
  double initial_feasibility = 0.0;
  double harmonic_feasibility = 0.0;
  double tangent_activation_feasibility = 0.0;
  double tangent_activation_step_inf = 0.0;
  // Wall-clock profiling of the large equality-chain Phase I. These are
  // diagnostics only and do not affect the restoration trajectory.
  double harmonic_seconds = 0.0;
  double tangent_activation_seconds = 0.0;
  double legacy_gate_seconds = 0.0;
  double exact_restoration_seconds = 0.0;
  double minimax_seconds = 0.0;
  double restart_seconds = 0.0;
  double total_seconds = 0.0;
  int tangent_activation_gradient_rows = 0;
  int tangent_activation_line_quadratics = 0;
  int tangent_activation_candidates = 0;
  int harmonic_unknowns = 0;
  int harmonic_graph_edges = 0;
  double final_feasibility = 0.0;
  double final_merit = 0.0;
  double final_damping = 0.0;
};

// Separate work for an explicitly declared mixed-startup feasibility phase.
// Layouts of Settings, Result and RestorationResult remain unchanged.
struct NativeRestorationWork {
  int factorization_calls=0,backend_regularization_attempts=0,linear_solve_calls=0,outer_attempts=0,accepted_steps=0;
  double factorization_seconds=0,linear_solve_seconds=0;
};
RestorationResult restore_mixed_startup(const QCQPProblem& problem,const Vec& start,
    const Settings& settings,NativeRestorationWork* work=nullptr);

class FeasibilityRestoration {
 public:
  explicit FeasibilityRestoration(const Settings& settings) : st_(settings) {}
  RestorationResult solve(const QCQPProblem& p, const Vec& start) const;
  // Bounded affine elastic epigraph probe, followed by ordinary original
  // feasibility restoration with the SAME remaining iteration/time allowance.
  RestorationResult solve_elastic(const QCQPProblem& p,const Vec& start) const;
  // Late Phase I only: one sparse row-space run; no dense fallback/restarts.
  RestorationResult solve_sparse_only(const QCQPProblem& p,const Vec& start) const;
  RestorationResult polish(const QCQPProblem& p, const Vec& start, int max_iterations) const;

 private:
  const Settings& st_;
};

// Local equality retraction used after an SQP trial step.  It is deliberately
// local: unlike Phase I it performs no restarts and does not optimize the
// objective.  It only removes the O(||d||^2) equality defect while respecting
// variable bounds.
Vec retract_equalities(const QCQPProblem& p, const Vec& start,
                       int max_iterations, double tolerance,
                       double regularization, double fraction_to_boundary);

// Target-aware local equality correction.  The target values are the residuals
// predicted by the scaled linearized equality model.  For a trial z=x+alpha*d,
// a classical SQP defect correction uses
//
//   target_j = h_j(x) + grad h_j(x)^T (z-x),
//
// and removes only h_j(z)-target_j.  Hence the correction is O(||z-x||^2)
// and tends to zero with the line-search step.  Structured t=|q| lift links
// are eliminated and corrected in the reduced original-variable space.
Vec retract_equalities_to_targets(const QCQPProblem& p, const Vec& start,
                                  const Vec& targets,
                                  int max_iterations, double tolerance,
                                  double regularization,
                                  double fraction_to_boundary);

// Full lifted-space target correction used by Lazy-DCA SOC.  Unlike the
// Phase-I reduced manifold projection, this routine keeps every equality
// (including t^2-q^2 links) in the Newton system and does not project
// auxiliary variables before forming the residual.  Consequently the
// correction is exactly zero when start already matches targets and tends to
// zero with the scaled line-search defect, even for badly scaled lifted
// models.
Vec retract_equalities_to_targets_full_space(
    const QCQPProblem& p, const Vec& start, const Vec& targets,
    int max_iterations, double tolerance, double regularization,
    double fraction_to_boundary);

} // namespace lazyipdca::qcqp
