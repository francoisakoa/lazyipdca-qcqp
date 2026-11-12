#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <memory>
#include <string>
#include <vector>

namespace lazyipdca::qcqp {

struct SymmetricTriplet {
  int i = 0;
  int j = 0;
  double value = 0.0;
};

class SymmetricFactorization {
 public:
  virtual ~SymmetricFactorization() = default;
  virtual bool factor(int n, int primal_dimension,
                      const std::vector<SymmetricTriplet>& entries,
                      std::string& message) = 0;
  virtual bool solve(Vec& rhs, std::string& message) const = 0;
  // Numerical quality recovery on the identical cached matrix. False means
  // unsupported, capped, or failed refactorization; never accept the old solve.
  virtual bool increase_quality(std::string&) { return false; }
  virtual double pivot_tolerance() const { return 0.0; }
  // Read-only curvature probes may reuse a COMPLETE nonsingular factor with
  // rejected inertia. They never authorize a Newton step, and keep the same
  // solve residual/refinement checks. A partial/singular factor is unavailable.
  virtual int curvature_defect() const { return 0; }
  virtual bool solve_curvature_probe(Vec&, std::string&) const { return false; }
  virtual int failed_pivot_coordinate() const { return -1; }
  // Diagnostics refer to the matrix actually factorized by this backend.
  virtual double last_backward_error() const { return 0.0; }
  // Backward error before iterative refinement on the most recent solve.
  // This distinguishes a solve that was already accurate from one repaired
  // by one or more refinement steps.
  virtual double last_initial_backward_error() const { return last_backward_error(); }
  virtual double last_residual_inf() const { return 0.0; }
  virtual int last_refinement_steps() const { return 0; }
  // Number of symbolic analyses performed by this factor object.  Dense
  // LAPACK has no separate symbolic phase and reports zero.
  virtual int symbolic_analyses() const { return 0; }
  // Number of numerical factorizations that reused an already valid symbolic
  // analysis.  For sparse backends this counts reuse both within one convex
  // subproblem and, when the factor object is persistent, across successive
  // Lazy-DCA majorants.
  virtual int symbolic_reuses() const { return 0; }
  // Number of times an existing symbolic cache had to be replaced because a
  // genuinely new structural nonzero (or a dimension change) appeared.
  virtual int symbolic_invalidations() const { return 0; }
  // MA27 backend-call timing; other G10 backends retain their original behavior
  // and return zero when this split is not measured.
  virtual double symbolic_seconds() const { return 0.0; }
  virtual double numeric_seconds() const { return 0.0; }
  virtual double assembly_seconds() const { return 0.0; }
  virtual double linear_solve_seconds() const { return 0.0; }
  // True when the most recent numerical factorization used an existing
  // symbolic state instead of running METIS + symbolic analysis again.
  virtual bool last_symbolic_reused() const { return false; }
  virtual bool has_symbolic_cache() const { return false; }
  // Sparse numerical assembly diagnostics.  A scatter build records the
  // one-time mapping from the caller's stable triplet layout to the cached CSC
  // envelope.  A reuse avoids sorting, duplicate-pattern construction, and
  // associative lookup for the numerical refactorization.  A layout
  // invalidation is deliberately distinct from a symbolic invalidation: a
  // reordered caller layout can require a new scatter while retaining the
  // same matrix graph and symbolic factorization.
  virtual int numeric_scatter_builds() const { return 0; }
  virtual int numeric_scatter_reuses() const { return 0; }
  virtual int numeric_scatter_invalidations() const { return 0; }
  virtual bool last_numeric_scatter_reused() const { return false; }
  // Optional numerical diagnostics used by the sign-aware cholesk adapter.
  // Backends that do not expose scalar LDL^T pivots retain the sentinel values.
  virtual int last_positive_pivots() const { return -1; }
  virtual int last_negative_pivots() const { return -1; }
  virtual int last_zero_pivots() const { return -1; }
  virtual double minimum_abs_pivot() const { return 0.0; }
  virtual double maximum_abs_pivot() const { return 0.0; }
  virtual int minimum_abs_pivot_coordinate() const { return -1; }
  // Diagonal regularization applied inside the backend on its last successful
  // factorization.  The primal value is an additive shift; the dual value may
  // be a negative-diagonal floor. Explicit shifts already present in `entries`
  // are excluded.
  virtual double last_primal_regularization() const { return 0.0; }
  virtual double last_dual_regularization() const { return 0.0; }
  virtual int last_regularization_attempts() const { return 0; }
  virtual const char* name() const = 0;
};

std::unique_ptr<SymmetricFactorization>
make_symmetric_factorization(const Settings& settings, std::string& message);

// Explicit bounded-workspace factory for sparse-only feasibility recovery.
// Keeps the requested MUMPS+METIS backend; unsupported controls are refused.
// The ordinary factory and all main KKT settings remain unchanged.
std::unique_ptr<SymmetricFactorization> make_memory_bounded_symmetric_factorization(
    const Settings& settings,std::string& message,int maximum_workspace_mb);

bool linear_backend_compiled(const std::string& backend);
std::string compiled_linear_backends();

} // namespace lazyipdca::qcqp
