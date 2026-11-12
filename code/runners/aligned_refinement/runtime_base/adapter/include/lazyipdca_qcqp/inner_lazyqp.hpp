#pragma once
#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"

#include <memory>
#include <string>

namespace lazyipdca::qcqp {

// Exact affine substitution used by the optional LazyQP flow-lift route.
// The full ConvexSubproblem remains the public/outer model. This object owns
// a private reduced copy together with y_full = offset + T*y_reduced.
struct ExactAffineSubproblemReduction {
  ConvexSubproblem reduced;
  int full_variable_dimension = 0;
  int full_equality_dimension = 0;
  Vec offset;
  std::vector<SparseRow> transform_rows;
  std::vector<int> retained_variables;
  std::vector<int> retained_equalities;
  std::vector<int> full_to_reduced_variable;
  std::vector<int> full_to_reduced_equality;
  std::vector<ExactAffineElimination> eliminations;

  Vec expand_primal(const Vec& reduced_primal) const;
};

// Build and independently validate the affine composition of every quadratic
// model in p. On any mismatch this fails closed so p can be solved unchanged.
bool build_exact_affine_subproblem_reduction(
    const ConvexSubproblem& p, ExactAffineSubproblemReduction& reduction,
    std::string& message, bool sparse_checks = false);

// Numerical sufficient certificate of positive curvature on ker(A). All
// affine eliminations are validated before checking the unregularized saddle
// matrix inertia; singular or unsupported models are rejected.
bool certify_affine_tangent_inertia(const ConvexSubproblem& model,
    SymmetricFactorization& factor, std::string& message);

// Solver for an already-convex QCQP majorant.  No DC convexification occurs
// here.  The slack remains an independent primal variable.  At a trial point,
// the exact quadratic remainder is absorbed into the slack only while strict
// positivity is preserved; the unabsorbed part remains in the nonlinear
// primal residual and is corrected by later Newton steps.  One common
// primal-dual step length preserves the coupling of the KKT direction.
class ConvexQCQPLazyQPKernel {
 public:
  explicit ConvexQCQPLazyQPKernel(const Settings& settings) : st_(settings) {}
  InnerResult solve(const ConvexSubproblem& p) const;
  void clear_factorization_cache() const;
 private:
  const Settings& st_;
  SymmetricFactorization* acquire_factorization(bool reduced,bool hybrid,
                                                  std::string& message) const;
  struct KktTripletLayoutCache {
    int dimension = 0;
    bool valid = false;
    std::vector<SymmetricTriplet> entries;

    void clear() {
      dimension = 0;
      valid = false;
      entries.clear();
    }
  };
  // Full and inequality-reduced KKT systems have different dimensions and
  // sparsity graphs.  Keep independent symbolic states so a guarded reduced
  // fallback cannot evict the full-system METIS/LDL analysis (or vice versa).
  mutable std::unique_ptr<SymmetricFactorization> full_factor_;
  mutable std::unique_ptr<SymmetricFactorization> reduced_factor_;
  mutable std::unique_ptr<SymmetricFactorization> hybrid_factor_;
  // Full and inequality-reduced KKT systems keep independent, persistent
  // caller layouts beside their independent factor objects.  This covers
  // both condensed and bordered low-rank representations, including the
  // private affine-reduced subproblem: once a graph is established, later
  // Newton iterations and majorants update values in-place and hand the same
  // vector storage to the backend's positional numerical scatter.
  mutable KktTripletLayoutCache full_triplets_;
  mutable KktTripletLayoutCache reduced_triplets_;
  mutable KktTripletLayoutCache hybrid_triplets_;
  mutable std::string factor_backend_;
  mutable bool factor_metis_ = false;
  mutable unsigned factor_seed_ = 0;
};

} // namespace lazyipdca::qcqp
