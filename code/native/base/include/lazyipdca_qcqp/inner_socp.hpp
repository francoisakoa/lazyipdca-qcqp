#pragma once

#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <string>

namespace lazyipdca::qcqp {

// Structural description of the exact rotated-cone epigraph of a convex
// QCQP majorant.  The implementation deliberately keeps the Gram factors
// implicit: eliminating the cone coordinates gives the same sparse
// primal-dual equations as the native nonlinear-residual kernel, without
// materialising one dense R_i with H_i=R_i^T R_i per quadratic constraint.
struct CondensedSOCPFormulation {
  int variable_dimension = 0;
  int affine_equalities = 0;
  int objective_rotated_cones = 0;
  int inequality_rotated_cones = 0;
  int linear_nonnegative_cones = 0;
  long long explicit_lift_cone_dimension = 0;
  // The objective cone is interpreted on Ay=b.  If COND contributes signed
  // equality-Lagrangian curvature, a Finsler/PSD completion which vanishes on
  // that affine slice is implicit before the Gram factor is eliminated.
  bool equality_slice_psd_completion = false;
};

// Build and audit the exact conic representation.  Quadratic equalities are
// intentionally absent: LIN and COND continue to publish their affine SQP
// equation, and COND retains its independent exact quadratic retraction.
bool build_condensed_socp_formulation(const ConvexSubproblem& problem,
                                      CondensedSOCPFormulation& formulation,
                                      std::string& message);

// Determinant of the rotated cone obtained from
//   q(y)=c+l^T y+1/2 y^T H y <= 0:
//   (-(c+l^T y), 1, R y) in Q_r, H=R^T R.
// It is exactly -2*q(y).  For an objective epigraph q(y)<=tau it is
// 2*(tau-q(y)).  This helper is public so equivalence can be tested without
// constructing a dense Gram factor.
double condensed_socp_rotated_determinant(
    const ConvexQuadraticModel& model, const Vec& y,
    double epigraph_value, bool objective_epigraph);

// Exact Schur-condensed SOCP backend.  Cone primal/dual auxiliaries are
// analytically eliminated; the current affine equalities, warm starts,
// working-set state, KKT regularisation and fail-closed certificates are
// therefore preserved byte-for-byte by the proven LazyQP numerical kernel.
class ConvexQCQPSOCPKernel {
 public:
  explicit ConvexQCQPSOCPKernel(const Settings& settings)
      : st_(settings), condensed_kernel_(settings) {}

  InnerResult solve(const ConvexSubproblem& problem) const;
  void clear_factorization_cache() const {
    condensed_kernel_.clear_factorization_cache();
  }

 private:
  const Settings& st_;
  mutable ConvexQCQPLazyQPKernel condensed_kernel_;
};

} // namespace lazyipdca::qcqp
