#include "lazyipdca_qcqp/curvature.hpp"

#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

int main() {
  Settings st;
  st.verbose = false;
  st.exact_eigen_threshold = 64;
  st.max_rank_per_quadratic = 8;
  st.curvature_tolerance = 1e-10;
  st.curvature_margin = 1e-8;
  CurvatureOracle oracle(st);

  // Contract 1: an affine constraint (zero Hessian) is already PSD and must
  // not acquire an artificial scalar or low-rank correction.
  SymMatrix zero;
  zero.n = 3;
  Convexification affine_cvx;
  oracle.ensure_convex(zero, affine_cvx, 0.0);
  if (affine_cvx.rank() != 0 || affine_cvx.scalar_shift != 0.0) {
    std::cerr << "affine inequality was unnecessarily convexified\n";
    return 1;
  }

  // Contract 2: an indefinite inequality is corrected only to PSD.
  SymMatrix qi;
  qi.n = 2;
  qi.entries = {{0,0,-2.0}, {1,1,1.0}};
  Convexification ineq_cvx;
  oracle.ensure_convex(qi, ineq_cvx, 0.0);
  auto ei = oracle.smallest_eigenpair(qi, ineq_cvx);
  if (ei.lambda_min < -1e-8) {
    std::cerr << "inequality convexification is not PSD: lambda=" << ei.lambda_min << "\n";
    return 2;
  }
  if (ineq_cvx.rank() < 1) {
    std::cerr << "indefinite inequality retained no corrective direction\n";
    return 3;
  }

  // Contract 3: the objective receives the strictly positive model margin.
  Convexification obj_cvx;
  oracle.ensure_convex(qi, obj_cvx, st.curvature_margin);
  auto eo = oracle.smallest_eigenpair(qi, obj_cvx);
  if (eo.lambda_min < st.curvature_margin - 1e-7) {
    std::cerr << "objective model misses the curvature margin: lambda=" << eo.lambda_min << "\n";
    return 4;
  }

  // Contract 4: an explicit Lagrangian rank budget is independent of the
  // per-quadratic default.  With only two retained directions, the remaining
  // negative curvature must be handled by the scalar safeguard.
  SymMatrix q4; q4.n=4;
  q4.entries={{0,0,-4.0},{1,1,-3.0},{2,2,-2.0},{3,3,-1.0}};
  Convexification cap2;
  const int calls_before_bulk = oracle.calls();
  oracle.ensure_convex(q4,cap2,0.0,2);
  const int bulk_call_delta = oracle.calls() - calls_before_bulk;
  if (bulk_call_delta != 1) {
    std::cerr << "exact bulk curvature contract failed: oracle calls="
              << bulk_call_delta << " (expected 1)\n";
    return 5;
  }
  if(cap2.rank()!=2 || !(cap2.scalar_shift>0.0)){
    std::cerr << "explicit rank cap contract failed: rank="<<cap2.rank()
              << " shift="<<cap2.scalar_shift<<"\n";
    return 6;
  }

  std::cout << "[PASS] curvature contracts: affine untouched, inequality PSD, objective PD, split rank cap, one exact eigendecomposition\n";
  return 0;
}
