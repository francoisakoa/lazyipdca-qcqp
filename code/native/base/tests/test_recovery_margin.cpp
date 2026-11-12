#include "lazyipdca_qcqp/curvature.hpp"
#include <cmath>
#include <iostream>
using namespace lazyipdca::qcqp;
int main() {
  Settings st;
  st.max_rank_per_quadratic = 2;
  st.exact_eigen_threshold = 64;
  st.curvature_margin = 1e-8;
  st.curvature_tolerance = 1e-9;
  SymMatrix zero; zero.n = 4;
  Convexification cvx;
  CurvatureOracle oracle(st);
  oracle.ensure_convex(zero, cvx, st.curvature_margin);
  if (cvx.rank() != 2) {
    std::cerr << "expected preserved low-rank budget behavior, rank=" << cvx.rank() << "\n";
    return 1;
  }
  if (!(cvx.scalar_shift >= st.curvature_margin - st.curvature_tolerance)) {
    std::cerr << "missing scalar fallback, shift=" << cvx.scalar_shift << "\n";
    return 2;
  }
  auto eig = oracle.smallest_eigenpair(zero, cvx);
  if (eig.lambda_min < st.curvature_margin - 2e-9) {
    std::cerr << "objective margin not restored, lambda=" << eig.lambda_min << "\n";
    return 3;
  }
  std::cout << "[PASS] proximal stabilization remains separate from lazy curvature\n";
  return 0;
}
