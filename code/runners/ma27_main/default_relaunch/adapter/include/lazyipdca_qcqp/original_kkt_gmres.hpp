#pragma once

#include "lazyipdca_qcqp/model.hpp"

#include <functional>
#include <string>

namespace lazyipdca::qcqp {

using OriginalKktOperator = std::function<void(const Vec&, Vec&)>;
using OriginalKktPreconditioner =
    std::function<bool(const Vec&, Vec&, std::string&)>;

struct OriginalKktGmresResult {
  bool converged = false;
  int iterations = 0;
  double residual_2 = 0.0;
  std::string message;
};

// Flexible right-preconditioned GMRES for A d = rhs.  A is the exact,
// unregularized KKT operator.  The preconditioner may use a numerically
// regularized factorization and may vary with the right-hand side.
OriginalKktGmresResult original_kkt_right_preconditioned_gmres(
    const OriginalKktOperator& apply_A,
    const OriginalKktPreconditioner& apply_Minv,
    const Vec& rhs,
    Vec& correction,
    int max_iterations,
    double absolute_tolerance);

}  // namespace lazyipdca::qcqp
