#pragma once

#include <cmath>
#include <limits>

namespace lazyipdca::qcqp {

// A semantic inequality row carries its primal residual, not its independent
// slack value, across DCA majorants.  If
//
//     r_pri = phi_old(y_old) + s_old,
//
// then the same residual at the origin of the new majorant requires
//
//     s_new = r_pri - phi_new(0).
//
// Preserve that value exactly whenever it is strictly positive.  A positive
// interior push is permitted only when the exact transported value is
// non-positive, since in that case no strictly interior slack can preserve the
// residual.
struct SingleLoopSlackRemap {
  bool valid = false;
  bool residual_preserved = false;
  double slack = std::numeric_limits<double>::quiet_NaN();
};

inline SingleLoopSlackRemap remap_single_loop_slack_residual(
    double old_primal_slack_residual, double new_phi_at_origin,
    double positive_push) {
  SingleLoopSlackRemap result;
  if (!std::isfinite(old_primal_slack_residual) ||
      !std::isfinite(new_phi_at_origin) || !std::isfinite(positive_push) ||
      !(positive_push > 0.0))
    return result;

  const double exact_slack =
      old_primal_slack_residual - new_phi_at_origin;
  if (!std::isfinite(exact_slack)) return result;

  result.valid = true;
  if (exact_slack > 0.0) {
    result.residual_preserved = true;
    result.slack = exact_slack;
  } else {
    result.slack = positive_push;
  }
  return result;
}

}  // namespace lazyipdca::qcqp
