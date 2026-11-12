#pragma once

#include "lazyipdca_qcqp/model.hpp"
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp::detail {

// Frozen V1/V2 exports historically used either infinity sign for an absent
// bound side. The active-side contract is finiteness. Put absent sides into
// canonical order before validation and any presolve comparison or merging;
// never change a finite bound, a start coordinate, or the input file itself.
// NaNs are deliberately retained so the ordinary model validator rejects them.
inline void canonicalize_qcq_absent_bound_sentinels(QCQPProblem& problem) {
  const double infinity = std::numeric_limits<double>::infinity();
  for (double& lower : problem.lb)
    if (std::isinf(lower)) lower = -infinity;
  for (double& upper : problem.ub)
    if (std::isinf(upper)) upper = infinity;
}

} // namespace lazyipdca::qcqp::detail
