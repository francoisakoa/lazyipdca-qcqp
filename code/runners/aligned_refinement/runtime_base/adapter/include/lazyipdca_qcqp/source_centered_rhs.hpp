#pragma once
#include <cmath>

namespace lazyipdca::qcqp::detail {
// Source primal-barrier equality rows may have free, very large multipliers.
// Preserve the small product error before division by a very small slack.
// The non-source branch is deliberately the historical binary64 expression.
inline double source_centered_complementarity_rhs(bool source_primal_barrier,
    double slack,double multiplier,double mu) {
  return source_primal_barrier?std::fma(slack,multiplier,-mu):slack*multiplier-mu;
}
} // namespace lazyipdca::qcqp::detail
