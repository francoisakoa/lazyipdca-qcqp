#pragma once
#include "lazyipdca_qcqp/lazy_newton.hpp"

namespace lazyipdca::qcqp {
// A fixed DC majorant is advanced privately. Only an original-filter accepted
// endpoint changes the public outer point, multipliers and linearization.
Result solve_filter_dca(const QCQPProblem &, const Settings &);
} // namespace lazyipdca::qcqp
