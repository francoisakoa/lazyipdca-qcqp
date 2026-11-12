#pragma once
#include "lazyipdca_qcqp/curvature.hpp"
#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include "lazyipdca_qcqp/settings.hpp"

namespace lazyipdca::qcqp {

class Solver {
 public:
  explicit Solver(Settings settings = {});
  Result solve(const QCQPProblem& problem);

 private:
  Result solve_once(const QCQPProblem& problem);
  Settings st_;
};

} // namespace lazyipdca::qcqp
