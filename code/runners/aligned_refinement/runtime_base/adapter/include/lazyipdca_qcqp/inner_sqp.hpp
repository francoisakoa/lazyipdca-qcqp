#pragma once
#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/settings.hpp"

namespace lazyipdca::qcqp {

class ConvexQCQPFilterSQP {
 public:
  explicit ConvexQCQPFilterSQP(const Settings& settings) : st_(settings) {}
  InnerResult solve(const ConvexSubproblem& p) const;
 private:
  const Settings& st_;
};

class ConvexQCQPTrustRegionSQP {
 public:
  explicit ConvexQCQPTrustRegionSQP(const Settings& settings) : st_(settings) {}
  InnerResult solve(const ConvexSubproblem& p) const;
 private:
  const Settings& st_;
};

} // namespace lazyipdca::qcqp
