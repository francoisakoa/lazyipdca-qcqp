#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace lazyipdca::qcqp::study {
// Match QCQPProblem::validate and the actual V1/V2 reader/workspace contract:
// only finite entries represent active bound sides. The historical format can
// also carry lb=ub=+infinity or lb=ub=-infinity for a free coordinate. Rejecting
// those accepted sentinels here changes the model contract before any solve.
// Keep the original ordered-bound check, reject NaNs/nonfinite starts, and
// never replace an absent side by a finite numerical surrogate box. Every
// finite value remains a genuine side, however large.
inline void validate_h4_domain(const QCQPProblem& p) {
  if(p.n<=0||p.x0.size()!=std::size_t(p.n)||p.lb.size()!=std::size_t(p.n)||p.ub.size()!=std::size_t(p.n))
    throw std::invalid_argument("H4 domain: inconsistent dimension");
  for(int j=0;j<p.n;++j) {
    const double lower=p.lb[j],upper=p.ub[j];
    if(!std::isfinite(p.x0[j]))
      throw std::invalid_argument("H4 domain: initial coordinate must be finite at index "+std::to_string(j));
    if(std::isnan(lower)||std::isnan(upper)||lower>upper)
      throw std::invalid_argument("H4 domain: invalid bound side at index "+std::to_string(j)+
          " (lb="+std::to_string(lower)+", ub="+std::to_string(upper)+")");
  }
}
}
