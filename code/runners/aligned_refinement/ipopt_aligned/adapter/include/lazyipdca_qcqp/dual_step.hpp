#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace lazyipdca::qcqp::detail {

struct PrimalDualStep {
  double primal=0.0;
  double bound_dual=0.0;
  double equality_dual=0.0;
};

// Ipopt backtracks the primal variables/slacks only. Positive bound/slack
// multipliers keep their independent fraction-to-boundary step and equality
// multipliers use alpha_for_y=primal. The coupled policy is retained solely
// for reproducible comparison with releases through R5.
inline PrimalDualStep primal_dual_step(double primal_max,double dual_max,
                                       double backtrack,
                                       const std::string& policy) {
  if(policy=="ipopt")
    return {primal_max*backtrack,dual_max,primal_max*backtrack};
  if(policy=="coupled") {
    const double step=dual_max*backtrack;
    return {primal_max*backtrack,step,step};
  }
  throw std::invalid_argument("dual-step policy must be ipopt or coupled");
}

inline double common_feasible_step(double primal,double equality_dual,double bound_dual) {
  return std::min({primal,equality_dual,bound_dual});
}

inline bool compare_common_step(double original_primal,double old_dual,double split_dual,
                                double primal_trigger,double dual_growth) {
  return std::isfinite(original_primal)&&std::isfinite(old_dual)&&std::isfinite(split_dual)&&
         original_primal<=primal_trigger &&
         split_dual>=dual_growth*std::max(old_dual,1e-300);
}

} // namespace lazyipdca::qcqp::detail
