#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace lazyipdca::qcqp::detail {
// Observe accepted historical states only. Small steps alone do not diagnose
// failure: require sustained lack of progress in BOTH original KKT and dual
// residuals. A zero proximal term is a valid unregularized trajectory and
// must not hide filter stagnation. Reject negative/nonfinite diagnostics;
// no model, accepted point or proximal coefficient is changed here.
struct HistoricalEqualityWatch {
  int streak=0,last_generation=-1;
  static double physical_step_threshold() {
    return std::sqrt(std::numeric_limits<double>::epsilon());
  }
  // Compare the ACTUAL rounded accepted coordinates before committing them.
  // A global norm would hide motion of a small coordinate beside a large one.
  // Extended intermediates avoid overflow in a finite double displacement.
  template<class Vector>
  static double relative_primal_step(const Vector& before,const Vector& after) {
    if(before.size()!=after.size()||before.empty())
      return std::numeric_limits<double>::quiet_NaN();
    long double largest=0;
    for(std::size_t j=0;j<before.size();++j) {
      if(!std::isfinite(before[j])||!std::isfinite(after[j]))
        return std::numeric_limits<double>::quiet_NaN();
      const long double a=before[j],b=after[j];
      const long double step=std::abs(b-a)/(1+std::max(std::abs(a),std::abs(b)));
      if(!std::isfinite(step))return std::numeric_limits<double>::quiet_NaN();
      largest=std::max(largest,step);
    }
    return static_cast<double>(largest);
  }
  bool observe(bool enabled,int generation,double alpha,double proximal,
      double old_kkt,double kkt,double old_dual,double dual,double tolerance,
      double physical_relative_step=std::numeric_limits<double>::infinity()) {
    if(generation==last_generation)return false;
    last_generation=generation;
    const bool finite=std::isfinite(alpha)&&std::isfinite(proximal)&&
        std::isfinite(old_kkt)&&std::isfinite(kkt)&&std::isfinite(old_dual)&&
        std::isfinite(dual)&&std::isfinite(tolerance);
    const bool tiny_physical_step=std::isfinite(physical_relative_step)&&
        physical_relative_step>=0&&physical_relative_step<=physical_step_threshold();
    const bool stalled=enabled&&finite&&alpha>0&&(alpha<=1e-3||tiny_physical_step)&&proximal>=0&&
        tolerance>0&&kkt>tolerance&&dual>tolerance&&old_kkt>=0&&old_dual>=0&&
        static_cast<long double>(kkt)>=.99L*old_kkt&&
        static_cast<long double>(dual)>=.99L*old_dual;
    streak=stalled?streak+1:0;
    return streak>=8;
  }
};
} // namespace lazyipdca::qcqp::detail
