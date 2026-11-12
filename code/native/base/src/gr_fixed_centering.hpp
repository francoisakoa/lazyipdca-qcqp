#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {

inline bool gr_fixed_centering_converged(double original_barrier,double regularized_barrier,
    double mu,double primal,double dual,double ptol,double dtol) {
  const double threshold=10*mu;
  return std::isfinite(original_barrier)&&std::isfinite(regularized_barrier)&&
      std::isfinite(mu)&&mu>0&&std::isfinite(threshold)&&std::isfinite(primal)&&std::isfinite(dual)&&
      original_barrier<=threshold&&regularized_barrier<=threshold&&primal<=ptol&&dual<=dtol;
}

// A diagnostic inner phase, not a new continuation or acceptance rule.
// Trigger counts refer to distinct accepted states; failed outer retries
// cannot manufacture stability. At most two phases share the solve budget.
struct GrFixedCenteringPolicy {
  int limit=0,remaining=0,attempts=0,streak=0,last_generation=-1;
  int next_generation=0;
  explicit GrFixedCenteringPolicy(int steps):limit(steps) {
    if(steps<0||steps>8)throw std::invalid_argument("fixed centering steps must be in [0,8]");
  }
  bool active() const {return remaining>0;}
  bool observe(bool enabled,int generation,double primal,double dual,double comp,
      double centering,double mu,double ptol,double dtol,double ctol) {
    if(active()||limit==0||attempts>=2)return false;
    if(generation==last_generation)return false;
    last_generation=generation;
    if(!enabled){streak=0;return false;}
    if(generation<next_generation){streak=0;return false;}
    const bool finite=std::isfinite(primal)&&std::isfinite(dual)&&std::isfinite(comp)&&
        std::isfinite(centering)&&std::isfinite(mu)&&mu>0;
    // This diagnostic targets complementarity after the original primal
    // and dual blocks already pass. A looser neighborhood would spend its
    // finite phase budget on ordinary feasibility/optimality progress.
    const bool eligible=finite&&primal<=ptol&&dual<=dtol&&comp>ctol&&centering>10*mu;
    streak=eligible?streak+1:0;
    if(streak<3)return false;
    ++attempts;remaining=limit;streak=0;return true;
  }
  bool accepted(int generation,bool converged=false) {
    if(!active())return false;
    --remaining;
    if(converged||remaining==0){remaining=0;next_generation=generation+8;return true;}
    return false;
  }
  void abort(int generation) {remaining=0;streak=0;next_generation=generation+8;}
};

// Ensures interrupted/exceptional phase exits cannot export a staged phase.
struct GrFixedCenteringScope {
  std::function<void()> rollback;
  ~GrFixedCenteringScope() {if(rollback)rollback();}
};
} // namespace lazyipdca::qcqp::detail
