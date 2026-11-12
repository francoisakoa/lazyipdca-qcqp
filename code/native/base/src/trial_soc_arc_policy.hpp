#pragma once
#include "nonlinear_trial_soc_policy.hpp"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {
// Observe, never alter, the existing accepted-point barrier gate. An event
// applies only to that SAME accepted-step generation and exact new mu.
// Rejected directions may retry it; one accepted step consumes it naturally.
// Recoveries that replace/reset an anchor must explicitly reset this record.
struct TrialSocArcBarrierEvent {
  bool armed=false;
  int generation=-1;
  double previous_mu=0,target_mu=0;
  void reset() {armed=false;generation=-1;previous_mu=target_mu=0;}
  void observe(int accepted_generation,double before,double after) {
    reset();
    if(accepted_generation<0||!(before>0&&after>0&&after<before)||
        !std::isfinite(before)||!std::isfinite(after))return;
    armed=true;generation=accepted_generation;previous_mu=before;target_mu=after;
  }
  bool current(int accepted_generation,double mu)const {
    return armed&&accepted_generation==generation&&std::isfinite(mu)&&mu==target_mu;
  }
};

// Connected interval, starting at t=0, on which c+b*t+a*t*t remains
// nonnegative. Coefficients are normalized before forming the discriminant.
// An upward tangency does not leave that interval. The final realized
// fraction-to-boundary check remains mandatory at the proposed double point.
inline long double trial_soc_arc_boundary_limit(long double c,long double b,long double a) {
  const long double unbounded=std::numeric_limits<long double>::infinity();
  if(!(c>0)||!std::isfinite(c)||!std::isfinite(b)||!std::isfinite(a))return 0;
  const long double normalization=std::max({std::abs(c),std::abs(b),std::abs(a)});
  c/=normalization;b/=normalization;a/=normalization;
  if(a==0)return b<0?-c/b:unbounded;
  if(a>0&&b>=0)return unbounded;
  const long double discriminant=std::fma(b,b,-4*a*c);
  if(a>0&&discriminant<=0)return unbounded;
  if(!(discriminant>=0)||!std::isfinite(discriminant))return 0;
  const long double square_root=std::sqrt(discriminant);
  // Avoid cancellation in both the small-positive-root and the very-large
  // positive-root cases. c>0 implies a<0 has exactly one positive root.
  const long double root=a<0&&b>=0?(-b-square_root)/(2*a):
      (2*c)/(-b+square_root);
  return root>0?root:0;
}

struct TrialSocArcInitialScale {
  double initial_t=0;
  long double size_cap=0,boundary_cap=0;
  bool valid=false;
};

// The EXISTING fraction-to-boundary parameter supplies both the permitted
// slack/dual displacement and the interior safety factor. No fitted radius
// or new application-dependent threshold enters this geometric proposal.
template<class AnchorType,class DirectionType>
TrialSocArcInitialScale trial_soc_arc_initial_scale(const AnchorType& anchor,
    const DirectionType& source,const DirectionType& increment,double source_alpha,double tau) {
  TrialSocArcInitialScale out;
  if(!(source_alpha>0&&source_alpha<=1&&tau>0&&tau<1)||
      !std::isfinite(source_alpha)||!std::isfinite(tau)||
      source.x.size()!=increment.x.size()||anchor.s.size()!=source.s.size()||
      anchor.s.size()!=increment.s.size()||anchor.z.size()!=source.z.size()||
      anchor.z.size()!=increment.z.size()||anchor.active.size()!=anchor.s.size()||
      anchor.active.size()!=anchor.z.size())return out;
  long double source_norm=0,increment_norm=0;
  for(std::size_t j=0;j<source.x.size();++j) {
    if(!std::isfinite(source.x[j])||!std::isfinite(increment.x[j]))return out;
    source_norm=std::max(source_norm,std::abs(static_cast<long double>(source.x[j])));
    increment_norm=std::max(increment_norm,std::abs(static_cast<long double>(increment.x[j])));
  }
  const long double unbounded=std::numeric_limits<long double>::infinity();
  // Same 1e-16 zero-direction floor as the existing realized SOC size guard.
  out.size_cap=increment_norm>0?static_cast<long double>(source_alpha)*
      std::max(1e-16L,source_norm)/increment_norm:unbounded;
  out.boundary_cap=unbounded;
  for(std::size_t i=0;i<anchor.active.size();++i)if(anchor.active[i]) {
    out.boundary_cap=std::min(out.boundary_cap,trial_soc_arc_boundary_limit(
        static_cast<long double>(tau)*anchor.s[i],static_cast<long double>(source_alpha)*source.s[i],increment.s[i]));
    out.boundary_cap=std::min(out.boundary_cap,trial_soc_arc_boundary_limit(
        static_cast<long double>(tau)*anchor.z[i],static_cast<long double>(source_alpha)*source.z[i],increment.z[i]));
  }
  const long double cap=std::min({1.0L,out.size_cap,out.boundary_cap});
  const long double proposed=static_cast<long double>(tau)*cap;
  if(!(proposed>0&&proposed<1)||!std::isfinite(proposed))return out;
  out.initial_t=static_cast<double>(proposed);
  // Round toward the interior if the long-double proposal rounded upward.
  if(static_cast<long double>(out.initial_t)>proposed)
    out.initial_t=std::nextafter(out.initial_t,0.0);
  out.valid=out.initial_t>0&&out.initial_t<1&&std::isfinite(out.initial_t);
  return out;
}

// A private arc may reuse only the FIRST accepted-trial SOC solve. Later
// increments depend on already corrected provisional points and cannot be
// treated as a single second-order term. The first RHS can itself contain
// linear-solve/rounding residuals: this is a proposal, not an assertion of
// exact cancellation. Every realized point still needs the original filter.
inline bool trial_soc_arc_admitted(bool enabled,bool barrier_transition,bool first_available,
    bool corrected_selected,const char* reason,const NonlinearTrialSocEvidence& evidence) {
  return enabled&&barrier_transition&&first_available&&!corrected_selected&&evidence.eligible()&&
      (std::strcmp(reason,"total-correction-size")==0||
       std::strcmp(reason,"realized-total-correction-size")==0||
       std::strcmp(reason,"fraction-to-boundary")==0);
}

// Evaluating this direction at a=t*source_alpha realizes the SAME arc in
// every primal/dual block: anchor + t*source_alpha*p + t*t*delta. The filter
// slope must be that of p at the anchor, not this t-dependent secant.
template<class DirectionType>
DirectionType trial_soc_arc_direction(const DirectionType& source,
    const DirectionType& first_increment,double source_alpha,double t) {
  if(!(source_alpha>0&&source_alpha<=1&&t>0&&t<1)||
      !std::isfinite(source_alpha)||!std::isfinite(t))
    throw std::invalid_argument("invalid trial SOC arc scale");
  DirectionType combined=source;
  const double weight=t/source_alpha;
  auto add=[&](auto& target,const auto& increment) {
    if(target.size()!=increment.size())
      throw std::invalid_argument("inconsistent trial SOC arc dimension");
    for(std::size_t j=0;j<target.size();++j)target[j]+=weight*increment[j];
  };
  add(combined.x,first_increment.x);add(combined.s,first_increment.s);
  add(combined.z,first_increment.z);add(combined.lambda,first_increment.lambda);
  return combined;
}

// The filter checks x/s/z, but equality multipliers do not enter its point
// reduction. Check the REALIZED update in every block before any commit;
// finite directions alone do not exclude overflow in anchor+a*direction.
template<class AnchorType,class DirectionType>
bool trial_soc_arc_finite_update(const AnchorType& anchor,
    const DirectionType& direction,double alpha) {
  if(!(alpha>0&&alpha<=1)||!std::isfinite(alpha))return false;
  auto block=[&](const auto& origin,const auto& step) {
    if(origin.size()!=step.size())return false;
    for(std::size_t j=0;j<origin.size();++j)
      if(!std::isfinite(origin[j])||!std::isfinite(step[j])||
          !std::isfinite(origin[j]+alpha*step[j]))return false;
    return true;
  };
  return block(anchor.x,direction.x)&&block(anchor.s,direction.s)&&
      block(anchor.z,direction.z)&&block(anchor.lambda,direction.lambda);
}
} // namespace lazyipdca::qcqp::detail
