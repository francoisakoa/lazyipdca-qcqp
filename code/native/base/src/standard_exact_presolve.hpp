#pragma once
#include "lazyipdca_qcqp/standard_auto.hpp"
#include "native_signed_square_reduction.hpp"

namespace lazyipdca::qcqp::detail {
// Cheap negative screen before constructing a reduced model. In particular,
// ordinary bilinear systems do not trigger a second scan of every linear row.
inline bool native_signed_square_candidate(const QCQPProblem& p) {
  for(const auto& q:p.equalities) {
    if(q.constant!=0||!q.signed_square.empty())continue;
    int i=-1,j=-1;double a=0,b=0;bool valid=true;
    for(const auto& h:q.hessian.entries)if(h.v!=0) {
      if(h.i!=h.j||j>=0){valid=false;break;}
      if(i<0){i=h.i;a=h.v;}else{j=h.i;b=h.v;}
    }
    if(!valid||i<0||j<0||i==j||a!=-b)continue;
    bool linear=false;q.for_each_linear_nonzero([&](int,double c){linear=linear||c!=0;});
    if(linear)continue;
    auto domain=[&](int source,int auxiliary) {
      return p.lb[auxiliary]==0&&std::isinf(p.ub[auxiliary])&&p.ub[auxiliary]>0&&
          std::isfinite(p.x0[source])&&p.x0[auxiliary]==std::abs(p.x0[source]);
    };
    if(domain(i,j)||domain(j,i))return true;
  }
  return false;
}
// Give an already supported exact presolve priority over a newly selected
// full-source treatment. This applies only to the reference curvature policy:
// the HH/HS/SH/SS studies deliberately retain a quadratic source model.
// The reduction validates every use, domain, and supplied start before it is
// selected. It never changes the supplied start or fixes an orthant.
inline StandardAutoSelection resolve_standard_with_exact_presolve(
    const QCQPProblem& problem,const Settings& requested) {
  auto selected=resolve_standard_auto(problem,requested);
  if(!selected.enabled||requested.convexification_study!="off"||
      requested.majorant_equality_stabilization||problem.has_signed_square()||
      !native_signed_square_candidate(problem))
    return selected;
  NativeSignedSquareReduction native(problem);
  if(!native.changed())return selected;
  selected.settings=requested;
  selected.settings.standard_auto_resolved=true;
  selected.settings.outer_algorithm="lazy-dca";
  selected.settings.dca_engine="majorants";
  selected.enabled=false;
  selected.mode="native-signed-square";
  selected.reason="certified-exact-signed-square-presolve-priority";
  return selected;
}
} // namespace lazyipdca::qcqp::detail
