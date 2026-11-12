#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <cmath>

namespace lazyipdca::qcqp::detail {
// A square root-finding problem has neither an objective gradient nor a
// tangent optimization step. Keep the historical original-feasibility filter
// for its exact normal equations; source stabilization remains available to
// all optimization models. Never infer this from a small objective gradient.
inline bool constant_square_feasibility_filter(const QCQPProblem& p,const Settings& s) {
  if(s.standard_r10_policy!="auto"||p.n<=0||!p.inequalities.empty()||
      !p.objective.signed_square.empty()||p.has_signed_square())return false;
  bool constant=true;
  p.objective.for_each_linear_nonzero([&](int,double v){if(v!=0)constant=false;});
  for(const auto& t:p.objective.hessian.entries)if(t.v!=0)constant=false;
  if(!constant)return false;
  int free=0;
  for(int j=0;j<p.n;++j) {
    const bool lo=std::isfinite(p.lb[j]),up=std::isfinite(p.ub[j]);
    if(lo&&up&&p.lb[j]==p.ub[j])continue;
    if(lo||up)return false;
    ++free;
  }
  return free>0&&p.equalities.size()==static_cast<std::size_t>(free);
}
}
