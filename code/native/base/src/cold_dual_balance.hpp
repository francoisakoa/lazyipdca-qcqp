#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace lazyipdca::qcqp::detail {
struct AffineObjectiveColumn {int coordinate=-1;double coefficient=0;const char* reason="objective-not-one-affine-coordinate";};
inline bool curved_column(const QuadraticFunction& q,int j) {
  for(const auto& e:q.hessian.entries)if(e.v!=0&&(e.i==j||e.j==j))return true;
  for(const auto& e:q.signed_square)if(e.second!=0&&e.first==j)return true;
  return false;
}
inline double linear_column(const QuadraticFunction& q,int j) {
  double out=0;q.for_each_linear_nonzero([&](int k,double a){if(k==j)out+=a;});return out;
}
// Structural ORIGINAL coefficients, never a zero gradient observed at x0.
// The orientation makes the admission invariant to flipping the epigraph sign.
inline AffineObjectiveColumn affine_objective_column(const QCQPProblem& p) {
  AffineObjectiveColumn out;int count=0;
  for(const auto& e:p.objective.hessian.entries)if(e.v!=0)return out;
  for(const auto& e:p.objective.signed_square)if(e.second!=0)return out;
  p.objective.for_each_linear_nonzero([&](int j,double a){if(a!=0){++count;out.coordinate=j;out.coefficient=a;}});
  if(count!=1||!std::isfinite(out.coefficient)||out.coefficient==0){out.coordinate=-1;return out;}
  if(out.coordinate<0||out.coordinate>=p.n){out.coordinate=-1;return out;}
  const int j=out.coordinate;const double orientation=out.coefficient>0?1:-1;
  for(const auto& q:p.equalities)if(curved_column(q,j)||linear_column(q,j)!=0) {
    out.coordinate=-1;out.reason="objective-coordinate-in-original-equality";return out;
  }
  for(const auto& q:p.inequalities) {
    const double a=linear_column(q,j);
    if(curved_column(q,j)||!std::isfinite(a)||orientation*a>0) {
      out.coordinate=-1;out.reason="original-inequality-column-not-monotone-affine";return out;
    }
  }
  if((orientation>0&&std::isfinite(p.ub[j]))||(orientation<0&&std::isfinite(p.lb[j]))) {
    out.coordinate=-1;out.reason="opposing-original-bound";return out;
  }
  out.reason="original-affine-objective-column";return out;
}
struct ColdDualBalancePlan {bool allowed=false;double raw=0,scale=1;const char* reason="invalid-cold-barrier-state";};
inline ColdDualBalancePlan cold_dual_balance_plan(long double objective_column,long double constraint_column,
    int contributors,double actual_mean,double barrier_floor) {
  ColdDualBalancePlan out;
  if(contributors<2){out.reason="fewer-than-two-contributors";return out;}
  if(!std::isfinite(objective_column)||!std::isfinite(constraint_column)||objective_column==0||
      !std::isfinite(actual_mean)||!(actual_mean>0)||!std::isfinite(barrier_floor)||barrier_floor<0)return out;
  const long double raw=-objective_column/constraint_column;
  if(!std::isfinite(raw)||!(raw>0&&raw<1)){out.reason="dual-column-already-balanced-or-too-small";return out;}
  const long double clipped=std::max(raw,static_cast<long double>(barrier_floor)/actual_mean);
  out.raw=double(raw);out.scale=double(clipped);
  if(static_cast<long double>(out.scale)<clipped)out.scale=std::nextafter(out.scale,1.0);
  if(!std::isfinite(out.scale)||!(out.scale>0&&out.scale<1)){out.reason="barrier-floor-prevents-scaling";return out;}
  out.allowed=true;out.reason="coefficient-verified-column-balance";return out;
}
}
