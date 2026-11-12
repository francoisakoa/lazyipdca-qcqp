#pragma once
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>
#include "lazyipdca_qcqp/model.hpp"
namespace lazyipdca::qcqp::detail {
// Independently verify every original function before applying positive parts:
// std::max(0,NaN) must never turn an invalid row into apparent feasibility.
inline std::pair<long double,long double> native_affine_original_residuals(
    const QCQPProblem& problem,const Vec& x) {
  const auto invalid=std::make_pair(std::numeric_limits<long double>::infinity(),
      std::numeric_limits<long double>::infinity());
  if(x.size()!=std::size_t(problem.n))return invalid;
  for(double v:x)if(!std::isfinite(v))return invalid;
  if(!std::isfinite(problem.objective.value(x)))return invalid;
  std::pair<long double,long double> out{0,0};
  auto add=[&](long double v){out.first=std::max(out.first,v);out.second+=v;};
  for(const auto& q:problem.equalities){double v=q.value(x);if(!std::isfinite(v))return invalid;add(std::abs(v));}
  for(const auto& q:problem.inequalities){double v=q.value(x);if(!std::isfinite(v))return invalid;add(std::max(0.,v));}
  for(int j=0;j<problem.n;++j){
    if(std::isfinite(problem.lb[j]))add(std::max(0.,problem.lb[j]-x[j]));
    if(std::isfinite(problem.ub[j]))add(std::max(0.,x[j]-problem.ub[j]));
  }
  return out;
}
// Permission for a private feasibility-phase anchor, never an original KKT
// certificate. All rejected/accepted native work stays in the global budget.
inline bool native_affine_partial_progress(bool phase_solved,bool bounds_ok,
    long double candidate_max,long double candidate_l1,
    long double reference_max,long double reference_l1,double elapsed,double limit) {
  return !phase_solved&&bounds_ok&&std::isfinite(candidate_max)&&std::isfinite(candidate_l1)&&
      std::isfinite(reference_max)&&std::isfinite(reference_l1)&&candidate_max>=0&&candidate_l1>=0&&
      reference_max>0&&reference_l1>0&&candidate_max<.9L*reference_max&&candidate_l1<.9L*reference_l1&&
      std::isfinite(elapsed)&&elapsed>=0&&std::isfinite(limit)&&limit>0&&elapsed<limit;
}
}
