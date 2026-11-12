#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace lazyipdca::qcqp::detail {

struct BoundedNormalColumn {
  std::vector<std::pair<int,double>> entries;
  double lower=-std::numeric_limits<double>::infinity();
  double upper= std::numeric_limits<double>::infinity();
};

// Execution work survives a rejected or nonrepresentable multiplier candidate.
struct BoundedNormalRestartWork {
  int sweeps=0;
  double seconds=0;
};

struct BoundedNormalRestart {
  Vec multipliers,residual;
  int sweeps=0;
  double seconds=0;
  bool valid=false;
};

// Reconstruct a small multiplier representative from zero while holding every
// primal coordinate and original normal fixed. Column bounds are supplied by
// the caller, including any original-complementarity caps. This helper does
// not accept a nonlinear step or claim a KKT certificate: the caller must
// independently check its original residuals and globalization policy.
inline BoundedNormalRestart bounded_normal_restart(
    const Vec& gradient,const std::vector<BoundedNormalColumn>& columns,
    double budget_seconds,int max_sweeps=80000,
    BoundedNormalRestartWork* execution_work=nullptr) {
  using L=long double;using Clock=std::chrono::steady_clock;
  const auto began=Clock::now();
  auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-began).count();};
  BoundedNormalRestart result;
  if(execution_work)*execution_work={};
  auto finish=[&](bool valid=false) {
    result.seconds=elapsed();result.valid=valid;
    if(execution_work){execution_work->sweeps=result.sweeps;execution_work->seconds=result.seconds;}
    if(!valid){result.multipliers.clear();result.residual.clear();}
    return result;
  };
  if(!(budget_seconds>0)||max_sweeps<1||gradient.empty()||
     !std::all_of(gradient.begin(),gradient.end(),[](double v){return std::isfinite(v);}))return finish();
  struct Work {
    std::vector<std::pair<int,double>> entries;
    double scale=1;
    L u=0,lower=0,upper=0,norm=0;
  };
  std::vector<Work> work;work.reserve(columns.size());
  std::size_t nonzeros=0;
  for(const auto& input:columns) {
    if(std::isnan(input.lower)||std::isnan(input.upper)||input.lower>input.upper||
       input.lower==std::numeric_limits<double>::infinity()||
       input.upper==-std::numeric_limits<double>::infinity())return finish();
    Work c;double scale=0;
    for(auto [j,a]:input.entries) {
      if(j<0||std::size_t(j)>=gradient.size()||!std::isfinite(a))return finish();
      scale=std::max(scale,std::abs(a));
    }
    c.scale=scale>0?scale:1;
    c.lower=L(input.lower)*c.scale;c.upper=L(input.upper)*c.scale;
    c.u=std::min(c.upper,std::max(c.lower,L(0)));
    for(auto [j,a]:input.entries)if(a!=0) {
      const double value=a/c.scale;c.entries.push_back({j,value});
      c.norm+=L(value)*value;
    }
    nonzeros+=c.entries.size();
    if(nonzeros>128u*1024u*1024u/sizeof(std::pair<int,double>))return finish();
    work.push_back(std::move(c));
    if(elapsed()>=budget_seconds)return finish();
  }
  std::vector<L> residual;
  auto reconstruct=[&]() {
    residual.assign(gradient.begin(),gradient.end());
    for(const auto& c:work)for(auto [j,a]:c.entries)residual[j]+=c.u*a;
  };
  reconstruct();
  for(;result.sweeps<max_sweeps&&elapsed()<budget_seconds;++result.sweeps) {
    bool changed=false;
    for(auto& c:work) {
      if(!(c.norm>0))continue;
      L product=0;for(auto [j,a]:c.entries)product+=L(a)*residual[j];
      const L next=std::min(c.upper,std::max(c.lower,c.u-product/c.norm));
      const L delta=next-c.u;c.u=next;changed|=delta!=0;
      for(auto [j,a]:c.entries)residual[j]+=delta*a;
    }
    if(result.sweeps%8==0)reconstruct();
    // At a fixed point, further sweeps cannot improve the bounded fit.
    if(!changed){++result.sweeps;break;}
  }
  result.multipliers.reserve(work.size());
  for(std::size_t i=0;i<work.size();++i) {
    const double value=double(work[i].u/work[i].scale);
    if(!std::isfinite(value))return finish();
    result.multipliers.push_back(std::min(columns[i].upper,std::max(columns[i].lower,value)));
  }
  // Publish the residual of the returned binary64 multipliers against the
  // ORIGINAL normals, freshly accumulated in long double; do not publish an
  // optimistic residual belonging only to the normalized internal state.
  residual.assign(gradient.begin(),gradient.end());
  for(std::size_t i=0;i<columns.size();++i)
    for(auto [j,a]:columns[i].entries)residual[j]+=L(result.multipliers[i])*a;
  result.residual.reserve(residual.size());
  for(L r:residual) {
    const double value=double(r);
    if(!std::isfinite(value))return finish();
    result.residual.push_back(value);
  }
  return finish(true);
}

} // namespace lazyipdca::qcqp::detail
