#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <map>
namespace lazyipdca::qcqp::detail {
struct ConvexAggregateCertificate {
  bool verified=false,lower=false;
  int bound_variable=-1,equality_sign=0;
  std::string bound_weight,minimum,weight_norm,primal_lower_bound;
  std::vector<int> inequality_rows;
  std::vector<std::string> inequality_weights;
};
// Sum affine equality balances, eliminate one finite original bound, and
// complete squares after adding nonnegative diagonal quadratic inequalities.
// A positive global minimum of this nonpositive combination is a certificate
// of infeasibility. Rational arithmetic retains every binary64 coefficient.
inline ConvexAggregateCertificate convex_aggregate_certificate(const QCQPProblem& p,double tolerance,double deadline){
  using Q=boost::multiprecision::cpp_rational;
  ConvexAggregateCertificate out;
  auto expired=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline;};
  auto bounded=[](const Q& v){auto n=numerator(v);if(n<0)n=-n;auto d=denominator(v);return (n==0||boost::multiprecision::msb(n)<16384)&&boost::multiprecision::msb(d)<16384;};
  if(p.n<1||p.n>1000000||p.equalities.empty()||p.inequalities.empty()||expired())return out;
  std::map<int,Q> aggregate;Q constant=0;
  for(const auto& q:p.equalities){
    if(expired()||!q.signed_square.empty())return out;
    for(const auto& h:q.hessian.entries)if(h.v!=0)return out;
    constant+=Q(q.constant);q.for_each_linear_nonzero([&](int j,double v){aggregate[j]+=Q(v);});
  }
  std::vector<int> pivots;
  for(auto [j,a]:aggregate)if(a!=0&&(std::isfinite(p.lb[j])||std::isfinite(p.ub[j])))pivots.push_back(j);
  if(pivots.empty())return out;
  std::vector<int> first(p.n,-1);std::vector<Q> first_diagonal(p.n);
  for(int i=0;i<int(p.inequalities.size());++i){
    const auto& q=p.inequalities[i];bool valid=q.signed_square.empty();
    q.for_each_linear_nonzero([&](int,double a){if(a!=0)valid=false;});
    for(const auto& h:q.hessian.entries)if(h.v!=0&&(h.i!=h.j||h.v<0))valid=false;
    if(valid)for(const auto& h:q.hessian.entries)if(h.v>0&&first[h.i]<0){first[h.i]=i;first_diagonal[h.i]=Q(h.v)/2;}
    if(expired())return out;
  }
  for(int t:pivots)for(bool lower:{true,false}){
    const double bound=lower?p.lb[t]:p.ub[t];if(!std::isfinite(bound)||expired())continue;
    const int sign=(aggregate[t]>0?1:-1)*(lower?1:-1);const Q weight=aggregate[t]>0?aggregate[t]:-aggregate[t];
    std::map<int,Q> linear;for(auto [j,a]:aggregate)linear[j]=sign*a;linear[t]=0;
    Q c=sign*constant+(lower?weight*Q(bound):-weight*Q(bound));
    std::map<int,Q> selected;bool covered=true;
    for(auto [j,a]:linear)if(a!=0){
      if(expired())return out;
      if(first[j]<0){covered=false;break;}
      selected[first[j]]+=(a<0?-a:a)/(2*first_diagonal[j]);
    }
    if(!covered)continue;
    std::map<int,Q> diagonal;Q norm=Q(p.equalities.size())+weight;
    for(auto [i,w]:selected){
      if(expired()||!bounded(w)||!bounded(c)||!bounded(norm))return out;
      const auto& q=p.inequalities[i];c+=w*Q(q.constant);norm+=w;
      for(const auto& h:q.hessian.entries)if(h.v!=0)diagonal[h.i]+=w*Q(h.v)/2;
    }
    Q minimum=c;
    for(auto [j,a]:linear)if(a!=0){if(expired()||!bounded(minimum))return out;if(diagonal[j]<=0){covered=false;break;}minimum-=a*a/(4*diagonal[j]);}
    if(!covered||minimum<=0||norm<=0||minimum<=Q(tolerance)*norm||expired())continue;
    // All entries above are the original polynomials. Completing each exact
    // nonnegative diagonal square independently proves the global bound.
    out.verified=true;out.lower=lower;out.bound_variable=t;out.equality_sign=sign;
    out.bound_weight=weight.str();out.minimum=minimum.str();out.weight_norm=norm.str();
    Q lower_bound=minimum/norm;out.primal_lower_bound=lower_bound.str();
    for(auto [i,w]:selected){out.inequality_rows.push_back(i);out.inequality_weights.push_back(w.str());}
    return out;
  }
  return out;
}
}
