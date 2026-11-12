#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <numeric>
#include <map>
#include <queue>

namespace lazyipdca::qcqp::detail {
struct AffineCycleCertificate {
  bool verified=false;
  std::vector<int> rows;
  std::vector<std::string> weights;
  std::string constant,primal_lower_bound;
};
// Exact contradiction among original affine difference equalities. No
// constraint is removed and no tolerance is used to decide consistency.
inline AffineCycleCertificate affine_cycle_certificate(const QCQPProblem& p,double deadline) {
  using Q=boost::multiprecision::cpp_rational;
  AffineCycleCertificate out;
  auto expired=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline;};
  auto bounded=[](const Q& q){
    const auto num=numerator(q),den=denominator(q);
    return (num==0||boost::multiprecision::msb(num<0?-num:num)<16384)&&
        boost::multiprecision::msb(den)<16384;
  };
  if(p.n<=0||p.n>1000000||p.equalities.size()>1000000||expired())return out;
  const int anchor=p.n;
  std::vector<int> parent(p.n+1),rank(p.n+1,0);std::iota(parent.begin(),parent.end(),0);
  std::vector<Q> delta(p.n+1);
  struct Edge {int to,row,sign;double coefficient;};
  std::vector<std::vector<Edge>> forest(p.n+1);
  auto find=[&](int j){Q d=0;while(parent[j]!=j){d+=delta[j];j=parent[j];}return std::make_pair(j,d);};
  auto verify=[&](const std::map<int,Q>& weights){
    std::map<int,Q> linear;Q c=0,norm=0;
    for(const auto& [r,w]:weights){
      if(expired()||!bounded(w))return false;
      const auto& q=p.equalities[r];
      if(!q.signed_square.empty())return false;
      for(const auto& h:q.hessian.entries)if(h.v!=0)return false;
      c+=w*Q(q.constant);norm+=w<0?-w:w;
      q.for_each_linear_nonzero([&](int j,double a){linear[j]+=w*Q(a);});
    }
    if(c==0||norm==0)return false;
    for(const auto& [j,a]:linear)if(a!=0)return false;
    if(expired())return false;
    for(const auto& [r,w]:weights)if(w!=0){out.rows.push_back(r);out.weights.push_back(w.str());}
    out.constant=c.str();Q lower=(c<0?-c:c)/norm;out.primal_lower_bound=lower.str();out.verified=true;return true;
  };
  for(int r=0;r<int(p.equalities.size())&&!expired();++r){
    const auto& q=p.equalities[r];bool affine=q.signed_square.empty();
    for(const auto& h:q.hessian.entries)if(h.v!=0){affine=false;break;}
    if(!affine)continue;
    std::vector<std::pair<int,double>> a;
    q.for_each_linear_nonzero([&](int j,double v){if(v!=0&&a.size()<3)a.push_back({j,v});});
    if(a.empty()){if(q.constant!=0&&verify({{r,Q(1)}}))return out;continue;}
    if(a.size()>2||(a.size()==2&&Q(a[0].second)+Q(a[1].second)!=0))continue;
    const int i=a[0].first,j=a.size()==2?a[1].first:anchor;
    const double coefficient=a[0].second;const Q target=-Q(q.constant)/Q(coefficient);
    if(!bounded(target))return out;
    const auto [ri,di]=find(i);const auto [rj,dj]=find(j);
    if(ri!=rj){
      Q difference=target-di+dj;if(!bounded(difference))return out;
      if(rank[ri]<rank[rj]){parent[ri]=rj;delta[ri]=difference;}
      else {parent[rj]=ri;delta[rj]=-difference;if(rank[ri]==rank[rj])++rank[ri];}
      forest[i].push_back({j,r,1,coefficient});forest[j].push_back({i,r,-1,coefficient});continue;
    }
    if(di-dj==target)continue;
    // Recover a cycle in the union forest, then independently sum every
    // original polynomial with the exact proposed rational weights.
    std::vector<int> previous(p.n+1,-1);std::vector<Edge> arrived(p.n+1);
    std::queue<int> pending;pending.push(i);previous[i]=i;
    while(!pending.empty()&&previous[j]<0&&!expired()){
      int u=pending.front();pending.pop();
      for(const auto& e:forest[u])if(previous[e.to]<0){previous[e.to]=u;arrived[e.to]=e;pending.push(e.to);}
    }
    if(previous[j]<0)return out;
    std::map<int,Q> weights;weights[r]=Q(1)/Q(coefficient);
    for(int u=j;u!=i;u=previous[u]){
      const auto& e=arrived[u];weights[e.row]-=Q(e.sign)/Q(e.coefficient);
    }
    if(verify(weights))return out;
  }
  return out;
}
} // namespace lazyipdca::qcqp::detail
