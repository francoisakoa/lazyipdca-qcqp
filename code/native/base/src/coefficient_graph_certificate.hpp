#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <chrono>
#include <numeric>
#include <queue>
#include <set>
#include <cstdint>

namespace lazyipdca::qcqp::detail {
struct CoefficientGraphCertificate {
  bool constructed=false;
  int vertices=0,edges=0,components=0,independent=0;
  Vec x,multipliers;
  std::string reason;
};
// Exact coefficient-domain certificate, not a model selected by a label.
// Q0 consists of disjoint constant negative blocks beta_a * 11^T.
// One equality is sigma/2 * ||x||^2+c=0, sigma>0,c<0. Every other
// equality is gamma_e times an inner product between the SAME two vertices
// in each block coordinate. A maximal independent set S supplies a KKT point:
//   x_i=s for i in S in one coordinate, zero elsewhere; s^2=-2c/(sigma*|S|).
//   lambda_sphere=-beta*|S|/sigma.
// For each unselected vertex assign one incident selected neighbour's row
// lambda_e=-beta*|S|/gamma_e. This cancels its objective gradient exactly.
// The caller MUST audit all ORIGINAL functions and multiplier signs. This
// establishes first-order stationarity only, never global or second-order
// optimality. No input point, coefficient or equality is replaced.
inline CoefficientGraphCertificate coefficient_graph_certificate(
    const QCQPProblem& p,const Settings& st,double deadline) {
  using Clock=std::chrono::steady_clock;
  CoefficientGraphCertificate out;
  auto expired=[&]{return std::chrono::duration<double>(Clock::now().time_since_epoch()).count()>=deadline;};
  auto reject=[&](const char* reason){out.reason=reason;out.x.clear();out.multipliers.clear();return out;};
  if(st.standard_r10_policy!="auto"||st.convexification_study!="off"||
      (st.dca_engine!="auto"&&st.dca_engine!="majorants")||
      (st.outer_algorithm!="auto"&&st.outer_algorithm!="lazy-dca")||
      st.study_controller!="r3"||st.convexification_schedule!="lazy"||
      st.rank_presolve_context||st.original_private_phase||
      (st.accepted_checkpoint_callback&&!st.accepted_checkpoint_builtin)||
      !p.inequalities.empty()||p.equalities.size()<2||p.n<2||p.n>80000||
      p.has_signed_square()||p.objective.hessian.entries.size()>2000000||expired())
    return reject("outside-coefficient-certificate-domain");
  for(int j=0;j<p.n;++j)if(std::isfinite(p.lb[j])||std::isfinite(p.ub[j])||!std::isfinite(p.x0[j]))return reject("finite-bounds-or-nonfinite-start");
  bool linear=false;p.objective.for_each_linear_nonzero([&](int,double){linear=true;});
  if(linear||p.objective.hessian.entries.empty())return reject("objective-not-pure-negative-blocks");
  // Preserve every already feasible supplied point and its ordinary trajectory.
  double initial=0;
  for(const auto& q:p.equalities){
    q.for_each_linear_nonzero([&](int,double){linear=true;});
    if(linear)return reject("affine-source-coefficient-present");
    initial=std::max(initial,std::abs(q.value(p.x0)));
  }
  if(!std::isfinite(initial)||initial<=st.feasibility_tolerance)return reject("feasible-start-preserved");
  std::vector<int> parent(p.n);std::iota(parent.begin(),parent.end(),0);
  auto find=[&](int j){while(parent[j]!=j){parent[j]=parent[parent[j]];j=parent[j];}return j;};
  std::vector<std::uint64_t> keys;keys.reserve(p.objective.hessian.entries.size());
  for(const auto& h:p.objective.hessian.entries)if(h.v!=0){
    if(h.v>=0)return reject("objective-block-not-negative");
    parent[find(h.i)]=find(h.j);
    keys.push_back(std::uint64_t(h.i)*p.n+h.j);
  }
  if(expired())return reject("objective-scan-deadline");
  std::sort(keys.begin(),keys.end());
  if(std::adjacent_find(keys.begin(),keys.end())!=keys.end())return reject("duplicate-objective-coefficient");
  std::vector<std::vector<int>> components;
  std::vector<int> root_to_component(p.n,-1),component(p.n),vertex(p.n);
  for(int j=0;j<p.n;++j){int root=find(j),&a=root_to_component[root];if(a<0){a=int(components.size());components.push_back({});}component[j]=a;vertex[j]=int(components[a].size());components[a].push_back(j);}
  const int d=int(components.size()),N=int(components[0].size());
  if(d<1||d>16||N<2||N>5000)return reject("component-workspace-cost-guard");
  for(const auto& a:components)if(int(a.size())!=N)return reject("unequal-block-coordinate-sizes");
  std::vector<double> beta(d,0);
  std::vector<std::size_t> count(d,0);
  for(const auto& h:p.objective.hessian.entries)if(h.v!=0){
    int a=component[h.i];if(component[h.j]!=a)return reject("objective-component-mismatch");
    if(count[a]&&beta[a]!=h.v)return reject("nonconstant-objective-block");
    beta[a]=h.v;++count[a];
  }
  for(int a=0;a<d;++a)if(count[a]!=std::size_t(N)*(N+1)/2)return reject("incomplete-negative-objective-block");
  struct Edge{int i,j,row;double gamma;};std::vector<Edge> edges;
  std::vector<std::set<int>> graph(N);
  int sphere=-1;double sigma=0,c=0;
  for(int e=0;e<int(p.equalities.size());++e){
    if(expired())return reject("constraint-scan-deadline");
    const auto& q=p.equalities[e];
    int nh=0;bool diagonal=true;double diagonal_coefficient=0;std::vector<unsigned char> seen(p.n,0);
    for(const auto& h:q.hessian.entries)if(h.v!=0){
      ++nh;if(h.i!=h.j||h.v<=0||seen[h.i]||(diagonal_coefficient&&diagonal_coefficient!=h.v))diagonal=false;
      seen[h.i]=1;diagonal_coefficient=h.v;
    }
    if(diagonal&&nh==p.n&&q.constant<0){
      if(sphere>=0)return reject("multiple-sphere-rows");
      sphere=e;sigma=diagonal_coefficient;c=q.constant;continue;
    }
    if(q.constant!=0||nh!=d)return reject("constraint-not-sphere-or-inner-product");
    int i=-1,j=-1;double gamma=0;std::vector<unsigned char> used(d,0);
    for(const auto& h:q.hessian.entries)if(h.v!=0){
      const int a=component[h.i];int u=vertex[h.i],v=vertex[h.j];if(u>v)std::swap(u,v);
      if(component[h.j]!=a||u==v||used[a])return reject("invalid-inner-product-coordinate");
      if(i<0){i=u;j=v;gamma=h.v;}
      if(i!=u||j!=v||gamma!=h.v)return reject("unaligned-inner-product-blocks");
      used[a]=1;
    }
    edges.push_back({i,j,e,gamma});graph[i].insert(j);graph[j].insert(i);
  }
  if(sphere<0||edges.empty()||edges.size()>100000)return reject("missing-sphere-or-graph-workspace-guard");
  using Item=std::pair<int,int>;
  std::priority_queue<Item,std::vector<Item>,std::greater<Item>> queue;
  std::vector<int> degree(N);std::vector<unsigned char> available(N,1),selected(N,0);
  for(int j=0;j<N;++j){degree[j]=int(graph[j].size());queue.push({degree[j],j});}
  int k=0;
  while(!queue.empty()){
    if(expired())return reject("independent-set-deadline");
    auto [deg,j]=queue.top();queue.pop();if(!available[j]||degree[j]!=deg)continue;
    selected[j]=1;++k;std::vector<int> removed{j};
    for(int u:graph[j])if(available[u])removed.push_back(u);
    for(int u:removed)available[u]=0;
    for(int u:removed)for(int v:graph[u])if(available[v]){--degree[v];queue.push({degree[v],v});}
  }
  int chosen=int(std::min_element(beta.begin(),beta.end())-beta.begin());
  const long double s2=-2.L*c/(static_cast<long double>(sigma)*k);
  const double s=static_cast<double>(std::sqrt(s2));
  if(!(s>0)||!std::isfinite(s))return reject("nonrepresentable-sphere-point");
  out.x.assign(p.n,0);out.multipliers.assign(p.equalities.size(),0);
  for(int j=0;j<N;++j)if(selected[j])out.x[components[chosen][j]]=s;
  out.multipliers[sphere]=static_cast<double>(-static_cast<long double>(beta[chosen])*k/sigma);
  auto covered=selected;
  for(const auto& edge:edges){
    int unselected=-1;
    if(selected[edge.i]&&!covered[edge.j])unselected=edge.j;
    else if(selected[edge.j]&&!covered[edge.i])unselected=edge.i;
    if(unselected>=0){out.multipliers[edge.row]=static_cast<double>(-static_cast<long double>(beta[chosen])*k/edge.gamma);covered[unselected]=1;}
  }
  if(std::any_of(covered.begin(),covered.end(),[](auto b){return !b;})||
      !std::all_of(out.multipliers.begin(),out.multipliers.end(),[](double a){return std::isfinite(a);})||expired())return reject("certificate-coverage-finiteness-or-deadline");
  out.constructed=true;out.vertices=N;out.edges=int(edges.size());out.components=d;out.independent=k;
  out.reason="exact-negative-block-objective-sphere-and-orthogonality-graph";return out;
}
} // namespace lazyipdca::qcqp::detail
