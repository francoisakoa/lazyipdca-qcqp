#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/terminal_multiplier_fit.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <tuple>
#include <optional>
#include "rank_structural_retry.hpp"
#include "rank_retry_statistics.hpp"

namespace lazyipdca::qcqp {
namespace {
using L=long double;
using Pair=std::pair<int,int>;
using Key=std::array<int,4>;
using Clock=std::chrono::steady_clock;
Pair pair(int a,int b){return {std::min(a,b),std::max(a,b)};}
// Error-free expansion of a sum of input doubles. A transformed coefficient
// must be exactly representable as a double; otherwise the reduction is refused.
struct ExactSum {
  Vec partials;
  void add(double x) {
    if(!std::isfinite(x))throw std::runtime_error("rank-presolve: coefficient overflow");
    std::size_t i=0;
    for(double y:partials){
      if(std::abs(x)<std::abs(y))std::swap(x,y);
      const double hi=x+y;
      if(!std::isfinite(hi))throw std::runtime_error("rank-presolve: coefficient sum overflow");
      const double lo=y-(hi-x);
      if(lo!=0.0)partials[i++]=lo;
      x=hi;
    }
    partials.resize(i);
    if(x!=0.0)partials.push_back(x);
  }
  int sign() const {return partials.empty()?0:(partials.back()>0?1:-1);}
  double exact() const {
    if(partials.size()>1)throw std::runtime_error("rank-presolve: unrepresentable exact coefficient sum");
    return partials.empty()?0.0:partials[0];
  }
};
// Exact sign of a*b*b+c for moderate finite inputs. FMA TwoProduct
// expansions are error-free here: exponent guards rule out overflow and
// subnormal loss even for the low parts of these three-factor products.
bool radial_implies_bound(double a,double b,double c) {
  if(!(a>0)||!(c<0)||!std::isfinite(b)||b==0)return false;
  for(double x:{a,b,c})if(!std::isfinite(x)||std::ilogb(std::abs(x)) < -128 || std::ilogb(std::abs(x))>128)return false;
  const double p=a*b,e=std::fma(a,b,-p);
  const double pp=p*b,pe=std::fma(p,b,-pp);
  const double ep=e*b,ee=std::fma(e,b,-ep);
  ExactSum sum;for(double x:{pp,pe,ep,ee,c})sum.add(x);
  return sum.sign()>=0;
}
// An exact singleton affine row can be represented by a box side only if
// its root is exactly a binary64 number. FMA certifies a*b+c == 0 without
// an inexact quotient test; the exponent guard prevents a nonzero exact
// product residual from underflowing to zero. Refuse uncertain arithmetic.
bool exact_affine_root(double a,double c,double& bound) {
  if(a==0||!std::isfinite(a)||!std::isfinite(c))return false;
  bound=-c/a;
  if(!std::isfinite(bound))return false;
  if(bound==0)return c==0;
  const int exponent=std::ilogb(std::abs(a))+std::ilogb(std::abs(bound));
  if(exponent < -900 || exponent > 900)return false;
  return std::fma(a,bound,c)==0;
}
std::map<Pair,double> hessian(const QuadraticFunction& q) {
  std::map<Pair,ExactSum> sums;
  for(auto e:q.hessian.entries)if(e.v!=0)sums[{e.i,e.j}].add(e.v);
  std::map<Pair,double> out;
  for(const auto& it:sums){const double v=it.second.exact();if(v!=0)out[it.first]=v;}
  return out;
}
bool no_linear(const QuadraticFunction& q) {
  bool empty=true;q.for_each_linear_nonzero([&](int,double v){if(v!=0)empty=false;});return empty;
}
struct Signature { Key vars; double r=0,t=0; int equation=-1; };
bool leaf_signature(const QuadraticFunction& q,int id,Signature& out) {
  if(q.constant!=0||!no_linear(q)||q.hessian.entries.size()>8)return false;
  const auto H=hessian(q);
  std::vector<int> diag;std::set<int> vars;
  for(auto e:H){vars.insert(e.first.first);vars.insert(e.first.second);
    if(e.first.first==e.first.second)diag.push_back(e.first.first);}
  if(diag.size()!=2||vars.size()!=4)return false;
  const int u=diag[0],v=diag[1];
  const double d=H.at({u,u});
  if(d!=H.at({v,v}))return false;
  const double r=d*0.5;
  if(r==0||2*r!=d)return false;
  std::vector<int> parents;for(int j:vars)if(j!=u&&j!=v)parents.push_back(j);
  auto at=[&](int a,int b){auto it=H.find(pair(a,b));return it==H.end()?0.0:it->second;};
  for(int swap=0;swap<2;++swap){
    const int a=parents[swap],b=parents[1-swap];
    const double t=at(a,v);
    if(at(a,u)!=-r||at(b,v)!=-r||at(b,u)!=-t||at(u,v)!=0||at(a,b)!=0)continue;
    std::map<Pair,double> expected={{{u,u},d},{{v,v},d}};
    expected[pair(a,u)]=-r;expected[pair(b,v)]=-r;
    if(t!=0){expected[pair(a,v)]=t;expected[pair(b,u)]=-t;}
    if(expected!=H)continue;
    out={{u,v,a,b},r,t,id};return true;
  }
  return false;
}
bool independent(double a,double b,double c,double d) {
  const L ad=L(a)*d,bc=L(b)*c;
  const L det=ad-bc,mag=std::abs(ad)+std::abs(bc);
  // A conservative arithmetic guard, not a numerical rank decision used to
  // delete rows. Independence is certified only away from product roundoff.
  return std::isfinite(det)&&std::abs(det)>64*std::numeric_limits<L>::epsilon()*mag;
}
template<class Map>
QuadraticFunction substitute_map(const QuadraticFunction& q,Map map,int n) {
  QuadraticFunction out;out.name=q.name;out.constant=q.constant;
  out.hessian.n=n;out.compact_linear_dimension=n;
  std::map<int,ExactSum> ls;std::map<Pair,ExactSum> hs;
  q.for_each_linear_nonzero([&](int j,double v){if(v!=0)ls[map(j)].add(v);});
  for(auto e:q.hessian.entries){
    if(e.v==0)continue;
    const int a=map(e.i),b=map(e.j);
    // An old off-diagonal term has twice a diagonal's stored-Hessian weight.
    const double v=(e.i!=e.j&&a==b)?2*e.v:e.v;
    hs[pair(a,b)].add(v);
  }
  for(const auto& it:ls){double v=it.second.exact();if(v!=0)out.compact_linear.push_back({it.first,v});}
  for(const auto& it:hs){double v=it.second.exact();if(v!=0)out.hessian.entries.push_back({it.first.first,it.first.second,v});}
  return out;
}
QuadraticFunction substitute(const QuadraticFunction& q,const std::vector<int>& map,int n) {
  return substitute_map(q,[&](int j){return map[j];},n);
}
std::size_t qhash(const QuadraticFunction& q) {
  std::size_t h=1469598103934665603ULL;
  auto mix=[&](std::size_t v){h^=v;h*=1099511628211ULL;};
  auto bits=[&](double d){std::uint64_t b=0;if(d==0)d=0;std::memcpy(&b,&d,sizeof b);mix(b);};
  bits(q.constant);
  q.for_each_linear_nonzero([&](int j,double d){mix(j);bits(d);});mix(0xabcdef);
  for(auto e:q.hessian.entries){mix(e.i);mix(e.j);bits(e.v);}return h;
}
bool same(const QuadraticFunction& a,const QuadraticFunction& b) {
  if(a.constant!=b.constant||a.compact_linear!=b.compact_linear||a.hessian.entries.size()!=b.hessian.entries.size())return false;
  for(std::size_t i=0;i<a.hessian.entries.size();++i){auto x=a.hessian.entries[i],y=b.hessian.entries[i];
    if(x.i!=y.i||x.j!=y.j||x.v!=y.v)return false;}
  return true;
}
void gradient(const QuadraticFunction& q,const Vec& x,L weight,std::vector<L>& g) {
  if(weight==0)return;
  q.for_each_linear_nonzero([&](int j,double a){g[j]+=weight*L(a);});
  for(auto e:q.hessian.entries){g[e.i]+=weight*L(e.v)*x[e.j];if(e.i!=e.j)g[e.j]+=weight*L(e.v)*x[e.i];}
}
L derivative(const QuadraticFunction& q,const Vec& x,int j) {
  L v=0;q.for_each_linear_nonzero([&](int k,double a){if(k==j)v+=a;});
  for(auto e:q.hessian.entries){if(e.i==j)v+=L(e.v)*x[e.j];if(e.j==j&&e.i!=j)v+=L(e.v)*x[e.i];}return v;
}
} // namespace

ConstraintPresolve::ConstraintPresolve(const QCQPProblem& p,bool remove_implied_bounds,
    bool remove_equality_implied_bounds,bool enable_exact_fixed,
    bool remove_affine_equality_implied_bounds):original_(p) {
  const auto start=Clock::now();stats_.enabled=true;stats_.reason="no-proven-reduction";
  stats_.original_variables=p.n;stats_.original_equalities=p.equalities.size();
  stats_.original_inequalities=p.inequalities.size();
  // An explicit radial lower inequality c+a(u^2+v^2)<=0, c>0,a<0,
  // excludes the zero-voltage branch on every feasible point, not just x0.
  std::map<Pair,int> nonzero;
  for(int i=0;i<(int)p.inequalities.size();++i){const auto& q=p.inequalities[i];
    if(!(q.constant>0)||!no_linear(q)||q.hessian.entries.size()!=2)continue;
    const auto H=hessian(q);if(H.size()!=2)continue;
    auto a=H.begin(),b=std::next(a);
    if(a->first.first==a->first.second&&b->first.first==b->first.second&&
       a->second<0&&a->second==b->second)nonzero[{a->first.first,b->first.first}]=i;
  }
  std::map<Key,std::vector<Signature>> groups;
  for(int e=0;e<(int)p.equalities.size();++e){Signature s;
    if(leaf_signature(p.equalities[e],e,s))groups[s.vars].push_back(s);}
  std::vector<int> parent(p.n);std::iota(parent.begin(),parent.end(),0);
  auto root=[&](int j){while(parent[j]!=j)j=parent[j];return j;};
  // Each candidate contraction is checked on every affected original
  // polynomial BEFORE commit. Reject only that contraction when an exact
  // coalesced coefficient is not representable, instead of losing all leaves.
  auto function=[&](int id)->const QuadraticFunction& {
    if(id==0)return p.objective;
    if(id<=int(p.equalities.size()))return p.equalities[id-1];
    return p.inequalities[id-1-p.equalities.size()];
  };
  std::vector<std::vector<int>> incidence(p.n);
  for(int id=0;id<1+int(p.equalities.size()+p.inequalities.size());++id){
    const auto& q=function(id);
    q.for_each_linear_nonzero([&](int j,double v){if(v!=0)incidence[j].push_back(id);});
    for(auto e:q.hessian.entries)if(e.v!=0){incidence[e.i].push_back(id);if(e.i!=e.j)incidence[e.j].push_back(id);}
  }
  for(auto& ids:incidence){std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());}
  std::set<int> child_used,equation_used;
  for(const auto& entry:groups){const auto& v=entry.second;if(v.size()<2)continue;
    ++stats_.leaf_candidates;
    const auto k=entry.first;const int u=k[0],w=k[1],a=k[2],b=k[3];
    auto nz=nonzero.find({u,w});
    const bool bound_nonzero=p.lb[u]>0||p.ub[u]<0||p.lb[w]>0||p.ub[w]<0;
    if(nz==nonzero.end()&&!bound_nonzero){++stats_.nonzero_rejected;continue;}
    if((p.x0[u]==0&&p.x0[w]==0)||p.x0[u]!=p.x0[a]||p.x0[w]!=p.x0[b]){++stats_.start_rejected;continue;}
    if(child_used.count(u)||child_used.count(w)||root(u)!=u||root(w)!=w)continue;
    int j=-1;
    for(int i=1;i<(int)v.size();++i)if(independent(v[0].r,v[0].t,v[i].r,v[i].t)) {j=i;break;}
    if(j<0||equation_used.count(v[0].equation)||equation_used.count(v[j].equation))continue;
    const int ra=root(a),rb=root(b);
    // Disjoint coordinate pairs form a forest. Cross-axis identification or
    // cycles are not part of this certified reduction.
    if(ra==u||ra==w||rb==u||rb==w||ra==rb)continue;
    parent[u]=ra;parent[w]=rb;
    std::vector<int> affected=incidence[u];affected.insert(affected.end(),incidence[w].begin(),incidence[w].end());
    std::sort(affected.begin(),affected.end());affected.erase(std::unique(affected.begin(),affected.end()),affected.end());
    bool exact=true;
    try {for(int id:affected)(void)substitute_map(function(id),root,p.n);}
    catch(const std::runtime_error&){exact=false;}
    if(!exact){parent[u]=u;parent[w]=w;++stats_.arithmetic_rejected;continue;}
    for(Pair merge:{Pair{u,ra},Pair{w,rb}}){
      auto& dst=incidence[merge.second];const auto& src=incidence[merge.first];
      dst.insert(dst.end(),src.begin(),src.end());std::sort(dst.begin(),dst.end());dst.erase(std::unique(dst.begin(),dst.end()),dst.end());
      incidence[merge.first].clear();
    }
    leaves_.push_back({u,w,a,b,v[0].equation,v[j].equation,nz==nonzero.end()?-1:nz->second,
                       v[0].r,v[0].t,v[j].r,v[j].t});
    parent[u]=ra;parent[w]=rb;child_used.insert(u);child_used.insert(w);
    equation_used.insert(v[0].equation);equation_used.insert(v[j].equation);
  }
  // Enforce pairwise-disjoint voltage coordinates for adjoint reconstruction.
  std::map<int,int> mate;
  bool compatible=true;
  for(const auto& e:leaves_)for(Pair pr:{Pair{e.u,e.v},Pair{e.a,e.b}}){
    if((mate.count(pr.first)&&mate[pr.first]!=pr.second)||(mate.count(pr.second)&&mate[pr.second]!=pr.first))compatible=false;
    mate[pr.first]=pr.second;mate[pr.second]=pr.first;
  }
  auto build=[&](){
    reduced_={};reduced_.name=p.name;variable_map_.assign(p.n,-1);
    std::vector<int> roots;
    for(int j=0;j<p.n;++j)if(root(j)==j){variable_map_[j]=roots.size();roots.push_back(j);}
    reduced_.n=roots.size();for(int j=0;j<p.n;++j)variable_map_[j]=variable_map_[root(j)];
    reduced_.x0.resize(roots.size());reduced_.lb.resize(roots.size());reduced_.ub.resize(roots.size());
    lower_origins_=upper_origins_=roots;
    for(int k=0;k<(int)roots.size();++k){int j=roots[k];reduced_.x0[k]=p.x0[j];reduced_.lb[k]=p.lb[j];reduced_.ub[k]=p.ub[j];}
    for(int j=0;j<p.n;++j){int k=variable_map_[j];
      if(p.lb[j]>reduced_.lb[k]){reduced_.lb[k]=p.lb[j];lower_origins_[k]=j;}
      if(p.ub[j]<reduced_.ub[k]){reduced_.ub[k]=p.ub[j];upper_origins_[k]=j;}
      if(p.x0[j]!=reduced_.x0[k])throw std::runtime_error("rank-presolve: initial point not preserved");
    }
    reduced_.objective=substitute(p.objective,variable_map_,reduced_.n);
    eq_origins_.clear();eq_groups_.assign(p.equalities.size(),-1);ineq_origins_.clear();ineq_groups_.assign(p.inequalities.size(),-1);
    stats_.duplicate_equalities=stats_.zero_equalities=stats_.duplicate_inequalities=stats_.constant_inequalities=0;
    std::unordered_map<std::size_t,std::vector<int>> eqseen,inseen;
    for(int e=0;e<(int)p.equalities.size();++e){auto q=substitute(p.equalities[e],variable_map_,reduced_.n);
      if(q.compact_linear.empty()&&q.hessian.entries.empty()&&q.constant==0){++stats_.zero_equalities;continue;}
      auto& bucket=eqseen[qhash(q)];int found=-1;
      for(int i:bucket)if(same(q,reduced_.equalities[i])){found=i;break;}
      if(found>=0){++stats_.duplicate_equalities;eq_groups_[e]=found;continue;}
      eq_groups_[e]=reduced_.equalities.size();
      bucket.push_back(reduced_.equalities.size());eq_origins_.push_back(e);reduced_.equalities.push_back(std::move(q));
    }
    for(int i=0;i<(int)p.inequalities.size();++i){auto q=substitute(p.inequalities[i],variable_map_,reduced_.n);
      if(q.compact_linear.empty()&&q.hessian.entries.empty()&&q.constant<=0){++stats_.constant_inequalities;continue;}
      auto& bucket=inseen[qhash(q)];int found=-1;
      for(int j:bucket)if(same(q,reduced_.inequalities[j])){found=j;break;}
      if(found>=0){++stats_.duplicate_inequalities;ineq_groups_[i]=found;continue;}
      const int index=reduced_.inequalities.size();bucket.push_back(index);ineq_groups_[i]=index;
      ineq_origins_.push_back(i);reduced_.inequalities.push_back(std::move(q));
    }
    // All selected balance equations must have vanished identically.
    for(const auto& e:leaves_)for(int id:{e.e1,e.e2}){
      const auto q=substitute(p.equalities[id],variable_map_,reduced_.n);
      if(q.constant!=0||!q.compact_linear.empty()||!q.hessian.entries.empty())
        throw std::runtime_error("rank-presolve: balance identity not exact");
    }
    reduced_.validate();
  };
  try{if(!compatible)throw std::runtime_error("rank-presolve: overlapping coordinate pairs");build();}
  catch(const std::runtime_error&){
    // Fail closed. Keep the original variables if coefficient coalescing would
    // round; exact duplicate-row reduction is still permissible.
    ++stats_.arithmetic_rejected;leaves_.clear();std::iota(parent.begin(),parent.end(),0);build();
  }
  // Recover eliminated leaves before their parent (adjoint elimination).
  std::map<int,int> child_index;for(int i=0;i<(int)leaves_.size();++i){child_index[leaves_[i].u]=i;child_index[leaves_[i].v]=i;}
  std::vector<int> count(leaves_.size(),0),next(leaves_.size(),-1);
  for(int i=0;i<(int)leaves_.size();++i){auto it=child_index.find(leaves_[i].a);
    if(it!=child_index.end()){next[i]=it->second;++count[it->second];}}
  std::deque<int> ready;for(int i=0;i<(int)count.size();++i)if(count[i]==0)ready.push_back(i);
  while(!ready.empty()){int i=ready.front();ready.pop_front();lift_order_.push_back(i);
    if(next[i]>=0&&--count[next[i]]==0)ready.push_back(next[i]);}
  if(lift_order_.size()!=leaves_.size())throw std::runtime_error("rank-presolve: reconstruction graph is cyclic");
  std::vector<unsigned char> needed(p.n,0);
  for(const auto& e:leaves_){needed[e.u]=1;needed[e.v]=1;}
  if(!leaves_.empty())for(int id=0;id<1+int(p.equalities.size()+p.inequalities.size());++id){
    const auto& q=function(id);bool touches=false;
    q.for_each_linear_nonzero([&](int j,double v){if(v!=0&&needed[j])touches=true;});
    for(auto e:q.hessian.entries)if(e.v!=0&&(needed[e.i]||needed[e.j])){touches=true;break;}
    if(touches)lift_function_ids_.push_back(id);
  }
  stats_.leaf_pairs=leaves_.size();stats_.reduced_variables=reduced_.n;
  // Expose EXACTLY FIXED coordinates certified by opposing affine singleton
  // sides and/or original box sides. Keep every nonfixing singleton and its
  // historical inequality/slack trajectory, including redundant rows. Broad
  // box folding would change scaling, startup and globalization needlessly.
  // Neither the input nor its serialized x0 is changed.
  affine_lower_sources_.assign(reduced_.n,-1);
  affine_upper_sources_.assign(reduced_.n,-1);
  affine_inequality_sources_.assign(p.inequalities.size(),-1);
  std::vector<AffineBound> lower_proof(reduced_.n),upper_proof(reduced_.n);
  Vec proposed_lower=reduced_.lb,proposed_upper=reduced_.ub;
  std::vector<int> singleton_variable(reduced_.inequalities.size(),-1);
  std::vector<QuadraticFunction> kept_inequalities;
  std::vector<int> kept_origins,old_to_new(reduced_.inequalities.size(),-1);
  for(int i=0;i<(int)reduced_.inequalities.size();++i) {
    const auto& q=reduced_.inequalities[i];
    int variable=-1,count=0;double coefficient=0,bound=0;
    q.for_each_linear_nonzero([&](int j,double a){variable=j;coefficient=a;++count;});
    const bool affine=q.signed_square.empty()&&p.inequalities[ineq_origins_[i]].signed_square.empty()&&
        std::none_of(q.hessian.entries.begin(),q.hessian.entries.end(),
        [](const auto& e){return e.v!=0;});
    const bool singleton=affine&&count==1;
    if(enable_exact_fixed&&remove_implied_bounds&&singleton&&exact_affine_root(coefficient,q.constant,bound)) {
      singleton_variable[i]=variable;
      if(coefficient>0&&bound<proposed_upper[variable]) {
        proposed_upper[variable]=bound;
        upper_proof[variable]={variable,1,ineq_origins_[i],coefficient,bound};
      } else if(coefficient<0&&bound>proposed_lower[variable]) {
        proposed_lower[variable]=bound;
        lower_proof[variable]={variable,-1,ineq_origins_[i],coefficient,bound};
      }
    } else if(enable_exact_fixed&&remove_implied_bounds&&singleton)++stats_.arithmetic_rejected;
  }
  std::vector<char> newly_fixed(reduced_.n,0);
  for(int j=0;j<reduced_.n;++j)if(reduced_.lb[j]!=reduced_.ub[j]&&
      std::isfinite(proposed_lower[j])&&proposed_lower[j]==proposed_upper[j]) {
    newly_fixed[j]=1;reduced_.lb[j]=proposed_lower[j];reduced_.ub[j]=proposed_upper[j];
  }
  for(int i=0;i<(int)reduced_.inequalities.size();++i) {
    const int j=singleton_variable[i];
    if(j>=0&&newly_fixed[j])continue; // original row is audited after substitution
    old_to_new[i]=kept_inequalities.size();
    kept_origins.push_back(ineq_origins_[i]);kept_inequalities.push_back(reduced_.inequalities[i]);
  }
  for(int j=0;j<reduced_.n;++j)for(int side:{-1,1}) {
    if(!newly_fixed[j])continue;
    const auto& proof=side<0?lower_proof[j]:upper_proof[j];
    if(proof.inequality<0)continue;
    const int id=affine_bounds_.size();affine_bounds_.push_back(proof);
    (side<0?affine_lower_sources_[j]:affine_upper_sources_[j])=id;
    affine_inequality_sources_[proof.inequality]=id;
  }
  for(int& group:ineq_groups_)if(group>=0)group=old_to_new[group];
  reduced_.inequalities=std::move(kept_inequalities);ineq_origins_=std::move(kept_origins);
  stats_.reduced_equalities=reduced_.equalities.size();stats_.reduced_inequalities=reduced_.inequalities.size();
  int ob=0,nb=0;for(int j=0;j<p.n;++j)ob+=std::isfinite(p.lb[j])+std::isfinite(p.ub[j]);
  for(int j=0;j<reduced_.n;++j)nb+=std::isfinite(reduced_.lb[j])+std::isfinite(reduced_.ub[j]);
  stats_.duplicate_bounds=std::max(0,ob-nb);
  // Keep the radial inequality. Drop only a box SIDE which that retained
  // inequality globally implies, with a proof at the exact binary coefficients.
  // This handles a radius vs bound mismatch at the last bit in the correct
  // direction; the stronger quadratic is never replaced by a weaker box.
  if(remove_implied_bounds)for(int i=0;i<(int)reduced_.inequalities.size();++i){
    const auto& q=reduced_.inequalities[i];
    if(!(q.constant<0)||!no_linear(q))continue;
    bool diagonal=true;
    for(auto e:q.hessian.entries)if(e.i!=e.j||!(e.v>0)){diagonal=false;break;}
    if(!diagonal)continue;
    for(auto e:q.hessian.entries){const int j=e.i;const double a=e.v*0.5;
      if(2*a!=e.v||reduced_.lb[j]==reduced_.ub[j])continue;
      if(reduced_.ub[j]>0&&radial_implies_bound(a,reduced_.ub[j],q.constant)){
        reduced_.ub[j]=INFINITY;implied_bounds_.push_back({j,1,i});
      }
      if(reduced_.lb[j]<0&&radial_implies_bound(a,reduced_.lb[j],q.constant)){
        reduced_.lb[j]=-INFINITY;implied_bounds_.push_back({j,-1,i});
      }
    }
  }
  // A retained affine singleton equality fixes one coordinate exactly. A
  // box side implied by this equality has no independent domain restriction.
  // Keeping it as a positive-slack barrier row creates an empty interior when
  // the equality fixes the coordinate onto that side. Remove only sides with
  // an exact binary64 root proof, retaining the equality and the original x0.
  // The full original audit and omitted-bound residual remain authoritative;
  // omitted bound multipliers lift to zero, with no multiplier refit required.
  if(remove_affine_equality_implied_bounds)for(int e=0;e<(int)reduced_.equalities.size();++e) {
    const auto& q=reduced_.equalities[e];
    if(!q.signed_square.empty())continue;
    try {
      if(!hessian(q).empty())continue;
      int variable=-1,count=0;double coefficient=0.0;
      q.for_each_linear_nonzero([&](int j,double a) {
        if(a!=0.0){variable=j;coefficient=a;++count;}
      });
      double root=0;
      if(count!=1||!exact_affine_root(coefficient,q.constant,root))continue;
      // Do not alter the historical exact-fixed-box substitution.
      if(reduced_.lb[variable]==reduced_.ub[variable])continue;
      if(std::isfinite(reduced_.lb[variable])&&root>=reduced_.lb[variable]) {
        reduced_.lb[variable]=-INFINITY;
        equality_implied_bounds_.push_back({variable,-1,eq_origins_[e]});
      }
      if(std::isfinite(reduced_.ub[variable])&&root<=reduced_.ub[variable]) {
        reduced_.ub[variable]=INFINITY;
        equality_implied_bounds_.push_back({variable,1,eq_origins_[e]});
      }
    } catch(const std::runtime_error&) {++stats_.arithmetic_rejected;}
  }
  // An equality h=0 retains both signs. Use its radial positive sign only
  // for the exact implication proof, without changing the retained equation.
  if(remove_equality_implied_bounds)for(int e=0;e<(int)reduced_.equalities.size();++e) {
    const auto& q=reduced_.equalities[e];
    if(q.constant==0||!std::isfinite(q.constant)||!no_linear(q)||!q.signed_square.empty()||
        !p.equalities[eq_origins_[e]].signed_square.empty())continue;
    try {
      const auto H=hessian(q);if(H.empty())continue;
      const double sign=q.constant<0?1:-1,constant=sign*q.constant;
      bool diagonal=true;
      for(auto entry:H)if(entry.first.first!=entry.first.second||!(sign*entry.second>0)) {
        diagonal=false;break;
      }
      if(!diagonal)continue;
      std::vector<std::array<int,3>> proven;
      for(auto entry:H) {
        const int j=entry.first.first;const double diagonal_value=sign*entry.second,a=diagonal_value*.5;
        if(2*a!=diagonal_value||reduced_.lb[j]==reduced_.ub[j])continue;
        if(reduced_.ub[j]>0&&radial_implies_bound(a,reduced_.ub[j],constant))
          proven.push_back({j,1,eq_origins_[e]});
        if(reduced_.lb[j]<0&&radial_implies_bound(a,reduced_.lb[j],constant))
          proven.push_back({j,-1,eq_origins_[e]});
      }
      // A failed arithmetic proof cannot leave a partly omitted row domain.
      for(auto side:proven) {
        if(side[1]>0)reduced_.ub[side[0]]=INFINITY;
        else reduced_.lb[side[0]]=-INFINITY;
        equality_implied_bounds_.push_back(side);
      }
    } catch(const std::runtime_error&) {++stats_.arithmetic_rejected;}
  }
  stats_.implied_bound_sides=implied_bounds_.size()+equality_implied_bounds_.size();
  reduced_.validate();
  stats_.applied=stats_.implied_bound_sides>0||reduced_.n!=p.n||reduced_.equalities.size()!=p.equalities.size()||reduced_.inequalities.size()!=p.inequalities.size();
  if(stats_.applied)stats_.reason="coefficient-identities-nonzero-domain-exact-sums";
  build_audit_plan();
  stats_.preparation_seconds=std::chrono::duration<double>(Clock::now()-start).count();
}

OmittedBoundResidual ConstraintPresolve::omitted_bound_residual(const Vec& point) const {
  if(point.size()!=std::size_t(reduced_.n))
    throw std::invalid_argument("rank-presolve: omitted-bound point dimension mismatch");
  for(double value:point)if(!std::isfinite(value))
    throw std::invalid_argument("rank-presolve: nonfinite omitted-bound point");
  OmittedBoundResidual residual;L maximum=0,sum=0;
  for(auto side:equality_implied_bounds_) {
    const int j=side[0];
    const L violation=side[1]<0?L(original_.lb[lower_origins_[j]])-point[j]:
        L(point[j])-original_.ub[upper_origins_[j]];
    const L positive=std::max(0.L,violation);
    maximum=std::max(maximum,positive);sum+=positive;++residual.sides;
  }
  residual.maximum=static_cast<double>(maximum);residual.sum=sum;
  return residual;
}

void ConstraintPresolve::complete_affine_fixed_duals(Result& r) const {
  if(affine_bounds_.empty()||r.x.size()!=std::size_t(reduced_.n)||
     r.certificate_equality_multipliers.size()!=reduced_.equalities.size()||
     r.certificate_inequality_multipliers.size()!=reduced_.inequalities.size()||
     r.certificate_lower_multipliers.size()!=r.x.size()||r.certificate_upper_multipliers.size()!=r.x.size())return;
  detail::DirectOriginalAuditView valid{reduced_,r};
  if(!valid.validate(false))return;
  std::vector<int> fixed;
  for(int j=0;j<reduced_.n;++j)if((affine_lower_sources_[j]>=0||affine_upper_sources_[j]>=0)&&
      std::isfinite(reduced_.lb[j])&&reduced_.lb[j]==reduced_.ub[j])fixed.push_back(j);
  if(fixed.empty())return;
  // Workspace eliminates exact fixed coordinates and exports zero bound
  // multipliers for them. Complete their two-sided adjoint before mapping
  // the artificial side back to its original inequality. This is the same
  // nonnegative decomposition used by the original fixed-bound audit.
  std::vector<L> g(reduced_.n,0);gradient(reduced_.objective,r.x,1,g);
  for(int e=0;e<(int)reduced_.equalities.size();++e)
    gradient(reduced_.equalities[e],r.x,r.certificate_equality_multipliers[e],g);
  for(int i=0;i<(int)reduced_.inequalities.size();++i)
    gradient(reduced_.inequalities[i],r.x,r.certificate_inequality_multipliers[i],g);
  for(int j:fixed)if(std::isfinite(g[j])) {
    r.certificate_lower_multipliers[j]=double(std::max(L(0),g[j]));
    r.certificate_upper_multipliers[j]=double(std::max(L(0),-g[j]));
  }
}

void ConstraintPresolve::lift_checkpoint(Result& r) const {
  const auto saved=stats_;
  try {lift(r);}catch(...){stats_=saved;throw;}
  stats_=saved;
}

void ConstraintPresolve::lift(Result& r) const {
  ++stats_.audit_lift_calls;
  if(r.x.size()!=(std::size_t)reduced_.n||r.certificate_equality_multipliers.size()!=eq_origins_.size()||
     r.certificate_inequality_multipliers.size()!=ineq_origins_.size()||
     r.certificate_lower_multipliers.size()!=r.x.size()||r.certificate_upper_multipliers.size()!=r.x.size())
    throw std::runtime_error("rank-presolve: incomplete reduced state");
  complete_affine_fixed_duals(r);
  Vec x(original_.n),eq(original_.equalities.size(),0),in(original_.inequalities.size(),0);
  Vec lo(original_.n,0),up(original_.n,0);
  for(int j=0;j<original_.n;++j)x[j]=r.x[variable_map_[j]];
  for(int e=0;e<(int)eq_origins_.size();++e)eq[eq_origins_[e]]=r.certificate_equality_multipliers[e];
  for(int i=0;i<(int)ineq_origins_.size();++i)in[ineq_origins_[i]]=r.certificate_inequality_multipliers[i];
  for(int k=0;k<reduced_.n;++k){
    if(affine_lower_sources_[k]<0)lo[lower_origins_[k]]=r.certificate_lower_multipliers[k];
    if(affine_upper_sources_[k]<0)up[upper_origins_[k]]=r.certificate_upper_multipliers[k];
  }
  for(const auto& b:affine_bounds_) {
    const double multiplier=b.side<0?r.certificate_lower_multipliers[b.variable]:r.certificate_upper_multipliers[b.variable];
    in[b.inequality]=multiplier/std::abs(b.coefficient);
    if(!std::isfinite(in[b.inequality]))throw std::runtime_error("rank-presolve: affine-bound multiplier overflow");
  }
  std::vector<L> g(original_.n,0);
  for(int id:lift_function_ids_) {
    if(id==0)gradient(original_.objective,x,1,g);
    else if(id<=int(eq.size()))gradient(original_.equalities[id-1],x,eq[id-1],g);
    else {const int i=id-1-eq.size();gradient(original_.inequalities[i],x,in[i],g);}
  }
  for(int j=0;j<original_.n;++j)g[j]+=L(up[j])-lo[j];
  for(int id:lift_order_){const auto& e=leaves_[id];
    const auto& q1=original_.equalities[e.e1];const auto& q2=original_.equalities[e.e2];
    const L a=derivative(q1,x,e.u),b=derivative(q2,x,e.u),c=derivative(q1,x,e.v),d=derivative(q2,x,e.v);
    const L det=a*d-b*c,mag=std::abs(a*d)+std::abs(b*c);
    if(!std::isfinite(det)||std::abs(det)<=64*std::numeric_limits<L>::epsilon()*mag)
      throw std::runtime_error("rank-presolve: multiplier lift singular at a zero-voltage trial");
    const L rhs1=-g[e.u],rhs2=-g[e.v];
    eq[e.e1]=double((rhs1*d-b*rhs2)/det);eq[e.e2]=double((a*rhs2-rhs1*c)/det);
    if(!std::isfinite(eq[e.e1])||!std::isfinite(eq[e.e2]))throw std::runtime_error("rank-presolve: nonfinite multiplier reconstruction");
    gradient(q1,x,eq[e.e1],g);gradient(q2,x,eq[e.e2],g);
  }
  if(r.termination_scaling_available){
    Vec es(eq.size(),1),is(in.size(),1),sl(in.size(),0);
    for(int e=0;e<(int)eq_origins_.size();++e)es[eq_origins_[e]]=r.termination_equality_scales[e];
    for(int i=0;i<(int)in.size();++i){const int k=ineq_groups_[i];
      if(k>=0){is[i]=r.termination_inequality_scales[k];sl[i]=r.termination_inequality_slacks[k];}
      else sl[i]=std::max(0.0,-original_.inequalities[i].value(x));
    }
    r.termination_equality_scales=std::move(es);r.termination_inequality_scales=std::move(is);r.termination_inequality_slacks=std::move(sl);
  }
  r.x=std::move(x);r.certificate_equality_multipliers=std::move(eq);r.certificate_inequality_multipliers=std::move(in);
  r.certificate_lower_multipliers=std::move(lo);r.certificate_upper_multipliers=std::move(up);
}

bool ConstraintPresolve::audit_full(const Settings& st,Result& r,detail::OriginalSignedResidual* signed_out) const {
  ++stats_.audit_full_calls;
  if(signed_out)*signed_out={};
  detail::DirectOriginalAuditView check{reduced_,r};
  if(!check.validate(st.termination_test=="ipopt") ||
      (r.termination_scaling_available && (r.termination_equality_scales.size()!=eq_origins_.size() ||
       r.termination_inequality_scales.size()!=ineq_origins_.size() || r.termination_inequality_slacks.size()!=ineq_origins_.size()))){
    r.termination_audit_valid=false;r.acceptable_point=false;return false;}
  // Audit the actual original input, then retain the reduced coordinate vectors
  // expected by the running Workspace. All scalar stop decisions are ORIGINAL.
  ++stats_.audit_result_copies;
  Result full=r;
  try{lift(full);}catch(const std::runtime_error&){r.termination_audit_valid=false;r.acceptable_point=false;return false;}
  Settings original_settings=st;original_settings.rank_presolve_context=nullptr;
  detail::OriginalSignedResidual residual;
  stats_.audit_value_evaluations+=1+original_.equalities.size()+original_.inequalities.size();
  stats_.audit_gradient_evaluations+=1+original_.equalities.size()+original_.inequalities.size();
  const bool strict=detail::audit_original_termination(original_,original_settings,full,nullptr,true,signed_out?&residual:nullptr);
  if(signed_out&&residual.valid){
    signed_out->dual.assign(reduced_.n,0);std::vector<L> g(reduced_.n,0);
    for(int j=0;j<original_.n;++j)g[variable_map_[j]]+=L(residual.dual[j]);
    for(int j=0;j<reduced_.n;++j)signed_out->dual[j]=double(g[j]);
    for(int e:eq_origins_)signed_out->equality.push_back(residual.equality[e]);
    for(int i:ineq_origins_)signed_out->inequality.push_back(residual.inequality[i]);
    signed_out->valid=true;
  }
  full.x=std::move(r.x);
  full.certificate_equality_multipliers=std::move(r.certificate_equality_multipliers);
  full.certificate_inequality_multipliers=std::move(r.certificate_inequality_multipliers);
  full.certificate_lower_multipliers=std::move(r.certificate_lower_multipliers);
  full.certificate_upper_multipliers=std::move(r.certificate_upper_multipliers);
  full.termination_equality_scales=std::move(r.termination_equality_scales);
  full.termination_inequality_scales=std::move(r.termination_inequality_scales);
  full.termination_inequality_slacks=std::move(r.termination_inequality_slacks);
  r=std::move(full);return strict;
}

// Arithmetic recipes retain the original order, including diagonal factors.
// Substitution is used only for coordinates, NEVER to combine coefficients in
// an audit. Sharing a value requires exact recipe equality, not a hash alone.
// Algebraic duplicates with different floating-point recipes are evaluated
// separately to preserve the old original certificate even under cancellation.
namespace {
std::vector<std::uint64_t> audit_recipe(const QuadraticFunction& q,const std::vector<int>& map) {
  std::vector<std::uint64_t> v;
  auto bits=[](double x){std::uint64_t b;std::memcpy(&b,&x,sizeof b);return b;};
  v.push_back(bits(q.constant));
  q.for_each_linear_nonzero([&](int j,double x){v.push_back(1);v.push_back(map[j]);v.push_back(bits(x));});
  v.push_back(2);
  for(auto e:q.hessian.entries){v.push_back(e.i==e.j?3:4);v.push_back(map[e.i]);v.push_back(map[e.j]);v.push_back(bits(e.v));}
  return v;
}
std::size_t recipe_hash(const std::vector<std::uint64_t>& v){
  std::size_t h=1469598103934665603ULL;
  for(auto b:v){h^=b;h*=1099511628211ULL;}return h;
}
}
void ConstraintPresolve::build_audit_plan() {
  const auto& p=original_;
  audit_eq_sources_.assign(p.equalities.size(),-1);
  audit_leaf_sources_.assign(p.equalities.size(),-1);
  for(int e=0;e<(int)eq_origins_.size();++e)audit_eq_sources_[eq_origins_[e]]=e;
  for(int i=0;i<(int)leaves_.size();++i){audit_leaf_sources_[leaves_[i].e1]=2*i;audit_leaf_sources_[leaves_[i].e2]=2*i+1;}
  audit_lower_sources_.assign(p.n,-1);audit_upper_sources_.assign(p.n,-1);
  for(int k=0;k<reduced_.n;++k){
    if(affine_lower_sources_[k]<0)audit_lower_sources_[lower_origins_[k]]=k;
    if(affine_upper_sources_[k]<0)audit_upper_sources_[upper_origins_[k]]=k;
  }
  auto function=[&](int id)->const QuadraticFunction&{
    if(id==0)return p.objective;
    if(id<=int(p.equalities.size()))return p.equalities[id-1];
    return p.inequalities[id-1-p.equalities.size()];
  };
  const int count=1+p.equalities.size()+p.inequalities.size();
  audit_value_sources_.resize(count);audit_value_slots_.assign(count,-1);
  std::unordered_map<std::size_t,std::vector<int>> buckets;
  for(int id=0;id<count;++id){
    const auto recipe=audit_recipe(function(id),variable_map_);
    auto& bucket=buckets[recipe_hash(recipe)];int origin=id;
    for(int prior:bucket)if(recipe==audit_recipe(function(prior),variable_map_)){origin=prior;break;}
    audit_value_sources_[id]=origin;
    if(origin==id)bucket.push_back(id);
    else {
      if(audit_value_slots_[origin]<0)audit_value_slots_[origin]=audit_value_slot_count_++;
      audit_value_slots_[id]=audit_value_slots_[origin];++stats_.audit_shared_values;
    }
  }
  // All algebraic duplicate groups are compiled into shared monomial programs.
  // Reordered formulas reuse the monomial products, then sum them in the old
  // order. This removes repeated duplicate coefficient scans without replacing
  // the precise original audit by a differently-rounded reduced-polynomial sum.
  std::vector<int> eq_count(eq_origins_.size(),0),in_count(ineq_origins_.size(),0);
  for(int g:eq_groups_)if(g>=0)++eq_count[g];
  for(int g:ineq_groups_)if(g>=0)++in_count[g];
  std::vector<unsigned char> eq_variants(eq_count.size(),0),in_variants(in_count.size(),0);
  for(int e=0;e<(int)eq_groups_.size();++e){const int g=eq_groups_[e];
    if(g>=0&&audit_value_sources_[1+e]!=audit_value_sources_[1+eq_origins_[g]])eq_variants[g]=1;}
  for(int i=0;i<(int)ineq_groups_.size();++i){const int g=ineq_groups_[i];
    if(g>=0&&audit_value_sources_[1+p.equalities.size()+i]!=audit_value_sources_[1+p.equalities.size()+ineq_origins_[g]])in_variants[g]=1;}
  audit_program_ids_.assign(count,-1);
  std::map<std::tuple<int,int,int,std::uint64_t>,int> terms;
  auto program=[&](int id){
    const int origin=audit_value_sources_[id];
    if(audit_program_ids_[origin]>=0){audit_program_ids_[id]=audit_program_ids_[origin];return;}
    const auto& q=function(origin);AuditProgram prog;prog.constant=q.constant;
    auto term=[&](int kind,int i,int j,double coefficient){
      std::uint64_t b;std::memcpy(&b,&coefficient,sizeof b);
      auto key=std::make_tuple(kind,i,j,b);auto it=terms.find(key);int index;
      if(it==terms.end()){index=audit_terms_.size();audit_terms_.push_back({i,j,kind,coefficient});terms.emplace(key,index);}
      else index=it->second;
      prog.terms.push_back(index);
    };
    q.for_each_linear_nonzero([&](int j,double a){term(0,variable_map_[j],-1,a);});
    for(auto e:q.hessian.entries)term(e.i==e.j?1:2,variable_map_[e.i],variable_map_[e.j],e.v);
    audit_program_ids_[id]=audit_program_ids_[origin]=audit_programs_.size();audit_programs_.push_back(std::move(prog));
  };
  for(int e=0;e<(int)p.equalities.size();++e){const int g=eq_groups_[e];
    if(g>=0&&eq_count[g]>1){
      if(eq_variants[g])program(1+e);
      if(eq_origins_[g]!=e)++stats_.audit_duplicate_rows_covered;
    }
  }
  for(int i=0;i<(int)p.inequalities.size();++i){const int g=ineq_groups_[i];
    if(g>=0&&in_count[g]>1){
      const int id=1+p.equalities.size()+i;
      if(in_variants[g])program(id);
      if(ineq_origins_[g]!=i){++stats_.audit_duplicate_rows_covered;
        if(audit_value_sources_[id]!=audit_value_sources_[1+p.equalities.size()+ineq_origins_[g]])++stats_.audit_recipe_exceptions;}

    }
  }
  // Values can be shared across equality/inequality families. Propagate the
  // compiled program to all aliases of its first evaluation.
  for(int id=0;id<count;++id)if(audit_program_ids_[audit_value_sources_[id]]>=0)
    audit_program_ids_[id]=audit_program_ids_[audit_value_sources_[id]];
  stats_.audit_unique_terms=audit_terms_.size();
}

struct ConstraintPresolve::AuditView {
  const ConstraintPresolve& c;
  Result& r;
  Vec leaf_multipliers,original_x;
  std::vector<L> values,dual_pullback,term_values;
  std::vector<unsigned char> term_ready;
  AuditView(const ConstraintPresolve& c_,Result& r_):c(c_),r(r_),leaf_multipliers(2*c.leaves_.size(),0),values(c.audit_value_slot_count_,0),term_values(c.audit_terms_.size(),0),term_ready(c.audit_terms_.size(),0){}
  double x(int j)const{return original_x[j];}
  double equality(std::size_t e)const {
    const int k=c.audit_eq_sources_[e];if(k>=0)return r.certificate_equality_multipliers[k];
    const int leaf=c.audit_leaf_sources_[e];return leaf>=0?leaf_multipliers[leaf]:0;
  }
  double inequality(std::size_t i)const {
    const int source=c.affine_inequality_sources_[i];
    if(source>=0) {
      const auto& b=c.affine_bounds_[source];
      const double multiplier=b.side<0?r.certificate_lower_multipliers[b.variable]:r.certificate_upper_multipliers[b.variable];
      return multiplier/std::abs(b.coefficient);
    }
    const int k=c.ineq_groups_[i];return k>=0&&c.ineq_origins_[k]==(int)i?r.certificate_inequality_multipliers[k]:0;
  }
  double lower(int j)const{const int k=c.audit_lower_sources_[j];return k>=0?r.certificate_lower_multipliers[k]:0;}
  double upper(int j)const{const int k=c.audit_upper_sources_[j];return k>=0?r.certificate_upper_multipliers[k]:0;}
  double equality_scale(std::size_t e)const{const int k=c.audit_eq_sources_[e];return k>=0?r.termination_equality_scales[k]:1;}
  double inequality_scale(std::size_t i)const{const int k=c.ineq_groups_[i];return k>=0?r.termination_inequality_scales[k]:1;}
  double slack(std::size_t i)const {
    const int k=c.ineq_groups_[i];if(k>=0)return r.termination_inequality_slacks[k];
    // Removed constant rows have a zero multiplier. Match legacy double slack
    // reconstruction exactly; do not invent an interior slack for retained rows.
    const auto& q=c.original_.inequalities[i];double linear=0,quadratic=0;
    q.for_each_linear_nonzero([&](int j,double a){linear+=a*x(j);});
    for(auto e:q.hessian.entries){if(e.i==e.j)quadratic+=e.v*x(e.i)*x(e.i);else quadratic+=2.0*e.v*x(e.i)*x(e.j);}
    return std::max(0.0,-(q.constant+linear+0.5*quadratic));
  }
  void add_gradient(const QuadraticFunction& q,L weight,std::vector<L>& g)const {
    if(weight==0)return;
    q.for_each_linear_nonzero([&](int j,double a){g[j]+=weight*L(a);});
    for(auto e:q.hessian.entries){g[e.i]+=weight*L(e.v)*x(e.j);if(e.i!=e.j)g[e.j]+=weight*L(e.v)*x(e.i);}
  }
  L component(const QuadraticFunction& q,int j)const {
    L v=0;q.for_each_linear_nonzero([&](int k,double a){if(k==j)v+=a;});
    for(auto e:q.hessian.entries){if(e.i==j)v+=L(e.v)*x(e.j);if(e.j==j&&e.i!=j)v+=L(e.v)*x(e.i);}return v;
  }
  bool validate(bool native) {
    detail::DirectOriginalAuditView reduced_view{c.reduced_,r};
    if(!reduced_view.validate(native))return false;
    // The legacy lift accessed scale/slack vectors whenever available. Keep
    // malformed states fail-closed even in project mode, without unchecked reads.
    if(r.termination_scaling_available &&
       (r.termination_equality_scales.size()!=c.eq_origins_.size() ||
        r.termination_inequality_scales.size()!=c.ineq_origins_.size() ||
        r.termination_inequality_slacks.size()!=c.ineq_origins_.size()))return false;
    // A single contiguous primal-coordinate buffer is cheaper than millions
    // of indirect lookups. It is not a Result copy or a multiplier/slack lift.
    original_x.resize(c.original_.n);
    for(int j=0;j<c.original_.n;++j)original_x[j]=r.x[c.variable_map_[j]];
    ++c.stats_.audit_coordinate_expansions;
    if(native)for(int i=0;i<(int)c.ineq_groups_.size();++i)if(c.ineq_groups_[i]<0 && !std::isfinite(slack(i)))return false;
    if(c.leaves_.empty())return true;
    // Small adjoint reconstruction uses exactly the old operations/order. Only
    // the eliminated multipliers are materialized; no original x/slack arrays.
    std::vector<L> g(c.original_.n,0);
    for(int id:c.lift_function_ids_){
      if(id==0)add_gradient(c.original_.objective,1,g);
      else if(id<=int(c.original_.equalities.size()))add_gradient(c.original_.equalities[id-1],equality(id-1),g);
      else {const int i=id-1-c.original_.equalities.size();add_gradient(c.original_.inequalities[i],inequality(i),g);}
    }
    for(int j=0;j<c.original_.n;++j)g[j]+=L(upper(j))-lower(j);
    for(int id:c.lift_order_){const auto& e=c.leaves_[id];
      const auto& q1=c.original_.equalities[e.e1];const auto& q2=c.original_.equalities[e.e2];
      const L a=component(q1,e.u),b=component(q2,e.u),cc=component(q1,e.v),d=component(q2,e.v);
      const L det=a*d-b*cc,mag=std::abs(a*d)+std::abs(b*cc);
      if(!std::isfinite(det)||std::abs(det)<=64*std::numeric_limits<L>::epsilon()*mag)return false;
      const L rhs1=-g[e.u],rhs2=-g[e.v];
      leaf_multipliers[2*id]=double((rhs1*d-b*rhs2)/det);
      leaf_multipliers[2*id+1]=double((a*rhs2-rhs1*cc)/det);
      if(!std::isfinite(leaf_multipliers[2*id])||!std::isfinite(leaf_multipliers[2*id+1]))return false;
      add_gradient(q1,leaf_multipliers[2*id],g);add_gradient(q2,leaf_multipliers[2*id+1],g);
    }
    return true;
  }
  void set_fixed(int,double,double)const{} // original-only changes never enter Workspace
  L evaluate(const QuadraticFunction& q,L weight,std::size_t id,std::vector<L>& g,
      const std::vector<std::vector<int>>*) {
    const int slot=c.audit_value_slots_[id];
    if(c.audit_value_sources_[id]!=(int)id){
      ++c.stats_.audit_value_reuses;
      if(weight!=0){++c.stats_.audit_gradient_evaluations;add_gradient(q,weight,g);}
      return values[slot];
    }
    ++c.stats_.audit_value_evaluations;
    if(weight!=0)++c.stats_.audit_gradient_evaluations;
    L value;
    const int program=c.audit_program_ids_[id];
    if(program>=0){
      ++c.stats_.audit_ordered_sums;
      const auto& instructions=c.audit_programs_[program];value=instructions.constant;
      for(int index:instructions.terms){
        if(!term_ready[index]){
          const auto& term=c.audit_terms_[index];
          term_values[index]=term.kind==0?L(term.coefficient)*r.x[term.i]:
            (term.kind==1?L(.5):L(1))*L(term.coefficient)*r.x[term.i]*r.x[term.j];
          term_ready[index]=1;++c.stats_.audit_term_evaluations;
        } else ++c.stats_.audit_term_reuses;
        value+=term_values[index];
      }
      add_gradient(q,weight,g);
    } else value=detail::original_audit_evaluate(q,[&](int j){return x(j);},weight,g,nullptr,id);
    if(slot>=0)values[slot]=value;
    return value;
  }
  void begin_signed(detail::OriginalSignedResidual& v){
    v.dual.assign(c.reduced_.n,0);v.equality.resize(c.eq_origins_.size());v.inequality.resize(c.ineq_origins_.size());
    dual_pullback.assign(c.reduced_.n,0);
  }
  void write_equality(detail::OriginalSignedResidual& v,std::size_t e,double x)const{
    const int k=c.audit_eq_sources_[e];if(k>=0)v.equality[k]=x;
  }
  void write_inequality(detail::OriginalSignedResidual& v,std::size_t i,double x)const{
    const int k=c.ineq_groups_[i];if(k>=0 && c.ineq_origins_[k]==(int)i)v.inequality[k]=x;
  }
  void write_dual(detail::OriginalSignedResidual&,int j,double x){dual_pullback[c.variable_map_[j]]+=L(x);}
  void end_signed(detail::OriginalSignedResidual& v)const{
    for(int j=0;j<c.reduced_.n;++j)v.dual[j]=double(dual_pullback[j]);
    v.valid=true;
  }
};

namespace {
bool identical_audit_fields(const Result& a,const Result& b){
  auto equal=[](double x,double y){return x==y||(std::isnan(x)&&std::isnan(y));};
  // Includes every decision input/output and objective-history field. Integers
  // are kept exact; no tolerance can conceal a changed strict/acceptable test.
#define AUDIT_SAME(field) if(!equal(a.field,b.field))return false
  AUDIT_SAME(termination_scaling_available);AUDIT_SAME(termination_audit_valid);
  AUDIT_SAME(acceptable_point);AUDIT_SAME(acceptable_convergence);AUDIT_SAME(ipopt_desired_convergence);
  AUDIT_SAME(termination_square_problem);AUDIT_SAME(fixed_multipliers_reconstructed);
  AUDIT_SAME(objective);AUDIT_SAME(feasibility.equality_inf);AUDIT_SAME(feasibility.inequality_inf);AUDIT_SAME(feasibility.bound_inf);AUDIT_SAME(feasibility.total_inf);
  AUDIT_SAME(dual_infeasibility_inf);AUDIT_SAME(complementarity_inf);AUDIT_SAME(model_stationarity_inf);
  AUDIT_SAME(raw_kkt_error);AUDIT_SAME(acceptable_error);AUDIT_SAME(scaled_nlp_error);AUDIT_SAME(original_normalized_nlp_error);
  AUDIT_SAME(termination_internal_dual_inf);AUDIT_SAME(termination_internal_primal_inf);AUDIT_SAME(termination_internal_compl_inf);
  AUDIT_SAME(termination_dual_inf);AUDIT_SAME(termination_compl_inf);
  AUDIT_SAME(termination_sd);AUDIT_SAME(termination_sc);AUDIT_SAME(dual_error_scaling);AUDIT_SAME(complementarity_error_scaling);
  AUDIT_SAME(constraint_multiplier_l1);AUDIT_SAME(bound_slack_multiplier_l1);
  AUDIT_SAME(ipopt_dual_multiplier_dimension);AUDIT_SAME(ipopt_complementarity_multiplier_dimension);
  AUDIT_SAME(ipopt_y_c_dimension);AUDIT_SAME(ipopt_y_d_dimension);AUDIT_SAME(ipopt_z_L_dimension);AUDIT_SAME(ipopt_z_U_dimension);AUDIT_SAME(ipopt_v_L_dimension);AUDIT_SAME(ipopt_v_U_dimension);
  AUDIT_SAME(termination_previous_objective);AUDIT_SAME(termination_current_objective);AUDIT_SAME(termination_objective_iteration);AUDIT_SAME(termination_objective_change);
#undef AUDIT_SAME
  return a.termination_test==b.termination_test && a.x==b.x &&
    a.certificate_equality_multipliers==b.certificate_equality_multipliers && a.certificate_inequality_multipliers==b.certificate_inequality_multipliers &&
    a.certificate_lower_multipliers==b.certificate_lower_multipliers && a.certificate_upper_multipliers==b.certificate_upper_multipliers &&
    a.termination_equality_scales==b.termination_equality_scales && a.termination_inequality_scales==b.termination_inequality_scales &&
    a.termination_inequality_slacks==b.termination_inequality_slacks;
}
}
bool ConstraintPresolve::audit(const Settings& st,Result& r,detail::OriginalSignedResidual* signed_out)const {
  if(st.dca_rank_audit_mode!="full"&&st.dca_rank_audit_mode!="mapped"&&st.dca_rank_audit_mode!="check")
    throw std::invalid_argument("dca-rank-audit-mode must be full, mapped or check");
  const auto begin=Clock::now();++stats_.audit_calls;stats_.audit_mode=st.dca_rank_audit_mode;
  complete_affine_fixed_duals(r);
  bool strict=false;
  if(st.dca_rank_audit_mode=="full")strict=audit_full(st,r,signed_out);
  else if(st.dca_rank_audit_mode=="check"){
    ++stats_.audit_check_calls;++stats_.audit_result_copies;
    Result reference=r;detail::OriginalSignedResidual reference_residual;
    const bool old=audit_full(st,reference,signed_out?&reference_residual:nullptr);
    AuditView view(*this,r);++stats_.audit_mapped_calls;
    strict=detail::audit_original_termination_impl(original_,st,r,view,nullptr,true,signed_out);
    if(old!=strict || (reference.termination_audit_valid!=r.termination_audit_valid) ||
        (r.termination_audit_valid && !identical_audit_fields(reference,r)) ||
        (signed_out && r.termination_audit_valid && (reference_residual.valid!=signed_out->valid||
          reference_residual.dual!=signed_out->dual||reference_residual.equality!=signed_out->equality||reference_residual.inequality!=signed_out->inequality)))
      throw std::runtime_error("rank-presolve: mapped audit differs from full original audit (check mode)");
  } else {
    AuditView view(*this,r);++stats_.audit_mapped_calls;
    strict=detail::audit_original_termination_impl(original_,st,r,view,nullptr,true,signed_out);
  }
  stats_.audit_seconds+=std::chrono::duration<double>(Clock::now()-begin).count();return strict;
}
bool rank_presolve_audit(const ConstraintPresolve* p,const QCQPProblem&,const Settings& s,Result& r,detail::OriginalSignedResidual* v){return p->audit(s,r,v);}

std::string ConstraintPresolve::report_json() const {
  std::ostringstream os;os<<std::setprecision(17);
  os<<"{\"schema\":\"exact-rank-presolve-v1\",\"applied\":"<<(changed()?"true":"false")
    <<",\"original_variables\":"<<original_.n<<",\"reduced_variables\":"<<reduced_.n
    <<",\"original_equalities\":"<<original_.equalities.size()<<",\"reduced_equalities\":"<<reduced_.equalities.size()
    <<",\"original_inequalities\":"<<original_.inequalities.size()<<",\"reduced_inequalities\":"<<reduced_.inequalities.size()
    <<",\"leaf_candidates\":"<<stats_.leaf_candidates<<",\"leaf_pairs\":"<<stats_.leaf_pairs
    <<",\"nonzero_rejected\":"<<stats_.nonzero_rejected<<",\"start_rejected\":"<<stats_.start_rejected
    <<",\"arithmetic_rejected\":"<<stats_.arithmetic_rejected<<",\"duplicate_equalities\":"<<stats_.duplicate_equalities
    <<",\"zero_equalities\":"<<stats_.zero_equalities<<",\"duplicate_inequalities\":"<<stats_.duplicate_inequalities
    <<",\"constant_inequalities\":"<<stats_.constant_inequalities<<",\"merged_bound_sides\":"<<stats_.duplicate_bounds
    <<",\"implied_bound_sides\":"<<stats_.implied_bound_sides
    <<",\"affine_singleton_bound_sides\":"<<affine_bounds_.size()
    <<",\"equality_implied_bound_sides\":"<<equality_implied_bounds_.size()
    <<",\"initial_lift_error\":0,\"preparation_seconds\":"<<stats_.preparation_seconds<<",\"leaves\":[";
  bool comma=false;for(auto e:leaves_){if(comma)os<<',';comma=true;
    os<<"{\"child\":["<<e.u<<','<<e.v<<"],\"parent\":["<<e.a<<','<<e.b<<"],\"equalities\":["<<e.e1<<','<<e.e2
      <<"],\"nonzero_inequality\":"<<e.nonzero_inequality<<",\"coefficients\":["<<e.r1<<','<<e.t1<<','<<e.r2<<','<<e.t2<<"]}";}
  os<<"],\"implied_bounds\":[";comma=false;
  for(auto e:implied_bounds_){if(comma)os<<',';comma=true;os<<'['<<e[0]<<','<<e[1]<<','<<e[2]<<']';}
  os<<"],\"equality_implied_bounds\":[";comma=false;
  for(auto e:equality_implied_bounds_){if(comma)os<<',';comma=true;os<<'['<<e[0]<<','<<e[1]<<','<<e[2]<<']';}
  os<<"],\"affine_singleton_bounds\":[";comma=false;
  for(const auto& b:affine_bounds_){if(comma)os<<',';comma=true;
    os<<"{\"variable\":"<<b.variable<<",\"side\":"<<b.side<<",\"original_inequality\":"<<b.inequality
      <<",\"coefficient\":"<<b.coefficient<<",\"bound\":"<<b.bound<<'}';}
  os<<"]";
  auto array=[&](const char* name,const std::vector<int>& data){os<<",\""<<name<<"\":[";bool first=true;
    for(int v:data){if(!first)os<<',';first=false;os<<v;}os<<']';};
  array("variable_map",variable_map_);array("equality_origins",eq_origins_);
  array("inequality_origins",ineq_origins_);array("inequality_groups",ineq_groups_);
  array("lower_bound_origins",lower_origins_);array("upper_bound_origins",upper_origins_);
  os<<"}";return os.str();
}
} // namespace lazyipdca::qcqp

#include "lazyipdca_qcqp/solver.hpp"
#include <fstream>
#include <iostream>
namespace lazyipdca::qcqp {
Result solve_with_rank_presolve(const QCQPProblem& p,const Settings& st) {
  if((st.dca_engine!="filter"&&st.dca_engine!="majorants")||
     (st.outer_algorithm!="lazy-dca"&&st.outer_algorithm!="auto"))
    throw std::invalid_argument("rank-presolve requires the filter or majorants Lazy-DCA engine");
  if(!st.dca_endpoint_start_file.empty())
    throw std::invalid_argument("rank-presolve requires a cold start; original endpoint files cannot be reused implicitly");
  const auto all_begin=Clock::now();
  const double started_absolute=std::chrono::duration<double>(all_begin.time_since_epoch()).count();
  auto attempt=[&](const Settings& st,bool enable_exact_fixed)->std::optional<Result> {
  const auto begin=Clock::now();
  const bool equality_bounds=st.dca_rank_implied_bounds&&st.dca_engine=="majorants"&&
      (st.factorial_active()?st.factorial_globalization=="hs":
       (st.convexification_study=="HH"||st.convexification_study=="H0"||
        st.convexification_study=="HS"||st.convexification_study=="SH"));
  const bool affine_equality_bounds=st.dca_rank_implied_bounds&&
      st.dca_engine=="majorants"&&st.factorial_r3_controls();
  ConstraintPresolve reduction(p,st.dca_rank_implied_bounds,equality_bounds,
      enable_exact_fixed,affine_equality_bounds);
  if(enable_exact_fixed&&reduction.affine_bounds().empty())return std::nullopt;
  const double preparation=std::chrono::duration<double>(Clock::now()-begin).count();
  if(enable_exact_fixed&&std::isfinite(st.time_limit_seconds)&&preparation>=st.time_limit_seconds)return std::nullopt;
  if(!st.dca_rank_presolve_report_file.empty()){
    std::ofstream f(st.dca_rank_presolve_report_file);
    if(!f)throw std::runtime_error("cannot create rank-presolve report: "+st.dca_rank_presolve_report_file);
    f<<reduction.report_json()<<'\n';
    if(!f)throw std::runtime_error("cannot write rank-presolve report");
  }
  if(st.live_display_level!=0){const auto& s=reduction.statistics();
    std::cout<<"[DCA-RANK-START] applied="<<s.applied<<" leaves="<<s.leaf_pairs
      <<" variables="<<p.n<<"->"<<s.reduced_variables
      <<" equalities="<<p.equalities.size()<<"->"<<s.reduced_equalities
      <<" inequalities="<<p.inequalities.size()<<"->"<<s.reduced_inequalities
      <<" audit_mode="<<st.dca_rank_audit_mode<<" audit_shared_values="<<s.audit_shared_values<<" audit_ordered_variant_rows="<<s.audit_recipe_exceptions
      <<" implied_bound_sides="<<s.implied_bound_sides<<" exact_x0=1 arithmetic_rejected="<<s.arithmetic_rejected<<" seconds="<<preparation<<'\n';
    if(!reduction.affine_bounds().empty())std::cout<<"[DCA-AFFINE-BOUNDS] tightened_sides="<<reduction.affine_bounds().size()
      <<" fixed_coordinates="<<std::count_if(reduction.affine_bounds().begin(),reduction.affine_bounds().end(),[&](const auto& b){
        return reduction.reduced().lb[b.variable]==reduction.reduced().ub[b.variable];})
      <<" original_x0_preserved=1 dual_mapping=original-singleton-inequality\n";
  }
  Settings inner=st;inner.dca_rank_presolve=false;
  inner.rank_presolve_context=reduction.changed()?&reduction:nullptr;
  if(std::isfinite(st.time_limit_seconds))inner.time_limit_seconds=std::max(1e-6,st.time_limit_seconds-preparation);
  Result r=Solver(inner).solve(reduction.changed()?reduction.reduced():p);
  const auto post=Clock::now();
  r.rank_presolve=reduction.statistics();r.rank_presolve.preparation_seconds=preparation;
  if(reduction.changed()) {
    const Status status=r.status;
    try {
      reduction.lift(r);
      Settings original_st=st;original_st.rank_presolve_context=nullptr;
      ++reduction.stats_.audit_final_full_calls;
      bool strict=detail::audit_original_termination(p,original_st,r);
      // The inner solver deliberately cannot fit ORIGINAL multipliers while
      // an exact coordinate/row reduction is installed. Once lift has rebuilt
      // every original row and bound, apply the same fixed-primal rescue as
      // the unreduced path, inside the ORIGINAL solve's remaining deadline.
      // A duplicate row alone must not silently disable this certificate.
      const double remaining=st.time_limit_seconds-
          std::chrono::duration<double>(Clock::now()-begin).count();
      const bool fitted=!strict&&detail::terminal_original_multiplier_fit(
          p,original_st,r,std::min(5.,remaining));
      if(fitted)strict=true;
      r.rank_presolve.postsolve_valid=r.termination_audit_valid;
      if(!fitted&&((status==Status::Solved&&!strict)||
         ((status==Status::SolvedAcceptable||status==Status::SolvedAcceptableOnFailure)&&!r.acceptable_point))){
        r.status=Status::NumericalError;r.message="rank-presolve: original postsolve certificate failed";
      } else if(status==Status::Solved)r.message="original QCQP KKT certificate satisfied after exact coefficient reduction and multiplier reconstruction";
      if(!fitted)r.termination_multiplier_source="rank-presolve-adjoint-original-extended-audit";
    } catch(const std::exception& e) {
      // Never export a reduced vector as if it were an original endpoint.
      r.status=Status::NumericalError;r.message=e.what();r.rank_presolve.postsolve_valid=false;
      if(r.x.size()==std::size_t(reduction.reduced().n)) {
        Vec lifted(p.n);for(int j=0;j<p.n;++j)lifted[j]=r.x[reduction.variable_map()[j]];r.x=std::move(lifted);
      }
      if(r.x.size()!=std::size_t(p.n))r.x=p.x0;
      r.objective=p.objective.value(r.x);r.feasibility=feasibility(p,r.x);
      r.certificate_equality_multipliers.assign(p.equalities.size(),0);
      r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
      r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);
      r.termination_scaling_available=false;
      r.termination_audit_valid=false;r.acceptable_point=false;r.acceptable_convergence=false;
      r.dual_infeasibility_inf=r.scaled_nlp_error=r.raw_kkt_error=INFINITY;
    }
  } else r.rank_presolve.postsolve_valid=r.termination_audit_valid;
  {const bool valid=r.rank_presolve.postsolve_valid;
   r.rank_presolve=reduction.statistics();r.rank_presolve.postsolve_valid=valid;
   r.rank_presolve.preparation_seconds=preparation;}
  r.rank_presolve.postsolve_seconds=std::chrono::duration<double>(Clock::now()-post).count();
  r.lazy_newton.cost_seconds[(int)NewtonCost::Preparation]+=preparation;
  r.lazy_newton.cost_seconds[(int)NewtonCost::Certificate]+=r.rank_presolve.postsolve_seconds;
  r.elapsed_seconds=std::chrono::duration<double>(Clock::now()-begin).count();
  r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
  if(st.live_display_level!=0)std::cout<<"[DCA-RANK-POSTSOLVE] original_audit="<<r.termination_audit_valid
      <<" status="<<status_string(r.status)<<" primal="<<r.feasibility.total_inf
      <<" dual="<<r.dual_infeasibility_inf<<" compl="<<r.complementarity_inf<<'\n';
  if(st.live_display_level!=0){const auto& a=r.rank_presolve;
    std::cout<<"[DCA-RANK-AUDIT] mode="<<a.audit_mode<<" calls="<<a.audit_calls<<" mapped="<<a.audit_mapped_calls
      <<" full="<<a.audit_full_calls<<" values="<<a.audit_value_evaluations<<" reused="<<a.audit_value_reuses
      <<" ordered_sums="<<a.audit_ordered_sums<<" term_reuses="<<a.audit_term_reuses
      <<" result_copies="<<a.audit_result_copies<<" lift_calls="<<a.audit_lift_calls<<" final_full="<<a.audit_final_full_calls
      <<" seconds="<<a.audit_seconds<<'\n';}
  return r;
  };
  // The first attempt is exactly the historical presolve and numerical path.
  // In particular a strict result is returned before probing new fixed sides.
  Result primary=*attempt(st,false);
  const auto first_budget=detail::rank_structural_retry_budget(st,primary,
      std::chrono::duration<double>(Clock::now()-all_begin).count(),started_absolute);
  if(!first_budget.allowed)return primary;
  Settings retry=st;
  retry.time_limit_seconds=first_budget.seconds;
  retry.max_outer_iterations=first_budget.outer;
  retry.restoration_max_iterations=first_budget.restoration;
  retry.restoration_shared_iteration_budget=true;
  retry.restoration_deadline_seconds=first_budget.restoration_deadline;
  // Keep the caller's recovery permission: automatic startup requires it.
  // A shared zero iteration allowance suppresses actual Phase-I work without
  // changing the startup policy or reopening a restoration budget.
  if(st.live_display_level!=0)std::cout<<"[DCA-STRUCTURAL-RETRY] event=probe primary_status="
    <<status_string(primary.status)<<" configuration="<<st.convexification_study
    <<" same_configuration=1 original_frozen_start=1 hidden_reference_fallback=0"
    <<" primary_outer="<<primary.outer_iterations<<" remaining_outer="<<first_budget.outer
    <<" primary_restoration="<<primary.restoration_iterations<<" remaining_restoration="<<first_budget.restoration
    <<" remaining_seconds="<<first_budget.seconds<<" restoration_deadline="<<first_budget.restoration_deadline<<'\n';
  auto recovered=attempt(retry,true);
  if(!recovered) {
    // Even a refused proof consumes real preparation time, never new tours.
    const double total=std::chrono::duration<double>(Clock::now()-all_begin).count();
    const double overhead=std::max(0.0,total-primary.elapsed_seconds);
    primary.elapsed_seconds=total;primary.lazy_newton.solver_wall_seconds=total;
    primary.lazy_newton.cost_seconds[(int)NewtonCost::Preparation]+=overhead;
    return primary;
  }
  Result r=std::move(*recovered);
  detail::merge_rank_retry_work(r,primary);
  r.elapsed_seconds=std::chrono::duration<double>(Clock::now()-all_begin).count();
  r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
  if(st.live_display_level!=0)std::cout<<"[DCA-STRUCTURAL-RETRY] event=finished primary_status="
    <<status_string(primary.status)<<" final_status="<<status_string(r.status)
    <<" same_configuration=1 original_frozen_start=1 original_postsolve_audit="<<r.rank_presolve.postsolve_valid
    <<" cumulative_outer="<<r.outer_iterations<<" outer_limit="<<st.max_outer_iterations
    <<" cumulative_restoration="<<r.restoration_iterations<<" restoration_limit="<<st.restoration_max_iterations
    <<" cumulative_seconds="<<r.elapsed_seconds<<" time_limit="<<st.time_limit_seconds<<'\n';
  return r;
}
} // namespace lazyipdca::qcqp
