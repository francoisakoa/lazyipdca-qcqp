#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <map>
#include <set>
#include <string>

namespace lazyipdca::qcqp {
// Exact structural elimination of t^2-x^2=0, t>=0, into x*abs(x).
// This helper never fixes an orthant. Unsupported uses leave the input alone.
class NativeSignedSquareReduction {
 public:
  struct Link { int x=-1,t=-1,e=-1; double scale=1; };
  QCQPProblem reduced;
  std::vector<Link> links;
  std::vector<int> map,variables,eqmap;
  std::string reason;
  explicit NativeSignedSquareReduction(const QCQPProblem& original):original_(original) {
    std::vector<Link> candidates;
    for(int e=0;e<int(original.equalities.size());++e) {
      const auto& q=original.equalities[e];
      if(q.constant!=0||!q.signed_square.empty())continue;
      bool linear=false;q.for_each_linear_nonzero([&](int,double a){linear=linear||a!=0;});
      if(linear)continue;
      std::map<int,double> diagonal;bool simple=true;
      for(auto h:q.hessian.entries)if(h.v!=0) {
        if(h.i!=h.j||diagonal.count(h.i)){simple=false;break;}
        diagonal[h.i]=h.v;
      }
      if(!simple||diagonal.size()!=2)continue;
      auto a=diagonal.begin(),b=std::next(a);
      if(a->second!=-b->second)continue;
      // Prefer the positive diagonal as auxiliary; a negative equality scale
      // is allowed when only the other coordinate has the required domain.
      int t=a->second>0?a->first:b->first,x=a->second>0?b->first:a->first;
      auto eligible=[&](int src,int aux){return original.lb[aux]==0&&
          std::isinf(original.ub[aux])&&original.ub[aux]>0&&
          std::isfinite(original.x0[src])&&original.x0[aux]==std::abs(original.x0[src]);};
      if(!eligible(x,t)){std::swap(x,t);if(!eligible(x,t))continue;}
      const double scale=diagonal.at(t)*0.5;
      if(!std::isfinite(scale)||scale==0||scale*2!=diagonal.at(t))continue;
      candidates.push_back({x,t,e,scale});
    }
    if(candidates.empty()){reason="no exact absolute-value link";return;}
    std::set<int> sources,auxiliary,equations;
    for(auto l:candidates) {
      if(!sources.insert(l.x).second||!auxiliary.insert(l.t).second){reason="overlapping absolute links";return;}
      equations.insert(l.e);
    }
    for(auto j:sources)if(auxiliary.count(j)){reason="nested absolute links";return;}
    auto compatible=[&](const QuadraticFunction& q) {
      bool valid=q.signed_square.empty();
      q.for_each_linear_nonzero([&](int j,double a){if(a!=0&&auxiliary.count(j))valid=false;});
      for(auto h:q.hessian.entries)if(h.v!=0) {
        for(auto l:candidates)if(h.i==l.t||h.j==l.t) {
          const int other=h.i==l.t?h.j:h.i;
          if(other!=l.x&&other!=l.t)valid=false;
        }
      }
      return valid;
    };
    if(!compatible(original.objective)){reason="unsupported objective auxiliary use";return;}
    for(int e=0;e<int(original.equalities.size());++e)
      if(!equations.count(e)&&!compatible(original.equalities[e])){reason="unsupported equality auxiliary use";return;}
    for(const auto& q:original.inequalities)if(!compatible(q)){reason="unsupported inequality auxiliary use";return;}
    links=std::move(candidates);map.assign(original.n,-1);
    for(int j=0;j<original.n;++j)if(!auxiliary.count(j)){map[j]=int(variables.size());variables.push_back(j);}
    reduced.name=original.name;reduced.n=int(variables.size());
    for(int j:variables){reduced.x0.push_back(original.x0[j]);reduced.lb.push_back(original.lb[j]);reduced.ub.push_back(original.ub[j]);}
    reduced.objective=reduce(original.objective);
    for(int e=0;e<int(original.equalities.size());++e)if(!equations.count(e)) {
      eqmap.push_back(e);reduced.equalities.push_back(reduce(original.equalities[e]));
    }
    for(const auto& q:original.inequalities)reduced.inequalities.push_back(reduce(q));
    reduced.validate();reason="exact native signed-square reduction";
  }
  bool changed()const{return !links.empty();}
  // Lift primal and certificate multipliers. The caller MUST then perform a
  // fresh audit against original_, and may report Solved only if that passes.
  void lift(Result& r)const {
    if(!changed())return;
    if(r.x.size()!=variables.size()||r.certificate_equality_multipliers.size()!=eqmap.size()||
       r.certificate_inequality_multipliers.size()!=original_.inequalities.size()||
       r.certificate_lower_multipliers.size()!=variables.size()||
       r.certificate_upper_multipliers.size()!=variables.size())
      throw std::runtime_error("native signed-square postsolve certificate dimension mismatch");
    Vec x(original_.n,0),lower(original_.n,0),upper(original_.n,0),lambda(original_.equalities.size(),0);
    for(std::size_t j=0;j<variables.size();++j){const int k=variables[j];x[k]=r.x[j];lower[k]=r.certificate_lower_multipliers[j];upper[k]=r.certificate_upper_multipliers[j];}
    for(std::size_t e=0;e<eqmap.size();++e)lambda[eqmap[e]]=r.certificate_equality_multipliers[e];
    for(const auto& l:links) {
      x[l.t]=std::abs(x[l.x]);long double cross=0,square=0;
      auto accumulate=[&](const QuadraticFunction& q,long double weight){
        for(auto h:q.hessian.entries) {
          if((h.i==l.x&&h.j==l.t)||(h.i==l.t&&h.j==l.x))cross+=weight*h.v;
          else if(h.i==l.t&&h.j==l.t)square+=weight*h.v;
        }
      };
      accumulate(original_.objective,1);
      for(int e:eqmap)accumulate(original_.equalities[e],lambda[e]);
      for(std::size_t i=0;i<original_.inequalities.size();++i)
        accumulate(original_.inequalities[i],r.certificate_inequality_multipliers[i]);
      const int sign=x[l.x]>0?1:(x[l.x]<0?-1:0);
      // grad_t L = cross*x + square*abs(x). Divide symbolically by t,
      // rather than numerically by abs(x), including the exact zero case.
      lambda[l.e]=double(-(cross*sign+square)/(2*static_cast<long double>(l.scale)));
      if(!std::isfinite(lambda[l.e]))throw std::runtime_error("native signed-square multiplier overflow");
    }
    if(r.termination_scaling_available) {
      if(r.termination_equality_scales.size()!=eqmap.size())throw std::runtime_error("native signed-square scaling dimension mismatch");
      Vec scales(original_.equalities.size(),1);
      for(std::size_t e=0;e<eqmap.size();++e)scales[eqmap[e]]=r.termination_equality_scales[e];
      r.termination_equality_scales=std::move(scales);
    }
    r.x=std::move(x);r.certificate_lower_multipliers=std::move(lower);r.certificate_upper_multipliers=std::move(upper);
    r.certificate_equality_multipliers=std::move(lambda);r.termination_audit_valid=false;
    r.termination_multiplier_source="native-signed-square-exact-lift-original-audit-pending";
  }
 private:
  const QCQPProblem& original_;
  QuadraticFunction reduce(const QuadraticFunction& q)const {
    QuadraticFunction out;out.name=q.name;out.constant=q.constant;out.linear.assign(reduced.n,0);out.hessian.n=reduced.n;
    q.for_each_linear_nonzero([&](int j,double a){if(map[j]>=0)out.linear[map[j]]=a;});
    for(auto h:q.hessian.entries) {
      if(map[h.i]>=0&&map[h.j]>=0){out.hessian.entries.push_back({map[h.i],map[h.j],h.v});continue;}
      if(h.v==0)continue;
      bool transferred=false;
      for(const auto& l:links) {
        if(h.i==l.t&&h.j==l.t){out.hessian.entries.push_back({map[l.x],map[l.x],h.v});transferred=true;break;}
        if((h.i==l.x&&h.j==l.t)||(h.i==l.t&&h.j==l.x)){out.signed_square.push_back({map[l.x],h.v});transferred=true;break;}
      }
      if(!transferred)throw std::runtime_error("unvalidated native signed-square coefficient");
    }
    return out;
  }
};
} // namespace lazyipdca::qcqp
