#pragma once
// Deterministic internal initialization for a coefficient-detected bilinear
// flow structure. This builds a feasible point; it does not certify optimality.
// The caller retains the original frozen start and audits the original model.
#include "lazyipdca_qcqp/model.hpp"
#include <chrono>
#include <set>

namespace lazyipdca::qcqp::detail {
struct NativeBilinearStart {
  bool admitted=false,success=false;
  Vec x;
  std::string reason;
  double initial_feasibility=std::numeric_limits<double>::infinity();
  double final_feasibility=std::numeric_limits<double>::infinity();
  double objective=std::numeric_limits<double>::quiet_NaN();
};
inline NativeBilinearStart native_bilinear_start(const QCQPProblem& p,double absolute_deadline_seconds){
  NativeBilinearStart out;
  auto expired=[&](){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=absolute_deadline_seconds;};
  auto reject=[&](const char* why){out.reason=why;out.success=false;out.x.clear();return out;};
  if(expired())return reject("bilinear-start deadline reached");
  try{p.validate();}catch(const std::exception&){return reject("bilinear-start invalid model");}
  if(p.has_signed_square())return reject("bilinear-start signed squares unsupported");
  auto affine=[](const QuadraticFunction& q){for(auto e:q.hessian.entries)if(e.v!=0)return false;return true;};
  if(!affine(p.objective))return reject("bilinear-start objective not affine");
  for(const auto& q:p.inequalities)if(!affine(q)||q.constant>0)return reject("bilinear-start inequality structure unsupported");
  struct Definition{int z,a,b;};
  std::vector<Definition> definitions;
  std::vector<int> group(p.n,-1),definition(p.n,-1);
  std::vector<bool> nonlinear(p.n,false);
  for(const auto& q:p.equalities)for(auto e:q.hessian.entries)if(e.v){nonlinear[e.i]=true;nonlinear[e.j]=true;}
  Vec base(p.n,0.0);int groups=0;
  for(const auto& q:p.equalities){
    if(expired())return reject("bilinear-start deadline reached");
    if(affine(q)){
      if(q.constant!=-1)return reject("bilinear-start affine equality not unit simplex");
      std::vector<int> members;bool valid=true;
      q.for_each_linear_nonzero([&](int j,double a){if(a!=1||group[j]!=-1||p.lb[j]!=0||p.ub[j]!=1)valid=false;members.push_back(j);});
      if(!valid||members.empty())return reject("bilinear-start overlapping or unsupported simplex");
      for(int j:members){group[j]=groups;base[j]=1.0/members.size();}++groups;
    }else{
      if(q.constant!=0)return reject("bilinear-start product constant nonzero");
      int z=-1,nlin=0;double pivot=0;
      q.for_each_linear_nonzero([&](int j,double a){z=j;pivot=a;++nlin;});
      SymTriplet term;int nh=0;for(auto e:q.hessian.entries)if(e.v){term=e;++nh;}
      if(nlin!=1||nh!=1||term.i==term.j||(pivot!=1&&pivot!=-1)||term.v!=-pivot||nonlinear[z]||definition[z]!=-1)
        return reject("bilinear-start equality not unique positive unit product");
      definition[z]=static_cast<int>(definitions.size());definitions.push_back({z,term.i,term.j});
    }
  }
  if(!groups||definitions.empty())return reject("bilinear-start no simplex/product structure");
  std::vector<int> flow,index(p.n,-1);
  for(int j=0;j<p.n;++j){
    if(!std::isfinite(p.x0[j]))return reject("bilinear-start nonfinite frozen start");
    if(group[j]>=0){if(definition[j]>=0)return reject("bilinear-start simplex also product");}
    else{if(p.lb[j]!=0)return reject("bilinear-start non-simplex lower bound not zero");if(definition[j]<0){index[j]=flow.size();flow.push_back(j);}}
  }
  if(flow.empty())return reject("bilinear-start no flow coordinates");
  for(auto& d:definitions){
    if(group[d.b]>=0)std::swap(d.a,d.b);
    if(group[d.a]<0||index[d.b]<0)return reject("bilinear-start product is not simplex times flow");
  }
  const std::size_t nf=flow.size(),mi=p.inequalities.size();
  if(nf>std::numeric_limits<std::size_t>::max()/std::max<std::size_t>(mi,1)||8.L*nf*(mi+2)>128.L*1024*1024)
    return reject("bilinear-start dense row workspace exceeds 128 MiB");
  // Substitute z=q*y at fixed uniform simplex values. Constant contributions
  // from simplex coordinates remain constants in these private affine rows.
  auto row=[&](const QuadraticFunction& q,Vec& a,double& c){
    a.assign(nf,0);c=q.constant;
    q.for_each_linear_nonzero([&](int j,double v){
      if(index[j]>=0)a[index[j]]+=v;
      else if(definition[j]>=0){const auto& d=definitions[definition[j]];a[index[d.b]]+=v*base[d.a];}
      else c+=v*base[j];
    });
  };
  Vec A;A.reserve(nf*mi);Vec constants(mi),cost,temporary;double objective_constant=0;
  row(p.objective,cost,objective_constant);
  std::vector<std::size_t> active;
  for(std::size_t i=0;i<mi;++i){
    if(expired())return reject("bilinear-start deadline reached");
    row(p.inequalities[i],temporary,constants[i]);
    if(constants[i]>0)return reject("bilinear-start uniform zero flow is infeasible");
    A.insert(A.end(),temporary.begin(),temporary.end());if(constants[i]==0)active.push_back(i);
  }
  Vec upper;upper.reserve(nf);for(int j:flow)upper.push_back(p.ub[j]);
  for(auto d:definitions)if(base[d.a]>0)upper[index[d.b]]=std::min(upper[index[d.b]],p.ub[d.z]/base[d.a]);
  for(double a:cost)if(!std::isfinite(a))return reject("bilinear-start nonfinite substituted objective");
  for(double a:A)if(!std::isfinite(a))return reject("bilinear-start nonfinite substituted row");
  Vec direction(nf,0);
  auto add_pair=[&](std::size_t j,std::size_t k,double ratio){
    const double scale=std::max(1.0,ratio);direction[j]+=1.0/scale;direction[k]+=ratio/scale;
  };
  for(std::size_t j=0;j<nf;++j){
    if(expired())return reject("bilinear-start deadline reached");
    std::vector<std::size_t> positive;
    for(std::size_t r:active)if(A[r*nf+j]>1e-12)positive.push_back(r);
    if(positive.empty()){if(cost[j]<0)direction[j]+=1;continue;}
    // Intersection of negative supports, in ascending original coordinate order.
    for(std::size_t k=j+1;k<nf;++k){
      bool candidate=true;for(std::size_t r:positive)if(A[r*nf+k]>=-1e-12){candidate=false;break;}
      if(!candidate)continue;
      double low=0,high=std::numeric_limits<double>::infinity();bool valid=true;
      for(std::size_t r:active){double a=A[r*nf+j],b=A[r*nf+k];
        if(b<0)low=std::max(low,-a/b);else if(b>0)high=std::min(high,-a/b);else if(a>1e-12){valid=false;break;}}
      if(!valid||low>high+1e-12||high<=0)continue;
      low=std::max(low,0.0);double ratio=std::isfinite(high)?(low+high)/2:low+std::max(1.0,low);
      if(!std::isfinite(ratio)||cost[j]+ratio*cost[k]>=0)continue;
      for(std::size_t r:active)if(A[r*nf+j]+ratio*A[r*nf+k]>1e-9){valid=false;break;}
      if(valid)add_pair(j,k,ratio);
    }
  }
  double alpha=std::numeric_limits<double>::infinity();bool any=false;
  for(std::size_t i=0;i<mi;++i){
    if(expired())return reject("bilinear-start deadline reached");
    double v=0;for(std::size_t j=0;j<nf;++j)v+=A[i*nf+j]*direction[j];
    if(v>1e-10)alpha=std::min(alpha,-constants[i]/v);
  }
  for(std::size_t j=0;j<nf;++j)if(direction[j]>0){any=true;alpha=std::min(alpha,upper[j]/direction[j]);}
  if(!std::isfinite(alpha))alpha=1;
  alpha=std::max(0.0,0.9*alpha);
  out.x=base;
  if(any)for(std::size_t j=0;j<nf;++j)out.x[flow[j]]=alpha*direction[j];
  for(auto d:definitions)out.x[d.z]=out.x[d.a]*out.x[d.b];
  for(double v:out.x)if(!std::isfinite(v))return reject("bilinear-start nonfinite constructed coordinate");
  out.admitted=true;
  out.initial_feasibility=feasibility(p,p.x0).total_inf;
  for(const auto& q:p.equalities)if(!std::isfinite(q.value(out.x)))return reject("bilinear-start nonfinite original equality value");
  for(const auto& q:p.inequalities)if(!std::isfinite(q.value(out.x)))return reject("bilinear-start nonfinite original inequality value");
  out.final_feasibility=feasibility(p,out.x).total_inf;
  out.objective=p.objective.value(out.x);
  if(expired())return reject("bilinear-start deadline reached");
  if(!std::isfinite(out.final_feasibility)||!std::isfinite(out.objective)||out.final_feasibility>1e-8)
    return reject("bilinear-start original physical feasibility audit failed");
  out.success=true;out.reason=any?"uniform simplex plus summed feasible descent rays":"uniform simplex with zero flows";
  return out;
}
} // namespace lazyipdca::qcqp::detail
