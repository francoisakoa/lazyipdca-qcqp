#include <stdexcept>
#include <iostream>
#include "../src/atomic_newton.cpp"
using namespace lazyipdca::qcqp;
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
QCQPProblem problem(){
  QCQPProblem p;p.n=4;p.name="g14_cache_nonlinear_reduction";
  p.lb={-2,-2,-2,0.25};p.ub={2,2,2,0.25};p.x0={0.2,0.3,0.04,0.25};
  p.objective.linear={0,0,0,3};p.objective.hessian.n=4;
  p.objective.hessian.entries={{0,0,2},{1,1,4},{2,2,3}};
  QuadraticFunction q;q.linear={0,0,1,0};q.hessian.n=4;q.hessian.entries={{0,0,-2}};
  p.equalities.push_back(q);q.linear=Vec(4,0);q.hessian.entries={{0,0,2},{1,1,1}};q.constant=-1;
  p.inequalities.push_back(q);return p;
}
void same_matrix(const DcaAggregate::Preparation& a,const DcaAggregate::Preparation& b){
  require(a.hessian==b.hessian && a.weak_border==b.weak_border && a.dimension==b.dimension,
      "cache differs from fresh original Hessian/layout");
  require(a.unshifted.size()==b.unshifted.size(),"matrix length differs");
  for(std::size_t i=0;i<a.unshifted.size();++i){const auto x=a.unshifted[i],y=b.unshifted[i];
    require(x.i==y.i && x.j==y.j && x.value==y.value,"matrix entries/order differ");}
}
int main(){try{
  Settings st;st.linear_backend=st.dca_aggregate_backend="native";st.dca_aggregate=true;
  st.lazy_newton_lazy_constraints=false;st.lazy_newton_reduction=true;st.live_display_level=0;
  auto p=problem();LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;DcaAggregate a(st);
  DcaAggregate::Preparation cached,fresh;
  auto compare=[&](){a.prepare(w,stats,cached,true,1);a.prepare(w,stats,fresh,false,1);same_matrix(cached,fresh);};
  compare();const int builds=stats.aggregate_preparation_builds;
  a.prepare(w,stats,cached,true,1);require(stats.aggregate_preparation_builds==builds && stats.aggregate_preparation_hits==1,"same-anchor cache miss");
  int invalidations=0;
  auto changed=[&](){compare();require(stats.aggregate_preparation_invalidations==++invalidations,"changed state not invalidated");};
  w.x[0]+=0.05;changed();
  w.lambda[0]+=0.7;changed();
  w.z[0]+=0.25;changed();
  w.s[0]*=0.7;changed();
  w.z[0]=1e6*w.s[0];changed();
  require(cached.weak_border[0]>=0,"strong row did not create border");
  w.active[0]=0;changed();require(cached.weak_border[0]<0,"inactive row left a border");
  w.active[0]=1;changed();
  w.objective.model_active=true;changed();require(!w.objective.model_active,"original geometry not restored");
  DcaAggregate::Preparation next_transaction;
  const int before=stats.aggregate_preparation_builds;
  a.prepare(w,stats,next_transaction,true,2);require(stats.aggregate_preparation_builds==before+1,"cache leaked to next transaction");
  st.max_outer_iterations=6;st.dca_aggregate_strict_probe_stop=false;
  st.dca_aggregate_preparation_cache=false;auto r0=solve_filter_dca(p,st);
  st.dca_aggregate_preparation_cache=true;auto r1=solve_filter_dca(p,st);
  require(r0.status==r1.status && r0.x==r1.x &&
      r0.certificate_equality_multipliers==r1.certificate_equality_multipliers &&
      r0.certificate_inequality_multipliers==r1.certificate_inequality_multipliers &&
      r0.certificate_lower_multipliers==r1.certificate_lower_multipliers &&
      r0.certificate_upper_multipliers==r1.certificate_upper_multipliers,
      "cache modified nonlinear solve / multipliers");
  std::cout<<"PASS exact matrix, nonlinear geometry, lambda/z/slack/active/model invalidation, next-anchor lifetime, solve parity; invalidations="<<invalidations<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
