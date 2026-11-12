#include <stdexcept>
#include <iostream>
#include "../src/atomic_newton.cpp"
using namespace lazyipdca::qcqp;
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
QCQPProblem simple(){
  QCQPProblem p;p.n=2;p.name="g14_strict_rho";p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.x0={.002,.001};p.objective.linear={0,0};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,2},{1,1,3}};return p;
}
struct Output{Result r;DcaFilterStatistics stats;};
Output run(bool cache,bool shortcut){
  auto p=simple();Settings s;s.linear_backend=s.dca_aggregate_backend="native";s.dca_aggregate=true;
  s.lazy_newton_lazy_constraints=false;s.live_display_level=0;
  s.dca_aggregate_preparation_cache=cache;s.dca_aggregate_strict_probe_stop=shortcut;
  s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  Output out;detail::NewtonCostLedger costs(out.r.lazy_newton,Clock::now());Workspace w(p,s,out.r.lazy_newton,costs);
  DcaModels models(w,out.stats);DcaAggregate a(s);a.accepted_rho=.01;
  auto point=dca_original(w,w.x,w.lambda,w.z);detail::OriginalDcaFilter filter(std::max(1.,10*point.theta),s.stationarity_tolerance);
  require(a.attempt(w,models,point,filter,1),"analytical rho contest failed");
  w.export_state(out.r);require(detail::audit_original_termination(p,s,out.r,&w.original_linear_support),"endpoint fails unchanged original strict audit");
  return out;
}
int main(){try{
  const auto uncached=run(false,false),cached=run(true,false),fast=run(true,true);
  require(uncached.r.x==cached.r.x && cached.r.x==fast.r.x,"strict shortcut/cache changed analytical endpoint");
  require(uncached.stats.aggregate_preparation_builds==6 && cached.stats.aggregate_preparation_builds==1 &&
      cached.stats.aggregate_preparation_hits==5,"did not exercise six probes / five hits");
  require(uncached.stats.aggregate_certificates==cached.stats.aggregate_certificates,
      "preparation cache wrongly reused numeric factorization");
  require(fast.stats.aggregate_certificates==1 && fast.stats.aggregate_strict_probe_successes==1 &&
      fast.stats.aggregate_strict_probe_skipped==5,"strict certificate did not skip five nonduplicate probes");
  auto p=simple();Settings s;s.linear_backend=s.dca_aggregate_backend="native";s.dca_aggregate=true;
  s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  s.live_display_level=0;LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(p,s,ns,costs);
  DcaAggregate a(s);DcaFilterStatistics stats;DcaAggregate::Candidate c;
  c.x=w.x;c.s=w.s;c.z=w.z;c.lambda=w.lambda;c.point={};c.point.kkt=0;c.point.normalized_kkt=0;
  const Vec old_x=w.x,old_s=w.s,old_z=w.z,old_l=w.lambda;
  require(!a.strict_candidate(w,c,stats,2),"tiny log indicators incorrectly certified real dual residual");
  require(w.x==old_x && w.s==old_s && w.z==old_z && w.lambda==old_l,"candidate audit modified anchor");
  // An acceptable-only candidate cannot short-circuit the rho search or
  // synthesize the consecutive acceptable-iterate rule.
  c.x={3e-6,0};c.point=dca_original(w,c.x,c.lambda,c.z);
  const int checks_before=stats.aggregate_strict_probe_checks;
  require(!a.strict_candidate(w,c,stats,2) && stats.aggregate_strict_probe_checks==checks_before,
      "acceptable-only point invoked strict shortcut");
  p.n=2;p.lb={-INFINITY,0.25};p.ub={INFINITY,0.25};p.x0={0,0.25};
  p.objective.linear={-1,7};p.objective.hessian.entries={{0,0,2}};
  QuadraticFunction q;q.linear={1,0};q.hessian.n=2;p.inequalities={q};
  s.termination_test="ipopt";s.stationarity_tolerance=1e-8;s.dual_infeasibility_tolerance=1;
  s.feasibility_tolerance=s.complementarity_tolerance=1e-4;
  LazyNewtonStatistics ns2;detail::NewtonCostLedger costs2(ns2,Clock::now());Workspace w2(p,s,ns2,costs2);DcaAggregate a2(s);
  c.x=w2.x;c.lambda=w2.lambda;c.z=w2.z;c.s=w2.s;c.point={};c.point.kkt=0;c.point.normalized_kkt=0;
  c.z[0]=w2.objective.scale/w2.inequalities[0].scale;
  c.s[0]=w2.inequalities[0].scale*0.1;
  require(!a2.strict_candidate(w2,c,stats,3),"actual slack complementarity was ignored");
  c.s[0]=w2.inequalities[0].scale*1e-12;
  require(a2.strict_candidate(w2,c,stats,3),"full native strict certificate with reconstructed fixed multiplier failed");
  c.z[0]*=-1;
  require(!a2.strict_candidate(w2,c,stats,3),"negative inequality multiplier certified");
  require(stats.aggregate_strict_probe_checks==4 && stats.aggregate_strict_probe_failures==3,
      "full rejection gates not exercised");
  std::cout<<"PASS six-to-one preparation, identical factor count with cache, strict 6-to-1 factors, full dual/slack/fixed-bound audit, rollback; skipped="<<fast.stats.aggregate_strict_probe_skipped<<"\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
