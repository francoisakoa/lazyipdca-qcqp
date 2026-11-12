#include <iostream>
#include <stdexcept>
#include "../src/atomic_newton.cpp"
using namespace lazyipdca::qcqp;
void require(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
QCQPProblem small(double curvature=2){
  QCQPProblem p;p.n=2;p.name="g15_finish";p.lb={-INFINITY,0.25};p.ub={INFINITY,0.25};
  p.x0={2e-5,0.25};p.objective.linear={0,3};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,curvature}};return p;
}
int main(){try{
  Settings st;st.linear_backend=st.dca_aggregate_backend="native";st.dca_aggregate=true;
  st.live_display_level=0;st.termination_test="ipopt";st.stationarity_tolerance=1e-8;
  st.feasibility_tolerance=st.complementarity_tolerance=1e-4;st.dual_infeasibility_tolerance=1;
  st.acceptable_objective_change_tolerance=1e100;st.acceptable_tolerance=1e-4;st.acceptable_dual_tolerance=1e-4;st.acceptable_iterations=0;st.max_outer_iterations=5;
  st.dca_strict_finish=true;st.dca_strict_finish_max=2;
  auto p=small();auto r=solve_filter_dca(p,st);
  require(r.status==Status::Solved,"precise finish did not solve analytical QP");
  require(r.dca_filter.strict_finish_attempts==1&&r.dca_filter.strict_finish_acceptances==1&&
      r.dca_filter.strict_finish_solved==1,"finish was unreachable before accepted path / disabled streak");
  require(detail::audit_original_termination(p,st,r),"unchanged full certificate failed");
  require(r.fixed_multipliers_reconstructed==1,"fixed multiplier not reconstructed");
  st.dca_strict_finish=false;auto old=solve_filter_dca(p,st);
  require(old.dca_filter.strict_finish_attempts==0,"disabled flag ignored");
  // Nonlinear equality defect below global control tol but above strict tol:
  // the finishing SOC must use the requested precision and precise RHS.
  auto curved=small();curved.n=3;curved.lb={-INFINITY,-INFINITY,0.25};curved.ub={INFINITY,INFINITY,0.25};
  curved.x0={1+1e-5,(1+1e-5)*(1+1e-5),0.25};
  curved.objective.linear={-1,-1,3};curved.objective.hessian.n=3;
  curved.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction nonlinear;nonlinear.hessian.n=3;nonlinear.linear={0,1,0};nonlinear.hessian.entries={{0,0,-2}};
  curved.equalities={nonlinear};st.dca_strict_finish=true;st.stationarity_tolerance=1e-12;
  auto nonlinear_result=solve_filter_dca(curved,st);
  require(nonlinear_result.status==Status::Solved && nonlinear_result.dca_filter.strict_finish_acceptances>0,
      "nonlinear precise finish did not solve");
  require(nonlinear_result.dca_filter.soc_attempts>0,"strict precision normal correction not exercised");
  st.stationarity_tolerance=1e-8;
  // Failed unshifted inertia must leave the acceptable saddle intact.
  p=small(-2);st.dca_strict_finish=true;LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;DcaModels models(w,stats);DcaAggregate a(st);
  auto point=dca_original(w,w.x,w.lambda,w.z);detail::OriginalDcaFilter f(1,st.dca_control_tolerance);
  auto x=w.x,s=w.s,z=w.z,l=w.lambda;const auto before=point.kkt;
  require(!dca_strict_finish(w,models,a,point,f,st,w.s,1),"uncertified saddle accepted");
  require(x==w.x&&s==w.s&&z==w.z&&l==w.lambda&&point.kkt==before,
      "failed precise finish mutated incumbent");
  require(stats.strict_finish_rejections==1,"rejection counter missing");
  // Cancellation: g=1e16 + 1 - 1e16. The signed vector and scalar
  // certificate must share the exact original extended accumulation stream.
  QCQPProblem c;c.n=1;c.lb={-INFINITY};c.ub={INFINITY};c.x0={0};
  c.objective.linear={1e16};c.objective.hessian.n=1;
  QuadraticFunction q;q.hessian.n=1;q.linear={1};c.equalities={q,q};
  Settings check;Result cr;cr.x={0};cr.certificate_equality_multipliers={1,-1e16};
  cr.certificate_lower_multipliers={0};cr.certificate_upper_multipliers={0};
  detail::OriginalSignedResidual signed_r;
  detail::audit_original_termination(c,check,cr,nullptr,true,&signed_r);
  require(signed_r.valid&&signed_r.dual.size()==1&&signed_r.dual[0]==1&&cr.dual_infeasibility_inf==1,
      "signed original RHS differs from full extended certificate");
  cr.certificate_lower_multipliers[0]=-1;
  require(!detail::audit_original_termination(c,check,cr,nullptr,true,&signed_r)&&!signed_r.valid,
      "invalid audit reused stale signed residual");
  std::cout<<"PASS acceptable->strict without streak, actual finish admission, fixed bound, disabled flag, inertia rollback, signed cancellation and stale output rejection\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
