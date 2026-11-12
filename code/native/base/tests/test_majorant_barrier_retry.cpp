#include "../src/majorant_barrier_retry.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool b,const char* text){if(!b)throw std::runtime_error(text);}
int main(){try{
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.max_outer_iterations=3000;st.restoration_max_iterations=40;st.time_limit_seconds=1200;
  st.feasibility_tolerance=st.stationarity_tolerance=st.dual_infeasibility_tolerance=
      st.complementarity_tolerance=1e-6;
  QCQPProblem p;p.n=1;p.x0={.5};p.lb={0};p.ub={2};p.objective.hessian.n=1;
  QuadraticFunction h;h.hessian.n=1;h.constant=-1;h.hessian.entries={{0,0,2}};p.equalities={h};
  Result primary;primary.status=Status::InnerFailure;primary.x={1};
  primary.termination_audit_valid=true;primary.feasibility.total_inf=0;
  primary.dual_infeasibility_inf=1;primary.complementarity_inf=1e-7;
  primary.outer_iterations=114;primary.restoration_iterations=18;
  primary.restoration_total_seconds=.5;
  auto budget=detail::majorant_barrier_retry_budget(p,st,primary,10,1000);
  check(budget.allowed&&budget.outer==2886&&budget.restoration==22&&budget.seconds==1190&&
      budget.restoration_deadline==1060,"retry acquired new physical budgets");
  check(p.x0==Vec{.5}&&primary.x==Vec{1},"eligibility changed original or primary point");
  for(Status status:{Status::Solved,Status::SolvedAcceptable,Status::SolvedAcceptableOnFailure,
      Status::MaxOuterIterations,Status::TimeLimit,Status::RestorationFailure,
      Status::NumericalError,Status::TrustRegionFailure}) {
    auto r=primary;r.status=status;
    check(!detail::majorant_barrier_retry_budget(p,st,r,10,1000).allowed,
        "success, exhausted budget or nonadmitted status triggered retry");
  }
  for(Status status:{Status::InnerFailure,Status::FactorizationFailure,Status::LineSearchFailure}) {
    auto r=primary;r.status=status;
    check(detail::majorant_barrier_retry_budget(p,st,r,10,1000).allowed,
        "terminal feasible numerical failure excluded");
  }
  auto blocked=st;blocked.majorant_barrier_retry_consumed=true;
  check(!detail::majorant_barrier_retry_budget(p,blocked,primary,10,1000).allowed,
      "nested or second native retry acquired new allowance");
  blocked=st;blocked.majorant_complete_barrier_gate=true;
  check(!detail::majorant_barrier_retry_budget(p,blocked,primary,10,1000).allowed,
      "complete-gate child recursively reentered recovery");
  auto affine=p;affine.equalities[0].hessian.entries.clear();affine.equalities[0].linear={1};
  check(!detail::majorant_barrier_retry_budget(affine,st,primary,10,1000).allowed,
      "affine equality problem acquired nonlinear barrier retry");
  auto infeasible=primary;infeasible.feasibility.total_inf=1.01e-6;
  check(!detail::majorant_barrier_retry_budget(p,st,infeasible,10,1000).allowed,
      "original-infeasible anchor admitted");
  auto unaudited=primary;unaudited.termination_audit_valid=false;
  check(!detail::majorant_barrier_retry_budget(p,st,unaudited,10,1000).allowed,
      "unaudited point admitted");
  auto exhausted=primary;exhausted.outer_iterations=3000;
  check(!detail::majorant_barrier_retry_budget(p,st,exhausted,10,1000).allowed&&
      !detail::majorant_barrier_retry_budget(p,st,primary,1200,1000).allowed,
      "exhausted outer or time budget reset");
  Result retry=primary;retry.status=Status::Solved;retry.outer_iterations=2000;
  retry.restoration_iterations=20;retry.dual_infeasibility_inf=5e-7;
  check(detail::majorant_barrier_retry_strict(st,retry,budget,1170),"strict within-budget retry rejected");
  auto wrong=retry;wrong.outer_iterations=2887;
  check(!detail::majorant_barrier_retry_strict(st,wrong,budget,1170),"retry overspent outer budget");
  wrong=retry;wrong.restoration_iterations=23;
  check(!detail::majorant_barrier_retry_strict(st,wrong,budget,1170),"retry overspent restoration budget");
  check(!detail::majorant_barrier_retry_strict(st,retry,budget,1200.01),"late certificate accepted");
  wrong=retry;wrong.dual_infeasibility_inf=1.0001e-6;
  check(!detail::majorant_barrier_retry_strict(st,wrong,budget,1170),"original dual tolerance relaxed");
  wrong=retry;wrong.complementarity_inf=1.0001e-6;
  check(!detail::majorant_barrier_retry_strict(st,wrong,budget,1170),"original complementarity tolerance relaxed");
  wrong=retry;wrong.termination_audit_valid=false;
  check(!detail::majorant_barrier_retry_strict(st,wrong,budget,1170),"unaudited retry endpoint accepted");
  std::cout<<"majorant_barrier_retry PASS primary-success-preserved=1 global-outer-time-restoration=1"
      <<" original-frozen-start=1 recursion-guard=1 no-infeasible-retry=1 strict-original-gates=1\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
