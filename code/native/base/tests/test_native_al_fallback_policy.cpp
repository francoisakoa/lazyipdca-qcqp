#include "lazyipdca_qcqp/native_al_fallback_policy.hpp"
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
QCQPProblem product() {
  QCQPProblem p;p.n=3;p.name="anonymous";p.x0={.2,.8,.16};p.lb={0,0,0};p.ub={1,1,1};
  p.objective.hessian.n=3;p.objective.linear={0,0,-1};
  QuadraticFunction h;h.hessian.n=3;h.linear={0,0,1};h.hessian.entries={{0,1,-1}};
  p.equalities.push_back(h);p.validate();return p;
}
Settings settings() {Settings s;s.linear_backend="mumps";s.linear_metis=true;s.time_limit_seconds=100;return s;}
Result certified(const QCQPProblem& p) {
  Result r;r.status=Status::Solved;r.x=p.x0;r.termination_audit_valid=true;
  r.certificate_equality_multipliers.assign(p.equalities.size(),0);
  r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
  r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);
  r.feasibility.total_inf=r.dual_infeasibility_inf=r.complementarity_inf=1e-8;return r;
}
void structural_policy() {
  auto p=product();auto s=settings();
  require(detail::native_al_fallback_eligibility(p,s).eligible,"native coefficient domain refused");
  const auto frozen=p.x0;p.name="different family label";p.x0={.7,.3,.21};
  require(detail::native_al_fallback_eligibility(p,s).eligible,"selection depends on name/start");
  require(frozen==Vec({.2,.8,.16}),"frozen start mutated");
  for(const char* mode:{"off","HH","HS","SH","SS","SS_legacy"}) {
    s.convexification_study=mode;
    require(detail::native_al_fallback_eligibility(p,s).eligible,"variant name changed AL eligibility");
  }
  p.objective.hessian.entries={{0,0,1e-300}};
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"tiny genuine objective curvature ignored");
  p=product();QuadraticFunction q;q.hessian.n=3;q.hessian.entries={{0,0,2}};p.inequalities.push_back(q);
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"quadratic source IQ admitted");
  p=product();p.equalities[0].hessian.entries={{0,1,0.}};
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"stored zero interpreted as qEQ");
  p=product();p.equalities[0].signed_square={{0,1}};
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"signed square admitted");
}
void explicit_settings() {
  const auto p=product();auto s=settings();s.native_al_fallback="off";
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"off ignored");
  s=settings();s.native_augmented_repair=true;
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"explicit nativeAL recursed");
  s=settings();s.product_definition_presolve=true;
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"explicit product presolve overridden");
  s=settings();s.power2_scaling=true;
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"explicit scaling overridden");
  s=settings();s.termination_test="ipopt";
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"alternate termination overridden");
  s=settings();s.linear_metis=false;
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"METIS requirement ignored");
  s=settings();s.inner_backend="ipopt";
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"external optimization backend admitted");
  s=settings();s.accepted_checkpoint_callback=[](const QCQPProblem&,Result&,const ConstraintPresolve*,int,double){};
  require(!detail::native_al_fallback_eligibility(p,s).eligible,"custom callback silently skipped");
  s.accepted_checkpoint_builtin=true;
  require(detail::native_al_fallback_eligibility(p,s).eligible,"built-in observer rejected");
  s=settings();s.native_al_fallback="invalid";bool threw=false;
  try {detail::native_al_fallback_eligibility(p,s);}catch(const std::invalid_argument&){threw=true;}
  require(threw,"invalid policy accepted");
}
void strict_selection_and_memory() {
  const auto p=product();auto s=settings();auto r=certified(p);
  require(detail::native_al_fallback_choose_candidate(p,s,r,99),"strict candidate refused inside budget");
  require(!detail::native_al_fallback_choose_candidate(p,s,r,100),"deadline candidate accepted");
  require(!detail::native_al_fallback_choose_candidate(p,s,r,101),"over-budget candidate accepted");
  r.status=Status::SolvedAcceptable;
  require(!detail::native_al_fallback_choose_candidate(p,s,r,1),"acceptable status counted strict success");
  r=certified(p);r.dual_infeasibility_inf=1e-5;
  require(!detail::native_al_fallback_choose_candidate(p,s,r,1),"nonKKT endpoint selected");
  r=certified(p);r.certificate_lower_multipliers.pop_back();
  require(!detail::native_al_fallback_choose_candidate(p,s,r,1),"wrong original multiplier dimension accepted");
  r=certified(p);r.certificate_lower_multipliers[0]=-1e-20;
  require(!detail::native_al_fallback_choose_candidate(p,s,r,1),"negative bound multiplier accepted");
  r=certified(p);r.certificate_equality_multipliers[0]=std::numeric_limits<double>::infinity();
  require(!detail::native_al_fallback_choose_candidate(p,s,r,1),"nonfinite equality multiplier accepted");
  r=certified(p);r.feasibility.total_inf=std::numeric_limits<double>::quiet_NaN();
  require(!detail::native_al_fallback_choose_candidate(p,s,r,1),"NaN certificate accepted");
  r=certified(p);s.stationarity_tolerance=1e-9;
  require(!detail::native_al_fallback_choose_candidate(p,s,r,1),"stricter user tolerance weakened");
  require(detail::native_al_fallback_dense_admitted(2364),"safe dense dimension refused");
  require(!detail::native_al_fallback_dense_admitted(2365),"dense dimension above 128MiB admitted");
  require(!detail::native_al_fallback_dense_admitted(1000000000),"dense estimate overflow");
}
}
int main() {
  try {structural_policy();explicit_settings();strict_selection_and_memory();
    std::cout<<"native AL fallback coefficient/settings/strict budget/memory policy PASS\n";return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
