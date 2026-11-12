#include "../src/standard_exact_presolve.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
int checks=0;
void require(bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);}
QuadraticFunction q(int n){QuadraticFunction x;x.linear.assign(n,0);x.hessian.n=n;return x;}
QCQPProblem linked() {
  QCQPProblem p;p.n=3;p.name="anonymous";p.x0={.25,.0625,.25};
  p.lb={-1,-INFINITY,0};p.ub={1,INFINITY,INFINITY};
  p.objective=q(3);p.objective.linear[1]=1;
  auto h=q(3);h.hessian.entries={{0,0,-2},{2,2,2}};p.equalities.push_back(h);
  h=q(3);h.linear[1]=1;h.hessian.entries={{0,2,-1}};p.equalities.push_back(h);
  p.validate();return p;
}
Settings settings(){Settings st;st.linear_backend="mumps";st.dca_engine="majorants";st.majorant_startup_policy="auto";st.dca_rank_presolve=true;return st;}
void precedence() {
  const auto p=linked();auto st=settings();
  require(detail::resolve_standard_auto(p,st).enabled,"fixture no longer exercises full-source selection");
  const auto r=detail::resolve_standard_with_exact_presolve(p,st);
  require(r.mode=="native-signed-square"&&!r.enabled,"exact native presolve did not precede R10");
  require(!r.settings.majorant_equality_stabilization&&r.settings.standard_auto_resolved&&r.settings.dca_engine=="majorants", "exact presolve route was not retained");
  require(r.settings.majorant_startup_policy==st.majorant_startup_policy&&r.settings.dca_rank_presolve==st.dca_rank_presolve&&r.settings.lazy_newton_initial_mu==st.lazy_newton_initial_mu,"historical startup changed");
  require(p.x0==Vec({.25,.0625,.25}),"supplied start changed");
  auto renamed=p;renamed.name="arbitrary_new_dataset";
  require(detail::resolve_standard_with_exact_presolve(renamed,st).reason==r.reason,"instance name controls exact presolve");
  st.dca_engine="auto";
  require(detail::resolve_standard_with_exact_presolve(p,st).settings.dca_engine=="majorants","automatic engine missed exact native presolve");
}
void guards() {
  auto p=linked();auto st=settings();
  for(const char* study:{"HH","HS","SH","SS"}) {
    st.convexification_study=study;const auto r=detail::resolve_standard_with_exact_presolve(p,st);
    require(r.enabled&&r.settings.convexification_study==study,"curvature study silently became reference policy");
  }
  st=settings();st.convexification_study="SS_legacy";
  require(detail::resolve_standard_with_exact_presolve(p,st).reason=="SS_legacy-incompatible-preserved","historical scalar control changed");
  st=settings();st.majorant_equality_stabilization=true;st.majorant_equality_stabilization_explicit=true;
  require(detail::resolve_standard_with_exact_presolve(p,st).enabled,"explicit stabilization ON ignored");
  st=settings();st.standard_r10_policy="off";
  require(detail::resolve_standard_with_exact_presolve(p,st).reason=="explicit-standard-policy-off","explicit standard OFF ignored");
  st=settings();st.dca_engine="legacy";
  require(detail::resolve_standard_with_exact_presolve(p,st).reason=="explicit-engine-preserved","explicit legacy engine changed");
  st=settings();p.objective.linear[2]=1;
  require(detail::resolve_standard_with_exact_presolve(p,st).enabled,"unsupported auxiliary use lost standard treatment");
  p=linked();p.x0[2]=.5;
  require(detail::resolve_standard_with_exact_presolve(p,st).enabled,"inconsistent start was projected");
  p=linked();p.ub[2]=2;
  require(detail::resolve_standard_with_exact_presolve(p,st).enabled,"finite auxiliary upper bound eliminated");
  p=linked();p.equalities[0].hessian.entries[0].v=-2.000000000001;
  require(detail::resolve_standard_with_exact_presolve(p,st).enabled,"inexact square identity eliminated");
  p=linked();p.equalities.erase(p.equalities.begin());
  require(detail::resolve_standard_with_exact_presolve(p,st).enabled,"ordinary bilinear equality lost stabilized treatment");
}
}
int main(){try {precedence();guards();std::cout<<"standard_exact_presolve PASS checks="<<checks<<" optimizer_calls=0\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
