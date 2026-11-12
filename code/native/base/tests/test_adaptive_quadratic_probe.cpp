#include "adaptive_equality_probe.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){try {
 int checks=0;auto require=[&](bool good,const char* text){++checks;if(!good)throw std::runtime_error(text);};
 QCQPProblem p;p.name="arbitrary_diagonal_convex_equality_model";p.n=2;
 p.x0={1,1};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
 p.objective.linear={0,0};p.objective.hessian.n=2;p.objective.hessian.entries={{1,1,2}};
 QuadraticFunction h;h.linear={0,1};h.hessian.n=2;h.hessian.entries={{0,0,-2}};p.equalities.push_back(h);
 Settings s;s.linear_backend="mumps";s.dca_engine="majorants";s.standard_r10_policy="auto";s.max_outer_iterations=5000;
 const Vec frozen=p.x0;
 for(const char* mode:{"off","HH","HS","SH","SS"}) {
  s.convexification_study=mode;const auto selected=detail::resolve_standard_auto(p,s);
  require(selected.enabled,"stabilized recovery permission lost");
  require(detail::adaptive_equality_probe_eligible(p,s,selected),"healthy quadratic equality primary not observed");
  require(selected.settings.convexification_study==mode,"variant substituted");
 }
 s.convexification_study="off";const auto selected=detail::resolve_standard_auto(p,s);
 auto renamed=p;renamed.name="a_completely_unrelated_name";
 require(detail::adaptive_equality_probe_eligible(renamed,s,detail::resolve_standard_auto(renamed,s)),"name changes admission");
 require(p.x0==frozen,"supplied start changed");
 auto explicit_settings=s;explicit_settings.majorant_equality_stabilization_explicit=true;
 require(!detail::adaptive_equality_probe_eligible(p,explicit_settings,selected),"explicit stabilization overridden");
 auto disabled=selected;disabled.enabled=false;
 require(!detail::adaptive_equality_probe_eligible(p,s,disabled),"unselected extension observed");
 auto signed_problem=p;signed_problem.objective.hessian.entries={{0,0,-1},{1,1,2}};
 require(!detail::adaptive_equality_probe_eligible(signed_problem,s,detail::resolve_standard_auto(signed_problem,s)),"signed quadratic route changed");
 auto affine=p;affine.objective.hessian.entries.clear();affine.objective.linear={1,0};
 require(detail::adaptive_equality_probe_eligible(affine,s,detail::resolve_standard_auto(affine,s)),"existing affine observation lost");
 Result observed;observed.outer_iterations=6;observed.restoration_iterations=2;
 require(detail::adaptive_remaining_outer(s,observed)==4994,"new outer budget invented");
 require(detail::adaptive_remaining_restoration(s,observed)==598,"new restoration budget invented");
 auto short_budget=s;short_budget.max_outer_iterations=7;
 require(!detail::adaptive_equality_probe_eligible(p,short_budget,selected),"small explicit allowance ignored");
 std::cout<<"PASS adaptive quadratic primary: "<<checks<<" checks; optimizer_calls=0\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
