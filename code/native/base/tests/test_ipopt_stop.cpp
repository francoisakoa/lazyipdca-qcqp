#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/dca_filter.hpp"
#include <stdexcept>
#include <iostream>
using namespace lazyipdca::qcqp;
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(){
  Settings s;s.termination_test="ipopt";s.stationarity_tolerance=1e-6;
  s.feasibility_tolerance=1e-4;s.dual_infeasibility_tolerance=1;s.complementarity_tolerance=1e-4;
  s.acceptable_tolerance=1e-6;s.acceptable_dual_tolerance=1e10;
  s.acceptable_constraint_tolerance=1e-2;s.acceptable_complementarity_tolerance=1e-2;
  QCQPProblem p;p.n=2;p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};p.x0={0,0};
  p.objective.linear={-100,.05};p.objective.hessian.n=2;
  QuadraticFunction eq;eq.linear={1,0};eq.hessian.n=2;p.equalities={eq};
  Result r;r.x=p.x0;r.certificate_equality_multipliers={100};r.certificate_lower_multipliers={0,0};r.certificate_upper_multipliers={0,0};
  r.termination_scaling_available=true;r.termination_objective_scale=1e-5;r.termination_equality_scales={1e-3};
  check(detail::audit_original_termination(p,s,r),"scaled strict point rejected by raw dual");
  check(r.dual_infeasibility_inf==.05&&std::abs(r.scaled_nlp_error-5e-7)<1e-20,"two-level scaling incorrect");
  s.dual_infeasibility_tolerance=.01;check(!detail::audit_original_termination(p,s,r),"absolute raw dual bypassed");
  s.dual_infeasibility_tolerance=1;r.termination_objective_scale=1;
  check(!detail::audit_original_termination(p,s,r),"raw dual below 1 falsely implies success");
  r.termination_objective_scale=1e-5;r.termination_equality_scales={1e-8};
  detail::audit_original_termination(p,s,r);check(std::abs(r.termination_sd-1000)<1e-10,"scaled multipliers not used");
  // Slack complementarity cannot be replaced by multiplier*g(x).
  QuadraticFunction in;in.linear={0,0};in.hessian.n=2;p.inequalities={in};
  r.certificate_inequality_multipliers={1};r.termination_inequality_scales={1};r.termination_inequality_slacks={.5};
  check(!detail::audit_original_termination(p,s,r)&&r.complementarity_inf==0&&r.termination_compl_inf==.5,"actual slack complementarity ignored");
  r.termination_inequality_slacks={0};check(detail::audit_original_termination(p,s,r),"zero complementarity rejected");
  r.termination_equality_scales[0]=NAN;check(!detail::audit_original_termination(p,s,r)&&!r.termination_audit_valid,"NaN scaling accepted");
  r.termination_equality_scales[0]=1e-8;
  s.stationarity_tolerance=1e-14;
  r.outer_iterations=1;detail::audit_original_termination(p,s,r);
  check(r.acceptable_point,"acceptable scaled point rejected");
  const double saved=r.termination_objective_change;
  detail::audit_original_termination(p,s,r);check(r.termination_objective_change==saved,"reaudit fabricated objective progress");
  // Every criterion, not just the dual guard, is mandatory.
  IpoptStopSnapshot v;v.free_variables=2;v.equality_dimension=1;
  v.dual_dimension=1;v.compl_dimension=0;v.internal_dual=5e-7;v.internal_primal=0;v.internal_compl=0;
  v.unscaled_dual=.05;v.unscaled_primal=0;v.unscaled_compl=0;v.objective_change=0;
  check(ipopt_stop_decision(v,s).acceptable,"eligible acceptable point rejected");
  v.objective_change=1;s.acceptable_objective_change_tolerance=.01;
  check(!ipopt_stop_decision(v,s).acceptable,"objective-change gate bypassed");
  v.internal_dual=1;v.unscaled_dual=1;v.objective_change=0;
  check(!ipopt_stop_decision(v,s).acceptable,"constant objective replaced NLP error");
  // Specific 3.14.11 square-problem behavior is explicit, never n_reduced==me.
  v.free_variables=1;v.equality_dimension=1;
  check(ipopt_stop_decision(v,s).desired,"bundled Ipopt square-problem rule missing");
  detail::AcceptableCounter c;
  check(!c.observe(0,true,2,true)&&c.observe(1,true,2,true),"native initial-iterate counting incorrect");
  check(c.observe(1,true,2,true)&&c.count==2,"same iteration counted twice");
  check(!c.observe(2,false,2,true)&&c.count==0,"ineligible point failed to reset streak");
  check(!c.observe(3,true,0,true),"acceptable-iter=0 did not disable stopping");
  // Changing stopping ceilings must not change the first two DCA steps.
  QCQPProblem small;small.n=2;small.lb={-3,-3};small.ub={3,3};small.x0={.3,.7};
  small.objective.linear={-1,-2};small.objective.hessian.n=2;
  small.objective.hessian.entries={{0,0,1},{1,1,4}};
  QuadraticFunction ce;ce.linear={1,1};ce.constant=-1;ce.hessian.n=2;small.equalities={ce};
  Settings a;a.termination_test="ipopt";a.acceptable_iterations=0;a.stationarity_tolerance=1e-30;
  a.feasibility_tolerance=1e-4;a.dual_infeasibility_tolerance=1;a.complementarity_tolerance=1e-4;
  a.max_outer_iterations=2;a.live_display_level=0;a.dca_control_tolerance=1e-6;
  Settings b=a;b.stationarity_tolerance=1e-40;b.feasibility_tolerance=100;
  b.dual_infeasibility_tolerance=100;b.complementarity_tolerance=100;
  const auto first=solve_filter_dca(small,a),second=solve_filter_dca(small,b);
  check(first.outer_iterations==2&&second.outer_iterations==2,"controller test did not take two steps");
  check(first.x==second.x&&first.certificate_equality_multipliers==second.certificate_equality_multipliers&&
        first.kkt_factorizations==second.kkt_factorizations,"stop settings changed DCA direction/controller");
  std::cout<<"ipopt_stop: two-level scaling, raw ceilings, slacks, objective, square rule, counter, controller isolation PASS\n";
}
