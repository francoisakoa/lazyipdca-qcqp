#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
QCQPProblem base(double lower,double upper) {
  QCQPProblem p;p.n=2;p.name="anonymous_affine_bound_fixture";
  p.x0={7,8};p.lb={lower,-10};p.ub={upper,10};
  p.objective.hessian.n=2;p.objective.linear={0,-2};p.objective.hessian.entries={{1,1,1}};
  return p;
}
void row(QCQPProblem& p,double a,double c) {
  QuadraticFunction q;q.hessian.n=2;q.linear={a,0};q.constant=c;p.inequalities.push_back(q);
}
Result endpoint(const QCQPProblem& p,double x,double lower_dual,double upper_dual) {
  Result r;r.x={x,2};r.certificate_equality_multipliers.assign(p.equalities.size(),0);
  r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
  r.certificate_lower_multipliers={lower_dual,0};r.certificate_upper_multipliers={upper_dual,0};
  r.termination_scaling_available=true;r.termination_objective_scale=1;
  r.termination_equality_scales.assign(p.equalities.size(),1);
  r.termination_inequality_scales.assign(p.inequalities.size(),1);
  r.termination_inequality_slacks.assign(p.inequalities.size(),0);return r;
}
void fixed_side(double a,double bound,bool opposite_original) {
  const bool upper=a>0;
  auto p=base(upper?bound:-10,upper?10:bound);
  if(!opposite_original){p.lb[0]=-10;p.ub[0]=10;row(p,-a,a*bound);}
  row(p,a,-a*bound);p.objective.linear[0]=upper?-6:8;
  p.validate();ConstraintPresolve reduced(p,true,false,true);
  check(reduced.reduced().lb[0]==bound&&reduced.reduced().ub[0]==bound,"singleton did not expose fixed coordinate");
  check(reduced.reduced().x0==p.x0,"serialized frozen x0 changed");
  check(reduced.reduced().inequalities.empty(),"converted singleton remained in strict barrier");
  const double lower=upper?0:8,up=upper?6:0;
  auto small=endpoint(reduced.reduced(),bound,lower,up);auto full=small;
  reduced.lift(full);
  check(full.certificate_inequality_multipliers.back()==(upper?6:8)/std::abs(a),"original inequality multiplier incorrectly scaled");
  check(full.certificate_lower_multipliers[0]==0&&full.certificate_upper_multipliers[0]==0,"artificial box multiplier leaked to original side");
  Settings st;st.termination_test="project";st.rank_presolve_context=&reduced;
  st.dca_rank_audit_mode="check";
  check(detail::audit_original_termination(reduced.reduced(),st,small),"mapped/full original KKT audit did not certify analytic solution");
  check(small.certificate_lower_multipliers==Vec({lower,0})&&small.certificate_upper_multipliers==Vec({up,0}),"mapped audit changed working bound duals");
  auto raw=endpoint(reduced.reduced(),bound,0,0);
  check(detail::audit_original_termination(reduced.reduced(),st,raw),"Workspace zero fixed-coordinate duals were not completed before original audit");
  check(raw.certificate_lower_multipliers==Vec({lower,0})&&raw.certificate_upper_multipliers==Vec({up,0}),"wrong fixed-coordinate adjoint completion");
  st.termination_test="ipopt";
  raw=endpoint(reduced.reduced(),bound,0,0);
  check(detail::audit_original_termination(reduced.reduced(),st,raw),"native mapped/full audit failed with converted singleton side");
  st.termination_test="project";
  // Audit failure must remain visible on every original row after conversion.
  full.x[0]=upper?bound+0.25:bound-0.25;Settings direct=st;direct.rank_presolve_context=nullptr;
  check(!detail::audit_original_termination(p,direct,full),"removed original inequality was no longer audited");
}
void strongest_and_redundant() {
  auto p=base(2,10);row(p,2,-8);row(p,4,-8);row(p,8,-16);
  p.objective.linear[0]=-12;ConstraintPresolve reduced(p,true,false,true);
  check(reduced.reduced().ub[0]==2&&reduced.affine_bounds().size()==1,"strongest singleton bound not selected");
  auto r=endpoint(reduced.reduced(),2,0,12);reduced.lift(r);
  check(r.certificate_inequality_multipliers==Vec({0,3,0}),"duplicate/weak singleton multiplier provenance wrong");
  check(r.certificate_upper_multipliers[0]==0,"weaker original upper box kept artificial dual");
  p=base(-10,1);row(p,2,-8);ConstraintPresolve redundant(p,true,false,true);
  check(redundant.affine_bounds().empty()&&redundant.reduced().inequalities.size()==1,"nonfixing redundant singleton changed historical route");
  r=endpoint(redundant.reduced(),1,0,12);redundant.lift(r);
  check(r.certificate_inequality_multipliers[0]==0&&r.certificate_upper_multipliers[0]==12,"original stronger bound lost multiplier");
}
void arithmetic_and_optout() {
  auto p=base(-10,10);row(p,3,-1);ConstraintPresolve nonbinary(p,true,false,true);
  check(nonbinary.reduced().inequalities.size()==1&&nonbinary.reduced().ub[0]==10,"nonrepresentable 1/3 root rounded into model");
  p=base(0,10);row(p,2,0);ConstraintPresolve disabled(p,false,false,true);
  check(disabled.reduced().inequalities.size()==1&&disabled.reduced().ub[0]==10,"implied-bound option was ignored");
  p=base(-10,10);row(p,0,-1);ConstraintPresolve constant(p,true,false,true);
  check(constant.affine_bounds().empty(),"constant inequality treated as a bound");
  p=base(1,10);row(p,std::ldexp(1.0,-900),-std::ldexp(1.0,-900));ConstraintPresolve extreme(p,true,false,true);
  check(extreme.reduced().ub[0]==1,"exact moderate-product root was unnecessarily rejected");
  p=base(-10,10);const double tiny=std::numeric_limits<double>::denorm_min();row(p,3*tiny,-tiny);ConstraintPresolve underflow(p,true,false,true);
  check(underflow.reduced().inequalities.size()==1,"underflow made an inexact affine root look exact");
  p=base(-10,10);row(p,1,0);p.inequalities[0].hessian.entries={{0,0,2}};ConstraintPresolve nonlinear(p,true,false,true);
  check(nonlinear.reduced().inequalities.size()==1,"quadratic row was incorrectly converted to a bound");
  p=base(-10,10);row(p,2,-4);ConstraintPresolve nonfixing(p,true,false,true);
  check(nonfixing.reduced().lb==p.lb&&nonfixing.reduced().ub==p.ub&&
        nonfixing.reduced().inequalities.size()==p.inequalities.size()&&nonfixing.affine_bounds().empty(),
        "nonfixing tighter singleton changed historical bounds or inequalities");
}
}
int main(){try{
  for(double a:{2.0,0.5,-4.0,-0.25})for(double b:{0.0,0.5,-0.5})for(bool original:{false,true})fixed_side(a,b,original);
  strongest_and_redundant();arithmetic_and_optout();
  std::cout<<"affine singleton bounds: exact roots, fixed coordinates, signs, provenance and mapped/full original audits PASS\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
