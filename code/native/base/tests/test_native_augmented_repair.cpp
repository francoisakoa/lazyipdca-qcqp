#include "../src/native_augmented_repair.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace al=lazyipdca::qcqp::detail::native_augmented;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
Settings controls(){Settings s;s.linear_backend="mumps";s.linear_metis=true;s.live_display_level=0;s.verbose=false;s.time_limit_seconds=10;s.max_outer_iterations=200;s.feasibility_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=s.stationarity_tolerance=1e-6;return s;}
QuadraticFunction quad(int n,double c,Vec linear,std::vector<SymTriplet> h){QuadraticFunction q;q.name="analytic";q.constant=c;q.linear=std::move(linear);q.hessian.n=n;q.hessian.entries=std::move(h);return q;}
QCQPProblem empty(int n){QCQPProblem p;p.name="generic_control";p.n=n;p.x0.assign(n,.2);p.lb.assign(n,0);p.ub.assign(n,4);p.objective=quad(n,0,Vec(n,0),{});return p;}
void derivative_and_box(){
 auto s=controls();auto p=empty(2);p.x0={1.3,2.4};p.ub={4,8};p.objective=quad(2,0,{-8,0},{{0,1,1}});p.equalities={quad(2,0,{-2,1},{})};p.inequalities={quad(2,-2,{1,1},{})};p.validate();
 al::NativeAL k(p,s,al::Clock::now(),10);Vec y{.7},z{.4};const double rho=3,eps=1e-5;auto a=k.evaluate(p.x0,y,z,rho,true);double ge=0,he=0;
 for(int j=0;j<2;++j){Vec plus=p.x0,minus=p.x0;plus[j]+=eps;minus[j]-=eps;auto b=k.evaluate(plus,y,z,rho,false),c=k.evaluate(minus,y,z,rho,false);ge=std::max(ge,std::abs((double)((b.value-c.value)/(2*eps))-a.gradient[j]));for(int i=0;i<2;++i)he=std::max(he,std::abs((b.gradient[i]-c.gradient[i])/(2*eps)-a.hessian[i*2+j]));}
 require(ge<1e-7&&he<1e-7,"native PHR gradient/Hessian differs from analytic finite differences");
 auto d=k.direction(p.x0,a);require(dot(a.gradient,d)<0&&k.solves==1,"native MUMPS direction not descent");
 auto box=empty(2);box.x0={.2,.3};box.ub={1,1};box.objective=quad(2,0,{-4,-2},{{0,0,2},{0,1,1},{1,1,2}});box.validate();
 al::NativeAL kb(box,s,al::Clock::now(),10);auto e=kb.evaluate(box.x0,{},{},1,true),unused=e;auto db=kb.box_direction(box.x0,e);(void)unused;
 require(std::abs(box.x0[0]+db[0]-1)<1e-10&&std::abs(box.x0[1]+db[1]-.5)<1e-10,"native bound-face Newton misses analytic box minimizer");
}
class RejectedSolve final:public SymmetricFactorization{
 public:mutable int calls=0;mutable std::vector<Vec> inputs;
 bool factor(int,int,const std::vector<SymmetricTriplet>&,std::string&)override{return true;}
 bool solve(Vec& rhs,std::string&)const override{inputs.push_back(rhs);++calls;if(calls==1){rhs.assign(rhs.size(),999);return false;}for(double& v:rhs)v*=.5;return true;}
 const char* name()const override{return "mutating-rejected-solve-test";}
};
void retry_pristine_rhs(){RejectedSolve f;Vec pristine{2,4},rhs=pristine;std::string message;require(!al::solve_from_pristine_rhs(f,pristine,rhs,message),"injected rejected solve accepted");require(al::solve_from_pristine_rhs(f,pristine,rhs,message),"second injected solve rejected");require(f.inputs[0]==pristine&&f.inputs[1]==pristine&&rhs==Vec({1,2}),"retry used a failed solver's overwritten RHS");}
void stable_phr_value_and_finite_guards(){
 auto p=empty(1);p.inequalities={quad(1,1e-4,{0},{})};auto s=controls();al::NativeAL k(p,s,al::Clock::now(),10);auto e=k.evaluate(p.x0,{},Vec{1e20},1,true);require(e.value>0.999999999999e16L&&e.value<1.000000000001e16L,"large-multiplier PHR value lost rho*g by cancellation");
 auto bad=empty(1);bad.x0={1};bad.equalities={quad(1,0,{0},{{0,0,1e308}})};al::NativeAL b(bad,s,al::Clock::now(),10);bool rejected=false;try{b.evaluate(bad.x0,Vec{0},{},10,true);}catch(const std::exception&){rejected=true;}require(rejected,"nonfinite augmented gradient hidden by projected-gradient norm");
}
void stable_merit_differences(){
 auto s=controls();
 auto p=empty(1);p.objective=quad(1,1e30,{1e-12},{});al::NativeAL k(p,s,al::Clock::now(),10);
 const long double tiny=k.merit_difference(Vec{0},Vec{1},{},{},1);
 require(std::abs(tiny-1e-12L)<1e-27L,"merit difference lost a tiny objective change against a large constant");
 require(p.objective.value(Vec{0})==p.objective.value(Vec{1}),"large-constant regression does not exercise value-subtraction loss");
 p=empty(1);p.inequalities={quad(1,0,{1e-24},{})};al::NativeAL huge(p,s,al::Clock::now(),10);
 const long double large_z=huge.merit_difference(Vec{0},Vec{1},{},Vec{1e20},1);
 require(std::abs(large_z-1e-4L)<1e-19L,"active PHR difference lost z*dg by subtracting nearby multiplier values");
 // Check all PHR branches, including a change across the active threshold.
 p=empty(1);p.inequalities={quad(1,0,{1},{})};al::NativeAL branch(p,s,al::Clock::now(),10);
 for(const auto pair:std::vector<std::pair<double,double>>{{-2,-1},{-1,1},{1,-1},{1,2}}){
   Vec x{pair.first},t{pair.second};const auto diff=branch.merit_difference(x,t,{},Vec{0},2);
   const auto direct=branch.evaluate(t,{},Vec{0},2,false,false).value-branch.evaluate(x,{},Vec{0},2,false,false).value;
   require(std::abs(diff-direct)<1e-18L,"PHR difference branch differs from direct moderate value");
 }
 p=empty(2);p.objective=quad(2,5,{.7,-.2},{{0,0,2},{0,1,-.3},{1,1,4}});
 p.equalities={quad(2,-1,{.3,-.1},{{0,1,.8}})};p.inequalities={quad(2,.4,{-.1,.2},{{0,0,1.4},{0,1,-.9}})};
 al::NativeAL mixed(p,s,al::Clock::now(),10);Vec x{.8,.3},t{.8001,.2997},y{-.7},z{.4};
 const auto direct=mixed.evaluate(t,y,z,3,false,false).value-mixed.evaluate(x,y,z,3,false,false).value;
 const auto difference=mixed.merit_difference(x,t,y,z,3);
 require(std::abs(difference-direct)<2e-15L,"quadratic augmented difference differs from moderate direct values");
 const double eps=1e-5;auto grad=mixed.evaluate(x,y,z,3,false).gradient;
 for(int j=0;j<2;++j){Vec plus=x,minus=x;plus[j]+=eps;minus[j]-=eps;
   auto numerical=(mixed.merit_difference(x,plus,y,z,3)-mixed.merit_difference(x,minus,y,z,3))/(2*eps);
   require(std::abs(numerical-grad[j])<1e-7L,"augmented merit-difference finite difference disagrees with analytic gradient");
 }
 bool refused=false;try{mixed.merit_difference(x,Vec{std::numeric_limits<double>::quiet_NaN(),0},y,z,3);}catch(const std::exception&){refused=true;}
 require(refused,"merit difference concealed a nonfinite trial");
}
void full_original_product_lift(){
 auto p=empty(3);p.x0={.2,.4,1.5};p.lb={0,0,0};p.ub={1,2,2};p.objective=quad(3,0,{0,0,-1},{});p.equalities={quad(3,0,{0,0,1},{{0,1,-1}})};p.validate();const auto initial=p.x0;auto s=controls();s.feasibility_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=s.stationarity_tolerance=1e-9;
 auto r=detail::solve_native_augmented_repair(p,s);require(p.x0==initial,"native repair mutated original start");require(r.status==Status::Solved&&r.termination_audit_valid,"generic product model not original-KKT solved");require(r.x.size()==3&&r.certificate_equality_multipliers.size()==1,"private-dimensional certificate leaked");require(std::abs(r.objective+2)<1e-8&&r.feasibility.total_inf<=1e-9&&r.dual_infeasibility_inf<=1e-9&&r.complementarity_inf<=1e-9,"generic product original raw KKT/optimum failed");require(r.native_augmented.initial_projection&&r.native_augmented.reduced_variables==2,"product projection/reduced dimension not recorded");require(r.native_augmented.accepted_steps<=s.max_outer_iterations,"accepted Newton budget exceeded");
}
void limits_and_admission(){
 auto p=empty(1);p.objective=quad(1,0,{-1},{});p.equalities={quad(1,-1,{0},{{0,0,2}})};p.validate();auto s=controls();s.time_limit_seconds=0;auto t=detail::solve_native_augmented_repair(p,s);require(t.status==Status::TimeLimit&&t.native_augmented.accepted_steps==0&&t.native_augmented.factorizations==0&&t.x.size()==1,"zero global deadline performed native steps or lost original vectors");
 s=controls();s.max_outer_iterations=1;auto r=detail::solve_native_augmented_repair(p,s);require(r.native_augmented.accepted_steps<=1&&r.status!=Status::Solved,"one-step cap not enforced on nonlinear feasibility problem");
 s=controls();s.max_outer_iterations=0;auto z=detail::solve_native_augmented_repair(p,s);require(z.status==Status::MaxOuterIterations&&z.native_augmented.accepted_steps==0&&z.native_augmented.factorizations==0,"zero accepted-step cap ignored");
 s=controls();s.linear_backend="native";auto w=detail::solve_native_augmented_repair(p,s);require(w.status==Status::NumericalError&&!w.native_augmented.admitted&&w.native_augmented.factorizations==0,"cross-backend repair silently admitted");
 auto big=empty(3000);s=controls();auto b=detail::solve_native_augmented_repair(big,s);require(b.status==Status::NumericalError&&!b.native_augmented.admitted&&b.native_augmented.factorizations==0,"dense private Hessian admission cap ignored");
}
int main(){derivative_and_box();retry_pristine_rhs();stable_phr_value_and_finite_guards();stable_merit_differences();full_original_product_lift();limits_and_admission();std::cout<<"NATIVE_AUGMENTED_REPAIR_TESTS_PASS derivatives box RHS-retry stable-PHR stable-merit-difference finite original-dual-lift deadline accepted-step admission\n";}
