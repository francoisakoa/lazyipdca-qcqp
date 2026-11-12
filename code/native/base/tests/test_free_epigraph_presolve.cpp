#include "lazyipdca_qcqp/free_epigraph_presolve.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
using Exact=boost::multiprecision::cpp_rational;
namespace {
void check(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
QCQPProblem fixture(){
  QCQPProblem p;p.name="coefficient_leaf_fixture";p.n=2;p.x0={0,-0.0};p.lb={-INFINITY,0};p.ub={INFINITY,INFINITY};
  p.objective.constant=1;p.objective.linear={-2,0};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,2}};
  QuadraticFunction q;q.linear={0,0};q.hessian.n=2;q.hessian.entries={{0,0,2},{1,1,-2.5}};p.inequalities={q};p.validate();return p;
}
Result state(const QCQPProblem& reduced,double y){
  Result r;r.x={y};r.certificate_equality_multipliers.assign(reduced.equalities.size(),0);
  r.certificate_inequality_multipliers.assign(reduced.inequalities.size(),0);
  r.certificate_lower_multipliers.assign(reduced.n,0);r.certificate_upper_multipliers.assign(reduced.n,0);return r;
}
}
int main(){try {
  auto p=fixture();FreeEpigraphPresolve reduction(p);
  check(reduction.changed()&&reduction.reduced().n==1&&reduction.reduced().inequalities.empty(),"exact leaf not removed");
  check(reduction.statistics().initial_point_preserved,"initial point proof unavailable");
  auto initial=state(reduction.reduced(),p.x0[0]);reduction.lift(initial);
  check(std::memcmp(initial.x.data(),p.x0.data(),2*sizeof(double))==0,"signed-zero start was not bit preserved");
  for(double y:{-1e100,-1.0,-1e-100,0.,std::nextafter(1.,2.),1e100}) {
    auto r=state(reduction.reduced(),y);reduction.lift(r);
    check(r.x.size()==2&&r.x[0]==y&&r.x[1]>=0&&std::isfinite(r.x[1]),"invalid full coordinate lift");
    check(Exact(y)*Exact(y)-Exact(1.25)*Exact(r.x[1])*Exact(r.x[1])<=0,"outward leaf lift infeasible in exact binary rational arithmetic");
    check(r.certificate_inequality_multipliers==Vec{0}&&r.certificate_lower_multipliers==Vec({0,0})&&r.certificate_upper_multipliers==Vec({0,0}),"zero dual algebra not preserved");
  }
  auto optimum=state(reduction.reduced(),1);reduction.lift(optimum);
  check(p.objective.value(optimum.x)==0&&2*optimum.x[0]-2==0,"objective/stationarity transfer failed");
  auto scaled=state(reduction.reduced(),1);scaled.termination_scaling_available=true;reduction.lift(scaled);
  check(scaled.termination_inequality_scales==Vec{1}&&scaled.termination_inequality_slacks.size()==1,"original scaling dimensions not restored");
  auto short_state=state(reduction.reduced(),1);short_state.certificate_lower_multipliers.clear();bool refused=false;
  try{reduction.lift(short_state);}catch(const std::runtime_error&){refused=true;}check(refused,"incomplete snapshot was accepted");

  auto reject=[&](QCQPProblem q,const char* reason){q.validate();FreeEpigraphPresolve r(q);check(!r.changed(),reason);};
  auto q=p;q.ub[1]=2;reject(q,"finite upper leaf bound ignored");
  q=p;q.lb[1]=-1;reject(q,"negative lower leaf bound ignored");
  q=p;q.objective.linear[1]=.1;reject(q,"objective incidence ignored");
  q=p;q.x0[1]=1;reject(q,"noncanonical start projected implicitly");
  q=p;q.inequalities[0].constant=1;reject(q,"infeasible initial row projected implicitly");
  q=p;q.inequalities[0].hessian.entries.push_back({0,1,1});reject(q,"mixed leaf term ignored");
  q=p;q.equalities.push_back(q.inequalities[0]);reject(q,"other-row leaf incidence ignored");
  q=p;q.n=3;q.x0.push_back(0);q.lb.push_back(0);q.ub.push_back(INFINITY);q.objective.linear.push_back(0);q.objective.hessian.n=3;
  q.inequalities[0].linear.push_back(0);q.inequalities[0].hessian.n=3;q.inequalities[0].hessian.entries.push_back({2,2,-2});
  reject(q,"ambiguous multi-leaf row not refused");
  q=fixture();q.x0[0]=1;q.inequalities[0].constant=-1;q.inequalities[0].hessian.entries[0].v=std::nextafter(2.,3.);
  reject(q,"exact positive initial defect was hidden by floating roundoff");
  std::cout<<"PASS coefficient-only free epigraph proof, exact outward lift, full dual dimensions, canonical start and negative guards\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
