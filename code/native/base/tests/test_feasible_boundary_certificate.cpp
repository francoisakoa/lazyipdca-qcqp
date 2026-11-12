#include "../src/feasible_boundary_certificate.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
QCQPProblem problem(){
  QCQPProblem p;p.name="boundary_equality_generic";p.n=2;p.x0={.3,.5};
  p.lb={-1,-INFINITY};p.ub={1,INFINITY};p.objective.linear={1,2};p.objective.hessian.n=2;
  QuadraticFunction a;a.constant=-1;a.linear={0,0};a.hessian.n=2;a.hessian.entries={{0,0,2}};
  QuadraticFunction b;b.linear={-1,1};b.hessian.n=2;p.equalities={a,b};return p;
}
Result failure(){Result r;r.status=Status::InnerFailure;r.message="precision rejection";r.x={1,1};r.objective=3;r.feasibility.total_inf=0;r.dca_filter.majorant_reject_precision=1;r.certificate_equality_multipliers={23,24};r.certificate_lower_multipliers={25,26};r.certificate_upper_multipliers={27,28};return r;}
void unchanged(const Result&a,const Result&b){check(a.status==b.status&&a.message==b.message&&a.x==b.x&&a.objective==b.objective&&a.certificate_equality_multipliers==b.certificate_equality_multipliers&&a.certificate_lower_multipliers==b.certificate_lower_multipliers&&a.certificate_upper_multipliers==b.certificate_upper_multipliers,"rejected fit changed result");}
}
int main(){
 Settings s;s.dca_engine="majorants";s.linear_backend="native";s.verbose=false;s.restoration_enabled=true;
 s.feasibility_tolerance=s.stationarity_tolerance=s.complementarity_tolerance=1e-6;s.kkt_backward_error_reject=1e-7;
 for(int mode=0;mode<5;++mode){auto p=problem();auto r=failure();
  if(mode==1)p.name="renamed_without_eigen";
  if(mode==2){p.equalities[0].constant*=13;for(auto&e:p.equalities[0].hessian.entries)e.v*=13;}
  if(mode==3){p.x0={.9,.1};}
  if(mode==4){r.status=Status::LineSearchFailure;r.dca_filter.majorant_last_rejection="barrier-filter";r.complementarity_inf=.1;r.dca_filter.majorant_reject_precision=0;}
  const Vec initial=p.x0,x=r.x;const auto a=detail::feasible_boundary_certificate(p,s,r,3);
  check(a.attempted&&a.accepted&&a.factorizations==1&&a.solves>=1,"valid fixed-point certificate rejected");
  check(r.x==x&&p.x0==initial&&r.objective==3,"primal or objective changed");
  check(r.status==Status::Solved&&r.termination_audit_valid&&r.termination_certificate_rescued,"missing original certificate");
  check(a.dual<=1e-10&&a.primal==0&&a.complementarity==0,"invalid numerical certificate");
  check(std::all_of(r.certificate_lower_multipliers.begin(),r.certificate_lower_multipliers.end(),[](double v){return v>=0&&v<1e-12;})&&std::all_of(r.certificate_upper_multipliers.begin(),r.certificate_upper_multipliers.end(),[](double v){return v>=0&&v<1e-12;}),"spurious bound multipliers");
 }
 for(Status status:{Status::Solved,Status::SolvedAcceptable,Status::TimeLimit,Status::MaxOuterIterations,Status::FactorizationFailure,Status::LineSearchFailure}){auto r=failure();r.status=status;auto old=r;auto a=detail::feasible_boundary_certificate(problem(),s,r,3);check(!a.attempted&&a.factorizations==0,"unrelated status activated rescue");unchanged(r,old);}
 for(int mode=0;mode<8;++mode){auto p=problem();auto r=failure();auto st=s;double budget=3;
  if(mode==0)r.dca_filter.majorant_reject_precision=0;
  if(mode==1)st.restoration_enabled=false;
  if(mode==2)st.dca_engine="filter";
  if(mode==3)r.feasibility.total_inf=.1;
  if(mode==4)p.ub[0]=2;
  if(mode==5)p.equalities.pop_back();
  if(mode==6)p.inequalities.push_back(p.equalities[0]);
  if(mode==7)budget=0;
  auto old=r;auto a=detail::feasible_boundary_certificate(p,st,r,budget);check(!a.attempted&&a.factorizations==0,"scope/precision/budget guard missed");unchanged(r,old);
 }
 {auto p=problem();p.equalities[0].constant=-2;auto r=failure();auto old=r;auto a=detail::feasible_boundary_certificate(p,s,r,3);check(!a.accepted&&a.factorizations==0,"stale primal metric escaped independent audit");unchanged(r,old);}
 {auto p=problem();p.equalities[0].linear={1,0};p.equalities[0].hessian.entries.clear();p.equalities[0].constant=-1;p.equalities[1]=p.equalities[0];auto r=failure();auto old=r;auto a=detail::feasible_boundary_certificate(p,s,r,3);check(!a.accepted,"deficient equality Jacobian accepted");unchanged(r,old);}
 {auto p=problem();auto r=failure();r.x[0]=NAN;auto old=r;auto a=detail::feasible_boundary_certificate(p,s,r,3);check(!a.attempted&&r.status==old.status&&std::isnan(r.x[0]),"nonfinite point not guarded");}
 std::cout<<"feasible boundary: original audit, fixed primal, failure-only trigger, no-name selection, rollback, deadline PASS\n";
}
