#include "../src/bounded_origin_certificate.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void check(bool yes,const char* message){if(!yes)throw std::runtime_error(message);}
QCQPProblem problem(double tiny=0){
 QCQPProblem p;p.n=2;p.name="generic_certificate";p.lb={0,1.4};p.ub={6,1.6};p.x0={4,1.5};
 p.objective.linear={2,.3};p.objective.hessian.n=2;
 QuadraticFunction q;q.linear={1,tiny};q.hessian.n=2;q.constant=-1.4*tiny;p.equalities={q};
 return p;
}
Result failure(Status status=Status::InnerFailure){Result r;r.status=status;r.x={4,1.5};r.message="preserve failure";r.objective=123;r.certificate_equality_multipliers={91};return r;}
void unchanged(const Result&r,Status status=Status::InnerFailure){check(r.status==status&&r.x==Vec({4,1.5})&&r.objective==123&&r.message=="preserve failure"&&r.certificate_equality_multipliers==Vec({91}),"failed candidate changed result");}
}
int main(){
 Settings st;st.verbose=false;st.linear_backend="native";st.feasibility_tolerance=st.stationarity_tolerance=st.complementarity_tolerance=1e-6;st.kkt_backward_error_reject=1e-7;
 for(double tiny:{0.,1e-18})for(Status trigger:{Status::InnerFailure,Status::FactorizationFailure,Status::LineSearchFailure}){
  auto p=problem(tiny);auto r=failure(trigger);auto a=detail::bounded_origin_certificate(p,st,r,5);
  check(a.accepted&&a.factorizations==1&&a.solves>=1,"valid strict certificate rejected");
  check(r.status==Status::Solved&&r.x==Vec({0,1.4})&&p.x0==Vec({4,1.5}),"candidate or frozen start changed incorrectly");
  check(r.termination_audit_valid&&a.primal<=1e-6&&a.dual<=1e-6&&a.complementarity==0,"original certificate audit missing");
  check(a.expanded_relative_residual<=st.kkt_backward_error_reject,"expanded adjoint residual not guarded");
  check(r.certificate_inequality_multipliers.empty()&&r.certificate_lower_multipliers[1]>.29,"bound certificate missing");
 }
 for(Status skip:{Status::Solved,Status::SolvedAcceptable,Status::TimeLimit,Status::MaxOuterIterations}){
  auto r=failure(skip);auto a=detail::bounded_origin_certificate(problem(),st,r,5);check(!a.attempted,"wrong trigger");unchanged(r,skip);
 }
 {auto r=failure();auto a=detail::bounded_origin_certificate(problem(),st,r,0);check(!a.attempted&&a.factorizations==0,"expired budget performed work");unchanged(r);}
 {auto p=problem();p.equalities[0].constant=1;auto r=failure();auto a=detail::bounded_origin_certificate(p,st,r,5);check(!a.accepted&&a.factorizations==0,"infeasible origin accepted");unchanged(r);}
 {auto p=problem();p.objective.linear[1]=-1;auto r=failure();auto a=detail::bounded_origin_certificate(p,st,r,5);check(!a.accepted&&a.factorizations==0,"unsupported wrong-sign gradient accepted");unchanged(r);}
 {auto p=problem();p.equalities[0].linear[1]=1;p.equalities[0].constant=-1.4;auto r=failure();auto a=detail::bounded_origin_certificate(p,st,r,5);check(!a.accepted&&a.factorizations==0,"rectangular Jacobian accepted");unchanged(r);}
 {auto p=problem(1e-15);p.objective.linear={1e12,0};auto r=failure();auto a=detail::bounded_origin_certificate(p,st,r,5);check(!a.accepted&&a.factorizations==1&&a.dual>1e-6,"ignored numerical column escaped original audit");unchanged(r);}
 {auto p=problem();p.equalities[0].constant=std::numeric_limits<double>::quiet_NaN();auto r=failure();auto a=detail::bounded_origin_certificate(p,st,r,5);check(!a.accepted&&a.factorizations==0,"nonfinite candidate accepted");unchanged(r);}
 {auto p=problem();auto q=p.equalities[0];q.constant=.1;q.linear={0,0};p.inequalities={q};auto r=failure();auto a=detail::bounded_origin_certificate(p,st,r,5);check(!a.accepted&&a.factorizations==0,"violated original inequality accepted");unchanged(r);}
 {auto p=problem();p.lb[1]=p.ub[1]=1.4;p.equalities[0].linear[1]=1;p.equalities[0].constant=-1.4;p.objective.linear[1]=-8;auto r=failure();auto a=detail::bounded_origin_certificate(p,st,r,5);check(a.accepted&&a.dual<1e-10&&r.certificate_upper_multipliers[1]>9.9,"fixed-column certificate rejected");}
 std::cout<<"Bounded-origin strict certificate, numerical-support audit, rollback and deadlines: PASS\n";
}
