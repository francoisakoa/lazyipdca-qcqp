#include "common_factor_substitution.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;using namespace lazyipdca::qcqp::detail;
void check(bool v,const char* s){if(!v)throw std::runtime_error(s);}
QCQPProblem model(){
 QCQPProblem p;p.name="anonymous";p.n=2;p.x0={2,1};p.lb={0,.5};p.ub={4,3};
 p.objective.hessian.n=2;p.objective.linear={0,0};p.objective.hessian.entries={{0,1,-1}};
 QuadraticFunction q;q.hessian.n=2;q.linear={1,0};q.hessian.entries={{0,1,1}};q.constant=-8;
 p.inequalities.push_back(q);p.validate();return p;
}
int main(){
 Settings st;st.linear_backend="mumps";st.native_al_fallback="off";st.dca_engine="majorants";st.live_display_level=0;
 const double end=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+30;
 auto p=model();auto x0=p.x0;CommonFactorSubstitution sub(p,st,end);
 check(sub.applied&&p.x0==x0,"domain or original frozen start");
 Vec x={2,2},u={2,4};check(sub.model.objective.value(u)==p.objective.value(x),"objective polynomial identity");
 check(sub.model.inequalities[0].value(u)==p.inequalities[0].value(x),"row polynomial identity");
 auto renamed=p;renamed.name="arbitrary";check(CommonFactorSubstitution(renamed,st,end).applied,"name-dependent detection");
 auto tiny=p;tiny.inequalities[0].constant=0x1p-53;check(CommonFactorSubstitution(tiny,st,end).model.inequalities[0].constant==0x1p-53,"tiny constant dropped");
 auto bad=p;bad.objective.hessian.entries.push_back({1,1,1e-300});check(!CommonFactorSubstitution(bad,st,end).applied,"tiny curvature ignored");
 bad=p;bad.lb[0]=-1;check(!CommonFactorSubstitution(bad,st,end).applied,"negative pivot domain admitted");
 for(const char* c:{"HH","HS","SH","SS","SS_legacy"}){auto s=st;s.convexification_study=c;check(!CommonFactorSubstitution(p,s,end).applied,"study model changed");}
 check(!CommonFactorSubstitution(p,st,0).applied,"expired preprocessing");
 st.max_outer_iterations=1000;st.time_limit_seconds=30;const auto r=Solver(st).solve(p);
 check(r.status==Status::Solved&&r.common_factor_substitution&&r.termination_audit_valid,"native original KKT rejected");
 check(std::abs(r.objective+6)<=1e-6&&r.feasibility.total_inf<=1e-6,"wrong optimum or original feasibility");
 check(!r.native_al_fallback.attempted,"AL attempted");
 std::cout<<"common_factor_substitution PASS polynomial identity, domains, tiny terms, names, original KKT\n";
}
