#include "positive_square_epigraph_substitution.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;using namespace lazyipdca::qcqp::detail;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
QCQPProblem model(){
 QCQPProblem p;p.name="anonymous";p.n=3;p.x0={1,0,0};p.lb={0,-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY,INFINITY};
 p.objective.hessian.n=3;p.objective.linear={1,0,0};
 QuadraticFunction h;h.hessian.n=3;h.linear={0,1,0};h.constant=-2;p.equalities.push_back(h);h.linear={0,0,1};h.constant=-3;p.equalities.push_back(h);
 QuadraticFunction g;g.hessian.n=3;g.linear={0,0,0};g.constant=-0x1p-53;g.hessian.entries={{0,0,-2},{1,1,2},{2,2,2}};p.inequalities.push_back(g);p.validate();return p;
}
int main(){
 Settings st;st.linear_backend="mumps";st.native_al_fallback="off";st.dca_engine="majorants";st.live_display_level=0;
 st.max_outer_iterations=1000;st.time_limit_seconds=30;
 const double deadline=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+30;
 auto p=model();const auto before=p.x0;PositiveSquareEpigraphSubstitution sub(p,st,deadline);
 check(sub.applied&&p.x0==before,"domain rejected or original start overwritten");
 check(sub.model.inequalities[0].constant==p.inequalities[0].constant,"tiny original constant dropped");
 check(sub.model.inequalities[0].hessian.entries.size()==2&&sub.model.inequalities[0].compact_linear==std::vector<std::pair<int,double>>{{0,-1}},"incorrect exact transformed polynomial");
 Vec u={13,2,3},x={std::sqrt(13.),2,3};
 check(std::abs(sub.model.inequalities[0].value(u)-p.inequalities[0].value(x))<=1e-13,"point map violates original polynomial");
 Result mapped;mapped.x=u;mapped.certificate_equality_multipliers={-4,-6};mapped.certificate_inequality_multipliers={1};mapped.certificate_lower_multipliers={0,0,0};mapped.certificate_upper_multipliers={0,0,0};
 check(sub.lift(mapped),"positive inverse map rejected");auto grad=p.objective.gradient(mapped.x);
 for(int i=0;i<2;++i)axpy(mapped.certificate_equality_multipliers[i],p.equalities[i].gradient(mapped.x),grad);
 axpy(mapped.certificate_inequality_multipliers[0],p.inequalities[0].gradient(mapped.x),grad);
 check(inf_norm(grad)<=1e-13,"original KKT multiplier chain rule incorrect");
 auto renamed=p;renamed.name="unrelated";check(PositiveSquareEpigraphSubstitution(renamed,st,deadline).applied,"label-dependent eligibility");
 auto bad=p;bad.inequalities[0].hessian.entries.push_back({1,2,1e-300});check(!PositiveSquareEpigraphSubstitution(bad,st,deadline).applied,"tiny offdiagonal ignored");
 bad=p;bad.equalities[0].linear[0]=1e-300;check(!PositiveSquareEpigraphSubstitution(bad,st,deadline).applied,"tiny epigraph equality coefficient ignored");
 bad=p;bad.x0[0]=.1;check(!PositiveSquareEpigraphSubstitution(bad,st,deadline).applied,"rounded initial square admitted as exact");
 for(const char* c:{"HH","HS","SH","SS","SS_legacy"}){auto explicit_st=st;explicit_st.convexification_study=c;check(!PositiveSquareEpigraphSubstitution(p,explicit_st,deadline).applied,"curvature study model changed");}
 check(!PositiveSquareEpigraphSubstitution(p,st,0).applied,"expired transformation");
 auto result=Solver(st).solve(p);
 check(result.status==Status::Solved&&result.positive_square_epigraph_substitution&&result.termination_audit_valid,"native postsolve KKT not qualified");
 check(feasibility(p,result.x).total_inf<=1e-6&&!result.native_al_fallback.attempted,"original primal or AL contract");
 std::cout<<"positive_square_epigraph_substitution PASS exact domain, polynomial identity, multiplier chain rule, names, tiny terms, frozen start and native original audit\n";
}
