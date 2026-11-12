#include "lazyipdca_qcqp/terminal_multiplier_fit.hpp"
#include <cassert>
using namespace lazyipdca::qcqp;
QCQPProblem model(int n){QCQPProblem p;p.n=n;p.name="generic-initial-certificate";p.x0.assign(n,0);p.lb.assign(n,-2);p.ub.assign(n,2);p.objective.hessian.n=n;p.objective.linear.assign(n,0);return p;}
Result state(const QCQPProblem& p,const Vec& x){Result r;r.status=Status::MaxOuterIterations;r.x=x;r.feasibility.total_inf=0;r.dual_infeasibility_inf=1;r.complementarity_inf=1;r.certificate_equality_multipliers.assign(p.equalities.size(),0);r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);return r;}
int main(){Settings st;st.live_display_level=0;st.termination_test="project";st.feasibility_tolerance=st.stationarity_tolerance=st.dual_infeasibility_tolerance=st.complementarity_tolerance=1e-6;
 auto p=model(1);p.objective.linear[0]=1;p.lb[0]=-1;p.ub[0]=1;QuadraticFunction h;h.hessian.n=1;h.constant=-1;h.linear={0};h.hessian.entries={{0,0,2}};p.equalities={h};
 auto r=state(p,{1});auto x=r.x;assert(detail::terminal_original_multiplier_fit(p,st,r,.1));assert(r.x==x&&r.status==Status::Solved);assert(r.feasibility.total_inf<=1e-6&&r.dual_infeasibility_inf<=1e-6&&r.complementarity_inf<=1e-6);for(auto a:r.certificate_lower_multipliers)assert(a>=0);for(auto a:r.certificate_upper_multipliers)assert(a>=0);
 auto infeasible=state(p,{1.1});x=infeasible.x;assert(!detail::terminal_original_multiplier_fit(p,st,infeasible,.1));assert(infeasible.x==x&&infeasible.status!=Status::Solved);
 auto tangent=model(2);tangent.objective.linear[1]=1;h.hessian.n=2;h.linear={0,0};tangent.equalities={h,h};auto notstationary=state(tangent,{1,0});x=notstationary.x;assert(!detail::terminal_original_multiplier_fit(tangent,st,notstationary,.1));assert(notstationary.x==x&&notstationary.status!=Status::Solved);
}
