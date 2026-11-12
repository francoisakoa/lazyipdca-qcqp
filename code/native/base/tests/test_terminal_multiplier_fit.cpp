#include "lazyipdca_qcqp/terminal_multiplier_fit.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include <stdexcept>
#include <iostream>
using namespace lazyipdca::qcqp;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){
 Settings st;st.termination_test="project";st.feasibility_tolerance=1e-6;st.stationarity_tolerance=1e-6;st.dual_infeasibility_tolerance=1e-6;st.complementarity_tolerance=1e-6;
 QCQPProblem p;p.name="bounded-linear-certificate";p.n=1;p.lb={0};p.ub={INFINITY};p.x0={1e-9};p.objective.linear={5e-6};p.objective.hessian.n=1;
 Result r;r.status=Status::MaxOuterIterations;r.x=p.x0;r.feasibility.total_inf=0;r.dual_infeasibility_inf=5e-6;r.complementarity_inf=0;r.certificate_lower_multipliers={0};r.certificate_upper_multipliers={0};
 const Vec original=r.x;
 require(detail::terminal_original_multiplier_fit(p,st,r,1),"valid fixed-primal fit rejected");
 require(r.x==original&&r.status==Status::Solved,"fit moved primal or failed to certify");
 require(r.certificate_lower_multipliers[0]>=0&&r.certificate_upper_multipliers[0]==0,"invalid nonnegative or absent-bound multiplier");
 require(r.dual_infeasibility_inf<=1e-6&&r.complementarity_inf<=1e-6,"original KKT residual rejected");
 p.x0={1};Result bad;bad.status=Status::MaxOuterIterations;bad.x=p.x0;bad.feasibility.total_inf=0;bad.dual_infeasibility_inf=5e-6;bad.complementarity_inf=0;
 require(!detail::terminal_original_multiplier_fit(p,st,bad,1),"inactive bound invented an invalid certificate");
 require(bad.x==p.x0&&bad.status==Status::MaxOuterIterations,"failed fit changed accepted state");
 Result infeasible=bad;infeasible.x={-1};infeasible.feasibility.total_inf=1;
 require(!detail::terminal_original_multiplier_fit(p,st,infeasible,1),"infeasible point certified");
 Result timed=bad;require(!detail::terminal_original_multiplier_fit(p,st,timed,0),"zero budget ignored");
 // A far auxiliary multiplier guess cannot preclude an exact fixed-primal
 // ORIGINAL certificate. This is a bound optimum with multiplier 2,000,000.
 QCQPProblem far=p;far.name="far-original-bound-certificate";
 far.x0={0};far.objective.linear={2e6};
 Result far_fit;far_fit.status=Status::InnerFailure;far_fit.x=far.x0;
 far_fit.feasibility.total_inf=0;far_fit.dual_infeasibility_inf=2e6;
 far_fit.complementarity_inf=0;
 require(detail::terminal_original_multiplier_fit(far,st,far_fit,1),
     "far auxiliary dual guess excluded valid original certificate");
 require(far_fit.x==far.x0&&far_fit.status==Status::Solved&&
     far_fit.certificate_lower_multipliers[0]==2e6&&
     far_fit.dual_infeasibility_inf<=st.dual_infeasibility_tolerance&&
     far_fit.complementarity_inf<=st.complementarity_tolerance,
     "far fit changed point or failed strict original audit");
 Result far_bad;far_bad.status=Status::InnerFailure;far_bad.x={1};
 far_bad.feasibility.total_inf=0;far_bad.dual_infeasibility_inf=2e6;
 const Result unchanged=far_bad;
 require(!detail::terminal_original_multiplier_fit(far,st,far_bad,1),
     "feasible nonstationary point received fabricated far certificate");
 require(far_bad.x==unchanged.x&&far_bad.status==unchanged.status&&
     far_bad.certificate_lower_multipliers==unchanged.certificate_lower_multipliers,
     "rejected far certificate changed accepted result");
 // Nearly parallel original equality normals make cyclic coordinate descent
 // converge slowly. This fixed primal point has an exact certificate, but
 // needs more than the former 2000-sweep cap to reach the same strict target.
 QCQPProblem slow;slow.name="near-dependent-original-normals";slow.n=2;
 slow.lb={-INFINITY,-INFINITY};slow.ub={INFINITY,INFINITY};slow.x0={0,0};
 slow.objective.linear={0,1e-5};slow.objective.hessian.n=2;
 QuadraticFunction a;a.linear={1,0};a.hessian.n=2;
 QuadraticFunction b;b.linear={1,.02};b.hessian.n=2;
 slow.equalities={a,b};
 Result fit;fit.status=Status::MaxOuterIterations;fit.x=slow.x0;
 fit.feasibility.total_inf=0;fit.dual_infeasibility_inf=1e-5;fit.complementarity_inf=0;
 require(detail::terminal_original_multiplier_fit(slow,st,fit,1),"slow valid multiplier fit stopped prematurely");
 require(fit.x==slow.x0&&fit.status==Status::Solved,"slow fit moved primal or failed original audit");
 require(fit.dual_infeasibility_inf<=1e-6,"slow original certificate exceeds strict target");
 // Exact duplicate elimination must postpone, rather than permanently
 // suppress, fixed-primal ORIGINAL multiplier recovery.
 QCQPProblem duplicate=p;duplicate.x0={1e-9};
 QuadraticFunction row;row.hessian.n=1;row.linear={-1};
 duplicate.inequalities={row,row};
 ConstraintPresolve reduction(duplicate);
 require(reduction.changed(),"duplicate-row test did not exercise exact presolve");
 Result reduced;reduced.status=Status::MaxOuterIterations;reduced.x=reduction.reduced().x0;
 reduced.feasibility.total_inf=0;reduced.dual_infeasibility_inf=5e-6;
 reduced.complementarity_inf=0;
 reduced.certificate_equality_multipliers.assign(reduction.reduced().equalities.size(),0);
 reduced.certificate_inequality_multipliers.assign(reduction.reduced().inequalities.size(),0);
 reduced.certificate_lower_multipliers.assign(reduction.reduced().n,0);
 reduced.certificate_upper_multipliers.assign(reduction.reduced().n,0);
 Settings inner=st;inner.rank_presolve_context=&reduction;
 require(!detail::terminal_original_multiplier_fit(reduction.reduced(),inner,reduced,1),
     "reduced-space multipliers advertised as original certificate");
 reduction.lift(reduced);
 require(detail::terminal_original_multiplier_fit(duplicate,st,reduced,1)&&
     reduced.x==duplicate.x0&&reduced.status==Status::Solved,
     "exact postsolve lift lost the original fixed-primal certificate");
 require(reduced.certificate_inequality_multipliers.size()==2,
     "postsolve fit omitted duplicate original rows");
 const auto original_strict=reduced;
 require(!detail::terminal_original_multiplier_fit(duplicate,st,reduced,1)&&
     reduced.x==original_strict.x&&reduced.status==original_strict.status&&
     reduced.certificate_inequality_multipliers==original_strict.certificate_inequality_multipliers&&
     reduced.certificate_lower_multipliers==original_strict.certificate_lower_multipliers,
     "already strict postsolve certificate changed");
 Result nonstationary=original_strict;nonstationary.status=Status::MaxOuterIterations;
 nonstationary.x={1};nonstationary.feasibility.total_inf=0;
 nonstationary.dual_infeasibility_inf=5e-6;
 const auto bad_original=nonstationary;
 require(!detail::terminal_original_multiplier_fit(duplicate,st,nonstationary,1)&&
     nonstationary.x==bad_original.x&&nonstationary.status==bad_original.status,
     "postsolve fit invented a stationary point from inactive original rows");
 // Assembly itself is charged against the fitting budget. A large original
 // box must not enter its quadratic bound-column loop after budget expiry.
 QCQPProblem large;large.n=20000;large.x0=Vec(large.n,1e-9);
 large.lb=Vec(large.n,0);large.ub=Vec(large.n,INFINITY);
 large.objective.hessian.n=large.n;large.objective.linear=Vec(large.n,5e-6);
 Result expired;expired.status=Status::MaxOuterIterations;expired.x=large.x0;
 expired.feasibility.total_inf=0;expired.dual_infeasibility_inf=5e-6;
 const auto timer=std::chrono::steady_clock::now();
 require(!detail::terminal_original_multiplier_fit(large,st,expired,1e-9),
     "assembly ignored its expired wall-clock budget");
 require(std::chrono::duration<double>(std::chrono::steady_clock::now()-timer).count()<1.,
     "large bound assembly was not interrupted at its budget");
 require(expired.x==large.x0&&expired.status==Status::MaxOuterIterations&&
     expired.certificate_lower_multipliers.empty(),"budget rejection mutated accepted state");
 std::cout<<"terminal_multiplier_fit: valid / complementarity-limited / infeasible / zero budget / near-dependent normals / duplicate-postlift / strict-unchanged / original-nonstationary-reject / assembly-deadline PASS\n";
}
