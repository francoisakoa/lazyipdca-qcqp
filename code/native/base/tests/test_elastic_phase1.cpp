#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/inner_ipm.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(){try {
 QCQPProblem p;p.name="incompatible_tangents";p.n=3;p.x0={.5,7,3};
 p.lb={0,-INFINITY,3};p.ub={2,INFINITY,3};p.objective.linear={2,0,0};p.objective.hessian.n=3;
 QuadraticFunction h;h.constant=-1;h.linear={0,0,0};h.hessian.n=3;h.hessian.entries={{0,0,2}};
 QuadraticFunction a;a.constant=-1;a.linear={1,0,0};a.hessian.n=3;
 QuadraticFunction g;g.constant=.25;g.linear={-1,0,0};g.hessian.n=3;
 p.equalities={h,a};p.inequalities={g};p.validate();
 Settings s;s.linear_backend="mumps";s.linear_metis=true;s.verbose=false;s.live_display_level=0;
 s.restoration_failed_linearization_context=true;s.restoration_shared_iteration_budget=true;
 s.feasibility_tolerance=1e-6;s.restoration_max_iterations=240;
 auto now=[](){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();};
 s.restoration_deadline_seconds=now()+10;
 auto r=FeasibilityRestoration(s).solve_elastic(p,p.x0);
 check(r.success&&feasibility(p,r.x).total_inf<=1e-6,"incompatible tangent Phase I did not recover original feasibility");
 check(r.minimax_qp_solves>0,"elastic epigraph not exercised");
 check(p.x0==Vec({.5,7,3})&&r.x[1]==7&&r.x[2]==3,"frozen start or absent/fixed bound changed");
 check(r.iterations<=240,"shared budget overrun");
 for(int budget:{0,1,4,12}){auto limited=s;limited.restoration_max_iterations=budget;
 auto b=FeasibilityRestoration(limited).solve_elastic(p,p.x0);check(b.iterations<=budget,"small budget overrun");}
 auto expired=s;expired.restoration_deadline_seconds=0;
 auto e=FeasibilityRestoration(expired).solve_elastic(p,p.x0);
 check(!e.success&&e.iterations==0&&e.x==p.x0,"expired elastic solve moved point");
 ConvexSubproblem q;q.nvar=1;q.objective.linear={0};q.objective.base_hessian.n=1;q.equalities.nvar=1;
 auto ir=ConvexQCQPInteriorPoint(expired).solve(q);
 check(!ir.success&&ir.iterations==0,"expired convex solve accepted point");
 // A separate mean-complementarity target must override even the guarded
 // last-iteration fallback, without changing ordinary residual tolerances.
 ConvexSubproblem c;c.nvar=1;c.equalities.nvar=1;c.objective.linear={0};c.objective.base_hessian.n=1;
 c.objective.base_hessian.entries={{0,0,1}};
 ConvexQuadraticModel row;row.constant=-1;row.linear={-1};row.base_hessian.n=1;c.inequalities={row};
 auto cs=s;cs.restoration_deadline_seconds=now()+10;cs.inner_tolerance=1e-7;cs.max_inner_iterations=100;
 c.complementarity_tolerance=1e-13;cs.minimum_positive=1e-18;
 auto ci=ConvexQCQPInteriorPoint(cs).solve(c);
 check(ci.success&&ci.complementarity<=1e-13&&ci.inequality_slacks.size()==1,"strict separate complementarity target ignored");
 cs.max_inner_iterations=1;cs.inner_tolerance=100;cs.inner_iteration_limit_mu_fallback=100;
 auto short_ci=ConvexQCQPInteriorPoint(cs).solve(c);
 check(!short_ci.success,"last-iteration fallback bypassed explicit complementarity target");
 std::cout<<"PASS elastic Phase I: incompatible tangents, original rows, fixed/unbounded variables, shared budgets, deadline\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
