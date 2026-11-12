#include "convex_aggregate_certificate.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <stdexcept>
#include <iostream>
using namespace lazyipdca::qcqp;using namespace lazyipdca::qcqp::detail;
void check(bool v,const char* m){if(!v)throw std::runtime_error(m);}
QCQPProblem model(){QCQPProblem p;p.n=2;p.name="anonymous";p.x0={0,0};p.lb={2,-INFINITY};p.ub={INFINITY,INFINITY};
 p.objective.hessian.n=2;p.objective.linear={-1,0};QuadraticFunction q;q.hessian.n=2;q.linear={-1,1};p.equalities.push_back(q);
 q.linear={0,0};q.hessian.entries={{1,1,2}};q.constant=-1;p.inequalities.push_back(q);p.validate();return p;}
int main(){double end=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+20;auto p=model();auto c=convex_aggregate_certificate(p,1e-6,end);
 check(c.verified&&c.bound_variable==0&&c.equality_sign==-1,"diagonal inequality/affine/bound contradiction missing");
 using Q=boost::multiprecision::cpp_rational;check(Q(c.minimum)>Q(1e-6)*Q(c.weight_norm),"lower bound does not exclude tolerance");
 auto renamed=p;renamed.name="other";check(convex_aggregate_certificate(renamed,1e-6,end).verified,"name-dependent detection");
 auto feasible=p;feasible.lb[0]=.5;check(!convex_aggregate_certificate(feasible,1e-6,end).verified,"feasible model rejected");
 auto nonlinear=p;nonlinear.equalities[0].hessian.entries={{0,0,1e-300}};check(!convex_aggregate_certificate(nonlinear,1e-6,end).verified,"tiny equality curvature discarded");
 auto indefinite=p;indefinite.inequalities[0].hessian.entries.push_back({0,0,-1e-300});check(!convex_aggregate_certificate(indefinite,1e-6,end).verified,"tiny negative curvature discarded");
 check(!convex_aggregate_certificate(p,1e-6,0).verified,"expired certificate");
 Settings st;st.native_al_fallback="off";auto r=Solver(st).solve(p);check(r.status==Status::Infeasible&&r.convex_infeasibility_certificate&&!r.termination_audit_valid,"infeasibility counted as success");
 std::cout<<"convex_aggregate_certificate PASS original coefficients, diagonal completion, bound, names, tiny terms, no KKT success\n";
}
