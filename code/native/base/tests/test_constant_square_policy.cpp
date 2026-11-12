#include "lazyipdca_qcqp/constant_square_policy.hpp"
#include <stdexcept>
#include <limits>
using namespace lazyipdca::qcqp;
int main() {
 auto check=[](bool b){if(!b)throw std::runtime_error("constant square policy");};
 QCQPProblem p;p.n=2;p.x0={0,0};p.lb={-INFINITY,INFINITY};p.ub={INFINITY,INFINITY};
 p.objective.hessian.n=2;p.objective.linear={0,0};
 QuadraticFunction h;h.hessian.n=2;h.linear={0,0};h.constant=-1;h.hessian.entries={{0,0,2}};
 p.equalities={h,h};Settings s;check(detail::constant_square_feasibility_filter(p,s));
 p.name="unrelated";check(detail::constant_square_feasibility_filter(p,s));
 s.standard_r10_policy="off";check(!detail::constant_square_feasibility_filter(p,s));s.standard_r10_policy="auto";
 p.objective.linear[0]=1;check(!detail::constant_square_feasibility_filter(p,s));p.objective.linear[0]=0;
 p.objective.hessian.entries={{0,0,2}};check(!detail::constant_square_feasibility_filter(p,s));p.objective.hessian.entries.clear();
 p.lb[0]=0;check(!detail::constant_square_feasibility_filter(p,s));p.lb[0]=-INFINITY;
 p.inequalities={h};check(!detail::constant_square_feasibility_filter(p,s));p.inequalities.clear();
 p.equalities.pop_back();check(!detail::constant_square_feasibility_filter(p,s));
 p.lb[0]=p.ub[0]=0;check(detail::constant_square_feasibility_filter(p,s));
}
