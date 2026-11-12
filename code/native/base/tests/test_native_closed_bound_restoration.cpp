#include "lazyipdca_qcqp/restoration.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){
 QCQPProblem p;p.n=2;p.name="anonymous_boundary";p.x0={1,1};p.lb={0,-INFINITY};p.ub={INFINITY,INFINITY};p.objective.hessian.n=2;p.objective.linear={0,0};
 QuadraticFunction h;h.hessian.n=2;h.linear={1,0};p.equalities.push_back(h);h.linear={0,0};h.constant=-1;h.signed_square={{1,1}};p.equalities.push_back(h);
 Settings st;st.linear_backend="mumps";st.restoration_max_iterations=8;st.restoration_max_restarts=0;st.feasibility_tolerance=1e-8;
 auto r=FeasibilityRestoration(st).solve(p,p.x0);
 if(!r.success||r.final_feasibility>st.feasibility_tolerance||p.x0!=Vec({1,1})||r.x[0]<0)throw std::runtime_error("native closed-bound original feasibility contract");
 auto renamed=p;renamed.name="unrelated";auto again=FeasibilityRestoration(st).solve(renamed,renamed.x0);
 if(r.x!=again.x)throw std::runtime_error("name-dependent Phase I");
 std::cout<<"native_closed_bound_restoration PASS native C1 bounds, original feasibility, unchanged input, renamed model\n";
}
