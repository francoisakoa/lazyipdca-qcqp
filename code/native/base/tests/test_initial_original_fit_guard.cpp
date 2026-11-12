#include "lazyipdca_qcqp/terminal_multiplier_fit.hpp"
#include <cassert>
using namespace lazyipdca::qcqp;
int main(){
 QCQPProblem p;p.n=101;p.equalities.resize(101);Settings st;st.feasibility_tolerance=1e-6;
 assert(detail::initial_original_multiplier_fit_admitted(p,st,2.4e-8,60));
 assert(!detail::initial_original_multiplier_fit_admitted(p,st,2e-6,60));
 assert(!detail::initial_original_multiplier_fit_admitted(p,st,2e-8,.004));
 assert(!detail::initial_original_multiplier_fit_admitted(p,st,INFINITY,60));
 assert(!detail::initial_original_multiplier_fit_admitted(p,st,-1,60));
 auto neq=p;neq.equalities.pop_back();assert(!detail::initial_original_multiplier_fit_admitted(neq,st,0,60));
 auto large=p;large.n=10000;assert(!detail::initial_original_multiplier_fit_admitted(large,st,0,60));
 auto many=p;many.inequalities.resize(2000);assert(!detail::initial_original_multiplier_fit_admitted(many,st,0,60));
 auto work=p;work.n=400;work.equalities.resize(400);assert(!detail::initial_original_multiplier_fit_admitted(work,st,0,60));
 auto cost=p;cost.objective.hessian.entries.resize(1000001);assert(!detail::initial_original_multiplier_fit_admitted(cost,st,0,60));
}
