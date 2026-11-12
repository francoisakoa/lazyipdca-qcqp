#include "../src/cold_dual_balance.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main() {
 int n=0;auto check=[&](bool b,const char* why){++n;if(!b)throw std::runtime_error(why);};
 auto model=[](){QCQPProblem p;p.n=2;p.x0={.5,1};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};p.objective.linear={0,1};p.objective.hessian.n=2;
  QuadraticFunction g;g.constant=0;g.linear={0,-1};g.hessian.n=2;g.hessian.entries={{0,0,2}};p.inequalities.push_back(g);g.constant=1;g.linear={-2,-1};p.inequalities.push_back(g);return p;};
 auto p=model();auto col=detail::affine_objective_column(p);check(col.coordinate==1&&col.coefficient==1,"affine epigraph column rejected");
 auto q=p;q.objective.hessian.entries={{1,1,2}};check(detail::affine_objective_column(q).coordinate<0,"zero current gradient hid objective curvature");
 q=p;q.objective.linear[0]=.1;check(detail::affine_objective_column(q).coordinate<0,"extra objective coordinate admitted");
 q=p;QuadraticFunction h;h.linear={0,0};h.hessian.n=2;h.hessian.entries={{1,1,2}};q.equalities.push_back(h);check(detail::affine_objective_column(q).coordinate<0,"equality Hessian column at zero admitted");
 q=p;h.hessian.entries.clear();h.linear={0,1};q.equalities.push_back(h);check(detail::affine_objective_column(q).coordinate<0,"linear equality epigraph dependence admitted");
 q=p;q.inequalities[0].linear[1]=1;check(detail::affine_objective_column(q).coordinate<0,"opposing inequality admitted");
 q=p;q.inequalities[0].hessian.entries.push_back({1,1,2});check(detail::affine_objective_column(q).coordinate<0,"inequality Hessian column admitted");
 q=p;q.ub[1]=3;check(detail::affine_objective_column(q).coordinate<0,"opposing native upper bound admitted");
 q=p;q.objective.linear[1]=-1;for(auto& g:q.inequalities)g.linear[1]=1;q.ub[1]=3;check(detail::affine_objective_column(q).coordinate==1,"objective coordinate sign flip changed admission");
 q=p;q.objective.linear.clear();q.objective.compact_linear_dimension=2;q.objective.compact_linear={{1,1}};check(detail::affine_objective_column(q).coordinate==1,"sparse original objective not handled");
 auto plan=detail::cold_dual_balance_plan(1,-2,2,1,1e-7);check(plan.allowed&&plan.scale==.5,"ratio did not balance epigraph stationarity");
 plan=detail::cold_dual_balance_plan(-1,2,2,1,1e-7);check(plan.allowed&&plan.scale==.5,"flipped column ratio wrong");
 plan=detail::cold_dual_balance_plan(1,-2,2,1,.75);check(plan.allowed&&plan.scale==.75,"barrier floor clip wrong");
 check(!detail::cold_dual_balance_plan(1,-2,2,1,1).allowed,"barrier floor scaling>=1 admitted");
 check(!detail::cold_dual_balance_plan(1,-2,1,1,0).allowed,"single contributor admitted");
 for(double c:{-1.,-.5,0.,2.,double(INFINITY)})check(!detail::cold_dual_balance_plan(1,c,2,1,0).allowed,"unnecessary or invalid ratio admitted");
 check(!detail::cold_dual_balance_plan(1,-2,2,0,0).allowed,"nonpositive actual complementarity mean admitted");
 check(!detail::cold_dual_balance_plan(1,-2,2,1,-1).allowed,"negative barrier floor admitted");
 // Scaling all constraint multipliers preserves every other affine-objective
 // stationarity component by gamma; the objective coordinate is balanced.
 for(double gamma:{.5,.75}) {const double old_other=9,old_epi=-1;const double next_other=gamma*old_other,next_epi=1+gamma*(old_epi-1);check(std::max(std::abs(next_other),std::abs(next_epi))<9,"global dual algebra did not decrease");check(4*(.25*gamma)==gamma,"relative complementarity not preserved");}
 std::cout<<"cold_dual_balance PASS checks="<<n<<" coefficient_admission=1 sign_flip=1 actual_mean_floor=1 dual_algebra=1\n";
}
