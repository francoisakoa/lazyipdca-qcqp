#include "../src/atomic_newton.cpp"
#include <iostream>
using namespace lazyipdca::qcqp;
int main() {
 int checks=0;auto check=[&](bool b,const char* why){++checks;if(!b)throw std::runtime_error(why);};
 auto model=[](){QCQPProblem p;p.n=2;p.x0={.5,1};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};p.objective.linear={0,1};p.objective.hessian.n=2;QuadraticFunction g;g.linear={0,-1};g.hessian.n=2;g.hessian.entries={{0,0,2}};p.inequalities.push_back(g);g.constant=1;g.linear={-2,-1};p.inequalities.push_back(g);return p;};
 auto settings=[](){Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";s.majorant_startup_policy="legacy";s.lazy_newton_initial_mu=1;s.lazy_newton_adaptive_barrier=false;s.lazy_newton_lazy_constraints=false;s.linear_backend="mumps";s.linear_metis=true;s.verbose=false;s.live_display_level=0;s.lazy_newton_reduction=false;s.lazy_newton_objective_gradient_target=1000;s.lazy_newton_constraint_gradient_target=1000;return s;};
 for(int mode=0;mode<4;++mode) {
  auto p=model();auto off=settings();if(mode==1){p.objective.linear[1]=-1;for(auto& g:p.inequalities)g.linear[1]=1;}
  if(mode==2){off.stationarity_tolerance=8.25;off.complementarity_tolerance=8.25;}
  if(mode==3){p.lb[0]=p.ub[0]=.5;}
  auto on=off;on.majorant_initial_dual_balance=true;
  LazyNewtonStatistics sa,sb;detail::NewtonCostLedger ca(sa,Clock::now()),cb(sb,Clock::now());Workspace a(p,off,sa,ca),b(p,on,sb,cb);
  const double gamma=mode==2?.75:.5;
  check(a.x==b.x&&a.s==b.s&&a.h==b.h&&a.rp==b.rp,"cold primal/slacks/residuals changed");
  check(inf_norm(b.rd)<inf_norm(a.rd),"actual complete dual norm did not decrease");
  for(int i=0;i<a.mi;++i)if(a.active[i])check(b.z[i]==gamma*a.z[i],"active dual not commonly scaled");
  check(sb.initial_barrier==gamma*sa.initial_barrier,"initial mu metadata not scaled from actual products");
  check(sa.factor_attempts==0&&sb.factor_attempts==0,"initialization performed factorization");
 }
 for(int context=0;context<3;++context) {
  auto p=model();auto off=settings(),on=off;on.majorant_initial_dual_balance=true;
  if(context==0)on.dca_endpoint_start_file="explicit-endpoint.json";
  if(context==1)on.restoration_failed_linearization_context=true;
  if(context==2)on.restoration_private_precision_context=true;
  LazyNewtonStatistics sa,sb;detail::NewtonCostLedger ca(sa,Clock::now()),cb(sb,Clock::now());Workspace a(p,off,sa,ca),b(p,on,sb,cb);
  check(a.x==b.x&&a.s==b.s&&a.z==b.z&&a.lambda==b.lambda&&a.rd==b.rd,"warm/private context was reinitialized");
 }
 // Equality multipliers use the same scaling, including their signs.
 {
  auto p=model();QuadraticFunction h;h.constant=-.5;h.linear={1,0};h.hessian.n=2;p.equalities.push_back(h);auto s=settings();
  LazyNewtonStatistics stats;detail::NewtonCostLedger ledger(stats,Clock::now());Workspace w(p,s,stats,ledger);w.lambda[0]=3;w.refresh_majorant_residuals_from_cache();const auto x=w.x,slack=w.s;const double before=inf_norm(w.rd);w.balance_cold_duals();
  check(w.lambda[0]==1.5&&inf_norm(w.rd)<before,"equality dual not scaled with inequality duals");check(w.x==x&&w.s==slack,"common dual scaling changed primal state");
 }
 // A proposed active dual underflow is rejected before any mutation.
 {
  auto p=model();QuadraticFunction g;g.constant=-10;g.linear={0,0};g.hessian.n=2;g.hessian.entries={{0,0,2}};p.inequalities.push_back(g);auto s=settings();
  LazyNewtonStatistics stats;detail::NewtonCostLedger ledger(stats,Clock::now());Workspace w(p,s,stats,ledger);w.z[2]=std::numeric_limits<double>::denorm_min();w.refresh_majorant_residuals_from_cache();const auto z=w.z,lambda=w.lambda,rd=w.rd,slack=w.s,x=w.x;const auto wide=w.majorant_rd_extended;w.balance_cold_duals();
  check(w.z==z&&w.lambda==lambda&&w.rd==rd&&w.majorant_rd_extended==wide&&w.x==x&&w.s==slack,"underflow rejection did not retain exact source state");
 }
 // Force the transactional residual guard to reject a non-improvement.
 {
  auto p=model();auto s=settings();LazyNewtonStatistics stats;detail::NewtonCostLedger ledger(stats,Clock::now());Workspace w(p,s,stats,ledger);w.rd.assign(w.n,0);w.majorant_rd_extended.assign(w.n,0);const auto z=w.z,lambda=w.lambda,rd=w.rd;const auto wide=w.majorant_rd_extended;w.balance_cold_duals();
  check(w.z==z&&w.lambda==lambda&&w.rd==rd&&w.majorant_rd_extended==wide,"whole-dual guard rollback was not exact");
 }
 // The original coordinate survives a nontrivial exact rank map uniquely.
 {
  QCQPProblem p;p.n=5;p.x0={.5,0,.5,0,1};p.lb=Vec(5,-2);p.ub=Vec(5,2);p.lb[4]=-INFINITY;p.ub[4]=INFINITY;p.objective.linear={0,0,0,0,1};p.objective.hessian.n=5;
  for(auto rt:{std::pair<double,double>{2,3},{3,-2}}) {QuadraticFunction h;h.linear=Vec(5,0);h.hessian.n=5;auto add=[&](int i,int j,double v){h.hessian.entries.push_back({std::min(i,j),std::max(i,j),v});};add(2,2,2*rt.first);add(3,3,2*rt.first);add(2,0,-rt.first);add(3,1,-rt.first);add(3,0,rt.second);add(2,1,-rt.second);std::sort(h.hessian.entries.begin(),h.hessian.entries.end(),[](auto a,auto b){return std::pair{a.i,a.j}<std::pair{b.i,b.j};});p.equalities.push_back(h);}
  QuadraticFunction g;g.linear=Vec(5,0);g.hessian.n=5;g.constant=.25;g.hessian.entries={{2,2,-2},{3,3,-2}};p.inequalities.push_back(g);g.constant=0;g.linear[4]=-1;g.hessian.entries={{0,0,2}};p.inequalities.push_back(g);g.constant=1;g.linear[2]=-2;g.hessian.entries={{2,2,2}};p.inequalities.push_back(g);
  ConstraintPresolve reduction(p);check(reduction.changed()&&reduction.reduced().n==3,"rank-map fixture did not contract exact leaf");auto off=settings();off.rank_presolve_context=&reduction;auto on=off;on.majorant_initial_dual_balance=true;
  LazyNewtonStatistics sa,sb;detail::NewtonCostLedger ca(sa,Clock::now()),cb(sb,Clock::now());Workspace a(reduction.reduced(),off,sa,ca),b(reduction.reduced(),on,sb,cb);
  check(a.x==b.x&&a.s==b.s&&inf_norm(b.rd)<inf_norm(a.rd),"original-to-reduced-to-free balance failed");check(sb.initial_barrier==.5,"rank-map barrier balance missing");
 }
 std::cout<<"cold_dual_balance_workspace PASS checks="<<checks<<" x_s_unchanged=1 sign_flip=1 floor=1 contexts=1 rankmap=1 exact_rollback=1\n";
}
