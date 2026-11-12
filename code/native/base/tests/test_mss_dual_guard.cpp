// Regression: a small accepted multiplier vector does not justify accepting
// an enormous freshly recomputed vector on a numerically vanishing row.
#include <cstddef>
#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
static void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
static void exercise(double epsilon,bool ill_conditioned) {
  QCQPProblem p;p.name="dual_candidate_magnitude_guard";p.n=2;p.x0={0,0};
  p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);p.objective.linear={1,1};p.objective.hessian.n=2;
  QuadraticFunction a;a.name="first";a.linear={1,0};a.hessian.n=2;
  QuadraticFunction b;b.name="second";b.linear={0,epsilon};b.hessian.n=2;
  p.equalities={a,b};p.validate();
  Settings st;st.dca_engine="majorants";st.linear_backend="native";
  st.lazy_newton_reduction=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;st.verbose=false;st.live_display_level=0;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.lambda={-1,0};w.evaluate();
  auto metrics=dca_outer_metrics(w);const Vec x=w.x,old_lambda=w.lambda;
  const double before=metrics.dual;DcaFilterStatistics stats;
  const bool kept=dca_majorant_recalc_lambda(w,metrics,stats,40,Clock::now(),30,false,true);
  require(w.x==x,"multiplier correction modified the primal point");
  require(metrics.dual<=before,"correction worsened original stationarity");
  if(ill_conditioned) {
    require(!kept&&w.lambda==old_lambda,"unstable candidate accepted despite its huge multiplier norm");
    require(metrics.dual>st.stationarity_tolerance,"unresolved residual disguised as solved");
  } else {
    require(kept,"healthy multiplier correction disabled");
    require(metrics.dual<1e-6,"healthy full-rank correction not accurate");
  }
}
int main(){try{exercise(1,false);exercise(1e-16,true);
 std::cout<<"PASS original-space multiplier-candidate guard, healthy path, primal immutability\n";
 return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
