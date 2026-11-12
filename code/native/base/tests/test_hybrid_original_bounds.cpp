#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void close(double a,double b,const char* message){require(std::abs(a-b)<=1e-12*std::max({1.,std::abs(a),std::abs(b)}),message);}
QCQPProblem radial_problem() {
  QCQPProblem p;p.name="radial_domain_filter_contract";p.n=3;p.x0={1,0,2};
  p.lb={-1,-1,2};p.ub={1,1,2};p.objective.linear={0,0,0};p.objective.hessian.n=3;
  QuadraticFunction h;h.name="unit_circle";h.constant=-1;h.linear={0,0,0};h.hessian.n=3;
  h.hessian.entries={{0,0,2},{1,1,2}};p.equalities={h};p.validate();return p;
}
void filter_contract(double target) {
  auto p=radial_problem();ConstraintPresolve reduced(p,true,true);
  require(reduced.has_omitted_equality_bounds(),"equality implied sides not registered");
  require(reduced.reduced().x0==p.x0,"initial point changed");
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.convexification_study="HH";st.linear_backend="mumps";st.linear_metis=true;
  st.rank_presolve_context=&reduced;st.lazy_newton_constraint_gradient_target=target;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(reduced.reduced(),st,ns,costs);
  require(w.n==2&&w.original.n==3,"fixed-coordinate test did not reduce the private dimension");
  Settings base_st=st;base_st.rank_presolve_context=nullptr;
  LazyNewtonStatistics base_ns;detail::NewtonCostLedger base_costs(base_ns,Clock::now());
  Workspace base(reduced.reduced(),base_st,base_ns,base_costs);
  Direction d;d.x={.05,-1.25};d.s.assign(w.mi,0);d.z.assign(w.mi,0);d.lambda.assign(w.me,0);
  DcaMajorantBarrierPath path(w,d),base_path(base,d);
  const double scale=std::min(1.,target);
  for(double alpha:{0.,.125,1.}) {
    const auto point=path.point(alpha,0),plain=base_path.point(alpha,0);
    const auto rounded=path.rounded_point(alpha,0),plain_rounded=base_path.rounded_point(alpha,0);
    Vec trial=w.x;for(int j=0;j<w.n;++j)trial[j]+=alpha*d.x[j];
    const auto bound=w.omitted_original_bounds(trial);
    close(point.theta-plain.theta,static_cast<double>(scale*bound.sum),"normal path omitted original sides from theta");
    close(rounded.theta-plain_rounded.theta,static_cast<double>(scale*bound.sum),"SOC rounded path omitted original sides from theta");
    require(point.omitted_bound_sides==4&&rounded.omitted_bound_sides==4,"some original sides not checked");
    close(point.omitted_bound_max,bound.maximum,"trial physical bound maximum differs");
    require(point.primal>=bound.maximum&&point.primal_internal>=scale*bound.maximum,
        "trial physical/internal primal omits a removed bound");
  }
  w.x={1.05,-1.25};base.x=w.x;w.evaluate();base.evaluate();
  const auto metric=dca_outer_metrics(w),plain_metric=dca_outer_metrics(base);
  const auto bound=w.omitted_original_bounds(w.x);
  close(metric.barrier.theta-plain_metric.barrier.theta,static_cast<double>(scale*bound.sum),
      "current outer metric omits original sides");
  require(metric.primal>=.25&&metric.barrier.primal>=.25,
      "current original primal does not include bound violation");
  const auto current=DcaMajorantBarrierPath::current(w,0),plain_current=DcaMajorantBarrierPath::current(base,0);
  close(current.theta-plain_current.theta,static_cast<double>(scale*bound.sum),
      "cached current filter point omits original sides");
  const auto invalid=w.omitted_original_bounds({INFINITY,0});
  require(!std::isfinite(invalid.maximum)&&!std::isfinite(invalid.sum),"nonfinite point escaped original bound contract");
  const auto no_contract=base.omitted_original_bounds(base.x);
  require(no_contract.sides==0&&no_contract.maximum==0&&no_contract.sum==0,"off path changed domain");
}
}
int main(){try{filter_contract(100);filter_contract(.25);std::cout<<"hybrid_original_bounds PASS\n";return 0;}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
