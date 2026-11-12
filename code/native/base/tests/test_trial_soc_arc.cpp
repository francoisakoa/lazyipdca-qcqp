#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;

int main() {
  int checks=0;
  auto check=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
  check(!Settings{}.majorant_trial_soc_arc,"legacy default changed");
  detail::NonlinearTrialSocEvidence nonlinear(1e-6),affine(1e-6);
  nonlinear.add_row(0,0,4.41,4.41,1);
  affine.add_row(.1,.05,.05,0,1);
  check(detail::trial_soc_arc_admitted(true,true,true,false,"total-correction-size",nonlinear),"size refusal not admitted");
  check(detail::trial_soc_arc_admitted(true,true,true,false,"fraction-to-boundary",nonlinear),"boundary refusal not admitted");
  check(!detail::trial_soc_arc_admitted(false,true,true,false,"total-correction-size",nonlinear),"disabled policy changed R9 fallback");
  check(!detail::trial_soc_arc_admitted(true,true,true,true,"total-correction-size",nonlinear),"successful SOC replaced");
  check(!detail::trial_soc_arc_admitted(true,true,true,false,"total-correction-size",affine),"affine row admitted");
  check(!detail::trial_soc_arc_admitted(true,true,false,false,"total-correction-size",nonlinear),"missing first correction admitted");
  check(!detail::trial_soc_arc_admitted(true,true,true,false,"audited-correction-rejected",nonlinear),"unreliable linear solve admitted");
  check(!detail::trial_soc_arc_admitted(true,false,true,false,"total-correction-size",nonlinear),"nontransition step admitted");
  detail::TrialSocArcBarrierEvent event;
  check(!event.current(0,.01),"startup incorrectly armed a barrier event");
  event.observe(3,.01,.002);
  check(event.current(3,.002),"barrier decrease did not arm its accepted generation");
  check(event.previous_mu==.01&&event.target_mu==.002&&event.generation==3,"barrier observation changed values");
  check(event.current(3,.002),"rejected direction consumed the barrier event");
  check(!event.current(4,.002),"next accepted generation retained stale barrier event");
  check(!event.current(3,.001),"changed mu retained stale barrier event");
  event.observe(4,.002,.002);
  check(!event.current(4,.002),"held barrier armed an event");
  event.observe(5,.002,.01);
  check(!event.current(5,.01),"barrier increase armed an event");
  event.observe(6,.01,.002);event.reset();
  check(!event.current(6,.002),"recovery reset retained barrier event");
  event.observe(7,.01,INFINITY);
  check(!event.armed,"nonfinite barrier armed an event");
  event.observe(-1,.01,.002);
  check(!event.armed,"invalid generation armed an event");
  event.observe(8,.01,0);
  check(!event.armed,"zero barrier armed an event");
  auto close_root=[&](long double actual,long double expected,const char* why) {
    check(std::isfinite(actual)&&std::abs(actual-expected)<=1e-15L*std::max(1.0L,std::abs(expected)),why);
  };
  close_root(detail::trial_soc_arc_boundary_limit(1,-2,0),.5L,"linear boundary root incorrect");
  check(std::isinf(detail::trial_soc_arc_boundary_limit(1,2,0)),"increasing linear path was bounded");
  close_root(detail::trial_soc_arc_boundary_limit(1,0,-1),1,"negative curvature root incorrect");
  close_root(detail::trial_soc_arc_boundary_limit(1,-3,1),(3-std::sqrt(5.0L))/2,"wrong first positive root");
  check(std::isinf(detail::trial_soc_arc_boundary_limit(1,-2,1)),"upward tangency disconnected nonnegative domain");
  check(std::isinf(detail::trial_soc_arc_boundary_limit(1,1,1)),"unbounded quadratic path was bounded");
  close_root(detail::trial_soc_arc_boundary_limit(1,2,-1),1+std::sqrt(2.0L),"large positive root branch incorrect");
  close_root(detail::trial_soc_arc_boundary_limit(1,-2,-1),std::sqrt(2.0L)-1,"small positive root branch incorrect");
  close_root(detail::trial_soc_arc_boundary_limit(1e300L,-3e300L,1e300L),(3-std::sqrt(5.0L))/2,"large coefficient normalization failed");
  close_root(detail::trial_soc_arc_boundary_limit(1e-300L,-3e-300L,1e-300L),(3-std::sqrt(5.0L))/2,"small coefficient normalization failed");
  if(std::numeric_limits<long double>::max_exponent10>650) {
    const long double exponent_scale=std::pow(10.0L,600);
    close_root(detail::trial_soc_arc_boundary_limit(1e-300L,1e300L,-1e-300L)/exponent_scale,1,"large root suffered cancellation");
    close_root(detail::trial_soc_arc_boundary_limit(1e-300L,-1e300L,-1e-300L)*exponent_scale,1,"small root suffered cancellation");
  }
  check(detail::trial_soc_arc_boundary_limit(0,-1,0)==0,"nonpositive origin boundary admitted");
  check(detail::trial_soc_arc_boundary_limit(1,INFINITY,-1)==0,"nonfinite boundary coefficient admitted");
  // A radius-four tangent has delta_x=-8. Its geometric size cap is 1/2,
  // rather than a fitted backtracking grid; the old fraction parameter gives
  // an interior proposal and the corrected circle defect is exactly 64*t^4.
  {
    struct Origin {Vec s,z;std::vector<bool> active;} origin;
    Direction tangent,delta;tangent.x={0,4};delta.x={-8,0};
    const auto scale=detail::trial_soc_arc_initial_scale(origin,tangent,delta,1,.995);
    check(scale.valid,"radius-four geometric scale invalid");
    close_root(scale.size_cap,.5L,"radius-four size cap incorrect");
    check(std::isinf(scale.boundary_cap),"empty barrier added a boundary cap");
    check(std::abs(scale.initial_t-.995*.5)<1e-15,"existing fraction safety not applied to cap");
    const double t=scale.initial_t,x=1-8*t*t,y=4*t;
    check(std::abs(x*x+y*y-1-64*t*t*t*t)<2e-12,"radius-four arc lost fourth-order circle defect");
    const auto invalid=detail::trial_soc_arc_initial_scale(origin,tangent,delta,1,1);
    check(!invalid.valid,"unit safety fraction admitted");
  }
  // Every block must reject both a nonfinite direction and overflow of the
  // realized point, particularly lambda, which is invisible to the filter.
  for(auto member:{&Direction::x,&Direction::s,&Direction::z,&Direction::lambda}) {
    Direction origin,step;origin.x=origin.s=origin.z=origin.lambda={0};
    step.x=step.s=step.z=step.lambda={0};
    check(detail::trial_soc_arc_finite_update(origin,step,.5),"finite zero update rejected");
    (step.*member)[0]=INFINITY;
    check(!detail::trial_soc_arc_finite_update(origin,step,.5),"nonfinite update block admitted");
    (step.*member)[0]=1.5e308;(origin.*member)[0]=1.5e308;
    check(!detail::trial_soc_arc_finite_update(origin,step,.75),"realized update overflow admitted");
    (step.*member).clear();
    check(!detail::trial_soc_arc_finite_update(origin,step,.5),"inconsistent update dimension admitted");
  }

  // A genuine frozen KKT solve produces a large tangent on the unit circle.
  // Its first full SOC violates the size and slack-domain guards, while a
  // damped arc removes the quadratic defect to fourth order. This exercise
  // includes a nonlinear inequality and its original barrier residual.
  QCQPProblem p;p.name="circle_and_nonlinear_slack_arc";p.n=2;p.x0={1,0};
  p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.linear={0,-3.15};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction h;h.constant=-1;h.linear={0,0};h.hessian.n=2;
  h.hessian.entries={{0,0,2},{1,1,2}};p.equalities.push_back(h);
  QuadraticFunction g;g.constant=-4;g.linear={0,0};g.hessian.n=2;
  g.hessian.entries={{1,1,2}};p.inequalities.push_back(g);
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.linear_backend="mumps";st.linear_metis=true;st.verbose=false;st.live_display_level=0;
  st.lazy_newton_reduction=false;st.lazy_newton_atomic_regularization=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_objective_gradient_target=1000;st.lazy_newton_constraint_gradient_target=1000;
  LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());
  Workspace w(p,st,stats,costs);w.s={4};w.z={.25};w.lambda={0};w.evaluate();w.update_matrix();
  check(w.factorize(0),"fixture KKT factorization failed");
  const auto anchor_x=w.x,anchor_s=w.s,anchor_z=w.z,anchor_lambda=w.lambda;
  const auto attempts=stats.factor_attempts;auto* factor=w.current_factor;const double tau=w.regularization;
  const auto retained=w.retained;
  const Direction source=w.direction({0},w.rd,w.h,w.rp,0,true,DirectionPurpose::Complete,true);
  check(std::abs(source.x[0])<1e-12&&std::abs(source.x[1]-2.1)<1e-12,"large tangent fixture changed");
  Vec source_x=w.x,source_s=w.s;axpy(1,source.x,source_x);axpy(1,source.s,source_s);
  const double eq_rhs=w.equalities[0].original_at(source_x);
  const double primal_rhs=w.inequalities[0].original_at(source_x)+source_s[0];
  const Direction increment=w.direction({0},Vec(w.n,0),{eq_rhs},{primal_rhs},0,true,DirectionPurpose::Complete,true);
  check(inf_norm(increment.x)>inf_norm(source.x),"fixture did not refuse full SOC size");
  Direction full=source;
  axpy(1,increment.x,full.x);axpy(1,increment.s,full.s);axpy(1,increment.z,full.z);axpy(1,increment.lambda,full.lambda);
  check(w.fraction(full,st.lazy_newton_fraction_to_boundary)<1,"fixture did not refuse full SOC domain");
  const auto geometric=detail::trial_soc_arc_initial_scale(w,source,increment,1,st.lazy_newton_fraction_to_boundary);
  check(geometric.valid,"true KKT arc geometric scale invalid");
  const auto initial_arc=detail::trial_soc_arc_direction(source,increment,1,geometric.initial_t);
  check(w.fraction(initial_arc,st.lazy_newton_fraction_to_boundary)>=geometric.initial_t,"geometric initial arc failed actual boundary guard");
  check(geometric.initial_t>0&&geometric.initial_t<geometric.boundary_cap,"geometric initial arc omitted interior safety");
  const DcaMajorantBarrierPath source_path(w,source);
  const auto from=DcaMajorantBarrierPath::current(w,1);
  DcaMajorantBarrierFilter filter(from.theta);
  check(filter.diagnose(from,source_path.rounded_point(1,1),source_path.slope(1),1,1e-4).accepted,"source filter fixture changed");
  double previous_h=INFINITY;
  for(double t:{.5,.25,.125}) {
    const auto arc=detail::trial_soc_arc_direction(source,increment,1,t);
    const double a=t;
    check(w.fraction(arc,st.lazy_newton_fraction_to_boundary)>=a,"damped arc still outside slack domain");
    Vec x=w.x,s=w.s,z=w.z,lambda=w.lambda,source_same_x=w.x;
    axpy(a,arc.x,x);axpy(a,arc.s,s);axpy(a,arc.z,z);axpy(a,arc.lambda,lambda);axpy(a,source.x,source_same_x);
    auto identity=[&](const Vec& origin,const Vec& d,const Vec& delta,const Vec& actual) {
      for(std::size_t j=0;j<origin.size();++j)
        check(std::abs(actual[j]-(origin[j]+t*d[j]+t*t*delta[j]))<2e-13,"primal-dual arc scaling identity failed");
    };
    identity(w.x,source.x,increment.x,x);identity(w.s,source.s,increment.s,s);
    identity(w.z,source.z,increment.z,z);identity(w.lambda,source.lambda,increment.lambda,lambda);
    const double actual_h=std::abs(w.equalities[0].original_at(x));
    check(std::abs(actual_h-.25*std::pow(t*2.1,4))<2e-12,"circle arc is not fourth-order corrected");
    check(actual_h<previous_h,"arc equality residual failed to shrink");previous_h=actual_h;
    check(actual_h<std::abs(w.equalities[0].original_at(source_same_x)),"arc not better than same-alpha source");
    check(std::abs(w.inequalities[0].original_at(x)+s[0])<2e-12,"full primal slack correction failed");
    const DcaMajorantBarrierPath arc_path(w,arc);
    const auto point=arc_path.rounded_point(a,1),source_same=source_path.rounded_point(a,1);
    check(point.positive&&point.finite&&point.theta<source_same.theta,"arc original theta/domain check failed");
    check(filter.diagnose(from,point,source_path.slope(1),a,1e-4).accepted,"arc original filter rejected");
    check(filter.size()==0,"private arc changed filter history");
    check(stats.factor_attempts==attempts&&w.current_factor==factor&&w.regularization==tau&&w.retained==retained,"private arc changed frozen factor/model");
    check(w.x==anchor_x&&w.s==anchor_s&&w.z==anchor_z&&w.lambda==anchor_lambda,"private arc mutated accepted anchor");
  }
  // Source alpha is not necessarily one. All four update blocks must retain
  // alpha_new=t*alpha_source rather than silently using a different dual step.
  for(double alpha:{.5,.2}) {
    const double t=.25,a=t*alpha;
    const auto arc=detail::trial_soc_arc_direction(source,increment,alpha,t);
    for(std::size_t j=0;j<source.x.size();++j)
      check(std::abs(a*arc.x[j]-(t*alpha*source.x[j]+t*t*increment.x[j]))<1e-13,"nonunit source-alpha arc identity failed");
  }
  bool refused=false;
  try{(void)detail::trial_soc_arc_direction(source,increment,0,.5);}catch(const std::invalid_argument&){refused=true;}
  check(refused,"invalid scale admitted");
  std::cout<<"trial_soc_arc PASS checks="<<checks
      <<" genuine_frozen_KKT=1 size_and_domain_refusal=1 same_alpha_comparison=1 fourth_order_circle=1 private_anchor=1\n";
}
