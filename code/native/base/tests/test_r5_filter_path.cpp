#include "../src/atomic_newton.cpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const std::string& message) {
  if(!ok)throw std::runtime_error(message);
}
void close(double actual,double expected,const std::string& name) {
  const double tolerance=2e-12*std::max({1.0,std::abs(actual),std::abs(expected)});
  check(std::isfinite(actual)&&std::isfinite(expected)&&std::abs(actual-expected)<=tolerance,
        name+": actual="+std::to_string(actual)+" expected="+std::to_string(expected));
}
void same_point(const DcaPoint& a,const DcaPoint& b,const std::string& context) {
  close(a.objective,b.objective,context+" objective");
  close(a.theta,b.theta,context+" theta");
  close(a.lagrangian,b.lagrangian,context+" frozen-multiplier Lagrangian");
  close(a.current_lagrangian,b.current_lagrangian,context+" current Lagrangian");
  close(a.primal,b.primal,context+" original-unit feasibility");
  close(a.dual,b.dual,context+" original-unit stationarity");
  close(a.comp,b.comp,context+" original-unit complementarity");
  close(a.kkt,b.kkt,context+" KKT");
  close(a.primal_inf,b.primal_inf,context+" primal infinity norm");
  close(a.normalized_kkt,b.normalized_kkt,context+" normalized KKT");
  close(a.guard_reference_kkt,b.guard_reference_kkt,context+" guard reference");
  check(a.within_acceptable_caps==b.within_acceptable_caps,context+" acceptable caps");
  check(a.distinct_from_anchor==b.distinct_from_anchor,context+" anchor identity");
  check(a.feasible_reference_initialized==b.feasible_reference_initialized,
        context+" feasible reference");
  check(!a.acceptable_certified,context+" polynomial trial falsely became a certificate");
}
std::vector<std::uint64_t> counters(const LazyNewtonStatistics& statistics) {
  std::vector<std::uint64_t> out;
  for(const auto& phase:statistics.calls)for(const auto& family:phase) {
    out.insert(out.end(),{family.model_values,family.model_gradients,family.values,
      family.gradients,family.cached_gradient_accumulations,family.jacobian_products,
      family.hessian_assemblies,family.hessian_products});
  }
  return out;
}
QCQPProblem problem() {
  QCQPProblem p;p.name="r5_exact_original_filter_path";p.n=3;
  p.x0={1.7,-0.4,0.8};p.lb={-2,-3,-INFINITY};p.ub={4,INFINITY,2};
  p.objective.constant=7.3;p.objective.linear={3.7,-2.1,0.6};
  p.objective.hessian.n=3;p.objective.hessian.entries={{0,0,-4},{0,1,1.3},{1,1,3},{2,2,2}};
  QuadraticFunction h;h.constant=-2.3;h.linear={0.2,-2,1};h.hessian.n=3;
  h.hessian.entries={{0,0,2},{0,2,0.7},{1,1,-0.4}};p.equalities.push_back(h);
  QuadraticFunction h2;h2.constant=0.3;h2.linear={1.7,0.3,-0.8};h2.hessian.n=3;
  h2.hessian.entries={{1,2,0.9}};p.equalities.push_back(h2);
  QuadraticFunction g;g.constant=1.1;g.linear={0.4,-1,2};g.hessian.n=3;
  g.hessian.entries={{0,0,-3},{0,1,0.4},{1,1,0.5},{2,2,1.7}};p.inequalities.push_back(g);
  QuadraticFunction g2;g2.constant=-3;g2.linear={1,0.2,-0.6};g2.hessian.n=3;
  g2.hessian.entries={{0,0,1},{0,2,0.3},{2,2,2}};p.inequalities.push_back(g2);
  p.validate();return p;
}
Settings settings() {
  Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
  s.linear_backend="native";s.dca_filter_curvature_backend="native";
  s.lazy_newton_atomic_regularization=false;s.lazy_newton_lazy_constraints=false;
  s.lazy_newton_reduction=false;s.verbose=false;s.live_display_level=0;
  s.lazy_newton_objective_gradient_target=0.4;
  s.lazy_newton_constraint_gradient_target=0.25;
  return s;
}
void polynomial_filter_matches_original() {
  for(bool lean:{false,true})for(const std::string termination:{"ipopt","raw"}) {
    auto p=problem();auto st=settings();st.dca_aggregate_lean=lean;st.termination_test=termination;
    // Exercise both sides of acceptable-cap normalization, while never
    // admitting a polynomial evaluation as an independent certificate.
    for(bool wide_caps:{false,true}) {
      if(wide_caps) {
        st.acceptable_tolerance=1e6;st.acceptable_constraint_tolerance=1e6;
        st.acceptable_dual_tolerance=1e6;st.acceptable_complementarity_tolerance=1e6;
      }
      LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
      Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
      DcaInequalityModels models(w,stats);models.build();
      check(w.objective.scale<1 && w.equalities[0].scale<1 &&
            w.inequalities[0].scale<1 && w.inequalities.back().scale<1,
            "filter path fixture does not exercise nontrivial objective/row/bound scaling");
      check(w.inequalities[0].model_active && !w.objective.model_active &&
            !w.equalities[0].model_active,"filter fixture changed the majorant formulation");
      w.lambda={-2.1,0.8};
      for(int i=0;i<w.mi;++i){w.z[i]=0.3+0.05*i;w.s[i]=0.4+0.1*i;}
      // Inactive rows still count for feasibility, but not stationarity or
      // complementarity. This also tests generated bound family normalization.
      w.active.back()=0;
      w.evaluate();w.update_matrix();
      Direction d;d.x={-0.63,0.71,-0.22};d.lambda={0.37,-0.19};
      d.z.resize(w.mi);d.s.resize(w.mi);
      for(int i=0;i<w.mi;++i){d.z[i]=0.011*(i+1);d.s[i]=-0.014*(i+1);}
      const auto before=counters(ns);
      const auto old_x=w.x,old_lambda=w.lambda,old_z=w.z,old_s=w.s;
      DcaMajorantFilterPath path(w,d);
      const auto anchor=DcaMajorantFilterPath::current(w);
      check(counters(ns)==before,"polynomial preparation reevaluated original functions");
      Vec grad(w.n,0);w.objective.original_gradient_at(w.x,1,grad);
      close(path.objective_slope(),dot(grad,d.x),"exact original objective slope");
      const auto auxiliary_hessian=w.hessian_product(Vec{0.7,-0.4,0.2});
      for(double alpha:{0.0,0.03125,0.25,0.63,1.0}) {
        const auto calls_before=counters(ns);
        const auto actual=path.point(alpha);
        check(counters(ns)==calls_before,"polynomial trial repeated values/gradients/Hessians");
        auto x=w.x,lambda=w.lambda,z=w.z;
        axpy(alpha,d.x,x);axpy(alpha,d.lambda,lambda);axpy(alpha,d.z,z);
        const auto expected=dca_original(w,x,lambda,z,&w.lambda,&w.z);
        same_point(actual,expected,"alpha="+std::to_string(alpha));
        if(alpha==0)same_point(anchor,expected,"cached current point");
      }
      check(w.x==old_x&&w.lambda==old_lambda&&w.z==old_z&&w.s==old_s,
            "original filter trials mutated primal-dual state");
      check(w.inequalities[0].model_active&&
            auxiliary_hessian==w.hessian_product(Vec{0.7,-0.4,0.2}),
            "original filter evaluation replaced the auxiliary Hessian");
    }
  }
}
QCQPProblem barrier_problem() {
  QCQPProblem p;p.name="r7_exact_original_barrier_path";p.n=4;
  p.x0={1.1,0.7,-0.2,0.8};p.lb={-2,-3,-INFINITY,-INFINITY};p.ub={4,INFINITY,2,3};
  p.objective.constant=1.3;p.objective.linear={0.7,-0.1,0.6,0.2};
  p.objective.hessian.n=4;p.objective.hessian.entries={{0,0,-2},{0,1,0.3},{1,1,3},{2,2,2},{3,3,1}};
  QuadraticFunction h;h.constant=-1.2;h.linear={0.2,-2,1,0.3};h.hessian.n=4;
  h.hessian.entries={{0,0,2},{0,2,0.7},{1,1,-0.4}};p.equalities.push_back(h);
  QuadraticFunction g;g.constant=0.4;g.linear={0.3,-0.2,0.1,-0.7};g.hessian.n=4;
  g.hessian.entries={{0,2,-0.3},{0,3,-1},{1,2,1},{1,3,-0.3}};p.inequalities.push_back(g);
  p.validate();return p;
}
DcaMajorantBarrierPoint direct_barrier_point(Workspace& w,const Direction& d,double alpha,double mu) {
  auto x=w.x,slack=w.s,multiplier=w.z;
  axpy(alpha,d.x,x);axpy(alpha,d.s,slack);axpy(alpha,d.z,multiplier);
  DcaMajorantBarrierPoint p;
  p.objective=w.objective.original_at(x);p.phi=p.objective;
  for(int e=0;e<w.me;++e) {
    const double residual=std::abs(w.equalities[e].original_at(x));
    p.theta+=residual;p.primal=std::max(p.primal,residual/w.equalities[e].scale);
    p.primal_internal=std::max(p.primal_internal,residual);
  }
  for(int i=0;i<w.mi;++i) {
    const double value=w.inequalities[i].original_at(x);
    const double residual=w.active[i]?std::abs(value+slack[i]):std::max(0.0,value);
    p.theta+=residual;p.primal=std::max(p.primal,residual/w.inequalities[i].scale);
    p.primal_internal=std::max(p.primal_internal,residual);
    if(w.active[i]) {
      p.positive=p.positive&&slack[i]>0&&multiplier[i]>0;
      if(slack[i]>0)p.phi-=mu*std::log(slack[i]);
    }
  }
  if(!p.positive){p.phi=INFINITY;p.finite=false;}
  return p;
}
void same_barrier_point(const DcaMajorantBarrierPoint& actual,
                        const DcaMajorantBarrierPoint& expected,const std::string& context) {
  close(actual.objective,expected.objective,context+" objective");
  close(actual.phi,expected.phi,context+" barrier objective");
  close(actual.theta,expected.theta,context+" augmented feasibility sum");
  close(actual.primal,expected.primal,context+" original-unit primal norm");
  close(actual.primal_internal,expected.primal_internal,context+" internal primal norm");
  check(actual.positive==expected.positive&&actual.finite==expected.finite,
        context+" positivity/finiteness mismatch");
}
void polynomial_barrier_matches_original_with_majorants_active() {
  auto p=barrier_problem();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  check(models.inequality[0].half_spectral,"barrier path fixture lacks a matrix majorant correction");
  check(w.objective.scale<1&&w.equalities[0].scale<1&&w.inequalities[0].scale<1,
        "barrier path fixture lacks nontrivial frozen scaling");
  w.lambda={-1.7};
  for(int i=0;i<w.mi;++i){w.s[i]=0.7+0.1*i;w.z[i]=0.3+0.07*i;}
  w.active.back()=0;w.evaluate();w.update_matrix();
  Direction d;d.x={-0.63,0.71,-0.22,0.28};d.lambda={0.37};
  d.s.resize(w.mi);d.z.resize(w.mi);
  for(int i=0;i<w.mi;++i){d.s[i]=-0.03*(i+1);d.z[i]=0.01*(i+1);}
  // Inactive slacks are absent from the barrier and positivity constraint.
  d.s.back()=-100;d.z.back()=-100;
  const auto state_x=w.x,state_s=w.s,state_z=w.z;
  const auto before=counters(ns);
  DcaMajorantBarrierPath path(w,d);
  check(counters(ns)==before,"barrier path preparation repeated function evaluations");
  const auto auxiliary_hessian=w.hessian_product(Vec{0.7,-0.4,0.2,0.3});
  auto separated_trial=w.x;axpy(1,d.x,separated_trial);
  check(w.inequalities[0].at(separated_trial)-w.inequalities[0].original_at(separated_trial)>0.01,
        "original filter fixture does not distinguish the inequality from its majorant");
  Vec objective_gradient(w.n,0);w.objective.add_gradient_at(w.x,1,objective_gradient);
  for(double mu:{0.0,0.012,0.1,2.0}) {
    double expected_slope=dot(objective_gradient,d.x);
    for(int i=0;i<w.mi;++i)if(w.active[i])expected_slope-=mu*d.s[i]/w.s[i];
    close(path.slope(mu),expected_slope,"exact original barrier slope");
    for(double alpha:{0.0,0.2,0.7,1.0}) {
      const auto calls=counters(ns);
      const auto actual=path.point(alpha,mu);
      Vec committed,expected_slacks=w.s;
      path.write_slacks(alpha,committed);axpy(alpha,d.s,expected_slacks);
      check(committed==expected_slacks,"filter commit changed the DOUBLE affine slack tuple");
      for(int i=0;i<w.mi;++i)
        check(committed[i]==path.slack(i,alpha),"filter and commit disagree on canonical slack rounding");
      check(counters(ns)==calls,"barrier trial/commit repeated values/gradients/Hessians");
      const auto expected=direct_barrier_point(w,d,alpha,mu);
      same_barrier_point(actual,expected,"barrier alpha="+std::to_string(alpha));
      if(alpha==1) {
        auto trial_x=w.x;axpy(alpha,d.x,trial_x);
        const double trial_s=w.s[0]+alpha*d.s[0];
        const double auxiliary_theta=expected.theta
            -std::abs(w.inequalities[0].original_at(trial_x)+trial_s)
            +std::abs(w.inequalities[0].at(trial_x)+trial_s);
        check(std::abs(actual.theta-auxiliary_theta)>0.01,
              "filter feasibility still contains the convexified inequality");
      }
      const auto initial=direct_barrier_point(w,d,0,mu);
      close(actual.phi_change,expected.phi-initial.phi,"exact barrier objective change");
      if(alpha==0)same_barrier_point(DcaMajorantBarrierPath::current(w,mu),expected,"cached barrier point");
    }
  }
  check(w.x==state_x&&w.s==state_s&&w.z==state_z,"barrier trials mutated the primal-dual state");
  check(auxiliary_hessian==w.hessian_product(Vec{0.7,-0.4,0.2,0.3})&&w.inequalities[0].model_active,
        "barrier globalization altered the auxiliary Hessian");
  for(bool negative_slack:{false,true}) {
    auto invalid=d;
    if(negative_slack)invalid.s[0]=-2*w.s[0];else invalid.z[0]=-2*w.z[0];
    DcaMajorantBarrierPath invalid_path(w,invalid);
    const auto trial=invalid_path.point(1,0.1);
    check(!trial.positive&&!trial.finite,
          "barrier path admitted a nonpositive active slack or multiplier");
  }
}
void affine_slacks_retain_original_quadratic_residual() {
  QCQPProblem p;p.name="r9_affine_commit_not_quadratic_slack_transport";p.n=1;
  p.x0={2};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};
  p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,2}};
  QuadraticFunction g;g.constant=1;g.linear={0};g.hessian.n=1;
  g.hessian.entries={{0,0,-2}};p.inequalities.push_back(g);p.validate();
  auto st=settings();st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  check(w.inequalities[0].scale==1,"affine analytic fixture changed units");
  w.s={3};w.z={0.1};w.evaluate();
  Direction d;d.x={-0.5};d.s={-2};d.z={0};
  const auto calls=counters(ns);DcaMajorantBarrierPath path(w,d);
  for(double alpha:{0.0,0.125,0.5,1.0}) {
    Vec committed;path.write_slacks(alpha,committed);
    const double x=2-0.5*alpha;
    check(committed[0]==3-2*alpha&&committed[0]==path.slack(0,alpha),
          "affine slack secretly contains a quadratic correction");
    close(1-x*x+committed[0],-0.25*alpha*alpha,
          "affine commit incorrectly cancels original quadratic curvature");
    close(path.point(alpha,0.1).theta,0.25*alpha*alpha,
          "filter does not evaluate the original inequality with the committed affine slack");
  }
  check(counters(ns)==calls,"affine curve/commit repeated original evaluations");
}

void rounded_slack_change_cannot_invent_barrier_progress() {
  QCQPProblem p;p.name="r9_sub_ulp_slack_commit";p.n=1;
  p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};p.objective.hessian.n=1;
  const double large=std::ldexp(1.0,53);
  QuadraticFunction g;g.constant=-large;g.linear={0};g.hessian.n=1;
  p.inequalities.push_back(g);p.validate();auto st=settings();
  st.lazy_newton_constraint_gradient_target=100;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();w.s={large};w.z={1};w.evaluate();
  Direction d;d.x={0};d.s={0.5};d.z={0};
  const auto calls=counters(ns);DcaMajorantBarrierPath path(w,d);
  Vec committed;path.write_slacks(1,committed);
  check(committed[0]==w.s[0]&&committed[0]==path.slack(0,1),
        "sub-ulp fixture did not preserve the committed double slack");
  const auto trial=path.point(1,1);
  check(trial.finite&&trial.phi_change==0,
        "filter invented barrier progress below the committed slack's precision");
  check(counters(ns)==calls,"rounded slack handling repeated original evaluations");
}

void barrier_changes_survive_large_objective_offset() {
  QCQPProblem p;p.name="r5_barrier_change_without_objective_cancellation";p.n=1;
  p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective.constant=1e16;
  p.objective.linear={1};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,1}};
  QuadraticFunction g;g.constant=-1;g.linear={0};g.hessian.n=1;p.inequalities.push_back(g);
  p.validate();auto st=settings();st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  w.s[0]=100;w.z[0]=0.001;w.evaluate();
  Direction objective_step;objective_step.x={-1e-7};objective_step.s={0};objective_step.z={0};
  DcaMajorantBarrierPath objective_path(w,objective_step);
  const auto from=DcaMajorantBarrierPath::current(w,0.1),to=objective_path.point(1,0.1);
  check(to.phi==from.phi,"large objective fixture does not expose subtraction cancellation");
  close(to.phi_change,-1e-7+0.5e-14,"small objective change under large constant offset");
  check(to.phi_change<0,"objective cancellation erased the descent signal");
  Direction slack_step;slack_step.x={0};slack_step.s={-99};slack_step.z={0.00099};
  DcaMajorantBarrierPath slack_path(w,slack_step);
  const auto slack_to=slack_path.point(1,0.1);
  close(slack_to.phi_change,0.1*std::log(100.0),"barrier log-ratio under large objective offset");
  check(slack_to.phi_change>0,"objective cancellation erased the barrier increase");
}

void augmented_feasibility_improvement_can_increase_original_complementarity() {
  QCQPProblem p;p.name="r5_slack_feasibility_vs_original_complementarity";p.n=1;
  p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1}};
  QuadraticFunction g;g.constant=-1;g.linear={0};g.hessian.n=1;p.inequalities.push_back(g);
  p.validate();auto st=settings();st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  w.s[0]=100;w.z[0]=0.001;w.evaluate();
  Direction d;d.x={0};d.s={-99};d.z={0.00099};
  constexpr double mu=0.1;
  DcaMajorantBarrierPath path(w,d);
  const auto from=DcaMajorantBarrierPath::current(w,mu),to=path.point(1,mu);
  close(from.theta,99,"counterexample initial augmented residual");
  close(to.theta,0,"counterexample corrected augmented residual");
  check(to.phi>from.phi,"counterexample must increase the barrier objective");
  const double original_comp0=std::abs(w.z[0]*w.inequalities[0].original_at(w.x));
  const double original_comp1=std::abs((w.z[0]+d.z[0])*w.inequalities[0].original_at(w.x));
  check(original_comp1>original_comp0,"counterexample must increase original complementarity");
  DcaMajorantBarrierFilter filter(from.theta);
  const auto decision=filter.diagnose(from,to,path.slope(mu),1,st.line_search_armijo);
  check(decision.accepted&&decision.feasibility,
        "augmented-feasibility filter rejected a valid slack correction because phi increased");
  // The filter controls the step; it does not certify the final solution.
  auto x=w.x,z=w.z;axpy(1,d.z,z);
  check(dca_original(w,x,w.lambda,z).comp>st.complementarity_tolerance,
        "accepted slack correction was incorrectly treated as original optimality");
}

void barrier_objective_progress_has_its_own_filter_margin() {
  // Cold-start line-search point: augmented feasibility improves only
  // slightly, while the barrier improvement is meaningful on its own scale.
  DcaMajorantBarrierPoint from;from.theta=904;from.phi=0;
  auto to=from;constexpr double alpha=4.7e-7;
  to.theta=from.theta*(1-alpha);to.phi=to.phi_change=-0.00014963;
  DcaMajorantBarrierFilter filter(from.theta);
  const auto local=filter.diagnose(from,to,-18195,alpha,1e-4);
  check(local.accepted&&local.objective&&!local.feasibility,
        "cold-start barrier decrease was rejected using the larger feasibility margin");
  filter.insert(from);
  const auto historical=filter.diagnose(from,to,-18195,alpha,1e-4);
  check(historical.accepted&&historical.objective,
        "filter history retained the wrong feasibility margin for the barrier objective");
}

void variable_barrier_filter_preserves_history_and_reweights_exactly() {
  DcaMajorantBarrierPoint history;history.theta=1;history.objective=1;history.slack_logs=2;
  history.phi=history.objective-0.1*history.slack_logs;
  DcaMajorantBarrierFilter filter(history.theta,true);filter.insert(history);
  // Locally improving but historically dominated; changing a private inner
  // barrier must not erase the previously achieved feasibility/objective pair.
  DcaMajorantBarrierPoint from;from.theta=2;from.objective=4;from.slack_logs=0;from.phi=4;
  auto to=from;to.theta=1.5;to.objective=2;to.phi=2;to.phi_change=-2;
  check(std::string(filter.diagnose(from,to,-2,1,1e-4).reason)=="filter-history",
        "dominated trial bypassed initial history");
  filter.rebase(0.3);
  check(filter.size()==1&&std::string(filter.diagnose(from,to,-2,1,1e-4).reason)=="filter-history",
        "barrier continuation cleared original filter history");
  // Reweighting is necessary: at mu=2 the historical phi is -3, not 0.8.
  to.phi=0;to.phi_change=-4;
  filter.rebase(2);
  check(std::string(filter.diagnose(from,to,-4,1,1e-4).reason)=="filter-history",
        "retained history was compared at a stale barrier parameter");
  // A point dominating at one barrier cannot delete an entry which wins at
  // another. Objective/log ordering certifies domination for every mu>=0.
  auto crossing=history;crossing.objective=0;crossing.slack_logs=0;crossing.phi=0;
  filter.insert(crossing);check(filter.size()==2,"one-mu dominance incorrectly pruned a crossing history");
  auto uniform=crossing;uniform.theta=0.5;uniform.objective=-1;uniform.slack_logs=3;uniform.phi=-7;
  filter.insert(uniform);check(filter.size()==1,"uniformly dominated history was not pruned");
}

void phase1_progress_is_filter_guarded_and_not_an_original_certificate() {
  QCQPProblem p;p.name="phase1_original_filter_scope";p.n=1;p.x0={2};
  p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1}};
  QuadraticFunction g;g.constant=-100;g.linear={0};g.hessian.n=1;g.hessian.entries={{0,0,200}};
  p.inequalities.push_back(g);p.validate();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(p,st,ns,costs);
  w.s[0]=w.inequalities[0].scale;w.z[0]=0.1;w.evaluate();
  const auto original=dca_outer_metrics(w);const auto from=original.filter_point(0);
  const double target=detail::phase1_feasibility_target(original.primal,st.feasibility_tolerance);
  w.x[0]=1.1;w.s[0]=16*std::numeric_limits<double>::epsilon()*w.inequalities[0].scale;w.evaluate();
  const auto candidate=dca_outer_metrics(w);auto to=candidate.filter_point(0);
  to.phi_change=to.objective-from.objective;
  check(candidate.primal<=target&&candidate.primal>st.feasibility_tolerance,
      "phase-I target confused with the final original tolerance");
  DcaMajorantBarrierFilter filter(from.theta,true);
  check(filter.diagnose(from,to,0,1,st.line_search_armijo).accepted,
      "fresh original phase-I reduction rejected by an empty original filter");
  check(dca_original(w,w.x,w.lambda,w.z).primal>st.feasibility_tolerance,
      "phase-I progress was incorrectly promoted to an original KKT certificate");
  auto dominating=to;dominating.theta=to.theta*0.5;dominating.objective=dominating.phi=to.phi-1;
  filter.insert(dominating);
  check(std::string(filter.diagnose(from,to,0,1,st.line_search_armijo).reason)=="filter-history",
      "phase-I reduction bypassed previously accepted original filter history");
}

void filter_requires_local_and_historical_progress() {
  detail::OriginalDcaFilter filter(20,1e-7);
  detail::OriginalFilterPoint from,candidate;
  from.objective=10;from.theta=1;from.kkt=10;
  candidate=from;candidate.objective-=1e-9;
  auto insufficient=filter.diagnose(from,candidate,1,1e-7);
  check(!insufficient.accepted&&std::string(insufficient.reason)=="local-progress",
        "nondomination alone bypassed sufficient decrease");
  candidate.objective=9;
  check(filter.diagnose(from,candidate,1,1e-7).accepted,
        "sufficient original-objective decrease was rejected by an empty filter");
  detail::OriginalFilterPoint historical;historical.objective=0;historical.theta=1;
  historical.kkt=10;filter.insert(historical);
  from.objective=10;from.theta=5;candidate.objective=9;candidate.theta=2;
  const auto blocked=filter.diagnose(from,candidate,1,1e-7);
  check(!blocked.accepted&&blocked.history_checked&&
        std::string(blocked.reason)=="filter-history"&&blocked.history_index==0,
        "local improvement bypassed a dominating historical filter entry");
  candidate.theta=0.5;
  check(filter.diagnose(from,candidate,1,1e-7).accepted,
        "sufficient feasibility decrease failed valid original filter history");
}
void geometric_outer_path_keeps_actual_slacks_and_original_guards() {
  QCQPProblem p;p.name="geometric_original_filter";p.n=1;p.x0={.1};
  p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={1};p.objective.hessian.n=1;
  QuadraticFunction g;g.constant=-1;g.linear={1};g.hessian.n=1;p.inequalities.push_back(g);
  auto st=settings();st.lazy_newton_objective_gradient_target=1000;st.lazy_newton_constraint_gradient_target=1000;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(p,st,ns,costs);
  w.s[0]=.5;w.z[0]=1;w.evaluate();
  Direction d;d.x={.1};d.s={.5};d.z={.1};
  const auto old_x=w.x,old_s=w.s,old_z=w.z;const auto factors=ns.factor_attempts;
  DcaMajorantBarrierPath affine(w,d),geometric(w,d,true);
  check(!affine.geometric_slacks&&geometric.geometric_slacks,"slack path selection failed");
  check(affine.slack(0,1)==geometric.slack(0,1)&&geometric.slack(0,0)==w.s[0],"curved path changed a canonical endpoint");
  close(geometric.slack(0,.5),std::sqrt(.5),"geometric midpoint");
  close(geometric.slope(1),.1-std::log(2.),"exact curved original-barrier slope");
  Vec actual;geometric.write_slacks(.5,actual);
  HybridInnerOutcome model;model.proximal_coordinates=w.retained;model.objective_scale=w.objective.scale;
  const auto curved_gate=hybrid_outer_inner_gate(w,d,.5,1e-4,st,model,&actual);
  const auto affine_gate=hybrid_outer_inner_gate(w,d,.5,1e-4,st,model);
  close(curved_gate.final.primal,.85-std::sqrt(.5),"model gate used actual curved slacks");
  close(affine_gate.final.primal,.10,"default model gate remains affine");
  check(w.x==old_x&&w.s==old_s&&w.z==old_z&&ns.factor_attempts==factors,"path or model trial changed anchor/factors");
  d.z[0]=-2;DcaMajorantBarrierPath bad_dual(w,d,true);
  check(!bad_dual.point(1,1e-4).positive,"geometric slacks hid a nonpositive affine dual");
  const double a=w.fraction(d,.995,2);
  check(bad_dual.point(a,1e-4).positive,"dual-only fraction guard inconsistent with curved slacks");
  d.s[0]=-1;DcaMajorantBarrierPath bad_slack(w,d,true);
  check(!bad_slack.geometric_slacks&&!bad_slack.point(1,1e-4).positive,"invalid geometric endpoint bypassed domain guard");
  auto bounded=p;bounded.inequalities.clear();bounded.lb={-1};bounded.ub={1};
  bounded.objective.hessian.entries={{0,0,2}};
  const auto convex=study::construct_hybrid(1,bounded.objective.hessian.entries);
  check(convex.original_convex&&convex.rho==0&&convex.atoms==0,"convex source was corrected for a curved slack path");
  Workspace bw(bounded,st,ns,costs);std::fill(bw.active.begin(),bw.active.end(),0);bw.evaluate();
  Direction bd;bd.x={3};bd.s.assign(bw.mi,0);bd.z.assign(bw.mi,0);
  DcaMajorantBarrierPath bounded_curve(bw,bd,true);
  close(bounded_curve.point(1,0).primal,2.1,"curved path omitted an inactive original finite bound");
}

}
int main() {
  try {
    geometric_outer_path_keeps_actual_slacks_and_original_guards();
    polynomial_filter_matches_original();
    filter_requires_local_and_historical_progress();
    variable_barrier_filter_preserves_history_and_reweights_exactly();
    phase1_progress_is_filter_guarded_and_not_an_original_certificate();
    barrier_objective_progress_has_its_own_filter_margin();
    polynomial_barrier_matches_original_with_majorants_active();
    affine_slacks_retain_original_quadratic_residual();
    rounded_slack_change_cannot_invent_barrier_progress();
    barrier_changes_survive_large_objective_offset();
    augmented_feasibility_improvement_can_increase_original_complementarity();
    std::cout<<"PASS R9 affine filter: exact original QCQP polynomials, scaling and inactive bounds, "
      "original barrier/slack filter, no repeated evaluations, unchanged auxiliary Hessian and sufficient decrease\n";
  } catch(const std::exception& e) {
    std::cerr<<"FAIL R9 affine filter path: "<<e.what()<<'\n';return 1;
  }
}
