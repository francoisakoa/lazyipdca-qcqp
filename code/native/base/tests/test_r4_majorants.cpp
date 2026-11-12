#include "../src/atomic_newton.cpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef LAZYIPDCA_HAVE_OPENMP
#include <omp.h>
#endif

using namespace lazyipdca::qcqp;

namespace {
// Test-only characterization of auxiliary KKT; absent from the R6 engine.
struct DcaAuxiliaryMetrics {
  double primal=0,dual=0,complementarity=0,function_complementarity=0,kkt=0;
  bool finite=true;
};

DcaAuxiliaryMetrics dca_auxiliary_metrics(const Workspace& w) {
  DcaAuxiliaryMetrics m;
  m.dual=inf_norm(w.rd)/w.objective.scale;
  for(int e=0;e<w.me;++e)
    m.primal=std::max(m.primal,std::abs(w.h[e])/w.equalities[e].scale);
  for(int i=0;i<w.mi;++i) {
    const auto& q=w.inequalities[i];
    m.primal=std::max(m.primal,std::max(0.0,q.value)/q.scale);
    if(w.active[i]) {
      m.primal=std::max(m.primal,std::abs(w.rp[i])/q.scale);
      m.complementarity=std::max(m.complementarity,std::abs(w.s[i]*w.z[i])/w.objective.scale);
      m.function_complementarity=std::max(m.function_complementarity,
          std::abs(q.value*w.z[i])/w.objective.scale);
      if(!(w.s[i]>0) || !(w.z[i]>=0))m.finite=false;
    }
  }
  m.kkt=std::max({m.primal,m.dual,m.complementarity,m.function_complementarity});
  m.finite=m.finite&&std::isfinite(m.kkt)&&std::isfinite(m.dual)&&
      finite(w.rd)&&finite(w.h)&&finite(w.rp)&&finite(w.x)&&finite(w.s)&&finite(w.z)&&finite(w.lambda);
  if(!m.finite)m.kkt=INFINITY;
  return m;
}

bool dca_auxiliary_converged(const DcaAuxiliaryMetrics& m,const Settings& st) {
  return m.finite&&m.primal<=st.feasibility_tolerance&&
      m.dual<=st.dual_infeasibility_tolerance&&
      m.complementarity<=st.complementarity_tolerance&&
      m.function_complementarity<=st.complementarity_tolerance&&
      m.kkt<=st.stationarity_tolerance;
}

void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance,
           const std::string& message) {
  check(std::isfinite(actual) && std::abs(actual-expected) <= tolerance,
        message+": actual="+std::to_string(actual)+
        " expected="+std::to_string(expected));
}
std::vector<std::uint64_t> cache_call_counts(const LazyNewtonStatistics& stats) {
  std::vector<std::uint64_t> counts;
  for(const auto& phase:stats.calls)for(const auto& row:phase)
    counts.insert(counts.end(),{row.values,row.gradients,row.model_values,row.model_gradients,
      row.cached_gradient_accumulations,row.jacobian_products,row.hessian_assemblies,row.hessian_products});
  return counts;
}
void outer_newton_contract(const Result& result) {
  check(result.dca_filter.majorant_outer_only_enabled,
        "majorant solver did not select the single outer-loop engine");
  check(result.dca_filter.majorant_original_filter_enabled &&
        !result.dca_filter.majorant_slack_transport_enabled &&
        result.dca_filter.majorant_slack_transports==0,
        "original filter used auxiliary feasibility or transported the tested slacks");
  check(!result.lazy_newton.predictor_corrector_enabled &&
        !result.dca_filter.majorant_startup_predictor_corrector_enabled &&
        result.dca_filter.majorant_startup_predictor_corrector_steps==0 &&
        result.dca_filter.corrector_steps==0 && result.dca_filter.centering_trials==0 &&
        result.lazy_newton.predictor_corrector_fallbacks==0,
        "outer-only majorants used a predictor/corrector or an alternate direction");
  check(result.dca_filter.majorant_inner_convergences==0 &&
        result.dca_filter.majorant_barrier_phases==0,
        "outer-only majorants waited for an auxiliary convergence or barrier phase");
  check(result.dca_filter.majorant_original_cache_evaluations==result.lazy_newton.accepted_steps,
        "accepted point did not receive exactly one original cache evaluation");
  check(result.lazy_newton.linear_solves==result.lazy_newton.accepted_steps+
        result.lazy_newton.direction_corrections,
        "outer-only oracle used a RHS solve outside accepted Newton steps or numerical refinement");
  check(result.lazy_newton.direction_residual_evaluations==result.lazy_newton.accepted_steps+
        result.lazy_newton.direction_corrections,
        "outer-only oracle duplicated the complete expanded Newton audit");
}
Settings settings() {
  Settings s;
  s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
  s.linear_backend="native";s.dca_filter_curvature_backend="native";
  s.dca_aggregate=false;s.dca_aggregate_only=false;s.dca_krylov=false;
  s.dca_filter_atom_events=false;s.dca_filter_max_atoms=0;
  s.lazy_newton_atomic_regularization=false;
  s.lazy_newton_lazy_constraints=false;s.lazy_newton_reduction=false;
  s.verbose=false;s.live_display_level=0;s.termination_test="ipopt";
  s.max_outer_iterations=100;s.dca_filter_inner_max=100;
  s.time_limit_seconds=30;s.acceptable_iterations=0;s.acceptable_on_failure=false;
  s.stationarity_tolerance=1e-7;s.feasibility_tolerance=1e-7;
  s.dual_infeasibility_tolerance=1e-7;s.complementarity_tolerance=1e-7;
  return s;
}
QCQPProblem nonlinear_problem() {
  QCQPProblem p;p.name="r4_original_objective_and_nonlinear_equalities";p.n=3;
  p.x0={2.0,0.5,4.25};p.lb=Vec(3,-INFINITY);p.ub=Vec(3,INFINITY);
  // Intentionally nonconvex f: convexifying it would violate the contract.
  p.objective.constant=0.7;p.objective.linear={0.2,-0.1,0.3};
  p.objective.hessian.n=3;
  p.objective.hessian.entries={{0,0,-1},{1,1,4},{2,2,2}};
  QuadraticFunction h;h.name="x0_squared_plus_x1_squared_minus_x2";
  h.linear={0,0,-1};h.hessian.n=3;h.hessian.entries={{0,0,2},{1,1,2}};
  p.equalities.push_back(h);
  QuadraticFunction g;g.name="nonconvex_inequality";g.constant=1;
  g.linear={0,0,0.3};g.hessian.n=3;g.hessian.entries={{0,0,-2},{1,1,0.5}};
  p.inequalities.push_back(g);p.validate();return p;
}
QCQPProblem scalar_problem() {
  QCQPProblem p;p.name="r4_auxiliary_optimum_is_not_original_optimum";p.n=1;
  p.x0={2};p.lb={-INFINITY};p.ub={INFINITY};
  p.objective.linear={0};p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,2}}; // x^2
  QuadraticFunction g;g.constant=1;g.linear={0};g.hessian.n=1;
  g.hessian.entries={{0,0,-2}}; // 1-x^2 <= 0
  p.inequalities.push_back(g);p.validate();return p;
}
Vec gradient(const Quadratic& q,const Vec& x,bool original) {
  Vec result(x.size(),0);
  if(original)q.original_gradient_at(x,1,result);
  else q.add_gradient_at(x,1,result);
  return result;
}
void anchor_cache_is_current(const Workspace& w) {
  Vec dual=gradient(w.objective,w.x,true);
  for(int e=0;e<w.me;++e) {
    close(w.h[e],w.equalities[e].original_at(w.x),1e-12,"equality cache stale after model construction");
    w.equalities[e].original_gradient_at(w.x,w.lambda[e],dual);
  }
  for(int i=0;i<w.mi;++i) {
    close(w.rp[i],w.inequalities[i].at(w.x)+w.s[i],1e-12,
          "primal slack residual cache stale after model construction");
    if(w.active[i])w.inequalities[i].original_gradient_at(w.x,w.z[i],dual);
  }
  for(int j=0;j<w.n;++j)
    close(w.rd[j],dual[j],1e-12,"stationarity cache stale after model construction");
}

void precise_majorant_values_preserve_legacy_and_gradients() {
  const double large=std::ldexp(1.0,54);
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Quadratic q;q.ledger=&costs;q.linear={{0,large},{1,1},{2,-large}};
  q.scale=0.25;q.compile();
  const Vec point={1,1,1},center={0,0,0};
  // Exact linear value is one. Legacy double addition loses the middle
  // term; only the majorant route opts into accurate value accumulation.
  q.evaluate(point);const auto original_gradient=q.gradient;
  close(q.value,0,0,"legacy value arithmetic unexpectedly changed");
  const auto calls_before=ns.calls[1][0].values;
  q.majorant_values_extended=true;
  close(q.original_at(point),0.25,0,"extended original value lost cancellation or scaling");
  check(ns.calls[1][0].values==calls_before+1,
        "extended value evaluation added a redundant evaluation traversal");
  q.evaluate(point);
  check(q.gradient==original_gradient,"extended value precision changed the gradient arithmetic");
  q.evaluate(center);q.set_model(center,{},{});
  close(q.at(point),0.25,0,"extended tangent model value was lost or scaled twice");
  q.evaluate(point);
  check(q.gradient==original_gradient,"extended model value precision changed its tangent gradient");
  q.majorant_values_extended=false;
  close(q.at(point),0,0,"legacy tangent model arithmetic unexpectedly changed");

  QCQPProblem p;p.name="r5_workspace_precise_majorant_value_activation";p.n=3;
  p.x0=point;p.lb=Vec(3,0);p.ub=Vec(3,2);
  p.objective.linear={large,1,-large};p.objective.hessian.n=3;
  QuadraticFunction h;h.linear={1,0,0};h.constant=-1;h.hessian.n=3;
  p.equalities.push_back(h);
  QuadraticFunction g;g.linear={0,1,0};g.constant=-2;g.hessian.n=3;
  p.inequalities.push_back(g);p.validate();
  auto st=settings();st.lazy_newton_objective_gradient_target=large/4;
  LazyNewtonStatistics modern_stats;detail::NewtonCostLedger modern_costs(modern_stats,Clock::now());
  Workspace modern(p,st,modern_stats,modern_costs);
  close(modern.objective.value,0.25,0,"majorant Workspace enabled precision after its first value cache");
  check(modern.objective.majorant_values_extended,"majorant objective did not enable precise values");
  for(const auto* family:{&modern.equalities,&modern.inequalities})
    for(const auto& row:*family)check(row.majorant_values_extended,
        "majorant equality/inequality/generated bound missed precise-value activation");
  st.dca_engine="filter";
  LazyNewtonStatistics legacy_stats;detail::NewtonCostLedger legacy_costs(legacy_stats,Clock::now());
  Workspace legacy(p,st,legacy_stats,legacy_costs);
  close(legacy.objective.value,0,0,"legacy Workspace value arithmetic changed");
  check(!legacy.objective.majorant_values_extended,"legacy objective unexpectedly opted into precise values");
  for(const auto* family:{&legacy.equalities,&legacy.inequalities})
    for(const auto& row:*family)check(!row.majorant_values_extended,
        "legacy row unexpectedly opted into precise values");
  check(modern.objective.gradient==legacy.objective.gradient,
        "majorant-only value precision changed the shared objective gradient");
}

void precise_majorant_gradients_and_weighted_stationarity() {
  const double large=std::ldexp(1.0,54);
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Quadratic legacy;legacy.ledger=&costs;legacy.scale=0.25;legacy.linear={{0,1}};
  legacy.hessian={{0,0,large},{0,1,-large},{1,1,large}};legacy.compile();
  auto precise=legacy;precise.majorant_gradients_extended=true;
  precise.majorant_values_extended=true;
  const Vec point={1,1},center={0,0};
  legacy.evaluate(point);
  const auto gradient_calls=ns.calls[1][0].gradients;
  precise.evaluate(point);
  check(ns.calls[1][0].gradients==gradient_calls+1,
        "precise gradient accumulation added a redundant gradient evaluation");
  close(legacy.gradient[0],0,0,"legacy original-gradient arithmetic changed");
  close(precise.gradient[0],0.25,0,"precise original gradient lost the small linear term");
  close(precise.gradient[1],0,0,"precise original gradient changed its exact zero component");
  // The PSD quadratic part has its null direction at (1,1). Its tangent
  // model has the same derivative there and must preserve the small term.
  legacy.evaluate(center);precise.evaluate(center);
  const std::vector<SymTriplet> model_hessian={{0,0,large/4},{0,1,-large/4},{1,1,large/4}};
  const std::map<Pair,int> slots={{{0,0},0},{{0,1},1},{{1,1},2}};
  legacy.set_model(center,model_hessian,slots);precise.set_model(center,model_hessian,slots);
  legacy.evaluate(point);precise.evaluate(point);
  close(legacy.gradient[0],0,0,"legacy tangent-gradient arithmetic changed");
  close(precise.gradient[0],0.25,0,"precise tangent gradient lost cancellation or applied scaling twice");

  QCQPProblem p;p.name="r5_precise_weighted_stationarity";p.n=1;
  p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={-1};p.objective.hessian.n=1;
  for(double coefficient:{large,-large,1.0}) {
    QuadraticFunction g;g.constant=-1;g.linear={coefficient};g.hessian.n=1;
    p.inequalities.push_back(g);
  }
  p.validate();auto st=settings();st.lazy_newton_objective_gradient_target=2*large;
  st.lazy_newton_constraint_gradient_target=2*large;
  LazyNewtonStatistics modern_stats;detail::NewtonCostLedger modern_costs(modern_stats,Clock::now());
  Workspace modern(p,st,modern_stats,modern_costs);modern.z=Vec(3,1);modern.evaluate();
  st.dca_engine="filter";
  LazyNewtonStatistics old_stats;detail::NewtonCostLedger old_costs(old_stats,Clock::now());
  Workspace old(p,st,old_stats,old_costs);old.z=Vec(3,1);old.evaluate();
  close(old.rd[0],1,0,"legacy weighted stationarity arithmetic changed");
  close(modern.rd[0],0,0,"precise weighted stationarity lost the cancelling objective gradient");
  check(modern.objective.majorant_gradients_extended&&!old.objective.majorant_gradients_extended,
        "precise core-gradient activation escaped the majorant route");
}

void infeasible_initial_slacks_preserve_capped_duals_and_center_products() {
  QCQPProblem p;p.name="r6_infeasible_initial_slack_centralization";p.n=2;
  p.x0={2,-1};p.lb={-4,0};p.ub={1,4};
  p.objective.linear={0.3,-0.4};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,2},{1,1,2}};
  QuadraticFunction h;h.constant=-1;h.linear={1,1};h.hessian.n=2;p.equalities.push_back(h);
  for(const auto& coefficients:std::vector<Vec>{{-1.5,1,0},{-5,-1,0},{1.0001,0,1}}) {
    QuadraticFunction g;g.constant=coefficients[0];g.linear={coefficients[1],coefficients[2]};
    g.hessian.n=2;p.inequalities.push_back(g);
  }
  p.validate();auto st=settings();st.lazy_newton_adaptive_barrier=true;
  st.lazy_newton_initial_mu=1e-3;st.lazy_newton_constraint_gradient_target=0.5;
  LazyNewtonStatistics modern_stats;detail::NewtonCostLedger modern_costs(modern_stats,Clock::now());
  Workspace modern(p,st,modern_stats,modern_costs);
  st.dca_engine="filter";
  LazyNewtonStatistics legacy_stats;detail::NewtonCostLedger legacy_costs(legacy_stats,Clock::now());
  Workspace legacy(p,st,legacy_stats,legacy_costs);
  const double mu=modern_stats.initial_barrier;
  check(mu>0.01&&mu==legacy_stats.initial_barrier,
        "infeasible initialization fixture did not retain its raised nominal barrier");
  check(modern.x==p.x0&&legacy.x==p.x0&&modern.lambda==legacy.lambda&&modern.z==legacy.z,
        "initial slack adjustment changed x, equality multipliers or capped inequality multipliers");
  bool modified_native=false,modified_bound=false,unchanged_interior=false;
  for(int i=0;i<modern.mi;++i) {
    const double value=modern.inequalities[i].value;
    const double old_slack=std::max(0.01,-value);
    const double old_dual=std::min(1.0,mu/old_slack);
    close(legacy.s[i],old_slack,0,"legacy initial slack formula changed");
    close(modern.z[i],old_dual,0,"initial inequality multiplier differs from the former capped formula");
    close(modern.s[i]*modern.z[i],mu,1e-15,"initial slack/dual product is not the nominal barrier");
    close(modern.rp[i],value+modern.s[i],0,"initial slack residual cache does not match the changed slack");
    if(modern.s[i]!=legacy.s[i]) {
      if(modern.inequality_kind[i]==0)modified_native=true;else modified_bound=true;
    } else if(-value>mu)unchanged_interior=true;
  }
  check(modified_native&&modified_bound&&unchanged_interior,
        "initialization fixture missed a violated inequality, violated bound or unchanged interior row");
  const auto cached_h=modern.h,cached_rp=modern.rp,cached_rd=modern.rd;
  modern.evaluate();
  for(int e=0;e<modern.me;++e)close(modern.h[e],cached_h[e],1e-12,"initial equality cache differs from a full evaluation");
  for(int i=0;i<modern.mi;++i)close(modern.rp[i],cached_rp[i],1e-12,"initial slack cache differs from a full evaluation");
  for(int j=0;j<modern.n;++j) {
    close(modern.rd[j],cached_rd[j],1e-12,"initial dual cache differs from a full evaluation");
    close(modern.rd[j],legacy.rd[j],1e-12,"initial slack adjustment changed stationarity");
  }
}

void cached_residual_refresh_matches_full_evaluation() {
  auto p=nonlinear_problem();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);
  auto same_vector=[&](const Vec& a,const Vec& b,const std::string& name) {
    check(a.size()==b.size(),name+" dimension mismatch");
    for(std::size_t j=0;j<a.size();++j)close(a[j],b[j],1e-12,name);
  };
  const auto initial_h=w.h,initial_rp=w.rp,initial_rd=w.rd;
  w.evaluate();
  same_vector(initial_h,w.h,"initial cached equality residual");
  same_vector(initial_rp,w.rp,"initial cached slack residual");
  same_vector(initial_rd,w.rd,"initial cached stationarity residual");
  w.lambda[0]=-3.7;w.z[0]=2.3;w.s[0]=0.21;
  auto evaluations=[&]() {
    std::vector<std::uint64_t> calls;
    for(const auto& phase:ns.calls)for(const auto& row:phase)
      calls.insert(calls.end(),{row.values,row.gradients,row.model_values,row.model_gradients});
    return calls;
  };
  const auto before=evaluations();
  w.refresh_majorant_residuals_from_cache();
  check(evaluations()==before,"cached residual refresh reevaluated QCQP functions or gradients");
  const auto refreshed_h=w.h,refreshed_rp=w.rp,refreshed_rd=w.rd;
  w.evaluate();
  same_vector(refreshed_h,w.h,"updated cached equality residual");
  same_vector(refreshed_rp,w.rp,"updated cached slack residual");
  same_vector(refreshed_rd,w.rd,"updated cached stationarity residual");
}

void model_contract_and_transport() {
  auto p=nonlinear_problem();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);
  const Vec anchor=w.x;
  const auto initial_calls=cache_call_counts(ns);
  models.prepare_curvature_once();
  const auto prepared_curvatures=models.curvature_builds;
  models.prepare_curvature_once();
  check(models.curvature_builds==prepared_curvatures&&cache_call_counts(ns)==initial_calls,
        "one-time curvature preparation repeated a certificate or evaluated QCQP functions");
  models.reanchor_from_cache();
  check(cache_call_counts(ns)==initial_calls,
        "initial model preparation failed to reuse the original Workspace caches");
  anchor_cache_is_current(w);
  check(!w.objective.model_active && !w.equalities[0].model_active,
        "majorants altered the original objective or nonlinear equality");
  check(w.inequalities[0].model_active,"nonconvex inequality has no majorant");
  check(models.rho_max>0,"negative inequality curvature has no scalar correction");
  auto& g=w.inequalities[0];
  close(g.at(anchor),g.original_at(anchor),1e-12,"majorant value tangency");
  check(gradient(g,anchor,false)==gradient(g,anchor,true),"majorant gradient tangency");
  for(const Vec& d:std::vector<Vec>{{0.2,-0.3,0.1},{-0.7,0.4,-0.5},
                                    {1.1,-0.2,0.0},{0,0,1},{1,1,1}}) {
    Vec plus=anchor,minus=anchor;axpy(1,d,plus);axpy(-1,d,minus);
    for(const Vec* x:{&plus,&minus}) {
      close(w.objective.at(*x),w.objective.original_at(*x),1e-12,"original f changed away from anchor");
      close(w.equalities[0].at(*x),w.equalities[0].original_at(*x),1e-12,"nonlinear equality was linearized");
      check(gradient(w.objective,*x,false)==gradient(w.objective,*x,true),"objective gradient changed");
      check(gradient(w.equalities[0],*x,false)==gradient(w.equalities[0],*x,true),"equality gradient changed");
      check(g.at(*x)>=g.original_at(*x)-1e-11,"inequality model is not a majorant");
    }
    check(g.at(plus)+g.at(minus)-2*g.at(anchor)>=-1e-11,
          "convexified inequality has negative directional curvature");
  }

  // Deliberately infeasible in the slack equation: s is NOT inferred as -g.
  w.x={1.3,0.1,2.0};w.s[0]=0.25;w.z[0]=1.7;w.lambda[0]=-3.1;
  const auto x=w.x,z=w.z,y=w.lambda;
  const double model_value=g.at(w.x),original_value=g.original_at(w.x);
  const double old_slack=w.s[0],old_residual=model_value+old_slack;
  const double gap=model_value-original_value;
  check(gap>0.1 && std::abs(old_residual)>0.1,"transport test failed to exercise a nonzero model and primal gap");
  const double expected_slack=old_slack+gap;
  const double expected_complementarity=w.z[0]*old_slack+w.z[0]*gap;
  Result exported;w.export_state(exported);
  close(exported.termination_inequality_slacks[0],expected_slack/g.scale,1e-12,
        "terminal audit did not receive transported original slack");
  models.transport_slacks();
  check(!g.model_active,"transport did not leave original-constraint coordinates");
  check(w.x==x && w.z==z && w.lambda==y,"slack transport altered primal or multiplier coordinates");
  close(w.s[0],expected_slack,1e-12,"wrong sign of majorant-to-original slack shift");
  close(g.original_at(w.x)+w.s[0],old_residual,1e-12,"transport changed infeasible primal slack residual");
  close(w.z[0]*w.s[0],expected_complementarity,1e-12,"transport lost complementarity gap");
  const auto once=w.s;models.transport_slacks();
  check(w.s==once,"slack transport applied the model gap twice");

  check(stats.curvature_factors==0 && stats.rho_refinement_trials==0,
        "direct inequality correction used inertia certificates or rho refinement");
  const auto before_builds=models.curvature_builds;
  const auto before_reuses=models.curvature_reuses;
  const double previous_rho=models.rho_max;
  w.evaluate();
  const auto before_reanchor_calls=cache_call_counts(ns);
  models.reanchor_from_cache();
  check(cache_call_counts(ns)==before_reanchor_calls,
        "reanchor_from_cache repeated a function, gradient or Hessian traversal");
  check(w.x==x && w.z==z && w.lambda==y,
        "cached reanchoring changed a primal or multiplier coordinate");
  anchor_cache_is_current(w);
  check(models.curvature_builds==before_builds && models.curvature_reuses>before_reuses,
        "constant QCQP inequality curvature was reconstructed after reanchoring");
  close(models.rho_max,previous_rho,0,"reanchoring changed fixed inequality curvature");
  check(w.s==once,"reanchoring changed the already transported slack");
  close(g.at(w.x),g.original_at(w.x),1e-12,"new majorant is not value-tangent");
  check(gradient(g,w.x,false)==gradient(g,w.x,true),"new majorant is not gradient-tangent");
  close(g.at(w.x)+w.s[0],old_residual,1e-12,"reanchoring lost original slack residual");
  for(int step=0;step<12;++step) {
    w.x[0]-=0.013;w.x[1]+=0.007;w.x[2]-=0.003;
    const double preserved_residual=g.at(w.x)+w.s[0];
    const auto kept_x=w.x,kept_z=w.z,kept_lambda=w.lambda;
    models.transport_slacks();w.evaluate();
    const auto before_reanchor=cache_call_counts(ns);
    models.reanchor_from_cache();
    check(cache_call_counts(ns)==before_reanchor,
          "successive cached reanchoring reevaluated the accepted point");
    check(w.x==kept_x&&w.z==kept_z&&w.lambda==kept_lambda,
          "transport/reanchoring changed the accepted multipliers");
    anchor_cache_is_current(w);
    check(models.curvature_builds==before_builds,
          "repeated accepted-point reanchoring reconstructed constant curvature");
    close(g.at(w.x)+w.s[0],preserved_residual,1e-11,
          "accepted-point reanchoring changed the auxiliary primal residual");
    close(g.at(w.x),g.original_at(w.x),1e-12,"successive reanchor lost tangency");
    check(w.s[0]>0,"accepted-point transport destroyed slack positivity");
  }
}

void accepted_original_state_survives_discard_and_cached_reanchor() {
  auto p=nonlinear_problem();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);
  models.prepare_curvature_once();models.reanchor_from_cache();
  w.x={1.3,0.1,2.0};w.s[0]=0.25;w.z[0]=1.7;w.lambda[0]=-3.1;
  auto& g=w.inequalities[0];
  check(g.at(w.x)-g.original_at(w.x)>0.1,
        "discard fixture lacks a nonzero majorization gap");
  const auto accepted_x=w.x,accepted_s=w.s,accepted_z=w.z,accepted_lambda=w.lambda;
  const auto cached_h=w.h,cached_rp=w.rp,cached_rd=w.rd;
  const auto cached_gradient=g.gradient;const auto cached_gradient_extended=g.gradient_extended;
  const double cached_value=g.value;
  const auto before=cache_call_counts(ns);
  models.discard();models.discard();
  check(!g.model_active&&cache_call_counts(ns)==before,
        "discard reevaluated functions or left the auxiliary model installed");
  check(w.x==accepted_x&&w.s==accepted_s&&w.z==accepted_z&&w.lambda==accepted_lambda,
        "discard changed the accepted primal-dual point or added the majorization gap to its slack");
  check(w.h==cached_h&&w.rp==cached_rp&&w.rd==cached_rd&&g.value==cached_value&&
        g.gradient==cached_gradient&&g.gradient_extended==cached_gradient_extended,
        "discard mutated caches instead of only disabling models");
  Result exported;w.export_state(exported);
  close(exported.termination_inequality_slacks[0],accepted_s[0]/g.scale,1e-12,
        "original audit exported a different slack from the one accepted by the filter");
  w.evaluate();
  const double original_residual=g.original_at(w.x)+accepted_s[0];
  close(w.rp[0],original_residual,1e-12,"original evaluation retained the auxiliary residual");
  const auto evaluated_h=w.h,evaluated_rp=w.rp,evaluated_rd=w.rd;
  const auto before_reanchor=cache_call_counts(ns);
  models.reanchor_from_cache();
  check(cache_call_counts(ns)==before_reanchor,
        "original-filter reanchor traversed coefficients after the original point was evaluated");
  check(w.x==accepted_x&&w.s==accepted_s&&w.z==accepted_z&&w.lambda==accepted_lambda&&
        w.h==evaluated_h&&w.rp==evaluated_rp&&w.rd==evaluated_rd,
        "reanchor changed the accepted point or its original residuals");
  anchor_cache_is_current(w);
  close(g.at(w.x)+w.s[0],original_residual,1e-12,
        "new auxiliary anchor is inconsistent with the accepted original slack equation");
  check(stats.majorant_slack_transports==0,"discard/reanchor invoked legacy slack transport");
}

void single_accepted_step_exports_exact_affine_slack_and_dual() {
  // For f=x^2, g=1-x^2, x0=2, s0=3, z0=0.1/3, the anchor is
  // primal feasible and centered. Irrespective of the accepted alpha and
  // numerical Hessian shift, the Newton slack equation gives ds=4*dx and
  // complementarity gives dz=-(z0/s0)*ds. This independent identity detects
  // any post-filter gap transport, dual reset, or unequal primal/dual step.
  auto p=scalar_problem();auto st=settings();st.max_outer_iterations=1;
  st.lazy_newton_initial_mu=0.1;
  st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  const auto result=solve_filter_dca(p,st);
  check(result.lazy_newton.accepted_steps==1,
        "affine-slack endpoint fixture did not accept one original-filter step");
  const double delta=result.x.at(0)-2.0;
  check(delta*delta>1e-3,"affine-slack endpoint fixture has a negligible majorization gap");
  const double expected_slack=3.0+4.0*delta;
  const double initial_dual=0.1/3.0;
  const double expected_dual=initial_dual-(initial_dual/3.0)*4.0*delta;
  check(expected_slack>0&&expected_dual>0,"one-step affine slack/dual lost positivity");
  close(result.termination_inequality_slacks.at(0),expected_slack,1e-12,
        "accepted slack was changed after the original filter test");
  close(result.certificate_inequality_multipliers.at(0),expected_dual,1e-12,
        "accepted multiplier was reset or did not use the common filter step");
  check(std::abs(result.termination_inequality_slacks[0]-(expected_slack+delta*delta))>1e-3,
        "terminal slack contains the discarded majorization gap");
  check(result.status!=Status::Solved&&result.status!=Status::SolvedAcceptable,
        "one accepted filter point was promoted to original optimality");
  outer_newton_contract(result);
}

void primal_centering_gate_excludes_only_the_original_dual() {
  QCQPProblem p;p.name="r7_primal_centering_gate_components";p.n=2;
  p.x0={0,0};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={100,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction h;h.linear={0,1};h.hessian.n=2;p.equalities.push_back(h);
  QuadraticFunction g;g.constant=-1;g.linear={0,0};g.hessian.n=2;p.inequalities.push_back(g);
  p.validate();auto st=settings();st.lazy_newton_initial_mu=0.1;
  st.lazy_newton_objective_gradient_target=1000;
  st.lazy_newton_constraint_gradient_target=100;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);
  constexpr double mu=0.1,gate_threshold=10*mu;
  auto metrics=dca_outer_metrics(w);
  check(metrics.residual(mu)>gate_threshold,
        "dual-only gate fixture does not expose a large original stationarity residual");
  close(metrics.primal_centering_residual(mu),0,1e-14,
        "dual-only residual still enters the primal/centering barrier gate");
  w.lambda[0]=1000;w.evaluate();
  metrics=dca_outer_metrics(w);
  close(metrics.primal_centering_residual(mu),0,1e-14,
        "changing only equality multipliers changed the primal/centering gate");
  check(metrics.residual(mu)>=1000,
        "full residual diagnostic incorrectly omitted the original dual");

  w.x[1]=2;w.evaluate();metrics=dca_outer_metrics(w);
  close(metrics.barrier_primal_residual,2,1e-14,"gate omitted original nonlinear/equality feasibility");
  check(metrics.primal_centering_residual(mu)>gate_threshold,
        "large equality residual no longer blocks a barrier decrease");
  w.x[1]=0;w.s[0]=3;w.evaluate();metrics=dca_outer_metrics(w);
  close(metrics.barrier_primal_residual,2,1e-14,"gate omitted the original slack equation");
  check(metrics.primal_centering_residual(mu)>gate_threshold,
        "large original g+s residual no longer blocks a barrier decrease");
  w.s[0]=1;w.z[0]=2;w.evaluate();
  const auto calls_before=cache_call_counts(ns);metrics=dca_outer_metrics(w);
  close(metrics.barrier_primal_residual,0,1e-14,"centering fixture is not slack/equality feasible");
  close(metrics.primal_centering_residual(mu),1.9,1e-14,
        "barrier gate omitted the centered complementarity product residual");
  check(metrics.primal_centering_residual(mu)>gate_threshold,
        "poor complementarity centering no longer blocks a barrier decrease");
  check(cache_call_counts(ns)==calls_before,
        "splitting the barrier gate from its dual diagnostic reevaluated QCQP functions");
}

void accepted_primal_centered_point_reduces_mu_despite_large_dual() {
  auto p=scalar_problem();auto st=settings();st.lazy_newton_initial_mu=0.1;
  st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  st.max_outer_iterations=1;const auto first=solve_filter_dca(p,st);
  check(first.lazy_newton.accepted_steps==1,
        "primal-centered scalar gate fixture did not accept its initial step");
  // The existing affine-slack oracle has x≈1.25375, s=.015, z≈.0665.
  // Its actual original slack equation is imperfect but below 10*mu;
  // original stationarity remains above that threshold.
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);
  w.x=first.x;w.s[0]=first.termination_inequality_slacks.at(0);
  w.z[0]=first.certificate_inequality_multipliers.at(0);w.evaluate();
  const auto current=dca_outer_metrics(w);
  constexpr double mu=0.1;
  check(current.primal_centering_residual(mu)<=10*mu&&current.residual(mu)>10*mu,
        "accepted scalar state does not isolate the removed dual-only barrier gate");
  check(current.dual>10*mu,
        "accepted scalar state has no substantial original stationarity error");
  close(first.dca_filter.majorant_barrier_mu,mu,1e-14,
        "outer budget exit performed an unused barrier update");
  st.max_outer_iterations=2;const auto second=solve_filter_dca(p,st);
  check(second.lazy_newton.accepted_steps==2,
        "primal-centered scalar state did not advance into the second outer step");
  close(second.dca_filter.majorant_barrier_mu,0.2*mu,1e-14,
        "large original dual still prevented the eligible outer barrier decrease");
  check(second.dca_filter.majorant_mu_updates==1,
        "primal-centering gate introduced more than one barrier update between two accepted steps");
  auto certified=first;
  check(!detail::audit_original_termination(p,st,certified)&&first.status!=Status::Solved,
        "removing dual from the barrier gate also removed it from original KKT certification");
  outer_newton_contract(first);outer_newton_contract(second);
}

void auxiliary_oracle_and_original_certificate() {
  auto p=scalar_problem();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  auto& q=w.inequalities[0];
  // Analytic DC majorant at a=2: 1-x^2+(x-2)^2 = 5-4x.
  close(models.rho_max,2.0,2e-6,"scalar inequality correction differs from the PSD threshold");
  for(double x:{0.0,1.0,1.25,2.0,3.0})
    close(q.at(Vec{x}),5-4*x,5e-6,"scalar auxiliary differs from analytic affine majorant");
  w.x={1.25};w.s[0]=0;w.z[0]=0.625;w.evaluate();
  check(inf_norm(w.rd)<5e-6 && inf_norm(w.rp)<5e-6,
        "analytic auxiliary KKT point was not recognized by model equations");
  const auto original=dca_original(w,w.x,w.lambda,w.z);
  check(original.dual>0.9 && original.comp>0.3,
        "auxiliary multipliers falsely satisfy original KKT at x=1.25");
  Result candidate;w.export_state(candidate);
  check(!detail::audit_original_termination(p,st,candidate,&w.original_linear_support),
        "auxiliary optimum was falsely certified as original optimum");

  st.max_outer_iterations=2;st.dca_filter_inner_max=1;
  auto one=solve_filter_dca(p,st);
  check(one.status!=Status::Solved && one.status!=Status::SolvedAcceptable &&
        one.status!=Status::SolvedAcceptableOnFailure,
        "two accepted incomplete majorant steps falsely ended the outer iteration");
  check(one.dca_filter.majorant_reanchors>=1 &&
        one.dca_filter.majorant_inner_convergences==0,
        std::string("filter-accepted point was not reanchored before auxiliary convergence: ")+
        status_string(one.status)+" / "+one.message);
  check(one.dca_filter.majorant_barrier_line_acceptances>=1 && one.dca_filter.filter_trials>=1,
        "majorant step was committed without the original barrier/slack filter");
  check(one.dca_filter.private_line_trials==0,
        "R6 silently used a private auxiliary merit search");
  check(one.dca_filter.majorant_barrier_filter_enabled &&
        one.dca_filter.original_line_acceptances==0,
        "barrier-globalized route was falsely reported as original-filter acceptance");
  check(one.dca_filter.majorant_filter_reanchor_enabled &&
        one.dca_filter.majorant_outer_only_enabled &&
        one.dca_filter.maximum_model_steps==1,
        "accepted point did not trigger exactly one majorant step before early reanchoring");
  check(one.dca_filter.aggregate_certificates==0 && one.dca_filter.krylov_trials==0,
        "majorants route invoked original aggregate or Krylov fallback");

  st.max_outer_iterations=100;st.dca_filter_inner_max=1;
  st.lazy_newton_predictor_corrector=true; // Legacy flag must not activate a second RHS in R6.
  auto full=solve_filter_dca(p,st);
  check(full.status==Status::Solved,std::string("successive majorants did not solve scalar oracle: ")+
        status_string(full.status)+" / "+full.message);
  close(full.x.at(0),1.0,2e-5,"successive majorants converged to the wrong original point");
  check(full.dca_filter.majorant_reanchors>1,
        "original solution was obtained without repeated filter-accepted majorant steps");
  check(full.dca_filter.aggregate_certificates==0 && full.dca_filter.krylov_trials==0,
        "original aggregate or Krylov path contaminated the majorant solve");
  check(full.dca_filter.curvature_factors==0 && full.dca_filter.rho_refinement_trials==0,
        "majorants solve used inequality inertia certificates or rho bisection");
  check(detail::audit_original_termination(p,st,full),"final original certificate does not reproduce");
  outer_newton_contract(full);
  check(full.dca_filter.majorant_reanchors==full.lazy_newton.accepted_steps-1,
        "strict original convergence prepared an unused following majorant");
  check(full.lazy_newton.predictor_corrector_requested,
        "legacy predictor/corrector request was not recorded faithfully");
  st.lazy_newton_predictor_corrector=false;st.dca_filter_inner_max=17;
  auto flag_off=solve_filter_dca(p,st);outer_newton_contract(flag_off);
  check(flag_off.status==Status::Solved && flag_off.x==full.x &&
        flag_off.objective==full.objective &&
        flag_off.certificate_inequality_multipliers==full.certificate_inequality_multipliers &&
        flag_off.lazy_newton.accepted_steps==full.lazy_newton.accepted_steps &&
        flag_off.lazy_newton.linear_solves==full.lazy_newton.linear_solves,
        "legacy PC/inner-budget flags changed the single outer-loop trajectory");

  p.x0={0.5}; // Initially infeasible original inequality.
  st.lazy_newton_predictor_corrector=true;st.dca_filter_inner_max=1;
  auto infeasible=solve_filter_dca(p,st);
  check(infeasible.status==Status::Solved,
        std::string("outer-only infeasible scalar oracle did not converge strictly: ")+status_string(infeasible.status)+" / "+infeasible.message);
  close(infeasible.x.at(0),1.0,2e-5,"infeasible scalar oracle converged to the wrong original point");
  outer_newton_contract(infeasible);
  check(detail::audit_original_termination(p,st,infeasible),
        "infeasible outer-only oracle lacks its independent original certificate");

}

void pure_dual_step_requires_and_reaches_the_original_certificate() {
  QCQPProblem p;p.name="r5_pure_dual_newton_step";p.n=1;
  p.x0={1};p.lb={-INFINITY};p.ub={INFINITY};
  p.objective.linear={1};p.objective.hessian.n=1;
  QuadraticFunction h;h.constant=-1;h.linear={1};h.hessian.n=1;
  p.equalities.push_back(h);p.validate();auto st=settings();
  Result initial;initial.x=p.x0;initial.certificate_equality_multipliers={0};
  initial.certificate_lower_multipliers={0};initial.certificate_upper_multipliers={0};
  check(!detail::audit_original_termination(p,st,initial),
        "primal feasibility with the wrong multiplier was certified as solved");
  auto result=solve_filter_dca(p,st);
  check(result.status==Status::Solved,
        std::string("valid pure-dual Newton step was rejected: ")+status_string(result.status)+" / "+result.message);
  check(result.x==p.x0&&result.step_inf==0,
        "pure-dual fixture unexpectedly relied on primal or objective improvement");
  check(result.lazy_newton.accepted_steps==1,
        "pure-dual optimum was claimed without exactly one committed multiplier correction");
  check(result.lazy_newton.dual_only_acceptances==0 && result.dca_filter.filter_trials>=1,
        "pure-dual step bypassed the common filter using the old tiny-primal shortcut");
  check(result.dca_filter.majorant_startup_predictor_corrector_steps==0 &&
        result.dca_filter.majorant_terminal_single_newton_steps==1 &&
        result.dca_filter.corrector_steps==0,
        "pure-dual feasible problem entered the startup predictor-corrector phase");
  check(result.certificate_equality_multipliers.size()==1,
        "pure-dual solution lost its original equality multiplier");
  close(result.certificate_equality_multipliers[0],-1,1e-13,
        "pure-dual step did not produce the required original KKT multiplier");
  check(detail::audit_original_termination(p,st,result),
        "pure-dual accepted step failed its independent original certificate");
  outer_newton_contract(result);
  check(result.dca_filter.majorant_reanchors==0,
        "pure-dual certified solution constructed a needless next majorant");
}

void outer_newton_preserves_its_single_direction_when_feasibility_worsens() {
  QCQPProblem p;p.name="r5_terminal_mode_is_not_reactivated";p.n=2;
  p.x0={1,0};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.constant=1.04;p.objective.linear={-0.4,-2};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,2},{1,1,2}};
  QuadraticFunction h;h.constant=-1;h.linear={0,0};h.hessian.n=2;
  h.hessian.entries={{0,0,2},{1,1,2}};p.equalities.push_back(h);
  QuadraticFunction g;g.constant=-1;g.linear={0,0};g.hessian.n=2;p.inequalities.push_back(g);
  p.validate();auto st=settings();st.lazy_newton_predictor_corrector=true;
  st.max_outer_iterations=1;const auto first=solve_filter_dca(p,st);
  check(first.lazy_newton.accepted_steps==1,
        "nonlinear feasibility fixture failed to accept its initial terminal Newton step");
  check(std::abs(first.x[0]*first.x[0]+first.x[1]*first.x[1]-1)>0.1,
        "nonlinear fixture did not develop a substantial equality residual after the first step");
  st.max_outer_iterations=2;const auto second=solve_filter_dca(p,st);
  check(second.lazy_newton.accepted_steps==2 &&
        second.dca_filter.majorant_terminal_single_newton_steps==2 &&
        second.dca_filter.majorant_startup_predictor_corrector_steps==0 &&
        !second.lazy_newton.predictor_corrector_enabled,
        "outer-only engine added a second direction after nonlinear feasibility worsened");
  check(second.status!=Status::Solved&&second.status!=Status::SolvedAcceptable&&
        second.status!=Status::SolvedAcceptableOnFailure,
        "two filter-accepted nonlinear steps were falsely certified as optimal");
  outer_newton_contract(first);outer_newton_contract(second);
}

void tight_paired_spectrum_majorant() {
  // Analytic OPF angle block: H=[0 B;B' 0], B=[-t -1;1 -t].
  // Its eigenvalues are +/-sqrt(1+t*t), each twice. The positive/negative
  // eigenspaces provide an independent exact oracle for both PSD claims.
  constexpr double t=0.3;
  const double sigma=std::hypot(1.0,t);
  QCQPProblem p;p.name="r5_tight_angle_majorant";p.n=4;
  p.x0={1.1,0.7,-0.2,0.8};p.lb=Vec(4,-INFINITY);p.ub=Vec(4,INFINITY);
  p.objective.linear=Vec(4,0);p.objective.hessian.n=4;
  p.objective.hessian.entries={{0,0,2},{1,1,2},{2,2,2},{3,3,2}};
  QuadraticFunction g;g.constant=0.4;g.linear={0.3,-0.2,0.1,-0.7};g.hessian.n=4;
  g.hessian.entries={{0,2,-t},{0,3,-1},{1,2,1},{1,3,-t}};
  p.inequalities.push_back(g);p.validate();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();auto& q=w.inequalities[0];
  const auto anchor=w.x;
  check(q.model_active,"angle constraint has no convex inequality majorant");
  close(q.at(anchor),q.original_at(anchor),1e-12,"tight majorant value tangency");
  check(gradient(q,anchor,false)==gradient(q,anchor,true),"tight majorant gradient tangency");
  std::set<std::pair<int,int>> reserved;
  for(auto entry:q.hessian)reserved.insert(upper(entry.i,entry.j));
  std::vector<Vec> corrected(4,Vec(4,0)),correction(4,Vec(4,0));
  for(auto entry:q.model_hessian) {
    check(reserved.count(upper(entry.i,entry.j))!=0,
          "tight majorant introduced a new Hessian sparsity coordinate");
    corrected[entry.i][entry.j]+=entry.v;
    if(entry.i!=entry.j)corrected[entry.j][entry.i]+=entry.v;
  }
  correction=corrected;
  for(auto entry:q.hessian) {
    correction[entry.i][entry.j]-=q.scale*entry.v;
    if(entry.i!=entry.j)correction[entry.j][entry.i]-=q.scale*entry.v;
  }
  auto product=[](const std::vector<Vec>& matrix,const Vec& v) {
    Vec result(v.size(),0);for(std::size_t i=0;i<v.size();++i)result[i]=dot(matrix[i],v);
    return result;
  };
  double trace=0;for(int j=0;j<4;++j)trace+=correction[j][j];
  close(trace,2*sigma*q.scale,1e-11,"tight angle correction did not halve isotropic trace");
  std::vector<Vec> directions;
  for(int sign:{-1,1})for(const Vec& u:std::vector<Vec>{{1,0},{0,1}})
    directions.push_back({u[0],u[1],sign*(-t*u[0]+u[1])/sigma,
                         sign*(-u[0]-t*u[1])/sigma});
  directions.push_back({0.37,-0.12,0.53,0.24});
  for(const auto& d:directions) {
    const double model_curvature=dot(d,product(corrected,d));
    const double gap_curvature=dot(d,product(correction,d));
    check(model_curvature>=-1e-12 && gap_curvature>=-1e-12,
          "tight angle model or its dominance correction is not PSD");
    check(gap_curvature<=sigma*q.scale*dot(d,d)+1e-11,
          "tight correction exceeds the old isotropic majorant");
    auto x=anchor;axpy(1,d,x);
    close(q.at(x)-q.original_at(x),0.5*gap_curvature,1e-11,
          "tight model gap disagrees with independently reconstructed correction");
  }
  const auto d=directions.back();
  axpy(1,d,w.x);w.s[0]=0.23;w.z[0]=1.7;
  const auto correction_d=product(correction,d);
  const double gap=0.5*dot(d,correction_d),primal_residual=q.at(w.x)+w.s[0];
  const auto measured=models.gaps(w.x,w.z);
  close(measured.value_gap_inf,gap/q.scale,1e-11,"tight gap reporting omitted Hessian cross terms");
  close(measured.dual_gap_inf,1.7*inf_norm(correction_d)/w.objective.scale,1e-11,
        "tight stationarity gap omitted correction cross terms");
  close(measured.weighted_value_gap_inf,1.7*gap/w.objective.scale,1e-11,
        "tight complementarity gap omitted correction cross terms");
  const auto builds=models.curvature_builds;
  models.transport_slacks();
  close(q.original_at(w.x)+w.s[0],primal_residual,1e-11,
        "tight majorant slack transport changed the infeasible primal residual");
  close(w.s[0],0.23+gap,1e-11,"tight slack transport used the old isotropic gap");
  models.build();
  check(models.curvature_builds==builds,"tight majorant reanchor recomputed its spectrum");
  anchor_cache_is_current(w);
}

void fixed_barrier_newton_escapes_predictor_corrector_false_fixed_point() {
  QCQPProblem p;p.name="r5_fixed_barrier_predictor_cross_term_bias";p.n=1;
  p.x0={3.0/11};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={1};
  p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,-1}};
  QuadraticFunction g;g.linear={-1};g.hessian.n=1;p.inequalities.push_back(g);p.validate();
  auto st=settings();st.kkt_backward_error_reject=1e-10;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  w.x[0]=w.s[0]=3.0/11;w.z[0]=8.0/11;w.evaluate();w.update_matrix();
  check(inf_norm(w.rd)<1e-15&&inf_norm(w.rp)<1e-15,
        "fixed-barrier counterexample must start stationary and slack-feasible");
  check(w.factorize(0)&&w.regularization==0,"positive scalar condensed KKT required an unexpected shift");
  const double mu=24.0/3025,complementarity=w.s[0]*w.z[0];
  const auto affine=w.direction(Vec{complementarity},w.rd,w.h,w.rp);
  close(affine.s[0],-24.0/55,1e-13,"counterexample affine slack direction");
  close(affine.z[0],24.0/55,1e-13,"counterexample affine multiplier direction");
  const double true_residual=complementarity-mu;
  const double biased_residual=true_residual+affine.s[0]*affine.z[0];
  close(true_residual,576.0/3025,1e-15,"counterexample true barrier residual");
  check(true_residual>10*mu&&std::abs(biased_residual)<1e-15,
        "predictor cross term did not hide the unsolved fixed-barrier equation");
  const auto biased=w.direction(Vec{biased_residual},w.rd,w.h,w.rp);
  check(inf_norm(biased.x)<1e-13&&inf_norm(biased.s)<1e-13&&inf_norm(biased.z)<1e-13,
        "biased predictor-corrector counterexample did not produce a false fixed point");
  const auto exact=w.direction(Vec{true_residual},w.rd,w.h,w.rp);
  check(inf_norm(exact.x)>0.1&&exact.residual<=st.kkt_backward_error_reject,
        "one fixed-barrier Newton RHS failed to escape the false fixed point");
  close(w.s[0]*exact.z[0]+w.z[0]*exact.s[0],-true_residual,1e-13,
        "fixed-barrier direction fails the actual complementarity Newton equation");
}

void regularization_memory_requires_optimality_progress() {
  // For the concave but bounded objective, early objective-descent steps move
  // away from its stationary maximum. An accepted full step consequently
  // need not improve stationarity and cannot justify discarding a useful
  // positive-definiteness shift. This is deliberately independent of the
  // controller's particular numerical contraction threshold.
  QCQPProblem p;p.name="r5_full_step_objective_descent_without_dual_progress";p.n=1;
  p.x0={1};p.lb={-10};p.ub={10};p.objective.linear={0};
  p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,-2}};p.validate();
  auto st=settings();st.max_outer_iterations=2;st.dca_filter_inner_max=1;
  const auto result=solve_filter_dca(p,st);
  check(result.lazy_newton.accepted_steps==2 && result.lazy_newton.full_steps==2,
        "rho-memory fixture failed to produce two full original-filter steps");
  check(result.x[0]>p.x0[0] && result.objective < -1,
        "rho-memory fixture failed to improve the concave original objective");
  check(result.lazy_newton.regularization_increases>0 &&
        result.lazy_newton.final_regularization>0,
        "indefinite auxiliary fixture did not require a numerical shift");
  check(result.lazy_newton.regularization_decreases==0,
        "objective-only full steps prematurely discarded numerical rho memory");
  check(result.status!=Status::Solved && result.status!=Status::SolvedAcceptable &&
        result.status!=Status::SolvedAcceptableOnFailure,
        "filter acceptance was mistaken for certified optimality");
}

void cancelled_extended_newton_equations() {
  QCQPProblem p;p.name="r5_extended_audit_preserves_equations";p.n=1;
  p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};
  p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,1}};
  const double large=std::ldexp(1.0,54),small=std::ldexp(1.0,-54);
  for(int i=0;i<3;++i) {
    QuadraticFunction g;g.constant=-small;g.linear={1};g.hessian.n=1;
    if(i==0)g.hessian.entries={{0,0,-2}};
    p.inequalities.push_back(g);
  }
  p.validate();auto st=settings();st.kkt_backward_error_reject=1e-10;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  w.s=Vec(3,small);w.z=Vec(3,1);w.evaluate();w.update_matrix();
  check(w.factorize(0),"extended-residual fixture factorization failed");
  Direction d;d.x={0};d.s=Vec(3,0);d.z={large,-large,1};
  const Vec rc={-1,1,-small},dual={-1},eq,primal(3,0);
  // All data are binary powers, so the exact four-block Newton residual is
  // zero. Sequential double accumulation loses the -1 before cancellation.
  Vec inaccurate=dual;
  for(int i=0;i<w.mi;++i)w.inequalities[i].add_gradient(d.z[i],inaccurate);
  check(inf_norm(inaccurate)>0.5,"extended-audit fixture did not expose cancellation");
  Vec ed(1),eh,ep(3),ec(3);double de=0,ee=0,pe=0,ce=0;
  const auto first=w.majorant_direction_residual_extended(d,rc,dual,eq,primal,
      ed,eh,ep,ec,de,ee,pe,ce);
  check(first<=st.kkt_backward_error_reject&&de==0&&ee==0&&pe==0&&ce==0,
        "accurate reaccumulation rejected an exact auxiliary Newton direction");
  const auto second=w.majorant_direction_residual_extended(d,rc,Vec{-0.5},eq,primal,
      ed,eh,ep,ec,de,ee,pe,ce);
  close(de,0.5,0,"extended audit hid a genuine stationarity residual");
  check(second>st.kkt_backward_error_reject,
        "extended-precision recovery relaxed the unchanged rejection tolerance");
  check(w.inequalities[0].model_active,"extended audit disabled the inequality majorant");
}

void near_active_bound_reconstruction() {
  QCQPProblem p;p.name="r4_singleton_bound_expanded_residual";p.n=2;
  p.x0={1e-5,0.3};p.lb={-INFINITY,-INFINITY};p.ub={0,INFINITY};
  p.objective.linear={0.7,-0.2};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,2},{1,1,3}};
  QuadraticFunction equality;equality.linear={0,1};equality.hessian.n=2;
  p.equalities.push_back(equality);
  QuadraticFunction g;g.constant=-2;g.linear={0,0};g.hessian.n=2;
  g.hessian.entries={{0,0,-2}};p.inequalities.push_back(g);p.validate();
  std::vector<std::string> backends={"native"};
  if(linear_backend_compiled("mumps"))backends.push_back("mumps");
  for(const auto& backend:backends) {
    auto st=settings();st.linear_backend=backend;
    st.kkt_backward_error_reject=1e-10;st.lazy_newton_sparse_recovery=false;
    LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
    Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
    DcaInequalityModels models(w,stats);models.build();
    const int native=0,bound=1;
    check(w.inequality_kind[bound]==1 && w.jr[bound].support.size()==1 &&
          w.jr[bound].dense<0,"near-active bound was not initially condensed");
    w.s[native]=2;w.z[native]=0.25;w.s[bound]=1e-24;w.z[bound]=1;w.lambda[0]=0.8;
    w.evaluate();w.update_matrix();
    const auto auxiliary_column0=w.hessian_product(Vec{1,0});
    const auto auxiliary_column1=w.hessian_product(Vec{0,1});
    close(auxiliary_column0[0],2.0,1e-12,"auxiliary inequality curvature missing before recovery");
    close(auxiliary_column1[1],3.0,1e-12,"original objective Hessian changed before recovery");
    auto same_hessian=[&](){return w.hessian_product(Vec{1,0})==auxiliary_column0 &&
                                 w.hessian_product(Vec{0,1})==auxiliary_column1;};
    check(w.inequalities[native].model_active,"reconstruction fixture lost its inequality majorant");
    Vec rc(w.mi);
    auto complementarity=[&](){for(int i=0;i<w.mi;++i)rc[i]=w.s[i]*w.z[i];};
    complementarity();
    check(w.factorize(0),"initial condensed factor failed: "+w.message);

    // Inspect a real raw backsolve before expanded refinement. A tiny backend
    // error does not certify reconstructed dz when z/s has erased Hessian
    // digits in the condensed Gram coefficient. All four Newton equations
    // below are checked independently using the preserved auxiliary Hessian.
    auto expanded_residual=[&](const Direction& d) {
      Vec dual=w.rd;
      axpy(d.x[0],auxiliary_column0,dual);axpy(d.x[1],auxiliary_column1,dual);
      for(int i=0;i<w.mi;++i)axpy(d.z[i],gradient(w.inequalities[i],w.x,false),dual);
      for(int e=0;e<w.me;++e)axpy(d.lambda[e],gradient(w.equalities[e],w.x,false),dual);
      double error=inf_norm(dual),scale=std::max(1.0,inf_norm(w.rd));
      for(int e=0;e<w.me;++e) {
        error=std::max(error,std::abs(w.h[e]+dot(gradient(w.equalities[e],w.x,false),d.x)));
        scale=std::max(scale,std::abs(w.h[e]));
      }
      for(int i=0;i<w.mi;++i) {
        const double jdx=dot(gradient(w.inequalities[i],w.x,false),d.x);
        error=std::max({error,std::abs(w.rp[i]+jdx+d.s[i]),
                       std::abs(rc[i]+w.z[i]*d.s[i]+w.s[i]*d.z[i])});
        scale=std::max({scale,std::abs(w.rp[i]),std::abs(rc[i])});
      }
      return error/scale;
    };
    const auto raw=w.direction(rc,w.rd,w.h,w.rp,1);
    check(expanded_residual(raw)>1e-3,
          "near-active fixture did not expose cancellation in reconstructed bound dual");
    const int old_dimension=w.dim;
    check(w.recover_condensation("original-direction-residual"),
          "actual expanded-residual failure did not permit singleton-bound promotion");
    check(w.dim==old_dimension+1 && w.jr[bound].dense>=0 &&
          ns.promoted_bounds==1 && ns.promoted_inequalities==0,
          "recovery did not promote precisely the near-active singleton bound");
    check(same_hessian() && w.inequalities[native].model_active,
          "border promotion replaced or altered the auxiliary Hessian");
    check(w.factorize(0),"bordered factor failed: "+w.message);
    const auto recovered=w.direction(rc,w.rd,w.h,w.rp);
    check(recovered.residual<=st.kkt_backward_error_reject &&
          expanded_residual(recovered)<=st.kkt_backward_error_reject,
          "bordered direction failed the unchanged expanded Newton tolerance");
    check(w.regularization==0,"equivalent border recovery introduced primal regularization");

    const int dimension=w.dim,border=w.jr[bound].dense;
    std::vector<std::pair<int,int>> pattern;
    for(const auto& entry:w.kkt)pattern.push_back({entry.i,entry.j});
    const int analyses=w.factor->symbolic_analyses(),reuses=w.factor->symbolic_reuses();
    for(int iteration=0;iteration<3;++iteration) {
      w.x[0]+=1e-6;w.s[bound]*=10;w.z[bound]+=0.125;
      w.evaluate();w.update_matrix();complementarity();
      check(w.dim==dimension && w.jr[bound].dense==border,"bound border was not persistent");
      std::vector<std::pair<int,int>> current;
      for(const auto& entry:w.kkt)current.push_back({entry.i,entry.j});
      check(current==pattern,"successive numeric updates changed the promoted KKT pattern");
      check(same_hessian(),"numeric update lost auxiliary Hessian curvature");
      check(w.factorize(0),"persistent bordered refactorization failed: "+w.message);
      const auto direction=w.direction(rc,w.rd,w.h,w.rp);
      check(expanded_residual(direction)<=st.kkt_backward_error_reject,
            "persistent bound border violated an expanded Newton equation");
    }
    if(backend=="mumps")
      check(w.factor->symbolic_analyses()==analyses && w.factor->symbolic_reuses()>=reuses+3,
            "persistent bound border repeated MUMPS symbolic analyses");
  }

  // The legacy route retains its historical singleton exclusion.
  auto legacy=settings();legacy.dca_engine="filter";
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,legacy,ns,costs);DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  w.s[0]=2;w.z[0]=0.25;w.s[1]=1e-24;w.z[1]=1;w.evaluate();w.update_matrix();
  check(!w.recover_condensation("original-direction-residual") && w.jr[1].dense<0,
        "R4 bound recovery changed the legacy controller");
}

void parallel_model_parity() {
#ifdef LAZYIPDCA_HAVE_OPENMP
  struct RestoreThreads {
    int count=omp_get_max_threads(),dynamic=omp_get_dynamic();
    ~RestoreThreads(){omp_set_num_threads(count);omp_set_dynamic(dynamic);}
  } restore;
  omp_set_dynamic(0);
  auto p=nonlinear_problem();
  const auto base=p.inequalities.front();
  p.inequalities.clear();
  // More than one dynamic scheduling block, with distinct negative modes.
  for(int i=0;i<1032;++i) {
    auto q=base;q.name="parallel_inequality_"+std::to_string(i);
    q.constant-=0.01*i;q.hessian.entries[0].v-=0.025*(i%13);
    if(i%3==0)q.hessian.entries.push_back({0,1,0.017*(i%7)});
    p.inequalities.push_back(std::move(q));
  }
  QuadraticFunction retained_equality;retained_equality.name="retained_nonlinear_equality";
  retained_equality.constant=-1;retained_equality.linear=Vec(3,0);
  retained_equality.hessian.n=3;retained_equality.hessian.entries={{0,1,1}};
  p.equalities.push_back(retained_equality);
  p.validate();auto st=settings();st.lazy_newton_reduction=true;
  auto snapshot=[&](int threads) {
    omp_set_num_threads(threads);
    LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
    Workspace w(p,st,ns,costs);DcaFilterStatistics stats;
    DcaInequalityModels models(w,stats);models.build();
    check(models.curvature_threads==threads,"parallel majorant builder did not use the requested OpenMP team");
    Vec values;
    w.update_geometry();
    values.insert(values.end(),w.tvalue.begin(),w.tvalue.end());
    for(const auto* rows:{&w.jr,&w.ar}) for(const auto& row:*rows)
      values.insert(values.end(),row.values.begin(),row.values.end());
    for(auto& q:w.inequalities) {
      for(auto e:q.model_hessian) {
        values.push_back(e.i);values.push_back(e.j);values.push_back(e.v);
      }
      for(const Vec& x:std::vector<Vec>{{2.0,0.5,4.25},{1.3,0.1,2.0},{-0.3,1.7,-2.5}}) {
        values.push_back(q.at(x));
        const auto g=gradient(q,x,false);values.insert(values.end(),g.begin(),g.end());
      }
    }
    Direction d;d.x={0.1,-0.07,0.03};d.lambda=Vec(w.me,0.01);
    d.s=Vec(w.mi,0.002);d.z=Vec(w.mi,0.001);
    DcaMajorantBarrierPath path(w,d);
    for(double alpha:{0.0,0.25,1.0}) {
      const auto point=path.point(alpha,0.1);
      check(point.positive&&point.finite,"parallel barrier fixture became invalid");
      values.insert(values.end(),{point.objective,point.phi,point.phi_change,
        point.theta,point.primal,point.primal_internal,path.slope(0.1)});
    }
    w.x={1.3,0.1,2.0};models.transport_slacks();
    values.insert(values.end(),w.s.begin(),w.s.end());
    const auto built=models.curvature_builds;models.build();
    w.update_geometry();
    values.insert(values.end(),w.tvalue.begin(),w.tvalue.end());
    for(const auto* rows:{&w.jr,&w.ar}) for(const auto& row:*rows)
      values.insert(values.end(),row.values.begin(),row.values.end());
    check(models.curvature_builds==built,"parallel reanchor repeated constant Hessian correction");
    check(stats.curvature_factors==0 && stats.rho_refinement_trials==0,
          "parallel model preparation invoked inertia or rho refinement");
    values.push_back(models.rho_max);values.push_back(models.curvature_builds);
    values.push_back(models.curvature_reuses);
    return values;
  };
  const auto serial=snapshot(1),parallel=snapshot(4);
  check(serial==parallel,"one/four-thread majorants changed coefficients, gradients or transported slacks");

  auto oracle=scalar_problem();
  omp_set_num_threads(1);const auto first=solve_filter_dca(oracle,st);
  omp_set_num_threads(4);const auto second=solve_filter_dca(oracle,st);
  check(first.status==Status::Solved && second.status==first.status &&
        first.x==second.x && first.objective==second.objective &&
        first.certificate_inequality_multipliers==second.certificate_inequality_multipliers &&
        first.outer_iterations==second.outer_iterations && first.inner_iterations==second.inner_iterations,
        "one/four-thread solve changed scientific endpoint or iteration count");
#endif
}

void exhausted_budget_is_not_success() {
  auto p=scalar_problem();auto st=settings();
  st.max_outer_iterations=1;st.dca_filter_inner_max=1;
  auto capped=solve_filter_dca(p,st);
  check(capped.status!=Status::Solved && capped.status!=Status::SolvedAcceptable &&
        capped.status!=Status::SolvedAcceptableOnFailure,
        "exhausted accepted-step budget was promoted to success");
  check(capped.dca_filter.majorant_inner_convergences==0,
        "one inner step was falsely declared a converged model");
  st.dca_filter_inner_max=100;st.time_limit_seconds=1e-12;
  auto timed=solve_filter_dca(p,st);
  check(timed.status==Status::TimeLimit,"exhausted time budget did not report TimeLimit");
  check(timed.dca_filter.majorant_inner_convergences==0,
        "time-limit exit invented an auxiliary convergence");
}
}

int main() {
  try {
    precise_majorant_values_preserve_legacy_and_gradients();
    precise_majorant_gradients_and_weighted_stationarity();
    infeasible_initial_slacks_preserve_capped_duals_and_center_products();
    cached_residual_refresh_matches_full_evaluation();
    model_contract_and_transport();
    accepted_original_state_survives_discard_and_cached_reanchor();
    single_accepted_step_exports_exact_affine_slack_and_dual();
    primal_centering_gate_excludes_only_the_original_dual();
    accepted_primal_centered_point_reduces_mu_despite_large_dual();
    auxiliary_oracle_and_original_certificate();
    pure_dual_step_requires_and_reaches_the_original_certificate();
    outer_newton_preserves_its_single_direction_when_feasibility_worsens();
    tight_paired_spectrum_majorant();
    fixed_barrier_newton_escapes_predictor_corrector_false_fixed_point();
    regularization_memory_requires_optimality_progress();
    cancelled_extended_newton_equations();
    near_active_bound_reconstruction();
    parallel_model_parity();
    exhausted_budget_is_not_success();
    std::cout<<"PASS R7 primal-centering gate experiment: original f/h, convex tangent inequalities, persistent rhoI, "
      "original filter and unchanged accepted slacks, legacy transport, cached reanchoring and budget exits\n";
  } catch(const std::exception& e) {
    std::cerr<<"FAIL R7 primal-centering gate experiment: "<<e.what()<<'\n';return 1;
  }
}
