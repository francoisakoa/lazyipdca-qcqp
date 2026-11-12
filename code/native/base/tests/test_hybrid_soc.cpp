#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
int checks=0;
void require(bool condition,const char* message) {
  ++checks;if(!condition)throw std::runtime_error(message);
}
void rounded_original_defect() {
  QCQPProblem p;p.name="rounded_soc";p.n=2;
  p.x0={.9999999999987516,-1.580141875776555e-6};
  p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.constant=-100;p.objective.linear={-1,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,200},{1,1,200}};
  QuadraticFunction h;h.constant=-1;h.linear={0,0};h.hessian.n=2;
  h.hessian.entries={{0,0,2},{1,1,2}};p.equalities.push_back(h);
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.linear_backend="mumps";st.linear_metis=true;st.verbose=false;st.live_display_level=0;
  st.lazy_newton_reduction=false;st.lazy_newton_atomic_regularization=false;
  st.lazy_newton_objective_gradient_target=1000;st.lazy_newton_constraint_gradient_target=1000;
  LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());
  Workspace w(p,st,stats,costs);
  w.objective.majorant_values_extended=true;w.objective.majorant_gradients_extended=true;
  w.equalities[0].majorant_values_extended=true;w.equalities[0].majorant_gradients_extended=true;
  w.evaluate();w.update_matrix();
  require(w.factorize(0),"SOC fixture factorization failed");
  Direction step;step.x={.9999999999987641-p.x0[0],-1.5722412288183977e-6-p.x0[1]};
  step.lambda={0};
  const DcaMajorantBarrierPath path(w,step);
  Vec trial=w.x;axpy(1,step.x,trial);
  const double full=w.equalities[0].original_at(trial);
  const double curvature=double(path.equalities[0].quadratic);
  require(full>curvature&&curvature>0,"full defect did not include affine rounding residual");
  const auto original_x=w.x,original_lambda=w.lambda;
  const auto attempts=stats.factor_attempts;auto* factor=w.current_factor;
  const auto correction=w.direction({},Vec(w.n,0),{full},{},0,true,DirectionPurpose::Complete,true);
  require(stats.factor_attempts==attempts&&w.current_factor==factor,"SOC refactored instead of reusing factors");
  require(w.x==original_x&&w.lambda==original_lambda,"frozen correction changed anchor");
  Direction combined=step;axpy(1,correction.x,combined.x);axpy(1,correction.lambda,combined.lambda);
  DcaMajorantBarrierPath corrected(w,combined);
  const auto before=DcaMajorantBarrierPath::current(w,0);
  const auto rejected=path.rounded_point(1,0),accepted=corrected.rounded_point(1,0);
  DcaMajorantBarrierFilter filter(before.theta);
  require(!filter.diagnose(before,rejected,path.slope(0),1,1e-4).accepted,"uncorrected rounding increase passed original filter");
  require(filter.diagnose(before,accepted,path.slope(0),1,1e-4).accepted,"full original-residual SOC failed original filter");
  require(accepted.phi_change<0&&accepted.theta<rejected.theta,"SOC did not improve actual rounded original point");
  Direction curvature_combined=step;
  const auto curvature_correction=w.direction({},Vec(w.n,0),{curvature},{},0,true,DirectionPurpose::Complete,true);
  axpy(1,curvature_correction.x,curvature_combined.x);
  Vec curvature_trial=w.x;axpy(1,curvature_combined.x,curvature_trial);
  require(curvature_trial==trial,"curvature-only SOC unexpectedly fixed representable defect fixture");
}
void restored_anchor_requires_matching_numeric_factors() {
  for(const std::string mode:{"HH","HS","SH"})for(bool reassemble_before_stale:{false,true}) {
    QCQPProblem p;p.name="hybrid_soc_factor_lifecycle";p.n=2;
    p.x0={.5,.5};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
    p.objective.linear={0,0};p.objective.hessian.n=2;
    p.objective.hessian.entries={{0,0,2},{1,1,2}};
    QuadraticFunction h;h.constant=-1;h.linear={1,1};h.hessian.n=2;
    p.equalities.push_back(h);
    QuadraticFunction g;g.constant=-4;g.linear={0,0};g.hessian.n=2;
    g.hessian.entries={{0,0,2}};p.inequalities.push_back(g);
    Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
    st.convexification_study=mode;st.linear_backend="mumps";st.linear_metis=true;
    st.verbose=false;st.live_display_level=0;st.lazy_newton_reduction=false;
    st.lazy_newton_atomic_regularization=false;st.lazy_newton_lazy_constraints=false;
    st.lazy_newton_objective_gradient_target=1000;st.lazy_newton_constraint_gradient_target=1000;
    LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());
    Workspace w(p,st,stats,costs);w.s={3.75};w.z={1};w.evaluate();
    const HybridInnerAnchor anchor(w);const double tau=.1;
    // A valid private Newton point need not have the accepted anchor's s/z,
    // inequality Jacobian or dual-weighted Hessian. This is exactly the state
    // lifecycle of a successful private inner solve followed by anchor restore.
    w.x={.9,.1};w.s={1e-10};w.z={1e6};w.evaluate();w.update_matrix();
    require(w.factorize(tau),"private fixture factorization failed");
    anchor.restore(w);
    // false reproduces the private affine repair after a moved Newton point:
    // evaluate updated function gradients but left old matrix caches/factors.
    // true reproduces the old outer-SOC restore+reassembly without refactoring.
    if(reassemble_before_stale)w.update_matrix();
    const auto anchor_x=w.x,anchor_s=w.s,anchor_z=w.z,anchor_lambda=w.lambda;
    const auto before_stale=stats.factor_attempts;
    bool stale_rejected=false;
    try {w.direction({0},Vec(2,0),{.25},{0},0,true,DirectionPurpose::Complete,true);}
    catch(const std::runtime_error& error) {
      stale_rejected=std::string(error.what()).find("frozen-factor correction")!=std::string::npos;
    }
    require(stale_rejected,"mismatched private factors unexpectedly passed strict correction audit");
    require(stats.factor_attempts==before_stale,"frozen stale solve changed factors");
    w.update_matrix();
    require(w.factorize(tau,true),"matching frozen anchor factorization failed");
    const auto fresh_attempts=stats.factor_attempts;auto* fresh_factor=w.current_factor;
    const Direction correction=w.direction({0},Vec(2,0),{.25},{0},0,true,DirectionPurpose::Complete,true);
    require(std::isfinite(correction.residual)&&correction.residual<=st.kkt_backward_error_reject,
        "matching factor correction failed the unchanged strict audit");
    require(stats.factor_attempts==fresh_attempts&&w.current_factor==fresh_factor,
        "SOC refinement did not retain the matching frozen factors");
    require(w.regularization==tau&&w.objective.study_model_scalar==0&&
        w.inequalities[0].study_model_scalar==0&&w.objective.model_atoms.empty()&&
        w.inequalities[0].model_atoms.empty(),"frozen correction changed convex source curvature");
    require(w.x==anchor_x&&w.s==anchor_s&&w.z==anchor_z&&w.lambda==anchor_lambda,
        "factor lifecycle changed accepted primal/dual state");
    bool changed_tau_rejected=false;
    try {w.factorize(2*tau,true);}catch(const std::invalid_argument&){changed_tau_rejected=true;}
    require(changed_tau_rejected&&w.regularization==tau,
        "frozen factorization accepted a different model tau");
    require(fresh_attempts==before_stale+1,"SOC performed more than one explicit anchor factorization");
  }
}
}
void adjacent_rounding_is_only_a_candidate() {
  const Vec before={1.,-1.},small={-1e-20,1e-20};Vec point=before;
  require(hybrid_adjacent_realization(point,before,small),"sub-ulp projection realization missing");
  require(point[0]==std::nextafter(1.,-INFINITY)&&point[1]==std::nextafter(-1.,INFINITY),"realization does not follow solved correction");
  const Vec changed=point;
  require(!hybrid_adjacent_realization(point,before,small)&&point==changed,"represented correction was adjusted twice");
  point=before;require(!hybrid_adjacent_realization(point,before,Vec(2,0))&&point==before,"zero correction invented a displacement");
}
int main(){rounded_original_defect();restored_anchor_requires_matching_numeric_factors();adjacent_rounding_is_only_a_candidate();std::cout<<"hybrid_soc PASS checks="<<checks<<" rounded_original_defect=1 unchanged_factors=1 original_filter_required=1\n";}
