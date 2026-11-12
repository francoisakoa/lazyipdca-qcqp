#include "../src/atomic_newton.cpp"
#include <iostream>
using namespace lazyipdca::qcqp;
int main(){
  int checks=0;auto require=[&](bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);};
  auto problem=[](){QCQPProblem p;p.name="generic_terminal_certificate";p.n=1;p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};p.objective.hessian.n=1;return p;};
  auto state=[](double x,int equality=0,int inequality=0){Result r;r.x={x};r.certificate_equality_multipliers.assign(equality,0);r.certificate_inequality_multipliers.assign(inequality,0);r.certificate_lower_multipliers={0};r.certificate_upper_multipliers={0};return r;};
  try {
    Settings settings;settings.time_limit_seconds=1;settings.feasibility_tolerance=settings.stationarity_tolerance=settings.dual_infeasibility_tolerance=settings.complementarity_tolerance=1e-6;
    for(const std::string mode:{"HH","HS","SH"}) {
      Settings st=settings;st.convexification_study=mode;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";st.linear_backend="mumps";st.linear_metis=true;st.dca_pair_elimination=false;
      auto p=problem();p.objective.constant=1;p.objective.linear={-2};p.objective.hessian.entries={{0,0,2}};p.validate();
      LazyNewtonStatistics statistics;detail::NewtonCostLedger ledger(statistics,Clock::now());Workspace w(p,st,statistics,ledger);
      const HybridInnerAnchor accepted(w);w.x={1};w.evaluate();
      HybridInnerOutcome outcome;outcome.model_gate.model_decrease=false;outcome.model_value=5;outcome.anchor_model_value=1;outcome.reason="inner-iteration-budget";
      require(outcome.model_value>outcome.anchor_model_value,"test did not block the proximal-model merit");
      hybrid_try_terminal_original(w,st,Clock::now(),1,outcome);
      require(outcome.terminal_original_audit.strict&&outcome.terminal_original_state,"true original KKT endpoint rejected");
      require(!outcome.success&&!outcome.model_gate.model_decrease,"terminal audit bypassed a model gate");
      accepted.restore(w);require(w.x==accepted.x,"private solver anchor not restorable");
      outcome.terminal_original_state->restore(w);require(w.x==Vec{1},"terminal certified primal not transported");
      // Real gate computation: an inexact affine feasibility residual can
      // pass every fixed-model KKT target yet increase the exact-L1 merit.
      // Its original KKT residual is already below the unchanged tolerances.
      auto flat=problem();QuadraticFunction linear;linear.linear={1};linear.hessian.n=1;flat.equalities={linear};flat.validate();
      LazyNewtonStatistics fs;DcaFilterStatistics ms;detail::NewtonCostLedger fl(fs,Clock::now());Workspace fw(flat,st,fs,fl);DcaInequalityModels fm(fw,ms);
      const HybridInnerAnchor flat_anchor(fw);fm.reanchor_from_cache();fw.x={1e-10};fw.lambda={-1e-9};fw.penalty=1e6;fw.evaluate();
      HybridInnerOutcome merit_only;merit_only.tau_prox=10;merit_only.proximal_coordinates=fw.retained;merit_only.reason="inner-iteration-budget";
      hybrid_model_relative_gate(fw,flat_anchor,st,merit_only);
      const auto& mg=merit_only.model_gate;
      require(mg.positive_margin&&mg.stationarity_ok&&mg.feasibility_ok&&mg.complementarity_ok&&!mg.model_decrease,"generic fixed model did not fail only merit");
      hybrid_try_terminal_original(fw,st,Clock::now(),1,merit_only);
      require(merit_only.terminal_original_audit.strict&&merit_only.terminal_original_state&&!merit_only.success,"merit-only failure hid a true original KKT point");
      fm.discard();flat_anchor.restore(fw);require(fw.x==flat_anchor.x,"merit-only proof changed the accepted anchor");
      // Export must rescale existing algorithm multipliers, not fit new ones.
      auto scaled=problem();scaled.x0={.5};scaled.objective.hessian.entries={{0,0,2000}};
      QuadraticFunction g;g.constant=100;g.linear={-100};g.hessian.n=1;scaled.inequalities={g};scaled.validate();
      st.lazy_newton_objective_gradient_target=st.lazy_newton_constraint_gradient_target=1;
      LazyNewtonStatistics ss;detail::NewtonCostLedger sl(ss,Clock::now());Workspace sw(scaled,st,ss,sl);
      sw.x={1};sw.z[0]=20*sw.objective.scale/sw.inequalities[0].scale;sw.evaluate();
      Result exported;sw.export_state(exported);
      require(std::abs(exported.certificate_inequality_multipliers[0]-20)<1e-12,"terminal multiplier rescaling wrong");
      HybridInnerOutcome scaled_outcome;scaled_outcome.reason="inner-iteration-budget";
      hybrid_try_terminal_original(sw,st,Clock::now(),1,scaled_outcome);
      require(scaled_outcome.terminal_original_audit.strict,"scaled original KKT endpoint rejected");
      // A feasible zero-objective point is KKT with zero multipliers even
      // when its inherited barrier state is far from complementary.
      auto pure=problem();QuadraticFunction side;side.constant=-1;side.linear={1};side.hessian.n=1;
      pure.inequalities={side};pure.validate();
      LazyNewtonStatistics zs;detail::NewtonCostLedger zl(zs,Clock::now());Workspace zw(pure,st,zs,zl);
      const Vec zx=zw.x;require(inf_norm(zw.z)>0,"zero-dual test did not inherit barrier multipliers");
      require(hybrid_try_zero_gradient_certificate(zw,st,Clock::now()),"feasible zero objective was not certified");
      require(zw.x==zx&&inf_norm(zw.z)==0,"zero-gradient certificate moved primal or retained barrier dual");
      zw.x={2};zw.z.assign(zw.mi,1);zw.evaluate();const Vec rejected_z=zw.z;
      require(!hybrid_try_zero_gradient_certificate(zw,st,Clock::now())&&zw.z==rejected_z,
          "zero objective hid primal violation or failed rollback");
      zw.x={0};zw.objective.linear={{0,1}};zw.objective.compile();zw.evaluate();
      require(!hybrid_try_zero_gradient_certificate(zw,st,Clock::now())&&zw.z==rejected_z,
          "nonzero objective gradient was discarded");
      // f=0 does not certify an infeasible nonlinear equality.
      auto infeasible=problem();QuadraticFunction h;h.constant=-1;h.linear={0};h.hessian.n=1;h.hessian.entries={{0,0,2}};infeasible.equalities={h};
      auto r=state(0,1);auto audit=detail::audit_hybrid_terminal_original(infeasible,st,r,[](){return 0.;});
      require(audit.attempted&&!audit.strict&&audit.primal==1,"zero objective hid primal violation");
      // Failure must leave the private state unchanged and retain no endpoint.
      const Vec saved=sw.x;sw.z[0]=0;sw.evaluate();HybridInnerOutcome rejected;
      hybrid_try_terminal_original(sw,st,Clock::now(),1,rejected);
      require(!rejected.terminal_original_state&&sw.x==saved,"failed terminal audit changed the primal state");
    }
    auto p=problem();p.objective.hessian.entries={{0,0,1}};auto bad=state(.01);
    require(!detail::audit_hybrid_terminal_original(p,settings,bad,[](){return 0.;}).strict,"stationarity violation accepted");
    auto comp=problem();comp.objective.linear={1};QuadraticFunction g;g.linear={-1};g.hessian.n=1;comp.inequalities={g};auto cr=state(.01,0,1);cr.certificate_inequality_multipliers={1};
    require(!detail::audit_hybrid_terminal_original(comp,settings,cr,[](){return 0.;}).strict,"complementarity violation accepted");
    comp.objective.linear={-1};cr=state(0,0,1);cr.certificate_inequality_multipliers={-1};
    require(!detail::audit_hybrid_terminal_original(comp,settings,cr,[](){return 0.;}).strict,"negative dual accepted");
    auto zero=problem();auto zr=state(0);
    auto before=detail::audit_hybrid_terminal_original(zero,settings,zr,[](){return 1.;});
    require(before.expired&&!before.attempted&&!before.strict,"expired pre-audit deadline accepted");
    int clock=0;auto during=detail::audit_hybrid_terminal_original(zero,settings,zr,[&](){return clock++?1.:0.;});
    require(during.expired&&during.attempted&&!during.strict,"deadline crossed during audit accepted");
    auto ready=detail::audit_hybrid_terminal_original(zero,settings,zr,[](){return .1;});
    require(ready.strict&&detail::hybrid_terminal_commit_open(ready,.2,1),"valid timely terminal commit rejected");
    require(!detail::hybrid_terminal_commit_open(ready,1,1),"deadline crossed before commit accepted");
    Settings tighter=settings;tighter.dual_infeasibility_tolerance=1e-8;auto tr=state(1e-7);
    require(!detail::audit_hybrid_terminal_original(p,tighter,tr,[](){return 0.;}).strict,"raw dual tolerance relaxed");
    std::cout<<"hybrid_terminal_certificate PASS checks="<<checks<<" modes=HH,HS,SH model_merit_unchanged=1 strict_original_only=1 multiplier_rescaling=1 deadlines=1\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
