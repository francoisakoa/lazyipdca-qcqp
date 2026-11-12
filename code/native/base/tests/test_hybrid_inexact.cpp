#include "lazyipdca_qcqp/hybrid_inexact.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
int main(){
  const auto startup_plan=hybrid_inner::initial_interior_start(
      {-.0004,-2,42},{.01,2,1},{.01,.00005,.33},{0,1});
  require(startup_plan.eligible&&startup_plan.slack[0]==.0004&&startup_plan.slack[1]==2,
      "strict first-model interior startup was not constructed");
  require(std::abs(startup_plan.target_mu-1e-4)<1e-19&&std::abs(startup_plan.dual[0]-.25)<1e-15,
      "startup_plan positive complementarity mean was not preserved");
  require(startup_plan.slack[2]==1&&startup_plan.dual[2]==.33,"startup_plan startup changed inactive auxiliary variables");
  for(int i:{0,1})require(std::abs(startup_plan.slack[i]*startup_plan.dual[i]-startup_plan.target_mu)<1e-19,
      "startup_plan model is not centered at its recorded mean");
  require(!hybrid_inner::initial_interior_start({0},{1},{1},{0}).eligible,
      "non-strict model received a zero slack");
  require(!hybrid_inner::initial_interior_start({1},{1},{1},{0}).eligible,
      "violated inequality received an invented interior startup");
  require(!hybrid_inner::initial_interior_start({-1},{1},{1},{0,0}).eligible,
      "duplicated active row silently changed its startup weight");
  require(!hybrid_inner::initial_interior_start({-1},{1},{1},{1}).eligible,
      "invalid active row entered startup");
  require(!hybrid_inner::initial_interior_start({-1},{1},{NAN},{0}).eligible,
      "nonfinite startup_plan multiplier accepted");
  require(!hybrid_inner::initial_interior_start({-1},{1e308},{1e308},{0}).eligible,
      "overflowed startup_plan mean accepted");
  {
    // Near a degenerate equality/bound face, z amplifies a tiny primal
    // reconstruction mismatch into a rejected complementarity equation.
    const auto b=hybrid_inner::balance_slack_direction(1e-16L,1e-6L,1e9,-1e-16);
    require(b.changed&&b.before>8e-7L&&b.after<1e-14L,
        "auxiliary reconstruction did not balance both original equations");
    for(double z:{1e-12,1e-3,1.,1e3,1e12})
      for(long double a:{-3.L,0.L,1e-16L,2.L})
        for(long double c:{-7.L,0.L,1e-6L,4.L}) {
          const auto r=hybrid_inner::balance_slack_direction(a,c,z,static_cast<double>(-a));
          require(r.after<=r.before,"balanced slack worsened unchanged expanded audit");
          const long double exact=-(a+c)/(1+static_cast<long double>(z));
          const double center=static_cast<double>(exact);
          auto error=[&](double v){return std::max(std::abs(a+v),std::abs(c+static_cast<long double>(z)*v));};
          for(double candidate:{center,std::nextafter(center,-INFINITY),std::nextafter(center,INFINITY)})
            require(r.after<=error(candidate),"representable neighbor improves selected slack");
        }
    require(!hybrid_inner::balance_slack_direction(0,0,1,0).changed,
        "zero auxiliary equations invented movement");
    for(double invalid:{-1.,0.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
      require(!hybrid_inner::balance_slack_direction(1,1,invalid,-1).changed,
          "invalid multiplier changed a slack direction");
  }
  hybrid_inner::FullResidualL2 before_full,after_full;
  before_full.add(.09,1);after_full.add(.089,1);
  for(int j=0;j<17;++j){before_full.add(.001,1);after_full.add(.01,1);}
  require(.089<.09&&after_full.norm()>before_full.norm(),
      "family maximum hides aggregate nonlinear KKT growth counterexample lost");
  hybrid_inner::FullResidualL2 normalized;
  normalized.add(3,3);normalized.add(-4,2);
  require(std::abs(normalized.norm()-std::sqrt(5.))<1e-15,"full residual has incorrect scaling");
  hybrid_inner::FullResidualL2 invalid;
  invalid.add(1,0);require(!std::isfinite(invalid.norm()),"zero-scale residual accepted");
  hybrid_inner::FullResidualL2 nonfinite;
  nonfinite.add(INFINITY,1);require(!std::isfinite(nonfinite.norm()),"nonfinite full residual accepted");
  require(hybrid_inner::model_penalty_seed(99.5)>99.5,"penalty not above dual estimate");
  require(hybrid_inner::model_penalty_seed(99.5)<101,"stale penalty inherited");
  for(double theta:{1.,1e-6,1e-12,1e-18,1e-100}) {
    const auto sigma=hybrid_inner::descent_penalty(1.,0.,theta);
    require(sigma<2,"vanishing residual creates absolute penalty blow-up");
    for(double ratio:{-2.,0.,1.,1000.}) {
      const double slope=ratio*theta;
      const auto p=hybrid_inner::descent_penalty(1.,slope,theta);
      require(static_cast<long double>(slope)-static_cast<long double>(p)*theta<0,
          "selected penalty does not give descent");
      require(p<2002,"penalty depends on residual magnitude rather than ratio");
    }
  }
  require(hybrid_inner::descent_penalty(5047.,0.,1e-12)==5047.,
      "penalty decreased inside a fixed model");
  require(hybrid_inner::descent_penalty(1.,2.,0.)==1.,"invented division at zero infeasibility");
  // f(x)=-x, g(x)=x-1 at x=1-delta is genuinely interior. The positive
  // multiplier 1 gives an approximate-KKT residual zero and gap delta;
  // this must be called epsilon-KKT, not an exact normal at that point.
  hybrid_inner::NormalColumn near_normal{{{0,1}},1,true};
  const auto near=hybrid_inner::epsilon_normal({-1},{near_normal},{-1e-13});
  require(near.dual_feasible&&near.fit.residual_l2==0&&near.fit.sweeps==0,
      "supplied nonnegative approximate KKT multipliers not reconstructed");
  require(hybrid_inner::check_model(near.fit.residual_l2,1,1,.25,0,0,1e-10,
      near.complementarity,1e-10,0,1).passed,"near-interior epsilon KKT rejected");
  const auto far=hybrid_inner::epsilon_normal({-1},{near_normal},{-1});
  require(far.dual_feasible&&!hybrid_inner::check_model(far.fit.residual_l2,1,1,.25,0,
      0,1e-10,far.complementarity,1e-10,0,1).passed,
      "far-interior noncomplementary normal accepted");
  auto negative=near_normal;negative.multiplier=-1;negative.values={{0,-1}};
  require(!hybrid_inner::epsilon_normal({-1},{negative},{0}).dual_feasible,
      "negative multiplier clipped into an epsilon certificate");
  const auto wrong_gradient=hybrid_inner::epsilon_normal({-2},{near_normal},{-1e-13});
  require(!hybrid_inner::check_model(wrong_gradient.fit.residual_l2,1,1,.25,0,
      0,1e-10,wrong_gradient.complementarity,1e-10,0,1).passed,
      "omitted objective/proximal gradient hidden by epsilon certificate");
  require(!hybrid_inner::epsilon_normal({NAN},{near_normal},{0}).dual_feasible,
      "nonfinite epsilon objective gradient accepted");
  require(!hybrid_inner::epsilon_normal({-1},{near_normal},{}).dual_feasible,
      "epsilon certificate with missing constraint value accepted");
  // An old almost-zero barrier must not trap an infeasible frozen model at
  // tiny slacks. The restart is dimensionally consistent under objective
  // and individual row scaling; it is not a feasibility acceptance test.
  for(double objective_scale:{.01,1.,100.}) {
    const auto reset=hybrid_inner::central_restart(8.8e-8,objective_scale,1e-10,1e-30);
    require(reset.needed,"degenerate inherited centrality not detected");
    require(std::abs(reset.target_mu/objective_scale-8.8e-8)<1e-22,
        "restart target changes under objective scaling");
    for(double row_scale:{1e-6,1.,1e6}) {
      const double s=row_scale*reset.normalized_slack_floor,z=reset.target_mu/s;
      require(s>0&&z>0&&std::isfinite(s)&&std::isfinite(z),"nonpositive central startup");
      require(std::abs(s*z/reset.target_mu-1)<5e-16,"restarted complementarity not centered");
      require(std::abs(s/row_scale-reset.normalized_slack_floor)<1e-19,
          "restart slack changes under row scaling");
    }
  }
  require(!hybrid_inner::central_restart(1e-8,1.,1e-10,1e-6).needed,
      "adequate complementarity was unnecessarily reset");
  const auto precision_recovery=hybrid_inner::central_restart(1e-8,1.,1e-10,1e-6,true);
  require(precision_recovery.needed&&precision_recovery.target_mu==1e-6,
      "observed numerical failure cannot rebalance at its current complementarity");
  require(!hybrid_inner::central_restart(0,1.,0,0).needed,"zero scale restart invented");
  for(const auto& bad:std::vector<std::vector<double>>{
      {NAN,1,1e-10,0},{1e-8,0,1e-10,0},{1e-8,1,-1,0},{1e-8,1,1e-10,-1}})
    require(!hybrid_inner::central_restart(bad[0],bad[1],bad[2],bad[3]).needed,
        "invalid central restart arguments accepted");
  using hybrid_inner::Residual;using hybrid_inner::check;
  using hybrid_inner::check_progress;
  // Reproduce the rounded no-progress acceptance observed on PGLIB case14:
  // a tiny alpha makes Armijo's decrement disappear, while the old rounding
  // allowance admitted an unchanged residual and merit for all 80 iterations.
  require(!check_progress(true,41.37040883354858,41.37040883354858,
      1,1,-1,6.4e-185,1e-4,64*2.220446049250313e-16).passed,
      "rounded no-progress trial accepted");
  require(!check_progress(false,1,.5,2,1,-1,.5,1e-4,0).passed,
      "unchanged represented state accepted");
  require(check_progress(true,1,.5,2,3,1,.5,1e-4,0).residual_contraction,
      "strict perturbed KKT contraction rejected");
  // For a linear residual and an exact Newton direction, a fraction t gives
  // R(x+t*d)=(1-t)*R(x). Backtracking alpha acts AFTER that fraction. The
  // sufficient decrease must use t*alpha, not alpha alone.
  require(!check_progress(true,1,1-1e-6,2,3,1,1,1e-4,0).residual_contraction,
      "small-fraction regression no longer distinguishes the old contract");
  require(check_progress(true,1,1-1e-6,2,3,1,1,1e-4,0,1e-6).residual_contraction,
      "exact Newton residual contraction rejected after fraction-to-boundary");
  require(check_progress(true,1,1-2.5e-7,2,3,1,.25,1e-4,0,1e-6).residual_contraction,
      "backtracking was not combined with the boundary fraction");
  require(!check_progress(true,1,1,2,3,1,1,1e-4,0,1e-6).passed,
      "small boundary fraction manufactured no-progress acceptance");
  require(!check_progress(true,1,1.000001,2,3,1,1,1e-4,0,1e-6).passed,
      "small boundary fraction accepted residual increase");
  for(double fraction:{0.,-1.,1.01,std::numeric_limits<double>::quiet_NaN()})
    require(!check_progress(true,1,.5,2,1,-1,.5,1e-4,0,fraction).passed,
        "invalid boundary fraction accepted");
  for(double fraction:{1.,.1,1e-6}) {
    require(check_progress(true,1,1,2,1,-1,.5,1e-4,0,fraction).merit_decrease,
        "boundary fraction changed an already normalized merit Armijo test");
    require(!check_progress(true,1,1,2,1.99999,-1,.5,1e-4,0,fraction).merit_decrease,
        "boundary fraction weakened the merit Armijo requirement");
  }
  require(check_progress(true,1,1,2,1,-1,.5,1e-4,0).merit_decrease,
      "represented Armijo merit decrease rejected");
  require(!check_progress(true,1,1,2,1.99999,-1,.5,1e-4,0).passed,
      "insufficient Armijo decrease accepted");
  require(!check_progress(true,1,1,1,1,-1,1,1e-4,1).passed,
      "large rounding allowance manufactured progress");
  require(!check_progress(true,NAN,0,2,1,-1,.5,1e-4,0).passed,
      "nonfinite progress quantities accepted");
  require(!check_progress(true,1,.5,2,1,-1,0,1e-4,0).passed,
      "zero alpha trial accepted");
  require(!check_progress(true,1,.5,2,1,-1,.5,1e-4,-1).passed,
      "negative rounding allowance accepted");
  // Merit and its derivative must be in the SAME units. Here the actual
  // normalized Armijo decrement is .0002; the represented decrease .0001
  // is insufficient. Leaving the slope in internally scaled units (.1)
  // incorrectly accepts it, which reproduces the former unit mismatch.
  require(!check_progress(true,1,1,10,9.9999,-1/.1,.2,1e-4,0).passed,
      "normalized Armijo test accepted insufficient decrease");
  require(check_progress(true,1,1,10,9.9999,-1,.2,1e-4,0).passed,
      "scaling-mismatch counterexample lost");
  for(double scale:{.01,.1,1.,10.,100.}) {
    const double merit_before=10*scale/scale,merit_after=9.99*scale/scale;
    const double smooth_slope=-1*scale,penalty=2,theta=.1*scale;
    const double normalized_slope=(smooth_slope-penalty*theta)/scale;
    require(check_progress(true,1,1,merit_before,merit_after,
        normalized_slope,.2,1e-4,0).passed,"common-unit scaling changed Armijo decision");
  }
  Residual initial{10,2,1};
  require(check({0.01,0.1,0.05},initial,1,.2,1e-10,4,5).passed,"valid relative model step rejected");
  require(!check({0.3,0.1,0.05},initial,1,.2,1e-10,4,5).passed,"dual residual gate bypassed");
  require(!check({0.01,.5,0.05},initial,1,.2,1e-10,4,5).passed,"model primal gate bypassed");
  require(!check({0.01,.1,.3},initial,1,.2,1e-10,4,5).passed,"centering gate bypassed");
  require(!check({0.01,.1,.05},initial,1,.2,1e-10,6,5).passed,"model increase accepted");
  require(!check({0.01,.1,.05},initial,0,.2,1e-10,4,5).passed,"zero step accepts nonzero residual");
  require(!check({NAN,.1,.05},initial,1,.2,1e-10,4,5).passed,"nonfinite residual accepted");
  // A small residual is not enough if the accepted alpha reduces the actual
  // displacement: always recompute the target for the accepted point.
  require(check({.1,.1,.05},initial,1,.2,1e-10,4,5).passed,"reference target");
  require(!check({.1,.1,.05},initial,.1,.2,1e-10,4,5).passed,"damped outer gate not recomputed");
  // Exact initial centering is not a demand to oversolve the new barrier
  // iterate to absolute tolerance. The O(mu) allowance disappears with mu.
  require(check({.001,0,1e-6},{1,0,0},.1,.2,1e-10,4,5,1).passed,"inexact central neighbourhood rejected");
  require(!check({.001,0,1e-6},{1,0,0},.1,.2,1e-10,4,5,0).passed,"zero-mu centering relaxed");
  require(!check({.001,0,.3},{1,0,0},.1,.2,1e-10,4,5,1).passed,"central neighbourhood exceeded");
  require(!check({.001,0,0},{1,0,0},.1,.2,1e-10,4,5,-1).passed,"negative barrier accepted");
  // A reduced-KKT primal diagonal is a proximal term on retained coordinates.
  // Its gradient must match its energy, and an eliminated pivot is untouched.
  const std::vector<double> anchor{1,2,3},point{1.3,-4,3.4};
  const std::vector<int> retained{0,2};std::vector<double> gradient(3,0);
  const double tau=.7,energy=hybrid_inner::retained_proximal(point,anchor,retained,tau,&gradient);
  require(std::abs(energy-.5*tau*(.3*.3+.4*.4))<1e-14,"wrong retained proximal energy");
  require(gradient[1]==0,"proximal term changed an eliminated pivot");
  for(int j:retained) {
    auto left=point,right=point;const double delta=1e-5;left[j]-=delta;right[j]+=delta;
    const double fd=(hybrid_inner::retained_proximal(right,anchor,retained,tau)-
        hybrid_inner::retained_proximal(left,anchor,retained,tau))/(2*delta);
    require(std::abs(fd-gradient[j])<1e-10,"proximal gradient/energy inconsistent");
  }
  // The Newton equation of a convex quadratic + frozen proximal term has
  // stationarity at the endpoint only when the SAME term is in the residual.
  // Without it, repeating the regularized step solves a different problem.
  const double q=2,c=-3,t=4,a=.5,newton=-(q*a+c)/(q+t),endpoint=a+newton;
  std::vector<double> stationarity{q*endpoint+c};
  hybrid_inner::retained_proximal({endpoint},{a},{0},t,&stationarity);
  require(std::abs(stationarity[0])<1e-14,"regularized Newton/residual model mismatch");
  require(std::abs(q*endpoint+c)>.1,"test failed to expose omitted proximal gradient");
  require(hybrid_inner::proximal_curvature_deficit(199,200)==0,"existing positive objective curvature ignored");
  require(hybrid_inner::proximal_curvature_deficit(199,0)==199,"missing equality proximal deficit");
  require(hybrid_inner::proximal_curvature_deficit(199,100)==99,"wrong equality proximal deficit");
  // A feasible numerical active set needs nonnegative inequality normals;
  // unconstrained equality normals may have either sign. Inactive normals
  // must be omitted by the caller, never added merely to cancel a gradient.
  hybrid_inner::NormalColumn inequality{{{0,-1}},0,true};
  auto fit=hybrid_inner::fit_normal({2},{inequality});
  require(fit.residual_l2<1e-14&&std::abs(fit.multipliers[0]-2)<1e-14,"active normal fit failed");
  fit=hybrid_inner::fit_normal({-2},{inequality});
  require(std::abs(fit.residual_l2-2)<1e-14&&fit.multipliers[0]==0,"negative inequality multiplier admitted");
  auto equality=inequality;equality.nonnegative=false;
  require(hybrid_inner::fit_normal({-2},{equality}).residual_l2<1e-14,"signed equality normal rejected");
  auto gate=hybrid_inner::check_model(.01,.2,1,.25,0,1e-12,1e-10,1e-12,1e-10,4,5);
  require(gate.passed&&std::abs(gate.kappa-.25)<1e-14,"valid manuscript finite-accuracy gate failed");
  require(!hybrid_inner::check_model(.01,.2,0,.25,0,0,1e-10,0,1e-10,4,5).passed,"uncertified zero margin accepted");
  require(!hybrid_inner::check_model(.01,.2,1,.5,0,0,1e-10,0,1e-10,4,5).passed,"kappa >= margin/2 accepted");
  require(!hybrid_inner::check_model(.1,.2,1,.25,0,0,1e-10,0,1e-10,4,5).passed,"non-relative stationarity accepted");
  require(!hybrid_inner::check_model(.01,.2,1,.25,0,1e-3,1e-10,0,1e-10,4,5).passed,"infeasible model accepted");
  require(!hybrid_inner::check_model(.01,.2,1,.25,0,0,1e-10,1e-3,1e-10,4,5).passed,"complementarity bypassed");
  // The centrality target uses the scaled SUM complementarity budget.
  // Duplicating rows must halve mu rather than double its normalized cost.
  for(double scale:{.01,1.,100.}) {
    const auto floor=hybrid_inner::complementarity_barrier_floor(scale,1e-10,7);
    const auto duplicate=hybrid_inner::complementarity_barrier_floor(scale,1e-10,14);
    require(floor.value>0&&floor.value<=.25L*scale*1e-10L/7,
        "floor exceeds normalized complementarity budget");
    require(std::abs(floor.value/scale-(.25e-10/7))<1e-25,
        "objective rescaling changed normalized barrier floor");
    require(duplicate.value==floor.value/2,
        "duplicated model rows changed total complementarity reservation");
  }
  require(hybrid_inner::complementarity_barrier_floor(1,1e-10,0).value==0,
      "empty model divided by zero or created barrier");
  require(hybrid_inner::complementarity_barrier_floor(1,0,7).value==0,
      "zero complementarity tolerance created positive floor");
  const double largest=std::numeric_limits<double>::max(),
      smallest=std::numeric_limits<double>::denorm_min();
  const auto saturated=hybrid_inner::complementarity_barrier_floor(largest,largest,1);
  require(saturated.saturated&&saturated.value==largest,"overflowed barrier floor was not safely capped");
  const auto underflow=hybrid_inner::complementarity_barrier_floor(smallest,smallest,7);
  require(underflow.underflow&&underflow.value==0,"underflow invented an excessive positive floor");
  const auto mixed=hybrid_inner::complementarity_barrier_floor(largest,smallest,7);
  require(mixed.value>0&&std::isfinite(mixed.value),"extreme cancelling scales lost a representable barrier floor");
  for(int bad=0;bad<6;++bad) {
    bool rejected=false;
    try {hybrid_inner::complementarity_barrier_floor(bad==0?0:bad==1?INFINITY:1,
        bad==2?-1:bad==3?NAN:1e-10,bad==4?-1:7,bad==5?1:.25);}
    catch(const std::invalid_argument&) {rejected=true;}
    require(rejected,"invalid barrier floor controls accepted");
  }
  // Mehrotra's affine predictor determines sigma by the cubic centrality
  // ratio; the corrector target also respects the common decay safeguard.
  auto prediction=hybrid_inner::mehrotra_target({2},{3},{-1},{-1},{0},1,1,6,.2);
  require(std::abs(prediction.affine_mu-2)<1e-14&&std::abs(prediction.sigma-1.0/27)<1e-14,
      "Mehrotra affine/cubic prediction inconsistent");
  require(std::abs(prediction.target-2.0/9)<1e-14,"Mehrotra target inconsistent");
  prediction=hybrid_inner::mehrotra_target({2},{3},{-1},{-1},{0},1,1,1e-100,.2);
  require(std::abs(prediction.target-2.0/9)<1e-14,"obsolete target prevents model recentering");
  prediction=hybrid_inner::mehrotra_target({2},{3},{-2},{-3},{0},1,1,1e-100,.2,1e-12);
  require(prediction.target==1e-12,"explicit finite-accuracy barrier floor missing");
  const auto budget_floor=hybrid_inner::complementarity_barrier_floor(.008,1e-10,1581);
  prediction=hybrid_inner::mehrotra_target({2},{3},{-2},{-3},{0},1,1,1e-100,.2,budget_floor.value);
  require(prediction.target==budget_floor.value&&prediction.target>1e-100,
      "old tiny target prevented a tolerance-derived centrality increase");
  const auto actual_gate=hybrid_inner::check_model(0,.1,1,.2,0,0,1e-10,
      1e-5,1e-10,0,1);
  require(!actual_gate.passed,"centrality floor bypassed actual complementarity certificate");
  std::cout<<"hybrid_inexact_gate PASS\n";
}
