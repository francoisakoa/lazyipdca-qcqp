#include "../src/small_step_soc_policy.hpp"
#include <iostream>
#include <limits>
#include <string>
using lazyipdca::qcqp::detail::SmallStepSocPolicy;
using lazyipdca::qcqp::detail::NonlinearStepSocPolicy;
int main(){int checks=0;auto check=[&](bool x,const char* why){++checks;if(!x)throw std::runtime_error(why);};
  SmallStepSocPolicy historical(0,.125);check(historical.enabled(),"default changed historical SOC");
  check(!historical.observe_accepted(.1)&&historical.enabled(),"historical latch changed");
  SmallStepSocPolicy p(4,.125);check(!p.enabled(),"SOC enabled before accepted evidence");
  for(int i=0;i<3;++i)check(!p.observe_accepted(.125)&&!p.enabled(),"patience was shortened");
  check(!p.observe_accepted(.126)&&p.streak()==0,"larger accepted step did not reset streak");
  for(int i=0;i<3;++i)check(!p.observe_accepted(.01),"latch activated too early");
  check(p.observe_accepted(.125)&&p.enabled()&&p.latched(),"four accepted steps did not latch SOC");
  check(!p.observe_accepted(1)&&p.enabled(),"latched recovery was accidentally disabled");
  for(double a:{0.,-1.,1.1,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}){
    bool rejected=false;try{p.observe_accepted(a);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"invalid accepted alpha admitted");}
  bool rejected=false;try{SmallStepSocPolicy invalid(-1,.125);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"invalid patience admitted");
  rejected=false;try{SmallStepSocPolicy invalid(4,0);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"invalid threshold admitted");
  NonlinearStepSocPolicy off(0,.125);
  for(int i=0;i<5;++i)check(!off.observe_accepted(.01,0.,1.,1.,0.)&&!off.enabled(),
      "disabled window policy changed historical admission");
  check(off.window_size()==0,"disabled policy accumulated a window");

  // Replay only the observed alpha/theta sequence at accepted iterations
  // 11--14. The log did not contain equality defect values: supply a material
  // synthetic defect to verify admission CONDITIONAL on that extra evidence.
  const double alphas[]={.017913084281394459,.14324590308020527,
      .028028456641146667,.17059973301388609};
  const double thetas[]={.0035192762987863524,.0037415171352442474,
      .12419455978809091,.12581433650032284,.28554767541960352};
  NonlinearStepSocPolicy window(4,.125);
  SmallStepSocPolicy consecutive(4,.125);
  for(int i=0;i<4;++i) {
    const bool fired=window.observe_accepted(alphas[i],thetas[i],thetas[i+1],.01,1e-12);
    check(fired==(i==3),"trace window triggered before full evidence or missed iteration 14");
    check(!consecutive.observe_accepted(alphas[i]),"trace unexpectedly passed consecutive latch");
  }
  check(window.enabled()&&window.latched()&&window.window_size()==4,"window did not latch");
  check(std::abs(window.mean_alpha()-.08994679425415812)<1e-15,"trace mean alpha wrong");
  check(std::string(window.reason())=="short-window-nonlinear-feasibility-loss","trigger reason absent");
  check(!window.observe_accepted(1.,1.,0.,0.,0.)&&window.enabled(),"latched SOC became unavailable");

  NonlinearStepSocPolicy improving(4,.125),affine(4,.125),tiny_defect(4,.125);
  for(int i=0;i<10;++i) {
    check(!improving.observe_accepted(.01,10.-i,9.-i,.01,1e-12),
        "improving original theta enabled extra SOC");
    check(!affine.observe_accepted(.01,i,i+1,0.,0.),"affine equality activated window SOC");
    check(!tiny_defect.observe_accepted(.01,i,i+1,1e-12,1e-12),"defect at floor activated SOC");
  }
  check(!improving.enabled()&&!affine.enabled()&&!tiny_defect.enabled(),"unqualified evidence latched SOC");

  NonlinearStepSocPolicy bounded(2,.125);
  check(!bounded.observe_accepted(.01,0.,1.,1.,0.),"partial history triggered SOC");
  check(!bounded.observe_accepted(1.,1.,2.,1.,0.),"large mean was admitted");
  check(!bounded.observe_accepted(.01,2.,3.,1.,0.),"old small step survived bounded window");
  check(bounded.observe_accepted(.01,3.,4.,1.,0.)&&bounded.window_size()==2,
      "old large step did not expire from window");
  NonlinearStepSocPolicy boundary(1,.125),rounding(1,.125);
  check(boundary.observe_accepted(.125,1.,2.,1.,0.),"mean threshold equality rejected");
  check(!rounding.observe_accepted(.125,1.,std::nextafter(1.,2.),1.,0.),
      "roundoff-only theta change activated SOC");

  NonlinearStepSocPolicy valid(4,.125);
  const double invalids[]={-1.,std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()};
  for(double value:invalids)for(int field=0;field<5;++field) {
    double data[]={.01,1.,2.,1.,0.};data[field]=value;
    bool bad=false;try{valid.observe_accepted(data[0],data[1],data[2],data[3],data[4]);}
      catch(const std::invalid_argument&){bad=true;}
    check(bad&&valid.accepted()==0&&valid.window_size()==0,"invalid observation changed policy state");
  }
  rejected=false;try{valid.observe_accepted(0.,1.,2.,1.,0.);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"zero accepted alpha admitted");
  rejected=false;try{NonlinearStepSocPolicy invalid(-1,.125);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"negative nonlinear window admitted");
  rejected=false;try{NonlinearStepSocPolicy invalid(4,0);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"zero nonlinear step threshold admitted");
  NonlinearStepSocPolicy restart(4,.125);
  for(int i=0;i<3;++i)restart.observe_accepted(.01,1,2,1,0);
  restart.clear_window();check(restart.window_size()==0&&!restart.enabled(),"cold retry retained stale window");
  check(!restart.observe_accepted(.01,1,2,1,0),"cold retry mixed old and new anchors");
  std::cout<<"small_step_soc_policy PASS checks="<<checks
      <<" historical_default=1 accepted_streak=1 threshold_boundary=1 latch=1 invalid_controls=1"
      <<" nonlinear_window=1 trace_alpha_theta_conditional_replay=1 affine_no_trigger=1 bounded_history=1\n";
}
