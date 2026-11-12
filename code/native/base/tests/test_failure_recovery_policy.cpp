#include "../src/failure_recovery_policy.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(){try{
  using detail::failure_recovery_budget;
  check(failure_recovery_budget(401,381,40).allowed,"mixed 401 incorrectly excluded");
  check(failure_recovery_budget(2070,1981,0).allowed,"2070 equality-only incorrectly excluded");
  check(failure_recovery_budget(2883,2883,0).allowed,"square recovery incorrectly excluded");
  check(!failure_recovery_budget(30000,30000,0).allowed,"large dense recovery not bounded");
  check(!failure_recovery_budget(std::numeric_limits<int>::max(),std::numeric_limits<std::size_t>::max(),0).allowed,"size estimate overflow");
  check(!failure_recovery_budget(0,1,0).allowed,"invalid dimension admitted");
  check(!failure_recovery_budget(5,0,4).allowed,"inequality-only model admitted");
  check(failure_recovery_budget(5,0,4,true).allowed,"explicit hybrid inequality-only recovery excluded");
  check(!failure_recovery_budget(5,0,0,true).allowed,"empty feasibility model admitted");
  // Curved equations may already be feasible while the remaining affine
  // equations still prevent the rejected Newton model from being usable.
  // Admission depends on the original residual, not on a nonzero Hessian.
  using detail::failed_linearization_recovery_eligible;
  for(int route=0;route<3;++route)
    check(failed_linearization_recovery_eligible(route==0,route==1,route==2,
        true,false,0.06697147697923356,1e-6),
        "material affine remainder was excluded from bounded recovery");
  check(!failed_linearization_recovery_eligible(false,false,false,true,false,1.,1e-6),
      "an ordinary iteration acquired failure recovery");
  check(!failed_linearization_recovery_eligible(true,false,false,false,false,1.,1e-6),
      "disabled restoration admitted");
  check(!failed_linearization_recovery_eligible(true,false,false,true,true,1.,1e-6),
      "spent anchor recovery restarted");
  for(double residual:{0.,1e-6,-1.,double(NAN),double(INFINITY)})
    check(!failed_linearization_recovery_eligible(true,false,false,true,false,residual,1e-6),
        "small or invalid original residual admitted");
  for(double residual:{1.0001e-6,5e-6,8.953425182234298e-5,9.811625089128917e-5,100*1e-6}) {
    check(failed_linearization_recovery_eligible(true,false,false,true,false,residual,1e-6),
        "rejected numerical direction abandoned a still-infeasible anchor");
    check(failed_linearization_recovery_eligible(false,true,false,true,false,residual,1e-6),
        "exhausted original filter abandoned a still-infeasible anchor");
    check(failed_linearization_recovery_eligible(false,false,true,true,false,residual,1e-6),
        "failed model abandoned a still-infeasible anchor");
    check(!failed_linearization_recovery_eligible(false,true,false,true,false,residual,1e-6,false),
        "disabled gap repair changed historical filter admission");
    check(!failed_linearization_recovery_eligible(false,false,true,true,false,residual,1e-6,false),
        "disabled gap repair changed historical model admission");
    check(failed_linearization_recovery_eligible(true,false,false,true,false,residual,1e-6,false),
        "disabled gap repair changed historical numerical recovery");
  }
  for(double tolerance:{0.,-1.,double(NAN),double(INFINITY)})
    check(!failed_linearization_recovery_eligible(true,false,false,true,false,1.,tolerance),
        "invalid feasibility tolerance admitted");
  using detail::majorant_barrier_continuation;
  check(majorant_barrier_continuation(1.,1e-8,10.)==.2,
      "accepted boundary progress skipped historical continuation");
  check(majorant_barrier_continuation(1.,1e-8,10.001)==1.,
      "insufficient progress reduced barrier");
  check(majorant_barrier_continuation(1.,.5,0.)==.5,
      "continuation crossed fixed barrier floor");
  check(majorant_barrier_continuation(.5,.5,0.)==.5,
      "barrier at floor changed");
  for(double residual:{-1.,double(NAN),double(INFINITY)})
    check(majorant_barrier_continuation(1.,1e-8,residual)==1.,
        "invalid continuation residual changed barrier");
  double recovery_mu=1.;
  // Each filter-accepted Phase-I anchor gets ONE historical update. The
  // decreasing original residual does not otherwise change the barrier.
  for(double residual:{1.58,.067,.00284,.0001205,5e-6}) {
    const double next=majorant_barrier_continuation(recovery_mu,1e-8,residual);
    check(next==std::max(1e-8,.2*recovery_mu),
        "accepted recovery anchor retained cold barrier or skipped a stage");
    recovery_mu=next;
  }
  using detail::failed_direction_proximal_retry_eligible;
  for(const char* reason:{"lazy-newton expanded Newton residual rejected: 1e-5",
      "backward error too large","nonfinite direction"})
    for(int used=0;used<3;++used)
      check(failed_direction_proximal_retry_eligible(reason,used,false,4,1.,1200.),
          "rejected numerical direction lost bounded explicit-proximal retry");
  const std::string numerical="expanded Newton residual rejected";
  check(!failed_direction_proximal_retry_eligible(numerical,3,false,4,1.,1200.),"retry allowance restarted");
  check(!failed_direction_proximal_retry_eligible(numerical,-1,false,4,1.,1200.),"invalid retry history admitted");
  check(!failed_direction_proximal_retry_eligible(numerical,0,true,4,1.,1200.),"atomic regularization replaced");
  check(!failed_direction_proximal_retry_eligible(numerical,0,false,1,1.,1200.),"unchargeable outer retry admitted");
  check(!failed_direction_proximal_retry_eligible(numerical,0,false,4,1200.,1200.),"expired retry admitted");
  check(!failed_direction_proximal_retry_eligible(numerical,0,false,4,NAN,1200.),"invalid clock admitted");
  for(const char* reason:{"original-filter-rejected","inertia mismatch","line-search failure"})
    check(!failed_direction_proximal_retry_eligible(reason,0,false,4,1.,1200.),
        "non-numerical failure acquired direction-precision retry");
  using detail::lower_proximal_probe_eligible;
  check(lower_proximal_probe_eligible(false,false,false,1e-11,1e-6,2,1.,1200.),
      "original-feasible exhausted filter lost downward probe");
  check(!lower_proximal_probe_eligible(true,false,false,0.,1e-6,2,1.,1200.),"downward probe repeated same anchor");
  check(!lower_proximal_probe_eligible(false,false,false,1.0001e-6,1e-6,2,1.,1200.),"infeasible anchor acquired downward probe");
  check(!lower_proximal_probe_eligible(false,true,false,0.,1e-6,2,1.,1200.),"square normal metric changed");
  check(!lower_proximal_probe_eligible(false,false,true,0.,1e-6,2,1.,1200.),"atomic matrix policy changed");
  check(!lower_proximal_probe_eligible(false,false,false,0.,1e-6,1,1.,1200.),"uncharged downward probe admitted");
  check(!lower_proximal_probe_eligible(false,false,false,0.,1e-6,2,1200.,1200.),"expired downward probe admitted");
  for(double primal:{-1.,double(NAN),double(INFINITY)})
    check(!lower_proximal_probe_eligible(false,false,false,primal,1e-6,2,1.,1200.),"invalid primal acquired downward probe");
  using detail::lower_proximal_proposal;
  check(lower_proximal_proposal(2e8,0.)==1.,"large auxiliary damping was not probed downward");
  check(std::abs(lower_proximal_proposal(.2,0.)-.02)<1e-17,"small auxiliary damping did not decrease geometrically");
  check(lower_proximal_proposal(20.,3.)==3.,"downward proposal crossed certified margin floor");
  check(lower_proximal_proposal(3.,3.)==3.,"certified floor was reduced");
  check(lower_proximal_proposal(2.,3.)==2.,"invalid below-floor matrix changed silently");
  for(double invalid:{-1.,double(NAN),double(INFINITY)})
    check(lower_proximal_proposal(20.,invalid)==20.,"invalid floor changed matrix");
  {
    bool pending=true;double memory=1.;
    check(detail::restore_lower_proximal_probe(pending,2e8,memory)&&!pending&&memory==2e8,
        "failed optional lower matrix did not request a rebuild of the original matrix");
    check(!detail::restore_lower_proximal_probe(pending,7.,memory)&&memory==2e8,
        "rollback repeated or changed an unrelated matrix");
    bool used=true;
    check(!lower_proximal_probe_eligible(used,false,false,0.,1e-6,10,1.,1200.),
        "rollback reopened the spent lower-probe budget");
    pending=true;bool invalid=false;
    try{detail::restore_lower_proximal_probe(pending,-1.,memory);}
    catch(const std::invalid_argument&){invalid=true;}
    check(invalid&&pending&&memory==2e8,"invalid rollback seed changed probe state");
  }
  {
    const Vec g{0.,-100.,7.},old_s{std::numeric_limits<double>::epsilon()*16,100.,2.},old_z{1.,.01,3.};
    const std::vector<bool> active{true,true,false};
    Vec s,z;long double gain=0;
    check(detail::direction_centering_candidate(g,active,old_s,old_z,6.4e-5,s,z,gain),
        "balanced auxiliary candidate rejected");
    check(s[0]==.008&&std::abs(s[0]*z[0]-6.4e-5)<1e-19&&
        s[1]==old_s[1]&&std::abs(s[1]*z[1]-6.4e-5)<1e-19,
        "candidate did not center active products or shrank slacks");
    check(s[2]==old_s[2]&&z[2]==old_z[2],"inactive auxiliary state changed");
    check(std::abs(gain-(std::log(static_cast<long double>(s[0]))-
        std::log(static_cast<long double>(old_s[0]))))<1e-17L,
        "geometric slack path lost exact logarithmic slope");
    const Vec kept_s=s,kept_z=z;const long double kept_gain=gain;
    check(!detail::direction_centering_candidate(g,active,old_s,old_z,0.,s,z,gain)&&
        s==kept_s&&z==kept_z&&gain==kept_gain,"invalid candidate mutated outputs");
    check(!detail::direction_centering_candidate(g,active,Vec{0,100,2},old_z,1.,s,z,gain),
        "noninterior accepted state admitted");
    check(!detail::direction_centering_candidate(g,active,Vec{1,100,2},old_z,1.,s,z,gain),
        "no-slack-progress state invented barrier progress");
  }
  detail::SharedPhase1RecoveryBudget shared(240);
  check(shared.remaining()==240,"initial late phase-I budget incorrect");
  shared.charge(13);check(shared.remaining()==227,"first recovery was not charged");
  check(shared.bind_deadline(160)==160&&shared.bind_deadline(220)==160,
      "new accepted anchor restarted shared recovery deadline");
  check(shared.bind_deadline(150)==150,"earlier external deadline not preserved");
  shared.charge(227);check(shared.remaining()==0,"successive recoveries exceeded global allowance");
  bool overrun=false;try{shared.charge(1);}catch(const std::runtime_error&){overrun=true;}
  check(overrun,"exhausted shared allowance silently restarted");
  check(detail::phase1_reopen_after_dca(true,true,true,1e-12),"certified changed DCA primal failed to rearm");
  check(!detail::phase1_reopen_after_dca(true,true,true,0),"duals-only or same-anchor update rearmed recovery");
  check(!detail::phase1_reopen_after_dca(true,false,true,1),"uncertified direction rearmed recovery");
  check(!detail::phase1_reopen_after_dca(true,true,false,1),"filter-rejected point rearmed recovery");
  check(!detail::phase1_reopen_after_dca(false,true,true,1),"legacy path acquired recovery policy");
  check(!detail::phase1_reopen_after_dca(true,true,true,NAN),"nonfinite primal rearmed recovery");
  {
    double best=detail::phase1_best_reference(INFINITY,.001915846);
    check(detail::phase1_reopen_after_dca(true,true,true,1),"changed accepted primal cannot rearm");
    // A degraded accepted DCA anchor is eligible, but cannot erase the
    // independently verified restoration history or weaken its next target.
    const double before=159.725;
    const double target=detail::phase1_feasibility_target(std::min(before,best),1e-6);
    check(std::abs(target-.0001915846)<1e-18,"rearm weakened best Phase-I contraction");
    check(!detail::phase1_verified_reduction(std::min(before,best),.83299,1e-6),
        "degraded restoration admitted after rearm");
    check(detail::phase1_verified_reduction(std::min(before,best),5.775e-5,1e-6),
        "genuine improvement of historical restoration reference rejected");
    best=detail::phase1_best_reference(best,5.775e-5);
    check(best==5.775e-5&&detail::phase1_best_reference(best,.1)==best,
        "a larger later point reset the best reference");
    check(detail::phase1_best_reference(best,1e-8)==1e-8,
        "new independently verified improvement not recorded");
    for(double bad:{double(NAN),-1.}) {
      bool invalid=false;try{detail::phase1_best_reference(bad,0);}catch(const std::invalid_argument&){invalid=true;}
      check(invalid,"invalid prior reference admitted");
    }
  }
  check(std::abs(detail::phase1_feasibility_target(0.4,1e-6)-0.04)<1e-16,"phase-I target not decade reduction");
  check(detail::phase1_feasibility_target(2e-6,1e-6)==1e-6,"phase-I target weakens final feasibility floor");
  check(detail::phase1_private_precision_context(true,1e-7,1e-6),"finer private phase lost relative descent");
  for(double target:{1e-6,1.,0.,-1.,double(NAN),double(INFINITY)})
    check(!detail::phase1_private_precision_context(true,target,1e-6),"coarse/invalid phase acquired relative descent");
  check(!detail::phase1_private_precision_context(false,1e-7,1e-6),"legacy phase acquired precision policy");
  const double private_target=detail::phase1_private_precision_target(1e-6,2.8e-7);
  check(private_target>0&&private_target<2.8e-7&&private_target<1e-6&&
      std::abs(private_target-2.8e-8)<1e-22,
      "original-filter precision was stopped at the final KKT tolerance");
  double target=1e-6,private_primal=2.8e-7;
  for(int step=0;step<12;++step) {
    const double next=detail::phase1_private_precision_target(target,private_primal);
    check(next>0&&next<target&&next<private_primal,
        "private continuation repeated a precision target");
    target=private_primal=next;
  }
  check(detail::phase1_private_precision_target(1e-6,0)==0,
      "zero physical residual created an endlessly repeated private target");
  const double denorm_target=std::numeric_limits<double>::denorm_min();
  check(detail::phase1_private_precision_target(denorm_target,denorm_target)==0,
      "minimum representable target failed to terminate");
  for(int invalid=0;invalid<4;++invalid) {
    bool rejected=false;try{detail::phase1_private_precision_target(
        invalid==0?0:invalid==1?NAN:1e-6,invalid==2?-1:invalid==3?INFINITY:1e-7);}
    catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"invalid private precision controls admitted");
  }
  bool invalid_phase1=false;try{detail::phase1_feasibility_target(NAN,1e-6);}catch(const std::invalid_argument&){invalid_phase1=true;}
  check(invalid_phase1,"nonfinite phase-I target admitted");
  bool phase1_spent=true;
  for(int k=0;k<1000;++k)if(detail::phase1_verified_reduction(1.,1.,1e-6))phase1_spent=false;
  check(phase1_spent,"unchanged restoration anchor reopens the attempt budget");
  check(!detail::phase1_verified_reduction(1.,0.5,1e-6),"partial reduction below decade contract reopened recovery");
  check(!detail::phase1_verified_reduction(1.,NAN,1e-6),"nonfinite restoration progress admitted");
  check(!detail::phase1_verified_reduction(std::min(100.,0.04),0.01,1e-6),
      "DCA infeasibility increase resets previous phase-I contraction history");
  check(detail::phase1_verified_reduction(std::min(100.,0.04),0.003,1e-6),
      "new phase-I anchor beating previous stage by a decade rejected");
  double phase1_before=1.;int stages=0;
  while(phase1_before>100e-6) {
    const double after=detail::phase1_feasibility_target(phase1_before,1e-6);
    check(detail::phase1_verified_reduction(phase1_before,after,1e-6),"valid decade phase-I progress lost");
    phase1_before=after;++stages;check(stages<=5,"phase-I stage count not contraction bounded");
  }
  for(double scale:{1e-200,1.,1e200})
    check(detail::failed_linearization_relative_descent({scale,0},{-1e-8,0}),
        "gradient units changed the private descent decision");
  check(detail::failed_linearization_relative_descent({1e308,1e308},{-1e308,-1e308}),
      "finite extreme descent overflowed");
  check(!detail::failed_linearization_relative_descent({1,0},{1,0}),"ascent accepted");
  check(!detail::failed_linearization_relative_descent({1,0},{0,1}),"orthogonal direction accepted");
  check(!detail::failed_linearization_relative_descent({0,0},{1,0}),"zero gradient accepted");
  check(!detail::failed_linearization_relative_descent({1},{0}),"zero direction accepted");
  check(!detail::failed_linearization_relative_descent({NAN},{-1}),"nonfinite gradient accepted");
  check(!detail::failed_linearization_relative_descent({1},{-INFINITY}),"nonfinite direction accepted");
  check(detail::phase1_deadline_open(59.9,1200.,100.,100.1),"open phase-I clocks rejected");
  check(!detail::phase1_deadline_open(60.,1200.,100.1,100.1),"local deadline equality accepted a late point");
  check(!detail::phase1_deadline_open(1200.,1200.,100.,200.),"global deadline equality accepted a late point");
  check(!detail::phase1_deadline_open(1200.1,1200.,100.,200.),"expired global clock accepted");
  check(!detail::phase1_deadline_open(NAN,1200.,100.,200.),"nonfinite phase-I clock accepted");
  int tiny=0,full=0;
  // No alpha appears in the contraction criterion: both histories trigger
  // the SAME subsequently guarded probe, without weakening the certificate.
  for(int k=0;k<3000;++k){
    tiny=detail::weak_dual_contraction_streak(tiny,100,99.99,1e-6);
    full=detail::weak_dual_contraction_streak(full,100,99.99,1e-6);
  }
  check(tiny==12&&tiny==full,"weak contraction streak lost or overflowed");
  check(detail::weak_dual_contraction_streak(12,100,80,1e-6)==0,"real contraction not reset");
  check(detail::weak_dual_contraction_streak(12,1e-6,1e-6,1e-6)==0,"certified dual triggers probe");
  check(detail::weak_dual_contraction_streak(12,100,INFINITY,1e-6)==0,"nonfinite history triggers probe");
  check(detail::small_step_proximal_trigger(true,1e-9,3,0.0,1e-6),"accepted vanishing step not detected");
  check(!detail::small_step_proximal_trigger(true,1e-9,2,0.0,1e-6),"premature proximal trigger");
  check(!detail::small_step_proximal_trigger(true,1,12,0.0,1e-6),"full step changes policy");
  check(!detail::small_step_proximal_trigger(false,1e-9,12,0.0,1e-6),"ineligible route changed");
  check(!detail::small_step_proximal_trigger(true,0,12,0.0,1e-6),"zero/rejected step treated as commit");
  check(!detail::small_step_proximal_trigger(true,NAN,12,0.0,1e-6),"invalid alpha triggers proposal");
  check(!detail::small_step_proximal_trigger(true,0.00564634,3,0.0286624,1e-6),
        "short infeasible progress incorrectly classified as a feasible dual stall");
  check(!detail::small_step_proximal_trigger(true,1e-9,12,1.0001e-6,1e-6),
        "infeasible point triggers dual-stall metric");
  check(detail::small_step_proximal_trigger(true,1e-9,3,1e-6,1e-6),
        "feasible boundary excluded from dual-stall metric");
  check(!detail::small_step_proximal_trigger(true,1e-9,3,NAN,1e-6),"NaN primal admitted");
  check(!detail::small_step_proximal_trigger(true,1e-9,3,INFINITY,1e-6),"infinite primal admitted");
  check(!detail::small_step_proximal_trigger(true,1e-9,3,-1,1e-6),"negative residual admitted");
  check(!detail::small_step_proximal_trigger(true,1e-9,3,0,0),"zero tolerance admitted");
  check(!detail::small_step_proximal_trigger(true,1e-9,3,0,INFINITY),"invalid tolerance admitted");
  {
    // Memory can be below an intrinsic positive-margin floor or an actual
    // numeric diagonal. Increasing memory alone must not count as a retry.
    const double floor=.001*.0080268698505021174;
    const double base=detail::proximal_retry_base(0,floor);
    const double changed=detail::small_step_proximal_proposal(base,0,1,1e-12);
    check(base==floor&&changed>floor&&std::max(changed,floor)>floor,
        "intrinsic-floor retry leaves effective tau unchanged");
    for(double memory:{0.,1e-12,.1,7.})for(double actual:{0.,1e-5,1.,100.}) {
      const double seed=detail::proximal_retry_base(memory,actual);
      const double next=detail::small_step_proximal_proposal(seed,0,1,1e-12);
      check(seed>=memory&&seed>=actual&&next>actual&&next>memory,
          "effective model metric does not change on retry");
    }
    const double capped=detail::proximal_retry_base(0,1e12);
    check(!(detail::small_step_proximal_proposal(capped,0,1,1e-12)>capped),
        "capped metric falsely counts an unchanged retry");
    for(double bad:{-1.,double(INFINITY),double(NAN)}) {
      check(std::isnan(detail::proximal_retry_base(0,bad)),"invalid actual tau admitted");
      check(std::isnan(detail::proximal_retry_base(bad,0)),"invalid remembered tau admitted");
    }
  }
  check(detail::small_step_proximal_proposal(0,100,10,1e-6)==10,"initial explicit metric incorrect");
  check(detail::small_step_proximal_proposal(5,100,10,1e-6)==50,"metric growth incorrect");
  check(detail::small_step_proximal_proposal(1e12,100,10,1e-6)==1e12,"metric cap exceeded");
  check(detail::small_step_proximal_proposal(5,INFINITY,10,1e-6)==5,"nonfinite proposal retained");
  {
    using detail::hybrid_full_objective_proximal_decay;
    // min .5*x^2 subject to 1-x<=0. The exact proximal model at a=2,
    // tau=100 has p=200/101>1,z_new=0. Its KKT residual is exactly zero
    // and its objective decreases, but the ORIGINAL KKT norm increases
    // from 1 (z_old=1) to p. KKT monotonicity is not a damping certificate.
    const double a=2,tau=100,p=tau*a/(1+tau),old_z=1,new_z=0;
    const double old_kkt=std::max(std::abs(a-old_z),std::abs(old_z*(1-a)));
    const double new_kkt=std::max(std::abs(p-new_z),std::abs(new_z*(1-p)));
    check(std::abs(p+tau*(p-a))<1e-12&&p>1&&.5*p*p<.5*a*a&&new_kkt>old_kkt,
        "exact convex model does not expose nonmonotone original KKT");
    check(hybrid_full_objective_proximal_decay(true,true,true,true,true,1,tau,.001),
        "certified full objective step cannot release excess proximal damping");
    for(int omitted=0;omitted<5;++omitted) {
      bool gates[5]={true,true,true,true,true};gates[omitted]=false;
      check(!hybrid_full_objective_proximal_decay(gates[0],gates[1],gates[2],gates[3],gates[4],1,tau,.001),
          "uncertified, historical or feasibility-only step changes damping policy");
    }
    for(double alpha:{.5,.9999999999999999,double(NAN),double(INFINITY)})
      check(!hybrid_full_objective_proximal_decay(true,true,true,true,true,alpha,tau,.001),
          "partial or invalid step releases damping");
    for(double invalid:{-1.,double(NAN),double(INFINITY)}) {
      check(!hybrid_full_objective_proximal_decay(true,true,true,true,true,1,invalid,0),
          "invalid damping admitted");
      check(!hybrid_full_objective_proximal_decay(true,true,true,true,true,1,tau,invalid),
          "invalid certified floor admitted");
    }
    check(!hybrid_full_objective_proximal_decay(true,true,true,true,true,1,.001,.001),
        "a required certified margin floor is treated as excess damping");
  }
  std::cout<<"PASS recovery memory estimates, mixed/square/2070 dimensions, overflow, weak-contraction history\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
