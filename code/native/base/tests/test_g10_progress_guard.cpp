#include "../src/dca_progress_guard.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using lazyipdca::qcqp::detail::DcaAcceptedProgress;
using lazyipdca::qcqp::detail::dca_kkt_growth_allowed;
using lazyipdca::qcqp::detail::dca_kkt_growth_ceiling;
using lazyipdca::qcqp::detail::dca_inherit_kkt_reference;
using lazyipdca::qcqp::detail::dca_inherit_phase_kkt_reference;
using lazyipdca::qcqp::detail::dca_private_continuation_allowed;
using lazyipdca::qcqp::detail::dca_local_candidate_not_dominated;
using lazyipdca::qcqp::detail::dca_aggregate_initial_rho;
using lazyipdca::qcqp::detail::dca_aggregate_next_rho;
using lazyipdca::qcqp::detail::dca_rho_probe_candidate_allowed;
using lazyipdca::qcqp::detail::dca_rho_probe_schedule;
using lazyipdca::qcqp::detail::dca_feasible_kkt_recovery_allowed;
using lazyipdca::qcqp::detail::dca_rho_candidate_better;
using lazyipdca::qcqp::Settings;

void require(bool value,const char* message){
  if(!value)throw std::runtime_error(message);
}

int main(){try{
  const Settings defaults;
  require(defaults.dca_kkt_growth_max==10.0,
          "validated aggregate KKT-growth default regressed");
  require(defaults.dca_aggregate_progress_window==0,
          "experimental progress response must remain opt-in");
  require(defaults.acceptable_tolerance==1e-5 &&
          defaults.acceptable_constraint_tolerance==1e-6 &&
          defaults.acceptable_dual_tolerance==1e-5 &&
          defaults.acceptable_complementarity_tolerance==1e-6,
          "validated acceptable defaults regressed");

  require(dca_kkt_growth_ceiling(500,1.5)==750,"wrong KKT ceiling");
  require(dca_kkt_growth_allowed(500,750,1.5),"boundary KKT rejected");
  require(!dca_kkt_growth_allowed(500,751,1.5),"KKT growth escaped global guard");
  require(!dca_kkt_growth_allowed(500,INFINITY,1.5),"nonfinite KKT accepted");
  require(dca_kkt_growth_ceiling(4.39372704e-3,10.0,1e-4)<4.4e-2,
          "near-solution KKT ceiling retained the obsolete unit floor");
  require(!dca_kkt_growth_allowed(4.39372704e-3,8.02631285e-1,10.0,1e-4),
          "case30000 0.0044 -> 0.803 KKT regression was accepted");
  require(dca_kkt_growth_allowed(1e-8,1e-4,10.0,1e-4),
          "tolerance-derived roundoff floor was not honored");
  double reference=dca_inherit_kkt_reference(4.4e-3,4.4e-3,3.4e-4,2.28e-2);
  require(reference==4.4e-3,"local KKT low-water mark was not retained");
  require(!dca_kkt_growth_allowed(reference,1.38e-1,10.0,1e-4),
          "case30000 cumulative KKT ratchet escaped persistent reference");
  reference=dca_inherit_phase_kkt_reference(
      3.2e-3,3.2e-3,1.3e-4,3.9e-3,2e-8,1e-6);
  require(reference==3.9e-3,
          "infeasible KKT low-water mark leaked into the feasible phase");
  reference=dca_inherit_phase_kkt_reference(
      3.9e-3,reference,2e-8,3.67e-3,2e-8,1e-6,true);
  require(reference==3.67e-3,
          "feasible KKT low-water mark was not inherited");
  reference=dca_inherit_phase_kkt_reference(
      3.67e-3,reference,2e-8,5e-3,2e-4,1e-6,true);
  reference=dca_inherit_phase_kkt_reference(
      5e-3,reference,2e-4,4.2e-3,2e-8,1e-6,true);
  require(reference==3.67e-3,
          "feasible KKT reference was reset by a later infeasible-to-feasible recrossing");
  require(!dca_private_continuation_allowed(1.55224339e-4,1.4e-4),
          "case30000 outer 145 private continuation remained enabled");
  require(dca_private_continuation_allowed(31.0,7.8e-2),
          "global private continuation was disabled outside the local zone");
  require(!dca_local_candidate_not_dominated(
              1.6e-3,1.6e-3,1.1e-5,2.5e-3,3.5e-4),
          "case30000 dominated local step was accepted");
  require(dca_local_candidate_not_dominated(
              5.4e-1,5.4e-1,3.9e-3,6.9e-1,2.6e-3),
          "local feasibility-improving tradeoff was rejected");
  require(dca_local_candidate_not_dominated(
              1.9e-1,1.9e-1,3.3e-4,1.0e-1,1.3e-2),
          "new KKT low-water mark was rejected");
  require(dca_aggregate_initial_rho(2e-2,1e-4,1e-6,true)==2e-2,
          "certified rho was not preserved before feasibility");
  require(dca_aggregate_initial_rho(2e-2,6e-10,1e-6,true)==0.0,
          "exact KKT was not retried after primal certification");
  require(std::abs(dca_aggregate_next_rho(0.0,2e-2,true)-4e-3)<1e-15,
          "rho recovery did not start at 0.2 of the certificate");
  require(std::abs(dca_aggregate_next_rho(4e-3,2e-2,true)-2e-2)<1e-15,
          "rho recovery skipped the previous certificate");
  require(std::abs(dca_aggregate_next_rho(2e-2,2e-2,true)-2e-1)<1e-15,
          "rho escalation above the certificate regressed");
  require(std::abs(dca_aggregate_next_rho(4.0,20.0,false)-40.0)<1e-15,
          "global historical rho ladder was changed");
  require(!dca_rho_probe_candidate_allowed(1.88e-3,1.90e-3,1.90e-6,1e-6),
          "case30000 R7D6 rho=0.004 regression was accepted");
  require(!dca_rho_probe_candidate_allowed(1.88e-3,1.70e-3,1.01e-6,1e-6),
          "rho probe lost certified primal feasibility");
  require(!dca_rho_probe_candidate_allowed(1.88e-3,1.90e-3,8e-7,1e-6),
          "rho probe worsened original KKT");
  require(dca_rho_probe_candidate_allowed(1.88e-3,1.50e-3,8e-7,1e-6),
          "rho probe rejected genuine original progress");
  require(dca_rho_probe_candidate_allowed(3.9e-3,3.67e-3,2e-8,1e-6),
          "case30000 gradual feasible KKT progress was rejected");
  require(dca_feasible_kkt_recovery_allowed(
              1.48e-2,2.3e-3,3.0e-12,7.4e-7,1e-6),
          "case30000 R7D8 feasible KKT recovery was rejected");
  require(!dca_feasible_kkt_recovery_allowed(
              1.48e-2,2.3e-3,3.0e-12,1.01e-6,1e-6),
          "KKT recovery was allowed to lose primal feasibility");
  require(!dca_feasible_kkt_recovery_allowed(
              1.48e-2,1.34e-2,3.0e-12,7.4e-7,1e-6),
          "KKT recovery accepted less than ten percent contraction");
  require(dca_feasible_kkt_recovery_allowed(
              1.48e-2,0.9*1.48e-2,3.0e-12,7.4e-7,1e-6),
          "KKT recovery rejected its exact contraction boundary");
  require(!dca_feasible_kkt_recovery_allowed(
              1.48e-2,2.3e-3,1.01e-6,7.4e-7,1e-6),
          "KKT recovery treated an infeasible origin as terminal");
  require(!dca_feasible_kkt_recovery_allowed(
              0.0,0.0,3.0e-12,7.4e-7,1e-6),
          "KKT recovery accepted a zero current residual");
  require(!dca_feasible_kkt_recovery_allowed(
              INFINITY,2.3e-3,3.0e-12,7.4e-7,1e-6),
          "KKT recovery accepted a nonfinite current residual");
  require(!dca_kkt_growth_allowed(1.8e-3,2.0e-2,10.0,1e-4),
          "KKT recovery could escape the persistent low-water ceiling");
  const auto schedule=dca_rho_probe_schedule(1e-2,1e3);
  require(std::find(schedule.begin(),schedule.end(),2e-2)!=schedule.end(),
          "case30000 historical rho=0.02 was omitted from probe schedule");
  require(dca_rho_candidate_better(1.50e-3,8e-7,0.004,
                                   1.70e-3,6e-10,0.02,0.02),
          "smallest original KKT did not win rho selection");
  require(!dca_rho_candidate_better(1.90e-3,8e-7,0.004,
                                    1.70e-3,6e-10,0.02,0.02),
          "smaller rho overrode a better original KKT");
  require(!dca_rho_candidate_better(1.70e-3,6e-10,0.004,
                                    1.70e-3,6e-10,0.02,0.02),
          "tie did not prefer the previous certified rho");

  DcaAcceptedProgress progress(2);
  auto d=progress.observe(1.2e5,3.1e-2,1.2e5,true,1e-6,1e-6);
  require(!d.aggregate_stall,"single aggregate point declared stalled");
  d=progress.observe(1.195e5,3.1e-2,1.195e5,true,1e-6,1e-6);
  require(d.aggregate_stall&&d.aggregate_reference_kkt==1.2e5,
          "slow aggregate streak was not detected");

  progress.clear();
  progress.observe(100,1e-2,100,true,1e-6,1e-6);
  d=progress.observe(98,1e-2,98,true,1e-6,1e-6);
  require(!d.aggregate_stall,"meaningful aggregate contraction rejected");

  progress.clear();
  progress.observe(2400,5e-9,2400,false,1e-6,1e-6);
  d=progress.observe(2400,4e-9,2400,false,1e-6,1e-6);
  require(d.near_feasible_dual_stall&&d.dual_reference==2400,
          "near-feasible dual stall was not detected");
  progress.clear();
  progress.observe(2400,1e-3,2400,false,1e-6,1e-6);
  d=progress.observe(2400,1e-3,2400,false,1e-6,1e-6);
  require(!d.near_feasible_dual_stall,"infeasible iterates triggered dual rescue");

  DcaAcceptedProgress disabled(0);
  disabled.observe(1,0,1,true,1e-6,1e-6);
  d=disabled.observe(1,0,1,true,1e-6,1e-6);
  require(!d.aggregate_stall&&!d.near_feasible_dual_stall,"disabled monitor fired");
  std::cout<<"Global KKT guard and accepted-iterate progress monitor passed\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
