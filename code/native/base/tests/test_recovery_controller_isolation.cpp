#include "../src/failure_recovery_policy.hpp"
#include <iostream>
#include <stdexcept>
#include <limits>
using namespace lazyipdca::qcqp::detail;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){try {
  DualRecoveryHistory history;
  int proposals=0;
  for(int k=1;k<=12;++k) {
    history.record(100,99.9,1e-6);
    if(small_step_proximal_trigger(true,1e-5,history.proximal_steps,0,1e-6)) {
      const int retained=history.curvature_steps;
      history.proximal_proposed();++proposals;
      require(history.curvature_steps==retained,"proximal retry erased original-dual history");
    }
  }
  require(proposals==4,"proximal cadence changed");
  require(history.curvature_steps==12,"curvature probe starved by repeated proximal retries");
  history.curvature_probed();require(history.curvature_steps==0&&history.proximal_steps==0,"probe reset");
  history.record(100,100,1e-6);history.record(100,80,1e-6);
  require(history.curvature_steps==0&&history.proximal_steps==0,"real contraction ignored");
  for(int k=0;k<3000;++k)history.record(100,100,1e-6);
  require(history.curvature_steps==12&&history.proximal_steps==12,"bounded counters");
  require(!small_step_proximal_trigger(true,1e-5,history.proximal_steps,0.0286624,1e-6),"CORE2 infeasible guard lost");
  require(!small_step_proximal_trigger(false,1e-5,history.proximal_steps,0,1e-6),"eligibility ignored");
  history.record(100,std::numeric_limits<double>::infinity(),1e-6);
  require(history.curvature_steps==0&&history.proximal_steps==0,"invalid residual history");
  std::cout<<"PASS separate histories, bounded cadence, curvature eligibility, CORE2 feasibility guard\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
