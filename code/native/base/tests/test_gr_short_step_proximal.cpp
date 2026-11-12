#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "gr_short_step_proximal.hpp"
#include "failure_recovery_policy.hpp"
namespace policy=lazyipdca::qcqp::detail;
static int checks=0;
static void require(bool p,const char* message){++checks;if(!p)throw std::runtime_error(message);}
int main() {
  try {
    using Gate=DcaGrShortStepProximal;
    require(Gate::trigger(true,.001,3,.005,.00495),"infeasible physical plateau not admitted");
    require(Gate::trigger(true,.001,3,10,9),"exact ten percent boundary not admitted");
    require(Gate::trigger(true,.001,3,10,10.1),"worsening physical residual not admitted");
    require(!Gate::trigger(true,.001,3,10,8.99),"improving physical residual admitted");
    require(!Gate::trigger(false,.001,3,10,10),"disabled additional gate active");
    require(!Gate::trigger(true,.01,3,10,10),"large accepted alpha admitted");
    require(!Gate::trigger(true,.5,3,10,10),"half step admitted");
    require(!Gate::trigger(true,0,3,10,10),"zero alpha admitted");
    require(!Gate::trigger(true,-.1,3,10,10),"negative alpha admitted");
    require(!Gate::trigger(true,.001,2,10,10),"insufficient weak dual history admitted");
    require(!Gate::trigger(true,.001,-1,10,10),"negative weak dual history admitted");
    for(double invalid:{NAN,INFINITY,-INFINITY}) {
      require(!Gate::trigger(true,invalid,3,10,10),"nonfinite alpha admitted");
      require(!Gate::trigger(true,.001,3,invalid,10),"nonfinite previous physical residual admitted");
      require(!Gate::trigger(true,.001,3,10,invalid),"nonfinite current physical residual admitted");
    }
    require(!Gate::trigger(true,.001,3,-1,10),"negative previous physical residual admitted");
    require(!Gate::trigger(true,.001,3,10,-1),"negative current physical residual admitted");
    require(Gate::trigger(true,.001,3,0,0),"zero residual boundary invalid");
    require(Gate::trigger(true,.001,3,std::numeric_limits<double>::max(),std::numeric_limits<double>::max()),"finite large residual overflowed gate");

    // Exercise the ACTUAL existing dual history. A >10% contraction resets
    // the evidence; a near tolerance residual also resets it.
    policy::DualRecoveryHistory history;
    for(int i=0;i<3;++i)history.record(1,.99,1e-6);
    require(history.proximal_steps==3&&Gate::trigger(true,.001,history.proximal_steps,.005,.00495),"actual weak dual history did not admit plateau");
    history.record(1,.8,1e-6);
    require(history.proximal_steps==0&&!Gate::trigger(true,.001,history.proximal_steps,.005,.00495),"improving dual failed to reset additional policy");
    for(int i=0;i<3;++i)history.record(1,.99,1e-6);
    history.record(1,1e-6,1e-6);
    require(history.proximal_steps==0,"near-tolerance dual failed to reset history");
    for(int i=0;i<3;++i)history.record(1,.99,1e-6);
    history.proximal_proposed();require(history.proximal_steps==0,"existing proposal did not reset its own history");

    // The old policy stays exactly available. Infeasible plateaus constitute
    // the focused extension; objective/constraint functions are untouched.
    const bool legacy_infeasible=policy::small_step_proximal_trigger(true,.001,3,.00495,1e-6);
    require(!legacy_infeasible&&Gate::trigger(true,.001,3,.005,.00495),"extension did not isolate former infeasible-gate gap");
    require(policy::small_step_proximal_trigger(true,.001,3,1e-7,1e-6),"historical feasible policy changed");
    for(double alpha:{.0001,.001,.009,.01,.5})for(int weak:{0,2,3,12})for(double primal:{0.,1e-7,.005}) {
      const bool legacy=policy::small_step_proximal_trigger(true,alpha,weak,primal,1e-6);
      const bool disabled_combined=legacy||Gate::trigger(false,alpha,weak,primal,primal);
      require(disabled_combined==legacy,"flag0 legacy parity changed");
    }
    const double proposal=policy::small_step_proximal_proposal(6e-5,.04,1e3,1e-8);
    require(std::abs(proposal-6e-4)<1e-18,"existing next-model proposal changed");
    std::cout<<"gr_short_step_proximal_actual_helper PASS checks="<<checks<<"\n";return 0;
  }catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<" checks="<<checks<<"\n";return 1;}
}
