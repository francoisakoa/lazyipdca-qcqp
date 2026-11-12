#include "shared_restoration_budget.hpp"
#include <climits>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp::detail;
namespace {
int checks=0;
void require(bool ok,const char* message) {
  ++checks;if(!ok)throw std::runtime_error(message);
}

void nested_work_conserves_the_original_allowance() {
  for(int budget:{0,1,2,3,199,200,201,599,600,601,1000}) {
    for(int outer:{0,1,2,8,200,5000}) {
      const int affine=affine_start_iteration_allowance(true,budget,outer);
      require(affine>=0&&affine<=budget,"affine phase created restoration work");
      require(affine<=200,"affine historical cap changed");
      require(affine==0||affine<outer,"affine phase consumed original-model outer reserve");
      for(int affine_used:{0,affine/2,affine}) {
        const int child=remaining_shared_restoration(budget,affine_used);
        require(child+affine_used==budget,"continuation inherited a replenished budget");
        const int startup=std::min(80,child);
        const int late=remaining_shared_restoration(child,startup);
        const int consumed=affine_used+startup+late;
        require(consumed==budget,"nested phases lost or duplicated attempted work");
        require(!restoration_step_available(true,budget,consumed),
            "exhausted nested phases admitted an extra terminal certificate");
      }
    }
  }
}

void observed_overrun_regression() {
  const int affine=affine_start_iteration_allowance(true,600,5000);
  const int inner=remaining_shared_restoration(600,affine);
  require(affine==200&&inner==400,"600-step parent was not debited before recursion");
  const int late=remaining_shared_restoration(inner,80);
  require(late==320,"startup plus late recovery can still spend 600 after affine work");
  const int actual=affine+80+late;
  require(actual==600&&!restoration_step_available(true,600,actual),"801-step route remains possible");
  require(!restoration_step_available(true,600,600),"600+terminal=601 remains possible");
  const int reported_overrun=801;
  require(remaining_shared_restoration(600,reported_overrun)==0,"overrun granted new work");
  require(reported_overrun==801,"actual work was hidden by clamping a counter");
}

void partial_work_and_historical_callers() {
  const int reserved=affine_start_iteration_allowance(true,600,5000);
  require(reserved==200,"historical affine envelope unexpectedly changed");
  const int actual=40;
  require(remaining_shared_restoration(600,actual)==560,"unused reservation was charged as executed work");
  require(restoration_step_available(true,600,599),"last available certificate step was lost");
  for(int limit:{-1,0,1,600}) {
    require(affine_start_iteration_allowance(false,limit,5000)==200,"unshared affine policy changed");
    require(restoration_step_available(false,limit,801),"unshared terminal policy changed");
  }
  require(affine_start_iteration_allowance(true,INT_MAX,INT_MIN)==0,"negative outer bound overflowed");
  require(affine_start_iteration_allowance(true,INT_MAX,INT_MAX)==200,"large outer bound overflowed");
  require(remaining_shared_restoration(INT_MAX,INT_MAX-1)==1,"last large-budget step lost");
  require(remaining_shared_restoration(600,-1)==0,"invalid negative counter granted work");
}
}

int main(){try {
  nested_work_conserves_the_original_allowance();observed_overrun_regression();partial_work_and_historical_callers();
  std::cout<<"PASS shared restoration accounting checks="<<checks
      <<" nested_budget=1 actual_work_preserved=1 legacy_policy_preserved=1 optimizer_calls=0\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
