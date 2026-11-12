#include "gr_fixed_centering.hpp"
#include <iostream>
#include <stdexcept>

using lazyipdca::qcqp::detail::GrFixedCenteringPolicy;
using lazyipdca::qcqp::detail::GrFixedCenteringScope;
using lazyipdca::qcqp::detail::gr_fixed_centering_converged;
namespace {
int checks=0;
void require(bool v,const char* message){++checks;if(!v)throw std::runtime_error(message);}
bool observe(GrFixedCenteringPolicy& p,int generation,bool enabled=true,double primal=1e-7,
    double dual=1e-10,double comp=2e-6,double centering=.03,double mu=1e-7) {
  return p.observe(enabled,generation,primal,dual,comp,centering,mu,1e-6,1e-6,1e-6);
}
}
int main(){try {
  GrFixedCenteringPolicy off(0);
  for(int i=0;i<20;++i)require(!observe(off,i),"disabled phase activated");
  GrFixedCenteringPolicy p(2);
  require(!observe(p,1),"unstable first anchor triggered");
  require(!observe(p,1),"retry manufactured stability");
  require(!observe(p,2),"second anchor triggered");
  require(observe(p,3),"stable complementarity stall did not trigger");
  require(p.active()&&p.remaining==2&&p.attempts==1,"phase budget incorrect");
  require(!observe(p,4),"nested centering phase");
  require(!p.accepted(4)&&p.remaining==1,"first centering step ended early");
  require(p.accepted(5)&&!p.active(),"centering budget unbounded");
  for(int i=6;i<13;++i)require(!observe(p,i),"cooldown bypassed");
  require(!observe(p,13)&&!observe(p,14)&&observe(p,15),"second stable phase unavailable");
  p.abort(15);
  require(!p.active()&&p.remaining==0,"abort failed");
  for(int i=16;i<40;++i)require(!observe(p,i),"global two-phase limit bypassed");
  for(auto condition:{0,1,2,3,4}) {
    GrFixedCenteringPolicy gate(1);
    for(int i=0;i<5;++i)require(!observe(gate,i,condition!=0,
        condition==1?1e-3:1e-7,condition==2?1e-3:1e-10,
        condition==3?1e-7:2e-6,condition==4?1e-8:.03),"ineligible residual gate triggered");
  }
  GrFixedCenteringPolicy reset(1);
  require(!observe(reset,0)&&!observe(reset,1),"premature trigger");
  require(!observe(reset,2,false),"disabled gate triggered");
  require(!observe(reset,3)&&!observe(reset,4)&&observe(reset,5),"unstable window was not reset");
  for(bool test_primal:{true,false}) {
    GrFixedCenteringPolicy strict(2);
    for(int i=0;i<12;++i)require(!observe(strict,i,true,test_primal?5e-6:1e-7,
        test_primal?1e-10:5e-6),"phase consumed before strict primal/dual feasibility");
    require(strict.attempts==0,"infeasible near-KKT neighborhood consumed phase budget");
  }
  GrFixedCenteringPolicy boundary(1);
  require(!observe(boundary,0,true,1e-6,1e-6)&&!observe(boundary,1,true,1e-6,1e-6)&&
      observe(boundary,2,true,1e-6,1e-6),"strict original tolerance boundary rejected");
  GrFixedCenteringPolicy early(8);
  require(!observe(early,0)&&!observe(early,1)&&observe(early,2),"early-finish setup failed");
  require(!early.accepted(3,false)&&early.remaining==7,"unconverged accepted work was not charged");
  require(early.accepted(4,true)&&early.remaining==0&&!early.active(),"converged phase forced cap steps");
  require(early.next_generation==12&&early.attempts==1,"early finish lost cooldown/attempt charge");
  require(gr_fixed_centering_converged(1e-6,1e-6,1e-7,1e-6,1e-6,1e-6,1e-6),"both barrier boundary gates rejected");
  require(!gr_fixed_centering_converged(2e-6,1e-7,1e-7,1e-7,1e-7,1e-6,1e-6),"missing original barrier gate");
  require(!gr_fixed_centering_converged(1e-7,2e-6,1e-7,1e-7,1e-7,1e-6,1e-6),"missing regularized barrier gate");
  require(!gr_fixed_centering_converged(1e-7,1e-7,1e-7,2e-6,1e-7,1e-6,1e-6),"missing physical primal gate");
  require(!gr_fixed_centering_converged(1e-7,1e-7,1e-7,1e-7,2e-6,1e-6,1e-6),"missing physical dual gate");
  require(!gr_fixed_centering_converged(INFINITY,1e-7,1e-7,1e-7,1e-7,1e-6,1e-6),"nonfinite original barrier gate");
  require(!gr_fixed_centering_converged(1e-7,INFINITY,1e-7,1e-7,1e-7,1e-6,1e-6),"nonfinite regularized barrier gate");
  require(!gr_fixed_centering_converged(1e-7,1e-7,0,1e-7,1e-7,1e-6,1e-6),"zero target barrier admitted");
  int rollbacks=0;
  {GrFixedCenteringScope scope;scope.rollback=[&](){++rollbacks;};}
  require(rollbacks==1,"scope exit lost pending transaction");
  try{GrFixedCenteringScope scope;scope.rollback=[&](){++rollbacks;};throw std::runtime_error("probe");}
  catch(const std::runtime_error&){}
  require(rollbacks==2,"exception lost pending transaction");
  {GrFixedCenteringScope scope;scope.rollback=[&](){++rollbacks;};scope.rollback={};}
  require(rollbacks==2,"committed transaction rolled back");
  bool rejected=false;try{GrFixedCenteringPolicy bad(9);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"unbounded requested phase accepted");
  std::cout<<"GR_FIXED_CENTERING_PASS checks="<<checks<<" optimizer_calls=0\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
