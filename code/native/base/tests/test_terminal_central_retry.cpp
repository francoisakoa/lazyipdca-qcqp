#include "lazyipdca_qcqp/majorant_terminal_retry.hpp"
#include "lazyipdca_qcqp/majorant_startup_retry.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace lazyipdca::qcqp::detail;
int main() {
  int checks=0;
  auto check=[&](bool v,const char* msg){++checks;if(!v)throw std::runtime_error(msg);};
  auto guard=[](bool auto_mode=true,bool central=false,bool used=false,bool mixed=true,
      bool signed_sq=false,bool lazy_obj=false,double initial_mu=.0442552,
      double primal=.0288,double dual=6055.,double comp=10400.,int remaining=2686,double seconds=570.) {
    return terminal_central_retry_needed(auto_mode,central,used,mixed,signed_sq,lazy_obj,
      initial_mu,primal,dual,comp,1e-6,1e-6,1e-6,remaining,seconds);
  };
  check(guard(),"terminal legacy reset must be eligible");
  check(!guard(false),"explicit legacy must not change");
  check(!guard(true,true),"central first must not reset");
  check(!guard(true,false,true),"shared retry already spent");
  check(!guard(true,false,false,false),"not mixed");
  check(!guard(true,false,false,true,true),"signed square protected");
  check(!guard(true,false,false,true,false,true),"lazy objective protected");
  for(double mu:{0.,-1.,1.,2.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
    check(!guard(true,false,false,true,false,false,mu),"invalid or already centered initial mu");
  check(guard(true,false,false,true,false,false,1e-4),"low initial mu terminal eligibility");
  check(!guard(true,false,false,true,false,false,.044,0.),"feasible primal protected");
  check(!guard(true,false,false,true,false,false,.044,.0288,6055,10400,0),"iterations exhausted");
  check(!guard(true,false,false,true,false,false,.044,.0288,6055,10400,1,0),"time exhausted");
  check(!guard(true,false,false,true,false,false,.044,.0288,-1),"negative dual");
  AutoStartupRetry budget;
  check(guard(true,false,budget.used()),"fresh shared quota");
  budget.mark_used();check(!guard(true,false,budget.used()),"one-shot shared quota enforced");
  std::cout<<"TERMINAL_CENTRAL_RETRY_POLICY=PASS checks="<<checks<<"\n";
}
