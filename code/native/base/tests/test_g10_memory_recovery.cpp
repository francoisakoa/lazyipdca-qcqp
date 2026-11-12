#include "../src/mumps_memory_recovery.hpp"
#include "../src/mumps_ordering_guard.hpp"
#include <stdexcept>
#include <iostream>
#include <vector>
using lazyipdca::qcqp::detail::recover_mumps_memory;
using lazyipdca::qcqp::detail::learned_mumps_memory_margin;
using lazyipdca::qcqp::detail::remember_mumps_memory_margin;
using lazyipdca::qcqp::detail::reset_mumps_memory_margin_for_test;
void require(bool b){if(!b)throw std::runtime_error("memory recovery contract failed");}
int main(){try{
  using lazyipdca::qcqp::detail::mumps_requested_ordering_verified;
  std::string message;
  require(mumps_requested_ordering_verified(true,5,1,message));
  for(int fallback:{0,1,2,3,4,6,7})
    require(!mumps_requested_ordering_verified(true,fallback,1,message));
  for(int analysis:{0,2,7})
    require(!mumps_requested_ordering_verified(true,5,analysis,message));
  require(mumps_requested_ordering_verified(false,0,2,message));
  reset_mumps_memory_margin_for_test();
  int code=-9,margin=20,calls=0;
  std::vector<int> margins;
  const int attempts=recover_mumps_memory(code,margin,[&]{
    ++calls;margins.push_back(margin);code=margin<2560?-9:0;
  });
  require(code==0 && attempts==7 && calls==7 && margin==2560);
  require(learned_mumps_memory_margin()==2560);
  require(margins==std::vector<int>({40,80,160,320,640,1280,2560}));
  code=-8;margin=20;calls=0;
  require(recover_mumps_memory(code,margin,[&]{++calls;code=-9;})==9);
  require(margin==6400 && calls==9 && code==-9);
  require(learned_mumps_memory_margin()==2560);
  for(int status:{0,-10,-13}){
    code=status;margin=20;
    require(recover_mumps_memory(code,margin,[&]{throw std::runtime_error("unexpected retry");})==0);
    require(code==status && margin==20);
  }
  code=-9;margin=6400;
  require(recover_mumps_memory(code,margin,[&]{throw std::runtime_error("cap exceeded");})==0);
  using lazyipdca::qcqp::detail::mumps_reanalyze_after_memory_failure;
  for(int failure:{-8,-9}) {
    require(mumps_reanalyze_after_memory_failure(true,true,failure));
    require(!mumps_reanalyze_after_memory_failure(false,true,failure));
    require(!mumps_reanalyze_after_memory_failure(true,false,failure));
  }
  for(int other:{-10,-13,0,1})require(!mumps_reanalyze_after_memory_failure(true,true,other));
  reset_mumps_memory_margin_for_test();remember_mumps_memory_margin(160);
  remember_mumps_memory_margin(80);require(learned_mumps_memory_margin()==160);
  std::cout<<"MUMPS -8/-9: same numerical transaction, finite cap, process-wide learned margin, no retry on other errors; requested METIS ordering and sequential analysis verified\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
