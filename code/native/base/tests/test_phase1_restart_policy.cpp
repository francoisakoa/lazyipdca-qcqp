#include "../src/restoration_restart_policy.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){
  int checks=0;auto require=[&](bool ok,const char* message){++checks;if(!ok)throw std::runtime_error(message);};
  for(int total:{0,1,7,8,136,240,2147483647})for(int directions:{1,2,8,16}) {
    int remaining=total,visited=0;long long used=0;
    for(int left=directions;left>0&&remaining>0;--left) {
      const int share=detail::shared_restart_allowance(remaining,left,80);
      require(share>0&&share<=remaining&&share<=80,"shared allowance violated a cap");
      remaining-=share;used+=share;++visited;
    }
    require(used<=total,"restarts exceeded common total");
    if(total>=directions)require(visited==directions,"early directions starved a later restart");
  }
  require(detail::shared_restart_allowance(136,8,80)==17,"expected equal remaining share lost");
  require(detail::shared_restart_allowance(0,8,80)==0,"spent total reopened");
  require(detail::shared_restart_allowance(136,8,0)==0,"zero explicit restart cap reopened");
  for(auto v:{-1,0}){bool rejected=false;try{detail::shared_restart_allowance(2,v,80);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"invalid restart count admitted");}
  const double frozen=160.9;
  for(double current:{160.9,16.09,2.6,.26,.026})
    require(detail::phase1_global_restart_reference(current,frozen)==frozen,"decade contraction shrank global exploration reference");
  require(detail::phase1_global_restart_reference(200,frozen)==200,"current larger reference was suppressed");
  std::cout<<"phase1_restart_policy PASS checks="<<checks<<" shared_cap=1 remaining_directions_covered=1 frozen_global_radius=1\n";
}
