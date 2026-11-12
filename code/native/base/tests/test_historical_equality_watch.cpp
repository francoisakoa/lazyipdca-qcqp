#include "historical_equality_watch.hpp"
#include <stdexcept>
#include <iostream>
#include <limits>
#include <vector>
using namespace lazyipdca::qcqp::detail;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
bool stalled(HistoricalEqualityWatch& p,int generation,double proximal){
  return p.observe(true,generation,1e-4,proximal,1,1,1,1,1e-6);
}
void eight_distinct_states(double proximal){
  HistoricalEqualityWatch p;
  for(int i=1;i<=7;++i)check(!stalled(p,i,proximal),"trigger before eight states");
  check(p.streak==7,"seven stalled states must be counted, including zero proximal");
  check(!stalled(p,7,proximal)&&p.streak==7,"duplicate generation advances watch");
  check(stalled(p,8,proximal),"eight stalled states must trigger, including zero proximal");
  check(!stalled(p,8,proximal)&&p.streak==8,"duplicate trigger is not isolated");
  check(!p.observe(true,9,1e-4,proximal,1,.5,1,.5,1e-6)&&p.streak==0,
      "healthy progress must reset watch");
}
void healthy_trajectories(){
  HistoricalEqualityWatch both,dual_only,kkt_only,full,disabled,converged;
  double kkt=2,dual=1;
  for(int i=1;i<=32;++i){
    const double next_kkt=.9*kkt,next_dual=.9*dual;
    check(!both.observe(true,i,1e-4,0,kkt,next_kkt,dual,next_dual,1e-6),
        "healthy primal-dual progress interrupted");
    check(!dual_only.observe(true,i,1e-4,0,2,2,dual,next_dual,1e-6),
        "healthy dual progress interrupted");
    check(!kkt_only.observe(true,i,1e-4,0,kkt,next_kkt,.001,.001,1e-6),
        "healthy KKT progress interrupted");
    check(!full.observe(true,i,1,0,1,1,1,1,1e-6),"full step misclassified");
    check(!disabled.observe(false,i,1e-4,0,1,1,1,1,1e-6),"disabled watch active");
    check(!converged.observe(true,i,1e-4,0,1e-6,1e-6,1e-6,1e-6,1e-6),
        "converged residuals misclassified");
    kkt=next_kkt;dual=next_dual;
  }
}
void invalid_state_resets(){
  const double nan=std::numeric_limits<double>::quiet_NaN();
  const double inf=std::numeric_limits<double>::infinity();
  const double samples[][8]={
    {1e-4,-1,1,1,1,1,1e-6,1},
    {1e-4,nan,1,1,1,1,1e-6,1},
    {1e-4,inf,1,1,1,1,1e-6,1},
    {1e-4,-inf,1,1,1,1,1e-6,1},
    {0,0,1,1,1,1,1e-6,1},
    {-1e-4,0,1,1,1,1,1e-6,1},
    {nan,0,1,1,1,1,1e-6,1},
    {1e-4,0,1,inf,1,1,1e-6,1},
    {1e-4,0,1,1,1,inf,1e-6,1},
    {1e-4,0,-1,1,1,1,1e-6,1},
    {1e-4,0,1,1,-1,1,1e-6,1},
    {1e-4,0,1,1,1,1,0,1},
    {1e-4,0,1,1,1,1,nan,1},
    {1e-4,0,1,1,1,1,1e-6,0}
  };
  for(const auto& s:samples){
    HistoricalEqualityWatch p;
    for(int i=1;i<=7;++i)check(!stalled(p,i,0),"premature prefix trigger");
    check(!p.observe(s[7]!=0,8,s[0],s[1],s[2],s[3],s[4],s[5],s[6])&&p.streak==0,
        "invalid or disabled diagnostic must reset streak");
    for(int i=9;i<=15;++i)check(!stalled(p,i,0),"old streak survived invalid diagnostic");
    check(stalled(p,16,0),"valid eight-state observation cannot recover after reset");
  }
}
void physical_step_observation(){
  using Watch=HistoricalEqualityWatch;
  const double threshold=Watch::physical_step_threshold();
  const std::vector<double> old={1,1e12,0};
  const std::vector<double> next={1+1e-9,1e12+1,1e-10};
  const auto saved_old=old,saved_next=next;
  const double metric=Watch::relative_primal_step(old,next);
  check(metric>0&&metric<threshold,"actual tiny coordinatewise step was not recognized");
  check(old==saved_old&&next==saved_next,"physical observation modified an accepted vector");
  check(Watch::relative_primal_step(old,old)==0,"zero rounded displacement is not zero");
  check(Watch::relative_primal_step(std::vector<double>{1e30,0},
                                  std::vector<double>{1e30,1e-4})>threshold,
        "large coordinate concealed real motion of a small coordinate");
  const double huge=std::numeric_limits<double>::max();
  const double opposite=Watch::relative_primal_step(std::vector<double>{huge},std::vector<double>{-huge});
  check(std::isfinite(opposite)&&opposite>1.9,"finite huge displacement overflowed");
  Watch full;
  for(int i=1;i<=7;++i)
    check(!full.observe(true,i,1,0,1,1,1,1,1e-6,metric),"physical branch fired before eight states");
  check(!full.observe(true,7,1,0,1,1,1,1,1e-6,metric)&&full.streak==7,
        "physical branch double-counted a generation");
  check(full.observe(true,8,1,0,1,1,1,1,1e-6,metric),
        "full-alpha tiny physical stagnation was missed");
  check(!full.observe(true,8,1,0,1,1,1,1,1e-6,metric)&&full.streak==8,
        "physical branch repeated its trigger for one state");
  Watch exact,above;
  for(int i=1;i<=8;++i) {
    check(exact.observe(true,i,1,0,1,1,1,1,1e-6,threshold)==(i==8),
          "physical threshold boundary is not inclusive");
    check(!above.observe(true,i,1,0,1,1,1,1,1e-6,std::nextafter(threshold,INFINITY)),
          "step above the physical threshold was called tiny");
  }
  Watch healthy_kkt,healthy_dual,disabled,converged;
  for(int i=1;i<=16;++i) {
    check(!healthy_kkt.observe(true,i,1,0,1,.98,1,1,1e-6,metric),
          "tiny physical step with real KKT contraction was interrupted");
    check(!healthy_dual.observe(true,i,1,0,1,1,1,.98,1e-6,metric),
          "tiny physical step with real dual contraction was interrupted");
    check(!disabled.observe(false,i,1,0,1,1,1,1,1e-6,metric),"disabled physical watch active");
    check(!converged.observe(true,i,1,0,1e-6,1e-6,1e-6,1e-6,1e-6,metric),
          "physical watch interrupted a converged state");
  }
  const double nan=std::numeric_limits<double>::quiet_NaN();
  for(double invalid:{nan,std::numeric_limits<double>::infinity(),-1.0}) {
    Watch guard;
    for(int i=1;i<=7;++i)check(!guard.observe(true,i,1,0,1,1,1,1,1e-6,metric),"bad prefix");
    check(!guard.observe(true,8,1,0,1,1,1,1,1e-6,invalid)&&guard.streak==0,
          "invalid physical diagnostic completed a streak");
  }
  check(std::isnan(Watch::relative_primal_step(std::vector<double>{nan},std::vector<double>{0}))&&
        std::isnan(Watch::relative_primal_step(std::vector<double>{0},std::vector<double>{INFINITY}))&&
        std::isnan(Watch::relative_primal_step(std::vector<double>{0},std::vector<double>{0,1}))&&
        std::isnan(Watch::relative_primal_step(std::vector<double>{},std::vector<double>{})),
        "invalid vector measurement must remain unknown");
}
int main(){
  try{
    eight_distinct_states(1e4);
    eight_distinct_states(0);
    eight_distinct_states(-0.0);
    healthy_trajectories();
    invalid_state_resets();
    physical_step_observation();
    std::cout<<"historical_equality_watch PASS zero/positive proximal, eight distinct accepted states, "
      "healthy KKT/dual progress, coordinatewise physical full-step stagnation, "
      "disabled/converged guards, invalid diagnostic reset\n";
    return 0;
  }catch(const std::exception& error){
    std::cerr<<"historical_equality_watch FAIL: "<<error.what()<<'\n';return 1;
  }
}
