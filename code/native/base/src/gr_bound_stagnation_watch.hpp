#pragma once
#include <cmath>
namespace lazyipdca::qcqp::detail {
// Admit the existing consistent-bound path when an accepted vanishing step
// repeatedly prevents progress. Counts refer to distinct accepted states.
struct GrBoundStagnationWatch {
  int last_generation=-1,count=0;
  double reference_kkt=0,reference_centering=0,barrier=-1;
  bool active=false;
  bool observe(bool enabled,int generation,double alpha,double kkt,
               double centering,double mu,double tolerance){
    if(active||generation==last_generation)return false;
    last_generation=generation;
    const bool eligible=enabled&&std::isfinite(alpha)&&alpha>0&&alpha<=1e-3&&
        std::isfinite(kkt)&&std::isfinite(centering)&&std::isfinite(mu)&&mu>0&&
        kkt>tolerance&&centering>10*mu;
    if(!eligible){count=0;return false;}
    if(count==0||mu!=barrier||kkt<.99*reference_kkt||centering<.99*reference_centering){
      reference_kkt=kkt;reference_centering=centering;barrier=mu;count=1;return false;
    }
    if(++count<8)return false;
    active=true;return true;
  }
};
}
