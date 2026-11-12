#pragma once
#include "lazyipdca_qcqp/model.hpp"
namespace lazyipdca::qcqp {
struct HybridSocSize {
 bool passed=false;
 double scaled_step=0,scaled_correction=0,limit=0;
};
// A globalization SOC need not solve the original affine subproblem again.
// Its SOURCE direction must be certified separately. This size certificate
// bounds the primal correction by O(||alpha*d||^2) in fixed anchor units.
inline HybridSocSize hybrid_source_soc_size(const Vec& anchor,const Vec& direction,
    double alpha,const Vec& correction) {
 HybridSocSize r;if(anchor.empty()||anchor.size()!=direction.size()||anchor.size()!=correction.size()||
     !std::isfinite(alpha)||!(alpha>0&&alpha<=1))return r;
 long double step=0,delta=0;
 for(std::size_t j=0;j<anchor.size();++j){
  if(!std::isfinite(anchor[j])||!std::isfinite(direction[j])||!std::isfinite(correction[j]))return r;
  const long double scale=1+std::abs((long double)anchor[j]);
  step=std::max(step,std::abs((long double)alpha*direction[j])/scale);
  delta=std::max(delta,std::abs((long double)correction[j])/scale);
 }
 const long double cap=std::min(step,step*step);
 r.scaled_step=double(step);r.scaled_correction=double(delta);r.limit=double(cap);
 r.passed=std::isfinite(r.scaled_step)&&std::isfinite(r.scaled_correction)&&std::isfinite(r.limit)&&delta<=cap;
 return r;
}
}
