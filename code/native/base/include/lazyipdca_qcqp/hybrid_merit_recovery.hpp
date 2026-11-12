#pragma once
#include "lazyipdca_qcqp/hybrid_inexact.hpp"
namespace lazyipdca::qcqp {
// Private globalization fallback, never a model or original KKT certificate.
// Require a representable Armijo decrease beyond its rounding allowance,
// physical primal contraction (or its already certified floor), and bounded
// complete KKT residual growth. The unchanged full-model gate must still pass.
inline bool hybrid_merit_recovery_allowed(const hybrid_inner::ProgressGate& progress,
    double before_merit,double after_merit,double rounding,double before_primal,
    double after_primal,double primal_floor,double before_norm,double after_norm,double primal_alpha) {
 for(double v:{before_merit,after_merit,rounding,before_primal,after_primal,primal_floor,before_norm,after_norm,primal_alpha})
  if(!std::isfinite(v))return false;
 if(!progress.merit_decrease||rounding<0||before_primal<0||after_primal<0||primal_floor<0||
     before_norm<0||after_norm<0||!(primal_alpha>0&&primal_alpha<=1))return false;
 if(!((long double)before_merit-after_merit>rounding))return false;
 const bool primal=before_primal<=primal_floor?after_primal<=primal_floor:
     after_primal<before_primal&&after_primal<=(1-1e-4*primal_alpha)*before_primal;
 return primal&&(long double)after_norm<=2*(long double)before_norm;
}
}
