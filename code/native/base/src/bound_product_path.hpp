#pragma once
// Pure scalar arc helper, included beside the barrier path. The includer
// supplies <cmath> and <limits>; no namespace or solver state is introduced.
struct DcaBoundProductPath {
  static long double target_product(double s0,double z0,double alpha,double mu) {
    if(!std::isfinite(s0)||!(s0>0)||!std::isfinite(z0)||!(z0>0)||
        !std::isfinite(alpha)||alpha<0||alpha>1||!std::isfinite(mu)||!(mu>0))
      return std::numeric_limits<long double>::quiet_NaN();
    const long double a=alpha,p0=static_cast<long double>(s0)*z0;
    // Convex combination of positive products: do not form p0+a*(mu-p0),
    // whose cancellation at alpha=1 could lose a much smaller positive mu.
    return std::fma(1-a,p0,a*static_cast<long double>(mu));
  }
  static double dual(double s0,double z0,double ds,double alpha,double mu,double actual_slack) {
    if(!std::isfinite(ds)||!std::isfinite(actual_slack)||!(actual_slack>0))
      return std::numeric_limits<double>::quiet_NaN();
    const long double product=target_product(s0,z0,alpha,mu);
    if(!std::isfinite(product)||!(product>0))return std::numeric_limits<double>::quiet_NaN();
    if(alpha==0)return z0;
    const long double value=product/actual_slack;
    if(!std::isfinite(value)||!(value>0)||value>std::numeric_limits<double>::max())
      return std::numeric_limits<double>::quiet_NaN();
    const double rounded=static_cast<double>(value);
    // Underflow is rejected as a domain failure; no dual floor changes the arc.
    return std::isfinite(rounded)&&rounded>0?rounded:std::numeric_limits<double>::quiet_NaN();
  }
  static long double complementarity_direction_error(double s0,double z0,double ds,double dz,double mu) {
    if(!std::isfinite(s0)||!(s0>0)||!std::isfinite(z0)||!(z0>0)||
        !std::isfinite(ds)||!std::isfinite(dz)||!std::isfinite(mu)||!(mu>0))
      return std::numeric_limits<long double>::quiet_NaN();
    const long double rc=std::fma(static_cast<long double>(s0),static_cast<long double>(z0),-static_cast<long double>(mu));
    return std::fma(static_cast<long double>(s0),static_cast<long double>(dz),
        std::fma(static_cast<long double>(z0),static_cast<long double>(ds),rc));
  }
  // Arc derivative minus the COMPUTED Newton dual direction. This is zero
  // for exact Newton equations, and -ec/s0 for an inexact audited solve.
  static long double tangency_defect(double s0,double z0,double ds,double dz,double mu) {
    return -complementarity_direction_error(s0,z0,ds,dz,mu)/s0;
  }
};
