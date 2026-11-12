#pragma once
// Pure original-bound slack path. The includer supplies cmath and limits.
// g0 and ga are DOUBLE original row evaluations, as used by the actual merit.
struct DcaConsistentBoundSlack {
  static long double origin_link(double g0,double s0) {
    if(!std::isfinite(g0)||!std::isfinite(s0)||!(s0>0))
      return std::numeric_limits<long double>::quiet_NaN();
    return static_cast<long double>(g0)+s0;
  }
  static double value(double g0,double s0,double ga,double alpha) {
    const long double b0=origin_link(g0,s0);
    if(!std::isfinite(b0)||!std::isfinite(ga)||!std::isfinite(alpha)||alpha<0||alpha>1)
      return std::numeric_limits<double>::quiet_NaN();
    if(alpha==0)return s0;
    const long double slack=(1-static_cast<long double>(alpha))*b0-static_cast<long double>(ga);
    if(!std::isfinite(slack)||std::abs(slack)>std::numeric_limits<double>::max())
      return std::numeric_limits<double>::quiet_NaN();
    // Nonpositive/underflowed values are rejected by the actual domain gate.
    // Never floor a slack or weaken an acceptance/termination tolerance.
    return static_cast<double>(slack);
  }
  static long double slope(double g0,double s0,long double original_row_slope) {
    const long double b0=origin_link(g0,s0);
    if(!std::isfinite(b0)||!std::isfinite(original_row_slope))
      return std::numeric_limits<long double>::quiet_NaN();
    return -original_row_slope-b0;
  }
  // Path ds minus the computed Newton ds is minus its bound-link row error.
  static long double tangent_defect(double g0,double s0,long double row_slope,double computed_ds) {
    return slope(g0,s0,row_slope)-static_cast<long double>(computed_ds);
  }
};
