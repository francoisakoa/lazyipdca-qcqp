#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp::detail {

// Curvature-only realization of the ALREADY AUDITED native Newton direction.
// It preserves the actual affine source F residual pointwise, even if g is
// rounded or nonlinear. It is not an original KKT certificate.
class GrSourceNewtonCurveRow {
  double s0_=0,z0_=0,mu_=0,rho_=0,ds_=0,dz_=0;
  long double weight_=0;
  bool valid_=false;
  struct Sum {
    long double hi=0,lo=0;
    void add(long double x) {
      const long double t=hi+x;
      lo+=std::abs(hi)>=std::abs(x)?(hi-t)+x:(x-t)+hi;hi=t;
    }
    long double value()const{return hi+lo;}
  };
 public:
  struct Point {bool valid=false;double slack=0,dual=0;};
  GrSourceNewtonCurveRow()=default;
  GrSourceNewtonCurveRow(double s,double z,double mu,double rho,double ds,double dz)
      :s0_(s),z0_(z),mu_(mu),rho_(rho),ds_(ds),dz_(dz) {
    for(double x:{s,z,mu,rho,ds,dz})if(!std::isfinite(x))return;
    if(!(s>0&&mu>0&&rho>0))return;
    weight_=static_cast<long double>(mu)/s/s;
    valid_=std::isfinite(weight_)&&weight_>0;
  }
  bool valid()const{return valid_;}
  Point at(double alpha)const {
    // Match the historical affine arrays' double arithmetic exactly. The
    // staging overload below should be used when those arrays already exist.
    const double affine_s=s0_+alpha*ds_,affine_z=z0_+alpha*dz_;
    return at(alpha,affine_s,affine_z);
  }
  Point at(double alpha,double supplied_affine_s,double supplied_affine_z)const {
    Point out;
    if(!valid_||!std::isfinite(alpha)||alpha<0||alpha>1||
        !std::isfinite(supplied_affine_s)||!std::isfinite(supplied_affine_z))return out;
    if(alpha==0){
      if(supplied_affine_s!=s0_||supplied_affine_z!=z0_)return out;
      out.valid=true;out.slack=s0_;out.dual=z0_;return out;
    }
    const long double S0=s0_,Z0=z0_,M=mu_,P=rho_;
    const long double affine_s=supplied_affine_s,affine_z=supplied_affine_z;
    const long double u=affine_s-S0,v=affine_z-Z0;
    if(!(affine_s>0)||!std::isfinite(affine_s)||!std::isfinite(affine_z))return out;
    const long double t=P*weight_;
    Sum bs;bs.add(affine_s);bs.add(t*(S0-u));
    const long double b=bs.value(),c=weight_*u*u;
    if(!std::isfinite(b)||!std::isfinite(c)||c<0)return out;
    long double q=0;
    if(c>0) {
      const long double discriminant=std::hypot(b,2*std::sqrt(P)*std::sqrt(c));
      q=b>=0?2*c/(discriminant+b):(discriminant-b)/(2*P);
    }
    if(!(q>=0)||!std::isfinite(q))return out;
    Sum s;s.add(affine_s);s.add(P*q);
    Sum z;z.add(affine_z);z.add(q);
    out.slack=static_cast<double>(s.value());out.dual=static_cast<double>(z.value());
    if(!(out.slack>0)||!std::isfinite(out.slack)||!std::isfinite(out.dual))return Point{};
    // Audit the rounded pair actually staged/committed. These are local
    // backward-error checks, independent of the original KKT tolerances.
    const long double S=out.slack,Z=out.dual,delta_s=S-affine_s,delta_z=Z-affine_z;
    Sum f;f.add(delta_s);f.add(-P*delta_z);
    const long double inverse_delta=M*(S0-S)/S/S0;
    Sum r;r.add(Z-Z0);r.add(-inverse_delta);r.add(-v);r.add(-weight_*u);
    const long double sf=1+std::abs(delta_s)+std::abs(P*delta_z);
    const long double sr=1+std::abs(Z-Z0)+std::abs(inverse_delta)+std::abs(v)+std::abs(weight_*u);
    constexpr long double tolerance=64*std::numeric_limits<double>::epsilon();
    if(!std::isfinite(f.value())||!std::isfinite(r.value())||!std::isfinite(sf)||!std::isfinite(sr)||
        std::abs(f.value())>tolerance*sf||std::abs(r.value())>tolerance*sr)return Point{};
    out.valid=true;return out;
  }
};
// A numerical refusal leaves exactly the source affine arrays already staged
// from the audited native direction. It never changes that direction.
inline bool gr_source_newton_curve_or_affine(const GrSourceNewtonCurveRow& row,double alpha,
    long double minimum_slack,double& staged_slack,double& staged_dual) {
  if(!std::isfinite(minimum_slack)||minimum_slack<0)return false;
  const auto point=row.at(alpha,staged_slack,staged_dual);
  if(!point.valid||!(static_cast<long double>(point.slack)>=minimum_slack))return false;
  staged_slack=point.slack;staged_dual=point.dual;return true;
}
}
