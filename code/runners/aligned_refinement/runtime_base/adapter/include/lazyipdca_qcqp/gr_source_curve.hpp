#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp::detail {
class GrSourceCurveRow {
  double g0_=0,s0_=0,z0_=0,center_=0,mu_=0,rho_=0;
  long double f0_=0,r0_=0,b0_=0,product_=0,forcing_=0;
  bool valid_=false;
  struct Sum {
    long double hi=0,lo=0;
    void add(long double x) {
      const long double t=hi+x;
      lo+=std::abs(hi)>=std::abs(x)?(hi-t)+x:(x-t)+hi;hi=t;
    }
    void product(long double a,long double b) {
      const long double p=a*b;add(p);add(std::fma(a,b,-p));
    }
    void quotient(long double a,long double b) {
      const long double q=a/b;add(q);add(std::fma(-q,b,a)/b);
    }
    long double value()const{return hi+lo;}
  };
  struct Audit {
    bool valid=false;long double maximum=INFINITY,forward=INFINITY,correction=0;
  };
  Audit audit(double alpha,long double dg,double s,double z)const {
    Audit out;
    if(!(s>0)||!std::isfinite(s)||!std::isfinite(z))return out;
    const long double S=s,S0=s0_,Z=z,Z0=z0_,A=alpha,P=rho_,M=mu_;
    const long double ds=S-S0,dz=Z-Z0,secant=M/S/S0;
    Sum fp;fp.add(dg);fp.add(ds);fp.add(-P*dz);fp.add(A*f0_);
    Sum rs;rs.add(dz);rs.add(secant*ds);rs.add(A*r0_);
    const long double f=fp.value(),r=rs.value();
    const long double sf=1+std::abs(dg)+std::abs(ds)+std::abs(P*dz)+std::abs(A*f0_);
    const long double sr=1+std::abs(dz)+std::abs(secant*ds)+std::abs(A*r0_);
    const long double weight=M/S/S;
    Sum numerator;numerator.add(r);numerator.add(-weight*f);
    const long double correction=numerator.value()/(1+P*weight);
    out.maximum=std::max(std::abs(f)/sf,std::abs(r)/sr);
    out.forward=std::abs(correction)/std::max(1.L,std::abs(Z));out.correction=correction;
    out.valid=std::isfinite(out.maximum)&&std::isfinite(out.forward)&&std::isfinite(correction);
    return out;
  }
 public:
  struct Tangent {bool valid=false;double slack=0,dual=0;};
  struct Point {bool valid=false;double slack=0,dual=0;};
  GrSourceCurveRow()=default;
  GrSourceCurveRow(double g,double s,double z,double center,double mu,double rho)
      :g0_(g),s0_(s),z0_(z),center_(center),mu_(mu),rho_(rho) {
    for(double v:{g,s,z,center,mu,rho})if(!std::isfinite(v))return;
    if(!(s>0&&mu>0&&rho>0))return;
    const long double G=g,S=s,Z=z,E=center,M=mu,P=rho;
    product_=P*M;
    Sum f;f.add(G);f.add(S);f.product(-P,Z);f.product(P,E);f0_=f.value();
    Sum r;r.add(Z);r.quotient(-M,S);r0_=r.value();
    const long double product_low=std::fma(P,M,-product_);
    const long double q=product_/S;
    const long double qlow=(std::fma(-q,S,product_)+product_low)/S;
    Sum b;b.add(q);b.add(qlow);b.add(-S);b0_=b.value();
    Sum force;force.add(G);force.add(S);force.product(P,E);force.add(-q);force.add(-qlow);forcing_=force.value();
    valid_=std::isfinite(product_)&&product_>0&&std::isfinite(f0_)&&
        std::isfinite(r0_)&&std::isfinite(b0_)&&std::isfinite(forcing_);
  }
  bool valid()const{return valid_;}
  long double primal_residual()const{return f0_;}
  long double slack_gradient()const{return r0_;}
  Tangent tangent(long double dg)const {
    Tangent out;if(!valid_||!std::isfinite(dg))return out;
    const long double weight=static_cast<long double>(mu_)/s0_/s0_;
    Sum a;a.add(dg);a.add(forcing_);
    const long double ds=-a.value()/(1+static_cast<long double>(rho_)*weight);
    Sum z;
    if(static_cast<long double>(rho_)*weight>=1) {
      z.add(dg);z.add(ds);z.add(f0_);
      const long double numerator=z.value();z=Sum{};z.add(numerator/rho_);
    } else {z.add(-r0_);z.add(-weight*ds);}
    out.slack=static_cast<double>(ds);out.dual=static_cast<double>(z.value());
    out.valid=std::isfinite(ds)&&std::isfinite(z.value())&&std::isfinite(out.slack)&&std::isfinite(out.dual);
    return out;
  }
  Point at(double alpha,long double dg_actual)const {
    Point out;
    if(!valid_||!std::isfinite(alpha)||alpha<0||alpha>1||!std::isfinite(dg_actual))return out;
    if(alpha==0){if(dg_actual!=0)return out;out.valid=true;out.slack=s0_;out.dual=z0_;return out;}
    const long double a=alpha,S0=s0_,M=mu_,P=rho_,Z0=z0_;
    Sum delta;delta.add(dg_actual);delta.add(a*forcing_);
    Sum bt;bt.add(b0_);bt.add(delta.value());const long double b=bt.value();
    if(!std::isfinite(b))return out;
    const long double disc=std::hypot(b,2*std::sqrt(product_));
    const long double root=b>=0?2*product_/(disc+b):(disc-b)/2;
    const double rounded=static_cast<double>(root);
    if(!(rounded>0)||!std::isfinite(root)||!std::isfinite(rounded))return out;
    constexpr long double tolerance=64*std::numeric_limits<double>::epsilon();
    // The well-conditioned reconstruction normally passes immediately.
    // Fallback below is bounded and never changes the acceptance accuracy.
    {
      Sum z;
      if(P*M/root/root>=1) {
        z.add(dg_actual);z.add(root-S0);z.add(a*f0_);
        const long double numerator=z.value();z=Sum{};z.add(Z0);z.add(numerator/P);
      } else {z.add(Z0);z.add(M*(S0-root)/root/S0);z.add(-a*r0_);}
      const double dual=static_cast<double>(z.value());
      const auto check=audit(alpha,dg_actual,rounded,dual);
      if(check.valid&&check.maximum<=8*std::numeric_limits<double>::epsilon()&&
          check.forward<=8*std::numeric_limits<double>::epsilon()) {
        out.valid=true;out.slack=rounded;out.dual=dual;return out;
      }
    }
    const double slacks[]={rounded,std::nextafter(rounded,0.),std::nextafter(rounded,INFINITY)};
    long double best=INFINITY;
    for(double s:slacks) {
      if(!(s>0)||!std::isfinite(s))continue;
      const long double S=s;
      // The two algebraically equivalent reconstructions condition differently.
      // Both the unrounded root and the actual rounded slack provide candidates.
      for(long double trial_s:{root,S}) {
        Sum zp;zp.add(dg_actual);zp.add(trial_s-S0);zp.add(a*f0_);
        const long double primal=Z0+zp.value()/P;
        Sum zg;zg.add(Z0);zg.add(M*(S0-trial_s)/trial_s/S0);zg.add(-a*r0_);
        const long double gradient=zg.value();
        const long double secant=M/trial_s/S0;
        const long double ds=-delta.value()/(1+P*secant);
        Sum zi;zi.add(Z0);zi.add(-a*r0_);zi.add(-secant*ds);
        for(long double initial:{primal,gradient,zi.value()}) {
          if(!std::isfinite(initial))continue;
          double z=static_cast<double>(initial);
          for(int refinement=0;refinement<2;++refinement) {
            const double duals[]={z,std::nextafter(z,-INFINITY),std::nextafter(z,INFINITY)};
            for(double candidate:duals) {
              const auto check=audit(alpha,dg_actual,s,candidate);
              if(!check.valid||check.maximum>tolerance||check.forward>tolerance)continue;
              const long double score=std::max(check.maximum,check.forward);
              if(score<best){best=score;out.valid=true;out.slack=s;out.dual=candidate;}
            }
            const auto check=audit(alpha,dg_actual,s,z);
            if(!check.valid)break;
            z=static_cast<double>(static_cast<long double>(z)-check.correction);
          }
        }
      }
    }
    return out;
  }
};
// Transactional realization of ONE source row. A numerical refusal leaves
// exactly the already staged affine row, using the same audited tangent.
inline bool gr_source_curve_or_affine(const GrSourceCurveRow& row,double alpha,
    long double delta_g,long double minimum_slack,double& staged_slack,double& staged_dual) {
  if(!std::isfinite(minimum_slack)||minimum_slack<0)return false;
  const auto point=row.at(alpha,delta_g);
  if(!point.valid||!(static_cast<long double>(point.slack)>=minimum_slack))return false;
  staged_slack=point.slack;staged_dual=point.dual;return true;
}
}
