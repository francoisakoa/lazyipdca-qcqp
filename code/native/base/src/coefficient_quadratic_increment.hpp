#pragma once
#include <cmath>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {
// Difference of ORIGINAL stored functions at two actual binary64 points.
// Constants cancel algebraically. No cached gradient, auxiliary Hessian or
// nominal alpha*d enters this calculation. The secant identity also avoids
// subtracting two rounded products near a nonzero anchor.
class CoefficientIncrementSum {
  long double hi_=0,lo_=0;
 public:
  void add(long double x) {
    const long double t=hi_+x;
    lo_+=std::abs(hi_)>=std::abs(x)?(hi_-t)+x:(x-t)+hi_;hi_=t;
  }
  void product(long double a,long double b) {
    const long double p=a*b;add(p);
    if(std::isfinite(p))add(std::fma(a,b,-p));
  }
  void triple(long double a,long double b,long double c,long double factor=1.L) {
    const long double p=a*b;
    product(factor*p,c);
    if(std::isfinite(p))product(factor*std::fma(a,b,-p),c);
  }
  long double get()const{return hi_+lo_;}
};

template<class Point,class Linear,class Hessian,class SignedSquare>
long double coefficient_quadratic_increment(const Point& x,const Point& trial,
    const Linear& linear,const Hessian& hessian,const SignedSquare& signed_square,
    double scale=1.) {
  if(x.size()!=trial.size()||!std::isfinite(scale))
    throw std::invalid_argument("original coefficient increment: invalid dimension or scale");
  CoefficientIncrementSum delta;
  auto difference=[&](int j) {
    if(j<0||static_cast<std::size_t>(j)>=x.size()||
        !std::isfinite(x[j])||!std::isfinite(trial[j]))
      throw std::invalid_argument("original coefficient increment: invalid coordinate");
    return static_cast<long double>(trial[j])-static_cast<long double>(x[j]);
  };
  for(const auto& e:linear)delta.product(e.second,difference(e.first));
  for(const auto& e:hessian) {
    const long double di=difference(e.i),dj=difference(e.j);
    const long double si=static_cast<long double>(trial[e.i])+x[e.i];
    const long double sj=static_cast<long double>(trial[e.j])+x[e.j];
    if(e.i==e.j)delta.triple(e.v,di,si,.5L);
    else {delta.triple(e.v,di,sj,.5L);delta.triple(e.v,dj,si,.5L);}
  }
  for(const auto& e:signed_square) {
    const long double d=difference(e.first),a=x[e.first],b=trial[e.first];
    if((a>=0&&b>=0)||(a<=0&&b<=0))
      delta.triple(e.second,d,a+b,(a<0||b<0)?-1.L:1.L);
    else {
      // Across zero the two signed squares have opposite signs, so this
      // subtraction adds their magnitudes rather than cancelling them.
      delta.product(e.second,b*std::abs(b)-a*std::abs(a));
    }
  }
  CoefficientIncrementSum scaled;scaled.product(scale,delta.get());
  const long double result=scaled.get();
  if(!std::isfinite(result))
    throw std::runtime_error("original coefficient increment is nonfinite");
  return result;
}
} // namespace lazyipdca::qcqp::detail
