#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace lazyipdca::qcqp::hybrid_slack {
// Stable log(b/a), including ratios too extreme for b/a or log1p((b-a)/a).
inline long double log_ratio(double a,double b) {
  if(!(a>0&&b>0)||!std::isfinite(a)||!std::isfinite(b))
    throw std::invalid_argument("geometric slack endpoints must be positive and finite");
  const long double x=a,y=b;
  if(y>=.5L*x&&y<=2.L*x)return std::log1p((y-x)/x);
  return std::log(y)-std::log(x);
}
struct Curve {
  double first=1,last=1;
  long double rate=0;
  Curve()=default;
  Curve(double a,double b):first(a),last(b),rate(log_ratio(a,b)){}
  double value(double alpha) const {
    if(!std::isfinite(alpha)||alpha<0||alpha>1)
      throw std::invalid_argument("geometric slack alpha outside [0,1]");
    // Preserve the exact binary endpoint certified by the model gate.
    if(alpha==0)return first;
    if(alpha==1)return last;
    long double point;
    if(std::abs(rate)<=1)point=static_cast<long double>(first)*std::exp(alpha*rate);
    else point=std::exp((1-static_cast<long double>(alpha))*std::log(static_cast<long double>(first))+
                        static_cast<long double>(alpha)*std::log(static_cast<long double>(last)));
    // Only contain rounding in the endpoint interval. This is not a change
    // to an acceptance tolerance, domain bound or directional derivative.
    return std::clamp(static_cast<double>(point),std::min(first,last),std::max(first,last));
  }
  long double actual_log_change(double actual) const {return log_ratio(first,actual);}
};
}
