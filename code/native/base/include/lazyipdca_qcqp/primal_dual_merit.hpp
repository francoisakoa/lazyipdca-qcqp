#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace lazyipdca::qcqp::detail::gr {
// Native convention L=f+y'C. C contains ORIGINAL h and source g+s;
// b contains the UNSTABILIZED canonical-bound link residuals g_B+s_B.
// The source center, mu, rho and kappa are frozen during one line search.
struct State { double objective=0; std::vector<double> c,y,s,b; };
struct Parameters { double mu=0,rho=0,kappa=0; std::vector<double> center; };
struct Direction { double objective=0; std::vector<double> c,y,s,b; };
struct Result {
  bool valid=false;
  long double value=std::numeric_limits<long double>::quiet_NaN();
  long double barrier=0,center=0,constraint=0,shifted=0,hard_bounds=0;
  long double true_c_inf=0,regularized_c_inf=0;
};
struct KappaChoice {bool valid=false;double value=0;};
struct Sum {
  long double hi=0,lo=0;
  void add(long double x) {
    const long double t=hi+x;
    lo+=std::abs(hi)>=std::abs(x)?(hi-t)+x:(x-t)+hi;hi=t;
  }
  void product(long double a,long double b) {
    const long double p=a*b;add(p);
    if(std::isfinite(p))add(std::fma(a,b,-p));
  }
  long double get()const{return hi+lo;}
};
inline bool finite(const std::vector<double>& a) {
  for(double v:a)if(!std::isfinite(v))return false;
  return true;
}
inline bool valid(const State& a,const Parameters& p) {
  // With no inequality or bound barrier terms the exact parameter is zero.
  // Keep positivity for every state that contains an effective barrier row.
  if(!std::isfinite(a.objective)||!std::isfinite(p.mu)||p.mu<0||
      (p.mu==0&&(!a.s.empty()||!a.b.empty()))||
      !std::isfinite(p.rho)||!(p.rho>0)||!std::isfinite(p.kappa)||p.kappa<0||
      a.c.size()!=a.y.size()||a.c.size()!=p.center.size()||
      !finite(a.c)||!finite(a.y)||!finite(a.s)||!finite(a.b)||!finite(p.center))return false;
  for(double s:a.s)if(!(s>0))return false;
  return true;
}
inline Result finish(Result r) {
  Sum s;s.add(r.barrier);s.add(r.center);s.add(r.constraint);s.add(r.shifted);s.add(r.hard_bounds);
  r.value=s.get();r.valid=std::isfinite(r.value)&&std::isfinite(r.barrier)&&
      std::isfinite(r.center)&&std::isfinite(r.constraint)&&std::isfinite(r.shifted)&&
      std::isfinite(r.hard_bounds)&&std::isfinite(r.true_c_inf)&&std::isfinite(r.regularized_c_inf);
  return r;
}
inline long double shifted(const State& a,const Parameters& p,std::size_t i) {
  return std::fma(-static_cast<long double>(p.rho),
      static_cast<long double>(a.y[i])-p.center[i],static_cast<long double>(a.c[i]));
}
inline Result evaluate(const State& a,const Parameters& p) {
  Result r;if(!valid(a,p))return r;
  Sum phi,linear,cquad,fquad,hard;phi.add(a.objective);
  for(double s:a.s)phi.product(-static_cast<long double>(p.mu),std::log(static_cast<long double>(s)));
  for(std::size_t i=0;i<a.c.size();++i) {
    const long double c=a.c[i],f=shifted(a,p,i);
    linear.product(p.center[i],c);cquad.product(c/(2*static_cast<long double>(p.rho)),c);
    fquad.product(f/(2*static_cast<long double>(p.rho)),f);
    r.true_c_inf=std::max(r.true_c_inf,std::abs(c));r.regularized_c_inf=std::max(r.regularized_c_inf,std::abs(f));
  }
  for(double b:a.b)hard.product(p.kappa,std::abs(static_cast<long double>(b)));
  r.barrier=phi.get();r.center=linear.get();r.constraint=cquad.get();r.shifted=fquad.get();r.hard_bounds=hard.get();
  return finish(r);
}
inline bool valid_direction(const State& a,const Direction& d) {
  return std::isfinite(d.objective)&&d.c.size()==a.c.size()&&d.y.size()==a.y.size()&&
      d.s.size()==a.s.size()&&d.b.size()==a.b.size()&&finite(d.c)&&finite(d.y)&&finite(d.s)&&finite(d.b);
}
// One-sided L1 derivative also handles an exactly zero hard-bound residual.
inline long double l1_derivative(double b,double db) {
  return b>0?static_cast<long double>(db):(b<0?-static_cast<long double>(db):std::abs(static_cast<long double>(db)));
}
inline Result directional_with_barrier_slope(const State& a,const Direction& d,
    const Parameters& p,long double phi_slope) {
  Result r;if(!valid(a,p)||!valid_direction(a,d)||!std::isfinite(phi_slope))return r;
  Sum linear,cquad,fquad,hard;
  for(std::size_t i=0;i<a.c.size();++i) {
    const long double c=a.c[i],f=shifted(a,p,i),dc=d.c[i];
    const long double df=std::fma(-static_cast<long double>(p.rho),static_cast<long double>(d.y[i]),dc);
    linear.product(p.center[i],dc);cquad.product(c/p.rho,dc);fquad.product(f/p.rho,df);
    r.true_c_inf=std::max(r.true_c_inf,std::abs(c));r.regularized_c_inf=std::max(r.regularized_c_inf,std::abs(f));
  }
  for(std::size_t i=0;i<a.b.size();++i)hard.product(p.kappa,l1_derivative(a.b[i],d.b[i]));
  r.barrier=phi_slope;r.center=linear.get();r.constraint=cquad.get();r.shifted=fquad.get();r.hard_bounds=hard.get();
  return finish(r);
}
inline Result directional(const State& a,const Direction& d,const Parameters& p) {
  if(!valid(a,p)||!valid_direction(a,d))return {};
  Sum phi;phi.add(d.objective);
  for(std::size_t i=0;i<a.s.size();++i)phi.product(-static_cast<long double>(p.mu),static_cast<long double>(d.s[i])/a.s[i]);
  return directional_with_barrier_slope(a,d,p,phi.get());
}
inline bool same_dimensions(const State& a,const State& b) {
  return a.c.size()==b.c.size()&&a.y.size()==b.y.size()&&a.s.size()==b.s.size()&&a.b.size()==b.b.size();
}
// Never subtract two complete merit scores. The caller supplies the original
// barrier difference, e.g. the already compensated canonical path difference.
inline Result change_with_barrier_delta_impl(const State& a,const State& b,
    const Parameters& p,long double phi_delta,const std::vector<long double>* source_delta) {
  Result r;if(!valid(a,p)||!valid(b,p)||!same_dimensions(a,b)||!std::isfinite(phi_delta))return r;
  if(source_delta) {
    if(source_delta->size()!=a.c.size())return r;
    for(long double d:*source_delta)if(!std::isfinite(d))return r;
  }
  Sum linear,cquad,fquad,hard;
  for(std::size_t i=0;i<a.c.size();++i) {
    const long double dc=source_delta?(*source_delta)[i]:static_cast<long double>(b.c[i])-a.c[i];
    const long double dy=static_cast<long double>(b.y[i])-a.y[i];
    const long double fa=shifted(a,p,i),df=std::fma(-static_cast<long double>(p.rho),dy,dc);
    linear.product(p.center[i],dc);
    cquad.product((static_cast<long double>(a.c[i])+.5L*dc)/p.rho,dc);
    fquad.product((fa+.5L*df)/p.rho,df);
    r.true_c_inf=std::max(r.true_c_inf,std::abs(static_cast<long double>(b.c[i])));
    r.regularized_c_inf=std::max(r.regularized_c_inf,std::abs(shifted(b,p,i)));
  }
  for(std::size_t i=0;i<a.b.size();++i)hard.product(p.kappa,
      std::abs(static_cast<long double>(b.b[i]))-std::abs(static_cast<long double>(a.b[i])));
  r.barrier=phi_delta;r.center=linear.get();r.constraint=cquad.get();r.shifted=fquad.get();r.hard_bounds=hard.get();
  return finish(r);
}
inline Result change_with_barrier_delta(const State& a,const State& b,
    const Parameters& p,long double phi_delta) {
  return change_with_barrier_delta_impl(a,b,p,phi_delta,nullptr);
}
// source_delta contains ORIGINAL coefficient differences at the ACTUAL trial
// point, plus the ACTUAL source slack increments. State c values remain the
// original feasibility observations, including for true/regularized C logs.
// The merit, frozen parameters, bound penalty and Armijo rule are unchanged.
inline Result change_with_barrier_delta(const State& a,const State& b,
    const Parameters& p,long double phi_delta,const std::vector<long double>& source_delta) {
  return change_with_barrier_delta_impl(a,b,p,phi_delta,&source_delta);
}
// objective_delta must itself be evaluated from the ORIGINAL coefficients or
// an accurate path formula; subtracting rounded huge objective values loses
// information before this helper can recover it.
inline Result change(const State& a,const State& b,const Parameters& p,long double objective_delta) {
  if(!valid(a,p)||!valid(b,p)||!same_dimensions(a,b)||!std::isfinite(objective_delta))return {};
  Sum phi;phi.add(objective_delta);
  for(std::size_t i=0;i<a.s.size();++i) {
    const long double old=a.s[i],now=b.s[i],ds=now-old;
    const long double logs=std::abs(ds)<=.5L*old?std::log1p(ds/old):std::log(now)-std::log(old);
    phi.product(-static_cast<long double>(p.mu),logs);
  }
  return change_with_barrier_delta(a,b,p,phi.get());
}
inline Result change_with_constraint_delta(const State& a,const State& b,
    const Parameters& p,long double objective_delta,const std::vector<long double>& source_delta) {
  if(!valid(a,p)||!valid(b,p)||!same_dimensions(a,b)||!std::isfinite(objective_delta))return {};
  Sum phi;phi.add(objective_delta);
  for(std::size_t i=0;i<a.s.size();++i) {
    const long double old=a.s[i],now=b.s[i],ds=now-old;
    const long double logs=std::abs(ds)<=.5L*old?std::log1p(ds/old):std::log(now)-std::log(old);
    phi.product(-static_cast<long double>(p.mu),logs);
  }
  return change_with_barrier_delta(a,b,p,phi.get(),source_delta);
}
inline KappaChoice choose_kappa(double current,const std::vector<double>& zB,const std::vector<double>& dzB) {
  KappaChoice out;if(!std::isfinite(current)||current<0||zB.size()!=dzB.size()||!finite(zB)||!finite(dzB))return out;
  long double k=current;
  for(std::size_t i=0;i<zB.size();++i)k=std::max(k,2*std::abs(static_cast<long double>(zB[i])+dzB[i]));
  if(!std::isfinite(k)||k>std::numeric_limits<double>::max())return out;
  out.value=static_cast<double>(k);
  if(static_cast<long double>(out.value)<k)out.value=std::nextafter(out.value,std::numeric_limits<double>::infinity());
  out.valid=std::isfinite(out.value);return out;
}
inline bool armijo(const Result& delta,const Result& slope,double alpha,double constant) {
  return delta.valid&&slope.valid&&slope.value<0&&std::isfinite(alpha)&&alpha>0&&alpha<=1&&
      std::isfinite(constant)&&constant>0&&constant<1&&delta.value<=static_cast<long double>(constant)*alpha*slope.value;
}
} // namespace lazyipdca::qcqp::detail::gr
