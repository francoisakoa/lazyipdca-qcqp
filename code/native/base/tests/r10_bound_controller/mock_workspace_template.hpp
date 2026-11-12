#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "lazyipdca_qcqp/hybrid_slack_path.hpp"
using Vec=std::vector<double>;
namespace hybrid_slack=lazyipdca::qcqp::hybrid_slack;
struct SymTriplet {int i=0,j=0;double v=0;};
struct Quadratic {
  double constant=0,value=0,scale=1;
  bool majorant_gradients_extended=false;
  std::vector<int> support;
  Vec coefficients,gradient;
  std::vector<long double> gradient_extended;
  std::vector<SymTriplet> hessian,base_hessian;
  std::vector<std::pair<int,double>> signed_square;
  double original_at(const Vec& x)const {
    long double f=constant;
    for(std::size_t i=0;i<support.size();++i)f+=static_cast<long double>(coefficients[i])*x[support[i]];
    for(auto e:hessian)f+=(e.i==e.j?.5L:1.L)*e.v*x[e.i]*x[e.j];
    for(auto e:signed_square)f+=static_cast<long double>(e.second)*x[e.first]*std::abs(x[e.first]);
    return static_cast<double>(scale*f);
  }
  void at_anchor(const Vec& x) {
    value=original_at(x);gradient.resize(support.size());gradient_extended.resize(support.size());
    for(std::size_t k=0;k<support.size();++k) {
      long double g=scale*coefficients[k];const int j=support[k];
      for(auto e:hessian) {
        if(e.i==j)g+=scale*e.v*x[e.j];
        if(e.j==j&&e.i!=e.j)g+=scale*e.v*x[e.i];
      }
      for(auto e:signed_square)if(e.first==j)g+=2*scale*e.second*std::abs(x[j]);
      gradient[k]=static_cast<double>(g);gradient_extended[k]=g;
    }
  }
};
struct Direction {Vec x,s,z,lambda;};
struct Omitted {int sides=0;double maximum=0;long double sum=0;};
struct RankContext {bool has_omitted_equality_bounds()const{return false;}};
struct MockSettings {std::shared_ptr<RankContext> rank_presolve_context;};
struct Workspace {
  int n=0,me=0,mi=0;
  Quadratic objective;std::vector<Quadratic> equalities,inequalities;
  Vec x,s,z,lambda,rd;std::vector<int> active,inequality_kind,free_source;
  MockSettings st;
  bool source_primal_barrier(int i)const{return free_source.at(i)!=0;}
  bool requires_positive_dual(int i)const{return !source_primal_barrier(i);}
  Omitted omitted_original_bounds(const Vec&)const{return {};}
  double omitted_original_bound_scale()const{return 1;}
};
inline void axpy(double alpha,const Vec& d,Vec& point) {
  if(d.size()!=point.size())throw std::runtime_error("test axpy shape mismatch");
  for(std::size_t i=0;i<point.size();++i)point[i]+=alpha*d[i];
}
inline bool finite(const Vec& values) {
  for(double value:values)if(!std::isfinite(value))return false;
  return true;
}
