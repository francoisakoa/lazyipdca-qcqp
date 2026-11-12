#include "lazyipdca_qcqp/direct_ipopt.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
using boost::multiprecision::cpp_rational;

namespace {
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
double old_value(const QuadraticFunction& q,const Vec& x) {
  double linear=0,quadratic=0;
  q.for_each_linear_nonzero([&](int j,double a){linear+=a*x[j];});
  for(auto e:q.hessian.entries) {
    const double product=e.v*x[e.i]*x[e.j];
    quadratic+=e.i==e.j?product:2*product;
  }
  return q.constant+linear+0.5*quadratic;
}
double rational_value(const QuadraticFunction& q,const Vec& x) {
  cpp_rational value=q.constant;
  q.for_each_linear_nonzero([&](int j,double a){value+=cpp_rational(a)*cpp_rational(x[j]);});
  for(auto e:q.hessian.entries) {
    cpp_rational term=cpp_rational(e.v)*cpp_rational(x[e.i])*cpp_rational(x[e.j]);
    value+=e.i==e.j?term/2:term;
  }
  return value.convert_to<double>();
}
QCQPProblem problem(QuadraticFunction q,Vec x) {
  QCQPProblem p;p.n=static_cast<int>(x.size());p.name="accurate_callback_fixture";
  q.hessian.n=p.n;p.objective=std::move(q);p.x0=std::move(x);
  p.lb.assign(p.n,-INFINITY);p.ub.assign(p.n,INFINITY);p.validate();return p;
}
void accuracy(const QCQPProblem& p,bool old_must_fail) {
  OriginalQCQPDerivatives callback(p);
  const double actual=callback.objective_value(p.x0),expected=rational_value(p.objective,p.x0);
  check(std::isfinite(actual),"accurate callback returned nonfinite value");
  const double tolerance=4*std::numeric_limits<double>::epsilon()*
      std::max(std::numeric_limits<double>::min(),std::abs(expected));
  if(std::abs(actual-expected)>tolerance) {
    std::cerr<<std::setprecision(17)<<"actual="<<actual<<" rational="<<expected<<'\n';
    throw std::runtime_error("adaptive callback differs from independent exact rational oracle");
  }
  if(old_must_fail) {
    const double old=old_value(p.objective,p.x0);
    check(!std::isfinite(old)||std::abs(old-expected)>1e-7,
        "regression fixture no longer distinguishes the original inaccurate callback");
  }
}
}
int main(int argc,char** argv) {
  try {
    QuadraticFunction affine;affine.linear={1,1,1};
    accuracy(problem(affine,{4.080442022756951e19,97.29420375586804,-4.080442022756951e19}),true);
    QuadraticFunction mixed;mixed.constant=97;mixed.linear={-1e9};mixed.hessian.entries={{0,0,2}};
    accuracy(problem(mixed,{1e9}),true); // cancellation across constant/linear/quadratic
    QuadraticFunction triple;triple.linear={0,0,0,0};
    triple.hessian.entries={{0,1,1+std::ldexp(1.,-52)},{2,3,-1}};
    accuracy(problem(triple,{1e8,1e8,1e8,1e8}),true); // product rounding as well as summation
    QuadraticFunction extended;extended.linear={0,0};extended.hessian.entries={{0,1,1e-300}};
    accuracy(problem(extended,{1e-100,1e300}),false); // recover intermediate double underflow
    QuadraticFunction overflow;overflow.constant=97;overflow.linear={0,0};
    overflow.hessian.entries={{0,0,2},{1,1,-2}};
    accuracy(problem(overflow,{1e154,1e154}),true); // finite value despite intermediate double overflow
    QuadraticFunction nonfinite;nonfinite.linear={0};nonfinite.hessian.entries={{0,0,2}};
    auto too_large=problem(nonfinite,{1e200});OriginalQCQPDerivatives overflow_callback(too_large);
    check(!std::isfinite(overflow_callback.objective_value(too_large.x0)),
        "genuinely nonfinite binary64 value was manufactured as finite");

    QuadraticFunction ordinary;ordinary.constant=.5;ordinary.linear={1,0,-2};
    ordinary.hessian.entries={{0,0,2},{0,2,3}};
    auto normal=problem(ordinary,{1,2,3});OriginalQCQPDerivatives d(normal);
    const double fast=d.objective_value(normal.x0),historical=old_value(normal.objective,normal.x0);
    check(std::memcmp(&fast,&historical,sizeof(double))==0,"well-conditioned callback fast path changed");

    if(argc==3) {
      auto p=read_qcq(argv[1]);std::ifstream point(argv[2]);Vec x(p.n);
      for(double& value:x)check(bool(point>>value),"cannot read the complete frozen point");
      OriginalQCQPDerivatives callback(p);Vec values;callback.constraint_values(x,values);
      double max_error=0;
      for(std::size_t row=0;row<values.size();++row) {
        const auto& q=row<p.equalities.size()?p.equalities[row]:p.inequalities[row-p.equalities.size()];
        const double exact=rational_value(q,x);
        max_error=std::max(max_error,std::abs(values[row]-exact));
        check(std::abs(values[row]-exact)<=16*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(exact)),
            "public frozen point callback differs from exact rational polynomial");
      }
      check(values.size()>26,"expected enough rows for public point regression");
      std::cout<<std::setprecision(17)<<"PUBLIC_POINT row26_old="<<old_value(p.equalities[26],x)
        <<" row26_accurate="<<values[26]<<" exact="<<rational_value(p.equalities[26],x)
        <<" max_row_error="<<max_error<<'\n';
    } else check(argc==1,"usage: test_direct_ipopt_value_accuracy [model.qcq point.txt]");
    std::cout<<"PASS adaptive direct callback values against exact binary-rational oracle; "
                 "constant/linear/quadratic cancellation and triple-product errors covered\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
