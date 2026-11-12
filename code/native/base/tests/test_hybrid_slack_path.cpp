#include "lazyipdca_qcqp/hybrid_slack_path.hpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
using namespace lazyipdca::qcqp;
int main(){int checks=0;auto require=[&](bool ok,const char* m){++checks;if(!ok)throw std::runtime_error(m);};
  auto bits=[](double x){std::uint64_t v;std::memcpy(&v,&x,sizeof(x));return v;};
  const double tiny=std::numeric_limits<double>::denorm_min(),large=std::numeric_limits<double>::max();
  for(auto endpoints:{std::pair<double,double>{tiny,large},{large,tiny},{1.,std::nextafter(1.,2.)},{1e-300,1e-250},{.5,.5},{.5,1.}}) {
    hybrid_slack::Curve c(endpoints.first,endpoints.second);
    require(bits(c.value(0))==bits(endpoints.first)&&bits(c.value(1))==bits(endpoints.second),"geometric endpoint bits changed");
    require(std::isfinite(c.rate),"finite endpoints produced nonfinite derivative");
    for(double a:{0.,.125,.5,.75,1.}) {
      const double s=c.value(a);require(std::isfinite(s)&&s>0,"geometric slack left its original positive domain");
      require(s>=std::min(c.first,c.last)&&s<=std::max(c.first,c.last),"geometric slack left endpoint interval");
      require(std::isfinite(c.actual_log_change(s)),"rounded slack log change nonfinite");
    }
  }
  hybrid_slack::Curve c(.5,2.);const long double h=1e-6L;
  const long double derivative=(std::log(static_cast<long double>(c.value(double(h))))-std::log(.5L))/h;
  require(std::abs(derivative-c.rate)<1e-9L,"log-barrier derivative does not match geometric curve");
  hybrid_slack::Curve subulp(1.,std::nextafter(1.,2.));
  require(subulp.value(1e-6)==1.&&subulp.actual_log_change(subulp.value(1e-6))==0,
      "unrepresented slack move manufactured barrier progress");
  require(subulp.rate>0,"stable log1p lost adjacent endpoint derivative");
  for(double bad:{0.,-1.,double(INFINITY),double(NAN)}) {bool reject=false;try{hybrid_slack::Curve invalid(1,bad);}catch(const std::invalid_argument&){reject=true;}require(reject,"invalid slack endpoint admitted");}
  bool reject=false;try{c.value(-1);}catch(const std::invalid_argument&){reject=true;}require(reject,"invalid alpha admitted");
  std::cout<<"hybrid_slack_path PASS checks="<<checks<<" exact_endpoints=1 exact_log_derivative=1 rounded_progress=1 extreme_log_ratio=1\n";
}
