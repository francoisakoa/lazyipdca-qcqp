#include "lazyipdca_qcqp/primal_dual_merit.hpp"
#include "coefficient_quadratic_increment.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace lazyipdca::qcqp;
struct Entry {int i,j;double v;};
using Vec=std::vector<double>;
using Linear=std::vector<std::pair<int,double>>;
using Hessian=std::vector<Entry>;
static int checks=0;
static void require(bool pass,const char* message) {
  ++checks;if(!pass)throw std::runtime_error(message);
}
static void close(long double actual,long double wanted,const char* message) {
  require(std::abs(actual-wanted)<=64*std::numeric_limits<long double>::epsilon()*
      std::max(std::numeric_limits<long double>::min(),std::abs(wanted)),message);
}
int main()try {
  auto delta=[](const Vec& a,const Vec& b,const Linear& l,const Hessian& h,
      const Linear& ss=Linear{},double scale=1.) {
    return detail::coefficient_quadratic_increment(a,b,l,h,ss,scale);
  };
  const double next=std::nextafter(1.,2.);
  require(delta({1.},{next},{{0,1.}},{})==std::ldexp(1.L,-52),
      "actual adjacent binary64 linear displacement lost");
  require(delta({1.},{next},{{0,std::ldexp(1.,-20)}},{})==std::ldexp(1.L,-72),
      "sub-1e-20 coefficient increment lost");
  require(delta({1e16},{1e16+.1},{{0,1.}},{})==0,
      "nominal displacement used instead of actual rounded point");
  close(delta({1.,2.},{next,2.}, {},{{0,1,3.}}),6*std::ldexp(1.L,-52),
      "off-diagonal Hessian convention differs");
  close(delta({1.},{next},{},{{0,0,2.}}),
      std::ldexp(1.L,-52)*(2+std::ldexp(1.L,-52)),"diagonal secant differs");
  require(delta({0.,0.,0.},{1.,1.,1.},{{0,1e30},{1,1.},{2,-1e30}},{})==1,
      "compensated coefficient cancellation lost");
  close(delta({-2.},{-1.},{},{},{{0,3.}}),9,"negative signed square branch");
  close(delta({-2.},{1.},{},{},{{0,3.}}),15,"zero-crossing signed square branch");
  close(delta({2.},{-1.},{},{},{{0,3.}}),-15,"reverse zero-crossing signed square branch");
  close(delta({0.},{-1.},{},{},{{0,3.}}),-3,"signed square zero anchor");
  close(delta({1.,2.},{2.,3.},{{0,2.}},{{0,1,4.}},{},.125),2.25L,
      "stored scale or symmetric off-diagonal convention");
  bool rejected=false;try{delta({1.},{1.,2.},{{0,1.}},{});}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"mismatched point dimension accepted");
  rejected=false;try{delta({1.},{INFINITY},{{0,1.}},{});}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"nonfinite trial accepted");

  // A source g=-1e12+x with slack=1e12 has a nonzero true change
  // hidden by rounded g evaluations. The large multiplier makes its lost
  // contribution decisive without changing Armijo or any feasibility gate.
  namespace gr=detail::gr;
  gr::State a;a.c={0};a.y={1e8};a.s={1e12};
  gr::State b=a;
  gr::Parameters p;p.mu=.1;p.rho=.1;p.center={1e8};
  const long double dc=-std::ldexp(1.L,-30);
  const auto old=gr::change(a,b,p,0);
  const auto improved=gr::change_with_constraint_delta(a,b,p,0,{dc});
  require(old.valid&&old.value==0,"rounded replay path changed");
  close(improved.value,1e8L*dc+dc*dc/static_cast<long double>(p.rho),
      "coefficient C change not used by all merit components");
  gr::Result slope;slope.valid=true;slope.value=1e8L*dc;
  require(!gr::armijo(old,slope,1,1e-4)&&gr::armijo(improved,slope,1,1e-4),
      "stable source increment did not preserve the unchanged Armijo rule");
  require(improved.true_c_inf==0,"coefficient increment silently changed observed feasibility");
  require(!gr::change_with_constraint_delta(a,b,p,0,{}).valid,"bad delta dimension accepted");
  require(!gr::change_with_constraint_delta(a,b,p,0,{INFINITY}).valid,"nonfinite source delta accepted");
  require(!gr::change_with_barrier_delta(a,b,p,0,{NAN}).valid,"NaN source delta accepted");
  a.b={-.2};b.b={.1};p.kappa=1.2;
  const auto old_bound=gr::change(a,b,p,0);
  const auto new_bound=gr::change_with_constraint_delta(a,b,p,0,{0});
  require(old_bound.value==new_bound.value&&old_bound.hard_bounds==new_bound.hard_bounds,
      "zero source change modified hard-bound penalty");
  // Actual source slack movement must be included, even if it differs from
  // a nominal affine ds. This helper is also used by the production caller.
  detail::CoefficientIncrementSum source;
  source.add(delta({1.},{next},{{0,1.}},{}));
  source.add(static_cast<long double>(next)-1.);
  require(source.get()==std::ldexp(1.L,-51),"actual staged slack increment lost");
  std::cout<<"coefficient_constraint_increment PASS checks="<<checks<<"\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
