#include "../src/accurate_product_sum.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
using lazyipdca::qcqp::detail::AccurateProductSum;
using boost::multiprecision::cpp_rational;
namespace {
void check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
// All finite doubles are binary rationals, so this oracle has no floating
// product or summation rounding and shares no implementation with the audit.
void fixture(const std::vector<std::pair<double,double>>& products,
             long double tolerance) {
  AccurateProductSum sum;cpp_rational exact=0;
  for(auto [a,b]:products){sum.add_product(a,b);exact+=cpp_rational(a)*cpp_rational(b);}
  const long double oracle=exact.convert_to<long double>();
  if(std::abs(sum.value()-oracle)>tolerance) {
    std::cerr<<std::setprecision(25)<<"actual="<<sum.value()<<" rational="<<oracle
      <<" tolerance="<<tolerance<<'\n';
    check(false,"compensated audit differs from exact rational oracle");
  }
}
}
int main() {
  try {
    fixture({{1e18,1},{1e-8,1},{-1e18,1}},1e-25L);
    fixture({{1e16,1+std::ldexp(1.,-52)},{-1e16,1}},1e-18L);
    fixture({{1e14,1.23456789012345},{2e-8,1},{-1e14,1.23456789012345}},1e-25L);
    // The compensated sum remains a finite precision calculation. Its
    // absolute error on this1e40 dynamic range is far below the Newton gate.
    fixture({{1e-10,1e-10},{1e30,1e-10},{-1e30,1e-10}},1e-19L);
    // A real violation must remain rejected after huge cancellation. In
    // particular the new arithmetic cannot reinterpret a backend berr or
    // change the existing absolute expanded direction threshold of1e-7.
    AccurateProductSum invalid;invalid.add_product(1e18,1);
    invalid.add_product(2e-7,1);invalid.add_product(-1e18,1);
    check(std::abs(invalid.value())>1e-7L,"true expanded residual was concealed");
    AccurateProductSum valid;valid.add_product(1e18,1);
    valid.add_product(2e-8,1);valid.add_product(-1e18,1);
    check(std::abs(valid.value())<1e-7L,"valid expanded residual was spuriously rejected");
    std::cout<<"PASS accurate product sums against exact rational oracle; strict rejection retained\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
