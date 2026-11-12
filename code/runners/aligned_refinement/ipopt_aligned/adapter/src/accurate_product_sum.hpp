#pragma once
#include <cmath>

namespace lazyipdca::qcqp::detail {
// The expanded Newton audit can contain individually huge multiplier terms
// whose exact sum is small. An ordinary long-double running sum still loses
// that small sum. Keep its summation error, and split every DOUBLE product
// into an exactly rounded product and its FMA remainder before accumulating.
// This changes the evaluation accuracy, never the audited equations or gate.
class AccurateProductSum {
  long double sum_=0,correction_=0;
  void add_scaled_product(double first,double second,long double factor) {
    const double product=first*second;
    if(std::isfinite(product)) {
      add(factor*static_cast<long double>(product));
      add(factor*static_cast<long double>(std::fma(first,second,-product)));
    } else {
      add(factor*static_cast<long double>(first)*second);
    }
  }
 public:
  AccurateProductSum()=default;
  explicit AccurateProductSum(double value):sum_(value){}
  void add(long double value) {
    const long double next=sum_+value;
    correction_+=std::abs(sum_)>=std::abs(value)?
        (sum_-next)+value:(value-next)+sum_;
    sum_=next;
  }
  void add_product(double first,double second) {
    const double product=first*second;
    if(std::isfinite(product)) {
      add(product);
      add(std::fma(first,second,-product));
    } else {
      // Long double may have a larger exponent range than double. Preserve
      // that range while callers continue their mandatory finite checks.
      add(static_cast<long double>(first)*second);
    }
  }
  // The quadratic callbacks need the unrounded product of three binary64
  // operands. Retain both parts of the first product, then split each second
  // product with FMA. The factor is an exact power of two for QCQP diagonals.
  void add_triple_product(double first,double second,double third,
                          long double factor=1.L) {
    const double product=first*second;
    if(std::isnormal(product)) {
      add_scaled_product(product,third,factor);
      add_scaled_product(std::fma(first,second,-product),third,factor);
    } else {
      // Preserve the extended exponent range for overflow or underflow in
      // an intermediate product. Callers retain their existing finite gate.
      add(factor*static_cast<long double>(first)*second*third);
    }
  }
  long double value() const {return sum_+correction_;}
};
}
