#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
namespace lazyipdca::qcqp {
// One common backtracking parameter, independently feasible primal/dual
// fractions. A small dual fraction must not force the same tiny primal step.
inline std::pair<double,double> hybrid_split_fractions(double primal,double dual,double beta) {
 if(!std::isfinite(primal)||!std::isfinite(dual)||!std::isfinite(beta)||
     primal<0||primal>1||dual<0||dual>1||beta<0||beta>1)
  throw std::invalid_argument("invalid primal-dual backtracking fractions");
 return {primal*beta,dual*beta};
}
}
