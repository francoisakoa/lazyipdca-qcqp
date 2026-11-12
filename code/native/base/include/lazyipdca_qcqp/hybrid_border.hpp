#pragma once
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lazyipdca::qcqp::detail {
// Append one exact algebraic low-rank border. The Schur complement of the
// independent -1 pivot is +coefficient * u u'. Only nonzero entries of u are
// stored: no explicit quadratic outer product is inserted into the base KKT.
// Numerical regularization belongs in the supplied base block, not in this
// mathematical coefficient. Backend residual checks remain independent.
inline std::size_t append_hybrid_border(int primal_dimension,
    const Vec& projected,double coefficient,int& dimension,
    std::vector<SymmetricTriplet>& matrix,Vec& values) {
  if(primal_dimension<0 || projected.size()!=std::size_t(primal_dimension) ||
     dimension<primal_dimension || matrix.size()!=values.size() ||
     !std::isfinite(coefficient) || coefficient<0)
    throw std::invalid_argument("invalid hybrid border dimensions or positive coefficient");
  bool nonzero=false;
  for(double value:projected) {
    if(!std::isfinite(value))throw std::invalid_argument("nonfinite projected hybrid atom");
    nonzero=nonzero||value!=0;
  }
  if(coefficient==0 || !nonzero)return 0;
  if(dimension==std::numeric_limits<int>::max())throw std::overflow_error("hybrid border dimension overflow");
  const double weight=std::sqrt(coefficient);const int row=dimension;
  std::vector<SymmetricTriplet> added;Vec coefficients;
  for(int j=0;j<primal_dimension;++j)if(projected[j]!=0) {
    const double value=weight*projected[j];
    if(!std::isfinite(value))throw std::overflow_error("hybrid border coefficient overflow");
    added.push_back({j,row,0});coefficients.push_back(value);
  }
  added.push_back({row,row,0});coefficients.push_back(-1);
  matrix.insert(matrix.end(),added.begin(),added.end());
  values.insert(values.end(),coefficients.begin(),coefficients.end());++dimension;
  return added.size();
}
} // namespace lazyipdca::qcqp::detail
