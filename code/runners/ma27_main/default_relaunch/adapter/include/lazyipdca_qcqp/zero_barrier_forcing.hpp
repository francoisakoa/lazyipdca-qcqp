#pragma once
#include <algorithm>
#include <cmath>

namespace lazyipdca::qcqp::detail {
// Pure equalities have no positive barrier parameter to provide a decreasing
// inner accuracy. Use a geometric forcing sequence at accepted stabilization
// epochs. This is never an original KKT stopping tolerance. Positive-barrier
// systems continue to use their existing 10*mu gate.
inline double zero_barrier_epoch_forcing(double reference,int epochs,double terminal) {
  if(!std::isfinite(terminal)||terminal<=0)return 0;
  if(!std::isfinite(reference)||reference<0||epochs<0)return terminal;
  return std::max(terminal,reference*std::pow(0.2,epochs));
}
}
