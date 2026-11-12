#pragma once
#include <algorithm>
#include <stdexcept>
namespace lazyipdca::qcqp::detail {
// One remaining allowance must cover every still-authorized direction. The
// ceiling division avoids starving the final directions; no call can consume
// more than the remaining total or the historical per-restart cap.
inline int shared_restart_allowance(int remaining,int directions,int per_restart_cap) {
  if(remaining<0||directions<=0||per_restart_cap<0)
    throw std::invalid_argument("invalid shared restoration restart allowance");
  if(remaining==0||per_restart_cap==0)return 0;
  const int share=remaining/directions+(remaining%directions!=0);
  return std::min(per_restart_cap,share);
}
// A private Phase-I anchor may contract by decades. Global exploration keeps
// the scale of the frozen supplied start, while local radii still use the
// current best point. Radius caps are applied by the existing caller.
inline double phase1_global_restart_reference(double current,double frozen_start) {
  return std::max(current,frozen_start);
}
}
