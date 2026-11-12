#pragma once
#include <algorithm>

namespace lazyipdca::qcqp::detail {
// A remaining allowance is permission for FUTURE work. Never clamp the actual
// counters when charging attempted work: an upstream overrun must stay visible.
inline int remaining_shared_restoration(int limit,int consumed) {
  if(limit<=0||consumed<0||consumed>=limit)return 0;
  return limit-consumed;
}

// The affine startup is an explicit restoration phase and also consumes outer
// work. Leave an outer step for the original-model continuation. Historical
// callers without a shared restoration contract keep their existing allowance.
inline int affine_start_iteration_allowance(bool shared,int restoration_limit,int outer_limit) {
  if(outer_limit<=1)return 0;
  const int outer_allowance=std::min(200,outer_limit-1);
  return shared?std::min(outer_allowance,std::max(0,restoration_limit)):outer_allowance;
}

inline bool restoration_step_available(bool shared,int limit,int consumed) {
  return !shared||remaining_shared_restoration(limit,consumed)>0;
}

// The existing terminal certificate trial is work in the SAME restoration
// budget. A known original-feasible candidate must not lose its one trial
// solely because a failed primary phase spent every available iteration.
inline int primary_restoration_allowance(bool shared,int limit,bool reserve_terminal) {
  if(!shared||!reserve_terminal)return limit;
  return std::max(0,limit-1);
}
} // namespace lazyipdca::qcqp::detail
