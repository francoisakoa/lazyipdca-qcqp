#pragma once
#include <algorithm>
#include <atomic>
namespace lazyipdca::qcqp::detail {
inline std::atomic<int>& mumps_memory_margin_cache() {
  static std::atomic<int> margin{20};
  return margin;
}
inline int learned_mumps_memory_margin() {
  return mumps_memory_margin_cache().load(std::memory_order_relaxed);
}
inline void remember_mumps_memory_margin(int margin) {
  margin=std::clamp(margin,20,6400);
  auto& cache=mumps_memory_margin_cache();
  int known=cache.load(std::memory_order_relaxed);
  while(known<margin&&!cache.compare_exchange_weak(
      known,margin,std::memory_order_relaxed,std::memory_order_relaxed)) {}
}
inline void reset_mumps_memory_margin_for_test(int margin=20) {
  mumps_memory_margin_cache().store(std::clamp(margin,20,6400),
                                    std::memory_order_relaxed);
}
inline bool mumps_reanalyze_after_memory_failure(bool enabled,bool reused,int code) {
  return enabled&&reused&&(code==-8||code==-9);
}
// Retry only numeric factorization. The symbolic state and matrix are unchanged.
// The historical margin is used at initialization. Retain increased workspace
// while the symbolic cache remains valid; allocation is not an acceptance rule.
template<class Integer, class Retry>
int recover_mumps_memory(Integer& code, Integer& margin, Retry retry, bool learn_successful_margin=true) {
  int attempts=0;
  while((code==-8 || code==-9) && margin<6400) {
    margin=std::min<Integer>(6400,std::max<Integer>(40,2*margin));
    retry();++attempts;
  }
  // A saturated unsuccessful ladder proves no useful allocation policy.
  // Do not poison future contexts with workspace that never factored.
  if(code>=0&&learn_successful_margin)remember_mumps_memory_margin(static_cast<int>(margin));
  return attempts;
}
}
