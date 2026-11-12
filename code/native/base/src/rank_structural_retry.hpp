#pragma once
#include "lazyipdca_qcqp/result.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <algorithm>
#include <cmath>

namespace lazyipdca::qcqp::detail {
// Structural recovery is a second solve of the SAME configuration, never a
// solver/profile fallback. Exhausted iteration/time budgets do not trigger it.
inline bool rank_structural_retry_status(Status status) {
  switch(status) {
    case Status::RestorationFailure:case Status::InnerFailure:
    case Status::FactorizationFailure:case Status::LineSearchFailure:
    case Status::TrustRegionFailure:case Status::NumericalError:return true;
    default:return false;
  }
}
struct RankStructuralRetryBudget {
  bool allowed=false;
  int outer=0,restoration=0;
  double seconds=0,restoration_deadline=INFINITY;
};
inline RankStructuralRetryBudget rank_structural_retry_budget(
    const Settings& st,const Result& primary,double elapsed,double started_absolute) {
  RankStructuralRetryBudget b;
  b.outer=std::max(0,st.max_outer_iterations-std::max(0,primary.outer_iterations));
  const int restoration_limit=std::max(0,std::min(st.max_outer_iterations,st.restoration_max_iterations));
  b.restoration=std::max(0,restoration_limit-std::max(0,primary.restoration_iterations));
  b.seconds=std::isfinite(st.time_limit_seconds)?std::max(0.0,st.time_limit_seconds-elapsed):INFINITY;
  b.restoration_deadline=st.restoration_deadline_seconds;
  // A restarted private Phase I must not acquire a fresh sixty-second window.
  // This conservative physical-start deadline can be earlier than the first
  // late Phase-I start, but never grants additional private restoration time.
  if(primary.restoration_iterations>0||primary.restoration_total_seconds>0)
    b.restoration_deadline=std::min(b.restoration_deadline,started_absolute+60.0);
  b.allowed=rank_structural_retry_status(primary.status)&&b.outer>0&&b.seconds>1e-6;
  return b;
}
} // namespace lazyipdca::qcqp::detail
