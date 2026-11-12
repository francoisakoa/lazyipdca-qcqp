#pragma once
#include "rank_structural_retry.hpp"

namespace lazyipdca::qcqp::detail {
// Native Lazy recovery of an otherwise terminal numerical failure. No data,
// start, profile, backend, function correction or final tolerance is changed.
// A historical success, budget exhaustion, infeasible point, affine equality
// model, or already nested/used retry never acquires this recovery.
inline bool majorant_barrier_retry_eligible(const QCQPProblem& problem,
    const Settings& st,const Result& primary) {
  if(st.majorant_barrier_retry_consumed||st.majorant_complete_barrier_gate||
      st.outer_algorithm!="lazy-dca"||st.dca_engine!="majorants"||
      st.rank_presolve_context||!st.dca_endpoint_start_file.empty()||
      problem.has_signed_square()||primary.x.size()!=std::size_t(problem.n)||
      !std::all_of(primary.x.begin(),primary.x.end(),[](double v){return std::isfinite(v);})||
      !primary.termination_audit_valid||!std::isfinite(primary.feasibility.total_inf)||
      primary.feasibility.total_inf>st.feasibility_tolerance)return false;
  if(primary.status!=Status::InnerFailure&&primary.status!=Status::FactorizationFailure&&
      primary.status!=Status::LineSearchFailure)return false;
  return std::any_of(problem.equalities.begin(),problem.equalities.end(),[](const auto& q) {
    return std::any_of(q.hessian.entries.begin(),q.hessian.entries.end(),
        [](const auto& a){return a.v!=0;});
  });
}
inline RankStructuralRetryBudget majorant_barrier_retry_budget(const QCQPProblem& problem,
    const Settings& st,const Result& primary,double elapsed,double started_absolute) {
  auto budget=rank_structural_retry_budget(st,primary,elapsed,started_absolute);
  budget.allowed=budget.allowed&&majorant_barrier_retry_eligible(problem,st,primary);
  return budget;
}
inline bool majorant_barrier_retry_strict(const Settings& st,const Result& retry,
    const RankStructuralRetryBudget& budget,double total_seconds) {
  const double target=std::min(st.stationarity_tolerance,st.dual_infeasibility_tolerance);
  return budget.allowed&&retry.status==Status::Solved&&retry.termination_audit_valid&&
      retry.outer_iterations>=0&&retry.outer_iterations<=budget.outer&&
      retry.restoration_iterations>=0&&retry.restoration_iterations<=budget.restoration&&
      std::isfinite(total_seconds)&&total_seconds>=0&&total_seconds<=st.time_limit_seconds&&
      std::isfinite(retry.feasibility.total_inf)&&retry.feasibility.total_inf<=st.feasibility_tolerance&&
      std::isfinite(retry.dual_infeasibility_inf)&&retry.dual_infeasibility_inf<=target&&
      std::isfinite(retry.complementarity_inf)&&retry.complementarity_inf<=st.complementarity_tolerance;
}
} // namespace lazyipdca::qcqp::detail
