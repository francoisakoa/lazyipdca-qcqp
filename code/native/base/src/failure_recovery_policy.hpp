#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <string>

namespace lazyipdca::qcqp::detail {
// Admission estimate for the existing dense residual models / legacy normal
// matrices, NOT a bound on process RSS or a sparse factor's fill. Long double
// arithmetic avoids size_t overflow before deciding whether to allocate.
// Legacy callers admit one late recovery. Hybrid stages may reopen only
// after original-filtered decade progress; each entry retains a 60 s deadline.
struct FailureRecoveryBudget {
  long double estimated_workspace_bytes=0;
  static constexpr long double limit_bytes=512.0L*1024*1024;
  bool allowed=false;
};
inline FailureRecoveryBudget failure_recovery_budget(int n,std::size_t eq,std::size_t ineq,
                                                    bool admit_inequality_only=false) {
  FailureRecoveryBudget out;
  if(n<=0||(eq==0&&(!admit_inequality_only||ineq==0)))return out;
  const long double N=n,M=static_cast<long double>(eq)+static_cast<long double>(ineq);
  out.estimated_workspace_bytes=sizeof(double)*(4*N*M+4*N*N+64*N+8*M);
  out.allowed=std::isfinite(out.estimated_workspace_bytes)&&
      out.estimated_workspace_bytes<=out.limit_bytes;
  return out;
}
inline FailureRecoveryBudget failure_recovery_budget(const QCQPProblem& p,bool admit_inequality_only=false) {
  auto out=failure_recovery_budget(p.n,p.equalities.size(),p.inequalities.size(),admit_inequality_only);
  out.allowed=out.allowed&&!p.has_signed_square();
  return out;
}
// A failed linearization can leave only affine rows infeasible after curved
// rows have been restored. Curvature therefore cannot determine whether the
// ORIGINAL anchor needs recovery. This only admits a bounded private attempt;
// callers retain their storage/deadline budgets and original-filter commit.
inline bool failed_linearization_recovery_eligible(bool numerical_failure,
    bool filter_failure,bool inner_model_failure,bool restoration_enabled,
    bool already_attempted,double original_primal,double final_tolerance,
    bool repair_small_infeasibility=true) {
  return (numerical_failure||filter_failure||inner_model_failure)&&
      restoration_enabled&&!already_attempted&&
      std::isfinite(original_primal)&&original_primal>=0&&
      std::isfinite(final_tolerance)&&final_tolerance>0&&
      // Exhausting a direction, original filter or model leaves no usable
      // next step. The interval (tol,100*tol] is still ORIGINAL-infeasible;
      // excluding it strands a small residual while proximal retries cannot
      // alter the affine primal equations. Admit the same bounded private
      // Phase I there. Its unchanged original filter and final KKT audit
      // remain mandatory, so this changes permission rather than acceptance.
      original_primal>((repair_small_infeasibility||numerical_failure)?1:100)*final_tolerance;
}
// One historical barrier-continuation update after an ORIGINAL-filter
// acceptance. Recovery anchors use exactly the same residual gate and bounded
// factor as DCA anchors; this is not an extra inner barrier solve or a KKT test.
inline double majorant_barrier_continuation(double mu,double floor,double residual) {
  if(!std::isfinite(mu)||!std::isfinite(floor)||!std::isfinite(residual)||
      floor<0||residual<0||!(mu>floor)||residual>10*mu)return mu;
  return std::max(floor,0.2*mu);
}
// This is permission to rebuild a DIFFERENT auxiliary proximal matrix after
// an expanded direction has been rejected. It never authorizes that direction
// or changes a function correction. Three charged outer retries per anchor.
inline bool failed_direction_proximal_retry_eligible(const std::string& failure,
    int used,bool atomic_regularization,int remaining_outer,double elapsed,double limit) {
  const bool numerical=failure.find("expanded Newton residual rejected")!=std::string::npos||
      failure.find("backward error")!=std::string::npos||failure.find("nonfinite")!=std::string::npos;
  return numerical&&used>=0&&used<3&&!atomic_regularization&&remaining_outer>1&&
      std::isfinite(elapsed)&&elapsed>=0&&std::isfinite(limit)&&elapsed<limit;
}
// Transactional balanced auxiliary state at a FIXED original primal point.
// Increasing a positive slack along its geometric path makes log(s) affine;
// -mu*log_gain is the exact original barrier slope/difference at alpha=1.
// The caller must still check every original row and the UNCHANGED filter.
// No bound, function coefficient or primal coordinate is added or changed.
template<class Active>
inline bool direction_centering_candidate(const Vec& original_values,const Active& active,
    const Vec& old_s,const Vec& old_z,double mu,Vec& new_s,Vec& new_z,long double& log_gain) {
  if(original_values.size()!=active.size()||old_s.size()!=active.size()||old_z.size()!=active.size()||
      !std::isfinite(mu)||!(mu>0))return false;
  Vec s=old_s,z=old_z;long double gain=0;
  const double balanced=std::sqrt(mu);
  for(std::size_t i=0;i<active.size();++i)if(active[i]) {
    if(!std::isfinite(original_values[i])||!std::isfinite(old_s[i])||!(old_s[i]>0)||
        !std::isfinite(old_z[i])||!(old_z[i]>0))return false;
    s[i]=std::max({old_s[i],-original_values[i],balanced});z[i]=mu/s[i];
    if(!std::isfinite(s[i])||!std::isfinite(z[i])||!(s[i]>0)||!(z[i]>0))return false;
    gain+=std::log(static_cast<long double>(s[i]))-std::log(static_cast<long double>(old_s[i]));
  }
  if(!std::isfinite(gain)||!(gain>0)||!std::isfinite(-static_cast<long double>(mu)*gain))return false;
  new_s.swap(s);new_z.swap(z);log_gain=gain;return true;
}
// One shared late Phase-I allowance per physical solve. Startup restoration
// remains separate. Neither a new DCA anchor nor a finer private stage resets it.
struct SharedPhase1RecoveryBudget {
  int limit=0,used=0;bool started=false;
  double deadline=std::numeric_limits<double>::infinity();
  explicit SharedPhase1RecoveryBudget(int count):limit(std::max(0,count)){}
  int remaining()const{return limit-used;}
  void charge(int count) {
    if(count<0||count>remaining())throw std::runtime_error("late phase-I shared iteration budget exceeded");
    used+=count;
  }
  double bind_deadline(double proposed) {
    if(!std::isfinite(proposed))throw std::invalid_argument("invalid late phase-I deadline");
    deadline=started?std::min(deadline,proposed):proposed;started=true;return deadline;
  }
};
inline bool phase1_reopen_after_dca(bool hybrid,bool model_certified,
    bool original_filter_accepted,double actual_primal_displacement) {
  return hybrid&&model_certified&&original_filter_accepted&&
      std::isfinite(actual_primal_displacement)&&actual_primal_displacement>0;
}
// Reopening eligibility does not erase a verified Phase-I contraction.
// Accepted DCA steps may increase feasibility while reducing the objective;
// later recovery must still beat the best previously accepted Phase-I point.
inline double phase1_best_reference(double previous,double accepted_primal) {
  if(std::isnan(previous)||previous<0||!std::isfinite(accepted_primal)||accepted_primal<0)
    throw std::invalid_argument("invalid best phase-I reference");
  return std::min(previous,accepted_primal);
}
// A feasibility-restoration target is not a KKT tolerance. A successful
// phase-I reduction supplies only a new original-QCQP anchor; the controller
// must independently evaluate every original constraint and its own filter.
inline double phase1_feasibility_target(double before,double final_tolerance) {
  if(!std::isfinite(before)||before<0||!std::isfinite(final_tolerance)||!(final_tolerance>0))
    throw std::invalid_argument("invalid phase-I feasibility target");
  return std::max(final_tolerance,0.1*before);
}
// Relative tiny-slope eligibility is needed only after the original filter
// has required private restoration precision below the fixed final tolerance.
// In coarse phases the historical absolute slope gate avoids unproductive
// microscopic moves. This switch changes no actual merit line-search test.
inline bool phase1_private_precision_context(bool hybrid,double target,double final_tolerance) {
  return hybrid&&std::isfinite(target)&&target>0&&std::isfinite(final_tolerance)&&
      final_tolerance>0&&target<final_tolerance;
}
// Once an improving private Phase-I point is rejected by the ORIGINAL
// filter, its next precision target is independent of the final KKT tolerance.
// Zero means there is no strictly smaller representable positive target.
inline double phase1_private_precision_target(double previous_target,double primal) {
  if(!std::isfinite(previous_target)||!(previous_target>0)||
     !std::isfinite(primal)||primal<0)
    throw std::invalid_argument("invalid private phase-I precision target");
  if(primal==0)return 0;
  const double target=std::max(std::numeric_limits<double>::denorm_min(),
      0.1*std::min(previous_target,primal));
  return std::isfinite(target)&&target>0&&target<previous_target&&target<primal?target:0;
}
inline bool phase1_verified_reduction(double before,double after,double final_tolerance) {
  return std::isfinite(after)&&after>=0&&after<before&&
      after<=phase1_feasibility_target(before,final_tolerance);
}
// Relative descent of an already computed private restoration direction.
// Normalize each finite vector before long-double accumulation so even
// extreme objective units cannot overflow the angle test. This is exactly
// g'd < -1e-14 ||g|| ||d|| after cancelling the two positive scale factors.
inline bool failed_linearization_relative_descent(const Vec& gradient,const Vec& direction) {
  if(gradient.size()!=direction.size()||gradient.empty())return false;
  double gs=0,ds=0;
  for(std::size_t j=0;j<gradient.size();++j) {
    if(!std::isfinite(gradient[j])||!std::isfinite(direction[j]))return false;
    gs=std::max(gs,std::abs(gradient[j]));ds=std::max(ds,std::abs(direction[j]));
  }
  if(!(gs>0&&ds>0))return false;
  long double gd=0,gg=0,dd=0;
  for(std::size_t j=0;j<gradient.size();++j) {
    const long double g=static_cast<long double>(gradient[j])/gs,
        d=static_cast<long double>(direction[j])/ds;
    gd+=g*d;gg+=g*g;dd+=d*d;
  }
  return std::isfinite(gd)&&std::isfinite(gg)&&std::isfinite(dd)&&gg>0&&dd>0&&
      gd < -1e-14L*std::sqrt(gg)*std::sqrt(dd);
}
// Check both clocks using ONE fresh sample immediately before a private
// stage or an original-anchor commit. A solver success cannot waive either
// deadline, including a final factorization which completes after expiry.
inline bool phase1_deadline_open(double elapsed,double global_limit,double now,double deadline) {
  return std::isfinite(elapsed)&&elapsed>=0&&elapsed<global_limit&&
      std::isfinite(now)&&now<deadline;
}
// Count weak contraction independently of alpha: a microscopic accepted
// step is evidence of stagnation, not a reason to erase its history. This
// policy only enables a subsequently audited, filter-tested direction probe.
inline int weak_dual_contraction_streak(int old,double before,double after,double tolerance) {
  if(!std::isfinite(before)||!std::isfinite(after)||!(tolerance>0)||
      after<=10*tolerance||after<0.9*before)return 0;
  return std::min(12,std::max(0,old)+1);
}
// Each controller owns its retry history. A proximal change does not
// establish a contraction of the original stationarity and must not erase
// the accumulated evidence needed by the independently audited curvature
// probe. A genuine contraction resets both histories through record().
struct DualRecoveryHistory {
  int curvature_steps=0;
  int proximal_steps=0;
  void record(double before,double after,double tolerance) {
    curvature_steps=weak_dual_contraction_streak(curvature_steps,before,after,tolerance);
    proximal_steps=weak_dual_contraction_streak(proximal_steps,before,after,tolerance);
  }
  void reset() { curvature_steps=proximal_steps=0; }
  void proximal_proposed() { proximal_steps=0; }
  void curvature_probed() { reset(); }
};
// Explicit proximal control after a genuinely stagnant accepted small step.
// This computes a proposal only; it neither edits the point/filter nor accepts
// an unaudited direction. Invalid controls and capped proposals are no-ops.
// The stored proposal can lie below the effective frozen model diagonal
// after a certified floor or backend recovery. A retry must exceed BOTH;
// growing memory alone can otherwise repeat an identical subproblem.
inline double proximal_retry_base(double remembered,double actual) {
  if(!std::isfinite(remembered)||remembered<0||!std::isfinite(actual)||actual<0)
    return std::numeric_limits<double>::quiet_NaN();
  return std::max(remembered,actual);
}
// A filter failure at an ORIGINAL-feasible anchor does not establish that
// greater proximal damping will help. Offer one downward matrix probe per
// actual primal anchor. Its factorization, expanded equations and ORIGINAL
// filter must all pass; this proposal is no convexity/overdamping certificate.
inline bool lower_proximal_probe_eligible(bool used,bool square_feasibility,
    bool atomic_regularization,double primal,double feasibility_tolerance,
    int remaining_outer,double elapsed,double limit) {
  return !used&&!square_feasibility&&!atomic_regularization&&remaining_outer>1&&
      std::isfinite(primal)&&primal>=0&&std::isfinite(feasibility_tolerance)&&
      feasibility_tolerance>0&&primal<=feasibility_tolerance&&
      std::isfinite(elapsed)&&elapsed>=0&&std::isfinite(limit)&&elapsed<limit;
}
inline double lower_proximal_proposal(double actual,double certified_floor) {
  if(!std::isfinite(actual)||actual<0||!std::isfinite(certified_floor)||certified_floor<0||
      !(actual>certified_floor))return actual;
  const double proposal=std::max(certified_floor,std::min(1.,.1*actual));
  return proposal<actual?proposal:actual;
}
// A rejected optional lower matrix is not evidence that the ORIGINAL
// matrix failed. Return true so numerical callers rebuild the saved matrix
// before they consider any unrelated restoration/increase policy.
inline bool restore_lower_proximal_probe(bool& pending,double previous_seed,double& memory) {
  if(!pending)return false;
  if(!std::isfinite(previous_seed)||previous_seed<0)
    throw std::invalid_argument("invalid saved lower-proximal seed");
  memory=previous_seed;pending=false;return true;
}
inline double small_step_proximal_proposal(double rho,double gradient_inf,
                                         double x_inf,double initial) {
  if(!std::isfinite(rho)||rho<0||!std::isfinite(gradient_inf)||gradient_inf<0||
     !std::isfinite(x_inf)||x_inf<0||!std::isfinite(initial)||initial<=0)return rho;
  const double next=std::max({initial,10*rho,gradient_inf/std::max(1.0,x_inf)});
  return std::isfinite(next)&&next>rho&&next<=1e12?next:rho;
}
// A fully certified, objective-filter-accepted hybrid step is evidence that
// the next model can try less EXPLICIT proximal damping. Original KKT norms
// need not be monotone, particularly after multiplier refits. This is only a
// next-model proposal; its margin floor and all acceptance gates are rebuilt.
inline bool hybrid_full_objective_proximal_decay(bool hybrid,bool model_relative,
    bool model_certified,bool original_filter_accepted,bool objective_condition,
    double alpha,double tau,double certified_floor) {
  return hybrid&&model_relative&&model_certified&&original_filter_accepted&&
      objective_condition&&std::isfinite(alpha)&&alpha==1.0&&
      std::isfinite(tau)&&std::isfinite(certified_floor)&&certified_floor>=0&&
      tau>certified_floor;
}
inline bool small_step_proximal_trigger(bool eligible,double alpha,int weak_steps,
                                        double original_primal,double feasibility_tolerance) {
  // A weak dual contraction during nonlinear feasibility recovery is not
  // evidence of a stalled feasible optimization step. In that regime the
  // filter may legitimately accept short steps that improve feasibility.
  // Escalate this dual-stall metric only at an originally feasible point;
  // genuine filter exhaustion retains its separate bounded retry policy.
  return eligible&&std::isfinite(original_primal)&&original_primal>=0&&
      std::isfinite(feasibility_tolerance)&&feasibility_tolerance>0&&
      original_primal<=feasibility_tolerance&&std::isfinite(alpha)&&
      alpha>0&&alpha<1e-2&&weak_steps>=3;
}

}
