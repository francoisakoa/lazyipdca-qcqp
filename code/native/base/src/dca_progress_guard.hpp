#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace lazyipdca::qcqp::detail {

inline double dca_kkt_growth_ceiling(double from_kkt, double growth_max,
                                     double absolute_floor = 0.0) {
  // Never hide a relative KKT explosion behind a unit-size floor.  The old
  // max(1, growth*from) admitted 4.4e-3 -> 8.0e-1 on case30000.  Callers may
  // supply a small, tolerance-derived floor to avoid decisions on roundoff.
  return std::max(std::max(0.0, absolute_floor), growth_max * from_kkt);
}

inline bool dca_kkt_growth_allowed(double from_kkt, double to_kkt,
                                   double growth_max,
                                   double absolute_floor = 0.0) {
  return std::isfinite(to_kkt) &&
         to_kkt <= dca_kkt_growth_ceiling(from_kkt, growth_max,
                                          absolute_floor);
}

inline bool dca_local_guard_active(double kkt, double primal,
                                   double kkt_trigger,
                                   double primal_trigger) {
  return std::isfinite(kkt) && std::isfinite(primal) &&
         kkt <= kkt_trigger && primal <= primal_trigger;
}

inline double dca_inherit_kkt_reference(double from_kkt,
                                        double from_reference,
                                        double from_primal,
                                        double to_kkt,
                                        double kkt_trigger = 1.0,
                                        double primal_trigger = 0.1) {
  if (!dca_local_guard_active(from_kkt, from_primal,
                              kkt_trigger, primal_trigger))
    return to_kkt;
  const double reference = std::isfinite(from_reference)
                               ? std::min(from_kkt, from_reference)
                               : from_kkt;
  return std::min(reference, to_kkt);
}

inline double dca_inherit_phase_kkt_reference(
    double from_kkt, double from_reference, double from_primal,
    double to_kkt, double to_primal, double feasibility_tolerance,
    bool feasible_reference_initialized = false,
    double kkt_trigger = 1.0, double primal_trigger = 0.1) {
  // A KKT low-water mark obtained before primal certification must not become
  // the stationarity target for the feasible phase.  At the crossing, start a
  // new low-water history from the first feasible point.  Thereafter the
  // ordinary persistent local reference applies again.
  if (!feasible_reference_initialized && std::isfinite(to_primal) &&
      to_primal <= feasibility_tolerance)
    return to_kkt;
  return dca_inherit_kkt_reference(from_kkt, from_reference, from_primal,
                                    to_kkt, kkt_trigger, primal_trigger);
}

inline bool dca_private_continuation_allowed(double kkt, double primal,
                                             double kkt_trigger = 1.0,
                                             double primal_trigger = 0.1) {
  // Private model iterates are useful globally, but once the original
  // residual is local they can consume every inner/rebuild slot without
  // producing an admissible original endpoint.
  return !dca_local_guard_active(kkt, primal, kkt_trigger, primal_trigger);
}

inline bool dca_local_candidate_not_dominated(double from_kkt,
                                              double reference_kkt,
                                              double from_primal,
                                              double to_kkt,
                                              double to_primal,
                                              double kkt_trigger = 1.0,
                                              double primal_trigger = 0.1) {
  if (!dca_local_guard_active(from_kkt, from_primal,
                              kkt_trigger, primal_trigger))
    return true;
  const double reference = std::isfinite(reference_kkt)
                               ? std::min(from_kkt, reference_kkt)
                               : from_kkt;
  // A local candidate may trade stationarity for better feasibility, or
  // feasibility for a new KKT low-water mark.  It may not worsen both.
  return to_kkt <= reference || to_primal <= from_primal;
}

// Keep the last certified curvature shift while feasibility is still being
// built. Once the original primal residual is below tolerance, stationarity
// is the remaining task: probe the exact KKT first, then recover through the
// previous certified shift if its inertia is not valid.
inline double dca_aggregate_initial_rho(double previous_rho, double primal,
                                        double feasibility_tolerance,
                                        bool local_guard) {
  if (std::isfinite(primal) && primal <= feasibility_tolerance) return 0.0;
  return local_guard ? std::max(0.0, previous_rho) : 0.0;
}

inline double dca_aggregate_next_rho(double rho, double previous_rho,
                                     bool recover_previous_certificate) {
  previous_rho = std::max(0.0, previous_rho);
  if (rho == 0.0) return std::max(1e-6, 0.2 * previous_rho);
  if (recover_previous_certificate && rho < previous_rho)
    return std::min(previous_rho, 10.0 * rho);
  return 10.0 * rho;
}

// Preserve the old proposals first. Before abandonment, insert the previous
// certified value (only if within the CURRENT cap) and the cap itself. Every
// proposal is freshly factorized; remembered values carry no stale certificate.
inline std::vector<double> dca_complete_inertia_schedule(
    double initial,double previous,double cap,bool recover_previous,int budget=10) {
  std::vector<double> out;
  auto add=[&](double value){
    if(std::isfinite(value)&&value>=0&&value<=cap&&
       std::find(out.begin(),out.end(),value)==out.end())out.push_back(value);
  };
  double rho=initial;
  for(int k=0;k<budget&&std::isfinite(rho)&&rho<=cap;++k){
    add(rho);
    const double next=dca_aggregate_next_rho(rho,previous,recover_previous);
    if(!(next>rho))break;
    rho=next;
  }
  add(previous);add(cap);
  return out;
}

inline bool dca_rho_probe_candidate_allowed(double current_kkt, double to_kkt,
                                            double to_primal,
                                            double feasibility_tolerance) {
  // Compare with the current feasible iterate, not with a low-water mark from
  // an earlier (possibly infeasible) phase.  The persistent KKT guard and the
  // original filter are still applied by the caller after this prefilter.
  return std::isfinite(current_kkt) && std::isfinite(to_kkt) &&
         std::isfinite(to_primal) && to_primal <= feasibility_tolerance &&
         to_kkt < current_kkt;
}

inline bool dca_feasible_kkt_recovery_allowed(
    double current_kkt, double to_kkt, double current_primal,
    double to_primal, double feasibility_tolerance,
    double required_contraction = 0.9) {
  // The persistent low-water mark limits nonmonotone growth, but it must not
  // prevent recovery after such a step has already been accepted.  On
  // case30000 R7D8, 1.8e-3 -> 1.48e-2 was legal under the x10 watchdog, then
  // a certified feasible 2.3e-3 aggregate point was incorrectly rejected
  // only because it did not immediately beat the old 1.8e-3 record.
  //
  // Require a material contraction of the CURRENT KKT and preserve primal
  // feasibility on both sides.  This is an additional recovery route; the
  // persistent growth ceiling and the original filter remain mandatory.
  return std::isfinite(current_kkt) && current_kkt > 0.0 &&
         std::isfinite(to_kkt) && std::isfinite(current_primal) &&
         std::isfinite(to_primal) &&
         current_primal <= feasibility_tolerance &&
         to_primal <= feasibility_tolerance &&
         required_contraction > 0.0 && required_contraction < 1.0 &&
         to_kkt <= required_contraction * current_kkt;
}

inline std::vector<double> dca_rho_probe_schedule(double anchor_rho,
                                                   double rho_max) {
  anchor_rho = std::max(0.0, anchor_rho);
  std::vector<double> rhos = {0.0};
  if (anchor_rho > 0.0) {
    const double recovery=std::max(1e-6, 0.2 * anchor_rho);
    if (recovery <= rho_max) rhos.push_back(recovery);
    if (0.5 * anchor_rho <= rho_max) rhos.push_back(0.5 * anchor_rho);
    if (anchor_rho <= rho_max) rhos.push_back(anchor_rho);
    if (2.0 * anchor_rho <= rho_max) rhos.push_back(2.0 * anchor_rho);
    if (10.0 * anchor_rho <= rho_max) rhos.push_back(10.0 * anchor_rho);
  }
  std::sort(rhos.begin(), rhos.end());
  rhos.erase(std::unique(rhos.begin(), rhos.end()), rhos.end());
  return rhos;
}

inline bool dca_rho_candidate_better(double candidate_kkt,
                                     double candidate_primal,
                                     double candidate_rho,
                                     double best_kkt,double best_primal,
                                     double best_rho,double anchor_rho) {
  if (!std::isfinite(best_kkt)) return true;
  const double scale=std::max({1.0,std::abs(candidate_kkt),std::abs(best_kkt)});
  const double tolerance=32.0*std::numeric_limits<double>::epsilon()*scale;
  if (candidate_kkt < best_kkt-tolerance) return true;
  if (std::abs(candidate_kkt-best_kkt)>tolerance) return false;
  if (candidate_primal < best_primal) return true;
  if (candidate_primal > best_primal) return false;
  return std::abs(candidate_rho-anchor_rho)<std::abs(best_rho-anchor_rho);
}

struct DcaProgressDecision {
  bool aggregate_stall = false;
  bool near_feasible_dual_stall = false;
  double aggregate_reference_kkt = 0.0;
  double dual_reference = 0.0;
};

// Observe accepted outer iterates only. Aggregate and dual histories are kept
// separately: a non-aggregate transaction breaks an aggregate streak, while a
// loss of original feasibility breaks a near-feasible dual streak.
class DcaAcceptedProgress {
 public:
  explicit DcaAcceptedProgress(int window, double contraction = 0.99)
      : window_(window), contraction_(contraction) {}

  DcaProgressDecision observe(double kkt, double primal, double dual,
                              bool aggregate, double feasibility_tolerance,
                              double dual_tolerance) {
    DcaProgressDecision decision;
    if (window_ <= 0 || !std::isfinite(kkt) || !std::isfinite(primal) ||
        !std::isfinite(dual)) {
      aggregate_kkt_.clear();
      dual_residual_.clear();
      return decision;
    }

    if (!aggregate) aggregate_kkt_.clear();
    else {
      aggregate_kkt_.push_back(kkt);
      trim(aggregate_kkt_);
      if (static_cast<int>(aggregate_kkt_.size()) == window_) {
        decision.aggregate_reference_kkt = aggregate_kkt_.front();
        decision.aggregate_stall =
            kkt >= contraction_ * decision.aggregate_reference_kkt;
        if (decision.aggregate_stall) aggregate_kkt_.clear();
      }
    }

    if (primal > feasibility_tolerance || dual <= dual_tolerance)
      dual_residual_.clear();
    else {
      dual_residual_.push_back(dual);
      trim(dual_residual_);
      if (static_cast<int>(dual_residual_.size()) == window_) {
        decision.dual_reference = dual_residual_.front();
        decision.near_feasible_dual_stall =
            dual >= contraction_ * decision.dual_reference;
        if (decision.near_feasible_dual_stall) dual_residual_.clear();
      }
    }
    return decision;
  }

  void clear() {
    aggregate_kkt_.clear();
    dual_residual_.clear();
  }

 private:
  void trim(std::deque<double>& values) const {
    while (static_cast<int>(values.size()) > window_) values.pop_front();
  }

  int window_ = 0;
  double contraction_ = 0.99;
  std::deque<double> aggregate_kkt_, dual_residual_;
};

} // namespace lazyipdca::qcqp::detail
