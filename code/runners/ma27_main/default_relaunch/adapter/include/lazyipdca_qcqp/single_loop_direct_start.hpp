#pragma once

#include "lazyipdca_qcqp/model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp {

struct DirectStartRecoveryCheck {
  bool finite_original_evaluation = false;
  bool bounds_strict = false;
  bool below_theta_cap = false;
  bool contracts_theta = false;
  bool all_violated_inequalities_modeled = false;
  double theta_target = 0.0;
  double roundoff = 0.0;

  bool accepted() const {
    return finite_original_evaluation && bounds_strict && below_theta_cap &&
        contracts_theta && all_violated_inequalities_modeled;
  }
};

// Fletcher--Leyffer h-step used only before the direct start has first entered
// the strict inequality tube.  Objective change is deliberately absent: a
// lower objective can never compensate for increased original-QCQP violation.
inline DirectStartRecoveryCheck check_direct_start_recovery_step(
    const Feasibility& current, const Feasibility& trial, double alpha,
    double feasibility_tolerance, double theta_cap, double gamma_h,
    bool finite_original_evaluation,
    bool all_violated_inequalities_modeled) {
  DirectStartRecoveryCheck out;
  out.finite_original_evaluation=finite_original_evaluation &&
      std::isfinite(current.total_inf) && std::isfinite(trial.equality_inf) &&
      std::isfinite(trial.inequality_inf) &&
      std::isfinite(trial.bound_inf) && std::isfinite(trial.total_inf) &&
      std::isfinite(alpha) && alpha>0.0 &&
      std::isfinite(feasibility_tolerance) && feasibility_tolerance>=0.0 &&
      std::isfinite(theta_cap) && theta_cap>0.0 &&
      std::isfinite(gamma_h) && gamma_h>0.0 && gamma_h<1.0;
  out.all_violated_inequalities_modeled=
      all_violated_inequalities_modeled;
  if(!out.finite_original_evaluation) return out;

  out.roundoff=8192.0*std::numeric_limits<double>::epsilon()*
      std::max({1.0,std::abs(current.total_inf),
                std::abs(trial.total_inf),std::abs(theta_cap)});
  // Treat evaluation uncertainty conservatively.  Adding roundoff to the
  // target makes an unchanged trial acceptable once backtracking drives
  // gamma_h*alpha*theta below that allowance.  Contract a lower bound on the
  // current violation and compare an upper bound on the trial instead.
  const double current_lower=
      std::max(0.0,current.total_inf-out.roundoff);
  const double contracted_lower=std::fma(
      -gamma_h*alpha,current_lower,current_lower);
  out.theta_target=std::max(feasibility_tolerance,contracted_lower);
  out.bounds_strict=
      trial.bound_inf<=feasibility_tolerance+out.roundoff;
  out.below_theta_cap=trial.total_inf<=theta_cap+out.roundoff;
  // Entering the requested tube is terminal for this recovery phase.  Away
  // from it the conservative comparison guarantees that an effectively zero
  // step cannot pass merely because alpha is tiny.
  const bool enters_tube=
      trial.total_inf<=feasibility_tolerance+out.roundoff;
  out.contracts_theta=enters_tube ||
      trial.total_inf+out.roundoff<=out.theta_target;
  return out;
}

} // namespace lazyipdca::qcqp
