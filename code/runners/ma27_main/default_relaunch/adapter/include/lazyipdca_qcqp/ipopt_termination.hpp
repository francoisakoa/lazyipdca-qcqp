#pragma once

#include "lazyipdca_qcqp/settings.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lazyipdca::qcqp {

// Desired-convergence test matching Ipopt 3.14's multiplier scaling structure.
//
// IpoptCalculatedQuantities::ComputeOptimalityErrorScaling computes
//   s_c = max(1, (||z_L||_1+||z_U||_1+||v_L||_1+||v_U||_1)
//                    /(dim(z_L)+dim(z_U)+dim(v_L)+dim(v_U))/s_max)
//   s_d = max(1, (||y_c||_1+||y_d||_1+||z_L||_1+||z_U||_1
//                    +||v_L||_1+||v_U||_1)
//                    /(dim(y_c)+dim(y_d)+dim(z_L)+dim(z_U)+dim(v_L)+dim(v_U))/s_max).
// Empty multiplier collections have scaling one.  In particular, the
// denominators are the ACTUAL dimensions of the multiplier vectors, not the
// primal/slack dimensions n and m from the implementation paper shorthand.
struct IpoptTerminationMetrics {
  double dual_infeasibility = std::numeric_limits<double>::infinity();
  double constraint_violation = std::numeric_limits<double>::infinity();
  double complementarity = std::numeric_limits<double>::infinity();
  double dual_scaling = 1.0;
  double complementarity_scaling = 1.0;
  double scaled_dual_infeasibility = std::numeric_limits<double>::infinity();
  double scaled_complementarity = std::numeric_limits<double>::infinity();
  double overall_nlp_error = std::numeric_limits<double>::infinity();
  double y_l1 = 0.0;
  double z_l1 = 0.0;
  int dual_multiplier_dimension = 0;
  int complementarity_multiplier_dimension = 0;
  bool finite = false;
  bool converged = false;
};

inline IpoptTerminationMetrics ipopt_desired_convergence_metrics(
    double dual_inf,
    double constr_viol,
    double compl_inf,
    double y_l1,
    double z_l1,
    int dual_multiplier_dimension,
    int complementarity_multiplier_dimension,
    const Settings& st) {
  IpoptTerminationMetrics out;
  out.dual_infeasibility = dual_inf;
  out.constraint_violation = constr_viol;
  out.complementarity = compl_inf;
  out.y_l1 = y_l1;
  out.z_l1 = z_l1;
  out.dual_multiplier_dimension = std::max(0, dual_multiplier_dimension);
  out.complementarity_multiplier_dimension =
      std::max(0, complementarity_multiplier_dimension);

  const double smax = st.nlp_error_scaling_max;
  if (!(smax > 0.0) || !std::isfinite(smax) ||
      !std::isfinite(dual_inf) || !std::isfinite(constr_viol) ||
      !std::isfinite(compl_inf) || !std::isfinite(y_l1) ||
      !std::isfinite(z_l1) || y_l1 < 0.0 || z_l1 < 0.0) {
    return out;
  }

  if (out.complementarity_multiplier_dimension == 0) {
    out.complementarity_scaling = 1.0;
  } else {
    const double avg_z = z_l1 /
        static_cast<double>(out.complementarity_multiplier_dimension);
    out.complementarity_scaling = std::max(smax, avg_z) / smax;
  }

  if (out.dual_multiplier_dimension == 0) {
    out.dual_scaling = 1.0;
  } else {
    const double avg_yz = (y_l1 + z_l1) /
        static_cast<double>(out.dual_multiplier_dimension);
    out.dual_scaling = std::max(smax, avg_yz) / smax;
  }

  out.scaled_dual_infeasibility = dual_inf / out.dual_scaling;
  out.scaled_complementarity = compl_inf / out.complementarity_scaling;
  out.overall_nlp_error = std::max({out.scaled_dual_infeasibility,
                                    constr_viol,
                                    out.scaled_complementarity});
  out.finite = std::isfinite(out.overall_nlp_error);
  out.converged = out.finite &&
      out.overall_nlp_error <= st.stationarity_tolerance &&
      dual_inf <= st.dual_infeasibility_tolerance &&
      constr_viol <= st.feasibility_tolerance &&
      compl_inf <= st.complementarity_tolerance;
  return out;
}

} // namespace lazyipdca::qcqp
