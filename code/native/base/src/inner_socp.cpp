#include "lazyipdca_qcqp/inner_socp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace lazyipdca::qcqp {
namespace {

bool finite_model(const ConvexQuadraticModel& model, int n,
                  std::string& message) {
  if (static_cast<int>(model.linear.size()) != n ||
      model.base_hessian.n != n || model.shift_dimension < 0 ||
      model.shift_dimension > n || !std::isfinite(model.constant) ||
      !std::isfinite(model.scalar_shift)) {
    message = "invalid convex quadratic dimensions or scalar data";
    return false;
  }
  for (double value : model.linear) {
    if (!std::isfinite(value)) {
      message = "nonfinite convex quadratic linear coefficient";
      return false;
    }
  }
  try {
    model.base_hessian.validate();
  } catch (const std::exception& e) {
    message = e.what();
    return false;
  }
  // A negative explicit shift or atom cannot be the Gram part of a rotated
  // cone.  The sparse base Hessian is trusted only through the
  // ConvexSubproblem contract; LIN/COND construct and certify that model
  // before this backend is entered.
  if (model.scalar_shift < 0.0) {
    message = "negative scalar convexification shift in SOCP model";
    return false;
  }
  for (const auto& term : model.low_rank) {
    if (!std::isfinite(term.beta) || term.beta < 0.0 ||
        static_cast<int>(term.direction.size()) != n) {
      message = "invalid PSD atom in SOCP model";
      return false;
    }
    for (double value : term.direction) {
      if (!std::isfinite(value)) {
        message = "nonfinite SOCP atom direction";
        return false;
      }
    }
  }
  return true;
}

bool has_quadratic_part(const ConvexQuadraticModel& model) {
  if (model.shift_dimension > 0 && model.scalar_shift != 0.0) return true;
  for (const auto& entry : model.base_hessian.entries)
    if (entry.v != 0.0) return true;
  for (const auto& term : model.low_rank)
    if (term.beta != 0.0) return true;
  return false;
}

} // namespace

bool build_condensed_socp_formulation(const ConvexSubproblem& problem,
                                      CondensedSOCPFormulation& formulation,
                                      std::string& message) {
  formulation = {};
  if (problem.nvar <= 0 || problem.equalities.nvar != problem.nvar ||
      problem.equalities.rows.size() != problem.equalities.rhs.size()) {
    message = "invalid affine equality system in SOCP formulation";
    return false;
  }
  if (!finite_model(problem.objective, problem.nvar, message)) return false;
  for (std::size_t row = 0; row < problem.equalities.rows.size(); ++row) {
    if (!std::isfinite(problem.equalities.rhs[row])) {
      message = "nonfinite affine equality right-hand side in SOCP formulation";
      return false;
    }
    for (const auto& [column, value] : problem.equalities.rows[row].entries) {
      if (column < 0 || column >= problem.nvar || !std::isfinite(value)) {
        message = "invalid affine equality coefficient in SOCP formulation";
        return false;
      }
    }
  }
  for (const auto& inequality : problem.inequalities)
    if (!finite_model(inequality, problem.nvar, message)) return false;

  formulation.variable_dimension = problem.nvar;
  formulation.affine_equalities =
      static_cast<int>(problem.equalities.rows.size());
  formulation.equality_slice_psd_completion =
      formulation.affine_equalities > 0 &&
      has_quadratic_part(problem.objective);
  if (has_quadratic_part(problem.objective)) {
    formulation.objective_rotated_cones = 1;
    formulation.explicit_lift_cone_dimension +=
        static_cast<long long>(problem.nvar) + 2;
  }
  for (const auto& inequality : problem.inequalities) {
    if (has_quadratic_part(inequality)) {
      ++formulation.inequality_rotated_cones;
      formulation.explicit_lift_cone_dimension +=
          static_cast<long long>(problem.nvar) + 2;
    } else {
      ++formulation.linear_nonnegative_cones;
      ++formulation.explicit_lift_cone_dimension;
    }
  }
  message = "exact rotated-cone formulation built with implicit Gram factors";
  return true;
}

double condensed_socp_rotated_determinant(
    const ConvexQuadraticModel& model, const Vec& y,
    double epigraph_value, bool objective_epigraph) {
  const double value = model.value(y);
  return objective_epigraph ? 2.0 * (epigraph_value - value)
                            : -2.0 * value;
}

InnerResult ConvexQCQPSOCPKernel::solve(
    const ConvexSubproblem& problem) const {
  InnerResult out;
  out.socp_formulation_attempts = 1;
  CondensedSOCPFormulation formulation;
  std::string message;
  if (!build_condensed_socp_formulation(problem, formulation, message)) {
    out.socp_formulation_rejections = 1;
    out.socp_formulation_audit_failures = 1;
    out.message = "SOCP formulation rejected: " + message;
    return out;
  }

  out = condensed_kernel_.solve(problem);
  // The assignment above imports the numerical result, whose SOCP counters
  // are zero.  Account for this wrapper invocation exactly once.
  out.socp_formulation_attempts = 1;
  out.socp_formulation_builds = 1;
  out.socp_objective_rotated_cones =
      formulation.objective_rotated_cones;
  out.socp_inequality_rotated_cones =
      formulation.inequality_rotated_cones;
  out.socp_linear_nonnegative_cones =
      formulation.linear_nonnegative_cones;
  out.socp_affine_equalities = formulation.affine_equalities;
  out.socp_explicit_lift_cone_dimension =
      formulation.explicit_lift_cone_dimension;
  out.socp_condensed_newton_iterations = out.iterations;

  if (!out.y.empty() && static_cast<int>(out.y.size()) == problem.nvar) {
    double violation = 0.0;
    for (const auto& inequality : problem.inequalities) {
      const double determinant = condensed_socp_rotated_determinant(
          inequality, out.y, 0.0, false);
      if (!std::isfinite(determinant)) {
        ++out.socp_formulation_audit_failures;
        violation = std::numeric_limits<double>::infinity();
        break;
      }
      violation = std::max(violation, std::max(0.0, -determinant));
    }
    out.socp_max_rotated_cone_violation = violation;
  }

  std::ostringstream prefix;
  prefix << "exact Schur-condensed SOCP"
         << "[eq=" << st_.dca_equality_model
         << ",qcones=" << formulation.inequality_rotated_cones
         << ",lcones=" << formulation.linear_nonnegative_cones << "]";
  out.message = prefix.str() + ": " + out.message;
  return out;
}

} // namespace lazyipdca::qcqp
