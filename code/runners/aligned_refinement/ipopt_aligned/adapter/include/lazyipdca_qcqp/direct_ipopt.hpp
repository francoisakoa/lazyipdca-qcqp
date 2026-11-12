#pragma once

#include "lazyipdca_qcqp/model.hpp"

#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lazyipdca::qcqp {

// Options for solving the original, unmodified QCQP with Ipopt.  This is a
// reference route: no DC majorant, working set, flow-lift elimination, or
// equality condensation is applied.
struct DirectIpoptSettings {
  double tolerance = 1.0e-8;
  double constraint_violation_tolerance = 1.0e-6;
  double dual_infeasibility_tolerance = 1.0e-6;
  double complementarity_tolerance = 1.0e-6;
  int acceptable_iterations = 0;
  double acceptable_tolerance = 1.0e-6;
  double acceptable_constraint_violation_tolerance = 1.0e-2;
  double acceptable_dual_infeasibility_tolerance = 1.0e10;
  double acceptable_complementarity_tolerance = 1.0e-2;
  double acceptable_objective_change_tolerance = 1e20;
  double nlp_error_scaling_max = 100.0;
  double time_limit_seconds = 7200.0;
  int max_iterations = 5000;
  int print_level = 5;
  std::string linear_solver = "mumps";
  std::string mu_strategy = "monotone";
  // Optional path passed verbatim to Ipopt's official `hsllib` option.
  // Leaving it empty preserves Ipopt's own dynamic-loader search policy.
  std::string hsl_library;
  bool derivative_test = false;
  bool print_timing_statistics = false;
};

// This fixed contract is checked by the solver API as well as the CLI.
inline void validate_aligned_ipopt_settings(const DirectIpoptSettings& s) {
  if(s.mu_strategy!="adaptive"&&s.mu_strategy!="monotone")
    throw std::invalid_argument("mu_strategy must be adaptive or monotone");
  if(s.linear_solver!="ma27"&&s.linear_solver!="mumps")
    throw std::invalid_argument("linear_solver must be ma27 or mumps");
  if(s.tolerance!=1e-8||s.constraint_violation_tolerance!=1e-6||
     s.dual_infeasibility_tolerance!=1e-6||s.complementarity_tolerance!=1e-6||
     s.acceptable_iterations!=0||s.max_iterations!=5000||s.time_limit_seconds!=7200.0||
     s.derivative_test||s.print_timing_statistics||s.print_level<0||s.print_level>12)
    throw std::invalid_argument("aligned Ipopt fixed tolerance/budget contract violated");
  if(s.linear_solver=="mumps"&&!s.hsl_library.empty())
    throw std::invalid_argument("hsllib is only admitted for ma27");
}

// A deterministic sparse derivative plan for an original QCQP.  Its public
// structure/value accessors make the exact TNLP contract testable even in a
// build where Ipopt itself is unavailable.
class OriginalQCQPDerivatives {
 public:
  explicit OriginalQCQPDerivatives(const QCQPProblem& problem);

  int variables() const { return problem_->n; }
  int constraints() const {
    return static_cast<int>(constraints_.size());
  }
  int equalities() const {
    return static_cast<int>(problem_->equalities.size());
  }

  const std::vector<int>& jacobian_rows() const { return jacobian_rows_; }
  const std::vector<int>& jacobian_columns() const {
    return jacobian_columns_;
  }
  const std::vector<int>& hessian_rows() const { return hessian_rows_; }
  const std::vector<int>& hessian_columns() const {
    return hessian_columns_;
  }

  double objective_value(const Vec& x) const;
  void objective_gradient(const Vec& x, Vec& values) const;
  void constraint_values(const Vec& x, Vec& values) const;
  void jacobian_values(const Vec& x, Vec& values) const;
  void lagrangian_hessian_values(double objective_factor,
                                 const Vec& constraint_multipliers,
                                 Vec& values) const;

 private:
  struct GradientTerm {
    std::size_t first_position = 0;
    std::size_t second_position = 0;
    int first_variable = 0;
    int second_variable = 0;
    double value = 0.0;
  };

  struct QuadraticPlan {
    const QuadraticFunction* function = nullptr;
    std::size_t jacobian_offset = 0;
    std::vector<int> columns;
    std::vector<double> linear_values;
    std::vector<GradientTerm> gradient_terms;
  };

  double value(const QuadraticPlan& plan, const Vec& x) const;

  const QCQPProblem* problem_ = nullptr;
  std::vector<const QuadraticFunction*> constraints_;
  QuadraticPlan objective_plan_;
  std::vector<QuadraticPlan> constraint_plans_;
  std::vector<int> jacobian_rows_;
  std::vector<int> jacobian_columns_;
  std::vector<int> hessian_rows_;
  std::vector<int> hessian_columns_;
  // Entry zero is the objective; entries 1..m follow the constraint order.
  std::vector<std::vector<std::pair<std::size_t, double>>>
      hessian_contributions_;
};

struct DirectIpoptResult {
  double parse_seconds=0, derivative_plan_seconds=0, setup_seconds=0;
  bool aligned_contract_verified=false;
  bool option_file_disabled=false;
  std::string mu_strategy;
  double core_s=0, postsolve_seconds=0;
  std::string library_defaults_json="{}";
  std::string representation_overrides_json="[]";
  std::string start_source="embedded_x0";
  std::string linear_solver;
  int requested_mumps_ordering=0;
  double wall_limit_seconds=0;
  int max_iterations=0;
  bool ipopt_native_square_problem = false;
  double original_normalized_nlp_error = std::numeric_limits<double>::infinity();
  double ipopt_native_dual_inf = std::numeric_limits<double>::infinity();
  double ipopt_native_primal_inf = std::numeric_limits<double>::infinity();
  double ipopt_native_compl_inf = std::numeric_limits<double>::infinity();

  bool available = false;
  bool success = false;
  bool ipopt_termination_accepted = false;
  bool ipopt_desired_convergence = false;
  bool acceptable_absolute_limits_pass = false;
  bool acceptable_convergence = false;
  std::string status = "unavailable";
  std::string message;
  std::string application_status;
  std::string solver_return;
  int application_status_code = 0;
  int solver_return_code = 0;
  int iterations = 0;
  int n = 0;
  int equalities = 0;
  int inequalities = 0;
  int acceptable_required = 0;
  std::size_t jacobian_nnz = 0;
  std::size_t lagrangian_hessian_nnz = 0;
  double objective = std::numeric_limits<double>::infinity();
  Feasibility feasibility{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity()};
  double dual_infeasibility_inf = std::numeric_limits<double>::infinity();
  double complementarity_inf = std::numeric_limits<double>::infinity();
  double scaled_nlp_error = std::numeric_limits<double>::infinity();
  double ipopt_internal_scaled_nlp_error =
      std::numeric_limits<double>::infinity();
  double ipopt_internal_unscaled_nlp_error =
      std::numeric_limits<double>::infinity();
  double acceptable_tolerance = std::numeric_limits<double>::infinity();
  double acceptable_constraint_violation_tolerance =
      std::numeric_limits<double>::infinity();
  double acceptable_dual_infeasibility_tolerance =
      std::numeric_limits<double>::infinity();
  double acceptable_complementarity_tolerance =
      std::numeric_limits<double>::infinity();
  double dual_error_scaling = 1.0;
  double complementarity_error_scaling = 1.0;
  double constraint_multiplier_l1 = 0.0;
  double bound_slack_multiplier_l1 = 0.0;
  double inequality_multiplier_sign_violation = 0.0;
  double bound_multiplier_sign_violation = 0.0;
  double elapsed_seconds = 0.0;
  Vec x;
  Vec equality_multipliers;
  Vec inequality_multipliers;
  Vec lower_bound_multipliers;
  Vec upper_bound_multipliers;
};

DirectIpoptResult solve_direct_ipopt(const QCQPProblem& problem,
                                     const DirectIpoptSettings& settings);

// Classify an Ipopt endpoint without conflating strict desired convergence
// with Ipopt's explicitly configured acceptable termination.  The latter is
// accepted only when Ipopt's own scaled NLP error and independently
// recomputed original-QCQP absolute ceilings all pass.
void classify_direct_ipopt_result(const DirectIpoptSettings& settings,
                                  DirectIpoptResult& result);

std::string direct_ipopt_result_json(const DirectIpoptResult& result,
                                     const std::string& problem_name);
void write_direct_ipopt_result_json(const DirectIpoptResult& result,
                                    const std::string& problem_name,
                                    const std::string& path);

} // namespace lazyipdca::qcqp
