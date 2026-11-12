#include "lazyipdca_qcqp/direct_ipopt.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace lazyipdca::qcqp;

namespace {

bool close(double first, double second, double tolerance = 1.0e-13) {
  return std::abs(first - second) <= tolerance;
}

QuadraticFunction function(const char* name, double constant, Vec linear,
                           std::vector<SymTriplet> hessian) {
  QuadraticFunction result;
  result.name = name;
  result.constant = constant;
  result.linear = std::move(linear);
  result.hessian.n = 3;
  result.hessian.entries = std::move(hessian);
  return result;
}

void compactify(QuadraticFunction& function, int dimension) {
  function.compact_linear_dimension = dimension;
  for (int column = 0; column < dimension; ++column)
    if (function.linear[column] != 0.0)
      function.compact_linear.emplace_back(column, function.linear[column]);
  function.linear.clear();
}

} // namespace

int main() {
  const double infinity = std::numeric_limits<double>::infinity();
  QCQPProblem problem;
  problem.name = "direct_structure";
  problem.n = 3;
  problem.x0 = {1.0, 2.0, 3.0};
  problem.lb = {-infinity, -infinity, -infinity};
  problem.ub = {infinity, infinity, infinity};
  problem.objective = function(
      "objective", 0.5, {1.0, 0.0, -2.0},
      {{0, 0, 2.0}, {0, 2, 3.0}});
  problem.equalities.push_back(function(
      "quadratic_equality", -1.0, {0.0, 4.0, 0.0},
      {{0, 1, 5.0}, {2, 2, -2.0}}));
  problem.inequalities.push_back(function(
      "quadratic_inequality", 2.0, {-1.0, 0.0, 0.0},
      {{1, 2, 7.0}}));
  problem.validate();

  OriginalQCQPDerivatives derivatives(problem);
  const std::vector<int> expected_jacobian_rows{0, 0, 0, 1, 1, 1};
  const std::vector<int> expected_jacobian_columns{0, 1, 2, 0, 1, 2};
  const std::vector<int> expected_hessian_rows{0, 1, 2, 2, 2};
  const std::vector<int> expected_hessian_columns{0, 0, 0, 1, 2};
  if (derivatives.jacobian_rows() != expected_jacobian_rows ||
      derivatives.jacobian_columns() != expected_jacobian_columns ||
      derivatives.hessian_rows() != expected_hessian_rows ||
      derivatives.hessian_columns() != expected_hessian_columns) {
    std::cerr << "[FAIL] sparse derivative structure mismatch\n";
    return 1;
  }

  const Vec x{1.0, 2.0, 3.0};
  if (!close(derivatives.objective_value(x), 5.5)) {
    std::cerr << "[FAIL] objective value mismatch\n";
    return 2;
  }

  Vec values;
  derivatives.objective_gradient(x, values);
  const Vec expected_gradient{12.0, 0.0, 1.0};
  if (values != expected_gradient) {
    std::cerr << "[FAIL] objective gradient mismatch\n";
    return 3;
  }

  derivatives.constraint_values(x, values);
  if (values.size() != 2 || !close(values[0], 8.0) ||
      !close(values[1], 43.0)) {
    std::cerr << "[FAIL] constraint value mismatch\n";
    return 4;
  }

  derivatives.jacobian_values(x, values);
  const Vec expected_jacobian{10.0, 9.0, -6.0, -1.0, 21.0, 14.0};
  if (values != expected_jacobian) {
    std::cerr << "[FAIL] constraint Jacobian mismatch\n";
    return 5;
  }

  derivatives.lagrangian_hessian_values(2.0, {3.0, -4.0}, values);
  const Vec expected_hessian{4.0, 15.0, 6.0, -28.0, -6.0};
  if (values != expected_hessian) {
    std::cerr << "[FAIL] Lagrangian Hessian mismatch\n";
    return 6;
  }

  // This equality contribution is the direct-QCQP correctness guard: the
  // convex-subproblem Ipopt adapter intentionally never needed it.
  derivatives.lagrangian_hessian_values(0.0, {1.0, 0.0}, values);
  if (!close(values[1], 5.0) || !close(values[4], -2.0)) {
    std::cerr << "[FAIL] nonlinear equality Hessian was omitted\n";
    return 7;
  }

  // V2 QCQ files keep linear coefficients sparse.  The direct Ipopt adapter
  // must preserve exactly the same plans, values and derivatives without
  // materializing one dense vector per constraint.
  auto compact_problem = problem;
  compactify(compact_problem.objective, compact_problem.n);
  for (auto& equality : compact_problem.equalities)
    compactify(equality, compact_problem.n);
  for (auto& inequality : compact_problem.inequalities)
    compactify(inequality, compact_problem.n);
  compact_problem.validate();
  OriginalQCQPDerivatives compact(compact_problem);
  if (compact.jacobian_rows() != derivatives.jacobian_rows() ||
      compact.jacobian_columns() != derivatives.jacobian_columns() ||
      compact.hessian_rows() != derivatives.hessian_rows() ||
      compact.hessian_columns() != derivatives.hessian_columns() ||
      !close(compact.objective_value(x), derivatives.objective_value(x))) {
    std::cerr << "[FAIL] compact V2 derivative plan differs from dense V1\n";
    return 8;
  }
  Vec dense_values;
  compact.objective_gradient(x, values);
  derivatives.objective_gradient(x, dense_values);
  if (values != dense_values) return 9;
  compact.constraint_values(x, values);
  derivatives.constraint_values(x, dense_values);
  if (values != dense_values) return 10;
  compact.jacobian_values(x, values);
  derivatives.jacobian_values(x, dense_values);
  if (values != dense_values) return 11;
  compact.lagrangian_hessian_values(2.0, {3.0, -4.0}, values);
  derivatives.lagrangian_hessian_values(2.0, {3.0, -4.0}, dense_values);
  if (values != dense_values) return 12;

  std::cout << "[PASS] direct Ipopt dense-V1/compact-V2 sparse objective, "
               "Jacobian and Lagrangian Hessian contract\n";
  return 0;
}
