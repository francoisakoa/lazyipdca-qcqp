#include "lazyipdca_qcqp/condensed_retraction.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

using namespace lazyipdca::qcqp;

namespace {

QuadraticFunction quadratic(int n, const std::string& name) {
  QuadraticFunction q;
  q.name = name;
  q.linear.assign(static_cast<std::size_t>(n), 0.0);
  q.hessian.n = n;
  return q;
}

QCQPProblem flow_lift_problem(bool duplicate_balance = false,
                              bool corrupt_signature = false) {
  QCQPProblem p;
  p.name = "synthetic_pglib_flow_lift";
  p.n = 6; // four physical coordinates followed by P/Q auxiliaries
  p.lb.assign(6, -std::numeric_limits<double>::infinity());
  p.ub.assign(6, std::numeric_limits<double>::infinity());
  p.lb[3] = 0.0;
  p.ub[3] = 2.0;
  p.x0 = {0.4, -0.2, 0.3, 0.0, 0.13, -0.04};
  p.objective = quadratic(6, "objective");
  p.objective.linear[0] = 1.0;
  p.objective.hessian.entries.push_back({1, 1, 2.0});

  auto balance = quadratic(6, "balance_P_1");
  balance.constant = -0.3;
  balance.linear[0] = 1.0;
  balance.linear[1] = 2.0;
  balance.linear[2] = 1.0;
  p.equalities.push_back(balance);
  if (duplicate_balance) {
    balance.name = "balance_P_duplicate";
    p.equalities.push_back(balance);
  }

  // s_P - z0^2 - 0.5 z1 z2 = 0.
  auto hp = quadratic(6, "flowP_f_0");
  hp.linear[4] = 1.0;
  hp.hessian.entries.push_back({0, 0, -2.0});
  hp.hessian.entries.push_back({1, 2, -0.5});
  p.equalities.push_back(hp);

  // -s_Q - z1^2 - z2 z3 = 0 (exercise the generic sigma=-1 path).
  auto hq = quadratic(6, "flowQ_f_0");
  hq.linear[5] = -1.0;
  hq.hessian.entries.push_back({1, 1, -2.0});
  hq.hessian.entries.push_back({2, 3, -1.0});
  p.equalities.push_back(hq);

  auto slim = quadratic(6, "slim_f_0");
  slim.constant = -4.0;
  slim.hessian.entries.push_back({4, 4, 2.0});
  slim.hessian.entries.push_back({5, 5, 2.0});
  if (corrupt_signature)
    slim.hessian.entries.push_back({0, 4, 1e-3});
  p.inequalities.push_back(std::move(slim));
  p.validate();
  return p;
}

QCQPProblem case118_shape_problem() {
  constexpr int physical = 344;
  constexpr int balances = 236;
  constexpr int limited_branches = 186;
  constexpr int endpoints = 2 * limited_branches;
  constexpr int lifts = 2 * endpoints;
  constexpr int n = physical + lifts;
  QCQPProblem p;
  p.name = "case118_apparent_flow_shape";
  p.n = n;
  p.x0.assign(n, 0.1);
  p.lb.assign(n, -std::numeric_limits<double>::infinity());
  p.ub.assign(n, std::numeric_limits<double>::infinity());
  for (int j = 0; j < physical; ++j) {
    p.lb[static_cast<std::size_t>(j)] = -10.0;
    p.ub[static_cast<std::size_t>(j)] = 10.0;
  }
  p.objective = quadratic(n, "objective");
  p.objective.linear[physical - 1] = 1.0;
  for (int i = 0; i < balances; ++i) {
    auto h = quadratic(n, "balance_" + std::to_string(i));
    h.linear[static_cast<std::size_t>(i)] = 1.0;
    h.constant = -p.x0[static_cast<std::size_t>(i)];
    p.equalities.push_back(std::move(h));
  }
  for (int endpoint = 0; endpoint < endpoints; ++endpoint) {
    const int branch = endpoint / 2;
    const std::string suffix =
        std::string(endpoint % 2 == 0 ? "f_" : "t_") +
        std::to_string(branch);
    const int zp = endpoint % physical;
    const int zq = (endpoint * 7 + 3) % physical;
    const int sp = physical + 2 * endpoint;
    const int sq = sp + 1;
    auto hp = quadratic(n, "flowP_" + suffix);
    hp.linear[static_cast<std::size_t>(sp)] = 1.0;
    hp.hessian.entries.push_back({zp, zp, -2.0});
    hp.constant = p.x0[static_cast<std::size_t>(zp)] *
                  p.x0[static_cast<std::size_t>(zp)] -
                  p.x0[static_cast<std::size_t>(sp)];
    p.equalities.push_back(std::move(hp));
    auto hq = quadratic(n, "flowQ_" + suffix);
    hq.linear[static_cast<std::size_t>(sq)] = 1.0;
    hq.hessian.entries.push_back({zq, zq, -2.0});
    hq.constant = p.x0[static_cast<std::size_t>(zq)] *
                  p.x0[static_cast<std::size_t>(zq)] -
                  p.x0[static_cast<std::size_t>(sq)];
    p.equalities.push_back(std::move(hq));
    auto slim = quadratic(n, "slim_" + suffix);
    slim.constant = -4.0;
    slim.hessian.entries.push_back({sp, sp, 2.0});
    slim.hessian.entries.push_back({sq, sq, 2.0});
    p.inequalities.push_back(std::move(slim));
  }
  p.validate();
  return p;
}

double max_difference(const Vec& a, const Vec& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    result = std::max(result, std::abs(a[i] - b[i]));
  return result;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void compare_maps(const QCQPProblem& p, const Settings& settings,
                  bool bound_aware, double tolerance,
                  bool expect_positive_regularization) {
  CondensedQuadraticRetraction schur(settings, true);
  CondensedQuadraticRetraction bordered(settings, false);
  std::string message;
  const bool schur_ok = bound_aware
      ? schur.factor_bound_aware(p, p.x0, message)
      : schur.factor(p, p.x0, message);
  require(schur_ok, "Schur factorization failed: " + message);
  const bool bordered_ok = bound_aware
      ? bordered.factor_bound_aware(p, p.x0, message)
      : bordered.factor(p, p.x0, message);
  require(bordered_ok, "bordered factorization failed: " + message);
  require(schur.flow_lift_schur_active(), "Schur route was not activated");
  require(schur.flow_lift_schur_detected_variables() == 2,
          "wrong number of detected flow auxiliaries");
  const int expected_core = static_cast<int>(p.equalities.size()) - 2;
  require(schur.normal_factor_dimension() == 4 + expected_core,
          "wrong reduced normal dimension");
  require(bordered.normal_factor_dimension() ==
              p.n + static_cast<int>(p.equalities.size()),
          "historical bordered dimension changed");
  if (expect_positive_regularization) {
    require(schur.regularization() > 0.0 && bordered.regularization() > 0.0,
            "rank-deficient test did not exercise positive regularization");
  } else {
    require(schur.regularization() == 0.0 && bordered.regularization() == 0.0,
            "full-rank test unexpectedly used regularization");
  }

  Vec distinct_rhs(p.equalities.size(), 0.0);
  for (std::size_t i = 0; i < distinct_rhs.size(); ++i)
    distinct_rhs[i] = (i % 2 == 0 ? 0.17 : -0.31) *
                      static_cast<double>(i + 1);
  const std::vector<Vec> right_hand_sides = {
      Vec(p.equalities.size(), 1.0), std::move(distinct_rhs),
      Vec(p.equalities.size(), 0.0)};
  for (std::size_t k = 0; k + 1 < right_hand_sides.size(); ++k) {
    Vec a, b;
    require(schur.apply_right_inverse(right_hand_sides[k], a, message),
            "Schur solve failed: " + message);
    require(bordered.apply_right_inverse(right_hand_sides[k], b, message),
            "bordered solve failed: " + message);
    require(max_difference(a, b) <= tolerance,
            "Schur/bordered right-inverse mismatch");
  }

  const Vec v = {0.7, -0.4, 0.2, 0.3, -0.8, 0.5};
  const Vec w = {-0.1, 0.9, 0.6, -0.2, 0.4, -0.7};
  Vec pv_schur, pv_bordered, pw_schur;
  require(schur.project_tangent(v, pv_schur, message),
          "Schur tangent projection failed: " + message);
  require(bordered.project_tangent(v, pv_bordered, message),
          "bordered tangent projection failed: " + message);
  require(schur.project_tangent(w, pw_schur, message),
          "second Schur tangent projection failed: " + message);
  require(max_difference(pv_schur, pv_bordered) <= 4.0 * tolerance,
          "Schur/bordered tangent projection mismatch");
  require(std::abs(dot(v, pw_schur) - dot(pv_schur, w)) <=
              20.0 * tolerance,
          "Schur tangent projector lost symmetry");
  if (bound_aware)
    require(std::abs(pv_schur[3]) <= tolerance,
            "bound-aware Schur route moved a locked coordinate");
}

} // namespace

int main() {
  try {
    Settings settings;
    settings.linear_backend = "native";
    settings.kkt_adaptive_regularization = true;
    settings.kkt_backward_error_tolerance = 1e-10;
    settings.kkt_backward_error_reject = 1e-8;
    settings.dca_condensed_regularization = 1e-8;
    settings.dca_condensed_regularization_max = 1e-4;
    settings.dca_condensed_cached_regularization = false;
    settings.dca_condensed_bound_aware_normal = true;

    const auto regular = flow_lift_problem();
    CondensedQuadraticRetraction default_guard(settings);
    std::string message;
    require(default_guard.factor(regular, regular.x0, message),
            "default-off factorization failed: " + message);
    require(!default_guard.flow_lift_schur_active() &&
                default_guard.flow_lift_schur_detected_variables() == 0,
            "default constructor bypassed the opt-in flow-lift guard");
    compare_maps(regular, settings, false, 2e-10, false);
    compare_maps(regular, settings, true, 2e-10, false);

    const auto rank_deficient = flow_lift_problem(true, false);
    compare_maps(rank_deficient, settings, false, 2e-7, true);

    const auto corrupt = flow_lift_problem(false, true);
    CondensedQuadraticRetraction guarded(settings, true);
    require(guarded.factor(corrupt, corrupt.x0, message),
            "guarded fallback factorization failed: " + message);
    require(!guarded.flow_lift_schur_active() &&
                guarded.flow_lift_schur_detected_variables() == 0,
            "corrupt flow-lift signature did not fail closed");
    require(guarded.normal_factor_dimension() ==
                corrupt.n + static_cast<int>(corrupt.equalities.size()),
            "corrupt signature did not retain bordered dimension");

    const auto case118_shape = case118_shape_problem();
    CondensedQuadraticRetraction case118_schur(settings, true);
    require(case118_schur.factor(case118_shape, case118_shape.x0, message),
            "case118-shape Schur factorization failed: " + message);
    require(case118_schur.flow_lift_schur_active() &&
                case118_schur.flow_lift_schur_detected_variables() == 744,
            "case118-shape detector did not recognize 744 auxiliaries");
    require(case118_schur.normal_factor_dimension() == 580,
            "case118-shape normal factor was not reduced from 2068 to 580");
    Vec rhs(case118_shape.equalities.size(), 1e-3), correction;
    require(case118_schur.apply_right_inverse(rhs, correction, message),
            "case118-shape Schur solve failed: " + message);
    double residual = 0.0;
    for (std::size_t i = 0; i < case118_shape.equalities.size(); ++i)
      residual = std::max(
          residual,
          std::abs(dot(case118_shape.equalities[i].gradient(case118_shape.x0),
                       correction) - rhs[i]));
    require(residual <= 2e-10,
            "case118-shape Schur correction failed the original Jacobian");

    std::cout << "[PASS] exact PGLib flow-lift Schur normal projector parity\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
