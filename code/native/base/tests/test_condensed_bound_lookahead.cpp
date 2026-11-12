#include "lazyipdca_qcqp/condensed_retraction.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_future_face_problem() {
  QCQPProblem p;
  p.name = "condensed_bound_lookahead";
  p.n = 3;
  p.x0 = {0.5, -0.5, 0.0};
  p.lb = {0.0, -std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity()};
  p.ub.assign(3, std::numeric_limits<double>::infinity());
  p.objective.name = "zero";
  p.objective.linear.assign(3, 0.0);
  p.objective.hessian.n = 3;

  // h(x)=x0+x1+x2^2.  The tangent step reaches x0's lower bound and
  // leaves the exact quadratic defect x2^2=0.04 for the normal correction.
  QuadraticFunction h;
  h.name = "future_face_parabola";
  h.linear = {1.0, 1.0, 0.0};
  h.hessian.n = 3;
  h.hessian.entries = {{2, 2, 2.0}};
  p.equalities.push_back(std::move(h));
  p.validate();
  return p;
}

Settings test_settings() {
  Settings st;
  st.linear_backend = "native";
  st.dca_condensed_bound_aware_normal = true;
  st.dca_condensed_retraction_tolerance = 1e-12;
  st.dca_condensed_retraction_max_ratio = 4.0;
  st.inner_tolerance = 1e-12;
  return st;
}

} // namespace

int main() {
  try {
    const QCQPProblem p = make_future_face_problem();
    const Vec step = {-0.5, 0.5, 0.2};
    std::string message;

    CondensedQuadraticRetraction historical(test_settings());
    if (!historical.factor_bound_aware(p, p.x0, message))
      throw std::runtime_error("historical factorization failed: " + message);
    const auto leaking = historical.retract(step);
    if (!leaking.success || !(leaking.feasibility.bound_inf > 1e-3))
      throw std::runtime_error(
          "synthetic problem no longer exposes the future-face leak");

    CondensedQuadraticRetraction guarded(test_settings());
    if (guarded.active_bound_coordinates(p, p.x0) != 0 ||
        guarded.active_bound_coordinates(p, p.x0, step) != 1)
      throw std::runtime_error("look-ahead face was not identified exactly");
    if (!guarded.factor_bound_aware(p, p.x0, step, message))
      throw std::runtime_error("look-ahead factorization failed: " + message);
    const auto kept = guarded.retract(step);
    if (!kept.success || guarded.locked_coordinates() != 1 ||
        kept.feasibility.equality_inf > 1e-12 ||
        kept.feasibility.bound_inf != 0.0 ||
        std::abs(kept.x[0]) > 1e-14 ||
        std::abs(kept.x[1] + 0.04) > 1e-11 ||
        std::abs(kept.x[2] - 0.2) > 1e-14) {
      throw std::runtime_error(
          "look-ahead normal map did not preserve the future bound face");
    }

    std::cout << "[PASS] condensed bound look-ahead prevents future-face leak"
              << " historical_bound=" << leaking.feasibility.bound_inf
              << " guarded_bound=" << kept.feasibility.bound_inf << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
