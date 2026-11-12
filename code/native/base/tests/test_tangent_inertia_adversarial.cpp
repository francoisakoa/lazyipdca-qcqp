#include "lazyipdca_qcqp/condensed_retraction.hpp"
#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem coordinate_manifold(bool duplicate_rows = false,
                                bool lock_hidden_mode = false) {
  constexpr int n = 8;
  const double inf = std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name = duplicate_rows ? "duplicate_rows" : "hidden_negative_mode";
  p.n = n;
  p.x0.assign(n, 0.0);
  p.lb.assign(n, -inf);
  p.ub.assign(n, inf);
  if (lock_hidden_mode) p.lb[7] = p.ub[7] = 0.0;
  p.objective.name = "zero";
  p.objective.linear.assign(n, 0.0);
  p.objective.hessian.n = n;
  for (int row = 0; row < 2; ++row) {
    QuadraticFunction h;
    h.name = "coordinate_" + std::to_string(row);
    h.linear.assign(n, 0.0);
    const int coordinate = duplicate_rows ? 0 : row;
    h.linear[coordinate] = 1.0;
    h.hessian.n = n;
    h.hessian.entries.push_back({coordinate, coordinate, 2.0});
    p.equalities.push_back(std::move(h));
  }
  p.validate();
  return p;
}

SymMatrix hidden_mode_hessian() {
  SymMatrix h;
  h.n = 8;
  for (int j = 0; j < 8; ++j)
    h.entries.push_back({j, j, j == 7 ? -1.0 : 1.0});
  return h;
}

}  // namespace

int main() {
  try {
    Settings st;
    st.linear_backend = "native";
    st.kkt_adaptive_regularization = true;
    st.dca_condensed_regularization = 1e-12;
    st.dca_condensed_regularization_max = 1e-4;
    st.fixed_bound_tolerance = 1e-12;
    std::string message;

    const QCQPProblem p = coordinate_manifold();
    CondensedQuadraticRetraction geometry(st);
    if (!geometry.factor(p, p.x0, message))
      throw std::runtime_error("normal factorization failed: " + message);
    const SymMatrix h = hidden_mode_hessian();
    Convexification none;
    if (geometry.certify_tangent_convex_by_inertia(h, none, -1e-10, message))
      throw std::runtime_error("inertia accepted a hidden tangent eigenvalue");

    Convexification corrected;
    Vec hidden(8, 0.0);
    hidden[7] = 1.0;
    corrected.terms.push_back({2.0, hidden});
    if (!geometry.certify_tangent_convex_by_inertia(
            h, corrected, -1e-10, message))
      throw std::runtime_error("rank-one correction rejected: " + message);

    const QCQPProblem locked = coordinate_manifold(false, true);
    CondensedQuadraticRetraction locked_geometry(st);
    if (!locked_geometry.factor(locked, locked.x0, message) ||
        !locked_geometry.certify_tangent_convex_by_inertia(
            h, none, -1e-10, message))
      throw std::runtime_error("fixed-coordinate negative mode retained: " +
                               message);

    Convexification omitted;
    const auto omitted_result = geometry.ensure_tangent_convex_multi_ritz(
        h, omitted, 0.0, 0);
    if (!omitted_result.success || omitted_result.rank_added != 0 ||
        omitted_result.oracle_calls != 0 ||
        omitted_result.scalar_safeguard_updates <= 0)
      throw std::runtime_error("inertia did not repair omitted Ritz mode: " +
                               omitted_result.message);

    SymMatrix capped_h = h;
    for (auto& entry : capped_h.entries)
      if (entry.i >= 5) entry.v = -1.0;
    Convexification capped;
    const auto capped_result = geometry.ensure_tangent_convex_multi_ritz(
        capped_h, capped, 0.0, 1);
    if (!capped_result.success || capped_result.rank_added != 1 ||
        capped_result.scalar_safeguard_updates <= 0 ||
        capped.scalar_shift <= 0.0)
      throw std::runtime_error("rank-cap recovery not exercised: " +
                               capped_result.message);

    const QCQPProblem duplicate = coordinate_manifold(true, false);
    CondensedQuadraticRetraction duplicate_geometry(st);
    if (!duplicate_geometry.factor(duplicate, duplicate.x0, message))
      throw std::runtime_error("regularized duplicate normal factor failed: " +
                               message);
    Convexification duplicate_cvx;
    const auto duplicate_result =
        duplicate_geometry.ensure_tangent_convex_multi_ritz(
            h, duplicate_cvx, 0.0, 0);
    if (duplicate_result.success)
      throw std::runtime_error("rank-deficient J passed exact inertia");

    // Integration contract: the solver discards the failed proposal and
    // enters the unchanged historical route, rather than accepting or aborting
    // on the uncertified model.
    QCQPProblem fallback_problem = duplicate;
    fallback_problem.objective.linear[7] = -1.0;
    fallback_problem.objective.hessian = h;
    fallback_problem.validate();
    Settings fallback_settings = st;
    fallback_settings.verbose = false;
    fallback_settings.outer_algorithm = "lazy-dca";
    fallback_settings.inner_backend = "lazyqp";
    fallback_settings.dca_equality_model = "condensed-retraction";
    fallback_settings.dca_condensed_projected_multi_ritz_inertia = true;
    fallback_settings.restoration_enabled = false;
    fallback_settings.max_outer_iterations = 1;
    fallback_settings.max_inner_iterations = 100;
    fallback_settings.dca_local_corrector = false;
    fallback_settings.dca_iterated_soc = false;
    fallback_settings.dca_equality_retraction = false;
    const Result fallback_result = Solver(fallback_settings).solve(
        fallback_problem);
    if (fallback_result.lazy_dca_condensed_projected_multi_ritz_fallbacks < 1)
      throw std::runtime_error(
          "rank-deficient J did not enter the historical solver fallback: " +
          fallback_result.message);

    Convexification invalid = corrected;
    invalid.terms.front().beta = -1.0;
    if (geometry.certify_tangent_convex_by_inertia(
            h, invalid, -1e-10, message))
      throw std::runtime_error("negative low-rank weight was certified");

    std::cout << "[PASS] hidden mode, fixed face, rank cap, rank-deficient J, "
                 "and invalid correction\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
