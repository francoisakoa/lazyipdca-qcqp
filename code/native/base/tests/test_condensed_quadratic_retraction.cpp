#include "lazyipdca_qcqp/condensed_retraction.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

QuadraticFunction circle_constraint(int n, int i, int j,
                                    const std::string& name) {
  QuadraticFunction h;
  h.name = name;
  h.constant = -1.0;
  h.linear.assign(static_cast<std::size_t>(n), 0.0);
  h.hessian.n = n;
  h.hessian.entries = {{i, i, 2.0}, {j, j, 2.0}};
  return h;
}

QCQPProblem sphere_problem() {
  QCQPProblem p;
  p.name = "condensed_sphere";
  p.n = 3;
  p.x0 = {1.0, 0.0, 0.0};
  p.lb.assign(3, -kInf);
  p.ub.assign(3, kInf);
  p.objective.name = "zero";
  p.objective.linear.assign(3, 0.0);
  p.objective.hessian.n = 3;
  QuadraticFunction h;
  h.name = "unit_sphere";
  h.constant = -1.0;
  h.linear.assign(3, 0.0);
  h.hessian.n = 3;
  h.hessian.entries = {{0, 0, 2.0}, {1, 1, 2.0}, {2, 2, 2.0}};
  p.equalities.push_back(std::move(h));
  return p;
}

QCQPProblem high_dimensional_sphere_problem(int n) {
  QCQPProblem p;
  p.name = "condensed_high_dimensional_sphere";
  p.n = n;
  p.x0.assign(static_cast<std::size_t>(n), 0.0);
  p.x0[0] = 1.0;
  p.lb.assign(static_cast<std::size_t>(n), -kInf);
  p.ub.assign(static_cast<std::size_t>(n), kInf);
  p.objective.name = "zero";
  p.objective.linear.assign(static_cast<std::size_t>(n), 0.0);
  p.objective.hessian.n = n;
  QuadraticFunction h;
  h.name = "unit_sphere";
  h.constant = -1.0;
  h.linear.assign(static_cast<std::size_t>(n), 0.0);
  h.hessian.n = n;
  for (int j = 0; j < n; ++j) h.hessian.entries.push_back({j, j, 2.0});
  p.equalities.push_back(std::move(h));
  return p;
}

QCQPProblem product_of_circles_problem() {
  QCQPProblem p;
  p.name = "condensed_product_of_circles";
  p.n = 4;
  p.x0 = {1.0, 0.0, 1.0, 0.0};
  p.lb.assign(4, -kInf);
  p.ub.assign(4, kInf);
  p.objective.name = "zero";
  p.objective.linear.assign(4, 0.0);
  p.objective.hessian.n = 4;
  p.equalities.push_back(circle_constraint(4, 0, 1, "circle_1"));
  p.equalities.push_back(circle_constraint(4, 2, 3, "circle_2"));
  return p;
}

QCQPProblem coupled_quadratic_manifold_problem() {
  QCQPProblem p;
  p.name = "condensed_coupled_quadratic_manifold";
  p.n = 3;
  p.x0 = {1.0, 0.0, 0.0};
  p.lb.assign(3, -kInf);
  p.ub.assign(3, kInf);
  p.objective.name = "zero";
  p.objective.linear.assign(3, 0.0);
  p.objective.hessian.n = 3;

  // h1=x0*x1+x2=0: exercises a cross-Hessian coefficient and a linear term.
  QuadraticFunction h1;
  h1.name = "cross_and_linear";
  h1.linear = {0.0, 0.0, 1.0};
  h1.hessian.n = 3;
  h1.hessian.entries = {{0, 1, 1.0}};
  p.equalities.push_back(std::move(h1));

  QuadraticFunction h2;
  h2.name = "unit_xy_circle";
  h2.constant = -1.0;
  h2.linear.assign(3, 0.0);
  h2.hessian.n = 3;
  h2.hessian.entries = {{0, 0, 2.0}, {1, 1, 2.0}};
  p.equalities.push_back(std::move(h2));
  return p;
}

QCQPProblem solver_circle_problem() {
  QCQPProblem p;
  p.name = "condensed_solver_circle";
  p.n = 2;
  p.x0 = {1.0, 0.0};
  p.lb = {-2.0, -2.0};
  p.ub = {2.0, 2.0};
  // f(x,y)=(y-1)^2.  The tangent direction at (1,0) is nonzero and
  // the exact retraction keeps every accepted trial on the circle.
  p.objective.name = "f";
  p.objective.constant = 1.0;
  p.objective.linear = {0.0, -2.0};
  p.objective.hessian.n = 2;
  p.objective.hessian.entries = {{1, 1, 2.0}};
  p.equalities.push_back(circle_constraint(2, 0, 1, "unit_circle"));
  return p;
}


QCQPProblem duplicate_circle_equalities_problem() {
  QCQPProblem p = solver_circle_problem();
  p.name = "condensed_duplicate_circle_equalities";
  p.equalities.push_back(p.equalities.front());
  p.equalities.back().name = "unit_circle_duplicate";
  return p;
}

QCQPProblem solver_circle_multiplier_curvature_problem() {
  QCQPProblem p;
  p.name = "condensed_solver_circle_multiplier_curvature";
  p.n = 2;
  p.x0 = {1.0, 0.0};
  p.lb = {-2.0, -2.0};
  p.ub = {2.0, 2.0};
  // At x0, the minimum-residual equality multiplier is lambda=1/2 for the
  // convention grad f + lambda grad h=0 in the normal coordinate.  Hence the
  // equality-Lagrangian Hessian restricted to the y tangent is exactly +1.
  p.objective.name = "linear_with_normal_and_tangent_components";
  p.objective.linear = {-1.0, -0.1};
  p.objective.hessian.n = 2;
  p.equalities.push_back(circle_constraint(2, 0, 1, "unit_circle"));
  return p;
}

QCQPProblem equality_only_negative_tangent_problem() {
  QCQPProblem p;
  p.name = "condensed_equality_only_negative_tangent";
  p.n = 4;
  p.x0 = {1.0, 0.0, 0.0, 0.0};
  p.lb.assign(4, -kInf);
  p.ub.assign(4, kInf);
  // The y direction is tangent to the circle and has negative curvature.  The
  // nonzero tangent linear term forces an actual step.  With no finite bounds
  // and no inequalities, the convex majorant must use the dedicated sparse
  // bordered equality-QP path rather than the general LazyQP IPM.
  p.objective.name = "negative_tangent_curvature";
  p.objective.linear = {0.0, -0.2, 0.0, 0.0};
  p.objective.hessian.n = 4;
  p.objective.hessian.entries = {
      {1, 1, -1.0}, {2, 2, 1.0}, {3, 3, 1.0}};
  p.equalities.push_back(circle_constraint(4, 0, 1, "unit_circle"));
  return p;
}

QCQPProblem ill_conditioned_near_feasible_linear_manifold() {
  QCQPProblem p;
  p.name = "condensed_near_feasible_baseline_ratio";
  p.n = 2;
  // h(x)=1e-3*x0 has residual 5e-10, already inside the configured
  // retraction tolerance, but its fixed-normal correction is 5e-7.
  p.x0 = {5e-7, 0.0};
  p.lb.assign(2, -kInf);
  p.ub.assign(2, kInf);
  p.objective.name = "zero";
  p.objective.linear.assign(2, 0.0);
  p.objective.hessian.n = 2;
  QuadraticFunction h;
  h.name = "small_jacobian_linear_equality";
  h.linear = {1e-3, 0.0};
  h.hessian.n = 2;
  p.equalities.push_back(std::move(h));
  return p;
}

Settings test_settings() {
  Settings st;
  st.linear_backend = linear_backend_compiled("ldl") ? "ldl" : "native";
  st.verbose = false;
  st.minimum_step_length = 1e-13;
  st.dca_condensed_retraction_tolerance = 1e-12;
  st.dca_condensed_retraction_max_ratio = 4.0;
  st.dca_condensed_newton_iterations = 12;
  st.dca_condensed_krylov_iterations = 12;
  st.dca_condensed_krylov_relative_tolerance = 1e-10;
  st.dca_condensed_continuation_attempts = 24;
  st.dca_condensed_lanczos_steps = 12;
  st.dca_condensed_lanczos_starts = 3;
  return st;
}

double convexified_quadratic(const SymMatrix& B,
                             const Convexification& cvx,
                             const Vec& v) {
  double q = B.quad(v) + cvx.scalar_shift * dot(v, v);
  for (const auto& term : cvx.terms) {
    const double a = dot(term.direction, v);
    q += term.beta * a * a;
  }
  return q;
}

int test_scalar_exact_retraction_and_tangent_curvature() {
  const QCQPProblem p = sphere_problem();
  Settings st = test_settings();
  CondensedQuadraticRetraction retraction(st);
  std::string message;
  if (!retraction.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] scalar normal factorization: " << message << "\n";
    return 1;
  }

  Vec projected;
  if (!retraction.project_tangent({4.0, 2.0, -3.0}, projected, message)) {
    std::cerr << "[FAIL] tangent projection: " << message << "\n";
    return 2;
  }
  if (std::abs(projected[0]) > 1e-11) {
    std::cerr << "[FAIL] tangent projector retained a normal component: "
              << projected[0] << "\n";
    return 3;
  }

  const Vec step = {0.0, 0.4, 0.2};
  const auto rr = retraction.retract(step);
  if (!rr.success) {
    std::cerr << "[FAIL] scalar condensed retraction: " << rr.message << "\n";
    return 4;
  }
  if (rr.equality_residual_inf > 5e-12 ||
      std::abs(p.equalities[0].value(rr.x)) > 5e-12) {
    std::cerr << "[FAIL] scalar retraction is not exact: residual="
              << rr.equality_residual_inf << "\n";
    return 5;
  }
  if (!(rr.x[0] > 0.0 && rr.x[0] < 1.0)) {
    std::cerr << "[FAIL] scalar retraction selected the wrong circle branch: x0="
              << rr.x[0] << "\n";
    return 6;
  }

  SymMatrix B;
  B.n = 3;
  B.entries = {{1, 1, -2.0}, {2, 2, 4.0}};
  Convexification cvx;
  const auto tc = retraction.ensure_tangent_convex(B, cvx, 0.0, 4);
  if (!tc.success || (tc.rank_added == 0 && tc.scalar_shift_added == 0.0)) {
    std::cerr << "[FAIL] tangent negative curvature was not corrected: "
              << tc.message << " lambda0=" << tc.initial_min_eigenvalue
              << " lambda1=" << tc.final_min_eigenvalue << "\n";
    return 7;
  }
  const Vec negative_tangent = {0.0, 1.0, 0.0};
  if (convexified_quadratic(B, cvx, negative_tangent) < -1e-8) {
    std::cerr << "[FAIL] corrected model remains negative on a known tangent direction\n";
    return 8;
  }
  return 0;
}

int test_newton_krylov_retraction() {
  const QCQPProblem p = product_of_circles_problem();
  Settings st = test_settings();
  CondensedQuadraticRetraction retraction(st);
  std::string message;
  if (!retraction.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] multi-equality normal factorization: " << message << "\n";
    return 11;
  }
  const auto rr = retraction.retract({0.0, 0.3, 0.0, 0.4});
  if (!rr.success) {
    std::cerr << "[FAIL] Newton--Krylov condensed retraction: " << rr.message << "\n";
    return 12;
  }
  if (rr.newton_iterations <= 0 || rr.krylov_iterations <= 0) {
    std::cerr << "[FAIL] multi-equality path did not exercise Newton--Krylov\n";
    return 13;
  }
  if (rr.equality_residual_inf > 5e-11) {
    std::cerr << "[FAIL] multi-equality exact residual="
              << rr.equality_residual_inf << "\n";
    return 14;
  }


  const QCQPProblem coupled = coupled_quadratic_manifold_problem();
  CondensedQuadraticRetraction coupled_retraction(st);
  if (!coupled_retraction.factor(coupled, coupled.x0, message)) {
    std::cerr << "[FAIL] coupled sparse normal factorization: " << message << "\n";
    return 15;
  }
  const auto coupled_rr = coupled_retraction.retract({0.0, 0.2, -0.2});
  if (!coupled_rr.success || coupled_rr.equality_residual_inf > 5e-11) {
    std::cerr << "[FAIL] cross/linear sparse Jacobian retraction: "
              << coupled_rr.message << " residual="
              << coupled_rr.equality_residual_inf << "\n";
    return 16;
  }
  return 0;
}

int test_near_feasible_baseline_ratio() {
  const QCQPProblem p = ill_conditioned_near_feasible_linear_manifold();
  Settings st = test_settings();
  st.dca_condensed_retraction_tolerance = 1e-9;
  st.dca_condensed_retraction_max_ratio = 2.0;
  CondensedQuadraticRetraction retraction(st);
  std::string message;
  if (!retraction.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] near-feasible baseline factorization: "
              << message << "\n";
    return 28;
  }
  const auto rr = retraction.retract({0.0, 1e-7});
  if (!rr.success || rr.baseline_correction_inf < 4e-7 ||
      rr.incremental_correction_inf > 1e-12 ||
      rr.correction_ratio > st.dca_condensed_retraction_max_ratio) {
    std::cerr << "[FAIL] near-feasible baseline was not removed from the "
                 "retraction ratio: success=" << rr.success
              << " corr=" << rr.correction_inf
              << " base=" << rr.baseline_correction_inf
              << " incremental=" << rr.incremental_correction_inf
              << " ratio=" << rr.correction_ratio
              << " message=" << rr.message << "\n";
    return 29;
  }
  return 0;
}

int test_verified_iterated_scalar_curvature_safeguard() {
  constexpr int n = 65;
  const QCQPProblem p = high_dimensional_sphere_problem(n);
  Settings st = test_settings();
  st.linear_backend = "native";
  st.random_seed = 0;
  st.dca_condensed_lanczos_steps = 2;
  st.dca_condensed_lanczos_starts = 1;
  st.dca_condensed_scalar_safeguard_updates = 8;

  CondensedQuadraticRetraction retraction(st);
  std::string message;
  if (!retraction.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] scalar-safeguard normal factorization: "
              << message << "\n";
    return 17;
  }

  // With seed zero and a two-vector Lanczos space, the historical one-shot
  // safeguard estimates -31.5788, shifts by 34.7367, and a fresh verification
  // exposes residual curvature near -0.5569.  This deterministic case models
  // the CATENA failure without depending on CUTEst.
  SymMatrix B;
  B.n = n;
  for (int j = 1; j < n; ++j)
    B.entries.push_back({j, j, -0.01 * static_cast<double>(j * j)});
  Convexification cvx;
  const auto tc = retraction.ensure_tangent_convex(B, cvx, 0.0, 0);
  if (!tc.success || tc.scalar_safeguard_updates < 2 ||
      tc.final_min_eigenvalue < -10.0 * st.curvature_tolerance) {
    std::cerr << "[FAIL] verified scalar safeguard did not recover the "
                 "one-shot Lanczos miss: updates="
              << tc.scalar_safeguard_updates << " lambda0="
              << tc.initial_min_eigenvalue << " lambda1="
              << tc.final_min_eigenvalue << " shift="
              << tc.scalar_shift_added << " message=" << tc.message << "\n";
    return 18;
  }
  if (!cvx.terms.empty()) {
    std::cerr << "[FAIL] scalar-safeguard test unexpectedly grew the low-rank border\n";
    return 19;
  }
  return 0;
}

int test_equality_only_bordered_low_rank_route() {
  const QCQPProblem p = equality_only_negative_tangent_problem();
  Settings st = test_settings();
  st.outer_algorithm = "lazy-dca";
  st.dca_equality_model = "condensed-retraction";
  st.inner_backend = "lazyqp";
  st.restoration_enabled = false;
  st.restoration_tangent_activation = false;
  st.dca_local_corrector = false;
  st.dca_iterated_soc = false;
  st.second_order_correction = false;
  st.dca_equality_retraction = false;
  st.dca_step_control = false;
  st.dca_extra_proximal_regularization = 1.0;
  st.max_outer_iterations = 1;
  st.max_inner_iterations = 100;
  st.feasibility_tolerance = 1e-8;
  st.stationarity_tolerance = 1e-8;
  st.dual_infeasibility_tolerance = 1e-8;
  st.complementarity_tolerance = 1e-8;
  st.inner_tolerance = 1e-10;
  st.line_search_max_steps = 40;

  const Result r = Solver(st).solve(p);
  const double initial_funnel=std::sqrt(st.feasibility_tolerance);
  if(r.lazy_dca_equality_objective_steps<1 ||
     r.lazy_dca_equality_feasibility_steps!=0 ||
     std::abs(r.lazy_dca_final_equality_funnel-initial_funnel)>
         64.0*std::numeric_limits<double>::epsilon()*initial_funnel) {
    std::cerr << "[FAIL] an objective-only step contracted the equality funnel: "
              << r.lazy_dca_final_equality_funnel << "/" << initial_funnel
              << " objective/h=" << r.lazy_dca_equality_objective_steps
              << "/" << r.lazy_dca_equality_feasibility_steps << "\n";
    return 68;
  }
  if (st.dca_condensed_projected_multi_ritz_inertia ||
      r.lazy_dca_condensed_projected_multi_ritz_batches != 0 ||
      r.lazy_dca_condensed_projected_multi_ritz_fallbacks != 0) {
    std::cerr << "[FAIL] generic default did not preserve the historical "
                 "curvature route\n";
    return 30;
  }
  if (r.lazy_dca_condensed_tangent_rank_max <= 0) {
    std::cerr << "[FAIL] equality-only route did not detect tangent negative curvature: "
              << r.message << "\n";
    return 31;
  }
  const int expected_minimum_dimension = p.n +
      static_cast<int>(p.equalities.size()) +
      r.lazy_dca_condensed_tangent_rank_max;
  if (r.lazyqp_full_kkt_factorizations <= 0 ||
      r.lazyqp_max_full_kkt_dimension < expected_minimum_dimension) {
    std::cerr << "[FAIL] equality-only low-rank correction was not represented by "
                 "the sparse bordered KKT system: fact="
              << r.lazyqp_full_kkt_factorizations << " dimension="
              << r.lazyqp_max_full_kkt_dimension << " expected-at-least="
              << expected_minimum_dimension << " message=" << r.message << "\n";
    return 32;
  }
  if (r.lazy_dca_condensed_accepted_steps <= 0 ||
      r.feasibility.equality_inf > 5e-10) {
    std::cerr << "[FAIL] equality-only bordered route did not accept an exact "
                 "manifold step: accepted="
              << r.lazy_dca_condensed_accepted_steps << " equality="
              << r.feasibility.equality_inf << " status=" << status_string(r.status)
              << " message=" << r.message << "\n";
    return 33;
  }
  if (!(r.objective < p.objective.value(p.x0))) {
    std::cerr << "[FAIL] equality-only bordered manifold step did not reduce the objective\n";
    return 34;
  }
  return 0;
}


int test_fast_exact_same_trajectory_parity() {
  const QCQPProblem p = product_of_circles_problem();
  const Vec step = {0.0, 0.55, 0.0, 0.45};

  Settings historical = test_settings();
  historical.dca_condensed_affine_normal_reuse = false;
  historical.dca_condensed_quadratic_linesearch_reuse = false;
  CondensedQuadraticRetraction baseline(historical);
  std::string message;
  if (!baseline.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] r1 parity baseline factorization: " << message << "\n";
    return 41;
  }
  const auto blind = baseline.retract(step);
  if (!blind.success) {
    std::cerr << "[FAIL] r1 parity historical retraction: " << blind.message << "\n";
    return 42;
  }

  Settings accelerated = historical;
  accelerated.dca_condensed_affine_normal_reuse = true;
  accelerated.dca_condensed_quadratic_linesearch_reuse = true;
  CondensedQuadraticRetraction fast(accelerated);
  if (!fast.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] r1 parity accelerated factorization: " << message << "\n";
    return 43;
  }
  const auto kept = fast.retract(step);
  if (!kept.success) {
    std::cerr << "[FAIL] r1 parity accelerated retraction: " << kept.message << "\n";
    return 44;
  }

  double dx = 0.0;
  for (std::size_t i = 0; i < kept.x.size(); ++i)
    dx = std::max(dx, std::abs(kept.x[i] - blind.x[i]));
  if (dx > 2e-12 ||
      kept.continuation_attempts != blind.continuation_attempts ||
      kept.continuation_acceptances != blind.continuation_acceptances ||
      kept.newton_iterations != blind.newton_iterations ||
      kept.krylov_iterations != blind.krylov_iterations ||
      kept.line_search_backtracks != blind.line_search_backtracks) {
    std::cerr << "[FAIL] r1 algebraic acceleration changed the certified retraction trajectory: dx="
              << dx << " cont=" << blind.continuation_attempts << "->"
              << kept.continuation_attempts << " newton=" << blind.newton_iterations
              << "->" << kept.newton_iterations << " krylov=" << blind.krylov_iterations
              << "->" << kept.krylov_iterations << " backtracks="
              << blind.line_search_backtracks << "->" << kept.line_search_backtracks
              << "\n";
    return 45;
  }
  if (kept.affine_normal_map_reuses <= 0 ||
      kept.quadratic_linesearch_reuses <= 0 ||
      kept.quadratic_linesearch_exact_checks <= 0 ||
      fast.zero_rhs_solve_skips() <= 0) {
    std::cerr << "[FAIL] parity test did not exercise the algebraic and "
                 "proven-zero accelerations: affine="
              << kept.affine_normal_map_reuses << " quadratic="
              << kept.quadratic_linesearch_reuses << "/"
              << kept.quadratic_linesearch_exact_checks << " early/rows="
              << kept.quadratic_linesearch_early_rejections << "/"
              << kept.quadratic_linesearch_rows_skipped << " zero-rhs="
              << fast.zero_rhs_solve_skips() << "\n";
    return 46;
  }
  const auto screened = fast.retract({0.0, 5.0, 0.0, 4.0});
  if (screened.quadratic_linesearch_early_rejections <= 0 ||
      screened.quadratic_linesearch_rows_skipped <= 0) {
    std::cerr << "[FAIL] aggressive parity trial did not exercise the "
                 "quadratic line-search early-rejection screen: early/rows="
              << screened.quadratic_linesearch_early_rejections << "/"
              << screened.quadratic_linesearch_rows_skipped << "\n";
    return 48;
  }
  if (kept.normal_solves > blind.normal_solves) {
    std::cerr << "[FAIL] r1 algebraic reuse increased normal solves: "
              << blind.normal_solves << " -> " << kept.normal_solves << "\n";
    return 47;
  }
  return 0;
}

int test_r2_sparse_parallel_and_cached_regularization_parity() {
  const QCQPProblem p = coupled_quadratic_manifold_problem();
  const Vec step = {0.0, 0.23, -0.23};
  std::string message;

  Settings dense = test_settings();
  dense.dca_condensed_sparse_equality_evaluation = false;
  dense.dca_condensed_cached_regularization = false;
  dense.dca_condensed_parallel_equalities = false;
  CondensedQuadraticRetraction baseline(dense);
  if (!baseline.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] r2 dense parity factorization: " << message << "\n";
    return 51;
  }
  const auto dense_rr = baseline.retract(step);
  if (!dense_rr.success) {
    std::cerr << "[FAIL] r2 dense parity retraction: " << dense_rr.message << "\n";
    return 52;
  }

  Settings sparse = dense;
  sparse.dca_condensed_sparse_equality_evaluation = true;
  CondensedQuadraticRetraction sparse_route(sparse);
  if (!sparse_route.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] r2 sparse parity factorization: " << message << "\n";
    return 53;
  }
  const auto sparse_rr = sparse_route.retract(step);
  if (!sparse_rr.success) {
    std::cerr << "[FAIL] r2 sparse parity retraction: " << sparse_rr.message << "\n";
    return 54;
  }

  auto trajectory_equal = [](const CondensedRetractionResult& a,
                             const CondensedRetractionResult& b) {
    double dx = 0.0;
    for (std::size_t i = 0; i < a.x.size(); ++i)
      dx = std::max(dx, std::abs(a.x[i] - b.x[i]));
    return dx <= 2e-12 &&
        a.continuation_attempts == b.continuation_attempts &&
        a.continuation_acceptances == b.continuation_acceptances &&
        a.newton_iterations == b.newton_iterations &&
        a.krylov_iterations == b.krylov_iterations &&
        a.line_search_backtracks == b.line_search_backtracks;
  };
  if (!trajectory_equal(dense_rr, sparse_rr) ||
      sparse_route.sparse_equality_evaluations() <= 0) {
    std::cerr << "[FAIL] r2 sparse equality evaluator changed the retraction trajectory\n";
    return 55;
  }

  Settings parallel = sparse;
  parallel.dca_condensed_parallel_equalities = true;
  parallel.dca_condensed_parallel_min_equalities = 1;
  CondensedQuadraticRetraction parallel_route(parallel);
  if (!parallel_route.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] r2 parallel parity factorization: " << message << "\n";
    return 56;
  }
  const auto parallel_rr = parallel_route.retract(step);
  if (!parallel_rr.success || !trajectory_equal(sparse_rr, parallel_rr)) {
    std::cerr << "[FAIL] r2 equality-row parallelism changed the retraction trajectory\n";
    return 57;
  }

  // A duplicated equality makes J rank deficient at every point.  The first
  // two factorizations establish the same positive regularization; the third
  // may then seed that value directly.  The resulting operator is the same,
  // but one doomed zero-regularization numerical factorization is removed.
  const QCQPProblem duplicate = duplicate_circle_equalities_problem();
  Settings cached = test_settings();
  cached.dca_condensed_cached_regularization = true;
  cached.dca_condensed_sparse_equality_evaluation = true;
  CondensedQuadraticRetraction cached_route(cached);
  for (int rep = 0; rep < 3; ++rep) {
    if (!cached_route.factor(duplicate, duplicate.x0, message)) {
      std::cerr << "[FAIL] r2 cached regularization factorization: "
                << message << " rep=" << rep << "\n";
      return 58;
    }
  }
  if (cached_route.regularization() <= 0.0) {
    std::cerr << "[FAIL] r2 cached-regularization test did not require a positive safeguard\n";
    return 59;
  }
  if (cached_route.cached_regularization_attempts() <= 0 ||
      cached_route.cached_regularization_hits() <= 0) {
    std::cerr << "[FAIL] r2 positive normal regularization was not reused\n";
    return 60;
  }
  return 0;
}

int test_bound_face_normal_map() {
  Settings st = test_settings();
  st.dca_condensed_bound_aware_normal = true;
  st.inner_tolerance = 1e-12;

  QCQPProblem p;
  p.name = "condensed_bound_face";
  p.n = 3;
  p.x0 = {0.0, 0.0, 0.0};
  p.lb = {0.0, -kInf, -kInf};
  p.ub = {kInf, kInf, kInf};
  p.objective.name = "zero";
  p.objective.linear.assign(3, 0.0);
  p.objective.hessian.n = 3;
  QuadraticFunction h;
  h.name = "x0_plus_x1_plus_x2_squared";
  h.linear = {1.0, 1.0, 0.0};
  h.hessian.n = 3;
  h.hessian.entries = {{2, 2, 2.0}};
  p.equalities.push_back(std::move(h));
  p.validate();

  const Vec step = {0.0, 0.0, 0.2};
  std::string message;
  CondensedQuadraticRetraction historical(st);
  if (!historical.factor(p, p.x0, message)) {
    std::cerr << "[FAIL] bound-face historical factorization: " << message << "\n";
    return 61;
  }
  const auto leaking = historical.retract(step);
  if (!leaking.success || !(leaking.feasibility.bound_inf > 1e-3)) {
    std::cerr << "[FAIL] synthetic bound-face case no longer reproduces the historical leak\n";
    return 62;
  }

  CondensedQuadraticRetraction guarded(st);
  if (!guarded.factor_bound_aware(p, p.x0, message)) {
    std::cerr << "[FAIL] bound-face guarded factorization: " << message << "\n";
    return 63;
  }
  const auto kept = guarded.retract(step);
  if (!kept.success || guarded.locked_coordinates() != 1 ||
      kept.feasibility.bound_inf != 0.0 ||
      kept.equality_residual_inf > 1e-12 ||
      std::abs(kept.x[0]) > 1e-14 ||
      std::abs(kept.x[1] + 0.04) > 1e-11 ||
      std::abs(kept.x[2] - 0.2) > 1e-14) {
    std::cerr << "[FAIL] bound-face normal map did not preserve the active face: "
              << kept.message << " x=" << kept.x[0] << "," << kept.x[1]
              << "," << kept.x[2] << " eq=" << kept.equality_residual_inf
              << " bound=" << kept.feasibility.bound_inf << " locks="
              << guarded.locked_coordinates() << "\n";
    return 64;
  }

  // If the only first-order equality normal is the locked coordinate, the
  // active-face map has no local right inverse.  It must reject cleanly rather
  // than return the historical bound-infeasible branch as a false success.
  QCQPProblem rank_loss = p;
  rank_loss.name = "condensed_bound_face_rank_loss";
  rank_loss.n = 2;
  rank_loss.x0 = {0.0, 0.0};
  rank_loss.lb = {0.0, -kInf};
  rank_loss.ub = {kInf, kInf};
  rank_loss.objective.linear.assign(2, 0.0);
  rank_loss.objective.hessian.n = 2;
  rank_loss.equalities.clear();
  QuadraticFunction singular;
  singular.name = "x0_plus_x1_squared";
  singular.linear = {1.0, 0.0};
  singular.hessian.n = 2;
  singular.hessian.entries = {{1, 1, 2.0}};
  rank_loss.equalities.push_back(std::move(singular));
  rank_loss.validate();
  const Vec rank_step = {0.0, 0.2};
  CondensedQuadraticRetraction no_range(st);
  if (!no_range.factor_bound_aware(rank_loss, rank_loss.x0, message)) {
    std::cerr << "[FAIL] rank-loss normal map did not reach guarded solve: "
              << message << "\n";
    return 65;
  }
  const auto rejected = no_range.retract(rank_step);
  if (rejected.success) {
    std::cerr << "[FAIL] rank-loss bound map returned a false successful retraction\n";
    return 66;
  }
  return 0;
}

int test_solver_route() {
  const QCQPProblem p = solver_circle_problem();
  Settings st = test_settings();
  st.outer_algorithm = "lazy-dca";
  st.dca_equality_model = "condensed-retraction";
  st.inner_backend = "lazyqp";
  st.restoration_enabled = false;
  st.restoration_tangent_activation = false;
  st.dca_local_corrector = false;
  st.dca_iterated_soc = false;
  st.second_order_correction = false;
  st.dca_equality_retraction = false;  // generic SOC retraction must be irrelevant
  st.dca_equality_funnel_auto = true;
  st.max_outer_iterations = 1;
  st.max_inner_iterations = 500;
  st.feasibility_tolerance = 1e-7;
  st.stationarity_tolerance = 1e-7;
  st.dual_infeasibility_tolerance = 1e-7;
  st.complementarity_tolerance = 1e-7;
  st.inner_tolerance = 1e-10;
  st.line_search_max_steps = 40;

  const Result r = Solver(st).solve(p);
  if (r.effective_outer_algorithm != "lazy-dca" ||
      r.lazy_dca_equality_model != "condensed-retraction") {
    std::cerr << "[FAIL] wrong solver route/model: route="
              << r.effective_outer_algorithm << " model="
              << r.lazy_dca_equality_model << "\n";
    return 21;
  }
  if (r.lazy_dca_condensed_factorizations <= 0 ||
      r.lazy_dca_condensed_retraction_attempts <= 0 ||
      r.lazy_dca_condensed_retraction_acceptances <= 0 ||
      r.lazy_dca_condensed_accepted_steps <= 0) {
    std::cerr << "[FAIL] condensed solver path not exercised: factors="
              << r.lazy_dca_condensed_factorizations << " attempts="
              << r.lazy_dca_condensed_retraction_attempts << " retracts="
              << r.lazy_dca_condensed_retraction_acceptances << " accepted-steps="
              << r.lazy_dca_condensed_accepted_steps << " status="
              << status_string(r.status) << " message=" << r.message << "\n";
    return 22;
  }
  if (r.lazy_dca_condensed_bound_face_factorizations != 0 ||
      r.lazy_dca_condensed_max_active_bound_coordinates != 0) {
    std::cerr << "[FAIL] interior condensed route built an unnecessary bound-face factor\n";
    return 67;
  }
  if (r.soc_factorizations != 0 || r.lazy_dca_corrector_attempts != 0) {
    std::cerr << "[FAIL] condensed route stacked the legacy SOC/corrector\n";
    return 23;
  }
  if (r.feasibility.equality_inf > 2e-10) {
    std::cerr << "[FAIL] solver accepted an off-manifold point: eq="
              << r.feasibility.equality_inf << "\n";
    return 24;
  }
  if (!(r.objective < p.objective.value(p.x0))) {
    std::cerr << "[FAIL] condensed solver step did not improve the objective: f="
              << r.objective << "\n";
    return 25;
  }


  const QCQPProblem multiplier_problem =
      solver_circle_multiplier_curvature_problem();
  Settings multiplier_settings = st;
  multiplier_settings.max_outer_iterations = 1;
  const Result multiplier_result = Solver(multiplier_settings).solve(
      multiplier_problem);
  if (!(multiplier_result.lazy_dca_condensed_initial_tangent_min_eigenvalue > 0.5 &&
        multiplier_result.lazy_dca_condensed_initial_tangent_min_eigenvalue < 1.5)) {
    std::cerr << "[FAIL] signed equality multiplier curvature was not used: lambda_tan="
              << multiplier_result.lazy_dca_condensed_initial_tangent_min_eigenvalue
              << " status=" << status_string(multiplier_result.status)
              << " message=" << multiplier_result.message << "\n";
    return 26;
  }
  if (multiplier_result.lazy_dca_condensed_tangent_rank_max != 0 ||
      multiplier_result.lazy_dca_condensed_tangent_shift_max != 0.0) {
    std::cerr << "[FAIL] positive signed equality curvature was unnecessarily corrected\n";
    return 27;
  }
  return 0;
}

} // namespace

int main() {
  if (const int rc = test_scalar_exact_retraction_and_tangent_curvature()) return rc;
  if (const int rc = test_newton_krylov_retraction()) return rc;
  if (const int rc = test_near_feasible_baseline_ratio()) return rc;
  if (const int rc = test_verified_iterated_scalar_curvature_safeguard()) return rc;
  if (const int rc = test_equality_only_bordered_low_rank_route()) return rc;
  if (const int rc = test_fast_exact_same_trajectory_parity()) return rc;
  if (const int rc = test_r2_sparse_parallel_and_cached_regularization_parity()) return rc;
  if (const int rc = test_bound_face_normal_map()) return rc;
  if (const int rc = test_solver_route()) return rc;
  std::cout << "[PASS] condensed quadratic retraction: exact scalar branch, "
               "Newton--Krylov multi-equality branch, tangent curvature, "
               "verified iterative scalar curvature safeguard, "
               "sparse bordered equality-QP, r1/r2 same-trajectory parity, "
               "near-feasible incremental correction guard, cached regularization, "
               "proven-zero screens, optional equality-row parallelism, bound-face normal map, "
               "and Lazy-DCA integration\n";
  return 0;
}
