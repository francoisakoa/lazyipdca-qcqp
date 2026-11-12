#include "lazyipdca_qcqp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

QuadraticFunction quadratic(int n, const std::string& name,
                            double constant = 0.0) {
  QuadraticFunction q;
  q.name = name;
  q.constant = constant;
  q.linear.assign(static_cast<std::size_t>(n), 0.0);
  q.hessian.n = n;
  return q;
}

// Compact OPF analogue.  x and y are physical coordinates, while p and q
// are the strictly named apparent-flow auxiliaries.  At the feasible origin,
// the tangent flow equations give dp=dq=0.  The first retraction therefore
// exposes slim_f_0 only after p=x^2 is restored, exercising the guarded
// post-retraction working-set expansion.  The negative x curvature also
// forces the projected multi-Ritz proposal through its exact inertia gate.
QCQPProblem compact_apparent_flow_problem() {
  constexpr int n = 4;
  constexpr int x = 0;
  constexpr int y = 1;
  constexpr int p_flow = 2;
  constexpr int q_flow = 3;
  const double inf = std::numeric_limits<double>::infinity();

  QCQPProblem problem;
  problem.name = "pglib_structural_hotpaths";
  problem.n = n;
  problem.x0.assign(n, 0.0);
  problem.lb = {-1.0, -1.0, -inf, -inf};
  problem.ub = {1.0, 1.0, inf, inf};

  problem.objective = quadratic(n, "objective");
  problem.objective.linear[x] = -1.0;
  problem.objective.hessian.entries.push_back({x, x, -0.2});
  problem.objective.hessian.entries.push_back({y, y, 2.0});

  // p-x^2=0 and q-y^2=0.  Each auxiliary is unbounded, objective-free,
  // occurs in exactly one equality and in its unique matching slim row.
  auto flow_p = quadratic(n, "flowP_f_0");
  flow_p.linear[p_flow] = 1.0;
  flow_p.hessian.entries.push_back({x, x, -2.0});
  problem.equalities.push_back(std::move(flow_p));

  auto flow_q = quadratic(n, "flowQ_f_0");
  flow_q.linear[q_flow] = 1.0;
  flow_q.hessian.entries.push_back({y, y, -2.0});
  problem.equalities.push_back(std::move(flow_q));

  // p^2+q^2 <= (1/4)^2.  The solution is
  // (x,y,p,q)=(1/2,0,1/4,0), with a regular original-QCQP KKT point.
  auto slim = quadratic(n, "slim_f_0", -0.0625);
  slim.hessian.entries.push_back({p_flow, p_flow, 2.0});
  slim.hessian.entries.push_back({q_flow, q_flow, 2.0});
  problem.inequalities.push_back(std::move(slim));

  problem.validate();
  return problem;
}

Settings hotpath_settings() {
  Settings settings;
  settings.verbose = false;
  settings.outer_algorithm = "lazy-dca";
  settings.inner_backend = "lazyqp";
  settings.linear_backend = "native";
  settings.dca_equality_model = "condensed-retraction";

  settings.lazyqp_flow_lift_elimination = true;
  settings.dca_condensed_flow_lift_schur = true;
  settings.dca_condensed_projected_multi_ritz_inertia = true;
  settings.dca_condensed_post_retraction_working_set_expansion = true;
  settings.dca_condensed_post_retraction_working_set_max_expansions = 4;
  settings.dca_condensed_tangent_curvature = true;

  settings.restoration_enabled = false;
  settings.restoration_tangent_activation = false;
  settings.dca_step_control = false;
  settings.dca_local_corrector = false;
  settings.dca_iterated_soc = false;
  settings.second_order_correction = false;
  settings.terminal_kkt_enabled = true;
  settings.dca_condensed_alpha_predictor = false;

  settings.max_outer_iterations = 200;
  settings.max_inner_iterations = 1000;
  settings.max_working_set_expansions = 8;
  settings.line_search_max_steps = 50;
  settings.time_limit_seconds = 30.0;
  settings.feasibility_tolerance = 1e-7;
  settings.stationarity_tolerance = 1e-6;
  settings.dual_infeasibility_tolerance = 1e-6;
  settings.complementarity_tolerance = 1e-6;
  settings.inner_tolerance = 1e-10;
  settings.working_set_activation = 1e-4;
  settings.dca_condensed_retraction_tolerance = 1e-11;
  settings.dca_condensed_retraction_max_ratio = 8.0;
  settings.dca_condensed_newton_iterations = 16;
  settings.dca_condensed_krylov_iterations = 16;
  settings.dca_condensed_krylov_relative_tolerance = 1e-10;
  settings.dca_condensed_continuation_attempts = 24;
  settings.dca_condensed_lanczos_steps = 12;
  settings.dca_condensed_lanczos_starts = 2;
  settings.dca_condensed_max_tangent_rank = 4;
  settings.dca_condensed_scalar_safeguard_updates = 8;
  settings.exact_eigen_threshold = 1;
  settings.curvature_safety = 1e-8;
  settings.random_seed = 7;
  return settings;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    const QCQPProblem problem = compact_apparent_flow_problem();
    const Settings settings = hotpath_settings();
    const Result result = Solver(settings).solve(problem);

    require(result.status == Status::Solved,
            std::string("combined structural route did not solve: ") +
                status_string(result.status) + " / " + result.message);
    require(result.feasibility.equality_inf <= settings.feasibility_tolerance &&
                result.feasibility.inequality_inf <=
                    settings.feasibility_tolerance &&
                result.feasibility.bound_inf <= settings.feasibility_tolerance,
            "combined route violated an original constraint");
    require(std::isfinite(result.dual_infeasibility_inf) &&
                result.dual_infeasibility_inf <=
                    settings.dual_infeasibility_tolerance &&
                std::isfinite(result.complementarity_inf) &&
                result.complementarity_inf <=
                    settings.complementarity_tolerance,
            "combined route failed the original-QCQP KKT certificate");

    require(result.lazy_dca_condensed_flow_lift_schur_detected_variables == 2,
            "flow-lift Schur detector did not recognize p/q");
    require(result.lazy_dca_condensed_flow_lift_schur_factor_dimension == 2,
            "flow-lift Schur factor did not reduce 6 to 2");
    require(result.lazy_dca_condensed_flow_lift_schur_factorizations > 0 &&
                result.lazy_dca_condensed_flow_lift_schur_fallbacks == 0,
            "flow-lift Schur route fell back");

    require(result.lazyqp_affine_elimination_attempts > 0 &&
                result.lazyqp_affine_elimination_acceptances > 0 &&
                result.lazyqp_affine_elimination_fallbacks == 0,
            "exact LazyQP flow-lift elimination was not accepted cleanly");
    require(result.lazyqp_affine_eliminated_variables == 2 &&
                result.lazyqp_affine_eliminated_equalities == 2 &&
                result.lazyqp_affine_reduced_variables == 2 &&
                result.lazyqp_affine_reduced_equalities == 0,
            "exact affine elimination reported wrong reduced dimensions");
    // With two physical bounds on each side and the active slim row, the
    // reduced full KKT has dimension 2+0+5=7.  The unreduced counterpart
    // would have dimension 4+2+5=11.
    require(result.lazyqp_max_full_kkt_dimension <= 7,
            "LazyQP factorization leaked the full-dimensional flow lifts: " +
                std::to_string(result.lazyqp_max_full_kkt_dimension));

    require(result.lazy_dca_condensed_projected_multi_ritz_batches > 0 &&
                result.lazy_dca_condensed_projected_multi_ritz_fallbacks == 0 &&
                result.lazy_dca_condensed_tangent_inertia_attempts > 0 &&
                result.lazy_dca_condensed_tangent_inertia_acceptances > 0,
            "projected multi-Ritz/inertia route was not accepted cleanly");

    // This compact geometry is intentionally constructed to expose slim only
    // after the first nonlinear equality retraction.
    require(result.lazy_dca_condensed_post_retraction_omitted_offenders > 0 &&
                result.lazy_dca_condensed_post_retraction_working_set_expansions >
                    0 &&
                result.lazy_dca_condensed_post_retraction_expansion_fallbacks ==
                    0,
            "post-retraction working-set expansion was not exercised cleanly");

    require(std::abs(result.x[0] - 0.5) <= 2e-5 &&
                std::abs(result.x[1]) <= 2e-5 &&
                std::abs(result.x[2] - 0.25) <= 2e-5 &&
                std::abs(result.x[3]) <= 2e-5,
            "combined route converged to the wrong compact OPF point");

    // r3p20 route: the exact ambient sparse-block certificate removes every
    // projected curvature application.  The retraction object must therefore
    // be factored even with no active bound face, and it must be the only
    // normal operator used by the solve.
    Settings block_settings = hotpath_settings();
    block_settings.dca_condensed_projected_multi_ritz_inertia = false;
    block_settings.dca_condensed_block_exact_curvature = true;
    block_settings.exact_eigen_threshold = 2;
    const Result block_result = Solver(block_settings).solve(problem);
    require(block_result.status == Status::Solved,
            std::string("block-exact single-normal route did not solve: ") +
                status_string(block_result.status) + " / " +
                block_result.message);
    require(block_result.feasibility.total_inf <=
                block_settings.feasibility_tolerance &&
                block_result.dual_infeasibility_inf <=
                block_settings.dual_infeasibility_tolerance &&
                block_result.complementarity_inf <=
                block_settings.complementarity_tolerance,
            "block-exact route failed the original-QCQP certificate");
    require(block_result.lazy_dca_condensed_block_exact_attempts > 0 &&
                block_result.lazy_dca_condensed_block_exact_acceptances ==
                    block_result.lazy_dca_condensed_block_exact_attempts &&
                block_result.lazy_dca_condensed_block_exact_max_block_dimension <=
                    block_settings.exact_eigen_threshold,
            "block-exact certificate was not accepted on every tangent model");
    require(block_result.lazy_dca_condensed_geometry_normal_solves == 0,
            "single-normal route still applied the geometry normal map");
    require(block_result.lazy_dca_condensed_retraction_normal_solves > 0 &&
                block_result.lazy_dca_condensed_factorizations > 0,
            "single-normal route did not build and apply its retraction map");
    require(block_result.lazy_dca_condensed_projected_multi_ritz_batches == 0 &&
                block_result.lazy_dca_condensed_tangent_inertia_attempts == 0,
            "single-normal route leaked into projected curvature/inertia");
    require(std::abs(block_result.x[0] - 0.5) <= 2e-5 &&
                std::abs(block_result.x[1]) <= 2e-5 &&
                std::abs(block_result.x[2] - 0.25) <= 2e-5 &&
                std::abs(block_result.x[3]) <= 2e-5,
            "block-exact route converged to the wrong compact OPF point");

    std::cout << "[PASS] PGLib structural hotpaths"
              << " obj=" << result.objective
              << " feas=" << result.feasibility.total_inf
              << " dual=" << result.dual_infeasibility_inf
              << " schur="
              << result.lazy_dca_condensed_flow_lift_schur_factor_dimension
              << " elim=" << result.lazyqp_affine_eliminated_variables
              << " reduced=" << result.lazyqp_affine_reduced_variables << "/"
              << result.lazyqp_affine_reduced_equalities
              << " ritz="
              << result.lazy_dca_condensed_projected_multi_ritz_batches
              << " ws="
              << result.lazy_dca_condensed_post_retraction_working_set_expansions
              << " block="
              << block_result.lazy_dca_condensed_block_exact_acceptances
              << " geometry_solves="
              << block_result.lazy_dca_condensed_geometry_normal_solves
              << " outer/inner=" << result.outer_iterations << "/"
              << result.inner_iterations << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
