#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lazyipdca::qcqp;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool close(double a, double b, double tolerance = 2e-11) {
  return std::isfinite(a) && std::isfinite(b) &&
      std::abs(a - b) <= tolerance *
          std::max({1.0, std::abs(a), std::abs(b)});
}

double max_difference(const Vec& a, const Vec& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    result = std::max(result, std::abs(a[i] - b[i]));
  return result;
}

double row_dot(const SparseRow& row, const Vec& x) {
  double result = 0.0;
  for (const auto& [j, value] : row.entries)
    result += value * x[static_cast<std::size_t>(j)];
  return result;
}

Vec transpose_transform(const ExactAffineSubproblemReduction& reduction,
                        const Vec& full) {
  Vec mapped(static_cast<std::size_t>(reduction.reduced.nvar), 0.0);
  require(static_cast<int>(full.size()) == reduction.full_variable_dimension,
          "invalid full vector in transpose transform");
  for (int j = 0; j < reduction.full_variable_dimension; ++j)
    for (const auto& [k, value] :
         reduction.transform_rows[static_cast<std::size_t>(j)].entries)
      mapped[static_cast<std::size_t>(k)] +=
          value * full[static_cast<std::size_t>(j)];
  return mapped;
}

Vec stationarity(const ConvexSubproblem& problem, const Vec& primal,
                 const Vec& equality_duals, const Vec& inequality_duals) {
  require(equality_duals.size() == problem.equalities.rows.size(),
          "equality dual dimension mismatch");
  require(inequality_duals.size() == problem.inequalities.size(),
          "inequality dual dimension mismatch");
  Vec residual = problem.objective.gradient(primal);
  for (std::size_t r = 0; r < problem.equalities.rows.size(); ++r)
    for (const auto& [j, value] : problem.equalities.rows[r].entries)
      residual[static_cast<std::size_t>(j)] +=
          equality_duals[r] * value;
  for (std::size_t i = 0; i < problem.inequalities.size(); ++i) {
    const Vec gradient = problem.inequalities[i].gradient(primal);
    axpy(inequality_duals[i], gradient, residual);
  }
  return residual;
}

ConvexSubproblem make_problem() {
  ConvexSubproblem problem;
  problem.nvar = 4;
  problem.original_dimension = 4;
  problem.prefer_feasible_centered_start = true;

  // This deliberately exercises every representation carried by a
  // ConvexQuadraticModel.  The prefix shift includes the +1 flow pivot but
  // excludes the -1 pivot, so composing it cannot accidentally look like a
  // scalar identity in the two retained coordinates.
  problem.objective.constant = 0.7;
  problem.objective.linear = {-0.8, 0.3, 0.2, -0.4};
  problem.objective.base_hessian.n = 4;
  problem.objective.base_hessian.entries = {
      {0, 0, 4.0}, {0, 1, 0.25}, {0, 2, -0.3},
      {1, 1, 3.0}, {1, 3, 0.2},
      {2, 2, 2.0}, {2, 3, 0.1}, {3, 3, 2.5}};
  problem.objective.shift_dimension = 3;
  problem.objective.scalar_shift = 0.4;
  problem.objective.low_rank.push_back(
      {0.7, {0.5, -0.25, 0.75, -0.4}});

  problem.equalities.nvar = 4;
  // Positive pivot: y_2 = 0.5 - 2 y_0 + y_1.
  problem.equalities.rows.push_back({{{0, 2.0}, {1, -1.0}, {2, 1.0}}});
  problem.equalities.rhs.push_back(0.5);
  // Negative pivot: y_3 = 0.25 - y_0 + 3 y_1.
  problem.equalities.rows.push_back({{{0, -1.0}, {1, 3.0}, {3, -1.0}}});
  problem.equalities.rhs.push_back(-0.25);
  // One retained physical equality.
  problem.equalities.rows.push_back({{{0, 0.75}, {1, -1.25}}});
  problem.equalities.rhs.push_back(0.1);

  ConvexQuadraticModel disk;
  disk.constant = -10.0;
  disk.linear = {0.1, -0.2, 0.05, -0.1};
  disk.base_hessian.n = 4;
  disk.base_hessian.entries = {
      {0, 0, 0.2}, {0, 3, 0.1}, {1, 1, 0.3},
      {2, 2, 1.4}, {3, 3, 1.6}};
  disk.shift_dimension = 4;
  disk.scalar_shift = 0.15;
  disk.low_rank.push_back({0.6, {-0.2, 0.4, 0.5, -0.3}});
  problem.inequalities.push_back(std::move(disk));

  // Consistent primal continuation and deliberately distinct dual entries
  // make index/order mistakes in the warm-start maps visible.
  problem.has_primal_warm_start = true;
  problem.initial_primal = {0.11, -0.07, 0.21, -0.07};
  problem.has_dual_warm_start = true;
  problem.initial_equality_duals = {0.8, -0.6, 0.25};
  problem.initial_inequality_duals = {0.4};
  problem.exact_affine_eliminations = {{0, 2}, {1, 3}};
  return problem;
}

void check_model_parity(const ConvexQuadraticModel& full,
                        const ConvexQuadraticModel& reduced,
                        const ExactAffineSubproblemReduction& map,
                        const Vec& z, const std::string& label) {
  const Vec y = map.expand_primal(z);
  require(close(full.value(y), reduced.value(z)),
          label + " value changed under exact affine substitution");
  const Vec expected = transpose_transform(map, full.gradient(y));
  const Vec actual = reduced.gradient(z);
  require(max_difference(expected, actual) <= 2e-11,
          label + " gradient violates the T^T chain rule");
}

Vec reconstruct_full_equality_duals(
    const ConvexSubproblem& full,
    const ExactAffineSubproblemReduction& reduction,
    const Vec& full_primal, const Vec& reduced_equality_duals,
    const Vec& inequality_duals) {
  Vec lambda(full.equalities.rows.size(), 0.0);
  require(reduced_equality_duals.size() ==
              reduction.retained_equalities.size(),
          "invalid reduced equality dual dimension");
  for (std::size_t k = 0; k < reduction.retained_equalities.size(); ++k)
    lambda[static_cast<std::size_t>(reduction.retained_equalities[k])] =
        reduced_equality_duals[k];

  // Before adding the eliminated-row multipliers, the pivot coordinate of
  // stationarity is independent of every other equality by the validated
  // diagonal-elimination contract.  Its multiplier is therefore exact.
  Vec partial = stationarity(full, full_primal, lambda, inequality_duals);
  for (const auto& elimination : reduction.eliminations) {
    double pivot = 0.0;
    for (const auto& [j, value] : full.equalities.rows[
             static_cast<std::size_t>(elimination.equality_row)].entries)
      if (j == elimination.variable) pivot += value;
    require(pivot == 1.0 || pivot == -1.0,
            "recovery saw a noncanonical affine pivot");
    lambda[static_cast<std::size_t>(elimination.equality_row)] =
        -partial[static_cast<std::size_t>(elimination.variable)] / pivot;
  }
  return lambda;
}

void expect_rejected(ConvexSubproblem problem, const std::string& label) {
  ExactAffineSubproblemReduction reduction;
  reduction.reduced.nvar = 123; // prove that failure clears stale output
  std::string message;
  require(!build_exact_affine_subproblem_reduction(
              problem, reduction, message),
          label + " was not rejected");
  require(!message.empty(), label + " produced no failure diagnostic");
  require(reduction.reduced.nvar == 0 && reduction.offset.empty() &&
              reduction.eliminations.empty(),
          label + " left a partially usable reduction behind");
}

} // namespace

int main() {
  try {
    const ConvexSubproblem full = make_problem();
    ExactAffineSubproblemReduction reduction;
    std::string message;
    require(build_exact_affine_subproblem_reduction(
                full, reduction, message),
            "exact affine reduction failed: " + message);

    require(reduction.full_variable_dimension == 4 &&
                reduction.reduced.nvar == 2,
            "wrong full/reduced variable dimensions");
    require(reduction.full_equality_dimension == 3 &&
                reduction.reduced.equalities.rows.size() == 1,
            "wrong full/reduced equality dimensions");
    require(reduction.retained_variables == std::vector<int>({0, 1}) &&
                reduction.retained_equalities == std::vector<int>({2}),
            "retained variable/equality order changed");
    require(max_difference(reduction.offset, {0.0, 0.0, 0.5, 0.25}) == 0.0,
            "nonzero affine offset was assembled incorrectly");
    require(reduction.reduced.objective.shift_dimension == 0 &&
                reduction.reduced.objective.scalar_shift == 0.0,
            "prefix scalar shift was not composed into the reduced Hessian");
    require(reduction.reduced.objective.low_rank.size() == 1 &&
                reduction.reduced.inequalities[0].low_rank.size() == 1,
            "low-rank curvature was not preserved by congruence");

    const Vec z = {0.37, -0.19};
    const Vec y = reduction.expand_primal(z);
    require(max_difference(y, {0.37, -0.19, -0.43, -0.69}) <= 1e-15,
            "positive/negative pivot expansion is wrong");
    require(reduction.expand_primal({1.0}).empty(),
            "dimension-invalid expansion did not fail closed");
    check_model_parity(full.objective, reduction.reduced.objective,
                       reduction, z, "objective");
    for (std::size_t i = 0; i < full.inequalities.size(); ++i)
      check_model_parity(full.inequalities[i],
                         reduction.reduced.inequalities[i], reduction, z,
                         "inequality " + std::to_string(i));

    for (int r = 0; r < 2; ++r)
      require(std::abs(row_dot(full.equalities.rows[static_cast<std::size_t>(r)], y) -
                       full.equalities.rhs[static_cast<std::size_t>(r)]) <= 1e-15,
              "expanded primal violates an eliminated equality");
    const double full_core = row_dot(full.equalities.rows[2], y) -
        full.equalities.rhs[2];
    const double reduced_core = row_dot(reduction.reduced.equalities.rows[0], z) -
        reduction.reduced.equalities.rhs[0];
    require(close(full_core, reduced_core),
            "retained equality residual changed under substitution");

    require(reduction.reduced.has_primal_warm_start &&
                max_difference(reduction.reduced.initial_primal,
                               {0.11, -0.07}) == 0.0,
            "primal warm start was not restricted in retained order");
    require(reduction.reduced.has_dual_warm_start &&
                max_difference(reduction.reduced.initial_equality_duals,
                               {0.25}) == 0.0 &&
                max_difference(reduction.reduced.initial_inequality_duals,
                               {0.4}) == 0.0,
            "dual warm start was not mapped in retained/original order");
    require(reduction.reduced.exact_affine_eliminations.empty(),
            "reduced problem retained recursive elimination hints");

    // A direct continuation can originate at a point with a small residual
    // in the rows selected for exact pivot elimination.  Restricting its
    // retained coordinates then moves the eliminated variables.  Its
    // independent slacks must move by the opposite inequality-value defect so
    // that g+s is invariant; copying s verbatim was the case118 warm-state
    // rejection in the first single-loop implementation.
    ConvexSubproblem slack_transport_problem = make_problem();
    slack_transport_problem.initial_primal[2] += 0.09;
    slack_transport_problem.initial_primal[3] -= 0.06;
    slack_transport_problem.has_inequality_slack_warm_start = true;
    slack_transport_problem.initial_inequality_slacks = {10.0};
    ExactAffineSubproblemReduction slack_transport;
    require(build_exact_affine_subproblem_reduction(
                slack_transport_problem, slack_transport, message),
            "slack-transport affine reduction failed: " + message);
    require(slack_transport.reduced.has_inequality_slack_warm_start &&
                slack_transport.reduced.initial_inequality_slacks.size() == 1,
            "positive affine slack transport was discarded");
    const double full_warm_rpri =
        slack_transport_problem.inequalities[0].value(
            slack_transport_problem.initial_primal) +
        slack_transport_problem.initial_inequality_slacks[0];
    const double reduced_warm_rpri =
        slack_transport.reduced.inequalities[0].value(
            slack_transport.reduced.initial_primal) +
        slack_transport.reduced.initial_inequality_slacks[0];
    require(close(full_warm_rpri, reduced_warm_rpri, 2e-13),
            "affine reduction changed the transported primal-slack residual");
    require(std::abs(slack_transport.reduced.initial_inequality_slacks[0] -
                     slack_transport_problem.initial_inequality_slacks[0]) >
                1e-4,
            "slack-transport fixture did not move the eliminated primal");

    // KKT covariance at an arbitrary state: after exact recovery of the two
    // discarded equality multipliers, the full stationarity vector is the
    // reduced stationarity on retained coordinates and zero on both pivots.
    const Vec reduced_lambda = {-0.35};
    const Vec inequality_duals = {0.27};
    const Vec reduced_stationarity = stationarity(
        reduction.reduced, z, reduced_lambda, inequality_duals);
    const Vec full_lambda = reconstruct_full_equality_duals(
        full, reduction, y, reduced_lambda, inequality_duals);
    const Vec full_stationarity = stationarity(
        full, y, full_lambda, inequality_duals);
    require(max_difference(full_stationarity,
                           {reduced_stationarity[0],
                            reduced_stationarity[1], 0.0, 0.0}) <= 3e-11,
            "full/reduced KKT stationarity is not covariant");

    // Black-box parity uses the production kernel on the unchanged full model
    // and on the production-built reduced model.  Hints are cleared from the
    // reference so this remains an independent baseline once the opt-in solve
    // wrapper starts consuming them.
    ConvexSubproblem full_reference = full;
    full_reference.exact_affine_eliminations.clear();
    full_reference.has_primal_warm_start = false;
    full_reference.has_dual_warm_start = false;
    full_reference.initial_primal.clear();
    full_reference.initial_equality_duals.clear();
    full_reference.initial_inequality_duals.clear();
    ConvexSubproblem reduced_problem = reduction.reduced;
    reduced_problem.has_primal_warm_start = false;
    reduced_problem.has_dual_warm_start = false;
    reduced_problem.initial_primal.clear();
    reduced_problem.initial_equality_duals.clear();
    reduced_problem.initial_inequality_duals.clear();

    Settings settings;
    settings.verbose = false;
    settings.linear_backend = "native";
    settings.lazyqp_kkt_form = "full";
    settings.lazyqp_low_rank_form = "condensed";
    settings.max_inner_iterations = 250;
    settings.inner_tolerance = 1e-9;
    settings.inner_iteration_limit_mu_fallback = 1e-7;
    const InnerResult full_result =
        ConvexQCQPLazyQPKernel(settings).solve(full_reference);
    const InnerResult reduced_result =
        ConvexQCQPLazyQPKernel(settings).solve(reduced_problem);
    require(full_result.success && reduced_result.success,
            "full/reduced LazyQP solve failed: full='" + full_result.message +
                "' reduced='" + reduced_result.message + "'");
    const Vec expanded_solution = reduction.expand_primal(reduced_result.y);
    require(max_difference(full_result.y, expanded_solution) <= 3e-6,
            "exact affine reduction changed the LazyQP minimizer");
    require(close(full.objective.value(full_result.y),
                  full.objective.value(expanded_solution), 3e-6),
            "exact affine reduction changed the solved objective");
    require(full_result.lazyqp_full_kkt_dimension == 8 &&
                reduced_result.lazyqp_full_kkt_dimension == 4,
            "test did not exercise the expected 8-to-4 KKT reduction");

    ConvexSubproblem default_off_problem = full_reference;
    default_off_problem.exact_affine_eliminations =
        full.exact_affine_eliminations;
    const InnerResult default_off_result =
        ConvexQCQPLazyQPKernel(settings).solve(default_off_problem);
    require(default_off_result.success &&
                default_off_result.lazyqp_affine_elimination_attempts == 0 &&
                default_off_result.lazyqp_affine_elimination_acceptances == 0 &&
                default_off_result.lazyqp_affine_elimination_fallbacks == 0 &&
                default_off_result.lazyqp_full_kkt_dimension == 8,
            "default-off setting consumed affine elimination hints");
    require(max_difference(default_off_result.y, full_result.y) <= 3e-6,
            "default-off hint path changed the historical full solution");

    const Vec recovered_solution_lambda = reconstruct_full_equality_duals(
        full, reduction, expanded_solution, reduced_result.equality_duals,
        reduced_result.inequality_duals);
    const Vec recovered_solution_stationarity = stationarity(
        full, expanded_solution, recovered_solution_lambda,
        reduced_result.inequality_duals);
    require(inf_norm(recovered_solution_stationarity) <= 3e-6,
            "expanded reduced solution fails the full KKT stationarity test");
    for (std::size_t r = 0; r < full.equalities.rows.size(); ++r)
      require(std::abs(row_dot(full.equalities.rows[r], expanded_solution) -
                       full.equalities.rhs[r]) <= 3e-7,
              "expanded reduced solution fails a full equality");

    // Exercise the public opt-in wrapper as well as the transformation helper.
    // Seed it from the reference solution so both primal and full-dual warm
    // maps are accepted on the reduced problem and survive result expansion.
    ConvexSubproblem wrapped_problem = full;
    wrapped_problem.initial_primal = expanded_solution;
    wrapped_problem.initial_equality_duals = recovered_solution_lambda;
    wrapped_problem.initial_inequality_duals =
        reduced_result.inequality_duals;
    Settings wrapped_settings = settings;
    wrapped_settings.lazyqp_flow_lift_elimination = true;
    wrapped_settings.inner_dual_warm_start = true;
    wrapped_settings.inner_inequality_dual_warm_start = true;
    const InnerResult wrapped_result =
        ConvexQCQPLazyQPKernel(wrapped_settings).solve(wrapped_problem);
    require(wrapped_result.success,
            "opt-in affine-elimination solve failed: " +
                wrapped_result.message);
    require(wrapped_result.lazyqp_inexact_forcing_model_evaluations == 0,
            "disabled forcing evaluated the affine-reduced/full models");
    require(wrapped_result.lazyqp_affine_elimination_attempts == 1 &&
                wrapped_result.lazyqp_affine_elimination_acceptances == 1 &&
                wrapped_result.lazyqp_affine_elimination_fallbacks == 0,
            "opt-in wrapper did not accept exactly one affine reduction");
    require(wrapped_result.lazyqp_affine_eliminated_variables == 2 &&
                wrapped_result.lazyqp_affine_eliminated_equalities == 2 &&
                wrapped_result.lazyqp_affine_reduced_variables == 2 &&
                wrapped_result.lazyqp_affine_reduced_equalities == 1,
            "opt-in wrapper reported wrong reduction dimensions");
    require(wrapped_result.primal_warm_start_used &&
                wrapped_result.dual_warm_start_used &&
                wrapped_result.full_dual_warm_start_used &&
                !wrapped_result.dual_warm_start_rejected,
            "mapped primal/full-dual warm starts were not accepted");
    require(wrapped_result.y.size() == 4 &&
                wrapped_result.equality_duals.size() == 3 &&
                wrapped_result.inequality_duals.size() == 1 &&
                wrapped_result.inequality_slacks.size() == 1,
            "opt-in wrapper leaked reduced result dimensions");
    require(close(full.inequalities[0].value(wrapped_result.y) +
                      wrapped_result.inequality_slacks[0],
                  wrapped_result.primal_residual_inf, 3e-6) ||
                close(std::abs(full.inequalities[0].value(wrapped_result.y) +
                                   wrapped_result.inequality_slacks[0]),
                      wrapped_result.primal_residual_inf, 3e-6),
            "opt-in wrapper did not preserve/revalidate the inequality slack");
    require(close(wrapped_result.inequality_slacks[0] *
                      wrapped_result.inequality_duals[0],
                  wrapped_result.complementarity, 3e-6),
            "opt-in wrapper did not preserve/revalidate complementarity");
    require(max_difference(wrapped_result.y, full_result.y) <= 3e-6,
            "opt-in wrapper changed the full-space solution");
    require(inf_norm(stationarity(full, wrapped_result.y,
                                  wrapped_result.equality_duals,
                                  wrapped_result.inequality_duals)) <= 3e-6,
            "opt-in wrapper returned a non-KKT full-space result");

    // A forcing success in reduced coordinates must be independently
    // recertified after expansion, including against the full model at d=0;
    // the nonzero affine offset above makes a reduced-zero comparison invalid.
    ConvexSubproblem forcing_problem=full;
    forcing_problem.has_primal_warm_start=false;
    forcing_problem.has_dual_warm_start=false;
    forcing_problem.initial_primal.clear();
    forcing_problem.initial_equality_duals.clear();
    forcing_problem.initial_inequality_duals.clear();
    Settings forcing_settings=settings;
    forcing_settings.lazyqp_flow_lift_elimination=true;
    forcing_settings.lazyqp_inexact_forcing=true;
    forcing_settings.lazyqp_forcing_eta=0.9;
    forcing_settings.lazyqp_forcing_target_tolerance=10.0;
    forcing_settings.lazyqp_forcing_primal_tolerance=1e-4;
    forcing_settings.lazyqp_forcing_min_iterations=2;
    forcing_settings.lazyqp_forcing_fast_iterations=8;
    const InnerResult forcing_result=
        ConvexQCQPLazyQPKernel(forcing_settings).solve(forcing_problem);
    require(forcing_result.success &&
                forcing_result.lazyqp_inexact_forcing_attempted,
            "affine-reduced forcing path failed: "+forcing_result.message);
    require(forcing_result.lazyqp_inexact_forcing_accepted &&
                forcing_result.lazyqp_affine_elimination_acceptances==1 &&
                forcing_result.lazyqp_affine_elimination_fallbacks==0,
            "full-space forcing recertification rejected a valid exact map: "+
                forcing_result.message);

    // A stale/duplicated hint must attempt the guarded route once, reject it,
    // and solve the untouched full problem with no reduced dimensions exposed.
    ConvexSubproblem corrupt_wrapper = full_reference;
    corrupt_wrapper.exact_affine_eliminations = {{0, 2}, {1, 2}};
    const InnerResult fallback_result =
        ConvexQCQPLazyQPKernel(wrapped_settings).solve(corrupt_wrapper);
    require(fallback_result.success,
            "corrupt-hint full fallback failed: " + fallback_result.message);
    require(fallback_result.lazyqp_affine_elimination_attempts == 1 &&
                fallback_result.lazyqp_affine_elimination_acceptances == 0 &&
                fallback_result.lazyqp_affine_elimination_fallbacks == 1,
            "corrupt hint did not fail closed through the wrapper");
    require(fallback_result.y.size() == 4 &&
                fallback_result.equality_duals.size() == 3 &&
                fallback_result.lazyqp_full_kkt_dimension == 8,
            "corrupt hint did not retain the full problem dimensions");
    require(max_difference(fallback_result.y, full_result.y) <= 3e-6,
            "corrupt-hint fallback changed the historical full solution");

    // The same corrupt hint is not allowed to invoke that historical full
    // fallback from inside an active single-loop transaction.  Construction
    // fails before the reduced kernel runs, so this is a terminal fail-closed
    // result with zero KKT work and no full-dimensional fallback telemetry.
    Settings single_loop_affine_settings=wrapped_settings;
    single_loop_affine_settings.dca_single_loop_primal_dual=true;
    single_loop_affine_settings.dca_single_loop_inner_steps=1;
    single_loop_affine_settings.dca_single_loop_forcing_eta=0.9;
    single_loop_affine_settings.qp_mehrotra=true;

    // A rejected reduced-coordinate transaction must still return the full
    // origin snapshot needed by an outer proximal retry.  In particular, z/s
    // are not transformed by affine variable elimination and must not be
    // dropped while primal/equality-dual coordinates are expanded.
    Settings retry_snapshot_settings=single_loop_affine_settings;
    retry_snapshot_settings.dca_single_loop_forcing_eta=1e-15;
    ConvexSubproblem retry_snapshot_problem=full;
    retry_snapshot_problem.allow_dca_single_loop=true;
    const InnerResult retry_snapshot=
        ConvexQCQPLazyQPKernel(retry_snapshot_settings).solve(
            retry_snapshot_problem);
    require(!retry_snapshot.success &&
                retry_snapshot.dca_single_loop_retryable &&
                retry_snapshot.lazyqp_affine_elimination_attempts==1 &&
                retry_snapshot.lazyqp_affine_elimination_acceptances==0 &&
                retry_snapshot.lazyqp_affine_elimination_fallbacks==0,
            "affine-reduced guard rejection was not retryable: "+
                retry_snapshot.message);
    require(retry_snapshot.dca_single_loop_initial_primal.size()==4 &&
                retry_snapshot.dca_single_loop_initial_equality_duals.size()==3 &&
                retry_snapshot.dca_single_loop_initial_inequality_duals.size()==1 &&
                retry_snapshot.dca_single_loop_initial_inequality_slacks.size()==1 &&
                std::all_of(
                    retry_snapshot.dca_single_loop_initial_inequality_duals.begin(),
                    retry_snapshot.dca_single_loop_initial_inequality_duals.end(),
                    [](double value){ return std::isfinite(value)&&value>0.0; }) &&
                std::all_of(
                    retry_snapshot.dca_single_loop_initial_inequality_slacks.begin(),
                    retry_snapshot.dca_single_loop_initial_inequality_slacks.end(),
                    [](double value){ return std::isfinite(value)&&value>0.0; }),
            "affine-reduced retry lost its full initial primal-dual snapshot");
    require(inf_norm(retry_snapshot.dca_single_loop_initial_primal)>1e-3,
            "affine retry fixture no longer exercises a nonzero expanded origin");

    ConvexSubproblem corrupt_single_loop=corrupt_wrapper;
    corrupt_single_loop.allow_dca_single_loop=true;
    const InnerResult fail_closed_construction=
        ConvexQCQPLazyQPKernel(single_loop_affine_settings).solve(
            corrupt_single_loop);
    require(!fail_closed_construction.success &&
                !fail_closed_construction.dca_single_loop_accepted &&
                !fail_closed_construction.dca_single_loop_retryable &&
                !fail_closed_construction.dca_single_loop_strict_fallback &&
                fail_closed_construction.factorizations==0 &&
                fail_closed_construction.lazyqp_full_kkt_factorizations==0 &&
                fail_closed_construction.lazyqp_affine_elimination_attempts==1 &&
                fail_closed_construction.lazyqp_affine_elimination_acceptances==0 &&
                fail_closed_construction.lazyqp_affine_elimination_fallbacks==0 &&
                fail_closed_construction.message.find("failed closed")!=
                    std::string::npos,
            "single-loop affine construction failure escaped to a strict full solve");

    // A failure returned by the recursively reduced kernel is equally
    // fail-closed.  Disable Mehrotra only at the API level to make that
    // reduced solve reject its single-loop configuration deterministically;
    // the wrapper must not clear allow_dca_single_loop and solve(full).
    Settings invalid_reduced_settings=single_loop_affine_settings;
    invalid_reduced_settings.qp_mehrotra=false;
    ConvexSubproblem invalid_reduced_problem=full;
    invalid_reduced_problem.allow_dca_single_loop=true;
    const InnerResult fail_closed_reduced_solve=
        ConvexQCQPLazyQPKernel(invalid_reduced_settings).solve(
            invalid_reduced_problem);
    require(!fail_closed_reduced_solve.success &&
                !fail_closed_reduced_solve.dca_single_loop_accepted &&
                !fail_closed_reduced_solve.dca_single_loop_strict_fallback &&
                fail_closed_reduced_solve.factorizations==0 &&
                fail_closed_reduced_solve.lazyqp_full_kkt_factorizations==0 &&
                fail_closed_reduced_solve.lazyqp_affine_elimination_attempts==1 &&
                fail_closed_reduced_solve.lazyqp_affine_elimination_acceptances==0 &&
                fail_closed_reduced_solve.lazyqp_affine_elimination_fallbacks==0 &&
                fail_closed_reduced_solve.message.find("failed closed")!=
                    std::string::npos,
            "single-loop reduced-solve failure escaped to a strict full solve");

    ConvexSubproblem corrupt_hint = full;
    corrupt_hint.exact_affine_eliminations[1].variable = 2;
    expect_rejected(std::move(corrupt_hint), "duplicate-variable hint");

    ConvexSubproblem corrupt_signature = full;
    for (auto& entry : corrupt_signature.equalities.rows[0].entries)
      if (entry.first == 2) entry.second = 1.0 + 1e-12;
    expect_rejected(std::move(corrupt_signature), "nonexact pivot signature");

    ConvexSubproblem reused_pivot = full;
    reused_pivot.equalities.rows[2].entries.push_back({2, 1e-12});
    expect_rejected(std::move(reused_pivot), "pivot-column reuse");

    // Exercise a late rejection, after dimensions/maps have begun to be
    // assembled.  Fail-closed means this must clear those partial products as
    // rigorously as the early signature checks above.
    ConvexSubproblem eliminates_everything;
    eliminates_everything.nvar = 1;
    eliminates_everything.original_dimension = 1;
    eliminates_everything.prefer_feasible_centered_start = true;
    eliminates_everything.objective.linear = {0.0};
    eliminates_everything.objective.base_hessian.n = 1;
    eliminates_everything.equalities.nvar = 1;
    eliminates_everything.equalities.rows.push_back({{{0, 1.0}}});
    eliminates_everything.equalities.rhs.push_back(0.0);
    eliminates_everything.exact_affine_eliminations = {{0, 0}};
    expect_rejected(std::move(eliminates_everything),
                    "all-variable late rejection");

    std::cout << "[PASS] exact affine ConvexSubproblem reduction"
              << " n=4->2 eq=3->1 kkt=8->4"
              << " solution_delta="
              << max_difference(full_result.y, expanded_solution) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
