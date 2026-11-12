#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

ConvexQuadraticModel affine_row(double constant,double slope) {
  ConvexQuadraticModel row;
  row.constant=constant;
  row.linear={slope};
  row.base_hessian.n=1;
  return row;
}

ConvexSubproblem bounded_descent_problem() {
  ConvexSubproblem p;
  p.nvar=1;
  p.original_dimension=1;
  p.prefer_feasible_centered_start=true;
  p.objective.linear={-101.0};
  p.objective.base_hessian.n=1;
  p.objective.base_hessian.entries={{0,0,1.0}};
  p.inequalities={affine_row(-1.0,1.0),affine_row(-1.0,-1.0)};
  p.equalities.nvar=1;
  return p;
}

Settings strict_settings() {
  Settings settings;
  settings.verbose=false;
  settings.linear_backend="native";
  settings.lazyqp_kkt_form="full";
  settings.max_inner_iterations=100;
  settings.inner_tolerance=1e-9;
  settings.inner_iteration_limit_mu_fallback=1e-7;
  return settings;
}

QCQPProblem forcing_globalization_problem() {
  QCQPProblem problem;
  problem.name="forcing_transaction";
  problem.n=2;
  problem.x0={0.6,0.64};
  problem.lb={-2.0,-2.0};
  problem.ub={2.0,2.0};
  problem.objective.linear={1.0,0.0};
  problem.objective.hessian.n=2;
  problem.objective.hessian.entries={{0,0,-4.0},{1,1,-2.0}};
  QuadraticFunction equality;
  equality.name="quadratic_manifold";
  equality.constant=-1.0;
  equality.linear={0.0,1.0};
  equality.hessian.n=2;
  equality.hessian.entries={{0,0,2.0}}; // x0^2+x1-1=0
  problem.equalities={equality};
  problem.validate();
  return problem;
}

} // namespace

int main() {
  try {
    const ConvexSubproblem problem=bounded_descent_problem();
    const Settings strict=strict_settings();
    const InnerResult reference=ConvexQCQPLazyQPKernel(strict).solve(problem);
    if(!reference.success)
      throw std::runtime_error("strict LazyQP reference failed: "+reference.message);
    if(reference.lazyqp_inexact_forcing_model_evaluations!=0)
      throw std::runtime_error(
          "disabled forcing performed forcing-only model evaluations");

    // Poison every forcing parameter while leaving the feature disabled.  A
    // cleanly gated implementation must neither inspect/evaluate the forcing
    // contract nor perturb the historical numerical trajectory.
    Settings disabled_poisoned=strict;
    disabled_poisoned.lazyqp_inexact_forcing=false;
    disabled_poisoned.lazyqp_forcing_eta=
        std::numeric_limits<double>::quiet_NaN();
    disabled_poisoned.lazyqp_forcing_target_tolerance=
        std::numeric_limits<double>::quiet_NaN();
    disabled_poisoned.lazyqp_forcing_primal_tolerance=
        std::numeric_limits<double>::quiet_NaN();
    disabled_poisoned.lazyqp_forcing_min_iterations=-17;
    disabled_poisoned.lazyqp_forcing_fast_iterations=-31;
    const InnerResult disabled=
        ConvexQCQPLazyQPKernel(disabled_poisoned).solve(problem);
    const bool exact_parity=disabled.success==reference.success &&
        disabled.iterations==reference.iterations &&
        disabled.factorizations==reference.factorizations &&
        disabled.y==reference.y &&
        disabled.equality_duals==reference.equality_duals &&
        disabled.inequality_duals==reference.inequality_duals &&
        disabled.inequality_slacks==reference.inequality_slacks &&
        disabled.dual_inf==reference.dual_inf &&
        disabled.equality_inf==reference.equality_inf &&
        disabled.inequality_inf==reference.inequality_inf &&
        disabled.primal_residual_inf==reference.primal_residual_inf &&
        disabled.complementarity==reference.complementarity;
    if(disabled.lazyqp_inexact_forcing_model_evaluations!=0 || !exact_parity)
      throw std::runtime_error(
          "disabled forcing was not evaluation-free and numerically inert");

    Settings forced=strict;
    forced.lazyqp_inexact_forcing=true;
    forced.lazyqp_forcing_eta=0.5;
    forced.lazyqp_forcing_target_tolerance=2.0;
    forced.lazyqp_forcing_primal_tolerance=1e-6;
    forced.lazyqp_forcing_min_iterations=1;
    forced.lazyqp_forcing_fast_iterations=3;
    const InnerResult early=ConvexQCQPLazyQPKernel(forced).solve(problem);
    if(!early.success || !early.lazyqp_inexact_forcing_attempted ||
       !early.lazyqp_inexact_forcing_accepted)
      throw std::runtime_error("guarded forcing did not accept a useful early state: "+
                               early.message);
    if(early.lazyqp_inexact_forcing_model_evaluations<3)
      throw std::runtime_error(
          "enabled forcing did not account for its model evaluations");
    if(!(early.iterations<reference.iterations))
      throw std::runtime_error("forcing did not reduce LazyQP iterations");
    if(early.primal_residual_inf>forced.lazyqp_forcing_primal_tolerance ||
       early.inequality_inf>forced.lazyqp_forcing_primal_tolerance)
      throw std::runtime_error("forcing accepted a primal-inaccurate state");

    // An unattainable fast target must not turn into an inner failure or a
    // relaxed result: the same kernel continues to its historical strict KKT
    // contract and reproduces the reference minimizer.
    Settings fallback=forced;
    fallback.lazyqp_forcing_eta=1e-4;
    fallback.lazyqp_forcing_target_tolerance=1e6;
    fallback.lazyqp_forcing_primal_tolerance=1e6;
    fallback.lazyqp_forcing_fast_iterations=1;
    fallback.lazyqp_forcing_min_iterations=1;
    const InnerResult recovered=
        ConvexQCQPLazyQPKernel(fallback).solve(problem);
    if(!recovered.success || !recovered.lazyqp_inexact_forcing_attempted ||
       recovered.lazyqp_inexact_forcing_accepted ||
       !recovered.lazyqp_inexact_forcing_strict_fallback)
      throw std::runtime_error(
          "noncontracted model residual did not fall back strictly");
    if(!(recovered.lazyqp_inexact_forcing_target<=
         fallback.lazyqp_forcing_eta*
             recovered.lazyqp_inexact_forcing_initial_residual*
             (1.0+1e-12)) ||
       !(recovered.lazyqp_inexact_forcing_target<
         recovered.lazyqp_inexact_forcing_outer_target))
      throw std::runtime_error(
          "forcing target was not limited by relative KKT contraction");
    if(std::abs(recovered.y[0]-reference.y[0])>1e-8 ||
       recovered.dual_inf>100.0*strict.inner_tolerance ||
       recovered.primal_residual_inf>100.0*strict.inner_tolerance)
      throw std::runtime_error("strict fallback changed the certified solution");

    // A problem already satisfying the strict contract at initialization is
    // still classified as a strict fallback, so attempted = accepted +
    // strict_fallback remains true for every eligible subproblem.
    ConvexSubproblem stationary;
    stationary.nvar=1;
    stationary.original_dimension=1;
    stationary.prefer_feasible_centered_start=true;
    stationary.objective.linear={0.0};
    stationary.objective.base_hessian.n=1;
    stationary.objective.base_hessian.entries={{0,0,2.0}};
    stationary.equalities.nvar=1;
    const InnerResult immediate=
        ConvexQCQPLazyQPKernel(forced).solve(stationary);
    if(!immediate.success || !immediate.lazyqp_inexact_forcing_attempted ||
       immediate.lazyqp_inexact_forcing_accepted ||
       !immediate.lazyqp_inexact_forcing_strict_fallback)
      throw std::runtime_error(
          "forcing counters did not classify immediate strict convergence: " +
          std::to_string(immediate.lazyqp_inexact_forcing_attempted) + "/" +
          std::to_string(immediate.lazyqp_inexact_forcing_accepted) + "/" +
          std::to_string(immediate.lazyqp_inexact_forcing_strict_fallback) +
          " " + immediate.message);

    // Force an alpha=1-only globalization on a curved equality manifold.  A
    // useful inexact direction is rejected once; Solver must transactionally
    // re-solve that identical majorant under the strict contract instead of
    // exposing LineSearchFailure.  This deterministic problem exercises the
    // outer fallback rather than only the kernel-local fast-window fallback.
    Settings outer=strict;
    outer.outer_algorithm="lazy-dca";
    outer.dca_equality_model="condensed-retraction";
    outer.max_outer_iterations=40;
    outer.max_inner_iterations=100;
    outer.feasibility_tolerance=1e-6;
    outer.stationarity_tolerance=1e-5;
    outer.dual_infeasibility_tolerance=1e-5;
    outer.complementarity_tolerance=1e-5;
    outer.lazyqp_inexact_forcing=true;
    outer.lazyqp_forcing_eta=0.9;
    outer.lazyqp_forcing_max_tolerance=10.0;
    outer.lazyqp_forcing_min_iterations=1;
    outer.lazyqp_forcing_fast_iterations=3;
    outer.line_search_armijo=0.8;
    outer.line_search_max_steps=1;
    const Result transactional=Solver(outer).solve(
        forcing_globalization_problem());
    if(transactional.status!=Status::Solved ||
       transactional.lazyqp_inexact_forcing_acceptances<1 ||
       transactional.lazyqp_inexact_forcing_outer_retries!=1 ||
       transactional.lazyqp_inexact_forcing_model_evaluations<1)
      throw std::runtime_error(
          "outer forcing rejection did not aggregate its public telemetry: "+
          transactional.message);
    if(transactional.feasibility.total_inf>outer.feasibility_tolerance ||
       transactional.dual_infeasibility_inf>
           outer.dual_infeasibility_tolerance ||
       transactional.complementarity_inf>
           outer.complementarity_tolerance)
      throw std::runtime_error(
          "transactional forcing fallback bypassed final KKT certification");

    std::cout << "[PASS] guarded LazyQP forcing early="<<early.iterations
              << " strict="<<reference.iterations
              << " fallback="<<recovered.iterations
              << " target="<<early.lazyqp_inexact_forcing_target
              << " outer_retry="
              << transactional.lazyqp_inexact_forcing_outer_retries<<"\n";
    return 0;
  } catch(const std::exception& exception) {
    std::cerr << "[FAIL] "<<exception.what()<<"\n";
    return 1;
  }
}
