#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static ConvexQuadraticModel linear_row(double constant, double a) {
  ConvexQuadraticModel q;
  q.constant=constant;
  q.linear={a};
  q.base_hessian.n=1;
  return q;
}

static ConvexSubproblem first_problem() {
  ConvexSubproblem p;
  p.nvar=1;
  p.original_dimension=1;
  p.prefer_feasible_centered_start=true;
  // min 0.5 y^2 - 101 y, -1 <= y <= 1.  The upper-bound multiplier is 100.
  p.objective.linear={-101.0};
  p.objective.base_hessian.n=1;
  p.objective.base_hessian.entries={{0,0,1.0}};
  p.inequalities={linear_row(-1.0,+1.0),linear_row(-1.0,-1.0)};
  p.equalities.nvar=1;
  return p;
}

static ConvexSubproblem continuation_problem(const InnerResult* seed,
                                             bool deliberately_bad=false) {
  ConvexSubproblem p;
  p.nvar=1;
  p.original_dimension=1;
  p.prefer_feasible_centered_start=true;
  // Same absolute QP expressed in displacement coordinates at y_abs=1.
  // Its solution is d=0 and the preceding multiplier is already KKT-close.
  p.objective.linear={-100.0};
  p.objective.base_hessian.n=1;
  p.objective.base_hessian.entries={{0,0,1.0}};
  p.inequalities={linear_row(0.0,+1.0),linear_row(-2.0,-1.0)};
  p.equalities.nvar=1;
  if(seed) {
    p.has_dual_warm_start=true;
    p.initial_equality_duals=seed->equality_duals;
    p.initial_inequality_duals=seed->inequality_duals;
    if(deliberately_bad)
      p.initial_inequality_duals.assign(2,1.0e12);
  }
  return p;
}

int main() {
  try {
    Settings cold_settings;
    cold_settings.verbose=false;
    cold_settings.max_inner_iterations=200;
    cold_settings.inner_tolerance=1e-8;
    cold_settings.inner_iteration_limit_mu_fallback=1e-6;
    cold_settings.lazyqp_kkt_form="full";

    const InnerResult first=ConvexQCQPLazyQPKernel(cold_settings).solve(first_problem());
    if(!first.success || first.inequality_duals.size()!=2)
      throw std::runtime_error("failed to construct continuation multiplier state");
    if(first.inequality_duals[0]<50.0)
      throw std::runtime_error("first problem did not recover the expected active multiplier");

    const InnerResult cold=ConvexQCQPLazyQPKernel(cold_settings).solve(
        continuation_problem(nullptr));
    if(!cold.success)
      throw std::runtime_error("cold continuation reference failed");

    Settings warm_settings=cold_settings;
    warm_settings.inner_dual_warm_start=true;
    warm_settings.inner_inequality_dual_warm_start=true;
    warm_settings.lazyqp_warm_start_guard_factor=2.0;
    const InnerResult warm=ConvexQCQPLazyQPKernel(warm_settings).solve(
        continuation_problem(&first));
    if(!warm.success)
      throw std::runtime_error("guarded continuation failed: "+warm.message);
    if(!warm.dual_warm_start_used || !warm.full_dual_warm_start_used ||
       warm.dual_warm_start_rejected)
      throw std::runtime_error("valid full multiplier continuation was not accepted");
    if(warm.message.find("nonlinear-residual")==std::string::npos)
      throw std::runtime_error("full warm start escaped to the legacy IPM");
    if(std::abs(warm.y[0])>2e-6 || std::abs(cold.y[0])>2e-6)
      throw std::runtime_error("cold/warm continuation returned the wrong displacement");
    if(!(warm.iterations<cold.iterations))
      throw std::runtime_error("accepted continuation did not reduce Newton iterations");

    // Outer multiplier mapping represents inactive inequalities by exact
    // zeros.  r3p23 replaced those entries by the O(1) centered cold dual,
    // which needlessly restores a large barrier.  The new policy is opt-in,
    // strictly positive, componentwise complementarity-capped and guarded by
    // the same all-residual comparison as every full warm start.
    ConvexSubproblem mapped_zero_problem=continuation_problem(&first);
    mapped_zero_problem.dual_warm_start_is_mapped=true;
    mapped_zero_problem.initial_inequality_duals[1]=0.0;
    const InnerResult mapped_zero_r3p23=
        ConvexQCQPLazyQPKernel(warm_settings).solve(mapped_zero_problem);
    if(!mapped_zero_r3p23.success ||
       mapped_zero_r3p23.lazyqp_zero_dual_continuation_attempts!=0)
      throw std::runtime_error("disabled zero-dual continuation changed r3p23");

    Settings barrier_settings=warm_settings;
    barrier_settings.lazyqp_zero_dual_continuation=true;
    barrier_settings.lazyqp_zero_dual_mu_cap=1.0e-4;
    // ZERO_DUAL remains an independent optimization.  Poison the disabled
    // forcing contract so this test also proves that the successful repair
    // does not inspect or evaluate any forcing-only model quantity.
    barrier_settings.lazyqp_inexact_forcing=false;
    barrier_settings.lazyqp_forcing_eta=
        std::numeric_limits<double>::quiet_NaN();
    barrier_settings.lazyqp_forcing_target_tolerance=
        std::numeric_limits<double>::quiet_NaN();
    barrier_settings.lazyqp_forcing_primal_tolerance=
        std::numeric_limits<double>::quiet_NaN();
    barrier_settings.lazyqp_forcing_min_iterations=-17;
    barrier_settings.lazyqp_forcing_fast_iterations=-31;
    const InnerResult mapped_zero_barrier=
        ConvexQCQPLazyQPKernel(barrier_settings).solve(mapped_zero_problem);
    if(!mapped_zero_barrier.success ||
       !mapped_zero_barrier.dual_warm_start_used ||
       !mapped_zero_barrier.full_dual_warm_start_used ||
       mapped_zero_barrier.dual_warm_start_rejected ||
       mapped_zero_barrier.lazyqp_zero_dual_continuation_attempts!=1 ||
       mapped_zero_barrier.lazyqp_zero_dual_continuation_mapped_attempts!=1 ||
       mapped_zero_barrier.lazyqp_zero_dual_continuation_direct_attempts!=0 ||
       mapped_zero_barrier.lazyqp_zero_dual_continuation_acceptances!=1 ||
       mapped_zero_barrier.lazyqp_zero_dual_continuation_fallbacks!=0 ||
       mapped_zero_barrier.lazyqp_zero_dual_continuation_repairs!=1 ||
       mapped_zero_barrier.lazyqp_inexact_forcing_model_evaluations!=0)
      throw std::runtime_error("mapped zero barrier continuation was not accepted");
    if(!(mapped_zero_barrier.initial_complementarity<
         mapped_zero_r3p23.initial_complementarity) ||
       !(mapped_zero_barrier.initial_complementarity<=1.01e-4))
      throw std::runtime_error("mapped zero barrier did not bound initial mu");

    ConvexSubproblem direct_zero_problem=mapped_zero_problem;
    direct_zero_problem.dual_warm_start_is_mapped=false;
    direct_zero_problem.initial_inequality_duals[1]=
        barrier_settings.minimum_positive;
    const InnerResult direct_zero_barrier=
        ConvexQCQPLazyQPKernel(barrier_settings).solve(direct_zero_problem);
    if(!direct_zero_barrier.success ||
       direct_zero_barrier.lazyqp_zero_dual_continuation_attempts!=1 ||
       direct_zero_barrier.lazyqp_zero_dual_continuation_mapped_attempts!=0 ||
       direct_zero_barrier.lazyqp_zero_dual_continuation_direct_attempts!=1 ||
       direct_zero_barrier.lazyqp_zero_dual_continuation_acceptances!=1)
      throw std::runtime_error(
          "direct numerical-floor provenance was not recorded");
    if(!(direct_zero_barrier.initial_complementarity<=1.01e-4))
      throw std::runtime_error(
          "direct numerical-floor repair did not preserve the cap");

    // Dual infeasibility has a sign: a negative inequality multiplier is not
    // interpreted as a small nonnegative zero.  The complete warm state is
    // rejected and the untouched cold solve is recovered.
    ConvexSubproblem negative_dual_problem=mapped_zero_problem;
    negative_dual_problem.initial_inequality_duals[1]=-1.0e-9;
    const InnerResult negative_dual=
        ConvexQCQPLazyQPKernel(barrier_settings).solve(
            negative_dual_problem);
    if(!negative_dual.success ||
       !negative_dual.dual_warm_start_rejected ||
       negative_dual.dual_warm_start_used ||
       negative_dual.lazyqp_zero_dual_continuation_attempts!=0 ||
       negative_dual.lazyqp_zero_dual_continuation_acceptances!=0 ||
       negative_dual.lazyqp_zero_dual_continuation_fallbacks!=0 ||
       std::abs(negative_dual.y[0]-cold.y[0])>1e-10)
      throw std::runtime_error(
          "negative inequality multiplier was not rejected fail-closed");

    // A mixed stale/zero multiplier vector exercises the fail-closed path:
    // even a valid zero repair must not rescue a continuation whose complete
    // dual residual fails the guard.  The returned solution is the strict cold
    // reference, not a partially accepted hybrid state.
    ConvexSubproblem destructive_zero=continuation_problem(&first);
    destructive_zero.dual_warm_start_is_mapped=true;
    destructive_zero.initial_inequality_duals={1.0e12,0.0};
    const InnerResult barrier_fallback=
        ConvexQCQPLazyQPKernel(barrier_settings).solve(destructive_zero);
    if(!barrier_fallback.success ||
       !barrier_fallback.dual_warm_start_rejected ||
       barrier_fallback.dual_warm_start_used ||
       barrier_fallback.lazyqp_zero_dual_continuation_attempts!=1 ||
       barrier_fallback.lazyqp_zero_dual_continuation_mapped_attempts!=1 ||
       barrier_fallback.lazyqp_zero_dual_continuation_direct_attempts!=0 ||
       barrier_fallback.lazyqp_zero_dual_continuation_acceptances!=0 ||
       barrier_fallback.lazyqp_zero_dual_continuation_fallbacks!=1 ||
       barrier_fallback.lazyqp_zero_dual_continuation_repairs!=1)
      throw std::runtime_error(
          "mapped zero continuation did not fail closed: success="+
          std::to_string(barrier_fallback.success)+" rejected="+
          std::to_string(barrier_fallback.dual_warm_start_rejected)+" used="+
          std::to_string(barrier_fallback.dual_warm_start_used)+" try="+
          std::to_string(barrier_fallback.lazyqp_zero_dual_continuation_attempts)+
          " ok="+
          std::to_string(barrier_fallback.lazyqp_zero_dual_continuation_acceptances)+
          " fb="+
          std::to_string(barrier_fallback.lazyqp_zero_dual_continuation_fallbacks)+
          " repairs="+
          std::to_string(barrier_fallback.lazyqp_zero_dual_continuation_repairs));
    if(std::abs(barrier_fallback.y[0]-cold.y[0])>1e-10)
      throw std::runtime_error("zero-dual fallback changed the cold solution");

    const InnerResult rejected=ConvexQCQPLazyQPKernel(warm_settings).solve(
        continuation_problem(&first,true));
    if(!rejected.success || !rejected.dual_warm_start_rejected ||
       rejected.dual_warm_start_used)
      throw std::runtime_error("destructive warm state was not rejected safely");
    if(std::abs(rejected.y[0])>2e-6)
      throw std::runtime_error("guard fallback changed the cold solution");

    std::cout << "[PASS] guarded full-dual continuation"
              << " first_z="<<first.inequality_duals[0]
              << " cold_it="<<cold.iterations
              << " warm_it="<<warm.iterations
              << " r3p23_zero_mu="<<mapped_zero_r3p23.initial_complementarity
              << " barrier_zero_mu="<<mapped_zero_barrier.initial_complementarity
              << " rejected_it="<<rejected.iterations<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
