#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

ConvexQuadraticModel objective_model() {
  ConvexQuadraticModel model;
  model.linear={-1.0};
  model.base_hessian.n=1;
  model.base_hessian.entries.push_back({0,0,1.0});
  return model;
}

ConvexQuadraticModel upper_bound(double upper) {
  ConvexQuadraticModel model;
  model.constant=-upper;
  model.linear={1.0};
  model.base_hessian.n=1;
  return model;
}

ConvexSubproblem one_dimensional_majorant() {
  ConvexSubproblem p;
  p.nvar=1;
  p.original_dimension=1;
  p.objective=objective_model();
  p.equalities.nvar=1;
  p.inequalities.push_back(upper_bound(2.0));
  p.prefer_feasible_centered_start=true;
  p.allow_dca_single_loop=true;
  p.has_primal_warm_start=true;
  p.initial_primal={0.0};
  p.has_dual_warm_start=true;
  p.initial_equality_duals={};
  p.initial_inequality_duals={0.1};
  p.has_inequality_slack_warm_start=true;
  p.initial_inequality_slacks={2.0};
  return p;
}

Settings settings(double eta) {
  Settings st;
  st.linear_backend="native";
  st.max_inner_iterations=100;
  st.inner_tolerance=1e-9;
  st.qp_model_solved_tol=1e-11;
  st.inner_iteration_limit_mu_fallback=1e-8;
  st.inner_dual_warm_start=true;
  st.inner_inequality_dual_warm_start=true;
  st.lazyqp_warm_start_guard_factor=100.0;
  st.dca_single_loop_primal_dual=true;
  st.dca_single_loop_inner_steps=1;
  st.dca_single_loop_forcing_eta=eta;
  st.qp_mehrotra=true;
  st.verbose=false;
  return st;
}

bool same_vector(const Vec& lhs,const Vec& rhs,double tolerance=1e-10) {
  if(lhs.size()!=rhs.size()) return false;
  for(std::size_t i=0;i<lhs.size();++i)
    if(std::abs(lhs[i]-rhs[i])>tolerance) return false;
  return true;
}

bool same_strict_state(const InnerResult& lhs,const InnerResult& rhs) {
  return same_vector(lhs.y,rhs.y) &&
      same_vector(lhs.equality_duals,rhs.equality_duals) &&
      same_vector(lhs.inequality_duals,rhs.inequality_duals) &&
      same_vector(lhs.inequality_slacks,rhs.inequality_slacks) &&
      std::abs(lhs.dual_inf-rhs.dual_inf)<=1e-10 &&
      std::abs(lhs.equality_inf-rhs.equality_inf)<=1e-10 &&
      std::abs(lhs.inequality_inf-rhs.inequality_inf)<=1e-10 &&
      std::abs(lhs.primal_residual_inf-rhs.primal_residual_inf)<=1e-10 &&
      std::abs(lhs.complementarity-rhs.complementarity)<=1e-10;
}

} // namespace

int main() {
  try {
    const ConvexSubproblem p=one_dimensional_majorant();

    const Settings provisional_settings=settings(0.999999);
    const InnerResult provisional=
        ConvexQCQPLazyQPKernel(provisional_settings).solve(p);
    if(provisional.success || !provisional.dca_single_loop_attempted ||
       !provisional.dca_single_loop_accepted ||
       provisional.dca_single_loop_strict_fallback ||
       !provisional.dca_single_loop_state_transport_used ||
       provisional.dca_single_loop_steps!=1 ||
       provisional.dca_single_loop_factorizations!=1)
      throw std::runtime_error(
          "one-step continuation did not return a distinct provisional state");
    if(provisional.y.size()!=1 ||
       provisional.equality_duals.size()!=0 ||
       provisional.inequality_duals.size()!=1 ||
       provisional.inequality_slacks.size()!=1 ||
       !(provisional.inequality_duals[0]>0.0) ||
       !(provisional.inequality_slacks[0]>0.0) ||
       !(provisional.dca_single_loop_final_residual<
         provisional.dca_single_loop_initial_residual) ||
       !(p.objective.value(provisional.y)<
         p.objective.value(p.initial_primal)))
      throw std::runtime_error(
          "provisional state failed its finite/interior/contraction/decrease contract");

    // The outer filter is the final authority for the original QCQP, but it may
    // receive only a direction which passed the local one-step transaction.
    // An impossible contraction target must therefore roll back to the exact
    // initial state and request a different proximal majorant; finite/interior
    // values alone are not sufficient.
    const Settings delegated_settings=settings(1e-15);
    const InnerResult delegated=
        ConvexQCQPLazyQPKernel(delegated_settings).solve(p);
    if(delegated.success || delegated.dca_single_loop_accepted ||
       !delegated.dca_single_loop_retryable ||
       delegated.factorizations!=1 || delegated.iterations!=1 ||
       !same_vector(delegated.y,p.initial_primal) ||
       !same_vector(delegated.inequality_duals,
                    p.initial_inequality_duals) ||
       !same_vector(delegated.inequality_slacks,
                    p.initial_inequality_slacks))
      throw std::runtime_error(
          "failed single-loop contraction did not roll back transactionally");

    Settings strict_settings=delegated_settings;
    strict_settings.dca_single_loop_primal_dual=false;
    ConvexSubproblem strict_problem=p;
    strict_problem.allow_dca_single_loop=false;
    const InnerResult strict=
        ConvexQCQPLazyQPKernel(strict_settings).solve(strict_problem);

    // "One step" is an upper bound for a nonterminal majorant, not an order
    // to perturb a state which already meets the strict inner contract.  A
    // fully transported strict solution is allowed to return with zero KKT
    // work, and that is the only zero-step exception.
    ConvexSubproblem already_solved=p;
    already_solved.initial_primal=strict.y;
    already_solved.initial_equality_duals=strict.equality_duals;
    already_solved.initial_inequality_duals=strict.inequality_duals;
    already_solved.initial_inequality_slacks=strict.inequality_slacks;
    const InnerResult zero_step=
        ConvexQCQPLazyQPKernel(provisional_settings).solve(already_solved);
    if(!zero_step.success || zero_step.dca_single_loop_attempted ||
       zero_step.dca_single_loop_accepted ||
       zero_step.dca_single_loop_steps!=0 || zero_step.factorizations!=0)
      throw std::runtime_error(
          "already strict transported state did not take the zero-step exit");

    // Merely setting the primary-only hint cannot change historical behavior
    // while the master option is off.
    ConvexSubproblem strict_with_hint=p;
    const InnerResult strict_hint=
        ConvexQCQPLazyQPKernel(strict_settings).solve(strict_with_hint);
    if(!strict_hint.success || !same_strict_state(strict_hint,strict) ||
       strict_hint.factorizations!=strict.factorizations ||
       strict_hint.iterations!=strict.iterations ||
       strict_hint.dca_single_loop_attempted ||
       strict_hint.dca_single_loop_accepted)
      throw std::runtime_error(
          "default-off single-loop hint changed the historical solve");

    // Kernel API users do not pass through CLI validation.  Invalid eta must
    // fail closed before a factorization instead of weakening the guard.
    const InnerResult invalid=
        ConvexQCQPLazyQPKernel(settings(1.0)).solve(p);
    if(invalid.success || invalid.dca_single_loop_attempted ||
       invalid.dca_single_loop_accepted || invalid.factorizations!=0 ||
       invalid.message!="invalid single-loop primal-dual configuration")
      throw std::runtime_error(
          "invalid API contraction setting did not fail closed");

    // The public integer is retained for CLI/API compatibility, but active
    // single-loop execution is a one-step protocol rather than a tunable
    // truncated inner solve.  API callers requesting two steps must fail
    // before either a predictor or a factorization is attempted.  A zero-step
    // return is reserved solely for an already strictly solved input state.
    Settings invalid_step_settings=settings(0.9);
    invalid_step_settings.dca_single_loop_inner_steps=2;
    const InnerResult invalid_steps=
        ConvexQCQPLazyQPKernel(invalid_step_settings).solve(p);
    if(invalid_steps.success || invalid_steps.dca_single_loop_attempted ||
       invalid_steps.dca_single_loop_accepted ||
       invalid_steps.dca_single_loop_steps!=0 ||
       invalid_steps.factorizations!=0 ||
       invalid_steps.message!="invalid single-loop primal-dual configuration")
      throw std::runtime_error(
          "multi-step API request did not fail closed before factorization");

    Settings invalid_reuse_settings=settings(0.9);
    invalid_reuse_settings.lazyqp_numeric_kkt_reuse=true;
    const InnerResult invalid_reuse=
        ConvexQCQPLazyQPKernel(invalid_reuse_settings).solve(p);
    if(invalid_reuse.success || invalid_reuse.dca_single_loop_attempted ||
       invalid_reuse.dca_single_loop_accepted ||
       invalid_reuse.dca_single_loop_steps!=0 ||
       invalid_reuse.dca_single_loop_factorizations!=0 ||
       invalid_reuse.factorizations!=0 ||
       invalid_reuse.lazyqp_numeric_kkt_reuse_attempts!=0 ||
       invalid_reuse.lazyqp_numeric_kkt_reuse_acceptances!=0 ||
       invalid_reuse.lazyqp_numeric_kkt_reuse_fallbacks!=0 ||
       invalid_reuse.lazyqp_numeric_kkt_reuse_iterations!=0 ||
       invalid_reuse.message!="invalid single-loop primal-dual configuration")
      throw std::runtime_error(
          "numerical KKT reuse API request did not fail before factorization");

    ConvexSubproblem empty_barrier=p;
    empty_barrier.inequalities.clear();
    empty_barrier.initial_inequality_duals.clear();
    empty_barrier.initial_inequality_slacks.clear();
    const InnerResult rejected_empty_barrier=
        ConvexQCQPLazyQPKernel(provisional_settings).solve(empty_barrier);
    if(rejected_empty_barrier.success ||
       rejected_empty_barrier.dca_single_loop_attempted ||
       rejected_empty_barrier.dca_single_loop_accepted ||
       rejected_empty_barrier.factorizations!=0 ||
       rejected_empty_barrier.message!=
           "single-loop primal-dual majorant requires at least one inequality row")
      throw std::runtime_error(
          "empty single-loop barrier escaped to an equality-QP solve");

    // A malformed slack vector invalidates the complete transported state;
    // lambda/z must not be silently combined with reconstructed cold slacks.
    ConvexSubproblem malformed=p;
    malformed.initial_inequality_slacks={-1.0};
    const InnerResult rejected_transport=
        ConvexQCQPLazyQPKernel(provisional_settings).solve(malformed);
    if(rejected_transport.success ||
       !rejected_transport.dca_single_loop_attempted ||
       !rejected_transport.dca_single_loop_accepted ||
       !rejected_transport.dca_single_loop_state_transport_rejected ||
       rejected_transport.dca_single_loop_state_transport_used)
      throw std::runtime_error(
          "invalid complete transport did not use a centered provisional state");

    // A cold first majorant receives one centered predictor-corrector step;
    // strict bootstrap solves are forbidden in single-loop mode.
    ConvexSubproblem bootstrap=p;
    bootstrap.has_dual_warm_start=false;
    bootstrap.initial_inequality_duals.clear();
    bootstrap.has_inequality_slack_warm_start=false;
    bootstrap.initial_inequality_slacks.clear();
    const InnerResult first=
        ConvexQCQPLazyQPKernel(provisional_settings).solve(bootstrap);
    if(first.success || !first.dca_single_loop_attempted ||
       !first.dca_single_loop_accepted ||
       first.factorizations!=first.dca_single_loop_factorizations ||
       first.factorizations!=1)
      throw std::runtime_error(
          "cold first majorant did not use exactly one provisional step");

    std::cout << "[PASS] single-loop provisional/fallback/transport/bootstrap"
              << "/api/default-off"
              << " residual="<<provisional.dca_single_loop_initial_residual
              << "->"<<provisional.dca_single_loop_final_residual
              << " delegated_fact="<<delegated.factorizations<<"\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] "<<error.what()<<"\n";
    return 1;
  }
}
