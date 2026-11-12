#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_problem() {
  QCQPProblem p;
  p.name="single_loop_working_set_transport";
  p.n=1;
  p.x0={0.0};
  p.lb={-10.0};
  p.ub={10.0};

  // min 0.5*x^2+2*x, whose solution x=-2 moves away from the
  // initially-near-active (and ultimately irrelevant) side constraint.
  p.objective.name="objective";
  p.objective.linear={2.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};

  QuadraticFunction side;
  side.name="transient_side";
  side.constant=-5.0e-5;
  side.linear={1.0};
  side.hessian.n=1;
  p.inequalities.push_back(std::move(side));
  p.validate();
  return p;
}

QCQPProblem make_addition_problem() {
  QCQPProblem p;
  p.name="single_loop_working_set_addition";
  p.n=1;
  p.x0={-2.0};
  p.lb={-10.0};
  p.ub={10.0};

  // With the extra proximal term, the first exact majorant endpoint is -0.5.
  // The side row is omitted at x=-2 and remains feasible there, but enters the
  // activation band at the next accepted origin.  This exercises a genuinely
  // new row in direct continuation rather than same-outer violation expansion.
  p.objective.name="objective";
  p.objective.linear={-1.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};

  QuadraticFunction side;
  side.name="late_side";
  side.constant=0.4995; // x <= -0.4995
  side.linear={1.0};
  side.hessian.n=1;
  p.inequalities.push_back(std::move(side));
  p.validate();
  return p;
}

QCQPProblem make_same_outer_expansion_problem() {
  QCQPProblem p;
  p.name="single_loop_same_outer_working_set_expansion";
  p.n=1;
  p.x0={0.0};
  p.lb={-10.0};
  p.ub={10.0};

  // The row x<=0.01 is safely outside the initial 1e-3 activation band, but
  // the first bounds-only one-PC direction moves toward the unconstrained
  // minimizer x=2 and violates it.  The row must be added and initialized by
  // semantic key at the same outer origin, never by a nonzero primal seed
  // that disables the single-loop route.
  p.objective.name="objective";
  p.objective.linear={-2.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};

  QuadraticFunction side;
  side.name="same_outer_side";
  side.constant=-1.0e-2;
  side.linear={1.0};
  side.hessian.n=1;
  p.inequalities.push_back(std::move(side));
  p.validate();
  return p;
}

Settings settings() {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.inner_backend="lazyqp";
  st.linear_backend="native";
  st.verbose=false;
  st.restoration_enabled=false;
  st.dca_local_corrector=false;
  st.second_order_correction=false;
  st.terminal_kkt_enabled=false;
  st.max_outer_iterations=100;
  st.max_inner_iterations=200;
  st.feasibility_tolerance=1e-8;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-9;
  st.dca_extra_proximal_regularization=1.0;
  st.working_set_activation=1e-4;
  st.working_set_drop_slack=2e-4;
  st.working_set_drop_multiplier=1e-6;
  st.working_set_drop_patience=1;
  st.inner_dual_warm_start=true;
  st.inner_inequality_dual_warm_start=true;
  st.lazyqp_warm_start_guard_factor=100.0;
  st.dca_single_loop_primal_dual=true;
  st.dca_single_loop_inner_steps=1;
  st.dca_single_loop_forcing_eta=0.999999;
  st.dca_inner_retry_initial_prox=1.0;
  st.dca_inner_retry_max=6;
  return st;
}

} // namespace

int main() {
  try {
    const Settings st=settings();
    const Result result=Solver(st).solve(make_problem());
    if(result.status!=Status::Solved)
      throw std::runtime_error("solve failed: "+result.message);
    if(result.working_set_drops<1)
      throw std::runtime_error("test did not drop the transient row");
    if(result.dca_single_loop_keyed_transport_churn_acceptances<1 ||
       result.dca_single_loop_keyed_transport_dropped_rows<1)
      throw std::runtime_error(
          "row-keyed continuation did not survive the working-set drop");
    if(result.dca_single_loop_keyed_transport_matched_rows<2)
      throw std::runtime_error(
          "stable lower/upper-bound rows were not matched by identity");
    if(result.dca_single_loop_keyed_transport_rejections!=0)
      throw std::runtime_error(
          "valid keyed continuation was rejected before the warm guard");
    if(result.dca_single_loop_attempts<1 ||
       result.dca_single_loop_state_transport_acceptances<1)
      throw std::runtime_error(
          "keyed proposal did not reach the guarded single-loop kernel");
    if(result.multiplier_refit_attempts!=0 ||
       result.dca_single_loop_full_refit_calls!=1 ||
       result.dca_single_loop_strict_fallbacks!=0 ||
       result.dca_single_loop_retraction_fallbacks!=0 ||
       result.lazy_dca_equality_retraction_attempts!=0 ||
       result.lazy_dca_condensed_retraction_attempts!=0 ||
       result.kkt_factorizations!=result.dca_single_loop_factorizations)
      throw std::runtime_error(
          "single-loop route performed a path refit or repeated its final fit");
    if(std::abs(result.x[0]+2.0)>2e-4 ||
       result.feasibility.total_inf>st.feasibility_tolerance)
      throw std::runtime_error("wrong certified endpoint");

    Settings add_settings=st;
    add_settings.working_set_activation=1e-3;
    add_settings.working_set_dropping=false;
    const Result added=Solver(add_settings).solve(make_addition_problem());
    if(added.status!=Status::Solved)
      throw std::runtime_error("addition solve failed: "+added.message);
    if(added.working_set_additions<1 ||
       added.dca_single_loop_keyed_transport_churn_acceptances<1 ||
       added.dca_single_loop_keyed_transport_new_rows<1)
      throw std::runtime_error(
          "row-keyed continuation did not initialize the added row");
    if(added.dca_single_loop_keyed_transport_rejections!=0 ||
       added.dca_single_loop_state_transport_acceptances<1)
      throw std::runtime_error(
          "added-row continuation failed its transactional warm guard");
    if(added.multiplier_refit_attempts!=0 ||
       added.dca_single_loop_full_refit_calls!=1 ||
       added.dca_single_loop_strict_fallbacks!=0 ||
       added.dca_single_loop_retraction_fallbacks!=0 ||
       added.lazy_dca_equality_retraction_attempts!=0 ||
       added.lazy_dca_condensed_retraction_attempts!=0 ||
       added.kkt_factorizations!=added.dca_single_loop_factorizations)
      throw std::runtime_error(
          "added-row route violated the terminal-only full-refit contract");
    if(std::abs(added.x[0]+0.4995)>2e-4 ||
       added.feasibility.total_inf>add_settings.feasibility_tolerance)
      throw std::runtime_error("addition problem returned wrong endpoint");

    const Result same_outer=Solver(add_settings).solve(
        make_same_outer_expansion_problem());
    if(same_outer.status!=Status::Solved)
      throw std::runtime_error(
          "same-outer expansion solve failed: "+same_outer.message);
    if(same_outer.working_set_additions<1 ||
       same_outer.dca_single_loop_keyed_transport_new_rows<1 ||
       same_outer.dca_single_loop_keyed_transport_churn_acceptances<1)
      throw std::runtime_error(
          "same-outer expansion did not initialize its new semantic row");
    if(same_outer.dca_single_loop_strict_fallbacks!=0 ||
       same_outer.dca_single_loop_retraction_fallbacks!=0 ||
       same_outer.lazy_dca_equality_retraction_attempts!=0 ||
       same_outer.lazy_dca_condensed_retraction_attempts!=0 ||
       same_outer.kkt_factorizations!=
           same_outer.dca_single_loop_factorizations)
      throw std::runtime_error(
          "same-outer working-set expansion escaped into a strict solve: strict="+
          std::to_string(same_outer.dca_single_loop_strict_fallbacks)+
          " kkt/single="+std::to_string(same_outer.kkt_factorizations)+"/"+
          std::to_string(same_outer.dca_single_loop_factorizations));
    if(same_outer.multiplier_refit_attempts!=0 ||
       same_outer.dca_single_loop_full_refit_calls!=1 ||
       same_outer.feasibility.total_inf>add_settings.feasibility_tolerance)
      throw std::runtime_error(
          "same-outer expansion violated the terminal-only fit contract");

    Settings packaging_settings=add_settings;
    packaging_settings.max_outer_iterations=0;
    const Result far_packaging=Solver(packaging_settings).solve(
        make_same_outer_expansion_problem());
    if(far_packaging.status!=Status::MaxOuterIterations ||
       far_packaging.dca_single_loop_full_refit_calls!=0 ||
       far_packaging.multiplier_refit_attempts!=0 ||
       far_packaging.termination_multiplier_fit_converged)
      throw std::runtime_error(
          "final packaging fitted multipliers away from termination");

    std::cout << "[PASS] keyed single-loop working-set transport"
              << " drops="<<result.working_set_drops
              << " churn="
              << result.dca_single_loop_keyed_transport_churn_acceptances
              << " matched="
              << result.dca_single_loop_keyed_transport_matched_rows
              << " attempts="<<result.dca_single_loop_attempts
              << " added="
              << added.dca_single_loop_keyed_transport_new_rows
              << " same-outer-fact="
              << same_outer.dca_single_loop_factorizations
              << " x="<<result.x[0]<<"\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] "<<error.what()<<"\n";
    return 1;
  }
}
