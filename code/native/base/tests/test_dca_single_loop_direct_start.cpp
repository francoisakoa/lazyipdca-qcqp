#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/single_loop_direct_start.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

QuadraticFunction affine_row(const char* name, double constant,
                             double linear) {
  QuadraticFunction row;
  row.name=name;
  row.constant=constant;
  row.linear={linear};
  row.hessian.n=1;
  return row;
}

QCQPProblem side_start(double scale=1.0) {
  QCQPProblem p;
  p.name="single_loop_direct_side_start";
  p.n=1;
  p.x0={0.0};
  p.lb={-2.0};
  p.ub={2.0};
  p.objective.name="objective";
  // The side-recovery direction x<0 must also be a descent direction for the
  // one-model predictor-corrector acceptance contract.
  p.objective.linear={1.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};
  p.inequalities.push_back(affine_row("side_1",4.0e-3*scale,1.0));
  p.inequalities.push_back(affine_row("side_2",5.0e-3*scale,2.0));
  p.validate();
  return p;
}

QCQPProblem bound_start() {
  QCQPProblem p;
  p.name="single_loop_direct_bound_reject";
  p.n=1;
  p.x0={1.005};
  p.lb={-2.0};
  p.ub={1.0};
  p.objective.name="objective";
  p.objective.linear={0.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};
  p.equalities.push_back(affine_row("at_upper_bound",-1.0,1.0));
  p.validate();
  return p;
}

QCQPProblem equality_start() {
  QCQPProblem p;
  p.name="single_loop_direct_equality_reject";
  p.n=1;
  p.x0={0.0};
  p.lb={-2.0};
  p.ub={2.0};
  p.objective.name="objective";
  p.objective.linear={0.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};
  p.equalities.push_back(affine_row("small_equality_violation",5.0e-3,1.0));
  p.validate();
  return p;
}

QCQPProblem equality_only_unbounded() {
  QCQPProblem p;
  p.name="single_loop_equality_only_reject";
  p.n=1;
  p.x0={0.0};
  p.lb={-std::numeric_limits<double>::infinity()};
  p.ub={std::numeric_limits<double>::infinity()};
  p.objective.name="objective";
  p.objective.linear={0.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};
  p.equalities.push_back(affine_row("fixed_origin",0.0,1.0));
  p.validate();
  return p;
}

QCQPProblem fixed_bound_only() {
  QCQPProblem p;
  p.name="single_loop_fixed_bound_only_reject";
  p.n=1;
  p.x0={0.0};
  p.lb={0.0};
  p.ub={0.0};
  p.objective.name="objective";
  p.objective.linear={1.0};
  p.objective.hessian.n=1;
  p.validate();
  return p;
}

QCQPProblem nonfinite_start() {
  QCQPProblem p=side_start();
  p.name="single_loop_direct_nonfinite_reject";
  p.x0={std::numeric_limits<double>::infinity()};
  p.validate();
  return p;
}

Settings settings() {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.inner_backend="lazyqp";
  st.linear_backend="native";
  st.verbose=false;
  st.restoration_enabled=true;
  st.restoration_max_iterations=8;
  st.restoration_max_restarts=0;
  st.restoration_minimax_enabled=false;
  st.max_outer_iterations=0;
  st.max_inner_iterations=20;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-6;
  st.inner_tolerance=1e-9;
  st.terminal_kkt_enabled=false;
  st.dca_local_corrector=false;
  st.second_order_correction=false;
  st.working_set_dropping=false;
  st.dca_full_constraints_for_affine_objective=false;
  st.dca_single_loop_primal_dual=true;
  st.dca_single_loop_forcing_eta=0.999999;
  st.dca_single_loop_direct_start=true;
  st.dca_single_loop_direct_start_theta_cap=1e-2;
  st.dca_single_loop_direct_start_gamma_h=5e-2;
  return st;
}

} // namespace

int main() {
  try {
    const Settings st=settings();
    const auto require_api_rejection=
        [](const Settings& invalid,const QCQPProblem& problem,
           const char* expected) {
      try {
        (void)Solver(invalid).solve(problem);
      } catch(const std::runtime_error& error) {
        if(std::string(error.what()).find(expected)!=std::string::npos)
          return;
        throw;
      }
      throw std::runtime_error(
          std::string("single-loop API configuration bypass: ")+expected);
    };

    Settings invalid=st;
    invalid.outer_algorithm="auto";
    require_api_rejection(invalid,side_start(),"outer_algorithm=lazy-dca");
    invalid=st;
    invalid.inner_backend="filter-sqp";
    require_api_rejection(invalid,side_start(),"inner_backend=lazyqp");
    invalid=st;
    invalid.qp_mehrotra=false;
    require_api_rejection(invalid,side_start(),"requires Mehrotra");
    invalid=st;
    invalid.dca_single_loop_inner_steps=2;
    require_api_rejection(invalid,side_start(),"exactly one inner step");
    invalid=st;
    invalid.dca_single_loop_retraction_fallback_only=true;
    require_api_rejection(
        invalid,side_start(),"forbids auxiliary retraction fallback");
    invalid=st;
    invalid.lazyqp_numeric_kkt_reuse=true;
    require_api_rejection(
        invalid,side_start(),"forbids numerical KKT reuse");
    require_api_rejection(
        st,equality_only_unbounded(),
        "requires an inequality or nonfixed finite bound");
    require_api_rejection(
        st,fixed_bound_only(),
        "requires an inequality or nonfixed finite bound");

    Feasibility current;
    current.inequality_inf=8e-3;
    current.total_inf=8e-3;
    Feasibility sufficient;
    sufficient.inequality_inf=7.5e-3;
    sufficient.total_inf=7.5e-3;
    const auto sufficient_check=check_direct_start_recovery_step(
        current,sufficient,1.0,1e-6,1e-2,5e-2,true,true);
    if(!sufficient_check.accepted())
      throw std::runtime_error("valid h-step contraction was rejected");

    Feasibility micro=sufficient;
    micro.inequality_inf=7.7e-3;
    micro.total_inf=7.7e-3;
    const auto micro_check=check_direct_start_recovery_step(
        current,micro,1.0,1e-6,1e-2,5e-2,true,true);
    if(micro_check.accepted() || micro_check.contracts_theta)
      throw std::runtime_error(
          "sub-gamma h-step reduction was incorrectly accepted");

    // Regression: the old `target + roundoff` predicate accepted an unchanged
    // trial after enough backtracking because the requested decrease vanished
    // beneath its absolute allowance.
    const auto unresolved_micro_check=check_direct_start_recovery_step(
        current,current,1e-12,1e-6,1e-2,5e-2,true,true);
    if(unresolved_micro_check.accepted() ||
       unresolved_micro_check.contracts_theta)
      throw std::runtime_error(
          "unresolved direct-start h-step decrease was accepted");

    const double evaluation_roundoff=
        8192.0*std::numeric_limits<double>::epsilon();
    const double small_alpha=std::ldexp(1.0,-24);
    const double old_target=(1.0-5e-2*small_alpha)*current.total_inf;
    Feasibility old_tolerance_trial=current;
    old_tolerance_trial.inequality_inf=
        old_target+0.5*evaluation_roundoff;
    old_tolerance_trial.total_inf=old_tolerance_trial.inequality_inf;
    const auto old_tolerance_check=check_direct_start_recovery_step(
        current,old_tolerance_trial,small_alpha,1e-6,1e-2,5e-2,true,true);
    if(old_tolerance_check.accepted() ||
       old_tolerance_check.contracts_theta)
      throw std::runtime_error(
          "additive-roundoff direct-start microstep regression");

    Feasibility tube_entry=current;
    tube_entry.inequality_inf=1e-6;
    tube_entry.total_inf=1e-6;
    const auto tube_entry_check=check_direct_start_recovery_step(
        current,tube_entry,small_alpha,1e-6,1e-2,5e-2,true,true);
    if(!tube_entry_check.accepted() || !tube_entry_check.contracts_theta)
      throw std::runtime_error(
          "terminal direct-start recovery tube entry was rejected");

    // Even a dramatic objective decrease cannot authorize a trial whose
    // exact original-QCQP theta increases: objective values are intentionally
    // absent from the h-only predicate.
    const double current_objective=10.0;
    const double lower_trial_objective=0.0;
    Feasibility theta_up=current;
    theta_up.inequality_inf=9e-3;
    theta_up.total_inf=9e-3;
    const auto theta_up_check=check_direct_start_recovery_step(
        current,theta_up,1.0,1e-6,1e-2,5e-2,true,true);
    if(!(lower_trial_objective<current_objective) ||
       theta_up_check.accepted() || theta_up_check.contracts_theta)
      throw std::runtime_error(
          "objective decrease compensated for increased theta");
    const auto unmodeled_check=check_direct_start_recovery_step(
        current,sufficient,1.0,1e-6,1e-2,5e-2,true,false);
    if(unmodeled_check.accepted() ||
       unmodeled_check.all_violated_inequalities_modeled)
      throw std::runtime_error("h-step accepted an unmodeled violated row");

    const Result accepted=Solver(st).solve(side_start());
    if(accepted.dca_single_loop_direct_start_attempts!=1 ||
       accepted.dca_single_loop_direct_start_acceptances!=1 ||
       accepted.dca_single_loop_direct_start_rejections_nonfinite!=0 ||
       accepted.dca_single_loop_direct_start_rejections_bounds!=0 ||
       accepted.dca_single_loop_direct_start_rejections_equalities!=0 ||
       accepted.dca_single_loop_direct_start_rejections_theta!=0)
      throw std::runtime_error("eligible finite start did not bypass Phase I");
    if(std::abs(accepted.dca_single_loop_direct_start_initial_theta-5e-3)>
           1e-14 ||
       accepted.dca_single_loop_direct_start_violated_inequalities!=2 ||
       accepted.dca_single_loop_direct_start_working_set_rows!=2 ||
       accepted.active_inequalities!=2)
      throw std::runtime_error(
          "direct start did not seed every initially violated side row");
    if(accepted.dca_single_loop_direct_start_filter_initializations!=1)
      throw std::runtime_error("direct start did not initialize the filter");
    if(accepted.restoration_iterations!=0 ||
       accepted.restoration_linear_solves!=0)
      throw std::runtime_error("eligible start unexpectedly entered Phase I");

    Settings recovery_settings=st;
    recovery_settings.max_outer_iterations=40;
    recovery_settings.max_inner_iterations=100;
    recovery_settings.dca_extra_proximal_regularization=1.0;
    const Result recovered=Solver(recovery_settings).solve(side_start());
    if(recovered.status!=Status::Solved ||
       recovered.dca_single_loop_direct_start_recovery_attempts<1 ||
       recovered.dca_single_loop_direct_start_recovery_acceptances<1 ||
       recovered.dca_single_loop_direct_start_strict_latches!=1 ||
       recovered.feasibility.total_inf>
           recovery_settings.feasibility_tolerance)
      throw std::runtime_error(
          "direct h-step did not enter and irreversibly latch the strict tube");
    if(recovered.dca_single_loop_retraction_fallbacks!=0 ||
       recovered.lazy_dca_equality_retraction_attempts!=0 ||
       recovered.lazy_dca_condensed_retraction_attempts!=0)
      throw std::runtime_error(
          "single-loop direct recovery scheduled an auxiliary retraction");

    const Result over_cap=Solver(st).solve(side_start(4.0));
    if(over_cap.dca_single_loop_direct_start_attempts!=1 ||
       over_cap.dca_single_loop_direct_start_acceptances!=0 ||
       over_cap.dca_single_loop_direct_start_rejections_theta!=1 ||
       !(over_cap.restoration_initial_feasibility>st.feasibility_tolerance))
      throw std::runtime_error("over-cap start did not retain Phase I");
    if(over_cap.dca_single_loop_direct_start_filter_initializations!=0 ||
       over_cap.dca_single_loop_direct_start_working_set_rows!=0)
      throw std::runtime_error("rejected over-cap start leaked bypass state");

    const Result bad_bound=Solver(st).solve(bound_start());
    if(bad_bound.dca_single_loop_direct_start_attempts!=1 ||
       bad_bound.dca_single_loop_direct_start_acceptances!=0 ||
       bad_bound.dca_single_loop_direct_start_rejections_bounds!=1 ||
       bad_bound.dca_single_loop_direct_start_filter_initializations!=0)
      throw std::runtime_error("bound-violating start did not fail closed");

    const Result bad_equality=Solver(st).solve(equality_start());
    if(bad_equality.dca_single_loop_direct_start_attempts!=1 ||
       bad_equality.dca_single_loop_direct_start_acceptances!=0 ||
       bad_equality.dca_single_loop_direct_start_rejections_equalities!=1 ||
       bad_equality.dca_single_loop_direct_start_filter_initializations!=0 ||
       !(bad_equality.restoration_initial_feasibility>
         st.feasibility_tolerance) ||
       bad_equality.restoration_iterations<1)
      throw std::runtime_error(
          "equality-infeasible start bypassed Phase I side-route guard");

    const Result nonfinite=Solver(st).solve(nonfinite_start());
    if(nonfinite.dca_single_loop_direct_start_attempts!=1 ||
       nonfinite.dca_single_loop_direct_start_acceptances!=0 ||
       nonfinite.dca_single_loop_direct_start_rejections_nonfinite!=1 ||
       nonfinite.dca_single_loop_direct_start_filter_initializations!=0)
      throw std::runtime_error("non-finite original evaluation did not fail closed");

    Settings ordinary=st;
    ordinary.dca_single_loop_primal_dual=false;
    const Result not_single_loop=Solver(ordinary).solve(side_start());
    if(not_single_loop.dca_single_loop_direct_start_attempts!=0 ||
       not_single_loop.dca_single_loop_direct_start_acceptances!=0 ||
       !(not_single_loop.restoration_initial_feasibility>
         ordinary.feasibility_tolerance))
      throw std::runtime_error(
          "direct-start option escaped the single-loop route");

    std::cout << "[PASS] fail-closed single-loop direct start"
              << " theta="
              << accepted.dca_single_loop_direct_start_initial_theta
              << " violated/ws/filter="
              << accepted.dca_single_loop_direct_start_violated_inequalities
              << "/"
              << accepted.dca_single_loop_direct_start_working_set_rows
              << "/"
              << accepted.dca_single_loop_direct_start_filter_initializations
              << " recovery="
              << recovered.dca_single_loop_direct_start_recovery_acceptances
              << " latch="
              << recovered.dca_single_loop_direct_start_strict_latches
              << "\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
