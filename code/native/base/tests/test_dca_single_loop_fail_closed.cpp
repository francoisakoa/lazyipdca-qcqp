#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem inactive_side_problem() {
  QCQPProblem p;
  p.name="single_loop_empty_dynamic_majorant";
  p.n=1;
  p.x0={0.0};
  p.lb={-std::numeric_limits<double>::infinity()};
  p.ub={std::numeric_limits<double>::infinity()};
  p.objective.name="objective";
  p.objective.linear={-1.0};
  p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1.0}};

  QuadraticFunction inactive;
  inactive.name="inactive_side";
  inactive.constant=-10.0;
  inactive.linear={1.0};
  inactive.hessian.n=1;
  p.inequalities.push_back(std::move(inactive));
  p.validate();
  return p;
}

QCQPProblem quadratic_equality_problem() {
  QCQPProblem p;
  p.name="single_loop_quadratic_equality_terminal_rescue_guard";
  p.n=2;
  p.x0={1.0,0.0};
  p.lb={-2.0,-2.0};
  p.ub={2.0,2.0};
  p.objective.name="f";
  p.objective.constant=1.0;
  p.objective.linear={0.0,-2.0};
  p.objective.hessian.n=2;
  p.objective.hessian.entries={{1,1,2.0}};
  QuadraticFunction circle;
  circle.name="unit_circle";
  circle.constant=-1.0;
  circle.linear={0.0,0.0};
  circle.hessian.n=2;
  circle.hessian.entries={{0,0,2.0},{1,1,2.0}};
  p.equalities.push_back(std::move(circle));
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
  st.max_outer_iterations=2;
  st.max_inner_iterations=100;
  st.feasibility_tolerance=1e-8;
  st.stationarity_tolerance=1e-7;
  st.dual_infeasibility_tolerance=1e-7;
  st.complementarity_tolerance=1e-7;
  st.inner_tolerance=1e-9;
  st.working_set_activation=1e-4;
  st.working_set_dropping=false;
  st.dca_local_corrector=false;
  st.second_order_correction=false;
  st.dca_single_loop_primal_dual=true;
  st.dca_single_loop_inner_steps=1;
  st.dca_single_loop_forcing_eta=0.999999;
  st.dca_extra_proximal_regularization=1.0;
  st.dca_inner_retry_max=0;
  return st;
}

} // namespace

int main() {
  try {
    Settings st=settings();
    st.terminal_kkt_enabled=false;
    const Result empty=Solver(st).solve(inactive_side_problem());
    if(empty.status!=Status::InnerFailure ||
       empty.inner_iterations!=0 || empty.kkt_factorizations!=0 ||
       empty.dca_single_loop_attempts!=0 ||
       empty.dca_single_loop_factorizations!=0 ||
       empty.dca_single_loop_strict_fallbacks!=0 ||
       empty.multiplier_refit_attempts!=0 ||
       empty.message.find(
           "single-loop primal-dual majorant requires at least one inequality row")==
           std::string::npos)
      throw std::runtime_error(
          "empty dynamic majorant did not fail before equality-QP factorization: "+
          empty.message);

    Settings rescue=settings();
    rescue.terminal_kkt_enabled=true;
    rescue.terminal_kkt_start_outer=0;
    rescue.terminal_kkt_trigger_stationarity=10.0;
    rescue.line_search_max_steps=0;
    rescue.max_outer_iterations=1;
    const Result guarded=Solver(rescue).solve(quadratic_equality_problem());
    if(guarded.status!=Status::LineSearchFailure ||
       guarded.dca_single_loop_attempts!=1 ||
       guarded.dca_single_loop_factorizations!=1 ||
       guarded.kkt_factorizations!=guarded.dca_single_loop_factorizations ||
       guarded.terminal_kkt_attempts!=0 ||
       guarded.terminal_kkt_iterations!=0 ||
       guarded.terminal_kkt_accepted_steps!=0 ||
       guarded.terminal_kkt_active_set_drops!=0 ||
       guarded.terminal_kkt_rejected_steps!=0 ||
       guarded.terminal_kkt_skipped_unstable!=0 ||
       guarded.terminal_kkt_active_stability!=0 ||
       guarded.multiplier_refit_attempts!=0 ||
       guarded.dca_single_loop_full_refit_calls!=0 ||
       guarded.x.size()!=2 ||
       std::abs(guarded.x[0]-1.0)>1e-15 ||
       std::abs(guarded.x[1])>1e-15)
      throw std::runtime_error(
          "quadratic-equality single-loop failure entered terminal rescue: "+
          guarded.message+" status="+status_string(guarded.status)+
          " attempts="+std::to_string(guarded.dca_single_loop_attempts)+
          " factors="+std::to_string(guarded.dca_single_loop_factorizations)+
          "/"+std::to_string(guarded.kkt_factorizations)+
          " terminal="+std::to_string(guarded.terminal_kkt_attempts)+
          "/"+std::to_string(guarded.terminal_kkt_iterations)+
          "/"+std::to_string(guarded.terminal_kkt_accepted_steps)+
          " x="+(guarded.x.empty()?std::string("empty"):
                   std::to_string(guarded.x[0])));

    std::cout << "[PASS] single-loop empty-majorant and terminal-rescue guards"
              << " factors="<<guarded.dca_single_loop_factorizations
              << " terminal="<<guarded.terminal_kkt_attempts<<"\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] "<<error.what()<<"\n";
    return 1;
  }
}
