#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {
QCQPProblem fixture() {
  QCQPProblem problem;
  problem.name="EQUALITY_QP_LARGE_RHO_ORIGINAL_KKT_GMRES";
  problem.n=3;
  problem.x0={0.0,0.0,0.0};
  problem.lb=Vec(3,-std::numeric_limits<double>::infinity());
  problem.ub=Vec(3, std::numeric_limits<double>::infinity());
  problem.objective.name="f";
  problem.objective.linear={0.00506482150846793,-0.07492446090182754,
                            0.00034325554814259};
  problem.objective.hessian.n=3;
  problem.objective.hessian.entries={
      {0,0,0.00022493313960010},{0,1,0.00019405763736605},
      {0,2,0.00102004745339937},{1,1,0.00016746748587407},
      {1,2,0.00088012770828845},{2,2,0.00462600890575227}};

  QuadraticFunction first;
  first.name="h1";
  first.constant=-0.00785554306227220;
  first.linear={0.00708388554811141,-0.06775419489083805,
                0.01035262800475593};
  first.hessian.n=3;
  QuadraticFunction second;
  second.name="h2";
  second.constant=0.00035192342367547;
  second.linear={0.00025585560569825,0.00015282729197326,
                 -0.00096146689664057};
  second.hessian.n=3;
  problem.equalities={first,second};
  return problem;
}

Settings settings() {
  Settings value;
  value.outer_algorithm="lazy-dca";
  value.verbose=false;
  value.restoration_enabled=false;
  value.kkt_adaptive_regularization=true;
  value.kkt_rho_x_floor=1e-2;
  value.kkt_primal_regularization=1e-2;
  value.kkt_rho_x_max=1e-2;
  value.kkt_delta_dual_floor=1e-12;
  value.kkt_dual_regularization=1e-12;
  value.kkt_delta_dual_max=1e-12;
  value.inner_tolerance=1e-8;
  value.feasibility_tolerance=1e-6;
  value.stationarity_tolerance=1e-6;
  value.dual_infeasibility_tolerance=1e-6;
  value.complementarity_tolerance=1e-6;
  value.max_outer_iterations=5;
  value.max_inner_iterations=100;
  return value;
}
}  // namespace

int main() {
  try {
    Settings disabled_settings=settings();
    disabled_settings.equality_qp_original_kkt_gmres=false;
    const Result disabled=Solver(disabled_settings).solve(fixture());
    if(disabled.status==Status::Solved)
      throw std::runtime_error("fixture no longer requires the GMRES rescue");
    if(disabled.equality_qp_original_kkt_gmres_attempts!=0)
      throw std::runtime_error("disabled GMRES path was entered");

    Settings enabled_settings=settings();
    const Result result=Solver(enabled_settings).solve(fixture());
    if(result.status!=Status::Solved)
      throw std::runtime_error("expected solved, got "+
          std::string(status_string(result.status))+" / "+result.message);
    if(!(result.final_kkt_rho_x>=9.99e-3))
      throw std::runtime_error("test did not retain the intended large rho_x");
    if(!(result.last_inner_dual_inf<=enabled_settings.inner_tolerance) ||
       !(result.last_inner_equality_inf<=enabled_settings.inner_tolerance))
      throw std::runtime_error("original equality-QP certificate was not reached");
    if(result.equality_qp_original_kkt_gmres_attempts!=1 ||
       result.equality_qp_original_kkt_gmres_acceptances!=1 ||
       result.equality_qp_original_kkt_gmres_iterations<=0)
      throw std::runtime_error("GMRES rescue counters are inconsistent");

    std::cout << "[PASS] large rho_x is used only as a KKT preconditioner"
              << " rho_x=" << result.final_kkt_rho_x
              << " inner_dual=" << result.last_inner_dual_inf
              << " inner_eq=" << result.last_inner_equality_inf
              << " gmres_it="
              << result.equality_qp_original_kkt_gmres_iterations << "\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
