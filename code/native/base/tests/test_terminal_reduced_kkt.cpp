#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p;
  p.name = "terminal_reduced_kkt";
  p.n = 2;
  p.x0 = {0.4, 0.84};
  p.lb = {-2.0, -2.0};
  p.ub = { 2.0,  2.0};

  p.objective.name = "objective";
  p.objective.linear = {-0.2, 0.0};
  p.objective.hessian.n = 2;
  p.objective.hessian.entries = {{0,0,-1.0}, {1,1,0.5}};

  QuadraticFunction h;
  h.name = "eq";
  h.constant = -1.0;
  h.linear = {0.0, 1.0};
  h.hessian.n = 2;
  h.hessian.entries = {{0,0,2.0}};
  p.equalities.push_back(h);

  QuadraticFunction g;
  g.name = "ineq";
  g.constant = -0.8;
  g.linear = {1.0, 0.0};
  g.hessian.n = 2;
  g.hessian.entries = {{1,1,-1.0}};
  p.inequalities.push_back(g);

  Settings st;
  st.outer_algorithm = "legacy-sqp";
  st.terminal_kkt_enabled = true;
  st.verbose = false;
  st.max_outer_iterations = 40;
  st.max_inner_iterations = 200;
  st.feasibility_tolerance = 1e-7;
  st.stationarity_tolerance = 1e-6;
  st.inner_tolerance = 1e-7;
  st.exact_eigen_threshold = 50;
  st.max_rank_per_quadratic = 8;
  st.max_lagrangian_rank = 8;
  st.terminal_kkt_start_outer = 0;
  st.terminal_kkt_trigger_stationarity = 1e3;
  st.terminal_kkt_max_iterations = 12;

  Solver solver(st);
  Result r = solver.solve(p);
  if (r.status != Status::Solved) {
    std::cerr << "[FAIL] terminal KKT problem not solved: " << status_string(r.status)
              << " stat=" << r.model_stationarity_inf << "\n";
    return 1;
  }
  if (r.terminal_kkt_iterations <= 0 || r.terminal_kkt_accepted_steps <= 0) {
    std::cerr << "[FAIL] reduced KKT phase was not exercised: it=" << r.terminal_kkt_iterations
              << " accept=" << r.terminal_kkt_accepted_steps << "\n";
    return 2;
  }
  if (r.feasibility.total_inf > st.feasibility_tolerance ||
      r.model_stationarity_inf > st.stationarity_tolerance) {
    std::cerr << "[FAIL] residual contract: feas=" << r.feasibility.total_inf
              << " stat=" << r.model_stationarity_inf << "\n";
    return 3;
  }
  std::cout << "[PASS] reduced KKT it=" << r.terminal_kkt_iterations
            << " accept=" << r.terminal_kkt_accepted_steps
            << " jacrank=" << r.active_jacobian_rank
            << " tangent=" << r.tangent_dimension
            << " redneg=" << r.reduced_negative_eigenvalues
            << " stat=" << r.model_stationarity_inf << "\n";
  return 0;
}
