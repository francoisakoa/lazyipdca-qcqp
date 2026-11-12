#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
#include <sstream>
#include <string>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p;
  p.name = "qp_dual_multiplier_path";
  p.n = 2;
  p.x0 = {0.4, 0.84};
  p.lb = {-2.0, -2.0};
  p.ub = { 2.0,  2.0};

  p.objective.name = "objective";
  p.objective.constant = 0.0;
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
  st.verbose = true;
  st.max_outer_iterations = 40;
  st.max_inner_iterations = 200;
  st.feasibility_tolerance = 1e-7;
  st.stationarity_tolerance = 1e-6;
  st.inner_tolerance = 1e-7;
  st.exact_eigen_threshold = 50;
  st.max_rank_per_quadratic = 8;

  std::ostringstream capture;
  auto* old = std::cout.rdbuf(capture.rdbuf());
  Solver solver(st);
  Result r = solver.solve(p);
  std::cout.rdbuf(old);

  const std::string log = capture.str();
  if (log.find("mult=legacy") == std::string::npos) {
    std::cerr << "[FAIL] initial legacy multiplier phase not observed\n";
    return 1;
  }
  if (log.find("mult=qpdual") == std::string::npos) {
    std::cerr << "[FAIL] QP-dual multiplier update not observed\n";
    return 2;
  }
  if (r.status != Status::Solved) {
    std::cerr << "[FAIL] expected solved, got " << status_string(r.status) << "\n";
    return 3;
  }
  if (r.feasibility.total_inf > st.feasibility_tolerance ||
      r.model_stationarity_inf > st.stationarity_tolerance) {
    std::cerr << "[FAIL] terminal KKT residuals too large: feas=" << r.feasibility.total_inf
              << " stat=" << r.model_stationarity_inf << "\n";
    return 4;
  }
  std::cout << "[PASS] qp-dual multiplier path f=" << r.objective
            << " feas=" << r.feasibility.total_inf
            << " stat=" << r.model_stationarity_inf << "\n";
  return 0;
}
