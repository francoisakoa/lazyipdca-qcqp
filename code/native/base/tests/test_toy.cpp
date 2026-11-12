#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p;
  p.name = "toy_nonconvex_qcqp";
  p.n = 2;
  p.x0 = {0.4, 0.84};
  p.lb = {-2.0, -2.0};
  p.ub = { 2.0,  2.0};

  // min -0.5*x0^2 + 0.25*x1^2 - 0.2*x0
  p.objective.name = "objective";
  p.objective.constant = 0.0;
  p.objective.linear = {-0.2, 0.0};
  p.objective.hessian.n = 2;
  p.objective.hessian.entries = {{0,0,-1.0}, {1,1,0.5}};

  // quadratic equality x0^2 + x1 - 1 = 0
  QuadraticFunction h;
  h.name = "circle_like_eq";
  h.constant = -1.0;
  h.linear = {0.0, 1.0};
  h.hessian.n = 2;
  h.hessian.entries = {{0,0,2.0}};
  p.equalities.push_back(h);

  // nonconvex inequality -0.5*x1^2 + x0 - 0.8 <= 0
  QuadraticFunction g;
  g.name = "nonconvex_ineq";
  g.constant = -0.8;
  g.linear = {1.0, 0.0};
  g.hessian.n = 2;
  g.hessian.entries = {{1,1,-1.0}};
  p.inequalities.push_back(g);

  Settings st;
  st.verbose = false;
  st.max_outer_iterations = 80;
  st.max_inner_iterations = 100;
  st.feasibility_tolerance = 1e-6;
  st.inner_tolerance = 1e-7;
  st.step_tolerance = 1e-8;
  st.exact_eigen_threshold = 50;
  st.max_rank_per_quadratic = 8;

  Solver solver(st);
  Result r = solver.solve(p);
  std::cout << "status=" << status_string(r.status)
            << " f=" << r.objective
            << " feas=" << r.feasibility.total_inf
            << " rank0=" << r.objective_rank
            << " rankI=" << r.inequality_total_rank << "\n";

  if (r.feasibility.total_inf > 5e-5) return 1;
  if (r.objective_rank < 1) return 2;
  if (r.inequality_total_rank < 1) return 3;
  return 0;
}
