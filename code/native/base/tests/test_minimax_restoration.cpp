#include "lazyipdca_qcqp/restoration.hpp"
#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

static QuadraticFunction quadratic_constraint(int n, int j, double a, double c, double hdiag,
                                              const std::string& name) {
  QuadraticFunction q;
  q.name = name;
  q.constant = c;
  q.linear.assign(n, 0.0);
  q.linear[j] = a;
  q.hessian.n = n;
  if (hdiag != 0.0) q.hessian.entries.push_back({j, j, hdiag});
  return q;
}

int main() {
  // Deliberately bypass the L2 Newton budget so the test exercises the
  // terminal L-infinity epigraph restoration directly on a many-inequality
  // model.  The feasible set is x_j >= 1 for all j.
  QCQPProblem p;
  p.name = "minimax_many_inequalities";
  p.n = 80;
  p.x0.assign(p.n, 0.0);
  p.lb.assign(p.n, -INFINITY);
  p.ub.assign(p.n, INFINITY);
  p.objective.name = "obj";
  p.objective.linear.assign(p.n, 0.0);
  p.objective.hessian.n = p.n;

  for (int rep = 0; rep < 3; ++rep) {
    for (int j = 0; j < p.n; ++j) {
      // 1 - x_j + 0.01*x_j^2 <= 0 (H_jj=0.02 in 0.5*x^T H x convention).
      p.inequalities.push_back(quadratic_constraint(
          p.n, j, -1.0, 1.0, 0.02, "g_" + std::to_string(rep) + "_" + std::to_string(j)));
    }
  }
  p.validate();

  Settings st;
  st.restoration_max_iterations = 0;
  st.restoration_max_restarts = 0;
  st.restoration_minimax_enabled = true;
  st.restoration_minimax_max_iterations = 8;
  st.restoration_minimax_proximal = 1e-8;
  st.max_inner_iterations = 300;
  st.inner_tolerance = 1e-8;
  st.feasibility_tolerance = 1e-7;
  st.verbose = false;

  FeasibilityRestoration restoration(st);
  RestorationResult r = restoration.solve(p, p.x0);

  if (!r.success) {
    std::cerr << "[FAIL] minimax restoration did not converge: theta=" << r.final_feasibility << "\n";
    return 1;
  }
  if (r.minimax_qp_solves <= 0 || r.minimax_accepted_steps <= 0) {
    std::cerr << "[FAIL] minimax path was not exercised\n";
    return 2;
  }
  if (r.final_feasibility > st.feasibility_tolerance) {
    std::cerr << "[FAIL] final feasibility=" << r.final_feasibility << "\n";
    return 3;
  }
  std::cout << "[PASS] minimax theta=" << r.final_feasibility
            << " qp=" << r.minimax_qp_solves
            << " inner=" << r.minimax_inner_iterations
            << " fact=" << r.minimax_factorizations
            << " accept=" << r.minimax_accepted_steps << "\n";
  return 0;
}
