#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/settings.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static QuadraticFunction qfun(std::string name, int n, double c, Vec lin,
                              std::initializer_list<SymTriplet> H) {
  QuadraticFunction q;
  q.name = std::move(name);
  q.constant = c;
  q.linear = std::move(lin);
  q.hessian.n = n;
  q.hessian.entries.assign(H.begin(), H.end());
  return q;
}

static void run_branch(double q0, double rhs_sign) {
  QCQPProblem p;
  p.name = "ABS_LINK_RESTORATION";
  p.n = 3; // q, y, t
  p.x0 = {q0, q0, 7.0}; // deliberately inconsistent auxiliary start
  p.lb = {-10.0, -10.0, 0.0};
  p.ub = { 10.0,  10.0, INFINITY};
  p.objective = qfun("objective", 3, 0.0, Vec(3, 0.0), {});

  // q*t = +/-4. In 1/2 x'Hx, H(q,t)=1 gives q*t.
  p.equalities.push_back(qfun("pipe", 3, -4.0*rhs_sign, Vec(3, 0.0),
                              {{0,2,rhs_sign}}));
  // y-q = 0
  p.equalities.push_back(qfun("linear_coupling", 3, 0.0, {-1.0,1.0,0.0}, {}));
  // t^2-q^2 = 0. Name and algebra match the CUTEst lift adapter.
  p.equalities.push_back(qfun("abs_link_x0", 3, 0.0, Vec(3, 0.0),
                              {{0,0,-2.0},{2,2,2.0}}));
  p.validate();

  Settings st;
  st.verbose = false;
  st.feasibility_tolerance = 1e-9;
  st.restoration_max_iterations = 200;
  st.restoration_max_restarts = 8;
  st.restoration_restart_iterations = 100;

  FeasibilityRestoration restoration(st);
  const auto r = restoration.solve(p, p.x0);
  if (!r.success) throw std::runtime_error("structured abs-link restoration did not converge");
  if (r.final_feasibility > 1e-9) throw std::runtime_error("feasibility contract failed");
  if (std::abs(r.x[2] - std::abs(r.x[0])) > 1e-12)
    throw std::runtime_error("abs-link manifold was not enforced exactly");
  if (std::abs(r.x[1] - r.x[0]) > 1e-9)
    throw std::runtime_error("coupling equality failed");
}

int main() {
  try {
    run_branch(0.5, +1.0);
    run_branch(-0.5, -1.0);
    std::cout << "[PASS] structured abs-link restoration contract\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
