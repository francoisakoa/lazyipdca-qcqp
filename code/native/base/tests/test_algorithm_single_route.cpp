#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static QCQPProblem stationary_quadratic_equality() {
  QCQPProblem p;
  p.name="AUTO_ALIAS_QUADRATIC_EQUALITY";
  p.n=1;
  p.x0={1.0}; p.lb={-2.0}; p.ub={2.0};
  p.objective.name="f"; p.objective.constant=1.0;
  p.objective.linear={-2.0}; p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,2.0}}; // (x-1)^2
  QuadraticFunction h;
  h.name="h"; h.constant=-1.0; h.linear={0.0}; h.hessian.n=1;
  h.hessian.entries={{0,0,2.0}}; // x^2-1=0
  p.equalities.push_back(h);
  return p;
}

static QCQPProblem stationary_small_nonlinear_inequality() {
  QCQPProblem p;
  p.name="AUTO_ALIAS_SMALL_NONLINEAR";
  p.n=1;
  p.x0={0.0}; p.lb={-2.0}; p.ub={2.0};
  p.objective.name="f"; p.objective.constant=0.0;
  p.objective.linear={0.0}; p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,2.0}}; // x^2
  QuadraticFunction g;
  g.name="g"; g.constant=-1.0; g.linear={0.0}; g.hessian.n=1;
  g.hessian.entries={{0,0,2.0}}; // x^2-1<=0
  p.inequalities.push_back(g);
  return p;
}

int main() {
  try {
    Settings st;
    st.outer_algorithm="auto";
    st.auto_sqp_max_dimension=64; // deliberately nonzero: v0.4.14 ignores it.
    st.verbose=false;
    st.restoration_enabled=false;
    st.max_outer_iterations=20;
    st.max_inner_iterations=200;
    st.feasibility_tolerance=1e-7;
    st.stationarity_tolerance=1e-6;
    st.dual_infeasibility_tolerance=1e-6;
    st.complementarity_tolerance=1e-6;
    st.inner_tolerance=1e-8;

    const Result eq=Solver(st).solve(stationary_quadratic_equality());
    if(eq.effective_outer_algorithm!="lazy-dca")
      throw std::runtime_error("auto alias switched a quadratic equality away from lazy-dca");
    if(eq.status!=Status::Solved || !eq.ipopt_desired_convergence)
      throw std::runtime_error("single-route quadratic-equality smoke did not certify");

    const Result ineq=Solver(st).solve(stationary_small_nonlinear_inequality());
    if(ineq.effective_outer_algorithm!="lazy-dca")
      throw std::runtime_error("auto alias switched a small nonlinear QCQP away from lazy-dca");
    if(ineq.status!=Status::Solved || !ineq.ipopt_desired_convergence)
      throw std::runtime_error("single-route nonlinear-inequality smoke did not certify");

    Settings cmp=st;
    cmp.outer_algorithm="legacy-sqp";
    const Result explicit_cmp=Solver(cmp).solve(stationary_quadratic_equality());
    if(explicit_cmp.effective_outer_algorithm!="legacy-sqp")
      throw std::runtime_error("explicit comparison algorithm was not preserved");

    std::cout << "[PASS] auto is a lazy-dca alias; no structure-dependent route remains\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
