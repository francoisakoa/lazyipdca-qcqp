#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

int main() {
  // Start exactly on x0^2+x1^2=1.  The convex objective asks for d1=1 while
  // the classical SQP equality linearization at (1,0) imposes d0=0.  With SOC
  // disabled, a systematic nonlinear projection would accept alpha=1; the
  // single-route implementation must instead backtrack the linearized step
  // until the ORIGINAL quadratic equality remains inside the validated tube.
  QCQPProblem p;
  p.name="lazy_dca_linearized_circle";
  p.n=2;
  p.x0={1.0,0.0};
  p.lb={-2.0,-2.0}; p.ub={2.0,2.0};
  p.objective.name="f";
  p.objective.constant=1.0;
  p.objective.linear={0.0,-2.0};
  p.objective.hessian.n=2;
  p.objective.hessian.entries={{1,1,2.0}}; // (x1-1)^2
  QuadraticFunction h;
  h.name="unit_circle"; h.constant=-1.0; h.linear={0.0,0.0}; h.hessian.n=2;
  h.hessian.entries={{0,0,2.0},{1,1,2.0}};
  p.equalities.push_back(h);

  Settings st;
  st.outer_algorithm="lazy-dca";
  st.verbose=false;
  st.restoration_enabled=false;
  st.second_order_correction=false;
  st.dca_equality_retraction=false;
  st.dca_equality_funnel_auto=false;
  st.max_outer_iterations=1;
  st.max_inner_iterations=300;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-8;
  st.dual_infeasibility_tolerance=1e-8;
  st.complementarity_tolerance=1e-8;
  st.inner_tolerance=1e-10;
  st.line_search_max_steps=40;
  st.minimum_step_length=1e-12;

  const Result r=Solver(st).solve(p);
  if(r.effective_outer_algorithm!="lazy-dca") {
    std::cerr << "[FAIL] wrong route: " << r.effective_outer_algorithm << "\n";
    return 1;
  }
  if(r.lazy_dca_linearized_equality_steps!=1) {
    std::cerr << "[FAIL] linearized equality step counter="
              << r.lazy_dca_linearized_equality_steps << "\n";
    return 2;
  }
  if(!(r.lazy_dca_last_equality_step_alpha<1.0) ||
     !(r.lazy_dca_last_equality_step_alpha>0.0)) {
    std::cerr << "[FAIL] expected a genuine backtracked SQP-linearized step, alpha="
              << r.lazy_dca_last_equality_step_alpha << "\n";
    return 3;
  }
  if(r.soc_attempts!=0 || r.soc_accepted_steps!=0) {
    std::cerr << "[FAIL] SOC was disabled but used\n";
    return 4;
  }
  if(r.feasibility.equality_inf>st.feasibility_tolerance) {
    std::cerr << "[FAIL] original quadratic equality violation="
              << r.feasibility.equality_inf << "\n";
    return 5;
  }
  if(r.line_search_backtracks<1) {
    std::cerr << "[FAIL] no line-search backtrack; a hidden projection may remain\n";
    return 6;
  }
  std::cout << "[PASS] quadratic equality linearized, alpha="
            << r.lazy_dca_last_equality_step_alpha
            << " backtracks=" << r.line_search_backtracks
            << " eq=" << r.feasibility.equality_inf << "\n";
  return 0;
}
