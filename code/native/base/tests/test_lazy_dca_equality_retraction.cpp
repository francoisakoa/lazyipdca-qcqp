#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

static QCQPProblem circle_problem() {
  QCQPProblem p;
  p.name="lazy_dca_explicit_equality_retraction";
  p.n=2;
  p.x0={1.0,0.0};
  p.lb={-2.0,-2.0}; p.ub={2.0,2.0};
  p.objective.name="f";
  p.objective.constant=1.0;
  p.objective.linear={0.0,-2.0};
  p.objective.hessian.n=2;
  p.objective.hessian.entries={{1,1,2.0}};
  QuadraticFunction h;
  h.name="unit_circle"; h.constant=-1.0; h.linear={0.0,0.0}; h.hessian.n=2;
  h.hessian.entries={{0,0,2.0},{1,1,2.0}};
  p.equalities.push_back(h);
  return p;
}

static Settings base() {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.verbose=false;
  st.restoration_enabled=false;
  st.dca_local_corrector=false;
  st.second_order_correction=false;
  st.max_outer_iterations=1;
  st.max_inner_iterations=300;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-8;
  st.dual_infeasibility_tolerance=1e-8;
  st.complementarity_tolerance=1e-8;
  st.inner_tolerance=1e-10;
  st.line_search_max_steps=40;
  st.minimum_step_length=1e-12;
  return st;
}

int main() {
  const QCQPProblem p=circle_problem();
  Settings strict=base();
  strict.dca_equality_retraction=false;
  strict.dca_equality_funnel_auto=false;
  const Result r0=Solver(strict).solve(p);

  Settings guarded=base();
  guarded.dca_equality_retraction=true;
  guarded.dca_equality_funnel_auto=true;
  guarded.dca_equality_retraction_max_ratio=2.0;
  const Result r1=Solver(guarded).solve(p);

  if(r1.effective_outer_algorithm!="lazy-dca") {
    std::cerr << "[FAIL] route changed\n"; return 1;
  }
  if(r1.lazy_dca_linearized_equality_steps!=1) {
    std::cerr << "[FAIL] primary equality was not linearized\n"; return 2;
  }
  if(r1.lazy_dca_equality_retraction_attempts<=0 ||
     r1.lazy_dca_equality_retraction_acceptances<=0) {
    std::cerr << "[FAIL] explicit retraction not accepted\n"; return 3;
  }
  if(r1.feasibility.equality_inf>guarded.feasibility_tolerance) {
    std::cerr << "[FAIL] retracted point infeasible: "
              << r1.feasibility.equality_inf << "\n"; return 4;
  }
  if(!(r1.lazy_dca_last_equality_step_alpha>
       100.0*r0.lazy_dca_last_equality_step_alpha)) {
    std::cerr << "[FAIL] retraction did not prevent step collapse: strict="
              << r0.lazy_dca_last_equality_step_alpha << " retract="
              << r1.lazy_dca_last_equality_step_alpha << "\n"; return 5;
  }
  if(!(r1.objective<r0.objective)) {
    std::cerr << "[FAIL] objective not improved\n"; return 6;
  }
  std::cout << "[PASS] explicit SQP equality retraction preserves Lazy-DCA route"
            << " strict_alpha=" << r0.lazy_dca_last_equality_step_alpha
            << " retract_alpha=" << r1.lazy_dca_last_equality_step_alpha
            << " correction=" << r1.lazy_dca_max_equality_retraction_inf
            << " eq=" << r1.feasibility.equality_inf << "\n";
  return 0;
}
