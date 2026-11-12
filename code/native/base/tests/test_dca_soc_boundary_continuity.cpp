#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>

using namespace lazyipdca::qcqp;

static QCQPProblem problem_at_feasibility_boundary() {
  QCQPProblem p;
  p.name="dca_soc_boundary_continuity";
  p.n=3;
  p.x0={1.0,1.0,0.0};
  p.lb={-2.0,0.0,-2.0};
  p.ub={ 2.0,2.0, 2.0};

  p.objective.constant=1.0;
  p.objective.linear={0.0,0.0,-2.0};
  p.objective.hessian.n=3;
  p.objective.hessian.entries={{2,2,2.0}}; // (y-1)^2

  QuadraticFunction link;
  link.name="abs_link_0";
  link.linear={0.0,0.0,0.0};
  link.hessian.n=3;
  link.hessian.entries={{0,0,-2.0},{1,1,2.0}};
  p.equalities.push_back(std::move(link));

  // At x0 this row has residual 0.99e-6 and a weak normal derivative 0.004
  // along the lifted manifold.  The tangent objective step produces a strong
  // quadratic defect 80*y^2.  A full-restoration SOC has a fixed ~2.5e-4
  // correction as alpha->0 and accepts only an uncorrected 2^-17 step.  A
  // scaled-linearization defect SOC vanishes with alpha and accepts 2^-2.
  QuadraticFunction h;
  h.name="weak_boundary_equality";
  h.constant=-0.004+0.99e-6;
  h.linear={0.002,0.002,0.0};
  h.hessian.n=3;
  h.hessian.entries={{2,2,160.0}};
  p.equalities.push_back(std::move(h));
  return p;
}

int main() {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.verbose=false;
  st.restoration_enabled=false;
  st.dca_equality_retraction=false;
  st.dca_equality_funnel_auto=false;
  st.dca_local_corrector=false;
  st.max_outer_iterations=1;
  st.max_inner_iterations=500;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-8;
  st.dual_infeasibility_tolerance=1e-8;
  st.complementarity_tolerance=1e-8;
  st.inner_tolerance=1e-10;
  st.line_search_max_steps=32;
  st.minimum_step_length=1e-10;
  st.second_order_correction=true;
  st.second_order_correction_max_ratio=2.0;
  st.dca_iterated_soc=true;
  st.dca_iterated_soc_max_iterations=12;
  st.dca_iterated_soc_max_ratio=2.0;
  st.dca_iterated_soc_trigger_ratio=1e12;

  const Result r=Solver(st).solve(problem_at_feasibility_boundary());
  if(r.effective_outer_algorithm!="lazy-dca") {
    std::cerr << "[FAIL] route changed\n";
    return 1;
  }
  if(r.lazy_dca_linearized_equality_steps!=1 || r.soc_accepted_steps<=0) {
    std::cerr << "[FAIL] defect SOC not used: eqsteps="
              << r.lazy_dca_linearized_equality_steps
              << " soc=" << r.soc_accepted_steps << "\n";
    return 2;
  }
  if(r.feasibility.total_inf>st.feasibility_tolerance) {
    std::cerr << "[FAIL] final original-QCQP tube lost: "
              << r.feasibility.total_inf << "\n";
    return 3;
  }
  if(!(r.lazy_dca_last_equality_step_alpha>=3.0e-5)) {
    std::cerr << "[FAIL] SOC still has a non-vanishing correction floor: alpha="
              << r.lazy_dca_last_equality_step_alpha << "\n";
    return 4;
  }

  std::cout << "[PASS] boundary SOC is continuous in the scaled step"
            << " alpha=" << r.lazy_dca_last_equality_step_alpha
            << " theta=" << r.feasibility.total_inf
            << " soc=" << r.soc_accepted_steps << "\n";
  return 0;
}
