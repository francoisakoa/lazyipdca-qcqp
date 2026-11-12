#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p;
  p.name="flat_tangent_majorant";
  p.n=2;
  p.x0={1.0,0.0};
  p.lb={-100.0,-100.0};
  p.ub={100.0,100.0};
  p.objective.name="linear_descent";
  p.objective.constant=0.0;
  p.objective.linear={0.0,-10.0};
  p.objective.hessian.n=2;
  QuadraticFunction h;
  h.name="unit_circle";
  h.constant=-1.0;
  h.linear={0.0,0.0};
  h.hessian.n=2;
  h.hessian.entries={{0,0,2.0},{1,1,2.0}};
  p.equalities.push_back(h);

  Settings st;
  st.outer_algorithm="lazy-dca";
  st.verbose=false;
  st.restoration_enabled=false;
  st.max_outer_iterations=1;
  st.max_inner_iterations=200;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-10;
  st.dca_local_corrector=false;
  st.second_order_correction=false;
  st.dca_equality_retraction=true;
  st.dca_step_control=true;
  st.dca_step_target_absolute=1.0;
  st.dca_step_target_relative=0.01;
  st.dca_step_target_maximum=2.0;
  st.dca_step_control_max_retries=8;

  const Result r=Solver(st).solve(p);
  if(r.effective_outer_algorithm!="lazy-dca") {
    std::cerr << "[FAIL] route=" << r.effective_outer_algorithm << "\n";
    return 1;
  }
  if(r.lazy_dca_step_control_predictions+r.lazy_dca_step_control_retries<=0) {
    std::cerr << "[FAIL] oversized flat direction was not controlled proximally\n";
    return 2;
  }
  if(!(r.lazy_dca_last_raw_direction_inf<=
       1.05*r.lazy_dca_last_step_target_inf)) {
    std::cerr << "[FAIL] accepted raw direction="
              << r.lazy_dca_last_raw_direction_inf << " target="
              << r.lazy_dca_last_step_target_inf << "\n";
    return 3;
  }
  if(!(r.lazy_dca_max_model_proximal>1.0)) {
    std::cerr << "[FAIL] explicit model proximal term not increased: "
              << r.lazy_dca_max_model_proximal << "\n";
    return 4;
  }
  if(r.feasibility.equality_inf>st.feasibility_tolerance) {
    std::cerr << "[FAIL] equality violation=" << r.feasibility.equality_inf << "\n";
    return 5;
  }
  std::cout << "[PASS] step control predict/retry="
            << r.lazy_dca_step_control_predictions << "/"
            << r.lazy_dca_step_control_retries << " raw="
            << r.lazy_dca_last_raw_direction_inf << " target="
            << r.lazy_dca_last_step_target_inf << " prox="
            << r.lazy_dca_max_model_proximal << " alpha="
            << r.lazy_dca_last_equality_step_alpha << "\n";
  return 0;
}
