#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>

using namespace lazyipdca::qcqp;

static QCQPProblem problem_at_equality_and_inequality_boundary() {
  QCQPProblem p;
  p.name="dca_soc_inequality_defect_continuity";
  p.n=3; // q, t, y
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

  // The current point is 0.99e-6 infeasible but inside the configured final
  // tube.  The tangent step has a large y component, so this equality creates
  // the Maratos defect that invokes the one-shot SOC.
  QuadraticFunction h;
  h.name="weak_boundary_equality";
  h.constant=-0.004+0.99e-6;
  h.linear={0.002,0.002,0.0};
  h.hessian.n=3;
  h.hessian.entries={{2,2,160.0}};
  p.equalities.push_back(std::move(h));

  // The scaled linearized model of this inequality remains positive for the
  // small line-search steps used near the equality boundary.  Its nonlinear
  // defect is only O(alpha^2).  A SOC that imposes g(z)+Jp<=0 erases the
  // pre-existing 0.99e-6 residual and develops a non-vanishing correction
  // floor: the accepted step is 2^-17 and no SOC is accepted.  The corrected
  // all-constraint defect model
  //
  //   g(z)+Jp <= max(0, g(x)+J(x)(z-x))
  //
  // accepts a SOC step of at least 2^-15 on this deterministic instance.
  QuadraticFunction g;
  g.name="positive_scaled_model_residual";
  g.constant=0.99e-6;
  g.linear={0.0,0.0,-2.0e-6};
  g.hessian.n=3;
  g.hessian.entries={{2,2,1.0e-6}}; // +0.5e-6 y^2
  p.inequalities.push_back(std::move(g));
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
  st.line_search_max_steps=40;
  st.minimum_step_length=1e-12;
  st.second_order_correction=true;
  st.second_order_correction_max_ratio=2.0;
  // Exercise the fallback minimum-norm SOC QP directly; the preliminary
  // iterated equality-normal correction is covered by separate tests.
  st.dca_iterated_soc=false;
  st.second_order_correction_trigger_ratio=1e12;

  const Result r=Solver(st).solve(problem_at_equality_and_inequality_boundary());
  if(r.effective_outer_algorithm!="lazy-dca") {
    std::cerr << "[FAIL] route changed\n";
    return 1;
  }
  if(r.lazy_dca_linearized_equality_steps!=1 || r.soc_accepted_steps<=0) {
    std::cerr << "[FAIL] all-constraint defect SOC not used: eqsteps="
              << r.lazy_dca_linearized_equality_steps
              << " soc=" << r.soc_accepted_steps << " status="
              << status_string(r.status) << "\n";
    return 2;
  }
  if(r.feasibility.total_inf>st.feasibility_tolerance) {
    std::cerr << "[FAIL] final original-QCQP tube lost: "
              << r.feasibility.total_inf << "\n";
    return 3;
  }
  if(!(r.lazy_dca_last_equality_step_alpha>=3.0e-5)) {
    std::cerr << "[FAIL] inequality residual created a non-vanishing SOC floor: alpha="
              << r.lazy_dca_last_equality_step_alpha << " status="
              << status_string(r.status) << "\n";
    return 4;
  }

  std::cout << "[PASS] SOC preserves a positive scaled inequality residual"
            << " alpha=" << r.lazy_dca_last_equality_step_alpha
            << " theta=" << r.feasibility.total_inf
            << " soc=" << r.soc_accepted_steps << "\n";
  return 0;
}
