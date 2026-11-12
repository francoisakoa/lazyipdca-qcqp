#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

static QCQPProblem lifted_problem() {
  // A lifted CORE-like geometry: t^2-q^2=0 with t>=0, plus a second
  // quadratic equality q^2+y^2=1.  At (q,t,y)=(1,1,0), the classical SQP
  // linearizations force dq=dt=0 while the objective asks for dy=1.  A
  // one-shot SOC leaves a second-order defect; the guarded iterated SOC must
  // restore both original equalities without changing the outer route.
  QCQPProblem p;
  p.name="lazy_dca_iterated_soc_abs_link";
  p.n=3;
  p.x0={1.0,1.0,0.0};
  p.lb={-2.0,0.0,-2.0};
  p.ub={ 2.0,2.0, 2.0};

  p.objective.name="f";
  p.objective.constant=1.0;
  p.objective.linear={0.0,0.0,-2.0};
  p.objective.hessian.n=3;
  p.objective.hessian.entries={{2,2,2.0}}; // (y-1)^2

  QuadraticFunction link;
  link.name="abs_link_0";
  link.constant=0.0;
  link.linear={0.0,0.0,0.0};
  link.hessian.n=3;
  link.hessian.entries={{0,0,-2.0},{1,1,2.0}}; // t^2-q^2=0
  p.equalities.push_back(std::move(link));

  QuadraticFunction h;
  h.name="unit_circle";
  h.constant=-1.0;
  h.linear={0.0,0.0,0.0};
  h.hessian.n=3;
  h.hessian.entries={{0,0,2.0},{2,2,2.0}}; // q^2+y^2=1
  p.equalities.push_back(std::move(h));
  return p;
}

static Settings base_settings() {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.verbose=false;
  st.restoration_enabled=false;
  st.dca_local_corrector=false;
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
  st.second_order_correction=true;
  st.dca_iterated_soc_max_iterations=12;
  st.dca_iterated_soc_max_ratio=2.0;
  st.dca_iterated_soc_trigger_ratio=1e6;
  return st;
}

int main() {
  const QCQPProblem p=lifted_problem();

  Settings one=base_settings();
  one.dca_iterated_soc=false;
  const Result r1=Solver(one).solve(p);

  Settings iter=base_settings();
  iter.dca_iterated_soc=true;
  const Result r2=Solver(iter).solve(p);

  if(r1.effective_outer_algorithm!="lazy-dca" ||
     r2.effective_outer_algorithm!="lazy-dca") {
    std::cerr << "[FAIL] outer route changed\n";
    return 1;
  }
  if(r2.lazy_dca_linearized_equality_steps!=1) {
    std::cerr << "[FAIL] equality was not handled by its SQP linearization\n";
    return 2;
  }
  if(r2.lazy_dca_iterated_soc_attempts<=0 ||
     r2.lazy_dca_iterated_soc_acceptances<=0 ||
     r2.soc_accepted_steps<=0) {
    std::cerr << "[FAIL] iterated SOC was not exercised/accepted: "
              << r2.lazy_dca_iterated_soc_attempts << "/"
              << r2.lazy_dca_iterated_soc_acceptances << " soc="
              << r2.soc_accepted_steps << "\n";
    return 3;
  }
  if(r2.feasibility.equality_inf>iter.feasibility_tolerance) {
    std::cerr << "[FAIL] original equalities not restored: "
              << r2.feasibility.equality_inf << "\n";
    return 4;
  }
  if(!(r2.lazy_dca_last_equality_step_alpha>
       8.0*r1.lazy_dca_last_equality_step_alpha)) {
    std::cerr << "[FAIL] iterated SOC did not materially enlarge the accepted step: one="
              << r1.lazy_dca_last_equality_step_alpha << " iter="
              << r2.lazy_dca_last_equality_step_alpha << "\n";
    return 5;
  }
  if(!(r2.objective < r1.objective)) {
    std::cerr << "[FAIL] iterated SOC objective did not improve: one="
              << r1.objective << " iter=" << r2.objective << "\n";
    return 6;
  }

  std::cout << "[PASS] linearized lifted equalities + guarded iterated SOC"
            << " alpha_one=" << r1.lazy_dca_last_equality_step_alpha
            << " alpha_iter=" << r2.lazy_dca_last_equality_step_alpha
            << " eq=" << r2.feasibility.equality_inf
            << " f=" << r2.objective << "\n";
  return 0;
}
