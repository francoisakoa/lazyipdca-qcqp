#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <iostream>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p;
  p.name="elastic_circle";
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

  Settings st;
  st.outer_algorithm="lazy-dca";
  // This test isolates the experimental elastic equality globalization.  Its
  // tiny 11-by-11 barrier KKT system has no pivoting margin and is not a
  // backend-parity contract; sparse LDL is covered separately.  Use the
  // deterministic dense reference backend so an available LDL implementation
  // (notably the validation stub) cannot change the trajectory under test.
  st.linear_backend="native";
  st.dca_equality_model="elastic-split";
  st.verbose=false;
  st.restoration_enabled=false;
  st.restoration_tangent_activation=false;
  st.second_order_correction=false;
  st.dca_equality_retraction=false;
  st.dca_local_corrector=false;
  st.dca_iterated_soc=false;
  st.dca_equality_funnel_auto=false;
  st.max_outer_iterations=1;
  st.max_inner_iterations=300;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-9;
  st.line_search_max_steps=40;
  st.minimum_step_length=1e-12;

  const Result r=Solver(st).solve(p);
  if(r.effective_outer_algorithm!="lazy-dca" || r.lazy_dca_equality_model!="elastic-split") {
    std::cerr << "[FAIL] wrong equality route/model: route="<<r.effective_outer_algorithm
              <<" model="<<r.lazy_dca_equality_model<<"\n";
    return 1;
  }
  if(r.lazy_dca_elastic_subproblems<=0 || r.lazy_dca_elastic_accepted_steps<=0) {
    std::cerr << "[FAIL] elastic path not exercised sub="<<r.lazy_dca_elastic_subproblems
              <<" acc="<<r.lazy_dca_elastic_accepted_steps<<"\n";
    return 2;
  }
  if(r.soc_factorizations!=0 || r.lazy_dca_corrector_attempts!=0) {
    std::cerr << "[FAIL] elastic path unexpectedly stacked SOC/corrector: soc="
              <<r.soc_factorizations<<" corrector="<<r.lazy_dca_corrector_attempts<<"\n";
    return 3;
  }
  if(!(r.objective < 1.0) || r.feasibility.equality_inf > 1.1e-6) {
    std::cerr << "[FAIL] elastic step made no controlled progress: f="<<r.objective
              <<" eq="<<r.feasibility.equality_inf<<"\n";
    return 4;
  }
  std::cout << "[PASS] elastic-split equality path sub="<<r.lazy_dca_elastic_subproblems
            <<" acc="<<r.lazy_dca_elastic_accepted_steps
            <<" eta="<<r.lazy_dca_elastic_last_eta
            <<" f="<<r.objective<<" eq="<<r.feasibility.equality_inf<<"\n";
  return 0;
}
