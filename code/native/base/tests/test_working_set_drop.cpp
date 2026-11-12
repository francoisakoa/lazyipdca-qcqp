#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;

int main(){
  QCQPProblem p; p.name="working_set_drop"; p.n=1; p.x0={0.0}; p.lb={-10.0}; p.ub={10.0};
  // min 0.5 (x+2)^2 up to an irrelevant constant: 0.5 x^2 + 2x.
  p.objective.name="obj"; p.objective.linear={2.0}; p.objective.hessian.n=1; p.objective.hessian.entries={{0,0,1.0}};
  // Near-active at the start, but strongly slack at the minimizer x=-2.
  QuadraticFunction g; g.name="spurious"; g.constant=-5e-5; g.linear={1.0}; g.hessian.n=1; p.inequalities.push_back(g);
  Settings st; st.outer_algorithm="legacy-sqp"; st.inner_backend="legacy-ipm"; st.verbose=false; st.max_outer_iterations=20; st.max_inner_iterations=100;
  st.feasibility_tolerance=1e-7; st.stationarity_tolerance=2e-4; st.terminal_kkt_enabled=false;
  st.proximal_regularization=1.0; st.working_set_activation=1e-4; st.working_set_drop_slack=2e-4; st.working_set_drop_multiplier=1e-9;
  Solver solver(st); Result r=solver.solve(p);
  if(r.status!=Status::Solved){std::cerr<<"[FAIL] status="<<status_string(r.status)<<" stat="<<r.model_stationarity_inf<<"\n";return 1;}
  if(r.working_set_drops<1){std::cerr<<"[FAIL] expected a working-set drop, got "<<r.working_set_drops<<"\n";return 2;}
  if(r.feasibility.total_inf>st.feasibility_tolerance){std::cerr<<"[FAIL] feasibility="<<r.feasibility.total_inf<<"\n";return 3;}
  std::cout<<"[PASS] drops="<<r.working_set_drops<<" x="<<r.x[0]<<" stat="<<r.model_stationarity_inf<<"\n";
  return 0;
}
