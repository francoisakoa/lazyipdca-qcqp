#include "lazyipdca_qcqp/solver.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;

int main(){
  // Classic Maratos geometry: minimize -x2 on the unit disk.  Once the QP-dual
  // path activates, the tangent half-space admits a tangential step whose
  // nonlinear constraint defect is O(||d||^2); the SOC must repair it without
  // falling back to the objective-blind Phase-I polisher.
  QCQPProblem p; p.name="soc_unit_disk"; p.n=2; p.x0={1.0,0.0};
  p.lb={-2.0,-2.0}; p.ub={2.0,2.0};
  p.objective.name="obj"; p.objective.linear={0.0,-1.0}; p.objective.hessian.n=2;
  QuadraticFunction g; g.name="unit_disk"; g.constant=-1.0; g.linear={0.0,0.0};
  g.hessian.n=2; g.hessian.entries={{0,0,2.0},{1,1,2.0}}; p.inequalities.push_back(g);

  Settings st; st.outer_algorithm="legacy-sqp"; st.verbose=false; st.max_outer_iterations=300; st.max_inner_iterations=300;
  st.feasibility_tolerance=1e-7; st.stationarity_tolerance=1e-4;
  st.proximal_regularization=10.0; st.terminal_kkt_enabled=false;
  st.phase2_linearized_inequalities=true; st.second_order_correction=true;

  Result r=Solver(st).solve(p);
  if(r.status!=Status::Solved){
    std::cerr<<"[FAIL] SOC problem status="<<status_string(r.status)
             <<" feas="<<r.feasibility.total_inf<<" stat="<<r.model_stationarity_inf<<"\n";
    return 1;
  }
  if(r.soc_attempts<1 || r.soc_accepted_steps<1){
    std::cerr<<"[FAIL] SOC path not exercised attempts="<<r.soc_attempts
             <<" accepted="<<r.soc_accepted_steps<<"\n";
    return 2;
  }
  if(r.feasibility.total_inf>st.feasibility_tolerance){
    std::cerr<<"[FAIL] SOC feasibility="<<r.feasibility.total_inf<<"\n";
    return 3;
  }
  std::cout<<"[PASS] SOC attempts="<<r.soc_attempts
           <<" accept="<<r.soc_accepted_steps
           <<" backtracks="<<r.line_search_backtracks
           <<" stat="<<r.model_stationarity_inf<<"\n";

  // Regression for v0.2.13: force the SOC size guard to reject every useful
  // correction.  The feasible-axis line search must fall back to the proven
  // local feasibility repair and still converge; it must not collapse to tiny
  // alpha merely because SOC was rejected.
  Settings sf=st;
  sf.second_order_correction_max_ratio=1e-14;
  Result rf=Solver(sf).solve(p);
  if(rf.status!=Status::Solved || rf.feasibility.total_inf>sf.feasibility_tolerance){
    std::cerr<<"[FAIL] rejected-SOC fallback status="<<status_string(rf.status)
             <<" feas="<<rf.feasibility.total_inf
             <<" stat="<<rf.model_stationarity_inf<<"\n";
    return 4;
  }
  if(rf.soc_attempts<1 || rf.soc_accepted_steps!=0 || rf.restoration_iterations<1){
    std::cerr<<"[FAIL] rejected-SOC fallback not exercised attempts="<<rf.soc_attempts
             <<" accepted="<<rf.soc_accepted_steps
             <<" repair_it="<<rf.restoration_iterations<<"\n";
    return 5;
  }
  std::cout<<"[PASS] rejected-SOC fallback attempts="<<rf.soc_attempts
           <<" repair_it="<<rf.restoration_iterations
           <<" backtracks="<<rf.line_search_backtracks
           <<" stat="<<rf.model_stationarity_inf<<"\n";
  return 0;
}
