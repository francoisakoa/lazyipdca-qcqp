#include "lazyipdca_qcqp/restoration.hpp"

#include <cmath>
#include <iostream>
#include <limits>

using namespace lazyipdca::qcqp;

int main() {
  constexpr int n=520;
  constexpr int me=500;
  QCQPProblem p;
  p.name="near_feasible_sparse_inequality_restoration";
  p.n=n;
  p.x0.assign(n,0.0);
  p.lb.assign(n,-std::numeric_limits<double>::infinity());
  p.ub.assign(n, std::numeric_limits<double>::infinity());
  p.objective.name="objective";
  p.objective.linear.assign(n,0.0);
  p.objective.hessian.n=n;

  // Sparse equalities preserve a long feasible manifold.  The initial point
  // satisfies every equality, but one sparse inequality needs a small move
  // along that manifold, mirroring a power-flow start with a violated voltage
  // lower bound.
  for(int i=0;i<me;++i) {
    QuadraticFunction h;
    h.name="link_"+std::to_string(i);
    h.linear.assign(n,0.0);
    h.linear[i]=1.0;
    h.linear[i+1]=-1.0;
    h.hessian.n=n;
    p.equalities.push_back(std::move(h));
  }
  QuadraticFunction active;
  active.name="active_floor";
  active.constant=5e-3;
  active.linear.assign(n,0.0);
  active.linear[me]=-1.0; // 5e-3 - x[me] <= 0
  active.hessian.n=n;
  p.inequalities.push_back(std::move(active));
  for(int i=0;i<32;++i) {
    QuadraticFunction inactive;
    inactive.name="inactive_"+std::to_string(i);
    inactive.constant=-1.0;
    inactive.linear.assign(n,0.0);
    inactive.linear[me+1+(i%19)]=1e-3;
    inactive.hessian.n=n;
    p.inequalities.push_back(std::move(inactive));
  }
  p.validate();

  Settings st;
  st.verbose=false;
  st.linear_backend="native";
  st.kkt_adaptive_regularization=true;
  st.restoration_max_iterations=12;
  st.restoration_max_restarts=0;
  st.restoration_minimax_enabled=false;
  st.feasibility_tolerance=1e-9;

  const RestorationResult r=FeasibilityRestoration(st).solve(p,p.x0);
  if(!r.success || r.final_feasibility>st.feasibility_tolerance) {
    std::cerr << "[FAIL] sparse active-row restoration did not converge"
              << " theta="<<r.final_feasibility
              << " message="<<r.message<<"\n";
    return 1;
  }
  if(r.row_space_solves<=0 || r.matrix_free_solves!=0 ||
     r.exact_hessian_solves!=0) {
    std::cerr << "[FAIL] near-feasible sparse row-space gate not used"
              << " row_space="<<r.row_space_solves
              << " matrix_free="<<r.matrix_free_solves
              << " exact="<<r.exact_hessian_solves<<"\n";
    return 2;
  }
  if(r.iterations>4) {
    std::cerr << "[FAIL] sparse active-row restoration took "
              <<r.iterations<<" iterations\n";
    return 3;
  }
  std::cout << "[PASS] near-feasible sparse inequalities use compact row-space LM"
            << " iterations="<<r.iterations
            << " solves="<<r.row_space_solves
            << " theta="<<r.final_feasibility<<"\n";
  return 0;
}
