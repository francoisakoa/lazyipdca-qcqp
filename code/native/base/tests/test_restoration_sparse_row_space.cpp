#include "lazyipdca_qcqp/restoration.hpp"
#include <cmath>
#include <iostream>
#include <limits>

using namespace lazyipdca::qcqp;

int main() {
  constexpr int m=200;
  QCQPProblem p;
  p.name="catenary_like_sparse_row_space";
  p.n=3*m;
  p.x0.assign(p.n,0.0);
  p.lb.assign(p.n,-std::numeric_limits<double>::infinity());
  p.ub.assign(p.n, std::numeric_limits<double>::infinity());
  p.objective.name="f";
  p.objective.constant=0.0;
  p.objective.linear.assign(p.n,0.0);
  p.objective.hessian.n=p.n;
  for(int i=0;i<m;++i) {
    QuadraticFunction h;
    h.name="chain_"+std::to_string(i);
    h.constant=-1.0;
    h.linear.assign(p.n,0.0);
    h.linear[3*i]=1.0;
    h.linear[3*i+1]=2.0;
    if(i+1<m) h.linear[3*(i+1)]=-0.5;
    h.hessian.n=p.n;
    p.equalities.push_back(std::move(h));
  }

  Settings st;
  st.verbose=false;
  st.linear_backend="native";
  st.kkt_adaptive_regularization=true;
  st.restoration_max_iterations=20;
  st.restoration_max_restarts=0;
  st.restoration_minimax_enabled=false;
  st.feasibility_tolerance=1e-9;
  const RestorationResult r=FeasibilityRestoration(st).solve(p,p.x0);
  if(!r.success) {
    std::cerr << "[FAIL] row-space restoration failed: " << r.message
              << " theta=" << r.final_feasibility << "\n"; return 1;
  }
  if(r.row_space_solves<=0) {
    std::cerr << "[FAIL] sparse row-space path not used\n"; return 2;
  }
  if(r.final_feasibility>st.feasibility_tolerance) {
    std::cerr << "[FAIL] final feasibility=" << r.final_feasibility << "\n"; return 3;
  }
  std::cout << "[PASS] large equality-only Phase-I uses row-space LM"
            << " n=" << p.n << " m=" << m
            << " solves=" << r.row_space_solves
            << " theta=" << r.final_feasibility << "\n";
  return 0;
}
