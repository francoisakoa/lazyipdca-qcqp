#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>
#include <string>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p = read_qcq(std::string(LAZYIPDCA_SOURCE_DIR) + "/tests/toy.qcq");
  Settings st;
  st.verbose=false;
  st.linear_backend="native";
  st.kkt_adaptive_regularization=true;
  st.kkt_primal_regularization=1e-12;
  st.kkt_dual_regularization=1e-12;
  st.kkt_rho_x_floor=1e-12;
  st.kkt_rho_x_growth=10.0;
  st.kkt_rho_x_decay=0.5;
  st.kkt_rho_x_max=1e-2;
  st.kkt_rho_x_clean_outer=3;
  st.kkt_delta_dual_base=1e-8;
  st.kkt_delta_dual_floor=1e-12;
  st.kkt_delta_dual_max=1e-4;
  st.kkt_delta_dual_inverse=true;
  st.kkt_backward_error_tolerance=1e-9;
  st.kkt_backward_error_reject=1e-7;
  st.kkt_iterative_refinement_steps=3;
  st.max_outer_iterations=100;
  st.max_inner_iterations=500;

  const Result r=Solver(st).solve(p);
  if(r.status!=Status::Solved){
    std::cerr << "native adaptive KKT failed: " << r.message << "\n";
    return 1;
  }
  if(!(r.final_kkt_rho_x>=st.kkt_rho_x_floor) || !(r.max_kkt_rho_x>=r.final_kkt_rho_x)){
    std::cerr << "rho_x diagnostics not wired\n";
    return 2;
  }
  if(!(r.final_kkt_delta_d>0.0) || !(r.max_kkt_delta_d>=r.final_kkt_delta_d)){
    std::cerr << "delta_d diagnostics not wired\n";
    return 3;
  }
  if(!std::isfinite(r.max_kkt_backward_error) || r.max_kkt_backward_error>st.kkt_backward_error_reject){
    std::cerr << "native backward-error diagnostic invalid: " << r.max_kkt_backward_error << "\n";
    return 4;
  }
  if(r.final_proximal_regularization>1e-4){
    std::cerr << "toy unexpectedly inflated SQP prox: " << r.final_proximal_regularization << "\n";
    return 5;
  }
  std::cout << "[PASS] native common KKT rho_x=" << r.final_kkt_rho_x
            << " delta_d=" << r.final_kkt_delta_d
            << " berr=" << r.max_kkt_backward_error
            << " prox=" << r.final_proximal_regularization << "\n";
  return 0;
}
