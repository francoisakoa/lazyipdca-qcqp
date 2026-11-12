#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

int main() {
  try {
    QCQPProblem p;
    p.name="EQUALITY_QP_DUAL_SHIFT";
    p.n=1;
    p.x0={1.0};
    p.lb={-std::numeric_limits<double>::infinity()};
    p.ub={ std::numeric_limits<double>::infinity()};
    p.objective.name="f"; p.objective.constant=0.0;
    p.objective.linear={1000.0}; p.objective.hessian.n=1;
    p.objective.hessian.entries={{0,0,1.0}};
    QuadraticFunction h;
    h.name="h"; h.constant=0.0; h.linear={1.0}; h.hessian.n=1;
    p.equalities.push_back(h);

    Settings st;
    st.outer_algorithm="lazy-dca";
    st.verbose=false;
    st.restoration_enabled=false;
    st.kkt_adaptive_regularization=true;
    st.kkt_delta_dual_base=1e-8;   // historical value that must not seed this path
    st.kkt_delta_dual_floor=1e-12;
    st.kkt_dual_regularization=1e-12;
    st.inner_tolerance=1e-8;
    st.feasibility_tolerance=1e-7;
    st.stationarity_tolerance=1e-6;
    st.dual_infeasibility_tolerance=1e-6;
    st.complementarity_tolerance=1e-6;
    st.max_outer_iterations=10;

    Result r=Solver(st).solve(p);
    if(r.status!=Status::Solved)
      throw std::runtime_error("expected solved, got "+std::string(status_string(r.status))+" / "+r.message);
    if(!(r.last_inner_equality_inf<=st.inner_tolerance))
      throw std::runtime_error("original equality residual exceeds inner tolerance");
    if(!(r.final_kkt_delta_d<=1.01e-12))
      throw std::runtime_error("equality-QP incorrectly started from barrier dual shift");
    if(std::abs(r.x[0])>st.feasibility_tolerance)
      throw std::runtime_error("wrong equality solution");

    std::cout << "[PASS] equality-QP uses delta floor residual="
              << r.last_inner_equality_inf << " delta=" << r.final_kkt_delta_d << "\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
