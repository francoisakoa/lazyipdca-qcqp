#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

int main() {
  try {
    // Reproduce CATENA/CATENARY's failure mechanism in one dimension.
    // Solving the regularized KKT system with rho_x=1e-12 at y=5e4 leaves
    // an ORIGINAL-model stationarity residual rho_x*y ~= 5e-8.  The solver
    // must remove this numerical regularization bias rather than weakening
    // inner_tol=1e-8.
    QCQPProblem p;
    p.name = "EQUALITY_QP_ORIGINAL_KKT_REFINEMENT";
    p.n = 2;
    p.x0 = {0.0,0.0};
    p.lb = {-std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()};
    p.ub = { std::numeric_limits<double>::infinity(),
             std::numeric_limits<double>::infinity()};
    p.objective.name = "f";
    p.objective.constant = 0.0;
    p.objective.linear = {0.0,0.0};
    p.objective.hessian.n = 2;

    QuadraticFunction h;
    h.name = "h_bias";
    h.constant = -50000.0;
    h.linear = {1.0,0.0};
    h.hessian.n = 2;
    p.equalities.push_back(h);

    // A large, already-satisfied equality RHS reproduces the v0.4.22 bug:
    // its scale inflated a single combined refinement guard and stopped
    // correction while the independent dual block still had a 5e-8 defect.
    QuadraticFunction hscale;
    hscale.name = "h_large_rhs";
    hscale.constant = -1e8;
    hscale.linear = {0.0,1e8};
    hscale.hessian.n = 2;
    p.equalities.push_back(hscale);

    Settings st;
    st.outer_algorithm = "lazy-dca";
    st.verbose = false;
    st.restoration_enabled = false;
    st.kkt_adaptive_regularization = true;
    st.kkt_rho_x_floor = 1e-12;
    st.kkt_primal_regularization = 1e-12;
    st.kkt_delta_dual_floor = 1e-12;
    st.kkt_dual_regularization = 1e-12;
    st.inner_tolerance = 1e-8;
    st.feasibility_tolerance = 1e-6;
    st.stationarity_tolerance = 1e-6;
    st.dual_infeasibility_tolerance = 1e-6;
    st.complementarity_tolerance = 1e-6;
    st.max_outer_iterations = 5;

    const Result r = Solver(st).solve(p);
    if (r.status != Status::Solved)
      throw std::runtime_error("expected solved, got " +
          std::string(status_string(r.status)) + " / " + r.message);
    if (!(r.last_inner_dual_inf <= st.inner_tolerance))
      throw std::runtime_error("original-KKT dual residual was not refined below inner_tol");
    if (!(r.last_inner_equality_inf <= st.inner_tolerance))
      throw std::runtime_error("original-KKT equality residual exceeds inner_tol");
    if (!(std::abs(r.x[0] - 50000.0) <= 1e-5) ||
        !(std::abs(r.x[1] - 1.0) <= 1e-10))
      throw std::runtime_error("wrong equality-QP solution");
    if (r.equality_qp_original_kkt_gmres_attempts != 0)
      throw std::runtime_error("GMRES changed a historically successful refinement path");

    std::cout << "[PASS] original equality-QP KKT refinement removes rho_x bias"
              << " inner_dual=" << r.last_inner_dual_inf
              << " inner_eq=" << r.last_inner_equality_inf
              << " refine=" << r.kkt_refinement_steps << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
