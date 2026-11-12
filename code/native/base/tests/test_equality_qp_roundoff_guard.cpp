#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

int main() {
  try {
    // The equality-only DCA subproblem is d = 1e4.  With rho_x=1e-12,
    // the numerically regularized KKT system leaves a true-model dual
    // residual of about rho_x*|d| = 1e-8, i.e. exactly at inner_tol.
    // The residual evaluation may land a few ulps above 1e-8 and must not
    // turn that boundary case into an InnerFailure.
    QCQPProblem p;
    p.name = "EQUALITY_QP_ROUNDOFF_GUARD";
    p.n = 1;
    p.x0 = {0.0};
    p.lb = {-std::numeric_limits<double>::infinity()};
    p.ub = { std::numeric_limits<double>::infinity()};
    p.objective.name = "f";
    p.objective.constant = 0.0;
    p.objective.linear = {0.0};
    p.objective.hessian.n = 1;

    QuadraticFunction h;
    h.name = "h";
    h.constant = -10000.0;
    h.linear = {1.0};
    h.hessian.n = 1;
    p.equalities.push_back(h);

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
    if (!(r.feasibility.total_inf <= st.feasibility_tolerance))
      throw std::runtime_error("final equality infeasible");
    if (!(std::abs(r.x[0] - 10000.0) <= 1e-5))
      throw std::runtime_error("wrong equality solution");

    std::cout << "[PASS] equality-QP roundoff guard accepts tol+O(eps) boundary"
              << " inner_dual=" << r.last_inner_dual_inf
              << " inner_eq=" << r.last_inner_equality_inf << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
