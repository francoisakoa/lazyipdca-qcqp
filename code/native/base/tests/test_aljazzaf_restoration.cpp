#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/settings.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

int main() {
  try {
    constexpr int n = 1000;
    constexpr int n1 = 500;
    constexpr double biga = 100.0;
    const double f = (biga*biga - 1.0) / (n - 1.0);

    QCQPProblem p;
    p.name = "ALJAZZAF_PHASE1_EXACT";
    p.n = n;
    p.x0.assign(n, 0.0);
    p.lb.assign(n, -INFINITY);
    p.ub.assign(n, INFINITY);
    p.objective.name = "objective";
    p.objective.linear.assign(n, 0.0);
    p.objective.hessian.n = n;

    QuadraticFunction h;
    h.name = "H";
    h.constant = 1.0;
    h.linear.assign(n, 0.0);
    h.linear[0] = -1.0;
    h.hessian.n = n;

    // Exact CUTEst/SIF equality:
    // 1 - x1 + sum_{i=2..500} B_i x_i^2
    //       + sum_{i=501..1000} B_i (x_i-1)^2 = 0,
    // B_i = 1 + (i-1)*(100^2-1)/(1000-1).
    for (int j = 1; j < n; ++j) {
      const double B = 1.0 + j*f;
      h.hessian.entries.push_back({j,j,2.0*B});
      if (j >= n1) {
        h.linear[j] = -2.0*B;
        h.constant += B;
      }
    }
    p.equalities.push_back(std::move(h));
    p.validate();

    const double theta0 = feasibility(p, p.x0).total_inf;
    if (std::abs(theta0 - 3751377.1261261264) > 1e-6)
      throw std::runtime_error("ALJAZZAF reconstruction mismatch");

    Settings st;
    st.verbose = false;
    st.feasibility_tolerance = 1e-7;
    st.restoration_max_iterations = 100;
    st.restoration_max_restarts = 0;
    st.restoration_exact_hessian_threshold = 256; // force matrix-free at n=1000

    FeasibilityRestoration restoration(st);
    const RestorationResult r = restoration.solve(p, p.x0);
    if (!r.success)
      throw std::runtime_error("ALJAZZAF restoration failed theta=" + std::to_string(r.final_feasibility));
    if (r.final_feasibility > 1e-7)
      throw std::runtime_error("ALJAZZAF feasibility regression");
    if (r.iterations > 40)
      throw std::runtime_error("ALJAZZAF restoration unexpectedly slow");
    if (r.matrix_free_solves <= 0 || r.cg_iterations <= 0)
      throw std::runtime_error("ALJAZZAF matrix-free exact-Hessian path not exercised");

    std::cout << "[PASS] ALJAZZAF exact-QCQP Phase-I theta=" << r.final_feasibility
              << " iterations=" << r.iterations
              << " mf_solves=" << r.matrix_free_solves
              << " cg=" << r.cg_iterations << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
