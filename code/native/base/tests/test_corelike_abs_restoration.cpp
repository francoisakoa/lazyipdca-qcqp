#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/settings.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace lazyipdca::qcqp;

static QuadraticFunction make_q(std::string name, int n, double c, Vec lin,
                                std::vector<SymTriplet> H = {}) {
  QuadraticFunction q;
  q.name = std::move(name);
  q.constant = c;
  q.linear = std::move(lin);
  q.hessian.n = n;
  q.hessian.entries = std::move(H);
  return q;
}

int main() {
  try {
    constexpr int m = 20;
    // q[0:m], p[0:m] (m+1 pressures), t[0:m]
    const int q0 = 0;
    const int p0 = m;
    const int t0 = m + (m + 1);
    const int n = 3*m + 1;

    QCQPProblem p;
    p.name = "CORE_LIKE_ABS_REDUCED_PHASE1";
    p.n = n;
    p.x0.assign(n, 0.0);
    p.lb.assign(n, -INFINITY);
    p.ub.assign(n,  INFINITY);
    p.objective = make_q("objective", n, 0.0, Vec(n, 0.0));

    // Exact target flows alternate sign and stay away from zero.  Pressures
    // are recursively chosen so that p_i-p_{i+1}+a_i q_i|q_i| = 0.
    std::vector<double> qstar(m), pstar(m+1), a(m);
    pstar[0] = 4200.0;
    for (int i = 0; i < m; ++i) {
      qstar[i] = (i % 3 == 0 ? -1.0 : 1.0) * (1.0 + 0.08*i);
      a[i] = 0.2 + 0.03*(i % 7);
      pstar[i+1] = pstar[i] + a[i]*qstar[i]*std::abs(qstar[i]);
    }

    for (int i = 0; i < m; ++i) {
      p.lb[q0+i] = -20.0; p.ub[q0+i] = 20.0;
      p.lb[t0+i] = 0.0;   p.ub[t0+i] = 20.0;
    }
    for (int i = 0; i <= m; ++i) {
      p.lb[p0+i] = 1000.0; p.ub[p0+i] = 7000.0;
    }

    // Fix p_0 to remove the pressure translation nullspace.
    p.lb[p0] = pstar[0]; p.ub[p0] = pstar[0];

    // Node-like linear equations pin the relative flow pattern while leaving
    // one scalar degree of freedom: q_i - s_i q_0 = rhs_i.
    for (int i = 1; i < m; ++i) {
      Vec lin(n, 0.0);
      const double ratio = qstar[i] / qstar[0];
      lin[q0+i] = 1.0;
      lin[q0] = -ratio;
      p.equalities.push_back(make_q("node_" + std::to_string(i), n, 0.0, std::move(lin)));
    }

    // One linear equation fixes q_0 at its exact target.
    {
      Vec lin(n, 0.0); lin[q0] = 1.0;
      p.equalities.push_back(make_q("flow_anchor", n, -qstar[0], std::move(lin)));
    }

    // Pipe equations with q_i*t_i plus pressure differences.
    for (int i = 0; i < m; ++i) {
      Vec lin(n, 0.0);
      lin[p0+i] = 1.0;
      lin[p0+i+1] = -1.0;
      std::vector<SymTriplet> H{{q0+i, t0+i, a[i]}};
      p.equalities.push_back(make_q("pipe_" + std::to_string(i), n, 0.0, std::move(lin), std::move(H)));

      // Exact lift t_i^2-q_i^2=0.
      p.equalities.push_back(make_q("abs_link_x" + std::to_string(i), n, 0.0, Vec(n,0.0),
                                     {{q0+i,q0+i,-2.0},{t0+i,t0+i,2.0}}));
    }

    // Deliberately poor, badly scaled starting point with inconsistent t.
    for (int i = 0; i < m; ++i) {
      p.x0[q0+i] = (i % 2 ? -0.25 : 0.25) * (1.0 + i);
      p.x0[t0+i] = 7.0 + 0.2*i;
    }
    for (int i = 0; i <= m; ++i) p.x0[p0+i] = 5200.0 - 35.0*i;
    p.x0[p0] = pstar[0];
    p.validate();

    Settings st;
    st.verbose = false;
    st.feasibility_tolerance = 1e-8;
    st.restoration_max_iterations = 400;
    st.restoration_max_restarts = 6;
    st.restoration_restart_iterations = 200;

    FeasibilityRestoration restoration(st);
    const RestorationResult r = restoration.solve(p, p.x0);
    if (!r.success) {
      throw std::runtime_error("CORE-like reduced abs restoration failed: theta=" + std::to_string(r.final_feasibility));
    }
    if (r.final_feasibility > 1e-8)
      throw std::runtime_error("CORE-like feasibility tolerance failed");
    if (r.structured_abs_links != m)
      throw std::runtime_error("CORE-like structured link count diagnostic failed");
    if (r.effective_dimension != n-m)
      throw std::runtime_error("CORE-like effective dimension diagnostic failed");
    for (int i = 0; i < m; ++i) {
      if (std::abs(r.x[t0+i] - std::abs(r.x[q0+i])) > 1e-11)
        throw std::runtime_error("CORE-like abs manifold consistency failed");
    }

    std::cout << "[PASS] CORE-like reduced abs-manifold Phase-I contract theta="
              << r.final_feasibility << " iterations=" << r.iterations
              << " restarts=" << r.restarts << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
