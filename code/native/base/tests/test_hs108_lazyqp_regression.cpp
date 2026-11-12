#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

QuadraticFunction quadratic(const std::string& name, int n, double constant = 0.0) {
  QuadraticFunction q;
  q.name = name;
  q.constant = constant;
  q.linear.assign(n, 0.0);
  q.hessian.n = n;
  return q;
}

void add_square(QuadraticFunction& q, int i, double scale = 1.0) {
  q.hessian.entries.push_back({i, i, 2.0 * scale});
}

void add_difference_square(QuadraticFunction& q, int i, int j,
                           double scale = 1.0) {
  q.hessian.entries.push_back({i, i, 2.0 * scale});
  q.hessian.entries.push_back({i, j, -2.0 * scale});
  q.hessian.entries.push_back({j, j, 2.0 * scale});
}

QCQPProblem hs108() {
  constexpr int n = 9;
  QCQPProblem p;
  p.name = "HS108_LAZYQP_REGRESSION";
  p.n = n;
  p.x0.assign(n, 1.0);
  p.lb.assign(n, -INFINITY);
  p.ub.assign(n, INFINITY);
  p.lb[8] = 0.0;

  p.objective = quadratic("objective", n);
  p.objective.hessian.entries = {
      {0, 3, -0.5}, {1, 2, 0.5}, {2, 8, -0.5},
      {4, 7, -0.5}, {4, 8, 0.5}, {5, 6, 0.5}};

  auto c1 = quadratic("c1", n, -1.0);
  add_square(c1, 2); add_square(c1, 3);
  p.inequalities.push_back(std::move(c1));

  auto c2 = quadratic("c2", n, -1.0);
  add_square(c2, 4); add_square(c2, 5);
  p.inequalities.push_back(std::move(c2));

  auto c3 = quadratic("c3", n, -1.0);
  add_square(c3, 8);
  p.inequalities.push_back(std::move(c3));

  auto c4 = quadratic("c4", n, -1.0);
  add_square(c4, 0); add_difference_square(c4, 1, 8);
  p.inequalities.push_back(std::move(c4));

  auto c5 = quadratic("c5", n, -1.0);
  add_difference_square(c5, 0, 4); add_difference_square(c5, 1, 5);
  p.inequalities.push_back(std::move(c5));

  auto c6 = quadratic("c6", n, -1.0);
  add_difference_square(c6, 0, 6); add_difference_square(c6, 1, 7);
  p.inequalities.push_back(std::move(c6));

  auto c7 = quadratic("c7", n, -1.0);
  add_difference_square(c7, 2, 4); add_difference_square(c7, 3, 5);
  p.inequalities.push_back(std::move(c7));

  auto c8 = quadratic("c8", n, -1.0);
  add_difference_square(c8, 2, 6); add_difference_square(c8, 3, 7);
  p.inequalities.push_back(std::move(c8));

  auto c9 = quadratic("c9", n, -1.0);
  add_square(c9, 6); add_difference_square(c9, 7, 8);
  p.inequalities.push_back(std::move(c9));

  auto c10 = quadratic("c10", n);
  add_difference_square(c10, 2, 8, -1.0);
  p.inequalities.push_back(std::move(c10));

  auto c11 = quadratic("c11", n);
  c11.hessian.entries = {{4, 7, -1.0}, {5, 6, 1.0}};
  p.inequalities.push_back(std::move(c11));

  auto c12 = quadratic("c12", n);
  c12.hessian.entries = {{0, 3, -1.0}, {1, 2, 1.0}};
  p.inequalities.push_back(std::move(c12));

  auto c13 = quadratic("c13", n);
  c13.hessian.entries = {{4, 8, 1.0}};
  p.inequalities.push_back(std::move(c13));

  p.validate();
  return p;
}

}  // namespace

int main() {
  try {
    Settings st;
    st.verbose = std::getenv("LAZYIPDCA_TEST_VERBOSE") != nullptr;
    st.outer_algorithm = "lazy-dca";
    st.inner_backend = "lazyqp";
    st.linear_backend = "native";
    st.max_outer_iterations = 1000;
    st.max_inner_iterations = 1000;
    st.time_limit_seconds = 30.0;
    st.feasibility_tolerance = 1e-6;
    st.stationarity_tolerance = 1e-6;
    st.dual_infeasibility_tolerance = 1e-6;
    st.complementarity_tolerance = 1e-6;
    st.inner_tolerance = 1e-8;
    st.inner_iteration_limit_mu_fallback = 1e-7;
    st.restoration_max_iterations = 600;
    st.restoration_max_restarts = 16;
    st.dca_inner_retry_max = 3;
    st.dca_inner_retry_initial_prox = 1e-6;
    st.dca_local_corrector = true;
    st.inner_dual_warm_start = true;
    st.inner_inequality_dual_warm_start = true;
    st.kkt_adaptive_regularization = true;
    const Result r = Solver(st).solve(hs108());
    if (r.status != Status::Solved)
      throw std::runtime_error(std::string("HS108 did not converge: ") +
          status_string(r.status) + " / " + r.message);
    if (r.feasibility.total_inf > 1e-6 ||
        r.dual_infeasibility_inf > 1e-6 ||
        std::abs(r.objective + 0.8660254) > 2e-5) {
      std::cerr << "[HS108-DIAGNOSTIC] status=" << status_string(r.status)
                << " objective=" << r.objective
                << " feasibility=" << r.feasibility.total_inf
                << " dual=" << r.dual_infeasibility_inf
                << " desired=" << (r.ipopt_desired_convergence ? 1 : 0)
                << "\n";
      throw std::runtime_error("HS108 final certificate or objective regressed");
    }
    if (r.inner_iteration_limit_fallback_acceptances < 1 ||
        r.lazy_dca_proximal_retries != 0)
      throw std::runtime_error("HS108 did not use the guarded inexact-majorant continuation");
    if (r.working_set_additions < 2 ||
        r.working_set_primal_transports < 2)
      throw std::runtime_error("HS108 did not exercise working-set primal transport");

    std::cout << "[PASS] HS108 LazyQP regression f=" << r.objective
              << " feas=" << r.feasibility.total_inf
              << " dual=" << r.dual_infeasibility_inf
              << " outer=" << r.outer_iterations
              << " inner=" << r.inner_iterations
              << " transports=" << r.working_set_primal_transports << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
