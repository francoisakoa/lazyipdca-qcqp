#include "lazyipdca_qcqp/dca_filter.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void independent_certificate(const Result &r) {
  require(r.status == Status::Solved, r.message.c_str());
  require(r.x.size() == 3 && r.certificate_inequality_multipliers.size() == 1,
          "missing original coefficient certificate");
  const double diagonal[3] = {-1.2, -1.0, 0.0};
  const double z = r.certificate_inequality_multipliers[0];
  require(std::isfinite(z) && z >= 0, "invalid inequality dual sign");
  double g = -1, objective = 1.5 * r.x[2], stationarity = 0;
  for (int j = 0; j < 3; ++j) {
    require(std::isfinite(r.x[j]) && r.x[j] >= -2 - 1e-6 && r.x[j] <= 2 + 1e-6,
            "nonfinite or infeasible original point");
    g += r.x[j] * r.x[j];
    objective += 0.5 * diagonal[j] * r.x[j] * r.x[j];
    double gradient = (diagonal[j] + 2 * z) * r.x[j] + (j == 2 ? 1.5 : 0.0);
    if (r.certificate_lower_multipliers.size() == 3) gradient -= r.certificate_lower_multipliers[j];
    if (r.certificate_upper_multipliers.size() == 3) gradient += r.certificate_upper_multipliers[j];
    stationarity = std::max(stationarity, std::abs(gradient));
  }
  require(g <= 1e-6 && std::abs(z * g) <= 1e-6 && stationarity <= 1e-6,
          "independent original KKT check rejected");
  require(std::abs(objective + 1.5) <= 1e-5, "wrong original objective");
}
}
int main() {
  try {
    QCQPProblem p;
    // Genuine DC objective, convex sphere constraint. At (0,0,-1), z=0.75,
    // the tangent Lagrangian Hessian is diag(0.3,0.5), whereas the objective
    // alone has two negative modes. Its DC corrections produce local modes
    // approximately 0.8 and 2/3. Unlike a concave sphere constraint, this
    // example does not require combining positive O(error^2) DC slack gaps.
    p.name = "g6_two_slow_dc_objective_modes";
    p.n = 3; p.x0 = {0.1, 0.1, -std::sqrt(0.98)};
    p.lb = {-2, -2, -2}; p.ub = {2, 2, 2};
    p.objective.linear = {0, 0, 1.5}; p.objective.hessian.n = 3;
    p.objective.hessian.entries = {{0, 0, -1.2}, {1, 1, -1.0}};
    QuadraticFunction inequality;
    inequality.constant = -1; inequality.linear = {0, 0, 0}; inequality.hessian.n = 3;
    inequality.hessian.entries = {{0, 0, 2}, {1, 1, 2}, {2, 2, 2}};
    p.inequalities = {inequality};
    Settings s; s.linear_backend = "native"; s.dca_engine = "filter";
    require(!s.dca_filter_anderson, "Anderson acceleration must remain opt-in");
    s.max_outer_iterations = 1000; s.lazy_newton_activation_margin = 1;
    s.dca_filter_extrapolation = false; // Isolate the new multi-mode proposal.
    s.dca_filter_model_reuse = false;
    s.feasibility_tolerance = s.stationarity_tolerance =
        s.dual_infeasibility_tolerance = s.complementarity_tolerance = 1e-6;
    const auto baseline = solve_filter_dca(p, s);
    independent_certificate(baseline);
    s.dca_filter_anderson = true;
    const auto accelerated = solve_filter_dca(p, s);
    independent_certificate(accelerated);
    std::cout << "Anderson probe: baseline=" << baseline.outer_iterations
              << " accelerated=" << accelerated.outer_iterations
              << " trials=" << accelerated.dca_filter.extrapolation_trials
              << " acceptances=" << accelerated.dca_filter.extrapolation_acceptances << '\n';
    require(accelerated.dca_filter.extrapolation_acceptances > 0,
            "bounded multi-mode acceleration not exercised");
    require(accelerated.outer_iterations < baseline.outer_iterations,
            "multi-mode proposal did not reduce DC iterations");
    require(accelerated.dca_filter.restorations == 0, "unexpected restoration");
    std::cout << "PASS original nonconvex objective KKT; DC iterations "
              << baseline.outer_iterations << " -> " << accelerated.outer_iterations
              << ", accepted Anderson proposals "
              << accelerated.dca_filter.extrapolation_acceptances << '\n';
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
