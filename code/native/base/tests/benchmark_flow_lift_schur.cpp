#include "lazyipdca_qcqp/condensed_retraction.hpp"
#include "lazyipdca_qcqp/io.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

struct Timing {
  double factor_seconds = 0.0;
  double solve_seconds = 0.0;
  int factor_dimension = 0;
  int detected_lifts = 0;
  double checksum = 0.0;
};

Timing run(const QCQPProblem& problem, const Settings& settings,
           bool schur, int solves) {
  CondensedQuadraticRetraction projector(settings, schur);
  std::string message;
  const auto factor_start = std::chrono::steady_clock::now();
  if (!projector.factor(problem, problem.x0, message))
    throw std::runtime_error(message);
  const auto factor_stop = std::chrono::steady_clock::now();

  Vec rhs(problem.equalities.size(), 0.0), correction;
  const auto solve_start = std::chrono::steady_clock::now();
  double checksum = 0.0;
  for (int k = 0; k < solves; ++k) {
    for (std::size_t i = 0; i < rhs.size(); ++i)
      rhs[i] = std::sin(0.01 * static_cast<double>((k + 1) * (i + 3)));
    if (!projector.apply_right_inverse(rhs, correction, message))
      throw std::runtime_error(message);
    checksum += correction[static_cast<std::size_t>(k % problem.n)];
  }
  const auto solve_stop = std::chrono::steady_clock::now();
  Timing out;
  out.factor_seconds = std::chrono::duration<double>(
      factor_stop - factor_start).count();
  out.solve_seconds = std::chrono::duration<double>(
      solve_stop - solve_start).count();
  out.factor_dimension = projector.normal_factor_dimension();
  out.detected_lifts = projector.flow_lift_schur_detected_variables();
  out.checksum = checksum;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3) {
      std::cerr << "usage: benchmark_flow_lift_schur MODEL.qcq [SOLVES]\n";
      return 2;
    }
    const int solves = argc == 3 ? std::stoi(argv[2]) : 512;
    Settings settings;
    settings.linear_backend = "native";
    settings.kkt_adaptive_regularization = true;
    settings.kkt_backward_error_tolerance = 1e-9;
    settings.kkt_backward_error_reject = 1e-7;
    settings.kkt_iterative_refinement_steps = 3;
    const QCQPProblem problem = read_qcq(argv[1]);
    const Timing reduced = run(problem, settings, true, solves);
    const Timing full = run(problem, settings, false, solves);
    std::cout << "schur factor_dim=" << reduced.factor_dimension
              << " lifts=" << reduced.detected_lifts
              << " factor_s=" << reduced.factor_seconds
              << " solves_s=" << reduced.solve_seconds << '\n'
              << "full factor_dim=" << full.factor_dimension
              << " factor_s=" << full.factor_seconds
              << " solves_s=" << full.solve_seconds << '\n'
              << "speedup_solve=" << full.solve_seconds /
                     std::max(1e-15, reduced.solve_seconds)
              << " checksum_delta=" << std::abs(full.checksum - reduced.checksum)
              << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << '\n';
    return 1;
  }
}
