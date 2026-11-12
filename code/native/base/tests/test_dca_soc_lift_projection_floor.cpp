#include "lazyipdca_qcqp/restoration.hpp"

#include <cmath>
#include <iostream>

using namespace lazyipdca::qcqp;

int main() {
  // CORE-like lifted state.  The lift mismatch is tiny, but a strongly scaled
  // coupling equality amplifies an exact t=|q| projection into an O(1e-4)
  // correction.  A defect-only SQP SOC with zero defect must return zero.
  QCQPProblem p;
  p.name = "dca_soc_lift_projection_floor";
  p.n = 3; // q, t, y
  p.lb = {-2.0, 0.0, -2.0};
  p.ub = { 2.0, 2.0,  2.0};
  p.x0 = {1.0, 1.0 + 5.0e-13, -4.9901e-4};
  p.objective.linear.assign(3, 0.0);
  p.objective.hessian.n = 3;

  QuadraticFunction link;
  link.name = "abs_link_0";
  link.linear.assign(3, 0.0);
  link.hessian.n = 3;
  link.hessian.entries = {{0,0,-2.0},{1,1,2.0}};
  p.equalities.push_back(link);

  QuadraticFunction coupling;
  coupling.name = "scaled_coupling";
  coupling.linear = {-1.0e9, 1.0e9, 1.0};
  coupling.hessian.n = 3;
  p.equalities.push_back(coupling);

  Vec targets;
  for (const auto& h : p.equalities) targets.push_back(h.value(p.x0));

  Vec reduced = retract_equalities_to_targets(
      p, p.x0, targets, 12, 1e-14, 1e-12, 0.995);
  Vec full = retract_equalities_to_targets_full_space(
      p, p.x0, targets, 12, 1e-14, 1e-12, 0.995);

  Vec dr = reduced;
  Vec df = full;
  axpy(-1.0, p.x0, dr);
  axpy(-1.0, p.x0, df);
  const double reduced_inf = inf_norm(dr);
  const double full_inf = inf_norm(df);

  double target_error = 0.0;
  for (int i = 0; i < static_cast<int>(p.equalities.size()); ++i)
    target_error = std::max(target_error,
        std::abs(p.equalities[i].value(full) - targets[i]));

  if (!(reduced_inf > 1e-5)) {
    std::cerr << "[FAIL] regression did not expose projection floor: "
              << reduced_inf << "\n";
    return 1;
  }
  if (!(full_inf < 1e-12 && target_error < 1e-12)) {
    std::cerr << "[FAIL] full-space zero-defect correction was not continuous: "
              << "corr=" << full_inf << " target=" << target_error << "\n";
    return 2;
  }

  std::cout << "[PASS] full-space defect SOC removes lift-projection floor"
            << " reduced_corr=" << reduced_inf
            << " full_corr=" << full_inf
            << " target_error=" << target_error << "\n";
  return 0;
}
