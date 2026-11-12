#include "lazyipdca_qcqp/inner_ipm.hpp"
#include <iostream>

using namespace lazyipdca::qcqp;

static ConvexSubproblem make_problem() {
  ConvexSubproblem p;
  p.nvar = 1;
  p.original_dimension = 1;
  p.objective.linear = {0.0};
  p.objective.base_hessian.n = 1;
  p.objective.shift_dimension = 1;
  p.objective.scalar_shift = 1.0;

  // A deliberately inactive inequality with a large slack.  Its dual residual
  // converges much faster than absolute complementarity s*z, which makes it a
  // deterministic contract for the guarded max-iteration fallback.
  ConvexQuadraticModel g;
  g.constant = -1000.0;
  g.linear = {1.0};
  g.base_hessian.n = 1;
  p.inequalities.push_back(g);
  return p;
}

int main() {
  const auto p = make_problem();

  Settings guarded;
  guarded.max_inner_iterations = 5;
  guarded.inner_tolerance = 1e-7;
  guarded.inner_iteration_limit_mu_fallback = 1e-5;
  ConvexQCQPInteriorPoint ipm_guarded(guarded);
  const InnerResult a = ipm_guarded.solve(p);
  if (!a.success || !a.used_iteration_limit_fallback) {
    std::cerr << "guarded fallback was not used at the inner iteration limit\n";
    return 1;
  }
  if (!(a.dual_inf <= guarded.inner_tolerance &&
        a.primal_residual_inf <= guarded.inner_tolerance &&
        a.complementarity > guarded.inner_tolerance &&
        a.complementarity <= guarded.inner_iteration_limit_mu_fallback)) {
    std::cerr << "guarded fallback accepted outside its contract\n";
    return 2;
  }

  Settings blocked = guarded;
  blocked.inner_iteration_limit_mu_fallback = 1e-8;
  ConvexQCQPInteriorPoint ipm_blocked(blocked);
  const InnerResult b = ipm_blocked.solve(p);
  if (b.success || b.used_iteration_limit_fallback) {
    std::cerr << "fallback ignored its complementarity guard\n";
    return 3;
  }

  Settings strict = guarded;
  strict.max_inner_iterations = 6;
  ConvexQCQPInteriorPoint ipm_strict(strict);
  const InnerResult c = ipm_strict.solve(p);
  if (!c.success || c.used_iteration_limit_fallback ||
      c.complementarity > strict.inner_tolerance) {
    std::cerr << "normal strict convergence was changed by fallback support\n";
    return 4;
  }

  std::cout << "[PASS] strict IPM preserved; guarded complementarity fallback is max-iteration only\n";
  return 0;
}
