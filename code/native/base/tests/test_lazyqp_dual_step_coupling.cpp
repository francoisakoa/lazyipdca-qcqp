#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static ConvexSubproblem make_dual_jump_problem() {
  ConvexSubproblem p;
  p.nvar=2;
  p.original_dimension=2;
  p.prefer_feasible_centered_start=true;

  // A nearly flat objective direction creates a very large primal Newton
  // component.  Its dual component must be damped whenever that primal
  // component is backtracked by the exact nonlinear residual test.
  p.objective.constant=0.0;
  p.objective.linear={-0.9,-0.0945696};
  p.objective.base_hessian.n=2;
  p.objective.base_hessian.entries={{1,1,1e-16}};

  // x + 0.5e-14 y^2 <= 0 and x >= -1.  The optimum is
  // x=-1, y=sqrt(2e14).  The problem is convex and strictly feasible.
  ConvexQuadraticModel curved;
  curved.constant=0.0;
  curved.linear={1.0,0.0};
  curved.base_hessian.n=2;
  curved.base_hessian.entries={{1,1,1e-14}};
  p.inequalities.push_back(curved);

  ConvexQuadraticModel x_lower;
  x_lower.constant=-1.0;
  x_lower.linear={-1.0,0.0};
  x_lower.base_hessian.n=2;
  p.inequalities.push_back(x_lower);

  // Reproduce the many-row barrier scaling of the failure without adding
  // curvature or changing the minimizer.  These rows are strictly inactive.
  for(int i=0;i<100;++i) {
    ConvexQuadraticModel redundant;
    redundant.constant=-1e-4;
    redundant.linear={0.0,0.0};
    redundant.base_hessian.n=2;
    p.inequalities.push_back(redundant);
  }

  p.equalities.nvar=2;
  return p;
}

int main() {
  try {
    Settings st;
    st.verbose=false;
    st.max_inner_iterations=100;
    st.inner_tolerance=1e-7;
    st.inner_iteration_limit_mu_fallback=1e-6;

    const InnerResult r=ConvexQCQPLazyQPKernel(st).solve(make_dual_jump_problem());
    if(!r.success)
      throw std::runtime_error("coupled-step LazyQP failed: "+r.message);
    if(r.message.find("nonlinear-residual")==std::string::npos)
      throw std::runtime_error("dual-step regression did not use nonlinear-residual LazyQP");
    if(r.y.size()!=2 || !std::isfinite(r.y[0]) || !std::isfinite(r.y[1]))
      throw std::runtime_error("dual-step regression returned invalid primal variables");

    const double expected_y=std::sqrt(2.0e14);
    if(std::abs(r.y[0]+1.0)>1e-5 ||
       std::abs(r.y[1]-expected_y)>2e-5*expected_y)
      throw std::runtime_error("dual-step regression converged to the wrong minimizer");
    if(r.dual_inf>5e-6 || r.primal_residual_inf>5e-6 ||
       r.inequality_inf>5e-6 || r.complementarity>2e-6)
      throw std::runtime_error("dual-step regression KKT residual is too large");
    if(r.lazyqp_scaled_split_steps<1 ||
       r.lazyqp_max_dual_primal_step_ratio<=100.0)
      throw std::runtime_error("scaled split-step protection was not exercised");
    if(r.iterations>=60)
      throw std::runtime_error("dual-step regression lost the scaled split-step escape");

    std::cout << "[PASS] LazyQP coupled-step regression it="<<r.iterations
              << " split="<<r.lazyqp_scaled_split_steps
              << " max_d_over_p="<<r.lazyqp_max_dual_primal_step_ratio
              << " dual="<<r.dual_inf
              << " pri="<<r.primal_residual_inf
              << " mu="<<r.complementarity
              << " y="<<r.y[1]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
