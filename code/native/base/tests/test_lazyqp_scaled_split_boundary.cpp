#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static ConvexQuadraticModel linear_row(double constant, double a0, double a1) {
  ConvexQuadraticModel q;
  q.constant=constant;
  q.linear={a0,a1};
  q.base_hessian.n=2;
  return q;
}

static ConvexSubproblem make_boundary_release_problem() {
  ConvexSubproblem p;
  p.nvar=2;
  p.original_dimension=2;
  p.prefer_feasible_centered_start=true;

  // Minimize -y_1 over a strongly curved convex boundary.  Several nearly
  // coincident active rows and many inactive barrier rows reproduce the
  // degeneracy that makes one primal slack limit alpha_P while the dual
  // fraction-to-boundary step remains close to one.
  p.objective.constant=0.0;
  p.objective.linear={0.0,-1.0};
  p.objective.base_hessian.n=2;

  ConvexQuadraticModel curved;
  curved.constant=0.0;
  curved.linear={1.0,0.0};
  curved.base_hessian.n=2;
  curved.base_hessian.entries={{1,1,1.0e4}}; // y_0 + 5000 y_1^2 <= 0
  p.inequalities.push_back(curved);

  p.inequalities.push_back(linear_row(-1.0,-1.0,0.0)); // y_0 >= -1
  p.inequalities.push_back(linear_row( 0.0, 0.0,-1.0)); // y_1 >= 0
  p.inequalities.push_back(linear_row(-1.0, 0.0, 1.0)); // y_1 <= 1

  for(int i=0;i<4;++i) {
    ConvexQuadraticModel duplicate=curved;
    duplicate.constant=-(i+1)*1.0e-12;
    p.inequalities.push_back(std::move(duplicate));
  }
  for(int i=0;i<100;++i)
    p.inequalities.push_back(linear_row(-1.0e-4,0.0,0.0));

  p.equalities.nvar=2;
  return p;
}

int main() {
  try {
    Settings st;
    st.verbose=false;
    st.max_inner_iterations=300;
    st.inner_tolerance=1e-7;
    st.inner_iteration_limit_mu_fallback=1e-6;

    const InnerResult r=ConvexQCQPLazyQPKernel(st).solve(make_boundary_release_problem());
    if(!r.success)
      throw std::runtime_error("scaled split-step LazyQP stalled: "+r.message);
    if(r.y.size()!=2)
      throw std::runtime_error("boundary-release regression returned wrong dimension");

    const double expected_y1=std::sqrt(2.0e-4);
    if(std::abs(r.y[0]+1.0)>2e-5 || std::abs(r.y[1]-expected_y1)>2e-6)
      throw std::runtime_error("boundary-release regression converged to the wrong minimizer");
    if(r.dual_inf>1e-5 || r.primal_residual_inf>1e-5 ||
       r.inequality_inf>1e-5 || r.complementarity>2e-6)
      throw std::runtime_error("boundary-release regression KKT residual is too large");
    if(r.lazyqp_scaled_split_steps<1)
      throw std::runtime_error("scaled split-step path was not exercised");
    if(std::max(r.lazyqp_max_primal_dual_step_ratio,
                r.lazyqp_max_dual_primal_step_ratio)<=10.0)
      throw std::runtime_error("fraction-to-boundary scales never separated");
    if(r.iterations>=80)
      throw std::runtime_error("primal boundary again throttled the dual Newton step");

    std::cout << "[PASS] LazyQP scaled split boundary release it="<<r.iterations
              << " split="<<r.lazyqp_scaled_split_steps
              << " max_p_over_d="<<r.lazyqp_max_primal_dual_step_ratio
              << " max_d_over_p="<<r.lazyqp_max_dual_primal_step_ratio
              << " dual="<<r.dual_inf
              << " pri="<<r.primal_residual_inf
              << " mu="<<r.complementarity
              << " y1="<<r.y[1]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
