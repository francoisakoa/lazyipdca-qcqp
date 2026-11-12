#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static ConvexSubproblem make_curvature_wall_problem() {
  ConvexSubproblem p;
  p.nvar=2;
  p.original_dimension=2;
  p.prefer_feasible_centered_start=true;

  // Mildly regularized affine objective: drive y_1 upward.
  p.objective.constant=0.0;
  p.objective.linear={0.0,-1.0};
  p.objective.base_hessian.n=2;
  p.objective.base_hessian.entries={{0,0,1e-4},{1,1,1e-4}};

  // Strongly curved row active at the centered start:
  //   y_0 + 0.5e6 y_1^2 <= 0.
  // A tangential Newton displacement in y_1 has a large O(alpha^2)
  // remainder.  v0.4.4 forced that remainder into the slack and stalled.
  ConvexQuadraticModel curved;
  curved.constant=0.0;
  curved.linear={1.0,0.0};
  curved.base_hessian.n=2;
  curved.base_hessian.entries={{1,1,1e6}};

  ConvexQuadraticModel xlo;
  xlo.constant=-1.0;
  xlo.linear={-1.0,0.0};
  xlo.base_hessian.n=2; // y_0 >= -1

  ConvexQuadraticModel ylo;
  ylo.constant=0.0;
  ylo.linear={0.0,-1.0};
  ylo.base_hessian.n=2; // y_1 >= 0

  ConvexQuadraticModel yhi;
  yhi.constant=-1.0;
  yhi.linear={0.0,1.0};
  yhi.base_hessian.n=2; // y_1 <= 1

  p.inequalities={curved,xlo,ylo,yhi};
  p.equalities.nvar=2;
  return p;
}

int main() {
  try {
    Settings st;
    st.verbose=false;
    st.max_inner_iterations=400;
    st.inner_tolerance=1e-8;
    st.inner_iteration_limit_mu_fallback=1e-6;

    ConvexQCQPLazyQPKernel qp(st);
    const InnerResult r=qp.solve(make_curvature_wall_problem());
    if(!r.success)
      throw std::runtime_error("nonlinear-residual LazyQP stalled on curvature-wall regression: "+r.message);
    if(r.message.find("nonlinear-residual")==std::string::npos)
      throw std::runtime_error("curvature-wall regression did not use nonlinear-residual LazyQP path");
    if(r.y.size()!=2)
      throw std::runtime_error("curvature-wall regression returned wrong dimension");

    const double ystar=std::sqrt(2.0e-6);
    if(std::abs(r.y[0]+1.0)>2e-5 || std::abs(r.y[1]-ystar)>2e-6)
      throw std::runtime_error("curvature-wall regression converged to wrong point");
    if(r.dual_inf>1e-5 || r.primal_residual_inf>1e-5 || r.inequality_inf>1e-5)
      throw std::runtime_error("curvature-wall regression residual too large");
    if(r.iterations>=100)
      throw std::runtime_error("curvature-wall regression lost the intended fast escape from the exact-slack wall");

    std::cout << "[PASS] LazyQP curvature-wall regression it="<<r.iterations
              << " dual="<<r.dual_inf
              << " pri="<<r.primal_residual_inf
              << " y="<<r.y[1]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
