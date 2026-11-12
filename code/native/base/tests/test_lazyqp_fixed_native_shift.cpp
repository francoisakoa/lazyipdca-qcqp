#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

static ConvexSubproblem make_flat_problem() {
  ConvexSubproblem p;
  p.nvar=1;
  p.original_dimension=1;
  p.prefer_feasible_centered_start=true;

  // With no objective Hessian and no constraint Jacobian, the primal KKT row
  // is identically zero unless the Settings-level primal shift is assembled.
  p.objective.constant=0.0;
  p.objective.linear={-1.0};
  p.objective.base_hessian.n=1;

  ConvexQuadraticModel inactive;
  inactive.constant=-1.0;
  inactive.linear={0.0};
  inactive.base_hessian.n=1;
  p.inequalities.push_back(inactive);

  p.equalities.nvar=1;
  return p;
}

int main() {
  try {
    Settings no_shift;
    no_shift.verbose=false;
    no_shift.linear_backend="native";
    no_shift.kkt_adaptive_regularization=false;
    no_shift.kkt_primal_regularization=0.0;
    no_shift.kkt_dual_regularization=0.0;
    no_shift.max_inner_iterations=1;

    const InnerResult singular=ConvexQCQPLazyQPKernel(no_shift).solve(make_flat_problem());
    if(singular.message.find("KKT factorization failed")==std::string::npos)
      throw std::runtime_error("unshifted flat KKT did not expose the singular native system");

    Settings shifted=no_shift;
    shifted.kkt_primal_regularization=1e-6;
    shifted.kkt_dual_regularization=1e-8;
    const InnerResult regularized=ConvexQCQPLazyQPKernel(shifted).solve(make_flat_problem());

    if(regularized.message.find("KKT factorization failed")!=std::string::npos)
      throw std::runtime_error("fixed native KKT shifts were not assembled");
    if(regularized.factorizations<1)
      throw std::runtime_error("regularized native KKT was not factored");
    if(std::abs(regularized.final_kkt_rho_x-shifted.kkt_primal_regularization)>1e-18)
      throw std::runtime_error("reported rho_x does not match the assembled fixed shift");
    if(std::abs(regularized.final_kkt_delta_d-shifted.kkt_dual_regularization)>1e-18)
      throw std::runtime_error("reported delta_d does not match the assembled fixed shift");

    std::cout << "[PASS] LazyQP native fixed KKT shift rho_x="
              << regularized.final_kkt_rho_x
              << " delta_d=" << regularized.final_kkt_delta_d
              << " fact=" << regularized.factorizations << "\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
