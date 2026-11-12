#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static ConvexSubproblem make_problem() {
  ConvexSubproblem p;
  p.nvar=2;
  p.original_dimension=2;
  p.prefer_feasible_centered_start=true;

  p.objective.linear={-1.0,-0.2};
  p.objective.base_hessian.n=2;
  p.objective.base_hessian.entries={{0,0,0.2},{1,1,0.2}};
  p.objective.low_rank.push_back({1.0,{1.0,1.0}});

  ConvexQuadraticModel curved;
  curved.constant=-1.0;
  curved.linear={0.0,0.0};
  curved.base_hessian.n=2;
  curved.low_rank.push_back({2.0,{1.0,0.0}}); // x_0^2 <= 1
  p.inequalities.push_back(curved);

  ConvexQuadraticModel lower;
  lower.constant=-1.0;
  lower.linear={0.0,-1.0};
  lower.base_hessian.n=2;
  p.inequalities.push_back(lower);

  ConvexQuadraticModel upper;
  upper.constant=-1.0;
  upper.linear={0.0,1.0};
  upper.base_hessian.n=2;
  p.inequalities.push_back(upper);

  p.equalities.nvar=2;
  return p;
}

int main() {
  try {
    Settings condensed;
    condensed.verbose=false;
    condensed.max_inner_iterations=200;
    condensed.inner_tolerance=1e-8;
    condensed.inner_iteration_limit_mu_fallback=1e-6;
    // Deliberately choose the opposite curvature-oracle threshold: the KKT
    // representation must now be controlled only by lazyqp_low_rank_form.
    condensed.exact_eigen_threshold=0;
    condensed.lazyqp_low_rank_form="condensed";
    condensed.lazyqp_kkt_form="full";

    Settings bordered=condensed;
    bordered.exact_eigen_threshold=256;
    bordered.lazyqp_low_rank_form="bordered";

    const InnerResult rc=ConvexQCQPLazyQPKernel(condensed).solve(make_problem());
    const InnerResult rb=ConvexQCQPLazyQPKernel(bordered).solve(make_problem());
    if(!rc.success || !rb.success)
      throw std::runtime_error("condensed/bordered low-rank solve failed");
    if(rc.y.size()!=2 || rb.y.size()!=2)
      throw std::runtime_error("condensed/bordered low-rank dimension mismatch");
    if(std::abs(rc.y[0]-rb.y[0])>1e-7 || std::abs(rc.y[1]-rb.y[1])>1e-7)
      throw std::runtime_error("condensed and bordered low-rank KKT solutions differ");
    if(std::abs(rc.y[0]-1.0)>5e-5 || std::abs(rc.y[1]+2.0/3.0)>5e-5)
      throw std::runtime_error("low-rank condensation returned wrong minimizer");
    if(rc.lazyqp_full_kkt_dimension!=5 || rb.lazyqp_full_kkt_dimension!=7)
      throw std::runtime_error("low-rank form did not control condensed/bordered KKT dimensions independently");

    std::cout << "[PASS] exact low-rank condensation matches bordered KKT"
              << " y="<<rc.y[0]<<","<<rc.y[1]
              << " dim_condensed="<<rc.lazyqp_full_kkt_dimension
              << " dim_bordered="<<rb.lazyqp_full_kkt_dimension
              << " it="<<rc.iterations<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
