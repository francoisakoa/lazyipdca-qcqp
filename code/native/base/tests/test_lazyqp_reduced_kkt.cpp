#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static ConvexQuadraticModel linear_row(double constant, double a0, double a1) {
  ConvexQuadraticModel q;
  q.constant=constant;
  q.linear={a0,a1};
  q.base_hessian.n=2;
  return q;
}

static ConvexSubproblem make_problem() {
  ConvexSubproblem p;
  p.nvar=2;
  p.original_dimension=2;
  p.prefer_feasible_centered_start=true;
  p.objective.linear={0.0,-1.0};
  p.objective.base_hessian.n=2;
  // Keep one PSD term as an explicit KKT border.  The inequality-reduced
  // representation must eliminate only z and retain this column/diagonal
  // exactly, rather than forcing dense low-rank condensation.
  p.objective.low_rank.push_back({0.25,{1.0,0.0}});

  ConvexQuadraticModel curved;
  curved.constant=0.0;
  curved.linear={1.0,0.0};
  curved.base_hessian.n=2;
  curved.base_hessian.entries={{1,1,1.0e4}};
  p.inequalities.push_back(curved);
  p.inequalities.push_back(linear_row(-1.0,-1.0,0.0));
  p.inequalities.push_back(linear_row( 0.0, 0.0,-1.0));
  p.inequalities.push_back(linear_row(-1.0, 0.0, 1.0));
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
    Settings common;
    common.verbose=false;
    common.max_inner_iterations=300;
    common.inner_tolerance=1e-7;
    common.inner_iteration_limit_mu_fallback=1e-6;
    common.exact_eigen_threshold=256;
    common.lazyqp_low_rank_form="bordered";

    Settings full=common;
    full.lazyqp_kkt_form="full";
    Settings reduced=common;
    reduced.lazyqp_kkt_form="reduced";
    reduced.lazyqp_numeric_kkt_reuse=true;
    Settings reduced_reference=reduced;
    reduced_reference.lazyqp_numeric_kkt_reuse=false;
    Settings guarded_fallback=reduced;
    guarded_fallback.lazyqp_numeric_kkt_reuse_max_iterations=1;

    const InnerResult rf=ConvexQCQPLazyQPKernel(full).solve(make_problem());
    const InnerResult rr=ConvexQCQPLazyQPKernel(reduced).solve(make_problem());
    const InnerResult rr_reference=
        ConvexQCQPLazyQPKernel(reduced_reference).solve(make_problem());
    const InnerResult rr_fallback=
        ConvexQCQPLazyQPKernel(guarded_fallback).solve(make_problem());
    if(!rf.success || !rr.success || !rr_reference.success ||
       !rr_fallback.success)
      throw std::runtime_error("full/reduced LazyQP solve failed: full='"+rf.message+
                               "' reduced='"+rr.message+"'");
    if(rf.y.size()!=2 || rr.y.size()!=2)
      throw std::runtime_error("full/reduced solution dimension mismatch");
    if(std::abs(rf.y[0]-rr.y[0])>2e-6 || std::abs(rf.y[1]-rr.y[1])>2e-7)
      throw std::runtime_error("inequality-reduced KKT changed the minimizer");
    if(rf.lazyqp_full_kkt_factorizations<1 || rf.lazyqp_reduced_kkt_factorizations!=0)
      throw std::runtime_error("forced full KKT path was not exercised exclusively");
    if(rr.lazyqp_reduced_kkt_factorizations<1 || rr.lazyqp_reduced_kkt_fallbacks!=0)
      throw std::runtime_error("forced reduced KKT path was not exercised cleanly");
    if(rf.lazyqp_kkt_dimension!=111 || rf.lazyqp_full_kkt_dimension!=111)
      throw std::runtime_error("unexpected full KKT dimension");
    if(rr.lazyqp_kkt_dimension!=3 || rr.lazyqp_full_kkt_dimension!=111)
      throw std::runtime_error("unexpected reduced/full KKT dimensions");
    if(rr.dual_inf>1e-5 || rr.primal_residual_inf>1e-5 ||
       rr.inequality_inf>1e-5 || rr.complementarity>2e-6)
      throw std::runtime_error("reduced KKT solution residual is too large");
    if(std::abs(rr.y[0]-rr_reference.y[0])>2e-7 ||
       std::abs(rr.y[1]-rr_reference.y[1])>2e-7) {
      std::ostringstream message;
      message << "residual-certified numerical reuse changed the minimizer: reuse=("
              << rr.y[0] << "," << rr.y[1] << ") reference=("
              << rr_reference.y[0] << "," << rr_reference.y[1] << ") delta=("
              << std::abs(rr.y[0]-rr_reference.y[0]) << ","
              << std::abs(rr.y[1]-rr_reference.y[1]) << ") residuals reuse="
              << rr.dual_inf << "/" << rr.primal_residual_inf << "/"
              << rr.complementarity << " reference=" << rr_reference.dual_inf
              << "/" << rr_reference.primal_residual_inf << "/"
              << rr_reference.complementarity;
      message << " iterations=" << rr.iterations << "/"
              << rr_reference.iterations << " factors="
              << rr.lazyqp_reduced_kkt_factorizations << "/"
              << rr_reference.lazyqp_reduced_kkt_factorizations
              << " reuse=" << rr.lazyqp_numeric_kkt_reuse_attempts << "/"
              << rr.lazyqp_numeric_kkt_reuse_acceptances << "/"
              << rr.lazyqp_numeric_kkt_reuse_fallbacks;
      throw std::runtime_error(message.str());
    }
    if(rr.lazyqp_numeric_kkt_reuse_attempts<1 ||
       rr.lazyqp_numeric_kkt_reuse_acceptances<1 ||
       !(rr.lazyqp_reduced_kkt_factorizations<
         rr_reference.lazyqp_reduced_kkt_factorizations))
      throw std::runtime_error("numerical KKT reuse did not avoid a factorization");
    if(rr_fallback.lazyqp_numeric_kkt_reuse_fallbacks<1 ||
       std::abs(rr_fallback.y[0]-rr_reference.y[0])>2e-7 ||
       std::abs(rr_fallback.y[1]-rr_reference.y[1])>2e-7)
      throw std::runtime_error("failed Krylov reuse did not refactorize safely");

    std::cout << "[PASS] exact bordered inequality KKT reduction"
              << " full_dim="<<rf.lazyqp_kkt_dimension
              << " reduced_dim="<<rr.lazyqp_kkt_dimension
              << " full_fact="<<rf.lazyqp_full_kkt_factorizations
              << " reduced_fact="<<rr.lazyqp_reduced_kkt_factorizations
              << "/"<<rr_reference.lazyqp_reduced_kkt_factorizations
              << " krylov="<<rr.lazyqp_numeric_kkt_reuse_iterations
              << " guarded_fallback="
              << rr_fallback.lazyqp_numeric_kkt_reuse_fallbacks
              << " it_full="<<rf.iterations
              << " it_reduced="<<rr.iterations<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
