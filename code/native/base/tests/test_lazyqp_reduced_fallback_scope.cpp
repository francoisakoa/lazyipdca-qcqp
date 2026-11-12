#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

int main() {
  try {
    ConvexSubproblem p;
    p.nvar=1;
    p.original_dimension=1;
    p.prefer_feasible_centered_start=true;
    p.objective.linear={0.0};
    p.objective.base_hessian.n=1;
    p.objective.base_hessian.entries={{0,0,1.0}};
    // Keep a real low-rank border so the invalid reduced denominator must
    // fall back to the complete (y,z,border) system without changing either
    // the RHS dimension or the border layout.
    p.objective.low_rank.push_back({0.5,{1.0}});
    p.equalities.nvar=1;

    // The centered start has s=1e308 and z=minimum_positive, so s/z overflows.
    // The reduced denominator guard must switch once to the full KKT before any
    // reduced factorization is attempted.  The full KKT may still reject this
    // deliberately extreme model, but it must never receive a reduced-size
    // matrix or fail later with an RHS-dimension mismatch.
    ConvexQuadraticModel g;
    g.constant=-1.0e308;
    g.linear={0.0};
    g.base_hessian.n=1;
    p.inequalities={g};

    Settings st;
    st.verbose=false;
    st.linear_backend="native";
    st.lazyqp_kkt_form="reduced";
    st.lazyqp_low_rank_form="bordered";
    st.kkt_adaptive_regularization=false;
    st.max_inner_iterations=2;

    const InnerResult r=ConvexQCQPLazyQPKernel(st).solve(p);
    if(r.success)
      throw std::runtime_error("overflow regression unexpectedly solved");
    if(r.lazyqp_reduced_kkt_fallbacks!=1)
      throw std::runtime_error("invalid Schur denominator did not trigger exactly one fallback");
    if(r.lazyqp_reduced_kkt_factorizations!=0)
      throw std::runtime_error("invalid reduced matrix was still sent to a factorization backend");
    if(r.lazyqp_full_kkt_factorizations<1)
      throw std::runtime_error("full KKT fallback was not attempted");
    if(r.message.find("RHS dimension mismatch")!=std::string::npos)
      throw std::runtime_error("reduced/full fallback still corrupted the active KKT dimension");

    std::cout << "[PASS] reduced Schur fallback scope"
              << " fallback="<<r.lazyqp_reduced_kkt_fallbacks
              << " reduced_fact="<<r.lazyqp_reduced_kkt_factorizations
              << " full_fact="<<r.lazyqp_full_kkt_factorizations
              << " message='"<<r.message<<"'\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
