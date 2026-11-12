#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

using namespace lazyipdca::qcqp;

int main() {
  QCQPProblem p = read_qcq(std::string(LAZYIPDCA_SOURCE_DIR) + "/tests/toy.qcq");
  Settings base;
  base.verbose=false;
  base.max_outer_iterations=400;
  base.max_inner_iterations=500;
  base.feasibility_tolerance=1e-7;
  base.stationarity_tolerance=1e-6;
  base.dual_infeasibility_tolerance=1e-6;
  base.complementarity_tolerance=1e-6;

  Settings sn=base; sn.linear_backend="native";
  Result rn=Solver(sn).solve(p);
  if (rn.status != Status::Solved || !rn.ipopt_desired_convergence) {
    std::cerr << "native failed original-QCQP certificate: status="
              << static_cast<int>(rn.status)
              << " dual=" << rn.dual_infeasibility_inf
              << " comp=" << rn.complementarity_inf
              << " nlp=" << rn.scaled_nlp_error << "\n";
    return 1;
  }

  // QDLDL has no numerical pivoting and is safe for our regularized KKT
  // systems only when the factorization has exactly the expected inertia.
  // QDLDL_factor returns the number of positive D pivots; make sure the
  // backend rejects a deliberately wrong primal dimension instead of merely
  // accepting any successful factorization.
  if (linear_backend_compiled("qdldl")) {
    Settings si=base; si.linear_backend="qdldl"; si.linear_metis=true;
    std::string msg;
    auto fac = make_symmetric_factorization(si,msg);
    if (!fac) { std::cerr << "qdldl factory failed: " << msg << "\n"; return 4; }
    const std::vector<SymmetricTriplet> pd2 = {{0,0,2.0},{1,1,3.0}};
    msg.clear();
    if (fac->factor(2,1,pd2,msg)) {
      std::cerr << "qdldl inertia guard accepted 2 positive pivots with expected primal dimension 1\n";
      return 5;
    }
    if (msg.find("inertia mismatch") == std::string::npos) {
      std::cerr << "qdldl wrong-inertia failure did not report inertia mismatch: " << msg << "\n";
      return 6;
    }
    msg.clear();
    if (!fac->factor(2,2,pd2,msg)) {
      std::cerr << "qdldl inertia guard rejected matching positive-definite inertia: " << msg << "\n";
      return 7;
    }
    std::cout << "[PASS] qdldl inertia guard rejects wrong positive-pivot count\n";
  }

  if (linear_backend_compiled("mumps")) {
    Settings st=base; st.linear_backend="mumps"; st.kkt_adaptive_regularization=true;
    std::string msg; auto fac=make_symmetric_factorization(st,msg);
    const std::vector<SymmetricTriplet> matrix={{0,0,2.0},{1,0,1.0},{1,1,-1.0}};
    if(!fac || !fac->factor(2,1,matrix,msg) || fac->pivot_tolerance()!=0.01) return 40;
    const int analyses=fac->symbolic_analyses();
    if(!fac->increase_quality(msg) || fac->pivot_tolerance()!=0.1 ||
       fac->symbolic_analyses()!=analyses) return 41;
    Vec rhs={4.0,-1.0};
    if(!fac->solve(rhs,msg) || std::abs(rhs[0]-1)>1e-12 || std::abs(rhs[1]-2)>1e-12) { std::cerr << "pivot solve: " << msg << " x=" << rhs[0] << "," << rhs[1] << "\n"; return 42; }
    if(fac->increase_quality(msg)) return 43;
    if(!fac->factor(2,1,matrix,msg) || fac->pivot_tolerance()!=0.1) return 44;
    std::cout << "[PASS] MUMPS pivot .01 -> .1, same matrix and symbolic analysis, cap and persistence\n";
  }

  for (const std::string b : {"ldl","qdldl","cholesk","mumps"}) {
    if (!linear_backend_compiled(b)) {
      std::cout << "[SKIP] " << b << " not compiled\n";
      continue;
    }

    // Repeated numerical refactorizations of an unchanged KKT pattern must
    // reuse the symbolic ordering/elimination analysis.
    {
      Settings sc=base; sc.linear_backend=b; sc.linear_metis=true;
      std::string msg;
      auto fac=make_symmetric_factorization(sc,msg);
      if(!fac){ std::cerr << b << " cache factory failed: " << msg << "\n"; return 8; }
      // Unsorted duplicate triplets exercise aggregation; the off-diagonal is
      // then removed and reintroduced.  Every sparse backend must retain a
      // monotone structural envelope instead of letting a numerical zero evict
      // METIS / symbolic analysis.
      const std::vector<SymmetricTriplet> a1={
          {1,1,1.0},{0,1,0.1},{0,0,1.0},{1,1,2.0},{0,0,1.0}};
      const std::vector<SymmetricTriplet> a2={
          {1,1,2.0},{0,0,2.0},{1,1,3.0},{0,0,2.0}};
      const std::vector<SymmetricTriplet> a3={
          {1,1,5.0},{0,1,-0.2},{0,0,4.0}};
      if(!fac->factor(2,2,a1,msg)){ std::cerr << b << " cache first factor failed: " << msg << "\n"; return 9; }
      const int nsa=fac->symbolic_analyses();
      const int nsi=fac->symbolic_invalidations();
      msg.clear();
      if(!fac->factor(2,2,a2,msg)){ std::cerr << b << " cache zero-pattern refactor failed: " << msg << "\n"; return 10; }
      msg.clear();
      if(!fac->factor(2,2,a3,msg)){ std::cerr << b << " cache pattern reactivation failed: " << msg << "\n"; return 11; }
      if(nsa<=0 || fac->symbolic_analyses()!=nsa ||
         fac->symbolic_invalidations()!=nsi || fac->symbolic_reuses()<2){
        std::cerr << b << " monotone symbolic envelope was not reused: analyses "
                  << nsa << " -> " << fac->symbolic_analyses()
                  << " invalidations " << nsi << " -> " << fac->symbolic_invalidations()
                  << " reuses=" << fac->symbolic_reuses() << "\n"; return 12;
      }
      std::cout << "[PASS] " << b
                << " symbolic envelope survives a temporary structural zero\n";
    }
    Settings s=base; s.linear_backend=b; s.linear_metis=true;
    Result r=Solver(s).solve(p);
    if (r.status != Status::Solved) { std::cerr << b << " failed: " << r.message << "\n"; return 2; }
    // Backend parity is defined on the ORIGINAL-QCQP KKT certificate.  With
    // SQP-linearized quadratic equalities, an accepted step may be backtracked
    // or SOC-corrected, so the fresh terminal fit is allowed to replace the
    // full-step inner multiplier map.
    const double obj_tol = 0.1 * base.stationarity_tolerance *
                           std::max(1.0, std::abs(rn.objective));
    const bool cert_ok = r.ipopt_desired_convergence &&
                         r.feasibility.total_inf <= base.feasibility_tolerance &&
                         r.dual_infeasibility_inf <= base.dual_infeasibility_tolerance &&
                         r.complementarity_inf <= base.complementarity_tolerance &&
                         r.scaled_nlp_error <= base.stationarity_tolerance;
    if (std::abs(r.objective-rn.objective)>obj_tol || !cert_ok) {
      std::cerr << b << " parity failure f=" << r.objective
                << " df=" << std::abs(r.objective-rn.objective)
                << " obj_tol=" << obj_tol
                << " feas=" << r.feasibility.total_inf
                << " model_stat=" << r.model_stationarity_inf
                << " dual=" << r.dual_infeasibility_inf
                << " comp=" << r.complementarity_inf
                << " nlp=" << r.scaled_nlp_error << "\n";
      return 3;
    }
    std::cout << "[PASS] " << b << " f=" << r.objective
              << " feas=" << r.feasibility.total_inf
              << " model_stat=" << r.model_stationarity_inf
              << " dual=" << r.dual_infeasibility_inf
              << " comp=" << r.complementarity_inf
              << " nlp=" << r.scaled_nlp_error << "\n";
  }
  std::cout << "[PASS] compiled backends: " << compiled_linear_backends() << "\n";
  return 0;
}
