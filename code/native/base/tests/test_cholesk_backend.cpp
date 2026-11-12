#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace lazyipdca::qcqp;

namespace {
double max_abs(std::initializer_list<double> values) {
  double result=0.0;
  for(double value:values) result=std::max(result,std::abs(value));
  return result;
}

bool check_dense_border_ordering() {
  for(const std::string backend:{"cholesk","ldl"}) {
    if(!linear_backend_compiled(backend))continue;
    for(bool metis:{false,true})for(int kind=0;kind<3;++kind) {
      const int np=kind==1?513:519,ne=kind==1?512:(kind==2?2:1),n=np+ne;
      std::vector<SymmetricTriplet> entries;
      for(int j=0;j<np;++j)entries.push_back({j,j,4.0});
      if(kind==1) {
        for(int a=0;a<ne;++a) {
          entries.push_back({0,np+a,1.0});entries.push_back({a+1,np+a,1.0});
        }
      } else {
        for(int j=1;j<np;++j)entries.push_back({0,j,0.0625});
        entries.push_back({0,np,1.0});
        if(kind==2) {
          // The two local equality rows are dependent until the shared hub
          // is eliminated. The candidate order must fail closed and retry
          // the original order, without adding any numerical regularization.
          entries.push_back({1,np,1.0});
          entries.push_back({0,np+1,-1.0});entries.push_back({1,np+1,1.0});
        }
      }
      Vec expected(n),rhs(n,0.0);
      for(int j=0;j<n;++j)expected[j]=0.125*(j%11-5);
      for(const auto& e:entries) {
        rhs[e.i]+=e.value*expected[e.j];
        if(e.i!=e.j)rhs[e.j]+=e.value*expected[e.i];
      }
      Settings st;st.linear_backend=backend;st.linear_metis=metis;
      st.kkt_adaptive_regularization=true;st.kkt_interleaved_equality_ordering=true;
      st.kkt_primal_first_ordering=false;
      std::string message;auto factor=make_symmetric_factorization(st,message);
      Vec actual=rhs;
      if(!factor || !factor->factor(n,np,entries,message) || !factor->solve(actual,message)) {
        std::cerr<<"dense-border fixture "<<backend<<" metis="<<metis
                 <<" kind="<<kind<<": "<<message<<"\n";return false;
      }
      double error=0.0;Vec residual=rhs;
      for(int j=0;j<n;++j)error=std::max(error,std::abs(actual[j]-expected[j]));
      for(const auto& e:entries) {
        residual[e.i]-=e.value*actual[e.j];
        if(e.i!=e.j)residual[e.j]-=e.value*actual[e.i];
      }
      if(error>1e-8 || inf_norm(residual)>1e-8 ||
         factor->last_primal_regularization()!=0.0 ||
         factor->last_dual_regularization()!=0.0) return false;
      if(kind==2 && factor->symbolic_analyses()<2) {
        std::cerr<<"dependent local rows did not exercise the original-order retry\n";return false;
      }
      if(!factor->factor(n,np,entries,message) || !factor->last_symbolic_reused())return false;
    }
  }
  return true;
}

}

int main() {
  if(!check_dense_border_ordering())return 24;
  if(!linear_backend_compiled("cholesk")) {
    std::cerr<<"cholesk backend is not compiled\n";
    return 1;
  }

  // Two identical equality rows make the unregularized KKT singular.  In
  // caller-adaptive mode the backend must factor the exact matrix and fail
  // closed instead of silently changing an inertia certificate.
  const std::vector<SymmetricTriplet> dependent={
      {0,0,1.0},{0,1,1.0},{0,2,1.0},{1,1,0.0},{2,2,0.0}};
  Settings exact;
  exact.linear_backend="cholesk";
  exact.kkt_adaptive_regularization=true;
  exact.live_display_level=0;
  std::string message;
  auto exact_factor=make_symmetric_factorization(exact,message);
  if(!exact_factor) {
    std::cerr<<"exact cholesk factory failed: "<<message<<"\n";
    return 2;
  }
  if(exact_factor->factor(3,1,dependent,message)) {
    std::cerr<<"exact cholesk accepted a rank-deficient unshifted KKT\n";
    return 3;
  }
  if(exact_factor->last_primal_regularization()!=0.0 ||
     exact_factor->last_dual_regularization()!=0.0) {
    std::cerr<<"exact cholesk inserted a hidden regularization\n";
    return 4;
  }

  // The same system is valid after the caller explicitly assembles the dual
  // shift.  This is the path used by Lazy's adaptive KKT controller and by
  // exact curvature/inertia certificates.
  auto explicitly_shifted=dependent;
  explicitly_shifted.push_back({1,1,-1e-8});
  explicitly_shifted.push_back({2,2,-1e-8});
  message.clear();
  if(!exact_factor->factor(3,1,explicitly_shifted,message)) {
    std::cerr<<"explicitly shifted cholesk KKT failed: "<<message<<"\n";
    return 5;
  }

  // Fixed-mode compatibility adapts the legacy diagpert behavior: primal and
  // dual shifts are distinct and the dependent dual block is repaired.
  Settings fixed;
  fixed.linear_backend="cholesk";
  fixed.kkt_adaptive_regularization=false;
  fixed.kkt_primal_regularization=0.0;
  fixed.kkt_dual_regularization=0.0;
  fixed.kkt_rho_x_floor=0.0;
  fixed.kkt_delta_dual_floor=1e-12;
  fixed.kkt_rho_x_max=1e-2;
  fixed.kkt_delta_dual_max=1e-4;
  fixed.kkt_regularization_retry_max=12;
  fixed.kkt_backward_error_tolerance=1e-12;
  fixed.kkt_backward_error_reject=1e-9;
  fixed.kkt_iterative_refinement_steps=5;
  fixed.live_display_level=0;
  auto fixed_factor=make_symmetric_factorization(fixed,message);

  // A nonsingular saddle matrix with a structural zero on the dual diagonal
  // must remain exact: the configured floors are activated only after a
  // numerical failure, not injected into every Lazy Newton system.
  const std::vector<SymmetricTriplet> exact_saddle={
      {0,0,2.0},{0,1,1.0},{1,1,0.0}};
  message.clear();
  if(!fixed_factor || !fixed_factor->factor(2,1,exact_saddle,message)) {
    std::cerr<<"exact-first fixed cholesk factor failed: "<<message<<"\n";
    return 6;
  }
  if(fixed_factor->last_primal_regularization()!=0.0 ||
     fixed_factor->last_dual_regularization()!=0.0 ||
     fixed_factor->last_regularization_attempts()!=1) {
    std::cerr<<"exact-first fixed cholesk applied an unnecessary shift\n";
    return 7;
  }
  Vec exact_rhs={3.0,1.0};
  if(!fixed_factor->solve(exact_rhs,message) ||
     max_abs({2.0*exact_rhs[0]+exact_rhs[1]-3.0,
              exact_rhs[0]-1.0})>1e-12) {
    std::cerr<<"exact-first fixed cholesk solve failed: "<<message<<"\n";
    return 8;
  }

  message.clear();
  if(!fixed_factor || !fixed_factor->factor(3,1,dependent,message)) {
    std::cerr<<"dual-regularized cholesk factor failed: "<<message<<"\n";
    return 9;
  }
  const double primal_shift=fixed_factor->last_primal_regularization();
  const double dual_shift=fixed_factor->last_dual_regularization();
  if(!(dual_shift>0.0) || !(primal_shift>=0.0) ||
     fixed_factor->last_positive_pivots()!=1 ||
     fixed_factor->last_negative_pivots()!=2) {
    std::cerr<<"invalid cholesk shift/inertia diagnostics: primal="
             <<primal_shift<<" dual="<<dual_shift<<" inertia="
             <<fixed_factor->last_positive_pivots()<<"/"
             <<fixed_factor->last_negative_pivots()<<"\n";
    return 10;
  }
  Vec rhs={3.0,1.0,1.0};
  message.clear();
  if(!fixed_factor->solve(rhs,message)) {
    std::cerr<<"dual-regularized cholesk solve failed: "<<message<<"\n";
    return 11;
  }
  const double residual=max_abs({
      (1.0+primal_shift)*rhs[0]+rhs[1]+rhs[2]-3.0,
      rhs[0]-dual_shift*rhs[1]-1.0,
      rhs[0]-dual_shift*rhs[2]-1.0});
  if(!(residual<=1e-7) ||
     !(fixed_factor->last_backward_error()<=fixed.kkt_backward_error_reject)) {
    std::cerr<<"dual-regularized cholesk residual failed: residual="<<residual
             <<" berr="<<fixed_factor->last_backward_error()<<"\n";
    return 12;
  }

  const int analyses=fixed_factor->symbolic_analyses();
  message.clear();
  if(!fixed_factor->factor(3,1,dependent,message) ||
     fixed_factor->symbolic_analyses()!=analyses ||
     fixed_factor->symbolic_reuses()<1) {
    std::cerr<<"cholesk symbolic reuse failed: "<<message<<" analyses="
             <<analyses<<"->"<<fixed_factor->symbolic_analyses()
             <<" reuses="<<fixed_factor->symbolic_reuses()<<"\n";
    return 13;
  }

  // A clean exact factorization between two singular systems must not erase
  // the last stable repair.  With a one-step decay, the next failed exact
  // attempt should warm-start directly from half the previous dual shift.
  Settings memory=fixed;
  memory.kkt_delta_dual_floor=1e-14;
  memory.cholesk_relative_pivot_tolerance=1e-12;
  memory.cholesk_equilibration=false;
  memory.kkt_rho_x_clean_outer=3;
  memory.kkt_delta_dual_decay=0.5;
  auto memory_factor=make_symmetric_factorization(memory,message);
  const std::vector<SymmetricTriplet> memory_singular={
      {0,0,1.0},{1,1,0.0}};
  message.clear();
  if(!memory_factor || !memory_factor->factor(2,1,memory_singular,message)) {
    std::cerr<<"initial memory repair failed: "<<message<<"\n";
    return 14;
  }
  const double remembered_dual=memory_factor->last_dual_regularization();
  if(!(remembered_dual>memory.kkt_delta_dual_floor)) {
    std::cerr<<"memory fixture did not require a grown dual shift: "
             <<remembered_dual<<"\n";
    return 15;
  }
  message.clear();
  if(!memory_factor->factor(2,1,exact_saddle,message)) {
    std::cerr<<"intermediate clean factorization failed: "<<message<<"\n";
    return 16;
  }
  message.clear();
  if(!memory_factor->factor(2,1,memory_singular,message)) {
    std::cerr<<"warm-started memory repair failed: "<<message<<"\n";
    return 17;
  }
  if(memory_factor->last_regularization_attempts()!=2 ||
     memory_factor->last_dual_regularization()<
         memory.kkt_delta_dual_decay*remembered_dual) {
    std::cerr<<"stable dual shift was not reused after a clean factorization: "
             <<"remembered="<<remembered_dual
             <<" reused="<<memory_factor->last_dual_regularization()
             <<" attempts="<<memory_factor->last_regularization_attempts()<<"\n";
    return 18;
  }

  // A slightly negative primal direction requires only the primal ladder.
  const std::vector<SymmetricTriplet> wrong_primal={
      {0,0,-1e-8},{1,1,1.0},{2,2,-1.0}};
  Settings primal=fixed;
  primal.kkt_delta_dual_floor=0.0;
  auto primal_factor=make_symmetric_factorization(primal,message);
  message.clear();
  if(!primal_factor || !primal_factor->factor(3,2,wrong_primal,message)) {
    std::cerr<<"primal-regularized cholesk factor failed: "<<message<<"\n";
    return 19;
  }
  if(!(primal_factor->last_primal_regularization()>1e-8) ||
     primal_factor->last_regularization_attempts()<2 ||
     primal_factor->last_positive_pivots()!=2 ||
     primal_factor->last_negative_pivots()!=1) {
    std::cerr<<"primal cholesk ladder did not repair inertia: primal="
             <<primal_factor->last_primal_regularization()<<" dual="
             <<primal_factor->last_dual_regularization()<<" attempts="
             <<primal_factor->last_regularization_attempts()<<"\n";
    return 20;
  }

  // Majorants keep curvature recovery in outer: do not exhaust the tiny
  // numerical-shift ladder on a diagnosed inertia mismatch.
  Settings outer_owned=primal;outer_owned.dca_engine="majorants";
  auto outer_factor=make_symmetric_factorization(outer_owned,message);
  if(!outer_factor || outer_factor->factor(3,2,wrong_primal,message) ||
     outer_factor->last_regularization_attempts()!=1 ||
     message.find("inertia mismatch")==std::string::npos) {
    std::cerr<<"majorant inertia failure was not returned directly to outer\n";
    return 22;
  }

  // A unit-independent rescue: an absolute 1e-12 shift would be rounded
  // away on this 1e12 block, whose equilibrated last pivot is about 1e-15.
  Settings scaled=outer_owned;scaled.cholesk_equilibration=true;
  scaled.kkt_rho_x_floor=1e-12;scaled.kkt_rho_x_max=1e-8;
  scaled.cholesk_relative_pivot_tolerance=1e-14;
  auto scaled_factor=make_symmetric_factorization(scaled,message);
  const std::vector<SymmetricTriplet> large_weak={
      {0,0,1e12},{0,1,1e12},{1,1,1e12+0.001}};
  if(!scaled_factor || !scaled_factor->factor(2,2,large_weak,message) ||
     scaled_factor->last_regularization_attempts()!=2 ||
     scaled_factor->last_primal_regularization()!=1e-12) {
    std::cerr<<"scale-aware primal rescue failed: "<<message<<"\n";
    return 23;
  }

  // End-to-end Lazy-DCA parity on the bundled nonlinear toy model.
  QCQPProblem problem=read_qcq(std::string(LAZYIPDCA_SOURCE_DIR)+"/tests/toy.qcq");
  Settings lazy;
  lazy.linear_backend="cholesk";
  lazy.kkt_adaptive_regularization=true;
  lazy.verbose=false;
  lazy.max_outer_iterations=400;
  lazy.max_inner_iterations=500;
  lazy.feasibility_tolerance=1e-7;
  lazy.stationarity_tolerance=1e-6;
  lazy.dual_infeasibility_tolerance=1e-6;
  lazy.complementarity_tolerance=1e-6;
  const Result result=Solver(lazy).solve(problem);
  if(result.status!=Status::Solved || !result.ipopt_desired_convergence) {
    std::cerr<<"Lazy-DCA/cholesk failed: "<<result.message
             <<" feas="<<result.feasibility.total_inf
             <<" dual="<<result.dual_infeasibility_inf
             <<" comp="<<result.complementarity_inf<<"\n";
    return 21;
  }

  std::cout<<"[PASS] cholesk exact-mode guard, independent primal/dual "
              "regularization, refinement, symbolic reuse and Lazy-DCA path\n";
  return 0;
}
