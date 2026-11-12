#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace lazyipdca::qcqp;

namespace {
int idx(int joint,int coordinate) { return 3*joint+coordinate; }

void add_square_difference(QuadraticFunction& h,int a,int b) {
  h.hessian.entries.push_back({std::min(a,b),std::min(a,b),2.0});
  h.hessian.entries.push_back({std::max(a,b),std::max(a,b),2.0});
  h.hessian.entries.push_back({std::min(a,b),std::max(a,b),-2.0});
}

QCQPProblem make_catenary_like(int beams) {
  constexpr double fraction=0.6;
  constexpr double total_mass=500.0;
  constexpr double gravity=9.81;
  const double mg=(total_mass/beams)*gravity;

  QCQPProblem p;
  p.name="CATENARY_multiplier_transport";
  p.n=3*(beams+1);
  p.x0.assign(p.n,0.0);
  p.lb.assign(p.n,-std::numeric_limits<double>::infinity());
  p.ub.assign(p.n, std::numeric_limits<double>::infinity());
  p.objective.name="gravity";
  p.objective.linear.assign(p.n,0.0);
  p.objective.hessian.n=p.n;
  p.objective.linear[idx(0,1)]=0.5*mg;
  for(int i=1;i<beams;++i) p.objective.linear[idx(i,1)]=mg;
  p.objective.linear[idx(beams,1)]=0.5*mg;

  for(int c=0;c<3;++c) p.lb[idx(0,c)]=p.ub[idx(0,c)]=0.0;
  const double length=fraction*beams;
  p.lb[idx(beams,0)]=p.ub[idx(beams,0)]=length;

  for(int i=1;i<=beams;++i) {
    QuadraticFunction h;
    h.name="beam_"+std::to_string(i);
    h.constant=-1.0;
    h.linear.assign(p.n,0.0);
    h.hessian.n=p.n;
    add_square_difference(h,idx(i,0),idx(i-1,0));
    add_square_difference(h,idx(i,1),idx(i-1,1));
    // Deliberately retain the historical CATENARY cross-coordinate defect:
    // (X_i-Z_{i-1})^2 rather than (Z_i-Z_{i-1})^2.
    add_square_difference(h,idx(i,0),idx(i-1,2));
    p.equalities.push_back(std::move(h));
  }

  for(int i=1;i<=beams;++i) p.x0[idx(i,0)]=fraction;
  p.x0[idx(beams,0)]=length;
  return p;
}
}

int main() {
  Settings st;
  st.verbose=std::getenv("LAZYIPDCA_TEST_VERBOSE")!=nullptr;
  st.outer_algorithm="lazy-dca";
  st.inner_backend="lazyqp";
  st.linear_backend=linear_backend_compiled("ldl") ? "ldl" : "native";
  st.kkt_adaptive_regularization=true;
  st.max_outer_iterations=100;
  st.max_inner_iterations=1000;
  st.time_limit_seconds=20.0;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-8;
  st.inner_iteration_limit_mu_fallback=1e-7;
  st.dca_step_control=true;
  st.dca_equality_model="linearized";
  // This regression isolates multiplier transport.  Tangent activation has its
  // own dedicated regression and must not perturb this trajectory.
  st.restoration_tangent_activation=false;
  st.dca_step_target_absolute=1.0;
  st.dca_step_target_relative=1e-2;
  st.dca_step_target_maximum=10.0;

  const QCQPProblem p=make_catenary_like(50);
  const Result r=Solver(st).solve(p);

  if(r.status!=Status::Solved || !r.ipopt_desired_convergence ||
     r.feasibility.total_inf>st.feasibility_tolerance ||
     r.dual_infeasibility_inf>st.dual_infeasibility_tolerance) {
    std::cerr << "[FAIL] CATENARY multiplier transport did not converge"
              << " status="<<status_string(r.status)
              << " objective="<<r.objective
              << " feas="<<r.feasibility.total_inf
              << " dual="<<r.dual_infeasibility_inf
              << " outer="<<r.outer_iterations
              << " message="<<r.message<<"\n";
    return 1;
  }
  if(r.multiplier_refit_attempts<=0 || r.multiplier_refit_successes<=0 ||
     r.multiplier_row_space_factorizations<=0 ||
     r.multiplier_row_space_solves<=0) {
    std::cerr << "[FAIL] sparse multiplier refit path was not exercised"
              << " refit="<<r.multiplier_refit_attempts<<"/"
              <<r.multiplier_refit_successes
              << " fact="<<r.multiplier_row_space_factorizations
              << " solve="<<r.multiplier_row_space_solves<<"\n";
    return 2;
  }
  // The continuation fallback must be completely dormant on the established
  // CATENARY trajectory: it is a numerical-failure recovery policy, not a new
  // multiplier update rule.
  if(r.multiplier_refit_fallback_reuses!=0) {
    std::cerr << "[FAIL] CATENARY unexpectedly used the last-valid multiplier "
                 "fallback: " << r.multiplier_refit_fallback_reuses << "\n";
    return 8;
  }
  if(r.terminal_kkt_accepted_steps<=0) {
    std::cerr << "[FAIL] guarded sparse terminal KKT correction was not used\n";
    return 3;
  }
  if(p.n<=64 || r.terminal_kkt_iterations<=r.terminal_kkt_attempts) {
    std::cerr << "[FAIL] large equality-only terminal Newton did not take "
                 "autonomous control: n=" << p.n
              << " attempts=" << r.terminal_kkt_attempts
              << " iterations=" << r.terminal_kkt_iterations << "\n";
    return 4;
  }
  // r3p24 commits only objective-monotone terminal steps. A later step in the
  // autonomous sequence may therefore be rolled back while the accepted
  // prefix and the surrounding Lazy-DCA route still obtain the full original
  // KKT certificate.
  if(r.terminal_kkt_attempts!=1 || r.terminal_kkt_accepted_steps<1 ||
     r.terminal_kkt_accepted_steps>r.terminal_kkt_iterations) {
    std::cerr << "[FAIL] guarded autonomous terminal Newton accounting is "
                 "inconsistent: attempts="
              << r.terminal_kkt_attempts
              << " iterations=" << r.terminal_kkt_iterations
              << " accepted=" << r.terminal_kkt_accepted_steps << "\n";
    return 7;
  }
  for(int c=0;c<3;++c) {
    if(r.x[idx(0,c)]!=p.lb[idx(0,c)]) {
      std::cerr << "[FAIL] terminal Newton moved a fixed left endpoint: c="
                << c << " value=" << r.x[idx(0,c)] << "\n";
      return 5;
    }
  }
  if(r.x[idx(50,0)]!=p.lb[idx(50,0)]) {
    std::cerr << "[FAIL] terminal Newton moved the fixed right endpoint: value="
              << r.x[idx(50,0)] << "\n";
    return 6;
  }
  std::cout << "[PASS] CATENARY sparse multiplier transport"
            << " outer="<<r.outer_iterations
            << " dual="<<r.dual_infeasibility_inf
            << " refit="<<r.multiplier_refit_successes
            << " fallback="<<r.multiplier_refit_fallback_reuses
            << " rowspace="<<r.multiplier_row_space_factorizations<<"/"
            <<r.multiplier_row_space_solves
            << " terminal="<<r.terminal_kkt_accepted_steps
            << " autonomous="<<r.terminal_kkt_iterations<<"/"
            <<r.terminal_kkt_attempts
            << "\n";
  return 0;
}
