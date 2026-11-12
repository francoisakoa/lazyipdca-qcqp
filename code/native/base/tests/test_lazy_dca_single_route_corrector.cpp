#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef LAZYIPDCA_SOURCE_DIR
#define LAZYIPDCA_SOURCE_DIR "."
#endif

using namespace lazyipdca::qcqp;

static void run_case(const std::string& name, double objective_reference,
                     double objective_tolerance, int max_outer) {
  const std::string path=std::string(LAZYIPDCA_SOURCE_DIR)+
      "/tests/regression/"+name+".qcq";
  QCQPProblem p=read_qcq(path);
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.verbose=false;
  st.max_outer_iterations=max_outer;
  st.max_inner_iterations=1000;
  st.time_limit_seconds=20.0;
  st.feasibility_tolerance=1e-6;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-8;
  st.inner_iteration_limit_mu_fallback=1e-7;
  st.inner_dual_warm_start=true;
  st.inner_inequality_dual_warm_start=true;
  st.dca_local_corrector=true;
  st.dca_equality_retraction=false;
  st.dca_equality_funnel_auto=false;

  const Result r=Solver(st).solve(p);
  if(r.effective_outer_algorithm!="lazy-dca")
    throw std::runtime_error(name+": route changed to "+r.effective_outer_algorithm);
  if(r.status!=Status::Solved || !r.ipopt_desired_convergence)
    throw std::runtime_error(name+": not solved: "+status_string(r.status)+" / "+r.message);
  if(r.feasibility.total_inf>1e-6 || r.dual_infeasibility_inf>1e-6 ||
     r.complementarity_inf>1e-6 || r.scaled_nlp_error>1e-6)
    throw std::runtime_error(name+": original-QCQP KKT certificate failed");
  if(std::abs(r.objective-objective_reference)>objective_tolerance)
    throw std::runtime_error(name+": objective regression");
  if(r.lazy_dca_corrector_attempts<=0 || r.lazy_dca_corrector_acceptances<=0)
    throw std::runtime_error(name+": local corrector was not exercised");

  std::cout << "[PASS] " << name
            << " route=" << r.effective_outer_algorithm
            << " f=" << r.objective
            << " feas=" << r.feasibility.total_inf
            << " dual=" << r.dual_infeasibility_inf
            << " outer/inner=" << r.outer_iterations << "/" << r.inner_iterations
            << " corrector=" << r.lazy_dca_corrector_acceptances
            << "\n";
}

int main() {
  try {
    run_case("BT1",-1.0,1e-7,100);
    run_case("DISCS",16.6416298,1e-5,250);
    run_case("HS116",97.5875096,1e-5,350);
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
