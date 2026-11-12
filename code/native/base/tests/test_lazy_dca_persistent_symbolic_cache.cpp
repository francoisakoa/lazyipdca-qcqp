#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_problem() {
  // min x subject to 1-x^2 <= 0 and 1/2 <= x <= 3, starting at x=2.
  // The DCA majorant of the concave row is
  //   1+x_k^2-2 x_k x <= 0,
  // hence x_{k+1}=(x_k+1/x_k)/2.  Several outer majorants have the
  // same KKT sparsity pattern before converging to x=1.
  QCQPProblem p;
  p.name="persistent_symbolic_outer_sequence";
  p.n=1;
  p.x0={2.0};
  p.lb={0.5};
  p.ub={3.0};
  p.objective.name="f";
  p.objective.linear={1.0};
  p.objective.hessian.n=1;

  QuadraticFunction outside_unit_interval;
  outside_unit_interval.name="outside_unit_interval";
  outside_unit_interval.constant=1.0;
  outside_unit_interval.linear={0.0};
  outside_unit_interval.hessian.n=1;
  outside_unit_interval.hessian.entries={{0,0,-2.0}};
  p.inequalities={outside_unit_interval};
  p.validate();
  return p;
}

Settings make_settings(bool persistent) {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.inner_backend="lazyqp";
  st.linear_backend="ldl";
  st.linear_metis=true;
  st.kkt_persistent_symbolic_cache=persistent;
  st.kkt_adaptive_regularization=true;
  st.lazyqp_kkt_form="full";
  st.verbose=false;
  st.max_outer_iterations=30;
  st.max_inner_iterations=200;
  st.feasibility_tolerance=1e-8;
  st.stationarity_tolerance=1e-7;
  st.dual_infeasibility_tolerance=1e-7;
  st.complementarity_tolerance=1e-7;
  st.inner_tolerance=1e-9;
  st.inner_iteration_limit_mu_fallback=1e-7;
  st.dca_full_constraints_for_affine_objective=true;
  st.dca_affine_full_constraint_limit=16;
  // Isolate symbolic reuse from the independently tested lazy working-set
  // policy and optional local accelerations.  The KKT envelope must keep one
  // fixed sparsity pattern throughout this regression.
  st.dca_affine_box_bounded_lazy=false;
  st.dca_local_corrector=false;
  st.dca_iterated_soc=false;
  st.second_order_correction=false;
  st.terminal_kkt_enabled=false;
  return st;
}

} // namespace

int main() {
  try {
    if(!linear_backend_compiled("ldl")) {
      std::cout << "[SKIP] SuiteSparse LDL+METIS not compiled\n";
      return 0;
    }

    const QCQPProblem p=make_problem();
    const Result cached=Solver(make_settings(true)).solve(p);
    const Result uncached=Solver(make_settings(false)).solve(p);
    Settings reduced_settings=make_settings(true);
    reduced_settings.lazyqp_kkt_form="reduced";
    reduced_settings.lazyqp_low_rank_form="bordered";
    const Result reduced=Solver(reduced_settings).solve(p);

    if(cached.status!=Status::Solved)
      throw std::runtime_error("cached Lazy-DCA sequence failed: "+cached.message);
    if(uncached.status!=Status::Solved)
      throw std::runtime_error("uncached Lazy-DCA sequence failed: "+uncached.message);
    if(cached.outer_iterations<3 || uncached.outer_iterations<3)
      throw std::runtime_error("regression did not exercise several outer majorants");
    if(cached.x.size()!=1 || std::abs(cached.x[0]-1.0)>2e-5)
      throw std::runtime_error("cached sequence returned the wrong solution");
    if(std::abs(uncached.x[0]-cached.x[0])>1e-7)
      throw std::runtime_error("cache changed the Lazy-DCA solution");
    if(cached.kkt_symbolic_analyses!=1)
      throw std::runtime_error("persistent Solver path did not retain one symbolic analysis");
    if(cached.kkt_symbolic_reuses<=0 || cached.kkt_symbolic_invalidations!=0)
      throw std::runtime_error("persistent Solver path did not report clean symbolic reuse");
    if(cached.kkt_numeric_scatter_builds!=1 ||
       cached.kkt_numeric_scatter_reuses<1 ||
       cached.kkt_numeric_scatter_invalidations!=0)
      throw std::runtime_error("Solver did not aggregate LDL scatter telemetry");
    if(reduced.status!=Status::Solved ||
       reduced.lazyqp_kkt_triplet_layout_builds!=1 ||
       reduced.lazyqp_kkt_triplet_layout_reuses<1 ||
       reduced.lazyqp_kkt_triplet_layout_invalidations!=0)
      throw std::runtime_error("Solver did not aggregate KKT triplet-layout telemetry");
    if(uncached.kkt_symbolic_analyses<=cached.kkt_symbolic_analyses)
      throw std::runtime_error("cache-disable reference did not repeat symbolic analyses");

    std::cout << "[PASS] persistent symbolic cache crosses Lazy-DCA majorants"
              << " outer="<<cached.outer_iterations
              << " cached_analysis="<<cached.kkt_symbolic_analyses
              << " cached_reuse="<<cached.kkt_symbolic_reuses
              << " cached_invalidation="<<cached.kkt_symbolic_invalidations
              << " scatter="<<cached.kkt_numeric_scatter_builds<<"/"
              << cached.kkt_numeric_scatter_reuses<<"/"
              << cached.kkt_numeric_scatter_invalidations
              << " triplets="<<reduced.lazyqp_kkt_triplet_layout_builds<<"/"
              << reduced.lazyqp_kkt_triplet_layout_reuses<<"/"
              << reduced.lazyqp_kkt_triplet_layout_invalidations
              << " uncached_analysis="<<uncached.kkt_symbolic_analyses
              << " x="<<cached.x[0]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
