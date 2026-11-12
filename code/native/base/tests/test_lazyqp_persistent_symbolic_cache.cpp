#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"

#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

ConvexQuadraticModel linear_row(double constant,double a) {
  ConvexQuadraticModel q;
  q.constant=constant;
  q.linear={a};
  q.base_hessian.n=1;
  return q;
}

ConvexSubproblem make_problem(bool drop_second_gradient=false,
                              bool add_third_inequality=false) {
  ConvexSubproblem p;
  p.nvar=1;
  p.original_dimension=1;
  p.prefer_feasible_centered_start=true;
  p.objective.linear={-0.2};
  p.objective.base_hessian.n=1;
  p.objective.base_hessian.entries={{0,0,1.0}};
  p.equalities.nvar=1;
  p.inequalities={linear_row(-1.0,+1.0),
                  linear_row(-1.0,drop_second_gradient?0.0:-1.0)};
  if(add_third_inequality)
    p.inequalities.push_back(linear_row(-2.0,+0.5));
  return p;
}

Settings ldl_settings(bool persistent) {
  Settings st;
  st.inner_backend="lazyqp";
  st.linear_backend="ldl";
  st.linear_metis=true;
  st.kkt_persistent_symbolic_cache=persistent;
  st.kkt_adaptive_regularization=true;
  st.lazyqp_kkt_form="full";
  st.max_inner_iterations=100;
  st.inner_tolerance=1e-8;
  st.inner_iteration_limit_mu_fallback=1e-6;
  st.verbose=false;
  return st;
}

} // namespace

int main() {
  try {
    if(!linear_backend_compiled("ldl")) {
      std::cout << "[SKIP] SuiteSparse LDL+METIS not compiled\n";
      return 0;
    }

    Settings cached_settings=ldl_settings(true);
    ConvexQCQPLazyQPKernel cached(cached_settings);
    const InnerResult first=cached.solve(make_problem(false,false));
    const InnerResult second=cached.solve(make_problem(true,false));
    const InnerResult grown=cached.solve(make_problem(true,true));

    if(!first.success || !second.success || !grown.success)
      throw std::runtime_error("persistent-cache regression problem failed");
    if(first.symbolic_analyses!=1)
      throw std::runtime_error("first LDL solve did not perform exactly one symbolic analysis");
    if(first.numeric_scatter_builds!=1 || first.numeric_scatter_reuses<1 ||
       first.numeric_scatter_invalidations!=0)
      throw std::runtime_error("first LDL solve did not expose scatter build/reuse deltas");
    if(first.lazyqp_kkt_triplet_layout_builds!=1 ||
       first.lazyqp_kkt_triplet_layout_reuses<1 ||
       first.lazyqp_kkt_triplet_layout_invalidations!=0)
      throw std::runtime_error(
          "first full KKT did not retain its raw triplet layout");
    if(second.symbolic_analyses!=0 || second.symbolic_reuses<=0)
      throw std::runtime_error("second LDL solve did not reuse the cross-subproblem symbolic cache");
    if(second.symbolic_invalidations!=0)
      throw std::runtime_error("temporary structural zero invalidated the monotone symbolic envelope");
    if(second.numeric_scatter_builds!=1 ||
       second.numeric_scatter_invalidations!=1 ||
       second.numeric_scatter_reuses<1)
      throw std::runtime_error("changed raw layout did not expose scatter rebuild deltas");
    if(second.lazyqp_kkt_triplet_layout_builds!=1 ||
       second.lazyqp_kkt_triplet_layout_invalidations!=1 ||
       second.lazyqp_kkt_triplet_layout_reuses<1)
      throw std::runtime_error(
          "changed full KKT graph did not rebuild its triplet layout once");
    if(grown.symbolic_analyses!=1 || grown.symbolic_invalidations!=1)
      throw std::runtime_error("dimension growth did not invalidate and rebuild the cache exactly once");
    if(grown.numeric_scatter_builds!=1 ||
       grown.numeric_scatter_invalidations!=1)
      throw std::runtime_error("dimension growth did not expose scatter invalidation/build deltas");

    // Full and inequality-reduced KKT representations must have independent
    // factor objects.  A guarded change of representation must not evict the
    // symbolic state of the other representation.
    Settings split_settings=ldl_settings(true);
    ConvexQCQPLazyQPKernel split_cache(split_settings);
    split_settings.lazyqp_kkt_form="reduced";
    const InnerResult reduced_first=split_cache.solve(make_problem(false,false));
    split_settings.lazyqp_kkt_form="full";
    const InnerResult full_first=split_cache.solve(make_problem(false,false));
    split_settings.lazyqp_kkt_form="reduced";
    const InnerResult reduced_second=split_cache.solve(make_problem(false,false));
    if(!reduced_first.success || !full_first.success || !reduced_second.success)
      throw std::runtime_error("separate full/reduced cache regression failed");
    if(reduced_first.symbolic_analyses!=1 || full_first.symbolic_analyses!=1)
      throw std::runtime_error("first use of each KKT representation was not analyzed exactly once");
    if(reduced_second.symbolic_analyses!=0 || reduced_second.symbolic_reuses<=0 ||
       reduced_second.symbolic_invalidations!=0)
      throw std::runtime_error("full KKT use evicted the reduced symbolic cache");
    if(reduced_first.lazyqp_kkt_triplet_layout_builds!=1 ||
       full_first.lazyqp_kkt_triplet_layout_builds!=1 ||
       reduced_second.lazyqp_kkt_triplet_layout_builds!=0 ||
       reduced_second.lazyqp_kkt_triplet_layout_invalidations!=0 ||
       reduced_second.lazyqp_kkt_triplet_layout_reuses<1 ||
       reduced_second.numeric_scatter_builds!=0 ||
       reduced_second.numeric_scatter_invalidations!=0 ||
       reduced_second.numeric_scatter_reuses<1)
      throw std::runtime_error(
          "full KKT use evicted the reduced triplet/scatter caches");

    Settings uncached_settings=ldl_settings(false);
    ConvexQCQPLazyQPKernel uncached(uncached_settings);
    const InnerResult cold1=uncached.solve(make_problem(false,false));
    const InnerResult cold2=uncached.solve(make_problem(true,false));
    if(!cold1.success || !cold2.success)
      throw std::runtime_error("nonpersistent reference solve failed");
    if(cold1.symbolic_analyses!=1 || cold2.symbolic_analyses!=1)
      throw std::runtime_error("cache-disable switch did not recreate the LDL symbolic state");

    std::cout << "[PASS] persistent LDL symbolic cache"
              << " first_analysis="<<first.symbolic_analyses
              << " cross_solve_analysis="<<second.symbolic_analyses
              << " cross_solve_reuse="<<second.symbolic_reuses
              << " growth_analysis="<<grown.symbolic_analyses
              << " growth_invalidation="<<grown.symbolic_invalidations
              << " split_reduced_reuse="<<reduced_second.symbolic_reuses
              << " uncached_second_analysis="<<cold2.symbolic_analyses
              << "\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
