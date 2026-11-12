#include "lazyipdca_qcqp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

// At (0,0), the affine tangent equation of y-x^2=0 is d_y=0.  Hence the
// uncorrected endpoint remains strictly below y<=c independently of d_x, while
// its exact condensed retraction has y=d_x^2 and exposes the omitted side
// constraint.  This is the minimal deterministic analogue of a flow limit
// first violated by the case118 equality correction.
QCQPProblem make_post_retraction_side_problem() {
  const double inf=std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name="condensed_post_retraction_working_set";
  p.n=2;
  p.x0={0.0,0.0};
  p.lb={-1.0,-inf};
  p.ub={ 1.0, inf};

  p.objective.name="minus_x";
  p.objective.linear={-1.0,0.0};
  p.objective.hessian.n=2;

  QuadraticFunction h;
  h.name="parabola";
  h.linear={0.0,1.0};
  h.hessian.n=2;
  h.hessian.entries={{0,0,-2.0}};  // y-x^2=0
  p.equalities.push_back(std::move(h));

  QuadraticFunction g;
  g.name="side_cap";
  g.constant=-0.09;
  g.linear={0.0,1.0};              // y-0.09<=0
  g.hessian.n=2;
  p.inequalities.push_back(std::move(g));

  p.validate();
  return p;
}

Settings settings(bool expand_after_retraction) {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.inner_backend="lazyqp";
  st.linear_backend="native";
  st.dca_equality_model="condensed-retraction";
  st.verbose=false;

  st.restoration_enabled=false;
  st.restoration_tangent_activation=false;
  st.dca_step_control=false;
  st.dca_local_corrector=false;
  st.dca_iterated_soc=false;
  st.second_order_correction=false;
  st.terminal_kkt_enabled=false;
  st.dca_condensed_alpha_predictor=false;
  st.dca_condensed_post_retraction_working_set_expansion=
      expand_after_retraction;

  st.max_outer_iterations=200;
  st.max_inner_iterations=1000;
  st.max_working_set_expansions=8;
  st.line_search_max_steps=40;
  st.feasibility_tolerance=1e-7;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-10;
  st.working_set_activation=1e-4;
  st.dca_condensed_retraction_tolerance=1e-11;
  st.dca_condensed_retraction_max_ratio=4.0;
  st.dca_condensed_newton_iterations=12;
  st.dca_condensed_krylov_iterations=12;
  st.dca_condensed_krylov_relative_tolerance=1e-10;
  st.dca_condensed_continuation_attempts=24;
  return st;
}

void require_certificate(const Result& r,const Settings& st,
                         const std::string& route) {
  if(r.status!=Status::Solved)
    throw std::runtime_error(route+" route did not solve: "+r.message);
  if(r.feasibility.equality_inf>st.feasibility_tolerance ||
     r.feasibility.inequality_inf>st.feasibility_tolerance ||
     r.feasibility.bound_inf>st.feasibility_tolerance)
    throw std::runtime_error(route+" route violated an original constraint");
  if(!std::isfinite(r.dual_infeasibility_inf) ||
     r.dual_infeasibility_inf>st.dual_infeasibility_tolerance ||
     !std::isfinite(r.complementarity_inf) ||
     r.complementarity_inf>st.complementarity_tolerance)
    throw std::runtime_error(route+" route did not retain the exact KKT test");
}

} // namespace

int main() {
  try {
    const QCQPProblem p=make_post_retraction_side_problem();
    const Settings historical_settings=settings(false);
    const Settings expanded_settings=settings(true);
    Settings capped_settings=settings(true);
    capped_settings.dca_condensed_post_retraction_working_set_max_expansions=0;
    const Result historical=Solver(historical_settings).solve(p);
    const Result expanded=Solver(expanded_settings).solve(p);
    const Result capped=Solver(capped_settings).solve(p);

    require_certificate(historical,historical_settings,"historical");
    require_certificate(expanded,expanded_settings,"post-retraction expansion");
    require_certificate(capped,capped_settings,"capped historical fallback");

    if(historical.lazy_dca_condensed_post_retraction_working_set_expansions!=0)
      throw std::runtime_error("disabled route unexpectedly expanded its working set");
    if(historical.lazy_dca_condensed_post_retraction_omitted_offenders<=0)
      throw std::runtime_error(
          "test did not expose an omitted post-retraction offender");
    if(expanded.lazy_dca_condensed_post_retraction_omitted_offenders<=0 ||
       expanded.lazy_dca_condensed_post_retraction_working_set_expansions<=0)
      throw std::runtime_error(
          "enabled route did not re-solve the enlarged majorant at the same outer");
    if(expanded.working_set_primal_transports<=0)
      throw std::runtime_error(
          "post-retraction expansion discarded the solved primal majorant state");
    if(expanded.lazy_dca_condensed_post_retraction_expansion_fallbacks!=0)
      throw std::runtime_error("small test unexpectedly exhausted the expansion cap");
    if(capped.lazy_dca_condensed_post_retraction_working_set_expansions!=0 ||
       capped.lazy_dca_condensed_post_retraction_expansion_fallbacks<=0)
      throw std::runtime_error(
          "zero expansion cap did not select the historical alpha fallback");

    // With no expansion budget, enabling the detector must be observational:
    // the complete historical line-search trajectory is retained.
    if(capped.outer_iterations!=historical.outer_iterations ||
       capped.line_search_backtracks!=historical.line_search_backtracks ||
       std::abs(capped.objective-historical.objective)>2e-12 ||
       std::abs(capped.x[0]-historical.x[0])>2e-12 ||
       std::abs(capped.x[1]-historical.x[1])>2e-12)
      throw std::runtime_error(
          "expansion-cap fallback did not reproduce the historical trajectory");

    const double objective_scale=std::max(
        {1.0,std::abs(historical.objective),std::abs(expanded.objective)});
    if(std::abs(historical.objective-expanded.objective)>
           5e-6*objective_scale ||
       std::abs(historical.x[0]-expanded.x[0])>5e-6 ||
       std::abs(historical.x[1]-expanded.x[1])>5e-6)
      throw std::runtime_error(
          "guarded working-set expansion changed the certified solution materially");

    std::cout << "[PASS] condensed post-retraction working-set expansion"
              << " objective=" << expanded.objective
              << " equality=" << expanded.feasibility.equality_inf
              << " inequality=" << expanded.feasibility.inequality_inf
              << " dual=" << expanded.dual_infeasibility_inf
              << " omitted="
              << expanded.lazy_dca_condensed_post_retraction_omitted_offenders
              << " modeled="
              << expanded.lazy_dca_condensed_post_retraction_modeled_offenders
              << " expansions="
              << expanded.lazy_dca_condensed_post_retraction_working_set_expansions
              << " transports=" << expanded.working_set_primal_transports
              << " backtracks=" << historical.line_search_backtracks << "->"
              << expanded.line_search_backtracks
              << " outer=" << historical.outer_iterations << "->"
              << expanded.outer_iterations << "\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
