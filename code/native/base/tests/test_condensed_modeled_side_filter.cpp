#include "lazyipdca_qcqp/solver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

// At the origin the tangent equation of y-x^2=0 is d_y=0, so the raw
// majorant step respects y<=0.09.  Its exact condensed retraction has
// y=(x+d_x)^2 and can cross that already-modeled side constraint.  This is
// the two-variable analogue of the modeled apparent-flow Maratos effect.
QCQPProblem make_modeled_side_problem() {
  const double inf=std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name="condensed_modeled_side_filter";
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
  g.name="modeled_side_cap";
  g.constant=-0.09;
  g.linear={0.0,1.0};              // y-0.09<=0
  g.hessian.n=2;
  p.inequalities.push_back(std::move(g));

  p.validate();
  return p;
}

QCQPProblem make_initially_side_infeasible_problem() {
  const double inf=std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name="condensed_modeled_side_filter_infeasible_start";
  p.n=3;
  // Equality-feasible, but z<=0 is violated.  The condensed correction for
  // y-x^2 has no z component, so it cannot create an authentic side excursion.
  p.x0={0.0,0.0,0.1};
  p.lb={-1.0,-inf,-1.0};
  p.ub={ 1.0, inf, 1.0};

  p.objective.name="minus_x";
  p.objective.linear={-1.0,0.0,0.0};
  p.objective.hessian.n=3;

  QuadraticFunction h;
  h.name="parabola";
  h.linear={0.0,1.0,0.0};
  h.hessian.n=3;
  h.hessian.entries={{0,0,-2.0}};
  p.equalities.push_back(std::move(h));

  QuadraticFunction g;
  g.name="preexisting_z_violation";
  g.linear={0.0,0.0,1.0};
  g.hessian.n=3;
  p.inequalities.push_back(std::move(g));
  p.validate();
  return p;
}

Settings settings(bool modeled_side_filter) {
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
  st.dca_condensed_post_retraction_working_set_expansion=false;

  st.max_outer_iterations=200;
  st.max_inner_iterations=1000;
  st.line_search_max_steps=40;
  st.feasibility_tolerance=1e-7;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-10;
  // Keep y<=0.09 in the solved majorant from the initial point and make the
  // transient tube wide enough to admit the deterministic alpha=1/2 step.
  st.working_set_activation=0.2;
  st.dca_condensed_modeled_side_filter=modeled_side_filter;
  st.dca_condensed_modeled_side_filter_limit=0.2;
  st.dca_condensed_modeled_side_recovery_margin=1.0;
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
    const QCQPProblem p=make_modeled_side_problem();
    const QCQPProblem infeasible_start=
        make_initially_side_infeasible_problem();
    const Settings historical_settings=settings(false);
    const Settings filtered_settings=settings(true);
    Settings one_outer_settings=filtered_settings;
    one_outer_settings.max_outer_iterations=1;
    Settings two_outer_settings=filtered_settings;
    two_outer_settings.max_outer_iterations=2;
    Settings infeasible_historical_settings=settings(false);
    infeasible_historical_settings.max_outer_iterations=1;
    Settings infeasible_filtered_settings=settings(true);
    infeasible_filtered_settings.max_outer_iterations=1;

    // The generic default is part of the compatibility contract: application
    // runners must opt into the transient filter explicitly.
    if(Settings{}.dca_condensed_modeled_side_filter)
      throw std::runtime_error("modeled-side filter is not default-off");

    const Result historical=Solver(historical_settings).solve(p);
    const Result filtered=Solver(filtered_settings).solve(p);
    const Result one_outer=Solver(one_outer_settings).solve(p);
    const Result two_outer=Solver(two_outer_settings).solve(p);
    const Result infeasible_historical=
        Solver(infeasible_historical_settings).solve(infeasible_start);
    const Result infeasible_filtered=
        Solver(infeasible_filtered_settings).solve(infeasible_start);

    require_certificate(historical,historical_settings,"historical");
    require_certificate(filtered,filtered_settings,"filtered");

    if(historical.lazy_dca_condensed_modeled_side_filter_attempts!=0 ||
       historical.lazy_dca_condensed_modeled_side_filter_acceptances!=0 ||
       historical.lazy_dca_condensed_modeled_side_filter_recovery_acceptances!=0 ||
       historical.lazy_dca_condensed_modeled_side_filter_fallbacks!=0 ||
       historical.lazy_dca_condensed_modeled_side_filter_recovery_margin_rows!=0 ||
       historical.lazy_dca_condensed_modeled_side_filter_max_violation!=0.0)
      throw std::runtime_error("disabled filter changed its diagnostic counters");

    if(filtered.lazy_dca_condensed_modeled_side_filter_attempts<=0 ||
       filtered.lazy_dca_condensed_modeled_side_filter_acceptances<=0 ||
       !(filtered.lazy_dca_condensed_modeled_side_filter_max_violation>
         filtered_settings.feasibility_tolerance))
      throw std::runtime_error(
          "enabled route did not exercise a modeled transient side step");

    // A transient step is permitted only as an iterate.  With the run stopped
    // immediately after that first accepted excursion, SOLVED must remain
    // impossible while the original inequality is outside the strict tube.
    if(one_outer.lazy_dca_condensed_modeled_side_filter_acceptances<=0)
      throw std::runtime_error("one-outer route did not accept the test excursion");
    if(one_outer.feasibility.inequality_inf<=
       one_outer_settings.feasibility_tolerance)
      throw std::runtime_error("one-outer route did not remain transiently infeasible");
    if(one_outer.status==Status::Solved)
      throw std::runtime_error("transient side tube bypassed strict termination");

    // The next accepted relaxed step must contract the side violation.  This
    // pins the recovery branch, rather than merely observing filter attempts
    // on an otherwise historical trajectory.
    if(two_outer.lazy_dca_condensed_modeled_side_filter_acceptances<2 ||
       two_outer.lazy_dca_condensed_modeled_side_filter_recovery_acceptances<=0 ||
       two_outer.lazy_dca_condensed_modeled_side_filter_recovery_margin_rows<=0 ||
       !(two_outer.feasibility.inequality_inf<
         one_outer.feasibility.inequality_inf))
      throw std::runtime_error("modeled-side recovery did not reduce violation");
    if(two_outer.feasibility.inequality_inf>
           two_outer_settings.feasibility_tolerance &&
       two_outer.status==Status::Solved)
      throw std::runtime_error("recovery iterate bypassed strict termination");

    // An arbitrary infeasible Phase-II start carries no recovery provenance.
    // With restoration deliberately disabled in settings(), enabling the
    // filter must leave this one-outer historical trajectory untouched.
    if(infeasible_filtered.lazy_dca_condensed_modeled_side_filter_attempts!=0 ||
       infeasible_filtered.lazy_dca_condensed_modeled_side_filter_acceptances!=0 ||
       infeasible_filtered.
           lazy_dca_condensed_modeled_side_filter_recovery_acceptances!=0 ||
       infeasible_filtered.
           lazy_dca_condensed_modeled_side_filter_recovery_margin_rows!=0 ||
       infeasible_filtered.lazy_dca_condensed_modeled_side_filter_fallbacks!=0 ||
       infeasible_filtered.
           lazy_dca_condensed_modeled_side_filter_max_violation!=0.0)
      throw std::runtime_error(
          "infeasible start entered relaxed recovery without provenance");
    if(infeasible_filtered.status!=infeasible_historical.status ||
       infeasible_filtered.outer_iterations!=
           infeasible_historical.outer_iterations ||
       infeasible_filtered.inner_iterations!=
           infeasible_historical.inner_iterations ||
       infeasible_filtered.line_search_backtracks!=
           infeasible_historical.line_search_backtracks ||
       std::abs(infeasible_filtered.objective-
                infeasible_historical.objective)>2e-12 ||
       std::abs(infeasible_filtered.x[0]-infeasible_historical.x[0])>2e-12 ||
       std::abs(infeasible_filtered.x[1]-infeasible_historical.x[1])>2e-12)
      throw std::runtime_error(
          "filter changed the historical infeasible-start trajectory");

    const double objective_scale=std::max(
        {1.0,std::abs(historical.objective),std::abs(filtered.objective)});
    if(std::abs(historical.objective-filtered.objective)>
           5e-6*objective_scale ||
       std::abs(historical.x[0]-filtered.x[0])>5e-6 ||
       std::abs(historical.x[1]-filtered.x[1])>5e-6)
      throw std::runtime_error(
          "transient filter changed the certified solution materially");

    std::cout << "[PASS] condensed modeled-side filter"
              << " objective=" << filtered.objective
              << " equality=" << filtered.feasibility.equality_inf
              << " inequality=" << filtered.feasibility.inequality_inf
              << " dual=" << filtered.dual_infeasibility_inf
              << " attempts="
              << filtered.lazy_dca_condensed_modeled_side_filter_attempts
              << " acceptances="
              << filtered.lazy_dca_condensed_modeled_side_filter_acceptances
              << " fallbacks="
              << filtered.lazy_dca_condensed_modeled_side_filter_fallbacks
              << " max-side="
              << filtered.lazy_dca_condensed_modeled_side_filter_max_violation
              << " outer=" << historical.outer_iterations << "->"
              << filtered.outer_iterations
              << " one-outer-status=" << static_cast<int>(one_outer.status)
              << " one-outer-side=" << one_outer.feasibility.inequality_inf
              << " two-outer-side=" << two_outer.feasibility.inequality_inf
              << " recovery="
              << filtered.
                  lazy_dca_condensed_modeled_side_filter_recovery_acceptances
              << " infeasible-start-attempts="
              << infeasible_filtered.
                  lazy_dca_condensed_modeled_side_filter_attempts
              << "\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
