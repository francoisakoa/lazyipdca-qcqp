#include "lazyipdca_qcqp/standard_auto.hpp"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace lazyipdca::qcqp;
namespace {
int checks=0;
void require(bool value,const char* message) {
  ++checks;if(!value)throw std::runtime_error(message);
}
struct Environment {
  std::vector<std::pair<std::string,std::string>> saved;
  std::vector<std::string> absent;
  Environment() {
    for(const char* key:{"LAZY_ORIGINAL_STABILIZED_MERIT","LAZY_ORIGINAL_SOURCE_PRIMAL_BARRIER",
        "LAZY_ORIGINAL_BOUND_PRODUCT_PATH","LAZY_ORIGINAL_GR_SHORT_STEP_PROX",
        "LAZY_ORIGINAL_GR_CONSISTENT_BOUND_SLACKS","LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER",
        "LAZY_ORIGINAL_GR_SOC","LAZY_EXACT_AFFINE_FACE_RESEARCH"}) {
      if(const char* value=std::getenv(key))saved.push_back({key,value});else absent.push_back(key);
      unsetenv(key);
    }
  }
  ~Environment() {
    for(const auto& key:absent)unsetenv(key.c_str());
    for(const auto& item:saved)setenv(item.first.c_str(),item.second.c_str(),1);
  }
};
QCQPProblem product() {
  QCQPProblem p;p.name="anonymous_structure";p.n=3;
  p.x0={.25,.75,.1875};p.lb={0,0,0};p.ub={1,1,1};
  p.objective.linear={0,0,-1};p.objective.hessian.n=3;
  QuadraticFunction h;h.linear={0,0,1};h.hessian.n=3;h.hessian.entries={{0,1,-1}};
  p.equalities.push_back(h);h.linear={1,1,0};h.constant=-1;h.hessian.entries.clear();
  p.equalities.push_back(h);p.validate();return p;
}
Settings native_settings() {Settings st;st.linear_backend="mumps";return st;}
void structural_selection() {
  const auto p=product();const auto requested=native_settings();
  const auto selected=detail::resolve_standard_auto(p,requested);const auto& st=selected.settings;
  require(selected.enabled&&st.dca_engine=="majorants","plain native auto did not select eligible structure");
  require(st.majorant_equality_stabilization&&st.original_stabilized_merit&&st.original_source_primal_barrier,
          "standard auto omitted original source stabilization/barrier");
  require(st.original_bound_product_path&&st.original_gr_short_step_prox&&st.original_gr_consistent_bound_slacks,
          "standard auto omitted native R10 mechanisms");
  require(st.original_gr_consistent_bound_trigger=="on_failure","standard repair is not failure-triggered");
  require(st.original_exact_affine_faces,"standard auto omitted certified face discovery");
  require(st.study_controller==requested.study_controller&&st.convexification_schedule==requested.convexification_schedule&&
          st.convexification_study==requested.convexification_study,"auto changed curvature configuration");
  require(st.termination_test==requested.termination_test&&st.stationarity_tolerance==requested.stationarity_tolerance&&
          st.feasibility_tolerance==requested.feasibility_tolerance&&
          st.dual_infeasibility_tolerance==requested.dual_infeasibility_tolerance&&
          st.complementarity_tolerance==requested.complementarity_tolerance,
          "standard selection changed original termination tolerances or semantics");
  auto renamed=p;renamed.name="unrelated_name_from_a_different_library";
  renamed.x0={.99,.01,.00001};
  const auto again=detail::resolve_standard_auto(renamed,requested);
  require(again.enabled&&again.mode==selected.mode&&again.reason==selected.reason,
          "selection depends on label or supplied start");
  require(renamed.x0==Vec({.99,.01,.00001})&&p.x0==Vec({.25,.75,.1875}),
          "policy replaced a supplied start");
  require(requested.dca_engine=="auto"&&!requested.original_stabilized_merit,
          "resolver mutated caller settings");
}
void unrelated_quadratics_and_explicit_controls() {
  const auto p=product();const auto requested=native_settings();
  auto altered=p;altered.objective.hessian.entries={{0,0,2}};
  auto selected=detail::resolve_standard_auto(altered,requested);
  require(!selected.enabled&&selected.settings.dca_engine=="legacy"&&
          !selected.settings.majorant_equality_stabilization,"unrelated quadratic objective changed route");
  altered=p;QuadraticFunction g;g.linear={0,0,0};g.hessian.n=3;g.hessian.entries={{0,0,2}};
  altered.inequalities.push_back(g);selected=detail::resolve_standard_auto(altered,requested);
  require(!selected.enabled&&selected.settings.dca_engine=="legacy"&&
          !selected.settings.original_stabilized_merit,"unrelated quadratic inequality changed route");
  altered=p;altered.equalities[0].hessian.entries.clear();
  require(!detail::resolve_standard_auto(altered,requested).enabled,"affine-only problem enters equality R10");
  altered=p;altered.equalities[0].signed_square={{0,1}};
  require(!detail::resolve_standard_auto(altered,requested).enabled,"signed-square domain entered original QCQP R10");
  for(const char* engine:{"legacy","filter"}) {
    auto st=requested;st.dca_engine=engine;selected=detail::resolve_standard_auto(p,st);
    require(!selected.enabled&&selected.settings.dca_engine==engine,"explicit engine changed");
  }
  auto st=requested;st.standard_r10_policy="off";selected=detail::resolve_standard_auto(p,st);
  require(!selected.enabled&&!selected.settings.original_stabilized_merit,"explicit standard OFF ignored");
  st=requested;st.majorant_equality_stabilization_explicit=true;st.majorant_equality_stabilization=false;
  require(!detail::resolve_standard_auto(p,st).enabled,"explicit source stabilization OFF ignored");
  st=requested;st.linear_backend="native";
  require(!detail::resolve_standard_auto(p,st).enabled,"unsupported explicit backend changed");
  st=requested;st.product_definition_presolve=true;
  require(!detail::resolve_standard_auto(p,st).enabled,"explicit alternate presolve overridden");
  st=requested;st.original_gr_consistent_bound_trigger="always";st.standard_r10_policy="off";
  selected=detail::resolve_standard_auto(p,st);
  require(selected.settings.original_gr_consistent_bound_trigger=="always","OFF mutated historical repair definition");
}
void six_curvature_definitions() {
  const auto p=product();
  for(const char* mode:{"off","HH","HS","SH","SS","SS_legacy"}) {
    auto requested=native_settings();requested.convexification_study=mode;
    requested.dca_engine="majorants";
    if(std::string(mode)=="SS_legacy")requested.majorant_conditional_retraction=false;
    const auto selected=detail::resolve_standard_auto(p,requested);
    require(selected.settings.convexification_study==mode&&
            selected.settings.convexification_schedule==requested.convexification_schedule&&
            selected.settings.study_controller==requested.study_controller&&
            selected.settings.majorant_conditional_retraction==requested.majorant_conditional_retraction,
            "one of six campaign curvature definitions changed");
    require(selected.enabled==(std::string(mode)!="SS_legacy"),"six campaign compatibility domain changed");
    if(std::string(mode)=="SS_legacy")require(selected.reason=="SS_legacy-incompatible-preserved",
            "SS_legacy fallback lacks explicit incompatibility reason");
  }
}
void campaign_route_and_frozen_start() {
  auto p=product();p.x0={.11,.89,.0979};const auto frozen=p.x0;
  for(const char* mode:{"off","HH"}) {
    auto requested=native_settings();requested.dca_engine="majorants";
    requested.convexification_study=mode;requested.dca_rank_presolve=true;
    requested.dca_rank_implied_bounds=true;requested.majorant_startup_policy="auto";
    requested.free_epigraph_presolve=true;
    if(std::string(mode)=="HH") {
      requested.accepted_checkpoint_builtin=true;
      requested.accepted_checkpoint_callback=[](const QCQPProblem&,Result&,const ConstraintPresolve*,int,double){};
    }
    const auto selected=detail::resolve_standard_auto(p,requested);const auto& st=selected.settings;
    require(selected.enabled&&st.majorant_equality_stabilization&&st.original_stabilized_merit,
            "actual campaign rank/startup argv blocks R3/HH R10 selection");
    require(!st.dca_rank_presolve&&!st.free_epigraph_presolve&&st.majorant_startup_policy=="legacy"&&st.lazy_newton_initial_mu==.1,
            "actual campaign did not select the full-source R10 startup/operator");
    require(st.original_exact_affine_faces&&!st.accepted_checkpoint_callback,
            "actual HH builtin observer prevented the R10 exact-face mechanism");
    require(p.x0==frozen,"actual campaign policy changed a frozen supplied start");
    auto explicit_mu=requested;explicit_mu.lazy_newton_initial_mu=.023;
    require(detail::resolve_standard_auto(p,explicit_mu).settings.lazy_newton_initial_mu==.023,
            "nondefault supplied barrier changed");
    explicit_mu=requested;explicit_mu.lazy_newton_initial_mu_explicit=true;
    require(detail::resolve_standard_auto(p,explicit_mu).settings.lazy_newton_initial_mu==requested.lazy_newton_initial_mu,
            "explicit default-valued barrier changed");
    requested.standard_r10_policy="off";
    const auto off=detail::resolve_standard_auto(p,requested);
    require(!off.enabled&&off.settings.dca_rank_presolve&&off.settings.majorant_startup_policy=="auto",
            "OFF did not retain actual campaign rank/startup definitions");
  }
  auto requested=native_settings();requested.outer_algorithm="auto";
  const auto selected=detail::resolve_standard_auto(p,requested);
  require(selected.enabled&&selected.settings.outer_algorithm=="lazy-dca",
          "top-level algorithm auto did not resolve consistently");
  requested=native_settings();
  requested.accepted_checkpoint_callback=[](const QCQPProblem&,Result&,const ConstraintPresolve*,int,double){};
  const auto custom=detail::resolve_standard_auto(p,requested);
  require(!custom.enabled&&bool(custom.settings.accepted_checkpoint_callback)&&
          custom.reason=="custom-observer-preserved-R10-unsupported",
          "custom API observer silently discarded by automatic R10 selection");
}
void environment_overrides() {
  const auto p=product();const auto requested=native_settings();
  setenv("LAZY_ORIGINAL_STABILIZED_MERIT","0",1);
  require(!detail::resolve_standard_auto(p,requested).enabled,"ENV GR OFF ignored");
  unsetenv("LAZY_ORIGINAL_STABILIZED_MERIT");
  setenv("LAZY_ORIGINAL_GR_SHORT_STEP_PROX","0",1);
  auto selected=detail::resolve_standard_auto(p,requested);
  require(selected.enabled&&!selected.settings.original_gr_short_step_prox,"ENV short-step OFF ignored");
  unsetenv("LAZY_ORIGINAL_GR_SHORT_STEP_PROX");
  setenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_SLACKS","0",1);
  selected=detail::resolve_standard_auto(p,requested);
  require(selected.enabled&&!selected.settings.original_gr_consistent_bound_slacks,"ENV bound repair OFF ignored");
  unsetenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_SLACKS");
  setenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER","always",1);
  require(detail::resolve_standard_auto(p,requested).settings.original_gr_consistent_bound_trigger=="always",
          "explicit research always variant changed");
  unsetenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER");
}
}
int main() {
  try {Environment clean;structural_selection();unrelated_quadratics_and_explicit_controls();
    six_curvature_definitions();campaign_route_and_frozen_start();environment_overrides();
    std::cout<<"standard_auto_policy PASS checks="<<checks<<" optimizer_calls=0\n";
  } catch(const std::exception& error) {std::cerr<<"standard_auto_policy FAIL "<<error.what()<<'\n';return 1;}
}
