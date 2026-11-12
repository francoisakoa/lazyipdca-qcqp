#include "lazyipdca_qcqp/standard_auto.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main() {
  int checks=0;
  auto require=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
  QCQPProblem p;p.name="anonymous";p.n=2;p.x0={1,0};
  p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.linear={0,0};p.objective.hessian.n=2;
  QuadraticFunction h;h.linear={0,0};h.hessian.n=2;
  h.constant=-1;h.hessian.entries={{0,0,2},{1,1,2}};
  p.equalities={h,h};
  Settings st;st.linear_backend="mumps";st.dca_engine="majorants";
  for(const auto& H:std::vector<std::vector<SymTriplet>>{
      {{0,0,-2},{1,1,-2}},{{0,0,-2},{1,1,2}},{{0,1,-1}}}) {
    p.objective.hessian.entries=H;const auto original=p.objective.hessian.entries;
    require(!detail::original_diagonal_convex_equality_domain(p),"signed objective was mislabeled PSD");
    require(detail::original_quadratic_equality_domain(p),"quadratic equality domain rejected signed curvature");
    for(const char* mode:{"off","HH","HS","SH","SS"}) {
      st.convexification_study=mode;const auto selected=detail::resolve_standard_auto(p,st);
      require(selected.enabled&&selected.settings.majorant_equality_stabilization,"stabilized route not selected");
      require(selected.reason=="exact-signed-quadratic-objective-equality-domain-eligible","signed domain lacks its own reason");
      require(selected.settings.convexification_study==mode,"curvature policy was replaced");
      require(selected.settings.stationarity_tolerance==st.stationarity_tolerance&&
          selected.settings.feasibility_tolerance==st.feasibility_tolerance,"termination tolerances changed");
      detail::validate_majorant_equality_stabilization_problem(p,selected.settings);
    }
    require(p.x0==Vec({1,0})&&p.objective.hessian.entries.size()==original.size(),"source model or start changed");
  }
  st.convexification_study="off";
  auto q=p;q.lb[0]=0;require(!detail::resolve_standard_auto(q,st).enabled,"finite bound silently admitted");
  q=p;q.inequalities.push_back(h);require(!detail::resolve_standard_auto(q,st).enabled,"source inequality silently admitted");
  q=p;q.equalities[0].hessian.entries.clear();q.equalities[1].hessian.entries.clear();
  require(!detail::resolve_standard_auto(q,st).enabled,"affine-only model domain broadened");
  st.standard_r10_policy="off";require(!detail::resolve_standard_auto(p,st).enabled,"explicit off ignored");
  st.standard_r10_policy="auto";st.convexification_study="SS_legacy";
  require(!detail::resolve_standard_auto(p,st).enabled,"historical control replaced");
  st.convexification_study="off";st.majorant_equality_stabilization_explicit=true;st.majorant_equality_stabilization=false;
  require(!detail::resolve_standard_auto(p,st).enabled,"explicit stabilization off ignored");
  std::cout<<"signed_quadratic_equality_domain PASS checks="<<checks<<" optimizer_calls=0\n";
}
