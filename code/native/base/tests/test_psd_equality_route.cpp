#include "lazyipdca_qcqp/standard_auto.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main() {
  int checks=0;
  auto require=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
  QCQPProblem p;p.name="generic";p.n=2;p.x0={1,1};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.linear={0,0};p.objective.hessian.n=2;p.objective.hessian.entries={{1,1,2}};
  QuadraticFunction h;h.linear={0,1};h.hessian.n=2;h.hessian.entries={{0,0,-2}};p.equalities.push_back(h);
  Settings st;st.linear_backend="mumps";st.dca_engine="majorants";
  const auto frozen=p.x0;
  for(const char* mode:{"off","HH","HS","SH","SS"}) {
    st.convexification_study=mode;
    auto route=detail::resolve_standard_auto(p,st);
    require(route.enabled,"certified diagonal PSD objective not admitted");
    require(route.settings.majorant_equality_stabilization&&route.settings.original_stabilized_merit,"GR not active");
    require(route.settings.convexification_study==mode,"curvature configuration changed");
    detail::validate_majorant_equality_stabilization_problem(p,route.settings);
  }
  st.convexification_study="off";
  require(p.x0==frozen,"frozen start changed");
  auto altered=p;altered.objective.hessian.entries.push_back({0,0,-1e-300});
  require(!detail::original_diagonal_convex_equality_domain(altered),"negative curvature was mislabeled convex");
  require(detail::resolve_standard_auto(altered,st).enabled,"signed quadratic objective not admitted");
  altered=p;altered.objective.hessian.entries={{0,0,2},{0,1,1},{1,1,2}};
  require(!detail::original_diagonal_convex_equality_domain(altered),"offdiagonal objective was mislabeled diagonal");
  require(detail::resolve_standard_auto(altered,st).enabled,"quadratic offdiagonal objective not admitted");
  altered=p;altered.lb[0]=0;
  require(!detail::resolve_standard_auto(altered,st).enabled,"finite bound admitted through extension");
  altered=p;QuadraticFunction g;g.linear={1,0};g.hessian.n=2;altered.inequalities.push_back(g);
  require(!detail::resolve_standard_auto(altered,st).enabled,"source inequality admitted through extension");
  altered=p;altered.equalities.clear();
  require(!detail::resolve_standard_auto(altered,st).enabled,"no equality admitted");
  altered=p;altered.equalities[0].hessian.entries.clear();
  require(!detail::resolve_standard_auto(altered,st).enabled,"affine-only equalities admitted through extension");
  altered=p;altered.objective.signed_square={{0,1}};
  require(!detail::resolve_standard_auto(altered,st).enabled,"signed-square objective admitted");
  altered=p;altered.lb[0]=altered.ub[0]=1;
  require(detail::resolve_standard_auto(altered,st).enabled,"exact fixed-coordinate PSD reference extension rejected");
  for(const char* study:{"HH","HS","SH","SS"}){auto requested=st;requested.convexification_study=study;require(!detail::resolve_standard_auto(altered,requested).enabled,"fixed-coordinate extension changed a study route");}
  st.standard_r10_policy="off";
  require(!detail::resolve_standard_auto(p,st).enabled,"explicit policy OFF ignored");
  st.standard_r10_policy="auto";st.convexification_study="SS_legacy";
  require(!detail::resolve_standard_auto(p,st).enabled,"historical control changed");
  st.convexification_study="off";st.majorant_equality_stabilization_explicit=true;st.majorant_equality_stabilization=false;
  require(!detail::resolve_standard_auto(p,st).enabled,"explicit stabilization OFF ignored");
  std::cout<<"diagonal_psd_equality_route PASS checks="<<checks<<" optimizer_calls=0\n";
}
