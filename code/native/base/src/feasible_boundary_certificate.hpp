#pragma once
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <chrono>
#include <map>

namespace lazyipdca::qcqp::detail {
struct FeasibleBoundaryCertificateAttempt {
  bool attempted=false,accepted=false;
  int supported_columns=0,equalities=0,factorizations=0,solves=0;
  double seconds=0,initial_primal=INFINITY,primal=INFINITY,dual=INFINITY,complementarity=INFINITY,expanded_relative_residual=INFINITY;
  std::string backend,message;
};
// Recover a certificate only after a precision rejection at a feasible boundary
// point with a square equality Jacobian. No primal displacement is permitted.
// The existing bounded-origin recovery runs first and is left byte-for-byte intact.
// A full-rank equality Jacobian leaves no tangent degree of freedom with which
// to open a bound. A strictly positive barrier can then drive huge cancelling
// equality/bound multipliers; the original KKT certificate does not require them.
// No benchmark-name tests, threshold relaxation or linear-backend substitution.
inline FeasibleBoundaryCertificateAttempt feasible_boundary_certificate(
    const QCQPProblem& p,const Settings& requested,Result& result,double budget_seconds) {
  using Clock=std::chrono::steady_clock;using L=long double;
  FeasibleBoundaryCertificateAttempt out;const auto start=Clock::now();
  auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-start).count();};
  auto expired=[&]{return !(budget_seconds>0)||elapsed()>=budget_seconds;};
  auto finish=[&](const char* reason){out.message=reason;out.seconds=elapsed();return out;};
  const bool precision_failure=result.status==Status::InnerFailure &&
    (result.dca_filter.majorant_reject_precision>0 || result.message.find("non-finite")!=std::string::npos);
  const bool barrier_failure=result.status==Status::LineSearchFailure &&
    result.dca_filter.majorant_last_rejection=="barrier-filter" &&
    std::isfinite(result.complementarity_inf) && result.complementarity_inf>requested.complementarity_tolerance;
  if(requested.dca_engine!="majorants" || !requested.restoration_enabled ||
     (!precision_failure && !barrier_failure))
    return finish("boundary-barrier failure trigger absent");
  if(expired() || p.equalities.empty() || !p.inequalities.empty() ||
     result.x.size()!=static_cast<std::size_t>(p.n) || !std::all_of(result.x.begin(),result.x.end(),[](double v){return std::isfinite(v);}))
    return finish("scope or budget absent");
  if(!std::isfinite(result.feasibility.total_inf) ||
     result.feasibility.total_inf>requested.feasibility_tolerance)
    return finish("current point is not original-feasible");
  int free_variables=0;bool contact=false;
  for(int j=0;j<p.n;++j) {
    const bool fixed=std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) && p.lb[j]==p.ub[j];
    if(fixed)continue;
    ++free_variables;
    contact=contact || (std::isfinite(p.lb[j]) && std::abs(result.x[j]-p.lb[j])<=requested.feasibility_tolerance)
                    || (std::isfinite(p.ub[j]) && std::abs(result.x[j]-p.ub[j])<=requested.feasibility_tolerance);
  }
  if(!contact || free_variables!=static_cast<int>(p.equalities.size()))
    return finish("square equality / boundary contact trigger absent");
  out.attempted=true;out.initial_primal=result.feasibility.total_inf;
  const Vec x=result.x; // This exact bit pattern must be retained on success.
  struct Row {L value=0;std::vector<std::pair<int,double>> gradient;};
  auto row=[&](const QuadraticFunction& q,bool gradient) {
    Row r;r.value=q.constant;std::map<int,L> g;
    q.for_each_linear_nonzero([&](int j,double a){r.value+=L(a)*x[j];if(gradient)g[j]+=a;});
    for(auto h:q.hessian.entries){r.value+=(h.i==h.j?0.5L:1.L)*h.v*L(x[h.i])*x[h.j];if(gradient){g[h.i]+=L(h.v)*x[h.j];if(h.i!=h.j)g[h.j]+=L(h.v)*x[h.i];}}
    for(auto t:q.signed_square){r.value+=L(t.second)*x[t.first]*std::abs(L(x[t.first]));if(gradient)g[t.first]+=2*L(t.second)*std::abs(L(x[t.first]));}
    if(!std::isfinite(r.value))throw std::runtime_error("nonfinite candidate value");
    for(auto [j,a]:g)if(a!=0){const double v=double(a);if(!std::isfinite(v))throw std::runtime_error("nonfinite candidate gradient");r.gradient.push_back({j,v});}
    return r;
  };
  try {
    std::vector<Row> eq;eq.reserve(p.equalities.size());std::vector<char> supported(p.n,0);Vec column_norm(p.n,0);double jacobian_norm=0;
    L primal=0;
    for(const auto& q:p.equalities){if(expired())return finish("elapsed-time limit");eq.push_back(row(q,true));primal=std::max(primal,std::abs(eq.back().value));
      for(auto [j,v]:eq.back().gradient)if(!(std::isfinite(p.lb[j])&&p.lb[j]==p.ub[j])){column_norm[j]=std::max(column_norm[j],std::abs(v));jacobian_norm=std::max(jacobian_norm,std::abs(v));}}
    for(const auto& q:p.inequalities){if(expired())return finish("elapsed-time limit");primal=std::max(primal,row(q,false).value);}
    out.primal=double(primal);
    if(!std::isfinite(out.primal)||primal>requested.feasibility_tolerance)return finish("current point is not independently original-feasible");
    const Row objective=row(p.objective,true);if(!std::isfinite(objective.value))return finish("nonfinite candidate objective");
    Vec gradient(p.n,0);for(auto [j,v]:objective.gradient)gradient[j]=v;
    // Numerical exporters can leave roundoff-sized derivatives in otherwise
    // empty columns. Ignore them only in the certificate fit, retaining every
    // original coefficient for multiplier reconstruction and the final audit.
    const double support_floor=64*std::numeric_limits<double>::epsilon()*jacobian_norm;
    for(int j=0;j<p.n;++j)supported[j]=column_norm[j]>support_floor;
    std::vector<int> columns,map(p.n,-1);
    for(int j=0;j<p.n;++j)if(supported[j]){map[j]=int(columns.size());columns.push_back(j);}
    const int m=int(eq.size()),n=int(columns.size());out.equalities=m;out.supported_columns=n;
    if(n!=m||n==0)return finish("supported equality Jacobian is not square");
    // Unsupported columns cannot be repaired by equality multipliers. Their
    // objective gradient must already admit the exact active-bound sign.
    for(int j=0;j<p.n;++j)if(!supported[j]&&gradient[j]!=0) {
      const bool lower=std::isfinite(p.lb[j])&&x[j]==p.lb[j]&&gradient[j]>0;
      const bool upper=std::isfinite(p.ub[j])&&x[j]==p.ub[j]&&gradient[j]<0;
      if(!lower&&!upper)return finish("unsupported gradient has no active-bound certificate");
    }
    std::vector<SymmetricTriplet> matrix;Vec scaling(m,1),rhs(n+m,0);
    for(int j=0;j<n;++j){matrix.push_back({j,j,1});rhs[j]=-gradient[columns[j]];}
    for(int e=0;e<m;++e){
      double norm=0;for(auto [j,v]:eq[e].gradient)if(map[j]>=0)norm=std::max(norm,std::abs(v));
      if(!(norm>0)||!std::isfinite(norm))return finish("zero supported equality row");
      scaling[e]=norm;matrix.push_back({n+e,n+e,0});
      for(auto [j,v]:eq[e].gradient)if(map[j]>=0)matrix.push_back({map[j],n+e,v/norm});
    }
    Settings st=requested;
    st.kkt_adaptive_regularization=false;st.cholesk_adaptive_regularization=false;
    st.kkt_primal_regularization=0;st.kkt_dual_regularization=0;
    st.kkt_primal_first_ordering=true;st.kkt_interleaved_equality_ordering=false;st.kkt_preserve_structural_zeros=true;
    std::string why;auto factor=make_symmetric_factorization(st,why);
    if(!factor){out.message=why;out.seconds=elapsed();return out;}
    out.backend=factor->name();
    if(expired())return finish("elapsed-time limit");
    ++out.factorizations;if(!factor->factor(n+m,n,matrix,why)){out.message="certificate factorization: "+why;out.seconds=elapsed();return out;}
    if(expired())return finish("elapsed-time limit");
    Vec solution=rhs;++out.solves;
    if(!factor->solve(solution,why)){out.message="certificate adjoint solve: "+why;out.seconds=elapsed();return out;}
    // Refine the unperturbed symmetric system with the same selected factor.
    // The final original KKT audit is stricter than a successful backend call.
    for(int refinement=0;refinement<3&&!expired();++refinement) {
      std::vector<L> residual(rhs.begin(),rhs.end());
      for(auto a:matrix){residual[a.i]-=L(a.value)*solution[a.j];if(a.i!=a.j)residual[a.j]-=L(a.value)*solution[a.i];}
      double norm=0;Vec correction(n+m);for(int j=0;j<n+m;++j){correction[j]=double(residual[j]);norm=std::max(norm,std::abs(correction[j]));}
      if(norm<=0.01*requested.stationarity_tolerance)break;
      ++out.solves;if(!factor->solve(correction,why))return finish("certificate refinement rejected");
      axpy(1,correction,solution);
    }
    if(expired())return finish("elapsed-time limit");
    std::vector<L> expanded_residual(rhs.begin(),rhs.end());
    for(auto a:matrix){expanded_residual[a.i]-=L(a.value)*solution[a.j];if(a.i!=a.j)expanded_residual[a.j]-=L(a.value)*solution[a.i];}
    L residual_norm=0,rhs_norm=1;
    for(int j=0;j<n+m;++j){if(!std::isfinite(solution[j])||!std::isfinite(expanded_residual[j]))return finish("nonfinite adjoint solution");residual_norm=std::max(residual_norm,std::abs(expanded_residual[j]));rhs_norm=std::max(rhs_norm,std::abs(L(rhs[j])));}
    out.expanded_relative_residual=double(residual_norm/rhs_norm);
    if(!(out.expanded_relative_residual<=requested.kkt_backward_error_reject))return finish("unperturbed expanded adjoint residual rejected");
    Result candidate=result;candidate.x=x;candidate.certificate_equality_multipliers.assign(m,0);
    candidate.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
    candidate.certificate_lower_multipliers.assign(p.n,0);candidate.certificate_upper_multipliers.assign(p.n,0);
    std::vector<L> stationarity(gradient.begin(),gradient.end());
    for(int e=0;e<m;++e){const double multiplier=solution[n+e]/scaling[e];candidate.certificate_equality_multipliers[e]=multiplier;
      if(!std::isfinite(multiplier))return finish("nonfinite candidate multiplier");
      for(auto [j,v]:eq[e].gradient)stationarity[j]+=L(multiplier)*v;}
    for(int j=0;j<p.n;++j){
      if(!std::isfinite(stationarity[j]))return finish("nonfinite candidate stationarity");
      if(std::isfinite(p.lb[j])&&x[j]==p.lb[j]&&stationarity[j]>0)candidate.certificate_lower_multipliers[j]=double(stationarity[j]);
      if(std::isfinite(p.ub[j])&&x[j]==p.ub[j]&&stationarity[j]<0)candidate.certificate_upper_multipliers[j]=double(-stationarity[j]);
    }
    Settings audit=requested;audit.rank_presolve_context=nullptr;audit.termination_test="project";
    candidate.termination_scaling_available=false;candidate.termination_audit_valid=false;
    const bool strict=audit_original_termination(p,audit,candidate)&&candidate.feasibility.total_inf<=requested.feasibility_tolerance&&
        candidate.dual_infeasibility_inf<=requested.stationarity_tolerance&&candidate.complementarity_inf<=requested.complementarity_tolerance;
    out.primal=candidate.feasibility.total_inf;out.dual=candidate.dual_infeasibility_inf;out.complementarity=candidate.complementarity_inf;
    if(!strict)return finish("original KKT certificate rejected");
    if(expired())return finish("elapsed-time limit");
    candidate.original_failure_status=status_string(result.status);candidate.original_failure_message=result.message;
    candidate.status=Status::Solved;candidate.message="fixed-primal feasible-boundary adjoint satisfies strict independent original QCQP KKT";
    candidate.termination_multiplier_source="feasible-boundary-fixed-primal-adjoint-original-audit";
    candidate.termination_certificate_rescued=true;
    result=std::move(candidate);out.accepted=true;return finish("strict original KKT certificate");
  } catch(const std::exception& e){out.message=e.what();out.seconds=elapsed();return out;}
}
} // namespace lazyipdca::qcqp::detail
