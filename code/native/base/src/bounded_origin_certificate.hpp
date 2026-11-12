#pragma once
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <chrono>
#include <map>

namespace lazyipdca::qcqp::detail {
// Admission for reserving one unit for the ALREADY existing terminal trial.
// This is only original primal feasibility, never a KKT certificate, and
// never replaces the supplied initial point of the primary Lazy trajectory.
inline bool bounded_origin_reservation_candidate(const QCQPProblem& p,double tolerance) {
  if(p.equalities.empty()||p.n<=0||!(tolerance>=0)||!std::isfinite(tolerance))return false;
  Vec point(p.n,0.0);
  for(int j=0;j<p.n;++j) {
    if(std::isfinite(p.lb[j]))point[j]=std::max(point[j],p.lb[j]);
    if(std::isfinite(p.ub[j]))point[j]=std::min(point[j],p.ub[j]);
  }
  // std::max can hide NaN, so test each original row before aggregation.
  // Clamping has already satisfied every finite bound side.
  double primal=0;
  for(const auto& q:p.equalities) {
    const double value=q.value(point);
    if(!std::isfinite(value))return false;
    primal=std::max(primal,std::abs(value));
  }
  for(const auto& q:p.inequalities) {
    const double value=q.value(point);
    if(!std::isfinite(value))return false;
    primal=std::max(primal,std::max(0.0,value));
  }
  return primal<=tolerance;
}
struct BoundedOriginCertificateAttempt {
  bool attempted=false,accepted=false;
  int supported_columns=0,equalities=0,factorizations=0,solves=0;
  double seconds=0,initial_primal=INFINITY,primal=INFINITY,dual=INFINITY,complementarity=INFINITY,expanded_relative_residual=INFINITY;
  std::string backend,message;
};
// A terminal certificate candidate, never a barrier-filter step. It is accepted
// only by a fresh ORIGINAL audit. Failed attempts leave every result field intact.
inline BoundedOriginCertificateAttempt bounded_origin_certificate(
    const QCQPProblem& p,const Settings& requested,Result& result,double budget_seconds) {
  using Clock=std::chrono::steady_clock;using L=long double;
  BoundedOriginCertificateAttempt out;const auto start=Clock::now();
  auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-start).count();};
  auto expired=[&]{return !(budget_seconds>0)||elapsed()>=budget_seconds;};
  auto finish=[&](const char* reason){out.message=reason;out.seconds=elapsed();return out;};
  if(result.status!=Status::InnerFailure&&result.status!=Status::FactorizationFailure&&result.status!=Status::LineSearchFailure)
    return finish("failure trigger absent");
  if(expired()||p.equalities.empty())return finish("no budget or equality rows");
  out.attempted=true;out.initial_primal=result.feasibility.total_inf;
  Vec x(p.n,0);
  for(int j=0;j<p.n;++j){if(std::isfinite(p.lb[j]))x[j]=std::max(x[j],p.lb[j]);if(std::isfinite(p.ub[j]))x[j]=std::min(x[j],p.ub[j]);}
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
    if(!std::isfinite(out.primal)||primal>requested.feasibility_tolerance)return finish("bounded origin is not original-feasible");
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
    candidate.status=Status::Solved;candidate.message="bounded-origin candidate satisfies strict independent original QCQP KKT";
    candidate.termination_multiplier_source="bounded-origin-adjoint-original-audit";
    result=std::move(candidate);out.accepted=true;return finish("strict original KKT certificate");
  } catch(const std::exception& e){out.message=e.what();out.seconds=elapsed();return out;}
}
} // namespace lazyipdca::qcqp::detail
