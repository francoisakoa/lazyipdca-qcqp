#pragma once
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/bounded_normal_active_set.hpp"
#include <chrono>
#include <iostream>

namespace lazyipdca::qcqp::detail {
// Admit one small initial fixed-primal certificate fit before copying or
// re-auditing a workspace. This is a cost guard, not a KKT acceptance test.
// The existing terminal fitter verifies every original constraint, bound,
// multiplier sign and complementarity product before returning success.
inline bool initial_original_multiplier_fit_admitted(const QCQPProblem& p,
    const Settings& st,double original_primal,double remaining_seconds) {
  if(!(remaining_seconds>=0.005)||!std::isfinite(original_primal)||
      original_primal<0||original_primal>st.feasibility_tolerance||p.n<1||p.n>512||
      p.equalities.size()!=std::size_t(p.n)||p.has_signed_square()||
      (st.rank_presolve_context&&st.rank_presolve_context->changed()))return false;
  const long double n=p.n,columns=p.equalities.size()+p.inequalities.size()+2*n;
  // Bound the same dense rank-revealing search used at termination. In
  // particular, n=10000 exits above without a point/result/model copy.
  if(columns>1536||n*columns>1000000||
      n*columns*std::min(n,columns)>25000000||
      48*n*columns>32.0L*1024*1024)return false;
  std::size_t coefficients=p.objective.hessian.entries.size();
  for(const auto& q:p.equalities){coefficients+=q.hessian.entries.size();if(coefficients>1000000)return false;}
  for(const auto& q:p.inequalities){coefficients+=q.hessian.entries.size();if(coefficients>1000000)return false;}
  return true;
}

// Fixed-primal recovery of a tolerance-valid ORIGINAL KKT multiplier vector.
// Bounds on nonnegative multipliers enforce the original complementarity
// tolerance. This changes no primal coordinate, function model or tolerance.
inline bool terminal_original_multiplier_fit(const QCQPProblem& p,const Settings& st,
                                             Result& result,double budget_seconds) {
  using L=long double;using Clock=std::chrono::steady_clock;
  const auto began=Clock::now();
  auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-began).count();};
  if(result.status==Status::Solved||!(budget_seconds>0)||p.has_signed_square()||
     (st.rank_presolve_context&&st.rank_presolve_context->changed())||
     result.x.size()!=std::size_t(p.n)||!std::all_of(result.x.begin(),result.x.end(),[](double v){return std::isfinite(v);})||
     !(result.feasibility.total_inf<=st.feasibility_tolerance))return false;
  const double target=std::min(st.stationarity_tolerance,st.dual_infeasibility_tolerance);
  // A feasible recovered anchor can have poor auxiliary multipliers, e.g.
  // after Phase-I reset. Their residual does not determine whether ORIGINAL
  // KKT multipliers exist at this fixed x. Retain the historical coordinate
  // fast path for near certificates; send a distant guess directly to the
  // same bounded, rank-revealing least-squares search below.
  const bool near_certificate=std::isfinite(result.dual_infeasibility_inf)&&
      result.dual_infeasibility_inf<=100*target;
  struct Column {std::vector<std::pair<int,double>> entries;double u=0,scale=1,cap=INFINITY;L norm=0;int kind=0,index=0;};
  std::vector<Column> columns;std::size_t nonzeros=0;
  const Vec point=result.x,gradient=p.objective.gradient(point);
  auto add=[&](const Vec& g,double value,double old,int kind,int index) {
    Column c;c.kind=kind;c.index=index;c.scale=inf_norm(g);
    if(!(c.scale>0)||!std::isfinite(c.scale))return;
    if(kind&&value!=0)c.cap=.8*st.complementarity_tolerance*c.scale/std::abs(value);
    c.u=old*c.scale;if(kind)c.u=std::min(c.cap,std::max(0.,c.u));
    for(int j=0;j<p.n;++j)if(g[j]!=0){const double a=g[j]/c.scale;c.entries.push_back({j,a});c.norm+=L(a)*a;}
    nonzeros+=c.entries.size();columns.push_back(std::move(c));
  };
  auto old=[&](const Vec& a,int j){return j<int(a.size())&&std::isfinite(a[j])?a[j]:0.;};
  // Assembly is part of the same wall-clock/memory budget as the fit.
  // In particular a postsolve with many original rows must not spend an
  // unbounded time building columns before entering the bounded iteration.
  auto assembly_allowed=[&] {
    return elapsed()<budget_seconds&&
        nonzeros<=128u*1024u*1024u/sizeof(std::pair<int,double>);
  };
  for(int i=0;i<int(p.equalities.size());++i) {
    if(!assembly_allowed())return false;
    add(p.equalities[i].gradient(point),0,old(result.certificate_equality_multipliers,i),0,i);
  }
  for(int i=0;i<int(p.inequalities.size());++i) {
    if(!assembly_allowed())return false;
    add(p.inequalities[i].gradient(point),p.inequalities[i].value(point),
        old(result.certificate_inequality_multipliers,i),1,i);
  }
  for(int j=0;j<p.n;++j) {
    if(!assembly_allowed())return false;
    Vec g(p.n,0);
    if(std::isfinite(p.lb[j])){g[j]=-1;add(g,p.lb[j]-point[j],old(result.certificate_lower_multipliers,j),2,j);}
    if(std::isfinite(p.ub[j])){g[j]=1;add(g,point[j]-p.ub[j],old(result.certificate_upper_multipliers,j),3,j);}
  }
  if(columns.empty()||nonzeros*sizeof(std::pair<int,double>)>128u*1024u*1024u)return false;
  std::vector<L> residual;
  auto reconstruct=[&]() {
    residual.assign(gradient.begin(),gradient.end());
    for(const auto& c:columns)for(auto [j,a]:c.entries)residual[j]+=L(c.u)*a;
    L norm=0;for(L r:residual)norm=std::max(norm,std::abs(r));return double(norm);
  };
  const double before=reconstruct();double after=before;int sweeps=0;bool accepted=false;
  auto admit=[&]() {
    Result candidate=result;
    candidate.certificate_equality_multipliers.assign(p.equalities.size(),0);
    candidate.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
    candidate.certificate_lower_multipliers.assign(p.n,0);candidate.certificate_upper_multipliers.assign(p.n,0);
    for(const auto& c:columns) {
      Vec* destination=c.kind==0?&candidate.certificate_equality_multipliers:
        c.kind==1?&candidate.certificate_inequality_multipliers:
        c.kind==2?&candidate.certificate_lower_multipliers:&candidate.certificate_upper_multipliers;
      (*destination)[c.index]=c.u/c.scale;
    }
    Settings check=st;check.termination_test="project";
    candidate.termination_scaling_available=false;candidate.termination_audit_valid=false;
    const bool strict=detail::audit_original_termination(p,check,candidate)&&
      candidate.feasibility.total_inf<=st.feasibility_tolerance&&
      candidate.dual_infeasibility_inf<=target&&candidate.complementarity_inf<=st.complementarity_tolerance;
    if(!strict||candidate.x!=point||elapsed()>=budget_seconds)return false;
    candidate.original_failure_status=status_string(result.status);candidate.original_failure_message=result.message;
    candidate.status=Status::Solved;candidate.message="fixed-primal bounded multiplier fit satisfies independent original QCQP KKT";
    candidate.termination_multiplier_source="fixed-primal-complementarity-bounded-fit-original-audit";
    candidate.termination_certificate_rescued=true;result=std::move(candidate);return true;
  };
  bool dense_attempted=false;
  auto dense_fit=[&]() {
    dense_attempted=true;
    if(elapsed()>=budget_seconds)return false;
    std::vector<detail::BoundedNormalColumn> normals;
    for(const auto& c:columns) {
      detail::BoundedNormalColumn normal;normal.entries=c.entries;
      if(c.kind){normal.lower=0;normal.upper=c.cap;}
      normals.push_back(std::move(normal));
    }
    const auto dense=detail::bounded_normal_active_set(gradient,normals,budget_seconds-elapsed());
    bool strict=false;
    if(dense.valid&&dense.multipliers.size()==columns.size()) {
      for(std::size_t k=0;k<columns.size();++k)columns[k].u=dense.multipliers[k];
      after=reconstruct();strict=after<=.9*target&&admit();
    }
    if(st.live_display_level>0)std::cout<<"[TERMINAL-MULTIPLIER-SVD] accepted="<<strict
      <<" iterations="<<dense.sweeps<<" dual="<<after<<" seconds="<<dense.seconds
      <<" near_certificate="<<near_certificate
      <<" x_bitwise_unchanged=1 original_KKT_audit_required=1"<<std::endl;
    return strict;
  };
  if(!near_certificate)accepted=dense_fit();
  // Near-dependent original normals can need many inexpensive coordinate
  // sweeps. The caller's wall-clock allowance is authoritative; retain only a
  // generous finite safety guard, rather than discarding an improving fit
  // after a few milliseconds with most of its certified budget unused.
  for(;!accepted&&near_certificate&&sweeps<100000&&elapsed()<budget_seconds;++sweeps) {
    for(auto& c:columns) {
      L product=0;for(auto [j,a]:c.entries)product+=L(a)*residual[j];
      double next=c.u-double(product/c.norm);
      if(c.kind)next=std::min(c.cap,std::max(0.,next));
      const double delta=next-c.u;c.u=next;
      for(auto [j,a]:c.entries)residual[j]+=L(delta)*a;
    }
    if(sweeps%8!=0)continue;
    after=reconstruct();
    if(!(after<=.9*target))continue;
    if(admit()){accepted=true;break;}
  }
  if(!accepted&&!dense_attempted&&elapsed()<budget_seconds)accepted=dense_fit();
  if(st.live_display_level>0)std::cout<<"[TERMINAL-MULTIPLIER-FIT] accepted="<<accepted
    <<" sweeps="<<sweeps+1<<" dual="<<before<<"->"<<after
    <<" seconds="<<elapsed()<<" x_bitwise_unchanged=1 original_KKT_audit_required=1"<<std::endl;
  return accepted;
}

} // namespace lazyipdca::qcqp::detail
