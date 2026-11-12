#include "lazyipdca_qcqp/inner_sqp.hpp"
#include "lazyipdca_qcqp/inner_lazyqp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

namespace lazyipdca::qcqp {
namespace {

struct FilterPair { double theta=0.0; double phi=0.0; };

double row_dot(const SparseRow& row, const Vec& x) {
  double s=0.0;
  for (const auto& [j,a] : row.entries) s += a*x[j];
  return s;
}

double violation(const ConvexSubproblem& p, const Vec& y,
                 double* eq_out=nullptr, double* ineq_out=nullptr) {
  double eq=0.0, ineq=0.0;
  for (int r=0;r<(int)p.equalities.rows.size();++r)
    eq=std::max(eq,std::abs(row_dot(p.equalities.rows[r],y)-p.equalities.rhs[r]));
  for (const auto& q:p.inequalities)
    ineq=std::max(ineq,std::max(0.0,q.value(y)));
  if(eq_out) *eq_out=eq;
  if(ineq_out) *ineq_out=ineq;
  return std::max(eq,ineq);
}

void add_scaled_model_hessian(ConvexQuadraticModel& dst,
                              const ConvexQuadraticModel& src,
                              double coeff) {
  if (!(coeff>0.0)) return;
  if(dst.base_hessian.n==0) dst.base_hessian.n=src.base_hessian.n;
  for(const auto& e:src.base_hessian.entries)
    dst.base_hessian.entries.push_back({e.i,e.j,coeff*e.v});
  dst.shift_dimension=std::max(dst.shift_dimension,src.shift_dimension);
  dst.scalar_shift += coeff*src.scalar_shift;
  for(const auto& t:src.low_rank) {
    LowRankTerm u=t;
    u.beta*=coeff;
    dst.low_rank.push_back(std::move(u));
  }
}

ConvexSubproblem sqp_qp(const ConvexSubproblem& p, const Vec& y,
                        const Vec& mu, double trust_radius) {
  ConvexSubproblem q;
  q.nvar=p.nvar;
  q.original_dimension=p.original_dimension;
  q.objective.constant=p.objective.value(y);
  q.objective.linear=p.objective.gradient(y);
  q.objective.base_hessian.n=p.nvar;
  add_scaled_model_hessian(q.objective,p.objective,1.0);
  for(int i=0;i<(int)p.inequalities.size() && i<(int)mu.size();++i)
    add_scaled_model_hessian(q.objective,p.inequalities[i],std::max(0.0,mu[i]));

  q.equalities.nvar=p.nvar;
  for(int r=0;r<(int)p.equalities.rows.size();++r) {
    q.equalities.rows.push_back(p.equalities.rows[r]);
    q.equalities.rhs.push_back(p.equalities.rhs[r]-row_dot(p.equalities.rows[r],y));
  }
  for(const auto& c:p.inequalities) {
    ConvexQuadraticModel lin;
    lin.constant=c.value(y);
    lin.linear=c.gradient(y);
    lin.base_hessian.n=p.nvar;
    lin.shift_dimension=p.nvar;
    q.inequalities.push_back(std::move(lin));
  }
  if(std::isfinite(trust_radius)) {
    ConvexQuadraticModel tr;
    tr.constant=-0.5*trust_radius*trust_radius;
    tr.linear.assign(p.nvar,0.0);
    tr.base_hessian.n=p.nvar;
    for(int j=0;j<p.nvar;++j) tr.base_hessian.entries.push_back({j,j,1.0});
    tr.shift_dimension=p.nvar;
    q.inequalities.push_back(std::move(tr));
  }
  return q;
}

bool filter_ok(const std::vector<FilterPair>& filter,double theta,double phi,
               const Settings& st) {
  for(const auto& e:filter) {
    const bool tgood=theta <= (1.0-st.inner_sqp_filter_gamma_theta)*e.theta;
    const bool fgood=phi <= e.phi-st.inner_sqp_filter_gamma_f*std::max(1.0,e.theta);
    if(!tgood && !fgood) return false;
  }
  return true;
}

void filter_add(std::vector<FilterPair>& filter,double theta,double phi) {
  filter.erase(std::remove_if(filter.begin(),filter.end(),[&](const FilterPair& e){
    return theta<=e.theta && phi<=e.phi;
  }),filter.end());
  filter.push_back({theta,phi});
  if(filter.size()>64) filter.erase(filter.begin());
}

void accumulate(InnerResult& out,const InnerResult& step) {
  out.factorizations += step.factorizations;
  out.symbolic_analyses += step.symbolic_analyses;
  out.symbolic_reuses += step.symbolic_reuses;
  out.symbolic_invalidations += step.symbolic_invalidations;
  out.numeric_scatter_builds += step.numeric_scatter_builds;
  out.numeric_scatter_reuses += step.numeric_scatter_reuses;
  out.numeric_scatter_invalidations += step.numeric_scatter_invalidations;
  out.lazyqp_inexact_forcing_model_evaluations +=
      step.lazyqp_inexact_forcing_model_evaluations;
  out.lazyqp_model_support_fastpath_requests +=
      step.lazyqp_model_support_fastpath_requests;
  out.lazyqp_model_support_fastpath_effective_subproblems +=
      step.lazyqp_model_support_fastpath_effective_subproblems;
  out.lazyqp_model_support_builds += step.lazyqp_model_support_builds;
  out.lazyqp_model_support_eligible_models +=
      step.lazyqp_model_support_eligible_models;
  out.lazyqp_model_support_generic_fallback_models +=
      step.lazyqp_model_support_generic_fallback_models;
  out.lazyqp_model_support_value_evaluations +=
      step.lazyqp_model_support_value_evaluations;
  out.lazyqp_model_support_gradient_evaluations +=
      step.lazyqp_model_support_gradient_evaluations;
  out.lazyqp_kkt_triplet_layout_builds +=
      step.lazyqp_kkt_triplet_layout_builds;
  out.lazyqp_kkt_triplet_layout_reuses +=
      step.lazyqp_kkt_triplet_layout_reuses;
  out.lazyqp_kkt_triplet_layout_invalidations +=
      step.lazyqp_kkt_triplet_layout_invalidations;
  out.max_kkt_rho_x=std::max(out.max_kkt_rho_x,step.max_kkt_rho_x);
  out.max_kkt_delta_d=std::max(out.max_kkt_delta_d,step.max_kkt_delta_d);
  out.max_kkt_backward_error=std::max(out.max_kkt_backward_error,step.max_kkt_backward_error);
  out.max_kkt_residual_inf=std::max(out.max_kkt_residual_inf,step.max_kkt_residual_inf);
  out.kkt_regularization_retries += step.kkt_regularization_retries;
  out.kkt_refinement_steps += step.kkt_refinement_steps;
  out.equality_qp_original_kkt_gmres_attempts +=
      step.equality_qp_original_kkt_gmres_attempts;
  out.equality_qp_original_kkt_gmres_acceptances +=
      step.equality_qp_original_kkt_gmres_acceptances;
  out.equality_qp_original_kkt_gmres_iterations +=
      step.equality_qp_original_kkt_gmres_iterations;
  out.final_kkt_rho_x=step.final_kkt_rho_x;
  out.final_kkt_delta_d=step.final_kkt_delta_d;
}

void final_diagnostics(const ConvexSubproblem& p,InnerResult& out,const Vec& y,
                       const Vec& lambda,const Vec& mu) {
  out.y=y;
  out.equality_duals=lambda;
  out.inequality_duals=mu;
  double eq=0.0,ineq=0.0;
  violation(p,y,&eq,&ineq);
  out.equality_inf=eq;
  out.inequality_inf=ineq;
  out.primal_residual_inf=std::max(eq,ineq);
  Vec grad=p.objective.gradient(y);
  for(int r=0;r<(int)p.equalities.rows.size() && r<(int)lambda.size();++r)
    for(const auto& [j,a]:p.equalities.rows[r].entries) grad[j]+=lambda[r]*a;
  double comp=0.0;
  for(int i=0;i<(int)p.inequalities.size() && i<(int)mu.size();++i) {
    Vec g=p.inequalities[i].gradient(y);
    axpy(std::max(0.0,mu[i]),g,grad);
    comp=std::max(comp,std::abs(std::max(0.0,mu[i])*p.inequalities[i].value(y)));
  }
  out.dual_inf=inf_norm(grad);
  out.complementarity=comp;
}

} // namespace

InnerResult ConvexQCQPFilterSQP::solve(const ConvexSubproblem& p) const {
  InnerResult out;
  if(p.nvar<=0) { out.message="invalid convex subproblem"; return out; }
  Vec y(p.nvar,0.0), lambda(p.equalities.rows.size(),0.0), mu(p.inequalities.size(),0.0);
  std::vector<FilterPair> filter;
  filter_add(filter,violation(p,y),p.objective.value(y));

  Settings qp_st=st_;
  qp_st.inner_dual_warm_start=false;
  qp_st.inner_inequality_dual_warm_start=false;
  qp_st.max_inner_iterations=std::max(30,std::min(st_.max_inner_iterations,250));
  ConvexQCQPLazyQPKernel qp_solver(qp_st);

  for(int k=0;k<st_.max_inner_iterations;++k) {
    out.iterations=k+1;
    const double theta=violation(p,y);
    ConvexSubproblem q=sqp_qp(p,y,mu,std::numeric_limits<double>::infinity());
    InnerResult step=qp_solver.solve(q);
    accumulate(out,step);
    if(!step.success || (int)step.y.size()!=p.nvar) {
      out.message="filter-SQP QP step failed: "+step.message;
      final_diagnostics(p,out,y,lambda,mu);
      return out;
    }
    Vec d=step.y;
    const double dnorm=inf_norm(d);
    double alpha=1.0;
    bool accepted=false;
    Vec trial=y;
    for(int ls=0;ls<st_.inner_sqp_line_search_max_steps;++ls) {
      trial=y;
      axpy(alpha,d,trial);
      const double ttheta=violation(p,trial);
      const double tphi=p.objective.value(trial);
      const bool progress=(ttheta <= (1.0-st_.inner_sqp_filter_gamma_theta)*theta) ||
          (tphi <= p.objective.value(y)-st_.inner_sqp_filter_gamma_f*alpha*std::max(1.0,dnorm));
      if(progress && filter_ok(filter,ttheta,tphi,st_)) { accepted=true; break; }
      alpha*=st_.inner_sqp_line_search_contraction;
    }
    if(!accepted) {
      out.message="filter-SQP line search failed";
      final_diagnostics(p,out,y,lambda,mu);
      return out;
    }
    y=trial;
    if((int)step.equality_duals.size()==(int)lambda.size()) lambda=step.equality_duals;
    if((int)step.inequality_duals.size()>=(int)mu.size())
      for(int i=0;i<(int)mu.size();++i) mu[i]=std::max(0.0,step.inequality_duals[i]);
    filter_add(filter,violation(p,y),p.objective.value(y));
    final_diagnostics(p,out,y,lambda,mu);
    if(st_.verbose)
      std::cout << "[INNER-FILTER-SQP] k="<<k<<" phi="<<p.objective.value(y)
                <<" theta="<<out.primal_residual_inf<<" dual="<<out.dual_inf
                <<" step="<<alpha*dnorm<<"\n";
    if(out.primal_residual_inf<=st_.inner_tolerance &&
       out.dual_inf<=10.0*st_.inner_tolerance &&
       out.complementarity<=std::max(10.0*st_.inner_tolerance,st_.inner_iteration_limit_mu_fallback) &&
       alpha*dnorm<=std::sqrt(std::max(st_.inner_tolerance,1e-16))) {
      out.success=true;
      out.message="convex QCQP solved by filter-SQP";
      return out;
    }
  }
  final_diagnostics(p,out,y,lambda,mu);
  out.success = out.primal_residual_inf<=10.0*st_.inner_tolerance &&
                out.dual_inf<=100.0*st_.inner_tolerance;
  out.message=out.success?"convex QCQP filter-SQP acceptable limit":"filter-SQP iteration limit";
  return out;
}

InnerResult ConvexQCQPTrustRegionSQP::solve(const ConvexSubproblem& p) const {
  InnerResult out;
  if(p.nvar<=0) { out.message="invalid convex subproblem"; return out; }
  Vec y(p.nvar,0.0), lambda(p.equalities.rows.size(),0.0), mu(p.inequalities.size(),0.0);
  double delta=std::min(st_.inner_sqp_tr_max_radius,std::max(st_.inner_sqp_tr_min_radius,st_.inner_sqp_tr_initial_radius));
  Settings qp_st=st_;
  qp_st.inner_dual_warm_start=false;
  qp_st.inner_inequality_dual_warm_start=false;
  qp_st.max_inner_iterations=std::max(30,std::min(st_.max_inner_iterations,250));
  ConvexQCQPLazyQPKernel qp_solver(qp_st);

  for(int k=0;k<st_.max_inner_iterations;++k) {
    out.iterations=k+1;
    final_diagnostics(p,out,y,lambda,mu);
    if(out.primal_residual_inf<=st_.inner_tolerance &&
       out.dual_inf<=10.0*st_.inner_tolerance &&
       out.complementarity<=std::max(10.0*st_.inner_tolerance,st_.inner_iteration_limit_mu_fallback)) {
      out.success=true;
      out.message="convex QCQP solved by TR-SQP";
      return out;
    }
    const double theta=violation(p,y);
    const double phi=p.objective.value(y);
    ConvexSubproblem q=sqp_qp(p,y,mu,delta);
    InnerResult step=qp_solver.solve(q);
    accumulate(out,step);
    if(!step.success || (int)step.y.size()!=p.nvar) {
      delta*=st_.inner_sqp_tr_shrink;
      if(delta<st_.inner_sqp_tr_min_radius) {
        out.message="TR-SQP QP step failed at minimum radius: "+step.message;
        final_diagnostics(p,out,y,lambda,mu);
        return out;
      }
      continue;
    }
    Vec d=step.y;
    Vec trial=y; axpy(1.0,d,trial);
    const double ttheta=violation(p,trial);
    const double tphi=p.objective.value(trial);
    const double sigma=10.0*(1.0+std::accumulate(mu.begin(),mu.end(),0.0));
    const double psi=phi+sigma*theta;
    const double tpsi=tphi+sigma*ttheta;
    const double qphi=q.objective.value(d);
    const double pred=std::max(1e-16,phi-qphi+sigma*theta);
    const double ared=psi-tpsi;
    const double rho=ared/pred;
    const bool accept=ared>0.0 && rho>=st_.inner_sqp_tr_eta_accept;
    if(accept) {
      y=trial;
      if((int)step.equality_duals.size()==(int)lambda.size()) lambda=step.equality_duals;
      if((int)step.inequality_duals.size()>=(int)mu.size())
        for(int i=0;i<(int)mu.size();++i) mu[i]=std::max(0.0,step.inequality_duals[i]);
      if(rho>=st_.inner_sqp_tr_eta_expand && norm2(d)>=0.8*delta)
        delta=std::min(st_.inner_sqp_tr_max_radius,st_.inner_sqp_tr_expand*delta);
    } else {
      delta=std::max(st_.inner_sqp_tr_min_radius,st_.inner_sqp_tr_shrink*delta);
    }
    final_diagnostics(p,out,y,lambda,mu);
    if(st_.verbose)
      std::cout << "[INNER-TR-SQP] k="<<k<<" phi="<<p.objective.value(y)
                <<" theta="<<out.primal_residual_inf<<" dual="<<out.dual_inf
                <<" rho="<<rho<<" Delta="<<delta<<"\n";
    if(accept && out.primal_residual_inf<=st_.inner_tolerance &&
       out.dual_inf<=10.0*st_.inner_tolerance &&
       out.complementarity<=std::max(10.0*st_.inner_tolerance,st_.inner_iteration_limit_mu_fallback) &&
       inf_norm(d)<=std::sqrt(std::max(st_.inner_tolerance,1e-16))) {
      out.success=true;
      out.message="convex QCQP solved by TR-SQP";
      return out;
    }
    if(delta<=st_.inner_sqp_tr_min_radius*(1.0+1e-12) && !accept) break;
  }
  final_diagnostics(p,out,y,lambda,mu);
  out.success = out.primal_residual_inf<=10.0*st_.inner_tolerance &&
                out.dual_inf<=100.0*st_.inner_tolerance;
  out.message=out.success?"convex QCQP TR-SQP acceptable limit":"TR-SQP iteration/radius limit";
  return out;
}

} // namespace lazyipdca::qcqp
