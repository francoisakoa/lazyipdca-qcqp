#pragma once
#include "model.hpp"
#include "result.hpp"
#include "settings.hpp"
#include "ipopt_termination.hpp"
#include "ipopt_stop.hpp"
#include "rank_presolve.hpp"
#include <stdexcept>
namespace lazyipdca::qcqp::detail {
struct AcceptableCounter {
  int last=-1,count=0;
  bool observe(int iteration,bool eligible,int required,bool include_initial=false){
    if(required<=0){count=0;last=iteration;return false;}
    if(iteration==last)return count>=required;
    if(last>=0 && iteration!=last+1)count=0;
    last=iteration;
    count=(iteration>0 || include_initial) && eligible?count+1:0;
    return count>=required;
  }
};
inline void validate_acceptable(const Settings& s){
  if(s.dca_rank_audit_mode!="full"&&s.dca_rank_audit_mode!="mapped"&&s.dca_rank_audit_mode!="check")
    throw std::invalid_argument("dca-rank-audit-mode must be full, mapped or check");
  if(s.termination_test!="project"&&s.termination_test!="ipopt")
    throw std::invalid_argument("termination-test must be project or ipopt");
  if(s.acceptable_iterations<0)throw std::invalid_argument("acceptable-iter must be nonnegative");
  for(double v:{s.acceptable_tolerance,s.acceptable_constraint_tolerance,s.acceptable_dual_tolerance,
                s.acceptable_complementarity_tolerance,s.nlp_error_scaling_max})
    if(!(v>0)||!std::isfinite(v))throw std::invalid_argument("acceptable tolerances and s_max must be finite and positive");
  if(!std::isfinite(s.acceptable_objective_change_tolerance)||s.acceptable_objective_change_tolerance<0)
    throw std::invalid_argument("acceptable-obj-change-tol must be finite and nonnegative");
}
// Signed ORIGINAL-coordinate residual, accumulated in the same order and
// precision as the certificate. It is only populated on a complete valid audit.
struct OriginalSignedResidual {
  std::vector<double> dual, equality, inequality;
  bool valid=false;
};
// The scalar certificate is shared by the direct and mapped audit paths.
// Mapped access changes neither the original row dimensions nor normalizations.
// Evaluate an original polynomial in its original summation order. Access may
// substitute coordinates, but it must never coalesce/reorder coefficients here.
template<class X>
inline long double original_audit_evaluate(const QuadraticFunction& q,const X& x,
    long double weight,std::vector<long double>& g,
    const std::vector<std::vector<int>>* support,std::size_t id) {
  using L=long double;
  L value=q.constant;
  auto term=[&](int j,double a){value+=L(a)*x(j);g[j]+=weight*L(a);};
  if(q.has_compact_linear())for(auto a:q.compact_linear)term(a.first,a.second);
  else if(support && id<support->size())for(int j:(*support)[id])term(j,q.linear[j]);
  else for(int j=0;j<(int)q.linear.size();++j)if(q.linear[j]!=0)term(j,q.linear[j]);
  for(auto a:q.hessian.entries){
    value+=(a.i==a.j?L(.5):L(1))*L(a.v)*x(a.i)*x(a.j);
    g[a.i]+=weight*L(a.v)*x(a.j);if(a.i!=a.j)g[a.j]+=weight*L(a.v)*x(a.i);
  }
  for(auto e:q.signed_square){const L a=x(e.first);value+=L(e.second)*a*std::abs(a);g[e.first]+=weight*2*L(e.second)*std::abs(a);}
  return value;
}
struct DirectOriginalAuditView {
  const QCQPProblem& p;
  Result& r;
  bool validate(bool native)const {
    auto all_finite=[](const auto& a){for(double v:a)if(!std::isfinite(v))return false;return true;};
    const auto& eq=r.certificate_equality_multipliers;const auto& in=r.certificate_inequality_multipliers;
    const auto& lo=r.certificate_lower_multipliers;const auto& up=r.certificate_upper_multipliers;
    if(r.x.size()!=std::size_t(p.n)||eq.size()!=p.equalities.size()||in.size()!=p.inequalities.size()||
      lo.size()!=r.x.size()||up.size()!=r.x.size()||!all_finite(r.x)||!all_finite(eq)||!all_finite(in)||!all_finite(lo)||!all_finite(up))return false;
    for(const auto* a:{&in,&lo,&up})for(double v:*a)if(v<0)return false;
    if(native){
      if(!r.termination_scaling_available || !(r.termination_objective_scale>0) || !std::isfinite(r.termination_objective_scale) ||
         r.termination_equality_scales.size()!=eq.size() || r.termination_inequality_scales.size()!=in.size() ||
         r.termination_inequality_slacks.size()!=in.size())return false;
      for(const auto* scales:{&r.termination_equality_scales,&r.termination_inequality_scales})
        for(double v:*scales)if(!(v>0)||!std::isfinite(v))return false;
      for(double v:r.termination_inequality_slacks)if(v<0||!std::isfinite(v))return false;
    }
    return true;
  }
  double x(int j)const{return r.x[j];}
  double equality(std::size_t e)const{return r.certificate_equality_multipliers[e];}
  double inequality(std::size_t i)const{return r.certificate_inequality_multipliers[i];}
  double lower(int j)const{return r.certificate_lower_multipliers[j];}
  double upper(int j)const{return r.certificate_upper_multipliers[j];}
  double equality_scale(std::size_t e)const{return r.termination_equality_scales[e];}
  double inequality_scale(std::size_t i)const{return r.termination_inequality_scales[i];}
  double slack(std::size_t i)const{return r.termination_inequality_slacks[i];}
  void set_fixed(int j,double lo,double up){r.certificate_lower_multipliers[j]=lo;r.certificate_upper_multipliers[j]=up;}
  long double evaluate(const QuadraticFunction& q,long double w,std::size_t id,
      std::vector<long double>& g,const std::vector<std::vector<int>>* support)const{
    return original_audit_evaluate(q,[&](int j){return x(j);},w,g,support,id);
  }
  void begin_signed(OriginalSignedResidual& v)const {v.dual.resize(p.n);v.equality.resize(p.equalities.size());v.inequality.resize(p.inequalities.size());}
  void write_equality(OriginalSignedResidual& v,std::size_t i,double x)const{v.equality[i]=x;}
  void write_inequality(OriginalSignedResidual& v,std::size_t i,double x)const{v.inequality[i]=x;}
  void write_dual(OriginalSignedResidual& v,int i,double x)const{v.dual[i]=x;}
  void end_signed(OriginalSignedResidual& v)const{v.valid=true;}
};

template<class AuditView>
inline bool audit_original_termination_impl(const QCQPProblem& p,const Settings& s,Result& r,
    AuditView& state,const std::vector<std::vector<int>>* support,bool reconstruct_fixed,
    OriginalSignedResidual* signed_residual){
  if(signed_residual){*signed_residual={};}
  const bool native = s.termination_test=="ipopt";
  r.termination_test = native ? "ipopt-3.14.11" : "project";
  r.acceptable_point=false;r.termination_audit_valid=false;r.acceptable_convergence=false;r.ipopt_desired_convergence=false;
  r.raw_kkt_error=r.acceptable_error=r.scaled_nlp_error=r.dual_infeasibility_inf=r.complementarity_inf=INFINITY;
  r.termination_internal_dual_inf=r.termination_internal_primal_inf=r.termination_internal_compl_inf=INFINITY;
  r.termination_dual_inf=r.termination_compl_inf=r.original_normalized_nlp_error=INFINITY;
  if(!state.validate(native))return false;
  using L=long double;
  const L sf=native?r.termination_objective_scale:1;
  std::vector<L> g(p.n,0);L comp=0,eqerr=0,inerr=0,berr=0;
  if(signed_residual)state.begin_signed(*signed_residual);
  L scaled_primal=0,canonical_comp=0,scaled_y_l1=0,scaled_z_l1=0;
  auto evaluate=[&](const QuadraticFunction& q,L weight,std::size_t id){
    return state.evaluate(q,weight,id,g,support);
  };
  const L objective=evaluate(p.objective,1,0);if(!std::isfinite(objective))return false;
  L y_l1=0,z_l1=0;
  for(std::size_t e=0;e<p.equalities.size();++e){
    const double multiplier_value=state.equality(e);
    const L v=evaluate(p.equalities[e],multiplier_value,1+e);if(!std::isfinite(v))return false;
    if(signed_residual)state.write_equality(*signed_residual,e,double(v));
    eqerr=std::max(eqerr,std::abs(v));y_l1+=std::abs(L(multiplier_value));
    const L sc=native?state.equality_scale(e):1;
    scaled_primal=std::max(scaled_primal,std::abs(sc*v));
    scaled_y_l1+=std::abs(sf*L(multiplier_value)/sc);
  }
  for(std::size_t i=0;i<p.inequalities.size();++i){
    const double multiplier_value=state.inequality(i);
    const L v=evaluate(p.inequalities[i],multiplier_value,1+p.equalities.size()+i);if(!std::isfinite(v))return false;
    if(signed_residual)state.write_inequality(*signed_residual,i,double(v));
    inerr=std::max(inerr,v);comp=std::max(comp,std::abs(L(multiplier_value)*v));
    y_l1+=std::abs(L(multiplier_value));z_l1+=std::abs(L(multiplier_value));
    const L sc=native?state.inequality_scale(i):1;
    scaled_primal=std::max(scaled_primal,sc*v);
    const L multiplier=std::abs(sf*L(multiplier_value)/sc);
    // Native g multiplier is BOTH y_d and v_U; d-stationarity is eliminated.
    scaled_y_l1+=multiplier;scaled_z_l1+=multiplier;
    canonical_comp=std::max(canonical_comp,native?std::abs(L(multiplier_value)*state.slack(i)):std::abs(L(multiplier_value)*v));
  }
  L dual=0,free_dual=0;
  int finite_lower_nonfixed=0,finite_upper_nonfixed=0,free_variables=0;
  r.fixed_multipliers_reconstructed=0;
  for(int j=0;j<p.n;++j){
    if(!std::isfinite(g[j]))return false;
    double lower=state.lower(j),upper=state.upper(j);
    const double point=state.x(j);
    const bool exact_fixed=std::isfinite(p.lb[j])&&p.lb[j]==p.ub[j];
    if(reconstruct_fixed && exact_fixed){
      lower=double(std::max(L(0),g[j]));upper=double(std::max(L(0),-g[j]));++r.fixed_multipliers_reconstructed;
      state.set_fixed(j,lower,upper);
    }
    if(!std::isfinite(lower)||!std::isfinite(upper))return false;
    const bool fixed=native?exact_fixed:(std::isfinite(p.lb[j])&&std::isfinite(p.ub[j])&&std::abs(p.ub[j]-p.lb[j])<=s.fixed_bound_tolerance);
    if(!fixed)++free_variables;
    if(std::isfinite(p.lb[j])){
      berr=std::max(berr,L(p.lb[j])-point);
      const L c=std::abs(L(lower)*(L(point)-p.lb[j]));comp=std::max(comp,c);
      if(!fixed){z_l1+=lower;scaled_z_l1+=sf*L(lower);++finite_lower_nonfixed;canonical_comp=std::max(canonical_comp,c);}
    } else if(lower!=0)return false;
    if(std::isfinite(p.ub[j])){
      berr=std::max(berr,L(point)-p.ub[j]);
      const L c=std::abs(L(upper)*(L(p.ub[j])-point));comp=std::max(comp,c);
      if(!fixed){z_l1+=upper;scaled_z_l1+=sf*L(upper);++finite_upper_nonfixed;canonical_comp=std::max(canonical_comp,c);}
    } else if(upper!=0)return false;
    const L signed_dual=g[j]-L(lower)+L(upper);
    if(signed_residual)state.write_dual(*signed_residual,j,double(signed_dual));
    const L stationarity=std::abs(signed_dual);
    dual=std::max(dual,stationarity);if(!fixed)free_dual=std::max(free_dual,stationarity);
  }
  const L primal=std::max({eqerr,inerr,berr});scaled_primal=std::max(scaled_primal,berr);
  const int yc=static_cast<int>(p.equalities.size()),yd=static_cast<int>(p.inequalities.size());
  const int zl=finite_lower_nonfixed,zu=finite_upper_nonfixed,vl=0,vu=yd;
  const int complementarity_dimension=zl+zu+vl+vu;
  const int dual_dimension=yc+yd+complementarity_dimension;
  auto metrics=ipopt_desired_convergence_metrics(double(dual),double(primal),double(comp),double(y_l1),double(z_l1),
      dual_dimension,complementarity_dimension,s);
  r.original_normalized_nlp_error=metrics.overall_nlp_error;
  bool acceptable=metrics.finite && metrics.overall_nlp_error<=s.acceptable_tolerance && primal<=s.acceptable_constraint_tolerance &&
    dual<=s.acceptable_dual_tolerance && comp<=s.acceptable_complementarity_tolerance;
  r.termination_dual_inf=double(dual);r.termination_compl_inf=double(comp);
  if(native){
    // Re-auditing an outer iteration must not turn objective change into zero.
    if(r.termination_objective_iteration!=r.outer_iterations){
      r.termination_previous_objective=r.termination_current_objective;
      r.termination_current_objective=double(sf*objective);
      r.termination_objective_iteration=r.outer_iterations;
      r.termination_objective_change=std::abs(r.termination_current_objective-r.termination_previous_objective)/
        std::max(1.0,std::abs(r.termination_current_objective));
    }
    IpoptStopSnapshot v;
    v.internal_dual=double(sf*free_dual);v.internal_primal=double(scaled_primal);v.internal_compl=double(sf*canonical_comp);
    v.unscaled_dual=double(free_dual);v.unscaled_primal=double(primal);v.unscaled_compl=double(canonical_comp);
    v.y_l1=double(scaled_y_l1);v.z_l1=double(scaled_z_l1);v.dual_dimension=dual_dimension;v.compl_dimension=complementarity_dimension;
    v.free_variables=free_variables;v.equality_dimension=yc;v.objective_change=r.termination_objective_change;
    const auto d=ipopt_stop_decision(v,s);metrics=d.metrics;acceptable=d.acceptable;
    r.termination_square_problem=d.square;
    r.termination_internal_dual_inf=v.internal_dual;r.termination_internal_primal_inf=v.internal_primal;r.termination_internal_compl_inf=v.internal_compl;
    r.termination_dual_inf=v.unscaled_dual;r.termination_compl_inf=v.unscaled_compl;
  }
  const L raw=std::max({primal,dual,comp});if(!std::isfinite(raw)||!metrics.finite)return false;
  r.objective=double(objective);r.feasibility={double(eqerr),double(inerr),double(berr),double(primal)};
  r.dual_infeasibility_inf=double(dual);r.complementarity_inf=double(comp);r.model_stationarity_inf=double(dual);
  r.raw_kkt_error=double(raw);r.acceptable_error=r.scaled_nlp_error=metrics.overall_nlp_error;
  r.termination_sd=r.dual_error_scaling=metrics.dual_scaling;r.termination_sc=r.complementarity_error_scaling=metrics.complementarity_scaling;
  r.constraint_multiplier_l1=metrics.y_l1;r.bound_slack_multiplier_l1=metrics.z_l1;
  r.ipopt_dual_multiplier_dimension=dual_dimension;r.ipopt_complementarity_multiplier_dimension=complementarity_dimension;
  r.ipopt_y_c_dimension=yc;r.ipopt_y_d_dimension=yd;r.ipopt_z_L_dimension=zl;r.ipopt_z_U_dimension=zu;r.ipopt_v_L_dimension=vl;r.ipopt_v_U_dimension=vu;
  if(signed_residual)state.end_signed(*signed_residual);
  r.termination_audit_valid=true;r.ipopt_desired_convergence=metrics.converged;r.acceptable_point=acceptable;
  return r.ipopt_desired_convergence;
}
// Public dispatch. The final postsolve and independent audit always use the
// direct original-input path (rank_presolve_context=nullptr).
inline bool audit_original_termination(const QCQPProblem& p,const Settings& s,Result& r,
  const std::vector<std::vector<int>>* support=nullptr,bool reconstruct_fixed=true,
  OriginalSignedResidual* signed_residual=nullptr){
  if(s.rank_presolve_context && &p==&s.rank_presolve_context->reduced())
    return rank_presolve_audit(s.rank_presolve_context,p,s,r,signed_residual);
  DirectOriginalAuditView view{p,r};
  return audit_original_termination_impl(p,s,r,view,support,reconstruct_fixed,signed_residual);
}
}
