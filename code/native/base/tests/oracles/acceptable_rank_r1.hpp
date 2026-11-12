// Frozen RANK_START_R1 certificate oracle. Only include paths, namespace and
// rank dispatch are adapted; formulas and accumulation code are unchanged.
#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include "lazyipdca_qcqp/ipopt_termination.hpp"
#include "lazyipdca_qcqp/ipopt_stop.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include <stdexcept>
namespace lazyipdca::qcqp::rank_r1_oracle {
using namespace lazyipdca::qcqp::detail;
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
// Original functions are accumulated in extended precision. Scaling and the
// canonical original slacks are exported by the Workspace, not guessed from
// the residual magnitudes. No direction or factorization is changed here.
inline bool audit_original_termination(const QCQPProblem& p,const Settings& s,Result& r,
  const std::vector<std::vector<int>>* support=nullptr,bool reconstruct_fixed=true,
  OriginalSignedResidual* signed_residual=nullptr){
  if(signed_residual){*signed_residual={};}
  const bool native = s.termination_test=="ipopt";
  r.termination_test = native ? "ipopt-3.14.11" : "project";
  r.acceptable_point=false;r.termination_audit_valid=false;r.acceptable_convergence=false;r.ipopt_desired_convergence=false;
  r.raw_kkt_error=r.acceptable_error=r.scaled_nlp_error=r.dual_infeasibility_inf=r.complementarity_inf=INFINITY;
  r.termination_internal_dual_inf=r.termination_internal_primal_inf=r.termination_internal_compl_inf=INFINITY;
  r.termination_dual_inf=r.termination_compl_inf=r.original_normalized_nlp_error=INFINITY;
  auto& eq=r.certificate_equality_multipliers;auto& in=r.certificate_inequality_multipliers;
  auto& lo=r.certificate_lower_multipliers;auto& up=r.certificate_upper_multipliers;
  auto all_finite=[](const auto& a){for(double v:a)if(!std::isfinite(v))return false;return true;};
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
  using L=long double;
  const L sf=native?r.termination_objective_scale:1;
  std::vector<L> g(p.n,0);L comp=0,eqerr=0,inerr=0,berr=0;
  if(signed_residual){signed_residual->dual.resize(p.n);signed_residual->equality.resize(eq.size());signed_residual->inequality.resize(in.size());}
  L scaled_primal=0,canonical_comp=0,scaled_y_l1=0,scaled_z_l1=0;
  auto evaluate=[&](const QuadraticFunction& q,L weight,std::size_t id){
    L value=q.constant;
    auto term=[&](int j,double a){value+=L(a)*r.x[j];g[j]+=weight*L(a);};
    if(q.has_compact_linear())for(auto a:q.compact_linear)term(a.first,a.second);
    else if(support && id<support->size())for(int j:(*support)[id])term(j,q.linear[j]);
    else for(int j=0;j<p.n;++j)if(q.linear[j]!=0)term(j,q.linear[j]);
    for(auto a:q.hessian.entries){
      value+=(a.i==a.j?L(.5):L(1))*L(a.v)*r.x[a.i]*r.x[a.j];
      g[a.i]+=weight*L(a.v)*r.x[a.j];if(a.i!=a.j)g[a.j]+=weight*L(a.v)*r.x[a.i];
    }
    return value;
  };
  const L objective=evaluate(p.objective,1,0);if(!std::isfinite(objective))return false;
  L y_l1=0,z_l1=0;
  for(std::size_t e=0;e<eq.size();++e){
    const L v=evaluate(p.equalities[e],eq[e],1+e);if(!std::isfinite(v))return false;
    if(signed_residual)signed_residual->equality[e]=double(v);
    eqerr=std::max(eqerr,std::abs(v));y_l1+=std::abs(L(eq[e]));
    const L sc=native?r.termination_equality_scales[e]:1;
    scaled_primal=std::max(scaled_primal,std::abs(sc*v));
    scaled_y_l1+=std::abs(sf*L(eq[e])/sc);
  }
  for(std::size_t i=0;i<in.size();++i){
    const L v=evaluate(p.inequalities[i],in[i],1+eq.size()+i);if(!std::isfinite(v))return false;
    if(signed_residual)signed_residual->inequality[i]=double(v);
    inerr=std::max(inerr,v);comp=std::max(comp,std::abs(L(in[i])*v));
    y_l1+=std::abs(L(in[i]));z_l1+=std::abs(L(in[i]));
    const L sc=native?r.termination_inequality_scales[i]:1;
    scaled_primal=std::max(scaled_primal,sc*v);
    const L multiplier=std::abs(sf*L(in[i])/sc);
    // Native g multiplier is BOTH y_d and v_U; d-stationarity is eliminated.
    scaled_y_l1+=multiplier;scaled_z_l1+=multiplier;
    canonical_comp=std::max(canonical_comp,native?std::abs(L(in[i])*r.termination_inequality_slacks[i]):std::abs(L(in[i])*v));
  }
  L dual=0,free_dual=0;
  int finite_lower_nonfixed=0,finite_upper_nonfixed=0,free_variables=0;
  r.fixed_multipliers_reconstructed=0;
  for(int j=0;j<p.n;++j){
    if(!std::isfinite(g[j]))return false;
    const bool exact_fixed=std::isfinite(p.lb[j])&&p.lb[j]==p.ub[j];
    if(reconstruct_fixed && exact_fixed){
      lo[j]=double(std::max(L(0),g[j]));up[j]=double(std::max(L(0),-g[j]));++r.fixed_multipliers_reconstructed;
    }
    if(!std::isfinite(lo[j])||!std::isfinite(up[j]))return false;
    const bool fixed=native?exact_fixed:(std::isfinite(p.lb[j])&&std::isfinite(p.ub[j])&&std::abs(p.ub[j]-p.lb[j])<=s.fixed_bound_tolerance);
    if(!fixed)++free_variables;
    if(std::isfinite(p.lb[j])){
      berr=std::max(berr,L(p.lb[j])-r.x[j]);
      const L c=std::abs(L(lo[j])*(L(r.x[j])-p.lb[j]));comp=std::max(comp,c);
      if(!fixed){z_l1+=lo[j];scaled_z_l1+=sf*L(lo[j]);++finite_lower_nonfixed;canonical_comp=std::max(canonical_comp,c);}
    } else if(lo[j]!=0)return false;
    if(std::isfinite(p.ub[j])){
      berr=std::max(berr,L(r.x[j])-p.ub[j]);
      const L c=std::abs(L(up[j])*(L(p.ub[j])-r.x[j]));comp=std::max(comp,c);
      if(!fixed){z_l1+=up[j];scaled_z_l1+=sf*L(up[j]);++finite_upper_nonfixed;canonical_comp=std::max(canonical_comp,c);}
    } else if(up[j]!=0)return false;
    const L signed_dual=g[j]-L(lo[j])+L(up[j]);
    if(signed_residual)signed_residual->dual[j]=double(signed_dual);
    const L stationarity=std::abs(signed_dual);
    dual=std::max(dual,stationarity);if(!fixed)free_dual=std::max(free_dual,stationarity);
  }
  const L primal=std::max({eqerr,inerr,berr});scaled_primal=std::max(scaled_primal,berr);
  const int yc=static_cast<int>(eq.size()),yd=static_cast<int>(in.size());
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
  if(signed_residual)signed_residual->valid=true;
  r.termination_audit_valid=true;r.ipopt_desired_convergence=metrics.converged;r.acceptable_point=acceptable;
  return r.ipopt_desired_convergence;
}
}
