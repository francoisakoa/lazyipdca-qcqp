#pragma once
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <chrono>
#include <iostream>

namespace lazyipdca::qcqp::detail {
inline double exact_power2_scale(double value,int exponent){
  if(!std::isfinite(value))throw std::runtime_error("power2 scaling: nonfinite finite datum");
  const double result=std::scalbn(value,exponent);
  if(!std::isfinite(result)||std::scalbn(result,-exponent)!=value||
      (value==0&&std::signbit(value)!=std::signbit(result)))
    throw std::runtime_error("power2 scaling: dyadic datum not exactly representable");
  return result;
}
inline int dyadic_ceil_exponent(double value){
  if(value==0)return -100000;
  int exponent=0;const double fraction=std::frexp(std::abs(value),&exponent);
  return fraction==.5?exponent-1:exponent;
}
class Power2Scaling {
 public:
  const QCQPProblem& original;QCQPProblem scaled;
  std::vector<int> variable_exp,equality_exp,inequality_exp;
  int objective_exp=0;
  explicit Power2Scaling(const QCQPProblem& p):original(p){
    p.validate();if(p.has_signed_square())throw std::runtime_error("power2 scaling: signed-square unsupported");
    variable_exp.assign(p.n,0);
    for(int j=0;j<p.n;++j){
      if(!std::isfinite(p.x0[j]))throw std::runtime_error("power2 scaling: nonfinite start");
      long double size=std::max(1.L,std::abs(static_cast<long double>(p.x0[j])));
      if(std::isfinite(p.lb[j])&&std::isfinite(p.ub[j]))size=std::max(size,static_cast<long double>(p.ub[j])-p.lb[j]);
      int exponent=0;const long double fraction=std::frexp(size,&exponent);variable_exp[j]=fraction==.5L?exponent-1:exponent;
      if(variable_exp[j]<0||variable_exp[j]>1023)throw std::runtime_error("power2 scaling: variable scale out of range");
    }
    auto row_exp=[&](const QuadraticFunction& q,bool objective){
      int maximum=objective?-100000:dyadic_ceil_exponent(q.constant);
      q.for_each_linear_nonzero([&](int j,double a){if(a)maximum=std::max(maximum,dyadic_ceil_exponent(a)+variable_exp[j]);});
      for(auto e:q.hessian.entries)if(e.v)maximum=std::max(maximum,dyadic_ceil_exponent(e.v)+variable_exp[e.i]+variable_exp[e.j]);
      // Target coefficient magnitude <=64. Only downscale row/objective units.
      return std::min(0,6-maximum);
    };
    objective_exp=row_exp(p.objective,true);for(const auto& q:p.equalities)equality_exp.push_back(row_exp(q,false));for(const auto& q:p.inequalities)inequality_exp.push_back(row_exp(q,false));
    scaled.name=p.name;scaled.n=p.n;
    for(int j=0;j<p.n;++j){scaled.x0.push_back(exact_power2_scale(p.x0[j],-variable_exp[j]));scaled.lb.push_back(std::isfinite(p.lb[j])?exact_power2_scale(p.lb[j],-variable_exp[j]):p.lb[j]);scaled.ub.push_back(std::isfinite(p.ub[j])?exact_power2_scale(p.ub[j],-variable_exp[j]):p.ub[j]);}
    auto rewrite=[&](const QuadraticFunction& q,int exponent){
      QuadraticFunction out;out.name=q.name;out.constant=exact_power2_scale(q.constant,exponent);out.compact_linear_dimension=p.n;out.hessian.n=p.n;
      q.for_each_linear_nonzero([&](int j,double a){if(a)out.compact_linear.push_back({j,exact_power2_scale(a,exponent+variable_exp[j])});});
      for(auto e:q.hessian.entries)if(e.v)out.hessian.entries.push_back({e.i,e.j,exact_power2_scale(e.v,exponent+variable_exp[e.i]+variable_exp[e.j])});return out;
    };
    scaled.objective=rewrite(p.objective,objective_exp);for(int i=0;i<int(p.equalities.size());++i)scaled.equalities.push_back(rewrite(p.equalities[i],equality_exp[i]));for(int i=0;i<int(p.inequalities.size());++i)scaled.inequalities.push_back(rewrite(p.inequalities[i],inequality_exp[i]));scaled.validate();
  }
  void lift(Result& r)const{
    if(r.x.size()!=std::size_t(original.n)||r.certificate_lower_multipliers.size()!=r.x.size()||r.certificate_upper_multipliers.size()!=r.x.size()||r.certificate_equality_multipliers.size()!=equality_exp.size()||r.certificate_inequality_multipliers.size()!=inequality_exp.size())throw std::runtime_error("power2 scaling: incomplete certificate");
    for(int j=0;j<original.n;++j){r.x[j]=exact_power2_scale(r.x[j],variable_exp[j]);r.certificate_lower_multipliers[j]=exact_power2_scale(r.certificate_lower_multipliers[j],-objective_exp-variable_exp[j]);r.certificate_upper_multipliers[j]=exact_power2_scale(r.certificate_upper_multipliers[j],-objective_exp-variable_exp[j]);}
    for(int i=0;i<int(equality_exp.size());++i)r.certificate_equality_multipliers[i]=exact_power2_scale(r.certificate_equality_multipliers[i],equality_exp[i]-objective_exp);
    for(int i=0;i<int(inequality_exp.size());++i)r.certificate_inequality_multipliers[i]=exact_power2_scale(r.certificate_inequality_multipliers[i],inequality_exp[i]-objective_exp);
    r.termination_scaling_available=false;r.termination_equality_scales.clear();r.termination_inequality_scales.clear();r.termination_inequality_slacks.clear();
  }
};
inline Result solve_with_power2_scaling(const QCQPProblem& p,const Settings& st){
  using Clock=std::chrono::steady_clock;const auto began=Clock::now();auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-began).count();};
  Power2Scaling scaling(p);Settings inner=st;inner.power2_scaling=false;
  int smallest_row=0,largest_variable=0,smallest_variable=1024;for(int e:scaling.equality_exp)smallest_row=std::min(smallest_row,e);for(int e:scaling.inequality_exp)smallest_row=std::min(smallest_row,e);for(int e:scaling.variable_exp){largest_variable=std::max(largest_variable,e);smallest_variable=std::min(smallest_variable,e);}
  // Convert tolerances toward the ORIGINAL physical units, never relax them.
  inner.feasibility_tolerance=exact_power2_scale(std::min(1e-6,st.feasibility_tolerance),std::min(smallest_row,-largest_variable));
  inner.dual_infeasibility_tolerance=exact_power2_scale(std::min(1e-6,st.dual_infeasibility_tolerance),scaling.objective_exp+smallest_variable);
  inner.complementarity_tolerance=exact_power2_scale(std::min(1e-6,st.complementarity_tolerance),scaling.objective_exp);
  inner.stationarity_tolerance=std::min({st.stationarity_tolerance,inner.feasibility_tolerance,inner.dual_infeasibility_tolerance,inner.complementarity_tolerance});
  inner.dca_control_tolerance=std::min(st.dca_control_tolerance,inner.stationarity_tolerance);
  inner.acceptable_iterations=0;inner.acceptable_on_failure=false;
  const double preparation=elapsed();
  if(st.live_display_level)std::cout<<"[POWER2-SCALING] variables="<<p.n<<" D_exp="<<smallest_variable<<":"<<largest_variable<<" row_min_exp="<<smallest_row<<" F_exp="<<scaling.objective_exp<<" physical_start_unchanged=1 exact_dyadic_data=1 inner_feas_tol="<<inner.feasibility_tolerance<<" inner_dual_tol="<<inner.dual_infeasibility_tolerance<<" inner_compl_tol="<<inner.complementarity_tolerance<<std::endl;
  if(std::isfinite(st.time_limit_seconds))inner.time_limit_seconds=std::max(0.,st.time_limit_seconds-elapsed());
  Result r;
  if(inner.time_limit_seconds>0)r=Solver(inner).solve(scaling.scaled);
  else {r.status=Status::TimeLimit;r.message="power2 scaling: preparation deadline";r.x=scaling.scaled.x0;r.certificate_equality_multipliers.assign(p.equalities.size(),0);r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);}
  const auto post=Clock::now();
  try {scaling.lift(r);Settings original=st;original.rank_presolve_context=nullptr;original.power2_scaling=false;const bool audited=audit_original_termination(p,original,r);const bool strict=audited&&r.feasibility.total_inf<=std::min(1e-6,st.feasibility_tolerance)&&r.dual_infeasibility_inf<=std::min(1e-6,st.dual_infeasibility_tolerance)&&r.complementarity_inf<=std::min(1e-6,st.complementarity_tolerance);
    if((r.status==Status::Solved||r.status==Status::SolvedAcceptable||r.status==Status::SolvedAcceptableOnFailure)&&!strict){r.status=Status::NumericalError;r.message="power2 scaling: ORIGINAL unscaled KKT audit failed";r.acceptable_point=false;r.acceptable_convergence=false;}else if(r.status==Status::Solved)r.message+="; exact dyadic scaling and ORIGINAL KKT audit";
    r.termination_multiplier_source="power2-original-unit-lift:"+r.termination_multiplier_source;
  }catch(const std::exception& e){r.status=Status::NumericalError;r.message=e.what();r.x=p.x0;r.certificate_equality_multipliers.assign(p.equalities.size(),0);r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);r.termination_scaling_available=false;r.termination_audit_valid=false;r.acceptable_point=false;r.acceptable_convergence=false;r.objective=p.objective.value(r.x);r.feasibility=feasibility(p,r.x);r.dual_infeasibility_inf=r.complementarity_inf=r.raw_kkt_error=INFINITY;}
  r.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=preparation;r.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=std::chrono::duration<double>(Clock::now()-post).count();r.elapsed_seconds=elapsed();r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
  if(std::isfinite(st.time_limit_seconds)&&r.elapsed_seconds>st.time_limit_seconds){r.status=Status::TimeLimit;r.message="power2 scaling: original solve/audit deadline";}
  return r;
}
} // namespace lazyipdca::qcqp::detail
