#pragma once
// Internal native PHR augmented-Lagrangian repair. Ported from the frozen
// successful research kernel; original QCQP remains the sole final KKT model.
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "product_definition_presolve.hpp"
#include "power2_scaling.hpp"
#include "native_bilinear_start.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace lazyipdca::qcqp::detail::native_augmented {
using Clock=std::chrono::steady_clock;
inline bool solve_from_pristine_rhs(SymmetricFactorization& factor,const Vec& pristine,Vec& rhs,std::string& message){
  rhs=pristine;return factor.solve(rhs,message);
}
struct SparseQuadratic {
  double c;std::vector<int> support;Vec lin;std::vector<SymTriplet> h;
  explicit SparseQuadratic(const QuadraticFunction& q):c(q.constant) {
    if(!q.signed_square.empty())throw std::runtime_error("AL: signed squares unsupported");
    std::set<int> s;q.for_each_linear_nonzero([&](int j,double a){if(a)s.insert(j);});
    for(auto e:q.hessian.entries)if(e.v){s.insert(e.i);s.insert(e.j);}
    support.assign(s.begin(),s.end());lin.assign(support.size(),0);std::map<int,int> ix;
    for(int k=0;k<(int)support.size();++k)ix[support[k]]=k;
    q.for_each_linear_nonzero([&](int j,double a){if(a)lin[ix[j]]+=a;});
    for(auto e:q.hessian.entries)if(e.v)h.push_back({ix[e.i],ix[e.j],e.v});
  }
  long double value_extended(const Vec& x,Vec* gradient=nullptr)const {
    long double v=c;if(gradient)*gradient=lin;
    for(int k=0;k<(int)support.size();++k)v+=(long double)lin[k]*x[support[k]];
    for(auto e:h){double a=x[support[e.i]],b=x[support[e.j]];v+=(e.i==e.j?.5L:1.L)*e.v*a*b;
      if(gradient){(*gradient)[e.i]+=e.v*b;if(e.i!=e.j)(*gradient)[e.j]+=e.v*a;}}
    return v;
  }
  double evaluate(const Vec& x,Vec* gradient=nullptr)const {
    return static_cast<double>(value_extended(x,gradient));
  }
  // Form q(t)-q(x) directly. The constant cancels symbolically and the
  // products use their symmetric secant form, so neither a large constant
  // nor a small displacement requires subtraction of rounded values.
  long double difference(const Vec& x,const Vec& t)const {
    long double delta=0;
    for(int k=0;k<(int)support.size();++k)
      delta+=(long double)lin[k]*((long double)t[support[k]]-x[support[k]]);
    for(auto e:h){
      const int i=support[e.i],j=support[e.j];
      const long double di=(long double)t[i]-x[i],dj=(long double)t[j]-x[j];
      if(i==j)delta+=.5L*e.v*di*((long double)t[i]+x[i]);
      else delta+=.5L*e.v*(di*((long double)t[j]+x[j])+dj*((long double)t[i]+x[i]));
    }
    return delta;
  }
};
struct Eval {long double value=0;Vec gradient,hessian,eq,iq;double projected=0;};
struct NativeAL {
  const QCQPProblem& p;SparseQuadratic obj;std::vector<SparseQuadratic> eq,iq;
  Settings factor_settings;std::unique_ptr<SymmetricFactorization> factor;int factors=0,solves=0,iterations=0;
  double shift=0;Clock::time_point began;double seconds_limit;bool box_qp=false,critical_only=false;
  NativeAL(const QCQPProblem& model,const Settings& selected,Clock::time_point global_began,double seconds,bool box=true,bool critical=true):p(model),obj(model.objective),factor_settings(selected),began(global_began),seconds_limit(seconds),box_qp(box),critical_only(critical){
    if(24.L*p.n*p.n>128.L*1024*1024)throw std::runtime_error("AL: dense private Hessian construction exceeds 128 MiB admission budget");
    for(const auto& q:p.equalities)eq.emplace_back(q);for(const auto& q:p.inequalities)iq.emplace_back(q);
    factor_settings.linear_backend="mumps";factor_settings.linear_metis=true;factor_settings.kkt_adaptive_regularization=true;
    factor_settings.kkt_primal_regularization=0;factor_settings.kkt_dual_regularization=0;
    std::string msg;factor=make_symmetric_factorization(factor_settings,msg);if(!factor)throw std::runtime_error(msg);
  }
  bool expired()const{return std::chrono::duration<double>(Clock::now()-began).count()>=seconds_limit;}
  long double merit_difference(const Vec& x,const Vec& t,const Vec& lambda,const Vec& z,double rho)const{
    if(!(rho>0)||!std::isfinite(rho))throw std::runtime_error("native AL: invalid merit-difference penalty");
    long double delta=obj.difference(x,t);
    for(int i=0;i<(int)eq.size();++i){
      const long double h0=eq[i].value_extended(x),dh=eq[i].difference(x,t);
      if(!std::isfinite(h0)||!std::isfinite(dh))throw std::runtime_error("native AL: nonfinite equality merit difference");
      delta+=((long double)lambda[i]+(long double)rho*h0)*dh+.5L*rho*dh*dh;
    }
    for(int i=0;i<(int)iq.size();++i){
      const long double g0=iq[i].value_extended(x),g1=iq[i].value_extended(t),dg=iq[i].difference(x,t);
      if(!std::isfinite(g0)||!std::isfinite(g1)||!std::isfinite(dg))throw std::runtime_error("native AL: nonfinite inequality merit difference");
      const long double a0=(long double)z[i]+(long double)rho*g0,a1=(long double)z[i]+(long double)rho*g1;
      if(a0>0&&a1>0)delta+=a0*dg+.5L*rho*dg*dg;
      else if(a1>0)delta+=.5L*a1*a1/rho;
      else if(a0>0)delta-=.5L*a0*a0/rho;
    }
    if(!std::isfinite(delta))throw std::runtime_error("native AL: nonfinite augmented merit difference");
    return delta;
  }
  Eval evaluate(const Vec& x,const Vec& lambda,const Vec& z,double rho,bool hessian,bool derivative=true)const{
    Eval out;if(derivative)out.gradient.assign(p.n,0);if(hessian)out.hessian.assign((std::size_t)p.n*p.n,0);
    Vec g;
    auto row=[&](const SparseQuadratic& q,double w,double normal){
      if(derivative){for(int k=0;k<(int)q.support.size();++k)out.gradient[q.support[k]]+=w*g[k];}
      if(hessian){
        for(auto e:q.h){int i=q.support[e.i],j=q.support[e.j];out.hessian[(std::size_t)i*p.n+j]+=w*e.v;if(i!=j)out.hessian[(std::size_t)j*p.n+i]+=w*e.v;}
        if(normal)for(int a=0;a<(int)q.support.size();++a)for(int b=0;b<=a;++b){int i=q.support[a],j=q.support[b];double v=normal*g[a]*g[b];out.hessian[(std::size_t)i*p.n+j]+=v;if(i!=j)out.hessian[(std::size_t)j*p.n+i]+=v;}
      }
    };
    out.value=obj.value_extended(x,derivative?&g:nullptr);row(obj,1,0);
    for(int i=0;i<(int)eq.size();++i){double v=eq[i].evaluate(x,derivative?&g:nullptr);out.eq.push_back(v);
      out.value+=(long double)lambda[i]*v+.5L*rho*v*v;row(eq[i],lambda[i]+rho*v,rho);}
    for(int i=0;i<(int)iq.size();++i){double v=iq[i].evaluate(x,derivative?&g:nullptr);out.iq.push_back(v);double a=std::max(0.,z[i]+rho*v);
      // Algebraically identical PHR value without subtracting two nearby
      // rounded doubles when z is much larger than rho*g. Keep the branch
      // used by the generalized gradient and Hessian above.
      if(a>0)out.value+=(long double)z[i]*v+.5L*rho*v*v;
      else out.value-=.5L*(long double)z[i]*z[i]/rho;
      row(iq[i],a,a>0?rho:0);}
    if(!std::isfinite(out.value))throw std::runtime_error("native AL: nonfinite augmented value");
    for(double v:out.eq)if(!std::isfinite(v))throw std::runtime_error("native AL: nonfinite equality value");
    for(double v:out.iq)if(!std::isfinite(v))throw std::runtime_error("native AL: nonfinite inequality value");
    if(derivative)for(double v:out.gradient)if(!std::isfinite(v))throw std::runtime_error("native AL: nonfinite gradient component");
    if(hessian)for(double v:out.hessian)if(!std::isfinite(v))throw std::runtime_error("native AL: nonfinite Hessian component");
    if(derivative)for(int j=0;j<p.n;++j){double a=out.gradient[j];if((x[j]<=p.lb[j]&&a>0)||(x[j]>=p.ub[j]&&a<0))a=0;out.projected=std::max(out.projected,std::abs(a));}
    return out;
  }
  Vec box_direction(const Vec& x,const Eval& e){
    const int n=p.n;Vec d(n,0),candidate(n,0);std::vector<int> active(n,0),all;
    for(int j=0;j<n;++j){if(p.lb[j]==p.ub[j])active[j]=2;else{all.push_back(j);if(x[j]<=p.lb[j]&&e.gradient[j]>0)active[j]=-1;if(x[j]>=p.ub[j]&&e.gradient[j]<0)active[j]=1;}}
    std::string msg;double delta=std::max(0.,shift);
    auto solve_face=[&](const std::vector<int>& free,Vec& rhs,bool full){
      std::vector<SymmetricTriplet> entries;entries.reserve(free.size()*(free.size()+1)/2);
      std::vector<std::size_t> diag;
      for(int a=0;a<(int)free.size();++a)for(int b=a;b<(int)free.size();++b){if(a==b)diag.push_back(entries.size());entries.push_back({a,b,e.hessian[(std::size_t)free[a]*n+free[b]]});}
      const Vec pristine_rhs=rhs;
      for(int attempt=0;attempt<(full?22:1)&&!expired();++attempt){
        for(int a=0;a<(int)free.size();++a)entries[diag[a]].value=e.hessian[(std::size_t)free[a]*n+free[a]]+delta;
        ++factors;if(factor->factor(free.size(),free.size(),entries,msg)){++solves;if(solve_from_pristine_rhs(*factor,pristine_rhs,rhs,msg))return true;}
        if(full)delta=delta>0?10*delta:1e-8;
      }
      return false;
    };
    // Certify a single SPD quadratic on ALL nonfixed coordinates before
    // changing its bound active set. Principal face systems share this shift.
    Vec scratch;for(int j:all)scratch.push_back(-e.gradient[j]);
    if(all.empty())return d;
    if(!critical_only&&!solve_face(all,scratch,true))return direction(x,e);
    shift=delta;
    for(int sweep=0;sweep<3*n+20&&!expired();++sweep){
      std::vector<int> free;for(int j=0;j<n;++j)if(!active[j])free.push_back(j);
      candidate=d;
      if(!free.empty()){
        Vec rhs;
        for(int i:free){long double a=-e.gradient[i];for(int j=0;j<n;++j)if(active[j])a-=(long double)e.hessian[(std::size_t)i*n+j]*d[j];rhs.push_back((double)a);}
        if(!solve_face(free,rhs,critical_only))return direction(x,e);
        shift=delta;
        for(int k=0;k<(int)free.size();++k)candidate[free[k]]=rhs[k];
      }
      double alpha=1;int hit=-1,side=0;
      for(int j:free){double a=candidate[j]-d[j];if(a<0&&x[j]+candidate[j]<p.lb[j]){double b=(p.lb[j]-x[j]-d[j])/a;if(b<alpha){alpha=std::max(0.,b);hit=j;side=-1;}}
        if(a>0&&x[j]+candidate[j]>p.ub[j]){double b=(p.ub[j]-x[j]-d[j])/a;if(b<alpha){alpha=std::max(0.,b);hit=j;side=1;}}}
      for(int j:free)d[j]+=alpha*(candidate[j]-d[j]);
      if(hit>=0){d[hit]=(side<0?p.lb[hit]:p.ub[hit])-x[hit];active[hit]=side;continue;}
      int release=-1;double wrong=1e-9;
      for(int i=0;i<n;++i)if(active[i]&&active[i]!=2){long double a=e.gradient[i]+delta*d[i];for(int j=0;j<n;++j)a+=(long double)e.hessian[(std::size_t)i*n+j]*d[j];double violation=active[i]<0?-(double)a:(double)a;if(violation>wrong){wrong=violation;release=i;}}
      if(release<0){long double slope=0;for(int j=0;j<n;++j)slope+=(long double)e.gradient[j]*d[j];if(slope<0)return d;return direction(x,e);}
      active[release]=0;
    }
    return direction(x,e);
  }
  Vec direction(const Vec& x,const Eval& e){
    std::vector<int> free;
    for(int j=0;j<p.n;++j)if(p.lb[j]!=p.ub[j]&&!((x[j]<=p.lb[j]&&e.gradient[j]>0)||(x[j]>=p.ub[j]&&e.gradient[j]<0)))free.push_back(j);
    Vec d(p.n,0);if(free.empty())return d;
    std::vector<SymmetricTriplet> entries;entries.reserve(free.size()*(free.size()+1)/2);
    std::vector<std::size_t> diag;
    for(int a=0;a<(int)free.size();++a)for(int b=a;b<(int)free.size();++b){if(a==b)diag.push_back(entries.size());entries.push_back({a,b,e.hessian[(std::size_t)free[a]*p.n+free[b]]});}
    double delta=std::max(0.,shift);std::string msg;
    for(int retry=0;retry<22&&!expired();++retry){
      for(int a=0;a<(int)free.size();++a)entries[diag[a]].value=e.hessian[(std::size_t)free[a]*p.n+free[a]]+delta;
      ++factors;if(factor->factor(free.size(),free.size(),entries,msg)){
        Vec rhs;for(int j:free)rhs.push_back(-e.gradient[j]);++solves;
        if(factor->solve(rhs,msg)){long double slope=0;for(int a=0;a<(int)free.size();++a)slope+=(long double)e.gradient[free[a]]*rhs[a];
          if(slope<0&&std::isfinite(slope)){for(int a=0;a<(int)free.size();++a)d[free[a]]=rhs[a];shift=delta;return d;}}
      }
      delta=delta>0?10*delta:1e-8;
    }
    // A rejected Newton matrix does not authorize an inaccurate step.
    // Fall back to a diagonally scaled projected-gradient descent direction.
    for(int j:free)d[j]=-e.gradient[j]/std::max(1.,std::abs(e.hessian[(std::size_t)j*p.n+j]));
    return d;
  }
  int minimize(Vec& x,const Vec& lambda,const Vec& z,double rho,double tolerance,int budget){
    for(int it=0;it<budget&&!expired();++it){
      Eval e=evaluate(x,lambda,z,rho,true);if(!std::isfinite(e.value)||!std::isfinite(e.projected))return -2;
      if(e.projected<=tolerance)return it;
      Vec d=box_qp?box_direction(x,e):direction(x,e),trial(p.n);bool accepted=false;double alpha=1;
      for(int pass=0;pass<2&&!accepted;++pass){
        if(pass)for(int j=0;j<p.n;++j)d[j]=-e.gradient[j]/std::max(1.,std::abs(e.hessian[(std::size_t)j*p.n+j]));
        for(int ls=0;ls<45&&!expired();++ls){
          long double slope=0;double step=0;
          for(int j=0;j<p.n;++j){trial[j]=std::clamp(x[j]+alpha*d[j],p.lb[j],p.ub[j]);slope+=(long double)e.gradient[j]*(trial[j]-x[j]);step=std::max(step,std::abs(trial[j]-x[j]));}
          if(slope<0&&merit_difference(x,trial,lambda,z,rho)<=1e-4L*slope){accepted=true;break;}
          alpha*=.5;
          if(step<=1e-16*(1+inf_norm(x)))break;
        }
        if(!accepted)alpha=1;
      }
      if(!accepted)return -1;
      x.swap(trial);++iterations;if(alpha>.5)shift*=.2;
    }
    return budget;
  }
};

} // namespace lazyipdca::qcqp::detail::native_augmented

namespace lazyipdca::qcqp::detail {
inline Result solve_native_augmented_repair(const QCQPProblem& original,const Settings& st){
  using native_augmented::Clock;
  const auto began=Clock::now();
  auto elapsed=[&](){return std::chrono::duration<double>(Clock::now()-began).count();};
  auto expired=[&](){return elapsed()>=st.time_limit_seconds;};
  Result result;
  result.native_augmented.enabled=true;
  result.native_augmented.original_variables=original.n;
  result.native_augmented.reduced_variables=original.n;
  result.native_augmented.seed_policy="original.x0; exact unit-product manifold projection; dyadic coordinates; bound projection";
  result.x=original.x0;
  result.certificate_equality_multipliers.assign(original.equalities.size(),0);
  result.certificate_inequality_multipliers.assign(original.inequalities.size(),0);
  result.certificate_lower_multipliers.assign(original.n,0);
  result.certificate_upper_multipliers.assign(original.n,0);
  result.status=Status::MaxOuterIterations;
  result.message="native augmented repair: accepted-step or AL-stage budget exhausted";
  Settings auditst=st;
  auditst.rank_presolve_context=nullptr;
  auditst.termination_test="project";
  auditst.acceptable_iterations=0;
  auditst.acceptable_on_failure=false;
  const double primal_cap=std::min({1e-6,st.feasibility_tolerance,st.stationarity_tolerance});
  const double dual_cap=std::min({1e-6,st.dual_infeasibility_tolerance,st.stationarity_tolerance});
  const double comp_cap=std::min({1e-6,st.complementarity_tolerance,st.stationarity_tolerance});
  auditst.feasibility_tolerance=primal_cap;
  auditst.dual_infeasibility_tolerance=dual_cap;
  auditst.complementarity_tolerance=comp_cap;
  double certificate_seconds=0,preparation_seconds=0;
  int factors=0,solves=0,accepted=0,chunks=0,stages=0,private_variables=original.n;
  bool admitted=false,initial_projection=false;
  double projection_inf=0;
  std::string reason;
  std::string seed_policy="original.x0; exact unit-product manifold projection; dyadic coordinates; bound projection";
  auto audit=[&](Result& candidate){
    candidate.termination_audit_valid=false;
    const auto at=Clock::now();
    const bool audit_ok=audit_original_termination(original,auditst,candidate);
    certificate_seconds+=std::chrono::duration<double>(Clock::now()-at).count();
    return audit_ok&&candidate.termination_audit_valid&&
      std::isfinite(candidate.feasibility.total_inf)&&std::isfinite(candidate.dual_infeasibility_inf)&&
      std::isfinite(candidate.complementarity_inf)&&candidate.feasibility.total_inf<=primal_cap&&
      candidate.dual_infeasibility_inf<=dual_cap&&candidate.complementarity_inf<=comp_cap;
  };
  const int level=st.live_display_level<0?((st.live_iterations||st.verbose)?1:0):st.live_display_level;
  bool have_audit=false;
  std::unique_ptr<native_augmented::NativeAL> kernel;
  try{
    original.validate();
    if(!(primal_cap>0)||!(dual_cap>0)||!(comp_cap>0)||
       std::isnan(st.time_limit_seconds)||st.time_limit_seconds<0)
      throw std::runtime_error("native AL: invalid physical tolerance or time budget");
    if(st.linear_backend!="mumps"||!st.linear_metis)
      throw std::runtime_error("native AL admission requires the selected MUMPS+METIS backend");
    if(original.has_signed_square())throw std::runtime_error("native AL admission: signed-square functions unsupported");
    if(expired()){
      result.status=Status::TimeLimit;reason="global deadline exhausted before preparation";
    }else if(st.max_outer_iterations<=0){
      reason="accepted Newton step budget is zero";
    }else{
      // Initialization is an explicit algorithmic phase. The provided model
      // and its frozen x0 remain unchanged and alone define the final audit.
      QCQPProblem initialized_model=original;
      const double deadline=std::chrono::duration<double>(began.time_since_epoch()).count()+st.time_limit_seconds;
      const auto ray=native_bilinear_start(original,deadline);
      if(ray.success){
        const double current_feasibility=feasibility(original,original.x0).total_inf;
        const double current_objective=original.objective.value(original.x0);
        bool finite_original=std::isfinite(current_feasibility)&&std::isfinite(current_objective);
        for(const auto& q:original.equalities)finite_original=finite_original&&std::isfinite(q.value(original.x0));
        for(const auto& q:original.inequalities)finite_original=finite_original&&std::isfinite(q.value(original.x0));
        const bool preserve=finite_original&&current_feasibility<=1e-6&&current_objective<=ray.objective;
        if(!preserve){initialized_model.x0=ray.x;seed_policy="original.x0 retained in model; internal uniform-simplex feasible-ray phase; exact unit-product and dyadic maps; bound projection";}
        else seed_policy="provided original.x0 already feasible and no worse than internal ray candidate; exact unit-product and dyadic maps; bound projection";
      }else seed_policy+="; ray initialization not admitted: "+ray.reason;
      ProductDefinitionPresolve reduction(initialized_model);
      const QCQPProblem& reduced=reduction.changed()?reduction.reduced:initialized_model;
      Power2Scaling scaling(reduced);
      const QCQPProblem& p=scaling.scaled;
      private_variables=p.n;
      result.native_augmented.reduced_variables=p.n;
      if(24.L*p.n*p.n>128.L*1024*1024)
        throw std::runtime_error("native AL admission: dense private Hessian construction exceeds 128 MiB");
      if(expired()){
        result.status=Status::TimeLimit;reason="global deadline exhausted during preparation";
      }else{
        kernel=std::make_unique<native_augmented::NativeAL>(p,st,began,st.time_limit_seconds,true,true);
        admitted=true;
        Vec x=p.x0;
        for(int j=0;j<p.n;++j)x[j]=std::clamp(x[j],p.lb[j],p.ub[j]);
        auto state=[&](const Vec& point,const Vec& lambda,const Vec& z){
          Vec gradient=p.objective.gradient(point);
          for(int i=0;i<(int)lambda.size();++i)axpy(lambda[i],p.equalities[i].gradient(point),gradient);
          for(int i=0;i<(int)z.size();++i)if(z[i])axpy(z[i],p.inequalities[i].gradient(point),gradient);
          for(double v:gradient)if(!std::isfinite(v))throw std::runtime_error("native AL: nonfinite certificate gradient");
          Result r;r.x=point;r.certificate_equality_multipliers=lambda;r.certificate_inequality_multipliers=z;
          r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);
          for(int j=0;j<p.n;++j){if(point[j]<=p.lb[j])r.certificate_lower_multipliers[j]=std::max(0.,gradient[j]);if(point[j]>=p.ub[j])r.certificate_upper_multipliers[j]=std::max(0.,-gradient[j]);}
          scaling.lift(r);if(reduction.changed())reduction.lift(r);return r;
        };
        Vec lambda(p.equalities.size(),0),z(p.inequalities.size(),0);
        // Record the actual physical initialization, including the explicit
        // product-manifold and bound projections, without changing original.x0.
        Result initialized=state(x,lambda,z);
        for(int j=0;j<original.n;++j){
          const double change=std::abs(initialized.x[j]-original.x0[j]);
          if(!std::isfinite(change))throw std::runtime_error("native AL: nonfinite initialization projection");
          projection_inf=std::max(projection_inf,change);
          initial_projection=initial_projection||change!=0;
        }
        result=std::move(initialized);
        preparation_seconds=elapsed();
        if(level)std::cout<<"[NATIVE-AUGMENTED-START] original_n="<<original.n<<" private_n="<<p.n
          <<" equations="<<p.equalities.size()<<" inequalities="<<p.inequalities.size()
          <<" exact_product_projection="<<initial_projection<<" projection_inf="<<projection_inf
          <<" accepted_step_budget="<<st.max_outer_iterations<<" seed_policy="<<seed_policy<<" backend=mumps-metis"<<std::endl;
        double rho=10,previous=INFINITY,best=INFINITY;
        bool solved=false;
        // Chunks and multiplier stages are both bounded. A chunk can end
        // before approximate inner stationarity and does not then raise rho.
        for(int outer=0;outer<120&&!kernel->expired()&&kernel->iterations<st.max_outer_iterations;++outer){
          const int budget=std::min(500,st.max_outer_iterations-kernel->iterations);
          double tolerance=std::max(1e-12,std::min(.05,.1*std::pow(.3,stages)));
          const int inner=kernel->minimize(x,lambda,z,rho,tolerance,budget);
          ++chunks;
          native_augmented::Eval e=kernel->evaluate(x,lambda,z,rho,false);
          double violation=0;Vec next_lambda=lambda,next_z=z;
          for(int i=0;i<(int)lambda.size();++i){violation=std::max(violation,std::abs(e.eq[i]));next_lambda[i]+=rho*e.eq[i];}
          for(int i=0;i<(int)z.size();++i){violation=std::max(violation,std::abs(std::max(e.iq[i],-z[i]/rho)));next_z[i]=std::max(0.,z[i]+rho*e.iq[i]);}
          Result r=state(x,next_lambda,next_z);
          solved=audit(r);have_audit=true;
          const double physical_primal=r.feasibility.total_inf;
          const double score=r.termination_audit_valid?std::max({r.feasibility.total_inf/primal_cap,r.dual_infeasibility_inf/dual_cap,r.complementarity_inf/comp_cap}):INFINITY;
          if(level){std::ostringstream line;line<<std::setprecision(12)<<"[NATIVE-AUGMENTED] chunk="<<chunks<<" stage="<<stages
            <<" inner="<<inner<<" rho="<<rho<<" shift="<<kernel->shift<<" local_v="<<violation
            <<" original_f="<<r.objective<<" primal="<<r.feasibility.total_inf<<" dual="<<r.dual_infeasibility_inf
            <<" complementarity="<<r.complementarity_inf<<" factors="<<kernel->factors<<" accepted="<<kernel->iterations;std::cout<<line.str()<<std::endl;}
          if(score<best||solved){best=score;result=std::move(r);}
          if(solved){result.status=Status::Solved;reason="ORIGINAL raw QCQP KKT verified";break;}
          if(kernel->expired())break;
          if(!(inner>=0&&inner<budget))continue;
          lambda.swap(next_lambda);z.swap(next_z);++stages;
          if(outer>0&&violation>.3*previous&&physical_primal>std::min(2e-7,.2*primal_cap))rho=std::min(1e12,5*rho);
          previous=violation;
        }
        factors=kernel->factors;solves=kernel->solves;accepted=kernel->iterations;
        if(!solved){result.status=expired()?Status::TimeLimit:Status::MaxOuterIterations;reason=expired()?"global deadline exhausted":"accepted Newton step or AL-stage budget exhausted";}
      }
    }
  }catch(const std::exception& e){
    result.status=Status::NumericalError;reason=e.what();
  }
  if(kernel){factors=kernel->factors;solves=kernel->solves;accepted=kernel->iterations;kernel.reset();}
  if(preparation_seconds==0)preparation_seconds=std::max(0.,elapsed()-certificate_seconds);
  // Even refused/limited attempts export the complete original coordinates.
  // Audit success never overrides an exhausted global deadline or an error.
  if(!have_audit||!result.termination_audit_valid){try{audit(result);}catch(const std::exception& e){result.termination_audit_valid=false;if(reason.empty())reason=e.what();}}
  if(expired()){result.status=Status::TimeLimit;reason="global deadline exhausted (preparation, core and original audit included)";}
  result.message="native augmented repair: "+reason;
  result.termination_multiplier_source="native-PHR-AL-product-and-dyadic-original-lift";
  result.linear_solver_backend="native-augmented/mumps-metis";
  result.effective_outer_algorithm="native-augmented-repair";
  result.outer_iterations=chunks;result.inner_iterations=accepted;result.kkt_factorizations=factors;
  result.elapsed_seconds=elapsed();
  auto& stats=result.native_augmented;
  stats.enabled=true;stats.admitted=admitted;stats.original_variables=original.n;
  stats.reduced_variables=private_variables;
  stats.initial_projection=initial_projection;stats.initial_projection_inf=projection_inf;
  stats.stages=stages;stats.accepted_steps=accepted;stats.factorizations=factors;stats.linear_solves=solves;
  stats.preparation_seconds=preparation_seconds;stats.certificate_seconds=certificate_seconds;
  stats.core_seconds=std::max(0.,result.elapsed_seconds-preparation_seconds-certificate_seconds);
  stats.reason=reason;stats.seed_policy=seed_policy;
  result.lazy_newton.core_solve_seconds=stats.core_seconds;result.lazy_newton.solver_wall_seconds=result.elapsed_seconds;
  result.lazy_newton.cost_seconds[(int)NewtonCost::Preparation]+=preparation_seconds;
  result.lazy_newton.cost_seconds[(int)NewtonCost::Certificate]+=certificate_seconds;
  result.acceptable_point=false;result.acceptable_convergence=false;
  return result;
}
} // namespace lazyipdca::qcqp::detail
