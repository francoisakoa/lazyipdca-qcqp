#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>

namespace lazyipdca::qcqp::detail {
struct PositiveSquareEpigraphSubstitution {
  bool applied=false;
  int variable=-1;
  QCQPProblem model;
  std::string reason;
  PositiveSquareEpigraphSubstitution(const QCQPProblem& p,const Settings& st,double deadline) {
    using Exact=boost::multiprecision::cpp_rational;
    auto expired=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline;};
    if(st.standard_r10_policy!="auto"||st.convexification_study!="off"||
        (st.dca_engine!="auto"&&st.dca_engine!="majorants")||
        (st.outer_algorithm!="auto"&&st.outer_algorithm!="lazy-dca")||
        st.study_controller!="r3"||st.rank_presolve_context||st.original_private_phase||
        st.product_definition_presolve||st.power2_scaling||
        (st.accepted_checkpoint_callback&&!st.accepted_checkpoint_builtin)||
        p.equalities.empty()||p.inequalities.empty()||p.has_signed_square()||expired())return;
    auto affine=[](const QuadraticFunction& q){return std::all_of(q.hessian.entries.begin(),q.hessian.entries.end(),[](const auto& h){return h.v==0;});};
    if(!affine(p.objective))return;
    int t=-1,terms=0;p.objective.for_each_linear_nonzero([&](int j,double a){++terms;if(a>0)t=j;});
    if(terms!=1||t<0||!std::isfinite(p.lb[t])||p.lb[t]<0||std::isfinite(p.ub[t])||!std::isfinite(p.x0[t])||p.x0[t]<0)return;
    for(int j=0;j<p.n;++j)if(j!=t&&(std::isfinite(p.lb[j])||std::isfinite(p.ub[j])))return;
    for(const auto& q:p.equalities){
      if(!affine(q)||expired())return;
      bool independent=true;q.for_each_linear_nonzero([&](int j,double){if(j==t)independent=false;});if(!independent)return;
    }
    for(const auto& q:p.inequalities){
      bool valid=true;int negative=0;
      q.for_each_linear_nonzero([&](int,double){valid=false;});
      for(const auto& h:q.hessian.entries)if(h.v!=0){
        if(h.i!=h.j)valid=false;
        else if(h.i==t){if(h.v>=0||Exact(.5*h.v)!=Exact(h.v)/2)valid=false;else ++negative;}
        else if(h.v<0)valid=false;
      }
      if(!valid||negative!=1||expired())return;
    }
    const double x0=p.x0[t]*p.x0[t],lb=p.lb[t]*p.lb[t];
    if(!std::isfinite(x0)||!std::isfinite(lb)||Exact(x0)!=Exact(p.x0[t])*Exact(p.x0[t])||Exact(lb)!=Exact(p.lb[t])*Exact(p.lb[t]))return;
    model=p;model.x0[t]=x0;model.lb[t]=lb;
    for(auto& q:model.inequalities){
      q.linear.clear();q.compact_linear_dimension=p.n;q.compact_linear.clear();
      std::vector<SymTriplet> hessian;hessian.reserve(q.hessian.entries.size());
      for(const auto& h:q.hessian.entries){
        if(h.i==t&&h.j==t&&h.v!=0)q.compact_linear.push_back({t,.5*h.v});
        else hessian.push_back(h);
      }
      q.hessian.entries=std::move(hessian);
    }
    if(expired()){model=QCQPProblem{};return;}
    model.validate();variable=t;applied=true;
    reason="exact-u-equals-positive-epigraph-square; monotone-affine-objective; all-constants-retained";
  }
  // F(u)=c*u+c0 and f(t)=c*t+c0 have the same minimizers for t>=0,c>0.
  // At t>0 the original equality and inequality multipliers are y/(2t),
  // z/(2t), while the epigraph's lower-bound multiplier is unchanged.
  bool lift(Result& r) const {
    const int t=variable;
    if(!applied||r.x.size()!=std::size_t(model.n)||r.certificate_lower_multipliers.size()!=r.x.size()||r.certificate_upper_multipliers.size()!=r.x.size()||!std::isfinite(r.x[t])||r.x[t]<=0)return false;
    const double value=std::sqrt(r.x[t]);const double scale=.5/value;
    if(!std::isfinite(value)||!std::isfinite(scale))return false;
    r.x[t]=value;
    for(auto& a:r.certificate_equality_multipliers)a*=scale;
    for(auto& a:r.certificate_inequality_multipliers)a*=scale;
    for(int j=0;j<model.n;++j)if(j!=t){r.certificate_lower_multipliers[j]*=scale;r.certificate_upper_multipliers[j]*=scale;}
    r.termination_scaling_available=false;r.termination_audit_valid=false;
    return true;
  }
};
} // namespace lazyipdca::qcqp::detail
