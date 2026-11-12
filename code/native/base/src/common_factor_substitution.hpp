#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include <chrono>

namespace lazyipdca::qcqp::detail {
// q(x)=c+a*t+sum b_j*t*x_j. With t>0, u_j=t*x_j
// gives an affine polynomial without changing any coefficient or constant.
// A finite box on x_j becomes L_j*t<=u_j<=U_j*t.
struct CommonFactorSubstitution {
  bool applied=false;
  int variable=-1, source_inequalities=0;
  QCQPProblem model;
  struct Side {int variable;bool lower;};
  std::vector<Side> sides;
  CommonFactorSubstitution(const QCQPProblem& p,const Settings& st,double deadline) {
    auto expired=[&]{return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=deadline;};
    if(st.standard_r10_policy!="auto"||st.convexification_study!="off"||
       (st.dca_engine!="auto"&&st.dca_engine!="majorants")||
       (st.outer_algorithm!="auto"&&st.outer_algorithm!="lazy-dca")||
       st.study_controller!="r3"||st.rank_presolve_context||st.original_private_phase||
       st.product_definition_presolve||st.power2_scaling||
       (st.accepted_checkpoint_callback&&!st.accepted_checkpoint_builtin)||p.has_signed_square()||expired())return;
    std::vector<int> candidates;
    auto first=[&](const QuadraticFunction& q){
      if(!candidates.empty())return;
      for(const auto& h:q.hessian.entries)if(h.v!=0){
        if(h.i!=h.j)candidates={h.i,h.j};else candidates={h.i};break;
      }
    };
    first(p.objective);for(const auto& q:p.equalities)first(q);for(const auto& q:p.inequalities)first(q);
    for(int t:candidates){
      // Strictly positive frozen t allows a regular internal map. Zero is
      // permitted by the domain, but a zero final t cannot use this inverse.
      if(!std::isfinite(p.lb[t])||p.lb[t]<0||!(p.x0[t]>0))continue;
      bool valid=true;
      for(int j=0;j<p.n;++j)if(j!=t&&(!std::isfinite(p.lb[j])||!std::isfinite(p.ub[j])))valid=false;
      auto check=[&](const QuadraticFunction& q){
        std::vector<char> seen(p.n,0);
        q.for_each_linear_nonzero([&](int j,double a){if(a!=0&&j!=t)valid=false;});
        for(const auto& h:q.hessian.entries)if(h.v!=0){
          if(h.i==h.j||(h.i!=t&&h.j!=t))valid=false;
          else {int j=h.i==t?h.j:h.i;if(seen[j])valid=false;seen[j]=1;}
        }
      };
      check(p.objective);for(const auto& q:p.equalities)check(q);for(const auto& q:p.inequalities)check(q);
      if(!valid||expired())continue;
      model=p;source_inequalities=int(p.inequalities.size());
      for(int j=0;j<p.n;++j)if(j!=t){
        model.x0[j]=p.x0[t]*p.x0[j];
        if(!std::isfinite(model.x0[j])){valid=false;break;}
        model.lb[j]=-INFINITY;model.ub[j]=INFINITY;
      }
      if(!valid)continue;
      auto transform=[&](QuadraticFunction& q){
        Vec a(p.n,0.);q.for_each_linear_nonzero([&](int j,double v){a[j]+=v;});
        for(const auto& h:q.hessian.entries)if(h.v!=0)a[h.i==t?h.j:h.i]=h.v;
        q.hessian.entries.clear();q.linear=std::move(a);q.compact_linear.clear();q.compact_linear_dimension=0;
      };
      transform(model.objective);for(auto& q:model.equalities)transform(q);for(auto& q:model.inequalities)transform(q);
      sides.clear();
      for(int j=0;j<p.n;++j)if(j!=t)for(bool lower:{true,false}){
        QuadraticFunction q;q.name="common-factor-box";q.hessian.n=p.n;q.linear.assign(p.n,0.);
        q.linear[t]=lower?p.lb[j]:-p.ub[j];q.linear[j]=lower?-1.:1.;
        model.inequalities.push_back(std::move(q));sides.push_back({j,lower});
      }
      if(expired())return;
      model.validate();variable=t;applied=true;return;
    }
  }
  bool lift(const QCQPProblem& p,Result& r) const {
    const int t=variable;
    if(!applied||r.x.size()!=std::size_t(p.n)||!(r.x[t]>0)||!std::isfinite(r.x[t])||
       r.certificate_equality_multipliers.size()!=p.equalities.size()||
       r.certificate_inequality_multipliers.size()!=p.inequalities.size()+sides.size()||
       r.certificate_lower_multipliers.size()!=r.x.size()||r.certificate_upper_multipliers.size()!=r.x.size())return false;
    const double pivot=r.x[t];
    for(int j=0;j<p.n;++j)if(j!=t){r.x[j]/=pivot;if(!std::isfinite(r.x[j]))return false;}
    double defect=0;
    for(std::size_t k=0;k<sides.size();++k){
      auto side=sides[k];double z=r.certificate_inequality_multipliers[source_inequalities+k];
      if(!std::isfinite(z)||z<0)return false;
      const int j=side.variable;
      if(side.lower){r.certificate_lower_multipliers[j]=pivot*z;defect+=z*(p.lb[j]-r.x[j]);}
      else {r.certificate_upper_multipliers[j]=pivot*z;defect+=z*(r.x[j]-p.ub[j]);}
    }
    // Exact transformed KKT has defect=0. Do not hide a nonzero defect in a
    // bound multiplier: leave it for the full original stationarity audit.
    (void)defect;
    r.certificate_inequality_multipliers.resize(source_inequalities);
    r.termination_scaling_available=false;r.termination_audit_valid=false;return true;
  }
};
} // namespace lazyipdca::qcqp::detail
