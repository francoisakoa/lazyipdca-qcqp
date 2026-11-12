#include <iostream>
#include <stdexcept>
#include "../src/atomic_newton.cpp"
using namespace lazyipdca::qcqp;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
QCQPProblem problem(){
  QCQPProblem p;p.n=4;p.name="g16_lean_nonlinear_reduction";
  p.lb={-2,-2,-2,0.25};p.ub={2,2,2,0.25};p.x0={0.2,0.3,0.04,0.25};
  p.objective.linear={0,0,0,3};p.objective.hessian.n=4;
  p.objective.hessian.entries={{0,0,2},{1,1,4},{2,2,3}};
  QuadraticFunction q;q.linear={0,0,1,0};q.hessian.n=4;q.hessian.entries={{0,0,-2}};
  p.equalities.push_back(q);
  q.linear=Vec(4,0);q.hessian.entries={{0,0,2},{1,1,1}};q.constant=-1;
  p.inequalities.push_back(q);
  q.hessian.entries={{0,0,-1},{1,1,2}};q.constant=-3;
  p.inequalities.push_back(q);return p;
}
void same_point(const DcaPoint& a,const DcaPoint& b){
#define SAME(field) require(a.field==b.field,"lean changed original point " #field)
  SAME(objective);SAME(theta);SAME(kkt);SAME(lagrangian);SAME(primal_inf);
  SAME(normalized_kkt);SAME(acceptable_certified);SAME(within_acceptable_caps);
  SAME(distinct_from_anchor);SAME(primal);SAME(dual);SAME(comp);
  SAME(current_lagrangian);SAME(guard_reference_kkt);SAME(feasible_reference_initialized);
#undef SAME
}
std::uint64_t values(const detail::NewtonCostLedger& costs){
  std::uint64_t sum=0;for(int f=0;f<4;++f)sum+=costs.active_calls[f].values;return sum;
}
void original_norms(){
  auto p=problem();
  for(bool reduced:{false,true}){
    Settings st;st.dca_engine="filter";st.linear_backend="native";
    st.lazy_newton_reduction=reduced;st.lazy_newton_lazy_constraints=false;st.live_display_level=0;
    LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
    Workspace w(p,st,ns,costs);DcaFilterStatistics stats;w.dca_stats=&stats;
    // Non-unit scales, nonlinear equalities, fixed coordinates, native rows,
    // converted bounds, inactive rows and distinct anchor multipliers.
    w.objective.scale=0.007;w.equalities[0].scale=0.013;
    w.inequalities[0].scale=0.023;w.inequalities.back().scale=0.041;
    for(const auto* profile:{"project","ipopt"})for(bool inactive:{false,true}){
      st.termination_test=profile;w.active[1]=!inactive;
      for(int trial=0;trial<12;++trial){
        Vec x=w.x,lambda=w.lambda,z=w.z;
        for(int j=0;j<w.n;++j)x[j]+=(trial-5)*0.019*(j+1);
        for(int e=0;e<w.me;++e)lambda[e]=(trial-6)*0.17;
        for(int i=0;i<w.mi;++i)z[i]=(i+1)*0.013*(trial+1);
        Vec anchor_lambda=lambda,anchor_z=z;
        for(double& v:anchor_lambda)v+=0.31;
        for(double& v:anchor_z)v+=0.29;
        st.dca_aggregate_lean=false;
        const auto before=values(costs);
        auto legacy=dca_original(w,x,lambda,z,&anchor_lambda,&anchor_z);
        const auto legacy_values=values(costs)-before;
        st.dca_aggregate_lean=true;
        const auto second=values(costs),saved=stats.original_point_value_reuses;
        auto lean=dca_original(w,x,lambda,z,&anchor_lambda,&anchor_z);
        const auto lean_values=values(costs)-second;
        same_point(legacy,lean);
        require(legacy_values==1+2*std::uint64_t(w.me+w.mi),"legacy value count is not double traversal");
        require(lean_values==1+std::uint64_t(w.me+w.mi),"lean did not remove duplicate values");
        require(stats.original_point_value_reuses-saved==std::uint64_t(w.me+w.mi),"saved value counter incorrect");
      }
    }
    require(stats.original_point_calls==96,"original point calls not counted");
    require(stats.original_point_seconds>=0,"invalid original point timer");
  }
}
void same_quadratic_model(const Quadratic& a,const Quadratic& b){
  require(a.model_active==b.model_active && a.model_constant==b.model_constant &&
      a.model_gradient==b.model_gradient && a.model_hessian.size()==b.model_hessian.size(),
      "deferred layout changed a majorant");
  for(std::size_t i=0;i<a.model_hessian.size();++i){const auto x=a.model_hessian[i],y=b.model_hessian[i];
    require(x.i==y.i && x.j==y.j && x.v==y.v,"deferred majorant Hessian differs");}
}
void deferred_majorant(){
  auto p=problem();Settings st;st.dca_engine="filter";st.linear_backend="native";
  st.dca_aggregate=true;st.dca_aggregate_only=true;st.dca_aggregate_backend="native";
  st.lazy_newton_lazy_constraints=false;st.live_display_level=0;
  LazyNewtonStatistics n0,n1;detail::NewtonCostLedger c0(n0,Clock::now()),c1(n1,Clock::now());
  Settings old_st=st,lean_st=st;old_st.dca_aggregate_lean=false;lean_st.dca_aggregate_lean=true;
  Workspace w0(p,old_st,n0,c0);DcaFilterStatistics s0;DcaModels old(w0,s0);
  Workspace w1(p,lean_st,n1,c1);DcaFilterStatistics s1;DcaModels lean(w1,s1);
  require(old.layout_ready && s0.model_layout_builds==1,"legacy layout not prepared");
  require(!lean.layout_ready && s1.model_layout_builds==0 && lean.original_b.empty() &&
      lean.reduced_scatter.empty() && lean.inequality.empty() && lean.center.empty(),
      "aggregate-only prepared unused majorant data");
  require(lean.slots==old.slots,"aggregate diagonal slots changed");
  // Exercise the actual first consumer, including a subsequent rebuild. A
  // future fallback must not read an uninitialized deferred layout.
  for(bool restoration:{true,false}){
    old.build(0,restoration);lean.build(0,restoration);
    require(lean.layout_ready && s1.model_layout_builds==1,"layout not built exactly once");
    same_quadratic_model(w0.objective,w1.objective);
    for(int e=0;e<w0.me;++e)same_quadratic_model(w0.equalities[e],w1.equalities[e]);
    for(int i=0;i<w0.mi;++i)same_quadratic_model(w0.inequalities[i],w1.inequalities[i]);
    require(s0.models==s1.models && s0.curvature_factors==s1.curvature_factors,
        "deferred layout changed model/certificate counts");
  }
}
void same_endpoint(const Result& a,const Result& b){
  require(a.status==b.status && a.x==b.x && a.objective==b.objective &&
      a.outer_iterations==b.outer_iterations && a.inner_iterations==b.inner_iterations &&
      a.certificate_equality_multipliers==b.certificate_equality_multipliers &&
      a.certificate_inequality_multipliers==b.certificate_inequality_multipliers &&
      a.certificate_lower_multipliers==b.certificate_lower_multipliers &&
      a.certificate_upper_multipliers==b.certificate_upper_multipliers &&
      a.dca_filter.models==b.dca_filter.models &&
      a.dca_filter.aggregate_certificates==b.dca_filter.aggregate_certificates,
      "lean changed solve endpoint or iteration/certificate counts");
}
void solve_routes(){
  auto p=problem();
  for(bool aggregate_only:{false,true}){
    Settings st;st.linear_backend="native";st.dca_engine="filter";
    st.dca_aggregate=aggregate_only;st.dca_aggregate_only=aggregate_only;
    st.dca_aggregate_backend="native";st.dca_aggregate_rho_max=1e8;
    st.lazy_newton_lazy_constraints=false;st.live_display_level=0;st.max_outer_iterations=8;
    st.dca_aggregate_lean=false;auto old=solve_filter_dca(p,st);
    st.dca_aggregate_lean=true;auto lean=solve_filter_dca(p,st);
    same_endpoint(old,lean);
    if(aggregate_only)require(lean.dca_filter.model_layout_builds==0 && lean.dca_filter.models==0,
        "aggregate-only constructed a majorant layout");
    else require(lean.dca_filter.model_layout_builds==1 && lean.dca_filter.models>0,
        "majorant route not covered by parity test");
    require(lean.dca_filter.original_point_calls>0 && lean.dca_filter.original_point_value_reuses>0,
        "solver did not connect original point counters");
  }
}
}
int main(){try{
  original_norms();deferred_majorant();solve_routes();
  std::cout<<"PASS exact original norms, saved evaluations, deferred majorant/restoration rebuild, solve route parity\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
