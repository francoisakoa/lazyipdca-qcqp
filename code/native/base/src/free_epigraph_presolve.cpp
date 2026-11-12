#include "lazyipdca_qcqp/free_epigraph_presolve.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>

namespace lazyipdca::qcqp {
namespace {
using Exact=boost::multiprecision::cpp_rational;
using Clock=std::chrono::steady_clock;
Exact exact_value(const QuadraticFunction& q,const Vec& x,int omit=-1) {
  Exact value(q.constant);
  q.for_each_linear_nonzero([&](int j,double a){if(j!=omit)value+=Exact(a)*Exact(x[j]);});
  for(const auto& e:q.hessian.entries) {
    if(e.i==omit||e.j==omit)continue;
    Exact term=Exact(e.v)*Exact(x[e.i])*Exact(x[e.j]);
    value+=e.i==e.j?term/2:term;
  }
  return value;
}
double lift_coordinate(const QCQPProblem& p,const FreeEpigraphPresolve::Leaf& leaf,const Vec& x) {
  const Exact q=exact_value(p.inequalities[leaf.row],x,leaf.variable);
  if(q<=0)return std::copysign(0.0,p.x0[leaf.variable]);
  const Exact target=q/(-Exact(leaf.diagonal)/2);
  const long double wide=target.convert_to<long double>();
  double t=static_cast<double>(std::sqrt(wide));
  if(!(t>=0)||!std::isfinite(t))throw std::runtime_error("free-epigraph: nonfinite outward lift");
  // Exact comparisons of the original binary64 coefficients and point make
  // every outward rounding step verifiable, independent of sqrt rounding.
  int steps=0;
  while(Exact(t)*Exact(t)<target) {
    t=std::nextafter(t,INFINITY);
    if(!std::isfinite(t)||++steps>8)throw std::runtime_error("free-epigraph: outward lift not representable");
  }
  return t;
}
}

FreeEpigraphPresolve::FreeEpigraphPresolve(const QCQPProblem& p):original_(p),reduced_(p) {
  p.validate();stats_.enabled=true;stats_.original_variables=p.n;stats_.reduced_variables=p.n;
  stats_.original_inequalities=p.inequalities.size();stats_.reduced_inequalities=p.inequalities.size();
  stats_.reason="no-proven-free-epigraph-leaf";
  if(p.has_signed_square()){stats_.reason="signed-square-model-not-supported";return;}
  if(std::any_of(p.x0.begin(),p.x0.end(),[](double v){return !std::isfinite(v);})) {
    stats_.reason="nonfinite-start-not-supported";return;
  }
  std::vector<std::vector<int>> incidence(p.n);
  auto add=[&](const QuadraticFunction& q,int id){
    q.for_each_linear_nonzero([&](int j,double a){if(a)incidence[j].push_back(id);});
    for(const auto& e:q.hessian.entries)if(e.v){incidence[e.i].push_back(id);incidence[e.j].push_back(id);}
  };
  add(p.objective,0);
  for(int e=0;e<int(p.equalities.size());++e)add(p.equalities[e],1+e);
  const int first_inequality=1+p.equalities.size();
  for(int i=0;i<int(p.inequalities.size());++i)add(p.inequalities[i],first_inequality+i);
  for(auto& ids:incidence){std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());}
  std::vector<Leaf> candidates;std::vector<int> row_count(p.inequalities.size(),0);
  for(int j=0;j<p.n;++j) {
    const auto& ids=incidence[j];if(ids.size()!=1||ids[0]<first_inequality)continue;
    const int row=ids[0]-first_inequality;const auto& q=p.inequalities[row];
    bool linear=false;q.for_each_linear_nonzero([&](int k,double a){if(k==j&&a)linear=true;});
    if(linear)continue;
    int count=0;double diagonal=0;bool pure=true;
    for(const auto& e:q.hessian.entries)if(e.v&&(e.i==j||e.j==j)) {
      ++count;pure=pure&&e.i==j&&e.j==j;diagonal=e.v;
    }
    if(count!=1||!pure||!(diagonal<0))continue;
    ++stats_.candidates;++row_count[row];
    if(p.lb[j]!=0||p.ub[j]!=INFINITY){++stats_.bounds_rejected;continue;}
    if(p.x0[j]!=0||exact_value(q,p.x0,j)>0){++stats_.start_rejected;continue;}
    candidates.push_back({j,row,diagonal});
  }
  std::vector<bool> removed(p.n,false),removed_row(p.inequalities.size(),false);
  for(const auto& leaf:candidates) {
    // Multi-leaf rows need a group lift; do not silently choose one branch.
    if(row_count[leaf.row]!=1){++stats_.ambiguous_rejected;continue;}
    leaves_.push_back(leaf);removed[leaf.variable]=true;removed_row[leaf.row]=true;
  }
  if(leaves_.empty())return;
  for(int j=0;j<p.n;++j)if(!removed[j])kept_variables_.push_back(j);
  if(kept_variables_.empty()){leaves_.clear();stats_.reason="no-retained-coordinate";return;}
  for(int i=0;i<int(p.inequalities.size());++i)if(!removed_row[i])kept_inequalities_.push_back(i);
  std::vector<int> map(p.n,-1);for(int k=0;k<int(kept_variables_.size());++k)map[kept_variables_[k]]=k;
  QCQPProblem reduced;reduced.name=p.name;reduced.n=kept_variables_.size();
  for(int j:kept_variables_){reduced.x0.push_back(p.x0[j]);reduced.lb.push_back(p.lb[j]);reduced.ub.push_back(p.ub[j]);}
  auto rewrite=[&](const QuadraticFunction& q){
    QuadraticFunction out;out.name=q.name;out.constant=q.constant;out.hessian.n=reduced.n;
    out.compact_linear_dimension=reduced.n;
    q.for_each_linear_nonzero([&](int j,double a){if(a){
      if(map[j]<0)throw std::runtime_error("free-epigraph: nonlocal linear incidence");
      out.compact_linear.emplace_back(map[j],a);
    }});
    for(const auto& e:q.hessian.entries)if(e.v){
      if(map[e.i]<0||map[e.j]<0)throw std::runtime_error("free-epigraph: nonlocal quadratic incidence");
      out.hessian.entries.push_back({map[e.i],map[e.j],e.v});
    }
    return out;
  };
  reduced.objective=rewrite(p.objective);
  for(const auto& q:p.equalities)reduced.equalities.push_back(rewrite(q));
  for(int i:kept_inequalities_)reduced.inequalities.push_back(rewrite(p.inequalities[i]));
  reduced.validate();reduced_=std::move(reduced);
  Result initial;initial.x=reduced_.x0;initial.certificate_equality_multipliers.assign(reduced_.equalities.size(),0);
  initial.certificate_inequality_multipliers.assign(reduced_.inequalities.size(),0);
  initial.certificate_lower_multipliers.assign(reduced_.n,0);initial.certificate_upper_multipliers.assign(reduced_.n,0);
  lift(initial);
  stats_.initial_point_preserved=initial.x.size()==p.x0.size()&&std::memcmp(initial.x.data(),p.x0.data(),p.x0.size()*sizeof(double))==0;
  if(!stats_.initial_point_preserved)throw std::runtime_error("free-epigraph: initial point not bit preserved");
  stats_.applied=true;stats_.selected=leaves_.size();stats_.reduced_variables=reduced_.n;
  stats_.reduced_inequalities=reduced_.inequalities.size();stats_.reason="exact-objective-free-negative-square-leaves";
}

void FreeEpigraphPresolve::lift(Result& r) const {
  if(!changed())return;
  if(r.x.size()!=std::size_t(reduced_.n)||r.certificate_equality_multipliers.size()!=original_.equalities.size()||
      r.certificate_inequality_multipliers.size()!=kept_inequalities_.size()||
      r.certificate_lower_multipliers.size()!=r.x.size()||r.certificate_upper_multipliers.size()!=r.x.size())
    throw std::runtime_error("free-epigraph: incomplete reduced state");
  if(std::any_of(r.x.begin(),r.x.end(),[](double v){return !std::isfinite(v);}))
    throw std::runtime_error("free-epigraph: nonfinite reduced point");
  Vec x=original_.x0,lower(original_.n,0),upper(original_.n,0),ineq(original_.inequalities.size(),0);
  for(int k=0;k<int(kept_variables_.size());++k){const int j=kept_variables_[k];
    x[j]=r.x[k];lower[j]=r.certificate_lower_multipliers[k];upper[j]=r.certificate_upper_multipliers[k];
  }
  for(int k=0;k<int(kept_inequalities_.size());++k)ineq[kept_inequalities_[k]]=r.certificate_inequality_multipliers[k];
  for(const auto& leaf:leaves_)x[leaf.variable]=lift_coordinate(original_,leaf,x);
  if(r.termination_scaling_available) {
    if(r.termination_inequality_scales.size()!=kept_inequalities_.size()||r.termination_inequality_slacks.size()!=kept_inequalities_.size())
      throw std::runtime_error("free-epigraph: incomplete scaling state");
    Vec scale(original_.inequalities.size(),1),slack(original_.inequalities.size(),0);
    for(int k=0;k<int(kept_inequalities_.size());++k){const int i=kept_inequalities_[k];
      scale[i]=r.termination_inequality_scales[k];slack[i]=r.termination_inequality_slacks[k];
    }
    for(const auto& leaf:leaves_)slack[leaf.row]=std::max(0.,-original_.inequalities[leaf.row].value(x));
    r.termination_inequality_scales=std::move(scale);r.termination_inequality_slacks=std::move(slack);
  }
  r.x=std::move(x);r.certificate_lower_multipliers=std::move(lower);r.certificate_upper_multipliers=std::move(upper);
  r.certificate_inequality_multipliers=std::move(ineq);
}

Result solve_with_free_epigraph_presolve(const QCQPProblem& p,const Settings& st) {
  const auto began=Clock::now();auto elapsed=[&]{return std::chrono::duration<double>(Clock::now()-began).count();};
  FreeEpigraphPresolve reduction(p);FreeEpigraphStatistics stats=reduction.statistics();stats.preparation_seconds=elapsed();
  Settings inner=st;inner.free_epigraph_presolve=false;
  const auto& model=reduction.changed()?reduction.reduced():p;
  if(st.live_display_level)std::cout<<"[FREE-EPIGRAPH-PRESOLVE] applied="<<reduction.changed()<<" candidates="<<stats.candidates
      <<" leaves="<<stats.selected<<" variables="<<p.n<<"->"<<model.n<<" inequalities="<<p.inequalities.size()<<"->"<<model.inequalities.size()
      <<" exact_x0="<<stats.initial_point_preserved<<" no_objective_change=1 zero_row_and_leaf_bound_duals=1"<<std::endl;
  if(reduction.changed()&&st.accepted_checkpoint_callback) {
    const auto observer=st.accepted_checkpoint_callback;
    inner.accepted_checkpoint_callback=[&,observer](const QCQPProblem& current,Result& snapshot,const ConstraintPresolve* context,int iteration,double) {
      const auto start=Clock::now();
      try {
        if(std::isfinite(st.time_limit_seconds)&&elapsed()>=st.time_limit_seconds)
          throw std::runtime_error("free-epigraph checkpoint deadline before lift");
        if(&current!=&reduction.reduced()&&!(context&&&context->original()==&reduction.reduced()&&&context->reduced()==&current))
          throw std::runtime_error("free-epigraph checkpoint model owner unavailable");
        Result full=snapshot;if(context)context->lift_checkpoint(full);reduction.lift(full);
        if(std::isfinite(st.time_limit_seconds)&&elapsed()>=st.time_limit_seconds)
          throw std::runtime_error("free-epigraph checkpoint deadline after lift");
        observer(p,full,nullptr,iteration,elapsed());++stats.checkpoint_lifts;
      }catch(...){stats.checkpoint_seconds+=std::chrono::duration<double>(Clock::now()-start).count();throw;}
      stats.checkpoint_seconds+=std::chrono::duration<double>(Clock::now()-start).count();
    };
  }
  if(std::isfinite(st.time_limit_seconds))inner.time_limit_seconds=std::max(0.,st.time_limit_seconds-elapsed());
  Result r;
  if(inner.time_limit_seconds>0)r=Solver(inner).solve(model);
  else {r.status=Status::TimeLimit;r.message="free-epigraph: preparation deadline";r.x=model.x0;
    r.certificate_equality_multipliers.assign(model.equalities.size(),0);r.certificate_inequality_multipliers.assign(model.inequalities.size(),0);
    r.certificate_lower_multipliers.assign(model.n,0);r.certificate_upper_multipliers.assign(model.n,0);}
  const auto post=Clock::now();
  if(reduction.changed()) {
    const auto child_message=r.message;
    const auto child_multiplier_source=r.termination_multiplier_source;
    try {
      reduction.lift(r);Settings original=st;original.rank_presolve_context=nullptr;original.free_epigraph_presolve=false;
      const bool strict=detail::audit_original_termination(p,original,r);stats.postsolve_valid=r.termination_audit_valid;
      if((r.status==Status::Solved&&!strict)||((r.status==Status::SolvedAcceptable||r.status==Status::SolvedAcceptableOnFailure)&&!r.acceptable_point)) {
        r.status=Status::NumericalError;r.message="free-epigraph: full original postsolve certificate failed";
      } else if(r.status==Status::Solved)r.message=child_message+"; exact free-epigraph lift and full original audit";
      r.termination_multiplier_source="free-epigraph-zero-duals-full-original-audit:"+child_multiplier_source;
    }catch(const std::exception& error) {
      r.status=Status::NumericalError;r.message=error.what();stats.postsolve_valid=false;
      // A failed lift must never export a private-dimensional endpoint.
      r.x=p.x0;r.certificate_equality_multipliers.assign(p.equalities.size(),0);r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
      r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);
      r.termination_scaling_available=false;r.termination_audit_valid=false;r.acceptable_point=false;r.acceptable_convergence=false;
      r.objective=p.objective.value(r.x);r.feasibility=feasibility(p,r.x);r.dual_infeasibility_inf=r.complementarity_inf=r.scaled_nlp_error=r.raw_kkt_error=INFINITY;
    }
  } else stats.postsolve_valid=r.termination_audit_valid;
  stats.postsolve_seconds=std::chrono::duration<double>(Clock::now()-post).count();
  r.lazy_newton.cost_seconds[int(NewtonCost::Preparation)]+=stats.preparation_seconds;
  r.lazy_newton.cost_seconds[int(NewtonCost::Certificate)]+=stats.postsolve_seconds;
  r.free_epigraph=stats;r.elapsed_seconds=elapsed();r.lazy_newton.solver_wall_seconds=r.elapsed_seconds;
  if(std::isfinite(st.time_limit_seconds)&&r.elapsed_seconds>st.time_limit_seconds) {
    r.status=Status::TimeLimit;r.message="free-epigraph: original solve/audit deadline";
  }
  if(st.live_display_level&&reduction.changed())std::cout<<"[FREE-EPIGRAPH-POSTSOLVE] original_n="<<r.x.size()
      <<" audit_valid="<<stats.postsolve_valid<<" status="<<status_string(r.status)<<" primal="<<r.feasibility.total_inf
      <<" dual="<<r.dual_infeasibility_inf<<" compl="<<r.complementarity_inf<<" elapsed="<<r.elapsed_seconds<<std::endl;
  return r;
}
}
