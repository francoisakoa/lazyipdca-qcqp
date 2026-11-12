#pragma once
// Native, same-execution certificate finishing. This is not an endpoint reader,
// warm-start interface, primal optimizer, or relaxation of original tolerances.
#include "acceptable.hpp"
#include "rank_presolve.hpp"
#include "bounded_normal_restart.hpp"
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>

namespace lazyipdca::qcqp::detail {
using NativeFinishClock=std::chrono::steady_clock;
inline double native_finish_now() {
  return std::chrono::duration<double>(NativeFinishClock::now().time_since_epoch()).count();
}
inline bool native_finish_bits_equal(const Vec& a,const Vec& b) {
  return a.size()==b.size()&&(a.empty()||std::memcmp(a.data(),b.data(),a.size()*sizeof(double))==0);
}
inline bool native_finish_finite(const Vec& a) {
  return std::all_of(a.begin(),a.end(),[](double x){return std::isfinite(x);});
}
inline Result native_finish_zero_state(const QCQPProblem& p,const Vec& x) {
  Result r;r.status=Status::FactorizationFailure;r.x=x;
  r.certificate_equality_multipliers.assign(p.equalities.size(),0);
  r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
  r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);
  return r;
}
inline bool native_finish_primal(const QCQPProblem& p,const Vec& x,double& primal,double& objective) {
  using L=long double;
  if(x.size()!=std::size_t(p.n)||!native_finish_finite(x))return false;
  auto value=[&](const QuadraticFunction& q,L& v) {
    v=q.constant;
    q.for_each_linear_nonzero([&](int j,double a){v+=L(a)*x[j];});
    for(auto h:q.hessian.entries)v+=(h.i==h.j?.5L:1.L)*L(h.v)*x[h.i]*x[h.j];
    for(auto e:q.signed_square)v+=L(e.second)*x[e.first]*std::abs(L(x[e.first]));
    return std::isfinite(v)&&std::isfinite(double(v));
  };
  L maximum=0,v=0;
  if(!value(p.objective,v))return false;objective=double(v);
  for(const auto& q:p.equalities){if(!value(q,v))return false;maximum=std::max(maximum,std::abs(v));}
  for(const auto& q:p.inequalities){if(!value(q,v))return false;maximum=std::max(maximum,v);}
  for(int j=0;j<p.n;++j) {
    if(std::isnan(p.lb[j])||std::isnan(p.ub[j]))return false;
    if(std::isfinite(p.lb[j]))maximum=std::max(maximum,L(p.lb[j])-x[j]);
    if(std::isfinite(p.ub[j]))maximum=std::max(maximum,L(x[j])-p.ub[j]);
  }
  primal=double(maximum);return std::isfinite(primal)&&primal>=0;
}

struct NativeOriginalView;
inline thread_local const NativeOriginalView* native_original_view_current=nullptr;
// A view maps a COPY from its current model to the owning parent model. The
// normal Solver return path still performs its ordinary lift exactly once.
struct NativeOriginalView {
  const QCQPProblem* model;
  const QCQPProblem* original;
  const Settings* original_settings;
  const NativeOriginalView* parent;
  std::function<bool(Result&)> lift_to_parent;
  double absolute_deadline;
  bool supported=true;
  NativeOriginalView(const QCQPProblem& m,const QCQPProblem& p,const Settings& st,
      std::function<bool(Result&)> lift,double deadline)
    :model(&m),original(&p),original_settings(&st),parent(native_original_view_current),
     lift_to_parent(std::move(lift)),absolute_deadline(deadline) {
    supported=!parent||parent->model==original;
    if(parent)absolute_deadline=std::min(absolute_deadline,parent->absolute_deadline);
  }
  const QCQPProblem& root_problem() const {return parent?parent->root_problem():*original;}
  const Settings& root_settings() const {return parent?parent->root_settings():*original_settings;}
  bool lift_copy(Result& r) const {
    if(!supported||!lift_to_parent(r))return false;
    return !parent||parent->lift_copy(r);
  }
};
struct NativeOriginalViewScope {
  const NativeOriginalView* previous;
  explicit NativeOriginalViewScope(const NativeOriginalView& v):previous(native_original_view_current) {
    native_original_view_current=&v;
  }
  NativeOriginalViewScope(const NativeOriginalViewScope&)=delete;
  ~NativeOriginalViewScope(){native_original_view_current=previous;}
};

enum class NativeAnchorOrigin { AcceptedR3Step, RestorationResult, HistoricalReturnedState };
struct NativeAnchorStore;
inline thread_local NativeAnchorStore* native_anchor_store_current=nullptr;
struct NativeAnchorIdentityAlias;
inline thread_local const NativeAnchorIdentityAlias* native_anchor_identity_alias_current=nullptr;
// Declare identity only at an owning native site that has just copied its
// complete model and changed only x0. This is an execution-scoped provenance
// edge, never a search by dimension, coefficients, filename or endpoint.
struct NativeAnchorIdentityAlias {
  const QCQPProblem* copied_model;
  const QCQPProblem* parent_model;
  const NativeAnchorStore* owner;
  const void* execution_token;
  const NativeAnchorIdentityAlias* previous;
  NativeAnchorIdentityAlias(const QCQPProblem& copied,const QCQPProblem& parent);
  NativeAnchorIdentityAlias(const NativeAnchorIdentityAlias&)=delete;
  ~NativeAnchorIdentityAlias(){native_anchor_identity_alias_current=previous;}
};
struct NativeAnchorStore {
  const QCQPProblem& canonical;
  const Settings& canonical_settings;
  const NativeOriginalView* view;
  double deadline;
  bool enabled;
  std::unique_ptr<Result> anchor;
  double original_primal=INFINITY,original_objective=INFINITY;
  NativeAnchorOrigin origin=NativeAnchorOrigin::AcceptedR3Step;
  int observations=0,primal_audits=0,ownership_refusals=0,stored=0;
  double observation_seconds=0;
  const void* execution_token;
  NativeAnchorStore(const QCQPProblem& p,const Settings& st,double absolute_deadline,bool on)
    :canonical(p),canonical_settings(st),view(native_original_view_current),deadline(absolute_deadline),enabled(on),execution_token(this) {
    if(view){enabled=enabled&&view->supported&&view->model==&canonical;deadline=std::min(deadline,view->absolute_deadline);}
    const auto& full=original_problem();
    const long double cells=3.L*p.n+p.equalities.size()+p.inequalities.size()+
        3.L*full.n+full.equalities.size()+full.inequalities.size();
    enabled=enabled&&cells*sizeof(double)<=32.L*1024*1024;
  }
  const QCQPProblem& original_problem()const{return view?view->root_problem():canonical;}
  const Settings& original_settings()const{return view?view->root_settings():canonical_settings;}
  double original_tolerance()const{return std::min(1e-6,original_settings().feasibility_tolerance);}
  bool wants(double private_hint,bool private_phase)const {
    return enabled&&!private_phase&&native_anchor_store_current==this&&native_finish_now()<deadline&&
        std::isfinite(private_hint)&&private_hint>=0&&private_hint<=original_tolerance();
  }
  bool lifted(Result& copy)const {
    try{return !view||view->lift_copy(copy);}catch(...){return false;}
  }
  void observe(const QCQPProblem& model,const ConstraintPresolve* reduction,const Vec& point,
      double private_hint,bool private_phase,NativeAnchorOrigin why) {
    if(!wants(private_hint,private_phase))return;
    const double began=native_finish_now();++observations;
    auto done=[&]{observation_seconds+=native_finish_now()-began;};
    try {
      if(point.size()!=std::size_t(model.n)||!native_finish_finite(point)){done();return;}
      Result candidate=native_finish_zero_state(model,point);
      const QCQPProblem* owned_model=&model;
      // Nested copies are visited from the innermost live scope to its
      // owning parent. Each edge must belong to this same anchor execution.
      for(auto alias=native_anchor_identity_alias_current;alias;alias=alias->previous)
        if(alias->owner==this&&alias->execution_token==execution_token&&alias->copied_model==owned_model)owned_model=alias->parent_model;
      if(owned_model!=&canonical) {
        if(!reduction||&reduction->original()!=&canonical||&reduction->reduced()!=owned_model){++ownership_refusals;done();return;}
        reduction->lift_checkpoint(candidate);
      }
      double private_primal=INFINITY,unused=0;
      if(!native_finish_primal(canonical,candidate.x,private_primal,unused)||private_primal>original_tolerance()){done();return;}
      Result full=candidate;
      if(!lifted(full)){++ownership_refusals;done();return;}
      double score=INFINITY,objective=INFINITY;++primal_audits;
      if(!native_finish_primal(original_problem(),full.x,score,objective)||score>original_tolerance()||native_finish_now()>=deadline){done();return;}
      if(!anchor||score<original_primal||(score==original_primal&&objective<original_objective)) {
        // All certificate arrays are rebuilt by the fixed-primal fit later.
        candidate=native_finish_zero_state(canonical,candidate.x);
        candidate.feasibility.total_inf=private_primal;
        candidate.dual_infeasibility_inf=INFINITY;candidate.complementarity_inf=INFINITY;
        anchor=std::make_unique<Result>(std::move(candidate));
        original_primal=score;original_objective=objective;origin=why;++stored;
      }
    }catch(...){++ownership_refusals;}
    done();
  }
};
inline NativeAnchorIdentityAlias::NativeAnchorIdentityAlias(
    const QCQPProblem& copied,const QCQPProblem& parent)
  :copied_model(&copied),parent_model(&parent),owner(native_anchor_store_current),
   execution_token(owner?owner->execution_token:nullptr),previous(native_anchor_identity_alias_current) {
  native_anchor_identity_alias_current=this;
}
struct NativeAnchorCaptureScope {
  NativeAnchorStore* previous;
  explicit NativeAnchorCaptureScope(NativeAnchorStore& s):previous(native_anchor_store_current){native_anchor_store_current=&s;}
  NativeAnchorCaptureScope(const NativeAnchorCaptureScope&)=delete;
  ~NativeAnchorCaptureScope(){native_anchor_store_current=previous;}
};
inline bool native_anchor_capture_wanted(double hint,bool private_phase) {
  return native_anchor_store_current&&native_anchor_store_current->wants(hint,private_phase);
}
inline void native_anchor_observe(const QCQPProblem& model,const ConstraintPresolve* reduction,
    const Vec& point,double hint,bool private_phase,NativeAnchorOrigin origin) {
  if(native_anchor_store_current)native_anchor_store_current->observe(model,reduction,point,hint,private_phase,origin);
}

struct NativeSparseFixedPrimalWork {
  bool attempted=false,accepted=false;
  std::size_t columns=0,nonzeros=0;
  int sweeps=0;
  double assembly_seconds=0,coordinate_seconds=0,audit_seconds=0,total_seconds=0;
};
// No dense method, external optimizer, factorization or direct linear solve.
// The sparse bounded normal method starts its multipliers from zero.
inline bool native_sparse_fixed_primal_fit(const QCQPProblem& p,const Settings& st,
    Result& r,double budget_seconds,NativeSparseFixedPrimalWork& work) {
  const double start=native_finish_now();
  auto elapsed=[&]{return native_finish_now()-start;};
  auto done=[&](bool ok){work.total_seconds=elapsed();return ok;};
  if(!(budget_seconds>.005)||p.has_signed_square()||st.rank_presolve_context||
      r.x.size()!=std::size_t(p.n)||!native_finish_finite(r.x))return done(false);
  double primal=INFINITY,objective=INFINITY;
  if(!native_finish_primal(p,r.x,primal,objective)||primal>st.feasibility_tolerance)return done(false);
  const Vec point=r.x;const Vec gradient=p.objective.gradient(point);
  if(!native_finish_finite(gradient))return done(false);
  std::vector<BoundedNormalColumn> normals;
  std::vector<std::pair<int,int>> destinations;
  auto open=[&]{return elapsed()<budget_seconds&&work.nonzeros<=64u*1024u*1024u/sizeof(std::pair<int,double>);};
  auto add=[&](const Vec& g,double value,int kind,int index) {
    if(!native_finish_finite(g)||!std::isfinite(value))return false;
    BoundedNormalColumn c;
    if(kind){c.lower=0;c.upper=value==0?INFINITY:.8*st.complementarity_tolerance/std::abs(value);}
    for(int j=0;j<p.n;++j)if(g[j]!=0)c.entries.push_back({j,g[j]});
    if(c.entries.empty())return true;
    work.nonzeros+=c.entries.size();normals.push_back(std::move(c));destinations.push_back({kind,index});
    return open();
  };
  for(int i=0;i<int(p.equalities.size());++i)if(!open()||!add(p.equalities[i].gradient(point),0,0,i))return done(false);
  for(int i=0;i<int(p.inequalities.size());++i)if(!open()||!add(p.inequalities[i].gradient(point),p.inequalities[i].value(point),1,i))return done(false);
  for(int j=0;j<p.n;++j) {
    if(!open())return done(false);Vec g(p.n,0);
    if(std::isfinite(p.lb[j])){g[j]=-1;if(!add(g,p.lb[j]-point[j],2,j))return done(false);g[j]=0;}
    if(std::isfinite(p.ub[j])){g[j]=1;if(!add(g,point[j]-p.ub[j],3,j))return done(false);}
  }
  work.attempted=true;work.columns=normals.size();work.assembly_seconds=elapsed();
  const double remaining=budget_seconds-elapsed();
  const double allowance=std::max(0.,std::min(.8*remaining,remaining-.005));
  BoundedNormalRestartWork coordinate_work;
  const auto fit=bounded_normal_restart(gradient,normals,allowance,80000,&coordinate_work);
  work.sweeps=coordinate_work.sweeps;work.coordinate_seconds=coordinate_work.seconds;
  if(!fit.valid||fit.multipliers.size()!=normals.size()||!open())return done(false);
  Result candidate=r;candidate.x=point;
  candidate.certificate_equality_multipliers.assign(p.equalities.size(),0);
  candidate.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
  candidate.certificate_lower_multipliers.assign(p.n,0);candidate.certificate_upper_multipliers.assign(p.n,0);
  for(std::size_t j=0;j<destinations.size();++j) {
    const auto [kind,index]=destinations[j];
    Vec* out=kind==0?&candidate.certificate_equality_multipliers:kind==1?&candidate.certificate_inequality_multipliers:
        kind==2?&candidate.certificate_lower_multipliers:&candidate.certificate_upper_multipliers;
    (*out)[index]=fit.multipliers[j];
  }
  const double audit_start=native_finish_now();
  Settings check=st;check.rank_presolve_context=nullptr;check.termination_test="project";
  check.acceptable_iterations=0;check.acceptable_on_failure=false;
  const bool strict=audit_original_termination(p,check,candidate)&&
      candidate.feasibility.total_inf<=check.feasibility_tolerance&&
      candidate.dual_infeasibility_inf<=std::min(check.stationarity_tolerance,check.dual_infeasibility_tolerance)&&
      candidate.complementarity_inf<=check.complementarity_tolerance;
  work.audit_seconds=native_finish_now()-audit_start;
  if(!strict||!native_finish_bits_equal(candidate.x,point)||elapsed()>=budget_seconds)return done(false);
  candidate.status=Status::Solved;candidate.message="native same-execution fixed-primal sparse multiplier certificate";
  candidate.termination_multiplier_source="native-same-execution-sparse-fixed-primal-current-model-audit";
  r=std::move(candidate);work.accepted=true;return done(true);
}

struct NativeAnchorFinishOutcome {
  bool attempted=false,accepted=false,private_strict=false;
  int outer_charge=0;
  double seconds=0,original_primal=INFINITY,original_dual=INFINITY,original_complementarity=INFINITY;
  NativeSparseFixedPrimalWork work;
};
inline NativeAnchorFinishOutcome native_original_anchor_finish(NativeAnchorStore& store,
    Result& selected,double allowance,int remaining_outer,int level) {
  NativeAnchorFinishOutcome out;const double began=native_finish_now();
  allowance=std::min(allowance,store.deadline-began);
  if(native_anchor_store_current!=&store||!store.enabled||!store.anchor||store.execution_token!=&store||remaining_outer<1||!(allowance>.01))return out;
  out.attempted=true;out.outer_charge=1;
  Result candidate=selected;candidate.x=store.anchor->x;const Vec native_point=candidate.x;
  Settings fit=store.canonical_settings;const auto& original_settings=store.original_settings();
  fit.rank_presolve_context=nullptr;fit.termination_test="project";
  fit.feasibility_tolerance=std::min(1e-6,original_settings.feasibility_tolerance);
  fit.stationarity_tolerance=std::min(1e-6,original_settings.stationarity_tolerance);
  fit.dual_infeasibility_tolerance=std::min(1e-6,original_settings.dual_infeasibility_tolerance);
  fit.complementarity_tolerance=std::min(1e-6,original_settings.complementarity_tolerance);
  out.private_strict=native_sparse_fixed_primal_fit(store.canonical,fit,candidate,.9*allowance,out.work);
  if(out.private_strict&&native_finish_bits_equal(candidate.x,native_point)) {
    Result original=candidate;
    if(store.lifted(original)) {
      Settings audit=original_settings;audit.rank_presolve_context=nullptr;audit.termination_test="project";
      audit.acceptable_iterations=0;audit.acceptable_on_failure=false;
      audit.feasibility_tolerance=fit.feasibility_tolerance;audit.stationarity_tolerance=fit.stationarity_tolerance;
      audit.dual_infeasibility_tolerance=fit.dual_infeasibility_tolerance;audit.complementarity_tolerance=fit.complementarity_tolerance;
      const bool strict=audit_original_termination(store.original_problem(),audit,original)&&
          original.feasibility.total_inf<=audit.feasibility_tolerance&&
          original.dual_infeasibility_inf<=std::min(audit.stationarity_tolerance,audit.dual_infeasibility_tolerance)&&
          original.complementarity_inf<=audit.complementarity_tolerance;
      out.original_primal=original.feasibility.total_inf;out.original_dual=original.dual_infeasibility_inf;
      out.original_complementarity=original.complementarity_inf;
      if(strict&&native_finish_now()-began<allowance&&native_finish_now()<store.deadline) {
        // Keep PRIVATE vectors here. The normal owning wrapper lifts once on
        // return; original above was a disposable qualification copy only.
        candidate.termination_multiplier_source="native-same-execution-fixed-primal-full-original-copy-audited";
        selected=std::move(candidate);out.accepted=true;
      }
    }
  }
  out.seconds=native_finish_now()-began;
  if(level>0)std::cout<<"[NATIVE-ORIGINAL-ANCHOR-FINISH] attempted="<<out.attempted<<" accepted="<<out.accepted
      <<" private_fit_strict="<<out.private_strict<<" outer_charge="<<out.outer_charge<<" restoration_charge=0"
      <<" coordinate_sweeps="<<out.work.sweeps<<" columns="<<out.work.columns<<" nonzeros="<<out.work.nonzeros
      <<" factor_calls=0 direct_solve_calls=0 spectral_calls=0 assembly_seconds="<<out.work.assembly_seconds
      <<" coordinate_seconds="<<out.work.coordinate_seconds<<" audit_seconds="<<out.work.audit_seconds
      <<" phase_seconds="<<out.seconds<<" full_original_primal="<<out.original_primal<<" full_original_dual="<<out.original_dual
      <<" full_original_compl="<<out.original_complementarity<<" native_same_execution=1 external_endpoint=0"
      <<" x_bitwise_unchanged=1 private_optimization_tolerance_unchanged=1 full_original_audit_required=1"<<std::endl;
  return out;
}
} // namespace lazyipdca::qcqp::detail
