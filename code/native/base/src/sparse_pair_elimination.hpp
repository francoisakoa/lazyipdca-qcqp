#pragma once
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <iostream>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace lazyipdca::qcqp::detail {
// Congruence elimination of [h a; a 0] variable/equality pivots; optional
// negative equality diagonals use the exact [h a; a d] inverse, det < 0.
// Each pair contributes inertia (1,1,0). A stable structural layout is
// compiled once; subsequent matrices only scatter and update numeric slots.
// The fill envelope includes potential zeros, so a coefficient crossing zero
// cannot silently omit a Schur term. No dense null-space basis is formed.
class SparsePairElimination {
  struct Neighbor { int node,u_slot,v_slot; double u=0,v=0; };
  struct Update { int slot,i,j; };
  struct PairStep {
    int p,e,a_slot,h_slot,e_slot;
    double a=0,h=0,d=0,invpp=0,invpe=0,invee=0;
    std::vector<Neighbor> neighbors;
    std::vector<Update> updates;
  };
  std::vector<PairStep> steps_;
  std::vector<int> kept_,input_slots_,output_slots_;
  std::vector<std::pair<int,int>> input_pattern_;
  std::vector<SymmetricTriplet> output_pattern_;
  Vec values_;
  int dimension_=0,primal_=0,degree_budget_=0,fill_budget_=0;
  bool cached_=false,allow_regularized_=false;

  static std::pair<int,int> upper(int i,int j) {
    return {std::min(i,j),std::max(i,j)};
  }
  bool same_pattern(int n,int primal,const std::vector<SymmetricTriplet>& entries,
                    int degree_budget,int fill_budget,bool allow_regularized) const {
    if(!cached_ || n!=dimension_ || primal!=primal_ || degree_budget!=degree_budget_ ||
       fill_budget!=fill_budget_ || allow_regularized!=allow_regularized_ ||
       entries.size()!=input_pattern_.size())return false;
    for(std::size_t k=0;k<entries.size();++k)
      if(upper(entries[k].i,entries[k].j)!=input_pattern_[k])return false;
    return true;
  }
  bool load_step(PairStep& s) {
    s.a=values_[s.a_slot];s.h=values_[s.h_slot];s.d=values_[s.e_slot];
    double rowmax=std::max(std::abs(s.a),std::abs(s.d));
    for(auto& q:s.neighbors) {
      q.u=q.u_slot<0?0:values_[q.u_slot];
      q.v=q.v_slot<0?0:values_[q.v_slot];
      rowmax=std::max(rowmax,std::abs(q.v));
    }
    if(!(allow_regularized_?s.d<=0:s.d==0)||!std::isfinite(s.a)||!std::isfinite(s.h)||
       !std::isfinite(s.d)||!std::isfinite(rowmax)||std::abs(s.a)<1e-4*std::max(1.0,rowmax))return false;
    const double determinant=s.h*s.d-s.a*s.a;
    if(!(determinant<0)||!std::isfinite(determinant))return false;
    s.invpp=s.d/determinant;s.invpe=-s.a/determinant;s.invee=s.h/determinant;
    return std::isfinite(s.invpp)&&std::isfinite(s.invpe)&&std::isfinite(s.invee)&&
      std::max({std::abs(s.invpp),std::abs(s.invpe),std::abs(s.invee)})<=1e8;
  }
  static double change(const PairStep& s,int i,int j) {
    const auto& x=s.neighbors[i];const auto& y=s.neighbors[j];
    return s.invpp*x.u*y.u+s.invpe*(x.u*y.v+x.v*y.u)+s.invee*x.v*y.v;
  }
  bool update_step(const PairStep& s) {
    for(const auto& u:s.updates) {
      values_[u.slot]-=change(s,u.i,u.j);
      if(!std::isfinite(values_[u.slot]))return false;
    }
    return true;
  }
  std::vector<SymmetricTriplet> unreduced(const std::vector<SymmetricTriplet>& entries) {
    // A failed numeric pivot does not justify an approximate elimination.
    // Return the original matrix and identity reconstruction for this factor.
    ++numeric_fallbacks;cached_=false;steps_.clear();
    pairs=fill=0;reduced_primal=primal_;
    kept_.resize(dimension_);std::iota(kept_.begin(),kept_.end(),0);
    return entries;
  }
  std::vector<SymmetricTriplet> output() const {
    auto out=output_pattern_;
    for(std::size_t k=0;k<out.size();++k)out[k].value=values_[output_slots_[k]];
    return out;
  }
public:
  int pairs=0,fill=0,reduced_primal=0;
  int plan_builds=0,plan_reuses=0,plan_invalidations=0,numeric_fallbacks=0;

  std::vector<SymmetricTriplet> reduce(int n,int primal,const std::vector<SymmetricTriplet>& entries,
                                      int degree_budget=24,int fill_budget=64,bool allow_regularized=false) {
    if(same_pattern(n,primal,entries,degree_budget,fill_budget,allow_regularized)) {
      ++plan_reuses;
      std::fill(values_.begin(),values_.end(),0.0);
      for(std::size_t k=0;k<entries.size();++k)values_[input_slots_[k]]+=entries[k].value;
      for(auto& s:steps_)if(!load_step(s)||!update_step(s))return unreduced(entries);
      return output();
    }
    if(cached_)++plan_invalidations;
    ++plan_builds;
    dimension_=n;primal_=primal;degree_budget_=degree_budget;fill_budget_=fill_budget;
    allow_regularized_=allow_regularized;
    cached_=false;steps_.clear();kept_.clear();values_.clear();
    input_pattern_.clear();input_slots_.clear();output_pattern_.clear();output_slots_.clear();
    pairs=fill=0;reduced_primal=primal;
    std::vector<std::map<int,int>> rows(n);
    std::unordered_map<std::uint64_t,int> slots;
    slots.reserve(entries.size()*2+n);
    auto slot=[&](int i,int j) {
      auto [a,b]=upper(i,j);
      const auto key=(std::uint64_t(a)<<32)|std::uint32_t(b);
      auto it=slots.find(key);
      if(it!=slots.end())return it->second;
      int s=values_.size();slots.emplace(key,s);values_.push_back(0);
      rows[a][b]=s;rows[b][a]=s;return s;
    };
    std::vector<int> diagonal(n);
    for(int i=0;i<n;++i)diagonal[i]=slot(i,i);
    for(const auto& t:entries) {
      if(t.i<0||t.i>=n||t.j<0||t.j>=n)throw std::runtime_error("pair elimination index out of range");
      input_pattern_.push_back(upper(t.i,t.j));
      int s=slot(t.i,t.j);input_slots_.push_back(s);values_[s]+=t.value;
    }
    std::vector<unsigned char> alive(n,1),auxiliary(n,0);
    for(int j=primal;j<n;++j)if(values_[diagonal[j]]<0) {
      int primal_neighbors=0;
      for(const auto& [i,s]:rows[j])if(i<primal)++primal_neighbors;
      // Negative -1/gamma diagonals also occur on SPARSE equality normals.
      // Those columns remain part of the degree budget; only genuinely
      // dense auxiliaries (such as spectral atom borders) are exempt.
      auxiliary[j]=primal_neighbors>=std::max(3,primal/4);
    }
    auto degree=[&](int i) {
      int d=0;for(const auto& [j,s]:rows[i])if(!auxiliary[j])++d;return d;
    };
    const int total_budget=std::max(64,int(entries.size()/2));
    for(int e=primal;e<n;++e) {
      if(!alive[e] || !(allow_regularized?values_[diagonal[e]]<=0:values_[diagonal[e]]==0) ||
         n-2*(pairs+1)<1)continue;
      int best=-1,best_cost=degree_budget+1;
      double rowmax=0;for(const auto& [j,s]:rows[e])rowmax=std::max(rowmax,std::abs(values_[s]));
      for(const auto& [p,s]:rows[e])if(p<primal&&alive[p]&&
          std::abs(values_[s])>=1e-4*std::max(1.0,rowmax)) {
        // Dense negative-diagonal auxiliary borders already connect every
        // primal coordinate. They must not veto a sparse equality pivot;
        // their real updates and any new edges still count in the fill cap.
        int cost=degree(p)+degree(e);
        if(cost<best_cost){best=p;best_cost=cost;}
      }
      if(best<0)continue;
      int p=best;
      PairStep step{p,e,rows[e].at(p),diagonal[p],diagonal[e],0,0,0,0,0,0,{},{}};
      std::set<int> neighbors;
      for(const auto& [j,s]:rows[p])if(alive[j]&&j!=p&&j!=e)neighbors.insert(j);
      for(const auto& [j,s]:rows[e])if(alive[j]&&j!=p&&j!=e)neighbors.insert(j);
      for(int j:neighbors)step.neighbors.push_back({j,
        rows[p].count(j)?rows[p].at(j):-1,rows[e].count(j)?rows[e].at(j):-1});
      if(!load_step(step))continue;
      int extra=0;bool finite=true;
      const int available_fill=std::min(fill_budget,total_budget-fill);
      // Only pairs incident to a structural equality neighbor can change.
      // Retain an update even when its CURRENT numeric value is zero.
      for(int i=0;i<(int)step.neighbors.size()&&extra<=available_fill;++i)
        for(int j=i;j<(int)step.neighbors.size()&&extra<=available_fill;++j) {
        if(!allow_regularized&&step.neighbors[i].v_slot<0&&step.neighbors[j].v_slot<0)continue;
        int r=step.neighbors[i].node,c=step.neighbors[j].node;
        if(!rows[r].count(c))++extra;
        finite=finite&&std::isfinite(change(step,i,j));
      }
      if(!finite||extra>fill_budget||fill+extra>total_budget)continue;
      for(int i=0;i<(int)step.neighbors.size();++i)for(int j=i;j<(int)step.neighbors.size();++j) {
        if(!allow_regularized&&step.neighbors[i].v_slot<0&&step.neighbors[j].v_slot<0)continue;
        step.updates.push_back({slot(step.neighbors[i].node,step.neighbors[j].node),i,j});
      }
      if(!update_step(step))return unreduced(entries);
      for(int j:neighbors){rows[j].erase(p);rows[j].erase(e);}
      rows[p].clear();rows[e].clear();alive[p]=alive[e]=false;
      steps_.push_back(std::move(step));++pairs;--reduced_primal;fill+=extra;
    }
    std::vector<int> index(n,-1);
    for(int i=0;i<n;++i)if(alive[i]){index[i]=kept_.size();kept_.push_back(i);}
    for(int i:kept_)for(const auto& [j,s]:rows[i])if(j>=i&&alive[j]) {
      output_pattern_.push_back({index[i],index[j],0});output_slots_.push_back(s);
    }
    cached_=true;
    return output();
  }
  int dimension()const{return kept_.size();}
  int original_coordinate(int coordinate)const {
    return coordinate>=0&&coordinate<(int)kept_.size()?kept_[coordinate]:-1;
  }
  template<class Solve> bool solve_with(Vec& rhs,std::string& message,Solve solve)const {
    if((int)rhs.size()!=dimension_)return false;
    Vec b=rhs;
    for(const auto& s:steps_)for(const auto& q:s.neighbors)
      b[q.node]-=q.u*(s.invpp*b[s.p]+s.invpe*b[s.e])+q.v*(s.invpe*b[s.p]+s.invee*b[s.e]);
    Vec r; r.reserve(kept_.size());for(int i:kept_)r.push_back(b[i]);
    if(!solve(r,message))return false;
    rhs.assign(dimension_,0);for(int k=0;k<(int)kept_.size();++k)rhs[kept_[k]]=r[k];
    for(auto it=steps_.rbegin();it!=steps_.rend();++it) {
      double bp=b[it->p],be=b[it->e];
      for(const auto& q:it->neighbors){bp-=q.u*rhs[q.node];be-=q.v*rhs[q.node];}
      rhs[it->p]=it->invpp*bp+it->invpe*be;rhs[it->e]=it->invpe*bp+it->invee*be;
    }
    return true;
  }
  bool solve(SymmetricFactorization& factor,Vec& rhs,std::string& message)const {
    return solve_with(rhs,message,[&](Vec& b,std::string& m){return factor.solve(b,m);});
  }
};

// Adapter for the existing symmetric-factorization contract. Inertia is
// preserved up to one positive and one negative eigenvalue per eliminated
// pair. Negative equality completion diagonals are allowed here, so the same
// algebra also applies to curvature certificates with -I/gamma normal blocks.
class PairReducedFactorization final : public SymmetricFactorization {
public:
  using Factory=std::function<std::unique_ptr<SymmetricFactorization>(std::string&)>;
private:
  using Clock=std::chrono::steady_clock;
  std::unique_ptr<SymmetricFactorization> base_,full_;
  SymmetricFactorization* current_=nullptr;
  Factory full_factory_;
  SparsePairElimination elimination_;
  bool reduced_=false,verbose_=false;
  int backend_plan_build_=-1;
  int retired_analyses_=0,retired_reuses_=0,retired_invalidations_=0;
  int retired_scatter_builds_=0,retired_scatter_reuses_=0,retired_scatter_invalidations_=0;
  inline static std::atomic<int> next_id_{0};
  int id_=++next_id_;
  static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
  }
  bool prepare_reduced_backend(std::string& message) {
    if(backend_plan_build_<0||backend_plan_build_==elimination_.plan_builds)return true;
    // A reduced coordinate number is meaningful only within its elimination
    // plan. Unioning unrelated graphs merely because their dimensions agree
    // can turn a sparse cached envelope into a nearly dense one.
    if(!full_factory_)return false;
    auto replacement=full_factory_(message);
    if(!replacement)return false;
    retired_analyses_+=base_->symbolic_analyses();
    retired_reuses_+=base_->symbolic_reuses();
    retired_invalidations_+=base_->symbolic_invalidations();
    retired_scatter_builds_+=base_->numeric_scatter_builds();
    retired_scatter_reuses_+=base_->numeric_scatter_reuses();
    retired_scatter_invalidations_+=base_->numeric_scatter_invalidations();
    base_=std::move(replacement);current_=base_.get();++reduced_backend_resets;
    return true;
  }
public:
  int factor_calls=0,reduced_factor_calls=0,full_factor_calls=0,minimum_dimension=0;
  int reduced_backend_resets=0;
  double reduction_seconds=0,reduced_factor_seconds=0,full_factor_seconds=0;
  explicit PairReducedFactorization(std::unique_ptr<SymmetricFactorization> base,
                                   Factory full_factory={},bool verbose=false)
    :base_(std::move(base)),current_(base_.get()),full_factory_(std::move(full_factory)),verbose_(verbose){}
  int underlying_factorizations()const{return reduced_factor_calls+full_factor_calls;}
  bool factor(int n,int primal,const std::vector<SymmetricTriplet>& entries,std::string& message)override {
    const int underlying_before=underlying_factorizations();
    ++factor_calls;
    reduced_=false;current_=base_.get();
    auto started=Clock::now();
    auto matrix=elimination_.reduce(n,primal,entries,24,256,true);
    const double reduction=seconds(started);
    reduction_seconds+=reduction;
    if(!minimum_dimension||elimination_.dimension()<minimum_dimension)minimum_dimension=elimination_.dimension();
    double reduced_factor=0,full_factor=0;bool ok=false;
    if(elimination_.pairs) {
      started=Clock::now();
      if(prepare_reduced_backend(message)) {
        ++reduced_factor_calls;backend_plan_build_=elimination_.plan_builds;
        ok=base_->factor(elimination_.dimension(),elimination_.reduced_primal,matrix,message);
      }
      reduced_factor=seconds(started);reduced_factor_seconds+=reduced_factor;
      if(ok)reduced_=true;
      // Failed reduced inertia is checked again on the original matrix.
      // A separate persistent backend keeps both symbolic layouts available.
    }
    if(!ok) {
      ++full_factor_calls;started=Clock::now();
      if(!full_&&full_factory_)full_=full_factory_(message);
      current_=full_?full_.get():base_.get();
      ok=current_->factor(n,primal,entries,message);
      full_factor=seconds(started);full_factor_seconds+=full_factor;
    }
    if(verbose_)std::cout<<"[SPARSE-PAIR] id="<<id_<<" call="<<factor_calls
      <<" dim="<<n<<" primal="<<primal<<" reduced="<<elimination_.dimension()
      <<" min_dim="<<minimum_dimension<<" nnz="<<entries.size()<<" reduced_nnz="<<matrix.size()
      <<" pairs="<<elimination_.pairs<<" fill="<<elimination_.fill
      <<" builds="<<elimination_.plan_builds<<" reuses="<<elimination_.plan_reuses
      <<" invalidations="<<elimination_.plan_invalidations<<" numeric_fallbacks="<<elimination_.numeric_fallbacks
      <<" reduced_calls="<<reduced_factor_calls<<" full_calls="<<full_factor_calls
      <<" actual_calls="<<underlying_factorizations()<<" actual_this="<<underlying_factorizations()-underlying_before
      <<" backend_resets="<<reduced_backend_resets
      <<" reduce_s="<<reduction<<" factor_reduced_s="<<reduced_factor<<" factor_full_s="<<full_factor
      <<" total_reduce_s="<<reduction_seconds<<" total_factor_reduced_s="<<reduced_factor_seconds
      <<" total_factor_full_s="<<full_factor_seconds<<" separate_full="<<bool(full_)
      <<" using_reduced="<<reduced_<<" ok="<<ok<<'\n';
    return ok;
  }
  bool solve(Vec& rhs,std::string& message)const override {
    return reduced_?elimination_.solve(*current_,rhs,message):current_->solve(rhs,message);
  }
  bool solve_curvature_probe(Vec& rhs,std::string& message)const override {
    return reduced_?elimination_.solve_with(rhs,message,[&](Vec& b,std::string& m){
      return current_->solve_curvature_probe(b,m);
    }):current_->solve_curvature_probe(rhs,message);
  }
  int curvature_defect()const override{return current_->curvature_defect();}
  int failed_pivot_coordinate()const override {
    return reduced_?elimination_.original_coordinate(current_->failed_pivot_coordinate()):current_->failed_pivot_coordinate();
  }
  double last_backward_error()const override{return current_->last_backward_error();}
  double last_initial_backward_error()const override{return current_->last_initial_backward_error();}
  double last_residual_inf()const override{return current_->last_residual_inf();}
  int last_refinement_steps()const override{return current_->last_refinement_steps();}
  // Every eliminated nonsingular indefinite 2x2 block contributes (1,1,0).
  // Diagnostics must refer to the caller's ORIGINAL matrix, not its Schur
  // complement. Preserve unknown (-1) and never add pairs to a full fallback.
  int last_positive_pivots()const override{
    const int count=current_->last_positive_pivots();
    return reduced_&&count>=0?count+elimination_.pairs:count;
  }
  int last_negative_pivots()const override{
    const int count=current_->last_negative_pivots();
    return reduced_&&count>=0?count+elimination_.pairs:count;
  }
  int last_zero_pivots()const override{return current_->last_zero_pivots();}
  double minimum_abs_pivot()const override{return current_->minimum_abs_pivot();}
  double maximum_abs_pivot()const override{return current_->maximum_abs_pivot();}
  int minimum_abs_pivot_coordinate()const override {
    return reduced_?elimination_.original_coordinate(current_->minimum_abs_pivot_coordinate())
                   :current_->minimum_abs_pivot_coordinate();
  }
  double last_primal_regularization()const override{return current_->last_primal_regularization();}
  double last_dual_regularization()const override{return current_->last_dual_regularization();}
  int last_regularization_attempts()const override{return current_->last_regularization_attempts();}
  int symbolic_analyses()const override{return retired_analyses_+base_->symbolic_analyses()+(full_?full_->symbolic_analyses():0);}
  int symbolic_reuses()const override{return retired_reuses_+base_->symbolic_reuses()+(full_?full_->symbolic_reuses():0);}
  int symbolic_invalidations()const override{return retired_invalidations_+base_->symbolic_invalidations()+(full_?full_->symbolic_invalidations():0);}
  bool last_symbolic_reused()const override{return current_->last_symbolic_reused();}
  bool has_symbolic_cache()const override{return current_->has_symbolic_cache();}
  int numeric_scatter_builds()const override{return retired_scatter_builds_+base_->numeric_scatter_builds()+(full_?full_->numeric_scatter_builds():0);}
  int numeric_scatter_reuses()const override{return retired_scatter_reuses_+base_->numeric_scatter_reuses()+(full_?full_->numeric_scatter_reuses():0);}
  int numeric_scatter_invalidations()const override{return retired_scatter_invalidations_+base_->numeric_scatter_invalidations()+(full_?full_->numeric_scatter_invalidations():0);}
  bool last_numeric_scatter_reused()const override{return current_->last_numeric_scatter_reused();}
  const char* name()const override{return base_->name();}
  const SparsePairElimination& elimination()const{return elimination_;}
};

inline std::unique_ptr<SymmetricFactorization> with_sparse_pair_elimination(
    std::unique_ptr<SymmetricFactorization> base,
    PairReducedFactorization::Factory full_factory={},bool verbose=false) {
  if(!base)return nullptr;
  return std::make_unique<PairReducedFactorization>(std::move(base),std::move(full_factory),verbose);
}
}
