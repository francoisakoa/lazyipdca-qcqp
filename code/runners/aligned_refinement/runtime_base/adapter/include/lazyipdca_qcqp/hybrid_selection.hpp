#pragma once
#include "lazyipdca_qcqp/convexification_study.hpp"
#include "lazyipdca_qcqp/hybrid_spectrum.hpp"
#include <functional>

namespace lazyipdca::qcqp::study {
// Equations (12)--(20) of the supplied LazyIPDCA QP manuscript. Spectral
// quantities select candidates only; hybrid_certificate is the separate proof.
struct HybridSelectionSettings {
  int candidates=8,lanczos_steps=24,lanczos_starts=2,additions=2;
  double support_energy=.999999,strength=1,beta_offset=1e-6,negative_tolerance=1e-8;
  double scalar_safety=1.1,scalar_floor=.1;
  double score_activity=.5,score_scalar=.5,score_support=.1,score_density=.1,score_weight=.05;
  double min_lift=1e-4,min_activity=1e-2,min_score=1e-4;
};
struct HybridSelectionContext {
  int ambient=0;
  // Full-space mode: n_free=ambient, no unannounced active-face projection.
  std::vector<double> gradient,last_displacement;
};
struct HybridSelectionRecord {
  int pool_raw=0,pool_orthogonalized=0,pool_dependent_dropped=0,pool_nonnegative_dropped=0;
  int candidate=0,greedy_iteration=0,nnz=0,n=0,nfree=0,support_cap=0;
  double lambda0=0,lambda_plus=0,rayleigh=0,beta=0,alpha=0,chi=0;
  double l1_squared=0,retained_energy=0,lift=0,theta=0,sparsity=0,density=0;
  double rho_before=0,rho_limit=0,rho_scalar=0,residual_target=0,certified_residual=0,installed_residual=0,score=0;
  bool evaluated=false,admissible=false,selected=false,certificate=false,redundant=false,energy_cap_rejected=false,screened_before_certificate=false;
};
using HybridSelectionObserver=std::function<void(const HybridSelectionRecord&)>;

inline HybridAtom hybrid_sparsify_candidate(const std::vector<double>& vector,
    const std::vector<int>& support,int cap,double energy,double& retained) {
  if(vector.size()!=support.size()||cap<1||!(energy>0&&energy<=1))
    throw std::runtime_error("invalid hybrid sparsification input");
  std::vector<int> order(vector.size());std::iota(order.begin(),order.end(),0);
  long double total=0;for(double v:vector){if(!std::isfinite(v))throw std::runtime_error("nonfinite Ritz candidate");total+=(long double)v*v;}
  HybridAtom atom;retained=0;if(!(total>0))return atom;
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){return std::abs(vector[a])>std::abs(vector[b]);});
  long double norm=0;std::vector<std::pair<int,double>> entries;
  for(int j:order) {
    if(vector[j]==0)break;
    entries.push_back({support[j],vector[j]});norm+=(long double)vector[j]*vector[j];
    if(int(entries.size())>=cap||norm>=energy*total)break;
  }
  retained=double(norm/total);if(!(norm>0))return atom;
  std::sort(entries.begin(),entries.end());const double length=std::sqrt(double(norm));
  for(auto e:entries){atom.indices.push_back(e.first);atom.values.push_back(e.second/length);}
  return atom;
}
inline bool hybrid_redundant(const HybridAtom& candidate,const std::vector<HybridAtom>& retained) {
  for(const auto& previous:retained) {
    long double inner=0,norm=0;for(double v:previous.values)norm+=(long double)v*v;
    std::size_t a=0,b=0;
    while(a<candidate.indices.size()&&b<previous.indices.size()) {
      if(candidate.indices[a]==previous.indices[b])inner+=(long double)candidate.values[a++]*previous.values[b++];
      else if(candidate.indices[a]<previous.indices[b])++a;else ++b;
    }
    if(norm>0&&std::abs(inner)/std::sqrt(norm)>1-1e-10L)return true;
  }
  return false;
}
inline double hybrid_normalized_alignment(const HybridAtom& atom,const std::vector<double>& vector) {
  long double norm=0,inner=0;for(double v:vector){if(!std::isfinite(v))throw std::runtime_error("nonfinite selection context");norm+=(long double)v*v;}
  if(!(norm>0))return 0;
  for(std::size_t j=0;j<atom.indices.size();++j) {
    if(atom.indices[j]<0||std::size_t(atom.indices[j])>=vector.size())throw std::runtime_error("selection context dimension mismatch");
    inner+=(long double)atom.values[j]*vector[atom.indices[j]];
  }
  return double(std::abs(inner)/std::sqrt(norm));
}
inline int hybrid_select_enrich(Correction& correction,int ambient,int max_atoms,int support_cap,
    const HybridSelectionSettings& settings,const HybridSelectionContext& context,
    double margin=0,const HybridSelectionObserver& observer={}) {
  if(correction.family!="hybrid"||correction.original_convex||correction.support.empty()||
     int(correction.hybrid_atoms.size())>=max_atoms||settings.additions<=0)return 0;
  const auto positive=[](double v){return std::isfinite(v)&&v>0;};
  const auto nonnegative=[](double v){return std::isfinite(v)&&v>=0;};
  if(settings.candidates<1||settings.candidates>256||settings.additions>256||support_cap<1||
     !positive(settings.negative_tolerance)||!positive(settings.strength)||!positive(settings.beta_offset)||
     !(settings.scalar_safety>1)||!std::isfinite(settings.scalar_safety)||
     !nonnegative(settings.scalar_floor)||settings.scalar_floor>1||
     !positive(settings.support_energy)||settings.support_energy>1||
     !nonnegative(settings.score_activity)||!nonnegative(settings.score_scalar)||
     !nonnegative(settings.score_support)||!nonnegative(settings.score_density)||!nonnegative(settings.score_weight)||
     !nonnegative(settings.min_lift)||!nonnegative(settings.min_activity)||!nonnegative(settings.min_score))
    throw std::runtime_error("invalid hybrid composite-selector settings");
  if(ambient<=0||context.ambient!=ambient||context.gradient.size()!=std::size_t(ambient)||
     (!context.last_displacement.empty()&&context.last_displacement.size()!=std::size_t(ambient)))
    throw std::runtime_error("hybrid selector context dimension mismatch");
  auto spectrum=hybrid_spectrum(correction,settings.candidates,settings.lanczos_steps,
      settings.lanczos_starts,settings.negative_tolerance);
  const int pool_raw=spectrum.raw_candidates,pool_orthogonalized=spectrum.orthogonalized_candidates;
  const int pool_dependent_dropped=spectrum.dependent_dropped,pool_nonnegative_dropped=spectrum.nonnegative_projected_dropped;
  std::vector<HybridAtom> pool;std::vector<double> retention;
  for(const auto& vector:spectrum.vectors) {
    double retained=0;auto atom=hybrid_sparsify_candidate(vector,correction.support,
        support_cap,settings.support_energy,retained);
    if(!atom.indices.empty()){pool.push_back(std::move(atom));retention.push_back(retained);}
  }
  std::vector<bool> used(pool.size(),false);int inserted=0;
  for(int greedy=0;greedy<settings.additions&&int(correction.hybrid_atoms.size())<max_atoms;++greedy) {
    if(greedy)spectrum=hybrid_spectrum(correction,1,settings.lanczos_steps,settings.lanczos_starts,settings.negative_tolerance);
    // Outside structural support Q+H vanishes. Those zero ambient eigenvalues
    // matter to full-space lift, even though no search/storage is needed there.
    const double lambda0=correction.support.size()<std::size_t(ambient)?std::min(0.0,spectrum.lambda_min):spectrum.lambda_min;
    const double spectral_scale=std::max(settings.negative_tolerance,std::abs(std::min(lambda0,-settings.negative_tolerance)));
    const double rho_scalar=settings.scalar_safety*std::abs(std::min(0.0,lambda0));
    int best=-1;double best_score=-INFINITY;Correction best_metric;HybridSelectionRecord best_record;
    for(std::size_t index=0;index<pool.size();++index)if(!used[index]) {
      HybridSelectionRecord record;record.candidate=int(index);record.greedy_iteration=greedy;
      record.pool_raw=pool_raw;record.pool_orthogonalized=pool_orthogonalized;
      record.pool_dependent_dropped=pool_dependent_dropped;record.pool_nonnegative_dropped=pool_nonnegative_dropped;
      record.lambda0=lambda0;record.n=record.nfree=ambient;record.nnz=pool[index].indices.size();
      record.support_cap=support_cap;record.rho_before=correction.rho;record.rho_limit=std::max(correction.rho_limit,correction.rho);
      record.retained_energy=retention[index];record.rho_scalar=rho_scalar;
      record.energy_cap_rejected=record.retained_energy+16*std::numeric_limits<double>::epsilon()<settings.support_energy;
      if(record.energy_cap_rejected){if(observer)observer(record);continue;}
      record.redundant=hybrid_redundant(pool[index],correction.hybrid_atoms);
      auto atom=pool[index];std::vector<double> local(correction.support.size(),0);
      for(std::size_t j=0;j<atom.indices.size();++j)
        local[std::lower_bound(correction.support.begin(),correction.support.end(),atom.indices[j])-correction.support.begin()]=atom.values[j];
      auto product=hybrid_apply_support(correction,local);long double rayleigh=0;
      for(std::size_t j=0;j<local.size();++j)rayleigh+=(long double)local[j]*product[j];
      record.rayleigh=double(rayleigh);
      if(!(record.rayleigh<-settings.negative_tolerance)||record.redundant) {
        if(observer)observer(record);continue;
      }
      atom.weight=settings.strength*(-record.rayleigh+settings.beta_offset);record.beta=atom.weight;
      if(!(atom.weight>0)||!std::isfinite(atom.weight))throw std::runtime_error("nonfinite hybrid candidate strength");
      Correction trial=correction;trial.hybrid_atoms.push_back(atom);hybrid_assemble_trace(trial);
      record.lambda_plus=hybrid_spectrum(trial,1,settings.lanczos_steps,settings.lanczos_starts,settings.negative_tolerance).lambda_min;
      if(correction.support.size()<std::size_t(ambient))record.lambda_plus=std::min(0.0,record.lambda_plus);
      record.lift=std::max(0.0,record.lambda_plus-lambda0)/spectral_scale;
      record.alpha=std::max(hybrid_normalized_alignment(atom,context.gradient),
        context.last_displacement.empty()?0:hybrid_normalized_alignment(atom,context.last_displacement));
      record.chi=std::min(1.0,std::sqrt(double(ambient))*record.alpha);
      long double l1=0;for(double v:atom.values)l1+=std::abs(v);record.l1_squared=double(l1*l1);
      record.sparsity=double(record.nnz)/ambient;
      record.density=std::max(0.0,record.l1_squared-1)/std::max(1,record.nnz-1);
      const double spectral_residual=settings.scalar_safety*std::abs(std::min(0.0,record.lambda_plus));
      record.theta=rho_scalar>0?(spectral_residual+atom.weight*record.alpha*record.alpha)/rho_scalar:1;
      record.residual_target=std::max(spectral_residual,settings.scalar_floor*rho_scalar);
      record.score=record.lift*(1+settings.score_activity*record.chi)+
        settings.score_scalar*std::clamp(record.theta,0.0,1.0)-settings.score_support*record.sparsity*record.sparsity-
        settings.score_density*record.density-settings.score_weight*atom.weight/spectral_scale;
      // These quantities are selection estimates, not PSD proofs. If any
      // required selection test already fails, no model can be installed;
      // avoid exact PSD refinement of this rejected temporary candidate.
      if(record.lift<settings.min_lift||std::sqrt(double(ambient))*record.alpha<settings.min_activity||record.score<settings.min_score) {
        record.screened_before_certificate=true;
        if(observer)observer(record);
        continue;
      }
      record.certified_residual=hybrid_certified_residual(trial,correction.rho,margin);
      // Ritz and the nominal scalar floor select proposals; the installed
      // correction uses the separately certified residual, without restoring
      // an excess merely because the spectral proposal is conservative.
      trial.rho=record.certified_residual;
      record.installed_residual=trial.rho;record.certificate=hybrid_certificate(trial,trial.rho,margin);
      record.evaluated=true;
      // Atom enrichment must not force a scalar increase. Function PSD checks
      // and the original-step transaction remain separate obligations.
      record.admissible=record.certificate&&trial.rho<=correction.rho&&trial.rho<=record.rho_limit&&record.lift>=settings.min_lift&&
        std::sqrt(double(ambient))*record.alpha>=settings.min_activity&&record.score>=settings.min_score;
      if(observer)observer(record);
      if(record.admissible&&record.score>best_score) {
        best=int(index);best_score=record.score;best_metric=std::move(trial);best_record=record;
      }
    }
    if(best<0)break;
    used[best]=true;best_record.selected=true;if(observer)observer(best_record);
    best_metric.refined=true;best_metric.atom_batches=correction.atom_batches+1;
    best_metric.atom_certificate_checks=correction.atom_certificate_checks+1;
    best_metric.certificate="hybrid-composite-selector-fullspace-certified";
    best_metric.rank_upper=best_metric.rho>0?int(best_metric.support.size()):std::min(ambient,best_metric.atoms);
    correction=std::move(best_metric);++inserted;
  }
  return inserted;
}
struct HybridDownweightActivity {
  double denominator=0,penalty=0;bool active=false;
};
inline HybridDownweightActivity hybrid_downweight_activity(double beta,double projection,
    double scale,double multiplier,double objective_before,double objective_after,
    double epsilon,double dominance) {
  if(!std::isfinite(beta)||beta<0||!std::isfinite(projection)||!std::isfinite(scale)||scale<0||
     !std::isfinite(multiplier)||!std::isfinite(objective_before)||!std::isfinite(objective_after)||
     !(epsilon>0)||!std::isfinite(epsilon)||!(dominance>0)||!std::isfinite(dominance))
    throw std::runtime_error("invalid downweight activity input");
  HybridDownweightActivity out;
  out.denominator=std::max(objective_before-objective_after,epsilon*(1+std::abs(objective_after)));
  out.penalty=std::max(0.0,multiplier)*scale*beta*projection*projection;
  out.active=out.penalty>dominance*out.denominator;return out;
}
} // namespace lazyipdca::qcqp::study
