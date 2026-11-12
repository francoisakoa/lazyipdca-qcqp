#pragma once
#include "lazyipdca_qcqp/hybrid_selection.hpp"

namespace lazyipdca::qcqp::study {
// Representation changes are not curvature changes. Compare the actual
// rank-one sums and support scalar exactly over the supplied binary doubles,
// without rounding away small but genuine changes or allocating ambient n^2.
inline bool same_effective_correction(const Correction& a,const Correction& b) {
  if(a.hybrid_atoms.empty()&&b.hybrid_atoms.empty()&&a.support==b.support)
    return a.rho==b.rho;
  auto difference=hybrid_exact_gram(a);
  for(const auto& entry:hybrid_exact_gram(b))difference[entry.first]-=entry.second;
  for(int j:a.support)difference[{j,j}]+=Rational(a.rho);
  for(int j:b.support)difference[{j,j}]-=Rational(b.rho);
  for(const auto& entry:difference)if(entry.second!=0)return false;
  return true;
}
// If Q+C=0 exactly, any other globally convexifying D satisfies
// D-C=Q+D >= 0. Thus C is already the Loewner-minimal correction;
// neither extra atoms nor a scalar/atom re-encoding can reduce it.
inline bool correction_cancels_original(const Correction& c) {
  auto model=hybrid_exact_gram(c);
  for(int j:c.support)model[{j,j}]+=Rational(c.rho);
  for(const auto& e:c.original)
    model[{std::min(e.i,e.j),std::max(e.i,e.j)}]+=Rational(e.v);
  for(const auto& entry:model)if(entry.second!=0)return false;
  return true;
}
// Recovery is distinct from the ordinary per-atom composite score. If the
// least eigenvalue is multiple, one useful rank-one correction cannot lift
// that eigenvalue: certify the proposed negative-subspace batch as a whole.
struct HybridRecoveryRecord {
  int pool_raw=0,pool_orthogonalized=0,pool_dependent_dropped=0,pool_nonnegative_dropped=0;
  int negative_candidates=0,added=0;
  double rho_before=0,rho_after=0,scalar_baseline=0;
  double lambda0=0,lambda_plus=0,lift=0,alpha=0,chi=0,theta=0,score=0;
  double support_cost=0,density_cost=0,weight_cost=0;
  bool certificate=false,improved=false,selection_screened=false;
};
inline int hybrid_recover_negative_subspace(Correction& correction,int ambient,
    int max_atoms,int support_cap,const HybridSelectionSettings& settings,
    double margin,HybridRecoveryRecord& record,const HybridSelectionContext* context=nullptr) {
  record={};record.rho_before=record.rho_after=correction.rho;
  if(correction.family!="hybrid"||correction.original_convex||correction.support.empty()||
     max_atoms<=int(correction.hybrid_atoms.size())||settings.additions<=0)return 0;
  auto spectrum=hybrid_spectrum(correction,settings.candidates,
      settings.lanczos_steps,settings.lanczos_starts,settings.negative_tolerance);
  record.negative_candidates=spectrum.vectors.size();
  record.pool_raw=spectrum.raw_candidates;record.pool_orthogonalized=spectrum.orthogonalized_candidates;
  record.pool_dependent_dropped=spectrum.dependent_dropped;record.pool_nonnegative_dropped=spectrum.nonnegative_projected_dropped;
  record.lambda0=spectrum.lambda_min;
  if(correction.support.size()<std::size_t(ambient))record.lambda0=std::min(0.0,record.lambda0);
  if(context&&(context->ambient!=ambient||context->gradient.size()!=std::size_t(ambient)||
     (!context->last_displacement.empty()&&context->last_displacement.size()!=std::size_t(ambient))))
    throw std::runtime_error("hybrid batch selection context dimension mismatch");
  Correction trial=correction;
  const int batch_limit=std::min(max_atoms,int(correction.hybrid_atoms.size())+settings.additions);
  for(const auto& vector:spectrum.vectors) {
    if(int(trial.hybrid_atoms.size())>=batch_limit)break;
    double retained=0;
    auto atom=hybrid_sparsify_candidate(vector,correction.support,support_cap,
        settings.support_energy,retained);
    if(atom.indices.empty()||retained+16*std::numeric_limits<double>::epsilon()<settings.support_energy||
       hybrid_redundant(atom,trial.hybrid_atoms))continue;
    std::vector<double> local(correction.support.size(),0);
    long double squared_norm=0;
    for(std::size_t j=0;j<atom.indices.size();++j) {
      local[std::lower_bound(correction.support.begin(),correction.support.end(),atom.indices[j])-correction.support.begin()]=atom.values[j];
      squared_norm+=(long double)atom.values[j]*atom.values[j];
    }
    const auto product=hybrid_apply_support(trial,local);
    long double rayleigh=0;for(std::size_t j=0;j<local.size();++j)rayleigh+=(long double)local[j]*product[j];
    if(!(rayleigh < -settings.negative_tolerance)||!(squared_norm>0))continue;
    // No heuristic positive padding: independent exact certification below
    // supplies any necessary residual scalar. The atom itself stays PSD.
    atom.weight=double(-rayleigh/(squared_norm*squared_norm));
    if(!(atom.weight>0)||!std::isfinite(atom.weight))continue;
    trial.hybrid_atoms.push_back(std::move(atom));++record.added;
  }
  if(!record.added)return 0;
  record.lambda_plus=hybrid_spectrum(trial,1,settings.lanczos_steps,
      settings.lanczos_starts,settings.negative_tolerance).lambda_min;
  if(correction.support.size()<std::size_t(ambient))record.lambda_plus=std::min(0.0,record.lambda_plus);
  const double scale=std::max(settings.negative_tolerance,std::abs(std::min(record.lambda0,-settings.negative_tolerance)));
  record.lift=std::max(0.0,record.lambda_plus-record.lambda0)/scale;
  if(context) {
    long double weighted_activity=0,total_weight=0;
    for(std::size_t k=correction.hybrid_atoms.size();k<trial.hybrid_atoms.size();++k) {
      const auto& atom=trial.hybrid_atoms[k];
      const double activity=std::max(hybrid_normalized_alignment(atom,context->gradient),
        context->last_displacement.empty()?0:hybrid_normalized_alignment(atom,context->last_displacement));
      weighted_activity+=(long double)atom.weight*activity*activity;total_weight+=atom.weight;
      const double sparsity=double(atom.indices.size())/ambient;
      long double l1=0;for(double value:atom.values)l1+=std::abs(value);
      record.support_cost+=sparsity*sparsity;
      record.density_cost+=std::max(0.0,double(l1*l1)-1)/std::max(1,int(atom.indices.size())-1);
      record.weight_cost+=atom.weight/scale;
    }
    record.alpha=total_weight>0?std::sqrt(double(weighted_activity/total_weight)):0;
    record.chi=std::min(1.0,std::sqrt(double(ambient))*record.alpha);
    const double rho_scalar=settings.scalar_safety*std::abs(std::min(0.0,record.lambda0));
    const double spectral_residual=settings.scalar_safety*std::abs(std::min(0.0,record.lambda_plus));
    record.theta=rho_scalar>0?(spectral_residual+double(weighted_activity))/rho_scalar:1;
    record.score=record.lift*(1+settings.score_activity*record.chi)+
      settings.score_scalar*std::clamp(record.theta,0.0,1.0)-
      settings.score_support*record.support_cost-settings.score_density*record.density_cost-
      settings.score_weight*record.weight_cost;
    record.selection_screened=record.lift<settings.min_lift||
      std::sqrt(double(ambient))*record.alpha<settings.min_activity||record.score<settings.min_score;
    if(record.selection_screened)return 0;
  }
  Correction scalar_only=correction;
  record.scalar_baseline=hybrid_certified_residual(scalar_only,correction.rho,margin);
  trial.rho=hybrid_certified_residual(trial,correction.rho,margin);
  record.rho_after=trial.rho;record.certificate=hybrid_certificate(trial,trial.rho,margin);
  record.improved=record.certificate&&trial.rho<record.scalar_baseline*(1-1e-10);
  if(!record.improved)return 0;
  trial.refined=true;trial.atom_batches=correction.atom_batches+1;
  trial.atom_certificate_checks=correction.atom_certificate_checks+1;
  trial.certificate=context?"hybrid-negative-subspace-composite-batch-exact-certificate":"hybrid-negative-subspace-recovery-exact-certificate";
  trial.rank_upper=trial.rho>0?int(trial.support.size()):std::min(ambient,trial.atoms);
  correction=std::move(trial);return record.added;
}
} // namespace lazyipdca::qcqp::study
