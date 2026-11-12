#include "lazyipdca_qcqp/hybrid_recovery.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;
int main(){
 int checks=0;auto check=[&](bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);};
 study::HybridSelectionSettings settings;study::HybridRecoveryRecord report;
 auto tied=study::construct_hybrid(2,{{0,0,-2},{1,1,-2}}),original=tied;
 study::HybridSelectionContext context;context.ambient=2;context.gradient={1,1};
 check(study::hybrid_select_enrich(tied,2,2,2,settings,context)==0,"control did not reproduce tied-minimum screening");
 check(study::hybrid_recover_negative_subspace(tied,2,2,2,settings,0,report)==2,"negative eigenspace batch was not constructed");
 check(tied.rho==0&&tied.hybrid_atoms.size()==2&&tied.rank_upper==2,"rank-two batch did not remove the scalar deficit");
 check(report.improved&&report.certificate&&study::hybrid_certificate(tied,tied.rho),"batch is not independently PSD certified");
 // A rank-two representation of 2 I is not a new function metric.
 check(study::correction_cancels_original(original)&&study::correction_cancels_original(tied),
     "exact isotropic cancellation did not prove minimal correction");
 check(study::same_effective_correction(original,tied),"isotropic scalar-to-atoms equivalence not detected");
 auto changed=tied;changed.hybrid_atoms[0].weight=std::nextafter(changed.hybrid_atoms[0].weight,INFINITY);
 check(!study::correction_cancels_original(changed),"one-ulp nonzero model was erased by cancellation proof");
 check(!study::same_effective_correction(original,changed),"one-ulp genuine curvature change was rounded away");
 auto reordered=tied;std::reverse(reordered.hybrid_atoms.begin(),reordered.hybrid_atoms.end());
 check(study::same_effective_correction(tied,reordered),"atom order changed effective-correction comparison");
 // Realification of one complex channel h=(1+i,2-i). Its negative rank-two
 // Gram matrix has two equal eigenvalues -7 and two zero eigenvalues.
 std::vector<double> u={1,2,1,-1},v={-1,1,1,2};std::vector<SymTriplet> q;
 for(int i=0;i<4;++i)for(int j=i;j<4;++j){double a=-u[i]*u[j]-v[i]*v[j];if(a)q.push_back({i,j,a});}
 auto channel=study::construct_hybrid(4,q);const auto channel_initial=channel;const double before=channel.rho;
 check(study::hybrid_recover_negative_subspace(channel,4,2,4,settings,0,report)==2,"complex-channel rank-two batch missing");
 check(!study::same_effective_correction(channel_initial,channel),"rank-two correction on four-dimensional support was mistaken for scalar I");
 check(channel.rho<before*1e-10&&study::hybrid_certificate(channel,channel.rho),"complex-channel batch failed its full-space certificate");
 for(const auto& atom:channel.hybrid_atoms)check(atom.weight>=0,"negative atom weight");
 auto partial_channel=channel_initial;
 check(study::hybrid_recover_negative_subspace(partial_channel,4,1,4,settings,0,report)==0,"loose initial Gershgorin bound admitted a useless partial batch");
 check(partial_channel.hybrid_atoms.empty()&&partial_channel.rho==channel_initial.rho,"rejected partial channel batch mutated the metric");
 auto convex=study::construct_hybrid(2,{{0,0,1},{0,1,2},{1,1,4}}),saved=convex;
 check(study::hybrid_recover_negative_subspace(convex,2,2,2,settings,0,report)==0,"convex source was enriched");
 check(convex.rho==0&&convex.hybrid_atoms.empty(),"convex source changed");
 auto scalar=study::construct_support_scalar(2,{{0,0,-2},{1,1,-2}});
 check(study::hybrid_recover_negative_subspace(scalar,2,2,2,settings,0,report)==0&&scalar.hybrid_atoms.empty(),"scalar family silently became hybrid");
 auto limited=original;
 check(study::hybrid_recover_negative_subspace(limited,2,1,2,settings,0,report)==0,"one atom falsely lifted a double least eigenvalue");
 check(limited.rho==original.rho&&limited.hybrid_atoms.empty(),"failed limited batch mutated the accepted metric");
 auto normal=original;
 check(study::hybrid_recover_negative_subspace(normal,2,2,2,settings,0,report,&context)==2,"normal block score rejected the joint spectral lift");
 check(!report.selection_screened&&report.score>=settings.min_score&&report.certificate,"normal batch bypassed score or certificate gates");
 study::HybridSelectionContext orthogonal;orthogonal.ambient=4;orthogonal.gradient={3,0,-1,2};
 auto inactive=channel_initial;
 check(study::hybrid_recover_negative_subspace(inactive,4,2,4,settings,0,report,&orthogonal)==0&&report.selection_screened,"normal batch ignored the activity threshold");
 check(inactive.rho==channel_initial.rho&&inactive.hybrid_atoms.empty(),"screened normal batch changed the metric");
 auto threshold=original;settings.min_lift=2;
 check(study::hybrid_recover_negative_subspace(threshold,2,2,2,settings,0,report,&context)==0&&report.selection_screened,"normal batch ignored the lift threshold");
 settings.min_lift=1e-4;settings.min_score=1e6;
 check(study::hybrid_recover_negative_subspace(threshold,2,2,2,settings,0,report,&context)==0&&report.selection_screened,"normal batch ignored the score threshold");
 settings.min_score=1e-4;
 settings.additions=1;limited=original;
 check(study::hybrid_recover_negative_subspace(limited,2,2,2,settings,0,report)==0,"per-event addition budget was ignored");
 settings.additions=0;
 check(study::hybrid_recover_negative_subspace(limited,2,2,2,settings,0,report)==0,"zero-addition ablation was ignored");
 std::cout<<"hybrid_recovery PASS checks="<<checks<<" repeated_minimum=1 complex_rank_two=1 convex_C0=1 scalar_family_preserved=1 budgets=1 failed_trial_unchanged=1\n";
}
