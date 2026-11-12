#include "lazyipdca_qcqp/hybrid_selection.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main() {
 try {
  study::HybridSelectionSettings settings;
  study::HybridSelectionContext context;context.ambient=3;context.gradient={1,1,0};
  auto c=study::construct_hybrid(3,{{0,0,-4},{1,1,-1},{2,2,2}});c.rho=4.4;
  std::vector<study::HybridSelectionRecord> records;
  const int added=study::hybrid_select_enrich(c,3,8,3,settings,context,0,
    [&](const auto& record){records.push_back(record);});
  require(added==2&&c.hybrid_atoms.size()==2,"finite-pool greedy selector missed two negative directions");
  require(study::hybrid_certificate(c,c.rho),"composite selector bypassed full-space certificate");
  std::vector<study::HybridSelectionRecord> selected;
  for(const auto& r:records)if(r.selected) {
   selected.push_back(r);
   const double scale=std::max(settings.negative_tolerance,std::abs(std::min(r.lambda0,-settings.negative_tolerance)));
   const double score=r.lift*(1+settings.score_activity*r.chi)+settings.score_scalar*std::clamp(r.theta,0.,1.)-
     settings.score_support*r.sparsity*r.sparsity-settings.score_density*r.density-settings.score_weight*r.beta/scale;
   require(std::abs(r.score-score)<1e-12,"composite score does not follow equation 18");
   require(r.installed_residual==r.certified_residual&&r.residual_target>=settings.scalar_floor*r.rho_scalar,
     "H4 installed scalar must come from the certificate, not the spectral proposal floor");
   require(r.admissible&&r.certificate&&r.alpha>0,"uncertified or inadmissible candidate selected");
  }
  require(selected.size()==2&&selected[0].greedy_iteration==0&&selected[1].greedy_iteration==1,
    "greedy iteration accounting missing");
  require(std::abs(selected[0].lambda0+4)<1e-12&&std::abs(selected[1].lambda0+1)<1e-12,
    "remaining candidate was not rescored after first insertion");
  double energy=0;
  auto sparse=study::hybrid_sparsify_candidate({.8,.6,0},{0,1,2},3,.6,energy);
  require(sparse.indices==std::vector<int>({0})&&std::abs(energy-.64)<1e-12,"smallest energy-carrying support not selected");
  sparse=study::hybrid_sparsify_candidate({.8,.6,0},{0,1,2},1,.99,energy);
  require(sparse.indices.size()==1&&energy<.99,"support cap did not limit energy retention");
  auto inactive=study::construct_hybrid(3,{{0,0,-4},{1,1,-1},{2,2,2}});inactive.rho=4.4;
  auto no_activity=context;no_activity.gradient={0,0,1};
  bool activity_screened=false;
  require(study::hybrid_select_enrich(inactive,3,8,3,settings,no_activity,0,[&](const auto& record){
    activity_screened|=record.screened_before_certificate&&!record.evaluated&&!record.certificate;
  })==0&&inactive.hybrid_atoms.empty()&&activity_screened,
    "activity admissibility threshold ignored");
  auto score_reject=settings;score_reject.min_score=100;
  bool score_screened=false;
  require(study::hybrid_select_enrich(inactive,3,8,3,score_reject,context,0,[&](const auto& record){
    score_screened|=record.screened_before_certificate&&!record.evaluated&&!record.certificate;
  })==0&&score_screened,"score admissibility threshold ignored");
  auto repeated=study::construct_hybrid(3,{{0,0,-4},{1,1,-4},{2,2,2}});
  bool lift_screened=false;
  require(study::hybrid_select_enrich(repeated,3,8,3,settings,context,0,[&](const auto& record){
    lift_screened|=record.screened_before_certificate&&record.lift<settings.min_lift&&!record.certificate;
  })==0&&repeated.hybrid_atoms.empty()&&lift_screened,"zero-lift candidate was unnecessarily certified");
  auto one=inactive;
  require(study::hybrid_select_enrich(one,3,1,3,settings,context)==1&&one.hybrid_atoms.size()==1,"global/per-function atom limit ignored");
  auto scalar=study::construct_support_scalar(3,{{0,0,-4},{1,1,-1},{2,2,2}});
  require(study::hybrid_select_enrich(scalar,3,8,3,settings,context)==0&&scalar.hybrid_atoms.empty(),"S role received selector atoms");
  study::HybridSelectionContext rotated_context;rotated_context.ambient=2;rotated_context.gradient={1,-1};
  auto rotated=study::construct_hybrid(2,{{0,0,1},{0,1,2},{1,1,1}});rotated.rho=1.1;
  bool cap_rejected=false;
  require(study::hybrid_select_enrich(rotated,2,8,1,settings,rotated_context,0,[&](const auto& record){cap_rejected|=record.energy_cap_rejected;})==0&&cap_rejected,
    "sparsified direction accepted after losing negative curvature");
  auto increase=study::construct_hybrid(3,{{0,0,-4},{1,1,-1},{2,2,2}});increase.rho=4;increase.rho_limit=4.4;
  auto weak=settings;weak.strength=.01;weak.additions=1;
  require(study::hybrid_select_enrich(increase,3,8,3,weak,context)==1&&increase.rho<4&&
    study::hybrid_certificate(increase,increase.rho),"heuristic spectral floor restored certified scalar excess");
  study::HybridAtom redundant;redundant.indices={0};redundant.values={-1};
  study::HybridAtom previous;previous.indices={0};previous.values={1};
  require(study::hybrid_redundant(redundant,{previous}),"parallel opposite-sign candidate not recognized as redundant");
  auto partial=study::construct_hybrid(3,{{0,0,1},{0,2,2},{2,2,1}});partial.rho=1.1;
  study::HybridSelectionContext partial_context;partial_context.ambient=3;partial_context.gradient={1,0,-1};
  auto one_addition=settings;one_addition.additions=1;bool zero_ambient_floor=false;
  require(study::hybrid_select_enrich(partial,3,1,3,one_addition,partial_context,0,[&](const auto& record){
    if(record.selected)zero_ambient_floor=record.lambda_plus==0;
  })==1&&zero_ambient_floor,"omitted zero ambient eigenvalues overstated full-space lift");
  bool invalid=false;auto bad=settings;bad.negative_tolerance=0;
  try{study::hybrid_select_enrich(inactive,3,8,3,bad,context);}catch(const std::runtime_error&){invalid=true;}
  require(invalid,"unsafe zero spectral tolerance was accepted");
  const auto fall=study::hybrid_downweight_activity(2,.1,1,1,10,9,1e-12,.5);
  const auto rise=study::hybrid_downweight_activity(2,.1,1,1,10,11,1e-12,.5);
  require(fall.denominator==1&&!fall.active,"paper objective decrease denominator wrong");
  require(std::abs(rise.denominator-12e-12)<1e-25&&rise.active,"objective increase incorrectly counted as progress");
  const auto zero_dual=study::hybrid_downweight_activity(2,.1,1,0,10,11,1e-12,.5);
  require(zero_dual.penalty==0&&!zero_dual.active,"inactive constraint dual penalty incorrectly active");
  std::cout<<"hybrid_composite_selection PASS pool=2 greedy_rescore=1 energy_cap=1 activity_rejection=1 score_rejection=1 certified_scalar_without_spectral_floor=1 mixed_scalar_guard=1 ambient_zero_floor=1 paper_downweight=1\n";
  return 0;
 }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
