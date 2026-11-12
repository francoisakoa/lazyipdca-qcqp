#include "lazyipdca_qcqp/hybrid_selection.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main() {
  int checks=0;
  auto require=[&](bool b,const char* m){++checks;if(!b)throw std::runtime_error(m);};
  const std::vector<std::vector<SymTriplet>> psd={
    {},{{0,0,0}},{{0,0,1},{1,1,1}},{{0,0,1},{0,1,2},{1,1,4}}
  };
  for(const auto& h:std::vector<std::vector<SymTriplet>>{{},{{0,0,1}}})
    for(double margin:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
      bool rejected=false;
      try{study::construct_hybrid(3,h,margin);}catch(const std::runtime_error&){rejected=true;}
      require(rejected,"PSD early return accepted an invalid margin");
    }
  for(const auto& h:psd)for(const std::string mode:{"HH","HS","SH"})for(bool objective:{false,true})for(double margin:{0.,.5}) {
    const bool hybrid=study::correction_family(mode,objective)=="hybrid";
    auto c=hybrid?study::construct_hybrid(3,h,margin):study::construct_support_scalar(3,h,margin);
    require(c.original_convex&&c.classification_known,"PSD source classification lost");
    require(c.rho==0&&c.atoms==0&&c.hybrid_atoms.empty()&&c.correction.empty(),"certified PSD function received correction");
    require(study::hybrid_certificate(c,0),"uncorrected PSD source certificate failed");
    study::HybridSelectionContext context;context.ambient=3;context.gradient={1,1,1};
    study::HybridSelectionSettings settings;
    require(study::hybrid_select_enrich(c,3,8,3,settings,context)==0,"PSD source received eager/lazy atom");
  }
  // A source zero eigenvalue does not acquire a strong-convexity certificate
  // just because a caller requested a function margin.
  auto singular=study::construct_hybrid(3,{{0,0,1},{0,1,2},{1,1,4}},.5);
  require(!study::hybrid_certificate(singular,0,.5),"missing strong margin was falsely certified");
  auto safe=study::construct_hybrid(2,{{0,0,-1},{1,1,1}});
  study::HybridAtom atom;atom.weight=1;atom.indices={0};atom.values={1};
  safe.hybrid_atoms.push_back(atom);study::hybrid_assemble_trace(safe);
  require(study::hybrid_downweight(safe,2,0,0),"safe complete atom withdrawal refused");
  require(safe.hybrid_atoms.empty()&&safe.atoms==0&&safe.rho==1,"atom was not actually erased");
  require(study::hybrid_certificate(safe,safe.rho),"withdrawal bypassed model certificate");
  auto unsafe=safe;unsafe.rho=0;unsafe.hybrid_atoms.push_back(atom);study::hybrid_assemble_trace(unsafe);
  const auto before=unsafe;
  require(!study::hybrid_downweight(unsafe,2,0,0),"necessary atom removed");
  require(unsafe.rho==before.rho&&unsafe.hybrid_atoms.size()==before.hybrid_atoms.size()&&unsafe.hybrid_atoms[0].weight==before.hybrid_atoms[0].weight,"refused withdrawal mutated state");
  require(!study::hybrid_downweight(unsafe,2,0,-.1)&&!study::hybrid_downweight(unsafe,2,0,1),"invalid attenuation accepted");
  auto tiny=safe;auto tiny_atom=atom;tiny_atom.weight=std::numeric_limits<double>::denorm_min();
  tiny.hybrid_atoms.push_back(tiny_atom);study::hybrid_assemble_trace(tiny);
  require(!study::hybrid_downweight(tiny,2,0,.75),"unrepresented weight change was reported as a reduction");
  require(tiny.hybrid_atoms.size()==1&&tiny.hybrid_atoms[0].weight==tiny_atom.weight,"refused rounded reduction mutated the atom");
  require(study::hybrid_downweight(tiny,2,0,.25)&&tiny.hybrid_atoms.empty()&&tiny.atoms==0,"underflowed atom was retained instead of removed");
  std::vector<SymTriplet> large;for(int j=0;j<20;++j)large.push_back({j,j,j==0?-2.:1.});
  auto l=study::construct_support_scalar(20,large);l.rho=2.2;
  const double refined=study::hybrid_certified_residual(l,l.rho);
  require(refined==2&&refined<l.rho&&study::hybrid_certificate(l,refined),"large support scalar excess not refinable");
  study::HybridSelectionSettings settings;settings.additions=1;settings.strength=.01;
  study::HybridSelectionContext context;context.ambient=3;context.gradient={1,1,0};
  auto c=study::construct_hybrid(3,{{0,0,-4},{1,1,-1},{2,2,2}});c.rho_limit=4.4;
  bool target_exceeds_installed=false;
  require(study::hybrid_select_enrich(c,3,8,3,settings,context,0,[&](const auto& r){
    if(r.selected){target_exceeds_installed=r.residual_target>r.installed_residual;require(r.installed_residual==r.certified_residual,"heuristic floor replaced certificate");}
  })==1,"safe weak candidate rejected");
  require(target_exceeds_installed&&c.rho<4&&study::hybrid_certificate(c,c.rho),"spectral proposal floor restored scalar excess");
  std::cout<<"h4_correction_guards PASS checks="<<checks<<" C0_convex=1 remove_or_attenuate=1 scalar_refine_large=1 no_spectral_floor=1\n";
}
