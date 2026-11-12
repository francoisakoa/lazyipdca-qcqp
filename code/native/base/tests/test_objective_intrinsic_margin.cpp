#include "lazyipdca_qcqp/convexification_study.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;using namespace lazyipdca::qcqp::study;
int main(){int checks=0;auto require=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
 auto make=[](int n,std::vector<SymTriplet> q){Correction c;c.original=std::move(q);c.support.resize(n);std::iota(c.support.begin(),c.support.end(),0);return c;};
 auto exact=[&](const Correction& c,double gamma){const int n=c.support.size();auto gram=hybrid_exact_gram(c);for(auto e:c.original)gram[{std::min(e.i,e.j),std::max(e.i,e.j)}]+=Rational(e.v);std::vector<Rational> a(n*n);for(auto [ij,v]:gram){auto[i,j]=ij;a[i*n+j]+=v;if(i!=j)a[j*n+i]+=v;}for(int i=0;i<n;++i)a[i*n+i]-=Rational(gamma);return rational_psd(std::move(a),n);};
 for(int n:{2,20,64}) {
  auto c=make(n,{{0,0,1},{0,1,2},{1,1,5}});for(int i=2;i<n;++i)c.original.push_back({i,i,1});
  std::shared_ptr<const ObjectiveMarginCertificate> cache;
  auto proof=certified_objective_implicit_lower(c,cache);
  require(certified_objective_base_lower(n,c.original)==-1,"test is not non-DD");
  require(proof->lower>.17&&proof->lower<.18&&proof->interval_certified,"non-DD SPD objective margin was not proved");
  require(exact(c,proof->lower),"interval margin disagrees with independent exact Bareiss");
  require(certified_objective_implicit_lower(c,cache)==proof,"constant full coefficient certificate was not cached");
  auto changed=c;changed.original[0].v=0;auto different=certified_objective_implicit_lower(changed,cache);
  require(different!=proof&&different->lower==0,"changed base reused a stale positive margin");
  require(exact(c,proof->lower)&&!exact(c,.18),"exact proof controls do not bracket lambda minimum");
 }
 auto semi=make(2,{{0,0,1},{0,1,1},{1,1,1}});std::shared_ptr<const ObjectiveMarginCertificate> cache;
 auto proof=certified_objective_implicit_lower(semi,cache);require(proof->lower==0&&!proof->interval_certified,"singular PSD objective acquired a positive margin");
 auto omitted=make(3,{{0,0,1},{0,1,2},{1,1,5}});proof=certified_objective_implicit_lower(omitted,cache);require(proof->lower==0,"missing ambient coordinate was ignored");
 auto atom=make(2,{{0,0,-2},{1,1,1}});atom.hybrid_atoms={HybridAtom{3,{0},{1},0}};
 proof=certified_objective_implicit_lower(atom,cache);require(proof->lower>.99&&proof->interval_certified&&exact(atom,proof->lower),"actual atom border was omitted from objective certificate");
 const auto old=proof;atom.hybrid_atoms[0].weight=1;proof=certified_objective_implicit_lower(atom,cache);require(proof!=old&&proof->lower==0,"atom weight mutation reused stale certificate");
 atom.hybrid_atoms[0].weight=3;atom.hybrid_atoms[0].values[0]=.5;proof=certified_objective_implicit_lower(atom,cache);require(proof->lower==0,"atom vector mutation reused stale certificate");
 auto unsafe=atom;unsafe.hybrid_atoms[0].weight=-1;bool rejected=false;try{certified_objective_implicit_lower(unsafe,cache);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"negative border weight accepted");
 auto large=make(129,{{0,0,1},{0,1,2},{1,1,5}});proof=certified_objective_implicit_lower(large,cache);require(proof->lower==0&&proof->checks==0,"large dimension exceeded bounded search scope");
 auto dd=make(2,{{0,0,2},{0,1,.25},{1,1,3}});proof=certified_objective_implicit_lower(dd,cache);require(proof->lower==1.75&&proof->checks==0,"cheap positive exact Gershgorin bound changed");
 require(certified_proximal_margin_floor(.001,.08)==0,"known objective curvature still receives artificial proximal floor");
 std::cout<<"objective_intrinsic_margin PASS checks="<<checks<<" independent_exact_Bareiss=1 singular_control=1 installed_atoms=1 cache_invalidation=1\n";
}
