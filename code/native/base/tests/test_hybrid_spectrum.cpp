#include "lazyipdca_qcqp/hybrid_spectrum.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;
void demand(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){
 try {
  auto small=study::construct_hybrid(5,{{1,1,-3},{4,4,2}});
  small.rho=99; // Deliberately excluded from all candidate-selection estimates.
  auto spectrum=study::hybrid_spectrum(small,8,24,2,1e-8);
  demand(std::abs(spectrum.lambda_min+3)<1e-12&&spectrum.vectors.size()==1,"small compact spectrum incorrect");
  demand(spectrum.vectors[0].size()==2&&std::abs(spectrum.vectors[0][0]-1)<1e-12,"compact coordinates or canonical sign wrong");
  auto applied=study::hybrid_apply_support(small,{2,-3});
  demand(applied[0]==-6&&applied[1]==-6,"compact application includes rho or wrong indices");
  study::HybridAtom atom;atom.weight=4;atom.indices={1};atom.values={1};small.hybrid_atoms.push_back(atom);
  spectrum=study::hybrid_spectrum(small,8,24,2,1e-8);
  demand(std::abs(spectrum.lambda_min-1)<1e-12&&spectrum.vectors.empty(),"actual atom Gram ignored");
  const double a=1/std::sqrt(2.0),b=1/std::sqrt(6.0),d=1/std::sqrt(3.0);
  const double vectors[3][3]={{a,-a,0},{b,b,-2*b},{d,d,d}};
  const double eigenvalues[3]={-7,-1,3};std::vector<SymTriplet> matrix;
  for(int i=0;i<3;++i)for(int j=i;j<3;++j){double value=0;for(int k=0;k<3;++k)value+=eigenvalues[k]*vectors[k][i]*vectors[k][j];matrix.push_back({i,j,value});}
  auto rotated=study::construct_hybrid(3,matrix);spectrum=study::hybrid_spectrum(rotated,8,24,2,1e-8);
  demand(spectrum.vectors.size()==2&&std::abs(spectrum.lambda_min+7)<1e-11,"rotated negative candidate pool incorrect");
  demand(std::abs(spectrum.rayleigh[1]+1)<1e-11,"second rotated negative direction missing");
  demand(std::abs(study::hybrid_spectrum_detail::dot(spectrum.vectors[0],spectrum.vectors[1]))<1e-12,"small candidate pool not independent");
  for(int i=3;i<40;++i)matrix.push_back({i,i,1.+i*.1});
  auto large=study::construct_hybrid(40,matrix);auto first=study::hybrid_spectrum(large,8,24,2,1e-8);
  auto second=study::hybrid_spectrum(large,8,24,2,1e-8);
  demand(first.vectors.size()>=2&&std::abs(first.lambda_min+7)<1e-8,"large rotated Lanczos minimum missing");
  demand(std::abs(first.rayleigh[1]+1)<1e-7,"large Lanczos finite negative pool missing second direction");
  demand(first.lambda_min==second.lambda_min&&first.vectors==second.vectors,"Lanczos search is nondeterministic");
  // Independent deterministic starts can return correlated vectors spanning
  // one repeated negative eigenspace. The retained candidate basis must span
  // both directions, rather than spend a rank-two batch on overlapping rays.
  std::vector<SymTriplet> repeated;for(int i=0;i<64;++i)repeated.push_back({i,i,i<2?-5.:1.});
  auto repeated_metric=study::construct_hybrid(64,repeated);
  auto repeated_pool=study::hybrid_spectrum(repeated_metric,8,24,2,1e-8);
  demand(repeated_pool.vectors.size()==2&&repeated_pool.orthogonalized_candidates>0,"repeated large eigenspace was not diversified");
  demand(std::abs(study::hybrid_spectrum_detail::dot(repeated_pool.vectors[0],repeated_pool.vectors[1]))<1e-13,"repeated Ritz candidates remain correlated");
  for(std::size_t j=0;j<repeated_pool.vectors.size();++j)
    demand(std::abs(study::hybrid_spectrum_detail::dot(repeated_pool.vectors[j],study::hybrid_apply_support(repeated_metric,repeated_pool.vectors[j]))-repeated_pool.rayleigh[j])<1e-13,"projected candidate kept a stale Rayleigh value");
  std::vector<std::vector<double>> basis={{1,0,0}};
  std::vector<double> near={1,1e-7,0};
  demand(study::hybrid_spectrum_detail::orthogonalize_candidate(near,basis)&&std::abs(near[0])<1e-15&&std::abs(near[1]-1)<1e-15,"stable MGS lost a resolvable near-collinear direction");
  near={1,0,0};demand(!study::hybrid_spectrum_detail::orthogonalize_candidate(near,basis),"dependent direction was normalized as new curvature");
  std::vector<SymTriplet> wide;for(int i=0;i<20000;++i)wide.push_back({i,i,i==0?-5.:i==1?-2.:1.});
  auto bounded=study::construct_hybrid(20000,wide);spectrum=study::hybrid_spectrum(bounded,2,5,2,1e-8);
  demand(spectrum.vectors.size()==2&&std::abs(spectrum.lambda_min+5)<1e-9&&std::abs(spectrum.rayleigh[1]+2)<1e-9,"bounded large support search incorrect");
  bool invalid=false;try{study::hybrid_spectrum(large,8,1,2,1e-8);}catch(const std::runtime_error&){invalid=true;}
  demand(invalid,"invalid Lanczos dimension silently accepted");
  invalid=false;large.original[0].v=INFINITY;try{study::hybrid_spectrum(large,8,24,2,1e-8);}catch(const std::runtime_error&){invalid=true;}
  demand(invalid,"nonfinite spectrum silently accepted");
  std::cout<<"hybrid_spectrum PASS: compact, rho-exclusion, actual-atoms, rotated-pool, deterministic-Lanczos, repeated-Lanczos-QR, near-collinear, bounded-n20000, invalid-input\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
