#include "lazyipdca_qcqp/convexification_study.hpp"
#include <iostream>
#include <stdexcept>
#include <random>
using namespace lazyipdca::qcqp;
void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
// Slow rational Schur oracle, independent of the integer Bareiss arithmetic.
bool rational_schur_oracle(std::vector<study::Rational> a,int n) {
  for(int k=0;k<n;++k) {
    if(a[k*n+k]<0)return false;
    if(a[k*n+k]==0){for(int j=k+1;j<n;++j)if(a[k*n+j]!=0)return false;continue;}
    for(int i=k+1;i<n;++i)for(int j=i;j<n;++j)
      a[j*n+i]=a[i*n+j]-a[i*n+k]*a[k*n+j]/a[k*n+k];
    for(int i=k+1;i<n;++i)for(int j=i;j<n;++j)a[i*n+j]=a[j*n+i];
  }
  return true;
}
void verify_fraction_free_psd() {
  using R=study::Rational;
  // Singular pivots after positive pivots, zero leading pivots, and exact
  // indefinite 2x2 blocks must have the same classification as Schur LDL.
  for(auto a:std::vector<std::vector<R>>{
      {4,0,0,0,0,0,0,0,9}, {0,0,0,0,4,2,0,2,1},
      {0,0,1,0,1,0,1,0,0}, {R(1)/3,R(1)/2,R(1)/2,R(3)/4}}) {
    const int n=a.size()==9?3:2;
    require(study::rational_psd(a,n)==rational_schur_oracle(a,n),"fraction-free singular or rational mismatch");
  }
  std::mt19937 random(94017);
  for(int n=1;n<=8;++n)for(int trial=0;trial<12;++trial) {
    std::vector<R> b(n*n),a(n*n,0);
    for(auto& v:b)v=R(int(random()%17)-8)/(1u<<(random()%9));
    if(trial%2==0) {
      // Exact Gram with varying rank, including zero and singular PSD.
      const int rank=trial%(n+1);
      for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<rank;++k)a[i*n+j]+=b[k*n+i]*b[k*n+j];
    } else for(int i=0;i<n;++i)for(int j=i;j<n;++j)a[i*n+j]=a[j*n+i]=b[i*n+j];
    require(study::rational_psd(a,n)==rational_schur_oracle(a,n),"fraction-free exact oracle mismatch");
  }
  // Clearing dyadic denominators must retain terms far below machine scale.
  std::vector<R> subnormal={R(1),R(1e-100),R(1e-100),R(1e-200)};
  require(study::rational_psd(subnormal,2)==rational_schur_oracle(subnormal,2),"fraction-free tiny term lost");
  bool rejected=false;try{study::rational_psd({1,2,3,4},2);}catch(const std::runtime_error&){rejected=true;}
  require(rejected,"nonsymmetric exact certificate accepted");
}
int main(){
  verify_fraction_free_psd();
  for(auto h:std::vector<std::vector<SymTriplet>>{
      {{0,0,1},{0,1,2},{1,1,4}}, // singular PSD non-DD
      {{0,0,1},{0,1,2},{1,1,5}}, // SPD non-DD
      {},{{0,0,0},{1,1,2}}}) {
    for(auto family:{"scalar","atoms"}) {
      auto c=study::construct(2,h,family,true);
      require(c.original_convex,"convex function not classified");
      require(c.rho==0&&c.atoms==0&&c.correction.empty(),"convex function corrected");
    }
  }
  const std::vector<SymTriplet> h={{0,0,-1},{0,1,2},{1,1,1}};
  for(auto family:{"scalar","atoms"})for(bool precise:{false,true}) {
    auto c=study::construct(2,h,family,precise);
    auto a=study::dense(c.hessian,c.support);
    for(int i=0;i<2;++i)a[i*2+i]+=c.rho;
    require(study::exact_psd(a,2),"model not PSD");
    auto correction=study::dense(c.correction,c.support);
    for(int i=0;i<2;++i)correction[i*2+i]+=c.rho;
    require(study::exact_psd(correction,2),"correction not PSD");
    if(std::string(family)=="atoms")require(c.rho==0&&c.atoms>0,"atoms not pure");
    else require(c.atoms==0&&c.rho>0,"scalar not pure");
  }
  auto initial=study::construct(2,h,"scalar",false),refined=study::construct(2,h,"scalar",true);
  require(refined.rho<initial.rho&&refined.rho>0,"scalar excess not reduced");
  // Ambient scalar is independent of support. Its omitted coordinate remains
  // rho, not zero, in the true model Q+rho I.
  auto scalar=study::construct(3,{{0,0,-2}},"scalar",true);
  require(scalar.rho>=2,"ambient scalar threshold");
  auto rounded=study::construct(2,{{0,0,1e16},{0,1,1}},"atoms",false);
  require(rounded.rank_upper==2&&rounded.atoms>=2,"assembly padding omitted from atom/rank metadata");
  auto mixed=study::construct(3,{{0,0,-1},{0,1,1},{0,2,1e-100}},"scalar",false);
  auto mixed_dense=study::dense(mixed.original,mixed.support);
  require(study::exact_psd_shift(mixed_dense,3,mixed.rho),"exact Gersh bound lost small term");
  auto progressive=study::construct(3,{{0,0,-3},{1,1,-2},{2,2,1}},"atoms",true);
  require(progressive.atom_batches==2 && progressive.atom_certificate_checks>=2,"atomic enrichment not observed");
  require(study::exact_psd(study::dense(progressive.hessian,progressive.support),3),"enriched model not certified");
  std::cout<<"convexification_study_certificates PASS\n";
}
