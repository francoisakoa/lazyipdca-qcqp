#include "lazyipdca_qcqp/convexification_study.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
using namespace lazyipdca::qcqp;
using namespace lazyipdca::qcqp::study;
int main(int argc,char** argv){
 if(argc==2) {
  std::ifstream stream(argv[1]);int n,m;if(!(stream>>n>>m)||n<0||m<0)return 2;
  std::vector<SymTriplet> matrix;for(int k=0;k<m;++k){int i,j;double v;if(!(stream>>i>>j>>v))return 2;matrix.push_back({i,j,v});}
  for(const std::string family:{"hybrid","scalar"}) {
   const auto c=family=="hybrid"?construct_hybrid(n,matrix):construct_support_scalar(n,matrix);
   const auto& p=c.sparse_psd_proof;
   std::cout<<std::setprecision(17)<<"{\"family\":\""<<family<<"\",\"rho\":"<<c.rho<<",\"atoms\":"<<c.atoms
    <<",\"original_convex\":"<<(c.original_convex?"true":"false")<<",\"certified\":"<<(p.certified?"true":"false")
    <<",\"precision_bits\":"<<p.precision_bits<<",\"positive_pivots\":"<<p.positive_pivots<<",\"updates\":"<<p.updates
    <<",\"minimum_pivot_lower\":"<<p.minimum_pivot_lower<<",\"peak_entries\":"<<p.peak_entries<<",\"seconds\":"<<p.seconds<<"}\n";
   if(!p.certified||!c.original_convex||c.rho!=0||c.atoms)return 1;
  }
  return 0;
 }
 int checks=0;auto check=[&](bool value,const char* why){++checks;if(!value)throw std::runtime_error(why);};
 auto arithmetic=[&](auto sample){
  using F=decltype(sample);using I=sparse_psd_detail::Bounds<F>;
  auto inside=[&](const I& a,const Rational& lo,const Rational& hi){check(Rational(a.lo)<=lo&&Rational(a.hi)>=hi,"interval arithmetic failed exact containment");};
  for(double a:{-3.,-.1,0.,.1,3.,1e-300})for(double b:{-2.,.2,2.,1e-300}) {
   I x{F(a),F(a)},y{F(b),F(b)};Rational qa(a),qb(b);
   inside(sparse_psd_detail::add(x,y),qa+qb,qa+qb);
   inside(sparse_psd_detail::subtract(x,y),qa-qb,qa-qb);
   inside(sparse_psd_detail::multiply(x,y),qa*qb,qa*qb);
   I positive{F(.2),F(3.)};Rational p0=qa/Rational(.2),p1=qa/Rational(3.);
   inside(sparse_psd_detail::divide_positive(x,positive),std::min(p0,p1),std::max(p0,p1));
  }
 };
 arithmetic((long double)0);arithmetic(boost::multiprecision::cpp_bin_float_quad(0));
 for(double epsilon:{0x1p-20,0x1p-40,0x1p-50})for(int sign:{-1,0,1}) {
   std::vector<SymTriplet> q={{0,0,1},{0,1,2},{1,1,4+sign*epsilon}};
   const bool exact=exact_psd(dense(q,{0,1}),2);const auto proof=verified_sparse_psd(2,q);
   check(!proof.certified||exact,"near-singular indefinite matrix falsely certified");
   if(sign>0)check(proof.certified,"representable near-singular SPD control was not certified");
 }
 std::mt19937 rng(20261022);
 for(int n=1;n<=8;++n)for(int sample=0;sample<20;++sample) {
   std::vector<SymTriplet> q;std::vector<int> support(n);std::iota(support.begin(),support.end(),0);
   for(int i=0;i<n;++i)for(int j=i;j<n;++j){double v=int(rng()%7)-3;if(i==j&&sample%3==0)v+=6*n;if(v)q.push_back({i,j,v});}
   const bool exact=exact_psd(dense(q,support),n);const auto proof=verified_sparse_psd(n,q);
   check(!proof.certified||exact,"interval factor disagrees with exact Bareiss");
   const auto wide=verified_sparse_psd_precision<boost::multiprecision::cpp_bin_float_quad>(n,q,{});
   check(!wide.certified||exact,"wide interval factor disagrees with exact Bareiss");
   if(sample%3==0)check(wide.certified,"wide SPD control was not certified");
   if(sample%3==0)check(proof.certified,"strong SPD control was not certified");
 }
 // Connected non-DD Gram matrix R'R, R upper bidiagonal with alternating
 // diagonal 2/1 and superdiagonal 1.5. Scalar Gershgorin has a deficit.
 std::vector<SymTriplet> q;const int n=40;
 for(int i=0;i<n;++i){double r=i%2?1:2;q.push_back({i,i,r*r+(i?2.25:0)});if(i+1<n)q.push_back({i,i+1,1.5*r});}
 const auto proof=verified_sparse_psd(n,q);check(proof.certified&&proof.positive_pivots==n,"large connected non-DD SPD matrix rejected");
 const auto legacy=construct(n,q,"scalar",false);
 check(!legacy.sparse_psd_proof.certified&&legacy.sparse_psd_proof.reason=="not-started"&&
       !legacy.classification_known&&legacy.rho>0,
       "large sparse source classification silently changed a historical study family");
 for(const auto& family:std::vector<std::string>{"hybrid","scalar"}) {
   const auto c=family=="hybrid"?construct_hybrid(n,q):construct_support_scalar(n,q);
   check(c.original_convex&&c.classification_known&&c.rho==0&&c.hybrid_atoms.empty(),"large convex function received correction");
   check(c.sparse_psd_proof.certified,"function factory did not retain its verified global proof");
 }
 for(const bool positive:{true,false}) {
   const double off=positive?.5:2.;
   std::vector<SymTriplet> mixed={{0,0,1},{1,1,1},{0,1,0x1p100},{1,0,off},{0,1,-0x1p100}};
   const auto r=verified_sparse_psd(2,mixed);
   check(r.certified==positive,"reverse repeated coefficients or cancellation produced a false sign");
   const auto tiny=verified_sparse_psd(2,{{0,0,1e-300},{1,1,1e300},{0,1,off}});
   check(tiny.certified==positive,"mixed-magnitude SPD/indefinite control failed");
 }
 std::vector<SymTriplet> repeated;
 for(const auto& e:q){repeated.push_back({e.j,e.i,e.v*.5});repeated.push_back({e.i,e.j,e.v*.5});}
 check(verified_sparse_psd(n,repeated).certified,"reversed/repeated sparse SPD matrix was rejected");
 SparsePsdBudget budget;budget.max_entries=1;
 check(!verified_sparse_psd(n,q,budget).certified,"entry budget was ignored");
 budget={};budget.max_updates=0;
 check(!verified_sparse_psd(n,q,budget).certified,"operation budget was ignored");
 budget={};budget.max_seconds=1e-300;
 check(!verified_sparse_psd(n,q,budget).certified,"time budget was ignored");
 budget={};budget.max_dimension=1;
 check(!verified_sparse_psd(n,q,budget).certified,"dimension budget was ignored");
 check(!verified_sparse_psd(2,{{0,0,INFINITY}}).certified,"infinite coefficient accepted");
 check(!verified_sparse_psd(2,{{-1,0,1}}).certified,"invalid index accepted");
 check(!verified_sparse_psd(2,{{0,0,NAN}}).certified,"nonfinite coefficient accepted");
 check(!verified_sparse_psd(2,{{0,1,1}}).certified,"zero-pivot indefinite matrix accepted");
 check(verified_sparse_psd(3,{}).certified,"exact zero PSD rows rejected");
 std::cout<<"verified_sparse_psd PASS checks="<<checks<<" exact_Bareiss=1 near_singular=1 arithmetic_containment=1 large_non_DD_C0=1 fail_closed_budgets=1\n";
}
