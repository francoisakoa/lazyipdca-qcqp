#include "lazyipdca_qcqp/convexification_study.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;using namespace lazyipdca::qcqp::study;
int main(){int checks=0;auto check=[&](bool b,const char* why){++checks;if(!b)throw std::runtime_error(why);};
 auto make=[](int n){std::vector<SymTriplet> q;for(int i=0;i<n;++i)q.push_back({i,i,i==0?-4.:1.});auto c=construct_hybrid(n,q);HybridAtom a;a.weight=3;for(int i=0;i<n;++i){a.indices.push_back(i);a.values.push_back(i==0?.9:.1);}c.hybrid_atoms.push_back(a);hybrid_assemble_trace(c);return c;};
 for(int n:{17,20}){
  auto c=make(n);auto gram=hybrid_exact_gram(c);for(auto e:c.original)gram[{e.i,e.j}]+=Rational(e.v);const double old=hybrid_gershgorin_shift(gram);
  const auto base=hybrid_implicit_intervals(c,.3);for(int i:c.support)gram[{i,i}]-=Rational(.3);
  for(std::size_t k=0;k<base.native.size();++k){auto a=base.native[k];auto b=base.wide[k];auto value=gram[{a.i,a.j}];check(Rational(a.value.lo)<=value&&Rational(a.value.hi)>=value,"native input excludes exact triple product");check(Rational(b.value.lo)<=value&&Rational(b.value.hi)>=value,"wide input excludes exact triple product");}
  const auto started=std::chrono::steady_clock::now();const double rho=hybrid_certified_residual(c,4);const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  check(rho<3&&rho<old&&rho>=0,"implicit refinement did not reduce excess scalar");check(hybrid_certificate(c,rho),"refined certificate lost");
  check(c.implicit_interval_certificate&&c.implicit_interval_certificate->refinement_complete,"finished scalar search was not cached");
  check(hybrid_certified_residual(c,rho)==rho,"constant Hessian refinement was recomputed inconsistently");
  auto exact=hybrid_exact_gram(c);for(auto e:c.original)exact[{e.i,e.j}]+=Rational(e.v);std::vector<Rational> full(n*n);for(auto e:exact){auto [i,j]=e.first;full[i*n+j]+=e.second;if(i!=j)full[j*n+i]+=e.second;}for(int i=0;i<n;++i)full[i*n+i]+=Rational(rho);check(rational_psd(full,n),"interval proof disagrees with exact Bareiss");
  c.hessian.clear();for(int i=0;i<n;++i)c.hessian.push_back({i,i,1e9});check(!hybrid_certificate(c,0),"diagnostic Hessian replaced implicit matrix");check(hybrid_certificate(c,rho+1e-6),"monotone cache failed");
  auto bad=c;bad.hybrid_atoms[0].weight=0;check(!hybrid_certificate(bad,rho),"atom mutation reused stale cache");bad=c;bad.hybrid_atoms[0].weight=-1;check(!hybrid_certificate(bad,rho),"negative atom passed cache");check(!hybrid_certificate(c,rho,1),"margin mutation reused stale cache");
  bad=c;bad.hybrid_atoms[0].values[0]=0;check(!hybrid_certificate(bad,rho),"atom vector mutation reused stale cache");
  bad=c;bad.original[0].v=-8;check(!hybrid_certificate(bad,rho),"source Hessian mutation reused stale cache");
  check(!hybrid_certificate(c,-1),"negative scalar passed cache");
  bad=c;bad.hybrid_atoms[0].indices[0]=n+1;check(!hybrid_certificate(bad,rho),"out-of-support atom passed cache");
  std::cout<<"n="<<n<<" gersh="<<old<<" rho="<<rho<<" seconds="<<seconds<<'\n';
 }
 bool rounded_case=false;
 for(double beta:{.1,.3,1.,3.,7.})for(double value:{.1,.2,.3,.7,.9})if(!rounded_case) {
  const Rational exact=Rational(beta)*Rational(value)*Rational(value);const double rounded=exact.convert_to<double>();
  if(Rational(rounded)<=exact)continue;
  std::vector<SymTriplet> q;for(int i=0;i<17;++i)q.push_back({i,i,i==0?-rounded:1.});
  auto near=construct_hybrid(17,q);HybridAtom a;a.weight=beta;a.indices={0};a.values={value};near.hybrid_atoms={a};
  near.hessian=q;near.hessian[0].v=0; // rounded diagnostic appears PSD
  check(!hybrid_certificate(near,0),"rounded-away exact negative curvature was accepted");
  check(Rational(rounded)>exact,"rounding counterexample is not exact");rounded_case=true;
 }
 check(rounded_case,"no exact Gram rounding counterexample found");
 auto legacy=make(20);legacy.support_scalar=false;
 auto legacy_matrix=hybrid_exact_gram(legacy);for(auto e:legacy.original)legacy_matrix[{e.i,e.j}]+=Rational(e.v);
 const double legacy_gersh=hybrid_gershgorin_shift(legacy_matrix);
 check(hybrid_certified_residual(legacy,legacy_gersh)==legacy_gersh,"legacy ambient scalar path changed");
 auto c=make(129);auto exact=hybrid_exact_gram(c);for(auto e:c.original)exact[{e.i,e.j}]+=Rational(e.v);const auto g=hybrid_gershgorin_shift(exact);check(hybrid_certified_residual(c,g)==g,"large-support path changed");
 std::cout<<"implicit_interval_psd PASS checks="<<checks<<" exact_triple_products=1 Bareiss=1 cache_invalidation=1 diagnostic_H_ignored=1\n";
}
