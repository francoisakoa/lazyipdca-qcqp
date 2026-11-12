#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/verified_sparse_psd.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <limits>
#include <numeric>
#include <memory>
#include <chrono>
#include <cstdint>
extern "C" void dsyev_(const char*,const char*,const int*,double*,const int*,double*,double*,const int*,int*);
namespace lazyipdca::qcqp::study {
using Rational=boost::multiprecision::cpp_rational;
// Exact fraction-free Schur complements. All function coefficients and atom
// products are dyadic rationals; clearing a common positive denominator and
// Bareiss elimination avoids a rational GCD at every arithmetic operation.
// No numerical tolerance or eigenvalue estimate is accepted as proof of PSD.
inline bool rational_psd(std::vector<Rational> a,int n) {
  if(a.size()!=std::size_t(n*n))throw std::runtime_error("study matrix dimension mismatch");
  for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)if(a[i*n+j]!=a[j*n+i])throw std::runtime_error("study nonsymmetric exact certificate input");
  using Integer=boost::multiprecision::cpp_int;
  Integer common=1;
  for(const auto& value:a) {
    const Integer d=denominator(value);
    if(common%d!=0) {
      Integer x=common,y=d;
      while(y!=0){Integer remainder=x%y;x=std::move(y);y=std::move(remainder);}
      common=(common/x)*d;
    }
  }
  std::vector<Integer> b;b.reserve(a.size());
  for(const auto& value:a)b.push_back(numerator(value)*(common/denominator(value)));
  Integer previous=1;
  for(int k=0;k<n;++k) {
    const Integer pivot=b[k*n+k];
    if(pivot<0)return false;
    if(pivot==0) {for(int j=k+1;j<n;++j)if(b[k*n+j]!=0)return false;continue;}
    for(int i=k+1;i<n;++i)for(int j=i;j<n;++j)
      b[j*n+i]=(pivot*b[i*n+j]-b[i*n+k]*b[k*n+j])/previous;
    for(int i=k+1;i<n;++i)for(int j=i;j<n;++j)b[i*n+j]=b[j*n+i];
    previous=pivot;
  }
  return true;
}
inline bool exact_psd(const std::vector<double>& matrix,int n) {
  std::vector<Rational> a;a.reserve(matrix.size());
  for(double v:matrix){if(!std::isfinite(v))throw std::runtime_error("study nonfinite matrix");a.emplace_back(v);}
  return rational_psd(std::move(a),n);
}
inline bool exact_psd_sum(const std::vector<double>& a,const std::vector<double>& b,int n,int sign=1) {
  std::vector<Rational> sum;sum.reserve(a.size());
  for(std::size_t k=0;k<a.size();++k)sum.push_back(Rational(a[k])+sign*Rational(b[k]));
  return rational_psd(std::move(sum),n);
}
inline bool exact_psd_shift(const std::vector<double>& matrix,int n,double rho) {
  std::vector<Rational> shifted;shifted.reserve(matrix.size());
  for(double v:matrix)shifted.emplace_back(v);
  for(int j=0;j<n;++j)shifted[j*n+j]+=Rational(rho);
  return rational_psd(std::move(shifted),n);
}
inline double upward(long double v) {
  double d=static_cast<double>(v);
  if(static_cast<long double>(d)<v)d=std::nextafter(d,INFINITY);
  return d;
}
inline double rational_upward(const Rational& value) {
  double d=value.convert_to<double>();
  if(!std::isfinite(d))throw std::runtime_error("study curvature bound exceeds finite double range");
  while(Rational(d)<value)d=std::nextafter(d,INFINITY);
  if(!std::isfinite(d))throw std::runtime_error("study curvature upper bound overflow");
  return d;
}
// Certified lower bound for the explicit objective model base. Positive
// rank-one borders may be ignored by Weyl monotonicity. Ambient coordinates
// absent from this matrix contribute exact zero eigenvalues to the bound.
inline double certified_objective_base_lower(int ambient_n,const std::vector<SymTriplet>& model_hessian) {
  if(ambient_n<0)throw std::runtime_error("negative objective model dimension");
  if(!ambient_n){if(!model_hessian.empty())throw std::runtime_error("nonempty zero-dimensional objective model");return 0;}
  std::map<std::pair<int,int>,Rational> matrix;
  for(const auto& e:model_hessian) {
    if(e.i<0||e.j<0||e.i>=ambient_n||e.j>=ambient_n||!std::isfinite(e.v))
      throw std::runtime_error("invalid objective model coefficient");
    matrix[{std::min(e.i,e.j),std::max(e.i,e.j)}]+=Rational(e.v);
  }
  std::map<int,Rational> row_bound;
  for(const auto& e:matrix) {
    const int i=e.first.first,j=e.first.second;
    if(i==j)row_bound[i]+=e.second;
    else {const Rational magnitude=e.second<0?-e.second:e.second;row_bound[i]-=magnitude;row_bound[j]-=magnitude;}
  }
  Rational lower=row_bound.size()<std::size_t(ambient_n)?Rational(0):row_bound.begin()->second;
  for(const auto& row:row_bound)if(row.second<lower)lower=row.second;
  return -rational_upward(-lower);
}
// The algorithmic proximal term is separate from function convexification.
// These directed-rounding helpers combine an independently certified model
// lower bound with a nonnegative retained-coordinate proximal diagonal.
inline double certified_proximal_margin_floor(double requested_margin,double model_lower) {
  if(!std::isfinite(requested_margin)||requested_margin<0||
     !std::isfinite(model_lower)||model_lower<0)
    throw std::invalid_argument("invalid certified proximal margin data");
  const Rational difference=Rational(requested_margin)-Rational(model_lower);
  return difference>0?rational_upward(difference):0;
}
inline double certified_objective_margin_sum(double model_lower,double tau,double scale=1) {
  if(!std::isfinite(model_lower)||model_lower<0||!std::isfinite(tau)||tau<0||
     !std::isfinite(scale)||scale<=0)
    throw std::invalid_argument("invalid certified objective margin data");
  const Rational total=(Rational(model_lower)+Rational(tau))/Rational(scale);
  return total>0?-rational_upward(-total):0;
}
struct HybridAtom {
  double weight=0;std::vector<int> indices;std::vector<double> values;int strikes=0;
};
struct HybridIntervalCertificate {std::string key;double rho=0;int precision_bits=0;bool refinement_complete=false;};
struct HybridSpectrumCache;
// Certificates depend on the frozen function matrix and the installed metric,
// never on the current primal point, atom activity or filter history. Copies
// made for a rollback share this bounded, content-addressed query memo. Only
// complete deterministic answers are retained; a budget-limited failure is not
// evidence that a later query could not certify the same matrix.
struct HybridCertificateMemo {
  struct Entry {std::string key;bool result=false;std::shared_ptr<const HybridIntervalCertificate> interval;};
  std::vector<Entry> entries;
  std::size_t bytes=0;
  std::uint64_t hits=0,misses=0,evictions=0,inconclusive=0,downweight_hits=0,gram_hits=0,gram_misses=0,refusals_invalidated=0;
  double lookup_seconds=0,certificate_seconds=0;
  std::uint64_t spectrum_hits=0,spectrum_misses=0;
  std::size_t spectrum_entries=0,spectrum_bytes=0;
  double spectrum_seconds=0;
  static constexpr std::size_t max_entries=128,max_bytes=8*1024*1024;
};
struct HybridGramSnapshot {
  std::string key;
  std::map<std::pair<int,int>,Rational> gram;
};
struct Correction {
  mutable std::shared_ptr<const HybridIntervalCertificate> implicit_interval_certificate;
  mutable std::shared_ptr<HybridCertificateMemo> certificate_memo;
  mutable std::shared_ptr<const HybridGramSnapshot> exact_gram_snapshot;
  mutable std::shared_ptr<HybridSpectrumCache> spectrum_cache;
  SparsePsdProof sparse_psd_proof;
  std::vector<int> support;
  std::vector<HybridAtom> hybrid_atoms;
  std::vector<SymTriplet> original, correction, hessian;
  double rho=0, rho_limit=0, original_lower=0, final_lower=0, padding=0;
  int atoms=0, rank_upper=0, atom_batches=0, atom_certificate_checks=0;
  bool original_convex=false, classification_known=false, refined=false;
  // HH/HS/SH/SS use rho P_support. Legacy scalar modes continue to use rho I.
  bool support_scalar=false;
  std::string certificate="zero",family;
};
inline bool support_study_mode(const std::string& mode) {
  return mode=="HH"||mode=="HS"||mode=="SH"||mode=="SS";
}
inline std::string correction_family(const std::string& mode,bool objective) {
  if(mode=="U"||mode=="F")return "scalar";
  const char kind=mode.at(objective?0:1);
  return kind=='H'?"hybrid":kind=='S'?"scalar":"atoms";
}
inline std::vector<double> dense(const std::vector<SymTriplet>& h,const std::vector<int>& support) {
  const int n=support.size();std::vector<double> a(n*n,0);
  for(auto e:h) {
    const int i=std::lower_bound(support.begin(),support.end(),e.i)-support.begin();
    const int j=std::lower_bound(support.begin(),support.end(),e.j)-support.begin();
    a[i*n+j]+=e.v;if(i!=j)a[j*n+i]+=e.v;
  }
  return a;
}
inline void assemble(Correction& c) {
  std::map<std::pair<int,int>,double> h;
  for(auto e:c.original)h[{e.i,e.j}]+=e.v;
  for(auto e:c.correction)h[{e.i,e.j}]+=e.v;
  c.hessian.clear();for(auto e:h)c.hessian.push_back({e.first.first,e.first.second,e.second});
}
inline Correction construct(int ambient,const std::vector<SymTriplet>& input,
                             const std::string& family,bool precise=false,int limit=16,
                             bool verified_large_source=false) {
  Correction c;c.family=family;c.refined=precise;
  std::map<std::pair<int,int>,double> canonical;
  for(auto e:input){if(!std::isfinite(e.v))throw std::runtime_error("study nonfinite curvature");canonical[{std::min(e.i,e.j),std::max(e.i,e.j)}]+=e.v;}
  std::set<int> support;for(auto e:canonical)if(e.second!=0){c.original.push_back({e.first.first,e.first.second,e.second});support.insert(e.first.first);support.insert(e.first.second);}
  c.support.assign(support.begin(),support.end());const int n=c.support.size();
  if(!n){c.classification_known=c.original_convex=true;return c;}
  std::map<int,Rational> exact_diagonal,exact_radius;
  double scale=1;
  for(auto e:c.original){scale=std::max(scale,std::abs(e.v));if(e.i==e.j){exact_diagonal[e.i]+=Rational(e.v);}else{exact_radius[e.i]+=Rational(std::abs(e.v));exact_radius[e.j]+=Rational(std::abs(e.v));}}
  Rational exact_lower=exact_diagonal[c.support[0]]-exact_radius[c.support[0]];
  for(int j:c.support){const Rational bound=exact_diagonal[j]-exact_radius[j];if(bound<exact_lower)exact_lower=bound;}
  c.original_lower=-rational_upward(-exact_lower);
  if(exact_lower>=0){c.classification_known=c.original_convex=true;c.hessian=c.original;c.final_lower=c.original_lower;c.certificate="gershgorin-PSD";return c;}
  if(n>limit&&verified_large_source) {
    std::vector<SymTriplet> local;local.reserve(c.original.size());
    for(const auto& e:c.original) {
      const int i=std::lower_bound(c.support.begin(),c.support.end(),e.i)-c.support.begin();
      const int j=std::lower_bound(c.support.begin(),c.support.end(),e.j)-c.support.begin();
      local.push_back({i,j,e.v});
    }
    c.sparse_psd_proof=verified_sparse_psd(n,local);
    if(c.sparse_psd_proof.certified) {
      c.classification_known=c.original_convex=true;c.hessian=c.original;c.final_lower=0;
      c.certificate="verified-sparse-interval-LDL-PSD";return c;
    }
  }
  auto a=n<=limit?dense(c.original,c.support):std::vector<double>{};
  if(n<=limit) {
    c.classification_known=true;c.original_convex=exact_psd(a,n);
    if(c.original_convex){c.hessian=c.original;c.final_lower=0;c.certificate="exact-dyadic-LDL-PSD";return c;}
  }
  if(family=="scalar") {
    c.rho=rational_upward(-exact_lower);c.final_lower=0;c.certificate="gershgorin-shift";
    if(precise&&n<=limit) {
      std::vector<double> eig=a,values(n),work(std::max(1,3*n));int info=0,lwork=work.size();char vectors='N',up='U';
      dsyev_(&vectors,&up,&n,eig.data(),&n,values.data(),work.data(),&lwork,&info);
      if(info||!std::isfinite(values.front()))throw std::runtime_error("study scalar spectrum failed");
      double pad=64*std::numeric_limits<double>::epsilon()*scale*n;
      for(int retry=0;retry<16;++retry,pad*=2) {
        const double proposed=std::max(0.0,-values.front())+pad;
        if(exact_psd_shift(a,n,proposed)){c.rho=std::min(c.rho,proposed);c.padding=pad;c.certificate="spectral-proposal-exact-dyadic-PSD";break;}
      }
    }
    c.rank_upper=c.rho>0?ambient:0;c.hessian=c.original;return c;
  }
  if(family!="atoms")throw std::runtime_error("invalid study correction family");
  // Pure sparse atoms. Each off-diagonal b is canceled by
  // |b|(ei-sign(b)ej)(ei-sign(b)ej)' . Remaining deficits use ei ei'.
  std::map<int,double> diagC;
  for(auto e:c.original)if(e.i!=e.j){c.correction.push_back({e.i,e.j,-e.v});++c.atoms;}
  for(int j:c.support) {
    const Rational negative=-exact_diagonal[j];
    const Rational needed=exact_radius[j]>negative?exact_radius[j]:negative;
    const double v=rational_upward(needed);
    diagC[j]=v;if(v>0)c.correction.push_back({j,j,v});
  }
  c.atom_batches=c.atoms?1:0;
  c.rank_upper=n;c.certificate="edge-atoms-DD-C-diagonal-QplusC";c.final_lower=0;
  assemble(c);
  for(auto& e:c.hessian)if(e.i==e.j) {
    const Rational target=exact_diagonal[e.i]+Rational(diagC[e.i]);
    e.v=rational_upward(target);
    while(Rational(e.v)-exact_diagonal[e.i]<exact_radius[e.i]||e.v<0)e.v=std::nextafter(e.v,INFINITY);
    const Rational actual_diagonal=Rational(e.v)-exact_diagonal[e.i];
    if(actual_diagonal>exact_radius[e.i])++c.atoms;
    const Rational extra=actual_diagonal-Rational(diagC[e.i]);
    if(extra>0)c.padding=std::max(c.padding,rational_upward(extra));
  }
  if(n<=limit) {
    auto actual=dense(c.hessian,c.support);
    if(!exact_psd(actual,n)||!exact_psd_sum(actual,a,n,-1))throw std::runtime_error("study graph stored-matrix certificate failed");
  }
  if(precise&&n<=limit) {
    std::vector<double> vectors=a,values(n),work(std::max(1,3*n));int info=0,lwork=work.size();char job='V',up='U';
    dsyev_(&job,&up,&n,vectors.data(),&n,values.data(),work.data(),&lwork,&info);
    if(info)throw std::runtime_error("study atoms spectrum failed");
    double pad=64*std::numeric_limits<double>::epsilon()*scale*n;
    for(int retry=0;retry<16;++retry,pad*=2) {
      std::vector<double> correction(n*n,0);int count=0;
      // Demand-driven enrichment within the curvature oracle. Never expose
      // an incomplete, indefinite model to the outer solver. Each added
      // eigen-direction is followed by an exact certificate check.
      for(int i=0;i<n;++i)correction[i*n+i]=pad;
      for(int k=0;k<n;++k)if(values[k]<0) {
        ++count;
        for(int i=0;i<n;++i)for(int j=i;j<n;++j){correction[i*n+j]+=(-values[k])*vectors[i+n*k]*vectors[j+n*k];correction[j*n+i]=correction[i*n+j];}
        ++c.atom_certificate_checks;
        if(exact_psd(correction,n)&&exact_psd_sum(a,correction,n))break;
      }
      auto total=a;for(int i=0;i<n*n;++i)total[i]+=correction[i];
      // Both assembled matrices are checked exactly on their stored doubles.
      if(exact_psd(correction,n)&&exact_psd_sum(a,correction,n)&&exact_psd(total,n)&&exact_psd_sum(total,a,n,-1)) {
        c.correction.clear();for(int i=0;i<n;++i)for(int j=i;j<n;++j)if(correction[i*n+j]!=0)c.correction.push_back({c.support[i],c.support[j],correction[i*n+j]});
        c.padding=pad;c.atoms=count+n;c.rank_upper=n;c.atom_batches=count;
        c.certificate="spectral-atoms-plus-coordinate-padding-exact-dyadic-PSD";assemble(c);break;
      }
    }
  }
  return c;
}

// Hybrid metric: explicit sparse rank-one atoms are retained separately from
// the ambient scalar. The certificate evaluates their dyadic products exactly.
inline std::string hybrid_interval_key(const Correction& c,double margin) {
  std::string key;
  auto add=[&](const auto& value){key.append(reinterpret_cast<const char*>(&value),sizeof(value));};
  add(margin);add(c.support.size());for(int i:c.support)add(i);
  add(c.original.size());for(const auto& e:c.original){add(e.i);add(e.j);add(e.v);}
  add(c.hybrid_atoms.size());for(const auto& a:c.hybrid_atoms){add(a.weight);add(a.indices.size());for(int i:a.indices)add(i);add(a.values.size());for(double v:a.values)add(v);}
  return key;
}
inline HybridCertificateMemo& hybrid_query_memo(const Correction& c) {
  if(!c.certificate_memo)c.certificate_memo=std::make_shared<HybridCertificateMemo>();
  return *c.certificate_memo;
}
inline bool hybrid_cache_key_bounded(const Correction& c,std::size_t limit=HybridCertificateMemo::max_bytes) {
  if(c.support.size()>128||limit<128)return false;
  std::size_t remaining=limit-128;
  auto take=[&](std::size_t count,std::size_t width){if(count>remaining/width)return false;remaining-=count*width;return true;};
  if(!take(c.support.size(),sizeof(int))||!take(c.original.size(),2*sizeof(int)+sizeof(double)))return false;
  if(!take(c.hybrid_atoms.size(),sizeof(double)+2*sizeof(std::size_t)))return false;
  for(const auto& a:c.hybrid_atoms)if(!take(a.indices.size(),sizeof(int))||!take(a.values.size(),sizeof(double)))return false;
  if(c.support.size()<=16)for(const auto* h:{&c.hessian,&c.correction})
    if(!take(h->size(),2*sizeof(int)+sizeof(double)))return false;
  return true;
}
inline std::string hybrid_query_key(const Correction& c,double rho,double margin,char kind='C') {
  std::string key(1,kind);key+=hybrid_interval_key(c,margin);
  auto add=[&](const auto& value){key.append(reinterpret_cast<const char*>(&value),sizeof(value));};
  add(rho);add(c.support_scalar);
  // The small exact path also certifies the rounded diagnostic matrices and
  // their stored difference. Include these inputs, even if a caller mutates
  // them without using hybrid_assemble_trace.
  if(c.support.size()<=16) {
    for(const auto* entries:{&c.hessian,&c.correction}) {
      add(entries->size());for(const auto& e:*entries){add(e.i);add(e.j);add(e.v);}
    }
  }
  return key;
}
inline const HybridCertificateMemo::Entry* hybrid_find_query(HybridCertificateMemo& memo,const std::string& key) {
  const auto started=std::chrono::steady_clock::now();
  const auto it=std::find_if(memo.entries.begin(),memo.entries.end(),[&](const auto& e){return e.key==key;});
  memo.lookup_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  if(it==memo.entries.end()){++memo.misses;return nullptr;}
  ++memo.hits;return &*it;
}
inline void hybrid_remember_query(HybridCertificateMemo& memo,std::string key,bool result,
    std::shared_ptr<const HybridIntervalCertificate> interval={}) {
  auto entry_bytes=[](const std::string& k,const std::shared_ptr<const HybridIntervalCertificate>& p) {
    return k.size()+sizeof(HybridCertificateMemo::Entry)+(p?p->key.size()+sizeof(HybridIntervalCertificate):0);
  };
  const auto bytes=entry_bytes(key,interval);
  if(bytes>HybridCertificateMemo::max_bytes)return;
  while(!memo.entries.empty()&&(memo.entries.size()>=HybridCertificateMemo::max_entries||memo.bytes+bytes>HybridCertificateMemo::max_bytes)) {
    memo.bytes-=entry_bytes(memo.entries.front().key,memo.entries.front().interval);
    memo.entries.erase(memo.entries.begin());++memo.evictions;
  }
  memo.bytes+=bytes;memo.entries.push_back({std::move(key),result,std::move(interval)});
}
inline void hybrid_forget_refusals(HybridCertificateMemo& memo) {
  // A refusal is not an indefiniteness theorem. Newly learned positive
  // certificates can supersede an earlier inconclusive deterministic answer.
  const auto before=memo.entries.size();
  memo.entries.erase(std::remove_if(memo.entries.begin(),memo.entries.end(),[](const auto& e){return !e.result;}),memo.entries.end());
  memo.refusals_invalidated+=before-memo.entries.size();memo.bytes=0;
  for(const auto& e:memo.entries)memo.bytes+=e.key.size()+sizeof(HybridCertificateMemo::Entry)+
    (e.interval?e.interval->key.size()+sizeof(HybridIntervalCertificate):0);
}
inline std::map<std::pair<int,int>,Rational> hybrid_exact_gram(const Correction& c) {
  const bool cache=c.support_scalar&&hybrid_cache_key_bounded(c);
  const auto key=cache?hybrid_interval_key(c,0):std::string{};
  if(cache&&c.exact_gram_snapshot&&c.exact_gram_snapshot->key==key) {
    ++hybrid_query_memo(c).gram_hits;return c.exact_gram_snapshot->gram;
  }
  if(cache)++hybrid_query_memo(c).gram_misses;
  std::map<std::pair<int,int>,Rational> h;
  for(const auto& a:c.hybrid_atoms) {
    if(!(a.weight>=0)||!std::isfinite(a.weight)||a.indices.size()!=a.values.size())throw std::runtime_error("invalid hybrid atom");
    for(std::size_t i=0;i<a.indices.size();++i)for(std::size_t j=i;j<a.indices.size();++j)
      h[{std::min(a.indices[i],a.indices[j]),std::max(a.indices[i],a.indices[j])}]+=Rational(a.weight)*Rational(a.values[i])*Rational(a.values[j]);
  }
  // One immutable snapshot per correction copy. Large-support QCQPs retain
  // their historical memory policy; at most 8192 exact entries are retained.
  if(cache&&key.size()<=HybridCertificateMemo::max_bytes&&h.size()<=8192)
    c.exact_gram_snapshot=std::make_shared<const HybridGramSnapshot>(HybridGramSnapshot{key,h});
  return h;
}
inline double hybrid_gershgorin_shift(const std::map<std::pair<int,int>,Rational>& h,double margin=0) {
  std::map<int,Rational> d,r;
  for(const auto& e:h)if(e.first.first==e.first.second)d[e.first.first]+=e.second;
    else {const Rational a=e.second<0?-e.second:e.second;r[e.first.first]+=a;r[e.first.second]+=a;}
  Rational deficit=0;
  for(const auto& e:d){Rational value=Rational(margin)+r[e.first]-e.second;if(value>deficit)deficit=value;}
  for(const auto& e:r){Rational value=Rational(margin)+e.second-d[e.first];if(value>deficit)deficit=value;}
  return rational_upward(deficit);
}
inline void hybrid_assemble_trace(Correction& c) {
  auto gram=hybrid_exact_gram(c);std::map<std::pair<int,int>,Rational> total=gram;
  for(auto e:c.original)total[{e.i,e.j}]+=Rational(e.v);
  c.correction.clear();for(auto e:gram)if(e.second!=0)c.correction.push_back({e.first.first,e.first.second,e.second.convert_to<double>()});
  c.hessian.clear();for(auto e:total)if(e.second!=0)c.hessian.push_back({e.first.first,e.first.second,e.second.convert_to<double>()});
  c.atoms=int(c.hybrid_atoms.size());
  c.rank_upper=c.rho>0?int(c.support.size()):std::min(int(c.support.size()),c.atoms);
}
struct HybridImplicitIntervals {
  using Wide=boost::multiprecision::cpp_bin_float_quad;
  int n=0;
  std::vector<SparseIntervalTriplet<long double>> native;
  std::vector<SparseIntervalTriplet<Wide>> wide;
};
template<class Float> inline sparse_psd_detail::Bounds<Float> hybrid_rational_bounds(const Rational& value) {
  using namespace sparse_psd_detail;
  const Float nearest=value.convert_to<Float>();
  if(!number_finite(nearest))throw std::runtime_error("implicit certificate conversion overflow");
  Float lo=nearest,hi=nearest;
  if(Rational(lo)>value)lo=down(lo);
  if(Rational(hi)<value)hi=up(hi);
  if(Rational(lo)>value||Rational(hi)<value)throw std::runtime_error("implicit certificate conversion did not enclose exact input");
  return {lo,hi};
}
inline HybridImplicitIntervals hybrid_implicit_intervals(const Correction& c,double margin) {
  HybridImplicitIntervals base;base.n=c.support.size();
  auto exact=hybrid_exact_gram(c);for(const auto& e:c.original)exact[{e.i,e.j}]+=Rational(e.v);
  for(int i:c.support)exact[{i,i}]-=Rational(margin);
  for(const auto& e:exact) {
    const auto a=std::lower_bound(c.support.begin(),c.support.end(),e.first.first),b=std::lower_bound(c.support.begin(),c.support.end(),e.first.second);
    if(a==c.support.end()||b==c.support.end()||*a!=e.first.first||*b!=e.first.second)throw std::runtime_error("implicit certificate index outside support");
    const int i=a-c.support.begin(),j=b-c.support.begin();
    base.native.push_back({i,j,hybrid_rational_bounds<long double>(e.second)});
    base.wide.push_back({i,j,hybrid_rational_bounds<HybridImplicitIntervals::Wide>(e.second)});
  }
  return base;
}
inline SparsePsdProof hybrid_interval_check(const HybridImplicitIntervals& base,double rho,const SparsePsdBudget& budget) {
  using namespace sparse_psd_detail;
  auto native=base.native;
  for(auto& e:native)if(e.i==e.j)e.value=add(e.value,Bounds<long double>{rho,rho});
  auto first=verified_sparse_psd_intervals(base.n,native,budget);
  if(first.certified||first.reason!="pivot-not-strictly-positive"||!(first.undecided_pivot_lower<=0&&first.undecided_pivot_upper>0)||first.seconds>=budget.max_seconds||first.updates>=budget.max_updates)return first;
  auto wide=base.wide;using W=HybridImplicitIntervals::Wide;
  for(auto& e:wide)if(e.i==e.j)e.value=add(e.value,Bounds<W>{W(rho),W(rho)});
  auto remaining=budget;remaining.max_seconds-=first.seconds;remaining.max_updates-=first.updates;
  auto second=verified_sparse_psd_intervals(base.n,wide,remaining);second.seconds+=first.seconds;second.updates+=first.updates;
  second.first_stage_reason=first.reason;return second;
}
inline bool hybrid_interval_answer_reusable(const SparsePsdProof& proof,const SparsePsdBudget& budget) {
  if(proof.certified)return true;
  // A first precision stage may report its undecided pivot while declining
  // the wider stage because the shared budget has already been exhausted.
  // The pivot reason alone is therefore insufficient to identify a complete
  // deterministic rejection.
  return proof.reason.find("budget")==std::string::npos&&
    proof.seconds<budget.max_seconds&&proof.updates<budget.max_updates;
}
// A positive objective margin is proved on the *installed* model, including
// its separately stored rank-one borders. A positive LDL pivot is not used as
// an eigenvalue bound: the proof is explicitly of A-gamma*I >= 0.
struct ObjectiveMarginCertificate {
  std::string matrix_key,certificate_key,kind="exact-ambient-Gershgorin",reason="Gershgorin-baseline";
  double lower=0,seconds=0;
  int checks=0,precision_bits=0;
  bool interval_certified=false;
};
inline SparsePsdProof objective_interval_margin_check(const HybridImplicitIntervals& base,double gamma,const SparsePsdBudget& budget) {
  if(!std::isfinite(gamma)||gamma<0)throw std::invalid_argument("invalid objective margin candidate");
  using namespace sparse_psd_detail;
  auto shifted=base;using Wide=HybridImplicitIntervals::Wide;
  for(auto& e:shifted.native)if(e.i==e.j)e.value=subtract(e.value,Bounds<long double>{gamma,gamma});
  for(auto& e:shifted.wide)if(e.i==e.j)e.value=subtract(e.value,Bounds<Wide>{Wide(gamma),Wide(gamma)});
  return hybrid_interval_check(shifted,0,budget);
}
// c.original must contain the actual scaled binary64 base coefficients and
// c.hybrid_atoms the actual stored beta/directions, not a rounded Gram trace.
// Its support must cover ALL ambient coordinates. The caller retains the
// separate function/model PSD certificates which justify the baseline zero.
inline std::shared_ptr<const ObjectiveMarginCertificate> certified_objective_implicit_lower(
    const Correction& c,std::shared_ptr<const ObjectiveMarginCertificate>& cache) {
  const auto started=std::chrono::steady_clock::now();
  const int n=c.support.size();for(int i=0;i<n;++i)if(c.support[i]!=i)throw std::invalid_argument("objective margin requires full ambient support");
  const auto key=hybrid_interval_key(c,0);
  if(cache&&cache->matrix_key==key&&cache->certificate_key==hybrid_interval_key(c,cache->lower))return cache;
  ObjectiveMarginCertificate result;result.matrix_key=key;
  result.lower=std::max(0.0,certified_objective_base_lower(n,c.original));
  for(const auto& a:c.hybrid_atoms) {
    if(!std::isfinite(a.weight)||a.weight<0||a.indices.size()!=a.values.size())throw std::invalid_argument("invalid objective margin atom");
    int previous=-1;for(std::size_t j=0;j<a.indices.size();++j) {
      if(a.indices[j]<=previous||a.indices[j]>=n||!std::isfinite(a.values[j]))throw std::invalid_argument("invalid objective margin atom layout");
      previous=a.indices[j];
    }
  }
  // Preserve the cheap established certificate when it already supplies a
  // positive margin; no eigenvalue search is needed for diagonally dominant
  // models. The bounded refinement targets otherwise unproved SPD models.
  if(n>0&&n<=128&&result.lower==0) {
    auto exact=hybrid_exact_gram(c);for(const auto& e:c.original)exact[{std::min(e.i,e.j),std::max(e.i,e.j)}]+=Rational(e.v);
    Rational diagonal=exact[{0,0}];for(int i=1;i<n;++i)diagonal=std::min(diagonal,exact[{i,i}]);
    if(diagonal>0) {try {
      const auto base=hybrid_implicit_intervals(c,0);double hi=rational_upward(diagonal),lo=0;
      std::size_t updates=0;result.reason="no-positive-shift-certified";
      for(int query=0;query<12;++query) {
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        if(elapsed>=1||updates>=4000000){result.reason="objective-margin-search-budget";break;}
        const double gamma=lo+(hi-lo)*.5;if(!(gamma>lo&&gamma<hi))break;
        SparsePsdBudget budget;budget.max_seconds=1-elapsed;budget.max_updates=4000000-updates;
        const auto proof=objective_interval_margin_check(base,gamma,budget);updates+=proof.updates;++result.checks;
        if(proof.certified) {
          lo=gamma;result.lower=gamma;result.interval_certified=true;result.precision_bits=proof.precision_bits;
          result.kind="exact-ambient-implicit-interval";result.reason="A-minus-gamma-I-positive-interval-LDL";
        } else if(proof.reason=="time-budget"||proof.reason=="operation-budget") {result.reason="objective-margin-search-budget";break;}
        else hi=gamma; // inconclusive is only a search decision, never a proof
      }
    } catch(const std::runtime_error&) {result.reason="objective-margin-interval-inconclusive";}
    } else result.reason="no-positive-diagonal-upper-bound";
  } else if(n>128)result.reason="dimension-budget-Gershgorin-baseline";
  result.certificate_key=hybrid_interval_key(c,result.lower);
  result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  cache=std::make_shared<const ObjectiveMarginCertificate>(std::move(result));return cache;
}

inline bool hybrid_cached_interval_certificate(const Correction& c,double rho,double margin) {
  const auto& proof=c.implicit_interval_certificate;
  return proof&&rho>=proof->rho&&proof->key==hybrid_interval_key(c,margin);
}
inline void hybrid_store_interval_certificate(const Correction& c,double rho,double margin,int precision_bits,bool complete=false) {
  const auto key=hybrid_interval_key(c,margin);
  if(c.implicit_interval_certificate&&c.implicit_interval_certificate->key==key&&c.implicit_interval_certificate->rho<=rho) {
    if(complete&&!c.implicit_interval_certificate->refinement_complete) {
      auto copy=*c.implicit_interval_certificate;copy.refinement_complete=true;c.implicit_interval_certificate=std::make_shared<const HybridIntervalCertificate>(std::move(copy));
    }
    return;
  }
  c.implicit_interval_certificate=std::make_shared<const HybridIntervalCertificate>(HybridIntervalCertificate{key,rho,precision_bits,complete});
  if(c.certificate_memo)hybrid_forget_refusals(*c.certificate_memo);
}
inline bool hybrid_certificate_uncached(const Correction& c,double rho,double margin,bool& reusable) {
  reusable=true;
  if(!(rho>=0)||!std::isfinite(rho)||!(margin>=0)||!std::isfinite(margin))return false;
  // C=rho P + sum beta vv' is PSD independently of the model certificate.
  for(const auto& a:c.hybrid_atoms){if(!(a.weight>=0)||!std::isfinite(a.weight)||a.indices.size()!=a.values.size())return false;for(double v:a.values)if(!std::isfinite(v))return false;}
  for(const auto& e:c.original)if(!std::isfinite(e.v))return false;
  const int n=c.support.size();
  if(c.support_scalar&&n>16&&n<=128&&hybrid_cached_interval_certificate(c,rho,margin))return true;
  auto implicit=hybrid_exact_gram(c);
  for(auto e:c.original)implicit[{e.i,e.j}]+=Rational(e.v);
  if(n>16) {
    if(rho>=hybrid_gershgorin_shift(implicit,margin))return true;
    if(n>128||!c.support_scalar)return false;
    SparsePsdBudget budget;budget.max_seconds=1;budget.max_updates=4000000;
    try {
      const auto proof=hybrid_interval_check(hybrid_implicit_intervals(c,margin),rho,budget);
      if(proof.certified)hybrid_store_interval_certificate(c,rho,margin,proof.precision_bits);
      if(!hybrid_interval_answer_reusable(proof,budget))reusable=false;
      return proof.certified;
    }catch(const std::runtime_error&){reusable=false;return false;}
  }
  std::map<int,int> pos;for(int i=0;i<n;++i)pos[c.support[i]]=i;
  std::vector<Rational> model(n*n,0);
  for(auto e:implicit){int i=pos[e.first.first],j=pos[e.first.second];model[i*n+j]+=e.second;if(i!=j)model[j*n+i]+=e.second;}
  for(int i=0;i<n;++i)model[i*n+i]+=Rational(rho)-Rational(margin);
  if(!rational_psd(std::move(model),n))return false;
  const auto stored=dense(c.hessian,c.support),original=dense(c.original,c.support);
  std::vector<Rational> stored_difference(n*n);
  for(int i=0;i<n*n;++i)stored_difference[i]=Rational(stored[i])-Rational(original[i]);
  for(int i=0;i<n;++i)stored_difference[i*n+i]+=Rational(rho);
  return exact_psd_shift(stored,n,rho-margin)&&exact_psd_shift(dense(c.correction,c.support),n,rho)&&rational_psd(std::move(stored_difference),n);
}
inline bool hybrid_certificate(const Correction& c,double rho,double margin=0,bool* reusable_out=nullptr) {
  // Keep key construction itself bounded for very large original functions.
  // The optimization targets the already bounded medium-support oracle;
  // large-support certificates retain their historical path and memory use.
  if(!c.support_scalar||!hybrid_cache_key_bounded(c)) {
    bool reusable=false;const bool result=hybrid_certificate_uncached(c,rho,margin,reusable);
    if(reusable_out)*reusable_out=reusable;return result;
  }
  auto& memo=hybrid_query_memo(c);const auto key=hybrid_query_key(c,rho,margin);
  if(c.support.size()>16&&std::isfinite(rho)&&rho>=0&&std::isfinite(margin)&&margin>=0&&
      hybrid_cached_interval_certificate(c,rho,margin)) {if(reusable_out)*reusable_out=true;return true;}
  if(const auto* hit=hybrid_find_query(memo,key)) {
    const bool result=hit->result;const auto p=hit->interval;
    if(result&&p) {
      if(!c.implicit_interval_certificate||c.implicit_interval_certificate->key!=p->key||
          p->rho<c.implicit_interval_certificate->rho||
          (p->rho==c.implicit_interval_certificate->rho&&p->refinement_complete)) {
        c.implicit_interval_certificate=p;hybrid_forget_refusals(memo);
      }
    }
    if(reusable_out)*reusable_out=true;return result;
  }
  const auto started=std::chrono::steady_clock::now();bool reusable=false;
  const bool result=hybrid_certificate_uncached(c,rho,margin,reusable);
  memo.certificate_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  const auto interval=result&&c.implicit_interval_certificate&&
    c.implicit_interval_certificate->key==hybrid_interval_key(c,margin)?c.implicit_interval_certificate:nullptr;
  if(reusable)hybrid_remember_query(memo,key,result,interval);
  else ++memo.inconclusive;
  if(reusable_out)*reusable_out=reusable;return result;
}
inline double hybrid_certified_residual(Correction& c,double cap,double margin=0) {
  const auto refinement_started=std::chrono::steady_clock::now();
  hybrid_assemble_trace(c);
  if(!hybrid_certificate(c,cap,margin)) {
    auto m=hybrid_exact_gram(c);for(auto e:c.original)m[{e.i,e.j}]+=Rational(e.v);
    cap=std::max(cap,hybrid_gershgorin_shift(m,margin));
    for(int trial=0;trial<16&&!hybrid_certificate(c,cap,margin);++trial)
      cap=std::nextafter(std::max(cap,64*std::numeric_limits<double>::epsilon())*2,INFINITY);
    if(!hybrid_certificate(c,cap,margin))throw std::runtime_error("hybrid PSD certificate failed");
  }
  if(c.support.size()>16){
    auto implicit=hybrid_exact_gram(c);for(auto e:c.original)implicit[{e.i,e.j}]+=Rational(e.v);
    double hi=std::min(cap,hybrid_gershgorin_shift(implicit,margin));
    if(c.support.size()>128||!c.support_scalar)return hi;
    if(hybrid_cached_interval_certificate(c,0,margin))return 0;
    if(c.implicit_interval_certificate&&c.implicit_interval_certificate->refinement_complete&&c.implicit_interval_certificate->key==hybrid_interval_key(c,margin))return std::min(hi,c.implicit_interval_certificate->rho);
    if(c.implicit_interval_certificate&&c.implicit_interval_certificate->key==hybrid_interval_key(c,margin))hi=std::min(hi,c.implicit_interval_certificate->rho);
    if(std::chrono::duration<double>(std::chrono::steady_clock::now()-refinement_started).count()>=1)return hi;
    HybridImplicitIntervals base;try{base=hybrid_implicit_intervals(c,margin);}catch(const std::runtime_error&){return hi;}
    std::size_t updates=0;double lo=0;bool complete=true;
    for(int it=0;it<49;++it) {
      const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-refinement_started).count();
      if(elapsed>=1||updates>=4000000){complete=false;break;}
      const double trial=it==0?0:lo+(hi-lo)*.5;if(it&&!(trial>lo&&trial<hi))break;
      SparsePsdBudget budget;budget.max_seconds=1-elapsed;budget.max_updates=4000000-updates;
      const auto proof=hybrid_interval_check(base,trial,budget);updates+=proof.updates;
      if(proof.certified){hi=trial;hybrid_store_interval_certificate(c,trial,margin,proof.precision_bits);if(trial==0){hybrid_store_interval_certificate(c,0,margin,proof.precision_bits,true);return 0;}}
      else if(proof.reason=="time-budget"||proof.reason=="operation-budget"){complete=false;break;}
      else if(it)lo=trial;
    }
    if(complete&&c.implicit_interval_certificate&&c.implicit_interval_certificate->key==hybrid_interval_key(c,margin))hybrid_store_interval_certificate(c,hi,margin,c.implicit_interval_certificate->precision_bits,true);
    return hi;
  }
  double lo=0,hi=cap;
  if(hybrid_certificate(c,0,margin))return 0;
  for(int it=0;it<48;++it){double mid=lo+(hi-lo)*0.5;if(hybrid_certificate(c,mid,margin))hi=mid;else lo=mid;}
  return hi;
}
inline Correction construct_hybrid(int ambient,const std::vector<SymTriplet>& input,double margin=0) {
  if(!std::isfinite(margin)||margin<0)throw std::runtime_error("invalid hybrid function margin");
  auto c=construct(ambient,input,"scalar",false,16,true);c.family="hybrid";c.support_scalar=true;
  // A request for numerical/model strong convexity must not change a source
  // function which already has an exact PSD certificate. Any missing reduced
  // margin belongs to the separately traced proximal/KKT policy.
  if(c.original_convex) {
    c.rho=c.rho_limit=0;hybrid_assemble_trace(c);
    c.certificate="hybrid-original-PSD-no-correction";
    return c;
  }
  // Initial certified scalar metric; no unrequested atom construction.
  c.rho=std::max(c.rho,margin);hybrid_assemble_trace(c);
  c.rho_limit=c.rho;
  if(!hybrid_certificate(c,c.rho,margin))throw std::runtime_error("initial hybrid scalar certificate failed");
  c.certificate="hybrid-initial-exact-scalar";c.rank_upper=c.rho>0?int(c.support.size()):0;
  return c;
}
inline Correction construct_support_scalar(int ambient,const std::vector<SymTriplet>& input,double margin=0) {
  auto c=construct_hybrid(ambient,input,margin);c.family="scalar";
  c.certificate="support-scalar-initial-exact-certificate";
  return c;
}
inline bool hybrid_enrich(Correction& c,int ambient,int max_atoms,int support_cap,double scalar_decay,double margin=0) {
  if(c.family!="hybrid"||c.original_convex||int(c.hybrid_atoms.size())>=max_atoms||c.support.empty())return false;
  Correction trial=c;HybridAtom atom;
  const auto gram=hybrid_exact_gram(c);auto total=gram;
  for(auto e:c.original)total[{e.i,e.j}]+=Rational(e.v);
  const int n=c.support.size();
  if(n<=16) {
    auto h=dense(c.hessian,c.support);std::vector<double> values(n),work(std::max(1,3*n));int info=0,lwork=work.size();char job='V',up='U';
    dsyev_(&job,&up,&n,h.data(),&n,values.data(),work.data(),&lwork,&info);
    if(info||!(values[0]<-64*std::numeric_limits<double>::epsilon()))return false;
    std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return std::abs(h[a])>std::abs(h[b]);});
    for(int k=0;k<std::min(n,std::max(1,support_cap));++k){atom.indices.push_back(c.support[order[k]]);atom.values.push_back(h[order[k]]);}
    // Canonical index order is required by exact Gram assembly.
    std::vector<std::pair<int,double>> v;for(std::size_t j=0;j<atom.indices.size();++j)v.push_back({atom.indices[j],atom.values[j]});
    std::sort(v.begin(),v.end());atom.indices.clear();atom.values.clear();double norm=0;
    for(auto e:v){atom.indices.push_back(e.first);atom.values.push_back(e.second);norm+=e.second*e.second;}
    if(!(norm>0))return false;for(auto& value:atom.values)value/=std::sqrt(norm);
    long double rayleigh=0;
    for(auto e:total)for(std::size_t i=0;i<atom.indices.size();++i)if(atom.indices[i]==e.first.first)
      for(std::size_t j=0;j<atom.indices.size();++j)if(atom.indices[j]==e.first.second)
        rayleigh+=(e.first.first==e.first.second?1:2)*e.second.convert_to<long double>()*atom.values[i]*atom.values[j];
    if(!(rayleigh<0))return false;
    atom.weight=upward(-rayleigh*(1-std::clamp(scalar_decay,0.0,0.95)));
  } else {
    // Bounded sparse candidates on large supports: negative coordinates or
    // bilinear edge directions. No dense eigensystem or unverified Ritz floor.
    double worst=0;int first=-1,second=-1;double sign=0;
    std::map<int,double> diag;for(auto e:total)if(e.first.first==e.first.second)diag[e.first.first]=e.second.convert_to<double>();
    for(auto e:diag)if(e.second<worst){worst=e.second;first=e.first;second=-1;}
    if(support_cap>=2)for(auto e:total)if(e.first.first!=e.first.second){const double b=e.second.convert_to<double>(),rq=0.5*(diag[e.first.first]+diag[e.first.second]-2*std::abs(b));if(rq<worst){worst=rq;first=e.first.first;second=e.first.second;sign=b>0?-1:1;}}
    if(first<0)return false;
    atom.indices={first};atom.values={1};
    if(second>=0){atom.indices.push_back(second);atom.values.push_back(sign);atom.weight=std::abs(total[{first,second}].convert_to<double>());}
    else atom.weight=-worst*(1-std::clamp(scalar_decay,0.0,0.95));
  }
  if(!(atom.weight>0)||!std::isfinite(atom.weight))return false;
  trial.hybrid_atoms.push_back(atom);trial.rho=hybrid_certified_residual(trial,c.rho,margin);
  // A true scalar increase caused only by adding an atom is rejected.
  if(trial.rho>c.rho||!hybrid_certificate(trial,trial.rho,margin))return false;
  trial.refined=true;trial.atom_batches=c.atom_batches+1;trial.atom_certificate_checks=c.atom_certificate_checks+1;
  trial.certificate="hybrid-exact-atoms-plus-certified-residual";trial.rank_upper=trial.rho>0?int(trial.support.size()):std::min(ambient,trial.atoms);
  c=std::move(trial);return true;
}
inline bool hybrid_downweight(Correction& c,int ambient,std::size_t index,double factor,double margin=0) {
  if(c.family!="hybrid"||index>=c.hybrid_atoms.size()||!std::isfinite(factor)||factor<0||factor>=1)return false;
  // The rejected copy previously discarded its proof and repeated the same
  // exact Gram assembly/LDL on every activity event. A failed deterministic
  // query can be reused even though the optimization metric was rolled back.
  auto* memo=c.support_scalar&&hybrid_cache_key_bounded(c)?&hybrid_query_memo(c):nullptr;std::string key;
  if(memo) {
    key=hybrid_query_key(c,c.rho,margin,'D');
    key.append(reinterpret_cast<const char*>(&ambient),sizeof(ambient));
    key.append(reinterpret_cast<const char*>(&index),sizeof(index));
    key.append(reinterpret_cast<const char*>(&factor),sizeof(factor));
    if(hybrid_find_query(*memo,key)){++memo->downweight_hits;return false;}
  }
  Correction trial=c;
  const double next_weight=factor==0?0:trial.hybrid_atoms[index].weight*factor;
  if(factor>0&&!(next_weight<trial.hybrid_atoms[index].weight))return false;
  const bool removed=next_weight==0;
  if(removed)trial.hybrid_atoms.erase(trial.hybrid_atoms.begin()+index);
  else trial.hybrid_atoms[index].weight=next_weight;
  hybrid_assemble_trace(trial);
  // Keep residual rho unchanged: if decreasing the atom would violate the
  // certificate, rollback rather than silently compensating by a scalar jump.
  bool reusable=false;
  if(!hybrid_certificate(trial,trial.rho,margin,&reusable)) {
    if(reusable&&memo)hybrid_remember_query(*memo,std::move(key),false);
    return false;
  }
  if(!removed)trial.hybrid_atoms[index].strikes=0;
  trial.refined=true;trial.certificate=removed?"hybrid-certified-atom-removal":"hybrid-certified-atom-downweight";
  trial.rank_upper=trial.rho>0?int(trial.support.size()):std::min(ambient,trial.atoms);c=std::move(trial);return true;
}
} // namespace lazyipdca::qcqp::study
