#pragma once
#include <vector>
namespace lazyipdca::qcqp::study {
struct Correction;
// Declare the selection-only interface before the correction header: it also
// includes the selector, which calls this interface from its inline methods.
struct HybridSpectrum {
  double lambda_min=0;
  std::vector<std::vector<double>> vectors;
  std::vector<double> rayleigh;
  int raw_candidates=0,orthogonalized_candidates=0,dependent_dropped=0,nonnegative_projected_dropped=0;
};
inline std::vector<double> hybrid_apply_support(const Correction&,const std::vector<double>&);
inline HybridSpectrum hybrid_spectrum(const Correction&,int,int,int,double);
}
#include "lazyipdca_qcqp/convexification_study.hpp"
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace lazyipdca::qcqp::study {
// These are numerical Rayleigh/Ritz estimates for candidate selection only.
// They are never lower bounds or substitutes for hybrid_certificate().
namespace hybrid_spectrum_detail {
inline double dot(const std::vector<double>& a,const std::vector<double>& b) {
  if(a.size()!=b.size())throw std::runtime_error("hybrid spectrum vector dimension mismatch");
  long double value=0;for(std::size_t j=0;j<a.size();++j)value+=(long double)a[j]*b[j];
  const double out=double(value);
  if(!std::isfinite(out))throw std::runtime_error("hybrid spectrum nonfinite inner product");
  return out;
}
inline double normalize(std::vector<double>& x) {
  const double norm=std::sqrt(std::max(0.0,dot(x,x)));
  if(norm>0)for(auto& value:x)value/=norm;
  return norm;
}
// A pool from multiple starts need not be orthogonal in a repeated eigenspace.
// This is candidate diversification only, never a PSD certificate. Its
// roundoff-scale dependency threshold is unrelated to any KKT tolerance.
inline bool orthogonalize_candidate(std::vector<double>& vector,
                                   const std::vector<std::vector<double>>& basis) {
  for(int pass=0;pass<2;++pass)for(const auto& old:basis) {
    const double component=dot(vector,old);
    for(std::size_t i=0;i<vector.size();++i)vector[i]-=component*old[i];
  }
  const double norm=normalize(vector);
  return norm>128*std::numeric_limits<double>::epsilon()*std::max(1.0,std::sqrt(double(vector.size())));
}
struct Operator {
  int n=0;
  struct Entry {int i=0,j=0;double value=0;};
  struct Atom {double weight=0;std::vector<int> index;std::vector<double> value;};
  std::vector<Entry> entries;
  std::vector<Atom> atoms;
  explicit Operator(const Correction& c):n(int(c.support.size())) {
    if(!std::is_sorted(c.support.begin(),c.support.end())||
       std::adjacent_find(c.support.begin(),c.support.end())!=c.support.end())
      throw std::runtime_error("hybrid spectrum support is not sorted unique");
    auto compact=[&](int global){const auto it=std::lower_bound(c.support.begin(),c.support.end(),global);
      if(it==c.support.end()||*it!=global)throw std::runtime_error("hybrid spectrum coefficient outside support");
      return int(it-c.support.begin());};
    entries.reserve(c.original.size());
    for(const auto& e:c.original){if(!std::isfinite(e.v))throw std::runtime_error("hybrid spectrum nonfinite Hessian");entries.push_back({compact(e.i),compact(e.j),e.v});}
    atoms.reserve(c.hybrid_atoms.size());
    for(const auto& a:c.hybrid_atoms) {
      if(!std::isfinite(a.weight)||a.weight<0||a.indices.size()!=a.values.size())
        throw std::runtime_error("hybrid spectrum invalid atom");
      Atom atom;atom.weight=a.weight;atom.value=a.values;atom.index.reserve(a.indices.size());
      for(std::size_t j=0;j<a.indices.size();++j){if(!std::isfinite(a.values[j]))throw std::runtime_error("hybrid spectrum nonfinite atom");atom.index.push_back(compact(a.indices[j]));}
      atoms.push_back(std::move(atom));
    }
  }
  std::vector<double> apply(const std::vector<double>& x) const {
    if(x.size()!=std::size_t(n))throw std::runtime_error("hybrid spectrum operator dimension mismatch");
    std::vector<long double> acc(n,0);
    for(double value:x)if(!std::isfinite(value))throw std::runtime_error("hybrid spectrum nonfinite vector");
    for(const auto& e:entries){acc[e.i]+=(long double)e.value*x[e.j];if(e.i!=e.j)acc[e.j]+=(long double)e.value*x[e.i];}
    for(const auto& a:atoms){long double projection=0;for(std::size_t j=0;j<a.index.size();++j)projection+=(long double)a.value[j]*x[a.index[j]];
      const long double beta=(long double)a.weight*projection;for(std::size_t j=0;j<a.index.size();++j)acc[a.index[j]]+=beta*a.value[j];}
    std::vector<double> out(n);for(int j=0;j<n;++j){out[j]=double(acc[j]);if(!std::isfinite(out[j]))throw std::runtime_error("hybrid spectrum operator overflow");}
    return out;
  }
};
inline std::vector<double> diagonalize(std::vector<double>& a,int n) {
  std::vector<double> eigenvalues(n),work(std::max(1,3*n));
  int info=0,lwork=int(work.size());char vectors='V',triangle='U';
  dsyev_(&vectors,&triangle,&n,a.data(),&n,eigenvalues.data(),work.data(),&lwork,&info);
  if(info)throw std::runtime_error("hybrid spectrum LAPACK eigensystem failed");
  for(double value:eigenvalues)if(!std::isfinite(value))throw std::runtime_error("hybrid spectrum nonfinite eigenvalue");
  for(double value:a)if(!std::isfinite(value))throw std::runtime_error("hybrid spectrum nonfinite eigenvector");
  return eigenvalues;
}
inline std::uint64_t next(std::uint64_t& state) {
  state+=0x9e3779b97f4a7c15ULL;std::uint64_t z=state;
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;
  return z^(z>>31);
}
} // namespace hybrid_spectrum_detail

inline std::vector<double> hybrid_apply_support(const Correction& c,const std::vector<double>& x) {
  return hybrid_spectrum_detail::Operator(c).apply(x);
}

inline HybridSpectrum hybrid_spectrum_uncached(const Correction& c,int max_candidates,
                                     int lanczos_steps,int starts,double negative_tolerance) {
  using namespace hybrid_spectrum_detail;
  if(max_candidates<0||max_candidates>256||lanczos_steps<2||lanczos_steps>256||
     starts<1||starts>16||!std::isfinite(negative_tolerance)||negative_tolerance<0)
    throw std::runtime_error("hybrid spectrum invalid bounded-search settings");
  const Operator op(c);const int n=op.n;HybridSpectrum out;if(!n)return out;
  out.lambda_min=std::numeric_limits<double>::infinity();
  std::vector<std::pair<double,std::vector<double>>> candidates;
  auto assess=[&](std::vector<double> vector) {
    if(!(normalize(vector)>0))throw std::runtime_error("hybrid spectrum zero Ritz vector");
    const double rayleigh=dot(vector,op.apply(vector));
    out.lambda_min=std::min(out.lambda_min,rayleigh);
    if(rayleigh < -negative_tolerance&&max_candidates) candidates.emplace_back(rayleigh,std::move(vector));
  };
  if(n<=32) {
    std::vector<double> matrix(std::size_t(n)*n,0),unit(n,0);
    for(int j=0;j<n;++j){unit[j]=1;const auto column=op.apply(unit);unit[j]=0;for(int i=0;i<n;++i)matrix[i+std::size_t(j)*n]=column[i];}
    diagonalize(matrix,n);
    // DSYEV stores normalized eigenvectors in columns, in ascending order.
    for(int column=0;column<n;++column){std::vector<double> vector(n);for(int i=0;i<n;++i)vector[i]=matrix[i+std::size_t(column)*n];assess(std::move(vector));}
  } else {
    const int capacity=std::min(n,lanczos_steps);
    for(int start=0;start<starts;++start) {
      std::uint64_t seed=0x487962726964ULL+std::uint64_t(start)*0x9e3779b97f4a7c15ULL;
      std::vector<double> q(n);
      for(auto& value:q)value=double(next(seed)>>11)*(1.0/9007199254740992.0)-0.5;
      if(!(normalize(q)>0))throw std::runtime_error("hybrid spectrum zero deterministic start");
      std::vector<std::vector<double>> basis;basis.reserve(capacity);
      std::vector<double> projected(std::size_t(capacity)*capacity,0);
      for(int j=0;j<capacity;++j) {
        basis.push_back(std::move(q));auto residual=op.apply(basis.back());
        const double scale=std::sqrt(std::max(0.0,dot(residual,residual)));
        // Compute B'AB, rather than treating reorthogonalization errors as an
        // exactly tridiagonal Lanczos recurrence.
        for(int i=0;i<=j;++i){const double entry=dot(basis[i],residual);projected[i+std::size_t(j)*capacity]=entry;projected[j+std::size_t(i)*capacity]=entry;}
        // Two full passes keep the Ritz pool independent even for clustered
        // spectra. Basis dimension and number of starts are explicitly capped.
        for(int pass=0;pass<2;++pass)for(const auto& old:basis){const double component=dot(old,residual);for(int i=0;i<n;++i)residual[i]-=component*old[i];}
        const double norm=normalize(residual);
        if(j+1==capacity||norm<=128*std::numeric_limits<double>::epsilon()*std::max(1.0,scale))break;
        q=std::move(residual);
      }
      const int m=int(basis.size());std::vector<double> small(std::size_t(m)*m);
      for(int j=0;j<m;++j)for(int i=0;i<m;++i)small[i+std::size_t(j)*m]=projected[i+std::size_t(j)*capacity];
      const auto values=diagonalize(small,m);
      // Keep only the finite requested negative pool, but always assess the
      // leftmost Ritz vector so positive-curvature estimates are available.
      for(int column=0;column<m;++column) {
        if(column>0&&(column>=max_candidates||values[column]>=-negative_tolerance))break;
        std::vector<double> vector(n,0);
        for(int j=0;j<m;++j){const double weight=small[j+std::size_t(column)*m];for(int i=0;i<n;++i)vector[i]+=weight*basis[j][i];}
        assess(std::move(vector));
      }
    }
  }
  if(!std::isfinite(out.lambda_min))throw std::runtime_error("hybrid spectrum produced no finite estimate");
  std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.first<b.first;});
  out.raw_candidates=int(candidates.size());
  for(auto& candidate:candidates) {
    if(!orthogonalize_candidate(candidate.second,out.vectors)){++out.dependent_dropped;continue;}
    if(!out.vectors.empty())++out.orthogonalized_candidates;
    // Projection may change the Rayleigh value or remove negative curvature.
    // Re-evaluate the actual implicit Q+atoms operator before later scoring.
    const double rayleigh=dot(candidate.second,op.apply(candidate.second));
    if(!(rayleigh < -negative_tolerance)){++out.nonnegative_projected_dropped;continue;}
    out.lambda_min=std::min(out.lambda_min,rayleigh);
    // A canonical sign makes repeated runs and tied candidate order stable.
    const auto largest=std::max_element(candidate.second.begin(),candidate.second.end(),[](double a,double b){return std::abs(a)<std::abs(b);});
    if(largest!=candidate.second.end()&&*largest<0)for(auto& value:candidate.second)value=-value;
    out.rayleigh.push_back(rayleigh);out.vectors.push_back(std::move(candidate.second));
    if(int(out.vectors.size())>=max_candidates)break;
  }
  return out;
}

// The spectral oracle has fixed deterministic starts and no wall-time budget.
// It depends only on Q+sum(beta vv') and the bounded search settings. Primal
// gradients/displacements are deliberately excluded: the caller must always
// recompute activity, composite scores and admissibility at the current point.
struct HybridSpectrumCache {
  struct Entry {std::string key;HybridSpectrum value;std::size_t bytes=0;};
  std::vector<Entry> entries;std::size_t bytes=0;
  static constexpr std::size_t max_entries=32,max_bytes=1024*1024;
};
inline HybridSpectrum hybrid_spectrum(const Correction& c,int max_candidates,
                                     int lanczos_steps,int starts,double negative_tolerance) {
  if(!c.support_scalar||!hybrid_cache_key_bounded(c,HybridSpectrumCache::max_bytes))
    return hybrid_spectrum_uncached(c,max_candidates,lanczos_steps,starts,negative_tolerance);
  auto key=hybrid_interval_key(c,0);
  auto add=[&](const auto& value){key.append(reinterpret_cast<const char*>(&value),sizeof(value));};
  add(max_candidates);add(lanczos_steps);add(starts);add(negative_tolerance);
  if(!c.spectrum_cache)c.spectrum_cache=std::make_shared<HybridSpectrumCache>();
  auto& cache=*c.spectrum_cache;auto& counters=hybrid_query_memo(c);
  const auto hit=std::find_if(cache.entries.begin(),cache.entries.end(),[&](const auto& e){return e.key==key;});
  if(hit!=cache.entries.end()){++counters.spectrum_hits;return hit->value;}
  ++counters.spectrum_misses;const auto started=std::chrono::steady_clock::now();
  auto out=hybrid_spectrum_uncached(c,max_candidates,lanczos_steps,starts,negative_tolerance);
  counters.spectrum_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  std::size_t bytes=key.size()+sizeof(HybridSpectrumCache::Entry)+out.rayleigh.size()*sizeof(double);
  for(const auto& v:out.vectors)bytes+=sizeof(v)+v.size()*sizeof(double);
  if(bytes<=HybridSpectrumCache::max_bytes) {
    while(!cache.entries.empty()&&(cache.entries.size()>=HybridSpectrumCache::max_entries||cache.bytes+bytes>HybridSpectrumCache::max_bytes)) {
      cache.bytes-=cache.entries.front().bytes;cache.entries.erase(cache.entries.begin());
    }
    cache.bytes+=bytes;cache.entries.push_back({std::move(key),out,bytes});
  }
  counters.spectrum_entries=cache.entries.size();counters.spectrum_bytes=cache.bytes;
  return out;
}
} // namespace lazyipdca::qcqp::study
