#include "lazyipdca_qcqp/native_restoration_trace.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#if defined(LAZYIPDCA_HAVE_HSL) || defined(LAZYIPDCA_HAVE_MUMPS)
#include "stable_upper_assembly.hpp"
#endif
#ifdef LAZYIPDCA_HAVE_HSL
#include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <iostream>
#include "mumps_memory_recovery.hpp"
#include "mumps_ordering_guard.hpp"
#include <memory>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

extern "C" {
void dsytrf_(const char* uplo, const int* n, double* a, const int* lda, int* ipiv,
             double* work, const int* lwork, int* info);
void dsytrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda,
             const int* ipiv, double* b, const int* ldb, int* info);
}

#ifdef LAZYIPDCA_HAVE_LDL
extern "C" {
#include <ldl.h>
}
#endif
#ifdef LAZYIPDCA_HAVE_QDLDL
#include <qdldl.h>
#endif
#ifdef LAZYIPDCA_HAVE_METIS
extern "C" {
#include <metis.h>
}
#endif
#ifdef LAZYIPDCA_HAVE_MUMPS
#include <dmumps_c.h>
#endif

namespace lazyipdca::qcqp {
namespace {

struct AggregatedEntry {
  int i=0, j=0;
  double value=0.0;
};
using EntryList = std::vector<AggregatedEntry>;

#if defined(LAZYIPDCA_HAVE_HSL) || defined(LAZYIPDCA_HAVE_MUMPS)
struct LinearPhaseTime {
  double& total;
  std::chrono::steady_clock::time_point started=std::chrono::steady_clock::now();
  bool stopped=false;
  explicit LinearPhaseTime(double& seconds):total(seconds){}
  void stop(){if(!stopped){total+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();stopped=true;}}
  ~LinearPhaseTime(){stop();}
};
#endif

inline std::uint64_t upper_key(int i, int j) {
  // Column-major key: sorting yields CSC-ready order (column, then row).
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(j)) << 32) |
         static_cast<std::uint32_t>(i);
}

inline int upper_key_row(std::uint64_t key) {
  return static_cast<int>(static_cast<std::uint32_t>(key));
}

inline int upper_key_col(std::uint64_t key) {
  return static_cast<int>(static_cast<std::uint32_t>(key >> 32));
}

constexpr bool stable_counting_upper_order(int n, std::size_t entry_count) {
  // With fewer than two entries there is no duplicate accumulation order to
  // preserve.  Otherwise this is exactly the counting-sort predicate used by
  // sort_upper_entries below.
  if(entry_count<2) return true;
  const std::size_t ns=static_cast<std::size_t>(std::max(0,n));
  return n>0 && ns<=1000000u &&
      ns<=std::max<std::size_t>(4096u,8u*entry_count);
}

static_assert(stable_counting_upper_order(1,0));
static_assert(stable_counting_upper_order(1000000,125000));
static_assert(!stable_counting_upper_order(1000001,1000001));

void sort_upper_entries(int n, EntryList& entries) {
  if(entries.size()<2) return;

  // The triplet coordinates are bounded integers.  On the medium-size KKT
  // systems for which repeated assembly dominates the solve, two stable
  // counting passes (row, then column) produce the same CSC-ready (column,row)
  // order in O(nnz+n), without the O(nnz log nnz) comparison sort paid at every
  // Newton iteration.  Avoid an O(n) workspace on extremely large, very sparse
  // matrices, where comparison sorting remains cheaper.
  const std::size_t m=entries.size();
  const bool use_counting=stable_counting_upper_order(n,m);
  if(!use_counting) {
    std::sort(entries.begin(),entries.end(),[](const AggregatedEntry& a,
                                               const AggregatedEntry& b){
      return upper_key(a.i,a.j)<upper_key(b.i,b.j);
    });
    return;
  }

  EntryList buffer(m);
  const std::size_t ns=static_cast<std::size_t>(n);
  std::vector<std::size_t> offsets(ns+1,0);

  // Least-significant key first: stable row ordering.
  for(const auto& e:entries) ++offsets[static_cast<std::size_t>(e.i)+1];
  for(std::size_t k=1;k<offsets.size();++k) offsets[k]+=offsets[k-1];
  for(const auto& e:entries)
    buffer[offsets[static_cast<std::size_t>(e.i)]++]=e;

  // Most-significant key second: stable column ordering.
  std::fill(offsets.begin(),offsets.end(),0);
  for(const auto& e:buffer) ++offsets[static_cast<std::size_t>(e.j)+1];
  for(std::size_t k=1;k<offsets.size();++k) offsets[k]+=offsets[k-1];
  for(const auto& e:buffer)
    entries[offsets[static_cast<std::size_t>(e.j)]++]=e;
}

EntryList aggregate_upper(int n, const std::vector<SymmetricTriplet>& entries,
                          int primal_dimension=-1, double primal_reg=0.0,
                          double dual_reg=0.0, bool preserve_zeros=false, int dual_end=-1) {
  EntryList tmp;
  tmp.reserve(entries.size() + (primal_dimension >= 0 ? static_cast<std::size_t>(n) : 0));
  for (const auto& e : entries) {
    if (e.i < 0 || e.i >= n || e.j < 0 || e.j >= n || !std::isfinite(e.value)) continue;
    int i=e.i, j=e.j;
    if (i>j) std::swap(i,j);
    tmp.push_back({i,j,e.value});
  }
  if (primal_dimension >= 0) {
    for (int i=0;i<n;++i) {
      if(i<primal_dimension)tmp.push_back({i,i,primal_reg});
      else if(dual_end<0||i<dual_end)tmp.push_back({i,i,-dual_reg});
    }
  }
  sort_upper_entries(n,tmp);
  EntryList out;
  out.reserve(tmp.size());
  for (std::size_t k=0;k<tmp.size();) {
    const int i=tmp[k].i, j=tmp[k].j;
    double sum=0.0;
    do { sum += tmp[k].value; ++k; } while (k<tmp.size() && tmp[k].i==i && tmp[k].j==j);
    if (preserve_zeros || std::abs(sum)>=1e-30) out.push_back({i,j,sum});
  }
  return out;
}

[[maybe_unused]] std::vector<std::uint64_t> entry_pattern(const EntryList& a) {
  std::vector<std::uint64_t> out; out.reserve(a.size());
  for (const auto& e:a) out.push_back(upper_key(e.i,e.j));
  return out;
}

inline bool pattern_contains(const std::vector<std::uint64_t>& envelope,
                             const std::vector<std::uint64_t>& current) {
  return std::includes(envelope.begin(),envelope.end(),current.begin(),current.end());
}

inline std::vector<std::uint64_t>
pattern_union(const std::vector<std::uint64_t>& a,
              const std::vector<std::uint64_t>& b) {
  std::vector<std::uint64_t> out;
  out.reserve(a.size()+b.size());
  std::set_union(a.begin(),a.end(),b.begin(),b.end(),std::back_inserter(out));
  return out;
}

inline double symmetric_inf_norm(int n, const EntryList& a) {
  std::vector<double> rows(static_cast<std::size_t>(n),0.0);
  for (const auto& e:a) {
    const double av=std::abs(e.value);
    rows[e.i]+=av;
    if(e.i!=e.j) rows[e.j]+=av;
  }
  double m=0.0; for(double v:rows) m=std::max(m,v);
  return m;
}

inline void symmetric_residual_into(const EntryList& a, const Vec& x,
                                    const Vec& b, Vec& r) {
  r = b;
  for(const auto& e:a){
    r[e.i]-=e.value*x[e.j];
    if(e.i!=e.j) r[e.j]-=e.value*x[e.i];
  }
}

inline Vec symmetric_residual(const EntryList& a, const Vec& x, const Vec& b) {
  Vec r;
  symmetric_residual_into(a,x,b,r);
  return r;
}

inline double backward_error_into_known_binf(const EntryList& a, double ainf,
                                             double binf, const Vec& x,
                                             const Vec& b, Vec& residual,
                                             double* rinf=nullptr) {
  symmetric_residual_into(a,x,b,residual);
  const double ri=inf_norm(residual);
  if(rinf) *rinf=ri;
  const double den=ainf*inf_norm(x)+binf;
  return ri/std::max(den,std::numeric_limits<double>::min());
}

inline double backward_error_into(const EntryList& a, double ainf,
                                  const Vec& x, const Vec& b, Vec& residual,
                                  double* rinf=nullptr) {
  return backward_error_into_known_binf(a,ainf,inf_norm(b),x,b,residual,rinf);
}

inline double backward_error(const EntryList& a, double ainf,
                             const Vec& x, const Vec& b, double* rinf=nullptr) {
  Vec residual;
  return backward_error_into(a,ainf,x,b,residual,rinf);
}

#include "hsl_ma27_factorization.hpp"

// Delay a small set of globally shared primal coordinates. Equality rows
// with local primal neighbours may then be eliminated with their local block,
// before those shared coordinates. This is only a candidate ordering: a zero
// pivot retries the historical all-primal-dependencies order on the SAME matrix
// and SAME backend. Inertia, refinement and residual checks remain mandatory.
[[maybe_unused]] std::vector<char> dense_primal_border_order(
    int n,int np,const std::vector<std::uint64_t>& pattern,
    std::vector<int>& perm,std::vector<int>& iperm,bool enabled) {
  if(!enabled || n<512 || np<=0 || np>=n)return {};
  std::vector<std::size_t> degree(static_cast<std::size_t>(np),0);
  for(const auto key:pattern) {
    const int i=upper_key_row(key),j=upper_key_col(key);
    if(i==j)continue;
    if(i<np)++degree[i];
    if(j<np)++degree[j];
  }
  std::vector<char> hubs(static_cast<std::size_t>(n),0);
  int count=0;
  for(int j=0;j<np;++j)if(degree[j]>=64) {hubs[j]=1;++count;}
  if(count==0 || count*8>=np)return {};
  std::stable_partition(perm.begin(),perm.end(),[&](int v){return !hubs[v];});
  for(int j=0;j<n;++j)iperm[perm[j]]=j;
  return hubs;
}

[[maybe_unused]] void local_primal_dependencies(
    std::vector<std::vector<int>>& neighbours,const std::vector<char>& hubs) {
  if(hubs.empty())return;
  for(auto& row:neighbours) {
    const bool local=std::any_of(row.begin(),row.end(),[&](int p){return !hubs[p];});
    if(local)row.erase(std::remove_if(row.begin(),row.end(),
        [&](int p){return hubs[p];}),row.end());
  }
}

class DenseLapackFactor final : public SymmetricFactorization {
 public:
  explicit DenseLapackFactor(const Settings& st) : st_(st) {}

  bool factor(int n, int primal_dimension, const std::vector<SymmetricTriplet>& entries,
              std::string& message) override {
    probe_ready_=false;curvature_defect_=0;
    n_ = n;
    a_.assign(static_cast<std::size_t>(n)*n, 0.0);
    piv_.assign(n, 0);
    matrix_ = st_.kkt_adaptive_regularization
        ? aggregate_upper(n, entries)
        : aggregate_upper(n, entries, primal_dimension,
                          st_.kkt_primal_regularization,
                          st_.kkt_dual_regularization);
    matrix_inf_norm_ = symmetric_inf_norm(n, matrix_);
    for (const auto& e : matrix_) {
      const int i=e.i, j=e.j; const double v=e.value;
      a_[i + j*n] += v;
      if (i != j) a_[j + i*n] += v;
    }

    // Symmetric-indefinite LAPACK with numerical pivoting.  The matrix passed
    // here is the SAME rho_x/delta_d stabilized KKT used by the other common
    // backends when adaptive KKT regularization is enabled.
    const char uplo='L';
    int info=0, lwork=-1;
    double work_query=0.0;
    dsytrf_(&uplo,&n_,a_.data(),&n_,piv_.data(),&work_query,&lwork,&info);
    if (info != 0 || !std::isfinite(work_query)) {
      message = "native LAPACK KKT workspace query failed, dsytrf info=" + std::to_string(info);
      return false;
    }
    lwork=std::max(1,static_cast<int>(work_query));
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsytrf_(&uplo,&n_,a_.data(),&n_,piv_.data(),work.data(),&lwork,&info);
    if (info != 0) {
      message = "native LAPACK symmetric KKT factorization failed, dsytrf info=" + std::to_string(info);
      return false;
    }

    const double zero_threshold = st_.native_inertia_relative_zero_tolerance *
                                  matrix_inf_norm_;
    int positive=0, negative=0, zero=0;
    auto classify = [&](double value, double threshold) {
      if (!std::isfinite(value) || std::abs(value) <= threshold) ++zero;
      else if (value > 0.0) ++positive;
      else ++negative;
    };
    for (int k=0;k<n_;) {
      if (piv_[k] > 0) {
        const double d=a_[k+k*n_];
        classify(d, zero_threshold);
        ++k;
      } else {
        if (k+1>=n_ || piv_[k+1] >= 0) {
          message="native LAPACK malformed 2x2 pivot block in dsytrf factors";
          return false;
        }
        const double aa=a_[k+k*n_];
        const double bb=a_[k+1+k*n_];
        const double cc=a_[k+1+(k+1)*n_];
        const double scale=std::max({std::abs(aa),std::abs(bb),std::abs(cc)});
        if (!(scale>0.0) || !std::isfinite(scale)) {
          zero+=2;
        } else {
          const double as=aa/scale, bs=bb/scale, cs=cc/scale;
          const double det=as*cs-bs*bs;
          const double tr=as+cs;
          if (zero_threshold > 0.0) {
            // Stable small eigenvalue of the normalized 2x2 D block.
            const double gap = std::hypot(as-cs, 2.0*bs);
            const double large = 0.5*(tr + std::copysign(gap, tr));
            const double small = large != 0.0 ? det/large : 0.0;
            classify(large, zero_threshold/scale);
            classify(small, zero_threshold/scale);
          }
          else if (det < 0.0) { ++positive; ++negative; }
          else if (det > 0.0) {
            if (tr > 0.0) positive+=2;
            else if (tr < 0.0) negative+=2;
            else zero+=2;
          } else {
            ++zero;
            if (tr>0.0) ++positive;
            else if (tr<0.0) ++negative;
            else ++zero;
          }
        }
        k+=2;
      }
    }
    probe_ready_=zero==0;
    curvature_defect_=probe_ready_?std::max(0,primal_dimension-positive):0;
    const int expected_negative=n_-primal_dimension;
    if (zero!=0 || positive!=primal_dimension || negative!=expected_negative) {
      std::ostringstream os;
      os << "native LAPACK inertia mismatch: (+,-,0)=(" << positive << ","
         << negative << "," << zero << "), expected=(" << primal_dimension
         << "," << expected_negative << ",0)";
      message=os.str();
      return false;
    }
    last_backward_error_=0.0;
    last_residual_inf_=0.0;
    last_refinement_steps_=0;
    return true;
  }

  bool solve(Vec& rhs, std::string& message) const override {
    if (static_cast<int>(rhs.size()) != n_) {
      message="native LAPACK RHS dimension mismatch"; return false;
    }
    const Vec b=rhs;
    Vec x=b;
    if(!solve_raw(x,message)) return false;
    double rinf=0.0;
    double berr=backward_error(matrix_,matrix_inf_norm_,x,b,&rinf);
    int refinement=0;
    while(berr>st_.kkt_backward_error_tolerance &&
          refinement<st_.kkt_iterative_refinement_steps) {
      Vec corr=symmetric_residual(matrix_,x,b);
      if(!solve_raw(corr,message)) return false;
      axpy(1.0,corr,x);
      ++refinement;
      berr=backward_error(matrix_,matrix_inf_norm_,x,b,&rinf);
    }
    rhs=std::move(x);
    last_backward_error_=berr;
    last_residual_inf_=rinf;
    last_refinement_steps_=refinement;
    if(!std::isfinite(berr)) {
      message="native LAPACK produced non-finite backward error"; return false;
    }
    if(berr>st_.kkt_backward_error_reject) {
      std::ostringstream os;
      os << "native LAPACK backward error too large after refinement: " << berr;
      message=os.str();
      return false;
    }
    return true;
  }

  int curvature_defect() const override {return curvature_defect_;}
  bool solve_curvature_probe(Vec& rhs,std::string& message) const override {
    return probe_ready_&&curvature_defect_>0&&solve(rhs,message);
  }
  double last_backward_error() const override { return last_backward_error_; }
  double last_residual_inf() const override { return last_residual_inf_; }
  int last_refinement_steps() const override { return last_refinement_steps_; }
  const char* name() const override { return "native_dense_lapack_symmetric_indefinite_refined"; }

 private:
  bool solve_raw(Vec& rhs, std::string& message) const {
    const char uplo='L'; const int nrhs=1; int info=0;
    dsytrs_(&uplo,&n_,&nrhs,a_.data(),&n_,piv_.data(),rhs.data(),&n_,&info);
    if (info != 0) {
      message="native LAPACK symmetric KKT solve failed, dsytrs info="+std::to_string(info);
      return false;
    }
    for(double v:rhs) if(!std::isfinite(v)) {
      message="native LAPACK produced non-finite solution"; return false;
    }
    return true;
  }

  const Settings& st_;
  bool probe_ready_=false;int curvature_defect_=0;
  int n_=0;
  std::vector<double> a_;
  std::vector<int> piv_;
  EntryList matrix_;
  double matrix_inf_norm_=0.0;
  mutable double last_backward_error_=0.0;
  mutable double last_residual_inf_=0.0;
  mutable int last_refinement_steps_=0;
};

#if defined(LAZYIPDCA_HAVE_LDL) && defined(LAZYIPDCA_HAVE_METIS)
// Historical LazyIPDCA-QP style sparse LDL^T backend: SuiteSparse LDL with
// METIS nested-dissection ordering.  This is a no-numerical-pivot LDL factorization.
// Fixed regularization is applied uniformly by every backend when adaptive
// regularization is disabled.  In adaptive mode the caller inserts the current
// primal/dual shifts explicitly, preventing backend-dependent double shifts.
class SuiteSparseLdlMetisFactor final : public SymmetricFactorization {
 public:
  explicit SuiteSparseLdlMetisFactor(const Settings& st) : st_(st) {
    equilibrate_=st.kkt_equilibration;
  }

  bool factor(int n, int primal_dimension,
              const std::vector<SymmetricTriplet>& entries,
              std::string& message) override {
    probe_ready_=false;curvature_defect_=0;
    failed_pivot_=-1;
    last_numeric_scatter_reused_=false;

    // Once the symbolic envelope and the caller's raw triplet layout are
    // stable, assemble values through a positional scatter.  The first call
    // builds raw_entry_to_envelope_; subsequent calls verify the coordinates
    // exactly in O(nnz) and perform no sorting, pattern construction, binary
    // search, or associative lookup.  Accumulation follows the caller order,
    // followed by fixed diagonal regularization, exactly matching the stable
    // counting-sort path in aggregate_upper().
    const bool fixed_regularization=!st_.kkt_adaptive_regularization;
    bool scatter_layout_matches=numeric_scatter_valid_ && n==scatter_n_ &&
        fixed_regularization==scatter_fixed_regularization_ &&
        entries.size()==raw_entry_keys_.size();
    if(scatter_layout_matches) {
      for(std::size_t k=0;k<entries.size();++k) {
        int i=entries[k].i,j=entries[k].j;
        if(i<0 || i>=n || j<0 || j>=n) {
          scatter_layout_matches=false;
          break;
        }
        if(i>j) std::swap(i,j);
        if(upper_key(i,j)!=raw_entry_keys_[k]) {
          scatter_layout_matches=false;
          break;
        }
      }
    }
    if(numeric_scatter_valid_ && !scatter_layout_matches) {
      ++numeric_scatter_invalidations_;
      clear_numeric_scatter();
    }

    bool direct_numeric_hit=false;
    const bool direct_entry_count_valid=st_.kkt_adaptive_regularization ||
        (n>=0 && entries.size()<=std::numeric_limits<std::size_t>::max()-
                                      static_cast<std::size_t>(n));
    const std::size_t direct_entry_count=entries.size()+
        ((!st_.kkt_adaptive_regularization && direct_entry_count_valid)
             ? static_cast<std::size_t>(n) : 0u);
    const bool direct_numeric_candidate=scatter_layout_matches &&
        symbolic_valid_ && n==n_ &&
        (!(st_.kkt_primal_first_ordering||st_.kkt_interleaved_equality_ordering) || symbolic_primal_dimension_==primal_dimension) &&
        direct_entry_count_valid &&
        stable_counting_upper_order(n,direct_entry_count) &&
        raw_entry_to_envelope_.size()==entries.size() &&
        (!fixed_regularization ||
         regularization_to_envelope_.size()==static_cast<std::size_t>(n)) &&
        envelope_values_.size()==input_pattern_.size() &&
        input_to_ax_.size()==input_pattern_.size() &&
        Ax_.size()==input_pattern_.size();
    if(direct_numeric_candidate) {
      std::fill(envelope_values_.begin(),envelope_values_.end(),0.0);
      bool valid=true;
      auto accumulate_direct = [&](std::size_t slot, double value) {
        if(!valid) return;
        if(slot>=envelope_values_.size() || !std::isfinite(value)) {
          valid=false;
          return;
        }
        double& destination=envelope_values_[slot];
        destination+=value;
        if(!std::isfinite(destination)) valid=false;
      };
      for(std::size_t k=0;k<entries.size();++k)
        accumulate_direct(raw_entry_to_envelope_[k],entries[k].value);
      if(valid && fixed_regularization) {
        for(int i=0;i<n;++i) {
          const double regularization=i<primal_dimension
              ? st_.kkt_primal_regularization
              : -st_.kkt_dual_regularization;
          // aggregate_upper appends zero diagonal entries but removes them at
          // the common 1e-30 threshold.  Avoid treating such an absent zero
          // key as structural growth on every direct attempt.
          if(regularization==0.0) continue;
          accumulate_direct(
              regularization_to_envelope_[static_cast<std::size_t>(i)],
              regularization);
        }
      }
      if(valid) {
        std::fill(Ax_.begin(),Ax_.end(),0.0);
        matrix_.clear();
        matrix_.reserve(input_pattern_.size());
        for(std::size_t slot=0;slot<input_pattern_.size();++slot) {
          const double value=envelope_values_[slot];
          if(st_.kkt_preserve_structural_zeros?value==0.0:std::abs(value)<1e-30) continue;
          Ax_[input_to_ax_[slot]]=value;
          matrix_.push_back({upper_key_row(input_pattern_[slot]),
                             upper_key_col(input_pattern_[slot]),value});
        }
        matrix_inf_norm_=symmetric_inf_norm(n,matrix_);
        last_symbolic_reused_=true;
        ++symbolic_reuses_;
        last_numeric_scatter_reused_=true;
        ++numeric_scatter_reuses_;
        direct_numeric_hit=true;
      } else if(numeric_scatter_valid_) {
        // The only structural failure possible here is a cached sentinel for
        // a fixed diagonal that has changed from exact zero to nonzero.  Fail
        // closed through the historical aggregation path and rebuild the
        // scatter against the resulting envelope.  Non-finite input values
        // also fall back, but retain the valid layout for a later finite call.
        bool missing_required_slot=false;
        if(fixed_regularization) {
          for(int i=0;i<n;++i) {
            const double regularization=i<primal_dimension
                ? st_.kkt_primal_regularization
                : -st_.kkt_dual_regularization;
            if(regularization!=0.0 &&
               regularization_to_envelope_[static_cast<std::size_t>(i)]>=
                   input_pattern_.size()) {
              missing_required_slot=true;
              break;
            }
          }
        }
        if(missing_required_slot) {
          ++numeric_scatter_invalidations_;
          clear_numeric_scatter();
          scatter_layout_matches=false;
        }
      }
    }

    if(!direct_numeric_hit) {
      EntryList M = st_.kkt_adaptive_regularization
          ? aggregate_upper(n, entries,-1,0.0,0.0,st_.kkt_preserve_structural_zeros)
          : aggregate_upper(n, entries, primal_dimension,
                            st_.kkt_primal_regularization,
                            st_.kkt_dual_regularization,st_.kkt_preserve_structural_zeros);
      matrix_=M;
      matrix_inf_norm_=symmetric_inf_norm(n,M);
      const auto current_pattern=entry_pattern(M);
      const bool same_dimension=symbolic_valid_ && n==n_ &&
          (!(st_.kkt_primal_first_ordering||st_.kkt_interleaved_equality_ordering) || symbolic_primal_dimension_==primal_dimension);
      const bool reusable=same_dimension &&
          pattern_contains(input_pattern_,current_pattern);
      last_symbolic_reused_=reusable;

      if (!reusable) {
        if(symbolic_valid_) ++symbolic_invalidations_;
        n_=n;symbolic_primal_dimension_=primal_dimension;
        clear_numeric_scatter();
        // Keep a monotone structural envelope while the KKT dimension is fixed.
        // Entries that temporarily become zero remain explicit zero slots in Ax,
        // so a disappearing Jacobian/Hessian coefficient does not invalidate
        // METIS or the LDL elimination tree.  Only a genuinely new structural
        // entry grows the envelope and triggers a new symbolic analysis.
        input_pattern_=same_dimension
            ? pattern_union(input_pattern_,current_pattern)
            : current_pattern;

        perm_.resize(n); iperm_.resize(n);
        for(int i=0;i<n;++i) perm_[i]=iperm_[i]=i;

        if(st_.linear_metis && n>1) {
#ifdef METIS_NOPTIONS
          using MetisIndex=idx_t;
#else
          using MetisIndex=idxtype;
#endif
          std::vector<std::vector<MetisIndex>> adj(n);
          for(const std::uint64_t key:input_pattern_) {
            const int i=upper_key_row(key), j=upper_key_col(key);
            if(i==j) continue;
            adj[i].push_back(static_cast<MetisIndex>(j));
            adj[j].push_back(static_cast<MetisIndex>(i));
          }
          std::vector<MetisIndex> xadj(n+1,0), adjncy;
          for(int i=0;i<n;++i) {
            auto& ai=adj[i];
            std::sort(ai.begin(),ai.end());
            ai.erase(std::unique(ai.begin(),ai.end()),ai.end());
            xadj[i+1]=xadj[i]+static_cast<MetisIndex>(ai.size());
            adjncy.insert(adjncy.end(),ai.begin(),ai.end());
          }
#ifdef METIS_NOPTIONS
          std::vector<MetisIndex> p(n),ip(n),opts(METIS_NOPTIONS);
          METIS_SetDefaultOptions(opts.data());
          opts[METIS_OPTION_NUMBERING]=0;
          opts[METIS_OPTION_SEED]=static_cast<MetisIndex>(st_.random_seed);
          MetisIndex nv=static_cast<MetisIndex>(n);
          const int rc=METIS_NodeND(&nv,xadj.data(),
              adjncy.empty()?nullptr:adjncy.data(),nullptr,opts.data(),p.data(),ip.data());
          if(rc!=METIS_OK) {
            message="METIS_NodeND failed for SuiteSparse LDL backend, code="+
                std::to_string(rc);
            symbolic_valid_=false;
            return false;
          }
#else
          std::vector<MetisIndex> p(n),ip(n);
          int nv=n,numflag=0,opts[8]={0,0,0,0,0,0,0,0};
          METIS_NodeND(&nv,xadj.data(),adjncy.empty()?nullptr:adjncy.data(),
                       &numflag,opts,p.data(),ip.data());
#endif
          for(int i=0;i<n;++i) {
            perm_[i]=static_cast<int>(p[i]);
            iperm_[i]=static_cast<int>(ip[i]);
          }
        }

        const auto hubs=dense_primal_border_order(n,primal_dimension,input_pattern_,
            perm_,iperm_,!dense_border_disabled_ && st_.kkt_interleaved_equality_ordering &&
            !st_.kkt_primal_first_ordering);
        dense_border_active_=!hubs.empty();

        if(st_.kkt_interleaved_equality_ordering) {
          // Keep METIS locality: a zero-diagonal dual must follow
          // all incident primals. This is a structural order, not a claim of
          // numerical pivot safety; inertia/residual checks remain mandatory.
          std::vector<std::vector<int>> neighbours(n),pending(n);
          for(const auto key:input_pattern_) {
            const int i=upper_key_row(key),j=upper_key_col(key);
            if(i<primal_dimension&&j>=primal_dimension)neighbours[j].push_back(i);
          }
          local_primal_dependencies(neighbours,hubs);
          std::vector<int> ordered;ordered.reserve(n);
          std::vector<unsigned char> emitted(n,0);
          for(int v:perm_) {
            if(v<primal_dimension) {
              ordered.push_back(v);emitted[v]=1;
              for(int d:pending[v]){ordered.push_back(d);emitted[d]=1;}
            } else {
              bool ready=true;int last_primal=-1;
              for(int p:neighbours[v]) {
                ready=ready&&emitted[p];
                if(last_primal<0||iperm_[p]>iperm_[last_primal])last_primal=p;
              }
              if(ready||last_primal<0){ordered.push_back(v);emitted[v]=1;}
              else pending[last_primal].push_back(v);
            }
          }
          perm_.swap(ordered);
          for(int i=0;i<n;++i)iperm_[perm_[i]]=i;
        }
        if(st_.kkt_primal_first_ordering) {
          std::stable_partition(perm_.begin(),perm_.end(),
              [&](int j){return j<primal_dimension;});
          for(int i=0;i<n;++i)iperm_[perm_[i]]=i;
        }

        struct IndexedPermutedEntry {
          int i=0,j=0;
          double value=0.0;
          std::size_t input_index=0;
        };
        std::vector<IndexedPermutedEntry> P;
        P.reserve(input_pattern_.size());
        std::size_t current=0;
        for(std::size_t k=0;k<input_pattern_.size();++k) {
          const std::uint64_t key=input_pattern_[k];
          const int old_i=upper_key_row(key), old_j=upper_key_col(key);
          double value=0.0;
          if(current<current_pattern.size() && current_pattern[current]==key) {
            value=M[current].value;
            ++current;
          }
          int i=iperm_[old_i],j=iperm_[old_j];
          if(i>j) std::swap(i,j);
          P.push_back({i,j,value,k});
        }
        std::sort(P.begin(),P.end(),[](const IndexedPermutedEntry& a,
                                       const IndexedPermutedEntry& b) {
          return upper_key(a.i,a.j)<upper_key(b.i,b.j);
        });

        Ap_.assign(n+1,0);
        for(const auto& e:P) ++Ap_[e.j+1];
        for(int j=0;j<n;++j) Ap_[j+1]+=Ap_[j];
        Ai_.resize(P.size());
        Ax_.resize(P.size());
        input_to_ax_.resize(P.size());
        for(std::size_t k=0;k<P.size();++k) {
          Ai_[k]=P[k].i;
          Ax_[k]=P[k].value;
          input_to_ax_[P[k].input_index]=k;
        }

        envelope_values_.assign(input_pattern_.size(),0.0);

        Lp_.resize(n+1); Parent_.resize(n); Lnz_.resize(n); Flag_.resize(n);
        ldl_symbolic(n,Ap_.data(),Ai_.data(),Lp_.data(),Parent_.data(),
                     Lnz_.data(),Flag_.data(),nullptr,nullptr);
        if(Lp_[n]<0) {
          message="SuiteSparse LDL symbolic analysis produced invalid L size";
          symbolic_valid_=false;
          return false;
        }
        Li_.resize(static_cast<std::size_t>(Lp_[n]));
        Lx_.resize(static_cast<std::size_t>(Lp_[n]));
        D_.resize(n); Y_.resize(n); Pattern_.resize(n);
        symbolic_valid_=true;
        ++symbolic_analyses_;
      } else {
        if(input_to_ax_.size()!=input_pattern_.size() || Ax_.size()!=input_pattern_.size()) {
          message="SuiteSparse LDL cached numeric scatter map is inconsistent";
          symbolic_valid_=false;
          return false;
        }
        std::fill(Ax_.begin(),Ax_.end(),0.0);
        std::size_t envelope=0;
        for(std::size_t current=0;current<current_pattern.size();++current) {
          while(envelope<input_pattern_.size() &&
                input_pattern_[envelope]<current_pattern[current]) ++envelope;
          if(envelope>=input_pattern_.size() ||
             input_pattern_[envelope]!=current_pattern[current]) {
            message="SuiteSparse LDL cached envelope lost a current pattern entry";
            symbolic_valid_=false;
            return false;
          }
          Ax_[input_to_ax_[envelope]]=M[current].value;
        }
        ++symbolic_reuses_;
      }

      // Build the raw-layout scatter only after the monotone envelope and CSC
      // permutation map are known to be valid.  lower_bound is paid once per
      // layout; the hot numerical path above subsequently performs only
      // indexed additions.  If an input coordinate was removed by the 1e-30
      // aggregation threshold, leave the cache disabled so a later nonzero
      // value necessarily returns through envelope growth and symbolic
      // invalidation rather than being silently dropped.
      if(!numeric_scatter_valid_) {
        std::vector<std::uint64_t> candidate_keys;
        std::vector<std::size_t> candidate_scatter;
        std::vector<std::size_t> candidate_regularization;
        candidate_keys.reserve(entries.size());
        candidate_scatter.reserve(entries.size());
        bool scatter_complete=stable_counting_upper_order(n,direct_entry_count);
        for(const auto& entry:entries) {
          int i=entry.i,j=entry.j;
          if(i<0 || i>=n || j<0 || j>=n) {
            scatter_complete=false;
            break;
          }
          if(i>j) std::swap(i,j);
          const std::uint64_t key=upper_key(i,j);
          candidate_keys.push_back(key);
          const auto found=std::lower_bound(
              input_pattern_.begin(),input_pattern_.end(),key);
          if(found==input_pattern_.end() || *found!=key) {
            scatter_complete=false;
            break;
          }
          candidate_scatter.push_back(
              static_cast<std::size_t>(found-input_pattern_.begin()));
        }
        if(scatter_complete && fixed_regularization) {
          candidate_regularization.assign(static_cast<std::size_t>(n),
                                           input_pattern_.size());
          for(int i=0;i<n;++i) {
            const double regularization=i<primal_dimension
                ? st_.kkt_primal_regularization
                : -st_.kkt_dual_regularization;
            if(regularization==0.0) continue;
            const std::uint64_t key=upper_key(i,i);
            const auto found=std::lower_bound(
                input_pattern_.begin(),input_pattern_.end(),key);
            if(found==input_pattern_.end() || *found!=key) {
              scatter_complete=false;
              break;
            }
            candidate_regularization[static_cast<std::size_t>(i)]=
                static_cast<std::size_t>(found-input_pattern_.begin());
          }
        }
        if(scatter_complete && candidate_keys.size()==entries.size() &&
           candidate_scatter.size()==entries.size()) {
          raw_entry_keys_=std::move(candidate_keys);
          raw_entry_to_envelope_=std::move(candidate_scatter);
          regularization_to_envelope_=std::move(candidate_regularization);
          scatter_n_=n;
          scatter_fixed_regularization_=fixed_regularization;
          numeric_scatter_valid_=true;
          ++numeric_scatter_builds_;
        }
      }
    }

    // Equilibrate numeric values only: the graph, positional scatter, and
    // original matrix used for backward-error checks remain unchanged.
    // Positive diagonal congruence preserves the inertia certificate.
    if(equilibrate_&&!equilibrate_numeric(message))return false;

    failed_pivot_=-1;
    reported_positive_=reported_negative_=reported_zero_=-1;
    std::fill(Flag_.begin(),Flag_.end(),0);
    const int d=ldl_numeric(n,Ap_.data(),Ai_.data(),Ax_.data(),
                            Lp_.data(),Parent_.data(),Lnz_.data(),
                            Li_.data(),Lx_.data(),D_.data(),Y_.data(),
                            Pattern_.data(),Flag_.data(),nullptr,nullptr);
    if(d!=n) {
      if(dense_border_active_ && !dense_border_disabled_) {
        dense_border_disabled_=true;dense_border_active_=false;symbolic_valid_=false;
        return factor(n,primal_dimension,entries,message);
      }
      failed_pivot_=perm_[d];
      message="SuiteSparse LDL numeric factorization failed at pivot "+
          std::to_string(d)+" of "+std::to_string(n)+" original="+std::to_string(perm_[d]);
      return false;
    }

    int positive=0,negative=0,zero=0;
    for(double v:D_) {
      if(!std::isfinite(v)||v==0.0) ++zero;
      else if(v>0.0) ++positive;
      else ++negative;
    }
    reported_positive_=positive;reported_negative_=negative;reported_zero_=zero;
    probe_ready_=zero==0;
    curvature_defect_=probe_ready_?std::max(0,primal_dimension-positive):0;
    const int expected_negative=n-primal_dimension;
    if(zero!=0 || positive!=primal_dimension || negative!=expected_negative) {
      std::ostringstream os;
      os << "SuiteSparse LDL inertia mismatch: (+,-,0)=(" << positive << ","
         << negative << "," << zero << "), expected=(" << primal_dimension
         << "," << expected_negative << ",0)";
      message=os.str();
      return false;
    }
    return true;
  }

  bool solve(Vec& rhs, std::string& message) const override {
    if (static_cast<int>(rhs.size()) != n_) {
      message="SuiteSparse LDL RHS dimension mismatch"; return false;
    }
    scratch_b_=rhs;
    scratch_x_=rhs;
    if(!solve_raw(scratch_x_,scratch_perm_,message)) return false;
    double rinf=0.0;
    const double binf=inf_norm(scratch_b_);
    double berr=backward_error_into_known_binf(
        matrix_,matrix_inf_norm_,binf,scratch_x_,scratch_b_,scratch_residual_,&rinf);
    last_initial_backward_error_=berr;
    int refinement=0;
    while(berr>st_.kkt_backward_error_tolerance &&
          refinement<st_.kkt_iterative_refinement_steps){
      // scratch_residual_ already holds b-Ax from backward_error_into().
      if(!solve_raw(scratch_residual_,scratch_perm_,message)) return false;
      axpy(1.0,scratch_residual_,scratch_x_);
      ++refinement;
      berr=backward_error_into_known_binf(
          matrix_,matrix_inf_norm_,binf,scratch_x_,scratch_b_,scratch_residual_,&rinf);
    }
    rhs=scratch_x_;
    last_backward_error_=berr;
    last_residual_inf_=rinf;
    last_refinement_steps_=refinement;
    if(!std::isfinite(berr)){
      message="SuiteSparse LDL produced non-finite backward error"; return false;
    }
    if(berr>st_.kkt_backward_error_reject){
      std::ostringstream os;
      os << "SuiteSparse LDL backward error too large after refinement: " << berr;
      message=os.str();
      return false;
    }
    return true;
  }

  int curvature_defect() const override {return curvature_defect_;}
  bool solve_curvature_probe(Vec& rhs,std::string& message) const override {
    return probe_ready_&&curvature_defect_>0&&solve(rhs,message);
  }
  int failed_pivot_coordinate() const override { return failed_pivot_; }
  double last_backward_error() const override { return last_backward_error_; }
  double last_initial_backward_error() const override { return last_initial_backward_error_; }
  double last_residual_inf() const override { return last_residual_inf_; }
  int last_refinement_steps() const override { return last_refinement_steps_; }
  int symbolic_analyses() const override { return symbolic_analyses_; }
  int symbolic_reuses() const override { return symbolic_reuses_; }
  int symbolic_invalidations() const override { return symbolic_invalidations_; }
  bool last_symbolic_reused() const override { return last_symbolic_reused_; }
  bool has_symbolic_cache() const override { return symbolic_valid_; }
  int numeric_scatter_builds() const override { return numeric_scatter_builds_; }
  int numeric_scatter_reuses() const override { return numeric_scatter_reuses_; }
  int numeric_scatter_invalidations() const override {
    return numeric_scatter_invalidations_;
  }
  bool last_numeric_scatter_reused() const override {
    return last_numeric_scatter_reused_;
  }
  int last_positive_pivots() const override {return reported_positive_;}
  int last_negative_pivots() const override {return reported_negative_;}
  int last_zero_pivots() const override {return reported_zero_;}
  const char* name() const override {
    return "suitesparse_ldl_metis_persistent_symbolic";
  }

 private:
  int reported_positive_=-1,reported_negative_=-1,reported_zero_=-1;
  bool equilibrate_numeric(std::string& message) {
    equilibration_scale_.assign(static_cast<std::size_t>(n_),1.0);
    equilibration_norm_.resize(static_cast<std::size_t>(n_));
    equilibration_step_.resize(static_cast<std::size_t>(n_));
    constexpr double minimum_scale=1e-100,maximum_scale=1e100;
    // Three bounded symmetric Ruiz max-norm passes, in permuted coordinates.
    // Every refactor starts from freshly assembled unscaled Ax_; scaling is
    // never compounded with a factorization from a previous numeric update.
    for(int pass=0;pass<3;++pass) {
      std::fill(equilibration_norm_.begin(),equilibration_norm_.end(),0.0);
      for(int j=0;j<n_;++j)for(int k=Ap_[j];k<Ap_[j+1];++k) {
        const double a=std::abs(Ax_[k]);
        if(!std::isfinite(a)) {
          message="SuiteSparse LDL equilibration encountered nonfinite coefficient";return false;
        }
        const int i=Ai_[k];
        equilibration_norm_[i]=std::max(equilibration_norm_[i],a);
        equilibration_norm_[j]=std::max(equilibration_norm_[j],a);
      }
      for(int i=0;i<n_;++i) {
        const double norm=equilibration_norm_[i];
        // A zero row is left unchanged; the ordinary LDL singularity check
        // still rejects it. No diagonal shift is introduced by equilibration.
        const double proposed=norm>0?1.0/std::sqrt(norm):1.0;
        const double step=std::clamp(proposed,minimum_scale/equilibration_scale_[i],
                                    maximum_scale/equilibration_scale_[i]);
        const double cumulative=equilibration_scale_[i]*step;
        if(!(step>0)||!std::isfinite(step)||!(cumulative>0)||!std::isfinite(cumulative)) {
          message="SuiteSparse LDL equilibration produced invalid diagonal scale";return false;
        }
        equilibration_step_[i]=step;
        equilibration_scale_[i]=cumulative;
      }
      for(int j=0;j<n_;++j)for(int k=Ap_[j];k<Ap_[j+1];++k) {
        // A wide intermediate avoids overflow of either product before the
        // second diagonal factor brings it back into double range.
        const long double product=static_cast<long double>(Ax_[k])*
            equilibration_step_[Ai_[k]]*equilibration_step_[j];
        const double scaled=static_cast<double>(product);
        if(!std::isfinite(scaled)||(Ax_[k]!=0.0&&scaled==0.0)) {
          message="SuiteSparse LDL equilibration would overflow or erase a coefficient";return false;
        }
        Ax_[k]=scaled;
      }
    }
    return true;
  }
  void clear_numeric_scatter() {
    numeric_scatter_valid_=false;
    scatter_n_=0;
    scatter_fixed_regularization_=false;
    raw_entry_keys_.clear();
    raw_entry_to_envelope_.clear();
    regularization_to_envelope_.clear();
  }

  bool solve_raw(Vec& rhs, Vec& perm_work, std::string& message) const {
    if(static_cast<int>(perm_work.size())!=n_)
      perm_work.resize(static_cast<std::size_t>(n_));
    if(equilibrate_) {
      for(int i=0;i<n_;++i) {
        const double value=equilibration_scale_[i]*rhs[perm_[i]];
        if(!std::isfinite(value)||(rhs[perm_[i]]!=0.0&&value==0.0)) {
          message="SuiteSparse LDL equilibration overflowed or erased a right hand side entry";return false;
        }
        perm_work[static_cast<std::size_t>(i)]=value;
      }
    } else for (int i=0;i<n_;++i) perm_work[static_cast<std::size_t>(i)]=rhs[perm_[i]];
    auto* self=const_cast<SuiteSparseLdlMetisFactor*>(this);
    ldl_lsolve(n_,perm_work.data(),self->Lp_.data(),self->Li_.data(),self->Lx_.data());
    ldl_dsolve(n_,perm_work.data(),self->D_.data());
    ldl_ltsolve(n_,perm_work.data(),self->Lp_.data(),self->Li_.data(),self->Lx_.data());
    if(equilibrate_) {
      for(int i=0;i<n_;++i)rhs[perm_[i]]=equilibration_scale_[i]*perm_work[static_cast<std::size_t>(i)];
    } else for (int i=0;i<n_;++i) rhs[perm_[i]]=perm_work[static_cast<std::size_t>(i)];
    for (double v:rhs) if(!std::isfinite(v)) {
      message="SuiteSparse LDL produced non-finite solution"; return false;
    }
    return true;
  }
  const Settings& st_;
  bool equilibrate_=false;
  Vec equilibration_scale_,equilibration_norm_,equilibration_step_;
  bool probe_ready_=false;int curvature_defect_=0;
  int symbolic_primal_dimension_=-1,failed_pivot_=-1;
  int n_=0, symbolic_analyses_=0, symbolic_reuses_=0, symbolic_invalidations_=0;
  bool dense_border_active_=false,dense_border_disabled_=false;
  bool symbolic_valid_=false;
  bool last_symbolic_reused_=false;
  int numeric_scatter_builds_=0;
  int numeric_scatter_reuses_=0;
  int numeric_scatter_invalidations_=0;
  int scatter_n_=0;
  bool numeric_scatter_valid_=false;
  bool scatter_fixed_regularization_=false;
  bool last_numeric_scatter_reused_=false;
  std::vector<std::uint64_t> input_pattern_;
  std::vector<std::uint64_t> raw_entry_keys_;
  std::vector<std::size_t> raw_entry_to_envelope_;
  std::vector<std::size_t> regularization_to_envelope_;
  std::vector<double> envelope_values_;
  std::vector<std::size_t> input_to_ax_;
  std::vector<int> perm_, iperm_;
  std::vector<int> Ap_, Ai_, Lp_, Parent_, Lnz_, Flag_, Li_, Pattern_;
  std::vector<double> Ax_, Lx_, D_, Y_;
  EntryList matrix_;
  double matrix_inf_norm_=0.0;
  mutable Vec scratch_b_, scratch_x_, scratch_residual_, scratch_perm_;
  mutable double last_backward_error_=0.0;
  mutable double last_initial_backward_error_=0.0;
  mutable double last_residual_inf_=0.0;
  mutable int last_refinement_steps_=0;
};
#endif

#if defined(LAZYIPDCA_HAVE_QDLDL) && defined(LAZYIPDCA_HAVE_METIS)
class QdldlMetisFactor final : public SymmetricFactorization {
 public:
  explicit QdldlMetisFactor(const Settings& st) : st_(st) {}

  // In fixed-mode Cholesk, try its already authorized numerical repair
  // ladder before discarding a sparse border ordering. Exact callers retain
  // the original immediate, unshifted ordering fallback.
  void allow_dense_border_fallback(bool allowed) { allow_dense_border_fallback_=allowed; }
  bool has_dense_border_fallback() const {
    return dense_border_active_ && !dense_border_disabled_;
  }
  void use_original_border_order() {
    dense_border_disabled_=true;dense_border_active_=false;symbolic_valid_=false;
  }

  bool factor(int n, int primal_dimension,
              const std::vector<SymmetricTriplet>& entries,
              std::string& message) override {
    positive_pivots_=negative_pivots_=zero_pivots_=-1;
    minimum_abs_pivot_=maximum_abs_pivot_=0.0;
    minimum_abs_pivot_coordinate_=-1;
    probe_ready_=false;
    curvature_defect_=0;
    // QDLDL's symbolic routine rejects an empty CSC column. KKT dual
    // diagonals are commonly exact structural zeros, so retain one explicit
    // diagonal slot for every coordinate in both caller-adaptive and fixed
    // modes. This changes only the symbolic envelope, never a matrix value.
    EntryList M = st_.kkt_adaptive_regularization
        ? aggregate_upper(n,entries,primal_dimension,0.0,0.0,true)
        : aggregate_upper(n,entries,primal_dimension,
                          st_.kkt_primal_regularization,
                          st_.kkt_dual_regularization,true);
    const auto current_pattern=entry_pattern(M);
    const bool same_dimension=symbolic_valid_ && n==n_ &&
        (!(st_.kkt_primal_first_ordering||st_.kkt_interleaved_equality_ordering) ||
         symbolic_primal_dimension_==primal_dimension);
    const bool reusable=same_dimension &&
        pattern_contains(input_pattern_,current_pattern);
    last_symbolic_reused_=reusable;

    if(!reusable) {
      if(symbolic_valid_) ++symbolic_invalidations_;
      n_=n;
      symbolic_primal_dimension_=primal_dimension;
      input_pattern_=same_dimension
          ? pattern_union(input_pattern_,current_pattern)
          : current_pattern;

      perm_.resize(n); iperm_.resize(n);
      for(int i=0;i<n;++i) perm_[i]=iperm_[i]=i;
      if(st_.linear_metis && n>1) {
#ifdef METIS_NOPTIONS
        using MetisIndex=idx_t;
#else
        using MetisIndex=idxtype;
#endif
        std::vector<std::vector<MetisIndex>> adj(n);
        for(const std::uint64_t key:input_pattern_) {
          const int i=upper_key_row(key),j=upper_key_col(key);
          if(i==j) continue;
          adj[i].push_back(static_cast<MetisIndex>(j));
          adj[j].push_back(static_cast<MetisIndex>(i));
        }
        std::vector<MetisIndex> xadj(n+1,0),adjncy;
        for(int i=0;i<n;++i) {
          auto& ai=adj[i];
          std::sort(ai.begin(),ai.end());
          ai.erase(std::unique(ai.begin(),ai.end()),ai.end());
          xadj[i+1]=xadj[i]+static_cast<MetisIndex>(ai.size());
          adjncy.insert(adjncy.end(),ai.begin(),ai.end());
        }
#ifdef METIS_NOPTIONS
        std::vector<MetisIndex> p(n),ip(n),opts(METIS_NOPTIONS);
        METIS_SetDefaultOptions(opts.data());
        opts[METIS_OPTION_NUMBERING]=0;
        opts[METIS_OPTION_SEED]=static_cast<MetisIndex>(st_.random_seed);
        MetisIndex nv=static_cast<MetisIndex>(n);
        const int rc=METIS_NodeND(&nv,xadj.data(),
            adjncy.empty()?nullptr:adjncy.data(),nullptr,opts.data(),p.data(),ip.data());
        if(rc!=METIS_OK) {
          message="METIS_NodeND failed for QDLDL backend, code="+
              std::to_string(rc);
          symbolic_valid_=false;
          return false;
        }
#else
        std::vector<MetisIndex> p(n),ip(n);
        int nv=n,numflag=0,opts[8]={0,0,0,0,0,0,0,0};
        METIS_NodeND(&nv,xadj.data(),adjncy.empty()?nullptr:adjncy.data(),
                     &numflag,opts,p.data(),ip.data());
#endif
        for(int i=0;i<n;++i) {
          perm_[i]=static_cast<int>(p[i]);
          iperm_[i]=static_cast<int>(ip[i]);
        }
      }

      // QDLDL has no numerical pivoting. Preserve METIS locality while
      // enforcing the KKT dependency order requested by Lazy: a zero-diagonal
      // equality coordinate may only be emitted after all incident primal
      // coordinates. This is the re-entrant counterpart of chold's primal /
      // dual tiers and is essential for exact (unshifted) saddle matrices.
      const auto hubs=dense_primal_border_order(n,primal_dimension,input_pattern_,
          perm_,iperm_,!dense_border_disabled_ && st_.kkt_interleaved_equality_ordering &&
          !st_.kkt_primal_first_ordering);
      dense_border_active_=!hubs.empty();

      if(st_.kkt_interleaved_equality_ordering) {
        std::vector<std::vector<int>> neighbours(n),pending(n);
        for(const std::uint64_t key:input_pattern_) {
          const int i=upper_key_row(key),j=upper_key_col(key);
          if(i<primal_dimension&&j>=primal_dimension)
            neighbours[static_cast<std::size_t>(j)].push_back(i);
        }
        local_primal_dependencies(neighbours,hubs);
        std::vector<int> ordered;ordered.reserve(static_cast<std::size_t>(n));
        std::vector<unsigned char> emitted(static_cast<std::size_t>(n),0);
        for(int v:perm_) {
          if(v<primal_dimension) {
            ordered.push_back(v);emitted[static_cast<std::size_t>(v)]=1;
            for(int d:pending[static_cast<std::size_t>(v)]) {
              ordered.push_back(d);emitted[static_cast<std::size_t>(d)]=1;
            }
          } else {
            bool ready=true;int last_primal=-1;
            for(int p:neighbours[static_cast<std::size_t>(v)]) {
              ready=ready&&emitted[static_cast<std::size_t>(p)];
              if(last_primal<0||iperm_[static_cast<std::size_t>(p)]>
                                  iperm_[static_cast<std::size_t>(last_primal)])
                last_primal=p;
            }
            if(ready||last_primal<0) {
              ordered.push_back(v);emitted[static_cast<std::size_t>(v)]=1;
            } else pending[static_cast<std::size_t>(last_primal)].push_back(v);
          }
        }
        if(ordered.size()!=static_cast<std::size_t>(n)) {
          message="QDLDL interleaved ordering lost a KKT coordinate";
          symbolic_valid_=false;
          return false;
        }
        perm_.swap(ordered);
        for(int i=0;i<n;++i)iperm_[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])]=i;
      }
      if(st_.kkt_primal_first_ordering) {
        std::stable_partition(perm_.begin(),perm_.end(),
            [&](int j){return j<primal_dimension;});
        for(int i=0;i<n;++i)iperm_[static_cast<std::size_t>(perm_[static_cast<std::size_t>(i)])]=i;
      }

      struct IndexedPermutedEntry {
        int i=0,j=0;
        double value=0.0;
        std::size_t input_index=0;
      };
      std::vector<IndexedPermutedEntry> P;
      P.reserve(input_pattern_.size());
      std::size_t current=0;
      for(std::size_t k=0;k<input_pattern_.size();++k) {
        const std::uint64_t key=input_pattern_[k];
        const int old_i=upper_key_row(key),old_j=upper_key_col(key);
        double value=0.0;
        if(current<current_pattern.size() && current_pattern[current]==key) {
          value=M[current].value;
          ++current;
        }
        int i=iperm_[old_i],j=iperm_[old_j];
        if(i>j) std::swap(i,j);
        P.push_back({i,j,value,k});
      }
      std::sort(P.begin(),P.end(),[](const IndexedPermutedEntry& a,
                                     const IndexedPermutedEntry& b) {
        return upper_key(a.i,a.j)<upper_key(b.i,b.j);
      });

      Ap_.assign(n+1,0);
      for(const auto& e:P) ++Ap_[e.j+1];
      for(int j=0;j<n;++j) Ap_[j+1]+=Ap_[j];
      Ai_.resize(P.size()); Ax_.resize(P.size());
      input_to_ax_.resize(P.size());
      for(std::size_t k=0;k<P.size();++k) {
        Ai_[k]=static_cast<QDLDL_int>(P[k].i);
        Ax_[k]=static_cast<QDLDL_float>(P[k].value);
        input_to_ax_[P[k].input_index]=k;
      }

      etree_.resize(n); Lnz_.resize(n); iwork_.resize(static_cast<std::size_t>(3)*n);
      bwork_.resize(n); fwork_.resize(n); Lp_.resize(n+1); D_.resize(n); Dinv_.resize(n);
      const QDLDL_int lnz=QDLDL_etree(static_cast<QDLDL_int>(n),Ap_.data(),Ai_.data(),
          iwork_.data(),Lnz_.data(),etree_.data());
      if(lnz<0) {
        message="QDLDL_etree failed, code="+
            std::to_string(static_cast<long long>(lnz));
        symbolic_valid_=false;
        return false;
      }
      if(st_.live_display_level>=4)
        std::cout<<"[QDLDL-SYMBOLIC] n="<<n<<" primal="<<primal_dimension
                 <<" input_nnz="<<P.size()<<" factor_nnz="<<lnz
                 <<" dense_border="<<dense_border_active_<<std::endl;
      Li_.resize(static_cast<std::size_t>(lnz));
      Lx_.resize(static_cast<std::size_t>(lnz));
      symbolic_valid_=true;
      ++symbolic_analyses_;
    } else {
      if(input_to_ax_.size()!=input_pattern_.size() ||
         Ax_.size()!=input_pattern_.size()) {
        message="QDLDL cached numeric scatter map is inconsistent";
        symbolic_valid_=false;
        return false;
      }
      std::fill(Ax_.begin(),Ax_.end(),static_cast<QDLDL_float>(0));
      std::size_t envelope=0;
      for(std::size_t current=0;current<current_pattern.size();++current) {
        while(envelope<input_pattern_.size() &&
              input_pattern_[envelope]<current_pattern[current]) ++envelope;
        if(envelope>=input_pattern_.size() ||
           input_pattern_[envelope]!=current_pattern[current]) {
          message="QDLDL cached envelope lost a current pattern entry";
          symbolic_valid_=false;
          return false;
        }
        Ax_[input_to_ax_[envelope]]=static_cast<QDLDL_float>(M[current].value);
      }
      ++symbolic_reuses_;
    }

    // Numeric-only symmetric Ruiz scaling. The original sparsity graph is
    // untouched, congruence preserves inertia, and Cholesk still evaluates
    // its backward error against the unscaled caller matrix.
    if(st_.kkt_equilibration&&!equilibrate_numeric(message)) return false;

    const QDLDL_int pos=QDLDL_factor(static_cast<QDLDL_int>(n),Ap_.data(),Ai_.data(),Ax_.data(),
        Lp_.data(),Li_.data(),Lx_.data(),D_.data(),Dinv_.data(),Lnz_.data(),etree_.data(),
        bwork_.data(),iwork_.data(),fwork_.data());
    if(pos<0) {
      if(allow_dense_border_fallback_ && has_dense_border_fallback()) {
        dense_border_disabled_=true;dense_border_active_=false;symbolic_valid_=false;
        return factor(n,primal_dimension,entries,message);
      }
      message="QDLDL_factor failed: zero pivot / non-factorizable KKT";
      return false;
    }
    positive_pivots_=static_cast<int>(pos);
    negative_pivots_=n-positive_pivots_;
    zero_pivots_=0;
    double minimum=std::numeric_limits<double>::infinity();
    double maximum=0.0;
    int minimum_coordinate=-1;
    for(int k=0;k<n;++k) {
      const double pivot=static_cast<double>(D_[static_cast<std::size_t>(k)]);
      if(!std::isfinite(pivot) || pivot==0.0) {
        ++zero_pivots_;
        continue;
      }
      const double magnitude=std::abs(pivot);
      maximum=std::max(maximum,magnitude);
      if(magnitude<minimum) {
        minimum=magnitude;
        minimum_coordinate=perm_[static_cast<std::size_t>(k)];
      }
    }
    minimum_abs_pivot_=std::isfinite(minimum)?minimum:0.0;
    maximum_abs_pivot_=maximum;
    minimum_abs_pivot_coordinate_=minimum_coordinate;
    probe_ready_=zero_pivots_==0;
    curvature_defect_=probe_ready_
        ?std::max(0,primal_dimension-positive_pivots_):0;
    if(zero_pivots_!=0) {
      message="QDLDL factor contains non-finite or zero diagonal pivots";
      return false;
    }
    if(pos!=static_cast<QDLDL_int>(primal_dimension)) {
      message="QDLDL inertia mismatch: positive pivots="+
          std::to_string(static_cast<long long>(pos))+", expected="+
          std::to_string(primal_dimension);
      return false;
    }
    return true;
  }

  bool solve(Vec& rhs, std::string& message) const override {
    if(static_cast<int>(rhs.size())!=n_) {
      message="QDLDL RHS dimension mismatch";
      return false;
    }
    std::vector<QDLDL_float> x(n_);
    if(st_.kkt_equilibration) {
      for(int i=0;i<n_;++i) {
        const double value=equilibration_scale_[static_cast<std::size_t>(i)]*
                           rhs[perm_[static_cast<std::size_t>(i)]];
        if(!std::isfinite(value)||(rhs[perm_[static_cast<std::size_t>(i)]]!=0.0&&value==0.0)) {
          message="QDLDL equilibration overflowed or erased a right hand side entry";
          return false;
        }
        x[static_cast<std::size_t>(i)]=static_cast<QDLDL_float>(value);
      }
    } else for(int i=0;i<n_;++i) x[i]=rhs[perm_[i]];
    QDLDL_solve(static_cast<QDLDL_int>(n_),Lp_.data(),Li_.data(),Lx_.data(),Dinv_.data(),x.data());
    if(st_.kkt_equilibration) {
      for(int i=0;i<n_;++i)
        rhs[perm_[static_cast<std::size_t>(i)]]=
            equilibration_scale_[static_cast<std::size_t>(i)]*
            static_cast<double>(x[static_cast<std::size_t>(i)]);
    } else for(int i=0;i<n_;++i) rhs[perm_[i]]=x[i];
    for(double v:rhs) if(!std::isfinite(v)) {
      message="QDLDL produced non-finite solution";
      return false;
    }
    return true;
  }
  int curvature_defect() const override { return curvature_defect_; }
  bool solve_curvature_probe(Vec& rhs,std::string& message) const override {
    return probe_ready_&&curvature_defect_>0&&solve(rhs,message);
  }
  int symbolic_analyses() const override { return symbolic_analyses_; }
  int symbolic_reuses() const override { return symbolic_reuses_; }
  int symbolic_invalidations() const override { return symbolic_invalidations_; }
  bool last_symbolic_reused() const override { return last_symbolic_reused_; }
  bool has_symbolic_cache() const override { return symbolic_valid_; }
  int last_positive_pivots() const override { return positive_pivots_; }
  int last_negative_pivots() const override { return negative_pivots_; }
  int last_zero_pivots() const override { return zero_pivots_; }
  double minimum_abs_pivot() const override { return minimum_abs_pivot_; }
  double maximum_abs_pivot() const override { return maximum_abs_pivot_; }
  int minimum_abs_pivot_coordinate() const override {
    return minimum_abs_pivot_coordinate_;
  }
  const char* name() const override { return "qdldl_metis_persistent_symbolic_envelope"; }
 private:
  bool equilibrate_numeric(std::string& message) {
    equilibration_scale_.assign(static_cast<std::size_t>(n_),1.0);
    equilibration_norm_.resize(static_cast<std::size_t>(n_));
    equilibration_step_.resize(static_cast<std::size_t>(n_));
    constexpr double minimum_scale=1e-100,maximum_scale=1e100;
    for(int pass=0;pass<3;++pass) {
      std::fill(equilibration_norm_.begin(),equilibration_norm_.end(),0.0);
      for(int j=0;j<n_;++j)for(QDLDL_int k=Ap_[static_cast<std::size_t>(j)];
          k<Ap_[static_cast<std::size_t>(j+1)];++k) {
        const double a=std::abs(static_cast<double>(Ax_[static_cast<std::size_t>(k)]));
        if(!std::isfinite(a)) {
          message="QDLDL equilibration encountered nonfinite coefficient";
          return false;
        }
        const int i=static_cast<int>(Ai_[static_cast<std::size_t>(k)]);
        equilibration_norm_[static_cast<std::size_t>(i)]=std::max(
            equilibration_norm_[static_cast<std::size_t>(i)],a);
        equilibration_norm_[static_cast<std::size_t>(j)]=std::max(
            equilibration_norm_[static_cast<std::size_t>(j)],a);
      }
      for(int i=0;i<n_;++i) {
        const double norm=equilibration_norm_[static_cast<std::size_t>(i)];
        const double proposed=norm>0.0?1.0/std::sqrt(norm):1.0;
        const double prior=equilibration_scale_[static_cast<std::size_t>(i)];
        const double step=std::clamp(proposed,minimum_scale/prior,maximum_scale/prior);
        const double cumulative=prior*step;
        if(!(step>0.0)||!std::isfinite(step)||!(cumulative>0.0)||!std::isfinite(cumulative)) {
          message="QDLDL equilibration produced invalid diagonal scale";
          return false;
        }
        equilibration_step_[static_cast<std::size_t>(i)]=step;
        equilibration_scale_[static_cast<std::size_t>(i)]=cumulative;
      }
      for(int j=0;j<n_;++j)for(QDLDL_int k=Ap_[static_cast<std::size_t>(j)];
          k<Ap_[static_cast<std::size_t>(j+1)];++k) {
        const int i=static_cast<int>(Ai_[static_cast<std::size_t>(k)]);
        const double original=static_cast<double>(Ax_[static_cast<std::size_t>(k)]);
        const long double product=static_cast<long double>(original)*
            equilibration_step_[static_cast<std::size_t>(i)]*
            equilibration_step_[static_cast<std::size_t>(j)];
        const double scaled=static_cast<double>(product);
        if(!std::isfinite(scaled)||(original!=0.0&&scaled==0.0)) {
          message="QDLDL equilibration would overflow or erase a coefficient";
          return false;
        }
        Ax_[static_cast<std::size_t>(k)]=static_cast<QDLDL_float>(scaled);
      }
    }
    return true;
  }
  const Settings& st_;
  int n_=0,symbolic_primal_dimension_=-1;
  int positive_pivots_=-1,negative_pivots_=-1,zero_pivots_=-1;
  int minimum_abs_pivot_coordinate_=-1;
  double minimum_abs_pivot_=0.0,maximum_abs_pivot_=0.0;
  bool probe_ready_=false;
  int curvature_defect_=0;
  int symbolic_analyses_=0, symbolic_reuses_=0, symbolic_invalidations_=0;
  bool dense_border_active_=false,dense_border_disabled_=false;
  bool allow_dense_border_fallback_=true;
  bool symbolic_valid_=false;
  bool last_symbolic_reused_=false;
  std::vector<std::uint64_t> input_pattern_;
  std::vector<std::size_t> input_to_ax_;
  std::vector<int> perm_,iperm_;
  std::vector<QDLDL_int> Ap_,Ai_,Lp_,Li_,etree_,Lnz_,iwork_;
  std::vector<QDLDL_float> Ax_,Lx_,D_,Dinv_,fwork_;
  std::vector<QDLDL_bool> bwork_;
  Vec equilibration_scale_,equilibration_norm_,equilibration_step_;
};

// Modern adaptation of the supplied legacy cholesk/chold factorization.
//
// The legacy code's useful numerical contract is a no-pivot sparse LDL^T for
// the quasi-definite KKT form, with the expected sign attached to every primal
// and dual pivot and independent diagonal perturbations on both blocks.  Its
// original implementation is process-global and depends on private IPDCA/HSL
// headers.  This adapter keeps that contract while delegating the sparse
// symbolic/numeric kernel to re-entrant QDLDL + METIS.  In caller-adaptive mode
// it factors the exact supplied matrix.  In fixed mode it may grow the two
// already-authorized KKT shifts independently until inertia and a relative
// pivot guard are both satisfied.
class CholeskFactor final : public SymmetricFactorization {
 public:
  explicit CholeskFactor(const Settings& st)
      : st_(st), kernel_settings_(st), kernel_(kernel_settings_) {
    kernel_settings_.linear_backend="qdldl";
    kernel_settings_.kkt_equilibration=st_.cholesk_equilibration;
  }

  bool factor(int n,int primal_dimension,
              const std::vector<SymmetricTriplet>& entries,
              std::string& message) override {
    return factor_impl(n,primal_dimension,entries,message,false,0.0,0.0);
  }

  bool solve(Vec& rhs,std::string& message) const override {
    return solve_checked(rhs,message,true);
  }

  bool increase_quality(std::string& message) override {
    if(!cached_ || explicit_shift_mode_ || !st_.cholesk_adaptive_regularization)
      return false;
    // The supplied chold code grows an absolute epsdiag (starting at 1e-14),
    // independently of unrelated large coefficients elsewhere in the KKT.
    // A global matrix-norm floor destroys that property on heterogeneous
    // power-system models, where one block can be 1e12 times another.
    constexpr double automatic_floor=1e-14;
    const double primal_cap=std::max(last_primal_regularization_,
                                     std::max(0.0,st_.kkt_rho_x_max));
    const double dual_cap=std::max(last_dual_regularization_,
                                   std::max(0.0,st_.kkt_delta_dual_max));
    const double primal_floor=std::min(primal_cap,std::max({
        0.0,st_.kkt_primal_regularization,st_.kkt_rho_x_floor,automatic_floor}));
    const double dual_floor=std::min(dual_cap,std::max({
        0.0,st_.kkt_dual_regularization,st_.kkt_delta_dual_floor,automatic_floor}));
    const double next_primal=grown(last_primal_regularization_,primal_floor,primal_cap,
                                   st_.kkt_rho_x_growth);
    const double next_dual=grown(last_dual_regularization_,dual_floor,dual_cap,
                                 st_.kkt_delta_dual_growth);
    if(!(next_primal>last_primal_regularization_) &&
       !(next_dual>last_dual_regularization_)) return false;
    return factor_impl(cached_n_,cached_primal_dimension_,cached_entries_,message,
                       true,next_primal,next_dual);
  }

  double pivot_tolerance() const override {
    return st_.cholesk_relative_pivot_tolerance;
  }
  int curvature_defect() const override { return kernel_.curvature_defect(); }
  bool solve_curvature_probe(Vec& rhs,std::string& message) const override {
    if(kernel_.curvature_defect()<=0) return false;
    return solve_checked(rhs,message,false);
  }
  int failed_pivot_coordinate() const override {
    return kernel_.failed_pivot_coordinate();
  }
  double last_backward_error() const override { return last_backward_error_; }
  double last_initial_backward_error() const override {
    return last_initial_backward_error_;
  }
  double last_residual_inf() const override { return last_residual_inf_; }
  int last_refinement_steps() const override { return last_refinement_steps_; }
  int symbolic_analyses() const override { return kernel_.symbolic_analyses(); }
  int symbolic_reuses() const override { return kernel_.symbolic_reuses(); }
  int symbolic_invalidations() const override {
    return kernel_.symbolic_invalidations();
  }
  bool last_symbolic_reused() const override {
    return kernel_.last_symbolic_reused();
  }
  bool has_symbolic_cache() const override { return kernel_.has_symbolic_cache(); }
  int last_positive_pivots() const override {
    return kernel_.last_positive_pivots();
  }
  int last_negative_pivots() const override {
    return kernel_.last_negative_pivots();
  }
  int last_zero_pivots() const override { return kernel_.last_zero_pivots(); }
  double minimum_abs_pivot() const override {
    return kernel_.minimum_abs_pivot();
  }
  double maximum_abs_pivot() const override {
    return kernel_.maximum_abs_pivot();
  }
  int minimum_abs_pivot_coordinate() const override {
    return kernel_.minimum_abs_pivot_coordinate();
  }
  double last_primal_regularization() const override {
    return last_primal_regularization_;
  }
  double last_dual_regularization() const override {
    return last_dual_regularization_;
  }
  int last_regularization_attempts() const override {
    return last_regularization_attempts_;
  }
  const char* name() const override {
    return "cholesk_qdldlt_metis_sign_regularized_refined";
  }

 private:
  double grown(double value,double floor,double cap,double growth) const {
    if(!(cap>value)) return value;
    growth=std::max(1.1,growth);
    const double candidate=value>0.0?value*growth:floor;
    return std::min(cap,std::max(floor,candidate));
  }

  bool factor_impl(int n,int primal_dimension,
                   const std::vector<SymmetricTriplet>& entries,
                   std::string& message,bool forced,double forced_primal,
                   double forced_dual) {
    ready_=false;
    last_regularization_attempts_=0;
    last_backward_error_=last_initial_backward_error_=last_residual_inf_=0.0;
    last_refinement_steps_=0;
    if(n<0 || primal_dimension<0 || primal_dimension>n) {
      message="Cholesk received invalid KKT dimensions";
      return false;
    }
    if(!std::isfinite(st_.cholesk_relative_pivot_tolerance) ||
       st_.cholesk_relative_pivot_tolerance<0.0 ||
       !std::isfinite(st_.kkt_rho_x_growth) || st_.kkt_rho_x_growth<=1.0 ||
       !std::isfinite(st_.kkt_rho_x_decay) || st_.kkt_rho_x_decay<=0.0 ||
       st_.kkt_rho_x_decay>1.0 ||
       !std::isfinite(st_.kkt_delta_dual_growth) || st_.kkt_delta_dual_growth<=1.0 ||
       !std::isfinite(st_.kkt_delta_dual_decay) || st_.kkt_delta_dual_decay<=0.0 ||
       st_.kkt_delta_dual_decay>1.0 || st_.kkt_regularization_retry_max<0) {
      message="invalid Cholesk pivot/regularization controls";
      return false;
    }

    cached_=true;
    cached_n_=n;
    cached_primal_dimension_=primal_dimension;
    cached_entries_=entries;
    const EntryList unshifted=aggregate_upper(n,entries);
    unshifted_matrix_inf_norm_=symmetric_inf_norm(n,unshifted);
    constexpr double automatic_floor=1e-14;

    explicit_shift_mode_=st_.kkt_adaptive_regularization;
    const bool repair=!explicit_shift_mode_&&st_.cholesk_adaptive_regularization;
    const bool defer_border_fallback=repair && st_.dca_engine=="majorants";
    kernel_.allow_dense_border_fallback(!defer_border_fallback);
    const double base_primal=explicit_shift_mode_
        ?0.0:std::max(0.0,st_.kkt_primal_regularization);
    const double base_dual=explicit_shift_mode_
        ?0.0:std::max(0.0,st_.kkt_dual_regularization);
    const double primal_cap=repair
        ?std::max(base_primal,std::max(0.0,st_.kkt_rho_x_max)):base_primal;
    const double dual_cap=repair
        ?std::max(base_dual,std::max(0.0,st_.kkt_delta_dual_max)):base_dual;
    const double primal_floor=repair
        ?std::min(primal_cap,std::max({base_primal,st_.kkt_rho_x_floor,
                                      automatic_floor})):base_primal;
    const double dual_floor=repair
        ?std::min(dual_cap,std::max({base_dual,st_.kkt_delta_dual_floor,
                                    automatic_floor})):base_dual;

    double primal=base_primal,dual=base_dual;
    // Cholesk assembles its regularization explicitly so the kernel never
    // applies a second uniform shift. In particular, the dual control is a
    // floor on weak/zero dual diagonals (the legacy MAX(dm,epsdiag) rule), not
    // a perturbation of already well-scaled inequality/atom borders.
    kernel_settings_.kkt_adaptive_regularization=true;
    if(!explicit_shift_mode_) {
      if(forced) {
        primal=std::clamp(forced_primal,base_primal,primal_cap);
        dual=std::clamp(forced_dual,base_dual,dual_cap);
      } else if(repair) {
        // Try the caller's matrix first.  The diagonal floors are safeguards,
        // not a permanent modification of every Newton equation.  This is
        // especially important for very ill-conditioned OPF KKT systems:
        // even a 1e-12 dual shift can have a large forward effect although
        // the shifted-system backward error is tiny.  If the exact numeric
        // factorization fails, the retry ladder below activates the smallest
        // authorized primal/dual floor and grows only the deficient block.
        primal=base_primal;
        dual=base_dual;
      }
    }

    const int retry_budget=repair?st_.kkt_regularization_retry_max:0;
    std::string last_reason;
    Vec repair_scale; // Lazy: no extra matrix traversal on an exact success.
    for(int attempt=0;attempt<=retry_budget;++attempt) {
      ++last_regularization_attempts_;
      kernel_settings_.kkt_primal_regularization=0.0;
      kernel_settings_.kkt_dual_regularization=0.0;
      last_primal_regularization_=explicit_shift_mode_?0.0:primal;
      last_dual_regularization_=explicit_shift_mode_?0.0:dual;
      std::vector<SymmetricTriplet> trial_entries=entries;
      if(!explicit_shift_mode_) {
        if(st_.dca_engine=="majorants"&&(primal!=0.0||dual!=0.0)&&repair_scale.empty()) {
          repair_scale.assign(static_cast<std::size_t>(n),1.0);
          for(const auto& entry:unshifted) {
            repair_scale[entry.i]=std::max(repair_scale[entry.i],std::abs(entry.value));
            repair_scale[entry.j]=std::max(repair_scale[entry.j],std::abs(entry.value));
          }
        }
        if(primal!=0.0) for(int i=0;i<primal_dimension;++i)
          trial_entries.push_back({i,i,primal*(repair_scale.empty()?1.0:repair_scale[i])});
        if(dual!=0.0) {
          Vec input_diagonal(static_cast<std::size_t>(n),0.0);
          for(const auto& entry:unshifted) if(entry.i==entry.j)
            input_diagonal[static_cast<std::size_t>(entry.i)]=entry.value;
          for(int i=primal_dimension;i<n;++i) {
            const double diagonal=input_diagonal[static_cast<std::size_t>(i)];
            const double floor=dual*(repair_scale.empty()?1.0:repair_scale[i]);
            double addition=0.0;
            if(diagonal>=0.0) addition=-floor;
            else if(diagonal>-floor) addition=-floor-diagonal;
            if(addition!=0.0) trial_entries.push_back({i,i,addition});
          }
        }
      }
      matrix_=aggregate_upper(n,trial_entries);
      matrix_inf_norm_=symmetric_inf_norm(n,matrix_);

      std::string kernel_message;
      const bool factored=kernel_.factor(
          n,primal_dimension,trial_entries,kernel_message);
      const double minimum=kernel_.minimum_abs_pivot();
      // Use the pivot coordinate's own diagonal scale. The original chold
      // implementation guarded min|D| against an absolute epsdiag; making
      // that test relative to the GLOBAL KKT norm rejects valid 1e-8 dual
      // floors whenever an unrelated primal coefficient is around 1e12.
      // The local form keeps an absolute guard below unit scale and a relative
      // guard for genuinely large diagonal coordinates. Inertia and the solve
      // backward error remain independent acceptance gates.
      double pivot_diagonal=0.0;
      const int pivot_coordinate=kernel_.minimum_abs_pivot_coordinate();
      if(pivot_coordinate>=0)for(const auto& entry:matrix_)
        if(entry.i==pivot_coordinate&&entry.j==pivot_coordinate) {
          pivot_diagonal=entry.value;
          break;
        }
      const double pivot_scale=kernel_settings_.kkt_equilibration
          ?1.0:std::max(1.0,std::abs(pivot_diagonal));
      const bool weak_pivot=factored &&
          minimum<=st_.cholesk_relative_pivot_tolerance*pivot_scale;
      if(factored&&!weak_pivot) {
        ready_=true;
        const bool same_memory_dimensions=
            last_success_n_==n&&last_success_primal_dimension_==primal_dimension;
        const bool repaired=last_regularization_attempts_>1 || forced;
        if(!same_memory_dimensions) {
          last_success_primal_regularization_=0.0;
          last_success_dual_regularization_=0.0;
          clean_regularization_successes_=0;
        }
        if(repaired) {
          // Retain the last shifts which actually repaired this KKT.  A later
          // exact success must not erase them immediately: neighbouring DCA
          // systems commonly alternate between a clean and a singular
          // numeric factorization while sharing the same symbolic structure.
          last_success_primal_regularization_=last_primal_regularization_;
          last_success_dual_regularization_=last_dual_regularization_;
          clean_regularization_successes_=0;
        } else if(same_memory_dimensions &&
                  (last_success_primal_regularization_>0.0 ||
                   last_success_dual_regularization_>0.0)) {
          ++clean_regularization_successes_;
          const int clean_window=std::max(1,st_.kkt_rho_x_clean_outer);
          if(clean_regularization_successes_>=clean_window) {
            last_success_primal_regularization_ *= st_.kkt_rho_x_decay;
            last_success_dual_regularization_ *= st_.kkt_delta_dual_decay;
            if(last_success_primal_regularization_<=primal_floor)
              last_success_primal_regularization_=0.0;
            if(last_success_dual_regularization_<=dual_floor)
              last_success_dual_regularization_=0.0;
            clean_regularization_successes_=0;
          }
        }
        last_success_n_=n;
        last_success_primal_dimension_=primal_dimension;
        message.clear();
        if(last_regularization_attempts_>1 && st_.live_display_level>=2)
          std::cout<<"[CHOLESK-REG] attempts="<<last_regularization_attempts_
                   <<" primal="<<last_primal_regularization_
                   <<" dual="<<last_dual_regularization_
                   <<" pivot_ratio="<<minimum/pivot_scale
                   <<" status=accepted\n";
        return true;
      }

      if(weak_pivot) {
        std::ostringstream os;
        os<<"relative pivot "<<minimum/pivot_scale
          <<" is below "<<st_.cholesk_relative_pivot_tolerance
          <<" (coordinate="<<pivot_coordinate
          <<", |D|="<<minimum<<", local_scale="<<pivot_scale<<")";
        last_reason=os.str();
      } else last_reason=kernel_message;
      // The outer majorant controller owns tangent-inertia correction.
      // Do not spend a numerical pivot-repair ladder on a curvature failure.
      if(st_.dca_engine=="majorants"&&!factored&&
         kernel_message.find("inertia mismatch")!=std::string::npos &&
         kernel_.last_positive_pivots()<primal_dimension)break;
      if(attempt==retry_budget) break;

      bool grow_primal=false,grow_dual=false;
      if(weak_pivot) {
        const int coordinate=kernel_.minimum_abs_pivot_coordinate();
        grow_primal=coordinate>=0&&coordinate<primal_dimension;
        grow_dual=coordinate>=primal_dimension;
        // In a saddle system a small primal-labelled LDL pivot can be
        // controlled by the constraint Schur complement. If a primal rescue
        // still leaves a weak pivot, activate the complementary dual floor.
        if(grow_primal&&primal>base_primal)grow_dual=true;
      } else if(kernel_.last_positive_pivots()>=0) {
        grow_primal=kernel_.last_positive_pivots()<primal_dimension;
        grow_dual=kernel_.last_negative_pivots()<n-primal_dimension;
      }
      if(!grow_primal&&!grow_dual) grow_primal=grow_dual=true;
      double warm_primal=primal_floor,warm_dual=dual_floor;
      if(st_.cholesk_regularization_warm_start&&
         last_success_n_==n&&last_success_primal_dimension_==primal_dimension) {
        if(last_success_primal_regularization_>0.0)
          warm_primal=std::min(primal_cap,std::max(primal_floor,
              st_.kkt_rho_x_decay*last_success_primal_regularization_));
        if(last_success_dual_regularization_>0.0)
          warm_dual=std::min(dual_cap,std::max(dual_floor,
              st_.kkt_delta_dual_decay*last_success_dual_regularization_));
      }
      const double next_primal=grow_primal
          ?(primal==base_primal&&warm_primal>primal?warm_primal:
            grown(primal,primal_floor,primal_cap,st_.kkt_rho_x_growth)):primal;
      const double next_dual=grow_dual
          ?(dual==base_dual&&warm_dual>dual?warm_dual:
            grown(dual,dual_floor,dual_cap,st_.kkt_delta_dual_growth)):dual;
      if(!(next_primal>primal) && !(next_dual>dual)) break;
      if(st_.live_display_level>=3)
        std::cout<<"[CHOLESK-REG] attempt="<<attempt+1
                 <<" reason=\""<<last_reason<<"\""
                 <<" primal="<<primal<<"->"<<next_primal
                 <<" dual="<<dual<<"->"<<next_dual<<"\n";
      primal=next_primal;
      dual=next_dual;
    }
    // A genuine curvature defect belongs to the outer controller. An
    // ordering change cannot correct inertia, so do not densify for it.
    // Otherwise retain the original-order rescue, but only after the
    // finite local numerical ladder. Both phases keep the same backend,
    // pivot threshold and regularization caps. There are at most two phases.
    if(defer_border_fallback && kernel_.has_dense_border_fallback() &&
       last_reason.find("inertia mismatch")==std::string::npos) {
      const int first_phase_attempts=last_regularization_attempts_;
      if(st_.live_display_level>=2)
        std::cout<<"[CHOLESK-ORDER-RETRY] local_attempts="<<first_phase_attempts
                 <<" fallback=original-order same_backend=1"<<std::endl;
      kernel_.use_original_border_order();
      const bool ok=factor_impl(n,primal_dimension,entries,message,
                               forced,forced_primal,forced_dual);
      last_regularization_attempts_+=first_phase_attempts;
      return ok;
    }
    std::ostringstream os;
    os<<"Cholesk factorization rejected after "<<last_regularization_attempts_
      <<" attempt(s): "<<last_reason
      <<"; primal_regularization="<<last_primal_regularization_
      <<" dual_regularization="<<last_dual_regularization_;
    message=os.str();
    return false;
  }

  bool solve_checked(Vec& rhs,std::string& message,bool require_ready) const {
    if(require_ready&&!ready_) {
      message="Cholesk solve requires a successful inertia and pivot certificate";
      return false;
    }
    if(static_cast<int>(rhs.size())!=cached_n_) {
      message="Cholesk RHS dimension mismatch";
      return false;
    }
    scratch_b_=rhs;
    scratch_x_=rhs;
    if(!kernel_.solve(scratch_x_,message)) return false;
    const double binf=inf_norm(scratch_b_);
    double rinf=0.0;
    double berr=backward_error_into_known_binf(
        matrix_,matrix_inf_norm_,binf,scratch_x_,scratch_b_,scratch_residual_,&rinf);
    last_initial_backward_error_=berr;
    int refinement=0;
    while(berr>st_.kkt_backward_error_tolerance &&
          refinement<st_.kkt_iterative_refinement_steps) {
      if(!kernel_.solve(scratch_residual_,message)) return false;
      axpy(1.0,scratch_residual_,scratch_x_);
      ++refinement;
      berr=backward_error_into_known_binf(
          matrix_,matrix_inf_norm_,binf,scratch_x_,scratch_b_,scratch_residual_,&rinf);
    }
    rhs=scratch_x_;
    last_backward_error_=berr;
    last_residual_inf_=rinf;
    last_refinement_steps_=refinement;
    if(!std::isfinite(berr)) {
      message="Cholesk produced non-finite backward error";
      return false;
    }
    if(berr>st_.kkt_backward_error_reject) {
      std::ostringstream os;
      os<<"Cholesk backward error too large after refinement: "<<berr;
      message=os.str();
      return false;
    }
    return true;
  }

  const Settings& st_;
  Settings kernel_settings_;
  QdldlMetisFactor kernel_;
  int cached_n_=0,cached_primal_dimension_=0;
  int last_success_n_=-1,last_success_primal_dimension_=-1;
  bool cached_=false,ready_=false,explicit_shift_mode_=false;
  std::vector<SymmetricTriplet> cached_entries_;
  EntryList matrix_;
  double matrix_inf_norm_=0.0,unshifted_matrix_inf_norm_=0.0;
  double last_primal_regularization_=0.0,last_dual_regularization_=0.0;
  double last_success_primal_regularization_=0.0;
  double last_success_dual_regularization_=0.0;
  int clean_regularization_successes_=0;
  int last_regularization_attempts_=0;
  mutable Vec scratch_b_,scratch_x_,scratch_residual_;
  mutable double last_backward_error_=0.0;
  mutable double last_initial_backward_error_=0.0;
  mutable double last_residual_inf_=0.0;
  mutable int last_refinement_steps_=0;
};
#endif

#ifdef LAZYIPDCA_HAVE_MUMPS
class MumpsMetisFactor final : public SymmetricFactorization {
 public:
  explicit MumpsMetisFactor(const Settings& st,int maximum_workspace_mb=0)
      : st_(st),maximum_workspace_mb_(maximum_workspace_mb) {}
  ~MumpsMetisFactor() override { finish(); }

  bool factor(int n, int primal_dimension,
              const std::vector<SymmetricTriplet>& entries,
              std::string& message) override {
    LinearPhaseTime assembly_time(assembly_seconds_);
    // Only the aggregate controller opts into its requested structural-zero
    // policy and compiled triplet layout. Other MUMPS callers retain their
    // historical zero filtering and assembly path.
    const bool preserve_zeros=st_.dca_aggregate &&
        st_.dca_aggregate_kkt_layout=="persistent" && st_.kkt_preserve_structural_zeros;
    const bool cached=st_.dca_aggregate && st_.dca_aggregate_fast_assembly &&
        st_.kkt_dual_regularization_end<0 &&
        scatter_.assemble(n,entries,st_.kkt_adaptive_regularization?-1:primal_dimension,
            st_.kkt_primal_regularization,st_.kkt_dual_regularization,
            preserve_zeros,matrix_);
    if(!cached)matrix_=st_.kkt_adaptive_regularization
        ? aggregate_upper(n,entries,-1,0.0,0.0,preserve_zeros)
        : aggregate_upper(n,entries,primal_dimension,
                          st_.kkt_primal_regularization,
                          st_.kkt_dual_regularization,preserve_zeros,
                          st_.kkt_dual_regularization_end);
    const EntryList& M=matrix_;
    expected_negative_=n-primal_dimension;
    matrix_inf_norm_=symmetric_inf_norm(n,M);
    const auto uncached_pattern=cached?std::vector<std::uint64_t>{}:entry_pattern(M);
    const auto& current_pattern=cached?scatter_.pattern():uncached_pattern;
    const bool same_dimension=initialized_ && analyzed_ && n==n_;
    const bool reusable=same_dimension &&
        pattern_contains(pattern_,current_pattern);
    last_symbolic_reused_=reusable;

    if(!reusable) {
      if(initialized_ && analyzed_) ++symbolic_invalidations_;
      // A capped Phase-I factor may retain one admitted matrix pattern,
      // never an accumulating union of changing numerical cliques. Ordinary
      // main KKT contexts keep their historical symbolic envelope policy.
      const std::vector<std::uint64_t> envelope=same_dimension&&maximum_workspace_mb_==0
          ? pattern_union(pattern_,current_pattern)
          : current_pattern;
      assembly_time.stop();
      // Context lifecycle and graph/index preparation belong to the symbolic
      // phase. No numeric factorization is timed inside this scope.
      LinearPhaseTime symbolic_time(symbolic_seconds_);
      finish();
      n_=n;
      pattern_=envelope;
      irn_.clear(); jcn_.clear(); a_.clear();
      irn_.reserve(pattern_.size());
      jcn_.reserve(pattern_.size());
      a_.assign(pattern_.size(),0.0);

      std::size_t current=0;
      for(std::size_t k=0;k<pattern_.size();++k) {
        const std::uint64_t key=pattern_[k];
        irn_.push_back(static_cast<MUMPS_INT>(upper_key_row(key)+1));
        jcn_.push_back(static_cast<MUMPS_INT>(upper_key_col(key)+1));
        if(current<current_pattern.size() && current_pattern[current]==key) {
          a_[k]=M[current].value;
          ++current;
        }
      }

      id_={};
      id_.job=-1; id_.par=1; id_.sym=2; id_.comm_fortran=-987654;
      dmumps_c(&id_); initialized_=true;
      id_.icntl[0]=-1; id_.icntl[1]=-1; id_.icntl[2]=-1; id_.icntl[3]=0;
      // The explicit capped Phase-I context owns a separate workspace policy.
      // A successful uncapped KKT margin can exceed this factor's hard cap
      // even when the native default factors the same small sparse matrix.
      const int learned_margin=maximum_workspace_mb_>0?20:
          detail::learned_mumps_memory_margin();
      if(learned_margin>id_.icntl[13]) {
        id_.icntl[13]=static_cast<MUMPS_INT>(learned_margin);
        if(st_.live_display_level>0)
          std::cout<<"[MUMPS-MEMORY] learned_margin_percent="<<learned_margin
                   <<" scope=process\n";
      }
      if(st_.linear_metis) {
        id_.icntl[6]=5;
        id_.icntl[27]=1;
      }
      id_.cntl[0]=quality_pivot_;
      // Matching is a numerical ordering aid for an explicitly requested
      // recovery context. METIS still orders its compressed symmetric graph.
      // Ordinary successful factors retain their historical ordering policy.
      if(st_.mumps_symmetric_matching) {
        id_.icntl[5]=5;  // weighted matching, with numerical values supplied
        id_.icntl[11]=2; // METIS on the matching-compressed symmetric graph
      }
      id_.icntl[12]=1;
      if(maximum_workspace_mb_>0) {
        id_.icntl[22]=static_cast<MUMPS_INT>(maximum_workspace_mb_);
        if(st_.live_display_level>0)
          std::cout<<"[MUMPS-WORKSPACE-CAP] megabytes="<<maximum_workspace_mb_
                   <<" scope=explicit-sparse-restoration main_KKT_unchanged=1"
                   <<" envelope_entries="<<a_.size()<<" cumulative_pattern_union=0\n";
      }
      id_.n=static_cast<MUMPS_INT>(n);
      id_.nnz=static_cast<MUMPS_INT8>(a_.size());
      id_.irn=irn_.data(); id_.jcn=jcn_.data(); id_.a=a_.data();
      id_.job=1;
      dmumps_c(&id_);
      if(id_.infog[0]<0) {
        message="MUMPS analysis failed INFOG(1)="+
            std::to_string(id_.infog[0])+" INFOG(2)="+
            std::to_string(id_.infog[1]);
        finish();
        return false;
      }
      if(!verify_requested_ordering(message,"initial")) {
        finish();
        return false;
      }
      analyzed_=true;
      ++symbolic_analyses_;
    } else {
      if(a_.size()!=pattern_.size()) {
        message="MUMPS cached envelope storage is inconsistent";
        finish();
        return false;
      }
      std::fill(a_.begin(),a_.end(),0.0);
      std::size_t envelope=0;
      for(std::size_t current=0;current<current_pattern.size();++current) {
        while(envelope<pattern_.size() &&
              pattern_[envelope]<current_pattern[current]) ++envelope;
        if(envelope>=pattern_.size() ||
           pattern_[envelope]!=current_pattern[current]) {
          message="MUMPS cached envelope lost a current pattern entry";
          finish();
          return false;
        }
        a_[envelope]=M[current].value;
      }
      id_.a=a_.data();
      ++symbolic_reuses_;
    }

    assembly_time.stop();
    return numeric_factor(message);
  }

  double pivot_tolerance() const override { return quality_pivot_; }
  bool increase_quality(std::string& message) override {
    if(!initialized_ || !analyzed_ || quality_pivot_>=0.1) return false;
    quality_pivot_=std::min(0.1,std::sqrt(quality_pivot_));
    id_.cntl[0]=quality_pivot_;
    return numeric_factor(message);
  }

  bool numeric_factor(std::string& message) {
    const MUMPS_INT margin_before_numeric=id_.icntl[13];
    const auto attempt_numeric=[&]() {
      LinearPhaseTime numeric_time(numeric_seconds_);
      id_.job=2;
      dmumps_c(&id_);
      if(st_.mumps_memory_recovery) {
        detail::recover_mumps_memory(id_.infog[0],id_.icntl[13],[&](){
          if(st_.live_display_level>0)
            std::cout<<"[MUMPS-MEMORY] retry_same_matrix=1 error="<<id_.infog[0]
                     <<" margin_percent="<<id_.icntl[13]<<"\n";
          id_.job=2;dmumps_c(&id_);
        },maximum_workspace_mb_==0);
      }
    };
    attempt_numeric();
    if(detail::mumps_reanalyze_after_memory_failure(st_.mumps_memory_recovery,
        last_symbolic_reused_,id_.infog[0])) {
      const int code=id_.infog[0],saturated_margin=id_.icntl[13];
      // Values can change markedly while the graph is identical. An old
      // numerical ordering/scaling can exhaust real workspace even though a
      // fresh analysis of this SAME sparse matrix factors in the normal budget.
      // Try one new analysis only. Neither a_, its envelope/pattern, the KKT
      // diagonal nor the configured backend is changed by this recovery.
      {
        LinearPhaseTime symbolic_time(symbolic_seconds_);
        id_.icntl[13]=margin_before_numeric;
        id_.job=1;dmumps_c(&id_);
        ++symbolic_invalidations_;last_symbolic_reused_=false;
        if(id_.infog[0]<0) {
          message="MUMPS same-matrix reanalysis failed INFOG(1)="+
              std::to_string(id_.infog[0])+" INFOG(2)="+std::to_string(id_.infog[1]);
          analyzed_=false;return false;
        }
        if(!verify_requested_ordering(message,"same-matrix-reanalysis")) {
          analyzed_=false;return false;
        }
        ++symbolic_analyses_;
      }
      if(st_.live_display_level>0)
        std::cout<<"[MUMPS-MEMORY-REANALYSIS] same_matrix=1 pattern_unchanged=1"
          <<" attempt=1 previous_error="<<code<<" saturated_margin="<<saturated_margin
          <<" restored_margin="<<id_.icntl[13]<<" n="<<n_<<" nnz="<<a_.size()
          <<" tau_and_function_corrections_unchanged=1\n";
      attempt_numeric();
    }
    if(id_.infog[0]<0) {
      message="MUMPS factorization failed INFOG(1)="+
          std::to_string(id_.infog[0])+" INFOG(2)="+
          std::to_string(id_.infog[1]);
      // INFOG(13) counts repeated delays, so a count greater than n proves
      // repeated numerical pivot rejection. A capped factor is never
      // diagnosed this way. The marker
      // authorizes only a bounded refinement preconditioner at the caller;
      // it is not an inertia/curvature diagnosis of the failed matrix.
      if(maximum_workspace_mb_==0 && id_.infog[0]==-9 && id_.infog[12]>n_) {
        message+=" pivot_delay_workspace_exhaustion=1 delayed_pivots="+
            std::to_string(id_.infog[12]);
        if(st_.live_display_level>0)
          std::cout<<"[MUMPS-PIVOT-DELAY] n="<<n_<<" nnz="<<a_.size()
              <<" delayed_pivots="<<id_.infog[12]<<" max_front="<<id_.infog[10]
              <<" estimated_mb="<<id_.infog[15]<<" allocated_mb="<<id_.infog[17]
              <<" margin_percent="<<id_.icntl[13]
              <<" bounded_unperturbed_refinement_eligible=1 curvature_claim=0\n";
      }
      return false;
    }
    if(st_.mumps_symmetric_matching && st_.live_display_level>0)
      std::cout<<"[MUMPS-MATCHED-RECOVERY] n="<<n_<<" nnz="<<a_.size()
          <<" matching_ordering="<<id_.infog[23]<<" matching_method="<<id_.infog[22]
          <<" metis_ordering="<<id_.infog[6]<<" delayed_pivots="<<id_.infog[12]
          <<" max_front="<<id_.infog[10]<<" estimated_mb="<<id_.infog[15]
          <<" allocated_mb="<<id_.infog[17]<<" negative_pivots="<<id_.infog[11]
          <<" expected_negative="<<expected_negative_<<" numeric_seconds_total="<<numeric_seconds_
          <<" unperturbed_audit_required=1\n";
    const int negative_pivots=static_cast<int>(id_.infog[11]);
    const int expected_negative=expected_negative_;
    if(negative_pivots!=expected_negative) {
      message="MUMPS inertia mismatch: negative pivots="+
          std::to_string(negative_pivots)+", expected="+
          std::to_string(expected_negative);
      return false;
    }
    last_backward_error_=0.0;
    last_residual_inf_=0.0;
    last_refinement_steps_=0;
    return true;
  }

  bool verify_requested_ordering(std::string& message,const char* phase) const {
    if(st_.linear_metis)
      std::cout<<"[MUMPS-ORDERING] phase="<<phase
               <<" requested_ordering=5 requested_analysis=1"
               <<" actual_ordering="<<id_.infog[6]
               <<" actual_analysis="<<id_.infog[31]<<"\n";
    if(!detail::mumps_requested_ordering_verified(st_.linear_metis,
        static_cast<int>(id_.infog[6]),static_cast<int>(id_.infog[31]),message))return false;
    if(st_.mumps_symmetric_matching && (id_.infog[23]!=2 || id_.infog[22]!=5)) {
      message="MUMPS requested weighted matching ordering was not honored: INFOG(24)="+
          std::to_string(id_.infog[23])+" INFOG(23)="+std::to_string(id_.infog[22]);
      return false;
    }
    return true;
  }

  bool solve(Vec& rhs, std::string& message) const override {
    LinearPhaseTime solve_time(linear_solve_seconds_);
    if(static_cast<int>(rhs.size())!=n_) {
      message="MUMPS RHS dimension mismatch";
      return false;
    }
    const Vec b=rhs;
    Vec x=b;
    if(!solve_raw(x,message)) return false;
    double rinf=0.0;
    double berr=backward_error(matrix_,matrix_inf_norm_,x,b,&rinf);
    int refinement=0;
    while(berr>st_.kkt_backward_error_tolerance &&
          refinement<st_.kkt_iterative_refinement_steps) {
      Vec corr=symmetric_residual(matrix_,x,b);
      if(!solve_raw(corr,message)) return false;
      axpy(1.0,corr,x);
      ++refinement;
      berr=backward_error(matrix_,matrix_inf_norm_,x,b,&rinf);
    }
    rhs=std::move(x);
    last_backward_error_=berr;
    last_residual_inf_=rinf;
    last_refinement_steps_=refinement;
    if(!std::isfinite(berr)) {
      message="MUMPS produced non-finite backward error";
      return false;
    }
    if(berr>st_.kkt_backward_error_reject) {
      std::ostringstream os;
      os << "MUMPS backward error too large after refinement: " << berr;
      message=os.str();
      return false;
    }
    return true;
  }

  double last_backward_error() const override { return last_backward_error_; }
  double last_residual_inf() const override { return last_residual_inf_; }
  int last_refinement_steps() const override { return last_refinement_steps_; }
  int symbolic_analyses() const override { return symbolic_analyses_; }
  int symbolic_reuses() const override { return symbolic_reuses_; }
  int symbolic_invalidations() const override { return symbolic_invalidations_; }
  double assembly_seconds() const override { return assembly_seconds_; }
  double symbolic_seconds() const override { return symbolic_seconds_; }
  double numeric_seconds() const override { return numeric_seconds_; }
  double linear_solve_seconds() const override { return linear_solve_seconds_; }
  int numeric_scatter_builds() const override { return scatter_.builds(); }
  int numeric_scatter_reuses() const override { return scatter_.reuses(); }
  int numeric_scatter_invalidations() const override { return scatter_.invalidations(); }
  bool last_numeric_scatter_reused() const override { return scatter_.last_reused(); }
  bool last_symbolic_reused() const override { return last_symbolic_reused_; }
  bool has_symbolic_cache() const override { return initialized_ && analyzed_; }
  int last_negative_pivots() const override {return initialized_ && id_.infog[0]>=0?int(id_.infog[11]):-1;}
  int last_positive_pivots() const override {const int neg=last_negative_pivots();return neg>=0?n_-neg:-1;}
  const char* name() const override {
    return "mumps_sym_indef_metis_persistent_symbolic_envelope_refined";
  }

 private:
  bool solve_raw(Vec& rhs, std::string& message) const {
    auto* self=const_cast<MumpsMetisFactor*>(this);
    self->rhs_=rhs;
    self->id_.rhs=self->rhs_.data();
    self->id_.nrhs=1;
    self->id_.lrhs=static_cast<MUMPS_INT>(n_);
    self->id_.job=3;
    dmumps_c(&self->id_);
    if(self->id_.infog[0]<0) {
      message="MUMPS solve failed INFOG(1)="+
          std::to_string(self->id_.infog[0])+" INFOG(2)="+
          std::to_string(self->id_.infog[1]);
      return false;
    }
    rhs=self->rhs_;
    for(double v:rhs) if(!std::isfinite(v)) {
      message="MUMPS produced non-finite solution";
      return false;
    }
    return true;
  }

  void finish() {
    if(initialized_) {
      id_.job=-2;
      dmumps_c(&id_);
    }
    initialized_=false;
    analyzed_=false;
    n_=0;
    irn_.clear(); jcn_.clear(); a_.clear(); pattern_.clear();
  }

  const Settings& st_;
  int maximum_workspace_mb_=0;
  int n_=0,expected_negative_=0;
  double quality_pivot_=0.01;
  int symbolic_analyses_=0, symbolic_reuses_=0, symbolic_invalidations_=0;
  bool initialized_=false, analyzed_=false;
  bool last_symbolic_reused_=false;
  std::vector<std::uint64_t> pattern_;
  mutable DMUMPS_STRUC_C id_{};
  std::vector<MUMPS_INT> irn_,jcn_;
  std::vector<double> a_;
  mutable Vec rhs_;
  EntryList matrix_;
  detail::StableUpperAssemblyCache scatter_;
  double assembly_seconds_=0.0,symbolic_seconds_=0.0,numeric_seconds_=0.0;
  mutable double linear_solve_seconds_=0.0;
  double matrix_inf_norm_=0.0;
  mutable double last_backward_error_=0.0;
  mutable double last_residual_inf_=0.0;
  mutable int last_refinement_steps_=0;
};
#endif

} // namespace

bool linear_backend_compiled(const std::string& backend) {
  if (backend=="native") return true;
#ifdef LAZYIPDCA_HAVE_HSL
  if(backend=="ma27") return true;
#endif
#if defined(LAZYIPDCA_HAVE_LDL) && defined(LAZYIPDCA_HAVE_METIS)
  if (backend=="ldl") return true;
#endif
#if defined(LAZYIPDCA_HAVE_QDLDL) && defined(LAZYIPDCA_HAVE_METIS)
  if (backend=="qdldl") return true;
  if (backend=="cholesk") return true;
#endif
#ifdef LAZYIPDCA_HAVE_MUMPS
  if (backend=="mumps") return true;
#endif
  return false;
}

std::string compiled_linear_backends() {
  std::string s="native";
#ifdef LAZYIPDCA_HAVE_HSL
  s+=",ma27";
#endif
#if defined(LAZYIPDCA_HAVE_LDL) && defined(LAZYIPDCA_HAVE_METIS)
  s += ",ldl";
#endif
#if defined(LAZYIPDCA_HAVE_QDLDL) && defined(LAZYIPDCA_HAVE_METIS)
  s += ",qdldl";
  s += ",cholesk";
#endif
#ifdef LAZYIPDCA_HAVE_MUMPS
  s += ",mumps";
#endif
  return s;
}

std::unique_ptr<SymmetricFactorization> make_memory_bounded_symmetric_factorization(
    const Settings& settings,std::string& message,int maximum_workspace_mb) {
#ifdef LAZYIPDCA_HAVE_MUMPS
  if(settings.linear_backend=="mumps"&&settings.linear_metis&&
      maximum_workspace_mb>0&&maximum_workspace_mb<=128)
    return trace_native_restoration_factor(std::make_unique<MumpsMetisFactor>(settings,maximum_workspace_mb));
#endif
  message="bounded sparse restoration requires MUMPS+METIS and workspace cap in 1..128 MB";
  return {};
}

std::unique_ptr<SymmetricFactorization>
make_symmetric_factorization_untraced(const Settings& settings, std::string& message) {
#ifdef LAZYIPDCA_HAVE_HSL
  if(settings.linear_backend=="ma27") {
    try {
      if(!std::isfinite(settings.hsl_pivot_tolerance)||settings.hsl_pivot_tolerance<=0||settings.hsl_pivot_tolerance>0.5||
         !std::isfinite(settings.hsl_memory_factor)||settings.hsl_memory_factor<=1||settings.hsl_memory_factor>64)
        throw std::runtime_error("MA27 requires pivot tolerance in (0,0.5] and memory factor in (1,64]");
      return std::make_unique<Ma27Factor>(settings);
    }catch(const std::exception& e){message=e.what();return {};}
  }
#endif
  if (settings.linear_backend=="native") return std::make_unique<DenseLapackFactor>(settings);
#if defined(LAZYIPDCA_HAVE_LDL) && defined(LAZYIPDCA_HAVE_METIS)
  if (settings.linear_backend=="ldl") return std::make_unique<SuiteSparseLdlMetisFactor>(settings);
#endif
#if defined(LAZYIPDCA_HAVE_QDLDL) && defined(LAZYIPDCA_HAVE_METIS)
  if (settings.linear_backend=="qdldl") return std::make_unique<QdldlMetisFactor>(settings);
  if (settings.linear_backend=="cholesk") {
    if(!std::isfinite(settings.cholesk_relative_pivot_tolerance) ||
       settings.cholesk_relative_pivot_tolerance<0.0) {
      message="Cholesk relative pivot tolerance must be finite and nonnegative";
      return {};
    }
    return std::make_unique<CholeskFactor>(settings);
  }
#endif
#ifdef LAZYIPDCA_HAVE_MUMPS
  if (settings.linear_backend=="mumps") return std::make_unique<MumpsMetisFactor>(settings);
#endif
  message="linear backend '"+settings.linear_backend+"' is not compiled; available: "+compiled_linear_backends();
  return {};
}

std::unique_ptr<SymmetricFactorization>
make_symmetric_factorization(const Settings& settings,std::string& message) {
  return trace_native_restoration_factor(make_symmetric_factorization_untraced(settings,message));
}

} // namespace lazyipdca::qcqp
