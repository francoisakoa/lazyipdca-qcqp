#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace lazyipdca::qcqp::study {
struct SparsePsdBudget {
  std::size_t max_entries=2000000,max_updates=30000000;
  int max_dimension=200000;
  double max_seconds=5;
};
struct SparsePsdProof {
  bool certified=false;
  std::string reason="not-started";
  int positive_pivots=0,exact_zero_rows=0,precision_bits=0;
  std::string first_stage_reason;
  std::size_t updates=0,peak_entries=0;
  double seconds=0;
  long double undecided_pivot_lower=0,undecided_pivot_upper=0;
  long double minimum_pivot_lower=std::numeric_limits<long double>::infinity();
};
namespace sparse_psd_detail {
template<class Float> struct Bounds {Float lo=0,hi=0;};
template<class Float> inline bool number_finite(const Float& value){using std::isfinite;return isfinite(value);}
template<class Float> inline bool finite(const Bounds<Float>& a){return number_finite(a.lo)&&number_finite(a.hi)&&a.lo<=a.hi;}
template<class Float> inline bool zero(const Bounds<Float>& a){return a.lo==0&&a.hi==0;}
template<class Float> inline Float down(const Float& x){using std::nextafter;return nextafter(x,-std::numeric_limits<Float>::infinity());}
template<class Float> inline Float up(const Float& x){using std::nextafter;return nextafter(x,std::numeric_limits<Float>::infinity());}
template<class Float> inline Float rounded(const Float& value){return value;}
template<> inline long double rounded(const long double& value){volatile long double stored=value;return stored;}
template<class F> inline Bounds<F> add(const Bounds<F>& a,const Bounds<F>& b){
  if(zero(a))return b;if(zero(b))return a;
  const F lo=rounded(F(a.lo+b.lo)),hi=rounded(F(a.hi+b.hi));return {down(lo),up(hi)};
}
template<class F> inline Bounds<F> subtract(const Bounds<F>& a,const Bounds<F>& b){
  if(zero(b))return a;
  const F lo=rounded(F(a.lo-b.hi)),hi=rounded(F(a.hi-b.lo));return {down(lo),up(hi)};
}
template<class F> inline Bounds<F> multiply(const Bounds<F>& a,const Bounds<F>& b){
  if(zero(a)||zero(b))return {};
  const F p0=rounded(F(a.lo*b.lo)),p1=rounded(F(a.lo*b.hi)),p2=rounded(F(a.hi*b.lo)),p3=rounded(F(a.hi*b.hi));
  return {down(std::min({p0,p1,p2,p3})),up(std::max({p0,p1,p2,p3}))};
}
template<class F> inline Bounds<F> divide_positive(const Bounds<F>& a,const Bounds<F>& b){
  if(zero(a))return {};
  const F p0=rounded(F(a.lo/b.lo)),p1=rounded(F(a.lo/b.hi)),p2=rounded(F(a.hi/b.lo)),p3=rounded(F(a.hi/b.hi));
  return {down(std::min({p0,p1,p2,p3})),up(std::max({p0,p1,p2,p3}))};
}
} // namespace sparse_psd_detail

// Sparse interval Schur elimination with a symmetric minimum-degree pivot
// ordering. A floating factorization/inertia is never used as a certificate.
// Every input binary64 coefficient is represented exactly in long double;
// each arithmetic operation is widened by one adjacent long-double value.
// Positive pivot lower endpoints prove SPD by Schur-complement induction.
template<class Float,class Entry,class Convert> inline SparsePsdProof verified_sparse_psd_initial(int n,const std::vector<Entry>& input,
                                         const Convert& convert,const SparsePsdBudget& budget={}) {
  using namespace sparse_psd_detail;using Clock=std::chrono::steady_clock;
  using Interval=Bounds<Float>;
  const auto started=Clock::now();SparsePsdProof proof;proof.precision_bits=std::numeric_limits<Float>::digits;
  auto finish=[&](const char* reason,bool certified=false){proof.reason=reason;proof.certified=certified;proof.seconds=std::chrono::duration<double>(Clock::now()-started).count();return proof;};
#if defined(__FAST_MATH__) || defined(__ASSOCIATIVE_MATH__) || defined(__RECIPROCAL_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__>0)
  return finish("unsafe-floating-point-build");
#endif
  if(std::numeric_limits<Float>::radix!=2||
     std::numeric_limits<Float>::digits<std::numeric_limits<double>::digits||
     std::numeric_limits<Float>::max_exponent<std::numeric_limits<double>::max_exponent||
     std::numeric_limits<Float>::min_exponent>std::numeric_limits<double>::min_exponent)
    return finish("unsupported-floating-point-format");
  if(n<0||n>budget.max_dimension||!(budget.max_seconds>0)||!std::isfinite(budget.max_seconds))return finish("dimension-or-time-budget");
  using Row=std::unordered_map<int,Interval>;
  std::vector<Row> rows(n);std::vector<unsigned char> active(n,1);
  std::size_t entries=0;
  auto set=[&](int i,int j,const Interval& value){
    if(!rows[i].count(j)){if(entries>=budget.max_entries)return false;++entries;}
    rows[i][j]=value;proof.peak_entries=std::max(proof.peak_entries,entries);return true;
  };
  std::size_t input_count=0;
  for(const auto& e:input) {
    if((++input_count&4095)==0&&std::chrono::duration<double>(Clock::now()-started).count()>budget.max_seconds)return finish("time-budget");
    if(e.i<0||e.j<0||e.i>=n||e.j>=n)return finish("invalid-input");
    Interval value=convert(e);if(!finite(value))return finish("invalid-input");if(zero(value))continue;
    const auto it=rows[e.i].find(e.j);
    if(it!=rows[e.i].end())value=add(it->second,value);
    if(!finite(value))return finish("nonfinite-input-sum");
    if(!set(e.i,e.j,value)||(e.i!=e.j&&!set(e.j,e.i,value)))return finish("entry-budget");
  }
  using Key=std::pair<std::size_t,int>;
  std::priority_queue<Key,std::vector<Key>,std::greater<Key>> queue;
  for(int i=0;i<n;++i) {
    if(queue.size()>=budget.max_entries)return finish("ordering-entry-budget");
    if((i&4095)==0&&std::chrono::duration<double>(Clock::now()-started).count()>budget.max_seconds)return finish("time-budget");
    queue.push({rows[i].size()-(rows[i].count(i)?1:0),i});
  }
  for(int eliminated=0;eliminated<n;++eliminated) {
    if(std::chrono::duration<double>(Clock::now()-started).count()>budget.max_seconds)return finish("time-budget");
    while(!queue.empty()) {
      const auto [degree,k]=queue.top();
      const auto current=rows[k].size()-(rows[k].count(k)?1:0);
      if(!active[k]||degree!=current)queue.pop();else break;
    }
    if(queue.empty())return finish("ordering-exhausted");
    const int k=queue.top().second;queue.pop();
    std::vector<std::pair<int,Interval>> column;
    for(const auto& item:rows[k])if(item.first!=k&&!zero(item.second))column.push_back(item);
    std::sort(column.begin(),column.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    auto diagonal=rows[k].find(k);const Interval pivot=diagonal==rows[k].end()?Interval{}:diagonal->second;
    if(zero(pivot)&&column.empty()){++proof.exact_zero_rows;active[k]=0;entries-=rows[k].size();rows[k].clear();continue;}
    if(!finite(pivot)||!(pivot.lo>0)){proof.undecided_pivot_lower=static_cast<long double>(pivot.lo);proof.undecided_pivot_upper=static_cast<long double>(pivot.hi);return finish("pivot-not-strictly-positive");}
    ++proof.positive_pivots;proof.minimum_pivot_lower=std::min(proof.minimum_pivot_lower,std::nextafter(static_cast<long double>(pivot.lo),-std::numeric_limits<long double>::infinity()));
    std::vector<Interval> ratios;ratios.reserve(column.size());
    for(const auto& a:column){auto value=divide_positive(a.second,pivot);if(!finite(value))return finish("nonfinite-ratio");ratios.push_back(value);}
    for(std::size_t a=0;a<column.size();++a)for(std::size_t b=a;b<column.size();++b) {
      if(++proof.updates>budget.max_updates)return finish("operation-budget");
      if((proof.updates&4095)==0&&std::chrono::duration<double>(Clock::now()-started).count()>budget.max_seconds)return finish("time-budget");
      const int i=column[a].first,j=column[b].first;
      const auto found=rows[i].find(j);const Interval old=found==rows[i].end()?Interval{}:found->second;
      const Interval value=subtract(old,multiply(ratios[a],column[b].second));
      if(!finite(value))return finish("nonfinite-schur-update");
      if(!set(i,j,value)||(i!=j&&!set(j,i,value)))return finish("entry-budget");
    }
    active[k]=0;entries-=rows[k].size();rows[k].clear();
    for(const auto& a:column) {
      const int i=a.first;entries-=rows[i].erase(k);
      if(queue.size()>=budget.max_entries)return finish("ordering-entry-budget");
      queue.push({rows[i].size()-(rows[i].count(i)?1:0),i});
    }
  }
  return finish("positive-interval-LDL-pivots",true);
}
template<class Float> struct SparseIntervalTriplet {int i=0,j=0;sparse_psd_detail::Bounds<Float> value;};
template<class Float> inline SparsePsdProof verified_sparse_psd_intervals(int n,
    const std::vector<SparseIntervalTriplet<Float>>& input,const SparsePsdBudget& budget={}) {
  return verified_sparse_psd_initial<Float>(n,input,[](const auto& e){return e.value;},budget);
}
template<class Float> inline SparsePsdProof verified_sparse_psd_precision(int n,
    const std::vector<SymTriplet>& input,const SparsePsdBudget& budget={}) {
  return verified_sparse_psd_initial<Float>(n,input,[](const auto& e){return sparse_psd_detail::Bounds<Float>{Float(e.v),Float(e.v)};},budget);
}
// Retry only numerical interval ambiguity in the same bounded computation.
// Software binary113 has round-to-nearest arithmetic and adjacent-value
// nextafter; expression templates are disabled in cpp_bin_float_quad.
inline SparsePsdProof verified_sparse_psd(int n,const std::vector<SymTriplet>& input,
                                         const SparsePsdBudget& budget={}) {
  auto first=verified_sparse_psd_precision<long double>(n,input,budget);
  if(first.certified||first.reason!="pivot-not-strictly-positive"||
     !(first.undecided_pivot_lower<=0&&first.undecided_pivot_upper>0)||
     first.seconds>=budget.max_seconds)return first;
  auto remaining=budget;remaining.max_seconds-=first.seconds;remaining.max_updates-=first.updates;
  auto second=verified_sparse_psd_precision<boost::multiprecision::cpp_bin_float_quad>(n,input,remaining);
  second.first_stage_reason=first.reason;second.seconds+=first.seconds;
  second.updates+=first.updates;second.peak_entries=std::max(second.peak_entries,first.peak_entries);
  return second;
}
} // namespace lazyipdca::qcqp::study
