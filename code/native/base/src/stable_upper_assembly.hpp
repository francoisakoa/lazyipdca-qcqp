#pragma once
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace lazyipdca::qcqp::detail {
// Compile the caller's triplet layout once. Numeric reuse preserves the exact
// duplicate accumulation sequence of stable counting-sort assembly, including
// fixed diagonal shifts appended after every caller entry. The output pattern
// still uses the backend's existing zero policy: this cache changes neither
// its symbolic envelope nor its ordering or factorization parameters.
class StableUpperAssemblyCache {
 public:
  template<class Entry>
  bool assemble(int n,const std::vector<SymmetricTriplet>& input,int primal,
                double primal_reg,double dual_reg,bool preserve_zeros,
                std::vector<Entry>& output) {
    last_reused_=false;
    const bool fixed=primal>=0;
    if(n<0||input.size()>std::numeric_limits<std::size_t>::max()-
          (fixed?static_cast<std::size_t>(n):0))return unavailable();
    const std::size_t count=input.size()+(fixed?static_cast<std::size_t>(n):0);
    // The historical assembler uses an unstable comparison sort outside
    // this range; leave that path untouched rather than change roundoff.
    const std::size_t ns=static_cast<std::size_t>(n);
    if(count>=2&&!(n>0&&ns<=1000000u&&ns<=std::max<std::size_t>(4096u,8u*count)))
      return unavailable();
    bool matches=valid_&&n==n_&&fixed==fixed_&&input.size()==raw_count_;
    for(std::size_t k=0;k<input.size();++k) {
      const auto& e=input[k];
      if(e.i<0||e.j<0||e.i>=n||e.j>=n||!std::isfinite(e.value))return unavailable();
      if(matches&&keys_[k]!=key(e.i,e.j))matches=false;
    }
    if(!matches) {
      if(valid_)++invalidations_;
      n_=n;fixed_=fixed;raw_count_=input.size();
      keys_.resize(count);
      for(std::size_t k=0;k<input.size();++k)keys_[k]=key(input[k].i,input[k].j);
      if(fixed)for(int j=0;j<n;++j)keys_[input.size()+static_cast<std::size_t>(j)]=key(j,j);
      order_.resize(count);std::iota(order_.begin(),order_.end(),std::size_t(0));
      if(count>=2) {
        buffer_.resize(count);offsets_.assign(ns+1,0);
        for(const auto k:keys_)++offsets_[row(k)+1];
        for(std::size_t j=1;j<offsets_.size();++j)offsets_[j]+=offsets_[j-1];
        for(std::size_t k=0;k<count;++k)buffer_[offsets_[row(keys_[k])]++]=k;
        std::fill(offsets_.begin(),offsets_.end(),0);
        for(const auto k:buffer_)++offsets_[column(keys_[k])+1];
        for(std::size_t j=1;j<offsets_.size();++j)offsets_[j]+=offsets_[j-1];
        for(const auto k:buffer_)order_[offsets_[column(keys_[k])]++]=k;
      }
      unique_.clear();slots_.resize(count);
      for(const auto k:order_) {
        if(unique_.empty()||unique_.back()!=keys_[k])unique_.push_back(keys_[k]);
        slots_[k]=unique_.size()-1;
      }
      valid_=true;++builds_;
    }else{last_reused_=true;++reuses_;}
    values_.assign(unique_.size(),0.0);
    for(std::size_t k=0;k<input.size();++k)values_[slots_[k]]+=input[k].value;
    if(fixed)for(int j=0;j<n;++j)
      values_[slots_[input.size()+static_cast<std::size_t>(j)]]+=j<primal?primal_reg:-dual_reg;
    output.clear();output.reserve(unique_.size());
    pattern_.clear();pattern_.reserve(unique_.size());
    for(std::size_t k=0;k<unique_.size();++k) {
      const double value=values_[k];
      if(preserve_zeros||std::abs(value)>=1e-30) {
        output.push_back({static_cast<int>(row(unique_[k])),static_cast<int>(column(unique_[k])),value});
        pattern_.push_back(unique_[k]);
      }
    }
    return true;
  }
  const std::vector<std::uint64_t>& pattern() const{return pattern_;}
  int builds()const{return builds_;}
  int reuses()const{return reuses_;}
  int invalidations()const{return invalidations_;}
  bool last_reused()const{return last_reused_;}
 private:
  static std::uint64_t key(int i,int j) {
    if(i>j)std::swap(i,j);
    return(static_cast<std::uint64_t>(static_cast<std::uint32_t>(j))<<32)|static_cast<std::uint32_t>(i);
  }
  static std::size_t row(std::uint64_t k){return static_cast<std::uint32_t>(k);}
  static std::size_t column(std::uint64_t k){return static_cast<std::uint32_t>(k>>32);}
  bool unavailable(){if(valid_)++invalidations_;valid_=false;return false;}
  bool valid_=false,fixed_=false,last_reused_=false;
  int n_=0,builds_=0,reuses_=0,invalidations_=0;
  std::size_t raw_count_=0;
  std::vector<std::uint64_t> keys_,unique_,pattern_;
  std::vector<std::size_t> slots_,order_,buffer_,offsets_;
  Vec values_;
};
} // namespace lazyipdca::qcqp::detail
