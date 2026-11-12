#pragma once

#include "lazyipdca_qcqp/linear_solver.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lazyipdca::qcqp::detail {

// An inequality keeps its border identity after its first promotion. Numeric
// classification is still free to change at every anchor: a condensed or
// inactive assigned row becomes an independent -1 pivot. Reserved slots are
// also independent -1 pivots. Consequently they do not change the Schur
// operator, and each contributes exactly one negative eigenvalue.
//
// This is an amortized envelope, not a claim of one symbolic analysis per
// solve. The first promotion of an inequality adds its actual support; new
// support entries and capacity growth must invalidate backend symbolic state.
// We never reserve coupling edges for every inequality, nor condense dense
// gradients here. The caller must retain the sparse condensed envelope and
// enable preservation of structural zeros in its factorization backend.
class DcaStableBorderSlots {
 public:
  void configure(int primal,int equalities,int inequalities,double reserve_factor=1.25) {
    if(primal<0 || equalities<0 || inequalities<0 ||
       primal>std::numeric_limits<int>::max()-equalities ||
       primal+equalities>std::numeric_limits<int>::max()-inequalities)
      throw std::invalid_argument("invalid persistent border dimensions");
    if(!std::isfinite(reserve_factor) || reserve_factor<1.0)
      throw std::invalid_argument("persistent border reserve must be finite and >= 1");
    reserve_factor_=reserve_factor;
    if(configured_ && primal_==primal && equalities_==equalities &&
       original_count_==inequalities)return;
    primal_=primal;equalities_=equalities;original_count_=inequalities;
    row_to_slot_.assign(static_cast<std::size_t>(inequalities),-1);
    requested_.assign(static_cast<std::size_t>(inequalities),0);
    slots_.clear();capacity_=0;configured_=true;++generation_;
  }

  // Duplicate IDs are harmless; assignment order is canonical even when the
  // caller traverses rows in a different order. Slots are never recycled.
  void update(const std::vector<int>& requested_rows) {
    if(!configured_)throw std::logic_error("persistent border is not configured");
    for(int original:requested_rows)
      if(original<0 || original>=original_count_)
        throw std::out_of_range("persistent border original row");
    std::fill(requested_.begin(),requested_.end(),0);
    for(int original:requested_rows)requested_[original]=1;
    for(int original=0;original<original_count_;++original) {
      if(!requested_[original] || row_to_slot_[original]>=0)continue;
      row_to_slot_[original]=static_cast<int>(slots_.size());
      slots_.push_back(Slot{original,{},{},{}});
      ++generation_;
    }
    const int required=static_cast<int>(slots_.size());
    if(required>capacity_) {
      // Headroom is bounded by the original inequality count, and only new
      // independent dummy diagonals are introduced for the spare capacity.
      const double desired=std::ceil(reserve_factor_*static_cast<double>(required))+64.0;
      capacity_=desired>=static_cast<double>(original_count_)
          ? original_count_:static_cast<int>(desired);
      ++generation_;
    }
  }

  int row_for(int original)const {
    check_original(original);
    return row_to_slot_[original]<0?-1:base_dimension()+row_to_slot_[original];
  }
  int active_row_for(int original)const {
    check_original(original);
    return requested_[original]?row_for(original):-1;
  }
  int base_dimension()const{return primal_+equalities_;}
  int dimension()const{return base_dimension()+capacity_;}
  int capacity()const{return capacity_;}
  int assigned_count()const{return static_cast<int>(slots_.size());}
  std::uint64_t generation()const{return generation_;}

  // Rows have .support and .values vectors (Workspace::jr satisfies this
  // contract). Values may change without a graph rebuild. Support unions are
  // retained even if the row later becomes condensed/inactive or its current
  // gradient support shrinks; missing coordinates are emitted as exact zeros.
  template<class Rows,class Scalars>
  void append_borders(const Rows& rows,const Scalars& s,const Scalars& z,
                      std::vector<SymmetricTriplet>& matrix) {
    if(static_cast<int>(rows.size())!=original_count_ ||
       static_cast<int>(s.size())!=original_count_ ||
       static_cast<int>(z.size())!=original_count_)
      throw std::invalid_argument("persistent border input dimensions");
    for(std::size_t k=0;k<slots_.size();++k) {
      Slot& slot=slots_[k];const int original=slot.original;
      const auto& gradient=rows[original];
      synchronize_support(slot,gradient.support,gradient.values.size());
      const int row=base_dimension()+static_cast<int>(k);
      const bool active=requested_[original]!=0;
      matrix.push_back({row,row,active?-s[original]/z[original]:-1.0});
      for(std::size_t j=0;j<slot.support.size();++j) {
        const int source=slot.source_offsets[j];
        matrix.push_back({slot.support[j],row,
            active && source>=0?gradient.values[source]:0.0});
      }
    }
    for(int k=static_cast<int>(slots_.size());k<capacity_;++k) {
      const int row=base_dimension()+k;matrix.push_back({row,row,-1.0});
    }
  }

 private:
  struct Slot {
    int original;
    std::vector<int> support,current_support,source_offsets;
  };
  void check_original(int original)const {
    if(original<0 || original>=original_count_)
      throw std::out_of_range("persistent border original row");
  }
  void synchronize_support(Slot& slot,const std::vector<int>& support,
                           std::size_t values_size) {
    if(support.size()!=values_size)
      throw std::invalid_argument("persistent border gradient dimensions");
    if(slot.current_support==support)return;
    for(int column:support)
      if(column<0 || column>=primal_)
        throw std::out_of_range("persistent border gradient column");
    // Dense gradients can contain tens of thousands of entries. Build this
    // index only when the support changes; repeated numeric preparation
    // takes the equality fast path above. Preserve first-seen emission order.
    std::unordered_map<int,std::size_t> positions;
    positions.reserve(slot.support.size()+support.size());
    for(std::size_t j=0;j<slot.support.size();++j)
      positions.emplace(slot.support[j],j);
    slot.source_offsets.assign(slot.support.size(),-1);
    for(std::size_t j=0;j<support.size();++j) {
      const auto inserted=positions.emplace(support[j],slot.support.size());
      if(inserted.second) {
        slot.support.push_back(support[j]);
        slot.source_offsets.push_back(static_cast<int>(j));++generation_;
      }else {
        const std::size_t position=inserted.first->second;
        if(slot.source_offsets[position]>=0)
          throw std::invalid_argument("persistent border duplicate gradient column");
        slot.source_offsets[position]=static_cast<int>(j);
      }
    }
    slot.current_support=support;
  }
  bool configured_=false;
  int primal_=0,equalities_=0,original_count_=0,capacity_=0;
  double reserve_factor_=1.25;
  std::uint64_t generation_=0;
  std::vector<int> row_to_slot_;
  std::vector<unsigned char> requested_;
  std::vector<Slot> slots_;
};

} // namespace lazyipdca::qcqp::detail
