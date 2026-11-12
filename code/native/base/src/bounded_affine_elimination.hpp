#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lazyipdca::qcqp::detail {

// Materializing T' H T stores one HessianMap for every contributing pair,
// not merely one for every distinct reduced KKT entry. A sparse affine lift
// can therefore turn a diagonal H into cubic-sized bookkeeping. Admit the
// substitution BEFORE constructing T, reduced Jacobians or the pair maps.
// Refusal retains the original variable and equality in the sparse KKT.
struct AffineEliminationBudget {
  std::vector<std::size_t> chosen_width;
  std::size_t original_entries=0, proposed_entries=0, admitted_entries=0;
  std::size_t budget_bytes=0, proposed_bytes=0, admitted_bytes=0;
  std::size_t candidates=0, refused=0;
  bool arithmetic_saturated=false;
};

inline std::size_t affine_saturating_add(std::size_t a,std::size_t b,bool& saturated) {
  if(b>std::numeric_limits<std::size_t>::max()-a) {
    saturated=true;return std::numeric_limits<std::size_t>::max();
  }
  return a+b;
}
inline std::size_t affine_saturating_multiply(std::size_t a,std::size_t b,bool& saturated) {
  if(a&&b>std::numeric_limits<std::size_t>::max()/a) {
    saturated=true;return std::numeric_limits<std::size_t>::max();
  }
  return a*b;
}
inline std::size_t affine_triangle(std::size_t width,bool& saturated) {
  if(width==std::numeric_limits<std::size_t>::max()) {
    saturated=true;return width;
  }
  const auto next=width+1;
  return width%2?affine_saturating_multiply(width,next/2,saturated):
      affine_saturating_multiply(width/2,next,saturated);
}

// pattern is the unique ORIGINAL Hessian structural support, including the
// reserved diagonal and small study-model completions. proposed_width[j]
// is one for a retained coordinate, and the affine dependency count for a
// proposed eliminated coordinate. The injectable budget is unit-test-only;
// no instance name, CLI tolerance or optimization data enters this policy.
inline AffineEliminationBudget bounded_affine_elimination(
    const std::vector<std::size_t>& proposed_width,
    const std::vector<std::pair<int,int>>& pattern,
    std::size_t bytes_per_map,std::size_t test_budget_bytes=0) {
  if(!bytes_per_map)throw std::invalid_argument("zero affine-elimination map size");
  AffineEliminationBudget out;
  const std::size_t n=proposed_width.size();
  out.original_entries=pattern.size();
  const std::size_t original_bytes=affine_saturating_multiply(
      out.original_entries,bytes_per_map,out.arithmetic_saturated);
  // A fixed cap prevents large inputs from granting unbounded fill, while
  // the sparse-size factor keeps ordinary useful reductions unchanged.
  constexpr std::size_t floor=8u*1024u*1024u,ceiling=512u*1024u*1024u;
  const auto scaled=affine_saturating_multiply(original_bytes,8,out.arithmetic_saturated);
  out.budget_bytes=test_budget_bytes?test_budget_bytes:
      std::min(ceiling,std::max(floor,scaled));
  const std::size_t max_entries=out.budget_bytes/bytes_per_map;
  std::vector<std::vector<std::size_t>> incident(n);
  auto pair_count=[&](const std::pair<int,int>& edge,const std::vector<std::size_t>& widths) {
    return edge.first==edge.second?affine_triangle(widths[edge.first],out.arithmetic_saturated):
        affine_saturating_multiply(widths[edge.first],widths[edge.second],out.arithmetic_saturated);
  };
  for(std::size_t k=0;k<pattern.size();++k) {
    const auto edge=pattern[k];
    if(edge.first<0||edge.second<edge.first||std::size_t(edge.second)>=n)
      throw std::invalid_argument("invalid affine-elimination Hessian support");
    incident[edge.first].push_back(k);
    if(edge.first!=edge.second)incident[edge.second].push_back(k);
    out.proposed_entries=affine_saturating_add(out.proposed_entries,
        pair_count(edge,proposed_width),out.arithmetic_saturated);
  }
  for(auto width:proposed_width)out.candidates+=width!=1;
  out.proposed_bytes=affine_saturating_multiply(
      out.proposed_entries,bytes_per_map,out.arithmetic_saturated);
  if(out.proposed_entries<std::numeric_limits<std::size_t>::max()&&out.proposed_entries<=max_entries) {
    out.chosen_width=proposed_width;out.admitted_entries=out.proposed_entries;
  } else {
    out.chosen_width.assign(n,1);out.admitted_entries=out.original_entries;
    // Constant substitutions cannot increase fill and must not be rejected
    // merely because another proposed pivot would be expensive.
    for(int pass=0;pass<2;++pass)for(std::size_t j=0;j<n;++j) {
      const auto width=proposed_width[j];
      if(width==1||(pass==0)!=(width==0))continue;
      std::size_t before=0,after=0;
      for(auto k:incident[j])before=affine_saturating_add(before,
          pair_count(pattern[k],out.chosen_width),out.arithmetic_saturated);
      out.chosen_width[j]=width;
      for(auto k:incident[j])after=affine_saturating_add(after,
          pair_count(pattern[k],out.chosen_width),out.arithmetic_saturated);
      const auto remainder=out.admitted_entries-before;
      const auto trial=affine_saturating_add(remainder,after,out.arithmetic_saturated);
      if(width==0||(trial<std::numeric_limits<std::size_t>::max()&&trial<=max_entries))out.admitted_entries=trial;
      else {out.chosen_width[j]=1;++out.refused;}
    }
  }
  out.admitted_bytes=affine_saturating_multiply(
      out.admitted_entries,bytes_per_map,out.arithmetic_saturated);
  return out;
}

} // namespace lazyipdca::qcqp::detail
