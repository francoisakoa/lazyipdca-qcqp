#pragma once
#include "trial_soc_arc_policy.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <map>
#include <string>
#include <vector>

namespace lazyipdca::qcqp::detail {
struct SocDefectResiduals {
  long double equality=0,slack=0;
  bool valid=true;
  void add_equality(long double value,double scale) {
    if(!std::isfinite(value)||!std::isfinite(scale)||!(scale>0)){valid=false;return;}
    const auto v=std::abs(value)/scale;
    if(!std::isfinite(v)){valid=false;return;}
    equality=std::max(equality,v);
  }
  void add_slack(long double value,double scale) {
    if(!std::isfinite(value)||!std::isfinite(scale)||!(scale>0)){valid=false;return;}
    const auto v=std::abs(value)/scale;
    if(!std::isfinite(v)){valid=false;return;}
    slack=std::max(slack,v);
  }
};
template<class WorkspaceType>
SocDefectResiduals soc_defect_anchor_residuals(const WorkspaceType& w) {
  SocDefectResiduals out;
  for(int e=0;e<w.me;++e)out.add_equality(w.equalities[e].value,w.equalities[e].scale);
  for(int i=0;i<w.mi;++i)if(w.active[i])out.add_slack(
      static_cast<long double>(w.inequalities[i].value)+w.s[i],w.inequalities[i].scale);
  return out;
}
template<class WorkspaceType,class DirectionType>
SocDefectResiduals soc_defect_trial_residuals(const WorkspaceType& w,
    const DirectionType& d,double alpha) {
  SocDefectResiduals out;
  if(!trial_soc_arc_finite_update(w,d,alpha)){out.valid=false;return out;}
  auto x=w.x,s=w.s;
  for(std::size_t j=0;j<x.size();++j)x[j]+=alpha*d.x[j];
  for(std::size_t i=0;i<s.size();++i)s[i]+=alpha*d.s[i];
  for(int e=0;e<w.me;++e)out.add_equality(w.equalities[e].original_at(x),w.equalities[e].scale);
  for(int i=0;i<w.mi;++i)if(w.active[i])out.add_slack(
      static_cast<long double>(w.inequalities[i].original_at(x))+s[i],w.inequalities[i].scale);
  return out;
}
inline bool soc_defect_within(long double value,long double anchor,double tolerance) {
  const long double limit=std::max(anchor,static_cast<long double>(tolerance));
  const long double floor=64*std::numeric_limits<double>::epsilon()*std::max({1.0L,value,limit});
  return std::isfinite(value)&&std::isfinite(limit)&&value>=0&&value-limit<=floor;
}

struct SocDefectProjection {
  bool valid=false;
  const char* reason="unprepared-geometry";
  long double correction_inf=0;
  int block_count=0,leaf_count=0;
};

// Coefficient-only certificate of an optional SOURCE-level-set retraction.
// No KKT pivots, elimination maps, problem names, row-count patterns or model
// Hessians enter detection. The original diagonal equalities must form
// disjoint definite blocks. Every remaining affine equality needs its OWN
// unbounded raw unit-coefficient leaf, occurring in no other affine row.
// Only O(n + equalities + nonlinear diagonal entries) metadata is retained;
// dense affine coefficients are borrowed from their original rows at use.
struct SocDefectGeometry {
  struct Term {int coordinate=0;long double weight=0,center=0;};
  struct Block {int equality=0,sign=1;std::vector<Term> terms;};
  struct Leaf {int equality=0,pivot=0;double coefficient=0;};
  bool supported=false;
  std::string reason="not-inspected";
  int dimension=0,equality_count=0,inequality_count=0;
  std::vector<Block> blocks;
  std::vector<Leaf> leaves;

  template<class WorkspaceType>
  bool inspect(const WorkspaceType& w) {
    *this=SocDefectGeometry{};
    dimension=w.n;equality_count=w.me;inequality_count=w.mi;
    auto decline=[&](const char* why){reason=why;return false;};
    if(w.n<=0||w.me<=0)return decline("no-equality-geometry");
    std::vector<int> owner(w.n,-1),affine_occurrences(w.n,0),affine;
    for(int e=0;e<w.me;++e) {
      const auto& q=w.equalities[e];
      if(!q.signed_square.empty())return decline("native-signed-square-equality");
      if(!std::isfinite(q.constant)||!std::isfinite(q.scale)||!(q.scale>0))
        return decline("invalid-original-row-coefficients");
      // Workspace reserves exact-zero sparse Hessian slots. They are
      // structural metadata, not curvature or nonlinear variable support.
      std::map<int,long double> diagonal;
      for(const auto& entry:q.hessian) {
        if(!std::isfinite(entry.v))return decline("nonfinite-original-hessian");
        if(entry.v==0)continue;
        if(entry.i!=entry.j||entry.i<0||entry.i>=w.n)
          return decline("nondefinite-diagonal-equality");
        diagonal[entry.i]+=entry.v;
      }
      Block block;block.equality=e;
      for(const auto& entry:diagonal) {
        if(entry.second==0)continue;
        const int sign=entry.second>0?1:-1;
        if(block.terms.empty())block.sign=sign;
        if(sign!=block.sign)return decline("indefinite-diagonal-equality");
        if(owner[entry.first]!=-1)return decline("overlapping-nonlinear-equality-blocks");
        owner[entry.first]=e;
        long double linear=0;
        for(const auto& term:q.linear)if(term.first==entry.first)linear+=term.second;
        const long double center=-linear/entry.second;
        const long double weight=.5L*std::abs(entry.second);
        if(!std::isfinite(center)||!std::isfinite(weight)||!(weight>0))
          return decline("nonfinite-completed-square");
        block.terms.push_back({entry.first,weight,center});
      }
      if(block.terms.empty()){affine.push_back(e);continue;}
      for(const auto& entry:q.linear) {
        if(!std::isfinite(entry.second))return decline("nonfinite-original-linear-term");
        if(entry.second==0)continue;
        if(entry.first<0||entry.first>=w.n||owner[entry.first]!=e)
          return decline("linear-term-outside-definite-block");
      }
      blocks.push_back(std::move(block));
    }
    if(blocks.empty())return decline("no-nonlinear-equality-block");
    for(int e:affine)for(const auto& entry:w.equalities[e].linear) {
      if(entry.first<0||entry.first>=w.n||!std::isfinite(entry.second))
        return decline("invalid-affine-coefficient");
      if(entry.second!=0)++affine_occurrences[entry.first];
    }
    for(int e:affine) {
      const auto& q=w.equalities[e];Leaf leaf;leaf.equality=e;bool found=false;
      for(const auto& entry:q.linear) {
        const int j=entry.first;
        if(owner[j]>=0||affine_occurrences[j]!=1||!(entry.second==1||entry.second==-1))continue;
        const int original=w.free[j];
        if(original<0||static_cast<std::size_t>(original)>=w.original.lb.size()||
            static_cast<std::size_t>(original)>=w.original.ub.size())return decline("invalid-original-bound-map");
        if(w.original.lb[original]!=-std::numeric_limits<double>::infinity()||
            w.original.ub[original]!=std::numeric_limits<double>::infinity())continue;
        leaf.pivot=j;leaf.coefficient=entry.second;found=true;break;
      }
      if(!found)return decline("affine-row-without-independent-unbounded-unit-leaf");
      leaves.push_back(leaf);
    }
    supported=true;reason="verified-disjoint-definite-diagonal-blocks-and-independent-affine-leaves";
    return true;
  }

  template<class WorkspaceType,class DirectionType,class PathType>
  SocDefectProjection project(const WorkspaceType& w,const DirectionType& source,
      const PathType& path,double alpha,DirectionType& combined)const {
    SocDefectProjection out;out.block_count=static_cast<int>(blocks.size());
    out.leaf_count=static_cast<int>(leaves.size());
    auto decline=[&](const char* why){out.reason=why;return out;};
    if(!supported||w.n!=dimension||w.me!=equality_count||w.mi!=inequality_count)
      return decline("unprepared-or-changed-geometry");
    if(!trial_soc_arc_finite_update(w,source,alpha))return decline("invalid-source-update");
    auto projected=w.x,source_x=w.x,slacks=w.s;
    for(int j=0;j<w.n;++j)projected[j]=source_x[j]=w.x[j]+alpha*source.x[j];
    for(int i=0;i<w.mi;++i)slacks[i]=w.s[i]+alpha*source.s[i];
    for(const auto& block:blocks) {
      const auto& q=w.equalities[block.equality];
      if(!std::isfinite(q.scale)||!(q.scale>0))return decline("invalid-original-row-scale");
      long double anchor_radius=0,source_radius=0;
      for(const auto& term:block.terms) {
        const long double a=static_cast<long double>(w.x[term.coordinate])-term.center;
        const long double b=static_cast<long double>(source_x[term.coordinate])-term.center;
        anchor_radius+=term.weight*a*a;source_radius+=term.weight*b*b;
      }
      // The target is h(anchor)+alpha*J_h(anchor)*p, INCLUDING the actual
      // source linear-solve residual, never an idealized (1-alpha)*h(anchor).
      // Constants cancel analytically in this completed-square radius update.
      const long double target_radius=anchor_radius+block.sign*static_cast<long double>(alpha)*
          path.equalities[block.equality].linear/q.scale;
      if(!(anchor_radius>0)||!(source_radius>0)||!std::isfinite(anchor_radius)||
          !std::isfinite(source_radius)||!std::isfinite(target_radius)||target_radius<0)
        return decline("invalid-or-singular-radial-target");
      const long double source_root=std::sqrt(source_radius),target_root=std::sqrt(target_radius);
      const long double delta_factor=(target_radius-source_radius)/
          (source_root*(target_root+source_root));
      if(!std::isfinite(delta_factor))return decline("nonfinite-radial-factor");
      for(const auto& term:block.terms) {
        // source+(factor-1)*(source-center) avoids subtracting a huge center
        // from itself when a shifted block needs only a tiny correction.
        const long double value=static_cast<long double>(source_x[term.coordinate])+
            delta_factor*(static_cast<long double>(source_x[term.coordinate])-term.center);
        projected[term.coordinate]=static_cast<double>(value);
        if(!std::isfinite(projected[term.coordinate]))return decline("nonfinite-radial-coordinate");
      }
    }
    for(const auto& leaf:leaves) {
      const auto& q=w.equalities[leaf.equality];
      long double other=q.constant;
      for(const auto& entry:q.linear)if(entry.first!=leaf.pivot)
        other+=static_cast<long double>(entry.second)*projected[entry.first];
      const long double target=path.equalities[leaf.equality].constant+
          static_cast<long double>(alpha)*path.equalities[leaf.equality].linear;
      const long double value=(target/q.scale-other)/leaf.coefficient;
      projected[leaf.pivot]=static_cast<double>(value);
      if(!std::isfinite(projected[leaf.pivot]))return decline("nonfinite-affine-leaf-coordinate");
    }
    for(int i=0;i<w.mi;++i)if(w.active[i]) {
      const long double target=path.inequalities[i].constant+w.s[i]+
          static_cast<long double>(alpha)*(path.inequalities[i].linear+source.s[i]);
      slacks[i]=static_cast<double>(target-w.inequalities[i].original_at(projected));
      if(!std::isfinite(slacks[i]))return decline("nonfinite-affine-slack-transport");
    }
    combined=source; // z and lambda retain the PURE SOURCE update.
    for(int j=0;j<w.n;++j)combined.x[j]=(projected[j]-w.x[j])/alpha;
    for(int i=0;i<w.mi;++i)combined.s[i]=(slacks[i]-w.s[i])/alpha;
    if(!trial_soc_arc_finite_update(w,combined,alpha))return decline("nonfinite-realized-retraction");
    for(int j=0;j<w.n;++j)out.correction_inf=std::max(out.correction_inf,
        std::abs(static_cast<long double>(w.x[j]+alpha*combined.x[j])-source_x[j]));
    out.valid=std::isfinite(out.correction_inf);out.reason=out.valid?"coefficient-source-levelset-proposal":"nonfinite-realized-size";
    return out;
  }
};
} // namespace lazyipdca::qcqp::detail
