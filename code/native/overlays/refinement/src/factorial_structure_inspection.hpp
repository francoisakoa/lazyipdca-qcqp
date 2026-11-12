#pragma once
#include "factorial_trace.hpp"
#include "native_signed_square_reduction.hpp"
#include "closed_zero_face.hpp"
#include "exact_redundant_equalities.hpp"
#include "affine_face_quadratic_rewrite.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include <chrono>
#include <set>
namespace lazyipdca::qcqp::detail {
struct FactorialStructureInspection {
  bool reduction_changed=false,unresolved=false,has_signed_square=false;
  bool retain_original_qcqp=false;
  std::size_t potential_shapes=0,models=0,rank_models=0,exact_phase_models=0;
  std::vector<NativeSignedSquareReduction::Link> original_links;
  std::string reason;
  std::set<std::string> seen;
  double deadline=0;
  static double now(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  static void number(FactorialSHA256& h,double d){std::uint64_t u;std::memcpy(&u,&d,sizeof(u));h.word(u);}
  static std::string signature(const QCQPProblem& p,bool rank,bool structural,bool rewrite) {
    FactorialSHA256 h;h.word(p.n);h.word(rank);h.word(structural);h.word(rewrite);h.vector(p.x0);h.vector(p.lb);h.vector(p.ub);
    auto row=[&](const QuadraticFunction& q){number(h,q.constant);std::uint64_t n=0;q.for_each_linear_nonzero([&](int,double){++n;});h.word(n);q.for_each_linear_nonzero([&](int i,double v){h.word(i);number(h,v);});h.word(q.hessian.entries.size());for(auto e:q.hessian.entries){h.word(e.i);h.word(e.j);number(h,e.v);}h.word(q.signed_square.size());for(auto e:q.signed_square){h.word(e.first);number(h,e.second);}};
    row(p.objective);h.word(p.equalities.size());for(const auto& q:p.equalities)row(q);h.word(p.inequalities.size());for(const auto& q:p.inequalities)row(q);return h.finish();
  }
  void inspect(const QCQPProblem& p,bool rank,bool structural,bool rewrite,int depth) {
    if(now()>=deadline||depth>12||models>=64){unresolved=true;reason="structural-inspection-work-limit";return;}
    if(!seen.insert(signature(p,rank,structural,rewrite)).second)return;
    ++models;has_signed_square=has_signed_square||p.has_signed_square();
    potential_shapes+=NativeSignedSquareReduction::potential_link_shapes(p);
    if(retain_original_qcqp) {
      // The primary experiment disables this nonlinear route by construction.
      // Candidate shapes are reported, but cannot change the supplied QCQP or
      // any of its exact quadratic/affine reductions in this intervention.
      if(models==1)reason="primary-original-QCQP-native-signed-square-route-disabled";
    } else {
      const NativeSignedSquareReduction r(p);reduction_changed=reduction_changed||r.changed();
      if(models==1){original_links=r.links;reason=r.reason;}
    }
    // With no quadratic source equality, exact affine substitutions and row
    // deletion cannot create one. This covers regime I without optimizer work.
    const bool quadratic_eq=std::any_of(p.equalities.begin(),p.equalities.end(),[](const auto& q){return !q.signed_square.empty()||std::any_of(q.hessian.entries.begin(),q.hessian.entries.end(),[](auto e){return e.v!=0;});});
    if(!quadratic_eq)return;
    if(structural) {
      ClosedZeroFace face(p,deadline);
      if(face.reason=="deadline"){unresolved=true;reason="zero-face-inspection-deadline";return;}
      const auto& phase=face.applied()?face.model:p;
      ExactRedundantEqualities rows(phase,1e-6,deadline);
      if(rows.reason.find("deadline")!=std::string::npos){unresolved=true;reason="exact-row-inspection-deadline";return;}
      if(face.applied()||rows.changed()) {
        ++exact_phase_models;inspect(rows.reduced,rank,true,rewrite,depth+1);
      }
      if(!rewrite) {
        AffineFaceQuadraticRewrite affine(p,deadline);
        if(now()>=deadline){unresolved=true;reason="affine-rewrite-inspection-deadline";return;}
        if(affine.applied()){++exact_phase_models;inspect(affine.model,rank,true,true,depth+1);}
      }
    }
    if(rank) {
      // Both globalization columns and both possible exact-fixed attempts.
      // These are actual native coefficient reductions, never numerical solves.
      for(bool hs:{false,true})for(bool exact_fixed:{false,true}) {
        if(retain_original_qcqp&&hs)continue; // common_r3_guards_v1, both G columns
        if(now()>=deadline){unresolved=true;reason="rank-inspection-deadline";return;}
        ConstraintPresolve reduced(p,true,hs,exact_fixed,!hs);
        if(exact_fixed&&reduced.affine_bounds().empty())continue;
        if(reduced.changed()){++rank_models;inspect(reduced.reduced(),false,false,rewrite,depth+1);}
      }
    }
  }
  explicit FactorialStructureInspection(const QCQPProblem& p,bool original_qcqp=false):retain_original_qcqp(original_qcqp) {
    deadline=now()+300;
    try{inspect(p,true,true,false,0);}catch(const std::exception& e){unresolved=true;reason=e.what();}
    // Any candidate shape remains a conservative unresolved case, even if
    // current bounds/start/compatibility keep the raw reduction inactive.
    if(!retain_original_qcqp&&potential_shapes>0)unresolved=true;
  }
  bool eligible()const{return !reduction_changed&&!unresolved&&!has_signed_square;}
};
}
