#pragma once
#include "model.hpp"
#include "result.hpp"
#include <array>
#include <string>
#include <vector>

namespace lazyipdca::qcqp {
struct Settings;
namespace detail { struct OriginalSignedResidual; }
struct OmittedBoundResidual {
  double maximum=0;
  long double sum=0;
  int sides=0;
};

// Exact coefficient reductions. No rank threshold and no active-set heuristic
// are used to delete a constraint. The unmodified input remains the audit model.
class ConstraintPresolve {
 public:
  struct Leaf {
    int u=-1,v=-1,a=-1,b=-1,e1=-1,e2=-1,nonzero_inequality=-1;
    double r1=0,t1=0,r2=0,t2=0;
  };
  struct AffineBound {
    int variable=-1,side=0,inequality=-1;
    double coefficient=0,bound=0;
  };
  explicit ConstraintPresolve(const QCQPProblem& original,bool remove_implied_bounds=true,
      bool remove_equality_implied_bounds=false,bool enable_exact_fixed=false,
      bool remove_affine_equality_implied_bounds=false);
  const QCQPProblem& reduced() const { return reduced_; }
  const QCQPProblem& original() const { return original_; }
  const RankPresolveStatistics& statistics() const { return stats_; }
  const std::vector<Leaf>& leaves() const { return leaves_; }
  const std::vector<AffineBound>& affine_bounds() const { return affine_bounds_; }
  const std::vector<int>& variable_map() const { return variable_map_; }
  bool has_omitted_equality_bounds() const { return !equality_implied_bounds_.empty(); }
  OmittedBoundResidual omitted_bound_residual(const Vec& reduced_point) const;
  bool changed() const { return stats_.applied; }
  void lift(Result& result) const;
  // Snapshot mapping leaves live audit counters unchanged.
  void lift_checkpoint(Result& result) const;
  bool audit(const Settings&, Result&, detail::OriginalSignedResidual*) const;
  std::string report_json() const;
 private:
  friend Result solve_with_rank_presolve(const QCQPProblem&,const Settings&);
  struct AuditView;
  void complete_affine_fixed_duals(Result&) const;
  void build_audit_plan();
  bool audit_full(const Settings&,Result&,detail::OriginalSignedResidual*) const;
  std::vector<int> audit_eq_sources_,audit_leaf_sources_,audit_lower_sources_,audit_upper_sources_;
  std::vector<int> audit_value_sources_,audit_value_slots_;
  int audit_value_slot_count_=0;
  struct AuditTerm {int i=-1,j=-1,kind=0;double coefficient=0;};
  struct AuditProgram {double constant=0;std::vector<int> terms;};
  std::vector<AuditTerm> audit_terms_;
  std::vector<AuditProgram> audit_programs_;
  std::vector<int> audit_program_ids_;
  const QCQPProblem& original_;
  QCQPProblem reduced_;
  mutable RankPresolveStatistics stats_; // per-solve counters; no process-global cache
  std::vector<Leaf> leaves_;
  // A reduced box side obtained from an ORIGINAL affine inequality. The
  // bound multiplier is mapped back with z_i = side_dual/abs(coefficient).
  std::vector<AffineBound> affine_bounds_;
  std::vector<int> affine_lower_sources_,affine_upper_sources_,affine_inequality_sources_;
  std::vector<std::array<int,3>> implied_bounds_; // variable, side (-1 lower/+1 upper), inequality
  std::vector<std::array<int,3>> equality_implied_bounds_; // reduced variable, side, original equality
  std::vector<int> variable_map_,eq_origins_,eq_groups_,ineq_origins_,ineq_groups_;
  std::vector<int> lower_origins_,upper_origins_,lift_order_,lift_function_ids_;
};

Result solve_with_rank_presolve(const QCQPProblem&,const Settings&);

bool rank_presolve_audit(const ConstraintPresolve*,const QCQPProblem&,
                        const Settings&,Result&,detail::OriginalSignedResidual*);
} // namespace lazyipdca::qcqp
