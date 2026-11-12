#pragma once
// v0.4.28r3p28 — sparse single-loop kernel for the Lazy-DCA primary majorant.
//
// This kernel is the "graft" of the lazy-newton machinery under the Lazy-DCA
// envelope: the outer loop keeps building its DC majorant, its accept/reject
// test, its proximal retries, working-set transport and restoration exactly
// as before; only the one-step predictor-corrector transaction on the
// already-convex ConvexSubproblem is executed by a lean sparse kernel
// (support-based models, condensed short rows, SuiteSparse LDL with AMD/METIS
// ordering, inertia-checked factorization, iterative refinement).
//
// It honours the r3p25+ single-loop contract of ConvexQCQPLazyQPKernel:
//   * exactly one transactional predictor-corrector step;
//   * success=true only for a strictly/guardedly solved convex majorant;
//   * success=false + dca_single_loop_accepted for a contracted provisional
//     state (the outer original-QCQP filter commits or rejects the point);
//   * dca_single_loop_retryable with the rolled-back start state otherwise.
// Any subproblem outside that contract (auxiliary/restoration QPs, standalone
// callers) is delegated unchanged to ConvexQCQPLazyQPKernel.
#include "lazyipdca_qcqp/inner_ipm.hpp"

namespace lazyipdca::qcqp {

class SparseSingleLoopKernel {
 public:
  explicit SparseSingleLoopKernel(const Settings& settings) : st_(settings) {}
  InnerResult solve(const ConvexSubproblem& p) const;

 private:
  const Settings& st_;
};

} // namespace lazyipdca::qcqp
