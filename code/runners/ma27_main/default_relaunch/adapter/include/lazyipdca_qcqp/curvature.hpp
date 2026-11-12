#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <random>
#include <vector>

namespace lazyipdca::qcqp {

struct EigenEstimate {
  double lambda_min = 0.0;
  Vec vector;
  bool exact = false;
};

// Certificate returned by the sparse structural-block exact oracle.  The
// oracle is deliberately fail-closed: available is true only when the current
// convexification has no low-rank terms, every structural block can be
// diagonalized exactly under the configured threshold, and zero coordinates
// outside the structural support already meet the requested margin to the
// configured tolerance.
struct BlockExactConvexificationResult {
  bool available = false;
  int block_count = 0;
  int active_dimension = 0;
  int max_block_dimension = 0;
  // Dense blocks request only the lowest rank_cap+1 modes from LAPACK.  A
  // failed partial-spectrum kernel is never accepted silently: the same
  // block is recomputed by the historical full-spectrum DSYEV path.
  int partial_spectrum_attempts = 0;
  int partial_spectrum_acceptances = 0;
  int partial_spectrum_fallbacks = 0;
  int rank_added = 0;
  double scalar_shift_added = 0.0;
  double initial_min_eigenvalue = 0.0;
  double final_min_eigenvalue = 0.0;
};

class CurvatureOracle {
 public:
  explicit CurvatureOracle(const Settings& settings);
  EigenEstimate smallest_eigenpair(const SymMatrix& Q, const Convexification& cvx);
  void ensure_convex(const SymMatrix& Q, Convexification& cvx, double target_margin, int max_rank = -1);
  BlockExactConvexificationResult ensure_convex_block_exact(
      const SymMatrix& Q, Convexification& cvx, double target_margin,
      int max_rank = -1);
  int calls() const { return calls_; }
  double seconds() const { return seconds_; }
  int partial_spectrum_workspace_queries() const {
    return partial_spectrum_workspace_queries_;
  }

 private:
  const Settings& st_;
  std::mt19937 rng_;
  int calls_ = 0;
  double seconds_ = 0.0;
  // DSYEVR mutates all of its dense/work arrays.  They therefore belong to
  // one already-stateful oracle instance (calls_ and rng_ are stateful too),
  // never to shared global storage.  A workspace queried for the largest
  // dimension seen so far is valid for every later smaller block.
  std::vector<double> partial_matrix_workspace_;
  Vec partial_eigenvalue_workspace_;
  std::vector<double> partial_eigenvector_workspace_;
  Vec partial_real_workspace_;
  std::vector<int> partial_support_workspace_;
  std::vector<int> partial_integer_workspace_;
  int partial_workspace_dimension_ = 0;
  int partial_real_workspace_size_ = 0;
  int partial_integer_workspace_size_ = 0;
  int partial_spectrum_workspace_queries_ = 0;

  Vec apply(const SymMatrix& Q, const Convexification& cvx, const Vec& x) const;
  bool dense_lowest_eigenpairs(const std::vector<double>& input, int n,
                               int requested, Vec& eigenvalues,
                               std::vector<Vec>& eigenvectors);
  EigenEstimate exact_smallest(const SymMatrix& Q, const Convexification& cvx);
  EigenEstimate lanczos_smallest(const SymMatrix& Q, const Convexification& cvx);
};

} // namespace lazyipdca::qcqp
