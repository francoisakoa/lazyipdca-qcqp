#include "lazyipdca_qcqp/curvature.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>

extern "C" {
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda,
            double* w, double* work, const int* lwork, int* info);
void dsyevr_(const char* jobz, const char* range, const char* uplo,
             const int* n, double* a, const int* lda, const double* vl,
             const double* vu, const int* il, const int* iu,
             const double* abstol, int* m, double* w, double* z,
             const int* ldz, int* isuppz, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info);
}

namespace lazyipdca::qcqp {
namespace {

inline void normalize(Vec& v) {
  const double n = norm2(v);
  if (!(n > 0.0) || !std::isfinite(n)) throw std::runtime_error("curvature: cannot normalize vector");
  for (double& x : v) x /= n;
}

std::pair<Vec, std::vector<Vec>> dense_eigendecomposition(std::vector<double> a, int n) {
  Vec w(n, 0.0);
  if (n == 0) return {w, {}};
  const char jobz = 'V', uplo = 'U';
  int info = 0, lwork = -1;
  double wkopt = 0.0;
  dsyev_(&jobz, &uplo, &n, a.data(), &n, w.data(), &wkopt, &lwork, &info);
  if (info != 0) throw std::runtime_error("LAPACK dsyev workspace query failed");
  lwork = std::max(1, static_cast<int>(wkopt));
  Vec work(lwork, 0.0);
  dsyev_(&jobz, &uplo, &n, a.data(), &n, w.data(), work.data(), &lwork, &info);
  if (info != 0) throw std::runtime_error("LAPACK dsyev failed");
  std::vector<Vec> vecs(n, Vec(n, 0.0));
  // LAPACK columns are eigenvectors, eigenvalues ascending.
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      vecs[j][i] = a[i + j*n];
  return {w, vecs};
}

} // namespace

CurvatureOracle::CurvatureOracle(const Settings& settings)
  : st_(settings), rng_(settings.random_seed) {}

bool CurvatureOracle::dense_lowest_eigenpairs(
    const std::vector<double>& input, int n, int requested,
    Vec& eigenvalues, std::vector<Vec>& eigenvectors) {
  eigenvalues.clear();
  eigenvectors.clear();
  if (n <= 0 || requested <= 0 || requested > n ||
      input.size() != static_cast<std::size_t>(n) * n)
    return false;

  const char jobz = 'V';
  const char range = 'I';
  const char uplo = 'U';
  const double vl = 0.0;
  const double vu = 0.0;
  const int il = 1;
  const int iu = requested;
  // LAPACK's default absolute tolerance selects its safe machine-precision
  // path.  Count, INFO and finiteness are checked before the partial result
  // can enter the exact block-certificate construction.
  const double abstol = 0.0;
  const int lda = n;
  const int ldz = n;
  int found = 0;
  int info = 0;

  const std::size_t matrix_size =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  const std::size_t vector_size =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(requested);
  partial_matrix_workspace_.resize(matrix_size);
  partial_eigenvalue_workspace_.resize(static_cast<std::size_t>(n));
  partial_eigenvector_workspace_.resize(vector_size);
  partial_support_workspace_.resize(
      static_cast<std::size_t>(2 * std::max(1, requested)));

  // DSYEVR's real and integer workspace requirements depend on N, not on the
  // selected eigenvalue count.  Retain the query for the largest block ever
  // seen by this oracle and reuse it for every same/smaller later block.
  if (n > partial_workspace_dimension_) {
    partial_matrix_workspace_ = input;
    int query_lwork = -1;
    int query_liwork = -1;
    double real_query = 0.0;
    int integer_query = 0;
    dsyevr_(&jobz, &range, &uplo, &n, partial_matrix_workspace_.data(),
            &lda, &vl, &vu, &il, &iu, &abstol, &found,
            partial_eigenvalue_workspace_.data(),
            partial_eigenvector_workspace_.data(), &ldz,
            partial_support_workspace_.data(), &real_query, &query_lwork,
            &integer_query, &query_liwork, &info);
    ++partial_spectrum_workspace_queries_;
    if (info != 0 || !std::isfinite(real_query) || real_query < 1.0 ||
        integer_query < 1)
      return false;
    partial_real_workspace_size_ =
        std::max(1, static_cast<int>(real_query));
    partial_integer_workspace_size_ = std::max(1, integer_query);
    partial_real_workspace_.resize(
        static_cast<std::size_t>(partial_real_workspace_size_));
    partial_integer_workspace_.resize(
        static_cast<std::size_t>(partial_integer_workspace_size_));
    partial_workspace_dimension_ = n;
  }

  if (partial_real_workspace_size_ < 1 ||
      partial_integer_workspace_size_ < 1)
    return false;
  // Workspace queries and eigensolves may both overwrite A.
  partial_matrix_workspace_ = input;
  found = 0;
  info = 0;
  dsyevr_(&jobz, &range, &uplo, &n, partial_matrix_workspace_.data(),
          &lda, &vl, &vu, &il, &iu, &abstol, &found,
          partial_eigenvalue_workspace_.data(),
          partial_eigenvector_workspace_.data(), &ldz,
          partial_support_workspace_.data(), partial_real_workspace_.data(),
          &partial_real_workspace_size_, partial_integer_workspace_.data(),
          &partial_integer_workspace_size_, &info);
  if (info != 0 || found != requested) return false;

  eigenvalues.assign(partial_eigenvalue_workspace_.begin(),
                     partial_eigenvalue_workspace_.begin() + found);
  eigenvectors.assign(static_cast<std::size_t>(found),
                      Vec(static_cast<std::size_t>(n), 0.0));
  for (int column = 0; column < found; ++column) {
    if (!std::isfinite(eigenvalues[static_cast<std::size_t>(column)]))
      return false;
    for (int row = 0; row < n; ++row) {
      const double value =
          partial_eigenvector_workspace_[static_cast<std::size_t>(row) +
              static_cast<std::size_t>(column) * n];
      if (!std::isfinite(value)) return false;
      eigenvectors[static_cast<std::size_t>(column)]
                  [static_cast<std::size_t>(row)] = value;
    }
  }
  return true;
}

Vec CurvatureOracle::apply(const SymMatrix& Q, const Convexification& cvx, const Vec& x) const {
  Vec y = Q.apply(x);
  if (cvx.scalar_shift != 0.0) {
    for (int i = 0; i < Q.n; ++i) y[i] += cvx.scalar_shift * x[i];
  }
  for (const auto& t : cvx.terms) {
    const double a = t.beta * dot(t.direction, x);
    axpy(a, t.direction, y);
  }
  return y;
}

EigenEstimate CurvatureOracle::exact_smallest(const SymMatrix& Q, const Convexification& cvx) {
  ++calls_;
  const int n = Q.n;
  std::vector<double> A(static_cast<std::size_t>(n)*n, 0.0);
  for (const auto& e : Q.entries) {
    A[e.i + e.j*n] += e.v;
    if (e.i != e.j) A[e.j + e.i*n] += e.v;
  }
  for (int i = 0; i < n; ++i) A[i + i*n] += cvx.scalar_shift;
  for (const auto& t : cvx.terms) {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        A[i + j*n] += t.beta * t.direction[i] * t.direction[j];
  }
  auto [w, V] = dense_eigendecomposition(std::move(A), n);
  EigenEstimate out;
  out.lambda_min = w.empty() ? 0.0 : w.front();
  out.vector = V.empty() ? Vec{} : V.front();
  out.exact = true;
  return out;
}

EigenEstimate CurvatureOracle::lanczos_smallest(const SymMatrix& Q, const Convexification& cvx) {
  ++calls_;
  const int n = Q.n;
  const int mmax = std::max(2, std::min(st_.lanczos_steps, n));
  std::normal_distribution<double> normal(0.0, 1.0);

  EigenEstimate best;
  best.lambda_min = std::numeric_limits<double>::infinity();
  best.vector.assign(n, 0.0);
  best.exact = false;

  const int starts = std::max(1, st_.lanczos_starts);
  for (int s = 0; s < starts; ++s) {
    Vec q(n, 0.0);
    for (double& v : q) v = normal(rng_);
    normalize(q);

    std::vector<Vec> basis;
    basis.reserve(mmax);
    Vec alpha, beta;
    Vec qprev(n, 0.0);
    double bprev = 0.0;

    for (int k = 0; k < mmax; ++k) {
      basis.push_back(q);
      Vec z = apply(Q, cvx, q);
      if (k > 0) axpy(-bprev, qprev, z);
      const double a = dot(q, z);
      alpha.push_back(a);
      axpy(-a, q, z);

      // Full reorthogonalization is deliberate: the oracle is diagnostic and
      // stability matters more than saving a handful of dot products.
      for (const auto& v : basis) {
        const double proj = dot(v, z);
        axpy(-proj, v, z);
      }

      const double b = norm2(z);
      if (k + 1 >= mmax || b <= 1e-14) break;
      beta.push_back(b);
      qprev = q;
      q = z;
      for (double& v : q) v /= b;
      bprev = b;
    }

    const int m = static_cast<int>(alpha.size());
    std::vector<double> T(static_cast<std::size_t>(m)*m, 0.0);
    for (int i = 0; i < m; ++i) T[i + i*m] = alpha[i];
    for (int i = 0; i + 1 < m; ++i) {
      T[i + (i+1)*m] = beta[i];
      T[(i+1) + i*m] = beta[i];
    }
    auto [evals, evecs] = dense_eigendecomposition(std::move(T), m);
    if (evals.empty()) continue;
    Vec ritz(n, 0.0);
    const Vec& coeff = evecs.front();
    for (int j = 0; j < m; ++j) axpy(coeff[j], basis[j], ritz);
    normalize(ritz);
    // Rayleigh quotient against the original operator is more trustworthy than
    // the raw tridiagonal value after finite precision reorthogonalization.
    const double lambda = dot(ritz, apply(Q, cvx, ritz));
    if (lambda < best.lambda_min) {
      best.lambda_min = lambda;
      best.vector = std::move(ritz);
    }
  }

  if (!std::isfinite(best.lambda_min)) throw std::runtime_error("Lanczos curvature oracle failed");
  return best;
}

EigenEstimate CurvatureOracle::smallest_eigenpair(const SymMatrix& Q, const Convexification& cvx) {
  if (Q.n <= st_.exact_eigen_threshold) return exact_smallest(Q, cvx);
  return lanczos_smallest(Q, cvx);
}

BlockExactConvexificationResult CurvatureOracle::ensure_convex_block_exact(
    const SymMatrix& Q, Convexification& cvx, double target_margin,
    int max_rank) {
  BlockExactConvexificationResult out;

  // This route is a certificate, not a heuristic.  Existing low-rank terms
  // may couple otherwise independent structural blocks, so refuse them rather
  // than silently deriving a block partition for the wrong operator.  The
  // target/shift guard similarly certifies the zero eigenspace outside the
  // stored structural support without materializing one direction per zero.
  if (!cvx.terms.empty() || !std::isfinite(cvx.scalar_shift) ||
      !std::isfinite(target_margin) ||
      !std::isfinite(st_.curvature_tolerance) ||
      !std::isfinite(st_.curvature_safety) ||
      st_.curvature_tolerance < 0.0 ||
      !std::isfinite(target_margin - cvx.scalar_shift) ||
      target_margin - cvx.scalar_shift > st_.curvature_tolerance) {
    return out;
  }
  Q.validate();
  if (Q.n == 0) {
    out.available = true;
    return out;
  }

  // Partition from every stored nonzero, before numerical aggregation.  In
  // particular, duplicate off-diagonal triplets that cancel still define one
  // structural component, while their values sum to zero in the dense block.
  std::vector<int> parent(static_cast<std::size_t>(Q.n));
  std::vector<int> tree_rank(static_cast<std::size_t>(Q.n), 0);
  std::vector<unsigned char> active(static_cast<std::size_t>(Q.n), 0);
  std::iota(parent.begin(), parent.end(), 0);
  auto find_root = [&](int node) {
    int root = node;
    while (parent[static_cast<std::size_t>(root)] != root)
      root = parent[static_cast<std::size_t>(root)];
    while (parent[static_cast<std::size_t>(node)] != node) {
      const int next = parent[static_cast<std::size_t>(node)];
      parent[static_cast<std::size_t>(node)] = root;
      node = next;
    }
    return root;
  };
  auto unite = [&](int left, int right) {
    int lroot = find_root(left);
    int rroot = find_root(right);
    if (lroot == rroot) return;
    if (tree_rank[static_cast<std::size_t>(lroot)] <
        tree_rank[static_cast<std::size_t>(rroot)])
      std::swap(lroot, rroot);
    parent[static_cast<std::size_t>(rroot)] = lroot;
    if (tree_rank[static_cast<std::size_t>(lroot)] ==
        tree_rank[static_cast<std::size_t>(rroot)])
      ++tree_rank[static_cast<std::size_t>(lroot)];
  };
  for (const auto& entry : Q.entries) {
    if (entry.v == 0.0) continue;
    active[static_cast<std::size_t>(entry.i)] = 1;
    active[static_cast<std::size_t>(entry.j)] = 1;
    if (entry.i != entry.j) unite(entry.i, entry.j);
  }

  std::vector<std::vector<int>> blocks;
  std::vector<int> root_to_block(static_cast<std::size_t>(Q.n), -1);
  for (int index = 0; index < Q.n; ++index) {
    if (!active[static_cast<std::size_t>(index)]) continue;
    const int root = find_root(index);
    int& block_index = root_to_block[static_cast<std::size_t>(root)];
    if (block_index < 0) {
      block_index = static_cast<int>(blocks.size());
      blocks.emplace_back();
    }
    blocks[static_cast<std::size_t>(block_index)].push_back(index);
    ++out.active_dimension;
  }
  out.block_count = static_cast<int>(blocks.size());
  for (const auto& block : blocks) {
    out.max_block_dimension = std::max(
        out.max_block_dimension, static_cast<int>(block.size()));
    if (static_cast<int>(block.size()) > st_.exact_eigen_threshold)
      return out;
  }

  const int rank_cap = std::max(
      0, max_rank >= 0 ? max_rank : st_.max_rank_per_quadratic);

  std::vector<int> block_of(static_cast<std::size_t>(Q.n), -1);
  std::vector<int> local_of(static_cast<std::size_t>(Q.n), -1);
  for (int block_index = 0;
       block_index < static_cast<int>(blocks.size()); ++block_index) {
    const auto& block = blocks[static_cast<std::size_t>(block_index)];
    for (int local = 0; local < static_cast<int>(block.size()); ++local) {
      const int global = block[static_cast<std::size_t>(local)];
      block_of[static_cast<std::size_t>(global)] = block_index;
      local_of[static_cast<std::size_t>(global)] = local;
    }
  }

  struct BlockMode {
    double eigenvalue = 0.0;
    int block = -1;
    int local_order = -1;
    Vec local_direction;
    double correction = 0.0;
  };
  std::vector<BlockMode> modes;
  modes.reserve(static_cast<std::size_t>(out.active_dimension + 1));

  // calls() tracks logical oracle certifications, consistently with the
  // monolithic exact route, rather than exposing the number of independent
  // LAPACK kernels used to realize one structural certificate.
  ++calls_;

  std::vector<std::vector<double>> dense_blocks;
  dense_blocks.reserve(blocks.size());
  for (const auto& block : blocks) {
    const std::size_t dimension = block.size();
    dense_blocks.emplace_back(dimension * dimension, 0.0);
  }
  for (const auto& entry : Q.entries) {
    if (entry.v == 0.0) continue;
    const int block_index = block_of[static_cast<std::size_t>(entry.i)];
    const int row = local_of[static_cast<std::size_t>(entry.i)];
    const int column = local_of[static_cast<std::size_t>(entry.j)];
    const int dimension = static_cast<int>(
        blocks[static_cast<std::size_t>(block_index)].size());
    auto& dense = dense_blocks[static_cast<std::size_t>(block_index)];
    dense[static_cast<std::size_t>(row + column * dimension)] += entry.v;
    if (row != column)
      dense[static_cast<std::size_t>(column + row * dimension)] += entry.v;
  }

  for (int block_index = 0;
       block_index < static_cast<int>(blocks.size()); ++block_index) {
    const auto& block = blocks[static_cast<std::size_t>(block_index)];
    const int dimension = static_cast<int>(block.size());
    auto& dense = dense_blocks[static_cast<std::size_t>(block_index)];
    for (int index = 0; index < dimension; ++index)
      dense[static_cast<std::size_t>(index + index * dimension)] +=
          cvx.scalar_shift;

    Vec eigenvalues;
    std::vector<Vec> eigenvectors;
    if (dimension == 1) {
      // PGLIB Hessians contain many scalar generator-cost blocks.  Calling
      // dsyev for each of them would turn an O(n) scan into thousands of tiny
      // LAPACK dispatches over the outer iterations.
      eigenvalues.push_back(dense[0]);
      eigenvectors.push_back(Vec{1.0});
    } else {
      // At most rank_cap modes can receive low-rank corrections.  The next
      // global mode alone determines the scalar safeguard, so retaining the
      // lowest rank_cap+1 modes from every independent block is sufficient:
      // no omitted mode can precede the first represented uncorrected mode.
      const int requested = rank_cap >= dimension ? dimension : rank_cap + 1;
      ++out.partial_spectrum_attempts;
      if (dense_lowest_eigenpairs(dense, dimension, requested, eigenvalues,
                                  eigenvectors)) {
        ++out.partial_spectrum_acceptances;
      } else {
        ++out.partial_spectrum_fallbacks;
        auto decomposition = dense_eigendecomposition(dense, dimension);
        eigenvalues = std::move(decomposition.first);
        eigenvectors = std::move(decomposition.second);
      }
    }
    for (int order = 0; order < static_cast<int>(eigenvalues.size()); ++order) {
      BlockMode mode;
      mode.eigenvalue = eigenvalues[static_cast<std::size_t>(order)];
      if (!std::isfinite(mode.eigenvalue)) return out;
      mode.block = block_index;
      mode.local_order = order;
      mode.local_direction = std::move(
          eigenvectors[static_cast<std::size_t>(order)]);
      modes.push_back(std::move(mode));
    }
  }

  // Every coordinate outside the structural support contributes the zero
  // eigenvalue of Q, hence scalar_shift for the current model.  One sentinel
  // represents that multiplicity: the admission guard guarantees it can
  // never consume a low-rank slot, and one copy suffices for both minima and
  // the scalar safeguard.
  if (out.active_dimension < Q.n) {
    BlockMode zero_mode;
    zero_mode.eigenvalue = cvx.scalar_shift;
    modes.push_back(std::move(zero_mode));
  }
  std::sort(modes.begin(), modes.end(), [](const BlockMode& left,
                                           const BlockMode& right) {
    if (left.eigenvalue != right.eigenvalue)
      return left.eigenvalue < right.eigenvalue;
    if (left.block != right.block) return left.block < right.block;
    return left.local_order < right.local_order;
  });
  if (modes.empty()) return out;
  out.initial_min_eigenvalue = modes.front().eigenvalue;

  std::vector<LowRankTerm> additions;
  additions.reserve(static_cast<std::size_t>(rank_cap));
  std::size_t first_uncorrected = 0;
  while (first_uncorrected < modes.size() &&
         modes[first_uncorrected].eigenvalue <
             target_margin - st_.curvature_tolerance &&
         static_cast<int>(additions.size()) < rank_cap) {
    BlockMode& mode = modes[first_uncorrected];
    // A zero-support sentinel cannot enter this branch by construction.
    if (mode.block < 0) return out;
    const double deficit = target_margin - mode.eigenvalue;
    const double beta = std::max(
        deficit * (1.0 + st_.curvature_safety),
        st_.curvature_tolerance);
    if (!(beta > 0.0) || !std::isfinite(beta)) return out;
    Vec direction(static_cast<std::size_t>(Q.n), 0.0);
    const auto& block = blocks[static_cast<std::size_t>(mode.block)];
    for (int local = 0; local < static_cast<int>(block.size()); ++local) {
      direction[static_cast<std::size_t>(
          block[static_cast<std::size_t>(local)])] =
          mode.local_direction[static_cast<std::size_t>(local)];
    }
    additions.push_back({beta, std::move(direction)});
    mode.correction = beta;
    ++first_uncorrected;
  }

  double scalar_addition = 0.0;
  if (first_uncorrected < modes.size() &&
      modes[first_uncorrected].eigenvalue < target_margin) {
    scalar_addition =
        (target_margin - modes[first_uncorrected].eigenvalue) *
        (1.0 + st_.curvature_safety);
    if (!(scalar_addition > 0.0) || !std::isfinite(scalar_addition))
      return out;
  }

  double final_minimum = std::numeric_limits<double>::infinity();
  for (const auto& mode : modes) {
    final_minimum = std::min(
        final_minimum,
        mode.eigenvalue + mode.correction + scalar_addition);
  }
  if (!std::isfinite(final_minimum) ||
      final_minimum < target_margin - st_.curvature_tolerance) {
    return out;
  }

  Convexification updated = cvx;
  updated.scalar_shift += scalar_addition;
  updated.terms.insert(updated.terms.end(),
                       std::make_move_iterator(additions.begin()),
                       std::make_move_iterator(additions.end()));
  cvx = std::move(updated);

  out.available = true;
  out.rank_added = static_cast<int>(cvx.terms.size());
  out.scalar_shift_added = scalar_addition;
  out.final_min_eigenvalue = final_minimum;
  return out;
}

void CurvatureOracle::ensure_convex(const SymMatrix& Q, Convexification& cvx, double target_margin, int max_rank) {
  struct Timer {
    double& seconds;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    ~Timer() { seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(); }
  } timer{seconds_};
  if (Q.n == 0) return;

  // Optional proof-oriented fallback. Because low-rank terms are PSD, a
  // Gershgorin lower bound for Q + rho I is also a valid lower bound after
  // adding the retained corrections.
  if (st_.strict_scalar_certificate) {
    const double glb = Q.gershgorin_lower_bound();
    if (glb + cvx.scalar_shift < target_margin) {
      cvx.scalar_shift = target_margin - glb;
    }
    return;
  }

  const int rank_cap = (max_rank >= 0) ? max_rank : st_.max_rank_per_quadratic;

  // Exact small/medium path: diagonalize ONCE and use all negative
  // eigenvectors returned by the same dsyev call.  The former implementation
  // called dsyev again after every rank-one correction (A4X12: about 44 full
  // eigendecompositions per outer iteration).  Because the eigenvectors of the
  // current symmetric operator are orthonormal, correcting several of them in
  // one shot is algebraically equivalent and avoids that O(r*n^3) repetition.
  if (Q.n <= st_.exact_eigen_threshold) {
    ++calls_;
    const int n = Q.n;
    std::vector<double> A(static_cast<std::size_t>(n)*n, 0.0);
    for (const auto& e : Q.entries) {
      A[e.i + e.j*n] += e.v;
      if (e.i != e.j) A[e.j + e.i*n] += e.v;
    }
    for (int i = 0; i < n; ++i) A[i + i*n] += cvx.scalar_shift;
    for (const auto& t : cvx.terms) {
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          A[i + j*n] += t.beta * t.direction[i] * t.direction[j];
    }
    auto [w, V] = dense_eigendecomposition(std::move(A), n);
    int room = std::max(0, rank_cap - static_cast<int>(cvx.terms.size()));
    int first_uncorrected = 0;
    while (first_uncorrected < n &&
           w[first_uncorrected] < target_margin - st_.curvature_tolerance &&
           room > 0) {
      const double deficit = target_margin - w[first_uncorrected];
      const double beta = std::max(deficit * (1.0 + st_.curvature_safety),
                                   st_.curvature_tolerance);
      cvx.terms.push_back({beta, V[first_uncorrected]});
      ++first_uncorrected;
      --room;
    }

    // If the rank budget did not cover every negative direction, shift the
    // remaining orthogonal subspace once.  No second eigendecomposition is
    // necessary: corrections along earlier eigenvectors do not change the
    // uncorrected eigenvalues from this decomposition.
    if (first_uncorrected < n && w[first_uncorrected] < target_margin) {
      cvx.scalar_shift += (target_margin - w[first_uncorrected]) *
                          (1.0 + st_.curvature_safety);
    }
    return;
  }

  // Large-scale path: keep the incremental Lanczos oracle.
  while (static_cast<int>(cvx.terms.size()) < rank_cap) {
    auto eig = smallest_eigenpair(Q, cvx);
    if (eig.lambda_min >= target_margin - st_.curvature_tolerance) return;

    const double deficit = target_margin - eig.lambda_min;
    const double beta = std::max(deficit * (1.0 + st_.curvature_safety), st_.curvature_tolerance);
    cvx.terms.push_back({beta, eig.vector});
  }

  // Once the rank budget is exhausted, preserve robustness with a scalar
  // safeguard rather than silently accepting an indefinite model.
  auto eig = smallest_eigenpair(Q, cvx);
  if (eig.lambda_min < target_margin) {
    cvx.scalar_shift += (target_margin - eig.lambda_min) * (1.0 + st_.curvature_safety);
  }
}

} // namespace lazyipdca::qcqp
