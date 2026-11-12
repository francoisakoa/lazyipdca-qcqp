#include "lazyipdca_qcqp/condensed_retraction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

extern "C" {
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a,
            const int* lda, double* w, double* work, const int* lwork,
            int* info);
}

namespace lazyipdca::qcqp {
namespace {

constexpr double kTiny = 64.0 * std::numeric_limits<double>::epsilon();

bool finite_vec(const Vec& x) {
  for (double v : x)
    if (!std::isfinite(v)) return false;
  return true;
}

void normalize(Vec& v) {
  const double n = norm2(v);
  if (!(n > 0.0) || !std::isfinite(n))
    throw std::runtime_error("condensed retraction: cannot normalize vector");
  for (double& a : v) a /= n;
}

std::pair<Vec, std::vector<Vec>> dense_eigendecomposition(std::vector<double> a,
                                                           int n) {
  Vec w(std::max(0, n), 0.0);
  if (n == 0) return {w, {}};
  const char jobz = 'V';
  const char uplo = 'U';
  int info = 0;
  int lwork = -1;
  double wkopt = 0.0;
  dsyev_(&jobz, &uplo, &n, a.data(), &n, w.data(), &wkopt, &lwork, &info);
  if (info != 0 || !std::isfinite(wkopt))
    throw std::runtime_error("condensed retraction: LAPACK dsyev workspace query failed");
  lwork = std::max(1, static_cast<int>(wkopt));
  Vec work(static_cast<std::size_t>(lwork), 0.0);
  dsyev_(&jobz, &uplo, &n, a.data(), &n, w.data(), work.data(), &lwork, &info);
  if (info != 0)
    throw std::runtime_error("condensed retraction: LAPACK dsyev failed");
  std::vector<Vec> V(static_cast<std::size_t>(n), Vec(static_cast<std::size_t>(n), 0.0));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      V[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] =
          a[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n];
  return {std::move(w), std::move(V)};
}

inline double max_abs_value(const Vec& x) {
  double m = 0.0;
  for (double v : x) m = std::max(m, std::abs(v));
  return m;
}

void add_retraction_work(const CondensedRetractionResult& attempted,
                         CondensedRetractionResult& result) {
  result.continuation_attempts += attempted.continuation_attempts;
  result.continuation_acceptances += attempted.continuation_acceptances;
  result.newton_iterations += attempted.newton_iterations;
  result.krylov_iterations += attempted.krylov_iterations;
  result.line_search_backtracks += attempted.line_search_backtracks;
  result.normal_solves += attempted.normal_solves;
  result.affine_normal_map_reuses += attempted.affine_normal_map_reuses;
  result.newton_endpoint_reuses += attempted.newton_endpoint_reuses;
  result.quadratic_linesearch_reuses +=
      attempted.quadratic_linesearch_reuses;
  result.quadratic_linesearch_predictor_builds +=
      attempted.quadratic_linesearch_predictor_builds;
  result.quadratic_linesearch_exact_checks +=
      attempted.quadratic_linesearch_exact_checks;
  result.quadratic_linesearch_early_rejections +=
      attempted.quadratic_linesearch_early_rejections;
  result.quadratic_linesearch_rows_skipped +=
      attempted.quadratic_linesearch_rows_skipped;
}

} // namespace

CondensedQuadraticRetraction::CondensedQuadraticRetraction(
    const Settings& settings, bool allow_flow_lift_schur)
    : st_(settings), allow_flow_lift_schur_(allow_flow_lift_schur),
      factor_settings_(settings), tangent_certificate_settings_(settings),
      rng_(settings.random_seed + 428u) {
  // The normal metric and dual regularization are assembled explicitly in the
  // bordered matrix.  No backend may add an unreported KKT perturbation.
  factor_settings_.kkt_adaptive_regularization = true;
  // The scaled normal operator must detect rank loss up to roundoff. A
  // sign-only native inertia check can accept a spurious O(epsilon) pivot
  // for duplicate equality rows, depending on the BLAS implementation.
  // Keep this guard local: Newton and tangent-curvature settings retain
  // their original inertia policy.
  factor_settings_.native_inertia_relative_zero_tolerance = std::max(
      factor_settings_.native_inertia_relative_zero_tolerance, kTiny);
  // The certificate must factor exactly the assembled bordered matrix; with
  // the common factor interface, adaptive mode means no implicit fixed shift.
  tangent_certificate_settings_.kkt_adaptive_regularization = true;
  std::string message;
  factor_ = make_symmetric_factorization(factor_settings_, message);
}

bool CondensedQuadraticRetraction::detect_flow_lift_schur_structure(
    const QCQPProblem& problem) {
  if (flow_lift_structure_problem_ == &problem)
    return flow_lift_schur_candidate_;

  flow_lift_structure_problem_ = &problem;
  flow_lifts_.clear();
  flow_core_equalities_.clear();
  flow_base_variables_.clear();
  flow_full_to_base_.assign(static_cast<std::size_t>(problem.n), -1);
  flow_equality_to_lift_.assign(problem.equalities.size(), -1);
  flow_lift_schur_candidate_ = false;
  if (!allow_flow_lift_schur_ || problem.equalities.empty()) return false;
  auto reject_structure = [&]() {
    flow_lifts_.clear();
    flow_core_equalities_.clear();
    flow_base_variables_.clear();
    std::fill(flow_full_to_base_.begin(), flow_full_to_base_.end(), -1);
    std::fill(flow_equality_to_lift_.begin(),
              flow_equality_to_lift_.end(), -1);
    return false;
  };

  const int n = problem.n;
  std::vector<int> equality_linear_incidence(static_cast<std::size_t>(n), 0);
  std::vector<int> equality_hessian_incidence(static_cast<std::size_t>(n), 0);
  std::vector<int> inequality_incidence(static_cast<std::size_t>(n), 0);
  std::vector<int> last_inequality(static_cast<std::size_t>(n), -1);
  std::vector<unsigned char> objective_incidence(static_cast<std::size_t>(n), 0);

  for (int j = 0; j < n; ++j)
    if (problem.objective.linear[static_cast<std::size_t>(j)] != 0.0)
      objective_incidence[static_cast<std::size_t>(j)] = 1;
  for (const auto& entry : problem.objective.hessian.entries) {
    objective_incidence[static_cast<std::size_t>(entry.i)] = 1;
    objective_incidence[static_cast<std::size_t>(entry.j)] = 1;
  }
  for (const auto& equality : problem.equalities) {
    for (int j = 0; j < n; ++j)
      if (equality.linear[static_cast<std::size_t>(j)] != 0.0)
        ++equality_linear_incidence[static_cast<std::size_t>(j)];
    for (const auto& entry : equality.hessian.entries) {
      ++equality_hessian_incidence[static_cast<std::size_t>(entry.i)];
      if (entry.j != entry.i)
        ++equality_hessian_incidence[static_cast<std::size_t>(entry.j)];
    }
  }
  for (int i = 0; i < static_cast<int>(problem.inequalities.size()); ++i) {
    const auto& inequality = problem.inequalities[static_cast<std::size_t>(i)];
    for (int j = 0; j < n; ++j) {
      if (inequality.linear[static_cast<std::size_t>(j)] != 0.0 &&
          last_inequality[static_cast<std::size_t>(j)] != i) {
        ++inequality_incidence[static_cast<std::size_t>(j)];
        last_inequality[static_cast<std::size_t>(j)] = i;
      }
    }
    for (const auto& entry : inequality.hessian.entries) {
      for (int j : {entry.i, entry.j}) {
        if (last_inequality[static_cast<std::size_t>(j)] != i) {
          ++inequality_incidence[static_cast<std::size_t>(j)];
          last_inequality[static_cast<std::size_t>(j)] = i;
        }
      }
    }
  }

  std::unordered_map<std::string, int> inequality_by_name;
  for (int i = 0; i < static_cast<int>(problem.inequalities.size()); ++i)
    inequality_by_name.emplace(
        problem.inequalities[static_cast<std::size_t>(i)].name, i);
  std::unordered_map<std::string, int> slim_lift_counts;
  std::vector<unsigned char> auxiliary(static_cast<std::size_t>(n), 0);

  for (int i = 0; i < static_cast<int>(problem.equalities.size()); ++i) {
    const auto& equality = problem.equalities[static_cast<std::size_t>(i)];
    std::string suffix;
    if (equality.name.rfind("flowP_", 0) == 0)
      suffix = equality.name.substr(6);
    else if (equality.name.rfind("flowQ_", 0) == 0)
      suffix = equality.name.substr(6);
    else
      continue;
    const std::string slim_name = "slim_" + suffix;
    const auto slim_it = inequality_by_name.find(slim_name);
    if (slim_it == inequality_by_name.end()) return reject_structure();
    const auto& slim = problem.inequalities[
        static_cast<std::size_t>(slim_it->second)];

    int candidate = -1;
    double sign = 0.0;
    for (int j = 0; j < n; ++j) {
      const double a = equality.linear[static_cast<std::size_t>(j)];
      if (a != 1.0 && a != -1.0) continue;
      if (equality_linear_incidence[static_cast<std::size_t>(j)] != 1 ||
          equality_hessian_incidence[static_cast<std::size_t>(j)] != 0 ||
          objective_incidence[static_cast<std::size_t>(j)] != 0 ||
          inequality_incidence[static_cast<std::size_t>(j)] != 1 ||
          std::isfinite(problem.lb[static_cast<std::size_t>(j)]) ||
          std::isfinite(problem.ub[static_cast<std::size_t>(j)]))
        continue;

      double diagonal = 0.0;
      bool off_diagonal = false;
      for (const auto& entry : slim.hessian.entries) {
        if (entry.i == j && entry.j == j) diagonal += entry.v;
        else if (entry.i == j || entry.j == j) off_diagonal = true;
      }
      if (!(diagonal > 0.0) || off_diagonal ||
          slim.linear[static_cast<std::size_t>(j)] != 0.0)
        continue;
      if (candidate >= 0) return reject_structure();
      candidate = j;
      sign = a;
    }
    if (candidate < 0 || auxiliary[static_cast<std::size_t>(candidate)])
      return reject_structure();
    auxiliary[static_cast<std::size_t>(candidate)] = 1;
    flow_equality_to_lift_[static_cast<std::size_t>(i)] =
        static_cast<int>(flow_lifts_.size());
    flow_lifts_.push_back({i, candidate, sign});
    ++slim_lift_counts[slim_name];
  }

  // A PGLib apparent-power disk must contain exactly its real and imaginary
  // auxiliaries.  Requiring complete pairs prevents a partially recognized
  // export from silently changing the normal operator.
  if (flow_lifts_.size() < 2) return reject_structure();
  for (const auto& item : slim_lift_counts)
    if (item.second != 2) return reject_structure();

  for (int i = 0; i < static_cast<int>(problem.equalities.size()); ++i)
    if (flow_equality_to_lift_[static_cast<std::size_t>(i)] < 0)
      flow_core_equalities_.push_back(i);
  for (int j = 0; j < n; ++j) {
    if (auxiliary[static_cast<std::size_t>(j)]) continue;
    flow_full_to_base_[static_cast<std::size_t>(j)] =
        static_cast<int>(flow_base_variables_.size());
    flow_base_variables_.push_back(j);
  }
  if (flow_base_variables_.empty()) return reject_structure();

  // The unique-column tests above imply this condition, but check it
  // explicitly so future changes to the detector fail closed.
  for (int equality_index : flow_core_equalities_) {
    const auto& equality =
        problem.equalities[static_cast<std::size_t>(equality_index)];
    for (const auto& lift : flow_lifts_) {
      const int j = lift.variable;
      if (equality.linear[static_cast<std::size_t>(j)] != 0.0)
        return reject_structure();
      for (const auto& entry : equality.hessian.entries)
        if (entry.i == j || entry.j == j) return reject_structure();
    }
  }

  flow_lift_schur_candidate_ = true;
  return true;
}

void CondensedQuadraticRetraction::build_sparse_equality_data(
    const QCQPProblem& problem) {
  if (equality_sparse_problem_ == &problem &&
      static_cast<int>(equality_sparse_data_.size()) ==
          static_cast<int>(problem.equalities.size()))
    return;

  equality_sparse_data_.clear();
  equality_sparse_data_.resize(problem.equalities.size());
  std::vector<int> position(static_cast<std::size_t>(problem.n), -1);
  std::vector<int> touched;

  for (std::size_t qi = 0; qi < problem.equalities.size(); ++qi) {
    const auto& q = problem.equalities[qi];
    auto& data = equality_sparse_data_[qi];
    data.linear_entries.clear();
    data.gradient_components.clear();

    std::vector<int> support;
    support.reserve(q.hessian.entries.size() * 2 + 8);
    for (int j = 0; j < problem.n; ++j) {
      const double a = q.linear[static_cast<std::size_t>(j)];
      if (a != 0.0) {
        data.linear_entries.emplace_back(j, a);
        if (!fixed_[static_cast<std::size_t>(j)]) support.push_back(j);
      }
    }
    for (const auto& e : q.hessian.entries) {
      if (!fixed_[static_cast<std::size_t>(e.i)]) support.push_back(e.i);
      if (e.i != e.j && !fixed_[static_cast<std::size_t>(e.j)])
        support.push_back(e.j);
    }
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end()), support.end());
    data.gradient_components.reserve(support.size());
    touched.clear();
    for (int j : support) {
      SparseGradientComponent component;
      component.index = j;
      component.linear = q.linear[static_cast<std::size_t>(j)];
      position[static_cast<std::size_t>(j)] =
          static_cast<int>(data.gradient_components.size());
      touched.push_back(j);
      data.gradient_components.push_back(std::move(component));
    }
    // Preserve the exact per-coordinate accumulation order of SymMatrix::apply:
    // iterate the stored Hessian entries once and append each contribution in
    // that same order to the affected component.
    for (const auto& e : q.hessian.entries) {
      if (!fixed_[static_cast<std::size_t>(e.i)]) {
        const int pos = position[static_cast<std::size_t>(e.i)];
        if (pos >= 0)
          data.gradient_components[static_cast<std::size_t>(pos)]
              .hessian_terms.emplace_back(e.j, e.v);
      }
      if (e.i != e.j && !fixed_[static_cast<std::size_t>(e.j)]) {
        const int pos = position[static_cast<std::size_t>(e.j)];
        if (pos >= 0)
          data.gradient_components[static_cast<std::size_t>(pos)]
              .hessian_terms.emplace_back(e.i, e.v);
      }
    }
    for (int j : touched) position[static_cast<std::size_t>(j)] = -1;
  }
  equality_sparse_problem_ = &problem;
}

void CondensedQuadraticRetraction::build_sparse_inequality_data(
    const QCQPProblem& problem) {
  if (inequality_sparse_problem_ == &problem &&
      inequality_sparse_data_.size() == problem.inequalities.size())
    return;

  inequality_sparse_data_.clear();
  inequality_sparse_data_.resize(problem.inequalities.size());
  for (std::size_t qi = 0; qi < problem.inequalities.size(); ++qi) {
    const auto& q = problem.inequalities[qi];
    auto& entries = inequality_sparse_data_[qi].linear_entries;
    entries.clear();
    for (int j = 0; j < problem.n; ++j) {
      const double a = q.linear[static_cast<std::size_t>(j)];
      if (a != 0.0) entries.emplace_back(j, a);
    }
  }
  inequality_sparse_problem_ = &problem;
}

bool CondensedQuadraticRetraction::parallel_equalities_enabled() const {
#ifdef LAZYIPDCA_HAVE_OPENMP
  return st_.dca_condensed_parallel_equalities &&
      me_ >= std::max(1, st_.dca_condensed_parallel_min_equalities);
#else
  return false;
#endif
}

double CondensedQuadraticRetraction::sparse_equality_value(
    int equality_index, const Vec& x) const {
  const auto& q = problem_->equalities[static_cast<std::size_t>(equality_index)];
  const auto& data = equality_sparse_data_[static_cast<std::size_t>(equality_index)];
  double linear_sum = 0.0;
  for (const auto& [j, a] : data.linear_entries)
    linear_sum += a * x[static_cast<std::size_t>(j)];
  double quadratic_sum = 0.0;
  for (const auto& e : q.hessian.entries) {
    if (e.i == e.j)
      quadratic_sum += e.v * x[static_cast<std::size_t>(e.i)] *
          x[static_cast<std::size_t>(e.i)];
    else
      quadratic_sum += 2.0 * e.v * x[static_cast<std::size_t>(e.i)] *
          x[static_cast<std::size_t>(e.j)];
  }
  return q.constant + linear_sum + 0.5 * quadratic_sum;
}

double CondensedQuadraticRetraction::sparse_inequality_value(
    int inequality_index, const Vec& x) const {
  const auto& q =
      problem_->inequalities[static_cast<std::size_t>(inequality_index)];
  const auto& data = inequality_sparse_data_[
      static_cast<std::size_t>(inequality_index)];
  double linear_sum = 0.0;
  // The support was collected by increasing coordinate, exactly the order of
  // the nonzero products in dot(q.linear, x).
  for (const auto& [j, a] : data.linear_entries)
    linear_sum += a * x[static_cast<std::size_t>(j)];
  double quadratic_sum = 0.0;
  // Keep SymMatrix::quad's stored-triplet order and operation shape.  This is
  // an exact evaluator, not a reordered sparse reduction.
  for (const auto& e : q.hessian.entries) {
    if (e.i == e.j)
      quadratic_sum += e.v * x[static_cast<std::size_t>(e.i)] *
          x[static_cast<std::size_t>(e.i)];
    else
      quadratic_sum += 2.0 * e.v * x[static_cast<std::size_t>(e.i)] *
          x[static_cast<std::size_t>(e.j)];
  }
  return q.constant + linear_sum + 0.5 * quadratic_sum;
}

void CondensedQuadraticRetraction::inequality_values_into(
    const Vec& x, Vec& values) const {
  const int mi = static_cast<int>(problem_->inequalities.size());
  values.resize(static_cast<std::size_t>(mi));
  // Skipping zero*Inf would change the dense evaluator's nonfinite semantics.
  // Retraction points are finite, while this guard keeps the public evaluator
  // behavior intact for arbitrary inputs.
  const bool sparse = finite_vec(x) &&
      inequality_sparse_problem_ == problem_ &&
      static_cast<int>(inequality_sparse_data_.size()) == mi;
  for (int i = 0; i < mi; ++i) {
    values[static_cast<std::size_t>(i)] = sparse
        ? sparse_inequality_value(i, x)
        : problem_->inequalities[static_cast<std::size_t>(i)].value(x);
  }
}

Feasibility CondensedQuadraticRetraction::feasibility_from_equality_values(
    const Vec& x, const Vec& values, Vec* inequality_values) const {
  Feasibility f;
  for (double value : values) f.equality_inf = std::max(f.equality_inf, std::abs(value));
  Vec& gvalues = inequality_values
      ? *inequality_values
      : inequality_values_work_;
  inequality_values_into(x, gvalues);
  for (double value : gvalues)
    f.inequality_inf = std::max(f.inequality_inf, std::max(0.0, value));
  for (int j = 0; j < n_; ++j) {
    if (std::isfinite(problem_->lb[static_cast<std::size_t>(j)]))
      f.bound_inf = std::max(
          f.bound_inf,
          std::max(0.0, problem_->lb[static_cast<std::size_t>(j)] -
                            x[static_cast<std::size_t>(j)]));
    if (std::isfinite(problem_->ub[static_cast<std::size_t>(j)]))
      f.bound_inf = std::max(
          f.bound_inf,
          std::max(0.0, x[static_cast<std::size_t>(j)] -
                            problem_->ub[static_cast<std::size_t>(j)]));
  }
  f.total_inf = std::max({f.equality_inf, f.inequality_inf, f.bound_inf});
  return f;
}

Feasibility CondensedQuadraticRetraction::evaluate_feasibility(const Vec& x) const {
  if (!problem_ || static_cast<int>(x.size()) != n_)
    return Feasibility{};
  const Vec values = equality_values(x);
  return feasibility_from_equality_values(x, values);
}

void CondensedQuadraticRetraction::equality_values_into(const Vec& x, Vec& values) const {
  values.resize(static_cast<std::size_t>(me_));
  const bool sparse = st_.dca_condensed_sparse_equality_evaluation &&
      static_cast<int>(equality_sparse_data_.size()) == me_;
  const bool parallel = parallel_equalities_enabled();
  if (sparse) sparse_equality_evaluations_ += me_;
  if (parallel) ++parallel_equality_batches_;
#ifdef LAZYIPDCA_HAVE_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
  for (int i = 0; i < me_; ++i) {
    values[static_cast<std::size_t>(i)] = sparse
        ? sparse_equality_value(i, x)
        : problem_->equalities[static_cast<std::size_t>(i)].value(x);
  }
}

Vec CondensedQuadraticRetraction::equality_values(const Vec& x) const {
  Vec values;
  equality_values_into(x, values);
  return values;
}

CondensedQuadraticRetraction::SparseGradientRow
CondensedQuadraticRetraction::sparse_gradient(int equality_index,
                                               const Vec& x) const {
  SparseGradientRow row;
  if (st_.dca_condensed_sparse_equality_evaluation &&
      static_cast<int>(equality_sparse_data_.size()) == me_) {
    const auto& data =
        equality_sparse_data_[static_cast<std::size_t>(equality_index)];
    row.entries.reserve(data.gradient_components.size());
    for (const auto& component : data.gradient_components) {
      double value = 0.0;
      for (const auto& [j, a] : component.hessian_terms)
        value += a * x[static_cast<std::size_t>(j)];
      value += component.linear;
      if (value != 0.0) row.entries.emplace_back(component.index, value);
    }
    return row;
  }

  const auto& q = problem_->equalities[static_cast<std::size_t>(equality_index)];
  Vec dense = q.gradient(x);
  row.entries.reserve(q.hessian.entries.size() * 2 + 8);
  for (int j = 0; j < n_; ++j) {
    const double a = dense[static_cast<std::size_t>(j)];
    if (!fixed_[static_cast<std::size_t>(j)] && a != 0.0)
      row.entries.emplace_back(j, a);
  }
  return row;
}

double CondensedQuadraticRetraction::sparse_row_dot(
    const SparseGradientRow& row, const Vec& v) const {
  double value = 0.0;
  for (const auto& [j, a] : row.entries)
    value += a * v[static_cast<std::size_t>(j)];
  return value;
}

double CondensedQuadraticRetraction::symmetric_bilinear(
    const SymMatrix& matrix, const Vec& x, const Vec& y) const {
  double value = 0.0;
  for (const auto& e : matrix.entries) {
    if (e.i == e.j) {
      value += e.v * x[static_cast<std::size_t>(e.i)] *
          y[static_cast<std::size_t>(e.i)];
    } else {
      value += e.v *
          (x[static_cast<std::size_t>(e.i)] * y[static_cast<std::size_t>(e.j)] +
           x[static_cast<std::size_t>(e.j)] * y[static_cast<std::size_t>(e.i)]);
    }
  }
  return value;
}

Vec CondensedQuadraticRetraction::base_jacobian_product(const Vec& v) const {
  Vec out(static_cast<std::size_t>(me_), 0.0);
  const bool parallel = parallel_equalities_enabled();
  if (parallel) ++parallel_equality_batches_;
#ifdef LAZYIPDCA_HAVE_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
  for (int i = 0; i < me_; ++i)
    out[static_cast<std::size_t>(i)] =
        sparse_row_dot(base_gradients_[static_cast<std::size_t>(i)], v);
  return out;
}

void CondensedQuadraticRetraction::jacobian_product_from_delta_into(
    const Vec& delta, const Vec& v, Vec& out) const {
  out.resize(static_cast<std::size_t>(me_));
  const bool parallel = parallel_equalities_enabled();
  if (parallel) ++parallel_equality_batches_;
#ifdef LAZYIPDCA_HAVE_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
  for (int i = 0; i < me_; ++i) {
    const auto& q = problem_->equalities[static_cast<std::size_t>(i)];
    out[static_cast<std::size_t>(i)] =
        sparse_row_dot(base_gradients_[static_cast<std::size_t>(i)], v) +
        symmetric_bilinear(q.hessian, delta, v);
  }
}

Vec CondensedQuadraticRetraction::jacobian_product_from_delta(
    const Vec& delta, const Vec& v) const {
  Vec out;
  jacobian_product_from_delta_into(delta, v, out);
  return out;
}

void CondensedQuadraticRetraction::jacobian_product_at_into(
    const Vec& point, const Vec& v, Vec& out) const {
  gmres_point_delta_ = point;
  axpy(-1.0, base_, gmres_point_delta_);
  jacobian_product_from_delta_into(gmres_point_delta_, v, out);
}

Vec CondensedQuadraticRetraction::jacobian_product_at(const Vec& point,
                                                       const Vec& v) const {
  Vec out;
  jacobian_product_at_into(point, v, out);
  return out;
}

bool CondensedQuadraticRetraction::coordinate_on_active_bound(
    const QCQPProblem& problem, const Vec& base, int j,
    const Vec* predicted_step) const {
  const double lower = problem.lb[static_cast<std::size_t>(j)];
  const double upper = problem.ub[static_cast<std::size_t>(j)];
  const bool hard_fixed = std::isfinite(lower) && std::isfinite(upper) &&
      std::abs(upper - lower) <= st_.fixed_bound_tolerance;
  if (hard_fixed || !st_.dca_condensed_bound_aware_normal) return false;
  const bool has_lookahead = predicted_step &&
      predicted_step->size() == base.size();
  const double step = has_lookahead
      ? (*predicted_step)[static_cast<std::size_t>(j)] : 0.0;
  const double endpoint = base[static_cast<std::size_t>(j)] + step;
  const double scale = std::max(
      {1.0, std::abs(base[static_cast<std::size_t>(j)]),
       has_lookahead ? std::abs(endpoint) : 0.0,
       std::isfinite(lower) ? std::abs(lower) : 0.0,
       std::isfinite(upper) ? std::abs(upper) : 0.0});
  const double tolerance = std::max(
      {st_.fixed_bound_tolerance,
       st_.dca_condensed_bound_activity_tolerance * scale,
       10.0 * st_.inner_tolerance * scale,
       64.0 * std::numeric_limits<double>::epsilon() * scale});
  const bool active_at_base =
      (std::isfinite(lower) &&
       base[static_cast<std::size_t>(j)] <= lower + tolerance) ||
      (std::isfinite(upper) &&
       base[static_cast<std::size_t>(j)] >= upper - tolerance);
  if (active_at_base || !has_lookahead) return active_at_base;

  // Do not lock a coordinate merely because it is near a face when the raw
  // tangent step moves it strictly back into the box.
  return (std::isfinite(lower) && step < 0.0 &&
          endpoint <= lower + tolerance) ||
         (std::isfinite(upper) && step > 0.0 &&
          endpoint >= upper - tolerance);
}

int CondensedQuadraticRetraction::active_bound_coordinates(
    const QCQPProblem& problem, const Vec& base) const {
  if (static_cast<int>(base.size()) != problem.n) return 0;
  int count = 0;
  for (int j = 0; j < problem.n; ++j)
    if (coordinate_on_active_bound(problem, base, j)) ++count;
  return count;
}

int CondensedQuadraticRetraction::active_bound_coordinates(
    const QCQPProblem& problem, const Vec& base,
    const Vec& predicted_step) const {
  if (static_cast<int>(base.size()) != problem.n ||
      static_cast<int>(predicted_step.size()) != problem.n)
    return 0;
  int count = 0;
  for (int j = 0; j < problem.n; ++j)
    if (coordinate_on_active_bound(problem, base, j, &predicted_step)) ++count;
  return count;
}

bool CondensedQuadraticRetraction::factor(const QCQPProblem& problem,
                                           const Vec& base,
                                           std::string& message) {
  return factor_impl(problem, base, false, nullptr, message);
}

bool CondensedQuadraticRetraction::factor_bound_aware(
    const QCQPProblem& problem, const Vec& base, std::string& message) {
  return factor_impl(problem, base, true, nullptr, message);
}

bool CondensedQuadraticRetraction::factor_bound_aware(
    const QCQPProblem& problem, const Vec& base,
    const Vec& predicted_step, std::string& message) {
  if (static_cast<int>(predicted_step.size()) != problem.n) {
    message = "condensed retraction look-ahead step dimension mismatch";
    return false;
  }
  return factor_impl(problem, base, true, &predicted_step, message);
}

bool CondensedQuadraticRetraction::factor_impl(
    const QCQPProblem& problem, const Vec& base, bool bound_aware,
    const Vec* predicted_step, std::string& message) {
  factored_ = false;
  problem_ = &problem;
  base_ = base;
  n_ = problem.n;
  me_ = static_cast<int>(problem.equalities.size());
  if (static_cast<int>(base.size()) != n_) {
    message = "condensed retraction base dimension mismatch";
    return false;
  }

  const std::vector<unsigned char> previous_locked = fixed_;
  fixed_.assign(static_cast<std::size_t>(n_), 0);
  locked_coordinates_ = 0;
  for (int j = 0; j < n_; ++j) {
    const double lower = problem.lb[static_cast<std::size_t>(j)];
    const double upper = problem.ub[static_cast<std::size_t>(j)];
    const bool hard_fixed = std::isfinite(lower) && std::isfinite(upper) &&
        std::abs(upper - lower) <= st_.fixed_bound_tolerance;
    const bool locked = hard_fixed ||
        (bound_aware &&
         coordinate_on_active_bound(problem, base, j, predicted_step));
    fixed_[static_cast<std::size_t>(j)] = locked ? 1 : 0;
    if (locked) ++locked_coordinates_;
  }

  // Sparse gradient supports depend on the locked-coordinate mask.  Rebuild
  // only when that mask changes; the QCQP itself remains cached otherwise.
  if (previous_locked != fixed_) equality_sparse_problem_ = nullptr;

  build_sparse_equality_data(problem);
  build_sparse_inequality_data(problem);
  base_values_ = equality_values(base_);
  base_gradients_.clear();
  base_gradients_.resize(static_cast<std::size_t>(me_));
  const bool parallel_rows = parallel_equalities_enabled();
  if (parallel_rows) ++parallel_equality_batches_;
#ifdef LAZYIPDCA_HAVE_OPENMP
#pragma omp parallel for schedule(static) if(parallel_rows)
#endif
  for (int i = 0; i < me_; ++i)
    base_gradients_[static_cast<std::size_t>(i)] = sparse_gradient(i, base_);
  row_scale_.assign(static_cast<std::size_t>(me_), 1.0);
  int nonfinite_row = -1;
#ifdef LAZYIPDCA_HAVE_OPENMP
#pragma omp parallel for schedule(static) if(parallel_rows)
#endif
  for (int i = 0; i < me_; ++i) {
    double nrm2 = 0.0;
    for (const auto& entry :
         base_gradients_[static_cast<std::size_t>(i)].entries)
      nrm2 += entry.second * entry.second;
    const double nrm = std::sqrt(std::max(0.0, nrm2));
    if (!std::isfinite(nrm)) {
#ifdef LAZYIPDCA_HAVE_OPENMP
#pragma omp critical
#endif
      { if (nonfinite_row < 0) nonfinite_row = i; }
    } else {
      row_scale_[static_cast<std::size_t>(i)] =
          nrm > 1e-16 ? 1.0 / nrm : 1.0;
    }
  }
  if (nonfinite_row >= 0) {
    message = "nonfinite equality Jacobian row in condensed retraction";
    return false;
  }

  flow_lift_schur_active_ = false;
  flow_lift_schur_candidate_ = detect_flow_lift_schur_structure(problem);
  normal_factor_dimension_ = n_ + me_;

  if (me_ == 0) {
    regularization_ = 0.0;
    factored_ = true;
    message.clear();
    return true;
  }
  if (!factor_) {
    std::string backend_message;
    factor_ = make_symmetric_factorization(factor_settings_, backend_message);
    if (!factor_) {
      message = "condensed normal factor backend unavailable: " + backend_message;
      return false;
    }
  }

  const double fallback_reg = std::max(1e-18, st_.dca_condensed_regularization);
  const double max_reg = std::max(fallback_reg, st_.dca_condensed_regularization_max);
  const bool cached_seed = st_.dca_condensed_cached_regularization &&
      cached_positive_regularization_streak_ >= 2 &&
      cached_positive_regularization_ > 0.0 &&
      std::isfinite(cached_positive_regularization_);
  double reg = cached_seed ? cached_positive_regularization_ : 0.0;
  bool cached_seed_active = cached_seed;
  if (cached_seed) ++cached_regularization_attempts_;
  std::string last;

  auto factor_matrix = [&](int dimension, int primal_dimension,
                           const std::vector<SymmetricTriplet>& matrix,
                           std::string& current) {
    const int sym_before = factor_->symbolic_analyses();
    const int reuse_before = factor_->symbolic_reuses();
    const int inval_before = factor_->symbolic_invalidations();
    ++factorization_count_;
    const bool success =
        factor_->factor(dimension, primal_dimension, matrix, current);
    symbolic_analyses_ += factor_->symbolic_analyses() - sym_before;
    symbolic_reuses_ += factor_->symbolic_reuses() - reuse_before;
    symbolic_invalidations_ += factor_->symbolic_invalidations() - inval_before;
    return success;
  };

  auto build_full_matrix = [&](double current_reg) {
    std::vector<SymmetricTriplet> matrix;
    std::size_t nnz = static_cast<std::size_t>(n_ + me_);
    for (const auto& row : base_gradients_) nnz += row.entries.size();
    matrix.reserve(nnz);
    // Historical Euclidean minimum-norm bordered operator.
    for (int j = 0; j < n_; ++j) matrix.push_back({j, j, 1.0});
    for (int i = 0; i < me_; ++i) {
      matrix.push_back({n_ + i, n_ + i, -current_reg});
      const double scale = row_scale_[static_cast<std::size_t>(i)];
      const auto& row = base_gradients_[static_cast<std::size_t>(i)];
      for (const auto& [j, value] : row.entries) {
        const double a = scale * value;
        if (a != 0.0) matrix.push_back({j, n_ + i, a});
      }
    }
    return matrix;
  };

  auto build_flow_lift_schur_matrix = [&](double current_reg) {
    const int nz = static_cast<int>(flow_base_variables_.size());
    const int mb = static_cast<int>(flow_core_equalities_.size());
    std::vector<SymmetricTriplet> matrix;
    std::size_t nnz = static_cast<std::size_t>(nz + mb);
    for (const auto& lift : flow_lifts_) {
      const auto& row = base_gradients_[static_cast<std::size_t>(lift.equality)];
      nnz += row.entries.size() * (row.entries.size() + 1) / 2;
    }
    for (int equality : flow_core_equalities_)
      nnz += base_gradients_[static_cast<std::size_t>(equality)].entries.size();
    matrix.reserve(nnz);
    for (int j = 0; j < nz; ++j) matrix.push_back({j, j, 1.0});

    // Eliminating each unique auxiliary and its equality multiplier from
    // [I J_s^T; J_s -reg I] adds C^T w C to the base-variable metric.
    for (const auto& lift : flow_lifts_) {
      const int equality = lift.equality;
      const double scale = row_scale_[static_cast<std::size_t>(equality)];
      const double denominator = scale * scale + current_reg;
      const double weight = scale * scale / denominator;
      const auto& row = base_gradients_[static_cast<std::size_t>(equality)];
      for (std::size_t a = 0; a < row.entries.size(); ++a) {
        const int ja = row.entries[a].first;
        if (ja == lift.variable) continue;
        const int ra = flow_full_to_base_[static_cast<std::size_t>(ja)];
        if (ra < 0) continue;
        for (std::size_t b = a; b < row.entries.size(); ++b) {
          const int jb = row.entries[b].first;
          if (jb == lift.variable) continue;
          const int rb = flow_full_to_base_[static_cast<std::size_t>(jb)];
          if (rb < 0) continue;
          matrix.push_back({std::min(ra, rb), std::max(ra, rb),
                            weight * row.entries[a].second *
                                row.entries[b].second});
        }
      }
    }
    for (int k = 0; k < mb; ++k) {
      const int equality = flow_core_equalities_[static_cast<std::size_t>(k)];
      matrix.push_back({nz + k, nz + k, -current_reg});
      const double scale = row_scale_[static_cast<std::size_t>(equality)];
      for (const auto& [j, value] :
           base_gradients_[static_cast<std::size_t>(equality)].entries) {
        const int reduced = flow_full_to_base_[static_cast<std::size_t>(j)];
        if (reduced >= 0 && value != 0.0)
          matrix.push_back({reduced, nz + k, scale * value});
      }
    }
    return matrix;
  };

  for (int attempt = 0; attempt < 16; ++attempt) {
    std::string current;
    bool used_schur = false;
    bool success = false;
    if (flow_lift_schur_candidate_) {
      const int nz = static_cast<int>(flow_base_variables_.size());
      const int dimension =
          nz + static_cast<int>(flow_core_equalities_.size());
      auto reduced = build_flow_lift_schur_matrix(reg);
      ++flow_lift_schur_factorizations_;
      success = factor_matrix(dimension, nz, reduced, current);
      used_schur = success;
      if (!success) {
        const std::string reduced_message = current;
        auto full = build_full_matrix(reg);
        std::string full_message;
        success = factor_matrix(n_ + me_, n_, full, full_message);
        if (success) ++flow_lift_schur_fallbacks_;
        current = success ? std::string{}
                          : ("flow-lift Schur: " + reduced_message +
                             "; bordered fallback: " + full_message);
      }
    } else {
      auto full = build_full_matrix(reg);
      success = factor_matrix(n_ + me_, n_, full, current);
    }
    if (success) {
      regularization_ = reg;
      flow_lift_schur_active_ = used_schur;
      normal_factor_dimension_ = used_schur
          ? static_cast<int>(flow_base_variables_.size() +
                             flow_core_equalities_.size())
          : n_ + me_;
      if (cached_seed_active) ++cached_regularization_hits_;
      if (reg > 0.0) {
        const double scale = std::max({1.0, std::abs(reg),
                                      std::abs(cached_positive_regularization_)});
        if (cached_positive_regularization_streak_ > 0 &&
            std::abs(reg - cached_positive_regularization_) <=
                8.0 * std::numeric_limits<double>::epsilon() * scale) {
          ++cached_positive_regularization_streak_;
        } else {
          cached_positive_regularization_ = reg;
          cached_positive_regularization_streak_ = 1;
        }
      } else {
        cached_positive_regularization_ = 0.0;
        cached_positive_regularization_streak_ = 0;
      }
      factored_ = true;
      message.clear();
      return true;
    }
    last = current;
    if (cached_seed_active) {
      // The cached positive seed is only an acceleration.  If it ever stops
      // working, immediately restore the historical ladder beginning at zero.
      cached_seed_active = false;
      cached_positive_regularization_streak_ = 0;
      reg = 0.0;
      continue;
    }
    if (reg == 0.0) reg = fallback_reg;
    else reg *= std::max(1.1, st_.dca_condensed_regularization_growth);
    if (reg > max_reg * (1.0 + 1e-12)) break;
  }

  message = "condensed bordered normal factorization failed through regularization " +
      std::to_string(max_reg) + ": " + last;
  return false;
}

bool CondensedQuadraticRetraction::apply_right_inverse(const Vec& rhs,
                                                        Vec& correction,
                                                        std::string& message) const {
  return apply_right_inverse_impl(rhs, correction, message, false);
}

bool CondensedQuadraticRetraction::apply_right_inverse_impl(
    const Vec& rhs, Vec& correction, std::string& message,
    bool use_intermediate_accuracy) const {
  if (!factored_) {
    message = "condensed normal operator is not factored";
    return false;
  }
  if (static_cast<int>(rhs.size()) != me_) {
    message = "condensed normal rhs dimension mismatch";
    return false;
  }
  if (me_ == 0) {
    correction.assign(static_cast<std::size_t>(n_), 0.0);
    message.clear();
    return true;
  }

  bool zero_rhs = true;
  for (double value : rhs) {
    if (value != 0.0) {
      zero_rhs = false;
      break;
    }
  }
  if (zero_rhs) {
    correction.resize(static_cast<std::size_t>(n_));
    std::fill(correction.begin(), correction.end(), 0.0);
    ++zero_rhs_solve_skips_;
    message.clear();
    return true;
  }

  const std::size_t normal_dim = static_cast<std::size_t>(
      flow_lift_schur_active_ ? normal_factor_dimension_ : n_ + me_);
  if (normal_work_.size() != normal_dim)
    normal_work_.assign(normal_dim, 0.0);
  else
    std::fill(normal_work_.begin(), normal_work_.end(), 0.0);
  if (flow_lift_schur_active_) {
    const int nz = static_cast<int>(flow_base_variables_.size());
    for (const auto& lift : flow_lifts_) {
      const int equality = lift.equality;
      const double scale = row_scale_[static_cast<std::size_t>(equality)];
      const double denominator = scale * scale + regularization_;
      const double scaled_rhs =
          scale * rhs[static_cast<std::size_t>(equality)];
      const double coefficient = scale * scaled_rhs / denominator;
      for (const auto& [j, value] :
           base_gradients_[static_cast<std::size_t>(equality)].entries) {
        if (j == lift.variable) continue;
        const int reduced = flow_full_to_base_[static_cast<std::size_t>(j)];
        if (reduced >= 0)
          normal_work_[static_cast<std::size_t>(reduced)] +=
              coefficient * value;
      }
    }
    for (int k = 0; k < static_cast<int>(flow_core_equalities_.size()); ++k) {
      const int equality = flow_core_equalities_[static_cast<std::size_t>(k)];
      normal_work_[static_cast<std::size_t>(nz + k)] =
          row_scale_[static_cast<std::size_t>(equality)] *
          rhs[static_cast<std::size_t>(equality)];
    }
  } else {
    for (int i = 0; i < me_; ++i)
      normal_work_[static_cast<std::size_t>(n_ + i)] =
          row_scale_[static_cast<std::size_t>(i)] *
          rhs[static_cast<std::size_t>(i)];
  }

  // The factor object retains a reference to factor_settings_.  Temporarily
  // changing only this private copy lets intermediate Newton--GMRES products
  // avoid certification-grade iterative refinement without changing the
  // global KKT settings.  The scope guard restores the strict policy before
  // every return, including a backend failure.
  const bool relaxed = use_intermediate_accuracy &&
      st_.dca_condensed_adaptive_normal_refinement &&
      st_.dca_condensed_normal_intermediate_backward_error_tolerance >
          st_.kkt_backward_error_tolerance;
  struct RestoreNormalAccuracy {
    Settings& settings;
    double tolerance;
    double reject;
    ~RestoreNormalAccuracy() {
      settings.kkt_backward_error_tolerance = tolerance;
      settings.kkt_backward_error_reject = reject;
    }
  } restore{factor_settings_, factor_settings_.kkt_backward_error_tolerance,
            factor_settings_.kkt_backward_error_reject};
  if (relaxed) {
    factor_settings_.kkt_backward_error_tolerance =
        st_.dca_condensed_normal_intermediate_backward_error_tolerance;
    factor_settings_.kkt_backward_error_reject =
        st_.dca_condensed_normal_intermediate_backward_error_reject;
    ++relaxed_normal_solves_;
  } else {
    ++strict_normal_solves_;
  }
  ++solve_count_;
  if (!factor_->solve(normal_work_, message)) {
    if (relaxed) relaxed_normal_solve_failed_ = true;
    return false;
  }
  const int refine = factor_->last_refinement_steps();
  normal_refinement_steps_ += refine;
  if (refine == 0) ++normal_refinement_zero_solves_;
  else if (refine == 1) ++normal_refinement_one_solves_;
  else if (refine == 2) ++normal_refinement_two_solves_;
  else ++normal_refinement_three_plus_solves_;
  const double initial_berr = factor_->last_initial_backward_error();
  const double final_berr = factor_->last_backward_error();
  if (relaxed && std::isfinite(final_berr) &&
      final_berr > st_.kkt_backward_error_tolerance)
    ++relaxed_accepts_above_strict_tolerance_;
  max_normal_initial_backward_error_ = std::max(
      max_normal_initial_backward_error_, initial_berr);
  sum_normal_initial_backward_error_ += initial_berr;
  sum_normal_final_backward_error_ += final_berr;
  max_normal_backward_error_ = std::max(max_normal_backward_error_, final_berr);
  if (flow_lift_schur_active_) {
    correction.assign(static_cast<std::size_t>(n_), 0.0);
    for (int k = 0; k < static_cast<int>(flow_base_variables_.size()); ++k)
      correction[static_cast<std::size_t>(
          flow_base_variables_[static_cast<std::size_t>(k)])] =
          normal_work_[static_cast<std::size_t>(k)];
    for (const auto& lift : flow_lifts_) {
      const int equality = lift.equality;
      const double scale = row_scale_[static_cast<std::size_t>(equality)];
      const double denominator = scale * scale + regularization_;
      double c_delta = 0.0;
      for (const auto& [j, value] :
           base_gradients_[static_cast<std::size_t>(equality)].entries)
        if (j != lift.variable)
          c_delta += value * correction[static_cast<std::size_t>(j)];
      const double scaled_rhs =
          scale * rhs[static_cast<std::size_t>(equality)];
      const double lambda =
          (scale * c_delta - scaled_rhs) / denominator;
      correction[static_cast<std::size_t>(lift.variable)] =
          -lift.sign * scale * lambda;
    }
  } else {
    correction.assign(normal_work_.begin(), normal_work_.begin() + n_);
  }
  for (int j = 0; j < n_; ++j)
    if (fixed_[static_cast<std::size_t>(j)]) correction[static_cast<std::size_t>(j)] = 0.0;
  if (!finite_vec(correction)) {
    message = "condensed normal solve produced a nonfinite correction";
    return false;
  }
  message.clear();
  return true;
}

bool CondensedQuadraticRetraction::project_tangent(const Vec& v,
                                                    Vec& projected,
                                                    std::string& message) const {
  if (static_cast<int>(v.size()) != n_) {
    message = "condensed tangent projection dimension mismatch";
    return false;
  }
  projected = v;
  for (int j = 0; j < n_; ++j)
    if (fixed_[static_cast<std::size_t>(j)]) projected[static_cast<std::size_t>(j)] = 0.0;
  if (me_ == 0) {
    message.clear();
    return true;
  }

  // Two projection passes suppress residual normal components when a tiny
  // rank-deficiency regularization was required in the bordered factorization.
  for (int refinement = 0; refinement < 2; ++refinement) {
    const Vec normal_residual = base_jacobian_product(projected);
    if (inf_norm(normal_residual) <=
        128.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, inf_norm(projected)))
      break;
    Vec normal;
    if (!apply_right_inverse(normal_residual, normal, message)) return false;
    axpy(-1.0, normal, projected);
    for (int j = 0; j < n_; ++j)
      if (fixed_[static_cast<std::size_t>(j)]) projected[static_cast<std::size_t>(j)] = 0.0;
  }
  if (!finite_vec(projected)) {
    message = "condensed tangent projection produced a nonfinite vector";
    return false;
  }
  message.clear();
  return true;
}

bool CondensedQuadraticRetraction::solve_scalar_quadratic(
    const Vec& step, double tau, double target, double previous_z,
    double& z, std::string& message) const {
  Vec s = step;
  for (double& v : s) v *= tau;
  Vec point = base_;
  axpy(1.0, s, point);

  Vec unit(1, 1.0), y;
  if (!apply_right_inverse(unit, y, message)) return false;
  const QuadraticFunction& h = problem_->equalities.front();
  const double a = 0.5 * h.hessian.quad(y);
  const double b = dot(h.gradient(point), y);
  const double c = h.value(point) - target;
  const double scale = std::max({1.0, std::abs(a), std::abs(b), std::abs(c)});
  const double zero = 128.0 * std::numeric_limits<double>::epsilon() * scale;

  std::vector<double> roots;
  if (std::abs(a) <= zero) {
    if (std::abs(b) <= zero) {
      if (std::abs(c) <= std::max(st_.dca_condensed_retraction_tolerance, zero)) {
        z = previous_z;
        message.clear();
        return true;
      }
      message = "scalar condensed equation is locally constant and inconsistent";
      return false;
    }
    roots.push_back(-c / b);
  } else {
    double disc = b * b - 4.0 * a * c;
    const double disc_tol = 256.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, b * b, std::abs(4.0 * a * c)});
    if (disc < -disc_tol) {
      message = "scalar condensed quadratic has no real local branch";
      return false;
    }
    disc = std::max(0.0, disc);
    const double root_disc = std::sqrt(disc);
    const double q = -0.5 * (b + std::copysign(root_disc, b == 0.0 ? 1.0 : b));
    if (q != 0.0) {
      roots.push_back(q / a);
      roots.push_back(c / q);
    } else {
      roots.push_back((-b + root_disc) / (2.0 * a));
      roots.push_back((-b - root_disc) / (2.0 * a));
    }
  }

  double best = std::numeric_limits<double>::infinity();
  double selected = previous_z;
  for (double r : roots) {
    if (!std::isfinite(r)) continue;
    const double distance = std::abs(r - previous_z);
    if (distance < best) {
      best = distance;
      selected = r;
    }
  }
  if (!std::isfinite(best)) {
    message = "scalar condensed quadratic produced no finite root";
    return false;
  }
  z = selected;
  message.clear();
  return true;
}

bool CondensedQuadraticRetraction::gmres(const Vec& current_point,
                                          const Vec& rhs, Vec& solution,
                                          Vec* mapped_solution,
                                          int& iterations,
                                          std::string& message,
                                          bool use_intermediate_accuracy) const {
  const int m = me_;
  solution.resize(static_cast<std::size_t>(m));
  std::fill(solution.begin(), solution.end(), 0.0);
  if (mapped_solution) {
    mapped_solution->resize(static_cast<std::size_t>(n_));
    std::fill(mapped_solution->begin(), mapped_solution->end(), 0.0);
  }
  iterations = 0;
  if (m == 0) return true;
  const double beta = norm2(rhs);
  if (!(beta > 0.0)) return true;

  const int kmax = std::max(1, std::min(m, st_.dca_condensed_krylov_iterations));
  const double tol = std::max(
      st_.dca_condensed_retraction_tolerance * 0.1,
      st_.dca_condensed_krylov_relative_tolerance * beta);

  bool grew = false;
  if (static_cast<int>(gmres_v_.size()) < kmax + 1) {
    gmres_v_.resize(static_cast<std::size_t>(kmax + 1));
    grew = true;
  }
  for (int i = 0; i <= kmax; ++i) {
    if (static_cast<int>(gmres_v_[static_cast<std::size_t>(i)].size()) != m) {
      gmres_v_[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(m));
      grew = true;
    }
  }
  if (mapped_solution) {
    if (static_cast<int>(gmres_mapped_basis_.size()) < kmax) {
      gmres_mapped_basis_.resize(static_cast<std::size_t>(kmax));
      grew = true;
    }
    for (int i = 0; i < kmax; ++i) {
      if (static_cast<int>(gmres_mapped_basis_[static_cast<std::size_t>(i)].size()) != n_) {
        gmres_mapped_basis_[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_));
        grew = true;
      }
    }
  }
  const std::size_t hsize = static_cast<std::size_t>(kmax + 1) *
                            static_cast<std::size_t>(kmax);
  if (gmres_h_.size() != hsize) { gmres_h_.resize(hsize); grew = true; }
  if (static_cast<int>(gmres_cs_.size()) != kmax) { gmres_cs_.resize(kmax); grew = true; }
  if (static_cast<int>(gmres_sn_.size()) != kmax) { gmres_sn_.resize(kmax); grew = true; }
  if (static_cast<int>(gmres_g_.size()) != kmax + 1) { gmres_g_.resize(kmax + 1); grew = true; }
  if (static_cast<int>(gmres_y_.size()) != kmax) { gmres_y_.resize(kmax); grew = true; }
  if (static_cast<int>(gmres_point_delta_.size()) != n_) { gmres_point_delta_.resize(n_); grew = true; }
  if (static_cast<int>(gmres_normal_.size()) != n_) { gmres_normal_.resize(n_); grew = true; }
  if (static_cast<int>(gmres_w_.size()) != m) { gmres_w_.resize(m); grew = true; }
  if (grew) ++gmres_workspace_growths_;

  std::fill(gmres_h_.begin(), gmres_h_.end(), 0.0);
  std::fill(gmres_cs_.begin(), gmres_cs_.end(), 0.0);
  std::fill(gmres_sn_.begin(), gmres_sn_.end(), 0.0);
  std::fill(gmres_g_.begin(), gmres_g_.end(), 0.0);
  std::fill(gmres_y_.begin(), gmres_y_.end(), 0.0);
  auto H = [&](int i, int j) -> double& {
    return gmres_h_[static_cast<std::size_t>(i) * static_cast<std::size_t>(kmax) +
                    static_cast<std::size_t>(j)];
  };

  gmres_v_[0] = rhs;
  for (double& a : gmres_v_[0]) a /= beta;
  gmres_g_[0] = beta;
  gmres_point_delta_ = current_point;
  axpy(-1.0, base_, gmres_point_delta_);
  int used = 0;

  for (int j = 0; j < kmax; ++j) {
    if (!apply_right_inverse_impl(gmres_v_[static_cast<std::size_t>(j)],
                                  gmres_normal_, message,
                                  use_intermediate_accuracy)) return false;
    if (mapped_solution)
      gmres_mapped_basis_[static_cast<std::size_t>(j)] = gmres_normal_;
    jacobian_product_from_delta_into(gmres_point_delta_, gmres_normal_, gmres_w_);

    for (int i = 0; i <= j; ++i) {
      H(i,j) = dot(gmres_v_[static_cast<std::size_t>(i)], gmres_w_);
      axpy(-H(i,j), gmres_v_[static_cast<std::size_t>(i)], gmres_w_);
    }
    // Preserve the historical second MGS pass exactly; only storage is reused.
    for (int i = 0; i <= j; ++i) {
      const double h2 = dot(gmres_v_[static_cast<std::size_t>(i)], gmres_w_);
      H(i,j) += h2;
      axpy(-h2, gmres_v_[static_cast<std::size_t>(i)], gmres_w_);
    }
    H(j+1,j) = norm2(gmres_w_);
    auto& next = gmres_v_[static_cast<std::size_t>(j+1)];
    if (H(j+1,j) > 1e-14) {
      next = gmres_w_;
      for (double& a : next) a /= H(j+1,j);
    } else {
      std::fill(next.begin(), next.end(), 0.0);
    }

    for (int i = 0; i < j; ++i) {
      const double h0 = H(i,j);
      const double h1 = H(i+1,j);
      H(i,j) = gmres_cs_[static_cast<std::size_t>(i)] * h0 +
               gmres_sn_[static_cast<std::size_t>(i)] * h1;
      H(i+1,j) = -gmres_sn_[static_cast<std::size_t>(i)] * h0 +
                  gmres_cs_[static_cast<std::size_t>(i)] * h1;
    }

    const double a = H(j,j);
    const double b = H(j+1,j);
    const double r = std::hypot(a,b);
    gmres_cs_[static_cast<std::size_t>(j)] = r > 0.0 ? a/r : 1.0;
    gmres_sn_[static_cast<std::size_t>(j)] = r > 0.0 ? b/r : 0.0;
    H(j,j) = r;
    H(j+1,j) = 0.0;

    const double gj = gmres_g_[static_cast<std::size_t>(j)];
    gmres_g_[static_cast<std::size_t>(j)] =
        gmres_cs_[static_cast<std::size_t>(j)] * gj;
    gmres_g_[static_cast<std::size_t>(j+1)] =
        -gmres_sn_[static_cast<std::size_t>(j)] * gj;
    used = j + 1;
    ++iterations;
    if (std::abs(gmres_g_[static_cast<std::size_t>(j+1)]) <= tol) break;
  }

  if (used == 0) {
    message = "condensed GMRES built no Krylov direction";
    return false;
  }
  std::fill(gmres_y_.begin(), gmres_y_.end(), 0.0);
  for (int i = used - 1; i >= 0; --i) {
    double sum = gmres_g_[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < used; ++j)
      sum -= H(i,j) * gmres_y_[static_cast<std::size_t>(j)];
    const double diag = H(i,i);
    if (std::abs(diag) <= 1e-16 || !std::isfinite(diag)) {
      message = "condensed GMRES triangular solve is singular";
      return false;
    }
    gmres_y_[static_cast<std::size_t>(i)] = sum/diag;
  }
  for (int j = 0; j < used; ++j) {
    axpy(gmres_y_[static_cast<std::size_t>(j)],
         gmres_v_[static_cast<std::size_t>(j)], solution);
    if (mapped_solution)
      axpy(gmres_y_[static_cast<std::size_t>(j)],
           gmres_mapped_basis_[static_cast<std::size_t>(j)], *mapped_solution);
  }
  if (!finite_vec(solution) || (mapped_solution && !finite_vec(*mapped_solution))) {
    message = "condensed GMRES produced a nonfinite Newton coordinate";
    return false;
  }
  message.clear();
  return true;
}

bool CondensedQuadraticRetraction::solve_condensed_newton(
    const Vec& step, double tau, const Vec& targets, Vec& z,
    CondensedRetractionResult& diagnostics, std::string& message,
    bool use_intermediate_accuracy) const {
  Vec& scaled_step = newton_scaled_step_;
  scaled_step = step;
  for (double& v : scaled_step) v *= tau;

  const double target_scale = std::max(1.0, max_abs_value(targets));
  const double tolerance = std::max(
      st_.dca_condensed_retraction_tolerance,
      kTiny * target_scale);

  // Reuse all Newton/line-search storage across iterations.  This preserves
  // the historical arithmetic but removes thousands of short-lived vectors.
  Vec& correction = newton_correction_;
  Vec& point = newton_point_;
  Vec& residual = newton_residual_;
  Vec& rhs = newton_rhs_;
  Vec& dz = newton_dz_;
  Vec& mapped_dz = newton_mapped_dz_;
  Vec& slope = newton_slope_;
  Vec& curvature = newton_curvature_;
  Vec& ztrial = newton_ztrial_;
  Vec& ptrial = newton_ptrial_;
  Vec& ctrial = newton_ctrial_;
  Vec& rtrial = newton_rtrial_;
  const bool need_mapped_direction =
      st_.dca_condensed_affine_normal_reuse ||
      st_.dca_condensed_quadratic_linesearch_reuse;

  // Materialize the exact state once.  Every accepted line-search endpoint
  // below is checked by equality_values_into and becomes the next Newton
  // state directly, so rebuilding Y*z and reevaluating the same point at the
  // top of the following iteration would be redundant.  retract_impl still
  // performs its independent final reconstruction and certification.
  if (!apply_right_inverse_impl(z, correction, message,
                                use_intermediate_accuracy)) return false;
  point = base_;
  axpy(1.0, scaled_step, point);
  axpy(1.0, correction, point);
  equality_values_into(point, residual);
  for (int i = 0; i < me_; ++i)
    residual[static_cast<std::size_t>(i)] -=
        targets[static_cast<std::size_t>(i)];

  for (int it = 0; it < st_.dca_condensed_newton_iterations; ++it) {
    const double r0 = inf_norm(residual);
    if (r0 <= tolerance) {
      message.clear();
      return true;
    }

    ++diagnostics.newton_iterations;
    rhs = residual;
    for (double& a : rhs) a = -a;
    int krylov = 0;
    bool have_direction = gmres(
        point, rhs, dz, need_mapped_direction ? &mapped_dz : nullptr,
        krylov, message, use_intermediate_accuracy);
    diagnostics.krylov_iterations += krylov;
    if (!have_direction) {
      dz = rhs;
      message.clear();
      if (need_mapped_direction &&
          !apply_right_inverse_impl(dz, mapped_dz, message,
                                    use_intermediate_accuracy)) return false;
    }

    auto line_search = [&](const Vec& search_dz,
                           const Vec& mapped_search_dz) -> int {
      auto exact_check = [&](double alpha) -> int {
        ztrial = z;
        axpy(alpha, search_dz, ztrial);
        if (st_.dca_condensed_affine_normal_reuse) {
          ptrial = point;
          axpy(alpha, mapped_search_dz, ptrial);
          ++diagnostics.affine_normal_map_reuses;
        } else {
          if (!apply_right_inverse_impl(ztrial, ctrial, message,
                                        use_intermediate_accuracy)) return -1;
          ptrial = base_;
          axpy(1.0, scaled_step, ptrial);
          axpy(1.0, ctrial, ptrial);
        }
        equality_values_into(ptrial, rtrial);
        if (st_.dca_condensed_quadratic_linesearch_reuse)
          ++diagnostics.quadratic_linesearch_exact_checks;
        for (int i = 0; i < me_; ++i)
          rtrial[static_cast<std::size_t>(i)] -=
              targets[static_cast<std::size_t>(i)];
        const double rt = inf_norm(rtrial);
        if (!(std::isfinite(rt) &&
              (rt <= tolerance || rt < (1.0 - 1e-4 * alpha) * r0)))
          return 0;

        z = ztrial;
        // Both buffers contain values produced by the exact endpoint check.
        // Swapping avoids an O(n+m) copy while retaining their allocations for
        // the next candidate.
        point.swap(ptrial);
        residual.swap(rtrial);
        ++diagnostics.newton_endpoint_reuses;
        return 1;
      };

      // Full Newton steps are common near convergence.  Certify alpha=1
      // immediately; only a rejection justifies assembling and scanning the
      // quadratic residual predictor for contracted candidates.
      if (st_.dca_condensed_quadratic_linesearch_reuse)
        ++diagnostics.quadratic_linesearch_reuses;
      int exact_status = exact_check(1.0);
      if (exact_status != 0) return exact_status;
      ++diagnostics.line_search_backtracks;

      if (st_.dca_condensed_quadratic_linesearch_reuse) {
        ++diagnostics.quadratic_linesearch_predictor_builds;
        jacobian_product_at_into(point, mapped_search_dz, slope);
        curvature.resize(static_cast<std::size_t>(me_));
        std::fill(curvature.begin(), curvature.end(), 0.0);
        const bool parallel = parallel_equalities_enabled();
        if (parallel) ++parallel_equality_batches_;
#ifdef LAZYIPDCA_HAVE_OPENMP
#pragma omp parallel for schedule(static) if(parallel)
#endif
        for (int i = 0; i < me_; ++i) {
          const auto& h = problem_->equalities[static_cast<std::size_t>(i)];
          curvature[static_cast<std::size_t>(i)] =
              0.5 * symmetric_bilinear(h.hessian, mapped_search_dz,
                                       mapped_search_dz);
        }
      }

      double alpha = 0.5;
      for (int ls = 1; ls < 14; ++ls) {
        bool needs_exact_check = true;
        if (st_.dca_condensed_quadratic_linesearch_reuse) {
          double rt_poly = 0.0;
          const double alpha2 = alpha * alpha;
          bool screen_passes = true;
          const double armijo_bound = (1.0 - 1e-4 * alpha) * r0;
          const double accept_bound = std::max(tolerance, armijo_bound);
          constexpr double roundoff_factor =
              256.0 * std::numeric_limits<double>::epsilon();
          for (int i = 0; i < me_; ++i) {
            const double ri = residual[static_cast<std::size_t>(i)] +
                alpha * slope[static_cast<std::size_t>(i)] +
                alpha2 * curvature[static_cast<std::size_t>(i)];
            rt_poly = std::max(rt_poly, std::abs(ri));
            const double partial_guard = roundoff_factor *
                std::max({1.0, r0, rt_poly});
            if (rt_poly > accept_bound + partial_guard) {
              screen_passes = false;
              ++diagnostics.quadratic_linesearch_early_rejections;
              diagnostics.quadratic_linesearch_rows_skipped += me_ - i - 1;
              break;
            }
          }
          ++diagnostics.quadratic_linesearch_reuses;
          const double roundoff_guard =
              roundoff_factor *
              std::max({1.0, r0, rt_poly});
          needs_exact_check = screen_passes &&
              std::isfinite(rt_poly) && rt_poly <= accept_bound + roundoff_guard;
        }

        if (needs_exact_check) {
          exact_status = exact_check(alpha);
          if (exact_status != 0) return exact_status;
        }

        ++diagnostics.line_search_backtracks;
        alpha *= 0.5;
      }
      return 0;
    };

    int ls_status = line_search(dz, mapped_dz);
    if (ls_status < 0) return false;
    bool accepted = ls_status > 0;
    if (!accepted) {
      dz = rhs;
      if (need_mapped_direction &&
          !apply_right_inverse_impl(dz, mapped_dz, message,
                                    use_intermediate_accuracy)) return false;
      ls_status = line_search(dz, mapped_dz);
      if (ls_status < 0) return false;
      accepted = ls_status > 0;
    }
    if (!accepted) {
      message = "condensed equality Newton step failed to reduce the exact quadratic residual";
      return false;
    }
  }

  if (inf_norm(residual) <= tolerance) {
    message.clear();
    return true;
  }
  message = "condensed equality Newton iteration limit reached";
  return false;
}


CondensedRetractionResult CondensedQuadraticRetraction::retract(
    const Vec& step, const Vec& final_targets,
    bool allow_adaptive_normal) const {
  const bool use_intermediate_accuracy = allow_adaptive_normal &&
      st_.dca_condensed_adaptive_normal_refinement &&
      st_.dca_condensed_normal_intermediate_backward_error_tolerance >
          st_.kkt_backward_error_tolerance;
  if (!use_intermediate_accuracy)
    return retract_impl(step, final_targets, false);

  CondensedRetractionResult relaxed =
      retract_impl(step, final_targets, true);
  if (relaxed.success && !relaxed_normal_solve_failed_) return relaxed;

  // A relaxed solve is only an acceleration.  Any failure replays the whole
  // continuation from z=0 with the historical strict factor policy.  This is
  // deliberately coarser than retrying one RHS: it prevents a partly relaxed
  // Newton path from leaking into the accepted endpoint.
  ++strict_retry_retractions_;
  CondensedRetractionResult strict =
      retract_impl(step, final_targets, false);
  add_retraction_work(relaxed, strict);
  if (!strict.success && !relaxed.message.empty()) {
    strict.message += " (after adaptive normal retry: " + relaxed.message + ")";
  }
  return strict;
}

CondensedRetractionResult CondensedQuadraticRetraction::retract_impl(
    const Vec& step, const Vec& final_targets,
    bool use_intermediate_accuracy) const {
  if (use_intermediate_accuracy) relaxed_normal_solve_failed_ = false;
  CondensedRetractionResult out;
  out.x = base_;
  if (!factored_) {
    out.message = "condensed normal operator is not factored";
    return out;
  }
  if (static_cast<int>(step.size()) != n_) {
    out.message = "condensed retraction step dimension mismatch";
    return out;
  }

  Vec& targets = continuation_targets_;
  targets = final_targets;
  if (targets.empty()) targets.assign(static_cast<std::size_t>(me_), 0.0);
  if (static_cast<int>(targets.size()) != me_) {
    out.message = "condensed retraction target dimension mismatch";
    return out;
  }
  if (me_ == 0) {
    axpy(1.0, step, out.x);
    const Vec no_equalities;
    out.feasibility = feasibility_from_equality_values(
        out.x, no_equalities, &out.inequality_values);
    out.success = finite_vec(out.x);
    out.message = out.success ? "no nonlinear equality to retract" : "nonfinite trial";
    return out;
  }

  const int solves_before = solve_count_;
  Vec& z = continuation_z_;
  z.resize(static_cast<std::size_t>(me_));
  std::fill(z.begin(), z.end(), 0.0);
  Vec& stage_targets = continuation_stage_targets_;
  stage_targets.resize(static_cast<std::size_t>(me_));
  Vec& trial_z = continuation_trial_z_;
  double tau = 0.0;
  double delta = 1.0;
  std::string last_message;
  const int max_attempts = std::max(1, st_.dca_condensed_continuation_attempts);

  while (tau < 1.0 - 8.0 * std::numeric_limits<double>::epsilon() &&
         out.continuation_attempts < max_attempts) {
    const double next = std::min(1.0, tau + delta);
    for (int i = 0; i < me_; ++i)
      stage_targets[static_cast<std::size_t>(i)] =
          (1.0 - next) * base_values_[static_cast<std::size_t>(i)] +
          next * targets[static_cast<std::size_t>(i)];

    trial_z = z;
    bool ok = false;
    ++out.continuation_attempts;
    if (me_ == 1) {
      double scalar = trial_z[0];
      ok = solve_scalar_quadratic(step, next, stage_targets[0], z[0],
                                  scalar, last_message);
      if (ok) trial_z[0] = scalar;
    } else {
      ok = solve_condensed_newton(step, next, stage_targets, trial_z,
                                  out, last_message,
                                  use_intermediate_accuracy);
    }

    if (!ok && use_intermediate_accuracy && relaxed_normal_solve_failed_) {
      out.normal_solves = solve_count_ - solves_before;
      out.message = "adaptive condensed normal solve failed: " + last_message;
      return out;
    }

    if (ok) {
      // Keep both persistent buffers allocated across continuation stages.
      z = trial_z;
      tau = next;
      ++out.continuation_acceptances;
      delta = std::min(1.0 - tau, std::max(delta, 2.0 * delta));
    } else {
      delta *= 0.5;
      if (delta < std::max(st_.minimum_step_length, 1e-10)) break;
    }
  }

  out.normal_solves = solve_count_ - solves_before;
  if (tau < 1.0 - 8.0 * std::numeric_limits<double>::epsilon()) {
    out.message = "condensed branch continuation failed before tau=1: " + last_message;
    return out;
  }

  Vec correction;
  if (!apply_right_inverse(z, correction, out.message)) {
    out.normal_solves = solve_count_ - solves_before;
    return out;
  }
  out.x = base_;
  axpy(1.0, step, out.x);
  axpy(1.0, correction, out.x);
  out.correction_inf = inf_norm(correction);
  out.incremental_correction_inf = out.correction_inf;
  out.normal_coordinate_inf = inf_norm(z);
  Vec residual = equality_values(out.x);
  out.feasibility = feasibility_from_equality_values(
      out.x, residual, &out.inequality_values);
  for (int i = 0; i < me_; ++i)
    residual[static_cast<std::size_t>(i)] -= targets[static_cast<std::size_t>(i)];
  out.equality_residual_inf = inf_norm(residual);
  const double scale = std::max({1.0, max_abs_value(base_values_),
                                 max_abs_value(targets)});
  const double tolerance = std::max(st_.dca_condensed_retraction_tolerance,
                                    kTiny * scale);
  const double ratio_denominator = std::max(
      {st_.minimum_step_length, inf_norm(step),
       st_.dca_condensed_retraction_tolerance});
  out.correction_ratio = out.correction_inf / ratio_denominator;

  // The continuation is anchored at h(base), whereas the final target is
  // normally zero.  If base is feasible only to the requested numerical
  // tolerance, Y(target-h(base)) is a step-independent normal correction.
  // Comparing that term with alpha*d makes the ratio diverge as alpha -> 0
  // and caused CATENA's terminal no-op backtracking cascade.  Remove only the
  // fixed-normal linear prediction, and only in this near-feasible regime;
  // all ordinary retractions keep the historical total-correction guard.
  double base_target_residual_inf = 0.0;
  for (int i = 0; i < me_; ++i)
    base_target_residual_inf = std::max(
        base_target_residual_inf,
        std::abs(base_values_[static_cast<std::size_t>(i)] -
                 targets[static_cast<std::size_t>(i)]));
  if (out.correction_ratio > st_.dca_condensed_retraction_max_ratio &&
      base_target_residual_inf <= tolerance) {
    Vec baseline_rhs(static_cast<std::size_t>(me_), 0.0);
    for (int i = 0; i < me_; ++i)
      baseline_rhs[static_cast<std::size_t>(i)] =
          targets[static_cast<std::size_t>(i)] -
          base_values_[static_cast<std::size_t>(i)];
    Vec baseline_correction;
    std::string baseline_message;
    if (apply_right_inverse(baseline_rhs, baseline_correction,
                            baseline_message)) {
      out.baseline_correction_inf = inf_norm(baseline_correction);
      Vec incremental_correction = correction;
      axpy(-1.0, baseline_correction, incremental_correction);
      out.incremental_correction_inf = inf_norm(incremental_correction);
      out.correction_ratio = out.incremental_correction_inf /
                             ratio_denominator;
    }
  }
  out.normal_solves = solve_count_ - solves_before;
  if (!finite_vec(out.x) || !std::isfinite(out.equality_residual_inf)) {
    out.message = "condensed retraction produced a nonfinite point";
    return out;
  }
  if (out.equality_residual_inf > tolerance) {
    out.message = "condensed retraction did not satisfy the exact quadratic equality target";
    return out;
  }
  if (out.correction_ratio > st_.dca_condensed_retraction_max_ratio) {
    out.message = "condensed normal correction exceeded the configured step ratio";
    return out;
  }
  out.success = true;
  out.message = exact_right_inverse()
      ? "exact fixed-normal condensed quadratic retraction"
      : "regularized fixed-normal condensed quadratic retraction";
  return out;
}

Vec CondensedQuadraticRetraction::apply_model(
    const SymMatrix& hessian, const Convexification& convexification,
    const Vec& v) const {
  Vec y = hessian.apply(v);
  if (convexification.scalar_shift != 0.0)
    for (int j = 0; j < n_; ++j)
      y[static_cast<std::size_t>(j)] +=
          convexification.scalar_shift * v[static_cast<std::size_t>(j)];
  for (const auto& term : convexification.terms) {
    const double coefficient = term.beta * dot(term.direction, v);
    axpy(coefficient, term.direction, y);
  }
  return y;
}

bool CondensedQuadraticRetraction::certify_tangent_convex_by_inertia(
    const SymMatrix& hessian, const Convexification& convexification,
    double floor, std::string& message) {
  if (!factored_) {
    message = "cannot certify tangent inertia before factorizing the normal operator";
    return false;
  }
  if (!exact_right_inverse()) {
    // A regularized normal factor means LICQ was not established by the
    // projector factorization.  Tiny numerical pivots in a duplicated J row
    // must not be mistaken for a valid bordered-inertia certificate.
    message = "tangent inertia certificate requires an unregularized full-row-rank Jacobian";
    return false;
  }
  if (hessian.n != n_) {
    message = "tangent inertia Hessian dimension mismatch";
    return false;
  }
  if (!std::isfinite(floor) ||
      !std::isfinite(convexification.scalar_shift)) {
    message = "nonfinite tangent inertia threshold or scalar shift";
    return false;
  }

  std::vector<int> free_index(static_cast<std::size_t>(n_), -1);
  int nfree = 0;
  for (int j = 0; j < n_; ++j)
    if (!fixed_[static_cast<std::size_t>(j)])
      free_index[static_cast<std::size_t>(j)] = nfree++;
  if (nfree < me_) {
    message = "tangent inertia certificate has fewer free variables than equalities";
    return false;
  }

  int positive_terms = 0;
  for (const auto& term : convexification.terms) {
    if (!std::isfinite(term.beta) || term.beta < 0.0 ||
        static_cast<int>(term.direction.size()) != n_) {
      message = "invalid low-rank term in tangent inertia certificate";
      return false;
    }
    if (term.beta > 0.0) ++positive_terms;
  }

  const int total = nfree + me_ + positive_terms;
  std::vector<SymmetricTriplet> K;
  std::size_t reserve = static_cast<std::size_t>(total) +
      hessian.entries.size();
  for (const auto& row : base_gradients_) reserve += row.entries.size();
  reserve += static_cast<std::size_t>(positive_terms) * nfree;
  K.reserve(reserve);

  for (const auto& entry : hessian.entries) {
    const int i = free_index[static_cast<std::size_t>(entry.i)];
    const int j = free_index[static_cast<std::size_t>(entry.j)];
    if (i >= 0 && j >= 0 && entry.v != 0.0)
      K.push_back({i, j, entry.v});
  }
  const double diagonal_shift = convexification.scalar_shift - floor;
  for (int j = 0; j < nfree; ++j)
    K.push_back({j, j, diagonal_shift});

  // Nonzero row scaling is a congruence and leaves ker(J) unchanged.
  for (int i = 0; i < me_; ++i) {
    const int dual = nfree + i;
    K.push_back({dual, dual, 0.0});
    const double scale = row_scale_[static_cast<std::size_t>(i)];
    for (const auto& entry :
         base_gradients_[static_cast<std::size_t>(i)].entries) {
      const int j = free_index[static_cast<std::size_t>(entry.first)];
      const double value = scale * entry.second;
      if (j >= 0 && value != 0.0) K.push_back({j, dual, value});
    }
  }

  // Eliminating a -I border adds beta*u*u^T without forming it densely.
  int term_index = 0;
  for (const auto& term : convexification.terms) {
    if (!(term.beta > 0.0)) continue;
    const int border = nfree + me_ + term_index++;
    const double root_beta = std::sqrt(term.beta);
    if (!std::isfinite(root_beta)) {
      message = "nonfinite low-rank scale in tangent inertia certificate";
      return false;
    }
    for (int original = 0; original < n_; ++original) {
      const int j = free_index[static_cast<std::size_t>(original)];
      const double value = root_beta *
          term.direction[static_cast<std::size_t>(original)];
      if (j >= 0 && value != 0.0) K.push_back({j, border, value});
    }
    K.push_back({border, border, -1.0});
  }

  if (!tangent_certificate_factor_) {
    std::string backend_message;
    tangent_certificate_factor_ = make_symmetric_factorization(
        tangent_certificate_settings_, backend_message);
    if (!tangent_certificate_factor_) {
      message = "tangent inertia backend unavailable: " + backend_message;
      return false;
    }
  }
  const int symbolic_before = tangent_certificate_factor_->symbolic_analyses();
  const int reuse_before = tangent_certificate_factor_->symbolic_reuses();
  const int invalid_before = tangent_certificate_factor_->symbolic_invalidations();
  ++tangent_inertia_factorizations_;
  const bool success = tangent_certificate_factor_->factor(
      total, nfree, K, message);
  tangent_inertia_symbolic_analyses_ +=
      tangent_certificate_factor_->symbolic_analyses() - symbolic_before;
  tangent_inertia_symbolic_reuses_ +=
      tangent_certificate_factor_->symbolic_reuses() - reuse_before;
  tangent_inertia_symbolic_invalidations_ +=
      tangent_certificate_factor_->symbolic_invalidations() - invalid_before;
  if (!success) return false;
  message.clear();
  return true;
}

bool CondensedQuadraticRetraction::smallest_projected_eigenpair(
    const SymMatrix& hessian, const Convexification& convexification,
    double& lambda, Vec& vector, std::string& message) {
  ++tangent_curvature_calls_;
  const int mmax = std::max(2, std::min(n_, st_.dca_condensed_lanczos_steps));
  const int starts = std::max(1, st_.dca_condensed_lanczos_starts);
  std::normal_distribution<double> normal(0.0, 1.0);
  lambda = std::numeric_limits<double>::infinity();
  vector.assign(static_cast<std::size_t>(n_), 0.0);
  bool found_tangent = false;

  for (int start = 0; start < starts; ++start) {
    Vec q(static_cast<std::size_t>(n_), 0.0);
    for (double& a : q) a = normal(rng_);
    if (!project_tangent(q, q, message)) return false;
    if (norm2(q) <= 1e-13) {
      // Deterministic coordinate probes identify a nontrivial tangent space
      // even if a random vector was almost entirely normal.
      bool coordinate_found = false;
      for (int j = 0; j < n_; ++j) {
        Vec e(static_cast<std::size_t>(n_), 0.0);
        e[static_cast<std::size_t>(j)] = 1.0;
        if (!project_tangent(e, q, message)) return false;
        if (norm2(q) > 1e-13) {
          coordinate_found = true;
          break;
        }
      }
      if (!coordinate_found) continue;
    }
    normalize(q);
    found_tangent = true;

    std::vector<Vec> basis;
    basis.reserve(static_cast<std::size_t>(mmax));
    Vec alpha;
    Vec beta;
    Vec qprev(static_cast<std::size_t>(n_), 0.0);
    double bprev = 0.0;

    for (int k = 0; k < mmax; ++k) {
      basis.push_back(q);
      Vec z = apply_model(hessian, convexification, q);
      if (!project_tangent(z, z, message)) return false;
      if (k > 0) axpy(-bprev, qprev, z);
      const double a = dot(q, z);
      alpha.push_back(a);
      axpy(-a, q, z);
      for (const auto& v : basis) {
        const double projection = dot(v, z);
        axpy(-projection, v, z);
      }
      // Reproject after finite-precision orthogonalization so every retained
      // Ritz direction remains in ker(J_base), including after regularization.
      if (!project_tangent(z, z, message)) return false;
      const double b = norm2(z);
      if (k + 1 >= mmax || b <= 1e-14) break;
      beta.push_back(b);
      qprev = q;
      q = z;
      for (double& a2 : q) a2 /= b;
      bprev = b;
    }

    const int m = static_cast<int>(alpha.size());
    if (m == 0) continue;
    std::vector<double> T(static_cast<std::size_t>(m) * m, 0.0);
    for (int i = 0; i < m; ++i)
      T[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m] =
          alpha[static_cast<std::size_t>(i)];
    for (int i = 0; i + 1 < m; ++i) {
      const double b = beta[static_cast<std::size_t>(i)];
      T[static_cast<std::size_t>(i) + static_cast<std::size_t>(i + 1) * m] = b;
      T[static_cast<std::size_t>(i + 1) + static_cast<std::size_t>(i) * m] = b;
    }
    auto decomposition = dense_eigendecomposition(std::move(T), m);
    if (decomposition.first.empty()) continue;
    Vec ritz(static_cast<std::size_t>(n_), 0.0);
    const Vec& coefficient = decomposition.second.front();
    for (int j = 0; j < m; ++j)
      axpy(coefficient[static_cast<std::size_t>(j)],
           basis[static_cast<std::size_t>(j)], ritz);
    if (!project_tangent(ritz, ritz, message)) return false;
    if (norm2(ritz) <= 1e-14) continue;
    normalize(ritz);
    Vec Av = apply_model(hessian, convexification, ritz);
    if (!project_tangent(Av, Av, message)) return false;
    const double rq = dot(ritz, Av);
    if (rq < lambda) {
      lambda = rq;
      vector = std::move(ritz);
    }
  }

  if (!found_tangent) {
    lambda = std::numeric_limits<double>::infinity();
    vector.assign(static_cast<std::size_t>(n_), 0.0);
    message = "equality manifold has zero numerical tangent dimension";
    return true;
  }
  if (!std::isfinite(lambda)) {
    message = "projected Lanczos failed to return a finite Ritz value";
    return false;
  }
  message.clear();
  return true;
}

bool CondensedQuadraticRetraction::projected_ritz_batch(
    const SymMatrix& hessian, const Convexification& convexification,
    double target_margin, int max_vectors, double& lambda,
    std::vector<std::pair<double, Vec>>& candidates,
    std::string& message) {
  ++tangent_curvature_calls_;
  candidates.clear();
  lambda = std::numeric_limits<double>::infinity();
  if (max_vectors <= 0) {
    message.clear();
    return true;
  }

  int nfree = 0;
  for (unsigned char locked : fixed_)
    if (!locked) ++nfree;
  // With full-row-rank J, nfree-me is the exact tangent dimension.  For a
  // modest nullity, one full tangent Krylov basis is both cheaper than several
  // short starts and able to expose all modes.  Inertia remains authoritative
  // if dependent rows make this estimate optimistic.
  const int tangent_dimension_estimate = std::max(0, nfree - me_);
  const bool compact_tangent = tangent_dimension_estimate >= 2 &&
      tangent_dimension_estimate <= 128;
  const int requested_steps = compact_tangent
      ? std::max(st_.dca_condensed_lanczos_steps,
                 tangent_dimension_estimate)
      : st_.dca_condensed_lanczos_steps;
  const int mmax = std::max(2, std::min(n_, requested_steps));
  const int starts = compact_tangent
      ? 1 : std::max(1, st_.dca_condensed_lanczos_starts);
  std::normal_distribution<double> normal(0.0, 1.0);
  bool found_tangent = false;
  std::vector<std::pair<double, Vec>> raw_candidates;
  raw_candidates.reserve(static_cast<std::size_t>(starts) *
                         std::max(1, max_vectors));

  for (int start = 0; start < starts; ++start) {
    Vec q(static_cast<std::size_t>(n_), 0.0);
    for (double& value : q) value = normal(rng_);
    if (!project_tangent(q, q, message)) return false;
    if (norm2(q) <= 1e-13) {
      bool coordinate_found = false;
      for (int j = 0; j < n_; ++j) {
        Vec e(static_cast<std::size_t>(n_), 0.0);
        e[static_cast<std::size_t>(j)] = 1.0;
        if (!project_tangent(e, q, message)) return false;
        if (norm2(q) > 1e-13) {
          coordinate_found = true;
          break;
        }
      }
      if (!coordinate_found) continue;
    }
    normalize(q);
    found_tangent = true;

    std::vector<Vec> basis;
    basis.reserve(static_cast<std::size_t>(mmax));
    Vec alpha;
    Vec beta;
    Vec qprev(static_cast<std::size_t>(n_), 0.0);
    double bprev = 0.0;
    for (int k = 0; k < mmax; ++k) {
      basis.push_back(q);
      Vec z = apply_model(hessian, convexification, q);
      if (!project_tangent(z, z, message)) return false;
      if (k > 0) axpy(-bprev, qprev, z);
      const double a = dot(q, z);
      alpha.push_back(a);
      axpy(-a, q, z);
      for (const auto& v : basis)
        axpy(-dot(v, z), v, z);
      if (!project_tangent(z, z, message)) return false;
      const double b = norm2(z);
      if (k + 1 >= mmax || b <= 1e-14) break;
      beta.push_back(b);
      qprev = q;
      q = z;
      for (double& value : q) value /= b;
      bprev = b;
    }

    const int m = static_cast<int>(alpha.size());
    if (m == 0) continue;
    std::vector<double> T(static_cast<std::size_t>(m) * m, 0.0);
    for (int i = 0; i < m; ++i)
      T[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m] =
          alpha[static_cast<std::size_t>(i)];
    for (int i = 0; i + 1 < m; ++i) {
      const double b = beta[static_cast<std::size_t>(i)];
      T[static_cast<std::size_t>(i) + static_cast<std::size_t>(i + 1) * m] = b;
      T[static_cast<std::size_t>(i + 1) + static_cast<std::size_t>(i) * m] = b;
    }
    auto decomposition = dense_eigendecomposition(std::move(T), m);
    if (decomposition.first.empty()) continue;

    const int probe = std::min(
        m, std::max(1, std::min(m, 2 * max_vectors + 2)));
    for (int eigen = 0; eigen < probe; ++eigen) {
      if (eigen > 0 &&
          decomposition.first[static_cast<std::size_t>(eigen)] >=
              target_margin + 10.0 * st_.curvature_tolerance)
        break;
      Vec ritz(static_cast<std::size_t>(n_), 0.0);
      const Vec& coefficient =
          decomposition.second[static_cast<std::size_t>(eigen)];
      for (int j = 0; j < m; ++j)
        axpy(coefficient[static_cast<std::size_t>(j)],
             basis[static_cast<std::size_t>(j)], ritz);
      if (!project_tangent(ritz, ritz, message)) return false;
      if (norm2(ritz) <= 1e-14) continue;
      normalize(ritz);
      // The final inertia check certifies acceptance, so this Rayleigh
      // quotient needs no extra diagnostic tangent projection.
      const double rq = dot(ritz, apply_model(
          hessian, convexification, ritz));
      if (!std::isfinite(rq)) {
        message = "projected multi-Ritz produced a nonfinite Rayleigh quotient";
        return false;
      }
      lambda = std::min(lambda, rq);
      if (rq < target_margin - st_.curvature_tolerance)
        raw_candidates.emplace_back(rq, std::move(ritz));
    }
  }

  if (!found_tangent) {
    lambda = std::numeric_limits<double>::infinity();
    message = "equality manifold has zero numerical tangent dimension";
    return true;
  }
  if (!std::isfinite(lambda)) {
    message = "projected multi-Ritz failed to return a finite Ritz value";
    return false;
  }
  std::sort(raw_candidates.begin(), raw_candidates.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });
  for (auto& raw : raw_candidates) {
    Vec direction = std::move(raw.second);
    for (const auto& accepted : candidates)
      axpy(-dot(accepted.second, direction), accepted.second, direction);
    if (!project_tangent(direction, direction, message)) return false;
    const double length = norm2(direction);
    if (!(length > 1e-8) || !std::isfinite(length)) continue;
    for (double& value : direction) value /= length;
    const double rq = dot(direction, apply_model(
        hessian, convexification, direction));
    if (!std::isfinite(rq)) {
      message = "orthogonalized projected multi-Ritz candidate is nonfinite";
      return false;
    }
    lambda = std::min(lambda, rq);
    if (rq < target_margin - st_.curvature_tolerance)
      candidates.emplace_back(rq, std::move(direction));
    if (static_cast<int>(candidates.size()) >= max_vectors) break;
  }
  message.clear();
  return true;
}

TangentConvexificationResult
CondensedQuadraticRetraction::ensure_tangent_convex_multi_ritz(
    const SymMatrix& hessian, Convexification& convexification,
    double target_margin, int max_rank) {
  TangentConvexificationResult out;
  if (!factored_) {
    out.message = "cannot test tangent curvature before factorizing the normal operator";
    return out;
  }
  if (hessian.n != n_) {
    out.message = "tangent curvature Hessian dimension mismatch";
    return out;
  }

  const int calls_before = tangent_curvature_calls_;
  const int rank_cap = std::max(0, max_rank);
  const double acceptance_floor =
      target_margin - 10.0 * st_.curvature_tolerance;
  bool recorded_initial = false;
  double lambda = std::numeric_limits<double>::infinity();
  std::string message;

  while (out.rank_added < rank_cap) {
    std::vector<std::pair<double, Vec>> candidates;
    const int room = rank_cap - out.rank_added;
    double batch_lambda = std::numeric_limits<double>::infinity();
    if (!projected_ritz_batch(hessian, convexification, target_margin,
                              room, batch_lambda, candidates, message)) {
      out.message = message;
      out.oracle_calls = tangent_curvature_calls_ - calls_before;
      return out;
    }
    lambda = batch_lambda;
    if (!recorded_initial) {
      out.initial_min_eigenvalue =
          std::isfinite(lambda) ? lambda : target_margin;
      recorded_initial = true;
    }
    if (candidates.empty()) break;
    for (auto& candidate : candidates) {
      const double beta = std::max(
          st_.curvature_tolerance,
          (target_margin - candidate.first) * (1.0 + st_.curvature_safety));
      if (!(beta > 0.0) || !std::isfinite(beta)) {
        out.message = "nonfinite projected multi-Ritz correction";
        out.oracle_calls = tangent_curvature_calls_ - calls_before;
        return out;
      }
      convexification.terms.push_back({beta, std::move(candidate.second)});
      ++out.rank_added;
      if (out.rank_added >= rank_cap) break;
    }
    if (certify_tangent_convex_by_inertia(
            hessian, convexification, acceptance_floor, message)) {
      out.success = true;
      out.final_min_eigenvalue =
          std::isfinite(lambda) ? lambda : target_margin;
      out.oracle_calls = tangent_curvature_calls_ - calls_before;
      out.message = "projected multi-Ritz model certified by bordered inertia";
      return out;
    }
  }

  // Missing Krylov modes or an exhausted rank cap are handled only by a
  // bounded monotone PSD scalar shift, with inertia as the acceptance gate.
  const int scalar_cap =
      std::max(1, st_.dca_condensed_scalar_safeguard_updates);
  double increment = std::max(
      st_.curvature_tolerance,
      0.01 * std::max(1.0, std::abs(convexification.scalar_shift)));
  if (std::isfinite(lambda) && lambda < target_margin)
    increment = std::max(increment,
        (target_margin - lambda) * (1.0 + st_.curvature_safety));
  for (int update = 0; update <= scalar_cap; ++update) {
    if (certify_tangent_convex_by_inertia(
            hessian, convexification, acceptance_floor, message)) {
      out.success = true;
      out.final_min_eigenvalue =
          std::isfinite(lambda) ? lambda : target_margin;
      out.oracle_calls = tangent_curvature_calls_ - calls_before;
      out.message = "projected multi-Ritz/scalar model certified by bordered inertia";
      return out;
    }
    if (update == scalar_cap || !(increment > 0.0) ||
        !std::isfinite(increment))
      break;
    convexification.scalar_shift += increment;
    out.scalar_shift_added += increment;
    ++out.scalar_safeguard_updates;
    increment *= 2.0;
  }

  if (!recorded_initial) out.initial_min_eigenvalue = target_margin;
  out.final_min_eigenvalue =
      std::isfinite(lambda) ? lambda : target_margin;
  out.oracle_calls = tangent_curvature_calls_ - calls_before;
  out.message = "projected multi-Ritz inertia certificate failed: " + message;
  return out;
}

TangentConvexificationResult
CondensedQuadraticRetraction::ensure_tangent_convex(
    const SymMatrix& hessian, Convexification& convexification,
    double target_margin, int max_rank) {
  TangentConvexificationResult out;
  if (!factored_) {
    out.message = "cannot test tangent curvature before factorizing the normal operator";
    return out;
  }
  if (hessian.n != n_) {
    out.message = "tangent curvature Hessian dimension mismatch";
    return out;
  }

  std::string message;
  double lambda = 0.0;
  Vec direction;
  const int calls_before = tangent_curvature_calls_;
  if (!smallest_projected_eigenpair(hessian, convexification, lambda,
                                    direction, message)) {
    out.message = message;
    return out;
  }
  out.initial_min_eigenvalue = std::isfinite(lambda) ? lambda : target_margin;
  const int rank_cap = std::max(0, max_rank);

  while (std::isfinite(lambda) &&
         lambda < target_margin - st_.curvature_tolerance &&
         out.rank_added < rank_cap) {
    const double deficit = target_margin - lambda;
    const double beta = std::max(
        st_.curvature_tolerance,
        deficit * (1.0 + st_.curvature_safety));
    convexification.terms.push_back({beta, direction});
    ++out.rank_added;
    if (!smallest_projected_eigenpair(hessian, convexification, lambda,
                                      direction, message)) {
      out.message = message;
      return out;
    }
  }

  const double acceptance_floor =
      target_margin - 10.0 * st_.curvature_tolerance;
  const int scalar_update_cap =
      std::max(1, st_.dca_condensed_scalar_safeguard_updates);
  auto add_verified_scalar_shift = [&]() -> bool {
    const double deficit = target_margin - lambda;
    const double shift = std::max(
        st_.curvature_tolerance,
        deficit * (1.0 + st_.curvature_safety));
    if (!(shift > 0.0) || !std::isfinite(shift)) {
      message = "non-finite condensed scalar curvature safeguard";
      return false;
    }
    convexification.scalar_shift += shift;
    out.scalar_shift_added += shift;
    ++out.scalar_safeguard_updates;
    return smallest_projected_eigenpair(
        hessian, convexification, lambda, direction, message);
  };

  // Preserve the historical arithmetic and oracle sequence whenever its
  // one-shot scalar safeguard succeeds.  Only a formerly failing branch is
  // allowed to perform additional monotone scalar updates.
  if (std::isfinite(lambda) && lambda < target_margin) {
    if (!add_verified_scalar_shift()) {
      out.message = message;
      return out;
    }
  }

  if (std::isfinite(lambda) && lambda < acceptance_floor) {
    while (std::isfinite(lambda) && lambda < acceptance_floor &&
           out.scalar_safeguard_updates < scalar_update_cap) {
      if (!add_verified_scalar_shift()) {
        out.message = message;
        return out;
      }
      // The first nonnegative estimate on this recovery-only branch is
      // checked with an independent projected-Lanczos start.  If that check
      // exposes another negative direction, the loop raises the scalar shift
      // again instead of converting a recoverable rank-cap event into a
      // numerical error.
      if (std::isfinite(lambda) && lambda >= acceptance_floor) {
        if (!smallest_projected_eigenpair(hessian, convexification, lambda,
                                          direction, message)) {
          out.message = message;
          return out;
        }
      }
    }
  }
  out.final_min_eigenvalue = std::isfinite(lambda) ? lambda : target_margin;
  out.oracle_calls = tangent_curvature_calls_ - calls_before;
  out.success = !std::isfinite(lambda) || lambda >= acceptance_floor;
  out.message = out.success
      ? "equality-Lagrangian Hessian convexified on the equality tangent space"
      : "projected tangent curvature remained below the requested margin";
  return out;
}

} // namespace lazyipdca::qcqp
