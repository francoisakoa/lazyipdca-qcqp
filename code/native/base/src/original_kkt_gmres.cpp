#include "lazyipdca_qcqp/original_kkt_gmres.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace lazyipdca::qcqp {
namespace {

double dot_local(const Vec& a, const Vec& b) {
  double sum = 0.0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

double norm2_local(const Vec& x) {
  return std::sqrt(std::max(0.0, dot_local(x, x)));
}

void axpy_local(double alpha, const Vec& x, Vec& y) {
  const std::size_t n = std::min(x.size(), y.size());
  for (std::size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

bool finite_vector(const Vec& x) {
  return std::all_of(x.begin(), x.end(),
                     [](double value) { return std::isfinite(value); });
}

}  // namespace

OriginalKktGmresResult original_kkt_right_preconditioned_gmres(
    const OriginalKktOperator& apply_A,
    const OriginalKktPreconditioner& apply_Minv,
    const Vec& rhs,
    Vec& correction,
    int max_iterations,
    double absolute_tolerance) {
  OriginalKktGmresResult out;
  const int dimension = static_cast<int>(rhs.size());
  correction.assign(rhs.size(), 0.0);
  if (dimension == 0) {
    out.converged = true;
    out.message = "empty system";
    return out;
  }
  const int krylov_dimension =
      std::max(0, std::min(max_iterations, dimension));
  if (krylov_dimension == 0) {
    out.message = "GMRES iteration limit is zero";
    return out;
  }
  if (!(absolute_tolerance > 0.0) || !std::isfinite(absolute_tolerance)) {
    out.message = "invalid GMRES absolute tolerance";
    return out;
  }

  const double beta = norm2_local(rhs);
  out.residual_2 = beta;
  if (!std::isfinite(beta) || !finite_vector(rhs)) {
    out.message = "non-finite original-KKT residual";
    return out;
  }
  if (beta <= absolute_tolerance) {
    out.converged = true;
    out.message = "original KKT already satisfies tolerance";
    return out;
  }

  std::vector<Vec> basis(static_cast<std::size_t>(krylov_dimension + 1),
                         Vec(rhs.size(), 0.0));
  std::vector<Vec> preconditioned(
      static_cast<std::size_t>(krylov_dimension), Vec(rhs.size(), 0.0));
  std::vector<double> hessenberg(
      static_cast<std::size_t>(krylov_dimension + 1) * krylov_dimension, 0.0);
  std::vector<double> cosine(static_cast<std::size_t>(krylov_dimension), 0.0);
  std::vector<double> sine(static_cast<std::size_t>(krylov_dimension), 0.0);
  std::vector<double> rotated_rhs(
      static_cast<std::size_t>(krylov_dimension + 1), 0.0);
  auto h = [&](int row, int column) -> double& {
    return hessenberg[static_cast<std::size_t>(row) * krylov_dimension +
                       column];
  };

  for (int i = 0; i < dimension; ++i)
    basis[0][static_cast<std::size_t>(i)] =
        rhs[static_cast<std::size_t>(i)] / beta;
  rotated_rhs[0] = beta;

  int used = 0;
  std::string failure;
  for (int column = 0; column < krylov_dimension; ++column) {
    if (!apply_Minv(basis[static_cast<std::size_t>(column)],
                    preconditioned[static_cast<std::size_t>(column)],
                    failure)) {
      out.message = "original-KKT GMRES preconditioner failed: " + failure;
      break;
    }
    if (preconditioned[static_cast<std::size_t>(column)].size() != rhs.size() ||
        !finite_vector(preconditioned[static_cast<std::size_t>(column)])) {
      out.message = "original-KKT GMRES produced an invalid preconditioned vector";
      break;
    }

    Vec work(rhs.size(), 0.0);
    apply_A(preconditioned[static_cast<std::size_t>(column)], work);
    if (work.size() != rhs.size() || !finite_vector(work)) {
      out.message = "original-KKT GMRES operator returned an invalid vector";
      break;
    }

    // Two-pass modified Gram--Schmidt is used because this path is entered
    // precisely for ill-conditioned original KKT systems.  The second pass
    // improves orthogonality without changing any established solver path.
    for (int pass = 0; pass < 2; ++pass) {
      for (int row = 0; row <= column; ++row) {
        const double projection =
            dot_local(basis[static_cast<std::size_t>(row)], work);
        h(row, column) += projection;
        axpy_local(-projection, basis[static_cast<std::size_t>(row)], work);
      }
    }
    h(column + 1, column) = norm2_local(work);
    if (h(column + 1, column) > 0.0) {
      for (int i = 0; i < dimension; ++i)
        basis[static_cast<std::size_t>(column + 1)]
             [static_cast<std::size_t>(i)] =
            work[static_cast<std::size_t>(i)] / h(column + 1, column);
    }

    for (int row = 0; row < column; ++row) {
      const double upper = h(row, column);
      const double lower = h(row + 1, column);
      h(row, column) = cosine[static_cast<std::size_t>(row)] * upper +
                       sine[static_cast<std::size_t>(row)] * lower;
      h(row + 1, column) =
          -sine[static_cast<std::size_t>(row)] * upper +
          cosine[static_cast<std::size_t>(row)] * lower;
    }

    const double denominator =
        std::hypot(h(column, column), h(column + 1, column));
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
      out.message = "original-KKT GMRES Arnoldi breakdown";
      break;
    }
    cosine[static_cast<std::size_t>(column)] =
        h(column, column) / denominator;
    sine[static_cast<std::size_t>(column)] =
        h(column + 1, column) / denominator;
    h(column, column) = denominator;
    h(column + 1, column) = 0.0;

    const double current = rotated_rhs[static_cast<std::size_t>(column)];
    const double next = rotated_rhs[static_cast<std::size_t>(column + 1)];
    rotated_rhs[static_cast<std::size_t>(column)] =
        cosine[static_cast<std::size_t>(column)] * current +
        sine[static_cast<std::size_t>(column)] * next;
    rotated_rhs[static_cast<std::size_t>(column + 1)] =
        -sine[static_cast<std::size_t>(column)] * current +
        cosine[static_cast<std::size_t>(column)] * next;
    used = column + 1;
    out.iterations = used;
    out.residual_2 =
        std::abs(rotated_rhs[static_cast<std::size_t>(column + 1)]);
    if (out.residual_2 <= absolute_tolerance) break;
  }

  if (used == 0) return out;
  Vec coefficients(static_cast<std::size_t>(used), 0.0);
  for (int row = used - 1; row >= 0; --row) {
    double value = rotated_rhs[static_cast<std::size_t>(row)];
    for (int column = row + 1; column < used; ++column)
      value -= h(row, column) * coefficients[static_cast<std::size_t>(column)];
    const double diagonal = h(row, row);
    if (diagonal == 0.0 || !std::isfinite(diagonal)) {
      out.message = "original-KKT GMRES triangular solve breakdown";
      correction.assign(rhs.size(), 0.0);
      return out;
    }
    coefficients[static_cast<std::size_t>(row)] = value / diagonal;
  }
  for (int column = 0; column < used; ++column)
    axpy_local(coefficients[static_cast<std::size_t>(column)],
               preconditioned[static_cast<std::size_t>(column)], correction);

  // The Arnoldi estimate alone can under-report the true residual after loss
  // of orthogonality.  Certification and the public result always use this
  // explicitly recomputed residual.
  Vec applied(rhs.size(), 0.0);
  apply_A(correction, applied);
  if (applied.size() != rhs.size() || !finite_vector(applied) ||
      !finite_vector(correction)) {
    out.message = "original-KKT GMRES explicit residual is invalid";
    correction.assign(rhs.size(), 0.0);
    out.residual_2 = std::numeric_limits<double>::infinity();
    return out;
  }
  for (std::size_t i = 0; i < rhs.size(); ++i) applied[i] = rhs[i] - applied[i];
  out.residual_2 = norm2_local(applied);
  out.converged =
      std::isfinite(out.residual_2) && out.residual_2 <= absolute_tolerance;
  if (out.converged) {
    out.message = "original-KKT GMRES converged";
  } else if (out.message.empty()) {
    std::ostringstream message;
    message << "original-KKT GMRES stopped; explicit residual2="
            << out.residual_2;
    out.message = message.str();
  }
  return out;
}

}  // namespace lazyipdca::qcqp
