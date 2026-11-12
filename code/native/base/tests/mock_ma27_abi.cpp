// TEST DOUBLE ONLY: dense LAPACK, not HSL. Never install, auto-discover or
// benchmark this library. It exercises the MA27 ABI and adapter checks only.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

extern "C" {
void dsyev_(const char*, const char*, const int*, double*, const int*, double*,
            double*, const int*, int*);
void dgesv_(const int*, const int*, double*, const int*, int*, double*,
            const int*, int*);
}

namespace {
int analyses = 0;
int analysis_flag = -1;
std::vector<int> analysis_order;

std::vector<double> matrix(int n, int ne, const int* row, const int* col,
                           const double* values) {
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0);
  for (int k = 0; k < ne; ++k) {
    const int i = row[k] - 1, j = col[k] - 1;
    out[i + j * n] += values[k];
    if (i != j) out[j + i * n] += values[k];
  }
  return out;
}

int inertia(int n, const double* values, int& rank) {
  std::vector<double> copy(values, values + n * n), eigenvalues(n);
  std::vector<double> work(std::max(1, 3 * n));
  const int lwork = static_cast<int>(work.size());
  int info = 0;
  const char job = 'N', uplo = 'U';
  dsyev_(&job, &uplo, &n, copy.data(), &n, eigenvalues.data(), work.data(),
         &lwork, &info);
  if (info != 0) { rank = 0; return 0; }
  int negative = 0;
  rank = 0;
  for (double value : eigenvalues) {
    negative += value < -1e-12;
    rank += std::abs(value) > 1e-12;
  }
  return negative;
}
}

extern "C" {
int mock_ma27_test_double() { return 1; }
int mock_ma27_analysis_count() { return analyses; }
int mock_ma27_last_flag() { return analysis_flag; }
int mock_ma27_order_at(int i) {
  return i >= 0 && i < static_cast<int>(analysis_order.size())
             ? analysis_order[i] : -1;
}

void ma27id_(int* icntl, double* cntl) {
  std::fill(icntl, icntl + 30, 0);
  std::fill(cntl, cntl + 5, 0);
}

void ma27ad_(int* n, int* ne, const int*, const int*, int*, int*, int* keep,
             int*, int* steps, int* flag, int*, double*, int* info, double*) {
  std::fill(info, info + 20, 0);
  info[4] = *ne;
  info[5] = *n + 2;
  *steps = *n;
  ++analyses;
  analysis_flag = *flag;
  analysis_order.assign(*n, 0);
  std::vector<int> seen(*n, 0);
  if (*flag != 0 && *flag != 1) { info[0] = -99; return; }
  for (int i = 0; i < *n; ++i) {
    const int position = *flag == 1 ? keep[i] - 1 : i;
    if (position < 0 || position >= *n || seen[position]++) {
      info[0] = -99;
      return;
    }
    analysis_order[i] = position + 1;
    keep[i] = position + 1;
  }
}

void ma27bd_(int* n, int* ne, const int* row, const int* col, double* values,
             int* la, int* iw, int* liw, int* keep, int*, int* maxfrt,
             int*, int*, double*, int* info) {
  std::fill(info, info + 20, 0);
  // Understate storage at analysis to exercise both workspace retry paths.
  if (*la < (*n) * (*n)) { info[0] = -4; info[1] = (*n) * (*n); return; }
  if (*liw < 2 * (*n) + 3) { info[0] = -3; info[1] = 2 * (*n) + 3; return; }
  const auto dense = matrix(*n, *ne, row, col, values);
  int rank = 0;
  info[14] = inertia(*n, dense.data(), rank);
  if (rank != *n) { info[0] = -5; return; }
  for (int i = 0; i < *n; ++i) iw[i] = keep[i] - 1;
  for (int j = 0; j < *n; ++j)
    for (int i = 0; i < *n; ++i)
      values[iw[i] + iw[j] * (*n)] = dense[i + j * (*n)];
  *maxfrt = *n;
}

void ma27cd_(int* n, double* values, int*, int* iw, int*, double*, int*,
             double* rhs, int*, int*, int*, int* info) {
  info[0] = 0;
  if (std::getenv("MOCK_MA27_BAD_SOLVE")) {
    std::fill(rhs, rhs + *n, 0);
    return;
  }
  std::vector<double> copy(values, values + (*n) * (*n)), ordered(*n);
  std::vector<int> pivots(*n);
  for (int i = 0; i < *n; ++i) ordered[iw[i]] = rhs[i];
  const int nrhs = 1;
  dgesv_(n, &nrhs, copy.data(), n, pivots.data(), ordered.data(), n, info);
  for (int i = 0; i < *n; ++i) rhs[i] = ordered[iw[i]];
}
}
