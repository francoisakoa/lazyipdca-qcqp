#include "../src/dca_curvature.hpp"
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {
constexpr int n = 24;
constexpr double margin = 1e-4;
using Normals = std::vector<std::vector<std::pair<int, double>>>;
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
double smallest(Vec matrix) {
  Vec eigenvalues(n), work(3 * n);
  const char vectors = 'N', triangle = 'U';
  int dimension = n, workspace = work.size(), info = 0;
  dsyev_(&vectors, &triangle, &dimension, matrix.data(), &dimension,
         eigenvalues.data(), work.data(), &workspace, &info);
  require(info == 0 && std::isfinite(eigenvalues.front()), "independent spectrum failed");
  return eigenvalues.front();
}
void verify(const std::vector<SymTriplet> &h, const detail::DcaCurvature &c,
            const Normals &normals) {
  Vec a(n * n, 0), correction(n * n, 0);
  for (auto e : h) {
    a[e.i + n * e.j] += e.v;
    if (e.i != e.j) a[e.j + n * e.i] += e.v;
  }
  for (int i = 0; i < n; ++i) correction[i + n * i] = c.rho;
  auto rank_one = [&](double beta, const Vec &v) {
    require(beta >= 0 && std::isfinite(beta), "invalid correction weight");
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) correction[i + n * j] += beta * v[i] * v[j];
  };
  for (auto atom : c.atoms) {
    Vec v(n, 0);
    v[atom.i] = atom.u;
    if (atom.i != atom.j) v[atom.j] = atom.v;
    rank_one(atom.beta, v);
  }
  for (const auto &atom : c.spectral_atoms) rank_one(atom.beta, atom.direction);
  for (const auto &row : normals) {
    Vec v(n, 0);
    for (auto e : row) v[e.first] += e.second;
    rank_one(c.normal_weight, v);
  }
  require(smallest(correction) > -1e-8, "transported DCA correction is not PSD");
  for (std::size_t k = 0; k < a.size(); ++k) a[k] += correction[k];
  require(smallest(a) > margin * 0.5, "transported full majorant is not strictly convex");
}
bool identical_atoms(const std::vector<LowRankTerm> &left,
                     const std::vector<LowRankTerm> &right) {
  if (left.size() != right.size()) return false;
  for (std::size_t i = 0; i < left.size(); ++i)
    if (left[i].beta != right[i].beta || left[i].direction != right[i].direction) return false;
  return true;
}
} // namespace

int main() {
  try {
    Settings settings;
    settings.linear_backend = "native";
    settings.dca_filter_block_ritz = true;
    settings.dca_filter_certificate_transport = true;
    detail::DcaCurvature curvature(settings);
    std::vector<SymTriplet> h;
    for (int i = 0; i < n; ++i) h.push_back({i, i, 3 + 0.04 * i});
    h[0].v = -4;
    h[2].v = -1.7;
    h.back().v = -8;
    h.push_back({0, 1, 0.6});
    h.push_back({2, 3, 0.35});
    Normals normals = {{{1, 0}, {n - 1, 1}}};
    int remaining = 4;
    curvature.ensure(n, h, margin, true, remaining, 2, 1e-8, &normals);
    require(curvature.normal_weight > 0 && !curvature.spectral_atoms.empty(),
            "test did not establish a normal-plus-atomic certificate");
    require(curvature.certificate_transports == 0, "transport used before first certificate");
    verify(h, curvature, normals);

    const int factors = curvature.certificates, additions = curvature.additions;
    const auto atoms = curvature.spectral_atoms;
    const int budget = remaining;
    for (int step = 1; step <= 3; ++step) {
      h[0].v -= 1e-8;
      normals[0][0].second += 1e-8;
      curvature.ensure(n, h, margin, true, remaining, 2, 1e-8, &normals);
      require(curvature.certificate_transports == step,
              "small certified perturbation was not transported");
      require(curvature.certificates == factors, "transport unnecessarily factorized the matrix");
      require(curvature.additions == additions && remaining == budget &&
                  identical_atoms(atoms, curvature.spectral_atoms),
              "transport changed its retained atom correction");
      verify(h, curvature, normals);
    }
    require(curvature.normal_weight > 1e6, "test did not exercise the gamma restart bound");
    h[0].v -= 1e-8;
    curvature.ensure(n, h, margin, false, remaining, 2, 1e-8, &normals);
    require(curvature.certificate_transports == 3 && curvature.certificate_transport_streak == 0,
            "unbounded chain of transported certificates");
    require(curvature.certificates > factors && curvature.normal_weight <= 1e6,
            "full recertification failed to restart the normal-weight search");
    verify(h, curvature, normals);

    const int transported_before_large_change = curvature.certificate_transports;
    h[2].v -= 0.05;
    curvature.ensure(n, h, margin, false, remaining, 2, 1e-8, &normals);
    require(curvature.certificate_transports == transported_before_large_change,
            "large perturbation bypassed a complete certificate");
    verify(h, curvature, normals);

    // The inexpensive Frobenius bound is valid only for an actual matrix
    // layout with unique coordinates; duplicate entries must use the fallback.
    const double shared = normals[0][0].second * 0.5;
    normals = {{{1, shared}, {1, shared}, {n - 1, 1}}};
    curvature.ensure(n, h, margin, true, remaining, 2, 1e-8, &normals);
    require(curvature.certificate_transports == transported_before_large_change,
            "changed normal pattern used an invalid perturbation bound");
    verify(h, curvature, normals);
    std::cout << "PASS transported PSD certificate, fixed atoms and bounded fallback\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
