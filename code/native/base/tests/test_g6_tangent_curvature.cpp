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
double smallest(Vec matrix, int dimension) {
  Vec values(dimension), work(3 * dimension);
  const char vectors = 'N', triangle = 'U';
  int info = 0, workspace = work.size();
  dsyev_(&vectors, &triangle, &dimension, matrix.data(), &dimension,
         values.data(), work.data(), &workspace, &info);
  require(info == 0 && std::isfinite(values.front()), "independent eigensystem failed");
  return values.front();
}
// Explicit orthonormal tangent basis for N=(sin(angle),0,...,cos(angle)).
std::vector<Vec> tangent_basis(double angle) {
  std::vector<Vec> basis(n - 1, Vec(n, 0));
  basis[0][0] = std::cos(angle);
  basis[0][n - 1] = -std::sin(angle);
  for (int i = 1; i < n - 1; ++i) basis[i][i] = 1;
  return basis;
}
Normals normal(double angle) {
  return {{{0, std::sin(angle)}, {n - 1, std::cos(angle)}}};
}
std::vector<SymTriplet> matrix(double angle, bool negative_tangent) {
  const auto basis = tangent_basis(angle);
  Vec dense(n * n, 0);
  for (int k = 0; k < n - 1; ++k) {
    const double eigenvalue = negative_tangent && k == 0 ? -4.0 :
                             negative_tangent && k == 2 ? -1.7 : 3 + 0.04 * k;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        dense[i + n * j] += eigenvalue * basis[k][i] * basis[k][j];
  }
  Vec v(n, 0);
  v[0] = std::sin(angle);
  v[n - 1] = std::cos(angle);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) dense[i + n * j] -= 1e5 * v[i] * v[j];
  std::vector<SymTriplet> h;
  for (int i = 0; i < n; ++i)
    for (int j = i; j < n; ++j)
      if (i == j || dense[i + n * j] != 0) h.push_back({i, j, dense[i + n * j]});
  return h;
}
double verify(const std::vector<SymTriplet> &h, const detail::DcaCurvature &curvature,
              double angle, int cap) {
  Vec dense(n * n, 0), correction(n * n, 0);
  for (auto e : h) {
    dense[e.i + n * e.j] += e.v;
    if (e.i != e.j) dense[e.j + n * e.i] += e.v;
  }
  for (int i = 0; i < n; ++i) correction[i + n * i] = curvature.rho;
  auto rank_one = [&](double beta, const Vec &v) {
    require(beta >= 0 && std::isfinite(beta), "invalid PSD atom weight");
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) correction[i + n * j] += beta * v[i] * v[j];
  };
  for (auto atom : curvature.atoms) {
    Vec v(n, 0);
    v[atom.i] = atom.u;
    if (atom.i != atom.j) v[atom.j] = atom.v;
    rank_one(atom.beta, v);
  }
  for (const auto &atom : curvature.spectral_atoms) rank_one(atom.beta, atom.direction);
  require(smallest(correction, n) >= -1e-8, "partial-Lagrangian correction is not PSD");
  for (std::size_t k = 0; k < dense.size(); ++k) dense[k] += correction[k];
  const auto basis = tangent_basis(angle);
  Vec projected((n - 1) * (n - 1), 0);
  for (int j = 0; j < n - 1; ++j)
    for (int i = 0; i < n - 1; ++i)
      for (int b = 0; b < n; ++b)
        for (int a = 0; a < n; ++a)
          projected[i + (n - 1) * j] += basis[i][a] * dense[a + n * b] * basis[j][b];
  require(smallest(projected, n - 1) > 0.5 * margin,
          "affine-domain majorant is not strictly convex");
  require(int(curvature.atoms.size() + curvature.stored_spectral_atoms()) <= cap,
          "atom cap exceeded");
  return smallest(dense, n);
}
} // namespace

int main() {
  try {
    Settings settings;
    settings.linear_backend = "native";
    settings.dca_filter_tangent_curvature = true;
    settings.dca_filter_block_ritz = true;
    settings.dca_filter_certificate_transport = true;
    for (double angle : {0.0, 0.31}) {
      auto normals = normal(angle);
      const auto normal_only = matrix(angle, false);
      detail::DcaCurvature c(settings);
      int budget = 4;
      c.ensure(n, normal_only, margin, true, budget, 2, 1e-8, &normals);
      require(c.tangent_certificate, "tangent certificate mode not enabled");
      require(c.rho == 0 && c.additions == 0,
              "normal-only negative curvature caused a tangent correction");
      require(verify(normal_only, c, angle, 4) < -1e4,
              "test failed to retain the harmless negative normal curvature");

      const auto h = matrix(angle, true);
      for (int cap : {0, 1, 2}) {
        detail::DcaCurvature correction(settings);
        int remaining = cap;
        correction.ensure(n, h, margin, true, remaining, 2, 1e-8, &normals);
        verify(h, correction, angle, cap);
        if (cap < 2) require(correction.rho > 0, "rank-limited complement is missing");
        if (cap > 0) require(correction.additions > 0, "negative tangent mode was not captured");
        const int additions = correction.additions;
        const double changed_angle = angle + 1e-9;
        auto changed_normals = normal(changed_angle);
        correction.ensure(n, h, margin, true, remaining, 2, 1e-8, &changed_normals);
        require(correction.additions == additions, "normal drift admitted a new atom");
        require(correction.certificate_transports == 0,
                "full-space gamma transport was applied to a tangent certificate");
        verify(h, correction, changed_angle, cap);
      }
    }
    std::cout << "PASS tangent inertia: independent restricted spectra and normal-only curvature\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
