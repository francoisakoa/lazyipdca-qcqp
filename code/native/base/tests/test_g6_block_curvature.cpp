#include "../src/dca_curvature.hpp"
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {
constexpr int dimension = 24;
constexpr double margin = 1e-4;
using Normals = std::vector<std::vector<std::pair<int, double>>>;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

Vec spectrum(Vec matrix) {
  Vec values(dimension), work(3 * dimension);
  const char vectors = 'N', triangle = 'U';
  int info = 0, workspace = work.size(), n = dimension;
  dsyev_(&vectors, &triangle, &n, matrix.data(), &n, values.data(),
         work.data(), &workspace, &info);
  require(info == 0, "independent dense eigensystem failed");
  for (double value : values) require(std::isfinite(value), "nonfinite eigenvalue");
  return values;
}

// Assemble from exported coefficients and factors, without corrected(), the
// bordered matrix, or the production factorization/inertia implementation.
void verify_certificate(const std::vector<SymTriplet> &h,
                        const detail::DcaCurvature &curvature,
                        const Normals &normals, int cap) {
  Vec original(dimension * dimension, 0), correction(original.size(), 0);
  for (const auto &entry : h) {
    original[entry.i + dimension * entry.j] += entry.v;
    if (entry.i != entry.j) original[entry.j + dimension * entry.i] += entry.v;
  }
  require(std::isfinite(curvature.rho) && curvature.rho >= 0,
          "invalid scalar complement");
  require(std::isfinite(curvature.normal_weight) && curvature.normal_weight >= 0,
          "invalid normal completion");
  for (int i = 0; i < dimension; ++i) correction[i + dimension * i] = curvature.rho;
  auto add_rank_one = [&](double weight, const Vec &direction) {
    require(std::isfinite(weight) && weight >= 0, "non-PSD atom weight");
    for (int j = 0; j < dimension; ++j)
      for (int i = 0; i < dimension; ++i)
        correction[i + dimension * j] += weight * direction[i] * direction[j];
  };
  for (const auto &atom : curvature.atoms) {
    Vec direction(dimension, 0);
    direction[atom.i] = atom.u;
    if (atom.j != atom.i) direction[atom.j] = atom.v;
    add_rank_one(atom.beta, direction);
  }
  for (const auto &atom : curvature.spectral_atoms) add_rank_one(atom.beta, atom.direction);
  for (const auto &row : normals) {
    Vec direction(dimension, 0);
    for (const auto &entry : row) direction[entry.first] += entry.second;
    add_rank_one(curvature.normal_weight, direction);
  }
  require(spectrum(correction).front() >= -1e-8, "DCA correction is not PSD");
  for (std::size_t k = 0; k < original.size(); ++k) original[k] += correction[k];
  require(spectrum(original).front() > 0.5 * margin,
          "final full-space majorant lost its strict-convexity certificate");
  require(int(curvature.atoms.size() + curvature.stored_spectral_atoms()) <= cap &&
              curvature.last_peak_stored_atoms <= cap,
          "stored atom capacity exceeded");
}

std::vector<SymTriplet> indefinite_matrix() {
  std::vector<SymTriplet> h;
  for (int i = 0; i < dimension; ++i) h.push_back({i, i, 3 + 0.04 * i});
  h[0].v = -4;
  h[2].v = -1.7;
  h.back().v = -8; // negative only in the equality-normal direction
  h.push_back({0, 1, 0.6});
  h.push_back({2, 3, 0.35});
  return h;
}

void exercise_caps(const Settings &settings, bool block) {
  Settings configured = settings;
  configured.dca_filter_block_ritz = block;
  const auto h = indefinite_matrix();
  const Normals normals = {{{dimension - 1, 1}}};
  for (int cap : {0, 1, 2}) {
    detail::DcaCurvature curvature(configured);
    int remaining = cap;
    curvature.ensure(dimension, h, margin, true, remaining, 2, 1e-8, &normals);
    verify_certificate(h, curvature, normals, cap);
    require(remaining >= 0, "atom budget became negative");
    if (cap == 0) require(curvature.additions == 0, "zero capacity admitted an atom");
    else {
      require(curvature.stored_spectral_atoms() > 0,
              "negative tangent curvature produced no spectral correction");
      require(curvature.inverse_probes > 0, "test did not exercise the inverse oracle");
    }
    // There are two independent negative tangent modes. A rank-zero/rank-one
    // atom correction cannot make both positive without a scalar complement.
    if (cap < 2) require(curvature.rho > 0, "missing complementary scalar curvature");
    const int additions = curvature.additions, factors = curvature.certificates;
    curvature.ensure(dimension, h, margin, true, remaining, 2, 1e-8, &normals);
    require(curvature.additions == additions && curvature.certificates == factors,
            "identical model invalidated its cached certificate");
    verify_certificate(h, curvature, normals, cap);
  }
}
} // namespace

int main() {
  try {
    Settings settings;
    settings.linear_backend = "native";
    settings.dca_filter_rho_refinements = 3;
    exercise_caps(settings, false);
    exercise_caps(settings, true);
    settings.dca_filter_block_ritz = true;

    // Leave capacity available: non-events and moving normals must not spend
    // it, even though the old full-space certificate must be updated.
    detail::DcaCurvature curvature(settings);
    auto h = indefinite_matrix();
    Normals normals = {{{dimension - 1, 1}}};
    int remaining = 4;
    curvature.ensure(dimension, h, margin, true, remaining, 2, 1e-8, &normals);
    verify_certificate(h, curvature, normals, 4);
    const int initial_additions = curvature.additions;
    require(remaining > 0, "test did not retain admission capacity");

    normals = {{{1, 0.002}, {dimension - 1, std::sqrt(1 - 0.002 * 0.002)}}};
    const int initial_updates = curvature.updates;
    curvature.ensure(dimension, h, margin, true, remaining, 2, 1e-8, &normals);
    require(curvature.updates == initial_updates && curvature.additions == initial_additions,
            "normal drift was treated as a Hessian admission event");
    verify_certificate(h, curvature, normals, 4);

    // Reverse one captured negative mode and couple its subspace to another:
    // old weights can decrease and rotate, so an old certificate is unsafe.
    h[0].v = 2.2;
    h[2].v = -3.2;
    h.push_back({0, 2, 0.8});
    h.push_back({0, 3, -0.4});
    curvature.ensure(dimension, h, margin, false, remaining, 2, 1e-8, &normals);
    require(curvature.additions == initial_additions, "atom admitted outside an event");
    verify_certificate(h, curvature, normals, 4);

    // A newly negative independent direction may be admitted on the next real
    // Hessian event. Validate the complete result after all batch reweighting.
    h[5].v = -2.5;
    curvature.ensure(dimension, h, margin, true, remaining, 2, 1e-8, &normals);
    require(remaining >= 0, "post-update admission exceeded capacity");
    verify_certificate(h, curvature, normals, 4);
    std::cout << "PASS block Ritz: independent PSD spectra, caps and Hessian events\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
