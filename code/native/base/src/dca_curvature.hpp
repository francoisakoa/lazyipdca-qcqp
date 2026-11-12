#pragma once
#include "sparse_pair_elimination.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <map>
#include <chrono>
#include <numeric>
extern "C" void dsyev_(const char *, const char *, const int *, double *, const int *, double *,
                       double *, const int *, int *);

namespace lazyipdca::qcqp::detail {
// Sparse rank-one atoms are either coordinate or two-coordinate directions.
// They are admitted by a negative Rayleigh witness, never by a positive probe.
// The residual scalar is independently certified, including when the atom cap
// is zero. Symbolic LDL data persist across updates of a matrix family.
struct DcaTimedScope {
  double &seconds;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  explicit DcaTimedScope(double &value) : seconds(value) {}
  ~DcaTimedScope() { seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};
struct DcaAtom {
  int i = 0, j = 0;
  double u = 1, v = 0, beta = 0;
};
class DcaCurvature {
  Settings settings_;
  std::unique_ptr<SymmetricFactorization> factor_;
  // All scalar/gamma trials of one correction share the same sparse border.
  // Retain storage across models and update only the trial diagonals until an
  // atom rebalancing changes that border. The factorization still receives
  // exactly the same triplet order and arithmetic as a fresh assembly.
  std::vector<SymmetricTriplet> factor_entries_;
  std::vector<std::size_t> primal_diagonal_slots_, normal_diagonal_slots_;
  std::vector<SymTriplet> previous_;
  int dimension_ = -1;
  double previous_margin_ = 0;
  std::vector<Vec> atom_basis_;
  std::vector<std::vector<std::pair<int, double>>> previous_normals_;

public:
  struct RitzEstimate {
    bool available = false;
    double value = 0;
    double residual = 0;
  };
  std::vector<DcaAtom> atoms;
  std::vector<LowRankTerm> spectral_atoms;
  int basis_compressions = 0, retired_directions = 0;
  int last_peak_stored_atoms = 0;
  int rho_refinement_trials = 0, rho_refinement_acceptances = 0;
  int certificate_transport_trials = 0, certificate_transports = 0, certificate_transport_streak = 0;
  int inverse_probes = 0, factor_reuses = 0, subspace_spectra = 0, dependent_witnesses = 0;
  int ritz_shift_proposals = 0, gershgorin_fallback_uses = 0;
  double ensure_seconds = 0, factor_seconds = 0, inverse_seconds = 0, subspace_seconds = 0;
  double rho = 0, normal_weight = 0;
  bool tangent_certificate = false;
  int small_spectra = 0;
  int updates = 0, certificates = 0, reuses = 0, probes = 0, additions = 0;
  explicit DcaCurvature(const Settings &s) : settings_(s) {
    if (s.dca_filter_curvature_backend != "same")
      settings_.linear_backend = s.dca_filter_curvature_backend;
    settings_.kkt_adaptive_regularization = true;
    settings_.kkt_primal_regularization = 0;
    settings_.kkt_dual_regularization = 0;
    settings_.kkt_preserve_structural_zeros = true;
    settings_.kkt_interleaved_equality_ordering = false;
    settings_.kkt_primal_first_ordering = false;
    settings_.native_inertia_relative_zero_tolerance = 0;
    tangent_certificate = s.dca_filter_tangent_curvature;
    if (tangent_certificate) settings_.kkt_interleaved_equality_ordering = true;
  }
  static std::vector<SymTriplet> canonical(int n, const std::vector<SymTriplet> &h) {
    bool ordered = true;
    int diagonal_count = 0;
    for (std::size_t k = 0; k < h.size(); ++k) {
      const auto e = h[k];
      if (e.i < 0 || e.j < e.i || e.j >= n || !std::isfinite(e.v) ||
          (k && !(h[k-1].i < e.i || (h[k-1].i == e.i && h[k-1].j < e.j)))) {
        ordered = false; break;
      }
      diagonal_count += e.i == e.j;
    }
    if (ordered && diagonal_count == n) return h;
    // Contiguous stable reduction avoids one heap allocation per coefficient.
    // Stability preserves the summation order of duplicate input entries.
    std::vector<SymTriplet> a;
    a.reserve(n + h.size());
    for (int i = 0; i < n; ++i)
      a.push_back({i, i, 0});
    for (auto e : h) {
      if (e.i < 0 || e.j < 0 || e.i >= n || e.j >= n)
        throw std::runtime_error("invalid DCA curvature index");
      if (e.i > e.j)
        std::swap(e.i, e.j);
      a.push_back(e);
    }
    std::stable_sort(a.begin(), a.end(),
                     [](auto x, auto y) { return x.i < y.i || (x.i == y.i && x.j < y.j); });
    std::size_t write = 0;
    for (std::size_t read = 0; read < a.size();) {
      auto entry = a[read++];
      while (read < a.size() && a[read].i == entry.i && a[read].j == entry.j)
        entry.v += a[read++].v;
      if (!std::isfinite(entry.v))
        throw std::runtime_error("nonfinite DCA curvature coefficient");
      a[write++] = entry;
    }
    a.resize(write);
    return a;
  }
  std::vector<SymTriplet> corrected(int n, const std::vector<SymTriplet> &h, double shift) const {
    if (atoms.empty()) {
      auto out = canonical(n, h);
      for (auto &e : out) if (e.i == e.j) {
        e.v += shift;
        if (!std::isfinite(e.v)) throw std::runtime_error("nonfinite DCA scalar correction");
      }
      return out;
    }
    auto out = h;
    for (int i = 0; i < n; ++i)
      out.push_back({i, i, shift});
    for (auto a : atoms) {
      out.push_back({a.i, a.i, a.beta * a.u * a.u});
      if (a.j != a.i) {
        out.push_back({a.i, a.j, a.beta * a.u * a.v});
        out.push_back({a.j, a.j, a.beta * a.v * a.v});
      }
    }
    return canonical(n, out);
  }
  std::size_t stored_spectral_atoms() const { return atom_basis_.size(); }
  // Rebalance the entire captured subspace instead of independently covering
  // overlapping Rayleigh witnesses. The resulting PSD correction is a proposal;
  // the bordered inertia certificate still checks the full reduced operator.
  void rebalance(int n, const std::vector<SymTriplet> &h,
                 const std::vector<std::vector<std::pair<int, double>>> *normals,
                 double gamma, double margin) {
    const int rank = atom_basis_.size();
    spectral_atoms.clear();
    if (!rank) return;
    DcaTimedScope timer(subspace_seconds);
    ++subspace_spectra;
    Vec projected(rank * rank, 0), values(rank), work(std::max(1, 3 * rank));
    for (int j = 0; j < rank; ++j) {
      const auto &v = atom_basis_[j];
      Vec hv(n, 0);
      for (auto e : h) {
        hv[e.i] += e.v * v[e.j];
        if (e.i != e.j) hv[e.j] += e.v * v[e.i];
      }
      // A normal completion certifies the full operator, but its artificial
      // positive curvature must not erase useful captured negative modes.
      // The legacy weighting is available only as an explicit control.
      if (!tangent_certificate && settings_.dca_filter_normal_atom_weighting && normals && gamma > 0)
        for (const auto &row : *normals) {
          double a = 0;
          for (auto e : row) a += e.second * v[e.first];
          for (auto e : row) hv[e.first] += gamma * a * e.second;
        }
      for (int i = 0; i <= j; ++i)
        projected[i + rank * j] = dot(atom_basis_[i], hv);
    }
    int info = 0, lwork = work.size();
    const char vectors = 'V', triangle = 'U';
    dsyev_(&vectors, &triangle, &rank, projected.data(), &rank, values.data(), work.data(), &lwork, &info);
    if (info || !finite_values(values))
      throw std::runtime_error("DCA atom subspace eigensystem failed");
    const double padding = std::max(margin, std::sqrt(std::numeric_limits<double>::epsilon()) *
                                               std::max({1.0, std::abs(values.front()), std::abs(values.back())}));
    for (int j = 0; j < rank; ++j)
      if (values[j] < -1e-12) {
        Vec direction(n, 0);
        for (int i = 0; i < rank; ++i)
          axpy(projected[i + rank * j], atom_basis_[i], direction);
        spectral_atoms.push_back({-values[j] + padding, std::move(direction)});
      }
  }
  static bool finite_values(const Vec &v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
  }
  // Direct symmetric Lanczos/Ritz estimate on the CURRENT atom-corrected
  // primal operator.  This is deliberately a proposal, never a certificate:
  // the bordered inertia test below remains the sole normal-path authority.
  // The residual is retained to avoid starting the shift search from an
  // over-optimistic unconverged Ritz value.
  RitzEstimate smallest_ritz(int n, const std::vector<SymTriplet> &base) const {
    RitzEstimate out;
    if (n <= 0) {
      out.available = true;
      return out;
    }
    auto apply = [&](const Vec &x) {
      Vec y(n, 0);
      for (const auto e : base) {
        y[e.i] += e.v * x[e.j];
        if (e.i != e.j) y[e.j] += e.v * x[e.i];
      }
      for (const auto &atom : spectral_atoms) {
        if (!(atom.beta > 0) || static_cast<int>(atom.direction.size()) != n) continue;
        const double weight = atom.beta * dot(atom.direction, x);
        axpy(weight, atom.direction, y);
      }
      return y;
    };

    const int limit = std::min(n, 24);
    std::vector<Vec> basis;
    basis.reserve(limit);
    Vec v(n);
    for (int i = 0; i < n; ++i)
      v[i] = std::sin((i + 1) * 0.7548776662466927) +
             std::cos((i + 1) * 1.618033988749895);
    for (int k = 0; k < limit; ++k) {
      for (int sweep = 0; sweep < 2; ++sweep)
        for (const auto &q : basis) axpy(-dot(q, v), q, v);
      const double length = norm2(v);
      if (!(length > 1e-13) || !std::isfinite(length)) break;
      for (double &entry : v) entry /= length;
      basis.push_back(v);
      v = apply(basis.back());
    }
    const int m = static_cast<int>(basis.size());
    if (!m) return out;

    Vec projected(m * m, 0), values(m), work(std::max(1, 3 * m));
    std::vector<Vec> images;
    images.reserve(m);
    for (int j = 0; j < m; ++j) images.push_back(apply(basis[j]));
    for (int j = 0; j < m; ++j)
      for (int i = 0; i <= j; ++i) {
        const double value = 0.5 * (dot(basis[i], images[j]) +
                                    dot(basis[j], images[i]));
        projected[i + m * j] = value;
        projected[j + m * i] = value;
      }
    int info = 0, lwork = static_cast<int>(work.size());
    const char vectors = 'V', triangle = 'U';
    dsyev_(&vectors, &triangle, &m, projected.data(), &m, values.data(),
           work.data(), &lwork, &info);
    if (info || !finite_values(values)) return out;

    Vec direction(n, 0);
    for (int j = 0; j < m; ++j) axpy(projected[j], basis[j], direction);
    const double direction_norm = norm2(direction);
    if (!(direction_norm > 0) || !std::isfinite(direction_norm)) return out;
    for (double &entry : direction) entry /= direction_norm;
    Vec residual = apply(direction);
    axpy(-values.front(), direction, residual);
    out.available = std::isfinite(values.front()) && finite_values(residual);
    out.value = values.front();
    out.residual = norm2(residual);
    return out;
  }
  void compact_inactive_space() {
    const std::size_t active = spectral_atoms.size(), stored = atom_basis_.size();
    const std::size_t keep = std::max<std::size_t>(1, settings_.dca_atom_basis_keep);
    const std::size_t tail = std::max<std::size_t>(0, settings_.dca_atom_basis_tail);
    if (stored <= std::max<std::size_t>(keep, 2 * active + tail)) return;
    // Keep the exact currently applied correction; only its search-space cache
    // is compacted. The next Hessian update rebalances and recertifies as usual.
    std::vector<Vec> kept;
    auto retain = [&](Vec v) {
      for (int sweep = 0; sweep < 2; ++sweep)
        for (const auto &u : kept) axpy(-dot(u, v), u, v);
      const double length = norm2(v);
      if (!(length > 1e-8)) return;
      for (double &x : v) x /= length;
      kept.push_back(std::move(v));
    };
    for (const auto &a : spectral_atoms) retain(a.direction);
    for (std::size_t i = stored > tail ? stored - tail : 0; i < stored; ++i) retain(atom_basis_[i]);
    retired_directions += stored - kept.size();
    atom_basis_.swap(kept);
    ++basis_compressions;
  }
  bool ensure(int n, const std::vector<SymTriplet> &input, double margin, bool atom_event,
              int &remaining_atoms, int batch, double min_relative_gain,
              const std::vector<std::vector<std::pair<int, double>>> *normals = nullptr) {
    DcaTimedScope timer(ensure_seconds);
    last_peak_stored_atoms = atoms.size() + atom_basis_.size();
    const auto h = canonical(n, input);
    bool same = n == dimension_ && h.size() == previous_.size() && margin == previous_margin_;
    if (same)
      for (std::size_t k = 0; k < h.size(); ++k)
        if (h[k].i != previous_[k].i || h[k].j != previous_[k].j || h[k].v != previous_[k].v) {
          same = false;
          break;
        }
    const bool hessian_changed = !same;
    if (same && (!normals || *normals == previous_normals_)) {
      ++reuses;
      return false;
    }
    if (hessian_changed)
      ++updates;
    // A sufficient perturbation certificate can retain the ENTIRE old PSD
    // atom correction. For E=N_new-N_old, Young's inequality gives
    //   ||N_old v||^2 <= 1.25 ||N_new v||^2 + 5 ||E v||^2.
    // Together with DeltaB >= -dB I this certifies the new majorant using
    // gamma_new=1.25 gamma_old and rho_new=rho_old+dB+5 gamma_old ||E||^2.
    // No atom weight is changed and no inverse probe may use the old factor.
    bool transport_eligible = !tangent_certificate && settings_.dca_filter_certificate_transport &&
                              certificate_transport_streak < 3 && n == dimension_ &&
                              margin == previous_margin_ && h.size() == previous_.size() &&
                              normals && !previous_normals_.empty() &&
                              normals->size() == previous_normals_.size() &&
                              normal_weight > 0 && std::isfinite(normal_weight) &&
                              rho >= 0 && std::isfinite(rho);
    if (transport_eligible) {
      for (std::size_t k = 0; k < h.size(); ++k)
        if (h[k].i != previous_[k].i || h[k].j != previous_[k].j) {
          transport_eligible = false; break;
        }
      for (std::size_t row = 0; transport_eligible && row < normals->size(); ++row) {
        if ((*normals)[row].size() != previous_normals_[row].size()) {
          transport_eligible = false; break;
        }
        for (std::size_t j = 0; j < (*normals)[row].size(); ++j)
          if ((*normals)[row][j].first != previous_normals_[row][j].first ||
              (*normals)[row][j].first < 0 || (*normals)[row][j].first >= n ||
              (j && (*normals)[row][j - 1].first >= (*normals)[row][j].first)) {
            transport_eligible = false; break;
          }
      }
    }
    if (transport_eligible) {
      ++certificate_transport_trials;
      Vec diagonal(n, 0), radius(n, 0), old_rows(n, 0), new_rows(n, 0);
      for (std::size_t k = 0; k < h.size(); ++k) {
        const auto entry = h[k];
        const double difference = entry.v - previous_[k].v;
        old_rows[entry.i] += std::abs(previous_[k].v);
        new_rows[entry.i] += std::abs(entry.v);
        if (entry.i == entry.j) diagonal[entry.i] += difference;
        else {
          radius[entry.i] += std::abs(difference);
          radius[entry.j] += std::abs(difference);
          old_rows[entry.j] += std::abs(previous_[k].v);
          new_rows[entry.j] += std::abs(entry.v);
        }
      }
      double deficit = 0;
      for (int i = 0; i < n; ++i) deficit = std::max(deficit, radius[i] - diagonal[i]);
      Vec difference_columns(n, 0), old_columns(n, 0), new_columns(n, 0);
      double frobenius_squared = 0, difference_infinity = 0,
             old_infinity = 0, new_infinity = 0;
      bool normal_values_finite = true;
      for (std::size_t row = 0; row < normals->size(); ++row) {
        double difference_sum = 0, old_sum = 0, new_sum = 0;
        for (std::size_t j = 0; j < (*normals)[row].size(); ++j) {
          const auto entry = (*normals)[row][j];
          const double old_value = previous_normals_[row][j].second;
          if (!std::isfinite(entry.second) || !std::isfinite(old_value)) {
            normal_values_finite = false; break;
          }
          const double difference = entry.second - old_value;
          difference_sum += std::abs(difference);
          difference_columns[entry.first] += std::abs(difference);
          frobenius_squared += difference * difference;
          old_sum += std::abs(old_value);
          new_sum += std::abs(entry.second);
          old_columns[entry.first] += std::abs(old_value);
          new_columns[entry.first] += std::abs(entry.second);
        }
        difference_infinity = std::max(difference_infinity, difference_sum);
        old_infinity = std::max(old_infinity, old_sum);
        new_infinity = std::max(new_infinity, new_sum);
      }
      const double error_squared = std::min(frobenius_squared,
                                           inf_norm(difference_columns) * difference_infinity);
      const double scale = std::max({1.0, rho, inf_norm(old_rows), inf_norm(new_rows),
                                     normal_weight * inf_norm(old_columns) * old_infinity,
                                     1.25 * normal_weight * inf_norm(new_columns) * new_infinity});
      const double padding = 64 * std::numeric_limits<double>::epsilon() * scale;
      const double increment = deficit + 5 * normal_weight * error_squared + padding;
      const double allowed = std::min(1e-3, std::max(1e-6, 0.05 * rho));
      const double next_rho = std::nextafter(rho + increment, INFINITY);
      const double next_gamma = std::nextafter(1.25 * normal_weight, INFINITY);
      if (normal_values_finite && finite_values(diagonal) && finite_values(radius) &&
          std::isfinite(error_squared) && error_squared >= 0 &&
          std::isfinite(increment) && increment >= 0 && increment <= allowed &&
          std::isfinite(next_rho) && std::isfinite(next_gamma)) {
        rho = next_rho;
        normal_weight = next_gamma;
        previous_ = h;
        previous_normals_ = *normals;
        ++certificate_transports;
        ++certificate_transport_streak;
        return true;
      }
    }
    certificate_transport_streak = 0;
    // Reweight only after an actual Hessian change. A decrease invalidates
    // the old certificate; the complete corrected matrix is checked below.
    for (auto &a : atoms) {
      double ray = 0;
      for (auto e : h) {
        double vi = e.i == a.i ? a.u : e.i == a.j ? a.v : 0;
        double vj = e.j == a.i ? a.u : e.j == a.j ? a.v : 0;
        ray += (e.i == e.j ? 1 : 2) * e.v * vi * vj;
      }
      a.beta = std::max(0.0, -1.05 * ray);
    }
    rebalance(n, h, normals, normal_weight, margin);
    auto base = corrected(n, h, 0);
    auto lower = [&](const std::vector<SymTriplet> &a) {
      SymMatrix m;
      m.n = n;
      m.entries = a;
      return m.gershgorin_lower_bound();
    };
    double lower_bound = lower(base);
    auto admit_sparse = [&]() {
      const int additions_before = additions;
      if (lower_bound < margin && atom_event && hessian_changed && remaining_atoms > 0) {
        for (int k = 0; k < std::min(batch, remaining_atoms); ++k) {
          Vec diagonal(n, 0);
          for (auto e : base)
            if (e.i == e.j)
              diagonal[e.i] = e.v;
          DcaAtom best;
          double eig = 0;
          for (int i = 0; i < n; ++i)
            if (diagonal[i] < eig) {
              eig = diagonal[i];
              best = {i, i, 1, 0, 0};
            }
          for (auto e : base)
            if (e.i != e.j && e.v != 0) {
              const double a = diagonal[e.i], b = diagonal[e.j];
              const double l = 0.5 * (a + b - std::hypot(a - b, 2 * e.v));
              if (l < eig) {
                double u = e.v, v = l - a, norm = std::hypot(u, v);
                if (norm > 0) {
                  eig = l;
                  best = {e.i, e.j, u / norm, v / norm, 0};
                }
              }
            }
          ++probes;
          if (eig >= -std::max(1e-12, min_relative_gain * std::max(1.0, -lower_bound)))
            break;
          best.beta = -1.05 * eig;
          atoms.push_back(best);
          ++additions;
          base = corrected(n, h, 0);
          lower_bound = lower(base);
        }
        remaining_atoms -= additions - additions_before;
      }
    };
    // Optional normal completion is implicit: B+H+rho*I+gamma*A'A is
    // globally positive definite iff [B+H+rho*I-margin*I A'; A -I/gamma]
    // has n positive and m negative eigenvalues. On A*d=-h, its added
    // objective is the constant gamma*||h||^2/2. Newton uses the row-equivalent
    // unaugmented system and shifted equality duals, avoiding Gram fill.
    // Research alternative: a ZERO normal block certifies strict positivity
    // on ker(N) instead. H+rho I remains PSD, so the partial-Lagrangian
    // majorization still holds globally; convexity of the subproblem is on
    // its affine equality domain. A positive normal completion exists, but
    // its value is not computed or claimed by this mode (normal_weight=1 is
    // only an internal marker selecting the equality border).
    // Gershgorin is retained only as an emergency finite fallback when the
    // numerical inertia certificate cannot be constructed or repeatedly
    // fails. It no longer seeds, caps, or short-circuits the normal search.
    double gershgorin_fallback = std::max(0.0, margin - lower_bound) +
        (lower_bound < margin ? 1e-12 * std::max(1.0, std::abs(lower_bound)) : 0.0);
    int correction_version = 0, factored_version = -1;
    double factored_shift = 0, factored_gamma = 0;
    int assembled_version = -1, assembled_dimension = 0;
    bool assembled_normal_border = false;
    auto positive = [&](double shift, double gamma) {
      if (factored_version == correction_version && shift == factored_shift && gamma == factored_gamma) {
        ++factor_reuses;
        return true; // same successful numeric factor, also valid for inverse probes
      }
      if (!factor_) {
        std::string message;
        factor_ = make_symmetric_factorization(settings_, message);
        if(settings_.dca_pair_elimination)factor_=with_sparse_pair_elimination(std::move(factor_),
            [this](std::string& m){return make_symmetric_factorization(settings_,m);},settings_.live_display_level>=4);
        if (!factor_)
          return lower_bound + shift >= margin;
      }
      const bool normal_border = gamma > 0 && normals;
      if (assembled_version != correction_version || normal_border != assembled_normal_border) {
        factor_entries_.clear();
        factor_entries_.reserve(base.size() + 4 * n);
        primal_diagonal_slots_.clear();
        normal_diagonal_slots_.clear();
        for (auto e : base) {
          if (e.i == e.j) primal_diagonal_slots_.push_back(factor_entries_.size());
          factor_entries_.push_back({e.i, e.j, e.v + 0.0});
        }
        assembled_dimension = n;
        if (normal_border) {
          assembled_dimension += normals->size();
          for (int e = 0; e < (int)normals->size(); ++e) {
            normal_diagonal_slots_.push_back(factor_entries_.size());
            factor_entries_.push_back({n + e, n + e, 0.0});
            for (auto a : (*normals)[e])
              factor_entries_.push_back({a.first, n + e, a.second});
          }
        }
        for (const auto &atom : spectral_atoms) {
          if (!(atom.beta > 0)) continue;
          const int row = assembled_dimension++;
          factor_entries_.push_back({row, row, -1.0});
          const double weight = std::sqrt(atom.beta);
          for (int j = 0; j < n; ++j)
            factor_entries_.push_back({j, row, weight * atom.direction[j]});
        }
        assembled_version = correction_version;
        assembled_normal_border = normal_border;
      }
      for (std::size_t slot : primal_diagonal_slots_)
        factor_entries_[slot].value = base[slot].v + (shift - margin);
      for (std::size_t slot : normal_diagonal_slots_)
        factor_entries_[slot].value = tangent_certificate ? 0.0 : -1 / gamma;
      std::string message;
      const auto* pair_factor=settings_.dca_pair_elimination
          ?dynamic_cast<const PairReducedFactorization*>(factor_.get()):nullptr;
      const int underlying_before=pair_factor?pair_factor->underlying_factorizations():0;
      ++certificates;
      factored_version = -1;
      bool ok;
      { DcaTimedScope factor_time(factor_seconds); ok = factor_->factor(assembled_dimension, n, factor_entries_, message); }
      // One certificate request may factor both the reduced matrix and its
      // full fallback. Report every actual numeric factorization.
      if(pair_factor)certificates+=pair_factor->underlying_factorizations()-underlying_before-1;
      if (ok) {
        factored_version = correction_version;
        factored_shift = shift;
        factored_gamma = gamma;
      }
      return ok;
    };
    auto ritz_proposal = [&]() {
      ++ritz_shift_proposals;
      const RitzEstimate estimate = smallest_ritz(n, base);
      const double floor = estimate.available
          ? estimate.value - 2.0 * std::max(0.0, estimate.residual)
          : margin;
      return std::max(2 * margin + 1e-12,
                      estimate.available ? margin - floor : 2 * margin + 1e-12);
    };
    auto certify_positive_shift = [&](double gamma, double initial) {
      double trial = std::max(2 * margin + 1e-12, initial);
      for (int k = 0; k < 32; ++k) {
        if (positive(trial, gamma)) {
          rho = trial;
          return true;
        }
        if (!std::isfinite(trial) || trial > std::numeric_limits<double>::max() / 10)
          break;
        trial *= 10;
      }
      // Ultimate proof-oriented escape hatch. This is intentionally outside
      // the Ritz/inertia loop and must not constrain successful normal runs.
      if (std::isfinite(gershgorin_fallback) &&
          positive(gershgorin_fallback, gamma)) {
        ++gershgorin_fallback_uses;
        rho = gershgorin_fallback;
        return true;
      }
      return false;
    };
    const double previous_rho = rho, previous_normal_weight = normal_weight;
    normal_weight = 0;
    if (!normals && n <= 16 && lower_bound < margin) {
      admit_sparse();
      // Small constant inequality blocks are certified once, exactly by LAPACK.
      Vec dense(n * n, 0), values(n), work(std::max(1, 3 * n));
      for (auto e : base) {
        dense[e.i + n * e.j] = e.v;
        dense[e.j + n * e.i] = e.v;
      }
      int info = 0, lwork = work.size();
      const char no = 'N', upper_triangle = 'U';
      dsyev_(&no, &upper_triangle, &n, dense.data(), &n, values.data(), work.data(), &lwork, &info);
      ++small_spectra;
      if (info != 0 || !std::isfinite(values[0]))
        throw std::runtime_error("DCA small-block curvature certificate failed");
      const double padding =
          64 * std::numeric_limits<double>::epsilon() * std::max({1.0, std::abs(values.front()), std::abs(values.back())});
      rho = std::max(0.0, margin - values[0] + padding);
    } else {
      bool zero = false;
      if (normals && !normals->empty()) {
        if (tangent_certificate) {
          normal_weight = 1; // implicit extension, not a numerical gamma
          zero = positive(0, normal_weight);
        } else {
          for (double gamma = std::min(1e6, std::max(1e2, previous_normal_weight)); gamma <= 1e6; gamma *= 100) {
            normal_weight = gamma;
            if (positive(0, gamma)) {
              zero = true;
              break;
            }
          }
        }
      } else
        zero = positive(0, 0);
      if (!zero && !normals) {
        admit_sparse();
        ++correction_version;
        gershgorin_fallback = std::max(0.0, margin - lower_bound) +
            1e-12 * std::max(1.0, std::abs(lower_bound));
        zero = positive(0, normal_weight);
      }
      if (zero)
        rho = 0;
      else {
        // Lanczos/Ritz proposes the scale; inertia alone accepts it. Retain a
        // decayed successful shift as a warm start when it is more cautious.
        // A full-space Ritz value is not a tangent-space Ritz value. With an
        // equality border, let bordered inertia bracket the tangent minimum;
        // the subsequent inverse Ritz probe is already projected onto ker(A).
        const bool bordered = normals && !normals->empty();
        const double initial = bordered
            ? std::max(2 * margin + 1e-12, 0.2 * previous_rho)
            : std::max(ritz_proposal(), 0.2 * previous_rho);
        if (!certify_positive_shift(normal_weight, initial))
          throw std::runtime_error("DCA Ritz/inertia curvature certificate failed");
      }
    }
    // Every normal-path endpoint has a full inertia certificate for the
    // CURRENT correction. Gershgorin can appear only in the escape hatch.
    auto refine_complement = [&]() {
      if (!(rho > 0)) return;
      double low = 0, high = rho;
      for (int k = 0; k < settings_.dca_filter_rho_refinements; ++k) {
        ++rho_refinement_trials;
        const double trial = 0.5 * (low + high);
        if (positive(trial, normal_weight)) { high = trial; ++rho_refinement_acceptances; }
        else low = trial;
      }
      rho = high;
    };
    // Research control: use one inverse Krylov space for the entire admission
    // batch. The individual Rayleigh gates below still inspect the CURRENT
    // correction after every accepted direction. Only the final correction is
    // exposed to the model, and it receives the same full-space certificate.
    const bool block_ritz = settings_.dca_filter_block_ritz;
    const bool defer_refinement = block_ritz && rho > 0 && atom_event &&
                                  hessian_changed && remaining_atoms > 0 && batch > 0 && normals;
    if (!defer_refinement) refine_complement();
    const int additions_before_refinement = additions;
    if (rho > 0 && atom_event && hessian_changed && remaining_atoms > 0 && normals) {
      const int allowance = std::min(batch, remaining_atoms);
      for (int attempt = 0; attempt < allowance && rho > 0; ++attempt) {
        if (!positive(rho, normal_weight))
          break;
        // Reuse the certified shifted operator. Its largest inverse
        // eigenvalue targets the MOST NEGATIVE unshifted curvature, whereas
        // an unshifted inverse can be dominated by tiny positive eigenvalues.
        // A direct Rayleigh test alone authorizes the new PSD atom.
        const int dim = n + (normal_weight > 0 ? (int)normals->size() : 0) +
                        (int)std::count_if(spectral_atoms.begin(), spectral_atoms.end(),
                                           [](const auto &atom) { return atom.beta > 0; });
        std::vector<Vec> basis, images;
        Vec v(n);
        for (int j = 0; j < n; ++j)
          v[j] = std::sin((j + 1) * 1.618033988749895);
        int projection_solves = 0;
        if (tangent_certificate && normal_weight > 0) {
          // The primal block of this inverse maps every RHS into ker(N).
          // Start there to avoid an artificial normal component from the
          // initial random vector in the finite Krylov/Ritz approximation.
          Vec rhs(dim, 0);
          std::copy(v.begin(), v.end(), rhs.begin());
          std::string message;
          ++inverse_probes;
          bool solved;
          { DcaTimedScope inverse_time(inverse_seconds); solved = factor_->solve(rhs, message); }
          if (!solved) break;
          v.assign(rhs.begin(), rhs.begin() + n);
          projection_solves = 1;
        }
        const int probe_depth = std::max(1, settings_.dca_atom_probe_depth -
            (settings_.dca_atom_probe_count_projection ? projection_solves : 0));
        for (int it = 0; it < std::min(n, probe_depth); ++it) {
          const double length = norm2(v);
          if (!(length > 1e-12))
            break;
          for (double &z : v)
            z /= length;
          basis.push_back(v);
          Vec rhs(dim, 0);
          std::copy(v.begin(), v.end(), rhs.begin());
          std::string message;
          ++inverse_probes;
          bool solved;
          { DcaTimedScope inverse_time(inverse_seconds); solved = factor_->solve(rhs, message); }
          if (!solved) {
            basis.pop_back();
            break;
          }
          images.emplace_back(rhs.begin(), rhs.begin() + n);
          v = images.back();
          for (int sweep = 0; sweep < 2; ++sweep)
            for (const auto &u : basis)
              axpy(-dot(u, v), u, v);
        }
        const int m = basis.size();
        if (m == 0)
          break;
        Vec reduced(m * m), values(m), work(std::max(1, 3 * m));
        for (int j = 0; j < m; ++j)
          for (int i = 0; i < m; ++i)
            reduced[i + m * j] = 0.5 * (dot(basis[i], images[j]) + dot(basis[j], images[i]));
        int info = 0, lwork = work.size();
        const char vectors = 'V', triangle = 'U';
        dsyev_(&vectors, &triangle, &m, reduced.data(), &m, values.data(), work.data(), &lwork,
               &info);
        if (info || !(values[m - 1] > 0))
          break;
        int admitted_from_probe = 0;
        const int candidate_count = block_ritz ? m : 1;
        for (int candidate = 0; candidate < candidate_count &&
                                admitted_from_probe < (block_ritz ? allowance : 1); ++candidate) {
          const int column = m - 1 - candidate;
          if (!(values[column] > 0)) continue;
          Vec direction(n, 0);
          for (int j = 0; j < m; ++j)
            axpy(reduced[j + m * column], basis[j], direction);
          double length = norm2(direction);
          if (!(length > 0)) continue;
          for (double &x : direction) x /= length;
          double ray = 0;
          for (auto e : base)
            ray += (e.i == e.j ? 1 : 2) * e.v * direction[e.i] * direction[e.j];
          double tangent_residual = 0;
          for (const auto &row : *normals) {
            double v = 0;
            for (auto a : row) v += a.second * direction[a.first];
            if (tangent_certificate) tangent_residual = std::isfinite(v)
                ? std::max(tangent_residual, std::abs(v)) : INFINITY;
            else ray += normal_weight * v * v;
          }
          if (tangent_certificate &&
              tangent_residual > 1e-7 * std::max(1.0, norm2(direction))) continue;
          // A prior candidate may have been reweighted into this direction.
          // Never infer another admission from the old projected eigenvalue.
          for (const auto &a : spectral_atoms) {
            double v = dot(a.direction, direction);
            ray += a.beta * v * v;
          }
          ++probes;
          if (!std::isfinite(ray) || ray >= -std::max(1e-10, min_relative_gain)) continue;
          for (int sweep = 0; sweep < 2; ++sweep)
            for (const auto &u : atom_basis_) axpy(-dot(u, direction), u, direction);
          const double independent = norm2(direction);
          if (!(independent > 1e-6)) { ++dependent_witnesses; continue; }
          for (double &v : direction) v /= independent;
          atom_basis_.push_back(std::move(direction));
          rebalance(n, h, normals, normal_weight, margin);
          ++correction_version;
          --remaining_atoms;
          ++additions;
          ++admitted_from_probe;
        }
        if (!admitted_from_probe) break;
        if (positive(0, normal_weight)) {
          rho = 0;
        } else {
          // Rotating/reducing old weights is not a monotone PSD update.
          // The pre-admission shift MUST be recertified after rebalancing.
          const bool bordered = normals && !normals->empty();
          double initial = std::max(rho, 2 * margin + 1e-12);
          if (!bordered) {
            const RitzEstimate updated_ritz = smallest_ritz(n, base);
            ++ritz_shift_proposals;
            const double updated_floor = updated_ritz.available
                ? updated_ritz.value - 2.0 * std::max(0.0, updated_ritz.residual)
                : margin;
            initial = std::max(initial, updated_ritz.available
                ? margin - updated_floor : 0.0);
          }
          if (!certify_positive_shift(normal_weight, initial))
            throw std::runtime_error("DCA post-atom Ritz/inertia certificate failed");
        }
        if (block_ritz) break; // no stale factor is used after batch rebalancing
      }
    }

    if (defer_refinement || additions != additions_before_refinement) refine_complement();
    last_peak_stored_atoms = std::max(last_peak_stored_atoms, int(atoms.size() + atom_basis_.size()));
    compact_inactive_space();
    previous_ = h;
    dimension_ = n;
    previous_margin_ = margin;
    if (normals)
      previous_normals_ = *normals;
    return true;
  }
};
} // namespace lazyipdca::qcqp::detail
