#include "lazyipdca_qcqp/curvature.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool near(double left, double right, double tolerance = 1e-10) {
  return std::abs(left - right) <=
         tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

Vec apply_convexification(const Convexification& cvx, const Vec& x) {
  Vec result(x.size(), 0.0);
  for (std::size_t i = 0; i < x.size(); ++i)
    result[i] = cvx.scalar_shift * x[i];
  for (const auto& term : cvx.terms)
    axpy(term.beta * dot(term.direction, x), term.direction, result);
  return result;
}

void test_global_block_merge_and_rank_cap() {
  Settings settings;
  settings.verbose = false;
  settings.exact_eigen_threshold = 2;
  settings.max_rank_per_quadratic = 8;
  settings.curvature_tolerance = 1e-12;
  settings.curvature_safety = 0.10;
  CurvatureOracle oracle(settings);

  SymMatrix q;
  q.n = 8;
  q.entries = {
      {0, 0, -4.0},
      // These duplicates cancel numerically, but both stored nonzeros must
      // keep coordinates 0 and 1 in one structural component.
      {0, 1, 2.0},
      {0, 1, -2.0},
      {1, 1, -1.0},
      {2, 2, -3.0},
      {2, 3, 0.5},
      {3, 3, 2.0},
      {4, 4, 5.0},
      // Stored numerical zeros neither activate nor connect coordinates.
      {5, 6, 0.0},
  };

  Convexification cvx;
  const auto result = oracle.ensure_convex_block_exact(q, cvx, 0.0, 2);
  require(result.available, "block-exact certificate unexpectedly unavailable");
  require(result.block_count == 3, "structural duplicate partition is wrong");
  require(result.active_dimension == 5, "active structural dimension is wrong");
  require(result.max_block_dimension == 2, "maximum block dimension is wrong");
  require(result.partial_spectrum_attempts == 2 &&
              result.partial_spectrum_acceptances == 2 &&
              result.partial_spectrum_fallbacks == 0,
          "partial-spectrum block counters are wrong");
  require(oracle.partial_spectrum_workspace_queries() == 1,
          "equal-size blocks did not reuse one DSYEVR workspace query");
  require(result.rank_added == 2 && cvx.rank() == 2,
          "global rank budget was not enforced");
  require(near(result.initial_min_eigenvalue, -4.0),
          "initial exact minimum is wrong");

  // The two global lowest modes are -4 on coordinate 0 and the negative
  // eigenmode of block {2,3}.  The remaining -1 mode is handled by exactly
  // the same scalar safeguard as the monolithic exact path.
  require(std::abs(cvx.terms[0].direction[0]) > 1.0 - 1e-10,
          "first global curvature mode is wrong");
  require(std::abs(cvx.terms[1].direction[2]) > 1e-3 &&
              std::abs(cvx.terms[1].direction[3]) > 1e-3,
          "second correction did not come from the globally lower block");
  for (int index : {0, 1, 4, 5, 6, 7}) {
    require(std::abs(cvx.terms[1].direction[static_cast<std::size_t>(index)]) <
                1e-12,
            "block eigenvector leaked outside its structural component");
  }
  require(near(result.scalar_shift_added, 1.1),
          "rank-cap scalar safeguard differs from the exact path");
  require(near(cvx.scalar_shift, result.scalar_shift_added),
          "reported scalar increment differs from the model");
  require(near(result.final_min_eigenvalue, 0.1),
          "algebraic final exact minimum is wrong");

  Settings dense_settings = settings;
  dense_settings.exact_eigen_threshold = q.n;
  CurvatureOracle dense_oracle(dense_settings);
  const EigenEstimate verified = dense_oracle.smallest_eigenpair(q, cvx);
  require(verified.exact, "dense verification did not use the exact oracle");
  require(near(verified.lambda_min, result.final_min_eigenvalue),
          "reported block certificate disagrees with dense LAPACK");
  require(verified.lambda_min >= -settings.curvature_tolerance,
          "block-exact convexification is not certified PSD");
}

void test_zero_spectrum_outside_support() {
  Settings settings;
  settings.verbose = false;
  settings.exact_eigen_threshold = 2;
  settings.curvature_tolerance = 1e-12;
  settings.curvature_safety = 0.10;
  CurvatureOracle oracle(settings);

  SymMatrix q;
  q.n = 4;
  q.entries = {{0, 0, 2.0}};
  Convexification cvx;
  const double target = 0.5 * settings.curvature_tolerance;
  const auto result = oracle.ensure_convex_block_exact(q, cvx, target, 4);
  require(result.available, "zero-support spectrum was not certified");
  require(result.block_count == 1 && result.active_dimension == 1 &&
              result.max_block_dimension == 1,
          "zero-support structural statistics are wrong");
  require(result.rank_added == 0 && cvx.rank() == 0,
          "outside zeros consumed low-rank corrections");
  require(near(result.initial_min_eigenvalue, 0.0),
          "outside zero eigenvalues were omitted from the initial spectrum");
  const double expected_shift = target * (1.0 + settings.curvature_safety);
  require(near(result.scalar_shift_added, expected_shift, 1e-12),
          "outside zero eigenspace did not receive the exact scalar safeguard");
  require(near(result.final_min_eigenvalue, expected_shift, 1e-12),
          "outside zero eigenspace is missing from the final spectrum");

  SymMatrix zero;
  zero.n = 6;
  zero.entries = {{1, 4, 0.0}};
  Convexification zero_cvx;
  const auto zero_result =
      oracle.ensure_convex_block_exact(zero, zero_cvx, 0.0, 0);
  require(zero_result.available && zero_result.block_count == 0 &&
              zero_result.active_dimension == 0 &&
              zero_result.max_block_dimension == 0 &&
              zero_result.rank_added == 0 &&
              zero_result.scalar_shift_added == 0.0 &&
              zero_result.initial_min_eigenvalue == 0.0 &&
              zero_result.final_min_eigenvalue == 0.0,
          "all-zero structural spectrum contract failed");
}

void test_case118_scale_structural_shape() {
  Settings settings;
  settings.verbose = false;
  settings.exact_eigen_threshold = 256;
  settings.curvature_tolerance = 1e-12;
  settings.curvature_safety = 0.10;
  CurvatureOracle oracle(settings);

  SymMatrix q;
  q.n = 1088;
  q.entries.reserve(2 * 235 + 236 + 54);
  // One case118-scale structural chain.  Cancelled numerical edges keep the
  // spectrum diagonal and analytically checkable while preserving a genuine
  // 236-coordinate structural component for the partitioner.
  for (int index = 0; index + 1 < 236; ++index) {
    q.entries.push_back({index, index + 1, 1.0});
    q.entries.push_back({index, index + 1, -1.0});
  }
  for (int index = 0; index < 236; ++index) {
    double diagonal = 2.0;
    if (index == 0) diagonal = -10.0;
    if (index == 1) diagonal = -8.0;
    if (index == 2) diagonal = -6.0;
    q.entries.push_back({index, index, diagonal});
  }
  // Fifty-four independent scalar generator-style blocks.  Two negative
  // modes interleave globally with the chain modes.
  for (int offset = 0; offset < 54; ++offset) {
    double diagonal = 3.0;
    if (offset == 0) diagonal = -9.0;
    if (offset == 1) diagonal = -7.0;
    const int index = 236 + offset;
    q.entries.push_back({index, index, diagonal});
  }

  Convexification cvx;
  const int calls_before = oracle.calls();
  const auto result = oracle.ensure_convex_block_exact(q, cvx, 0.0, 3);
  require(result.available, "case118-scale block certificate unavailable");
  require(oracle.calls() - calls_before == 1,
          "one structural certificate counted as multiple oracle calls");
  require(result.block_count == 55 && result.active_dimension == 290 &&
              result.max_block_dimension == 236,
          "case118-scale structural shape was not recovered");
  require(result.partial_spectrum_attempts == 1 &&
              result.partial_spectrum_acceptances == 1 &&
              result.partial_spectrum_fallbacks == 0,
          "case118-scale block did not use one partial spectrum");
  require(oracle.partial_spectrum_workspace_queries() == 1,
          "case118-scale block did not initialize one DSYEVR workspace");
  require(result.rank_added == 3 && cvx.rank() == 3,
          "case118-scale global rank cap is wrong");
  require(near(result.initial_min_eigenvalue, -10.0),
          "case118-scale initial minimum is wrong");
  require(near(result.scalar_shift_added, 7.7),
          "case118-scale scalar safeguard is wrong");
  require(near(result.final_min_eigenvalue, 0.7),
          "case118-scale final exact minimum is wrong");
  require(result.final_min_eigenvalue >= -settings.curvature_tolerance,
          "case118-scale result is not analytically PSD");

  // Global order must be chain(-10), singleton(-9), chain(-8).
  require(std::abs(cvx.terms[0].direction[0]) > 1.0 - 1e-10 &&
              std::abs(cvx.terms[1].direction[236]) > 1.0 - 1e-10 &&
              std::abs(cvx.terms[2].direction[1]) > 1.0 - 1e-10,
          "case118-scale modes were not merged in global eigenvalue order");

  Convexification repeated;
  const auto repeated_result =
      oracle.ensure_convex_block_exact(q, repeated, 0.0, 3);
  require(repeated_result.available &&
              repeated_result.partial_spectrum_acceptances == 1,
          "repeated case118-scale partial spectrum failed");
  require(oracle.partial_spectrum_workspace_queries() == 1,
          "repeated case118-scale call queried DSYEVR workspace again");
}

void test_partial_spectrum_matches_full_spectrum_beyond_rank_cap() {
  Settings settings;
  settings.verbose = false;
  settings.exact_eigen_threshold = 16;
  settings.curvature_tolerance = 1e-12;
  settings.curvature_safety = 0.10;

  SymMatrix q;
  q.n = 7;
  for (int i = 0; i < q.n; ++i)
    q.entries.push_back({i, i, -5.0 + 1.7 * i});
  // A connected block with a simple spectrum makes the low-rank correction
  // operator directly comparable despite the arbitrary signs of eigenvectors.
  for (int i = 0; i + 1 < q.n; ++i)
    q.entries.push_back({i, i + 1, 0.13 + 0.01 * i});

  CurvatureOracle partial_oracle(settings);
  Convexification partial;
  const auto result =
      partial_oracle.ensure_convex_block_exact(q, partial, 0.0, 2);
  require(result.available, "partial-spectrum certificate unavailable");
  require(result.rank_added == 2 && result.scalar_shift_added > 0.0,
          "fixture did not exceed the low-rank cap");
  require(result.partial_spectrum_attempts == 1 &&
              result.partial_spectrum_acceptances == 1 &&
              result.partial_spectrum_fallbacks == 0,
          "partial-spectrum kernel was not accepted exactly once");

  CurvatureOracle full_oracle(settings);
  Convexification full;
  full_oracle.ensure_convex(q, full, 0.0, 2);
  require(full.rank() == partial.rank(),
          "partial and full spectra used different low-rank counts");
  require(near(full.scalar_shift, partial.scalar_shift, 1e-9),
          "partial and full spectra selected different scalar safeguards");

  for (int probe = 0; probe < 4; ++probe) {
    Vec x(static_cast<std::size_t>(q.n), 0.0);
    for (int i = 0; i < q.n; ++i)
      x[static_cast<std::size_t>(i)] =
          std::sin(0.37 * (1 + i) * (1 + probe));
    const Vec partial_action = apply_convexification(partial, x);
    const Vec full_action = apply_convexification(full, x);
    for (int i = 0; i < q.n; ++i)
      require(near(partial_action[static_cast<std::size_t>(i)],
                   full_action[static_cast<std::size_t>(i)], 2e-8),
              "partial-spectrum correction differs from full DSYEV");
  }

  const EigenEstimate partial_min =
      full_oracle.smallest_eigenpair(q, partial);
  const EigenEstimate full_min = full_oracle.smallest_eigenpair(q, full);
  require(near(partial_min.lambda_min, full_min.lambda_min, 2e-8) &&
              partial_min.lambda_min >= -settings.curvature_tolerance,
          "partial spectrum lost the full-spectrum PSD certificate");
}

void test_fail_closed_guards() {
  Settings settings;
  settings.verbose = false;
  settings.exact_eigen_threshold = 2;
  settings.curvature_tolerance = 1e-12;
  CurvatureOracle oracle(settings);

  SymMatrix diagonal;
  diagonal.n = 3;
  diagonal.entries = {{0, 0, -1.0}};

  Convexification with_term;
  Vec direction(3, 0.0);
  direction[0] = 1.0;
  with_term.terms.push_back({2.0, direction});
  const Convexification before_term = with_term;
  const auto term_result =
      oracle.ensure_convex_block_exact(diagonal, with_term, 0.0, 2);
  require(!term_result.available, "existing low-rank coupling was accepted");
  require(with_term.rank() == before_term.rank() &&
              with_term.scalar_shift == before_term.scalar_shift &&
              with_term.terms[0].beta == before_term.terms[0].beta &&
              with_term.terms[0].direction == before_term.terms[0].direction,
          "low-rank guard mutated the convexification");

  Convexification insufficient_zero_shift;
  const auto target_result = oracle.ensure_convex_block_exact(
      diagonal, insufficient_zero_shift,
      2.0 * settings.curvature_tolerance, 2);
  require(!target_result.available,
          "uncertified outside-zero target was accepted");
  require(insufficient_zero_shift.rank() == 0 &&
              insufficient_zero_shift.scalar_shift == 0.0,
          "target guard mutated the convexification");

  SymMatrix oversized;
  oversized.n = 5;
  oversized.entries = {{0, 1, 1.0}, {1, 2, 1.0}, {3, 3, -1.0}};
  Convexification oversized_cvx;
  const auto oversized_result =
      oracle.ensure_convex_block_exact(oversized, oversized_cvx, 0.0, 2);
  require(!oversized_result.available,
          "oversized structural component was diagonalized");
  require(oversized_result.block_count == 2 &&
              oversized_result.active_dimension == 4 &&
              oversized_result.max_block_dimension == 3,
          "oversized refusal did not report structural statistics");
  require(oversized_cvx.rank() == 0 && oversized_cvx.scalar_shift == 0.0,
          "oversized-component guard mutated the convexification");
}

}  // namespace

int main() {
  try {
    test_global_block_merge_and_rank_cap();
    test_zero_spectrum_outside_support();
    test_case118_scale_structural_shape();
    test_partial_spectrum_matches_full_spectrum_beyond_rank_cap();
    test_fail_closed_guards();
    std::cout << "[PASS] exact structural-block curvature certificate\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << "\n";
    return 1;
  }
}
