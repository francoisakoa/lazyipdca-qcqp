#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/condensed_retraction.hpp"

#include <iostream>
#include <limits>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem clustered_problem() {
  constexpr int n = 513;
  const double inf = std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name = "projected_multi_ritz_inertia";
  p.n = n;
  p.x0.assign(n, 0.0);
  p.x0[0] = 1.0;
  p.lb.assign(n, -inf);
  p.ub.assign(n, inf);
  p.objective.name = "clustered_indefinite_diagonal";
  p.objective.linear.assign(n, 0.0);
  p.objective.linear[0] = 1000.0;
  p.objective.linear[1] = -1.0;
  p.objective.hessian.n = n;
  for (int j = 0; j < n; ++j) {
    const double diagonal = j <= 1 ? -1000.0 :
        (j == 2 ? -500.0 : (j == 3 ? -100.0 : 10.0 + 0.01 * j));
    p.objective.hessian.entries.push_back({j, j, diagonal});
  }
  QuadraticFunction h;
  h.name = "unit_first_coordinate";
  h.constant = -1.0;
  h.linear.assign(n, 0.0);
  h.hessian.n = n;
  h.hessian.entries.push_back({0, 0, 2.0});
  p.equalities.push_back(std::move(h));
  p.validate();
  return p;
}

}  // namespace

int main() {
  try {
    Settings st;
    st.verbose = false;
    st.outer_algorithm = "lazy-dca";
    st.inner_backend = "lazyqp";
    st.linear_backend = "native";
    st.max_outer_iterations = 1;
    st.max_inner_iterations = 100;
    st.time_limit_seconds = 30.0;
    st.feasibility_tolerance = 1e-6;
    st.stationarity_tolerance = 1e-6;
    st.dual_infeasibility_tolerance = 1e-6;
    st.complementarity_tolerance = 1e-6;
    st.dca_equality_model = "condensed-retraction";
    st.dca_condensed_tangent_curvature = true;
    st.dca_condensed_projected_multi_ritz_inertia = true;
    st.exact_eigen_threshold = 256;
    st.dca_condensed_lanczos_steps = 24;
    st.dca_condensed_lanczos_starts = 2;
    st.dca_condensed_max_tangent_rank = 8;
    st.dca_condensed_scalar_safeguard_updates = 8;
    st.curvature_safety = 0.1;
    st.random_seed = 3;
    st.dca_local_corrector = false;
    st.dca_iterated_soc = false;
    st.dca_equality_retraction = false;

    const QCQPProblem p = clustered_problem();
    CondensedQuadraticRetraction geometry(st);
    std::string message;
    if (!geometry.factor(p, p.x0, message)) {
      std::cerr << "[FAIL] normal factorization: " << message << "\n";
      return 1;
    }
    Convexification none;
    if (geometry.certify_tangent_convex_by_inertia(
            p.objective.hessian, none, -1e-8, message)) {
      std::cerr << "[FAIL] inertia accepted a known indefinite tangent Hessian\n";
      return 1;
    }

    const Result r = Solver(st).solve(p);
    if (r.lazy_dca_condensed_projected_multi_ritz_batches < 1 ||
        r.lazy_dca_condensed_projected_multi_ritz_batches > 2 ||
        r.lazy_dca_condensed_tangent_rank_max < 1 ||
        r.lazy_dca_condensed_projected_multi_ritz_fallbacks != 0 ||
        r.lazy_dca_condensed_tangent_inertia_attempts < 1 ||
        r.lazy_dca_condensed_tangent_inertia_acceptances < 1) {
      std::cerr << "[FAIL] multi-Ritz/inertia route contract: batches="
                << r.lazy_dca_condensed_projected_multi_ritz_batches
                << " rank=" << r.lazy_dca_condensed_tangent_rank_max
                << " fallback="
                << r.lazy_dca_condensed_projected_multi_ritz_fallbacks
                << " inertia="
                << r.lazy_dca_condensed_tangent_inertia_attempts << "/"
                << r.lazy_dca_condensed_tangent_inertia_acceptances
                << " message=" << r.message << "\n";
      return 1;
    }
    std::cout << "[PASS] projected multi-Ritz batches="
              << r.lazy_dca_condensed_projected_multi_ritz_batches
              << " rank=" << r.lazy_dca_condensed_tangent_rank_max
              << " inertia="
              << r.lazy_dca_condensed_tangent_inertia_attempts << "/"
              << r.lazy_dca_condensed_tangent_inertia_acceptances << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
