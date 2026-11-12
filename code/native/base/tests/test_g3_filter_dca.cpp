#include "lazyipdca_qcqp/dca_filter.hpp"
#include "../src/dca_filter_contract.hpp"
#include "../src/dca_curvature.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
QuadraticFunction q(int n, Vec g, std::vector<SymTriplet> h, double c = 0) {
  QuadraticFunction f;
  f.linear = g;
  f.hessian.n = n;
  f.hessian.entries = h;
  f.constant = c;
  return f;
}
Settings settings() {
  Settings s;
  s.dca_engine = "filter";
  s.linear_backend = "native";
  s.max_outer_iterations = 500;
  s.live_display_level = 0;
  s.dca_filter_atom_every = 2;
  s.dca_filter_max_atoms = 5;
  s.feasibility_tolerance = s.stationarity_tolerance = s.dual_infeasibility_tolerance =
      s.complementarity_tolerance = 1e-6;
  return s;
}
int main() {
  try {
    detail::OriginalDcaFilter filter;
    detail::OriginalFilterPoint base{2, 1, 1};
    require(!filter.acceptable(base, {3, 1, 0.1}, 1, 1e-6), "model-only progress accepted");
    require(filter.acceptable(base, {3, 0.5, 1}, 0, 1e-6),
            "original feasibility progress rejected");
    require(!filter.acceptable({1, 0, 1}, {1, 0, 1}, 0, 1e-6), "unchanged feasible point accepted");
    require(filter.acceptable({1, 0, 1}, {1, 0, 0.1}, 0, 1e-6),
            "terminal original KKT contraction rejected");
    require(filter.acceptable({1, 0, 0.1, 1}, {1.00000001, 1e-9, 0.01, 0.99999999}, 0, 1e-6),
            "fixed-multiplier original Lagrangian safeguard rejected");
    require(!filter.acceptable({1, 0, 0.1, 1}, {1.00000001, 1e-9, 0.2, 0.99999999}, 0, 1e-6),
            "terminal safeguard accepted worse original KKT");
    Settings s = settings();
    detail::DcaCurvature c(s);
    int available = 1;
    std::vector<SymTriplet> h = {{0, 0, 1}, {0, 1, 2}, {1, 1, 1}};
    c.ensure(2, h, 1e-8, true, available, 2, 1e-6);
    require(available == 0 && c.atoms.size() == 1, "atom cap not enforced");
    auto a = c.corrected(2, h, c.rho);
    double d0 = 0, d1 = 0, o = 0;
    for (auto e : a) {
      if (e.i == 0 && e.j == 0)
        d0 = e.v;
      else if (e.i == 1 && e.j == 1)
        d1 = e.v;
      else
        o = e.v;
    }
    require(d0 > 0 && d1 > 0 && d0 * d1 > o * o, "corrected objective not positive definite");
    int calls = c.certificates;
    c.ensure(2, h, 1e-8, true, available, 2, 1e-6);
    require(c.certificates == calls && c.reuses == 1, "constant Hessian refactorized");
    c.ensure(2, {{0, 0, 1}, {1, 1, 1}}, 1e-8, false, available, 2, 1e-6);
    require(c.rho == 0 && c.atoms[0].beta == 0, "unnecessary curvature not retired");
    detail::DcaCurvature noatoms(s);
    available = 0;
    noatoms.ensure(2, h, 1e-8, true, available, 2, 1e-6);
    require(noatoms.atoms.empty() && noatoms.rho > 1, "scalar completion missing at zero atom cap");
    detail::DcaCurvature normal(s);
    available = 0;
    std::vector<std::vector<std::pair<int, double>>> arows = {{{0, 1}}};
    normal.ensure(2, {{0, 0, -2}, {1, 1, 1}}, 1e-8, true, available, 2, 1e-6, &arows);
    require(normal.rho == 0 && normal.normal_weight > 2,
            "normal-only negative curvature damped tangent space");
    require(-2 + normal.normal_weight > 0,
            "implicit normal completion not globally positive definite");
    auto changed = arows;
    changed[0][0].second = 0;
    normal.ensure(2, {{0, 0, -2}, {1, 1, 1}}, 1e-8, false, available, 2, 1e-6, &changed);
    require(normal.rho > 2, "changed equality Jacobian reused stale normal certificate");
    detail::DcaCurvature spectral(s);
    available = 2;
    std::vector<SymTriplet> diagonal = {{0, 0, -2}, {1, 1, -1}, {2, 2, 3}, {3, 3, 0.1}};
    std::vector<std::vector<std::pair<int, double>>> normal4 = {{{3, 1}}};
    spectral.ensure(4, diagonal, 1e-8, true, available, 2, 1e-6, &normal4);
    require(spectral.inverse_probes > 0 && spectral.spectral_atoms.size() == 2 && available == 0,
            "shifted inverse atom admission did not run");
    require(spectral.rho == 0, "two negative directions left an unnecessary scalar complement");
    // Independent dense eigenspectrum of the completed model, including borders.
    Vec matrix(16, 0), eigenvalues(4), work(12);
    for (auto e : diagonal)
      matrix[e.i + 4 * e.j] = e.v;
    for (const auto &atom : spectral.spectral_atoms)
      for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
          matrix[i + 4 * j] += atom.beta * atom.direction[i] * atom.direction[j];
    matrix[15] += spectral.normal_weight;
    int n4 = 4, lwork = 12, info = 0;
    const char no = 'N', upper = 'U';
    dsyev_(&no, &upper, &n4, matrix.data(), &n4, eigenvalues.data(), work.data(), &lwork, &info);
    require(info == 0 && eigenvalues[0] > 1e-8,
            "bordered atomic curvature certificate is incorrect");
    bool rejected = false;
    try {
      detail::DcaCurvature invalid(s);
      invalid.ensure(1, {{0, 0, std::numeric_limits<double>::infinity()}}, 0, false, available, 2,
                     1e-6);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "nonfinite curvature was accepted");
    QCQPProblem p;
    p.n = 2;
    p.name = "g3_box_qp";
    p.x0 = {0.2, 0.3};
    p.lb = {0, 0};
    p.ub = {3, 3};
    p.objective = q(2, {-2, -4}, {{0, 0, 2}, {1, 1, 2}}, 5);
    auto r = solve_filter_dca(p, s);
    require(r.status == Status::Solved, r.message.c_str());
    require(std::abs(r.x[0] - 1) < 1e-5 && std::abs(r.x[1] - 2) < 1e-5,
            "convex QP minimizer wrong");
    require(r.dca_filter.used && r.effective_outer_algorithm == "lazy-dca" && !r.atomic_engine_used,
            "algorithm identity incorrect");
    require(r.dca_filter.first_step_acceptances > 0, "first acceptable private step not exercised");
    p.name = "g3_curved_equality";
    p.x0 = {0.3, 0.09};
    p.lb = {0.1, 0.01};
    p.ub = {3, 9};
    p.objective = q(2, {-1, -1}, {{0, 0, 1}, {1, 1, 1}}, 1);
    p.equalities = {q(2, {0, -1}, {{0, 0, 2}})};
    r = solve_filter_dca(p, s);
    require(r.status == Status::Solved, r.message.c_str());
    require(std::abs(r.x[0] - 1) < 2e-5 && std::abs(r.x[1] - 1) < 2e-5,
            "curved equality optimum wrong");
    require(r.feasibility.equality_inf < 1e-6 && r.dual_infeasibility_inf < 1e-6,
            "original equality KKT failure");
    p.name = "g3_nonconvex_qp";
    p.x0 = {0.2, 0.2};
    p.lb = {-1, -1};
    p.ub = {1, 1};
    p.equalities = {q(2, {1, -1}, {})};
    p.objective = q(2, {0, 0}, {{0, 0, -2}, {1, 1, -2}});
    r = solve_filter_dca(p, s);
    require(r.status == Status::Solved, r.message.c_str());
    require(r.objective < -1.99999, "nonconvex QP did not improve original objective");
    require(r.dca_filter.atoms_lagrangian + r.dca_filter.atoms_inequalities <=
                s.dca_filter_max_atoms,
            "global atom storage cap exceeded");
    p.n = 4;
    p.name = "g3_reduced_atomic_border";
    p.x0 = {0.2, 0.3, 0.1, 0};
    p.lb = Vec(4, -1);
    p.ub = Vec(4, 1);
    p.equalities = {q(4, {0, 0, 0, 1}, {})};
    p.objective = q(4, {0, 0, 0, 0}, {{0, 0, -2}, {1, 1, -2}, {2, 2, 3}});
    s.dca_filter_atom_every = 1;
    r = solve_filter_dca(p, s);
    require(r.status == Status::Solved, r.message.c_str());
    require(r.objective < -1.99999 && r.dca_filter.spectral_inverse_solves > 0,
            "reduced atomic border did not solve nonconvex QP");
    require(r.lazy_newton.maximum_direction_residual <= 1e-7,
            "expanded atomic direction failed residual audit");
    std::cout << "PASS original filter, atomic/scalar certificates, cache, curved equalities and "
                 "nonconvex QP\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
