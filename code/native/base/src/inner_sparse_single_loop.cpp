// v0.4.28r3p28 — sparse single-loop kernel (see the header for the contract).
#include "lazyipdca_qcqp/inner_sparse_single_loop.hpp"
#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "sparse_ldl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lazyipdca::qcqp {
namespace {

using sparse_ldl::Clock;
using sparse_ldl::seconds_since;
using sparse_ldl::SparseLdl;

inline bool finite_vec(const Vec& v) {
  for (double x : v) if (!std::isfinite(x)) return false;
  return true;
}
inline double max_abs(const Vec& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

// Sparse view of a ConvexQuadraticModel.  The prefix scalar shift and the
// low-rank terms are kept symbolically: the shift is a diagonal, the low-rank
// terms are bordered into the KKT system (one auxiliary unknown per term).
struct SModel {
  double c = 0.0;
  std::vector<int> lin_idx;
  std::vector<double> lin_val;
  std::vector<int> hi, hj;         // upper triplets, global indices
  std::vector<double> hv;
  double shift = 0.0;
  int shift_dim = 0;
  const std::vector<LowRankTerm>* low_rank = nullptr;
  std::vector<int> support;        // sorted union of linear and Hessian indices
  bool dense_gradient = false;     // gradient may leave the support (shift/low-rank)

  static SModel build(const ConvexQuadraticModel& m, int n) {
    SModel s;
    s.c = m.constant;
    for (int j = 0; j < n; ++j) if (m.linear[j] != 0.0) { s.lin_idx.push_back(j); s.lin_val.push_back(m.linear[j]); }
    std::unordered_map<std::int64_t, double> acc;
    for (const auto& e : m.base_hessian.entries) {
      if (e.v == 0.0) continue;
      const int a = std::min(e.i, e.j), b = std::max(e.i, e.j);
      acc[(static_cast<std::int64_t>(a) << 32) | b] += e.v;
    }
    std::vector<std::pair<std::int64_t, double>> ents(acc.begin(), acc.end());
    std::sort(ents.begin(), ents.end());
    for (const auto& e : ents) {
      s.hi.push_back(static_cast<int>(e.first >> 32));
      s.hj.push_back(static_cast<int>(e.first & 0xffffffff));
      s.hv.push_back(e.second);
    }
    s.shift = m.scalar_shift; s.shift_dim = std::min(n, std::max(0, m.shift_dimension));
    if (s.shift == 0.0) s.shift_dim = 0;
    s.low_rank = &m.low_rank;
    s.support = s.lin_idx;
    s.support.insert(s.support.end(), s.hi.begin(), s.hi.end());
    s.support.insert(s.support.end(), s.hj.begin(), s.hj.end());
    std::sort(s.support.begin(), s.support.end());
    s.support.erase(std::unique(s.support.begin(), s.support.end()), s.support.end());
    s.dense_gradient = s.shift_dim > 0 || !m.low_rank.empty();
    return s;
  }
  double value(const Vec& y) const {
    double v = c;
    for (std::size_t t = 0; t < lin_idx.size(); ++t) v += lin_val[t] * y[lin_idx[t]];
    double q = 0.0;
    for (std::size_t t = 0; t < hv.size(); ++t)
      q += (hi[t] == hj[t]) ? hv[t] * y[hi[t]] * y[hi[t]] : 2.0 * hv[t] * y[hi[t]] * y[hj[t]];
    v += 0.5 * q;
    for (int i = 0; i < shift_dim; ++i) v += 0.5 * shift * y[i] * y[i];
    for (const auto& t : *low_rank) { const double a = dot(t.direction, y); v += 0.5 * t.beta * a * a; }
    return v;
  }
  // Sparse gradient at y as (index, value) pairs; exact (dense terms included).
  void gradient(const Vec& y, std::vector<std::pair<int, double>>& out, Vec& work) const {
    out.clear();
    if (!dense_gradient) {
      for (int j : support) work[j] = 0.0;
      for (std::size_t t = 0; t < lin_idx.size(); ++t) work[lin_idx[t]] += lin_val[t];
      for (std::size_t t = 0; t < hv.size(); ++t) {
        work[hi[t]] += hv[t] * y[hj[t]];
        if (hi[t] != hj[t]) work[hj[t]] += hv[t] * y[hi[t]];
      }
      for (int j : support) if (work[j] != 0.0) out.emplace_back(j, work[j]);
      return;
    }
    const int n = static_cast<int>(y.size());
    std::fill(work.begin(), work.end(), 0.0);
    for (std::size_t t = 0; t < lin_idx.size(); ++t) work[lin_idx[t]] += lin_val[t];
    for (std::size_t t = 0; t < hv.size(); ++t) {
      work[hi[t]] += hv[t] * y[hj[t]];
      if (hi[t] != hj[t]) work[hj[t]] += hv[t] * y[hi[t]];
    }
    for (int i = 0; i < shift_dim; ++i) work[i] += shift * y[i];
    for (const auto& t : *low_rank) { const double a = t.beta * dot(t.direction, y); for (int j = 0; j < n; ++j) work[j] += a * t.direction[j]; }
    for (int j = 0; j < n; ++j) if (work[j] != 0.0) out.emplace_back(j, work[j]);
  }
};

struct State {
  Vec y, lambda, z, s;
  Vec rdual, req, phi, rpri;
  std::vector<std::vector<std::pair<int, double>>> grad;  // per inequality row
  double dual_inf = 0, eq_inf = 0, pri_inf = 0, ineq_inf = 0, mu = 0;
  double residual() const {
    double c = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) c = std::max(c, std::abs(s[i] * z[i]));
    return std::max({dual_inf, eq_inf, pri_inf, ineq_inf, c});
  }
};

struct Kernel {
  const ConvexSubproblem& p;
  const Settings& st;
  int n, me, mi;
  SModel obj;
  std::vector<SModel> ineq;
  std::vector<std::vector<std::pair<int, double>>> eq_rows;  // sparse equality rows
  Vec work;
  // KKT structure
  int nb = 0;                       // border unknowns
  std::vector<std::pair<int, int>> border_owner;  // (model index: -1 objective / row i, term index)
  std::vector<char> augmented;      // per inequality row
  std::vector<int> row_kkt;         // inequality row -> KKT row or -1 (condensed)
  int N = 0, naug = 0;
  std::vector<int> ti, tj, slot;
  std::vector<double> tv, reg;
  std::vector<int> diag_slot, eq_diag_slot, ineq_diag_slot, border_diag_slot;
  std::unordered_map<std::int64_t, int> offdiag;
  std::vector<int> obj_hslot;
  std::vector<std::vector<int>> ineq_hslot, ineq_cslot, ineq_jslot, border_slots;
  std::vector<std::vector<int>> eq_jslot, obj_border_slots;
  int expected_pos = 0, expected_neg = 0;

  Kernel(const ConvexSubproblem& pp, const Settings& ss) : p(pp), st(ss) {
    n = p.nvar; me = static_cast<int>(p.equalities.rows.size()); mi = static_cast<int>(p.inequalities.size());
    work.assign(n, 0.0);
    obj = SModel::build(p.objective, n);
    ineq.reserve(mi);
    for (const auto& g : p.inequalities) ineq.push_back(SModel::build(g, n));
    eq_rows.resize(me);
    for (int r = 0; r < me; ++r) {
      for (const auto& e : p.equalities.rows[r].entries) if (e.second != 0.0) eq_rows[r].push_back(e);
      std::sort(eq_rows[r].begin(), eq_rows[r].end());
    }
  }

  void evaluate(State& e) const {
    e.rdual.assign(n, 0.0);
    std::vector<std::pair<int, double>> g;
    Vec w(n, 0.0);
    obj.gradient(e.y, g, w);
    for (const auto& [j, v] : g) e.rdual[j] += v;
    e.req.assign(me, 0.0);
    for (int r = 0; r < me; ++r) {
      double v = -p.equalities.rhs[r];
      for (const auto& [j, a] : eq_rows[r]) { v += a * e.y[j]; e.rdual[j] += e.lambda[r] * a; }
      e.req[r] = v;
    }
    e.phi.assign(mi, 0.0); e.rpri.assign(mi, 0.0); e.grad.resize(mi);
    e.ineq_inf = 0.0;
    for (int i = 0; i < mi; ++i) {
      e.phi[i] = ineq[i].value(e.y);
      ineq[i].gradient(e.y, e.grad[i], w);
      for (const auto& [j, v] : e.grad[i]) e.rdual[j] += e.z[i] * v;
      e.rpri[i] = e.phi[i] + e.s[i];
      e.ineq_inf = std::max(e.ineq_inf, std::max(0.0, e.phi[i]));
    }
    e.dual_inf = max_abs(e.rdual); e.eq_inf = max_abs(e.req); e.pri_inf = max_abs(e.rpri);
    e.mu = mi ? dot(e.s, e.z) / mi : 0.0;
  }

  // ---------------------------------------------------------------- structure
  void build_structure(const State& e, SparseLdl& ldl, bool& reused, std::vector<int>& cached_ti, std::vector<int>& cached_tj, std::vector<int>& cached_slot) {
    ti.clear(); tj.clear(); offdiag.clear();
    auto add = [&](int i, int j) { if (i > j) std::swap(i, j); ti.push_back(i); tj.push_back(j); return static_cast<int>(ti.size()) - 1; };
    auto pair_slot = [&](int a, int b) {
      if (a == b) return diag_slot[a];
      const int lo = std::min(a, b), hi = std::max(a, b);
      const std::int64_t key = (static_cast<std::int64_t>(lo) << 32) | hi;
      auto it = offdiag.find(key);
      if (it == offdiag.end()) it = offdiag.emplace(key, add(lo, hi)).first;
      return it->second;
    };
    // border unknowns: objective low-rank terms then inequality low-rank terms
    border_owner.clear();
    for (std::size_t k = 0; k < p.objective.low_rank.size(); ++k) border_owner.emplace_back(-1, static_cast<int>(k));
    for (int i = 0; i < mi; ++i) for (std::size_t k = 0; k < p.inequalities[i].low_rank.size(); ++k) border_owner.emplace_back(i, static_cast<int>(k));
    nb = static_cast<int>(border_owner.size());
    augmented.assign(mi, 0); row_kkt.assign(mi, -1);
    naug = 0;
    for (int i = 0; i < mi; ++i) {
      const bool cond = !ineq[i].dense_gradient && static_cast<int>(e.grad[i].size()) <= st.ln_condense_max_support;
      augmented[i] = cond ? 0 : 1;
      if (!cond) row_kkt[i] = n + nb + me + naug++;
    }
    N = n + nb + me + naug;
    diag_slot.assign(n, -1);
    for (int j = 0; j < n; ++j) diag_slot[j] = add(j, j);
    border_diag_slot.assign(nb, -1);
    for (int b = 0; b < nb; ++b) border_diag_slot[b] = add(n + b, n + b);
    eq_diag_slot.assign(me, -1); ineq_diag_slot.assign(mi, -1);
    obj_hslot.assign(obj.hv.size(), -1);
    for (std::size_t t = 0; t < obj.hv.size(); ++t) obj_hslot[t] = pair_slot(obj.hi[t], obj.hj[t]);
    obj_border_slots.assign(p.objective.low_rank.size(), {});
    int b = 0;
    for (std::size_t k = 0; k < p.objective.low_rank.size(); ++k, ++b) {
      const Vec& v = p.objective.low_rank[k].direction;
      for (int j = 0; j < n; ++j) if (v[j] != 0.0) obj_border_slots[k].push_back(add(j, n + b));
    }
    ineq_hslot.assign(mi, {}); ineq_cslot.assign(mi, {}); ineq_jslot.assign(mi, {}); border_slots.assign(mi, {});
    for (int i = 0; i < mi; ++i) {
      const SModel& m = ineq[i];
      ineq_hslot[i].resize(m.hv.size());
      for (std::size_t t = 0; t < m.hv.size(); ++t) ineq_hslot[i][t] = pair_slot(m.hi[t], m.hj[t]);
      for (std::size_t k = 0; k < p.inequalities[i].low_rank.size(); ++k, ++b) {
        const Vec& v = p.inequalities[i].low_rank[k].direction;
        for (int j = 0; j < n; ++j) if (v[j] != 0.0) border_slots[i].push_back(add(j, n + b));
      }
      if (augmented[i]) {
        const int row = row_kkt[i];
        ineq_jslot[i].resize(e.grad[i].size());
        for (std::size_t t = 0; t < e.grad[i].size(); ++t) ineq_jslot[i][t] = add(e.grad[i][t].first, row);
        ineq_diag_slot[i] = add(row, row);
      } else {
        const std::size_t k = e.grad[i].size();
        ineq_cslot[i].resize(k * (k + 1) / 2);
        std::size_t c = 0;
        for (std::size_t a = 0; a < k; ++a)
          for (std::size_t bb = a; bb < k; ++bb, ++c) ineq_cslot[i][c] = pair_slot(e.grad[i][a].first, e.grad[i][bb].first);
      }
    }
    eq_jslot.assign(me, {});
    for (int r = 0; r < me; ++r) {
      eq_jslot[r].resize(eq_rows[r].size());
      for (std::size_t t = 0; t < eq_rows[r].size(); ++t) eq_jslot[r][t] = add(eq_rows[r][t].first, n + nb + r);
      eq_diag_slot[r] = add(n + nb + r, n + nb + r);
    }
    tv.assign(ti.size(), 0.0); reg.assign(N, 0.0);
    reused = (ti == cached_ti && tj == cached_tj && cached_slot.size() == ti.size());
    if (!reused) {
      std::string msg;
      ldl.ordering = st.ln_ordering;
      ldl.reset_ordering();
      if (!ldl.analyze(N, ti, tj, slot, st.linear_metis, st.random_seed, msg))
        throw std::runtime_error("sparse single-loop kernel: " + msg);
      cached_ti = ti; cached_tj = tj; cached_slot = slot;
    } else {
      slot = cached_slot;
    }
  }

  // Numeric assembly; expected inertia is recorded.
  void assemble(const State& e, double delta_x, double delta_c, bool least_squares = false) {
    std::fill(tv.begin(), tv.end(), 0.0);
    expected_pos = n; expected_neg = me + naug;
    if (least_squares) {
      // [I + sum_condensed g g', A'; A, -delta_c] with unit slack columns on
      // augmented rows: multiplier least-squares fit at the current point.
      // Reduced-space fit emulation: the stationarity of variables that the
      // exact affine eliminations would remove is enforced (W^-1-weighted fit: small diagonal weight),
      // matching LazyQP's reduced dual bootstrap in the full coordinates.
      for (int j = 0; j < n; ++j) tv[diag_slot[j]] = 1.0;
      for (const auto& el : p.exact_affine_eliminations)
        if (el.variable >= 0 && el.variable < n) tv[diag_slot[el.variable]] = 1e-6;
      for (int b = 0; b < nb; ++b) { tv[border_diag_slot[b]] = -1.0; ++expected_neg; }
      // inequality multipliers stay fixed in the fit: decouple their rows
      for (int i = 0; i < mi; ++i) {
        if (augmented[i]) {
          for (std::size_t t = 0; t < e.grad[i].size(); ++t) tv[ineq_jslot[i][t]] = 0.0;
          tv[ineq_diag_slot[i]] = -1.0;
        }
      }
      for (int r = 0; r < me; ++r) {
        for (std::size_t t = 0; t < eq_rows[r].size(); ++t) tv[eq_jslot[r][t]] = eq_rows[r][t].second;
        tv[eq_diag_slot[r]] = -std::max(delta_c, 1e-8);
        reg[n + nb + r] = 0.0;
      }
      return;
    }
    // diagonal: shift terms via prefix sums
    Vec dshift(n + 1, 0.0);
    if (obj.shift_dim > 0) { dshift[0] += obj.shift; dshift[obj.shift_dim] -= obj.shift; }
    for (int i = 0; i < mi; ++i) if (ineq[i].shift_dim > 0) { dshift[0] += e.z[i] * ineq[i].shift; dshift[ineq[i].shift_dim] -= e.z[i] * ineq[i].shift; }
    double run = 0.0;
    for (int j = 0; j < n; ++j) { run += dshift[j]; tv[diag_slot[j]] = delta_x + run; }
    for (std::size_t t = 0; t < obj.hv.size(); ++t) tv[obj_hslot[t]] += obj.hv[t];
    int b = 0;
    for (std::size_t k = 0; k < p.objective.low_rank.size(); ++k, ++b) {
      const double beta = p.objective.low_rank[k].beta;
      const Vec& v = p.objective.low_rank[k].direction;
      std::size_t c = 0;
      for (int j = 0; j < n; ++j) if (v[j] != 0.0) tv[obj_border_slots[k][c++]] = v[j];
      // border pivot: Schur complement -v (pivot)^-1 v' = beta v v'  =>  pivot = -1/beta
      const double piv = (beta != 0.0) ? -1.0 / beta : -1e-30;
      tv[border_diag_slot[b]] = piv;
      if (piv < 0.0) ++expected_neg; else ++expected_pos;
    }
    for (int i = 0; i < mi; ++i) {
      const SModel& m = ineq[i];
      const double zi = e.z[i];
      for (std::size_t t = 0; t < m.hv.size(); ++t) tv[ineq_hslot[i][t]] += zi * m.hv[t];
      std::size_t c = 0;
      for (std::size_t k = 0; k < p.inequalities[i].low_rank.size(); ++k, ++b) {
        const double beta = zi * p.inequalities[i].low_rank[k].beta;
        const Vec& v = p.inequalities[i].low_rank[k].direction;
        for (int j = 0; j < n; ++j) if (v[j] != 0.0) tv[border_slots[i][c++]] = v[j];
        const double piv = (beta != 0.0) ? -1.0 / beta : -1e-30;
        tv[border_diag_slot[b]] = piv;
        if (piv < 0.0) ++expected_neg; else ++expected_pos;
      }
      if (augmented[i]) {
        for (std::size_t t = 0; t < e.grad[i].size(); ++t) tv[ineq_jslot[i][t]] = e.grad[i][t].second;
        tv[ineq_diag_slot[i]] = -e.s[i] / zi;
      } else {
        const double w = zi / e.s[i];
        const std::size_t k = e.grad[i].size();
        std::size_t cc = 0;
        for (std::size_t a = 0; a < k; ++a)
          for (std::size_t bb = a; bb < k; ++bb, ++cc) tv[ineq_cslot[i][cc]] += w * e.grad[i][a].second * e.grad[i][bb].second;
      }
    }
    for (int r = 0; r < me; ++r) {
      for (std::size_t t = 0; t < eq_rows[r].size(); ++t) tv[eq_jslot[r][t]] = eq_rows[r][t].second;
      tv[eq_diag_slot[r]] = -delta_c;
      reg[n + nb + r] = -delta_c;
    }
  }

  void matvec(const std::vector<double>& v, std::vector<double>& out, bool unregularized) const {
    std::fill(out.begin(), out.end(), 0.0);
    for (std::size_t t = 0; t < ti.size(); ++t) {
      out[ti[t]] += tv[t] * v[tj[t]];
      if (ti[t] != tj[t]) out[tj[t]] += tv[t] * v[ti[t]];
    }
    if (unregularized) for (int k = 0; k < N; ++k) out[k] -= reg[k] * v[k];
  }

  double solve_refined(const SparseLdl& ldl, const std::vector<double>& b, std::vector<double>& d, int& sweeps) const {
    d = b; ldl.solve(d);
    const double bn = std::max(1e-300, max_abs(b));
    std::vector<double> res(N), corr(N), best = d;
    double best_berr = std::numeric_limits<double>::infinity();
    for (int sweep = 0; sweep <= st.ln_kkt_refine; ++sweep) {
      matvec(d, res, true);
      for (int k = 0; k < N; ++k) res[k] = b[k] - res[k];
      const double berr = max_abs(res) / bn;
      if (berr < best_berr) { best_berr = berr; best = d; }
      if (berr <= st.ln_kkt_berr_tol || sweep == st.ln_kkt_refine) break;
      corr = res; ldl.solve(corr);
      for (int k = 0; k < N; ++k) d[k] += corr[k];
      ++sweeps;
    }
    d = best;
    return best_berr;
  }

  // Right-hand side for a complementarity target t_i (per row): row rhs
  // r_i = -phi_i - t_i / z_i, condensed rows fold into the primal block.
  void rhs(const State& e, const Vec& target, std::vector<double>& b, Vec& cond_r) const {
    b.assign(N, 0.0);
    for (int j = 0; j < n; ++j) b[j] = -e.rdual[j];
    for (int r = 0; r < me; ++r) b[n + nb + r] = -e.req[r];
    cond_r.assign(mi, 0.0);
    for (int i = 0; i < mi; ++i) {
      const double ri = -e.phi[i] - target[i] / e.z[i];
      if (augmented[i]) b[row_kkt[i]] = ri;
      else {
        cond_r[i] = ri;
        const double w = e.z[i] / e.s[i];
        for (const auto& [j, g] : e.grad[i]) b[j] += w * ri * g;
      }
    }
  }
  void unpack(const State& e, const std::vector<double>& d, const Vec& cond_r, Vec& dy, Vec& dlam, Vec& dz, Vec& ds) const {
    dy.assign(n, 0.0); for (int j = 0; j < n; ++j) dy[j] = d[j];
    dlam.assign(me, 0.0); for (int r = 0; r < me; ++r) dlam[r] = d[n + nb + r];
    dz.assign(mi, 0.0); ds.assign(mi, 0.0);
    for (int i = 0; i < mi; ++i) {
      double gd = 0.0;
      for (const auto& [j, g] : e.grad[i]) gd += g * dy[j];
      dz[i] = augmented[i] ? d[row_kkt[i]] : (e.z[i] / e.s[i]) * (gd - cond_r[i]);
      ds[i] = -(e.rpri[i] + gd);
    }
  }
};

} // namespace

InnerResult SparseSingleLoopKernel::solve(const ConvexSubproblem& p) const {
  // The cold first majorant (no transported dual state) keeps LazyQP's
  // reduced dual bootstrap; the sparse kernel takes over from the first
  // transported state onwards.
  const bool cold_first = !(st_.inner_dual_warm_start && p.has_dual_warm_start);
  const bool single_loop_route = st_.dca_single_loop_primal_dual && p.allow_dca_single_loop &&
                                 p.prefer_feasible_centered_start && !p.inequalities.empty() && !cold_first;
  if (!single_loop_route) {
    // Outside the single-loop primary-majorant contract: delegate unchanged.
    ConvexQCQPLazyQPKernel qp(st_);
    return qp.solve(p);
  }
  InnerResult out;
  const auto t0 = Clock::now();
  const int n = p.nvar, me = static_cast<int>(p.equalities.rows.size()), mi = static_cast<int>(p.inequalities.size());
  if (n <= 0 || static_cast<int>(p.objective.linear.size()) != n || static_cast<int>(p.equalities.rhs.size()) != me) {
    out.message = "sparse single-loop kernel: invalid convex subproblem dimensions";
    return out;
  }
  if (st_.dca_single_loop_inner_steps != 1 || !(st_.dca_single_loop_forcing_eta > 0.0 && st_.dca_single_loop_forcing_eta < 1.0)) {
    out.message = "sparse single-loop kernel: invalid single-loop configuration";
    return out;
  }
  static thread_local SparseLdl ldl;
  static thread_local std::vector<int> cached_ti, cached_tj, cached_slot;
  try {
    Kernel K(p, st_);
    // ---- start state: primal warm start, centered cold duals/slacks, optional dual/slack transport
    State e;
    e.y = (p.has_primal_warm_start && static_cast<int>(p.initial_primal.size()) == n && finite_vec(p.initial_primal))
              ? p.initial_primal : Vec(n, 0.0);
    out.primal_warm_start_used = p.has_primal_warm_start && static_cast<int>(p.initial_primal.size()) == n;
    e.lambda.assign(me, 0.0); e.z.assign(mi, 1.0); e.s.assign(mi, 1.0);
    {
      std::vector<std::pair<int, double>> g; Vec w(n, 0.0);
      K.obj.gradient(e.y, g, w);
      double gmax = 0.0; for (const auto& [j, v] : g) gmax = std::max(gmax, std::abs(v));
      const double gscale = std::max(1e-2, std::min(1.0, gmax));
      const double target_mu = std::max({st_.lazyqp_barrier_mu0, gscale, 100.0 * st_.inner_tolerance});
      const double slack_push = std::max(1e-4, 10.0 * st_.inner_tolerance);
      for (int i = 0; i < mi; ++i) {
        const double phi = K.ineq[i].value(e.y);
        const double raw = std::max(0.0, -phi);
        e.s[i] = std::max(raw, slack_push);
        if (e.s[i] > raw) ++out.initialization_bound_pushes;
        e.z[i] = std::clamp(target_mu / std::max(e.s[i], target_mu), st_.minimum_positive, 10.0);
      }
    }
    K.evaluate(e);
    const double cold_residual = e.residual();
    const bool warm_dims = st_.inner_dual_warm_start && p.has_dual_warm_start &&
                           static_cast<int>(p.initial_equality_duals.size()) == me &&
                           static_cast<int>(p.initial_inequality_duals.size()) == mi &&
                           finite_vec(p.initial_equality_duals) && finite_vec(p.initial_inequality_duals);
    if (warm_dims) {
      State w = e;
      w.lambda = p.initial_equality_duals;
      const bool full = st_.inner_inequality_dual_warm_start;
      bool constructed = true;
      if (full) {
        const bool slack_warm = p.has_inequality_slack_warm_start &&
            static_cast<int>(p.initial_inequality_slacks.size()) == mi && finite_vec(p.initial_inequality_slacks) &&
            std::all_of(p.initial_inequality_slacks.begin(), p.initial_inequality_slacks.end(), [](double v) { return v > 0.0; });
        for (int i = 0; i < mi; ++i) {
          const double phi = K.ineq[i].value(w.y);
          const double raw = std::max(0.0, -phi);
          const double push = std::max(st_.minimum_positive,
              std::min(st_.qp_bound_push * std::max(1.0, std::abs(phi)),
                       st_.inner_bound_frac * std::max(st_.inner_bound_push_reference, raw)));
          w.s[i] = slack_warm ? std::max(st_.minimum_positive, p.initial_inequality_slacks[i]) : std::max(raw, push);
          const double zi = p.initial_inequality_duals[i];
          if (std::isfinite(zi) && zi > st_.minimum_positive) {
            w.z[i] = zi;
          } else if (st_.lazyqp_zero_dual_continuation && std::isfinite(zi) && zi >= 0.0 && zi <= st_.minimum_positive) {
            // zero-dual continuation (LazyQP r3p23 policy): keep the entry
            // strictly positive with componentwise complementarity <= mu_cap.
            ++out.lazyqp_zero_dual_continuation_repairs;
            const double candidate = std::clamp(st_.lazyqp_zero_dual_mu_cap / std::max(w.s[i], st_.lazyqp_zero_dual_mu_cap),
                                                st_.minimum_positive, 10.0);
            if (std::isfinite(candidate) && candidate > 0.0) w.z[i] = candidate; else { constructed = false; w.z[i] = e.z[i]; }
          } else {
            if (st_.lazyqp_zero_dual_continuation && std::isfinite(zi) && zi < 0.0) constructed = false;
            w.z[i] = e.z[i];
          }
        }
        if (st_.lazyqp_zero_dual_continuation) ++out.lazyqp_zero_dual_continuation_attempts;
      }
      K.evaluate(w);
      // guard: finite and not catastrophically worse than the centered start
      const double wr = w.residual();
      const double guard = std::max(1.0, st_.lazyqp_warm_start_guard_factor) * std::max(cold_residual, st_.inner_tolerance);
      if (constructed && std::isfinite(wr) && wr <= 1e3 * guard) {
        if (st_.lazyqp_zero_dual_continuation && full) ++out.lazyqp_zero_dual_continuation_acceptances;
        e = std::move(w);
        out.dual_warm_start_used = true; out.full_dual_warm_start_used = full;
      } else out.dual_warm_start_rejected = true;
    }
    // ---- reduced dual bootstrap (first majorant only): least-squares
    // equality multipliers from the identity-block system, accepted only
    // when they lower the dual residual of the centered start.
    bool reused0 = false;
    K.build_structure(e, ldl, reused0, cached_ti, cached_tj, cached_slot);
    if (reused0) ++out.symbolic_reuses; else ++out.symbolic_analyses;
    double bootstrap_reference = -1.0;
    if (p.allow_dca_single_loop_dual_bootstrap) {
      out.dca_single_loop_reduced_bootstrap_attempted = true;
      K.assemble(e, 0.0, st_.ln_delta_c, true);
      ++out.factorizations;
      const bool ls_ok = ldl.factor(K.tv, K.slot);
      if (ls_ok && ldl.npos == K.expected_pos && ldl.nneg == K.expected_neg) {
        std::vector<double> bb(K.N, 0.0), dd;
        for (int j = 0; j < n; ++j) bb[j] = -e.rdual[j];
        dd = bb; ldl.solve(dd);
        State boot = e;
        for (int r = 0; r < me; ++r) boot.lambda[r] += dd[n + K.nb + r];
        if (finite_vec(boot.lambda)) {
          K.evaluate(boot);
          if (std::isfinite(boot.dual_inf) && boot.dual_inf < e.dual_inf) {
            e = std::move(boot);
            out.dca_single_loop_reduced_bootstrap_accepted = true;
            bootstrap_reference = cold_residual;  // contract against the centered start
          }
        }
      }
    }
    // ---- transaction start snapshot
    const State start = e;
    const double state_start_residual = e.residual();
    const double start_residual = bootstrap_reference > 0.0 ? std::max(state_start_residual, bootstrap_reference)
                                                            : state_start_residual;
    const double start_primal = std::max({e.eq_inf, e.pri_inf, e.ineq_inf});
    const double start_model = p.objective.value(e.y);
    out.dca_single_loop_attempted = true;
    out.dca_single_loop_initial_residual = start_residual;
    out.dca_single_loop_initial_primal = e.y;
    out.dca_single_loop_initial_equality_duals = e.lambda;
    out.dca_single_loop_initial_inequality_duals = e.z;
    out.dca_single_loop_initial_inequality_slacks = e.s;
    auto rollback = [&](const std::string& reason) {
      out.dca_single_loop_retryable = true;
      out.dca_single_loop_accepted = false;
      out.success = false;
      out.dca_single_loop_final_residual = e.residual();
      out.dca_single_loop_steps = 0;
      out.message = "sparse single-loop step rejected: " + reason;
      e = start;
      out.y = e.y; out.equality_duals = e.lambda; out.inequality_duals = e.z; out.inequality_slacks = e.s;
      out.dual_inf = e.dual_inf; out.equality_inf = e.eq_inf; out.inequality_inf = e.ineq_inf;
      out.primal_residual_inf = e.pri_inf; out.complementarity = e.mu;
      return out;
    };
    // ---- factorization with inertia control (structure built above)
    out.lazyqp_kkt_dimension = K.N; out.lazyqp_full_kkt_dimension = K.N;
    double delta = 0.0;
    bool factored = false;
    for (int attempt = 0; attempt < 40; ++attempt) {
      K.assemble(e, delta, st_.ln_delta_c);
      ++out.factorizations;
      const bool ok = ldl.factor(K.tv, K.slot);
      if (ok && ldl.npos == K.expected_pos && ldl.nneg == K.expected_neg) { factored = true; break; }
      delta = (delta == 0.0) ? st_.ln_delta_x_init : 8.0 * delta;
      ++out.kkt_regularization_retries;
      if (delta > st_.ln_delta_x_max) break;
    }
    out.final_kkt_rho_x = delta; out.max_kkt_rho_x = delta;
    out.final_kkt_delta_d = st_.ln_delta_c; out.max_kkt_delta_d = st_.ln_delta_c;
    out.dca_single_loop_factorizations = out.factorizations;
    if (!factored) return rollback("inertia correction failed");
    // ---- Mehrotra predictor-corrector (affine predictor, centering corrector)
    std::vector<double> b, d;
    Vec cond_r, target(mi, 0.0), dy, dlam, dz, ds;
    int sweeps = 0;
    K.rhs(e, target, b, cond_r);
    double berr = K.solve_refined(ldl, b, d, sweeps);
    K.unpack(e, d, cond_r, dy, dlam, dz, ds);
    double alpha_aff_p = 1.0, alpha_aff_d = 1.0;
    for (int i = 0; i < mi; ++i) {
      if (ds[i] < 0.0) alpha_aff_p = std::min(alpha_aff_p, -e.s[i] / ds[i]);
      if (dz[i] < 0.0) alpha_aff_d = std::min(alpha_aff_d, -e.z[i] / dz[i]);
    }
    double mu_aff = 0.0;
    for (int i = 0; i < mi; ++i) mu_aff += (e.s[i] + alpha_aff_p * ds[i]) * (e.z[i] + alpha_aff_d * dz[i]);
    mu_aff = mi ? mu_aff / mi : 0.0;
    double sigma = 0.0;
    if (mi && e.mu > 0.0) {
      const double ratio = std::clamp(mu_aff / e.mu, 0.0, 1.0);
      sigma = std::min({ratio * ratio * ratio, std::clamp(st_.lazyqp_barrier_r, 0.0, 0.95), st_.qp_sigma_max});
    }
    // LazyQP single-loop convention: use the second-order product at the
    // admissible affine endpoint, not the unscaled (possibly huge) one.
    const double affine_scale = alpha_aff_p * alpha_aff_d;
    for (int i = 0; i < mi; ++i) target[i] = sigma * e.mu - affine_scale * ds[i] * dz[i];
    K.rhs(e, target, b, cond_r);
    berr = std::max(berr, K.solve_refined(ldl, b, d, sweeps));
    K.unpack(e, d, cond_r, dy, dlam, dz, ds);
    out.kkt_refinement_steps = sweeps;
    out.max_kkt_backward_error = berr;
    if (!finite_vec(dy) || !finite_vec(dlam) || !finite_vec(dz) || !finite_vec(ds)) return rollback("nonfinite direction");
    // Normal/tangential split (LazyQP adaptive normal step): the equality
    // feasibility correction is taken in full when it keeps every slack
    // strictly positive; only the tangential part is scaled by alpha_p.
    Vec dy_n(n, 0.0), ds_n(mi, 0.0), dy_t = dy, ds_t = ds;
    bool normal_split = false;
    {
      std::vector<double> bn(K.N, 0.0), dn;
      double req_inf = 0.0;
      for (int r = 0; r < me; ++r) { bn[n + K.nb + r] = -e.req[r]; req_inf = std::max(req_inf, std::abs(e.req[r])); }
      // full linearized feasibility right-hand side, zero dual/centering part
      for (int i = 0; i < mi; ++i) {
        const double ri = -e.rpri[i];
        if (K.augmented[i]) bn[K.row_kkt[i]] = ri;
        else { const double w = e.z[i] / e.s[i]; for (const auto& [j, g] : e.grad[i]) bn[j] += w * ri * g; }
      }
      if (req_inf > st_.inner_tolerance || e.pri_inf > st_.inner_tolerance) {
        int sw = 0;
        K.solve_refined(ldl, bn, dn, sw);
        bool interior_n = finite_vec(dn);
        if (interior_n) {
          for (int j = 0; j < n; ++j) dy_n[j] = dn[j];
          for (int i = 0; i < mi; ++i) {
            double gd = 0.0;
            for (const auto& [j, g] : e.grad[i]) gd += g * dy_n[j];
            ds_n[i] = -(e.rpri[i] + gd);
            if (!(e.s[i] + ds_n[i] > 0.0)) { interior_n = false; break; }
          }
        }
        if (interior_n) {
          normal_split = true;
          for (int j = 0; j < n; ++j) dy_t[j] = dy[j] - dy_n[j];
          for (int i = 0; i < mi; ++i) ds_t[i] = ds[i] - ds_n[i];
        } else { std::fill(dy_n.begin(), dy_n.end(), 0.0); std::fill(ds_n.begin(), ds_n.end(), 0.0); }
      }
    }
    // ---- common primal-dual step with nonlinear residual globalization
    // Split primal/dual fraction-to-boundary maxima scaled by one common
    // globalization factor tau (LazyQP split-step convention).
    const double tau_ftb = std::clamp(st_.fraction_to_boundary, 0.0, 0.999999999999);
    double alpha_p_max = 1.0, alpha_d_max = 1.0;
    for (int i = 0; i < mi; ++i) {
      const double s_base = e.s[i] + ds_n[i];
      if (ds_t[i] < 0.0) alpha_p_max = std::min(alpha_p_max, -tau_ftb * s_base / ds_t[i]);
      if (dz[i] < 0.0) alpha_d_max = std::min(alpha_d_max, -tau_ftb * e.z[i] / dz[i]);
    }
    const double alpha_max = std::min(alpha_p_max, alpha_d_max);
    const double primal_now = std::max(e.eq_inf, e.pri_inf);
    const double primal_limit = std::max({100.0 * st_.inner_tolerance, 10.0 * std::max(primal_now, st_.inner_tolerance), 0.10 * e.mu});
    const double dual_limit = std::max({100.0 * st_.inner_tolerance, 10.0 * std::max(e.dual_inf, st_.inner_tolerance), 0.10 * e.mu});
    const double residual_now = std::max({e.dual_inf, e.eq_inf, e.pri_inf});
    const bool centrality_phase = residual_now <= std::max(1e-5, 100.0 * st_.inner_tolerance);
    const double mu_limit = centrality_phase ? std::max(st_.inner_iteration_limit_mu_fallback, 0.95 * e.mu)
                                             : std::max(100.0 * st_.inner_tolerance, 10.0 * std::max(e.mu, st_.inner_tolerance));
    const double beta = std::clamp(st_.lazyqp_line_search_beta, 0.05, 0.95);
    double tau = 1.0, alpha = alpha_max, alpha_p = alpha_p_max, alpha_d = alpha_d_max, alpha_lambda = 0.0;
    State trial;
    bool accepted = false;
    for (int bt = 0; bt < 12; ++bt) {
      alpha_p = tau * alpha_p_max; alpha_d = tau * alpha_d_max; alpha = std::min(alpha_p, alpha_d);
      trial.y = e.y; trial.lambda = e.lambda; trial.z = e.z; trial.s = e.s;
      for (int j = 0; j < n; ++j) trial.y[j] += dy_n[j] + alpha_p * dy_t[j];
      // equality multipliers are sign-free: their step is chosen after the
      // trial by a one-dimensional residual projection (LazyQP convention)
      bool trial_valid = true;
      for (int i = 0; i < mi && trial_valid; ++i) {
        trial.z[i] = e.z[i] + alpha_d * dz[i];
        if (!(trial.z[i] > 0.0) || !std::isfinite(trial.z[i])) { trial_valid = false; break; }
        trial.z[i] = std::max(st_.minimum_positive, trial.z[i]);
        double gd = 0.0;
        for (const auto& [j, g] : e.grad[i]) gd += g * (dy_n[j] + alpha_p * dy_t[j]);
        double s_lin = e.s[i] + ds_n[i] + alpha_p * ds_t[i];
        if (!(s_lin > 0.0) || !std::isfinite(s_lin)) { trial_valid = false; break; }
        s_lin = std::max(st_.minimum_positive, s_lin);
        // Exact second-order slack correction (LazyQP make_trial): absorb the
        // quadratic remainder q while s stays safely inside s>0, otherwise
        // damp only the correction, never the primal Newton step.
        const double phi_t = K.ineq[i].value(trial.y);
        const double q = phi_t - (e.phi[i] + gd);
        if (!std::isfinite(q)) { trial_valid = false; break; }
        double gamma = 1.0;
        if (q > 0.0) {
          const double floor = std::max(st_.minimum_positive, (1.0 - tau_ftb) * s_lin);
          if (s_lin - q <= floor) gamma = std::clamp((s_lin > floor) ? (s_lin - floor) / q : 0.0, 0.0, 1.0);
        }
        trial.s[i] = std::max(st_.minimum_positive, s_lin - gamma * q);
        if (!(trial.s[i] > 0.0) || !std::isfinite(trial.s[i])) { trial_valid = false; break; }
      }
      if (!trial_valid) { tau *= beta; if (tau < 1e-14) break; continue; }
      K.evaluate(trial);
      {
        Vec gdir(n, 0.0);
        for (int r = 0; r < me; ++r) for (const auto& [j, a] : K.eq_rows[r]) gdir[j] += a * dlam[r];
        double gg = 0.0; for (double v : gdir) gg += v * v;
        if (gg > 0.0) {
          auto value = [&](double al) { double m = 0.0; for (int j = 0; j < n; ++j) m = std::max(m, std::abs(trial.rdual[j] + al * gdir[j])); return m; };
          double lo = 0.0, hi = 1.0;
          for (int it = 0; it < 64; ++it) { const double l = (2.0 * lo + hi) / 3.0, rr = (lo + 2.0 * hi) / 3.0; if (value(l) <= value(rr)) hi = rr; else lo = l; }
          double best = 0.0, bestv = value(0.0);
          for (double cand : {lo, 0.5 * (lo + hi), hi, 1.0}) { const double v = value(cand); if (v < bestv) { bestv = v; best = cand; } }
          if (best > 0.0) {
            for (int r = 0; r < me; ++r) trial.lambda[r] += best * dlam[r];
            for (int j = 0; j < n; ++j) trial.rdual[j] += best * gdir[j];
            trial.dual_inf = max_abs(trial.rdual);
          }
          alpha_lambda = best;
        }
      }
      const bool finite = finite_vec(trial.y) && finite_vec(trial.lambda) && std::isfinite(trial.residual());
      const bool ok = finite && std::max(trial.eq_inf, trial.pri_inf) <= primal_limit &&
                      trial.dual_inf <= dual_limit && trial.mu <= mu_limit;
      if (ok) { accepted = true; break; }
      tau *= beta;
      if (tau < 1e-14) break;
    }
    (void)alpha_lambda;
    if (!accepted) return rollback("nonlinear residual globalization failed");
    if (std::max(alpha_p / std::max(alpha_d, 1e-300), alpha_d / std::max(alpha_p, 1e-300)) > 10.0) ++out.lazyqp_scaled_split_steps;
    out.lazyqp_max_primal_dual_step_ratio = alpha_p / std::max(alpha_d, 1e-300);
    out.lazyqp_max_dual_primal_step_ratio = alpha_d / std::max(alpha_p, 1e-300);
    e = std::move(trial);
    // Complementarity floor for inactive multipliers: an affine-dominated
    // step (sigma -> 0) drives inactive z to the numerical floor; the outer
    // transport would then repair them at mu_cap, destroying the centering
    // of the next majorant.  Keep every product s*z >= mu/kappa instead.
    if (st_.ln_transport_complementarity_floor > 0.0 && e.mu > 0.0) {
      bool floored = false;
      for (int i = 0; i < mi; ++i) {
        if (e.z[i] > 10.0 * st_.minimum_positive) continue;  // only entries at the numerical floor
        const double zmin = e.mu / (st_.ln_transport_complementarity_floor * e.s[i]);
        if (e.z[i] < zmin) { e.z[i] = zmin; floored = true; }
      }
      if (floored) K.evaluate(e);
    }
    out.iterations = 1;
    out.dca_single_loop_steps = 1;
    out.dca_single_loop_factorizations = out.factorizations;
    const double final_residual = e.residual();
    out.dca_single_loop_final_residual = final_residual;
    out.lazyqp_min_globalization_tau = tau;
    // ---- contract
    const double final_primal = std::max({e.eq_inf, e.pri_inf, e.ineq_inf});
    const double final_model = p.objective.value(e.y);
    const double model_noise = 128.0 * std::numeric_limits<double>::epsilon() * std::max({1.0, std::abs(start_model), std::abs(final_model)});
    const double residual_factor = 1.0 - (1.0 - st_.dca_single_loop_forcing_eta) * std::clamp(alpha, 0.0, 1.0);
    const double residual_target = residual_factor * std::max(start_residual, std::numeric_limits<double>::min());
    const bool interior = std::all_of(e.s.begin(), e.s.end(), [](double v) { return v > 0.0; }) &&
                          std::all_of(e.z.begin(), e.z.end(), [](double v) { return v > 0.0; });
    const bool outer_filter_progress = p.allow_dca_single_loop_outer_filter_progress && !st_.lazyqp_inexact_forcing;
    const bool contracted = outer_filter_progress || final_residual <= residual_target;
    const bool primal_ok = final_primal <= primal_limit;
    const bool model_ok = final_model < start_model - model_noise;
    const bool feasibility_ok = final_primal < (1.0 - st_.dca_feasibility_step_fraction) * std::max(start_primal, st_.inner_tolerance);
    const double model_tol = std::max(st_.inner_tolerance, st_.qp_model_solved_tol);
    const double strict_residual = std::max({e.dual_inf, e.eq_inf, e.pri_inf});
    const bool strict = strict_residual <= model_tol && e.mu <= model_tol;
    const bool guarded = strict_residual <= 100.0 * st_.inner_tolerance && e.mu <= st_.inner_iteration_limit_mu_fallback;
    out.y = e.y; out.equality_duals = e.lambda; out.inequality_duals = e.z; out.inequality_slacks = e.s;
    out.dual_inf = e.dual_inf; out.equality_inf = e.eq_inf; out.inequality_inf = e.ineq_inf;
    out.primal_residual_inf = e.pri_inf; out.complementarity = e.mu;
    if (interior && (strict || guarded)) {
      out.success = true; out.dca_single_loop_accepted = false;
      out.message = guarded && !strict ? "sparse single-loop step reached the inexact nonlinear-residual contract"
                                       : "sparse single-loop step solved the convex QCQP strictly";
    } else if (interior && contracted && primal_ok && (model_ok || (outer_filter_progress && feasibility_ok))) {
      out.success = false; out.dca_single_loop_accepted = true;
      out.message = "guarded provisional sparse single-loop primal-dual state";
    } else {
      std::string why = !interior ? "lost interiority" : !contracted ? "residual not contracted" : !primal_ok ? "primal residual above limit" : "model not decreased";
      return rollback(why);
    }
    if (st_.verbose)
      std::printf("[sparse-single-loop] N=%d aug=%d nb=%d alpha_p/d/lambda=%.3e/%.3e/%.3e normal=%d eq %.1e->%.1e |dy|=%.1e sigma=%.2e mu %.2e->%.2e residual %.2e -> %.2e delta_x=%.1e berr=%.1e %s (%.2f ms)\n",
                  K.N, K.naug, K.nb, alpha_p, alpha_d, alpha_lambda, (int)normal_split, start.eq_inf, e.eq_inf, max_abs(dy), sigma, start.mu, e.mu, start_residual, final_residual, delta, berr,
                  out.success ? "solved" : "provisional", 1e3 * seconds_since(t0));
    return out;
  } catch (const std::exception& ex) {
    InnerResult failed;
    failed.message = std::string("sparse single-loop kernel: ") + ex.what();
    return failed;
  }
}

} // namespace lazyipdca::qcqp
