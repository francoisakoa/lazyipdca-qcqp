#pragma once
// v0.4.28r3p28 — shared sparse symmetric LDL' wrapper (SuiteSparse LDL with
// AMD or METIS fill-reducing ordering) used by the lazy-newton route and by the
// sparse single-loop kernel grafted into Lazy-DCA.  Internal header: it is not
// part of the public include/ contract.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#if defined(LAZYIPDCA_HAVE_LDL) && defined(LAZYIPDCA_HAVE_METIS)
extern "C" {
#include <ldl.h>
}
extern "C" {
#include <metis.h>
}
#define LAZYIPDCA_LN_HAVE_SPARSE 1
#endif
#if defined(LAZYIPDCA_HAVE_AMD)
#include <amd.h>
#endif

namespace lazyipdca::qcqp::sparse_ldl {

using Clock = std::chrono::steady_clock;
inline double seconds_since(const Clock::time_point& t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Sparse symmetric LDL' (SuiteSparse LDL + METIS nested dissection).
// ---------------------------------------------------------------------------
class SparseLdl {
 public:
  int n = 0;
  int npos = 0, nneg = 0, nzero = 0;
  int factor_nnz = 0;
  double t_metis = 0.0;
  std::string ordering = "metis";
  bool analyze(int dim, const std::vector<int>& ti, const std::vector<int>& tj,
               std::vector<int>& slot, bool use_metis, unsigned seed, std::string& msg) {
#ifndef LAZYIPDCA_LN_HAVE_SPARSE
    (void)dim; (void)ti; (void)tj; (void)slot; (void)use_metis; (void)seed;
    msg = "lazy-newton requires the SuiteSparse LDL + METIS backend";
    return false;
#else
    n = dim;
    // adjacency for ordering.  The fill-reducing ordering is computed once
    // on the first structure; rows appended later by the lazy working set
    // are ordered last (a sparse border), so a structure growth only costs a
    // symbolic re-analysis, never a new METIS call.
    perm.assign(n, 0); iperm.assign(n, 0);
    for (int i = 0; i < n; ++i) perm[i] = iperm[i] = i;
    if (!base_perm.empty() && static_cast<int>(base_perm.size()) <= n) {
      const int n0 = static_cast<int>(base_perm.size());
      for (int i = 0; i < n0; ++i) perm[i] = base_perm[i];
      for (int i = n0; i < n; ++i) perm[i] = i;
      for (int i = 0; i < n; ++i) iperm[perm[i]] = i;
#if defined(LAZYIPDCA_HAVE_AMD)
    } else if (ordering == "amd" && n > 1) {
#ifdef LAZYIPDCA_HAVE_AMD
      const auto tm = Clock::now();
      std::vector<std::vector<int>> adj(n);
      for (std::size_t t = 0; t < ti.size(); ++t) {
        if (ti[t] == tj[t]) continue;
        adj[ti[t]].push_back(tj[t]);
        adj[tj[t]].push_back(ti[t]);
      }
      std::vector<int> xadj(n + 1, 0), adjncy;
      for (int i = 0; i < n; ++i) {
        auto& a = adj[i];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        xadj[i + 1] = xadj[i] + static_cast<int>(a.size());
        adjncy.insert(adjncy.end(), a.begin(), a.end());
      }
      std::vector<int> P(n);
      double Control[AMD_CONTROL], Info[AMD_INFO];
      amd_defaults(Control);
      const int rc = amd_order(n, xadj.data(), adjncy.data(), P.data(), Control, Info);
      if (rc != AMD_OK && rc != AMD_OK_BUT_JUMBLED) { msg = "amd_order failed, code=" + std::to_string(rc); return false; }
      for (int i = 0; i < n; ++i) { perm[i] = P[i]; iperm[P[i]] = i; }
      base_perm = perm;
      t_metis = seconds_since(tm);
#else
      msg = "AMD ordering requested but not compiled";
      return false;
#endif
#endif
    } else if (use_metis && n > 1) {
      const auto tm = Clock::now();
#ifdef METIS_NOPTIONS
      using MetisIndex = idx_t;
#else
      using MetisIndex = idxtype;
#endif
      std::vector<std::vector<MetisIndex>> adj(n);
      for (std::size_t t = 0; t < ti.size(); ++t) {
        if (ti[t] == tj[t]) continue;
        adj[ti[t]].push_back(tj[t]);
        adj[tj[t]].push_back(ti[t]);
      }
      std::vector<MetisIndex> xadj(n + 1, 0), adjncy;
      for (int i = 0; i < n; ++i) {
        auto& a = adj[i];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        xadj[i + 1] = xadj[i] + static_cast<MetisIndex>(a.size());
        adjncy.insert(adjncy.end(), a.begin(), a.end());
      }
      std::vector<MetisIndex> p(n), ip(n);
#ifdef METIS_NOPTIONS
      std::vector<MetisIndex> opts(METIS_NOPTIONS);
      METIS_SetDefaultOptions(opts.data());
      opts[METIS_OPTION_NUMBERING] = 0;
      opts[METIS_OPTION_SEED] = static_cast<MetisIndex>(seed);
      MetisIndex nv = n;
      const int rc = METIS_NodeND(&nv, xadj.data(), adjncy.empty() ? nullptr : adjncy.data(),
                                  nullptr, opts.data(), p.data(), ip.data());
      if (rc != METIS_OK) { msg = "METIS_NodeND failed, code=" + std::to_string(rc); return false; }
#else
      (void)seed;
      int nv = n, numflag = 0, mopts[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      METIS_NodeND(&nv, xadj.data(), adjncy.empty() ? nullptr : adjncy.data(), &numflag, mopts,
                   p.data(), ip.data());
#endif
      for (int i = 0; i < n; ++i) { perm[i] = static_cast<int>(p[i]); iperm[i] = static_cast<int>(ip[i]); }
      base_perm = perm;
      t_metis = seconds_since(tm);
    }
    // unique permuted upper pattern
    struct E { int i, j; std::size_t t; };
    std::vector<E> P;
    P.reserve(ti.size());
    for (std::size_t t = 0; t < ti.size(); ++t) {
      int i = iperm[ti[t]], j = iperm[tj[t]];
      if (i > j) std::swap(i, j);
      P.push_back({i, j, t});
    }
    std::sort(P.begin(), P.end(), [](const E& a, const E& b) {
      return a.j != b.j ? a.j < b.j : a.i < b.i;
    });
    Ap.assign(n + 1, 0);
    Ai.clear(); Ai.reserve(P.size());
    slot.assign(ti.size(), -1);
    int last_i = -1, last_j = -1;
    for (const auto& e : P) {
      if (e.i != last_i || e.j != last_j) {
        Ai.push_back(e.i);
        ++Ap[e.j + 1];
        last_i = e.i; last_j = e.j;
      }
      slot[e.t] = static_cast<int>(Ai.size()) - 1;
    }
    for (int j = 0; j < n; ++j) Ap[j + 1] += Ap[j];
    Ax.assign(Ai.size(), 0.0);
    Lp.resize(n + 1); Parent.resize(n); Lnz.resize(n); Flag.resize(n);
    ldl_symbolic(n, Ap.data(), Ai.data(), Lp.data(), Parent.data(), Lnz.data(), Flag.data(),
                 nullptr, nullptr);
    if (Lp[n] < 0) { msg = "ldl_symbolic produced an invalid L size"; return false; }
    factor_nnz = Lp[n];
    Li.resize(Lp[n]); Lx.resize(Lp[n]);
    D.resize(n); Y.resize(n); Pattern.resize(n);
    work.resize(n);
    return true;
#endif
  }
  // tv is indexed by triplet; slot maps to Ax
  bool factor(const std::vector<double>& tv, const std::vector<int>& slot) {
#ifndef LAZYIPDCA_LN_HAVE_SPARSE
    (void)tv; (void)slot; return false;
#else
    std::fill(Ax.begin(), Ax.end(), 0.0);
    for (std::size_t t = 0; t < tv.size(); ++t) Ax[slot[t]] += tv[t];
    const int d = ldl_numeric(n, Ap.data(), Ai.data(), Ax.data(), Lp.data(), Parent.data(),
                              Lnz.data(), Li.data(), Lx.data(), D.data(), Y.data(),
                              Pattern.data(), Flag.data(), nullptr, nullptr);
    npos = nneg = nzero = 0;
    if (d != n) { nzero = n - d; return false; }
    double dmax = 0.0;
    for (double v : D) dmax = std::max(dmax, std::abs(v));
    const double tiny = 1e-30 * std::max(1.0, dmax);
    for (double v : D) {
      if (!std::isfinite(v) || std::abs(v) <= tiny) ++nzero;
      else if (v > 0.0) ++npos;
      else ++nneg;
    }
    return nzero == 0;
#endif
  }
  void solve(std::vector<double>& b) const {
#ifdef LAZYIPDCA_LN_HAVE_SPARSE
    auto& w = const_cast<std::vector<double>&>(work);
    for (int i = 0; i < n; ++i) w[i] = b[perm[i]];
    auto* self = const_cast<SparseLdl*>(this);
    ldl_lsolve(n, w.data(), self->Lp.data(), self->Li.data(), self->Lx.data());
    ldl_dsolve(n, w.data(), self->D.data());
    ldl_ltsolve(n, w.data(), self->Lp.data(), self->Li.data(), self->Lx.data());
    for (int i = 0; i < n; ++i) b[perm[i]] = w[i];
#else
    (void)b;
#endif
  }

  void reset_ordering() { base_perm.clear(); }
  std::vector<int> zero_pivot_rows() const {
    std::vector<int> out;
    double dmax = 0.0;
    for (double v : D) dmax = std::max(dmax, std::abs(v));
    const double tiny = 1e-30 * std::max(1.0, dmax);
    for (int k = 0; k < n; ++k) if (!std::isfinite(D[k]) || std::abs(D[k]) <= tiny) out.push_back(perm[k]);
    return out;
  }
 private:
  std::vector<int> perm, iperm, base_perm, Ap, Ai, Lp, Parent, Lnz, Flag, Li, Pattern;
  std::vector<double> Ax, Lx, D, Y, work;
};


} // namespace lazyipdca::qcqp::sparse_ldl
