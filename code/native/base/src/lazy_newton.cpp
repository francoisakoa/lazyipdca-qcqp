// v0.4.28r3p28 — Lazy primal-dual Newton route (LazyPDN).
// See lazy_newton.hpp for the algorithmic contract.
#include "lazyipdca_qcqp/lazy_newton.hpp"
#include "lazyipdca_qcqp/qcqp_ipopt_termination.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparse_ldl.hpp"

extern "C" {
void dsyev_(const char* jobz, const char* uplo, const int* n, double* a, const int* lda,
            double* w, double* work, const int* lwork, int* info);
}

namespace lazyipdca::qcqp {
namespace {

using sparse_ldl::Clock;
using sparse_ldl::seconds_since;
using sparse_ldl::SparseLdl;
inline double inf_norm_ptr(const double* v, std::size_t n) {
  double m = 0.0;
  for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::abs(v[i]));
  return m;
}

// ---------------------------------------------------------------------------
// Sparse quadratic on its support, local indexing.
// ---------------------------------------------------------------------------
struct SQuad {
  double c = 0.0;
  std::vector<int> sup;          // sorted global variable indices
  std::vector<double> lin;       // linear coefficients (local)
  std::vector<int> hi, hj;       // Hessian upper triplets (local, hi<=hj)
  std::vector<double> hv;
  std::vector<int> pi, pj;       // PSD part (local upper triplets)
  std::vector<double> pv;
  bool psd = true;               // true if the PSD part equals the Hessian
  bool convex_zero = false;      // Hessian identically zero
  // KKT slots (filled at structure build): hessian slot per (hi,hj) entry,
  // PSD slot per (pi,pj) entry, Jacobian slot per support entry.
  std::vector<int> hslot, pslot, jslot;
  std::vector<int> cslot;   // condensed rank-one pattern: pair (a<=b) -> slot, -1 if a fixed var
  // scratch
  mutable std::vector<double> xl, gl;

  int k() const { return static_cast<int>(sup.size()); }

  void gather(const Vec& x) const {
    xl.resize(sup.size());
    for (std::size_t t = 0; t < sup.size(); ++t) xl[t] = x[sup[t]];
  }
  // value and local gradient at x (uses gather)
  double eval(const Vec& x) const {
    gather(x);
    gl.assign(sup.size(), 0.0);
    double q = 0.0;
    for (std::size_t t = 0; t < hv.size(); ++t) {
      const int a = hi[t], b = hj[t];
      const double v = hv[t];
      if (a == b) {
        q += v * xl[a] * xl[a];
        gl[a] += v * xl[a];
      } else {
        q += 2.0 * v * xl[a] * xl[b];
        gl[a] += v * xl[b];
        gl[b] += v * xl[a];
      }
    }
    double val = c + 0.5 * q;
    for (std::size_t t = 0; t < sup.size(); ++t) {
      val += lin[t] * xl[t];
      gl[t] += lin[t];
    }
    return val;
  }
  double value_only(const Vec& x) const {
    gather(x);
    double q = 0.0;
    for (std::size_t t = 0; t < hv.size(); ++t) {
      const int a = hi[t], b = hj[t];
      q += (a == b) ? hv[t] * xl[a] * xl[a] : 2.0 * hv[t] * xl[a] * xl[b];
    }
    double val = c + 0.5 * q;
    for (std::size_t t = 0; t < sup.size(); ++t) val += lin[t] * xl[t];
    return val;
  }
  // quadratic form 1/2 d'Qd for a global direction d
  double half_quad(const Vec& d) const {
    double q = 0.0;
    for (std::size_t t = 0; t < hv.size(); ++t) {
      const int a = sup[hi[t]], b = sup[hj[t]];
      q += (hi[t] == hj[t]) ? hv[t] * d[a] * d[a] : 2.0 * hv[t] * d[a] * d[b];
    }
    return 0.5 * q;
  }
  // g'd with the local gradient gl already evaluated
  double grad_dot(const Vec& d) const {
    double s = 0.0;
    for (std::size_t t = 0; t < sup.size(); ++t) s += gl[t] * d[sup[t]];
    return s;
  }
  double grad_inf() const { return inf_norm_ptr(gl.data(), gl.size()); }
};

void dc_split(SQuad& q) {
  const int k = q.k();
  q.convex_zero = q.hv.empty();
  if (q.hv.empty()) { q.psd = true; return; }
  // Exact sparse certificates first: a diagonal objective needs no dense
  // eigensolve, and symmetric diagonally dominant matrices are PSD.
  bool diagonal=true;
  Vec diag(k,0.0), radius(k,0.0);
  for(std::size_t t=0;t<q.hv.size();++t) {
    int a=q.hi[t],b=q.hj[t];double v=q.hv[t];
    if(a==b)diag[a]+=v;
    else {if(v!=0.0)diagonal=false;radius[a]+=std::abs(v);radius[b]+=std::abs(v);}
  }
  bool dominant=true;for(int a=0;a<k;++a)dominant=dominant&&diag[a]>=radius[a];
  if(dominant) {q.psd=true;q.pi=q.hi;q.pj=q.hj;q.pv=q.hv;return;}
  if(diagonal) {
    double amax=0;for(double v:diag)amax=std::max(amax,std::abs(v));
    const double tol=1e-12*std::max(1.0,amax);
    q.psd=*std::min_element(diag.begin(),diag.end())>=-tol;
    if(q.psd){q.pi=q.hi;q.pj=q.hj;q.pv=q.hv;return;}
    for(int a=0;a<k;++a)if(diag[a]>tol){q.pi.push_back(a);q.pj.push_back(a);q.pv.push_back(diag[a]);}
    return;
  }
  std::vector<double> A(static_cast<std::size_t>(k) * k, 0.0);
  for (std::size_t t = 0; t < q.hv.size(); ++t) {
    A[q.hi[t] + k * q.hj[t]] = q.hv[t];
    A[q.hj[t] + k * q.hi[t]] = q.hv[t];
  }
  std::vector<double> w(k), work(std::max(1, 8 * k));
  int info = 0, lwork = static_cast<int>(work.size());
  std::vector<double> V = A;
  dsyev_("V", "U", &k, V.data(), &k, w.data(), work.data(), &lwork, &info);
  if (info != 0) throw std::runtime_error("lazy-newton: dsyev failed in DC split");
  double amax = 0.0;
  for (double e : w) amax = std::max(amax, std::abs(e));
  const double tol = 1e-12 * std::max(1.0, amax);
  if (w.front() >= -tol) {
    q.psd = true;
    q.pi = q.hi; q.pj = q.hj; q.pv = q.hv;
    return;
  }
  q.psd = false;
  if (w.back() <= tol) return; // concave: PSD part is zero
  std::vector<double> P(static_cast<std::size_t>(k) * k, 0.0);
  for (int e = 0; e < k; ++e) {
    if (w[e] <= tol) continue;
    for (int a = 0; a < k; ++a)
      for (int b = 0; b < k; ++b) P[a + k * b] += w[e] * V[a + k * e] * V[b + k * e];
  }
  for (int a = 0; a < k; ++a)
    for (int b = a; b < k; ++b) {
      const double v = 0.5 * (P[a + k * b] + P[b + k * a]);
      if (std::abs(v) > 1e-15 * std::max(1.0, amax)) {
        q.pi.push_back(a); q.pj.push_back(b); q.pv.push_back(v);
      }
    }
}

SQuad make_squad(const QuadraticFunction& f, int n, bool want_split) {
  SQuad q;
  q.c = f.constant;
  std::vector<int> sup;
  f.for_each_linear_nonzero([&](int j,double){sup.push_back(j);});
  for (const auto& e : f.hessian.entries) if (e.v != 0.0) { sup.push_back(e.i); sup.push_back(e.j); }
  std::sort(sup.begin(), sup.end());
  sup.erase(std::unique(sup.begin(), sup.end()), sup.end());
  q.sup = sup;
  static thread_local std::vector<int> loc;
  if (static_cast<int>(loc.size()) < n) loc.assign(n, -1);
  for (std::size_t t = 0; t < sup.size(); ++t) loc[sup[t]] = static_cast<int>(t);
  q.lin.resize(sup.size());
  f.for_each_linear_nonzero([&](int j,double v){q.lin[loc[j]]=v;});
  // merge duplicate hessian triplets (sorted, local indices)
  std::vector<std::pair<std::int64_t, double>> ents;
  ents.reserve(f.hessian.entries.size());
  for (const auto& e : f.hessian.entries) {
    if (e.v == 0.0) continue;
    int a = loc[e.i], b = loc[e.j];
    if (a > b) std::swap(a, b);
    ents.emplace_back((static_cast<std::int64_t>(a) << 32) | b, e.v);
  }
  std::sort(ents.begin(), ents.end());
  for (std::size_t t = 0; t < ents.size(); ++t) {
    if (!q.hv.empty() && ents[t].first == ((static_cast<std::int64_t>(q.hi.back()) << 32) | q.hj.back())) {
      q.hv.back() += ents[t].second;
      continue;
    }
    q.hi.push_back(static_cast<int>(ents[t].first >> 32));
    q.hj.push_back(static_cast<int>(ents[t].first & 0xffffffff));
    q.hv.push_back(ents[t].second);
  }
  for (int v : sup) loc[v] = -1;
  if (want_split) dc_split(q);
  else { q.psd = true; q.convex_zero = q.hv.empty(); }
  return q;
}

// ---------------------------------------------------------------------------
// Solver state
// ---------------------------------------------------------------------------
struct LazyNewton {
  const QCQPProblem& p;
  const Settings& st;
  Result& r;
  Clock::time_point t0;

  int n = 0, nf = 0, me = 0, mi = 0;
  std::vector<int> free_of;   // variable -> free index or -1
  std::vector<int> var_of;    // free index -> variable
  std::vector<char> has_lb, has_ub;
  SQuad obj;
  std::vector<SQuad> eq, in;
  std::vector<double> eq_scale, in_scale;  // row scaling factors (scaled = factor * original)
  double obj_scale = 1.0;                  // objective scaling factor
  std::vector<char> eq_live;   // equality row has at least one free variable

  // state
  Vec x, zl, zu;               // zl/zu sized n (free entries used)
  Vec y;                       // equality multipliers (size me)
  Vec mu_in, s_in;             // inequality multipliers/slacks (size mi; 0 off working set)
  std::vector<char> inW;       // working-set membership
  std::vector<int> env_row;    // inequality -> envelope row index or -1
  std::vector<int> env_ineq;   // envelope row -> inequality
  std::vector<int> env_kkt_row;    // envelope row -> KKT row index, or -1 when condensed
  std::vector<int> aug_env_of_row; // KKT row (>= nf+me) -> envelope row
  std::vector<double> cond_r;      // inequality -> condensed row right-hand side of the current system
  std::vector<double> cond_r2;     // inequality -> accumulated correction right-hand side
  std::vector<int> inactive_count;
  Vec hval, gval;              // constraint values at x
  Vec hcorr, gcorr;            // exact second-order residuals of the last direction
  double fval = 0.0;
  double mu = 0.1;
  double nu = 1.0;             // merit penalty
  double delta_x_last = 0.0;

  // KKT structure
  int N = 0;
  std::vector<int> ti, tj, slot;
  std::vector<double> tv;
  std::vector<double> reg_dual;      // per KKT index: dual regularization value (0 on primal)
  std::vector<int> diag_slot;        // free index -> triplet index of the diagonal
  std::vector<int> dual_diag_slot;   // KKT index (>= nf) -> triplet index of the dual diagonal
  std::unordered_map<std::int64_t, int> offdiag;  // (fi,fj) -> triplet index
  bool structure_valid = false;
  SparseLdl ldl;

  // timers
  double t_setup = 0, t_factor = 0, t_assembly = 0, t_eval = 0, t_ls = 0, t_cert = 0, t_structure = 0;
  // Fletcher-Leyffer filter (theta = l1 constraint violation, phi = barrier objective)
  std::vector<std::pair<double, double>> filter;
  double filter_theta_max = std::numeric_limits<double>::infinity();
  double filter_mu = -1.0;
  bool filter_acceptable(double theta, double phi) const {
    if (theta > filter_theta_max) return false;
    for (const auto& f : filter)
      if (!(theta <= (1.0 - 1e-5) * f.first || phi <= f.second - 1e-8 * f.first)) return false;
    return true;
  }
  std::string last_inertia_message;

  LazyNewton(const QCQPProblem& pp, const Settings& ss, Result& rr) : p(pp), st(ss), r(rr) {}

  bool is_fixed(int j) const {
    return std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
           std::abs(p.ub[j] - p.lb[j]) <= st.fixed_bound_tolerance;
  }

  // ------------------------------------------------------------------ setup
  void setup() {
    const auto ts = Clock::now();
    n = p.n; me = static_cast<int>(p.equalities.size()); mi = static_cast<int>(p.inequalities.size());
    free_of.assign(n, -1); has_lb.assign(n, 0); has_ub.assign(n, 0);
    for (int j = 0; j < n; ++j) {
      if (is_fixed(j)) continue;
      free_of[j] = static_cast<int>(var_of.size());
      var_of.push_back(j);
      has_lb[j] = std::isfinite(p.lb[j]) ? 1 : 0;
      has_ub[j] = std::isfinite(p.ub[j]) ? 1 : 0;
    }
    nf = static_cast<int>(var_of.size());
    obj = make_squad(p.objective, n, true);
    eq.reserve(me); in.reserve(mi);
    // Equalities carry their exact signed curvature: no DC split is needed.
    for (const auto& h : p.equalities) eq.push_back(make_squad(h, n, false));
    for (const auto& g : p.inequalities) in.push_back(make_squad(g, n, true));
    eq_scale.assign(me, 1.0); in_scale.assign(mi, 1.0);
    eq_live.assign(me, 0);
    for (int j = 0; j < me; ++j)
      for (int v : eq[j].sup) if (free_of[v] >= 0) { eq_live[j] = 1; break; }
    inW.assign(mi, 0); env_row.assign(mi, -1); inactive_count.assign(mi, 0);
    mu_in.assign(mi, 0.0); s_in.assign(mi, 0.0);
    cond_r.assign(mi, 0.0); cond_r2.assign(mi, 0.0);
    y.assign(me, 0.0); zl.assign(n, 0.0); zu.assign(n, 0.0);
    hval.assign(me, 0.0); gval.assign(mi, 0.0); hcorr.assign(me, 0.0); gcorr.assign(mi, 0.0);
    // initial point: fixed variables pinned, others pushed inside the bounds
    x = p.x0;
    for (int j = 0; j < n; ++j) {
      if (free_of[j] < 0) { x[j] = p.lb[j]; continue; }
      const double l = p.lb[j], u = p.ub[j];
      if (has_lb[j] && has_ub[j]) {
        const double push = std::min(st.ln_bound_push * std::max(1.0, std::abs(l)),
                                     st.ln_bound_push * (u - l));
        const double pushu = std::min(st.ln_bound_push * std::max(1.0, std::abs(u)),
                                      st.ln_bound_push * (u - l));
        x[j] = std::min(std::max(x[j], l + push), u - pushu);
      } else if (has_lb[j]) {
        x[j] = std::max(x[j], l + st.ln_bound_push * std::max(1.0, std::abs(l)));
      } else if (has_ub[j]) {
        x[j] = std::min(x[j], u - st.ln_bound_push * std::max(1.0, std::abs(u)));
      }
    }
    mu = st.ln_mu_init;
    // Gradient-based scaling (Ipopt nlp_scaling_method=gradient-based analogue):
    // every row and the objective are scaled so that their gradient at the
    // start has infinity norm at most ln_scaling_max_gradient.
    if (st.ln_scaling_max_gradient > 0.0) {
      auto scale_of = [&](SQuad& q) {
        q.eval(x);
        const double g = q.grad_inf();
        return (g > st.ln_scaling_max_gradient) ? st.ln_scaling_max_gradient / g : 1.0;
      };
      auto apply = [](SQuad& q, double f) {
        if (f == 1.0) return;
        q.c *= f;
        for (double& v : q.lin) v *= f;
        for (double& v : q.hv) v *= f;
        for (double& v : q.pv) v *= f;
      };
      obj_scale = scale_of(obj); apply(obj, obj_scale);
      for (int j = 0; j < me; ++j) { eq_scale[j] = scale_of(eq[j]); apply(eq[j], eq_scale[j]); }
      for (int i = 0; i < mi; ++i) { in_scale[i] = scale_of(in[i]); apply(in[i], in_scale[i]); }
    }
    for (int j = 0; j < n; ++j) {
      if (free_of[j] < 0) continue;
      if (has_lb[j]) zl[j] = mu / (x[j] - p.lb[j]);
      if (has_ub[j]) zu[j] = mu / (p.ub[j] - x[j]);
    }
    t_setup += seconds_since(ts);
  }

  // ------------------------------------------------------------- evaluation
  void evaluate_all() {
    const auto ts = Clock::now();
    fval = obj.eval(x);
    for (int j = 0; j < me; ++j) hval[j] = eq[j].eval(x);
    for (int i = 0; i < mi; ++i) gval[i] = in[i].eval(x);
    t_eval += seconds_since(ts);
  }
  double ineq_scale(int i) const {
    return std::max(1.0, in[i].grad_inf());
  }

  // ----------------------------------------------------- working set update
  // Returns true when the KKT structure changed.
  bool update_working_set(bool initial) {
    bool changed = false;
    for (int i = 0; i < mi; ++i) {
      if (in[i].k() == 0) continue;
      const double sc = ineq_scale(i);
      const double dist = -gval[i] / sc;  // scaled signed distance (>0 feasible)
      if (!inW[i]) {
        if (initial && dist <= st.ln_envelope && env_row[i] < 0) {
          // structural reservation only: the row stays decoupled until activated
          env_row[i] = static_cast<int>(env_ineq.size());
          env_ineq.push_back(i);
          changed = true;
        }
        if (dist <= st.ln_activation) {
          inW[i] = 1; inactive_count[i] = 0;
          s_in[i] = std::max(-gval[i], std::max(mu, 1e-4 * sc));
          // Gentle activation: the multiplier starts as if the row stood at
          // unit scaled distance, so the dual residual jump is at most mu
          // and the barrier complementarity error stays below mu.
          mu_in[i] = mu / std::max(s_in[i], sc);
          ++r.working_set_additions;
          if (env_row[i] < 0) {
            env_row[i] = static_cast<int>(env_ineq.size());
            env_ineq.push_back(i);
            changed = true;
          }
        }
      } else if (!initial) {
        // A row leaves the working set only when dropping its barrier term
        // perturbs the barrier KKT conditions by at most mu (scaled distance
        // beyond ln_drop_distance and multiplier force below mu), for
        // ln_drop_patience consecutive iterations.
        if (dist > st.ln_drop_distance && mu_in[i] * sc <= mu) {
          if (++inactive_count[i] >= st.ln_drop_patience) {
            inW[i] = 0; mu_in[i] = 0.0; s_in[i] = 0.0; inactive_count[i] = 0;
            ++r.working_set_drops;
          }
        } else inactive_count[i] = 0;
      }
    }
    int cnt = 0;
    for (int i = 0; i < mi; ++i) cnt += inW[i];
    r.lazy_newton_working_set_max = std::max(r.lazy_newton_working_set_max, cnt);
    return changed;
  }

  // --------------------------------------------------------- KKT structure
  void build_structure() {
    const auto ts = Clock::now();
    const int nenv = static_cast<int>(env_ineq.size());
    env_kkt_row.assign(nenv, -1);
    aug_env_of_row.clear();
    for (int e = 0; e < nenv; ++e) {
      int free_support = 0;
      for (int v : in[env_ineq[e]].sup) free_support += (free_of[v] >= 0);
      if (free_support > st.ln_condense_max_support) {
        env_kkt_row[e] = nf + me + static_cast<int>(aug_env_of_row.size());
        aug_env_of_row.push_back(e);
      }
    }
    const int naug = static_cast<int>(aug_env_of_row.size());
    N = nf + me + naug;
    ti.clear(); tj.clear(); offdiag.clear();
    diag_slot.assign(nf, -1); dual_diag_slot.assign(N, -1); reg_dual.assign(N, 0.0);
    auto add = [&](int i, int j) { if (i > j) std::swap(i, j); ti.push_back(i); tj.push_back(j); return static_cast<int>(ti.size()) - 1; };
    for (int f = 0; f < nf; ++f) diag_slot[f] = add(f, f);
    auto hess_slots = [&](const std::vector<int>& sup, const std::vector<int>& li,
                          const std::vector<int>& lj, std::vector<int>& out) {
      out.assign(li.size(), -1);
      for (std::size_t t = 0; t < li.size(); ++t) {
        const int fa = free_of[sup[li[t]]], fb = free_of[sup[lj[t]]];
        if (fa < 0 || fb < 0) continue;
        if (fa == fb) { out[t] = diag_slot[fa]; continue; }
        const int a = std::min(fa, fb), b = std::max(fa, fb);
        const std::int64_t key = (static_cast<std::int64_t>(a) << 32) | b;
        auto it = offdiag.find(key);
        if (it == offdiag.end()) it = offdiag.emplace(key, add(a, b)).first;
        out[t] = it->second;
      }
    };
    hess_slots(obj.sup, obj.pi, obj.pj, obj.pslot);
    for (auto& q : eq) hess_slots(q.sup, q.hi, q.hj, q.hslot);
    for (auto& q : in) hess_slots(q.sup, q.pi, q.pj, q.pslot);
    if (st.ln_inequality_curvature != "majorant") for (auto& q : in) hess_slots(q.sup, q.hi, q.hj, q.hslot);
    auto jac_slots = [&](SQuad& q, int row) {
      q.jslot.assign(q.sup.size(), -1);
      for (std::size_t t = 0; t < q.sup.size(); ++t) {
        const int f = free_of[q.sup[t]];
        if (f < 0) continue;
        q.jslot[t] = add(f, row);
      }
    };
    for (int j = 0; j < me; ++j) {
      jac_slots(eq[j], nf + j);
      dual_diag_slot[nf + j] = add(nf + j, nf + j);
      reg_dual[nf + j] = -st.ln_delta_c;
    }
    for (int e = 0; e < nenv; ++e) {
      SQuad& q = in[env_ineq[e]];
      const int row = env_kkt_row[e];
      if (row >= 0) {
        q.cslot.clear();
        jac_slots(q, row);
        dual_diag_slot[row] = add(row, row);
      } else {
        // condensed row: rank-one pattern on the free support
        q.jslot.assign(q.sup.size(), -1);
        const int k = q.k();
        q.cslot.assign(static_cast<std::size_t>(k) * (k + 1) / 2, -1);
        std::size_t c = 0;
        for (int a = 0; a < k; ++a)
          for (int b = a; b < k; ++b, ++c) {
            const int fa = free_of[q.sup[a]], fb = free_of[q.sup[b]];
            if (fa < 0 || fb < 0) continue;
            if (fa == fb) { q.cslot[c] = diag_slot[fa]; continue; }
            const int lo = std::min(fa, fb), hi = std::max(fa, fb);
            const std::int64_t key = (static_cast<std::int64_t>(lo) << 32) | hi;
            auto it = offdiag.find(key);
            if (it == offdiag.end()) it = offdiag.emplace(key, add(lo, hi)).first;
            q.cslot[c] = it->second;
          }
      }
    }
    r.lazy_newton_condensed_rows_max = std::max(r.lazy_newton_condensed_rows_max, nenv - naug);
    r.lazy_newton_augmented_rows_max = std::max(r.lazy_newton_augmented_rows_max, naug);
    tv.assign(ti.size(), 0.0);
    std::string msg;
    ldl.ordering = st.ln_ordering;
    const double t_pattern = seconds_since(ts);
    if (!ldl.analyze(N, ti, tj, slot, st.linear_metis, st.random_seed, msg))
      throw std::runtime_error("lazy-newton: " + msg);
    if (st.verbose) std::printf("[structure] N=%d triplets=%zu pattern %.2f ms analyze %.2f ms (metis %.2f ms)\n", N, ti.size(), 1e3 * t_pattern, 1e3 * (seconds_since(ts) - t_pattern), 1e3 * ldl.t_metis);
    ++r.lazy_newton_symbolic_analyses;
    r.lazy_newton_kkt_dimension = std::max(r.lazy_newton_kkt_dimension, N);
    r.lazy_newton_kkt_nnz = std::max(r.lazy_newton_kkt_nnz, static_cast<int>(ti.size()));
    r.lazy_newton_factor_nnz = std::max(r.lazy_newton_factor_nnz, ldl.factor_nnz);
    structure_valid = true;
    t_structure += seconds_since(ts);
  }

  bool use_exact_curvature = false;   // current inequality curvature model
  double delta_c_eff = 1e-8;          // current dual regularization
  // Numeric assembly. Local gradients must be current (evaluate_all).
  void assemble(double delta_x) {
    const auto ts = Clock::now();
    std::fill(tv.begin(), tv.end(), 0.0);
    for (int f = 0; f < nf; ++f) {
      const int j = var_of[f];
      double d = delta_x;
      if (has_lb[j]) d += zl[j] / (x[j] - p.lb[j]);
      if (has_ub[j]) d += zu[j] / (p.ub[j] - x[j]);
      tv[diag_slot[f]] = d;
    }
    for (std::size_t t = 0; t < obj.pv.size(); ++t)
      if (obj.pslot[t] >= 0) tv[obj.pslot[t]] += obj.pv[t];
    for (int j = 0; j < me; ++j) {
      const SQuad& q = eq[j];
      const double yj = y[j];
      if (yj != 0.0)
        for (std::size_t t = 0; t < q.hv.size(); ++t)
          if (q.hslot[t] >= 0) tv[q.hslot[t]] += yj * q.hv[t];
      for (std::size_t t = 0; t < q.sup.size(); ++t)
        if (q.jslot[t] >= 0) tv[q.jslot[t]] = q.gl[t];
      reg_dual[nf + j] = -delta_c_eff;
      tv[dual_diag_slot[nf + j]] = reg_dual[nf + j];
    }
    for (std::size_t e = 0; e < env_ineq.size(); ++e) {
      const int i = env_ineq[e];
      const int row = env_kkt_row[e];
      const SQuad& q = in[i];
      if (inW[i]) {
        const double m = mu_in[i];
        if (use_exact_curvature) {
          for (std::size_t t = 0; t < q.hv.size(); ++t)
            if (q.hslot[t] >= 0) tv[q.hslot[t]] += m * q.hv[t];
        } else {
          for (std::size_t t = 0; t < q.pv.size(); ++t)
            if (q.pslot[t] >= 0) tv[q.pslot[t]] += m * q.pv[t];
        }
        if (row >= 0) {
          for (std::size_t t = 0; t < q.sup.size(); ++t)
            if (q.jslot[t] >= 0) tv[q.jslot[t]] = q.gl[t];
          tv[dual_diag_slot[row]] = -s_in[i] / m;
          reg_dual[row] = 0.0;
        } else {
          // condensed: primal block += (mu_i/s_i) grad g_i grad g_i'
          const double w = m / s_in[i];
          const int k = q.k();
          std::size_t c = 0;
          for (int a = 0; a < k; ++a)
            for (int b = a; b < k; ++b, ++c)
              if (q.cslot[c] >= 0) tv[q.cslot[c]] += w * q.gl[a] * q.gl[b];
        }
      } else if (row >= 0) {
        for (std::size_t t = 0; t < q.sup.size(); ++t)
          if (q.jslot[t] >= 0) tv[q.jslot[t]] = 0.0;
        tv[dual_diag_slot[row]] = -1.0;
        reg_dual[row] = 0.0;
      }
    }
    t_assembly += seconds_since(ts);
  }

  // y = K v with the assembled triplets, optionally removing the equality
  // regularization (the refinement target is the unregularized system).
  void kkt_matvec(const std::vector<double>& v, std::vector<double>& out, bool unregularized) const {
    std::fill(out.begin(), out.end(), 0.0);
    for (std::size_t t = 0; t < ti.size(); ++t) {
      const int i = ti[t], j = tj[t];
      const double a = tv[t];
      out[i] += a * v[j];
      if (i != j) out[j] += a * v[i];
    }
    if (unregularized)
      for (int k = nf; k < N; ++k) out[k] -= reg_dual[k] * v[k];
  }

  // Solve K d = b with refinement; returns relative backward error.
  double solve_refined(const std::vector<double>& b, std::vector<double>& d) {
    d = b;
    ldl.solve(d);
    const double bn = std::max(1e-300, inf_norm_ptr(b.data(), b.size()));
    std::vector<double> res(N), corr(N);
    double berr = std::numeric_limits<double>::infinity();
    std::vector<double> best = d; double best_berr = berr;
    for (int sweep = 0; sweep <= st.ln_kkt_refine; ++sweep) {
      kkt_matvec(d, res, true);
      for (int k = 0; k < N; ++k) res[k] = b[k] - res[k];
      berr = inf_norm_ptr(res.data(), res.size()) / bn;
      if (berr < best_berr) { best_berr = berr; best = d; }
      if (berr <= st.ln_kkt_berr_tol || sweep == st.ln_kkt_refine) break;
      corr = res;
      ldl.solve(corr);
      for (int k = 0; k < N; ++k) d[k] += corr[k];
      ++r.lazy_newton_refinement_sweeps;
    }
    d = best;
    return best_berr;
  }

  // Least-squares multiplier initialization (identity primal block).
  void initialize_multipliers(std::vector<double>& rx) {
    const auto ts = Clock::now();
    std::fill(tv.begin(), tv.end(), 0.0);
    for (int f = 0; f < nf; ++f) tv[diag_slot[f]] = 1.0;
    for (int j = 0; j < me; ++j) {
      const SQuad& q = eq[j];
      for (std::size_t t = 0; t < q.sup.size(); ++t) if (q.jslot[t] >= 0) tv[q.jslot[t]] = q.gl[t];
      tv[dual_diag_slot[nf + j]] = -std::max(st.ln_delta_c, 1e-8);
    }
    for (std::size_t e = 0; e < env_ineq.size(); ++e) {
      const int i = env_ineq[e];
      const int row = env_kkt_row[e];
      const SQuad& q = in[i];
      // Working-set rows carry an implicit unit slack column in the identity
      // block of the least-squares system: [I J'; J -I]; condensed rows add
      // the equivalent rank-one term grad g grad g' to the identity block.
      if (row >= 0) {
        for (std::size_t t = 0; t < q.sup.size(); ++t) if (q.jslot[t] >= 0) tv[q.jslot[t]] = inW[i] ? q.gl[t] : 0.0;
        tv[dual_diag_slot[row]] = -1.0;
      } else if (inW[i]) {
        const int k = q.k();
        std::size_t c = 0;
        for (int a = 0; a < k; ++a)
          for (int b = a; b < k; ++b, ++c)
            if (q.cslot[c] >= 0) tv[q.cslot[c]] += q.gl[a] * q.gl[b];
      }
    }
    ++r.lazy_newton_factorizations;
    if (!ldl.factor(tv, slot)) { t_factor += seconds_since(ts); return; }
    std::vector<double> b(N, 0.0), d(N, 0.0);
    errors(rx);  // rx = grad f - zl + zu with zero multipliers
    for (int f = 0; f < nf; ++f) b[f] = -rx[f];
    b = b; d = b; ldl.solve(d);
    double ymax = 0.0;
    for (int k = nf; k < N; ++k) ymax = std::max(ymax, std::abs(d[k]));
    std::vector<double> est_in(mi, 0.0);
    {
      Vec dxl(n, 0.0);
      for (int f = 0; f < nf; ++f) dxl[var_of[f]] = d[f];
      for (std::size_t e = 0; e < env_ineq.size(); ++e) {
        const int i = env_ineq[e];
        if (!inW[i]) continue;
        est_in[i] = env_kkt_row[e] >= 0 ? d[env_kkt_row[e]] : in[i].grad_dot(dxl);
        ymax = std::max(ymax, std::abs(est_in[i]));
      }
    }
    if (!std::isfinite(ymax)) { t_factor += seconds_since(ts); return; }
    if (st.verbose) {
      std::vector<int> idx(N - nf);
      for (int k = 0; k < N - nf; ++k) idx[k] = k;
      std::partial_sort(idx.begin(), idx.begin() + std::min(8, N - nf), idx.end(),
                        [&](int a, int b) { return std::abs(d[nf + a]) > std::abs(d[nf + b]); });
      std::printf("LS multiplier init: max |y| = %.3e\n", ymax);
      for (int k = 0; k < std::min(8, N - nf); ++k) {
        const int row = idx[k];
        const std::string name = row < me ? p.equalities[row].name : p.inequalities[env_ineq[aug_env_of_row[row - me]]].name;
        std::printf("   %-24s %12.4e\n", name.c_str(), d[nf + row]);
      }
    }
    if (ymax > st.ln_mult_init_max) { t_factor += seconds_since(ts); return; }
    for (int j = 0; j < me; ++j) y[j] = d[nf + j];
    for (std::size_t e = 0; e < env_ineq.size(); ++e) {
      const int i = env_ineq[e];
      if (!inW[i]) continue;
      // inequality multipliers must stay strictly positive
      mu_in[i] = std::max(mu / s_in[i], est_in[i]);
    }
    t_factor += seconds_since(ts);
  }

  // Factorize with inertia control. Returns the shift used (or -1 on failure).
  double factor_with_inertia(double delta_floor = 0.0) {
    const auto ts = Clock::now();
    int active_rows = me;
    for (int i = 0; i < mi; ++i) active_rows += inW[i];
    const int expected_neg = N - nf;  // every dual row, decoupled rows included
    (void)active_rows;
    double delta = delta_floor;
    // The unshifted system has structurally zero primal pivots for unbounded
    // variables without curvature; start from the previous shift reduced by
    // three, floored at ln_delta_x_min, instead of a guaranteed failure.
    if (delta == 0.0 && delta_x_last > 0.0)
      delta = std::max(st.ln_delta_x_min, delta_x_last / 3.0);
    delta_c_eff = st.ln_delta_c;
    use_exact_curvature = st.ln_inequality_curvature != "majorant";
    int dual_escalations = 0;
    for (int attempt = 0; attempt < 60; ++attempt) {
      assemble(delta);
      ++r.lazy_newton_factorizations;
      const bool ok = ldl.factor(tv, slot);
      if (ok && ldl.npos == nf && ldl.nneg == expected_neg) {
        delta_x_last = delta;
        r.lazy_newton_max_delta_x = std::max(r.lazy_newton_max_delta_x, delta);
        if (use_exact_curvature) ++r.lazy_newton_exact_curvature_steps; else ++r.lazy_newton_majorant_curvature_steps;
        t_factor += seconds_since(ts);
        return delta;
      }
      ++r.lazy_newton_inertia_corrections;
      if (ldl.npos >= nf && ldl.nneg < expected_neg && dual_escalations < 4) {
        // Too few negative pivots: the constraint rows are (numerically)
        // dependent.  Regularize the dual block (Ipopt jacobian
        // regularization analogue) before shifting the primal block.
        delta_c_eff = std::max(1e-8, delta_c_eff * 100.0);
        ++dual_escalations; ++r.lazy_newton_dual_regularizations;
        continue;
      }
      if (st.ln_inequality_curvature == "adaptive" && use_exact_curvature && delta >= st.ln_adaptive_curvature_delta) {
        // Adaptive DC fallback: the exact inequality curvature could not be
        // certified with a moderate shift; revert to the PSD majorant
        // curvature (always a valid convex model) and restart the shifts.
        use_exact_curvature = false;
        delta = (delta_x_last == 0.0) ? st.ln_delta_x_init : std::max(st.ln_delta_x_min, delta_x_last / 3.0);
        ++r.lazy_newton_curvature_fallbacks;
        continue;
      }
      if (delta == 0.0) delta = (delta_x_last == 0.0) ? st.ln_delta_x_init : std::max(1e-20, delta_x_last / 3.0);
      else delta *= 8.0;
      delta = std::max(delta, delta_floor);
      if (delta > st.ln_delta_x_max) break;
    }
    t_factor += seconds_since(ts);
    last_inertia_message = "inertia correction failed: npos=" + std::to_string(ldl.npos) +
        " nneg=" + std::to_string(ldl.nneg) + " nzero=" + std::to_string(ldl.nzero) +
        " expected (" + std::to_string(nf) + "," + std::to_string(expected_neg) + ")";
    for (int k : ldl.zero_pivot_rows()) {
      std::string name = k < nf ? ("var " + std::to_string(var_of[k])) :
          (k < nf + me ? ("eq " + p.equalities[k - nf].name) :
           ("ineq " + p.inequalities[env_ineq[aug_env_of_row[k - nf - me]]].name + (inW[env_ineq[aug_env_of_row[k - nf - me]]] ? " (W)" : " (off)")));
      if (k >= nf + me) { const int i = env_ineq[aug_env_of_row[k - nf - me]]; name += " s=" + std::to_string(s_in[i]) + " mu=" + std::to_string(mu_in[i]) + " g=" + std::to_string(gval[i]); }
      last_inertia_message += "; zero pivot at " + name;
    }
    return -1.0;
  }

  // ------------------------------------------------------------ residuals
  // Dual infeasibility of the current point with the current multipliers
  // (free variables only), constraint violation, complementarities.
  struct Err { double dual = 0, primal = 0, compl0 = 0, compl_mu = 0, primal_ws = 0;
               double dual_u = 0, primal_u = 0, compl_u = 0; };  // original (unscaled) units
  Err errors(std::vector<double>& rx) const {
    rx.assign(nf, 0.0);
    for (std::size_t t = 0; t < obj.sup.size(); ++t) {
      const int f = free_of[obj.sup[t]];
      if (f >= 0) rx[f] += obj.gl[t];
    }
    for (int j = 0; j < me; ++j) {
      if (y[j] == 0.0) continue;
      const SQuad& q = eq[j];
      for (std::size_t t = 0; t < q.sup.size(); ++t) {
        const int f = free_of[q.sup[t]];
        if (f >= 0) rx[f] += y[j] * q.gl[t];
      }
    }
    for (int i = 0; i < mi; ++i) {
      if (!inW[i] || mu_in[i] == 0.0) continue;
      const SQuad& q = in[i];
      for (std::size_t t = 0; t < q.sup.size(); ++t) {
        const int f = free_of[q.sup[t]];
        if (f >= 0) rx[f] += mu_in[i] * q.gl[t];
      }
    }
    Err e;
    for (int f = 0; f < nf; ++f) {
      const int j = var_of[f];
      rx[f] += -zl[j] + zu[j];
      e.dual = std::max(e.dual, std::abs(rx[f]));
      if (has_lb[j]) {
        const double c = zl[j] * (x[j] - p.lb[j]);
        e.compl0 = std::max(e.compl0, c); e.compl_mu = std::max(e.compl_mu, std::abs(c - mu));
      }
      if (has_ub[j]) {
        const double c = zu[j] * (p.ub[j] - x[j]);
        e.compl0 = std::max(e.compl0, c); e.compl_mu = std::max(e.compl_mu, std::abs(c - mu));
      }
    }
    for (int j = 0; j < me; ++j) {
      e.primal = std::max(e.primal, std::abs(hval[j]));
      e.primal_u = std::max(e.primal_u, std::abs(hval[j]) / eq_scale[j]);
    }
    for (int i = 0; i < mi; ++i) {
      e.primal = std::max(e.primal, std::max(0.0, gval[i]));
      e.primal_u = std::max(e.primal_u, std::max(0.0, gval[i]) / in_scale[i]);
      if (inW[i]) {
        e.primal_ws = std::max(e.primal_ws, std::abs(gval[i] + s_in[i]));
        e.compl0 = std::max(e.compl0, std::abs(mu_in[i] * gval[i]));
        e.compl_mu = std::max(e.compl_mu, std::abs(mu_in[i] * s_in[i] - mu));
      }
    }
    e.dual_u = e.dual / obj_scale;
    e.compl_u = e.compl0 / obj_scale;
    return e;
  }

  // Barrier merit at (x,s): f - mu*sum log s - mu*sum log(bounds) + nu*||c||_1
  double merit(const Vec& xt, const Vec& st_in, double& cnorm) const {
    double phi = obj.value_only(xt);
    cnorm = 0.0;
    for (int j = 0; j < n; ++j) {
      if (free_of[j] < 0) continue;
      if (has_lb[j]) { const double d = xt[j] - p.lb[j]; if (d <= 0.0) return std::numeric_limits<double>::infinity(); phi -= mu * std::log(d); }
      if (has_ub[j]) { const double d = p.ub[j] - xt[j]; if (d <= 0.0) return std::numeric_limits<double>::infinity(); phi -= mu * std::log(d); }
    }
    for (int j = 0; j < me; ++j) cnorm += std::abs(eq[j].value_only(xt));
    for (int i = 0; i < mi; ++i) {
      if (!inW[i]) continue;
      if (st_in[i] <= 0.0) return std::numeric_limits<double>::infinity();
      phi -= mu * std::log(st_in[i]);
      cnorm += std::abs(in[i].value_only(xt) + st_in[i]);
    }
    return phi + nu * cnorm;
  }

  // Largest alpha in (0,1] keeping the non-modeled inequalities feasible on
  // the segment x + alpha d, using the closed-form quadratic restriction.
  double lazy_segment_bound(const Vec& d, int& blocking) const {
    double alpha = 1.0; blocking = -1;
    for (int i = 0; i < mi; ++i) {
      if (inW[i] || in[i].k() == 0) continue;
      const SQuad& q = in[i];
      const double c = gval[i];
      if (c > 0.0) continue; // already violated: will be activated, cannot guard
      const double b = q.grad_dot(d);
      const double a = q.half_quad(d);
      // find smallest t>0 with a t^2 + b t + c = 0
      double t = std::numeric_limits<double>::infinity();
      if (std::abs(a) <= 1e-14 * std::max(1.0, std::abs(b))) {
        if (b > 0.0) t = -c / b;
      } else {
        const double disc = b * b - 4.0 * a * c;
        if (disc >= 0.0) {
          const double sq = std::sqrt(disc);
          const double r1 = (-b - sq) / (2.0 * a), r2 = (-b + sq) / (2.0 * a);
          if (r1 > 0.0) t = std::min(t, r1);
          if (r2 > 0.0) t = std::min(t, r2);
        }
      }
      if (t < alpha) { alpha = t; blocking = i; }
    }
    return alpha;
  }

  // ------------------------------------------------------------- iterate
  Status run() {
    setup();
    evaluate_all();
    update_working_set(true);
    build_structure();
    std::vector<double> rx, rhs(N), d(N), b2(N), d2(N), d2sum(N);
    initialize_multipliers(rx);
    auto resize_work = [&]() { rhs.assign(N, 0.0); d.assign(N, 0.0); b2.assign(N, 0.0); d2.assign(N, 0.0); d2sum.assign(N, 0.0); };
    Vec dx(n, 0.0), xt(n), st_trial(mi, 0.0), ds(mi, 0.0);
    const double tol_d = st.dual_infeasibility_tolerance, tol_p = st.feasibility_tolerance,
                 tol_c = st.complementarity_tolerance, tol = st.stationarity_tolerance;
    const double mu_min = std::min({tol, tol_c}) * obj_scale / 10.0;
    int iter = 0;
    int consecutive_failures = 0;
    double proximal_retry = 0.0;
    bool structure_dirty = false;
    if (st.live_iterations || st.verbose)
      std::printf("%4s %9s %12s %9s %9s %9s %9s %5s %7s %7s %8s %3s %s\n", "it", "mu", "objective",
                  "primal", "dual", "compl", "berr", "|W|", "alpha_p", "alpha_d", "delta_x", "ls", "flags");
    for (;;) {
      if (seconds_since(t0) > st.time_limit_seconds) { r.message = "time limit"; return Status::TimeLimit; }
      Err e = errors(rx);
      r.model_stationarity_inf = e.dual;
      const bool internal_converged = e.dual_u <= tol_d && e.primal_u <= tol_p && e.compl_u <= tol_c;
      if (internal_converged) {
        if (certify()) { r.message = "original-QCQP KKT test satisfied by lazy primal-dual Newton"; return Status::Solved; }
      }
      if (iter >= st.max_outer_iterations) { r.message = "maximum outer iterations"; return Status::MaxOuterIterations; }
      // barrier update (monotone Fiacco-McCormick)
      {
        const double e_mu = std::max({e.dual, e.primal, e.primal_ws, e.compl_mu});
        int updates = 0;
        while (e_mu <= st.ln_kappa_epsilon * mu && mu > mu_min && updates < 20) {
          mu = std::max(mu_min, std::min(st.ln_mu_kappa * mu, std::pow(mu, st.ln_mu_theta)));
          ++updates; ++r.lazy_newton_barrier_updates;
          if (e_mu > st.ln_kappa_epsilon * mu) break;
        }
      }
      if (structure_dirty) { build_structure(); resize_work(); structure_dirty = false; }
      // rhs (independent of the inertia shift)
      for (int f = 0; f < nf; ++f) {
        const int j = var_of[f];
        double v = rx[f] + zl[j] - zu[j];  // grad f + J'y + J_W'mu
        if (has_lb[j]) v -= mu / (x[j] - p.lb[j]);
        if (has_ub[j]) v += mu / (p.ub[j] - x[j]);
        rhs[f] = -v;
      }
      for (int j = 0; j < me; ++j) rhs[nf + j] = eq_live[j] ? -hval[j] : 0.0;
      std::fill(cond_r.begin(), cond_r.end(), 0.0);
      std::fill(cond_r2.begin(), cond_r2.end(), 0.0);
      for (std::size_t ee = 0; ee < env_ineq.size(); ++ee) {
        const int i = env_ineq[ee];
        const int row = env_kkt_row[ee];
        const double ri = inW[i] ? -(gval[i] + mu / mu_in[i]) : 0.0;
        if (row >= 0) rhs[row] = ri;
        else if (inW[i]) {
          cond_r[i] = ri;
          const double w = mu_in[i] / s_in[i];
          const SQuad& q = in[i];
          for (std::size_t t = 0; t < q.sup.size(); ++t) {
            const int f = free_of[q.sup[t]];
            if (f >= 0) rhs[f] += w * ri * q.gl[t];
          }
        }
      }
      // Newton system with inertia control and adaptive proximal shift: a
      // direction blocked far before unit length by the boundary or by a
      // non-modeled inequality is recomputed with a larger shift.
      double delta = -1.0, berr = 0.0, alpha_p = 0.0, alpha_lazy = 1.0;
      int blocking = -1;
      double delta_floor = proximal_retry;
      const double tau = std::max(st.ln_tau_min, 1.0 - mu);
      for (int prox = 0; prox < 8; ++prox) {
        delta = factor_with_inertia(delta_floor);
        if (delta < 0.0) { r.message = last_inertia_message; return Status::NumericalError; }
        berr = solve_refined(rhs, d);
        std::fill(dx.begin(), dx.end(), 0.0);
        for (int f = 0; f < nf; ++f) dx[var_of[f]] = d[f];
        // primal step bound from bounds and working-set slacks
        alpha_p = 1.0;
        int blk_var = -1; double blk_dx = 0.0, blk_dist = 0.0;
        for (int f = 0; f < nf; ++f) {
          const int j = var_of[f];
          if (has_lb[j] && dx[j] < 0.0) { const double a = -tau * (x[j] - p.lb[j]) / dx[j]; if (a < alpha_p) { alpha_p = a; blk_var = j; blk_dx = dx[j]; blk_dist = x[j] - p.lb[j]; } }
          if (has_ub[j] && dx[j] > 0.0) { const double a = tau * (p.ub[j] - x[j]) / dx[j]; if (a < alpha_p) { alpha_p = a; blk_var = j; blk_dx = dx[j]; blk_dist = p.ub[j] - x[j]; } }
        }
        if (st.verbose && blk_var >= 0 && alpha_p < 0.1)
          std::printf("   blocked by variable %d: dx=%.3e dist=%.3e zl=%.3e zu=%.3e |dx|inf=%.3e\n", blk_var, blk_dx, blk_dist, zl[blk_var], zu[blk_var], inf_norm(dx));
        for (std::size_t ee = 0; ee < env_ineq.size(); ++ee) {
          const int i = env_ineq[ee];
          if (!inW[i]) continue;
          const double dsi = -(gval[i] + in[i].grad_dot(dx) + s_in[i]);
          if (dsi < 0.0) alpha_p = std::min(alpha_p, -tau * s_in[i] / dsi);
        }
        alpha_lazy = lazy_segment_bound(dx, blocking);
        if (st.ln_prox_alpha_min <= 0.0 || std::min(alpha_p, alpha_lazy) >= st.ln_prox_alpha_min || prox == 7) break;
        delta_floor = std::max(delta * 10.0, st.ln_delta_x_init);
        ++r.lazy_newton_proximal_shifts;
      }
      (void)berr;
      // exact second-order residuals of the direction on the quadratic rows
      // (curvature information used for the correction and diagnostics)
      auto directional = [&](const Vec& dd, std::vector<double>& dmu, Vec& dsv, Vec& dzl, Vec& dzu) {
        dsv.assign(mi, 0.0); dmu.assign(mi, 0.0);
        for (std::size_t ee = 0; ee < env_ineq.size(); ++ee) {
          const int i = env_ineq[ee];
          if (!inW[i]) continue;
          const int row = env_kkt_row[ee];
          dmu[i] = row >= 0 ? d[row] : (mu_in[i] / s_in[i]) * (in[i].grad_dot(dd) - cond_r[i]);
          dsv[i] = -(gval[i] + in[i].grad_dot(dd) + s_in[i]);
        }
        dzl.assign(n, 0.0); dzu.assign(n, 0.0);
        for (int f = 0; f < nf; ++f) {
          const int j = var_of[f];
          if (has_lb[j]) dzl[j] = (mu - zl[j] * (x[j] - p.lb[j]) - zl[j] * dd[j]) / (x[j] - p.lb[j]);
          if (has_ub[j]) dzu[j] = (mu - zu[j] * (p.ub[j] - x[j]) + zu[j] * dd[j]) / (p.ub[j] - x[j]);
        }
      };
      std::vector<double> dmu; Vec dsv, dzl, dzu;
      directional(dx, dmu, dsv, dzl, dzu);

      // step bounds
      auto primal_bound = [&](const Vec& dd, const Vec& dss) {
        double a = 1.0;
        for (int f = 0; f < nf; ++f) {
          const int j = var_of[f];
          if (has_lb[j] && dd[j] < 0.0) a = std::min(a, -tau * (x[j] - p.lb[j]) / dd[j]);
          if (has_ub[j] && dd[j] > 0.0) a = std::min(a, tau * (p.ub[j] - x[j]) / dd[j]);
        }
        for (int i = 0; i < mi; ++i) if (inW[i] && dss[i] < 0.0) a = std::min(a, -tau * s_in[i] / dss[i]);
        return a;
      };
      auto dual_bound = [&](const std::vector<double>& dmm, const Vec& dl, const Vec& du) {
        double a = 1.0;
        for (int i = 0; i < mi; ++i) if (inW[i] && dmm[i] < 0.0) a = std::min(a, -tau * mu_in[i] / dmm[i]);
        for (int f = 0; f < nf; ++f) {
          const int j = var_of[f];
          if (has_lb[j] && dl[j] < 0.0) a = std::min(a, -tau * zl[j] / dl[j]);
          if (has_ub[j] && du[j] < 0.0) a = std::min(a, -tau * zu[j] / du[j]);
        }
        return a;
      };
      bool lazy_cut = false;
      if (alpha_lazy < alpha_p) { alpha_p = std::max(0.0, 0.95 * alpha_lazy); lazy_cut = true; ++r.lazy_newton_lazy_cuts; }
      double alpha_d = dual_bound(dmu, dzl, dzu);

      // merit penalty update
      const auto tls = Clock::now();
      double cnorm0 = 0.0;
      double dphi_b = obj.grad_dot(dx);
      double quad = 0.0;
      {
        // d'(W)d on the primal block (assembled values include Sigma and delta)
        for (std::size_t t = 0; t < ti.size(); ++t) {
          if (ti[t] >= nf || tj[t] >= nf) continue;
          const double a = tv[t] * d[ti[t]] * d[tj[t]];
          quad += (ti[t] == tj[t]) ? a : 2.0 * a;
        }
        for (int f = 0; f < nf; ++f) {
          const int j = var_of[f];
          if (has_lb[j]) dphi_b -= mu * dx[j] / (x[j] - p.lb[j]);
          if (has_ub[j]) dphi_b += mu * dx[j] / (p.ub[j] - x[j]);
        }
        for (int i = 0; i < mi; ++i) if (inW[i]) dphi_b -= mu * dsv[i] / s_in[i];
      }
      // Exact-penalty weight: at least the size of the updated multipliers
      // (exactness) and large enough for descent (Nocedal-Wright 18.36); it
      // is re-estimated every iteration and may decrease, at most tenfold,
      // so that a stale weight from an early long step cannot reject
      // second-order corrected steps later (Maratos effect).
      double nu_required = 0.0;
      for (int j = 0; j < me; ++j) nu_required = std::max(nu_required, std::abs(y[j] + d[nf + j]));
      for (std::size_t ee = 0; ee < env_ineq.size(); ++ee) {
        const int i = env_ineq[ee];
        if (inW[i]) nu_required = std::max(nu_required, std::abs(mu_in[i] + dmu[i]));
      }
      nu_required = 1.1 * nu_required + 1.0;
      if (cnorm0 > 1e-14) {
        const double nu_trial = (dphi_b + 0.5 * std::max(0.0, quad)) / (0.5 * cnorm0);
        nu_required = std::max(nu_required, nu_trial + 1.0);
      }
      nu = std::max(nu_required, 0.1 * nu);
      const double phi0 = merit(x, s_in, cnorm0);
      const double dphi = dphi_b - nu * cnorm0;
      const bool use_filter = st.ln_globalization == "filter";
      const double phib0 = phi0 - nu * cnorm0;   // barrier objective at the current point
      if (use_filter && filter_mu != mu) {
        // Ipopt-style reset at every barrier update
        filter.clear();
        filter_mu = mu;
        filter_theta_max = 1e4 * std::max(1.0, cnorm0);
      }
      bool filter_augment = false;
      // Acceptance test of a trial point (phi_trial includes nu*cnorm_trial).
      auto acceptable = [&](double a, double phi_trial, double cnorm_trial) {
        if (!std::isfinite(phi_trial)) return false;
        if (!use_filter) return phi_trial <= phi0 + 1e-4 * a * std::min(dphi, 0.0);
        const double phib_t = phi_trial - nu * cnorm_trial;
        if (!filter_acceptable(cnorm_trial, phib_t)) return false;
        const bool switching = dphi_b < 0.0 &&
            a * std::pow(-dphi_b, 2.3) > std::pow(cnorm0, 1.1);
        if (switching && phib_t <= phib0 + 1e-4 * a * dphi_b) { filter_augment = false; return true; }
        if (cnorm_trial <= (1.0 - 1e-5) * cnorm0 || phib_t <= phib0 - 1e-8 * cnorm0) { filter_augment = true; return true; }
        return false;
      };
      // Armijo backtracking with one exact quadratic correction attempt.
      double alpha = alpha_p;
      int ls = 0;
      bool accepted = false, corrected = false;
      Vec dx_used = dx, dsv_used = dsv;
      double cnorm_t = 0.0, phi_t = 0.0;
      for (; ls < st.ln_max_line_search; ++ls) {
        for (int j = 0; j < n; ++j) xt[j] = x[j] + alpha * dx_used[j];
        for (int i = 0; i < mi; ++i) st_trial[i] = inW[i] ? s_in[i] + alpha * dsv_used[i] : 0.0;
        phi_t = merit(xt, st_trial, cnorm_t);
        if (acceptable(alpha, phi_t, cnorm_t)) { accepted = true; break; }
        if (ls == 0 && st.ln_quadratic_correction && !lazy_cut) {
          // Exact quadratic (second-order) correction: for quadratic rows the
          // residual of the unit step is exactly 1/2 d'Qd, so the correction
          // needs no re-evaluation; up to ln_max_soc sweeps reuse the same
          // factorization, each one a Newton step on the constraint residual
          // with frozen Jacobian.
          Vec dxc(n, 0.0);
          for (int j = 0; j < n; ++j) dxc[j] = alpha * dx[j];
          std::fill(d2sum.begin(), d2sum.end(), 0.0);
          double res_prev = std::numeric_limits<double>::infinity();
          for (int sweep = 0; sweep < st.ln_max_soc; ++sweep) {
            ++r.lazy_newton_quadratic_corrections;
            double res = 0.0;
            for (int f = 0; f < nf; ++f) b2[f] = 0.0;
            for (int j = 0; j < me; ++j) {
              const double v = eq_live[j] ? (hval[j] + eq[j].grad_dot(dxc) + eq[j].half_quad(dxc)) - (1.0 - alpha) * hval[j] : 0.0;
              b2[nf + j] = -v; res = std::max(res, std::abs(v));
            }
            for (std::size_t ee = 0; ee < env_ineq.size(); ++ee) {
              const int i = env_ineq[ee];
              const double v = inW[i] ? in[i].half_quad(dxc) : 0.0;
              const int row = env_kkt_row[ee];
              if (row >= 0) b2[row] = -v;
              else if (inW[i]) {
                const double w = mu_in[i] / s_in[i];
                const SQuad& q = in[i];
                cond_r2[i] += -v;
                for (std::size_t t = 0; t < q.sup.size(); ++t) {
                  const int f = free_of[q.sup[t]];
                  if (f >= 0) b2[f] += w * (-v) * q.gl[t];
                }
              }
              res = std::max(res, std::abs(v));
            }
            if (res > 0.99 * res_prev) break;
            res_prev = res;
            solve_refined(b2, d2);
            for (int f = 0; f < nf; ++f) dxc[var_of[f]] += d2[f];
            for (int k = 0; k < N; ++k) d2sum[k] += d2[k];
            Vec dsc(mi, 0.0);
            for (int i = 0; i < mi; ++i) if (inW[i]) dsc[i] = -(gval[i] + in[i].grad_dot(dxc) + s_in[i]);
            double ap = primal_bound(dxc, dsc);
            int blk = -1;
            const double al = lazy_segment_bound(dxc, blk);
            if (al < ap) ap = 0.95 * al;
            if (ap < 0.5) break;
            for (int j = 0; j < n; ++j) xt[j] = x[j] + ap * dxc[j];
            for (int i = 0; i < mi; ++i) st_trial[i] = inW[i] ? s_in[i] + ap * dsc[i] : 0.0;
            double cn = 0.0;
            const double ph = merit(xt, st_trial, cn);
            if (acceptable(ap * alpha, ph, cn)) {
              accepted = true; corrected = true; ++r.lazy_newton_quadratic_correction_acceptances;
              dx_used = dxc; dsv_used = dsc; alpha = ap; phi_t = ph; cnorm_t = cn;
              break;
            }
          }
          if (accepted) break;
        }
        alpha *= 0.5;
        ++r.lazy_newton_line_search_backtracks;
      }
      t_ls += seconds_since(tls);
      if (!accepted) {
        ++consecutive_failures;
        if (consecutive_failures <= 3) {
          // proximal retry: force a larger inertia shift at the next factorization
          proximal_retry = (proximal_retry == 0.0) ? std::max(st.ln_delta_x_init, 10.0 * delta) : proximal_retry * 10.0;
          delta_x_last = proximal_retry;
          if (st.live_iterations || st.verbose)
            std::printf("%4d line search failed; proximal retry delta=%.1e\n", iter, proximal_retry);
          ++iter;
          continue;
        }
        r.message = "line search failure";
        return Status::LineSearchFailure;
      }
      consecutive_failures = 0; proximal_retry = 0.0;
      if (use_filter && filter_augment) filter.emplace_back(cnorm0, phib0);
      // dual direction for the corrected step is the Newton dual direction
      if (corrected) {
        // recompute the multiplier directions consistently with the corrected primal step
        Vec d2x(n, 0.0);
        for (int f = 0; f < nf; ++f) d2x[var_of[f]] = d2sum[f];
        for (std::size_t ee = 0; ee < env_ineq.size(); ++ee) {
          const int i = env_ineq[ee];
          if (!inW[i]) continue;
          const int row = env_kkt_row[ee];
          dmu[i] = row >= 0 ? d[row] + d2sum[row]
                            : dmu[i] + (mu_in[i] / s_in[i]) * (in[i].grad_dot(d2x) - cond_r2[i]);
        }
        for (int f = 0; f < nf; ++f) {
          const int j = var_of[f];
          if (has_lb[j]) dzl[j] = (mu - zl[j] * (x[j] - p.lb[j]) - zl[j] * dx_used[j]) / (x[j] - p.lb[j]);
          if (has_ub[j]) dzu[j] = (mu - zu[j] * (p.ub[j] - x[j]) + zu[j] * dx_used[j]) / (p.ub[j] - x[j]);
        }
        alpha_d = dual_bound(dmu, dzl, dzu);
      }
      // accept
      for (int j = 0; j < n; ++j) x[j] += alpha * dx_used[j];
      for (int i = 0; i < mi; ++i) if (inW[i]) s_in[i] += alpha * dsv_used[i];
      for (int j = 0; j < me; ++j) y[j] += alpha_d * (corrected ? d[nf + j] + d2sum[nf + j] : d[nf + j]);
      for (int i = 0; i < mi; ++i) if (inW[i]) mu_in[i] += alpha_d * dmu[i];
      for (int f = 0; f < nf; ++f) {
        const int j = var_of[f];
        if (has_lb[j]) {
          zl[j] += alpha_d * dzl[j];
          const double lo = mu / (1e10 * (x[j] - p.lb[j])), hi = 1e10 * mu / (x[j] - p.lb[j]);
          zl[j] = std::min(std::max(zl[j], lo), hi);
        }
        if (has_ub[j]) {
          zu[j] += alpha_d * dzu[j];
          const double lo = mu / (1e10 * (p.ub[j] - x[j])), hi = 1e10 * mu / (p.ub[j] - x[j]);
          zu[j] = std::min(std::max(zu[j], lo), hi);
        }
      }
      ++iter; ++r.lazy_newton_iterations;
      evaluate_all();
      if (update_working_set(false)) structure_dirty = true;
      int wcount = 0; for (int i = 0; i < mi; ++i) wcount += inW[i];
      if (st.live_iterations || st.verbose) {
        std::printf("%4d %9.2e %12.6e %9.2e %9.2e %9.2e %9.2e %5d %7.3f %7.3f %8.1e %3d %s%s%s  [scaled dual %.1e ws %.1e cmu %.1e |dx| %.1e]\n", iter, mu, fval,
                    e.primal_u, e.dual_u, e.compl_u, berr, wcount, alpha, alpha_d, delta, ls,
                    corrected ? "C" : "", lazy_cut ? "L" : "", structure_dirty ? "S" : "",
                    e.dual, e.primal_ws, e.compl_mu, inf_norm(dx_used));
        std::fflush(stdout);
      }
    }
  }

  // ---------------------------------------------------------- certificate
  bool certify() {
    const auto ts = Clock::now();
    ++r.lazy_newton_certificate_evaluations;
    QCQPTerminationMultipliers m;
    m.equality.assign(me, 0.0);
    for (int j = 0; j < me; ++j) m.equality[j] = y[j] * eq_scale[j] / obj_scale;
    m.inequality.assign(mi, 0.0);
    for (int i = 0; i < mi; ++i) if (inW[i]) m.inequality[i] = std::max(0.0, mu_in[i]) * in_scale[i] / obj_scale;
    m.lower_bound.assign(n, 0.0); m.upper_bound.assign(n, 0.0);
    for (int j = 0; j < n; ++j) {
      if (free_of[j] < 0) continue;
      if (has_lb[j]) m.lower_bound[j] = std::max(0.0, zl[j]) / obj_scale;
      if (has_ub[j]) m.upper_bound[j] = std::max(0.0, zu[j]) / obj_scale;
    }
    SparseQCQPEvaluation evaluator;
    const QCQPIpoptTermination c = qcqp_ipopt_termination_evaluated(p, x, m, st,evaluator);
    r.dual_infeasibility_inf = c.stop.dual_infeasibility;
    r.complementarity_inf = c.stop.complementarity;
    r.scaled_nlp_error = c.stop.overall_nlp_error;
    r.dual_error_scaling = c.stop.dual_scaling;
    r.complementarity_error_scaling = c.stop.complementarity_scaling;
    r.constraint_multiplier_l1 = c.stop.y_l1;
    r.bound_slack_multiplier_l1 = c.stop.z_l1;
    r.ipopt_dual_multiplier_dimension = c.stop.dual_multiplier_dimension;
    r.ipopt_complementarity_multiplier_dimension = c.stop.complementarity_multiplier_dimension;
    r.ipopt_y_c_dimension = c.y_c_dimension; r.ipopt_y_d_dimension = c.y_d_dimension;
    r.ipopt_z_L_dimension = c.z_L_dimension; r.ipopt_z_U_dimension = c.z_U_dimension;
    r.ipopt_v_L_dimension = c.v_L_dimension; r.ipopt_v_U_dimension = c.v_U_dimension;
    r.equality_multiplier_inf = c.equality_multiplier_inf;
    r.inequality_multiplier_inf = c.inequality_multiplier_inf;
    r.bound_multiplier_inf = c.bound_multiplier_inf;
    r.ipopt_desired_convergence = c.stop.converged;
    r.certificate_equality_multipliers = m.equality;
    r.certificate_inequality_multipliers = m.inequality;
    r.certificate_lower_multipliers = m.lower_bound;
    r.certificate_upper_multipliers = m.upper_bound;
    r.termination_multiplier_source = "lazy-newton-primal-dual";
    r.termination_multiplier_fit_converged = c.stop.converged;
    t_cert += seconds_since(ts);
    return c.stop.converged;
  }

  void finish(Status status) {
    r.status = status;
    r.x = x;
    r.objective = p.objective.value(x);
    r.feasibility = feasibility(p, x);
    r.outer_iterations = r.lazy_newton_iterations;
    r.inner_iterations = r.lazy_newton_iterations;
    int wcount = 0; for (int i = 0; i < mi; ++i) wcount += inW[i];
    r.active_inequalities = wcount;
    r.lazy_newton_working_set_final = wcount;
    r.lazy_newton_final_mu = mu;
    r.lazy_newton_objective_scale = obj_scale;
    r.lazy_newton_setup_seconds = t_setup;
    r.lazy_newton_factorization_seconds = t_factor;
    r.lazy_newton_assembly_seconds = t_assembly;
    r.lazy_newton_structure_seconds = t_structure;
    r.lazy_newton_evaluation_seconds = t_eval;
    r.lazy_newton_line_search_seconds = t_ls;
    r.lazy_newton_certificate_seconds = t_cert;
    r.tangent_dimension = nf - me;
    if (status != Status::Solved && r.certificate_equality_multipliers.empty()) certify();
    r.elapsed_seconds = seconds_since(t0);
  }
};

} // namespace

Result solve_lazy_newton(const QCQPProblem& problem, const Settings& st) {
  Result r;
  r.effective_outer_algorithm = "lazy-newton";
  r.lazy_dca_equality_model = "curvature-newton";
  r.linear_solver_backend = std::string("lazy-newton/suitesparse_ldl_") +
      ((st.ln_ordering == "amd") ? "amd" : "metis");
#if !defined(LAZYIPDCA_HAVE_AMD)
  r.linear_solver_backend="lazy-newton/suitesparse_ldl_metis";
#endif
  r.x = problem.x0;
  LazyNewton solver(problem, st, r);
  solver.t0 = Clock::now();
  try {
    const Status s = solver.run();
    solver.finish(s);
  } catch (const std::exception& e) {
    solver.finish(Status::NumericalError);
    r.message = std::string("lazy-newton: ") + e.what();
  }
  return r;
}

} // namespace lazyipdca::qcqp
