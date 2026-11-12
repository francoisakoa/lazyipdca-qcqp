// Internal controller compiled beside the shared reduced primal-dual Workspace.
// This file is intentionally included once by atomic_newton.cpp: numerical
// assembly, exact affine elimination, refinement and certificate code are shared.
namespace {
struct DcaPoint : detail::OriginalFilterPoint {
  double primal = 0, dual = 0, comp = 0;
  double current_lagrangian = 0;
  double guard_reference_kkt = INFINITY;
  bool feasible_reference_initialized = false;
};
double dca_kkt_guard_base(const DcaPoint &from) {
  return std::isfinite(from.guard_reference_kkt)
             ? std::min(from.kkt, from.guard_reference_kkt)
             : from.kkt;
}
double dca_kkt_ceiling(const Workspace &w, const DcaPoint &from) {
  return detail::dca_kkt_growth_ceiling(
      dca_kkt_guard_base(from), w.st.dca_kkt_growth_max,
      10.0 * w.st.acceptable_tolerance);
}
bool dca_kkt_allowed(const Workspace &w, const DcaPoint &from,
                     const DcaPoint &to,
                     bool *used_feasible_recovery = nullptr) {
  const bool below_persistent_ceiling = detail::dca_kkt_growth_allowed(
      dca_kkt_guard_base(from), to.kkt, w.st.dca_kkt_growth_max,
      10.0 * w.st.acceptable_tolerance);
  if (!below_persistent_ceiling) return false;
  const bool historical_tradeoff = detail::dca_local_candidate_not_dominated(
      from.kkt, from.guard_reference_kkt, from.primal, to.kkt, to.primal,
      1.0, w.st.dca_common_step_primal_trigger);
  const bool feasible_recovery = detail::dca_feasible_kkt_recovery_allowed(
      from.kkt, to.kkt, from.primal, to.primal,
      w.st.feasibility_tolerance);
  if (used_feasible_recovery)
    *used_feasible_recovery = feasible_recovery && !historical_tradeoff;
  return historical_tradeoff || feasible_recovery;
}
void dca_inherit_guard(const Workspace &w, const DcaPoint &from,
                       DcaPoint &to) {
  to.guard_reference_kkt = detail::dca_inherit_phase_kkt_reference(
      from.kkt, from.guard_reference_kkt, from.primal, to.kkt, to.primal,
      w.st.feasibility_tolerance, from.feasible_reference_initialized, 1.0,
      w.st.dca_common_step_primal_trigger);
  to.feasible_reference_initialized = from.feasible_reference_initialized ||
      to.primal <= w.st.feasibility_tolerance;
}
DcaPoint dca_original(Workspace &w, const Vec &x, const Vec &lambda, const Vec &z,
                      const Vec *anchor_lambda = nullptr, const Vec *anchor_z = nullptr) {
  std::optional<detail::DcaTimedScope> timer;
  if (w.dca_stats) {
    timer.emplace(w.dca_stats->original_point_seconds);
    ++w.dca_stats->original_point_calls;
    if (w.st.dca_aggregate_lean)
      w.dca_stats->original_point_value_reuses += w.me + w.mi;
  }
  DcaPoint p;
  p.objective = w.objective.original_at(x);
  p.lagrangian = p.objective;
  p.current_lagrangian = p.objective;
  const auto &fixed_lambda = anchor_lambda ? *anchor_lambda : lambda;
  const auto &fixed_z = anchor_z ? *anchor_z : z;
  Vec g(w.n, 0);
  long double y_l1=0, z_l1=0;
  // The internal norms visit the same rows, in the same order, as the
  // original-unit norms. Reuse each scalar value without retaining row arrays
  // or changing any gradient/normalization accumulation order.
  long double yi=0, zi=0;
  double pi=0;
  int native_inequalities=0;
  w.objective.original_gradient_at(x, 1, g);
  for (int e = 0; e < w.me; ++e) {
    auto &q = w.equalities[e];
    double h = q.original_at(x);
    p.lagrangian += fixed_lambda[e] * h;
    p.current_lagrangian += lambda[e] * h;
    p.theta += std::abs(h);
    p.primal = std::max(p.primal, std::abs(h) / q.scale);
    q.original_gradient_at(x, lambda[e], g);
    y_l1 += std::abs((long double)lambda[e] * q.scale / w.objective.scale);
    if (w.st.dca_aggregate_lean) {
      yi += std::abs(lambda[e]);
      pi = std::max(pi, std::abs(h));
    }
  }
  for (int i = 0; i < w.mi; ++i) {
    auto &q = w.inequalities[i];
    double h = q.original_at(x);
    if(w.inequality_kind[i]==0)++native_inequalities;
    if (w.active[i]) {
      const long double multiplier=std::abs((long double)z[i]*q.scale/w.objective.scale);
      z_l1+=multiplier;
      if(w.inequality_kind[i]==0)y_l1+=multiplier;
      p.lagrangian += fixed_z[i] * h;
      p.current_lagrangian += z[i] * h;
    }
    p.theta += std::max(0.0, h);
    p.primal = std::max(p.primal, std::max(0.0, h) / q.scale);
    if (w.active[i]) {
      q.original_gradient_at(x, z[i], g);
      p.comp = std::max(p.comp, std::abs(z[i] * h) / w.objective.scale);
    }
    if (w.st.dca_aggregate_lean) {
      pi = std::max(pi, w.inequality_kind[i]==0 ? std::max(0.0,h) : std::max(0.0,h)/q.scale);
      if (w.active[i]) {
        const double multiplier = w.inequality_kind[i]==0 ? z[i] : z[i]*q.scale;
        zi += std::abs(multiplier);
        if (w.inequality_kind[i]==0) yi += std::abs(multiplier);
      }
    }
  }
  p.dual = inf_norm(g) / w.objective.scale;
  p.kkt = std::max({p.primal, p.dual, p.comp});
  if (!finite(g) || !std::isfinite(p.objective) || !std::isfinite(p.theta))
    p.kkt = INFINITY;
  p.primal_inf = p.primal;
  const auto metrics=ipopt_desired_convergence_metrics(
      p.dual,p.primal,p.comp,double(y_l1),double(z_l1),
      w.me+native_inequalities+w.mi,w.mi,w.st);
  p.normalized_kkt=metrics.finite ? metrics.overall_nlp_error : INFINITY;
  p.within_acceptable_caps=metrics.finite &&
      p.normalized_kkt<=w.st.acceptable_tolerance &&
      p.primal<=w.st.acceptable_constraint_tolerance &&
      p.dual<=w.st.acceptable_dual_tolerance &&
      p.comp<=w.st.acceptable_complementarity_tolerance;
  {
    // Phase detection must use INTERNAL units for every termination profile.
    if (!w.st.dca_aggregate_lean) {
      for(int e=0;e<w.me;++e){yi+=std::abs(lambda[e]);pi=std::max(pi,std::abs(w.equalities[e].original_at(x)));}
      for(int i=0;i<w.mi;++i){
        const auto& q=w.inequalities[i];
        const double value=q.original_at(x);
        pi=std::max(pi,w.inequality_kind[i]==0?std::max(0.0,value):std::max(0.0,value)/q.scale);
        if(w.active[i]){
          const double multiplier=w.inequality_kind[i]==0?z[i]:z[i]*q.scale;
          zi+=std::abs(multiplier);if(w.inequality_kind[i]==0)yi+=std::abs(multiplier);
        }
      }
    }
    const auto m=ipopt_desired_convergence_metrics(p.dual*w.objective.scale,pi,p.comp*w.objective.scale,
        double(yi),double(zi),w.me+native_inequalities+w.mi,w.mi,w.st);
    p.normalized_kkt=m.finite?m.overall_nlp_error:INFINITY;
    // This only admits a candidate to the filter. The committed state is
    // subsequently audited with its ACTUAL transported slack complementarity.
    if(w.st.termination_test=="ipopt")p.within_acceptable_caps=m.finite&&m.overall_nlp_error<=w.st.acceptable_tolerance&&
        p.primal<=w.st.acceptable_constraint_tolerance&&p.dual<=w.st.acceptable_dual_tolerance&&
        p.comp<=w.st.acceptable_complementarity_tolerance;
  }
  p.distinct_from_anchor=p.within_acceptable_caps &&
      (x!=w.x || lambda!=w.lambda || z!=w.z);
  p.guard_reference_kkt = p.kkt;
  p.feasible_reference_initialized =
      p.primal <= w.st.feasibility_tolerance;
  return p;
}
#include "dca_precise_residual.hpp"
#include "dca_filter_diagnostics.hpp"
#include "dca_endpoint_start.hpp"
void dca_disable_models(Workspace &w) {
  w.objective.model_active = false;
  for (auto *group : {&w.equalities, &w.inequalities})
    for (auto &q : *group)
      q.model_active = false;
}
// Experimental, bounded acceleration of the accepted DC fixed-point map.
// This object never assembles a Hessian or solves a Newton system. Its small
// least-squares fit only mixes outputs of genuine DC steps. All proposals are
// audited below on the original problem, with the ordinary DC candidate kept
// intact as the fallback. Slacks have ORIGINAL-constraint coordinates here;
// model slacks from distinct DC centres must never be mixed directly.
struct DcaAnderson {
  struct Image { Vec x, lambda, z, original_s, residual; };
  std::vector<Image> history;
  Vec scaling;
  std::vector<unsigned char> active;
  void clear() { history.clear(); scaling.clear(); active.clear(); }
  bool propose(Workspace &w, const Vec &from_x, const Vec &from_lambda,
               const Vec &from_z, const Vec &to_x, const Vec &to_lambda,
               const Vec &to_z, const Vec &to_s, Image &proposal) {
    if (active != w.active) { clear(); active = w.active; }
    Image current;
    current.x = to_x; current.lambda = to_lambda; current.z = to_z;
    current.original_s = to_s;
    for (int i = 0; i < w.mi; ++i) if (w.active[i])
      current.original_s[i] += std::max(0.0, w.inequalities[i].at(to_x) -
                                                   w.inequalities[i].original_at(to_x));
    const int dimension = w.n + w.me + w.mi;
    const bool initial = scaling.empty();
    if (initial) scaling.resize(dimension);
    current.residual.resize(dimension);
    int offset = 0;
    auto append = [&](const Vec &from, const Vec &to, bool inequalities) {
      const double family_scale = 1.0 / std::sqrt(std::max<std::size_t>(1, from.size()));
      for (int j = 0; j < (int)from.size(); ++j, ++offset) {
        if (initial)
          scaling[offset] = family_scale /
              std::max({1.0, std::abs(from[j]), std::abs(to[j])});
        current.residual[offset] = inequalities && !w.active[j] ? 0.0 :
            scaling[offset] * (to[j] - from[j]);
      }
    };
    append(from_x, to_x, false); append(from_lambda, to_lambda, false);
    append(from_z, to_z, true);
    if (!finite(current.residual) || !finite(current.original_s)) { clear(); return false; }
    history.push_back(std::move(current));
    constexpr int depth = 4;
    if (history.size() > depth) history.erase(history.begin());
    const int count = history.size();
    if (count < 2) return false;
    double gram[depth][depth] = {}, lower[depth][depth] = {}, trace = 0;
    for (int i = 0; i < count; ++i) {
      for (int j = 0; j <= i; ++j)
        gram[i][j] = gram[j][i] = dot(history[i].residual, history[j].residual);
      trace += gram[i][i];
    }
    if (!(trace > 1e-300) || !std::isfinite(trace)) return false;
    const double ridge = 1e-8 * trace / count;
    for (int i = 0; i < count; ++i)
      for (int j = 0; j <= i; ++j) {
        double value = gram[i][j] + (i == j ? ridge : 0);
        for (int k = 0; k < j; ++k) value -= lower[i][k] * lower[j][k];
        if (i == j) {
          if (!(value > 0) || !std::isfinite(value)) return false;
          lower[i][j] = std::sqrt(value);
        } else lower[i][j] = value / lower[j][j];
      }
    double forward[depth] = {}, alpha[depth] = {}, denominator = 0;
    for (int i = 0; i < count; ++i) {
      double value = 1;
      for (int j = 0; j < i; ++j) value -= lower[i][j] * forward[j];
      forward[i] = value / lower[i][i];
    }
    for (int i = count - 1; i >= 0; --i) {
      double value = forward[i];
      for (int j = i + 1; j < count; ++j) value -= lower[j][i] * alpha[j];
      alpha[i] = value / lower[i][i]; denominator += alpha[i];
    }
    if (!(denominator > 0) || !std::isfinite(denominator)) return false;
    double coefficient_norm = 0, predicted_norm2 = 0;
    for (int i = 0; i < count; ++i) { alpha[i] /= denominator; coefficient_norm += std::abs(alpha[i]); }
    if (!std::isfinite(coefficient_norm) || coefficient_norm > 8.0) return false;
    for (int i = 0; i < count; ++i)
      for (int j = 0; j < count; ++j) predicted_norm2 += alpha[i] * alpha[j] * gram[i][j];
    if (!(predicted_norm2 <= 0.8 * gram[count - 1][count - 1])) return false;
    proposal.x.assign(w.n, 0); proposal.lambda.assign(w.me, 0);
    proposal.z.assign(w.mi, 0); proposal.original_s.assign(w.mi, 0);
    for (int i = 0; i < count; ++i) {
      axpy(alpha[i], history[i].x, proposal.x);
      axpy(alpha[i], history[i].lambda, proposal.lambda);
      axpy(alpha[i], history[i].z, proposal.z);
      axpy(alpha[i], history[i].original_s, proposal.original_s);
    }
    return finite(proposal.x) && finite(proposal.lambda) && finite(proposal.z) && finite(proposal.original_s);
  }
};
struct DcaModels {
  Workspace &w;
  DcaFilterStatistics &stats;
  detail::DcaCurvature lagrangian;
  std::vector<std::unique_ptr<detail::DcaCurvature>> inequality;
  std::map<Pair, int> slots;
  Vec center;
  struct InputCurvature { int equality; double scale, value; };
  struct ReducedScatter { int input, output, a, b; double multiplier; };
  std::vector<InputCurvature> curvature_input;
  std::vector<SymTriplet> original_b, canonical_b, reduced_b;
  std::vector<int> original_scatter, objective_diagonal;
  std::vector<ReducedScatter> reduced_scatter;
  std::vector<std::vector<std::pair<int, double>>> normals;
  int remaining;
  bool layout_ready = false;
  DcaModels(Workspace &work, DcaFilterStatistics &s)
      : w(work), stats(s), lagrangian(w.st),
        remaining(w.st.dca_filter_max_atoms) {
    CostScope layout(w.costs, NewtonCost::Preparation);
    for (int i = 0; i < (int)w.hpattern.size(); ++i)
      slots[w.hpattern[i]] = i;
    if (!(w.st.dca_aggregate_only && w.st.dca_aggregate_lean)) ensure_layout();
  }
  void ensure_layout() {
    if (layout_ready) return;
    CostScope layout(w.costs, NewtonCost::Preparation);
    detail::DcaTimedScope timer(stats.model_layout_seconds);
    ++stats.model_layout_builds;
    inequality.resize(w.mi);
    center = w.x;
    auto append = [&](const Quadratic &q, int equality) {
      for (auto e : q.hessian) {
        original_b.push_back({e.i, e.j, 0});
        curvature_input.push_back({equality, q.scale, e.v});
      }
    };
    append(w.objective, -1);
    for (int e = 0; e < w.me; ++e) append(w.equalities[e], e);
    canonical_b = detail::DcaCurvature::canonical(w.n, original_b);
    std::map<Pair, int> original_slots, reduced_slots;
    for (int k = 0; k < (int)canonical_b.size(); ++k)
      original_slots[{canonical_b[k].i, canonical_b[k].j}] = k;
    for (auto e : original_b) original_scatter.push_back(original_slots.at(upper(e.i, e.j)));
    for (int j : w.retained) objective_diagonal.push_back(original_slots.at({j, j}));
    for (int j = 0; j < w.nr; ++j) reduced_slots[{j, j}] = 0;
    for (auto e : original_b)
      for (int a = w.tbegin[e.i]; a < w.tbegin[e.i + 1]; ++a)
        for (int b = w.tbegin[e.j]; b < w.tbegin[e.j + 1]; ++b) {
          if (e.i == e.j && a > b) continue;
          reduced_slots[upper(w.tcolumn[a], w.tcolumn[b])] = 0;
        }
    for (auto &e : reduced_slots) {
      e.second = reduced_b.size(); reduced_b.push_back({e.first.first, e.first.second, 0});
    }
    for (int k = 0; k < (int)original_b.size(); ++k) {
      const auto e = original_b[k];
      for (int a = w.tbegin[e.i]; a < w.tbegin[e.i + 1]; ++a)
        for (int b = w.tbegin[e.j]; b < w.tbegin[e.j + 1]; ++b) {
          if (e.i == e.j && a > b) continue;
          const auto ij = upper(w.tcolumn[a], w.tcolumn[b]);
          reduced_scatter.push_back({k, reduced_slots.at(ij), a, b,
                                   e.i != e.j && ij.first == ij.second ? 2.0 : 1.0});
        }
    }
    for (const auto &row : w.ar) {
      normals.emplace_back();
      for (int j : row.support) normals.back().push_back({j, 0});
    }
    layout_ready = true;
  }
  void build(double extra_proximal = 0, bool restoration = false) {
    // A future caller that needs a majorant after a lean aggregate start gets
    // the complete historical layout before touching any of its arrays.
    ensure_layout();
    const auto started = Clock::now();
    dca_disable_models(w);
    center = w.x;
    w.evaluate();
    auto stamp = Clock::now();
    for (auto &e : canonical_b) e.v = 0;
    for (std::size_t k = 0; k < curvature_input.size(); ++k) {
      const auto t = curvature_input[k];
      original_b[k].v = t.equality < 0 ? t.scale * t.value : w.lambda[t.equality] * t.scale * t.value;
      canonical_b[original_scatter[k]].v += original_b[k].v;
    }
    const bool event = w.st.dca_filter_atom_events &&
                       ((stats.hessian_updates + 1) % w.st.dca_filter_atom_every == 0) &&
                       (!restoration || stats.models == 0);
    const int before = lagrangian.updates;
    int admission_budget = event ? std::min(remaining, w.st.dca_filter_atom_batch) : 0;
    // Exact affine elimination gives d=T*z+P*(-h_E), with E*T=I
    // for the retained-coordinate selector E. Curvature is certified in z.
    // The PSD correction is lifted as E'*(H_r+rho*I)*E, never as T*H_r*T'.
    Vec transform = w.tvalue;
    for (std::size_t t = 0; t < transform.size(); ++t) {
      auto [e, j] = w.tgradient[t];
      if (e >= 0)
        transform[t] = -w.equalities[e].gradient[j] / w.pivot_coefficient[e];
    }
    for (auto &e : reduced_b) e.v = 0;
    for (const auto &t : reduced_scatter)
      reduced_b[t.output].v += t.multiplier * original_b[t.input].v * transform[t.a] * transform[t.b];
    for (int i = 0; i < (int)w.ar.size(); ++i) {
      for (auto &entry : normals[i]) entry.second = 0;
      const auto &q = w.equalities[w.eq_retained[i]];
      for (auto t : w.ar[i].map)
        normals[i][t.output].second += q.gradient[t.gradient] * transform[t.t];
      // Positive row scaling preserves the affine tangent space. It gives
      // every equality normal a comparable completion strength and avoids
      // certifying a badly scaled augmented matrix just to ignore its normals.
      double length = 0;
      for (auto entry : normals[i]) length = std::hypot(length, entry.second);
      if (!std::isfinite(length)) throw std::runtime_error("nonfinite reduced equality normal");
      if (length > 0) for (auto &entry : normals[i]) entry.second /= length;
    }
    if (!restoration)
      lagrangian.ensure(w.nr, reduced_b, 1e-10, event, admission_budget, w.st.dca_filter_atom_batch,
                        w.st.dca_filter_atom_gain, &normals);
    const int curvature_peak = (restoration ? int(lagrangian.atoms.size() + lagrangian.stored_spectral_atoms())
                                           : lagrangian.last_peak_stored_atoms) + stats.atoms_inequalities;
    if (lagrangian.updates != before || (restoration && stats.models == 0)) {
      ++stats.hessian_updates;
      if (event)
        ++stats.atom_events;
    }
    auto obj = canonical_b;
    if (lagrangian.atoms.empty()) {
      for (int k : objective_diagonal) obj[k].v += lagrangian.rho + extra_proximal;
    } else {
      for (auto e : lagrangian.corrected(w.nr, {}, lagrangian.rho + extra_proximal))
        obj.push_back({w.retained[e.i], w.retained[e.j], e.v});
      obj = detail::DcaCurvature::canonical(w.n, obj);
    }
    // A sufficient full-space normal completion follows by a Schur bound in
    // coordinates (z, A_E*d). It is NOT assembled: both normal penalties are
    // constants on the complete affine equality slice. This avoids large
    // normal-only curvature in every optimization direction.
    long double pnorm2 = 0, tnorm2 = 0, bnorm2 = 0;
    for(double value:transform)tnorm2+=static_cast<long double>(value)*value;
    for (int e = 0; e < w.me; ++e)
      if (w.pivot_variable[e] >= 0) {
        const long double coefficient=w.pivot_coefficient[e];
        if(coefficient!=0)
          pnorm2=std::max(pnorm2,1/(coefficient*coefficient));
      }
    for (auto e : canonical_b)
      bnorm2+=(e.i==e.j?1.L:2.L)*static_cast<long double>(e.v)*e.v;
    // This is a diagnostic bound only; it must never stop a numerical
    // transaction.  In particular restoration discards the model below, so
    // avoid evaluating a potentially enormous, irrelevant expression first.
    stats.eliminated_normal_completion_weight=0;
    if(!restoration&&!lagrangian.tangent_certificate&&pnorm2>0) {
      const long double value=pnorm2*(std::sqrt(bnorm2)+tnorm2*bnorm2/1e-10L)+1e-10L;
      const long double cap=std::numeric_limits<double>::max();
      stats.eliminated_normal_completion_weight=
          std::isfinite(value)&&value<cap?static_cast<double>(value):static_cast<double>(cap);
    }
    stats.tangent_certificate = lagrangian.tangent_certificate;
    stats.certificate_transport_trials = lagrangian.certificate_transport_trials;
    stats.certificate_transports = lagrangian.certificate_transports;
    if (restoration) {
      obj.clear();
      for (int j = 0; j < w.n; ++j)
        obj.push_back({j, j, 1.0});
    }
    stats.rho_inequalities_max = 0;
    for (int i = 0; i < w.mi; ++i)
      if (w.active[i]) {
        auto &q = w.inequalities[i];
        std::vector<SymTriplet> hi;
        if (!q.hessian.empty() && !inequality[i]) {
          std::set<int> support;
          for (auto e : q.hessian) {
            support.insert(e.i);
            support.insert(e.j);
          }
          std::vector<int> indices(support.begin(), support.end());
          auto local = [&](int j) {
            return int(std::lower_bound(indices.begin(), indices.end(), j) - indices.begin());
          };
          for (auto e : q.hessian)
            hi.push_back({local(e.i), local(e.j), q.scale * e.v});
          if (!inequality[i])
            inequality[i] = std::make_unique<detail::DcaCurvature>(w.st);
          auto &c = *inequality[i];
          c.ensure(indices.size(), hi, 0, w.st.dca_filter_atom_events, admission_budget,
                   w.st.dca_filter_atom_batch, w.st.dca_filter_atom_gain);
          hi = c.corrected(indices.size(), hi, c.rho);
          for (auto &e : hi) {
            e.i = indices[e.i];
            e.j = indices[e.j];
          }
          stats.rho_inequalities_max = std::max(stats.rho_inequalities_max, c.rho);
        }
        if (inequality[i] && hi.empty()) {
          ++inequality[i]->reuses;
          q.model_center = &center;
          q.model_constant = q.value;
          q.model_gradient = q.gradient;
          q.model_active = true;
          stats.rho_inequalities_max = std::max(stats.rho_inequalities_max, inequality[i]->rho);
        } else
          q.set_model(center, std::move(hi), slots);
      }
    stats.normal_completion_weight = restoration || lagrangian.tangent_certificate ? 0 : lagrangian.normal_weight;
    stats.curvature_seconds += seconds(stamp);
    w.objective.set_model(center, std::move(obj), slots);
    w.objective.model_atoms.clear();
    w.objective.model_atom_support.clear();
    if (w.st.dca_filter_packed_atoms) w.objective.model_atom_support = w.retained;
    if (!restoration)
      for (const auto &a : lagrangian.spectral_atoms) {
        if (!(a.beta > 0))
          continue;
        LowRankTerm lifted;
        lifted.beta = a.beta;
        lifted.direction.assign(w.n, 0);
        for (int j = 0; j < w.nr; ++j)
          lifted.direction[w.retained[j]] = a.direction[j];
        w.objective.model_atoms.push_back(std::move(lifted));
      }
    if (restoration) {
      w.objective.model_constant = 0;
      std::fill(w.objective.model_gradient.begin(), w.objective.model_gradient.end(), 0);
    }
    for (auto &q : w.equalities)
      q.set_model(center, {}, slots);
    ++stats.models;
    stats.rho_lagrangian = restoration ? 1 : lagrangian.rho + extra_proximal;
    stats.atoms_lagrangian = lagrangian.atoms.size() + lagrangian.stored_spectral_atoms();
    stats.atoms_inequalities = 0;
    stats.curvature_factors = lagrangian.certificates;
    stats.curvature_reuses = lagrangian.reuses;
    for (const auto &c : inequality)
      if (c) {
        stats.atoms_inequalities += c->atoms.size();
        stats.curvature_factors += c->certificates;
        stats.curvature_reuses += c->reuses;
      }
    remaining = w.st.dca_filter_max_atoms - stats.atoms_lagrangian - stats.atoms_inequalities;
    if (remaining < 0) throw std::runtime_error("global DCA stored atom budget exceeded");
    stats.peak_stored_atoms = std::max({stats.peak_stored_atoms, curvature_peak,
                                      stats.atoms_lagrangian + stats.atoms_inequalities});
    stats.basis_compressions = lagrangian.basis_compressions;
    stats.atom_basis_keep = w.st.dca_atom_basis_keep;
    stats.atom_basis_tail = w.st.dca_atom_basis_tail;
    stats.atom_probe_depth = w.st.dca_atom_probe_depth;
    stats.atom_probe_count_projection = w.st.dca_atom_probe_count_projection;
    stats.retired_directions = lagrangian.retired_directions;
    stats.cumulative_atoms_lagrangian = lagrangian.additions;
    stats.spectral_inverse_solves = lagrangian.inverse_probes;
    stats.small_block_spectra = 0;
    stats.atoms_active_lagrangian = 0;
    stats.atoms_active_inequalities = 0;
    if (!restoration) {
      for (auto a : lagrangian.atoms)
        stats.atoms_active_lagrangian += a.beta > 0;
      for (const auto &a : lagrangian.spectral_atoms)
        stats.atoms_active_lagrangian += a.beta > 0;
    }
    for (const auto &c : inequality)
      if (c) {
        stats.small_block_spectra += c->small_spectra;
        for (auto a : c->atoms)
          stats.atoms_active_inequalities += a.beta > 0;
      }
    stats.rho_refinement_trials = lagrangian.rho_refinement_trials;
    stats.rho_refinement_acceptances = lagrangian.rho_refinement_acceptances;
    stats.curvature_factor_reuses = lagrangian.factor_reuses;
    stats.subspace_spectra = lagrangian.subspace_spectra;
    stats.dependent_witnesses = lagrangian.dependent_witnesses;
    stats.curvature_ritz_shift_proposals = lagrangian.ritz_shift_proposals;
    stats.curvature_gershgorin_fallbacks = lagrangian.gershgorin_fallback_uses;
    stats.curvature_ensure_seconds = lagrangian.ensure_seconds;
    stats.curvature_factor_seconds = lagrangian.factor_seconds;
    stats.curvature_inverse_seconds = lagrangian.inverse_seconds;
    stats.subspace_seconds = lagrangian.subspace_seconds;
    for (const auto &c : inequality) if (c) {
      stats.curvature_factor_reuses += c->factor_reuses;
      stats.curvature_ensure_seconds += c->ensure_seconds;
      stats.curvature_factor_seconds += c->factor_seconds;
      stats.curvature_inverse_seconds += c->inverse_seconds;
      stats.curvature_ritz_shift_proposals += c->ritz_shift_proposals;
      stats.curvature_gershgorin_fallbacks += c->gershgorin_fallback_uses;
    }
    stats.model_build_seconds += seconds(started);
  }
  void transport_slacks() {
    // Preserve g+s exactly across recentering. The correction is PSD, hence
    // the original slack is at least the positive model slack (up to roundoff).
    for (int i = 0; i < w.mi; ++i)
      if (w.active[i]) {
        auto &q = w.inequalities[i];
        const double delta = q.at(w.x) - q.original_at(w.x);
        w.s[i] += std::max(0.0, delta);
      }
  }
};
#include "dca_local_active.hpp"
#include "dca_aggregate.hpp"
#include "dca_krylov.hpp"
#include "dca_strict_finish.hpp"
void dca_display(const Workspace &w, const Result &r, const DcaPoint &p, int level, int inner,
                 double ap, double ad, int ls, double elapsed, int rebuild = -1) {
  const int k = r.outer_iterations;
  if (level == 0)
    return;
  if (k % w.st.live_iteration_header_every == 0) {
    std::cout << "\n iter      objective     inf_pr     inf_du   lg(mu)      ||d||   alpha_du   "
                 "alpha_pr   ls | inner/max rebuild/max";
    if (level >= 2)
      std::cout << " | aOE rOE<=      rhoOE   aI rI<=       rhoI    WS";
    if (level >= 3)
      std::cout << " |  mdl age nInt  nFac   curvFac   time(s)";
    std::cout << '\n';
  }
  double mu = 0;
  int count = 0;
  for (int i = 0; i < w.mi; ++i)
    if (w.active[i]) {
      mu += w.s[i] * w.z[i];
      ++count;
    }
  mu = count ? mu / count : 0;
  std::cout << std::setw(5) << k << ' ' << std::scientific << std::setprecision(7) << std::setw(15)
            << p.objective / w.objective.scale << std::setprecision(2) << ' ' << std::setw(10)
            << p.primal << ' ' << std::setw(10) << p.dual << std::fixed << std::setprecision(1)
            << ' ' << std::setw(8) << std::log10(std::max(mu, 1e-300)) << std::scientific
            << std::setprecision(2) << ' ' << std::setw(10) << r.step_inf << ' ' << std::setw(10)
            << ad << ' ' << std::setw(10) << ap << ' ' << std::setw(4) << ls
            << " | inner=";
  if (rebuild < 0) std::cout << '-';
  else std::cout << r.dca_filter.current_model_steps;
  std::cout << '/' << w.st.dca_filter_inner_max << " rebuild=";
  if (rebuild < 0) std::cout << '-';
  else std::cout << rebuild;
  std::cout << '/' << w.st.dca_filter_model_retries;
  auto &d = r.dca_filter;
  if (level >= 2)
    std::cout << " | " << std::setw(3) << d.atoms_lagrangian << ' ' << std::setw(5)
              << std::min(w.nr, d.atoms_active_lagrangian) << ' ' << std::setw(10)
              << d.rho_lagrangian << ' ' << std::setw(4) << d.atoms_inequalities << ' '
              << std::setw(4) << d.atoms_active_inequalities << ' ' << std::setw(10)
              << d.rho_inequalities_max << ' ' << std::setw(5) << count;
  if (level >= 3)
    std::cout << " | " << std::setw(4) << d.models << ' ' << std::setw(3) << d.current_model_steps
              << ' ' << std::setw(4) << inner << ' ' << std::setw(5)
              << r.lazy_newton.factor_attempts << ' ' << std::setw(9) << d.curvature_factors
              << std::fixed << std::setprecision(3) << ' ' << std::setw(9) << elapsed;
  std::cout << std::defaultfloat << std::endl;
}
// Completed transactions only: factor/line failures do not consume an inner
// step in this live counter. Rebuild zero is the initial model attempt.
void dca_budget(const Workspace &w, int level, int outer, const char *phase,
                int inner, int rebuild, const char *event) {
  if (level < 1) return;
  std::cout << "[DCA-BUDGET] outer=" << outer << " phase=" << phase
            << " inner=";
  if (inner < 0) std::cout << '-';
  else std::cout << inner;
  std::cout << '/' << w.st.dca_filter_inner_max << " rebuild=";
  if (rebuild < 0) std::cout << '-';
  else std::cout << rebuild;
  std::cout << '/' << w.st.dca_filter_model_retries << " event=" << event << std::endl;
}
#include "dca_inequality_majorant_models.hpp"
#include "dca_majorant_filter_path.hpp"
#include "dca_majorant_barrier_path.hpp"
#include "dca_inequality_majorant_engine.hpp"
} // namespace

Result solve_filter_dca(const QCQPProblem &problem, const Settings &requested) {
  if(requested.dca_engine=="majorants")
    return solve_inequality_majorants(problem,requested);
  const Settings stop_st = requested;
  Settings st = requested;
  if(st.termination_test=="ipopt") {
    if(!(st.dca_control_tolerance>0)||!std::isfinite(st.dca_control_tolerance))
      throw std::invalid_argument("DCA control tolerance must be finite and positive");
    st.stationarity_tolerance=st.feasibility_tolerance=
        st.dual_infeasibility_tolerance=st.complementarity_tolerance=st.dca_control_tolerance;
  }
  if (st.dca_filter_inner_max < 1 || st.dca_filter_inner_max > 100 || st.dca_filter_max_atoms < 0 ||
      st.dca_filter_atom_every < 1 || st.dca_filter_atom_batch < 1 ||
      st.dca_filter_model_retries < 0)
    throw std::invalid_argument("invalid DCA step/atom/retry limits");
  if(!std::isfinite(st.dca_aggregate_rho_max)||st.dca_aggregate_rho_max<0||
     !(st.dca_aggregate_primal_trigger>0)||
     (st.dca_aggregate_backend!="auto"&&st.dca_aggregate_backend!="cholesk"&&st.dca_aggregate_backend!="mumps"&&st.dca_aggregate_backend!="ldl"&&st.dca_aggregate_backend!="qdldl"&&st.dca_aggregate_backend!="native"&&st.dca_aggregate_backend!="ma27"))
    throw std::runtime_error("invalid aggregate DCA settings");
  if(st.dca_krylov_max_iterations<0||st.dca_krylov_max_iterations>128||st.dca_krylov_failure_cooldown<0)
    throw std::invalid_argument("invalid DCA Krylov budget/cooldown");
  if(!std::isfinite(st.dca_kkt_growth_max)||st.dca_kkt_growth_max<1.0||
     st.dca_aggregate_progress_window<0||st.dca_aggregate_progress_window>1000)
    throw std::invalid_argument("DCA KKT growth must be finite and >=1; progress window must be in [0,1000]");
  if(st.dca_strict_finish_max<0 || st.dca_strict_finish_max>100 ||
     st.dca_strict_finish_refinements<0 || st.dca_strict_finish_refinements>100 ||
     !(st.dca_strict_finish_eta>0&&st.dca_strict_finish_eta<1) || !std::isfinite(st.dca_strict_finish_eta) ||
     !(st.dca_strict_finish_mu_fraction>0&&st.dca_strict_finish_mu_fraction<1) || !std::isfinite(st.dca_strict_finish_mu_fraction))
    throw std::invalid_argument("strict finish budgets must be in [0,100] and eta/mu-fraction in (0,1)");
  if(st.dca_aggregate_retry_max<1 || st.dca_aggregate_retry_max>1000 ||
     !std::isfinite(st.dca_aggregate_rho_initial) || st.dca_aggregate_rho_initial<=0 ||
     !std::isfinite(st.dca_aggregate_rho_growth) || st.dca_aggregate_rho_growth<=1 ||
     !std::isfinite(st.dca_aggregate_rho_decay) || st.dca_aggregate_rho_decay<=0 || st.dca_aggregate_rho_decay>1)
    throw std::invalid_argument("invalid aggregate-only regularization budgets/factors");
  if(st.dca_aggregate_only && (!st.dca_aggregate || st.dca_filter_initial_restoration ||
      st.dca_local_active || st.dca_krylov || st.dca_aggregate_progress_window>0 ||
      std::isfinite(st.dca_aggregate_primal_trigger)))
    throw std::invalid_argument("aggregate-only requires aggregate, infinite primal trigger, and no initial/local/private recovery");
  const int level =
      st.live_display_level < 0 ? (st.live_iterations ? 1 : 0) : st.live_display_level;
  if (level < 0 || level > 4 || st.live_iteration_header_every < 1 || st.live_iteration_every < 1)
    throw std::invalid_argument("invalid DCA display level/period");
  if (st.dca_filter_rho_refinements < 0 || st.dca_filter_rho_refinements > 12)
    throw std::invalid_argument("DCA rho refinements must be in [0,12]");
  st.dca_engine = "filter";
  st.lazy_newton_atomic_regularization = false;
  st.verbose = false;
  st.live_curvature = level >= 4;
  st.lazy_newton_predictor_corrector = true;
  problem.validate();
  Result result;
  if (st.dca_filter_curvature_backend != "same" &&
      !linear_backend_compiled(st.dca_filter_curvature_backend))
    throw std::invalid_argument("requested DCA curvature backend is not compiled");
  detail::validate_acceptable(st);
  result.acceptable_required=st.acceptable_iterations;
  result.x = problem.x0;
  result.effective_outer_algorithm = "lazy-dca";
  result.lazy_dca_equality_model = "affine-partial-lagrangian-dc";
  result.linear_solver_backend = "filter-dca/" + st.linear_backend;
  result.dca_dual_step = st.dca_dual_step;
  result.dca_filter.used = true;
  result.dca_filter.kkt_growth_max=st.dca_kkt_growth_max;
  result.dca_filter.progress_window=st.dca_aggregate_progress_window;
  result.dca_filter.rho_complete_search_enabled=st.dca_rho_complete_search;
  result.dca_filter.strict_finish_enabled=st.dca_strict_finish;
  result.dca_filter.strict_finish_budget=st.dca_strict_finish_max;
  result.dca_filter.strict_finish_refinement_budget=st.dca_strict_finish_refinements;
  result.dca_filter.strict_finish_eta=st.dca_strict_finish_eta;
  result.dca_filter.strict_finish_mu_fraction=st.dca_strict_finish_mu_fraction;
  result.dca_filter.aggregate_enabled=st.dca_aggregate;
  result.dca_filter.aggregate_only_enabled=st.dca_aggregate_only;
  result.dca_filter.aggregate_affine_retry_enabled=st.dca_aggregate_affine_retry;
  result.dca_filter.aggregate_only_retry_budget=st.dca_aggregate_retry_max;
  result.dca_filter.aggregate_only_rho_initial=st.dca_aggregate_rho_initial;
  result.dca_filter.aggregate_only_rho_growth=st.dca_aggregate_rho_growth;
  result.dca_filter.aggregate_only_rho_decay=st.dca_aggregate_rho_decay;
  result.dca_filter.aggregate_only_rho_cap=st.dca_aggregate_rho_max;
  result.dca_filter.aggregate_preparation_cache_enabled=st.dca_aggregate_preparation_cache;
  result.dca_filter.aggregate_strict_probe_stop_enabled=st.dca_aggregate_strict_probe_stop;
  result.dca_filter.aggregate_kkt_layout=st.dca_aggregate_kkt_layout;
  result.dca_filter.aggregate_border_reserve=st.dca_aggregate_border_reserve;
  result.dca_filter.aggregate_rejection_policy=st.dca_aggregate_rejection_policy;
  result.dca_filter.aggregate_fast_assembly=st.dca_aggregate_fast_assembly;
  result.dca_filter.aggregate_rho_memory=st.dca_aggregate_rho_memory;
  result.dca_filter.aggregate_lean=st.dca_aggregate_lean;
  result.dca_filter.atom_basis_keep=st.dca_atom_basis_keep;
  result.dca_filter.atom_basis_tail=st.dca_atom_basis_tail;
  result.dca_filter.atom_probe_depth=st.dca_atom_probe_depth;
  result.dca_filter.atom_probe_count_projection=st.dca_atom_probe_count_projection;
  result.dca_filter.krylov_enabled=st.dca_krylov||st.dca_aggregate_progress_window>0||st.dca_aggregate;
  result.dca_filter.krylov_budget=st.dca_krylov_max_iterations;
  result.dca_filter.krylov_cooldown=st.dca_krylov_failure_cooldown;
  result.dca_filter.krylov_predictor_corrector=st.dca_krylov_predictor_corrector;
  result.dca_filter.initial_restoration_requested = st.dca_filter_initial_restoration;
  result.dca_filter.normal_atom_weighting = st.dca_filter_normal_atom_weighting;
  result.dca_filter.rho_refinement_budget = st.dca_filter_rho_refinements;
  result.dca_filter.active_evaluation = st.dca_filter_active_evaluation;
  result.dca_filter.packed_atoms = st.dca_filter_packed_atoms;
  result.dca_filter.lazy_merit = st.dca_filter_lazy_merit;
  result.dca_filter.block_ritz = st.dca_filter_block_ritz;
  result.dca_filter.anderson = st.dca_filter_anderson;
  result.dca_filter.certificate_transport = st.dca_filter_certificate_transport;
  result.dca_filter.sparse_fallback = st.dca_filter_sparse_fallback;
  result.dca_filter.equilibration = st.kkt_equilibration;
  result.dca_filter.curvature_backend = st.dca_filter_curvature_backend == "same" ? st.linear_backend : st.dca_filter_curvature_backend;
  result.lazy_newton.display_level = level;
  auto start = Clock::now();
  detail::NewtonCostLedger costs(result.lazy_newton, start);
  std::unique_ptr<Workspace> owner;
  Vec committed_x, committed_s, committed_z, committed_lambda;
  bool have_committed_state = false;
  Vec acceptable_x,acceptable_s,acceptable_z,acceptable_lambda;
  bool have_acceptable_state=false;
  double best_acceptable_error=INFINITY;
  int saved_acceptable_streak=0;
  try {
    bool all_fixed = true;
    for (int j = 0; j < problem.n; ++j)
      all_fixed = all_fixed && std::isfinite(problem.lb[j]) && problem.lb[j] == problem.ub[j];
    if (all_fixed) {
      result.x = problem.lb;
      result.certificate_equality_multipliers.assign(problem.equalities.size(), 0);
      result.certificate_inequality_multipliers.assign(problem.inequalities.size(), 0);
      result.certificate_lower_multipliers.assign(problem.n, 0);
      result.certificate_upper_multipliers.assign(problem.n, 0);
      if(st.termination_test=="ipopt") {
        result.termination_scaling_available=true;
        result.termination_objective_scale=1;
        result.termination_equality_scales.assign(problem.equalities.size(),1);
        result.termination_inequality_scales.assign(problem.inequalities.size(),1);
        result.termination_inequality_slacks.assign(problem.inequalities.size(),0);
      }
      bool ok;
      {
        CostScope certificate(costs, NewtonCost::Certificate);
        ok = detail::audit_original_termination(problem, stop_st, result);
      }
      result.status = ok ? Status::Solved : Status::RestorationFailure;
      result.message = ok ? "fixed point satisfies the original QCQP certificate"
                          : "fixed point violates original constraints";
      result.termination_multiplier_source = "filter-dca-original-extended-fixed-reconstructed";
      costs.finish();
      result.elapsed_seconds = seconds(start);
      return result;
    }
    {
      CostScope preparation(costs, NewtonCost::Preparation);
      owner = std::make_unique<Workspace>(problem, st, result.lazy_newton, costs);
    }
    auto &w = *owner;
    auto &stats = result.dca_filter;
    w.dca_stats = &stats;
    // A small feasibility defect does not justify a terminal barrier when
    // the cold start has a large dual residual. Center the INITIAL state only.
    const double initial_mu =
        std::max(st.lazy_newton_initial_mu, std::min(1.0, 0.01 * inf_norm(w.rd)));
    for (int i = 0; i < w.mi; ++i)
      if (w.active[i])
        w.z[i] = initial_mu / w.s[i];
    if(!st.dca_endpoint_start_file.empty())dca_endpoint_start(w,st.dca_endpoint_start_file);
    w.evaluate();
    DcaModels models(w, stats);
    DcaLocalActive local_active(st);
    DcaAggregate aggregate(st,&stop_st);
#ifdef G10_AGGREGATE_TEST_CONFIGURE
    G10_AGGREGATE_TEST_CONFIGURE(aggregate);
#endif
    DcaKrylov krylov;
    auto outer = dca_original(w, w.x, w.lambda, w.z);
    detail::DcaAcceptedProgress progress(st.dca_aggregate_progress_window);
    progress.observe(outer.kkt,outer.primal,outer.dual,false,
                     st.feasibility_tolerance,st.dual_infeasibility_tolerance);
    detail::OriginalDcaFilter filter(std::max(1.0, 10.0 * outer.theta),
                                     st.stationarity_tolerance);
    bool corrector_enabled = true;
    // The original filter globalizes the first DC majorant too. Restoration
    // remains the bounded-retry fallback; an initial phase is opt-in.
    bool phase_one = st.dca_filter_initial_restoration && outer.primal > 1e-2;
    bool model_ready = false, reuse_model = false;
    int model_steps = 0, completed_model_steps = 0;
    Vec previous_displacement;
    const bool anderson_enabled = st.dca_filter_anderson;
    DcaAnderson anderson;
    bool force_progress_krylov=false;
    int aggregate_progress_cooldown=0;
    auto observe_accepted_progress=[&](bool aggregate_path) {
      const auto decision=progress.observe(
          outer.kkt,outer.primal,outer.dual,aggregate_path,
          st.feasibility_tolerance,st.dual_infeasibility_tolerance);
      if(!decision.aggregate_stall&&!decision.near_feasible_dual_stall)return;
      ++stats.progress_events;
      if(decision.aggregate_stall){
        ++stats.aggregate_progress_stalls;
        aggregate_progress_cooldown=std::max(
            aggregate_progress_cooldown,st.dca_aggregate_progress_window);
        stats.progress_last_event="aggregate-stall";
      }
      if(decision.near_feasible_dual_stall){
        ++stats.dual_progress_stalls;
        force_progress_krylov=true;
        stats.progress_last_event="near-feasible-dual-stall";
      }
      model_ready=reuse_model=false;
      dca_disable_models(w);
      anderson.clear();previous_displacement.clear();
      if(level)std::cout<<"[DCA-PROGRESS] outer="<<result.outer_iterations
        <<" event="<<stats.progress_last_event
        <<" kkt="<<outer.kkt<<" dual="<<outer.dual
        <<" aggregate_reference="<<decision.aggregate_reference_kkt
        <<" dual_reference="<<decision.dual_reference
        <<" aggregate_cooldown="<<aggregate_progress_cooldown
        <<" force_krylov="<<force_progress_krylov<<"\n";
    };
    if (level) {
      CostScope display(costs, NewtonCost::Diagnostics);
      std::cout << (st.dca_aggregate ? "LazyIPDCA  |  aggregate convex models · original filter\n" : "LazyIPDCA  |  DC majorants · original filter\n");
      std::cout << "[DCA-NORMS] live=original-units-reduced-double terminal=original-full-extended-fixed-reconstructed primal_dual_compl=inf filter_theta=scaled-L1 terminal_primal=original-inf terminal_kkt=normalized-max acceptable=normalized-max\n";
      std::cout << "[DCA-STRICT-OPTIONS] error=scaled-nlp-max tol="
        << stop_st.stationarity_tolerance << " primal=" << stop_st.feasibility_tolerance
        << " dual=" << stop_st.dual_infeasibility_tolerance
        << " compl=" << stop_st.complementarity_tolerance
        << " s_max=" << stop_st.nlp_error_scaling_max << "\n";
      std::cout << "[DCA-ACCEPTABLE-OPTIONS] iter=" << stop_st.acceptable_iterations << " tol=" << stop_st.acceptable_tolerance
        << " primal=" << stop_st.acceptable_constraint_tolerance << " dual=" << stop_st.acceptable_dual_tolerance
        << " compl=" << stop_st.acceptable_complementarity_tolerance << " s_max=" << stop_st.nlp_error_scaling_max << "\n";
      std::cout << "[DCA-FINISH-OPTIONS] enabled=" << st.dca_strict_finish
                << " budget=" << st.dca_strict_finish_max << " refinements=" << st.dca_strict_finish_refinements
                << " eta=" << st.dca_strict_finish_eta << " mu_fraction=" << st.dca_strict_finish_mu_fraction << " rho_complete_search=" << st.dca_rho_complete_search << "\n";
      std::cout << "[DCA-AGGREGATE-OPTIONS] preparation_cache=" << st.dca_aggregate_preparation_cache
                << " strict_probe_stop=" << st.dca_aggregate_strict_probe_stop
                << " certificate=" << stop_st.termination_test << "\n";
      if(st.termination_test=="ipopt")
        std::cout << "[DCA-CONTROL] tolerance=" << st.dca_control_tolerance
                  << " independent_of_termination=1 reference=R7D10-project-runner\n";
      dca_display(w, result, outer, level, 0, 0, 0, 0, seconds(start));
    }
    result.status = Status::MaxOuterIterations;
    result.message = "Lazy-DCA outer iteration limit";
    bool stop = false;
    detail::AcceptableCounter acceptable_counter;
    for (int k = 0; k < st.max_outer_iterations && !stop; ++k) {
      if(seconds(start)>=st.time_limit_seconds){result.status=Status::TimeLimit;result.message="Lazy-DCA time limit";break;}
      // Only a complete audit of THIS incumbent may authorize continuation.
      // Do not reuse a Result flag left over from an earlier outer iteration.
      bool current_acceptable_certified=false;
      bool current_audited=false,acceptable_reached=false;
      if (phase_one && outer.primal <= 1e-4) {
        phase_one = false;
        model_ready = reuse_model = false;
        dca_disable_models(w);
        w.lambda.assign(w.me, 0);
        for (int i = 0; i < w.mi; ++i)
          if (w.active[i]) {
            w.s[i] = std::max(0.01, -w.inequalities[i].original_at(w.x));
            w.z[i] = initial_mu / w.s[i];
          }
        w.evaluate();
        outer = dca_original(w, w.x, w.lambda, w.z);
        progress.clear();
        progress.observe(outer.kkt,outer.primal,outer.dual,false,
                         st.feasibility_tolerance,st.dual_infeasibility_tolerance);
        force_progress_krylov=false;
        aggregate_progress_cooldown=0;
        corrector_enabled = true;
      }
      if (st.termination_test=="ipopt" || (outer.primal <= st.feasibility_tolerance &&
          outer.dual <= st.dual_infeasibility_tolerance &&
          outer.comp <= st.complementarity_tolerance)) {
        CostScope certificate(costs, NewtonCost::Certificate);
        w.export_state(result);
        if (detail::audit_original_termination(problem, stop_st, result, &w.original_linear_support)) {
          result.status = Status::Solved;
          result.message = st.termination_test=="ipopt" ? "Ipopt 3.14.11 desired test passed on original QCQP; raw audit retained separately" : "original QCQP KKT certificate satisfied by filter Lazy-DCA";
          break;
        }
        current_audited=true;
        current_acceptable_certified=result.termination_audit_valid && result.acceptable_point;
      }
      if(st.acceptable_iterations>0||st.acceptable_on_failure||st.dca_strict_finish){
        CostScope certificate(costs,NewtonCost::Certificate);
        if(!current_audited) {
          w.export_state(result);
          if(detail::audit_original_termination(problem,stop_st,result,&w.original_linear_support)){
            result.status=Status::Solved;result.message="original QCQP strict KKT certificate satisfied";break;
          }
        }
        current_acceptable_certified=result.termination_audit_valid && result.acceptable_point;
        const bool reached=st.acceptable_iterations>0 &&
          acceptable_counter.observe(result.outer_iterations,
            result.termination_audit_valid && result.acceptable_point,
            st.acceptable_iterations,st.termination_test=="ipopt");
        result.acceptable_streak=st.acceptable_iterations>0
          ?acceptable_counter.count:0;
        if(result.termination_audit_valid&&result.acceptable_point&&
           (!have_acceptable_state||result.acceptable_error<best_acceptable_error)) {
          acceptable_x=w.x;
          acceptable_s=w.s;
          // The current workspace may still contain slacks for an active DC
          // model.  Save the equivalent original-QCQP slacks so recovery is
          // independent of that model's lifetime.
          if(model_ready)
            for(int i=0;i<w.mi;++i)if(w.active[i])
              acceptable_s[i]+=std::max(
                  0.0,w.inequalities[i].at(w.x)-
                      w.inequalities[i].original_at(w.x));
          acceptable_z=w.z;
          acceptable_lambda=w.lambda;
          have_acceptable_state=true;
          best_acceptable_error=result.acceptable_error;
          saved_acceptable_streak=result.acceptable_streak;
        }
        if(level && (result.acceptable_point || result.acceptable_streak))
          std::cout<<"[DCA-ACCEPTABLE] outer="<<result.outer_iterations<<" streak="<<result.acceptable_streak
            <<"/"<<st.acceptable_iterations<<" error="<<result.acceptable_error
            <<" raw_pr="<<result.feasibility.total_inf<<" raw_du="<<result.dual_infeasibility_inf
            <<" raw_comp="<<result.complementarity_inf<<" sd="<<result.termination_sd<<" sc="<<result.termination_sc<<"\n";
        acceptable_reached=reached;
      }
      if(level && st.termination_test=="ipopt") {
        const auto saved_flags=std::cout.flags();const auto saved_precision=std::cout.precision();
        std::cout<<std::scientific<<std::setprecision(8)<<"[DCA-STOP] test=ipopt-3.14.11 outer="<<result.outer_iterations
          <<" E="<<result.scaled_nlp_error<<" dual_internal="<<result.termination_internal_dual_inf
          <<" dual_unscaled="<<result.termination_dual_inf<<" primal_internal="<<result.termination_internal_primal_inf
          <<" compl_internal="<<result.termination_internal_compl_inf<<" compl_unscaled="<<result.termination_compl_inf
          <<" sd="<<result.termination_sd<<" sc="<<result.termination_sc<<" objective_scale="<<result.termination_objective_scale
          <<" acceptable="<<result.acceptable_point<<" streak="<<result.acceptable_streak<<"/"<<st.acceptable_iterations<<"\n";
        std::cout.flags(saved_flags);std::cout.precision(saved_precision);
      }
      if (w.pivoted) stats.pivoted_backend = w.pivoted->name();
      const bool externally_acceptable = current_acceptable_certified;
      outer.acceptable_certified=externally_acceptable;
      // The accepted endpoint already carries its original residuals. Only
      // re-anchor the Lagrangian to its current multipliers for this filter.
      outer.lagrangian = outer.current_lagrangian;
      committed_x = w.x;
      committed_s = w.s;
      committed_z = w.z;
      committed_lambda = w.lambda;
      have_committed_state = true;
      const Vec &accepted_x = committed_x;
      const Vec &accepted_s = committed_s;
      const Vec &accepted_z = committed_z;
      const Vec &accepted_lambda = committed_lambda;
      Vec original_slacks = accepted_s;
      if (model_ready)
        for (int i = 0; i < w.mi; ++i)
          if (w.active[i])
            original_slacks[i] += std::max(0.0, w.inequalities[i].at(w.x) -
                                                       w.inequalities[i].original_at(w.x));
      if(st.dca_strict_finish && st.dca_aggregate && !phase_one && current_acceptable_certified &&
         stats.strict_finish_attempts<st.dca_strict_finish_max) {
        if(dca_strict_finish(w,models,aggregate,outer,filter,stop_st,original_slacks,result.outer_iterations+1)) {
          model_ready=reuse_model=false;anderson.clear();previous_displacement.clear();
          ++result.outer_iterations;observe_accepted_progress(false);
          result.step_inf=0;
          for(int j=0;j<w.n;++j)result.step_inf=std::max(result.step_inf,std::abs(w.x[j]-accepted_x[j]));
          // The next top-of-loop audit owns the status and the consecutive
          // counter. The local candidate audit does not increment the streak.
          continue;
        }
      }
      if(acceptable_reached){result.status=Status::SolvedAcceptable;
        result.message="original QCQP acceptable tolerances satisfied for consecutive accepted iterates";break;}
      if(st.dca_aggregate_only) {
        // No partial model has ever been installed on this route. Failed
        // transactions may update certificate memory but never the iterate.
        w.s=original_slacks;
        const bool ok=aggregate.attempt_single_loop(w,models,outer,filter,
            result.outer_iterations+1,start);
        if(ok) {
          model_ready=reuse_model=false;
          anderson.clear();previous_displacement.clear();
          ++result.outer_iterations;observe_accepted_progress(true);
          result.step_inf=0;
          for(int j=0;j<w.n;++j)result.step_inf=std::max(result.step_inf,std::abs(w.x[j]-accepted_x[j]));
          if(level)std::cout<<"  AGGREGATE-ONLY iter="<<result.outer_iterations
              <<" objective="<<outer.objective/w.objective.scale<<" original_kkt="<<outer.kkt
              <<" original_primal="<<outer.primal<<" rho="<<stats.aggregate_last_rho
              <<" majorants="<<stats.models<<" time="<<seconds(start)<<std::endl;
          continue;
        }
        w.x=accepted_x;w.s=accepted_s;w.z=accepted_z;w.lambda=accepted_lambda;
        result.status=std::string(aggregate.last_rejection)=="aggregate-only-time-limit"?
            Status::TimeLimit:Status::LineSearchFailure;
        result.message=std::string("Aggregate-only search stopped; no majorant/restoration fallback: ")+aggregate.last_rejection;
        break;
      }
      bool prefer_restoration = false;
      bool force_terminal_krylov = false;
      const bool aggregate_progress_suppressed=aggregate_progress_cooldown>0;
      if(aggregate_progress_suppressed)--aggregate_progress_cooldown;
      if (st.dca_aggregate && !aggregate_progress_suppressed && !force_progress_krylov &&
          !phase_one && outer.primal<=st.dca_aggregate_primal_trigger) {
        const bool objective_model=w.objective.model_active;
        std::vector<bool> equality_models,inequality_models;
        for(const auto& q:w.equalities)equality_models.push_back(q.model_active);
        for(const auto& q:w.inequalities)inequality_models.push_back(q.model_active);
        w.s = original_slacks;
        const bool local_ok = aggregate.attempt(
            w,models,outer,filter,result.outer_iterations+1,
            externally_acceptable);
        prefer_restoration = st.dca_restoration_compare && st.dca_filter_model_retries>0 && !local_ok && outer.primal > st.feasibility_tolerance &&
            std::string(aggregate.last_rejection)=="aggregate-curvature-certificate";
        if (local_ok) {
          model_ready = reuse_model = false;
          anderson.clear();previous_displacement.clear();
          ++result.outer_iterations;
          observe_accepted_progress(true);
          result.step_inf=0;
          for(int j=0;j<w.n;++j)result.step_inf=std::max(result.step_inf,std::abs(w.x[j]-accepted_x[j]));
          if (level) std::cout<<"  AGGREGATE-DCA iter="<<result.outer_iterations
                             <<" objective="<<outer.objective/w.objective.scale<<" original_kkt="<<outer.kkt
                             <<" original_primal="<<outer.primal<<" rho="<<stats.aggregate_last_rho
                             <<" inner=-/"<<st.dca_filter_inner_max
                             <<" rebuild=-/"<<st.dca_filter_model_retries
                             <<" time="<<seconds(start)<<std::endl;
          continue;
        }
        force_terminal_krylov = !externally_acceptable &&
            detail::dca_stationary_phase(outer.primal,outer.normalized_kkt,
                st.feasibility_tolerance,st.stationarity_tolerance) &&
            aggregate.stationarity_evidence.needs_rescue();
        if(force_terminal_krylov){
          ++stats.terminal_rescue_triggers;
          model_ready=reuse_model=false;
          if(level>0)std::cout<<"[DCA-KKT-RESCUE] outer="<<result.outer_iterations+1
              <<" phase=armed reason=zero-shift-uncertified-and-regularized-noncontraction"
              <<" original_kkt="<<outer.kkt<<" normalized_kkt="<<outer.normalized_kkt
              <<" inf_pr="<<outer.primal
              <<" minimum_certified_rho="<<aggregate.stationarity_evidence.minimum_certified_rho<<"\n";
        }
        if(force_terminal_krylov && st.dca_krylov_max_iterations>0 &&
           aggregate.root_factor_compatible(w)) {
          ++stats.terminal_rescue_trials;++stats.aggregate_root_trials;
          Direction seed;seed.x.assign(w.n,0);seed.lambda.assign(w.me,0);
          seed.s.assign(w.mi,0);seed.z.assign(w.mi,0);double alpha=0;
          const double target=0.001*w.objective.scale*
              std::min(st.complementarity_tolerance,st.stationarity_tolerance);
          const bool ok=krylov.attempt(w,stats,outer,filter,seed,seed,target,
              result.outer_iterations,start,alpha,outer,true,&aggregate);
          if(level>0)std::cout<<"[DCA-KKT-RESCUE] outer="<<result.outer_iterations+1
              <<" phase=aggregate-factor-reuse new_factorizations=0 outcome="<<(ok?"accepted":"rejected")
              <<" reason="<<(ok?"original-kkt-filter":stats.krylov_last_rejection)<<"\n";
          if(ok){
            auto candidate=krylov.proposal.point;
            w.x=std::move(krylov.proposal.x);w.s=std::move(krylov.proposal.s);
            w.z=std::move(krylov.proposal.z);w.lambda=std::move(krylov.proposal.lambda);
            dca_inherit_guard(w,outer,candidate);outer=candidate;
            ++stats.terminal_rescue_acceptances;++stats.aggregate_root_acceptances;
            ++stats.inner_steps;++result.lazy_newton.accepted_steps;
            result.inner_iterations=stats.inner_steps;++result.outer_iterations;
            model_ready=reuse_model=false;dca_disable_models(w);
            anderson.clear();previous_displacement.clear();observe_accepted_progress(false);
            result.step_inf=0;
            for(int j=0;j<w.n;++j)result.step_inf=std::max(result.step_inf,std::abs(w.x[j]-accepted_x[j]));
            continue;
          }
          // An inaccurate/nonprogressing original root changes no iterate.
          // The established majorant/rebuild fallback remains available.
        }
        if(level>=1)
          std::cout<<"[DCA-ROUTE] outer="<<result.outer_iterations+1
                   <<" from=aggregate to=majorant reason=\""
                   <<((aggregate.last_rejection&&*aggregate.last_rejection)
                          ?aggregate.last_rejection:"aggregate-not-accepted")
                   <<"\"\n";
        // A rejected local transaction must preserve the general DCA trajectory,
        // including model age, slacks, and cached model activation flags.
        w.s=accepted_s;w.objective.model_active=objective_model;
        for(int e=0;e<w.me;++e)w.equalities[e].model_active=equality_models[e];
        for(int i=0;i<w.mi;++i)w.inequalities[i].model_active=inequality_models[i];
      }
      if (st.dca_local_active && !phase_one && stats.models > 0 &&
          outer.primal <= st.dca_local_primal_trigger && outer.kkt <= st.dca_local_trigger) {
        const bool objective_model=w.objective.model_active;
        std::vector<bool> equality_models,inequality_models;
        for(const auto& q:w.equalities)equality_models.push_back(q.model_active);
        for(const auto& q:w.inequalities)inequality_models.push_back(q.model_active);
        w.s = original_slacks;
        const bool local_ok = local_active.attempt(w, models, outer, filter);
        if (local_ok) {
          model_ready = reuse_model = false;
          anderson.clear();previous_displacement.clear();
          ++result.outer_iterations;
          observe_accepted_progress(false);
          result.step_inf=0;
          for(int j=0;j<w.n;++j)result.step_inf=std::max(result.step_inf,std::abs(w.x[j]-accepted_x[j]));
          if (level) std::cout<<"  LOCAL-DCA iter="<<result.outer_iterations<<" original_kkt="<<outer.kkt
                             <<" original_primal="<<outer.primal
                             <<" inner=-/"<<st.dca_filter_inner_max
                             <<" rebuild=-/"<<st.dca_filter_model_retries
                             <<" time="<<seconds(start)<<std::endl;
          continue;
        }
        // A rejected local transaction must preserve the general DCA trajectory,
        // including model age, slacks, and cached model activation flags.
        w.s=accepted_s;w.objective.model_active=objective_model;
        for(int e=0;e<w.me;++e)w.equalities[e].model_active=equality_models[e];
        for(int i=0;i<w.mi;++i)w.inequalities[i].model_active=inequality_models[i];
      }
      if (reuse_model) {
        w.evaluate();
        // Recenter as soon as solving the frozen model ceases to represent
        // the original stationarity defect, even if its step budget remains.
        if (inf_norm(w.rd) < 0.5 * w.objective.scale * outer.dual)
          reuse_model = false;
      }
      bool accepted = false, expanded = false;
      int performed = 0, accepted_ls = 0, current_rebuild = -1;
      const char *current_phase = "model";
      enum class TerminalReject { Filter, Factorization, LinearSolve };
      TerminalReject terminal_reject=TerminalReject::Filter;
      if(prefer_restoration && level>0)
        std::cout<<"[DCA-SELECT] outer="<<result.outer_iterations+1
                 <<" reason=aggregate-uncertified-infeasible policy=compare-original-kkt memory_independent=1\n";
      auto trace_context = [&]() {
        return DcaTraceContext{result.outer_iterations+1,current_rebuild,completed_model_steps};
      };
      auto budget = [&](const char *event) {
        if (level < 1) return;
        CostScope display(costs, NewtonCost::Diagnostics);
        dca_budget(w, level, result.outer_iterations + 1, current_phase,
                   current_rebuild < 0 ? -1 : completed_model_steps, current_rebuild, event);
      };
      double accepted_ap = 0, accepted_ad = 0;
      struct PendingDcaCandidate {
        Vec x,s,z,lambda;
        DcaPoint point;
        double ap=0,ad=0;
        int ls=0;
      };
      std::optional<PendingDcaCandidate> pending;

      for (int rebuild = 0; rebuild <= st.dca_filter_model_retries && !accepted; ++rebuild) {
        if (seconds(start) >= st.time_limit_seconds) {
          stop = true;
          result.status = Status::TimeLimit;
          result.message = "Lazy-DCA time limit";
          budget("time-limit");
          break;
        }
        current_rebuild = rebuild;
        w.x = accepted_x;
        w.s = accepted_s;
        w.z = accepted_z;
        w.lambda = accepted_lambda;
        if (expanded)
          for (int i = 0; i < w.mi; ++i)
            if (w.active[i] && !(w.z[i] > 0)) {
              w.s[i] = std::max(0.01, -w.inequalities[i].original_at(w.x));
              w.z[i] = st.lazy_newton_initial_mu / w.s[i];
            }
        const bool restoration =
            phase_one || (rebuild == st.dca_filter_model_retries && rebuild > 0 &&
                          outer.primal > st.feasibility_tolerance);
        if (rebuild > 0)
          ++stats.model_retries;
        if (restoration)
          ++stats.restorations;
        const bool reuse = model_ready && reuse_model && !restoration && rebuild == 0 && !expanded;
        current_phase = restoration ? "restoration" : (reuse ? "reuse" : "model");
        if (!reuse) {
          completed_model_steps = 0;
          stats.current_model_steps = 0;
        }
        budget("begin");
        if (reuse) {
          ++stats.model_reuses;
        } else {
          w.s = original_slacks;
          // Newly active rows have no old positive model slack.
          for (int i = 0; i < w.mi; ++i)
            if (w.active[i] && !(w.s[i] > 0))
              w.s[i] = std::max(0.01, -w.inequalities[i].original_at(w.x));
          CostScope curvature(costs, NewtonCost::Atoms);
          const double model_proximal=
              rebuild > 0 ? std::pow(10.0, rebuild - 1) * 1e-3 : 0;
          result.lazy_dca_max_model_proximal=std::max(
              result.lazy_dca_max_model_proximal,model_proximal);
          models.build(model_proximal, restoration);
          model_ready = true;
          model_steps = 0;
          w.penalty = std::max(1.0, 1.1 * std::max(inf_norm(w.lambda), inf_norm(w.z)));
        }
        expanded = false;
        const int remaining_steps = st.dca_filter_inner_max - model_steps;
        for (int inner = 0; inner < remaining_steps && !accepted; ++inner) {
          try {
            terminal_reject=TerminalReject::Filter;
            if (seconds(start) >= st.time_limit_seconds) {
              stop = true;
              result.status = Status::TimeLimit;
              result.message = "Lazy-DCA time limit";
              budget("time-limit");
              break;
            }
            w.evaluate();
            w.update_matrix();
            if (!w.factorize(0)) {
              terminal_reject=TerminalReject::Factorization;
              result.message = "DCA majorant KKT factorization: " + w.message;
              dca_trace_unavailable(w,"majorant","factor","kkt-factorization",w.message,trace_context());
              budget("factor-reject");
              break;
            }
            double mu = 0;
            int count = 0;
            Vec rc(w.mi, 0);
            for (int i = 0; i < w.mi; ++i)
              if (w.active[i]) {
                rc[i] = w.s[i] * w.z[i];
                mu += rc[i];
                ++count;
              }
            mu = count ? mu / count : 0;
            if((force_terminal_krylov||force_progress_krylov) && !restoration && inner==0 &&
               st.dca_krylov_max_iterations>0){
              const bool terminal_rescue=force_terminal_krylov;
              if(terminal_rescue)++stats.terminal_rescue_trials;
              else ++stats.progress_rescue_trials;
              Direction seed;
              seed.x.assign(w.n,0);seed.lambda.assign(w.me,0);
              seed.s.assign(w.mi,0);seed.z.assign(w.mi,0);
              double rescue_alpha=0;
              const double root_target=0.001*w.objective.scale*
                  std::min(st.complementarity_tolerance,st.stationarity_tolerance);
              const bool rescued=krylov.attempt(w,stats,outer,filter,seed,seed,root_target,
                  result.outer_iterations,start,rescue_alpha,outer,true);
              if(level>0)std::cout<<"[DCA-KKT-RESCUE] outer="<<result.outer_iterations+1
                  <<" phase=before-majorant-line-search rebuild="<<rebuild
                  <<" outcome="<<(rescued?"accepted":"rejected")
                  <<" reason="<<(rescued?"original-kkt-filter":stats.krylov_last_rejection)
                  <<" original_kkt="<<(rescued?krylov.proposal.point.kkt:outer.kkt)<<"\n";
              if(rescued){
                auto candidate=krylov.proposal.point;
                if(candidate.primal>st.feasibility_tolerance)filter.insert(outer);
                w.x=std::move(krylov.proposal.x);w.s=std::move(krylov.proposal.s);
                w.z=std::move(krylov.proposal.z);w.lambda=std::move(krylov.proposal.lambda);
                dca_inherit_guard(w,outer,candidate);outer=candidate;
                accepted=true;accepted_ap=rescue_alpha;accepted_ad=krylov.accepted_dual_alpha;
                accepted_ls=krylov.accepted_line_steps;
                if(terminal_rescue)++stats.terminal_rescue_acceptances;
                else ++stats.progress_rescue_acceptances;
                force_terminal_krylov=force_progress_krylov=false;
                model_ready=reuse_model=false;dca_disable_models(w);
                anderson.clear();previous_displacement.clear();
                ++performed;++stats.inner_steps;++result.lazy_newton.accepted_steps;
                result.inner_iterations=stats.inner_steps;
                result.step_inf=0;
                for(int j=0;j<w.n;++j)result.step_inf=std::max(result.step_inf,std::abs(w.x[j]-accepted_x[j]));
                budget("original-kkt-accepted");
                break;
              }
              // Leave the causal flag armed: each subsequent rebuild supplies
              // a different, stronger certified preconditioner at this anchor.
            }
            Direction affine = w.direction(rc, w.rd, w.h, w.rp), direction = affine;
            double target = 0;
            if (count) {
              const double ap = w.fraction(affine, 1, 1), ad = w.fraction(affine, 1, 2);
              double amu = 0;
              for (int i = 0; i < w.mi; ++i)
                if (w.active[i])
                  amu += (w.s[i] + ap * affine.s[i]) * (w.z[i] + ad * affine.z[i]);
              amu /= count;
              const double sigma =
                  std::clamp(std::pow(std::max(0.0, amu) / std::max(mu, 1e-300), 3), 1e-8, 0.95);
              stats.last_sigma = sigma;
              ++stats.corrector_steps;
              target =
                  std::max(0.001 * w.objective.scale *
                               std::min(st.complementarity_tolerance, st.stationarity_tolerance),
                           sigma * mu);
              for (int i = 0; i < w.mi; ++i)
                if (w.active[i])
                  rc[i] = w.s[i] * w.z[i] - target +
                          (corrector_enabled ? affine.s[i] * affine.z[i] : 0);
              direction = w.direction(rc, w.rd, w.h, w.rp);
            }
            Vec tx(w.n), ts(w.mi), tz(w.mi), tl(w.me);
            bool private_ok = false;
            int line_steps = 0;
            double ap = 0, ad = 0;
            bool have_private_trial = false;
            const bool private_continuation_allowed =
                detail::dca_private_continuation_allowed(
                    outer.kkt, outer.primal, 1.0,
                    st.dca_common_step_primal_trigger);
            std::optional<DcaPoint> selected_original;
            Vec bx, bs, bz, bl;
            double bap = 0, bad = 0;
            int bls = 0;
            for (int correction = 0; correction < 2 && !private_ok; ++correction) {
              const double apmax = w.fraction(direction, st.lazy_newton_fraction_to_boundary, 1);
              const double admax = w.fraction(direction, st.lazy_newton_fraction_to_boundary, 2);
              const bool lazy_merit = st.dca_filter_lazy_merit;
              double theta = 0;
              if (lazy_merit) {
                for (double h : w.h) theta += std::abs(h);
                for (int i = 0; i < w.mi; ++i) if (w.active[i]) theta += std::abs(w.rp[i]);
              } else theta = w.theta(w.x, w.s);
              double slope = w.objective.directional(direction.x);
              for (int i = 0; i < w.mi; ++i)
                if (w.active[i])
                  slope -= target * direction.s[i] / w.s[i];
              w.penalty = std::max(w.penalty, 1.1 * std::max(inf_norm(w.lambda), inf_norm(w.z)));
              if (theta > 1e-14 && slope - w.penalty * theta >= -1e-8 * theta)
                w.penalty = std::max(w.penalty, (slope + std::abs(slope) + 1) / theta);
              stats.last_penalty = w.penalty;
              slope -= w.penalty * theta;
              double merit = 0, before = 0;
              bool merit_ready = false;
              auto prepare_merit = [&]() {
                if (merit_ready) return;
                merit = w.merit(w.x, w.s, target);
                before = w.perturbed_residual(w.x, w.lambda, w.s, w.z, target);
                merit_ready = true;
              };
              if (!lazy_merit) prepare_merit();
              for (int ls = 0; ls < st.line_search_max_steps; ++ls) {
                const double tau = std::pow(st.line_search_contraction, ls);
                const auto step=detail::primal_dual_step(apmax,admax,tau,st.dca_dual_step);
                ap=step.primal;ad=step.bound_dual;
                const double lambda_alpha=detail::dca_equality_step(
                    outer.primal,st.feasibility_tolerance,ap,step.equality_dual);
                line_steps = ls;
                for (int j = 0; j < w.n; ++j)
                  tx[j] = w.x[j] + ap * direction.x[j];
                for (int e = 0; e < w.me; ++e)
                  tl[e] = w.lambda[e] + lambda_alpha * direction.lambda[e];
                bool positive = true;
                for (int i = 0; i < w.mi; ++i) {
                  ts[i] = w.s[i] + ap * direction.s[i];
                  tz[i] = w.z[i] + ad * direction.z[i];
                  if (w.active[i] && (!(ts[i] > 0) || !(tz[i] > 0)))
                    positive = false;
                }
                // The outer globalization is on the ORIGINAL problem. A full
                // fraction-to-boundary step accepted there must not first be
                // shortened by a penalty inherited from an older DC model.
                if (positive && finite(tx) && finite(tl) && finite(tz)) {
                  CostScope original_gate(costs, NewtonCost::Globalization);
                  struct GateCandidate {Vec x,s,z,lambda;DcaPoint point;double ap=0,ad=0,pred=0;const char* stage=nullptr;};
                  auto gate=[&](Vec gx,Vec gs,Vec gz,Vec gl,double gap_alpha,double gad,
                                const char* gate_stage)->std::optional<GateCandidate>{
                  ++stats.original_line_trials;
                  auto full = dca_original(w, gx, gl, gz, &accepted_lambda, &accepted_z);
                  if(st.dca_dual_decomposition) {
                    const double hessian_gap=dca_original_hessian_gap(w,direction.x);
                    dca_trace_dual_decomposition(w,outer,full,w.x,w.lambda,w.z,direction,
                        gap_alpha,(st.dca_dual_step=="ipopt"?gap_alpha:
                            detail::dca_equality_step(outer.primal,st.feasibility_tolerance,gap_alpha,gad)),
                        gad,"majorant",gate_stage,direction.residual,hessian_gap,
                        trace_context());
                  }
                  const double pred = restoration ? 0 : w.objective.at(accepted_x) - w.objective.at(gx);
                  dca_trace_filter(w,filter,outer,full,pred,restoration,"majorant",
                                   gate_stage,ls,gap_alpha,gad,nullptr,-1,trace_context());
                  if (dca_kkt_allowed(w, outer, full) &&
                      filter.acceptable(outer, full, pred, st.feasibility_tolerance)) {
                    return GateCandidate{std::move(gx),std::move(gs),std::move(gz),std::move(gl),
                                         full,gap_alpha,gad,pred,gate_stage};
                  }
                  ++stats.original_line_rejections;
                  return std::nullopt;
                  };
                  const char* split_stage=correction==0?"trial":"centering-trial";
                  auto chosen=gate(tx,ts,tz,tl,ap,ad,split_stage);
                  if(st.dca_dual_step=="coupled" && chosen &&
                     detail::compare_common_step(outer.primal,outer.dual,chosen->point.dual,
                       st.dca_common_step_primal_trigger,st.dca_common_step_dual_growth)) {
                    const double common=detail::common_feasible_step(ap,lambda_alpha,ad);
                    if(common>0 && (std::abs(common-ap)>1e-15 || std::abs(common-ad)>1e-15)) {
                      Vec cx=w.x,cs=w.s,cz=w.z,cl=w.lambda;
                      axpy(common,direction.x,cx);axpy(common,direction.lambda,cl);
                      axpy(common,direction.z,cz);axpy(common,direction.s,cs);
                      auto synchronized=gate(std::move(cx),std::move(cs),std::move(cz),std::move(cl),
                                             common,common,"common-trial");
                      if(synchronized && (!chosen || synchronized->point.kkt<chosen->point.kkt))
                        chosen=std::move(synchronized);
                    }
                  }
                  if(chosen) {
                    tx=std::move(chosen->x);ts=std::move(chosen->s);tz=std::move(chosen->z);tl=std::move(chosen->lambda);
                    ap=chosen->ap;ad=chosen->ad;private_ok=true;selected_original=chosen->point;
                    ++stats.original_line_acceptances;
                    if(chosen->stage==std::string("common-trial") && level>0)
                      std::cout<<"[DCA-STEP-SELECT] path=majorant outer="<<result.outer_iterations+1
                               <<" mode=common-alpha alpha="<<ap<<" original_kkt="<<chosen->point.kkt<<"\n";
                    break;
                  }
                } else {
                  dca_trace_domain(w,"majorant","trial",ls,ap,ad,
                                   positive?"nonfinite-state":"nonpositive-state",-1,trace_context());
                }
                if (!private_continuation_allowed)
                  continue;
                prepare_merit();
                ++stats.private_line_trials;
                const double trial = positive ? w.merit(tx, ts, target) : INFINITY;
                const double noise =
                    16 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(merit));
                double trial_residual=std::numeric_limits<double>::quiet_NaN();
                bool trial_residual_evaluated=false;
                const bool merit_ok = std::isfinite(trial) && finite(tx) && finite(tl) && finite(tz) &&
                             ((ap >= 1e-6 && slope < 0 &&
                               trial <= merit + st.line_search_armijo * ap * slope + noise) ||
                              (trial <= merit + noise && before > 0 &&
                               (trial_residual_evaluated=true,
                                trial_residual=w.perturbed_residual(tx, tl, ts, tz, target)) <=
                                   (1 - 1e-4 * std::min(ap, ad)) * before));
                if(dca_filter_trace_enabled(w))
                  dca_trace_private(w,ls,ap,ad,positive,finite(tx)&&finite(tl)&&finite(tz),
                                    merit_ok,merit,trial,slope,theta,target,noise,before,
                                    trial_residual,trial_residual_evaluated,trace_context());
                if (!merit_ok) ++stats.private_line_rejections;
                if (merit_ok && !have_private_trial) {
                  bx = tx; bs = ts; bz = tz; bl = tl;
                  bap = ap; bad = ad; bls = ls;
                  have_private_trial = true;
                }
                // A private point is retained for continuation if the original
                // filter rejects this direction after a bounded backtracking.
                if (have_private_trial && ls >= 7) break;
              }
              if (!private_ok && count && correction == 0 && corrector_enabled) {
                for (int i = 0; i < w.mi; ++i)
                  if (w.active[i])
                    rc[i] = w.s[i] * w.z[i] - target;
                direction = w.direction(rc, w.rd, w.h, w.rp);
                ++result.lazy_newton.predictor_corrector_fallbacks;
                ++stats.centering_trials;
              }
            }
            if (!private_ok && have_private_trial) {
              tx.swap(bx); ts.swap(bs); tz.swap(bz); tl.swap(bl);
              ap = bap; ad = bad; line_steps = bls; private_ok = true;
            }
            if (!private_ok) {
              budget("line-reject");
              break;
            }
            ++performed;
            ++model_steps;
            stats.maximum_model_steps = std::max(stats.maximum_model_steps, model_steps);
            ++stats.inner_steps;
            result.inner_iterations = stats.inner_steps;
            ++result.lazy_newton.accepted_steps;
            const auto gate_start = Clock::now();
            CostScope globalization(costs, NewtonCost::Globalization);
            for (int i = 0; i < w.mi; ++i)
              if (!w.active[i] && w.inequalities[i].original_at(tx) >
                                      st.feasibility_tolerance * w.inequalities[i].scale) {
                w.active[i] = 1;
                dca_trace_domain(w,"majorant","commit",line_steps,ap,ad,"inactive-original-row",i,trace_context());
                expanded = true;
                ++result.lazy_newton.constraint_activations;
              }
            if (expanded) {
              pending.reset(); // the active-set contract changed; discard held candidates
              ++result.lazy_newton.working_set_retries;
              budget("working-set-retry");
              break;
            }
            DcaPoint candidate = selected_original ? *selected_original :
                dca_original(w, tx, tl, tz, &accepted_lambda, &accepted_z);
            const double predicted =
                restoration ? 0 : w.objective.at(accepted_x) - w.objective.at(tx);
            ++stats.filter_trials;
            accepted = dca_kkt_allowed(w, outer, candidate) &&
                       filter.acceptable(outer, candidate, predicted, st.feasibility_tolerance);
            dca_trace_filter(w,filter,outer,candidate,predicted,restoration,"majorant",prefer_restoration?"candidate":"commit",line_steps,ap,ad,nullptr,-1,trace_context());
            if (!accepted) {
              ++stats.filter_rejections;
              // Bounded second-order normal correction. The Jacobian remains
              // the frozen outer Jacobian, so the existing KKT factor is valid.
              // Every corrected endpoint is checked on original functions.
              if (!w.equalities.empty() && candidate.theta > 1e-14) {
                auto soc_start = Clock::now();
                ++stats.soc_attempts;
                Vec eq(w.me), zero_dual(w.n, 0), zero_primal(w.mi, 0), zero_comp(w.mi, 0);
                for (int e = 0; e < w.me; ++e)
                  eq[e] = w.equalities[e].original_at(tx);
                Direction normal = w.direction(zero_comp, zero_dual, eq, zero_primal);
                double radius = 0;
                for (int j = 0; j < w.n; ++j)
                  radius = std::max(radius, std::abs(tx[j] - accepted_x[j]));
                if (inf_norm(normal.x) <= std::max(1e-8, 0.5 * radius)) {
                  Vec cx = tx, cs = ts, cz = tz, cl = tl;
                  bool positive = true;
                  axpy(1, normal.x, cx);
                  axpy(1, normal.lambda, cl);
                  axpy(1, normal.s, cs);
                  axpy(1, normal.z, cz);
                  for (int i = 0; i < w.mi; ++i)
                    if (w.active[i] && (!(cs[i] > 0) || !(cz[i] > 0)))
                      positive = false;
                  if (positive && finite(cx) && finite(cl) && finite(cz)) {
                    auto corrected = dca_original(w, cx, cl, cz, &accepted_lambda, &accepted_z);
                    ++stats.filter_trials;
                    if(dca_filter_trace_enabled(w))
                      dca_trace_filter(w,filter,outer,corrected,
                                       restoration?0:w.objective.at(accepted_x)-w.objective.at(cx),
                                       restoration,"majorant","soc",line_steps,ap,ad,nullptr,-1,trace_context());
                    if (dca_kkt_allowed(w, outer, corrected) &&
                        filter.acceptable(
                            outer, corrected,
                            restoration ? 0 : w.objective.at(accepted_x) - w.objective.at(cx),
                            st.feasibility_tolerance)) {
                      tx.swap(cx);
                      ts.swap(cs);
                      tz.swap(cz);
                      tl.swap(cl);
                      candidate = corrected;
                      accepted = true;
                      ++stats.soc_acceptances;
                    }
                  }
                }
                stats.soc_seconds += seconds(soc_start);
              }
            }
            if (prefer_restoration && accepted && !restoration) {
              if (!pending || candidate.kkt < pending->point.kkt) {
                Vec os=ts;
                for(int i=0;i<w.mi;++i) if(w.active[i])
                  os[i]+=std::max(0.0,w.inequalities[i].at(tx)-w.inequalities[i].original_at(tx));
                pending=PendingDcaCandidate{tx,std::move(os),tz,tl,candidate,ap,ad,line_steps};
              }
              if(level>0)std::cout<<"[DCA-SELECT] outer="<<result.outer_iterations+1
                <<" decision=hold kind=majorant kkt="<<std::setprecision(17)<<candidate.kkt
                <<" rebuild="<<rebuild<<"\n";
              accepted=false;
              budget("candidate-held-for-restoration");
              break;
            }
            if (prefer_restoration && accepted && restoration && pending &&
                pending->point.kkt < candidate.kkt) {
              accepted=false;
              budget("majorant-preferred");
              break;
            }
            if(prefer_restoration && accepted && restoration && level>0)
              std::cout<<"[DCA-SELECT] outer="<<result.outer_iterations+1
                <<" decision=commit kind=restoration kkt="<<std::setprecision(17)<<candidate.kkt
                <<" majorant_kkt="<<(pending?pending->point.kkt:INFINITY)<<"\n";
            bool krylov_accelerated=false;
            if(accepted && st.dca_krylov && !restoration && rebuild==0 && inner==0) {
              const bool accelerated=krylov.attempt(
                  w,stats,outer,filter,affine,direction,target,
                  result.outer_iterations,start,ap,candidate,false);
              if(accelerated) {
                tx=std::move(krylov.proposal.x);ts=std::move(krylov.proposal.s);
                tz=std::move(krylov.proposal.z);tl=std::move(krylov.proposal.lambda);
                candidate=krylov.proposal.point;krylov_accelerated=true;model_ready=false;
                anderson.clear();previous_displacement.clear();
              }
            }
            bool extrapolated = krylov_accelerated;
            Vec displacement(w.n);
            for (int j = 0; j < w.n; ++j) displacement[j] = tx[j] - accepted_x[j];
            if (anderson_enabled && !krylov_accelerated) {
              const bool eligible = accepted && !restoration && model_steps == 1 &&
                  outer.primal <= 1e-4 &&
                  candidate.primal <= 1e-4 &&
                  outer.kkt <= 1.0 && candidate.kkt < outer.kkt;
              if (!eligible) anderson.clear();
              else {
                const auto boost_start = Clock::now();
                DcaAnderson::Image proposal;
                if (anderson.propose(w, accepted_x, accepted_lambda, accepted_z,
                                     tx, tl, tz, ts, proposal)) {
                  Vec original_s = ts;
                  for (int i = 0; i < w.mi; ++i) if (w.active[i])
                    original_s[i] += std::max(0.0, w.inequalities[i].at(tx) -
                                                        w.inequalities[i].original_at(tx));
                  for (double blend : {1.0, 0.5}) {
                    Vec cx = tx, cl = tl, cz = tz, cs = original_s;
                    for (int j = 0; j < w.n; ++j) cx[j] += blend * (proposal.x[j] - tx[j]);
                    for (int e = 0; e < w.me; ++e) cl[e] += blend * (proposal.lambda[e] - tl[e]);
                    bool positive = true;
                    for (int i = 0; i < w.mi; ++i) if (w.active[i]) {
                      cz[i] += blend * (proposal.z[i] - tz[i]);
                      cs[i] += blend * (proposal.original_s[i] - original_s[i]);
                      cs[i] -= std::max(0.0, w.inequalities[i].at(cx) -
                                                     w.inequalities[i].original_at(cx));
                      if (!(cz[i] > 0) || !(cs[i] > 0)) positive = false;
                    }
                    if (!positive || !finite(cx) || !finite(cl) || !finite(cz) || !finite(cs)) continue;
                    for (int i = 0; i < w.mi; ++i) if (!w.active[i] &&
                        w.inequalities[i].original_at(cx) > st.feasibility_tolerance * w.inequalities[i].scale)
                      positive = false;
                    if (!positive) continue;
                    ++stats.extrapolation_trials;
                    ++stats.anderson_trials;
                    auto accelerated = dca_original(w, cx, cl, cz, &accepted_lambda, &accepted_z);
                    // The standard accepted candidate is retained unless both
                    // ORIGINAL KKT contraction and the same ORIGINAL filter pass.
                    if (accelerated.primal <= std::max(st.feasibility_tolerance, candidate.primal) &&
                        accelerated.kkt <= 0.8 * candidate.kkt &&
                        filter.acceptable(outer, accelerated,
                            w.objective.at(accepted_x) - w.objective.at(cx), st.feasibility_tolerance)) {
                      tx.swap(cx); tl.swap(cl); tz.swap(cz); ts.swap(cs);
                      candidate = accelerated; extrapolated = true;
                      ++stats.extrapolation_acceptances;
                      ++stats.anderson_acceptances;
                      if (level >= 4) {
                        CostScope display(costs, NewtonCost::Diagnostics);
                        std::cout << "  DCA-ANDERSON history=" << anderson.history.size()
                                  << " blend=" << blend << " original_kkt=" << candidate.kkt
                                  << " accepted=yes" << std::endl;
                      }
                      break;
                    }
                  }
                }
                stats.extrapolation_seconds += seconds(boost_start);
              }
            }
            if (accepted && !extrapolated && !restoration && st.dca_filter_extrapolation &&
                outer.theta <= st.feasibility_tolerance && candidate.primal <= st.feasibility_tolerance &&
                candidate.kkt < outer.kkt && previous_displacement.size() == displacement.size()) {
              // A filtered acceleration of the DC fixed-point iteration, not a
              // second Hessian/Newton solve. Use only a stable local secant mode.
              const double old_norm = dot(previous_displacement, previous_displacement);
              const double new_norm = dot(displacement, displacement);
              const double product = dot(displacement, previous_displacement);
              const double contraction = old_norm > 0 ? product / old_norm : 0;
              if (new_norm > 1e-30 && contraction > 0.05 && contraction < 0.99 &&
                  product * product >= 0.98 * old_norm * new_norm) {
                const auto boost_start = Clock::now();
                double omega = std::min(16.0, 1.0 / (1.0 - contraction));
                for (int trial = 0; trial < 3 && omega > 1.1; ++trial, omega = 0.5 * (1 + omega)) {
                  Vec cx = accepted_x, cl = accepted_lambda, cz = accepted_z, cs = accepted_s;
                  axpy(omega, displacement, cx);
                  for (int e = 0; e < w.me; ++e) cl[e] += omega * (tl[e] - accepted_lambda[e]);
                  for (int i = 0; i < w.mi; ++i) if (w.active[i]) {
                    cz[i] = std::max(1e-16, accepted_z[i] + omega * (tz[i] - accepted_z[i]));
                    cs[i] = std::max(1e-16, accepted_s[i] + omega * (ts[i] - accepted_s[i]));
                  }
                  ++stats.extrapolation_trials;
                  auto accelerated = dca_original(w, cx, cl, cz, &accepted_lambda, &accepted_z);
                  if (accelerated.primal <= st.feasibility_tolerance &&
                      accelerated.kkt <= 0.8 * candidate.kkt &&
                      filter.acceptable(outer, accelerated, w.objective.at(accepted_x) - w.objective.at(cx),
                                        st.feasibility_tolerance)) {
                    tx.swap(cx); tl.swap(cl); tz.swap(cz); ts.swap(cs);
                    candidate = accelerated; extrapolated = true;
                    ++stats.extrapolation_acceptances;
                    break;
                  }
                }
                stats.extrapolation_seconds += seconds(boost_start);
              }
            }
            if (level >= 4) {
              CostScope display(costs, NewtonCost::Diagnostics);
              std::cout << "  DCA k=" << k << " inner=" << inner + 1 << " model=" << stats.models
                        << " original_filter=" << (accepted ? "accept" : "continue")
                        << " theta=" << candidate.theta << " kkt=" << candidate.kkt
                        << " df=" << candidate.objective - outer.objective
                        << " dL=" << candidate.lagrangian - outer.lagrangian
                        << " rho=" << stats.rho_lagrangian << " model_age=" << model_steps
                        << " penalty=" << stats.last_penalty << " sigma=" << stats.last_sigma
                        << " original_ls=" << stats.original_line_rejections
                        << " private_ls=" << stats.private_line_rejections
                        << " extrapolated=" << (extrapolated ? "yes" : "no") << std::endl;
            }
            stats.original_filter_seconds += seconds(gate_start);
            w.x.swap(tx);
            w.s.swap(ts);
            w.z.swap(tz);
            w.lambda.swap(tl);
            ++completed_model_steps;
            stats.current_model_steps = completed_model_steps;
            budget(accepted ? "accepted" : "private");
            if (accepted) {
              force_progress_krylov=false;
              if (inner == 0)
                ++stats.first_step_acceptances;
              if (candidate.theta > st.feasibility_tolerance)
                filter.insert(outer);
              dca_inherit_guard(w, outer, candidate);
              outer = candidate;
              accepted_ap = ap;
              accepted_ad = ad;
              accepted_ls = line_steps;
              double equality_defect = 0;
              for (const auto &q : w.equalities)
                equality_defect += std::abs(q.original_at(w.x) - q.at(w.x));
              if (extrapolated || restoration) previous_displacement.clear();
              else previous_displacement = displacement;
              reuse_model = st.dca_filter_model_reuse && !restoration && !extrapolated &&
                            model_steps < st.dca_filter_inner_max &&
                            equality_defect <= 0.1 * std::max(st.feasibility_tolerance, outer.theta);
              if(krylov_accelerated)dca_disable_models(w);
              result.step_inf = 0;
              for (int j = 0; j < w.n; ++j)
                result.step_inf = std::max(result.step_inf, std::abs(w.x[j] - accepted_x[j]));
            } else
              ++stats.private_continuations;
          } catch (const std::runtime_error &failure) {
            const std::string reason = failure.what();
            if (reason.find("Newton residual rejected") == std::string::npos &&
                reason.find("linear solve:") == std::string::npos)
              throw;
            w.message = reason;
            terminal_reject=TerminalReject::LinearSolve;
            if(level>=2)
              std::cout<<"[DCA-SOLVE-REJECT] outer="<<result.outer_iterations+1
                       <<" rebuild="<<rebuild<<" inner="<<inner
                       <<" reason=\""<<reason<<"\"\n";
            dca_trace_unavailable(w,"majorant","direction","linear-solve",reason,trace_context());
            budget("solve-reject");
            break; // reject the numerical transaction, rebuild at the accepted point
          }
        }
        if (stop)
          break;
        if (!accepted && model_steps >= st.dca_filter_inner_max)
          budget("inner-limit");
      }
      if (!accepted && pending && !stop) {
        w.x=std::move(pending->x);w.s=std::move(pending->s);
        w.z=std::move(pending->z);w.lambda=std::move(pending->lambda);
        dca_disable_models(w);model_ready=reuse_model=false;
        anderson.clear();previous_displacement.clear();
        if(pending->point.theta>st.feasibility_tolerance)filter.insert(outer);
        dca_inherit_guard(w,outer,pending->point);
        outer=pending->point;accepted=true;
        accepted_ap=pending->ap;accepted_ad=pending->ad;accepted_ls=pending->ls;
        result.step_inf=0;
        for(int j=0;j<w.n;++j)result.step_inf=std::max(result.step_inf,std::abs(w.x[j]-accepted_x[j]));
        if(level>0)std::cout<<"[DCA-SELECT] outer="<<result.outer_iterations+1
          <<" decision=commit kind=majorant kkt="<<std::setprecision(17)<<outer.kkt<<"\n";
        budget("selected-majorant");
      }
      if (!accepted) {
        w.x = accepted_x;
        w.s = accepted_s;
        w.z = accepted_z;
        w.lambda = accepted_lambda;
        if (!stop) {
          if(terminal_reject==TerminalReject::Factorization) {
            result.status=Status::FactorizationFailure;
            result.message="Lazy-DCA exhausted bounded majorant rebuilds after KKT factorization rejection: "+w.message;
          } else if(terminal_reject==TerminalReject::LinearSolve) {
            result.status=Status::InnerFailure;
            result.message="Lazy-DCA exhausted bounded majorant rebuilds after Newton solve rejection: "+w.message;
          } else {
            result.status=Status::LineSearchFailure;
            result.message="Lazy-DCA exhausted bounded majorant steps/rebuilds; original filter did not accept";
          }
          budget("exhausted");
        }
        break;
      }
      ++result.outer_iterations;
      observe_accepted_progress(false);
      if (level && result.outer_iterations % st.live_iteration_every == 0) {
        CostScope display(costs, NewtonCost::Diagnostics);
        dca_display(w, result, outer, level, performed, accepted_ap, accepted_ad, accepted_ls,
                    seconds(start), current_rebuild);
      }
    }
    {
      CostScope certificate(costs, NewtonCost::Certificate);
      w.export_state(result);
      if (detail::audit_original_termination(problem, stop_st, result, &w.original_linear_support)) {
        result.status = Status::Solved;
        result.message = st.termination_test=="ipopt" ? "Ipopt 3.14.11 desired test passed on original QCQP; raw audit retained separately" : "original QCQP KKT certificate satisfied by filter Lazy-DCA";
      } else if(result.status==Status::MaxOuterIterations && st.acceptable_iterations>0){
        const bool reached=acceptable_counter.observe(result.outer_iterations,
          result.termination_audit_valid && result.acceptable_point,st.acceptable_iterations,st.termination_test=="ipopt");
        result.acceptable_streak=acceptable_counter.count;
        if(reached){result.status=Status::SolvedAcceptable;result.message="original QCQP acceptable certificate at final accepted iterate";}
      }
    }
  } catch (const std::exception &e) {
    result.status = Status::NumericalError;
    result.message = e.what();
    if (owner) {
      if (have_committed_state) {
        owner->x = committed_x;
        owner->s = committed_s;
        owner->z = committed_z;
        owner->lambda = committed_lambda;
      }
      dca_disable_models(*owner);
      CostScope certificate(costs, NewtonCost::Certificate);
      owner->export_state(result);
      detail::audit_original_termination(problem, stop_st, result, &owner->original_linear_support);
    }
  }
  const bool recoverable_terminal_failure=
      result.status==Status::FactorizationFailure ||
      result.status==Status::LineSearchFailure ||
      result.status==Status::InnerFailure ||
      result.status==Status::RestorationFailure;
  if(st.acceptable_on_failure&&recoverable_terminal_failure&&owner&&
     have_acceptable_state&&
     (!result.termination_audit_valid||!result.acceptable_point)) {
    owner->x=acceptable_x;
    owner->s=acceptable_s;
    owner->z=acceptable_z;
    owner->lambda=acceptable_lambda;
    dca_disable_models(*owner);
    owner->export_state(result);
    detail::audit_original_termination(
        problem,stop_st,result,&owner->original_linear_support);
    result.acceptable_streak=saved_acceptable_streak;
  }
  if(st.acceptable_on_failure&&recoverable_terminal_failure&&
     result.termination_audit_valid&&result.acceptable_point) {
    result.original_failure_status=status_string(result.status);
    result.original_failure_message=result.message;
    result.status=Status::SolvedAcceptableOnFailure;
    result.message="restored original-QCQP point satisfies acceptable tolerances after terminal failure";
    result.acceptable_on_failure=true;
    result.acceptable_trigger="terminal_failure";
  }
  if(result.status==Status::SolvedAcceptable){
    result.acceptable_convergence=result.termination_audit_valid && result.acceptable_point &&
      result.acceptable_required>0 && result.acceptable_streak>=result.acceptable_required;
    if(!result.acceptable_convergence){result.status=Status::NumericalError;result.message="final acceptable certificate failed";}
  } else if(result.status==Status::SolvedAcceptableOnFailure) {
    result.acceptable_convergence=result.termination_audit_valid&&result.acceptable_point&&
      result.acceptable_on_failure&&result.acceptable_trigger=="terminal_failure";
    if(!result.acceptable_convergence){result.status=Status::NumericalError;result.message="acceptable-on-failure certificate failed";}
  }
  if (owner) {
    if (owner->pivoted) result.dca_filter.pivoted_backend = owner->pivoted->name();
    result.lazy_newton.final_active_constraints=std::count(owner->active.begin(),owner->active.end(),1);
    result.lazy_newton.symbolic_analyses = owner->factor->symbolic_analyses();
    result.lazy_newton.symbolic_reuses = owner->factor->symbolic_reuses();
    result.lazy_newton.scatter_builds = owner->factor->numeric_scatter_builds();
    result.lazy_newton.scatter_reuses = owner->factor->numeric_scatter_reuses();
  }
  result.termination_multiplier_source = "filter-dca-original-extended-fixed-reconstructed";
  result.kkt_factorizations =
      result.lazy_newton.factor_attempts + result.dca_filter.curvature_factors;
  result.objective_rank = result.dca_filter.atoms_lagrangian;
  if (level) {
    CostScope display(costs, NewtonCost::Diagnostics);
    std::cout << "\nEXIT: " << result.message << std::endl;
  }
  {
    CostScope cleanup(costs, NewtonCost::Cleanup);
    owner.reset();
  }
  costs.finish();
  result.elapsed_seconds = seconds(start);
  return result;
}
