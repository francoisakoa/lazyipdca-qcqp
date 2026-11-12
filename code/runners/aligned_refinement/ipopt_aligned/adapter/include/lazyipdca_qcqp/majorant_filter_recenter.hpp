#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
namespace lazyipdca::qcqp::detail {
// Dimensionless growth relative to a PHYSICAL complementarity reference.
// The caller uses min(initial complementarity, mu/objective scale), so a new
// barrier target is not compared against an obsolete large startup product.
// This gate cannot turn an unsuccessful direction into an accepted primal step.
inline bool filter_recenter_needed(bool already_used, double primal,
    double comp, double initial_comp, double feas_tol, double comp_tol) {
  return !already_used && std::isfinite(primal) && std::isfinite(comp) &&
      std::isfinite(initial_comp) && initial_comp >= 0 &&
      std::isfinite(feas_tol) && feas_tol > 0 &&
      std::isfinite(comp_tol) && comp_tol > 0 &&
      primal >= 0 && comp > 100*std::max(initial_comp,comp_tol);
}
// A smaller positive barrier changes the centrality equation. Reopen exactly
// one auxiliary reset in that new epoch; equal/increasing/invalid parameters
// never replenish permission. Every attempted reset still consumes an outer
// slot in the original caller, and the barrier floor bounds the epoch count.
inline bool filter_recenter_new_barrier_epoch(bool& used,double current,double next) {
  if(!std::isfinite(current)||!std::isfinite(next)||!(current>0)||!(next>0)||!(next<current))
    return false;
  used=false;return true;
}
// At a fixed barrier, a new source-feasibility epoch must halve the last
// recentered ORIGINAL primal residual and still be above final tolerance.
// Thus repeated permission cannot come from the same anchor or arbitrarily
// tiny progress; at most logarithmically many such epochs precede feasibility.
inline bool filter_recenter_new_primal_epoch(bool& used,double reference,
    double current,double tolerance) {
  if(!used||!std::isfinite(reference)||!std::isfinite(current)||
      !std::isfinite(tolerance)||!(tolerance>0)||!(current>tolerance)||
      !(reference>0)||!(current<=.5*reference))return false;
  used=false;return true;
}
// Build first, commit second. No primal coordinates are passed or changed.
// Retain the initialization rule s=max(.01,mu,-g); z=mu/s for active rows.
// Inactive rows keep their old state. Failure leaves output vectors intact.
template<class Active>
inline bool make_filter_recenter_state(const std::vector<double>& g,
    const Active& active, const std::vector<double>& old_s,
    const std::vector<double>& old_z, double mu,
    std::vector<double>& new_s, std::vector<double>& new_z) {
  if(g.size()!=active.size() || g.size()!=old_s.size() ||
      g.size()!=old_z.size() || !std::isfinite(mu) || !(mu>0)) return false;
  auto s=old_s,z=old_z;
  for(std::size_t i=0;i<g.size();++i) if(active[i]) {
    if(!std::isfinite(g[i])) return false;
    s[i]=std::max(std::max(0.01,mu),-g[i]); z[i]=mu/s[i];
    if(!std::isfinite(s[i])||!std::isfinite(z[i])||s[i]<=0||z[i]<=0) return false;
  }
  new_s.swap(s);new_z.swap(z);return true;
}
}
