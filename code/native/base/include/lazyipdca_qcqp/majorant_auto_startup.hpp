#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
namespace lazyipdca::qcqp::detail {
// The numerical selector takes no problem name, pathname or benchmark label.
// It is intentionally conservative: a viable first-order model is necessary,
// not sufficient, so nonlinear recovery remains allowed after a failed solve.
struct AutoStartupFeatures {
  bool mixed = false;
  bool affine_nonconstant_objective = false;
  bool curved_equalities = false;
  bool infeasible_curved_equality = false;
  bool curved_inequalities = false;
  bool native_signed_squares = false;
  bool blind_violated_row = false;
  bool finite_data = true;
  double scaled_objective_gradient = 0.0;
  long double estimated_dense_restoration_bytes = 0.0L;
};
struct AutoStartupDecision {
  bool central_first = false;
  bool bounded_restoration_probe = false;
  double initial_mu = 1e-4;
  double central_fallback_mu = 1e-4;
  const char* reason = "legacy-structure";
};
inline AutoStartupDecision choose_auto_startup(const AutoStartupFeatures& f,
                                             double historical_mu) {
  AutoStartupDecision d;d.initial_mu=historical_mu;
  if(!std::isfinite(historical_mu)||historical_mu<=0 || !f.finite_data ||
     !std::isfinite(f.scaled_objective_gradient)) {d.reason="nonfinite-data";return d;}
  if(f.native_signed_squares) {d.reason="native-signed-square";return d;}
  if(!f.mixed || !f.affine_nonconstant_objective ||
     !f.curved_equalities || !f.curved_inequalities) return d;
  // Keep successful cold feasibility repairs.  Infeasibility alone does not
  // justify spending the entire optimization budget in Phase I: a mixed
  // affine model whose violated rows have linear sensitivity can also enter
  // the native primal-dual engine. Blind rows keep their historical recovery.
  if(f.infeasible_curved_equality) {
    // A wall deadline does not bound the first dense Jacobian allocation.
    // Keep the native first-order route when the ordinary residual model
    // alone would exceed the existing 128 MiB restoration workspace scale.
    // No benchmark name or variable-count threshold participates.
    if(!f.blind_violated_row&&f.estimated_dense_restoration_bytes>128.0L*1024*1024) {
      d.initial_mu=std::max(historical_mu,std::min(1.0,f.scaled_objective_gradient));
      d.central_first=true;d.reason="first-order-ready-dense-restoration-memory-guard";return d;
    }
    d.bounded_restoration_probe=!f.blind_violated_row;
    d.central_fallback_mu=std::max(historical_mu,
        std::min(1.0,f.scaled_objective_gradient));
    d.reason="infeasible-nonlinear-equality-restoration-first";return d;
  }
  if(f.blind_violated_row) {d.reason="violated-row-without-linear-sensitivity";return d;}
  // Balance the central products against the scaled objective gradient.
  // The upper cap is the same unit cap used by the existing adaptive barrier.
  // No primal coordinate, final tolerance, budget or backend is changed.
  d.initial_mu=std::max(historical_mu,std::min(1.0,f.scaled_objective_gradient));
  d.central_first=true;d.reason="mixed-affine-first-order-ready";return d;
}
// This is a subdivision of the remaining GLOBAL allowance, never a new
// budget. No bound is applied to legacy, constant-objective or blind-row
// startup paths. A failed probe supplies no primal point or certificate.
inline double startup_restoration_probe_seconds(double remaining) {
  return std::isfinite(remaining)&&remaining>0 ? std::min(30.0,0.1*remaining) : 0.0;
}
} // namespace
