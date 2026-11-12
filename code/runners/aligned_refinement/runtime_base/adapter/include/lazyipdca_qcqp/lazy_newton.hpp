#pragma once
// v0.4.28r3p28 — Lazy primal-dual Newton route (LazyPDN).
//
// A single-loop primal-dual interior-point Newton method that keeps the
// three LazyIPDCA ingredients requested for the case118 speed target:
//   1. Lazy working set: only near-active inequalities enter the KKT
//      system; the others are enforced exactly on the search segment by the
//      closed-form roots of their quadratic restriction and are activated
//      when they become blocking.
//   2. DC majorants for inequalities: an inequality row in the working set
//      contributes only the PSD part of its Hessian (tight spectral DC split),
//      so the inequality contribution to the model Hessian is always a valid
//      convex majorant.
//   3. Curvature-aware equalities: the signed Lagrangian curvature
//      sum_j y_j Q_j enters the KKT primal block, its tangent convexity is
//      certified by the factorization inertia (with minimal inertia
//      shifts), and the exact second-order equality residual
//      1/2 d'Q_j d — known analytically for quadratic equalities without any
//      re-evaluation — drives an exact quadratic correction step.
//
// Every accepted solution is certified on the ORIGINAL QCQP through the
// shared qcqp_ipopt_termination() helper, exactly like the Lazy-DCA route.
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/result.hpp"
#include "lazyipdca_qcqp/settings.hpp"

namespace lazyipdca::qcqp {

Result solve_atomic_newton(const QCQPProblem& problem, const Settings& st);
bool certify_lazy_newton_original(const QCQPProblem&, const Settings&, Result&);
Result solve_lazy_newton(const QCQPProblem& problem, const Settings& st);

} // namespace lazyipdca::qcqp
