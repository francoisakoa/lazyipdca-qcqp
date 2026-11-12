#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace lazyipdca::qcqp {
namespace {

void add_sym_triplets(std::vector<SymmetricTriplet>& K, const SymMatrix& Q, double coeff) {
  if (coeff == 0.0) return;
  for (const auto& e : Q.entries) K.push_back({e.i, e.j, coeff * e.v});
}

void add_row_transpose_product(const SparseRow& row, double coeff, Vec& y) {
  for (const auto& [j, a] : row.entries) y[j] += coeff * a;
}

double row_dot(const SparseRow& row, const Vec& x) {
  double s = 0.0;
  for (const auto& [j, a] : row.entries) s += a * x[j];
  return s;
}

bool finite_vec(const Vec& x) {
  for (double v : x) if (!std::isfinite(v)) return false;
  return true;
}

double max_abs(const Vec& x) {
  double m = 0.0;
  for (double v : x) m = std::max(m, std::abs(v));
  return m;
}

double max_step_positive(const Vec& x, const Vec& dx, double fraction) {
  double a = 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (dx[i] < 0.0) a = std::min(a, -fraction * x[i] / dx[i]);
  }
  return std::max(0.0, std::min(1.0, a));
}

struct NewtonStep {
  Vec dy;
  Vec dlambda;
  Vec dz;
  Vec ds;
};

} // namespace

double ConvexQuadraticModel::value(const Vec& y) const {
  if (static_cast<int>(y.size()) != static_cast<int>(linear.size()))
    throw std::invalid_argument("ConvexQuadraticModel::value: dimension mismatch");
  double v = constant + dot(linear, y) + 0.5 * base_hessian.quad(y);
  for (int i = 0; i < shift_dimension; ++i) v += 0.5 * scalar_shift * y[i] * y[i];
  for (const auto& t : low_rank) {
    const double a = dot(t.direction, y);
    v += 0.5 * t.beta * a * a;
  }
  return v;
}

Vec ConvexQuadraticModel::gradient(const Vec& y) const {
  Vec g = base_hessian.apply(y);
  axpy(1.0, linear, g);
  for (int i = 0; i < shift_dimension; ++i) g[i] += scalar_shift * y[i];
  for (const auto& t : low_rank) {
    const double a = t.beta * dot(t.direction, y);
    axpy(a, t.direction, g);
  }
  return g;
}

InnerResult ConvexQCQPInteriorPoint::solve(const ConvexSubproblem& p) const {
  InnerResult out;
  auto expired=[&](){return std::isfinite(st_.restoration_deadline_seconds)&&
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()>=st_.restoration_deadline_seconds;};
  if(expired()){out.message="convex restoration deadline before initialization";return out;}
  const int n = p.nvar;
  const int me = static_cast<int>(p.equalities.rows.size());
  const int mi = static_cast<int>(p.inequalities.size());
  if (n <= 0 || static_cast<int>(p.objective.linear.size()) != n ||
      static_cast<int>(p.equalities.rhs.size()) != me) {
    out.message = "invalid convex subproblem dimensions";
    return out;
  }

  Vec y(n, 0.0), lambda(me, 0.0), z(mi, 1.0), s(mi, 1.0);

  // SQP-to-IPM dual warm start.  Keep y=0 because the inner variable is the
  // SQP displacement, but reuse the previous outer multipliers when their
  // mapping is known.  Slacks are initialized from g(0)+s=0 and then pushed
  // strictly inside s>0; nonnegative duals are pushed inside z>0 as well.
  const bool use_eq_warm = st_.inner_dual_warm_start && p.has_dual_warm_start &&
      static_cast<int>(p.initial_equality_duals.size()) == me &&
      static_cast<int>(p.initial_inequality_duals.size()) == mi;
  const bool use_ineq_warm = use_eq_warm && st_.inner_inequality_dual_warm_start;
  if (use_eq_warm) lambda = p.initial_equality_duals;

  int init_pushes = 0;
  const bool feasible_centered_start = p.prefer_feasible_centered_start && !use_ineq_warm;
  if (feasible_centered_start) {
    // Lazy-DCA-specific initialization.  The inner variable is d and d=0 is
    // the current outer iterate.  Once Phase I is feasible, the exact DC
    // majorants are tangent there, hence d=0 is already feasible up to outer
    // tolerance.  Do not destroy that information by imposing the historical
    // s>=1,z=1 cold start.  Instead use the natural slack and place (s,z) on
    // a modest central-path level.  This path is deliberately local to the
    // explicit convex Lazy-DCA subproblem; restoration and legacy SQP retain
    // their historical initialization.
    const double target_mu = std::max(1e-4, 100.0 * st_.inner_tolerance);
    const double slack_push = std::max(1e-4, 10.0 * st_.inner_tolerance);
    for (int i = 0; i < mi; ++i) {
      const double phi = p.inequalities[i].value(y);
      const double raw_slack = std::max(0.0, -phi);
      s[i] = std::max(raw_slack, slack_push);
      if (s[i] > raw_slack) ++init_pushes;
      z[i] = std::clamp(target_mu / std::max(s[i], st_.minimum_positive),
                        st_.minimum_positive, 1e8);
    }
  } else if (!use_ineq_warm) {
    // Historical inequality-barrier cold-start contract.  This remains the
    // initialization for restoration/SQP auxiliary solves and ordinary cold
    // starts outside the explicit Lazy-DCA convex-majorant path.
    for (int i = 0; i < mi; ++i) {
      const double phi = p.inequalities[i].value(y);
      s[i] = std::max(1.0, 1.0 - phi);
      z[i] = 1.0;
    }
  } else {
    // Full primal-dual inequality warm start.  Bound push is meaningful only
    // when both the slack geometry and the mapped inequality dual are warm.
    for (int i = 0; i < mi; ++i) {
      const double phi = p.inequalities[i].value(y);
      const double raw_slack = std::max(0.0, -phi);
      const double scale = std::max(1.0, std::abs(phi));
      const double rel_ref = std::max(st_.inner_bound_push_reference, raw_slack);
      const double push = std::max(st_.minimum_positive,
          std::min(st_.inner_bound_push * scale, st_.inner_bound_frac * rel_ref));
      s[i] = std::max(raw_slack, push);
      if (s[i] > raw_slack) ++init_pushes;

      double zi = p.initial_inequality_duals[i];
      if (!(zi > 0.0) || !std::isfinite(zi))
        zi = st_.inner_bound_mult_init_val > 0.0 ? st_.inner_bound_mult_init_val : 1.0;
      const double zpush = std::max(st_.minimum_positive,
          std::min(st_.inner_bound_push * std::max(1.0, std::abs(zi)),
                   st_.inner_bound_frac * std::max(st_.inner_bound_push_reference, std::abs(zi))));
      z[i] = std::max(zi, zpush);
    }
  }
  out.dual_warm_start_used = use_eq_warm;
  out.initialization_bound_pushes = init_pushes;
  {
    Vec init_rpri(mi,0.0);
    for(int i=0;i<mi;++i) init_rpri[i]=p.inequalities[i].value(y)+s[i];
    out.initial_primal_residual_inf=max_abs(init_rpri);
    out.initial_complementarity=mi ? dot(s,z)/static_cast<double>(mi) : 0.0;
  }
  if(st_.verbose){
    std::cout << "[INNER-INIT] warm=" << (use_eq_warm?1:0)
              << " ineqwarm=" << (use_ineq_warm?1:0)
              << " centered=" << (feasible_centered_start?1:0)
              << " pushes=" << init_pushes
              << " pri=" << std::scientific << out.initial_primal_residual_inf
              << " mu=" << out.initial_complementarity
              << "\n" << std::defaultfloat;
  }

  // Cache affine inequality Jacobians once per convex subproblem.  In the
  // v0.2.12+ Phase-II linearized model this includes all modeled original
  // inequalities and all simple bounds, eliminating millions of temporary
  // Vec allocations across a CUTEst run.
  std::vector<Vec> J(mi);
  std::vector<char> affine(mi,0);
  std::vector<std::vector<std::pair<int,double>>> J_sparse(mi);
  for(int i=0;i<mi;++i){
    const auto& c=p.inequalities[i];
    const bool is_affine=c.base_hessian.entries.empty() && c.low_rank.empty() &&
        (c.shift_dimension==0 || c.scalar_shift==0.0);
    if(!is_affine) continue;
    affine[i]=1;
    J[i]=c.linear;
    auto& sp=J_sparse[i];
    sp.reserve(c.linear.size());
    for(int j=0;j<n;++j) if(c.linear[j]!=0.0) sp.push_back({j,c.linear[j]});
  }

  std::string linear_message;
  auto factor=make_symmetric_factorization(st_,linear_message);
  if(!factor){
    out.message="linear factorization backend unavailable ["+st_.linear_backend+"]: "+linear_message;
    return out;
  }

  const bool adaptive_kkt = st_.kkt_adaptive_regularization &&
      (st_.linear_backend=="native" || st_.linear_backend=="ldl" ||
       st_.linear_backend=="qdldl" || st_.linear_backend=="cholesk" ||
       st_.linear_backend=="mumps");
  double rho_x = adaptive_kkt
      ? std::max(st_.kkt_rho_x_floor,st_.kkt_primal_regularization)
      : st_.kkt_primal_regularization;
  double delta_boost = 1.0;
  out.final_kkt_rho_x=rho_x; out.max_kkt_rho_x=rho_x;

  for (int iter = 0; iter < st_.max_inner_iterations; ++iter) {
    if(expired()){out.message="convex restoration deadline";return out;}
    Vec grad_obj = p.objective.gradient(y);
    Vec rdual = grad_obj;
    Vec req(me, 0.0), phi(mi, 0.0), rpri(mi, 0.0);

    for (int j = 0; j < me; ++j) {
      req[j] = row_dot(p.equalities.rows[j], y) - p.equalities.rhs[j];
      add_row_transpose_product(p.equalities.rows[j], lambda[j], rdual);
    }
    for (int i = 0; i < mi; ++i) {
      phi[i] = p.inequalities[i].value(y);
      if(!affine[i]) J[i] = p.inequalities[i].gradient(y);
      rpri[i] = phi[i] + s[i];
      if(affine[i]){
        for(const auto& [j,a]:J_sparse[i]) rdual[j] += z[i]*a;
      } else {
        axpy(z[i], J[i], rdual);
      }
    }

    const double mu = mi ? dot(s, z) / static_cast<double>(mi) : 0.0;
    const double dual_inf = max_abs(rdual);
    const double eq_inf = max_abs(req);
    const double pri_inf = max_abs(rpri);
    double model_viol = 0.0;
    for (double v : phi) model_viol = std::max(model_viol, std::max(0.0, v));

    out.dual_inf = dual_inf;
    out.equality_inf = eq_inf;
    out.inequality_inf = model_viol;
    out.primal_residual_inf = pri_inf;
    out.complementarity = mu;
    out.iterations = iter;

    const double residual_inf = std::max({dual_inf, eq_inf, pri_inf});
    const double mu_tolerance=std::isfinite(p.complementarity_tolerance)?
        std::min(st_.inner_tolerance,std::max(0.0,p.complementarity_tolerance)):st_.inner_tolerance;
    const bool strict_convergence =
        residual_inf <= st_.inner_tolerance && mu <= mu_tolerance;
    const bool final_iteration_fallback =
        (iter + 1 == st_.max_inner_iterations) &&
        residual_inf <= st_.inner_tolerance &&
        mu <= std::min(st_.inner_iteration_limit_mu_fallback,mu_tolerance);

    if (strict_convergence || final_iteration_fallback) {
      if(expired()){out.message="convex restoration deadline before acceptance";return out;}
      out.success = true;
      out.used_iteration_limit_fallback = final_iteration_fallback && !strict_convergence;
      out.message = out.used_iteration_limit_fallback
          ? "convex QCQP accepted at inner iteration limit: residuals converged and complementarity passed guarded fallback"
          : "convex QCQP primal-dual residuals converged";
      out.y = std::move(y);
      out.equality_duals = std::move(lambda);
      out.inequality_duals = std::move(z);
      out.inequality_slacks = std::move(s);
      out.symbolic_analyses=factor->symbolic_analyses();
      out.numeric_scatter_builds=factor->numeric_scatter_builds();
      out.numeric_scatter_reuses=factor->numeric_scatter_reuses();
      out.numeric_scatter_invalidations=
          factor->numeric_scatter_invalidations();
      return out;
    }

    // Collect positive-semidefinite low-rank Hessian terms for bordered KKT.
    struct Border { double gamma; const Vec* u; };
    std::vector<Border> borders;
    for (const auto& t : p.objective.low_rank)
      if (t.beta > 1e-18) borders.push_back({t.beta, &t.direction});
    for (int i = 0; i < mi; ++i) {
      if (z[i] <= 1e-18) continue;
      for (const auto& t : p.inequalities[i].low_rank) {
        const double gamma = z[i] * t.beta;
        if (gamma > 1e-18) borders.push_back({gamma, &t.direction});
      }
    }

    const int nb = static_cast<int>(borders.size());
    const int N = n + me + mi + nb;
    std::vector<SymmetricTriplet> K;
    K.reserve(p.objective.base_hessian.entries.size() + static_cast<std::size_t>(N) +
              static_cast<std::size_t>(me + mi + nb) * 8);

    // Lagrangian Hessian without low-rank terms.
    add_sym_triplets(K, p.objective.base_hessian, 1.0);
    for (int j = 0; j < p.objective.shift_dimension; ++j)
      K.push_back({j, j, p.objective.scalar_shift});
    for (int i = 0; i < mi; ++i) {
      if (z[i] <= 1e-18) continue;
      add_sym_triplets(K, p.inequalities[i].base_hessian, z[i]);
      for (int j = 0; j < p.inequalities[i].shift_dimension; ++j)
        K.push_back({j, j, z[i] * p.inequalities[i].scalar_shift});
    }

    // Equality Jacobian.
    const int eq0 = n;
    for (int r = 0; r < me; ++r)
      for (const auto& [j, a] : p.equalities.rows[r].entries)
        if (a != 0.0) K.push_back({j, eq0+r, a});

    // Inequality Jacobian and diagonal complementarity block -S/Z.
    const int in0 = n + me;
    for (int i = 0; i < mi; ++i) {
      if(affine[i]){
        for(const auto& [j,a]:J_sparse[i]) K.push_back({j,in0+i,a});
      } else {
        for (int j = 0; j < n; ++j) {
          const double a = J[i][j];
          if (a != 0.0) K.push_back({j, in0+i, a});
        }
      }
      const double zi = std::max(z[i], st_.minimum_positive);
      K.push_back({in0+i, in0+i, -s[i] / zi});
    }

    // Exact low-rank border: [K U; U^T -Gamma^{-1}].
    const int b0 = n + me + mi;
    for (int b = 0; b < nb; ++b) {
      const Vec& u = *borders[b].u;
      for (int j = 0; j < n; ++j) {
        const double a = u[j];
        if (a != 0.0) K.push_back({j, b0+b, a});
      }
      K.push_back({b0+b, b0+b, -1.0 / borders[b].gamma});
    }

    // Historical LazyIPDCA numerical stabilization is deliberately kept
    // outside the SQP/QCQP model.  rho_x regularizes only the primal Newton
    // block.  delta_d regularizes equality and inequality multipliers, not the
    // low-rank border variables.
    double dual_scale=max_abs(lambda);
    dual_scale=std::max(dual_scale,max_abs(z));
    double delta_d=st_.kkt_dual_regularization;
    if(adaptive_kkt){
      delta_d=st_.kkt_delta_dual_inverse
          ? st_.kkt_delta_dual_base/(1.0+dual_scale)
          : st_.kkt_delta_dual_base;
      delta_d=std::clamp(delta_d,st_.kkt_delta_dual_floor,st_.kkt_delta_dual_max);
      delta_d=std::min(st_.kkt_delta_dual_max,delta_d*delta_boost);
    }

    bool factor_ok=false;
    std::string factor_failure;
    for(int attempt=0;attempt<=std::max(0,st_.kkt_regularization_retry_max);++attempt){
      std::vector<SymmetricTriplet> Kreg=K;
      if(adaptive_kkt){
        for(int j=0;j<n;++j) Kreg.push_back({j,j,rho_x});
        for(int j=0;j<me;++j) Kreg.push_back({eq0+j,eq0+j,-delta_d});
        for(int i=0;i<mi;++i) Kreg.push_back({in0+i,in0+i,-delta_d});
      }
      linear_message.clear();
      ++out.factorizations;
      if(expired()){out.message="convex restoration deadline before factor";return out;}
      const bool factored=factor->factor(N,n,Kreg,linear_message);
      if(expired()){out.message="convex restoration deadline after factor";return out;}
      if(factored){ factor_ok=true; break; }
      factor_failure=linear_message;
      if(!adaptive_kkt || attempt==st_.kkt_regularization_retry_max) break;
      ++out.kkt_regularization_retries;
      const double old_rho=rho_x, old_delta=delta_d;
      rho_x=std::min(st_.kkt_rho_x_max,
          std::max(st_.kkt_rho_x_floor,
                   rho_x>0.0 ? rho_x*st_.kkt_rho_x_growth : st_.kkt_rho_x_floor));
      delta_boost=std::min(st_.kkt_delta_dual_max/std::max(delta_d,st_.kkt_delta_dual_floor),
                           delta_boost*st_.kkt_rho_x_growth);
      delta_d=std::min(st_.kkt_delta_dual_max,delta_d*st_.kkt_rho_x_growth);
      if(st_.verbose){
        std::cout << "[KKT-REG] iter="<<iter<<" attempt="<<attempt+1
                  << " reason=factor rho_x="<<std::scientific<<old_rho<<"->"<<rho_x
                  << " delta_d="<<old_delta<<"->"<<delta_d
                  << " msg="<<factor_failure<<"\n"<<std::defaultfloat;
      }
    }
    out.symbolic_analyses=factor->symbolic_analyses();
    out.numeric_scatter_builds=factor->numeric_scatter_builds();
    out.numeric_scatter_reuses=factor->numeric_scatter_reuses();
    out.numeric_scatter_invalidations=
        factor->numeric_scatter_invalidations();
    out.final_kkt_rho_x=rho_x;
    out.max_kkt_rho_x=std::max(out.max_kkt_rho_x,rho_x);
    out.final_kkt_delta_d=delta_d;
    out.max_kkt_delta_d=std::max(out.max_kkt_delta_d,delta_d);
    if(!factor_ok){
      out.message = "KKT factorization failed [" + st_.linear_backend + "]: " + factor_failure;
      return out;
    }

    auto solve_step = [&](const Vec& rcent) -> std::pair<bool, NewtonStep> {
      Vec rhs(N, 0.0);
      for (int j = 0; j < n; ++j) rhs[j] = -rdual[j];
      for (int j = 0; j < me; ++j) rhs[eq0+j] = -req[j];
      for (int i = 0; i < mi; ++i) {
        const double zi = std::max(z[i], st_.minimum_positive);
        rhs[in0+i] = rcent[i] / zi - rpri[i];
      }
      // border RHS remains zero.
      std::string solve_message;
      if (!factor->solve(rhs, solve_message)) {
        out.max_kkt_backward_error=std::max(out.max_kkt_backward_error,factor->last_backward_error());
        out.max_kkt_residual_inf=std::max(out.max_kkt_residual_inf,factor->last_residual_inf());
        out.kkt_refinement_steps += factor->last_refinement_steps();
        out.message = "KKT solve failed [" + st_.linear_backend + "]: " + solve_message;
        return {false, {}};
      }
      if(expired()){out.message="convex restoration deadline after solve";return {false,{}};}
      out.max_kkt_backward_error=std::max(out.max_kkt_backward_error,factor->last_backward_error());
      out.max_kkt_residual_inf=std::max(out.max_kkt_residual_inf,factor->last_residual_inf());
      out.kkt_refinement_steps += factor->last_refinement_steps();
      if(st_.verbose && adaptive_kkt &&
         (factor->last_backward_error()>st_.kkt_backward_error_good ||
          factor->last_refinement_steps()>0)){
        std::cout << "[KKT-SOLVE] iter="<<iter
                  << " rho_x="<<std::scientific<<rho_x
                  << " delta_d="<<delta_d
                  << " resid="<<factor->last_residual_inf()
                  << " berr="<<factor->last_backward_error()
                  << " refine="<<factor->last_refinement_steps()
                  << "\n"<<std::defaultfloat;
      }

      NewtonStep d;
      d.dy.assign(rhs.begin(), rhs.begin()+n);
      d.dlambda.assign(rhs.begin()+eq0, rhs.begin()+eq0+me);
      d.dz.assign(rhs.begin()+in0, rhs.begin()+in0+mi);
      d.ds.assign(mi, 0.0);
      for (int i = 0; i < mi; ++i) {
        double jd=0.0;
        if(affine[i]) for(const auto& [j,a]:J_sparse[i]) jd += a*d.dy[j];
        else jd=dot(J[i],d.dy);
        d.ds[i] = -rpri[i] - jd;
      }
      return {finite_vec(d.dy) && finite_vec(d.dz) && finite_vec(d.ds), std::move(d)};
    };

    Vec rcent_aff(mi, 0.0);
    for (int i = 0; i < mi; ++i) rcent_aff[i] = s[i] * z[i];
    auto [ok_aff, aff] = solve_step(rcent_aff);
    if (!ok_aff) {
      out.message = "affine predictor KKT solve failed";
      return out;
    }

    double alpha_aff_pri = mi ? max_step_positive(s, aff.ds, 1.0) : 1.0;
    double alpha_aff_dual = mi ? max_step_positive(z, aff.dz, 1.0) : 1.0;
    double mu_aff = 0.0;
    if (mi) {
      for (int i = 0; i < mi; ++i)
        mu_aff += (s[i] + alpha_aff_pri*aff.ds[i]) * (z[i] + alpha_aff_dual*aff.dz[i]);
      mu_aff /= static_cast<double>(mi);
    }
    double sigma = 0.0;
    if (mi && mu > 0.0) {
      const double ratio = std::clamp(mu_aff / mu, 0.0, 1.0);
      sigma = ratio * ratio * ratio;
      // When primal and dual residuals are already solved but complementarity
      // alone stalls, a nearly-unit Mehrotra sigma can keep the method on an
      // unnecessarily high barrier level indefinitely.  In the explicit
      // Lazy-DCA convex subproblem force a genuine barrier reduction.
      const double centrality_trigger = std::max(1e-5, 100.0 * st_.inner_tolerance);
      if (p.prefer_feasible_centered_start && residual_inf <= centrality_trigger &&
          mu > st_.inner_tolerance) {
        sigma = std::min(sigma, 0.10);
      }
    }

    Vec rcent(mi, 0.0);
    for (int i = 0; i < mi; ++i)
      rcent[i] = s[i]*z[i] + aff.ds[i]*aff.dz[i] - sigma*mu;
    auto [ok, d] = solve_step(rcent);
    if (!ok) {
      out.message = "corrector KKT solve failed";
      return out;
    }

    const double alpha_pri = mi ? max_step_positive(s, d.ds, st_.fraction_to_boundary) : 1.0;
    const double alpha_dual = mi ? max_step_positive(z, d.dz, st_.fraction_to_boundary) : 1.0;
    const double alpha_y = std::min(1.0, alpha_pri);

    axpy(alpha_y, d.dy, y);
    axpy(alpha_dual, d.dlambda, lambda); // equality duals are free; dual damping is adequate
    for (int i = 0; i < mi; ++i) {
      s[i] = std::max(st_.minimum_positive, s[i] + alpha_pri*d.ds[i]);
      z[i] = std::max(st_.minimum_positive, z[i] + alpha_dual*d.dz[i]);
    }

    if (!finite_vec(y) || !finite_vec(lambda) || !finite_vec(z) || !finite_vec(s)) {
      out.message = "non-finite primal-dual iterate in convex QCQP solver";
      return out;
    }
  }

  out.message = "convex QCQP inner iteration limit reached";
  out.y = std::move(y);
  out.equality_duals = std::move(lambda);
  out.inequality_duals = std::move(z);
  out.inequality_slacks = std::move(s);
  out.iterations = st_.max_inner_iterations;
  out.symbolic_analyses=factor->symbolic_analyses();
  out.numeric_scatter_builds=factor->numeric_scatter_builds();
  out.numeric_scatter_reuses=factor->numeric_scatter_reuses();
  out.numeric_scatter_invalidations=factor->numeric_scatter_invalidations();
  return out;
}

} // namespace lazyipdca::qcqp
