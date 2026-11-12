#include "lazyipdca_qcqp/inner_ipopt.hpp"

#ifdef LAZYIPDCA_HAVE_IPOPT
#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpSolveStatistics.hpp>
#include <coin-or/IpTNLP.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace lazyipdca::qcqp {
namespace {

using Ipopt::Index;
using Ipopt::Number;

void add_dense_hessian(const ConvexQuadraticModel& q, double coeff, std::vector<double>& H, int n) {
  if (coeff == 0.0) return;
  for (const auto& e : q.base_hessian.entries) {
    H[static_cast<std::size_t>(e.i)*n + e.j] += coeff*e.v;
    if (e.i != e.j) H[static_cast<std::size_t>(e.j)*n + e.i] += coeff*e.v;
  }
  for (int j = 0; j < q.shift_dimension; ++j)
    H[static_cast<std::size_t>(j)*n + j] += coeff*q.scalar_shift;
  for (const auto& t : q.low_rank) {
    const double beta = coeff*t.beta;
    if (beta == 0.0) continue;
    for (int i = 0; i < n; ++i) {
      const double ui = t.direction[i];
      if (ui == 0.0) continue;
      for (int j = 0; j < n; ++j) {
        const double uj = t.direction[j];
        if (uj != 0.0) H[static_cast<std::size_t>(i)*n+j] += beta*ui*uj;
      }
    }
  }
}

class SubproblemTNLP final : public Ipopt::TNLP {
 public:
  explicit SubproblemTNLP(const ConvexSubproblem& p) : p_(p) {}

  bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag,
                    IndexStyleEnum& index_style) override {
    n = p_.nvar;
    m = static_cast<Index>(p_.equalities.rows.size() + p_.inequalities.size());
    nnz_jac_g = n*m; // dense contract: small/reference backend, robust and simple
    nnz_h_lag = n*(n+1)/2;
    index_style = C_STYLE;
    return true;
  }

  bool get_bounds_info(Index n, Number* x_l, Number* x_u,
                       Index m, Number* g_l, Number* g_u) override {
    const Number inf = 2e19;
    for (Index i = 0; i < n; ++i) { x_l[i] = -inf; x_u[i] = inf; }
    const Index me = static_cast<Index>(p_.equalities.rows.size());
    for (Index i = 0; i < me; ++i) {
      g_l[i] = p_.equalities.rhs[i];
      g_u[i] = p_.equalities.rhs[i];
    }
    for (Index i = me; i < m; ++i) {
      g_l[i] = -inf;
      g_u[i] = 0.0;
    }
    return true;
  }

  bool get_starting_point(Index n, bool init_x, Number* x,
                          bool init_z, Number*, Number*,
                          Index, bool init_lambda, Number*) override {
    if (!init_x || init_z || init_lambda) return false;
    std::fill(x, x+n, 0.0);
    return true;
  }

  bool eval_f(Index n, const Number* x, bool, Number& obj_value) override {
    Vec y(x, x+n);
    obj_value = p_.objective.value(y);
    return std::isfinite(obj_value);
  }

  bool eval_grad_f(Index n, const Number* x, bool, Number* grad_f) override {
    Vec y(x, x+n);
    Vec g = p_.objective.gradient(y);
    std::copy(g.begin(), g.end(), grad_f);
    return true;
  }

  bool eval_g(Index n, const Number* x, bool, Index m, Number* g) override {
    Vec y(x, x+n);
    Index row = 0;
    for (const auto& r : p_.equalities.rows) {
      double s = 0.0;
      for (const auto& [j,a] : r.entries) s += a*y[j];
      g[row++] = s;
    }
    for (const auto& q : p_.inequalities) g[row++] = q.value(y);
    return row == m;
  }

  bool eval_jac_g(Index n, const Number* x, bool, Index m, Index,
                  Index* iRow, Index* jCol, Number* values) override {
    if (!values) {
      Index k = 0;
      for (Index r = 0; r < m; ++r)
        for (Index c = 0; c < n; ++c) { iRow[k] = r; jCol[k] = c; ++k; }
      return true;
    }
    Vec y(x, x+n);
    Index k = 0;
    for (const auto& r : p_.equalities.rows) {
      Vec dense(n,0.0);
      for (const auto& [j,a] : r.entries) dense[j] = a;
      for (Index c = 0; c < n; ++c) values[k++] = dense[c];
    }
    for (const auto& q : p_.inequalities) {
      Vec gg = q.gradient(y);
      for (Index c = 0; c < n; ++c) values[k++] = gg[c];
    }
    return true;
  }

  bool eval_h(Index n, const Number*, bool, Number obj_factor,
              Index /*m*/, const Number* lambda, bool, Index,
              Index* iRow, Index* jCol, Number* values) override {
    if (!values) {
      Index k = 0;
      for (Index i = 0; i < n; ++i)
        for (Index j = 0; j <= i; ++j) { iRow[k]=i; jCol[k]=j; ++k; }
      return true;
    }
    std::vector<double> H(static_cast<std::size_t>(n)*n,0.0);
    add_dense_hessian(p_.objective, obj_factor, H, n);
    const Index me = static_cast<Index>(p_.equalities.rows.size());
    for (Index i = 0; i < static_cast<Index>(p_.inequalities.size()); ++i)
      add_dense_hessian(p_.inequalities[i], lambda[me+i], H, n);
    Index k=0;
    for (Index i=0;i<n;++i)
      for(Index j=0;j<=i;++j) values[k++]=H[static_cast<std::size_t>(i)*n+j];
    return true;
  }

  void finalize_solution(Ipopt::SolverReturn status, Index n, const Number* x,
                         const Number*, const Number*, Index m, const Number* g,
                         const Number* lambda, Number,
                         const Ipopt::IpoptData*, Ipopt::IpoptCalculatedQuantities*) override {
    status_ = status;
    x_.assign(x,x+n);
    g_.assign(g,g+m);
    lambda_.assign(lambda,lambda+m);
  }

  Ipopt::SolverReturn status() const { return status_; }
  const Vec& x() const { return x_; }
  const Vec& g() const { return g_; }
  const Vec& multipliers() const { return lambda_; }

 private:
  const ConvexSubproblem& p_;
  Ipopt::SolverReturn status_ = Ipopt::INTERNAL_ERROR;
  Vec x_, g_, lambda_;
};

} // namespace

InnerResult ConvexQCQPIpopt::solve(const ConvexSubproblem& p) const {
  InnerResult out;
  auto app = IpoptApplicationFactory();
  app->Options()->SetIntegerValue("print_level", st_.verbose ? 0 : 0);
  app->Options()->SetIntegerValue("max_iter", st_.max_inner_iterations);
  app->Options()->SetNumericValue("tol", st_.inner_tolerance);
  app->Options()->SetNumericValue("constr_viol_tol", st_.inner_tolerance);
  app->Options()->SetNumericValue("dual_inf_tol", std::max(1e-10, st_.inner_tolerance));
  app->Options()->SetNumericValue("compl_inf_tol", std::max(1e-8, st_.inner_iteration_limit_mu_fallback));
  app->Options()->SetStringValue("mu_strategy", "adaptive");
  app->Options()->SetStringValue("hessian_approximation", "exact");
  app->Options()->SetStringValue("nlp_scaling_method", "gradient-based");
  app->Options()->SetNumericValue("bound_relax_factor", 0.0);
  app->Options()->SetStringValue("sb", "yes");
  app->Options()->SetStringValue("print_timing_statistics", "no");
  auto init = app->Initialize();
  if (init != Ipopt::Solve_Succeeded) {
    out.message = "Ipopt initialization failed";
    return out;
  }

  Ipopt::SmartPtr<SubproblemTNLP> raw = new SubproblemTNLP(p);
  Ipopt::SmartPtr<Ipopt::TNLP> nlp = raw;
  const auto status = app->OptimizeTNLP(nlp);
  auto stats = app->Statistics();
  out.iterations = Ipopt::IsValid(stats) ? static_cast<int>(stats->IterationCount()) : 0;
  out.factorizations = out.iterations;
  out.y = raw->x();
  const int me = static_cast<int>(p.equalities.rows.size());
  if (static_cast<int>(raw->multipliers().size()) >= me) {
    out.equality_duals.assign(raw->multipliers().begin(), raw->multipliers().begin()+me);
    out.inequality_duals.assign(raw->multipliers().begin()+me, raw->multipliers().end());
  }

  // Recompute transparent residual diagnostics at returned point.
  if (static_cast<int>(out.y.size()) == p.nvar) {
    Vec grad = p.objective.gradient(out.y);
    for (int j=0;j<me && j<(int)out.equality_duals.size();++j) {
      for (const auto& [c,a] : p.equalities.rows[j].entries) grad[c] += a*out.equality_duals[j];
      double rr=-p.equalities.rhs[j];
      for (const auto& [c,a] : p.equalities.rows[j].entries) rr += a*out.y[c];
      out.equality_inf = std::max(out.equality_inf,std::abs(rr));
    }
    double mu_sum=0.0; int mu_n=0;
    for (int i=0;i<(int)p.inequalities.size();++i) {
      const double phi=p.inequalities[i].value(out.y);
      out.inequality_inf=std::max(out.inequality_inf,std::max(0.0,phi));
      if (i<(int)out.inequality_duals.size()) {
        Vec gg=p.inequalities[i].gradient(out.y);
        axpy(out.inequality_duals[i],gg,grad);
        mu_sum += std::max(0.0,-phi)*std::max(0.0,out.inequality_duals[i]);
        ++mu_n;
      }
    }
    out.dual_inf=inf_norm(grad);
    out.primal_residual_inf=std::max(out.equality_inf,out.inequality_inf);
    out.complementarity=mu_n?mu_sum/mu_n:0.0;
  }

  out.success = (status == Ipopt::Solve_Succeeded || status == Ipopt::Solved_To_Acceptable_Level || status == Ipopt::Feasible_Point_Found);
  out.message = out.success ? "convex QCQP solved by Ipopt" : "Ipopt convex QCQP failed with status " + std::to_string(static_cast<int>(status));
  return out;
}

} // namespace lazyipdca::qcqp

#else
namespace lazyipdca::qcqp {
InnerResult ConvexQCQPIpopt::solve(const ConvexSubproblem&) const {
  InnerResult out; out.message = "Ipopt backend was not compiled"; return out;
}
}
#endif
