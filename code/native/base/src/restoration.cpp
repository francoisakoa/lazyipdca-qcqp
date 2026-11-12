#include "lazyipdca_qcqp/native_restoration_trace.hpp"
#include "restoration_restart_policy.hpp"
#include "failure_recovery_policy.hpp"
#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <iostream>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

extern "C" {
void dgesv_(const int*, const int*, double*, const int*, int*, double*, const int*, int*);
}

namespace lazyipdca::qcqp {

bool restoration_exact_sparse_row_dots_build_compatible() {
#if defined(__FAST_MATH__) || \
    (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
  return false;
#else
  return true;
#endif
}

namespace {

bool restoration_expired(const Settings& st) {
  return std::isfinite(st.restoration_deadline_seconds) &&
      std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()
          >= st.restoration_deadline_seconds;
}

struct ResidualModel {
  Vec residual;
  std::vector<Vec> rows;
  std::vector<const QuadraticFunction*> functions;
};

struct AbsLink {
  int equality_index = -1;
  int source = -1;
  int auxiliary = -1;
};

struct NewtonDirection {
  Vec d;
  bool success = false;
  bool exact_hessian = false;
  bool matrix_free = false;
  int linear_solves = 0;
  int cg_iterations = 0;
  double shift = 0.0;
};

// Storage shared by all exact-Hessian PCG solves of one local Phase-I run.
// Only allocation and scheduling change; all trajectory-defining scalar
// operations retain their established order.
struct ExactPcgWorkspace {
  Vec full;
  Vec yf;
  Vec hv;
  Vec lifted;
  Vec candidate_yf;
  Vec candidate_hv;
  Vec candidate_lifted;
  Vec b;
  Vec z;
  Vec r;
  Vec q;
  Vec pvec;
  Vec mz;
  Vec diag_full;
  Vec precond;
  Vec row_dots;
  Vec candidate_row_dots;
  int growths = 0;
  int matvecs = 0;
  int parallel_row_dot_batches = 0;
  int parallel_teams = 0;
  int sparse_row_dot_batches = 0;
  int sparse_row_dot_verifications = 0;
  int sparse_row_dot_fallbacks = 0;

  void ensure(int n, int nf, int rows) {
    bool grew = false;
    auto fit = [&](Vec& values, int size) {
      if (static_cast<int>(values.size()) != size) {
        values.resize(static_cast<std::size_t>(size));
        grew = true;
      }
    };
    fit(full,n);
    fit(yf,n);
    fit(hv,n);
    fit(lifted,n);
    fit(b,nf);
    fit(z,nf);
    fit(r,nf);
    fit(q,nf);
    fit(pvec,nf);
    fit(mz,nf);
    fit(diag_full,n);
    fit(precond,nf);
    fit(row_dots,rows);
    fit(candidate_row_dots,rows);
    if(grew) ++growths;
  }
};

// Shared by every exact PCG solve and restart belonging to one Phase-I call.
// A single bytewise disagreement permanently selects the historical dense Jv
// kernel for the remainder of that call.  This is deliberately separate from
// the allocation workspace so the safety latch also works when workspace
// reuse is disabled.
struct ExactSparseRowDotRuntimeState {
  bool dense_latched = false;
};

std::vector<AbsLink> detect_abs_links(const QCQPProblem& p) {
  std::vector<AbsLink> links;
  constexpr double tol = 1e-12;
  for (int eq_index = 0; eq_index < static_cast<int>(p.equalities.size()); ++eq_index) {
    const auto& h = p.equalities[eq_index];
    if (h.name.rfind("abs_link_", 0) != 0) continue;
    if (std::abs(h.constant) > tol || inf_norm(h.linear) > tol) continue;
    std::vector<SymTriplet> diag;
    bool offdiag = false;
    for (const auto& e : h.hessian.entries) {
      if (std::abs(e.v) <= tol) continue;
      if (e.i != e.j) { offdiag = true; break; }
      diag.push_back(e);
    }
    if (offdiag || diag.size() != 2) continue;
    const SymTriplet* pos = nullptr;
    const SymTriplet* neg = nullptr;
    for (const auto& e : diag) {
      if (e.v > 0.0) pos = &e;
      if (e.v < 0.0) neg = &e;
    }
    if (!pos || !neg) continue;
    const double scale = std::max({1.0, std::abs(pos->v), std::abs(neg->v)});
    if (std::abs(pos->v + neg->v) > 1e-10 * scale) continue;
    const int aux = pos->i;
    const int src = neg->i;
    if (!(std::isfinite(p.lb[aux]) && p.lb[aux] >= -tol)) continue;
    links.push_back({eq_index, src, aux});
  }
  return links;
}

void project_abs_links(const QCQPProblem& p, const std::vector<AbsLink>& links, Vec& x) {
  for (const auto& link : links) {
    double t = std::abs(x[link.source]);
    if (std::isfinite(p.lb[link.auxiliary])) t = std::max(t, p.lb[link.auxiliary]);
    if (std::isfinite(p.ub[link.auxiliary])) t = std::min(t, p.ub[link.auxiliary]);
    x[link.auxiliary] = t;
  }
}

std::vector<char> auxiliary_abs_mask(const QCQPProblem& p, const std::vector<AbsLink>& links) {
  std::vector<char> mask(p.n, 0);
  for (const auto& link : links) mask[link.auxiliary] = 1;
  return mask;
}

Vec clamp_to_bounds(const QCQPProblem& p, Vec x) {
  for (int j = 0; j < p.n; ++j) {
    if (std::isfinite(p.lb[j])) x[j] = std::max(x[j], p.lb[j]);
    if (std::isfinite(p.ub[j])) x[j] = std::min(x[j], p.ub[j]);
  }
  return x;
}

std::vector<char> fixed_mask(const QCQPProblem& p, double tol) {
  std::vector<char> fixed(p.n, 0);
  for (int j = 0; j < p.n; ++j) {
    fixed[j] = std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]) &&
               std::abs(p.ub[j] - p.lb[j]) <= tol;
  }
  return fixed;
}

double abs_sign(double q) {
  return q > 0.0 ? 1.0 : (q < 0.0 ? -1.0 : 0.0);
}

Vec reduce_gradient(Vec grad, const Vec& x, const std::vector<AbsLink>& links) {
  for (const auto& link : links) {
    const double s = abs_sign(x[link.source]);
    grad[link.source] += s * grad[link.auxiliary];
    grad[link.auxiliary] = 0.0;
  }
  return grad;
}

ResidualModel residual_model(const QCQPProblem& p, const Vec& x,
                             const std::vector<AbsLink>& links) {
  ResidualModel out;
  std::vector<char> skip_eq(p.equalities.size(), 0);
  for (const auto& link : links) {
    if (link.equality_index >= 0 && link.equality_index < static_cast<int>(skip_eq.size()))
      skip_eq[link.equality_index] = 1;
  }

  out.residual.reserve(p.equalities.size() + p.inequalities.size());
  out.rows.reserve(p.equalities.size() + p.inequalities.size());
  out.functions.reserve(p.equalities.size() + p.inequalities.size());

  for (int i = 0; i < static_cast<int>(p.equalities.size()); ++i) {
    if (skip_eq[i]) continue;
    const auto& h = p.equalities[i];
    out.residual.push_back(h.value(x));
    out.rows.push_back(links.empty() ? h.gradient(x) : reduce_gradient(h.gradient(x), x, links));
    out.functions.push_back(&h);
  }
  for (const auto& g : p.inequalities) {
    const double v = g.value(x);
    if (v > 0.0) {
      out.residual.push_back(v);
      out.rows.push_back(links.empty() ? g.gradient(x) : reduce_gradient(g.gradient(x), x, links));
      out.functions.push_back(&g);
    } else {
      out.residual.push_back(0.0);
      out.rows.emplace_back(p.n, 0.0);
      out.functions.push_back(&g);
    }
  }
  return out;
}

double squared_merit(const ResidualModel& rm) {
  return 0.5 * dot(rm.residual, rm.residual);
}

Vec merit_gradient(const ResidualModel& rm, int n) {
  Vec g(n, 0.0);
  for (int i = 0; i < static_cast<int>(rm.residual.size()); ++i) {
    const double r = rm.residual[i];
    if (r == 0.0) continue;
    axpy(r, rm.rows[i], g);
  }
  return g;
}

// Only the established sparse row-space LM path uses sparse storage.  Dense
// small-problem and abs-reduced paths keep their historical arithmetic.
using LegacySparseRow = std::vector<std::pair<int,double>>;
struct LegacyResidualJacobian {
  Vec residual;
  std::vector<Vec> rows;
  std::vector<LegacySparseRow> sparse_rows;
  bool sparse = false;
};

// One O(n) scratch vector is reused across all rows.  Clear only coordinates
// touched by the preceding row, preserve Hessian-triplet accumulation order,
// then add linear and signed-square terms in the model's established order.
// Sorted coordinates preserve the original left-to-right reduction order.
struct LegacySparseGradientWorkspace {
  Vec values;
  std::vector<char> marked;
  std::vector<int> touched;

  explicit LegacySparseGradientWorkspace(int n)
      : values(static_cast<std::size_t>(n),0.0),
        marked(static_cast<std::size_t>(n),0) {}

  LegacySparseRow gradient(const QuadraticFunction& q, const Vec& x) {
    for(int j:touched) { values[j]=0.0; marked[j]=0; }
    touched.clear();
    auto add=[&](int j,double v) {
      if(!marked[j]) { marked[j]=1; touched.push_back(j); }
      values[j]+=v;
    };
    for(const auto& e:q.hessian.entries) {
      add(e.i,e.v*x[e.j]);
      if(e.i!=e.j) add(e.j,e.v*x[e.i]);
    }
    q.for_each_linear_nonzero([&](int j,double v) {add(j,v);});
    for(const auto& e:q.signed_square)
      add(e.first,2*e.second*std::abs(x[e.first]));
    std::sort(touched.begin(),touched.end());
    LegacySparseRow result;
    result.reserve(touched.size());
    for(int j:touched) if(values[j]!=0.0) result.push_back({j,values[j]});
    return result;
  }
};

LegacyResidualJacobian legacy_residual_jacobian(const QCQPProblem& p, const Vec& x,
                                               bool sparse=false) {
  LegacyResidualJacobian out;
  out.residual.reserve(p.equalities.size() + p.inequalities.size());
  out.sparse=sparse;
  if(sparse) out.sparse_rows.reserve(p.equalities.size()+p.inequalities.size());
  else out.rows.reserve(p.equalities.size()+p.inequalities.size());
  LegacySparseGradientWorkspace workspace(sparse?p.n:0);
  for (const auto& h : p.equalities) {
    out.residual.push_back(h.value(x));
    if(sparse) out.sparse_rows.push_back(workspace.gradient(h,x));
    else out.rows.push_back(h.gradient(x));
  }
  for (const auto& g : p.inequalities) {
    const double v = g.value(x);
    if (v > 0.0) {
      out.residual.push_back(v);
      if(sparse) out.sparse_rows.push_back(workspace.gradient(g,x));
      else out.rows.push_back(g.gradient(x));
    } else {
      out.residual.push_back(0.0);
      if(sparse) out.sparse_rows.emplace_back();
      else out.rows.emplace_back(p.n, 0.0);
    }
  }
  return out;
}

// On a structured abs-lift manifold t=|q|, the auxiliary variables are not
// independent restoration variables.  The correct Phase-I Jacobian is the
// Jacobian of the reduced residual c(q,|q|,...), i.e.
//
//   d c / d q = c_q + sign(q) c_t,
//
// away from q=0.  At q=0 we use the Clarke-compatible value 0 for the
// derivative of |q|.  The explicit equations t^2-q^2=0 are omitted from the
// least-squares system because projection enforces them to machine precision.
// This makes the LM model live in the same space in which trial points are
// evaluated, instead of computing a 217-D step and then projecting it to a
// 157-D manifold.
LegacyResidualJacobian legacy_residual_jacobian_abs_reduced(const QCQPProblem& p, const Vec& x,
                                                const std::vector<AbsLink>& links) {
  LegacyResidualJacobian out;
  out.residual.reserve(p.equalities.size() + p.inequalities.size() - links.size());
  out.rows.reserve(p.equalities.size() + p.inequalities.size() - links.size());

  std::vector<char> skip_eq(p.equalities.size(), 0);
  for (const auto& link : links) {
    if (link.equality_index >= 0 &&
        link.equality_index < static_cast<int>(skip_eq.size()))
      skip_eq[link.equality_index] = 1;
  }

  auto reduce_gradient = [&](Vec grad) {
    for (const auto& link : links) {
      const double q = x[link.source];
      const double s = q > 0.0 ? 1.0 : (q < 0.0 ? -1.0 : 0.0);
      grad[link.source] += s * grad[link.auxiliary];
      grad[link.auxiliary] = 0.0;
    }
    return grad;
  };

  for (int i = 0; i < static_cast<int>(p.equalities.size()); ++i) {
    if (skip_eq[i]) continue;
    const auto& h = p.equalities[i];
    out.residual.push_back(h.value(x));
    out.rows.push_back(reduce_gradient(h.gradient(x)));
  }
  for (const auto& g : p.inequalities) {
    const double v = g.value(x);
    if (v > 0.0) {
      out.residual.push_back(v);
      out.rows.push_back(reduce_gradient(g.gradient(x)));
    } else {
      out.residual.push_back(0.0);
      out.rows.emplace_back(p.n, 0.0);
    }
  }
  return out;
}

double legacy_squared_merit(const LegacyResidualJacobian& rj) {
  return 0.5 * dot(rj.residual, rj.residual);
}

// Trial acceptance uses residual values only.  Building trial Jacobians here
// previously kept up to three dense m-by-n matrices alive simultaneously.
double legacy_squared_merit_values(const QCQPProblem& p, const Vec& x,
                                  const std::vector<AbsLink>& links) {
  std::vector<char> skip_eq;
  if(!links.empty()) {
    skip_eq.assign(p.equalities.size(),0);
    for(const auto& link:links)
      if(link.equality_index>=0 &&
         link.equality_index<static_cast<int>(skip_eq.size()))
        skip_eq[link.equality_index]=1;
  }
  double value=0.0;
  for(std::size_t i=0;i<p.equalities.size();++i) {
    if(!skip_eq.empty() && skip_eq[i]) continue;
    const double r=p.equalities[i].value(x);
    value+=r*r;
  }
  for(const auto& g:p.inequalities) {
    const double v=g.value(x);
    const double r=v>0.0?v:0.0;
    value+=r*r;
  }
  return 0.5*value;
}

double legacy_row_dot(const LegacyResidualJacobian& rj, int row, const Vec& v) {
  if(!rj.sparse) return dot(rj.rows[row],v);
  double value=0.0;
  for(const auto& e:rj.sparse_rows[row]) value+=e.second*v[e.first];
  return value;
}


bool solve_dense_system(std::vector<double> A, Vec b, Vec& x) {
  const int n = static_cast<int>(b.size());
  if (n == 0) { x.clear(); return true; }
  if (static_cast<int>(A.size()) != n*n) return false;
  std::vector<int> piv(n, 0);
  int nrhs = 1, info = 0;
  {NativeRestorationCall counted(NativeRestorationOperation::Dense);
  dgesv_(&n, &nrhs, A.data(), &n, piv.data(), b.data(), &n, &info);
  }
  if (info != 0) return false;
  x = std::move(b);
  for (double v : x) if (!std::isfinite(v)) return false;
  return true;
}

// Sparse row-space Levenberg--Marquardt direction for large equality-only
// restoration models.  CATENA/CATENARY have thousands of variables but only
// about one third as many short, chain-structured equalities.  Forming the
// historical n-by-n normal matrix destroys that sparsity.  Instead solve
//
//   (D J J^T D + damping I) lambda = D r,
//   d = -J^T D lambda,
//
// with the selected symmetric backend.  D equilibrates Jacobian rows.  The
// factor object is persistent, so LDL+METIS reuses its symbolic analysis while
// the numerical Jacobian values evolve.
bool sparse_underdetermined_equality_candidate(const QCQPProblem& p,
                                             int minimum_dimension) {
  const std::size_t m=p.equalities.size();
  if(!p.inequalities.empty() || m<8 || p.n<minimum_dimension ||
      m>=static_cast<std::size_t>(p.n))
    return false;
  // Structural support proxy.  The row-space path is aimed at short-chain
  // equations; avoid replacing the exact-Hessian restoration on large dense
  // nonlinear rows merely because m<n.
  constexpr std::size_t max_average_support=64;
  const std::size_t limit=max_average_support*m;
  std::size_t support=0;
  for(const auto& h:p.equalities) {
    h.for_each_linear_nonzero([&](int,double) {++support;});
    support+=h.signed_square.size();
    if(support>limit) return false;
    for(const auto& e:h.hessian.entries) {
      support += (e.i==e.j ? 1u : 2u);
      if(support>limit) return false;
    }
  }
  return support<=limit;
}

bool large_sparse_equality_chain_candidate(const QCQPProblem& p) {
  return sparse_underdetermined_equality_candidate(p,512);
}

// Near a feasible point, a large sparse model may have only a handful of
// violated inequalities in addition to a sparse, underdetermined equality
// Jacobian.  Forming and repeatedly applying the full exact merit Hessian is
// especially wasteful in that regime: inactive inequalities are represented
// by zero residual/Jacobian rows, while a row-space Gauss--Newton step can
// preserve all equalities and repair the few active inequalities directly.
//
// This gate is deliberately narrower than the general restoration route.  It
// applies only to medium/large sparse models, only from a genuinely
// near-feasible point, and only while the nonzero row system is
// underdetermined.  A failed probe falls back to the established exact-PCG
// restoration from the original point.
bool near_feasible_sparse_inequality_candidate(
    const QCQPProblem& p, const Vec& x, const Settings& st) {
  if(p.inequalities.empty() || p.n<512 || p.equalities.size()<8) return false;
  const double theta=feasibility(p,x).total_inf;
  const double gate=std::max(1e-2,1000.0*st.feasibility_tolerance);
  if(!(theta>st.feasibility_tolerance) || !std::isfinite(theta) || theta>gate)
    return false;

  const auto fixed=fixed_mask(p,st.fixed_bound_tolerance);
  const std::size_t free_count=static_cast<std::size_t>(
      std::count(fixed.begin(),fixed.end(),static_cast<char>(0)));
  std::size_t active_rows=p.equalities.size();
  std::size_t support=0;
  constexpr std::size_t max_average_support=64;

  auto add_support=[&](const QuadraticFunction& q) {
    q.for_each_linear_nonzero([&](int,double) {++support;});
    support+=q.signed_square.size();
    for(const auto& e:q.hessian.entries)
      if(e.v!=0.0) support+=(e.i==e.j ? 1u : 2u);
  };
  for(const auto& h:p.equalities) add_support(h);
  for(const auto& g:p.inequalities) {
    if(g.value(x)<=0.0) continue;
    ++active_rows;
    add_support(g);
  }
  if(active_rows<8 || active_rows>=free_count) return false;
  return support<=max_average_support*active_rows;
}

struct DisjointSet {
  explicit DisjointSet(int n) : parent(static_cast<std::size_t>(n)),
      rank(static_cast<std::size_t>(n),0) {
    std::iota(parent.begin(),parent.end(),0);
  }
  int find(int x) {
    int r=x;
    while(parent[static_cast<std::size_t>(r)]!=r)
      r=parent[static_cast<std::size_t>(r)];
    while(parent[static_cast<std::size_t>(x)]!=x) {
      const int p=parent[static_cast<std::size_t>(x)];
      parent[static_cast<std::size_t>(x)]=r;
      x=p;
    }
    return r;
  }
  void unite(int a,int b) {
    a=find(a); b=find(b);
    if(a==b) return;
    if(rank[static_cast<std::size_t>(a)]<rank[static_cast<std::size_t>(b)])
      std::swap(a,b);
    parent[static_cast<std::size_t>(b)]=a;
    if(rank[static_cast<std::size_t>(a)]==rank[static_cast<std::size_t>(b)])
      ++rank[static_cast<std::size_t>(a)];
  }
  std::vector<int> parent;
  std::vector<unsigned char> rank;
};

struct TangentActivationCandidate {
  bool attempted=false;
  bool accepted=false;
  int components=0;
  Vec x;
  double before=std::numeric_limits<double>::infinity();
  double after=std::numeric_limits<double>::infinity();
  double objective_before=std::numeric_limits<double>::infinity();
  double objective_after=std::numeric_limits<double>::infinity();
  double step_inf=0.0;
  int gradient_rows=0;
  int line_quadratics=0;
  int candidates=0;
};

TangentActivationCandidate objective_tangent_activation(
    const QCQPProblem& p, const Vec& start, const Settings& st);

struct HarmonicInitialization {
  bool attempted=false;
  bool accepted=false;
  int factorizations=0;
  Vec x;
  double before=std::numeric_limits<double>::infinity();
  double after=std::numeric_limits<double>::infinity();
  bool tangent_attempted=false;
  bool tangent_accepted=false;
  int tangent_components=0;
  double tangent_before=std::numeric_limits<double>::infinity();
  double tangent_after=std::numeric_limits<double>::infinity();
  double tangent_step_inf=0.0;
  double tangent_objective_before=std::numeric_limits<double>::infinity();
  double tangent_objective_after=std::numeric_limits<double>::infinity();
  double harmonic_seconds=0.0;
  double tangent_seconds=0.0;
  int graph_edges=0;
  int unknowns=0;
  int tangent_gradient_rows=0;
  int tangent_line_quadratics=0;
  int tangent_candidates=0;
  std::string message;
};

// Build a scale-free graph from off-diagonal quadratic couplings and
// harmonically interpolate components that have at least two fixed-bound
// anchors.  This is deliberately structural rather than name-specific.  It
// repairs pathological CUTEst starts such as CATENARY (all interior X values
// initialized at the same coordinate while the far endpoint is fixed) without
// changing components that have insufficient boundary information.
HarmonicInitialization anchored_harmonic_initialization(
    const QCQPProblem& p, const Vec& start, const Settings& st) {
  HarmonicInitialization out;
  const auto harmonic_t0=std::chrono::steady_clock::now();
  out.x=start;
  out.before=feasibility(p,start).total_inf;
  if(!large_sparse_equality_chain_candidate(p)||restoration_expired(st)) return out;
  out.attempted=true;

  using EdgeKey=std::pair<int,int>;
  std::map<EdgeKey,double> edge_weight;
  DisjointSet dsu(p.n);
  const double edge_floor=64.0*std::numeric_limits<double>::epsilon();
  for(const auto& h:p.equalities) {
    for(const auto& e:h.hessian.entries) {
      if(e.i==e.j || !std::isfinite(e.v) || std::abs(e.v)<=edge_floor) continue;
      const int a=std::min(e.i,e.j), b=std::max(e.i,e.j);
      edge_weight[{a,b}]+=std::abs(e.v);
      dsu.unite(a,b);
    }
  }
  out.graph_edges=static_cast<int>(edge_weight.size());
  if(edge_weight.empty()) {
    out.message="no off-diagonal quadratic coupling graph";
    return out;
  }

  const auto fixed=fixed_mask(p,st.fixed_bound_tolerance);
  std::vector<int> anchor_count(static_cast<std::size_t>(p.n),0);
  for(int j=0;j<p.n;++j)
    if(fixed[static_cast<std::size_t>(j)])
      ++anchor_count[static_cast<std::size_t>(dsu.find(j))];

  std::vector<int> unknown_index(static_cast<std::size_t>(p.n),-1);
  std::vector<int> unknown;
  unknown.reserve(static_cast<std::size_t>(p.n));
  for(int j=0;j<p.n;++j) {
    const int root=dsu.find(j);
    if(!fixed[static_cast<std::size_t>(j)] &&
       anchor_count[static_cast<std::size_t>(root)]>=2) {
      unknown_index[static_cast<std::size_t>(j)]=static_cast<int>(unknown.size());
      unknown.push_back(j);
    }
  }
  const int nu=static_cast<int>(unknown.size());
  out.unknowns=nu;
  if(nu==0) {
    out.message="no multiply anchored quadratic component";
    return out;
  }

  std::map<std::pair<int,int>,double> laplacian;
  Vec rhs(static_cast<std::size_t>(nu),0.0);
  for(const auto& [key,wraw]:edge_weight) {
    const int u=key.first, v=key.second;
    const int root=dsu.find(u);
    if(anchor_count[static_cast<std::size_t>(root)]<2) continue;
    const double w=std::max(edge_floor,wraw);
    const int iu=unknown_index[static_cast<std::size_t>(u)];
    const int iv=unknown_index[static_cast<std::size_t>(v)];
    if(iu>=0) {
      laplacian[{iu,iu}]+=w;
      if(iv>=0) laplacian[{std::min(iu,iv),std::max(iu,iv)}]-=w;
      else if(fixed[static_cast<std::size_t>(v)]) rhs[static_cast<std::size_t>(iu)]+=w*p.lb[v];
    }
    if(iv>=0) {
      laplacian[{iv,iv}]+=w;
      if(iu<0 && fixed[static_cast<std::size_t>(u)]) rhs[static_cast<std::size_t>(iv)]+=w*p.lb[u];
    }
  }

  std::vector<SymmetricTriplet> entries;
  entries.reserve(laplacian.size());
  double max_diag=1.0;
  for(const auto& [key,value]:laplacian) {
    if(value==0.0) continue;
    entries.push_back({key.first,key.second,value});
    if(key.first==key.second) max_diag=std::max(max_diag,std::abs(value));
  }
  if(entries.empty()) {
    out.message="empty anchored Laplacian";
    return out;
  }

  std::string message;
  auto factor=make_symmetric_factorization(st,message);
  if(!factor) {
    out.message="harmonic factor backend unavailable: "+message;
    return out;
  }
  if(restoration_expired(st))return out;
  bool solved=factor->factor(nu,nu,entries,message) && !restoration_expired(st) && factor->solve(rhs,message);
  ++out.factorizations;
  if(!solved && !restoration_expired(st)) {
    const double shift=1e-12*max_diag;
    for(int i=0;i<nu;++i) entries.push_back({i,i,shift});
    rhs.assign(static_cast<std::size_t>(nu),0.0);
    for(const auto& [key,wraw]:edge_weight) {
      const int u=key.first, v=key.second;
      const int root=dsu.find(u);
      if(anchor_count[static_cast<std::size_t>(root)]<2) continue;
      const double w=std::max(edge_floor,wraw);
      const int iu=unknown_index[static_cast<std::size_t>(u)];
      const int iv=unknown_index[static_cast<std::size_t>(v)];
      if(iu>=0 && iv<0 && fixed[static_cast<std::size_t>(v)])
        rhs[static_cast<std::size_t>(iu)]+=w*p.lb[v];
      if(iv>=0 && iu<0 && fixed[static_cast<std::size_t>(u)])
        rhs[static_cast<std::size_t>(iv)]+=w*p.lb[u];
    }
    solved=factor->factor(nu,nu,entries,message) && factor->solve(rhs,message);
    ++out.factorizations;
  }
  if(!solved) {
    out.message="anchored harmonic solve failed: "+message;
    return out;
  }

  Vec candidate=start;
  for(int i=0;i<nu;++i) candidate[static_cast<std::size_t>(unknown[static_cast<std::size_t>(i)])]=
      rhs[static_cast<std::size_t>(i)];
  candidate=clamp_to_bounds(p,std::move(candidate));
  for(int j=0;j<p.n;++j) if(fixed[static_cast<std::size_t>(j)]) candidate[j]=p.lb[j];
  out.after=feasibility(p,candidate).total_inf;
  const double noise=4096.0*std::numeric_limits<double>::epsilon()*
      std::max(1.0,out.before);
  if(std::isfinite(out.after) && out.after+noise<out.before) {
    out.x=std::move(candidate);
    out.accepted=true;
    out.message="anchored harmonic initialization improved feasibility";
  } else {
    out.message="anchored harmonic initialization did not improve feasibility";
  }

  out.harmonic_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-harmonic_t0).count();
  const Vec activation_start=out.accepted?out.x:start;
  const auto tangent_t0=std::chrono::steady_clock::now();
  const auto activation=objective_tangent_activation(p,activation_start,st);
  out.tangent_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-tangent_t0).count();
  out.tangent_attempted=activation.attempted;
  out.tangent_accepted=activation.accepted;
  out.tangent_components=activation.components;
  out.tangent_before=activation.before;
  out.tangent_after=activation.after;
  out.tangent_step_inf=activation.step_inf;
  out.tangent_objective_before=activation.objective_before;
  out.tangent_objective_after=activation.objective_after;
  out.tangent_gradient_rows=activation.gradient_rows;
  out.tangent_line_quadratics=activation.line_quadratics;
  out.tangent_candidates=activation.candidates;
  if(activation.accepted) {
    out.x=activation.x;
    out.after=activation.after;
    out.accepted=true;
    out.message="anchored harmonic plus objective tangent activation improved feasibility";
  }
  return out;
}


// Break an initial first-order degeneracy of a quadratic equality component.
// If a one-anchor component has a nonzero objective gradient but its equality
// Jacobian columns are all (numerically) zero, no multiplier can represent the
// objective gradient at that point.  A uniform objective step does not help a
// squared-difference chain because it keeps every edge difference zero.
//
// We instead use the graph distance from the real fixed anchor as a temporary
// virtual-anchor mode.  On a chain this is a linear ramp, hence every
// difference is nonzero.  Its scalar amplitude is selected by evaluating the
// exact quadratic equality residuals along the mode; no new constraint or
// permanent artificial anchor is introduced.
TangentActivationCandidate objective_tangent_activation(
    const QCQPProblem& p, const Vec& start, const Settings& st) {
  TangentActivationCandidate out;
  out.x=start;
  out.before=feasibility(p,start).total_inf;
  out.after=out.before;
  out.objective_before=p.objective.value(start);
  out.objective_after=out.objective_before;
  if(!st.restoration_tangent_activation ||
     !large_sparse_equality_chain_candidate(p)) return out;
  out.attempted=true;

  using EdgeKey=std::pair<int,int>;
  const double eps=std::numeric_limits<double>::epsilon();
  const double edge_floor=64.0*eps;
  std::map<EdgeKey,double> edge_weight;
  DisjointSet dsu(p.n);
  for(const auto& h:p.equalities) {
    for(const auto& e:h.hessian.entries) {
      if(e.i==e.j || !std::isfinite(e.v) || std::abs(e.v)<=edge_floor) continue;
      const int a=std::min(e.i,e.j), b=std::max(e.i,e.j);
      edge_weight[{a,b}]+=std::abs(e.v);
      dsu.unite(a,b);
    }
  }
  if(edge_weight.empty()) return out;

  std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(p.n));
  for(const auto& [key,w]:edge_weight) {
    if(!(w>edge_floor)) continue;
    adjacency[static_cast<std::size_t>(key.first)].push_back(key.second);
    adjacency[static_cast<std::size_t>(key.second)].push_back(key.first);
  }

  const auto fixed=fixed_mask(p,st.fixed_bound_tolerance);
  std::vector<int> anchor_count(static_cast<std::size_t>(p.n),0);
  std::vector<int> anchor(static_cast<std::size_t>(p.n),-1);
  std::map<int,std::vector<int>> components;
  for(int j=0;j<p.n;++j) {
    const int root=dsu.find(j);
    components[root].push_back(j);
    if(fixed[static_cast<std::size_t>(j)]) {
      ++anchor_count[static_cast<std::size_t>(root)];
      anchor[static_cast<std::size_t>(root)]=j;
    }
  }

  const Vec objective_gradient=p.objective.gradient(start);
  const double objective_scale=std::max(1.0,inf_norm(objective_gradient));
  const double gradient_tolerance=4096.0*eps*objective_scale;

  Vec jacobian_column_inf(static_cast<std::size_t>(p.n),0.0);
  double jacobian_scale=1.0;
  for(const auto& h:p.equalities) {
    const Vec row=h.gradient(start);
    ++out.gradient_rows;
    for(int j=0;j<p.n;++j) {
      const double a=std::abs(row[static_cast<std::size_t>(j)]);
      jacobian_column_inf[static_cast<std::size_t>(j)]=
          std::max(jacobian_column_inf[static_cast<std::size_t>(j)],a);
      jacobian_scale=std::max(jacobian_scale,a);
    }
  }
  const double jacobian_tolerance=16384.0*eps*jacobian_scale;

  Vec mode(static_cast<std::size_t>(p.n),0.0);
  for(const auto& [root,nodes]:components) {
    if(anchor_count[static_cast<std::size_t>(root)]!=1) continue;
    const int a0=anchor[static_cast<std::size_t>(root)];
    if(a0<0 || adjacency[static_cast<std::size_t>(a0)].empty()) continue;

    int significant=0, degenerate=0;
    double objective_mass=0.0, degenerate_mass=0.0;
    for(int j:nodes) {
      if(fixed[static_cast<std::size_t>(j)]) continue;
      const double gj=std::abs(objective_gradient[static_cast<std::size_t>(j)]);
      if(gj<=gradient_tolerance) continue;
      ++significant;
      objective_mass+=gj;
      if(jacobian_column_inf[static_cast<std::size_t>(j)]<=jacobian_tolerance) {
        ++degenerate;
        degenerate_mass+=gj;
      }
    }
    if(significant<2 || degenerate<2 ||
       degenerate_mass<0.5*std::max(objective_mass,gradient_tolerance)) continue;

    std::vector<int> distance(static_cast<std::size_t>(p.n),-1);
    std::queue<int> q;
    distance[static_cast<std::size_t>(a0)]=0;
    q.push(a0);
    while(!q.empty()) {
      const int u=q.front(); q.pop();
      for(int v:adjacency[static_cast<std::size_t>(u)]) {
        if(dsu.find(v)!=root || distance[static_cast<std::size_t>(v)]>=0) continue;
        distance[static_cast<std::size_t>(v)]=
            distance[static_cast<std::size_t>(u)]+1;
        q.push(v);
      }
    }

    double directional_dot=0.0;
    int farthest=0;
    for(int j:nodes) {
      if(fixed[static_cast<std::size_t>(j)]) continue;
      const int dj=distance[static_cast<std::size_t>(j)];
      if(dj<=0) continue;
      farthest=std::max(farthest,dj);
      directional_dot+=objective_gradient[static_cast<std::size_t>(j)]*
          static_cast<double>(dj);
    }
    const double dot_tolerance=4096.0*eps*objective_scale*
        std::max(1.0,static_cast<double>(farthest)*nodes.size());
    if(farthest<=0 || std::abs(directional_dot)<=dot_tolerance) continue;
    const double orientation=directional_dot>0.0?-1.0:1.0;
    for(int j:nodes) {
      if(fixed[static_cast<std::size_t>(j)]) continue;
      const int dj=distance[static_cast<std::size_t>(j)];
      if(dj>0) mode[static_cast<std::size_t>(j)]=
          orientation*static_cast<double>(dj);
    }

    double max_edge=0.0;
    for(const auto& [key,w]:edge_weight) {
      if(!(w>edge_floor) || dsu.find(key.first)!=root) continue;
      max_edge=std::max(max_edge,
          std::abs(mode[static_cast<std::size_t>(key.first)]-
                   mode[static_cast<std::size_t>(key.second)]));
    }
    if(!(max_edge>0.0) || !std::isfinite(max_edge)) {
      for(int j:nodes) mode[static_cast<std::size_t>(j)]=0.0;
      continue;
    }
    for(int j:nodes) mode[static_cast<std::size_t>(j)]/=max_edge;
    ++out.components;
  }
  if(out.components==0 || inf_norm(mode)==0.0) return out;

  double t_bound=std::numeric_limits<double>::infinity();
  for(int j=0;j<p.n;++j) {
    const double dj=mode[static_cast<std::size_t>(j)];
    if(dj>0.0 && std::isfinite(p.ub[static_cast<std::size_t>(j)]))
      t_bound=std::min(t_bound,
          (p.ub[static_cast<std::size_t>(j)]-start[static_cast<std::size_t>(j)])/dj);
    else if(dj<0.0 && std::isfinite(p.lb[static_cast<std::size_t>(j)]))
      t_bound=std::min(t_bound,
          (p.lb[static_cast<std::size_t>(j)]-start[static_cast<std::size_t>(j)])/dj);
  }
  if(!(t_bound>0.0)) return out;

  struct LineQuadratic { double a=0.0,b=0.0,c=0.0; };
  std::vector<LineQuadratic> line;
  line.reserve(p.equalities.size());
  std::vector<double> roots;
  roots.reserve(2*p.equalities.size());
  double coefficient_scale=1.0;
  for(const auto& h:p.equalities) {
    LineQuadratic qline;
    qline.c=h.value(start);
    qline.b=dot(h.gradient(start),mode);
    qline.a=0.5*h.hessian.quad(mode);
    coefficient_scale=std::max({coefficient_scale,std::abs(qline.a),
                                std::abs(qline.b),std::abs(qline.c)});
    line.push_back(qline);
    ++out.line_quadratics;
  }
  const double coefficient_tolerance=4096.0*eps*coefficient_scale;
  auto add_root=[&](double t) {
    if(std::isfinite(t) && t>0.0 &&
       (!std::isfinite(t_bound) || t<=t_bound*(1.0+1e-12)))
      roots.push_back(std::isfinite(t_bound)?std::min(t,t_bound):t);
  };
  for(const auto& qline:line) {
    if(std::abs(qline.a)>coefficient_tolerance) {
      const double disc=qline.b*qline.b-4.0*qline.a*qline.c;
      if(disc>=-coefficient_tolerance*coefficient_scale) {
        const double sd=std::sqrt(std::max(0.0,disc));
        add_root((-qline.b-sd)/(2.0*qline.a));
        add_root((-qline.b+sd)/(2.0*qline.a));
      }
    } else if(std::abs(qline.b)>coefficient_tolerance) {
      add_root(-qline.c/qline.b);
    }
  }
  std::sort(roots.begin(),roots.end());
  roots.erase(std::unique(roots.begin(),roots.end(),[&](double a,double b) {
    return std::abs(a-b)<=1e-12*std::max({1.0,std::abs(a),std::abs(b)});
  }),roots.end());

  std::vector<double> candidates;
  candidates.reserve(2*roots.size()+16);
  auto add_candidate=[&](double t) {
    if(!std::isfinite(t) || !(t>0.0)) return;
    if(std::isfinite(t_bound)) t=std::min(t,t_bound);
    if(t>0.0) candidates.push_back(t);
  };
  for(double t:roots) add_candidate(t);
  for(std::size_t i=1;i<roots.size();++i)
    add_candidate(0.5*(roots[i-1]+roots[i]));
  if(!roots.empty()) {
    const double lo=roots.front(), hi=roots.back();
    const double med=roots[roots.size()/2];
    add_candidate(0.5*(lo+hi));
    add_candidate(std::sqrt(0.5*(lo*lo+hi*hi)));
    add_candidate(0.5*med);
    add_candidate(med);
    add_candidate(1.5*med);
  } else {
    add_candidate(0.25); add_candidate(0.5);
    add_candidate(1.0); add_candidate(2.0);
  }
  std::sort(candidates.begin(),candidates.end());
  candidates.erase(std::unique(candidates.begin(),candidates.end(),
      [&](double a,double b) {
        return std::abs(a-b)<=1e-12*std::max({1.0,std::abs(a),std::abs(b)});
      }),candidates.end());
  out.candidates=static_cast<int>(candidates.size());

  const double objective_linear=dot(objective_gradient,mode);
  const double objective_quadratic=0.5*p.objective.hessian.quad(mode);
  double best_t=0.0;
  double best_theta=out.before;
  double best_merit=std::numeric_limits<double>::infinity();
  for(double t:candidates) {
    if(restoration_expired(st))break;
    double theta=0.0, merit=0.0;
    for(const auto& qline:line) {
      const double r=(qline.a*t+qline.b)*t+qline.c;
      theta=std::max(theta,std::abs(r));
      merit+=r*r;
    }
    const double obj=out.objective_before+t*objective_linear+
        t*t*objective_quadratic;
    const double objective_noise=4096.0*eps*
        std::max({1.0,std::abs(out.objective_before),std::abs(obj)});
    if(obj>out.objective_before+objective_noise) continue;
    const double theta_noise=4096.0*eps*std::max(1.0,best_theta);
    if(theta<best_theta-theta_noise ||
       (std::abs(theta-best_theta)<=theta_noise && merit<best_merit)) {
      best_t=t;
      best_theta=theta;
      best_merit=merit;
    }
  }
  if(!(best_t>0.0)) return out;

  Vec candidate=start;
  axpy(best_t,mode,candidate);
  candidate=clamp_to_bounds(p,std::move(candidate));
  for(int j=0;j<p.n;++j) if(fixed[static_cast<std::size_t>(j)])
    candidate[static_cast<std::size_t>(j)]=p.lb[static_cast<std::size_t>(j)];
  out.after=feasibility(p,candidate).total_inf;
  const double noise=4096.0*eps*std::max(1.0,out.before);
  if(std::isfinite(out.after) && out.after+noise<out.before) {
    out.x=std::move(candidate);
    out.objective_after=p.objective.value(out.x);
    out.step_inf=best_t*inf_norm(mode);
    out.accepted=true;
  }
  return out;
}

bool sparse_row_space_lm_direction(
    const LegacyResidualJacobian& rj, const std::vector<int>& free,
    double damping, SymmetricFactorization& factor, Vec& d,
    int full_dimension, std::string& message,
    bool compact_structurally_zero_rows=false,
    bool preserve_dense_columns=true,
    bool require_underdetermined=true) {
  const int original_m=static_cast<int>(rj.residual.size());
  if(original_m<=0 || free.empty() ||
     static_cast<int>(rj.sparse?rj.sparse_rows.size():rj.rows.size())!=original_m)
    return false;

  std::vector<char> is_free;
  if(rj.sparse) {
    is_free.assign(static_cast<std::size_t>(full_dimension),0);
    for(int j:free) is_free[j]=1;
  }
  std::vector<int> active_rows;
  active_rows.reserve(static_cast<std::size_t>(original_m));
  for(int a=0;a<original_m;++a) {
    bool structurally_nonzero=!compact_structurally_zero_rows;
    if(compact_structurally_zero_rows) {
      if(rj.sparse) {
        for(const auto& e:rj.sparse_rows[a]) if(is_free[e.first] && e.second!=0.0) {
          structurally_nonzero=true;
          break;
        }
      } else {
        for(int j:free) if(rj.rows[a][j]!=0.0) {
          structurally_nonzero=true;
          break;
        }
      }
    }
    if(structurally_nonzero) active_rows.push_back(a);
    else if(rj.residual[a]!=0.0) return false;
  }
  const int m=static_cast<int>(active_rows.size());
  if(m<=0 || (compact_structurally_zero_rows && require_underdetermined &&
              m>=static_cast<int>(free.size()))) return false;

  Vec row_scale(m,1.0);
  std::vector<std::vector<std::pair<int,double>>> by_variable(
      static_cast<std::size_t>(full_dimension));
  for(int a=0;a<m;++a) {
    const int row=active_rows[static_cast<std::size_t>(a)];
    double n2=0.0;
    if(rj.sparse) {
      for(const auto& e:rj.sparse_rows[row]) if(is_free[e.first] && e.second!=0.0)
        n2+=e.second*e.second;
    } else {
      for(int j:free) {
        const double v=rj.rows[static_cast<std::size_t>(row)][j];
        if(v!=0.0) n2+=v*v;
      }
    }
    row_scale[a]=std::max(1.0,std::sqrt(std::max(0.0,n2)));
  }
  if(rj.sparse) {
    for(int a=0;a<m;++a) {
      const int row=active_rows[static_cast<std::size_t>(a)];
      for(const auto& e:rj.sparse_rows[row]) if(is_free[e.first]) {
        const double v=e.second/row_scale[a];
        if(v!=0.0) by_variable[e.first].push_back({a,v});
      }
    }
  } else {
    for(int j:free) {
      auto& col=by_variable[static_cast<std::size_t>(j)];
      for(int a=0;a<m;++a) {
        const int row=active_rows[static_cast<std::size_t>(a)];
        const double v=rj.rows[static_cast<std::size_t>(row)][j]/row_scale[a];
        if(v!=0.0) col.push_back({a,v});
      }
    }
  }

  // A short sparse row does not imply a sparse row Gram matrix: a column
  // shared by q constraints creates q(q+1)/2 triplets. Preserve high-degree
  // columns U as a negative identity border instead of expanding their cliques:
  //
  // [ A_local A_local^T + shift I   U ] [lambda] = [D r]
  // [             U^T             -I ] [   y  ]   [ 0 ].
  //
  // Eliminating y gives exactly the historical LM system. The inertia is
  // (m positive, border_size negative), so every selected symmetric backend
  // retains its normal inertia and solve-residual checks. No backend fallback,
  // changed damping, modified starting point, or problem-specific detection.
  constexpr std::size_t dense_column_degree=64;
  std::vector<char> bordered(static_cast<std::size_t>(full_dimension),0);
  int border_size=0;
  std::size_t estimate=static_cast<std::size_t>(m);
  for(int j:free) {
    const std::size_t q=by_variable[static_cast<std::size_t>(j)].size();
    if(preserve_dense_columns && q>=dense_column_degree) {
      bordered[j]=1;
      ++border_size;
      estimate+=q+1;
    } else estimate+=q*(q+1)/2;
  }
  std::vector<SymmetricTriplet> normal;
  normal.reserve(estimate);
  int border=m;
  for(int j:free) {
    const auto& col=by_variable[static_cast<std::size_t>(j)];
    if(bordered[j]) {
      for(const auto& [a,v]:col) normal.push_back({a,border,v});
      normal.push_back({border,border,-1.0});
      ++border;
    } else {
      for(std::size_t a=0;a<col.size();++a)
        for(std::size_t b=0;b<=a;++b)
          normal.push_back({col[b].first,col[a].first,
                            col[b].second*col[a].second});
    }
  }
  const double diagonal_shift=std::max(
      1e-15,damping*std::max(1.0,static_cast<double>(free.size())/
                                      std::max(1,m)));
  for(int a=0;a<m;++a) normal.push_back({a,a,diagonal_shift});

  Vec lambda(m+border_size,0.0);
  for(int a=0;a<m;++a) {
    const int row=active_rows[static_cast<std::size_t>(a)];
    lambda[a]=rj.residual[static_cast<std::size_t>(row)]/row_scale[a];
  }
  if(!factor.factor(m+border_size,m,normal,message) || !factor.solve(lambda,message))
    return false;

  d.assign(static_cast<std::size_t>(full_dimension),0.0);
  for(int j:free) {
    double value=0.0;
    for(const auto& [a,v]:by_variable[static_cast<std::size_t>(j)])
      value+=v*lambda[a];
    d[j]=-value;
  }
  return std::isfinite(inf_norm(d));
}

// Apply T^T H T, where T is the local tangent map of t=|q|.
// Allocation-free counterpart of SymMatrix::apply used by the matrix-free
// Phase-I PCG loop.  The triplets and coordinates are visited in exactly the
// same order as the historical implementation; only the storage is retained
// between matvecs.
void reduced_hessian_apply_into(const QuadraticFunction& q, const Vec& x, const Vec& v,
                                const std::vector<AbsLink>& links,
                                Vec& lifted, Vec& y) {
  const auto& H=q.hessian;
  const Vec* input = &v;
  if (!links.empty()) {
    lifted = v;
    for (const auto& link : links)
      lifted[static_cast<std::size_t>(link.auxiliary)] =
          abs_sign(x[static_cast<std::size_t>(link.source)]) *
          v[static_cast<std::size_t>(link.source)];
    input = &lifted;
  }
  y.resize(static_cast<std::size_t>(H.n));
  std::fill(y.begin(), y.end(), 0.0);
  for (const auto& e : H.entries) {
    y[static_cast<std::size_t>(e.i)] +=
        e.v * (*input)[static_cast<std::size_t>(e.j)];
    if (e.i != e.j)
      y[static_cast<std::size_t>(e.j)] +=
          e.v * (*input)[static_cast<std::size_t>(e.i)];
  }
  for(auto e:q.signed_square)y[e.first]+=2*e.second*abs_sign(x[e.first])*(*input)[e.first];
  if (links.empty()) return;
  for (const auto& link : links) {
    y[static_cast<std::size_t>(link.source)] +=
        abs_sign(x[static_cast<std::size_t>(link.source)]) *
        y[static_cast<std::size_t>(link.auxiliary)];
    y[static_cast<std::size_t>(link.auxiliary)] = 0.0;
  }
}

void transformed_curvature_gershgorin(const ResidualModel& rm, const Vec& x,
                                      const std::vector<AbsLink>& links,
                                      Vec& diag, Vec& radius) {
  const int n = static_cast<int>(x.size());
  diag.assign(n, 0.0);
  radius.assign(n, 0.0);
  std::vector<int> map(n);
  Vec scale(n, 1.0);
  std::iota(map.begin(), map.end(), 0);
  for (const auto& link : links) {
    map[link.auxiliary] = link.source;
    scale[link.auxiliary] = abs_sign(x[link.source]);
  }

  for (int k = 0; k < static_cast<int>(rm.residual.size()); ++k) {
    const double w = rm.residual[k];
    if (w == 0.0) continue;
    const auto& H = rm.functions[k]->hessian;
    for(auto e:rm.functions[k]->signed_square)
      diag[map[e.first]]+=w*2*e.second*abs_sign(x[e.first])*scale[e.first]*scale[e.first];
    for (const auto& e : H.entries) {
      const int a = map[e.i];
      const int b = map[e.j];
      const double v = w * e.v * scale[e.i] * scale[e.j];
      if (e.i == e.j) {
        diag[a] += v;
      } else if (a == b) {
        diag[a] += 2.0*v;
      } else {
        radius[a] += std::abs(v);
        radius[b] += std::abs(v);
      }
    }
  }
}

using SparseRowPattern = std::vector<std::vector<int>>;

SparseRowPattern zero_skipping_pattern(const ResidualModel& rm,
                                       std::size_t dimension,
                                       bool& worthwhile) {
  SparseRowPattern pattern(rm.rows.size());
  std::size_t nonzeros = 0;
  for (std::size_t i = 0; i < rm.rows.size(); ++i) {
    const auto& row = rm.rows[i];
    auto& indices = pattern[i];
    indices.reserve(std::min<std::size_t>(row.size(), 64));
    for (std::size_t j = 0; j < row.size(); ++j) {
      if (row[j] != 0.0) {
        indices.push_back(static_cast<int>(j));
        ++nonzeros;
      }
    }
  }
  const std::size_t dense = rm.rows.size() * dimension;
  worthwhile = dense > 0 && nonzeros * 4 < dense;
  return pattern;
}

bool binary_equal(const Vec& a, const Vec& b) {
  return a.size() == b.size() &&
      (a.empty() || std::memcmp(a.data(), b.data(),
                                a.size() * sizeof(double)) == 0);
}

// The indices are collected by scanning the original dense row from left to
// right, so this performs precisely the same nonzero multiplications and
// additions, in precisely the same order, as dot(row,v).  The caller still
// validates the first complete row-dot batch byte-for-byte before selecting
// this path; this is a runtime guard against platform-specific signed-zero or
// floating-point-environment surprises.
double sparse_row_dot_preserving_order(const Vec& row,
                                       const std::vector<int>& indices,
                                       const Vec& v) {
  double value=0.0;
  for(int j:indices)
    value+=row[static_cast<std::size_t>(j)]*
        v[static_cast<std::size_t>(j)];
  return value;
}

// Complete the exact merit-Hessian product after the independent J-row dots
// have been computed.  The J^T(Jv) accumulation is deliberately serial and
// uses the same row and coordinate order as the historical fused kernel.
void exact_merit_hessian_finish_from_row_dots(
    const ResidualModel& rm, const Vec& x,
    const std::vector<AbsLink>& links, const Vec& v, double shift,
    const SparseRowPattern* sparse_axpy_rows, const Vec& row_dots,
    Vec& y, Vec& hv, Vec& lifted) {
  y.resize(v.size());
  std::fill(y.begin(), y.end(), 0.0);
  for(std::size_t i=0;i<rm.rows.size();++i) {
    const auto& row=rm.rows[i];
    const double s=row_dots[i];
    if(s==0.0) continue;
    if(sparse_axpy_rows && std::isfinite(s)) {
      for(int j:(*sparse_axpy_rows)[i])
        y[static_cast<std::size_t>(j)]+=
            s*row[static_cast<std::size_t>(j)];
    } else {
      axpy(s,row,y);
    }
  }
  for(int k=0;k<static_cast<int>(rm.residual.size());++k) {
    const double residual=rm.residual[static_cast<std::size_t>(k)];
    if(residual==0.0) continue;
    reduced_hessian_apply_into(
        *rm.functions[static_cast<std::size_t>(k)],x,v,links,
        lifted,hv);
    axpy(residual,hv,y);
  }
  for(int j=0;j<static_cast<int>(v.size());++j)
    y[static_cast<std::size_t>(j)]+=shift*v[static_cast<std::size_t>(j)];
}

void add_transformed_hessian_dense(std::vector<double>& A, int nf,
                                   const std::vector<int>& free_pos,
                                   const QuadraticFunction& q, double weight,
                                   const Vec& x, const std::vector<AbsLink>& links) {
  const auto& H=q.hessian;
  if (weight == 0.0) return;
  const int n = H.n;
  std::vector<int> map(n);
  Vec scale(n, 1.0);
  std::iota(map.begin(), map.end(), 0);
  for (const auto& link : links) {
    map[link.auxiliary] = link.source;
    scale[link.auxiliary] = abs_sign(x[link.source]);
  }
  for(auto e:q.signed_square) {
    const int j=free_pos[map[e.first]];
    if(j>=0)A[j+j*nf]+=weight*2*e.second*abs_sign(x[e.first])*scale[e.first]*scale[e.first];
  }
  for (const auto& e : H.entries) {
    const int ia = free_pos[map[e.i]];
    const int ib = free_pos[map[e.j]];
    if (ia < 0 || ib < 0) continue;
    const double v = weight * e.v * scale[e.i] * scale[e.j];
    A[ia + ib*nf] += v;
    if (e.i != e.j) A[ib + ia*nf] += v;
  }
}

bool pcg_exact(const ResidualModel& rm, const Vec& x,
               const std::vector<AbsLink>& links,
               const std::vector<int>& free, const Vec& grad,
               const Vec& curvature_diag, double shift,
               const Settings& st, Vec& d, int& cg_iterations,
               ExactPcgWorkspace* shared_workspace,
               ExactSparseRowDotRuntimeState& sparse_runtime) {
  NativeRestorationPcgCall counted;
  const int nf = static_cast<int>(free.size());
  if (nf == 0) return false;
  ExactPcgWorkspace local_workspace;
  ExactPcgWorkspace& ws=shared_workspace?*shared_workspace:local_workspace;
  ws.ensure(static_cast<int>(x.size()),nf,static_cast<int>(rm.rows.size()));

  std::fill(ws.diag_full.begin(),ws.diag_full.end(),shift);
  for (const auto& row : rm.rows)
    for (int j : free)
      ws.diag_full[static_cast<std::size_t>(j)]+=
          row[static_cast<std::size_t>(j)]*row[static_cast<std::size_t>(j)];
  for(int j:free)
    ws.diag_full[static_cast<std::size_t>(j)]+=
        curvature_diag[static_cast<std::size_t>(j)];
  const double micro_shift=shift*1e-6;
  for(int a=0;a<nf;++a) {
    const double denom=std::max(
        micro_shift,
        std::abs(ws.diag_full[static_cast<std::size_t>(free[a])]));
    ws.precond[static_cast<std::size_t>(a)]=std::max(1e-18,denom);
  }

  bool sparse_axpy_worthwhile = false;
  const SparseRowPattern sparse_rows =
      zero_skipping_pattern(rm, x.size(), sparse_axpy_worthwhile);
  bool sparse_axpy_verified = !sparse_axpy_worthwhile;
  bool sparse_axpy_enabled = false;
  // Under fast floating-point semantics the compiler may reassociate the
  // dense contiguous dot and the sparse indexed dot differently.  Do not pay
  // for a dense+sparse verification that is known to be non-portable: retain
  // the historical dense arithmetic for the complete PCG solve instead.
  const bool sparse_dot_worthwhile=
      restoration_exact_sparse_row_dots_build_compatible() &&
      st.restoration_exact_sparse_row_dots && sparse_axpy_worthwhile &&
      !sparse_runtime.dense_latched;
  bool sparse_dot_verified=!sparse_dot_worthwhile;
  bool sparse_dot_enabled=false;
  auto input_is_finite=[&]() {
    for(int a=0;a<nf;++a)
      if(!std::isfinite(ws.pvec[static_cast<std::size_t>(a)])) return false;
    return true;
  };
  auto serial_row_dots=[&]() {
    const bool finite=input_is_finite();
    if(!sparse_dot_verified && finite) {
      for(std::size_t row=0;row<rm.rows.size();++row) {
        ws.row_dots[row]=dot(rm.rows[row],ws.full);
        ws.candidate_row_dots[row]=sparse_row_dot_preserving_order(
            rm.rows[row],sparse_rows[row],ws.full);
      }
      ++ws.sparse_row_dot_verifications;
      sparse_dot_enabled=binary_equal(ws.row_dots,ws.candidate_row_dots);
      sparse_dot_verified=true;
      if(!sparse_dot_enabled) {
        ++ws.sparse_row_dot_fallbacks;
        sparse_runtime.dense_latched=true;
      }
      return;
    }
    if(sparse_dot_enabled && finite) {
      for(std::size_t row=0;row<rm.rows.size();++row)
        ws.row_dots[row]=sparse_row_dot_preserving_order(
            rm.rows[row],sparse_rows[row],ws.full);
      ++ws.sparse_row_dot_batches;
      return;
    }
    for(std::size_t row=0;row<rm.rows.size();++row)
      ws.row_dots[row]=dot(rm.rows[row],ws.full);
  };
  auto serial_matvec = [&](const Vec& input) {
    std::fill(ws.full.begin(), ws.full.end(), 0.0);
    for (int a = 0; a < nf; ++a)
      ws.full[static_cast<std::size_t>(free[a])] =
          input[static_cast<std::size_t>(a)];
    serial_row_dots();
    if (!sparse_axpy_verified) {
      // Preserve the historical dense dot products.  Only the structurally
      // zero AXPY coordinates are skipped, and only after the first complete
      // product agrees bit-for-bit with the historical kernel.
      exact_merit_hessian_finish_from_row_dots(
          rm,x,links,ws.full,shift,nullptr,ws.row_dots,ws.yf,ws.hv,
          ws.lifted);
      exact_merit_hessian_finish_from_row_dots(
          rm,x,links,ws.full,shift,&sparse_rows,ws.row_dots,
          ws.candidate_yf,ws.candidate_hv,ws.candidate_lifted);
      sparse_axpy_enabled=binary_equal(ws.yf,ws.candidate_yf);
      sparse_axpy_verified = true;
    } else {
      exact_merit_hessian_finish_from_row_dots(
          rm,x,links,ws.full,shift,
          sparse_axpy_enabled?&sparse_rows:nullptr,ws.row_dots,ws.yf,ws.hv,
          ws.lifted);
    }
    for (int a = 0; a < nf; ++a)
      ws.q[static_cast<std::size_t>(a)]=
          ws.yf[static_cast<std::size_t>(free[a])];
    ++ws.matvecs;
  };

  for(int a=0;a<nf;++a)
    ws.b[static_cast<std::size_t>(a)]=
        -grad[static_cast<std::size_t>(free[a])];
  const double bnorm=norm2(ws.b);
  if (!(bnorm > 0.0) || !std::isfinite(bnorm)) return false;

  std::fill(ws.z.begin(),ws.z.end(),0.0);
  ws.r=ws.b;
  for(int a=0;a<nf;++a)
    ws.mz[static_cast<std::size_t>(a)]=
        ws.r[static_cast<std::size_t>(a)]/
        ws.precond[static_cast<std::size_t>(a)];
  ws.pvec=ws.mz;
  double rz=dot(ws.r,ws.mz);
  if (!(rz > 0.0) || !std::isfinite(rz)) return false;

  const int maxit = std::min(st.restoration_cg_max_iterations, std::max(20, 2*nf));
  const double tol = st.restoration_cg_tolerance * std::max(1.0, bnorm);
  bool parallel=false;
#ifdef LAZYIPDCA_HAVE_OPENMP
  parallel= !std::isfinite(st.restoration_deadline_seconds) && st.restoration_exact_parallel_row_dots &&
      static_cast<int>(rm.rows.size())>=
          std::max(1,st.restoration_exact_parallel_min_rows);
#endif
  if(!parallel) {
    for(int it=0;it<maxit && !restoration_expired(st);++it) {
      serial_matvec(ws.pvec);
      const double pq=dot(ws.pvec,ws.q);
      if(!(pq>0.0) || !std::isfinite(pq)) return false;
      const double alpha=rz/pq;
      axpy(alpha,ws.pvec,ws.z);
      axpy(-alpha,ws.q,ws.r);
      cg_iterations=it+1;counted.iteration();
      if(norm2(ws.r)<=tol) break;
      for(int a=0;a<nf;++a)
        ws.mz[static_cast<std::size_t>(a)]=
            ws.r[static_cast<std::size_t>(a)]/
            ws.precond[static_cast<std::size_t>(a)];
      const double rz_new=dot(ws.r,ws.mz);
      if(!(rz_new>0.0) || !std::isfinite(rz_new)) return false;
      const double beta=rz_new/rz;
      for(int a=0;a<nf;++a)
        ws.pvec[static_cast<std::size_t>(a)]=
            ws.mz[static_cast<std::size_t>(a)]+
            beta*ws.pvec[static_cast<std::size_t>(a)];
      rz=rz_new;
    }
  } else {
#ifdef LAZYIPDCA_HAVE_OPENMP
    bool stop=false;
    bool numerical_failure=false;
    bool finite_input=true;
    ++ws.parallel_teams;
#pragma omp parallel shared(stop,numerical_failure,rz,sparse_axpy_verified,sparse_axpy_enabled,sparse_dot_verified,sparse_dot_enabled,finite_input)
    {
      for(int it=0;it<maxit;++it) {
#pragma omp single
        {
          std::fill(ws.full.begin(),ws.full.end(),0.0);
          for(int a=0;a<nf;++a)
            ws.full[static_cast<std::size_t>(free[a])]=
                ws.pvec[static_cast<std::size_t>(a)];
          finite_input=input_is_finite();
        }
#pragma omp for schedule(static)
        for(int row=0;row<static_cast<int>(rm.rows.size());++row) {
          const std::size_t k=static_cast<std::size_t>(row);
          if(!sparse_dot_verified && finite_input) {
            ws.row_dots[k]=dot(rm.rows[k],ws.full);
            ws.candidate_row_dots[k]=sparse_row_dot_preserving_order(
                rm.rows[k],sparse_rows[k],ws.full);
          } else if(sparse_dot_enabled && finite_input) {
            ws.row_dots[k]=sparse_row_dot_preserving_order(
                rm.rows[k],sparse_rows[k],ws.full);
          } else {
            ws.row_dots[k]=dot(rm.rows[k],ws.full);
          }
        }
#pragma omp single
        {
          if(!sparse_dot_verified && finite_input) {
            ++ws.sparse_row_dot_verifications;
            sparse_dot_enabled=
                binary_equal(ws.row_dots,ws.candidate_row_dots);
            sparse_dot_verified=true;
            if(!sparse_dot_enabled) {
              ++ws.sparse_row_dot_fallbacks;
              sparse_runtime.dense_latched=true;
            }
          } else if(sparse_dot_enabled && finite_input) {
            ++ws.sparse_row_dot_batches;
          }
          if(!sparse_axpy_verified) {
            exact_merit_hessian_finish_from_row_dots(
                rm,x,links,ws.full,shift,nullptr,ws.row_dots,ws.yf,ws.hv,
                ws.lifted);
            exact_merit_hessian_finish_from_row_dots(
                rm,x,links,ws.full,shift,&sparse_rows,ws.row_dots,
                ws.candidate_yf,ws.candidate_hv,ws.candidate_lifted);
            sparse_axpy_enabled=binary_equal(ws.yf,ws.candidate_yf);
            sparse_axpy_verified=true;
          } else {
            exact_merit_hessian_finish_from_row_dots(
                rm,x,links,ws.full,shift,
                sparse_axpy_enabled?&sparse_rows:nullptr,ws.row_dots,ws.yf,
                ws.hv,ws.lifted);
          }
          for(int a=0;a<nf;++a)
            ws.q[static_cast<std::size_t>(a)]=
                ws.yf[static_cast<std::size_t>(free[a])];
          ++ws.matvecs;
          ++ws.parallel_row_dot_batches;

          const double pq=dot(ws.pvec,ws.q);
          if(!(pq>0.0) || !std::isfinite(pq)) {
            numerical_failure=true;
            stop=true;
          } else {
            const double alpha=rz/pq;
            axpy(alpha,ws.pvec,ws.z);
            axpy(-alpha,ws.q,ws.r);
            cg_iterations=it+1;counted.iteration();
            if(norm2(ws.r)<=tol) {
              stop=true;
            } else {
              for(int a=0;a<nf;++a)
                ws.mz[static_cast<std::size_t>(a)]=
                    ws.r[static_cast<std::size_t>(a)]/
                    ws.precond[static_cast<std::size_t>(a)];
              const double rz_new=dot(ws.r,ws.mz);
              if(!(rz_new>0.0) || !std::isfinite(rz_new)) {
                numerical_failure=true;
                stop=true;
              } else {
                const double beta=rz_new/rz;
                for(int a=0;a<nf;++a)
                  ws.pvec[static_cast<std::size_t>(a)]=
                      ws.mz[static_cast<std::size_t>(a)]+
                      beta*ws.pvec[static_cast<std::size_t>(a)];
                rz=rz_new;
              }
            }
          }
        }
        if(stop) break;
      }
    }
    if(numerical_failure) return false;
#endif
  }

  d.assign(x.size(), 0.0);
  for (int a = 0; a < nf; ++a)
    d[static_cast<std::size_t>(free[a])]=ws.z[static_cast<std::size_t>(a)];
  return std::isfinite(inf_norm(d));
}

NewtonDirection exact_newton_direction(const QCQPProblem& p, const Vec& x,
                                       const ResidualModel& rm,
                                       const std::vector<AbsLink>& links,
                                       const std::vector<char>& active,
                                       const Vec& grad,const Vec& cdiag,
                                       const Vec& cradius,const Settings& st,
                                       double damping,
                                       ExactPcgWorkspace* pcg_workspace,
                                       ExactSparseRowDotRuntimeState& sparse_runtime) {
  NewtonDirection out;
  out.exact_hessian = true;
  std::vector<int> free;
  for (int j = 0; j < p.n; ++j) if (!active[j]) free.push_back(j);
  const int nf = static_cast<int>(free.size());
  if (nf == 0) return out;

  double lower = std::numeric_limits<double>::infinity();
  double scale = 1.0;
  for (int j : free) {
    lower = std::min(lower, cdiag[j] - cradius[j]);
    scale = std::max(scale, std::abs(cdiag[j]) + cradius[j]);
  }
  if (!std::isfinite(lower)) lower = 0.0;
  double shift = std::max(damping*scale, -lower + damping*scale);
  shift = std::max(1e-15, shift);

  for (int attempt = 0; attempt < 6 && !restoration_expired(st); ++attempt) {
    Vec d;
    bool ok = false;
    if (nf <= st.restoration_exact_hessian_threshold) {
      std::vector<int> pos(p.n, -1);
      for (int a = 0; a < nf; ++a) pos[free[a]] = a;
      std::vector<double> A(static_cast<std::size_t>(nf)*nf, 0.0);
      for (const auto& row : rm.rows) {
        for (int a = 0; a < nf; ++a) {
          const double ja = row[free[a]];
          if (ja == 0.0) continue;
          for (int b = 0; b <= a; ++b) {
            const double v = row[free[b]]*ja;
            A[b + a*nf] += v;
            if (a != b) A[a + b*nf] += v;
          }
        }
      }
      for (int k = 0; k < static_cast<int>(rm.residual.size()); ++k)
        add_transformed_hessian_dense(A, nf, pos, *rm.functions[k],
                                      rm.residual[k], x, links);
      for (int a = 0; a < nf; ++a) A[a + a*nf] += shift;
      Vec rhs(nf, 0.0), df;
      for (int a = 0; a < nf; ++a) rhs[a] = -grad[free[a]];
      ++out.linear_solves;
      ok = solve_dense_system(std::move(A), std::move(rhs), df);
      if (ok) {
        d.assign(p.n, 0.0);
        for (int a = 0; a < nf; ++a) d[free[a]] = df[a];
      }
    } else {
      out.matrix_free = true;
      ++out.linear_solves;
      int cgit = 0;
      ok = pcg_exact(rm,x,links,free,grad,cdiag,shift,st,d,cgit,
                     pcg_workspace,sparse_runtime);
      out.cg_iterations += cgit;
    }

    if (ok) {
      const double gd = dot(grad, d);
      const bool descent=st.restoration_failed_linearization_context&&st.restoration_private_precision_context?
          detail::failed_linearization_relative_descent(grad,d):
          (std::isfinite(gd) && gd < -1e-14*std::max(1.0, norm2(grad))*std::max(1.0, norm2(d)));
      if (descent) {
        out.d = std::move(d);
        out.success = true;
        out.shift = shift;
        return out;
      }
    }
    shift *= 10.0;
  }
  out.shift = shift;
  return out;
}

struct LegacyLocalRestoration {
  Vec x;
  int iterations = 0;
  int linear_solves = 0;
  int row_space_solves = 0;
  int line_search_backtracks = 0;
  double theta = std::numeric_limits<double>::infinity();
  double merit = std::numeric_limits<double>::infinity();
  double damping = 0.0;
};

#include "restoration_negative_curvature.hpp"
#include "restoration_column_lm.hpp"

LegacyLocalRestoration legacy_local_restore(const QCQPProblem& p, Vec x, const Settings& st,
                               int max_iterations,int maximum_workspace_mb=0,
                               bool closed_input_bounds=false) {
  // Preserve the historical interior trajectory until bound-limited progress
  // actually stalls. Phase I can then use the same finite input bounds as a
  // closed active set; no new point, bound, tolerance or budget is introduced.
  const bool adaptive_bound_closure=closed_input_bounds&&p.has_signed_square();
  if(adaptive_bound_closure)closed_input_bounds=false;
  int bound_stall=0;double bound_reference=INFINITY;
  const auto links = detect_abs_links(p);
  x = clamp_to_bounds(p, std::move(x));
  const auto fixed0 = fixed_mask(p, st.fixed_bound_tolerance);
  const auto aux0 = auxiliary_abs_mask(p, links);
  for (int j = 0; j < p.n; ++j) if (fixed0[j]) x[j] = p.lb[j];
  project_abs_links(p, links, x);

  LegacyLocalRestoration out;
  out.x = x;
  out.theta = feasibility(p, x).total_inf;
  out.merit = legacy_squared_merit_values(p,x,links);
  double damping = std::max(1e-15, st.restoration_lm_initial_damping);
  double best_theta = out.theta;
  double best_merit = out.merit;
  Vec best_x = x;
  int rejected = 0;
  // Keep the established row-space route for large, underdetermined equality
  // chains and add a tightly gated variant for large sparse models that are
  // already equality-feasible but have only a few violated inequalities.
  // Medium-sized dense problems still use their established exact-Hessian
  // route.
  const bool equality_chain_row_space=links.empty() &&
      large_sparse_equality_chain_candidate(p);
  const bool near_feasible_inequality_row_space=links.empty() &&
      near_feasible_sparse_inequality_candidate(p,x,st);
  const bool row_space_candidate=equality_chain_row_space ||
      near_feasible_inequality_row_space;
  if(near_feasible_inequality_row_space)
    damping=std::min(damping,1e-12);
  std::string row_space_message;
  auto row_space_factor=row_space_candidate
      ? (maximum_workspace_mb>0?
          make_memory_bounded_symmetric_factorization(st,row_space_message,maximum_workspace_mb):
          make_symmetric_factorization(st,row_space_message)) : nullptr;

  // A rank-deficient first-order residual model can be stationary while the
  // original quadratic feasibility merit has negative curvature. This is a
  // verified Phase-I step from x, not a replacement of the supplied x0.
  const auto curvature=restoration_negative_curvature_step(p,x,st);
  const int curvature_iterations=curvature.accepted?1:0;
  out.iterations=curvature_iterations;
  if(curvature.accepted) {
    out.merit=legacy_squared_merit_values(p,x,links);
    // Phase-I globalizes the squared merit; retain the existing best-point
    // safeguard in the infinity norm if this accepted step increases theta.
  }
  for (int it = 0; it < max_iterations-curvature_iterations && !restoration_expired(st); ++it) {
    out.iterations = it + 1 + curvature_iterations;
    const Feasibility fx = feasibility(p, x);
    if(adaptive_bound_closure&&!closed_input_bounds&&fx.total_inf>st.feasibility_tolerance){
      bool near_bound=false;
      for(int j=0;j<p.n;++j)if(!fixed0[j]){
        if(std::isfinite(p.lb[j])&&x[j]-p.lb[j]<=1e-8*std::max(1.,std::abs(p.lb[j])))near_bound=true;
        if(std::isfinite(p.ub[j])&&p.ub[j]-x[j]<=1e-8*std::max(1.,std::abs(p.ub[j])))near_bound=true;
      }
      if(!near_bound||fx.total_inf<.99*bound_reference){bound_stall=0;bound_reference=fx.total_inf;}
      else if(++bound_stall>=8){closed_input_bounds=true;
        if(st.live_display_level||st.verbose)std::cout<<"[RESTORATION-CLOSED-BOUND] trigger=observed-bound-stagnation primal="<<fx.total_inf
            <<" shared_iteration_budget=1 supplied_start_unchanged=1"<<std::endl;
      }
    }
    if (fx.total_inf < best_theta ||
        (fx.total_inf <= best_theta*(1.0 + 1e-12) && out.merit < best_merit)) {
      best_theta = fx.total_inf;
      best_merit = out.merit;
      best_x = x;
    }
    if (fx.total_inf <= st.feasibility_tolerance) break;

    const LegacyResidualJacobian rj = links.empty()
        ? legacy_residual_jacobian(p,x,row_space_candidate)
        : legacy_residual_jacobian_abs_reduced(p,x,links);
    const double phi = legacy_squared_merit(rj);
    const int m = static_cast<int>(rj.residual.size());
    if (m == 0) break;

    std::vector<char> active = fixed0;
    // Structured lift auxiliaries are eliminated from the Phase-I variables.
    // Their values are reconstructed exactly as |q| after every trial step.
    for (int j = 0; j < p.n; ++j) if (aux0[j]) active[j] = 1;
    Vec d(p.n, 0.0);
    bool have_direction = false;

    for (int aset = 0; aset < std::max(1, st.restoration_bound_active_passes) && !restoration_expired(st); ++aset) {
      std::vector<int> free;
      free.reserve(p.n);
      for (int j = 0; j < p.n; ++j) if (!active[j]) free.push_back(j);
      const int nf = static_cast<int>(free.size());
      if (nf == 0) break;

      bool direction_solved=false;
      ++out.linear_solves;
      if(row_space_candidate) {
        if(row_space_factor) {
          direction_solved=sparse_row_space_lm_direction(
              rj,free,damping,*row_space_factor,d,p.n,row_space_message,
              near_feasible_inequality_row_space,true,!closed_input_bounds);
          if(direction_solved) ++out.row_space_solves;
        }
      } else {
        std::vector<double> A(static_cast<std::size_t>(nf)*nf, 0.0);
        Vec rhs(nf, 0.0);
        Vec colnorm2(nf, 0.0);
        for (int row = 0; row < m; ++row) {
          const Vec& jr = rj.rows[row];
          const double rr = rj.residual[row];
          for (int a = 0; a < nf; ++a) {
            const double ja = jr[free[a]];
            rhs[a] -= ja * rr;
            colnorm2[a] += ja * ja;
            for (int b = 0; b <= a; ++b)
              A[b + a*nf] += jr[free[b]] * ja;
          }
        }
        for (int a = 0; a < nf; ++a) {
          for (int b = a + 1; b < nf; ++b) A[b + a*nf] = A[a + b*nf];
          A[a + a*nf] += damping * std::max(1.0, colnorm2[a]);
        }
        Vec df;
        direction_solved=solve_dense_system(std::move(A),std::move(rhs),df);
        if(direction_solved) {
          std::fill(d.begin(),d.end(),0.0);
          for(int a=0;a<nf;++a) d[free[a]]=df[a];
        }
      }
      if(!direction_solved) {
        damping*=10.0;
        continue;
      }

      std::vector<int> block;
      for (int j : free) {
        if (std::isfinite(p.lb[j]) && x[j] <= p.lb[j] + st.restoration_bound_activity_tolerance && d[j] < 0.0)
          block.push_back(j);
        else if (std::isfinite(p.ub[j]) && x[j] >= p.ub[j] - st.restoration_bound_activity_tolerance && d[j] > 0.0)
          block.push_back(j);
      }
      if (block.empty()) { have_direction = true; break; }
      for (int j : block) active[j] = 1;
    }

    if (!have_direction || inf_norm(d) <= st.restoration_step_tolerance) break;

    double alpha = 1.0;
    // Feasibility restoration has no log barrier. On the bounded recovery
    // route it may reach the first finite INPUT bound exactly; the existing
    // active-set pass then freezes any outward direction there. Retaining an
    // IPM interior fraction repeatedly approaches these same closed sides.
    const double boundary_fraction=closed_input_bounds?1.0:st.fraction_to_boundary;
    for (int j = 0; j < p.n; ++j) {
      if (d[j] > 0.0 && std::isfinite(p.ub[j])) {
        const double room = p.ub[j] - x[j];
        if (room <= 0.0) alpha = 0.0;
        else alpha = std::min(alpha, boundary_fraction * room / d[j]);
      } else if (d[j] < 0.0 && std::isfinite(p.lb[j])) {
        const double room = p.lb[j] - x[j];
        if (room >= 0.0) alpha = 0.0;
        else alpha = std::min(alpha, boundary_fraction * room / d[j]);
      }
    }
    alpha = std::max(0.0, std::min(1.0, alpha));
    if (!(alpha > 0.0)) break;

    Vec trial;
    double phi_trial=std::numeric_limits<double>::infinity();
    double rho=-1.0;
    bool accepted=false;
    const int restoration_ls_steps=row_space_candidate
        ? std::max(1,st.restoration_line_search_max_steps) : 1;
    for(int ls=0;ls<restoration_ls_steps && !restoration_expired(st);++ls) {
      Vec step=d;
      for(double& v:step) v*=alpha;
      trial=x;
      axpy(1.0,step,trial);
      trial=clamp_to_bounds(p,std::move(trial));
      for(int j=0;j<p.n;++j) if(fixed0[j]) trial[j]=p.lb[j];
      project_abs_links(p,links,trial);

      phi_trial=legacy_squared_merit_values(p,trial,links);
      if(std::isfinite(phi_trial) && phi_trial<phi) {
        Vec accepted_step=trial;
        for(int j=0;j<p.n;++j) accepted_step[j]-=x[j];
        Vec linearized=rj.residual;
        for(int row=0;row<m;++row)
          linearized[row]+=legacy_row_dot(rj,row,accepted_step);
        const double predicted=phi-0.5*dot(linearized,linearized);
        rho=predicted>1e-30 ? (phi-phi_trial)/predicted : -1.0;
        accepted=true;
        break;
      }
      ++out.line_search_backtracks;
      alpha*=st.restoration_line_search_contraction;
      if(alpha<st.minimum_step_length) break;
    }

    if (accepted) {
      x = std::move(trial);
      out.merit = phi_trial;
      rejected = 0;
      if (rho > 0.75) damping *= 0.25;
      else if (rho < 0.25) damping *= 2.0;
      damping = std::max(1e-15, damping);
    } else {
      ++rejected;
      damping *= rejected < 4 ? 4.0 : 10.0;
      if (damping > st.restoration_lm_max_damping) break;
    }
  }

  const Feasibility fcur = feasibility(p, x);
  if (fcur.total_inf < best_theta) {
    best_theta = fcur.total_inf;
    best_merit = legacy_squared_merit_values(p,x,links);
    best_x = x;
  }
  out.x = std::move(best_x);
  out.theta = best_theta;
  out.merit = best_merit;
  out.damping = damping;
  return out;
}


struct LocalRestoration {
  Vec x;
  int iterations = 0;
  int linear_solves = 0;
  int exact_hessian_solves = 0;
  int matrix_free_solves = 0;
  int cg_iterations = 0;
  int exact_workspace_growths = 0;
  int exact_matvecs = 0;
  int exact_parallel_row_dot_batches = 0;
  int exact_parallel_teams = 0;
  int exact_sparse_row_dot_batches = 0;
  int exact_sparse_row_dot_verifications = 0;
  int exact_sparse_row_dot_fallbacks = 0;
  int line_search_backtracks = 0;
  int stagnation_events = 0;
  double theta = std::numeric_limits<double>::infinity();
  double merit = std::numeric_limits<double>::infinity();
  double damping = 0.0;
};

LocalRestoration local_restore(const QCQPProblem& p, Vec x, const Settings& st,
                               int max_iterations,
                               ExactSparseRowDotRuntimeState& sparse_runtime) {
  const auto links = detect_abs_links(p);
  x = clamp_to_bounds(p, std::move(x));
  const auto fixed0 = fixed_mask(p, st.fixed_bound_tolerance);
  const auto aux0 = auxiliary_abs_mask(p, links);
  for (int j = 0; j < p.n; ++j) if (fixed0[j]) x[j] = p.lb[j];
  project_abs_links(p, links, x);

  LocalRestoration out;
  out.x = x;
  out.theta = feasibility(p, x).total_inf;
  out.merit = squared_merit(residual_model(p, x, links));
  double damping = std::max(1e-15, st.restoration_lm_initial_damping);
  double best_theta = out.theta;
  double best_merit = out.merit;
  Vec best_x = x;
  int no_progress = 0;
  ExactPcgWorkspace exact_pcg_workspace;

  // Intermediate dimensions enter exact restoration directly, bypassing the
  // small dense and large sparse Gauss-Newton probe. They need the same
  // certified saddle escape for pure quadratic feasibility.
  RestorationCurvatureStep curvature;
  if((p.n>64 && p.n<512) || (st.restoration_failed_linearization_context &&
      detail::failure_recovery_budget(p).allowed))
    curvature=restoration_negative_curvature_step(p,x,st);
  const int curvature_iterations=curvature.accepted?1:0;
  out.iterations=curvature_iterations;
  for (int it = 0; it < max_iterations-curvature_iterations && !restoration_expired(st); ++it) {
    out.iterations = it + 1 + curvature_iterations;
    const Feasibility fx = feasibility(p, x);
    const ResidualModel rm = residual_model(p, x, links);
    const double phi = squared_merit(rm);
    out.merit = phi;

    const double old_best_theta = best_theta;
    if (fx.total_inf < best_theta ||
        (fx.total_inf <= best_theta*(1.0 + 1e-12) && phi < best_merit)) {
      best_theta = fx.total_inf;
      best_merit = phi;
      best_x = x;
    }
    if (fx.total_inf <= st.feasibility_tolerance) break;

    const double rel_improve = (old_best_theta - best_theta)/std::max(1e-16, old_best_theta);
    if (rel_improve >= st.restoration_stagnation_relative_tolerance) no_progress = 0;
    else ++no_progress;
    if (no_progress >= st.restoration_stagnation_iterations) {
      ++out.stagnation_events;
      break;
    }

    const int m = static_cast<int>(rm.residual.size());
    if (m == 0) break;

    // These quantities depend on the residual model, not on the active-set
    // pass. Reuse them bit-for-bit across all passes at this Phase-I iterate.
    Vec grad=merit_gradient(rm,p.n);
    // The first-order least-squares model may become stationary only AFTER
    // the initial restoration step. An initial-only curvature probe misses
    // this degeneracy. The caller's failed-linearization context keeps all
    // historical successful startup trajectories unchanged.
    if(st.restoration_failed_linearization_context &&
       inf_norm(grad)<=1e-6*std::max(1.0,phi)) {
      const auto escape=restoration_negative_curvature_step(p,x,st);
      if(escape.accepted){damping=std::max(1e-15,st.restoration_lm_initial_damping);continue;}
    }
    Vec cdiag,cradius;
    transformed_curvature_gershgorin(rm,x,links,cdiag,cradius);

    std::vector<char> active = fixed0;
    for (int j = 0; j < p.n; ++j) if (aux0[j]) active[j] = 1;
    Vec d(p.n, 0.0);
    bool have_direction = false;
    NewtonDirection nd;

    for (int aset = 0; aset < std::max(1, st.restoration_bound_active_passes) && !restoration_expired(st); ++aset) {
      nd = exact_newton_direction(
          p,x,rm,links,active,grad,cdiag,cradius,st,damping,
          st.restoration_exact_workspace_reuse?&exact_pcg_workspace:nullptr,
          sparse_runtime);
      out.linear_solves += nd.linear_solves;
      out.cg_iterations += nd.cg_iterations;
      if (nd.exact_hessian) out.exact_hessian_solves += nd.linear_solves;
      if (nd.matrix_free) out.matrix_free_solves += nd.linear_solves;
      if (!nd.success) break;
      d = nd.d;

      std::vector<int> block;
      for (int j = 0; j < p.n; ++j) if (!active[j]) {
        if (std::isfinite(p.lb[j]) && x[j] <= p.lb[j] + st.restoration_bound_activity_tolerance && d[j] < 0.0)
          block.push_back(j);
        else if (std::isfinite(p.ub[j]) && x[j] >= p.ub[j] - st.restoration_bound_activity_tolerance && d[j] > 0.0)
          block.push_back(j);
      }
      if (block.empty()) { have_direction = true; break; }
      for (int j : block) active[j] = 1;
    }

    if (!have_direction || inf_norm(d) <= st.restoration_step_tolerance) {
      // Guaranteed descent fallback.  This is intentionally matrix-free and
      // only used if the safeguarded exact Newton system cannot produce a
      // descent direction.
      d.assign(p.n, 0.0);
      for (int j = 0; j < p.n; ++j) if (!active[j]) d[j] = -grad[j];
      const double gn = inf_norm(d);
      if (!(gn > st.restoration_step_tolerance) || !std::isfinite(gn)) break;
      const double inv = 1.0/std::max(1.0, gn);
      for (double& v : d) v *= inv;
      have_direction = true;
    }

    double alpha_max = 1.0;
    for (int j = 0; j < p.n; ++j) {
      if (d[j] > 0.0 && std::isfinite(p.ub[j])) {
        const double room = p.ub[j] - x[j];
        alpha_max = std::min(alpha_max, room > 0.0 ? st.fraction_to_boundary*room/d[j] : 0.0);
      } else if (d[j] < 0.0 && std::isfinite(p.lb[j])) {
        const double room = p.lb[j] - x[j];
        alpha_max = std::min(alpha_max, room < 0.0 ? st.fraction_to_boundary*room/d[j] : 0.0);
      }
    }
    alpha_max = std::max(0.0, std::min(1.0, alpha_max));
    if (!(alpha_max > 0.0)) break;

    const double gd = dot(grad, d);
    double alpha = alpha_max;
    bool accepted = false;
    Vec accepted_x;
    double accepted_phi = phi;
    double accepted_theta = fx.total_inf;
    Vec best_trial;
    double best_trial_theta = fx.total_inf;
    double best_trial_phi = phi;

    for (int ls = 0; ls < st.restoration_line_search_max_steps && !restoration_expired(st); ++ls) {
      Vec trial = x;
      axpy(alpha, d, trial);
      trial = clamp_to_bounds(p, std::move(trial));
      for (int j = 0; j < p.n; ++j) if (fixed0[j]) trial[j] = p.lb[j];
      project_abs_links(p, links, trial);
      const ResidualModel trm = residual_model(p, trial, links);
      const double phi_trial = squared_merit(trm);
      const double theta_trial = feasibility(p, trial).total_inf;

      if (theta_trial < best_trial_theta ||
          (theta_trial <= best_trial_theta*(1.0 + 1e-12) && phi_trial < best_trial_phi)) {
        best_trial_theta = theta_trial;
        best_trial_phi = phi_trial;
        best_trial = trial;
      }

      const bool armijo = std::isfinite(phi_trial) && gd < 0.0 &&
          phi_trial <= phi + st.restoration_armijo*alpha*gd;
      const bool theta_funnel = std::isfinite(phi_trial) &&
          theta_trial <= fx.total_inf*(1.0 - st.restoration_theta_fraction*alpha) &&
          phi_trial <= phi*st.restoration_theta_merit_growth;
      if (armijo || theta_funnel) {
        accepted = true;
        accepted_x = std::move(trial);
        accepted_phi = phi_trial;
        accepted_theta = theta_trial;
        break;
      }
      alpha *= st.restoration_line_search_contraction;
      ++out.line_search_backtracks;
    }

    if (!accepted && !best_trial.empty() &&
        best_trial_theta < fx.total_inf*(1.0 - st.restoration_min_theta_improvement) &&
        best_trial_phi <= phi*st.restoration_theta_merit_growth) {
      accepted = true;
      accepted_x = std::move(best_trial);
      accepted_phi = best_trial_phi;
      accepted_theta = best_trial_theta;
    }

    if (accepted) {
      x = std::move(accepted_x);
      out.merit = accepted_phi;
      if (accepted_theta < fx.total_inf*0.5) damping *= 0.25;
      else damping *= 0.7;
      damping = std::max(1e-15, damping);
    } else {
      damping *= 10.0;
      if (damping > st.restoration_lm_max_damping) {
        ++out.stagnation_events;
        break;
      }
    }
  }

  const Feasibility fcur = feasibility(p, x);
  const double pcur = squared_merit(residual_model(p, x, links));
  if (fcur.total_inf < best_theta ||
      (fcur.total_inf <= best_theta*(1.0 + 1e-12) && pcur < best_merit)) {
    best_theta = fcur.total_inf;
    best_merit = pcur;
    best_x = x;
  }
  out.x = std::move(best_x);
  out.theta = best_theta;
  out.merit = best_merit;
  out.damping = damping;
  out.exact_workspace_growths=exact_pcg_workspace.growths;
  out.exact_matvecs=exact_pcg_workspace.matvecs;
  out.exact_parallel_row_dot_batches=
      exact_pcg_workspace.parallel_row_dot_batches;
  out.exact_parallel_teams=exact_pcg_workspace.parallel_teams;
  out.exact_sparse_row_dot_batches=exact_pcg_workspace.sparse_row_dot_batches;
  out.exact_sparse_row_dot_verifications=
      exact_pcg_workspace.sparse_row_dot_verifications;
  out.exact_sparse_row_dot_fallbacks=
      exact_pcg_workspace.sparse_row_dot_fallbacks;
  return out;
}


struct MinimaxRestoration {
  Vec x;
  double theta = std::numeric_limits<double>::infinity();
  int qp_solves = 0;
  int inner_iterations = 0;
  int factorizations = 0;
  int accepted_steps = 0;
  int line_search_backtracks = 0;
};

SparseRow sparse_row_from_free_gradient(const Vec& g, const std::vector<int>& free, int t_index) {
  SparseRow row;
  row.entries.reserve(free.size());
  for (int a = 0; a < static_cast<int>(free.size()); ++a) {
    const double v = g[free[a]];
    if (std::abs(v) > 0.0) row.entries.push_back({a, v});
  }
  (void)t_index;
  return row;
}

ConvexQuadraticModel linear_model(int nvar, double constant,
                                  const std::vector<std::pair<int,double>>& entries) {
  ConvexQuadraticModel q;
  q.constant = constant;
  q.linear.assign(nvar, 0.0);
  q.base_hessian.n = nvar;
  for (const auto& [j, v] : entries) q.linear[j] += v;
  return q;
}

// Terminal Phase-I refinement aligned with the actual feasibility certificate.
//
// The exact-Newton restoration minimizes an L2 squared-residual merit.  Near a
// feasible boundary with many inequalities this can be almost stationary even
// though the worst violation is still well above feasibility_tolerance.  This
// routine solves the convex epigraph SQP model
//
//   min_{d,t} t + 0.5*eta*||d||^2
//   s.t. h(x)+Jh(x)d = 0,
//        g_i(x)+Jg_i(x)d <= t,
//        x+d within bounds,  t >= 0.
//
// and globalizes the step on the ORIGINAL QCQP maximum violation.  It is used
// only after the L2 Phase-I fails, so successful v0.2.2/v0.2.3-r2 trajectories
// (BT1/HIMMELBD/DISCS/CORE2) are not perturbed.
MinimaxRestoration minimax_restore(const QCQPProblem& p, Vec x, const Settings& st,
                                   int max_passes) {
  MinimaxRestoration out;
  x = clamp_to_bounds(p, std::move(x));
  const auto fixed = fixed_mask(p, st.fixed_bound_tolerance);
  for (int j = 0; j < p.n; ++j) if (fixed[j]) x[j] = p.lb[j];
  out.x = x;
  out.theta = feasibility(p, x).total_inf;

  // Structured t=|q| models already have their own reduced-manifold Phase I.
  // Do not build an epigraph model in the lifted coordinates.
  if (!detect_abs_links(p).empty() || out.theta <= st.feasibility_tolerance) return out;

  std::vector<int> free;
  free.reserve(p.n);
  for (int j = 0; j < p.n; ++j) if (!fixed[j]) free.push_back(j);
  if (free.empty()) return out;

  Settings inner_st = st;
  inner_st.max_inner_iterations = std::max(80, std::min(st.max_inner_iterations, 400));
  inner_st.inner_tolerance = std::min(st.inner_tolerance, 1e-9);
  inner_st.inner_iteration_limit_mu_fallback = std::min(st.inner_iteration_limit_mu_fallback, 1e-8);

  double eta = std::max(1e-10, st.restoration_minimax_proximal);

  for (int pass = 0; pass < max_passes && !restoration_expired(st); ++pass) {
    const Feasibility fx = feasibility(p, x);
    if (fx.total_inf <= st.feasibility_tolerance) break;

    const int nf = static_cast<int>(free.size());
    const int t_index = nf;
    const int nvar = nf + 1;
    ConvexSubproblem sub;
    sub.nvar = nvar;
    sub.original_dimension = nf;
    sub.objective.constant = 0.0;
    sub.objective.linear.assign(nvar, 0.0);
    sub.objective.linear[t_index] = 1.0;
    sub.objective.base_hessian.n = nvar;
    sub.objective.shift_dimension = nf;
    sub.objective.scalar_shift = eta;
    sub.equalities.nvar = nvar;

    const bool elastic=st.restoration_failed_linearization_context&&st.restoration_shared_iteration_budget;
    const double epigraph_offset=elastic?1+fx.total_inf:0;
    sub.objective.constant=epigraph_offset;
    // In failure recovery ALL linearized equations may be temporarily
    // violated through the SAME nonnegative epigraph variable. Thus even
    // incompatible tangent equalities define a feasible convex QP.
    for (const auto& h : p.equalities) {
      if(elastic) {
        const Vec gh=h.gradient(x);
        for(int sign:{-1,1}) {
          std::vector<std::pair<int,double>> entries;
          for(int a=0;a<nf;++a)if(gh[free[a]]!=0)entries.push_back({a,sign*gh[free[a]]});
          entries.push_back({t_index,-1});
          sub.inequalities.push_back(linear_model(nvar,sign*h.value(x)-epigraph_offset,entries));
        }
        continue;
      }
      const Vec gh = h.gradient(x);
      SparseRow row = sparse_row_from_free_gradient(gh, free, t_index);
      sub.equalities.rows.push_back(std::move(row));
      sub.equalities.rhs.push_back(-h.value(x));
    }

    sub.inequalities.reserve(p.inequalities.size() + 2*free.size() + 1);
    for (const auto& g : p.inequalities) {
      const Vec gg = g.gradient(x);
      std::vector<std::pair<int,double>> e;
      e.reserve(free.size()+1);
      for (int a = 0; a < nf; ++a) {
        const double v = gg[free[a]];
        if (v != 0.0) e.push_back({a, v});
      }
      e.push_back({t_index, -1.0});
      sub.inequalities.push_back(linear_model(nvar, g.value(x)-epigraph_offset, e));
    }

    // Keep x+d inside the original bounds.
    for (int a = 0; a < nf; ++a) {
      const int j = free[a];
      if (std::isfinite(p.ub[j]))
        sub.inequalities.push_back(linear_model(nvar, x[j]-p.ub[j], {{a, 1.0}}));
      if (std::isfinite(p.lb[j]))
        sub.inequalities.push_back(linear_model(nvar, p.lb[j]-x[j], {{a, -1.0}}));
    }
    // t >= 0.
    sub.inequalities.push_back(linear_model(nvar, -epigraph_offset, {{t_index, -1.0}}));

    ConvexQCQPInteriorPoint ipm(inner_st);
    ++out.qp_solves;
    InnerResult ir = ipm.solve(sub);
    out.inner_iterations += ir.iterations;
    out.factorizations += ir.factorizations;
    if (!ir.success || static_cast<int>(ir.y.size()) != nvar) {
      if (st.verbose >= 1) {
        std::fprintf(stderr,
            "[REST-MINIMAX-FAIL] pass=%d eta=%.12e backend=%s fact=%d inner=%d msg=%s\n",
            pass, eta, st.linear_backend.c_str(), ir.factorizations, ir.iterations,
            ir.message.c_str());
      }
      eta = std::min(1e4, 10.0*eta);
      continue;
    }

    Vec d(p.n, 0.0);
    for (int a = 0; a < nf; ++a) d[free[a]] = ir.y[a];
    if (!(inf_norm(d) > st.restoration_step_tolerance) || !std::isfinite(inf_norm(d))) break;

    bool accepted = false;
    Vec best_trial;
    double best_theta = fx.total_inf;
    double alpha = 1.0;
    for (int ls = 0; ls < st.restoration_line_search_max_steps && !restoration_expired(st); ++ls) {
      Vec trial = x;
      axpy(alpha, d, trial);
      trial = clamp_to_bounds(p, std::move(trial));
      for (int j = 0; j < p.n; ++j) if (fixed[j]) trial[j] = p.lb[j];
      const double th = feasibility(p, trial).total_inf;
      if (th < best_theta) { best_theta = th; best_trial = trial; }
      const double required = std::max(st.feasibility_tolerance,
          fx.total_inf*(1.0 - st.restoration_minimax_theta_fraction*alpha));
      if (std::isfinite(th) && th <= required) {
        x = std::move(trial);
        out.theta = th;
        ++out.accepted_steps;
        accepted = true;
        break;
      }
      alpha *= st.restoration_line_search_contraction;
      ++out.line_search_backtracks;
    }
    if (!accepted && !best_trial.empty() &&
        best_theta < fx.total_inf*(1.0 - st.restoration_min_theta_improvement)) {
      x = std::move(best_trial);
      out.theta = best_theta;
      ++out.accepted_steps;
      accepted = true;
    }
    if (!accepted) {
      eta = std::min(1e4, 10.0*eta);
      if (eta >= 1e4) break;
      continue;
    }

    out.x = x;
    eta = std::max(1e-12, 0.3*eta);
  }

  out.x = x;
  out.theta = feasibility(p, x).total_inf;
  return out;
}

Vec coordinate_restart(const QCQPProblem& p, int k, const std::vector<char>& fixed) {
  Vec v(p.n, 0.0);
  std::vector<int> free;
  for (int j = 0; j < p.n; ++j) if (!fixed[j]) free.push_back(j);
  if (free.empty()) return v;
  const std::size_t idx = (static_cast<std::size_t>(k)*104729u) % free.size();
  v[free[idx]] = (k % 2 == 0) ? 1.0 : -1.0;
  return v;
}

Vec small_problem_coordinate_restart(const QCQPProblem& p, int k, const std::vector<char>& fixed) {
  Vec v(p.n, 0.0);
  std::vector<int> free;
  for (int j = 0; j < p.n; ++j) if (!fixed[j]) free.push_back(j);
  if (free.empty()) return v;
  const int j = free[(k/2) % free.size()];
  v[j] = (k % 2 == 0) ? 1.0 : -1.0;
  return v;
}

void accumulate_local(RestorationResult& out, const LocalRestoration& loc) {
  out.iterations += loc.iterations;
  out.linear_solves += loc.linear_solves;
  out.exact_hessian_solves += loc.exact_hessian_solves;
  out.matrix_free_solves += loc.matrix_free_solves;
  out.cg_iterations += loc.cg_iterations;
  out.exact_workspace_growths += loc.exact_workspace_growths;
  out.exact_matvecs += loc.exact_matvecs;
  out.exact_parallel_row_dot_batches +=
      loc.exact_parallel_row_dot_batches;
  out.exact_parallel_teams += loc.exact_parallel_teams;
  out.exact_sparse_row_dot_batches += loc.exact_sparse_row_dot_batches;
  out.exact_sparse_row_dot_verifications +=
      loc.exact_sparse_row_dot_verifications;
  out.exact_sparse_row_dot_fallbacks += loc.exact_sparse_row_dot_fallbacks;
  out.line_search_backtracks += loc.line_search_backtracks;
  out.stagnation_events += loc.stagnation_events;
}

} // namespace

SparseRestorationBudget sparse_restoration_budget(const QCQPProblem& p,const Settings& st) {
  SparseRestorationBudget out;
  constexpr long double auxiliary_limit=384.0L*1024*1024;
  // Check linear-size bookkeeping BEFORE allocating structural work arrays.
  if(!st.restoration_failed_linearization_context||!st.restoration_shared_iteration_budget||
      st.linear_backend!="mumps"||!st.linear_metis||p.n<512||
      !p.inequalities.empty()||p.equalities.size()<8||
      p.equalities.size()>=static_cast<std::size_t>(p.n)||p.has_signed_square()||
      2048.0L*p.n+256.0L*p.equalities.size()>auxiliary_limit)return out;
  if(!large_sparse_equality_chain_candidate(p)||!detect_abs_links(p).empty())return out;
  const std::size_t n=static_cast<std::size_t>(p.n),m=p.equalities.size();
  std::vector<std::size_t> degree(n,0);std::vector<char> seen(n,0);
  std::vector<int> support;std::size_t entries=0;
  for(const auto& q:p.equalities) {
    support.clear();
    const auto add=[&](int j){if(!seen[j]){seen[j]=1;support.push_back(j);}};
    q.for_each_linear_nonzero([&](int j,double v){if(v!=0)add(j);});
    for(const auto& h:q.hessian.entries)if(h.v!=0){add(h.i);add(h.j);}
    if(support.size()>std::numeric_limits<std::size_t>::max()-entries)return out;
    entries+=support.size();
    for(int j:support){++degree[j];seen[j]=0;}
  }
  long double triplets=m;
  // Numerical cancellation or active bounds can lower a structural degree
  // below the border threshold. Bound BOTH representations over all possible
  // actual degrees: q+1 for a border, or at most 63*64/2 for a local clique.
  constexpr long double largest_unbordered_clique=63.0L*64/2;
  for(std::size_t q:degree)triplets+=q>=64?
      std::max(static_cast<long double>(q)+1,largest_unbordered_clique):
      static_cast<long double>(q)*(q+1)/2;
  // Conservative auxiliary construction envelope: Lanczos vectors, sparse
  // Jacobian/column views, graph/index copies, triplets and assembly scratch.
  // MUMPS numerical integer+real workspace has its SEPARATE explicit 128 MB
  // cap. Neither number is claimed to bound total process RSS/symbolic fill.
  const long double bytes=2048.0L*n+256.0L*m+128.0L*entries+256.0L*triplets;
  if(!std::isfinite(bytes)||bytes>auxiliary_limit||
      triplets>std::numeric_limits<std::size_t>::max())return out;
  out.allowed=true;out.jacobian_entries=entries;
  out.normal_triplets=static_cast<std::size_t>(triplets);out.estimated_auxiliary_bytes=bytes;
  return out;
}

RestorationResult FeasibilityRestoration::solve_sparse_only(
    const QCQPProblem& p,const Vec& start) const {
  RestorationResult out;out.x=start;
  const auto began=std::chrono::steady_clock::now();
  const auto finish=[&](const char* message) {
    out.message=message;out.total_seconds=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-began).count();return out;
  };
  if(restoration_expired(st_))return finish("sparse-only Phase-I deadline before admission");
  if(start.size()!=static_cast<std::size_t>(p.n)||
      !std::all_of(start.begin(),start.end(),[](double v){return std::isfinite(v);}))
    return finish("sparse-only Phase-I invalid starting coordinates");
  const auto budget=sparse_restoration_budget(p,st_);
  if(!budget.allowed)
    return finish("sparse-only Phase-I structural/workspace admission refused");
  if(restoration_expired(st_)||st_.restoration_max_iterations<=0)
    return finish("sparse-only Phase-I time/iteration budget exhausted");
  out.initial_feasibility=out.final_feasibility=feasibility(p,start).total_inf;
  if(st_.live_display_level>0)
    std::cout<<"[PHASE-I-SPARSE-ONLY] admitted=1 jacobian_entries="<<budget.jacobian_entries
      <<" normal_triplet_bound="<<budget.normal_triplets
      <<" auxiliary_byte_bound="<<static_cast<double>(budget.estimated_auxiliary_bytes)
      <<" factor_workspace_mb="<<budget.maximum_factor_workspace_mb
      <<" dense_fallback=0 exact_fallback=0 restarts=0\n";
  const auto local=legacy_local_restore(p,start,st_,st_.restoration_max_iterations,
      budget.maximum_factor_workspace_mb);
  out.iterations=local.iterations;out.linear_solves=local.linear_solves;
  out.row_space_solves=local.row_space_solves;
  out.line_search_backtracks=local.line_search_backtracks;
  // This entry point RETURNS HERE, even on stagnation. It cannot fall through
  // to local_restore/residual_model, a dense solve or a restart allocation.
  if(restoration_expired(st_)) {
    out.x=start;out.final_feasibility=out.initial_feasibility;out.success=false;
    return finish("sparse-only Phase-I deadline; private endpoint rolled back");
  }
  out.x=local.x;out.final_feasibility=local.theta;
  out.final_merit=local.merit;out.final_damping=local.damping;
  out.success=std::isfinite(local.theta)&&local.theta<=st_.feasibility_tolerance;
  return finish(out.success?"sparse-only Phase-I target reached; original filter/KKT still required":
      "sparse-only Phase-I stagnated; no dense fallback permitted");
}

RestorationResult FeasibilityRestoration::solve_elastic(const QCQPProblem& p,const Vec& start) const {
  const auto began=std::chrono::steady_clock::now();
  const long double n=p.n,m=2*p.equalities.size()+p.inequalities.size()+2*n+1;
  // Dense row vectors are part of this existing QP interface. Bound their
  // simultaneous storage before constructing any of them; this is not RSS
  // or sparse factor fill. Large cases keep the prior sparse-only route.
  const long double bytes=sizeof(double)*(6*n*m+8*n*n+64*n+16*m);
  if(!st_.restoration_failed_linearization_context||!st_.restoration_shared_iteration_budget||
      p.equalities.empty()||p.inequalities.empty()||p.has_signed_square()||
      !std::isfinite(bytes)||bytes>512.L*1024*1024||restoration_expired(st_))return solve(p,start);
  const int budget=std::max(0,st_.restoration_max_iterations);
  const auto mm=minimax_restore(p,start,st_,std::min(4,budget));
  Settings remaining=st_;remaining.restoration_max_iterations=std::max(0,budget-mm.qp_solves);
  auto out=FeasibilityRestoration(remaining).solve(p,mm.x);
  out.initial_feasibility=feasibility(p,start).total_inf;
  out.iterations+=mm.qp_solves;out.minimax_qp_solves+=mm.qp_solves;
  out.minimax_inner_iterations+=mm.inner_iterations;out.minimax_factorizations+=mm.factorizations;
  out.minimax_accepted_steps+=mm.accepted_steps;out.minimax_line_search_backtracks+=mm.line_search_backtracks;
  out.linear_solves+=mm.factorizations;
  out.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
  if(st_.live_display_level||st_.verbose)std::cout<<"[ELASTIC-PHASE-I] qp_passes="<<mm.qp_solves
      <<" inner_iterations="<<mm.inner_iterations<<" factorizations="<<mm.factorizations
      <<" accepted_private_steps="<<mm.accepted_steps<<" total_iterations="<<out.iterations
      <<" shared_budget="<<budget<<" primal="<<out.initial_feasibility<<"->"<<out.final_feasibility
      <<" linearized_equalities_relaxed=1 original_equalities_unchanged=1"
      <<" original_filter_required=1 final_KKT_certificate=0 backend="<<st_.linear_backend
      <<" estimated_auxiliary_bytes="<<static_cast<double>(bytes)<<std::endl;
  return out;
}

RestorationResult restore_mixed_startup(const QCQPProblem& p,const Vec& start,
    const Settings& st,NativeRestorationWork* work) {
  const auto begin=std::chrono::steady_clock::now();
  const auto* trace=active_native_restoration_trace;
  const NativeRestorationTrace trace_before=trace?*trace:NativeRestorationTrace{};
  auto checked_return=[&](RestorationResult result) {
    if(work&&trace) {
      const auto factors=trace->factorization_calls()-trace_before.factorization_calls();
      const auto solves=trace->linear_system_calls()-trace_before.linear_system_calls();
      add_native_restoration_count(work->factorization_calls,factors);
      add_native_restoration_count(work->linear_solve_calls,solves);
      add_native_restoration_count(work->backend_regularization_attempts,trace->backend_regularization_attempts-trace_before.backend_regularization_attempts);
      work->factorization_seconds+=trace->factor_api_seconds-trace_before.factor_api_seconds;
      work->linear_solve_seconds+=(trace->backend_solve_api_seconds-trace_before.backend_solve_api_seconds)+
          (trace->iterative_system_seconds-trace_before.iterative_system_seconds);
    }
    double maximum;long double merit;
    if(!column_lm_checked_values(p,result.x,maximum,merit)) {
      result.success=false;result.final_feasibility=std::numeric_limits<double>::infinity();
      result.message="mixed startup: rejected nonfinite restoration checkpoint";
    }
    result.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();return result;
  };
  auto invalid_start=[&](const Vec& point) {
    RestorationResult invalid;invalid.x=point;
    invalid.initial_feasibility=invalid.final_feasibility=std::numeric_limits<double>::infinity();
    invalid.message="mixed startup: nonfinite supplied or projected point or original row";
    return checked_return(std::move(invalid));
  };
  double supplied_maximum;long double supplied_merit;
  if(!column_lm_checked_values(p,start,supplied_maximum,supplied_merit))return invalid_start(start);
  const Vec initial=clamp_to_bounds(p,start);
  if(!column_lm_checked_values(p,initial,supplied_maximum,supplied_merit))return invalid_start(initial);
  if(restoration_expired(st)||st.restoration_max_iterations<=0)
    return checked_return(FeasibilityRestoration(st).solve(p,start));
  // Preserve the established bounded Gauss-Newton startup gate before
  // introducing a different scaled LM path. Small or structured models
  // already have an admitted historical restoration route; merely adding
  // the shared-budget accounting must not replace its feasible basin.
  // The ordinary solver charges all work to this same remaining allowance.
  if(p.n<=64||!detect_abs_links(p).empty())
    return checked_return(FeasibilityRestoration(st).solve(p,start));
  const auto admission=column_lm_admission(p,initial,st,true);
  if(st.live_display_level||st.verbose)std::cout<<"[MIXED-STARTUP-COLUMN-LM] admitted="<<admission.allowed
      <<" column_ratio="<<admission.column_ratio<<" free="<<admission.free.size()
      <<" triplets="<<admission.estimated_triplets<<" explicit_startup_context=1"
      <<" original_merit=1 original_KKT_required=1"<<std::endl;
  if(!admission.allowed)return checked_return(FeasibilityRestoration(st).solve(p,start));
  auto result=checked_return(column_scaled_lm_restore(p,initial,st,admission,true,work,true));
  if(work&&(st.live_display_level||st.verbose))std::cout<<"[MIXED-STARTUP-COLUMN-LM-WORK] factors="
      <<work->factorization_calls<<" solves="<<work->linear_solve_calls<<" model_attempts="<<work->outer_attempts<<" accepted_steps="<<work->accepted_steps
      <<" factor_seconds="<<work->factorization_seconds<<" solve_seconds="<<work->linear_solve_seconds
      <<" subset_of_preparation=1 shared_iteration_limit="<<st.restoration_max_iterations<<std::endl;
  return result;
}

RestorationResult FeasibilityRestoration::solve(const QCQPProblem& p, const Vec& start) const {
  RestorationResult out;
  ExactSparseRowDotRuntimeState sparse_runtime;
  const auto phase1_t0=std::chrono::steady_clock::now();
  if(st_.restoration_shared_iteration_budget&&st_.restoration_max_iterations<=0) {
    out.x=clamp_to_bounds(p,start);out.initial_feasibility=out.final_feasibility=feasibility(p,out.x).total_inf;
    out.success=out.final_feasibility<=st_.feasibility_tolerance;
    out.message="Phase-I shared iteration budget exhausted";return out;
  }
  if(!restoration_expired(st_)) {
    const Vec initial=clamp_to_bounds(p,start);
    const auto admission=column_lm_admission(p,initial,st_);
    if(admission.allowed) {
      const bool watched=st_.majorant_restoration_bound_stall_fix&&st_.majorant_adaptive_globalization&&
          st_.restoration_failed_linearization_context&&st_.restoration_shared_iteration_budget&&
          near_feasible_sparse_inequality_candidate(p,initial,st_);
      auto primary=column_scaled_lm_restore(p,initial,st_,admission,watched);
      int remaining=std::max(0,st_.restoration_max_iterations-primary.iterations);
      // A damped J'J model can stop at an infeasible stationary point of the
      // ORIGINAL squared residual. Its exact merit Hessian may still have
      // negative curvature. Release the unused private budget to the existing
      // certified curvature escape before returning a failed linearization.
      // Ordinary successful restoration trajectories are left unchanged.
      if(!primary.success&&remaining>1&&!restoration_expired(st_)&&
         st_.restoration_failed_linearization_context&&st_.restoration_shared_iteration_budget) {
        const auto residuals=residual_model(p,primary.x,{});
        const Vec gradient=merit_gradient(residuals,p.n);
        if(inf_norm(gradient)<=1e-6*std::max(1.0,squared_merit(residuals))) {
          Vec escaped=primary.x;
          const auto curvature=restoration_negative_curvature_step(p,escaped,st_,true);
          if(curvature.accepted) {
            Settings continuation=st_;
            continuation.restoration_max_iterations=remaining-1;
            const auto next_admission=column_lm_admission(p,escaped,continuation);
            if(next_admission.allowed) {
              auto next=column_scaled_lm_restore(p,escaped,continuation,next_admission,watched);
              next.initial_feasibility=primary.initial_feasibility;
              next.iterations+=primary.iterations+1;
              next.linear_solves+=primary.linear_solves;
              next.line_search_backtracks+=primary.line_search_backtracks;
              next.stagnation_events+=primary.stagnation_events;
              next.total_seconds=std::chrono::duration<double>(
                  std::chrono::steady_clock::now()-phase1_t0).count();
              if(st_.live_display_level||st_.verbose)
                std::cout<<"[RESTORATION-COLUMN-CURVATURE] accepted=1"
                  <<" merit_gradient_inf="<<inf_norm(gradient)
                  <<" iterations="<<next.iterations<<" shared_budget="<<st_.restoration_max_iterations
                  <<" primal="<<primary.initial_feasibility<<"->"<<primary.final_feasibility
                  <<"->"<<next.final_feasibility
                  <<" original_merit=1 original_filter_required=1 final_KKT_certificate=0"<<std::endl;
              primary=std::move(next);
              remaining=std::max(0,st_.restoration_max_iterations-primary.iterations);
            }
          }
        }
      }
      const bool stalled=watched&&primary.stagnation_events>0&&!primary.success&&remaining>0&&
          !restoration_expired(st_)&&near_feasible_sparse_inequality_candidate(p,primary.x,st_);
      if(!stalled)return primary;
      // The absolute deadline and iteration allowance are inherited from the
      // SAME private Phase I. No new anchor, filter entry or certificate is
      // accepted here; the outer original filter will judge the result.
      const auto legacy=legacy_local_restore(p,primary.x,st_,remaining,0,true);
      out=primary;out.x=legacy.x;out.final_feasibility=feasibility(p,out.x).total_inf;
      out.final_merit=legacy.merit;out.final_damping=legacy.damping;
      out.iterations+=legacy.iterations;out.linear_solves+=legacy.linear_solves;
      out.row_space_solves+=legacy.row_space_solves;out.line_search_backtracks+=legacy.line_search_backtracks;
      out.success=out.final_feasibility<=st_.feasibility_tolerance;
      out.message="original-residual-stalled column LM followed by closed-bound row-space recovery";
      out.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase1_t0).count();
      if(st_.live_display_level||st_.verbose)std::cout<<"[RESTORATION-BOUND-STALL-FALLBACK]"
        <<" column_iterations="<<primary.iterations<<" remaining_budget="<<remaining
        <<" rowspace_iterations="<<legacy.iterations<<" total_iterations="<<out.iterations
        <<" shared_budget="<<st_.restoration_max_iterations<<" closed_input_bounds=1"
        <<" primal="<<primary.initial_feasibility<<"->"<<primary.final_feasibility<<"->"<<out.final_feasibility
        <<" row_space_solves="<<out.row_space_solves<<" deadline_preserved=1"
        <<" original_merit=1 original_filter_required=1 final_KKT_certificate=0"<<std::endl;
      return out;
    }
  }
  const auto links = detect_abs_links(p);
  out.structured_abs_links = static_cast<int>(links.size());
  out.effective_dimension = p.n - static_cast<int>(links.size());

  Vec structured_start = clamp_to_bounds(p, start);
  project_abs_links(p, links, structured_start);
  out.initial_feasibility = feasibility(p, structured_start).total_inf;
  // A decade Phase-I continuation changes its private anchor, but should
  // not silently shrink the authorized GLOBAL restart search scale with
  // every contraction. Keep the frozen supplied problem start as reference;
  // local restart radii continue to use the current best residual. Existing
  // radius caps and the shared iteration/deadline budgets remain mandatory.
  const double global_restart_reference=st_.restoration_shared_iteration_budget?
      detail::phase1_global_restart_reference(out.initial_feasibility,
          feasibility(p,clamp_to_bounds(p,p.x0)).total_inf):
      out.initial_feasibility;
  out.harmonic_feasibility = out.initial_feasibility;
  out.tangent_activation_feasibility = out.initial_feasibility;
  const auto fixed = fixed_mask(p, st_.fixed_bound_tolerance);
  const auto aux = auxiliary_abs_mask(p, links);
  std::vector<char> restart_fixed = fixed;
  for (int j = 0; j < p.n; ++j) if (aux[j]) restart_fixed[j] = 1;

  // Stability gate: first reproduce the v0.2.2 Gauss-Newton Phase-I for a
  // short probe.  Problems that were already robust in v0.2.2 (BT1,
  // HIMMELBD, DISCS) therefore enter SQP from the same feasible point.
  // Structured abs-link problems keep the full legacy budget because v0.2.2
  // already solved CORE2 reliably on the reduced manifold.  Only unresolved
  // cases fall through to the exact-QCQP Newton restoration below.
  // Stability gate is intentionally limited to small ordinary QCQPs and to
  // structured abs-link models.  The small-problem threshold preserves the
  // v0.2.2 trajectories of BT1/HIMMELBD/DISCS without paying an O(n^3)
  // Gauss-Newton probe on medium/large instances such as A4X12/ALJAZZAF.
  const bool equality_only_row_space = links.empty() &&
      large_sparse_equality_chain_candidate(p);
  const bool near_feasible_inequality_row_space=links.empty() &&
      near_feasible_sparse_inequality_candidate(p,structured_start,st_);
  bool tangent_activated=false;
  if(equality_only_row_space && !restoration_expired(st_)) {
    const auto harmonic=anchored_harmonic_initialization(p,structured_start,st_);
    out.harmonic_seconds += harmonic.harmonic_seconds;
    out.tangent_activation_seconds += harmonic.tangent_seconds;
    out.harmonic_unknowns += harmonic.unknowns;
    out.harmonic_graph_edges += harmonic.graph_edges;
    out.tangent_activation_gradient_rows += harmonic.tangent_gradient_rows;
    out.tangent_activation_line_quadratics += harmonic.tangent_line_quadratics;
    out.tangent_activation_candidates += harmonic.tangent_candidates;
    out.harmonic_attempts += harmonic.attempted?1:0;
    out.harmonic_factorizations += harmonic.factorizations;
    if(harmonic.accepted) {
      structured_start=harmonic.x;
      out.harmonic_acceptances += 1;
      // HarmonicInitialization::after is the final (possibly activated)
      // start. Preserve the pre-activation harmonic residual separately.
      out.harmonic_feasibility=harmonic.tangent_attempted
          ? harmonic.tangent_before : harmonic.after;
    }
    tangent_activated=harmonic.tangent_accepted;
    out.tangent_activation_attempts += harmonic.tangent_attempted?1:0;
    out.tangent_activation_acceptances += harmonic.tangent_accepted?1:0;
    out.tangent_activation_components += harmonic.tangent_components;
    out.tangent_activation_feasibility = harmonic.tangent_attempted
        ? harmonic.tangent_after : feasibility(p,structured_start).total_inf;
    out.tangent_activation_step_inf = harmonic.tangent_step_inf;
    if(st_.verbose && harmonic.attempted) {
      std::cout << "[PHASE-I-HARMONIC] accepted="<<(harmonic.accepted?1:0)
                << " theta="<<std::scientific<<harmonic.before
                << " -> "<<(harmonic.tangent_attempted
                                ? harmonic.tangent_before : harmonic.after)
                << " unknown_factorizations="<<harmonic.factorizations
                << " graph_edges="<<harmonic.graph_edges
                << " unknowns="<<harmonic.unknowns
                << " seconds="<<harmonic.harmonic_seconds
                << "\n"<<std::defaultfloat;
      if(harmonic.tangent_attempted) {
        std::cout << "[PHASE-I-TANGENT-ACTIVATION] accepted="
                  <<(harmonic.tangent_accepted?1:0)
                  << " components="<<harmonic.tangent_components
                  << " theta="<<std::scientific<<harmonic.tangent_before
                  << " -> "<<harmonic.tangent_after
                  << " objective="<<harmonic.tangent_objective_before
                  << " -> "<<harmonic.tangent_objective_after
                  << " step="<<harmonic.tangent_step_inf
                  << " gradient_rows="<<harmonic.tangent_gradient_rows
                  << " line_quadratics="<<harmonic.tangent_line_quadratics
                  << " candidates="<<harmonic.tangent_candidates
                  << " seconds="<<harmonic.tangent_seconds
                  << "\n"<<std::defaultfloat;
      }
    }
  }
  const bool native_gauss_newton=p.has_signed_square();
  const bool use_legacy_gate = native_gauss_newton || !links.empty() || p.n <= 64 ||
      equality_only_row_space || near_feasible_inequality_row_space;
  LegacyLocalRestoration legacy;
  legacy.theta = std::numeric_limits<double>::infinity();
  legacy.merit = std::numeric_limits<double>::infinity();
  if (use_legacy_gate && !restoration_expired(st_)) {
    const int activated_probe_budget=std::max(8,2*static_cast<int>(
        std::ceil(std::log2(1.0+static_cast<double>(p.equalities.size())))));
    const int legacy_probe_budget = equality_only_row_space
        ? (tangent_activated
            ? std::min(st_.restoration_max_iterations,activated_probe_budget)
            : st_.restoration_max_iterations)
        : (near_feasible_inequality_row_space
            ? std::min(st_.restoration_max_iterations,8)
        : (links.empty() && !native_gauss_newton ? std::min(st_.restoration_max_iterations,120)
                         : st_.restoration_max_iterations));
    const auto legacy_t0=std::chrono::steady_clock::now();
    // Phase I has no logarithmic barrier. A native C1 signed-square model
    // may need an exact finite input bound to become feasible. Use its existing
    // closed-bound active-set path rather than repeatedly approaching that
    // same side with an IPM fraction. Ordinary historical models are unchanged.
    legacy = legacy_local_restore(p, structured_start, st_, std::max(1, legacy_probe_budget),0,native_gauss_newton);
    out.legacy_gate_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now()-legacy_t0).count();
    out.iterations += legacy.iterations;
    out.linear_solves += legacy.linear_solves;
    out.row_space_solves += legacy.row_space_solves;
    out.line_search_backtracks += legacy.line_search_backtracks;
    if (st_.verbose && equality_only_row_space) {
      std::cout << "[PHASE-I-LEGACY] iterations="<<legacy.iterations
                << " row_space_solves="<<legacy.row_space_solves
                << " backtracks="<<legacy.line_search_backtracks
                << " theta="<<std::scientific<<legacy.theta
                << " seconds="<<out.legacy_gate_seconds
                << "\n"<<std::defaultfloat;
    }
    if(st_.verbose && near_feasible_inequality_row_space) {
      std::cout << "[PHASE-I-SPARSE-ACTIVE-ROW] iterations="
                <<legacy.iterations
                << " row_space_solves="<<legacy.row_space_solves
                << " backtracks="<<legacy.line_search_backtracks
                << " theta="<<std::scientific<<out.initial_feasibility
                << " -> "<<legacy.theta
                << " seconds="<<out.legacy_gate_seconds
                << "\n"<<std::defaultfloat;
    }

    if (legacy.theta <= st_.feasibility_tolerance || native_gauss_newton) {
      out.x = legacy.x;
      out.final_feasibility = legacy.theta;
      out.final_merit = legacy.merit;
      out.final_damping = legacy.damping;
      out.success = legacy.theta <= st_.feasibility_tolerance;
      out.message = native_gauss_newton?"Phase-I native signed-square Gauss-Newton":"Phase-I legacy-stability gate converged";
      out.total_seconds=std::chrono::duration<double>(
          std::chrono::steady_clock::now()-phase1_t0).count();
      return out;
    }
  }

  // Unresolved cases start the exact Newton model from the original
  // structured point.  A failed Gauss-Newton endpoint is not used as a warm
  // start because it can sit in a heavily damped poor basin (ALJAZZAF).
  // For the large equality-chain path, preserve a genuinely improved
  // row-space endpoint instead of throwing it away and repeating a full
  // n-space restoration from the original pathological start.  Other models
  // retain the historical conservative restart policy.
  Vec exact_start=structured_start;
  if(equality_only_row_space && std::isfinite(legacy.theta) &&
     legacy.theta<feasibility(p,structured_start).total_inf)
    exact_start=legacy.x;
  if(st_.restoration_shared_iteration_budget&&out.iterations>=st_.restoration_max_iterations) {
    out.x=legacy.x;out.final_feasibility=legacy.theta;out.final_merit=legacy.merit;out.final_damping=legacy.damping;
    out.success=out.final_feasibility<=st_.feasibility_tolerance;
    out.message="Phase-I shared iteration budget exhausted after legacy probe";
    out.total_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-phase1_t0).count();
    return out;
  }
  int exact_budget=st_.restoration_shared_iteration_budget?
      std::max(0,st_.restoration_max_iterations-out.iterations):
      (equality_only_row_space?std::max(20,st_.restoration_max_iterations-legacy.iterations):st_.restoration_max_iterations);
  // Tiny ordinary models already authorize deterministic coordinate restarts.
  // Under the shared cap the first exact-Newton basin must not exhaust the
  // allowance before any of those authorized directions can be examined.
  // Successful legacy probes return above, so their trajectories are intact.
  if(st_.restoration_shared_iteration_budget&&links.empty()&&p.n<=16&&
      st_.restoration_max_restarts>0&&exact_budget>0) {
    const int remaining=exact_budget;
    exact_budget=detail::shared_restart_allowance(remaining,
        st_.restoration_max_restarts+1,remaining);
    if(st_.live_display_level||st_.verbose)
      std::cout<<"[RESTORATION-SHARED-INITIAL-SHARE] available="<<remaining
        <<" initial="<<exact_budget<<" authorized_restarts="<<st_.restoration_max_restarts
        <<" shared_limit="<<st_.restoration_max_iterations
        <<" original_rows_and_start_unchanged=1"<<std::endl;
  }
  const auto exact_t0=std::chrono::steady_clock::now();
  LocalRestoration best = local_restore(
      p,exact_start,st_,exact_budget,sparse_runtime);
  out.exact_restoration_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now()-exact_t0).count();
  accumulate_local(out, best);
  if (st_.verbose && equality_only_row_space) {
    std::cout << "[PHASE-I-EXACT] iterations="<<best.iterations
              << " linear_solves="<<best.linear_solves
              << " matrix_free="<<best.matrix_free_solves
              << " cg_iters="<<best.cg_iterations
              << " matvecs="<<best.exact_matvecs
              << " omp_batches/teams="
              << best.exact_parallel_row_dot_batches<<"/"
              << best.exact_parallel_teams
              << " sparse_dot_batches/checks/fallbacks="
              << best.exact_sparse_row_dot_batches<<"/"
              << best.exact_sparse_row_dot_verifications<<"/"
              << best.exact_sparse_row_dot_fallbacks
              << " backtracks="<<best.line_search_backtracks
              << " theta="<<std::scientific<<best.theta
              << " seconds="<<out.exact_restoration_seconds
              << "\n"<<std::defaultfloat;
  }
  if (use_legacy_gate && legacy.theta < best.theta) {
    best.x = legacy.x;
    best.theta = legacy.theta;
    best.merit = legacy.merit;
    best.damping = legacy.damping;
  }

  // If the squared-residual Newton model stalls, switch objectives rather than
  // merely spending more restarts on the same L2 stationary point.  The
  // epigraph QP minimizes the same L-infinity violation used by the final
  // feasibility certificate.  It is deliberately disabled for structured
  // abs-link models, whose reduced-manifold restoration already converges.
  if (best.theta > st_.feasibility_tolerance && st_.restoration_minimax_enabled && links.empty() &&
      (!st_.restoration_shared_iteration_budget||out.iterations<st_.restoration_max_iterations) &&
      !std::isfinite(st_.restoration_deadline_seconds)) {
    const auto minimax_t0=std::chrono::steady_clock::now();
    const int minimax_budget=st_.restoration_shared_iteration_budget?
        std::min(st_.restoration_minimax_max_iterations,st_.restoration_max_iterations-out.iterations):st_.restoration_minimax_max_iterations;
    MinimaxRestoration mm = minimax_restore(p, best.x, st_, minimax_budget);
    out.minimax_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now()-minimax_t0).count();
    out.minimax_qp_solves += mm.qp_solves;
    out.minimax_inner_iterations += mm.inner_iterations;
    out.minimax_factorizations += mm.factorizations;
    out.minimax_accepted_steps += mm.accepted_steps;
    out.minimax_line_search_backtracks += mm.line_search_backtracks;
    out.iterations += mm.qp_solves;
    out.linear_solves += mm.factorizations;
    if (mm.theta < best.theta) {
      best.x = std::move(mm.x);
      best.theta = mm.theta;
      best.merit = squared_merit(residual_model(p, best.x, links));
    }
  }

  const auto restart_t0=std::chrono::steady_clock::now();
  if (best.theta > st_.feasibility_tolerance && st_.restoration_max_restarts > 0) {
    std::mt19937 rng(st_.random_seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (int r = 0; r < st_.restoration_max_restarts && best.theta > st_.feasibility_tolerance && !restoration_expired(st_); ++r) {
      if(st_.restoration_shared_iteration_budget&&out.iterations>=st_.restoration_max_iterations)break;
      Vec dir(p.n, 0.0);
      bool global_restart = false;
      double radius = 1.0;
      Vec seed;

      if (links.empty() && p.n <= 16) {
        // Preserve the successful wide coordinate escape used by v0.2.2 on
        // tiny disconnected feasibility sets (notably HIMMELBD).
        dir = small_problem_coordinate_restart(p, r, restart_fixed);
        global_restart = true;
        const int shell = r/std::max(1, 2*p.n);
        radius = std::min(50.0, std::max(1.0, std::sqrt(std::max(0.0, global_restart_reference))))
                 * std::pow(2.0, shell);
        seed = structured_start;
      } else {
        const int mode = r % 4;
        const bool random_restart = (mode == 0 || mode == 2);
        if (random_restart) {
          double n2 = 0.0;
          for (int j = 0; j < p.n; ++j) if (!restart_fixed[j]) {
            dir[j] = normal(rng);
            n2 += dir[j]*dir[j];
          }
          if (n2 <= 0.0) break;
          const double inv = 1.0/std::sqrt(n2);
          for (double& v : dir) v *= inv;
        } else {
          dir = coordinate_restart(p, r, restart_fixed);
        }
        global_restart = (mode >= 2);
        const int shell = r/4;
        const double local_radius = std::min(4.0, std::max(1e-3, std::sqrt(std::max(0.0, best.theta))))
                                    * std::pow(1.35, shell);
        const double global_radius = std::min(50.0, std::max(1.0, std::sqrt(std::max(0.0, global_restart_reference))))
                                     * std::pow(1.20, shell);
        radius = global_restart ? global_radius : local_radius;
        seed = global_restart ? structured_start : best.x;
      }
      for (int j = 0; j < p.n; ++j) {
        double sj = global_restart ? 1.0 : std::max(1.0, std::abs(best.x[j]));
        if (std::isfinite(p.lb[j]) && std::isfinite(p.ub[j]))
          sj = std::max(1e-6, std::min(sj, 0.25*std::max(1e-12, p.ub[j]-p.lb[j])));
        seed[j] += radius*sj*dir[j];
      }
      seed = clamp_to_bounds(p, std::move(seed));
      for (int j = 0; j < p.n; ++j) if (fixed[j]) seed[j] = p.lb[j];
      project_abs_links(p, links, seed);

      // In a shared Phase-I budget, reserve a fair remaining share for each
      // authorized deterministic restart. Otherwise the first poor basins can
      // exhaust the cap before the remaining directions are even considered.
      // Historical callers retain their full per-restart trajectory.
      const int remaining_iterations=st_.restoration_max_iterations-out.iterations;
      const int remaining_restarts=std::max(1,st_.restoration_max_restarts-r);
      const int restart_budget=st_.restoration_shared_iteration_budget?
          detail::shared_restart_allowance(remaining_iterations,remaining_restarts,
              st_.restoration_restart_iterations):
          st_.restoration_restart_iterations;
      LocalRestoration cand = local_restore(
          p,std::move(seed),st_,restart_budget,
          sparse_runtime);
      accumulate_local(out, cand);
      ++out.restarts;
      if (cand.theta < best.theta ||
          (cand.theta <= best.theta*(1.0 + 1e-12) && cand.merit < best.merit)) best = std::move(cand);
    }
  }
  out.restart_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now()-restart_t0).count();

  out.x = best.x;
  out.final_feasibility = best.theta;
  out.final_merit = best.merit;
  out.final_damping = best.damping;
  out.success = best.theta <= st_.feasibility_tolerance;
  out.message = out.success ? "Phase-I hybrid restoration converged" :
      (restoration_expired(st_) ? "Phase-I elapsed-time limit" :
      "Phase-I hybrid restoration stagnated above feasibility tolerance");
  out.total_seconds=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-phase1_t0).count();
  if (st_.verbose && equality_only_row_space) {
    std::cout << "[PHASE-I-PROFILE] harmonic="<<out.harmonic_seconds
              << " tangent="<<out.tangent_activation_seconds
              << " legacy="<<out.legacy_gate_seconds
              << " exact="<<out.exact_restoration_seconds
              << " minimax="<<out.minimax_seconds
              << " restart="<<out.restart_seconds
              << " total="<<out.total_seconds
              << "\n";
  }
  return out;
}

RestorationResult FeasibilityRestoration::polish(const QCQPProblem& p, const Vec& start,
                                                  int max_iterations) const {
  RestorationResult out;
  const auto links = detect_abs_links(p);
  out.structured_abs_links = static_cast<int>(links.size());
  out.effective_dimension = p.n - static_cast<int>(links.size());
  Vec x0 = clamp_to_bounds(p, start);
  project_abs_links(p, links, x0);
  out.initial_feasibility = feasibility(p, x0).total_inf;
  LegacyLocalRestoration loc = legacy_local_restore(p, x0, st_, std::max(1, max_iterations));
  out.iterations += loc.iterations;
  out.linear_solves += loc.linear_solves;
  out.row_space_solves += loc.row_space_solves;
  out.x = loc.x;
  out.final_feasibility = loc.theta;
  out.final_merit = loc.merit;
  out.final_damping = loc.damping;
  out.success = loc.theta <= st_.feasibility_tolerance;
  out.message = out.success ? "legacy-stable feasibility polish converged" : "legacy-stable feasibility polish incomplete";
  return out;
}

Vec retract_equalities_to_targets(const QCQPProblem& p, const Vec& start,
                                  const Vec& targets,
                                  int max_iterations, double tolerance,
                                  double regularization,
                                  double fraction_to_boundary) {
  if (p.equalities.empty()) return clamp_to_bounds(p, start);
  if (targets.size() != p.equalities.size())
    throw std::invalid_argument("retract_equalities_to_targets: target dimension mismatch");

  const auto links = detect_abs_links(p);
  const auto auxiliary = auxiliary_abs_mask(p, links);
  std::vector<char> skip_eq(p.equalities.size(), 0);
  for (const auto& link : links)
    if (link.equality_index >= 0 &&
        link.equality_index < static_cast<int>(skip_eq.size()))
      skip_eq[link.equality_index] = 1;

  Vec x = clamp_to_bounds(p, start);
  const auto fixed = fixed_mask(p, 1e-14);
  for (int j = 0; j < p.n; ++j) if (fixed[j]) x[j] = p.lb[j];
  project_abs_links(p, links, x);

  // The explicit equations t^2-q^2=0 are enforced by projection.  Every other
  // equality is differentiated through t=|q|, exactly as in the validated
  // reduced Phase-I model.  Solving in the full lifted space and projecting
  // afterwards destroys the Newton direction and was the source of the
  // CORE2 correction floor in v0.4.16.
  std::vector<int> eq_indices;
  eq_indices.reserve(p.equalities.size());
  for (int i = 0; i < static_cast<int>(p.equalities.size()); ++i)
    if (!skip_eq[i]) eq_indices.push_back(i);

  for (int it = 0; it < max_iterations; ++it) {
    const int m = static_cast<int>(eq_indices.size());
    Vec r(m, 0.0);
    std::vector<Vec> J(m);
    double einf = 0.0;
    for (int a = 0; a < m; ++a) {
      const int i = eq_indices[a];
      r[a] = p.equalities[i].value(x) - targets[i];
      J[a] = reduce_gradient(p.equalities[i].gradient(x), x, links);
      einf = std::max(einf, std::abs(r[a]));
    }
    // Projected abs-link rows have target zero by construction.  Include their
    // actual residual in the stopping test without adding dependent rows to J.
    for (const auto& link : links)
      einf = std::max(einf, std::abs(p.equalities[link.equality_index].value(x)));
    if (einf <= tolerance || m == 0) break;

    std::vector<int> free;
    free.reserve(p.n);
    for (int j = 0; j < p.n; ++j)
      if (!fixed[j] && !auxiliary[j]) free.push_back(j);
    if (free.empty()) break;

    std::vector<double> A(static_cast<std::size_t>(m)*m, 0.0);
    for (int a = 0; a < m; ++a) {
      for (int b = 0; b <= a; ++b) {
        double value = 0.0;
        for (int j : free) value += J[a][j]*J[b][j];
        A[b + a*m] = value;
        A[a + b*m] = value;
      }
      A[a + a*m] += regularization;
    }
    Vec lambda;
    if (!solve_dense_system(std::move(A), r, lambda)) break;

    Vec w(p.n, 0.0);
    for (int a = 0; a < m; ++a)
      for (int j : free) w[j] -= J[a][j]*lambda[a];

    double alpha = 1.0;
    for (int j : free) {
      if (w[j] > 0.0 && std::isfinite(p.ub[j])) {
        const double room = p.ub[j]-x[j];
        alpha = std::min(alpha, room > 0.0 ? fraction_to_boundary*room/w[j] : 0.0);
      } else if (w[j] < 0.0 && std::isfinite(p.lb[j])) {
        const double room = p.lb[j]-x[j];
        alpha = std::min(alpha, room < 0.0 ? fraction_to_boundary*room/w[j] : 0.0);
      }
    }
    alpha = std::max(0.0, std::min(1.0, alpha));
    if (!(alpha > 0.0)) break;

    bool accepted = false;
    for (int ls = 0; ls < 16; ++ls) {
      Vec trial = x;
      axpy(alpha, w, trial);
      trial = clamp_to_bounds(p, std::move(trial));
      for (int j = 0; j < p.n; ++j) if (fixed[j]) trial[j] = p.lb[j];
      project_abs_links(p, links, trial);

      double ti = 0.0;
      for (int i : eq_indices)
        ti = std::max(ti, std::abs(p.equalities[i].value(trial)-targets[i]));
      for (const auto& link : links)
        ti = std::max(ti,
            std::abs(p.equalities[link.equality_index].value(trial)));
      if (ti < einf) {
        x = std::move(trial);
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }
  return x;
}

Vec retract_equalities_to_targets_full_space(
    const QCQPProblem& p, const Vec& start, const Vec& targets,
    int max_iterations, double tolerance, double regularization,
    double fraction_to_boundary) {
  if (p.equalities.empty()) return clamp_to_bounds(p, start);
  if (targets.size() != p.equalities.size())
    throw std::invalid_argument(
        "retract_equalities_to_targets_full_space: target dimension mismatch");

  Vec x = clamp_to_bounds(p, start);
  const auto fixed = fixed_mask(p, 1e-14);
  for (int j = 0; j < p.n; ++j) if (fixed[j]) x[j] = p.lb[j];

  // This is deliberately a full-space correction.  In particular, do not
  // call project_abs_links() here.  At a nearly feasible lifted point, that
  // projection can create an O(1) change in other badly scaled equalities even
  // when the requested SQP defect tends to zero.  Keeping the link rows in J
  // makes p=0 the exact correction for a zero defect.
  const int m = static_cast<int>(p.equalities.size());
  for (int it = 0; it < max_iterations; ++it) {
    Vec r(m, 0.0);
    std::vector<Vec> J(m);
    Vec row_scale(m, 1.0);
    double einf = 0.0;
    for (int i = 0; i < m; ++i) {
      r[i] = p.equalities[i].value(x) - targets[i];
      J[i] = p.equalities[i].gradient(x);
      double n2 = 0.0;
      for (int j = 0; j < p.n; ++j)
        if (!fixed[j]) n2 += J[i][j] * J[i][j];
      // Row scaling is essential on CORE-like models, where equality rows may
      // differ by many orders of magnitude.  It changes only the regularized
      // least-norm realization, not the target equations.
      row_scale[i] = std::max(1.0, std::sqrt(std::max(0.0, n2)));
      einf = std::max(einf, std::abs(r[i]));
    }
    if (einf <= tolerance) break;

    std::vector<int> free;
    free.reserve(p.n);
    for (int j = 0; j < p.n; ++j) if (!fixed[j]) free.push_back(j);
    if (free.empty()) break;

    // Minimum-norm normal correction with row equilibration:
    //   (D J J^T D + reg I) lambda = D r,
    //   w = -J^T D lambda,
    // where D_ii = 1/||J_i||.  The map r -> w is homogeneous, hence the
    // correction necessarily vanishes with the line-search defect.
    std::vector<double> A(static_cast<std::size_t>(m) * m, 0.0);
    Vec rhs(m, 0.0);
    for (int a = 0; a < m; ++a) {
      rhs[a] = r[a] / row_scale[a];
      for (int b = 0; b <= a; ++b) {
        double value = 0.0;
        for (int j : free) value += J[a][j] * J[b][j];
        value /= row_scale[a] * row_scale[b];
        A[b + a*m] = value;
        A[a + b*m] = value;
      }
      A[a + a*m] += std::max(0.0, regularization);
    }

    Vec lambda;
    if (!solve_dense_system(std::move(A), std::move(rhs), lambda)) break;

    Vec w(p.n, 0.0);
    for (int a = 0; a < m; ++a) {
      const double la = lambda[a] / row_scale[a];
      for (int j : free) w[j] -= J[a][j] * la;
    }
    if (!std::isfinite(inf_norm(w)) || inf_norm(w) == 0.0) break;

    double alpha = 1.0;
    for (int j : free) {
      if (w[j] > 0.0 && std::isfinite(p.ub[j])) {
        const double room = p.ub[j] - x[j];
        alpha = std::min(alpha,
            room > 0.0 ? fraction_to_boundary * room / w[j] : 0.0);
      } else if (w[j] < 0.0 && std::isfinite(p.lb[j])) {
        const double room = p.lb[j] - x[j];
        alpha = std::min(alpha,
            room < 0.0 ? fraction_to_boundary * room / w[j] : 0.0);
      }
    }
    alpha = std::max(0.0, std::min(1.0, alpha));
    if (!(alpha > 0.0)) break;

    bool accepted = false;
    for (int ls = 0; ls < 24; ++ls) {
      Vec trial = x;
      axpy(alpha, w, trial);
      trial = clamp_to_bounds(p, std::move(trial));
      for (int j = 0; j < p.n; ++j) if (fixed[j]) trial[j] = p.lb[j];

      double ti = 0.0;
      for (int i = 0; i < m; ++i)
        ti = std::max(ti,
            std::abs(p.equalities[i].value(trial) - targets[i]));
      if (ti < einf) {
        x = std::move(trial);
        accepted = true;
        break;
      }
      alpha *= 0.5;
    }
    if (!accepted) break;
  }
  return x;
}

Vec retract_equalities(const QCQPProblem& p, const Vec& start,
                       int max_iterations, double tolerance,
                       double regularization, double fraction_to_boundary) {
  return retract_equalities_to_targets(
      p, start, Vec(p.equalities.size(), 0.0), max_iterations, tolerance,
      regularization, fraction_to_boundary);
}

} // namespace lazyipdca::qcqp
