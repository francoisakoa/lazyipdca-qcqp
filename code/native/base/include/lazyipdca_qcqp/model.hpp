#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lazyipdca::qcqp {

using Vec = std::vector<double>;

inline double inf_norm(const Vec& x) {
  double m = 0.0;
  for (double v : x) m = std::max(m, std::abs(v));
  return m;
}

inline double dot(const Vec& a, const Vec& b) {
  if (a.size() != b.size()) throw std::invalid_argument("dot: dimension mismatch");
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

inline double norm2(const Vec& x) { return std::sqrt(std::max(0.0, dot(x, x))); }

inline void axpy(double alpha, const Vec& x, Vec& y) {
  if (x.size() != y.size()) throw std::invalid_argument("axpy: dimension mismatch");
  for (std::size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

struct SymTriplet {
  int i = 0;
  int j = 0;
  double v = 0.0;
};

// Symmetric sparse matrix stored as unique upper-triangular triplets (i <= j).
struct SymMatrix {
  int n = 0;
  std::vector<SymTriplet> entries;

  void validate() const {
    if (n < 0) throw std::invalid_argument("SymMatrix: negative dimension");
    for (const auto& e : entries) {
      if (e.i < 0 || e.j < 0 || e.i >= n || e.j >= n || e.i > e.j)
        throw std::invalid_argument("SymMatrix: invalid upper-triangular triplet");
      if (!std::isfinite(e.v)) throw std::invalid_argument("SymMatrix: non-finite value");
    }
  }

  Vec apply(const Vec& x) const {
    if (static_cast<int>(x.size()) != n) throw std::invalid_argument("SymMatrix::apply: dimension mismatch");
    Vec y(n, 0.0);
    for (const auto& e : entries) {
      y[e.i] += e.v * x[e.j];
      if (e.i != e.j) y[e.j] += e.v * x[e.i];
    }
    return y;
  }

  double quad(const Vec& x) const {
    if (static_cast<int>(x.size()) != n) throw std::invalid_argument("SymMatrix::quad: dimension mismatch");
    double q = 0.0;
    for (const auto& e : entries) {
      if (e.i == e.j) q += e.v * x[e.i] * x[e.i];
      else q += 2.0 * e.v * x[e.i] * x[e.j];
    }
    return q;
  }

  double gershgorin_lower_bound() const {
    Vec diag(n, 0.0), radius(n, 0.0);
    for (const auto& e : entries) {
      if (e.i == e.j) diag[e.i] += e.v;
      else {
        radius[e.i] += std::abs(e.v);
        radius[e.j] += std::abs(e.v);
      }
    }
    double lb = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) lb = std::min(lb, diag[i] - radius[i]);
    return n == 0 ? 0.0 : lb;
  }
};

struct LowRankTerm {
  double beta = 0.0;
  Vec direction;
};

struct Convexification {
  double scalar_shift = 0.0;
  std::vector<LowRankTerm> terms;

  int rank() const { return static_cast<int>(terms.size()); }
};

struct QuadraticFunction {
  std::string name;
  double constant = 0.0;
  Vec linear;
  SymMatrix hessian;
  // Native C1 piecewise quadratic c*x*abs(x), generalized Hessian 0 at x=0.
  std::vector<std::pair<int,double>> signed_square;
  // Optional owning representation for input models. A nonempty dense linear
  // vector takes precedence, so existing callers may replace it safely.
  int compact_linear_dimension=0;
  std::vector<std::pair<int,double>> compact_linear;
  bool has_compact_linear() const {return linear.empty()&&compact_linear_dimension>0;}
  template<class F> void for_each_linear_nonzero(F visit) const {
    if(has_compact_linear()){for(const auto& e:compact_linear)visit(e.first,e.second);}
    else for(int j=0;j<(int)linear.size();++j)if(linear[j]!=0.0)visit(j,linear[j]);
  }
  void materialize_linear() {
    if(!has_compact_linear())return;
    linear.assign(compact_linear_dimension,0.0);
    for(const auto& e:compact_linear)linear[e.first]=e.second;
    compact_linear.clear();compact_linear_dimension=0;
  }

  void validate(int n) const {
    if(has_compact_linear()) {
      if(compact_linear_dimension!=n)throw std::invalid_argument("QuadraticFunction: compact linear dimension mismatch");
      int previous=-1;for(auto e:compact_linear){if(e.first<=previous||e.first>=n||!std::isfinite(e.second))throw std::invalid_argument("QuadraticFunction: invalid compact linear coefficient");previous=e.first;}
    } else if (static_cast<int>(linear.size()) != n) throw std::invalid_argument("QuadraticFunction: linear dimension mismatch");
    if (hessian.n != n) throw std::invalid_argument("QuadraticFunction: Hessian dimension mismatch");
    if (!std::isfinite(constant)) throw std::invalid_argument("QuadraticFunction: non-finite constant");
    hessian.validate();
    for(auto e:signed_square)if(e.first<0||e.first>=n||!std::isfinite(e.second))throw std::invalid_argument("invalid native signed-square term");
  }

  double value(const Vec& x) const {
    double native=0;for(auto e:signed_square)native+=e.second*x[e.first]*std::abs(x[e.first]);
    if(!has_compact_linear())return constant + dot(linear,x) + 0.5*hessian.quad(x)+native;
    if((int)x.size()!=compact_linear_dimension)throw std::invalid_argument("QuadraticFunction: compact value dimension mismatch");
    double v=0;for(auto e:compact_linear)v+=e.second*x[e.first];
    return constant+v+0.5*hessian.quad(x)+native;
  }

  Vec gradient(const Vec& x) const {
    Vec g = hessian.apply(x);
    if(has_compact_linear()){for(auto e:compact_linear)g[e.first]+=e.second;}else axpy(1.0,linear,g);
    for(auto e:signed_square)g[e.first]+=2*e.second*std::abs(x[e.first]);
    return g;
  }
};

struct QCQPProblem {
  std::string name;
  int n = 0;
  Vec x0;
  Vec lb;
  Vec ub;
  QuadraticFunction objective;
  std::vector<QuadraticFunction> equalities;   // h_j(x) = 0
  std::vector<QuadraticFunction> inequalities; // g_i(x) <= 0

  bool has_signed_square() const {
    if(!objective.signed_square.empty())return true;
    for(const auto& q:equalities)if(!q.signed_square.empty())return true;
    for(const auto& q:inequalities)if(!q.signed_square.empty())return true;
    return false;
  }
  void validate() const {
    if (n <= 0) throw std::invalid_argument("QCQPProblem: n must be positive");
    if (static_cast<int>(x0.size()) != n || static_cast<int>(lb.size()) != n || static_cast<int>(ub.size()) != n)
      throw std::invalid_argument("QCQPProblem: bound/start dimension mismatch");
    objective.validate(n);
    for (const auto& q : equalities) q.validate(n);
    for (const auto& q : inequalities) q.validate(n);
    for (int i = 0; i < n; ++i) {
      if (std::isnan(lb[i]) || std::isnan(ub[i]) || std::isnan(x0[i]))
        throw std::invalid_argument("QCQPProblem: NaN in bounds/start");
      if (lb[i] > ub[i]) throw std::invalid_argument("QCQPProblem: inconsistent bounds");
    }
  }
};

struct Feasibility {
  double equality_inf = 0.0;
  double inequality_inf = 0.0;
  double bound_inf = 0.0;
  double total_inf = 0.0;
};

inline Feasibility feasibility(const QCQPProblem& p, const Vec& x) {
  Feasibility f;
  for (const auto& h : p.equalities) f.equality_inf = std::max(f.equality_inf, std::abs(h.value(x)));
  for (const auto& g : p.inequalities) f.inequality_inf = std::max(f.inequality_inf, std::max(0.0, g.value(x)));
  for (int i = 0; i < p.n; ++i) {
    if (std::isfinite(p.lb[i])) f.bound_inf = std::max(f.bound_inf, std::max(0.0, p.lb[i] - x[i]));
    if (std::isfinite(p.ub[i])) f.bound_inf = std::max(f.bound_inf, std::max(0.0, x[i] - p.ub[i]));
  }
  f.total_inf = std::max({f.equality_inf, f.inequality_inf, f.bound_inf});
  return f;
}

} // namespace lazyipdca::qcqp
