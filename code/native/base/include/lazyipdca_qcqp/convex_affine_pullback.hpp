#pragma once
#include "lazyipdca_qcqp/inner_ipm.hpp"
#include <map>
namespace lazyipdca::qcqp {
// Coordinate representation d=b+T*y. This changes no function correction.
// Rounded coefficients are only a solver representation; callers must audit
// the returned point against the original full certified model.
struct ConvexAffineCoordinates {
 int ambient=0,reduced=0;
 Vec offset;
 std::vector<SparseRow> rows;
 void validate() const {
  if(ambient<1||reduced<1||offset.size()!=std::size_t(ambient)||rows.size()!=offset.size())
   throw std::invalid_argument("affine coordinates: dimensions");
  for(double x:offset)if(!std::isfinite(x))throw std::invalid_argument("affine coordinates: offset");
  for(const auto& row:rows)for(auto a:row.entries)
   if(a.first<0||a.first>=reduced||!std::isfinite(a.second))throw std::invalid_argument("affine coordinates: row");
 }
 Vec expand(const Vec& y) const {
  validate();if(y.size()!=std::size_t(reduced))throw std::invalid_argument("affine coordinates: point");
  Vec d=offset;for(int i=0;i<ambient;++i){long double v=offset[i];for(auto a:rows[i].entries)v+=(long double)a.second*y[a.first];d[i]=double(v);}return d;
 }
 Vec transpose(const Vec& g) const {
  validate();if(g.size()!=std::size_t(ambient))throw std::invalid_argument("affine coordinates: gradient");
  std::vector<long double> sum(reduced,0);for(int i=0;i<ambient;++i)for(auto a:rows[i].entries)sum[a.first]+=(long double)a.second*g[i];
  Vec v(reduced);for(int i=0;i<reduced;++i)v[i]=double(sum[i]);return v;
 }
};
inline ConvexQuadraticModel convex_affine_pullback(const ConvexQuadraticModel& q,const ConvexAffineCoordinates& t) {
 t.validate();if(q.linear.size()!=std::size_t(t.ambient)||q.base_hessian.n!=t.ambient)
  throw std::invalid_argument("quadratic pullback: dimensions");
 ConvexQuadraticModel r;r.constant=q.value(t.offset);r.linear=t.transpose(q.gradient(t.offset));r.base_hessian.n=t.reduced;
 std::map<std::pair<int,int>,long double> h;
 auto add=[&](int i,int j,double v){for(auto a:t.rows.at(i).entries)for(auto b:t.rows.at(j).entries)
   if(a.first<=b.first)h[{a.first,b.first}]+=(long double)v*a.second*b.second;};
 for(auto e:q.base_hessian.entries){add(e.i,e.j,e.v);if(e.i!=e.j)add(e.j,e.i,e.v);}
 if(q.scalar_shift)for(int i=0;i<q.shift_dimension;++i)add(i,i,q.scalar_shift);
 for(auto e:h)if(e.second!=0){double v=double(e.second);if(!std::isfinite(v))throw std::runtime_error("quadratic pullback: Hessian overflow");r.base_hessian.entries.push_back({e.first.first,e.first.second,v});}
 for(const auto& a:q.low_rank)r.low_rank.push_back({a.beta,t.transpose(a.direction)});
 if(!std::isfinite(r.constant))throw std::runtime_error("quadratic pullback: constant overflow");
 for(double x:r.linear)if(!std::isfinite(x))throw std::runtime_error("quadratic pullback: gradient overflow");
 return r;
}
} // namespace lazyipdca::qcqp
