#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <chrono>
#include <memory>

namespace lazyipdca::qcqp::detail {
struct PositiveSquareEpigraphRestoration {
  bool eligible=false,attempted=false,accepted=false;
  int epigraph=-1,factors=0,solves=0;
  Vec x;
  double before=INFINITY,after=INFINITY,seconds=0;
  std::string reason;
};
// Feasibility only. Admit the complete exact coefficient domain
//   A*x+b=0 (independent of t), sum a_ij*x_j^2 - b_i*t^2+c_i <= 0,
//   a_ij>=0, b_i>0, t>=0, objective c*t+constant, c>0.
// Constants of either sign remain in every exact rational inequality check.
// No claim of SOCP equivalence or convexity is made. No input is overwritten.
inline PositiveSquareEpigraphRestoration restore_positive_square_epigraph(
    const QCQPProblem& p,const Settings& requested,double absolute_deadline) {
  using Exact=boost::multiprecision::cpp_rational;
  using Clock=std::chrono::steady_clock;
  PositiveSquareEpigraphRestoration out;
  const auto began=Clock::now();
  auto expired=[&]{return std::chrono::duration<double>(Clock::now().time_since_epoch()).count()>=absolute_deadline;};
  auto finish=[&](const std::string& reason){out.reason=reason;out.seconds=std::chrono::duration<double>(Clock::now()-began).count();return out;};
  if(!requested.restoration_enabled||requested.restoration_max_iterations<1||
      p.equalities.empty()||p.inequalities.empty()||p.has_signed_square()||
      (requested.linear_backend!="mumps"&&requested.linear_backend!="ma27")||!requested.linear_metis||
      requested.rank_presolve_context||expired())return finish("outside-domain-or-budget");
  auto affine=[](const QuadraticFunction& q){return std::all_of(q.hessian.entries.begin(),q.hessian.entries.end(),[](const auto& h){return h.v==0.;});};
  if(!affine(p.objective))return finish("objective-not-affine");
  int t=-1,objective_terms=0;
  p.objective.for_each_linear_nonzero([&](int j,double a){++objective_terms;if(a>0)t=j;});
  if(objective_terms!=1||t<0||!std::isfinite(p.lb[t])||p.lb[t]<0||std::isfinite(p.ub[t]))return finish("no-positive-objective-epigraph");
  for(int j=0;j<p.n;++j)if(j!=t&&(std::isfinite(p.lb[j])||std::isfinite(p.ub[j])))return finish("other-finite-bounds");
  std::size_t nnz=0;
  for(const auto& q:p.equalities){
    if(!affine(q))return finish("nonaffine-equality");
    bool independent=true;q.for_each_linear_nonzero([&](int j,double){++nnz;if(j==t)independent=false;});
    if(!independent)return finish("equality-uses-epigraph");
  }
  for(const auto& q:p.inequalities){
    if(expired())return finish("domain-scan-deadline");
    bool valid=true;int negative=0;
    q.for_each_linear_nonzero([&](int,double){valid=false;});
    for(const auto& h:q.hessian.entries)if(h.v!=0){
      if(h.i!=h.j)valid=false;
      else if(h.i==t){if(h.v>=0)valid=false;else ++negative;}
      else if(h.v<0)valid=false;
    }
    if(!valid||negative!=1)return finish("inequality-not-shared-negative-square");
  }
  out.eligible=true;out.epigraph=t;
  out.before=feasibility(p,p.x0).total_inf;
  if(!std::isfinite(out.before)||out.before<=requested.feasibility_tolerance)return finish("already-feasible-or-nonfinite");
  const std::size_t dimension=std::size_t(p.n)+p.equalities.size();
  if(dimension>150000||nnz>5000000||p.n+2.L*nnz+dimension>12000000)return finish("sparse-workspace-cost-guard");
  Settings linear=requested;
  linear.kkt_adaptive_regularization=false;linear.kkt_primal_regularization=0;
  linear.kkt_dual_regularization=0;linear.kkt_preserve_structural_zeros=true;
  std::string message;
  auto factor=make_memory_bounded_symmetric_factorization(linear,message,128);
  if(!factor)return finish("native-factor-unavailable:"+message);
  std::vector<SymmetricTriplet> matrix;matrix.reserve(p.n+2*nnz);
  for(int j=0;j<p.n;++j)matrix.push_back({j,j,1});
  for(std::size_t e=0;e<p.equalities.size();++e)p.equalities[e].for_each_linear_nonzero(
      [&](int j,double a){matrix.push_back({j,int(p.n+e),a});});
  Vec rhs(dimension,0);
  for(std::size_t e=0;e<p.equalities.size();++e)rhs[p.n+e]=-p.equalities[e].value(p.x0);
  if(expired())return finish("projection-pre-factor-deadline");
  out.attempted=true;++out.factors;
  if(!factor->factor(int(dimension),p.n,matrix,message))return finish("affine-normal-factor-rejected:"+message);
  if(expired())return finish("projection-post-factor-deadline");
  Vec direction=rhs;++out.solves;
  if(!factor->solve(direction,message))return finish("affine-normal-solve-rejected:"+message);
  // Audit the actual unshifted normal equations, independently of MUMPS' berr.
  std::vector<long double> residual(dimension,0);
  for(std::size_t j=0;j<dimension;++j)residual[j]=-rhs[j];
  for(const auto& h:matrix){residual[h.i]+=(long double)h.value*direction[h.j];if(h.i!=h.j)residual[h.j]+=(long double)h.value*direction[h.i];}
  long double expanded=0;for(auto a:residual)expanded=std::max(expanded,std::abs(a));
  if(!std::isfinite(expanded)||expanded>requested.kkt_backward_error_reject*std::max(1.,inf_norm(rhs)))return finish("affine-normal-expanded-residual-rejected");
  out.x=p.x0;
  for(int j=0;j<p.n;++j)out.x[j]+=direction[j];
  for(double v:out.x)if(!std::isfinite(v)){out.x.clear();return finish("nonfinite-projection");}
  for(const auto& q:p.equalities)if(std::abs(q.value(out.x))>0.1*requested.feasibility_tolerance){out.x.clear();return finish("original-affine-residual-rejected");}
  Exact radius2=0;
  for(const auto& q:p.inequalities){
    if(expired()){out.x.clear();return finish("exact-radius-deadline");}
    Exact value(q.constant),coefficient=0;
    for(const auto& h:q.hessian.entries)if(h.v!=0){
      if(h.i==t)coefficient=-Exact(h.v)/2;
      else value+=Exact(h.v)*Exact(out.x[h.i])*Exact(out.x[h.i])/2;
    }
    const Exact ratio=value/coefficient;if(ratio>radius2)radius2=ratio;
  }
  double radius=std::sqrt(radius2.convert_to<long double>());
  if(!std::isfinite(radius)){out.x.clear();return finish("nonfinite-radius");}
  for(int tries=0;Exact(radius)*Exact(radius)<radius2;++tries){
    if(tries>=8){out.x.clear();return finish("outward-radius-not-representable");}
    radius=std::nextafter(radius,INFINITY);
  }
  out.x[t]=std::max({p.lb[t],p.x0[t],radius});
  out.after=feasibility(p,out.x).total_inf;
  out.accepted=!expired()&&std::isfinite(out.after)&&out.after<=requested.feasibility_tolerance;
  if(!out.accepted)out.x.clear();
  return finish(out.accepted?"affine-normal-plus-exact-outward-negative-square-repair":"original-feasibility-audit-rejected");
}
} // namespace lazyipdca::qcqp::detail
