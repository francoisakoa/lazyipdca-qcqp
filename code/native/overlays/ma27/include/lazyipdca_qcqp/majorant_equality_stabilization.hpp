#pragma once
#include "lazyipdca_qcqp/model.hpp"
#include "lazyipdca_qcqp/settings.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdlib>

namespace lazyipdca::qcqp::detail {

inline bool original_boolean_mode(const char* name,bool fallback) {
  const char* flag=std::getenv(name);
  if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")
    throw std::invalid_argument(std::string(name)+" expects 0 or 1");
  return flag?std::string(flag)=="1":fallback;
}

inline bool original_stabilized_merit_requested(const Settings& st) {
  return original_boolean_mode("LAZY_ORIGINAL_STABILIZED_MERIT",st.original_stabilized_merit)&&!st.original_private_phase;
}

inline bool original_source_primal_barrier_requested(const Settings& st) {
  const bool gr_enabled=original_stabilized_merit_requested(st);
  const bool enabled=original_boolean_mode("LAZY_ORIGINAL_SOURCE_PRIMAL_BARRIER",
      st.original_source_primal_barrier||gr_enabled);
  if(st.original_private_phase)return false;
  if(enabled&&(!gr_enabled||!st.majorant_equality_stabilization))
    throw std::invalid_argument("source primal barrier requires GR merit and explicit source-constraint stabilization");
  return enabled;
}

// Conservative exact coefficient certificate for the barrier-free equality
// extension. Every nonzero objective Hessian entry is a nonnegative diagonal,
// so positive semidefiniteness needs no eigenvalue estimate or tolerance.
// No finite bound or source inequality is admitted through this extension.
inline bool original_diagonal_convex_equality_domain(const QCQPProblem& p) {
  // Reject ineligible structures before inspecting objective curvature. This
  // predicate has no model, result, multiplier or coefficient side effects.
  if(!p.inequalities.empty()||p.equalities.empty()||p.has_signed_square())return false;
  for(int j=0;j<p.n;++j){
    const bool lo=std::isfinite(p.lb[j]),up=std::isfinite(p.ub[j]);
    if((lo||up)&&!(lo&&up&&p.lb[j]==p.ub[j]))return false;
  }
  bool curved=false;
  for(const auto& h:p.objective.hessian.entries) {
    if(h.v==0.0)continue;
    if(!(h.i==h.j&&h.v>=0.0))return false;
    curved=true;
  }
  return curved&&std::any_of(p.equalities.begin(),p.equalities.end(),[](const auto& q){
    return std::any_of(q.hessian.entries.begin(),q.hessian.entries.end(),
        [](const auto& h){return h.v!=0.0;});});
}

// Bound-free pure equalities also admit an arbitrary signed quadratic
// objective. The existing Newton operator keeps its exact Hessian; inertia
// correction and the actual Gill--Robinson directional derivative, followed
// by the unchanged Armijo test, govern trial acceptance. This predicate is
// structural eligibility, not a claim of convexity or constraint qualification.
inline bool original_quadratic_equality_domain(const QCQPProblem& p) {
  if(original_diagonal_convex_equality_domain(p))return true;
  if(!p.inequalities.empty()||p.equalities.empty()||p.has_signed_square()||
      !std::all_of(p.lb.begin(),p.lb.end(),[](double v){return v==-INFINITY;})||
      !std::all_of(p.ub.begin(),p.ub.end(),[](double v){return v==INFINITY;}))return false;
  const bool curved=std::any_of(p.objective.hessian.entries.begin(),p.objective.hessian.entries.end(),
      [](const auto& h){return h.v!=0.0;});
  return curved&&std::any_of(p.equalities.begin(),p.equalities.end(),[](const auto& q){
    return std::any_of(q.hessian.entries.begin(),q.hessian.entries.end(),
        [](const auto& h){return h.v!=0.0;});});
}

inline void validate_majorant_equality_stabilization_problem(
    const QCQPProblem& problem,const Settings& st) {
  original_source_primal_barrier_requested(st);
  if(!st.majorant_equality_stabilization)return;
  auto affine=[](const QuadraticFunction& q) {
    return q.signed_square.empty()&&std::all_of(q.hessian.entries.begin(),q.hessian.entries.end(),
        [](const auto& h){return h.v==0.0;});
  };
  const bool original_r3=st.study_controller=="r3"&&
      (st.convexification_study=="off"||st.convexification_study=="HH"||
       st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS");
  if(st.dca_engine!="majorants"||(st.outer_algorithm!="lazy-dca"&&st.outer_algorithm!="auto")||
      st.native_augmented_repair||st.product_definition_presolve)
    throw std::invalid_argument("equality stabilization requires original majorants, no augmented repair and no global product substitution");
  if(!original_r3||problem.has_signed_square()||
      (!affine(problem.objective)&&!original_quadratic_equality_domain(problem))||
      !std::all_of(problem.inequalities.begin(),problem.inequalities.end(),affine))
    throw std::invalid_argument("equality stabilization requires original R3 affine objective/inequalities or a signed quadratic objective with bound-free pure equalities");
  if((st.linear_backend!="mumps"&&st.linear_backend!="ma27")||!st.linear_metis||st.inner_backend=="ipopt")
    throw std::invalid_argument("equality stabilization requires native MUMPS+METIS or MA27+METIS");
}

// V2: algorithmic dual stabilization of ALL source constraint equations,
// h=0 and g+s=0, NOT a backend rescue shift. Physical bound rows keep rho=0.
// F_h = h - rho*(lambda-lambda_E), hence Newton's equality equation is
// J_h*d_x - rho*d_lambda = -F_h. Original h is never overwritten.
struct MajorantEqualityStabilization {
  static constexpr const char* variant="all-source-constraint-dual-stabilization-v2";
  static constexpr const char* primal_barrier_variant="source-primal-barrier-free-duals-v3";
  bool enabled=false;
  double rho=1.0;
  static constexpr double minimum_rho=1e-6;
  Vec multiplier_center;
  Vec source_inequality_center;
  // Historical field name; V2 stores the true residual of all source C.
  double reference_true_equality=INFINITY;
  int epochs=0,last_update_generation=-1;

  void reset(const Vec& lambda,double true_equality,const Vec& z={}) {
    multiplier_center=lambda;rho=1.0;reference_true_equality=true_equality;
    source_inequality_center=z;
    epochs=0;last_update_generation=-1;
  }
  Vec residual(const Vec& h,const Vec& lambda) const {
    if(!enabled)return h;
    if(h.size()!=lambda.size()||lambda.size()!=multiplier_center.size())
      throw std::invalid_argument("equality stabilization dimension mismatch");
    Vec f(h.size());
    for(std::size_t e=0;e<h.size();++e) {
      f[e]=static_cast<double>(static_cast<long double>(h[e])-
          static_cast<long double>(rho)*(static_cast<long double>(lambda[e])-multiplier_center[e]));
      if(!std::isfinite(f[e]))throw std::runtime_error("nonfinite stabilized equality residual");
    }
    return f;
  }
  Vec primal_residual(const Vec& rp,const Vec& z,const std::vector<int>& kind) const {
    if(!enabled)return rp;
    if(rp.size()!=z.size()||z.size()!=kind.size()||z.size()!=source_inequality_center.size())
      throw std::invalid_argument("source constraint stabilization dimension mismatch");
    Vec f=rp;
    for(std::size_t i=0;i<rp.size();++i)if(kind[i]==0) {
      f[i]=static_cast<double>(static_cast<long double>(rp[i])-
          static_cast<long double>(rho)*(static_cast<long double>(z[i])-source_inequality_center[i]));
      if(!std::isfinite(f[i]))throw std::runtime_error("nonfinite stabilized source-inequality residual");
    }
    return f;
  }
  // Only an accepted outer anchor with a sufficiently solved regularized
  // barrier system can close an epoch. This is not an every-Newton update.
  bool update_epoch(const Vec& lambda,double true_equality,
      double stabilized_barrier_residual,double mu,int generation,const Vec* z=nullptr,
      double zero_barrier_tolerance=0.0) {
    // With pure equalities there is no barrier continuation parameter. The
    // stabilized inner equations use the caller's decreasing forcing accuracy
    // before an epoch closes. This does not define original KKT termination.
    // The positive-barrier gate remains 10*mu, independently of this argument.
    const double threshold=mu>0?10.0*mu:zero_barrier_tolerance;
    if(!enabled||generation==last_update_generation||!(mu>=0)||
        !std::isfinite(mu)||!std::isfinite(threshold)||threshold<0||!std::isfinite(true_equality)||
        !std::isfinite(stabilized_barrier_residual)||
        stabilized_barrier_residual>threshold)return false;
    if(lambda.size()!=multiplier_center.size())
      throw std::invalid_argument("equality stabilization center dimension mismatch");
    if(z&&z->size()!=source_inequality_center.size())
      throw std::invalid_argument("source constraint stabilization center dimension mismatch");
    if(true_equality<=0.2*reference_true_equality||rho<=minimum_rho) {
      multiplier_center=lambda;reference_true_equality=true_equality;
      if(z)source_inequality_center=*z;
    } else rho=std::max(minimum_rho,0.1*rho);
    ++epochs;last_update_generation=generation;return true;
  }
};

} // namespace lazyipdca::qcqp::detail
