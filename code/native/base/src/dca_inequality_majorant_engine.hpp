#include "defined_pivot_soc.hpp"
#include "gr_short_step_proximal.hpp"
#include "gr_kkt_soc.hpp"
#include "dca_square_continuation.hpp"

// Counters are interface calls: a DGESV call is a combined dense factor/solve,
// PCG is one iterative system attempt (iterations separately reported), and
// DSYEV is spectral work. Never infer a factorization from RR.linear_solves.
// These measured timers are included in the existing preparation/restoration
// wall time and are not added to the exclusive Newton cost ledger again.
void dca_merge_native_restoration_trace(Result& result,const NativeRestorationTrace& work,
                                       const char* phase,int level) {
  add_native_restoration_count(result.kkt_factorizations,work.factorization_calls());
  add_native_restoration_count(result.lazy_newton.factor_attempts,work.factorization_calls());
  add_native_restoration_count(result.lazy_newton.backend_regularization_attempts,work.backend_regularization_attempts);
  add_native_restoration_count(result.lazy_newton.linear_solves,work.linear_system_calls());
  if(level)write_native_restoration_trace(std::cout,work,phase);
}

// Included beside Workspace and DcaInequalityModels in atomic_newton.cpp.
// Inexact inequality majorization with a primal-dual barrier filter.
// Legacy modes use one Newton direction. HH retains the certified model
// across a bounded inexact inner solve and rechecks it at the accepted alpha.
// The original certificate is the sole success criterion.
// The filter compares the ORIGINAL fixed-mu barrier problem, including g+s
// with the affine candidate slacks. Acceptance commits exactly this state;
// no majorant-gap transport modifies it afterwards. Reanchoring affects only
// the next Newton model. A separately enabled, one-shot last-chance centrality
// reset may reinitialize auxiliary state and filter AFTER otherwise terminal
// filter exhaustion. It preserves x, deadlines and the original KKT audit.
// Exact QCQP stationarity decomposition, in internally scaled coordinates.
// At the anchor r+ = (1-a)r + a*(linear+Cdx) - a*Cdx + a*a*cross.
// All products are accumulated independently from original/model coefficients.
struct DcaTailTerms {
  std::vector<long double> initial,linear,added,cross;
  struct NativeTerm {int index;long double x,d,coefficient,multiplier,delta;};
  std::vector<NativeTerm> native;
  explicit DcaTailTerms(const Workspace& w,const Direction& d)
      :initial(w.rd.begin(),w.rd.end()),linear(initial),added(w.n,0),cross(w.n,0) {
    auto product=[&](const std::vector<SymTriplet>& h,long double weight,
                     std::vector<long double>& out) {
      for(const auto& e:h) {
        out[e.i]+=weight*e.v*d.x[e.j];
        if(e.i!=e.j)out[e.j]+=weight*e.v*d.x[e.i];
      }
    };
    auto term=[&](const Quadratic& q,double mult,double delta) {
      product(q.hessian,static_cast<long double>(mult)*q.scale,linear);
      product(q.hessian,static_cast<long double>(delta)*q.scale,cross);
      for(auto e:q.signed_square)
        native.push_back({e.first,w.x[e.first],d.x[e.first],
            static_cast<long double>(e.second)*q.scale,mult,delta});
      for(std::size_t k=0;k<q.support.size();++k)
        linear[q.support[k]]+=static_cast<long double>(delta)*q.gradient_extended[k];
      if(q.model_active) {
        product(q.model_hessian,mult,added);
        if(q.study_model_scalar)for(int j=0;j<w.n;++j)added[j]+=(long double)mult*q.study_model_scalar*d.x[j];
        product(q.hessian,-static_cast<long double>(mult)*q.scale,added);
        for(const auto& a:q.model_atoms) {
          long double dot=0;for(int j=0;j<w.n;++j)dot+=static_cast<long double>(a.direction[j])*d.x[j];
          for(int j=0;j<w.n;++j)added[j]+=mult*a.beta*dot*a.direction[j];
        }
      }
    };
    term(w.objective,1,0);
    for(int e=0;e<w.me;++e)term(w.equalities[e],w.lambda[e],d.lambda[e]);
    for(int i=0;i<w.mi;++i)if(w.active[i])term(w.inequalities[i],w.z[i],d.z[i]);
  }
  // A native gradient is 2*c*abs(x), so a crossing adds an exact remainder
  // to the ordinary quadratic stationarity identity. This also covers x=0,
  // where the chosen generalized Hessian is zero. There is no remainder or
  // allocation on the original purely quadratic route.
  std::vector<long double> native_defect(double alpha) const {
    if(native.empty())return {};
    std::vector<long double> defect(initial.size(),0);
    for(const auto& e:native) {
      const long double step=static_cast<long double>(alpha)*e.d,y=e.x+step;
      const int sign=(e.x>0)-(e.x<0);
      const long double remainder=sign==0?std::abs(step):
          ((sign>0&&y>=0)||(sign<0&&y<=0)?0:-2*sign*y);
      defect[e.index]+=2*e.coefficient*(e.multiplier+alpha*e.delta)*remainder;
    }
    return defect;
  }
  void report(const Workspace& w,int iteration,double a,const Vec& forcing) const {
    const auto defect=native_defect(a);
    long double lin=0,add=0,nonlinear=0,identity=0,pred=0,old=0,injected=0;
    for(int j=0;j<w.n;++j) {
      const long double e=linear[j]+added[j];
      const long double native_remainder=defect.empty()?0:defect[j];
      const long double prediction=(1-a)*initial[j]+a*e-a*added[j]+a*a*cross[j]+native_remainder;
      const long double rhs=forcing.empty()?0:forcing[j];
      injected=std::max(injected,std::abs(a*rhs));
      lin=std::max(lin,std::abs(a*(e-rhs)));add=std::max(add,std::abs(a*added[j]));
      nonlinear=std::max(nonlinear,std::abs(a*a*cross[j]+native_remainder));
      old=std::max(old,std::abs((1-a)*initial[j]));pred=std::max(pred,std::abs(prediction));
      identity=std::max(identity,std::abs(prediction-w.rd[j]));
    }
    const double scale=w.objective.scale;
    std::cout<<"[DUAL-DECOMPOSITION] iter="<<iteration<<" alpha="<<a
      <<" linear_error="<<double(lin/scale)<<" injected_rhs="<<double(injected/scale)<<" curvature="<<double(add/scale)
      <<" nonlinear="<<double(nonlinear/scale)<<" damping="<<double(old/scale)
      <<" predicted="<<double(pred/scale)<<" observed="<<inf_norm(w.rd)/scale
      <<" identity_error="<<double(identity/scale)<<std::endl;
  }
};

class DcaMajorantBarrierFilter {
  std::vector<DcaMajorantBarrierPoint> entries;
  double theta_cap,theta_min;
  bool variable_barrier=false;
public:
  struct Decision {
    bool accepted=false,feasibility=false,objective=false;
    // Paper (19) defines the switching condition independently of theta_min.
    // theta_min selects the mandatory-Armijo acceptance branch only. Filter
    // augmentation depends on (19) AND Armijo (20), not on which of (18a/b)
    // supplied the local acceptance verdict. See Waechter--Biegler sec.2.3.
    bool switching=false,armijo=false,f_type=false;
    const char* reason="nonfinite";
    bool augment_history(bool fix)const {
      return accepted&&(fix?!f_type:(feasibility&&!objective));
    }
  };
  explicit DcaMajorantBarrierFilter(double initial_theta,bool retain_barrier_history=false)
      : theta_cap(10.0*std::max(1.0,initial_theta)),
        theta_min(1e-4*std::max(1.0,initial_theta)),variable_barrier(retain_barrier_history) {}
  void rebase(double mu) {
    if(!(mu>=0)||!std::isfinite(mu))throw std::runtime_error("invalid filter barrier rebase");
    for(auto& e:entries) {
      e.phi=static_cast<double>(static_cast<long double>(e.objective)-mu*e.slack_logs);
      if(!std::isfinite(e.phi))throw std::runtime_error("nonfinite filter barrier rebase");
    }
  }
  void reset() {entries.clear();}
  std::size_t size() const {return entries.size();}
  Decision diagnose(const DcaMajorantBarrierPoint& from,
                    const DcaMajorantBarrierPoint& to,
                    double slope,double alpha,double armijo) const {
    Decision d;
    if(!from.finite||!to.finite||!to.positive||!std::isfinite(slope)||
        !(alpha>0&&alpha<=1))return d;
    if(to.theta>theta_cap) {d.reason="theta-cap";return d;}
    constexpr double gamma_theta=1e-5,gamma_phi=1e-8;
    // Ipopt Compare_le allowance: only arithmetic roundoff of the local
    // comparison, never a relaxation of any requested KKT tolerance.
    const auto compare_le=[](double lhs,double rhs,double reference) {
      return lhs-rhs<=10*std::numeric_limits<double>::epsilon()*std::abs(reference);
    };
    // Standard filter switching condition; logarithms avoid overflow when
    // comparing alpha*(-slope)^2.3 with theta^1.1. Armijo uses the accurately
    // accumulated phi DIFFERENCE and the standard local rounding allowance.
    const bool switching=slope<0&&
        (from.theta==0||std::log(alpha)+2.3*std::log(-slope)>
            1.1*std::log(from.theta));
    d.switching=switching;
    d.armijo=compare_le(to.phi_change,armijo*alpha*slope,from.phi);
    d.f_type=switching&&d.armijo;
    if(from.theta<=theta_min&&switching) {
      d.objective=d.armijo;
    } else {
      d.feasibility=compare_le(to.theta,(1-gamma_theta)*from.theta,from.theta);
      d.objective=compare_le(to.phi_change,-gamma_phi*from.theta,from.phi);
    }
    if(!(d.feasibility||d.objective)) {d.reason="barrier-progress";return d;}
    for(const auto& e:entries)
      if(!(to.theta<=(1-gamma_theta)*e.theta||to.phi<=e.phi-gamma_phi*e.theta)) {
        d.reason="filter-history";return d;
      }
    d.accepted=true;d.reason="accepted";return d;
  }
  void insert(const DcaMajorantBarrierPoint& point) {
    entries.erase(std::remove_if(entries.begin(),entries.end(),[&](const auto& e) {
      // At a variable barrier, dominance at one mu need not persist at
      // another. Prune only when dominance holds for EVERY nonnegative mu.
      return point.theta<=e.theta&&(variable_barrier?
          (point.objective<=e.objective&&point.slack_logs>=e.slack_logs):point.phi<=e.phi);
    }),entries.end());
    entries.push_back(point);
  }
};

// A single reduction of the original cache supplies the outer stopping gate,
// the fixed-mu filter reference and the next outer barrier decision.
struct DcaOuterMetrics {
  double primal=0,dual=0,comp=0,kkt=0,base_residual=0;
  double dual_internal_inf=0;
  double equality_primal=0,inequality_primal=0;
  double equality_internal_inf=0,slack_equation_internal_inf=0;
  double barrier_primal_residual=0;
  double product_min=INFINITY,product_max=0,product_mean=0;
  double source_product_min=INFINITY,source_product_max=0;
  double bound_product_min=INFINITY,bound_product_max=0;
  int source_product_count=0,bound_product_count=0;
  // Source slack variables are primal variables of the stabilized equality
  // formulation. Their multipliers are free during iteration. Retain the
  // state, not an old-mu residual, because the barrier gate can be rebased.
  std::vector<double> source_eta,source_slack;
  double source_multiplier_sign_inf=0;
  long double slack_logs=0;
  int active_count=0;
  DcaMajorantBarrierPoint barrier;

  double source_stationarity_residual(double mu) const {
    if(!std::isfinite(mu)||mu<0||source_eta.size()!=source_slack.size())return INFINITY;
    long double value=0;
    for(std::size_t i=0;i<source_eta.size();++i) {
      if(!std::isfinite(source_eta[i])||!std::isfinite(source_slack[i])||!(source_slack[i]>0))return INFINITY;
      const long double r=static_cast<long double>(source_eta[i])-static_cast<long double>(mu)/source_slack[i];
      value=std::max(value,std::abs(r));
    }
    return std::isfinite(value)&&value<=std::numeric_limits<double>::max()?static_cast<double>(value):INFINITY;
  }
  double source_centering_residual(double mu) const {
    return source_product_count?std::max(std::abs(source_product_min-mu),std::abs(source_product_max-mu)):0;
  }
  double bound_centering_residual(double mu) const {
    return bound_product_count?std::max(std::abs(bound_product_min-mu),std::abs(bound_product_max-mu)):0;
  }
  double residual(double mu) const {
    const double original=active_count?std::max({base_residual,std::abs(product_min-mu),
        std::abs(product_max-mu)}):base_residual;
    // Keep every historical product guard as well as the true source-slack
    // gradient. A small product residual alone can conceal a large gradient.
    return std::max(original,source_stationarity_residual(mu));
  }
  // Barrier progress requires primal feasibility and centering only. The
  // original dual residual remains mandatory in the final KKT certificate.
  double primal_centering_residual(double mu) const {
    const double original=active_count?std::max({barrier_primal_residual,std::abs(product_min-mu),
        std::abs(product_max-mu)}):barrier_primal_residual;
    return std::max(original,source_stationarity_residual(mu));
  }
  DcaMajorantBarrierPoint filter_point(double mu) const {
    auto point=barrier;
    point.slack_logs=slack_logs;
    point.phi=double(static_cast<long double>(point.objective)-mu*slack_logs);
    point.finite=point.finite&&std::isfinite(point.phi);
    return point;
  }
};

DcaOuterMetrics dca_outer_metrics(const Workspace& w) {
  using Acc=long double;
  DcaOuterMetrics m;
  double dual_internal=0;
  for(double value:w.rd) {
    if(!std::isfinite(value)){dual_internal=INFINITY;break;}
    dual_internal=std::max(dual_internal,std::abs(value));
  }
  m.dual=dual_internal/w.objective.scale;
  m.dual_internal_inf=dual_internal;
  m.base_residual=dual_internal;
  m.barrier.objective=w.objective.value;
  Acc theta=0,product_sum=0;
  for(int e=0;e<w.me;++e) {
    const double value=std::abs(w.equalities[e].value);
    m.equality_internal_inf=std::max(m.equality_internal_inf,value);
    m.equality_primal=std::max(m.equality_primal,value/w.equalities[e].scale);
    m.primal=std::max(m.primal,value/w.equalities[e].scale);
    m.base_residual=std::max(m.base_residual,value);
    m.barrier_primal_residual=std::max(m.barrier_primal_residual,value);
    m.barrier.primal=std::max(m.barrier.primal,value/w.equalities[e].scale);
    m.barrier.primal_internal=std::max(m.barrier.primal_internal,value);
    m.barrier.finite=m.barrier.finite&&std::isfinite(value);
    theta+=value;
  }
  for(int i=0;i<w.mi;++i) {
    const auto& q=w.inequalities[i];
    const double violation=std::max(0.0,q.value);
    m.inequality_primal=std::max(m.inequality_primal,violation/q.scale);
    m.primal=std::max(m.primal,violation/q.scale);
    Acc residual=violation;
    m.barrier.finite=m.barrier.finite&&std::isfinite(q.value);
    if(w.active[i]) {
      const bool positive=w.s[i]>0&&std::isfinite(w.s[i])&&std::isfinite(w.z[i])&&
          (!w.requires_positive_dual(i)||w.z[i]>0);
      m.barrier.positive=m.barrier.positive&&positive;
      m.barrier.finite=m.barrier.finite&&positive;
      const double product=w.s[i]*w.z[i];
      m.barrier.finite=m.barrier.finite&&std::isfinite(product);
      m.product_min=std::min(m.product_min,product);
      m.product_max=std::max(m.product_max,product);
      if(w.inequality_kind[i]==0) {
        ++m.source_product_count;
        m.source_product_min=std::min(m.source_product_min,product);
        m.source_product_max=std::max(m.source_product_max,product);
      } else {
        ++m.bound_product_count;
        m.bound_product_min=std::min(m.bound_product_min,product);
        m.bound_product_max=std::max(m.bound_product_max,product);
      }
      product_sum+=Acc(w.s[i])*w.z[i];
      ++m.active_count;
      if(w.source_primal_barrier(i)) {
        m.source_eta.push_back(w.z[i]);m.source_slack.push_back(w.s[i]);
        if(w.z[i]<0)m.source_multiplier_sign_inf=std::max(m.source_multiplier_sign_inf,
            static_cast<double>(-static_cast<Acc>(w.z[i])*q.scale/w.objective.scale));
      }
      m.comp=std::max(m.comp,double(std::abs(Acc(w.z[i])*q.value)/w.objective.scale));
      residual=std::abs(Acc(q.value)+w.s[i]);
      if(positive)m.slack_logs+=std::log(w.s[i]);
      m.slack_equation_internal_inf=std::max(m.slack_equation_internal_inf,std::abs(w.rp[i]));
      m.base_residual=std::max(m.base_residual,std::abs(w.rp[i]));
      m.barrier_primal_residual=std::max(m.barrier_primal_residual,std::abs(w.rp[i]));
    }
    theta+=residual;
    m.barrier.primal=std::max(m.barrier.primal,double(residual/q.scale));
    m.barrier.primal_internal=std::max(m.barrier.primal_internal,double(residual));
  }
  // Equality-implied sides are omitted only from the numerical barrier.
  // Their ORIGINAL physical violations remain part of every filter metric.
  const auto omitted=w.omitted_original_bounds(w.x);
  if(omitted.sides||omitted.maximum!=0||omitted.sum!=0) {
    const double scale=w.omitted_original_bound_scale();
    const double internal=scale*omitted.maximum;
    m.primal=std::max(m.primal,omitted.maximum);
    m.base_residual=std::max(m.base_residual,internal);
    m.barrier_primal_residual=std::max(m.barrier_primal_residual,internal);
    m.barrier.primal=std::max(m.barrier.primal,omitted.maximum);
    m.barrier.primal_internal=std::max(m.barrier.primal_internal,internal);
    m.barrier.omitted_bound_max=omitted.maximum;
    m.barrier.omitted_bound_theta=static_cast<double>(scale*omitted.sum);
    m.barrier.omitted_bound_sides=omitted.sides;
    theta+=scale*omitted.sum;
  }
  m.product_mean=m.active_count?double(product_sum/m.active_count):0;
  // This is the ORIGINAL inequality sign diagnostic, not an artificial
  // positivity restriction on the current source-equality multiplier.
  m.dual=std::max(m.dual,m.source_multiplier_sign_inf);
  m.kkt=std::max({m.primal,m.dual,m.comp});
  m.barrier.theta=double(theta);
  m.barrier.finite=m.barrier.finite&&std::isfinite(m.kkt)&&
      std::isfinite(m.base_residual)&&std::isfinite(m.product_mean)&&
      std::isfinite(m.barrier.objective)&&std::isfinite(m.barrier.theta)&&
      std::isfinite(m.barrier.primal)&&std::isfinite(m.slack_logs);
  if(!m.barrier.finite)throw std::runtime_error("nonfinite original outer-iteration cache");
  return m;
}



// Exact minimizer of max_j |r0_j + alpha*(r1_j-r0_j)| on [0,1]
// in working arithmetic, via the upper envelope of affine functions.
// No iterative scalar search, Jacobian products, or function evaluations.
double dca_dual_minimax_step(const Vec& r0,const Vec& r1) {
  struct Line {long double slope,intercept,start;};
  std::vector<Line> lines,hull;lines.reserve(2*r0.size());
  for(std::size_t j=0;j<r0.size();++j) {
    const long double b=r0[j],m=static_cast<long double>(r1[j])-b;
    lines.push_back({m,b,0});lines.push_back({-m,-b,0});
  }
  std::sort(lines.begin(),lines.end(),[](const Line& a,const Line& b){
    return a.slope<b.slope||(a.slope==b.slope&&a.intercept>b.intercept);
  });
  for(auto line:lines) {
    if(!hull.empty()&&line.slope==hull.back().slope)continue;
    line.start=-INFINITY;
    while(!hull.empty()) {
      line.start=(hull.back().intercept-line.intercept)/(line.slope-hull.back().slope);
      if(line.start>hull.back().start)break;
      hull.pop_back();
    }
    if(hull.empty())line.start=-INFINITY;
    hull.push_back(line);
  }
  long double best=INFINITY,chosen=1;
  for(std::size_t k=0;k<hull.size();++k) {
    const auto& line=hull[k];
    const long double left=std::max(0.0L,line.start);
    const long double right=std::min(1.0L,k+1<hull.size()?hull[k+1].start:INFINITY);
    if(left>right)continue;
    const long double alpha=line.slope<0?right:left;
    const long double value=line.intercept+alpha*line.slope;
    if(value<best||(value==best&&alpha>chosen)){best=value;chosen=alpha;}
  }
  return double(chosen);
}

// A dual-only regularized least-squares correction on the ORIGINAL cache.
// Solve (B B^T + eps I) u = -B rd, B = diag(||J_e||)^-1 J;
// delta_lambda = diag(||J_e||)^-1 u. No Hessian, matrix assembly or factorization.
// Incremental mode requires >=10% stationarity improvement. Restart mode
// also allows unchanged stationarity with >=10% smaller multipliers, so an
// independent primal component cannot hide an equality multiplier repair.
// Both modes leave x,s,z,mu and the original filter unchanged.
bool dca_majorant_recalc_lambda(Workspace& w,DcaOuterMetrics& current,
    DcaFilterStatistics& stats,int budget,const Clock::time_point& start,double time_limit,
    bool interpolate=false,bool restart=false) {
  const auto began=Clock::now();
  ++stats.majorant_lambda_attempts;stats.majorant_lambda_last_alpha=1;
  Vec target=w.rd;
  if(restart) {
    std::vector<long double> sum(w.n,0);w.objective.add_gradient_extended(1,sum);
    for(int i=0;i<w.mi;++i)if(w.active[i])w.inequalities[i].add_gradient_extended(w.z[i],sum);
    for(int j=0;j<w.n;++j)target[j]=double(sum[j]);
  }
  Vec scale(w.me,0),u(w.me,0),r(w.me,0),p(w.me,0),ap(w.me,0),jt(w.n,0);
  auto row_dot=[](const Quadratic& q,const Vec& v) {
    long double sum=0;
    const double* g=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
    for(std::size_t k=0;k<q.support.size();++k)sum+=static_cast<long double>(g[k])*v[q.support[k]];
    return double(sum);
  };
  auto inner=[](const Vec& a,const Vec& b) {
    long double sum=0;for(std::size_t k=0;k<a.size();++k)sum+=static_cast<long double>(a[k])*b[k];
    return double(sum);
  };
  for(int e=0;e<w.me;++e) {
    const auto& q=w.equalities[e];const double* g=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
    long double squared=0;for(std::size_t k=0;k<q.support.size();++k)squared+=static_cast<long double>(g[k])*g[k];
    scale[e]=squared>0?1/std::sqrt(double(squared)):0;
    r[e]=-scale[e]*row_dot(q,target);
  }
  p=r;double rr=inner(r,r);const double initial_rr=rr;
  constexpr double ridge=1e-10;
  for(int it=0;it<budget&&std::isfinite(rr)&&rr>1e-4*initial_rr&&rr>0;++it) {
    if(seconds(start)>=time_limit)break;
    std::fill(jt.begin(),jt.end(),0);
    for(int e=0;e<w.me;++e)w.equalities[e].add_gradient(scale[e]*p[e],jt);
    for(int e=0;e<w.me;++e)ap[e]=scale[e]*row_dot(w.equalities[e],jt)+ridge*p[e];
    const double curvature=inner(p,ap);
    if(!(curvature>0)||!std::isfinite(curvature))break;
    const double alpha=rr/curvature;
    for(int e=0;e<w.me;++e){u[e]+=alpha*p[e];r[e]-=alpha*ap[e];}
    ++stats.majorant_lambda_iterations;
    const double next=inner(r,r),beta=next/rr;
    for(int e=0;e<w.me;++e)p[e]=r[e]+beta*p[e];
    rr=next;
  }
  const Vec old_lambda=w.lambda;
  bool accepted=false;
  if(finite(u)&&seconds(start)<time_limit) {
    Vec trial=restart?Vec(w.me,0):old_lambda;for(int e=0;e<w.me;++e)trial[e]+=scale[e]*u[e];
    // Large multipliers on almost dependent equality rows can survive a
    // short normal-equation CG restart. Resolve its small row Gram spectrum
    // Inspect BOTH the accepted vector and the proposed CG vector: a small
    // accepted vector can generate a huge new component on a vanishing row.
    // only after this numerical symptom; original stationarity below still
    // decides whether the minimum-norm replacement is retained.
    const long double rows=w.me,cols=w.n;
    const bool spectrum_budget=sizeof(double)*(rows*rows+rows*cols+8*rows)<=64.0L*1024*1024 &&
        rows*rows*(cols+rows)<=1.0e9L;
    if(restart && spectrum_budget && seconds(start)<time_limit &&
       std::max(inf_norm(old_lambda),inf_norm(trial))>1000*std::max(1.0,inf_norm(target))) {
      const int m=w.me;Vec gram(m*m,0),values(m),work(3*m),dense(w.n*m,0),rhs(m,0);
      for(int e=0;e<m;++e) {
        const auto& q=w.equalities[e];const double* g=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
        for(std::size_t k=0;k<q.support.size();++k)dense[q.support[k]+w.n*e]=g[k];
        rhs[e]=-row_dot(q,target);
      }
      for(int e=0;e<m;++e)for(int f=0;f<=e;++f) {
        long double dot=0;for(int j=0;j<w.n;++j)dot+=static_cast<long double>(dense[j+w.n*e])*dense[j+w.n*f];
        gram[e+m*f]=gram[f+m*e]=double(dot);
      }
      const char vectors='V',upper='U';const int lwork=3*m;int info=0;
      dsyev_(&vectors,&upper,&m,gram.data(),&m,values.data(),work.data(),&lwork,&info);
      if(!info && finite(values) && values.back()>0) {
        Vec fit(m,0);int rank=0;
        const double cutoff=64*std::numeric_limits<double>::epsilon()*m*values.back();
        for(int k=0;k<m;++k)if(values[k]>cutoff) {
          ++rank;long double dot=0;for(int e=0;e<m;++e)dot+=static_cast<long double>(gram[e+m*k])*rhs[e];
          for(int e=0;e<m;++e)fit[e]+=gram[e+m*k]*double(dot/values[k]);
        }
        for(int e=0;e<m;++e)trial[e]=fit[e];
        if(w.st.live_display_level>=2)std::cout<<"[LAMBDA-NUMERICAL-NULLSPACE] rows="<<m<<" rank="<<rank
          <<" old_norm="<<inf_norm(old_lambda)<<" trial_norm="<<inf_norm(trial)<<std::endl;
      }
    }
    if(finite(trial)) {
      std::vector<long double> sum(w.n,0);
      w.objective.add_gradient_extended(1,sum);
      for(int e=0;e<w.me;++e)w.equalities[e].add_gradient_extended(trial[e],sum);
      for(int i=0;i<w.mi;++i)if(w.active[i])w.inequalities[i].add_gradient_extended(w.z[i],sum);
      Vec trial_rd(w.n);for(int j=0;j<w.n;++j)trial_rd[j]=double(sum[j]);
      if(interpolate&&finite(trial_rd)) {
        const double alpha=dca_dual_minimax_step(w.rd,trial_rd);
        stats.majorant_lambda_last_alpha=alpha;
        if(alpha<1) {
          ++stats.majorant_lambda_interpolations;
          for(int e=0;e<w.me;++e)trial[e]=old_lambda[e]+alpha*(trial[e]-old_lambda[e]);
          std::fill(sum.begin(),sum.end(),0);
          w.objective.add_gradient_extended(1,sum);
          for(int e=0;e<w.me;++e)w.equalities[e].add_gradient_extended(trial[e],sum);
          for(int i=0;i<w.mi;++i)if(w.active[i])w.inequalities[i].add_gradient_extended(w.z[i],sum);
          for(int j=0;j<w.n;++j)trial_rd[j]=double(sum[j]);
        }
      }
      const double dual=finite(trial_rd)?inf_norm(trial_rd)/w.objective.scale:INFINITY;
      accepted=std::isfinite(dual)&&(dual<=0.9*current.dual || (restart && dual<=current.dual && inf_norm(trial)<0.9*inf_norm(old_lambda)));
      if(accepted){
        w.lambda=std::move(trial);w.rd=std::move(trial_rd);
        current=dca_outer_metrics(w);++stats.majorant_lambda_acceptances;
      }
    }
  }
  stats.majorant_lambda_seconds+=seconds(began);
  return accepted;
}

// HH retains one certified convex barrier model across several primal-dual
// Newton steps. The original filter remains an independent OUTER acceptance
// gate. No relative inner test is inferred from a linear-system residual.
struct HybridInnerAnchor {
  Vec x,s,z,lambda;
  explicit HybridInnerAnchor(const Workspace& w):x(w.x),s(w.s),z(w.z),lambda(w.lambda){}
  void restore(Workspace& w) const {w.x=x;w.s=s;w.z=z;w.lambda=lambda;w.evaluate();}
};
// Realize an entirely sub-ulp audited correction by the adjacent floating
// vector in its solved direction. Callers must still check the complete
// fixed model, original filter, domain and displacement bound.
bool hybrid_adjacent_realization(Vec& point,const Vec& before,const Vec& correction) {
  if(point!=before)return false;
  bool changed=false;
  for(std::size_t j=0;j<point.size();++j)if(correction[j]!=0) {
    point[j]=std::nextafter(before[j],correction[j]>0?INFINITY:-INFINITY);
    changed=changed||point[j]!=before[j];
  }
  return changed;
}
struct HybridInnerOutcome {
  hybrid_inner::Residual initial,final;
  hybrid_inner::Gate gate;
  double anchor_merit=0,model_merit=0,step_inf=0,scaled_mu=0,relative=0,absolute=0;
  double tau_prox=0,proximal_energy_scaled=0,proximal_step_squared=0,objective_scale=1;
  double objective_base_lower=0;
  std::vector<int> proximal_coordinates;
  hybrid_inner::ModelGate model_gate;
  double barrier_mu=0,step_l2=0,normal_residual_l2=INFINITY,model_feasibility=INFINITY,
      normal_complementarity=INFINITY,activity_tolerance=0,minimum_multiplier=0,
      model_value=0,anchor_model_value=0,barrier_complementarity=0,penalty_sigma=0,activity_floor=0,
      point_inf=0,anchor_model_feasibility=0;
  int active_inequalities=0,inactive_inequalities=0,normal_sweeps=0,reduced_dimension=0;
  int normal_inequality_columns=0,equality_count=0,inequality_count=0;
  bool epsilon_normal_certificate=false;
  double epsilon_equality_gap=0;
  int slack_remaps=0,affine_repair_attempts=0,affine_repair_acceptances=0;
  double affine_repair_seconds=0;
  bool model_relative=false,qcqp_extension=false;
  Vec normal_gradient,normal_multipliers;
  Vec model_point,model_anchor;
  std::vector<SymTriplet> model_projection;
  std::vector<hybrid_inner::NormalColumn> normal_columns;
  std::vector<double> normal_values;
  std::vector<int> normal_kinds,normal_indices;
  int steps=0;
  bool success=false;
  // Populated only for an independently certified terminal endpoint of a
  // failed private solve. It never authorizes an ordinary outer step.
  std::shared_ptr<HybridInnerAnchor> terminal_original_state;
  detail::HybridTerminalAudit terminal_original_audit;
  std::string reason="not-started";
  std::string numerical_detail;
};
void hybrid_terminal_audit_trace(const HybridInnerOutcome& o,const Settings& st,
    int outer,const char* phase,const detail::HybridTerminalAudit& audit,bool committed=false,
    double commit_elapsed=INFINITY) {
  auto number=[](double value){std::ostringstream out;out<<std::setprecision(17);if(std::isfinite(value))out<<value;else out<<"null";return out.str();};
  std::cout<<"[HYBRID-TERMINAL-AUDIT] {\"outer\":"<<outer<<",\"configuration\":\""<<st.convexification_study
    <<"\",\"phase\":\""<<phase<<"\",\"attempted\":"<<(audit.attempted?"true":"false")
    <<",\"strict_original\":"<<(audit.strict?"true":"false")<<",\"expired\":"<<(audit.expired?"true":"false")
    <<",\"primal\":"<<number(audit.primal)<<",\"dual\":"<<number(audit.dual)
    <<",\"complementarity\":"<<number(audit.complementarity)
    <<",\"primal_tolerance\":"<<number(st.feasibility_tolerance)
    <<",\"dual_tolerance\":"<<number(std::min(st.stationarity_tolerance,st.dual_infeasibility_tolerance))
    <<",\"complementarity_tolerance\":"<<number(st.complementarity_tolerance)
    <<",\"elapsed_before\":"<<number(audit.elapsed_before)<<",\"elapsed_after\":"<<number(audit.elapsed_after)
    <<",\"seconds\":"<<number(audit.seconds)<<",\"time_limit\":"<<number(st.time_limit_seconds)
    <<",\"commit_elapsed\":"<<number(commit_elapsed)
    <<",\"committed\":"<<(committed?"true":"false")<<",\"inner_success\":"<<(o.success?"true":"false")
    <<",\"filter_accepted\":false,\"scope\":\"strict-original-QCQP-terminal-certificate\"}"<<std::endl;
}
// Zero multipliers are a legitimate ORIGINAL KKT certificate when the
// objective gradient vanishes. Barrier multipliers need not vanish at a
// feasible restoration endpoint. This is a terminal certificate proposal,
// never permission to accept an infeasible point or a nonzero gradient.
bool hybrid_try_zero_gradient_certificate(Workspace& w,const Settings& st,
    Clock::time_point start) {
  if(inf_norm(w.objective.gradient)!=0||seconds(start)>=st.time_limit_seconds)return false;
  const Vec saved_lambda=w.lambda,saved_z=w.z;
  std::fill(w.lambda.begin(),w.lambda.end(),0);std::fill(w.z.begin(),w.z.end(),0);
  w.evaluate();Result candidate;w.export_state(candidate);
  const auto audit=detail::audit_hybrid_terminal_original(w.original,st,candidate,
      [&](){return seconds(start);},&w.original_linear_support);
  if(st.live_display_level)std::cout<<"[ZERO-GRADIENT-ORIGINAL-KKT] accepted="<<audit.strict
      <<" primal="<<audit.primal<<" dual="<<audit.dual<<" complementarity="<<audit.complementarity
      <<" point_unchanged=1 all_original_rows_audited=1 final_tolerances_unchanged=1"<<std::endl;
  if(audit.strict&&detail::hybrid_terminal_commit_open(audit,seconds(start),st.time_limit_seconds))return true;
  w.lambda=saved_lambda;w.z=saved_z;w.evaluate();return false;
}
void hybrid_try_terminal_original(Workspace& w,const Settings& st,
    Clock::time_point start,int outer,HybridInnerOutcome& o) {
  if(o.success)return;
  CostScope certificate(w.costs,NewtonCost::Certificate);
  Result endpoint;w.export_state(endpoint);endpoint.outer_iterations=outer;
  o.terminal_original_audit=detail::audit_hybrid_terminal_original(w.original,st,endpoint,
      [&](){return seconds(start);},&w.original_linear_support);
  if(!o.terminal_original_audit.strict&&hybrid_try_zero_gradient_certificate(w,st,start)) {
    w.export_state(endpoint);
    o.terminal_original_audit=detail::audit_hybrid_terminal_original(w.original,st,endpoint,
        [&](){return seconds(start);},&w.original_linear_support);
  }
  if(o.terminal_original_audit.strict)
    o.terminal_original_state=std::make_shared<HybridInnerAnchor>(w);
  hybrid_terminal_audit_trace(o,st,outer,"private-endpoint",o.terminal_original_audit);
}
// A terminal iteration budget does not identify which mathematical gate
// failed. Keep the terminal reason and report the independently evaluated
// components; this changes neither tolerances nor inner/outer acceptance.
std::string hybrid_inner_failure_details(const HybridInnerOutcome& o) {
  std::ostringstream text;text<<std::setprecision(17)<<o.reason<<"; inner_steps="<<o.steps;
  if(!o.numerical_detail.empty())text<<"; numerical_cause="<<o.numerical_detail;
  if(o.model_relative) {
    const auto& g=o.model_gate;
    text<<"; positive_margin="<<g.positive_margin
        <<"; stationarity="<<o.normal_residual_l2<<"/"<<g.stationarity_target
        <<"; stationarity_ok="<<g.stationarity_ok
        <<"; feasibility="<<o.model_feasibility<<"/"<<g.feasibility_target
        <<"; feasibility_ok="<<g.feasibility_ok
        <<"; complementarity="<<std::max(o.normal_complementarity,o.barrier_complementarity)
        <<"/"<<g.complementarity_target
        <<"; complementarity_ok="<<g.complementarity_ok
        <<"; model_decrease="<<g.model_decrease;
  } else {
    text<<"; dual="<<o.final.dual<<"/"<<o.gate.dual_target
        <<"; primal="<<o.final.primal<<"/"<<o.gate.primal_target
        <<"; centering="<<o.final.centering<<"/"<<o.gate.centering_target
        <<"; relative_ok="<<o.gate.relative_ok
        <<"; model_decrease="<<o.gate.model_decrease;
  }
  return text.str();
}
Vec hybrid_model_dual(const Workspace& w,const HybridInnerAnchor& anchor,
    const HybridInnerOutcome& model) {
  Vec dual=w.rd;
  hybrid_inner::retained_proximal(w.x,anchor.x,model.proximal_coordinates,model.tau_prox,&dual);
  return dual;
}
double hybrid_model_merit(const Workspace& w,const HybridInnerAnchor& anchor,
    double mu,HybridInnerOutcome& model) {
  long double projected_square=0;
  for(int j:model.proximal_coordinates) {
    const long double d=static_cast<long double>(w.x[j])-anchor.x[j];
    projected_square+=d*d;
  }
  model.proximal_step_squared=static_cast<double>(projected_square);
  model.proximal_energy_scaled=hybrid_inner::retained_proximal(w.x,anchor.x,
      model.proximal_coordinates,model.tau_prox)/w.objective.scale;
  return w.merit(w.x,w.s,mu)/w.objective.scale+model.proximal_energy_scaled;
}
hybrid_inner::Residual hybrid_model_residual(const Workspace& w,double mu,
    const HybridInnerAnchor& anchor,const HybridInnerOutcome& model) {
  hybrid_inner::Residual r;
  r.dual=inf_norm(hybrid_model_dual(w,anchor,model))/w.objective.scale;
  for(int e=0;e<w.me;++e)r.primal=std::max(r.primal,std::abs(w.h[e])/w.equalities[e].scale);
  for(int i=0;i<w.mi;++i)if(w.active[i]) {
    if(!(w.s[i]>0&&w.z[i]>0))return {INFINITY,INFINITY,INFINITY};
    r.primal=std::max(r.primal,std::abs(w.rp[i])/w.inequalities[i].scale);
    r.centering=std::max(r.centering,std::abs(w.s[i]*w.z[i]-mu)/w.objective.scale);
  }
  return r;
}
double hybrid_model_progress_norm(const Workspace& w,double mu,
    const HybridInnerAnchor& anchor,const HybridInnerOutcome& model) {
  hybrid_inner::FullResidualL2 sum;
  for(double v:hybrid_model_dual(w,anchor,model))sum.add(v,w.objective.scale);
  for(int e=0;e<w.me;++e)sum.add(w.h[e],w.equalities[e].scale);
  for(int i=0;i<w.mi;++i)if(w.active[i]) {
    if(!(w.s[i]>0&&w.z[i]>0))return INFINITY;
    sum.add(w.rp[i],w.inequalities[i].scale);
    sum.add(static_cast<long double>(w.s[i])*w.z[i]-mu,w.objective.scale);
  }
  return sum.norm();
}
double hybrid_displacement(const Vec& x,const Vec& anchor) {
  double n=0;for(std::size_t j=0;j<x.size();++j)n=std::max(n,std::abs(x[j]-anchor[j]));return n;
}
Vec hybrid_reduced(const Workspace& w,const Vec& ambient) {
  Vec reduced(w.nr,0);
  for(int j=0;j<w.n;++j)for(int t=w.tbegin[j];t<w.tbegin[j+1];++t)
    reduced[w.tcolumn[t]]+=w.tvalue[t]*ambient[j];
  return reduced;
}
hybrid_inner::NormalColumn hybrid_normal_column(const Workspace& w,const Quadratic& q,
    double multiplier,bool nonnegative) {
  hybrid_inner::NormalColumn column;column.multiplier=multiplier;column.nonnegative=nonnegative;
  std::map<int,long double> values;
  const int* support=q.pooled_support?q.pooled_support:q.support.data();
  const double* gradient=q.pooled_gradient?q.pooled_gradient:q.gradient.data();
  for(std::size_t k=0;k<q.support.size();++k) {
    const int j=support[k];
    for(int t=w.tbegin[j];t<w.tbegin[j+1];++t)
      values[w.tcolumn[t]]+=static_cast<long double>(w.tvalue[t])*gradient[k]/q.scale;
  }
  for(auto a:values)if(a.second!=0)column.values.push_back({a.first,static_cast<double>(a.second)});
  return column;
}
void hybrid_model_relative_gate(Workspace& w,const HybridInnerAnchor& anchor,
    const Settings& st,HybridInnerOutcome& o,bool update_penalty=true) {
  struct PenaltyTransaction {
    double& value;double saved;bool commit=false;
    ~PenaltyTransaction(){if(!commit)value=saved;}
  } penalty_transaction{w.penalty,w.penalty};
  o.model_relative=true;o.reduced_dimension=w.nr;
  hybrid_model_merit(w,anchor,0,o);o.step_l2=std::sqrt(o.proximal_step_squared);
  o.objective_base_lower=w.hybrid_objective_base_lower;
  const double margin=study::certified_objective_margin_sum(0,
      w.hybrid_objective_margin(o.tau_prox),w.objective.scale);
  o.point_inf=inf_norm(w.x);
  const double roundoff=64*std::numeric_limits<double>::epsilon()*(1+o.point_inf);
  o.activity_floor=roundoff;
  o.activity_tolerance=std::max(roundoff,std::min(st.study_inner_activity,
      st.study_inner_relative*o.proximal_step_squared));
  const double feasibility_target=std::max(st.study_inner_feasibility,
      std::min(st.feasibility_tolerance,st.study_inner_relative*o.proximal_step_squared));
  const double complementarity_target=std::max(st.study_inner_complementarity,
      std::min(st.complementarity_tolerance,st.study_inner_relative*margin*o.proximal_step_squared));
  Vec ambient(w.n,0);w.objective.add_gradient(1,ambient);
  hybrid_inner::retained_proximal(w.x,anchor.x,o.proximal_coordinates,o.tau_prox,&ambient);
  for(double& a:ambient)a/=w.objective.scale;
  const Vec gradient=hybrid_reduced(w,ambient);
  std::vector<hybrid_inner::NormalColumn> columns;
  std::vector<hybrid_inner::NormalColumn> all_columns;
  std::vector<double> values,scales;std::vector<int> kinds,indices;
  std::vector<double> all_values,all_scales;std::vector<int> all_kinds,all_indices;
  o.equality_count=w.me;o.inequality_count=w.mi;
  o.model_feasibility=0;o.active_inequalities=0;o.inactive_inequalities=0;
  o.anchor_model_feasibility=0;
  auto append=[&](const Quadratic& q,double multiplier,bool nonnegative,int kind,int index) {
    all_columns.push_back(hybrid_normal_column(w,q,multiplier*q.scale/w.objective.scale,nonnegative));
    all_values.push_back(q.value/q.scale);all_scales.push_back(q.scale);all_kinds.push_back(kind);all_indices.push_back(index);
    if(!nonnegative||q.value/q.scale>=-o.activity_tolerance) {
      columns.push_back(all_columns.back());values.push_back(all_values.back());
      scales.push_back(q.scale);kinds.push_back(kind);indices.push_back(index);
    }
  };
  for(int e=0;e<w.me;++e) {
    o.model_feasibility=std::max(o.model_feasibility,std::abs(w.h[e])/w.equalities[e].scale);
    o.anchor_model_feasibility=std::max(o.anchor_model_feasibility,std::abs(w.equalities[e].at(anchor.x))/w.equalities[e].scale);
    append(w.equalities[e],w.lambda[e],false,0,e);
  }
  for(int i=0;i<w.mi;++i) {
    const double value=w.inequalities[i].value/w.inequalities[i].scale;
    o.anchor_model_feasibility=std::max(o.anchor_model_feasibility,std::max(0.0,w.inequalities[i].at(anchor.x)/w.inequalities[i].scale));
    o.model_feasibility=std::max(o.model_feasibility,std::max(0.0,value));
    append(w.inequalities[i],w.active[i]?w.z[i]:0.0,true,1,i);
    if(value>=-o.activity_tolerance) {
      ++o.active_inequalities;
    } else ++o.inactive_inequalities;
  }
  o.barrier_complementarity=0;
  for(int i=0;i<w.mi;++i)if(w.active[i])
    o.barrier_complementarity+=std::abs(w.z[i]*w.inequalities[i].value)/w.objective.scale;
  const double original_penalty=w.penalty;
  auto evaluate_certificate=[&](const hybrid_inner::NormalFit& fit,
      const std::vector<hybrid_inner::NormalColumn>& normals,
      const std::vector<double>& normal_values,const std::vector<double>& normal_scales) {
    o.normal_residual_l2=fit.residual_l2;o.normal_sweeps=fit.sweeps;
    o.normal_complementarity=0;o.minimum_multiplier=0;o.normal_inequality_columns=0;
    double multiplier_max=std::max(inf_norm(w.lambda),inf_norm(w.z));
    long double gap=0,equality_gap=0;
    for(std::size_t k=0;k<normals.size();++k) {
      multiplier_max=std::max(multiplier_max,std::abs(fit.multipliers[k])*w.objective.scale/normal_scales[k]);
      if(normals[k].nonnegative) {
        gap+=std::abs(static_cast<long double>(fit.multipliers[k])*normal_values[k]);
        o.minimum_multiplier=std::min(o.minimum_multiplier,fit.multipliers[k]);
        ++o.normal_inequality_columns;
      } else equality_gap+=std::abs(static_cast<long double>(fit.multipliers[k])*normal_values[k]);
    }
    o.normal_complementarity=static_cast<double>(gap);
    o.epsilon_equality_gap=static_cast<double>(equality_gap);
    // Compare endpoint and anchor with one common exact-L1 penalty. An
    // unsuccessful alternative certificate cannot leave an inflated penalty.
    const double certificate_penalty=hybrid_inner::model_penalty_seed(multiplier_max);
    w.penalty=update_penalty?std::max(original_penalty,certificate_penalty):original_penalty;
    o.penalty_sigma=w.penalty;
    o.model_value=hybrid_model_merit(w,anchor,0,o);
    o.anchor_model_value=w.merit(anchor.x,anchor.s,0)/w.objective.scale;
    o.model_gate=hybrid_inner::check_model(o.normal_residual_l2,o.step_l2,margin,
        st.study_inner_kappa_fraction,st.study_inner_absolute,o.model_feasibility,
        feasibility_target,std::max(o.normal_complementarity,o.barrier_complementarity),complementarity_target,
        o.model_value,o.anchor_model_value);
  };
  auto fit=hybrid_inner::NormalFit{};
  const auto epsilon=hybrid_inner::epsilon_normal(gradient,all_columns,all_values);
  o.epsilon_normal_certificate=false;
  if(epsilon.dual_feasible) {
    evaluate_certificate(epsilon.fit,all_columns,all_values,all_scales);
    if(o.model_gate.passed) {
      o.epsilon_normal_certificate=true;fit=epsilon.fit;
      columns=std::move(all_columns);values=std::move(all_values);scales=std::move(all_scales);
      kinds=std::move(all_kinds);indices=std::move(all_indices);
    }
  }
  if(!o.epsilon_normal_certificate) {
    fit=hybrid_inner::fit_normal(gradient,columns);
    evaluate_certificate(fit,columns,values,scales);
  }
  // A rejected normal fit is a diagnostic, not an accepted multiplier update.
  // Its trial penalty was used consistently for BOTH merit values above, but
  // must not poison the next Newton step or another certificate alternative.
  if(!o.model_gate.passed)w.penalty=original_penalty;
  else penalty_transaction.commit=true;
  if(w.n<=16) {
    o.normal_gradient=gradient;o.normal_columns=columns;o.normal_multipliers=fit.multipliers;
    o.normal_values=values;o.normal_kinds=kinds;o.normal_indices=indices;
    o.model_point=w.x;o.model_anchor=anchor.x;o.model_projection.clear();
    for(int j=0;j<w.n;++j)for(int t=w.tbegin[j];t<w.tbegin[j+1];++t)
      o.model_projection.push_back({j,w.tcolumn[t],w.tvalue[t]});
  }
}
void hybrid_model_relative_trace(const HybridInnerOutcome& o,const Settings& st,
    const char* phase,int outer,double alpha,bool accepted) {
  auto number=[](double v){return std::isfinite(v)?v:std::numeric_limits<double>::max();};
  const auto& g=o.model_gate;
  std::cout<<std::setprecision(17)<<"[HYBRID-MODEL-RELATIVE] {\"phase\":\""<<phase
    <<"\",\"outer\":"<<outer<<",\"steps\":"<<o.steps<<",\"barrier_mu\":"<<o.barrier_mu
    <<",\"actual_alpha\":"<<alpha<<",\"tau_prox\":"<<o.tau_prox<<",\"objective_scale\":"<<o.objective_scale
    <<",\"point_inf\":"<<o.point_inf<<",\"proximal_step_squared\":"<<o.proximal_step_squared
    <<",\"proximal_energy_scaled\":"<<o.proximal_energy_scaled
    <<",\"projection\":\"retained_coordinates\",\"reduced_dimension\":"<<o.reduced_dimension
    <<",\"margin_source\":\"certified-objective-base-plus-retained-tau\""
    <<",\"objective_base_certified_lower\":"<<o.objective_base_lower
    <<",\"margin_lower\":"<<g.margin
    <<",\"kappa_fraction\":"<<st.study_inner_kappa_fraction<<",\"kappa\":"<<g.kappa
    <<",\"step_l2\":"<<o.step_l2<<",\"residual_l2\":"<<number(o.normal_residual_l2)
    <<",\"stationarity_floor\":"<<st.study_inner_absolute<<",\"stationarity_target\":"<<g.stationarity_target
    <<",\"feasibility\":"<<number(o.model_feasibility)<<",\"feasibility_target\":"<<g.feasibility_target
    <<",\"feasibility_floor\":"<<st.study_inner_feasibility<<",\"feasibility_ceiling\":"<<st.feasibility_tolerance
    <<",\"complementarity_sum\":"<<number(o.normal_complementarity)
    <<",\"barrier_complementarity_sum\":"<<number(o.barrier_complementarity)
    <<",\"complementarity_target\":"<<g.complementarity_target<<",\"complementarity_floor\":"<<st.study_inner_complementarity
    <<",\"complementarity_ceiling\":"<<st.complementarity_tolerance<<",\"relative_coefficient\":"<<st.study_inner_relative
    <<",\"activity_tolerance\":"<<o.activity_tolerance<<",\"activity_ceiling\":"<<st.study_inner_activity
    <<",\"activity_floor\":"<<o.activity_floor
    <<",\"min_multiplier\":"<<o.minimum_multiplier<<",\"active_inequalities\":"<<o.active_inequalities
    <<",\"inactive_inequalities\":"<<o.inactive_inequalities<<",\"ls_sweeps\":"<<o.normal_sweeps
    <<",\"normal_certificate_kind\":\""<<(o.epsilon_normal_certificate?"epsilon-kkt":"finite-activity-fit")<<"\""
    <<",\"normal_inequality_columns\":"<<o.normal_inequality_columns
    <<",\"equality_count\":"<<o.equality_count<<",\"inequality_count\":"<<o.inequality_count
    <<",\"epsilon_budget\":"<<g.complementarity_target
    <<",\"epsilon_realized\":"<<number(o.normal_complementarity)
    <<",\"epsilon_equality_gap\":"<<number(o.epsilon_equality_gap)
    <<",\"dual_nonnegative\":"<<(o.minimum_multiplier>=0?"true":"false")
    <<",\"model_value\":"<<number(o.model_value)<<",\"anchor_model_value\":"<<number(o.anchor_model_value)
    <<",\"penalty_sigma\":"<<o.penalty_sigma
    <<",\"penalty_sigma_role\":\"certificate-trial-common-anchor-and-endpoint\""
    <<",\"affine_repair_attempts\":"<<o.affine_repair_attempts
    <<",\"affine_repair_acceptances\":"<<o.affine_repair_acceptances
    <<",\"affine_repair_seconds\":"<<o.affine_repair_seconds
    <<",\"penalty_fit_committed\":"<<(o.model_gate.passed?"true":"false")
    <<",\"anchor_model_feasibility\":"<<o.anchor_model_feasibility
    <<",\"anchor_role\":\""<<(o.anchor_model_feasibility<=st.feasibility_tolerance?"numerically-feasible-model-anchor":"infeasible-QCQP-linearization")<<"\""
    <<",\"model_value_kind\":\"zero-barrier-exact-L1-plus-proximal\""
    <<",\"model_decrease\":"<<(g.model_decrease?"true":"false")
    <<",\"positive_margin\":"<<(g.positive_margin?"true":"false")
    <<",\"stationarity_ok\":"<<(g.stationarity_ok?"true":"false")
    <<",\"feasibility_ok\":"<<(g.feasibility_ok?"true":"false")
    <<",\"complementarity_ok\":"<<(g.complementarity_ok?"true":"false")
    <<",\"inner_step_budget\":"<<st.study_inner_max
    <<",\"criterion_passed\":"<<(g.passed?"true":"false")<<",\"accepted\":"<<(accepted?"true":"false")
    <<",\"residual_kind\":\""<<(o.epsilon_normal_certificate?"epsilon-model-KKT":"finite-activity-model-KKT")
    <<"\",\"scope\":\"QCQP-approximate-model-KKT-extension\",\"reason\":\""<<o.reason<<"\"";
  if(!o.normal_gradient.empty()) {
    auto array=[&](const char* name,const Vec& values) {
      std::cout<<",\""<<name<<"\":[";for(std::size_t j=0;j<values.size();++j){if(j)std::cout<<',';std::cout<<number(values[j]);}std::cout<<']';
    };
    array("point",o.model_point);array("anchor",o.model_anchor);
    std::cout<<",\"T\":[";
    for(std::size_t k=0;k<o.model_projection.size();++k){if(k)std::cout<<',';const auto a=o.model_projection[k];std::cout<<'['<<a.i<<','<<a.j<<','<<a.v<<']';}
    std::cout<<"],\"reduced_gradient\":[";
    for(std::size_t j=0;j<o.normal_gradient.size();++j){if(j)std::cout<<',';std::cout<<number(o.normal_gradient[j]);}
    std::cout<<"],\"normal_columns\":[";
    for(std::size_t k=0;k<o.normal_columns.size();++k) {
      if(k)std::cout<<',';
      std::cout<<"{\"kind\":"<<o.normal_kinds[k]<<",\"index\":"<<o.normal_indices[k]
        <<",\"value\":"<<number(o.normal_values[k])<<",\"multiplier\":"<<number(o.normal_multipliers[k])<<",\"gradient\":[";
      Vec dense(o.reduced_dimension,0);for(auto a:o.normal_columns[k].values)dense[a.first]=a.second;
      for(std::size_t j=0;j<dense.size();++j){if(j)std::cout<<',';std::cout<<number(dense[j]);}
      std::cout<<"]}";
    }
    std::cout<<']';
  }
  std::cout<<"}"<<std::endl;
}
void hybrid_inner_trace(const HybridInnerOutcome& o,const char* phase,int outer,
    double mu,double alpha,bool accepted) {
  auto number=[](double v){return std::isfinite(v)?v:std::numeric_limits<double>::max();};
  std::cout<<std::setprecision(17)<<"[HYBRID-INEXACT] {\"phase\":\""<<phase
    <<"\",\"outer\":"<<outer<<",\"steps\":"<<o.steps<<",\"barrier_mu\":"<<mu
    <<",\"scaled_mu\":"<<number(o.scaled_mu)<<",\"inner_relative\":"<<o.relative<<",\"inner_absolute\":"<<o.absolute
    <<",\"tau_prox\":"<<number(o.tau_prox)<<",\"proximal_dimension\":"<<o.proximal_coordinates.size()
    <<",\"proximal_projection\":\"retained_coordinates\",\"proximal_energy_scaled\":"<<number(o.proximal_energy_scaled)
    <<",\"proximal_step_squared\":"<<number(o.proximal_step_squared)<<",\"objective_scale\":"<<number(o.objective_scale)
    <<",\"initial_primal\":"<<number(o.initial.primal)<<",\"initial_centering\":"<<number(o.initial.centering)
    <<",\"centering_reference\":"<<number(std::max(o.initial.centering,o.scaled_mu))<<",\"alpha\":"<<alpha<<",\"dual\":"<<number(o.final.dual)
    <<",\"dual_target\":"<<number(o.gate.dual_target)
    <<",\"primal\":"<<number(o.final.primal)<<",\"primal_target\":"<<number(o.gate.primal_target)
    <<",\"centering\":"<<number(o.final.centering)<<",\"centering_target\":"<<number(o.gate.centering_target)
    <<",\"step_inf\":"<<number(o.step_inf)<<",\"model_merit\":"<<number(o.model_merit)
    <<",\"anchor_model_merit\":"<<number(o.anchor_merit)
    <<",\"model_decrease\":"<<(o.gate.model_decrease?"true":"false")
    <<",\"relative_ok\":"<<(o.gate.relative_ok?"true":"false")
    <<",\"criterion_passed\":"<<(o.gate.passed?"true":"false")
    <<",\"accepted\":"<<(accepted?"true":"false")<<",\"reason\":\""<<o.reason
    <<"\",\"scope\":\"fixed-model-fixed-mu-proximal-barrier\"}"<<std::endl;
}
HybridInnerOutcome hybrid_inner_solve_barrier(Workspace& w,Direction& complete,double mu,
    const Settings& st,Clock::time_point start,int outer) {
  HybridInnerAnchor anchor(w);HybridInnerOutcome o;
  // The first direction was factored at this anchor. Freeze its algorithmic
  // proximal diagonal as part of the convex subproblem, including its gradient
  // and energy in every subsequent residual/RHS/acceptance test. Backend
  // numerical shifts remain audited preconditioners and are not included.
  o.tau_prox=w.regularization;o.proximal_coordinates=w.retained;o.objective_scale=w.objective.scale;
  o.initial=hybrid_model_residual(w,mu,anchor,o);o.scaled_mu=mu/w.objective.scale;
  o.relative=st.study_inner_relative;o.absolute=st.study_inner_absolute;
  o.anchor_merit=hybrid_model_merit(w,anchor,mu,o);
  o.model_merit=o.anchor_merit;o.final=o.initial;
  Direction step=complete;
  try {
    for(int inner=0;inner<st.study_inner_max;++inner) {
      if(seconds(start)>=st.time_limit_seconds){o.reason="time-limit";break;}
      if(inner) {
        w.update_matrix();
        if(!w.factorize(o.tau_prox)){o.reason="inner-factorization";break;}
        if(w.regularization!=o.tau_prox||w.retained!=o.proximal_coordinates) {
          o.reason="inner-frozen-proximal-changed";break;
        }
        Vec rc(w.mi,0);for(int i=0;i<w.mi;++i)if(w.active[i])rc[i]=w.centered_complementarity_rhs(i,mu);
        step=w.direction(rc,hybrid_model_dual(w,anchor,o),w.h,w.rp);
        if(w.regularization!=o.tau_prox||w.retained!=o.proximal_coordinates) {
          o.reason="inner-frozen-proximal-changed";break;
        }
      }
      const HybridInnerAnchor previous(w);
      const auto before=hybrid_model_residual(w,mu,anchor,o);
      double a=w.fraction(step,st.lazy_newton_fraction_to_boundary);bool moved=false;
      for(int ls=0;ls<st.line_search_max_steps&&seconds(start)<st.time_limit_seconds;++ls,a*=st.line_search_contraction) {
        w.x=previous.x;w.s=previous.s;w.z=previous.z;w.lambda=previous.lambda;
        axpy(a,step.x,w.x);axpy(a,step.s,w.s);axpy(a,step.z,w.z);axpy(a,step.lambda,w.lambda);
        try {
          w.evaluate();const auto trial=hybrid_model_residual(w,mu,anchor,o);
          if(trial.finite()&&trial.norm()<=(1-1e-4*a)*before.norm()) {moved=true;break;}
        }catch(const std::runtime_error&){}
      }
      if(!moved){previous.restore(w);o.reason="inner-residual-line-search";break;}
      ++o.steps;o.final=hybrid_model_residual(w,mu,anchor,o);
      o.step_inf=hybrid_displacement(w.x,anchor.x);
      o.model_merit=hybrid_model_merit(w,anchor,mu,o);
      o.gate=hybrid_inner::check(o.final,o.initial,o.step_inf,st.study_inner_relative,
          st.study_inner_absolute,o.model_merit,o.anchor_merit,o.scaled_mu);
      if(o.steps>=st.study_inner_min&&o.gate.passed) {o.success=true;o.reason="relative-barrier-model-criterion";break;}
      o.reason="inner-iteration-budget";
    }
    if(o.success) {
      complete.x=w.x;complete.s=w.s;complete.z=w.z;complete.lambda=w.lambda;
      for(int j=0;j<w.n;++j)complete.x[j]-=anchor.x[j];
      for(int i=0;i<w.mi;++i){complete.s[i]-=anchor.s[i];complete.z[i]-=anchor.z[i];}
      for(int e=0;e<w.me;++e)complete.lambda[e]-=anchor.lambda[e];
    }
    if(!o.success)hybrid_try_terminal_original(w,st,start,outer,o);
    anchor.restore(w);w.update_matrix();
  }catch(const std::runtime_error&) {
    o.success=false;o.reason="inner-numerical-error";anchor.restore(w);w.update_matrix();
  }
  hybrid_inner_trace(o,"inner",outer,mu,1,false);
  return o;
}
HybridInnerOutcome hybrid_inner_solve_relative(Workspace& w,Direction& complete,double mu,
    const Settings& st,Clock::time_point start,int outer) {
  HybridInnerAnchor anchor(w);HybridInnerOutcome o;
  const double inherited_penalty=w.penalty;
  const bool first_model_startup=!w.hybrid_initial_model_startup_consumed;
  const bool initial_outer_eligible=outer==1;
  // Consume even an ineligible first model: this is initialization, not an
  // indefinitely deferred event that may recenter multipliers near a solution.
  w.hybrid_initial_model_startup_consumed=true;
  w.penalty=hybrid_inner::model_penalty_seed(std::max(inf_norm(w.lambda),inf_norm(w.z)));
  o.model_relative=true;o.tau_prox=w.regularization;o.proximal_coordinates=w.retained;
  o.objective_scale=w.objective.scale;o.barrier_mu=mu;
  for(const auto& q:w.equalities)o.qcqp_extension=o.qcqp_extension||!q.hessian.empty();
  for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0)
    o.qcqp_extension=o.qcqp_extension||!w.inequalities[i].hessian.empty();
  std::vector<int> active;for(int i=0;i<w.mi;++i)if(w.active[i])active.push_back(i);
  int centrality_restarts=0;
  bool affine_repair_attempted=false;
  try {
    const auto remap_start=Clock::now();double slack_shift=0;
    // A strictly interior model anchor needs no artificial slack floor.
    // Project privately to its exact positive inequality slacks, retaining
    // the accepted primal point AND all multipliers. Commit this private
    // initialization only when the complete scaled KKT residual decreases.
    // The sole first-model initialization may also center positive duals.
    // Later near-active multipliers must never be reset by mean_mu/(-g).
    bool strict_interior=!active.empty();
    for(int i:active)strict_interior=strict_interior&&w.inequalities[i].value<0;
    int exact_slacks=0;double exact_before=0,exact_after=0;
    bool initial_dual_centered=false;double initial_target_mu=0,z_shift=0;
    long double mean_mu_before=0,mean_mu_after=0;
    for(int i:active)mean_mu_before+=static_cast<long double>(w.s[i])*w.z[i];
    if(!active.empty())mean_mu_before/=active.size();
    if(strict_interior) {
      const Vec saved_slacks=w.s;
      exact_before=hybrid_model_progress_norm(w,o.barrier_mu,anchor,o);
      if(first_model_startup&&initial_outer_eligible) {
        Vec values(w.mi);for(int i=0;i<w.mi;++i)values[i]=w.inequalities[i].value;
        const auto initial=hybrid_inner::initial_interior_start(values,w.s,w.z,active);
        if(initial.eligible) {
          w.s=initial.slack;w.z=initial.dual;
          o.barrier_mu=initial.target_mu;initial_target_mu=initial.target_mu;
          initial_dual_centered=true;
        }
      }
      if(!initial_dual_centered)for(int i:active)w.s[i]=-w.inequalities[i].value;
      w.evaluate();exact_after=hybrid_model_progress_norm(w,o.barrier_mu,anchor,o);
      // Startup is a private initialization and may increase its initial dual
      // residual. It accepts no outer point; all subsequent model/outer gates
      // are unchanged. Warm-start slack proposals still require contraction.
      if(initial_dual_centered||(std::isfinite(exact_after)&&exact_after<exact_before)) {
        for(int i:active)if(w.s[i]!=saved_slacks[i]) {
          slack_shift=std::max(slack_shift,std::abs(w.s[i]-saved_slacks[i]));
          ++exact_slacks;++o.slack_remaps;
        }
      } else {w.s=saved_slacks;w.evaluate();}
    }
    for(int i:active) {
      // A new majorant is tangent at the ORIGINAL accepted point. The old
      // majorant slack can be tiny while its new physical slack is large.
      // Transport only that positive gap, preserving outer x, z and metrics.
      const double remapped=std::max(w.s[i],-w.inequalities[i].value);
      if(remapped>w.s[i]){slack_shift=std::max(slack_shift,remapped-w.s[i]);w.s[i]=remapped;++o.slack_remaps;}
    }
    if(o.slack_remaps)w.evaluate();
    for(int i:active)mean_mu_after+=static_cast<long double>(w.s[i])*w.z[i];
    if(!active.empty())mean_mu_after/=active.size();
    for(int i:active)z_shift=std::max(z_shift,std::abs(w.z[i]-anchor.z[i]));
    std::cout<<std::setprecision(17)<<"[HYBRID-SUBMODEL-START] {\"outer\":"<<outer
      <<",\"policy\":\"initial-centered-or-residual-controlled-interior-slacks\",\"changed_slacks\":"<<o.slack_remaps
      <<",\"first_model_startup\":"<<(first_model_startup?"true":"false")
      <<",\"initial_outer_eligible\":"<<(initial_outer_eligible?"true":"false")
      <<",\"initial_startup_consumed\":true,\"initial_dual_centered\":"<<(initial_dual_centered?"true":"false")
      <<",\"initial_target_mu\":"<<initial_target_mu<<",\"outer_point_accepted\":false"
      <<",\"exact_interior_slacks\":"<<exact_slacks<<",\"exact_start_residual_before\":"<<exact_before
      <<",\"exact_start_residual_after\":"<<exact_after
      <<",\"mean_mu_before\":"<<static_cast<double>(mean_mu_before)<<",\"mean_mu_after\":"<<static_cast<double>(mean_mu_after)
      <<",\"lambda_inf_before\":"<<inf_norm(anchor.lambda)<<",\"lambda_inf_after\":"<<inf_norm(w.lambda)
      <<",\"z_inf_before\":"<<inf_norm(anchor.z)<<",\"z_inf_after\":"<<inf_norm(w.z)
      <<",\"slack_shift_inf\":"<<slack_shift<<",\"primal_x_shift\":0,\"dual_z_shift\":"<<z_shift<<",\"dual_lambda_shift\":0,\"tau_prox\":"<<o.tau_prox
      <<",\"inherited_penalty_sigma\":"<<inherited_penalty<<",\"model_penalty_sigma\":"<<w.penalty
      <<",\"penalty_policy\":\"per-fixed-model-dual-seed-transactional-fits\""
      <<",\"seconds\":"<<seconds(remap_start)<<",\"rollback_restores_accepted_state\":true";
    if(w.n<=16&&w.mi<=32) {
      auto array=[&](const char* name,const Vec& values) {
        std::cout<<",\""<<name<<"\":[";for(std::size_t j=0;j<values.size();++j){if(j)std::cout<<',';std::cout<<values[j];}std::cout<<']';
      };
      array("x_before",anchor.x);array("x_after",w.x);array("z_before",anchor.z);array("z_after",w.z);
      array("s_before",anchor.s);array("s_after",w.s);
      Vec g(w.mi),mask(w.mi);for(int i=0;i<w.mi;++i){g[i]=w.inequalities[i].value;mask[i]=w.active[i];}
      array("new_model_values",g);array("active_mask",mask);
    }
    std::cout<<"}"<<std::endl;
    auto central_restart=[&](bool numerical_failure,int inner,const char* trigger) {
      if(centrality_restarts!=0||active.empty()||seconds(start)>=st.time_limit_seconds)return false;
      double primal=0,mean=0;
      for(int e=0;e<w.me;++e)
        primal=std::max(primal,std::abs(w.h[e])/w.equalities[e].scale);
      for(int i:active) {
        primal=std::max(primal,std::abs(w.rp[i])/w.inequalities[i].scale);
        mean+=w.s[i]*w.z[i];
      }
      mean/=active.size();
      const auto reset=hybrid_inner::central_restart(primal,w.objective.scale,
          st.study_inner_feasibility,mean,numerical_failure);
      if(!reset.needed)return false;
      const auto before_reset=HybridInnerAnchor(w);
      anchor.restore(w);
      for(int i:active) {
        const double floor=w.inequalities[i].scale*reset.normalized_slack_floor;
        w.s[i]=std::max(floor,-w.inequalities[i].value);
        w.z[i]=reset.target_mu/w.s[i];
      }
      if(!finite(w.s)||!finite(w.z)) {before_reset.restore(w);return false;}
      w.evaluate();o.barrier_mu=reset.target_mu;++centrality_restarts;
      std::cout<<std::setprecision(17)<<"[HYBRID-CENTRAL-RESTART] {\"outer\":"<<outer
        <<",\"inner\":"<<inner+1<<",\"prior_mean_mu\":"<<mean
        <<",\"model_primal_residual\":"<<primal<<",\"target_mu\":"<<reset.target_mu
        <<",\"trigger\":\""<<trigger<<"\",\"restores_anchor_x\":true,\"outer_point_accepted\":false"
        <<",\"model_and_proximal_unchanged\":true,\"remaining_inner_budget\":"<<st.study_inner_max-inner-1
        <<"}"<<std::endl;
      return true;
    };
    if(!(w.hybrid_objective_margin(o.tau_prox)>0))
      o.reason="uncertified-positive-reduced-model-margin";
    else for(int inner=0;inner<st.study_inner_max;++inner) {
      try {
      if(seconds(start)>=st.time_limit_seconds){o.reason="time-limit";break;}
      hybrid_model_relative_gate(w,anchor,st,o);
      // A rounded affine projection can fail exact-L1 descent while every
      // model KKT component passes. Repair that same fixed model once; do not
      // modify its curvature, tau, tolerances, penalty policy or outer filter.
      if(!affine_repair_attempted && o.steps>=st.study_inner_min &&
          o.model_gate.positive_margin && o.model_gate.stationarity_ok &&
          o.model_gate.feasibility_ok && o.model_gate.complementarity_ok &&
          !o.model_gate.model_decrease && w.me>0 && inner+1<st.study_inner_max &&
          seconds(start)<st.time_limit_seconds) {
        affine_repair_attempted=true;
        const auto repair_start=Clock::now();const HybridInnerAnchor before(w);
        const auto before_outcome=o;const double before_penalty=w.penalty;
        auto* before_factor=w.current_factor;
        auto before_factor_attempts=w.stats.factor_attempts;
        const double before_tau=w.regularization;const auto before_retained=w.retained;
        bool repaired=false,directed_rounding=false,repair_point_factorized=false;
        double correction_inf=0;std::string repair_numerical_detail;
        const char* repair_reason="audited-correction-rejected";
        try {
          // The preceding private step updated x/s/z/lambda and gradients,
          // while numeric factors (and possibly assembly caches) still belong
          // to the previous Newton point. Match them to THIS repair endpoint,
          // keeping the same function models and proximal diagonal exactly.
          w.update_matrix();
          if(!w.factorize_affine_repair(before_tau))
            throw std::runtime_error("affine repair point factorization rejected: "+w.message);
          repair_point_factorized=true;
          before_factor=w.current_factor;before_factor_attempts=w.stats.factor_attempts;
          const Workspace::AffineRepairFactorPermit repair_factor(w);
          const Direction correction=w.direction(Vec(w.mi,0),Vec(w.n,0),w.h,Vec(w.mi,0),
              0,true,DirectionPurpose::Complete,true,&repair_factor);
          correction_inf=inf_norm(correction.x);
          const double step_bound=hybrid_displacement(before.x,anchor.x);
          if(w.current_factor==before_factor&&w.stats.factor_attempts==before_factor_attempts&&
              w.regularization==before_tau&&w.retained==before_retained&&
              correction_inf<=step_bound && w.fraction(correction,st.lazy_newton_fraction_to_boundary)>=1) {
            axpy(1,correction.x,w.x);axpy(1,correction.s,w.s);
            axpy(1,correction.z,w.z);axpy(1,correction.lambda,w.lambda);
            directed_rounding=hybrid_adjacent_realization(w.x,before.x,correction.x);
            repair_reason="nonrepresentable-or-size";
            if(w.x!=before.x&&finite(w.x)&&hybrid_displacement(w.x,before.x)<=step_bound) {
              w.evaluate();hybrid_model_relative_gate(w,anchor,st,o);
              repaired=o.model_gate.passed&&o.model_value<=o.anchor_model_value&&
                  seconds(start)<st.time_limit_seconds;
              repair_reason=repaired?"certified-model-repair":"model-merit-or-KKT-gate";
            }
          } else repair_reason="frozen-factor-or-size";
        } catch(const std::runtime_error& error) {repair_numerical_detail=error.what();}
        const double repair_seconds=seconds(repair_start);
        if(!repaired) {before.restore(w);w.penalty=before_penalty;o=before_outcome;}
        else ++o.steps;
        ++o.affine_repair_attempts;o.affine_repair_acceptances+=repaired;
        o.affine_repair_seconds+=repair_seconds;
        std::cout<<std::setprecision(17)<<"[HYBRID-AFFINE-REPAIR] {\"outer\":"<<outer
          <<",\"accepted\":"<<(repaired?"true":"false")
          <<",\"directed_rounding\":"<<(directed_rounding?"true":"false")
          <<",\"correction_inf\":"<<correction_inf<<",\"seconds\":"<<repair_seconds
          <<",\"factor_reused\":true,\"repair_point_factorized\":"<<(repair_point_factorized?"true":"false")
          <<",\"model_and_tau_unchanged\":true"
          <<",\"reason\":\""<<repair_reason<<"\",\"numerical_detail\":"<<std::quoted(repair_numerical_detail)
          <<",\"outer_filter_still_required\":true}"<<std::endl;
        // One repair consumes one of the common fixed-model iteration slots.
        ++inner;
      }
      if(o.steps>=st.study_inner_min&&o.model_gate.passed) {
        o.success=true;o.reason="finite-accuracy-model-relative-criterion";break;
      }
      if(inner||o.slack_remaps||initial_dual_centered) {
        w.update_matrix();
        if(!w.factorize(o.tau_prox)) {
          if(central_restart(true,inner,"factorization"))continue;
          o.reason="inner-factorization";o.numerical_detail=w.message;break;
        }
      }
      if(w.regularization!=o.tau_prox||w.retained!=o.proximal_coordinates) {
        o.reason="inner-frozen-proximal-changed";break;
      }
      const Vec dual=hybrid_model_dual(w,anchor,o);
      Vec rc(w.mi,0);Direction affine,step;
      hybrid_inner::MehrotraTarget prediction;
      double affine_alpha_primal=1,affine_alpha_dual=1,affine_audit=0;
      int predictor_solves=0,corrector_solves=0,centering_fallback=0;
      double mean_mu=0;for(int i:active)mean_mu+=w.s[i]*w.z[i];
      if(!active.empty())mean_mu/=active.size();
      const double previous_mu=o.barrier_mu;
      const auto floor_plan=hybrid_inner::complementarity_barrier_floor(
          w.objective.scale,st.study_inner_complementarity,static_cast<int>(active.size()));
      const double barrier_floor=floor_plan.value;
      if(st.study_inner_predictor_corrector&&!active.empty()) {
        for(int i:active)rc[i]=w.s[i]*w.z[i];
        affine=w.direction(rc,dual,w.h,w.rp);++predictor_solves;affine_audit=affine.residual;
        affine_alpha_primal=w.fraction(affine,1,1);affine_alpha_dual=w.fraction(affine,1,2);
        prediction=hybrid_inner::mehrotra_target(w.s,w.z,affine.s,affine.z,active,
            affine_alpha_primal,affine_alpha_dual,o.barrier_mu,st.study_inner_barrier_decay,barrier_floor);
        o.barrier_mu=prediction.target;
        for(int i:active)rc[i]=w.centered_complementarity_rhs(i,o.barrier_mu)+affine.s[i]*affine.z[i];
        step=w.direction(rc,dual,w.h,w.rp);++corrector_solves;
      } else {
        o.barrier_mu=active.empty()?0:std::max(std::numeric_limits<double>::min(),
            std::max(barrier_floor,st.study_inner_barrier_decay*mean_mu));
        for(int i:active)rc[i]=w.centered_complementarity_rhs(i,o.barrier_mu);
        step=w.direction(rc,dual,w.h,w.rp);++corrector_solves;
      }
      if(w.regularization!=o.tau_prox||w.retained!=o.proximal_coordinates) {
        o.reason="inner-frozen-proximal-changed";break;
      }
      const HybridInnerAnchor previous(w);HybridInnerAnchor best_state(previous);
      Direction best_step=step;double best_norm=INFINITY,best_alpha=0;
      double best_primal_fraction=0,best_dual_fraction=0;
      const Direction predictor_step=step;Direction centering_step;bool centering_cached=false;
      const double iteration_penalty=w.penalty;double best_penalty=w.penalty;
      bool moved=false,selected_separate_steps=false;double accepted_alpha=0;
      bool merit_found=false,merit_separate=false;HybridInnerAnchor merit_state(previous);
      Direction merit_step=step;double merit_norm=INFINITY,merit_primal=INFINITY,merit_alpha=0,merit_penalty=iteration_penalty;
      double merit_primal_fraction=0,merit_dual_fraction=0;
      int progress_guard_rejections=0;bool last_state_changed=false;
      double before_norm=0,after_norm=0;std::string acceptance="none";
      double primal_fraction=0,dual_fraction=0;
      for(int attempt=0;attempt<(st.study_inner_separate_steps?4:2)&&seconds(start)<st.time_limit_seconds;++attempt) {
        // Preserve every successful common-step PC/centering transition.
        // Independent fractions are a fallback only after BOTH common-step
        // trials fail their existing full KKT contraction guard.
        if(attempt>=2) {
          if(moved||merit_found)break;
          previous.restore(w);w.penalty=iteration_penalty;
          if(attempt==2)step=predictor_step;
          else {if(!centering_cached)break;step=centering_step;}
        }
        if(attempt==1) {
          if(!predictor_solves)break;
          previous.restore(w);
          for(int i:active)rc[i]=w.centered_complementarity_rhs(i,o.barrier_mu);
          try {
            step=w.direction(rc,hybrid_model_dual(w,anchor,o),w.h,w.rp);
            if(w.regularization!=o.tau_prox||w.retained!=o.proximal_coordinates)
              throw std::runtime_error("inner-frozen-proximal-changed");
          }
          catch(const std::runtime_error&) {
            // Comparing an extra centering candidate must not discard an
            // already audited, residual-contracting predictor/corrector step.
            if(moved&&w.regularization==o.tau_prox&&w.retained==o.proximal_coordinates)break;
            throw;
          }
          ++corrector_solves;centering_fallback=1;centering_step=step;centering_cached=true;
        }
        w.penalty=std::max(w.penalty,hybrid_inner::model_penalty_seed(
            std::max(inf_norm(w.lambda),inf_norm(w.z))));
        double smooth_slope=w.objective.directional(step.x);
        for(int j:o.proximal_coordinates)smooth_slope+=o.tau_prox*(w.x[j]-anchor.x[j])*step.x[j];
        for(int i:active)smooth_slope-=o.barrier_mu*step.s[i]/w.s[i];
        const double theta=w.theta(w.x,w.s);
        w.penalty=hybrid_inner::descent_penalty(w.penalty,smooth_slope,theta);
        // hybrid_model_merit is normalized by objective.scale. Its Armijo
        // derivative must use those same units; penalty selection above is
        // deliberately left in the internal unnormalized model units.
        const double slope=(smooth_slope-w.penalty*theta)/w.objective.scale;
        const double before_merit=hybrid_model_merit(w,anchor,o.barrier_mu,o);
        before_norm=hybrid_model_progress_norm(w,o.barrier_mu,anchor,o);
        const double before_primal=hybrid_model_residual(w,o.barrier_mu,anchor,o).primal;
        primal_fraction=w.fraction(step,st.lazy_newton_fraction_to_boundary,1);
        dual_fraction=w.fraction(step,st.lazy_newton_fraction_to_boundary,2);
        if(attempt<2)primal_fraction=dual_fraction=std::min(primal_fraction,dual_fraction);
        double a=1;
        for(int ls=0;ls<st.line_search_max_steps&&seconds(start)<st.time_limit_seconds;++ls,a*=st.line_search_contraction) {
          w.x=previous.x;w.s=previous.s;w.z=previous.z;w.lambda=previous.lambda;
          const auto fractions=hybrid_split_fractions(primal_fraction,dual_fraction,a);
          axpy(fractions.first,step.x,w.x);axpy(fractions.first,step.s,w.s);
          axpy(fractions.second,step.z,w.z);axpy(fractions.second,step.lambda,w.lambda);
          try {
            w.evaluate();const auto residual=hybrid_model_residual(w,o.barrier_mu,anchor,o);
            const double merit=hybrid_model_merit(w,anchor,o.barrier_mu,o);
            after_norm=hybrid_model_progress_norm(w,o.barrier_mu,anchor,o);
            const double rounding=64*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(before_merit));
            last_state_changed=w.x!=previous.x||w.s!=previous.s||
                w.z!=previous.z||w.lambda!=previous.lambda;
            const auto progress=hybrid_inner::check_progress(last_state_changed&&
                finite(w.x)&&finite(w.s)&&finite(w.z)&&finite(w.lambda),before_norm,
                after_norm,before_merit,merit,primal_fraction*slope,a,st.line_search_armijo,rounding,
                std::min(primal_fraction,dual_fraction));
            if(residual.finite()&&progress.residual_contraction) {
              if(!moved||after_norm<best_norm) {
                best_state=HybridInnerAnchor(w);best_step=step;
                best_norm=after_norm;best_alpha=a;selected_separate_steps=attempt>=2;best_penalty=w.penalty;
                best_primal_fraction=primal_fraction;best_dual_fraction=dual_fraction;
              }
              moved=true;acceptance="scaled-KKT-l2-contraction";break;
            }
            if(st.study_inner_merit_recovery&&residual.finite()&&
                hybrid_merit_recovery_allowed(progress,before_merit,merit,rounding,before_primal,
                    residual.primal,st.study_inner_feasibility,before_norm,after_norm,fractions.first)&&
                (!merit_found||residual.primal<merit_primal||
                    (residual.primal==merit_primal&&after_norm<merit_norm))) {
              merit_found=true;merit_state=HybridInnerAnchor(w);merit_step=step;
              merit_norm=after_norm;merit_primal=residual.primal;merit_alpha=a;
              merit_penalty=w.penalty;merit_separate=attempt>=2;
              merit_primal_fraction=primal_fraction;merit_dual_fraction=dual_fraction;
            }
            if(residual.finite()&&!progress.residual_contraction)++progress_guard_rejections;
          }catch(const std::runtime_error&){}
        }
      }
      if(!moved&&merit_found) {
        moved=true;best_state=merit_state;best_step=merit_step;best_penalty=merit_penalty;
        best_norm=merit_norm;best_alpha=merit_alpha;selected_separate_steps=merit_separate;
        best_primal_fraction=merit_primal_fraction;best_dual_fraction=merit_dual_fraction;
        acceptance="guarded-Armijo-physical-primal-recovery";
      }
      if(moved) {
        best_state.restore(w);w.penalty=best_penalty;step=best_step;
        after_norm=best_norm;accepted_alpha=best_alpha;
        primal_fraction=best_primal_fraction;dual_fraction=best_dual_fraction;
        last_state_changed=true;
      }
      std::cout<<std::setprecision(17)<<"[HYBRID-NEWTON] {\"outer\":"<<outer<<",\"inner\":"<<inner+1
        <<",\"mu_current\":"<<mean_mu<<",\"previous_target_mu\":"<<previous_mu
        <<",\"mu_affine\":"<<prediction.affine_mu<<",\"sigma\":"<<prediction.sigma
        <<",\"sigma_used\":"<<prediction.sigma_used<<",\"barrier_floor\":"<<barrier_floor
        <<",\"barrier_floor_coefficient\":0.25,\"barrier_floor_scope\":\"private-model-complementarity-budget\""
        <<",\"barrier_floor_underflow\":"<<(floor_plan.underflow?"true":"false")
        <<",\"barrier_floor_saturated\":"<<(floor_plan.saturated?"true":"false")
        <<",\"complementarity_floor\":"<<st.study_inner_complementarity
        <<",\"objective_scale\":"<<w.objective.scale<<",\"active_count\":"<<active.size()
        <<",\"sigma_exponent\":3,\"sigma_min\":0,\"sigma_max\":1,\"barrier_decay\":"<<st.study_inner_barrier_decay
        <<",\"target_mu\":"<<o.barrier_mu<<",\"predictor_solves\":"<<predictor_solves
        <<",\"corrector_solves\":"<<corrector_solves<<",\"centering_fallback\":"<<centering_fallback
        <<",\"separate_step_recovery_enabled\":"<<(st.study_inner_separate_steps?"true":"false")
        <<",\"selected_separate_steps\":"<<(selected_separate_steps?"true":"false")
        <<",\"merit_recovery_enabled\":"<<(st.study_inner_merit_recovery?"true":"false")
        <<",\"primal_fraction\":"<<primal_fraction<<",\"dual_fraction\":"<<dual_fraction
        <<",\"affine_alpha_primal\":"<<affine_alpha_primal<<",\"affine_alpha_dual\":"<<affine_alpha_dual
        <<",\"affine_direction_audit\":"<<affine_audit<<",\"direction_audit\":"<<step.residual
        <<",\"audit_threshold\":"<<st.kkt_backward_error_reject<<",\"alpha\":"<<accepted_alpha
        <<",\"effective_primal_alpha\":"<<accepted_alpha*primal_fraction
        <<",\"effective_dual_alpha\":"<<accepted_alpha*dual_fraction
        <<",\"residual_effective_alpha\":"<<accepted_alpha*std::min(primal_fraction,dual_fraction)
        <<",\"residual_before\":"<<before_norm<<",\"residual_after\":"<<after_norm
        <<",\"penalty_sigma\":"<<w.penalty<<",\"accepted\":"<<(moved?"true":"false")
        <<",\"progress_guard_rejections\":"<<progress_guard_rejections
        <<",\"last_trial_state_changed\":"<<(last_state_changed?"true":"false")
        <<",\"progress_policy\":\"KKT-contraction-first-guarded-Armijo-after-common-failure\""
        <<",\"residual_norm\":\"all-scaled-KKT-components-l2\""
        <<",\"direction_selection\":\"best-certified-PC-or-centering\""
        <<",\"acceptance\":\""<<acceptance<<"\"}"<<std::endl;
      if(!moved) {
        previous.restore(w);w.penalty=iteration_penalty;
        // A new frozen tangent model can inherit a nearly zero barrier from
        // the preceding model while its affine equations are still infeasible.
        // Give that fixed model one private central restart. This changes no
        // outer point, model coefficient, proximal term or acceptance target.
        if(central_restart(false,inner,"line-search"))continue;
        o.reason=progress_guard_rejections?"inner-no-certified-progress":"inner-Mehrotra-line-search";break;
      }
      ++o.steps;o.reason="inner-iteration-budget";
      hybrid_model_relative_gate(w,anchor,st,o);
      if(o.steps>=st.study_inner_min&&o.model_gate.passed) {
        o.success=true;o.reason="finite-accuracy-model-relative-criterion";break;
      }
      } catch(const std::runtime_error& error) {
        if(central_restart(true,inner,"numerical-direction"))continue;
        o.success=false;o.reason="inner-model-relative-numerical-error";
        o.numerical_detail=error.what();break;
      }
    }
    if(o.success) {
      complete.x=w.x;complete.s=w.s;complete.z=w.z;complete.lambda=w.lambda;
      for(int j=0;j<w.n;++j)complete.x[j]-=anchor.x[j];
      for(int i=0;i<w.mi;++i){complete.s[i]-=anchor.s[i];complete.z[i]-=anchor.z[i];}
      for(int e=0;e<w.me;++e)complete.lambda[e]-=anchor.lambda[e];
    }
    if(!o.success)hybrid_try_terminal_original(w,st,start,outer,o);
    anchor.restore(w);w.update_matrix();
  }catch(const std::runtime_error& error) {
    o.success=false;o.reason="inner-model-relative-numerical-error";o.numerical_detail=error.what();anchor.restore(w);w.update_matrix();
  }
  if(!o.success)w.penalty=inherited_penalty;
  hybrid_model_relative_trace(o,st,"inner",outer,1,false);return o;
}
// Explicit diagnostic alternative: SAME frozen certified majorant, affine
// equalities and retained-coordinate proximal term, solved by the native
// convex QCQP IPM. No original nonconvex Newton direction or solver fallback.
HybridInnerOutcome hybrid_inner_solve_native(Workspace& w,Direction& complete,double mu,
    const Settings& st,Clock::time_point start,int outer) {
  HybridInnerAnchor anchor(w);HybridInnerOutcome o;
  const bool reduce=st.study_inner_rule=="native-reduced-relative"||st.study_inner_rule=="native-reduced-direction-relative"||st.study_inner_rule=="native-reduced-direction-soc";
  o.model_relative=true;o.tau_prox=w.regularization;o.proximal_coordinates=w.retained;
  o.objective_scale=w.objective.scale;o.barrier_mu=mu;
  const long double n=w.n,m=w.mi+w.me;
  if(sizeof(double)*(6*n*m+8*n*n+64*n+16*m)>512.L*1024*1024) {
    o.reason="native-convex-model-memory-admission";return o;
  }
  const double inherited_penalty=w.penalty;
  w.penalty=hybrid_inner::model_penalty_seed(std::max(inf_norm(w.lambda),inf_norm(w.z)));
  const double model_penalty_seed=w.penalty;
  try {
    ConvexSubproblem p;p.nvar=w.n;p.original_dimension=w.n;p.equalities.nvar=w.n;
    auto model=[&](const Quadratic& q){
      ConvexQuadraticModel a;a.constant=q.value;a.linear=Vec(w.n,0);q.add_gradient(1,a.linear);
      a.base_hessian.n=w.n;a.base_hessian.entries=q.model_active?q.model_hessian:q.hessian;
      if(!q.model_active)for(auto& e:a.base_hessian.entries)e.v*=q.scale;
      if(q.model_active){a.low_rank=q.model_atoms;
        // Sparse atom support is a semantic restriction, not just storage.
        if(!q.model_atom_support.empty())for(auto& atom:a.low_rank){
          Vec support(w.n,0);for(int j:q.model_atom_support)support[j]=atom.direction[j];atom.direction=std::move(support);}
        if(q.study_model_scalar)for(int j=0;j<w.n;++j)a.base_hessian.entries.push_back({j,j,q.study_model_scalar});
      }
      return a;
    };
    p.objective=model(w.objective);
    for(int j:w.retained)p.objective.base_hessian.entries.push_back({j,j,o.tau_prox});
    for(const auto& q:w.equalities){Vec g(w.n,0);q.add_gradient(1,g);SparseRow row;
      for(int j=0;j<w.n;++j)if(g[j]!=0)row.entries.push_back({j,g[j]});
      p.equalities.rows.push_back(std::move(row));p.equalities.rhs.push_back(-q.value);}
    std::vector<int> active;for(int i=0;i<w.mi;++i)if(w.active[i]){active.push_back(i);p.inequalities.push_back(model(w.inequalities[i]));}
    ConvexAffineCoordinates coordinates;
    std::vector<int> retained_equalities;
    if(reduce) {
      w.update_geometry();coordinates.ambient=w.n;coordinates.reduced=w.nr;
      coordinates.offset.assign(w.n,0);coordinates.rows.resize(w.n);
      for(int j=0;j<w.n;++j){
        if(w.pivot_equation[j]>=0){const int e=w.pivot_equation[j];coordinates.offset[j]=-w.h[e]/w.pivot_coefficient[e];}
        for(int k=w.tbegin[j];k<w.tbegin[j+1];++k)coordinates.rows[j].entries.push_back({w.tcolumn[k],w.tvalue[k]});
      }
      ConvexSubproblem reduced;reduced.nvar=w.nr;reduced.original_dimension=w.nr;reduced.equalities.nvar=w.nr;
      reduced.objective=convex_affine_pullback(p.objective,coordinates);
      for(int e=0;e<w.me;++e)if(w.pivot_variable[e]<0){
        retained_equalities.push_back(e);Vec g(w.n,0);w.equalities[e].add_gradient(1,g);auto rg=coordinates.transpose(g);SparseRow row;
        for(int j=0;j<w.nr;++j)if(rg[j]!=0)row.entries.push_back({j,rg[j]});
        reduced.equalities.rows.push_back(std::move(row));
        reduced.equalities.rhs.push_back(-w.h[e]-dot(g,coordinates.offset));
      }
      for(const auto& q:p.inequalities)reduced.inequalities.push_back(convex_affine_pullback(q,coordinates));
      p=std::move(reduced);
    }
    Settings native=st;native.max_inner_iterations=st.study_inner_max;
    native.inner_tolerance=std::max(64*std::numeric_limits<double>::epsilon(),
        std::min(1e-9,w.objective.scale*std::min(st.stationarity_tolerance,st.dual_infeasibility_tolerance)*.01));
    // The outer certificate controls SUM |z*g| in ORIGINAL objective units.
    // A mean barrier target independent of row count/scale is insufficient.
    p.complementarity_tolerance=st.study_inner_complementarity*w.objective.scale/
        (4*std::max<std::size_t>(1,active.size()));
    native.inner_iteration_limit_mu_fallback=p.complementarity_tolerance;
    // A fixed 1e-12 positivity floor can make the stricter target impossible.
    // This is only IPM positivity protection, never function convexification.
    native.minimum_positive=std::min(st.minimum_positive,std::max(
        1024*std::numeric_limits<double>::min(),p.complementarity_tolerance*1e-4));
    native.restoration_deadline_seconds=std::min(st.restoration_deadline_seconds,
        std::chrono::duration<double>(start.time_since_epoch()).count()+st.time_limit_seconds);
    auto ir=ConvexQCQPInteriorPoint(native).solve(p);o.steps=std::min(st.study_inner_max,ir.iterations+1);
    o.reason="native-convex-"+ir.message;
    if(reduce && ir.y.size()==static_cast<std::size_t>(w.nr) &&
        ir.equality_duals.size()==retained_equalities.size() &&
        ir.inequality_duals.size()==active.size() && ir.inequality_slacks.size()==active.size()) {
      ir.y=coordinates.expand(ir.y);Vec lambda(w.me,0);
      for(std::size_t k=0;k<retained_equalities.size();++k)lambda[retained_equalities[k]]=ir.equality_duals[k];
      w.x=anchor.x;axpy(1,ir.y,w.x);w.lambda=lambda;
      for(std::size_t k=0;k<active.size();++k){w.z[active[k]]=ir.inequality_duals[k];w.s[active[k]]=ir.inequality_slacks[k];}
      w.evaluate();Vec lag=hybrid_model_dual(w,anchor,o);
      for(int j=0;j<w.n;++j)if(w.pivot_equation[j]>=0){int e=w.pivot_equation[j];lambda[e]=-lag[j]/w.pivot_coefficient[e];}
      ir.equality_duals=std::move(lambda);
      anchor.restore(w);
    }
    if(ir.y.size()==static_cast<std::size_t>(w.n)&&
        ir.equality_duals.size()==static_cast<std::size_t>(w.me)&&
        ir.inequality_duals.size()==active.size()&&ir.inequality_slacks.size()==active.size()) {
      w.x=anchor.x;axpy(1,ir.y,w.x);w.lambda=ir.equality_duals;
      long double sz=0;for(std::size_t k=0;k<active.size();++k){int i=active[k];w.z[i]=ir.inequality_duals[k];w.s[i]=ir.inequality_slacks[k];sz+=(long double)w.z[i]*w.s[i];}
      o.barrier_mu=active.empty()?0:static_cast<double>(sz/active.size());w.evaluate();
      hybrid_model_relative_gate(w,anchor,st,o);
      bool domain=finite(w.x)&&finite(w.lambda)&&finite(w.s)&&finite(w.z);
      for(int i:active)domain=domain&&w.s[i]>0&&w.z[i]>=0;
      o.success=native_model_certificate_authorized(domain,o.model_gate.passed,o.steps,
          st.study_inner_min,seconds(start),st.time_limit_seconds);
      if(o.success){
        complete.x=w.x;complete.s=w.s;complete.z=w.z;complete.lambda=w.lambda;
        for(int j=0;j<w.n;++j)complete.x[j]-=anchor.x[j];
        for(int i=0;i<w.mi;++i){complete.s[i]-=anchor.s[i];complete.z[i]-=anchor.z[i];}
        for(int e=0;e<w.me;++e)complete.lambda[e]-=anchor.lambda[e];
        o.reason="native-convex-same-model-relative-certificate";
      } else hybrid_try_terminal_original(w,st,start,outer,o);
    }
    std::cout<<"[HYBRID-NATIVE-CONVEX] outer="<<outer<<" native_success="<<ir.success
      <<" model_relative_pass="<<o.success<<" acceptance_authority=full-model-certificate"
      <<" kernel_absolute_success="<<ir.success<<" iterations="<<o.steps
      <<" affine_reduction="<<reduce<<" native_dimension="<<p.nvar<<" full_dimension="<<w.n
      <<" same_full_model_audit=1 exact_same_majorant=1 equalities_affine=1 historical_solver_fallback=0"<<std::endl;
    anchor.restore(w);w.update_matrix();
  }catch(const std::runtime_error& e){o.success=false;o.numerical_detail=e.what();o.reason="native-convex-error";anchor.restore(w);w.update_matrix();}
  const double trial_penalty=w.penalty;
  if(!o.success)w.penalty=inherited_penalty;
  std::cout<<"[HYBRID-NATIVE-PENALTY] outer="<<outer<<" inherited="<<inherited_penalty
      <<" model_seed="<<model_penalty_seed<<" trial="<<trial_penalty<<" installed="<<w.penalty
      <<" committed="<<o.success<<" per_model_seed=1 failed_trial_restored=1"<<std::endl;
  hybrid_model_relative_trace(o,st,"inner",outer,1,false);return o;
}
HybridInnerOutcome hybrid_inner_solve(Workspace& w,Direction& complete,double mu,
    const Settings& st,Clock::time_point start,int outer) {
  if(st.study_inner_rule=="native-relative"||st.study_inner_rule=="native-direction-relative"||st.study_inner_rule=="native-reduced-relative"||st.study_inner_rule=="native-reduced-direction-relative"||st.study_inner_rule=="native-direction-soc"||st.study_inner_rule=="native-reduced-direction-soc")return hybrid_inner_solve_native(w,complete,mu,st,start,outer);
  return st.study_inner_rule=="barrier-relative"?
      hybrid_inner_solve_barrier(w,complete,mu,st,start,outer):
      hybrid_inner_solve_relative(w,complete,mu,st,start,outer);
}
HybridInnerOutcome hybrid_outer_inner_gate(Workspace& w,const Direction& d,double a,
    double mu,const Settings& st,const HybridInnerOutcome& completed,const Vec* actual_slacks=nullptr) {
  HybridInnerAnchor anchor(w);auto o=completed;
  // Explicit alternative protocol: certify the FULL convex-model direction;
  // the outer filter/globalization checks the damped original trial separately.
  // Recompute even for SOC-combined proposals, never reuse a stale certificate.
  const bool direction_contract=st.study_inner_rule=="native-direction-relative"||st.study_inner_rule=="native-reduced-direction-relative"||st.study_inner_rule=="direction-relative"||st.study_inner_rule=="native-direction-soc"||st.study_inner_rule=="native-reduced-direction-soc";
  const double certificate_alpha=direction_contract?1:a;
  try {
    if(!(a>0&&a<=1))throw std::runtime_error("invalid globalization fraction");
    axpy(certificate_alpha,d.x,w.x);axpy(certificate_alpha,d.s,w.s);
    axpy(certificate_alpha,d.z,w.z);axpy(certificate_alpha,d.lambda,w.lambda);
    if(actual_slacks&&!direction_contract) {
      if(actual_slacks->size()!=w.s.size())throw std::runtime_error("outer slack-path dimension mismatch");
      w.s=*actual_slacks;
    }
    w.evaluate();
    if(o.model_relative) {
      hybrid_model_relative_gate(w,anchor,st,o,false);o.success=o.model_gate.passed;
      o.reason=direction_contract?(o.success?"full-direction-model-relative-criterion":"full-direction-model-relative-rejected"):
          (o.success?"actual-alpha-model-relative-criterion":"actual-alpha-model-relative-rejected");
      if(direction_contract)std::cout<<"[HYBRID-DIRECTION-CONTRACT] actual_alpha="<<a
          <<" certificate_alpha=1 model_certificate_pass="<<o.success
          <<" original_trial_filter_required=1 trial_stationarity_claimed=0"<<std::endl;
      anchor.restore(w);return o;
    }
    o.final=hybrid_model_residual(w,mu,anchor,o);o.step_inf=hybrid_displacement(w.x,anchor.x);
    o.model_merit=hybrid_model_merit(w,anchor,mu,o);
    o.gate=hybrid_inner::check(o.final,o.initial,o.step_inf,st.study_inner_relative,
        st.study_inner_absolute,o.model_merit,o.anchor_merit,o.scaled_mu);
    o.reason=o.gate.passed?"actual-alpha-relative-criterion":"actual-alpha-inner-criterion-rejected";
    o.success=o.gate.passed;
  }catch(const std::runtime_error&){o.success=false;o.gate.passed=false;o.reason="actual-alpha-nonfinite";}
  anchor.restore(w);return o;
}


Result solve_inequality_majorants(const QCQPProblem& problem,const Settings& requested,
                                 bool allow_affine_start_repair=true) {
  const int checkpoint_majorant_depth=requested.majorant_call_depth+1;

  problem.validate();
  detail::validate_gr_proximal_policy(requested.gr_proximal_policy);
  detail::validate_majorant_equality_stabilization_problem(problem,requested);
  const bool bound_product_curve=detail::original_boolean_mode("LAZY_ORIGINAL_BOUND_PRODUCT_PATH",
      requested.original_bound_product_path)&&
      !requested.original_private_phase;
  if(bound_product_curve&&(!requested.majorant_equality_stabilization||
      !detail::original_source_primal_barrier_requested(requested)))
    throw std::invalid_argument("bound product dual path requires GR merit, free source primal barrier and native source stabilization");
  const char* gr_soc_flag=std::getenv("LAZY_ORIGINAL_GR_SOC");
  if(gr_soc_flag&&std::string(gr_soc_flag)!="0"&&std::string(gr_soc_flag)!="1")
    throw std::invalid_argument("LAZY_ORIGINAL_GR_SOC expects 0 or 1");
  const bool gr_defined_soc=gr_soc_flag&&std::string(gr_soc_flag)=="1"&&
      !requested.original_private_phase;
  const bool gr_kkt_soc=requested.original_gr_kkt_soc&&!requested.original_private_phase;
  if(gr_kkt_soc&&gr_defined_soc)
    throw std::invalid_argument("choose either general KKT SOC or private-pivot GR SOC");
  if(gr_defined_soc&&!bound_product_curve)
    throw std::invalid_argument("GR defined-pivot SOC requires native GR, free source primal barrier and positive bound product path");
  const char* gr_soc_domain_flag=std::getenv("LAZY_ORIGINAL_GR_SOC_DOMAIN_DIAGNOSTIC");
  if(gr_soc_domain_flag&&std::string(gr_soc_domain_flag)!="0"&&std::string(gr_soc_domain_flag)!="1")
    throw std::invalid_argument("LAZY_ORIGINAL_GR_SOC_DOMAIN_DIAGNOSTIC expects 0 or 1");
  const bool gr_soc_domain_diagnostic=gr_soc_domain_flag&&std::string(gr_soc_domain_flag)=="1"&&
      !requested.original_private_phase;
  if(gr_soc_domain_diagnostic&&!gr_defined_soc)
    throw std::invalid_argument("SOC domain diagnostic requires the existing analytic GR SOC mode");
  const bool gr_short_step_prox=detail::original_boolean_mode("LAZY_ORIGINAL_GR_SHORT_STEP_PROX",
      requested.original_gr_short_step_prox)&&
      !requested.original_private_phase;
  if(gr_short_step_prox&&(!requested.majorant_equality_stabilization||
      !detail::original_source_primal_barrier_requested(requested)))
    throw std::invalid_argument("GR short-step proximal control requires native source primal barrier and stabilization");
  const bool gr_consistent_bound_slacks=detail::original_boolean_mode("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_SLACKS",
      requested.original_gr_consistent_bound_slacks)&&
      !requested.original_private_phase;
  if(gr_consistent_bound_slacks&&(!bound_product_curve||gr_defined_soc||
      !requested.majorant_equality_stabilization||!detail::original_source_primal_barrier_requested(requested)))
    throw std::invalid_argument("consistent bound slacks require native GR/source primal barrier/positive bound product path and analytic SOC disabled");
  // GR-BOUND-REPAIR-PARSER-BEGIN
  const char* bound_trigger_flag=std::getenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER");
  const std::string gr_bound_trigger=bound_trigger_flag?bound_trigger_flag:requested.original_gr_consistent_bound_trigger;
  if(gr_bound_trigger!="always"&&gr_bound_trigger!="on_failure")
    throw std::invalid_argument("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER expects always or on_failure");
  const bool gr_bound_repair_on_failure=gr_bound_trigger=="on_failure";
  // GR-BOUND-REPAIR-PARSER-END
  detail::GrBoundStagnationWatch gr_bound_watch;
  // A severely infeasible affine subsystem can swamp a mixed QCQP's first
  // Newton model. Restore that subsystem before constructing any nonlinear
  // model; this is an explicit native Phase I, not a replacement cold start.
  // Admission uses coefficients/residuals only and never an instance label.
  if(allow_affine_start_repair&&requested.restoration_enabled&&
      requested.majorant_startup_policy=="auto"&&!problem.has_signed_square()&&
      (!requested.rank_presolve_context||!requested.rank_presolve_context->changed())) {
    QCQPProblem affine=problem;
    affine.equalities.clear();affine.inequalities.clear();
    bool curved=false;double affine_relative_violation=0;
    auto admit_row=[&](const auto& q,bool equality) {
      if(!q.hessian.entries.empty()){curved=true;return;}
      double scale=std::abs(q.constant);
      q.for_each_linear_nonzero([&](int,double a){scale=std::max(scale,std::abs(a));});
      const double value=q.value(problem.x0);
      affine_relative_violation=std::max(affine_relative_violation,
          scale>0?(equality?std::abs(value):std::max(0.,value))/scale:0.);
      if(equality)affine.equalities.push_back(q);else affine.inequalities.push_back(q);
    };
    for(const auto& q:problem.equalities)admit_row(q,true);
    for(const auto& q:problem.inequalities)admit_row(q,false);
    const int affine_iteration_budget=detail::affine_start_iteration_allowance(
        requested.restoration_shared_iteration_budget,requested.restoration_max_iterations,
        requested.max_outer_iterations);
    if(curved&&!affine.equalities.empty()&&affine_relative_violation>100&&
        requested.time_limit_seconds>0&&affine_iteration_budget>0) {
      const auto began=Clock::now();
      affine.objective=QuadraticFunction{};
      affine.objective.linear.assign(problem.n,0);affine.objective.hessian.n=problem.n;
      Settings phase=requested;
      phase.original_private_phase=true;phase.majorant_call_depth=checkpoint_majorant_depth;
      // Startup restoration keeps its historical equations. Stabilization
      // begins only in the original-model R3 continuation below.
      phase.majorant_equality_stabilization=false;
      phase.convexification_study="off";phase.objective_lazy_convexification=false;
      phase.majorant_startup_policy="legacy";phase.restoration_enabled=false;
      phase.majorant_initial_restoration=false;phase.majorant_conditional_retraction=false;
      phase.rank_presolve_context=nullptr;
      phase.time_limit_seconds=std::min(30.,requested.time_limit_seconds);
      phase.max_outer_iterations=affine_iteration_budget;
      phase.live_display_level=0;phase.live_iterations=false;
      auto affine_result=solve_inequality_majorants(affine,phase,false);
      Result affine_work=affine_result;
      // Private artificial-objective histories are reported separately, never
      // inserted into the original objective's iteration-history plot.
      affine_work.lazy_newton.iteration_history.clear();
      auto original_violation=[&](const Vec& x) {
        return detail::native_affine_original_residuals(problem,x);
      };
      const auto before=original_violation(problem.x0);
      auto after=original_violation(affine_result.x);
      int used=affine_result.outer_iterations;
      auto phase_factorizations=affine_result.kkt_factorizations;
      bool minimum_norm_selected=false,minimum_norm_partial_selected=false;
      // A zero objective leaves free coordinates arbitrary. Also try the
      // convex minimum-norm point, but retain the historical point unless
      // BOTH original max violation and original L1 violation improve.
      // All attempted work shares the initial 30 s / 200 iteration envelope
      // and is charged to the full original solve, including rejected work.
      const double phase_remaining=phase.time_limit_seconds-seconds(began);
      if(phase_remaining>0&&phase.max_outer_iterations-used>0) {
        QCQPProblem norm_problem=affine;
        for(int j=0;j<problem.n;++j)norm_problem.objective.hessian.entries.push_back({j,j,1.});
        Settings norm_settings=phase;
        norm_settings.time_limit_seconds=phase_remaining;
        norm_settings.max_outer_iterations=phase.max_outer_iterations-used;
        auto norm_result=solve_inequality_majorants(norm_problem,norm_settings,false);
        const auto norm_violation=original_violation(norm_result.x);
        const bool norm_solved_accepted=norm_result.status==Status::Solved&&
            std::isfinite(norm_violation.first)&&std::isfinite(norm_violation.second)&&
            (affine_result.status!=Status::Solved||
             (norm_violation.first<after.first&&norm_violation.second<after.second));
        bool norm_bounds_ok=norm_result.x.size()==std::size_t(problem.n)&&finite(norm_result.x);
        if(norm_bounds_ok)for(int j=0;j<problem.n;++j) {
          if(std::isfinite(problem.lb[j])&&norm_result.x[j]<problem.lb[j]-requested.feasibility_tolerance)norm_bounds_ok=false;
          if(std::isfinite(problem.ub[j])&&norm_result.x[j]>problem.ub[j]+requested.feasibility_tolerance)norm_bounds_ok=false;
        }
        // This is a native feasibility phase, not an optimization endpoint.
        // A bounded partial minimum-norm iterate may be a better ORIGINAL
        // anchor even when the artificial phase objective has not converged.
        // Require independent finite original max AND L1 meaningful progress,
        // keep all phase work charged, and never infer an original KKT claim.
        const bool norm_partial_accepted=detail::native_affine_partial_progress(
            norm_result.status==Status::Solved,norm_bounds_ok,norm_violation.first,
            norm_violation.second,after.first,after.second,seconds(began),requested.time_limit_seconds);
        const bool norm_accepted=norm_solved_accepted||norm_partial_accepted;
        Result norm_work=norm_result;norm_work.lazy_newton.iteration_history.clear();
        detail::merge_rank_retry_work(affine_work,norm_work,false,false);
        used+=norm_result.outer_iterations;phase_factorizations+=norm_result.kkt_factorizations;
        if(requested.live_display_level>0)std::cout<<"[AFFINE-START-MINIMUM-NORM] selected="<<norm_accepted
          <<" original_primal="<<double(after.first)<<"->"<<double(norm_violation.first)
          <<" original_l1="<<double(after.second)<<"->"<<double(norm_violation.second)
          <<" iterations="<<norm_result.outer_iterations
          <<" partial_feasibility_candidate="<<norm_partial_accepted
          <<" phase_status="<<status_string(norm_result.status)<<" final_KKT_certificate=0"
          <<" shared_time_budget=1 shared_iteration_budget=1"<<std::endl;
        if(norm_accepted){affine_result=std::move(norm_result);after=norm_violation;
          minimum_norm_selected=true;minimum_norm_partial_selected=norm_partial_accepted;}
      }
      const double spent=seconds(began);
      const double phase_overhead=std::max(0.0,spent-affine_work.elapsed_seconds);
      affine_work.lazy_newton.cost_seconds[static_cast<int>(NewtonCost::Preparation)]+=phase_overhead;
      affine_work.lazy_newton.setup_seconds+=phase_overhead;
      affine_work.elapsed_seconds=spent;affine_work.lazy_newton.solver_wall_seconds=spent;
      const bool accepted=(affine_result.status==Status::Solved||minimum_norm_partial_selected)&&
          std::isfinite(after.first)&&std::isfinite(after.second)&&
          after.first<.9L*before.first&&after.second<.9L*before.second&&
          spent<requested.time_limit_seconds&&used<requested.max_outer_iterations;
      if(requested.live_display_level>0)std::cout<<"[AFFINE-START-RESTORATION] accepted="<<accepted
        <<" selection="<<(minimum_norm_selected?"minimum-norm":"historical")
        <<" objective_original_unchanged=1"
        <<" affine_relative_violation="<<affine_relative_violation
        <<" original_primal="<<double(before.first)<<"->"<<double(after.first)
        <<" original_l1="<<double(before.second)<<"->"<<double(after.second)
        <<" iterations="<<used<<" iteration_budget="<<affine_iteration_budget
        <<" shared_restoration_budget="<<requested.restoration_shared_iteration_budget
        <<" restoration_remaining="<<detail::remaining_shared_restoration(
            requested.restoration_max_iterations,used)<<" seconds="<<spent
        <<" native_backend="<<requested.linear_backend
        <<" input_x0_unchanged=1 explicit_algorithm_step=1 global_budget_preserved=1"<<std::endl;
      {
        QCQPProblem repaired=problem;if(accepted)repaired.x0=affine_result.x;
        Settings remaining=requested;
        remaining.majorant_call_depth=checkpoint_majorant_depth;
        remaining.time_limit_seconds-=spent;remaining.max_outer_iterations-=used;
        if(requested.restoration_shared_iteration_budget)
          remaining.restoration_max_iterations=detail::remaining_shared_restoration(
              requested.restoration_max_iterations,used);
        if(!(remaining.time_limit_seconds>0)||remaining.max_outer_iterations<1) {
          Result limited=affine_work;limited.x=problem.x0;
          limited.certificate_equality_multipliers.clear();limited.certificate_inequality_multipliers.clear();
          limited.certificate_lower_multipliers.clear();limited.certificate_upper_multipliers.clear();
          limited.termination_audit_valid=false;
          limited.acceptable_convergence=limited.acceptable_point=limited.acceptable_on_failure=false;
          limited.acceptable_streak=0;limited.acceptable_trigger.clear();
          limited.raw_kkt_error=limited.acceptable_error=INFINITY;
          limited.termination_scaling_available=false;
          limited.termination_equality_scales.clear();limited.termination_inequality_scales.clear();
          limited.termination_inequality_slacks.clear();
          limited.termination_internal_primal_inf=limited.termination_internal_dual_inf=limited.termination_internal_compl_inf=INFINITY;
          limited.termination_dual_inf=limited.termination_compl_inf=limited.original_normalized_nlp_error=INFINITY;
          limited.feasibility={INFINITY,INFINITY,INFINITY,INFINITY};
          limited.dual_infeasibility_inf=INFINITY;limited.complementarity_inf=INFINITY;
          limited.objective=problem.objective.value(problem.x0);
          limited.status=remaining.time_limit_seconds>0?Status::MaxOuterIterations:Status::TimeLimit;
          limited.message="budget exhausted during native affine startup restoration; no original KKT certificate";
          limited.elapsed_seconds=spent;limited.outer_iterations=used;
          limited.restoration_iterations+=used;limited.restoration_total_seconds+=spent;
          limited.kkt_factorizations=phase_factorizations;return limited;
        }
        // All functions, constraints, bounds and coordinates were copied
        // unchanged above; only the start may be the charged native affine
        // phase result. Keep exact ownership through this live recursive call.
        detail::NativeAnchorIdentityAlias native_repaired_identity(repaired,problem);
        auto continued=solve_inequality_majorants(repaired,remaining,false);
        const double continued_elapsed=continued.elapsed_seconds;
        detail::merge_rank_retry_work(continued,affine_work,false,true);
        continued.restoration_iterations+=used;
        continued.restoration_total_seconds+=spent;
        // The actual wrapper clock includes preparation/selection work beyond
        // the two child solve clocks; never add their inclusive times twice.
        continued.elapsed_seconds=continued_elapsed+spent;
        continued.lazy_newton.solver_wall_seconds=continued.elapsed_seconds;
        return continued;
      }
    }
  }
  Settings st=requested;
  const bool gr_requested=detail::original_stabilized_merit_requested(st);
  if(gr_requested&&!st.majorant_equality_stabilization&&allow_affine_start_repair)
    throw std::invalid_argument("GR merit requires explicit all-source constraint stabilization");
  const bool constant_square_filter=detail::constant_square_feasibility_filter(problem,st);
  const bool gr_globalization=gr_requested&&st.majorant_equality_stabilization&&!constant_square_filter;
  if(constant_square_filter&&st.live_display_level)
    std::cout<<"[CONSTANT-SQUARE-POLICY] globalization=original-feasibility-filter"
        <<" objective=exactly-constant bound_barriers=0 normal_equations=exact"
        <<" optimization_equality_stabilization_unchanged=1 original_KKT_required=1"<<std::endl;
  if(gr_globalization) {
    // First isolated merit diagnostic: do not commit a point through a
    // recovery whose only globalization authority is the historical filter.
    st.majorant_conditional_retraction=false;st.majorant_nonlinear_trial_soc=false;
    st.majorant_trial_soc_arc=false;st.majorant_soc_defect_control=false;
    st.majorant_recalc_lambda=false;st.majorant_lambda_interpolation=false;
    st.majorant_lambda_rho_retry=false;st.majorant_filter_recenter=false;
  }
  const bool hybrid_corrections=st.convexification_study=="HH"||
      st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS";
  if(st.study_controller!="r3"&&st.study_controller!="certified")throw std::invalid_argument("invalid study controller");
  const bool r3_hybrid=hybrid_corrections&&st.study_controller=="r3";
  const bool hybrid_mode=hybrid_corrections&&!r3_hybrid;
  if(hybrid_corrections)study::validate_h4_domain(problem);
  if(st.convexification_study!="off") {
    const std::set<std::string> modes={"U","F","SS","SS_legacy","SA","AS","AA","HH","HS","SH"};
    if(!modes.count(st.convexification_study)||
       (st.convexification_schedule!="lazy"&&st.convexification_schedule!="eager"))
      throw std::invalid_argument("invalid convexification study mode/schedule");
    if((st.convexification_study=="U"||st.convexification_study=="F")&&st.convexification_schedule!="eager")
      throw std::invalid_argument("fixed U/F require eager construction");
    if(st.linear_backend!="mumps"||!st.linear_metis||st.inner_backend=="ipopt")
      throw std::invalid_argument("study requires native MUMPS+METIS");
    if(problem.has_signed_square())throw std::invalid_argument("convexification study supports true QCQP, not signed-square piecewise functions");
    st.objective_lazy_convexification=true;
    st.majorant_curvature_corrections=0;
  }
  if(hybrid_mode) {
    if(st.study_inner_rule!="model-relative"&&st.study_inner_rule!="barrier-relative"&&st.study_inner_rule!="native-relative"&&st.study_inner_rule!="native-direction-relative"&&st.study_inner_rule!="native-reduced-relative"&&st.study_inner_rule!="native-reduced-direction-relative"&&st.study_inner_rule!="direction-relative"&&st.study_inner_rule!="native-direction-soc"&&st.study_inner_rule!="native-reduced-direction-soc")
      throw std::invalid_argument("unknown hybrid inner rule");
    if(st.study_inner_min<1||st.study_inner_max<st.study_inner_min||
       !std::isfinite(st.study_inner_relative)||st.study_inner_relative<=0||st.study_inner_relative>=1||
       !std::isfinite(st.study_inner_absolute)||st.study_inner_absolute<0)
      throw std::invalid_argument("invalid hybrid inner budget or relative accuracy");
    if(st.study_inner_rule!="barrier-relative") {
      if(!std::isfinite(st.study_inner_kappa_fraction)||st.study_inner_kappa_fraction<=0||
         st.study_inner_kappa_fraction+st.study_inner_relative>=.5||
         !std::isfinite(st.study_inner_margin)||st.study_inner_margin<=0||
         !std::isfinite(st.study_inner_barrier_decay)||st.study_inner_barrier_decay<=0||st.study_inner_barrier_decay>=1)
        throw std::invalid_argument("model-relative rule requires positive margin and kappa_fraction + relative < 1/2");
      for(double v:{st.study_inner_activity,st.study_inner_feasibility,st.study_inner_complementarity})
        if(!std::isfinite(v)||v<0)throw std::invalid_argument("invalid finite-accuracy model tolerance");
      if(st.study_inner_feasibility>st.feasibility_tolerance||st.study_inner_complementarity>st.complementarity_tolerance)
        throw std::invalid_argument("inner model floors exceed common feasibility/complementarity tolerances");
    }
  }
  if(st.majorant_startup_policy!="legacy"&&st.majorant_startup_policy!="auto")
    throw std::invalid_argument("unknown majorant startup policy");
  if(st.majorant_startup_policy=="auto" &&
     (!st.restoration_enabled || st.lazy_newton_initial_mu!=Settings{}.lazy_newton_initial_mu))
    throw std::invalid_argument("auto startup owns initial mu and keeps recovery enabled: remove --no-restoration and --lazy-newton-initial-mu");
  if(st.majorant_curvature_corrections<0||st.majorant_curvature_corrections>16)throw std::invalid_argument("invalid curvature correction budget");
  if(st.majorant_lambda_trigger!="consecutive"&&st.majorant_lambda_trigger!="window")throw std::invalid_argument("invalid lambda trigger");
  if(st.majorant_recalc_lambda_max_iterations<1)throw std::invalid_argument("invalid lambda projection budget");
  const int level=st.live_display_level<0?(st.live_iterations?1:0):st.live_display_level;
  if(level<0||level>4||st.live_iteration_every<1||st.max_outer_iterations<1)
    throw std::invalid_argument("invalid inequality-majorant iteration/display budget");
  for(double tolerance:{st.stationarity_tolerance,st.feasibility_tolerance,
      st.dual_infeasibility_tolerance,st.complementarity_tolerance})
    if(!(tolerance>0)||!std::isfinite(tolerance))
      throw std::invalid_argument("inequality-majorant tolerances must be positive and finite");
  if(!(st.time_limit_seconds>0)||!std::isfinite(st.time_limit_seconds)||
      !(st.line_search_contraction>0&&st.line_search_contraction<1)||
      !(st.line_search_armijo>0&&st.line_search_armijo<1)||st.line_search_max_steps<1||
      !(st.lazy_newton_fraction_to_boundary>0&&st.lazy_newton_fraction_to_boundary<1))
    throw std::invalid_argument("invalid inequality-majorant time/globalization settings");
  if(!st.dca_endpoint_start_file.empty())
    throw std::invalid_argument("inequality-majorant research route requires the problem's frozen initial point");
  if(st.objective_max_atoms<0||st.objective_initial_atoms<0||st.objective_atom_batch<1||
     !std::isfinite(st.objective_rho_decay)||st.objective_rho_decay<=0||st.objective_rho_decay>=1||
     !std::isfinite(st.objective_rho_excess)||st.objective_rho_excess<0)
    throw std::invalid_argument("invalid objective rho controls");
  st.dca_engine="majorants";
  st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;
  st.dca_aggregate=false;st.dca_aggregate_only=false;
  st.dca_krylov=false;st.dca_local_active=false;st.dca_strict_finish=false;
  st.verbose=false;st.live_curvature=level>=4;st.live_display_level=level;
  st.lazy_newton_predictor_corrector=hybrid_mode&&st.study_inner_rule!="barrier-relative"&&st.study_inner_predictor_corrector;
  Result result;
  result.x=problem.x0;
  result.effective_outer_algorithm="lazy-dca";
  result.lazy_dca_equality_model=(st.convexification_study=="off"||r3_hybrid)?"original-nonlinear-equalities":"linearized";
  result.linear_solver_backend="inequality-majorants/"+st.linear_backend;
  result.dca_dual_step="common";
  auto& stats=result.dca_filter;
  stats.used=true;stats.inequality_majorants_enabled=true;
  stats.majorant_filter_reanchor_enabled=true;
  stats.majorant_outer_only_enabled=!hybrid_mode;
  stats.majorant_barrier_filter_enabled=true;
  stats.majorant_original_filter_enabled=true;
  if(gr_globalization) {
    stats.majorant_original_filter_enabled=false;stats.majorant_barrier_filter_enabled=false;
    if(level)std::cout<<"[GLOBALIZATION-VARIANT] variant=original-Gill-Robinson-merit-v1"
        <<" operator="<<(detail::original_source_primal_barrier_requested(st)?
            detail::MajorantEqualityStabilization::primal_barrier_variant:detail::MajorantEqualityStabilization::variant)
        <<" original_filter=0 original_filter_fallback=0 legacy_SOC=0 gr_defined_soc="<<gr_defined_soc
        <<" gr_kkt_soc="<<gr_kkt_soc<<" lambda_recalc=0"
        <<" late_phase1=0 direction_centering=0 recenter=0 central_retry=0"
        <<" terminal_primal_recovery=0"
        <<" initial_phase1_unchanged=1 strict_original_KKT_required=1"
        <<" consistent_bound_master="<<gr_consistent_bound_slacks<<" consistent_bound_trigger="<<gr_bound_trigger
        <<" consistent_bound_retry="<<(gr_bound_repair_on_failure?"full-step-only":"full-line-search")
        <<" short_step_proximal="<<gr_short_step_prox
        <<" bound_dual_path="<<(bound_product_curve?"positive-product":"affine")
        <<" affine_dual_decomposition="<<(bound_product_curve?"disabled-curved-duals":"unchanged")<<std::endl;
  }
  stats.majorant_slack_transport_enabled=false;
  stats.majorant_terminal_single_newton=!hybrid_mode;
  if(hybrid_corrections)std::cout<<"[HYBRID-ARCHITECTURE] controller="<<st.study_controller
      <<" equalities="<<(r3_hybrid?"original-nonlinear":"linearized")
      <<" single_auxiliary_newton="<<r3_hybrid<<" inner_completion="<<hybrid_mode
      <<" original_filter="<<(!gr_globalization)<<" stabilization="<<(st.majorant_equality_stabilization?
          detail::MajorantEqualityStabilization::variant:(r3_hybrid?"historical-R3":"certified-model-proximal"))
      <<" function_corrections=certified-hybrid no_reference_fallback=1"<<std::endl;
  stats.aggregate_enabled=false;stats.aggregate_only_enabled=false;
  stats.krylov_enabled=false;stats.strict_finish_enabled=false;
  stats.curvature_backend=st.convexification_study=="off"?"direct-spectrum/gershgorin":"study-certified-scalar-or-atoms";
  stats.active_evaluation=false;
  result.lazy_newton.display_level=level;
  result.lazy_newton.predictor_corrector_requested=requested.lazy_newton_predictor_corrector;
  result.lazy_newton.predictor_corrector_enabled=st.lazy_newton_predictor_corrector;
  result.lazy_newton.sparse_certificate_enabled=st.lazy_newton_sparse_certificate;
  const auto start=Clock::now();
  detail::NewtonCostLedger costs(result.lazy_newton,start);
  std::unique_ptr<Workspace> owner;
  std::unique_ptr<DcaInequalityModels> models;
  bool audited_current=false,original_cache_current=false;
  // Original constant objectives admit the exact zero-multiplier KKT
  // certificate at every original-feasible point, independently of a barrier.
  bool original_constant_objective=problem.objective.signed_square.empty()&&
      std::all_of(problem.objective.hessian.entries.begin(),problem.objective.hessian.entries.end(),
          [](const auto& e){return e.v==0.0;});
  problem.objective.for_each_linear_nonzero([&](int,double v){
    if(v!=0.0)original_constant_objective=false;
  });
  auto audit_original=[&]() {
    CostScope certificate(costs,NewtonCost::Certificate);
    owner->export_state(result);
    bool passed=detail::audit_original_termination(problem,requested,result,
        &owner->original_linear_support);
    if(!passed&&(hybrid_mode||original_constant_objective)&&
        hybrid_try_zero_gradient_certificate(*owner,requested,start)) {
      owner->export_state(result);
      passed=detail::audit_original_termination(problem,requested,result,&owner->original_linear_support);
    }
    audited_current=true;
    return passed;
  };
  auto plausible_original=[&](const DcaOuterMetrics& current) {
    // This only triggers the authoritative original certificate. No auxiliary
    // convergence or slack-complementarity condition can block that test.
    return current.primal<=requested.feasibility_tolerance&&
        ((current.dual<=requested.dual_infeasibility_tolerance&&
          current.comp<=requested.complementarity_tolerance)||
         original_constant_objective||
         (hybrid_mode&&inf_norm(owner->objective.gradient)==0));
  };
  result.status=Status::MaxOuterIterations;
  result.message="inequality-majorant outer iteration limit";
  try {
    bool all_fixed=true;
    for(int j=0;j<problem.n;++j)
      all_fixed=all_fixed&&std::isfinite(problem.lb[j])&&problem.lb[j]==problem.ub[j];
    if(all_fixed) {
      result.x=problem.lb;
      result.certificate_equality_multipliers.assign(problem.equalities.size(),0);
      result.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
      result.certificate_lower_multipliers.assign(problem.n,0);
      result.certificate_upper_multipliers.assign(problem.n,0);
      if(requested.termination_test=="ipopt") {
        result.termination_scaling_available=true;result.termination_objective_scale=1;
        result.termination_equality_scales.assign(problem.equalities.size(),1);
        result.termination_inequality_scales.assign(problem.inequalities.size(),1);
        result.termination_inequality_slacks.assign(problem.inequalities.size(),0);
      }
      bool ok;
      {CostScope certificate(costs,NewtonCost::Certificate);
        ok=detail::audit_original_termination(problem,requested,result);}
      result.status=ok?Status::Solved:Status::RestorationFailure;
      result.message=ok?"fixed point satisfies original QCQP KKT":"fixed point violates original QCQP constraints";
      audited_current=true;
    } else {
      {CostScope preparation(costs,NewtonCost::Preparation);
        owner=std::make_unique<Workspace>(problem,st,result.lazy_newton,costs);
        owner->dca_stats=&stats;
        models=std::make_unique<DcaInequalityModels>(*owner,stats);}
      auto& w=*owner;
      DcaDefinedPivotSoc gr_soc_plan;
      if(gr_defined_soc) {
        CostScope preparation(costs,NewtonCost::Preparation);
        gr_soc_plan=DcaDefinedPivotSoc::build(w.n,w.equalities);
        if(level)std::cout<<"[GR-DEFINED-SOC-GEOMETRY] supported="<<gr_soc_plan.supported
            <<" curved_equalities="<<gr_soc_plan.curved_equalities
            <<" uncovered="<<gr_soc_plan.uncovered_rows.size()<<" reason="<<gr_soc_plan.reason
            <<" local_trial_retraction=1 global_product_elimination=0 Newton_KKT_correction_claimed=0"<<std::endl;
      }
      if(gr_globalization&&st.rank_presolve_context&&
          st.rank_presolve_context->has_omitted_equality_bounds())
        throw std::invalid_argument("GR merit prototype requires all canonical bound links retained");
      // DIAGNOSTIC ONLY: read the existing rank reporter to name original
      // source rows/physical bound coordinates. This changes no model/state.
      std::vector<int> diagnostic_inequality_origins,diagnostic_lower_origins,diagnostic_upper_origins;
      if(gr_soc_domain_diagnostic&&st.rank_presolve_context) {
        const std::string report=st.rank_presolve_context->report_json();
        auto indices=[&](const char* name) {
          std::vector<int> values;const std::string key=std::string("\"")+name+"\":[";
          auto pos=report.find(key);if(pos==std::string::npos)return values;pos+=key.size();
          while(pos<report.size()&&report[pos]!=']') {
            char* end=nullptr;const long value=std::strtol(report.c_str()+pos,&end,10);
            if(end==report.c_str()+pos||value<std::numeric_limits<int>::min()||value>std::numeric_limits<int>::max())break;
            values.push_back(static_cast<int>(value));pos=static_cast<std::size_t>(end-report.c_str());
            if(report[pos]==',')++pos;else if(report[pos]!=']')break;
          }
          return values;
        };
        diagnostic_inequality_origins=indices("inequality_origins");
        diagnostic_lower_origins=indices("lower_bound_origins");
        diagnostic_upper_origins=indices("upper_bound_origins");
      }
      if(gr_soc_domain_diagnostic)std::cout<<"[GR-SOC-DOMAIN-DIAGNOSTIC] enabled=1 default=0 numerical_behavior_unchanged=1"
          <<" original_index_scope="<<(st.rank_presolve_context?"rank-context-original":"workspace-input")
          <<" original_IQ_map_size="<<diagnostic_inequality_origins.size()
          <<" bound_map_size="<<diagnostic_lower_origins.size()<<std::endl;
      bool central_first=w.auto_startup.central_first;
      // Save the exact FREE-coordinate cold start before any restoration.
      // A retry will neither import an endpoint nor obtain another budget.
      const Vec auto_frozen_x=w.x;
      double auto_initial_mu=result.lazy_newton.initial_barrier;
      if(w.source_primal_barrier_mode)w.set_barrier_mu(auto_initial_mu);
      if(gr_globalization&&level)std::cout<<"[SOURCE-PRIMAL-BARRIER] enabled="<<w.source_primal_barrier_mode
          <<" source_IQ_equality_dual="<<(w.source_primal_barrier_mode?"free":"positive-PD")
          <<" source_slack_hessian="<<(w.source_primal_barrier_mode?"mu/s^2":"y/s")
          <<" all_slacks_positive=1 bound_duals_positive=1 original_terminal_signs_required=1"<<std::endl;
      detail::AutoStartupRetry auto_startup_retry;
      const bool auto_retry_eligible=!gr_globalization&&st.majorant_startup_policy=="auto" &&
          !central_first && w.me>0 && w.mi>0 && !problem.has_signed_square() &&
          !st.objective_lazy_convexification && finite(w.x) &&
          std::isfinite(auto_initial_mu) && auto_initial_mu>0 && auto_initial_mu<=1e-4;
      if(level && st.majorant_startup_policy=="auto")std::cout<<"[AUTO-STARTUP] selected="
          <<(central_first?"central-first":"legacy")<<" reason="<<w.auto_startup.reason
          <<" effective_initial_mu="<<result.lazy_newton.initial_barrier
          <<" startup_restoration_deferred="<<central_first
          <<" recovery_enabled="<<st.restoration_enabled<<" primal_unchanged=1"<<std::endl;
      if(level)std::cout<<"[MAJORANT-LINEAR] requested="<<st.linear_backend
          <<" metis="<<st.linear_metis<<" factory="<<w.factor->name()
          <<" cross_backend_fallback="<<st.dca_filter_sparse_fallback<<std::endl;
      // The startup restoration was introduced for a ZERO objective Hessian
      // (constant feasibility objectives and affine flat-chain starts). Applying
      // it to every equality-only problem moves a curved objective away from
      // its frozen start before the first Lazy model, and initializes the filter
      // cap from that unrelated feasible point. Use the ordinary original-model
      // direction for a curved objective. This is a structural test in FREE
      // coordinates, not a problem-name test, a curvature certificate, or a
      // change to the requested final KKT tolerances. Native signed squares
      // retain their separate existing startup policy.
      const bool free_objective_curvature=std::any_of(w.objective.hessian.begin(),
          w.objective.hessian.end(),[](const auto& e){return e.v!=0.0;});
      const bool equality_startup=w.mi==0&&!free_objective_curvature;
      // A nonlinear equality does not become a usable first-order model merely
      // because an inequality (or a free-variable bound) is present. Preserve
      // the historical equality-only path. Extend it only to an affine-objective
      // mixed model with an actually infeasible NONLINEAR equality, in free
      // coordinates. Linear/feasible equalities and curved objectives are no-ops.
      const bool mixed_affine_equality_startup=w.mi>0&&!free_objective_curvature&&
          std::any_of(w.equalities.begin(),w.equalities.end(),[&](const auto& q){
            return std::abs(q.value)/q.scale>st.feasibility_tolerance&&
                std::any_of(q.hessian.begin(),q.hessian.end(),
                    [](const auto& e){return e.v!=0.0;});
          });
      // For a square equality-only feasibility problem, the constraint
      // Jacobian determines the primal Newton step. A unit normal metric
      // supplies nonzero primal pivots without adding objective curvature:
      // [I J^T; J 0] [dx; eta] = [0; -h]. The original multipliers are zero,
      // not eta (which belongs solely to this minimum-norm normal subproblem).
      // Rank/pivot/expanded residual failure retains the old startup policy.
      bool square_feasibility=false;
      bool square_prepared=false;
      bool square_factor_failed=false;
      Direction square_first;
      // Exact removal of unused coordinates or retention of redundant
      // original equality rows can make a constant feasibility model tall.
      // The same minimum-norm normal system is valid when J*dx=-h is
      // consistent; its existing ORIGINAL residual guard still decides that.
      // Preserve the previous route for every nonconstant objective and
      // every underdetermined model.
      const bool square_candidate=w.mi==0&&w.n>0&&
          (w.me==w.n||(original_constant_objective&&w.me>w.n))&&
          !free_objective_curvature&&!problem.has_signed_square()&&
          std::all_of(w.objective.gradient.begin(),w.objective.gradient.end(),
              [](double value){return value==0.0;});
      // Natural-order Cholesk may have a much costlier normal factor than
      // the existing sparse restoration. Preserve a bounded cheap attempt:
      // one existing stagnation window per phase and 10% of the remaining
      // wall budget. This is backend/ordering policy, never a case-name rule.
      const Vec square_origin=square_candidate&&w.me==w.n?w.x:Vec{};
      const bool defer_square_normal=square_candidate&&st.restoration_enabled&&
          st.linear_backend=="cholesk"&&!st.linear_metis;
      auto square_direction=[&]() {
        // A feasibility normal step solves J*dx=-h. Its auxiliary normal
        // multiplier is not an optimization multiplier. Source stabilization
        // must not change that subproblem to J*dx-rho*d_lambda=-h.
        const double saved_rho=w.equality_stabilization.rho;
        const bool exact_normal=original_constant_objective&&w.equality_stabilization.enabled;
        auto restore_normal_scope=[&]() {
          if(exact_normal){w.equality_stabilization.rho=saved_rho;w.update_matrix();}
        };
        Direction d;
        try {
          if(exact_normal){w.equality_stabilization.rho=0.0;w.update_matrix();}
          square_factor_failed=!w.factorize(1.0);
          if(square_factor_failed)throw std::runtime_error(w.message);
          d=w.direction(Vec{},w.rd,w.h,Vec{});
          restore_normal_scope();
        } catch(...) {restore_normal_scope();throw;}
        d.lambda.assign(w.me,0.0);
        long double error=0;
        for(int e=0;e<w.me;++e) {
          const auto& q=w.equalities[e];long double residual=w.h[e];
          for(std::size_t k=0;k<q.support.size();++k)
            residual+=q.gradient_extended[k]*d.x[q.support[k]];
          error=std::max(error,std::abs(residual));
        }
        d.residual=double(error)/std::max(1.0,inf_norm(w.h));
        if(!std::isfinite(d.residual)||d.residual>st.kkt_backward_error_reject)
          throw std::runtime_error("square feasibility original linear residual rejected");
        if(level)std::cout<<"[MAJORANT-SQUARE-FEASIBILITY] usable=1 normal_metric="
            <<w.regularization<<" backend="<<w.current_factor->name()
            <<" original_linear_residual="<<d.residual<<" square_feasibility_v1=1"<<std::endl;
        return d;
      };
      if(square_candidate&&!defer_square_normal&&seconds(start)<st.time_limit_seconds&&
          dca_outer_metrics(w).primal>st.feasibility_tolerance) {
        w.update_matrix();
        try {
          square_first=square_direction();square_feasibility=true;square_prepared=true;
        } catch(const std::runtime_error& error) {
          if(level)std::cout<<"[MAJORANT-SQUARE-FEASIBILITY] usable=0 reason="
              <<error.what()<<" fallback=historical-startup"<<std::endl;
        }
      }
      bool equality_start_restored=false;
      if(level&&w.mi==0&&w.me>0)std::cout<<"[MAJORANT-STARTUP-POLICY]"
          <<" objective_free_curvature="<<free_objective_curvature
          <<" equality_zero_hessian_restore="<<(equality_startup&&!square_feasibility)
          <<" square_normal_selected="<<square_feasibility
          <<" structural_startup_v1=1"<<std::endl;
      if(level&&problem.has_signed_square())std::cout<<"[NATIVE-SIGNED-SQUARE-POLICY] startup_restoration="<<st.restoration_enabled
          <<" conditional_restoration="<<st.majorant_initial_restoration
          <<" original_filter=1 original_lifted_KKT=1"<<std::endl;
      if(level&&w.me>0&&w.mi>0)std::cout<<"[MAJORANT-MIXED-STARTUP]"
          <<" objective_free_curvature="<<free_objective_curvature
          <<" infeasible_nonlinear_equality_restore="<<mixed_affine_equality_startup
          <<" restoration_enabled="<<st.restoration_enabled<<" mixed_startup_v1=1"<<std::endl;
      if(!square_feasibility && (equality_startup || (mixed_affine_equality_startup&&!central_first) || problem.has_signed_square()) && w.me>0 && st.restoration_enabled &&
          dca_outer_metrics(w).primal>st.feasibility_tolerance) {
        CostScope preparation(costs,NewtonCost::Preparation);
        Settings recovery=st;
        recovery.restoration_deadline_seconds=std::min(recovery.restoration_deadline_seconds,
            std::chrono::duration<double>(start.time_since_epoch()).count()+st.time_limit_seconds);
        const bool bounded_startup_probe=w.auto_startup.bounded_restoration_probe&&
            mixed_affine_equality_startup&&!central_first;
        if(bounded_startup_probe) {
          const double elapsed=seconds(start);
          const double allowance=detail::startup_restoration_probe_seconds(
              std::max(0.0,st.time_limit_seconds-elapsed));
          recovery.restoration_deadline_seconds=std::min(recovery.restoration_deadline_seconds,
              std::chrono::duration<double>(start.time_since_epoch()).count()+elapsed+allowance);
          if(level)std::cout<<"[AUTO-STARTUP-RESTORATION-PROBE] seconds_cap="<<allowance
              <<" shared_global_budget=1 blind_violated_row=0"
              <<" success_keeps_historical_startup=1"<<std::endl;
        }
        if(defer_square_normal) {
          recovery.restoration_max_iterations=std::min(recovery.restoration_max_iterations,
              std::max(1,recovery.restoration_stagnation_iterations));
          const double now=seconds(start);
          recovery.restoration_deadline_seconds=std::min(recovery.restoration_deadline_seconds,
              std::chrono::duration<double>(start.time_since_epoch()).count()+now+
              0.1*std::max(0.0,st.time_limit_seconds-now));
          if(level)std::cout<<"[MAJORANT-SQUARE-POLICY] normal_deferred=1"
              <<" reason=natural-order-cholesk restoration_iteration_cap="
              <<recovery.restoration_max_iterations<<" wall_fraction_cap=0.1"<<std::endl;
        }
        recovery.restoration_max_restarts=0;
        recovery.restoration_minimax_enabled=false;
        NativeRestorationWork mixed_work;
        NativeRestorationTrace startup_work;
        auto rr=run_counted_native_restoration(startup_work,[&]() {
          return mixed_affine_equality_startup?
              restore_mixed_startup(problem,problem.x0,recovery,&mixed_work):
              FeasibilityRestoration(recovery).solve(problem,problem.x0);
        },[&](const NativeRestorationTrace& work) {
          dca_merge_native_restoration_trace(result,work,"startup",level);
        });
        result.restoration_total_seconds+=rr.total_seconds;
        result.restoration_iterations=rr.iterations;
        result.restoration_restarts=rr.restarts;
        result.restoration_linear_solves=rr.linear_solves;
        result.restoration_row_space_solves=rr.row_space_solves;
        result.restoration_exact_hessian_solves=rr.exact_hessian_solves;
        result.restoration_harmonic_factorizations=rr.harmonic_factorizations;
        result.restoration_harmonic_feasibility=rr.harmonic_feasibility;
        result.restoration_tangent_activation_components=rr.tangent_activation_components;
        result.restoration_tangent_activation_feasibility=rr.tangent_activation_feasibility;
        result.restoration_tangent_activation_step_inf=rr.tangent_activation_step_inf;
        result.restoration_initial_feasibility=rr.initial_feasibility;
        result.restoration_final_feasibility=rr.final_feasibility;
        // Preserve a same-execution native point before the private gate may
        // reject it. Capture checks every owning ORIGINAL row independently.
        if(detail::native_anchor_capture_wanted(rr.final_feasibility,st.original_private_phase)) {
          CostScope capture(costs,NewtonCost::Certificate);
          detail::native_anchor_observe(problem,st.rank_presolve_context,rr.x,
              rr.final_feasibility,st.original_private_phase,detail::NativeAnchorOrigin::RestorationResult);
        }
        result.restoration_harmonic_attempts=rr.harmonic_attempts;
        result.restoration_harmonic_acceptances=rr.harmonic_acceptances;
        result.restoration_tangent_activation_attempts=rr.tangent_activation_attempts;
        result.restoration_tangent_activation_acceptances=rr.tangent_activation_acceptances;
        const auto* original_domain=st.rank_presolve_context;
        const bool original_bounds_ok=!original_domain||!original_domain->has_omitted_equality_bounds()||
            (rr.x.size()==std::size_t(problem.n)&&finite(rr.x)&&
             original_domain->omitted_bound_residual(rr.x).maximum<=st.feasibility_tolerance);
        auto startup_original_metrics=[&](const Vec& point) {
          if(!original_domain)return detail::native_affine_original_residuals(problem,point);
          Result snapshot;snapshot.x=point;
          snapshot.certificate_equality_multipliers.assign(problem.equalities.size(),0);
          snapshot.certificate_inequality_multipliers.assign(problem.inequalities.size(),0);
          snapshot.certificate_lower_multipliers.assign(problem.n,0);
          snapshot.certificate_upper_multipliers.assign(problem.n,0);
          try {original_domain->lift_checkpoint(snapshot);}
          catch(...) {return std::make_pair((long double)INFINITY,(long double)INFINITY);}
          return detail::native_affine_original_residuals(original_domain->original(),snapshot.x);
        };
        bool startup_domain_ok=rr.x.size()==std::size_t(problem.n)&&finite(rr.x);
        if(startup_domain_ok)for(int j=0;j<problem.n;++j)
          if(rr.x[j]<problem.lb[j]||rr.x[j]>problem.ub[j]){startup_domain_ok=false;break;}
        const auto startup_before=startup_original_metrics(problem.x0);
        const auto startup_after=startup_original_metrics(rr.x);
        const bool mixed_partial=mixed_affine_equality_startup&&original_bounds_ok&&startup_domain_ok&&
            rr.iterations<=recovery.restoration_max_iterations&&
            detail::native_affine_partial_progress(rr.success,true,startup_after.first,startup_after.second,
                startup_before.first,startup_before.second,seconds(start),st.time_limit_seconds);
        const bool restoration_accepted=original_bounds_ok&&
            (rr.success || mixed_partial || (problem.has_signed_square() && rr.final_feasibility<0.5*rr.initial_feasibility));
        if(mixed_partial&&level)std::cout<<"[MIXED-STARTUP-PARTIAL-PROGRESS] accepted=1"
          <<" original_max="<<double(startup_before.first)<<"->"<<double(startup_after.first)
          <<" original_l1="<<double(startup_before.second)<<"->"<<double(startup_after.second)
          <<" model_attempts="<<rr.iterations<<" original_KKT_certificate=0"<<std::endl;
        if(level)std::cout<<"[MAJORANT-RESTORATION] trigger="
            <<(problem.has_signed_square()?"native-signed-square-startup":
                (mixed_affine_equality_startup?"affine-infeasible-nonlinear-equalities":"equality-only-startup"))
            <<" success="<<rr.success<<" accepted="<<restoration_accepted
            <<" original_masked_bounds_ok="<<original_bounds_ok
            <<" partial="<<(restoration_accepted&&!rr.success)
            <<" iterations="<<rr.iterations<<" seconds="<<rr.total_seconds
            <<" primal="<<rr.initial_feasibility<<"->"<<rr.final_feasibility
            <<" tangent="<<rr.tangent_activation_acceptances<<std::endl;
        if(restoration_accepted) {
          for(int j=0;j<w.n;++j)w.x[j]=rr.x[w.free[j]];
          w.evaluate();
          for(int i=0;i<w.mi;++i)if(w.active[i]){w.s[i]=std::max(1e-2,-w.inequalities[i].value);w.z[i]=std::min(1.,1./w.s[i]);}
          w.evaluate();equality_start_restored=true;
        }
        if(bounded_startup_probe&&!restoration_accepted&&seconds(start)<st.time_limit_seconds) {
          // No private Phase-I endpoint was imported. Reproduce the native
          // central-first initialization at the SAME frozen primal start,
          // before constructing an outer filter or consuming an outer step.
          const double central_mu=w.auto_startup.central_fallback_mu;
          w.x=auto_frozen_x;w.evaluate();
          for(int i=0;i<w.mi;++i) {
            w.s[i]=std::max(std::max(0.01,central_mu),-w.inequalities[i].value);
            if(w.active[i])w.z[i]=std::min(1.0,central_mu/w.s[i]);
          }
          std::fill(w.lambda.begin(),w.lambda.end(),0.0);
          w.evaluate();result.lazy_newton.initial_barrier=central_mu;
          central_first=true;w.auto_startup.central_first=true;
          w.auto_startup.initial_mu=central_mu;
          w.auto_startup.reason="bounded-cold-restoration-incomplete";
          if(st.majorant_initial_dual_balance)w.balance_cold_duals();
          auto_initial_mu=result.lazy_newton.initial_barrier;
          w.set_barrier_mu(auto_initial_mu);
          if(level)std::cout<<"[AUTO-STARTUP-RESTORATION-TRANSFER] accepted=1"
              <<" selected=central-first reason=bounded-cold-restoration-incomplete"
              <<" frozen_start_restored=1 private_endpoint_imported=0"
              <<" mu="<<auto_initial_mu<<" shared_global_budget=1"
              <<" original_filter_required=1 final_KKT_certificate=0"<<std::endl;
        }
        if(defer_square_normal&&!rr.success&&seconds(start)<st.time_limit_seconds) {
          // Do not import a failed least-squares stationary point. A smaller
          // residual alone does not ensure a useful Newton basin. The cold
          // point remains unchanged unless restoration has truly succeeded.
          w.update_matrix();
          try {
            square_first=square_direction();square_feasibility=true;square_prepared=true;
            if(level)std::cout<<"[MAJORANT-SQUARE-TRANSFER] accepted=1"
                <<" partial_restoration=0 original_start_preserved=1"
                <<" square_normal_selected=1"<<std::endl;
          } catch(const std::runtime_error& error) {
            if(level)std::cout<<"[MAJORANT-SQUARE-FEASIBILITY] usable=0 reason="
                <<error.what()<<" fallback=historical-startup"<<std::endl;
          }
        }
      }
      if(st.restoration_enabled&&!equality_start_restored&&
          detail::restoration_step_available(st.restoration_shared_iteration_budget,
              st.restoration_max_iterations,result.restoration_iterations)&&
          dca_outer_metrics(w).primal>st.feasibility_tolerance) {
        CostScope preparation(costs,NewtonCost::Preparation);
        const auto epigraph=detail::restore_positive_square_epigraph(problem,st,
            std::min(st.restoration_deadline_seconds,
                std::chrono::duration<double>(start.time_since_epoch()).count()+st.time_limit_seconds));
        result.restoration_total_seconds+=epigraph.seconds;
        result.restoration_linear_solves+=epigraph.solves;
        result.lazy_newton.factor_attempts+=epigraph.factors;
        result.lazy_newton.linear_solves+=epigraph.solves;
        if(epigraph.attempted)++result.restoration_iterations;
        if(epigraph.eligible&&level)std::cout<<"[POSITIVE-SQUARE-EPIGRAPH-RESTORATION] eligible=1"
            <<" attempted="<<epigraph.attempted<<" accepted="<<epigraph.accepted
            <<" variable="<<epigraph.epigraph<<" factors="<<epigraph.factors
            <<" solves="<<epigraph.solves<<" primal="<<epigraph.before<<"->"<<epigraph.after
            <<" seconds="<<epigraph.seconds<<" original_constants=exact"
            <<" frozen_input_unchanged=1 original_KKT_required=1 reason="<<epigraph.reason<<std::endl;
        if(epigraph.accepted) {
          for(int j=0;j<w.n;++j)w.x[j]=epigraph.x[w.free[j]];
          w.evaluate();
          for(int i=0;i<w.mi;++i)if(w.active[i]) {
            w.s[i]=std::max(1e-2,-w.inequalities[i].value);
            w.z[i]=std::min(1.,1./w.s[i]);
          }
          w.evaluate();equality_start_restored=true;
          result.restoration_initial_feasibility=epigraph.before;
          result.restoration_final_feasibility=epigraph.after;
        }
      }
      original_cache_current=true;
      DcaOuterMetrics current=dca_outer_metrics(w);
      if(w.equality_stabilization.enabled) {
        w.equality_stabilization.reset(w.lambda,w.true_source_constraint_residual(),w.z);
        if(level)std::cout<<"[EQUALITY-STABILIZATION] event=configured"
            <<" variant="<<w.source_stabilization_variant()
            <<" algorithmic_rho="<<w.equality_stabilization.rho
            <<" rho_floor="<<detail::MajorantEqualityStabilization::minimum_rho
            <<" all_equality_rows_retained="<<(w.mr==w.me&&w.nr==w.n)
            <<" equality_pivot_condensation=0 matched_normal_recovery=0"
            <<" physical_source_C_reference="<<w.true_source_constraint_residual()
            <<" source_IQ_slack_equations_stabilized=1 physical_bound_rho=0"
            <<" numerical_backend_shifts_separate=1 original_filter="<<(!gr_globalization)<<" original_KKT_unchanged=1"<<std::endl;
      }
      constexpr double barrier_tolerance_factor=10.0;
      const double barrier_floor=w.objective.scale*
          std::min(st.complementarity_tolerance,st.stationarity_tolerance)/
          (barrier_tolerance_factor+1.0);
      // Workspace initializes x,s,z,lambda once. Use its actual positive
      // complementarity mean, already in internal units, without any reset.
      double mu=current.active_count?std::max(barrier_floor,current.product_mean):0;
      detail::TrialSocArcBarrierEvent soc_arc_barrier_event;
      bool soc_defect_epoch=false,soc_defect_geometry_inspected=false;
      detail::SocDefectGeometry soc_defect_geometry;
      if(!(mu>=0)||!std::isfinite(mu))throw std::runtime_error("nonfinite initial barrier parameter");
      auto sync_outer_barrier=[&]() {
        if(w.source_primal_barrier_mode)w.set_barrier_mu(mu);
      };
      auto update_outer_matrix=[&]() {
        sync_outer_barrier();w.update_matrix();
      };
      sync_outer_barrier();
      result.lazy_newton.initial_barrier=mu;
      const double zero_barrier_forcing_reference=w.mi==0&&!free_objective_curvature?
          w.stabilized_barrier_residual(0.0):0.0;
      auto_initial_mu=mu;
      stats.majorant_barrier_mu=mu;
      stats.majorant_barrier_residual=current.residual(mu);
      // This is an initial auxiliary-Newton proximal term, not objective
      // convexification and not an invisible backend perturbation. It enters
      // the direction audit and follows the existing accepted-step decay.
      double rho_memory=equality_start_restored?1.0:0;
      double gr_kappa=1.0;
      detail::HistoricalEqualityWatch historical_watch;
      detail::GrFixedCenteringPolicy fixed_centering(st.gr_fixed_centering_steps);
      std::unique_ptr<HybridInnerAnchor> fixed_centering_anchor;
      detail::MajorantEqualityStabilization fixed_centering_stabilization;
      double fixed_centering_mu=0,fixed_centering_proximal=0,fixed_centering_kappa=0;
      double fixed_centering_initial_residual=INFINITY;
      bool fixed_centering_rolled_back=false;
      std::string study_last_margin_trace_key;
      std::unique_ptr<DcaMajorantBarrierFilter> filter;
      bool restoration_attempted=false;
      bool failed_linearization_recovery_attempted=false;
      int hybrid_phase1_stages=0;
      double hybrid_last_phase1_primal=INFINITY;
      detail::SharedPhase1RecoveryBudget late_phase1_budget(std::min(st.max_outer_iterations,st.restoration_max_iterations));
      // A structural retry inherits the remaining allowance. Its startup
      // restoration is work from that SAME allowance, before late Phase I.
      if(st.restoration_shared_iteration_budget)
        late_phase1_budget.charge(result.restoration_iterations);
      int hybrid_accepted_primal_generation=0;
      bool recovered_state_requires_audit=false;
      int filter_proximal_retries=0,hybrid_trial_retries=0;
      int direction_proximal_retries=0;
      bool direction_centering_used=false;
      bool lower_proximal_used=false,lower_proximal_pending=false;
      double lower_proximal_previous_seed=0,lower_proximal_previous_actual=0;
      std::unique_ptr<HybridInnerAnchor> lower_proximal_anchor;
      bool hybrid_skip_trial_once=false;
      bool filter_recenter_used=false;int filter_recenter_count=0;
      double filter_recenter_primal_reference=INFINITY;
      double initial_original_comp=current.comp;
      int dual_stalls=0,last_lambda_attempt=-8;
      bool lambda_rho_pending=false;
      detail::DualRecoveryHistory dual_recovery_history;
      detail::SmallStepSocPolicy small_step_soc(st.majorant_soc_small_step_patience,
          st.majorant_soc_small_step_threshold);
      detail::NonlinearStepSocPolicy nonlinear_step_soc(st.majorant_soc_small_step_patience,
          st.majorant_soc_small_step_threshold);
      const bool adaptive_globalization=st.majorant_adaptive_globalization;
      const bool deferred_r3_soc=models->study_r3_controller()&&
          st.majorant_conditional_retraction&&st.majorant_soc_small_step_patience>0;
      if(level&&deferred_r3_soc)std::cout<<"[MAJORANT-SOC-LATCH] event=configured patience="
          <<st.majorant_soc_small_step_patience<<" threshold="<<st.majorant_soc_small_step_threshold
          <<" enabled=0 accepted_steps_only=1 original_filter_unchanged=1"<<std::endl;
      Vec dual_window;
      auto fixed_centering_dual_history=dual_recovery_history;
      auto fixed_centering_small_step_soc=small_step_soc;
      auto fixed_centering_nonlinear_step_soc=nonlinear_step_soc;
      Vec fixed_centering_dual_window;
      int fixed_centering_dual_stalls=0,fixed_centering_last_lambda_attempt=0;
      bool fixed_centering_direction_centering_used=false,fixed_centering_lower_proximal_used=false;
      auto trace_barrier_components=[&](const char* source) {
        if(!gr_globalization||!st.gr_residual_components||level<2)return;
        const Vec eh=w.equality_newton_residual(),eg=w.primal_newton_residual();
        double regularized_gs=0;
        for(int i=0;i<w.mi;++i)if(w.active[i])regularized_gs=std::max(regularized_gs,std::abs(eg[i]));
        std::cout<<std::setprecision(17)<<"[GR-BARRIER-COMPONENTS] iter="<<result.outer_iterations
            <<" source="<<source<<" mu="<<mu<<" algorithmic_rho="<<w.equality_stabilization.rho
            <<" original_h="<<current.equality_internal_inf
            <<" original_g_plus_s="<<current.slack_equation_internal_inf
            <<" original_dual_internal="<<current.dual_internal_inf
            <<" original_dual_physical="<<current.dual
            <<" source_product_minus_mu="<<current.source_centering_residual(mu)
            <<" bound_product_minus_mu="<<current.bound_centering_residual(mu)
            <<" source_eta_minus_mu_over_s="<<current.source_stationarity_residual(mu)
            <<" original_barrier_residual="<<current.residual(mu)
            <<" regularized_h="<<inf_norm(eh)<<" regularized_g_plus_s="<<regularized_gs
            <<" regularized_barrier_residual="<<w.stabilized_barrier_residual(mu)
            <<" original_primal_physical="<<current.primal<<" original_compl_physical="<<current.comp
            <<" continuation_threshold="<<10*mu<<" fixed_centering="<<fixed_centering.active()
            <<" original_KKT_tolerances_unchanged=1"<<std::endl;
      };
      auto rollback_fixed_centering=[&](const char* reason) {
        if(!fixed_centering_anchor)return false;
        const double after=current.residual(mu);
        models->discard();models->study_rollback_trial();
        mu=fixed_centering_mu;w.equality_stabilization=fixed_centering_stabilization;
        rho_memory=fixed_centering_proximal;gr_kappa=fixed_centering_kappa;sync_outer_barrier();
        fixed_centering_anchor->restore(w);current=dca_outer_metrics(w);
        fixed_centering_anchor.reset();fixed_centering.abort(result.lazy_newton.accepted_steps);
        fixed_centering_rolled_back=true;
        dual_recovery_history=fixed_centering_dual_history;
        small_step_soc=fixed_centering_small_step_soc;nonlinear_step_soc=fixed_centering_nonlinear_step_soc;
        dual_window=fixed_centering_dual_window;dual_stalls=fixed_centering_dual_stalls;
        last_lambda_attempt=fixed_centering_last_lambda_attempt;
        direction_centering_used=fixed_centering_direction_centering_used;
        lower_proximal_used=fixed_centering_lower_proximal_used;
        original_cache_current=true;audited_current=false;recovered_state_requires_audit=true;
        filter_proximal_retries=0;direction_proximal_retries=0;
        lower_proximal_pending=false;lower_proximal_anchor.reset();
        stats.majorant_barrier_mu=mu;stats.majorant_barrier_residual=current.residual(mu);
        if(level)std::cout<<std::setprecision(17)<<"[GR-FIXED-CENTERING] event=rollback reason="<<reason
            <<" iter="<<result.outer_iterations<<" mu="<<mu<<" algorithmic_rho="<<w.equality_stabilization.rho
            <<" staged_barrier_residual="<<after<<" restored_barrier_residual="<<current.residual(mu)
            <<" attempt="<<fixed_centering.attempts<<" work_and_time_charged=1"
            <<" original_GR_required=1 entry_state_restored=1"<<std::endl;
        return true;
      };
      detail::GrFixedCenteringScope fixed_centering_scope;
      fixed_centering_scope.rollback=[&]() {
        if(!fixed_centering_anchor)return;
        try{rollback_fixed_centering("phase-scope-exit");}
        catch(const std::exception& e){result.status=Status::NumericalError;
          result.message=std::string("fixed-centering rollback failed: ")+e.what();}
      };
      auto rollback_lower_proximal=[&](const char* reason) {
        if(!lower_proximal_pending)return false;
        if(level)std::cout<<"[LOWER-PROXIMAL-PROBE] event=rollback reason="<<reason
            <<" proposed_memory="<<rho_memory<<" actual="<<w.regularization
            <<" restored_seed="<<lower_proximal_previous_seed
            <<" previous_actual="<<lower_proximal_previous_actual
            <<" primal_dual_state_unchanged=1 filter_reset=0 correction_refinement=0"<<std::endl;
        const bool restored=detail::restore_lower_proximal_probe(lower_proximal_pending,
            lower_proximal_previous_seed,rho_memory);
        models->discard();models->study_rollback_trial();
        if(lower_proximal_anchor)lower_proximal_anchor->restore(w);else w.evaluate();
        lower_proximal_anchor.reset();original_cache_current=true;
        if(models->study_hybrid_enabled())hybrid_skip_trial_once=true;
        return restored;
      };
      auto continue_barrier_after_original_acceptance=[&](const char* source) {
        // A restored anchor passed the SAME original filter as a DCA anchor.
        // It must not bypass the historical barrier-continuation policy and
        // retain a cold large mu while its interior slacks approach zero.
        // Use one bounded update, with the same spectral residual gate, floor
        // and filter rebase/reset policy. No failed direction is accepted.
        if(result.outer_iterations>=st.max_outer_iterations||seconds(start)>=st.time_limit_seconds)
          return;
        trace_barrier_components(source);
        if(fixed_centering.active()) {
          const bool converged=detail::gr_fixed_centering_converged(current.residual(mu),
              w.stabilized_barrier_residual(mu),mu,current.primal,current.dual,
              st.feasibility_tolerance,st.dual_infeasibility_tolerance);
          const bool finished=fixed_centering.accepted(result.lazy_newton.accepted_steps,converged);
          const bool parameters_fixed=mu==fixed_centering_mu&&
              w.equality_stabilization.rho==fixed_centering_stabilization.rho&&
              w.equality_stabilization.multiplier_center==fixed_centering_stabilization.multiplier_center&&
              w.equality_stabilization.source_inequality_center==fixed_centering_stabilization.source_inequality_center;
          if(!parameters_fixed)throw std::logic_error("fixed centering continuation parameters changed");
          if(level)std::cout<<std::setprecision(17)<<"[GR-FIXED-CENTERING] event="<<(finished?"phase-end":"accepted")
              <<" iter="<<result.outer_iterations<<" remaining="<<fixed_centering.remaining
              <<" mu="<<mu<<" algorithmic_rho="<<w.equality_stabilization.rho
              <<" barrier_residual_before="<<fixed_centering_initial_residual
              <<" barrier_residual_after="<<current.residual(mu)
              <<" original_primal="<<current.primal<<" original_dual="<<current.dual
              <<" original_compl="<<current.comp<<" parameters_fixed=1 original_GR_accepted=1"
              <<" finish_reason="<<(converged?"both-barrier-gates":(finished?"step-cap":"pending"))
              <<" dual_stationarity_RHS=zero other_Newton_rows=unchanged work_and_time_charged=1"<<std::endl;
          bool committed=false;
          if(finished) {
            // N is a cap, not a requirement to take flat extra steps after
            // BOTH original and regularized barrier gates have converged.
            // All staged steps already passed the unchanged original GR.
            if((converged||current.residual(mu)<=0.9*fixed_centering_initial_residual)&&
                current.primal<=st.feasibility_tolerance&&current.dual<=st.dual_infeasibility_tolerance) {
              fixed_centering_anchor.reset();
              committed=true;
              if(level)std::cout<<"[GR-FIXED-CENTERING] event=commit iter="<<result.outer_iterations
                  <<" commit_criterion="<<(converged?"both-barrier-gates":"residual-contraction-0.9")
                  <<" strict_physical_blocks_preserved=1 original_GR_unchanged=1"<<std::endl;
            } else rollback_fixed_centering("phase-contraction-or-physical-gate");
          }
          stats.majorant_barrier_mu=mu;stats.majorant_barrier_residual=current.residual(mu);
          // A converged committed phase now releases its fixed parameters
          // through the EXISTING epoch/mu gates below. It need not risk one
          // more old-mu Newton step merely to reach this continuation call.
          if(!(finished&&converged&&committed))return;
        }
        if(w.equality_stabilization.enabled) {
          const double stabilized=w.stabilized_barrier_residual(mu);
          const double old_rho=w.equality_stabilization.rho;
          const Vec old_center=w.equality_stabilization.multiplier_center;
          const Vec old_z_center=w.equality_stabilization.source_inequality_center;
          const double zero_barrier_epoch_tolerance=w.mi==0?
              detail::zero_barrier_epoch_forcing(zero_barrier_forcing_reference,
                  w.equality_stabilization.epochs,
                  std::min(st.feasibility_tolerance,st.dual_infeasibility_tolerance)):0.0;
          const bool updated=w.equality_stabilization.update_epoch(w.lambda,w.true_source_constraint_residual(),
              stabilized,mu,result.outer_iterations,&w.z,zero_barrier_epoch_tolerance);
          bool source_center_changed=false;
          for(int i=0;i<w.mi;++i)if(w.inequality_kind[i]==0&&
              old_z_center[i]!=w.equality_stabilization.source_inequality_center[i])source_center_changed=true;
          if(level>=2||updated)std::cout<<std::setprecision(17)
              <<"[EQUALITY-STABILIZATION] event="<<(updated?"epoch":"hold")
              <<" iteration="<<result.outer_iterations<<" source="<<source
              <<" regularized_barrier_residual="<<stabilized
              <<" threshold="<<(mu>0?10*mu:zero_barrier_epoch_tolerance)
              <<" zero_barrier_inner_KKT_tolerance="<<zero_barrier_epoch_tolerance
              <<" true_source_C="<<w.true_source_constraint_residual()
              <<" algorithmic_rho="<<old_rho<<"->"<<w.equality_stabilization.rho
              <<" multiplier_center_changed="<<(old_center!=w.equality_stabilization.multiplier_center)
              <<" source_IQ_center_changed="<<source_center_changed
              <<" epochs="<<w.equality_stabilization.epochs
              <<" positive_barrier_gate_unchanged=1 zero_barrier_forcing=geometric-terminal-floor"<<std::endl;
        }
        const bool spectral_gate=(models->positive_spectral_count>0||st.majorant_complete_barrier_gate)&&
            (current.primal>st.feasibility_tolerance||current.comp>st.complementarity_tolerance);
        const bool complete_gate=adaptive_globalization||spectral_gate||w.equality_stabilization.enabled;
        const double progress=complete_gate?current.residual(mu):current.primal_centering_residual(mu);
        const double next=current.active_count?
            detail::majorant_barrier_continuation(mu,barrier_floor,progress):mu;
        if(st.majorant_trial_soc_arc)
          soc_arc_barrier_event.observe(result.lazy_newton.accepted_steps,mu,next);
        if(st.majorant_soc_defect_control&&next>0&&next<mu) {
          soc_defect_epoch=true;
          if(level>=2)std::cout<<"[MAJORANT-SOC-DEFECT] event=barrier-epoch-available"
              <<" iter="<<result.outer_iterations<<" generation="<<result.lazy_newton.accepted_steps
              <<" previous_mu="<<mu<<" target_mu="<<next<<" source_affine_targets=1"<<std::endl;
        }
        if(adaptive_globalization&&level>=2) {
          std::cout<<"[MAJORANT-MU-GATE] iter="<<result.outer_iterations
              <<" mu="<<mu<<" proposed_mu="<<next<<" complete_gate="<<complete_gate
              <<" residual_internal="<<current.residual(mu)
              <<" primal_centering_internal="<<current.primal_centering_residual(mu)
              <<" h_internal="<<current.equality_internal_inf
              <<" g_plus_s_internal="<<current.slack_equation_internal_inf
              <<" sz_minus_mu_internal="<<(current.active_count?std::max(std::abs(current.product_min-mu),std::abs(current.product_max-mu)):0)
              <<" source_slack_gradient_internal="<<current.source_stationarity_residual(mu)
              <<" source_multiplier_sign_physical="<<current.source_multiplier_sign_inf
              <<" dual_original="<<current.dual<<" objective_scale="<<w.objective.scale
              <<" threshold="<<10*mu
              <<" decision="<<(next<mu?"decrease":(mu<=barrier_floor?"floor":"hold"))
              <<" source="<<source<<std::endl;
        }
        if(next<mu) {
          // A strictly lower barrier defines a new centrality equation.
          // Permit one auxiliary recenter for that new epoch after failure;
          // the same-mu one-shot guard and global outer/time budgets persist.
          detail::filter_recenter_new_barrier_epoch(filter_recenter_used,mu,next);
          mu=next;sync_outer_barrier();
          if(models->study_hybrid_controller_enabled())filter->rebase(mu);
          else {filter->reset();++stats.majorant_barrier_filter_resets;}
          ++stats.majorant_mu_updates;
          if(level>=2) {
            CostScope display(costs,NewtonCost::Diagnostics);
            std::cout<<"[MAJORANT-MU] iter="<<result.outer_iterations<<" mu="<<mu
                <<" filter_reset="<<(!models->study_hybrid_controller_enabled())
                <<" outer_update=1 source="<<source<<"\n";
          }
        }
        stats.majorant_barrier_mu=mu;stats.majorant_barrier_residual=current.residual(mu);
      };
      auto retry_direction_centering=[&](const std::string& failure,int outer) {
        if(gr_globalization)return false;
        if(direction_centering_used||!detail::failed_direction_proximal_retry_eligible(failure,
            0,false,st.max_outer_iterations-outer,seconds(start),st.time_limit_seconds))return false;
        direction_centering_used=true;
        models->discard();models->study_rollback_trial();w.evaluate();original_cache_current=true;
        const HybridInnerAnchor accepted(w);
        const auto before=dca_outer_metrics(w);current=before;
        Vec values(w.mi),central_s,central_z;long double log_gain=0;
        for(int i=0;i<w.mi;++i)values[i]=w.inequalities[i].value;
        if(!detail::direction_centering_candidate(values,w.active,w.s,w.z,mu,
            central_s,central_z,log_gain))return false;
        bool keep=false;const char* filter_reason="nonfinite-candidate";
        double theta=INFINITY,phi_change=INFINITY;
        try {
          w.s.swap(central_s);w.z.swap(central_z);w.evaluate();
          const auto candidate=dca_outer_metrics(w);
          const auto from=before.filter_point(mu);auto to=candidate.filter_point(mu);
          // The primal point and objective are BITWISE unchanged. Evaluate
          // the barrier difference from the geometric slack path directly.
          to.phi_change=static_cast<double>(-static_cast<long double>(mu)*log_gain);
          theta=to.theta;phi_change=to.phi_change;
          if(!filter)filter=std::make_unique<DcaMajorantBarrierFilter>(from.theta,
              models->study_hybrid_controller_enabled());
          const auto verdict=filter->diagnose(from,to,to.phi_change,1,st.line_search_armijo);
          filter_reason=verdict.reason;
          keep=verdict.accepted&&seconds(start)<st.time_limit_seconds&&w.x==accepted.x&&w.lambda==accepted.lambda;
          if(keep) {
            if(verdict.augment_history(st.majorant_filter_history_fix)&&from.theta>0)filter->insert(from);
            current=candidate;audited_current=false;recovered_state_requires_audit=true;
            if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
            if(models->study_hybrid_enabled())hybrid_skip_trial_once=true;
          }
        }catch(const std::runtime_error&){keep=false;}
        if(!keep){accepted.restore(w);current=before;}
        original_cache_current=true;
        if(level)std::cout<<"[DIRECTION-CENTERING-RECOVERY] iter="<<outer<<" accepted="<<keep
            <<" budget=1-per-primal-anchor mu="<<mu<<" theta="<<before.barrier.theta<<"->"<<theta
            <<" phi_change="<<phi_change<<" original_filter="<<filter_reason
            <<" geometric_slack_path=1 exact_barrier_slope=1 x_bitwise_unchanged=1 lambda_unchanged=1"
            <<" filter_reset=0 correction_refinement=0 expanded_audit_unchanged=1"
            <<" accepted_metric_digest="<<models->study_metric_digest()<<std::endl;
        return keep;
      };
      auto recover_failed_linearization=[&](const char* reason,const std::string& failure) {
        if(gr_globalization)return false;
        // The ordinary factor/direction has already failed. A curved
        // infeasible equality cannot be repaired by accepting a perturbed
        // Newton residual. Give the existing nonlinear feasibility solver
        // one bounded opportunity to construct a usable linearization.
        const bool numerical_failure=w.numeric_pivot_failure(failure) ||
          failure.find("inertia mismatch")!=std::string::npos ||
          failure.find("expanded Newton residual rejected")!=std::string::npos ||
          failure.find("backward error too large")!=std::string::npos;
        // A failed original filter with nonlinear infeasibility also needs
        // a fresh usable linearization. Bound storage rather than arbitrarily
        // excluding mixed n>256, square systems, or equality-only n>1024.
        const bool filter_failure=std::string(reason)=="filter-exhausted";
        const bool inner_model_failure=models->study_hybrid_controller_enabled()&&
            std::string(reason)=="hybrid-inner-failure";
        const bool phase1=true; // Candidate: original-filtered partial recovery for every controller.
        if(phase1&&(late_phase1_budget.remaining()<=0||
            (late_phase1_budget.started&&std::chrono::duration<double>(Clock::now().time_since_epoch()).count()>=late_phase1_budget.deadline)))
          return false;
        if(!detail::restoration_step_available(st.restoration_shared_iteration_budget,
            st.restoration_max_iterations,result.restoration_iterations))return false;
        // Preserve the explicitly selected conditional-restoration budget and
        // rollback contract; it must not silently acquire a second recovery.
        if(filter_failure&&st.majorant_initial_restoration)return false;
        const auto recovery_budget=detail::failure_recovery_budget(problem,phase1);
        Settings sparse_admission_settings=st;
        sparse_admission_settings.restoration_failed_linearization_context=phase1;
        sparse_admission_settings.restoration_shared_iteration_budget=phase1;
        const auto sparse_budget=phase1&&!recovery_budget.allowed?
            sparse_restoration_budget(problem,sparse_admission_settings):SparseRestorationBudget{};
        const bool sparse_only=phase1&&!recovery_budget.allowed&&sparse_budget.allowed;
        // A rejected linearization can leave only AFFINE rows infeasible:
        // the nonlinear Phase-I progress may already have satisfied every
        // curved row. Excluding that anchor because its remaining violation
        // has zero Hessian prevents the SAME bounded recovery from finishing.
        // Admission therefore follows the original residual, independently of
        // its curvature. This does not accept a failed direction, replenish a
        // budget, or bypass the original filter below.
        if(!detail::failed_linearization_recovery_eligible(numerical_failure,
             filter_failure,inner_model_failure,st.restoration_enabled,
             failed_linearization_recovery_attempted,current.primal,st.feasibility_tolerance,
             st.majorant_recovery_gap_fix) ||
           (!recovery_budget.allowed&&!sparse_only) ||
           seconds(start)>=st.time_limit_seconds)return false;
        failed_linearization_recovery_attempted=true;
        models->discard();models->study_rollback_trial();w.evaluate();
        const HybridInnerAnchor accepted_state(w);
        const auto accepted_metrics=dca_outer_metrics(w);
        const double before=accepted_metrics.primal;
        const double stage_reference=phase1?std::min(before,hybrid_last_phase1_primal):before;
        double restoration_target=phase1?
            detail::phase1_feasibility_target(stage_reference,st.feasibility_tolerance):st.feasibility_tolerance;
        Vec initial=problem.x0;
        for(int j=0;j<problem.n;++j)if(std::isfinite(problem.lb[j])&&problem.lb[j]==problem.ub[j])initial[j]=problem.lb[j];
        for(int j=0;j<w.n;++j)initial[w.free[j]]=w.x[j];
        Settings recovery_settings=st;
        recovery_settings.restoration_failed_linearization_context=true;
        if(phase1) {
          recovery_settings.feasibility_tolerance=restoration_target;
          recovery_settings.restoration_shared_iteration_budget=true;
        }
        // A tiny disconnected feasibility set can have a strict infeasible
        // local minimum with POSITIVE merit Hessian: negative-curvature
        // detection alone cannot escape. Reuse the existing deterministic
        // coordinate recovery, only after filter failure, within permission
        // and deadline. An explicit zero restart budget remains zero.
        recovery_settings.restoration_max_restarts=filter_failure&&problem.n<=16
            ? std::min(8,std::max(0,st.restoration_max_restarts)) : 0;
        recovery_settings.restoration_restart_iterations=std::min(80,st.restoration_restart_iterations);
        recovery_settings.restoration_minimax_enabled=false;
        recovery_settings.restoration_max_iterations=std::min(240,st.restoration_max_iterations);
        const double remaining=st.time_limit_seconds-seconds(start);
        recovery_settings.restoration_deadline_seconds=std::min(st.restoration_deadline_seconds,
          std::chrono::duration<double>(Clock::now().time_since_epoch()).count()+std::min(60.0,remaining));
        if(phase1)recovery_settings.restoration_deadline_seconds=
            late_phase1_budget.bind_deadline(recovery_settings.restoration_deadline_seconds);
        const auto began=Clock::now();
        auto recovery_time_open=[&]() {
          const auto now=Clock::now();
          return detail::phase1_deadline_open(std::chrono::duration<double>(now-start).count(),
              st.time_limit_seconds,std::chrono::duration<double>(now.time_since_epoch()).count(),
              recovery_settings.restoration_deadline_seconds);
        };
        if(level&&phase1)std::cout<<"[HYBRID-PHASE-I-START] trigger="<<reason
          <<" original_primal="<<before<<" stage_reference_primal="<<stage_reference
          <<" target="<<restoration_target
          <<" target_kind=original-maximum-constraint-violation factor=0.1"
          <<" final_kkt_feasibility_tolerance="<<st.feasibility_tolerance
          <<" accepted_dca_step=0 shared_recovery_budget=1"
          <<" workspace_route="<<(sparse_only?"sparse-only-capped":"historical-dense-admitted")
          <<" dense_fallback="<<!sparse_only<<std::endl;
        int phase1_iteration_budget=phase1?late_phase1_budget.remaining():
            recovery_settings.restoration_max_iterations;
        if(st.restoration_shared_iteration_budget)
          phase1_iteration_budget=std::min(phase1_iteration_budget,
              detail::remaining_shared_restoration(st.restoration_max_iterations,
                  result.restoration_iterations));
        if(phase1||st.restoration_shared_iteration_budget)
          recovery_settings.restoration_max_iterations=phase1_iteration_budget;
        int phase1_iterations_used=0,private_phase1_stages=0;
        double private_before=before,candidate_primal=INFINITY,candidate_theta=INFINITY;
        Vec recovery_point=initial;RestorationResult rr;
        bool keep=false;const char* recovery_filter_reason="not-evaluated";
        for(;;) {
          if(phase1) {
            if(!recovery_time_open())break;
            recovery_settings.feasibility_tolerance=restoration_target;
            recovery_settings.restoration_private_precision_context=
                detail::phase1_private_precision_context(phase1,restoration_target,st.feasibility_tolerance);
            recovery_settings.restoration_max_iterations=phase1_iteration_budget-phase1_iterations_used;
            if(recovery_settings.restoration_max_iterations<=0)break;
          }
          NativeRestorationTrace phase1_work;
          rr=run_counted_native_restoration(phase1_work,[&]() {
            return sparse_only?
                FeasibilityRestoration(recovery_settings).solve_sparse_only(problem,recovery_point):
                (phase1?FeasibilityRestoration(recovery_settings).solve_elastic(problem,recovery_point):
                  FeasibilityRestoration(recovery_settings).solve(problem,recovery_point));
          },[&](const NativeRestorationTrace& work) {
            dca_merge_native_restoration_trace(result,work,"late-native-restoration",level);
          });
          phase1_iterations_used+=rr.iterations;
          if(phase1)late_phase1_budget.charge(rr.iterations);
          result.restoration_iterations+=rr.iterations;
          result.restoration_restarts+=rr.restarts;
          result.restoration_matrix_free_solves+=rr.matrix_free_solves;
          result.restoration_cg_iterations+=rr.cg_iterations;
          result.restoration_exact_workspace_growths+=rr.exact_workspace_growths;
          result.restoration_exact_matvecs+=rr.exact_matvecs;
          result.restoration_exact_parallel_row_dot_batches+=rr.exact_parallel_row_dot_batches;
          result.restoration_exact_parallel_teams+=rr.exact_parallel_teams;
          result.restoration_exact_sparse_row_dot_batches+=rr.exact_sparse_row_dot_batches;
          result.restoration_exact_sparse_row_dot_verifications+=rr.exact_sparse_row_dot_verifications;
          result.restoration_exact_sparse_row_dot_fallbacks+=rr.exact_sparse_row_dot_fallbacks;
          result.restoration_line_search_backtracks+=rr.line_search_backtracks;
          result.restoration_minimax_qp_solves+=rr.minimax_qp_solves;
          result.restoration_minimax_inner_iterations+=rr.minimax_inner_iterations;
          result.restoration_minimax_factorizations+=rr.minimax_factorizations;
          result.restoration_minimax_accepted_steps+=rr.minimax_accepted_steps;
          result.restoration_minimax_line_search_backtracks+=rr.minimax_line_search_backtracks;
          result.restoration_stagnation_events+=rr.stagnation_events;
          result.restoration_harmonic_attempts+=rr.harmonic_attempts;
          result.restoration_harmonic_acceptances+=rr.harmonic_acceptances;
          result.restoration_harmonic_factorizations+=rr.harmonic_factorizations;
          result.restoration_harmonic_seconds+=rr.harmonic_seconds;
          result.restoration_tangent_activation_seconds+=rr.tangent_activation_seconds;
          result.restoration_legacy_gate_seconds+=rr.legacy_gate_seconds;
          result.restoration_exact_seconds+=rr.exact_restoration_seconds;
          result.restoration_restart_seconds+=rr.restart_seconds;
          result.restoration_total_seconds+=rr.total_seconds;
          result.restoration_linear_solves+=rr.linear_solves;
          result.restoration_row_space_solves+=rr.row_space_solves;
          result.restoration_exact_hessian_solves+=rr.exact_hessian_solves;
          result.restoration_tangent_activation_attempts+=rr.tangent_activation_attempts;
          result.restoration_tangent_activation_acceptances+=rr.tangent_activation_acceptances;
          result.restoration_initial_feasibility=before;
          result.restoration_final_feasibility=rr.final_feasibility;
        // Preserve a same-execution native point before the private gate may
        // reject it. Capture checks every owning ORIGINAL row independently.
        if(detail::native_anchor_capture_wanted(rr.final_feasibility,st.original_private_phase)) {
          CostScope capture(costs,NewtonCost::Certificate);
          detail::native_anchor_observe(problem,st.rank_presolve_context,rr.x,
              rr.final_feasibility,st.original_private_phase,detail::NativeAnchorOrigin::RestorationResult);
        }
          // Partial Phase-I progress is only a candidate: every original row
          // and the original filter are independently checked below. The shared
          // recovery allowance is never replenished by accepting this anchor.
          keep=finite(rr.x)&&(phase1 ? rr.final_feasibility<stage_reference :
              (rr.success&&rr.final_feasibility<=restoration_target))&&
            (phase1?recovery_time_open():seconds(start)<st.time_limit_seconds);
          recovery_filter_reason="not-evaluated";
          candidate_primal=rr.final_feasibility;candidate_theta=INFINITY;
          bool refine_private_phase1=false;double next_private_target=0;
          if(phase1) {
            try {
              if(keep) {
                for(int j=0;j<w.n;++j)w.x[j]=rr.x[w.free[j]];
                std::fill(w.lambda.begin(),w.lambda.end(),0);w.evaluate();
                for(int i=0;i<w.mi;++i)if(w.active[i]) {
                  // These are auxiliary strict-interior slacks, not new domain
                  // bounds. Their floor follows each row's physical scale and
                  // avoids overwhelming the original filter after restoration.
                  const double floor=std::max(std::numeric_limits<double>::min(),
                      16*std::numeric_limits<double>::epsilon()*std::max(w.inequalities[i].scale,
                      std::abs(w.inequalities[i].value)));
                  w.s[i]=std::max(floor,-w.inequalities[i].value);
                  const long double centered=static_cast<long double>(std::max(mu,barrier_floor))/w.s[i];
                  w.z[i]=std::max(std::numeric_limits<double>::min(),static_cast<double>(std::min(1.L,centered)));
                }
                w.evaluate();const auto candidate_metrics=dca_outer_metrics(w);
                candidate_primal=candidate_metrics.primal;candidate_theta=candidate_metrics.barrier.theta;
                const auto from=accepted_metrics.filter_point(mu);
                auto to=candidate_metrics.filter_point(mu);
                to.phi_change=static_cast<double>(static_cast<long double>(to.objective)-from.objective-
                    static_cast<long double>(mu)*(to.slack_logs-from.slack_logs));
                if(!filter)filter=std::make_unique<DcaMajorantBarrierFilter>(from.theta,true);
                const auto verdict=filter->diagnose(from,to,0,1,st.line_search_armijo);
                recovery_filter_reason=verdict.reason;
                keep=std::isfinite(candidate_primal)&&candidate_primal>=0&&
                    candidate_primal<stage_reference&&candidate_primal<before&&
                    to.theta<from.theta&&verdict.feasibility&&verdict.accepted&&recovery_time_open();
                if(!keep&&!verdict.accepted&&
                    detail::phase1_verified_reduction(private_before,candidate_primal,st.feasibility_tolerance)&&
                    phase1_iterations_used<phase1_iteration_budget&&recovery_time_open()) {
                  next_private_target=detail::phase1_private_precision_target(restoration_target,candidate_primal);
                  refine_private_phase1=next_private_target>0;
                }
                if(keep&&!recovery_time_open())keep=false;
                if(keep) {
                  if(verdict.augment_history(st.majorant_filter_history_fix)&&from.theta>0)filter->insert(from);
                  current=candidate_metrics;rho_memory=1.0;audited_current=false;
                  recovered_state_requires_audit=true;dual_recovery_history.reset();filter_proximal_retries=0;
                  if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
                  direction_proximal_retries=0;
                  direction_centering_used=false;
                  lower_proximal_used=lower_proximal_pending=false;
                  hybrid_skip_trial_once=true;
                  // Only a DIFFERENT anchor with independently recomputed
                  // original-filter feasibility progress may reopen phase I. A rejected/no-progress
                  // attempt stays spent; the global clock/outer budget never reset.
                  ++hybrid_phase1_stages;hybrid_last_phase1_primal=detail::phase1_best_reference(hybrid_last_phase1_primal,candidate_primal);
                  failed_linearization_recovery_attempted=candidate_primal<=st.feasibility_tolerance;
                }
              }
            }catch(const std::runtime_error&) {
              keep=false;recovery_filter_reason="nonfinite-original-phase-I-candidate";
            }
            if(!keep) {accepted_state.restore(w);current=accepted_metrics;}
          } else if(keep) {
            for(int j=0;j<w.n;++j)w.x[j]=rr.x[w.free[j]];
            std::fill(w.lambda.begin(),w.lambda.end(),0);w.evaluate();
            for(int i=0;i<w.mi;++i)if(w.active[i]) {
              w.s[i]=std::max(1e-2,-w.inequalities[i].value);
              w.z[i]=std::min(1.0,std::max(mu,barrier_floor)/w.s[i]);
            }
            w.evaluate();current=dca_outer_metrics(w);
            mu=current.active_count?std::max(barrier_floor,current.product_mean):0;sync_outer_barrier();
            if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
            filter.reset();rho_memory=1.0;audited_current=false;
            recovered_state_requires_audit=true;dual_recovery_history.reset();filter_proximal_retries=0;
            direction_proximal_retries=0;
            direction_centering_used=false;
            lower_proximal_used=lower_proximal_pending=false;
          }
          if(level&&phase1)std::cout<<"[HYBRID-PHASE-I-STAGE] stage="<<private_phase1_stages+1
            <<" target="<<restoration_target<<" private_primal="<<private_before<<"->"<<candidate_primal
            <<" accepted_original_anchor="<<keep<<" filter="<<recovery_filter_reason
            <<" private_refinement="<<refine_private_phase1
            <<" next_private_target="<<next_private_target
            <<" below_final_tolerance="<<(restoration_target<st.feasibility_tolerance)
            <<" relative_descent_private_precision="<<recovery_settings.restoration_private_precision_context
            <<" target_kind=private-restoration-precision-not-KKT"
            <<" iterations_used="<<phase1_iterations_used<<" iteration_budget="<<phase1_iteration_budget
            <<" solve_iterations_used="<<late_phase1_budget.used
            <<" solve_iteration_budget="<<late_phase1_budget.limit
            <<" primal_generation="<<hybrid_accepted_primal_generation
            <<" global_clock_preserved=1"<<std::endl;
          if(!phase1||keep||!refine_private_phase1)break;
          // No original anchor or filter was committed. Continue restoration
          // privately from its improving endpoint with the SAME deadline and
          // remaining iteration budget, until the ORIGINAL filter accepts.
          recovery_point=rr.x;private_before=candidate_primal;
          restoration_target=next_private_target;
          ++private_phase1_stages;
        }
        if(keep&&phase1)continue_barrier_after_original_acceptance("original-filtered-phase-I");
        if(level&&phase1)std::cout<<"[HYBRID-PHASE-I-END] trigger="<<reason
          <<" accepted="<<keep<<" phase1_target="<<restoration_target
          <<" original_primal="<<before<<"->"<<candidate_primal
          <<" original_theta="<<accepted_metrics.barrier.theta<<"->"<<candidate_theta
          <<" original_filter="<<recovery_filter_reason
          <<" accepted_dca_step=0 final_kkt_certificate=0"
          <<" accepted_stages="<<hybrid_phase1_stages
          <<" private_stages="<<private_phase1_stages+1
          <<" phase1_iterations_used="<<phase1_iterations_used
          <<" phase1_iteration_budget="<<phase1_iteration_budget
          <<" phase1_solve_iterations_used="<<late_phase1_budget.used
          <<" phase1_solve_iteration_budget="<<late_phase1_budget.limit
          <<" retry_budget_reopened="<<(keep&&!failed_linearization_recovery_attempted?1:0)
          <<" matrix_free_solves="<<rr.matrix_free_solves<<" row_space_solves="<<rr.row_space_solves
          <<" linear_solves="<<rr.linear_solves<<" deadline_seconds="<<std::min(60.0,remaining)<<std::endl;
        original_cache_current=true;
        if(level)std::cout<<"[FAILED-LINEARIZATION-RECOVERY] trigger="<<reason
          <<" accepted="<<keep<<" primal="<<before<<"->"<<rr.final_feasibility
          <<" iterations="<<rr.iterations<<" restarts="<<rr.restarts<<" seconds="<<seconds(began)
          <<" estimated_workspace_bytes="<<double(recovery_budget.estimated_workspace_bytes)
          <<" full_original_feasibility_required="<<(!phase1)
          <<" restoration_target="<<restoration_target<<std::endl;
        return keep;
      };
      auto conditional_restoration=[&](const char* reason) {
        if(gr_globalization)return false;
        // Enablement is permission, not a startup phase. Only a failed
        // direction/globalization with material infeasibility can trigger it.
        if(!st.majorant_initial_restoration || !st.restoration_enabled ||
            restoration_attempted || w.mi==0 || w.me==0 ||
            !detail::restoration_step_available(st.restoration_shared_iteration_budget,
                st.restoration_max_iterations,result.restoration_iterations)||
            current.primal<=100*st.feasibility_tolerance || seconds(start)>=st.time_limit_seconds)
          return false;
        restoration_attempted=true;
        const Vec saved_x=w.x,saved_s=w.s,saved_z=w.z,saved_lambda=w.lambda;
        const double initial_primal=current.primal;
        models->discard();models->study_rollback_trial();w.evaluate();
        const auto restoration_start=Clock::now();
        auto metric=dca_outer_metrics(w);
        result.restoration_initial_feasibility=metric.primal;
        const double nominal=mu;
        std::fill(w.lambda.begin(),w.lambda.end(),0.0);w.evaluate();
        for(int k=0;k<st.majorant_restoration_iterations &&
            detail::restoration_step_available(st.restoration_shared_iteration_budget,
                st.restoration_max_iterations,result.restoration_iterations)&&
            metric.primal>st.feasibility_tolerance && seconds(start)<st.time_limit_seconds;++k) {
          // Sparse proximal feasibility step: no objective gradient and zero
          // equality multipliers. Existing factors/layout and original rows.
          ++result.restoration_iterations;
          update_outer_matrix();
          if(!w.factorize(1.0))break;
          try {
            auto d=w.direction(Vec(w.mi,0),Vec(w.n,0),w.h,w.rp);
            ++result.restoration_linear_solves;
            DcaMajorantBarrierPath path(w,d);
            double a=w.fraction(d,st.lazy_newton_fraction_to_boundary);
            bool moved=false;
            for(int ls=0;ls<st.line_search_max_steps && seconds(start)<st.time_limit_seconds;++ls,a*=st.line_search_contraction) {
              const auto point=path.point(a,0);
              if(point.finite && point.positive && point.theta<(1-1e-4*a)*metric.barrier.theta) {
                axpy(a,d.x,w.x);path.write_slacks(a,w.s);
                for(int i=0;i<w.mi;++i)w.z[i]=w.active[i]?std::min(1.0,nominal/w.s[i]):0;
                w.evaluate();metric=dca_outer_metrics(w);moved=true;break;
              }
            }
            if(!moved)break;
          } catch(const std::runtime_error&) {break;}
        }
        result.restoration_final_feasibility=metric.primal;
        const bool improved=metric.primal<0.95*initial_primal && finite(w.x);
        if(!improved) {
          w.x=saved_x;w.s=saved_s;w.z=saved_z;w.lambda=saved_lambda;w.evaluate();
        }
        current=dca_outer_metrics(w);original_cache_current=true;audited_current=false;
        if(improved) {
          direction_proximal_retries=0;
          direction_centering_used=false;
          lower_proximal_used=lower_proximal_pending=false;
          mu=current.active_count?std::max(barrier_floor,current.product_mean):0;sync_outer_barrier();
          if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
          stats.majorant_barrier_mu=mu;rho_memory=0;
          filter.reset();
        }
        if(level)std::cout<<"[MAJORANT-CONDITIONAL-RESTORATION] trigger="<<reason
            <<" accepted="<<improved<<" steps="<<result.restoration_iterations
            <<" primal="<<initial_primal<<"->"<<metric.primal
            <<" seconds="<<seconds(restoration_start)<<std::endl;
        return improved;
      };
      bool square_continuation_attempted=false;
      auto recover_square_continuation=[&](int used_outer,const char* reason) {
        if(w.me!=w.n||!square_feasibility||!st.restoration_enabled||square_continuation_attempted||
            current.primal<=st.feasibility_tolerance)return 0;
        square_continuation_attempted=true;
        if(seconds(start)>=st.time_limit_seconds || used_outer>=st.max_outer_iterations) {
          result.status=seconds(start)>=st.time_limit_seconds?Status::TimeLimit:Status::MaxOuterIterations;
          result.message="square continuation shared budget exhausted";return 2;
        }
        int continuation_budget=st.max_outer_iterations-used_outer;
        if(st.restoration_shared_iteration_budget)
          continuation_budget=std::min(continuation_budget,
              detail::remaining_shared_restoration(st.restoration_max_iterations,
                  result.restoration_iterations));
        if(continuation_budget<=0)return 0;
        models->discard();models->study_rollback_trial();w.evaluate();
        const Vec saved_x=w.x,saved_lambda=w.lambda;
        const auto before=current;
        if(level)std::cout<<"[SQUARE-CONTINUATION] trigger="<<reason
            <<" original_primal="<<before.primal<<" backend="<<w.current_factor->name()
            <<" original_input_unchanged=1 intermediate_commit=0 square_continuation_v1=1"<<std::endl;
        DcaSquareContinuation helper(w,square_origin,start,st.time_limit_seconds,
            continuation_budget,level);
        auto recovered=helper.run();
        bool cold_restoration=false;
        Vec recovered_point=recovered.x;
        bool recovered_root=recovered.success;
        // A homotopy branch may fold before t=1. In a constant-objective
        // feasibility system retain the existing nonlinear Phase I from the
        // supplied frozen start as a last bounded recovery. No intermediate
        // homotopy point or borrowed solution becomes its initializer.
        const int remaining_restoration=detail::remaining_shared_restoration(
            detail::remaining_shared_restoration(st.restoration_max_iterations,
                result.restoration_iterations),recovered.iterations);
        if(!recovered_root&&!recovered.limited&&original_constant_objective&&
            (!st.restoration_shared_iteration_budget||remaining_restoration>0)&&
            seconds(start)<st.time_limit_seconds&&
            used_outer+recovered.iterations<st.max_outer_iterations) {
          Settings recovery=st;
          recovery.restoration_max_iterations=st.max_outer_iterations-used_outer-recovered.iterations;
          if(st.restoration_shared_iteration_budget)
            recovery.restoration_max_iterations=std::min(recovery.restoration_max_iterations,
                remaining_restoration);
          recovery.restoration_shared_iteration_budget=true;
          recovery.restoration_max_restarts=problem.n<=16?
              std::min(8,std::max(0,st.restoration_max_restarts)):0;
          recovery.restoration_restart_iterations=std::min(80,st.restoration_restart_iterations);
          recovery.restoration_minimax_enabled=false;
          recovery.restoration_deadline_seconds=std::min(recovery.restoration_deadline_seconds,
              std::chrono::duration<double>(start.time_since_epoch()).count()+st.time_limit_seconds);
          NativeRestorationTrace cold_work;
          const auto rr=run_counted_native_restoration(cold_work,[&]() {
            return FeasibilityRestoration(recovery).solve(problem,problem.x0);
          },[&](const NativeRestorationTrace& work) {
            dca_merge_native_restoration_trace(result,work,"square-cold-restoration",level);
          });
          result.restoration_total_seconds+=rr.total_seconds;
          result.restoration_iterations+=rr.iterations;
          result.restoration_linear_solves+=rr.linear_solves;
          stats.square_cold_restoration_attempted=true;
          stats.square_cold_restoration_iterations=rr.iterations;
          if(rr.final_feasibility<=st.feasibility_tolerance&&rr.x.size()==problem.x0.size()) {
            recovered_point.resize(w.n);
            for(int j=0;j<w.n;++j)recovered_point[j]=rr.x[w.free[j]];
            recovered_root=true;cold_restoration=true;
          }
        }
        stats.square_continuation_attempted=true;
        stats.square_continuation_stages=recovered.stages;
        stats.square_continuation_trials=recovered.trials;
        stats.square_continuation_iterations=recovered.iterations;
        stats.square_continuation_factors=recovered.factors;
        stats.square_continuation_solves=recovered.solves;
        stats.square_continuation_evaluations=recovered.evaluations;
        stats.square_continuation_gradient_evaluations=recovered.gradient_evaluations;
        stats.square_continuation_precision_rejections=recovered.precision_rejections;
        stats.square_continuation_krylov_iterations=recovered.krylov_iterations;
        stats.square_continuation_reused_directions=recovered.reused_directions;
        stats.square_continuation_parameter=recovered.parameter;
        stats.square_continuation_seconds=recovered.seconds;
        stats.square_continuation_linear_error=recovered.linear_error;
        stats.square_continuation_attempted_linear_error=recovered.attempted_linear_error;
        result.restoration_iterations+=recovered.iterations;
        result.restoration_linear_solves+=recovered.solves;
        result.lazy_newton.factor_attempts+=recovered.factors;
        result.restoration_initial_feasibility=before.primal;
        bool accepted=false;
        if(recovered_root && seconds(start)<st.time_limit_seconds) {
          w.x=recovered_point;std::fill(w.lambda.begin(),w.lambda.end(),0.0);
          w.evaluate();original_cache_current=true;audited_current=false;
          current=dca_outer_metrics(w);
          bool filter_pass=false;
          if(!gr_globalization) {
            filter_pass=!filter||filter->diagnose(before.filter_point(mu),
                current.filter_point(mu),0.0,1.0,st.line_search_armijo).accepted;
          } else {
            // The continuation does not bypass the selected globalization.
            // The original GR parameters and centers are frozen at entry.
            detail::gr::Parameters p;
            p.mu=mu;p.rho=w.equality_stabilization.rho;p.kappa=gr_kappa;
            p.center=w.equality_stabilization.multiplier_center;
            detail::gr::State a,b;detail::gr::Direction d;
            a.objective=w.objective.original_at(saved_x);
            b.objective=w.objective.original_at(w.x);
            for(int e=0;e<w.me;++e) {
              const auto& q=w.equalities[e];
              a.c.push_back(q.original_at(saved_x));b.c.push_back(q.original_at(w.x));
              a.y.push_back(saved_lambda[e]);b.y.push_back(w.lambda[e]);
              // Exact quadratic derivative at the saved original anchor.
              const auto path=DcaMajorantBarrierPath::prepare_original(q,[&]() {
                Vec dx(w.n);for(int j=0;j<w.n;++j)dx[j]=w.x[j]-saved_x[j];return dx;
              }(),saved_x);
              d.c.push_back(double(path.linear));d.y.push_back(w.lambda[e]-saved_lambda[e]);
            }
            const auto slope=detail::gr::directional(a,d,p);
            const auto delta=detail::gr::change(a,b,p,0.0L);
            filter_pass=detail::gr::armijo(delta,slope,1.0,st.line_search_armijo);
          }
          accepted=filter_pass&&plausible_original(current)&&audit_original();
          if(accepted) {
            result.status=Status::Solved;
            result.message=cold_restoration?
                "native feasibility recovery passes original globalization and original QCQP KKT":
                "continuation endpoint passes original globalization and original QCQP KKT";
          }
        }
        stats.square_continuation_accepted=accepted&&!cold_restoration;
        stats.square_cold_restoration_accepted=accepted&&cold_restoration;
        result.restoration_final_feasibility=accepted?current.primal:before.primal;
        if(!accepted) {
          w.x=saved_x;w.lambda=saved_lambda;w.evaluate();
          current=before;original_cache_current=true;audited_current=false;
        }
        if(level)std::cout<<"[SQUARE-CONTINUATION-END] accepted="<<accepted
            <<" terminal_route="<<(cold_restoration?"native-feasibility-restoration":"quadratic-continuation")
            <<" parameter="<<recovered.parameter<<" stages="<<recovered.stages
            <<" attempts="<<recovered.trials<<" corrector_steps="<<recovered.iterations
            <<" factors="<<recovered.factors<<" solves="<<recovered.solves
            <<" accepted_stage_linear_error="<<recovered.linear_error
            <<" attempted_stage_linear_error="<<recovered.attempted_linear_error
            <<" precision_rejections="<<recovered.precision_rejections
            <<" linear_krylov_iterations="<<recovered.krylov_iterations
            <<" reused_directions="<<recovered.reused_directions
            <<" original_primal="<<current.primal<<" seconds="<<recovered.seconds
            <<" reason="<<recovered.reason<<" original_filter="<<accepted
            <<" square_continuation_v1=1"<<std::endl;
        if(accepted)return 1;
        if(recovered.limited||seconds(start)>=st.time_limit_seconds) {
          result.status=seconds(start)>=st.time_limit_seconds?Status::TimeLimit:Status::MaxOuterIterations;
          result.message="square continuation shared budget exhausted";return 2;
        }
        return 0;
      };
      if(level) {
        CostScope display(costs,NewtonCost::Diagnostics);
        std::cout<<(gr_globalization?"LazyIPDCA | outer inequality majorants · original GR merit · original KKT\n":
            "LazyIPDCA | outer inequality majorants · original filter · original KKT\n")
          <<"[MAJORANTS] outer_only="<<(hybrid_mode?0:1)
          <<" predictor_corrector="<<st.lazy_newton_predictor_corrector<<" auxiliary_completion="<<(hybrid_mode?1:0)
          <<" inner_rule="<<(hybrid_mode?st.study_inner_rule:"historical-single-Newton")
          <<" original_filter="<<(!gr_globalization)<<" slack_transport="<<(hybrid_mode&&st.study_inner_rule!="barrier-relative")
          <<" slack_policy="<<(hybrid_mode&&st.study_inner_rule!="barrier-relative"?
              "positive-majorant-gap-remap":"historical-unchanged")
          <<" objective_convexification="<<(st.objective_lazy_convexification?"lazy":"off")
          <<" initial_mu="<<mu<<" initialization=actual-complementarity-mean"
          <<" predictor_corrector_requested="<<requested.lazy_newton_predictor_corrector<<"\n";
      }
      bool solved=plausible_original(current)&&audit_original();
      bool initial_fixed_primal_certificate=false;
      // A restored feasible point can already admit ORIGINAL KKT
      // multipliers even when its auxiliary barrier duals are not useful.
      // Try the existing sign/complementarity-bounded terminal fit once,
      // before a positive barrier moves a feasible boundary point away.
      // All admission checks precede the candidate copy and original audit.
      if(!solved&&detail::initial_original_multiplier_fit_admitted(
          problem,requested,current.primal,st.time_limit_seconds-seconds(start))) {
        CostScope certificate(costs,NewtonCost::Certificate);
        Result candidate=result;w.export_state(candidate);
        candidate.feasibility.total_inf=current.primal;
        candidate.dual_infeasibility_inf=current.dual;
        candidate.complementarity_inf=current.comp;
        const double allowance=std::min(.1,st.time_limit_seconds-seconds(start));
        const bool accepted=detail::terminal_original_multiplier_fit(problem,requested,candidate,allowance);
        ++result.multiplier_refit_attempts;
        if(level)std::cout<<"[INITIAL-ORIGINAL-MULTIPLIER-FIT] accepted="<<accepted
            <<" original_primal="<<current.primal<<" budget_seconds="<<allowance
            <<" x_bitwise_unchanged=1 original_bounds_and_signs_audited=1"<<std::endl;
        if(accepted&&seconds(start)<st.time_limit_seconds) {
          // No optimization step has failed at this initial certificate.
          // The terminal fitter's generic rescue provenance does not apply.
          candidate.original_failure_status.clear();
          candidate.original_failure_message.clear();
          candidate.termination_multiplier_source="initial-fixed-primal-complementarity-bounded-fit-original-audit";
          candidate.multiplier_refit_attempts=result.multiplier_refit_attempts;
          ++candidate.multiplier_refit_successes;
          result=std::move(candidate);audited_current=true;solved=true;
          initial_fixed_primal_certificate=true;
        }
      }
      if(solved) {
        result.status=Status::Solved;
        if(!initial_fixed_primal_certificate)
          result.message=equality_start_restored?"restored point satisfies original QCQP KKT":
              "initial point satisfies original QCQP KKT";
      } else if(seconds(start)>=st.time_limit_seconds) {
        result.status=Status::TimeLimit;result.message="outer-majorant time limit after preparation/restoration";
      } else {
        {CostScope preparation(costs,NewtonCost::Preparation);models->prepare_curvature_once();}
        if(level&&models->positive_spectral_count)std::cout<<"[MAJORANT-SPECTRAL] rows="
            <<models->positive_spectral_count<<" envelope_unchanged=1 primal_dual_barrier_gate=1"<<std::endl;
        for(int outer=0;outer<st.max_outer_iterations;++outer) {
          fixed_centering_rolled_back=false;
          if(seconds(start)>=st.time_limit_seconds) {
            result.status=Status::TimeLimit;result.message="outer-majorant time limit";break;
          }
          // A recovered state must be audited before attempting another
          // possibly singular direction (notably zero-objective feasibility).
          if(recovered_state_requires_audit) {
            recovered_state_requires_audit=false;
          if(plausible_original(current)&&audit_original()) {
              result.status=Status::Solved;
              result.message="failure-recovered point satisfies independent original QCQP KKT";
              break;
            }
          }
          if(fixed_centering.observe(gr_globalization&&!models->study_hybrid_controller_enabled()&&
                !square_feasibility&&!lower_proximal_pending&&
                st.max_outer_iterations-outer>st.gr_fixed_centering_steps,
              result.lazy_newton.accepted_steps,current.primal,current.dual,current.comp,
              std::max({current.source_stationarity_residual(mu),current.source_centering_residual(mu),
                current.bound_centering_residual(mu)}),mu,st.feasibility_tolerance,
              st.dual_infeasibility_tolerance,st.complementarity_tolerance)) {
            fixed_centering_anchor=std::make_unique<HybridInnerAnchor>(w);
            fixed_centering_stabilization=w.equality_stabilization;
            fixed_centering_mu=mu;fixed_centering_proximal=rho_memory;fixed_centering_kappa=gr_kappa;
            fixed_centering_initial_residual=current.residual(mu);
            fixed_centering_dual_history=dual_recovery_history;
            fixed_centering_small_step_soc=small_step_soc;fixed_centering_nonlinear_step_soc=nonlinear_step_soc;
            fixed_centering_dual_window=dual_window;fixed_centering_dual_stalls=dual_stalls;
            fixed_centering_last_lambda_attempt=last_lambda_attempt;
            fixed_centering_direction_centering_used=direction_centering_used;
            fixed_centering_lower_proximal_used=lower_proximal_used;
            if(level)std::cout<<std::setprecision(17)<<"[GR-FIXED-CENTERING] event=begin iter="<<outer
                <<" attempt="<<fixed_centering.attempts<<" maximum_attempts=2"
                <<" steps="<<fixed_centering.remaining<<" mu="<<mu
                <<" algorithmic_rho="<<w.equality_stabilization.rho
                <<" barrier_residual="<<fixed_centering_initial_residual
                <<" dual_stationarity_RHS=zero primal_and_centering_RHS=unchanged"
                <<" native_KKT_and_original_GR=1 original_KKT_tolerances_unchanged=1"
                <<" transaction=phase-entry-rollback work_and_time_charged=1"<<std::endl;
          }
          // Original convergence has already been tested at this accepted
          // state. Installing the next tangent model uses that cache only.
          if(models->study_hybrid_enabled()) {
            if(hybrid_skip_trial_once)hybrid_skip_trial_once=false;
            else models->study_begin_trial();
          }
          models->reanchor_from_cache();
          if(outer)++stats.majorant_reanchors;
          stats.majorant_outer_iterations=outer+1;result.outer_iterations=outer+1;
          stats.current_model_steps=0;
          if(level&&(outer==0||outer%st.live_iteration_every==0)) {
            CostScope display(costs,NewtonCost::Diagnostics);
            std::cout<<"[MAJORANT-ITER] iter="<<outer
              <<" objective="<<w.objective.value/w.objective.scale
              <<" original_primal="<<current.primal<<" original_dual="<<current.dual
              <<" original_compl="<<current.comp<<" mu="<<mu
              <<" atomsO="<<models->objective_admitted<<" rhoO="<<models->objective_rho<<" rhoO_floor="<<models->objective_floor
              <<" rhoO_excess="<<models->objective_excess
              <<" rhoI="<<stats.rho_inequalities_max<<" rhoKKT="<<rho_memory
              <<" seconds="<<seconds(start)<<std::endl;
          }
          auto from=current.filter_point(mu);
          if(!filter)filter=std::make_unique<DcaMajorantBarrierFilter>(from.theta,models->study_hybrid_controller_enabled());
          double study_equality_proximal=0,study_equality_absolute_bound=0,study_objective_curvature_lower=0;
          if(models->study_enabled()&&!r3_hybrid) {
            // Common proximal globalization for linearized quadratic
            // equalities. Their signed Lagrangian curvature is omitted from
            // the convex subproblem; do not let numerical rho decay below
            // this absolute-curvature bound. This is NOT C_f or C_g.
            for(int e=0;e<w.me;++e) {
              std::map<int,long double> row;
              for(auto h:w.equalities[e].hessian) {
                row[h.i]+=std::abs((long double)h.v);
                if(h.i!=h.j)row[h.j]+=std::abs((long double)h.v);
              }
              long double norm=0;for(auto r:row)norm=std::max(norm,r.second);
              study_equality_proximal+=std::abs(w.lambda[e])*w.equalities[e].scale*double(norm);
            }
            if(!std::isfinite(study_equality_proximal))throw std::runtime_error("nonfinite equality-linearization proximal bound");
            study_equality_absolute_bound=study_equality_proximal;
            if(hybrid_mode) {
              study::Correction actual_objective;actual_objective.original=w.objective.model_hessian;
              actual_objective.support.resize(w.n);std::iota(actual_objective.support.begin(),actual_objective.support.end(),0);
              for(const auto& border:w.objective.model_atoms) {
                study::HybridAtom atom;atom.weight=border.beta;
                for(int j=0;j<w.n;++j)if(border.direction[j]!=0){atom.indices.push_back(j);atom.values.push_back(border.direction[j]);}
                actual_objective.hybrid_atoms.push_back(std::move(atom));
              }
              const auto previous_certificate=w.hybrid_objective_margin_certificate;
              const auto proof=study::certified_objective_implicit_lower(actual_objective,w.hybrid_objective_margin_certificate);
              if(proof!=previous_certificate) {
                ++w.hybrid_objective_margin_serial;models->study_seconds+=proof->seconds;
                stats.majorant_curvature_seconds=models->study_seconds;
              }
              study_objective_curvature_lower=proof->lower;
              w.hybrid_objective_base_lower=study_objective_curvature_lower;
              // The equality model installed above is affine. Original
              // equality Hessians therefore do not contribute negative
              // curvature to this convex subproblem. Keep their absolute
              // bound as a diagnostic, not an unconditional proximal floor.
              // Original-function globalization below still checks every
              // trial and retains its common proximal retry policy.
              study_equality_proximal=0;
            }
            rho_memory=std::max(rho_memory,study_equality_proximal);
          }
          double study_margin_proximal_floor=0;
          double study_requested_model_margin=0;
          if(hybrid_mode&&st.study_inner_rule!="barrier-relative") {
            study_requested_model_margin=st.study_inner_margin*w.objective.scale;
            study_margin_proximal_floor=study::certified_proximal_margin_floor(
                study_requested_model_margin,study_objective_curvature_lower);
            rho_memory=std::max(rho_memory,study_margin_proximal_floor);
          }
          if(hybrid_mode) {
            std::cout<<std::setprecision(17)<<"[HYBRID-PROXIMAL-POLICY] {\"outer\":"<<outer+1
            <<",\"old_equality_absolute_bound\":"<<study_equality_absolute_bound
            <<",\"objective_model_certified_lower\":"<<study_objective_curvature_lower
            <<",\"equality_curvature_deficit\":"<<study_equality_proximal
            <<",\"proposed_tau\":"<<rho_memory<<",\"objective_scale\":"<<w.objective.scale
            <<",\"ambient_dimension\":"<<w.n
            <<",\"requested_model_margin\":"<<study_requested_model_margin
            <<",\"model_margin_proximal_floor\":"<<study_margin_proximal_floor
            <<",\"source\":\"certified-convex-model-plus-original-filter\""
            <<",\"original_equality_curvature_floor_applied\":false";
            const auto& margin_proof=*w.hybrid_objective_margin_certificate;
            const bool margin_fresh=margin_proof.matrix_key!=study_last_margin_trace_key;
            std::cout<<",\"objective_margin_certificate\":{\"serial\":"<<w.hybrid_objective_margin_serial
              <<",\"kind\":\""<<margin_proof.kind<<"\",\"reason\":\""<<margin_proof.reason
              <<"\",\"gamma\":"<<margin_proof.lower<<",\"interval_certified\":"<<(margin_proof.interval_certified?"true":"false")
              <<",\"precision_bits\":"<<margin_proof.precision_bits<<",\"checks\":"<<margin_proof.checks
              <<",\"seconds\":"<<margin_proof.seconds<<",\"reused\":"<<(margin_fresh?"false":"true")
              <<",\"exact_installed_base_plus_atoms\":true,\"LDL_pivot_used_as_eigenvalue\":false}";
            if((margin_fresh&&w.n<=128)||(w.n<=16&&w.me<=16)) {
              auto matrix=[&](const std::vector<SymTriplet>& h) {
                std::cout<<'[';for(std::size_t k=0;k<h.size();++k){if(k)std::cout<<',';std::cout<<'['<<h[k].i<<','<<h[k].j<<','<<h[k].v<<']';}std::cout<<']';
              };
              std::cout<<",\"objective_model_hessian\":";matrix(w.objective.model_hessian);
              std::cout<<",\"objective_model_atoms\":[";
              for(std::size_t k=0;k<w.objective.model_atoms.size();++k) {
                if(k)std::cout<<',';const auto& a=w.objective.model_atoms[k];
                std::cout<<"{\"weight\":"<<a.beta<<",\"indices\":[";bool first=true;
                for(int j=0;j<w.n;++j)if(a.direction[j]!=0){if(!first)std::cout<<',';first=false;std::cout<<j;}
                std::cout<<"],\"values\":[";first=true;
                for(double value:a.direction)if(value!=0){if(!first)std::cout<<',';first=false;std::cout<<value;}
                std::cout<<"]}";
              }
              std::cout<<']';
            }
            if(w.n<=16&&w.me<=16) {
              std::cout<<",\"equalities\":[";
              for(int e=0;e<w.me;++e){if(e)std::cout<<',';std::cout<<"{\"multiplier\":"<<w.lambda[e]<<",\"scale\":"<<w.equalities[e].scale<<",\"Q\":";std::cout<<'[';for(std::size_t j=0;j<w.equalities[e].hessian.size();++j){if(j)std::cout<<',';const auto& h=w.equalities[e].hessian[j];std::cout<<'['<<h.i<<','<<h.j<<','<<h.v<<']';}std::cout<<']';std::cout<<'}';}
              std::cout<<']';
            }
            std::cout<<"}"<<std::endl;study_last_margin_trace_key=margin_proof.matrix_key;
          }
          update_outer_matrix();
          Vec rc(w.mi,0);
          for(int i=0;i<w.mi;++i)if(w.active[i])rc[i]=w.centered_complementarity_rhs(i,mu);
          Direction direction;
          if(square_feasibility) {
            try {
              if(square_prepared){direction=std::move(square_first);square_prepared=false;}
              else direction=square_direction();
            } catch(const std::runtime_error& error) {
              if(recover_square_continuation(outer+1,"normal-direction"))break;
              if(recover_failed_linearization("square-direction",error.what()))continue;
              result.status=square_factor_failed?Status::FactorizationFailure:Status::InnerFailure;
              result.message=std::string("square feasibility direction rejected: ")+error.what();break;
            }
          } else {
          bool factored=false;
          if(lambda_rho_pending&&rho_memory>0) {
            ++stats.majorant_lambda_rho_probes;
            const double proposed=(models->study_enabled()&&!r3_hybrid)?std::max(0.2*rho_memory,study_equality_proximal):0.2*rho_memory;
            factored=w.factorize(proposed);
            if(level>=2)std::cout<<"[MAJORANT-LAMBDA-RHO] iter="<<outer
                <<" seed="<<rho_memory<<" trial="<<proposed<<" success="<<factored
                <<" actual="<<w.regularization<<std::endl;
            // A failed reduced-seed attempt must not remove the historical retry.
            if(!factored)factored=w.factorize(rho_memory);
          } else factored=w.factorize(rho_memory);
          lambda_rho_pending=false;
          if(!factored) {
            if(rollback_fixed_centering("factorization-rejected"))continue;
            if(rollback_lower_proximal("factorization-rejected"))continue;
            if(recover_failed_linearization("numeric-factorization",w.message))continue;
            if(conditional_restoration("factorization"))continue;
            ++stats.majorant_reject_factorization;stats.majorant_last_rejection="factorization";
            result.status=Status::FactorizationFailure;
            result.message="auxiliary KKT factorization failed: "+w.message;break;
          }
          if(lower_proximal_pending&&level)std::cout<<"[LOWER-PROXIMAL-PROBE] event=factored"
              <<" requested="<<rho_memory<<" actual="<<w.regularization
              <<" previous_actual="<<lower_proximal_previous_actual
              <<" effective_decrease="<<(w.regularization<lower_proximal_previous_actual)
              <<" expanded_audit_required=1 original_filter_required=1"<<std::endl;
          try {
            // The model-relative inner solver constructs and audits all its
            // own directions. Its input Direction is output-only. A redundant
            // outer solve can fail before the private numerical recovery is
            // reached; retain it for legacy and barrier-relative routes.
            if(!(models->study_hybrid_controller_enabled()&&st.study_inner_rule!="barrier-relative"))
              direction=w.direction(rc,fixed_centering.active()?Vec(w.n,0.0):w.rd,
                  w.equality_newton_residual(),w.primal_newton_residual());
          }
          catch(const std::runtime_error& e) {
            const std::string reason=e.what();
            if(rollback_fixed_centering("expanded-centering-direction-rejected"))continue;
            if(rollback_lower_proximal("expanded-direction-rejected"))continue;
            if(retry_direction_centering(reason,outer))continue;
            if(detail::failed_direction_proximal_retry_eligible(reason,direction_proximal_retries,
                st.lazy_newton_atomic_regularization,st.max_outer_iterations-outer,
                seconds(start),st.time_limit_seconds)) {
              const double previous_actual_tau=w.regularization;
              const double retry_base=detail::proximal_retry_base(rho_memory,previous_actual_tau);
              const double next=detail::small_step_proximal_proposal(retry_base,
                  inf_norm(w.rd),inf_norm(w.x),st.lazy_newton_regularization_initial);
              if(std::isfinite(next)&&next>retry_base&&next<=1e12) {
                ++direction_proximal_retries;
                const bool metric_rollback=models->study_trial_pending;
                models->discard();models->study_rollback_trial();w.evaluate();original_cache_current=true;
                if(models->study_hybrid_controller_enabled())hybrid_skip_trial_once=true;
                if(level)std::cout<<"[DIRECTION-PROXIMAL-RETRY] iter="<<outer
                    <<" retry="<<direction_proximal_retries<<" budget=3 rho="<<rho_memory<<"->"<<next
                    <<" previous_actual_tau="<<previous_actual_tau<<" retry_base="<<retry_base
                    <<" state_unchanged=1 original_filter_required=1 expanded_audit_unchanged=1"
                    <<" correction_refinement=0 metric_rollback="<<metric_rollback
                    <<" accepted_metric_digest="<<models->study_metric_digest()
                    <<" charged_outer_retry=1 recovery_order=proximal-before-phase-I cause="<<reason<<std::endl;
                rho_memory=next;continue;
              }
            }
            if(recover_failed_linearization("expanded-direction-residual",reason))continue;
            if(conditional_restoration("direction"))continue;
            const bool precision=reason.find("residual")!=std::string::npos||
                reason.find("backward error")!=std::string::npos||reason.find("nonfinite")!=std::string::npos;
            if(precision)++stats.majorant_reject_precision;
            else ++stats.majorant_reject_linear_backend;
            stats.majorant_last_rejection=precision?"direction-precision":"linear-backend";
            result.status=Status::InnerFailure;result.message="auxiliary KKT direction rejected: "+reason;break;
          }
          }
          HybridInnerOutcome hybrid_inner;
          if(models->study_hybrid_controller_enabled()) {
            hybrid_inner=hybrid_inner_solve(w,direction,mu,st,start,outer+1);
            if(hybrid_inner.model_relative) {
              stats.majorant_slack_transport_enabled=true;
              stats.majorant_slack_transports+=hybrid_inner.slack_remaps;
            }
            stats.inner_steps+=hybrid_inner.steps;result.inner_iterations=stats.inner_steps;
            if(hybrid_inner.terminal_original_state) {
              // Termination is independent of ordinary model/filter step
              // acceptance. Re-audit the saved endpoint after discarding the
              // temporary model, with fresh checks before audit AND commit.
              // A refused/expired certificate restores the accepted anchor.
              const HybridInnerAnchor accepted_anchor(w);
              models->discard();models->study_rollback_trial();
              detail::HybridTerminalAudit terminal;
              terminal.elapsed_before=terminal.elapsed_after=seconds(start);
              if(terminal.elapsed_before<st.time_limit_seconds) {
                hybrid_inner.terminal_original_state->restore(w);
                CostScope certificate(costs,NewtonCost::Certificate);
                w.export_state(result);
                terminal=detail::audit_hybrid_terminal_original(problem,requested,result,
                    [&](){return seconds(start);},&w.original_linear_support);
              } else terminal.expired=true;
              const double commit_elapsed=seconds(start);
              const bool committed=detail::hybrid_terminal_commit_open(terminal,commit_elapsed,st.time_limit_seconds);
              if(!committed)terminal.strict=false;
              if(!(commit_elapsed<st.time_limit_seconds))terminal.expired=true;
              hybrid_terminal_audit_trace(hybrid_inner,st,outer+1,"terminal-commit",terminal,committed,commit_elapsed);
              if(committed) {
                original_cache_current=true;audited_current=true;
                result.status=Status::Solved;
                result.message="private inner endpoint satisfies strict independent original QCQP KKT; no filter step accepted";
                break;
              }
              accepted_anchor.restore(w);original_cache_current=true;audited_current=false;
            }
            if(!hybrid_inner.success) {
              if(rollback_lower_proximal("hybrid-inner-rejected"))continue;
              const bool rejected_metric_trial=models->study_trial_pending;
              models->discard();models->study_rollback_trial();models->study_note_rejection("inner-criterion");
              w.evaluate();original_cache_current=true;
              if(rejected_metric_trial&&seconds(start)<st.time_limit_seconds&&hybrid_trial_retries++<4) {
                hybrid_skip_trial_once=true;continue;
              }
              // note_rejection has requested a different certified FUNCTION
              // correction at an infeasible anchor. Previously every proximal
              // retry set skip_trial_once, so that request could remain starved
              // until all retries were exhausted. Prepare it now and repeat a
              // solve only if the function metric actually changed. The trial
              // remains transactional: the original filter must still accept
              // it, and failure restores the preceding certified correction.
              if(!rejected_metric_trial&&models->study_model_failure_due&&
                  hybrid_trial_retries<4&&seconds(start)<st.time_limit_seconds&&
                  models->study_begin_trial("failed-model-before-proximal")) {
                ++hybrid_trial_retries;hybrid_skip_trial_once=true;
                if(level)std::cout<<"[INNER-FUNCTION-METRIC-RETRY] iter="<<outer
                  <<" retry="<<hybrid_trial_retries<<" tau_unchanged="<<rho_memory
                  <<" correction_changed=1 original_point_unchanged=1"
                  <<" original_filter_required=1 transactional_rollback=1"<<std::endl;
                continue;
              }
              // An unsuccessful convex model is not evidence that another
              // identical solve will become feasible. Try one explicit,
              // original-problem phase I before unchanged-model retries.
              if(recover_failed_linearization("hybrid-inner-failure",hybrid_inner.reason))continue;
              // Phase I has declined or failed without accepting a new
              // original anchor. Change the explicit proximal metric before
              // spending another solve on the same failed numerical model.
              // C_f/C_g stay fixed and the filter retry budget is shared.
              if(hybrid_inner.model_relative && filter_proximal_retries<4 &&
                  !st.lazy_newton_atomic_regularization &&
                  seconds(start)<st.time_limit_seconds) {
                const double retry_base=detail::proximal_retry_base(rho_memory,hybrid_inner.tau_prox);
                const double next=detail::small_step_proximal_proposal(retry_base,
                    inf_norm(w.rd),inf_norm(w.x),st.lazy_newton_regularization_initial);
                if(std::isfinite(next)&&next>retry_base&&next<=1e12) {
                  ++filter_proximal_retries;hybrid_skip_trial_once=true;
                  if(level)std::cout<<"[INNER-PROXIMAL-RETRY] iter="<<outer
                    <<" retry="<<filter_proximal_retries<<" rho="<<rho_memory<<"->"<<next
                    <<" previous_actual_tau="<<hybrid_inner.tau_prox<<" retry_base="<<retry_base
                    <<" tau_floor="<<study_margin_proximal_floor
                    <<" state_unchanged=1 original_filter_required=1 correction_refinement=0"
                    <<" accepted_metric_digest="<<models->study_metric_digest()
                    <<" recovery_order=phase-I-before-proximal inner_reason="<<hybrid_inner.reason<<std::endl;
                  rho_memory=next;continue;
                }
              }
              result.status=seconds(start)>=st.time_limit_seconds?Status::TimeLimit:Status::InnerFailure;
              result.message="hybrid convex barrier model did not satisfy inexact criterion: "+
                  hybrid_inner_failure_details(hybrid_inner);break;
            }
            if(hybrid_inner.model_relative&&mu!=hybrid_inner.barrier_mu) {
              // The fixed convex MODEL has been solved through an internal
              // barrier continuation. Globalization uses its final barrier
              // target consistently at both the accepted anchor and trial.
              mu=hybrid_inner.barrier_mu;stats.majorant_barrier_mu=mu;sync_outer_barrier();
              if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
              from=current.filter_point(mu);filter->rebase(mu);
              if(level>=2)std::cout<<"[HYBRID-FILTER-REBASE] iter="<<outer+1
                  <<" mu="<<mu<<" retained_entries="<<filter->size()<<" history_reset=0"<<std::endl;
            }
          }
          bool curvature_refined=false;Vec curvature_rhs;
          // Late finite-budget stationary iteration, preconditioned by the
          // current Lazy factor. Each trial solves the SAME auxiliary matrix
          // with a changed dual RHS and passes its strict expanded audit.
          if(st.majorant_curvature_corrections>0&&!st.objective_lazy_convexification&&
             current.primal<=st.feasibility_tolerance&&current.comp<=st.complementarity_tolerance&&
             current.dual>st.stationarity_tolerance&&!w.corrected()) {
            CostScope correction_cost(costs,NewtonCost::Direction);
            const auto correction_start=Clock::now();
            const Direction baseline=direction;
            DcaTailTerms terms(w,direction);
            auto norm=[](const std::vector<long double>& v) {
              long double r=0;for(auto x:v)r=std::max(r,std::abs(x));return r;
            };
            const double initial=double(norm(terms.linear));
            // Avoid spending solves when a plain Lazy direction already
            // predicts stationarity below tolerance, or when the curvature
            // defect is not the dominant error. In particular, do not chase
            // evaluation roundoff near the external stopping threshold.
            const bool eligible=initial>w.objective.scale*st.stationarity_tolerance&&
                initial>=0.25*inf_norm(w.rd)&&norm(terms.added)>=0.8*initial&&
                norm(terms.cross)<=0.1*initial;
            if(eligible)++stats.majorant_curvature_attempts;
            double last=initial;
            int used=0;bool aborted=false;
            try {
              for(int j=0;eligible&&j<st.majorant_curvature_corrections;++j) {
                if(seconds(start)>=st.time_limit_seconds)break;
                Vec rhs=w.rd;for(int k=0;k<w.n;++k)rhs[k]-=double(terms.added[k]);
                ++stats.majorant_curvature_solves;
                Direction trial=w.direction(rc,rhs,w.equality_newton_residual(),w.primal_newton_residual());
                if(w.corrected()){aborted=true;break;}
                DcaTailTerms next(w,trial);
                const double residual=double(norm(next.linear));
                if(!std::isfinite(residual)||residual>=0.9*last)break;
                direction=std::move(trial);terms=std::move(next);last=residual;++used;
                curvature_rhs=w.rd;for(int k=0;k<w.n;++k)curvature_rhs[k]-=rhs[k];
                if(residual<=0.05*inf_norm(w.rd))break;
              }
            } catch(const std::runtime_error& error) {
              // The already certified baseline remains available; recovery
              // may change a factor but never the current primal-dual state.
              aborted=true;
              if(level>=2)std::cout<<"[CURVATURE-CORRECTION-REJECT] reason="<<error.what()<<std::endl;
            }
            if(used&&!aborted) {
              const double a=w.fraction(direction,st.lazy_newton_fraction_to_boundary);
              const DcaMajorantBarrierPath trial_path(w,direction);
              const auto point=trial_path.point(a,mu);
              ++stats.majorant_filter_path_builds;++stats.majorant_filter_path_evaluations;++stats.filter_trials;
              long double predicted=0;
              const auto native_defect=terms.native_defect(a);
              for(int k=0;k<w.n;++k)predicted=std::max(predicted,std::abs(
                  (1-a)*terms.initial[k]+a*terms.linear[k]+a*a*terms.cross[k]+
                  (native_defect.empty()?0:native_defect[k])));
              curvature_refined=predicted<0.9*inf_norm(w.rd)&&
                  filter->diagnose(from,point,trial_path.slope(mu),a,st.line_search_armijo).accepted;
            }
            if(curvature_refined)++stats.majorant_curvature_acceptances;
            else {direction=baseline;curvature_rhs.clear();}
            stats.majorant_curvature_correction_seconds+=seconds(correction_start);
            if(level>=2)std::cout<<"[CURVATURE-CORRECTION] iter="<<outer
              <<" eligible="<<eligible<<" accepted="<<curvature_refined<<" solves="<<used
              <<" before="<<initial/w.objective.scale<<" after="<<last/w.objective.scale<<std::endl;
          }
          // A certified majorant can dominate the original stationarity
          // equation indefinitely. Probe the original Newton matrix only
          // after repeated weak contraction and a measured curvature defect.
          if(dual_recovery_history.curvature_steps>=12 && w.mi>0 && !problem.has_signed_square() &&
             (!st.objective_lazy_convexification||
              (models->study_r3_controller()&&models->study_objective.original_convex&&
               models->study_objective.rho==0&&models->study_objective.atoms==0))&&
             (!models->corrected_inequalities.empty()||
              (models->study_r3_controller()&&(stats.rho_inequalities_max>0||stats.atoms_inequalities>0)))) {
            DcaTailTerms defect(w,direction);
            // The explicit proximal term is artificial curvature too. Once
            // it dominates, inspecting only the inequality majorant would
            // make the reduction probe permanently ineligible.
            long double added=0;
            for(int j=0;j<w.n;++j)added=std::max(added,std::abs(defect.added[j]+
                static_cast<long double>(w.regularization)*direction.x[j]));
            if(added>=0.5*inf_norm(w.rd)) {
              const Direction baseline=direction;
              const double baseline_alpha=w.fraction(baseline,st.lazy_newton_fraction_to_boundary);
              long double baseline_prediction=0;
              for(int k=0;k<w.n;++k)baseline_prediction=std::max(baseline_prediction,std::abs(
                  (1-baseline_alpha)*defect.initial[k]+baseline_alpha*defect.linear[k]+
                  baseline_alpha*baseline_alpha*defect.cross[k]));
              const double baseline_rho=w.regularization;
              bool keep=false;
              models->discard();models->study_rollback_trial();update_outer_matrix();
              if(w.factorize(0.0)) {
                try {
                  Direction trial=w.direction(rc,w.rd,w.equality_newton_residual(),w.primal_newton_residual());
                  double a=w.fraction(trial,st.lazy_newton_fraction_to_boundary);
                  const DcaMajorantBarrierPath path(w,trial);
                  ++stats.majorant_filter_path_builds;
                  DcaTailTerms terms(w,trial);
                  // Testing only the full fraction falsely rejects usable
                  // directions. Search the same finite path as globalization;
                  // no filter reset, tolerance relaxation or state commit.
                  for(int ls=0;ls<st.line_search_max_steps &&
                      seconds(start)<st.time_limit_seconds;++ls,a*=st.line_search_contraction) {
                    const auto point=path.point(a,mu);
                    ++stats.majorant_filter_path_evaluations;++stats.filter_trials;
                    long double predicted=0;
                    for(int k=0;k<w.n;++k)predicted=std::max(predicted,std::abs(
                      (1-a)*terms.initial[k]+a*terms.linear[k]+a*a*terms.cross[k]));
                    const auto verdict=filter->diagnose(from,point,path.slope(mu),a,st.line_search_armijo);
                    // Filter acceptance alone is not a stationarity recovery:
                    // a feasibility step may actually increase the dual error.
                    keep=std::isfinite(predicted)&&predicted<baseline_prediction&&
                        predicted<(1-.1*a)*inf_norm(w.rd)&&verdict.accepted;
                    if(level>=2&&(ls==0||keep||ls+1==st.line_search_max_steps))
                      std::cout<<"[CURVATURE-PROBE-DETAIL] ls="<<ls<<" alpha="<<a
                        <<" predicted="<<double(predicted)<<" dual="<<inf_norm(w.rd)
                        <<" rho="<<w.regularization<<" filter="<<verdict.reason<<std::endl;
                    if(keep){direction=std::move(trial);break;}
                  }
                } catch(const std::runtime_error& e) {if(level>=2)std::cout<<"[CURVATURE-PROBE-ERROR] "<<e.what()<<std::endl;}
              }
              if(!keep) {
                models->reanchor_from_cache();update_outer_matrix();
                if(!w.factorize(baseline_rho)) {
                  result.status=Status::FactorizationFailure;
                  result.message="majorant factor could not be restored after original-curvature probe";break;
                }
                direction=baseline;
              }
              if(level)std::cout<<"[MAJORANT-CURVATURE-DEFECT] iter="<<outer
                <<" accepted="<<keep<<" added="<<double(added)<<" original_audit_required=1"<<std::endl;
              dual_recovery_history.curvature_probed();
            }
          }
          if(models->study_enabled()) {
            Vec endpoint=w.x;for(int j=0;j<w.n;++j)endpoint[j]+=direction.x[j];
            try {const auto gap=models->gaps(endpoint,w.z);
            std::cout<<"[CONVEXIFICATION-STEP] iter="<<outer<<" rhoKKT="<<std::setprecision(17)<<w.regularization
              <<" equality_linearization_proximal="<<study_equality_proximal
              <<" backend_numeric_primal="<<w.current_factor->last_primal_regularization()<<" backend_numeric_dual="<<w.current_factor->last_dual_regularization()<<" inequality_correction_value_inf="<<gap.value_gap_inf<<" inequality_correction_dual_inf="<<gap.dual_gap_inf
              <<" factor_attempts="<<result.lazy_newton.factor_attempts<<std::endl;
            }catch(const std::exception& error){std::cout<<"[CONVEXIFICATION-STEP] iter="<<outer<<" diagnostic_unavailable=1 reason="<<error.what()<<std::endl;}
          }
          // Numerical recovery inside the linear solver may change rho.
          // The unit metric belongs to the normal subproblem, not the
          // original (zero) Hessian or the auxiliary-curvature memory.
          rho_memory=square_feasibility?0.0:w.regularization;
          stats.majorant_regularization=rho_memory;
          // Realize the source barrier pair on a curve having EXACTLY the
          // already audited direction as tangent. Do not replace ds/dz by an
          // ideal exact-Newton pair using a differently rounded right hand side.
          // The actual affine trial source residual is preserved pointwise;
          // only its nonlinear source slack-gradient realization is corrected.
          bool gr_source_curve_enabled=gr_globalization&&w.source_primal_barrier_mode&&
              !models->study_hybrid_controller_enabled()&&!gr_kkt_soc&&!gr_defined_soc&&
              (!gr_consistent_bound_slacks||(gr_bound_repair_on_failure&&!gr_bound_watch.active))&&
              !fixed_centering.active()&&!square_feasibility&&
              std::isfinite(direction.residual)&&direction.residual<=st.kkt_backward_error_reject;
          std::vector<detail::GrSourceNewtonCurveRow> gr_source_curve_rows(w.mi);
          std::vector<unsigned char> gr_source_curve_eligible(w.mi,0);
          int gr_source_curve_count=0,gr_source_curve_pre_refused=0;
          if(gr_source_curve_enabled) {
            for(int i=0;i<w.mi;++i)if(w.active[i]&&w.source_primal_barrier(i)) {
              const auto& q=w.inequalities[i];
              if(!q.signed_square.empty()||std::any_of(q.hessian.begin(),q.hessian.end(),
                    [](const auto& h){return h.v!=0;})) {++gr_source_curve_pre_refused;continue;}
              gr_source_curve_rows[i]=detail::GrSourceNewtonCurveRow(w.s[i],w.z[i],mu,
                  w.equality_stabilization.rho,direction.s[i],direction.z[i]);
              if(!gr_source_curve_rows[i].valid()) {++gr_source_curve_pre_refused;continue;}
              gr_source_curve_eligible[i]=1;++gr_source_curve_count;
            }
            gr_source_curve_enabled=gr_source_curve_count>0;
            if(level>=2)std::cout<<std::setprecision(17)<<"[GR-SOURCE-CURVE] event=prepare iter="<<outer
                <<" admitted="<<gr_source_curve_enabled<<" rows="<<gr_source_curve_count
                <<" pre_refused_rows="<<gr_source_curve_pre_refused
                <<" tangent_relative_change=0 expanded_tangent_residual="<<direction.residual
                <<" tolerance="<<st.kkt_backward_error_reject
                <<" same_audited_direction_bitwise=1 same_model_mu_rho_centers_tau=1"
                <<" pair_homotopy=audited-Newton-affine new_factors=0 new_solves=0"
                <<" actual_GR_slope_recomputed=1 historical_fraction_retained=1"<<std::endl;
          }
          bool accepted=false,stop=false;
          double alpha=0;
          Vec accepted_slacks;
          Vec accepted_duals;
          Vec accepted_primal,accepted_equality_duals;
          bool accepted_analytic_soc=false,accepted_consistent_path=false,accepted_kkt_soc=false,accepted_source_staged=false;
          DcaGrKktSoc kkt_soc;
          bool retraction_attempted=false;
          double accepted_equality_defect_inf=0;
          bool accepted_equality_defect_measured=false;
          DcaMajorantBarrierPoint candidate;
          DcaMajorantBarrierFilter::Decision decision;
          HybridInnerOutcome hybrid_accepted_gate=hybrid_inner;
          const bool gr_consistent_bound_master=gr_consistent_bound_slacks;
          // GR-BOUND-REPAIR-CONTROL-BEGIN
          const bool bound_failed_only=gr_bound_repair_on_failure&&!gr_bound_watch.active;
          const int historical_bound_path_attempts=gr_consistent_bound_master&&bound_failed_only?2:1;
          // One bounded curved-source probe precedes the unchanged affine
          // and, when enabled, full-step consistent-bound repair attempts.
          const int bound_path_attempt_limit=historical_bound_path_attempts+(gr_source_curve_enabled?1:0);
          for(int bound_path_attempt=0;bound_path_attempt<bound_path_attempt_limit&&!accepted&&!stop;++bound_path_attempt) {
            const int historical_bound_attempt=bound_path_attempt-(gr_source_curve_enabled?1:0);
            const bool gr_consistent_bound_slacks=gr_consistent_bound_master&&
                (!bound_failed_only||historical_bound_attempt==1);
            const bool staged_source_curve=gr_source_curve_enabled&&bound_path_attempt==0;
            if(bound_path_attempt>0&&seconds(start)>=st.time_limit_seconds) {
              result.status=Status::TimeLimit;result.message="deadline before consistent-bound repair line search";
              stop=true;break;
            }
            // GR-BOUND-FULL-STEP-GATE-BEGIN
            if(historical_bound_attempt>0) {
              const double nominal_fraction=w.fraction(direction,st.lazy_newton_fraction_to_boundary,1);
              if(nominal_fraction!=1.0) {
                if(level>=2)std::cout<<"[GR-BOUND-PATH] iter="<<outer
                    <<" event=repair-skipped reason=nominal-fraction-below-one nominal_fraction="<<nominal_fraction
                    <<" full_step_only=1 original_proximal_retry_retained=1"<<std::endl;
                break;
              }
            }
            // GR-BOUND-FULL-STEP-GATE-END
            // GR-BOUND-REPAIR-CONTROL-BODY
            CostScope globalization(costs,NewtonCost::Globalization);
            const auto filter_start=Clock::now();
            const DcaMajorantBarrierPath path(w,direction,models->study_hybrid_controller_enabled(),bound_product_curve,nullptr,gr_consistent_bound_slacks,gr_bound_watch.active);
            detail::gr::Parameters gr_parameters;
            detail::gr::State gr_anchor;
            detail::gr::Result gr_slope,gr_change;
            auto gr_state_at=[&](double a,const Vec* actual_slacks=nullptr) {
              detail::gr::State state;Vec point=w.x;axpy(a,direction.x,point);
              state.objective=w.objective.original_at(point);
              for(int e=0;e<w.me;++e) {
                state.c.push_back(w.equalities[e].original_at(point));
                state.y.push_back(w.lambda[e]+a*direction.lambda[e]);
              }
              for(int i=0;i<w.mi;++i)if(w.active[i]) {
                const double slack=actual_slacks?actual_slacks->at(i):path.slack(i,a);
                state.s.push_back(slack);
                const double c=static_cast<double>(static_cast<long double>(w.inequalities[i].original_at(point))+slack);
                if(w.inequality_kind[i]==0) {
                  state.c.push_back(c);state.y.push_back(w.z[i]+a*direction.z[i]);
                } else state.b.push_back(c);
              }
              return state;
            };
            if(gr_globalization) {
              if(path.geometric_slacks)throw std::logic_error("GR original R3 diagnostic requires affine slack path");
              Vec bound_z,bound_dz;
              for(int i=0;i<w.mi;++i)if(w.active[i]&&w.inequality_kind[i]!=0) {
                bound_z.push_back(w.z[i]);bound_dz.push_back(direction.z[i]);
              }
              const auto choice=detail::gr::choose_kappa(gr_kappa,bound_z,bound_dz);
              if(!choice.valid)throw std::runtime_error("nonfinite GR hard-bound penalty");
              if(!fixed_centering.active())gr_kappa=choice.value;
              gr_parameters.mu=mu;
              gr_parameters.rho=w.equality_stabilization.rho;gr_parameters.kappa=gr_kappa;
              gr_parameters.center=w.equality_stabilization.multiplier_center;
              for(int i=0;i<w.mi;++i)if(w.active[i]&&w.inequality_kind[i]==0)
                gr_parameters.center.push_back(w.equality_stabilization.source_inequality_center[i]);
              gr_anchor=gr_state_at(0);
              detail::gr::Direction derivative;
              derivative.objective=static_cast<double>(path.objective.linear);
              for(int e=0;e<w.me;++e) {
                derivative.c.push_back(static_cast<double>(path.equalities[e].linear));
                derivative.y.push_back(direction.lambda[e]);
              }
              long double bound_slack_tangent_defect=0;
              for(int i=0;i<w.mi;++i)if(w.active[i]) {
                const long double ds_path=gr_consistent_bound_slacks?path.slack_slope(i):direction.s[i];
                derivative.s.push_back(static_cast<double>(ds_path));
                const double c=static_cast<double>(path.inequalities[i].linear+ds_path);
                if(w.inequality_kind[i]==0) {
                  derivative.c.push_back(c);derivative.y.push_back(direction.z[i]);
                } else {
                  derivative.b.push_back(gr_consistent_bound_slacks?static_cast<double>(path.bound_link_slope(i)):c);
                  bound_slack_tangent_defect=std::max(bound_slack_tangent_defect,std::abs(ds_path-direction.s[i]));
                }
              }
              gr_slope=detail::gr::directional(gr_anchor,derivative,gr_parameters);
              if(gr_consistent_bound_master&&level>=2)std::cout<<std::setprecision(17)
                  <<"[GR-BOUND-PATH] iter="<<outer<<" event=attempt attempt="<<bound_path_attempt
                  <<" path="<<(staged_source_curve?"source-residual-homotopy":(gr_consistent_bound_slacks?"consistent":"legacy-affine"))
                  <<" trigger="<<(gr_bound_watch.active?"accepted-short-step-stagnation":gr_bound_trigger)
                  <<" full_step_only="<<(gr_consistent_bound_slacks&&bound_failed_only)
                  <<" mu="<<mu<<" algorithmic_rho="<<gr_parameters.rho
                  <<" tau="<<w.regularization<<" kappa="<<gr_parameters.kappa
                  <<" same_anchor_direction_factor_centers=1 same_original_GR_required=1"<<std::endl;
              if(level>=2)std::cout<<std::setprecision(17)<<"[MAJORANT-GR-DIRECTION] iter="<<outer
                  <<" valid="<<gr_slope.valid<<" slope="<<double(gr_slope.value)
                  <<" rho="<<gr_parameters.rho<<" kappa="<<gr_parameters.kappa
                  <<" source_C_rows="<<gr_anchor.c.size()<<" hard_bound_links="<<gr_anchor.b.size()
                  <<" consistent_bound_slacks="<<gr_consistent_bound_slacks
                  <<" bound_slack_tangent_defect="<<bound_slack_tangent_defect
                  <<" source_curve_requested="<<staged_source_curve
                  <<" original_objective_and_Jacobians=1 actual_slope_required_negative=1"<<std::endl;
            }
            Direction analytic_soc;
            std::unique_ptr<DcaMajorantBarrierPath> analytic_soc_path;
            bool analytic_soc_attempted=false,analytic_soc_prepared=false;
            double analytic_soc_inf=0,analytic_soc_direction_inf=inf_norm(direction.x),analytic_soc_defect_inf=0;
            std::string analytic_soc_reason="not-attempted";
            auto prepare_analytic_soc=[&]() {
              if(analytic_soc_attempted)return;
              analytic_soc_attempted=true;++result.soc_attempts;++stats.soc_attempts;
              const auto began=Clock::now();
              std::vector<long double> defects(w.me);
              for(int e=0;e<w.me;++e) {
                defects[e]=path.equalities[e].quadratic;
                analytic_soc_defect_inf=std::max(analytic_soc_defect_inf,std::abs(double(defects[e])));
              }
              if(gr_soc_plan.apply(defects,analytic_soc.x,&analytic_soc_reason)) {
                analytic_soc.s.assign(w.mi,0);analytic_soc.z.assign(w.mi,0);analytic_soc.lambda.assign(w.me,0);
                for(int i=0;i<w.mi;++i)if(w.active[i])
                  analytic_soc.s[i]=-w.inequalities[i].directional(analytic_soc.x);
                analytic_soc_inf=inf_norm(analytic_soc.x);
                analytic_soc_prepared=finite(analytic_soc.x)&&finite(analytic_soc.s)&&
                    std::isfinite(analytic_soc_inf)&&analytic_soc_inf>0&&analytic_soc_defect_inf>0;
                analytic_soc_reason=analytic_soc_prepared?"analytic-normal-offset":"zero-or-nonfinite-normal-offset";
                if(analytic_soc_prepared) {
                  analytic_soc_path=std::make_unique<DcaMajorantBarrierPath>(w,direction,false,true,&analytic_soc);
                  ++stats.majorant_filter_path_builds;
                }
              }
              stats.soc_seconds+=seconds(began);
              if(level>=2)std::cout<<std::setprecision(17)<<"[GR-DEFINED-SOC-PREPARE] iter="<<outer
                  <<" prepared="<<analytic_soc_prepared<<" reason="<<analytic_soc_reason
                  <<" original_equality_defect="<<analytic_soc_defect_inf
                  <<" main_direction_inf="<<analytic_soc_direction_inf<<" normal_offset_inf="<<analytic_soc_inf
                  <<" new_factorizations=0 KKT_backsolves=0 source_eta_and_lambda_correction=0"<<std::endl;
            };
            auto gr_staged_state=[&](const Vec& point,const Vec& slacks,const Vec& duals,const Vec& eq_duals) {
              detail::gr::State state;state.objective=w.objective.original_at(point);
              for(int e=0;e<w.me;++e) {
                state.c.push_back(w.equalities[e].original_at(point));state.y.push_back(eq_duals[e]);
              }
              for(int i=0;i<w.mi;++i)if(w.active[i]) {
                state.s.push_back(slacks[i]);
                const double c=static_cast<double>(static_cast<long double>(w.inequalities[i].original_at(point))+slacks[i]);
                if(w.inequality_kind[i]==0){state.c.push_back(c);state.y.push_back(duals[i]);}
                else state.b.push_back(c);
              }
              return state;
            };
            // The values in State still describe feasibility at the actual
            // candidate. Only the merit DIFFERENCE uses original coefficient
            // increments, so neither double rounding of h(x) nor cancellation
            // in (g+s)_trial-(g+s)_anchor discards the source displacement.
            auto gr_actual_change=[&](const detail::gr::State& trial,
                const Vec& point,const Vec& slacks,long double objective_delta) {
              if(!st.gr_coefficient_constraint_increments)
                return detail::gr::change(gr_anchor,trial,gr_parameters,objective_delta);
              if(point.size()!=w.x.size()||slacks.size()!=w.s.size())
                throw std::invalid_argument("GR coefficient increment staged dimensions");
              std::vector<long double> source_delta;source_delta.reserve(gr_anchor.c.size());
              for(int e=0;e<w.me;++e)
                source_delta.push_back(w.equalities[e].original_difference(w.x,point));
              for(int i=0;i<w.mi;++i)if(w.active[i]&&w.inequality_kind[i]==0) {
                detail::CoefficientIncrementSum sum;
                sum.add(w.inequalities[i].original_difference(w.x,point));
                sum.add(static_cast<long double>(slacks[i])-w.s[i]);
                source_delta.push_back(sum.get());
              }
              return detail::gr::change_with_constraint_delta(
                  gr_anchor,trial,gr_parameters,objective_delta,source_delta);
            };
            auto equality_evidence=[&](const DcaMajorantBarrierPath& trial_path) {
              detail::NonlinearTrialSocEvidence evidence(st.feasibility_tolerance);
              const long double a=alpha;
              for(int e=0;e<w.me;++e) {
                const auto& p=trial_path.equalities[e];
                evidence.add_row(p.constant,p.constant+a*p.linear,p.at(a),
                    p.nonlinear_change(a),w.equalities[e].scale);
              }
              return evidence;
            };
            auto measure_accepted_defect=[&](const DcaMajorantBarrierPath& trial_path) {
              accepted_equality_defect_inf=0;
              accepted_equality_defect_measured=adaptive_globalization||st.majorant_nonlinear_trial_soc;
              if(accepted_equality_defect_measured)for(const auto& p:trial_path.equalities)
                accepted_equality_defect_inf=std::max(accepted_equality_defect_inf,
                    std::abs(double(p.nonlinear_change(static_cast<long double>(alpha)))));
            };
            if(models->study_hybrid_controller_enabled()&&level>=2)
              std::cout<<"[HYBRID-OUTER-PATH] iter="<<outer<<" slacks="<<path.slack_path_name()
                <<" x_z_lambda=affine exact_endpoints=1 actual_rounded_slacks=1"
                <<" original_filter_history_preserved=1 model_and_tau_unchanged=1"<<std::endl;
            ++stats.majorant_filter_path_builds;
            alpha=bound_product_curve?w.fraction(direction,st.lazy_newton_fraction_to_boundary,1):
                (models->study_hybrid_controller_enabled()?1.0:w.fraction(direction,st.lazy_newton_fraction_to_boundary));
            if(bound_product_curve&&(outer<20||outer%100==0)&&level>=2) {
              long double largest_ec=0,largest_derivative_defect=0,largest_relative_defect=0;
              int bound_rows=0;
              for(int i=0;i<w.mi;++i)if(w.active[i]&&w.inequality_kind[i]!=0) {
                ++bound_rows;
                const auto ec=DcaBoundProductPath::complementarity_direction_error(
                    w.s[i],w.z[i],gr_consistent_bound_slacks?static_cast<double>(path.slack_slope(i)):direction.s[i],direction.z[i],mu);
                const auto defect=DcaBoundProductPath::tangency_defect(
                    w.s[i],w.z[i],gr_consistent_bound_slacks?static_cast<double>(path.slack_slope(i)):direction.s[i],direction.z[i],mu);
                largest_ec=std::max(largest_ec,std::abs(ec));
                largest_derivative_defect=std::max(largest_derivative_defect,std::abs(defect));
                largest_relative_defect=std::max(largest_relative_defect,std::abs(defect)/
                    std::max({1.L,std::abs(static_cast<long double>(w.z[i])),std::abs(static_cast<long double>(direction.z[i]))}));
              }
              std::cout<<std::setprecision(17)<<"[BOUND-PRODUCT-PATH] iter="<<outer
                  <<" initial_alpha="<<alpha<<" bound_rows="<<bound_rows
                  <<" max_complementarity_direction_error="<<largest_ec
                  <<" max_tangent_defect="<<largest_derivative_defect
                  <<" max_relative_tangent_defect="<<largest_relative_defect
                  <<" mu="<<mu<<" kappa_frozen="<<gr_parameters.kappa
                  <<" exact_tangent_only_for_exact_Newton_equations=1"
                  <<" actual_merit_slope_recomputed="<<gr_consistent_bound_slacks
                  <<" staged_duals_required=1"<<std::endl;
            }
            if(w.source_primal_barrier_mode&&(outer<20||outer%100==0)&&level>=2) {
              const double slack_fraction=w.fraction(direction,st.lazy_newton_fraction_to_boundary,1);
              double bound_fraction=1,legacy_source_fraction=1,min_eta=INFINITY;
              long double min_chi=INFINITY,max_chi=-INFINITY;
              int source_count=0,negative_source_count=0;bool legacy_source_domain=true;
              for(int i=0;i<w.mi;++i)if(w.active[i]) {
                if(w.source_primal_barrier(i)) {
                  ++source_count;negative_source_count+=w.z[i]<0;min_eta=std::min(min_eta,w.z[i]);
                  const long double chi=static_cast<long double>(w.s[i])*w.z[i]/mu;
                  min_chi=std::min(min_chi,chi);max_chi=std::max(max_chi,chi);
                  legacy_source_domain=legacy_source_domain&&w.z[i]>0;
                  if(w.z[i]>0&&direction.z[i]<0)legacy_source_fraction=std::min(legacy_source_fraction,
                      -st.lazy_newton_fraction_to_boundary*w.z[i]/direction.z[i]);
                } else if(w.requires_positive_dual(i)&&direction.z[i]<0)
                  bound_fraction=std::min(bound_fraction,-st.lazy_newton_fraction_to_boundary*w.z[i]/direction.z[i]);
              }
              if(!legacy_source_domain)legacy_source_fraction=0;
              std::cout<<std::setprecision(17)<<"[SOURCE-SLACK-FRACTION] iter="<<outer
                  <<" actual="<<alpha<<" slacks="<<slack_fraction<<" bound_duals="<<bound_fraction
                  <<" old_source_positive_fraction="<<legacy_source_fraction
                  <<" old_source_domain="<<legacy_source_domain<<" source_rows="<<source_count
                  <<" negative_source_eta="<<negative_source_count<<" min_source_eta="<<(source_count?min_eta:0)
                  <<" chi_min="<<(source_count?min_chi:0)<<" chi_max="<<(source_count?max_chi:0)
                  <<" source_gradient_internal="<<w.source_stationarity_inf(mu)
                  <<" no_counterfactual_step_committed=1"<<std::endl;
            }
            // GR-BOUND-LINE-BUDGET-BEGIN
            const int bound_path_line_steps=gr_consistent_bound_slacks&&bound_failed_only?
                1:st.line_search_max_steps;
            // GR-BOUND-LINE-BUDGET-END
            double bound_path_last_tested_alpha=0;
            for(int ls=0;ls<bound_path_line_steps;++ls,alpha*=st.line_search_contraction) {
              if(seconds(start)>=st.time_limit_seconds) {
                result.status=Status::TimeLimit;result.message="outer-majorant time limit in filter";stop=true;break;
              }
              bound_path_last_tested_alpha=alpha;
              ++stats.majorant_barrier_line_trials;++stats.majorant_filter_path_evaluations;
              Vec trial_slacks,trial_primal,trial_duals,trial_equality_duals;
              int trial_source_curved_rows=0,trial_source_affine_rows=0,trial_source_refused_rows=0,trial_source_changed_rows=0;
              if(gr_consistent_bound_slacks||staged_source_curve) {
                // Evaluate and commit THESE arrays, including source free duals.
                // GR-CONSISTENT-BOUND-STAGING-BEGIN: evaluate and commit THESE arrays.
                path.write_primal(alpha,trial_primal);
                path.write_slacks(alpha,trial_slacks,&trial_primal);
                path.write_duals(alpha,mu,trial_duals,&trial_slacks);
                path.write_equality_duals(alpha,trial_equality_duals);
                // GR-CONSISTENT-BOUND-STAGING-END
                if(staged_source_curve)for(int i=0;i<w.mi;++i)if(w.active[i]&&w.source_primal_barrier(i)) {
                  if(!gr_source_curve_eligible[i]) {
                    ++trial_source_affine_rows;++trial_source_refused_rows;continue;
                  }
                  const double affine_s=trial_slacks[i],affine_z=trial_duals[i];
                  if(!detail::gr_source_newton_curve_or_affine(gr_source_curve_rows[i],alpha,
                      (1-static_cast<long double>(st.lazy_newton_fraction_to_boundary))*w.s[i],
                      trial_slacks[i],trial_duals[i])) {
                    // Keep THIS row's already staged affine values. They use
                    // the very same audited tangent and GR derivative at zero.
                    // No residual-homotopy claim is made for this fallback row.
                    ++trial_source_affine_rows;++trial_source_refused_rows;continue;
                  }
                  ++trial_source_curved_rows;
                  trial_source_changed_rows+=(trial_slacks[i]!=affine_s||trial_duals[i]!=affine_z);
                }
                candidate=path.staged_point(trial_primal,trial_slacks,trial_duals,mu);
              } else {
                if(path.geometric_slacks)path.write_slacks(alpha,trial_slacks);
                candidate=gr_globalization?path.rounded_point(alpha,mu):path.point(alpha,mu);
              }
              if(!candidate.positive) {
                ++stats.majorant_reject_domain;++stats.majorant_barrier_line_rejections;
                ++result.lazy_newton.line_search_backtracks;continue;
              }
              ++stats.filter_trials;
              if(gr_globalization) {
                if(!gr_consistent_bound_slacks&&!staged_source_curve)path.write_slacks(alpha,trial_slacks);
                const auto trial=(gr_consistent_bound_slacks||staged_source_curve)?
                    gr_staged_state(trial_primal,trial_slacks,trial_duals,trial_equality_duals):gr_state_at(alpha,&trial_slacks);
                Vec displacement(w.n),actual_primal(w.n);
                for(int j=0;j<w.n;++j) {
                  actual_primal[j]=(gr_consistent_bound_slacks||staged_source_curve)?trial_primal[j]:w.x[j]+alpha*direction.x[j];
                  displacement[j]=actual_primal[j]-w.x[j];
                }
                const auto actual_objective=DcaMajorantBarrierPath::prepare_original(w.objective,displacement,w.x);
                gr_change=gr_actual_change(trial,actual_primal,trial_slacks,actual_objective.change(1));
                decision={};
                decision.accepted=candidate.finite&&candidate.positive&&
                    detail::gr::armijo(gr_change,gr_slope,alpha,st.line_search_armijo);
                decision.objective=decision.accepted;decision.armijo=decision.accepted;
                decision.reason=!gr_slope.valid?"gr-invalid-slope":
                    (!(gr_slope.value<0)?"gr-non-descent":(decision.accepted?"gr-Armijo":"gr-Armijo-rejected"));
                if(level>=2&&(ls==0||decision.accepted||ls+1==st.line_search_max_steps))
                  std::cout<<std::setprecision(17)<<"[MAJORANT-GR-TRIAL] iter="<<outer
                      <<" ls="<<ls<<" alpha="<<alpha<<" accepted="<<decision.accepted
                      <<" reason="<<decision.reason<<" stable_merit_change="<<double(gr_change.value)
                      <<" slope="<<double(gr_slope.value)<<" original_C="<<double(gr_change.true_c_inf)
                      <<" original_filter=0 all_source_rows_evaluated=1"
                      <<" source_curve_attempt="<<staged_source_curve
                      <<" source_curved_rows="<<trial_source_curved_rows
                      <<" source_affine_rows="<<trial_source_affine_rows
                      <<" source_refused_rows="<<trial_source_refused_rows
                      <<" source_changed_rows="<<trial_source_changed_rows
                      <<" source_curve_realized="<<(staged_source_curve&&trial_source_changed_rows>0)
                      <<" source_increments="<<(st.gr_coefficient_constraint_increments?"coefficients":"rounded")
                      <<" rounded_merit_change="<<double(detail::gr::change(
                          gr_anchor,trial,gr_parameters,actual_objective.change(1)).value)
                      <<" same_Armijo=1"<<std::endl;
                if(gr_consistent_bound_slacks&&level>=2&&mu<=1e-4&&
                    (ls==0||decision.accepted||ls+1==st.line_search_max_steps)) {
                  long double anchor_B_l1=0,trial_B_l1=0,bound_link_target_error_l1=0;
                  for(std::size_t i=0;i<gr_anchor.b.size();++i) {
                    anchor_B_l1+=std::abs(static_cast<long double>(gr_anchor.b[i]));
                    trial_B_l1+=std::abs(static_cast<long double>(trial.b[i]));
                    bound_link_target_error_l1+=std::abs(static_cast<long double>(trial.b[i])-
                        (1-static_cast<long double>(alpha))*gr_anchor.b[i]);
                  }
                  std::cout<<std::setprecision(17)<<"[MAJORANT-GR-COMPONENTS] iter="<<outer<<" ls="<<ls
                      <<" alpha="<<alpha<<" accepted="<<decision.accepted<<" mu="<<mu
                      <<" stable_delta="<<gr_change.value<<" barrier="<<gr_change.barrier
                      <<" center="<<gr_change.center<<" C_quadratic="<<gr_change.constraint
                      <<" F_quadratic="<<gr_change.shifted<<" hard_bounds="<<gr_change.hard_bounds
                      <<" slope="<<gr_slope.value<<" slope_barrier="<<gr_slope.barrier
                      <<" slope_hard_bounds="<<gr_slope.hard_bounds<<" kappa="<<gr_parameters.kappa
                      <<" anchor_B_l1="<<anchor_B_l1<<" trial_B_l1="<<trial_B_l1
                      <<" bound_link_target_error_l1="<<bound_link_target_error_l1
                      <<" actual_original_rows=1 no_acceptance_relaxation=1"<<std::endl;
                }
                if(!gr_slope.valid||!(gr_slope.value<0))break;
              } else decision=filter->diagnose(from,candidate,path.slope(mu),alpha,st.line_search_armijo);
              if(decision.accepted&&models->study_hybrid_controller_enabled()) {
                auto gate=hybrid_outer_inner_gate(w,direction,alpha,mu,st,hybrid_inner,
                    path.geometric_slacks?&trial_slacks:nullptr);
                if(!gate.success) {
                  if(gate.model_relative)hybrid_model_relative_trace(gate,st,"outer-gate",outer+1,alpha,false);
                  else hybrid_inner_trace(gate,"outer-gate",outer+1,mu,alpha,false);
                  decision.accepted=false;decision.reason="hybrid-inner-relative-gate";
                } else hybrid_accepted_gate=gate;
              }
              // General equality-normal SOC: one frozen native KKT correction
              // per primary direction, reused over the bounded line search.
              // Unlike the private-pivot path, it may touch Hessian columns;
              // the actual rounded point is always evaluated from originals.
              if(gr_kkt_soc&&gr_globalization&&!decision.accepted&&gr_slope.valid&&
                  gr_slope.value<0&&w.me>0&&!models->study_hybrid_controller_enabled()) {
                const auto began=Clock::now();
                if(!kkt_soc.attempted) {
                  ++result.soc_attempts;++stats.soc_attempts;
                  const int factors_before=w.stats.factor_attempts;
                  kkt_soc.prepare(w,direction,start,st.time_limit_seconds);
                  result.soc_factorizations+=w.stats.factor_attempts-factors_before;
                  if(level>=2)std::cout<<std::setprecision(17)
                      <<"[GR-KKT-SOC-PREPARE] iter="<<outer<<" prepared="<<kkt_soc.prepared
                      <<" reason="<<std::quoted(kkt_soc.reason)
                      <<" defect_inf="<<kkt_soc.defect_inf<<" correction_inf="<<kkt_soc.correction_inf
                      <<" linear_residual="<<kkt_soc.correction.residual
                      <<" KKT_backsolves="<<kkt_soc.backsolves
                      <<" new_factorizations="<<(w.stats.factor_attempts-factors_before)
                      <<" correction_requests_per_direction=1 original_KKT_audit=1"<<std::endl;
                }
                bool soc_accepted=false,equality_improved=false;double correction_ratio=INFINITY;
                DcaGrKktSoc::EqualityNorms before_eq,after_eq;
                detail::gr::Result soc_change;
                DcaGrKktSoc::Staged base_trial,soc_trial;
                const bool staged=DcaGrKktSoc::time_open(start,st.time_limit_seconds)&&
                    kkt_soc.stage(path,alpha,mu,base_trial,soc_trial,correction_ratio);
                if(staged) {
                  ++stats.majorant_barrier_line_trials;++stats.majorant_filter_path_evaluations;
                  ++stats.filter_trials;
                  const auto soc_candidate=path.staged_point(soc_trial.x,soc_trial.s,soc_trial.z,mu);
                  before_eq=DcaGrKktSoc::original_equalities(w,base_trial.x);
                  after_eq=DcaGrKktSoc::original_equalities(w,soc_trial.x);
                  // The normal correction may redistribute equality error.
                  // Keep strict L1 improvement; the unchanged ORIGINAL GR
                  // Armijo test decides acceptance, without an extra max-norm gate.
                  equality_improved=before_eq.finite&&after_eq.finite&&
                      after_eq.l1<before_eq.l1;
                  if(soc_candidate.positive&&soc_candidate.finite&&equality_improved) {
                    const auto soc_state=gr_staged_state(soc_trial.x,soc_trial.s,soc_trial.z,soc_trial.lambda);
                    Vec displacement(w.n);for(int j=0;j<w.n;++j)displacement[j]=soc_trial.x[j]-w.x[j];
                    const auto original_objective=DcaMajorantBarrierPath::prepare_original(w.objective,displacement,w.x);
                    soc_change=gr_actual_change(soc_state,soc_trial.x,soc_trial.s,original_objective.change(1));
                    soc_accepted=detail::gr::armijo(soc_change,gr_slope,alpha,st.line_search_armijo)&&
                        DcaGrKktSoc::time_open(start,st.time_limit_seconds);
                    if(soc_accepted) {
                      accepted_primal=std::move(soc_trial.x);accepted_slacks=std::move(soc_trial.s);
                      accepted_duals=std::move(soc_trial.z);accepted_equality_duals=std::move(soc_trial.lambda);
                      candidate=soc_candidate;gr_change=soc_change;
                      decision.accepted=decision.objective=decision.armijo=true;
                      decision.reason="gr-frozen-KKT-SOC";accepted_kkt_soc=accepted=true;
                      accepted_equality_defect_measured=false;
                      ++result.soc_accepted_steps;++stats.soc_acceptances;
                    }
                  }
                }
                if(!soc_accepted)++result.soc_rejected_steps;
                stats.soc_seconds+=seconds(began);
                if(level>=2&&(ls==0||soc_accepted||ls+1==bound_path_line_steps))
                  std::cout<<std::setprecision(17)<<"[GR-KKT-SOC-TRIAL] iter="<<outer<<" ls="<<ls
                      <<" alpha="<<alpha<<" staged="<<staged<<" correction_ratio="<<correction_ratio
                      <<" equality_improved="<<equality_improved
                      <<" equality_inf_before="<<double(before_eq.inf)<<" equality_inf_after="<<double(after_eq.inf)
                      <<" equality_l1_before="<<double(before_eq.l1)<<" equality_l1_after="<<double(after_eq.l1)
                      <<" equality_policy=l1-strict"
                      <<" accepted="<<soc_accepted
                      <<" stable_merit_change="<<(soc_change.valid?double(soc_change.value):INFINITY)
                      <<" same_frozen_mu_rho_centers_kappa=1 actual_original_GR_Armijo=1"
                      <<" actual_four_vectors_staged=1 original_KKT_termination_required=1"<<std::endl;
                if(accepted_kkt_soc)break;
              }
              // A local equality-normal SOC is tried only after the unchanged
              // base GR candidate rejects. It never replaces the Newton model.
              if(gr_defined_soc&&gr_globalization&&!decision.accepted&&gr_slope.valid&&gr_slope.value<0&&
                  gr_soc_plan.supported&&gr_soc_plan.curved_equalities>0) {
                prepare_analytic_soc();
                const auto soc_trial_start=Clock::now();
                const bool ratio_ok=analytic_soc_prepared&&analytic_soc_direction_inf>0&&
                    static_cast<long double>(alpha)*analytic_soc_inf<=.5L*analytic_soc_direction_inf;
                bool soc_accepted=false;detail::gr::Result soc_change;
                if(ratio_ok) {
                  Vec soc_primal,soc_slacks,soc_duals,soc_equality_duals;
                  // GR-DEFINED-SOC-STAGING-BEGIN: these exact arrays are audited and committed.
                  analytic_soc_path->write_primal(alpha,soc_primal);
                  analytic_soc_path->write_slacks(alpha,soc_slacks);
                  analytic_soc_path->write_duals(alpha,mu,soc_duals,&soc_slacks);
                  analytic_soc_path->write_equality_duals(alpha,soc_equality_duals);
                  // GR-DEFINED-SOC-STAGING-END
                  ++stats.majorant_barrier_line_trials;++stats.majorant_filter_path_evaluations;++stats.filter_trials;
                  const auto soc_candidate=analytic_soc_path->staged_point(soc_primal,soc_slacks,soc_duals,mu);
                  if(gr_soc_domain_diagnostic&&ls==0) {
                    int bad_row=-1,bad_coordinate=-1,dominant_source=-1,beta_row=-1;
                    long double dominant_rs=-1,beta_cap=1;
                    for(int j=0;j<w.n;++j)if(!std::isfinite(soc_primal[j])){bad_coordinate=j;break;}
                    for(int i=0;i<w.mi;++i)if(w.active[i]) {
                      if(bad_row<0&&(!(soc_slacks[i]>0)||!std::isfinite(soc_slacks[i])||
                          !std::isfinite(soc_duals[i])||(w.requires_positive_dual(i)&&!(soc_duals[i]>0))))bad_row=i;
                      const double base_slack=w.s[i]+alpha*direction.s[i];
                      const double correction=(alpha*alpha)*analytic_soc.s[i];
                      if(base_slack>0&&correction<0) {
                        const long double cap=static_cast<long double>(base_slack)/-correction;
                        if(cap<beta_cap){beta_cap=cap;beta_row=i;}
                      }
                      if(w.source_primal_barrier(i)) {
                        const long double rs=static_cast<long double>(w.z[i])-static_cast<long double>(mu)/w.s[i];
                        if(std::abs(rs)>dominant_rs){dominant_rs=std::abs(rs);dominant_source=i;}
                      }
                    }
                    auto original_origin=[&](int i) {
                      const int origin=w.inequality_origin[i];const auto& map=w.inequality_kind[i]==0?
                          diagnostic_inequality_origins:(w.inequality_kind[i]<0?diagnostic_lower_origins:diagnostic_upper_origins);
                      return !st.rank_presolve_context?origin:(origin>=0&&origin<static_cast<int>(map.size())?map[origin]:-1);
                    };
                    std::cout<<std::setprecision(17)<<"[GR-SOC-DOMAIN-WITNESS] iter="<<outer<<" ls="<<ls<<" alpha="<<alpha
                        <<" candidate_positive="<<soc_candidate.positive<<" candidate_finite="<<soc_candidate.finite
                        <<" first_invalid_row="<<bad_row<<" first_nonfinite_coordinate="<<bad_coordinate
                        <<" beta_domain_cap="<<beta_cap<<" beta_limiting_row="<<beta_row;
                    if(bad_coordinate>=0)std::cout<<" coordinate_input_index="<<w.free[bad_coordinate]
                        <<" coordinate_actual="<<soc_primal[bad_coordinate];
                    if(bad_row>=0) {
                      const int i=bad_row,origin=w.inequality_origin[i],j=w.inequality_kind[i]?w.index[origin]:-1;
                      std::cout<<" kind="<<w.inequality_kind[i]<<" workspace_input_origin="<<origin
                          <<" original_origin="<<original_origin(i)<<" row_scale="<<w.inequalities[i].scale
                          <<" s_anchor="<<w.s[i]<<" ds="<<direction.s[i]<<" delta_s="<<analytic_soc.s[i]
                          <<" s_base="<<(w.s[i]+alpha*direction.s[i])
                          <<" normal_slack_correction="<<((alpha*alpha)*analytic_soc.s[i])
                          <<" s_actual="<<soc_slacks[i]<<" dual_actual="<<soc_duals[i]
                          <<" g_anchor="<<w.inequalities[i].value
                          <<" g_base="<<path.inequalities[i].at(alpha)
                          <<" g_SOC="<<analytic_soc_path->inequalities[i].at(alpha);
                      if(j>=0)std::cout<<" bound_free_coordinate="<<j<<" x_anchor="<<w.x[j]
                          <<" x_base="<<path.primal(j,alpha)<<" x_SOC="<<soc_primal[j]
                          <<" lower="<<w.original.lb[origin]<<" upper="<<w.original.ub[origin];
                    }
                    std::cout<<" no_candidate_or_step_modified=1"<<std::endl;
                    if(dominant_source>=0) {
                      const int i=dominant_source;
                      std::cout<<"[GR-SOC-SOURCE-RS-WITNESS] iter="<<outer<<" workspace_row="<<i
                          <<" workspace_input_origin="<<w.inequality_origin[i]<<" original_origin="<<original_origin(i)
                          <<" s_anchor="<<w.s[i]<<" eta_anchor="<<w.z[i]<<" mu="<<mu
                          <<" signed_Rs="<<(static_cast<long double>(w.z[i])-static_cast<long double>(mu)/w.s[i])
                          <<" abs_Rs="<<dominant_rs<<" s_base="<<(w.s[i]+alpha*direction.s[i])
                          <<" normal_slack_correction="<<((alpha*alpha)*analytic_soc.s[i])
                          <<" s_SOC="<<soc_slacks[i]<<" no_candidate_or_step_modified=1"<<std::endl;
                    }
                  }
                  if(soc_candidate.positive&&soc_candidate.finite&&finite(soc_equality_duals)) {
                    const auto soc_state=gr_staged_state(soc_primal,soc_slacks,soc_duals,soc_equality_duals);
                    Vec displacement(w.n);for(int j=0;j<w.n;++j)displacement[j]=soc_primal[j]-w.x[j];
                    const auto original_objective=DcaMajorantBarrierPath::prepare_original(w.objective,displacement,w.x);
                    soc_change=gr_actual_change(soc_state,soc_primal,soc_slacks,original_objective.change(1));
                    soc_accepted=detail::gr::armijo(soc_change,gr_slope,alpha,st.line_search_armijo);
                    if(soc_accepted&&seconds(start)<st.time_limit_seconds) {
                      accepted_primal=std::move(soc_primal);accepted_slacks=std::move(soc_slacks);
                      accepted_duals=std::move(soc_duals);accepted_equality_duals=std::move(soc_equality_duals);
                      candidate=soc_candidate;gr_change=soc_change;decision.accepted=true;
                      decision.objective=true;decision.armijo=true;decision.reason="gr-analytic-defined-pivot-SOC";
                      accepted_analytic_soc=true;accepted=true;
                      measure_accepted_defect(*analytic_soc_path);
                      ++result.soc_accepted_steps;++stats.soc_acceptances;
                    } else soc_accepted=false;
                  }
                }
                stats.soc_seconds+=seconds(soc_trial_start);
                if(level>=2&&(ls==0||soc_accepted||ls+1==st.line_search_max_steps))
                  std::cout<<std::setprecision(17)<<"[GR-DEFINED-SOC-TRIAL] iter="<<outer<<" ls="<<ls
                      <<" alpha="<<alpha<<" ratio_ok="<<ratio_ok<<" accepted="<<soc_accepted
                      <<" correction_ratio="<<(analytic_soc_direction_inf>0?alpha*analytic_soc_inf/analytic_soc_direction_inf:INFINITY)
                      <<" stable_merit_change="<<(soc_change.valid?double(soc_change.value):INFINITY)
                      <<" slope="<<double(gr_slope.value)<<" original_C="<<double(soc_change.true_c_inf)
                      <<" same_frozen_mu_rho_centers_kappa=1 actual_original_GR_Armijo=1 staged_all_four_vectors=1"<<std::endl;
                if(accepted_analytic_soc)break;
              }
              // An accepted source remains a valid fallback. A measured
              // nonlinear equality loss can admit a PRIVATE correction even
              // for a full step; it never revokes the source's filter verdict.
              const bool source_filter_accepted=decision.accepted;
              const bool hybrid_soc=models->study_hybrid_controller_enabled();
              const auto source_candidate=candidate;
              const auto source_decision=decision;
              detail::NonlinearTrialSocEvidence source_equality(st.feasibility_tolerance);
              const bool measure_trial_soc=st.majorant_nonlinear_trial_soc&&
                  st.majorant_conditional_retraction&&!hybrid_soc&&w.me>0&&source_filter_accepted;
              if(measure_trial_soc)source_equality=equality_evidence(path);
              const bool accepted_trial_soc=measure_trial_soc&&source_equality.eligible();
              if(measure_trial_soc&&level>=2)std::cout<<"[MAJORANT-TRIAL-SOC] iter="<<outer
                  <<" alpha="<<alpha<<" source_filter_accepted=1 admitted="<<accepted_trial_soc
                  <<" equality_anchor="<<source_equality.anchor_inf
                  <<" equality_affine="<<source_equality.affine_inf
                  <<" equality_trial="<<source_equality.trial_inf
                  <<" equality_defect="<<source_equality.defect_inf
                  <<" reason="<<source_equality.reason()<<" original_filter_unchanged=1"<<std::endl;
              if(source_filter_accepted) {
                if(gr_consistent_bound_slacks||staged_source_curve) {
                  accepted_primal=std::move(trial_primal);accepted_slacks=std::move(trial_slacks);
                  accepted_duals=std::move(trial_duals);accepted_equality_duals=std::move(trial_equality_duals);
                } else {
                  path.write_slacks(alpha,accepted_slacks);
                  if(bound_product_curve)path.write_duals(alpha,mu,accepted_duals,&accepted_slacks);
                }
                if(!accepted_trial_soc) {
                  measure_accepted_defect(path);accepted=true;break;
                }
              }
              const bool extended_accepted_soc=accepted_trial_soc&&
                  (st.majorant_trial_soc_max_corrections>1||st.majorant_trial_soc_full_primal);
              if(extended_accepted_soc&&!retraction_attempted) {
                retraction_attempted=true;++result.soc_attempts;++stats.soc_attempts;
                const auto ablation_start=Clock::now();
                const double saved_penalty=w.penalty,frozen_tau=w.regularization;
                const auto frozen_retained=w.retained;
                auto* frozen_factor=w.current_factor;
                const auto frozen_attempts=w.stats.factor_attempts;
                const double correction_bound=alpha*std::max(1e-16,inf_norm(direction.x));
                Direction working=direction,best_direction=direction;
                DcaMajorantBarrierPoint best_point=source_candidate;
                auto best_decision=source_decision;
                Vec best_slacks=accepted_slacks,total_x(w.n,0);
                bool best_corrected=false;
                Direction first_arc_increment;
                bool first_arc_increment_available=false;
                bool arc_selected=false;
                bool projection_selected=false;
                int corrections_done=0;
                const char* ablation_reason="budget-exhausted";
                // Every incremental RHS is the ACTUAL provisional residual
                // minus the SAME source affine target. Reusing just a fresh
                // quadratic remainder would double-count past corrections.
                auto provisional=[&](const Direction& d,Vec& x,Vec& slack,Vec& z,
                    Vec& eq_rhs,Vec& primal_rhs,double& hi,double& gi,double& ci) {
                  x=w.x;slack=w.s;z=w.z;
                  axpy(alpha,d.x,x);axpy(alpha,d.s,slack);axpy(alpha,d.z,z);
                  eq_rhs.assign(w.me,0);primal_rhs.assign(w.mi,0);hi=gi=ci=0;
                  const long double target=1-static_cast<long double>(alpha);
                  for(int e=0;e<w.me;++e) {
                    const long double actual=w.equalities[e].original_at(x);
                    eq_rhs[e]=double(actual-target*w.equalities[e].value);
                    hi=std::max(hi,std::abs(double(actual)));
                  }
                  for(int i=0;i<w.mi;++i)if(w.active[i]) {
                    const long double actual=static_cast<long double>(w.inequalities[i].original_at(x))+slack[i];
                    if(st.majorant_trial_soc_full_primal)
                      primal_rhs[i]=double(actual-target*(static_cast<long double>(w.inequalities[i].value)+w.s[i]));
                    gi=std::max(gi,std::abs(double(actual)));
                    ci=std::max(ci,std::abs(slack[i]*z[i]-mu));
                  }
                  if(!finite(x)||!finite(slack)||!finite(z)||!finite(eq_rhs)||!finite(primal_rhs)||
                      !std::isfinite(hi)||!std::isfinite(gi)||!std::isfinite(ci))
                    throw std::runtime_error("nonfinite provisional SOC residual");
                };
                try {
                  for(int correction_index=0;correction_index<st.majorant_trial_soc_max_corrections;++correction_index) {
                    if(seconds(start)>=st.time_limit_seconds){ablation_reason="deadline";break;}
                    Vec trial_x,trial_s,trial_z,eq_rhs,primal_rhs;double h_before=0,g_before=0,c_before=0;
                    provisional(working,trial_x,trial_s,trial_z,eq_rhs,primal_rhs,h_before,g_before,c_before);
                    Direction increment=w.direction(Vec(w.mi,0),Vec(w.n,0),eq_rhs,primal_rhs,
                        0,true,DirectionPurpose::Complete,true);
                    ++corrections_done;
                    if(st.majorant_trial_soc_arc&&correction_index==0&&
                        soc_arc_barrier_event.current(result.lazy_newton.accepted_steps,mu)) {
                      first_arc_increment=increment;first_arc_increment_available=true;
                    }
                    const bool unchanged=w.regularization==frozen_tau&&w.retained==frozen_retained&&
                        w.current_factor==frozen_factor&&w.stats.factor_attempts==frozen_attempts;
                    if(!unchanged){ablation_reason="frozen-factor-changed";break;}
                    Vec next_total=total_x;axpy(1,increment.x,next_total);
                    const double total_norm=inf_norm(next_total);
                    if(!finite(next_total)||total_norm>correction_bound){ablation_reason="total-correction-size";break;}
                    Direction next=working;
                    axpy(1/alpha,increment.x,next.x);axpy(1/alpha,increment.s,next.s);
                    axpy(1/alpha,increment.z,next.z);axpy(1/alpha,increment.lambda,next.lambda);
                    Vec realized_correction(w.n);
                    for(int j=0;j<w.n;++j)realized_correction[j]=
                        (w.x[j]+alpha*next.x[j])-(w.x[j]+alpha*direction.x[j]);
                    if(!finite(realized_correction)||inf_norm(realized_correction)>correction_bound) {
                      ablation_reason="realized-total-correction-size";break;
                    }
                    if(w.fraction(next,st.lazy_newton_fraction_to_boundary)<alpha) {
                      ablation_reason="fraction-to-boundary";break;
                    }
                    DcaMajorantBarrierPath next_path(w,next,false);++stats.majorant_filter_path_builds;
                    auto next_point=next_path.point(alpha,mu);
                    auto next_equality=equality_evidence(next_path);
                    auto next_decision=filter->diagnose(from,next_point,path.slope(mu),alpha,st.line_search_armijo);
                    ++stats.filter_trials;++stats.majorant_filter_path_evaluations;
                    double h_after=0,g_after=0,c_after=0;Vec ignored_eq,ignored_primal;
                    provisional(next,trial_x,trial_s,trial_z,ignored_eq,ignored_primal,h_after,g_after,c_after);
                    const bool valid=next_point.finite&&next_point.positive&&next_decision.accepted&&
                        source_equality.improved_by(next_equality)&&next_point.theta<best_point.theta;
                    if(level>=2)std::cout<<"[MAJORANT-SOC-ABLATION] iter="<<outer
                        <<" correction="<<correction_index+1<<" full_primal="<<st.majorant_trial_soc_full_primal
                        <<" alpha="<<alpha<<" h_before="<<h_before<<" h_after="<<h_after
                        <<" g_plus_s_before="<<g_before<<" g_plus_s_after="<<g_after
                        <<" sz_minus_mu_before="<<c_before<<" sz_minus_mu_after="<<c_after
                        <<" eq_rhs="<<inf_norm(eq_rhs)<<" primal_rhs="<<inf_norm(primal_rhs)
                        <<" total_correction="<<total_norm<<" correction_bound="<<correction_bound
                        <<" original_filter="<<next_decision.accepted<<" selected="<<valid
                        <<" frozen_factor="<<unchanged<<" rc_rhs=0 exact_provisional_residual=1"<<std::endl;
                    if(valid&&seconds(start)<st.time_limit_seconds) {
                      best_direction=next;best_point=next_point;best_decision=next_decision;
                      next_path.write_slacks(alpha,best_slacks);best_corrected=true;
                    }
                    working=std::move(next);total_x.swap(next_total);
                    // Recover the anchor's original feasibility neighborhood;
                    // otherwise spend only the remaining fixed correction budget.
                    double anchor_g=0;
                    for(int i=0;i<w.mi;++i)if(w.active[i])anchor_g=std::max(anchor_g,
                        std::abs(w.inequalities[i].value+w.s[i])/w.inequalities[i].scale);
                    double corrected_g=0;
                    for(int i=0;i<w.mi;++i)if(w.active[i])corrected_g=std::max(corrected_g,
                        std::abs(w.inequalities[i].original_at(trial_x)+trial_s[i])/w.inequalities[i].scale);
                    if(valid&&next_equality.trial_inf<=std::max<long double>(source_equality.anchor_inf,st.feasibility_tolerance)&&
                        (!st.majorant_trial_soc_full_primal||corrected_g<=std::max(anchor_g,st.feasibility_tolerance))) {
                      ablation_reason="anchor-feasibility-recovered";break;
                    }
                  }
                }catch(const std::runtime_error& error) {
                  ablation_reason="audited-correction-rejected";
                  if(level>=2)std::cout<<"[MAJORANT-SOC-ABLATION] iter="<<outer
                      <<" error="<<std::quoted(error.what())<<std::endl;
                }
                // A selected SOC can remain far outside the anchor's
                // feasibility neighborhood. If ORIGINAL coefficients verify
                // closed-form level-set geometry, repair the PURE SOURCE
                // point privately instead of forcing repeated tiny arcs.
                if(st.majorant_soc_defect_control&&soc_defect_epoch&&best_corrected&&
                    source_equality.eligible()&&seconds(start)<st.time_limit_seconds&&
                    w.regularization==frozen_tau&&w.retained==frozen_retained&&
                    w.current_factor==frozen_factor&&w.stats.factor_attempts==frozen_attempts) {
                  const auto anchor_residual=detail::soc_defect_anchor_residuals(w);
                  const auto selected_residual=detail::soc_defect_trial_residuals(w,best_direction,alpha);
                  const bool insufficient=!selected_residual.valid||
                      !detail::soc_defect_within(selected_residual.equality,anchor_residual.equality,st.feasibility_tolerance)||
                      (st.majorant_trial_soc_full_primal&&
                       !detail::soc_defect_within(selected_residual.slack,anchor_residual.slack,st.feasibility_tolerance));
                  if(insufficient) {
                    const bool prior_soc_selected=best_corrected;
                    const char* prior_soc_reason=ablation_reason;
                    if(!soc_defect_geometry_inspected) {
                      soc_defect_geometry_inspected=true;soc_defect_geometry.inspect(w);
                      if(level>=2)std::cout<<"[MAJORANT-SOC-DEFECT] event=coefficient-geometry"
                          <<" iter="<<outer<<" supported="<<soc_defect_geometry.supported
                          <<" blocks="<<soc_defect_geometry.blocks.size()<<" affine_leaves="<<soc_defect_geometry.leaves.size()
                          <<" reason="<<soc_defect_geometry.reason<<" KKT_pivots_used=0 dense_map_built=0"<<std::endl;
                    }
                    if(soc_defect_geometry.supported) {
                      const double source_alpha=alpha,source_slope=path.slope(mu);
                      double proposal_alpha=source_alpha;
                      for(int projection_ls=0;projection_ls<st.line_search_max_steps;
                          ++projection_ls,proposal_alpha*=st.line_search_contraction) {
                        if(seconds(start)>=st.time_limit_seconds){ablation_reason="deadline";break;}
                        if(!(proposal_alpha>0&&proposal_alpha<=1))break;
                        const bool unchanged=w.regularization==frozen_tau&&w.retained==frozen_retained&&
                            w.current_factor==frozen_factor&&w.stats.factor_attempts==frozen_attempts;
                        if(!unchanged)break;
                        Direction corrected;
                        const auto proposal=soc_defect_geometry.project(w,direction,path,proposal_alpha,corrected);
                        const double bound=proposal_alpha*std::max(1e-16,inf_norm(direction.x));
                        const bool size_ok=proposal.valid&&proposal.correction_inf<=bound;
                        const bool boundary_ok=size_ok&&w.fraction(corrected,st.lazy_newton_fraction_to_boundary)>=proposal_alpha;
                        bool filter_ok=false,improved=false,neighborhood=false,selected=false;
                        double source_h=INFINITY,projected_h=INFINITY,source_theta=INFINITY,projected_theta=INFINITY;
                        long double projected_g=INFINITY;
                        const char* reason=!proposal.valid?proposal.reason:(!size_ok?"realized-correction-size":"fraction-to-boundary");
                        if(boundary_ok) {
                          const DcaMajorantBarrierPath corrected_path(w,corrected,false);
                          ++stats.majorant_filter_path_builds;
                          const auto source_same=path.rounded_point(proposal_alpha,mu);
                          const auto point=corrected_path.rounded_point(proposal_alpha,mu);
                          stats.majorant_filter_path_evaluations+=2;
                          auto verdict=filter->diagnose(from,point,source_slope,proposal_alpha,st.line_search_armijo);
                          ++stats.filter_trials;
                          const auto residual=detail::soc_defect_trial_residuals(w,corrected,proposal_alpha);
                          detail::NonlinearTrialSocEvidence source_same_equality(st.feasibility_tolerance),
                              corrected_equality(st.feasibility_tolerance);
                          Vec source_x=w.x,corrected_x=w.x;
                          axpy(proposal_alpha,direction.x,source_x);axpy(proposal_alpha,corrected.x,corrected_x);
                          const long double a=proposal_alpha;
                          for(int e=0;e<w.me;++e) {
                            const auto& original=path.equalities[e];const auto& changed=corrected_path.equalities[e];
                            source_same_equality.add_row(original.constant,original.constant+a*original.linear,
                                w.equalities[e].original_at(source_x),original.nonlinear_change(a),w.equalities[e].scale);
                            corrected_equality.add_row(changed.constant,changed.constant+a*changed.linear,
                                w.equalities[e].original_at(corrected_x),changed.nonlinear_change(a),w.equalities[e].scale);
                          }
                          source_h=double(source_same_equality.trial_inf);projected_h=double(residual.equality);
                          projected_g=residual.slack;source_theta=source_same.theta;projected_theta=point.theta;
                          improved=source_same_equality.eligible()&&source_same_equality.improved_by(corrected_equality)&&
                              point.theta<source_same.theta;
                          neighborhood=anchor_residual.valid&&residual.valid&&
                              detail::soc_defect_within(residual.equality,anchor_residual.equality,st.feasibility_tolerance)&&
                              (!st.majorant_trial_soc_full_primal||
                               detail::soc_defect_within(residual.slack,anchor_residual.slack,st.feasibility_tolerance));
                          filter_ok=verdict.accepted;
                          selected=source_same.finite&&source_same.positive&&point.finite&&point.positive&&
                              improved&&neighborhood&&filter_ok&&seconds(start)<st.time_limit_seconds;
                          reason=selected?"source-levelset-projection-selected":(!neighborhood?"anchor-neighborhood":
                              (!improved?"same-alpha-not-improved":verdict.reason));
                          if(selected) {
                            corrected_path.write_slacks(proposal_alpha,best_slacks);
                            best_direction=std::move(corrected);best_point=point;best_decision=verdict;
                            best_corrected=true;projection_selected=true;alpha=proposal_alpha;
                            ablation_reason="coefficient-source-levelset-projection";
                          }
                        }
                        if(level>=2)std::cout<<"[MAJORANT-SOC-DEFECT] event=source-levelset-projection iter="<<outer
                            <<" trial="<<projection_ls+1<<" source_alpha="<<source_alpha<<" alpha="<<proposal_alpha
                            <<" selected="<<selected<<" reason="<<reason
                            <<" prior_soc_selected="<<prior_soc_selected<<" prior_soc_reason="<<prior_soc_reason
                            <<" scope=accepted-soc-insufficient"
                            <<" correction="<<proposal.correction_inf
                            <<" correction_bound="<<bound<<" boundary="<<boundary_ok<<" original_filter="<<filter_ok
                            <<" same_alpha_improved="<<improved<<" anchor_neighborhood="<<neighborhood
                            <<" source_h="<<source_h<<" projected_h="<<projected_h<<" projected_g_plus_s="<<projected_g
                            <<" source_theta="<<source_theta<<" projected_theta="<<projected_theta
                            <<" blocks="<<proposal.block_count<<" affine_leaves="<<proposal.leaf_count
                            <<" frozen_factor="<<unchanged<<" source_affine_targets=1 pure_source_duals=1"
                            <<" source_slope=1 no_new_solve=1 anchor_unchanged=1"<<std::endl;
                        if(selected)break;
                      }
                    }
                  }
                }
                if(detail::trial_soc_arc_admitted(st.majorant_trial_soc_arc,
                    st.majorant_trial_soc_arc&&soc_arc_barrier_event.current(result.lazy_newton.accepted_steps,mu),
                    first_arc_increment_available,best_corrected,ablation_reason,source_equality)) {
                  // Keep the first frozen-factor SOC as a private second-order
                  // arc proposal. In particular, do not rescale accumulated
                  // corrections from subsequent provisional residual solves.
                  const double source_alpha=alpha,source_slope=path.slope(mu);
                  const char* original_refusal=ablation_reason;
                  const auto arc_scale=detail::trial_soc_arc_initial_scale(w,direction,first_arc_increment,
                      source_alpha,st.lazy_newton_fraction_to_boundary);
                  double t=arc_scale.initial_t;
                  if(!arc_scale.valid&&level>=2)std::cout<<"[MAJORANT-SOC-ARC] iter="<<outer
                      <<" selected=0 reason=invalid-geometric-scale source_alpha="<<source_alpha
                      <<" barrier_drop_event=1 barrier_generation="<<soc_arc_barrier_event.generation
                      <<" accepted_generation="<<result.lazy_newton.accepted_steps
                      <<" previous_mu="<<soc_arc_barrier_event.previous_mu<<" target_mu="<<soc_arc_barrier_event.target_mu
                      <<" size_cap="<<arc_scale.size_cap
                      <<" boundary_cap="<<arc_scale.boundary_cap<<" anchor_unchanged=1"<<std::endl;
                  try {
                    for(int arc_ls=0;arc_ls<st.line_search_max_steps;
                        ++arc_ls,t*=st.line_search_contraction) {
                      if(seconds(start)>=st.time_limit_seconds){ablation_reason="deadline";break;}
                      const double arc_alpha=t*source_alpha;
                      if(!(t>0&&t<1&&arc_alpha>0))break;
                      const bool unchanged=w.regularization==frozen_tau&&w.retained==frozen_retained&&
                          w.current_factor==frozen_factor&&w.stats.factor_attempts==frozen_attempts;
                      if(!unchanged){ablation_reason="frozen-factor-changed";break;}
                      Direction arc=detail::trial_soc_arc_direction(direction,first_arc_increment,source_alpha,t);
                      const bool finite_arc=detail::trial_soc_arc_finite_update(w,arc,arc_alpha);
                      const double arc_bound=arc_alpha*std::max(1e-16,inf_norm(direction.x));
                      Vec arc_x=w.x,source_x=w.x,realized(w.n);
                      for(int j=0;j<w.n;++j) {
                        arc_x[j]+=arc_alpha*arc.x[j];source_x[j]+=arc_alpha*direction.x[j];
                        realized[j]=arc_x[j]-source_x[j];
                      }
                      const double actual_correction=inf_norm(realized);
                      const bool size_ok=finite_arc&&finite(arc_x)&&finite(realized)&&actual_correction<=arc_bound;
                      const bool boundary_ok=size_ok&&w.fraction(arc,st.lazy_newton_fraction_to_boundary)>=arc_alpha;
                      bool filter_ok=false,improved=false,selected=false;
                      const char* arc_reason=!size_ok?"realized-correction-size":"fraction-to-boundary";
                      double source_h=INFINITY,arc_h=INFINITY,source_theta=INFINITY,arc_theta=INFINITY;
                      if(boundary_ok) {
                        const DcaMajorantBarrierPath arc_path(w,arc,false);
                        ++stats.majorant_filter_path_builds;
                        const auto source_same=path.rounded_point(arc_alpha,mu);
                        const auto arc_point=arc_path.rounded_point(arc_alpha,mu);
                        stats.majorant_filter_path_evaluations+=2;
                        const auto arc_decision=filter->diagnose(from,arc_point,source_slope,
                            arc_alpha,st.line_search_armijo);
                        ++stats.filter_trials;
                        detail::NonlinearTrialSocEvidence source_same_equality(st.feasibility_tolerance),
                            arc_equality(st.feasibility_tolerance);
                        const long double a=arc_alpha;
                        for(int e=0;e<w.me;++e) {
                          const auto& ps=path.equalities[e];const auto& pa=arc_path.equalities[e];
                          source_same_equality.add_row(ps.constant,ps.constant+a*ps.linear,
                              w.equalities[e].original_at(source_x),ps.nonlinear_change(a),w.equalities[e].scale);
                          arc_equality.add_row(pa.constant,pa.constant+a*pa.linear,
                              w.equalities[e].original_at(arc_x),pa.nonlinear_change(a),w.equalities[e].scale);
                        }
                        source_h=double(source_same_equality.trial_inf);arc_h=double(arc_equality.trial_inf);
                        source_theta=source_same.theta;arc_theta=arc_point.theta;
                        improved=source_same_equality.eligible()&&source_same_equality.improved_by(arc_equality)&&
                            arc_point.theta<source_same.theta;
                        filter_ok=arc_decision.accepted;
                        selected=source_same.finite&&source_same.positive&&arc_point.finite&&arc_point.positive&&
                            improved&&filter_ok&&seconds(start)<st.time_limit_seconds;
                        arc_reason=selected?"corrected-arc-selected":(!improved?"same-alpha-not-improved":arc_decision.reason);
                        if(selected) {
                          arc_path.write_slacks(arc_alpha,best_slacks);
                          best_direction=std::move(arc);best_point=arc_point;best_decision=arc_decision;
                          best_corrected=true;arc_selected=true;alpha=arc_alpha;ablation_reason="damped-first-correction-arc";
                        }
                      }
                      if(level>=2)std::cout<<"[MAJORANT-SOC-ARC] iter="<<outer
                          <<" trial="<<arc_ls+1<<" source_alpha="<<source_alpha<<" alpha="<<arc_alpha
                          <<" t="<<t<<" original_refusal="<<original_refusal
                          <<" initial_t="<<arc_scale.initial_t<<" size_cap="<<arc_scale.size_cap
                          <<" boundary_cap="<<arc_scale.boundary_cap
                          <<" fraction_safety="<<st.lazy_newton_fraction_to_boundary
                          <<" barrier_drop_event=1 barrier_generation="<<soc_arc_barrier_event.generation
                          <<" accepted_generation="<<result.lazy_newton.accepted_steps
                          <<" previous_mu="<<soc_arc_barrier_event.previous_mu<<" target_mu="<<soc_arc_barrier_event.target_mu
                          <<" correction="<<actual_correction<<" correction_bound="<<arc_bound
                          <<" source_h_same_alpha="<<source_h<<" arc_h="<<arc_h
                          <<" source_theta_same_alpha="<<source_theta<<" arc_theta="<<arc_theta
                          <<" boundary="<<boundary_ok<<" original_filter="<<filter_ok
                          <<" same_alpha_improved="<<improved<<" selected="<<selected<<" reason="<<arc_reason
                          <<" frozen_factor="<<unchanged<<" first_increment_only=1 rc_rhs=0"
                          <<" source_slope=1 full_primal_dual_arc=1 anchor_unchanged=1"<<std::endl;
                      if(selected)break;
                    }
                  }catch(const std::exception& error) {
                    if(level>=2)std::cout<<"[MAJORANT-SOC-ARC] iter="<<outer
                        <<" selected=0 reason=private-probe-rejected error="<<std::quoted(error.what())
                        <<" anchor_unchanged=1"<<std::endl;
                  }
                }
                w.penalty=saved_penalty;stats.soc_seconds+=seconds(ablation_start);
                if(seconds(start)>=st.time_limit_seconds) {
                  result.status=Status::TimeLimit;result.message="deadline during iterated accepted-trial SOC";
                  stop=true;break;
                }
                if(best_corrected) {
                  // Commit only after the entire private probe finished within
                  // deadline. Later rejected increments never replace this best.
                  DcaMajorantBarrierPath best_path(w,best_direction,false);
                  measure_accepted_defect(best_path);
                  if((arc_selected||projection_selected)&&seconds(start)>=st.time_limit_seconds) {
                    result.status=Status::TimeLimit;result.message="deadline before accepted SOC arc commit";
                    stop=true;break;
                  }
                  accepted_slacks=best_slacks;
                  direction=std::move(best_direction);candidate=best_point;decision=best_decision;accepted=true;
                  ++result.soc_accepted_steps;++stats.soc_acceptances;
                  if(level>=2)std::cout<<"[MAJORANT-SOC-ABLATION] iter="<<outer
                      <<" event=best-corrected-accepted corrections="<<corrections_done
                      <<" full_primal="<<st.majorant_trial_soc_full_primal
                      <<" reason="<<ablation_reason<<" source_filter_accepted=1 original_filter=1"
                      <<" frozen_factor=1 no_late_commit=1"<<std::endl;
                  break;
                }
                ++result.soc_rejected_steps;
                if(level>=2)std::cout<<"[MAJORANT-SOC-ABLATION] iter="<<outer
                    <<" event=source-retained corrections="<<corrections_done
                    <<" reason="<<ablation_reason<<" original_source_unchanged=1"<<std::endl;
              }

              // Hybrid SOC is an independent trial of the SAME certified
              // convex model. A filter failure alone never authorizes it:
              // certify the uncorrected point first, keep Cq/tau and factors,
              // then require BOTH the model gate and original filter again.
              const bool source_soc=hybrid_soc&&(st.study_inner_rule=="direction-relative"||
                  st.study_inner_rule=="native-direction-soc"||st.study_inner_rule=="native-reduced-direction-soc");
              const bool original_soc=st.majorant_conditional_retraction&&
                  (!deferred_r3_soc||small_step_soc.enabled()||
                   (adaptive_globalization&&nonlinear_step_soc.enabled()));
              if(!extended_accepted_soc && (hybrid_soc||original_soc||accepted_trial_soc) && !retraction_attempted &&
                  // Repeated accepted small steps can exhibit a Maratos
                  // obstruction even while theta decreases. This changes
                  // only admission of a transactional proposal: the corrected
                  // point must still improve theta and pass the same filter.
                  w.me>0 && (hybrid_soc||accepted_trial_soc||
                    ((deferred_r3_soc||candidate.theta>from.theta)&&path.slope(mu)<0)) &&
                  seconds(start)<st.time_limit_seconds) {
                bool source_certified=true;HybridInnerOutcome source_gate=hybrid_inner;
                if(hybrid_soc){source_gate=hybrid_outer_inner_gate(
                    w,direction,alpha,mu,st,hybrid_inner,
                    path.geometric_slacks?&trial_slacks:nullptr);source_certified=source_gate.success;}
                if(source_certified) {
                retraction_attempted=true;++result.soc_attempts;++stats.soc_attempts;
                struct SocTimer {
                  DcaFilterStatistics& statistics;Clock::time_point began=Clock::now();
                  ~SocTimer(){statistics.soc_seconds+=seconds(began);}
                } soc_timer{stats};
                const double soc_penalty=w.penalty;
                const double soc_tau=w.regularization;
                const bool soc_frozen=hybrid_soc||deferred_r3_soc||accepted_trial_soc;
                const auto soc_retained=w.retained;
                auto* soc_factor=w.current_factor;
                auto soc_factor_attempts=w.stats.factor_attempts;
                bool soc_anchor_factorized=false;
                std::string soc_numerical_detail;
                const char* soc_reason="zero-defect";
                double defect_inf=0,correction_inf=0;
                bool soc_directed_rounding=false;
                try {
                  Vec defect(w.me,0),trial=w.x;
                  for(int j=0;j<w.n;++j)trial[j]+=alpha*direction.x[j];
                  for(int e=0;e<w.me;++e) {
                    if(hybrid_soc) {
                      // Full ORIGINAL residual, including the representable
                      // affine-equation error; alpha^2*d'Qd alone misses it.
                      const long double actual=w.equalities[e].original_at(trial);
                      defect[e]=double(actual-(1-static_cast<long double>(alpha))*
                          w.equalities[e].value);
                    } else defect[e]=path.equalities[e].signed_square.empty()?
                        double(alpha*alpha*path.equalities[e].quadratic):
                        double(path.equalities[e].nonlinear_change(alpha));
                    defect_inf=std::max(defect_inf,std::abs(defect[e]));
                  }
                  // Once accepted small steps established stagnation, a
                  // nonzero quadratic equality defect can obstruct the dual
                  // tail even below the FINAL feasibility tolerance. That
                  // tolerance is a stopping test, not a SOC relevance test.
                  // N=0 retains the historical threshold bit for bit.
                  if(defect_inf>((hybrid_soc||deferred_r3_soc||accepted_trial_soc)?0:st.feasibility_tolerance)) {
                    if(hybrid_soc) {
                      // The inner solve restores this accepted outer anchor
                      // but its last numeric factor belongs to a PRIVATE
                      // Newton point. Reassemble and explicitly factor the
                      // unchanged frozen model here, only when SOC is needed.
                      // All subsequent SOC solves/refinements reuse this same
                      // audited anchor factor. A failure rejects this probe.
                      update_outer_matrix();
                      if(!w.factorize(soc_tau,true))
                        throw std::runtime_error("SOC anchor factorization rejected: "+w.message);
                      soc_anchor_factorized=true;
                      soc_factor=w.current_factor;soc_factor_attempts=w.stats.factor_attempts;
                    }
                    Direction correction=w.direction(Vec(w.mi,0),Vec(w.n,0),defect,Vec(w.mi,0),
                        0,true,DirectionPurpose::Complete,soc_frozen);
                    correction_inf=inf_norm(correction.x);soc_reason="correction-size";
                    const bool unchanged=w.regularization==soc_tau&&w.retained==soc_retained&&
                        w.current_factor==soc_factor&&w.stats.factor_attempts==soc_factor_attempts;
                    if((!soc_frozen||unchanged) && correction_inf<=alpha*
                        (hybrid_soc?inf_norm(direction.x):std::max(1e-16,inf_norm(direction.x)))) {
                      Direction combined=direction;
                      axpy(1/alpha,correction.x,combined.x);
                      axpy(1/alpha,correction.s,combined.s);
                      // A source-certified SOC is a PRIMAL globalization
                      // correction; retain the certified source multipliers.
                      if(!source_soc){axpy(1/alpha,correction.z,combined.z);
                        axpy(1/alpha,correction.lambda,combined.lambda);}
                      if(hybrid_soc) {
                        Vec corrected_trial=w.x;
                        for(int j=0;j<w.n;++j)corrected_trial[j]+=alpha*combined.x[j];
                        soc_directed_rounding=hybrid_adjacent_realization(corrected_trial,trial,correction.x);
                        if(soc_directed_rounding)for(int j=0;j<w.n;++j)
                          combined.x[j]=(corrected_trial[j]-w.x[j])/alpha;
                        if(!finite(corrected_trial)||hybrid_displacement(corrected_trial,trial)>
                            alpha*inf_norm(direction.x))throw std::runtime_error("SOC realization exceeds step bound");
                      }
                      Vec corrected_slacks;
                      if(path.geometric_slacks) {
                        corrected_slacks=trial_slacks;axpy(1,correction.s,corrected_slacks);
                      }
                      soc_reason="fraction-to-boundary";
                      if(w.fraction(combined,st.lazy_newton_fraction_to_boundary,
                          path.geometric_slacks?2:0)>=alpha) {
                        DcaMajorantBarrierPath corrected_path(w,combined,path.geometric_slacks);
                        ++stats.majorant_filter_path_builds;
                        auto corrected_point=hybrid_soc?corrected_path.rounded_point(alpha,mu,
                            path.geometric_slacks?&corrected_slacks:nullptr):corrected_path.point(alpha,mu);
                        detail::NonlinearTrialSocEvidence corrected_equality(st.feasibility_tolerance);
                        if(accepted_trial_soc)corrected_equality=equality_evidence(corrected_path);
                        const bool corrected_equalities=!accepted_trial_soc||
                            source_equality.improved_by(corrected_equality);
                        auto corrected_decision=filter->diagnose(from,corrected_point,path.slope(mu),alpha,st.line_search_armijo);
                        ++stats.filter_trials;++stats.majorant_filter_path_evaluations;
                        bool corrected_model=true;
                        HybridInnerOutcome corrected_gate=hybrid_inner;
                        if(hybrid_soc) {
                          if(source_soc){
                            Vec actual_correction(w.n);
                            for(int j=0;j<w.n;++j)actual_correction[j]=
                                (w.x[j]+alpha*combined.x[j])-(w.x[j]+alpha*direction.x[j]);
                            const auto size=hybrid_source_soc_size(w.x,direction.x,alpha,actual_correction);
                            const auto rounded_source=path.rounded_point(alpha,mu,
                                path.geometric_slacks?&trial_slacks:nullptr);
                            corrected_gate=source_gate;
                            corrected_model=source_gate.success&&unchanged&&size.passed&&
                                rounded_source.finite&&corrected_point.positive&&
                                corrected_point.theta<rounded_source.theta;
                            corrected_gate.success=corrected_model;
                            std::cout<<"[HYBRID-SOURCE-SOC] source_certificate="<<source_gate.success
                                <<" correction_bound="<<size.passed<<" scaled_step="<<size.scaled_step
                                <<" scaled_correction="<<size.scaled_correction<<" limit="<<size.limit
                                <<" original_theta_before="<<rounded_source.theta<<" original_theta_after="<<corrected_point.theta
                                <<" source_multipliers_preserved=1 corrected_trial_model_KKT_claimed=0"
                                <<" original_filter_required=1"<<std::endl;
                          } else {
                            corrected_gate=hybrid_outer_inner_gate(w,combined,alpha,mu,st,hybrid_inner,
                                path.geometric_slacks?&corrected_slacks:nullptr);
                            corrected_model=corrected_gate.success;
                          }
                          if(corrected_gate.model_relative)hybrid_model_relative_trace(corrected_gate,st,
                              source_soc?"soc-source-certificate":"soc-model-gate",outer+1,alpha,false);
                          else hybrid_inner_trace(corrected_gate,"soc-model-gate",outer+1,mu,alpha,false);
                        }
                        soc_reason=!corrected_equalities?"equality-not-improved":
                            (corrected_model?corrected_decision.reason:"soc-model-gate");
                        if(corrected_model && corrected_decision.accepted &&
                            corrected_equalities &&
                            (hybrid_soc||corrected_point.theta<candidate.theta) &&
                            seconds(start)<st.time_limit_seconds) {
                          if(path.geometric_slacks)accepted_slacks=corrected_slacks;
                          else corrected_path.write_slacks(alpha,accepted_slacks);
                          measure_accepted_defect(corrected_path);
                          candidate=corrected_point;decision=corrected_decision;
                          direction=std::move(combined);accepted=true;
                          if(hybrid_soc)hybrid_accepted_gate=corrected_gate;
                          ++result.soc_accepted_steps;++stats.soc_acceptances;
                          if(level>=2)std::cout<<"[MAJORANT-SOC] iter="<<outer<<" accepted=1 alpha="<<alpha
                            <<" adaptive_small_step="<<deferred_r3_soc
                            <<" attempted_on_filter_accepted="<<source_filter_accepted
                            <<" factor_reused=1 anchor_factorized="<<soc_anchor_factorized
                            <<" full_original_residual="<<hybrid_soc
                            <<" directed_rounding="<<soc_directed_rounding<<" defect_inf="<<defect_inf<<" correction_inf="<<correction_inf
                            <<" model_gate="<<corrected_model<<" original_filter=1"<<std::endl;
                          if(level>=2&&accepted_trial_soc)std::cout<<"[MAJORANT-TRIAL-SOC] iter="<<outer
                              <<" alpha="<<alpha<<" event=corrected-accepted source_filter_accepted=1"
                              <<" equality_anchor="<<source_equality.anchor_inf
                              <<" equality_source="<<source_equality.trial_inf
                              <<" equality_corrected="<<corrected_equality.trial_inf
                              <<" theta_source="<<source_candidate.theta
                              <<" theta_corrected="<<corrected_point.theta
                              <<" original_filter=1 frozen_factor="<<unchanged<<std::endl;
                          break;
                        }
                      }
                    } else if(hybrid_soc&&!unchanged)soc_reason="frozen-factor-changed";
                  }
                } catch(const std::runtime_error& error) {
                  // Frozen correction solves cannot refactor or switch their
                  // backend. Failed SOC leaves the source path unchanged.
                  soc_reason="audited-correction-rejected";
                  soc_numerical_detail=error.what();
                }
                w.penalty=soc_penalty;
                ++result.soc_rejected_steps;
                if(level>=2)std::cout<<"[MAJORANT-SOC] iter="<<outer<<" accepted=0 alpha="<<alpha
                  <<" adaptive_small_step="<<deferred_r3_soc
                  <<" attempted_on_filter_accepted="<<source_filter_accepted
                  <<" factor_reused=1 anchor_factorized="<<soc_anchor_factorized
                  <<" full_original_residual="<<hybrid_soc
                  <<" directed_rounding="<<soc_directed_rounding<<" defect_inf="<<defect_inf<<" correction_inf="<<correction_inf
                  <<" reason="<<soc_reason<<" numerical_detail="<<std::quoted(soc_numerical_detail)
                  <<" original_path_unchanged=1"<<std::endl;
                }
              }
              if(source_filter_accepted) {
                // The frozen-factor proposal never altered source direction,
                // candidate or staged slacks. Keep that EXACT accepted source
                // if the correction failed; do not charge a filter rejection.
                // A proposal that used the remaining time cannot commit any
                // point after the solve's deadline.
                if(seconds(start)>=st.time_limit_seconds) {
                  result.status=Status::TimeLimit;
                  result.message="outer-majorant time limit in accepted-trial SOC";
                  stop=true;break;
                }
                candidate=source_candidate;decision=source_decision;
                measure_accepted_defect(path);accepted=true;
                if(level>=2)std::cout<<"[MAJORANT-TRIAL-SOC] iter="<<outer
                    <<" alpha="<<alpha<<" event=source-fallback source_filter_accepted=1"
                    <<" original_candidate_and_slacks_preserved=1 filter_rejection_charged=0"<<std::endl;
                break;
              }
              ++stats.filter_rejections;++stats.majorant_reject_globalization;
              ++stats.majorant_barrier_line_rejections;++result.lazy_newton.line_search_backtracks;
              if(level>=4||(level>=2&&(ls==0||ls+1==st.line_search_max_steps))) {
                CostScope display(costs,NewtonCost::Diagnostics);
            std::cout<<(gr_globalization?"[MAJORANT-GR] iter=":"[MAJORANT-FILTER] iter=")<<outer<<" direction="<<(square_feasibility?"square-feasibility-normal":"single-newton")
                  <<" ls="<<ls<<" alpha="<<alpha<<" accepted=0 reason="<<decision.reason
                  <<" from_theta="<<from.theta<<" theta="<<candidate.theta
                  <<" phi_change="<<candidate.phi_change<<" barrier_slope="<<path.slope(mu)<<" mu="<<mu<<"\n";
              }
            }
            if(analytic_soc_attempted&&!accepted_analytic_soc)++result.soc_rejected_steps;
            stats.original_filter_seconds+=seconds(filter_start);
            // GR-BOUND-REPAIR-NEXT-BEGIN
            if(accepted) {accepted_consistent_path=gr_consistent_bound_slacks;accepted_source_staged=staged_source_curve;}
            if(gr_consistent_bound_master&&level>=2)std::cout<<std::setprecision(17)
                <<"[GR-BOUND-PATH] iter="<<outer
                <<" event="<<(stop?"stopped":(accepted?"accepted":"line-search-failed"))
                <<" attempt="<<bound_path_attempt<<" path="<<(staged_source_curve?"source-residual-homotopy":(gr_consistent_bound_slacks?"consistent":"legacy-affine"))
                <<" trigger="<<(gr_bound_watch.active?"accepted-short-step-stagnation":gr_bound_trigger)
                <<" alpha="<<alpha<<" last_tested_alpha="<<bound_path_last_tested_alpha
                <<" full_step_only="<<(gr_consistent_bound_slacks&&bound_failed_only)
                <<" repair_skipped="<<(accepted&&!gr_consistent_bound_slacks)
                <<" accepted_state_unchanged_until_commit=1"<<std::endl;
            // GR-BOUND-REPAIR-NEXT-END
          }
          // GR-BOUND-REPAIR-CONTROL-END
          if(stop)break;
          if(!accepted) {
            if(recover_square_continuation(outer+1,"globalization-exhausted"))break;
            if(rollback_fixed_centering("original-GR-rejected-centering"))continue;
            rollback_lower_proximal("original-filter-rejected");
            if(models->study_hybrid_enabled()) {
              models->study_note_rejection("original-filter");
            }
            const bool lower_probe_admitted=detail::lower_proximal_probe_eligible(lower_proximal_used,square_feasibility,
                st.lazy_newton_atomic_regularization,current.primal,st.feasibility_tolerance,
                st.max_outer_iterations-outer,seconds(start),st.time_limit_seconds);
            if(gr_globalization&&level>=2) {
              const char* reason=lower_probe_admitted?"eligible":
                  (lower_proximal_used?"one-per-anchor-budget-used":
                  (square_feasibility?"square-feasibility-mode":
                  (st.lazy_newton_atomic_regularization?"atomic-regularization-mode":
                  (st.max_outer_iterations-outer<=1?"remaining-iteration-budget":
                  (!(current.primal<=st.feasibility_tolerance)?"original-primal-not-feasible":
                  "nonfinite-or-expired-time-budget")))));
              std::cout<<std::setprecision(17)<<"[GR-PROXIMAL-POLICY] event=lower-probe-admission iter="<<outer
                  <<" policy="<<st.gr_proximal_policy<<" admitted="<<lower_probe_admitted
                  <<" reason="<<reason<<" used="<<lower_proximal_used
                  <<" original_primal="<<current.primal<<" feasibility_tolerance="<<st.feasibility_tolerance
                  <<" remaining_outer="<<(st.max_outer_iterations-outer)
                  <<" existing_probe_preserved=1 original_merit_required=1"<<std::endl;
            }
            if(lower_probe_admitted) {
              const double actual=models->study_hybrid_controller_enabled()?hybrid_inner.tau_prox:w.regularization;
              const double certified_floor=std::max(study_margin_proximal_floor,study_equality_proximal);
              const double proposal=detail::lower_proximal_proposal(actual,certified_floor);
              if(std::isfinite(proposal)&&proposal>=certified_floor&&proposal<actual) {
                lower_proximal_used=lower_proximal_pending=true;
                lower_proximal_previous_seed=detail::proximal_retry_base(rho_memory,actual);
                lower_proximal_previous_actual=actual;
                lower_proximal_anchor=std::make_unique<HybridInnerAnchor>(w);
                const bool metric_rollback=models->study_trial_pending;
                models->discard();models->study_rollback_trial();w.evaluate();original_cache_current=true;
                if(models->study_hybrid_enabled())hybrid_skip_trial_once=true;
                if(level)std::cout<<"[LOWER-PROXIMAL-PROBE] event=proposed iter="<<outer
                    <<" budget=1-per-primal-anchor old_memory="<<rho_memory<<" previous_actual="<<actual
                    <<" proposed="<<proposal<<" certified_floor="<<certified_floor
                    <<" original_primal="<<current.primal<<" metric_rollback="<<metric_rollback
                    <<" accepted_metric_digest="<<models->study_metric_digest()
                    <<" state_unchanged=1 filter_reset=0 correction_refinement=0"
                    <<" charged_outer_retry=1 expanded_audit_required=1 original_filter_required=1"
                    <<" overregularization_certified=0"<<std::endl;
                rho_memory=proposal;continue;
              }
              if(gr_globalization&&level>=2)std::cout<<std::setprecision(17)
                  <<"[GR-PROXIMAL-POLICY] event=lower-probe-no-change iter="<<outer
                  <<" policy="<<st.gr_proximal_policy<<" previous_actual="<<actual
                  <<" proposed="<<proposal<<" certified_floor="<<certified_floor
                  <<" reason=no-admissible-strict-decrease existing_probe_preserved=1"<<std::endl;
            }
            if(models->study_hybrid_enabled()) {
              if(models->study_trial_pending&&seconds(start)<st.time_limit_seconds&&hybrid_trial_retries++<4) {
                models->discard();models->study_rollback_trial();w.evaluate();original_cache_current=true;
                hybrid_skip_trial_once=true;continue;
              }
            }
            auto_startup_retry.clear_window();
            auto retry_filter_proximal=[&]() {
              // Globalization failure is not an assertion of rank deficiency.
              // An explicit proximal retry never changes a function correction.
              if(!square_feasibility && !st.majorant_initial_restoration &&
                  filter_proximal_retries<4 && !st.lazy_newton_atomic_regularization) {
                const double previous_actual_tau=models->study_hybrid_controller_enabled()?
                    hybrid_inner.tau_prox:w.regularization;
                const double retry_base=models->study_hybrid_controller_enabled()?
                    detail::proximal_retry_base(rho_memory,previous_actual_tau):rho_memory;
                const double historical_next=detail::small_step_proximal_proposal(retry_base,
                    inf_norm(w.rd),inf_norm(w.x),st.lazy_newton_regularization_initial);
                const auto proximal_policy=detail::gr_globalization_proximal_growth(
                    gr_globalization,st.gr_proximal_policy,rho_memory,historical_next);
                const double next=proximal_policy.effective;
                if(gr_globalization&&level)std::cout<<std::setprecision(17)
                    <<"[GR-PROXIMAL-POLICY] event=line-search-failure iter="<<outer
                    <<" policy="<<st.gr_proximal_policy<<" old_memory="<<rho_memory
                    <<" previous_actual_tau="<<previous_actual_tau<<" retry_base="<<retry_base
                    <<" proposed="<<historical_next<<" effective="<<next
                    <<" blocked="<<proximal_policy.blocked<<" reason="<<proximal_policy.reason
                    <<" retry_consumed="<<(std::isfinite(next)&&next>retry_base&&next<=1e12)
                    <<" original_globalization_unchanged=1 numerical_inertia_recovery_unchanged=1"<<std::endl;
                if(std::isfinite(next)&&next>retry_base&&next<=1e12) {
                  ++filter_proximal_retries;
                  const bool metric_rollback=models->study_trial_pending;
                  // No accepted primal-dual state changes; release the model
                  // before rebuilding it with the new frozen proximal term.
                  models->discard();models->study_rollback_trial();
                  if(models->study_hybrid_controller_enabled())hybrid_skip_trial_once=true;
                  if(level)std::cout<<"[FILTER-PROXIMAL-RETRY] iter="<<outer
                    <<" retry="<<filter_proximal_retries<<" rho="<<rho_memory<<"->"<<next
                    <<" previous_actual_tau="<<previous_actual_tau<<" retry_base="<<retry_base
                    <<" tau_floor="<<study_margin_proximal_floor
                    <<" state_unchanged=1 original_filter_required="<<(!gr_globalization)
                    <<" correction_refinement=0 metric_rollback="<<metric_rollback
                    <<" accepted_metric_digest="<<models->study_metric_digest()
                    <<" recovery_order="<<(models->study_hybrid_controller_enabled()?
                        "proximal-before-restoration":"historical-restoration-before-proximal")<<std::endl;
                  rho_memory=next;return true;
                }
              }
              return false;
            };
            // A partial step from an infeasible anchor generally violates
            // the fixed model's affine equalities and cannot pass its inner
            // feasibility certificate. Re-solve with a larger EXPLICIT
            // proximal term before expensive nonlinear restoration. The
            // accepted anchor, C_f/C_g, filter and all budgets stay unchanged.
            if(models->study_hybrid_controller_enabled()&&retry_filter_proximal())continue;
            if(!central_first && recover_failed_linearization("filter-exhausted","original-filter-rejected"))continue;
            if(!central_first && conditional_restoration("filter-exhausted"))continue;
            if(!models->study_hybrid_controller_enabled()&&retry_filter_proximal())continue;
            // One auxiliary reset per barrier or verified source-feasibility
            // epoch, only after the original filter and proximal retries fail.
            detail::filter_recenter_new_primal_epoch(filter_recenter_used,
                filter_recenter_primal_reference,current.primal,st.feasibility_tolerance);
            if(st.majorant_filter_recenter && w.mi>0 && !problem.has_signed_square() &&
                detail::filter_recenter_needed(filter_recenter_used,current.primal,current.comp,
                    std::min(initial_original_comp,mu/w.objective.scale),
                    st.feasibility_tolerance,st.complementarity_tolerance) &&
                seconds(start)<st.time_limit_seconds) {
              filter_recenter_used=true;
              const double old_primal=current.primal,old_dual=current.dual,old_comp=current.comp;
              models->discard();models->study_rollback_trial(); w.evaluate();
              CostScope recenter_cost(costs,NewtonCost::Globalization);
              Vec original_values(w.mi),central_s,central_z;
              for(int i=0;i<w.mi;++i)original_values[i]=w.inequalities[i].value;
              if(!detail::make_filter_recenter_state(original_values,w.active,w.s,w.z,
                  mu,central_s,central_z)) {
                result.status=Status::LineSearchFailure;
                result.message="original filter exhausted; centrality reset is nonfinite";
                break;
              }
              w.s.swap(central_s);w.z.swap(central_z);
              std::fill(w.lambda.begin(),w.lambda.end(),0.0);
              w.evaluate();current=dca_outer_metrics(w);
              filter_recenter_primal_reference=old_primal;++filter_recenter_count;
              mu=current.active_count?std::max(barrier_floor,current.product_mean):0;sync_outer_barrier();
              if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
              filter.reset();rho_memory=0;filter_proximal_retries=0;
              audited_current=false;original_cache_current=true;
              dual_recovery_history.reset();dual_stalls=0;dual_window.clear();
              if(level)std::cout<<"[FILTER-RECENTER] count="<<filter_recenter_count
                <<" barrier_epoch="<<stats.majorant_mu_updates<<" mu="<<mu
                <<" primal="<<old_primal<<"->"<<current.primal
                <<" dual="<<old_dual<<"->"<<current.dual<<" comp="<<old_comp<<"->"<<current.comp
                <<" primal_unchanged=1 per_epoch_budget=1 original_audit_required=1"<<std::endl;
              continue;
            }
            // Central-first does not mean no recovery. Preserve its successful
            // proximal/recenter path, then recover only where it would terminate.
            if(central_first && recover_failed_linearization("filter-exhausted","original-filter-rejected"))continue;
            if(central_first && conditional_restoration("filter-exhausted"))continue;
            // R3: this location was a TERMINAL failure in R2. Do not touch a
            // successful R2 path, nor retry a central-first start. SAD-like
            // trajectories can start above 1e-4 and cannot use R2's tiny-step
            // low-mu window. One central cold retry remains possible here.
            if(detail::terminal_central_retry_needed(
                !gr_globalization&&st.majorant_startup_policy=="auto",central_first,auto_startup_retry.used(),
                w.me>0&&w.mi>0,problem.has_signed_square(),st.objective_lazy_convexification,
                auto_initial_mu,current.primal,current.dual,current.comp,
                st.feasibility_tolerance,st.dual_infeasibility_tolerance,st.complementarity_tolerance,
                st.max_outer_iterations-(outer+1),st.time_limit_seconds-seconds(start))) {
              CostScope retry_cost(costs,NewtonCost::Globalization);
              const double old_primal=current.primal,old_dual=current.dual,old_comp=current.comp,old_mu=mu;
              auto_startup_retry.mark_used();
              models->discard();models->study_rollback_trial();
              const Vec saved_x=w.x,saved_s=w.s,saved_z=w.z,saved_lambda=w.lambda;
              bool valid=false;
              try {
                w.x=auto_frozen_x;w.evaluate();
                valid=finite(w.x)&&std::isfinite(w.objective.value)&&finite(w.objective.gradient);
                Vec central_s=w.s,central_z=w.z;
                for(int e=0;e<w.me;++e)
                  valid=valid&&std::isfinite(w.equalities[e].value)&&finite(w.equalities[e].gradient);
                for(int i=0;i<w.mi;++i) {
                  valid=valid&&std::isfinite(w.inequalities[i].value)&&finite(w.inequalities[i].gradient);
                  central_s[i]=std::max(1.0,-w.inequalities[i].value);
                  central_z[i]=w.active[i]?std::min(1.0,1.0/central_s[i]):w.z[i];
                }
                valid=valid&&finite(central_s)&&finite(central_z);
                if(valid) {
                  w.s.swap(central_s);w.z.swap(central_z);
                  std::fill(w.lambda.begin(),w.lambda.end(),0.0);w.evaluate();
                  const auto candidate_metrics=dca_outer_metrics(w);
                  valid=std::isfinite(candidate_metrics.primal)&&std::isfinite(candidate_metrics.dual)&&
                      std::isfinite(candidate_metrics.comp)&&std::isfinite(candidate_metrics.product_mean);
                  if(valid)current=candidate_metrics;
                }
              } catch(const std::runtime_error&) {valid=false;}
              if(!valid) {
                w.x=saved_x;w.s=saved_s;w.z=saved_z;w.lambda=saved_lambda;
                w.evaluate();current=dca_outer_metrics(w);original_cache_current=true;
                if(level)std::cout<<"[AUTO-STARTUP-TERMINAL-RETRY] accepted=0 reason=nonfinite-central-state"
                    <<" previous_state_restored=1 shared_restart_budget=1"<<std::endl;
              } else {
                central_first=true;w.auto_startup.central_first=true;
                w.auto_startup.initial_mu=1.0;w.auto_startup.reason="terminal-filter-central-retry";
                mu=current.active_count?std::max(barrier_floor,current.product_mean):0;sync_outer_barrier();
                if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
                initial_original_comp=current.comp;
                nonlinear_step_soc.clear_window();
                filter.reset();rho_memory=0;filter_proximal_retries=0;lambda_rho_pending=false;
                dual_recovery_history.reset();dual_stalls=0;dual_window.clear();last_lambda_attempt=outer-8;
                audited_current=false;original_cache_current=true;recovered_state_requires_audit=true;
                stats.majorant_barrier_mu=mu;stats.majorant_barrier_residual=current.residual(mu);
                // Intentionally do not reset outer, the clock, recenter_used,
                // restoration_attempted, or failed_linearization_recovery_attempted.
                if(level)std::cout<<"[AUTO-STARTUP-TERMINAL-RETRY] accepted=1 iter="<<outer+1
                    <<" trigger=terminal-filter-failure mu="<<old_mu<<"->"<<mu
                    <<" primal="<<old_primal<<"->"<<current.primal
                    <<" dual="<<old_dual<<"->"<<current.dual
                    <<" comp="<<old_comp<<"->"<<current.comp
                    <<" reset_primal=frozen-x0 shared_restart_budget=1"
                    <<" global_iterations_preserved=1 global_time_preserved=1"
                    <<" recenter_budget_preserved=1 recovery_enabled="<<st.restoration_enabled<<std::endl;
                continue;
              }
            }
            stats.majorant_last_rejection=gr_globalization?"GR-merit":"barrier-filter";
            if(models->study_hybrid_controller_enabled()&&seconds(start)>=st.time_limit_seconds) {
              result.status=Status::TimeLimit;
              result.message="global time limit reached during original-filter recovery";
            } else {
              result.status=Status::LineSearchFailure;
              result.message=gr_globalization?"GR merit Armijo rejected the stabilized auxiliary Newton direction":
                  "original barrier filter rejected the single auxiliary Newton direction";
            }
            break;
          }
          if((st.majorant_soc_defect_control||accepted_analytic_soc||accepted_consistent_path||accepted_kkt_soc||accepted_source_staged)&&seconds(start)>=st.time_limit_seconds) {
            result.status=Status::TimeLimit;result.message="deadline before SOC repair commit";break;
          }
          if(!gr_globalization&&decision.augment_history(st.majorant_filter_history_fix)&&from.theta>0)filter->insert(from);
          Vec tx=w.x,ts=std::move(accepted_slacks),tz=w.z,tl=w.lambda;
          if(accepted_analytic_soc||accepted_consistent_path||accepted_kkt_soc||accepted_source_staged) {
            if(accepted_primal.size()!=w.x.size()||accepted_duals.size()!=w.z.size()||
                accepted_equality_duals.size()!=w.lambda.size()||ts.size()!=w.s.size())
              throw std::logic_error("accepted canonical curve state was not fully staged");
            tx=std::move(accepted_primal);tz=std::move(accepted_duals);tl=std::move(accepted_equality_duals);
          } else {
            axpy(alpha,direction.x,tx);
            if(bound_product_curve) {
              if(accepted_duals.size()!=w.z.size())
                throw std::logic_error("accepted bound product dual state was not staged");
              tz=std::move(accepted_duals);
            } else axpy(alpha,direction.z,tz);
            axpy(alpha,direction.lambda,tl);
          }
          if(!finite(tx)||!finite(ts)||!finite(tz)||!finite(tl))
            throw std::runtime_error("nonfinite filter-accepted primal-dual point");
          std::unique_ptr<DcaTailTerms> tail;
          if(st.majorant_dual_diagnostics&&!bound_product_curve){CostScope diagnostic_cost(costs,NewtonCost::Diagnostics);tail=std::make_unique<DcaTailTerms>(w,direction);}
          double step_inf=0;
          for(int j=0;j<w.n;++j)step_inf=std::max(step_inf,std::abs(tx[j]-w.x[j]));
          const double historical_relative_step=st.standard_equality_watch&&!gr_globalization?
              detail::HistoricalEqualityWatch::relative_primal_step(w.x,tx):
              std::numeric_limits<double>::infinity();
          const double previous_kkt=current.kkt;
          const double previous_physical_primal=current.primal;
          w.x.swap(tx);w.s.swap(ts);w.z.swap(tz);w.lambda.swap(tl);
          audited_current=false;original_cache_current=false;
          if(!models->study_hybrid_controller_enabled())++stats.inner_steps;
          ++stats.first_step_acceptances;
          stats.current_model_steps=models->study_hybrid_controller_enabled()?hybrid_inner.steps:1;
          stats.maximum_model_steps=std::max(stats.maximum_model_steps,stats.current_model_steps);
          ++stats.majorant_barrier_line_acceptances;
          if(!models->study_hybrid_controller_enabled())++stats.majorant_terminal_single_newton_steps;
          ++result.lazy_newton.accepted_steps;
          if(!w.corrected())++result.lazy_newton.unshifted_accepted_steps;
          if(alpha>=0.99)++result.lazy_newton.full_steps;
          result.lazy_newton.minimum_step=result.lazy_newton.accepted_steps==1?alpha:
              std::min(result.lazy_newton.minimum_step,alpha);
          result.inner_iterations=stats.inner_steps;result.step_inf=step_inf;
          // Commit exactly the x,s,z,lambda tested by the original filter.
          // Slacks were staged through the same canonical affine evaluator
          // used by the filter; no separate update can change their path.
          // Removing the old majorant changes no state or cached value. The
          // sole original evaluation precedes installation of the next model.
          const double previous_dual=current.dual;
          if(models->study_hybrid_enabled())models->study_commit_trial();
          if(models->study_hybrid_controller_enabled()) {
            if(hybrid_accepted_gate.model_relative)hybrid_model_relative_trace(hybrid_accepted_gate,st,"outer-gate",outer+1,alpha,true);
            else hybrid_inner_trace(hybrid_accepted_gate,"outer-gate",outer+1,mu,alpha,true);
          }
          if(models->study_enabled())models->study_accepted_trace(outer+1,alpha,study_equality_proximal);
          models->discard();models->study_rollback_trial();
          ++stats.majorant_original_cache_evaluations;
          w.evaluate();original_cache_current=true;
          current=dca_outer_metrics(w);
          const bool bound_stagnation=gr_bound_watch.observe(
              gr_globalization&&gr_consistent_bound_master&&gr_bound_repair_on_failure&&
              st.standard_r10_policy=="auto"&&st.convexification_study=="off"&&st.study_controller=="r3"&&
              !std::getenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER"),
              result.lazy_newton.accepted_steps,alpha,current.kkt,current.source_stationarity_residual(mu),mu,
              std::max({st.feasibility_tolerance,st.dual_infeasibility_tolerance,st.complementarity_tolerance}));
          if(bound_stagnation&&level)std::cout<<"[GR-BOUND-STAGNATION] accepted_states="<<gr_bound_watch.count
              <<" alpha="<<alpha<<" original_KKT="<<current.kkt
              <<" source_slack_gradient="<<current.source_stationarity_residual(mu)
              <<" mu="<<mu<<" action=existing-consistent-bound-path-next-model"
              <<" accepted_state_unchanged=1 same_actual_GR_Armijo_required=1"<<std::endl;
          const bool historical_stagnation=historical_watch.observe(
              st.standard_equality_watch&&!gr_globalization,result.lazy_newton.accepted_steps,
              alpha,w.regularization,previous_kkt,current.kkt,
              previous_dual,current.dual,std::max({st.feasibility_tolerance,
                  st.stationarity_tolerance,st.dual_infeasibility_tolerance,
                  st.complementarity_tolerance}),historical_relative_step);
          if(historical_stagnation) {
            result.status=Status::HistoricalStagnation;
            result.message="historical short-step stagnation; unused work released for native equality stabilization";
            if(level)std::cout<<"[HISTORICAL-EQUALITY-WATCH] event=stagnation"
                <<" accepted_generation="<<result.lazy_newton.accepted_steps
                <<" consecutive="<<historical_watch.streak<<" alpha="<<alpha
                <<" physical_relative_step="<<historical_relative_step
                <<" physical_step_threshold="<<detail::HistoricalEqualityWatch::physical_step_threshold()
                <<" proximal="<<w.regularization<<" previous_KKT="<<previous_kkt
                <<" current_KKT="<<current.kkt<<" previous_dual="<<previous_dual
                <<" current_dual="<<current.dual
                <<" original_filter_accepted=1 original_KKT_required=1"<<std::endl;
            break;
          }
          if(detail::phase1_reopen_after_dca(models->study_hybrid_controller_enabled(),
              hybrid_accepted_gate.success,decision.accepted,step_inf)) {
            const bool was_spent=failed_linearization_recovery_attempted;
            failed_linearization_recovery_attempted=false;
            ++hybrid_accepted_primal_generation;
            if(level>=2)std::cout<<"[HYBRID-PHASE-I-REARM] primal_generation="<<hybrid_accepted_primal_generation
              <<" previous_spent="<<was_spent<<" primal_displacement="<<step_inf
              <<" reference_reset=0 best_phase1_primal="<<hybrid_last_phase1_primal
              <<" shared_remaining="<<late_phase1_budget.remaining()
              <<" shared_budget="<<late_phase1_budget.limit<<std::endl;
          }
          dual_recovery_history.record(previous_dual,current.dual,st.stationarity_tolerance);
          if(adaptive_globalization&&level>=2)std::cout<<"[MAJORANT-PRIMAL-SPLIT] iter="<<outer+1
              <<" equalities="<<current.equality_primal<<" inequalities="<<current.inequality_primal
              <<" all_original="<<current.primal<<" theta="<<current.barrier.theta<<std::endl;
          if(adaptive_globalization&&deferred_r3_soc) {
            const double defect_floor=64*std::numeric_limits<double>::epsilon()*
                std::max({1.0,from.theta,candidate.theta});
            const bool activated=accepted_equality_defect_measured&&
                nonlinear_step_soc.observe_accepted(alpha,from.theta,
                candidate.theta,accepted_equality_defect_inf,defect_floor);
            if(level>=2)std::cout<<"[MAJORANT-NONLINEAR-STEP] iter="<<outer+1
                <<" alpha="<<alpha<<" window_size="<<nonlinear_step_soc.window_size()
                <<" mean_alpha="<<nonlinear_step_soc.mean_alpha()
                <<" equality_defect_measured="<<accepted_equality_defect_measured
                <<" equality_defect_internal="<<accepted_equality_defect_inf
                <<" defect_floor="<<defect_floor<<" theta_from="<<from.theta
                <<" theta_to="<<candidate.theta<<" soc_enabled="<<nonlinear_step_soc.enabled()
                <<" activated="<<activated<<" reason="<<nonlinear_step_soc.reason()
                <<" original_filter_unchanged=1"<<std::endl;
          }
          if(deferred_r3_soc&&small_step_soc.observe_accepted(alpha)&&level)
            std::cout<<"[MAJORANT-SOC-LATCH] event=activated iter="<<outer+1
                <<" accepted="<<small_step_soc.accepted()<<" streak="<<small_step_soc.streak()
                <<" patience="<<st.majorant_soc_small_step_patience
                <<" threshold="<<st.majorant_soc_small_step_threshold
                <<" original_filter_unchanged=1 correction_and_KKT_tolerances_unchanged=1"<<std::endl;
          filter_proximal_retries=0;
          direction_proximal_retries=0;
          if(lower_proximal_pending&&level)std::cout<<"[LOWER-PROXIMAL-PROBE] event="<<(gr_globalization?"GR-accepted":"filter-accepted")
              <<" actual="<<w.regularization<<" previous_actual="<<lower_proximal_previous_actual
              <<" effective_decrease="<<(w.regularization<lower_proximal_previous_actual)
              <<" primal_displacement="<<step_inf<<" original_filter="<<(!gr_globalization)<<" correction_refinement=0"<<std::endl;
          lower_proximal_pending=false;
          lower_proximal_anchor.reset();
          if(step_inf>0){direction_centering_used=false;lower_proximal_used=false;}
          if(tail){CostScope diagnostic_cost(costs,NewtonCost::Diagnostics);tail->report(w,outer+1,alpha,curvature_rhs);}
          // Near a rank-deficient equality Jacobian, an incremental multiplier
          // correction retains large null-space components. Refit from zero
          // after a regularized step, including the existing 1e-2 feasibility
          // neighborhood used by incremental dual recovery. No x/slack/filter
          // change is made; this is NOT a relaxed feasibility certificate.
          // Retain it only when stationarity
          // improves or stays unchanged while multiplier magnitude decreases.
          if(st.majorant_recalc_lambda && w.mi==0 && w.me>0 && rho_memory>0 &&
              current.primal<=1e-2) {
            const double before=current.dual,old_norm=inf_norm(w.lambda);
            const bool kept=dca_majorant_recalc_lambda(w,current,stats,
                st.majorant_recalc_lambda_max_iterations,start,st.time_limit_seconds,false,true);
            // Retain the accepted-step proximal memory after a multiplier restart.
              // The existing decay policy reduces it; a restart is not evidence of
              // positive curvature (CATENARY requires that distinction).
            if(level>=2)std::cout<<"[MAJORANT-LAMBDA-RESTART] iter="<<outer+1
                <<" accepted="<<kept<<" dual="<<before<<"->"<<current.dual
                <<" lambda_inf="<<old_norm<<"->"<<inf_norm(w.lambda)
                <<" rho_seed="<<rho_memory<<std::endl;
          }
          if(st.majorant_recalc_lambda && w.me>0) {
            dual_stalls=current.dual>=0.9*previous_dual?dual_stalls+1:0;
            dual_window.push_back(current.dual);
            if(dual_window.size()>4)dual_window.erase(dual_window.begin());
            const bool stalled=st.majorant_lambda_trigger=="consecutive"?dual_stalls>=3:
                (dual_window.size()==4&&dual_window.back()>=0.729*dual_window.front());
            const bool dominant=current.dual>10*std::max({current.primal,current.comp,st.stationarity_tolerance});
            // A mixed model can accumulate a large equality-normal dual
            // component while its primal iterate is still infeasible. Waiting
            // for the near-feasible gate then lets that component poison the
            // signed equality curvature indefinitely. Refit from zero only
            // while its initial barrier has never decreased, after the
            // existing stall/cooldown and a scale-free multiplier excess
            // symptom. Once barrier progress has occurred, preserve the
            // historical near-feasible incremental policy. The existing
            // original-stationarity acceptance
            // test is authoritative; no primal/slack/barrier/filter state moves.
            const double old_lambda_norm=inf_norm(w.lambda);
            const bool mixed_restart=w.mi>0&&rho_memory>0&&
                stats.majorant_mu_updates==0&&current.primal>1e-2&&
                std::isfinite(old_lambda_norm)&&
                old_lambda_norm>1000*std::max({1.0,current.primal,current.comp});
            if((current.primal<=1e-2||mixed_restart) && dominant && stalled && outer-last_lambda_attempt>=8) {
              CostScope correction(costs,NewtonCost::Direction);
              last_lambda_attempt=outer;dual_stalls=0;
              const double before=current.dual;
              const bool kept=dca_majorant_recalc_lambda(w,current,stats,st.majorant_recalc_lambda_max_iterations,start,st.time_limit_seconds,
                  mixed_restart?false:st.majorant_lambda_interpolation,mixed_restart);
              dual_window.clear();
              if(kept&&st.majorant_lambda_rho_retry)lambda_rho_pending=true;
              if(level>=2)std::cout<<"[MAJORANT-LAMBDA] iter="<<outer+1<<" accepted="<<kept
                  <<" dual="<<before<<"->"<<current.dual<<" alpha="<<stats.majorant_lambda_last_alpha
                  <<" mixed_restart="<<mixed_restart<<" lambda_inf="<<old_lambda_norm<<"->"<<inf_norm(w.lambda)
                  <<" primal_unchanged=1 slack_unchanged=1 barrier_unchanged=1 filter_unchanged=1"
                  <<" seconds_total="<<stats.majorant_lambda_seconds<<std::endl;
            }
          }
          stats.majorant_barrier_residual=current.residual(mu);
          if(level>=2) {
            CostScope display(costs,NewtonCost::Diagnostics);
            std::cout<<(gr_globalization?"[MAJORANT-GR] iter=":"[MAJORANT-FILTER] iter=")<<outer+1<<" accepted=1 alpha="<<alpha
              <<" direction="<<(accepted_kkt_soc?"frozen-KKT-SOC":square_feasibility?"square-feasibility-normal":
                  (models->study_hybrid_controller_enabled()?"inexact-convex-barrier":
                  (curvature_refined?"lazy-curvature-refined":"single-newton")))<<" original_filter="<<(!gr_globalization)
              <<" type="<<(gr_globalization?"Gill-Robinson-merit":(decision.objective?"barrier-objective":"slack-feasibility"))
              <<" theta="<<candidate.theta<<" phi_change="<<candidate.phi_change
              <<" original_masked_bound_sides="<<candidate.omitted_bound_sides
              <<" original_masked_bound_violation="<<candidate.omitted_bound_max
              <<" original_masked_bound_theta="<<candidate.omitted_bound_theta
              <<" original_primal="<<current.primal<<" original_dual="<<current.dual
              <<" original_compl="<<current.comp<<" mu="<<mu<<"\n";
          }
          if(!fixed_centering.active()&&detail::native_anchor_capture_wanted(current.primal,st.original_private_phase)) {
            CostScope capture(costs,NewtonCost::Certificate);
            Vec point=w.fixed;
            for(int j=0;j<w.n;++j)point[w.free[j]]=w.x[j];
            detail::native_anchor_observe(problem,st.rank_presolve_context,point,
                current.primal,st.original_private_phase,detail::NativeAnchorOrigin::AcceptedR3Step);
          }
          if(checkpoint_majorant_depth==1&&st.accepted_checkpoint_callback&&!fixed_centering.active()) {
            try {
              Result snapshot;w.export_state(snapshot);
              st.accepted_checkpoint_callback(problem,snapshot,st.rank_presolve_context,outer+1,seconds(start));
            }catch(const std::exception& error) {
              std::cerr<<"[ACCEPTED-CHECKPOINT] written=0 previous_file_preserved=1 reason="<<error.what()<<std::endl;
            }
          }
          if(plausible_original(current)&&audit_original()) {
            // An original KKT point is a valid committed phase endpoint.
            fixed_centering_anchor.reset();fixed_centering.remaining=0;
            result.status=Status::Solved;
            result.message=gr_globalization?"GR-merit-accepted point satisfies independent original QCQP KKT":
                "filter-accepted point satisfies independent original QCQP KKT";break;
          }
          if(outer+1>=st.max_outer_iterations)break;
          if(seconds(start)>=st.time_limit_seconds) {
            result.status=Status::TimeLimit;result.message="outer-majorant time limit after acceptance";break;
          }
          // The minimal spectral models remove artificial null-space
          // curvature. Until original primal feasibility and complementarity
          // are certified, use the complete primal-dual barrier residual:
          // primal/centering progress alone can otherwise shrink mu while
          // the new less-damped multiplier iterates diverge. After that gate,
          // preserve the existing fast dual-tail barrier update. This is one
          // conditional outer update, with the same strict original audit.
          continue_barrier_after_original_acceptance(gr_globalization?"original-GR-merit-DCA":"original-filtered-DCA");
          // Keep the diagnostic phase's parameter/model control fixed; no
          // continuation, proximal feedback or startup restart is interleaved.
          // A rolled-back phase must not feed its last staged alpha to those
          // controls after restoring the accepted entry state and histories.
          if(fixed_centering.active()||fixed_centering_rolled_back)continue;
          // R9-AUTO-STARTUP-R2: retain R1's startup decision and every successful
          // step until a prolonged, low-barrier primal-dual stall is observed.
          // One cold-primal central retry only, inside this SAME solve.
          if(auto_startup_retry.observe(auto_retry_eligible && !central_first,
                alpha,current.primal,current.dual,current.comp,mu,auto_initial_mu,
                st.feasibility_tolerance,st.dual_infeasibility_tolerance,
                st.max_outer_iterations-(outer+1),st.time_limit_seconds-seconds(start))) {
            CostScope retry_cost(costs,NewtonCost::Globalization);
            const double old_primal=current.primal,old_dual=current.dual,old_mu=mu;
            const int observed=auto_startup_retry.streak();
            auto_startup_retry.mark_used();
            models->discard();models->study_rollback_trial();
            // Compute a candidate state transactionally at the original x0.
            const Vec saved_x=w.x,saved_s=w.s,saved_z=w.z,saved_lambda=w.lambda;
            w.x=auto_frozen_x;
            w.evaluate();
            bool valid=finite(w.x)&&std::isfinite(w.objective.value)&&finite(w.objective.gradient);
            Vec central_s=w.s,central_z=w.z;
            for(int e=0;e<w.me;++e)
              valid=valid&&std::isfinite(w.equalities[e].value)&&finite(w.equalities[e].gradient);
            for(int i=0;i<w.mi;++i) {
              valid=valid&&std::isfinite(w.inequalities[i].value)&&finite(w.inequalities[i].gradient);
              // Exactly the previous frozen-start initialization with mu0=1.
              central_s[i]=std::max(1.0,-w.inequalities[i].value);
              central_z[i]=w.active[i]?std::min(1.0,1.0/central_s[i]):w.z[i];
            }
            valid=valid&&finite(central_s)&&finite(central_z);
            if(valid) {
              w.s.swap(central_s);w.z.swap(central_z);
              std::fill(w.lambda.begin(),w.lambda.end(),0.0);
              w.evaluate();
              const auto candidate_metrics=dca_outer_metrics(w);
              valid=std::isfinite(candidate_metrics.primal)&&std::isfinite(candidate_metrics.dual)&&
                    std::isfinite(candidate_metrics.comp)&&std::isfinite(candidate_metrics.product_mean);
              if(valid)current=candidate_metrics;
            }
            if(!valid) {
              w.x=saved_x;w.s=saved_s;w.z=saved_z;w.lambda=saved_lambda;
              w.evaluate();current=dca_outer_metrics(w);
              original_cache_current=true;
              if(level)std::cout<<"[AUTO-STARTUP-RETRY] accepted=0 reason=nonfinite-central-state"
                  <<" previous_state_restored=1 budget=1"<<std::endl;
            } else {
              central_first=true;
              w.auto_startup.central_first=true;w.auto_startup.initial_mu=1.0;
              w.auto_startup.reason="low-mu-stall-central-retry";
              mu=current.active_count?std::max(barrier_floor,current.product_mean):0;sync_outer_barrier();
              if(st.majorant_trial_soc_arc)soc_arc_barrier_event.reset();
            if(st.majorant_soc_defect_control){soc_defect_epoch=false;soc_defect_geometry_inspected=false;
              soc_defect_geometry=detail::SocDefectGeometry{};}
              initial_original_comp=current.comp;
              nonlinear_step_soc.clear_window();
              filter.reset();rho_memory=0;filter_proximal_retries=0;lambda_rho_pending=false;
              dual_recovery_history.reset();dual_stalls=0;dual_window.clear();last_lambda_attempt=outer-8;
              audited_current=false;original_cache_current=true;recovered_state_requires_audit=true;
              stats.majorant_barrier_mu=mu;stats.majorant_barrier_residual=current.residual(mu);
              if(level)std::cout<<"[AUTO-STARTUP-RETRY] accepted=1 iter="<<outer+1
                  <<" tiny_steps="<<observed<<" alpha="<<alpha
                  <<" mu="<<old_mu<<"->"<<mu<<" primal="<<old_primal<<"->"<<current.primal
                  <<" dual="<<old_dual<<"->"<<current.dual
                  <<" reset_primal=frozen-x0 restart_budget=1 global_iterations_preserved=1"
                  <<" global_time_preserved=1 recovery_enabled="<<st.restoration_enabled<<std::endl;
              continue;
            }
          }
          models->objective_accepted_step();
          // A technically accepted vanishing step is not useful globalization.
          // Retain the exact accepted point and filter, but reduce the next
          // auxiliary Newton step using an EXPLICIT proximal metric. Otherwise
          // the filter ceiling can trap the method without ever rejecting all
          // trials. Original direction and terminal audits are unchanged.
          const bool legacy_vanishing_stall=detail::small_step_proximal_trigger(
              !square_feasibility && w.mi>0 && !st.lazy_newton_atomic_regularization,
              alpha,dual_recovery_history.proximal_steps,current.primal,st.feasibility_tolerance);
          const bool gr_short_stall=DcaGrShortStepProximal::trigger(
              gr_short_step_prox&&!square_feasibility&&w.mi>0&&!st.lazy_newton_atomic_regularization,
              alpha,dual_recovery_history.proximal_steps,previous_physical_primal,current.primal);
          const bool vanishing_stall=legacy_vanishing_stall||gr_short_stall;
          const bool certified_full_objective_decay=detail::hybrid_full_objective_proximal_decay(
              models->study_hybrid_controller_enabled(),hybrid_accepted_gate.model_relative,
              hybrid_accepted_gate.success&&hybrid_accepted_gate.model_gate.passed,
              decision.accepted,decision.objective,alpha,rho_memory,study_margin_proximal_floor);
          detail::RhoDecayFeasibilityGate rho_feasibility_gate;
          if(st.majorant_rho_feasibility_guard&&!vanishing_stall&&rho_memory>0&&
              (current.kkt<=previous_kkt||certified_full_objective_decay)) {
            rho_feasibility_gate=detail::rho_decay_feasibility_gate(
                previous_physical_primal,current.primal,st.feasibility_tolerance);
            if(level>=2)std::cout<<std::setprecision(17)
                <<"[MAJORANT-RHO-FEASIBILITY-GATE] iter="<<outer+1
                <<" decision="<<(rho_feasibility_gate.allowed?"allow-decay":"hold")
                <<" reason="<<rho_feasibility_gate.reason
                <<" previous_physical_primal="<<previous_physical_primal
                <<" current_physical_primal="<<current.primal
                <<" allowed_physical_primal="<<rho_feasibility_gate.limit
                <<" rho_memory="<<rho_memory<<" alpha="<<alpha
                <<" original_KKT_before="<<previous_kkt<<" original_KKT_after="<<current.kkt
                <<" physical_residual_only=1 original_filter_unchanged=1"<<std::endl;
          }
          if(vanishing_stall) {
            const double historical_next=detail::small_step_proximal_proposal(rho_memory,
                inf_norm(w.rd),inf_norm(w.x),st.lazy_newton_regularization_initial);
            const auto proximal_policy=detail::gr_globalization_proximal_growth(
                gr_globalization,st.gr_proximal_policy,rho_memory,historical_next);
            const double next=proximal_policy.effective;
            if(gr_globalization&&level)std::cout<<std::setprecision(17)
                <<"[GR-PROXIMAL-POLICY] event=accepted-short-step iter="<<outer+1
                <<" policy="<<st.gr_proximal_policy<<" old_memory="<<rho_memory
                <<" proposed="<<historical_next<<" effective="<<next
                <<" blocked="<<proximal_policy.blocked<<" reason="<<proximal_policy.reason
                <<" alpha="<<alpha<<" original_globalization_unchanged=1"
                <<" numerical_inertia_recovery_unchanged=1"<<std::endl;
            if(std::isfinite(next)&&next>rho_memory&&next<=1e12) {
              if(level)std::cout<<"[SMALL-STEP-PROXIMAL] iter="<<outer+1
                <<" alpha="<<alpha<<" dual="<<current.dual
                <<" rho="<<rho_memory<<"->"<<next
                <<" trigger="<<(gr_short_stall?"GR-weak-primal-and-dual-short-step":"legacy-feasible-short-step")
                <<" weak_dual_steps="<<dual_recovery_history.proximal_steps
                <<" previous_physical_primal="<<previous_physical_primal<<" current_physical_primal="<<current.primal
                <<" accepted_state_unchanged=1 actual_original_globalization_required=1"<<std::endl;
              rho_memory=next;dual_recovery_history.proximal_proposed();
            }
          } else if(rho_memory>0&&(current.kkt<=previous_kkt||certified_full_objective_decay)&&
              (!st.majorant_rho_feasibility_guard||rho_feasibility_gate.allowed)) {
            const double before=rho_memory;
            rho_memory*=std::pow(0.5,alpha);
            if(rho_memory<=1e-10)rho_memory=0;
            ++result.lazy_newton.regularization_decreases;
            if(models->study_hybrid_controller_enabled()&&level>=2)
              std::cout<<std::setprecision(17)<<"[HYBRID-PROXIMAL-DECAY] iter="<<outer+1
                <<" reason="<<(certified_full_objective_decay?
                    "full-certified-objective-step":"original-KKT-nonincrease")
                <<" old_memory="<<before<<" new_memory="<<rho_memory
                <<" actual_model_tau="<<hybrid_accepted_gate.tau_prox
                <<" certified_floor="<<study_margin_proximal_floor
                <<" alpha="<<alpha<<" model_certificate="<<hybrid_accepted_gate.success
                <<" original_filter="<<decision.accepted<<" objective_condition="<<decision.objective
                <<" original_KKT_before="<<previous_kkt<<" original_KKT_after="<<current.kkt
                <<" correction_refinement=0 next_margin_recertified=1 original_filter_required=1"
                <<" accepted_metric_digest="<<models->study_metric_digest()<<std::endl;
          }
        }
      }
    }
  } catch(const std::exception& e) {
    result.status=Status::NumericalError;result.message=std::string("outer-majorant failure: ")+e.what();
    if(stats.majorant_last_rejection.empty())stats.majorant_last_rejection="evaluation-or-model";
  }
  // Export the last accepted state. A failed/limited iteration still receives
  // one independent original audit; an existing audit of this state is reused.
  if(owner) {
    try {
      // A rejected direction or an exception can leave the old model
      // installed at the last accepted point. Always release it, including
      // partial installations, without altering the accepted primal-dual state.
      if(models){models->discard();models->study_rollback_trial();}
      if(!original_cache_current) {
        owner->evaluate();original_cache_current=true;
      }
      if(!audited_current&&audit_original()) {
        result.status=Status::Solved;result.message="last accepted point satisfies independent original QCQP KKT";
      }
    } catch(const std::exception& e) {
      result.status=Status::NumericalError;result.message=std::string("outer-majorant final audit failed: ")+e.what();
      result.termination_audit_valid=false;owner->export_state(result);
    }
    result.lazy_newton.final_regularization=owner->regularization;
    result.lazy_newton.final_active_constraints=std::count(owner->active.begin(),owner->active.end(),1);
    result.lazy_newton.symbolic_analyses=owner->factor->symbolic_analyses();
    result.lazy_newton.symbolic_reuses=owner->factor->symbolic_reuses();
    result.lazy_newton.scatter_builds=owner->factor->numeric_scatter_builds();
    result.lazy_newton.scatter_reuses=owner->factor->numeric_scatter_reuses();
    if(owner->pivoted)stats.pivoted_backend=owner->pivoted->name();
  }
  if(!result.termination_certificate_rescued)
    result.termination_multiplier_source="inequality-majorants-original-extended-audit";
  result.kkt_factorizations=result.lazy_newton.factor_attempts;
  if(level)std::cout<<"[MAJORANT-OPTIONS] restoration="<<st.majorant_initial_restoration<<" retraction="<<st.majorant_conditional_retraction<<" hybrid_soc="<<hybrid_mode<<" gr_defined_soc="<<gr_defined_soc<<" gr_kkt_soc="<<gr_kkt_soc<<" consistent_bound_master="<<gr_consistent_bound_slacks<<" consistent_bound_trigger="<<gr_bound_trigger<<" soc_attempts="<<result.soc_attempts<<" soc_accepted="<<result.soc_accepted_steps<<" soc_rejected="<<result.soc_rejected_steps<<std::endl;
  if(level) {CostScope display(costs,NewtonCost::Diagnostics);std::cout<<"\nEXIT: "<<result.message<<std::endl;}
  {CostScope cleanup(costs,NewtonCost::Cleanup);models.reset();owner.reset();}
  {
    CostScope certificate(costs,NewtonCost::Certificate);
    detail::terminal_original_multiplier_fit(problem,requested,result,std::min(5.,st.time_limit_seconds-seconds(start)));
  }
  costs.finish();result.elapsed_seconds=seconds(start);
  return result;
}
