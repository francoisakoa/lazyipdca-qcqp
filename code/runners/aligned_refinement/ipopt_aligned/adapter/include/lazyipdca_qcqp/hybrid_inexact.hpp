#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

// Pure acceptance arithmetic, shared with independent small unit tests. The
// quantities refer to a frozen, fixed-mu convex barrier subproblem. They must
// never be interpreted as a zero-barrier original-QCQP certificate.
namespace lazyipdca::qcqp::hybrid_inner {
struct SlackBalance {
  double value=0;
  long double before=std::numeric_limits<long double>::infinity(),after=before;
  bool changed=false;
};
// A bordered Newton solve supplies dz separately. Reconstruct its auxiliary
// ds without privileging one of the ORIGINAL equations a+ds=0 and b+z*ds=0.
// For z>0 their minimax residual is attained at ds=-(a+b)/(1+z). Compare the
// actual represented old/root/adjacent doubles in extended precision. Only
// a strict improvement of the unchanged expanded row-pair audit is returned.
inline SlackBalance balance_slack_direction(long double a,long double b,
    double z,double old) {
  SlackBalance out;out.value=old;
  if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(z)||z<=0||!std::isfinite(old))
    return out;
  auto error=[&](double value) {
    return std::max(std::abs(a+value),std::abs(b+static_cast<long double>(z)*value));
  };
  out.before=out.after=error(old);
  const double balanced=static_cast<double>(-(a+b)/(1.L+z));
  for(double value:{balanced,std::nextafter(balanced,-INFINITY),std::nextafter(balanced,INFINITY)})
    if(std::isfinite(value)&&error(value)<out.after){out.value=value;out.after=error(value);}
  out.changed=out.value!=old;return out;
}
// Merit penalties belong to one fixed tangent model, not to the lifetime of
// the original nonconvex solve. The multiplier norm is in the SAME internal
// units as the residuals multiplied by sigma. This is a local exact-L1
// heuristic; the independent model certificate is still mandatory.
inline double model_penalty_seed(double multiplier_norm) {
  if(!std::isfinite(multiplier_norm)||multiplier_norm<0)
    throw std::invalid_argument("invalid model multiplier norm");
  const double seed=std::nextafter(std::max(1.0,multiplier_norm+1.0),
      std::numeric_limits<double>::infinity());
  if(!std::isfinite(seed))throw std::runtime_error("exact-L1 dual penalty overflow");
  return seed;
}
inline double descent_penalty(double current,double smooth_slope,double theta) {
  if(!std::isfinite(current)||current<0||!std::isfinite(smooth_slope)||
     !std::isfinite(theta)||theta<0)
    throw std::invalid_argument("invalid exact-L1 penalty arguments");
  if(theta==0||static_cast<long double>(smooth_slope)-
      static_cast<long double>(current)*theta<-1e-8L*theta)return current;
  // sigma >= 2 max(s,0)/theta + 1 gives s-sigma*theta <= -|s|-theta.
  // Adding 1 BEFORE division instead demands an absolute decrease of 1,
  // making sigma diverge even when both s and theta vanish together.
  const long double required=2*std::max(0.L,static_cast<long double>(smooth_slope))/theta+1;
  if(required>std::numeric_limits<double>::max())
    throw std::runtime_error("exact-L1 descent penalty overflow");
  const double rounded=std::nextafter(static_cast<double>(required),
      std::numeric_limits<double>::infinity());
  if(!std::isfinite(rounded))throw std::runtime_error("exact-L1 descent penalty overflow");
  return std::max(current,rounded);
}
struct NormalColumn {
  std::vector<std::pair<int,double>> values;
  double multiplier=0;
  bool nonnegative=false;
};
struct NormalFit {
  std::vector<double> residual,multipliers;
  double residual_l2=std::numeric_limits<double>::infinity();
  int sweeps=0;
};
// Constrained coordinate least squares. Any returned nonnegative multiplier
// vector is admissible for the SELECTED numerical active set. With a nonzero
// activity tolerance this is a finite-activity KKT residual, not the exact
// normal-cone distance at a genuinely interior point.
inline NormalFit fit_normal(const std::vector<double>& gradient,
    const std::vector<NormalColumn>& columns,int maximum_sweeps=100) {
  NormalFit fit;fit.residual=gradient;fit.multipliers.reserve(columns.size());
  std::vector<long double> norms;norms.reserve(columns.size());
  for(const auto& c:columns) {
    double m=c.nonnegative?std::max(0.0,c.multiplier):c.multiplier;
    fit.multipliers.push_back(m);long double norm=0;
    for(auto a:c.values) {
      if(a.first<0||std::size_t(a.first)>=gradient.size()||!std::isfinite(a.second))
        throw std::invalid_argument("invalid normal-cone column");
      norm+=static_cast<long double>(a.second)*a.second;
      fit.residual[a.first]+=m*a.second;
    }
    norms.push_back(norm);
  }
  for(int sweep=0;sweep<maximum_sweeps;++sweep) {
    long double largest_change=0;
    for(std::size_t k=0;k<columns.size();++k)if(norms[k]>0) {
      const auto& c=columns[k];long double product=0;
      for(auto a:c.values)product+=static_cast<long double>(a.second)*fit.residual[a.first];
      const double old=fit.multipliers[k];double next=old-static_cast<double>(product/norms[k]);
      if(c.nonnegative)next=std::max(0.0,next);
      const double delta=next-old;fit.multipliers[k]=next;
      largest_change=std::max(largest_change,std::abs(static_cast<long double>(delta))*std::sqrt(norms[k]));
      for(auto a:c.values)fit.residual[a.first]+=delta*a.second;
    }
    fit.sweeps=sweep+1;
    if(largest_change<=32*std::numeric_limits<double>::epsilon())break;
  }
  // Acceptance uses a fresh extended-precision reconstruction from the final
  // multipliers, never the incrementally updated coordinate-descent scratch.
  std::vector<long double> reconstructed(gradient.begin(),gradient.end());
  for(std::size_t k=0;k<columns.size();++k)for(auto a:columns[k].values)
    reconstructed[a.first]+=static_cast<long double>(fit.multipliers[k])*a.second;
  long double square=0;
  for(std::size_t j=0;j<reconstructed.size();++j) {
    fit.residual[j]=static_cast<double>(reconstructed[j]);square+=reconstructed[j]*reconstructed[j];
  }
  fit.residual_l2=static_cast<double>(std::sqrt(square));return fit;
}
struct EpsilonNormal {
  NormalFit fit;
  double complementarity=std::numeric_limits<double>::infinity();
  double minimum_multiplier=0;
  bool dual_feasible=false;
};
// Fresh approximate-KKT certificate using the supplied multipliers on ALL
// inequalities. This is an epsilon-normal certificate, not an exact normal
// cone at an interior point. In particular, negative multipliers are refused
// before fit_normal can apply its nonnegative initialization projection.
inline EpsilonNormal epsilon_normal(const std::vector<double>& gradient,
    const std::vector<NormalColumn>& columns,const std::vector<double>& values) {
  EpsilonNormal certificate;
  if(columns.size()!=values.size())return certificate;
  for(double v:gradient)if(!std::isfinite(v))return certificate;
  long double gap=0;
  for(std::size_t k=0;k<columns.size();++k) {
    const auto& column=columns[k];const double multiplier=column.multiplier;
    if(!std::isfinite(multiplier)||!std::isfinite(values[k]))return certificate;
    for(auto entry:column.values)
      if(entry.first<0||std::size_t(entry.first)>=gradient.size()||!std::isfinite(entry.second))
        return certificate;
    if(column.nonnegative) {
      certificate.minimum_multiplier=std::min(certificate.minimum_multiplier,multiplier);
      if(multiplier<0)return certificate;
      gap+=std::abs(static_cast<long double>(multiplier)*values[k]);
    }
  }
  certificate.fit=fit_normal(gradient,columns,0);
  certificate.complementarity=static_cast<double>(gap);
  certificate.dual_feasible=std::isfinite(certificate.fit.residual_l2)&&
      std::isfinite(certificate.complementarity);
  return certificate;
}
struct ModelGate {
  double margin=0,kappa=0,stationarity_target=0,feasibility_target=0,complementarity_target=0;
  bool positive_margin=false,stationarity_ok=false,feasibility_ok=false,
       complementarity_ok=false,model_decrease=false,passed=false;
};
inline ModelGate check_model(double residual_l2,double step_l2,double margin,
    double kappa_fraction,double stationarity_floor,double feasibility,
    double feasibility_target,double complementarity,double complementarity_target,
    double model_value,double anchor_model_value) {
  ModelGate gate;gate.margin=margin;gate.feasibility_target=feasibility_target;
  gate.complementarity_target=complementarity_target;
  for(double v:{residual_l2,step_l2,margin,stationarity_floor,feasibility,
      feasibility_target,complementarity,complementarity_target})
    if(!std::isfinite(v)||v<0)return gate;
  if(!std::isfinite(kappa_fraction)||kappa_fraction<=0||kappa_fraction>=.5||
      !std::isfinite(model_value)||!std::isfinite(anchor_model_value))return gate;
  gate.positive_margin=margin>0;gate.kappa=kappa_fraction*margin;
  gate.stationarity_target=stationarity_floor+gate.kappa*step_l2;
  gate.stationarity_ok=residual_l2<=gate.stationarity_target;
  gate.feasibility_ok=feasibility<=feasibility_target;
  gate.complementarity_ok=complementarity<=complementarity_target;
  const double roundoff=64*std::numeric_limits<double>::epsilon()*
      std::max({1.0,std::abs(model_value),std::abs(anchor_model_value)});
  gate.model_decrease=model_value<=anchor_model_value+roundoff;
  gate.passed=gate.positive_margin&&gate.stationarity_ok&&gate.feasibility_ok&&
      gate.complementarity_ok&&gate.model_decrease;return gate;
}
// Numerical centrality target, not an original-QCQP acceptance tolerance.
// On the central path g=-s, sum |z*g| / sf = m*mu/sf. Reserving one
// quarter of the smallest allowed model complementarity budget prevents an
// unnecessary approach to zero while leaving the actual model gate intact.
struct ComplementarityBarrierFloor {
  double value=0;
  bool underflow=false,saturated=false;
};
inline ComplementarityBarrierFloor complementarity_barrier_floor(
    double objective_scale,double complementarity_floor,int active_count,
    double fraction=.25) {
  if(!std::isfinite(objective_scale)||objective_scale<=0||
     !std::isfinite(complementarity_floor)||complementarity_floor<0||
     active_count<0||!std::isfinite(fraction)||fraction<=0||fraction>=1)
    throw std::invalid_argument("invalid complementarity barrier floor controls");
  ComplementarityBarrierFloor out;
  if(!active_count||complementarity_floor==0)return out;
  // Multiply mantissas, then combine exponents: no intermediate overflow
  // or underflow even where long double and double have the same range.
  int scale_exponent=0,tolerance_exponent=0,fraction_exponent=0;
  const long double scale_mantissa=std::frexp(objective_scale,&scale_exponent);
  const long double tolerance_mantissa=std::frexp(complementarity_floor,&tolerance_exponent);
  const long double fraction_mantissa=std::frexp(fraction,&fraction_exponent);
  const long double desired=std::ldexp(scale_mantissa*tolerance_mantissa*
      fraction_mantissa/active_count,
      scale_exponent+tolerance_exponent+fraction_exponent);
  if(desired>std::numeric_limits<double>::max()) {
    out.value=std::numeric_limits<double>::max();out.saturated=true;return out;
  }
  // Conservative evaluation also on platforms where long double == double.
  // Never round a tiny positive budget UP to min()/denorm_min().
  const long double guarded=desired*(1-8*std::numeric_limits<long double>::epsilon());
  out.value=static_cast<double>(guarded);
  if(out.value>0)out.value=std::nextafter(out.value,0.0);
  out.underflow=out.value==0;
  return out;
}
struct MehrotraTarget {double affine_mu=0,sigma=0,sigma_used=0,target=0;};
struct CentralRestart {
  double target_mu=0,normalized_slack_floor=0;
  bool needed=false;
};
// A fixed tangent model may inherit a complementarity scale much smaller
// than its primal residual. This is only a private interior initialization;
// it does not alter the model, its solution, or any acceptance tolerance.
inline CentralRestart central_restart(double primal,double objective_scale,
    double feasibility_floor,double mean_mu,bool numerical_failure=false) {
  CentralRestart plan;
  if(!std::isfinite(primal)||primal<0||!std::isfinite(objective_scale)||
     objective_scale<=0||!std::isfinite(feasibility_floor)||feasibility_floor<0||
     !std::isfinite(mean_mu)||mean_mu<0)return plan;
  const double normalized_mu=std::max(primal,feasibility_floor);
  plan.target_mu=objective_scale*normalized_mu;
  if(numerical_failure)plan.target_mu=std::max(plan.target_mu,mean_mu);
  if(!(plan.target_mu>0)||!std::isfinite(plan.target_mu)||
     (!numerical_failure&&mean_mu>=plan.target_mu))
    return plan;
  plan.normalized_slack_floor=std::sqrt(plan.target_mu/objective_scale);
  plan.needed=std::isfinite(plan.normalized_slack_floor)&&plan.normalized_slack_floor>0;
  return plan;
}
struct ProgressGate {
  bool residual_contraction=false,merit_decrease=false,passed=false;
};
// Accumulate EVERY scaled nonlinear KKT residual component. Taking the
// maximum of three family infinity norms can hide a large aggregate growth
// in feasibility while one dominant dual component decreases slightly.
struct FullResidualL2 {
  long double square=0;
  bool valid=true;
  void add(long double component,double scale) {
    if(!std::isfinite(component)||!std::isfinite(scale)||!(scale>0)) {
      valid=false;return;
    }
    const long double normalized=component/scale;
    square+=normalized*normalized;
    valid=valid&&std::isfinite(square);
  }
  double norm() const {
    return valid?static_cast<double>(std::sqrt(square)):
        std::numeric_limits<double>::infinity();
  }
};
struct InitialInteriorStart {
  std::vector<double> slack,dual;
  double target_mu=0;
  bool eligible=false;
};
// Initialization proposal, not an accepted descent step. Call at most once
// per physical solve; a non-interior first model must also consume that event.
// Repeating mean_mu/(-g) near later active sides can create huge duals.
inline InitialInteriorStart initial_interior_start(const std::vector<double>& values,
    const std::vector<double>& slack,const std::vector<double>& dual,
    const std::vector<int>& active) {
  InitialInteriorStart plan;
  if(values.size()!=slack.size()||values.size()!=dual.size()||active.empty())return plan;
  std::vector<unsigned char> seen(values.size(),0);long double mean=0;
  for(int i:active) {
    if(i<0||std::size_t(i)>=values.size()||seen[i]||!std::isfinite(values[i])||
       !(values[i]<0)||!std::isfinite(slack[i])||!std::isfinite(dual[i])||
       !(slack[i]>0&&dual[i]>0))return plan;
    seen[i]=1;mean+=static_cast<long double>(slack[i])*dual[i];
  }
  mean/=active.size();plan.target_mu=static_cast<double>(mean);
  if(!std::isfinite(plan.target_mu)||!(plan.target_mu>0))return plan;
  plan.slack=slack;plan.dual=dual;
  for(int i:active) {
    plan.slack[i]=-values[i];plan.dual[i]=static_cast<double>(mean/plan.slack[i]);
    if(!std::isfinite(plan.dual[i])||!(plan.dual[i]>0))return InitialInteriorStart{};
  }
  plan.eligible=true;return plan;
}
// Rounding tolerance can protect an Armijo comparison, but cannot manufacture
// progress. In particular, a vanishing step whose residual and merit both
// round to their previous values must not consume another inner iteration.
inline ProgressGate check_progress(bool state_changed,double residual_before,
    double residual_after,double merit_before,double merit_after,double slope,
    double alpha,double armijo,double rounding,double residual_fraction=1) {
  ProgressGate gate;
  for(double v:{residual_before,residual_after,merit_before,merit_after,slope,alpha,armijo,rounding,residual_fraction})
    if(!std::isfinite(v))return gate;
  if(!state_changed||residual_before<0||residual_after<0||
     alpha<=0||alpha>1||armijo<=0||armijo>=1||rounding<0||
     residual_fraction<=0||residual_fraction>1)return gate;
  // alpha is the line-search multiplier. A fraction-to-boundary may have
  // already shortened the Newton direction. Residual decrease must use the
  // actual step; the caller's merit slope already includes its primal fraction.
  gate.residual_contraction=residual_after<residual_before&&
      residual_after<=(1-1e-4*alpha*residual_fraction)*residual_before;
  gate.merit_decrease=slope<0&&merit_after<merit_before&&
      merit_after<=merit_before+armijo*alpha*slope+rounding;
  gate.passed=gate.residual_contraction||gate.merit_decrease;
  return gate;
}
inline MehrotraTarget mehrotra_target(const std::vector<double>& slack,
    const std::vector<double>& dual,const std::vector<double>& ds,
    const std::vector<double>& dz,const std::vector<int>& active,
    double alpha_primal,double alpha_dual,double previous_target_mu,double decay,
    double barrier_floor=0) {
  MehrotraTarget out;long double mean=0,affine=0;
  // A previous centrality target is NOT a bound on the new target: after an
  // outer model change the current complementarity/feasibility can be larger.
  (void)previous_target_mu;
  if(active.empty())return out;
  for(int i:active) {
    mean+=static_cast<long double>(slack.at(i))*dual.at(i);
    affine+=(static_cast<long double>(slack.at(i))+alpha_primal*ds.at(i))*
        (static_cast<long double>(dual.at(i))+alpha_dual*dz.at(i));
  }
  mean/=active.size();affine/=active.size();out.affine_mu=std::max(0.0,static_cast<double>(affine));
  if(mean>0)out.sigma=std::clamp(std::pow(out.affine_mu/static_cast<double>(mean),3),0.0,1.0);
  out.sigma_used=std::min(out.sigma,decay);
  out.target=std::max(barrier_floor,out.sigma_used*static_cast<double>(mean));
  // A representability safeguard, not a convergence/acceptance tolerance.
  out.target=std::max(std::numeric_limits<double>::min(),out.target);return out;
}
// Algorithmic proximal stabilization is distinct from the corrections of f
// and g_i.  E selects the coordinates retained by affine elimination, exactly
// matching the primal diagonal of the reduced auxiliary KKT matrix.
inline double retained_proximal(const std::vector<double>& point,
    const std::vector<double>& anchor,const std::vector<int>& retained,
    double tau,std::vector<double>* gradient=nullptr) {
  if(point.size()!=anchor.size()||(gradient&&gradient->size()!=point.size())||
     !std::isfinite(tau)||tau<0)
    throw std::invalid_argument("invalid frozen retained-coordinate proximal term");
  long double energy=0;
  for(int j:retained) {
    if(j<0||std::size_t(j)>=point.size())
      throw std::invalid_argument("invalid frozen proximal coordinate");
    const long double d=static_cast<long double>(point[j])-anchor[j];
    const long double g=static_cast<long double>(tau)*d;
    energy+=0.5L*g*d;
    if(gradient)(*gradient)[j]+=static_cast<double>(g);
  }
  return static_cast<double>(energy);
}
inline double proximal_curvature_deficit(double equality_absolute_bound,double objective_lower) {
  const long double deficit=std::max(0.0L,static_cast<long double>(equality_absolute_bound)-objective_lower);
  double rounded=static_cast<double>(deficit);
  if(static_cast<long double>(rounded)<deficit)rounded=std::nextafter(rounded,INFINITY);
  return rounded;
}
struct Residual {
  double dual=0,primal=0,centering=0;
  bool finite() const {
    return std::isfinite(dual)&&std::isfinite(primal)&&std::isfinite(centering)&&
      dual>=0&&primal>=0&&centering>=0;
  }
  double norm() const {return std::max({dual,primal,centering});}
};
struct Gate {
  double dual_target=0,primal_target=0,centering_target=0;
  bool relative_ok=false,model_decrease=false,passed=false;
};
inline Gate check(const Residual& value,const Residual& initial,double step,
    double relative,double absolute,double model_merit,double anchor_merit,
    double scaled_mu=0) {
  Gate gate;
  if(!value.finite()||!initial.finite()||!std::isfinite(step)||step<0||
     !std::isfinite(relative)||relative<=0||relative>=1||
     !std::isfinite(absolute)||absolute<0||!std::isfinite(model_merit)||
     !std::isfinite(anchor_merit)||!std::isfinite(scaled_mu)||scaled_mu<0)return gate;
  gate.dual_target=absolute+relative*step;
  gate.primal_target=absolute+relative*initial.primal;
  // An initially central point does not require machine-exact centrality
  // throughout an INEXACT fixed-mu model solve. Use a central-neighbourhood
  // forcing term O(mu), vanishing with mu; original KKT tolerances stay intact.
  gate.centering_target=absolute+relative*std::max(initial.centering,scaled_mu);
  gate.relative_ok=value.dual<=gate.dual_target&&value.primal<=gate.primal_target&&
    value.centering<=gate.centering_target;
  // Only a rounding-scale allowance, never a proportional relaxation of the
  // model-decrease test. Infeasible iterates use the fixed penalty merit.
  const double roundoff=64*std::numeric_limits<double>::epsilon()*
    std::max({1.0,std::abs(model_merit),std::abs(anchor_merit)});
  gate.model_decrease=model_merit<=anchor_merit+roundoff;
  gate.passed=gate.relative_ok&&gate.model_decrease;
  return gate;
}
} // namespace lazyipdca::qcqp::hybrid_inner
