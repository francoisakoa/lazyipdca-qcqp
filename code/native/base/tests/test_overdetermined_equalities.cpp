#include "../src/atomic_newton.cpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
using Exact=boost::multiprecision::cpp_rational;
void check(bool value,const char* message) {
  if(!value)throw std::runtime_error(message);
}
Exact magnitude(const Exact& value) {return value<0?-value:value;}
void include_max(Exact& maximum,const Exact& value) {
  const Exact absolute=magnitude(value);if(absolute>maximum)maximum=absolute;
}
std::vector<Exact> exact_gradient(const QuadraticFunction& q,
                                const std::vector<Exact>& point) {
  std::vector<Exact> result(point.size(),0);
  q.for_each_linear_nonzero([&](int j,double value){result[j]+=Exact(value);});
  for(const auto& entry:q.hessian.entries) {
    result[entry.i]+=Exact(entry.v)*point[entry.j];
    if(entry.i!=entry.j)result[entry.j]+=Exact(entry.v)*point[entry.i];
  }
  return result;
}
Exact exact_value(const QuadraticFunction& q,const std::vector<Exact>& point) {
  Exact result(q.constant);
  q.for_each_linear_nonzero([&](int j,double value){result+=Exact(value)*point[j];});
  for(const auto& entry:q.hessian.entries) {
    Exact term=Exact(entry.v)*point[entry.i]*point[entry.j];
    result+=entry.i==entry.j?term/2:term;
  }
  return result;
}
struct OriginalAudit {
  Exact raw=0,scale=1;
  Exact normalized() const {return raw/scale;}
  bool accepted(double threshold) const {return normalized()<=Exact(threshold);}
};
// Independent rational arithmetic: no backend residual, production summation
// helper, or floating product rounding is used in this oracle. All fixture
// coefficients, starts and returned direction entries are exact binary rationals.
OriginalAudit original_audit(const QCQPProblem& p,const Vec& x,const Direction& d,
                             double auxiliary_tau=0,
                             const std::vector<int>& retained={}) {
  std::vector<Exact> point(x.begin(),x.end()),trial=point;
  for(int j=0;j<p.n;++j)trial[j]+=Exact(d.x[j]);
  auto dual=exact_gradient(p.objective,trial);
  const auto rhs_dual=exact_gradient(p.objective,point);
  OriginalAudit result;
  for(const auto& value:rhs_dual)include_max(result.scale,value);
  for(std::size_t e=0;e<p.equalities.size();++e) {
    const auto gradient=exact_gradient(p.equalities[e],point);
    const Exact rhs=exact_value(p.equalities[e],point);
    include_max(result.scale,rhs);
    Exact row=rhs;
    for(int j=0;j<p.n;++j) {
      row+=gradient[j]*Exact(d.x[j]);
      dual[j]+=gradient[j]*Exact(d.lambda[e]);
    }
    include_max(result.raw,row);
  }
  // The auxiliary Newton matrix contains its explicit proximal tau I on
  // retained primal coordinates. This is part of the declared auxiliary
  // equation, unlike the factorizer's private primal/dual perturbations.
  // Include only this explicit model term; never include backend shifts or
  // change the independent original-QCQP KKT certificate.
  for(int j:retained)dual[j]+=Exact(auxiliary_tau)*Exact(d.x[j]);
  for(const auto& value:dual)include_max(result.raw,value);
  return result;
}
void oracle_controls() {
  const double threshold=1e-7;
  OriginalAudit scaled;scaled.raw=Exact(1.5e-7);scaled.scale=2;
  check(scaled.raw>Exact(threshold)&&scaled.accepted(threshold),
        "independent oracle confused an absolute and normalized residual");
  scaled.raw=Exact(2.5e-7);scaled.scale=3;
  check(scaled.raw>Exact(threshold)&&scaled.accepted(threshold),
        "independent oracle lost the quadratic RHS scale");
  scaled.raw=Exact(std::nextafter(threshold,INFINITY));scaled.scale=1;
  check(!scaled.accepted(threshold),"independent oracle weakened the Newton gate");
  scaled.raw=Exact(3.3e-7);scaled.scale=3;
  check(!scaled.accepted(threshold),"independent oracle accepted a scaled violation");
  // Nonunique equality multipliers may contain huge opposite components.
  // An actual residual of one must not disappear in their cancellation.
  const double large=1e30;
  const Exact cancelled=Exact(1)+Exact(large)+2*Exact(-0.5*large);
  check(cancelled==1,"independent rational cancellation oracle is inexact");
  scaled.raw=cancelled;scaled.scale=1;
  check(!scaled.accepted(threshold),"multiplier cancellation hid a true violation");

  // A private backend shift must not be hidden by adding it to the oracle.
  QCQPProblem p;p.n=1;p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};
  p.objective.linear={-1};p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,1}};
  const double tau=1e-6,private_shift=1e-3;
  Direction d;d.x={1/(1+tau+private_shift)};
  check(!original_audit(p,p.x0,d,tau,{0}).accepted(threshold),
        "oracle silently accepted a backend-perturbed equation");
  d.x={1/(1+tau)};
  check(original_audit(p,p.x0,d,tau,{0}).accepted(threshold),
        "oracle omitted the explicit auxiliary proximal term");
  check(!original_audit(p,p.x0,d).accepted(threshold),
        "forced proximal fixture no longer detects the old oracle omission");
}

// The third original equation is dependent, not a third independent
// restriction.  No rank presolve or function-name exception is involved.
QCQPProblem fixture(bool quadratic,bool inconsistent) {
  QCQPProblem p;p.name="anonymous_overdetermined_equalities";p.n=2;
  p.x0=quadratic?Vec{0.5,1}:Vec{0,0};
  p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={0,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction first,second,third;
  for(auto* q:{&first,&second,&third}) {
    q->linear={0,0};q->hessian.n=2;
  }
  first.constant=-1;second.constant=inconsistent?-3:-2;third.constant=quadratic?-4:-2;
  if(quadratic) {
    first.hessian.entries={{0,0,2}};
    second.hessian.entries={{0,0,4}};
    third.hessian.entries={{1,1,2}};
  } else {
    first.linear={1,0};second.linear={2,0};third.linear={0,1};
  }
  p.equalities={first,second,third};p.validate();return p;
}

void exercise(const std::string& backend,const std::string& configuration,
              bool quadratic,bool inconsistent,double requested_tau) {
  auto p=fixture(quadratic,inconsistent);
  Settings st;st.dca_engine="majorants";st.linear_backend=backend;
  st.convexification_study=configuration;
  st.lazy_newton_reduction=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;st.dca_filter_sparse_fallback=false;
  st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  st.kkt_backward_error_reject=1e-7;
  st.verbose=false;st.live_display_level=0;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);
  check(w.nr==2&&w.mr==3&&w.me==3,"overdetermined rows were silently removed");
  check(w.eq_retained==std::vector<int>({0,1,2}),"original multiplier indexing changed");
  check(w.equalities.size()==p.equalities.size(),"original nonlinear equations disappeared");
  check(w.objective.scale==1.0,"fixture objective scaling changed");
  for(const auto& q:w.equalities)
    check(q.scale==1.0,"fixture equality scaling changed");
  const auto old_x=w.x,old_h=w.h,old_lambda=w.lambda;
  w.update_matrix();
  bool rejected=false;Direction d;
  if(!w.factorize(requested_tau))rejected=true;
  else try { d=w.direction({},w.rd,w.h,{}); }
       catch(const std::runtime_error&) { rejected=true; }
  check(rejected==inconsistent,"consistent/inconsistent linearization classified incorrectly");
  check(w.x==old_x&&w.h==old_h&&w.lambda==old_lambda,"rank recovery changed accepted state");
  if(inconsistent)return;
  check(d.lambda.size()==p.equalities.size(),"missing original equality multipliers");
  const long double expected[2]={quadratic?0.75L:1.0L,quadratic?1.5L:2.0L};
  for(int j=0;j<2;++j)
    check(std::abs(d.x[j]-expected[j])<1e-7L,"incorrect independent analytic direction");
  // Check ALL original equations with the same documented RHS normalization
  // as the production gate. Raw absolute error is not that gate: RHS scales
  // are two (affine) and three (quadratic) in these fixtures. The threshold
  // itself remains exactly the binary64 setting 1e-7 on every backend.
  const auto audit=original_audit(p,old_x,d,w.regularization,w.retained);
  const auto unregularized_audit=original_audit(p,old_x,d);
  std::cout<<"[OVERDETERMINED-AUDIT] configuration="<<configuration
           <<" quadratic="<<quadratic<<" raw="<<std::setprecision(25)
           <<audit.raw.convert_to<long double>()
           <<" scale="<<audit.scale.convert_to<long double>()
           <<" normalized="<<audit.normalized().convert_to<long double>()
           <<" reported="<<d.residual<<" threshold="<<st.kkt_backward_error_reject
           <<" auxiliary_tau="<<w.regularization
           <<" requested_tau="<<requested_tau
           <<" backend_primal="<<w.current_factor->last_primal_regularization()
           <<" backend_dual="<<w.current_factor->last_dual_regularization()
           <<" unregularized_normalized="
           <<unregularized_audit.normalized().convert_to<long double>()
           <<" multipliers=";
  for(double value:d.lambda)std::cout<<value<<',';
  std::cout<<'\n';
  check(audit.accepted(st.kkt_backward_error_reject)&&
        std::isfinite(d.residual)&&d.residual<=st.kkt_backward_error_reject,
        "full original-coordinate auxiliary Newton audit failed");
  if(requested_tau>0)
    check(!unregularized_audit.accepted(st.kkt_backward_error_reject),
          "forced stabilization no longer reproduces the old oracle omission");
  if(quadratic) {
    // A tangent step does not satisfy these nonlinear equalities exactly.
    // Their original values must remain available to the outer filter.
    const Vec trial={old_x[0]+d.x[0],old_x[1]+d.x[1]};
    check(std::abs(w.equalities[0].at(trial))>0.5&&
          std::abs(w.equalities[1].at(trial))>1.0&&
          std::abs(w.equalities[2].at(trial))>2.0,
          "nonlinear equalities were replaced by their tangent equations");
  }
}
}

int main() {
  try {
    oracle_controls();
    if(!linear_backend_compiled("mumps")) {
      std::cout<<"SKIP: MUMPS unavailable\n";return 0;
    }
    for(const std::string configuration:{"off","HH","HS","SH"})
      for(bool quadratic:{false,true})
        for(bool inconsistent:{false,true})
          for(double tau:{0.0,1e-6})
            exercise("mumps",configuration,quadratic,inconsistent,tau);
    std::cout<<"overdetermined equality systems: retained equations, original multipliers, "
                "consistent Newton steps and inconsistent rejection PASS "
                "exact_rational_oracle=1 normalized_gate_unchanged=1 "
                "boundary_and_cancellation_controls=1 explicit_auxiliary_tau=1 "
                "backend_shifts_excluded=1 forced_nonzero_tau=1\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<std::endl;return 1;
  }
}
