#include <cstddef>
#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
detail::gr::State state(const Workspace& w,const DcaGrKktSoc::Staged& v) {
  detail::gr::State s;s.objective=w.objective.original_at(v.x);
  for(int e=0;e<w.me;++e){s.c.push_back(w.equalities[e].original_at(v.x));s.y.push_back(v.lambda[e]);}
  for(int i=0;i<w.mi;++i)if(w.active[i]) {
    s.s.push_back(v.s[i]);const double c=w.inequalities[i].original_at(v.x)+v.s[i];
    if(w.inequality_kind[i]==0){s.c.push_back(c);s.y.push_back(v.z[i]);}else s.b.push_back(c);
  }
  return s;
}
void fixture() {
  constexpr double mu=.01,alpha=.5;
  QCQPProblem p;p.name="curved_equality_without_private_pivots";p.n=2;
  p.x0={.8,.6};p.lb={0,0};p.ub={INFINITY,INFINITY};
  p.objective.hessian.n=2;p.objective.linear.resize(2);
  const Vec tangent={.6,-.8};
  for(int j=0;j<2;++j)p.objective.linear[j]=
      -(2+mu/(p.x0[j]*p.x0[j]))*tangent[j]-2*p.x0[j]+mu/p.x0[j];
  QuadraticFunction h;h.name="unit_circle";h.constant=-1;h.linear={0,0};h.hessian.n=2;
  h.hessian.entries={{0,0,2},{1,1,2}};p.equalities.push_back(h);p.validate();
  Settings st;st.dca_engine="majorants";st.outer_algorithm="lazy-dca";
  st.linear_backend="mumps";st.linear_metis=true;st.majorant_equality_stabilization=true;
  st.original_stabilized_merit=true;st.original_source_primal_barrier=true;
  st.original_gr_kkt_soc=true;st.lazy_newton_reduction=false;
  st.lazy_newton_lazy_constraints=false;st.lazy_newton_atomic_regularization=false;
  st.verbose=false;st.live_display_level=0;st.kkt_backward_error_reject=1e-7;
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.lambda={1};
  for(int i=0;i<w.mi;++i){w.s[i]=w.x[w.index[w.inequality_origin[i]]];w.z[i]=mu/w.s[i];}
  w.evaluate();w.equality_stabilization.multiplier_center=w.lambda;
  w.equality_stabilization.source_inequality_center=w.z;w.equality_stabilization.rho=.01;
  w.set_barrier_mu(mu);w.update_matrix();
  check(w.factorize(0)&&w.regularization==0,"real original KKT did not factor without proximal shift");
  Vec rc(w.mi);for(int i=0;i<w.mi;++i)rc[i]=w.s[i]*w.z[i]-mu;
  const auto d=w.direction(rc,w.rd,w.equality_newton_residual(),w.primal_newton_residual());
  check(std::abs(d.x[0]-.6)<1e-10&&std::abs(d.x[1]+.8)<1e-10,
        "real Newton solve did not recover the independent analytic tangent");
  const auto geometry=DcaDefinedPivotSoc::build(w.n,w.equalities);
  check(!geometry.supported&&geometry.curved_equalities==1,
        "fixture accidentally admits the old private-pivot SOC");
  const auto x=w.x,s=w.s,z=w.z,lambda=w.lambda,hessian=w.hvalue,base_matrix=w.base_values;
  const auto center=w.equality_stabilization.multiplier_center;
  auto* factor=w.current_factor;const auto factors=ns.factor_attempts,solves=ns.linear_solves;
  DcaGrKktSoc soc;
  check(soc.prepare(w,d,Clock::now(),60),"frozen MUMPS SOC correction failed its original audit");
  check(w.current_factor==factor&&ns.factor_attempts==factors&&soc.backsolves>=1&&soc.backsolves<=8,
        "SOC refactored or exceeded the frozen solve budget");
  const auto after_solves=ns.linear_solves;
  check(soc.prepare(w,d,Clock::now(),60)&&ns.linear_solves==after_solves,
        "reusing SOC performed another correction solve");
  check(soc.correction.residual<=st.kkt_backward_error_reject&&ns.linear_solves>solves,
        "SOC linear work was not counted and strictly audited");
  DcaMajorantBarrierPath path(w,d,false,true);
  DcaGrKktSoc::Staged baseline,corrected;double ratio=0;
  check(soc.stage(path,alpha,mu,baseline,corrected,ratio)&&ratio<=.5,
        "general SOC could not stage its bounded four-vector trial");
  const auto before_eq=DcaGrKktSoc::original_equalities(w,baseline.x);
  const auto after_eq=DcaGrKktSoc::original_equalities(w,corrected.x);
  const long double exact_corrected=static_cast<long double>(corrected.x[0])*corrected.x[0]+
      static_cast<long double>(corrected.x[1])*corrected.x[1]-1;
  check(after_eq.finite&&after_eq.inf<=before_eq.inf&&after_eq.l1<before_eq.l1&&
        std::abs(after_eq.inf-std::abs(exact_corrected))<1e-14,
        "SOC equality oracle failed to evaluate the original curved row at actual coordinates");
  check(soc.correction.x[0]!=0&&soc.correction.x[1]!=0,
        "SOC did not exercise a correction in original Hessian columns");
  const auto point=path.staged_point(corrected.x,corrected.s,corrected.z,mu);
  check(point.finite&&point.positive,"corrected trial violates the original positivity domain");
  DcaGrKktSoc::Staged origin{w.x,w.s,w.z,w.lambda};
  const auto a=state(w,origin),b=state(w,baseline),c=state(w,corrected);
  detail::gr::Parameters par;par.mu=mu;par.rho=w.equality_stabilization.rho;
  par.kappa=1;par.center=center;
  detail::gr::Direction gd;gd.objective=static_cast<double>(path.objective.linear);
  gd.c={static_cast<double>(path.equalities[0].linear)};gd.y=d.lambda;gd.s=d.s;
  for(int i=0;i<w.mi;++i)gd.b.push_back(static_cast<double>(path.inequalities[i].linear)+d.s[i]);
  const auto slope=detail::gr::directional(a,gd,par);
  auto change=[&](const detail::gr::State& trial,const Vec& primal) {
    Vec dx(w.n);for(int j=0;j<w.n;++j)dx[j]=primal[j]-w.x[j];
    return detail::gr::change(a,trial,par,
        DcaMajorantBarrierPath::prepare_original(w.objective,dx,w.x).change(1));
  };
  const auto ordinary=change(b,baseline.x),repaired=change(c,corrected.x);
  check(slope.valid&&slope.value<0&&!detail::gr::armijo(ordinary,slope,alpha,st.line_search_armijo),
        "ordinary curved trial did not actually fail original GR Armijo");
  check(detail::gr::armijo(repaired,slope,alpha,st.line_search_armijo),
        "SOC did not pass the unchanged original GR Armijo criterion");
  auto invalid=corrected;invalid.s[0]=-1;
  check(!path.staged_point(invalid.x,invalid.s,invalid.z,mu).positive,
        "nonpositive SOC slack was accepted");
  invalid=corrected;invalid.lambda[0]+=1e6;
  check(!detail::gr::armijo(change(state(w,invalid),invalid.x),slope,alpha,st.line_search_armijo),
        "invalid SOC multipliers bypassed actual original GR merit");
  DcaGrKktSoc expired;const auto before_expired=ns.linear_solves;
  check(!expired.prepare(w,d,Clock::now(),0)&&ns.linear_solves==before_expired,
        "zero available time performed a correction solve");
  DcaGrKktSoc late;
  check(!late.prepare(w,d,Clock::now()-std::chrono::seconds(2),1)&&ns.linear_solves==before_expired,
        "expired absolute time performed a correction solve");
  auto huge=soc;for(double& v:huge.correction.x)v*=1e9;
  DcaGrKktSoc::Staged unused_base,unused_corrected;
  check(!huge.stage(path,alpha,mu,unused_base,unused_corrected,ratio),
        "oversized SOC correction was staged");
  check(w.x==x&&w.s==s&&w.z==z&&w.lambda==lambda&&w.hvalue==hessian&&w.base_values==base_matrix&&
        w.current_factor==factor&&ns.factor_attempts==factors&&w.regularization==0&&
        w.equality_stabilization.multiplier_center==center&&w.equality_stabilization.rho==par.rho,
        "SOC trial changed the original anchor, matrix, factor or algorithmic parameters");
  std::cout<<"general_SOC original_EQ="<<static_cast<double>(before_eq.inf)<<"->"<<static_cast<double>(after_eq.inf)
           <<" original_GR="<<static_cast<double>(ordinary.value)<<"->"<<static_cast<double>(repaired.value)
           <<" backsolves="<<soc.backsolves<<" factor_reused=1\n";
}
}
int main() {
  if(!linear_backend_compiled("mumps")){std::cout<<"SKIP: real MUMPS+METIS required\n";return 0;}
  try{fixture();std::cout<<"general frozen KKT GR SOC: PASS\n";}
  catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
