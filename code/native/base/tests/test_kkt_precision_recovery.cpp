#include <cstddef>
#include "../src/atomic_newton.cpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sstream>
using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
// A large unrelated third diagonal masks the ill-conditioned two-coordinate
// Gram block from the historical GLOBAL Hessian-normalized guard. The full
// five-equation primal/slack/dual system has the analytic solution below.
void fixture(double curvature,double slack,bool allow_recovery,bool healthy,bool force_precision=false) {
  Settings st;st.dca_engine="majorants";st.linear_backend="mumps";
  st.lazy_newton_reduction=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;st.verbose=false;st.live_display_level=0;
  st.kkt_backward_error_reject=1e-7;
  QCQPProblem p;p.name="heterogeneous_gram_precision";p.n=3;
  p.x0={0,0,0};p.lb=Vec(3,-INFINITY);p.ub=Vec(3,INFINITY);
  p.objective.linear={0,0,0};p.objective.hessian.n=3;
  p.objective.hessian.entries={{0,0,curvature},{1,1,curvature},{2,2,1e12}};
  constexpr double a=1.23456789,primal=0.01,rc=0.003;
  QuadraticFunction q;q.name="linear_inequality";q.constant=-1;
  q.linear={1,a,0};q.hessian.n=3;p.inequalities.push_back(q);p.validate();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.s={slack};w.z={1};w.update_matrix();
  const auto x=w.x,s=w.s,z=w.z,h=w.hvalue;
  // Check the ACTUAL assembled binary64 Gram block. Its positive determinant
  // must exceed a conservative rounding envelope before any backend call.
  // Thus this precision test does not depend on factoring a rounded singular
  // (or indefinite) approximation of a mathematically positive matrix.
  long double g00=0,g11=0,g01=0;
  for(std::size_t k=0;k<w.kkt.size();++k) {
    const auto& e=w.kkt[k];
    if(e.i==0&&e.j==0)g00+=w.base_values[k];
    if(e.i==1&&e.j==1)g11+=w.base_values[k];
    if((e.i==0&&e.j==1)||(e.i==1&&e.j==0))g01+=w.base_values[k];
  }
  const long double determinant=g00*g11-g01*g01;
  const long double roundoff=128*std::numeric_limits<double>::epsilon()*
      (std::abs(g00*g11)+std::abs(g01*g01));
  check(g00>0&&g11>0&&determinant>roundoff,
        "analytic Gram fixture lacks a representable positive-curvature margin");
  if(!w.factorize(0)||w.regularization!=0) {
    std::ostringstream why;why<<"analytic matrix failed to factor without rho"
        <<" h="<<curvature<<" s="<<slack<<" rho="<<w.regularization
        <<" dense="<<w.ndense<<" reason="<<w.message;
    throw std::runtime_error(why.str());
  }
  // Independently solve H dx + J' dz = (a,-1,0), J dx + ds = -p,
  // ds + s dz = -rc. This oracle uses neither condensed data nor its audit.
  const long double aa=a,hh=curvature,ss=slack;
  const long double dz=(static_cast<long double>(primal)-rc)/((1+aa*aa)/hh+ss);
  const long double oracle[3]={(aa-dz)/hh,(-1-aa*dz)/hh,0};
  const auto full_error=[&](const Direction& v) {
    return std::max({
      std::abs(hh*v.x[0]+v.z[0]-aa),std::abs(hh*v.x[1]+aa*v.z[0]+1),
      std::abs(1e12L*v.x[2]),
      std::abs(static_cast<long double>(primal)+v.x[0]+aa*v.x[1]+v.s[0]),
      std::abs(static_cast<long double>(rc)+v.s[0]+ss*v.z[0])});
  };
  if(force_precision) {
    // Deterministically exercise the recovery contract even when this MUMPS
    // version solves the condensed fixture accurately without any recovery.
    Direction rejected;rejected.x={double(oracle[0]),double(oracle[1]),0};
    rejected.s={double(-static_cast<long double>(rc)-ss*dz)};
    rejected.z={double(dz)};
    check(full_error(rejected)/a<=st.kkt_backward_error_reject,
          "independent analytic direction is not initially valid");
    rejected.z[0]+=1e-3;
    Vec ed(3),eh,ep(1),ec(1);double de,ee,pe,ce;
    check(full_error(rejected)/a>st.kkt_backward_error_reject&&
          w.majorant_direction_residual_extended(rejected,{rc},{-a,1,0},{},{primal},
              ed,eh,ep,ec,de,ee,pe,ce)/a>st.kkt_backward_error_reject,
          "injected direction was not genuinely rejected by both original audits");
    check(w.recover_condensation_precision(rejected,a),
          "precision recovery did not promote the measured unsafe Gram row");
    check(ns.precision_condensation_recoveries==1&&ns.precision_condensation_rows==1&&w.ndense==1,
          "precision recovery did not retain exactly one equivalent border");
    check(w.factorize(0)&&w.regularization==0,
          "equivalent bordered fixture failed to factor without rho");
    check(!w.recover_condensation_precision(rejected,a)&&
          ns.precision_condensation_recoveries==1,
          "precision recovery repeated in the same assembly");
    check(full_error(rejected)/a>st.kkt_backward_error_reject,
          "promotion concealed the invalid original direction");
  }
  // Suppress only the new recovery to reproduce the historical failure path.
  if(!allow_recovery){w.precision_condensation_scanned=true;w.precision_quality_used=true;}
  Direction d;
  try { d=w.direction({rc},{-a,1,0},{},{primal}); }
  catch(const std::runtime_error& e) {
    check(!allow_recovery&&!healthy,"expanded recovery failed on the analytic KKT");
    check(std::string(e.what()).find("expanded Newton residual rejected")!=std::string::npos,
          "baseline failed for another reason");
    check(w.current_factor->last_backward_error()<1e-12,
          "fixture did not exhibit the normwise/expanded discrepancy");
    std::cout<<"historical_rejection h="<<curvature<<" s="<<slack<<std::endl;
    return;
  }
  for(int j=0;j<3;++j)
    check(std::abs(d.x[j]-oracle[j])<1e-7L*std::max(1.0L,std::abs(oracle[j])),
          "direction disagrees with the independent full KKT solution");
  const long double error=full_error(d);
  check(error/a<=st.kkt_backward_error_reject&&d.residual<=st.kkt_backward_error_reject,
        "recovery relaxed the strict expanded tolerance");
  check(w.x==x&&w.s==s&&w.z==z&&w.hvalue==h&&w.regularization==0,
        "equivalent recovery changed the problem, iterate or rho");
  check(ns.precision_condensation_recoveries<=1&&ns.precision_condensation_rows<=1,
        "precision recovery was not bounded");
  if(healthy)check(ns.factor_attempts==1&&ns.precision_condensation_recoveries==0&&w.ndense==0,
                   "healthy direction paid for a recovery or extra factorization");
  // A real error must remain visible, even after a successful recovery.
  auto wrong=d;wrong.z[0]+=1e-3;
  Vec ed(3),eh,ep(1),ec(1);double de,ee,pe,ce;
  check(w.majorant_direction_residual_extended(wrong,{rc},{-a,1,0},{},{primal},
        ed,eh,ep,ec,de,ee,pe,ce)/a>st.kkt_backward_error_reject,
        "recovery concealed a genuinely invalid direction");
  if(healthy) {
    // Exercise MUMPS's real quality callback on a known equivalent system.
    // Ordinary healthy direction calls above must not invoke it themselves.
    const auto before=w.current_factor->pivot_tolerance();
    const auto factors=ns.factor_attempts;
    check(w.recover_precision_pivoting(),"MUMPS quality recovery unavailable");
    check(w.current_factor->pivot_tolerance()>before&&ns.factor_attempts==factors+1,
          "quality recovery did not count its numeric refactorization");
    const auto improved=w.direction({rc},{-a,1,0},{},{primal});
    check(improved.residual<=st.kkt_backward_error_reject&&w.regularization==0,
          "quality recovery changed the original target");
    for(int j=0;j<3;++j)check(std::abs(improved.x[j]-oracle[j])<1e-10L,
                              "pivoting recovery changed the analytic solution");
    check(!w.recover_precision_pivoting()&&ns.factor_attempts==factors+1,
          "quality recovery repeated in the same outer assembly");
  }
  std::cout<<"full_kkt_pass h="<<curvature<<" s="<<slack<<" residual="<<double(error/a)
           <<" recovered="<<ns.precision_condensation_recoveries
           <<" forced_precision="<<force_precision<<std::endl;
}
}
int main() {
  try {
    if(!linear_backend_compiled("mumps")) { std::cout<<"SKIP: MUMPS unavailable\n";return 0; }
    fixture(1,1,true,true);
    for(auto values:{std::pair<double,double>{0.01,1e-10},{0.0001,1e-7}}) {
      fixture(values.first,values.second,false,false);
      fixture(values.first,values.second,true,false);
    }
    fixture(0.01,1e-10,true,false,true);
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
