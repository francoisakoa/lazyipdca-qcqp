#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void close(long double a,long double b,const char* text){check(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<5e-12L*std::max({1.L,std::abs(a),std::abs(b)}),text);}
QuadraticFunction q(int n){QuadraticFunction a;a.linear.assign(n,0);a.hessian.n=n;return a;}
void test(double x) {
 QCQPProblem p;p.name="native_sign_crossing";p.n=3;p.x0={x,.6,-.3};p.lb.assign(3,-INFINITY);p.ub.assign(3,INFINITY);
 p.objective=q(3);p.objective.linear={.4,-1,.7};p.objective.hessian.entries={{0,1,.7}};p.objective.signed_square={{0,-.7}};
 auto h=q(3);h.linear={.2,-.8,.5};h.constant=.7;h.signed_square={{0,1.2},{1,-.4}};p.equalities={h};
 auto g=q(3);g.linear={-.7,.1,.8};g.constant=-.9;g.hessian.entries={{0,0,-.2},{0,1,.3}};g.signed_square={{0,-1.7},{2,.8}};p.inequalities={g};
 Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";s.linear_backend="native";s.dca_filter_curvature_backend="native";s.lazy_newton_atomic_regularization=false;s.lazy_newton_lazy_constraints=false;s.lazy_newton_reduction=false;s.verbose=false;s.live_display_level=0;s.objective_lazy_convexification=true;
 LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());Workspace w(p,s,stats,costs);DcaFilterStatistics ds;DcaInequalityModels models(w,ds);models.build();
 Direction d;d.x={x==0?-.8:-2*x,-1.3,.8};d.s.assign(w.mi,0);d.z.assign(w.mi,0);d.lambda.assign(w.me,0);
 DcaMajorantBarrierPath path(w,d);
 for(double a:{0.,.1,.5,.75,1.}) {
  Vec y=w.x;axpy(a,d.x,y);auto point=path.point(a,.2);
  close(point.objective,p.objective.value(y)*w.objective.scale,"native objective along crossing path");
  close(path.equalities[0].at(a),p.equalities[0].value(y)*w.equalities[0].scale,"native equality along crossing path");
  const long double equality_remainder=w.equalities[0].scale*(p.equalities[0].value(y)-p.equalities[0].value(w.x)-a*dot(p.equalities[0].gradient(w.x),d.x));
  close(path.equalities[0].nonlinear_change(a),equality_remainder,"native exact SOC remainder across zero");
  close(path.inequalities[0].at(a),p.inequalities[0].value(y)*w.inequalities[0].scale,"native inequality along crossing path");
  const auto& iq=w.inequalities[0];const long double actual_gap=iq.model_at(y)-iq.original_at(y);
  const long double gap=models.correction_gap(0,y);close(gap,actual_gap,"native global majorant gap");check(gap>=-1e-14,"majorant failed across zero");
  std::vector<long double> dg(w.n,0);models.correction_gap(0,y,&dg,1);Vec mg(w.n,0),og(w.n,0);iq.add_gradient_at(y,1,mg);iq.original_gradient_at(y,1,og);for(int j=0;j<w.n;++j)close(dg[j],mg[j]-og[j],"native majorant gradient gap");
 }
 models.discard();w.x[0]=-x;w.evaluate();Vec actual(w.n,0);w.equalities[0].original_gradient_at(w.x,1,actual);const Vec expected=p.equalities[0].gradient(w.x);for(int j=0;j<w.n;++j)close(actual[j],w.equalities[0].scale*expected[j],"native original gradient after sign change");
}
}
int main(){try{for(double x:{-2.,0.,3.,1e-12})test(x);std::cout<<"native signed-square path and majorant: PASS\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
