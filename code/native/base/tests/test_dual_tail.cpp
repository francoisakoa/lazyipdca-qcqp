#include "../src/atomic_newton.cpp"
#include <iostream>
using namespace lazyipdca::qcqp;
namespace {
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
long double norm(const std::vector<long double>& v){long double r=0;for(auto x:v)r=std::max(r,std::abs(x));return r;}
void test(){
 Settings st;st.dca_engine="majorants";st.linear_backend="mumps";
 st.lazy_newton_reduction=false;st.lazy_newton_lazy_constraints=false;
 st.lazy_newton_atomic_regularization=false;st.verbose=false;st.live_display_level=0;
 QCQPProblem p;p.name="tail_identity";p.n=2;p.x0={.2,.3};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
 p.objective.linear={-.5,.2};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,2},{1,1,3}};
 QuadraticFunction q;q.constant=-1;q.linear={.1,.2};q.hessian.n=2;q.hessian.entries={{0,0,-2},{1,1,2}};p.inequalities.push_back(q);p.validate();
 LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(p,st,ns,costs);
 w.s={1};w.z={.4};w.evaluate();DcaFilterStatistics fs;DcaInequalityModels model(w,fs);
 model.prepare_curvature_once();model.reanchor_from_cache();w.update_matrix();require(w.factorize(0),"factor");
 const Vec rc={.2};Direction d=w.direction(rc,w.rd,w.h,w.rp);DcaTailTerms terms(w,d);
 require(norm(terms.added)>1e-4,"fixture missing curvature");
 const auto x=w.x,z=w.z;const auto factors=ns.factor_attempts;
 const auto before=norm(terms.linear);
 for(int k=0;k<4;++k){Vec rhs=w.rd;for(int j=0;j<w.n;++j)rhs[j]-=double(terms.added[j]);d=w.direction(rc,rhs,w.h,w.rp);terms=DcaTailTerms(w,d);require(d.residual<=st.kkt_backward_error_reject,"modified KKT audit");}
 require(ns.factor_attempts==factors,"unexpected refactorization");require(w.x==x&&w.z==z,"iterate changed");
 require(norm(terms.linear)<.1*before,"no original linear contraction");
 // Compare the full quadratic identity to an independent evaluation at a fractional step.
 const double a=.37;axpy(a,d.x,w.x);axpy(a,d.z,w.z);axpy(a,d.s,w.s);model.discard();w.evaluate();
 long double err=0;for(int j=0;j<w.n;++j){auto predicted=(1-a)*terms.initial[j]+a*terms.linear[j]+a*a*terms.cross[j];err=std::max(err,std::abs(predicted-w.rd[j]));}
 require(err<1e-12,"quadratic identity/sign/scaling");
 // Independent identity with a quadratic equality, multiplier change and
 // non-unit frozen objective/constraint scaling; no solve is used as oracle.
 QuadraticFunction eq;eq.constant=.1;eq.linear={.2,-.1};eq.hessian.n=2;eq.hessian.entries={{0,1,.3}};
 p.equalities.push_back(eq);p.validate();LazyNewtonStatistics ns2;detail::NewtonCostLedger costs2(ns2,Clock::now());
 Workspace v(p,st,ns2,costs2);v.objective.scale=.07;v.equalities[0].scale=.3;v.inequalities[0].scale=.2;
 v.lambda={.7};v.z={.4};v.evaluate();DcaFilterStatistics fs2;DcaInequalityModels m2(v,fs2);
 m2.prepare_curvature_once();m2.reanchor_from_cache();
 Direction arbitrary;arbitrary.x={.1,-.2};arbitrary.z={.03};arbitrary.s={.04};arbitrary.lambda={-.07};
 DcaTailTerms t2(v,arbitrary);axpy(a,arbitrary.x,v.x);axpy(a,arbitrary.z,v.z);axpy(a,arbitrary.lambda,v.lambda);
 m2.discard();v.evaluate();long double err2=0;
 for(int j=0;j<v.n;++j)err2=std::max(err2,std::abs((1-a)*t2.initial[j]+a*t2.linear[j]+a*a*t2.cross[j]-v.rd[j]));
 require(err2<1e-12,"quadratic equality/scaling identity");
 std::cout<<"identity_error="<<double(err)<<" linear_ratio="<<double(norm(terms.linear)/before)<<" reused_factor=1\n";
}
void native_crossing_identity() {
 Settings st;st.dca_engine="majorants";st.linear_backend="mumps";
 st.lazy_newton_reduction=false;st.lazy_newton_lazy_constraints=false;
 st.lazy_newton_atomic_regularization=false;st.verbose=false;st.live_display_level=0;
 QCQPProblem p;p.name="native_tail_crossings";p.n=3;p.x0={.2,-.3,0};
 p.lb=Vec(3,-INFINITY);p.ub=Vec(3,INFINITY);
 p.objective.linear={.1,-.2,.3};p.objective.hessian.n=3;
 p.objective.hessian.entries={{0,0,2},{1,1,3},{2,2,1}};p.objective.signed_square={{0,.8}};
 QuadraticFunction h;h.constant=.1;h.linear={.2,-.1,.3};h.hessian.n=3;
 h.hessian.entries={{0,1,.2}};h.signed_square={{1,-.7}};p.equalities.push_back(h);
 QuadraticFunction g;g.constant=-1;g.linear={.1,.2,-.2};g.hessian.n=3;
 g.signed_square={{2,.4}};p.inequalities.push_back(g);p.validate();
 LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(p,st,ns,costs);
 w.objective.scale=.07;w.equalities[0].scale=.3;w.inequalities[0].scale=.2;
 w.lambda={.7};w.z={.4};w.evaluate();DcaFilterStatistics stats;DcaInequalityModels models(w,stats);
 models.prepare_curvature_once();models.reanchor_from_cache();
 Direction d;d.x={-.7,.8,.6};d.lambda={-.07};d.z={.03};d.s={.04};
 DcaTailTerms terms(w,d);const auto x=w.x,z=w.z,lambda=w.lambda,s=w.s;
 models.discard();bool crossed=false;
 for(double a:{.2,.6,1.}) {
   w.x=x;w.z=z;w.lambda=lambda;w.s=s;
   axpy(a,d.x,w.x);axpy(a,d.z,w.z);axpy(a,d.lambda,w.lambda);axpy(a,d.s,w.s);w.evaluate();
   const auto defect=terms.native_defect(a);require(defect.size()==3,"native defect dimension");
   if(a>.5)require(std::abs(defect[0])>1e-4&&std::abs(defect[1])>1e-4,"fixture missed orthant crossings");
   require(std::abs(defect[2])>1e-4,"fixture missed departure from exact zero");
   for(int j=0;j<3;++j) {
     const long double prediction=(1-a)*terms.initial[j]+a*terms.linear[j]+a*a*terms.cross[j]+defect[j];
     require(std::abs(prediction-w.rd[j])<1e-12,"native stationarity identity across zero/scaling/multiplier update");
   }
   crossed=crossed||a>.5;
 }
 require(crossed,"native crossing coverage");
}
}
int main(){try{require(linear_backend_compiled("mumps"),"MUMPS required");test();native_crossing_identity();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
