#include <cmath>
#include <stdexcept>
#include <algorithm>
static int observed=0;
template<class W,class D,class V>
void observe_dual(W& w,const D& d,double delta,double rho,const V& dual,const V& eq,const V& primal,const V& comp){
  if(!(delta>0))throw std::runtime_error("dual shift absent");
  // Independent original-coordinate equations for the analytical QCQP below.
  double station[3]={2*d.x[0]+dual[0],3*d.x[1]+dual[1],4*d.x[2]+dual[2]};
  for(int j:w.retained)station[j]+=rho*d.x[j];
  station[0]+=(1+w.x[0])*d.lambda[0];station[1]+=(1+w.x[1])*d.lambda[0];station[2]+=d.lambda[1];
  // The first equality has curvature diag(1,0,0).
  station[0]+=w.lambda[0]*d.x[0];station[1]+=w.lambda[0]*d.x[1];
  for(int i=0;i<w.mi;++i){
    double jdx;
    if(w.inequality_kind[i]==0){station[0]+=d.z[i];station[1]+=0.5*d.z[i];jdx=d.x[0]+0.5*d.x[1];}
    else {int j=w.inequality_origin[i];double a=w.inequality_kind[i];station[j]+=a*d.z[i];jdx=a*d.x[j];}
    const double shift=w.inequality_kind[i]==0?delta:0;
    if(std::abs(primal[i]+jdx+d.s[i]-shift*d.z[i])>1e-11)throw std::runtime_error("inequality or bound equation wrong");
    if(std::abs(comp[i]+w.z[i]*d.s[i]+w.s[i]*d.z[i])>1e-11)throw std::runtime_error("complementarity equation wrong");
  }
  for(double x:station)if(std::abs(x)>1e-11)throw std::runtime_error("stationarity equation wrong");
  const double jd[2]={(1+w.x[0])*d.x[0]+(1+w.x[1])*d.x[1],d.x[2]};
  for(int e=0;e<2;++e)if(std::abs(eq[e]+jd[e]-(w.pivot_variable[e]<0?delta*d.lambda[e]:0))>1e-11)
    throw std::runtime_error("equality perturbation or condensation wrong");
  ++observed;
}
#define G10_DUAL_TEST_OBSERVER observe_dual
#include "../src/atomic_newton.cpp"
using namespace lazyipdca::qcqp;
// Inject a precision failure until the dual diagonal is changed; the actual
// perturbed solve uses the real native factorization, not a mock solution.
class PrecisionFailure final:public SymmetricFactorization {
 public:
  Settings settings;std::unique_ptr<SymmetricFactorization> delegate;
  bool repaired=false;int calls=0;int primal=0;int dual_calls=0;bool inject_inertia=false;
  PrecisionFailure(){settings.linear_backend="native";settings.kkt_adaptive_regularization=true;
    std::string m;delegate=make_symmetric_factorization(settings,m);}
  bool factor(int n,int np,const std::vector<SymmetricTriplet>& a,std::string& m) override {
    ++calls;primal=np;repaired=false;
    for(auto t:a)if(t.i==np && t.j==np && t.value<0)repaired=true;
    if(repaired && ++dual_calls==1 && inject_inertia){m="native inertia mismatch: injected";return false;}
    return delegate->factor(n,np,a,m);
  }
  bool solve(Vec& rhs,std::string& m)const override {
    if(!repaired){m="backward error injected by regression test";return false;}
    return delegate->solve(rhs,m);
  }
  const char* name()const override{return "precision-injection-native";}
};
QuadraticFunction q(Vec a,double c){QuadraticFunction f;f.linear=a;f.constant=c;f.hessian.n=a.size();return f;}
int main(){try{
 for(bool reduce:{false,true})for(int dense:{1,100})for(bool enabled:{false,true})for(bool inertia:{false,true}){
  QCQPProblem p;p.n=3;p.lb={-2,-2,-2};p.ub={2,2,2};p.x0={0.2,0.3,0.2};
  p.objective=q({0.4,-0.2,0.1},0);p.objective.hessian.entries={{0,0,2},{1,1,3},{2,2,4}};
  p.equalities={q({1,1,0},-1),q({0,0,1},-0.2)};p.equalities[0].hessian.entries={{0,0,1},{1,1,1}};
  p.inequalities={q({1,0.5,0},-2)};
  Settings st;st.linear_backend="native";st.dca_aggregate=true;st.dca_aggregate_backend="native";
  st.lazy_newton_lazy_constraints=false;st.lazy_newton_reduction=reduce;st.lazy_newton_dense_threshold=dense;
  st.dca_aggregate_adaptive_dual=enabled;st.live_display_level=0;
  LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());Workspace w(p,st,stats,costs);
  DcaFilterStatistics ds;DcaModels models(w,ds);auto outer=dca_original(w,w.x,w.lambda,w.z);
  detail::OriginalDcaFilter filter(std::max(1.0,10*outer.theta));DcaAggregate aggregate(st);
  auto injected=std::make_unique<PrecisionFailure>();auto* probe=injected.get();probe->inject_inertia=inertia;aggregate.factor=std::move(injected);
  const int before=observed;aggregate.attempt(w,models,outer,filter,1);
  if(enabled && (observed==before||!probe->repaired))throw std::runtime_error("recovery was not exercised");
  if(enabled && inertia && probe->dual_calls<2)throw std::runtime_error("inertia recovery not exercised");
  if(!enabled && (observed!=before||probe->repaired))throw std::runtime_error("disabled recovery changed system");
 }
 std::cout<<"PASS adaptive dual: injected precision failure, actual solves, independent full equations, condensed/bordered inequalities, equality elimination, bounds unperturbed, inertia recovery, disabled path\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
