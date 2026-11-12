#include "../src/atomic_newton.cpp"
#include <iostream>
using namespace lazyipdca::qcqp;
namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
void near(long double a,long double b,long double tolerance,const char* message) {
  require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tolerance,message);
}
Settings settings() {
  Settings st;st.dca_engine="majorants";st.linear_backend="native";
  st.lazy_newton_reduction=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;st.verbose=false;st.live_display_level=0;
  return st;
}
const Vec negative={7.0/9,-4.0/9,-4.0/9};
const Vec nullspace={-4.0/9,1.0/9,-8.0/9};
const Vec positive={-4.0/9,-8.0/9,1.0/9};
QCQPProblem fixture(bool sparse=false,double upper_eigenvalue=3) {
  QCQPProblem p;p.name="spectral_majorant_contract";p.n=3;p.x0={.2,-.3,.4};
  p.lb=Vec(3,-INFINITY);p.ub=Vec(3,INFINITY);
  p.objective.linear={.1,.2,.3};p.objective.hessian.n=3;
  p.objective.hessian.entries={{0,0,1},{1,1,1},{2,2,1}};
  QuadraticFunction q;q.constant=-2;q.linear={.3,-.1,.2};q.hessian.n=3;
  if(sparse)q.hessian.entries={{0,0,-2},{0,1,.5},{1,1,.2},{1,2,.4},{2,2,3}};
  else for(int i=0;i<3;++i)for(int j=i;j<3;++j)
    q.hessian.entries.push_back({i,j,-2*negative[i]*negative[j]+upper_eigenvalue*positive[i]*positive[j]});
  p.inequalities.push_back(q);p.validate();return p;
}
Vec product(const std::vector<SymTriplet>& h,const Vec& d) {
  Vec result(d.size(),0);
  for(const auto& e:h){result[e.i]+=e.v*d[e.j];if(e.i!=e.j)result[e.j]+=e.v*d[e.i];}
  return result;
}
void positive_part_and_gap() {
  auto p=fixture();auto st=settings();LazyNewtonStatistics ns;
  detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(p,st,ns,costs);
  constexpr double scale=.37,multiplier=1.7;
  w.inequalities[0].scale=scale;w.evaluate();DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();auto& q=w.inequalities[0];
  require(models.positive_spectral_count==1&&models.inequality[0].positive_spectral,
      "nonpaired dense block did not use its positive spectral part");
  require(!models.inequality[0].half_spectral,"nonpaired spectrum classified as paired");
  const auto anchor=w.x;
  near(q.at(anchor),q.original_at(anchor),1e-14,"majorant lost value tangency");
  // Analytic Householder eigenbasis: H=−2 uu'+3 vv'. This oracle does not
  // use LAPACK or the implementation's reconstructed eigendecomposition.
  for(const auto& e:q.model_hessian)
    near(e.v,3*scale*positive[e.i]*positive[e.j],2e-12,"positive-part coefficient differs from analytic PSD oracle");
  std::vector<Vec> directions={negative,nullspace,positive,{.4,-.7,.2},{1,0,0},{0,1,0},{0,0,1}};
  for(const auto& d:directions) {
    const auto hm=product(q.model_hessian,d),ho=product(q.hessian,d);
    const double model_curvature=dot(d,hm);
    const double correction_curvature=model_curvature-scale*dot(d,ho);
    require(model_curvature>=-1e-12&&correction_curvature>=-1e-12,
        "majorant or dominance correction lost PSD");
    Vec x=anchor;axpy(1,d,x);std::vector<long double> dual(3,0);
    const auto gap=models.correction_gap(0,x,&dual,multiplier);
    near(gap,.5*correction_curvature,2e-12,"cached correction gap disagrees with explicit curvature");
    near(gap,q.at(x)-q.original_at(x),2e-12,"cached correction gap disagrees with independent function values");
    for(int j=0;j<3;++j)
      near(dual[j],multiplier*(hm[j]-scale*ho[j]),2e-12,"weighted correction gradient disagrees with original/model coefficients");
  }
  near(dot(nullspace,product(q.model_hessian,nullspace)),0,2e-12,
      "positive spectral part unnecessarily penalizes a common null direction");
}
void bounded_dense_rank_two() {
  constexpr int n=128;constexpr double scale=.37;
  QCQPProblem p;p.name="bounded_dense_contract";p.n=n;p.x0=Vec(n,.1);
  p.lb=Vec(n,-INFINITY);p.ub=Vec(n,INFINITY);
  p.objective.linear=Vec(n,0);p.objective.hessian.n=n;
  for(int i=0;i<n;++i)p.objective.hessian.entries.push_back({i,i,1});
  QuadraticFunction q;q.constant=1;q.linear=Vec(n,0);q.hessian.n=n;
  // Analytic rank-two NSD matrix -u*u'-v*v', u=ones,
  // v=alternating signs. Two eigenvalues are -128;126 are zero.
  // The opposite-parity coefficients vanish and are absent from the source.
  for(int i=0;i<n;++i)for(int j=i;j<n;++j)if((i+j)%2==0)
    q.hessian.entries.push_back({i,j,-2});
  p.inequalities.push_back(q);p.validate();auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.inequalities[0].scale=scale;w.evaluate();
  const auto envelope=w.hpattern;DcaFilterStatistics stats;
  DcaInequalityModels models(w,stats);models.build();
  const auto& model=w.inequalities[0];
  require(models.inequality[0].positive_spectral,"bounded rank-two block used isotropic fallback");
  require(w.hpattern==envelope,"spectral construction changed precompiled envelope");
  std::set<Pair> compiled(envelope.begin(),envelope.end());
  for(const auto& e:model.model_hessian) {
    require(compiled.count(upper(e.i,e.j)),"dense spectral coefficient absent from compiled envelope");
    near(e.v,0,5e-10,"rank-two NSD model has artificial curvature");
  }
  Vec null(n,0);null[0]=1;null[2]=-1;
  Vec u(n,1),v(n);for(int i=0;i<n;++i)v[i]=i%2?-1:1;
  for(const Vec& d:std::vector<Vec>{null,u,v}) {
    const auto hm=product(model.model_hessian,d),ho=product(model.hessian,d);
    const double mc=dot(d,hm),correction=mc-scale*dot(d,ho);
    require(mc>=-1e-10&&correction>=-1e-10,"bounded majorant lost PSD or dominance");
    long double su=0,sv=0;for(int i=0;i<n;++i){su+=d[i];sv+=d[i]*(i%2?-1:1);}
    near(scale*dot(d,ho),-scale*(su*su+sv*sv),1e-8,"rank-two analytic curvature mismatch");
    Vec x=w.x;axpy(.1,d,x);
    near(models.correction_gap(0,x),model.at(x)-model.original_at(x),1e-9,
        "bounded majorant original/model gap mismatch");
  }
  near(dot(null,product(model.model_hessian,null)),0,1e-9,"bounded nullspace was damped");
}
void sparse_envelope_and_paired_preservation() {
  auto st=settings();
  for(bool sparse:{true,false}) {
    auto p=fixture(sparse,2);LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
    Workspace w(p,st,ns,costs);const auto envelope=w.hpattern;DcaFilterStatistics stats;
    DcaInequalityModels models(w,stats);models.build();
    require(w.hpattern==envelope,"curvature changed the precompiled symbolic envelope");
    require(!models.inequality[0].positive_spectral&&models.positive_spectral_count==0,
        "sparse/paired fixture did not preserve established majorant path");
    if(sparse) {
      require(!models.inequality[0].half_spectral,"sparse fixture accidentally paired");
      require(models.inequality[0].rho>0,"sparse fallback omitted convexification");
      Vec x=w.x;axpy(1,Vec{.2,-.4,.3},x);
      near(models.correction_gap(0,x),w.inequalities[0].at(x)-w.inequalities[0].original_at(x),2e-12,
          "sparse fallback gap mismatch");
    } else require(models.inequality[0].half_spectral,"paired spectrum no longer uses existing closed-form majorant");
  }
}
}
int main() {
  try {positive_part_and_gap();bounded_dense_rank_two();sparse_envelope_and_paired_preservation();
    std::cout<<"PASS spectral majorant: analytic PSD/dominance/nullspace, nonunit-scale gaps, sparse fallback, paired preservation\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
