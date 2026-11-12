#include "../src/sparse_pair_elimination.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void check(detail::SparsePairElimination& e,int n,int primal,
           const std::vector<SymmetricTriplet>& a,int degree=24,int budget=64,bool regularized=false) {
  Settings s;s.linear_backend="native";s.kkt_adaptive_regularization=true;
  s.kkt_primal_regularization=s.kkt_dual_regularization=0;
  auto reduced=e.reduce(n,primal,a,degree,budget,regularized);
  std::string msg;auto f=make_symmetric_factorization(s,msg);
  if(!f->factor(e.dimension(),e.reduced_primal,reduced,msg))throw std::runtime_error(msg);
  // All original basis right-hand sides validate the complete inverse,
  // including coupled equalities, Schur updates and back substitution.
  for(int j=0;j<n;++j){Vec b(n,0);b[j]=1;Vec x=b;
    if(!e.solve(*f,x,msg))throw std::runtime_error(msg);
    for(auto t:a){b[t.i]-=t.value*x[t.j];if(t.i!=t.j)b[t.j]-=t.value*x[t.i];}
    for(double r:b)if(!std::isfinite(r)||std::abs(r)>1e-10)throw std::runtime_error("original residual");
  }
}
}
int main(){try{
  for(int budget:{0,64}){
    detail::SparsePairElimination e;
    for(double scale:{0.5,2.0}){
      std::vector<SymmetricTriplet> a={{0,0,-2},{1,1,4},{2,2,3},{3,3,2},{0,1,0.2},{1,2,0.3},
        {0,4,scale},{1,4,1},{1,5,2},{2,5,1},{4,4,0},{5,5,0}};
      check(e,6,4,a,24,budget);
      if(budget&&e.pairs!=2)throw std::runtime_error("missing stable equality pivots");
    }
    if(e.plan_builds!=1||e.plan_reuses!=1)throw std::runtime_error("numeric plan was not reused");
  }
  {
    // Structural zero becomes nonzero without changing the cached envelope.
    detail::SparsePairElimination e;
    std::vector<SymmetricTriplet> a={{0,0,4},{1,1,5},{2,2,6},{3,3,0},
      {0,3,1},{1,3,0},{0,1,0}};
    check(e,4,3,a);
    a[5].value=0.4;a[6].value=0.1;
    check(e,4,3,a);
    if(e.plan_reuses!=1)throw std::runtime_error("zero crossing invalidated symbolic layout");
    a.push_back({1,2,0.25});
    check(e,4,3,a);
    if(e.plan_invalidations!=1)throw std::runtime_error("changed pattern not invalidated");
    // Losing the chosen pivot returns the original matrix for this factor.
    // The other equality coefficient still gives a well-posed KKT system.
    a[4].value=0;
    check(e,4,3,a);
    if(e.numeric_fallbacks!=1||e.pairs!=0||e.dimension()!=4)
      throw std::runtime_error("weak cached pivot did not fall back");
    check(e,4,3,a);
    if(e.pairs!=1)throw std::runtime_error("new stable pivot was not rebuilt");
  }
  {
    // A previously zero equality diagonal also invalidates a pair pivot.
    detail::SparsePairElimination e;
    std::vector<SymmetricTriplet> a={{0,0,4},{1,1,5},{2,2,0},{0,2,1}};
    check(e,3,2,a);a[2].value=-0.1;check(e,3,2,a);
    if(e.numeric_fallbacks!=1||e.pairs!=0)throw std::runtime_error("nonzero equality diagonal ignored");
  }
  {
    // General indefinite pairs retain regularized equality equations exactly.
    detail::SparsePairElimination e;
    std::vector<SymmetricTriplet> a={{0,0,4},{1,1,5},{2,2,6},{3,3,-1e-6},
      {4,4,-0.02},{0,3,1},{1,3,0.5},{1,4,1},{2,4,0.5},{0,1,0.2}};
    check(e,5,3,a,24,64,true);
    if(e.pairs!=2)throw std::runtime_error("regularized equality pairs not eliminated");
    a[3].value=-0.04;a[4].value=-0.001;
    check(e,5,3,a,24,64,true);
    if(e.plan_reuses!=1)throw std::runtime_error("regularized pair plan not reused");
    Settings s;s.linear_backend="native";s.kkt_adaptive_regularization=true;
    s.kkt_primal_regularization=s.kkt_dual_regularization=0;
    std::string message;
    int full_factory_calls=0;
    auto f=detail::with_sparse_pair_elimination(make_symmetric_factorization(s,message),
      [&](std::string& m){++full_factory_calls;return make_symmetric_factorization(s,m);});
    for(double diagonal:{-1e-6,-0.003}){
      a[3].value=diagonal;
      if(!f->factor(5,3,a,message))throw std::runtime_error(message);
      for(int j=0;j<5;++j){Vec b(5,0);b[j]=1;Vec x=b;
        if(!f->solve(x,message))throw std::runtime_error(message);
        for(auto t:a){b[t.i]-=t.value*x[t.j];if(t.i!=t.j)b[t.j]-=t.value*x[t.i];}
        for(double r:b)if(!std::isfinite(r)||std::abs(r)>1e-10)
          throw std::runtime_error("regularized wrapper original residual");
      }
    }
    a[2].value=-6;
    if(f->factor(5,3,a,message)||f->curvature_defect()==0)
      throw std::runtime_error("wrapper accepted deficient original inertia");
    // Rejected-inertia probes must use the full fallback's coordinates.
    Vec b(5,1),x=b;
    if(!f->solve_curvature_probe(x,message))throw std::runtime_error("curvature probe fallback unavailable");
    for(auto t:a){b[t.i]-=t.value*x[t.j];if(t.i!=t.j)b[t.j]-=t.value*x[t.i];}
    for(double r:b)if(!std::isfinite(r)||std::abs(r)>1e-10)
      throw std::runtime_error("curvature probe fallback original residual");
    if(full_factory_calls!=1)throw std::runtime_error("full backend was not created lazily");
    if(f->factor(5,3,a,message))throw std::runtime_error("repeated deficient inertia accepted");
    if(full_factory_calls!=1)throw std::runtime_error("full backend symbolic owner recreated");
    a[2].value=6;
    if(!f->factor(5,3,a,message))throw std::runtime_error("reduced factor did not recover after full fallback");
    b.assign(5,1);x=b;
    if(!f->solve(x,message))throw std::runtime_error(message);
    for(auto t:a){b[t.i]-=t.value*x[t.j];if(t.i!=t.j)b[t.j]-=t.value*x[t.i];}
    for(double r:b)if(!std::isfinite(r)||std::abs(r)>1e-10)
      throw std::runtime_error("switched persistent backend original residual");
    const auto* wrapped=dynamic_cast<detail::PairReducedFactorization*>(f.get());
    const int reduced_dimension=wrapped->elimination().dimension();
    a.push_back({0,2,0.01});a.push_back({0,4,0.25});
    if(!f->factor(5,3,a,message))throw std::runtime_error(message);
    if(wrapped->elimination().dimension()!=reduced_dimension||wrapped->reduced_backend_resets!=1||full_factory_calls!=2)
      throw std::runtime_error("changed elimination plan reused unrelated reduced backend coordinates");
    b.assign(5,1);x=b;
    if(!f->solve(x,message))throw std::runtime_error(message);
    for(auto t:a){b[t.i]-=t.value*x[t.j];if(t.i!=t.j)b[t.j]-=t.value*x[t.i];}
    for(double r:b)if(!std::isfinite(r)||std::abs(r)>1e-10)
      throw std::runtime_error("new reduced coordinate layout original residual");
    a.back().value=0.2;
    if(!f->factor(5,3,a,message)||wrapped->reduced_backend_resets!=1||full_factory_calls!=2)
      throw std::runtime_error("same-plan numeric update recreated reduced backend");
  }
  {
    // Many sparse regularized normal columns are NOT dense auxiliaries.
    // Their common primal variables have high degree and exceed this cap.
    detail::SparsePairElimination e;
    std::vector<SymmetricTriplet> a;
    for(int j=0;j<20;++j)a.push_back({j,j,3});
    for(int j=20;j<40;++j){a.push_back({j,j,-0.01});a.push_back({0,j,1});a.push_back({1,j,0.5});}
    check(e,40,20,a,8,256,true);
    if(e.pairs)throw std::runtime_error("sparse regularized normals escaped degree budget");
  }
  {
    // Many dense atom borders must not consume the sparse degree budget.
    detail::SparsePairElimination e;
    std::vector<SymmetricTriplet> a={{0,0,4},{1,1,5},{2,2,6},{3,3,0},{0,3,1},{1,3,0.5}};
    for(int k=4;k<24;++k){a.push_back({k,k,-1});for(int j=0;j<3;++j)a.push_back({j,k,0.02*(j+1)});}
    check(e,24,3,a,8,64);
    if(e.pairs!=1)throw std::runtime_error("dense auxiliary borders vetoed sparse pivot");
    for(auto& t:a)if(t.i<3&&t.j>=4)t.value*=2;
    check(e,24,3,a,8,64);
    if(e.plan_reuses!=1)throw std::runtime_error("dense border numeric plan not reused");
  }
  std::cout<<"PASS sparse pairs: original inverse, fill budgets, cache reuse, zero crossings, pattern and pivot fallback, regularized normals, dense borders\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
