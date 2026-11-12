#include "../src/atomic_newton.cpp"
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
QCQPProblem fixture(bool inconsistent,bool healthy){
  QCQPProblem p;p.name="anonymous_linear_algebra_fixture";p.n=2;p.x0={0,0};
  p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.linear={0,0};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,1}};
  if(healthy || inconsistent)p.objective.hessian.entries.push_back({1,1,1});
  QuadraticFunction q;q.linear={1,0};q.constant=-1;q.hessian.n=2;p.equalities={q};
  if(inconsistent){q.constant=-2;p.equalities.push_back(q);}return p;
}
int main(){try{
  for(const std::string backend:{"mumps","ldl"}){
    if(!linear_backend_compiled(backend))continue;
    for(int mode=0;mode<3;++mode){
      Settings s;s.dca_engine="majorants";s.linear_backend=backend;
      s.lazy_newton_reduction=false;s.lazy_newton_atomic_regularization=false;
      s.lazy_newton_lazy_constraints=false;s.verbose=false;s.live_display_level=0;
      s.dca_filter_sparse_fallback=false;
      auto p=fixture(mode==1,mode==2);
      LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());Workspace w(p,s,ns,costs);
      w.update_matrix();const auto initial=w.x;
      require(w.factorize(0),"compatible numerical rescue failed to factor");
      bool rejected=false;double dx=0,lambda=0,residual=INFINITY;
      try{
        auto d=w.direction({},w.rd,w.h,{});
        dx=d.x[0];lambda=d.lambda[0];residual=d.residual;
      }catch(const std::runtime_error&){rejected=true;}
      require(rejected==(mode==1),"wrong acceptance of compatible/incompatible system");
      if(!rejected){
        require(std::abs(dx-1)<1e-7 && std::abs(lambda+1)<1e-7,"analytic direction wrong");
        require(residual<=1e-7,"expanded audit was relaxed");
      }
      require(w.x==initial,"factor recovery modified accepted primal state");
      if(mode==2)require(ns.backend_regularization_attempts==0,"healthy path invoked rescue");
    }
  }
  std::cout<<"bounded pivot: analytic singular system, inconsistent rejection and healthy path PASS\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
