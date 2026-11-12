#include "lazyipdca_qcqp/linear_solver.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
using namespace lazyipdca::qcqp;
void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
int main(){try {
  require(linear_backend_compiled("cholesk"),"native Cholesk missing");
  // Independent dense hub, but two equality rows become dependent while
  // the hub is deferred. The unshifted full matrix remains nonsingular.
  const int np=519,n=np+2;
  std::vector<SymmetricTriplet> entries;
  for(int i=0;i<np;++i)entries.push_back({i,i,4});
  for(int j=1;j<np;++j)entries.push_back({0,j,0.0625});
  entries.push_back({0,np,1});entries.push_back({1,np,1});
  entries.push_back({0,np+1,-1});entries.push_back({1,np+1,1});
  Vec expected(n),b(n,0);for(int i=0;i<n;++i)expected[i]=0.125*(i%11-5);
  for(auto e:entries){b[e.i]+=e.value*expected[e.j];if(e.i!=e.j)b[e.j]+=e.value*expected[e.i];}
  for(bool metis:{false,true})for(int mode=0;mode<3;++mode) {
    Settings s;s.linear_backend="cholesk";s.linear_metis=metis;s.dca_engine="majorants";
    s.kkt_primal_first_ordering=false;s.kkt_interleaved_equality_ordering=true;
    s.kkt_primal_regularization=s.kkt_dual_regularization=0;
    s.cholesk_relative_pivot_tolerance=1e-14;
    s.kkt_adaptive_regularization=mode==0;
    s.cholesk_adaptive_regularization=true;
    s.kkt_delta_dual_floor=1e-12;s.kkt_delta_dual_max=mode==2?0:1e-8;
    s.kkt_rho_x_floor=0;s.kkt_rho_x_max=0;s.kkt_regularization_retry_max=5;
    s.kkt_backward_error_tolerance=1e-13;s.kkt_backward_error_reject=1e-9;
    s.kkt_iterative_refinement_steps=5;
    std::string msg;auto f=make_symmetric_factorization(s,msg);
    require(bool(f),msg);require(f->factor(n,np,entries,msg),msg);
    Vec x=b;require(f->solve(x,msg),msg);Vec residual=b;
    for(auto e:entries){residual[e.i]-=e.value*x[e.j];if(e.i!=e.j)residual[e.j]-=e.value*x[e.i];}
    require(inf_norm(residual)<1e-7,"original-system residual");
    require(f->last_positive_pivots()==np&&f->last_negative_pivots()==2,"inertia changed");
    if(mode==0||mode==2) {
      require(f->symbolic_analyses()>=2,"exact original-order rescue removed");
      require(f->last_primal_regularization()==0&&f->last_dual_regularization()==0,"hidden regularization");
    } else {
      require(f->symbolic_analyses()==1,"premature order discard before authorized local repair");
      require(f->last_dual_regularization()<=s.kkt_delta_dual_max,"dual cap exceeded");
    }
    require(f->last_regularization_attempts()<=2*(s.kkt_regularization_retry_max+1),"unbounded retries");
    require(f->factor(n,np,entries,msg),msg);x=b;require(f->solve(x,msg),msg);
    require(f->last_symbolic_reused(),"symbolic reuse lost");
  }
  std::cout<<"PASS deferred order rescue, exact/fixed/no-repair modes, METIS on/off, residual and inertia, bounded retries\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
