#include "lazyipdca_qcqp/linear_solver.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool b,const char* message){if(!b)throw std::runtime_error(message);}
int main(){try{
 for(const char* backend:{"native","ldl"})if(linear_backend_compiled(backend)) {
  Settings s;s.linear_backend=backend;s.kkt_adaptive_regularization=true;
  s.kkt_primal_regularization=0;s.kkt_dual_regularization=0;
  std::string message;auto f=make_symmetric_factorization(s,message);check(bool(f),"backend construction");
  Vec rhs={0,1,0};check(!f->solve_curvature_probe(rhs,message),"uninitialized factor was exposed");
  // x is fixed by the equality, y has negative tangent curvature. Negative
  // curvature in x alone must not be confused with this tangent defect.
  std::vector<SymmetricTriplet> k={{0,0,-4},{1,1,-1},{0,2,1},{2,2,0}};
  check(!f->factor(3,2,k,message),"indefinite tangent factor accepted as Newton");
  check(f->curvature_defect()==1,"wrong tangent inertia deficit");
  check(f->solve_curvature_probe(rhs,message),"complete rejected factor unavailable for checked probe");
  check(std::abs(rhs[0])<1e-12&&std::abs(rhs[1]+1)<1e-12&&std::abs(rhs[2])<1e-12,"probe lost tangent equations");
  check(f->last_backward_error()<=s.kkt_backward_error_reject,"probe bypassed residual rejection");
  k[1].value=2;check(f->factor(3,2,k,message),"normal-only negative curvature over-corrected");
  rhs={0,1,0};check(f->curvature_defect()==0&&!f->solve_curvature_probe(rhs,message),"accepted factor retained stale failure state");
  k[1].value=0;check(!f->factor(3,2,k,message),"singular factor accepted");
  check(f->curvature_defect()==0&&!f->solve_curvature_probe(rhs,message),"partial or singular factor reused as oracle");
 }
 std::cout<<"PASS checked tangent inverse probes, normal curvature and invalid factor rejection\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
