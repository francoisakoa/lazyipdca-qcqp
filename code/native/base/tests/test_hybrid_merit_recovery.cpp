#include "lazyipdca_qcqp/hybrid_merit_recovery.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
int main(){try {
 auto p=hybrid_inner::check_progress(true,10,11,20,18,-2,1,1e-4,1e-12);
 require(!p.residual_contraction&&p.merit_decrease,"Armijo-only counterexample absent");
 require(hybrid_merit_recovery_allowed(p,20,18,1e-12,1,.5,1e-10,10,11,1),"valid guarded private merit step rejected");
 require(!hybrid_inner::check_model(11,.1,1,.25,1e-10,.5,1e-6,1,1e-6,18,20).passed,"private globalization must not imply a full model certificate");
 require(!hybrid_merit_recovery_allowed(p,20,18,1e-12,1,1.1,1e-10,10,11,1),"primal growth accepted");
 require(!hybrid_merit_recovery_allowed(p,20,18,1e-12,1,.5,1e-10,10,21,1),"unbounded KKT growth accepted");
 require(!hybrid_merit_recovery_allowed(p,20,20-1e-12,1e-10,1,.5,1e-10,10,11,1),"rounding-sized decrease accepted");
 require(!hybrid_merit_recovery_allowed(p,20,18,1e-12,NAN,.5,1e-10,10,11,1),"nonfinite primal accepted");
 auto none=hybrid_inner::check_progress(false,10,11,20,18,-2,1,1e-4,1e-12);
 require(!hybrid_merit_recovery_allowed(none,20,18,1e-12,1,.5,1e-10,10,11,1),"unchanged state accepted");
 require(hybrid_merit_recovery_allowed(p,20,18,1e-12,0,0,1e-10,10,11,1),"exact affine feasibility cannot persist");
 std::cout<<"PASS guarded merit recovery: Armijo-only progress, physical primal contraction/floor, bounded KKT growth, rounding/no-op rejection, independent final certificate\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
