#include "lazyipdca_qcqp/majorant_equality_stabilization.hpp"
#include <iostream>
using namespace lazyipdca::qcqp;
using namespace lazyipdca::qcqp::detail;
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){
 MajorantEqualityStabilization a;a.enabled=true;a.reset({0},1);
 require(!a.update_epoch({1},.5,1e-7,0,1),"implicit zero-barrier gate changed");
 require(!a.update_epoch({1},.5,2e-6,0,1,nullptr,1e-6),"unsolved zero-barrier inner system admitted");
 require(a.update_epoch({1},.5,1e-7,0,1,nullptr,1e-6),"solved zero-barrier epoch not admitted");
 require(a.rho==.1&&a.epochs==1,"failed true-feasibility progress did not reduce rho");
 require(!a.update_epoch({1},.01,1e-7,0,1,nullptr,1e-6),"duplicate-generation epoch admitted");
 a.reset({0},1);
 require(!a.update_epoch({1},.1,.01,1e-4,1,nullptr,1),"positive-barrier gate relaxed");
 require(a.update_epoch({1},.1,1e-4,1e-4,1,nullptr,1e-6),"existing positive-barrier gate rejected");
 require(a.multiplier_center==Vec({1})&&a.rho==1,"successful progress did not update multiplier center");
 std::cout<<"PASS 8 zero-barrier epoch and positive-barrier isolation invariants\n";
}
