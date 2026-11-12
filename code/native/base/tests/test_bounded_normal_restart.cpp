#include "lazyipdca_qcqp/bounded_normal_restart.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main() {
 using detail::BoundedNormalColumn;
 // Parallel normals permit arbitrary huge canceling duals. Starting at zero
 // must recover the small representative without preserving that gauge.
 BoundedNormalColumn a;a.entries={{0,1}};
 auto r=detail::bounded_normal_restart({1},{a,a},1);
 require(r.valid&&r.multipliers.size()==2,"parallel-normal fit failed");
 require(std::abs(r.residual[0])<1e-14&&std::abs(r.multipliers[0])<=1&&
         std::abs(r.multipliers[1])<=1,"large canceling gauge retained");
 // A complementarity cap is authoritative, even when it prevents an exact
 // stationarity fit. No negative inequality multiplier may be invented.
 BoundedNormalColumn cap;cap.entries={{0,1}};cap.lower=1e-30;cap.upper=.5;
 r=detail::bounded_normal_restart({-2},{cap},1);
 require(r.valid&&r.multipliers[0]==.5&&r.residual[0]==-1.5,"complementarity cap ignored");
 r=detail::bounded_normal_restart({1},{cap},1);
 require(r.valid&&r.multipliers[0]>=cap.lower&&r.multipliers[0]<=cap.upper&&
         r.residual[0]>=1,"inequality dual sign violated");
 // Different scales must not corrupt the residual against the original
 // normals or alter the coefficient bounds.
 BoundedNormalColumn scaled;scaled.entries={{0,1e9},{1,2e9}};
 r=detail::bounded_normal_restart({-1,-2},{scaled},1);
 require(r.valid&&std::abs(r.multipliers[0]-1e-9)<1e-22&&
         inf_norm(r.residual)<1e-14,"scaled original normal residual wrong");
 auto none=detail::bounded_normal_restart({3,4},{},1);
 require(none.valid&&none.multipliers.empty()&&none.residual==Vec({3,4}),"normal invented");
 require(!detail::bounded_normal_restart({1},{cap},0).valid,"zero budget ignored");
 BoundedNormalColumn invalid=cap;invalid.entries={{1,1}};
 require(!detail::bounded_normal_restart({1},{invalid},1).valid,"invalid normal index accepted");
 invalid=cap;invalid.lower=1;invalid.upper=0;
 require(!detail::bounded_normal_restart({1},{invalid},1).valid,"invalid coefficient interval accepted");
 std::cout<<"bounded_normal_restart: gauge / complementarity caps / signs / scaling / invalid inputs PASS\n";
}
