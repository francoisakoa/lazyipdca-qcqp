#include "../src/rho_decay_feasibility_guard.hpp"
#include <iostream>
#include <stdexcept>
using lazyipdca::qcqp::detail::rho_decay_feasibility_gate;
int main() {
  int checks=0;auto check=[&](bool ok,const char* why){++checks;if(!ok)throw std::runtime_error(why);};
  const double eps=std::numeric_limits<double>::epsilon();
  check(rho_decay_feasibility_gate(0,1e-6,1e-6).allowed,"feasible point blocked");
  check(rho_decay_feasibility_gate(0,1e-6+eps,1e-6).allowed,"roundoff ignored floor missing");
  check(!rho_decay_feasibility_gate(0,1e-6+128*eps,1e-6).allowed,"material feasible-neighborhood loss allowed");
  check(rho_decay_feasibility_gate(.1,.09,1e-6).allowed,"improved infeasibility blocked");
  check(rho_decay_feasibility_gate(.1,.1+eps,1e-6).allowed,"roundoff increase blocked");
  check(!rho_decay_feasibility_gate(.1,.1001,1e-6).allowed,"infeasibility increase allowed");
  check(!rho_decay_feasibility_gate(.002009,.02175,1e-6).allowed,"dual improvement could hide primal loss");
  check(rho_decay_feasibility_gate(2e-7,9e-7,1e-6).allowed,"within final tolerance blocked");
  check(rho_decay_feasibility_gate(1e30,1e30*(1+eps),1e-6).allowed,"relative roundoff floor missing");
  for(double invalid:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    check(!rho_decay_feasibility_gate(invalid,0,1e-6).allowed,"invalid previous residual admitted");
    check(!rho_decay_feasibility_gate(0,invalid,1e-6).allowed,"invalid current residual admitted");
    check(!rho_decay_feasibility_gate(0,0,invalid).allowed,"invalid tolerance admitted");
  }
  check(!rho_decay_feasibility_gate(0,0,0).allowed,"zero tolerance admitted");
  std::cout<<"rho_decay_feasibility_guard PASS checks="<<checks<<"\n";
}
