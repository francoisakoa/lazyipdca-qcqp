#include "lazyipdca_qcqp/dual_step.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using lazyipdca::qcqp::detail::primal_dual_step;
using lazyipdca::qcqp::detail::common_feasible_step;
using lazyipdca::qcqp::detail::compare_common_step;

void require(bool condition,const char* message) {
  if(!condition)throw std::runtime_error(message);
}

int main() {
  try {
    // Reproduce two backtracks (tau=1/4). Ipopt keeps z at its independent
    // fraction-to-boundary step and advances free equality multipliers with x.
    const auto ipopt=primal_dual_step(0.8,0.6,0.25,"ipopt");
    require(std::abs(ipopt.primal-0.2)<1e-15,"Ipopt primal backtrack incorrect");
    require(std::abs(ipopt.bound_dual-0.6)<1e-15,"Ipopt bound-dual step was backtracked");
    require(std::abs(ipopt.equality_dual-0.2)<1e-15,"Ipopt alpha_for_y=primal incorrect");

    const auto coupled=primal_dual_step(0.8,0.6,0.25,"coupled");
    require(std::abs(coupled.primal-0.2)<1e-15,"coupled primal step incorrect");
    require(std::abs(coupled.bound_dual-0.15)<1e-15,"legacy bound-dual coupling incorrect");
    require(std::abs(coupled.equality_dual-0.15)<1e-15,"legacy equality-dual coupling incorrect");

    require(std::abs(common_feasible_step(0.0035,0.0136,0.0136)-0.0035)<1e-15,
            "common step is not jointly feasible");
    require(compare_common_step(0.023,92.4,640.0,0.1,1.25),
            "measured case30000 dual spike did not trigger common-step comparison");
    require(!compare_common_step(44.0,13500.0,25700.0,0.1,1.25),
            "far-from-feasible trajectory incorrectly triggered common-step comparison");
    require(!compare_common_step(0.023,92.4,110.0,0.1,1.25),
            "small dual variation incorrectly triggered common-step comparison");

    bool rejected=false;
    try { (void)primal_dual_step(1,1,1,"unknown"); }
    catch(const std::invalid_argument&) { rejected=true; }
    require(rejected,"invalid dual-step policy accepted");
    std::cout<<"Ipopt and legacy dual-step policies PASS\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
  }
}
