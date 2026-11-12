#include "lazyipdca_qcqp/direct_ipopt.hpp"
#include <iostream>
#include <limits>
using namespace lazyipdca::qcqp;
int main(){int n=0;auto ok=[&](DirectIpoptSettings s,bool accept){bool passed=true;try{validate_aligned_ipopt_settings(s);}catch(const std::invalid_argument&){passed=false;}if(passed!=accept)return false;++n;return true;};
for(const auto&b:{"ma27","mumps"})for(const auto&m:{"adaptive","monotone"}){DirectIpoptSettings s;s.linear_solver=b;s.mu_strategy=m;if(!ok(s,true))return 1;}
for(int i=0;i<15;++i){DirectIpoptSettings s;switch(i){case 0:s.mu_strategy="invalid";break;case 1:s.linear_solver="ma57";break;case 2:s.tolerance=1e-6;break;case 3:s.constraint_violation_tolerance=1e-4;break;case 4:s.dual_infeasibility_tolerance=1e-4;break;case 5:s.complementarity_tolerance=1e-4;break;case 6:s.acceptable_iterations=15;break;case 7:s.max_iterations=3000;break;case 8:s.time_limit_seconds=3600;break;case 9:s.time_limit_seconds=std::numeric_limits<double>::quiet_NaN();break;case 10:s.derivative_test=true;break;case 11:s.print_timing_statistics=true;break;case 12:s.print_level=-1;break;case 13:s.print_level=13;break;case 14:s.hsl_library="irrelevant_for_mumps";break;}if(!ok(s,false))return 2;}
DirectIpoptSettings s;s.linear_solver="ma27";s.hsl_library="declared_hsl.so";if(!ok(s,true))return 3;std::cout<<n<<" aligned API checks passed\n";}
