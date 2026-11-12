#include "gr_bound_stagnation_watch.hpp"
#include <stdexcept>
using lazyipdca::qcqp::detail::GrBoundStagnationWatch;
void check(bool b){if(!b)throw std::runtime_error("bound stagnation admission");}
int main(){
 GrBoundStagnationWatch w;
 for(int i=1;i<8;++i)check(!w.observe(true,i,1e-6,.01,.1,.001,1e-6));
 check(!w.observe(true,7,1e-6,.01,.1,.001,1e-6));
 check(w.observe(true,8,1e-6,.01,.1,.001,1e-6)&&w.active);
 check(!w.observe(true,9,1e-6,.01,.1,.001,1e-6));
 GrBoundStagnationWatch progress;
 for(int i=1;i<=20;++i)check(!progress.observe(true,i,1e-6,.01*std::pow(.98,i),.1,.001,1e-6));
 GrBoundStagnationWatch epochs;
 for(int i=1;i<=20;++i)check(!epochs.observe(true,i,1e-6,.01,.1,.001/std::pow(2,i),1e-6));
 GrBoundStagnationWatch disabled;
 for(int i=1;i<=20;++i)check(!disabled.observe(false,i,1e-6,.01,.1,.001,1e-6));
 GrBoundStagnationWatch normal;
 for(int i=1;i<=20;++i)check(!normal.observe(true,i,.1,.01,.1,.001,1e-6));
}
