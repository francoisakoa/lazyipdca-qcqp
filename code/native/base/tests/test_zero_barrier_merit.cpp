#include "lazyipdca_qcqp/primal_dual_merit.hpp"
#include <stdexcept>
#include <iostream>
using namespace lazyipdca::qcqp::detail;
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){
 gr::State a;a.objective=1;a.c={.2};a.y={.4};
 gr::Parameters p;p.mu=0;p.rho=.1;p.kappa=1;p.center={0};
 gr::Direction d;d.objective=-1;d.c={-.1};d.y={.2};
 const auto zero=gr::evaluate(a,p),slope=gr::directional(a,d,p);
 require(zero.valid&&slope.valid,"zero-barrier equality merit/slope must be finite");
 p.mu=.1;const auto positive=gr::evaluate(a,p);
 require(positive.valid&&positive.value==zero.value,"no-barrier merit must be independent of mu");
 p.mu=-1;require(!gr::valid(a,p),"negative mu accepted");
 p.mu=std::numeric_limits<double>::infinity();require(!gr::valid(a,p),"nonfinite mu accepted");
 p.mu=0;a.s={1};require(!gr::valid(a,p),"zero mu accepted with active slack barrier");
 p.mu=.1;require(gr::valid(a,p),"positive mu rejected with slack barrier");
 p.mu=0;a.s.clear();a.b={0};require(!gr::valid(a,p),"zero mu accepted with active bound link");
 a.b.clear();a.c.push_back(0);a.y.push_back(0);p.center.push_back(0);
 require(gr::evaluate(a,p).valid&&gr::evaluate(a,p).value==zero.value,"zero original equation changes merit");
 std::cout<<"PASS 8 zero-barrier domain and zero-equation invariants\n";
}
