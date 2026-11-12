#include "lazyipdca_qcqp/hybrid_source_soc.hpp"
#include "lazyipdca_qcqp/hybrid_inexact.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
int main(){try {
 // Original h(x)=x0^2+x1-1=0; anchor (0,1), source d=(t,0).
 // Its affine equality is d1=0. SOC delta=(0,-t^2) restores
 // ORIGINAL feasibility while deliberately not solving d1=0 again.
 for(double t:{.01,.1,1.}){
  Vec x={0,1},d={t,0},delta={0,-t*t};
  double source_h=t*t,corrected_h=t*t+(1-t*t)-1;
  require(std::abs(corrected_h)<1e-15&&source_h>0,"original curved equality not corrected");
  require(std::abs(delta[1])>1e-6,"affine-model residual conflict not exercised");
  auto size=hybrid_source_soc_size(x,d,1,delta);require(size.passed,"genuine second-order correction refused");
  auto source=hybrid_inner::check_model(0,t,1/t,.25,1e-10,0,1e-6,0,1e-6,-t/2,0);
  require(source.passed,"exact convex source minimizer not certified");
 }
 require(!hybrid_source_soc_size({0,1},{.1,0},1,{0,-.1}).passed,"first-order correction accepted as SOC");
 require(!hybrid_source_soc_size({0,1},{1e-10,0},1,{0,std::nextafter(1.,0.)-1}).passed,"rounded adjacent SOC violates order-two bound");
 require(!hybrid_source_soc_size({0,1},{.1,0},1,{0,NAN}).passed,"invalid SOC accepted");
 require(!hybrid_source_soc_size({0,1},{.1},1,{0,0}).passed,"SOC dimension mismatch accepted");
 std::cout<<"PASS source SOC: curved original equality, certified affine source, second-order size, first-order and invalid corrections refused\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
