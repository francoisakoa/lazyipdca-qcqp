#include "lazyipdca_qcqp/hybrid_split_step.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool b,const char*s){if(!b)throw std::runtime_error(s);}
int main(){try {
 // min x^2/2, x=1000, -x-1<=0. Initial x=0,s=z=lambda=1.
 // Exact Newton equations at target mu=.1 give dx=ds=1000,
 // dz=-1000.9, dlambda=-2000.9. Dual positivity restricts its
 // fraction, while the primal Newton step satisfies the affine equality.
 const double dx=1000,ds=1000,dz=-1000.9,dl=-2000.9,mu=.1;
 require(std::abs(dx+dl-dz)<1e-10&&-dx+ds==0&&std::abs(ds+dz+.9)<1e-10,"analytic KKT Newton equations changed");
 const double ad=.995/1000.9;auto a=hybrid_split_fractions(1,ad,1);
 double x=a.first*dx,s=1+a.first*ds,z=1+a.second*dz,l=1+a.second*dl;
 require(x==1000&&s>0&&z>0,"split step loses affine feasibility or positivity");
 double norm_before=std::hypot(1000.,1.-mu);
 double norm_after=std::sqrt((x+l-z)*(x+l-z)+(x-1000)*(x-1000)+(-x-1+s)*(-x-1+s)+(s*z-mu)*(s*z-mu));
 require(norm_after<norm_before,"split exact Newton example does not contract complete KKT residual");
 require(std::abs(ad*dx-1000)>999,"common fraction counterexample not exercised");
 auto b=hybrid_split_fractions(1,ad,.5);
 require(b.first==.5&&b.second==.5*ad&&1+b.second*dz>0,"shared backtracking lost domain");
 bool bad=false;try{hybrid_split_fractions(1,ad,NAN);}catch(const std::invalid_argument&){bad=true;}require(bad,"invalid fractions accepted");
 std::cout<<"PASS split primal-dual: exact anonymous KKT equations, full residual contraction, affine feasibility, positivity, common backtracking and invalid domain\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
