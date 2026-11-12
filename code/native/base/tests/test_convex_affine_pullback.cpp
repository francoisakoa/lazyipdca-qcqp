#include "lazyipdca_qcqp/convex_affine_pullback.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
int main(){try {
 ConvexAffineCoordinates t;t.ambient=4;t.reduced=2;t.offset={.5,-.25,3,0};t.rows.resize(4);
 t.rows[0].entries={{0,1},{1,-2}};t.rows[1].entries={{1,1}};t.rows[3].entries={{0,-.5},{1,1}};
 ConvexQuadraticModel q;q.constant=7;q.linear={1,-2,.5,3};q.base_hessian.n=4;
 // PSD base including a nonzero off-diagonal; nonzero affine offset and atoms.
 q.base_hessian.entries={{0,0,4},{0,1,2},{1,1,3},{2,2,2},{3,3,1}};
 q.scalar_shift=.125;q.shift_dimension=3;q.low_rank={{.75,{1,0,-1,2}},{2,{0,1,0,-1}}};
 auto r=convex_affine_pullback(q,t);
 for(int i=-4;i<=4;++i)for(int j=-4;j<=4;++j){Vec y={i*.25,j*.125};Vec d=t.expand(y);
  check(std::abs(q.value(d)-r.value(y))<1e-11,"value changed by affine pullback");
  auto a=t.transpose(q.gradient(d)),b=r.gradient(y);
  for(int k=0;k<2;++k)check(std::abs(a[k]-b[k])<1e-11,"chain-rule gradient changed");
  check(r.base_hessian.quad(y)>=-1e-12,"PSD base lost");
 }
 check(t.expand({0,0})==t.offset,"zero reduced point must retain affine offset");
 bool rejected=false;try{auto bad=t;bad.rows[0].entries={{2,1}};convex_affine_pullback(q,bad);}catch(const std::invalid_argument&){rejected=true;}
 check(rejected,"out-of-domain map accepted");
 // Equality x0+2*x1=5 is solved by d=(5-2*y,y), not by T*y alone.
 ConvexAffineCoordinates e;e.ambient=2;e.reduced=1;e.offset={5,0};e.rows.resize(2);e.rows[0].entries={{0,-2}};e.rows[1].entries={{0,1}};
 for(double v:{-2.,0.,3.}){auto d=e.expand({v});check(d[0]+2*d[1]==5,"affine equality not eliminated");}
 std::cout<<"PASS affine convex pullback: values, chain rule, off-diagonal Hessian, atoms, scalar support, fixed coordinate, affine offset and invalid map\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
