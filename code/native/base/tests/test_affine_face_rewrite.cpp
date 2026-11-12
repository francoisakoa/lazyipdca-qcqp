#include "affine_face_quadratic_rewrite.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;using namespace lazyipdca::qcqp::detail;
int main(){
 auto require=[](bool a,const char* s){if(!a)throw std::runtime_error(s);};
 QCQPProblem p;p.n=3;p.name="exact-face-fixture";p.x0={1,0,1};p.lb={0,0,0};p.ub={INFINITY,INFINITY,INFINITY};p.objective.hessian.n=3;p.objective.compact_linear_dimension=3;
 QuadraticFunction h;h.hessian.n=3;h.constant=-1;h.compact_linear_dimension=3;h.compact_linear={{0,1},{1,1}};p.equalities={h};
 QuadraticFunction q;q.hessian.n=3;q.compact_linear_dimension=3;q.hessian.entries={{0,0,-2},{1,1,2},{2,2,2}};p.inequalities={q};p.validate();
 AffineFaceQuadraticRewrite a(p,INFINITY);require(a.applied(),"exact proof absent");require(a.model.x0==p.x0,"x0 changed");require(a.model.inequalities[0].hessian.entries.size()==1,"square pair not removed");
 for(Vec x:std::vector<Vec>{{1,0,1},{.75,.25,.2},{.25,.75,.7},{.8,.3,.2}}){
   const double lhs=p.inequalities[0].value(x),rhs=a.model.inequalities[0].value(x)- (x[0]-x[1])*p.equalities[0].value(x);require(std::abs(lhs-rhs)<1e-15,"identity failed");
 }
 // At a nonzero h, the lifted gradient retains z*k*h*grad(t-u).
 // It must not be hidden by reusing a transformed-space success flag.
 { const Vec x={.8,.3,.2};const double z=3,k=-1;
   auto old_gradient=p.inequalities[0].gradient(x),new_gradient=a.model.inequalities[0].gradient(x);
   const auto hg=p.equalities[0].gradient(x);const double factor=k*(x[0]-x[1]);
   for(int j=0;j<3;++j){const double remainder=z*k*p.equalities[0].value(x)*(j==0?1:j==1?-1:0);
     require(std::abs(z*(old_gradient[j]-new_gradient[j]-factor*hg[j])-remainder)<1e-14,
         "off-manifold lifted-gradient remainder failed");}
 }
 Result r;r.x={.75,.25,.2};r.certificate_equality_multipliers={2};r.certificate_inequality_multipliers={3};require(a.lift(r),"lift rejected");require(r.certificate_equality_multipliers[0]==3.5,"dual lift sign failed");
 auto scaled=p;scaled.equalities[0].compact_linear={{0,2},{1,2}};scaled.equalities[0].constant=-2;
 AffineFaceQuadraticRewrite scale_proof(scaled,INFINITY);require(scale_proof.applied(),"exact scaled face rejected");
 Result invalid;r.x={.75,.25,.2};invalid.x=r.x;invalid.certificate_equality_multipliers={2};invalid.certificate_inequality_multipliers={INFINITY};
 require(!a.lift(invalid)&&invalid.certificate_equality_multipliers[0]==2,"nonfinite failed lift mutated certificate");
 require(!AffineFaceQuadraticRewrite(p,0).applied(),"expired proof accepted");
 auto oversized=p;oversized.name.assign(2000001,'x');
 require(!AffineFaceQuadraticRewrite(oversized,INFINITY).applied(),"oversized source copied before admission");
 auto bad=p;bad.equalities[0].compact_linear[1].second=2;require(!AffineFaceQuadraticRewrite(bad,INFINITY).applied(),"unequal coefficients accepted");
 bad=p;bad.equalities[0].hessian.entries={{0,0,2}};require(!AffineFaceQuadraticRewrite(bad,INFINITY).applied(),"quadratic face accepted");
 bad=p;bad.inequalities[0].hessian.entries.push_back({0,2,1});require(!AffineFaceQuadraticRewrite(bad,INFINITY).applied(),"cross term accepted");
 bad=p;bad.x0={0,0,1};require(!AffineFaceQuadraticRewrite(bad,INFINITY).applied(),"off-face raw start accepted");
 bad=p;bad.equalities[0].compact_linear={{0,3},{1,3}};bad.equalities[0].constant=-3;require(!AffineFaceQuadraticRewrite(bad,INFINITY).applied(),"inexact scale accepted");
 std::cout<<"PASS exact identity, raw start, dual lift, unequal/quadratic/cross/off-face/inexact rejection\n";
}
