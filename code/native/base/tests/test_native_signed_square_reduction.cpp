#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/native_signed_square_reduction.hpp"
#include <cassert>
#include <iostream>
using namespace lazyipdca::qcqp;
QuadraticFunction q(int n,const char* name){QuadraticFunction f;f.name=name;f.linear.assign(n,0);f.hessian.n=n;return f;}
QCQPProblem input(double x,double scale=1){
 QCQPProblem p;p.name="generic_abs";p.n=3;p.x0={x,2,std::abs(x)};p.lb={-INFINITY,-INFINITY,0};p.ub={INFINITY,INFINITY,INFINITY};
 p.objective=q(3,"f");p.objective.linear[1]=.1;p.objective.hessian.entries={{0,2,3},{2,2,2}};
 auto c=q(3,"arbitrary_name");c.hessian.entries={{0,0,-2*scale},{2,2,2*scale}};p.equalities.push_back(c);
 auto g=q(3,"law");g.linear[1]=1;g.hessian.entries={{0,2,2}};p.equalities.push_back(g);return p;
}
int main(){
 for(double x:{-2.,0.,3.,1e-300})for(double scale:{1.,-4.}) {
  auto p=input(x,scale);NativeSignedSquareReduction red(p);assert(red.changed());assert(red.reduced.n==2);assert(red.links.size()==1);
  assert(red.reduced.x0==Vec({x,2}));assert(red.reduced.objective.signed_square.size()==1);
  Result r;r.x=red.reduced.x0;r.certificate_equality_multipliers={7};r.certificate_lower_multipliers={0,0};r.certificate_upper_multipliers={0,0};red.lift(r);
  assert(r.x==p.x0);assert(!r.termination_audit_valid);
  Vec g=p.objective.gradient(r.x);for(int e=0;e<2;++e)axpy(r.certificate_equality_multipliers[e],p.equalities[e].gradient(r.x),g);
  assert(std::abs(g[2])<1e-12);assert(std::abs(g[0]-(34*std::abs(x)+2*x))<1e-12);assert(g[1]==7.1);
 }
 {auto p=input(1);p.objective.linear[2]=1;NativeSignedSquareReduction r(p);assert(!r.changed());}
 {auto p=input(1);p.objective.hessian.entries.push_back({1,2,1});NativeSignedSquareReduction r(p);assert(!r.changed());}
 {auto p=input(1);p.x0[2]=1.01;NativeSignedSquareReduction r(p);assert(!r.changed());}
 {auto p=input(1);p.lb[2]=.1;NativeSignedSquareReduction r(p);assert(!r.changed());}
 {auto p=input(1);p.ub[2]=2;NativeSignedSquareReduction r(p);assert(!r.changed());}
 {auto p=input(1);p.equalities[0].hessian.entries[0].v=-2.001;NativeSignedSquareReduction r(p);assert(!r.changed());}
 std::cout<<"native signed-square symbolic lift and KKT mapping: PASS\n";
}
