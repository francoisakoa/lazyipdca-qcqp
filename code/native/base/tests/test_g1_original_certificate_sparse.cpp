#include "lazyipdca_qcqp/lazy_newton.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool v,const char* message){if(!v)throw std::runtime_error(message);}
void near(double a,double b){check(std::abs(a-b)<=1e-12*std::max({1.0,std::abs(a),std::abs(b)}),"sparse audit differs from original dense equations");}
int main(){try{
 std::mt19937 rng(71);std::uniform_real_distribution<double> uniform(-1,1);
 for(int trial=0;trial<100;++trial) {
  QCQPProblem p;p.n=9;p.x0=Vec(p.n);p.lb=Vec(p.n,-2);p.ub=Vec(p.n,2);
  for(double& v:p.x0)v=uniform(rng);p.lb[0]=p.ub[0]=p.x0[0];p.lb[1]=-INFINITY;p.ub[2]=INFINITY;
  auto q=[&](){QuadraticFunction f;f.constant=uniform(rng);f.linear=Vec(p.n,0.0);f.hessian.n=p.n;
   for(int j=0;j<p.n;++j)if(j%3==trial%3)f.linear[j]=uniform(rng);
   for(int j=0;j<p.n;++j)for(int k=j;k<p.n;++k)if((j+k+trial)%4==0)f.hessian.entries.push_back({j,k,uniform(rng)});
   return f;};
  p.objective=q();for(int i=0;i<7;++i)p.equalities.push_back(q());for(int i=0;i<8;++i)p.inequalities.push_back(q());
  Result base;base.x=p.x0;base.certificate_equality_multipliers=Vec(7);for(double& v:base.certificate_equality_multipliers)v=uniform(rng);
  base.certificate_inequality_multipliers=Vec(8);for(double& v:base.certificate_inequality_multipliers)v=std::abs(uniform(rng));
  base.certificate_lower_multipliers=Vec(p.n,.2);base.certificate_upper_multipliers=Vec(p.n,.3);
  base.certificate_lower_multipliers[1]=0;base.certificate_upper_multipliers[2]=0;
  Settings sparse,dense;dense.lazy_newton_sparse_certificate=false;Result a=base,b=base;
  check(certify_lazy_newton_original(p,sparse,a)==certify_lazy_newton_original(p,dense,b),"acceptance differs from dense certificate");
  near(a.objective,b.objective);near(a.feasibility.equality_inf,b.feasibility.equality_inf);near(a.feasibility.inequality_inf,b.feasibility.inequality_inf);
  near(a.feasibility.bound_inf,b.feasibility.bound_inf);near(a.dual_infeasibility_inf,b.dual_infeasibility_inf);near(a.complementarity_inf,b.complementarity_inf);
 }
 QCQPProblem p;p.n=1;p.x0={1};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={-2};p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,2}};
 Result r;r.x={1};r.certificate_lower_multipliers={0};r.certificate_upper_multipliers={0};Settings s;
 check(certify_lazy_newton_original(p,s,r),"analytic original minimizer rejected");
 r.certificate_lower_multipliers[0]=1;
 check(!certify_lazy_newton_original(p,s,r)&&!r.ipopt_desired_convergence,"absent-bound dual or stale success flag accepted");
 r.certificate_lower_multipliers[0]=0;r.x[0]=NAN;check(!certify_lazy_newton_original(p,s,r),"nonfinite primal certified");
 std::cout<<"PASS original sparse/dense audit parity, fixed and absent bounds, invalid certificates\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
