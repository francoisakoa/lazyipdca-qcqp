#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool b,const char* m) { if(!b) throw std::runtime_error(m); }
void close(double a,double b) { require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<1e-10*(1+std::abs(a)),"affine arithmetic mismatch"); }
int main() {
 try {
  ConvexSubproblem p;
  p.nvar=p.original_dimension=8; p.prefer_feasible_centered_start=true;
  p.objective.linear.assign(8,0);p.objective.linear[1]=-1;
  p.objective.base_hessian.n=8;p.objective.shift_dimension=8;p.objective.scalar_shift=.2;
  for(int i=0;i<8;++i)p.objective.base_hessian.entries.push_back({i,i,2.0});
  p.objective.low_rank.push_back({.3,{1,0,0,0,0,0,0,1}});
  p.equalities.nvar=8;p.equalities.rows={{{{0,-2},{1,1},{7,1}}}};p.equalities.rhs={.5};
  p.exact_affine_eliminations={{0,7}};
  ConvexQuadraticModel g;g.constant=-10;g.linear.assign(8,0);g.linear[2]=.2;g.base_hessian.n=8;
  g.base_hessian.entries={{2,2,1},{2,7,.1},{7,7,1}};p.inequalities={g};
  ExactAffineSubproblemReduction dense,sparse;std::string message;
  require(build_exact_affine_subproblem_reduction(p,dense,message,false),message.c_str());
  require(build_exact_affine_subproblem_reduction(p,sparse,message,true),message.c_str());
  for(int k=0;k<10;++k) {
   Vec z(7);for(int j=0;j<7;++j)z[j]=.1*std::sin(j+k);
   close(dense.reduced.objective.value(z),sparse.reduced.objective.value(z));
   close(dense.reduced.inequalities[0].value(z),sparse.reduced.inequalities[0].value(z));
   auto a=dense.reduced.inequalities[0].gradient(z),b=sparse.reduced.inequalities[0].gradient(z);
   for(int j=0;j<7;++j)close(a[j],b[j]);
  }
  Settings st;st.verbose=false;st.linear_backend="native";st.lazyqp_flow_lift_elimination=true;
  st.inner_tolerance=1e-9;st.max_inner_iterations=100;
  auto a=ConvexQCQPLazyQPKernel(st).solve(p);
  st.lazyqp_affine_support_fastpath=true;
  auto b=ConvexQCQPLazyQPKernel(st).solve(p);
  require(a.success&&b.success,"sparse/dense affine solve failed");
  require(a.y.size()==b.y.size(),"primal size mismatch");
  for(std::size_t j=0;j<a.y.size();++j) close(a.y[j],b.y[j]);
  close(a.dual_inf,b.dual_inf);close(a.complementarity,b.complementarity);
  auto invalid=p;invalid.inequalities[0].linear[6]=std::numeric_limits<double>::quiet_NaN();
  require(!build_exact_affine_subproblem_reduction(invalid,sparse,message,true),"NaN bypassed sparse validation");
  require(sparse.reduced.nvar==0,"failed reduction retained stale state");

  ConvexSubproblem t;t.nvar=t.original_dimension=3;t.prefer_feasible_centered_start=true;
  t.objective.linear={0,0,0};t.objective.base_hessian.n=3;
  t.objective.base_hessian.entries={{0,0,-100},{1,1,2},{2,2,1}};
  t.equalities.nvar=3;t.equalities.rows={{{{0,1}}},{{{0,-1},{2,1}}}};t.equalities.rhs={0,0};
  t.exact_affine_eliminations={{1,2}};
  st.kkt_adaptive_regularization=true;st.linear_metis=false;
  auto factor=make_symmetric_factorization(st,message);
  require(bool(factor),"missing factor");
  require(certify_affine_tangent_inertia(t,*factor,message),"normal negative curvature incorrectly rejected");
  t.objective.base_hessian.entries[1].v=-2;
  require(!certify_affine_tangent_inertia(t,*factor,message),"negative tangent curvature accepted");
  t.objective.base_hessian.entries[1].v=2;
  t.equalities.rows.push_back(t.equalities.rows[0]);t.equalities.rhs.push_back(0);
  require(!certify_affine_tangent_inertia(t,*factor,message),"singular Jacobian accepted");
  std::cout<<"sparse affine audits and reduced tangent inertia PASS\n";
 } catch(const std::exception& e) { std::cerr<<e.what()<<"\n";return 1; }
}
