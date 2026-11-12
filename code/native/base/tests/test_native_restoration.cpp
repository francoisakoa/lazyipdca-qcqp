// Include the implementation to check its dense, matrix-free and preconditioner
// curvature representations independently. Checks remain active in Release.
#include "../src/restoration.cpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void check(bool yes,const char* message) {if(!yes)throw std::runtime_error(message);}
void close(double a,double b,const char* message) {check(std::abs(a-b)<1e-13,message);}
}
int main(){
 QuadraticFunction q;q.linear={0};q.hessian.n=1;q.hessian.entries={{0,0,2}};q.signed_square={{0,-1.5}};
 for(double x:{-2.,0.,3.}) {
  Vec point{x},v{4},scratch,y;std::vector<AbsLink>links;
  reduced_hessian_apply_into(q,point,v,links,scratch,y);
  const double h=2-3*((x>0)-(x<0));close(y[0],4*h,"native Hessian matvec");
  std::vector<double>A(1,0);add_transformed_hessian_dense(A,1,{0},q,1.3,point,links);close(A[0],1.3*h,"native dense Hessian");
  ResidualModel rm;rm.residual={1.3};rm.rows={{1}};rm.functions={&q};Vec diag,rad;
  transformed_curvature_gershgorin(rm,point,links,diag,rad);close(diag[0],1.3*h,"native Hessian preconditioner");check(rad[0]==0,"native diagonal radius");
 }
 QCQPProblem p;p.n=1;p.name="deadline";p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective.linear={0};p.objective.hessian.n=1;
 auto h=p.objective;h.constant=-1;h.linear={1};p.equalities={h};
 Settings st;st.verbose=false;st.restoration_deadline_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()-1;
 for(bool native:{false,true}) {
  if(native)p.equalities[0].signed_square={{0,1}};
  auto r=FeasibilityRestoration(st).solve(p,p.x0);
  check(r.iterations==0&&r.linear_solves==0&&!r.success&&r.x==p.x0,"expired deadline must preserve start and perform no iterations");
  check(r.message=="Phase-I elapsed-time limit","expired deadline message");
 }
 // Native Phase I deliberately uses the existing Gauss--Newton method; it
 // must not count these steps as solves of the exact merit Hessian.
 p.name="native_gauss_newton";p.x0={.5};p.equalities[0].linear={0};
 st.restoration_deadline_seconds=INFINITY;st.feasibility_tolerance=1e-8;
 auto r=FeasibilityRestoration(st).solve(p,p.x0);
 check(r.success&&r.final_feasibility<=st.feasibility_tolerance,"native Gauss-Newton feasibility");
 check(r.linear_solves>0&&r.exact_hessian_solves==0,"native Gauss-Newton accounting");
 std::cout<<"Native restoration curvature, Gauss-Newton accounting and deadlines: PASS\n";
}
