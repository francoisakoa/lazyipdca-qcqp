#include "lazyipdca_qcqp/inner_ipm.hpp"
#include "lazyipdca_qcqp/inner_lazyqp.hpp"
#include "lazyipdca_qcqp/inner_sqp.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static ConvexSubproblem make_problem(){
  ConvexSubproblem p; p.nvar=1; p.original_dimension=1;
  p.objective.constant=0.5; p.objective.linear={-1.0}; p.objective.base_hessian.n=1;
  p.objective.base_hessian.entries={{0,0,1.0}}; // 1/2 (y-1)^2
  ConvexQuadraticModel lo; lo.constant=0.0; lo.linear={-1.0}; lo.base_hessian.n=1; lo.shift_dimension=1; // y>=0
  ConvexQuadraticModel hi; hi.constant=-2.0; hi.linear={1.0}; hi.base_hessian.n=1; hi.shift_dimension=1; // y<=2
  p.inequalities={lo,hi};
  p.equalities.nvar=1;
  return p;
}


static ConvexSubproblem make_quadratic_problem(){
  ConvexSubproblem p; p.nvar=1; p.original_dimension=1;
  p.objective.constant=0.0; p.objective.linear={-1.0}; p.objective.base_hessian.n=1; // min -y
  ConvexQuadraticModel ball; ball.constant=-1.0; ball.linear={0.0}; ball.base_hessian.n=1;
  ball.base_hessian.entries={{0,0,2.0}}; // y^2 - 1 <= 0
  p.inequalities={ball}; p.equalities.nvar=1; p.prefer_feasible_centered_start=true;
  return p;
}


static ConvexSubproblem make_scaled_quadratic_problem(){
  ConvexSubproblem p; p.nvar=2; p.original_dimension=2;
  p.objective.constant=0.0; p.objective.linear={-0.0945696,0.0}; p.objective.base_hessian.n=2;
  ConvexQuadraticModel c1; c1.constant=-1e-4; c1.linear={0.0,0.0}; c1.base_hessian.n=2;
  c1.base_hessian.entries={{0,0,2e6},{1,1,2.0}}; // 1e6 x^2 + y^2 - 1e-4 <=0
  ConvexQuadraticModel c2; c2.constant=-1.0; c2.linear={0.0,0.0}; c2.base_hessian.n=2;
  c2.base_hessian.entries={{0,0,2.0},{1,1,2e6}};
  p.inequalities={c1,c2}; p.equalities.nvar=2; p.prefer_feasible_centered_start=true;
  return p;
}

static ConvexSubproblem make_active_boundary_quadratic_problem(){
  ConvexSubproblem p; p.nvar=1; p.original_dimension=1;
  p.objective.constant=0.0; p.objective.linear={-0.0945696}; p.objective.base_hessian.n=1;
  ConvexQuadraticModel c; c.constant=0.0; c.linear={-1.0}; c.base_hessian.n=1;
  c.base_hessian.entries={{0,0,1e6}}; // 5e5 x^2 - x <= 0, active at x=0
  p.inequalities={c}; p.equalities.nvar=1; p.prefer_feasible_centered_start=true;
  return p;
}

static void check_active_boundary_lazyqp(const InnerResult& r){
  if(!r.success) throw std::runtime_error("LazyQP active-boundary quadratic failed: "+r.message);
  if(r.message.find("nonlinear-residual")==std::string::npos)
    throw std::runtime_error("LazyQP active-boundary quadratic did not use nonlinear-residual path");
  const double xstar=2e-6;
  if(r.y.size()!=1 || std::abs(r.y[0]-xstar)>2e-7)
    throw std::runtime_error("LazyQP active-boundary quadratic wrong minimizer");
  if(r.primal_residual_inf>1e-6 || r.inequality_inf>1e-6 || r.dual_inf>1e-5)
    throw std::runtime_error("LazyQP active-boundary quadratic KKT residual too large");
  std::cout<<"[PASS] LazyQP active-boundary quadratic x="<<r.y[0]
           <<" dual="<<r.dual_inf<<" pri="<<r.primal_residual_inf
           <<" it="<<r.iterations<<"\n";
}

static void check_scaled_lazyqp(const InnerResult& r){
  if(!r.success) throw std::runtime_error("LazyQP scaled quadratic failed: "+r.message);
  if(r.message.find("nonlinear-residual")==std::string::npos)
    throw std::runtime_error("LazyQP scaled quadratic used fallback instead of primal barrier path");
  if(r.y.size()!=2 || std::abs(r.y[0]-1e-5)>2e-7 || std::abs(r.y[1])>1e-7)
    throw std::runtime_error("LazyQP scaled quadratic wrong minimizer");
  if(r.dual_inf>1e-6 || r.primal_residual_inf>1e-7)
    throw std::runtime_error("LazyQP scaled quadratic KKT residual too large");
  std::cout<<"[PASS] LazyQP scaled quadratic y="<<r.y[0]
           <<" dual="<<r.dual_inf<<" pri="<<r.primal_residual_inf
           <<" it="<<r.iterations<<"\n";
}

static void check_quad(const char* name,const InnerResult& r){
  if(!r.success) throw std::runtime_error(std::string(name)+" quadratic failed: "+r.message);
  if(std::string(name)=="LazyQP" && r.message.find("nonlinear-residual")==std::string::npos)
    throw std::runtime_error("LazyQP quadratic test used fallback instead of primal barrier path");
  if(r.y.size()!=1 || std::abs(r.y[0]-1.0)>5e-3)
    throw std::runtime_error(std::string(name)+" quadratic wrong minimizer y="+std::to_string(r.y.empty()?999:r.y[0]));
  if(r.primal_residual_inf>1e-5) throw std::runtime_error(std::string(name)+" quadratic primal residual");
  std::cout<<"[PASS] "<<name<<" quadratic y="<<r.y[0]<<" it="<<r.iterations<<" fact="<<r.factorizations<<"\n";
}

static void check(const char* name,const InnerResult& r){
  if(!r.success) throw std::runtime_error(std::string(name)+" failed: "+r.message);
  if(r.y.size()!=1 || std::abs(r.y[0]-1.0)>5e-4)
    throw std::runtime_error(std::string(name)+" wrong minimizer");
  if(r.primal_residual_inf>1e-5) throw std::runtime_error(std::string(name)+" primal residual");
  std::cout<<"[PASS] "<<name<<" y="<<r.y[0]<<" it="<<r.iterations<<" fact="<<r.factorizations<<"\n";
}

int main(){
  try{
    ConvexSubproblem p=make_problem();
    Settings st; st.verbose=false; st.inner_tolerance=1e-8; st.max_inner_iterations=100;
    ConvexQCQPLazyQPKernel lazyqp(st); check("LazyQP",lazyqp.solve(p));
    ConvexQCQPInteriorPoint ipm(st); check("legacy-IPM",ipm.solve(p));
    ConvexQCQPFilterSQP fsqp(st); check("filter-SQP",fsqp.solve(p));
    ConvexQCQPTrustRegionSQP trsqp(st); check("TR-SQP",trsqp.solve(p));
    ConvexSubproblem q=make_quadratic_problem();
    ConvexQCQPLazyQPKernel qlazy(st); check_quad("LazyQP",qlazy.solve(q));
    ConvexQCQPInteriorPoint qipm(st); check_quad("legacy-IPM",qipm.solve(q));
    ConvexSubproblem qs=make_scaled_quadratic_problem();
    ConvexQCQPLazyQPKernel qsl(st); check_scaled_lazyqp(qsl.solve(qs));
    ConvexSubproblem qb=make_active_boundary_quadratic_problem();
    ConvexQCQPLazyQPKernel qbl(st); check_active_boundary_lazyqp(qbl.solve(qb));
    return 0;
  }catch(const std::exception& e){ std::cerr<<"[FAIL] "<<e.what()<<"\n"; return 1; }
}
