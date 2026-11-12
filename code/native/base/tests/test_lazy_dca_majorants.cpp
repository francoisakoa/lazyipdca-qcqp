#include "lazyipdca_qcqp/curvature.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;

static double correction_quad(const Convexification& c,const Vec& d){
  double v=c.scalar_shift*dot(d,d);
  for(const auto& t:c.terms){ double a=dot(t.direction,d); v+=t.beta*a*a; }
  return v;
}

int main(){
  try{
    Settings st; st.verbose=false; st.exact_eigen_threshold=64; st.curvature_tolerance=1e-12;
    CurvatureOracle oracle(st);
    QuadraticFunction q;
    q.linear={0.3,-0.2}; q.hessian.n=2;
    q.hessian.entries={{0,0,-2.0},{1,1,1.0}};
    Convexification c; oracle.ensure_convex(q.hessian,c,0.0,8);
    if(c.rank()==0 && !(c.scalar_shift>0.0)) throw std::runtime_error("negative curvature was not corrected");
    Vec xk={0.4,-0.1};
    for(Vec d:std::vector<Vec>{{0.0,0.0},{0.2,-0.3},{-0.7,0.4},{1.1,-0.2}}){
      Vec x=xk; axpy(1.0,d,x);
      const double orig=q.value(x);
      const double maj=orig+0.5*correction_quad(c,d);
      if(maj+1e-12<orig) throw std::runtime_error("majorant inequality F<=F_k failed");
      if(d[0]==0.0 && d[1]==0.0 && std::abs(maj-orig)>1e-14)
        throw std::runtime_error("majorant tangency failed");
    }

    // One-dimensional nonconvex QCQP.  The constraint is initially omitted,
    // the first unconstrained convex candidate violates it, and the Lazy-DCA
    // working set must add its DC majorant and converge to x=+1.
    QCQPProblem p; p.name="lazy_dca_expand"; p.n=1; p.x0={1.5}; p.lb={-2.0}; p.ub={2.0};
    p.objective.name="f0"; p.objective.constant=0.0; p.objective.linear={0.0}; p.objective.hessian.n=1;
    p.objective.hessian.entries={{0,0,1.0}}; // 1/2 x^2
    QuadraticFunction g; g.name="g1"; g.constant=1.0; g.linear={0.0}; g.hessian.n=1;
    g.hessian.entries={{0,0,-2.0}}; // 1-x^2 <= 0
    p.inequalities.push_back(g);
    p.validate();

    Settings sd;
    sd.outer_algorithm="lazy-dca"; sd.inner_backend="ipm"; sd.verbose=false;
    sd.max_outer_iterations=300; sd.max_inner_iterations=300;
    sd.feasibility_tolerance=1e-6; sd.stationarity_tolerance=1e-5;
    sd.dual_infeasibility_tolerance=1e-5; sd.complementarity_tolerance=1e-5;
    sd.inner_tolerance=1e-8; sd.working_set_activation=1e-4;
    Solver solver(sd); Result r=solver.solve(p);
    if(r.status!=Status::Solved)
      throw std::runtime_error(std::string("Lazy-DCA did not solve expansion test: ")+status_string(r.status)+" / "+r.message);
    if(r.working_set_additions<1) throw std::runtime_error("lazy working-set expansion was not exercised");
    if(r.feasibility.total_inf>1e-6) throw std::runtime_error("final feasibility failed");
    if(std::abs(r.x[0]-1.0)>5e-4) throw std::runtime_error("unexpected DCA fixed point");
    if(r.inequality_total_rank<1) throw std::runtime_error("constraint lazy correction rank missing");
    std::cout << "[PASS] v0.4.0 DC majorant + lazy working-set contract x="<<r.x[0]
              <<" f="<<r.objective<<" add="<<r.working_set_additions<<" rank="<<r.inequality_total_rank<<"\n";
    return 0;
  }catch(const std::exception& e){ std::cerr<<"[FAIL] "<<e.what()<<"\n"; return 1; }
}
