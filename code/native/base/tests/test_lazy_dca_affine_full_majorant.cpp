#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

int main() {
  try {
    const double inf=std::numeric_limits<double>::infinity();

    // The affine objective is unbounded if the initially inactive convex row
    // is omitted.  This is the small exact analogue of A4X12's first v0.4.5
    // master: an affine objective and constraints that are responsible for
    // bounding the descent direction.
    QCQPProblem p;
    p.name="affine_full_majorant";
    p.n=1;
    p.x0={0.0};
    p.lb={-inf};
    p.ub={inf};
    p.objective.name="f";
    p.objective.linear={-1.0};
    p.objective.hessian.n=1;

    QuadraticFunction ball;
    ball.name="ball";
    ball.constant=-1.0;
    ball.linear={0.0};
    ball.hessian.n=1;
    ball.hessian.entries={{0,0,2.0}}; // x^2-1 <= 0
    p.inequalities={ball};
    p.validate();

    Settings st;
    st.outer_algorithm="lazy-dca";
    st.inner_backend="lazyqp";
    st.linear_backend="native";
    st.verbose=false;
    st.max_outer_iterations=20;
    st.max_inner_iterations=200;
    st.feasibility_tolerance=1e-6;
    st.stationarity_tolerance=1e-5;
    st.dual_infeasibility_tolerance=1e-5;
    st.complementarity_tolerance=1e-5;
    st.inner_tolerance=1e-8;
    st.inner_iteration_limit_mu_fallback=1e-6;
    // First reproduce the defective restricted master explicitly.  With the
    // safeguard disabled, the only bounding row is omitted at x0=0 and the
    // first convex subproblem cannot deliver an outer step.
    Settings restricted=st;
    restricted.dca_full_constraints_for_affine_objective=false;
    restricted.max_inner_iterations=8;
    const Result broken=Solver(restricted).solve(p);
    if(broken.status==Status::Solved || broken.outer_iterations!=0)
      throw std::runtime_error("restricted affine master unexpectedly entered optimization");

    st.dca_full_constraints_for_affine_objective=true;
    st.dca_affine_full_constraint_limit=16;

    const Result r=Solver(st).solve(p);
    if(r.status!=Status::Solved)
      throw std::runtime_error("affine full-majorant master failed: "+r.message);
    if(r.outer_iterations<1)
      throw std::runtime_error("solver never entered the outer optimization");
    if(r.x.size()!=1 || std::abs(r.x[0]-1.0)>5e-5)
      throw std::runtime_error("wrong affine full-majorant solution");
    if(r.feasibility.total_inf>st.feasibility_tolerance)
      throw std::runtime_error("affine full-majorant solution is infeasible");
    if(r.working_set_additions!=0)
      throw std::runtime_error("full-majorant mode unexpectedly used lazy activation");
    if(r.active_inequalities<1)
      throw std::runtime_error("bounding constraint was not retained");

    std::cout << "[PASS] affine-objective full majorant enters optimization"
              << " outer="<<r.outer_iterations
              << " inner="<<r.inner_iterations
              << " x="<<r.x[0]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
