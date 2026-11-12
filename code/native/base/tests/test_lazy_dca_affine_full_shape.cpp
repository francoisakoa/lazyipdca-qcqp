#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

int main(int argc, char** argv) {
  try {
    constexpr int n=130;
    constexpr int mi=384;
    const double inf=std::numeric_limits<double>::infinity();

    // A structural first-subproblem surrogate for A4X12: affine objective,
    // 130 variables, one equality, 384 quadratic inequalities, and many
    // negative-curvature rows.  The optimum is deliberately simple so this
    // test isolates model assembly, full affine-objective retention, low-rank
    // condensation, and the coupled LazyQP step rather than benchmark data.
    QCQPProblem p;
    p.name="affine_full_shape_130_1_384";
    p.n=n;
    p.x0.assign(n,0.0);
    p.lb.assign(n,-inf);
    p.ub.assign(n,inf);
    p.objective.name="f";
    p.objective.linear.assign(n,0.0);
    p.objective.linear[0]=-1.0;
    p.objective.hessian.n=n;

    QuadraticFunction equality;
    equality.name="fixed_tail";
    equality.linear.assign(n,0.0);
    equality.linear[n-1]=1.0;
    equality.hessian.n=n;
    p.equalities.push_back(equality);

    QuadraticFunction bounding_ball;
    bounding_ball.name="x0_ball";
    bounding_ball.constant=-1.0;
    bounding_ball.linear.assign(n,0.0);
    bounding_ball.hessian.n=n;
    bounding_ball.hessian.entries={{0,0,2.0}}; // x_0^2 <= 1
    p.inequalities.push_back(bounding_ball);

    for(int i=1;i<mi;++i) {
      QuadraticFunction g;
      g.name="g"+std::to_string(i);
      g.constant=-4.0;
      g.linear.assign(n,0.0);
      g.hessian.n=n;
      const int j=1+(i%128);
      const int k=1+((i+37)%128);
      if(i%2==0) {
        g.hessian.entries={{j,j,2.0}};
      } else if(j<k) {
        g.hessian.entries={{j,j,2.0},{k,k,-2.0}};
      } else {
        g.hessian.entries={{k,k,-2.0},{j,j,2.0}};
      }
      p.inequalities.push_back(std::move(g));
    }
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
    st.inner_tolerance=1e-7;
    st.inner_iteration_limit_mu_fallback=1e-6;
    st.dca_full_constraints_for_affine_objective=true;
    st.dca_affine_full_constraint_limit=4096;
    st.exact_eigen_threshold=256;
    st.max_rank_per_quadratic=24;
    st.lazyqp_kkt_form=(argc>1)?argv[1]:"auto";

    const Result r=Solver(st).solve(p);
    if(r.status!=Status::Solved)
      throw std::runtime_error("130/1/384 full-majorant surrogate failed: "+r.message);
    if(r.outer_iterations<1)
      throw std::runtime_error("130/1/384 surrogate never entered outer optimization");
    if(r.x.size()!=n || std::abs(r.x[0]-1.0)>1e-4)
      throw std::runtime_error("130/1/384 surrogate returned wrong minimizer");
    if(r.feasibility.total_inf>st.feasibility_tolerance)
      throw std::runtime_error("130/1/384 surrogate is infeasible");
    if(r.inequality_total_rank<100)
      throw std::runtime_error("negative-curvature rows were not retained/condensed");
    if(r.working_set_additions!=0)
      throw std::runtime_error("full-majorant surrogate unexpectedly used lazy activation");
    if(r.lazy_dca_corrector_attempts!=0)
      throw std::runtime_error("large affine-equality path unexpectedly entered the local corrector");
    if(r.lazy_dca_step_control_retries!=0)
      throw std::runtime_error("affine-equality A4X12-like path unexpectedly entered DCA step control");
    if(st.lazyqp_kkt_form=="full") {
      if(r.lazyqp_full_kkt_factorizations<1 || r.lazyqp_reduced_kkt_factorizations!=0 ||
         r.lazyqp_max_kkt_dimension!=515 || r.lazyqp_max_full_kkt_dimension!=515)
        throw std::runtime_error("130/1/384 surrogate did not use the requested full KKT path");
    } else {
      if(r.lazyqp_reduced_kkt_factorizations<1 || r.lazyqp_full_kkt_factorizations!=0)
        throw std::runtime_error("130/1/384 surrogate did not use the reduced KKT path");
      if(r.lazyqp_max_kkt_dimension!=131 || r.lazyqp_max_full_kkt_dimension!=515)
        throw std::runtime_error("130/1/384 surrogate reported wrong reduced/full dimensions");
    }

    std::cout << "[PASS] affine full-majorant shape 130/1/384"
              << " outer="<<r.outer_iterations
              << " inner="<<r.inner_iterations
              << " rank="<<r.inequality_total_rank
              << " kkt_dim="<<r.lazyqp_max_kkt_dimension
              << "/"<<r.lazyqp_max_full_kkt_dimension
              << " x0="<<r.x[0]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
