#include "lazyipdca_qcqp/solver.hpp"

#include <iostream>
#include <stdexcept>
#include <utility>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_lookahead_problem() {
  QCQPProblem p;
  p.name="lazy_dca_working_set_retention";
  p.n=2;
  p.x0={0.0,0.0};
  p.lb={-2.0,-10.0};
  p.ub={ 2.0, 10.0};

  // On the manifold y=x^2, f=-x+10*x^4.  At the origin the tangent model
  // does not yet see the quartic term, so its full trial crosses x<=1.  The
  // exact condensed retraction exposes that term and Armijo accepts a shorter
  // step, at which x<=1 is again materially slack.
  p.objective.name="quartic_on_manifold";
  p.objective.linear={-1.0,0.0};
  p.objective.hessian.n=2;
  p.objective.hessian.entries={{1,1,20.0}};

  QuadraticFunction h;
  h.name="parabola";
  h.linear={0.0,1.0};
  h.hessian.n=2;
  h.hessian.entries={{0,0,-2.0}}; // y-x^2=0
  p.equalities.push_back(std::move(h));

  QuadraticFunction g;
  g.name="lookahead_limit";
  g.constant=-1.0;
  g.linear={1.0,0.0};             // x-1<=0
  g.hessian.n=2;
  p.inequalities.push_back(std::move(g));
  p.validate();
  return p;
}

Settings settings() {
  Settings st;
  st.outer_algorithm="lazy-dca";
  st.inner_backend="lazyqp";
  st.linear_backend="native";
  st.dca_equality_model="condensed-retraction";
  st.verbose=false;
  st.restoration_enabled=false;
  st.dca_step_control=false;
  st.dca_local_corrector=false;
  st.dca_iterated_soc=false;
  st.second_order_correction=false;
  st.terminal_kkt_enabled=false;
  // Stop immediately after the iteration in which an old implementation
  // dropped and re-added the same look-ahead constraint.
  st.max_outer_iterations=2;
  st.max_inner_iterations=500;
  st.feasibility_tolerance=1e-7;
  st.stationarity_tolerance=1e-8;
  st.dual_infeasibility_tolerance=1e-8;
  st.complementarity_tolerance=1e-8;
  st.inner_tolerance=1e-10;
  st.working_set_activation=1e-4;
  st.working_set_drop_slack=2e-4;
  st.working_set_drop_multiplier=1e-9;
  return st;
}

} // namespace

int main() {
  try {
    const Result r=Solver(settings()).solve(make_lookahead_problem());
    if(r.status==Status::InnerFailure || r.status==Status::LineSearchFailure ||
       r.status==Status::NumericalError)
      throw std::runtime_error("look-ahead trajectory failed: "+r.message);
    if(r.working_set_primal_transports<1)
      throw std::runtime_error("test did not exercise full-trial working-set expansion");
    if(r.working_set_additions!=1)
      throw std::runtime_error("look-ahead constraint was re-added after a premature drop: additions="+
                               std::to_string(r.working_set_additions));
    if(r.working_set_drops!=0)
      throw std::runtime_error("fresh look-ahead constraint was dropped at the next accepted point: drops="+
                               std::to_string(r.working_set_drops));
    if(r.feasibility.total_inf>settings().feasibility_tolerance)
      throw std::runtime_error("retained working set damaged original-QCQP feasibility");

    std::cout << "[PASS] look-ahead constraint retained across shortened step"
              << " additions="<<r.working_set_additions
              << " drops="<<r.working_set_drops
              << " transports="<<r.working_set_primal_transports
              << " outer="<<r.outer_iterations
              << " x="<<r.x[0]<<","<<r.x[1]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
