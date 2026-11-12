#include "lazyipdca_qcqp/solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace lazyipdca::qcqp;

namespace {

QCQPProblem make_problem() {
  const double inf=std::numeric_limits<double>::infinity();
  QCQPProblem p;
  p.name="affine_box_bounded_condensed_lazy";
  p.n=3;
  p.x0={0.0,1.0,0.0};
  p.lb={-2.0,-inf,-inf};
  p.ub={ 2.0, inf, inf};
  p.objective.name="minus_x";
  p.objective.linear={-1.0,0.0,0.0};
  p.objective.hessian.n=3;

  QuadraticFunction circle;
  circle.name="unit_yz_circle";
  circle.constant=-1.0;
  circle.linear.assign(3,0.0);
  circle.hessian.n=3;
  circle.hessian.entries={{1,1,2.0},{2,2,2.0}};
  p.equalities.push_back(std::move(circle));

  QuadraticFunction ball;
  ball.name="x_ball";
  ball.constant=-1.0;
  ball.linear.assign(3,0.0);
  ball.hessian.n=3;
  ball.hessian.entries={{0,0,2.0}};
  p.inequalities.push_back(std::move(ball));

  // PGLib-like zero-cost rows that are safely slack at the start must not be
  // factored merely because the objective is affine.
  for(int i=0;i<24;++i) {
    QuadraticFunction slack;
    slack.name="slack_"+std::to_string(i);
    slack.constant=-100.0-static_cast<double>(i);
    slack.linear.assign(3,0.0);
    slack.hessian.n=3;
    slack.hessian.entries={{2,2,2.0}};
    p.inequalities.push_back(std::move(slack));
  }
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
  st.dca_local_corrector=false;
  st.dca_iterated_soc=false;
  st.second_order_correction=false;
  st.max_outer_iterations=20;
  st.max_inner_iterations=500;
  st.feasibility_tolerance=1e-7;
  st.stationarity_tolerance=1e-6;
  st.dual_infeasibility_tolerance=1e-6;
  st.complementarity_tolerance=1e-6;
  st.inner_tolerance=1e-9;
  st.dca_affine_full_constraint_limit=4096;
  st.dca_affine_box_bounded_lazy=true;
  return st;
}

} // namespace

int main() {
  try {
    const QCQPProblem p=make_problem();
    Settings lazy=settings();
    const Result sparse=Solver(lazy).solve(p);
    if(sparse.status!=Status::Solved)
      throw std::runtime_error("box-bounded condensed lazy route failed: "+
                               sparse.message);
    if(std::abs(sparse.x[0]-1.0)>2e-5 ||
       sparse.feasibility.total_inf>lazy.feasibility_tolerance)
      throw std::runtime_error("box-bounded condensed lazy route returned wrong point");
    if(sparse.working_set_additions<1 || sparse.active_inequalities!=1)
      throw std::runtime_error("box-bounded affine cost did not use lazy activation");

    Settings historical=lazy;
    historical.dca_affine_box_bounded_lazy=false;
    const Result full=Solver(historical).solve(p);
    if(full.status!=Status::Solved)
      throw std::runtime_error("historical full affine majorant failed: "+full.message);
    if(full.working_set_additions!=0 ||
       full.active_inequalities!=static_cast<int>(p.inequalities.size()))
      throw std::runtime_error("historical override did not retain the full majorant");

    // The box-boundedness certificate concerns the affine objective, not the
    // nonlinear-equality globalization selected by the user.  In particular,
    // selecting the classical linearized equality model must not silently
    // restore the O(m_I) full-majorant path.
    Settings linearized=lazy;
    linearized.dca_equality_model="linearized";
    const Result linearized_sparse=Solver(linearized).solve(p);
    if(linearized_sparse.status!=Status::Solved)
      throw std::runtime_error("box-bounded linearized lazy route failed: "+
                               linearized_sparse.message);
    if(std::abs(linearized_sparse.x[0]-1.0)>2e-5 ||
       linearized_sparse.feasibility.total_inf>linearized.feasibility_tolerance)
      throw std::runtime_error("box-bounded linearized lazy route returned wrong point");
    if(linearized_sparse.working_set_additions<1 ||
       linearized_sparse.active_inequalities!=1)
      throw std::runtime_error("linearized equality model did not honor sparse box-bounded activation");

    Settings linearized_historical=linearized;
    linearized_historical.dca_affine_box_bounded_lazy=false;
    const Result linearized_full=Solver(linearized_historical).solve(p);
    if(linearized_full.status!=Status::Solved)
      throw std::runtime_error("historical linearized full majorant failed: "+
                               linearized_full.message);
    if(linearized_full.working_set_additions!=0 ||
       linearized_full.active_inequalities!=static_cast<int>(p.inequalities.size()))
      throw std::runtime_error("linearized historical override did not retain the full majorant");

    std::cout << "[PASS] box-bounded affine condensed route starts lazy"
              << " ws="<<sparse.active_inequalities<<"/"<<p.inequalities.size()
              << " additions="<<sparse.working_set_additions
              << " full_ws="<<full.active_inequalities
              << " lin_ws="<<linearized_sparse.active_inequalities
              << " lin_full_ws="<<linearized_full.active_inequalities
              << " x="<<sparse.x[0]<<"\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr << "[FAIL] "<<e.what()<<"\n";
    return 1;
  }
}
