// Public-API regression: test the linked controller, not a private copy.
#include "lazyipdca_qcqp/dca_filter.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const char* msg) {if(!ok)throw std::runtime_error(msg);}
Settings settings() {
  Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
  s.linear_backend="native";s.dca_filter_curvature_backend="native";
  s.dca_aggregate=false;s.dca_aggregate_only=false;s.dca_krylov=false;
  s.lazy_newton_lazy_constraints=false;s.lazy_newton_atomic_regularization=false;
  s.verbose=false;s.live_display_level=2;s.max_outer_iterations=1;
  s.time_limit_seconds=10;s.termination_test="project";
  s.feasibility_tolerance=1e-8;s.stationarity_tolerance=1e-8;
  s.dual_infeasibility_tolerance=1e-8;s.complementarity_tolerance=1e-8;
  s.acceptable_iterations=0;s.acceptable_on_failure=false;
  return s;
}
QCQPProblem problem() {
  QCQPProblem p;p.name="quadratic_objective_nonlinear_equality_frozen_start";
  p.n=2;p.x0={3,1};p.lb={INFINITY,INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.constant=9;p.objective.linear={-6,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,2}}; // (x-3)^2
  QuadraticFunction h;h.constant=-1;h.linear={0,0};h.hessian.n=2;
  h.hessian.entries={{0,1,1}}; // x*y-1
  p.equalities.push_back(h);p.validate();return p;
}
struct Capture {
  std::ostringstream out;std::streambuf* previous;
  Capture():previous(std::cout.rdbuf(out.rdbuf())){}
  ~Capture(){std::cout.rdbuf(previous);}
};
void curved_objective_keeps_the_original_first_step() {
  const auto p=problem();auto st=settings();Result r;std::string log;
  {Capture c;r=solve_filter_dca(p,st);log=c.out.str();}
  check(r.restoration_iterations==0,"curved objective still enters startup restoration");
  check(r.status==Status::Solved,"one exact original step was not sufficient");
  check(std::abs(r.x[0]-3)<=1e-12&&std::abs(r.x[1]-1.0/3)<=1e-12,
        "first step did not keep the objective minimizer");
  check(log.find("objective_free_curvature=1 equality_zero_hessian_restore=0")!=std::string::npos,
        "structural startup decision not visible");
  check(r.lazy_newton.accepted_steps==1,"controller used an extra hidden solve phase");
  // Turning off ALL restoration must be an exact no-op for this curved case.
  st.restoration_enabled=false;Result diagnostic;
  {Capture c;diagnostic=solve_filter_dca(p,st);}
  check(diagnostic.x==r.x,"default curved route differs from direct original-model route");
}
void affine_startup_is_preserved() {
  auto p=problem();p.objective.constant=0;p.objective.linear={1,0};
  p.objective.hessian.entries.clear();Result r;std::string log;
  {Capture c;r=solve_filter_dca(p,settings());log=c.out.str();}
  check(r.restoration_iterations>0,"affine equality startup was disabled");
  check(log.find("objective_free_curvature=0 equality_zero_hessian_restore=1")!=std::string::npos,
        "affine startup guard incorrect");
}
void constant_negative_curvature_startup_is_preserved() {
  QCQPProblem p;p.name="constant_equalities_start_at_zero_jacobian";p.n=30;
  p.x0.assign(p.n,0);p.lb.assign(p.n,INFINITY);p.ub.assign(p.n,INFINITY);
  p.objective.compact_linear_dimension=p.n;p.objective.hessian.n=p.n;
  for(int i=0;i<10;++i){
    QuadraticFunction h;h.constant=-1;h.compact_linear_dimension=p.n;h.hessian.n=p.n;
    for(int j=0;j<3;++j)h.hessian.entries.push_back({3*i+j,3*i+j,2});
    p.equalities.push_back(h);
  }
  p.validate();
  auto st=settings();st.max_outer_iterations=50;Result r;std::string log;
  {Capture c;r=solve_filter_dca(p,st);log=c.out.str();}
  check(log.find("objective_free_curvature=0 equality_zero_hessian_restore=1")!=std::string::npos,
        "constant objective no longer enters the CYCLOOCF startup branch");
  check(r.status==Status::Solved,"constant feasibility negative-curvature regression");
  for(int i=0;i<10;++i)check(std::abs(r.x[3*i]*r.x[3*i]+r.x[3*i+1]*r.x[3*i+1]+r.x[3*i+2]*r.x[3*i+2]-1)<=1e-8,"constant feasibility certificate failed");
}
void fixed_curvature_does_not_disable_affine_free_startup() {
  auto p=problem();p.n=3;p.x0={3,1,2};p.lb={INFINITY,INFINITY,2};p.ub={INFINITY,INFINITY,2};
  p.objective.constant=0;p.objective.linear={1,0,0};p.objective.hessian.n=3;
  p.objective.hessian.entries={{2,2,2}};
  p.equalities[0].linear={0,0,0};p.equalities[0].hessian.n=3;p.validate();
  Result r;std::string log;{Capture c;r=solve_filter_dca(p,settings());log=c.out.str();}
  check(r.restoration_iterations>0,"fixed-only curvature disabled affine free startup");
  check(log.find("objective_free_curvature=0 equality_zero_hessian_restore=1")!=std::string::npos,
        "startup policy must inspect the free-space objective Hessian");
}
}
int main(){try{
 curved_objective_keeps_the_original_first_step();affine_startup_is_preserved();
 constant_negative_curvature_startup_is_preserved();fixed_curvature_does_not_disable_affine_free_startup();
 std::cout<<"PASS majorant startup: curved, affine, constant-curvature escape, fixed variables\n";
}catch(const std::exception& e){std::cerr<<"FAIL majorant startup: "<<e.what()<<'\n';return 1;}}
