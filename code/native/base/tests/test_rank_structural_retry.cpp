#include "../src/rank_structural_retry.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void budgets() {
  Settings s;s.max_outer_iterations=3000;s.restoration_max_iterations=600;s.time_limit_seconds=1200;
  Result primary;primary.status=Status::InnerFailure;primary.outer_iterations=51;
  primary.restoration_iterations=400;primary.restoration_total_seconds=2;
  auto b=detail::rank_structural_retry_budget(s,primary,75,1000);
  check(b.allowed&&b.outer==2949&&b.restoration==200&&b.seconds==1125,
        "retry reopened time, outer or Phase-I allowance");
  check(b.restoration_deadline==1060,"retry opened a new private sixty-second window");
  for(Status status:{Status::Solved,Status::SolvedAcceptable,Status::SolvedAcceptableOnFailure,
                    Status::TimeLimit,Status::MaxOuterIterations}) {
    primary.status=status;
    check(!detail::rank_structural_retry_budget(s,primary,75,1000).allowed,"healthy/exhausted solve triggered structural retry");
  }
  for(Status status:{Status::FactorizationFailure,Status::InnerFailure,Status::NumericalError,
                    Status::LineSearchFailure,Status::RestorationFailure,Status::TrustRegionFailure}) {
    primary.status=status;
    check(detail::rank_structural_retry_budget(s,primary,75,1000).allowed,"generic numerical failure was excluded");
  }
  primary.outer_iterations=3000;
  check(!detail::rank_structural_retry_budget(s,primary,75,1000).allowed,"outer budget was exceeded");
  primary.outer_iterations=51;
  check(!detail::rank_structural_retry_budget(s,primary,1200,1000).allowed,"wall budget was exceeded");
  primary.restoration_iterations=900;b=detail::rank_structural_retry_budget(s,primary,75,1000);
  check(b.allowed&&b.restoration==0,"exhausted Phase I acquired new iterations");
  s.restoration_deadline_seconds=1040;
  check(detail::rank_structural_retry_budget(s,primary,75,1000).restoration_deadline==1040,"explicit earlier private deadline was lost");
}
QCQPProblem problem(bool fixed) {
  QCQPProblem p;p.n=2;p.name="anonymous_structural_retry_control";p.x0={0.5,0.25};
  p.lb={fixed?0.0:-10.0,-10};p.ub={10,10};p.objective.hessian.n=2;
  p.objective.linear={-1,-2};p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction q;q.hessian.n=2;q.linear={1,0};q.constant=fixed?0:-3;p.inequalities.push_back(q);
  return p;
}
void defaults_and_healthy_route() {
  auto fixed=problem(true);ConstraintPresolve historical(fixed),proof(fixed,true,false,true);
  check(historical.affine_bounds().empty()&&historical.reduced().ub==fixed.ub&&
        historical.reduced().inequalities.size()==fixed.inequalities.size(),"exact-fixed proof activated eagerly by default");
  check(proof.affine_bounds().size()==1&&proof.reduced().lb[0]==proof.reduced().ub[0],"deferred exact-fixed proof unavailable");
  check(proof.reduced().x0==fixed.x0,"retry changed the frozen original start");
  if(!linear_backend_compiled("mumps"))return;
  auto p=problem(false);Settings st;st.linear_backend="mumps";st.dca_engine="majorants";
  st.outer_algorithm="lazy-dca";st.max_outer_iterations=300;st.time_limit_seconds=5;
  st.verbose=false;st.live_display_level=0;st.lazy_newton_reduction=false;
  st.lazy_newton_atomic_regularization=false;st.lazy_newton_lazy_constraints=false;
  for(const std::string configuration:{"off","HH","HS","SH"}) {
    st.convexification_study=configuration;
    Settings direct=st;direct.dca_rank_presolve=false;
    auto original=Solver(direct).solve(p);auto wrapped=solve_with_rank_presolve(p,st);
    check(original.status==Status::Solved&&wrapped.status==Status::Solved,"healthy native control did not solve");
    check(original.x==wrapped.x&&original.objective==wrapped.objective&&
          original.outer_iterations==wrapped.outer_iterations&&
          original.certificate_equality_multipliers==wrapped.certificate_equality_multipliers&&
          original.certificate_inequality_multipliers==wrapped.certificate_inequality_multipliers&&
          original.certificate_lower_multipliers==wrapped.certificate_lower_multipliers&&
          original.certificate_upper_multipliers==wrapped.certificate_upper_multipliers,
          "strict primary numerical result changed in structural wrapper");
  }
}
void zero_restoration_budget_native() {
  if(!linear_backend_compiled("mumps"))return;
  auto p=problem(true);ConstraintPresolve proof(p,true,false,true);
  Settings st;st.linear_backend="mumps";st.dca_engine="majorants";st.outer_algorithm="lazy-dca";
  st.majorant_startup_policy="auto";st.max_outer_iterations=300;st.time_limit_seconds=5;
  st.verbose=false;st.live_display_level=0;st.lazy_newton_reduction=false;
  st.lazy_newton_atomic_regularization=false;st.lazy_newton_lazy_constraints=false;
  st.restoration_enabled=true;st.restoration_max_iterations=0;
  st.restoration_shared_iteration_budget=true;st.dca_rank_presolve=false;
  st.rank_presolve_context=&proof;
  for(const std::string configuration:{"off","HH","HS","SH"}) {
    st.convexification_study=configuration;
    // This is the real secondary settings/model after an exhausted primary
    // Phase I: automatic startup keeps its permission, while the shared
    // allowance is exactly zero. It must return an endpoint, never throw the
    // automatic-startup/--no-restoration configuration error.
    auto result=Solver(st).solve(proof.reduced());
    check(result.x.size()==std::size_t(p.n),"zero-budget recovery did not return a native endpoint");
    check(result.restoration_iterations==0,"zero-budget recovery performed new Phase-I work");
    check(result.status==Status::Solved,"analytic zero-budget structural recovery did not solve");
    proof.lift(result);
    check(result.certificate_inequality_multipliers.size()==p.inequalities.size(),
          "zero-budget recovery endpoint lacks original inequality multipliers");
  }
}
}
int main(){try{
  budgets();defaults_and_healthy_route();zero_restoration_budget_native();
  std::cout<<"structural retry: deferred exact proofs, shared budgets and unchanged strict primary controls PASS\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
