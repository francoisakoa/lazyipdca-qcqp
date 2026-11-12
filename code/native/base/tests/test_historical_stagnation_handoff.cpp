#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "../src/majorant_barrier_retry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}

// This is a native routing witness, not a convergence benchmark. The free y
// direction makes stationarity impossible to repair by fitting multipliers.
// Thus a genuine watcher exit must reach its caller without another cold solve.
QCQPProblem problem(bool bounded_y) {
  QCQPProblem p;p.name=bounded_y?"watch_fit_success":"watch_handoff_without_KKT";
  p.n=2;p.x0={1.,1.};p.lb={-2.,bounded_y?0.:-INFINITY};
  p.ub={2.,bounded_y?4.:INFINITY};
  p.objective.linear={0.,1.};p.objective.hessian.n=2;
  QuadraticFunction h;h.hessian.n=2;h.linear={0.,0.};
  h.constant=-1.;h.hessian.entries={{0,0,2.}};p.equalities={h};
  p.validate();return p;
}

Settings settings() {
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.linear_backend="mumps";st.linear_metis=true;st.native_al_fallback="off";
  st.standard_r10_policy="off";st.standard_auto_resolved=true;
  st.standard_equality_watch=true;st.dca_rank_presolve=false;st.free_epigraph_presolve=false;
  st.majorant_startup_policy="legacy";st.restoration_enabled=false;
  st.restoration_shared_iteration_budget=true;st.restoration_max_iterations=0;
  st.max_outer_iterations=64;st.time_limit_seconds=5;
  // A large but supported proximal coefficient creates eight ACTUAL accepted
  // rounded tiny steps. No observer hook or simulated solver result is used.
  st.lazy_newton_regularization_initial=1e12;
  st.live_display_level=2;st.live_iterations=true;
  st.termination_test="project";st.acceptable_iterations=0;st.acceptable_on_failure=false;
  st.feasibility_tolerance=st.stationarity_tolerance=
      st.dual_infeasibility_tolerance=st.complementarity_tolerance=1e-6;
  return st;
}

struct Capture {
  std::ostringstream stream;std::streambuf* old=std::cout.rdbuf(stream.rdbuf());
  ~Capture(){std::cout.rdbuf(old);}
};

Result solve(const QCQPProblem& p,const Settings& st,std::string& trace) {
  Capture captured;Result r=Solver(st).solve(p);trace=captured.stream.str();return r;
}

void same_problem(const QCQPProblem& p,const QCQPProblem& before) {
  check(p.n==before.n&&p.x0==before.x0&&p.lb==before.lb&&p.ub==before.ub&&
      p.objective.linear==before.objective.linear&&p.objective.constant==before.objective.constant&&
      p.equalities.size()==1&&p.equalities[0].constant==before.equalities[0].constant&&
      p.equalities[0].linear==before.equalities[0].linear&&
      p.equalities[0].hessian.entries.size()==1&&p.equalities[0].hessian.entries[0].i==0&&
      p.equalities[0].hessian.entries[0].j==0&&p.equalities[0].hessian.entries[0].v==2.,
      "native routing changed original coefficients, bounds or start");
}

void verify_bounded_KKT(const Result& r) {
  check(r.x.size()==2&&r.certificate_equality_multipliers.size()==1&&
      r.certificate_lower_multipliers.size()==2&&r.certificate_upper_multipliers.size()==2,
      "protected fit did not export a full original certificate");
  const auto& x=r.x;const auto& lo=r.certificate_lower_multipliers;
  const auto& up=r.certificate_upper_multipliers;const double y=r.certificate_equality_multipliers[0];
  const double primal=std::max({std::abs(x[0]*x[0]-1),-2-x[0],x[0]-2,-x[1],x[1]-4,0.});
  const double dual=std::max({std::abs(2*x[0]*y-lo[0]+up[0]),
      std::abs(1-lo[1]+up[1]),-lo[0],-lo[1],-up[0],-up[1],0.});
  const double comp=std::max({std::abs(lo[0]*(x[0]+2)),std::abs(up[0]*(2-x[0])),
      std::abs(lo[1]*x[1]),std::abs(up[1]*(4-x[1]))});
  check(std::isfinite(primal)&&std::isfinite(dual)&&std::isfinite(comp)&&
      primal<=1e-6&&dual<=1e-6&&comp<=1e-6,
      "protected fit failed direct original two-variable KKT equations");
}
}

int main(){try {
  static_assert(static_cast<int>(Status::Solved)==0&&static_cast<int>(Status::InnerFailure)==4&&
      static_cast<int>(Status::LineSearchFailure)==6&&
      static_cast<int>(Status::SolvedAcceptableOnFailure)==11&&
      static_cast<int>(Status::HistoricalStagnation)==12,"existing status codes changed");
  check(std::string(status_string(Status::HistoricalStagnation))=="historical_stagnation",
      "structured handoff status has no stable exported name");
  check(linear_backend_compiled("mumps"),"routing integration requires native MUMPS");
  Settings st=settings();QCQPProblem p=problem(false);const auto before=p;
  std::string trace;Result handoff=solve(p,st,trace);
  check(trace.find("[HISTORICAL-EQUALITY-WATCH] event=stagnation")!=std::string::npos,
      "native integration did not trigger the real accepted-state watcher");
  check(handoff.status==Status::HistoricalStagnation,
      "watcher handoff was converted to failure or another terminal status");
  check(!handoff.native_barrier_retry.attempted&&
      trace.find("[NATIVE-BARRIER-RETRY] event=begin")==std::string::npos,
      "historical handoff acquired a parasitic cold retry");
  check(handoff.termination_audit_valid&&handoff.feasibility.total_inf<=1e-6&&
      handoff.dual_infeasibility_inf>=.99,
      "handoff lost its original audit or falsely certified the free y direction");
  check(handoff.outer_iterations>=8&&handoff.outer_iterations<16&&
      handoff.restoration_iterations==0&&handoff.elapsed_seconds<st.time_limit_seconds,
      "handoff consumed another solve allowance before returning");
  check(!handoff.native_al_fallback.attempted&&!handoff.native_al_fallback.accepted,
      "handoff changed the AL-off route");
  same_problem(p,before);

  // The admission functions are the actual policy called by Solver::solve.
  // Their decision depends on the status, never the wording of a message.
  Result primary=handoff;primary.message="ordinary numerical failure";
  auto budget=detail::majorant_barrier_retry_budget(p,st,primary,.1,1000);
  check(!budget.allowed&&!detail::rank_structural_retry_status(primary.status),
      "structured handoff admitted a cold structural or barrier retry");
  check(budget.outer==st.max_outer_iterations-primary.outer_iterations&&
      budget.restoration==0&&std::abs(budget.seconds-4.9)<1e-12,
      "handoff changed remaining work/time arithmetic");
  for(Status status:{Status::InnerFailure,Status::LineSearchFailure,Status::FactorizationFailure}) {
    primary.status=status;primary.message="historical short-step stagnation; arbitrary text";
    const auto actual=detail::majorant_barrier_retry_budget(p,st,primary,.1,1000);
    check(actual.allowed&&detail::rank_structural_retry_status(status)&&
        actual.outer==budget.outer&&actual.restoration==budget.restoration&&
        actual.seconds==budget.seconds,"genuine numerical failure lost its existing retry or budgets");
  }

  // A valid fixed-primal ORIGINAL certificate remains a success even if the
  // watcher fired first. The correction must not force a needless GR phase.
  QCQPProblem bounded=problem(true);const auto bounded_before=bounded;
  const Result fitted=solve(bounded,st,trace);
  check(trace.find("[HISTORICAL-EQUALITY-WATCH] event=stagnation")!=std::string::npos&&
      fitted.status==Status::Solved&&fitted.termination_audit_valid&&
      fitted.termination_certificate_rescued,
      "valid original terminal multiplier-fit success was sacrificed to force handoff");
  verify_bounded_KKT(fitted);same_problem(bounded,bounded_before);

  // Exercise the complete historical -> adaptive GR return path as well as
  // the standalone historical caller above. Unused budgets must be shared.
  Settings automatic=st;automatic.standard_auto_resolved=false;
  automatic.standard_r10_policy="auto";
  const Result continued=solve(p,automatic,trace);const auto& a=continued.adaptive_equality_probe;
  check(a.attempted&&a.transitioned&&a.probe_status=="historical_stagnation",
      "adaptive controller did not receive the structured watcher handoff");
  // The same-execution fixed-primal finish attempted one native phase before
  // GR. Its rejected certificate must debit exactly one outer, zero restoration.
  const auto finish_begin=trace.find("[NATIVE-ORIGINAL-ANCHOR-FINISH]");
  check(finish_begin!=std::string::npos,"adaptive native fixed-primal phase missing");
  const auto finish_end=trace.find('\n',finish_begin);
  const auto finish_trace=trace.substr(finish_begin,finish_end-finish_begin);
  check(finish_trace.find("attempted=1 accepted=0")!=std::string::npos&&
      finish_trace.find("outer_charge=1 restoration_charge=0")!=std::string::npos&&
      finish_trace.find("factor_calls=0 direct_solve_calls=0")!=std::string::npos,
      "rejected fixed-primal phase lost its exact work charge or invented a certificate");
  check(a.probe_outer==handoff.outer_iterations+1&&
      a.stabilized_outer_budget==automatic.max_outer_iterations-a.probe_outer&&
      a.within_global_budget&&continued.outer_iterations<=automatic.max_outer_iterations&&
      continued.restoration_iterations==0,
      "adaptive transfer reset or lost a work allowance");
  check(trace.find("[NATIVE-BARRIER-RETRY] event=begin")==std::string::npos&&
      !continued.native_al_fallback.attempted&&continued.status!=Status::Solved,
      "adaptive handoff introduced a cold retry, AL route or false KKT");
  same_problem(p,before);
  std::cout<<"historical_stagnation_handoff PASS real_MUMPS=1 historical_outer="
      <<handoff.outer_iterations<<" adaptive_probe_outer="<<a.probe_outer
      <<" adaptive_total_outer="<<continued.outer_iterations
      <<" native_failure_retries_preserved=1 original_fit_success_preserved=1"
      <<" frozen_input=1 budgets_shared=1 AL_off=1 old_status_codes_preserved=1\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
