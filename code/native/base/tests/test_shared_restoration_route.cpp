#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}

// The original nonlinear equality y^2+1=0 is infeasible. Its affine subsystem
// is far from the supplied start, so this exercises the real affine startup,
// original-model continuation and terminal recovery with tiny work allowances.
QCQPProblem problem() {
  QCQPProblem p;p.name="generic_nested_restoration_contract";p.n=2;
  p.x0={256.,0.};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.linear={1.,0.};p.objective.hessian.n=2;
  QuadraticFunction affine;affine.linear={1.,0.};affine.constant=-1.;affine.hessian.n=2;
  p.equalities.push_back(affine);
  QuadraticFunction curved;curved.linear={0.,0.};curved.constant=1.;curved.hessian.n=2;
  curved.hessian.entries={{1,1,2.}};p.equalities.push_back(curved);
  QuadraticFunction inequality;inequality.linear={1.,0.};inequality.constant=-512.;inequality.hessian.n=2;
  p.inequalities.push_back(inequality);p.validate();return p;
}

std::string frozen_coefficients(const QCQPProblem& p) {
  std::ostringstream out;out<<std::hexfloat<<p.n;
  for(const auto* values:{&p.x0,&p.lb,&p.ub})for(double value:*values)out<<' '<<value;
  auto function=[&](const QuadraticFunction& q) {
    out<<'|'<<q.constant;
    q.for_each_linear_nonzero([&](int index,double value){out<<' '<<index<<':'<<value;});
    for(auto h:q.hessian.entries)out<<' '<<h.i<<','<<h.j<<':'<<h.v;
  };
  function(p.objective);for(const auto& q:p.equalities)function(q);for(const auto& q:p.inequalities)function(q);
  return out.str();
}

struct Capture {
  std::ostringstream stream;std::streambuf* original=std::cout.rdbuf(stream.rdbuf());
  ~Capture(){std::cout.rdbuf(original);}
};
}

int main(){try {
  require(linear_backend_compiled("mumps"),"integration requires a real native MUMPS build");
  QCQPProblem p=problem();const std::string frozen=frozen_coefficients(p);
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.linear_backend="mumps";st.linear_metis=true;st.native_al_fallback="off";
  st.standard_r10_policy="off";st.standard_auto_resolved=true;
  st.dca_rank_presolve=false;st.free_epigraph_presolve=false;
  st.majorant_startup_policy="auto";st.majorant_adaptive_globalization=true;
  st.restoration_shared_iteration_budget=true;st.restoration_max_restarts=0;
  st.restoration_minimax_enabled=false;st.max_outer_iterations=12;st.time_limit_seconds=5;
  st.live_display_level=1;st.live_iterations=true;
  int affine_phases=0,terminal_budget_skips=0;
  for(int budget:{0,1,2,3}) {
    st.restoration_max_iterations=budget;
    Result result;std::string trace;
    {Capture capture;result=Solver(st).solve(p);trace=capture.stream.str();}
    require(result.restoration_iterations<=budget,"nested native solve exceeded shared restoration allowance");
    require(result.outer_iterations<=st.max_outer_iterations,"native solve exceeded shared outer allowance");
    require(frozen_coefficients(p)==frozen,"original coefficients, bounds or x0 changed");
    require(result.status!=Status::Solved,"infeasible original equality acquired a false KKT success");
    require(!result.native_al_fallback.attempted&&!result.native_al_fallback.accepted,
        "budget recovery introduced an AL optimizer");
    if(trace.find("[AFFINE-START-RESTORATION]")!=std::string::npos)++affine_phases;
    if(trace.find("[RESTORATION-BUDGET] phase=bounded-origin action=skip")!=std::string::npos)
      ++terminal_budget_skips;
    if(budget==0)require(trace.find("[AFFINE-START-RESTORATION]")==std::string::npos,
        "zero restoration allowance launched affine startup");
    std::cout<<"budget="<<budget<<" restoration="<<result.restoration_iterations
        <<" outer="<<result.outer_iterations<<" status="<<status_string(result.status)
        <<" seconds="<<result.elapsed_seconds<<'\n';
  }
  require(affine_phases>0,"integration did not reach affine startup");
  require(terminal_budget_skips>0,"integration did not reach the exhausted terminal-certificate guard");
  std::cout<<"PASS real Lazy/MUMPS shared restoration routes; affine_phases="<<affine_phases
      <<" terminal_budget_skips="<<terminal_budget_skips<<" original_input_preserved=1 AL_off=1\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
