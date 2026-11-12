// Integration regression through the PUBLIC solver API only. Link with either
// the archived hybrid library (must fail) or the corrected library (must pass).
// No private Workspace, candidate path, polynomial helper, or engine source
// is included, so the tested code is exactly the linked binary's controller.
#include "lazyipdca_qcqp/dca_filter.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {
void check(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance,
           const std::string& message) {
  check(std::isfinite(actual) && std::abs(actual-expected) <= tolerance,
        message+": actual="+std::to_string(actual)+
        " expected="+std::to_string(expected));
}
Settings settings() {
  Settings s;
  s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
  s.linear_backend="native";s.dca_filter_curvature_backend="native";
  s.dca_aggregate=false;s.dca_aggregate_only=false;s.dca_krylov=false;
  s.dca_filter_atom_events=false;s.dca_filter_max_atoms=0;
  s.lazy_newton_atomic_regularization=false;
  s.lazy_newton_lazy_constraints=false;s.lazy_newton_reduction=false;
  s.verbose=false;s.live_display_level=0;s.termination_test="ipopt";
  s.max_outer_iterations=100;s.dca_filter_inner_max=100;
  s.time_limit_seconds=30;s.acceptable_iterations=0;s.acceptable_on_failure=false;
  s.stationarity_tolerance=1e-7;s.feasibility_tolerance=1e-7;
  s.dual_infeasibility_tolerance=1e-7;s.complementarity_tolerance=1e-7;
  return s;
}
QCQPProblem scalar_problem() {
  QCQPProblem p;p.name="r4_auxiliary_optimum_is_not_original_optimum";p.n=1;
  p.x0={2};p.lb={-INFINITY};p.ub={INFINITY};
  p.objective.linear={0};p.objective.hessian.n=1;
  p.objective.hessian.entries={{0,0,2}}; // x^2
  QuadraticFunction g;g.constant=1;g.linear={0};g.hessian.n=1;
  g.hessian.entries={{0,0,-2}}; // 1-x^2 <= 0
  p.inequalities.push_back(g);p.validate();return p;
}
// Observe the point passed to the actual engine filter via its accepted-row
// diagnostics, then evaluate the EXPORTED endpoint independently. Checking
// only s=s0+alpha*ds would miss a filter using curved slacks while committing
// affine ones: that defective combination still satisfies the endpoint test
// above. This test never calls the candidate-path evaluator or its writers.
void engine_filter_acceptance_matches_committed_original_slacks() {
  auto p=scalar_problem();auto st=settings();st.max_outer_iterations=1;
  st.lazy_newton_initial_mu=0.1;
  st.lazy_newton_objective_gradient_target=100;
  st.lazy_newton_constraint_gradient_target=100;
  st.live_display_level=2;
  struct Capture {
    std::ostringstream output;
    std::streambuf* buffer;
    std::ios::fmtflags flags;
    std::streamsize precision;
    Capture():buffer(std::cout.rdbuf(output.rdbuf())),flags(std::cout.flags()),
              precision(std::cout.precision()) {
      std::cout.precision(std::numeric_limits<double>::max_digits10);
    }
    ~Capture() {
      std::cout.rdbuf(buffer);std::cout.flags(flags);std::cout.precision(precision);
    }
  };
  Result result;std::string transcript;
  {
    Capture capture;
    result=solve_filter_dca(p,st);
    transcript=capture.output.str();
  }
  check(result.lazy_newton.accepted_steps==1,
        "filter/commit fixture did not produce exactly one accepted engine step");
  std::string accepted_line,line;
  std::istringstream lines(transcript);
  int accepted_rows=0;
  while(std::getline(lines,line))
    if(line.rfind("[MAJORANT-FILTER]",0)==0&&line.find(" accepted=1 ")!=std::string::npos) {
      accepted_line=line;++accepted_rows;
    }
  check(accepted_rows==1,"engine trace does not identify its single accepted filter point");
  auto field=[&](const char* name) {
    const std::string key=std::string(name)+"=";
    std::istringstream fields(accepted_line);std::string token;
    while(fields>>token)if(token.rfind(key,0)==0)return std::stod(token.substr(key.size()));
    throw std::runtime_error(std::string("accepted filter trace omitted ")+name);
  };
  const double x=result.x.at(0),slack=result.termination_inequality_slacks.at(0);
  const double z=result.certificate_inequality_multipliers.at(0);
  const double delta=x-2.0;
  close(result.termination_objective_scale,1,0,"filter/commit objective fixture was rescaled");
  close(result.termination_inequality_scales.at(0),1,0,"filter/commit inequality fixture was rescaled");
  check(slack>0&&z>0&&delta*delta>1e-3,
        "filter/commit fixture has no positive state or measurable nonlinear gap");
  close(slack,3.0+4.0*delta,1e-12,
        "engine committed a curved or transported slack instead of its affine Newton slack");
  const double mu=field("mu");
  close(mu,0.1,1e-14,"filter/commit fixture changed mu during its first line search");
  const long double endpoint_theta=std::abs(1.0L-static_cast<long double>(x)*x+slack);
  const long double endpoint_phi_change=(static_cast<long double>(x)-2)*(x+2)-
      static_cast<long double>(mu)*std::log(static_cast<long double>(slack)/3);
  check(endpoint_theta>1e-3,
        "filter/commit fixture cannot distinguish original and majorant feasibility");
  close(field("theta"),double(endpoint_theta),1e-12,
        "filter accepted a different original g+s residual from the committed state");
  close(field("phi_change"),double(endpoint_phi_change),1e-12,
        "filter accepted a different logarithmic slack barrier from the committed state");
  // Independent first-step switching/Armijo check on that committed state.
  // Since theta0=0 and mu0=.1, alpha*phi'(0)=(4-.1*4/3)*(x-x0).
  const long double armijo_bound=static_cast<long double>(st.line_search_armijo)*
      (4.0L-0.1L*4/3)*delta;
  check(endpoint_phi_change<=armijo_bound,
        "committed point fails the original barrier acceptance used at its feasible anchor");
  check(!result.lazy_newton.predictor_corrector_enabled&&result.dca_filter.majorant_slack_transports==0,
        "fixture enabled a predictor-corrector or post-filter slack transport");
}

}
int main() {
  try {
    engine_filter_acceptance_matches_committed_original_slacks();
    std::cout<<"PASS engine filter/commit: independent original feasibility and slack barrier match the exported accepted state\n";
  } catch(const std::exception& e) {
    std::cerr<<"FAIL engine filter/commit: "<<e.what()<<'\n';return 1;
  }
}
