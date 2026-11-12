#include <cstddef>
#include "../src/atomic_newton.cpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;
namespace {
void check(bool ok,const char* message) {
  if(!ok)throw std::runtime_error(message);
}
constexpr double first_rhs=3.1415926535897931;
constexpr double second_rhs=2.7182818284590451;
constexpr double tolerance=1e-7;

// The first equation of J' dlambda = -dual requires dlambda[0] +
// dlambda[1] = -pi. With nearly dependent equality rows, the exact two
// multipliers have magnitude about 4e11. Their binary64 spacing prevents
// that sum from attaining the original RHS-relative tolerance. This is
// representational cancellation, not a low-quality residual calculation.
QCQPProblem problem(double epsilon,double curvature) {
  QCQPProblem p;p.name="nearly_dependent_kkt_precision";p.n=2;
  p.x0={0,0};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={0,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,curvature},{1,1,curvature}};
  QuadraticFunction q;q.name="first_row";q.linear={1,1};q.hessian.n=2;
  p.equalities.push_back(q);
  q.name="second_row";q.linear={1,1+epsilon};p.equalities.push_back(q);
  p.validate();return p;
}
Settings settings(bool recovery=true) {
  Settings st;st.dca_engine="majorants";st.linear_backend="mumps";st.linear_metis=true;
  st.lazy_newton_reduction=false;st.lazy_newton_lazy_constraints=false;
  st.lazy_newton_atomic_regularization=false;
  st.verbose=false;st.live_display_level=0;st.kkt_backward_error_reject=tolerance;
  if(!recovery)st.kkt_rho_x_max=st.kkt_delta_dual_max=0;
  return st;
}
long double independent_error(const QCQPProblem& p,const Direction& d,const Vec& eq) {
  // Direct full, unshifted four-equation oracle. It does not use Workspace
  // matrices, multipliers' norms, its audit or the preconditioner's shifts.
  const long double h=p.objective.hessian.entries[0].v;
  const long double a=p.equalities[1].linear[1];
  return std::max({
    std::abs(static_cast<long double>(first_rhs)+h*d.x[0]+d.lambda[0]+d.lambda[1]),
    std::abs(static_cast<long double>(second_rhs)+h*d.x[1]+d.lambda[0]+a*d.lambda[1]),
    std::abs(static_cast<long double>(eq[0])+d.x[0]+d.x[1]),
    std::abs(static_cast<long double>(eq[1])+d.x[0]+a*d.x[1])});
}
void cancellation_fixture(bool enabled,bool frozen=false,bool exhausted_time=false) {
  auto p=problem(1e-12,1e4);auto st=settings(enabled);
  if(exhausted_time)st.time_limit_seconds=0;
  const long double gap=static_cast<long double>(p.equalities[1].linear[1])-1;
  const double exact_second=static_cast<double>((first_rhs-second_rhs)/gap);
  const double spacing=std::nextafter(exact_second,INFINITY)-exact_second;
  const long double first_row_floor=std::abs(
      static_cast<long double>(first_rhs)-std::nearbyint(first_rhs/spacing)*spacing);
  check(first_row_floor>tolerance*first_rhs,
        "fixture does not establish a binary64 exact-multiplier rounding floor");
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.update_matrix();
  check(w.factorize(0)&&w.current_factor==w.factor.get()&&w.regularization==0,
        "cancellation fixture must start with a successful unshifted native factor");
  const auto x=w.x,lambda=w.lambda,h=w.hvalue,base=w.base_values;
  const auto entries=p.objective.hessian.entries;
  const auto prior_factors=ns.factor_attempts;
  const bool expect_recovery=enabled&&!frozen&&!exhausted_time;
  Direction d;std::string failure;
  try {d=w.direction({}, {first_rhs,second_rhs}, {0,0}, {},0,true,
                     DirectionPurpose::Complete,frozen);}
  catch(const std::runtime_error& e) {failure=e.what();}
  if(!expect_recovery) {
    check(!failure.empty(),"historical/frozen/exhausted fixture accepted invalid original KKT");
    check(failure.find(frozen?"frozen-factor correction residual rejected":
        "expanded Newton residual rejected")!=std::string::npos,
        "fixture was rejected for a different reason");
    check(w.current_factor->last_backward_error()<1e-12,
          "fixture does not demonstrate a small backend error with failed expanded residual");
    check(ns.backend_regularization_attempts==0,
          "disabled/frozen/exhausted recovery attempted a shifted preconditioner");
    if(frozen)check(ns.factor_attempts==prior_factors,
          "frozen correction changed its factorization");
    std::cout<<"expected_rejection frozen="<<frozen<<" exhausted_time="<<exhausted_time
             <<" backend_berr="<<w.current_factor->last_backward_error()<<'\n';
    return;
  }
  check(failure.empty(),"bounded native recovery did not resolve the expanded cancellation");
  const long double error=independent_error(p,d,{0,0});
  check(error/first_rhs<=tolerance&&d.residual<=tolerance,
        "recovery does not satisfy the independent original equations");
  check(ns.backend_regularization_attempts>=1&&ns.backend_regularization_attempts<=5,
        "recovery did not use its finite bounded preconditioner ladder");
  check(ns.maximum_backend_primal_regularization<=1e-8*(1+1e-12)&&
        ns.maximum_backend_dual_regularization<=1e-8*(1+1e-12),
        "recovery exceeded the existing numerical-shift cap");
  check(std::max(std::abs(d.lambda[0]),std::abs(d.lambda[1]))<100,
        "recovery retained the catastrophic cancelling multiplier representation");
  check(w.x==x&&w.lambda==lambda&&w.hvalue==h&&w.base_values==base&&w.regularization==0&&
        p.objective.hessian.entries[0].v==entries[0].v&&
        st.kkt_backward_error_reject==tolerance,
        "recovery changed the iterate, original model, proximal rho or tolerance");
  auto wrong=d;wrong.lambda[0]+=1e-3;
  Vec ed(2),eh(2),ep,ec;double de,ee,pe,ce;
  check(independent_error(p,wrong,{0,0})/first_rhs>tolerance&&
        w.majorant_direction_residual_extended(wrong,{}, {first_rhs,second_rhs}, {0,0}, {},
              ed,eh,ep,ec,de,ee,pe,ce)/first_rhs>tolerance,
        "an actually invalid original direction became acceptable");
  std::cout<<"recovered original_residual="<<static_cast<double>(error/first_rhs)
           <<" exact_multiplier_rounding_floor="<<static_cast<double>(first_row_floor)
           <<" attempts="<<ns.backend_regularization_attempts<<'\n';
}
void healthy_fixture() {
  auto p=problem(1e-4,1);auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.update_matrix();check(w.factorize(0),"healthy factor failed");
  const auto d=w.direction({}, {first_rhs,second_rhs}, {0,0}, {});
  check(independent_error(p,d,{0,0})/first_rhs<=tolerance,
        "healthy original KKT direction is invalid");
  check(ns.factor_attempts==1&&ns.backend_regularization_attempts==0,
        "healthy direction incurred an unnecessary recovery");
}
void inconsistent_fixture() {
  auto p=problem(0,1);auto st=settings();
  LazyNewtonStatistics ns;detail::NewtonCostLedger costs(ns,Clock::now());
  Workspace w(p,st,ns,costs);w.update_matrix();
  check(w.factorize(0),"bounded preconditioner for duplicate rows did not factor");
  bool rejected=false;
  try {w.direction({}, {first_rhs,second_rhs}, {0,1}, {});}
  catch(const std::runtime_error&) {rejected=true;}
  // Identical left hand sides with RHS 0 and -1 cannot both be within 1e-7.
  check(rejected,"inconsistent original equality equations were accepted");
  check(ns.backend_regularization_attempts<=5,
        "inconsistent equations caused an unbounded recovery ladder");
}
}
int main() {
  if(!linear_backend_compiled("mumps")) {
    std::cout<<"SKIP: real MUMPS+METIS backend required\n";return 0;
  }
  try {
    cancellation_fixture(false);
    cancellation_fixture(true);
    cancellation_fixture(true,true);
    cancellation_fixture(true,false,true);
    healthy_fixture();inconsistent_fixture();
    std::cout<<"expanded original KKT precision recovery: PASS\n";
  } catch(const std::exception& e) {
    std::cerr<<e.what()<<'\n';return 1;
  }
}
