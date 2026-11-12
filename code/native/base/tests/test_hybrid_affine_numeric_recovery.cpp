#include "../src/atomic_newton.cpp"
#include <iostream>
#include <stdexcept>

using namespace lazyipdca::qcqp;
namespace {
int checks=0;
constexpr double tau=.1,threshold=1e-7;
void require(bool ok,const char* message) {
  ++checks;if(!ok)throw std::runtime_error(message);
}
QCQPProblem problem(bool duplicate) {
  QCQPProblem p;p.name="fixed_hybrid_affine_repair";p.n=2;
  p.x0={.25,.5};p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);
  p.objective.linear={0,0};p.objective.hessian.n=2;
  p.objective.hessian.entries={{0,0,2},{1,1,2}};
  QuadraticFunction h;h.constant=-1;h.linear={1,1};h.hessian.n=2;
  p.equalities.push_back(h);
  if(!duplicate)h.linear[1]=-1;
  p.equalities.push_back(h);
  QuadraticFunction g;g.constant=-4;g.linear={0,0};g.hessian.n=2;
  g.hessian.entries={{0,0,2}};p.inequalities.push_back(g);
  p.validate();return p;
}
Settings settings() {
  Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";
  st.convexification_study="HH";st.study_controller="certified";
  st.study_inner_rule="model-relative";st.linear_backend="mumps";
  st.linear_metis=true;st.dca_pair_elimination=false;st.dca_rank_presolve=false;
  st.lazy_newton_reduction=false;st.lazy_newton_atomic_regularization=false;
  st.lazy_newton_lazy_constraints=false;st.verbose=false;st.live_display_level=0;
  st.lazy_newton_objective_gradient_target=1000;
  st.lazy_newton_constraint_gradient_target=1000;
  st.kkt_backward_error_reject=threshold;return st;
}
void prepare(Workspace& w) {
  w.s={3.9375};w.z={1};w.lambda={0,0};w.regularization=tau;
  w.evaluate();w.update_matrix();
}
long double independent_original_equations(const QCQPProblem& p,
    const Direction& d,const Vec& eq) {
  // Direct four-block oracle. Neither backend shifts nor Workspace's audit
  // or its assembled matrices enter these unperturbed model equations.
  long double e0=(4.L+tau)*d.x[0]+.5L*d.z[0];
  long double e1=(2.L+tau)*d.x[1];
  long double worst=0;
  for(int e=0;e<2;++e) {
    const auto& a=p.equalities[e].linear;
    e0+=static_cast<long double>(a[0])*d.lambda[e];
    e1+=static_cast<long double>(a[1])*d.lambda[e];
    worst=std::max(worst,std::abs(static_cast<long double>(eq[e])+
        static_cast<long double>(a[0])*d.x[0]+static_cast<long double>(a[1])*d.x[1]));
  }
  return std::max({worst,std::abs(e0),std::abs(e1),
      std::abs(.5L*d.x[0]+d.s[0]),std::abs(d.s[0]+3.9375L*d.z[0])});
}
template<class F> bool rejected(F&& f) {
  try {f();return false;}catch(const std::runtime_error&){return true;}
}
void recovered_fixed_system() {
  const auto p=problem(true);auto st=settings();
  LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());
  Workspace w(p,st,stats,costs);prepare(w);
  require(w.nr==2&&w.mr==2&&w.mi==1,"fixture reduced away its duplicate equations");
  // This is a genuinely singular original KKT with compatible duplicated
  // equations. The shifted factor is only a refinement preconditioner.
  require(!w.factorize(tau,true),"historical frozen SOC unexpectedly recovered a singular factor");
  require(stats.factor_attempts==1&&stats.backend_regularization_attempts==0,
      "historical frozen SOC entered a numeric recovery ladder");
  const Workspace::AffineRepairFactorPermit unchanged(w);
  const int prior_attempts=stats.factor_attempts;
  require(w.factorize_affine_repair(tau),"fixed-model affine repair failed numerical recovery");
  require(unchanged.same_model_and_point(w),"recovery changed tau, retained coordinates, blocks or anchor");
  require(w.current_factor==w.pivot_rescue.get(),"fixture did not exercise the recovered factor");
  require(stats.factor_attempts>prior_attempts&&stats.factor_attempts<=prior_attempts+9,
      "recovery factor attempts were missing or unbounded");
  require(stats.backend_regularization_attempts>=1&&stats.backend_regularization_attempts<=5,
      "shifted preconditioner work was not bounded and recorded");
  require(stats.maximum_backend_primal_regularization<=1e-8*(1+1e-12)&&
      stats.maximum_backend_dual_regularization<=1e-8*(1+1e-12),"existing numeric shift cap changed");
  require(stats.condensation_recoveries==0&&stats.pivoted_fallbacks==0&&
      stats.regularization_increases==0,"fixed repair changed representation, backend or mathematical tau");
  const Workspace::AffineRepairFactorPermit permit(w);
  const int attempts=stats.factor_attempts,solves=stats.linear_solves;
  const Vec eq=w.h;
  const Direction d=w.direction({0},{0,0},eq,{0},0,true,DirectionPurpose::Complete,true,&permit);
  const long double error=independent_original_equations(p,d,eq);
  require(error<=threshold&&d.residual<=threshold,"recovery failed independent unperturbed four-block audit");
  require(permit.matches(w)&&stats.factor_attempts==attempts&&stats.linear_solves>solves,
      "correction or refinement replaced its frozen factor or omitted linear work");
  require(stats.cost_seconds[int(NewtonCost::Factorization)]>0&&
      stats.cost_seconds[int(NewtonCost::Direction)]>0,"numeric work was omitted from its cost ledger");
  require(rejected([&]{w.direction({0},{0,0},eq,{0},0,true,DirectionPurpose::Complete,true);}),
      "SOC without a repair permit accepted a recovered factor");
  const int before_wrong_tau=stats.factor_attempts;
  bool wrong_tau=false;
  try {w.factorize_affine_repair(2*tau);}catch(const std::invalid_argument&){wrong_tau=true;}
  require(wrong_tau&&w.regularization==tau&&stats.factor_attempts==before_wrong_tau,
      "fixed repair accepted a changed mathematical tau");
  const double x=w.x[0];w.x[0]=std::nextafter(x,INFINITY);
  require(rejected([&]{w.direction({0},{0,0},eq,{0},0,true,DirectionPurpose::Complete,true,&permit);}),
      "permit survived an anchor change");w.x[0]=x;
  const double matrix=w.kkt[0].value;w.kkt[0].value=std::nextafter(matrix,INFINITY);
  require(rejected([&]{w.direction({0},{0,0},eq,{0},0,true,DirectionPurpose::Complete,true,&permit);}),
      "permit survived an unperturbed block change");w.kkt[0].value=matrix;
  const auto kept=w.retained;std::reverse(w.retained.begin(),w.retained.end());
  require(rejected([&]{w.direction({0},{0,0},eq,{0},0,true,DirectionPurpose::Complete,true,&permit);}),
      "permit survived a retained-coordinate change");w.retained=kept;
  auto wrong=d;wrong.lambda[0]+=1e-3;
  require(independent_original_equations(p,wrong,eq)>threshold,
      "independent oracle accepts an actually invalid original direction");
  Vec bad_eq=eq;bad_eq[1]+=1;
  const int before_bad_solves=stats.linear_solves;
  require(rejected([&]{w.direction({0},{0,0},bad_eq,{0},0,true,DirectionPurpose::Complete,true,&permit);}),
      "inconsistent original equations passed the shifted factor's audit");
  require(permit.matches(w)&&stats.factor_attempts==attempts&&stats.linear_solves>before_bad_solves,
      "rejected original audit changed the factor or lost its linear-work accounting");
  require(rejected([&]{w.direction({0},{NAN,0},eq,{0},0,true,DirectionPurpose::Complete,true,&permit);}),
      "nonfinite original RHS entered a recovered correction");
  require(rejected([&]{w.direction({0},{0,0},eq,{0},0,false,DirectionPurpose::Complete,true,&permit);}),
      "repair permit disabled its original expanded audit");
  require(permit.matches(w),"rejected nonfinite or unaudited correction changed its frozen state");
  require(w.factorize_affine_repair(tau),"second matching numeric preparation failed");
  require(rejected([&]{w.direction({0},{0,0},eq,{0},0,true,DirectionPurpose::Complete,true,&permit);}),
      "stale permit survived a refactorization");
  std::cout<<"fixed_model_recovered original_four_block_error="<<static_cast<double>(error)
      <<" shift="<<stats.final_backend_primal_regularization
      <<" factor_attempts="<<stats.factor_attempts<<" linear_solves="<<stats.linear_solves
      <<" tau_unchanged=1 original_threshold="<<threshold<<'\n';
}
void healthy_soc_contract() {
  const auto p=problem(false);auto st=settings();
  LazyNewtonStatistics stats;detail::NewtonCostLedger costs(stats,Clock::now());
  Workspace w(p,st,stats,costs);prepare(w);
  require(w.factorize(tau,true),"healthy historical frozen factor failed");
  const int attempts=stats.factor_attempts;auto* factor=w.current_factor;
  const auto d=w.direction({0},{0,0},w.h,{0},0,true,DirectionPurpose::Complete,true);
  require(independent_original_equations(p,d,w.h)<=threshold,"healthy SOC original audit failed");
  require(stats.factor_attempts==attempts&&w.current_factor==factor&&
      stats.backend_regularization_attempts==0,"healthy historical SOC contract changed");
}
}
int main() {
  try {
    require(linear_backend_compiled("mumps"),"real MUMPS backend is required; no skip qualifies this test");
    recovered_fixed_system();healthy_soc_contract();
    std::cout<<"hybrid_affine_numeric_recovery PASS checks="<<checks
      <<" unchanged_model=1 unchanged_SOC_contract=1 unperturbed_audit=1\n";
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
