#include "lazyipdca_qcqp/lazy_newton.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <iostream>
#include <stdexcept>
#include <limits>
using namespace lazyipdca::qcqp;
namespace {
QuadraticFunction q(int n,double c,Vec l,std::vector<SymTriplet> h) {
  QuadraticFunction f;f.constant=c;f.linear=std::move(l);f.hessian.n=n;f.hessian.entries=std::move(h);return f;
}
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
Settings settings() {
  Settings s;s.outer_algorithm="lazy-newton";s.ln_engine="atomic";s.linear_backend="native";s.verbose=false;
  s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
  s.max_outer_iterations=150;return s;
}
QCQPProblem normal_problem() {
  QCQPProblem p;p.name="negative-normal-curvature";p.n=2;p.x0={0.0,2.0};
  p.lb={-INFINITY,-5};p.ub={INFINITY,5};
  p.objective=q(2,0,{0,0},{{0,0,-100},{1,1,2}});
  p.equalities.push_back(q(2,0,{1,0},{}));return p;
}
void normal_curvature() {
  auto p=normal_problem();auto s=settings();
  for(bool reduction:{false,true}) {
    s.lazy_newton_reduction=reduction;auto r=Solver(s).solve(p);
    if(r.status!=Status::Solved)std::cerr<<r.message<<"\n";
    require(r.status==Status::Solved,"negative NORMAL curvature must remain admissible");
    require(r.lazy_newton.regularization_increases==0,"normal curvature caused a spurious correction");
    require(r.lazy_newton.unshifted_accepted_steps>0,"unshifted Newton was never used");
    require(r.lazy_newton.structure_builds==1,"structure was rebuilt");
    require(certify_lazy_newton_original(p,s,r),"original certificate rejected normal test");
    require(std::abs(r.x[1])<1e-5,"wrong normal-test optimum");
  }
  // A genuinely quadratic equality, with either sign of its multiplier.
  for(double sign:{-1.0,1.0}) {
    p.x0={1,2};p.objective=q(2,0,{100,0},{{0,0,-200},{1,1,2}});
    p.equalities={q(2,-sign,{0,0},{{0,0,2*sign}})};
    auto r=Solver(s).solve(p);
    require(r.status==Status::Solved,"quadratic normal-curvature solve failed");
    require(r.lazy_newton.regularization_increases==0,"quadratic normal curvature caused a spurious correction");
    require(std::abs(r.certificate_equality_multipliers[0]-50*sign)<1e-5,"signed equality multiplier was lost");
  }
}
void tangent_curvature() {
  auto p=normal_problem();p.name="negative-tangent-curvature";p.x0={0,0.2};p.lb[1]=-1;p.ub[1]=1;
  p.objective.hessian.entries={{0,0,-100},{1,1,-1}};
  auto s=settings();s.lazy_newton_initial_mu=1e-3;auto r=Solver(s).solve(p);
  if(r.status!=Status::Solved)std::cerr<<r.message<<"\n";
  require(r.status==Status::Solved,"bounded tangent-negative problem failed");
  require(r.lazy_newton.regularization_increases>0,"negative tangent curvature was not corrected");
  require(r.lazy_newton.regularization_decreases>0,"regularization was never decreased");
  require(std::abs(std::abs(r.x[1])-1)<1e-4,"wrong bounded concave optimum");
}
void nonlinear_reduction() {
  QCQPProblem p;p.name="nonlinear-lift-general-quadratic";p.n=3;p.x0={0.6,0.2,2.0};
  p.lb={0.1,-4,2};p.ub={2,4,2};
  p.objective=q(3,0.5,{0,-1,0},{{0,0,0.02},{0,1,0.1},{1,1,1},{0,2,0.05}});
  p.equalities.push_back(q(3,0,{0,1,0},{{0,0,-2}}));
  p.inequalities.push_back(q(3,-3,{0,0,0},{{0,1,0.5},{1,1,0.2}}));
  auto s=settings();Result reduced,full;
  for(bool reduction:{true,false}) {
    s.lazy_newton_reduction=reduction;auto r=Solver(s).solve(p);
    if(r.status!=Status::Solved)std::cerr<<r.message<<"\n";
    require(r.status==Status::Solved,"nonlinear lift solve failed");
    require(certify_lazy_newton_original(p,s,r),"nonlinear lift original certificate failed");
    require(r.x[2]==2.0,"fixed parameter was changed");
    require(r.lazy_newton.eliminated_variables==(reduction?1:0),"unexpected reduction dimension");
    if(reduction)reduced=r;else full=r;
  }
  require(std::abs(reduced.objective-full.objective)<1e-6,"reduction changed optimum");
  // No process-global model cache: changed coefficients must be compiled anew.
  p.equalities[0].hessian.entries[0].v=-4;
  auto changed=Solver(s).solve(p);require(changed.status==Status::Solved,"changed model failed");
  require(std::abs(changed.x[1]-2*changed.x[0]*changed.x[0])<1e-6,"stale model structure/numerics reused");
}
void invalid_certificates() {
  auto p=normal_problem();auto s=settings();auto r=Solver(s).solve(p);
  require(r.status==Status::Solved,"certificate baseline failed");
  auto bad=r;bad.certificate_upper_multipliers[1]=-1;
  require(!certify_lazy_newton_original(p,s,bad),"negative multiplier accepted");
  bad=r;bad.x[0]=1;require(!certify_lazy_newton_original(p,s,bad),"infeasible point accepted");
  bad=r;bad.x[1]=std::numeric_limits<double>::quiet_NaN();require(!certify_lazy_newton_original(p,s,bad),"NaN accepted");
  bad=r;bad.certificate_equality_multipliers.clear();require(!certify_lazy_newton_original(p,s,bad),"missing dual accepted");
  p.equalities.push_back(p.equalities[0]);p.x0={1,2};
  // Identical equalities are redundant but FEASIBLE. If a backend returns
  // a solution despite the singular multiplier system, the original KKT
  // certificate must pass. Contradictory duplicate rows must never solve.
  auto singular=Solver(s).solve(p);
  require(singular.status!=Status::Solved||certify_lazy_newton_original(p,s,singular),"uncertified redundant equality solution");
  p.equalities.back().constant=-1;
  auto inconsistent=Solver(s).solve(p);require(inconsistent.status!=Status::Solved,"inconsistent duplicate equalities falsely solved");
}
void lazy_activation() {
  QCQPProblem p;p.name="omitted-constraint-blocks-full-step";p.n=1;p.x0={-2};p.lb={-5};p.ub={5};
  p.objective=q(1,4.5,{-3},{{0,0,1}});p.inequalities.push_back(q(1,-1,{1},{}));
  auto s=settings();auto r=Solver(s).solve(p);
  require(r.status==Status::Solved,"lazy activation solve failed");
  require(r.lazy_newton.constraint_activations==1&&r.lazy_newton.working_set_retries>0,"blocking constraint was not activated transactionally");
  require(std::abs(r.x[0]-1)<1e-5,"omitted constraint was violated at termination");
}
void sparse_zero_envelope() {
  if(!linear_backend_compiled("ldl"))return;
  auto s=settings();s.linear_backend="ldl";s.kkt_adaptive_regularization=true;
  s.kkt_primal_first_ordering=true;s.kkt_preserve_structural_zeros=true;
  std::string message;auto f=make_symmetric_factorization(s,message);
  std::vector<SymmetricTriplet> a={{0,0,-2},{1,1,3},{0,2,1},{0,1,0},{1,2,0},{2,2,0}};
  require(f->factor(3,2,a,message),"normal-negative sparse factor failed");
  Vec rhs={1,2,0};require(f->solve(rhs,message),"sparse solve failed");
  require(std::abs(rhs[0])<1e-12&&std::abs(rhs[1]-2.0/3.0)<1e-12,"equality block was perturbed");
  a[3].value=0.1;a[4].value=0.2;
  require(f->factor(3,2,a,message),"zero-to-nonzero structure update failed");
  require(f->symbolic_analyses()==1&&f->numeric_scatter_builds()==1&&f->numeric_scatter_reuses()==1,"permanent envelope/scatter was not reused");
  a[1].value=-3;
  require(f->factor(3,1,a,message),"changed primal partition failed");
  require(f->symbolic_analyses()==2,"primal partition change retained a stale constrained ordering");
}
void determined_and_fixed() {
  auto s=settings();QCQPProblem p;p.n=1;p.x0={-1};p.lb={-5};p.ub={5};
  p.objective=q(1,0,{0},{{0,0,1}});p.equalities={q(1,-1,{1},{})};
  auto r=Solver(s).solve(p);require(r.status==Status::Solved&&std::abs(r.x[0]-1)<1e-6,"fully determined problem failed");
  p.lb={1};p.ub={1};r=Solver(s).solve(p);
  require(r.status==Status::Solved&&r.x[0]==1,"fully fixed problem failed");
  p.equalities[0].constant=-2;r=Solver(s).solve(p);require(r.status!=Status::Solved,"inconsistent fixed problem accepted");
}
void adaptive_start() {
  auto st=settings();QCQPProblem p;p.n=1;p.x0={-1};p.lb={-5};p.ub={5};
  p.objective=q(1,0,{0},{{0,0,1}});p.equalities={q(1,-1,{1},{})};
  auto adaptive=Solver(st).solve(p);
  require(adaptive.status==Status::Solved,"infeasible adaptive start failed");
  require(adaptive.lazy_newton.initial_barrier>st.lazy_newton_initial_mu,"infeasible start retained a terminal barrier");
  st.lazy_newton_adaptive_barrier=false;auto legacy=Solver(st).solve(p);
  require(legacy.status==Status::Solved,"fixed barrier control failed");
  require(legacy.lazy_newton.initial_barrier==st.lazy_newton_initial_mu,"barrier opt-out ignored");
  require(std::abs(adaptive.objective-legacy.objective)<1e-6,"adaptive barrier changed constrained optimum");
  p.x0={1};st.lazy_newton_adaptive_barrier=true;auto feasible=Solver(st).solve(p);
  require(feasible.status==Status::Solved,"feasible start regressed");
  require(feasible.lazy_newton.initial_barrier==st.lazy_newton_initial_mu,"feasible start was unnecessarily perturbed");
}
void interleaved_order() {
  if(!linear_backend_compiled("ldl"))return;
  auto st=settings();st.linear_backend="ldl";st.kkt_adaptive_regularization=true;
  st.kkt_interleaved_equality_ordering=true;st.kkt_preserve_structural_zeros=true;
  std::string msg;auto f=make_symmetric_factorization(st,msg);
  std::vector<SymmetricTriplet> a={{0,0,1},{1,1,2},{2,2,3},{3,3,0},{4,4,0},
      {0,3,1},{1,3,1},{1,4,1},{2,4,1},{0,2,0}};
  const Vec exact={1,2,-1,3,-2};
  for(double cross:{0.0,0.25,0.0}) {
    a.back().value=cross;Vec rhs(5,0.0);
    for(auto e:a){rhs[e.i]+=e.value*exact[e.j];if(e.i!=e.j)rhs[e.j]+=e.value*exact[e.i];}
    require(f->factor(5,3,a,msg),"interleaved exact-zero equality factor failed");
    require(f->solve(rhs,msg),"interleaved solve failed");
    for(int j=0;j<5;++j)require(std::abs(rhs[j]-exact[j])<1e-10,"interleaved permutation/scatter corrupted solution");
  }
  require(f->symbolic_analyses()==1&&f->numeric_scatter_builds()==1,"interleaved structural-zero pattern was rebuilt");
}

void sparse_normal_recovery() {
  if(!linear_backend_compiled("ldl"))return;
  // A genuine saddle matrix with a zero normal primal pivot. The equality
  // fixes x0=1; the tangent minimizer is x1=0, lambda=-1. Omitting the RHS
  // transformation would produce the wrong equality multiplier.
  auto st=settings();st.linear_backend="ldl";st.lazy_newton_reduction=false;
  QCQPProblem p;p.n=2;p.x0={0,3};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective=q(2,0,{1,0},{{1,1,2}});p.equalities={q(2,-1,{1,0},{})};
  for(bool recover:{false,true}) {
    st.lazy_newton_sparse_recovery=recover;
    auto r=Solver(st).solve(p);
    require(r.status==Status::Solved,"zero normal pivot recovery did not solve original QCQP");
    require(std::abs(r.x[0]-1)<1e-9&&std::abs(r.x[1])<1e-9,"recovery changed primal Newton step");
    require(std::abs(r.certificate_equality_multipliers[0]+1)<1e-9,"recovery omitted or corrupted RHS transformation");
    require(r.lazy_newton.regularization_increases==0,"normal recovery added spurious primal regularization");
    if(recover) {
      require(r.lazy_newton.normal_recovery_successes>0&&r.lazy_newton.pivoted_fallbacks==0,"sparse recovery did not replace native fallback");
      require(r.lazy_newton.normal_recovery_linear_solves>0,"recovery factor was not used");
    } else {
      require(r.lazy_newton.pivoted_fallbacks>0&&r.lazy_newton.native_linear_solves>0,"native fallback control was not exercised");
      require(r.lazy_newton.native_factor_seconds>=0&&r.lazy_newton.native_solve_seconds>=0,"invalid native timing");
    }
    require(certify_lazy_newton_original(p,st,r),"recovery original certificate failed");
  }
  // Equality scaling must not change the transported multiplier or solution.
  st.lazy_newton_sparse_recovery=true;
  p.equalities={q(2,-1e-4,{1e-4,0},{})};
  auto scaled=Solver(st).solve(p);
  require(scaled.status==Status::Solved&&std::abs(scaled.x[0]-1)<1e-9,"scaled normal equality recovery failed");
  require(std::abs(scaled.certificate_equality_multipliers[0]+1e4)<1e-5,"scaled equality multiplier transport failed");
  // Without an equality incident to a zero pivot, native 2x2 pivoting must
  // remain available. A bounded negative tangent direction is still corrected.
  p.n=2;p.x0={0.2,0.4};p.lb={-1,-1};p.ub={1,1};p.equalities.clear();
  p.objective=q(2,0,{0,0},{{0,1,1}});
  auto bounded=Solver(st).solve(p);
  require(bounded.status==Status::Solved,"general negative-curvature guard regressed");
  require(bounded.lazy_newton.normal_recovery_successes==0,"invented an equality for normal recovery");
  require(bounded.lazy_newton.regularization_increases>0,"negative tangent curvature bypassed inertia checks");
}

}
int main() {
  try {normal_curvature();tangent_curvature();nonlinear_reduction();invalid_certificates();lazy_activation();sparse_zero_envelope();determined_and_fixed();adaptive_start();interleaved_order();sparse_normal_recovery();}
  catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
  std::cout<<"lazy-newton tangent curvature, reduction and certificates: PASS\n";
}
