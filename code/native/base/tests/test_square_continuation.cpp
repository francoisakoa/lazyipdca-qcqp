// Public-controller tests: recovery, rollback, geometry and shared budgets.
#include "lazyipdca_qcqp/dca_filter.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
Settings settings(const std::string& backend="native") {
 Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
 s.linear_backend=backend;s.linear_metis=true;s.dca_filter_curvature_backend="same";
 s.dca_aggregate=false;s.dca_krylov=false;s.dca_filter_sparse_fallback=false;
 s.live_display_level=2;s.verbose=false;s.termination_test="project";
 s.max_outer_iterations=300;s.time_limit_seconds=10;
 s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-8;
 s.acceptable_iterations=0;s.acceptable_on_failure=false;
 s.restoration_max_iterations=20;s.restoration_max_restarts=0;return s;
}
QCQPProblem example() {
 QCQPProblem p;p.name="quadratic_square_recovery";p.n=2;
 p.x0={0,0};p.lb=p.ub=Vec(2,INFINITY);p.objective.constant=7;
 p.objective.linear={0,0};p.objective.hessian.n=2;
 QuadraticFunction a;a.constant=-2;a.linear={-1,3};a.hessian.n=2;
 a.hessian.entries={{0,0,6},{0,1,-4},{1,1,-2}};
 QuadraticFunction b;b.constant=0;b.linear={3,-2};b.hessian.n=2;
 b.hessian.entries={{0,0,4},{0,1,-7},{1,1,6}};
 p.equalities={a,b};p.validate();return p;
}
struct Run {Result r;std::string log;};
Run run(const QCQPProblem& p,const Settings& s) {
 std::ostringstream out;auto* old=std::cout.rdbuf(out.rdbuf());
 try{auto r=solve_filter_dca(p,s);std::cout.rdbuf(old);return {std::move(r),out.str()};}
 catch(...){std::cout.rdbuf(old);throw;}
}
void recovery(const std::string& backend) {
 auto p=example();const auto original=p.x0;auto s=settings(backend);
 s.restoration_enabled=false;auto before=run(p,s);
 require(before.r.status==Status::LineSearchFailure,"synthetic control must fail its Newton filter");
 require(!before.r.dca_filter.square_continuation_attempted,"disabled recovery was used");
 s.restoration_enabled=true;auto after=run(p,s);
 if(after.r.status!=Status::Solved)std::cerr<<after.log;
 require(after.r.status==Status::Solved,"continuation did not recover a feasible root");
 require(after.r.dca_filter.square_continuation_attempted&&after.r.dca_filter.square_continuation_accepted,"recovery not instrumented");
 require(after.r.dca_filter.square_continuation_parameter==1,"partial homotopy root certified");
 require(p.x0==original,"frozen initial point changed");
 for(const auto& q:p.equalities)require(std::abs(q.value(after.r.x))<=1e-8,"original equality not satisfied");
 require(after.r.objective==7,"original constant objective changed");
 for(double y:after.r.certificate_equality_multipliers)require(y==0,"stage multiplier leaked into original certificate");
 require(after.r.dca_filter.square_continuation_linear_error<=s.kkt_backward_error_reject,"unverified stage direction accepted");
 require(after.log.find("original_filter=1 square_continuation_v1=1")!=std::string::npos,"original filter not checked at commit");
 p.name="unrelated_renamed_system";auto renamed=run(p,s);
 require(renamed.r.x==after.r.x,"numerical correction depends on problem name");
}
void natural_linear_reuse() {
 if(!linear_backend_compiled("cholesk"))return;
 auto p=example();auto s=settings("cholesk");s.linear_metis=false;s.restoration_max_iterations=1;
 auto a=run(p,s);
 if(a.r.status!=Status::Solved)std::cerr<<a.log;
 require(a.r.status==Status::Solved,"natural-order continuation did not solve synthetic system");
 require(a.r.dca_filter.square_continuation_accepted,"natural-order recovery absent");
 require(a.r.dca_filter.square_continuation_krylov_iterations>0&&a.r.dca_filter.square_continuation_reused_directions>0,"factor reuse not exercised");
}
void translation_and_scaling() {
 auto p=example();auto s=settings();Vec a={2,-1};
 // h'(x)=h(x-a), so h'(a+y)=h(y). Nothing is replaced by a stored root.
 for(auto& q:p.equalities) {
  const auto old=q.linear;double shift=0;Vec ha(2,0);
  for(auto e:q.hessian.entries){shift+=(e.i==e.j?.5:1.)*e.v*a[e.i]*a[e.j];ha[e.i]+=e.v*a[e.j];if(e.i!=e.j)ha[e.j]+=e.v*a[e.i];}
  q.constant+=shift-old[0]*a[0]-old[1]*a[1];
  for(int j=0;j<2;++j)q.linear[j]-=ha[j];
 }
 p.x0=a;auto r=run(p,s);require(r.r.status==Status::Solved,"translated problem failed");
 for(auto& q:p.equalities)require(std::abs(q.value(r.r.x))<1e-8,"translated original residual");
 p=example();auto& q=p.equalities[0];q.constant*=3;for(auto& v:q.linear)v*=3;for(auto& e:q.hessian.entries)e.v*=3;
 auto scaled=run(p,s);require(scaled.r.status==Status::Solved,"scaled equations failed");
}
void rollback_and_budget() {
 auto p=example();p.n=1;p.x0={0};p.lb=p.ub=Vec(1,INFINITY);
 p.objective.linear={0};p.objective.hessian.n=1;
 QuadraticFunction q;q.constant=1;q.linear={1};q.hessian.n=1;q.hessian.entries={{0,0,2}};p.equalities={q};p.validate();
 auto s=settings();s.restoration_enabled=false;auto before=run(p,s);
 s.restoration_enabled=true;auto after=run(p,s);
 require(after.r.status!=Status::Solved,"infeasible scalar quadratic incorrectly solved");
 require(after.r.dca_filter.square_continuation_attempted&&!after.r.dca_filter.square_continuation_accepted,"failed recovery hidden");
 require(after.r.dca_filter.square_continuation_parameter<1,"failed homotopy parameter wrong");
 require(before.r.x==after.r.x,"a partial stage changed the accepted original state");
 p=example();s=settings();s.max_outer_iterations=10;
 auto limited=run(p,s);
 require(limited.r.status==Status::MaxOuterIterations,"shared iteration limit ignored");
 require(!limited.r.dca_filter.square_continuation_accepted,"incomplete stage certified at limit");
 require(limited.r.outer_iterations+limited.r.dca_filter.square_continuation_iterations<=10,"extra uncounted iteration budget");
}
void fixed_variable_reduction() {
 auto p=example();p.n=3;p.x0={0,0,3};p.lb=p.ub=Vec{INFINITY,INFINITY,3};
 p.objective.constant=-2;p.objective.linear={0,0,0};p.objective.hessian.n=3;p.objective.hessian.entries={{2,2,2}};
 for(auto& q:p.equalities){q.linear.push_back(0);q.hessian.n=3;}
 p.equalities[0].linear[0]-=6;p.equalities[0].hessian.entries.push_back({0,2,2});p.validate();
 auto a=run(p,settings());require(a.r.status==Status::Solved,"fixed-variable continuation failed");
 require(a.r.dca_filter.square_continuation_accepted&&a.r.x[2]==3,"fixed coordinate or free-variable policy wrong");
 for(auto& q:p.equalities)require(std::abs(q.value(a.r.x))<1e-8,"full original row after substitution wrong");
}
void no_op() {
 auto p=example();auto s=settings();
 p.x0={-0.18210605529715138,0.5387383482318471};auto already=run(p,s);
 require(already.r.status==Status::Solved&&!already.r.dca_filter.square_continuation_attempted,"feasible start unnecessarily continued");
 p=example();p.objective.linear[0]=1;s.max_outer_iterations=1;auto affine=run(p,s);
 require(!affine.r.dca_filter.square_continuation_attempted,"nonconstant objective was continued");
 p=example();p.objective.hessian.entries={{0,0,2}};auto curved=run(p,s);
 require(!curved.r.dca_filter.square_continuation_attempted,"curved objective was continued");
}
}
int main(){try {
 recovery("native");for(const std::string b:{"mumps","ldl","cholesk"})if(linear_backend_compiled(b))recovery(b);
 natural_linear_reuse();translation_and_scaling();rollback_and_budget();fixed_variable_reduction();no_op();
 std::cout<<"square continuation structural and rollback tests PASS\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
