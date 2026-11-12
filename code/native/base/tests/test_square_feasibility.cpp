// Structural normal-step tests through the public controller. No benchmark name.
#include "lazyipdca_qcqp/dca_filter.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool p,const char* message){if(!p)throw std::runtime_error(message);}
Settings options(const std::string& backend="native") {
 Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
 s.linear_backend=backend;s.linear_metis=true;s.dca_filter_curvature_backend="same";
 s.dca_aggregate=false;s.dca_aggregate_only=false;s.dca_krylov=false;
 s.lazy_newton_lazy_constraints=false;s.lazy_newton_atomic_regularization=false;
 s.dca_filter_sparse_fallback=false;s.live_display_level=2;s.verbose=false;
 s.termination_test="project";s.max_outer_iterations=40;s.time_limit_seconds=10;
 s.feasibility_tolerance=1e-8;s.stationarity_tolerance=1e-8;
 s.dual_infeasibility_tolerance=1e-8;s.complementarity_tolerance=1e-8;
 s.acceptable_iterations=0;s.acceptable_on_failure=false;
 s.restoration_max_iterations=20;s.restoration_max_restarts=0;
 return s;
}
QCQPProblem square() {
 QCQPProblem p;p.name="synthetic_regular_zero_objective";p.n=2;
 p.x0={1.2,2.1};p.lb.assign(2,INFINITY);p.ub.assign(2,INFINITY);
 p.objective.constant=7;p.objective.linear={0,0};p.objective.hessian.n=2;
 QuadraticFunction a;a.name="first";a.constant=-4;a.linear={0,0};a.hessian.n=2;
 a.hessian.entries={{0,0,2}};
 QuadraticFunction b;b.name="second";b.constant=-9;b.linear={0,0};b.hessian.n=2;
 b.hessian.entries={{1,1,2}};p.equalities={a,b};p.validate();return p;
}
struct Run {Result r;std::string text;};
Run run(const QCQPProblem& p,const Settings& s) {
 std::ostringstream o;auto* old=std::cout.rdbuf(o.rdbuf());
 try {auto r=solve_filter_dca(p,s);std::cout.rdbuf(old);return {std::move(r),o.str()};}
 catch(...){std::cout.rdbuf(old);throw;}
}
void regular_square(const std::string& backend) {
 auto p=square();auto s=options(backend);auto a=run(p,s);
 require(a.r.status==Status::Solved,"normal step did not solve regular square equations");
 require(a.r.restoration_iterations==0,"regular square unnecessarily restored");
 require(a.text.find("square_normal_selected=1")!=std::string::npos,"normal startup not selected");
 require(a.text.find("direction=square-feasibility-normal")!=std::string::npos,"normal direction unlabelled");
 require(std::abs(a.r.x[0]-2)<1e-8&&std::abs(a.r.x[1]-3)<1e-8,"wrong root");
 for(double x:a.r.certificate_equality_multipliers)require(x==0,"normal multiplier leaked into original KKT");
 auto renamed=p;renamed.name="renamed_unrelated_equations";
 require(run(renamed,s).r.x==a.r.x,"numeric path depends on problem name");
 s.restoration_enabled=false;auto b=run(p,s);
 require(b.r.x==a.r.x&&b.r.restoration_iterations==0,"normal step depends on a hidden restoration");
}
void protected_structures() {
 auto p=square();auto s=options();s.max_outer_iterations=1;
 p.objective.linear[0]=1;auto a=run(p,s);
 require(a.text.find("square_normal_selected=1")==std::string::npos,"affine objective treated as constant");
 require(a.r.restoration_iterations>0,"affine historical startup lost");
 p=square();p.objective.hessian.entries={{0,0,2}};p.objective.linear[0]=-2*p.x0[0];
 auto b=run(p,s);
 require(b.text.find("square_normal_selected=1")==std::string::npos,"zero gradient mistaken for constant objective");
 require(b.r.restoration_iterations==0,"curved objective startup regression");
 p=square();p.equalities.pop_back();auto c=run(p,s);
 require(c.text.find("square_normal_selected=1")==std::string::npos,"rectangular equations treated as square");
 require(c.r.restoration_iterations>0,"rectangular historical startup lost");
 p=square();QuadraticFunction g;g.constant=-10;g.linear={1,0};g.hessian.n=2;p.inequalities.push_back(g);auto d=run(p,s);
 require(d.text.find("square_normal_selected=1")==std::string::npos,"mixed constraints use square path");
 require(d.r.restoration_iterations>0,"mixed startup regression");
}
// Existing regression equations, with no benchmark name in the controller.
void failed_filter_coordinate_recovery() {
 auto p=square();p.name="anonymous_disconnected_quadratic_equalities";p.x0={1,1};
 p.lb=Vec(2,-INFINITY);p.ub=Vec(2,INFINITY);p.objective.constant=0;
 p.equalities[0].constant=-1;p.equalities[0].linear={0,12};
 p.equalities[0].hessian.entries={{0,0,2}};
 p.equalities[1].constant=-681;p.equalities[1].linear={84,2324};
 p.equalities[1].hessian.entries={{0,0,98},{1,1,98}};
 auto s=options("mumps");if(!linear_backend_compiled("mumps"))s=options();
 s.restoration_max_iterations=80;s.restoration_restart_iterations=80;s.restoration_max_restarts=8;
 auto a=run(p,s);
 require(a.text.find("trigger=filter-exhausted accepted=1")!=std::string::npos,
     "failed original filter did not invoke complete feasibility recovery");
 require(a.r.status==Status::Solved&&feasibility(p,a.r.x).total_inf<=s.feasibility_tolerance,
     "recovered state not independently certified");
 require(p.x0==Vec({1,1}),"filter recovery mutated the supplied start");
 require(a.r.restoration_restarts>0&&a.r.restoration_restarts<=8,
     "coordinate recoveries missing from result counters");
 s.restoration_max_restarts=0;auto b=run(p,s);
 require(b.r.status!=Status::Solved,"explicit no-restart permission ignored");
 require(b.text.find("trigger=filter-exhausted accepted=0")!=std::string::npos,
     "unsuccessful recovery falsely committed");
}
void natural_order_preserves_cheap_restoration() {
 if(!linear_backend_compiled("cholesk"))return;
 auto p=square();auto s=options("cholesk");s.linear_metis=false;
 auto a=run(p,s);require(a.r.status==Status::Solved,"natural-order cheap restoration failed");
 require(a.r.restoration_iterations>0,"cheap natural-order restoration discarded");
 require(a.text.find("normal_deferred=1")!=std::string::npos,"deferred policy unlabelled");
 s.restoration_enabled=false;auto b=run(p,s);
 require(b.r.status==Status::Solved&&b.r.restoration_iterations==0,"no-restoration ignored in natural ordering");
 require(b.text.find("square_normal_selected=1")!=std::string::npos,"disabled restoration did not use normal solve");
}
void failed_probe_does_not_commit() {
 auto p=square();p.x0={0,0};auto s=options();s.restoration_enabled=false;s.max_outer_iterations=1;
 auto a=run(p,s);require(a.r.status!=Status::Solved,"singular start incorrectly certified");
 require(a.r.x==p.x0,"failed normal probe changed the frozen point");
 require(a.text.find("usable=0")!=std::string::npos,"failed probe not recorded");
 require(a.text.find("fallback=historical-startup")!=std::string::npos,"historical policy not retained after failed probe");
}
void fixed_variables_and_row_scales() {
 auto p=square();p.n=3;p.x0.push_back(5);p.lb.push_back(5);p.ub.push_back(5);
 p.objective.linear.push_back(0);p.objective.hessian.n=3;p.objective.hessian.entries={{2,2,2}};
 for(auto& q:p.equalities){q.linear.push_back(0);q.hessian.n=3;}
 for(int k=0;k<2;++k) {
  double scale=k==0?1e-2:1e2;p.equalities[k].constant*=scale;
  for(auto& e:p.equalities[k].hessian.entries)e.v*=scale;
 }
 p.validate();auto a=run(p,options());require(a.r.status==Status::Solved,"fixed variables / row scaling failed");
 require(a.r.x[2]==5&&a.r.restoration_iterations==0,"fixed curvature changed structural decision");
 require(a.text.find("square_normal_selected=1")!=std::string::npos,"free-space square structure missed");
}
}
int main(){try {
 regular_square("native");
 for(const auto* backend:{"mumps","ldl","cholesk"})if(linear_backend_compiled(backend))regular_square(backend);
 protected_structures();failed_filter_coordinate_recovery();failed_probe_does_not_commit();fixed_variables_and_row_scales();natural_order_preserves_cheap_restoration();
 std::cout<<"PASS square feasibility: original KKT, same backend, names, fixed variables, rank failure, preserved startup\n";
}catch(const std::exception& e){std::cerr<<"FAIL square feasibility: "<<e.what()<<'\n';return 1;}}
