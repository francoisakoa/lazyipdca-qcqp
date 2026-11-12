// Exercise the public controller: no benchmark name or reconstructed benchmark.
#include "lazyipdca_qcqp/dca_filter.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
Settings settings(){
 Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
 s.linear_backend="native";s.dca_filter_curvature_backend="native";
 s.dca_aggregate=false;s.dca_aggregate_only=false;s.dca_krylov=false;
 s.lazy_newton_lazy_constraints=false;s.lazy_newton_atomic_regularization=false;
 s.verbose=false;s.live_display_level=2;s.max_outer_iterations=100;s.time_limit_seconds=10;
 s.termination_test="project";s.feasibility_tolerance=1e-8;s.stationarity_tolerance=1e-8;
 s.dual_infeasibility_tolerance=1e-8;s.complementarity_tolerance=1e-8;
 s.acceptable_iterations=0;s.acceptable_on_failure=false;return s;
}
struct Capture{std::ostringstream out;std::streambuf* p;Capture():p(std::cout.rdbuf(out.rdbuf())){}~Capture(){std::cout.rdbuf(p);}};
QCQPProblem sphere(bool extra){
 QCQPProblem p;p.name="flat_objective_infeasible_quadratic_equality";
 p.n=2;p.x0={0.1,0.1};p.lb={INFINITY,INFINITY};p.ub=p.lb;
 p.objective.linear={0,0};p.objective.hessian.n=2;
 QuadraticFunction h;h.constant=-2;h.linear={0,0};h.hessian.n=2;h.hessian.entries={{0,0,2},{1,1,2}};
 p.equalities.push_back(h);
 if(extra){h.constant=-3;p.inequalities.push_back(h);} // redundant on h=0
 p.validate();return p;
}
void redundant_inequality_does_not_disable_nonlinear_feasibility(){
 auto st=settings();Result a,b,c;std::string log;
 {Capture q;a=solve_filter_dca(sphere(false),st);}
 auto p=sphere(true);{Capture q;b=solve_filter_dca(p,st);log=q.out.str();}
 require(a.status==Status::Solved&&b.status==Status::Solved,"redundant inequality still disables restoration");
 require(b.restoration_iterations>0,"no nonlinear feasibility step");
 require(log.find("infeasible_nonlinear_equality_restore=1")!=std::string::npos,"missing structural decision");
 require(log.find("trigger=affine-infeasible-nonlinear-equalities")!=std::string::npos,"missing generic trigger");
 p.name="arbitrary_renamed_problem";{Capture q;c=solve_filter_dca(p,st);}
 require(b.x==c.x&&b.outer_iterations==c.outer_iterations,"problem name changes trajectory");
 st.restoration_enabled=false;{Capture q;c=solve_filter_dca(p,st);}
 require(c.restoration_iterations==0,"explicit restoration disable ignored");
}
void curved_objective_is_unchanged(){
 auto p=sphere(true);p.objective.hessian.entries={{0,0,2}};Result r;std::string log;
 {Capture q;r=solve_filter_dca(p,settings());log=q.out.str();}
 require(r.restoration_iterations==0,"curved objective restored");
 require(log.find("infeasible_nonlinear_equality_restore=0")!=std::string::npos,"curved guard absent");
}
void satisfied_nonlinear_equalities_are_unchanged(){
 auto p=sphere(true);p.x0={1,1};p.inequalities[0].hessian.entries.clear();
 p.inequalities[0].linear={1,0};p.inequalities[0].constant=1; // g violated, h satisfied
 auto st=settings();st.max_outer_iterations=1;Result r;
 {Capture q;r=solve_filter_dca(p,st);}
 require(r.restoration_iterations==0,"inequality violation alone triggers new startup");
}
void linear_equalities_are_unchanged(){
 auto p=sphere(true);p.equalities[0].hessian.entries.clear();p.equalities[0].linear={1,0};
 auto st=settings();st.max_outer_iterations=1;Result r;
 {Capture q;r=solve_filter_dca(p,st);}
 require(r.restoration_iterations==0,"linear equality triggers new startup");
}
void fixed_only_objective_curvature_does_not_hide_flat_free_space(){
 auto p=sphere(true);p.n=3;p.x0.push_back(0);p.lb.push_back(0);p.ub.push_back(0);
 p.objective.linear.push_back(0);p.objective.hessian.n=3;p.objective.hessian.entries={{2,2,2}};
 for(auto& h:p.equalities){h.linear.push_back(0);h.hessian.n=3;}
 for(auto& g:p.inequalities){g.linear.push_back(0);g.hessian.n=3;}
 p.validate();Result r;{Capture q;r=solve_filter_dca(p,settings());}
 require(r.restoration_iterations>0&&r.status==Status::Solved,"fixed-only curvature disabled needed free-space startup");
}
}
int main(){try{
 redundant_inequality_does_not_disable_nonlinear_feasibility();curved_objective_is_unchanged();
 satisfied_nonlinear_equalities_are_unchanged();linear_equalities_are_unchanged();
 fixed_only_objective_curvature_does_not_hide_flat_free_space();
 std::cout<<"mixed structural startup: 7 checks PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
