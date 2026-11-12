#include "lazyipdca_qcqp/majorant_auto_startup.hpp"
#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void ck(bool v,const char* s){if(!v)throw std::runtime_error(s);}
void decision_tests(){
 using namespace detail;AutoStartupFeatures f;
 f.mixed=f.affine_nonconstant_objective=f.curved_equalities=f.curved_inequalities=true;
 f.scaled_objective_gradient=100;
 auto d=choose_auto_startup(f,1e-4);ck(d.central_first&&d.initial_mu==1,"eligible");
 f.infeasible_curved_equality=true;
 d=choose_auto_startup(f,1e-4);ck(!d.central_first&&d.initial_mu==1e-4,"infeasible curved equality restores first");
 ck(d.bounded_restoration_probe&&d.central_fallback_mu==1,"ready mixed restoration is a bounded probe");
 f.estimated_dense_restoration_bytes=256.0L*1024*1024;
 d=choose_auto_startup(f,1e-4);
 ck(d.central_first&&!d.bounded_restoration_probe&&d.initial_mu==1,"large dense allocation not guarded");
 f.estimated_dense_restoration_bytes=128.0L*1024*1024;
 ck(choose_auto_startup(f,1e-4).bounded_restoration_probe,"within-budget small restoration changed");
 f.blind_violated_row=true;d=choose_auto_startup(f,1e-4);
 ck(!d.central_first&&!d.bounded_restoration_probe,"blind nonlinear model keeps historical restoration");
 f.estimated_dense_restoration_bytes=1024.0L*1024*1024;
 ck(!choose_auto_startup(f,1e-4).central_first,"memory gate silently assumes blind rows are linear-ready");
 f.estimated_dense_restoration_bytes=0;
 f.blind_violated_row=false;
 ck(startup_restoration_probe_seconds(7200)==30,"large allowance cap");
 ck(startup_restoration_probe_seconds(5)==.5,"remaining global fraction");
 ck(startup_restoration_probe_seconds(0)==0&&startup_restoration_probe_seconds(-1)==0,"exhausted budget");
 ck(startup_restoration_probe_seconds(INFINITY)==0&&startup_restoration_probe_seconds(NAN)==0,"nonfinite budget");
 f.infeasible_curved_equality=false;
 f.native_signed_squares=true;ck(!choose_auto_startup(f,1e-4).central_first,"signed protected");f.native_signed_squares=false;
 f.blind_violated_row=true;ck(!choose_auto_startup(f,1e-4).central_first,"blind protected");f.blind_violated_row=false;
 f.curved_inequalities=false;ck(!choose_auto_startup(f,1e-4).central_first,"bound only protected");f.curved_inequalities=true;
 f.affine_nonconstant_objective=false;ck(!choose_auto_startup(f,1e-4).central_first,"curved or constant protected");f.affine_nonconstant_objective=true;
 f.mixed=false;ck(!choose_auto_startup(f,1e-4).central_first,"equality only protected");f.mixed=true;
 f.finite_data=false;ck(!choose_auto_startup(f,1e-4).central_first,"nonfinite protected");f.finite_data=true;
 f.scaled_objective_gradient=0.03;d=choose_auto_startup(f,1e-4);ck(d.central_first&&d.initial_mu==0.03,"scale not always one");
 d=choose_auto_startup(f,0.8);ck(d.initial_mu==0.8,"do not lower the existing adaptive barrier");
 ck(Settings{}.restoration_enabled,"recovery default");ck(Settings{}.majorant_startup_policy=="legacy","legacy default");
}
Settings settings(){Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";
 s.linear_backend="native";s.dca_filter_curvature_backend="native";s.dca_aggregate=false;
 s.dca_aggregate_only=false;s.dca_krylov=false;s.lazy_newton_lazy_constraints=false;
 s.verbose=false;s.live_display_level=2;s.max_outer_iterations=1;s.time_limit_seconds=5;
 s.termination_test="project";s.majorant_startup_policy="auto";return s;}
QCQPProblem problem(){QCQPProblem p;p.name="anonymous";p.n=2;p.x0={1,1};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
 p.objective.linear={1,0};p.objective.hessian.n=2;
 QuadraticFunction h;h.constant=-1;h.linear={0,0};h.hessian.n=2;h.hessian.entries={{0,1,1}};p.equalities.push_back(h);
 QuadraticFunction g;g.constant=-4;g.linear={0,0};g.hessian.n=2;g.hessian.entries={{0,0,2},{1,1,2}};p.inequalities.push_back(g);p.validate();return p;}
struct Capture{std::ostringstream o;std::streambuf* old;Capture():old(std::cout.rdbuf(o.rdbuf())){}~Capture(){std::cout.rdbuf(old);}};
void linked_tests(){auto p=problem();auto s=settings();std::string log;Result r;
 {Capture c;r=solve_filter_dca(p,s);log=c.o.str();}
 ck(log.find("selected=central-first")!=std::string::npos,"linked selector");
 ck(log.find("recovery_enabled=1")!=std::string::npos,"recovery not disabled");
 ck(r.restoration_iterations==0,"no unsolicited initial restoration");
 p.name="arbitrary_other_library_or_renamed_problem";Result rr;
 {Capture c;rr=solve_filter_dca(p,s);}
 ck(r.x==rr.x&&r.objective==rr.objective,"renaming independence");
 {auto infeasible=p;infeasible.x0={0.5,0.5};Capture c;
  auto restored=solve_filter_dca(infeasible,s);
  ck(c.o.str().find("infeasible-nonlinear-equality-restoration-first")!=std::string::npos,"infeasible linked reason");
  ck(restored.restoration_iterations>0,"cold restoration actually executed");}
 {auto infeasible=p;infeasible.x0={0.5,0.5};
  infeasible.equalities[0].constant=1;
  infeasible.equalities[0].hessian.entries={{0,0,2}};
  Capture c;auto result=solve_filter_dca(infeasible,s);
  ck(c.o.str().find("AUTO-STARTUP-RESTORATION-TRANSFER")!=std::string::npos,"incomplete probe never transfers");
  ck(c.o.str().find("private_endpoint_imported=0")!=std::string::npos,"private endpoint imported");
  ck(result.status!=Status::Solved,"infeasible equality certified solved");
 }
 s.restoration_enabled=false;bool rejected=false;
 try{Capture c;solve_filter_dca(p,s);}catch(const std::invalid_argument&){rejected=true;}
 ck(rejected,"reject conflicting global disable");s.restoration_enabled=true;s.lazy_newton_initial_mu=1;rejected=false;
 try{Capture c;solve_filter_dca(p,s);}catch(const std::invalid_argument&){rejected=true;}
 ck(rejected,"reject fixed mu with auto");
}
int main(){try{decision_tests();linked_tests();std::cout<<"PASS auto startup: conservative structure, finite guards, scale, rename, recovery, conflicting options\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
