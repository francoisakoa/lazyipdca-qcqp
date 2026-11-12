#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <stdexcept>
#include <limits>
using namespace lazyipdca::qcqp;
void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
Settings settings(){Settings s;s.dca_aggregate=true;s.dca_aggregate_only=true;
 s.linear_backend=s.dca_aggregate_backend="native";s.lazy_newton_lazy_constraints=false;
 s.dca_rho_complete_search=false;s.dca_strict_finish=false;s.live_display_level=0;
 s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;
 return s;}
void no_majorants(const Result& r){require(r.dca_filter.models==0&&r.dca_filter.inner_steps==0&&r.dca_filter.model_reuses==0,
 "aggregate-only entered a private majorant/inner loop");}
int main(){try{
 const double inf=std::numeric_limits<double>::infinity();
 QCQPProblem p;p.n=2;p.name="single_loop_indefinite";p.lb={-inf,-inf};p.ub={inf,inf};p.x0={1,1};
 p.objective.linear={0,0};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,-50},{1,1,1}};
 auto s=settings();s.max_outer_iterations=1;s.dca_aggregate_rho_max=100;
 auto r=solve_filter_dca(p,s);no_majorants(r);
 require(r.dca_filter.aggregate_acceptances==1,"no certified aggregate step on diagonal problem");
 require(r.dca_filter.aggregate_last_rho==100,"cap was not tried/certified");
 require(r.dca_filter.aggregate_only_inertia_rejections>0,"wrong inertia not rejected");
 require(r.dca_filter.aggregate_preparation_builds==1&&r.dca_filter.aggregate_preparation_hits>0,"preparation not shared across regularizations");
 s.dca_aggregate_rho_max=20;r=solve_filter_dca(p,s);no_majorants(r);
 require(r.status!=Status::Solved&&r.status!=Status::SolvedAcceptable,"false convergence without inertia");
 require(r.x==p.x0&&r.dca_filter.aggregate_acceptances==0,"failed anchor was mutated");
 require(r.dca_filter.aggregate_only_exhaustions==1,"missing explicit search failure");
 p.objective.hessian.entries={{0,0,2},{1,1,4}};p.objective.linear={-1,-2};
 s.max_outer_iterations=50;s.dca_aggregate_rho_max=100;
 r=solve_filter_dca(p,s);no_majorants(r);require(r.status==Status::Solved,"convex QP not solved");
 require(inf_norm(p.objective.gradient(r.x))<=1e-6,"convex QP original stationarity failed");
 s.dca_filter_initial_restoration=true;bool rejected=false;
 try{(void)solve_filter_dca(p,s);}catch(const std::invalid_argument&){rejected=true;}
 require(rejected,"incompatible initial restoration accepted");
 std::cout<<"PASS aggregate-only: cap tried, wrong inertia rejected, no majorants, one-anchor cache, rollback, strict convex solution, incompatible flags\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
