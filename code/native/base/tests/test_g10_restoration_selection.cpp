#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(){try{
 QCQPProblem p;p.n=3;p.lb=Vec(3,-2);p.ub=Vec(3,2);p.x0={0.2,0.3,0.4};
 p.objective.linear=Vec(3,0);p.objective.hessian.n=3;p.objective.hessian.entries={{0,0,-2},{1,1,-4}};
 QuadraticFunction q;q.linear={0,0,1};q.hessian.n=3;q.hessian.entries={{0,0,-2}};p.equalities.push_back(q);
 Settings s;s.linear_backend="native";s.dca_engine="filter";s.live_display_level=1;s.max_outer_iterations=1;
 s.dca_kkt_growth_max=1e12; // this historical selection test isolates memory-independence
 s.dca_aggregate=true;s.dca_aggregate_backend="native";s.dca_aggregate_rho_max=0;
 std::ostringstream trace;auto* old=std::cout.rdbuf(trace.rdbuf());
 auto a=solve_filter_dca(p,s);s.mumps_memory_recovery=false;auto b=solve_filter_dca(p,s);
 std::cout.rdbuf(old);
 if(a.x!=b.x || a.certificate_equality_multipliers!=b.certificate_equality_multipliers || a.status!=b.status)
   throw std::runtime_error("selection depends on memory recovery setting");
 if(trace.str().find("policy=compare-original-kkt memory_independent=1")==std::string::npos ||
    trace.str().find("decision=hold kind=majorant")==std::string::npos ||
    trace.str().find("decision=commit kind=restoration")==std::string::npos ||
    a.outer_iterations!=1 || b.outer_iterations!=1)
   throw std::runtime_error("selection path was not exercised");
 std::cout<<trace.str()<<"Selection exercised from an infeasible state after aggregate inertia rejection; independent of allocation setting\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
