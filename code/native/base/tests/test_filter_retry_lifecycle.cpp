// Public-API regression test. The reference R4 must fail this test;
// the candidate only releases the installed model before retrying.
#include "lazyipdca_qcqp/dca_filter.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace lazyipdca::qcqp;
static void check(bool b,const std::string& m){if(!b)throw std::runtime_error(m);}
static int count(const std::string& s,const std::string& key){int n=0;std::size_t pos=0;while((pos=s.find(key,pos))!=std::string::npos){++n;pos+=key.size();}return n;}
static void test(const std::string& backend,double x,bool native,bool concave){
 QCQPProblem p;p.name=native?"native_nonlinear_retry":"renamed_nonlinear_retry";p.n=1;p.x0={x};p.lb={-10.};p.ub={10.};p.objective.linear={0};p.objective.hessian.n=1;
 QuadraticFunction h;h.constant=-1;h.linear={0};h.hessian.n=1;
 if(native)h.signed_square={{0,1}};else h.hessian.entries={{0,0,2}};
 p.equalities.push_back(h);
 if(concave){auto q=h;q.constant=-2;q.hessian.entries={{0,0,-2}};q.signed_square.clear();p.inequalities.push_back(q);}
 p.validate();
 Settings s;s.outer_algorithm="lazy-dca";s.dca_engine="majorants";s.linear_backend=backend;s.linear_metis=backend!="native";s.dca_filter_curvature_backend="same";s.dca_aggregate=false;s.dca_aggregate_only=false;s.dca_krylov=false;s.dca_filter_atom_events=false;s.dca_filter_max_atoms=0;s.lazy_newton_atomic_regularization=false;s.lazy_newton_lazy_constraints=false;s.lazy_newton_reduction=false;s.restoration_enabled=false;s.majorant_initial_restoration=false;s.verbose=false;s.live_display_level=2;s.max_outer_iterations=6;s.time_limit_seconds=10;s.line_search_max_steps=1;s.acceptable_iterations=0;s.acceptable_on_failure=false;s.termination_test="project";s.feasibility_tolerance=s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-6;s.kkt_backward_error_reject=1e-7;
 std::ostringstream log;auto* prev=std::cout.rdbuf(log.rdbuf());Result r;
 try {r=solve_filter_dca(p,s);}catch(...){std::cout.rdbuf(prev);throw;}
 std::cout.rdbuf(prev);
 check(count(log.str(),"[FILTER-PROXIMAL-RETRY]")>0,"retry fixture did not reach branch");
 check(r.status!=Status::NumericalError,"NumericalError: "+r.message);
 check(r.status==Status::LineSearchFailure,"rejected directions must remain failures, not successes");
 check(count(log.str(),"[FILTER-PROXIMAL-RETRY]")==4,"expected exactly four bounded retries");
 check(r.lazy_newton.accepted_steps==0,"a rejected direction was committed");
 check(r.x.size()==1&&std::memcmp(r.x.data(),p.x0.data(),sizeof(double))==0,"primal point changed without acceptance");
 check(r.restoration_iterations==0,"disabled restoration was used");
 check(log.str().find("discard the previous") == std::string::npos,"model lifecycle error remains");
 std::cout<<"PASS backend="<<backend<<" x="<<x<<" native="<<native<<" concave="<<concave<<" retries=4 no_accepted_step=1 primal_unchanged=1 terminal=line_search_failure\n";
}
int main(int argc,char** argv){try{const std::string b=argc>1?argv[1]:"native";for(double x:{.01,.1})for(bool native:{false,true})for(bool concave:{false,true})test(b,x,native,concave);return 0;}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
