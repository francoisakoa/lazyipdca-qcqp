#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/lazy_newton.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool b,const char* text){if(!b)throw std::runtime_error(text);}
struct Capture {std::ostringstream output;std::streambuf* before;Capture():before(std::cout.rdbuf(output.rdbuf())){}~Capture(){std::cout.rdbuf(before);}};
QuadraticFunction q(int n,double c,Vec linear,std::vector<SymTriplet> h) {
  QuadraticFunction a;a.constant=c;a.linear=linear;a.hessian.n=n;a.hessian.entries=h;return a;
}
Settings settings(const char* backend) {Settings s;s.outer_algorithm="lazy-newton";s.ln_engine="atomic";s.linear_backend=backend;s.verbose=false;
 s.live_curvature=true;s.lazy_newton_initial_mu=1e-3;s.max_outer_iterations=150;return s;}
void rotated_rank_one(const char* backend) {
  QCQPProblem p;p.n=2;p.x0={.2,.2};p.lb={-1,-1};p.ub={1,1};p.objective=q(2,0,{0,0},{{0,1,-1}});
  auto s=settings(backend);Capture capture;auto r=Solver(s).solve(p);
  if(r.status!=Status::Solved)throw std::runtime_error(r.message);
  require(certify_lazy_newton_original(p,s,r),"rotated atom original certificate");
  require(std::abs(r.x[0]-1)<1e-4&&std::abs(r.x[1]-1)<1e-4,"rotated concave optimum");
  require(r.lazy_newton.maximum_regularization==0&&r.lazy_newton.final_regularization==0,"atomic mode used uniform rho");
  require(r.lazy_newton.objective_atoms==1&&r.lazy_newton.atom_source_analyses==1,"rank-one spectrum was rebuilt or extra source atoms created");
  require(r.lazy_newton.atom_spectral_blocks==1&&r.lazy_newton.atom_peak_active==1,"non-axis atom lost rank one");
  bool corrected=false,unshifted=false;
  for(const auto& row:r.lazy_newton.iteration_history)if(row.has_direction) {
    if(row.atomic_active) {
      corrected=true;require(row.objective_atoms==1&&row.regularization_rank==1&&row.atom_ranks_exact,"wrong active atomic rank");
      require(std::abs(row.regularization_curvature-row.atomic_trace)<1e-8,"atom does not act along the negative rotated direction");
      require(row.regularization_curvature>0&&row.regularization==0,"correction was omitted from expanded direction");
    } else unshifted=true;
  }
  require(corrected&&(unshifted||r.lazy_newton.atom_final_trace<0.05*r.lazy_newton.atom_maximum_trace),"atom did not diminish toward Newton");
  require(r.lazy_newton.maximum_direction_residual<=s.kkt_backward_error_reject,"atomic original-direction audit bypassed");
  require(capture.output.str().find("[ATOMS-ADD]")!=std::string::npos,"atom event not flushed/displayed");
}
void no_atom_fast_path(const char* backend) {
  QCQPProblem p;p.n=2;p.x0={.2,.3};p.lb={-1,-1};p.ub={1,1};
  p.objective=q(2,0,{-.1,-.2},{{0,0,2},{0,1,.5},{1,1,2}});
  auto s=settings(backend);s.live_curvature=false;auto r=Solver(s).solve(p);
  s.lazy_newton_atomic_regularization=false;auto control=Solver(s).solve(p);
  require(r.status==Status::Solved&&control.status==Status::Solved,"convex comparison failed");
  require(r.x==control.x&&r.certificate_lower_multipliers==control.certificate_lower_multipliers&&
      r.certificate_upper_multipliers==control.certificate_upper_multipliers,"empty atomic path perturbed state");
  require(r.lazy_newton.atom_creations==0&&r.lazy_newton.atom_source_analyses==0&&
      r.lazy_newton.atom_analysis_seconds==0&&r.lazy_newton.atom_update_seconds==0,"empty path did atomic work");
  require(r.lazy_newton.factor_attempts==control.lazy_newton.factor_attempts,"empty atomic path added factors");
}
void equality_source(const char* backend) {
  QCQPProblem p;p.n=2;p.x0={.8,.6};p.lb={-2,-2};p.ub={2,2};
  p.objective=q(2,0,{1,0},{});p.equalities={q(2,-1,{0,0},{{0,0,2},{1,1,2}})};
  auto s=settings(backend);Capture capture;auto r=Solver(s).solve(p);
  if(r.status!=Status::Solved)throw std::runtime_error("equality fixture: "+r.message);
  require(r.x[0]<-.9999,"atomic circle failed to reach the minimum");
  require(certify_lazy_newton_original(p,s,r),"signed equality atom KKT failed");
  require(r.lazy_newton.equality_atoms>0,"signed equality fixture did not create source atoms");
  require(r.lazy_newton.maximum_regularization==0,"equality used uniform rho");
}
void bounded_large_block(const char* backend) {
  QCQPProblem p;p.n=20;p.x0=Vec(p.n,.2);p.lb=Vec(p.n,0);p.ub=Vec(p.n,1);
  std::vector<SymTriplet> h;for(int i=0;i<p.n;++i){h.push_back({i,i,1});if(i)h.push_back({i-1,i,-2});}
  p.objective=q(p.n,0,Vec(p.n,0),h);auto s=settings(backend);Capture capture;auto r=Solver(s).solve(p);
  if(r.status!=Status::Solved)throw std::runtime_error("large block fixture: "+r.message);
  require(r.lazy_newton.atom_large_blocks==1&&r.lazy_newton.atom_spectral_blocks==0,"large source triggered a dense spectrum");
  require(r.lazy_newton.objective_atoms>0&&r.lazy_newton.maximum_regularization==0,"large component was not corrected atomically");
  require(certify_lazy_newton_original(p,s,r),"large-block original KKT failed");
}
void changing_reduced_atom(const char* backend) {
  QCQPProblem p;p.n=2;p.x0={.4,.16};p.lb={.1,0};p.ub={2,1};
  p.objective=q(2,0,{0,0},{{1,1,-1}});p.equalities={q(2,0,{0,1},{{0,0,-2}})};
  auto s=settings(backend);Capture capture;auto r=Solver(s).solve(p);
  if(r.status!=Status::Solved)throw std::runtime_error("changing reduced atom: "+r.message);
  require(r.lazy_newton.eliminated_variables==1&&r.lazy_newton.objective_atoms>0,"changing projection was not exercised");
  require(std::abs(r.x[0]-1)<1e-4&&std::abs(r.x[1]-1)<1e-4,"wrong lifted concave optimum");
  require(r.lazy_newton.maximum_direction_residual<=s.kkt_backward_error_reject,"changing projection lost expanded direction accuracy");
  require(certify_lazy_newton_original(p,s,r),"changing reduced atom original certificate failed");
}
void null_curvature_completion(const char* backend) {
  QCQPProblem p;p.n=1;p.x0={0};p.lb={-INFINITY};p.ub={INFINITY};p.objective=q(1,0,{1},{});
  auto s=settings(backend);s.max_outer_iterations=1;Capture capture;auto r=Solver(s).solve(p);
  require(r.status==Status::MaxOuterIterations,"unbounded affine objective falsely certified or no regularized step");
  require(r.lazy_newton.completion_atoms==1&&r.lazy_newton.atom_source_analyses==0,"null curvature completion was not separate");
  require(r.lazy_newton.maximum_regularization==0,"completion reverted to uniform rho");
  require(!certify_lazy_newton_original(p,s,r),"regularization concealed original nonstationarity");
}
void qp_energy_support(const char* backend) {
  QCQPProblem p;p.n=2;p.x0={.2,.2};p.lb={-1,-1};p.ub={1,1};
  p.objective=q(2,0,{0,0},{{0,0,-1},{0,1,1e-4},{1,1,1}});
  auto s=settings(backend);s.lazy_newton_atom_energy_sparsify=true;Capture capture;auto r=Solver(s).solve(p);
  if(r.status!=Status::Solved)throw std::runtime_error("energy support: "+r.message);
  require(r.lazy_newton.atom_sparsifications==1&&r.lazy_newton.atom_dropped_coordinates==1,"weak eigenvector tail was retained");
  require(r.lazy_newton.objective_atoms==1,"energy support created redundant modes");
  require(certify_lazy_newton_original(p,s,r),"sparse source Rayleigh correction failed original KKT");
}
void qp_compensated_direction(const char* backend) {
  QCQPProblem p;p.n=2;p.x0={0,.2};p.lb={-10,-10};p.ub={10,10};
  p.objective=q(2,0,{0,0},{{0,0,-5},{1,1,-1}});
  p.inequalities={q(2,-.1,{0,0},{{0,0,6}})};
  auto s=settings(backend);s.lazy_newton_initial_mu=1;s.max_outer_iterations=1;s.lazy_newton_atom_rayleigh_gate=true;s.lazy_newton_atom_guided=false;
  Capture capture;auto r=Solver(s).solve(p);
  require(r.status==Status::MaxOuterIterations,"compensated gate failed to compute a valid step");
  require(r.lazy_newton.objective_atoms==1&&r.lazy_newton.atom_covered_rejections>0,"compensated source was still regularized");
  require(r.lazy_newton.atom_coverage_observations>0&&r.lazy_newton.atom_last_probe_coverage>=0&&
      r.lazy_newton.atom_last_probe_coverage<=1,"invalid scoped coverage diagnostic");
  require(r.lazy_newton.maximum_direction_residual<=s.kkt_backward_error_reject,"compensated gate bypassed full Newton audit");
}
void inequality_source(const char* backend) {
  QCQPProblem p;p.n=2;p.x0={.1,.2};p.lb={-2,-2};p.ub={2,2};
  p.objective=q(2,0,{-.1,-.2},{{0,0,.1},{1,1,.1}});
  p.inequalities={q(2,1,{0,0},{{0,0,-2},{1,1,-2}})};
  auto s=settings(backend);Capture capture;auto r=Solver(s).solve(p);
  if(r.status!=Status::Solved)throw std::runtime_error("inequality fixture: "+r.message);
  require(r.lazy_newton.inequality_atoms>0,"negative inequality source not attributed");
  require(r.lazy_newton.objective_atoms==0,"convex objective was assigned negative atoms");
  require(r.lazy_newton.maximum_regularization==0,"inequality atom used uniform rho");
  require(certify_lazy_newton_original(p,s,r),"inequality atom final KKT failed");
}
}
int main(){try{for(const char* backend:{"native","ldl"})if(linear_backend_compiled(backend)){
 rotated_rank_one(backend);no_atom_fast_path(backend);inequality_source(backend);equality_source(backend);bounded_large_block(backend);changing_reduced_atom(backend);null_curvature_completion(backend);qp_energy_support(backend);qp_compensated_direction(backend);}
 std::cout<<"PASS atomic rank-one, sources, retirement, original KKT and empty fast path\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
