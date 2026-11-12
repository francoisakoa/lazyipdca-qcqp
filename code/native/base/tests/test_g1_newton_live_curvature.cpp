#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
struct FlushedBuffer:std::stringbuf {int flushes=0;int sync() override {++flushes;return 0;}};
struct Capture {
  FlushedBuffer buffer;std::streambuf* before;
  Capture():before(std::cout.rdbuf(&buffer)){}
  ~Capture(){std::cout.rdbuf(before);}
};
QuadraticFunction q(double c,double linear,double h) {
  QuadraticFunction r;r.constant=c;r.linear={linear};r.hessian.n=1;
  if(h!=0)r.hessian.entries={{0,0,h}};return r;
}
Settings settings(){Settings s;s.outer_algorithm="lazy-newton";s.ln_engine="atomic";s.linear_backend="native";s.verbose=false;return s;}
void signed_inequality_and_no_perturbation() {
  QCQPProblem p;p.n=1;p.x0={2};p.lb={0};p.ub={5};
  p.objective=q(0,0,2);p.inequalities={q(1,0,-1)};
  auto s=settings();Result quiet;
  {Capture c;quiet=Solver(s).solve(p);require(c.buffer.str().empty(),"quiet Newton emits diagnostics");}
  s.live_curvature=true;s.live_iteration_every=3;s.live_iteration_header_every=2;
  Capture capture;auto r=Solver(s).solve(p);
  require(r.status==Status::Solved&&quiet.status==Status::Solved,"signed inequality fixture failed");
  require(r.x==quiet.x&&r.certificate_inequality_multipliers==quiet.certificate_inequality_multipliers,
          "diagnostic mode altered primal/dual trajectory");
  require(r.outer_iterations==quiet.outer_iterations,"diagnostics changed step count");
  require(quiet.lazy_newton.iteration_history.empty(),"quiet mode retained a history");
  const auto& stats=r.lazy_newton;bool negative_ineq=false;int steps=0;
  for(const auto& row:stats.iteration_history) {
    require(row.objective_atoms==0&&row.inequality_atoms==0&&row.objective_correction_rank==0&&row.inequality_correction_rank==0,
            "Newton invented separate DC atoms/ranks");
    if(row.event!="STEP")continue;
    ++steps;require(row.curvature_available,"nonzero step lacks curvature");
    require(std::abs(row.objective_curvature-2.0)<1e-12,"objective Rayleigh curvature is wrong");
    require(row.equality_curvature==0.0,"absent equality has curvature");
    negative_ineq=negative_ineq||row.inequality_curvature<0;
    require(row.active_quadratic_inequalities==1,"quadratic inequality omitted from diagnostic count");
    require(row.inequality_barrier_curvature>=0&&row.bound_barrier_curvature>=0,"barrier contribution lost PSD sign");
    const double sum=row.objective_curvature+row.equality_curvature+row.inequality_curvature+
                     row.inequality_barrier_curvature+row.bound_barrier_curvature+row.regularization_curvature;
    require(std::abs(sum-row.barrier_newton_curvature)<=1e-12*std::max(1.0,std::abs(sum)),"source decomposition does not sum");
  }
  require(negative_ineq&&steps==r.outer_iterations,"signed curvature or accepted steps missing");
  require(stats.iteration_history.front().event=="INIT"&&stats.iteration_history.back().event=="SOLVED","initial/final states missing");
  require(stats.iteration_history.back().inf_du<=1e-6,"final state was not refreshed");
  const std::string text=capture.buffer.str();
  for(const char* token:{"[ITER]","[CURV]","[KKT]","inf_eq","inf_ineq","inf_bnd","rho","rR","rN","aF/rF=0/0","aI/rI=0/0","SOLVED"})
    require(text.find(token)!=std::string::npos,"live output field missing");
  require(capture.buffer.flushes>1,"table was buffered until completion");
}
void global_correction_and_limit() {
  QCQPProblem p;p.n=1;p.x0={0.2};p.lb={-1};p.ub={1};p.objective=q(0,0,-1);
  auto s=settings();s.lazy_newton_atomic_regularization=false;s.live_curvature=true;s.lazy_newton_initial_mu=1e-3;
  Capture capture;auto r=Solver(s).solve(p);bool corrected=false;
  require(r.status==Status::Solved,"concave bounded fixture failed");
  for(const auto& row:r.lazy_newton.iteration_history)if(row.has_direction) {
    require(row.regularization_rank==(row.regularization>0?1:0),"global correction rank does not match reduced dimension");
    if(row.regularization>0){corrected=true;require(std::abs(row.regularization_curvature-row.regularization)<1e-12,"wrong 1-D regularization contribution");}
  }
  require(corrected,"negative tangent curvature fixture did not exercise correction");
  s.max_outer_iterations=1;s.live_iteration_every=100;
  auto limited=Solver(s).solve(p);
  require(limited.lazy_newton.iteration_history.back().event=="STOP","limit did not force final record");
  require(limited.lazy_newton.iteration_history.back().iteration==1,"limit record refers to old point");
  require(std::abs(limited.lazy_newton.iteration_history.back().objective-limited.objective)<1e-12,"limit objective is stale");
}
void signed_equality_normal_curvature() {
  QCQPProblem p;p.n=1;p.x0={2};p.lb={-5};p.ub={5};
  p.objective=q(0,1,0);p.equalities={q(-1,0,2)};
  auto s=settings();s.live_curvature=true;Capture capture;auto r=Solver(s).solve(p);
  require(r.status==Status::Solved,"quadratic equality fixture failed");
  bool negative=false;
  for(const auto& row:r.lazy_newton.iteration_history)if(row.curvature_available) {
    require(row.objective_curvature==0&&row.inequality_curvature==0,"linear objective/absent inequality acquired Hessian curvature");
    require(row.regularization_rank==0,"normal equality curvature was corrected unnecessarily");
    negative=negative||row.equality_curvature<0;
  }
  require(negative,"signed equality contribution missing from directional diagnostics");
}
void ill_conditioned_barrier_condensation() {
  if(!linear_backend_compiled("ldl"))return;
  // Convex Hessian I with a nearly active affine constraint. Its barrier
  // outer product can swallow tangent curvature in a condensed 2x2 matrix.
  QCQPProblem p;p.n=2;p.x0={0,0};p.lb={-INFINITY,-INFINITY};p.ub={INFINITY,INFINITY};
  p.objective.linear={-1e8,-1e8};p.objective.hessian.n=2;p.objective.hessian.entries={{0,0,1},{1,1,1}};
  QuadraticFunction g;g.constant=-1;g.linear={1,1};g.hessian.n=2;p.inequalities={g};
  auto s=settings();s.linear_backend="ldl";s.live_curvature=true;
  Capture capture;auto r=Solver(s).solve(p);
  require(r.status==Status::Solved,"stiff Gram fixture failed");
  require(std::abs(r.x[0]-.5)<1e-6&&std::abs(r.x[1]-.5)<1e-6,"promotion changed the constrained optimum");
  require(r.lazy_newton.condensation_recoveries==1&&r.lazy_newton.promoted_inequalities==1,
          "dangerous affine barrier term was not retained exactly once");
  require(r.lazy_newton.kkt_dimension==3&&r.lazy_newton.structure_builds==1,"persistent reduction lost during border expansion");
  require(r.lazy_newton.symbolic_analyses==2,"promoted layout was repeatedly rebuilt");
  require(r.lazy_newton.regularization_increases==0,"convex tangent acquired a spurious correction");
  require(r.lazy_newton.maximum_direction_residual<=s.kkt_backward_error_reject,"promotion bypassed original direction check");
  require(capture.buffer.str().find("[KKT-CONDENSATION-RECOVERY]")!=std::string::npos,"promotion event missing");
}
}
int main(){try{signed_inequality_and_no_perturbation();global_correction_and_limit();signed_equality_normal_curvature();ill_conditioned_barrier_condensation();
  std::cout<<"PASS: flushed Newton table, signed source curvature, global rank, unchanged trajectory\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
