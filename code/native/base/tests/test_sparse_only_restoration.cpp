#include "lazyipdca_qcqp/restoration.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "../src/failure_recovery_policy.hpp"
#include "../src/mumps_memory_recovery.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cstring>
using namespace lazyipdca::qcqp;
namespace {
int checks=0;
void require(bool condition,const char* message){++checks;if(!condition)throw std::runtime_error(message);}
QCQPProblem fixture(){
 QCQPProblem p;p.name="anonymous_sparse_recovery_fixture";p.n=30000;
 p.x0.assign(p.n,.5);p.lb.assign(p.n,-INFINITY);p.ub.assign(p.n,INFINITY);
 p.objective.compact_linear_dimension=p.n;p.objective.compact_linear={{p.n-1,1.}};p.objective.hessian.n=p.n;
 for(int i=0;i<10000;++i){QuadraticFunction q;q.constant=-1;q.compact_linear_dimension=p.n;q.hessian.n=p.n;q.hessian.entries={{i,i,2.}};p.equalities.push_back(std::move(q));}
 p.validate();return p;
}
Settings settings(){Settings s;s.dca_engine="majorants";s.linear_backend="mumps";s.linear_metis=true;
 s.verbose=false;s.live_display_level=0;s.restoration_failed_linearization_context=true;
 s.restoration_shared_iteration_budget=true;s.restoration_max_iterations=16;
 s.restoration_max_restarts=100;s.restoration_minimax_enabled=true;s.feasibility_tolerance=1e-8;
 s.restoration_deadline_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()+30.;return s;}
}
int main(){
 try{
  auto p=fixture();auto s=settings();const auto frozen=p.x0;
  require(!detail::failure_recovery_budget(p).allowed,"fixture unexpectedly fits historical dense guard");
  const auto b=sparse_restoration_budget(p,s);
  require(b.allowed&&b.jacobian_entries==10000&&b.maximum_factor_workspace_mb==128,"large sparse fixture admission incorrect");
  require(b.estimated_auxiliary_bytes<384.L*1024*1024,"sparse construction envelope exceeded");
  const auto result=FeasibilityRestoration(s).solve_sparse_only(p,p.x0);
  require(result.success&&result.final_feasibility<=s.feasibility_tolerance,"anonymous sparse-only Phase I failed");
  require(result.row_space_solves>0&&result.exact_hessian_solves==0&&result.matrix_free_solves==0,
      "sparse-only mode entered an exact/dense fallback");
  require(result.iterations<=s.restoration_max_iterations&&result.restarts==0&&result.minimax_qp_solves==0,
      "sparse-only mode spent another iteration/restart/minimax budget");
  require(p.x0==frozen,"sparse-only private solve modified the frozen start");
  // A dense shared Jacobian column must remain a one-column border,
  // rather than materializing its 10,000-by-10,000 row-space clique.
  auto bordered=p;
  for(auto& q:bordered.equalities){const int j=q.hessian.entries.front().i;
    q.hessian.entries.clear();q.constant=-2;q.compact_linear={{j,1.},{p.n-1,1.}};}
  const auto bb=sparse_restoration_budget(bordered,s);
  require(bb.allowed&&bb.jacobian_entries==20000&&bb.normal_triplets==30001,
      "high-degree sparse border exceeds structural accounting");
  const auto br=FeasibilityRestoration(s).solve_sparse_only(bordered,bordered.x0);
  require(br.success&&br.row_space_solves>0&&br.exact_hessian_solves==0,
      "high-degree bordered fixture did not use the capped sparse factor");
  auto short_budget=s;short_budget.restoration_max_iterations=1;
  const auto one=FeasibilityRestoration(short_budget).solve_sparse_only(p,p.x0);
  require(one.iterations<=1&&!one.success&&one.restarts==0&&one.exact_hessian_solves==0,
      "failed sparse run fell through to a dense continuation");
  auto expired=s;expired.restoration_deadline_seconds=0;
  const auto late=FeasibilityRestoration(expired).solve_sparse_only(p,p.x0);
  require(!late.success&&late.iterations==0&&late.linear_solves==0&&late.x==frozen,
      "expired sparse-only run accepted a point or allocated a solver");
  for(double nonfinite:{std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}) {
    auto invalid_start=p.x0;invalid_start[1]=nonfinite;
    const auto invalid=FeasibilityRestoration(s).solve_sparse_only(p,invalid_start);
    require(!invalid.success&&invalid.iterations==0&&invalid.linear_solves==0&&
        invalid.x.size()==invalid_start.size()&&std::memcmp(invalid.x.data(),
          invalid_start.data(),invalid_start.size()*sizeof(double))==0,
        "nonfinite unbounded start reported success or changed its rejected private state");
  }
  auto exhausted=s;exhausted.restoration_max_iterations=0;
  const auto empty=FeasibilityRestoration(exhausted).solve_sparse_only(p,p.x0);
  require(!empty.success&&empty.iterations==0&&empty.x==frozen,"exhausted sparse run accepted another stage");
  for(int omitted=0;omitted<4;++omitted){auto bad=s;
   if(omitted==0)bad.linear_backend="native";
   if(omitted==1)bad.linear_metis=false;
   if(omitted==2)bad.restoration_shared_iteration_budget=false;
   if(omitted==3)bad.restoration_failed_linearization_context=false;
   require(!sparse_restoration_budget(p,bad).allowed,"unsupported sparse-only controls admitted");}
  auto overflow=p;overflow.n=std::numeric_limits<int>::max();
  require(!sparse_restoration_budget(overflow,s).allowed,"linear-size overflow guard allocated structural arrays");
  QCQPProblem dense;dense.n=512;dense.x0.assign(512,.5);dense.lb.assign(512,-INFINITY);dense.ub.assign(512,INFINITY);
  dense.objective.compact_linear_dimension=512;dense.objective.hessian.n=512;
  for(int row=0;row<8;++row){QuadraticFunction q;q.compact_linear_dimension=512;q.hessian.n=512;for(int j=0;j<512;++j)q.hessian.entries.push_back({j,j,2.});dense.equalities.push_back(std::move(q));}
  require(!sparse_restoration_budget(dense,s).allowed,"dense Jacobian escaped structural preflight");
  // A structural degree of 64 can become an actual degree of 63 when
  // one derivative cancels. Its 2016-entry local clique must be budgeted,
  // even though its full-degree border would contain only 65 entries.
  QCQPProblem transition;transition.n=512;transition.objective.hessian.n=512;
  for(int j=0;j<64;++j){QuadraticFunction q;q.compact_linear_dimension=512;
    q.compact_linear={{j,1.}};q.hessian.n=512;q.hessian.entries={{511,511,2.}};
    transition.equalities.push_back(std::move(q));}
  const auto tb=sparse_restoration_budget(transition,s);
  require(tb.allowed&&tb.normal_triplets==64+64+2016,
      "structural degree-64 border misses its degree-63 numerical clique");
  auto mixed=p;QuadraticFunction g;g.compact_linear_dimension=p.n;g.hessian.n=p.n;mixed.inequalities.push_back(g);
  require(!sparse_restoration_budget(mixed,s).allowed,"unguarded mixed restoration entered sparse-only route");
  std::string message;
  // An uncapped main KKT can legitimately learn 6400%. Importing it into
  // a capped 128 MB context rejects even this 30k diagonal matrix (-19),
  // while the native 20% margin factors exactly the same matrix.
  detail::reset_mumps_memory_margin_for_test(6400);
  auto isolated=make_memory_bounded_symmetric_factorization(s,message,128);
  std::vector<SymmetricTriplet> small_diagonal;
  for(int j=0;j<p.n;++j)small_diagonal.push_back({j,j,2.});
  require(isolated&&isolated->factor(p.n,p.n,small_diagonal,message),
      "capped Phase-I factor inherited an incompatible uncapped margin");
  require(detail::learned_mumps_memory_margin()==6400,
      "capped context changed the uncapped process policy");
  detail::reset_mumps_memory_margin_for_test();
  int code=-9,margin=20;
  detail::recover_mumps_memory(code,margin,[&](){code=0;},false);
  require(code==0&&margin==40&&detail::learned_mumps_memory_margin()==20,
      "local capped retry exported its margin to main KKT contexts");
  // A changing numerical clique must not accumulate outside the explicit
  // capped context's one-matrix structural budget.
  auto traced=s;traced.live_display_level=1;
  auto changing=make_memory_bounded_symmetric_factorization(traced,message,128);
  std::vector<SymmetricTriplet> first,second,diagonal_only;
  for(int j=0;j<512;++j)diagonal_only.push_back({j,j,2.});
  first=second=diagonal_only;first.push_back({0,1,.1});second.push_back({2,3,.1});
  std::ostringstream captured;auto* old_output=std::cout.rdbuf(captured.rdbuf());
  const bool first_ok=changing->factor(512,512,first,message);
  const bool second_ok=changing->factor(512,512,second,message);
  std::cout.rdbuf(old_output);
  require(first_ok&&second_ok,"changing capped matrix patterns failed");
  const auto logs=captured.str();const std::string count="envelope_entries=513";
  const auto one_count=logs.find(count);
  require(one_count!=std::string::npos&&logs.find(count,one_count+count.size())!=std::string::npos&&
      logs.find("envelope_entries=514")==std::string::npos,
      "capped numerical patterns accumulated a cumulative symbolic envelope");
  require(changing->factor(512,512,diagonal_only,message)&&changing->last_symbolic_reused(),
      "contained capped pattern did not reuse its bounded single-matrix envelope");
  require(!make_memory_bounded_symmetric_factorization(s,message,0),"missing MUMPS workspace cap admitted");
  require(!make_memory_bounded_symmetric_factorization(s,message,129),"overbudget MUMPS workspace cap admitted");
  auto capped=make_memory_bounded_symmetric_factorization(s,message,1);
  require(bool(capped),"bounded MUMPS factory unavailable");
  std::vector<SymmetricTriplet> diagonal;for(int j=0;j<p.n;++j)diagonal.push_back({j,j,2.});
  require(!capped->factor(p.n,p.n,diagonal,message),"1 MB cap did not refuse oversized MUMPS workspace");
  std::cout<<"sparse_only_restoration PASS checks="<<checks<<" anonymous_n=30000 dense_fallback=0 shared_iteration_budget=1 expired_state_rollback=1 hard_MUMPS_cap_checked=1\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
