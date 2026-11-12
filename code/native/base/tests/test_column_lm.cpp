#include "../src/restoration.cpp"
#include <stdexcept>
#include <iostream>
using namespace lazyipdca::qcqp;
namespace {
void check(bool b,const char* why){if(!b)throw std::runtime_error(why);}
QCQPProblem example(){
  QCQPProblem p;p.name="independent_scaled_quadratic_with_redundancy";p.n=4;
  p.x0={0,0,7,3};p.lb=Vec(4,-INFINITY);p.ub=Vec(4,INFINITY);
  p.lb[3]=p.ub[3]=3;p.objective.linear=Vec(4,0);p.objective.hessian.n=4;
  QuadraticFunction a;a.constant=-1;a.linear={1e6,0,0,0};a.hessian.n=4;
  a.hessian.entries={{1,1,2}};
  QuadraticFunction b;b.constant=-.001;b.linear={1,.001,0,0};b.hessian.n=4;
  QuadraticFunction fixed;fixed.constant=-3;fixed.linear={0,0,0,1};fixed.hessian.n=4;
  p.equalities={a,b,a,fixed};p.validate();return p;
}
}
int main(){try{
  // Algebraic equivalence of normal and square-root augmented forms,
  // including redundant rows and a genuinely unused coordinate.
  for(const std::string backend:{"native","mumps","cholesk","ldl"})if(linear_backend_compiled(backend)) {
    Settings ls;ls.linear_backend=backend;ls.linear_metis=true;ls.kkt_adaptive_regularization=true;
    std::string msg;auto f=make_symmetric_factorization(ls,msg);
    std::vector<LegacySparseRow> B={{{0,2},{1,1}},{{0,4},{1,2}}};
    auto matrix=column_lm_augmented_entries(3,B,1e-6);
    check(f->factor(5,3,matrix,msg),"augmented LM factor failed");
    Vec u={0,0,0,-3,-6};check(f->solve(u,msg),"augmented LM solve failed");
    Vec residual(3);for(int j=0;j<3;++j)residual[j]=1e-6*u[j];
    for(int i=0;i<2;++i) {
      double by=3*(i+1);for(auto e:B[i])by+=e.second*u[e.first];
      for(auto e:B[i])residual[e.first]+=e.second*by;
    }
    check(inf_norm(residual)<1e-8&&u[2]==0,"augmented LM does not solve original normal equations");
  }
  auto p=example();Settings s;s.dca_engine="majorants";s.verbose=false;s.live_display_level=0;
  s.restoration_max_iterations=200;s.feasibility_tolerance=1e-6;
  s.linear_metis=true;s.kkt_primal_regularization=1e-3; // must NOT be added to explicit LM
  const auto a=column_lm_admission(p,p.x0,s);
  check(a.allowed&&a.free.size()==3&&a.column_ratio>=1e6,"ill-scaled pure feasibility not admitted");
  for(const std::string backend:{"native","mumps","cholesk","ldl"})if(linear_backend_compiled(backend)) {
    s.linear_backend=backend;
    for(int order=0;order<2;++order) {
      auto q=p;if(order){q.name="unrelated_name";std::reverse(q.equalities.begin(),q.equalities.end());}
      const auto r=FeasibilityRestoration(s).solve(q,q.x0);
      check(r.success&&feasibility(q,r.x).total_inf<=1e-6,"column LM failed original feasibility");
      check(q.x0==p.x0,"supplied point changed");
      check(r.x[2]==7&&r.x[3]==3,"unused/fixed coordinate changed");
      check(r.iterations<=s.restoration_max_iterations,"iteration budget exceeded");
      check(r.message.find("column-scaled LM")!=std::string::npos,"new path not exercised");
    }
  }
  auto bounded=p;bounded.lb[0]=-10;
  check(!column_lm_admission(bounded,bounded.x0,s).allowed,"free finite bound admitted");
  auto curved=p;curved.objective.hessian.entries={{0,0,1}};
  check(!column_lm_admission(curved,curved.x0,s).allowed,"curved objective admitted");
  auto affine=p;affine.objective.linear[0]=1;
  check(!column_lm_admission(affine,affine.x0,s).allowed,"nonconstant objective admitted");
  auto mixed=p;mixed.inequalities.push_back(mixed.equalities.front());
  check(!column_lm_admission(mixed,mixed.x0,s).allowed,"mixed problem admitted");
  // Mixed failed-model restoration: objective curvature is irrelevant to
  // Phase I, while actual finite, one-sided, fixed and absent bounds matter.
  QCQPProblem box;box.name="anonymous_mixed_normal_model";box.n=4;
  box.x0={0,0,7,3};box.lb={0,0,-INFINITY,3};box.ub={2,INFINITY,INFINITY,3};
  box.objective.linear={2,0,0,0};box.objective.hessian.n=4;
  box.objective.hessian.entries={{0,0,2}};
  QuadraticFunction balance;balance.constant=-1;balance.linear={1,1,0,0};balance.hessian.n=4;
  QuadraticFunction demand;demand.constant=.25;demand.linear={-1,0,0,0};demand.hessian.n=4;
  box.equalities={balance};box.inequalities={demand};box.validate();
  auto failure=s;failure.restoration_failed_linearization_context=true;
  failure.restoration_shared_iteration_budget=true;
  check(!column_lm_admission(box,box.x0,s).allowed,"ordinary startup mixed route changed");
  for(const std::string backend:{"native","mumps"})if(linear_backend_compiled(backend)) {
    failure.linear_backend=backend;
    for(int order=0;order<2;++order) {
      auto q=box;if(order){q.name="renamed_and_redundant";q.equalities.push_back(balance);}
      const auto admission=column_lm_admission(q,q.x0,failure);
      check(admission.allowed&&admission.free.size()==3,"mixed normal model not admitted");
      const auto r=column_scaled_lm_restore(q,q.x0,failure,admission);
      check(r.success&&feasibility(q,r.x).total_inf<=failure.feasibility_tolerance,
          "mixed column LM failed independent original feasibility");
      check(q.x0==box.x0&&r.x[2]==7&&r.x[3]==3,"mixed fixed/unused/start coordinates changed");
      check(r.x[0]>=0&&r.x[0]<=2&&r.x[1]>=0,"mixed original bound violated");
      check(r.iterations<=failure.restoration_max_iterations,"mixed recovery budget exceeded");
    }
  }
  // Physical maximum-residual history is independent of LM merit ratios.
  ColumnLmPhysicalStall stalled,contracting,below_floor;
  for(int k=0;k<16;++k) {
    check(!stalled.observe(5e-5,1e-6),"physical stall admitted before full window");
    check(!contracting.observe(5e-5*std::pow(.8,k),1e-6),"contracting physical residual stalled");
    check(!below_floor.observe(5e-7,1e-6),"already feasible physical residual stalled");
  }
  check(stalled.observe(5e-5,1e-6),"unchanged physical residual did not release unused Phase-I budget");
  check(!contracting.observe(5e-5*std::pow(.8,16),1e-6),"strong physical contraction stalled");
  check(!below_floor.observe(5e-7,1e-6),"feasibility floor admitted fallback");
  // A pure feasibility merit is defined on CLOSED original input bounds.
  // Landing on the first side avoids repeatedly taking an IPM interior fraction.
  QCQPProblem side;side.name="closed_original_bound";side.n=2;
  side.x0={0,7};side.lb={-INFINITY,-INFINITY};side.ub={.25,INFINITY};
  side.objective.linear={0,0};side.objective.hessian.n=2;
  QuadraticFunction residual;residual.constant=-1;residual.linear={1,0};residual.hessian.n=2;
  side.equalities={residual};side.validate();
  auto one=failure;one.restoration_max_iterations=1;
  const auto open_step=legacy_local_restore(side,side.x0,one,1);
  const auto closed_step=legacy_local_restore(side,side.x0,one,1,0,true);
  check(open_step.x[0]<side.ub[0]&&closed_step.x[0]==side.ub[0],"restoration did not reach first closed input bound exactly");
  check(closed_step.x[1]==side.x0[1]&&closed_step.iterations<=1,"closed-bound recovery changed unused coordinate or budget");
  check(closed_step.merit<open_step.merit&&closed_step.theta>one.feasibility_tolerance,
      "closed-bound merit improvement changed strict feasibility criterion");
  // Bound activation may reduce nf below the number of nonzero rows.
  // Positive LM damping keeps the row Gram SPD even with redundant rows.
  for(const std::string backend:{"native","mumps"})if(linear_backend_compiled(backend)) {
    Settings row_settings;row_settings.linear_backend=backend;row_settings.linear_metis=true;
    row_settings.kkt_adaptive_regularization=true;
    row_settings.kkt_primal_regularization=0;row_settings.kkt_dual_regularization=0;
    std::string message;auto factor=make_symmetric_factorization(row_settings,message);
    LegacyResidualJacobian redundant;redundant.sparse=true;
    redundant.residual={1,2};redundant.sparse_rows={{{0,1}},{{0,2}}};
    Vec step;
    check(!sparse_row_space_lm_direction(redundant,{0},1e-3,*factor,step,1,message,true,true,true),
        "historical underdetermined admission changed");
    check(sparse_row_space_lm_direction(redundant,{0},1e-3,*factor,step,1,message,true,true,false),
        "positive damped redundant row Gram rejected after bound activation");
    check(step.size()==1&&std::abs(step[0]+2./2.001)<1e-10,
        "overdetermined damped row Gram differs from independent one-column solution");
  }
  auto other=s;other.dca_engine="filter";
  check(!column_lm_admission(p,p.x0,other).allowed,"unrelated engine changed");
  auto expired=s;expired.restoration_deadline_seconds=0;
  auto r=column_scaled_lm_restore(p,p.x0,expired,a);
  check(!r.success&&r.iterations==0&&r.x==p.x0,"expired restoration claimed success or moved point");
  auto limited=s;limited.restoration_max_iterations=1;
  r=column_scaled_lm_restore(p,p.x0,limited,a);
  check(!r.success&&r.iterations<=1,"one-step budget was ignored");
  std::cout<<"PASS column LM: original feasibility, redundant/zero rows, unused/fixed values, names/order, all compiled backends, no double shift, deadline and budget\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
