#include "lazyipdca_qcqp/restoration.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
int main(){try{
  QCQPProblem p;p.name="incompatible_tangent_then_negative_merit_curvature";p.n=3;p.x0={0,0,0};
  p.lb=Vec(3,-INFINITY);p.ub=Vec(3,INFINITY);p.objective.linear={-1,2,-1};
  p.objective.hessian.n=3;p.objective.hessian.entries={{0,0,2},{1,1,2},{2,2,2}};
  QuadraticFunction a;a.constant=-5;a.linear={2,0,0};a.hessian.n=3;a.hessian.entries={{1,1,-2}};
  QuadraticFunction b;b.constant=-9;b.linear={3,0,0};b.hessian.n=3;b.hessian.entries={{2,2,-2}};
  p.equalities={a,b};
  Settings s;s.verbose=false;s.live_display_level=0;s.restoration_max_restarts=0;
  s.restoration_minimax_enabled=false;s.restoration_max_iterations=80;
  s.restoration_failed_linearization_context=true;
  const Vec frozen=p.x0;
  for(int variant=0;variant<2;++variant){
    if(variant){p.name="a_completely_different_name";std::swap(p.equalities[0],p.equalities[1]);}
    auto r=FeasibilityRestoration(s).solve(p,p.x0);
    require(r.success && feasibility(p,r.x).total_inf<=s.feasibility_tolerance,"stationary infeasible merit was not escaped");
    require(p.x0==frozen,"frozen starting point modified");
  }
  // A violated quadratic equality has zero first derivative at x0,
  // while an ordinary inequality and a nonconstant objective are present.
  // The failure-context restoration must activate curvature without names.
  QCQPProblem mixed;mixed.name="anonymous_mixed_stationary_merit";mixed.n=3;
  mixed.x0={0,0,0};mixed.lb=Vec(3,-INFINITY);mixed.ub=Vec(3,INFINITY);
  mixed.objective.linear={0,0,1};mixed.objective.hessian.n=3;
  QuadraticFunction sphere;sphere.constant=-1;sphere.linear=Vec(3,0);sphere.hessian.n=3;
  sphere.hessian.entries={{0,0,2},{1,1,2}};mixed.equalities={sphere};
  QuadraticFunction cap;cap.constant=-2;cap.linear={0,0,1};cap.hessian.n=3;mixed.inequalities={cap};
  auto mixed_result=FeasibilityRestoration(s).solve(mixed,mixed.x0);
  require(mixed_result.success && feasibility(mixed,mixed_result.x).total_inf<=s.feasibility_tolerance,
      "mixed zero-gradient violated equality was not restored");
  require(mixed.x0==Vec({0,0,0}),"mixed frozen starting point modified");
  auto coarse=s;coarse.feasibility_tolerance=.1;
  coarse.restoration_shared_iteration_budget=true;coarse.restoration_max_iterations=12;
  auto coarse_result=FeasibilityRestoration(coarse).solve(mixed,mixed.x0);
  require(coarse_result.success&&coarse_result.final_feasibility<=.1,
      "coarse target incorrectly excluded stationary infeasible merit");
  require(coarse_result.iterations<=12&&mixed.x0==Vec({0,0,0}),
      "coarse phase-I exceeded shared budget or modified frozen start");
  // Failure recovery must not depend on a small arbitrary dimension cutoff.
  for(int n:{401,2070}) {
    QCQPProblem padded=mixed;padded.name="dimension_not_instance_name";padded.n=n;
    padded.x0.resize(n,0);padded.lb.resize(n,-INFINITY);padded.ub.resize(n,INFINITY);
    padded.objective.linear.resize(n,0);padded.objective.hessian.n=n;
    for(auto& q:padded.equalities){q.linear.resize(n,0);q.hessian.n=n;}
    for(auto& q:padded.inequalities){q.linear.resize(n,0);q.hessian.n=n;}
    if(n==2070)padded.inequalities.clear();
    padded.validate();
    auto q=FeasibilityRestoration(s).solve(padded,padded.x0);
    require(q.success&&feasibility(padded,q.x).total_inf<=s.feasibility_tolerance,
      "dimension-budgeted stationary merit not escaped");
    require(padded.x0==Vec(n,0),"padded frozen start modified");
  }
  for(int equations:{2,3}) {
    QCQPProblem q;q.name="square_or_redundant_stationary_merit";q.n=2;
    q.x0={0,0};q.lb=Vec(2,-INFINITY);q.ub=Vec(2,INFINITY);
    q.objective.linear={1,0};q.objective.hessian.n=2;
    for(int i=0;i<equations;++i){QuadraticFunction c;c.constant=-1;c.linear=Vec(2,0);
      c.hessian.n=2;c.hessian.entries={{i%2,i%2,2}};q.equalities.push_back(c);}
    auto r=FeasibilityRestoration(s).solve(q,q.x0);
    require(r.success&&feasibility(q,r.x).total_inf<=s.feasibility_tolerance,
      "square/overdetermined stationary merit not escaped");
    require(q.x0==Vec(2,0),"square frozen start modified");
  }
  s.restoration_deadline_seconds=0;
  auto r=FeasibilityRestoration(s).solve(p,p.x0);
  require(!r.success,"expired restoration claimed success");
  std::cout<<"failed linearization: later saddle escape, renamed/reordered equations, frozen start and deadline PASS\n";
  return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
