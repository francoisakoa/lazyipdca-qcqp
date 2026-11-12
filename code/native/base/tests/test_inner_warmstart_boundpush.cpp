#include "lazyipdca_qcqp/inner_ipm.hpp"
#include <cmath>
#include <iostream>
using namespace lazyipdca::qcqp;

static ConvexSubproblem make_problem(bool warm, double z0) {
  ConvexSubproblem p;
  p.nvar=1; p.original_dimension=1;
  p.objective.linear={-1.0};
  p.objective.base_hessian.n=1;
  p.objective.base_hessian.entries.push_back({0,0,1.0});
  p.equalities.nvar=1;
  ConvexQuadraticModel g;
  g.constant=0.0;                 // active at y=0
  g.linear={1.0};                 // y <= 0
  g.base_hessian.n=1;
  p.inequalities.push_back(g);
  p.has_dual_warm_start=warm;
  if(warm){
    p.initial_equality_duals={};
    p.initial_inequality_duals={z0};
  }
  return p;
}

int main(){
  Settings st;
  st.inner_backend="native";
  st.linear_backend="native";
  st.inner_dual_warm_start=true;
  st.inner_inequality_dual_warm_start=false;
  st.inner_bound_push=1e-2;
  st.inner_bound_frac=1e-2;
  st.inner_bound_push_reference=1e-6;
  st.inner_bound_mult_init_val=1.0;
  st.max_inner_iterations=50;
  st.inner_tolerance=1e-9;

  ConvexQCQPInteriorPoint ipm(st);

  // Cold starts must retain the exact v0.2.30 initialization contract so the
  // new warm-start machinery cannot move Phase I/restoration/first-SQP paths.
  InnerResult cold=ipm.solve(make_problem(false,0.0));
  if(cold.dual_warm_start_used){ std::cerr<<"cold path marked warm\n"; return 1; }
  if(cold.initialization_bound_pushes!=0){ std::cerr<<"cold path was bound-pushed\n"; return 2; }
  if(std::abs(cold.initial_primal_residual_inf-1.0)>1e-12){
    std::cerr<<"cold v030 contract lost pri0="<<cold.initial_primal_residual_inf<<"\n"; return 3;
  }
  if(std::abs(cold.initial_complementarity-1.0)>1e-12){
    std::cerr<<"cold v030 contract lost mu0="<<cold.initial_complementarity<<"\n"; return 4;
  }

  // Equality-only warm start may reuse lambda, but it must preserve the
  // historical inequality-barrier state exactly: no slack/dual bound push.
  InnerResult eqwarm=ipm.solve(make_problem(true,0.0));
  if(!eqwarm.dual_warm_start_used){ std::cerr<<"equality warm start not used\n"; return 5; }
  if(eqwarm.initialization_bound_pushes!=0){ std::cerr<<"equality warm altered inequality barrier\n"; return 6; }
  if(std::abs(eqwarm.initial_primal_residual_inf-1.0)>1e-12){
    std::cerr<<"equality warm lost cold inequality pri0="<<eqwarm.initial_primal_residual_inf<<"\n"; return 7;
  }
  if(std::abs(eqwarm.initial_complementarity-1.0)>1e-12){
    std::cerr<<"equality warm lost cold inequality mu0="<<eqwarm.initial_complementarity<<"\n"; return 8;
  }

  // Full dual warm start explicitly enables bound push on the inequality pair.
  st.inner_inequality_dual_warm_start=true;
  ConvexQCQPInteriorPoint full_ipm(st);
  InnerResult full=full_ipm.solve(make_problem(true,2.0));
  if(full.initialization_bound_pushes<1){ std::cerr<<"full warm active slack not pushed\n"; return 9; }
  if(!(full.initial_primal_residual_inf>0.0 && full.initial_primal_residual_inf<=1.1e-8)){
    std::cerr<<"unexpected full warm pri0 "<<full.initial_primal_residual_inf<<"\n"; return 10;
  }
  if(!(full.initial_complementarity>0.0 && full.initial_complementarity<1e-5)){
    std::cerr<<"unexpected full warm complementarity "<<full.initial_complementarity<<"\n"; return 11;
  }

  std::cout<<"cold_pri="<<cold.initial_primal_residual_inf
           <<" cold_mu="<<cold.initial_complementarity
           <<" warm_pri="<<eqwarm.initial_primal_residual_inf
           <<" warm_mu="<<eqwarm.initial_complementarity
           <<" full_mu="<<full.initial_complementarity<<"\n";
  return 0;
}
