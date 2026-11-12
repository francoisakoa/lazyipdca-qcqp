#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
QuadraticFunction q(int n){QuadraticFunction f;f.hessian.n=n;f.compact_linear_dimension=n;return f;}
QuadraticFunction balance(int n,int u,int v,int a,int b,double r,double t){
  auto f=q(n);auto add=[&](int i,int j,double val){if(val!=0)f.hessian.entries.push_back({std::min(i,j),std::max(i,j),val});};
  add(u,u,2*r);add(v,v,2*r);add(u,a,-r);add(v,b,-r);add(v,a,t);add(u,b,-t);
  std::sort(f.hessian.entries.begin(),f.hessian.entries.end(),[](auto x,auto y){return std::pair{x.i,x.j}<std::pair{y.i,y.j};});return f;
}
void radial(QCQPProblem& p,int u,int v){auto f=q(p.n);f.constant=.25;f.hessian.entries={{u,u,-2},{v,v,-2}};p.inequalities.push_back(f);f.constant=-4;f.hessian.entries={{u,u,2},{v,v,2}};p.inequalities.push_back(f);}
QCQPProblem model(){
  QCQPProblem p;p.name="coefficient-leaf";p.n=4;p.x0={1,0,1,0};p.lb=Vec(4,-2);p.ub=Vec(4,2);
  p.objective=q(4);p.objective.constant=2.5;p.objective.compact_linear={{0,-2},{3,-1}};
  for(int j=0;j<4;++j)p.objective.hessian.entries.push_back({j,j,1});
  p.equalities={balance(4,2,3,0,1,2,3),balance(4,2,3,0,1,3,-2)};
  radial(p,0,1);radial(p,2,3);p.validate();return p;
}
Result result(const QCQPProblem& p,const Vec& x){
  Result r;r.x=x;r.certificate_equality_multipliers.assign(p.equalities.size(),0);r.certificate_inequality_multipliers.assign(p.inequalities.size(),0);
  r.certificate_lower_multipliers.assign(p.n,0);r.certificate_upper_multipliers.assign(p.n,0);return r;
}
int main(){try{
  auto p=model();ConstraintPresolve reduction(p);const auto& small=reduction.reduced();
  check(small.n==2&&small.equalities.empty(),"proved leaf not eliminated");
  check(small.inequalities.size()==2&&reduction.statistics().duplicate_inequalities==2,"duplicate radial bounds not merged");
  check(reduction.statistics().duplicate_bounds==4,"box sides not merged");
  auto r=result(small,{1,.5});reduction.lift(r);Settings st;
  check(detail::audit_original_termination(p,st,r),"original strict certificate after adjoint lift failed");
  check(inf_norm(r.certificate_equality_multipliers)>0,"multipliers not reconstructed");
  check(r.x==Vec({1,.5,1,.5}),"bad variable reconstruction");
  // Every lifted point lies exactly on the eliminated polynomial equations.
  for(Vec z: {Vec{.8,.2},Vec{1.3,-.7}}){auto t=result(small,z);reduction.lift(t);
    for(const auto& h:p.equalities)check(std::abs(h.value(t.x))<2e-14,"polynomial identity failed");}
  auto not_nonzero=model();not_nonzero.inequalities.clear();ConstraintPresolve no_nz(not_nonzero);
  check(no_nz.statistics().leaf_pairs==0,"zero-voltage branch silently discarded");
  auto different=model();different.x0[2]+=1e-12;ConstraintPresolve no_start(different);
  check(no_start.statistics().leaf_pairs==0&&no_start.statistics().start_rejected==1,"cold start silently projected");
  auto zero_start=model();zero_start.x0=Vec(4,0);ConstraintPresolve no_zero_start(zero_start);
  check(no_zero_start.statistics().leaf_pairs==0&&no_zero_start.statistics().start_rejected==1,"zero-voltage initial adjoint chart accepted");
  auto almost=model();almost.equalities[0].hessian.entries[0].v=std::nextafter(almost.equalities[0].hessian.entries[0].v,INFINITY);
  ConstraintPresolve no_approx(almost);check(no_approx.statistics().leaf_pairs==0,"almost-equal coefficients treated as exact");
  auto no_copy=p;no_copy.objective.compact_linear={{0,1.0},{2,1e-100}};ConstraintPresolve no_round(no_copy);
  check(no_round.statistics().leaf_pairs==0&&no_round.statistics().arithmetic_rejected==1,"inexact coefficient coalescing accepted");
  // An upper radius differing from a box endpoint at the last bit is retained.
  QCQPProblem scalar;scalar.n=2;scalar.name="not-implied-at-last-bit";scalar.x0={1,0};scalar.lb={.9,0};scalar.ub={1.1,0};scalar.objective=q(2);
  auto rad=q(2);rad.constant=-std::nextafter(1.21,INFINITY);rad.hessian.entries={{0,0,2},{1,1,2}};scalar.inequalities={rad};
  ConstraintPresolve no_box(scalar);check(no_box.reduced().inequalities.size()==1,"nearly redundant box constraint dropped");
  check(std::isinf(no_box.reduced().ub[0])&&no_box.reduced().lb[0]==.9,"quadratic-implied upper bound not removed");
  check(no_box.reduced().lb[1]==0&&no_box.reduced().ub[1]==0,"fixed variable changed");
  auto loose=scalar;loose.inequalities[0].constant=-std::nextafter(std::nextafter(1.21,INFINITY),INFINITY);
  ConstraintPresolve keep_box(loose);check(keep_box.reduced().ub[0]==1.1,"last-bit-stronger box discarded");
  auto tight=scalar;tight.ub[0]=1.05;ConstraintPresolve keep_tight(tight);
  check(keep_tight.reduced().ub[0]==1.05,"nonredundant box discarded");
  ConstraintPresolve box_off(scalar,false);check(box_off.reduced().ub[0]==1.1,"bound option not respected");
  auto cross=scalar;cross.inequalities[0].hessian.entries.push_back({0,1,.01});ConstraintPresolve keep_cross(cross);
  check(keep_cross.reduced().ub[0]==1.1,"cross-term bound implication assumed");
  // Equality-implied sides are opt-in. Retain the radial equality and exact
  // initial point while omitting only sides proved by h=0 (or its negative).
  // An equality-fixed boundary has no positive-slack interior. Prove and
  // omit only redundant sides, without projecting even an infeasible x0.
  QCQPProblem aff;aff.n=2;aff.name="affine-fixed-boundary";
  aff.x0={.25,2};aff.lb={0,-INFINITY};aff.ub={3,INFINITY};aff.objective=q(2);
  auto fixed_row=q(2);fixed_row.compact_linear={{0,2}};aff.equalities={fixed_row};aff.validate();
  ConstraintPresolve aff_legacy(aff),aff_proof(aff,false,false,false,true);
  check(aff_legacy.reduced().lb[0]==0&&aff_legacy.reduced().ub[0]==3,
      "legacy constructor changed affine sides");
  check(aff_proof.statistics().implied_bound_sides==2&&
      std::isinf(aff_proof.reduced().lb[0])&&std::isinf(aff_proof.reduced().ub[0])&&
      aff_proof.reduced().equalities.size()==1&&aff_proof.reduced().x0==aff.x0,
      "exact singleton proof or unchanged x0 contract failed");
  check(aff_proof.omitted_bound_residual({-1,2}).maximum==1,
      "removed affine-equality bound was lost from original residual");
  auto contradictory=aff;contradictory.lb[0]=1;
  ConstraintPresolve contradiction(contradictory,false,false,false,true);
  check(contradiction.reduced().lb[0]==1,
      "contradictory bound wrongly removed");
  auto inexact=aff;inexact.equalities[0].compact_linear={{0,3}};inexact.equalities[0].constant=-1;
  ConstraintPresolve not_dyadic(inexact,false,false,false,true);
  check(not_dyadic.statistics().implied_bound_sides==0,
      "inexact affine root wrongly certified");
  auto curved=aff;curved.equalities[0].hessian.entries={{0,0,2}};
  ConstraintPresolve not_affine(curved,false,false,false,true);
  check(not_affine.statistics().implied_bound_sides==0,
      "quadratic equality treated as affine");
  auto coupled=aff;coupled.equalities[0].compact_linear.push_back({1,1});
  ConstraintPresolve not_singleton(coupled,false,false,false,true);
  check(not_singleton.statistics().implied_bound_sides==0,
      "coupled affine equality treated as singleton");
  QCQPProblem circle;circle.n=2;circle.name="equality-unit-circle";
  circle.x0={1,0};circle.lb={-1,-1};circle.ub={1,1};circle.objective=q(2);
  auto unit=q(2);unit.constant=-1;unit.hessian.entries={{0,0,2},{1,1,2}};
  circle.equalities={unit};circle.validate();
  ConstraintPresolve eq_default(circle),eq_two_args(circle,false),eq_bounds(circle,false,true);
  check(eq_default.reduced().lb==circle.lb&&eq_default.reduced().ub==circle.ub&&
      !eq_default.has_omitted_equality_bounds()&&eq_two_args.reduced().ub==circle.ub,
      "existing constructor defaults removed equality-implied sides");
  check(eq_bounds.has_omitted_equality_bounds()&&eq_bounds.statistics().implied_bound_sides==4&&
      std::isinf(eq_bounds.reduced().lb[0])&&std::isinf(eq_bounds.reduced().lb[1])&&
      std::isinf(eq_bounds.reduced().ub[0])&&std::isinf(eq_bounds.reduced().ub[1]),
      "unit-circle equality did not prove all four box sides");
  check(eq_bounds.reduced().equalities.size()==1&&eq_bounds.reduced().equalities[0].constant==-1&&
      eq_bounds.reduced().equalities[0].hessian.entries[0].v==2&&
      eq_bounds.reduced().x0==circle.x0&&circle.equalities[0].constant==-1,
      "equality-bound proof changed the retained equation or start");
  const auto inside_residual=eq_bounds.omitted_bound_residual({1,0});
  const auto outside_residual=eq_bounds.omitted_bound_residual({1.25,-1.5});
  check(inside_residual.sides==4&&inside_residual.maximum==0&&inside_residual.sum==0&&
      outside_residual.sides==4&&outside_residual.maximum==.5&&outside_residual.sum==.75,
      "omitted equality-bound physical residual is incorrect");
  bool bad_dimension=false,bad_value=false;
  try{(void)eq_bounds.omitted_bound_residual({1});}catch(const std::invalid_argument&){bad_dimension=true;}
  try{(void)eq_bounds.omitted_bound_residual({1,NAN});}catch(const std::invalid_argument&){bad_value=true;}
  check(bad_dimension&&bad_value,"invalid omitted-bound point accepted");
  auto negative_circle=circle;negative_circle.equalities[0].constant=1;
  for(auto& entry:negative_circle.equalities[0].hessian.entries)entry.v=-entry.v;
  ConstraintPresolve negative_eq(negative_circle,false,true);
  check(negative_eq.statistics().implied_bound_sides==4&&
      negative_eq.reduced().equalities[0].constant==1&&
      negative_eq.reduced().equalities[0].hessian.entries[0].v==-2,
      "opposite equality sign was not proved without rewriting the equation");
  auto last_inside=circle;last_inside.ub[0]=std::nextafter(1.,0.);
  last_inside.lb[0]=std::nextafter(-1.,0.);
  ConstraintPresolve keep_last_inside(last_inside,false,true);
  check(keep_last_inside.reduced().ub[0]==last_inside.ub[0]&&
      keep_last_inside.reduced().lb[0]==last_inside.lb[0],
      "last-bit stronger equality box side was discarded");
  auto last_outside=circle;last_outside.ub[0]=std::nextafter(1.,INFINITY);
  last_outside.lb[0]=std::nextafter(-1.,-INFINITY);
  ConstraintPresolve drop_last_outside(last_outside,false,true);
  check(std::isinf(drop_last_outside.reduced().ub[0])&&std::isinf(drop_last_outside.reduced().lb[0]),
      "last-bit weaker equality box side was retained");
  auto fixed_circle=circle;fixed_circle.lb[1]=fixed_circle.ub[1]=0;
  ConstraintPresolve keep_eq_fixed(fixed_circle,false,true);
  check(keep_eq_fixed.reduced().lb[1]==0&&keep_eq_fixed.reduced().ub[1]==0&&
      keep_eq_fixed.statistics().implied_bound_sides==2,"equality proof removed a fixed side");
  auto shifted_circle=circle;shifted_circle.equalities[0].compact_linear={{0,1e-100}};
  auto cross_circle=circle;cross_circle.equalities[0].hessian.entries={{0,0,2},{0,1,1e-100},{1,1,2}};
  auto mixed_circle=circle;mixed_circle.equalities[0].hessian.entries[1].v=-2;
  auto same_sign_circle=circle;same_sign_circle.equalities[0].constant=1;
  for(const auto* ineligible:{&shifted_circle,&cross_circle,&mixed_circle,&same_sign_circle}) {
    ConstraintPresolve keep_ineligible(*ineligible,false,true);
    check(!keep_ineligible.has_omitted_equality_bounds()&&keep_ineligible.reduced().lb==ineligible->lb&&
        keep_ineligible.reduced().ub==ineligible->ub,"ineligible radial equality removed bounds");
  }
  auto duplicate_diagonal=circle;
  duplicate_diagonal.equalities[0].hessian.entries={{0,0,1},{0,0,1},{1,1,2}};
  ConstraintPresolve coalesced_eq(duplicate_diagonal,false,true);
  check(coalesced_eq.statistics().implied_bound_sides==4,
      "duplicate diagonal equality coefficients were not exactly coalesced");
  auto guarded_circle=circle;guarded_circle.equalities[0].constant=-1e100;
  guarded_circle.equalities[0].hessian.entries={{0,0,2e100},{1,1,2e100}};
  ConstraintPresolve guard_eq_arithmetic(guarded_circle,false,true);
  check(!guard_eq_arithmetic.has_omitted_equality_bounds(),"unsupported exact-proof exponents removed sides");
  auto inequality_first=circle;inequality_first.inequalities={unit};
  ConstraintPresolve ineq_only_residual(inequality_first,true,true);
  const auto no_equality_sides=ineq_only_residual.omitted_bound_residual({2,2});
  check(!ineq_only_residual.has_omitted_equality_bounds()&&no_equality_sides.sides==0&&
      no_equality_sides.maximum==0&&no_equality_sides.sum==0,
      "equality omitted-bound residual included historical inequality-removed sides");
  auto strongest=model();strongest.lb[2]=-1;strongest.ub[2]=1;
  auto leaf_unit=q(4);leaf_unit.constant=-1;leaf_unit.hessian.entries={{2,2,2},{3,3,2}};
  strongest.equalities.push_back(leaf_unit);
  ConstraintPresolve strongest_eq(strongest,false,true);
  const auto strongest_residual=strongest_eq.omitted_bound_residual({1.25,0});
  check(strongest_eq.statistics().leaf_pairs==1&&strongest_residual.maximum==.25&&
      strongest_residual.sum==.25,"omitted-bound residual did not use the strongest original bound origin");
  check(eq_bounds.report_json().find("\"equality_implied_bounds\":[[0,1,0],[0,-1,0],[1,1,0],[1,-1,0]]")!=std::string::npos&&
      eq_bounds.report_json().find("\"implied_bounds\":[]")!=std::string::npos,
      "equality and inequality bound origins were not reported separately");
  // Duplicate equality rows can be removed even without a voltage pattern.
  QCQPProblem dup=scalar;auto a=q(2);a.constant=-1;a.compact_linear={{0,1}};dup.equalities={a,a};dup.inequalities={rad,rad};
  ConstraintPresolve dd(dup);check(dd.statistics().duplicate_equalities==1&&dd.statistics().duplicate_inequalities==1,"global exact row duplicates missed");
  // Audit hook must certify ORIGINAL rows and must keep reduced vectors for RHS.
  Settings hooked=st;hooked.rank_presolve_context=&reduction;auto hr=result(small,{1,.5});
  detail::OriginalSignedResidual residual;
  check(detail::audit_original_termination(small,hooked,hr,nullptr,true,&residual),"hook rejected strict point");
  check(hr.x.size()==2&&residual.dual.size()==2&&residual.valid,"hook leaked original dimensions into Workspace");
  auto bad=result(small,{0,0});check(!detail::audit_original_termination(small,hooked,bad),"singular multiplier lift falsely passed");
  // Two-level forest with reversed parent coordinate orientation. The
  // adjoint lift must process the deepest child before its eliminated parent.
  QCQPProblem chain;chain.n=6;chain.name="chain";chain.x0={0,1,1,0,1,0};
  chain.lb=Vec(6,-2);chain.ub=Vec(6,2);chain.objective=q(6);
  for(int j=0;j<6;++j)chain.objective.compact_linear.push_back({j,double(j+1)});
  chain.equalities={balance(6,0,1,3,2,2,3),balance(6,0,1,3,2,3,-2),
                    balance(6,2,3,4,5,5,7),balance(6,2,3,4,5,7,-5)};
  radial(chain,0,1);radial(chain,2,3);radial(chain,4,5);
  ConstraintPresolve cc(chain);check(cc.statistics().leaf_pairs==2,"chain reduction missed");
  auto cr=result(cc.reduced(),{.8,.6});cr.certificate_inequality_multipliers={.5,.25};
  auto reduced_grad=cc.reduced().objective.gradient(cr.x);
  for(int i=0;i<2;++i)axpy(cr.certificate_inequality_multipliers[i],cc.reduced().inequalities[i].gradient(cr.x),reduced_grad);
  cc.lift(cr);auto full_grad=chain.objective.gradient(cr.x);
  for(int i=0;i<4;++i)axpy(cr.certificate_equality_multipliers[i],chain.equalities[i].gradient(cr.x),full_grad);
  for(int i=0;i<6;++i)axpy(cr.certificate_inequality_multipliers[i],chain.inequalities[i].gradient(cr.x),full_grad);
  for(int j=0;j<4;++j)check(std::abs(full_grad[j])<1e-12,"adjoint chain leaf residual");
  check(std::abs(full_grad[4]-reduced_grad[0])<1e-12&&std::abs(full_grad[5]-reduced_grad[1])<1e-12,"adjoint pullback identity");
  // Ipopt-style audit also receives full scale/slack vectors and returns a
  // reduced signed RHS, never an incomplete original multiplier vector.
  auto native=result(small,{1,.5});native.termination_scaling_available=true;
  native.termination_equality_scales.assign(small.equalities.size(),1);
  native.termination_inequality_scales.assign(small.inequalities.size(),1);
  for(const auto& g:small.inequalities)native.termination_inequality_slacks.push_back(-g.value(native.x));
  Settings ns=hooked;ns.termination_test="ipopt";
  check(detail::audit_original_termination(small,ns,native,nullptr,true,&residual),"native-style lifted audit failed");
  // Genuine solver integration, cold start, no endpoint and no fake Hessian.
  Settings solve;solve.outer_algorithm="lazy-dca";solve.dca_engine="filter";solve.linear_backend="native";
  solve.dca_aggregate=true;solve.dca_aggregate_backend="native";solve.live_display_level=0;solve.verbose=false;
  solve.dca_rank_presolve=true;solve.acceptable_iterations=0;solve.max_outer_iterations=150;
  solve.feasibility_tolerance=solve.stationarity_tolerance=solve.dual_infeasibility_tolerance=solve.complementarity_tolerance=1e-6;
  auto solved=Solver(solve).solve(p);
  check(solved.status==Status::Solved&&solved.x.size()==4&&solved.rank_presolve.postsolve_valid,"native presolved solve did not certify original model");
  check(detail::audit_original_termination(p,solve,solved),"independent re-audit failed");
  std::cout<<"PASS exact leaf, nonzero branch, x0, near coefficients, sum rounding, box, duplicates, hook, native solve\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
