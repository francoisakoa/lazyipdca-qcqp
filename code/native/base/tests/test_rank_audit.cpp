#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include "oracles/acceptable_rank_r1.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
namespace {
void require(bool x,const char* message){if(!x)throw std::runtime_error(message);}
QuadraticFunction q(int n){QuadraticFunction f;f.hessian.n=n;f.compact_linear_dimension=n;return f;}
QuadraticFunction balance(int n,int u,int v,int a,int b,double r,double t){
  auto f=q(n);auto add=[&](int i,int j,double val){if(val!=0)f.hessian.entries.push_back({std::min(i,j),std::max(i,j),val});};
  add(u,u,2*r);add(v,v,2*r);add(u,a,-r);add(v,b,-r);add(v,a,t);add(u,b,-t);
  std::sort(f.hessian.entries.begin(),f.hessian.entries.end(),[](auto x,auto y){return std::pair{x.i,x.j}<std::pair{y.i,y.j};});return f;
}
void radial(QCQPProblem& p,int u,int v){auto f=q(p.n);f.constant=.25;f.hessian.entries={{u,u,-2},{v,v,-2}};p.inequalities.push_back(f);f.constant=-4;f.hessian.entries={{u,u,2},{v,v,2}};p.inequalities.push_back(f);}
QCQPProblem leaf(){
  QCQPProblem p;p.name="leaf";p.n=4;p.x0={1,0,1,0};p.lb=Vec(4,-2);p.ub=Vec(4,2);
  p.objective=q(4);p.objective.constant=2.5;p.objective.compact_linear={{0,-2},{3,-1}};
  for(int j=0;j<4;++j)p.objective.hessian.entries.push_back({j,j,1});
  p.equalities={balance(4,2,3,0,1,2,3),balance(4,2,3,0,1,3,-2)};
  radial(p,0,1);radial(p,2,3);p.validate();return p;
}
QCQPProblem chain(){
  QCQPProblem p;p.n=6;p.name="reversed-chain";p.x0={0,1,1,0,1,0};p.lb=Vec(6,-2);p.ub=Vec(6,2);p.objective=q(6);
  for(int j=0;j<6;++j)p.objective.compact_linear.push_back({j,double(j+1)});
  p.equalities={balance(6,0,1,3,2,2,3),balance(6,0,1,3,2,3,-2),balance(6,2,3,4,5,5,7),balance(6,2,3,4,5,7,-5)};
  radial(p,0,1);radial(p,2,3);radial(p,4,5);return p;
}
QCQPProblem duplicates(){
  QCQPProblem p;p.n=3;p.name="duplicates-fixed-constants";p.x0={1,.5,0};p.lb={-2,-2,0};p.ub={2,2,0};p.objective=q(3);
  p.objective.compact_linear={{0,2},{1,-3},{2,5}};p.objective.hessian.entries={{0,0,2},{0,1,.5},{1,2,-2}};
  auto e=q(3);e.constant=-1;e.compact_linear={{0,1},{1,1}};
  p.equalities={e,e,q(3)};radial(p,0,1);p.inequalities.push_back(p.inequalities.back());
  auto near=p.inequalities.back();near.constant=std::nextafter(near.constant,INFINITY);p.inequalities.push_back(near);
  auto c=q(3);c.constant=-1;p.inequalities.push_back(c);return p;
}
QCQPProblem cancellation(){
  QCQPProblem p;p.name="different-roundoff-recipes";p.n=2;p.x0={1,.2};p.lb=Vec(2,-2);p.ub=Vec(2,2);p.objective=q(2);
  auto a=q(2);a.constant=-1;a.hessian.entries={{0,0,2e20},{0,0,2},{0,0,-2e20}};
  auto b=q(2);b.constant=-1;b.hessian.entries={{0,0,2}};
  p.inequalities={a,b,b};return p;
}
std::vector<std::pair<std::string,double>> metrics(const Result& r){
  std::vector<std::pair<std::string,double>> out;
#define FIELD(f) out.push_back({#f,double(r.f)})
  FIELD(termination_audit_valid);FIELD(acceptable_point);FIELD(acceptable_convergence);FIELD(ipopt_desired_convergence);
  FIELD(termination_square_problem);FIELD(fixed_multipliers_reconstructed);FIELD(objective);
  FIELD(feasibility.equality_inf);FIELD(feasibility.inequality_inf);FIELD(feasibility.bound_inf);FIELD(feasibility.total_inf);
  FIELD(dual_infeasibility_inf);FIELD(complementarity_inf);FIELD(model_stationarity_inf);
  FIELD(raw_kkt_error);FIELD(acceptable_error);FIELD(scaled_nlp_error);FIELD(original_normalized_nlp_error);
  FIELD(termination_internal_dual_inf);FIELD(termination_internal_primal_inf);FIELD(termination_internal_compl_inf);
  FIELD(termination_dual_inf);FIELD(termination_compl_inf);FIELD(termination_sd);FIELD(termination_sc);
  FIELD(dual_error_scaling);FIELD(complementarity_error_scaling);FIELD(constraint_multiplier_l1);FIELD(bound_slack_multiplier_l1);
  FIELD(ipopt_dual_multiplier_dimension);FIELD(ipopt_complementarity_multiplier_dimension);
  FIELD(ipopt_y_c_dimension);FIELD(ipopt_y_d_dimension);FIELD(ipopt_z_L_dimension);FIELD(ipopt_z_U_dimension);FIELD(ipopt_v_L_dimension);FIELD(ipopt_v_U_dimension);
  FIELD(termination_previous_objective);FIELD(termination_current_objective);FIELD(termination_objective_iteration);FIELD(termination_objective_change);
#undef FIELD
  return out;
}
void compare(const Result& a,const Result& b){
  const auto aa=metrics(a),bb=metrics(b);
  for(std::size_t i=0;i<aa.size();++i)if(aa[i].second!=bb[i].second && !(std::isnan(aa[i].second)&&std::isnan(bb[i].second))){
    std::ostringstream os;os<<std::setprecision(17)<<"metric mismatch "<<aa[i].first<<" "<<aa[i].second<<" vs "<<bb[i].second;throw std::runtime_error(os.str());}
  require(a.termination_test==b.termination_test,"termination label changed");
}
Result state(const QCQPProblem& p,std::mt19937_64& rng){
  std::uniform_real_distribution<double> d(.2,1.4);Result r;r.x=p.x0;
  for(int j=0;j<p.n;++j)r.x[j]=p.lb[j]==p.ub[j]?p.lb[j]:d(rng);
  for(std::size_t e=0;e<p.equalities.size();++e)r.certificate_equality_multipliers.push_back((d(rng)-.8)*1e4);
  for(std::size_t i=0;i<p.inequalities.size();++i)r.certificate_inequality_multipliers.push_back(d(rng)*1e5);
  for(int j=0;j<p.n;++j){r.certificate_lower_multipliers.push_back(std::isfinite(p.lb[j])?d(rng):0);r.certificate_upper_multipliers.push_back(std::isfinite(p.ub[j])?d(rng):0);}
  r.termination_scaling_available=true;r.termination_objective_scale=.013;r.outer_iterations=7;
  r.termination_previous_objective=2;r.termination_current_objective=3;r.termination_objective_iteration=6;
  for(std::size_t e=0;e<p.equalities.size();++e)r.termination_equality_scales.push_back(d(rng));
  for(std::size_t i=0;i<p.inequalities.size();++i){r.termination_inequality_scales.push_back(d(rng));r.termination_inequality_slacks.push_back(d(rng));}
  return r;
}
std::size_t compared=0;
void parity(const QCQPProblem& p,bool bounds){
  ConstraintPresolve reduction(p,bounds);std::mt19937_64 rng(20260920);
  for(const auto* test:{"project","ipopt"})for(double smax:{1.,100.,1e6})for(int it=0;it<40;++it){
    Settings st;st.termination_test=test;st.nlp_error_scaling_max=smax;st.rank_presolve_context=&reduction;st.dca_rank_audit_mode="mapped";
    auto small=state(reduction.reduced(),rng),full=small,unchanged=small;
    reduction.lift(full);auto direct=full;
    Settings plain=st;plain.rank_presolve_context=nullptr;
    rank_r1_oracle::OriginalSignedResidual old_res;
    const bool old=rank_r1_oracle::audit_original_termination(p,plain,full,nullptr,true,&old_res);
    detail::OriginalSignedResidual direct_res,new_res;
    const bool middle=detail::audit_original_termination(p,plain,direct,nullptr,true,&direct_res);
    const bool now=detail::audit_original_termination(reduction.reduced(),st,small,nullptr,true,&new_res);
    require(old==now&&old==middle,"strict decision changed");compare(full,small);compare(full,direct);
    require(old_res.dual==direct_res.dual && old_res.equality==direct_res.equality && old_res.inequality==direct_res.inequality,"direct kernel changed frozen signed residual");
    require(small.x==unchanged.x && small.certificate_equality_multipliers==unchanged.certificate_equality_multipliers && small.certificate_inequality_multipliers==unchanged.certificate_inequality_multipliers && small.certificate_lower_multipliers==unchanged.certificate_lower_multipliers && small.certificate_upper_multipliers==unchanged.certificate_upper_multipliers,"mapped audit changed working vectors");
    require(small.termination_equality_scales==unchanged.termination_equality_scales && small.termination_inequality_scales==unchanged.termination_inequality_scales && small.termination_inequality_slacks==unchanged.termination_inequality_slacks,"mapped audit changed live scales/slacks");
    std::vector<long double> pull(reduction.reduced().n,0);
    for(int j=0;j<p.n;++j)pull[reduction.variable_map()[j]]+=(long double)old_res.dual[j];
    for(int j=0;j<reduction.reduced().n;++j)require(new_res.dual[j]==double(pull[j]),"signed dual pullback changed");
    // Check mode additionally compares the equality/inequality pullbacks and
    // all metadata using the legacy lifted path. Re-audit same outer tests
    // objective-history preservation and invalidation on each changed state.
    st.dca_rank_audit_mode="check";
    require(detail::audit_original_termination(reduction.reduced(),st,small,nullptr,true,&new_res)==now,"repeat audit changed decision");
    compare(small,full);++compared;
  }
  require(reduction.statistics().audit_value_reuses>0,"no representative reuse exercised");
  require(reduction.statistics().audit_duplicate_rows_covered==reduction.statistics().duplicate_equalities+reduction.statistics().duplicate_inequalities,"not all duplicate groups covered");

}
}
int main(){try{
  for(bool bounds:{false,true}){parity(leaf(),bounds);parity(chain(),bounds);parity(duplicates(),bounds);parity(cancellation(),bounds);}
  ConstraintPresolve c(cancellation(),false);require(c.statistics().audit_recipe_exceptions>0,"algebraic duplicates with different evaluation order incorrectly shared");
  auto p=leaf();ConstraintPresolve r(p);std::mt19937_64 rng(5);Settings st;st.rank_presolve_context=&r;st.termination_test="ipopt";
  auto bad=state(r.reduced(),rng);bad.termination_inequality_slacks.clear();detail::OriginalSignedResidual sr;
  require(!detail::audit_original_termination(r.reduced(),st,bad,nullptr,true,&sr)&&!bad.termination_audit_valid&&!sr.valid,"missing slacks falsely certified");
  bad=state(r.reduced(),rng);bad.certificate_inequality_multipliers[0]=-1;
  require(!detail::audit_original_termination(r.reduced(),st,bad),"negative dual falsely certified");
  bad=state(r.reduced(),rng);bad.x[0]=NAN;require(!detail::audit_original_termination(r.reduced(),st,bad),"nonfinite x falsely certified");
  bad=state(r.reduced(),rng);bad.x.assign(r.reduced().n,0);
  require(!detail::audit_original_termination(r.reduced(),st,bad),"zero-voltage singular lift falsely certified");
  Settings solve;solve.outer_algorithm="lazy-dca";solve.dca_engine="filter";solve.linear_backend="native";solve.dca_aggregate=true;solve.dca_aggregate_backend="native";
  solve.live_display_level=0;solve.verbose=false;solve.dca_rank_presolve=true;solve.acceptable_iterations=0;solve.max_outer_iterations=150;
  solve.feasibility_tolerance=solve.stationarity_tolerance=solve.dual_infeasibility_tolerance=solve.complementarity_tolerance=1e-6;
  auto done=Solver(solve).solve(p);
  require(done.status==Status::Solved&&done.rank_presolve.postsolve_valid,"mapped solve failed original certificate");
  require(done.rank_presolve.audit_calls>0 && done.rank_presolve.audit_mapped_calls==done.rank_presolve.audit_calls,"mapped mode not used by controller");
  require(done.rank_presolve.audit_full_calls==0&&done.rank_presolve.audit_result_copies==0&&done.rank_presolve.audit_lift_calls==1&&done.rank_presolve.audit_final_full_calls==1,"full per-iteration work still present or final audit omitted");
  require(detail::audit_original_termination(p,solve,done),"final independent full re-audit failed");
  std::cout<<"PASS frozen R1 scalar/signed parity states="<<compared<<"; project+ipopt smax; leaves/chains; duplicate/fixed/constant/bound/roundoff; invalid states; no internal Result copies; mandatory full final audit\n";
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
