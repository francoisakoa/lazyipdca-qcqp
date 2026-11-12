#include "lazyipdca_qcqp/acceptable.hpp"
#include "lazyipdca_qcqp/dca_filter.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
void check(bool v,const char* msg){if(!v)throw std::runtime_error(msg);}
int main(){try{
 detail::AcceptableCounter c;
 check(!c.observe(0,true,2),"initial point counted");
 check(!c.observe(1,true,2)&&!c.observe(1,true,2),"duplicate counted");
 check(c.observe(2,true,2),"consecutive points not counted");
 check(!c.observe(3,false,2)&&c.count==0,"failure did not reset");
 check(!c.observe(4,true,2)&&!c.observe(6,true,2),"gap did not reset");
 check(!c.observe(7,true,0)&&c.count==0,"disable failed");
 QCQPProblem p;p.n=2;p.lb={0,-2};p.ub={0,2};p.x0={0,0};
 p.objective.linear={3,0.001};p.objective.hessian.n=2;
 Settings s;s.acceptable_tolerance=0.01;s.acceptable_dual_tolerance=0.01;
 Result r;r.x=p.x0;r.certificate_lower_multipliers={0,0};r.certificate_upper_multipliers={0,0};
 check(!detail::audit_original_termination(p,s,r),"false strict convergence");
 check(r.acceptable_point&&r.termination_audit_valid&&r.certificate_lower_multipliers[0]==3&&r.dual_infeasibility_inf==0.001,"fixed reconstruction or infinity norm incorrect");
 check(r.ipopt_z_L_dimension==1&&r.ipopt_z_U_dimension==1&&
       r.ipopt_complementarity_multiplier_dimension==2,
       "fixed variable leaked into Ipopt multiplier dimensions");
 r.certificate_lower_multipliers[0]=0;
 detail::audit_original_termination(p,s,r,nullptr,false);
 check(r.dual_infeasibility_inf==3&&!r.acceptable_point,"wrong fixed multiplier masked");
 p.objective.linear[1]=0;
 check(detail::audit_original_termination(p,s,r),"strict stationary fixed point rejected");
 r.x[0]=0.1;detail::audit_original_termination(p,s,r);
 check(!r.acceptable_point&&r.feasibility.total_inf>=0.1,"fixed primal violation masked");
 r.x[0]=0;r.certificate_upper_multipliers[1]=-1;detail::audit_original_termination(p,s,r);
 check(!r.termination_audit_valid&&!r.acceptable_point,"negative multiplier accepted");
 r.certificate_upper_multipliers[1]=0;r.x[1]=NAN;detail::audit_original_termination(p,s,r);
 check(!r.termination_audit_valid,"NaN accepted");

 // A large L1 sum must not replace the infinity norm in a stopping test.
 QCQPProblem many;many.n=1;many.lb={-INFINITY};many.ub={INFINITY};many.x0={0};many.objective.linear={0};many.objective.hessian.n=1;
 QuadraticFunction q;q.linear={0};q.hessian.n=1;q.constant=5e-7;many.equalities.assign(100,q);
 Result mr;mr.x={0};mr.certificate_equality_multipliers.assign(100,0);mr.certificate_lower_multipliers={0};mr.certificate_upper_multipliers={0};
 Settings ms;check(detail::audit_original_termination(many,ms,mr)&&mr.feasibility.total_inf==5e-7,"L1 substituted for max violation");
 many.equalities.resize(1);many.equalities[0].constant=0;many.equalities[0].linear={1};many.objective.linear={-9999.999};mr.certificate_equality_multipliers={10000};
 ms.acceptable_tolerance=1e-4;ms.acceptable_dual_tolerance=1e-4;detail::audit_original_termination(many,ms,mr);
 check(mr.termination_sd==100&&mr.acceptable_error<ms.acceptable_tolerance&&!mr.acceptable_point,"absolute dual ceiling bypassed by normalization");
 ms.acceptable_dual_tolerance=0.01;detail::audit_original_termination(many,ms,mr);check(mr.acceptable_point&&!mr.ipopt_desired_convergence,"acceptable confused with strict");
 ms.stationarity_tolerance=2e-5;ms.dual_infeasibility_tolerance=0.01;
 check(detail::audit_original_termination(many,ms,mr)&&
       mr.raw_kkt_error>ms.stationarity_tolerance&&
       mr.scaled_nlp_error<=ms.stationarity_tolerance,
       "strict convergence did not use scaled NLP error");

 // One g(x)<=0 multiplier is y_d in stationarity and v_U in the slack
 // complementarity block.  Both appearances must contribute to s_d.
 QCQPProblem one_ineq;one_ineq.n=1;one_ineq.lb={-INFINITY};one_ineq.ub={INFINITY};
 one_ineq.x0={0};one_ineq.objective.linear={0};one_ineq.objective.hessian.n=1;
 QuadraticFunction iq;iq.constant=-1;iq.linear={0};iq.hessian.n=1;
 one_ineq.inequalities={iq};
 Result ir;ir.x={0};ir.certificate_inequality_multipliers={10000};
 ir.certificate_lower_multipliers={0};ir.certificate_upper_multipliers={0};
 detail::audit_original_termination(one_ineq,ms,ir);
 check(ir.constraint_multiplier_l1==10000&&ir.bound_slack_multiplier_l1==10000&&
       ir.ipopt_y_d_dimension==1&&ir.ipopt_v_U_dimension==1&&
       ir.ipopt_dual_multiplier_dimension==2&&ir.dual_error_scaling==100,
       "Ipopt inequality multiplier scaling is not y_d/v_U consistent");
 p.lb={-2,-2};p.ub={2,2};p.objective.linear={0,0};p.objective.hessian.entries={{0,0,2},{1,1,2}};p.x0={0.8,0.6};
 s.linear_backend="native";s.dca_engine="filter";s.max_outer_iterations=1;s.acceptable_iterations=1;
 s.acceptable_tolerance=1;s.acceptable_constraint_tolerance=1;s.acceptable_dual_tolerance=100;s.acceptable_complementarity_tolerance=100;
 s.stationarity_tolerance=s.dual_infeasibility_tolerance=s.complementarity_tolerance=1e-14;
 auto a=solve_filter_dca(p,s);
 check(a.status==Status::SolvedAcceptable&&a.acceptable_convergence&&!a.ipopt_desired_convergence&&a.acceptable_streak==1,"end-to-end acceptable status failed");
 s.acceptable_iterations=0;auto b=solve_filter_dca(p,s);check(b.status!=Status::SolvedAcceptable&&!b.acceptable_convergence,"disabled status produced");
 s.acceptable_iterations=15;s.acceptable_tolerance=10;
 s.max_outer_iterations=2;s.line_search_max_steps=0;
 s.acceptable_on_failure=false;
 auto failed=solve_filter_dca(p,s);
 check(failed.status==Status::LineSearchFailure,
       "forced terminal globalization failure was not preserved");
 s.acceptable_on_failure=true;
 auto recovered=solve_filter_dca(p,s);
 if(recovered.status!=Status::SolvedAcceptableOnFailure)
   std::cerr<<"acceptable-on-failure diagnostic: status="<<status_string(recovered.status)
            <<" point="<<recovered.acceptable_point<<" valid="<<recovered.termination_audit_valid
            <<" streak="<<recovered.acceptable_streak<<"/"<<recovered.acceptable_required
            <<" message="<<recovered.message<<'\n';
 check(recovered.status==Status::SolvedAcceptableOnFailure&&
       recovered.acceptable_convergence&&recovered.acceptable_on_failure&&
       recovered.acceptable_trigger=="terminal_failure"&&
       recovered.original_failure_status=="line_search_failure"&&
       recovered.acceptable_streak<recovered.acceptable_required,
       "audited acceptable-on-failure status failed");
 s.acceptable_iterations=0;
 auto recovered_without_streak_rule=solve_filter_dca(p,s);
 check(recovered_without_streak_rule.status==Status::SolvedAcceptableOnFailure&&
       recovered_without_streak_rule.acceptable_convergence&&
       recovered_without_streak_rule.acceptable_streak==0&&
       recovered_without_streak_rule.acceptable_required==0,
       "acceptable-on-failure incorrectly depends on acceptable-iter");
 std::cout<<"acceptable counter, independent original audit, fixed signs, invalid states, and solver status PASS\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
