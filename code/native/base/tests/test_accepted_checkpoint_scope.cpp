#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include <iostream>
#include <stdexcept>
using namespace lazyipdca::qcqp;
int main(int argc,char**argv){

 Settings st;st.outer_algorithm="lazy-dca";st.dca_engine="majorants";st.linear_backend="native";
 st.live_display_level=0;st.max_outer_iterations=100;st.acceptable_iterations=0;st.time_limit_seconds=10;
 st.restoration_enabled=false;st.dca_rank_presolve=false;
 QCQPProblem p;p.name="same-name";p.n=1;p.x0={0};p.lb={-2};p.ub={2};
 p.objective.hessian.n=1;p.objective.hessian.entries={{0,0,2}};p.objective.linear={-2};p.objective.constant=1;
 int outer_observed=0,private_observed=0,private_accepted=0;bool inside=false,tried=false;
 st.accepted_checkpoint_callback=[&](const QCQPProblem&,Result&,const ConstraintPresolve*,int,double){
   if(inside){++private_observed;return;}
   ++outer_observed;
   if(!tried){tried=true;inside=true;auto private_model=p;private_model.x0={-.5};
     auto result=Solver(st).solve(private_model);private_accepted=result.lazy_newton.accepted_steps;inside=false;}
 };
 const auto r=Solver(st).solve(p);
 if(outer_observed<1||private_accepted<1||private_observed!=0)
   throw std::runtime_error("nested private accepted-state callback escaped depth scope");
 auto original=read_qcq_compact(argc==2?argv[1]:CHECKPOINT_RANK_FIXTURE);ConstraintPresolve rank(original);
 if(!rank.changed()||rank.reduced().n>=original.n)throw std::runtime_error("rank fixture did not reduce");
 Result snap;snap.x=rank.reduced().x0;
 snap.certificate_equality_multipliers.assign(rank.reduced().equalities.size(),0);
 snap.certificate_inequality_multipliers.assign(rank.reduced().inequalities.size(),.2);
 snap.certificate_lower_multipliers.assign(rank.reduced().n,0);snap.certificate_upper_multipliers.assign(rank.reduced().n,0);
 auto reference=snap;const auto before=rank.statistics();rank.lift_checkpoint(snap);const auto after=rank.statistics();
 if(before.audit_lift_calls!=after.audit_lift_calls||before.audit_gradient_evaluations!=after.audit_gradient_evaluations)
   throw std::runtime_error("checkpoint mapping mutated live audit counters");
 rank.lift(reference);
 if(snap.x!=reference.x||snap.certificate_equality_multipliers!=reference.certificate_equality_multipliers||
    snap.certificate_inequality_multipliers!=reference.certificate_inequality_multipliers||
    snap.certificate_lower_multipliers!=reference.certificate_lower_multipliers||
    snap.certificate_upper_multipliers!=reference.certificate_upper_multipliers)
   throw std::runtime_error("checkpoint lift differs from existing original mapping");
 std::cout<<"checkpoint scope PASS outer_callbacks="<<outer_observed<<" private_accepted="<<private_accepted
          <<" private_callbacks="<<private_observed<<" rank_original_dimensions="<<snap.x.size()<<" live_counters_unchanged=1\n";
}
