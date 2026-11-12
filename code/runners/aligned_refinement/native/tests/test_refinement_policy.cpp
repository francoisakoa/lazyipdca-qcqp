#include "factorial_trace.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <fstream>
using namespace lazyipdca::qcqp;
using namespace lazyipdca::qcqp::detail;
static int checks=0;
static void check(bool value){++checks;if(!value)throw std::runtime_error("check "+std::to_string(checks)+" failed");}
static Settings primary(const std::string& evolution,const std::string& g,const std::string& backend,const std::string& trace){
  Settings s;s.factorial_inequalities="hybrid";s.factorial_globalization=g;s.factorial_refinement=evolution;
  s.factorial_common_guards="r3";s.factorial_trace=trace;s.linear_backend=backend;s.linear_metis=true;
  s.native_al_fallback="off";s.restoration_shared_iteration_budget=true;s.restoration_max_iterations=600;
  s.max_outer_iterations=5000;s.time_limit_seconds=7200;factorial_configure(s);return s;
}
int main(int argc,char** argv){
 try{
  const std::string out=argc>1?argv[1]:".";std::filesystem::create_directories(out);
  Settings untouched;auto original=untouched;factorial_configure(untouched);
  check(untouched.convexification_study==original.convexification_study);
  check(untouched.objective_lazy_convexification==original.objective_lazy_convexification);
  check(!untouched.factorial_refinement_active());check(untouched.factorial_r3_controls());
  for(const auto& c:{std::string("cache"),std::string("hybrid")})for(const auto& g:{std::string("r3"),std::string("hs")}){
    Settings s;s.factorial_inequalities=c;s.factorial_globalization=g;s.factorial_trace=out+"/legacy";s.native_al_fallback="off";
    factorial_configure(s);check(s.convexification_study==(c=="cache"?"off":"HH"));
    check(s.factorial_r3_controls()==(g=="r3"));check(s.factorial_hs_guards()==(g=="hs"));
    check(factorial_legacy_objective_control(s)==(g=="hs"));check(!s.objective_lazy_convexification);
    check(s.majorant_conditional_retraction==(g=="hs"));check(s.factorial_refinement.empty());
  }
  for(const auto& b:{std::string("ma27"),std::string("mumps")}){
    std::string guard;
    for(const auto& g:{std::string("r3"),std::string("hs")})for(const auto& e:{std::string("frozen"),std::string("adaptive")}){
      auto s=primary(e,g,b,out+"/"+b+"_"+g+"_"+e+".jsonl");
      check(s.factorial_exact_objective());check(s.factorial_r3_controls());check(!s.factorial_hs_guards());
      check(!factorial_legacy_objective_control(s));check(s.convexification_study=="HH");
      check(s.majorant_adaptive_globalization==(g=="hs"));check(s.majorant_soc_small_step_patience==(g=="hs"?4:0));
      check(s.factorial_refinement_frozen()==(e=="frozen"));
      const auto h=factorial_guard_config_digest(s);check(h.size()==64);if(guard.empty())guard=h;check(guard==h);
      factorial_trace_begin(s);const std::string m0(64,'a');const auto phase=factorial_trace_metric_initial(s,m0,3,2,1);
      check(phase==1);check(s.factorial_trace_state->initial_model_events==1);
      factorial_trace_metric_current(s,phase,m0,m0);
      factorial_trace_commit(s,{0,1,2},{1,2},{3,4},{5},1,.1,1,"majorant_primary","none",true,0,0,0);
      check(s.factorial_trace_state->accepted_events==1);
      factorial_trace_finish(s,out+"/"+b+"_"+g+"_"+e+"_endpoint",1,0,1,1,"inequality-majorants/"+b);
      check(s.factorial_trace_state->success);
    }
  }
  auto zero=primary("frozen","r3","ma27",out+"/pre_model.jsonl");factorial_trace_begin(zero);
  factorial_trace_finish(zero,out+"/pre_model_endpoint",0,0,0.01,0,"native_exact_presolve");
  check(zero.factorial_trace_state->initial_model_events==0);check(zero.factorial_trace_state->primary_backend_factor_attempts==0);
  auto bad=[&](Settings s){bool rejected=false;try{factorial_configure(s);}catch(const std::runtime_error&){rejected=true;}check(rejected);};
  auto s=primary("frozen","r3","mumps",out+"/invalid");s.factorial_inequalities="cache";bad(s);
  s=primary("adaptive","r3","mumps",out+"/invalid");s.factorial_common_guards.clear();bad(s);
  s=primary("adaptive","hs","mumps",out+"/invalid");s.factorial_validation_legacy_objective=true;bad(s);
  s=primary("adaptive","hs","mumps",out+"/invalid");s.factorial_refinement="unlisted";bad(s);
  s=Settings{};s.factorial_common_guards="r3";bad(s);
  s=primary("adaptive","r3","mumps",out+"/invalid");s.factorial_common_guards="hs";bad(s);
  s=primary("frozen","r3","mumps",out+"/negative.jsonl");factorial_trace_begin(s);
  auto index=factorial_trace_metric_initial(s,std::string(64,'a'),2,1,0);bool rejected=false;
  try{factorial_trace_metric_current(s,index,std::string(64,'a'),std::string(64,'b'));}catch(const std::runtime_error&){rejected=true;}
  check(rejected);check(!s.factorial_trace_state->success);check(!s.factorial_trace_state->frozen_metric_invariant_verified);
  std::cout<<"checks="<<checks<<" passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
