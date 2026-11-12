#pragma once
#include "lazyipdca_qcqp/settings.hpp"
#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <stdexcept>
#include <cmath>

namespace lazyipdca::qcqp::detail {
// Research-only configuration. No historical Settings member is changed unless
// both independent factorial selectors are explicitly supplied.
inline void factorial_control_semantics(Settings& st) {
  if(!st.factorial_active())return;
  const bool hs=st.factorial_globalization=="hs";
  st.objective_lazy_convexification=st.factorial_validation_legacy_objective; // actual objective construction only
  st.majorant_conditional_retraction=hs;
  st.majorant_adaptive_globalization=hs;
  st.majorant_initial_dual_balance=hs;
  st.majorant_nonlinear_trial_soc=hs;
  st.majorant_trial_soc_full_primal=hs;
  st.majorant_trial_soc_arc=hs;
  st.majorant_soc_defect_control=hs;
  st.majorant_soc_small_step_patience=hs?4:0;
}
inline bool factorial_legacy_objective_control(const Settings& st) {
  return st.factorial_active()?st.factorial_hs_guards():st.objective_lazy_convexification;
}
inline void factorial_configure(Settings& st) {
  if(st.factorial_inequalities.empty()&&st.factorial_globalization.empty()) {
    if(st.factorial_validation_legacy_objective)throw std::runtime_error("factorial legacy-objective validation requires hybrid+hs");
    if(!st.factorial_refinement.empty()||!st.factorial_common_guards.empty())
      throw std::runtime_error("refinement/guard controls require both factorial selectors");
    return;
  }
  if((st.factorial_inequalities!="cache"&&st.factorial_inequalities!="hybrid")||
     (st.factorial_globalization!="r3"&&st.factorial_globalization!="hs"))
    throw std::runtime_error("both factorial selectors are required: inequalities cache|hybrid and globalization r3|hs");
  if(st.factorial_validation_legacy_objective&&
     (st.factorial_inequalities!="hybrid"||st.factorial_globalization!="hs"))
    throw std::runtime_error("legacy objective is restricted to the historical HH diagonal validation");
  if(!st.factorial_refinement.empty()&&st.factorial_refinement!="frozen"&&st.factorial_refinement!="adaptive")
    throw std::runtime_error("factorial refinement expects frozen|adaptive");
  if(!st.factorial_common_guards.empty()&&st.factorial_common_guards!="r3")
    throw std::runtime_error("factorial common guards currently supports r3 only");
  if(st.factorial_refinement_active()&&
     (st.factorial_inequalities!="hybrid"||st.factorial_validation_legacy_objective||st.factorial_common_guards.empty()))
    throw std::runtime_error("refinement ablation requires common hybrid initialization, exact objective and explicit common guards");
  if(st.study_controller!="r3"||st.native_al_fallback!="off"||!st.dca_endpoint_start_file.empty())
    throw std::runtime_error("factorial requires historical r3 controller, native-al-fallback off, and frozen x0");
  if(st.factorial_trace.empty())throw std::runtime_error("factorial runs require an accepted-direction trace");
  st.convexification_study=st.factorial_inequalities=="cache"?"off":"HH";
  st.majorant_curvature_corrections=0;
  factorial_control_semantics(st);
}
// SHA-256 over explicit little-endian lengths and binary64 bit patterns. No
// floating-point arithmetic or solver state mutation occurs in this observer.
class FactorialSHA256 {
  std::array<std::uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
  std::array<unsigned char,64> b{};std::uint64_t size=0;std::size_t used=0;
  static std::uint32_t r(std::uint32_t x,unsigned n){return(x>>n)|(x<<(32-n));}
  void block(){
    static constexpr std::uint32_t k[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t w[64];for(int i=0;i<16;++i)w[i]=(std::uint32_t(b[4*i])<<24)|(std::uint32_t(b[4*i+1])<<16)|(std::uint32_t(b[4*i+2])<<8)|b[4*i+3];
    for(int i=16;i<64;++i)w[i]=(r(w[i-2],17)^r(w[i-2],19)^(w[i-2]>>10))+w[i-7]+(r(w[i-15],7)^r(w[i-15],18)^(w[i-15]>>3))+w[i-16];
    auto a=h[0],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],j=h[7],v=h[1];
    for(int i=0;i<64;++i){auto t=j+(r(e,6)^r(e,11)^r(e,25))+((e&f)^(~e&g))+k[i]+w[i];auto q=(r(a,2)^r(a,13)^r(a,22))+((a&v)^(a&c)^(v&c));j=g;g=f;f=e;e=d+t;d=c;c=v;v=a;a=t+q;}
    h[0]+=a;h[1]+=v;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=j;
  }
public:
  void byte(unsigned char v){b[used++]=v;++size;if(used==64){block();used=0;}}
  void word(std::uint64_t v){for(int i=0;i<8;++i)byte(static_cast<unsigned char>(v>>(8*i)));}
  void text(const std::string& v){word(v.size());for(unsigned char c:v)byte(c);}
  void number(double d){std::uint64_t u;static_assert(sizeof(u)==sizeof(d));std::memcpy(&u,&d,sizeof(u));word(u);}
  void vector(const std::vector<double>& v){word(v.size());for(double d:v)number(d);}
  std::string finish(){auto bits=size*8;byte(0x80);while(used!=56)byte(0);for(int i=7;i>=0;--i)byte(static_cast<unsigned char>(bits>>(8*i)));std::ostringstream s;s<<std::hex<<std::setfill('0');for(auto v:h)s<<std::setw(8)<<v;return s.str();}
};
// This binds the declared guard/numerical policy once, before any native
// phase consumes its own initialization or presolve flags. The eight explicit
// globalization switches, trace paths and refinement selector are excluded.
// Per-phase consumed flags are recorded separately and may naturally differ
// after the two trajectories diverge.
inline std::string factorial_guard_config_digest(const Settings& st) {
  FactorialSHA256 h;h.text("common_r3_guards_v1");
  h.text("original_qcqp:no-native-signed-square-reduction");
  h.text(st.factorial_common_guards);h.word(st.factorial_r3_controls());h.word(st.factorial_hs_guards());
  h.text(st.study_controller);h.text(st.native_al_fallback);h.text(st.linear_backend);h.word(st.linear_metis);
  h.text(st.standard_r10_policy);h.word(st.dca_rank_presolve);h.word(st.dca_rank_implied_bounds);
  h.word(st.free_epigraph_presolve);h.text(st.majorant_startup_policy);h.word(st.standard_equality_watch);
  h.word(st.dca_filter_sparse_fallback);h.word(st.restoration_shared_iteration_budget);
  h.word(st.restoration_max_iterations);h.word(st.max_outer_iterations);
  h.vector({st.time_limit_seconds,st.feasibility_tolerance,st.stationarity_tolerance,
    st.dual_infeasibility_tolerance,st.complementarity_tolerance,st.lazy_newton_fraction_to_boundary});
  h.text("common-expanded-Newton-compensated-long-double-fma/inertia/original-KKT-v1");
  return h.finish();
}
inline std::string factorial_policies(const Settings& st) {
  const std::string c=st.factorial_active()?st.factorial_inequalities:(st.convexification_study=="off"?"cache":"hybrid");
  const std::string g=st.factorial_active()?st.factorial_globalization:(st.convexification_study=="off"?"r3":"hs");
  std::string out="\"inequality_policy\":\""+c+"\",\"globalization_policy\":\""+g+"\",\"objective_exact\":"+
      ((st.factorial_exact_objective()||(!st.factorial_active()&&!st.objective_lazy_convexification))?"true":"false");
  if(st.factorial_refinement_active())out+=",\"refinement_policy\":\""+st.factorial_refinement+
    "\",\"common_guards\":\""+st.factorial_common_guards+
    "\",\"guard_policy_version\":\"common_r3_guards_v1\",\"guard_fingerprint_scope\":\"complete_task_policy_before_phase_consumption\",\"common_representation\":\"original_qcqp\",\"guard_config_sha256\":\""+
    (st.factorial_trace_state?st.factorial_trace_state->guard_config_sha256:factorial_guard_config_digest(st))+"\"";
  return out;
}
inline void factorial_trace_begin(Settings& st) {
  if(st.factorial_trace.empty())return;
  st.factorial_trace_state=std::make_shared<FactorialTraceState>();
  if(st.factorial_refinement_active())st.factorial_trace_state->guard_config_sha256=factorial_guard_config_digest(st);
  std::ofstream out(st.factorial_trace,std::ios::trunc);
  out<<"{\"schema\":\"lazy-factorial-accepted-trace-v1\",\"event\":\"header\","<<factorial_policies(st)
     <<",\"state_encoding\":\"sha256-length-prefixed-binary64-le-x-s-z-lambda\",\"state_coordinates\":\"native-phase-internal\"}\n";
  if(!out)throw std::runtime_error("cannot create factorial accepted trace");
}
inline std::uint64_t factorial_trace_metric_initial(const Settings& st,const std::string& digest,
    std::size_t n,std::size_t mi,std::size_t me) {
  if(!st.factorial_refinement_active())return 0;
  if(!st.factorial_trace_state)throw std::runtime_error("refinement metric observer was not initialized");
  auto& s=*st.factorial_trace_state;const auto index=++s.initial_model_events;
  s.initial_hybrid_model_hashes.push_back(digest);
  if(s.initial_inequality_model_sha256.empty())s.initial_inequality_model_sha256=digest;
  s.current_phase_metric_index=index;s.current_initial_model_sha256=s.current_metric_sha256=digest;
  std::ofstream out(st.factorial_trace,std::ios::app);
  out<<"{\"event\":\"metric_initialized\","<<factorial_policies(st)
     <<",\"initialization\":\"common_certified_hybrid\",\"phase_metric_index\":"<<index
     <<",\"phase_n\":"<<n<<",\"phase_mi\":"<<mi<<",\"phase_me\":"<<me
     <<",\"initial_inequality_model_sha256\":\""<<digest<<"\"}\n";
  if(!out)throw std::runtime_error("refinement initialization trace failed");
  return index;
}
inline void factorial_trace_metric_current(const Settings& st,std::uint64_t index,
    const std::string& initial,const std::string& current) {
  if(!st.factorial_refinement_active())return;
  if(!st.factorial_trace_state)throw std::runtime_error("refinement metric observer was not initialized");
  auto& s=*st.factorial_trace_state;
  s.current_phase_metric_index=index;s.current_initial_model_sha256=initial;s.current_metric_sha256=current;
  if(st.factorial_refinement_frozen()&&initial!=current) {
    s.success=false;s.frozen_metric_invariant_verified=false;s.error="frozen inequality metric changed";
    throw std::runtime_error(s.error);
  }
}
inline void factorial_trace_commit(const Settings& st,const std::vector<double>& x,const std::vector<double>& s,
    const std::vector<double>& z,const std::vector<double>& l,int outer,double mu,double alpha,
    const char* origin,const char* postprocess,bool objective_zero,double primal,double dual,double comp) {
  if(st.factorial_trace.empty()||!st.factorial_trace_state)return;
  auto& state=*st.factorial_trace_state;
  try {
    FactorialSHA256 hash;hash.vector(x);hash.vector(s);hash.vector(z);hash.vector(l);
    std::ofstream out(st.factorial_trace,std::ios::app);
    out<<std::setprecision(17)<<"{\"event\":\"accepted\",\"accepted_index\":"<<++state.accepted_events
       <<",\"outer_iteration\":"<<outer<<','<<factorial_policies(st)<<",\"direction_origin\":\""<<origin
       <<"\",\"postprocess\":\""<<postprocess<<"\",\"final_origin\":\""<<(std::string(postprocess)=="none"?origin:postprocess)<<"\",\"objective_E0_zero\":"<<(objective_zero?"true":"false")
       <<",\"mu\":"<<mu<<",\"alpha\":"<<alpha<<",\"phase_n\":"<<x.size()<<",\"phase_mi\":"<<s.size()
       <<",\"phase_me\":"<<l.size()<<",\"state_sha256\":\""<<hash.finish()<<"\",\"primal\":"<<primal
       <<",\"dual\":"<<dual<<",\"complementarity\":"<<comp;
    if(st.factorial_refinement_active())out<<",\"phase_metric_index\":"<<state.current_phase_metric_index
       <<",\"initial_inequality_model_sha256\":\""<<state.current_initial_model_sha256
       <<"\",\"current_inequality_model_sha256\":\""<<state.current_metric_sha256
       <<"\",\"refinement_attempt_events\":"<<state.refinement_attempt_events
       <<",\"accepted_refinements\":"<<state.accepted_refinements
       <<",\"refinement_update_events\":"<<state.accepted_refinements
       <<",\"refinement_rollback_events\":"<<state.refinement_rollback_events;
    out<<"}\n";
    if(!out)throw std::runtime_error("accepted trace write failed");
    if(st.factorial_exact_objective()&&!objective_zero)throw std::runtime_error("exact objective invariant violated at accepted direction");
  }catch(const std::exception& e){state.success=false;state.error=e.what();}
}
inline void factorial_trace_state_event(const Settings& st,const std::vector<double>& x,
    const std::vector<double>& s,const std::vector<double>& z,const std::vector<double>& l,
    const char* event,int outer,double mu) {
  if(st.factorial_trace.empty()||!st.factorial_trace_state)return;
  try {
    FactorialSHA256 hash;hash.vector(x);hash.vector(s);hash.vector(z);hash.vector(l);
    std::ofstream out(st.factorial_trace,std::ios::app);
    out<<std::setprecision(17)<<"{\"event\":\""<<event<<"\",\"after_accepted_index\":"
       <<st.factorial_trace_state->accepted_events<<','<<factorial_policies(st)<<",\"outer_iteration\":"<<outer
       <<",\"mu\":"<<mu<<",\"phase_n\":"<<x.size()<<",\"phase_mi\":"<<s.size()
       <<",\"phase_me\":"<<l.size()<<",\"state_sha256\":\""<<hash.finish()<<"\"}\n";
    if(!out)throw std::runtime_error("state event trace write failed");
  }catch(const std::exception& e){st.factorial_trace_state->success=false;st.factorial_trace_state->error=e.what();}
}
inline std::string factorial_json_string(const std::string& s) {
  std::string out;for(unsigned char c:s){if(c=='"'||c=='\\')out+='\\';if(c>=32)out+=char(c);else out+=' ';}return out;
}
inline void factorial_trace_finish(const Settings& st,const std::string& endpoint,int outer,int restoration,double elapsed,std::uint64_t accepted,const std::string& actual_backend="") {
  if(st.factorial_trace.empty()||!st.factorial_trace_state||endpoint.empty())return;
  auto& s=*st.factorial_trace_state;
  if(s.accepted_events!=accepted){s.success=false;s.error="accepted trace count differs from endpoint";}
  const bool budget_verified=st.restoration_shared_iteration_budget&&restoration>=0&&
      restoration<=st.restoration_max_iterations&&restoration<=600&&outer>=0&&
      outer<=st.max_outer_iterations&&outer<=5000&&std::isfinite(elapsed)&&elapsed>=0&&
      elapsed<=st.time_limit_seconds&&elapsed<=7200;
  std::ofstream out(endpoint+".factorial.json");
  out<<"{\"schema\":\"lazy-factorial-endpoint-v1\","<<factorial_policies(st)
     <<",\"trace_success\":"<<(s.success?"true":"false")<<",\"accepted_events\":"<<s.accepted_events
     <<",\"budget_verified\":"<<(budget_verified?"true":"false")<<",\"endpoint_accepted_steps\":"<<accepted<<",\"outer_iterations\":"<<outer<<",\"restoration_iterations\":"<<restoration<<",\"seconds\":"<<std::setprecision(17)<<elapsed<<",\"error\":\""<<factorial_json_string(s.error)<<"\",\"legacy_objective_validation\":"<<(st.factorial_validation_legacy_objective?"true":"false");
  if(st.factorial_refinement_active()) {
    FactorialSHA256 h;for(const auto& v:s.initial_hybrid_model_hashes)h.text(v);
    out<<",\"initialization\":\"common_certified_hybrid\",\"initial_inequality_model_sha256\":\""<<s.initial_inequality_model_sha256
       <<"\",\"initial_hybrid_models_sha256\":\""<<h.finish()<<"\",\"initial_model_events\":"<<s.initial_model_events
       <<",\"refinement_attempt_events\":"<<s.refinement_attempt_events
       <<",\"accepted_refinements\":"<<s.accepted_refinements
       <<",\"refinement_update_events\":"<<s.accepted_refinements
       <<",\"refinement_rollback_events\":"<<s.refinement_rollback_events
       <<",\"metric_hash_evaluations\":"<<s.metric_hash_evaluations
       <<",\"native_solver_completed\":true,\"initial_model_state\":\""<<(s.initial_model_events?"reached":"not_reached")
       <<"\",\"initialization_calls\":"<<s.initial_model_events
       <<",\"primary_backend_factor_attempts\":"<<s.primary_backend_factor_attempts
       <<",\"primary_newton_backend_used\":"<<(s.primary_backend_factor_attempts?"true":"false")
       <<",\"native_pre_model_return\":"<<((s.initial_model_events==0&&accepted==0)?"true":"false")
       <<",\"frozen_metric_invariant_verified\":"<<(s.frozen_metric_invariant_verified?"true":"false")
       <<",\"requested_linear_backend\":\""<<(st.linear_backend=="mumps"&&st.linear_metis?"mumps_metis":st.linear_backend)
       <<"\",\"actual_linear_backend\":\""<<factorial_json_string(actual_backend)<<"\"";
  }
  out<<"}\n";
  if(!out)throw std::runtime_error("factorial endpoint sidecar write failed");
}
}
