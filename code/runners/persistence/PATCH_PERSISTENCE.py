#!/usr/bin/env python3
"""Add a generic deterministic R3 curvature-cache control in an isolated tree."""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil
HERE=Path(__file__).resolve().parent

SUPPORT=r'''#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace lazyipdca::qcqp::research {
// SHA-256 of a canonical little-endian, length-delimited binary64 stream.
// Used only in qualification traces, never in the timed cache comparison.
class Sha256 {
  std::array<std::uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
  std::array<unsigned char,64> block{};std::size_t used=0;std::uint64_t length=0;
  static std::uint32_t rr(std::uint32_t x,int n){return(x>>n)|(x<<(32-n));}
  void compress(){
    static constexpr std::uint32_t k[64]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::uint32_t w[64];for(int j=0;j<16;++j)w[j]=(std::uint32_t(block[4*j])<<24)|(std::uint32_t(block[4*j+1])<<16)|(std::uint32_t(block[4*j+2])<<8)|block[4*j+3];
    for(int j=16;j<64;++j){auto x=w[j-15],y=w[j-2];w[j]=w[j-16]+(rr(x,7)^rr(x,18)^(x>>3))+w[j-7]+(rr(y,17)^rr(y,19)^(y>>10));}
    auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];
    for(int j=0;j<64;++j){auto t=v+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^(~e&g))+k[j]+w[j];auto u=(rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c));v=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;}
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
  }
public:
  void byte(unsigned char x){block[used++]=x;++length;if(used==64){compress();used=0;}}
  void raw(const std::string& s){for(unsigned char x:s)byte(x);}
  void integer(std::uint64_t v){for(int i=0;i<8;++i){byte(static_cast<unsigned char>(v&255));v>>=8;}}
  void real(double v){static_assert(sizeof(double)==8,"binary64 required");std::uint64_t bits;std::memcpy(&bits,&v,8);integer(bits);}
  template<class V>void vector_real(const V& a){integer(a.size());for(double v:a)real(v);}
  template<class V>void vector_integer(const V& a){integer(a.size());for(auto v:a)integer(static_cast<std::uint64_t>(v));}
  std::string finish(){auto bits=length*8;byte(0x80);while(used!=56)byte(0);for(int i=7;i>=0;--i)byte(static_cast<unsigned char>(bits>>(8*i)));std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto v:h)out<<std::setw(8)<<v;return out.str();}
};
inline std::size_t& cache_instance_count(){static std::size_t n=0;return n;}
inline std::size_t next_cache_instance(){return ++cache_instance_count();}
inline void cache_session_completed(const std::string& mode,const std::string& path){
  if(path.empty())return;
  std::ofstream out(path,std::ios::app);
  if(!out)throw std::runtime_error("cannot complete curvature research evidence: "+path);
  out<<"{\"schema\":\"research-curvature-cache-session-v1\",\"mode\":\""<<mode
     <<"\",\"model_instances\":"<<cache_instance_count()<<",\"solver_returned\":true}\n";
  out.flush();if(!out)throw std::runtime_error("curvature session evidence write failed");
}
inline void validate_cache_mode(const std::string& mode,bool active,const std::string& engine,
    const std::string& study,bool objective_lazy){
  if(mode!="reuse"&&mode!="rebuild")throw std::invalid_argument("research-curvature-cache expects reuse|rebuild");
  if(active&&((engine!="majorants"&&engine!="auto")||study!="off"||objective_lazy))
    throw std::invalid_argument("curvature-cache ablation supports fixed R3 majorants only: study off and objective lazy convexification off");
}
}
'''

MEMBERS=r'''
  // Research bookkeeping is independent of the solver's state and factors.
  std::size_t research_instance=0,research_preparations=0,research_row_builds=0;
  std::size_t research_reanchors=0,research_trace_records=0,research_reuse_hits=0;
  double research_reset_seconds=0;
  std::string research_correction_identity;
  std::ofstream research_report_stream,research_trace_stream;
  bool research_active()const{return w.st.research_curvature_cache!="reuse"||
      !w.st.research_curvature_report.empty()||!w.st.research_curvature_trace.empty();}
  bool research_trace_enabled()const{return research_trace_stream.is_open();}
  void research_initialize(){
    research::validate_cache_mode(w.st.research_curvature_cache,research_active(),
        w.st.dca_engine,w.st.convexification_study,w.st.objective_lazy_convexification);
    if(!research_active())return;
    research_instance=research::next_cache_instance();
    auto open=[](std::ofstream& out,const std::string& path){if(path.empty())return;out.open(path,std::ios::app);if(!out)throw std::runtime_error("cannot open curvature research evidence: "+path);};
    open(research_report_stream,w.st.research_curvature_report);
    open(research_trace_stream,w.st.research_curvature_trace);
  }
  void research_prepared(){
    if(!research_active())return;
    ++research_preparations;
    for(const auto& q:w.inequalities)if(!q.hessian.empty()||!q.signed_square.empty())++research_row_builds;
  }
  void research_rebuild_before_anchor(){
    if(w.st.research_curvature_cache!="rebuild"||!research_reanchors)return;
    const auto reset_start=Clock::now();
    // Discard all numeric curvature data and remap it from the original
    // Hessians using the very same constructor. Keep the primal/dual state,
    // evaluator cache, envelope, symbolic slots and linear factor caches.
    inequality.assign(w.mi,Curvature{});corrected_inequalities.clear();
    curvature_ready=false;rho_max=0;positive_spectral_count=0;
    curvature_builds=0;small_spectra=0;analytic_spectra=0;gershgorin_certificates=0;
    research_reset_seconds+=seconds(reset_start);
    prepare_curvature_once();
  }
  std::string research_matrix_identity()const{
    research::Sha256 h;h.integer(w.n);h.integer(w.mi);h.real(rho_max);h.integer(positive_spectral_count);
    h.vector_integer(corrected_inequalities);
    for(const auto& c:inequality){
      h.integer(c.ready);h.integer(c.half_spectral);h.integer(c.positive_spectral);
      h.real(c.scale);h.real(c.rho);h.real(c.spectral_shift);h.real(c.native_shift);
      h.vector_integer(c.support);h.integer(c.hessian.size());
      for(const auto& e:c.hessian){h.integer(e.i);h.integer(e.j);h.real(e.v);}
    }
    // Include the coefficients actually installed in the model and the
    // retained symbolic mapping, not only the cached construction result.
    for(const auto& q:w.inequalities){
      h.integer(q.model_active);h.integer(q.model_hessian.size());
      for(const auto& e:q.model_hessian){h.integer(e.i);h.integer(e.j);h.real(e.v);}
      h.vector_integer(q.model_hi);h.vector_integer(q.model_hj);h.vector_integer(q.model_hglobal);
    }
    return h.finish();
  }
  void research_trace_state(const char* event,int outer,double mu,double rho,double kappa,
      double alpha=0,const Direction* direction=nullptr,const char* origin="none"){
    if(!research_trace_enabled())return;
    research::Sha256 state;state.vector_real(w.x);state.vector_real(w.lambda);
    state.vector_real(w.z);state.vector_real(w.s);state.vector_integer(w.active);
    state.real(w.regularization);state.real(w.penalty);state.real(w.source_barrier_mu);
    state.integer(w.source_primal_barrier_mode);state.integer(w.equality_stabilization.enabled);
    state.real(w.equality_stabilization.rho);state.real(w.equality_stabilization.reference_true_equality);
    state.vector_real(w.equality_stabilization.multiplier_center);
    state.vector_real(w.equality_stabilization.source_inequality_center);
    state.integer(w.equality_stabilization.epochs);state.integer(w.equality_stabilization.last_update_generation);
    state.real(mu);state.real(rho);state.real(kappa);state.real(alpha);
    state.integer(direction!=nullptr);if(direction){state.vector_real(direction->x);state.vector_real(direction->lambda);state.vector_real(direction->s);state.vector_real(direction->z);}
    research_trace_stream<<"{\"schema\":\"research-curvature-cache-trace-v1\",\"event\":\""<<event
      <<"\",\"model_instance\":"<<research_instance<<",\"mode\":\""<<w.st.research_curvature_cache
      <<"\",\"reanchor\":"<<research_reanchors<<",\"outer\":"<<outer
      <<",\"accepted_steps\":"<<w.stats.accepted_steps<<",\"origin\":\""<<origin
      <<"\",\"correction_identity\":\""<<research_correction_identity<<"\",\"state_identity\":\""<<state.finish()<<"\"}\n";
    research_trace_stream.flush();if(!research_trace_stream)throw std::runtime_error("curvature trace write failed");
    ++research_trace_records;
  }
  void research_anchored(){
    if(!research_active())return;
    if(research_reanchors&&w.st.research_curvature_cache=="reuse")++research_reuse_hits;
    ++research_reanchors;
    if(research_trace_enabled()){
      research_correction_identity=research_matrix_identity();
      research_trace_state("reanchor",-1,w.source_barrier_mu,0,0);
    }
    if(w.st.research_curvature_cache=="rebuild"){
      // Legacy reuse diagnostics must not claim cache hits in this control.
      stats.curvature_reuses=0;stats.majorant_curvature_reuses=0;
    }
  }
  void research_summary()noexcept{
    if(!research_report_stream.is_open())return;
    try{
      research_report_stream<<std::setprecision(17)
        <<"{\"schema\":\"research-curvature-cache-summary-v1\",\"model_instance\":"<<research_instance
        <<",\"mode\":\""<<w.st.research_curvature_cache<<"\",\"preparations\":"<<research_preparations
        <<",\"reanchors\":"<<research_reanchors<<",\"quadratic_row_builds\":"<<research_row_builds
        <<",\"corrected_rows\":"<<corrected_inequalities.size()<<",\"construction_seconds\":"<<curvature_seconds
        <<",\"rebuild_reset_seconds\":"<<research_reset_seconds<<",\"trace_records\":"<<research_trace_records
        <<",\"reuse_hits\":"<<research_reuse_hits<<",\"complete\":true}\n";
      research_report_stream.flush();
    }catch(...){/* Missing or incomplete evidence is rejected by the controller. */}
  }
'''

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_map(p):return {str(x.relative_to(p)):sha(x)for x in sorted(p.rglob('*'))if x.is_file()}
def patch(out):
    ref=HERE/'native/reference_source';out=Path(out).resolve()
    if out.exists():raise ValueError('New source directory required: '+str(out))
    if out==ref or ref in out.parents or out in ref.parents:raise ValueError('Separate source directory required')
    expected={k.split('/source/',1)[-1]:v for k,v in json.loads((HERE/'BASE_CONTEXT.json').read_text())['plan_header']['source_sha256'].items()}
    if len(expected)!=823 or source_map(ref)!=expected:raise ValueError('Authenticated 823-file reference changed')
    shutil.copytree(ref,out)
    def change(file,old,new):
        p=out/file;s=p.read_text()
        if s.count(old)!=1:raise ValueError('Ambiguous patch anchor: '+file+' '+old[:90])
        p.write_text(s.replace(old,new))
    change('include/lazyipdca_qcqp/settings.hpp','  std::string convexification_study = "off";',
      '  std::string research_curvature_cache="reuse";\n  std::string research_curvature_report,research_curvature_trace;\n  std::string convexification_study = "off";')
    change('src/main.cpp','      else if(a=="--no-dca-model-reuse") st.dca_filter_model_reuse=false;',
      '''      else if(a=="--research-curvature-cache")st.research_curvature_cache=need(a.c_str());
      else if(a=="--research-curvature-report")st.research_curvature_report=need(a.c_str());
      else if(a=="--research-curvature-trace")st.research_curvature_trace=need(a.c_str());
      else if(a=="--no-dca-model-reuse") st.dca_filter_model_reuse=false;''')
    change('src/main.cpp','    << "  --no-dca-model-reuse       G4 control: rebuild at each accepted point\\n"',
      '    << "  --research-curvature-cache reuse|rebuild  fixed R3 inequality curvature cache control\\n"\n    << "  --research-curvature-report FILE  append construction summary JSONL\\n"\n    << "  --research-curvature-trace FILE  qualification-only exact identity JSONL\\n"\n    << "  --no-dca-model-reuse       G4 control: rebuild at each accepted point\\n"')
    change('src/main.cpp','#include <set>','#include <set>\n#include "research_curvature_support.hpp"')
    change('src/main.cpp','    if(st.study_controller!="r3"&&st.study_controller!="certified")throw std::runtime_error("study-controller must be r3 or certified");',
      '''    research::validate_cache_mode(st.research_curvature_cache,
      st.research_curvature_cache!="reuse"||!st.research_curvature_report.empty()||!st.research_curvature_trace.empty(),
      st.dca_engine,st.convexification_study,st.objective_lazy_convexification);
    if(!st.research_curvature_report.empty()&&st.research_curvature_report==st.research_curvature_trace)
      throw std::invalid_argument("research report and trace paths must differ");
    for(const auto* path:{&st.research_curvature_report,&st.research_curvature_trace})if(!path->empty()){
      std::ofstream evidence(*path,std::ios::app);
      if(!evidence)throw std::invalid_argument("cannot initialize curvature research evidence: "+*path);
    }
    if(st.study_controller!="r3"&&st.study_controller!="certified")throw std::runtime_error("study-controller must be r3 or certified");''')
    change('src/atomic_newton.cpp','#include "lazyipdca_qcqp/native_original_anchor_finish.hpp"',
      '#include "research_curvature_support.hpp"\n#include "lazyipdca_qcqp/native_original_anchor_finish.hpp"')
    change('src/main.cpp','    const auto solver_end=std::chrono::steady_clock::now();',
      '    const auto solver_end=std::chrono::steady_clock::now();\n    research::cache_session_completed(st.research_curvature_cache,st.research_curvature_report);')
    change('src/dca_inequality_majorant_models.hpp','  ~DcaInequalityModels(){study_summary();}',
      MEMBERS+'\n  ~DcaInequalityModels(){research_summary();study_summary();}')
    change('src/dca_inequality_majorant_models.hpp','      : w(work), stats(diagnostics), inequality(work.mi) {',
      '      : w(work), stats(diagnostics), inequality(work.mi) {\n    research_initialize();')
    change('src/dca_inequality_majorant_models.hpp','    curvature_ready = true;\n    curvature_seconds += seconds(start);',
      '    curvature_ready = true;\n    curvature_seconds += seconds(start);\n    research_prepared();')
    change('src/dca_inequality_majorant_models.hpp','      throw std::runtime_error("discard the previous inequality model before reanchoring");\n    const auto start = Clock::now();',
      '      throw std::runtime_error("discard the previous inequality model before reanchoring");\n    research_rebuild_before_anchor();\n    const auto start = Clock::now();')
    change('src/dca_inequality_majorant_models.hpp','    stats.majorant_curvature_reuses = curvature_reuses;\n    stats.model_build_seconds += seconds(start);',
      '    stats.majorant_curvature_reuses = curvature_reuses;\n    stats.model_build_seconds += seconds(start);\n    research_anchored();')
    change('src/dca_inequality_majorant_engine.hpp','          models->reanchor_from_cache();\n          if(outer)++stats.majorant_reanchors;',
      '          models->reanchor_from_cache();\n          models->research_trace_state("controller",outer,mu,rho_memory,gr_kappa);\n          if(outer)++stats.majorant_reanchors;')
    change('src/dca_inequality_majorant_engine.hpp','          if(checkpoint_majorant_depth==1&&st.accepted_checkpoint_callback&&!fixed_centering.active()) {',
      '''          models->research_trace_state("accepted",outer+1,mu,rho_memory,gr_kappa,alpha,&direction,
            accepted_kkt_soc?"kkt-soc":bound_product_curve?"product-central":"main-accepted");
          if(checkpoint_majorant_depth==1&&st.accepted_checkpoint_callback&&!fixed_centering.active()) {''')
    (out/'src/research_curvature_support.hpp').write_text(SUPPORT)
    changed=[str(p.relative_to(out))for p in sorted(out.rglob('*'))if p.is_file()and(not(ref/p.relative_to(out)).exists()or p.read_bytes()!=(ref/p.relative_to(out)).read_bytes())]
    receipt=dict(schema='fixed-r3-curvature-cache-patch-v1',changed_files=changed,source_reference_sha256=source_map(ref),source_research_sha256=source_map(out),
      default_reuse_preserves_numerics=True,rebuild_scope='deterministic inequality curvature constructor at every model installation after first, including same-center restoration',
      trace_scope='SHA256 of binary64 matrices and primal-dual/slack/controller state at all reanchors and main accepted steps; rejected trial states are not traced',
      timed_trace_required=False,unsupported=['convexification-study modes','lazy adaptive objective correction'],optimizer_invocations=0)
    (out.parent/'PATCH_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    chunks=[]
    for rel in changed:chunks.extend(difflib.unified_diff((ref/rel).read_text().splitlines(True)if(ref/rel).exists()else[],(out/rel).read_text().splitlines(True),fromfile='reference/'+rel,tofile='research/'+rel))
    (out.parent/'PERSISTENCE.patch').write_text(''.join(chunks))
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=patch(a.out);print(json.dumps({'changed_files':result['changed_files'],'optimizer_invocations':0},indent=2))
