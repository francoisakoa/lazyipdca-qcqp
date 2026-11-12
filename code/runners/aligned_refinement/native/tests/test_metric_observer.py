#!/usr/bin/env python3
"""Compile actual new metric-observer methods against a small typed fixture.
This is not a complete native solver build and does not call an optimizer.
"""
from pathlib import Path
import argparse,subprocess,json,hashlib,struct
P=argparse.ArgumentParser();P.add_argument('--source',type=Path,required=True);P.add_argument('--out',type=Path,required=True);a=P.parse_args();a.out.mkdir(parents=True,exist_ok=True)
s=(a.source/'src/dca_convexification_study_models.hpp').read_text();methods=s[s.index('  std::string study_factorial_metric_sha256() const {'):s.index('  std::string study_metric_digest() const {')]
head=r'''
#include "factorial_trace.hpp"
#include <iostream>
#include <filesystem>
using namespace lazyipdca::qcqp;
struct SymTriplet{int i,j;double v;};
namespace study {struct Atom{double weight=0;int strikes=0;std::vector<int>indices;std::vector<double>values;};struct Correction{std::string family="hybrid";bool support_scalar=true;double rho=2;std::vector<int>support{0,1};std::vector<SymTriplet>original{{0,0,-1}},correction;std::vector<Atom>hybrid_atoms;};}
enum class NewtonCost{Diagnostics};struct CostScope{CostScope(int&,NewtonCost){}};
struct Row{double scale=1;};
struct Workspace{Settings st;int costs=0,n=2,mi=1,me=0;std::vector<int>inequality_kind{0};std::vector<Row>inequalities{{}};};
struct Model{
 Workspace&w;bool curvature_ready=true;std::vector<study::Correction>study_inequalities{{}};
 std::string factorial_initial_metric_sha256,factorial_current_metric_sha256;std::uint64_t factorial_phase_metric_index=0;
 Model(Workspace&q):w(q){}
'''
tail=r'''
};
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::filesystem::create_directories(argv[1]);int checks=0;
 auto ck=[&](bool v){++checks;if(!v)throw std::runtime_error("metric observer check failed "+std::to_string(checks));};
 auto setup=[&](const char*e,const char*name){Workspace w;w.st.factorial_inequalities="hybrid";w.st.factorial_globalization="r3";w.st.factorial_common_guards="r3";w.st.factorial_refinement=e;w.st.native_al_fallback="off";w.st.factorial_trace=std::string(argv[1])+name;detail::factorial_configure(w.st);detail::factorial_trace_begin(w.st);return w;};
 auto init=[&](Model&m){m.factorial_initial_metric_sha256=m.factorial_current_metric_sha256=m.study_factorial_metric_sha256();m.factorial_phase_metric_index=detail::factorial_trace_metric_initial(m.w.st,m.factorial_initial_metric_sha256,m.w.n,m.w.mi,m.w.me);};
 auto f=setup("frozen","/frozen.jsonl");Model m(f);init(m);auto h=m.factorial_initial_metric_sha256;ck(h.size()==64);m.study_factorial_snapshot();ck(f.st.factorial_trace_state->metric_hash_evaluations==1);
 m.study_inequalities[0].rho=3;bool rejected=false;try{m.study_factorial_snapshot(true);}catch(const std::runtime_error&){rejected=true;}ck(rejected);ck(!f.st.factorial_trace_state->frozen_metric_invariant_verified);
 auto g=setup("frozen","/final.jsonl");Model n(g);init(n);n.study_inequalities[0].rho=4;n.study_factorial_final_check();ck(!g.st.factorial_trace_state->success);ck(!g.st.factorial_trace_state->frozen_metric_invariant_verified);
 auto v=setup("adaptive","/adaptive.jsonl");Model q(v);init(q);ck(q.factorial_initial_metric_sha256==h);q.study_inequalities[0].rho=1;q.study_factorial_snapshot(true);ck(v.st.factorial_trace_state->success);ck(q.factorial_current_metric_sha256!=h);
 auto c=setup("frozen","/caches.jsonl");Model z(c);z.study_inequalities[0].hybrid_atoms.push_back({.5,0,{0,1},{1.,0.}});init(z);auto before=z.factorial_initial_metric_sha256;z.study_inequalities[0].hybrid_atoms[0].strikes=9;z.study_factorial_snapshot(true);ck(z.factorial_current_metric_sha256==before);z.study_inequalities[0].hybrid_atoms[0].weight=.75;z.study_factorial_final_check();ck(!c.st.factorial_trace_state->success);
 Workspace legacy;Model old(legacy);old.study_factorial_snapshot();old.study_factorial_final_check();ck(!legacy.st.factorial_trace_state);
 std::cout<<"checks="<<checks<<" passed; initial_sha256="<<h<<'\n';
}
'''
cpp=a.out/'metric_fixture.cpp';cpp.write_text(head+methods+tail)
cmd=['g++','-std=c++17','-O2','-Wall','-Wextra','-pedantic','-I',str(a.source/'include'),'-I',str(a.source/'src'),str(cpp),'-o',str(a.out/'metric_fixture')]
subprocess.run(cmd,check=True);r=subprocess.run([str(a.out/'metric_fixture'),str(a.out/'traces')],capture_output=True,text=True,check=True);print(r.stdout,end='')
h=hashlib.sha256()
def word(v):h.update(struct.pack('<Q',v))
def text(v):v=v.encode();word(len(v));h.update(v)
def vector(v):word(len(v));h.update(struct.pack('<'+'d'*len(v),*v))
text('hybrid-inequality-metric-binary64-v1');word(2);word(1);word(0);word(0);text('hybrid');word(1);vector([1.,2.]);word(2);word(0);word(1);word(1);word(0);word(0);vector([-1.]);word(0);word(0)
assert r.stdout.split('initial_sha256=')[1].strip()==h.hexdigest()
print('Independent Python binary64 metric SHA256 agrees')
(a.out/'RESULT.json').write_text(json.dumps(dict(passed=True,scope='Actual new metric observer methods, typed workspace fixture only; not a complete native solver build.',stdout=r.stdout,optimizer_invocations=0),indent=2)+'\n')
