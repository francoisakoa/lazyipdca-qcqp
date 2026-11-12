#!/usr/bin/env python3
"""Compile the production R10 controller fragments against deterministic storage.

This preserves the 25-check R10 controller/GR/commit regression. It imports no
optimizer, mutates no solver source, and writes generated test inputs only to
--output. Production fragments are re-extracted on each run so the assertions
exercise the current engine, original path, and Workspace fraction method.
"""
from pathlib import Path
import argparse
import hashlib,json,shutil,subprocess,time
ap=argparse.ArgumentParser()
ap.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2])
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--cxx',default='g++')
args=ap.parse_args()
assets=Path(__file__).resolve().parent
src=args.source.resolve()
folder=args.output.resolve()
folder.mkdir(parents=True,exist_ok=True)
shutil.copyfile(assets/'test_actual_r10_control.cpp',folder/'test_actual_r10_control.cpp')
engine=(src/'src/dca_inequality_majorant_engine.hpp').read_text()
def between(first,last):
    start=engine.index(first);start=engine.index('\n',start)+1
    return engine[start:engine.index(last,start)]
parser=between('// GR-BOUND-REPAIR-PARSER-BEGIN','// GR-BOUND-REPAIR-PARSER-END')
prefix=between('// GR-BOUND-REPAIR-CONTROL-BEGIN','// GR-BOUND-REPAIR-CONTROL-BODY')
suffix=between('// GR-BOUND-REPAIR-NEXT-BEGIN','// GR-BOUND-REPAIR-NEXT-END')
line_budget=between('// GR-BOUND-LINE-BUDGET-BEGIN','// GR-BOUND-LINE-BUDGET-END')
stage=between('// GR-CONSISTENT-BOUND-STAGING-BEGIN:','// GR-CONSISTENT-BOUND-STAGING-END')
transfer_start=engine.index('                if(gr_consistent_bound_slacks) {',engine.index('              if(source_filter_accepted) {'))
transfer=engine[transfer_start:engine.index('                if(!accepted_trial_soc)',transfer_start)]
commit_start=engine.index('          Vec tx=w.x,ts=std::move(accepted_slacks)');commit=engine[commit_start:engine.index('          std::unique_ptr<DcaTailTerms> tail;',commit_start)]
slope_start=engine.index('              detail::gr::Direction derivative;')
slope_end=engine.index(';',engine.index('              gr_slope=detail::gr::directional(',slope_start))+1
slope=engine[slope_start:slope_end]
deadline_start=engine.index('          if((st.majorant_soc_defect_control||accepted_analytic_soc||accepted_consistent_path)')
deadline_line=engine[deadline_start:engine.index('{',deadline_start)];deadline_expression=deadline_line.strip()[3:-1]
files={name:(src/'src'/name).read_text() for name in ['dca_majorant_barrier_path.hpp','consistent_bound_slack_path.hpp','bound_product_path.hpp']}
for name,text in files.items():(folder/name).write_text(text)
atomic=(src/'src/atomic_newton.cpp').read_text()
fraction_start=atomic.index('  double fraction(const Direction&')
fraction=atomic[fraction_start:atomic.index('  double theta(',fraction_start)]
mock=(assets/'mock_workspace_template.hpp').read_text()
mock=mock.replace('  Omitted omitted_original_bounds',fraction+'\n  Omitted omitted_original_bounds')
(folder/'mock_workspace.hpp').write_text(mock)
functions='''
#include <cstdlib>
#include "lazyipdca_qcqp/settings.hpp"
#include <iomanip>
#include <iostream>
namespace detail=lazyipdca::qcqp::detail;
enum class Status {Initial,TimeLimit};
struct Control {bool accepted=false,stop=false,consistent=false;Status status=Status::Initial;std::string message;};
static bool actual_parse_trigger(const char* value){if(value)setenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER",value,1);else unsetenv("LAZY_ORIGINAL_GR_CONSISTENT_BOUND_TRIGGER");
 const lazyipdca::qcqp::Settings requested;
 (void)requested;
'''+parser+'''return gr_bound_repair_on_failure;}
template<class Attempt>static Control actual_control(const Workspace& w,const Direction& direction,bool gr_consistent_bound_slacks,bool gr_bound_repair_on_failure,bool initially_stopped,Attempt attempt,bool measured_stall=false){
 struct {bool active;} gr_bound_watch{measured_stall};
 bool accepted=false,stop=initially_stopped,accepted_consistent_path=false;Control result;
 const bool gr_consistent_bound_master=gr_consistent_bound_slacks;
 const std::string gr_bound_trigger=gr_bound_repair_on_failure?"on_failure":"always";
 const int level=0,outer=0;double alpha=1,start=0,elapsed=0;
 struct {double time_limit_seconds,lazy_newton_fraction_to_boundary;} st{1,.995};auto seconds=[&](double){return elapsed;};
'''+prefix+'''
 const double bound_path_last_tested_alpha=alpha;
 attempt(gr_consistent_bound_slacks,bound_path_attempt,accepted,stop,elapsed);
'''+suffix+'''
 }
 result.accepted=accepted;result.stop=stop;result.consistent=accepted_consistent_path;return result;
}
static int actual_line_budget(bool gr_consistent_bound_slacks,bool gr_bound_repair_on_failure,int configured,bool measured_stall=false){
 const bool bound_failed_only=gr_bound_repair_on_failure&&!measured_stall;
 struct {int line_search_max_steps;} st{configured};
'''+line_budget+'''return bound_path_line_steps;
}
static bool actual_precommit_deadline(bool accepted_consistent_path,double elapsed){
 const bool accepted_analytic_soc=false;const double start=0;
 struct {bool majorant_soc_defect_control;double time_limit_seconds;} st{false,1};
 auto seconds=[&](double){return elapsed;};return '''+deadline_expression+''';
}
static detail::gr::Result actual_engine_slope(const Workspace& w,const Direction& direction,const DcaMajorantBarrierPath& path,const detail::gr::State& gr_anchor,const detail::gr::Parameters& gr_parameters,bool gr_consistent_bound_slacks){detail::gr::Result gr_slope;
'''+slope+'''(void)bound_slack_tangent_defect;return gr_slope;}
static Workspace actual_engine_commit(const Workspace& initial,const Direction& direction,const DcaMajorantBarrierPath& path,double alpha,double mu,bool gr_consistent_bound_slacks){Workspace w=initial;const bool accepted_analytic_soc=false,bound_product_curve=true,accepted_consistent_path=gr_consistent_bound_slacks;
 Vec trial_primal,trial_slacks,trial_duals,trial_equality_duals,accepted_primal,accepted_slacks,accepted_duals,accepted_equality_duals;
'''+stage+transfer+commit+'''w.x.swap(tx);w.s.swap(ts);w.z.swap(tz);w.lambda.swap(tl);return w;}
'''
(folder/'source_under_test.hpp').write_text('#include "mock_workspace.hpp"\n#include "lazyipdca_qcqp/primal_dual_merit.hpp"\n#include "dca_majorant_barrier_path.hpp"\n'+functions)
argv=[args.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(src/'include'),str(folder/'test_actual_r10_control.cpp'),'-o',str(folder/'test_actual_r10_control')]
t=time.monotonic();build=subprocess.run(argv,capture_output=True,text=True);run=None;output=build.stdout+build.stderr
if build.returncode==0:run=subprocess.run([str(folder/'test_actual_r10_control')],capture_output=True,text=True);output+=run.stdout+run.stderr
print(output,end='')
receipt={'schema':'R11-actual-R10-full-step-bound-controller-regression-v1','scope':'ACTUAL parser/2-attempt prefix with full-fraction gate/suffix/deadline/slope/stage/transfer/commit, actual source-extracted single-trial repair budget and Workspace fraction; actual full path and GR, mock only original-QCQP storage','source_sha256':{str(src/'src'/name):hashlib.sha256(text.encode()).hexdigest() for name,text in [('dca_inequality_majorant_engine.hpp',engine),('atomic_newton.cpp',atomic),*files.items()]},'compile_argv':argv,'compile_returncode':build.returncode,'test_returncode':run.returncode if run else None,'output':output,'seconds':time.monotonic()-t,'optimizer_calls':0,'actual_parser_extract':parser,'actual_control_prefix':prefix,'actual_control_suffix':suffix,'actual_repair_line_budget':line_budget,'actual_fraction_method':fraction,'actual_precommit_deadline':deadline_expression,'actual_transfer_extract':transfer,'actual_commit_extract':commit}
(folder/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(build.returncode or (run.returncode if run else 1))
