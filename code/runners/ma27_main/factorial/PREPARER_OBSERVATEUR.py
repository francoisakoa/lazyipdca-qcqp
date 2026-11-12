#!/usr/bin/env python3
"""Build an observation-only source tree from the hash-bound qualified sources.
No factorial numerical/control branch is applied to this baseline observer.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,difflib

def prepare(base:Path, research:Path, target:Path):
    if target.exists():raise RuntimeError('Observer target already exists: '+str(target))
    shutil.copytree(base,target)
    def change(rel,old,new):
        p=target/rel;s=p.read_text()
        if s.count(old)<1:raise RuntimeError((rel,s.count(old),old[:80]))
        p.write_text(s.replace(old,new))
    # Shared passive diagnostics data and writer; no research CLI selector is exposed.
    shutil.copy2(research/'src/factorial_trace.hpp',target/'src/factorial_trace.hpp')
    rsettings=(research/'include/lazyipdca_qcqp/settings.hpp').read_text()
    (target/'include/lazyipdca_qcqp/settings.hpp').write_text(rsettings)
    change('src/main.cpp','#include <set>','#include <set>\n#include "factorial_trace.hpp"')
    change('src/main.cpp','      else if(a=="--convexification-study") st.convexification_study=need("--convexification-study");','      else if(a=="--factorial-trace") st.factorial_trace=need("--factorial-trace");\n      else if(a=="--convexification-study") st.convexification_study=need("--convexification-study");')
    change('src/main.cpp','    Solver solver(st);','    detail::factorial_trace_begin(st);\n    Solver solver(st);')
    change('src/main.cpp','    const auto solver_end=std::chrono::steady_clock::now();','    const auto solver_end=std::chrono::steady_clock::now();\n    detail::factorial_trace_finish(st,json_path,r.outer_iterations,r.restoration_iterations,r.elapsed_seconds,r.lazy_newton.accepted_steps);')
    change('src/atomic_newton.cpp','#include "lazyipdca_qcqp/native_original_anchor_finish.hpp"','#include "lazyipdca_qcqp/native_original_anchor_finish.hpp"\n#include "factorial_trace.hpp"')
    e='src/dca_inequality_majorant_engine.hpp'
    change(e,'          bool curvature_refined=false;Vec curvature_rhs;','          bool factorial_original_curvature=false,factorial_soc_replaced=false;\n          bool curvature_refined=false;Vec curvature_rhs;')
    change(e,'                    if(keep){direction=std::move(trial);break;}','                    if(keep){direction=std::move(trial);factorial_original_curvature=true;break;}')
    change(e,'direction=std::move(best_direction);','direction=std::move(best_direction);factorial_soc_replaced=true;')
    change(e,'direction=std::move(combined);','direction=std::move(combined);factorial_soc_replaced=true;')
    change(e,'          models->discard();models->study_rollback_trial();\n          ++stats.majorant_original_cache_evaluations;','          const bool factorial_objective_zero=!w.objective.model_active&&w.objective.model_atoms.empty()&&w.objective.study_model_scalar==0;\n          models->discard();models->study_rollback_trial();\n          ++stats.majorant_original_cache_evaluations;')
    re=(research/e).read_text();start=re.index('          if(!st.factorial_trace.empty()) {\n            CostScope factorial_diagnostic_cost')
    stop=re.index('          const bool bound_stagnation=',start)
    change(e,'          current=dca_outer_metrics(w);\n          const bool bound_stagnation=', '          current=dca_outer_metrics(w);\n'+re[start:stop]+'          const bool bound_stagnation=')
    state_changes=[('          w.evaluate();equality_start_restored=true;', '          w.evaluate();equality_start_restored=true;\n          detail::factorial_trace_state_event(st,w.x,w.s,w.z,w.lambda,"startup_restoration_commit",0,result.lazy_newton.initial_barrier);'), ('          if(accepted) {\n            result.status=Status::Solved;', '          if(accepted) {\n            detail::factorial_trace_state_event(st,w.x,w.s,w.z,w.lambda,"state_recovery_commit",result.outer_iterations,mu);\n            result.status=Status::Solved;'), ('                if(level)std::cout<<"[AUTO-STARTUP-TERMINAL-RETRY] accepted=1 iter="', '                detail::factorial_trace_state_event(st,w.x,w.s,w.z,w.lambda,"terminal_central_restart_commit",outer+1,mu);\n                if(level)std::cout<<"[AUTO-STARTUP-TERMINAL-RETRY] accepted=1 iter="'), ('              if(level)std::cout<<"[AUTO-STARTUP-RETRY] accepted=1 iter="', '              detail::factorial_trace_state_event(st,w.x,w.s,w.z,w.lambda,"central_restart_commit",outer+1,mu);\n              if(level)std::cout<<"[AUTO-STARTUP-RETRY] accepted=1 iter="'), ('          stats.majorant_barrier_residual=current.residual(mu);\n          if(level>=2) {', '          detail::factorial_trace_state_event(st,w.x,w.s,w.z,w.lambda,"post_multiplier_update",outer+1,mu);\n          stats.majorant_barrier_residual=current.residual(mu);\n          if(level>=2) {')]
    for old,new in state_changes:change(e,old,new)
    originals={str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest()for p in base.rglob('*')if p.is_file()}
    changed=[]
    for p in target.rglob('*'):
        if p.is_file():
            rel=str(p.relative_to(target));new=hashlib.sha256(p.read_bytes()).hexdigest()
            if originals.get(rel)!=new:changed.append({'path':rel,'qualified_sha256':originals.get(rel),'observer_sha256':new})
    receipt={'schema':'lazy-factorial-baseline-observer-source-v1','qualified_source':str(base.resolve()),'qualified_files':len(originals),'changes':changed,'numerical_changes':False,'purpose':'accepted-commit passive observation only; no factorial numerical selectors exposed','gate_required':'original baseline vs observer exact final coordinates and nonclock counters; then observer trajectory vs research diagonals'}
    (target.parent/'OBSERVATEUR_SOURCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--research',type=Path,default=Path(__file__).parent/'source');p.add_argument('--out',required=True,type=Path);a=p.parse_args();print(json.dumps(prepare(a.baseline,a.research,a.out),indent=2))
