#!/usr/bin/env python3
"""Repeated same-QCQP comparison with original-coefficient endpoint checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from qcq_model_r3p23 import read_qcq
from live_process_r3p29r3 import run_logged
ROOT=Path(__file__).resolve().parents[1]
TOL=1e-6

def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def switch(command,positive,negative,enabled):
    command[:]=[v for v in command if v not in (positive,negative)]
    command.append(positive if enabled else negative)
def value(command,flag,item):
    while flag in command:
        i=command.index(flag);del command[i:i+2]
    command.extend([flag,str(item)])

def validate_endpoint(model,data,branch):
    if data.get('status')!='solved':raise ValueError(data.get('message','not solved'))
    x=data.get('x')
    if not isinstance(x,list) or len(x)!=model.n or not all(finite(v) for v in x):raise ValueError('invalid primal vector')
    residuals=model.residuals(x)
    names=('certificate_equality_multipliers','certificate_inequality_multipliers','certificate_lower_multipliers','certificate_upper_multipliers')
    vectors=[data.get(k) for k in names]
    sizes=(len(model.equalities),len(model.inequalities),model.n,model.n)
    independent=all(isinstance(v,list) and len(v)==n for v,n in zip(vectors,sizes))
    if independent:
        if not all(finite(x) for v in vectors for x in v):raise ValueError('nonfinite certificate')
        eq,ineq,lower,upper=vectors
        if any(x<0 for v in (ineq,lower,upper) for x in v):raise ValueError('negative inequality multiplier')
        residuals.update(model.kkt_residuals(x,eq+ineq,[u-l for u,l in zip(upper,lower)]))
        for j in range(model.n):
            if model.lb[j]==model.ub[j]:continue
            for bound,mult in ((model.lb[j],lower[j]),(model.ub[j],upper[j])):
                if math.isfinite(bound):residuals['complementarity_inf']=max(residuals['complementarity_inf'],abs(mult*(x[j]-bound)))
                elif mult!=0:raise ValueError('dual on absent bound')
    else:
        raise ValueError('certificate vectors required')
    for key in ('dual_infeasibility_inf','complementarity_inf','scaled_nlp_error'):
        if not finite(data.get(key)) or not 0<=data[key]<=TOL:raise ValueError('reported KKT failure: '+key)
    for key,v in residuals.items():
        if not finite(v) or not 0<=v<=TOL:raise ValueError('independent residual failure: '+key+'='+str(v))
    objective=model.objective_value(x)
    if not finite(objective) or not finite(data.get('objective')) or abs(objective-data['objective'])>1e-7*max(1,abs(objective)):raise ValueError('original objective mismatch')
    if branch=='single-loop':
        if data.get('dca_single_loop_full_refit_calls')!=1 or data.get('multiplier_refit_attempts')!=0:raise ValueError('single-loop multiplier contract violated')
        if data.get('lazy_dca_equality_retraction_attempts')!=0:raise ValueError('unexpected single-loop retraction')
    return objective,residuals,independent

def build_command(binary,model,result,profile,mode,threads,limit):
    if mode=='baseline':
        cmd=[str(binary),'--problem',str(model),'--json',str(result)]+list(profile)
        switch(cmd,'--lazyqp-affine-support-fastpath','--no-lazyqp-affine-support-fastpath',True)
        switch(cmd,'--restoration-exact-parallel-row-dots','--no-restoration-exact-parallel-row-dots',threads>1)
    else:
        cmd=[str(binary),'--problem',str(model),'--json',str(result),
             '--algorithm','lazy-newton','--linear-backend','ldl','--linear-metis','1',
             '--feas-tol','1e-6','--opt-tol','1e-6','--dual-inf-tol','1e-6','--compl-inf-tol','1e-6',
             '--lazy-newton-initial-mu','1e-4','--lazy-newton-predictor-corrector',
             '--lazy-newton-regularization-initial','1e-6','--lazy-newton-regularization-decay','0.2',
             '--lazy-newton-objective-gradient-target','100','--lazy-newton-constraint-gradient-target','100',
             '--lazy-newton-activation-margin','1','--lazy-newton-dense-threshold','32',
             '--max-outer','1000','--kkt-berr-reject','1e-7']
        if mode=='unreduced':cmd.append('--no-lazy-newton-reduction')
        if mode=='fresh-curvature':cmd.append('--no-lazy-newton-lazy-curvature')
        if mode=='all-inequalities':cmd.append('--no-lazy-newton-lazy-constraints')
    value(cmd,'--time-limit',limit);cmd.append('--quiet');return cmd

def validate_newton_mechanisms(data,mode):
    if data.get('effective_outer_algorithm')!='lazy-newton':raise ValueError('wrong algorithm')
    if data.get('lazy_newton_original_certificates',0)<1:raise ValueError('no original certificate')
    # General QCQPs need not admit elimination, an unshifted final factor,
    # or exactly one symbolic analysis. Report mechanisms without imposing
    # case118's performance contract on mathematical correctness.


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    source=ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--qcq',type=Path);source.add_argument('--case',type=Path)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--baseline-binary',type=Path)
    ap.add_argument('--reuse-qcq',action='store_true',help='Read the supplied frozen QCQP directly; hash checks remain active and its permissions are preserved')
    ap.add_argument('--no-predictor-corrector',action='store_true',help='Use the legacy centering profile')
    ap.add_argument('--centering-control',action='store_true',help='Also run r8 with the second-order corrector disabled')
    ap.add_argument('--packed-gradients',action='store_true',help='Use optional contiguous gradient buffers; measured production profile keeps individual row buffers')
    ap.add_argument('--packed-control',action='store_true',help='Also measure the same predictor-corrector profile with contiguous gradient buffers')
    ap.add_argument('--unpacked-control',action='store_true',help='Also run r8 with individual cached Jacobian allocations')
    ap.add_argument('--dense-input-control',action='store_true',help='Also run r8 with legacy dense input; potentially large RAM use')
    ap.add_argument('--skip-ipopt',action='store_true',help='Compare Lazy modes only; no new Ipopt quality or speed claim')
    ap.add_argument('--table-start',action='store_true',help='Use the original PGLib table start, without the power-flow initializer')
    ap.add_argument('--dense-certificate-control',action='store_true',help='Same algorithm with the original dense certificate implementation')
    ap.add_argument('--source-control',action='store_true',help='Same binary with r3p29r4 atom selection and retirement')
    ap.add_argument('--no-atom-guided',action='store_true')
    ap.add_argument('--no-atom-transport',action='store_true')
    ap.add_argument('--no-atom-step-decay',action='store_true')
    ap.add_argument('--atom-guided-after','--atom-plateau-batches',type=int,default=2)
    ap.add_argument('--atom-energy',action='store_true',help='Experimental QP-inspired energy support reduction')
    ap.add_argument('--atom-rayleigh',action='store_true',help='Experimental QP-inspired compensated-direction admission')
    ap.add_argument('--atom-decay',type=float,default=0.04,help='Atom weight retention; independent of the uniform control')
    ap.add_argument('--uniform-control',action='store_true',help='Same binary with parent uniform rho regularization')
    ap.add_argument('--native-control',action='store_true',help='Measure the same build with sparse recovery disabled')
    ap.add_argument('--display-level',type=int,choices=range(5),default=None,help='0 silent; 1 outer iterations; 2 numerical work and timing; 3 atoms/ranks; 4 full trace')
    ap.add_argument('--live-iterations',action='store_true',help='Stream both solver logs; enables the Newton iteration table')
    ap.add_argument('--live-curvature',action='store_true',help='Also report source-wise Newton curvature; implies --live-iterations')
    ap.add_argument('--live-iterations-every','--live-every',dest='live_every',type=int,default=1)
    ap.add_argument('--live-header-every',type=int,default=20)
    ap.add_argument('--ipopt-print-level',type=int,choices=range(13),default=None)
    ap.add_argument('--max-iter',type=int,default=1000)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--base',type=Path,default=Path('/data/projects/IPDCA_QCQP'))
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--warmups',type=int,default=1);ap.add_argument('--repetitions',type=int,default=5)
    ap.add_argument('--time-limit',type=float,default=1800)
    ap.add_argument('--target-seconds',type=float,default=4.0)
    args=ap.parse_args()
    if args.reuse_qcq and not args.qcq:ap.error('--reuse-qcq requires --qcq')
    if args.table_start and args.qcq:ap.error('--table-start requires --case; a frozen QCQP already contains its initial point')
    if not 0.0<args.atom_decay<1.0:ap.error('atom decay must lie strictly between zero and one')
    if not 0<=args.atom_guided_after<=14:ap.error("atom-guided-after must be between zero and fourteen")
    level=args.display_level if args.display_level is not None else (4 if args.live_curvature else 1 if args.live_iterations else 0)
    live=level>0
    if args.live_every<1 or args.live_header_every<1:ap.error('live intervals must be positive')
    ipopt_print_level=args.ipopt_print_level if args.ipopt_print_level is not None else (5 if live else 0)
    if args.max_iter<1 or args.threads<1 or args.warmups<0 or args.repetitions<1 or not finite(args.time_limit) or args.time_limit<=0 or not finite(args.target_seconds) or args.target_seconds<=0:ap.error('invalid counts or time limits')
    identity=json.loads((ROOT/'release_identity.json').read_text());branch=identity['branch']
    binary=args.binary.resolve(strict=True)
    banner=subprocess.check_output([str(binary),'--version'],text=True)
    if identity['build_id'] not in banner:ap.error('binary differs from release_identity.json')
    baseline=args.baseline_binary.resolve(strict=True) if args.baseline_binary else None
    if baseline:
        expected='0.4.28r3p29r7p1-lazy-columns'
        if expected not in subprocess.check_output([str(baseline),'--version'],text=True):ap.error('wrong parent binary')
    out=(args.output or args.base/'results'/('R3P29R8_'+(args.case or args.qcq).stem+'_' +str(time.time_ns()))).resolve()
    out.mkdir(parents=True,exist_ok=False);print('CAMPAIGN='+str(out),flush=True)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    shared_timing={}
    prepare_start=time.perf_counter()
    qcq=out/'problem.shared.qcq'
    if args.qcq:
        if args.reuse_qcq:qcq=args.qcq.resolve(strict=True)
        else:shutil.copyfile(args.qcq.resolve(strict=True),qcq)
        provenance={'qcq':str(args.qcq.resolve()),'sha256':sha(args.qcq)}
    else:
        case=args.case.resolve(strict=True)
        cmd=[sys.executable,str(ROOT/'python/export_pglib_qcqp_r3p23.py'),str(case),'--output',str(qcq),'--json',str(out/'export.json'),'--flow-limit','apparent','--power-flow-start','--no-power-flow-voltage-continuation']
        if args.table_start:cmd.remove('--power-flow-start')
        exported=run_logged(cmd,out/'export.log',env=env,live=level>=4)
        exported.check_returncode()
        provenance={'case':str(case),'sha256':sha(case),'initial_point':'pglib-table' if args.table_start else 'power-flow'}
    shared_timing['model_export_or_copy_seconds']=time.perf_counter()-prepare_start
    start=time.perf_counter();model=read_qcq(qcq)
    shared_timing['independent_model_read_seconds']=time.perf_counter()-start
    start=time.perf_counter();model_sha=sha(qcq)
    if not args.reuse_qcq:qcq.chmod(0o444)
    shared_timing['model_initial_hash_seconds']=time.perf_counter()-start
    profile=[]
    modes=([] if args.skip_ipopt else ['ipopt'])+(['baseline'] if baseline else [])+['newton']
    if args.native_control:modes+=['native-control']
    if args.uniform_control:modes+=['uniform-control']
    if args.source_control:modes+=['source-control']
    if args.dense_certificate_control:modes+=['dense-certificate-control']
    if args.centering_control:modes+=['centering-control']
    if args.unpacked_control:modes+=['unpacked-control']
    if args.packed_control:modes+=['packed-control']
    if args.dense_input_control:modes+=['dense-input-control']
    records=[]
    def run(mode,index,warmup):
        name=('warmup_' if warmup else 'run_')+str(index)+'_'+mode
        result_path=out/(name+'.json')
        if mode=='ipopt':
            cmd=[sys.executable,str(ROOT/'python/solve_qcq_ipopt_casadi_r3p23.py'),str(qcq),'--json',str(result_path),'--tol','1e-6','--constr-viol-tol','1e-6','--dual-inf-tol','1e-6','--compl-inf-tol','1e-6','--linear-solver','mumps','--print-level',str(ipopt_print_level),'--max-iter',str(args.max_iter),'--time-limit',str(args.time_limit)]
            if level==1:cmd.append('--no-runner-summary')
        else:
            cmd=build_command(baseline if mode=='baseline' else binary,qcq,result_path,profile,'newton',args.threads,args.time_limit)
            value(cmd,'--max-outer',args.max_iter)
            switch(cmd,'--lazy-newton-predictor-corrector','--no-lazy-newton-predictor-corrector',mode not in ('baseline','centering-control') and not args.no_predictor_corrector)
            if mode!='baseline':switch(cmd,'--lazy-newton-packed-gradients','--no-lazy-newton-packed-gradients',mode!='unpacked-control' and (args.packed_gradients or mode in ('packed-control','centering-control')))
            if mode=='dense-input-control':cmd.append('--dense-qcq-input')
            if mode=='native-control':cmd.append('--no-lazy-newton-sparse-recovery')
            if mode=='dense-certificate-control':cmd.append('--no-lazy-newton-sparse-certificate')
            if mode=='uniform-control':cmd.append('--lazy-newton-uniform-rho')
            if mode=='source-control':cmd+=['--no-lazy-newton-atom-guided','--no-lazy-newton-atom-transport','--no-lazy-newton-atom-step-decay']
            if mode not in ('baseline','uniform-control','source-control'):
                cmd+=['--lazy-newton-atom-guided-after',str(args.atom_guided_after)]
                if args.no_atom_guided:cmd.append('--no-lazy-newton-atom-guided')
                if args.no_atom_transport:cmd.append('--no-lazy-newton-atom-transport')
                if args.no_atom_step_decay:cmd.append('--no-lazy-newton-atom-step-decay')
            if mode!='baseline':cmd+=['--lazy-newton-atom-decay',str(args.atom_decay)]
            if args.atom_rayleigh and mode not in ('baseline','uniform-control','source-control'):cmd+=['--lazy-newton-atom-rayleigh','--no-lazy-newton-atom-guided']
            if args.atom_energy and mode not in ('baseline','uniform-control','source-control'):cmd.append('--lazy-newton-atom-energy')
            if mode!='baseline':cmd+=['--display-level',str(level)]
            elif live:cmd.append('--live-curvature' if level>=3 else '--live-iterations')
            if live:cmd+=['--live-iterations-every',str(args.live_every),'--live-header-every',str(args.live_header_every)]
        hash_start=time.perf_counter()
        record={'mode':mode,'index':index,'warmup':warmup,'command':cmd,'sha256_before':sha(qcq)}
        record['hash_before_seconds']=time.perf_counter()-hash_start
        if record['sha256_before']!=model_sha:raise ValueError('QCQP changed before run')
        start=time.perf_counter()
        try:
            r=run_logged(cmd,out/(name+'.log'),env=env,timeout=args.time_limit+60,live=live)
            record['returncode']=r.returncode
        except subprocess.TimeoutExpired:record['returncode']=-1
        record['process_seconds']=time.perf_counter()-start
        hash_start=time.perf_counter();record['sha256_after']=sha(qcq)
        record['hash_after_seconds']=time.perf_counter()-hash_start
        if record['sha256_after']!=model_sha:raise ValueError('QCQP changed during run')
        try:
            result_read_start=time.perf_counter();data=json.loads(result_path.read_text())
            sidecar=Path(str(result_path)+'.timing.json')
            if sidecar.exists():record['cli_timing']=json.loads(sidecar.read_text())
            record['result_read_seconds']=time.perf_counter()-result_read_start
            # Keep diagnostic evidence even when no valid endpoint exists.
            # Failed-run timings/objectives remain explicitly separate.
            record.update(solver_status=data.get('status'),solver_message=data.get('message'),
                          observed_iterations=data.get('iterations',data.get('outer_iterations')),
                          attempted_solver_seconds=data.get('solver_seconds',data.get('elapsed_seconds')))
            if mode!='ipopt':
                for k,v in data.items():
                    if k.startswith('lazy_newton_'):record[k]=v
            if record['returncode']!=0:raise ValueError('process failure: '+str(data.get('message')))
            if mode=='ipopt':
                record['ipopt_driver_timing']={k:data[k] for k in ('parse_seconds','model_build_seconds','solve_wall_seconds','total_wall_seconds') if k in data}
                if not data.get('success') or not data.get('kkt_passed') or data.get('model_sha256')!=model_sha:raise ValueError('Ipopt reference failed')
                objective=data['objective'];residuals=data['residuals'];seconds=data['solver_seconds'];iterations=data['iterations'];independent=True
                if any(not finite(v) or not 0<=v<=TOL for v in residuals.values()):raise ValueError('Ipopt residual failure')
            else:
                if mode!='baseline' and any(data.get(k)!=identity[k] for k in ('version','api_version','build_id')):raise ValueError('result identity mismatch')
                certificate_start=time.perf_counter()
                objective,residuals,independent=validate_endpoint(model,data,'lazy-newton')
                record['independent_certificate_seconds']=time.perf_counter()-certificate_start
                seconds=data['elapsed_seconds'];iterations=data['outer_iterations']
                for k,v in data.items():
                    if k.startswith('lazy_newton_') or k in ('inner_iterations','kkt_factorizations','dca_single_loop_factorizations','lazyqp_affine_elimination_fallbacks','model_value_calls','model_gradient_calls','multiplier_refit_attempts','restoration_total_seconds'):
                        record[k]=v
                if mode!='baseline':
                    validate_newton_mechanisms(data,mode)
                    record['backend_comparison_clean']=data.get('lazy_newton_pivoted_fallbacks',0)==0
            if not finite(objective) or not finite(seconds) or seconds<0:raise ValueError('nonfinite objective/time')
            record.update(valid=True,objective=objective,residuals=residuals,independent_certificate=independent,solver_seconds=seconds,iterations=iterations)
        except (ValueError,KeyError,OSError,TypeError) as e:record.update(valid=False,error=str(e))
        records.append(record);(out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(name,'PASS' if record['valid'] else 'FAIL',record.get('solver_seconds',record.get('error')),flush=True)
    for index in range(args.warmups):
        for mode in modes:run(mode,index+1,True)
    for index in range(args.repetitions):
        offset=index%len(modes)
        for mode in modes[offset:]+modes[:offset]:run(mode,index+1,False)
    references=[r['objective'] for r in records if r['mode']=='ipopt' and r['valid']]
    reference=statistics.median(references) if references else None
    for r in records:
        if r['valid'] and reference is not None:
            r['objective_gap_to_ipopt']=r['objective']-reference
            r['objective_matches_ipopt']=abs(r['objective']-reference)<=1e-5+1e-8*abs(reference)
    # A different local KKT point is not an invalid certificate. Keep solution
    # quality distinct from termination validity and from the speed target.
    summary={'schema':'lazyipdca-pglib-r3p29r8-v1','identity':identity,'model_sha256':model_sha,'provenance':provenance,'reused_qcq':args.reuse_qcq,'packed_gradients_requested':args.packed_gradients,'dimensions':[model.n,len(model.equalities),len(model.inequalities)],'tolerance':TOL,'live_output':live,'display_level':level,'live_curvature':level>=4,'shared_peripheral_seconds':shared_timing,'timing_contract':'COMPACT_NEWTON_R3P29R8.md','timings_include_live_diagnostics':live,'target_seconds':args.target_seconds,'environment':{k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')},'all_correct':all(r['valid'] for r in records),'objectives_match_ipopt':None if reference is None else all(r.get('objective_matches_ipopt',False) for r in records if r['valid']),'modes':{}}
    for mode in modes:
        rows=[r for r in records if r['mode']==mode and not r['warmup']];valid=[r for r in rows if r['valid']]
        result={'objectives':[r.get('objective') for r in rows],'pivoted_fallbacks':[r.get('lazy_newton_pivoted_fallbacks') for r in rows] if mode!='ipopt' else [],'measured_runs':len(rows),'valid_runs':len(valid),'all_certificates_independent':all(r.get('independent_certificate') for r in rows)}
        for field in ('atom_projection_seconds','atom_assembly_seconds','atom_product_seconds','atom_analysis_seconds','atom_update_seconds','atom_oracle_seconds','native_factor_seconds','native_solve_seconds','normal_recovery_factor_seconds','normal_recovery_solve_seconds','diagnostic_seconds'):
            key='lazy_newton_'+field
            if valid and all(key in r and finite(r[key]) for r in valid):
                result['median_'+field]=statistics.median(r[key] for r in valid)
        for field in ('core_solve_seconds','solver_wall_seconds'):
            key='lazy_newton_'+field
            if valid and all(key in r and finite(r[key]) for r in valid):result['median_'+field]=statistics.median(r[key] for r in valid)
        if valid and all('lazy_newton_cost_seconds' in r for r in valid):
            result['median_cost_seconds']={k:statistics.median(r['lazy_newton_cost_seconds'][k] for r in valid) for k in valid[0]['lazy_newton_cost_seconds']}
        if valid and all('cli_timing' in r for r in valid):
            result['median_cli_timing']={k:statistics.median(r['cli_timing'][k] for r in valid) for k,v in valid[0]['cli_timing'].items() if finite(v)}
        for field in ('hash_before_seconds','hash_after_seconds','result_read_seconds','independent_certificate_seconds'):
            if valid and all(field in r for r in valid):result['median_'+field]=statistics.median(r[field] for r in valid)
        result['solver_statuses']=[r.get('solver_status') for r in rows]
        result['observed_iterations']=[r.get('observed_iterations') for r in rows]
        for field in ('predictor_corrector_requested','predictor_corrector_initialization_guard','predictor_corrector_enabled','predictor_corrector_fallbacks','packed_gradients_enabled','atom_projection_builds','atom_projection_requests','atom_projection_updates','direction_corrections','direction_residual_evaluations','atom_adaptation_activated','atom_plateau_events','sparse_certificate_enabled','atom_guided_enabled','atom_guided_after','atom_guided_batches','atom_transport_enabled','atom_multiplier_transports','atom_step_decay_enabled','atom_oracle_calls','atom_oracle_solves','atom_oracle_failures','atom_tangent_witnesses','atom_maximum_tangent_error','atom_last_witness_coverage','atomic_regularization_enabled','atom_energy_sparsify_enabled','atom_rayleigh_gate_enabled','atom_sparsifications','atom_dropped_coordinates','atom_rayleigh_probes','atom_covered_rejections','atom_creations','atom_peak_active','atom_source_analyses','atom_completion_batches','objective_atoms','inequality_atoms','equality_atoms','completion_atoms','atom_final_rank','atom_ranks_exact','atom_coverage_observations','atom_last_probe_coverage','condensation_recoveries','promoted_inequalities','promoted_bounds','normal_recovery_attempts','normal_recovery_successes','normal_recovery_linear_solves','native_factor_successes','native_linear_solves','first_sparse_failure_iteration','last_sparse_failure_coordinate','last_native_iteration'):
            key='lazy_newton_'+field
            if rows and all(key in r for r in rows):result[field]=[r[key] for r in rows]
        if len(valid)==len(rows):
            result.update(median_solver_seconds=statistics.median(r['solver_seconds'] for r in rows),min_solver_seconds=min(r['solver_seconds'] for r in rows),max_solver_seconds=max(r['solver_seconds'] for r in rows),median_process_seconds=statistics.median(r['process_seconds'] for r in rows),iterations=[r['iterations'] for r in rows])
            result['target_met']=result['median_solver_seconds']<=args.target_seconds
        summary['modes'][mode]=result
    (out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    (out/'benchmark_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if level!=1:print(json.dumps(summary['modes'],indent=2),flush=True)
    print('CORRECTNESS='+('PASS' if summary['all_correct'] else 'FAIL'),flush=True)
    print('SUMMARY='+str(out/'benchmark_summary.json'),flush=True)
    return 0 if summary['all_correct'] else 1
if __name__=='__main__':raise SystemExit(main())
