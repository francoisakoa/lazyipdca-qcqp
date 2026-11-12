#!/usr/bin/env python3
"""Frozen-QCQP comparison: Ipopt, r3p27 control, sparse audits, tangent inertia.

Every measured endpoint is independently checked with the file coefficients
and the exported certificate multipliers. No tolerance is relaxed for timing.
"""
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

VERSION='0.4.28r3p27a1-lazy-equality-curvature'
BUILD_ID='v'+VERSION+'-api-4311'
TOL=1e-6
ROOT=Path(__file__).resolve().parents[1]
MODES=('ipopt','control','sparse','tangent')

def finite(v):
    return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_lazy(model,result,mode):
    if result.get('status')!='solved':
        raise ValueError(str(result.get('message','not solved')))
    if result.get('version')!=VERSION or result.get('build_id')!=BUILD_ID or result.get('api_version')!=4311:
        raise ValueError('incorrect solver identity')
    names=('certificate_equality_multipliers','certificate_inequality_multipliers',
           'certificate_lower_multipliers','certificate_upper_multipliers')
    vectors=[result.get(k) for k in names]
    sizes=(len(model.equalities),len(model.inequalities),model.n,model.n)
    for k,v,n in zip(names,vectors,sizes):
        if not isinstance(v,list) or len(v)!=n or not all(finite(x) for x in v):
            raise ValueError('invalid certificate vector '+k)
    equality,inequality,lower,upper=vectors
    if any(x<0 for v in (inequality,lower,upper) for x in v):
        raise ValueError('negative inequality/bound multiplier')
    x=result.get('x')
    if not isinstance(x,list) or len(x)!=model.n or not all(finite(v) for v in x):
        raise ValueError('invalid primal vector')
    residuals=model.residuals(x)
    residuals.update(model.kkt_residuals(x,equality+inequality,[u-l for u,l in zip(upper,lower)]))
    # Check the two bound multiplier sides separately as well.
    for j in range(model.n):
        if model.lb[j]==model.ub[j]:
            continue
        for b,z in ((model.lb[j],lower[j]),(model.ub[j],upper[j])):
            if math.isfinite(b):
                residuals['complementarity_inf']=max(residuals['complementarity_inf'],abs(z*(x[j]-b)))
            elif z!=0:
                raise ValueError('multiplier on an absent bound')
    for k,v in residuals.items():
        if not finite(v) or not 0<=v<=TOL:
            raise ValueError(f'independent original KKT failed: {k}={v}')
    objective=model.objective_value(x)
    if not finite(result.get('objective')) or abs(objective-result['objective'])>1e-7*max(1,abs(objective)):
        raise ValueError('objective mismatch with original file')
    for k in ('dual_infeasibility_inf','complementarity_inf','scaled_nlp_error'):
        if not finite(result.get(k)) or not 0<=result[k]<=TOL:
            raise ValueError('invalid reported KKT '+k)
    if result.get('dca_single_loop_full_refit_calls')!=1 or result.get('multiplier_refit_attempts')!=0:
        raise ValueError('single-loop final-only multiplier-fit contract violated')
    if result.get('lazy_dca_equality_retraction_attempts')!=0:
        raise ValueError('unexpected equality retraction')
    expected='tangent-inertia' if mode=='tangent' else 'condensed-retraction'
    if result.get('lazy_dca_equality_model')!=expected:
        raise ValueError('wrong equality model')
    if mode=='tangent' and result.get('equality_tangent_inertia_acceptances',0)<1:
        raise ValueError('tangent alternative never exercised')
    return objective,residuals

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--case',type=Path)
    source.add_argument('--qcq',type=Path)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--base',type=Path,default=Path('/data/projects/IPDCA_QCQP'))
    parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--time-limit',type=float,default=1800)
    parser.add_argument('--target-seconds',type=float,default=.2)
    args=parser.parse_args(argv)
    if args.repetitions<1 or args.warmups<0 or args.threads<1 or not finite(args.time_limit) or args.time_limit<=0 or not finite(args.target_seconds) or args.target_seconds<=0:
        parser.error('positive repetitions/threads/time limits required; warmups>=0')
    binary=args.binary.resolve(strict=True)
    banner=subprocess.check_output([str(binary),'--version'],text=True)
    if BUILD_ID not in banner: parser.error('the r3p27a1 binary is required')
    out=(args.output or args.base/'results'/('R3P27A1_CASE118_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    print('CAMPAIGN='+str(out),flush=True)
    env=dict(os.environ,OMP_NUM_THREADS=str(args.threads),OMP_DYNAMIC='FALSE',
             OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
    qcq=out/'case118.shared.qcq'
    if args.qcq:
        shutil.copyfile(args.qcq.resolve(strict=True),qcq)
        provenance={'input_qcq':str(args.qcq.resolve()),'input_sha256':digest(args.qcq)}
    else:
        case=args.case.resolve(strict=True)
        command=[sys.executable,str(ROOT/'python/export_pglib_qcqp_r3p23.py'),str(case),
                 '--output',str(qcq),'--json',str(out/'export.json'),'--flow-limit','apparent',
                 '--power-flow-start','--no-power-flow-voltage-continuation']
        with (out/'export.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=log,check=True)
        provenance={'case':str(case),'case_sha256':digest(case)}
    model=read_qcq(qcq)
    if (model.n,len(model.equalities),len(model.inequalities))!=(1088,980,979):
        raise ValueError('expected the unchanged 1088/980/979 case118 export')
    sha=digest(qcq);qcq.chmod(0o444)
    profile=json.loads((ROOT/'python/case118_profile_r3p27a1.json').read_text())
    records=[]
    def run(mode,index,warmup):
        stem=('warmup_' if warmup else 'run_')+str(index)+'_'+mode
        result_path=out/(stem+'.json')
        before=digest(qcq)
        if before!=sha: raise ValueError('model changed before '+stem)
        if mode=='ipopt':
            command=[sys.executable,str(ROOT/'python/solve_qcq_ipopt_casadi_r3p23.py'),str(qcq),
                     '--json',str(result_path),'--tol','1e-6','--constr-viol-tol','1e-6',
                     '--dual-inf-tol','1e-6','--compl-inf-tol','1e-6','--max-iter','3000',
                     '--time-limit',str(args.time_limit),'--linear-solver','mumps','--print-level','0']
        else:
            command=[str(binary),'--problem',str(qcq),'--json',str(result_path)]+profile+[
                '--time-limit',str(args.time_limit),'--quiet',
                '--lazyqp-affine-support-fastpath' if mode!='control' else '--no-lazyqp-affine-support-fastpath',
                '--dca-equality-model','tangent-inertia' if mode=='tangent' else 'condensed-retraction']
        record={'mode':mode,'index':index,'warmup':warmup,'command':command,'sha256_before':before}
        start=time.perf_counter()
        with (out/(stem+'.log')).open('w') as log:
            try:
                process=subprocess.run(command,env=env,stdout=log,stderr=log,timeout=args.time_limit+60)
                record['returncode']=process.returncode
            except subprocess.TimeoutExpired:
                record['returncode']=-1
        record['process_seconds']=time.perf_counter()-start
        record['sha256_after']=digest(qcq)
        if record['sha256_after']!=sha: raise ValueError('model changed after '+stem)
        try:
            result=json.loads(result_path.read_text())
            if record['returncode']!=0: raise ValueError('optimizer returned '+str(record['returncode'])+': '+str(result.get('message')))
            if mode=='ipopt':
                if not result.get('success') or not result.get('kkt_passed') or result.get('model_sha256')!=sha:
                    raise ValueError('Ipopt reference failed')
                residuals=result['residuals']
                if any(not finite(v) or v<0 or v>TOL for v in residuals.values()): raise ValueError('invalid Ipopt KKT')
                objective=result['objective'];seconds=result['solver_seconds'];iterations=result['iterations']
            else:
                objective,residuals=check_lazy(model,result,mode)
                seconds=result['elapsed_seconds'];iterations=result['outer_iterations']
                for k in ('inner_iterations','dca_single_loop_factorizations','equality_tangent_inertia_attempts',
                          'equality_tangent_inertia_acceptances','equality_tangent_stall_certificates',
                          'equality_model_seconds','primary_inner_seconds','primary_model_build_seconds','restoration_total_seconds'):
                    record[k]=result.get(k)
            if not finite(objective): raise ValueError('nonfinite objective')
            if not finite(seconds) or seconds<0: raise ValueError('invalid solve time')
            record.update(valid=True,objective=objective,residuals=residuals,solver_seconds=seconds,iterations=iterations)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            record.update(valid=False,error=str(exc))
        records.append(record)
        (out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(stem, 'PASS' if record['valid'] else 'FAIL', record.get('solver_seconds',record.get('error')),flush=True)
    for i in range(args.warmups):
        for mode in MODES:run(mode,i+1,True)
    for i in range(args.repetitions):
        order=MODES[i%4:]+MODES[:i%4]
        for mode in order:run(mode,i+1,False)
    reference=[r['objective'] for r in records if r['mode']=='ipopt' and r['valid']]
    if reference:
        fref=statistics.median(reference)
        for r in records:
            if r['valid'] and abs(r['objective']-fref)>1e-5+1e-8*abs(fref):
                r.update(valid=False,error='objective differs from Ipopt reference')
    else:
        for r in records:r.update(valid=False,error='no valid Ipopt reference')
    summary={'schema':'lazyipdca-case118-r3p27a1-v1','model_sha256':sha,'provenance':provenance,
             'dimensions':[1088,980,979],'tolerance':TOL,'target_seconds':args.target_seconds,
             'environment':{k:env[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS']},
             'all_correct':all(r['valid'] for r in records),'modes':{}}
    for mode in MODES:
        selected=[r for r in records if r['mode']==mode and not r['warmup']]
        valid=[r for r in selected if r['valid']]
        data={'valid_runs':len(valid),'measured_runs':len(selected)}
        if len(valid)==len(selected):
            data['median_solver_seconds']=statistics.median(r['solver_seconds'] for r in valid)
            data['median_process_seconds']=statistics.median(r['process_seconds'] for r in valid)
            data['iterations']=[r['iterations'] for r in valid]
            data['target_met']=data['median_solver_seconds']<=args.target_seconds
        summary['modes'][mode]=data
    (out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    (out/'benchmark_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary['modes'],indent=2),flush=True)
    print('CORRECTNESS='+('PASS' if summary['all_correct'] else 'FAIL'),flush=True)
    print('SUMMARY='+str(out/'benchmark_summary.json'),flush=True)
    return 0 if summary['all_correct'] else 1
if __name__=='__main__':
    raise SystemExit(main())
