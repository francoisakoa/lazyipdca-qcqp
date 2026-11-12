#!/usr/bin/env python3
"""Real MUMPS functional checks on two generic QCQPs, separate from timing.

Runs six fresh processes: reference/reuse/rebuild for a nonconvex inequality
and for a convex control. This is deliberately an explicit opt-in command.
It is not production MA27 qualification, an ASU run, or a speed experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REUSE_DIAGNOSTICS={'dca_filter_curvature_reuses','dca_filter_majorant_curvature_reuses'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core(value,exclude=()):
    if isinstance(value,dict):
        return {key:core(item,exclude) for key,item in value.items()
                if not key.endswith('_seconds') and key not in exclude}
    if isinstance(value,list):return [core(item,exclude) for item in value]
    if type(value)is float:return {'binary64':struct.pack('>d',value).hex()}
    return value


def read(path):return json.loads(Path(path).read_text())


def json_lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fixture(path,convex):
    def q(name,constant,linear,hessian):
        return '\n'.join(['quadratic','name '+name,'constant '+str(constant),
            'linear '+' '.join(map(str,linear)),'hessian_nnz '+str(len(hessian)),
            *[' '.join(map(str,entry)) for entry in hessian],'end_quadratic'])
    constant,diagonal,start,lower=(-1,2,[0,.5],[-4,-2]) if convex else (1,-2,[2,.5],[.1,-2])
    path.write_text('\n'.join(['LAZYIPDCA_QCQP_V1','name generic_curvature_cache_control',
        'n 2','x0 '+' '.join(map(str,start)),'lb '+' '.join(map(str,lower)),'ub 4 2','objective',
        q('objective',0,[1,0],[(1,1,2)]),'equalities 0','inequalities 1',
        q('constraint',constant,[0,0],[(0,0,diagonal)]),'end_problem'])+'\n')


def solve(binary,mode,model,folder,environment,display=0,require_majorants=True):
    folder.mkdir()
    command=[str(binary),'--problem',str(model),'--json',str(folder/'endpoint.json'),
        '--algorithm','lazy-dca','--dca-engine','majorants','--linear-backend','mumps','--linear-metis','1',
        '--native-al-fallback','off','--standard-r10-policy','off','--dca-rank-presolve','0',
        '--free-epigraph-presolve','0','--max-outer','200','--time-limit','30',
        '--termination-test','project','--feas-tol','1e-6','--opt-tol','1e-6',
        '--dual-inf-tol','1e-6','--compl-inf-tol','1e-6','--display-level',str(display)]
    if mode!='reference':command+=['--research-curvature-cache',mode,
        '--research-curvature-report',str(folder/'CURVATURE.jsonl'),
        '--research-curvature-trace',str(folder/'TRACE.jsonl')]
    (folder/'COMMAND.json').write_text(json.dumps(command,indent=2)+'\n')
    result=subprocess.run(command,capture_output=True,text=True,env=environment,timeout=40)
    (folder/'solver.log').write_text(result.stdout+result.stderr)
    if result.returncode!=0:raise ValueError('Functional fixture solve failed: '+str(folder))
    endpoint=read(folder/'endpoint.json')
    if require_majorants and endpoint.get('linear_solver_backend')!='inequality-majorants/mumps':raise ValueError('Actual MUMPS route absent')
    if endpoint.get('native_al_fallback',{}).get('attempted') is not False:raise ValueError('AL unexpectedly attempted')
    audit_command=[sys.executable,str(HERE/'support_chronos/support/AUDIT_ORIGINAL.py'),
        '--input',str(model),'--endpoint',str(folder/'endpoint.json'),'--out',str(folder/'AUDIT.json')]
    audit_process=subprocess.run(audit_command,capture_output=True,text=True,env=environment,timeout=30)
    (folder/'audit.log').write_text(audit_process.stdout+audit_process.stderr)
    audit=read(folder/'AUDIT.json')
    if audit_process.returncode!=0 or audit.get('strict_qcqp') is not True:
        raise ValueError('Original independent KKT audit is not strict: '+str(folder))
    return endpoint,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True,type=Path)
    parser.add_argument('--reference-binary',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    output=args.out.expanduser().resolve()
    if output.exists():raise ValueError('Fresh functional-test output required')
    if output==HERE or HERE in output.parents or output in HERE.parents:raise ValueError('Separate test output required')
    binary=args.binary.expanduser().resolve(strict=True);reference=args.reference_binary.expanduser().resolve(strict=True)
    output.mkdir(parents=True)
    environment={key:value for key,value in os.environ.items() if not key.startswith(('LAZY_','ELASTIC_','KERNEL_','IPDCA_'))}
    environment.update({key:'1' for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT'):environment.pop(key,None)
    rows=[]
    for label,convex in [('nonconvex',False),('convex',True)]:
        model=output/(label+'.qcq');fixture(model,convex);original=sha(model)
        values={mode:solve(reference if mode=='reference' else binary,mode,model,output/(label+'_'+mode),environment)
                for mode in ('reference','reuse','rebuild')}
        if sha(model)!=original:raise ValueError('Original fixture bytes changed')
        if core(values['reference'][0])!=core(values['reuse'][0]):raise ValueError('Default reuse changed reference numerics/work')
        if core(values['reuse'][0],REUSE_DIAGNOSTICS)!=core(values['rebuild'][0],REUSE_DIAGNOSTICS):
            raise ValueError('Rebuild changed non-cache endpoint/work fields')
        traces={mode:json_lines(output/(label+'_'+mode)/'TRACE.jsonl') for mode in ('reuse','rebuild')}
        raw_reports={mode:json_lines(output/(label+'_'+mode)/'CURVATURE.jsonl') for mode in ('reuse','rebuild')}
        reports={mode:[row for row in raw_reports[mode] if row.get('schema')=='research-curvature-cache-summary-v1']
                 for mode in ('reuse','rebuild')}
        for mode in ('reuse','rebuild'):
            sessions=[row for row in raw_reports[mode] if row.get('schema')=='research-curvature-cache-session-v1']
            if len(sessions)!=1 or sessions[0].get('solver_returned') is not True or sessions[0].get('mode')!=mode:
                raise ValueError('Completed native-session receipt absent')
            if sessions[0].get('model_instances')!=len(reports[mode]) or raw_reports[mode][-1]!=sessions[0]:
                raise ValueError('Model-instance count/session position differs')
        if not traces['reuse'] or core(traces['reuse'],{'mode'})!=core(traces['rebuild'],{'mode'}):
            raise ValueError('Matrix or trajectory trace differs')
        if len(reports['reuse'])!=len(reports['rebuild']) or not reports['reuse']:raise ValueError('Model-instance summaries absent')
        for mode in ('reuse','rebuild'):
            for row in reports[mode]:
                anchors=row['reanchors'];expected=1 if mode=='reuse' and anchors else anchors
                if row.get('complete') is not True or row['preparations']!=expected:
                    raise ValueError('Actual curvature preparation count differs from intervention')
                if row['reuse_hits']!=(max(0,anchors-1) if mode=='reuse' else 0):raise ValueError('Cache hit counter differs')
                if convex and row['corrected_rows']!=0:raise ValueError('Convex control falsely marked as corrected')
        if not convex and not any(row['reanchors']>1 and row['corrected_rows']>0 for row in reports['rebuild']):
            raise ValueError('Nonconvex fixture did not exercise repeated corrections')
        audit_keys=('objective','primal','dual','complementarity')
        if any(core({key:values[mode][1][key] for key in audit_keys})!=
               core({key:values['reference'][1][key] for key in audit_keys}) for mode in ('reuse','rebuild')):
            raise ValueError('Original numerical audit changed')
        rows.append(dict(fixture=label,input_sha256=original,all_original_KKT_strict=True,
            baseline_reuse_nonclock_endpoint_exact=True,reuse_rebuild_noncache_endpoint_exact=True,
            matrix_and_trajectory_trace_exact=True,trace_records=len(traces['reuse']),
            outer_iterations=values['reference'][0]['outer_iterations'],reports=reports))
    receipt=dict(schema='actual-native-curvature-cache-functional-check-v1',passed=True,
        binary=str(binary),binary_sha256=sha(binary),reference_binary=str(reference),reference_binary_sha256=sha(reference),
        optimizer_invocations=6,backend='mumps',production_MA27_qualification=False,
        production_timing_campaign_run=False,scope='Two generic fixtures only; exact intervention/trajectory and independent original KKT checks, no speed claim.',fixtures=rows)
    (output/'VALIDATION_NATIVE.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(receipt,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
