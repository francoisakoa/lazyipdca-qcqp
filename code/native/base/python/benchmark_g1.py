#!/usr/bin/env python3
"""Compare explicit engines on one immutable QCQP; certify original coefficients."""
import argparse,gzip,json,os,shutil,statistics,sys,tempfile,time
from pathlib import Path
from benchmark_pglib_r3p29r8 import validate_endpoint,sha
from qcq_model_r3p23 import read_qcq
from live_process_r3p29r3 import run_logged
ROOT=Path(__file__).resolve().parents[1]

def command(binary,qcq,out,mode,args):
    common=[str(binary),'--problem',str(qcq),'--json',str(out),'--linear-backend','ldl','--qp-backend','ldl',
            '--feas-tol','1e-6','--opt-tol','1e-6','--dual-inf-tol','1e-6','--compl-inf-tol','1e-6','--quiet']
    if mode=='graft':
        common+=json.loads((ROOT/'python/case118_control_profile_r3p28.json').read_text())
        common+=['--inner-backend','lazyqp-sparse','--ln-ordering','metis']
    else:
        common+=['--algorithm','lazy-newton','--ln-ordering','metis']
        if mode!='parent':common+=['--newton-engine','legacy' if mode=='legacy' else 'atomic']
        if mode in ('atomic','uniform'):
            common+=['--lazy-newton-predictor-corrector','--kkt-berr-reject','1e-7','--display-level',str(args.display_level)]
            if mode=='uniform':common+=['--lazy-newton-uniform']
        elif args.display_level>0:common+=['--live-iterations']
    return common+['--max-outer',str(args.max_iter),'--time-limit',str(args.time_limit)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument('--qcq',type=Path);source.add_argument('--problem',choices=['case118','case300','case1354','case2869'])
    p.add_argument('--binary',required=True,type=Path);p.add_argument('--baseline-binary',type=Path)
    p.add_argument('--modes',default='legacy,atomic',help='comma list: legacy,atomic,uniform,parent,graft (graft may be slow)')
    p.add_argument('--output',type=Path);p.add_argument('--base',type=Path,default=Path('/data/projects/IPDCA_QCQP'))
    p.add_argument('--display-level',type=int,choices=range(5),default=0)
    p.add_argument('--warmups',type=int,default=1);p.add_argument('--repetitions',type=int,default=5)
    p.add_argument('--max-iter',type=int,default=1000);p.add_argument('--time-limit',type=float,default=1800)
    p.add_argument('--target-seconds',type=float,default=2);a=p.parse_args()
    modes=a.modes.split(',')
    if len(set(modes))!=len(modes) or any(m not in ('legacy','atomic','uniform','parent','graft') for m in modes):p.error('invalid or duplicate modes')
    if a.repetitions<1 or a.warmups<0 or a.max_iter<1 or a.time_limit<=0:p.error('invalid campaign limits')
    if 'parent' in modes and not a.baseline_binary:p.error('parent requires --baseline-binary')
    binary=a.binary.resolve(strict=True);baseline=a.baseline_binary.resolve(strict=True) if a.baseline_binary else None
    out=a.output or a.base/'results'/('G1_'+(a.problem or a.qcq.stem)+'_'+str(time.time_ns()));out.mkdir(parents=True,exist_ok=False)
    print('CAMPAIGN='+str(out),flush=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
    for key in ('LD_PRELOAD','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS'):env.pop(key,None)
    started=time.perf_counter();rows=[]
    with tempfile.TemporaryDirectory(prefix='g1_input_',dir=out) as tmp:
        qcq=Path(tmp)/((a.problem or a.qcq.stem)+'.qcq')
        if a.problem:
            frozen=ROOT/'validation_g1/frozen'
            with gzip.open(frozen/(a.problem+'.qcq.gz'),'rb') as src,qcq.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
            expected=json.loads((frozen/'models.json').read_text())[a.problem]['model_sha256']
            if sha(qcq)!=expected:raise RuntimeError('frozen QCQP SHA256 mismatch')
        else:qcq=a.qcq.resolve(strict=True)
        model_sha=sha(qcq);model=read_qcq(qcq);input_seconds=time.perf_counter()-started
        print('MODEL_SHA256='+model_sha,flush=True)
        for i in range(-a.warmups,a.repetitions):
            order=modes if i<0 else modes[i%len(modes):]+modes[:i%len(modes)]
            for mode in order:
                label=('warmup_'+str(i+a.warmups+1) if i<0 else 'run_'+str(i+1))+'_'+mode
                result=out/(label+'.json');cmd=command(baseline if mode=='parent' else binary,qcq,result,mode,a)
                row={'mode':mode,'label':label,'measured':i>=0,'command':cmd,'model_sha256':model_sha}
                t=time.perf_counter()
                try:
                    proc=run_logged(cmd,out/(label+'.log'),env=env,timeout=a.time_limit+5,live=a.display_level>0)
                    row['process_seconds']=time.perf_counter()-t
                    data=json.loads(result.read_text())
                    row.update(status=data.get('status'),message=data.get('message'),solver_seconds=data.get('elapsed_seconds'),iterations=data.get('outer_iterations'))
                    if proc.returncode:raise RuntimeError('process failure: '+str(data.get('message')))
                    val,res,independent=validate_endpoint(model,data,'lazy-newton')
                    row.update(valid=True,objective=val,residuals=res,certificate_independent=independent,
                               core_seconds=data.get('lazy_newton_core_solve_seconds'),read_seconds=data.get('cli_read_seconds'),
                               atom_peak_active=data.get('lazy_newton_atom_peak_active'),atom_creations=data.get('lazy_newton_atom_creations'),
                               atom_analysis_seconds=data.get('lazy_newton_atom_analysis_seconds'),pivoted_fallbacks=data.get('lazy_newton_pivoted_fallbacks'))
                except Exception as exc:row.update(valid=False,error=str(exc),process_seconds=row.get('process_seconds',time.perf_counter()-t))
                rows.append(row);(out/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(label, 'PASS' if row['valid'] else 'FAIL',row.get('solver_seconds',row.get('error')),flush=True)
        if sha(qcq)!=model_sha:raise RuntimeError('QCQP modified during campaign')
    summary={'model_sha256':model_sha,'model_name':a.problem or a.qcq.stem,'input_preparation_and_python_model_seconds':input_seconds,
             'timing_contract':'solver_seconds = solver wall including preparation and original certificate, excluding CLI read and Python audit; core reported separately',
             'target_seconds':a.target_seconds,'modes':{}}
    valid=True
    for mode in modes:
        rr=[r for r in rows if r['mode']==mode and r['measured']];good=[r for r in rr if r['valid']]
        s={'measured_runs':len(rr),'valid_runs':len(good),'all_certificates_independent':len(good)==len(rr) and all(r['certificate_independent'] for r in good)}
        if good:
            vals=[r['solver_seconds'] for r in good];med=statistics.median(vals)
            s.update(median_solver_seconds=med,min_solver_seconds=min(vals),max_solver_seconds=max(vals),median_process_seconds=statistics.median(r['process_seconds'] for r in good),
                     iterations=[r['iterations'] for r in good],objectives=[r['objective'] for r in good],atom_peak_active=[r['atom_peak_active'] for r in good],
                     target_met=len(good)==len(rr) and med<a.target_seconds)
        summary['modes'][mode]=s;valid=valid and len(good)==len(rr)
    valid=valid and all(r['valid'] for r in rows)
    objectives=[r['objective'] for r in rows if r['valid'] and r['measured']]
    # Agreement is reported, not used as a claim of global optimality.
    summary['objective_spread']=max(objectives)-min(objectives) if objectives else None
    summary['correctness_pass']=valid
    (out/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('SUMMARY='+str(out/'benchmark_summary.json'));print('CORRECTNESS='+('PASS' if valid else 'FAIL'))
    return 0 if valid else 1
if __name__=='__main__':raise SystemExit(main())
