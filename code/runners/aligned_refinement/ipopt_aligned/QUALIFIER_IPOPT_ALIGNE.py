#!/usr/bin/env python3
"""Four actual backend/strategy smoke solves with independent analytic KKT audit."""
from pathlib import Path
import argparse,hashlib,json,math,os,subprocess,time,sys
if sys.flags.optimize:
    raise RuntimeError("Qualification requires Python assertions enabled; -O/PYTHONOPTIMIZE refused")

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

TOY='''LAZYIPDCA_QCQP_V1
name aligned_active_inequality_fixed_variable
n 3
x0 0.1 0.1 0.25
lb -2 -2 0.25
ub 2 2 0.25
objective
quadratic
name f
constant 1.0625
linear -3 0 -1
hessian_nnz 3
0 0 2
1 1 2
2 2 2
end_quadratic
equalities 1
quadratic
name affine_balance
constant -1
linear 1 1 0
hessian_nnz 0
end_quadratic
inequalities 1
quadratic
name active_quadratic
constant -0.25
linear 0 0 0
hessian_nnz 1
0 0 2
end_quadratic
end_problem
'''

def audit(e):
    x=e['x'];nu=e['equality_multipliers'];lam=e['inequality_multipliers'];zl=e['lower_bound_multipliers'];zu=e['upper_bound_multipliers']
    assert len(x)==3 and len(nu)==len(lam)==1 and len(zl)==len(zu)==3
    assert all(math.isfinite(v)for a in (x,nu,lam,zl,zu)for v in a)
    g=x[0]*x[0]-.25;lo=[-2,-2,.25];hi=[2,2,.25]
    p=max(abs(x[0]+x[1]-1),max(0,g),*[max(0,lo[i]-x[i],x[i]-hi[i])for i in range(3)])
    grad=[2*x[0]-3+nu[0]+2*x[0]*lam[0],2*x[1]+nu[0],2*x[2]-1]
    d=max(*[abs(grad[i]-zl[i]+zu[i])for i in range(3)],max(0,-lam[0]),*[max(0,-v)for v in zl+zu])
    c=max(abs(lam[0]*g),*[abs(zl[i]*(x[i]-lo[i]))for i in range(3)],*[abs(zu[i]*(hi[i]-x[i]))for i in range(3)])
    assert p<=1e-6 and d<=1e-6 and c<=1e-6,(p,d,c)
    assert max(abs(x[0]-.5),abs(x[1]-.5),abs(x[2]-.25))<=1e-5
    assert abs(lam[0]-3)<=1e-5 and abs(zu[2]-zl[2]-.5)<=1e-6
    return dict(primal=p,dual=d,complementarity=c,strict=True,method='independent analytic original-QCQP audit; includes fixed variable and all bounds')

def options(e,b,s):
    expected=dict(tol=1e-8,constr_viol_tol=1e-6,dual_inf_tol=1e-6,compl_inf_tol=1e-6,acceptable_iter=0,bound_relax_factor=0,max_iter=5000,max_cpu_time=7200,max_wall_time=7200,option_file_name='',print_user_options='yes',mu_strategy=s,linear_solver=b)
    assert e['adapter_version']=='ipopt-aligned-original-qcqp-v1'
    assert e['reference_policy']=='aligned_original_kkt_tolerances_v1'
    assert e['aligned_contract_verified']is True and e['option_file_disabled']is True
    assert e['linear_solver']==b and e['mu_strategy']==s and e['ipopt_version']=='3.14.11'
    assert all(e['effective_options'].get(k)==v for k,v in expected.items())
    assert e['effective_options']['nlp_scaling_method']=='gradient-based'
    assert e['effective_options']['hessian_approximation']=='exact'
    assert e['effective_options']['fixed_variable_treatment']=='make_parameter'
    assert e['requested_mumps_ordering']==0
    assert e['optimization_wall_limit_seconds']==7200 and e['max_iterations']==5000
    assert e['available']is True and isinstance(e['ipopt_application_status_code'],int)
    assert 0<=e['iterations']<=5000 and 0<=e['elapsed_seconds']<=7200
    return True

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--local-mumps-only',action='store_true');a=p.parse_args()
    build_path=a.build.resolve();cfg=json.loads(build_path.read_text());out=a.out.resolve()
    if out.exists():p.error('Qualification output must be fresh')
    if cfg['schema']!='isolated-ipopt-aligned-options-build-v1':p.error('Wrong isolated build schema')
    if not a.local_mumps_only and not cfg['production_dependencies_verified']:p.error('Production qualification requires verified campaign dependencies')
    binary=Path(cfg['binary']);assert sha(binary)==cfg['binary_sha256']
    assert sha(cfg['ipopt_library'])==cfg['ipopt_library_sha256']
    if not a.local_mumps_only:assert cfg.get('hsl_library')and sha(cfg['hsl_library'])==cfg['hsl_library_sha256']
    out.mkdir(parents=True);problem=out/'input.qcq';problem.write_text(TOY);start=out/'raw_start.json';start.write_text('{"x0":[0.1,0.1,0.25]}\n')
    # This must be ignored despite Initialize being invoked from this directory.
    (out/'ipopt.opt').write_text('mu_strategy definitely_invalid\ntol 0.5\nlinear_solver unavailable_backend\nmax_iter 0\n')
    env=dict(os.environ);env.update(cfg['runtime_environment'])
    for name in cfg.get('unset_environment',[]):env.pop(name,None)
    rows=[];began=time.monotonic()
    for b in (['mumps']if a.local_mumps_only else ['ma27','mumps']):
        for strategy in ('adaptive','monotone'):
            name=b+'_'+strategy;endpoint=out/(name+'.json');command=[str(binary),'--problem',str(problem),'--frozen-start',str(start),'--json',str(endpoint),'--mu-strategy',strategy,'--linear-solver',b,'--max-iter','5000','--time-limit','7200','--print-level','5']
            if b=='ma27':command+=['--hsl-library',cfg['hsl_library']]
            r=subprocess.run(command,cwd=out,env=env,capture_output=True,text=True,timeout=120)
            log=out/(name+'.log');log.write_text(r.stdout+r.stderr)
            row=dict(backend=b,strategy=strategy,command=command,returncode=r.returncode,log=str(log),log_sha256=sha(log))
            try:
                assert r.returncode==0 and endpoint.is_file(),r.stdout[-1500:]+r.stderr
                e=json.loads(endpoint.read_text());options(e,b,strategy);cert=audit(e)
                # The actual OptimizeTNLP banner identifies the backend used.
                assert ('running with linear solver '+b)in r.stdout.lower()
                row.update(passed=True,options_verified=True,original_audit=cert,endpoint=str(endpoint),endpoint_sha256=sha(endpoint))
            except Exception as ex:row.update(passed=False,error=type(ex).__name__+': '+str(ex))
            rows.append(row);print('IPOPT_ALIGNED_SMOKE',name,'PASS'if row['passed']else'FAIL',flush=True)
    assert sha(binary)==cfg['binary_sha256']and sha(cfg['ipopt_library'])==cfg['ipopt_library_sha256']
    if cfg.get('hsl_library'):assert sha(cfg['hsl_library'])==cfg['hsl_library_sha256']
    for values in (cfg.get('linked_libraries',{}),cfg.get('hsl_linked_libraries',{})):
        for item in values.values():assert sha(item['path'])==item['sha256']
    for name,value in cfg.get('ipopt_headers_sha256',{}).items():assert sha(Path(cfg['ipopt_headers'])/name)==value
    source=Path(cfg.get('source_directory',Path(__file__).resolve().parent/'adapter'))
    for name,value in cfg['source_sha256'].items():assert sha(source/name)==value
    complete=not a.local_mumps_only and len(rows)==4 and all(r['passed']for r in rows)
    report=dict(schema='ipopt-aligned-qualification-v1',passed=complete,local_partial_passed=a.local_mumps_only and all(r['passed']for r in rows),cases=len(rows),runs=len(rows),build=str(build_path),build_sha256=sha(build_path),binary_sha256=cfg['binary_sha256'],options_verified=all(r.get('options_verified',False)for r in rows),original_audit_verified=all(r.get('original_audit',{}).get('strict',False)for r in rows),backends=sorted({r['backend']for r in rows}),strategies=['adaptive','monotone'],ipopt_version=cfg['ipopt_version'],production_dependencies_verified=cfg['production_dependencies_verified'],ambient_hostile_option_file_ignored=all(r['passed']for r in rows),input_sha256=sha(problem),start_sha256=sha(start),dependencies_rehashed_after_smokes=True,seconds=time.monotonic()-began,smokes=rows)
    (out/'QUALIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ('passed','local_partial_passed','runs','options_verified','original_audit_verified')},indent=2))
    return 0 if (complete or report['local_partial_passed'])else 2
if __name__=='__main__':raise SystemExit(main())
