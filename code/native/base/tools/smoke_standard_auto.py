#!/usr/bin/env python3
"""Prepare or run three hand-derived fixtures and optional actual R3/HH argv.

All starts are fixed constants. The only optimizer process is the requested
native CLI. Endpoint certification independently evaluates the untouched
original QCQP with Fraction(binary64), including signed duals and both maximum
and sum complementarity products. The project criterion uses the maximum.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time


CASES = [
    dict(id='synthetic_no_slater_source_iq', n=1, x0=[0], lb=[0], ub=[1],
         objective=(0, [(0,-1)], []), eq=[], iq=[(0,[(0,1)],[])], expected=[0]),
    dict(id='synthetic_dependent_equalities', n=1, x0=[.25], lb=[0], ub=[2],
         objective=(0,[(0,1)],[]), eq=[(-1,[(0,1)],[]),(-2,[(0,2)],[])], iq=[], expected=[1]),
    dict(id='synthetic_bilinear_product', n=3, x0=[.25,.75,.1875], lb=[0,0,0], ub=[1,1,1],
         objective=(0,[(2,-1)],[]), eq=[(0,[(2,1)],[(0,1,-1)]),(-1,[(0,1),(1,1)],[])],
         iq=[], expected=[.5,.5,.25]),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def q_text(name, q):
    constant, linear, hessian = q
    lines = ['quadratic', 'name '+name, f'constant {constant:.17g}', 'linear_nnz '+str(len(linear))]
    lines += [f'{j} {v:.17g}' for j,v in linear]
    lines += ['hessian_nnz '+str(len(hessian))]+[f'{i} {j} {v:.17g}' for i,j,v in hessian]
    return lines+['end_quadratic']


def fixture(case):
    lines = ['LAZYIPDCA_QCQP_V2', 'name '+case['id'], 'n '+str(case['n'])]
    lines += [key+' '+' '.join(f'{v:.17g}' for v in case[key]) for key in ('x0','lb','ub')]
    lines += ['objective']+q_text('objective',case['objective'])
    for key,label in (('eq','equalities'),('iq','inequalities')):
        lines += [label+' '+str(len(case[key]))]
        for i,q in enumerate(case[key]):lines += q_text(label+'_'+str(i),q)
    return '\n'.join(lines+['end_problem'])+'\n'


def value(q, x):
    constant, linear, hessian = q
    result = F(constant)+sum((F(v)*x[j] for j,v in linear), F(0))
    for i,j,v in hessian:result += (F(1,2) if i==j else F(1))*F(v)*x[i]*x[j]
    return result


def gradient_add(q, x, weight, result):
    _,linear,hessian = q
    for j,v in linear:result[j] += weight*F(v)
    for i,j,v in hessian:
        result[i] += weight*F(v)*x[j]
        if i!=j:result[j] += weight*F(v)*x[i]


def audit(case, endpoint, tolerance):
    keys = ['x','certificate_equality_multipliers','certificate_inequality_multipliers',
            'certificate_lower_multipliers','certificate_upper_multipliers']
    dimensions = [case['n'],len(case['eq']),len(case['iq']),case['n'],case['n']]
    arrays=[]
    for key,size in zip(keys,dimensions):
        a=endpoint[key]
        if len(a)!=size or any(type(v) not in (int,float) or not math.isfinite(v) for v in a):
            raise ValueError('Malformed certificate array '+key)
        arrays.append(list(map(F,a)))
    x,y,z,zl,zu=arrays
    grad=[-l+u for l,u in zip(zl,zu)];gradient_add(case['objective'],x,F(1),grad)
    primal=F(0);products=[]
    for q,mult in zip(case['eq'],y):
        primal=max(primal,abs(value(q,x)));gradient_add(q,x,mult,grad)
    for q,mult in zip(case['iq'],z):
        g=value(q,x);primal=max(primal,g);products.append(abs(mult*g));gradient_add(q,x,mult,grad)
    for j in range(case['n']):
        primal=max(primal,F(case['lb'][j])-x[j],x[j]-F(case['ub'][j]))
        products.extend([abs(zl[j]*(x[j]-F(case['lb'][j]))),abs(zu[j]*(F(case['ub'][j])-x[j]))])
    stationarity=max(map(abs,grad),default=F(0));complementarity=sum(products,F(0))
    complementarity_max=max(products,default=F(0))
    min_dual=min(z+zl+zu,default=F(0))
    common=primal<=F(tolerance) and stationarity<=F(tolerance) and min_dual>=0
    strict=common and complementarity_max<=F(tolerance)
    return dict(method='independent exact Fraction(binary64) original QCQP',
                primal=float(primal),stationarity=float(stationarity),complementarity_sum=float(complementarity),
                complementarity_max=float(complementarity_max),
                minimum_signed_dual=float(min_dual),objective=float(value(case['objective'],x)),strict=bool(strict),
                strict_criterion='project: primal_inf, stationarity_inf, complementarity_max <= tolerance; signed duals >= 0',
                stronger_sum_pass=bool(common and complementarity<=F(tolerance)),
                tolerance=tolerance,expected_x=case['expected'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--profiles',type=Path,help='Also render the untouched actual campaign R3 and HH profiles on the bilinear fixture')
    ap.add_argument('--profile-contract',action='store_true',help='Include the actual campaign R3 and HH argv checks; requires --profiles')
    ap.add_argument('--seconds',type=float,default=30)
    ap.add_argument('--tolerance',type=float,default=1e-6)
    ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if args.profile_contract and not args.profiles:ap.error('--profile-contract requires --profiles')
    output=args.output.resolve();binary=args.binary.resolve();output.mkdir(parents=True,exist_ok=True)
    environment={k:v for k,v in os.environ.items() if not k.startswith(('LAZY_','ELASTIC_','KERNEL_'))}
    environment.update(OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
    profiles=json.loads(args.profiles.read_text())['profiles'] if args.profiles else {}
    cases=[]
    for case in CASES:
        folder=output/'fixtures';folder.mkdir(exist_ok=True)
        path=folder/(case['id']+'.qcq');path.write_text(fixture(case))
        start=folder/(case['id']+'.start.json');write_json(start,dict(x0=case['x0']))
        cases.append((case,'plain',path,start))
    if args.profile_contract:
        case=CASES[-1];path=output/'fixtures'/(case['id']+'.qcq');start=output/'fixtures'/(case['id']+'.start.json')
        cases.extend((case,mode,path,start) for mode in ('R3','HH'))
    results=[]
    for case,mode,path,start in cases:
        folder=output/(case['id']+'__'+mode)
        if (folder/'COMMAND.json').exists():raise SystemExit('Refusing to rerun existing smoke: '+str(folder))
        folder.mkdir(parents=True,exist_ok=True);endpoint=folder/'endpoint.json'
        if mode=='plain':
            command=[str(binary),'--problem',str(path),'--json',str(endpoint),
                '--max-outer','2000','--time-limit',str(args.seconds),
                '--termination-test','project','--acceptable-iter','0',
                '--display-level','2','--feas-tol',str(args.tolerance),'--opt-tol',str(args.tolerance),
                '--dual-inf-tol',str(args.tolerance),'--compl-inf-tol',str(args.tolerance)]
        else:
            mapping={'{binary}':str(binary),'{input}':str(path),'{endpoint}':str(endpoint),
                     '{backend}':'mumps','{seconds}':str(args.seconds),'{max_outer}':'2000'}
            command=[mapping.get(v,v) for v in profiles[mode]]
        command += ['--frozen-start',str(start)]
        receipt=dict(case=case['id'],mode=mode,argv=command,input_sha256=sha(path),start_sha256=sha(start),
            fixed_supplied_start=case['x0'],generated_initializer=False,external_optimizer=False,
            original_environment_overrides={},binary=str(binary),
            binary_sha256=sha(binary) if binary.is_file() else None,
            campaign_profiles_sha256=sha(args.profiles) if args.profiles else None,executed=args.run)
        write_json(folder/'COMMAND.json',receipt)
        if not args.run:results.append(receipt);continue
        began=time.monotonic()
        with (folder/'solver.log').open('w') as stream:
            try:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,
                                      timeout=args.seconds+15,cwd=folder)
                receipt['returncode']=result.returncode
            except subprocess.TimeoutExpired:receipt['returncode']='wall-timeout'
        receipt['seconds']=time.monotonic()-began
        log=(folder/'solver.log').read_text(errors='replace')
        receipt['standard_auto_lines']=[line for line in log.splitlines() if line.startswith('[STANDARD-AUTO]')]
        receipt['input_and_start_unchanged']=sha(path)==receipt['input_sha256'] and sha(start)==receipt['start_sha256']
        if endpoint.is_file():
            native=json.loads(endpoint.read_text());receipt['status']=native.get('status')
            try:receipt['original_kkt']=audit(case,native,args.tolerance)
            except (KeyError,ValueError,TypeError) as exc:receipt['audit_error']=str(exc)
        receipt['success']=receipt['input_and_start_unchanged'] and receipt.get('original_kkt',{}).get('strict',False)
        if case['id']=='synthetic_bilinear_product':
            receipt['auto_selected']=any('mode=native-R10-standard' in line for line in receipt['standard_auto_lines'])
            receipt['success']=receipt['success'] and receipt['auto_selected']
        write_json(folder/'RESULT.json',receipt);results.append(receipt)
        print(case['id'],mode,'PASS' if receipt['success'] else 'FAIL',round(receipt['seconds'],3),flush=True)
    summary=dict(schema='standard-auto-fixed-start-native-smokes-v1',executed=args.run,cases=results,
                 success=all(item.get('success',True) for item in results),optimizer_processes=len(results) if args.run else 0)
    write_json(output/'SUMMARY.json',summary)
    if args.run:raise SystemExit(0 if summary['success'] else 1)
    print('SMOKE_PLAN',output/'SUMMARY.json',flush=True)


if __name__=='__main__':main()
