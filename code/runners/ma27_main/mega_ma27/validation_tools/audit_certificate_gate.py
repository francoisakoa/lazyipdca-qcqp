#!/usr/bin/env python3
"""Pre-campaign qualification of original certificate auditing and frozen bounds.

Synthetic analytic certificates test the delivered audit_worker and tri-state
contract for every campaign method. No numerical optimizer is invoked and no
synthetic result is a campaign certificate. Frozen coefficients and raw starts
are inspected and hashed without rewriting their bytes.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
METHODS = ('R3','HH','HS','SH','SS','SS_legacy','ipopt_qcqp','ipopt_mumps')
TOLERANCE = 1e-6
EXPECTED_CASES = 4078
SCHEMA = 'original-certificate-audit-and-frozen-bound-gate-v1'
REPORT = Path('validation/AUDIT_CERTIFICATE_GATE.json')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def installed_sources(root):
    paths=('validation_tools/original_auditor.py','campaign/common/original_auditor.py','validation_tools/precise_original_auditor.py',
           'campaign/audit_worker.py','campaign/observation_contract.py','validation_tools/audit_certificate_gate.py','validation_tools/AUDITOR_IDENTITY.json')
    return {relative:sha(root/relative) for relative in paths}


def inventory(root,prepared=None):
    if prepared is None:
        for name in ('FROZEN_INVENTORY.json','PREPARATION.json'):
            if (root/name).is_file():
                prepared=json.loads((root/name).read_text());break
        else:
            raise ValueError('Frozen inventory required for certificate audit qualification')
    if not isinstance(prepared,dict) or not isinstance(prepared.get('cases'),list):
        raise ValueError('Frozen inventory cases must be a list')
    return prepared['cases']


def inventory_binding(cases):
    fields=('id','input','input_sha256','raw_x0_line_sha256','start_file','start_file_sha256',
            'problem_file','problem_sha256')
    return digest([{key:case[key] for key in fields if key in case} for case in cases])


def resolve_input(root,value):
    path=Path(value).expanduser()
    path=path if path.is_absolute() else root/'campaign'/path
    if path.is_symlink():
        raise ValueError('Symlink frozen file refused: '+str(path))
    return path.resolve(strict=True)


def check_frozen_header(path,expected_x0=None):
    with path.open('rb') as stream:
        lines=[stream.readline() for _ in range(6)]
    if lines[0].strip() not in (b'LAZYIPDCA_QCQP_V1',b'LAZYIPDCA_QCQP_V2'):
        raise ValueError('Frozen QCQP V1/V2 header required')
    parsed=[]
    for line,key in zip(lines[1:],(b'name',b'n',b'x0',b'lb',b'ub')):
        parts=line.split()
        if not parts or parts[0]!=key:
            raise ValueError('Missing frozen header field: '+key.decode())
        parsed.append(parts[1:])
    if len(parsed[1])!=1:
        raise ValueError('Invalid frozen dimension')
    n=int(parsed[1][0])
    if n<1:
        raise ValueError('Invalid frozen dimension')
    x0,lower,upper=(list(map(float,parts)) for parts in parsed[2:])
    if any(len(values)!=n for values in (x0,lower,upper)):
        raise ValueError('Frozen bound/start dimensions differ')
    if not all(math.isfinite(value) for value in x0):
        raise ValueError('Frozen raw start contains a nonfinite coordinate')
    if any(math.isnan(value) for values in (lower,upper) for value in values):
        raise ValueError('Frozen bound contains NaN')
    if any(math.isfinite(lo) and math.isfinite(hi) and lo>hi for lo,hi in zip(lower,upper)):
        raise ValueError('Frozen finite lower bound exceeds finite upper bound')
    x0_sha=hashlib.sha256(lines[3]).hexdigest()
    if expected_x0 is not None and expected_x0!=x0_sha:
        raise ValueError('Frozen raw x0 line differs from imported contract')
    return dict(n=n,raw_x0_line_sha256=x0_sha,
        absent_lower=sum(math.isinf(lo) for lo in lower),absent_upper=sum(math.isinf(hi) for hi in upper),
        noncanonical_lower_sentinels=sum(lo==math.inf for lo in lower),
        noncanonical_upper_sentinels=sum(hi==-math.inf for hi in upper),
        finite_bound_order_verified=True,raw_start_unchanged=True)


def preflight(root,cases,require_complete=True):
    if require_complete and len(cases)!=EXPECTED_CASES:
        raise ValueError('Complete frozen corpus required: 4078 cases')
    if len({case['id'] for case in cases})!=len(cases):
        raise ValueError('Duplicate frozen case identifier')
    records=[];seen={};started=time.monotonic()
    print('AUDIT_BOUND_PREFLIGHT',0,'/',len(cases),flush=True)
    for number,case in enumerate(cases,1):
        path=resolve_input(root,case['input']);expected=case['input_sha256']
        stat=path.stat();signature=(str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,expected)
        if signature not in seen:
            if sha(path)!=expected:
                raise ValueError('Frozen QCQP hash differs: '+case['id'])
            seen[signature]=check_frozen_header(path)
        header=seen[signature]
        if case.get('raw_x0_line_sha256') is not None and header['raw_x0_line_sha256']!=case['raw_x0_line_sha256']:
            raise ValueError('Frozen raw x0 line differs: '+case['id'])
        for field,key in (('start_file','start_file_sha256'),('problem_file','problem_sha256')):
            if case.get(field):
                extra=resolve_input(root,case[field])
                if not case.get(key) or sha(extra)!=case[key]:
                    raise ValueError('Frozen auxiliary input/start hash differs: '+case['id'])
        records.append(dict(id=case['id'],input_sha256=expected,**header))
        if number%100==0 or number==len(cases):
            print('AUDIT_BOUND_PREFLIGHT',number,'/',len(cases),flush=True)
    for signature in seen:
        path=Path(signature[0]);stat=path.stat()
        if (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)!=signature[1:6]:
            raise ValueError('Frozen input changed during bound preflight: '+str(path))
    return dict(cases_checked=len(records),unique_input_versions_checked=len(seen),
        noncanonical_bound_cases=sum(bool(r['noncanonical_lower_sentinels'] or r['noncanonical_upper_sentinels']) for r in records),
        cases_with_absent_bounds=sum(bool(r['absent_lower'] or r['absent_upper']) for r in records),
        finite_bound_order_verified=True,raw_start_bytes_and_hashes_verified=True,
        coefficients_or_starts_rewritten=False,seconds=time.monotonic()-started,records=records)


def quad(constant=0,linear=None,hessian=None):
    return (float(constant),list(linear or []),list(hessian or []))


def model_bytes(lower='inf',upper='-inf',obj=None,eq=(),ineq=()):
    obj=obj or quad(hessian=[(0,0,2)])
    def block(q,name):
        c,lin,hes=q
        return '\n'.join(['quadratic','name '+name,'constant '+str(c),'linear_nnz '+str(len(lin))]
            +[f'{j} {v}' for j,v in lin]+['hessian_nnz '+str(len(hes))]
            +[f'{i} {j} {v}' for i,j,v in hes]+['end_quadratic'])+'\n'
    text=f'LAZYIPDCA_QCQP_V2\nname analytic_auditor_gate\nn 1\nx0 0\nlb {lower}\nub {upper}\nobjective\n'
    text+=block(obj,'objective')+'equalities '+str(len(eq))+'\n'
    text+=''.join(block(q,'equality') for q in eq)+'inequalities '+str(len(ineq))+'\n'
    text+=''.join(block(q,'inequality') for q in ineq)+'end_problem\n'
    return text.encode()


def certificate_fixtures(root):
    worker=module(root/'campaign/audit_worker.py','_prelaunch_original_audit_worker')
    contract=module(root/'campaign/observation_contract.py','_prelaunch_original_observation_contract')
    scenarios=[]
    for lo in ('inf','-inf'):
        for hi in ('inf','-inf'):
            scenarios.append(dict(name='absent_bounds_'+lo+'_'+hi,input=model_bytes(lo,hi),x=0.0,expected=True))
    eq=quad(-1,hessian=[(0,0,2)])
    scenarios.extend([
        dict(name='stationarity_violation',input=model_bytes(),x=1.0,expected=False,check='dual'),
        dict(name='active_quadratic_equality',input=model_bytes('0','2',eq=[eq]),x=1.0,y=[-1.0],expected=True),
        dict(name='equality_primal_violation',input=model_bytes('0','3',eq=[eq]),x=2.0,y=[-1.0],expected=False,check='primal'),
        dict(name='active_quadratic_inequality',input=model_bytes('0','2',obj=quad(linear=[(0,-1)]),ineq=[eq]),x=1.0,z=[0.5],expected=True),
        dict(name='inequality_primal_violation',input=model_bytes('0','3',obj=quad(linear=[(0,-1)]),ineq=[eq]),x=2.0,z=[0.25],expected=False,check='primal'),
        dict(name='inequality_complementarity_violation',input=model_bytes('0','2',obj=quad(),ineq=[eq]),x=0.0,z=[0.5],expected=False,check='complementarity'),
        dict(name='inequality_dual_sign_violation',input=model_bytes('0','2',obj=quad(linear=[(0,1)]),ineq=[eq]),x=1.0,z=[-0.5],expected=False,check='dual'),
        dict(name='active_lower_bound',input=model_bytes('2','-inf',obj=quad(linear=[(0,1)])),x=2.0,zl=1.0,expected=True),
        dict(name='active_upper_bound',input=model_bytes('inf','3',obj=quad(linear=[(0,-1)])),x=3.0,zu=1.0,expected=True),
        dict(name='bound_complementarity_violation',input=model_bytes('0','2',obj=quad(linear=[(0,1)])),x=1.0,zl=1.0,expected=False,check='complementarity'),
        dict(name='absent_lower_multiplier_violation',input=model_bytes(obj=quad(linear=[(0,1)])),x=0.0,zl=1.0,expected=False,check='dual'),
        dict(name='finite_bound_inversion',input=model_bytes('2','1'),x=0.0,expected=None),
        dict(name='NaN_bound',input=model_bytes('nan','inf'),x=0.0,expected=None),
        dict(name='incomplete_certificate',input=model_bytes(),x=0.0,missing_multiplier=True,expected=None),
        dict(name='strict_residuals_budget_exceeded',input=model_bytes(),x=0.0,clock=7201.0,expected=None),
        dict(name='strict_residuals_signal_termination',input=model_bytes(),x=0.0,returncode=-15,expected=None),
        dict(name='strict_residuals_bad_route_or_backend',input=model_bytes(),x=0.0,route_pass=False,expected=None),
    ])
    records=[];started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='qcqp-auditor-gate-') as folder:
        folder=Path(folder)
        scalar=root/'campaign/common/original_auditor.py';precise=root/'validation_tools/precise_original_auditor.py'
        write(folder/'INSTALLATION.json',dict(native_auditor=str(scalar),native_auditor_sha256=sha(scalar),
            precise_auditor=str(precise),precise_auditor_sha256=sha(precise)))
        worker.ROOT=folder
        print('AUDIT_CERTIFICATE_FIXTURES',0,'/',len(scenarios)*len(METHODS),flush=True)
        for scenario in scenarios:
            inp=folder/'input.qcq';inp.write_bytes(scenario['input'])
            for method in METHODS:
                native=method not in ('ipopt_qcqp','ipopt_mumps')
                endpoint=dict(x=[scenario['x']],status='internal_status_ignored_by_certificate',
                    elapsed_seconds=scenario.get('clock',0.01),outer_iterations=0,restoration_iterations=0,
                    certificate_equality_multipliers=scenario.get('y',[]),certificate_inequality_multipliers=scenario.get('z',[]),
                    certificate_lower_multipliers=[scenario.get('zl',0.0)],certificate_upper_multipliers=[scenario.get('zu',0.0)],
                    native_al_fallback=dict(policy='off',enabled=False,attempted=False,accepted=False,endpoint_origin='lazy'))
                if not native:
                    endpoint.update(solver='Ipopt',core_s=scenario.get('clock',0.01),iterations=0,max_iterations=5000,optimization_wall_limit_seconds=7200)
                    for old,new in (('certificate_equality_multipliers','equality_multipliers'),('certificate_inequality_multipliers','inequality_multipliers'),
                                    ('certificate_lower_multipliers','lower_bound_multipliers'),('certificate_upper_multipliers','upper_bound_multipliers')):
                        endpoint[new]=endpoint.pop(old)
                if scenario.get('missing_multiplier'):
                    endpoint.pop('certificate_lower_multipliers' if native else 'lower_bound_multipliers')
                case=dict(id='analytic_auditor_gate',method=method,input=str(inp),input_sha256=sha(inp),
                    seconds=7200,max_outer=5000,restoration_max_iterations=600,solve_process_seconds=0.1,solve_external_timeout=False)
                write(folder/'case.json',case);write(folder/'endpoint.json',endpoint)
                audit=worker.run(folder/'case.json',folder/'endpoint.json',folder/'AUDIT.json')
                work=dict(returncode=scenario.get('returncode',0),interrupted=False,external_timeout=False,seconds=0.1)
                record=dict(method=method,work=work,audit_work=dict(returncode=0,interrupted=False,external_timeout=False),
                    audit=audit,route_policy_check=dict(passed=scenario.get('route_pass',True)),solution_origin='lazy' if native else 'ipopt')
                normalized=contract.normalize_result(record,integrity_verified=True)
                decision=normalized['strict_qcqp']
                if decision is not scenario['expected']:
                    raise ValueError('Certificate audit gate failed: '+scenario['name']+'/'+method+' expected '+str(scenario['expected'])+' got '+str(decision))
                if scenario.get('check') and not (audit.get(scenario['check'],0)>TOLERANCE):
                    raise ValueError('Certificate audit gate did not detect original '+scenario['check'])
                if decision is not None and audit.get('precision_agreement') is not True:
                    raise ValueError('Scalar/Decimal150 agreement absent in certificate gate')
                records.append(dict(scenario=scenario['name'],method=method,expected=scenario['expected'],
                    strict_original_qcqp=decision,primal=audit.get('primal'),dual=audit.get('dual'),complementarity=audit.get('complementarity'),
                    precision_agreement=audit.get('precision_agreement'),indeterminate_reasons=normalized['observation_indeterminate_reasons']))
            print('AUDIT_CERTIFICATE_FIXTURES',len(records),'/',len(scenarios)*len(METHODS),flush=True)
    return dict(tests=len(records),scenarios=len(scenarios),methods=list(METHODS),passed=True,
        numerical_solver_invocations=0,scope='Synthetic analytic auditor qualification, not numerical solver/profile or campaign qualification',
        true=sum(r['strict_original_qcqp'] is True for r in records),false=sum(r['strict_original_qcqp'] is False for r in records),
        indeterminate=sum(r['strict_original_qcqp'] is None for r in records),seconds=time.monotonic()-started,records=records)


def qualify(root,prepared=None,require_complete=True):
    root=Path(root).resolve(strict=True);started=time.monotonic()
    print('AUDIT_CERTIFICATE_GATE START',flush=True)
    report=dict(schema=SCHEMA,pass_=False,**{'pass':False},created_utc=datetime.now(timezone.utc).isoformat(),
        original_tolerance=TOLERANCE,coefficient_or_start_changes=False,numerical_solver_invocations=0,
        full_six_profile_campaign_qualification=False)
    try:
        sources=installed_sources(root);cases=inventory(root,prepared)
        identity=json.loads((root/'validation_tools/AUDITOR_IDENTITY.json').read_text())
        if (identity.get('scalar_sha256')!=sources['validation_tools/original_auditor.py']
                or sources['campaign/common/original_auditor.py']!=identity.get('scalar_sha256')
                or identity.get('precise_sha256')!=sources['validation_tools/precise_original_auditor.py']):
            raise ValueError('Auditor identity does not match installed scalar/precise sources')
        report.update(auditor_source_sha256=sources,inventory_binding_sha256=inventory_binding(cases))
        report['certificate_fixtures']=certificate_fixtures(root)
        report['frozen_preflight']=preflight(root,cases,require_complete)
        if installed_sources(root)!=sources:
            raise ValueError('Auditor source changed during qualification')
        report.update(pass_=True,**{'pass':True},seconds=time.monotonic()-started)
        write(root/REPORT,report)
        print('AUDIT_CERTIFICATE_GATE PASS',len(cases),'cases;',report['certificate_fixtures']['tests'],'analytic checks;0 solver invocations',flush=True)
        return report
    except Exception as error:
        report.update(error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-started)
        write(root/REPORT,report)
        print('AUDIT_CERTIFICATE_GATE FAIL',str(error),flush=True)
        raise


def verify_receipt(root,prepared=None):
    root=Path(root).resolve(strict=True)
    report=json.loads((root/REPORT).read_text())
    if report.get('schema')!=SCHEMA or report.get('pass') is not True or report.get('original_tolerance')!=TOLERANCE:
        raise ValueError('Passing original certificate audit gate receipt required')
    if report.get('auditor_source_sha256')!=installed_sources(root):
        raise ValueError('Qualified auditor source differs from receipt')
    cases=inventory(root,prepared)
    if report.get('inventory_binding_sha256')!=inventory_binding(cases) or report.get('frozen_preflight',{}).get('cases_checked')!=len(cases):
        raise ValueError('Qualified frozen input inventory differs from receipt')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--verify-receipt',action='store_true')
    args=parser.parse_args()
    report=verify_receipt(args.root) if args.verify_receipt else qualify(args.root)
    print(json.dumps(dict(passed=report['pass'],cases=report['frozen_preflight']['cases_checked'],
        analytic_checks=report['certificate_fixtures']['tests'],numerical_solver_invocations=0)),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
