#!/usr/bin/env python3
"""Inspect every frozen I/M model with the actual native structural inspector.

No numerical solve is requested. Missing or incomplete post-presolve coverage
blocks the factorial experiment. The independent all-corpus Ipopt run remains
separate from this gate.
"""
from __future__ import annotations
import argparse,collections,datetime,gzip,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import PROFILES

HERE=Path(__file__).resolve().parent
SCHEMA='factorial-corpus-coupling-gate-v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)
def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def common():
    for p in (HERE.parent/'CAMPAGNE.py',HERE.parent/'default_relaunch/CAMPAGNE.py'):
        if p.is_file():
            spec=importlib.util.spec_from_file_location('coupling_common',p)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
    raise ValueError('Shared frozen-input verifier CAMPAGNE.py unavailable')
def input_installation(path,root,contract_path):
    """Load the complete hash-bound path map, never reconstructing filenames."""
    path=Path(path).resolve();root=Path(root).resolve();contract_path=Path(contract_path).resolve()
    cfg=load(path);expected_contract=sha(contract_path)
    if cfg.get('schema')!='ipopt-defaults-full-corpus-installation-v1':
        raise ValueError('Explicit qualified default input installation required')
    if Path(cfg.get('root','')).resolve()!=root:raise ValueError('Input installation root differs')
    if cfg.get('contract_sha256')!=expected_contract:raise ValueError('Input installation corpus differs')
    installed_contract=Path(cfg.get('contract',''))
    if not installed_contract.is_file() or sha(installed_contract)!=expected_contract:
        raise ValueError('Installed input corpus file is absent or differs')
    cases=load(contract_path)['cases'];case_map={c['case_id']:c for c in cases}
    if len(cases)!=4078 or len(case_map)!=4078:
        raise ValueError('All 4078 unique frozen cases are required in input installation')
    bindings=cfg.get('frozen_bindings',{})
    if not isinstance(bindings,dict) or set(bindings)!=set(case_map):
        raise ValueError('Explicit frozen path bindings must cover exactly all 4078 cases')
    for ident,case in case_map.items():
        bound_input(cfg,case)
    cfg['_input_installation_path']=str(path)
    cfg['_input_installation_sha256']=sha(path)
    return cfg

def bound_input(cfg,case):
    ident=case['case_id'];binding=cfg.get('frozen_bindings',{}).get(ident)
    if not isinstance(binding,dict) or not isinstance(binding.get('input'),str) or not binding['input']:
        raise ValueError('Explicit frozen input path absent: '+ident)
    if binding.get('input_sha256')!=case['input_sha256']:
        raise ValueError('Explicit frozen input hash differs: '+ident)
    path=Path(binding['input'])
    if not path.is_absolute():raise ValueError('Explicit frozen input path must be absolute: '+ident)
    return path.resolve()

def build_identity(binary):
    candidates=[binary.parent/'BUILD_NATIVE.json',binary.parent.parent/'BUILD_NATIVE.json']
    for p in candidates:
        if not p.is_file():continue
        d=load(p)
        if d.get('binary_sha256')==sha(binary):
            expected=d.get('source_sha256');source=Path(d.get('source',''))
        elif d.get('research_binary_sha256')==sha(binary):
            expected=d.get('research_source_sha256');source=Path(d.get('source_directories',{}).get('research',HERE/'source'))
        else:continue
        if not isinstance(expected,dict) or not expected or not source.is_dir():continue
        actual={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()}
        if expected!=actual:raise ValueError('Research source differs from homogeneous native build')
        if not (d.get('homogeneous_compilation') is True or d.get('all_binaries_homogeneous') is True):
            raise ValueError('Homogeneous native build proof unavailable')
        return dict(research_source_sha256=actual,source_root=str(source.resolve()),
                    source_build_proof=str(p.resolve()),source_build_proof_sha256=sha(p))
    raise ValueError('No matching source-bound homogeneous BUILD_NATIVE.json beside the research binary')
def ma27_runtime_binding(binary,build_proof=None):
    candidates=[Path(build_proof)]if build_proof is not None else[binary.parent.parent/'BUILD_NATIVE.json']
    for path in candidates:
        if not path.is_file():continue
        proof=load(path)
        if proof.get('schema')!='lazy-factorial-three-native-builds-ma27-v2'or proof.get('native_backend')!='ma27'or proof.get('research_binary_sha256')!=sha(binary):continue
        if Path(proof['research_binary']).resolve()!=binary.resolve()or proof.get('mumps_compiled')is not False:raise ValueError('Coupling inspector MA27 binary route differs')
        from MA27_RUNTIME import verify_pin
        runtime_path=Path(proof['hsl_runtime_proof'])
        if sha(runtime_path)!=proof.get('hsl_runtime_proof_sha256'):raise ValueError('MA27 HSL runtime proof changed')
        runtime=verify_pin(load(runtime_path))
        if runtime['hsl_library_sha256']!=proof.get('hsl_library_sha256')or Path(runtime['hsl_library']).resolve()!=Path(proof['hsl_library']).resolve():raise ValueError('MA27 inspector HSL binding differs')
        environment=proof.get('runtime_environment',{})
        if environment.get('HSL_LIBRARY')!=proof['hsl_library']:raise ValueError('MA27 inspector runtime environment lacks the bound HSL library')
        return dict(native_backend='ma27',hsl_library=proof['hsl_library'],hsl_library_sha256=proof['hsl_library_sha256'],runtime_environment=environment,native_build_proof=str(path.resolve()),native_build_proof_sha256=sha(path))
    raise ValueError('Matching fresh MA27 runtime/build proof is required for coefficient inspection')

def inspect(case,root,binary,out,runner,timeout,input_config=None,runtime_config=None):
    cid=case['case_id'];folder=out/'inspections'/cid
    folder.mkdir(parents=True)
    payload=folder/'inspection.json';log=folder/'inspection.log'
    row={'case_id':cid,'dataset':case['dataset'],'regime':case['regime'],
         'input_sha256':case['input_sha256'],'start_sha256':case['start_binding']['start_sha256'],
         'passed':False,'signed_square_changed':None,'unresolved_structural_phase':True,
         'inspection_sha256':None,'missing':False,'inspection_file':str(payload.relative_to(out))}
    try:
        cfg=input_config if input_config is not None else {'root':str(root)}
        if input_config is not None:bound_input(cfg,case)
        inp,start=runner.verify_input(cfg,case)
        row['input_path']=str(Path(inp).resolve())
        if input_config is not None:row['input_installation_sha256']=input_config['_input_installation_sha256']
        if runtime_config is None:raise ValueError('Explicit matching MA27 runtime is required for structural inspection')
        command=PROFILES.make_command(None,'R3',binary,inp,payload,folder/'unused_trace.jsonl',backend='ma27',hsl_library=runtime_config['hsl_library'])
        row.update(native_backend='ma27',hsl_library_sha256=runtime_config['hsl_library_sha256'],native_build_proof_sha256=runtime_config['native_build_proof_sha256'])
        command+=['--factorial-structure-check']
        if start:command+=['--frozen-start',str(start)]
        atomic(folder/'COMMAND.json',{'command':command,'inspection_only':True})
        env=dict(os.environ);env.update(runtime_config['runtime_environment'])
        for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT'):env.pop(key,None)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='1'
        with log.open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env,
                                  timeout=timeout,check=False)
        row['returncode']=result.returncode
        if not payload.is_file():raise ValueError('Native coefficient inspection did not produce evidence')
        native=load(payload);row['inspection_sha256']=sha(payload)
        row['signed_square_changed']=native.get('native_signed_square_changed')
        row['unresolved_structural_phase']=(native.get('unresolved_structural_phase') is not False or
                                           native.get('post_presolve_absence_proved') is not True)
        row['native_schema']=native.get('schema')
        row['native_optimizer_invocations']=native.get('solver_invocations')
        row['post_presolve_covered']=native.get('post_presolve_absence_proved')
        row['inspected_models']=native.get('inspected_models')
        row['inspected_rank_models']=native.get('inspected_rank_models')
        row['inspected_exact_phase_models']=native.get('inspected_exact_phase_models')
        row['potential_signed_square_shape_count']=native.get('potential_signed_square_shape_count')
        row['native_reason']=native.get('reason')
        dimensions=case.get('dimensions',{})
        for key,native_key in (('n','n'),('equalities_total','equalities'),('inequalities_total','inequalities')):
            if key in dimensions and native.get(native_key)!=dimensions[key]:
                raise ValueError('Native structural inspector dimensions differ from the frozen contract: '+key)
        row['passed']=(result.returncode==0 and native.get('schema')=='lazy-factorial-structure-preflight-v1' and
                       native.get('inspection_only') is True and native.get('eligible') is True and
                       row['signed_square_changed'] is False and
                       row['unresolved_structural_phase'] is False and
                       native.get('solver_invocations')==0 and native.get('has_signed_square') is False and
                       native.get('potential_signed_square_shape_count')==0 and
                       type(native.get('inspected_models')) is int and native['inspected_models']>=1)
        if sha(inp)!=case['input_sha256']:raise ValueError('Frozen input changed during inspection')
        if (folder/'unused_trace.jsonl').exists():raise ValueError('Unexpected numerical iteration trace during structural preflight')
        if not row['passed']:row['error']='Native inspector reports changed, unresolved, or non-inspection-only structure'
    except FileNotFoundError as error:
        row.update(missing=True,passed=False,unresolved_structural_phase=True,error=str(error))
    except Exception as error:
        row.update(passed=False,unresolved_structural_phase=True,error=type(error).__name__+': '+str(error))
    return row
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','registration','contract','root','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--inspection-timeout',type=float,default=1800)
    p.add_argument('--input-installation',type=Path,required=True)
    p.add_argument('--build-proof',type=Path)
    a=p.parse_args();binary=a.binary.resolve();root=a.root.resolve();out=a.out.resolve()
    if out.exists():raise ValueError('Coefficient preflight requires a fresh output directory')
    initial_contract=sha(a.contract);initial_registration=sha(a.registration)
    contract=load(a.contract);registration=load(a.registration)
    if registration['inputs']['contract_file_sha256']!=initial_contract:raise ValueError('Preregistered corpus changed')
    cases=[c for c in contract['cases'] if c['regime'] in ('I','M')]
    regimes=dict(collections.Counter(c['regime'] for c in cases))
    if len(cases)!=1834 or regimes!={'I':1037,'M':797}:raise ValueError('Full1037I+797M native structural scan required')
    input_config=input_installation(a.input_installation,root,a.contract)
    pins=build_identity(binary);runtime_config=ma27_runtime_binding(binary,a.build_proof);initial_binary=sha(binary);runner=common()
    out.mkdir(parents=True)
    rows=[];rows_path=out/'COUPLING_CASES.jsonl'
    for i,case in enumerate(cases,1):
        row=inspect(case,root,binary,out,runner,a.inspection_timeout,input_config,runtime_config);rows.append(row)
        with rows_path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print('COUPLAGE_NATIF',i,'/1834',case['case_id'],
              'OK' if row['passed'] else 'BLOQUÉ',flush=True)
    changed=sum(r['signed_square_changed'] is True for r in rows)
    unresolved=sum(r['unresolved_structural_phase'] is not False for r in rows)
    missing=sum(r['missing'] for r in rows)
    unchanged=sha(binary)==initial_binary
    source_unchanged=build_identity(binary)==pins
    runtime_unchanged=ma27_runtime_binding(binary,a.build_proof)==runtime_config
    contract_unchanged=sha(a.contract)==initial_contract
    registration_unchanged=sha(a.registration)==initial_registration
    input_installation_unchanged=sha(a.input_installation)==input_config['_input_installation_sha256']
    passed=(all(r['passed'] for r in rows) and unchanged and source_unchanged and
            contract_unchanged and registration_unchanged and input_installation_unchanged and runtime_unchanged)
    gate={'schema':SCHEMA,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'passed':passed,'coupling_separation_verified':passed,'cases':len(cases),
          'regimes':regimes,'attempted_cases':len(rows),
          'inspected_cases':sum(r['inspection_sha256'] is not None for r in rows),'changed_cases':changed,
          'unresolved_cases':unresolved,'missing_cases':missing,
          'native_backend':'ma27','hsl_library_sha256':runtime_config['hsl_library_sha256'],'native_build_proof':runtime_config['native_build_proof'],'native_build_proof_sha256':runtime_config['native_build_proof_sha256'],'runtime_unchanged':runtime_unchanged,
          'factor_binary_sha256':initial_binary,'factor_binary':str(binary),
          'contract_sha256':initial_contract,'registration_sha256':initial_registration,
          'input_installation':input_config['_input_installation_path'],
          'input_installation_sha256':input_config['_input_installation_sha256'],
          'input_installation_unchanged':input_installation_unchanged,
          'rows_file':rows_path.name,'rows_sha256':sha(rows_path),**pins,
          'binary_unchanged':unchanged,'source_unchanged':source_unchanged,
          'contract_unchanged':contract_unchanged,'registration_unchanged':registration_unchanged,
          'numerical_optimizer_invocations':0,
          'scope':'Actual native coefficient inspection, including reachable post-presolve forms; no numerical solve or endpoint warm start.',
          'launch_authorized':False,
          'next_gate':'Passing structural inspection is necessary; the separate immutable50-case trajectory parity gate must also pass.'}
    atomic(out/'COUPLING_GATE.json',gate)
    print(json.dumps({'passed':passed,'changed':changed,'unresolved':unresolved,'missing':missing,
                      'receipt':str(out/'COUPLING_GATE.json')},indent=2))
    if not passed:raise SystemExit(2)
if __name__=='__main__':
    try:main()
    except Exception as error:print('COUPLAGE REFUSÉ — '+str(error),file=sys.stderr);raise SystemExit(2)
