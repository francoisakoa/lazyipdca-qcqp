#!/usr/bin/env python3
"""Fresh native MA27 qualification: six controls on generic I/M/E/Q fixtures.
These 24 short qualifications do not replace the 50-case factorial gate or a
full benchmark, and cannot qualify any historical MUMPS trajectory.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,subprocess,sys,time
import PROFILES
from MA27_RUNTIME import pin_hsl
HERE=Path(__file__).resolve().parent
METHODS=('R3','HH','HS','SH','SS','SS_legacy')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path):
 spec=importlib.util.spec_from_file_location('ma27_original_auditor',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def fixtures(out):
 out.mkdir(parents=True,exist_ok=True)
 def row(name,n,const,lin,h):return ['quadratic','name '+name,'constant '+str(const),'linear '+' '.join(map(str,lin)),'hessian_nnz '+str(len(h))]+[f'{i} {j} {v}'for i,j,v in h]+['end_quadratic']
 result={}
 for regime,source in [('I',HERE/'source/tests/hh_r3_inputs/convex_zero.qcq'),('M',HERE/'source/tests/hh_r3_inputs/mixed_curvature.qcq')]:
  p=out/(regime+'.qcq');p.write_bytes(source.read_bytes());result[regime]=p
 for regime in ('E','Q'):
  lines=['LAZYIPDCA_QCQP_V1','name generic_'+regime+'_ma27_control','n 3','x0 -0.8 0.4 -1.6','lb -2 -2 -4','ub 2 2 4','objective']
  if regime=='E':lines+=row('f',3,0,[1,.1,.03],[])
  else:lines+=row('f',3,0,[-.08,.08,0],[(0,0,.4),(1,1,.8),(2,2,.4)])
  lines+=['equalities 2']+row('ellipse',3,-1,[.2,-.1,0],[(0,0,2),(0,1,1),(1,1,3)])+row('affine',3,0,[-1,2,1],[])+['inequalities 0','end_problem']
  p=out/(regime+'.qcq');p.write_text('\n'.join(lines)+'\n');result[regime]=p
 return result

def qualify(binary,hsl_library,auditor_path,out,cpu=None,runtime_environment=None):
 binary=Path(binary).resolve();out=Path(out).resolve();auditor_path=Path(auditor_path).resolve()
 if out.exists():raise ValueError('Fresh MA27 qualification output required')
 out.mkdir(parents=True)
 proof=dict(schema='native-ma27-six-profiles-qualification-v1',passed=False,native_backend='ma27',binary=str(binary),binary_sha256=sha(binary)if binary.is_file()else None,hsl_library=str(Path(hsl_library).resolve()),hsl_library_sha256=sha(hsl_library)if Path(hsl_library).is_file()else None,profiles=list(METHODS),six_profiles_qualified=False,planned_native_runs=24,completed_native_runs=0,native_executions=0,qualification_limits=dict(seconds=60,outer_iterations=300,shared_restoration=600),benchmark_limits_unchanged=True,observations=[],failure=None)
 start=time.monotonic()
 try:
  runtime=pin_hsl(hsl_library);proof['hsl_runtime']=runtime
  backends=subprocess.check_output([str(binary),'--linear-backends'],text=True).strip().split(',')
  if'ma27'not in backends:raise ValueError('Native MA27 adapter is not compiled')
  if'mumps'in backends:raise ValueError('This MA27-only qualification requires MUMPS omitted, preventing hidden primary fallback')
  proof['compiled_backends']=backends;proof['mumps_compiled']=False
  auditor=load(auditor_path);proof['auditor']=str(auditor_path);proof['auditor_sha256']=sha(auditor_path)
  inputs=fixtures(out/'fixtures');proof['fixtures']={r:dict(path=str(p),sha256=sha(p))for r,p in inputs.items()}
  env=dict(os.environ,OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',HSL_LIBRARY=str(Path(hsl_library).resolve()))
  if runtime_environment is not None:env.update(runtime_environment)
  env['HSL_LIBRARY']=str(Path(hsl_library).resolve())
  for regime,inp in inputs.items():
   model=auditor.read_model(inp)
   for method in METHODS:
    d=out/'tasks'/(regime+'__'+method);d.mkdir(parents=True);endpoint=d/'endpoint.json';command=PROFILES.make_published_command(method,binary,inp,endpoint,backend='ma27',hsl_library=hsl_library)
    command[command.index('--time-limit')+1]='60';command[command.index('--max-outer')+1]='300'
    if cpu is not None:command=['taskset','-c',str(cpu)]+command
    (d/'COMMAND.json').write_text(json.dumps(command,indent=2)+'\n');record=dict(regime=regime,method=method,passed=False,command_sha256=sha(d/'COMMAND.json'),input_sha256=sha(inp),failure=None)
    proof['native_executions']+=1;began=time.monotonic()
    try:
     with(d/'solver.log').open('w')as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=65)
     proof['completed_native_runs']+=1;record['returncode']=r.returncode;data=json.loads(endpoint.read_text());record['endpoint_sha256']=sha(endpoint);audit=auditor.audit(model,data);record['original_audit']=audit
     backend=str(data.get('linear_solver_backend',''));attempts=int(data.get('lazy_newton_factor_attempts',0));failed=int(data.get('lazy_newton_backend_factor_failures',0));successful=max(0,attempts-failed)
     record.update(native_backend=backend,ma27_primary_factor_attempts=attempts,ma27_primary_successful_factors=successful,outer_iterations=data.get('outer_iterations'),restoration_iterations=data.get('restoration_iterations'),seconds=data.get('elapsed_seconds'),status=data.get('status'),pivoted_fallback=data.get('dca_filter_pivoted_backend'),native_factor_fallback_successes=data.get('lazy_newton_native_factor_successes'))
     log=(d/'solver.log').read_text(errors='replace')
     if'ma27'not in backend.lower()or successful<1 or data.get('lazy_newton_accepted_steps',0)<1:raise ValueError('No actual successful primary MA27 factor/accepted direction')
     if data.get('dca_filter_pivoted_backend')or data.get('lazy_newton_native_factor_successes',0):raise ValueError('A primary fallback factorization was used')
     if data.get('native_al_fallback',{}).get('enabled')is True or data.get('native_al_fallback',{}).get('active')is True:raise ValueError('Disallowed native AL phase')
     if not audit['strict']:raise ValueError('Independent original KKT certificate is not strict')
     if r.returncode!=0 or not 0<=float(data.get('elapsed_seconds',math.inf))<=60 or not 0<=int(data.get('outer_iterations',301))<=300 or not 0<=int(data.get('restoration_iterations',601))<=600:raise ValueError('Qualification completion/budget contract differs')
     record['passed']=True
    except Exception as e:record['failure']=type(e).__name__+': '+str(e)
    record['process_wall_seconds']=time.monotonic()-began;proof['observations'].append(record);(d/'QUALIFICATION.json').write_text(json.dumps(record,indent=2)+'\n');print('NATIVE_MA27_QUALIFICATION',regime,method,'PASS'if record['passed']else'BLOCKED',record.get('failure')or'',flush=True)
  proof['six_profiles_qualified']=len(proof['observations'])==24 and all(x['passed']for x in proof['observations']);proof['passed']=proof['six_profiles_qualified']
  if sha(binary)!=proof['binary_sha256']or sha(hsl_library)!=proof['hsl_library_sha256']:raise ValueError('Native/HSL binary changed during qualification')
  if not proof['passed']:proof['failure']='At least one real MA27/original-KKT qualification is incomplete or failed'
 except Exception as e:proof.update(passed=False,six_profiles_qualified=False,failure=type(e).__name__+': '+str(e))
 proof['seconds']=time.monotonic()-start;proof['profiles_factory_sha256']=sha(HERE/'PROFILES.py');proof['qualifier_sha256']=sha(Path(__file__));(out/'QUALIFICATION_MA27.json').write_text(json.dumps(proof,indent=2)+'\n');return proof

def main():
 p=argparse.ArgumentParser();p.add_argument('--binary',required=True,type=Path);p.add_argument('--hsl-library',required=True,type=Path);p.add_argument('--auditor',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--cpu',type=int);p.add_argument('--runtime-proof',type=Path);a=p.parse_args();runtime=json.loads(a.runtime_proof.read_text()).get('runtime_environment')if a.runtime_proof else None;proof=qualify(a.binary,a.hsl_library,a.auditor,a.out,a.cpu,runtime);print(json.dumps({k:proof[k]for k in ('schema','passed','six_profiles_qualified','failure')},indent=2));return 0 if proof['passed']else 2
if __name__=='__main__':raise SystemExit(main())
