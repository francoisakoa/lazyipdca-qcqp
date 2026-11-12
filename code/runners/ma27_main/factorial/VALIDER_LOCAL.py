#!/usr/bin/env python3
"""Local native smoke/parity checks. This is not the mandatory 50-case gate."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess,time
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('factorial_profiles',HERE/'PROFILES.py');profiles=importlib.util.module_from_spec(spec);spec.loader.exec_module(profiles)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(v):
 if isinstance(v,dict):return {k:normalize(x)for k,x in v.items()if not any(t in k for t in ('seconds','timing'))and k!='free_epigraph_checkpoint_lifts'}
 if isinstance(v,list):return [normalize(x)for x in v]
 return v

def differences(a,b,path=''):
 if type(a)!=type(b):return [path]
 if isinstance(a,dict):
  if a.keys()!=b.keys():return [path+'keys']
  return [p for k in a for p in differences(a[k],b[k],path+'.'+k)]
 if isinstance(a,list):
  if len(a)!=len(b):return [path+'length']
  return [p for i,(x,y)in enumerate(zip(a,b))for p in differences(x,y,path+'.'+str(i))]
 return []if a==b else[path]
def events(path):
 out=[]
 for line in path.read_text().splitlines():
  r=json.loads(line)
  if r.get('event')=='header':continue
  out.append({k:r.get(k)for k in ('event','accepted_index','after_accepted_index','state_sha256','phase_n','phase_mi','phase_me','outer_iteration','mu','alpha')})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--observer',required=True,type=Path);p.add_argument('--research',required=True,type=Path);p.add_argument('--reproduction',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--external-timeout',type=float,default=45);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 cases={'favorable':a.reproduction/'favorable/problem.qcq','boundary_n4':a.reproduction/'boundary_inputs/frozen/mitt_boundary_cont6_n0004/input.qcq','mixed_indefinite_objective':HERE/'source/tests/hh_r3_inputs/mixed_curvature.qcq'}
 env=dict(os.environ,OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
 records=[];checks=[]
 def run(case,kind,method,trace=False,legacy=False):
  d=a.out/case/(kind+'_'+method+('_legacy'if legacy else''));d.mkdir(parents=True,exist_ok=True);e=d/'endpoint.json';t=d/'accepted_trace.jsonl';binary={'baseline':a.baseline,'observer':a.observer,'research':a.research}[kind]
  if kind=='research':cmd=profiles.make_command(None,method,binary,cases[case],e,t,legacy)
  else:cmd=profiles.make_legacy_command(None,method,binary,cases[case],e,t if trace else None)
  (d/'COMMAND.json').write_text(json.dumps(cmd,indent=2)+'\n');record=dict(case=case,kind=kind,method=method,legacy_objective=legacy,binary_sha256=sha(binary),input_sha256=sha(cases[case]),command_sha256=sha(d/'COMMAND.json'))
  try:
   with(d/'solver.log').open('w')as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=a.external_timeout)
   endpoint=json.loads(e.read_text());record.update(returncode=q.returncode,status=endpoint['status'],endpoint_sha256=sha(e),outer_iterations=endpoint.get('outer_iterations'))
   if trace or kind=='research':record['sidecar']=json.loads(Path(str(e)+'.factorial.json').read_text());record['trace_sha256']=sha(t)
  except Exception as ex:record.update(censored=True,strict=None,error=type(ex).__name__+': '+str(ex))
  records.append(record);print(case,kind,method,record.get('status',record.get('error')),flush=True);return d,record
 for case in cases:
  for method,cell in [('R3','R3'),('HH','H0')]:
   b,br=run(case,'baseline',method);o,orr=run(case,'observer',method,True);r,rr=run(case,'research',cell,True,method=='HH')
   if any(x.get('censored')for x in (br,orr,rr)):checks.append(dict(case=case,diagonal=method,passed=False,censored=True));continue
   bj,oj,rj=[json.loads((d/'endpoint.json').read_text())for d in (b,o,r)]
   do=differences(normalize(bj),normalize(oj));dr=differences(normalize(oj),normalize(rj));trace_equal=events(o/'accepted_trace.jsonl')==events(r/'accepted_trace.jsonl')
   side_valid=all(x['sidecar']['trace_success']is True and x['sidecar']['budget_verified']is True for x in (orr,rr))
   checks.append(dict(case=case,diagonal=method,passed=not do and not dr and trace_equal and side_valid,baseline_observer_differences=do,observer_research_differences=dr,trajectory_equal=trace_equal,observation_counts={k:v.get('free_epigraph_checkpoint_lifts')for k,v in [('baseline',bj),('observer',oj),('research',rj)]},state_events_compared=len(events(o/'accepted_trace.jsonl'))))
  # Main exact-objective cells are separate from the legacy-objective HH gate.
  for cell in ('X1','X2','H0'):run(case,'research',cell,True,False)
 result={'schema':'native-factorial-local-smoke-parity-v1','mandatory_50_gate_passed':False,'scope':'three available controls including a mixed QCQP with an indefinite objective; not a corpus qualification','checks':checks,'all_local_diagonals_passed':all(x['passed']for x in checks),'records':records,'observer_count_exception':'free_epigraph_checkpoint_lifts only; reported separately; all numerical endpoint fields and nonclock solver counters compared exactly','clock_fields_excluded':['*seconds*','*timing*']}
 (a.out/'LOCAL_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print('LOCAL_DIAGONALS',result['all_local_diagonals_passed'],flush=True)
if __name__=='__main__':main()
