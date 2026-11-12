#!/usr/bin/env python3
"""One-fixture compatibility smoke of the six unchanged published profiles."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,subprocess
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_profiles',HERE/'PROFILES.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def numerical(v):
 if isinstance(v,dict):return {k:numerical(x)for k,x in v.items()if not any(t in k for t in ('seconds','timing'))}
 if isinstance(v,list):return [numerical(x)for x in v]
 return v

def main():
 a=argparse.ArgumentParser();a.add_argument('--baseline',required=True,type=Path);a.add_argument('--research',required=True,type=Path);a.add_argument('--out',required=True,type=Path);a.add_argument('--input',type=Path,default=HERE/'source/tests/hh_r3_inputs/mixed_curvature.qcq');a=a.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1');records=[]
 for method in ('R3','HH','HS','SH','SS','SS_legacy'):
  pair=[]
  for role,binary in [('qualified_original',a.baseline),('research_historical_flags',a.research)]:
   d=a.out/method/role;d.mkdir(parents=True,exist_ok=True);endpoint=d/'endpoint.json';cmd=p.make_published_command(method,binary,a.input,endpoint);(d/'COMMAND.json').write_text(json.dumps(cmd,indent=2)+'\n')
   with(d/'solver.log').open('w')as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=35)
   j=json.loads(endpoint.read_text());pair.append(j);print(method,role,r.returncode,j.get('status'),flush=True)
  records.append(dict(method=method,exact_nonclock_endpoint_and_counters=numerical(pair[0])==numerical(pair[1]),status=pair[0].get('status'),outer_iterations=pair[0].get('outer_iterations'),restoration_iterations=pair[0].get('restoration_iterations')))
 result={'schema':'native-six-historical-profiles-local-compatibility-v1','input':str(a.input),'input_sha256':sha(a.input),'baseline_binary_sha256':sha(a.baseline),'research_binary_sha256':sha(a.research),'source_profile_evidence_sha256':sha(HERE/'profile_evidence/PUBLISHED_SIX_PROFILES.json'),'new_factorial_flags_used':False,'all_passed':all(r['exact_nonclock_endpoint_and_counters']for r in records),'scope':'One native mixed-curvature QCQP with indefinite objective; twelve optimizer invocations; compatibility smoke, not full six-profile campaign qualification','records':records}
 (a.out/'SIX_PROFILES_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print('SIX_PROFILE_SMOKE',result['all_passed'],flush=True)
if __name__=='__main__':main()
