#!/usr/bin/env python3
"""Native regression on a SYNTHETIC perturbation, not case4020_goc."""
import argparse,json,math,os,subprocess,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--command',type=Path,required=True);a=p.parse_args()
base=json.loads(a.command.read_text());env=os.environ.copy()
for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):env[key]='1'
env['OMP_DYNAMIC']='FALSE';env['MKL_DYNAMIC']='FALSE'
with tempfile.TemporaryDirectory(prefix='auto_retry_test_') as tmp:
 tmp=Path(tmp)
 for policy in ('legacy','auto'):
  ep=tmp/(policy+'.json');cmd=[str(a.binary) if x=='@BINARY@' else str(a.input) if x=='@INPUT@' else str(ep) if x=='@OUTPUT@' else x for x in base]
  cmd[cmd.index('--majorant-startup-policy')+1]=policy
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=60)
  if not ep.is_file():raise RuntimeError('No endpoint: '+r.stdout[-3000:])
  d=json.loads(ep.read_text())
  if policy=='legacy':
   if r.returncode!=1 or d['status']!='max_outer_iterations':raise RuntimeError('Legacy control did not reproduce expected stall')
  else:
   if r.returncode or d['status']!='solved':raise RuntimeError('AUTO retry failed: '+r.stdout[-3000:])
   retry=[line for line in r.stdout.splitlines() if line.startswith('[AUTO-STARTUP-RETRY] accepted=1')]
   if len(retry)!=1 or 'reset_primal=frozen-x0' not in retry[0]:raise RuntimeError('Expected one declared cold-primal retry')
   if d['outer_iterations']<=64 or d['outer_iterations']>3000:raise RuntimeError('Global iteration counter was reset')
   for key in ('feasibility','dual_infeasibility_inf','complementarity_inf'):
    if not math.isfinite(d[key]) or not 0<=d[key]<=1e-6:raise RuntimeError('Invalid original KKT residual: '+key)
   print(retry[0]);print('PASS native synthetic recovery; original KKT and global counters preserved')
