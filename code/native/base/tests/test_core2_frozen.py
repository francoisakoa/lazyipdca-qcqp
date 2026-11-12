#!/usr/bin/env python3
"""Frozen CORE2 regression through the CLI and an independent original QCQP audit."""
import argparse, hashlib, json, os, pathlib, subprocess, tempfile, sys
from core2_audit import audit, read_model
ROOT=pathlib.Path(__file__).resolve().parent
EXPECTED='6d54ef837f9c2ed2738860daefc658310332272853856391cec19dbbe8f58b04'
p=argparse.ArgumentParser();p.add_argument('--binary',type=pathlib.Path,required=True)
p.add_argument('--backend',choices=['mumps','cholesk'],required=True)
p.add_argument('--variant',choices=['current','rho','tail'],required=True)
a=p.parse_args();inp=ROOT/'data/core2_frozen.qcq'
if hashlib.sha256(inp.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('CORE2 frozen input changed')
cmd=json.loads((ROOT/'data/core2_command.json').read_text());cmd[0]=str(a.binary.resolve())
def opt(k,v):
 if k in cmd:cmd[cmd.index(k)+1]=str(v)
 else:cmd.extend([k,str(v)])
opt('--problem',inp);opt('--linear-backend',a.backend);opt('--linear-metis',1)
opt('--majorant-lambda-interpolation',int(a.variant!='current'))
opt('--majorant-lambda-rho-retry',int(a.variant!='current'))
opt('--majorant-curvature-corrections',2 if a.variant=='tail' else 0)
env=dict(os.environ,OMP_NUM_THREADS='1',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1')
for k in ['LD_PRELOAD','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS']:env.pop(k,None)
with tempfile.TemporaryDirectory(prefix='core2_frozen_') as td:
 out=pathlib.Path(td)/'endpoint.json';opt('--json',out)
 run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,timeout=1800)
 ep=json.loads(out.read_text()) if out.exists() else {}
 cert=audit(read_model(inp),ep,1e-6)
 print(json.dumps({'backend':a.backend,'variant':a.variant,'status':ep.get('status'),'certificate':cert},allow_nan=False))
 if run.returncode or ep.get('status')!='solved' or not cert['strict']:
  print(run.stdout[-15000:]);raise RuntimeError('CORE2 strict certificate regression')
 if 'discard the previous inequality model' in run.stdout:raise RuntimeError('Model lifecycle error remains')
 if abs(cert['objective']-72.9)/72.9>1e-6:raise RuntimeError('CORE2 objective differs from archived success')
print('PASS frozen CORE2, same start, strict original QCQP, compatible objective')
