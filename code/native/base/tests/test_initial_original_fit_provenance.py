#!/usr/bin/env python3
"""An initial original KKT certificate must not report an optimization failure."""
import argparse,json,os,subprocess,tempfile
from fractions import Fraction as F
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='lazy_initial_original_certificate_') as directory:
 root=Path(directory);model=root/'model.qcq';endpoint=root/'endpoint.json'
 model.write_text('''LAZYIPDCA_QCQP_V1
name initial_original_certificate
n 1
x0 1
lb -INF
ub INF
objective
quadratic
name f
constant 0
linear 1
hessian_nnz 0
end_quadratic
equalities 1
quadratic
name h
constant -1
linear 0
hessian_nnz 1
0 0 2
end_quadratic
inequalities 0
end_problem
''')
 command=[str(a.binary.resolve()),'--problem',str(model),'--json',str(endpoint),'--algorithm','lazy-dca','--dca-engine','majorants','--linear-backend','mumps','--linear-metis','1','--native-al-fallback','off','--time-limit','30','--max-outer','5000','--feas-tol','1e-6','--dual-inf-tol','1e-6','--compl-inf-tol','1e-6','--opt-tol','1e-6','--display-level','0']
 run=subprocess.run(command,capture_output=True,text=True,timeout=40,env={**os.environ,'OMP_NUM_THREADS':'1','OMP_THREAD_LIMIT':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 assert run.returncode==0,(run.stdout,run.stderr)
 d=json.loads(endpoint.read_text());assert d['status']=='solved'
 assert d['x']==[1] and d['certificate_equality_multipliers']==[-.5]
 assert abs(F(d['x'][0])**2-1)<=F(1e-6)
 assert abs(1+2*F(d['x'][0])*F(d['certificate_equality_multipliers'][0]))<=F(1e-6)
 assert d['original_failure_status']=='' and d['original_failure_message']==''
 assert d['termination_certificate_rescued']
 assert d['termination_multiplier_source']=='initial-fixed-primal-complementarity-bounded-fit-original-audit'
 assert d['outer_iterations']==0 and d['lazy_newton_factor_attempts']==0
 assert not d['native_al_fallback']['attempted']
 print('initial_original_fit_provenance PASS original point and KKT multipliers preserved; no failed iteration reported; no AL')
