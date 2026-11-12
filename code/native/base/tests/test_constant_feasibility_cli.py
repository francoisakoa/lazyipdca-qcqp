#!/usr/bin/env python3
"""Portable native constant-objective pass/reject tests with exact binary64 audit."""
import argparse,json,math,os,subprocess,tempfile
from fractions import Fraction as F
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('binary',type=Path);a=p.parse_args();binary=a.binary.resolve()
def write(path,name,n,x0,lb,ub,c,eq):
 def q(name,c,linear,h):return '\n'.join(['quadratic','name '+name,'constant '+str(c),'linear '+' '.join(map(str,linear)),'hessian_nnz '+str(len(h)),*[' '.join(map(str,e))for e in h],'end_quadratic'])
 lines=['LAZYIPDCA_QCQP_V1','name '+name,'n '+str(n),'x0 '+' '.join(map(str,x0)),'lb '+' '.join(map(str,lb)),'ub '+' '.join(map(str,ub)),'objective',q('f',c,[0]*n,[]),'equalities '+str(len(eq)),*[q('h'+str(i),*v)for i,v in enumerate(eq)],'inequalities 0','end_problem'];path.write_text('\n'.join(lines)+'\n')
def exact_audit(d,lb,ub,eq):
 x=list(map(F,d['x']));y=list(map(F,d['certificate_equality_multipliers']));zl=list(map(F,d['certificate_lower_multipliers']));zu=list(map(F,d['certificate_upper_multipliers']));assert len(x)==len(lb)==len(ub)==len(zl)==len(zu)and len(y)==len(eq)
 primal=F(0);comp=F(0);stationarity=[u-l for l,u in zip(zl,zu)]
 for multiplier,(constant,linear,hessian)in zip(y,eq):
  value=F(constant)+sum(F(v)*x[j]for j,v in enumerate(linear))
  gradient=list(map(F,linear))
  for i,j,v in hessian:
   value+=(F(v,2)if i==j else F(v))*x[i]*x[j];gradient[i]+=F(v)*x[j]
   if i!=j:gradient[j]+=F(v)*x[i]
  primal=max(primal,abs(value))
  for j,v in enumerate(gradient):stationarity[j]+=multiplier*v
 dual=max([F(0),*map(abs,stationarity),*[-v for v in zl],*[-v for v in zu]])
 for j,(l,u)in enumerate(zip(lb,ub)):
  if math.isfinite(l):primal=max(primal,F(l)-x[j]);comp=max(comp,abs(zl[j]*(x[j]-F(l))))
  else:dual=max(dual,abs(zl[j]))
  if math.isfinite(u):primal=max(primal,x[j]-F(u));comp=max(comp,abs(zu[j]*(F(u)-x[j])))
  else:dual=max(dual,abs(zu[j]))
 return primal,dual,comp
circle=(-1,[0,0],[(0,0,2),(1,1,2)])
cases=[('negative_constant_bounded',2,[.5,.5],[-2,-2],[2,2],-7,[circle],True),('constant_redundant_equation',2,[.5,.5],[-math.inf]*2,[math.inf]*2,0,[circle,(0,[0,0],[])],True),('unused_coordinate_and_zero_equation',3,[.5,.5,.25],[-math.inf]*3,[math.inf]*3,0,[(-1,[0,0,0],[(0,0,2),(1,1,2)]),(0,[0,1,0],[]),(0,[0,0,0],[])],True),('constant_infeasible',1,[0],[-math.inf],[math.inf],0,[(1,[0],[(0,0,2)])],False),('constant_inconsistent_rows',1,[.5],[-math.inf],[math.inf],0,[(-1,[0],[(0,0,2)]),(-4,[0],[(0,0,2)])],False),('constant_bounded_infeasible',1,[.5],[0],[1],0,[(-2,[1],[])],False)]
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
env.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE');results=[]
with tempfile.TemporaryDirectory(prefix='lazy_constant_feasibility_')as folder:
 folder=Path(folder)
 for name,n,x0,lb,ub,c,eq,expected in cases:
  model=folder/(name+'.qcq');endpoint=folder/(name+'.json');write(model,name,n,x0,lb,ub,c,eq)
  cmd=[str(binary),'--problem',str(model),'--json',str(endpoint),'--algorithm','lazy-dca','--dca-engine','majorants','--linear-backend','mumps','--linear-metis','1','--time-limit','10','--max-outer','500','--native-al-fallback','off','--termination-test','project','--feas-tol','1e-6','--opt-tol','1e-6','--dual-inf-tol','1e-6','--compl-inf-tol','1e-6','--display-level','0']
  run=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=30)
  assert endpoint.is_file(),(name,run.returncode,run.stdout,run.stderr)
  data=json.loads(endpoint.read_text());residuals=exact_audit(data,lb,ub,eq);strict=max(residuals)<=F(1e-6)
  assert strict==expected,(name,strict,expected,list(map(float,residuals)),data.get('status'))
  assert not data.get('native_al_fallback',{}).get('attempted',False),(name,'AL invoked')
  if expected:assert data['status']=='solved'and data['objective']==c,(name,data['status'],data['objective'])
  else:assert data['status']!='solved',(name,'false solver success')
  if name=='unused_coordinate_and_zero_equation':
   assert data['x'][2]==.25,(name,'unused coordinate start not retained')
   assert data.get('restoration_iterations')==0,(name,'rectangular normal route was bypassed',data.get('restoration_iterations'))
  results.append({'case':name,'expected_strict':expected,'strict':strict,'PDC':list(map(float,residuals))})
print(json.dumps({'constant_feasibility_cli':'PASS','exact_binary64_audit':True,'AL_attempted':False,'cases':results}))
