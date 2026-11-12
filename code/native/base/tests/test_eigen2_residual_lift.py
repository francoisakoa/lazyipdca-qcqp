#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, sys, tempfile, types
from pathlib import Path
import numpy as np
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('export_cutest_qcqp',ROOT/'python'/'export_cutest_qcqp.py')
mod=importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(mod)
class FakeEig:
    name='EIGENA2'; n=6; m=3
    # D1,Q11,Q21,D2,Q12,Q22
    x0=np.array([1.,1.,0.,1.,0.,1.]); bl=np.full(6,-1e20); bu=np.full(6,1e20)
    cl=np.array([1.,0.,1.]); cu=np.array([1.,0.,1.]); is_eq_cons=np.ones(3,dtype=bool); is_linear_cons=np.zeros(3,dtype=bool)
    varnames=['D(1)','Q(1,1)','Q(2,1)','D(2)','Q(1,2)','Q(2,2)']
    def _unpack(self,x):
        D=np.array([x[0],x[3]],float); Q=np.array([[x[1],x[4]],[x[2],x[5]]],float); return D,Q
    def obj(self,x,gradient=False):
        D,Q=self._unpack(np.asarray(x,float)); A=np.diag([1.,2.]); R=np.zeros((2,2))
        for j in range(2):
            for i in range(2): R[i,j]=Q[j,i]*D[j]-sum(A[i,k]*Q[j,k] for k in range(2))
        f=float(np.sum(R*R)); g=np.zeros(6)
        for j in range(2):
            di=(0,3)[j]; g[di]=2*sum(R[i,j]*Q[j,i] for i in range(2))
            for k in range(2):
                qi=k*3+1+j; g[qi]=2*(R[k,j]*D[j]-sum(A[i,k]*R[i,j] for i in range(2)))
        return (f,g) if gradient else f
    def cons(self,x,index=None,gradient=False):
        D,Q=self._unpack(np.asarray(x,float)); vals=np.array([Q[:,0]@Q[:,0],Q[:,0]@Q[:,1],Q[:,1]@Q[:,1]])
        J=np.zeros((3,6));
        # q columns: col0 vars 1,2; col1 vars 4,5
        J[0,[1,2]]=2*Q[:,0]; J[1,[1,2]]=Q[:,1]; J[1,[4,5]]=Q[:,0]; J[2,[4,5]]=2*Q[:,1]
        if index is None: return (vals,J) if gradient else vals
        return (float(vals[index]),J[index]) if gradient else float(vals[index])
    def grad(self,x,index=None): return self.cons(x,index=index,gradient=True)[1]
    def isphess(self,x,cons_index=None):
        if cons_index is None: return sp.csr_matrix((6,6))
        H=np.zeros((6,6))
        if cons_index==0: H[1,1]=H[2,2]=2
        elif cons_index==1:
            H[1,4]=H[4,1]=1; H[2,5]=H[5,2]=1
        else: H[4,4]=H[5,5]=2
        return sp.csr_matrix(H)
sys.modules['pycutest']=types.SimpleNamespace(import_problem=lambda *a,**k:FakeEig())
with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'EIGENA2.qcq'
    info=mod.export_problem('EIGENA2',out,{},check=True,reformulation='auto')
    assert info['reformulation']=='eigen2_residual_exact_qcqp'
    assert info['n_original']==6 and info['n']==10
    assert info['equalities']==7 and info['inequalities']==0
    assert info['lift_validation_value'] < 1e-12
    assert info['lift_validation_gradient'] < 1e-11
print('[PASS] EIGEN*2 residual exact QCQP lift')
