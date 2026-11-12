#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, sys, tempfile, types
from pathlib import Path
import numpy as np
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('export_cutest_qcqp',ROOT/'python'/'export_cutest_qcqp.py')
mod=importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(mod)
class FakeBT2:
    name='BT2'; n=3; m=1
    x0=np.array([10.,10.,10.]); bl=np.full(3,-1e20); bu=np.full(3,1e20)
    # Match PyCUTEst builds that absorb the SIF group constant into c(x) and
    # expose the equality with zero lower/upper bounds.
    cl=np.array([0.0]); cu=np.array([0.0]); is_eq_cons=np.array([True]); is_linear_cons=np.array([False])
    def obj(self,x,gradient=False):
        x1,x2,x3=map(float,x); d=x2-x3
        v=(x1-1)**2+(x1-x2)**2+d**4
        g=np.array([2*(x1-1)+2*(x1-x2), -2*(x1-x2)+4*d**3, -4*d**3])
        return (v,g) if gradient else v
    def cons(self,x,index=None,gradient=False):
        x1,x2,x3=map(float,x); v=x1*(1+x2*x2)+x3**4-8.2426407
        g=np.array([1+x2*x2,2*x1*x2,4*x3**3])
        return (v,g) if gradient else v
    def grad(self,x,index=None): return self.cons(x,index=index,gradient=True)[1]
    def isphess(self,x,cons_index=None):
        if cons_index is None:
            raise RuntimeError('quartic objective Hessian must not be queried before the BT2 lift')
        x1,x2,x3=map(float,x)
        H=np.array([[0,2*x2,0],[2*x2,2*x1,0],[0,0,12*x3*x3]],float)
        return sp.csr_matrix(H)
sys.modules['pycutest']=types.SimpleNamespace(import_problem=lambda *a,**k:FakeBT2())
with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'BT2.qcq'
    info=mod.export_problem('BT2',out,{},check=True,reformulation='auto')
    assert info['reformulation']=='bt2_exact_qcqp'
    assert info['n_original']==3 and info['n']==6
    assert info['equalities']==4 and info['inequalities']==0
    assert info['lift_validation_value'] < 1e-12
    assert info['lift_validation_gradient'] < 1e-11
print('[PASS] BT2 exact quartic-to-QCQP lift')
