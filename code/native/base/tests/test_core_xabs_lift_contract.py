#!/usr/bin/env python3
"""Exact-lifting contract for a CUTEst-like q*abs(q) equality."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys, tempfile, types
import numpy as np
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('export_cutest_qcqp', ROOT/'python'/'export_cutest_qcqp.py')
mod=importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(mod)

class FakeCore:
    name='CORE2'; n=2; m=2
    x0=np.array([-0.5, 0.2])
    bl=np.array([-2.0, -2.0]); bu=np.array([2.0, 2.0])
    cl=np.array([0.0, -1.0e20]); cu=np.array([0.0, 1.0])
    is_eq_cons=np.array([True, False]); is_linear_cons=np.array([False, True])
    def obj(self,x,gradient=False):
        x=np.asarray(x); v=float(x[1]); g=np.array([0.0,1.0])
        return (v,g) if gradient else v
    def cons(self,x,index=None,gradient=False):
        x=np.asarray(x)
        vals=np.array([x[0]*abs(x[0])+2*x[1]-0.25, x[1]])
        J=np.array([[2*abs(x[0]),2.0],[0.0,1.0]])
        if index is None: return (vals,J) if gradient else vals
        return (float(vals[index]),J[index]) if gradient else float(vals[index])
    def grad(self,x,index=None):
        if index is None: return np.array([0.0,1.0])
        return self.cons(x,index=index,gradient=True)[1]
    def isphess(self,x,cons_index=None):
        if cons_index is None: return sp.csr_matrix((2,2))
        if cons_index==1: return sp.csr_matrix((2,2))
        s=1.0 if np.asarray(x)[0]>=0 else -1.0
        return sp.csr_matrix(np.diag([2*s,0.0]))

sys.modules['pycutest']=types.SimpleNamespace(import_problem=lambda *a,**k:FakeCore())
with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'core2.qcq'
    info=mod.export_problem('CORE2',out,{},check=True,reformulation='auto')
    assert info['reformulation']=='core_xabs_exact_qcqp'
    assert info['n_original']==2 and info['n']==3
    assert info['abs_lifts']==1
    assert info['equalities']==2  # original equality + t^2-q^2
    assert info['inequalities']==1
    assert info['lift_validation_value'] < 1e-12
    assert info['lift_validation_gradient'] < 1e-12
    text=out.read_text()
    assert 'name abs_link_x0' in text
    assert '0 2 1' in text or '0 2 1.0' in text
print('[PASS] exact q*abs(q) QCQP lifting contract')
