#!/usr/bin/env python3
"""CORE exact-lift contract with sign-fixed and sign-crossing pipe flows."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys, tempfile, types
import numpy as np
import scipy.sparse as sp

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('export_cutest_qcqp', ROOT/'python'/'export_cutest_qcqp.py')
mod=importlib.util.module_from_spec(SPEC); assert SPEC.loader; SPEC.loader.exec_module(mod)

class FakeCore1:
    name='CORE1'; n=3; m=3
    x0=np.array([0.5, -0.5, 0.2])
    # x0 is nonnegative-only, x1 nonpositive-only, x2 crosses zero.
    bl=np.array([0.0, -2.0, -2.0]); bu=np.array([2.0, 0.0, 2.0])
    cl=np.array([0.0, -1.0e20, 0.0]); cu=np.array([0.0, 4.0, 0.0])
    is_eq_cons=np.array([True, False, True])
    is_linear_cons=np.array([False, False, False])
    def obj(self,x,gradient=False):
        x=np.asarray(x); v=float(x[0]+x[1]+x[2]); g=np.ones(3)
        return (v,g) if gradient else v
    def cons(self,x,index=None,gradient=False):
        x=np.asarray(x)
        vals=np.array([
            2.0*x[0]*abs(x[0]) + x[2] - 0.7,
            3.0*x[1]*abs(x[1]) + x[0],
            1.5*x[2]*abs(x[2]) + x[1] + 0.44,
        ])
        J=np.array([
            [4.0*abs(x[0]), 0.0, 1.0],
            [1.0, 6.0*abs(x[1]), 0.0],
            [0.0, 1.0, 3.0*abs(x[2])],
        ])
        if index is None: return (vals,J) if gradient else vals
        return (float(vals[index]),J[index]) if gradient else float(vals[index])
    def grad(self,x,index=None):
        if index is None: return np.ones(3)
        return self.cons(x,index=index,gradient=True)[1]
    def isphess(self,x,cons_index=None):
        if cons_index is None: return sp.csr_matrix((3,3))
        x=np.asarray(x)
        coeff=(4.0,6.0,3.0)[cons_index]
        j=cons_index
        sign=1.0 if x[j]>=0.0 else -1.0
        H=np.zeros((3,3)); H[j,j]=coeff*sign
        return sp.csr_matrix(H)

sys.modules['pycutest']=types.SimpleNamespace(import_problem=lambda *a,**k:FakeCore1())
with tempfile.TemporaryDirectory() as td:
    out=Path(td)/'core1.qcq'
    info=mod.export_problem('CORE1',out,{},check=True,reformulation='auto')
    assert info['reformulation']=='core_xabs_exact_qcqp'
    assert info['n_original']==3 and info['n']==4
    assert info['abs_lifts']==1 and info['lift_variables']==[2]
    assert info['equalities']==3  # two original equalities plus one abs link
    assert info['inequalities']==1
    assert info['lift_validation_value'] < 1e-12
    assert info['lift_validation_gradient'] < 1e-12
    text=out.read_text()
    assert 'name abs_link_x2' in text
    assert 'name abs_link_x0' not in text
    assert 'name abs_link_x1' not in text
print('[PASS] CORE fixed-sign x*abs(x) rows remain direct quadratics')
