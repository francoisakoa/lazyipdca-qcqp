#!/usr/bin/env python3
"""Contract test for the PyCUTEst Hessian API used by the QCQP exporter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_cutest_qcqp", ROOT / "python" / "export_cutest_qcqp.py"
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


class FakeProblem:
    name = "FAKE_CONSTRAINED_QCQP"
    n = 2
    m = 1
    x0 = np.array([0.25, -0.5])
    bl = np.array([-1.0e20, -1.0e20])
    bu = np.array([+1.0e20, +1.0e20])
    cl = np.array([-1.0e20])
    cu = np.array([0.0])
    is_eq_cons = np.array([False])

    # objective = .5*x' diag(-2,4) x + [1,-3]'x + 2
    Q0 = np.array([[-2.0, 0.0], [0.0, 4.0]])
    c0 = np.array([1.0, -3.0])
    r0 = 2.0

    # constraint = .5*x' diag(2,0) x + [0,1]'x - 1 <= 0
    Q1 = np.array([[2.0, 0.0], [0.0, 0.0]])
    c1 = np.array([0.0, 1.0])
    r1 = -1.0

    def obj(self, x, gradient=False):
        x=np.asarray(x)
        v=float(self.r0 + self.c0@x + 0.5*x@self.Q0@x)
        if gradient:
            return v, self.c0 + self.Q0@x
        return v

    def cons(self, x, index=None, gradient=False):
        x=np.asarray(x)
        v=float(self.r1 + self.c1@x + 0.5*x@self.Q1@x)
        g=self.c1 + self.Q1@x
        if index is None:
            if gradient:
                return np.array([v]), np.array([g])
            return np.array([v])
        assert index == 0
        return (v, g) if gradient else v

    def grad(self, x, index=None):
        if index is None:
            return self.c0 + self.Q0@np.asarray(x)
        assert index == 0
        return self.c1 + self.Q1@np.asarray(x)

    def isphess(self, x, cons_index=None):
        del x
        return sp.coo_matrix(self.Q0 if cons_index is None else self.Q1)

    def sphess(self, x, v=None):
        raise AssertionError("exporter must not call sphess(x) for objective Hessian")


fake_pycutest = types.SimpleNamespace(
    import_problem=lambda *args, **kwargs: FakeProblem()
)
sys.modules["pycutest"] = fake_pycutest

with tempfile.TemporaryDirectory() as td:
    out=Path(td)/"fake.qcq"
    info=mod.export_problem("FAKE", out, {}, check=True)
    assert info["n"] == 2
    assert info["cutest_constraints"] == 1
    assert info["inequalities"] == 1
    text=out.read_text()
    assert "LAZYIPDCA_QCQP_V1" in text
    assert "name objective" in text
    assert "name c0_upper" in text

print("[PASS] PyCUTEst constrained Hessian API contract")
