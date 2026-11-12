#!/usr/bin/env python3
"""Reject smooth CUTEst models that only look quadratic at the expansion point."""
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


class Base:
    n = 2
    m = 1
    x0 = np.zeros(2)
    bl = np.array([-1.0e20, -1.0e20])
    bu = np.array([+1.0e20, +1.0e20])
    cl = np.array([-1.0e20])
    cu = np.array([0.0])
    is_eq_cons = np.array([False])

    def grad(self, x, index=None):
        if index is None:
            return self.obj(x, gradient=True)[1]
        return self.cons(x, index=index, gradient=True)[1]


class TrueQuadratic(Base):
    name = "TRUE_QUADRATIC"
    Q0 = np.diag([2.0, 4.0])
    Q1 = np.diag([2.0, 0.0])

    def obj(self, x, gradient=False):
        x = np.asarray(x)
        v = float(1.0 + x[0] - 2.0*x[1] + 0.5*x@self.Q0@x)
        g = np.array([1.0, -2.0]) + self.Q0@x
        return (v, g) if gradient else v

    def cons(self, x, index=None, gradient=False):
        x = np.asarray(x)
        v = float(-1.0 + x[1] + 0.5*x@self.Q1@x)
        g = np.array([0.0, 1.0]) + self.Q1@x
        if index is None:
            return (np.array([v]), np.array([g])) if gradient else np.array([v])
        assert index == 0
        return (v, g) if gradient else v

    def isphess(self, x, cons_index=None):
        del x
        return sp.coo_matrix(self.Q0 if cons_index is None else self.Q1)


class CubicObjective(TrueQuadratic):
    name = "CUBIC_OBJECTIVE"

    def obj(self, x, gradient=False):
        x = np.asarray(x)
        v = float(x[0]**3 + x[1]**2)
        g = np.array([3.0*x[0]**2, 2.0*x[1]])
        return (v, g) if gradient else v

    def isphess(self, x, cons_index=None):
        if cons_index is None:
            x = np.asarray(x)
            return sp.coo_matrix(np.diag([6.0*x[0], 2.0]))
        return sp.coo_matrix(self.Q1)


class QuarticConstraint(TrueQuadratic):
    name = "QUARTIC_CONSTRAINT"

    def cons(self, x, index=None, gradient=False):
        x = np.asarray(x)
        v = float(x[0]**4 + x[1] - 1.0)
        g = np.array([4.0*x[0]**3, 1.0])
        if index is None:
            return (np.array([v]), np.array([g])) if gradient else np.array([v])
        assert index == 0
        return (v, g) if gradient else v

    def isphess(self, x, cons_index=None):
        if cons_index is None:
            return sp.coo_matrix(self.Q0)
        x = np.asarray(x)
        return sp.coo_matrix(np.diag([12.0*x[0]**2, 0.0]))


PROBLEMS = {
    "TRUE": TrueQuadratic,
    "CUBIC": CubicObjective,
    "QUARTIC": QuarticConstraint,
}
sys.modules["pycutest"] = types.SimpleNamespace(
    import_problem=lambda name, **kwargs: PROBLEMS[name]()
)

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    info = mod.export_problem("TRUE", td / "true.qcq", {}, check=True, reformulation="none")
    assert info["quadratic_validation_probes"] >= 3
    assert info["quadratic_validation_objective_error"] < 1e-12
    assert info["quadratic_validation_constraint_error"] < 1e-12

    for name, expected in (("CUBIC", "objective"), ("QUARTIC", "constraints")):
        try:
            mod.export_problem(name, td / f"{name}.qcq", {}, check=True, reformulation="none")
        except RuntimeError as exc:
            text = str(exc)
            assert expected in text and "probe scale" in text, text
        else:
            raise AssertionError(f"{name} was incorrectly accepted as a QCQP")

print("[PASS] multi-scale exporter accepts true quadratics and rejects cubic/quartic models")
