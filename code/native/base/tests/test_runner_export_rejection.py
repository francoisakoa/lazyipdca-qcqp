#!/usr/bin/env python3
"""The CUTEst runner must turn a non-QCQP validation failure into explicit JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

FAKE_PYCUTEST = r'''
import numpy as np
import scipy.sparse as sp

class Cubic:
    name = "CUBIC_RUNNER"
    n = 1
    m = 0
    x0 = np.array([0.0])
    bl = np.array([-1.0e20])
    bu = np.array([+1.0e20])
    cl = np.array([])
    cu = np.array([])
    is_eq_cons = np.array([], dtype=bool)
    def obj(self, x, gradient=False):
        x = np.asarray(x)
        v = float(x[0]**3)
        g = np.array([3.0*x[0]**2])
        return (v, g) if gradient else v
    def isphess(self, x, cons_index=None):
        x = np.asarray(x)
        return sp.coo_matrix(np.array([[6.0*x[0]]]))
    def cons(self, x, gradient=False, index=None):
        if gradient:
            return np.array([]), np.empty((0,1))
        return np.array([])

def import_problem(name, **kwargs):
    assert name == "CUBIC_RUNNER"
    return Cubic()
'''

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "pycutest.py").write_text(FAKE_PYCUTEST)
    work = td / "work"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(td) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "python" / "run_cutest_qcqp.py"),
            "CUBIC_RUNNER",
            "--binary", "/bin/true",
            "--work-dir", str(work),
            "--reformulation", "none",
            "--quiet",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    assert proc.returncode == 2, proc.stdout
    assert "[EXPORT-REJECTED]" in proc.stdout, proc.stdout
    data = json.loads((work / "CUBIC_RUNNER.json").read_text())
    assert data["status"] == "not_qcqp", data
    assert data["quadratic_validation_passed"] is False, data
    assert data["dual_infeasibility_inf"] is None, data
    assert data["scaled_nlp_error"] is None, data

print("[PASS] runner records non-QCQP rejection with unavailable certificate")
