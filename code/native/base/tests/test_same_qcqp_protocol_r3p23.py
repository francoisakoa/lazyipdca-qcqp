#!/usr/bin/env python3
"""Self-contained contract test for the r3p23 same-file protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from export_pglib_qcqp_r3p23 import sha256_file  # noqa: E402
from run_pglib_qcqp_r3p23 import run_pair  # noqa: E402

QCQ = """LAZYIPDCA_QCQP_V1
name same_file_toy
n 1
x0 0
lb -INF
ub INF
objective
quadratic
name objective
constant 1
linear 0
hessian_nnz 1
0 0 2
end_quadratic
equalities 0
inequalities 0
end_problem
"""

FAKE_SOLVER = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys

if "--version" in sys.argv:
    print("fake-ipopt-direct" if "ipopt" in Path(sys.argv[0]).name
          else "fake-lazyipdca-r3p23")
    raise SystemExit(0)

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

casadi = "casadi" in Path(sys.argv[0]).name
model = Path(sys.argv[1] if casadi else value("--problem")).resolve()
output = Path(value("--json")).resolve()
digest = hashlib.sha256(model.read_bytes()).hexdigest()
common = {
    "objective": 1.0,
    "iterations": 3,
    "consumed_model": str(model),
    "consumed_sha256": digest,
}
native = {
    "status": "solved",
    "equality_violation": 0.0,
    "inequality_violation": 0.0,
    "bound_violation": 0.0,
    "dual_infeasibility_inf": 1.0e-8,
    "complementarity_inf": 0.0,
    "scaled_nlp_error": 1.0e-8,
    "elapsed_seconds": 0.02 if "ipopt" in Path(sys.argv[0]).name else 0.03,
    "outer_iterations": 2,
    "inner_iterations": 4,
    "kkt_factorizations": 4,
}
casadi_result = {
    "status": "Solve_Succeeded",
    "success": True,
    "kkt_passed": True,
    "solver_seconds": 0.02,
    "residuals": {
        "equality_inf": 0.0,
        "inequality_inf": 0.0,
        "bound_inf": 0.0,
        "dual_inf": 1.0e-8,
        "complementarity_inf": 0.0,
    },
    "dimensions": {"variables": 1, "equalities": 0, "inequalities": 0},
}
payload = {**common, **(casadi_result if casadi else native)}
output.write_text(json.dumps(payload))
'''


with tempfile.TemporaryDirectory() as temporary_name:
    temporary = Path(temporary_name)
    model = temporary / "one.qcq"
    model.write_text(QCQ)
    expected_hash = sha256_file(model)

    lazy = temporary / "fake_lazy"
    ipopt = temporary / "fake_ipopt"
    lazy.write_text(FAKE_SOLVER)
    ipopt.write_text(FAKE_SOLVER)
    lazy.chmod(lazy.stat().st_mode | stat.S_IXUSR)
    ipopt.chmod(ipopt.stat().st_mode | stat.S_IXUSR)

    pair = run_pair(
        model, lazy, ipopt, temporary / "pair",
        phase1_threads=1, max_outer=10, max_inner=10,
        ipopt_max_iterations=10, time_limit=10.0)
    assert pair["pair_passed"], pair
    assert pair["model_integrity_passed"], pair
    assert pair["model"]["sha256"] == expected_hash, pair
    assert pair["model"]["n"] == 1, pair
    assert pair["model"]["equalities"] == 0, pair
    assert pair["model"]["inequalities"] == 0, pair
    assert pair["execution_order"] == ["ipopt", "lazyipdca"], pair
    for label in ("ipopt", "lazyipdca"):
        result = pair[label]["result"]
        assert result["consumed_model"] == str(model.resolve()), result
        assert result["consumed_sha256"] == expected_hash, result
    stored = json.loads((temporary / "pair" / "summary.json").read_text())
    assert stored["pair_passed"] is True, stored
    assert stored["comparison"]["objective_absolute_difference"] == 0.0

    casadi = temporary / "fake_casadi_ipopt"
    casadi.write_text(FAKE_SOLVER)
    casadi.chmod(casadi.stat().st_mode | stat.S_IXUSR)
    casadi_pair = run_pair(
        model, lazy, casadi, temporary / "casadi_pair",
        ipopt_interface_name="casadi", phase1_threads=1,
        max_outer=10, max_inner=10, ipopt_max_iterations=10,
        time_limit=10.0)
    assert casadi_pair["pair_passed"], casadi_pair
    assert casadi_pair["ipopt_interface"] == "casadi", casadi_pair
    assert casadi_pair["ipopt"]["command"][1] == str(model.resolve())
    assert casadi_pair["ipopt"]["result"]["status"] == "solved"

print("[PASS] r3p23 gives Ipopt and LazyIPDCA one identical physical QCQP")
