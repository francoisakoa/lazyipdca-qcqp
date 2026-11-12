#!/usr/bin/env python3
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

binary = Path(sys.argv[1])
data = json.loads(subprocess.check_output([str(binary), "--qp-defaults"], text=True))
expected = {
    "qp_option": "C",
    "qp_amrs": True,
    "qp_init": 2,
    "qp_backend": "ldl",
    "qp_rho_trial_every": 2,
    "qp_max_atoms": 500,
    "qp_seed_coverage": 0.25,
    "qp_seed_stall_tol": 0,
    "qp_seed_patience": 999,
    "qp_atom_events": True,
    "qp_atom_event_every": 50,
    "qp_bound_push": 0,
    "qp_bp_push_kappa": 1e-4,
    "qp_bp_margin_tol": 1e-4,
    "qp_escapes": 3,
    "qp_mehrotra": True,
    "qp_sigma_max": 1e14,
    "qp_model_solved_tol": 1e-12,
    "qp_dca_stop": True,
    "qp_dca_stall_tol_rel": 1e-10,
    "qp_dca_stall_iters": 5,
    "qp_dca_monotone_tol_abs": 1e-6,
    "qp_bp_adaptive": True,
}
for key, value in expected.items():
    if data.get(key) != value:
        raise SystemExit(f"[FAIL] {key}: got {data.get(key)!r}, expected {value!r}")
print(f"[PASS] {len(expected)} qp_ defaults match profile C + AMRS")
