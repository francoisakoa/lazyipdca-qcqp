#!/usr/bin/env python3
"""Exercise the real CLI, campaign, original audit and timing/display contract."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from benchmark_pglib_r3p29r8 import validate_endpoint
from qcq_model_r3p23 import read_qcq

MODEL = """LAZYIPDCA_QCQP_V1
name g3_campaign_nonconvex
n 2
x0 0.2 0.2
lb -1 -1
ub 1 1
objective
quadratic
name objective
constant 0
linear 0 0
hessian_nnz 2
0 0 -2
1 1 -2
end_quadratic
equalities 1
quadratic
name affine
constant 0
linear 1 -1
hessian_nnz 0
end_quadratic
inequalities 0
end_problem
"""


def main():
    binary = Path(sys.argv[1]).resolve(strict=True)
    backend_query = subprocess.run(
        [str(binary), "--linear-backends"], capture_output=True,
        text=True, check=False)
    assert backend_query.returncode == 0, backend_query.stdout
    if "ldl" not in {item.strip() for item in
                     backend_query.stdout.strip().split(",")}:
        print("[SKIP] historical G3 campaign requires the uncompiled "
              "SuiteSparse LDL backend")
        return 77
    identity = json.loads((ROOT / "release_identity_g10.json").read_text())
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        problem = directory / "nonconvex.qcq"
        problem.write_text(MODEL)
        for level, cap in ((1, 0), (3, 1)):
            out = directory / ("campaign" + str(level))
            command = [sys.executable, str(ROOT / "python/benchmark_filter_dca_g10.py"),
                       "--qcq", str(problem), "--binary", str(binary), "--output", str(out),
                       "--skip-ipopt", "--warmups", "0", "--repetitions", "1",
                       "--display-level", str(level), "--max-atoms", str(cap),
                       "--atom-event-every", "1", "--time-limit", "20"]
            command += ["--dca-rho-refinements", str(0 if level == 1 else 2)]
            if level == 3: command += ["--dca-initial-restoration", "--dca-normal-weighted-atoms"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            assert process.returncode == 0, process.stdout + process.stderr
            data = json.loads((out / "run_1_filter-dca.json").read_text())
            summary = json.loads((out / "benchmark_summary.json").read_text())
            assert summary["all_correct"] and summary["modes"]["filter-dca"]["all_certificates_independent"]
            for key in ("version", "api_version", "build_id"):
                assert data[key] == identity[key], (key, data.get(key), identity[key])
                assert summary["identity"][key] == identity[key], key
            assert summary["schema"] == "filter-dca-g10-v1"
            assert summary["target_mode"] == "filter-dca"
            assert isinstance(summary["target_met"], bool)
            assert "CORRECTNESS=PASS" in process.stdout and "TARGET=" in process.stdout
            assert data["dca_filter_rho_refinement_budget"] == (0 if level == 1 else 2)
            assert data["dca_filter_initial_restoration_requested"] == (level == 3)
            assert data["dca_filter_normal_atom_weighting"] == (level == 3)
            assert data["objective"] < -1.99999
            assert data["dca_filter_atoms_lagrangian"] + data["dca_filter_atoms_inequalities"] <= cap
            assert data["dca_filter_peak_stored_atoms"] <= cap
            for flag in ("active_evaluation", "packed_atoms", "lazy_merit", "block_ritz", "anderson"):
                assert data["dca_filter_" + flag] is True, flag
            assert data["dca_filter_tangent_certificate"] is False
            assert data["dca_filter_curvature_backend"] == "ldl"
            assert data["dca_filter_inner_steps"] <= data["dca_filter_models"] * 8
            assert " iter " in process.stdout
            if level == 1:
                assert "aOE" not in process.stdout and "Timing (seconds" not in process.stdout
            else:
                assert "aOE" in process.stdout and "rOE<=" in process.stdout and "Core evaluations" in process.stdout
            costs = data["lazy_newton_cost_seconds"]
            assert all(v >= 0 for v in costs.values())
            assert abs(sum(costs.values()) - data["elapsed_seconds"]) < 1e-3
            core = sum(v for k, v in costs.items() if k not in ("preparation", "certificate", "diagnostics", "cleanup"))
            assert abs(core - data["lazy_newton_core_solve_seconds"]) < 1e-9
            calls = data["lazy_newton_evaluation_calls"]["core"]
            for family in calls.values():
                assert family["values"] >= family["model_values"]
                assert family["gradients"] >= family["model_gradients"]
            assert calls["objective"]["model_values"] > 0
            timing = json.loads((out / "run_1_filter-dca.json.timing.json").read_text())
            assert timing
            # A stale/incorrect multiplier must fail the separate coefficient audit.
            data["certificate_lower_multipliers"][0] += 10
            try:
                validate_endpoint(read_qcq(problem), data, "filter-dca")
            except ValueError:
                pass
            else:
                raise AssertionError("altered multiplier passed the independent certificate")
        assert problem.read_text() == MODEL, "input mutated"
        # A purported baseline must really be the unchanged G7, not the same
        # new binary used twice under two labels.
        out = directory / "wrong_baseline"
        rejected = subprocess.run([sys.executable, str(ROOT / "python/benchmark_filter_dca_g10.py"),
                                   "--qcq", str(problem), "--binary", str(binary),
                                   "--baseline-binary", str(binary), "--output", str(out),
                                   "--skip-ipopt", "--warmups", "0", "--repetitions", "1"],
                                  capture_output=True, text=True, timeout=20)
        assert rejected.returncode != 0 and "unchanged G7" in rejected.stderr
    print("PASS G9 campaign identity, displays, cap, cost accounting and independent rejection")


if __name__ == "__main__":
    raise SystemExit(main())
