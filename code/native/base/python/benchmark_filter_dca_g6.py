#!/usr/bin/env python3
"""G6 research profiles: same-input Lazy-DCA / Ipopt with independent original KKT audits."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from benchmark_pglib_r3p29r8 import finite, sha, validate_endpoint
from live_process_r3p29r3 import run_logged
from qcq_model_r3p23 import read_qcq

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6

# Explicit, reproducible switches: profiles do not depend on environment variables.
KERNEL_FLAGS = ("--dca-active-evaluation", "--dca-packed-atoms", "--dca-lazy-merit")
BLOCK_FLAGS = KERNEL_FLAGS + ("--dca-block-ritz", "--dca-anderson")
TANGENT_FLAGS = BLOCK_FLAGS + ("--dca-tangent-curvature",)
RESEARCH_PROFILES = {
    "control": (),
    "kernel": KERNEL_FLAGS,
    "block": BLOCK_FLAGS,
    "anderson": KERNEL_FLAGS + ("--dca-anderson",),
    "tangent": TANGENT_FLAGS,
    "tangent-mumps": TANGENT_FLAGS + (
        "--dca-curvature-backend", "mumps", "--dca-sparse-fallback"),
    "transport": BLOCK_FLAGS + ("--dca-certificate-transport",),
    "equilibrated": BLOCK_FLAGS + ("--ldl-equilibration",),
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def lazy_command(args, binary, qcq, result, mode):
    command = [str(binary), "--problem", str(qcq), "--json", str(result),
            "--algorithm", "lazy-dca", "--dca-engine", "filter",
            "--linear-backend", "ldl", "--linear-metis", "1",
            "--feas-tol", "1e-6", "--opt-tol", "1e-6",
            "--dual-inf-tol", "1e-6", "--compl-inf-tol", "1e-6",
            "--kkt-berr-reject", "1e-7", "--max-outer", str(args.max_iter),
            "--time-limit", str(args.time_limit), "--display-level", str(args.display_level),
            "--live-iterations-every", str(args.live_every),
            "--live-header-every", str(args.live_header_every),
            "--dca-inner-max", str(args.inner_max),
            "--max-atoms", str(0 if mode == "scalar-control" else args.max_atoms),
            "--atom-batch-size", str(args.atom_batch_size),
            "--atom-event-every", str(args.atom_event_every),
            "--atom-events" if args.atom_events else "--no-atom-events", "--quiet"]
    if mode != "baseline":
        command.extend(RESEARCH_PROFILES[args.research_profile])
        command += ["--dca-rho-refinements", str(args.rho_refinements)]
        if args.initial_restoration: command.append("--dca-initial-restoration")
        if args.normal_weighted_atoms: command.append("--dca-normal-weighted-atoms")
    if args.no_model_reuse and mode != "baseline": command.append("--no-dca-model-reuse")
    if args.no_extrapolation and mode != "baseline": command.append("--no-dca-extrapolation")
    return command


def audit_ipopt(model, data, digest):
    if not data.get("success") or not data.get("kkt_passed") or data.get("model_sha256") != digest:
        raise ValueError("Ipopt did not pass the original certificate")
    # Recompute again in this parent process, using exported primal/dual vectors.
    x, lam, bounds = data["x"], data["constraint_multipliers"], data["bound_multipliers"]
    if len(x) != model.n or len(bounds) != model.n or len(lam) != len(model.equalities) + len(model.inequalities):
        raise ValueError("invalid Ipopt certificate dimensions")
    if not all(finite(v) for vector in (x, lam, bounds) for v in vector):
        raise ValueError("nonfinite Ipopt certificate")
    residuals = dict(model.residuals(x), **model.kkt_residuals(x, lam, bounds))
    if any(not finite(v) or not 0 <= v <= TOL for v in residuals.values()):
        raise ValueError("independent Ipopt KKT failure: " + str(residuals))
    return model.objective_value(x), residuals, True


def campaign(args, out, temporary):
    identity = json.loads((ROOT / "release_identity_g6.json").read_text())
    binary = args.binary.resolve(strict=True)
    banner = subprocess.check_output([str(binary), "--version"], text=True)
    if identity["build_id"] not in banner:
        raise ValueError("binary does not match release_identity_g6.json")
    baseline = args.baseline_binary.resolve(strict=True) if args.baseline_binary else None
    parent_identity = json.loads((ROOT / "release_identity_g5.json").read_text())
    if baseline:
        parent_banner = subprocess.check_output([str(baseline), "--version"], text=True)
        if parent_identity["build_id"] not in parent_banner:
            raise ValueError("baseline must be the unchanged G5 parent binary")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC="FALSE",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    peripheral = {}
    start = time.perf_counter()
    if args.frozen:
        metadata = json.loads((ROOT / "validation_g1/frozen/models.json").read_text())[args.frozen]
        compressed = ROOT / "validation_g1/frozen" / (args.frozen + ".qcq.gz")
        if sha(compressed) != metadata["gzip_sha256"]:
            raise ValueError("frozen gzip SHA256 mismatch")
        qcq = temporary / (args.frozen + ".qcq")
        with gzip.open(compressed, "rb") as source, qcq.open("wb") as destination:
            shutil.copyfileobj(source, destination, 4 * 1024 * 1024)
        provenance = {"frozen": args.frozen, **metadata}
    elif args.qcq:
        qcq = args.qcq.resolve(strict=True)
        provenance = {"qcq": str(qcq)}
    else:
        case = args.case.resolve(strict=True)
        qcq = temporary / "export.qcq"
        command = [sys.executable, str(ROOT / "python/export_pglib_qcqp_r3p23.py"), str(case),
                   "--output", str(qcq), "--json", str(out / "export.json"), "--flow-limit", "apparent",
                   "--power-flow-start", "--no-power-flow-voltage-continuation"]
        run_logged(command, out / "export.log", env=env, timeout=args.time_limit + 60,
                   live=args.display_level >= 4).check_returncode()
        provenance = {"case": str(case), "case_sha256": sha(case), "initial_point": "power-flow"}
    peripheral["input_preparation_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    digest = sha(qcq)
    if args.frozen and digest != metadata["model_sha256"]:
        raise ValueError("frozen QCQP SHA256 mismatch")
    peripheral["initial_hash_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    model = read_qcq(qcq)
    peripheral["independent_model_read_seconds"] = time.perf_counter() - start
    modes = ([] if args.skip_ipopt else ["ipopt"]) + (["baseline"] if baseline else []) + ["filter-dca"]
    if args.scalar_control:
        modes.append("scalar-control")
    records = []

    def run(mode, index, warmup):
        name = ("warmup_" if warmup else "run_") + str(index) + "_" + mode
        output = out / (name + ".json")
        if mode == "ipopt":
            command = [sys.executable, str(ROOT / "python/solve_qcq_ipopt_casadi_r3p23.py"), str(qcq),
                       "--json", str(output), "--tol", "1e-6", "--constr-viol-tol", "1e-6",
                       "--dual-inf-tol", "1e-6", "--compl-inf-tol", "1e-6", "--linear-solver", "mumps",
                       "--print-level", str(args.ipopt_print_level), "--max-iter", str(args.max_iter),
                       "--time-limit", str(args.time_limit)]
        else:
            command = lazy_command(args, baseline if mode == "baseline" else binary, qcq, output, mode)
        row = {"mode": mode, "warmup": warmup, "index": index, "command": command, "valid": False}
        started = time.perf_counter()
        if sha(qcq) != digest:
            raise ValueError("input changed before run")
        row["hash_before_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        try:
            process = run_logged(command, out / (name + ".log"), env=env,
                                 timeout=args.time_limit + 60, live=args.display_level > 0)
            row["returncode"] = process.returncode
        except subprocess.TimeoutExpired:
            row["returncode"] = -1
        row["process_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        if sha(qcq) != digest:
            raise ValueError("input changed during run")
        row["hash_after_seconds"] = time.perf_counter() - started
        try:
            data = json.loads(output.read_text())
            row.update(status=data.get("status"), message=data.get("message"),
                       attempted_solver_seconds=data.get("elapsed_seconds", data.get("solver_seconds")),
                       observed_iterations=data.get("outer_iterations", data.get("iterations")))
            for key, value in data.items():
                if key.startswith("dca_filter_") or key in ("lazy_newton_cost_seconds", "lazy_newton_core_solve_seconds",
                        "lazy_newton_evaluation_calls", "lazy_newton_maximum_direction_residual", "lazy_newton_pivoted_fallbacks"):
                    row[key] = value
            if row["returncode"] != 0:
                raise ValueError("process failure: " + str(data.get("message", data.get("status"))))
            started = time.perf_counter()
            if mode == "ipopt":
                objective, residuals, independent = audit_ipopt(model, data, digest)
                elapsed, iterations = data["solver_seconds"], data["iterations"]
                row["ipopt_peripheral_seconds"] = {k: data[k] for k in ("parse_seconds", "model_build_seconds", "total_wall_seconds")}
            else:
                expected_identity = parent_identity if mode == "baseline" else identity
                if any(data.get(k) != expected_identity[k] for k in ("version", "api_version", "build_id")):
                    raise ValueError("result identity mismatch")
                if not data.get("dca_filter_used") or data.get("effective_outer_algorithm") != "lazy-dca":
                    raise ValueError("wrong algorithm; explicit filter Lazy-DCA required")
                objective, residuals, independent = validate_endpoint(model, data, "filter-dca")
                elapsed, iterations = data["elapsed_seconds"], data["outer_iterations"]
                row["cli_timing"] = json.loads(Path(str(output) + ".timing.json").read_text())
            row["independent_certificate_seconds"] = time.perf_counter() - started
            if not finite(elapsed) or elapsed < 0 or not finite(objective):
                raise ValueError("nonfinite result/time")
            row.update(valid=True, objective=objective, residuals=residuals, independent_certificate=independent,
                       solver_seconds=elapsed, iterations=iterations)
        except (ValueError, KeyError, OSError, TypeError) as error:
            row["error"] = str(error)
        records.append(row)
        write_json(out / "records.json", records)
        print(name, "PASS" if row["valid"] else "FAIL", row.get("solver_seconds", row.get("error")), flush=True)

    for i in range(args.warmups):
        for mode in modes:
            run(mode, i + 1, True)
    for i in range(args.repetitions):
        offset = i % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            run(mode, i + 1, False)
    reference_values = [r["objective"] for r in records if r["valid"] and r["mode"] == "ipopt"]
    reference = statistics.median(reference_values) if reference_values else None
    summary = {"schema": "filter-dca-g6-v1", "release_status": "research",
               "research_profile": args.research_profile,
               "research_flags": list(RESEARCH_PROFILES[args.research_profile]), "identity": identity, "model_sha256": digest,
               "provenance": provenance, "dimensions": [model.n, len(model.equalities), len(model.inequalities)],
               "tolerance": TOL, "target_seconds": args.target_seconds, "display_level": args.display_level,
               "settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
               "shared_peripheral_seconds": peripheral, "timing_contract": "INSTALL_AND_TEST_G6.md",
               "environment": {k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")},
               "all_correct": all(r["valid"] for r in records), "modes": {}}
    for mode in modes:
        rows = [r for r in records if r["mode"] == mode and not r["warmup"]]
        good = [r for r in rows if r["valid"]]
        entry = {"measured_runs": len(rows), "valid_runs": len(good), "target_met": False,
                 "all_certificates_independent": len(good) == len(rows) and all(r["independent_certificate"] for r in good),
                 "observed_iterations": [r.get("observed_iterations") for r in rows]}
        if len(good) == len(rows):
            entry.update(median_solver_seconds=statistics.median(r["solver_seconds"] for r in good),
                         min_solver_seconds=min(r["solver_seconds"] for r in good),
                         max_solver_seconds=max(r["solver_seconds"] for r in good),
                         median_process_seconds=statistics.median(r["process_seconds"] for r in good),
                         iterations=[r["iterations"] for r in good], objectives=[r["objective"] for r in good])
            entry["target_met"] = entry["median_solver_seconds"] <= args.target_seconds
            if mode != "ipopt":
                entry["median_core_solve_seconds"] = statistics.median(r["lazy_newton_core_solve_seconds"] for r in good)
                entry["median_cost_seconds"] = {k: statistics.median(r["lazy_newton_cost_seconds"][k] for r in good)
                                                  for k in good[0]["lazy_newton_cost_seconds"]}
                entry["atoms_lagrangian"] = [r["dca_filter_atoms_lagrangian"] for r in good]
                entry["atoms_inequalities"] = [r["dca_filter_atoms_inequalities"] for r in good]
                for field in ("models", "model_reuses", "curvature_factors", "inner_steps", "subspace_spectra"):
                    if "dca_filter_" + field in good[0]:
                        entry[field] = [r["dca_filter_" + field] for r in good]
            if reference is not None:
                entry["objective_gaps_to_ipopt"] = [r["objective"] - reference for r in good]
        summary["modes"][mode] = entry
    write_json(out / "benchmark_summary.json", summary)
    if args.display_level != 1:
        print(json.dumps(summary["modes"], indent=2), flush=True)
    print("CORRECTNESS=" + ("PASS" if summary["all_correct"] else "FAIL"), flush=True)
    print("SUMMARY=" + str(out / "benchmark_summary.json"), flush=True)
    return 0 if summary["all_correct"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frozen", "--problem", dest="frozen", choices=("case118", "case300", "case1354", "case2869"))
    source.add_argument("--qcq", type=Path)
    source.add_argument("--case", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path, help="unchanged G5 parent binary, same input and numerical tolerances")
    parser.add_argument("--research-profile", choices=tuple(RESEARCH_PROFILES), default="block",
                        help="experimental G6 profile (default: block); all solver commands are recorded")
    parser.add_argument("--dca-rho-refinements", dest="rho_refinements", type=int, choices=range(13), default=3)
    parser.add_argument("--dca-initial-restoration", dest="initial_restoration", action="store_true")
    parser.add_argument("--dca-normal-weighted-atoms", dest="normal_weighted_atoms", action="store_true")
    parser.add_argument("--no-extrapolation", action="store_true", help="disable guarded local DCA extrapolation")
    parser.add_argument("--no-model-reuse", action="store_true", help="rebuild at every accepted point in the selected G6 profile")
    parser.add_argument("--base", type=Path, default=Path("/data/projects/IPDCA_QCQP"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-ipopt", action="store_true")
    parser.add_argument("--scalar-control", action="store_true", help="also run the same DCA with max-atoms=0")
    parser.add_argument("--reuse-qcq", action="store_true", help="compatibility flag; QCQ inputs are always read without copying")
    parser.add_argument("--display-level", type=int, choices=range(5), default=0)
    parser.add_argument("--ipopt-print-level", type=int, choices=range(13), default=0)
    parser.add_argument("--live-every", type=int, default=1)
    parser.add_argument("--live-header-every", type=int, default=20)
    parser.add_argument("--inner-max", "--dca-inner-max", dest="inner_max", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=500)
    parser.add_argument("--atom-batch-size", type=int, default=2)
    parser.add_argument("--atom-event-every", type=int, default=1)
    parser.add_argument("--atom-events", dest="atom_events", action="store_true", default=True)
    parser.add_argument("--no-atom-events", dest="atom_events", action="store_false")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--time-limit", type=float, default=1800)
    parser.add_argument("--target-seconds", type=float, default=2)
    args = parser.parse_args()
    if min(args.threads, args.repetitions, args.max_iter, args.atom_batch_size, args.atom_event_every,
           args.live_every, args.live_header_every, args.inner_max) < 1 or args.inner_max > 100 or min(args.max_atoms, args.warmups) < 0:
        parser.error("invalid step, atom, repetition or display counts")
    if not all(math.isfinite(v) and v > 0 for v in (args.time_limit, args.target_seconds)):
        parser.error("time limits must be finite and positive")
    if args.reuse_qcq and not args.qcq:
        parser.error("--reuse-qcq requires --qcq")
    label = args.frozen or (args.qcq or args.case).stem
    out = (args.output or args.base / "results" / ("R3P28G6_" + label + "_" + str(time.time_ns()))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("CAMPAIGN=" + str(out), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="input_", dir=out) as temporary:
            return campaign(args, out, Path(temporary))
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        write_json(out / "campaign_error.json", {"correctness": False, "error": str(error)})
        print("CORRECTNESS=FAIL: " + str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
