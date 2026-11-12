#!/usr/bin/env python3
"""Benchmark Ipopt-direct and r3p23 on one immutable case118 QCQP export."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
from typing import Any

from export_pglib_qcqp_r3p23 import (
    export_case,
    sha256_file,
    write_json_atomic,
)
from run_pglib_qcqp_r3p23 import finite_float, run_pair

BENCHMARK_SCHEMA = "lazyipdca-ipopt-same-qcqp-benchmark-v1"


def parser() -> argparse.ArgumentParser:
    base = Path(os.environ.get("BASE", "/data/projects/IPDCA_QCQP"))
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case", type=Path,
                    default=base / "pglib-opf" /
                    "pglib_opf_case118_ieee.m")
    ap.add_argument("--lazy-binary", type=Path,
                    default=(Path(os.environ["BIN"])
                             if os.environ.get("BIN") else
                             root / "build" / "lazyipdca_qcqp_cli"))
    ap.add_argument("--ipopt-binary", type=Path,
                    default=(Path(os.environ["IPOPT_BIN"])
                             if os.environ.get("IPOPT_BIN") else
                             root / "build" / "ipopt_qcqp_cli"))
    ap.add_argument("--base", type=Path, default=base)
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--benchmark-id", default=None)
    ap.add_argument("--repetitions", "-n", type=int, default=5)
    ap.add_argument("--target-seconds", type=float, default=10.0,
                    help="upper bound for median LazyIPDCA solver time")
    ap.add_argument("--eq-tol", type=float, default=1.0e-6)
    ap.add_argument("--ineq-tol", type=float, default=1.0e-6)
    ap.add_argument("--bound-tol", type=float, default=1.0e-6)
    ap.add_argument("--dual-tol", type=float, default=1.0e-6)
    ap.add_argument("--compl-tol", type=float, default=1.0e-6)
    ap.add_argument("--objective-abs-tol", type=float, default=1.0e-3)
    ap.add_argument("--objective-rel-tol", type=float, default=1.0e-8)
    ap.add_argument("--flow-limit", choices=("current", "apparent", "none"),
                    default="apparent")
    ap.add_argument("--no-angle-limits", action="store_true")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--power-flow-start", dest="power_flow_start",
                       action="store_true")
    start.add_argument("--no-power-flow-start", dest="power_flow_start",
                       action="store_false")
    ap.set_defaults(power_flow_start=True)
    continuation = ap.add_mutually_exclusive_group()
    continuation.add_argument(
        "--power-flow-voltage-continuation",
        dest="power_flow_voltage_continuation", action="store_true")
    continuation.add_argument(
        "--no-power-flow-voltage-continuation",
        dest="power_flow_voltage_continuation", action="store_false")
    ap.set_defaults(power_flow_voltage_continuation=False)
    ap.add_argument("--linear-backend",
                    choices=("native", "ldl", "qdldl", "mumps"),
                    default="ldl")
    ap.add_argument("--phase1-threads", type=int, default=8)
    ap.add_argument("--max-outer", type=int, default=1000)
    ap.add_argument("--max-inner", type=int, default=1000)
    ap.add_argument("--ipopt-max-iterations", type=int, default=3000)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--ipopt-linear-solver", default="mumps")
    ap.add_argument("--ipopt-print-level", type=int, default=5)
    ap.add_argument("--ipopt-interface",
                    choices=("auto", "native", "casadi"), default="auto",
                    help="native uses --problem; CasADi uses positional MODEL")
    ap.add_argument("--lazy-extra-args", default="",
                    help="quoted additional r3p23 CLI arguments")
    ap.add_argument("--live-output", action="store_true")
    ap.add_argument("--stop-on-failure", action="store_true")
    return ap


def metric(result: dict[str, Any], name: str) -> float | None:
    return finite_float(result.get(name))


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.repetitions < 1:
        parser().error("--repetitions must be at least one")
    positive = {
        "target seconds": args.target_seconds,
        "equality tolerance": args.eq_tol,
        "inequality tolerance": args.ineq_tol,
        "bound tolerance": args.bound_tol,
        "dual tolerance": args.dual_tol,
        "complementarity tolerance": args.compl_tol,
        "objective absolute tolerance": args.objective_abs_tol,
        "objective relative tolerance": args.objective_rel_tol,
        "phase1 threads": args.phase1_threads,
        "max outer": args.max_outer,
        "max inner": args.max_inner,
        "Ipopt max iterations": args.ipopt_max_iterations,
        "time limit": args.time_limit,
    }
    invalid = next((name for name, value in positive.items()
                    if not math.isfinite(float(value)) or value <= 0), None)
    if invalid:
        parser().error(f"{invalid} must be finite and positive")

    case = args.case.expanduser().resolve()
    lazy_binary = args.lazy_binary.expanduser().resolve()
    ipopt_binary = args.ipopt_binary.expanduser().resolve()
    if not case.is_file():
        parser().error(f"case file is missing: {case}")
    for label, binary in (("LazyIPDCA", lazy_binary),
                          ("Ipopt-direct", ipopt_binary)):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser().error(
                f"{label} executable is missing or not executable: {binary}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    benchmark_id = (args.benchmark_id or
                    f"R3P23_SAME_QCQP_CASE118_{stamp}")
    results_root = (args.results_root or args.base / "results").resolve()
    benchmark_dir = results_root / benchmark_id
    shared_dir = benchmark_dir / "shared_model"
    runs_dir = benchmark_dir / "runs"
    shared_dir.mkdir(parents=True, exist_ok=False)
    runs_dir.mkdir()

    # This is the sole export in the entire benchmark.  Both binaries receive
    # this exact absolute path in every repetition; no per-run copy is made.
    model_path = shared_dir / f"{case.stem}.qcq"
    export_manifest_path = shared_dir / f"{case.stem}.export.json"
    try:
        export_manifest = export_case(
            case, model_path,
            flow_limit=args.flow_limit,
            angle_limits=not args.no_angle_limits,
            power_flow_start=args.power_flow_start,
            power_flow_voltage_continuation=
                args.power_flow_voltage_continuation)
        export_manifest["export_count"] = 1
        write_json_atomic(export_manifest_path, export_manifest)
        model_path.chmod(0o444)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[EXPORT FAILURE] {exc}", file=sys.stderr)
        return 2

    fingerprint = export_manifest["model_sha256"]
    print("[ONE SHARED QCQP EXPORT]")
    print(f"model      : {model_path}")
    print(f"sha256     : {fingerprint}")
    print(f"dimensions : {export_manifest['n']} / "
          f"{export_manifest['equalities']} / "
          f"{export_manifest['inequalities']}")

    rows: list[dict[str, Any]] = []
    pair_summaries: list[str] = []
    lazy_extra_args = shlex.split(args.lazy_extra_args)
    for index in range(1, args.repetitions + 1):
        order = "ipopt-first" if index % 2 else "lazy-first"
        run_dir = runs_dir / f"run_{index:03d}"
        print(f"\n[PAIR {index}/{args.repetitions}] order={order}", flush=True)
        try:
            pair = run_pair(
                model_path, lazy_binary, ipopt_binary, run_dir,
                order=order,
                linear_backend=args.linear_backend,
                phase1_threads=args.phase1_threads,
                max_outer=args.max_outer,
                max_inner=args.max_inner,
                ipopt_max_iterations=args.ipopt_max_iterations,
                time_limit=args.time_limit,
                equality_tolerance=args.eq_tol,
                inequality_tolerance=args.ineq_tol,
                bound_tolerance=args.bound_tol,
                dual_tolerance=args.dual_tol,
                complementarity_tolerance=args.compl_tol,
                objective_absolute_tolerance=args.objective_abs_tol,
                objective_relative_tolerance=args.objective_rel_tol,
                ipopt_linear_solver=args.ipopt_linear_solver,
                ipopt_print_level=args.ipopt_print_level,
                ipopt_interface_name=args.ipopt_interface,
                lazy_extra_args=lazy_extra_args,
                live_output=args.live_output)
            ipopt = pair["ipopt"]["result"]
            lazy = pair["lazyipdca"]["result"]
            comparison = pair["comparison"]
            row = {
                "repetition": index,
                "execution_order": order,
                "model_sha256": pair["model"]["sha256"],
                "model_integrity_passed": pair["model_integrity_passed"],
                "pair_passed": pair["pair_passed"],
                "ipopt_status": ipopt.get("status"),
                "ipopt_return_code": pair["ipopt"]["return_code"],
                "ipopt_kkt_pass": pair["ipopt"]["certificate_pass"],
                "ipopt_objective": metric(ipopt, "objective"),
                "ipopt_equality": metric(ipopt, "equality_violation"),
                "ipopt_inequality": metric(ipopt, "inequality_violation"),
                "ipopt_bound": metric(ipopt, "bound_violation"),
                "ipopt_dual": metric(ipopt, "dual_infeasibility_inf"),
                "ipopt_complementarity": metric(
                    ipopt, "complementarity_inf"),
                "ipopt_iterations": metric(ipopt, "iterations"),
                "ipopt_solver_seconds": metric(ipopt, "elapsed_seconds"),
                "ipopt_wall_seconds": pair["ipopt"]["wall_seconds"],
                "lazy_status": lazy.get("status"),
                "lazy_return_code": pair["lazyipdca"]["return_code"],
                "lazy_kkt_pass": pair["lazyipdca"]["certificate_pass"],
                "lazy_objective": metric(lazy, "objective"),
                "lazy_equality": metric(lazy, "equality_violation"),
                "lazy_inequality": metric(lazy, "inequality_violation"),
                "lazy_bound": metric(lazy, "bound_violation"),
                "lazy_dual": metric(lazy, "dual_infeasibility_inf"),
                "lazy_complementarity": metric(
                    lazy, "complementarity_inf"),
                "lazy_outer_iterations": metric(
                    lazy, "outer_iterations"),
                "lazy_inner_iterations": metric(
                    lazy, "inner_iterations"),
                "lazy_kkt_factorizations": metric(
                    lazy, "kkt_factorizations"),
                "lazy_solver_seconds": metric(lazy, "elapsed_seconds"),
                "lazy_wall_seconds": pair["lazyipdca"]["wall_seconds"],
                "objective_absolute_difference":
                    comparison["objective_absolute_difference"],
                "objective_relative_difference":
                    comparison["objective_relative_difference"],
                "objective_match": comparison["objective_match"],
                "summary_json": str(run_dir / "summary.json"),
                "error": "",
            }
            pair_summaries.append(str(run_dir / "summary.json"))
        except (OSError, RuntimeError, ValueError,
                subprocess.SubprocessError) as exc:
            row = {
                "repetition": index,
                "execution_order": order,
                "model_sha256": sha256_file(model_path),
                "model_integrity_passed": False,
                "pair_passed": False,
                "summary_json": str(run_dir / "summary.json"),
                "error": str(exc),
            }
            print(f"[PAIR FAILURE] {exc}", file=sys.stderr)
        rows.append(row)
        print(
            f"[PAIR RESULT] pass={row.get('pair_passed')} "
            f"ipopt_s={row.get('ipopt_solver_seconds')} "
            f"lazy_s={row.get('lazy_solver_seconds')} "
            f"objective_abs={row.get('objective_absolute_difference')}",
            flush=True)
        if args.stop_on_failure and not row.get("pair_passed", False):
            break

    final_hash = sha256_file(model_path)
    integrity_pass = final_hash == fingerprint and all(
        row.get("model_sha256") == fingerprint and
        row.get("model_integrity_passed", False) for row in rows)
    ipopt_times = [value for row in rows
                   if (value := finite_float(
                       row.get("ipopt_solver_seconds"))) is not None]
    lazy_times = [value for row in rows
                  if (value := finite_float(
                      row.get("lazy_solver_seconds"))) is not None]
    objective_differences = [value for row in rows
                             if (value := finite_float(row.get(
                                 "objective_absolute_difference")))
                             is not None]
    ipopt_median = median(ipopt_times)
    lazy_median = median(lazy_times)
    time_ratio = (
        lazy_median / ipopt_median
        if lazy_median is not None and ipopt_median is not None and
        ipopt_median > 0.0 else None)
    completed_pairs = sum(bool(row.get("pair_passed")) for row in rows)
    target_pass = lazy_median is not None and lazy_median < args.target_seconds
    benchmark_pass = (
        len(rows) == args.repetitions and
        completed_pairs == args.repetitions and
        integrity_pass and target_pass)

    csv_path = benchmark_dir / "benchmark.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    benchmark: dict[str, Any] = {
        "schema": BENCHMARK_SCHEMA,
        "benchmark_id": benchmark_id,
        "case": str(case),
        "shared_model": str(model_path),
        "shared_model_sha256": fingerprint,
        "shared_model_final_sha256": final_hash,
        "shared_model_integrity_passed": integrity_pass,
        "export_count": 1,
        "export_manifest": str(export_manifest_path),
        "dimensions": {
            key: export_manifest[key]
            for key in ("n", "equalities", "inequalities")},
        "lazy_binary": str(lazy_binary),
        "ipopt_binary": str(ipopt_binary),
        "requested_repetitions": args.repetitions,
        "completed_pairs": completed_pairs,
        "pair_summaries": pair_summaries,
        "rows": rows,
        "statistics": {
            "ipopt_solver_seconds_median": ipopt_median,
            "ipopt_solver_seconds_min": min(ipopt_times)
                if ipopt_times else None,
            "ipopt_solver_seconds_max": max(ipopt_times)
                if ipopt_times else None,
            "lazy_solver_seconds_median": lazy_median,
            "lazy_solver_seconds_min": min(lazy_times)
                if lazy_times else None,
            "lazy_solver_seconds_max": max(lazy_times)
                if lazy_times else None,
            "lazy_over_ipopt_median_ratio": time_ratio,
            "objective_absolute_difference_median":
                median(objective_differences),
            "objective_absolute_difference_max":
                max(objective_differences)
                if objective_differences else None,
        },
        "target_seconds": args.target_seconds,
        "target_metric": "median_lazyipdca_solver_elapsed_seconds",
        "target_passed": target_pass,
        "benchmark_passed": benchmark_pass,
        "csv": str(csv_path),
    }
    benchmark_json = benchmark_dir / "benchmark.json"
    write_json_atomic(benchmark_json, benchmark)

    def shown(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    lines = [
        "[R3P23 SAME-QCQP BENCHMARK]",
        f"benchmark={benchmark_id}",
        f"case={case}",
        f"shared_model={model_path}",
        f"shared_model_sha256={fingerprint}",
        f"shared_model_final_sha256={final_hash}",
        "export_count=1",
        f"dimensions={export_manifest['n']}/"
        f"{export_manifest['equalities']}/"
        f"{export_manifest['inequalities']}",
        f"pairs_passed={completed_pairs}/{args.repetitions}",
        f"ipopt_solver_seconds_median={shown(ipopt_median)}",
        f"lazy_solver_seconds_median={shown(lazy_median)}",
        f"lazy_over_ipopt_median_ratio={shown(time_ratio)}",
        "objective_absolute_difference_median="
        f"{shown(median(objective_differences))}",
        f"target_seconds={args.target_seconds}",
        "TARGET=" + ("PASS" if target_pass else "FAIL"),
        "MODEL_INTEGRITY=" + ("PASS" if integrity_pass else "FAIL"),
        "BENCHMARK=" + ("PASS" if benchmark_pass else "FAIL"),
        f"csv={csv_path}",
        f"json={benchmark_json}",
    ]
    summary_path = benchmark_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"summary={summary_path}")
    return 0 if benchmark_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
