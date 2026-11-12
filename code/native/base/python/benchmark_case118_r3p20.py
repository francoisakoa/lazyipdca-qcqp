#!/usr/bin/env python3
"""Repeat the r3p20 case118 runner and aggregate correctness and timings.

This program deliberately invokes no build system.  Every repetition uses the
already-compiled executable selected by --binary or the BIN environment
variable.  Unknown options are forwarded to run_pglib_qcqp_r3p20.py.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parser() -> argparse.ArgumentParser:
    description = (
        "Run PGLib case118 repeatedly with an existing r3p20 binary, then "
        "write benchmark.csv, benchmark.json, and summary.txt. Options not "
        "listed below are passed unchanged to the one-case PGLib runner."
    )
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_base = pathlib.Path(
        os.environ.get("BASE", "/data/projects/IPDCA_QCQP"))
    ap.add_argument("--case", type=pathlib.Path,
                    default=default_base / "pglib-opf" /
                    "pglib_opf_case118_ieee.m")
    ap.add_argument("--binary", type=pathlib.Path,
                    default=(pathlib.Path(os.environ["BIN"])
                             if os.environ.get("BIN") else None),
                    help="already-compiled lazyipdca_qcqp_cli")
    ap.add_argument("--base", type=pathlib.Path, default=default_base)
    ap.add_argument("--results-root", type=pathlib.Path, default=None,
                    help="parent directory for the timestamped benchmark")
    ap.add_argument("--benchmark-id", default=None)
    ap.add_argument("--repetitions", "-n", type=int, default=5)
    ap.add_argument("--target-seconds", type=float, default=10.0,
                    help="strict upper bound for median solver elapsed time")
    ap.add_argument("--eq-tol", type=float, default=1.0e-6)
    ap.add_argument("--ineq-tol", type=float, default=1.0e-6)
    ap.add_argument("--dual-tol", type=float, default=1.0e-6)
    ap.add_argument("--stop-on-failure", action="store_true",
                    help="stop after the first failed solve, KKT, or "
                         "block-exact structural check")
    return ap


def field(solver: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in solver:
            return solver[name]
    return None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args, forwarded = parser().parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.repetitions < 1:
        parser().error("--repetitions must be at least 1")
    if args.target_seconds <= 0.0:
        parser().error("--target-seconds must be positive")
    for name in ("eq_tol", "ineq_tol", "dual_tol"):
        if getattr(args, name) <= 0.0:
            parser().error(f"--{name.replace('_', '-')} must be positive")
    forbidden = {"--run-id", "--results-root", "--binary", "--base"}
    clash = next((option for option in forwarded if option in forbidden), None)
    if clash:
        parser().error(
            f"{clash} is managed by the benchmark; pass its benchmark-level "
            "form before the forwarded runner options")

    root = pathlib.Path(__file__).resolve().parents[1]
    runner = root / "python" / "run_pglib_qcqp_r3p20.py"
    binary = (args.binary or root / "build" /
              "lazyipdca_qcqp_cli").resolve()
    case = args.case.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser().error(f"compiled executable is missing or not executable: {binary}")
    if not case.is_file():
        parser().error(f"case file is missing: {case}")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    benchmark_id = (args.benchmark_id or
                    f"R3P20_CASE118_BENCH_{stamp}")
    results_parent = (args.results_root or
                      args.base / "results").resolve()
    benchmark_dir = results_parent / benchmark_id
    runs_root = benchmark_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for index in range(1, args.repetitions + 1):
        run_id = f"run_{index:03d}"
        command = [
            sys.executable, "-u", str(runner), str(case),
            "--base", str(args.base.resolve()),
            "--binary", str(binary),
            "--results-root", str(runs_root),
            "--run-id", run_id,
            *forwarded,
        ]
        print(f"\n[BENCH {index}/{args.repetitions}] {case.stem}", flush=True)
        wall_start = time.perf_counter()
        completed = subprocess.run(command, check=False)
        wall_seconds = time.perf_counter() - wall_start
        summary_path = runs_root / run_id / "summary.json"
        summary: dict[str, Any] = {}
        parse_error = ""
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
        solver = summary.get("solver", {})
        if not isinstance(solver, dict):
            solver = {}

        status = str(solver.get("status", "missing_summary"))
        solver_rc = solver.get("return_code", completed.returncode)
        equality = finite_float(field(
            solver, "equality_violation", "equality_inf"))
        inequality = finite_float(field(
            solver, "inequality_violation", "inequality_inf"))
        dual = finite_float(field(
            solver, "dual_infeasibility_inf", "dual_infeasibility"))
        solver_seconds = finite_float(field(
            solver, "elapsed_seconds", "elapsed"))
        objective = finite_float(solver.get("objective"))
        block_exact_attempts = finite_float(
            solver.get("lazy_dca_condensed_block_exact_attempts"))
        block_exact_acceptances = finite_float(
            solver.get("lazy_dca_condensed_block_exact_acceptances"))
        geometry_normal_solves = finite_float(
            solver.get("lazy_dca_condensed_geometry_normal_solves"))
        solved = completed.returncode == 0 and status == "solved"
        kkt_pass = (
            solved and equality is not None and inequality is not None and
            dual is not None and equality <= args.eq_tol and
            inequality <= args.ineq_tol and dual <= args.dual_tol
        )
        block_exact_pass = (
            solved and block_exact_attempts is not None and
            block_exact_attempts > 0.0 and
            block_exact_acceptances == block_exact_attempts and
            geometry_normal_solves == 0.0)
        time_pass = (solver_seconds is not None and
                     solver_seconds < args.target_seconds)
        row = {
            "repetition": index,
            "solver_status": status,
            "process_return_code": completed.returncode,
            "solver_return_code": solver_rc,
            "solved": solved,
            "kkt_pass": kkt_pass,
            "block_exact_pass": block_exact_pass,
            "time_pass_individual": time_pass,
            "objective": objective,
            "equality_violation": equality,
            "inequality_violation": inequality,
            "dual_infeasibility_inf": dual,
            "solver_seconds": solver_seconds,
            "wall_seconds": wall_seconds,
            "summary_json": str(summary_path),
            "parse_error": parse_error,
        }
        diagnostic_fields = (
            "outer_iterations",
            "inner_iterations",
            "line_search_backtracks",
            "lazy_dca_condensed_retraction_attempts",
            "lazy_dca_condensed_factorizations",
            "lazy_dca_condensed_normal_solves",
            "lazy_dca_condensed_geometry_normal_solves",
            "lazy_dca_condensed_normal_refinement_steps",
            "lazy_dca_condensed_geometry_normal_refinement_steps",
            "lazy_dca_condensed_retraction_normal_refinement_steps",
            "lazy_dca_condensed_relaxed_normal_solves",
            "lazy_dca_condensed_strict_normal_solves",
            "lazy_dca_condensed_relaxed_accepts_above_strict_tolerance",
            "lazy_dca_condensed_strict_retry_retractions",
            "lazy_dca_condensed_alpha_predictor_uses",
            "lazy_dca_condensed_alpha_predictor_full_step_resets",
            "lazy_dca_condensed_alpha_predictor_leading_retractions_skipped",
            "lazy_dca_condensed_alpha_predictor_historical_retries",
            "lazy_dca_condensed_flow_lift_schur_detected_variables",
            "lazy_dca_condensed_flow_lift_schur_factor_dimension",
            "lazy_dca_condensed_flow_lift_schur_factorizations",
            "lazy_dca_condensed_flow_lift_schur_fallbacks",
            "lazy_dca_condensed_post_retraction_modeled_offenders",
            "lazy_dca_condensed_post_retraction_omitted_offenders",
            "lazy_dca_condensed_post_retraction_working_set_expansions",
            "lazy_dca_condensed_post_retraction_expansion_fallbacks",
            "lazy_dca_condensed_block_exact_attempts",
            "lazy_dca_condensed_block_exact_acceptances",
            "lazy_dca_condensed_block_exact_blocks_max",
            "lazy_dca_condensed_block_exact_active_dimension_max",
            "lazy_dca_condensed_block_exact_max_block_dimension",
            "lazy_dca_condensed_projected_multi_ritz_batches",
            "lazy_dca_condensed_projected_multi_ritz_fallbacks",
            "lazy_dca_condensed_tangent_inertia_attempts",
            "lazy_dca_condensed_tangent_inertia_acceptances",
            "lazy_dca_condensed_tangent_inertia_shift_updates",
            "lazy_dca_condensed_tangent_inertia_shift_max",
            "lazyqp_affine_elimination_attempts",
            "lazyqp_affine_elimination_acceptances",
            "lazyqp_affine_elimination_fallbacks",
            "lazyqp_affine_eliminated_variables",
            "lazyqp_affine_eliminated_equalities",
            "lazyqp_affine_reduced_variables",
            "lazyqp_affine_reduced_equalities",
            "lazyqp_max_kkt_dimension",
            "lazyqp_max_full_kkt_dimension",
        )
        for diagnostic in diagnostic_fields:
            row[diagnostic] = solver.get(diagnostic)
        rows.append(row)
        print(
            "[BENCH RESULT] "
            f"status={status} rc={completed.returncode} "
            f"eq={equality} ineq={inequality} dual={dual} "
            f"solver_s={solver_seconds} wall_s={wall_seconds:.6f} "
            f"kkt={'PASS' if kkt_pass else 'FAIL'} "
            f"block={'PASS' if block_exact_pass else 'FAIL'}",
            flush=True,
        )
        if args.stop_on_failure and not (kkt_pass and block_exact_pass):
            break

    solver_times = [value for value in
                    (finite_float(row["solver_seconds"]) for row in rows)
                    if value is not None]
    wall_times = [float(row["wall_seconds"]) for row in rows]
    median_solver = median_or_none(solver_times)
    diagnostic_medians = {}
    for diagnostic in diagnostic_fields:
        values = [value for value in
                  (finite_float(row.get(diagnostic)) for row in rows)
                  if value is not None]
        diagnostic_medians[diagnostic] = median_or_none(values)
    all_completed = len(rows) == args.repetitions
    solved_count = sum(bool(row["solved"]) for row in rows)
    kkt_count = sum(bool(row["kkt_pass"]) for row in rows)
    block_exact_count = sum(bool(row["block_exact_pass"]) for row in rows)
    # The timing gate cannot hide a failed solve: every requested repetition
    # must solve and pass the explicit KKT checks before median time can pass.
    target_pass = (
        all_completed and solved_count == args.repetitions and
        kkt_count == args.repetitions and
        block_exact_count == args.repetitions and
        median_solver is not None and median_solver < args.target_seconds
    )
    aggregate = {
        "benchmark_id": benchmark_id,
        "case": str(case),
        "binary": str(binary),
        "requested_repetitions": args.repetitions,
        "completed_repetitions": len(rows),
        "solved": solved_count,
        "kkt_passed": kkt_count,
        "block_exact_passed": block_exact_count,
        "tolerances": {
            "equality": args.eq_tol,
            "inequality": args.ineq_tol,
            "dual": args.dual_tol,
        },
        "target_seconds": args.target_seconds,
        "target_metric": "median_solver_elapsed_seconds",
        "target_pass": target_pass,
        "solver_seconds": {
            "median": median_solver,
            "minimum": min(solver_times) if solver_times else None,
            "maximum": max(solver_times) if solver_times else None,
        },
        "wall_seconds": {
            "median": median_or_none(wall_times),
            "minimum": min(wall_times) if wall_times else None,
            "maximum": max(wall_times) if wall_times else None,
        },
        "forwarded_arguments": forwarded,
        "diagnostic_medians": diagnostic_medians,
        "runs": rows,
    }

    csv_path = benchmark_dir / "benchmark.csv"
    json_path = benchmark_dir / "benchmark.json"
    text_path = benchmark_dir / "summary.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = "PASS" if target_pass else "FAIL"
    text = (
        f"benchmark={benchmark_id}\n"
        f"case={case}\n"
        f"binary={binary}\n"
        f"solved={solved_count}/{args.repetitions}\n"
        f"kkt_passed={kkt_count}/{args.repetitions}\n"
        f"block_exact_passed={block_exact_count}/{args.repetitions}\n"
        f"eq_tol={args.eq_tol:.17g} ineq_tol={args.ineq_tol:.17g} "
        f"dual_tol={args.dual_tol:.17g}\n"
        f"solver_seconds_median={median_solver}\n"
        f"solver_seconds_min={aggregate['solver_seconds']['minimum']}\n"
        f"solver_seconds_max={aggregate['solver_seconds']['maximum']}\n"
        f"wall_seconds_median={aggregate['wall_seconds']['median']}\n"
        f"retraction_attempts_median="
        f"{diagnostic_medians['lazy_dca_condensed_retraction_attempts']}\n"
        f"condensed_factorizations_median="
        f"{diagnostic_medians['lazy_dca_condensed_factorizations']}\n"
        f"normal_solves_median="
        f"{diagnostic_medians['lazy_dca_condensed_normal_solves']}\n"
        f"geometry_normal_solves_median="
        f"{diagnostic_medians['lazy_dca_condensed_geometry_normal_solves']}\n"
        f"normal_refinement_steps_median="
        f"{diagnostic_medians['lazy_dca_condensed_normal_refinement_steps']}\n"
        f"relaxed_normal_solves_median="
        f"{diagnostic_medians['lazy_dca_condensed_relaxed_normal_solves']}\n"
        f"leading_retractions_skipped_median="
        f"{diagnostic_medians['lazy_dca_condensed_alpha_predictor_leading_retractions_skipped']}\n"
        f"flow_schur_lifts_median="
        f"{diagnostic_medians['lazy_dca_condensed_flow_lift_schur_detected_variables']}\n"
        f"flow_schur_dimension_median="
        f"{diagnostic_medians['lazy_dca_condensed_flow_lift_schur_factor_dimension']}\n"
        f"flow_schur_fallbacks_median="
        f"{diagnostic_medians['lazy_dca_condensed_flow_lift_schur_fallbacks']}\n"
        f"post_retraction_ws_expansions_median="
        f"{diagnostic_medians['lazy_dca_condensed_post_retraction_working_set_expansions']}\n"
        f"post_retraction_ws_fallbacks_median="
        f"{diagnostic_medians['lazy_dca_condensed_post_retraction_expansion_fallbacks']}\n"
        f"block_exact_attempts_median="
        f"{diagnostic_medians['lazy_dca_condensed_block_exact_attempts']}\n"
        f"block_exact_acceptances_median="
        f"{diagnostic_medians['lazy_dca_condensed_block_exact_acceptances']}\n"
        f"block_exact_blocks_max_median="
        f"{diagnostic_medians['lazy_dca_condensed_block_exact_blocks_max']}\n"
        f"block_exact_active_dimension_max_median="
        f"{diagnostic_medians['lazy_dca_condensed_block_exact_active_dimension_max']}\n"
        f"block_exact_max_block_dimension_median="
        f"{diagnostic_medians['lazy_dca_condensed_block_exact_max_block_dimension']}\n"
        f"multi_ritz_batches_median="
        f"{diagnostic_medians['lazy_dca_condensed_projected_multi_ritz_batches']}\n"
        f"multi_ritz_fallbacks_median="
        f"{diagnostic_medians['lazy_dca_condensed_projected_multi_ritz_fallbacks']}\n"
        f"inertia_attempts_median="
        f"{diagnostic_medians['lazy_dca_condensed_tangent_inertia_attempts']}\n"
        f"inertia_acceptances_median="
        f"{diagnostic_medians['lazy_dca_condensed_tangent_inertia_acceptances']}\n"
        f"affine_elimination_try_ok_fallback_median="
        f"{diagnostic_medians['lazyqp_affine_elimination_attempts']}/"
        f"{diagnostic_medians['lazyqp_affine_elimination_acceptances']}/"
        f"{diagnostic_medians['lazyqp_affine_elimination_fallbacks']}\n"
        f"affine_reduced_variables_equalities_median="
        f"{diagnostic_medians['lazyqp_affine_reduced_variables']}/"
        f"{diagnostic_medians['lazyqp_affine_reduced_equalities']}\n"
        f"lazyqp_kkt_reduced_full_median="
        f"{diagnostic_medians['lazyqp_max_kkt_dimension']}/"
        f"{diagnostic_medians['lazyqp_max_full_kkt_dimension']}\n"
        f"target_seconds={args.target_seconds:.17g}\n"
        f"target_metric=median_solver_elapsed_seconds\n"
        f"TARGET={verdict}\n"
        f"csv={csv_path}\njson={json_path}\n"
    )
    text_path.write_text(text, encoding="utf-8")
    print("\n[BENCHMARK SUMMARY]")
    print(text, end="")
    print(f"summary={text_path}")
    return 0 if target_pass else 1


if __name__ == "__main__":
    sys.exit(main())
