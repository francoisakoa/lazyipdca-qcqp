#!/usr/bin/env python3
"""Compare strict and r3p24 LazyIPDCA profiles on one immutable QCQP.

The input is an already-exported ``.qcq`` file.  This driver never exports or
copies the model: both profiles receive the same absolute path, and a SHA-256
check is performed before and after every solver process.
"""

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
import time
from typing import Any, Sequence

from export_pglib_qcqp_r3p23 import sha256_file, write_json_atomic
from run_pglib_qcqp_r3p23 import (
    finite_float,
    kkt_pass,
    lazy_command,
    load_result,
    qcq_dimensions,
)


BENCHMARK_SCHEMA = "lazyipdca-r3p24-same-qcqp-ab-benchmark-v1"
PAIR_SCHEMA = "lazyipdca-r3p24-same-qcqp-ab-pair-v1"
IPOPT_REFERENCE_SCHEMA = "lazyipdca-ipopt-same-qcqp-run-v1"
KKT_RESIDUAL_NAMES = (
    "equality_violation", "inequality_violation", "bound_violation",
    "dual_infeasibility_inf", "complementarity_inf",
)

DEFAULT_BASELINE_ARGS = (
    "--no-lazyqp-inexact-forcing "
    "--no-lazyqp-zero-dual-continuation "
    "--no-dca-condensed-outer-extrapolation "
    "--no-terminal-kkt"
)
DEFAULT_CANDIDATE_ARGS = (
    "--no-lazyqp-inexact-forcing "
    "--lazyqp-zero-dual-continuation "
    "--lazyqp-zero-dual-mu-cap 1e-4 "
    "--no-dca-condensed-outer-extrapolation "
    "--no-terminal-kkt"
)


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def option_enabled(
    arguments: Sequence[str],
    positive: str,
    negative: str,
) -> bool:
    enabled = False
    for argument in arguments:
        if argument == positive:
            enabled = True
        elif argument == negative:
            enabled = False
    return enabled


def require_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def objective_comparison(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    baseline_value = finite_float(baseline.get("objective"))
    candidate_value = finite_float(candidate.get("objective"))
    if baseline_value is None or candidate_value is None:
        return {
            "absolute_difference": None,
            "relative_difference": None,
            "match": False,
        }
    absolute = abs(candidate_value - baseline_value)
    scale = max(1.0, abs(baseline_value), abs(candidate_value))
    relative = absolute / scale
    return {
        "absolute_difference": absolute,
        "relative_difference": relative,
        "match": absolute <= absolute_tolerance + relative_tolerance * scale,
    }


def nonnegative_kkt_residuals(result: dict[str, Any]) -> bool:
    raw_values = [result.get(name) for name in KKT_RESIDUAL_NAMES]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in raw_values):
        return False
    values = [finite_float(value) for value in raw_values]
    return all(value is not None and value >= 0.0 for value in values)


def validate_ipopt_reference_summary(
    summary_path: Path,
    model: Path,
    expected_hash: str,
    *,
    equality_tolerance: float,
    inequality_tolerance: float,
    bound_tolerance: float,
    dual_tolerance: float,
    complementarity_tolerance: float,
) -> dict[str, Any]:
    """Validate a prior Ipopt-first r3p23 same-QCQP pair summary."""
    summary_path = summary_path.expanduser().resolve()
    model = model.expanduser().resolve()
    errors: list[str] = []
    payload: dict[str, Any] = {}
    try:
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
        else:
            errors.append("summary JSON is not an object")
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read summary: {exc}")

    if payload.get("schema") != IPOPT_REFERENCE_SCHEMA:
        errors.append("unexpected summary schema")

    model_record = payload.get("model")
    if not isinstance(model_record, dict):
        errors.append("missing model record")
        model_record = {}
    reference_model_path = model_record.get("path")
    try:
        same_path = (isinstance(reference_model_path, str) and
                     Path(reference_model_path).expanduser().resolve() == model)
    except OSError:
        same_path = False
    if not same_path:
        errors.append("reference model path does not match benchmark model")
    if model_record.get("sha256") != expected_hash:
        errors.append("reference model SHA-256 does not match benchmark model")

    integrity_hashes = payload.get("model_integrity_hashes")
    if (payload.get("model_integrity_passed") is not True or
            not isinstance(integrity_hashes, list) or
            len(integrity_hashes) < 3 or
            any(value != expected_hash for value in integrity_hashes)):
        errors.append("reference model-integrity hashes are incomplete or differ")
    if payload.get("execution_order") != ["ipopt", "lazyipdca"]:
        errors.append("reference was not produced with Ipopt first")

    ipopt = payload.get("ipopt")
    if not isinstance(ipopt, dict):
        errors.append("missing Ipopt record")
        ipopt = {}
    return_code = ipopt.get("return_code")
    if (not isinstance(return_code, int) or isinstance(return_code, bool) or
            return_code != 0):
        errors.append("Ipopt return code is not zero")
    if ipopt.get("certificate_pass") is not True:
        errors.append("Ipopt summary certificate did not pass")
    if ipopt.get("parse_error") != "":
        errors.append("Ipopt result has a parse error")

    result = ipopt.get("result")
    if not isinstance(result, dict):
        errors.append("missing Ipopt result")
        result = {}
    recomputed_kkt_pass = nonnegative_kkt_residuals(result) and kkt_pass(
        result,
        equality_tolerance=equality_tolerance,
        inequality_tolerance=inequality_tolerance,
        bound_tolerance=bound_tolerance,
        dual_tolerance=dual_tolerance,
        complementarity_tolerance=complementarity_tolerance,
    )
    if not recomputed_kkt_pass:
        errors.append("Ipopt result fails the current common KKT tolerances")

    # CasADi references carry these fields; the native TNLP summary currently
    # does not. Whenever provenance is present, it must agree with the outer
    # r3p23 model record rather than being silently ignored.
    for name in ("model_path", "consumed_model"):
        value = result.get(name)
        if value is not None:
            try:
                matches = isinstance(value, str) and Path(value).resolve() == model
            except OSError:
                matches = False
            if not matches:
                errors.append(f"Ipopt result {name} does not match benchmark model")
    for name in ("model_sha256", "input_sha256", "consumed_sha256"):
        value = result.get(name)
        if value is not None and value != expected_hash:
            errors.append(f"Ipopt result {name} does not match benchmark model")

    return {
        "path": str(summary_path),
        "summary_sha256": (
            sha256_file(summary_path) if summary_path.is_file() else None),
        "schema": payload.get("schema"),
        "model_path": reference_model_path,
        "model_sha256": model_record.get("sha256"),
        "execution_order": payload.get("execution_order"),
        "model_integrity_hashes": integrity_hashes,
        "model_integrity_passed":
            payload.get("model_integrity_passed") is True,
        "return_code": return_code,
        "reported_certificate_pass": ipopt.get("certificate_pass"),
        "recomputed_kkt_pass": recomputed_kkt_pass,
        "validation_passed": not errors,
        "errors": errors,
    }


def run_command(
    command: Sequence[str],
    log_path: Path,
    *,
    live_output: bool,
) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if live_output:
                print(line, end="", flush=True)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def run_profile_pair(
    model: Path,
    binary: Path,
    output_dir: Path,
    *,
    order: str,
    baseline_args: Sequence[str],
    candidate_args: Sequence[str],
    common_extra_args: Sequence[str] = (),
    linear_backend: str = "ldl",
    phase1_threads: int = 8,
    max_outer: int = 1000,
    max_inner: int = 1000,
    time_limit: float = 1800.0,
    equality_tolerance: float = 1.0e-6,
    inequality_tolerance: float = 1.0e-6,
    bound_tolerance: float = 1.0e-6,
    dual_tolerance: float = 1.0e-6,
    complementarity_tolerance: float = 1.0e-6,
    objective_absolute_tolerance: float = 1.0e-3,
    objective_relative_tolerance: float = 1.0e-8,
    live_output: bool = False,
) -> dict[str, Any]:
    """Run one strict/candidate pair without creating another QCQP file."""
    model = model.expanduser().resolve()
    binary = binary.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"exported QCQP is missing: {model}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(
            f"LazyIPDCA executable is missing or not executable: {binary}")
    if order not in {"baseline-first", "candidate-first"}:
        raise ValueError(f"unsupported execution order: {order}")

    output_dir.mkdir(parents=True, exist_ok=False)
    expected_hash = sha256_file(model)
    dimensions = qcq_dimensions(model)
    labels = (["baseline", "candidate"] if order == "baseline-first"
              else ["candidate", "baseline"])
    profile_args = {
        "baseline": [*common_extra_args, *baseline_args],
        "candidate": [*common_extra_args, *candidate_args],
    }
    result_paths = {
        label: output_dir / f"{label}.result.json"
        for label in ("baseline", "candidate")
    }
    log_paths = {
        label: output_dir / f"{label}.solve.log"
        for label in ("baseline", "candidate")
    }
    commands = {
        label: lazy_command(
            binary,
            model,
            result_paths[label],
            linear_backend=linear_backend,
            phase1_threads=phase1_threads,
            max_outer=max_outer,
            max_inner=max_inner,
            time_limit=time_limit,
            live_output=live_output,
            feasibility_tolerance=min(
                equality_tolerance, inequality_tolerance, bound_tolerance),
            dual_tolerance=dual_tolerance,
            complementarity_tolerance=complementarity_tolerance,
            extra_args=profile_args[label],
        )
        for label in ("baseline", "candidate")
    }

    records: dict[str, Any] = {}
    integrity_checks: list[dict[str, str]] = []
    for label in labels:
        before = sha256_file(model)
        integrity_checks.append({"point": f"before_{label}", "sha256": before})
        if before != expected_hash:
            raise RuntimeError(
                f"QCQP hash changed before {label}: "
                f"{expected_hash} -> {before}")
        result_paths[label].unlink(missing_ok=True)
        print(f"[{label.upper()}] same QCQP sha256={expected_hash}", flush=True)
        return_code, wall_seconds = run_command(
            commands[label], log_paths[label], live_output=live_output)
        after = sha256_file(model)
        integrity_checks.append({"point": f"after_{label}", "sha256": after})
        if after != expected_hash:
            raise RuntimeError(
                f"{label} modified the shared QCQP: "
                f"{expected_hash} -> {after}")

        result, parse_error = load_result(result_paths[label])
        certificate = nonnegative_kkt_residuals(result) and kkt_pass(
            result,
            equality_tolerance=equality_tolerance,
            inequality_tolerance=inequality_tolerance,
            bound_tolerance=bound_tolerance,
            dual_tolerance=dual_tolerance,
            complementarity_tolerance=complementarity_tolerance,
        )
        records[label] = {
            "profile_arguments": profile_args[label],
            "command": commands[label],
            "return_code": return_code,
            "wall_seconds": wall_seconds,
            "result_json": str(result_paths[label]),
            "log": str(log_paths[label]),
            "parse_error": parse_error,
            "certificate_passed": certificate,
            "result": result,
        }

    comparison = objective_comparison(
        records["baseline"]["result"],
        records["candidate"]["result"],
        absolute_tolerance=objective_absolute_tolerance,
        relative_tolerance=objective_relative_tolerance,
    )
    pair_passed = (
        all(records[label]["return_code"] == 0
            for label in ("baseline", "candidate")) and
        all(records[label]["certificate_passed"]
            for label in ("baseline", "candidate")) and
        comparison["match"] and
        all(item["sha256"] == expected_hash for item in integrity_checks)
    )
    payload = {
        "schema": PAIR_SCHEMA,
        "model": {
            "path": str(model),
            "sha256": expected_hash,
            "bytes": model.stat().st_size,
            **dimensions,
        },
        "model_integrity_checks": integrity_checks,
        "model_integrity_passed": all(
            item["sha256"] == expected_hash for item in integrity_checks),
        "binary": str(binary),
        "execution_order": labels,
        "tolerances": {
            "equality": equality_tolerance,
            "inequality": inequality_tolerance,
            "bound": bound_tolerance,
            "dual": dual_tolerance,
            "complementarity": complementarity_tolerance,
            "objective_absolute": objective_absolute_tolerance,
            "objective_relative": objective_relative_tolerance,
        },
        "baseline": records["baseline"],
        "candidate": records["candidate"],
        "comparison": comparison,
        "pair_passed": pair_passed,
    }
    write_json_atomic(output_dir / "summary.json", payload)
    return payload


def metric(result: dict[str, Any], name: str) -> float | None:
    return finite_float(result.get(name))


def valid_elapsed(value: Any) -> bool:
    converted = finite_float(value)
    return converted is not None and converted >= 0.0


def reported_numeric_factorizations(result: dict[str, Any]) -> float | None:
    """Return a disjoint count of reported numeric factorizations.

    The main KKT counter already contains SOC factorizations, and the
    restoration linear-solve counter already contains minimax solves.
    Matrix-free restoration solves are removed because they perform no numeric
    factorization. Flow-Schur, reduced/full KKT and bound-face counters are
    aliases of sites already represented here. The terminal solver did not
    expose an exact factorization counter in r3p23, so a run that actually
    enters it is reported as unavailable instead of inventing a number.
    """
    def valid_count(value: float | None) -> bool:
        return (
            value is not None and value >= 0.0 and
            float(value).is_integer())

    terminal_attempts = metric(result, "terminal_kkt_attempts")
    terminal_iterations = metric(result, "terminal_kkt_iterations")
    if not valid_count(terminal_attempts) or not valid_count(
            terminal_iterations):
        return None
    if ((terminal_attempts is not None and terminal_attempts > 0.0) or
            (terminal_iterations is not None and terminal_iterations > 0.0)):
        return None

    direct_names = (
        "kkt_factorizations",
        "lazy_dca_condensed_factorizations",
        "multiplier_row_space_factorizations",
        "restoration_harmonic_factorizations",
        "lazy_dca_condensed_tangent_inertia_attempts",
    )
    values = [metric(result, name) for name in direct_names]
    restoration_solves = metric(result, "restoration_linear_solves")
    matrix_free_solves = metric(result, "restoration_matrix_free_solves")
    if (any(not valid_count(value) for value in values) or
            not valid_count(restoration_solves) or
            not valid_count(matrix_free_solves) or
            matrix_free_solves > restoration_solves):
        return None
    values.append(restoration_solves - matrix_free_solves)
    return sum(value for value in values if value is not None)


def row_from_pair(index: int, order: str, pair: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "repetition": index,
        "execution_order": order,
        "model_sha256": pair["model"]["sha256"],
        "model_integrity_passed": pair["model_integrity_passed"],
        "pair_passed": pair["pair_passed"],
        "objective_absolute_difference":
            pair["comparison"]["absolute_difference"],
        "objective_relative_difference":
            pair["comparison"]["relative_difference"],
        "objective_match": pair["comparison"]["match"],
    }
    aliases = {
        "objective": "objective",
        "equality": "equality_violation",
        "inequality": "inequality_violation",
        "bound": "bound_violation",
        "dual": "dual_infeasibility_inf",
        "complementarity": "complementarity_inf",
        "outer_iterations": "outer_iterations",
        "inner_iterations": "inner_iterations",
        "kkt_factorizations": "kkt_factorizations",
        "solver_seconds": "elapsed_seconds",
        "zero_dual_attempts": "lazyqp_zero_dual_continuation_attempts",
        "zero_dual_mapped_attempts":
            "lazyqp_zero_dual_continuation_mapped_attempts",
        "zero_dual_direct_attempts":
            "lazyqp_zero_dual_continuation_direct_attempts",
        "zero_dual_acceptances":
            "lazyqp_zero_dual_continuation_acceptances",
        "zero_dual_fallbacks": "lazyqp_zero_dual_continuation_fallbacks",
        "zero_dual_repairs": "lazyqp_zero_dual_continuation_repairs",
    }
    for label in ("baseline", "candidate"):
        record = pair[label]
        result = record["result"]
        row[f"{label}_status"] = result.get("status")
        row[f"{label}_return_code"] = record["return_code"]
        row[f"{label}_kkt_pass"] = record["certificate_passed"]
        row[f"{label}_wall_seconds"] = record["wall_seconds"]
        row[f"{label}_reported_numeric_factorizations"] = (
            reported_numeric_factorizations(result))
        for destination, source in aliases.items():
            row[f"{label}_{destination}"] = metric(result, source)
    return row


def profile_medians(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    names = (
        "objective", "solver_seconds", "wall_seconds", "outer_iterations",
        "inner_iterations", "kkt_factorizations", "equality", "inequality",
        "reported_numeric_factorizations", "bound", "dual",
        "complementarity", "zero_dual_attempts",
        "zero_dual_mapped_attempts", "zero_dual_direct_attempts",
        "zero_dual_acceptances", "zero_dual_fallbacks",
        "zero_dual_repairs",
    )
    output: dict[str, Any] = {}
    for name in names:
        values = []
        for row in rows:
            value = finite_float(row.get(f"{label}_{name}"))
            if value is None:
                continue
            if name in {"solver_seconds", "wall_seconds"} and value < 0.0:
                continue
            values.append(value)
        output[f"{name}_median"] = median(values)
        if name in {"solver_seconds", "wall_seconds"}:
            output[f"{name}_min"] = min(values) if values else None
            output[f"{name}_max"] = max(values) if values else None
    return output


def safe_ratio(numerator: Any, denominator: Any) -> float | None:
    num = finite_float(numerator)
    den = finite_float(denominator)
    if num is None or den is None or den == 0.0:
        return None
    return num / den


def benchmark(
    *,
    model: Path,
    binary: Path,
    ipopt_reference_summary: Path,
    output_dir: Path,
    repetitions: int,
    baseline_args: Sequence[str],
    candidate_args: Sequence[str],
    common_extra_args: Sequence[str] = (),
    historical_target_seconds: float = 10.0,
    competitiveness_target_seconds: float = 0.5,
    factorization_budget: int = 45,
    stop_on_failure: bool = False,
    **pair_options: Any,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    require_positive("historical target", historical_target_seconds)
    require_positive("competitiveness target", competitiveness_target_seconds)
    require_positive("factorization budget", factorization_budget)

    model = model.expanduser().resolve()
    binary = binary.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"exported QCQP is missing: {model}")
    expected_hash = sha256_file(model)
    dimensions = qcq_dimensions(model)
    ipopt_reference = validate_ipopt_reference_summary(
        ipopt_reference_summary,
        model,
        expected_hash,
        equality_tolerance=float(
            pair_options.get("equality_tolerance", 1.0e-6)),
        inequality_tolerance=float(
            pair_options.get("inequality_tolerance", 1.0e-6)),
        bound_tolerance=float(pair_options.get("bound_tolerance", 1.0e-6)),
        dual_tolerance=float(pair_options.get("dual_tolerance", 1.0e-6)),
        complementarity_tolerance=float(
            pair_options.get("complementarity_tolerance", 1.0e-6)),
    )
    if not ipopt_reference["validation_passed"]:
        raise ValueError(
            "invalid Ipopt-first reference summary: " +
            "; ".join(ipopt_reference["errors"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir()

    rows: list[dict[str, Any]] = []
    summaries: list[str] = []
    for index in range(1, repetitions + 1):
        order = "baseline-first" if index % 2 else "candidate-first"
        run_dir = runs_dir / f"run_{index:03d}"
        print(f"\n[A/B {index}/{repetitions}] order={order}", flush=True)
        pair = run_profile_pair(
            model,
            binary,
            run_dir,
            order=order,
            baseline_args=baseline_args,
            candidate_args=candidate_args,
            common_extra_args=common_extra_args,
            **pair_options,
        )
        row = row_from_pair(index, order, pair)
        row["summary_json"] = str(run_dir / "summary.json")
        rows.append(row)
        summaries.append(str(run_dir / "summary.json"))
        print(
            "[A/B RESULT] "
            f"pass={row['pair_passed']} "
            f"baseline_s={row['baseline_solver_seconds']} "
            f"candidate_s={row['candidate_solver_seconds']} "
            f"baseline_outer/inner/fact="
            f"{row['baseline_outer_iterations']}/"
            f"{row['baseline_inner_iterations']}/"
            f"{row['baseline_kkt_factorizations']} "
            f"candidate_outer/inner/fact="
            f"{row['candidate_outer_iterations']}/"
            f"{row['candidate_inner_iterations']}/"
            f"{row['candidate_kkt_factorizations']}",
            flush=True,
        )
        if stop_on_failure and not pair["pair_passed"]:
            break

    final_hash = sha256_file(model)
    integrity_passed = (
        final_hash == expected_hash and
        all(row["model_sha256"] == expected_hash and
            row["model_integrity_passed"] for row in rows)
    )
    baseline_stats = profile_medians(rows, "baseline")
    candidate_stats = profile_medians(rows, "candidate")
    candidate_time = candidate_stats["solver_seconds_median"]
    baseline_time_available_runs = sum(
        valid_elapsed(row.get("baseline_solver_seconds"))
        for row in rows)
    candidate_time_available_runs = sum(
        valid_elapsed(row.get("candidate_solver_seconds"))
        for row in rows)
    timing_available = (
        len(rows) == repetitions and
        baseline_time_available_runs == repetitions and
        candidate_time_available_runs == repetitions)
    historical_passed = (
        timing_available and candidate_time is not None and
        candidate_time < historical_target_seconds)
    competitiveness_passed = (
        timing_available and candidate_time is not None and
        candidate_time < competitiveness_target_seconds)
    candidate_factorizations = candidate_stats[
        "reported_numeric_factorizations_median"]
    factorization_available_runs = sum(
        finite_float(row.get("candidate_reported_numeric_factorizations"))
        is not None for row in rows)
    factorization_budget_available = (
        len(rows) == repetitions and
        factorization_available_runs == repetitions)
    factorization_budget_passed = (
        factorization_budget_available and
        candidate_factorizations is not None and
        candidate_factorizations <= factorization_budget)
    pair_count = sum(bool(row["pair_passed"]) for row in rows)
    zero_dual_required = option_enabled(
        [*common_extra_args, *candidate_args],
        "--lazyqp-zero-dual-continuation",
        "--no-lazyqp-zero-dual-continuation",
    )
    zero_dual_engaged_runs = 0
    for row in rows:
        candidate_values = [
            finite_float(row.get(f"candidate_{name}"))
            for name in (
                "zero_dual_attempts", "zero_dual_mapped_attempts",
                "zero_dual_direct_attempts", "zero_dual_acceptances",
                "zero_dual_fallbacks", "zero_dual_repairs",
            )
        ]
        baseline_attempts = finite_float(
            row.get("baseline_zero_dual_attempts"))
        all_counters = [*candidate_values, baseline_attempts]
        if any(value is None or value < 0.0 or not value.is_integer()
               for value in all_counters):
            continue
        attempts, mapped, direct, acceptances, fallbacks, repairs = (
            value for value in candidate_values if value is not None)
        invariant = (
            attempts > 0.0 and acceptances > 0.0 and
            attempts == mapped + direct and
            attempts == acceptances + fallbacks and
            repairs >= attempts and
            baseline_attempts == 0.0)
        if invariant:
            zero_dual_engaged_runs += 1
    zero_dual_engagement_passed = (
        not zero_dual_required or
        (len(rows) == repetitions and
         zero_dual_engaged_runs == repetitions))
    protocol_passed = (
        len(rows) == repetitions and
        pair_count == repetitions and
        integrity_passed and timing_available and
        zero_dual_engagement_passed and
        ipopt_reference["validation_passed"])
    # The historical acceptance gate remains 10 s.  The 0.5 s competitive
    # objective is deliberately reported independently and is not hidden in
    # the protocol result.
    benchmark_passed = protocol_passed and historical_passed

    comparison = {
        "solver_speedup_baseline_over_candidate": safe_ratio(
            baseline_stats["solver_seconds_median"], candidate_time),
        "outer_ratio_candidate_over_baseline": safe_ratio(
            candidate_stats["outer_iterations_median"],
            baseline_stats["outer_iterations_median"]),
        "inner_ratio_candidate_over_baseline": safe_ratio(
            candidate_stats["inner_iterations_median"],
            baseline_stats["inner_iterations_median"]),
        "factorization_ratio_candidate_over_baseline": safe_ratio(
            candidate_stats["kkt_factorizations_median"],
            baseline_stats["kkt_factorizations_median"]),
        "reported_numeric_factorization_ratio_candidate_over_baseline":
            safe_ratio(candidate_factorizations,
                       baseline_stats[
                           "reported_numeric_factorizations_median"]),
        "objective_absolute_difference_median": median([
            value for row in rows
            if (value := finite_float(
                row.get("objective_absolute_difference"))) is not None
        ]),
    }

    csv_path = output_dir / "benchmark.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": BENCHMARK_SCHEMA,
        "benchmark_id": output_dir.name,
        "model": {
            "path": str(model),
            "sha256": expected_hash,
            "final_sha256": final_hash,
            "export_count": 0,
            **dimensions,
        },
        "model_integrity_passed": integrity_passed,
        "ipopt_first_reference": ipopt_reference,
        "candidate_feature": {
            "name": "lazyqp_zero_dual_continuation",
            "required": zero_dual_required,
            "engaged_runs": zero_dual_engaged_runs,
            "engagement_passed": zero_dual_engagement_passed,
        },
        "binary": str(binary),
        "repetitions_requested": repetitions,
        "pairs_passed": pair_count,
        "pair_summaries": summaries,
        "profiles": {
            "baseline_strict": list(baseline_args),
            "candidate_r3p24": list(candidate_args),
            "common_extra_arguments": list(common_extra_args),
        },
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "comparison": comparison,
        "targets": {
            "historical": {
                "seconds": historical_target_seconds,
                "metric": "candidate_median_solver_elapsed_seconds",
                "candidate_available_runs": candidate_time_available_runs,
                "baseline_available_runs": baseline_time_available_runs,
                "availability_required": repetitions,
                "passed": historical_passed,
                "required_for_benchmark_pass": True,
            },
            "competitiveness": {
                "seconds": competitiveness_target_seconds,
                "metric": "candidate_median_solver_elapsed_seconds",
                "candidate_available_runs": candidate_time_available_runs,
                "baseline_available_runs": baseline_time_available_runs,
                "availability_required": repetitions,
                "passed": competitiveness_passed,
                "required_for_benchmark_pass": False,
            },
            "reported_numeric_factorization_budget": {
                "maximum": factorization_budget,
                "metric":
                    "candidate_median_reported_numeric_factorizations",
                "available_runs": factorization_available_runs,
                "availability_required": repetitions,
                "passed": factorization_budget_passed,
                "required_for_benchmark_pass": False,
            },
        },
        "protocol_passed": protocol_passed,
        "benchmark_passed": benchmark_passed,
        "rows": rows,
        "csv": str(csv_path),
    }
    benchmark_json = output_dir / "benchmark.json"
    write_json_atomic(benchmark_json, payload)

    def shown(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    lines = [
        "[R3P24 STRICT-vs-CANDIDATE SAME-QCQP BENCHMARK]",
        f"benchmark={output_dir.name}",
        f"model={model}",
        f"model_sha256={expected_hash}",
        f"model_final_sha256={final_hash}",
        "export_count=0",
        f"dimensions={dimensions['n']}/"
        f"{dimensions['equalities']}/{dimensions['inequalities']}",
        f"pairs_passed={pair_count}/{repetitions}",
        "baseline_outer_inner_factorizations_median="
        f"{shown(baseline_stats['outer_iterations_median'])}/"
        f"{shown(baseline_stats['inner_iterations_median'])}/"
        f"{shown(baseline_stats['kkt_factorizations_median'])}",
        "candidate_outer_inner_factorizations_median="
        f"{shown(candidate_stats['outer_iterations_median'])}/"
        f"{shown(candidate_stats['inner_iterations_median'])}/"
        f"{shown(candidate_stats['kkt_factorizations_median'])}",
        "baseline_reported_numeric_factorizations_median="
        f"{shown(baseline_stats['reported_numeric_factorizations_median'])}",
        "candidate_reported_numeric_factorizations_median="
        f"{shown(candidate_factorizations)}",
        "baseline_solver_seconds_median="
        f"{shown(baseline_stats['solver_seconds_median'])}",
        "candidate_solver_seconds_median="
        f"{shown(candidate_stats['solver_seconds_median'])}",
        "baseline_objective_median="
        f"{shown(baseline_stats['objective_median'])}",
        "candidate_objective_median="
        f"{shown(candidate_stats['objective_median'])}",
        "solver_speedup_baseline_over_candidate="
        f"{shown(comparison['solver_speedup_baseline_over_candidate'])}",
        "objective_absolute_difference_median="
        f"{shown(comparison['objective_absolute_difference_median'])}",
        f"historical_target_seconds={historical_target_seconds}",
        "HISTORICAL_TARGET=" + ("PASS" if historical_passed else "FAIL"),
        f"competitiveness_target_seconds={competitiveness_target_seconds}",
        "COMPETITIVENESS_TARGET=" +
        ("PASS" if competitiveness_passed else "FAIL"),
        f"reported_numeric_factorization_budget={factorization_budget}",
        "FACTORIZATION_BUDGET=" +
        ("PASS" if factorization_budget_passed else "FAIL"),
        "ZERO_DUAL_CONTINUATION="
        f"{zero_dual_engaged_runs}/{repetitions} " +
        ("PASS" if zero_dual_engagement_passed else "FAIL"),
        "MODEL_INTEGRITY=" + ("PASS" if integrity_passed else "FAIL"),
        "IPOPT_FIRST_REFERENCE=" +
        ("PASS" if ipopt_reference["validation_passed"] else "FAIL"),
        "PROTOCOL=" + ("PASS" if protocol_passed else "FAIL"),
        "BENCHMARK=" + ("PASS" if benchmark_passed else "FAIL"),
        f"csv={csv_path}",
        f"json={benchmark_json}",
    ]
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"summary={summary_path}")
    return payload


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", type=Path, required=True,
                    help="already-exported immutable .qcq file")
    ap.add_argument(
        "--ipopt-reference-summary", type=Path, required=True,
        help="prior run_pglib_qcqp_r3p23.py --order ipopt-first summary.json",
    )
    ap.add_argument(
        "--binary", type=Path,
        default=(Path(os.environ["BIN"]) if os.environ.get("BIN") else
                 root / "build" / "lazyipdca_qcqp_cli"),
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--benchmark-id", default=None)
    ap.add_argument("--repetitions", "-n", type=int, default=5)
    ap.add_argument("--historical-target-seconds", type=float, default=10.0)
    ap.add_argument(
        "--competitiveness-target-seconds", type=float, default=0.5,
        help="reported separately; it does not replace the historical gate",
    )
    ap.add_argument(
        "--factorization-budget", type=int, default=45,
        help="informational median budget across reported numeric factor sites",
    )
    ap.add_argument("--baseline-args", default=DEFAULT_BASELINE_ARGS)
    ap.add_argument("--candidate-args", default=DEFAULT_CANDIDATE_ARGS)
    ap.add_argument("--common-extra-args", default="")
    ap.add_argument("--linear-backend",
                    choices=("native", "ldl", "qdldl", "mumps"),
                    default="ldl")
    ap.add_argument("--phase1-threads", type=int, default=8)
    ap.add_argument("--max-outer", type=int, default=1000)
    ap.add_argument("--max-inner", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--eq-tol", type=float, default=1.0e-6)
    ap.add_argument("--ineq-tol", type=float, default=1.0e-6)
    ap.add_argument("--bound-tol", type=float, default=1.0e-6)
    ap.add_argument("--dual-tol", type=float, default=1.0e-6)
    ap.add_argument("--compl-tol", type=float, default=1.0e-6)
    ap.add_argument("--objective-abs-tol", type=float, default=1.0e-3)
    ap.add_argument("--objective-rel-tol", type=float, default=1.0e-8)
    ap.add_argument("--live-output", action="store_true")
    ap.add_argument("--stop-on-failure", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    values = {
        "time limit": args.time_limit,
        "equality tolerance": args.eq_tol,
        "inequality tolerance": args.ineq_tol,
        "bound tolerance": args.bound_tol,
        "dual tolerance": args.dual_tol,
        "complementarity tolerance": args.compl_tol,
        "objective absolute tolerance": args.objective_abs_tol,
        "objective relative tolerance": args.objective_rel_tol,
        "historical target": args.historical_target_seconds,
        "competitiveness target": args.competitiveness_target_seconds,
        "factorization budget": args.factorization_budget,
        "phase1 threads": args.phase1_threads,
        "max outer": args.max_outer,
        "max inner": args.max_inner,
    }
    try:
        for name, value in values.items():
            require_positive(name, value)
        if args.repetitions < 1:
            raise ValueError("repetitions must be at least one")
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            base = Path(os.environ.get("BASE", "/data/projects/IPDCA_QCQP"))
            results_root = args.results_root or base / "results"
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            benchmark_id = (
                args.benchmark_id or f"R3P24_SAME_QCQP_AB_{stamp}")
            output_dir = results_root / benchmark_id
        payload = benchmark(
            model=args.model,
            binary=args.binary,
            ipopt_reference_summary=args.ipopt_reference_summary,
            output_dir=output_dir,
            repetitions=args.repetitions,
            baseline_args=shlex.split(args.baseline_args),
            candidate_args=shlex.split(args.candidate_args),
            common_extra_args=shlex.split(args.common_extra_args),
            historical_target_seconds=args.historical_target_seconds,
            competitiveness_target_seconds=
                args.competitiveness_target_seconds,
            factorization_budget=args.factorization_budget,
            stop_on_failure=args.stop_on_failure,
            linear_backend=args.linear_backend,
            phase1_threads=args.phase1_threads,
            max_outer=args.max_outer,
            max_inner=args.max_inner,
            time_limit=args.time_limit,
            equality_tolerance=args.eq_tol,
            inequality_tolerance=args.ineq_tol,
            bound_tolerance=args.bound_tol,
            dual_tolerance=args.dual_tol,
            complementarity_tolerance=args.compl_tol,
            objective_absolute_tolerance=args.objective_abs_tol,
            objective_relative_tolerance=args.objective_rel_tol,
            live_output=args.live_output,
        )
    except (OSError, RuntimeError, ValueError,
            subprocess.SubprocessError) as exc:
        print(f"[R3P24 BENCHMARK FAILURE] {exc}", file=sys.stderr)
        return 2
    return 0 if payload["benchmark_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
