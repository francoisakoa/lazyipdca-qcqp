#!/usr/bin/env python3
"""Verify the r3p27 reduced-versus-hybrid exact-case118 campaign.

Correctness is independent of timing: both LazyIPDCA modes must solve the
same immutable serialized QCQP, pass the original-QCQP certificate, agree
with the direct Ipopt objective, and satisfy the single-loop architecture
contract.  Median speed, paired speedups, and the 0.5 s target are reported
but never used as correctness gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence


API_VERSION = 4310
VERSION = "0.4.28r3p27-single-loop-hybrid-kkt"
BUILD_ID = "v0.4.28r3p27-single-loop-hybrid-kkt-api-4310"
SUMMARY_SCHEMA = "lazyipdca-r3p27-exact-case118-ab-summary-v1"

_BASE_PATH = Path(__file__).with_name("verify_r3p26_candidate.py")
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_lazyipdca_r3p26_verifier_base", _BASE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load verifier base: {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)
_BASE.API_VERSION = API_VERSION
_BASE.VERSION = VERSION
_BASE.BUILD_ID = BUILD_ID

VerificationError = _BASE.VerificationError


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def integer(record: dict[str, Any], key: str, label: str) -> int:
    return _BASE.integer(record, key, label)


def finite_number(record: dict[str, Any], key: str, label: str) -> float:
    return _BASE.finite_number(record, key, label)


def validate_ipopt_hash_log(path: Path, expected_hash: str) -> None:
    try:
        rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except OSError as exc:
        raise VerificationError(f"cannot read Ipopt hash log {path}: {exc}") from exc
    require(len(rows) == 1, "Ipopt hash log must contain exactly one row")
    fields = rows[0].split("\t")
    require(fields == ["ipopt", "0", expected_hash, expected_hash],
            "Ipopt pre/post model hash record is inconsistent")


def validate_warmup_hash_log(
    path: Path, expected_hash: str, expected_warmups: int
) -> None:
    try:
        rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except OSError as exc:
        raise VerificationError(f"cannot read warm-up hash log {path}: {exc}") from exc
    require(len(rows) == 2 * expected_warmups,
            f"warm-up hash log contains {len(rows)} rows, "
            f"expected {2 * expected_warmups}")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        fields = row.split("\t")
        require(len(fields) == 5, f"malformed warm-up hash row: {row!r}")
        mode = fields[0]
        try:
            index, return_code = int(fields[1]), int(fields[2])
        except ValueError as exc:
            raise VerificationError(f"malformed warm-up hash row: {row!r}") from exc
        require(mode in {"reduced", "hybrid"},
                f"invalid warm-up mode: {mode!r}")
        key = (mode, index)
        require(key not in seen and 1 <= index <= expected_warmups,
                f"invalid or duplicate warm-up row: {row!r}")
        seen.add(key)
        require(return_code == 0, f"{mode} warm-up {index} returned {return_code}")
        require(fields[3] == expected_hash and fields[4] == expected_hash,
                f"{mode} warm-up {index} changed the QCQP")
    expected = {
        (mode, index)
        for mode in ("reduced", "hybrid")
        for index in range(1, expected_warmups + 1)
    }
    require(seen == expected, "warm-up hash log is incomplete")


def validate_schedule(path: Path, expected_runs: int) -> None:
    try:
        rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except OSError as exc:
        raise VerificationError(f"cannot read measured schedule {path}: {exc}") from exc
    expected: list[tuple[int, int, str]] = []
    sequence = 0
    for pair in range(1, expected_runs + 1):
        order = ("reduced", "hybrid") if pair % 2 else ("hybrid", "reduced")
        for mode in order:
            sequence += 1
            expected.append((sequence, pair, mode))
    actual: list[tuple[int, int, str]] = []
    for row in rows:
        fields = row.split("\t")
        require(len(fields) == 3, f"malformed measured schedule row: {row!r}")
        try:
            actual.append((int(fields[0]), int(fields[1]), fields[2]))
        except ValueError as exc:
            raise VerificationError(
                f"malformed measured schedule row: {row!r}") from exc
    require(actual == expected,
            "measured order is not paired reduced/hybrid with alternating lead")


def validate_optimizer_order(
    path: Path, expected_warmups: int, expected_runs: int
) -> None:
    try:
        rows = [line.split("\t") for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise VerificationError(
            f"cannot read optimizer order log {path}: {exc}") from exc
    expected: list[tuple[int, str, str, int]] = [(1, "ipopt", "ipopt", 0)]
    sequence = 1
    for index in range(1, expected_warmups + 1):
        for mode in ("reduced", "hybrid"):
            sequence += 1
            expected.append((sequence, "warmup", mode, index))
    for pair in range(1, expected_runs + 1):
        order = ("reduced", "hybrid") if pair % 2 else ("hybrid", "reduced")
        for mode in order:
            sequence += 1
            expected.append((sequence, "measured", mode, pair))
    actual: list[tuple[int, str, str, int]] = []
    for fields in rows:
        require(len(fields) == 4,
                f"malformed optimizer order row: {fields!r}")
        try:
            actual.append((int(fields[0]), fields[1], fields[2], int(fields[3])))
        except ValueError as exc:
            raise VerificationError(
                f"malformed optimizer order row: {fields!r}") from exc
    require(actual == expected,
            "optimizer chronology is incomplete or Ipopt was not first")


def validate_fastpath(result: dict[str, Any], label: str) -> dict[str, int]:
    requests = integer(result, "lazyqp_model_support_fastpath_requests", label)
    effective = integer(
        result, "lazyqp_model_support_fastpath_effective_subproblems", label)
    builds = integer(result, "lazyqp_model_support_builds", label)
    eligible = integer(result, "lazyqp_model_support_eligible_models", label)
    generic = integer(
        result, "lazyqp_model_support_generic_fallback_models", label)
    values = integer(result, "lazyqp_model_support_value_evaluations", label)
    gradients = integer(result, "lazyqp_model_support_gradient_evaluations", label)
    require(requests > 0, f"{label}: model-support fast path was not requested")
    require(effective > 0 and builds > 0 and eligible > 0,
            f"{label}: model-support fast path was not exercised")
    require(values > 0 and gradients > 0,
            f"{label}: sparse support value/gradient kernels did no work")
    require(0 <= effective <= builds <= requests and eligible >= effective,
            f"{label}: invalid model-support lifecycle counters")
    require(generic >= 0, f"{label}: negative generic model fallback count")
    return {
        "requests": requests,
        "effective_subproblems": effective,
        "eligible_models": eligible,
        "generic_fallback_models": generic,
        "value_evaluations": values,
        "gradient_evaluations": gradients,
    }


def validate_kkt_mode(
    result: dict[str, Any], mode: str, label: str
) -> dict[str, int]:
    hybrid_attempts = integer(result, "lazyqp_hybrid_kkt_attempts", label)
    hybrid_factors = integer(result, "lazyqp_hybrid_kkt_factorizations", label)
    hybrid_fallbacks = integer(result, "lazyqp_hybrid_kkt_fallbacks", label)
    retained_total = integer(
        result, "lazyqp_hybrid_kkt_retained_rows_total", label)
    retained_max = integer(result, "lazyqp_hybrid_kkt_retained_rows_max", label)
    avoided = integer(
        result, "lazyqp_hybrid_kkt_outer_product_triplets_avoided", label)
    emitted = integer(
        result, "lazyqp_hybrid_kkt_border_triplets_emitted", label)
    reduced_factors = integer(result, "lazyqp_reduced_kkt_factorizations", label)
    require(reduced_factors > 0, f"{label}: reduced primal block was not exercised")

    if mode == "reduced":
        require((hybrid_attempts, hybrid_factors, hybrid_fallbacks,
                 retained_total, retained_max, avoided, emitted) == (0,) * 7,
                f"{label}: hybrid telemetry is nonzero in reduced control")
    else:
        require(hybrid_attempts > 0 and hybrid_factors > 0,
                f"{label}: hybrid KKT route was not exercised")
        require(hybrid_factors == hybrid_attempts,
                f"{label}: a hybrid factorization attempt was not accepted")
        require(hybrid_attempts == integer(
                    result, "dca_single_loop_factorizations", label),
                f"{label}: hybrid route did not own every single-loop factorization")
        require(retained_max == 2,
                f"{label}: case118 hybrid KKT did not retain exactly two rows")
        require(retained_total == 2 * hybrid_factors,
                f"{label}: retained-row accounting is inconsistent")
        require(avoided > emitted > 0,
                f"{label}: dense-row triplet reduction was not realized")
        require(hybrid_fallbacks == 0,
                f"{label}: hybrid KKT fell back")
    return {
        "hybrid_attempts": hybrid_attempts,
        "hybrid_factorizations": hybrid_factors,
        "hybrid_fallbacks": hybrid_fallbacks,
        "retained_rows_total": retained_total,
        "retained_rows_max": retained_max,
        "outer_product_triplets_avoided": avoided,
        "border_triplets_emitted": emitted,
        "max_kkt_dimension": integer(result, "lazyqp_max_kkt_dimension", label),
    }


def validate_mode(
    *, mode: str, run_dir: Path, expected_runs: int, model_hash: str,
    ipopt_objective: float, tolerance: float,
    objective_absolute_tolerance: float, objective_relative_tolerance: float,
    require_zero_dual: bool, require_model_support_fastpath: bool,
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    _BASE.validate_hash_log(
        run_dir / "model_hashes.tsv", model_hash, expected_runs)
    times: list[float] = []
    objectives: list[float] = []
    telemetry: list[dict[str, Any]] = []
    for index in range(1, expected_runs + 1):
        label = f"{mode} run {index}"
        result = _BASE.load_object(run_dir / f"run_{index}.json", label)
        single_loop = _BASE.validate_single_loop_contract(
            result, index, tolerance=tolerance,
            require_zero_dual=require_zero_dual,
        )
        fastpath = (validate_fastpath(result, label)
                    if require_model_support_fastpath else {})
        kkt_mode = validate_kkt_mode(result, mode, label)
        objective = finite_number(result, "objective", label)
        scale = max(1.0, abs(ipopt_objective), abs(objective))
        allowance = (objective_absolute_tolerance
                     + objective_relative_tolerance * scale)
        difference = abs(objective - ipopt_objective)
        require(difference <= allowance,
                f"{label}: objective difference {difference} exceeds {allowance}")
        elapsed = finite_number(result, "elapsed_seconds", label)
        require(elapsed >= 0.0, f"{label}: elapsed time is negative")
        times.append(elapsed)
        objectives.append(objective)
        telemetry.append({
            "single_loop": single_loop,
            "model_support": fastpath,
            "kkt": kkt_mode,
        })
    return times, objectives, telemetry


def validate_campaign(
    *, qcq: Path, ipopt_json: Path, ipopt_hashes: Path,
    reduced_dir: Path, hybrid_dir: Path, warmup_hashes: Path,
    warmups: int, schedule: Path, optimizer_order: Path,
    expected_runs: int, tolerance: float,
    target_seconds: float, objective_absolute_tolerance: float,
    objective_relative_tolerance: float, require_zero_dual: bool,
    require_model_support_fastpath: bool,
) -> dict[str, Any]:
    require(qcq.is_file(), f"QCQP does not exist: {qcq}")
    require(reduced_dir.is_dir(), f"reduced directory does not exist: {reduced_dir}")
    require(hybrid_dir.is_dir(), f"hybrid directory does not exist: {hybrid_dir}")
    model_hash = _BASE.sha256_file(qcq)
    validate_ipopt_hash_log(ipopt_hashes, model_hash)
    validate_warmup_hash_log(warmup_hashes, model_hash, warmups)
    validate_schedule(schedule, expected_runs)
    validate_optimizer_order(optimizer_order, warmups, expected_runs)

    ipopt = _BASE.load_object(ipopt_json, "Ipopt result")
    require(ipopt.get("schema") == "lazyipdca-qcqp-ipopt-direct-result-v1",
            "Ipopt result schema is invalid")
    ipopt_objective = _BASE.validate_ipopt(ipopt, model_hash, tolerance)
    ipopt_seconds = finite_number(ipopt, "solver_seconds", "Ipopt")
    require(ipopt_seconds >= 0.0, "Ipopt solver time is negative")
    ipopt_iterations = integer(ipopt, "iterations", "Ipopt")
    require(ipopt_iterations >= 0, "Ipopt iteration count is negative")
    common = {
        "expected_runs": expected_runs,
        "model_hash": model_hash,
        "ipopt_objective": ipopt_objective,
        "tolerance": tolerance,
        "objective_absolute_tolerance": objective_absolute_tolerance,
        "objective_relative_tolerance": objective_relative_tolerance,
        "require_zero_dual": require_zero_dual,
        "require_model_support_fastpath": require_model_support_fastpath,
    }
    reduced_times, reduced_objectives, reduced_telemetry = validate_mode(
        mode="reduced", run_dir=reduced_dir, **common)
    hybrid_times, hybrid_objectives, hybrid_telemetry = validate_mode(
        mode="hybrid", run_dir=hybrid_dir, **common)

    pair_allowance = (objective_absolute_tolerance
                      + objective_relative_tolerance
                      * max(1.0, abs(ipopt_objective)))
    for index, (reduced_objective, hybrid_objective) in enumerate(
        zip(reduced_objectives, hybrid_objectives), start=1
    ):
        require(abs(reduced_objective - hybrid_objective) <= pair_allowance,
                f"pair {index}: reduced/hybrid objectives disagree")

    reduced_median = statistics.median(reduced_times)
    hybrid_median = statistics.median(hybrid_times)
    paired_speedups = [
        100.0 * (reduced - hybrid) / reduced if reduced > 0.0 else 0.0
        for reduced, hybrid in zip(reduced_times, hybrid_times)
    ]
    dimension_deltas = [
        hybrid["kkt"]["max_kkt_dimension"]
        - reduced["kkt"]["max_kkt_dimension"]
        for reduced, hybrid in zip(reduced_telemetry, hybrid_telemetry)
    ]
    require(all(delta == 2 for delta in dimension_deltas),
            "hybrid KKT must add exactly two retained-row dimensions")
    return {
        "schema": SUMMARY_SCHEMA,
        "validation_passed": True,
        "model": str(qcq.resolve()),
        "model_sha256": model_hash,
        "ipopt_result": str(ipopt_json.resolve()),
        "ipopt_objective": ipopt_objective,
        "ipopt_solver_seconds": ipopt_seconds,
        "ipopt_iterations": ipopt_iterations,
        "warmups_per_mode": warmups,
        "runs_passed": {"reduced": expected_runs, "hybrid": expected_runs},
        "reduced": {
            "solver_seconds": reduced_times,
            "solver_seconds_min": min(reduced_times),
            "solver_seconds_median": reduced_median,
            "solver_seconds_max": max(reduced_times),
            "objectives": reduced_objectives,
            "telemetry": reduced_telemetry,
        },
        "hybrid": {
            "solver_seconds": hybrid_times,
            "solver_seconds_min": min(hybrid_times),
            "solver_seconds_median": hybrid_median,
            "solver_seconds_max": max(hybrid_times),
            "objectives": hybrid_objectives,
            "telemetry": hybrid_telemetry,
        },
        "paired_speedup_percent": paired_speedups,
        "paired_speedup_percent_median": statistics.median(paired_speedups),
        "max_kkt_dimension_deltas": dimension_deltas,
        "hybrid_faster": hybrid_median < reduced_median,
        "target_seconds": target_seconds,
        "target_passed": hybrid_median < target_seconds,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--qcq", required=True, type=Path)
    result.add_argument("--ipopt-json", required=True, type=Path)
    result.add_argument("--ipopt-hashes", required=True, type=Path)
    result.add_argument("--reduced-dir", required=True, type=Path)
    result.add_argument("--hybrid-dir", required=True, type=Path)
    result.add_argument("--warmup-hashes", required=True, type=Path)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--schedule", required=True, type=Path)
    result.add_argument("--optimizer-order", required=True, type=Path)
    result.add_argument("--expected-runs", type=int, default=5)
    result.add_argument("--tolerance", type=float, default=1e-6)
    result.add_argument("--target-seconds", type=float, default=0.5)
    result.add_argument("--objective-absolute-tolerance", type=float, default=1e-3)
    result.add_argument("--objective-relative-tolerance", type=float, default=1e-8)
    result.add_argument("--require-zero-dual", action="store_true")
    result.add_argument("--require-model-support-fastpath", action="store_true")
    result.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in (
        "tolerance", "target_seconds", "objective_absolute_tolerance",
        "objective_relative_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            print(f"error: --{name.replace('_', '-')} must be finite and positive",
                  file=sys.stderr)
            return 2
    if args.expected_runs < 1 or args.warmups < 0:
        print("error: run count must be positive and warmups nonnegative",
              file=sys.stderr)
        return 2
    try:
        summary = validate_campaign(
            qcq=args.qcq.resolve(),
            ipopt_json=args.ipopt_json.resolve(),
            ipopt_hashes=args.ipopt_hashes.resolve(),
            reduced_dir=args.reduced_dir.resolve(),
            hybrid_dir=args.hybrid_dir.resolve(),
            warmup_hashes=args.warmup_hashes.resolve(),
            warmups=args.warmups,
            schedule=args.schedule.resolve(),
            optimizer_order=args.optimizer_order.resolve(),
            expected_runs=args.expected_runs,
            tolerance=args.tolerance,
            target_seconds=args.target_seconds,
            objective_absolute_tolerance=args.objective_absolute_tolerance,
            objective_relative_tolerance=args.objective_relative_tolerance,
            require_zero_dual=args.require_zero_dual,
            require_model_support_fastpath=args.require_model_support_fastpath,
        )
    except VerificationError as exc:
        print(f"R3P27_AB_VALIDATION=FAIL: {exc}", file=sys.stderr)
        return 1

    output = args.output or args.hybrid_dir.parent / "benchmark_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"runs_passed={summary['runs_passed']['reduced']}/"
          f"{summary['runs_passed']['hybrid']} per_mode={args.expected_runs}")
    print(f"ipopt_solver_seconds={summary['ipopt_solver_seconds']} "
          f"iterations={summary['ipopt_iterations']}")
    print(f"reduced_solver_seconds_median="
          f"{summary['reduced']['solver_seconds_median']}")
    print(f"hybrid_solver_seconds_median="
          f"{summary['hybrid']['solver_seconds_median']}")
    print(f"paired_speedup_percent_median="
          f"{summary['paired_speedup_percent_median']}")
    print(f"HYBRID_FASTER={'PASS' if summary['hybrid_faster'] else 'FAIL'} "
          "(informative)")
    target_label = format(summary["target_seconds"], "g").replace(".", "_")
    print(f"TARGET_{target_label}="
          f"{'PASS' if summary['target_passed'] else 'FAIL'} (informative)")
    print("R3P27_AB_VALIDATION=PASS")
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
