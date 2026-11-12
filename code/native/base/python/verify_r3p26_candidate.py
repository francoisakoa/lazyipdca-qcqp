#!/usr/bin/env python3
"""Verify an r3p26 five-run campaign against one exact Ipopt QCQP run.

The verifier treats every input artifact as read-only and writes only a new
summary.  It hashes the serialized ``.qcq`` itself, checks the before/after
hashes recorded by the shell runner, validates the common original-QCQP
certificate, and enforces the r3p26 single-loop telemetry contract.  A timing
target is reported separately from correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence


API_VERSION = 4309
VERSION = "0.4.28r3p26-single-loop-corrected"
BUILD_ID = "v0.4.28r3p26-single-loop-corrected-api-4309"
SUMMARY_SCHEMA = "lazyipdca-r3p26-candidate-summary-v3"
RESIDUAL_KEYS = (
    "equality_violation",
    "inequality_violation",
    "bound_violation",
    "dual_infeasibility_inf",
    "complementarity_inf",
)
FORCING_ZERO_KEYS = (
    "lazyqp_inexact_forcing_attempts",
    "lazyqp_inexact_forcing_acceptances",
    "lazyqp_inexact_forcing_strict_fallbacks",
    "lazyqp_inexact_forcing_outer_retries",
    "lazyqp_inexact_forcing_model_evaluations",
)
PATH_REFIT_ZERO_KEYS = (
    "multiplier_refit_attempts",
    "multiplier_refit_successes",
)
RETRACTION_ZERO_KEYS = (
    "dca_single_loop_retraction_fallbacks",
    "lazy_dca_equality_retraction_attempts",
    "lazy_dca_equality_retraction_acceptances",
    "lazy_dca_equality_retraction_rejections",
    "lazy_dca_condensed_retraction_attempts",
    "lazy_dca_condensed_retraction_acceptances",
    "lazy_dca_condensed_retraction_rejections",
    "lazy_dca_condensed_retraction_normal_solves",
    "lazy_dca_condensed_retraction_normal_refinement_steps",
    "lazy_dca_condensed_strict_retry_retractions",
    "lazy_dca_condensed_post_retraction_modeled_offenders",
    "lazy_dca_condensed_post_retraction_omitted_offenders",
    "lazy_dca_condensed_post_retraction_working_set_expansions",
    "lazy_dca_condensed_post_retraction_expansion_fallbacks",
)
DIRECT_START_REJECTION_KEYS = (
    "dca_single_loop_direct_start_rejections_nonfinite",
    "dca_single_loop_direct_start_rejections_bounds",
    "dca_single_loop_direct_start_rejections_equalities",
    "dca_single_loop_direct_start_rejections_theta",
)
DIRECT_START_RECOVERY_REJECTION_KEYS = (
    "dca_single_loop_direct_start_recovery_rejections_nonfinite",
    "dca_single_loop_direct_start_recovery_rejections_bounds",
    "dca_single_loop_direct_start_recovery_rejections_cap",
    "dca_single_loop_direct_start_recovery_rejections_contraction",
    "dca_single_loop_direct_start_recovery_rejections_unmodeled",
)
DIRECT_START_ZERO_KEYS = (
    "dca_single_loop_direct_start_attempts",
    "dca_single_loop_direct_start_acceptances",
    *DIRECT_START_REJECTION_KEYS,
    "dca_single_loop_direct_start_violated_inequalities",
    "dca_single_loop_direct_start_working_set_rows",
    "dca_single_loop_direct_start_filter_initializations",
    "dca_single_loop_direct_start_recovery_attempts",
    "dca_single_loop_direct_start_recovery_acceptances",
    *DIRECT_START_RECOVERY_REJECTION_KEYS,
    "dca_single_loop_direct_start_recovery_working_set_additions",
    "dca_single_loop_direct_start_strict_latches",
)
TERMINAL_KKT_ZERO_KEYS = (
    "terminal_kkt_iterations",
    "terminal_kkt_accepted_steps",
    "terminal_kkt_active_set_drops",
    "terminal_kkt_attempts",
    "terminal_kkt_rejected_steps",
    "terminal_kkt_skipped_unstable",
    "terminal_kkt_active_stability",
)
EQUALITY_GMRES_ZERO_KEYS = (
    "equality_qp_original_kkt_gmres_attempts",
    "equality_qp_original_kkt_gmres_acceptances",
    "equality_qp_original_kkt_gmres_iterations",
)


class VerificationError(RuntimeError):
    """A campaign artifact violated a correctness or architecture gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"{label} {path} is not a JSON object")
    return payload


def finite_number(record: dict[str, Any], key: str, label: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label}: {key} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise VerificationError(f"{label}: {key} is not finite")
    return result


def integer(record: dict[str, Any], key: str, label: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationError(f"{label}: {key} is not an integer")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def validate_hash_log(
    path: Path, expected_hash: str, expected_runs: int
) -> None:
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    except OSError as exc:
        raise VerificationError(f"cannot read model hash log {path}: {exc}") from exc
    require(len(lines) == expected_runs,
            f"model hash log contains {len(lines)} rows, expected {expected_runs}")
    seen: set[int] = set()
    for line in lines:
        fields = line.split("\t")
        require(len(fields) == 4, f"malformed model hash row: {line!r}")
        try:
            index, return_code = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise VerificationError(f"malformed model hash row: {line!r}") from exc
        require(index not in seen and 1 <= index <= expected_runs,
                f"invalid or duplicate run index in model hash row: {line!r}")
        seen.add(index)
        require(return_code == 0, f"run {index}: process return code is {return_code}")
        require(fields[2] == expected_hash,
                f"run {index}: pre-solve model SHA-256 differs")
        require(fields[3] == expected_hash,
                f"run {index}: post-solve model SHA-256 differs")
    require(seen == set(range(1, expected_runs + 1)),
            "model hash log does not cover every expected run")


def validate_ipopt(
    result: dict[str, Any], expected_hash: str, tolerance: float
) -> float:
    require(result.get("model_sha256") == expected_hash,
            "Ipopt model SHA-256 differs from the current QCQP")
    require(result.get("success") is True, "Ipopt did not report success")
    require(result.get("kkt_passed") is True,
            "Ipopt did not pass its original-QCQP KKT gate")
    dimensions = result.get("dimensions")
    require(isinstance(dimensions, dict), "Ipopt dimensions are missing")
    require((dimensions.get("variables"), dimensions.get("equalities"),
             dimensions.get("inequalities")) == (1088, 980, 979),
            "Ipopt dimensions are not case118 1088/980/979")
    residuals = result.get("residuals")
    require(isinstance(residuals, dict), "Ipopt residuals are missing")
    for key in (
        "equality_inf", "inequality_inf", "bound_inf", "dual_inf",
        "complementarity_inf", "dual_sign_inf",
    ):
        value = finite_number(residuals, key, "Ipopt")
        require(0.0 <= value <= tolerance,
                f"Ipopt: {key}={value} exceeds {tolerance}")
    return finite_number(result, "objective", "Ipopt")


def validate_single_loop_contract(
    result: dict[str, Any], index: int, *, tolerance: float,
    require_zero_dual: bool,
) -> dict[str, int | float]:
    label = f"run {index}"
    require(result.get("schema") == "lazyipdca-qcqp-result-v2",
            f"{label}: unexpected result schema")
    require(result.get("api_version") == API_VERSION,
            f"{label}: API version is not {API_VERSION}")
    require(result.get("version") == VERSION,
            f"{label}: unexpected solver version")
    require(result.get("build_id") == BUILD_ID,
            f"{label}: unexpected build identifier")
    require(result.get("status") == "solved", f"{label}: status is not solved")
    require(result.get("ipopt_desired_convergence") is True,
            f"{label}: final original-QCQP certificate did not pass")

    for key in RESIDUAL_KEYS:
        value = finite_number(result, key, label)
        require(0.0 <= value <= tolerance,
                f"{label}: {key}={value} exceeds {tolerance}")

    attempts = integer(result, "dca_single_loop_attempts", label)
    acceptances = integer(result, "dca_single_loop_acceptances", label)
    steps = integer(result, "dca_single_loop_inner_steps", label)
    factors = integer(result, "dca_single_loop_factorizations", label)
    total_factors = integer(result, "kkt_factorizations", label)
    require(attempts > 0, f"{label}: single-loop path was not exercised")
    require(0 < acceptances <= attempts,
            f"{label}: invalid single-loop acceptance count")
    require(steps == attempts,
            f"{label}: every attempted majorant must take exactly one PC step")
    regularization_retries = integer(result, "kkt_regularization_retries", label)
    require(steps <= factors <= steps + regularization_retries,
            f"{label}: factorization count is inconsistent with one shared-factor "
            "PC step plus recorded numerical retries")
    require(total_factors == factors,
            f"{label}: KKT work exists outside the one-step single-loop path")
    require(integer(result, "dca_single_loop_strict_fallbacks", label) == 0,
            f"{label}: a forbidden strict/complete inner fallback ran")
    require(integer(
        result, "dca_single_loop_state_transport_acceptances", label
    ) > 0, f"{label}: primal-dual continuation state was never transported")
    require(integer(
        result, "dca_single_loop_keyed_transport_acceptances", label
    ) > 0, f"{label}: row-keyed lambda/z/s transport was never accepted")
    require(integer(result, "inner_iteration_limit_fallback_acceptances", label) == 0,
            f"{label}: historical iteration-limit inner acceptance ran")
    require(integer(result, "lazyqp_affine_elimination_fallbacks", label) == 0,
            f"{label}: affine reduction fell back to a full strict model")
    require(integer(result, "lazyqp_reduced_kkt_fallbacks", label) == 0,
            f"{label}: reduced KKT fell back to the full KKT form")
    require(integer(result, "lazyqp_full_kkt_factorizations", label) == 0,
            f"{label}: full-KKT factorization bypassed the candidate path")
    for key in (
        "lazy_dca_corrector_attempts",
        "lazy_dca_iterated_soc_attempts",
        "soc_attempts",
    ):
        require(integer(result, key, label) == 0,
                f"{label}: forbidden auxiliary correction ran ({key})")
    for key in (
        "lazyqp_numeric_kkt_reuse_attempts",
        "lazyqp_numeric_kkt_reuse_acceptances",
        "lazyqp_numeric_kkt_reuse_fallbacks",
        "lazyqp_numeric_kkt_reuse_iterations",
    ):
        require(integer(result, key, label) == 0,
                f"{label}: disabled numerical-factor reuse ran ({key})")

    for key in PATH_REFIT_ZERO_KEYS:
        require(integer(result, key, label) == 0,
                f"{label}: forbidden intermediate multiplier refit ({key})")
    refit_fallback_reuses = integer(
        result, "multiplier_refit_fallback_reuses", label)
    require(refit_fallback_reuses >= 0,
            f"{label}: invalid transported multiplier-reuse count")
    require(integer(result, "dca_single_loop_refit_fallbacks", label) == 0,
            f"{label}: an accepted endpoint failed multiplier transport")
    require(integer(result, "dca_single_loop_bootstrap_full_fits", label) == 0,
            f"{label}: a forbidden full multiplier bootstrap ran")
    require(integer(
        result, "dca_single_loop_reduced_bootstrap_attempts", label
    ) == 1, f"{label}: reduced first-model bootstrap must run once")
    require(integer(
        result, "dca_single_loop_reduced_bootstrap_acceptances", label
    ) == 1, f"{label}: reduced first-model bootstrap was not accepted")
    baseline_prox = finite_number(
        result, "dca_single_loop_baseline_proximal_regularization", label)
    require(math.isclose(baseline_prox, 5e-2, rel_tol=0.0, abs_tol=1e-15),
            f"{label}: effective single-loop baseline proximal curvature "
            f"is {baseline_prox}, expected 0.05")
    require(integer(result, "dca_single_loop_full_refit_calls", label) == 1,
            f"{label}: a solved run must perform exactly one final full refit")
    require(integer(result, "dca_single_loop_refits_skipped", label) > 0,
            f"{label}: transported multipliers did not replace any path refit")
    require(result.get("termination_multiplier_fit_converged") is True,
            f"{label}: final full multiplier fit did not converge")
    require(result.get("termination_multiplier_source") in {"fit", "fit-refined"},
            f"{label}: certificate did not use the independent final fit")

    for key in FORCING_ZERO_KEYS:
        require(integer(result, key, label) == 0,
                f"{label}: disabled forcing executed work ({key})")

    for key in RETRACTION_ZERO_KEYS:
        require(integer(result, key, label) == 0,
                f"{label}: forbidden auxiliary retraction work ran ({key})")
    require(integer(result, "dca_single_loop_retractions_skipped", label) > 0,
            f"{label}: no accepted endpoint recorded the retraction-free route")
    require(finite_number(
        result, "lazy_dca_max_equality_retraction_inf", label
    ) == 0.0, f"{label}: disabled retraction recorded a nonzero correction")

    for key in TERMINAL_KKT_ZERO_KEYS:
        require(integer(result, key, label) == 0,
                f"{label}: disabled terminal KKT work ran ({key})")
    for key in EQUALITY_GMRES_ZERO_KEYS:
        require(integer(result, key, label) == 0,
                f"{label}: disabled equality-QP GMRES work ran ({key})")

    require(integer(result, "kkt_symbolic_analyses", label) >= 1,
            f"{label}: no KKT symbolic structure was assembled")
    require(integer(result, "kkt_symbolic_reuses", label) >= 1,
            f"{label}: persistent KKT symbolic structure was not reused")
    require(integer(result, "lazyqp_kkt_triplet_layout_builds", label) >= 1,
            f"{label}: no KKT triplet layout was preassembled")
    require(integer(result, "lazyqp_kkt_triplet_layout_reuses", label) >= 1,
            f"{label}: preassembled KKT triplet layout was not reused")
    require(integer(result, "lazyqp_kkt_triplet_layout_builds", label) < factors,
            f"{label}: KKT triplets were rebuilt for every factorization")
    # The LazyQP layout cache is the mandatory structural scatter table: it
    # owns stable (i,j) slots and overwrites values in place.  SuiteSparse's
    # additional raw-triplet-to-CSC scatter is backend/regularization
    # dependent and legitimately reports 0/0 with adaptive KKT shifts.
    scatter_builds = integer(result, "kkt_numeric_scatter_builds", label)
    scatter_reuses = integer(result, "kkt_numeric_scatter_reuses", label)
    require(scatter_builds >= 0 and scatter_reuses >= 0,
            f"{label}: invalid backend numeric-scatter counters")
    if scatter_builds > 0 or scatter_reuses > 0:
        require(scatter_builds >= 1 and scatter_reuses >= 1,
                f"{label}: incomplete backend numeric-scatter lifecycle")
        require(scatter_builds < factors,
                f"{label}: LDL scatter was rebuilt for every factorization")
    require("persistent_symbolic" in str(result.get("linear_solver_backend", "")),
            f"{label}: result does not identify the persistent-symbolic LDL route")

    zero_attempts = integer(
        result, "lazyqp_zero_dual_continuation_attempts", label)
    zero_mapped = integer(
        result, "lazyqp_zero_dual_continuation_mapped_attempts", label)
    zero_direct = integer(
        result, "lazyqp_zero_dual_continuation_direct_attempts", label)
    zero_acceptances = integer(
        result, "lazyqp_zero_dual_continuation_acceptances", label)
    zero_fallbacks = integer(
        result, "lazyqp_zero_dual_continuation_fallbacks", label)
    require(zero_attempts == zero_mapped + zero_direct,
            f"{label}: ZERO_DUAL provenance counters do not partition attempts")
    require(zero_attempts == zero_acceptances + zero_fallbacks,
            f"{label}: ZERO_DUAL outcomes do not partition attempts")
    if require_zero_dual:
        require(zero_attempts > 0, f"{label}: ZERO_DUAL was not exercised")

    for key in DIRECT_START_ZERO_KEYS:
        require(integer(result, key, label) == 0,
                f"{label}: disabled direct-start path ran ({key})")

    return {
        "attempts": attempts,
        "acceptances": acceptances,
        "steps": steps,
        "factorizations": factors,
        "retraction_fallbacks": 0,
        "multiplier_model_reuses": refit_fallback_reuses,
        "zero_dual_attempts": zero_attempts,
    }


def validate_campaign(
    *, qcq: Path, ipopt_json: Path, run_dir: Path, expected_runs: int,
    tolerance: float, target_seconds: float,
    objective_absolute_tolerance: float,
    objective_relative_tolerance: float,
    require_zero_dual: bool,
) -> dict[str, Any]:
    require(qcq.is_file(), f"QCQP does not exist: {qcq}")
    require(run_dir.is_dir(), f"candidate run directory does not exist: {run_dir}")
    model_hash = sha256_file(qcq)
    validate_hash_log(run_dir / "model_hashes.tsv", model_hash, expected_runs)
    ipopt = load_object(ipopt_json, "Ipopt result")
    ipopt_objective = validate_ipopt(ipopt, model_hash, tolerance)

    records: list[dict[str, Any]] = []
    times: list[float] = []
    telemetry: list[dict[str, int | float]] = []
    for index in range(1, expected_runs + 1):
        path = run_dir / f"run_{index}.json"
        result = load_object(path, f"candidate run {index}")
        stats = validate_single_loop_contract(
            result, index, tolerance=tolerance,
            require_zero_dual=require_zero_dual,
        )
        objective = finite_number(result, "objective", f"run {index}")
        scale = max(1.0, abs(ipopt_objective), abs(objective))
        allowance = objective_absolute_tolerance + objective_relative_tolerance * scale
        difference = abs(objective - ipopt_objective)
        require(difference <= allowance,
                f"run {index}: objective difference {difference} exceeds {allowance}")
        elapsed = finite_number(result, "elapsed_seconds", f"run {index}")
        require(elapsed >= 0.0, f"run {index}: elapsed time is negative")
        records.append(result)
        times.append(elapsed)
        telemetry.append(stats)

    median_seconds = statistics.median(times)
    return {
        "schema": SUMMARY_SCHEMA,
        "model": str(qcq.resolve()),
        "model_sha256": model_hash,
        "ipopt_result": str(ipopt_json.resolve()),
        "runs_passed": len(records),
        "solver_seconds": times,
        "solver_seconds_min": min(times),
        "solver_seconds_median": median_seconds,
        "solver_seconds_max": max(times),
        "single_loop_telemetry": telemetry,
        "target_seconds": target_seconds,
        "target_passed": median_seconds < target_seconds,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--qcq", required=True, type=Path)
    result.add_argument("--ipopt-json", required=True, type=Path)
    result.add_argument("--run-dir", required=True, type=Path)
    result.add_argument("--expected-runs", type=int, default=5)
    result.add_argument("--tolerance", type=float, default=1e-6)
    result.add_argument("--target-seconds", type=float, default=0.5)
    result.add_argument("--objective-absolute-tolerance", type=float, default=1e-3)
    result.add_argument("--objective-relative-tolerance", type=float, default=1e-8)
    result.add_argument("--require-zero-dual", action="store_true")
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
    if args.expected_runs < 1:
        print("error: --expected-runs must be positive", file=sys.stderr)
        return 2

    try:
        summary = validate_campaign(
            qcq=args.qcq.resolve(),
            ipopt_json=args.ipopt_json.resolve(),
            run_dir=args.run_dir.resolve(),
            expected_runs=args.expected_runs,
            tolerance=args.tolerance,
            target_seconds=args.target_seconds,
            objective_absolute_tolerance=args.objective_absolute_tolerance,
            objective_relative_tolerance=args.objective_relative_tolerance,
            require_zero_dual=args.require_zero_dual,
        )
    except VerificationError as exc:
        print(f"R3P26_CANDIDATE_VALIDATION=FAIL: {exc}", file=sys.stderr)
        return 1

    output = args.output or args.run_dir.resolve() / "benchmark_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    for index, (elapsed, telemetry) in enumerate(
        zip(summary["solver_seconds"], summary["single_loop_telemetry"]),
        start=1,
    ):
        print(
            f"RUN={index}", f"solver_s={elapsed}",
            f"attempts/steps/fact={telemetry['attempts']}/"
            f"{telemetry['steps']}/{telemetry['factorizations']}",
            f"retractions={telemetry['retraction_fallbacks']}",
            f"zero_dual={telemetry['zero_dual_attempts']}",
        )
    print(f"R3P26_CANDIDATE_VALIDATION=PASS runs={summary['runs_passed']}")
    print(f"solver_seconds_median={summary['solver_seconds_median']}")
    target_label = format(summary["target_seconds"], "g").replace(".", "_")
    print(f"TARGET_{target_label}="
          f"{'PASS' if summary['target_passed'] else 'FAIL'}")
    print(f"summary={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
