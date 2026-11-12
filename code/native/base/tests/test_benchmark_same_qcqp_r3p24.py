#!/usr/bin/env python3
"""Contract tests for the r3p24 strict-vs-candidate same-QCQP benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from benchmark_same_qcqp_r3p24 import (  # noqa: E402
    DEFAULT_BASELINE_ARGS,
    DEFAULT_CANDIDATE_ARGS,
    benchmark,
    reported_numeric_factorizations,
    validate_ipopt_reference_summary,
)
from export_pglib_qcqp_r3p23 import sha256_file  # noqa: E402


QCQ = """LAZYIPDCA_QCQP_V1
name r3p24_same_qcqp_toy
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
import json
from pathlib import Path
import sys

if "--version" in sys.argv:
    print("fake-lazyipdca-r3p24")
    raise SystemExit(0)

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

model = Path(value("--problem")).resolve()
output = Path(value("--json")).resolve()
candidate = (
    "--lazyqp-zero-dual-continuation" in sys.argv or
    "--lazyqp-inexact-forcing" in sys.argv
)
if "--mutate-model" in sys.argv:
    model.write_text(model.read_text() + "# illicit mutation\n")

payload = {
    "status": "solved",
    "objective": 1.0 + (1.0e-10 if candidate else 0.0),
    "equality_violation": 0.0,
    "inequality_violation": 0.0,
    "bound_violation": 0.0,
    "dual_infeasibility_inf": 5.0e-8,
    "complementarity_inf": 2.0e-8,
    "scaled_nlp_error": 5.0e-8,
    "outer_iterations": 12 if candidate else 49,
    "inner_iterations": 50 if candidate else 711,
    "kkt_factorizations": 35 if candidate else 653,
    "lazy_dca_condensed_factorizations": 3 if candidate else 53,
    "multiplier_row_space_factorizations": 2 if candidate else 61,
    "restoration_linear_solves": 3 if candidate else 5,
    "restoration_matrix_free_solves": 1 if candidate else 2,
    "restoration_harmonic_factorizations": 2 if candidate else 4,
    "lazy_dca_condensed_tangent_inertia_attempts": 1 if candidate else 0,
    "terminal_kkt_attempts": 0,
    "terminal_kkt_iterations": 0,
    "lazyqp_zero_dual_continuation_attempts": 5 if candidate else 0,
    "lazyqp_zero_dual_continuation_mapped_attempts": 5 if candidate else 0,
    "lazyqp_zero_dual_continuation_direct_attempts": 0,
    "lazyqp_zero_dual_continuation_acceptances": 3 if candidate else 0,
    "lazyqp_zero_dual_continuation_fallbacks": 2 if candidate else 0,
    "lazyqp_zero_dual_continuation_repairs": 10 if candidate else 0,
    # Aliases of sites already counted above.  Huge values expose any
    # accidental double count in reported_numeric_factorizations().
    "lazyqp_reduced_kkt_factorizations": 10000,
    "lazy_dca_condensed_flow_lift_schur_factorizations": 20000,
    "restoration_minimax_factorizations": 30000,
    "soc_factorizations": 40000,
    "elapsed_seconds": 0.4 if candidate else 4.5,
}
if "--omit-factor-counter" in sys.argv:
    payload.pop("restoration_matrix_free_solves")
if "--omit-elapsed" in sys.argv:
    payload.pop("elapsed_seconds")
if "--negative-elapsed" in sys.argv:
    payload["elapsed_seconds"] = -0.1
if "--negative-dual" in sys.argv:
    payload["dual_infeasibility_inf"] = -1.0
if "--invalid-zero-dual-counters" in sys.argv and candidate:
    payload["lazyqp_zero_dual_continuation_mapped_attempts"] = -1
    payload["lazyqp_zero_dual_continuation_direct_attempts"] = 6
output.write_text(json.dumps(payload))
'''


def write_ipopt_reference(path: Path, model: Path) -> dict:
    digest = sha256_file(model)
    payload = {
        "schema": "lazyipdca-ipopt-same-qcqp-run-v1",
        "model": {
            "path": str(model.resolve()),
            "sha256": digest,
            "bytes": model.stat().st_size,
            "n": 1,
            "equalities": 0,
            "inequalities": 0,
        },
        "model_integrity_hashes": [digest, digest, digest],
        "model_integrity_passed": True,
        "execution_order": ["ipopt", "lazyipdca"],
        "ipopt": {
            "return_code": 0,
            "certificate_pass": True,
            "parse_error": "",
            "result": {
                "status": "solved",
                "objective": 1.0,
                "equality_violation": 0.0,
                "inequality_violation": 0.0,
                "bound_violation": 0.0,
                "dual_infeasibility_inf": 5.0e-8,
                "complementarity_inf": 2.0e-8,
                "model_path": str(model.resolve()),
                "model_sha256": digest,
            },
        },
    }
    path.write_text(json.dumps(payload))
    return payload


with tempfile.TemporaryDirectory() as temporary_name:
    temporary = Path(temporary_name)
    model = temporary / "one.qcq"
    model.write_text(QCQ)
    original_hash = sha256_file(model)
    ipopt_reference = temporary / "ipopt-first.summary.json"
    valid_reference_payload = write_ipopt_reference(ipopt_reference, model)
    solver = temporary / "fake_lazyipdca"
    solver.write_text(FAKE_SOLVER)
    solver.chmod(solver.stat().st_mode | stat.S_IXUSR)

    output = temporary / "benchmark"
    result = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=output,
        repetitions=3,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=shlex.split(DEFAULT_CANDIDATE_ARGS),
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.5,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    assert result["benchmark_passed"], result
    assert result["protocol_passed"], result
    assert result["model_integrity_passed"], result
    assert result["ipopt_first_reference"]["validation_passed"], result
    assert result["ipopt_first_reference"]["recomputed_kkt_pass"], result
    assert result["candidate_feature"]["required"] is True, result
    assert result["candidate_feature"]["engaged_runs"] == 3, result
    assert result["candidate_feature"]["engagement_passed"] is True, result
    assert result["model"]["export_count"] == 0, result
    assert result["model"]["sha256"] == original_hash, result
    assert result["model"]["path"] == str(model.resolve()), result
    assert result["targets"]["historical"]["seconds"] == 10.0, result
    assert result["targets"]["historical"]["passed"] is True, result
    assert result["targets"]["competitiveness"]["seconds"] == 0.5, result
    assert result["targets"]["competitiveness"]["passed"] is True, result
    factor_target = result["targets"][
        "reported_numeric_factorization_budget"]
    assert factor_target["maximum"] == 45, result
    assert factor_target["passed"] is True, result
    assert factor_target["required_for_benchmark_pass"] is False, result
    assert result["baseline"]["outer_iterations_median"] == 49.0, result
    assert result["baseline"]["inner_iterations_median"] == 711.0, result
    assert result["baseline"]["kkt_factorizations_median"] == 653.0, result
    assert result["candidate"]["outer_iterations_median"] == 12.0, result
    assert result["candidate"]["inner_iterations_median"] == 50.0, result
    assert result["candidate"]["kkt_factorizations_median"] == 35.0, result
    # Baseline: 653 + 53 + 61 + (5 - 2) + 4 + 0 = 774.
    # Candidate: 35 + 3 + 2 + (3 - 1) + 2 + 1 = 45.
    # Alias values 10,000 through 40,000 must not enter either sum.
    assert result["baseline"][
        "reported_numeric_factorizations_median"] == 774.0, result
    assert result["candidate"][
        "reported_numeric_factorizations_median"] == 45.0, result
    assert result["candidate"]["solver_seconds_median"] == 0.4, result
    assert result["baseline"]["objective_median"] == 1.0, result
    assert abs(result["candidate"]["objective_median"] - 1.0) < 1.0e-9
    assert result["comparison"][
        "solver_speedup_baseline_over_candidate"] == 11.25, result
    assert reported_numeric_factorizations({
        "kkt_factorizations": 1,
        "terminal_kkt_attempts": 1,
        "terminal_kkt_iterations": 0,
    }) is None
    assert reported_numeric_factorizations({
        "kkt_factorizations": 1,
        "terminal_kkt_attempts": 0,
        "terminal_kkt_iterations": 0,
    }) is None
    complete_counts = {
        "kkt_factorizations": 1,
        "lazy_dca_condensed_factorizations": 1,
        "multiplier_row_space_factorizations": 1,
        "restoration_harmonic_factorizations": 1,
        "lazy_dca_condensed_tangent_inertia_attempts": 0,
        "restoration_linear_solves": 1,
        "restoration_matrix_free_solves": 0,
        "terminal_kkt_attempts": 0,
        "terminal_kkt_iterations": 0,
    }
    assert reported_numeric_factorizations(complete_counts) == 5
    assert reported_numeric_factorizations({
        **complete_counts,
        "kkt_factorizations": -1,
    }) is None
    assert reported_numeric_factorizations({
        **complete_counts,
        "restoration_matrix_free_solves": -1,
    }) is None
    assert reported_numeric_factorizations({
        **complete_counts,
        "kkt_factorizations": 1.5,
    }) is None
    assert sha256_file(model) == original_hash
    stored = json.loads((output / "benchmark.json").read_text())
    assert stored["rows"][0]["baseline_kkt_pass"] is True, stored
    assert stored["rows"][0]["candidate_kkt_pass"] is True, stored
    assert stored["rows"][0]["objective_match"] is True, stored
    assert stored["rows"][1]["execution_order"] == "candidate-first", stored
    expected_model_arg = str(model.resolve())
    for label in ("baseline", "candidate"):
        command = json.loads((output / "runs" / "run_001" /
                              "summary.json").read_text())[label]["command"]
        assert command[command.index("--problem") + 1] == expected_model_arg

    # The new 0.5 s goal is deliberately independent of the historical 10 s
    # acceptance gate: missing 0.5 s must be visible without rewriting the
    # historical benchmark result.
    separated = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=temporary / "separate_targets",
        repetitions=1,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=shlex.split(DEFAULT_CANDIDATE_ARGS),
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.3,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    assert separated["targets"]["historical"]["passed"] is True, separated
    assert separated["targets"]["competitiveness"]["passed"] is False, separated
    assert separated["benchmark_passed"] is True, separated

    # Missing only the informational factorization budget must likewise leave
    # the historical benchmark gate untouched.
    factorization_miss = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=temporary / "factorization_budget_miss",
        repetitions=1,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=shlex.split(DEFAULT_CANDIDATE_ARGS),
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.5,
        factorization_budget=44,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    assert factorization_miss["targets"]["historical"]["passed"] is True
    assert factorization_miss["targets"]["competitiveness"]["passed"] is True
    assert factorization_miss["targets"][
        "reported_numeric_factorization_budget"]["passed"] is False
    assert factorization_miss["benchmark_passed"] is True

    factorization_unavailable = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=temporary / "factorization_unavailable",
        repetitions=1,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=[
            *shlex.split(DEFAULT_CANDIDATE_ARGS),
            "--omit-factor-counter",
        ],
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.5,
        factorization_budget=45,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    unavailable_target = factorization_unavailable["targets"][
        "reported_numeric_factorization_budget"]
    assert unavailable_target["available_runs"] == 0
    assert unavailable_target["passed"] is False
    assert factorization_unavailable["benchmark_passed"] is True

    timing_unavailable = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=temporary / "timing_unavailable",
        repetitions=1,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=[
            *shlex.split(DEFAULT_CANDIDATE_ARGS),
            "--omit-elapsed",
        ],
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.5,
        factorization_budget=45,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    assert timing_unavailable["targets"]["historical"][
        "candidate_available_runs"] == 0
    assert timing_unavailable["targets"]["historical"]["passed"] is False
    assert timing_unavailable["protocol_passed"] is False
    assert timing_unavailable["benchmark_passed"] is False

    negative_timing = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=temporary / "negative_timing",
        repetitions=1,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=[
            *shlex.split(DEFAULT_CANDIDATE_ARGS),
            "--negative-elapsed",
        ],
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.5,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    assert negative_timing["targets"]["historical"][
        "candidate_available_runs"] == 0
    assert negative_timing["targets"]["historical"]["passed"] is False
    assert negative_timing["protocol_passed"] is False

    negative_kkt = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=temporary / "negative_kkt",
        repetitions=1,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=[
            *shlex.split(DEFAULT_CANDIDATE_ARGS),
            "--negative-dual",
        ],
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.5,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    assert negative_kkt["rows"][0]["candidate_kkt_pass"] is False
    assert negative_kkt["rows"][0]["pair_passed"] is False
    assert negative_kkt["protocol_passed"] is False

    invalid_feature_counters = benchmark(
        model=model,
        binary=solver,
        ipopt_reference_summary=ipopt_reference,
        output_dir=temporary / "invalid_feature_counters",
        repetitions=1,
        baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
        candidate_args=[
            *shlex.split(DEFAULT_CANDIDATE_ARGS),
            "--invalid-zero-dual-counters",
        ],
        historical_target_seconds=10.0,
        competitiveness_target_seconds=0.5,
        linear_backend="native",
        phase1_threads=1,
        max_outer=100,
        max_inner=100,
        time_limit=10.0,
    )
    assert invalid_feature_counters["candidate_feature"][
        "engaged_runs"] == 0
    assert invalid_feature_counters["candidate_feature"][
        "engagement_passed"] is False
    assert invalid_feature_counters["protocol_passed"] is False

    validation_options = {
        "equality_tolerance": 1.0e-6,
        "inequality_tolerance": 1.0e-6,
        "bound_tolerance": 1.0e-6,
        "dual_tolerance": 1.0e-6,
        "complementarity_tolerance": 1.0e-6,
    }

    def rejected_reference(name, edit):
        payload = json.loads(json.dumps(valid_reference_payload))
        edit(payload)
        path = temporary / f"invalid-{name}.summary.json"
        path.write_text(json.dumps(payload))
        validation = validate_ipopt_reference_summary(
            path, model, original_hash, **validation_options)
        assert validation["validation_passed"] is False, validation
        assert validation["errors"], validation

    missing_validation = validate_ipopt_reference_summary(
        temporary / "missing-ipopt.summary.json",
        model,
        original_hash,
        **validation_options,
    )
    assert missing_validation["validation_passed"] is False
    rejected_reference(
        "hash", lambda data: data["model"].__setitem__("sha256", "0" * 64))
    rejected_reference(
        "path", lambda data: data["model"].__setitem__(
            "path", str(temporary / "different.qcq")))
    rejected_reference(
        "certificate", lambda data: data["ipopt"].__setitem__(
            "certificate_pass", False))
    rejected_reference(
        "return-code", lambda data: data["ipopt"].__setitem__(
            "return_code", 1))
    rejected_reference(
        "recomputed-kkt", lambda data: data["ipopt"]["result"].__setitem__(
            "dual_infeasibility_inf", 1.0e-2))
    rejected_reference(
        "order", lambda data: data.__setitem__(
            "execution_order", ["lazyipdca", "ipopt"]))

    rejected_output = temporary / "missing_reference_benchmark"
    try:
        benchmark(
            model=model,
            binary=solver,
            ipopt_reference_summary=temporary / "missing-ipopt.summary.json",
            output_dir=rejected_output,
            repetitions=1,
            baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
            candidate_args=shlex.split(DEFAULT_CANDIDATE_ARGS),
        )
        raise AssertionError("a missing Ipopt-first reference was not rejected")
    except ValueError as exc:
        assert "invalid Ipopt-first reference summary" in str(exc), exc
    assert not rejected_output.exists()

    mutable_model = temporary / "mutable.qcq"
    mutable_model.write_text(QCQ)
    mutable_reference = temporary / "mutable-ipopt-first.summary.json"
    write_ipopt_reference(mutable_reference, mutable_model)
    try:
        benchmark(
            model=mutable_model,
            binary=solver,
            ipopt_reference_summary=mutable_reference,
            output_dir=temporary / "mutation_benchmark",
            repetitions=1,
            baseline_args=shlex.split(DEFAULT_BASELINE_ARGS),
            candidate_args=[
                *shlex.split(DEFAULT_CANDIDATE_ARGS),
                "--mutate-model",
            ],
            linear_backend="native",
            phase1_threads=1,
            max_outer=10,
            max_inner=10,
            time_limit=10.0,
        )
        raise AssertionError("a QCQP mutation was not rejected")
    except RuntimeError as exc:
        assert "modified the shared QCQP" in str(exc), exc

print("[PASS] r3p24 strict/candidate benchmark preserves and measures one QCQP")
