#!/usr/bin/env python3
"""Contract tests for the standalone r3p27 exact-case118 A/B verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_r3p27_ab", ROOT / "python" / "verify_r3p27_ab.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(mode: str) -> dict:
    record = {
        "schema": "lazyipdca-qcqp-result-v2",
        "api_version": 4310,
        "version": "0.4.28r3p27-single-loop-hybrid-kkt",
        "build_id": "v0.4.28r3p27-single-loop-hybrid-kkt-api-4310",
        "status": "solved",
        "ipopt_desired_convergence": True,
        "objective": 10.0,
        "elapsed_seconds": 2.9 if mode == "reduced" else 2.2,
        "dca_single_loop_attempts": 4,
        "dca_single_loop_acceptances": 4,
        "dca_single_loop_inner_steps": 4,
        "dca_single_loop_factorizations": 4,
        "kkt_factorizations": 4,
        "kkt_regularization_retries": 0,
        "linear_solver_backend": "lazyqp/suitesparse_ldl_metis_persistent_symbolic",
        "dca_single_loop_strict_fallbacks": 0,
        "dca_single_loop_state_transport_acceptances": 3,
        "dca_single_loop_keyed_transport_acceptances": 3,
        "inner_iteration_limit_fallback_acceptances": 0,
        "lazyqp_affine_elimination_fallbacks": 0,
        "lazyqp_reduced_kkt_factorizations": 4,
        "lazyqp_reduced_kkt_fallbacks": 0,
        "lazyqp_full_kkt_factorizations": 0,
        "lazy_dca_corrector_attempts": 0,
        "lazy_dca_iterated_soc_attempts": 0,
        "soc_attempts": 0,
        "lazyqp_numeric_kkt_reuse_attempts": 0,
        "lazyqp_numeric_kkt_reuse_acceptances": 0,
        "lazyqp_numeric_kkt_reuse_fallbacks": 0,
        "lazyqp_numeric_kkt_reuse_iterations": 0,
        "dca_single_loop_full_refit_calls": 1,
        "dca_single_loop_bootstrap_full_fits": 0,
        "dca_single_loop_reduced_bootstrap_attempts": 1,
        "dca_single_loop_reduced_bootstrap_acceptances": 1,
        "dca_single_loop_baseline_proximal_regularization": 0.05,
        "dca_single_loop_refits_skipped": 3,
        "dca_single_loop_refit_fallbacks": 0,
        "multiplier_refit_fallback_reuses": 3,
        "termination_multiplier_fit_converged": True,
        "termination_multiplier_source": "fit",
        "dca_single_loop_retractions_skipped": 3,
        "lazy_dca_max_equality_retraction_inf": 0.0,
        "kkt_symbolic_analyses": 1,
        "kkt_symbolic_reuses": 3,
        "lazyqp_kkt_triplet_layout_builds": 1,
        "lazyqp_kkt_triplet_layout_reuses": 3,
        "kkt_numeric_scatter_builds": 1,
        "kkt_numeric_scatter_reuses": 3,
        "lazyqp_zero_dual_continuation_attempts": 2,
        "lazyqp_zero_dual_continuation_mapped_attempts": 2,
        "lazyqp_zero_dual_continuation_direct_attempts": 0,
        "lazyqp_zero_dual_continuation_acceptances": 1,
        "lazyqp_zero_dual_continuation_fallbacks": 1,
        "restoration_iterations": 3,
        "restoration_linear_solves": 3,
        "lazyqp_model_support_fastpath_requests": 4,
        "lazyqp_model_support_fastpath_effective_subproblems": 4,
        "lazyqp_model_support_builds": 4,
        "lazyqp_model_support_eligible_models": 4,
        "lazyqp_model_support_generic_fallback_models": 0,
        "lazyqp_model_support_value_evaluations": 12,
        "lazyqp_model_support_gradient_evaluations": 12,
        "lazyqp_max_kkt_dimension": 628 if mode == "reduced" else 630,
    }
    if mode == "reduced":
        record.update({
            "lazyqp_hybrid_kkt_attempts": 0,
            "lazyqp_hybrid_kkt_factorizations": 0,
            "lazyqp_hybrid_kkt_fallbacks": 0,
            "lazyqp_hybrid_kkt_retained_rows_total": 0,
            "lazyqp_hybrid_kkt_retained_rows_max": 0,
            "lazyqp_hybrid_kkt_outer_product_triplets_avoided": 0,
            "lazyqp_hybrid_kkt_border_triplets_emitted": 0,
        })
    else:
        record.update({
            "lazyqp_hybrid_kkt_attempts": 4,
            "lazyqp_hybrid_kkt_factorizations": 4,
            "lazyqp_hybrid_kkt_fallbacks": 0,
            "lazyqp_hybrid_kkt_retained_rows_total": 8,
            "lazyqp_hybrid_kkt_retained_rows_max": 2,
            "lazyqp_hybrid_kkt_outer_product_triplets_avoided": 10000,
            "lazyqp_hybrid_kkt_border_triplets_emitted": 400,
        })
    for key in MODULE._BASE.RESIDUAL_KEYS:
        record[key] = 1e-8
    for key in MODULE._BASE.FORCING_ZERO_KEYS + MODULE._BASE.PATH_REFIT_ZERO_KEYS:
        record[key] = 0
    for key in MODULE._BASE.RETRACTION_ZERO_KEYS:
        record[key] = 0
    for key in MODULE._BASE.DIRECT_START_ZERO_KEYS:
        record[key] = 0
    for key in MODULE._BASE.TERMINAL_KKT_ZERO_KEYS + MODULE._BASE.EQUALITY_GMRES_ZERO_KEYS:
        record[key] = 0
    return record


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_campaign(root: Path) -> tuple[dict, dict[str, dict]]:
    qcq = root / "case118.qcq"
    qcq.write_bytes(b"one immutable serialized QCQP\n")
    digest = hashlib.sha256(qcq.read_bytes()).hexdigest()
    ipopt = {
        "schema": "lazyipdca-qcqp-ipopt-direct-result-v1",
        "model_sha256": digest,
        "success": True,
        "kkt_passed": True,
        "objective": 10.0,
        "solver_seconds": 0.3,
        "iterations": 22,
        "dimensions": {
            "variables": 1088,
            "equalities": 980,
            "inequalities": 979,
        },
        "residuals": {
            "equality_inf": 1e-9,
            "inequality_inf": 0.0,
            "bound_inf": 0.0,
            "dual_inf": 1e-8,
            "complementarity_inf": 1e-8,
            "dual_sign_inf": 0.0,
        },
    }
    ipopt_path = root / "ipopt.json"
    write_json(ipopt_path, ipopt)
    ipopt_hashes = root / "ipopt_hashes.tsv"
    ipopt_hashes.write_text(f"ipopt\t0\t{digest}\t{digest}\n")
    warmup_hashes = root / "warmups.tsv"
    warmup_hashes.write_text(
        f"reduced\t1\t0\t{digest}\t{digest}\n"
        f"hybrid\t1\t0\t{digest}\t{digest}\n"
    )
    schedule = root / "schedule.tsv"
    rows = []
    sequence = 0
    for pair in range(1, 6):
        order = ("reduced", "hybrid") if pair % 2 else ("hybrid", "reduced")
        for mode in order:
            sequence += 1
            rows.append(f"{sequence}\t{pair}\t{mode}")
    schedule.write_text("\n".join(rows) + "\n")
    optimizer_order = root / "optimizer_order.tsv"
    order_rows = ["1\tipopt\tipopt\t0"]
    order_sequence = 1
    for index in range(1, 2):
        for mode in ("reduced", "hybrid"):
            order_sequence += 1
            order_rows.append(
                f"{order_sequence}\twarmup\t{mode}\t{index}")
    for pair in range(1, 6):
        order = ("reduced", "hybrid") if pair % 2 else ("hybrid", "reduced")
        for mode in order:
            order_sequence += 1
            order_rows.append(
                f"{order_sequence}\tmeasured\t{mode}\t{pair}")
    optimizer_order.write_text("\n".join(order_rows) + "\n")

    templates: dict[str, dict] = {}
    dirs: dict[str, Path] = {}
    for mode in ("reduced", "hybrid"):
        mode_dir = root / mode
        mode_dir.mkdir()
        dirs[mode] = mode_dir
        template = candidate(mode)
        templates[mode] = template
        hash_rows = []
        for index in range(1, 6):
            write_json(mode_dir / f"run_{index}.json", template)
            hash_rows.append(f"{index}\t0\t{digest}\t{digest}")
        (mode_dir / "model_hashes.tsv").write_text(
            "\n".join(hash_rows) + "\n")

    kwargs = {
        "qcq": qcq,
        "ipopt_json": ipopt_path,
        "ipopt_hashes": ipopt_hashes,
        "reduced_dir": dirs["reduced"],
        "hybrid_dir": dirs["hybrid"],
        "warmup_hashes": warmup_hashes,
        "warmups": 1,
        "schedule": schedule,
        "optimizer_order": optimizer_order,
        "expected_runs": 5,
        "tolerance": 1e-6,
        "target_seconds": 0.5,
        "objective_absolute_tolerance": 1e-3,
        "objective_relative_tolerance": 1e-8,
        "require_zero_dual": True,
        "require_model_support_fastpath": True,
    }
    return kwargs, templates


def expect_failure(kwargs: dict, path: Path, changed: dict) -> None:
    original = json.loads(path.read_text(encoding="utf-8"))
    write_json(path, changed)
    try:
        MODULE.validate_campaign(**kwargs)
    except MODULE.VerificationError:
        pass
    else:
        raise AssertionError("verifier accepted an invalid campaign")
    finally:
        write_json(path, original)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="r3p27-ab-verifier-") as temporary:
        kwargs, templates = build_campaign(Path(temporary))
        summary = MODULE.validate_campaign(**kwargs)
        assert summary["validation_passed"] is True
        assert summary["runs_passed"] == {"reduced": 5, "hybrid": 5}
        assert summary["hybrid_faster"] is True
        assert summary["target_passed"] is False
        assert summary["max_kkt_dimension_deltas"] == [2] * 5

        hybrid_path = kwargs["hybrid_dir"] / "run_1.json"
        broken = copy.deepcopy(templates["hybrid"])
        broken["lazyqp_hybrid_kkt_factorizations"] = 0
        expect_failure(kwargs, hybrid_path, broken)

        broken = copy.deepcopy(templates["hybrid"])
        broken["lazyqp_hybrid_kkt_retained_rows_max"] = 1
        expect_failure(kwargs, hybrid_path, broken)

        broken = copy.deepcopy(templates["hybrid"])
        broken["lazyqp_model_support_fastpath_requests"] = 0
        expect_failure(kwargs, hybrid_path, broken)

        reduced_path = kwargs["reduced_dir"] / "run_1.json"
        broken = copy.deepcopy(templates["reduced"])
        broken["lazyqp_hybrid_kkt_attempts"] = 1
        expect_failure(kwargs, reduced_path, broken)

        # A slower hybrid run is a valid measurement, not a validation error.
        for index in range(1, 6):
            path = kwargs["hybrid_dir"] / f"run_{index}.json"
            slower = copy.deepcopy(templates["hybrid"])
            slower["elapsed_seconds"] = 3.5
            write_json(path, slower)
        slower_summary = MODULE.validate_campaign(**kwargs)
        assert slower_summary["validation_passed"] is True
        assert slower_summary["hybrid_faster"] is False
        assert slower_summary["target_passed"] is False

        # Restore results, then prove that alternating paired order is gated.
        for index in range(1, 6):
            write_json(kwargs["hybrid_dir"] / f"run_{index}.json",
                       templates["hybrid"])
        schedule_path = kwargs["schedule"]
        original_schedule = schedule_path.read_text(encoding="utf-8")
        schedule_path.write_text(original_schedule.replace(
            "1\t1\treduced", "1\t1\thybrid", 1), encoding="utf-8")
        try:
            MODULE.validate_campaign(**kwargs)
        except MODULE.VerificationError:
            pass
        else:
            raise AssertionError("verifier accepted a non-alternating schedule")

    print("[PASS] r3p27 exact-case118 paired A/B verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
