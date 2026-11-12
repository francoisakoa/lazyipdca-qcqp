#!/usr/bin/env python3
"""Contract tests for the standalone r3p26 campaign verifier."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_r3p26_candidate", ROOT / "python" / "verify_r3p26_candidate.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate() -> dict:
    record = {
        "schema": "lazyipdca-qcqp-result-v2",
        "api_version": 4309,
        "version": "0.4.28r3p26-single-loop-corrected",
        "build_id": "v0.4.28r3p26-single-loop-corrected-api-4309",
        "status": "solved",
        "ipopt_desired_convergence": True,
        "objective": 10.0,
        "elapsed_seconds": 0.4,
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
    }
    for key in MODULE.RESIDUAL_KEYS:
        record[key] = 1e-8
    for key in MODULE.FORCING_ZERO_KEYS + MODULE.PATH_REFIT_ZERO_KEYS:
        record[key] = 0
    for key in MODULE.RETRACTION_ZERO_KEYS:
        record[key] = 0
    for key in MODULE.DIRECT_START_REJECTION_KEYS:
        record[key] = 0
    for key in MODULE.DIRECT_START_RECOVERY_REJECTION_KEYS:
        record.setdefault(key, 0)
    for key in MODULE.DIRECT_START_ZERO_KEYS:
        record[key] = 0
    for key in MODULE.TERMINAL_KKT_ZERO_KEYS + MODULE.EQUALITY_GMRES_ZERO_KEYS:
        record[key] = 0
    return record


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_campaign(root: Path) -> tuple[dict, dict]:
    qcq = root / "case118.qcq"
    qcq.write_bytes(b"one immutable serialized QCQP\n")
    digest = hashlib.sha256(qcq.read_bytes()).hexdigest()
    ipopt = {
        "model_sha256": digest,
        "success": True,
        "kkt_passed": True,
        "objective": 10.0,
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
    run_dir = root / "candidate"
    run_dir.mkdir()
    template = candidate()
    rows = []
    for index in range(1, 6):
        write_json(run_dir / f"run_{index}.json", template)
        rows.append(f"{index}\t0\t{digest}\t{digest}")
    (run_dir / "model_hashes.tsv").write_text("\n".join(rows) + "\n")
    kwargs = {
        "qcq": qcq,
        "ipopt_json": ipopt_path,
        "run_dir": run_dir,
        "expected_runs": 5,
        "tolerance": 1e-6,
        "target_seconds": 0.5,
        "objective_absolute_tolerance": 1e-3,
        "objective_relative_tolerance": 1e-8,
        "require_zero_dual": True,
    }
    return kwargs, template


def expect_failure(kwargs: dict, run_dir: Path, key: str, value: int) -> None:
    path = run_dir / "run_1.json"
    changed = copy.deepcopy(candidate())
    changed[key] = value
    write_json(path, changed)
    try:
        MODULE.validate_campaign(**kwargs)
    except MODULE.VerificationError:
        return
    raise AssertionError(f"verifier accepted invalid {key}={value}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="r3p26-verifier-") as temporary:
        kwargs, template = build_campaign(Path(temporary))
        summary = MODULE.validate_campaign(**kwargs)
        assert summary["runs_passed"] == 5
        assert summary["target_passed"] is True
        run_dir = kwargs["run_dir"]

        expect_failure(kwargs, run_dir, "dca_single_loop_strict_fallbacks", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir, "multiplier_refit_attempts", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir, "dca_single_loop_refit_fallbacks", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir, "dca_single_loop_bootstrap_full_fits", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(
            kwargs, run_dir, "dca_single_loop_reduced_bootstrap_attempts", 0)
        write_json(run_dir / "run_1.json", template)
        expect_failure(
            kwargs, run_dir, "dca_single_loop_reduced_bootstrap_acceptances", 0)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir, "dca_single_loop_inner_steps", 3)
        write_json(run_dir / "run_1.json", template)
        no_backend_scatter = copy.deepcopy(template)
        no_backend_scatter["kkt_numeric_scatter_builds"] = 0
        no_backend_scatter["kkt_numeric_scatter_reuses"] = 0
        write_json(run_dir / "run_1.json", no_backend_scatter)
        summary = MODULE.validate_campaign(**kwargs)
        assert summary["runs_passed"] == 5
        write_json(run_dir / "run_1.json", template)
        broken = copy.deepcopy(template)
        broken["dca_single_loop_baseline_proximal_regularization"] = 0.0
        write_json(run_dir / "run_1.json", broken)
        try:
            MODULE.validate_campaign(**kwargs)
        except MODULE.VerificationError:
            pass
        else:
            raise AssertionError(
                "verifier accepted a missing single-loop baseline curvature")
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir, "dca_single_loop_full_refit_calls", 0)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir, "lazy_dca_equality_retraction_attempts", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir,
                       "lazyqp_inexact_forcing_model_evaluations", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir,
                       "dca_single_loop_direct_start_strict_latches", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(kwargs, run_dir, "terminal_kkt_attempts", 1)
        write_json(run_dir / "run_1.json", template)
        expect_failure(
            kwargs, run_dir, "equality_qp_original_kkt_gmres_attempts", 1)

    print("[PASS] r3p26 exact-QCQP campaign telemetry verifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
