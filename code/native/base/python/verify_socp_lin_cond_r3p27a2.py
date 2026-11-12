#!/usr/bin/env python3
"""Fail-closed verification of the r3p27a2 SOCP LIN/COND pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

VERSION = "0.4.28r3p28-lazy-newton"
BUILD_ID = "v0.4.28r3p28-lazy-newton-api-4313"
TOLERANCE = 1e-6


def finite_below(value: object, limit: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= float(value) <= limit
    )


def verify(path: Path, route: str) -> dict:
    result = json.loads(path.read_text())
    if result.get("version") != VERSION or result.get("build_id") != BUILD_ID:
        raise ValueError(f"{path}: stale solver identity")
    if result.get("api_version") != 4313 or result.get("status") != "solved":
        raise ValueError(f"{path}: solver did not return an API-4313 solved result")
    if result.get("linear_solver_backend", "").split("/", 1)[0] != "socp":
        raise ValueError(f"{path}: numerical route is not SOCP")
    if result.get("lazy_dca_equality_model") != route:
        raise ValueError(f"{path}: wrong equality route")
    if result.get("socp_formulation_builds", 0) < 1:
        raise ValueError(f"{path}: conic formulation was never built")
    if result.get("socp_formulation_audit_failures") != 0:
        raise ValueError(f"{path}: conic audit failed")
    if not finite_below(result.get("socp_max_rotated_cone_violation"), TOLERANCE):
        raise ValueError(f"{path}: rotated-cone violation exceeds tolerance")
    for key in (
        "equality_violation",
        "inequality_violation",
        "bound_violation",
        "dual_infeasibility_inf",
        "complementarity_inf",
    ):
        if not finite_below(result.get(key), TOLERANCE):
            raise ValueError(f"{path}: {key} exceeds tolerance")
    if result.get("termination_multiplier_source") in (None, "", "unavailable"):
        raise ValueError(f"{path}: missing original-QCQP multiplier certificate")
    if route == "linearized":
        if result.get("lazy_dca_linearized_equality_steps", 0) < 1:
            raise ValueError(f"{path}: LIN affine equality route was not exercised")
        if result.get("lazy_dca_condensed_retraction_attempts", 0) != 0:
            raise ValueError(f"{path}: LIN unexpectedly used COND retraction")
    else:
        if result.get("lazy_dca_condensed_retraction_attempts", 0) < 1:
            raise ValueError(f"{path}: COND retraction was not attempted")
        if result.get("lazy_dca_condensed_retraction_acceptances", 0) < 1:
            raise ValueError(f"{path}: COND retraction was never accepted")
    return {
        "path": str(path.resolve()),
        "status": result["status"],
        "equality_model": route,
        "objective": result.get("objective"),
        "solver_core_seconds": result.get("elapsed_seconds"),
        "outer_iterations": result.get("outer_iterations"),
        "kkt_factorizations": result.get("kkt_factorizations"),
        "socp_formulation_builds": result.get("socp_formulation_builds"),
        "socp_max_rotated_cone_violation": result.get(
            "socp_max_rotated_cone_violation"
        ),
        "termination_multiplier_source": result.get(
            "termination_multiplier_source"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lin", type=Path, required=True)
    parser.add_argument("--cond", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lin = verify(args.lin, "linearized")
    cond = verify(args.cond, "condensed-retraction")
    objective_scale = max(
        1.0, abs(float(lin["objective"])), abs(float(cond["objective"]))
    )
    objective_difference = abs(float(lin["objective"]) - float(cond["objective"]))
    if objective_difference > 1e-6 * objective_scale:
        raise ValueError("LIN and COND objectives are not equivalent at the requested tolerance")

    summary = {
        "schema": "lazyipdca-socp-lin-cond-r3p27a2-v1",
        "version": VERSION,
        "build_id": BUILD_ID,
        "model_sha256": args.model_sha256,
        "validation_passed": True,
        "objective_difference": objective_difference,
        "runs": {"lin": lin, "cond": cond},
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
