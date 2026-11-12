#!/usr/bin/env python3
"""Export one PGLib/MATPOWER case once for a same-QCQP comparison.

r3p23 deliberately reuses the audited r3p22 algebra and serialization.  This
module adds provenance, an atomic write, and a SHA-256 identity contract; it
does not invoke either optimizer.  Consequently Ipopt and LazyIPDCA can only
be compared after both have consumed the very same immutable ``.qcq`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from export_pglib_qcqp_r3p22 import (  # exact, trajectory-preserving algebra
    BOUND_INFINITY,
    build_model,
    parse_matpower,
    validate,
    write_qcq,
)

EXPORT_SCHEMA = "lazyipdca-pglib-qcqp-export-v1"
EXPORT_REVISION = "r3p23-same-qcqp"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_native_value(value: Any) -> Any:
    """Recursively convert NumPy containers and scalars to JSON types.

    PGLib assembly intentionally uses NumPy throughout.  Even apparently
    scalar reductions such as ``sum(np.bool_(...))`` can therefore return a
    NumPy integer.  Normalize the complete payload at the serialization
    boundary instead of maintaining a fragile list of individual fields.
    """
    if isinstance(value, np.ndarray):
        return json_native_value(value.tolist())
    if isinstance(value, np.generic):
        return json_native_value(value.item())
    if isinstance(value, dict):
        normalized: dict[Any, Any] = {}
        for key, item in value.items():
            native_key = json_native_value(key)
            if not isinstance(native_key, (str, int, float, bool, type(None))):
                raise TypeError(
                    "JSON object key is not scalar after NumPy normalization: "
                    f"{type(native_key).__name__}")
            normalized[native_key] = json_native_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [json_native_value(item) for item in value]
    return value


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    normalized = json_native_value(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_case(
    case_path: Path,
    output: Path,
    *,
    flow_limit: str = "apparent",
    angle_limits: bool = True,
    power_flow_start: bool = True,
    power_flow_voltage_continuation: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Write and fingerprint the exact rectangular/lifted QCQP."""
    case_path = case_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if not case_path.is_file():
        raise FileNotFoundError(f"MATPOWER/PGLib case is missing: {case_path}")
    if flow_limit not in {"current", "apparent", "none"}:
        raise ValueError(f"unsupported flow limit: {flow_limit}")
    if output.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing QCQP without --force: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    case = parse_matpower(case_path)
    model = build_model(
        case,
        flow_limit=flow_limit,
        angle_limits=angle_limits,
        power_flow_start=power_flow_start,
        power_flow_voltage_continuation=power_flow_voltage_continuation,
    )
    (n, x0, lb, ub, objective, equalities, inequalities, _Y, _bus, _gen,
     _branch, _online, _base_mva, _bus_index, metadata, _gencost, _branches,
     _lift) = model
    metadata = dict(metadata)
    metadata.update(validate(model))

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".qcq", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_qcq(
            temporary, case_path.stem, n, x0, lb, ub, objective,
            equalities, inequalities)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    payload: dict[str, Any] = {
        "schema": EXPORT_SCHEMA,
        "export_revision": EXPORT_REVISION,
        "source_case": str(case_path),
        "source_case_sha256": sha256_file(case_path),
        "model": str(output),
        "model_sha256": sha256_file(output),
        "model_bytes": output.stat().st_size,
        "n": n,
        "equalities": len(equalities),
        "inequalities": len(inequalities),
        "flow_limit": flow_limit,
        "angle_limits": angle_limits,
        "power_flow_start": power_flow_start,
        "power_flow_voltage_continuation":
            power_flow_voltage_continuation,
        "finite_lower_bounds": sum(value > -BOUND_INFINITY for value in lb),
        "finite_upper_bounds": sum(value < BOUND_INFINITY for value in ub),
        "fixed_variables": sum(
            lower > -BOUND_INFINITY and upper < BOUND_INFINITY and
            lower == upper for lower, upper in zip(lb, ub)),
        "export_validation": metadata,
    }
    return json_native_value(payload)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Export one immutable QCQP for the r3p23 same-model test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("case", type=Path, help="MATPOWER/PGLib .m case")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None,
                    help="export manifest (default: OUTPUT.export.json)")
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
    ap.add_argument("--force", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = export_case(
            args.case, args.output,
            flow_limit=args.flow_limit,
            angle_limits=not args.no_angle_limits,
            power_flow_start=args.power_flow_start,
            power_flow_voltage_continuation=
                args.power_flow_voltage_continuation,
            force=args.force,
        )
        manifest = (args.json or
                    args.output.with_suffix(".export.json")).resolve()
        write_json_atomic(manifest, payload)
        payload["manifest"] = str(manifest)
        print(json.dumps(json_native_value(payload), indent=2,
                         sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[EXPORT FAILURE] {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
