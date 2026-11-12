#!/usr/bin/env python3
"""Run Ipopt-direct and LazyIPDCA on one identical exported QCQP file."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence

from export_pglib_qcqp_r3p22 import build_r3p22_solver_command
from export_pglib_qcqp_r3p23 import sha256_file, write_json_atomic

PAIR_SCHEMA = "lazyipdca-ipopt-same-qcqp-run-v1"


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def qcq_dimensions(path: Path) -> dict[str, int]:
    """Read only the dimension markers while validating the file envelope."""
    magic = False
    ended = False
    dimensions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number == 1:
                magic = line.strip() == "LAZYIPDCA_QCQP_V1"
            if line.startswith("n ") and "n" not in dimensions:
                dimensions["n"] = int(line.split()[1])
            elif line.startswith("equalities "):
                dimensions["equalities"] = int(line.split()[1])
            elif line.startswith("inequalities "):
                dimensions["inequalities"] = int(line.split()[1])
            elif line.strip() == "end_problem":
                ended = True
    if not magic or not ended or set(dimensions) != {
            "n", "equalities", "inequalities"}:
        raise ValueError(f"invalid or incomplete QCQP file: {path}")
    return dimensions


def ipopt_interface(binary: Path, requested: str = "auto") -> str:
    if requested not in {"auto", "native", "casadi"}:
        raise ValueError(f"unsupported Ipopt interface: {requested}")
    if requested != "auto":
        return requested
    name = binary.name.lower()
    if "casadi" in name or "run_exact_qcq_ipopt" in name:
        return "casadi"
    return "native"


def binary_version(binary: Path, *, interface: str = "native") -> str:
    if interface == "casadi":
        return "r3p23 CasADi/Ipopt exact-QCQP runner"
    completed = subprocess.run(
        [str(binary), "--version"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = completed.stdout.strip()
    return text if text else f"unavailable (return code {completed.returncode})"


def run_command(
    command: Sequence[str],
    log_path: Path,
    *,
    live_output: bool = False,
) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if live_output:
                print(line, end="", flush=True)
        return_code = process.wait()
    return return_code, time.perf_counter() - started


def load_result(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}, "result JSON is not an object"
        return value, ""
    except (OSError, json.JSONDecodeError) as exc:
        return {}, str(exc)


def normalize_ipopt_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize both native-TNLP and CasADi exact-QCQP JSON schemas."""
    if not isinstance(result.get("residuals"), dict):
        return result
    residuals = result["residuals"]
    dimensions = (result.get("dimensions")
                  if isinstance(result.get("dimensions"), dict) else {})
    normalized = dict(result)
    normalized["ipopt_return_status"] = result.get("status")
    successful = bool(result.get("success", False))
    certificate = bool(result.get("kkt_passed", False))
    normalized["status"] = (
        "solved" if successful and certificate else
        "kkt_failure" if successful else "ipopt_failure")
    aliases = {
        "equality_violation": "equality_inf",
        "inequality_violation": "inequality_inf",
        "bound_violation": "bound_inf",
        "dual_infeasibility_inf": "dual_inf",
        "complementarity_inf": "complementarity_inf",
        "scaled_nlp_error": "scaled_nlp_error",
    }
    for destination, source in aliases.items():
        if destination not in normalized:
            normalized[destination] = residuals.get(source)
    normalized.setdefault("elapsed_seconds", result.get("solver_seconds"))
    normalized.setdefault("ipopt_desired_convergence", certificate)
    normalized.setdefault("n", dimensions.get("variables"))
    normalized.setdefault("equalities", dimensions.get("equalities"))
    normalized.setdefault("inequalities", dimensions.get("inequalities"))
    normalized.setdefault(
        "jacobian_nnz", dimensions.get("jacobian_structural_nnz"))
    normalized.setdefault(
        "lagrangian_hessian_nnz",
        dimensions.get("lagrangian_hessian_structural_nnz"))
    return normalized


def kkt_pass(
    result: dict[str, Any],
    *,
    equality_tolerance: float,
    inequality_tolerance: float,
    bound_tolerance: float,
    dual_tolerance: float,
    complementarity_tolerance: float,
) -> bool:
    values = {
        "equality": finite_float(result.get("equality_violation")),
        "inequality": finite_float(result.get("inequality_violation")),
        "bound": finite_float(result.get("bound_violation")),
        "dual": finite_float(result.get("dual_infeasibility_inf")),
        "complementarity": finite_float(result.get("complementarity_inf")),
    }
    return (
        str(result.get("status")) == "solved" and
        values["equality"] is not None and
        values["equality"] <= equality_tolerance and
        values["inequality"] is not None and
        values["inequality"] <= inequality_tolerance and
        values["bound"] is not None and
        values["bound"] <= bound_tolerance and
        values["dual"] is not None and
        values["dual"] <= dual_tolerance and
        values["complementarity"] is not None and
        values["complementarity"] <= complementarity_tolerance
    )


def lazy_command(
    binary: Path,
    model: Path,
    result_json: Path,
    *,
    linear_backend: str,
    phase1_threads: int,
    max_outer: int,
    max_inner: int,
    time_limit: float,
    live_output: bool,
    feasibility_tolerance: float,
    dual_tolerance: float,
    complementarity_tolerance: float,
    extra_args: Sequence[str],
) -> list[str]:
    command = build_r3p22_solver_command(
        binary, model, result_json,
        linear_backend=linear_backend,
        equality_model="condensed-retraction",
        phase1_threads=phase1_threads,
        max_outer=max_outer,
        max_inner=max_inner,
        time_limit=time_limit,
        live_every=1,
        live_iterations=live_output,
        condensed_curvature=True,
        lazyqp_kkt_form="reduced",
        flow_lift_schur=True,
        post_retraction_ws_expansion=True,
        modeled_side_filter=True,
        modeled_side_filter_limit=3.2e-5,
        modeled_side_recovery_margin=4.0,
        block_exact_curvature=True,
        projected_multi_ritz_inertia=False,
        lazyqp_flow_lift_elimination=True,
    )
    # Later occurrences win in the r3p23 CLI.  These common tolerances make
    # the pair certificate independent of historical builder defaults.
    command += [
        "--feas-tol", str(feasibility_tolerance),
        "--opt-tol", str(dual_tolerance),
        "--dual-inf-tol", str(dual_tolerance),
        "--compl-inf-tol", str(complementarity_tolerance),
        *extra_args,
    ]
    return command


def ipopt_command(
    binary: Path,
    model: Path,
    result_json: Path,
    *,
    tolerance: float,
    equality_tolerance: float,
    dual_tolerance: float,
    complementarity_tolerance: float,
    max_iterations: int,
    time_limit: float,
    linear_solver: str,
    print_level: int,
    interface: str = "native",
) -> list[str]:
    prefix = ([str(binary), str(model)] if interface == "casadi" else
              [str(binary), "--problem", str(model)])
    return [
        *prefix, "--json", str(result_json),
        "--tol", str(tolerance),
        "--constr-viol-tol", str(equality_tolerance),
        "--dual-inf-tol", str(dual_tolerance),
        "--compl-inf-tol", str(complementarity_tolerance),
        "--max-iter", str(max_iterations),
        "--time-limit", str(time_limit),
        "--linear-solver", linear_solver,
        "--print-level", str(print_level),
    ]


def solver_record(
    *,
    command: Sequence[str],
    result_json: Path,
    log_path: Path,
    return_code: int,
    wall_seconds: float,
    result: dict[str, Any],
    parse_error: str,
    certificate_pass: bool,
) -> dict[str, Any]:
    return {
        "command": list(command),
        "result_json": str(result_json),
        "log": str(log_path),
        "return_code": return_code,
        "wall_seconds": wall_seconds,
        "parse_error": parse_error,
        "certificate_pass": certificate_pass,
        "result": result,
    }


def run_pair(
    model: Path,
    lazy_binary: Path,
    ipopt_binary: Path,
    output_dir: Path,
    *,
    order: str = "ipopt-first",
    linear_backend: str = "ldl",
    phase1_threads: int = 8,
    max_outer: int = 1000,
    max_inner: int = 1000,
    ipopt_max_iterations: int = 3000,
    time_limit: float = 1800.0,
    equality_tolerance: float = 1.0e-6,
    inequality_tolerance: float = 1.0e-6,
    bound_tolerance: float = 1.0e-6,
    dual_tolerance: float = 1.0e-6,
    complementarity_tolerance: float = 1.0e-6,
    objective_absolute_tolerance: float = 1.0e-3,
    objective_relative_tolerance: float = 1.0e-8,
    ipopt_linear_solver: str = "mumps",
    ipopt_print_level: int = 5,
    ipopt_interface_name: str = "auto",
    lazy_extra_args: Sequence[str] = (),
    live_output: bool = False,
) -> dict[str, Any]:
    model = model.resolve()
    lazy_binary = lazy_binary.resolve()
    ipopt_binary = ipopt_binary.resolve()
    output_dir = output_dir.resolve()
    if not model.is_file():
        raise FileNotFoundError(f"exported QCQP is missing: {model}")
    for label, binary in (("LazyIPDCA", lazy_binary),
                          ("Ipopt-direct", ipopt_binary)):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(
                f"{label} executable is missing or not executable: {binary}")
    if order not in {"ipopt-first", "lazy-first"}:
        raise ValueError(f"unsupported execution order: {order}")
    output_dir.mkdir(parents=True, exist_ok=False)

    dimensions = qcq_dimensions(model)
    expected_hash = sha256_file(model)
    model_record: dict[str, Any] = {
        "path": str(model),
        "sha256": expected_hash,
        "bytes": model.stat().st_size,
        **dimensions,
    }
    effective_ipopt_interface = ipopt_interface(
        ipopt_binary, ipopt_interface_name)

    result_paths = {
        "ipopt": output_dir / "ipopt.result.json",
        "lazyipdca": output_dir / "lazyipdca.result.json",
    }
    log_paths = {
        "ipopt": output_dir / "ipopt.solve.log",
        "lazyipdca": output_dir / "lazyipdca.solve.log",
    }
    commands = {
        "ipopt": ipopt_command(
            ipopt_binary, model, result_paths["ipopt"],
            tolerance=max(dual_tolerance, complementarity_tolerance),
            equality_tolerance=min(equality_tolerance,
                                   inequality_tolerance),
            dual_tolerance=dual_tolerance,
            complementarity_tolerance=complementarity_tolerance,
            max_iterations=ipopt_max_iterations,
            time_limit=time_limit,
            linear_solver=ipopt_linear_solver,
            print_level=ipopt_print_level,
            interface=effective_ipopt_interface,
        ),
        "lazyipdca": lazy_command(
            lazy_binary, model, result_paths["lazyipdca"],
            linear_backend=linear_backend,
            phase1_threads=phase1_threads,
            max_outer=max_outer,
            max_inner=max_inner,
            time_limit=time_limit,
            live_output=live_output,
            feasibility_tolerance=min(equality_tolerance,
                                      inequality_tolerance,
                                      bound_tolerance),
            dual_tolerance=dual_tolerance,
            complementarity_tolerance=complementarity_tolerance,
            extra_args=lazy_extra_args,
        ),
    }

    records: dict[str, Any] = {}
    integrity_hashes = [expected_hash]
    labels = (["ipopt", "lazyipdca"] if order == "ipopt-first" else
              ["lazyipdca", "ipopt"])
    for label in labels:
        print(f"[{label.upper()}] same QCQP sha256={expected_hash}", flush=True)
        result_paths[label].unlink(missing_ok=True)
        return_code, wall_seconds = run_command(
            commands[label], log_paths[label], live_output=live_output)
        result, parse_error = load_result(result_paths[label])
        if label == "ipopt":
            result = normalize_ipopt_result(result)
        certificate_pass = kkt_pass(
            result,
            equality_tolerance=equality_tolerance,
            inequality_tolerance=inequality_tolerance,
            bound_tolerance=bound_tolerance,
            dual_tolerance=dual_tolerance,
            complementarity_tolerance=complementarity_tolerance,
        )
        records[label] = solver_record(
            command=commands[label], result_json=result_paths[label],
            log_path=log_paths[label], return_code=return_code,
            wall_seconds=wall_seconds, result=result,
            parse_error=parse_error, certificate_pass=certificate_pass)
        observed_hash = sha256_file(model)
        integrity_hashes.append(observed_hash)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"{label} modified the shared QCQP: {expected_hash} -> "
                f"{observed_hash}")

    ipopt_objective = finite_float(
        records["ipopt"]["result"].get("objective"))
    lazy_objective = finite_float(
        records["lazyipdca"]["result"].get("objective"))
    objective_abs = (
        abs(ipopt_objective - lazy_objective)
        if ipopt_objective is not None and lazy_objective is not None
        else None)
    objective_scale = (
        max(1.0, abs(ipopt_objective), abs(lazy_objective))
        if ipopt_objective is not None and lazy_objective is not None
        else None)
    objective_rel = (
        objective_abs / objective_scale
        if objective_abs is not None and objective_scale is not None else None)
    objective_match = (
        objective_abs is not None and objective_scale is not None and
        objective_abs <= objective_absolute_tolerance +
        objective_relative_tolerance * objective_scale)
    pair_pass = (
        all(records[name]["return_code"] == 0 for name in records) and
        all(records[name]["certificate_pass"] for name in records) and
        objective_match and len(set(integrity_hashes)) == 1)

    payload: dict[str, Any] = {
        "schema": PAIR_SCHEMA,
        "model": model_record,
        "model_integrity_hashes": integrity_hashes,
        "model_integrity_passed": len(set(integrity_hashes)) == 1,
        "execution_order": labels,
        "versions": {
            "ipopt": binary_version(
                ipopt_binary, interface=effective_ipopt_interface),
            "lazyipdca": binary_version(lazy_binary),
        },
        "ipopt_interface": effective_ipopt_interface,
        "tolerances": {
            "equality": equality_tolerance,
            "inequality": inequality_tolerance,
            "bound": bound_tolerance,
            "dual": dual_tolerance,
            "complementarity": complementarity_tolerance,
            "objective_absolute": objective_absolute_tolerance,
            "objective_relative": objective_relative_tolerance,
        },
        "ipopt": records["ipopt"],
        "lazyipdca": records["lazyipdca"],
        "comparison": {
            "objective_absolute_difference": objective_abs,
            "objective_relative_difference": objective_rel,
            "objective_match": objective_match,
        },
        "pair_passed": pair_pass,
    }
    write_json_atomic(output_dir / "summary.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", type=Path, required=True,
                    help="the already-exported physical .qcq file")
    ap.add_argument("--lazy-binary", type=Path, required=True)
    ap.add_argument("--ipopt-binary", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--order", choices=("ipopt-first", "lazy-first"),
                    default="ipopt-first")
    ap.add_argument("--linear-backend",
                    choices=("native", "ldl", "qdldl", "mumps"),
                    default="ldl")
    ap.add_argument("--phase1-threads", type=int, default=8)
    ap.add_argument("--max-outer", type=int, default=1000)
    ap.add_argument("--max-inner", type=int, default=1000)
    ap.add_argument("--ipopt-max-iterations", type=int, default=3000)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--eq-tol", type=float, default=1.0e-6)
    ap.add_argument("--ineq-tol", type=float, default=1.0e-6)
    ap.add_argument("--bound-tol", type=float, default=1.0e-6)
    ap.add_argument("--dual-tol", type=float, default=1.0e-6)
    ap.add_argument("--compl-tol", type=float, default=1.0e-6)
    ap.add_argument("--objective-abs-tol", type=float, default=1.0e-3)
    ap.add_argument("--objective-rel-tol", type=float, default=1.0e-8)
    ap.add_argument("--ipopt-linear-solver", default="mumps")
    ap.add_argument("--ipopt-print-level", type=int, default=5)
    ap.add_argument("--ipopt-interface",
                    choices=("auto", "native", "casadi"), default="auto")
    ap.add_argument(
        "--lazy-extra-args", default="",
        help="quoted additional r3p23 CLI arguments")
    ap.add_argument("--live-output", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    positive = {
        "phase1 threads": args.phase1_threads,
        "max outer": args.max_outer,
        "max inner": args.max_inner,
        "Ipopt max iterations": args.ipopt_max_iterations,
        "time limit": args.time_limit,
        "equality tolerance": args.eq_tol,
        "inequality tolerance": args.ineq_tol,
        "bound tolerance": args.bound_tol,
        "dual tolerance": args.dual_tol,
        "complementarity tolerance": args.compl_tol,
    }
    invalid = next((name for name, value in positive.items()
                    if not math.isfinite(float(value)) or value <= 0), None)
    if invalid:
        parser().error(f"{invalid} must be finite and positive")
    try:
        payload = run_pair(
            args.model, args.lazy_binary, args.ipopt_binary, args.output_dir,
            order=args.order,
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
            lazy_extra_args=shlex.split(args.lazy_extra_args),
            live_output=args.live_output,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[PAIR FAILURE] {exc}", file=sys.stderr)
        return 2

    ipopt = payload["ipopt"]["result"]
    lazy = payload["lazyipdca"]["result"]
    comparison = payload["comparison"]
    print("\n[SAME-QCQP RESULT]")
    print(f"model sha256 : {payload['model']['sha256']}")
    print(f"dimensions   : {payload['model']['n']} / "
          f"{payload['model']['equalities']} / "
          f"{payload['model']['inequalities']}")
    print(f"Ipopt        : status={ipopt.get('status')} "
          f"objective={ipopt.get('objective')} "
          f"solver_s={ipopt.get('elapsed_seconds')}")
    print(f"LazyIPDCA    : status={lazy.get('status')} "
          f"objective={lazy.get('objective')} "
          f"solver_s={lazy.get('elapsed_seconds')}")
    print("objective abs: ", comparison["objective_absolute_difference"])
    print("PAIR=", "PASS" if payload["pair_passed"] else "FAIL", sep="")
    print(f"summary      : {args.output_dir.resolve() / 'summary.json'}")
    return 0 if payload["pair_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
