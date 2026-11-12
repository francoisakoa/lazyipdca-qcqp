#!/usr/bin/env python3
"""Export and solve one MATPOWER/PGLib QCQP with an existing r3p17 binary.

This runner never configures or compiles LazyIPDCA.  It creates a timestamped
result directory, streams the live iterations, and writes one summary.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys

from export_pglib_qcqp_r3p17 import (
    build_model,
    parse_matpower,
    solve_with_r3p17,
    validate,
    write_qcq,
)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", type=pathlib.Path,
                    help="MATPOWER/PGLib case .m file")
    ap.add_argument("--base", type=pathlib.Path,
                    default=pathlib.Path(os.environ.get(
                        "BASE", "/data/projects/IPDCA_QCQP")))
    ap.add_argument("--binary", type=pathlib.Path,
                    default=(pathlib.Path(os.environ["BIN"])
                             if os.environ.get("BIN") else None),
                    help="existing r3p17 CLI (default: BIN or ROOT/build/...)")
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    ap.add_argument("--results-root", type=pathlib.Path, default=None)
    ap.add_argument("--flow-limit", choices=("current", "apparent", "none"),
                    default="apparent")
    ap.add_argument("--no-angle-limits", action="store_true")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--power-flow-start", dest="power_flow_start",
                       action="store_true")
    start.add_argument("--no-power-flow-start", dest="power_flow_start",
                       action="store_false")
    ap.set_defaults(power_flow_start=True)
    voltage_continuation = ap.add_mutually_exclusive_group()
    voltage_continuation.add_argument(
        "--power-flow-voltage-continuation",
        dest="power_flow_voltage_continuation", action="store_true",
        help="opt in to bounded PV-voltage continuation when a raw power "
             "flow violates only lower-voltage constraints")
    voltage_continuation.add_argument(
        "--no-power-flow-voltage-continuation",
        dest="power_flow_voltage_continuation", action="store_false",
        help="retain the raw power-flow start (default)")
    ap.set_defaults(power_flow_voltage_continuation=False)
    ap.add_argument("--linear-backend",
                    choices=("native", "ldl", "qdldl", "mumps"),
                    default="ldl")
    ap.add_argument("--dca-equality-model",
                    choices=("linearized", "condensed-retraction",
                             "elastic-split"),
                    default="condensed-retraction")
    ap.add_argument("--lazyqp-kkt-form",
                    choices=("auto", "full", "reduced"),
                    default="reduced",
                    help="LazyQP KKT route; reduced retains the guarded "
                         "full-KKT fallback")
    ap.add_argument("--condensed-curvature", choices=("auto", "on", "off"),
                    default="auto")
    ap.add_argument("--phase1-threads", type=int, default=1)
    ap.add_argument("--max-outer", type=int, default=1000)
    ap.add_argument("--max-inner", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--live-every", type=int, default=1)
    ap.add_argument("--working-set-drop-patience", type=int, default=8,
                    help="accepted points required before an inequality may "
                         "leave the working set")
    ap.add_argument("--activation", type=float, default=5.0e-4,
                    help="Lazy-DCA inequality activation threshold")
    ap.add_argument("--no-live-iterations", action="store_true")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = (args.binary or root / "build" / "lazyipdca_qcqp_cli").resolve()
    case_path = args.case.resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"R3P17_PGLIB_{case_path.stem}_{timestamp}"
    results_root = (args.results_root or args.base / "results").resolve()
    out = results_root / run_id
    out.mkdir(parents=True, exist_ok=False)

    model_path = out / f"{case_path.stem}.qcq"
    export_json = out / f"{case_path.stem}.export.json"
    result_json = out / f"{case_path.stem}.result.json"
    solver_log = out / f"{case_path.stem}.solve.log"
    summary_json = out / "summary.json"

    summary = {
        "run_id": run_id,
        "case": str(case_path),
        "binary": str(binary),
        "flow_limit": args.flow_limit,
        "power_flow_voltage_continuation":
            args.power_flow_voltage_continuation,
        "dca_equality_model": args.dca_equality_model,
        "linear_backend": args.linear_backend,
        "lazyqp_kkt_form": args.lazyqp_kkt_form,
        "activation": args.activation,
        "paths": {
            "results": str(out),
            "model": str(model_path),
            "export_json": str(export_json),
            "result_json": str(result_json),
            "solver_log": str(solver_log),
        },
    }
    return_code = 2
    try:
        case = parse_matpower(case_path)
        model = build_model(
            case, flow_limit=args.flow_limit,
            angle_limits=not args.no_angle_limits,
            power_flow_start=args.power_flow_start,
            power_flow_voltage_continuation=
                args.power_flow_voltage_continuation)
        (n, x0, lb, ub, objective, equalities, inequalities, _Y, _bus,
         _gen, _branch, _online, _base_mva, _bus_index, metadata,
         _gencost, _branches, _lift) = model
        checks = validate(model)
        metadata.update(checks)
        write_qcq(model_path, case_path.stem, n, x0, lb, ub, objective,
                  equalities, inequalities)
        metadata["output"] = str(model_path)
        metadata["bytes"] = model_path.stat().st_size
        export_json.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        summary["export"] = metadata

        (return_code, result, version, _log,
         curvature_used) = solve_with_r3p17(
            binary, model_path, result_json, solver_log,
            linear_backend=args.linear_backend,
            equality_model=args.dca_equality_model,
            phase1_threads=args.phase1_threads,
            max_outer=args.max_outer, max_inner=args.max_inner,
            time_limit=args.time_limit, live_every=args.live_every,
            live_output=not args.no_live_iterations,
            condensed_curvature=args.condensed_curvature,
            working_set_drop_patience=args.working_set_drop_patience,
            lazyqp_kkt_form=args.lazyqp_kkt_form,
            activation=args.activation)
        summary["solver"] = dict(result)
        summary["solver"]["version"] = version
        summary["solver"]["return_code"] = return_code
        summary["solver"]["condensed_curvature"] = curvature_used
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        summary["solver"] = {
            "status": "runner_failure", "return_code": 2,
            "message": str(exc),
        }
        print(f"[FAIL] {exc}", file=sys.stderr)
        return_code = 2
    finally:
        summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    solver = summary["solver"]
    print("\n[PGLIB SUMMARY]")
    print(f"case       : {case_path.stem}")
    print(f"status     : {solver.get('status')}")
    print(f"objective  : {solver.get('objective')}")
    print(f"equality   : {solver.get('equality_violation')}")
    print(f"inequality : {solver.get('inequality_violation')}")
    print(f"dual       : {solver.get('dual_infeasibility_inf')}")
    print(f"results    : {out}")
    print(f"summary    : {summary_json}")
    return int(return_code)


if __name__ == "__main__":
    sys.exit(main())
