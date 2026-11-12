#!/usr/bin/env python3
"""Export and solve one MATPOWER/PGLib QCQP with an existing r3p19 binary.

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

from export_pglib_qcqp_r3p19 import (
    build_model,
    parse_matpower,
    solve_with_r3p19,
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
                    help="existing r3p19 CLI (default: BIN or ROOT/build/...)")
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
    normal_refinement = ap.add_mutually_exclusive_group()
    normal_refinement.add_argument(
        "--dca-condensed-adaptive-normal-refinement",
        dest="adaptive_normal_refinement", action="store_true",
        help="relax only intermediate condensed-normal solves (default)")
    normal_refinement.add_argument(
        "--no-dca-condensed-adaptive-normal-refinement",
        dest="adaptive_normal_refinement", action="store_false",
        help="use strict normal solves throughout for an A/B run")
    ap.set_defaults(adaptive_normal_refinement=True)
    ap.add_argument(
        "--dca-condensed-normal-intermediate-berr-tol", type=float,
        default=1.0e-6)
    ap.add_argument(
        "--dca-condensed-normal-intermediate-berr-reject", type=float,
        default=1.0e-4)
    ap.add_argument(
        "--dca-condensed-normal-strict-stationarity-factor", type=float,
        default=1.0e3,
        help="strict-normal gate as a multiple of dual_inf_tol")
    alpha_predictor = ap.add_mutually_exclusive_group()
    alpha_predictor.add_argument(
        "--dca-condensed-alpha-predictor", dest="alpha_predictor",
        action="store_true",
        help="reuse a guarded condensed line-search seed (default)")
    alpha_predictor.add_argument(
        "--no-dca-condensed-alpha-predictor", dest="alpha_predictor",
        action="store_false",
        help="restart every condensed outer line search at alpha=1")
    ap.set_defaults(alpha_predictor=True)
    ap.add_argument("--dca-condensed-alpha-growth", type=float, default=2.0)
    ap.add_argument("--dca-condensed-alpha-full-step-frequency", type=int,
                    default=8,
                    help="periodic alpha=1 safeguard interval")
    flow_lift_schur = ap.add_mutually_exclusive_group()
    flow_lift_schur.add_argument(
        "--dca-condensed-flow-lift-schur", dest="flow_lift_schur",
        action="store_true",
        help="use exact Schur elimination for recognized PGLib flow lifts "
             "(default for condensed PGLib runs)")
    flow_lift_schur.add_argument(
        "--no-dca-condensed-flow-lift-schur", dest="flow_lift_schur",
        action="store_false",
        help="retain the full bordered condensed-normal factorization")
    post_retraction_ws = ap.add_mutually_exclusive_group()
    post_retraction_ws.add_argument(
        "--dca-condensed-post-retraction-ws-expansion",
        dest="post_retraction_ws_expansion", action="store_true",
        help="re-solve the same outer model when retraction exposes an "
             "omitted side-constraint offender (default)")
    post_retraction_ws.add_argument(
        "--no-dca-condensed-post-retraction-ws-expansion",
        dest="post_retraction_ws_expansion", action="store_false",
        help="retain historical alpha backtracking for omitted offenders")
    ap.set_defaults(flow_lift_schur=True,
                    post_retraction_ws_expansion=True)
    ap.add_argument("--dca-condensed-post-retraction-ws-max", type=int,
                    default=8,
                    help="same-outer working-set expansions before guarded "
                         "historical fallback")
    projected_multi_ritz = ap.add_mutually_exclusive_group()
    projected_multi_ritz.add_argument(
        "--dca-condensed-projected-multi-ritz-inertia",
        dest="projected_multi_ritz_inertia", action="store_true",
        help="use batched projected Ritz proposals with an exact inertia "
             "gate (auto-enabled only with the native backend)")
    projected_multi_ritz.add_argument(
        "--no-dca-condensed-projected-multi-ritz-inertia",
        dest="projected_multi_ritz_inertia", action="store_false",
        help="retain the historical condensed curvature route (default for "
             "regularized sparse backends such as LDL)")
    lazyqp_flow_lift = ap.add_mutually_exclusive_group()
    lazyqp_flow_lift.add_argument(
        "--lazyqp-flow-lift-elimination",
        dest="lazyqp_flow_lift_elimination", action="store_true",
        help="exactly eliminate recognized PGLib flow lifts from LazyQP "
             "subproblems (default for condensed PGLib runs)")
    lazyqp_flow_lift.add_argument(
        "--no-lazyqp-flow-lift-elimination",
        dest="lazyqp_flow_lift_elimination", action="store_false",
        help="retain the unreduced LazyQP subproblem")
    ap.set_defaults(projected_multi_ritz_inertia=None,
                    lazyqp_flow_lift_elimination=True)
    ap.add_argument("--no-live-iterations", action="store_true")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = pathlib.Path(__file__).resolve().parents[1]
    binary = (args.binary or root / "build" / "lazyipdca_qcqp_cli").resolve()
    case_path = args.case.resolve()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"R3P19_PGLIB_{case_path.stem}_{timestamp}"
    results_root = (args.results_root or args.base / "results").resolve()
    out = results_root / run_id
    out.mkdir(parents=True, exist_ok=False)

    model_path = out / f"{case_path.stem}.qcq"
    export_json = out / f"{case_path.stem}.export.json"
    result_json = out / f"{case_path.stem}.result.json"
    solver_log = out / f"{case_path.stem}.solve.log"
    summary_json = out / "summary.json"
    condensed_profile = args.dca_equality_model == "condensed-retraction"
    # The exact inertia certificate requires an unregularized right inverse.
    # SuiteSparse LDL used a 1e-12 normal regularization on case118, making
    # every multi-Ritz proposal ineligible after its expensive projections.
    # Preserve explicit overrides while bypassing that path by default.
    projected_multi_ritz_effective = (
        args.projected_multi_ritz_inertia
        if args.projected_multi_ritz_inertia is not None
        else args.linear_backend == "native")
    projected_multi_ritz_policy = (
        "forced-on" if args.projected_multi_ritz_inertia is True else
        "forced-off" if args.projected_multi_ritz_inertia is False else
        "auto-native" if projected_multi_ritz_effective else
        "auto-regularized-backend-bypass")

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
        "condensed_phase2_controls": {
            "effective": condensed_profile,
            "adaptive_normal_refinement": args.adaptive_normal_refinement,
            "normal_intermediate_berr_tolerance":
                args.dca_condensed_normal_intermediate_berr_tol,
            "normal_intermediate_berr_reject":
                args.dca_condensed_normal_intermediate_berr_reject,
            "normal_strict_stationarity_factor":
                args.dca_condensed_normal_strict_stationarity_factor,
            "alpha_predictor": args.alpha_predictor,
            "alpha_growth": args.dca_condensed_alpha_growth,
            "alpha_full_step_frequency":
                args.dca_condensed_alpha_full_step_frequency,
            "flow_lift_schur": condensed_profile and args.flow_lift_schur,
            "post_retraction_ws_expansion":
                condensed_profile and args.post_retraction_ws_expansion,
            "post_retraction_ws_max":
                args.dca_condensed_post_retraction_ws_max,
            "projected_multi_ritz_inertia":
                condensed_profile and projected_multi_ritz_effective,
            "projected_multi_ritz_policy": projected_multi_ritz_policy,
            "lazyqp_flow_lift_elimination":
                condensed_profile and args.lazyqp_flow_lift_elimination,
        },
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
         curvature_used) = solve_with_r3p19(
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
            activation=args.activation,
            adaptive_normal_refinement=args.adaptive_normal_refinement,
            normal_intermediate_berr_tolerance=
                args.dca_condensed_normal_intermediate_berr_tol,
            normal_intermediate_berr_reject=
                args.dca_condensed_normal_intermediate_berr_reject,
            normal_strict_stationarity_factor=
                args.dca_condensed_normal_strict_stationarity_factor,
            alpha_predictor=args.alpha_predictor,
            alpha_growth=args.dca_condensed_alpha_growth,
            alpha_full_step_frequency=
                args.dca_condensed_alpha_full_step_frequency,
            flow_lift_schur=args.flow_lift_schur,
            post_retraction_ws_expansion=
                args.post_retraction_ws_expansion,
            post_retraction_ws_max=
                args.dca_condensed_post_retraction_ws_max,
            projected_multi_ritz_inertia=
                projected_multi_ritz_effective,
            lazyqp_flow_lift_elimination=
                args.lazyqp_flow_lift_elimination)
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
    reduced_vars = solver.get("lazyqp_affine_reduced_variables")
    reduced_eq = solver.get("lazyqp_affine_reduced_equalities")
    eliminated_vars = solver.get("lazyqp_affine_eliminated_variables")
    eliminated_eq = solver.get("lazyqp_affine_eliminated_equalities")
    affine_attempts = solver.get("lazyqp_affine_elimination_attempts")
    affine_exercised = (
        isinstance(affine_attempts, int) and affine_attempts > 0)
    lazyqp_full_vars = (
        reduced_vars + eliminated_vars
        if affine_exercised and isinstance(reduced_vars, int) and
        isinstance(eliminated_vars, int) else None)
    lazyqp_full_eq = (
        reduced_eq + eliminated_eq
        if affine_exercised and isinstance(reduced_eq, int) and
        isinstance(eliminated_eq, int)
        else None)
    affine_dims = (
        f"{lazyqp_full_vars}->{reduced_vars} / {lazyqp_full_eq}->{reduced_eq}"
        if lazyqp_full_vars is not None and lazyqp_full_eq is not None
        else "n/a")
    print("\n[PGLIB SUMMARY]")
    print(f"case       : {case_path.stem}")
    print(f"status     : {solver.get('status')}")
    print(f"objective  : {solver.get('objective')}")
    print(f"equality   : {solver.get('equality_violation')}")
    print(f"inequality : {solver.get('inequality_violation')}")
    print(f"dual       : {solver.get('dual_infeasibility_inf')}")
    print("flow Schur : "
          f"lifts={solver.get('lazy_dca_condensed_flow_lift_schur_detected_variables')} "
          f"dim={solver.get('lazy_dca_condensed_flow_lift_schur_factor_dimension')} "
          f"fact={solver.get('lazy_dca_condensed_flow_lift_schur_factorizations')} "
          f"fallback={solver.get('lazy_dca_condensed_flow_lift_schur_fallbacks')}")
    print("post-ret WS: "
          f"expand={solver.get('lazy_dca_condensed_post_retraction_working_set_expansions')} "
          f"fallback={solver.get('lazy_dca_condensed_post_retraction_expansion_fallbacks')}")
    print("multi-Ritz : "
          f"policy={projected_multi_ritz_policy} "
          f"batches={solver.get('lazy_dca_condensed_projected_multi_ritz_batches')} "
          f"fallback={solver.get('lazy_dca_condensed_projected_multi_ritz_fallbacks')} "
          f"inertia={solver.get('lazy_dca_condensed_tangent_inertia_attempts')}/"
          f"{solver.get('lazy_dca_condensed_tangent_inertia_acceptances')} "
          f"shift={solver.get('lazy_dca_condensed_tangent_inertia_shift_updates')} "
          f"max={solver.get('lazy_dca_condensed_tangent_inertia_shift_max')}")
    print("LazyQP lift: "
          f"try/ok/fb={solver.get('lazyqp_affine_elimination_attempts')}/"
          f"{solver.get('lazyqp_affine_elimination_acceptances')}/"
          f"{solver.get('lazyqp_affine_elimination_fallbacks')} "
          f"elim={solver.get('lazyqp_affine_eliminated_variables')}/"
          f"{solver.get('lazyqp_affine_eliminated_equalities')} "
          f"dims={affine_dims} "
          f"KKT={solver.get('lazyqp_max_kkt_dimension')}/"
          f"{solver.get('lazyqp_max_full_kkt_dimension')}")
    print(f"results    : {out}")
    print(f"summary    : {summary_json}")
    return int(return_code)


if __name__ == "__main__":
    sys.exit(main())
