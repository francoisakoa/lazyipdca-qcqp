#!/usr/bin/env python3
"""Contract test for python/export_pglib_qcqp_r3p21.py.

Self-contained: a small MATPOWER case is written to a temporary directory,
so the test needs neither network access nor the PGLib repository.

The test asserts independent algebraic, serialization, initialization,
flow-limit, and solver-invocation contracts.  Each catches a different class
of mutation; the core algebraic checks are:

1. Every assembled quadratic reproduces the value obtained from complex
    arithmetic  V .* conj(Y V),  V_u .* conj(I),  and the raw gencost
    polynomial.  Catches sign errors, transposed indices and the factor of
    two between  x^T M x  and  0.5 x^T H x.
2. The written .qcq file, parsed back and re-evaluated, agrees with the
    in-memory model.  Catches writer bugs: upper-triangle convention,
    triplet ordering, missing linear entries.
3. The imaginary part of the reference-bus voltage is a fixed variable.
    Without it the model is invariant under a global phase rotation, which
    puts a direction in the null space of the constraint Jacobian that no
    multiplier can cancel and the dual infeasibility never converges.
"""

import argparse
import math
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from export_pglib_qcqp_r3p21 import (  # noqa: E402
    BOUND_INFINITY, R3P21_API, R3P21_BUILD_ID, R3P21_VERSION, build_model,
    build_r3p21_solver_command, parse_matpower, solve_with_r3p21,
    validate, write_qcq,
)
from run_pglib_qcqp_r3p21 import parser as runner_parser  # noqa: E402

TINY_CASE = """
function mpc = tiny
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
mpc.bus = [
\t1\t3\t0.0\t0.0\t0.00\t0.00\t1\t1.00\t0.0\t230.0\t1\t1.10\t0.90;
\t2\t2\t30.0\t10.0\t0.00\t0.05\t1\t1.00\t0.0\t230.0\t1\t1.10\t0.90;
\t3\t1\t50.0\t20.0\t0.02\t0.00\t1\t1.00\t0.0\t230.0\t1\t1.10\t0.90;
];

%% generator data -- deliberately unbalanced: the tabulated output covers
%% 10 MW of an 80 MW load, and Qmax at bus 2 is tight enough to force the
%% PV -> PQ switch.  Both are needed for the power-flow assertions below to
%% have teeth.
mpc.gen = [
\t1\t5.0\t0.0\t100.0\t-100.0\t1.0\t100.0\t1\t30.0\t2.0;
\t2\t5.0\t0.0\t4.0\t-4.0\t1.0\t100.0\t1\t60.0\t3.0;
];

%% branch data
mpc.branch = [
\t1\t2\t0.01\t0.08\t0.020\t60.0\t0\t0\t0.00\t0.0\t1\t-25.0\t25.0;
\t1\t3\t0.02\t0.10\t0.030\t70.0\t0\t0\t0.98\t3.0\t1\t-25.0\t25.0;
\t2\t3\t0.03\t0.12\t0.010\t0.0\t0\t0\t0.00\t0.0\t1\t-25.0\t25.0;
];

%% generator cost data
mpc.gencost = [
\t2\t0.0\t0.0\t3\t0.110\t5.0\t20.0;
\t2\t0.0\t0.0\t3\t0.085\t1.2\t15.0;
];
"""


def parse_qcq(path):
    tokens = path.read_text().split()
    pos = 0

    def take():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def num():
        t = take()
        if t in ("INF", "+INF"):
            return math.inf
        if t == "-INF":
            return -math.inf
        return float(t)

    assert take() == "LAZYIPDCA_QCQP_V1"
    assert take() == "name"
    take()
    assert take() == "n"
    n = int(take())
    assert take() == "x0"
    x0 = np.array([num() for _ in range(n)])
    assert take() == "lb"
    lb = np.array([num() for _ in range(n)])
    assert take() == "ub"
    ub = np.array([num() for _ in range(n)])

    def quad():
        assert take() == "quadratic"
        assert take() == "name"
        name = take()
        assert take() == "constant"
        const = num()
        assert take() == "linear"
        lin = np.array([num() for _ in range(n)])
        assert take() == "hessian_nnz"
        nnz = int(take())
        trips = []
        for _ in range(nnz):
            i = int(take())
            j = int(take())
            trips.append((i, j, num()))
        assert take() == "end_quadratic"
        return name, const, lin, trips

    assert take() == "objective"
    obj = quad()
    assert take() == "equalities"
    eqs = [quad() for _ in range(int(take()))]
    assert take() == "inequalities"
    ineqs = [quad() for _ in range(int(take()))]
    assert take() == "end_problem"
    return n, x0, lb, ub, obj, eqs, ineqs


def evaluate(parsed_quad, x):
    """constant + linear.x + 0.5 x^T H x, H stored as its upper triangle."""
    _name, const, lin, trips = parsed_quad
    q = 0.0
    for i, j, v in trips:
        q += v * x[i] * x[i] if i == j else 2.0 * v * x[i] * x[j]
    return const + float(lin @ x) + 0.5 * q


def main(binary=None, linear_backend="ldl"):
    tmp = pathlib.Path(tempfile.mkdtemp())
    case_path = tmp / "tiny.m"
    case_path.write_text(TINY_CASE)

    failures = []
    if R3P21_VERSION != "0.4.28r3p21-partial-spectrum-sparse-kernels":
        failures.append(f"unexpected r3p21 version: {R3P21_VERSION}")
    if R3P21_API != 4304:
        failures.append(f"unexpected r3p21 API: {R3P21_API}")
    if R3P21_BUILD_ID != (
            "v0.4.28r3p21-partial-spectrum-sparse-kernels-api-4304"):
        failures.append(f"unexpected r3p21 build ID: {R3P21_BUILD_ID}")
    dimensions = {}
    for flow_limit in ("current", "apparent", "none"):
        case = parse_matpower(case_path)
        model = build_model(case, flow_limit=flow_limit, angle_limits=True)
        (n, x0, lb, ub, obj, eqs, ineqs, Y, bus, gen, branch, online,
         baseMVA, bus_index, meta, gencost, branches, lift) = model
        dimensions[flow_limit] = (n, len(eqs), len(ineqs))

        # --- 1. independent re-evaluation --------------------------------
        errors = validate(model, seed=7)
        for key in ("balance_error", "objective_error", "voltage_error",
                    "flow_error", "angle_error"):
            if not errors[key] <= 1e-9:
                failures.append(f"{flow_limit}: {key} = {errors[key]:.3e}")
        if not errors["probe_norm"] > 1e-3:
            failures.append(f"{flow_limit}: probe point did not move")

        # --- 2. file round-trip ------------------------------------------
        out = tmp / f"tiny_{flow_limit}.qcq"
        write_qcq(out, "tiny", n, x0, lb, ub, obj, eqs, ineqs)
        pn, px0, plb, pub, pobj, peqs, pineqs = parse_qcq(out)
        if pn != n or len(peqs) != len(eqs) or len(pineqs) != len(ineqs):
            failures.append(f"{flow_limit}: file dimensions differ")
            continue
        rng = np.random.default_rng(11)
        x = px0 + rng.normal(0.0, 0.05, pn)
        worst = abs(evaluate(pobj, x) - obj.value(x))
        for parsed, built in zip(peqs + pineqs, eqs + ineqs):
            worst = max(worst, abs(evaluate(parsed, x) - built.value(x)))
        if not worst <= 1e-9:
            failures.append(f"{flow_limit}: round-trip mismatch {worst:.3e}")
        if not np.allclose(px0, x0) or not np.allclose(
                np.where(np.isfinite(plb), plb, 0.0),
                np.where(lb > -BOUND_INFINITY, lb, 0.0)):
            failures.append(f"{flow_limit}: bounds or x0 not round-tripped")

        # --- 3. the phase rotation must be pinned ------------------------
        ref = int(meta["reference_bus"])
        r = bus_index[ref]
        if not (lb[2 * r + 1] == 0.0 and ub[2 * r + 1] == 0.0):
            failures.append(
                f"{flow_limit}: Im(V) at the reference bus is not fixed; "
                "the model keeps its global phase invariance")
        if not lb[2 * r] > 0.0:
            failures.append(
                f"{flow_limit}: Re(V) at the reference bus has no positive "
                "lower bound; the V -> -V reflection survives")

        print(f"[{flow_limit}] n={n} eq={len(eqs)} ineq={len(ineqs)} "
              f"worst_model={max(errors[k] for k in ('balance_error', 'objective_error', 'voltage_error', 'flow_error', 'angle_error')):.3e} "
              f"worst_file={worst:.3e}")

    # --- 4. power-flow start ------------------------------------------
    case = parse_matpower(case_path)
    model = build_model(case, flow_limit="apparent", angle_limits=True,
                        power_flow_start=True)
    (n, x0, lb, ub, obj, eqs, ineqs, Y, bus, gen, branch, online, baseMVA,
     bus_index, meta, gencost, branches, lift) = model
    note = str(meta["start"])
    if not note.startswith("power-flow"):
        failures.append(f"power flow did not run: {note}")
    else:
        residual = max(abs(q.value(x0)) for q in eqs)
        table = build_model(case, flow_limit="apparent", angle_limits=True)
        table_residual = max(abs(q.value(table[1])) for q in table[5])
        print(f"[power-flow] start residual {residual:.3e} "
              f"(table start {table_residual:.3e})")
        if not residual <= 1e-9:
            failures.append(f"power-flow start residual {residual:.3e} > 1e-9")
        if not residual < table_residual:
            failures.append("power-flow start is no better than the table")
        errors = validate(model, seed=3)
        for key in ("balance_error", "objective_error", "voltage_error",
                    "flow_error", "angle_error"):
            if not errors[key] <= 1e-9:
                failures.append(f"power-flow: {key} = {errors[key]:.3e}")

    # --- 5. flow-limit contracts ---------------------------------------
    none_model = build_model(case, flow_limit="none", angle_limits=True)
    none_n, _, _, _, _, none_eqs, none_ineqs = none_model[:7]
    forbidden_prefixes = ("ilim_", "slim_", "flowP_", "flowQ_")
    if any(q.name.startswith(forbidden_prefixes)
           for q in none_eqs + none_ineqs):
        failures.append("flow-limit none emitted a branch-flow constraint")
    expected_none_n = 2 * bus.shape[0] + 2 * len(online)
    if none_n != expected_none_n:
        failures.append(
            f"flow-limit none dimension {none_n} != {expected_none_n}")
    if not dimensions["none"][1] < dimensions["apparent"][1]:
        failures.append("flow-limit none retained apparent-flow equalities")
    if not dimensions["none"][2] < dimensions["current"][2]:
        failures.append("flow-limit none retained current-flow inequalities")

    # The lifted flow variables must stay unbounded.  A +/- rate box on them
    # A +/- rate box on them duplicates the convex |S|^2 <= rate^2 inequality
    # and makes the EQUALITY system unsolvable at an infeasible voltage
    # profile, which is exactly what the feasibility phase works on.
    nb = bus.shape[0]
    ng = len(online)
    base = 2 * nb + 2 * ng
    lifted_flow_branches = int(meta["lifted_flow_branches"])
    boxed = sum(1 for k in range(4 * lifted_flow_branches)
                if lb[base + k] > -BOUND_INFINITY or ub[base + k] < BOUND_INFINITY)
    if boxed:
        failures.append(f"{boxed} lifted flow variables carry a redundant box")
    limited_branches = sum(1 for item in branches if item[6][5] > 0.0)
    if lifted_flow_branches != limited_branches:
        failures.append(
            f"allocated {lifted_flow_branches} lifted branches for "
            f"{limited_branches} positive RATE_A limits")
    expected_n = 2 * nb + 2 * ng + 4 * limited_branches
    if n != expected_n:
        failures.append(
            f"apparent-flow dimension {n} != compact dimension {expected_n}")

    # --- 6. r3p21 runner defaults and solver-command contract -----------
    runner_defaults = runner_parser().parse_args([str(case_path)])
    expected_defaults = {
        "power_flow_voltage_continuation": False,
        "lazyqp_kkt_form": "reduced",
        "working_set_drop_patience": 8,
        "activation": 5.0e-4,
        "adaptive_normal_refinement": True,
        "dca_condensed_normal_intermediate_berr_tol": 1.0e-6,
        "dca_condensed_normal_intermediate_berr_reject": 1.0e-4,
        "dca_condensed_normal_strict_stationarity_factor": 1.0e3,
        "alpha_predictor": True,
        "dca_condensed_alpha_growth": 2.0,
        "dca_condensed_alpha_full_step_frequency": 8,
        "flow_lift_schur": True,
        "post_retraction_ws_expansion": True,
        "dca_condensed_post_retraction_ws_max": 8,
        "block_exact_curvature": True,
        "projected_multi_ritz_inertia": None,
        "lazyqp_flow_lift_elimination": True,
    }
    for key, expected in expected_defaults.items():
        actual = getattr(runner_defaults, key)
        if actual != expected:
            failures.append(
                f"r3p21 runner default {key}={actual!r}, expected "
                f"{expected!r}")
    runner_nonblock = runner_parser().parse_args([
        str(case_path), "--no-dca-condensed-block-exact-curvature"])
    if runner_nonblock.block_exact_curvature is not False:
        failures.append("r3p21 runner cannot explicitly disable block-exact")

    default_command = build_r3p21_solver_command(
        pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
        pathlib.Path("/result.json"), live_iterations=False)
    for flag, expected in (
            ("--lazyqp-kkt-form", "reduced"),
            ("--working-set-drop-patience", "8"),
            ("--activation", "0.0005"),
            ("--kkt-berr-tol", "1e-9"),
            ("--kkt-berr-reject", "1e-7"),
            ("--dca-condensed-normal-intermediate-berr-tol", "1e-06"),
            ("--dca-condensed-normal-intermediate-berr-reject", "0.0001"),
            ("--dca-condensed-normal-strict-stationarity-factor", "1000.0"),
            ("--dca-condensed-alpha-growth", "2.0"),
            ("--dca-condensed-alpha-full-step-frequency", "8"),
            ("--dca-condensed-post-retraction-ws-max", "8")):
        try:
            actual = default_command[default_command.index(flag) + 1]
        except (ValueError, IndexError):
            failures.append(f"r3p21 command misses {flag}")
        else:
            if actual != expected:
                failures.append(
                    f"r3p21 command {flag}={actual!r}, expected "
                    f"{expected!r}")
    for required in ("--dca-condensed-adaptive-normal-refinement",
                     "--dca-condensed-alpha-predictor",
                     "--dca-condensed-flow-lift-schur",
                     "--dca-condensed-post-retraction-ws-expansion",
                     "--dca-condensed-block-exact-curvature",
                     "--no-dca-condensed-projected-multi-ritz-inertia",
                     "--lazyqp-flow-lift-elimination"):
        if required not in default_command:
            failures.append(f"r3p21 command misses {required}")

    native_block_command = build_r3p21_solver_command(
        pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
        pathlib.Path("/result.json"), linear_backend="native",
        live_iterations=False)
    if "--dca-condensed-block-exact-curvature" not in native_block_command:
        failures.append("r3p21 native default did not enable block-exact")
    if ("--no-dca-condensed-projected-multi-ritz-inertia" not in
            native_block_command):
        failures.append(
            "r3p21 native block-exact route did not exclude multi-Ritz")

    native_multi_ritz_command = build_r3p21_solver_command(
        pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
        pathlib.Path("/result.json"), linear_backend="native",
        live_iterations=False, block_exact_curvature=False)
    for required in ("--no-dca-condensed-block-exact-curvature",
                     "--dca-condensed-projected-multi-ritz-inertia"):
        if required not in native_multi_ritz_command:
            failures.append(
                f"r3p21 native non-block auto policy misses {required}")

    ldl_nonblock_command = build_r3p21_solver_command(
        pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
        pathlib.Path("/result.json"), linear_backend="ldl",
        live_iterations=False, block_exact_curvature=False)
    for required in ("--no-dca-condensed-block-exact-curvature",
                     "--no-dca-condensed-projected-multi-ritz-inertia"):
        if required not in ldl_nonblock_command:
            failures.append(
                f"r3p21 LDL non-block auto policy misses {required}")

    try:
        build_r3p21_solver_command(
            pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
            pathlib.Path("/result.json"), live_iterations=False,
            block_exact_curvature=True,
            projected_multi_ritz_inertia=True)
    except ValueError:
        pass
    else:
        failures.append("r3p21 command accepted block-exact plus multi-Ritz")

    strict_command = build_r3p21_solver_command(
        pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
        pathlib.Path("/result.json"), live_iterations=False,
        adaptive_normal_refinement=False, alpha_predictor=False,
        flow_lift_schur=False, post_retraction_ws_expansion=False,
        block_exact_curvature=False,
        projected_multi_ritz_inertia=False,
        lazyqp_flow_lift_elimination=False)
    for required in ("--no-dca-condensed-adaptive-normal-refinement",
                     "--no-dca-condensed-alpha-predictor",
                     "--no-dca-condensed-flow-lift-schur",
                     "--no-dca-condensed-post-retraction-ws-expansion",
                     "--no-dca-condensed-block-exact-curvature",
                     "--no-dca-condensed-projected-multi-ritz-inertia",
                     "--no-lazyqp-flow-lift-elimination"):
        if required not in strict_command:
            failures.append(f"r3p21 A/B command misses {required}")

    for noncondensed_model in ("linearized", "elastic-split"):
        noncondensed_command = build_r3p21_solver_command(
            pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
            pathlib.Path("/result.json"),
            equality_model=noncondensed_model, live_iterations=False)
        leaked = [item for item in noncondensed_command
                  if item.startswith("--dca-condensed-adaptive-normal") or
                  item.startswith("--dca-condensed-normal-intermediate") or
                  item.startswith("--dca-condensed-normal-strict") or
                  item.startswith("--dca-condensed-alpha") or
                  item.startswith("--dca-condensed-flow-lift-schur") or
                  item.startswith("--no-dca-condensed-flow-lift-schur") or
                  item.startswith("--dca-condensed-post-retraction-ws") or
                  item.startswith("--no-dca-condensed-post-retraction-ws") or
                  item.startswith("--dca-condensed-block-exact") or
                  item.startswith("--no-dca-condensed-block-exact") or
                  item.startswith("--dca-condensed-projected-multi-ritz") or
                  item.startswith("--no-dca-condensed-projected-multi-ritz") or
                  item.startswith("--lazyqp-flow-lift-elimination") or
                  item.startswith("--no-lazyqp-flow-lift-elimination")]
        if leaked:
            failures.append(
                "condensed-only controls leaked into " +
                noncondensed_model + ": " + ", ".join(leaked))

    invalid_controls = (
        {"normal_intermediate_berr_tolerance": 0.0},
        {"normal_intermediate_berr_tolerance": 1.0e-3,
         "normal_intermediate_berr_reject": 1.0e-4},
        {"normal_strict_stationarity_factor": 0.0},
        {"alpha_growth": 1.0},
        {"alpha_full_step_frequency": 0},
        {"post_retraction_ws_max": -1},
    )
    for overrides in invalid_controls:
        try:
            build_r3p21_solver_command(
                pathlib.Path("/solver"), pathlib.Path("/problem.qcq"),
                pathlib.Path("/result.json"), live_iterations=False,
                **overrides)
        except ValueError:
            pass
        else:
            failures.append(
                "r3p21 command accepted invalid controls " + repr(overrides))

    # --- 7. real r3p21 load-and-solve contract -------------------------
    # This part is optional so the algebraic exporter tests remain usable on
    # machines where the C++ solver has not been built.  Release validation
    # passes --binary and therefore exercises the complete chain.
    if binary is not None:
        solver_model = tmp / "tiny_r3p21.qcq"
        solver_json = tmp / "tiny_r3p21.result.json"
        solver_log = tmp / "tiny_r3p21.solve.log"
        write_qcq(solver_model, "tiny_r3p21", n, x0, lb, ub,
                  obj, eqs, ineqs)
        command = build_r3p21_solver_command(
            pathlib.Path(binary), solver_model, solver_json,
            linear_backend=linear_backend,
            equality_model="condensed-retraction", phase1_threads=1,
            max_outer=1000, max_inner=1000, time_limit=60.0,
            live_iterations=False)
        for required in (
                "--equality-qp-original-kkt-gmres",
                "--dca-equality-model", "condensed-retraction",
                "--dca-condensed-curvature",
                "--dca-condensed-max-rank", "8",
                "--dca-affine-box-bounded-lazy",
                "--restoration-exact-workspace-reuse",
                "--restoration-exact-sparse-row-dots",
                "--lazyqp-kkt-form", "reduced",
                "--working-set-drop-patience", "8",
                "--activation", "0.0005",
                "--kkt-berr-tol", "1e-9",
                "--dca-condensed-adaptive-normal-refinement",
                "--dca-condensed-normal-intermediate-berr-tol", "1e-06",
                "--dca-condensed-normal-intermediate-berr-reject", "0.0001",
                "--dca-condensed-normal-strict-stationarity-factor", "1000.0",
                "--dca-condensed-alpha-predictor",
                "--dca-condensed-alpha-growth", "2.0",
                "--dca-condensed-alpha-full-step-frequency", "8",
                "--dca-condensed-flow-lift-schur",
                "--dca-condensed-post-retraction-ws-expansion",
                "--dca-condensed-post-retraction-ws-max",
                "--dca-condensed-block-exact-curvature",
                "--lazyqp-flow-lift-elimination"):
            if required not in command:
                failures.append(f"r3p21 command misses {required}")
        expected_multi_ritz = \
            "--no-dca-condensed-projected-multi-ritz-inertia"
        if expected_multi_ritz not in command:
            failures.append(f"r3p21 command misses {expected_multi_ritz}")
        try:
            rc, result, version, _log, curvature_used = solve_with_r3p21(
                pathlib.Path(binary), solver_model, solver_json, solver_log,
                linear_backend=linear_backend,
                equality_model="condensed-retraction", phase1_threads=1,
                max_outer=1000, max_inner=1000, time_limit=60.0,
                live_output=False, condensed_curvature="auto")
        except Exception as exc:  # contract failure must be reported cleanly
            failures.append(f"r3p21 invocation failed: {exc}")
        else:
            if R3P21_VERSION not in version:
                failures.append(f"wrong solver version: {version}")
            if rc != 0 or result.get("status") != "solved":
                failures.append(
                    f"r3p21 did not solve tiny model: rc={rc} "
                    f"status={result.get('status')}")
            for key in ("equality_violation", "inequality_violation",
                        "dual_infeasibility_inf"):
                value = result.get(key)
                if value is None or float(value) > 1e-6:
                    failures.append(f"r3p21 {key}={value} exceeds 1e-6")
            if result.get("lazy_dca_equality_model") != "condensed-retraction":
                failures.append("r3p21 did not use condensed-retraction")
            block_attempts = result.get(
                "lazy_dca_condensed_block_exact_attempts")
            block_acceptances = result.get(
                "lazy_dca_condensed_block_exact_acceptances")
            partial_attempts = result.get(
                "lazy_dca_condensed_block_partial_spectrum_attempts")
            partial_acceptances = result.get(
                "lazy_dca_condensed_block_partial_spectrum_acceptances")
            partial_fallbacks = result.get(
                "lazy_dca_condensed_block_partial_spectrum_fallbacks")
            geometry_solves = result.get(
                "lazy_dca_condensed_geometry_normal_solves")
            multi_ritz_batches = result.get(
                "lazy_dca_condensed_projected_multi_ritz_batches")
            if not isinstance(block_attempts, int) or block_attempts <= 0:
                failures.append(
                    f"r3p21 block-exact attempts={block_attempts!r}")
            if block_acceptances != block_attempts:
                failures.append(
                    "r3p21 block-exact attempts/acceptances differ: "
                    f"{block_attempts!r}/{block_acceptances!r}")
            if (not isinstance(partial_attempts, int) or
                    partial_attempts <= 0):
                failures.append(
                    f"r3p21 partial-spectrum attempts={partial_attempts!r}")
            if partial_acceptances != partial_attempts:
                failures.append(
                    "r3p21 partial-spectrum attempts/acceptances differ: "
                    f"{partial_attempts!r}/{partial_acceptances!r}")
            if partial_fallbacks != 0:
                failures.append(
                    "r3p21 partial-spectrum fallback count is nonzero: "
                    f"{partial_fallbacks!r}")
            if geometry_solves != 0:
                failures.append(
                    f"r3p21 used {geometry_solves!r} geometry normal solves")
            if multi_ritz_batches != 0:
                failures.append(
                    f"r3p21 used {multi_ritz_batches!r} multi-Ritz batches")
            if curvature_used not in {"on", "off-fallback"}:
                failures.append(
                    f"unexpected condensed-curvature route: {curvature_used}")
            if curvature_used == "off-fallback":
                first_json = solver_json.with_name(
                    solver_json.stem + ".curvature-on" + solver_json.suffix)
                first_log = solver_log.with_name(
                    solver_log.stem + ".curvature-on" + solver_log.suffix)
                if not first_json.is_file() or not first_log.is_file():
                    failures.append(
                        "guarded fallback did not preserve its curvature-on "
                        "JSON and log")
            print(f"[r3p21] status={result.get('status')} "
                  f"obj={result.get('objective'):.12g} "
                  f"eq={result.get('equality_violation'):.3e} "
                  f"ineq={result.get('inequality_violation'):.3e} "
                  f"dual={result.get('dual_infeasibility_inf'):.3e} "
                  f"curvature={curvature_used}")
    else:
        print("[r3p21] solver smoke skipped (pass --binary to enable it)")

    if failures:
        for f in failures:
            print("FAIL: " + f, file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path,
                        help="already compiled r3p21 lazyipdca_qcqp_cli")
    parser.add_argument("--linear-backend",
                        choices=("native", "ldl", "qdldl", "mumps"),
                        default="ldl")
    options = parser.parse_args()
    sys.exit(main(options.binary, options.linear_backend))
