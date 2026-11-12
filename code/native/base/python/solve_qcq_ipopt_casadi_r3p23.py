#!/usr/bin/env python3
"""Solve one exported LAZYIPDCA_QCQP_V1 model directly with Ipopt.

This is an original-QCQP reference runner.  It does not call LazyIPDCA and it
does not convexify, eliminate, or otherwise rewrite the exported variables or
constraints.  CasADi supplies exact sparse first and second derivatives to
Ipopt for the expressions read from the ``.qcq`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from qcq_model_r3p23 import QCQPModel, QuadraticFunction, read_qcq


SCHEMA = "lazyipdca-qcqp-ipopt-direct-result-v1"
RUNNER = "r3p23-casadi-ipopt-exact-qcq"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path, help="exported .qcq model")
    parser.add_argument(
        "--json", "--output", dest="json_path", type=Path, default=None,
        help="result JSON (default: MODEL.ipopt-direct.result.json)",
    )
    parser.add_argument(
        "--tol", type=float, default=1.0e-6,
        help="overall and default component tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--constr-viol-tol", type=float, default=None,
        help="constraint tolerance (default: --tol)",
    )
    parser.add_argument(
        "--dual-inf-tol", type=float, default=None,
        help="unscaled dual tolerance (default: --tol)",
    )
    parser.add_argument(
        "--compl-inf-tol", type=float, default=None,
        help="complementarity tolerance (default: --tol)",
    )
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--linear-solver", default="mumps")
    parser.add_argument("--print-level", type=int, default=5)
    parser.add_argument(
        "--no-solution-vector", action="store_true",
        help="omit the primal vector from JSON",
    )
    args = parser.parse_args(argv)
    for name in ("constr_viol_tol", "dual_inf_tol", "compl_inf_tol"):
        if getattr(args, name) is None:
            setattr(args, name, args.tol)
    for name in ("tol", "constr_viol_tol", "dual_inf_tol", "compl_inf_tol"):
        if not (math.isfinite(getattr(args, name)) and getattr(args, name) > 0.0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    if not (math.isfinite(args.time_limit) and args.time_limit > 0.0):
        parser.error("--time-limit must be finite and positive")
    if not (0 <= args.print_level <= 12):
        parser.error("--print-level must lie in [0, 12]")
    return args


def _quadratic_expression(ca: Any, quadratic: QuadraticFunction, x: Any) -> Any:
    terms = [ca.SX(quadratic.constant)]
    terms.extend(coefficient * x[index] for index, coefficient in quadratic.linear)
    for entry in quadratic.hessian:
        if entry.i == entry.j:
            terms.append(0.5 * entry.value * x[entry.i] * x[entry.i])
        else:
            terms.append(entry.value * x[entry.i] * x[entry.j])
    return ca.sum1(ca.vertcat(*terms))


def _build_solver(ca: Any, model: QCQPModel, args: argparse.Namespace) -> Any:
    x = ca.SX.sym("x", model.n)
    objective = _quadratic_expression(ca, model.objective, x)
    expressions = [
        _quadratic_expression(ca, quadratic, x)
        for quadratic in model.constraints
    ]
    constraints = ca.vertcat(*expressions) if expressions else ca.SX.zeros(0, 1)
    options = {
        "print_time": args.print_level > 0,
        "ipopt.sb": "yes",
        "ipopt.print_level": args.print_level,
        "ipopt.print_timing_statistics": "yes" if args.print_level > 0 else "no",
        "ipopt.tol": args.tol,
        "ipopt.constr_viol_tol": args.constr_viol_tol,
        "ipopt.dual_inf_tol": args.dual_inf_tol,
        "ipopt.compl_inf_tol": args.compl_inf_tol,
        "ipopt.max_iter": args.max_iter,
        "ipopt.max_cpu_time": args.time_limit,
        "ipopt.linear_solver": args.linear_solver,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.hessian_approximation": "exact",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.bound_relax_factor": 0.0,
        "ipopt.fixed_variable_treatment": "make_parameter",
    }
    return ca.nlpsol(
        "ipopt_direct_exact_qcq",
        "ipopt",
        {"x": x, "f": objective, "g": constraints},
        options,
    )


def _vector(value: Any) -> list[float]:
    return [float(item) for item in value.full().reshape((-1,)).tolist()]


def _finite_float(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    problem_path = args.problem.expanduser().resolve()
    if not problem_path.is_file():
        print(f"error: QCQ model is missing: {problem_path}", file=sys.stderr)
        return 2
    output_path = (
        args.json_path.expanduser().resolve()
        if args.json_path is not None
        else problem_path.with_name(problem_path.stem + ".ipopt-direct.result.json")
    )

    total_start = time.perf_counter()
    parse_start = time.perf_counter()
    try:
        model = read_qcq(problem_path)
    except (OSError, ValueError) as exc:
        print(f"error: cannot parse {problem_path}: {exc}", file=sys.stderr)
        return 2
    parse_seconds = time.perf_counter() - parse_start

    try:
        import casadi as ca
    except ImportError:
        print(
            "error: CasADi is required; install it in the active environment "
            "with `python -m pip install casadi`",
            file=sys.stderr,
        )
        return 2

    build_start = time.perf_counter()
    try:
        solver = _build_solver(ca, model, args)
    except Exception as exc:  # CasADi reports plugin/configuration errors here.
        print(f"error: cannot build the CasADi/Ipopt NLP: {exc}", file=sys.stderr)
        return 2
    build_seconds = time.perf_counter() - build_start

    equality_count = len(model.equalities)
    inequality_count = len(model.inequalities)
    lower_constraints = [0.0] * equality_count + [-math.inf] * inequality_count
    upper_constraints = [0.0] * (equality_count + inequality_count)
    initial_residuals = model.residuals(model.x0)

    solve_start = time.perf_counter()
    try:
        solution = solver(
            x0=model.x0,
            lbx=model.lb,
            ubx=model.ub,
            lbg=lower_constraints,
            ubg=upper_constraints,
        )
    except Exception as exc:
        solve_wall_seconds = time.perf_counter() - solve_start
        stats = solver.stats()
        payload = {
            "schema": SCHEMA,
            "runner": RUNNER,
            "problem": model.name,
            "model_path": str(problem_path),
            "model_sha256": _sha256(problem_path),
            "status": str(stats.get("return_status", type(exc).__name__)),
            "success": False,
            "message": str(exc),
            "iterations": int(stats.get("iter_count", 0) or 0),
            "parse_seconds": parse_seconds,
            "model_build_seconds": build_seconds,
            "solve_wall_seconds": solve_wall_seconds,
            "total_wall_seconds": time.perf_counter() - total_start,
            "initial_residuals": initial_residuals,
        }
        _write_json(output_path, payload)
        print(f"error: Ipopt solve failed: {exc}", file=sys.stderr)
        print(f"result: {output_path}", file=sys.stderr)
        return 1
    solve_wall_seconds = time.perf_counter() - solve_start
    stats = solver.stats()

    x_final = _vector(solution["x"])
    constraint_multipliers = _vector(solution["lam_g"])
    bound_multipliers = _vector(solution["lam_x"])
    objective = model.objective_value(x_final)
    primal = model.residuals(x_final)
    kkt = model.kkt_residuals(
        x_final, constraint_multipliers, bound_multipliers
    )
    residuals = {**primal, **kkt}
    kkt_passed = (
        primal["primal_inf"] <= args.constr_viol_tol
        and kkt["dual_inf"] <= args.dual_inf_tol
        and kkt["complementarity_inf"] <= args.compl_inf_tol
        and kkt["dual_sign_inf"] <= args.dual_inf_tol
    )
    success = bool(stats.get("success", False))
    solver_seconds = _finite_float(
        stats.get("t_wall_total"), solve_wall_seconds
    )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "runner": RUNNER,
        "problem": model.name,
        "model_path": str(problem_path),
        "model_sha256": _sha256(problem_path),
        "casadi_version": str(ca.__version__),
        "status": str(stats.get("return_status", "unknown")),
        "unified_status": str(stats.get("unified_return_status", "unknown")),
        "success": success,
        "kkt_passed": kkt_passed,
        "objective": objective,
        "casadi_objective": float(solution["f"]),
        "iterations": int(stats.get("iter_count", 0) or 0),
        "solver_seconds": solver_seconds,
        "solve_wall_seconds": solve_wall_seconds,
        "parse_seconds": parse_seconds,
        "model_build_seconds": build_seconds,
        "total_wall_seconds": time.perf_counter() - total_start,
        "dimensions": {
            "variables": model.n,
            "equalities": equality_count,
            "inequalities": inequality_count,
            "jacobian_structural_nnz": model.jacobian_structural_nnz(),
            "lagrangian_hessian_structural_nnz": (
                model.lagrangian_hessian_structural_nnz()
            ),
        },
        "initial_objective": model.objective_value(model.x0),
        "initial_residuals": initial_residuals,
        "residuals": residuals,
        "tolerances": {
            "overall": args.tol,
            "constraint": args.constr_viol_tol,
            "dual": args.dual_inf_tol,
            "complementarity": args.compl_inf_tol,
        },
        "ipopt_options": {
            "linear_solver": args.linear_solver,
            "max_iter": args.max_iter,
            "max_cpu_time": args.time_limit,
            "mu_strategy": "adaptive",
            "hessian_approximation": "exact",
            "nlp_scaling_method": "gradient-based",
            "bound_relax_factor": 0.0,
            "fixed_variable_treatment": "make_parameter",
        },
        "constraint_multipliers": constraint_multipliers,
        "bound_multipliers": bound_multipliers,
    }
    if not args.no_solution_vector:
        payload["x"] = x_final
    _write_json(output_path, payload)

    print("\n[IPOPT EXACT QCQP SUMMARY]")
    print(f"problem       : {model.name}")
    print(f"model sha256  : {payload['model_sha256']}")
    print(f"dimensions    : {model.n}/{equality_count}/{inequality_count}")
    print(f"status        : {payload['status']}")
    print(f"objective     : {objective:.17g}")
    print(f"equality      : {primal['equality_inf']:.17g}")
    print(f"inequality    : {primal['inequality_inf']:.17g}")
    print(f"bounds        : {primal['bound_inf']:.17g}")
    print(f"dual          : {kkt['dual_inf']:.17g}")
    print(f"complementarity: {kkt['complementarity_inf']:.17g}")
    print(f"iterations    : {payload['iterations']}")
    print(f"solver_s      : {solver_seconds:.9f}")
    print(f"solve_wall_s  : {solve_wall_seconds:.9f}")
    print(f"build_s       : {build_seconds:.9f}")
    print(f"KKT           : {'PASS' if kkt_passed else 'FAIL'}")
    print(f"result        : {output_path}")
    return 0 if success and kkt_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
