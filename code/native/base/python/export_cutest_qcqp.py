#!/usr/bin/env python3
"""Export CUTEst QCQPs to the LazyIPDCA-QCQP text format.

Direct mode accepts exactly quadratic smooth objectives/constraints.  ``auto``
also applies exact continuous QCQP liftings for selected structured CUTEst
models: CORE1/CORE2 (q*abs(q)), BT2 (quartic monomials through square
auxiliaries), BT4/OPTCDEG3/ORTHREGA/B/C (cubic terms), and
PATTERNLS/EIGENA2/EIGENB2/EIGENC2 (squared quadratic residuals).
PRODPL1 uses its native value/gradient-consistent quadratic coefficients.
New polynomial exports require a separate original-CUTEst KKT audit.
No finite-difference Hessian approximation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

BOUND_INFINITY = 1.0e19
REINTRODUCED = frozenset({"BT4", "OPTCDEG3", "ORTHREGA", "ORTHREGB", "ORTHREGC",
                            "PATTERNLS", "PRODPL1", "EIGENA2", "EIGENB2", "EIGENC2"})
QUADRATIC_CHECK_TOL = 1.0e-7
STRUCTURE_TOL = 1.0e-11

RawConstraint = Tuple[int, float, np.ndarray, sp.csr_matrix, float, float, bool]
QuadraticTuple = Tuple[str, float, np.ndarray, sp.csr_matrix]


def _parse_sif_param(text: str):
    if "=" not in text:
        raise argparse.ArgumentTypeError("SIF parameter must be KEY=VALUE")
    key, raw = text.split("=", 1)
    key, raw = key.strip(), raw.strip()
    if not key:
        raise argparse.ArgumentTypeError("empty SIF parameter name")
    try:
        if raw.lower().startswith(("0x", "+0x", "-0x")):
            value = int(raw, 0)
        elif any(c in raw.lower() for c in (".", "e")):
            value = float(raw)
        else:
            value = int(raw)
    except ValueError:
        value = raw
    return key, value


def _as_symmetric_sparse(H, n: int) -> sp.csr_matrix:
    A = sp.csr_matrix(H, shape=(n, n), dtype=float)
    A = (0.5 * (A + A.T)).tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    return A


def _resize_sparse(A: sp.spmatrix, n_new: int) -> sp.csr_matrix:
    C = sp.coo_matrix(A)
    return sp.coo_matrix((C.data, (C.row, C.col)), shape=(n_new, n_new), dtype=float).tocsr()


def _upper_triplets(A: sp.spmatrix):
    U = sp.triu(A, format="coo")
    order = np.lexsort((U.col, U.row))
    for k in order:
        v = float(U.data[k])
        if v != 0.0:
            yield int(U.row[k]), int(U.col[k]), v


def _recover_quadratic(value: float, grad: np.ndarray, H: sp.spmatrix, x: np.ndarray):
    grad = np.asarray(grad, dtype=float).reshape(-1)
    Hx = np.asarray(H @ x).reshape(-1)
    c = grad - Hx
    constant = float(value - 0.5 * x @ Hx - c @ x)
    return constant, c


def _finite_lower(v: float) -> bool:
    return math.isfinite(float(v)) and float(v) > -BOUND_INFINITY


def _finite_upper(v: float) -> bool:
    return math.isfinite(float(v)) and float(v) < BOUND_INFINITY


def _fmt(v: float) -> str:
    v = float(v)
    if v >= BOUND_INFINITY:
        return "INF"
    if v <= -BOUND_INFINITY:
        return "-INF"
    return f"{v:.17g}"


def _write_quadratic(f, name: str, constant: float, linear: np.ndarray, H: sp.spmatrix, sparse_format=False):
    trips = list(_upper_triplets(H))
    f.write("quadratic\n")
    f.write(f"name {name}\n")
    f.write(f"constant {_fmt(constant)}\n")
    if sparse_format:
        row = sp.csr_matrix(linear).reshape((1, H.shape[0]))
        row.sum_duplicates(); row.eliminate_zeros(); row.sort_indices()
        f.write(f"linear_nnz {row.nnz}\n")
        for j, value in zip(row.indices, row.data):
            f.write(f"{j} {_fmt(value)}\n")
    else:
        values = np.asarray(linear.toarray()).ravel() if sp.issparse(linear) else linear
        f.write("linear " + " ".join(_fmt(v) for v in values) + "\n")
    f.write(f"hessian_nnz {len(trips)}\n")
    for i, j, v in trips:
        f.write(f"{i} {j} {_fmt(v)}\n")
    f.write("end_quadratic\n")


def _bounded_probe_point(
    x0: np.ndarray,
    bl: np.ndarray,
    bu: np.ndarray,
    relative_scale: float,
    phase: int,
) -> np.ndarray:
    """Build a deterministic, bound-feasible validation point.

    A single O(1e-4) perturbation is insufficient to distinguish a quadratic
    from cubic/quartic CUTEst expressions near the expansion point.  This
    helper is used at several scales while keeping fixed variables fixed and
    staying strictly inside finite nonfixed bounds whenever possible.
    """
    x = np.array(x0, dtype=float, copy=True)
    for j in range(x.size):
        lo = float(bl[j]); up = float(bu[j]); xj = float(x0[j])
        lo_fin = _finite_lower(lo); up_fin = _finite_upper(up)
        if lo_fin and up_fin and abs(up-lo) <= 1e-14 * max(1.0, abs(lo), abs(up)):
            x[j] = lo
            continue

        margin = 1e-10 * max(1.0, abs(xj), abs(lo) if lo_fin else 0.0,
                             abs(up) if up_fin else 0.0)
        lower = lo + margin if lo_fin else -math.inf
        upper = up - margin if up_fin else math.inf
        if lower > upper:  # extremely narrow but nonfixed interval
            lower, upper = lo, up

        preferred = 1.0 if ((j + phase) % 2 == 0) else -1.0
        nominal = relative_scale * max(1.0, abs(xj))
        rooms = {
            1.0: max(0.0, upper-xj) if math.isfinite(upper) else math.inf,
            -1.0: max(0.0, xj-lower) if math.isfinite(lower) else math.inf,
        }
        direction = preferred if rooms[preferred] > 0.0 else -preferred
        room = rooms[direction]
        if room <= 0.0:
            continue
        step = nominal if not math.isfinite(room) else min(nominal, 0.5*room)
        if step <= 0.0:
            continue
        x[j] = min(upper, max(lower, xj + direction*step))
    return x


def _probe_point(x0: np.ndarray, bl: np.ndarray, bu: np.ndarray) -> np.ndarray:
    """Historical small probe retained for CORE structural checks."""
    return _bounded_probe_point(x0, bl, bu, 1e-4, 0)


def _quadratic_probe_points(x0: np.ndarray, bl: np.ndarray, bu: np.ndarray):
    specs = ((1e-4, 0), (1e-2, 1), (0.25, 0), (1.0, 1))
    points = []
    for scale, phase in specs:
        x = _bounded_probe_point(x0, bl, bu, scale, phase)
        if np.linalg.norm(x-x0, ord=np.inf) == 0.0:
            continue
        if any(np.array_equal(x, old_x) for _, old_x in points):
            continue
        points.append((scale, x))
    return points

def _evaluate_quad(constant: float, linear: np.ndarray, H: sp.spmatrix, x: np.ndarray):
    Hx = np.asarray(H @ x).reshape(-1)
    return float(constant + linear @ x + 0.5 * x @ Hx), linear + Hx


def _objective_data(prob, x0: np.ndarray, n: int):
    f0, grad0 = prob.obj(x0, gradient=True)
    # For constrained CUTEst problems, sphess(x) is a Lagrangian Hessian and
    # requires multipliers.  isphess(x) is the objective Hessian itself.
    H0 = _as_symmetric_sparse(prob.isphess(x0), n)
    obj_const, obj_linear = _recover_quadratic(float(f0), np.asarray(grad0), H0, x0)
    return float(f0), np.asarray(grad0, dtype=float), H0, obj_const, obj_linear


def _constraint_data(prob, x0: np.ndarray, n: int) -> List[RawConstraint]:
    m = int(prob.m)
    if m == 0:
        return []
    cval0 = np.asarray(prob.cons(x0), dtype=float).reshape(m)
    cl = np.asarray(prob.cl, dtype=float).reshape(m)
    cu = np.asarray(prob.cu, dtype=float).reshape(m)
    is_eq = np.asarray(prob.is_eq_cons, dtype=bool).reshape(m)
    out: List[RawConstraint] = []
    for i in range(m):
        grad_i = np.asarray(prob.grad(x0, index=i), dtype=float).reshape(n)
        Hi = _as_symmetric_sparse(prob.isphess(x0, cons_index=i), n)
        ci_const, ci_linear = _recover_quadratic(cval0[i], grad_i, Hi, x0)
        out.append((i, ci_const, ci_linear, Hi, cl[i], cu[i], bool(is_eq[i])))
    return out


def _constraint_values_and_jacobian(prob, x, m: int, n: int):
    """Use the vector PyCUTEst call when available, with a safe fallback."""
    try:
        values, jac = prob.cons(x, gradient=True)
        values = np.asarray(values, dtype=float).reshape(m)
        if sp.issparse(jac):
            jac = sp.csr_matrix(jac)
            if jac.shape != (m, n):
                raise ValueError("unexpected sparse Jacobian shape")
        else:
            jac = np.asarray(jac, dtype=float).reshape(m, n)
        return values, jac
    except Exception:
        values = np.empty(m, dtype=float)
        jac = np.empty((m, n), dtype=float)
        for i in range(m):
            value, grad = prob.cons(x, index=i, gradient=True)
            values[i] = float(value)
            jac[i, :] = np.asarray(grad, dtype=float).reshape(n)
        return values, jac


def _validate_direct_quadratic(prob, x0, bl, bu, obj_const, obj_linear, H0, constraints):
    probes = _quadratic_probe_points(x0, bl, bu)
    if not probes:
        # A fully fixed model is quadratic on its singleton feasible domain.
        return {"probes": 0, "objective_error": 0.0,
                "constraint_error": 0.0, "worst_constraint": -1}

    worst_obj = 0.0
    worst_con = 0.0
    worst_index = -1
    worst_scale = 0.0
    m = int(prob.m)
    n = int(prob.n)

    for probe_scale, xp in probes:
        fp, gp = prob.obj(xp, gradient=True)
        gp = np.asarray(gp, dtype=float).reshape(n)
        pred_f, pred_g = _evaluate_quad(obj_const, obj_linear, H0, xp)
        obj_scale = max(1.0, abs(float(fp)), abs(pred_f),
                        np.linalg.norm(gp, ord=np.inf),
                        np.linalg.norm(pred_g, ord=np.inf))
        obj_err = max(abs(float(fp)-pred_f),
                      np.linalg.norm(gp-pred_g, ord=np.inf)) / obj_scale
        worst_obj = max(worst_obj, float(obj_err))
        if obj_err > QUADRATIC_CHECK_TOL:
            raise RuntimeError(
                "objective is not numerically quadratic "
                f"(relative reconstruction error {obj_err:.3e}, "
                f"probe scale {probe_scale:g})"
            )

        if m == 0:
            continue
        values, jac = _constraint_values_and_jacobian(prob, xp, m, n)
        for i, ci_const, ci_linear, Hi, *_ in constraints:
            actual_grad = (np.asarray(jac.getrow(i).toarray()).reshape(n)
                           if sp.issparse(jac) else np.asarray(jac[i]).reshape(n))
            pred_value, pred_grad = _evaluate_quad(ci_const, ci_linear, Hi, xp)
            con_scale = max(1.0, abs(float(values[i])), abs(pred_value),
                            np.linalg.norm(actual_grad, ord=np.inf),
                            np.linalg.norm(pred_grad, ord=np.inf))
            err = max(abs(float(values[i])-pred_value),
                      np.linalg.norm(actual_grad-pred_grad, ord=np.inf)) / con_scale
            if err > worst_con:
                worst_con = float(err)
                worst_index = int(i)
                worst_scale = float(probe_scale)

    if worst_con > QUADRATIC_CHECK_TOL:
        raise RuntimeError(
            "constraints are not numerically quadratic "
            f"(worst relative reconstruction error {worst_con:.3e}, "
            f"constraint {worst_index}, probe scale {worst_scale:g})"
        )
    return {"probes": len(probes), "objective_error": worst_obj,
            "constraint_error": worst_con, "worst_constraint": worst_index}

def _canonicalize(raw_constraints: Sequence[RawConstraint], n: int):
    equalities: List[QuadraticTuple] = []
    inequalities: List[QuadraticTuple] = []
    eq_tol = 1e-12
    for i, constant, linear, H, lo, up, flagged_eq in raw_constraints:
        if linear.size != n or H.shape != (n, n):
            raise RuntimeError("internal canonicalization dimension mismatch")
        lo_fin = _finite_lower(lo)
        up_fin = _finite_upper(up)
        equal = lo_fin and up_fin and abs(float(up-lo)) <= eq_tol * max(1.0, abs(float(lo)), abs(float(up)))
        if flagged_eq or equal:
            if not (lo_fin or up_fin):
                raise RuntimeError(f"CUTEst constraint {i} is flagged equality but has no finite RHS")
            rhs = 0.5 * (float(lo)+float(up)) if lo_fin and up_fin else float(lo if lo_fin else up)
            equalities.append((f"c{i}_eq", constant-rhs, linear, H))
            continue
        if up_fin:
            inequalities.append((f"c{i}_upper", constant-float(up), linear, H))
        if lo_fin:
            inequalities.append((f"c{i}_lower", float(lo)-constant, -linear, -H))
    return equalities, inequalities


def _signed_interior_value(lo: float, up: float, sign: int, reference: float) -> float:
    """Return a nonzero bound-feasible point with the requested sign."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if sign < 0:
        return -_signed_interior_value(-up, -lo, 1, -reference)

    lower = float(lo) if _finite_lower(lo) else -math.inf
    upper = float(up) if _finite_upper(up) else math.inf
    a = max(0.0, lower)
    if math.isfinite(upper) and upper <= max(a, 0.0) + 1e-12:
        raise RuntimeError("requested positive branch is excluded by variable bounds")

    preferred = max(1e-4, abs(float(reference)))
    if math.isfinite(upper):
        margin = 1e-10 * max(1.0, abs(a), abs(upper))
        lo_i = max(a + margin, 1e-8)
        hi_i = upper - margin
        if hi_i <= lo_i:
            lo_i = max(a, 1e-12)
            hi_i = upper
        if hi_i <= 0.0:
            raise RuntimeError("requested positive branch is excluded by variable bounds")
        return float(min(hi_i, max(lo_i, preferred)))
    return float(max(a + max(1e-4, 0.25 * max(1.0, abs(a))), preferred))


def _opposite_sign_value(x: float, lo: float, up: float) -> float:
    """Choose a bounded nonzero point of sign opposite to x."""
    return _signed_interior_value(lo, up, -1 if x >= 0.0 else 1, x)


def _single_separable_diagonal(H: sp.spmatrix, constraint_index: int):
    """Return the unique nonlinear diagonal Hessian entry of a CORE row."""
    C = sp.coo_matrix(H)
    off = [(int(r), int(c), float(v)) for r, c, v in zip(C.row, C.col, C.data)
           if r != c and abs(float(v)) > STRUCTURE_TOL]
    diag = [(int(r), float(v)) for r, c, v in zip(C.row, C.col, C.data)
            if r == c and abs(float(v)) > STRUCTURE_TOL]
    if off or len(diag) != 1:
        raise RuntimeError(
            f"CORE structural check: nonlinear constraint {constraint_index} "
            "is not a single separable x*abs(x) term"
        )
    return diag[0]


def _core_xabs_lift(prob, x0, bl, bu, obj_const, obj_linear, H0):
    """Exact continuous QCQP representation for CORE1/CORE2 pipe laws.

    A variable whose feasible interval crosses zero receives the standard
    auxiliary t=|x| and the bilinear term alpha*x*t.  When the variable is
    sign-fixed by its bounds, x*|x| is already a quadratic polynomial on the
    feasible domain (x^2 on x>=0, -x^2 on x<=0); that row is retained directly
    and no artificial abs-link is introduced.
    """
    n = int(prob.n)
    m = int(prob.m)
    is_linear = np.asarray(prob.is_linear_cons, dtype=bool).reshape(m)
    nonlinear = [int(i) for i in np.flatnonzero(~is_linear)]
    if not nonlinear:
        raise RuntimeError("CORE x*abs(x) reformulation requested but CUTEst reports no nonlinear constraints")

    cval0 = np.asarray(prob.cons(x0), dtype=float).reshape(m)
    cl = np.asarray(prob.cl, dtype=float).reshape(m)
    cu = np.asarray(prob.cu, dtype=float).reshape(m)
    is_eq = np.asarray(prob.is_eq_cons, dtype=bool).reshape(m)

    # local[i] = (constant, linear, driver, alpha, direct_hessian, uses_lift)
    local = {}
    detected_lifts = {}
    driver_owner = {}

    for i in range(m):
        gi0 = np.asarray(prob.grad(x0, index=i), dtype=float).reshape(n)
        Hi0 = _as_symmetric_sparse(prob.isphess(x0, cons_index=i), n)
        if is_linear[i]:
            if Hi0.nnz and np.max(np.abs(Hi0.data)) > STRUCTURE_TOL:
                raise RuntimeError(f"CORE structural check: constraint {i} marked linear but Hessian is nonzero")
            const, linear = _recover_quadratic(cval0[i], gi0, sp.csr_matrix((n, n)), x0)
            local[i] = (const, linear, None, None, sp.csr_matrix((n, n)), False)
            continue

        # Identify the driver on a nonzero branch, rather than trusting a
        # potentially ambiguous Hessian returned exactly at x_j=0.
        j0, _ = _single_separable_diagonal(Hi0, i)
        can_negative = (not _finite_lower(bl[j0])) or float(bl[j0]) < -1e-12
        can_positive = (not _finite_upper(bu[j0])) or float(bu[j0]) > 1e-12
        if not can_negative and not can_positive:
            raise RuntimeError(f"CORE structural check: flow variable {j0} is fixed at zero")

        if can_positive:
            xp = np.array(x0, copy=True)
            xp[j0] = _signed_interior_value(bl[j0], bu[j0], +1, x0[j0])
            Hp = _as_symmetric_sparse(prob.isphess(xp, cons_index=i), n)
            jp, hp = _single_separable_diagonal(Hp, i)
        else:
            jp = j0
            hp = math.nan
        if can_negative:
            xn = np.array(x0, copy=True)
            xn[j0] = _signed_interior_value(bl[j0], bu[j0], -1, x0[j0])
            Hn = _as_symmetric_sparse(prob.isphess(xn, cons_index=i), n)
            jn, hn = _single_separable_diagonal(Hn, i)
        else:
            jn = j0
            hn = math.nan

        j = jp if can_positive else jn
        if (can_positive and jp != j) or (can_negative and jn != j):
            raise RuntimeError(f"CORE structural check: branch Hessians use different variables in constraint {i}")
        if j in driver_owner:
            raise RuntimeError(
                f"CORE structural check: nonlinear constraints {driver_owner[j]} and {i} "
                f"share flow variable {j}; expected one pipe-law row per variable"
            )
        driver_owner[j] = i

        crosses_zero = can_negative and can_positive
        if crosses_zero:
            reversal_error = abs(hp + hn)
            scale = max(1.0, abs(hp), abs(hn))
            if reversal_error > 1e-9 * scale:
                raise RuntimeError(
                    f"CORE structural check: Hessian of constraint {i} does not reverse sign across zero"
                )
            alpha = 0.5 * hp
            if not math.isfinite(alpha) or abs(alpha) <= STRUCTURE_TOL:
                raise RuntimeError(f"CORE structural check: invalid x*abs(x) coefficient in constraint {i}")
            # Recover the affine remainder with the branch Hessian consistent
            # with x0 whenever possible.  At x0=0 either branch gives the same
            # value and gradient remainder.
            use_positive = x0[j] >= 0.0
            Hbranch = Hp if use_positive else Hn
            const, linear = _recover_quadratic(cval0[i], gi0, Hbranch, x0)
            local[i] = (const, linear, j, alpha, None, True)
            detected_lifts[j] = alpha
        else:
            # Sign-fixed domain: x*|x| is already exactly quadratic.  Use the
            # Hessian from an interior point on the only feasible branch.
            Hbranch = Hp if can_positive else Hn
            const, linear = _recover_quadratic(cval0[i], gi0, Hbranch, x0)
            local[i] = (const, linear, j, None, Hbranch, False)

    if len(driver_owner) != len(nonlinear):
        raise RuntimeError(
            f"CORE structural check: got {len(driver_owner)} distinct flow variables "
            f"for {len(nonlinear)} nonlinear constraints"
        )

    lift_vars = sorted(detected_lifts)
    lift_pos = {j: n + k for k, j in enumerate(lift_vars)}
    n_new = n + len(lift_vars)

    raw: List[RawConstraint] = []
    original_lifted = []
    for i in range(m):
        const, lin0, j, alpha, direct_H, uses_lift = local[i]
        lin = np.zeros(n_new, dtype=float)
        lin[:n] = lin0
        if uses_lift:
            t = lift_pos[j]
            # In 1/2*y'H*y, symmetric H[j,t]=alpha yields alpha*x_j*t.
            H = sp.coo_matrix(
                ([alpha, alpha], ([j, t], [t, j])), shape=(n_new, n_new), dtype=float
            ).tocsr()
        else:
            H = _resize_sparse(direct_H, n_new)
        item = (i, const, lin, H, cl[i], cu[i], bool(is_eq[i]))
        raw.append(item)
        original_lifted.append((const, lin, H, j, uses_lift))

    equalities, inequalities = _canonicalize(raw, n_new)

    # t_j^2 - x_j^2 = 0, t_j >= 0. This enforces t_j = |x_j| exactly.
    for k, j in enumerate(lift_vars):
        t = n + k
        lin = np.zeros(n_new, dtype=float)
        H = sp.coo_matrix(
            ([-2.0, 2.0], ([j, t], [j, t])), shape=(n_new, n_new), dtype=float
        ).tocsr()
        equalities.append((f"abs_link_x{j}", 0.0, lin, H))

    x0_new = np.concatenate([x0, np.abs(x0[lift_vars])])
    bl_new = np.concatenate([bl, np.zeros(len(lift_vars))])
    bu_new = np.concatenate([bu, np.full(len(lift_vars), np.inf)])
    obj_linear_new = np.concatenate([obj_linear, np.zeros(len(lift_vars))])
    H0_new = _resize_sparse(H0, n_new)

    # Exactness checks on the initial point, a generic bound-feasible point,
    # and a sign-reversed point for every genuinely crossing flow variable.
    tests = [np.array(x0, copy=True), _probe_point(x0, bl, bu)]
    if lift_vars:
        xflip_all = np.array(x0, copy=True)
        for j in lift_vars:
            xflip_all[j] = _opposite_sign_value(x0[j], bl[j], bu[j])
        tests.append(xflip_all)
    worst_value = 0.0
    worst_grad = 0.0
    for xt in tests:
        yt = np.concatenate([xt, np.abs(xt[lift_vars])])
        c_true = np.asarray(prob.cons(xt), dtype=float).reshape(m)
        for i, (const, lin, H, driver, uses_lift) in enumerate(original_lifted):
            pv, gy = _evaluate_quad(const, lin, H, yt)
            scale = max(1.0, abs(float(c_true[i])))
            worst_value = max(worst_value, abs(float(c_true[i])-pv)/scale)
            if i in nonlinear:
                gx = np.array(gy[:n], copy=True)
                if uses_lift and abs(xt[driver]) > 1e-10:
                    t = lift_pos[driver]
                    gx[driver] += gy[t] * (1.0 if xt[driver] > 0.0 else -1.0)
                gtrue = np.asarray(prob.grad(xt, index=i), dtype=float).reshape(n)
                gscale = max(1.0, np.linalg.norm(gtrue, ord=np.inf))
                worst_grad = max(worst_grad, np.linalg.norm(gtrue-gx, ord=np.inf)/gscale)
    if worst_value > 5e-10 or worst_grad > 5e-9:
        raise RuntimeError(
            f"CORE exact-lift validation failed (value={worst_value:.3e}, gradient={worst_grad:.3e})"
        )

    return {
        "n": n_new,
        "x0": x0_new,
        "lb": bl_new,
        "ub": bu_new,
        "obj_const": obj_const,
        "obj_linear": obj_linear_new,
        "H0": H0_new,
        "equalities": equalities,
        "inequalities": inequalities,
        "reformulation": "core_xabs_exact_qcqp",
        "n_original": n,
        "abs_lifts": len(lift_vars),
        "lift_variables": lift_vars,
        "lift_validation_value": worst_value,
        "lift_validation_gradient": worst_grad,
    }



def _validate_constraints_quadratic(prob, x0, bl, bu, constraints):
    """Validate only the CUTEst constraints when the objective is lifted."""
    probes = _quadratic_probe_points(x0, bl, bu)
    if not probes or int(prob.m) == 0:
        return {"probes": len(probes), "constraint_error": 0.0, "worst_constraint": -1}
    n = int(prob.n); m = int(prob.m)
    worst = 0.0; worst_i = -1; worst_scale = 0.0
    for scale, xp in probes:
        values, jac = _constraint_values_and_jacobian(prob, xp, m, n)
        for i, ci_const, ci_linear, Hi, *_ in constraints:
            actual_grad = (np.asarray(jac.getrow(i).toarray()).reshape(n)
                           if sp.issparse(jac) else np.asarray(jac[i]).reshape(n))
            pv, pg = _evaluate_quad(ci_const, ci_linear, Hi, xp)
            denom = max(1.0, abs(float(values[i])), abs(pv),
                        np.linalg.norm(actual_grad, ord=np.inf),
                        np.linalg.norm(pg, ord=np.inf))
            err = max(abs(float(values[i])-pv),
                      np.linalg.norm(actual_grad-pg, ord=np.inf)) / denom
            if err > worst:
                worst, worst_i, worst_scale = float(err), int(i), float(scale)
    if worst > QUADRATIC_CHECK_TOL:
        raise RuntimeError(
            "constraints are not numerically quadratic "
            f"(worst relative reconstruction error {worst:.3e}, "
            f"constraint {worst_i}, probe scale {worst_scale:g})"
        )
    return {"probes": len(probes), "constraint_error": worst, "worst_constraint": worst_i}


def _bt2_exact_lift(prob, x0, bl, bu):
    """Exact six-variable QCQP lifting of CUTEst BT2.

    u=(x2-x3)^2, v=x2^2, w=x3^2.  The lifted objective is
      (x1-1)^2 + (x1-x2)^2 + u^2,
    and the original equality is x1*(1+v)+w^2 = rhs.
    """
    if int(prob.n) != 3 or int(prob.m) != 1:
        raise RuntimeError("BT2 exact lift expects n=3 and m=1")
    n0, n = 3, 6
    x1,x2,x3 = map(float, x0)
    u=(x2-x3)**2; v=x2*x2; w=x3*x3
    xnew=np.array([x1,x2,x3,u,v,w], dtype=float)
    lb=np.concatenate([np.asarray(bl,dtype=float), np.zeros(3)])
    ub=np.concatenate([np.asarray(bu,dtype=float), np.full(3,np.inf)])

    # Objective: 2*x1^2 - 2*x1*x2 + x2^2 - 2*x1 + 1 + u^2.
    lin=np.zeros(n); lin[0]=-2.0
    H=sp.dok_matrix((n,n), dtype=float)
    H[0,0]=4.0; H[1,1]=2.0; H[0,1]=-2.0; H[1,0]=-2.0; H[3,3]=2.0
    H=H.tocsr()

    cl=np.asarray(prob.cl,dtype=float).reshape(1); cu=np.asarray(prob.cu,dtype=float).reshape(1)
    if not (_finite_lower(cl[0]) and _finite_upper(cu[0])):
        raise RuntimeError("BT2 exact lift expects a finite equality RHS")
    constraint_bound=0.5*(float(cl[0])+float(cu[0]))
    # PyCUTEst installations differ in how an SIF equality group constant is
    # exposed: some return the raw element sum with cl=cu=8.2426407, while
    # others return element_sum-8.2426407 with cl=cu=0.  Recover the additive
    # convention from the actual value at x0 and form one invariant RHS for
    # the structured polynomial x1*(1+x2^2)+x3^4.
    cval0=float(prob.cons(np.asarray(x0,dtype=float), index=0))
    structured0=x1*(1.0+x2*x2)+x3**4
    constraint_shift=cval0-structured0
    rhs=constraint_bound-constraint_shift
    eq=[]
    # x1 + x1*v + w^2 - rhs = 0.
    l=np.zeros(n); l[0]=1.0
    C=sp.dok_matrix((n,n), dtype=float)
    C[0,4]=1.0; C[4,0]=1.0; C[5,5]=2.0
    eq.append(("bt2_original_eq", -rhs, l, C.tocsr()))
    # u - (x2-x3)^2 = 0.
    l=np.zeros(n); l[3]=1.0
    C=sp.dok_matrix((n,n), dtype=float)
    C[1,1]=-2.0; C[2,2]=-2.0; C[1,2]=2.0; C[2,1]=2.0
    eq.append(("bt2_u_def", 0.0, l, C.tocsr()))
    # v - x2^2 = 0.
    l=np.zeros(n); l[4]=1.0
    C=sp.dok_matrix((n,n), dtype=float); C[1,1]=-2.0
    eq.append(("bt2_v_def", 0.0, l, C.tocsr()))
    # w - x3^2 = 0.
    l=np.zeros(n); l[5]=1.0
    C=sp.dok_matrix((n,n), dtype=float); C[2,2]=-2.0
    eq.append(("bt2_w_def", 0.0, l, C.tocsr()))

    worst_v=0.0; worst_g=0.0
    tests=[np.asarray(x0,dtype=float)] + [xp for _,xp in _quadratic_probe_points(x0,bl,bu)]
    for xx in tests:
        a,b,c=map(float,xx)
        yy=np.array([a,b,c,(b-c)**2,b*b,c*c])
        pf,gy=_evaluate_quad(1.0,lin,H,yy)
        tf,tg=prob.obj(xx,gradient=True); tg=np.asarray(tg,dtype=float)
        # Chain rule through u,v,w for the lifted objective.
        gx=np.array(gy[:3],copy=True)
        gx[1]+=gy[3]*2*(b-c)+gy[4]*2*b
        gx[2]+=gy[3]*(-2)*(b-c)+gy[5]*2*c
        den=max(1.0,abs(float(tf)),abs(pf),np.linalg.norm(tg,np.inf),np.linalg.norm(gx,np.inf))
        worst_v=max(worst_v,abs(float(tf)-pf)/den)
        worst_g=max(worst_g,np.linalg.norm(tg-gx,np.inf)/den)
        cv,cg=prob.cons(xx,index=0,gradient=True); cg=np.asarray(cg,dtype=float)
        qv,qgy=_evaluate_quad(*eq[0][1:],yy)
        gxq=np.array(qgy[:3],copy=True)
        gxq[1]+=qgy[4]*2*b
        gxq[2]+=qgy[5]*2*c
        actual_residual=float(cv)-constraint_bound
        den=max(1.0,abs(actual_residual),abs(qv),np.linalg.norm(cg,np.inf),np.linalg.norm(gxq,np.inf))
        worst_v=max(worst_v,abs(actual_residual-qv)/den)
        worst_g=max(worst_g,np.linalg.norm(cg-gxq,np.inf)/den)
    if worst_v>5e-10 or worst_g>5e-9:
        raise RuntimeError(f"BT2 exact-lift validation failed (value={worst_v:.3e}, gradient={worst_g:.3e})")
    return {"n":n,"x0":xnew,"lb":lb,"ub":ub,"obj_const":1.0,"obj_linear":lin,"H0":H,
            "equalities":eq,"inequalities":[],"reformulation":"bt2_exact_qcqp",
            "n_original":n0,"abs_lifts":0,"lift_variables":[],
            "lift_validation_value":worst_v,"lift_validation_gradient":worst_g}


def _eigen2_matrix(kind: str, N: int) -> np.ndarray:
    A=np.zeros((N,N),dtype=float)
    if kind == "EIGENA2":
        np.fill_diagonal(A, np.arange(1,N+1,dtype=float))
    elif kind == "EIGENB2":
        np.fill_diagonal(A, 2.0)
        for i in range(N-1): A[i,i+1]=A[i+1,i]=-1.0
    elif kind == "EIGENC2":
        if N % 2 != 1:
            raise RuntimeError("EIGENC2 exact lift expects odd matrix dimension")
        M=(N-1)//2
        np.fill_diagonal(A, np.array([M-j for j in range(N)],dtype=float))
        for i in range(N-1): A[i,i+1]=A[i+1,i]=1.0
    else:
        raise RuntimeError(f"unsupported EIGEN*2 lift: {kind}")
    return A


def _eigen2_residual_lift(prob, x0, bl, bu):
    from cutest_residual_lifts import build_residual_lift
    return build_residual_lift(prob, x0, bl, bu, globals())

def _write_problem(output: Path, name: str, n: int, x0, bl, bu,
                   obj_const, obj_linear, H0, equalities, inequalities, sparse_format=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("LAZYIPDCA_QCQP_V2\n" if sparse_format else "LAZYIPDCA_QCQP_V1\n")
        f.write(f"name {name}\n")
        f.write(f"n {n}\n")
        f.write("x0 " + " ".join(_fmt(v) for v in x0) + "\n")
        f.write("lb " + " ".join(_fmt(v if _finite_lower(v) else -math.inf) for v in bl) + "\n")
        f.write("ub " + " ".join(_fmt(v if _finite_upper(v) else math.inf) for v in bu) + "\n")
        f.write("objective\n")
        _write_quadratic(f, "objective", obj_const, obj_linear, H0, sparse_format)
        f.write(f"equalities {len(equalities)}\n")
        for q in equalities:
            _write_quadratic(f, *q, sparse_format=sparse_format)
        f.write(f"inequalities {len(inequalities)}\n")
        for q in inequalities:
            _write_quadratic(f, *q, sparse_format=sparse_format)
        f.write("end_problem\n")


def export_problem(problem_name: str, output: Path, sif_params: Dict[str, object],
                   check: bool = True, reformulation: str = "auto"):
    import pycutest

    if reformulation not in {"auto", "none", "core-xabs", "bt2-exact", "eigen2-residual", "polynomial-exact"}:
        raise ValueError("reformulation must be auto, none, core-xabs, bt2-exact, eigen2-residual, or polynomial-exact")

    prob = pycutest.import_problem(
        problem_name,
        sifParams=sif_params or None,
        drop_fixed_variables=False,
    )
    n = int(prob.n)
    m = int(prob.m)
    x0 = np.asarray(prob.x0, dtype=float).reshape(n)
    bl = np.asarray(prob.bl, dtype=float).reshape(n)
    bu = np.asarray(prob.bu, dtype=float).reshape(n)

    pname = str(prob.name).strip().upper()
    use_core_lift = reformulation == "core-xabs" or (reformulation == "auto" and pname in {"CORE1", "CORE2"})
    use_bt2_lift = reformulation == "bt2-exact" or (reformulation == "auto" and pname == "BT2")
    use_eigen2_lift = reformulation == "eigen2-residual" or (reformulation == "auto" and pname in {"EIGENA2", "EIGENB2", "EIGENC2"})

    use_polynomial = reformulation == "polynomial-exact" or (reformulation == "auto" and pname in REINTRODUCED)
    if use_polynomial and pname not in REINTRODUCED:
        raise ValueError("No verified exact polynomial transformation for " + pname)
    validation = None
    if use_polynomial:
        if pname in {"BT4", "OPTCDEG3", "ORTHREGA", "ORTHREGB", "ORTHREGC"}:
            from cutest_cubic_lifts import build_cubic_lift
            model = build_cubic_lift(prob, x0, bl, bu, globals())
        elif pname == "PRODPL1":
            from cutest_residual_lifts import build_prodpl1_direct
            model = build_prodpl1_direct(prob, x0, bl, bu, globals())
        else:
            model = _eigen2_residual_lift(prob, x0, bl, bu)
    elif use_bt2_lift:
        # The original objective is quartic.  Do not probe its local Hessian
        # before entering the exact structured lifting route.
        model = _bt2_exact_lift(prob, x0, bl, bu)
    elif use_eigen2_lift:
        model = _eigen2_residual_lift(prob, x0, bl, bu)
    else:
        _, _, H0, obj_const, obj_linear = _objective_data(prob, x0, n)

    if use_core_lift:
        model = _core_xabs_lift(prob, x0, bl, bu, obj_const, obj_linear, H0)
    elif not use_bt2_lift and not use_eigen2_lift and not use_polynomial:
        constraints = _constraint_data(prob, x0, n)
        if check:
            validation = _validate_direct_quadratic(
                prob, x0, bl, bu, obj_const, obj_linear, H0, constraints
            )
        equalities, inequalities = _canonicalize(constraints, n)
        model = {
            "n": n,
            "x0": x0,
            "lb": bl,
            "ub": bu,
            "obj_const": obj_const,
            "obj_linear": obj_linear,
            "H0": H0,
            "equalities": equalities,
            "inequalities": inequalities,
            "reformulation": "none",
            "n_original": n,
            "abs_lifts": 0,
            "lift_variables": [],
        }

    _write_problem(
        output, pname, model["n"], model["x0"], model["lb"], model["ub"],
        model["obj_const"], model["obj_linear"], model["H0"],
        model["equalities"], model["inequalities"],
        sparse_format=use_polynomial or use_eigen2_lift,
    )
    original_audit = None
    if use_polynomial or use_eigen2_lift:
        from cutest_original_audit import original_audit_metadata
        original_audit = original_audit_metadata(prob, model, sif_params)
    return {
        "problem": pname,
        "n_original": int(model["n_original"]),
        "n": int(model["n"]),
        "cutest_constraints": m,
        "equalities": len(model["equalities"]),
        "inequalities": len(model["inequalities"]),
        "objective_hessian_nnz": int(sp.triu(model["H0"]).nnz),
        "reformulation": model["reformulation"],
        "abs_lifts": int(model["abs_lifts"]),
        "lift_variables": [int(j) for j in model["lift_variables"]],
        "lift_validation_value": model.get("lift_validation_value"),
        "lift_validation_gradient": model.get("lift_validation_gradient"),
        "quadratic_validation_probes": None if validation is None else validation["probes"],
        "quadratic_validation_objective_error": None if validation is None else validation["objective_error"],
        "quadratic_validation_constraint_error": None if validation is None else validation["constraint_error"],
        "quadratic_validation_worst_constraint": None if validation is None else validation["worst_constraint"],
        "original_audit": original_audit,
        "input_sha256": hashlib.sha256(output.read_bytes()).hexdigest() if original_audit is not None else None,
        "sif_params": sif_params,
        "lift_validation": model.get("lift_validation"),
        "lift_mapping": model.get("lift_mapping"),
        "lift_audit": model.get("lift_audit"),
        "lift_validation_hessian": model.get("lift_validation_hessian"),
        "output": str(output),
    }


def evaluate_cutest_solution(problem_name: str, x: Sequence[float], sif_params: Dict[str, object]):
    """Evaluate a returned original-variable vector on the unmodified CUTEst model."""
    import pycutest

    prob = pycutest.import_problem(
        problem_name, sifParams=sif_params or None, drop_fixed_variables=False
    )
    x = np.asarray(x, dtype=float).reshape(int(prob.n))
    f = float(prob.obj(x))
    bound_v = 0.0
    for j in range(int(prob.n)):
        if _finite_lower(prob.bl[j]):
            bound_v = max(bound_v, float(prob.bl[j]) - x[j])
        if _finite_upper(prob.bu[j]):
            bound_v = max(bound_v, x[j] - float(prob.bu[j]))
    cons_v = 0.0
    eq_v = 0.0
    ineq_v = 0.0
    if int(prob.m):
        c = np.asarray(prob.cons(x), dtype=float).reshape(int(prob.m))
        eq_flags = np.asarray(prob.is_eq_cons, dtype=bool)
        for i, value in enumerate(c):
            lo = float(prob.cl[i]); up = float(prob.cu[i])
            v = 0.0
            if _finite_lower(lo): v = max(v, lo - float(value))
            if _finite_upper(up): v = max(v, float(value) - up)
            cons_v = max(cons_v, v)
            if eq_flags[i]: eq_v = max(eq_v, v)
            else: ineq_v = max(ineq_v, v)
    total = max(0.0, bound_v, cons_v)
    return {
        "cutest_objective": f,
        "cutest_feasibility": total,
        "cutest_equality_violation": eq_v,
        "cutest_inequality_violation": ineq_v,
        "cutest_bound_violation": bound_v,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("problem")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sif-param", action="append", default=[], type=_parse_sif_param)
    ap.add_argument("--cache", type=Path, help="PyCUTEst cache directory")
    ap.add_argument("--no-check", action="store_true", help="skip direct numerical quadratic reconstruction check")
    ap.add_argument("--reformulation", choices=("auto", "none", "core-xabs", "bt2-exact", "eigen2-residual", "polynomial-exact"), default="auto")
    args = ap.parse_args()

    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
        os.environ["PYCUTEST_CACHE"] = str(args.cache.resolve())
    info = export_problem(
        args.problem, args.output, dict(args.sif_param),
        check=not args.no_check, reformulation=args.reformulation,
    )
    print("[CUTEst export]", " ".join(f"{k}={v}" for k, v in info.items()))


if __name__ == "__main__":
    main()
