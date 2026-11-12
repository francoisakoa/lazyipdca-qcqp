"""Exact, sparse CUTEst lifts for BT4, OPTCDEG3 and ORTHREG[A-C].

The oracle remains authoritative: the prescribed cubic structure is removed
from analytic derivatives, its quadratic remainder is recovered, and values,
gradients AND Hessians are checked at multiple points. No Hessian is densified.
Original variables retain their oracle order, bounds and initial values.

Quadratic tuples use sparse (1,n) linear rows and COO Hessians. Keeping every
constraint's linear vector/CSR row pointers of length n would cost O(m*n).
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

SUPPORTED = frozenset({"BT4", "OPTCDEG3", "ORTHREGA", "ORTHREGB", "ORTHREGC"})
_TOL = 1.0e-7


def _add(poly, key, value):
    key = tuple(sorted(key))
    poly[key] = poly.get(key, 0.0) + float(value)
    if poly[key] == 0.0:
        del poly[key]


def _plus(*polys):
    out = {}
    for poly in polys:
        for key, value in poly.items():
            _add(out, key, value)
    return out


def _scale(poly, value):
    return {key: value * c for key, c in poly.items() if value * c != 0.0}


def _mul(a, b):
    out = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            _add(out, ka + kb, va * vb)
    return out


def _poly_data(poly, x, n):
    """Exact monomial differentiation without division by coordinates."""
    value = 0.0
    grad = np.zeros(n)
    hh = defaultdict(float)
    for key, coef in poly.items():
        value += coef * math.prod(float(x[j]) for j in key)
        for a, i in enumerate(key):
            grad[i] += coef * math.prod(float(x[k]) for b, k in enumerate(key) if b != a)
            for b, j in enumerate(key):
                if a != b:
                    hh[i, j] += coef * math.prod(float(x[k]) for c, k in enumerate(key)
                                               if c != a and c != b)
    if hh:
        ij, vals = zip(*hh.items())
        rows, cols = zip(*ij)
        hess = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
        hess.eliminate_zeros()
    else:
        hess = sp.coo_matrix((n, n))
    return float(value), grad, hess


def _sparse_max(a):
    a = sp.coo_matrix(a)
    a.sum_duplicates()
    return float(np.max(np.abs(a.data), initial=0.0))


def _names(prob, attr, size):
    if getattr(prob, "_module", None) is not None:
        # PyCUTEst's historic width-10 native allocation is unsafe against
        # CUTEst C bridges writing 11 bytes/name. Never call that wrapper.
        from safe_cutest_names import safe_names
        value = safe_names(prob, attr)
    else:
        value = getattr(prob, attr, None)
        if callable(value):
            value = value()  # Pure Python independent test/model providers.
    if value is None or len(value) != size:
        raise RuntimeError(f"{prob.name}: exact lift requires {size} {attr}")
    result = [re.sub(r"\s+", "", str(v)).upper() for v in value]
    # SIFDecode releases expose indexed SIF names either as X(17) or X17.
    # Canonicalize only the indexed families of these named models; H11/G1
    # and BT4's scalar X1/X2/X3 must remain literal parameter names.
    pname = str(prob.name).strip().upper()
    prefixes = ("XYU" if attr == "varnames" else "BC") if pname == "OPTCDEG3" else (
        ("XYZ" if attr == "varnames" else "E") if pname.startswith("ORTHREG") else "")
    if prefixes:
        pattern = re.compile(r"([" + prefixes + r"])([0-9]+)$")
        result = [pattern.sub(r"\1(\2)", name) for name in result]
    if len(set(result)) != len(result):
        raise RuntimeError(f"{prob.name}: duplicate {attr}")
    return result, {name: i for i, name in enumerate(result)}


def _expected_names(actual, expected, label):
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise RuntimeError(f"unexpected {label}: missing={missing}, extra={extra}")


def _oracle(prob, x, index=None):
    if index is None:
        value, grad = prob.obj(x, gradient=True)
        H = prob.isphess(x)
    else:
        value, grad = prob.cons(x, index=index, gradient=True)
        H = prob.isphess(x, cons_index=index)
    grad = np.asarray(grad, dtype=float).reshape(int(prob.n))
    H = sp.coo_matrix(H, shape=(int(prob.n), int(prob.n)), dtype=float)
    H = sp.coo_matrix(0.5 * (H + H.T))
    H.sum_duplicates()
    if not np.isfinite(float(value)) or not np.all(np.isfinite(grad)) or not np.all(np.isfinite(H.data)):
        raise RuntimeError(f"{prob.name}: nonfinite oracle data at {index}")
    return float(value), grad, H


def _quadratic_remainder(prob, x0, cubic, index=None):
    n = len(x0)
    v, g, H = _oracle(prob, x0, index)
    cv, cg, cH = _poly_data(cubic, x0, n)
    H = sp.coo_matrix(H - cH)
    H.sum_duplicates()
    H.eliminate_zeros()
    Hx = np.asarray(H @ x0).ravel()
    linear = g - cg - Hx
    constant = v - cv - 0.5 * float(x0 @ Hx) - float(linear @ x0)
    poly = {(): constant} if constant else {}
    for j in np.flatnonzero(linear):
        _add(poly, (int(j),), linear[j])
    for i, j, a in zip(H.row, H.col, H.data):
        _add(poly, (int(i), int(j)), 0.5 * a)
    return poly


def _quad_tuple(label, poly, n):
    constant = float(poly.get((), 0.0))
    li, lv, hr, hc, hv = [], [], [], [], []
    for key, coef in poly.items():
        if len(key) == 1:
            li.append(key[0]); lv.append(coef)
        elif len(key) == 2:
            i, j = key
            if i == j:
                hr.append(i); hc.append(j); hv.append(2.0 * coef)
            else:
                hr.extend([i, j]); hc.extend([j, i]); hv.extend([coef, coef])
        elif len(key) > 2:
            raise RuntimeError("internal lift produced a nonquadratic expression")
    linear = sp.csr_matrix((lv, ([0] * len(li), li)), shape=(1, n))
    H = sp.coo_matrix((hv, (hr, hc)), shape=(n, n))
    H.sum_duplicates(); H.eliminate_zeros()
    return label, constant, linear, H


def _compose(poly, aux, n0):
    out = {}
    for key, coef in poly.items():
        term = {(): coef}
        for j in key:
            term = _mul(term, {(j,): 1.0} if j < n0 else aux[j - n0])
        out = _plus(out, term)
    return out


def _check_algebra(original, lifted, aux, n0):
    delta = _plus(original, _scale(_compose(lifted, aux, n0), -1.0))
    err = max((abs(v) for v in delta.values()), default=0.0)
    scale = max(1.0, max((abs(v) for v in original.values()), default=0.0))
    if err > 2e-13 * scale:
        raise RuntimeError(f"internal lift identity failed: {err / scale:.3e}")


def _validate(prob, x0, bl, bu, objective, constraints, helpers):
    probe_fn = helpers["_quadratic_probe_points"] if isinstance(helpers, dict) else helpers._quadratic_probe_points
    tests = [(0.0, np.asarray(x0))] + list(probe_fn(x0, bl, bu))
    worst = {"value": 0.0, "gradient": 0.0, "hessian": 0.0}
    for scale, x in tests:
        for index, poly in [(None, objective)] + list(enumerate(constraints)):
            actual_v, actual_g, actual_H = _oracle(prob, x, index)
            pv, pg, pH = _poly_data(poly, x, len(x))
            denom = max(1.0, abs(actual_v), abs(pv), float(np.max(np.abs(actual_g), initial=0)),
                        float(np.max(np.abs(pg), initial=0)))
            errors = {
                "value": abs(actual_v - pv) / denom,
                "gradient": float(np.max(np.abs(actual_g - pg), initial=0)) / denom,
                "hessian": _sparse_max(actual_H - pH) / max(1.0, _sparse_max(actual_H), _sparse_max(pH)),
            }
            for kind, err in errors.items():
                worst[kind] = max(worst[kind], err)
                if not math.isfinite(err) or err > _TOL:
                    raise RuntimeError(f"{prob.name}: exact cubic lift validation failed "
                                       f"({kind}={err:.3e}, constraint={index}, scale={scale:g})")
    return {"probes": len(tests), **worst}


def _structure(prob, x0, var, con):
    """Return cubic originals, their lifted replacements and auxiliary definitions."""
    name = str(prob.name).strip().upper()
    n0, m = int(prob.n), int(prob.m)
    aux, aux_names = [], []
    original = [{} for _ in range(m)]
    lifted = [{} for _ in range(m)]
    objective, obj_lifted = {}, {}

    def add_aux(label, poly):
        idx = n0 + len(aux)
        aux.append(poly); aux_names.append(label)
        return idx

    if name == "BT4":
        _expected_names(var, ["X1", "X2", "X3"], "BT4 variables")
        _expected_names(con, ["CON1", "CON2"], "BT4 constraints")
        j = var["X2"]
        z = add_aux("X2_squared", {(j, j): 1.0})
        objective = {(j, j, j): 1.0}
        obj_lifted = {tuple(sorted((j, z))): 1.0}
    elif name == "OPTCDEG3":
        if n0 < 5 or (n0 - 2) % 3:
            raise RuntimeError("OPTCDEG3: expected n=3*T+2 with fixed variables retained")
        T = (n0 - 2) // 3
        _expected_names(var, [f"{v}({t})" for t in range(T + 1) for v in ("X", "Y")] +
                        [f"U({t})" for t in range(T)], "OPTCDEG3 variables")
        _expected_names(con, [f"{c}({t})" for t in range(T) for c in ("B", "C")], "OPTCDEG3 constraints")
        for t in range(T):
            j, i = var[f"Y({t})"], con[f"C({t})"]
            # Exact analytic Hessian samples recover the SIF-decoded coefficient
            # (the decoder may round 1/T). This is polynomial coefficient
            # recovery, not an approximate Hessian used by the solver.
            xa = np.array(x0, copy=True); xb = np.array(x0, copy=True)
            xa[j] = 0.0; xb[j] = 1.0
            ha = sp.csr_matrix(prob.isphess(xa, cons_index=i))[j, j]
            hb = sp.csr_matrix(prob.isphess(xb, cons_index=i))[j, j]
            coef = float(hb - ha) / 6.0
            if not math.isfinite(coef) or abs(coef * T - 1.0) > 1e-7:
                raise RuntimeError(f"OPTCDEG3: unexpected cubic damping coefficient at t={t}: {coef}")
            z = add_aux(f"Y({t})_squared", {(j, j): 1.0})
            original[i] = {(j, j, j): coef}
            lifted[i] = {tuple(sorted((j, z))): coef}
    else:
        d = 3 if name == "ORTHREGB" else 2
        fixed = d * (d + 1) // 2 + d
        if n0 <= fixed or (n0 - fixed) % d:
            raise RuntimeError(f"{name}: inconsistent point/parameter count")
        points = (n0 - fixed) // d
        expected = [f"H{a+1}{b+1}" for a in range(d) for b in range(a, d)] + [f"G{a+1}" for a in range(d)]
        expected += [f"{v}({i})" for i in range(1, points + 1) for v in "XYZ"[:d]]
        _expected_names(var, expected, f"{name} variables")
        _expected_names(con, [f"E({i})" for i in range(1, points + 1)], f"{name} constraints")
        for point in range(1, points + 1):
            pp = [var[f"{v}({point})"] for v in "XYZ"[:d]]
            i = con[f"E({point})"]
            for a in range(d):
                hp = {}
                for b in range(d):
                    h = var[f"H{min(a,b)+1}{max(a,b)+1}"]
                    _add(hp, (h, pp[b]), 1.0)
                    _add(original[i], (pp[a], h, pp[b]), 1.0)
                z = add_aux(f"HP({point},{a+1})", hp)
                _add(lifted[i], (pp[a], z), 1.0)
    return objective, obj_lifted, original, lifted, aux, aux_names


def build_cubic_lift(prob, x0, bl, bu, helpers):
    """Build an exact model; helpers may be a namespace or exporter globals().

    ``lift_audit`` maps every original equality to its unchanged CUTEst index;
    definitions are appended in auxiliary order. All auxiliary variables are
    free: implied square nonnegativity is NOT added as a redundant bound.
    """
    name = str(prob.name).strip().upper()
    if name not in SUPPORTED:
        raise ValueError(f"unsupported cubic lift: {name}")
    n0, m = int(prob.n), int(prob.m)
    x0 = np.asarray(x0, dtype=float).reshape(n0)
    bl = np.asarray(bl, dtype=float).reshape(n0)
    bu = np.asarray(bu, dtype=float).reshape(n0)
    if not np.all(np.isfinite(x0)) or np.any(np.isnan(bl)) or np.any(np.isnan(bu)) or np.any(bl > bu):
        raise RuntimeError(f"{name}: invalid initial point or bounds")
    var_names, var = _names(prob, "varnames", n0)
    con_names, con = _names(prob, "connames", m)
    cl = np.asarray(prob.cl, dtype=float).reshape(m)
    cu = np.asarray(prob.cu, dtype=float).reshape(m)
    if np.any(~np.isfinite(cl)) or np.any(~np.isfinite(cu)) or np.any(np.abs(cl) >= 1e19) or np.any(np.abs(cu) >= 1e19) or np.any(np.abs(cl-cu) > 1e-12*np.maximum(1.0,np.maximum(np.abs(cl),np.abs(cu)))):
        raise RuntimeError(f"{name}: exact lift requires finite equalities")
    oc, ol, cc, lc, aux, aux_names = _structure(prob, x0, var, con)
    remainder = _quadratic_remainder(prob, x0, oc)
    obj_original, obj_lifted = _plus(remainder, oc), _plus(remainder, ol)
    _check_algebra(obj_original, obj_lifted, aux, n0)
    original, lifted = [], []
    for i in range(m):
        rem = _quadratic_remainder(prob, x0, cc[i], i)
        original.append(_plus(rem, cc[i]))
        lifted.append(_plus(rem, lc[i]))
        _check_algebra(original[-1], lifted[-1], aux, n0)
    check = _validate(prob, x0, bl, bu, obj_original, original, helpers)
    n = n0 + len(aux)
    xnew = np.concatenate([x0, [_poly_data(a, x0, n0)[0] for a in aux]])
    eq = [_quad_tuple(f"c{i}_eq", _plus(p, {(): -float(cl[i])}), n) for i, p in enumerate(lifted)]
    for k, definition in enumerate(aux):
        eq.append(_quad_tuple(f"lift_{k}_def", _plus({(n0+k,): 1.0}, _scale(definition, -1.0)), n))
    _, constant, linear, H = _quad_tuple("objective", obj_lifted, n)
    aux_meta = [{"index": n0+k, "name": aux_names[k], "equality_index": m+k,
                 "polynomial": [{"indices": list(key), "coefficient": float(value)} for key, value in sorted(definition.items())]}
                for k, definition in enumerate(aux)]
    return {"n": n, "n_original": n0, "x0": xnew,
            "lb": np.concatenate([bl, np.full(len(aux), -np.inf)]),
            "ub": np.concatenate([bu, np.full(len(aux), np.inf)]),
            "obj_const": constant, "obj_linear": np.asarray(linear.toarray()).ravel(), "H0": H,
            "equalities": eq, "inequalities": [], "abs_lifts": 0,
            "lift_variables": list(range(n0, n)), "reformulation": name.lower()+"_exact_qcqp",
            "lift_validation_value": check["value"], "lift_validation_gradient": check["gradient"],
            "lift_validation_hessian": check["hessian"], "lift_validation_probes": check["probes"],
            "lift_audit": {"schema": "CUTEST_EXACT_LIFT_1", "problem": name, "original_n": n0,
                           "original_m": m, "original_variable_indices": list(range(n0)),
                           "original_variable_names": var_names, "original_constraint_names": con_names,
                           "original_equalities": [{"cutest_index": i, "equality_index": i, "sign": 1.0} for i in range(m)],
                           "auxiliaries": aux_meta, "initial_original_unchanged": bool(np.array_equal(xnew[:n0], x0)),
                           "source": f"https://raw.githubusercontent.com/ralna/SIF/master/{name}.SIF"}}
