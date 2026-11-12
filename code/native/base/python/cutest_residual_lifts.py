"""Exact sparse residual lifts for polynomial CUTEst least-squares models.

Sources: the public ralna/SIF EIGENA2, EIGENB2, EIGENC2 and PATTERNLS SIFs.
All model expressions are checked against the *loaded*, unmodified CUTEst
value and gradient oracle at the original start and several other points.
The EIGEN route deliberately needs no individual Hessian native calls.

Quadratics use ``constant + linear @ x + .5*x.T@H@x``; linear is a CSR row.
No dense array with three dimensions, nor quartic objective Hessian, is built.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping

import numpy as np
import scipy.sparse as sp

SUPPORTED_RESIDUAL_LIFTS = frozenset({"PATTERNLS", "EIGENA2", "EIGENB2", "EIGENC2"})
VALUE_TOL = 5e-9
GRADIENT_TOL = 5e-8


def _helper(helpers, name):
    return helpers[name] if isinstance(helpers, Mapping) else getattr(helpers, name)


def _names(prob, attribute, count):
    if getattr(prob, "_module", None) is not None:
        # CUTEst's current C names ABI writes 11 bytes per name; affected
        # PyCUTEst wrappers allocate a 10-byte stride and corrupt native memory.
        # The safe helper reads the native names into a sufficiently sized
        # buffer and checks the ABI.  Do not call the unsafe wrapper first.
        from safe_cutest_names import safe_names
        result = safe_names(prob, attribute)
    else:
        # Pure Python independent test oracles may expose lists or methods.
        result = getattr(prob, attribute, None)
        if callable(result):
            result = result()
    if result is None:
        raise RuntimeError(f"{prob.name}: {attribute} are required to verify the structured lift")
    result = [re.sub(r"[\s()]", "", str(v)).upper() for v in result]
    if len(result) != count or len(set(result)) != count:
        raise RuntimeError(f"{prob.name}: invalid {attribute} count or duplicate names")
    return result


def _row(indices, values, size):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    out = sp.csr_matrix((values, (np.zeros(len(indices), dtype=np.int64), indices)), shape=(1, size))
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def _hessian(n, terms):
    """Build a symmetric Hessian from monomial (i,j,coefficient) triples."""
    rows, cols, values = [], [], []
    for i, j, coefficient in terms:
        if coefficient == 0.0:
            continue
        if i == j:
            rows.append(i); cols.append(i); values.append(2.0 * coefficient)
        else:
            rows.extend((i, j)); cols.extend((j, i)); values.extend((coefficient, coefficient))
    out = sp.coo_matrix((values, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def _eval(q, x):
    _, constant, linear, hessian = q
    hx = np.asarray(hessian @ x).ravel()
    return float(constant + (linear @ x)[0] + 0.5 * x @ hx), np.asarray(linear.toarray()).ravel() + hx


def _oracle_objective(prob, x):
    # Copy outputs before any subsequent native call: some wrappers reuse buffers.
    value, grad = prob.obj(np.ascontiguousarray(x, dtype=np.float64), gradient=True)
    value = float(value)
    grad = np.array(grad, dtype=np.float64, copy=True).reshape(int(prob.n))
    if not math.isfinite(value) or not np.all(np.isfinite(grad)):
        raise RuntimeError(f"{prob.name}: non-finite objective oracle during lift audit")
    return value, grad


def _oracle_constraints(prob, x):
    method = getattr(prob, "scons", None)
    if not callable(method):
        method = prob.cons
    values, jacobian = method(np.ascontiguousarray(x, dtype=np.float64), gradient=True)
    values = np.array(values, dtype=float, copy=True).reshape(int(prob.m))
    jacobian = sp.csr_matrix(jacobian, shape=(int(prob.m), int(prob.n)), dtype=float, copy=True)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(jacobian.data)):
        raise RuntimeError(f"{prob.name}: non-finite constraint oracle during lift audit")
    return values, jacobian


def _audit_points(x0, bl, bu, helpers):
    points = [np.array(x0, dtype=float, copy=True)]
    points += [np.array(x, dtype=float, copy=True)
               for _, x in _helper(helpers, "_quadratic_probe_points")(x0, bl, bu)]
    # Alternating probes alone have cancellation symmetries.  Add a reproducible
    # nonuniform probe, and respect fixed bounds without changing the real x0.
    rng = np.random.default_rng(917032)
    x = x0 + 0.37 * np.maximum(1.0, np.abs(x0)) * rng.uniform(-1.0, 1.0, len(x0))
    finite_lo = np.isfinite(bl) & (bl > -1e19)
    finite_up = np.isfinite(bu) & (bu < 1e19)
    x[finite_lo] = np.maximum(x[finite_lo], bl[finite_lo])
    x[finite_up] = np.minimum(x[finite_up], bu[finite_up])
    if not any(np.array_equal(x, other) for other in points):
        points.append(x)
    return points


def _audit(prob, points, residuals, original_constraints):
    worst_value = worst_gradient = worst_constraint_value = worst_constraint_gradient = 0.0
    for x in points:
        f, g = _oracle_objective(prob, x)
        pf = 0.0
        pg = np.zeros(int(prob.n))
        for q in residuals:
            r, dr = _eval(q, x)
            pf += r * r
            pg += (2.0 * r) * dr
        # Normalize values and gradients separately.  A large objective must not
        # hide an incorrect derivative or transpose in the structured formula.
        worst_value = max(worst_value, abs(f - pf) / max(1.0, abs(f), abs(pf)))
        worst_gradient = max(worst_gradient, float(np.linalg.norm(g - pg, np.inf)) /
                             max(1.0, float(np.linalg.norm(g, np.inf)), float(np.linalg.norm(pg, np.inf))))
        if original_constraints:
            values, jacobian = _oracle_constraints(prob, x)
            for i, q in enumerate(original_constraints):
                cv, cg = _eval(q, x)
                actual_g = jacobian.getrow(i).toarray().ravel()
                worst_constraint_value = max(worst_constraint_value,
                    abs(cv - values[i]) / max(1.0, abs(cv), abs(float(values[i]))))
                worst_constraint_gradient = max(worst_constraint_gradient,
                    float(np.linalg.norm(cg - actual_g, np.inf)) /
                    max(1.0, float(np.linalg.norm(cg, np.inf)), float(np.linalg.norm(actual_g, np.inf))))
    if max(worst_value, worst_constraint_value) > VALUE_TOL or max(worst_gradient, worst_constraint_gradient) > GRADIENT_TOL:
        raise RuntimeError(f"{prob.name}: exact residual-lift oracle audit failed "
            f"(objective value={worst_value:.3e}, gradient={worst_gradient:.3e}; "
            f"constraint value={worst_constraint_value:.3e}, gradient={worst_constraint_gradient:.3e})")
    return {"probes": len(points), "objective_value_error": worst_value,
            "objective_gradient_error": worst_gradient, "constraint_value_error": worst_constraint_value,
            "constraint_gradient_error": worst_constraint_gradient,
            "value_tolerance": VALUE_TOL, "gradient_tolerance": GRADIENT_TOL,
            "native_individual_hessian_calls": 0}


def _eigen_formulas(prob, x0):
    n = int(prob.n)
    N = int(round((math.sqrt(1 + 4 * n) - 1) / 2))
    if N < 1 or N * (N + 1) != n or int(prob.m) != N * (N + 1) // 2:
        raise RuntimeError(f"{prob.name}: dimensions do not match the EIGEN*2 SIF contract")
    names = _names(prob, "varnames", n)
    positions = {name: j for j, name in enumerate(names)}
    expected = {f"D{j + 1}" for j in range(N)} | {f"Q{i + 1},{j + 1}" for j in range(N) for i in range(N)}
    if set(names) != expected:
        raise RuntimeError(f"{prob.name}: variable names do not match D(j), Q(i,j)")
    didx = [positions[f"D{j + 1}"] for j in range(N)]
    qidx = np.array([[positions[f"Q{i + 1},{j + 1}"] for j in range(N)] for i in range(N)], dtype=int)
    name = str(prob.name).strip().upper()
    if name == "EIGENA2":
        diagonal, off_diagonal = np.arange(1, N + 1, dtype=float), 0.0
    elif name == "EIGENB2":
        diagonal, off_diagonal = np.full(N, 2.0), -1.0
    elif name == "EIGENC2":
        if N % 2 != 1:
            raise RuntimeError("EIGENC2 expects an odd matrix dimension")
        diagonal, off_diagonal = (N - 1) / 2 - np.arange(N, dtype=float), 1.0
    else:
        raise RuntimeError(f"unsupported residual model {name}")
    residuals = []
    # E(i,j) = Q(j,i)D(j) - sum_k A(i,k)Q(j,k).
    # Q's first and second index must not be swapped: the SIF uses Q transpose.
    for j in range(N):
        for i in range(N):
            indices, coefficients = [qidx[j, i]], [-diagonal[i]]
            if off_diagonal and i:
                indices.append(qidx[j, i - 1]); coefficients.append(-off_diagonal)
            if off_diagonal and i + 1 < N:
                indices.append(qidx[j, i + 1]); coefficients.append(-off_diagonal)
            residuals.append((f"E({i + 1},{j + 1})", 0.0, _row(indices, coefficients, n),
                              _hessian(n, [(didx[j], qidx[j, i], 1.0)])))
    names_c = _names(prob, "connames", int(prob.m))
    initial_values, _ = _oracle_constraints(prob, x0)
    constraints = []
    observed_pairs = set()
    for row, cname in enumerate(names_c):
        match = re.fullmatch(r"O(\d+),(\d+)", cname)
        if match is None:
            raise RuntimeError(f"{name}: unknown orthogonality constraint {cname}")
        i, j = (int(v) - 1 for v in match.groups())
        if not 0 <= i <= j < N or (i, j) in observed_pairs:
            raise RuntimeError(f"{name}: invalid orthogonality constraint {cname}")
        observed_pairs.add((i, j))
        H = _hessian(n, [(qidx[k, i], qidx[k, j], 1.0) for k in range(N)])
        # CUTEst may carry the SIF RHS in the group constant or the bounds.
        # Learn this additive constant at x0, then audit it elsewhere.
        constant = float(initial_values[row] - 0.5 * x0 @ (H @ x0))
        constraints.append((cname, constant, _row([], [], n), H))
    return residuals, constraints, {"matrix_dimension": N}


def _pattern_formulas(prob, m):
    n = int(prob.n)
    if n < 1 or int(prob.m) != 0:
        raise RuntimeError("PATTERNLS must be the unconstrained SIF least-squares model")
    names = _names(prob, "varnames", n)
    expected = {f"A{i}" for i in range(n)}
    if set(names) != expected:
        raise RuntimeError("PATTERNLS variable names do not match a(0), ..., a(N)")
    index = {int(name[1:]): pos for pos, name in enumerate(names)}
    # Each residual contains O(N) products; total Hessian storage is O(N^2),
    # not a dense tensor of O(N^3).  Original variable ordering is preserved.
    residuals = []
    N = n - 1
    angle = float(m) * (math.pi / 3.0)
    for k in range(n):
        terms = [(index[j], index[k + j], 2.0 * math.cos(angle * (k - j)))
                 for j in range(1, N - k + 1)]
        terms += [(index[j], index[k - j], math.cos(angle * (k - 2 * j)))
                  for j in range(k + 1)]
        residuals.append((f"C({k})", 0.0, _row([index[k]], [-1.0], n), _hessian(n, terms)))
    return residuals, [], {"N": N, "m": int(m)}


def _pad_hessian(H, n):
    C = H.tocoo()
    return sp.coo_matrix((C.data.copy(), (C.row, C.col)), shape=(n, n)).tocsr()


def _pad_linear(linear, n):
    return _row(linear.indices, linear.data, n)


def _canonicalize_original(prob, constraints, n, helpers):
    eq, ineq = [], []
    lower = _helper(helpers, "_finite_lower")
    upper = _helper(helpers, "_finite_upper")
    for i, (_, constant, linear, H) in enumerate(constraints):
        lo, up = float(prob.cl[i]), float(prob.cu[i])
        lo_ok, up_ok = lower(lo), upper(up)
        linear, H = _pad_linear(linear, n), _pad_hessian(H, n)
        is_equality = bool(prob.is_eq_cons[i]) or (lo_ok and up_ok and
            abs(up - lo) <= 1e-12 * max(1.0, abs(lo), abs(up)))
        if is_equality:
            if not (lo_ok or up_ok):
                raise RuntimeError(f"CUTEst equality {i} has no finite RHS")
            rhs = 0.5 * (lo + up) if lo_ok and up_ok else (lo if lo_ok else up)
            eq.append((f"c{i}_eq", constant - rhs, linear, H))
        else:
            if up_ok:
                ineq.append((f"c{i}_upper", constant - up, linear, H))
            if lo_ok:
                ineq.append((f"c{i}_lower", lo - constant, -linear, -H))
    return eq, ineq


def build_residual_lift(prob, x0, bl, bu, helpers):
    """Return an independently audited exact QCQP, preserving x0's first n entries."""
    name = str(prob.name).strip().upper()
    if name not in SUPPORTED_RESIDUAL_LIFTS:
        raise ValueError(f"unsupported residual lift {name}")
    n0 = int(prob.n)
    x0 = np.array(x0, dtype=np.float64, copy=True).reshape(n0)
    bl = np.array(bl, dtype=float, copy=True).reshape(n0)
    bu = np.array(bu, dtype=float, copy=True).reshape(n0)
    if not np.all(np.isfinite(x0)):
        raise RuntimeError(f"{name}: initial point is not finite")
    if name == "PATTERNLS":
        parameters = getattr(prob, "sifParams", None) or {}
        raw_m = parameters.get("m", 6)
        if not math.isfinite(float(raw_m)) or int(raw_m) != float(raw_m):
            raise RuntimeError("PATTERNLS SIF parameter m must be integral")
        residuals, constraints, params = _pattern_formulas(prob, int(raw_m))
    else:
        residuals, constraints, params = _eigen_formulas(prob, x0)
    audit = _audit(prob, _audit_points(x0, bl, bu, helpers), residuals, constraints)
    nr = len(residuals)
    n = n0 + nr
    xnew = np.empty(n, dtype=float)
    xnew[:n0] = x0
    for k, residual in enumerate(residuals):
        xnew[n0 + k] = _eval(residual, x0)[0]
    eq, ineq = _canonicalize_original(prob, constraints, n, helpers)
    original_equality_count = len(eq)
    for k, (_, constant, linear, H) in enumerate(residuals):
        indices = np.concatenate((linear.indices, [n0 + k]))
        coefficients = np.concatenate((-linear.data, [1.0]))
        eq.append((f"residual_definition_{k}", -constant, _row(indices, coefficients, n), -_pad_hessian(H, n)))
    objective_hessian = sp.diags(np.concatenate((np.zeros(n0), np.full(nr, 2.0))), format="csr")
    mapping = {"kind": "quadratic_residuals", "original_n": n0,
        "original_variable_indices": list(range(n0)), "auxiliary_indices": list(range(n0, n)),
        "original_constraint_count": int(prob.m), "original_equality_count": original_equality_count,
        "residual_count": nr, "parameters": params,
        "auxiliary_initialization": "r[k] = original quadratic residual q[k](x0)",
        "original_x0_preserved": bool(np.array_equal(xnew[:n0], x0))}
    return {"n": n, "x0": xnew, "lb": np.concatenate((bl, np.full(nr, -np.inf))),
        "ub": np.concatenate((bu, np.full(nr, np.inf))), "obj_const": 0.0,
        "obj_linear": _row([], [], n), "H0": objective_hessian,
        "equalities": eq, "inequalities": ineq,
        "reformulation": "patternls_residual_exact_qcqp" if name == "PATTERNLS" else "eigen2_residual_exact_qcqp",
        "n_original": n0, "abs_lifts": 0, "lift_variables": list(range(n0, n)),
        "lift_validation_value": max(audit["objective_value_error"], audit["constraint_value_error"]),
        "lift_validation_gradient": max(audit["objective_gradient_error"], audit["constraint_gradient_error"]),
        "lift_validation": audit, "lift_mapping": mapping}


def build_prodpl1_direct(prob, x0, bl, bu, helpers):
    """Reconstruct PRODPL1 using its exact value/gradient coefficient.

    The SIF's Fortran ``0.1`` is a single-precision literal.  The value and
    gradient use alpha=0.10000000149011612, whereas the constant expression
    ``2.0*(1.0-0.1)`` in its supplied Hessian rounds to 1.7999999523162842.
    These are inconsistent derivatives; raising the generic quadratic check
    tolerance would hide the issue.  Recover alpha from the known quadratic
    SIF expression and independently verify all values/gradients instead.
    No idealized decimal coefficient is substituted for the loaded oracle.
    """
    if str(prob.name).strip().upper() != "PRODPL1" or int(prob.n) != 60 or int(prob.m) != 29:
        raise RuntimeError("PRODPL1 dimensions do not match the production-planning SIF")
    n = int(prob.n)
    x0 = np.array(x0, dtype=float, copy=True).reshape(n)
    bl, bu = np.asarray(bl, dtype=float), np.asarray(bu, dtype=float)
    positions = {name: i for i, name in enumerate(_names(prob, "varnames", n))}
    names_c = _names(prob, "connames", int(prob.m))
    production = {}
    for t in range(1, 6):
        keys = [f"X{p:03d}{t:02d}" for p in range(1, 5)]
        if any(key not in positions for key in keys):
            raise RuntimeError("PRODPL1 production variable names do not match the SIF")
        production[t] = [positions[key] for key in keys]
    anchor = np.zeros(n, dtype=float)
    f0, g0 = _oracle_objective(prob, anchor)
    c0, J0 = _oracle_constraints(prob, anchor)
    objective = ("objective", f0, sp.csr_matrix(g0.reshape(1, n)), sp.csr_matrix((n, n)))
    constraints, coefficients, smooth_rows = [], [], []
    for row, cname in enumerate(names_c):
        H = sp.csr_matrix((n, n))
        match = re.fullmatch(r"SMOOTH\(?(\d+)\)?", cname)
        if match:
            t = int(match.group(1))
            if not 1 <= t <= 4 or t in smooth_rows:
                raise RuntimeError("PRODPL1 smoothing rows do not match the SIF")
            smooth_rows.append(t)
            test = anchor.copy()
            test[production[t][0]] = 1.0
            cv, Jv = _oracle_constraints(prob, test)
            # At (u,v)=(0,1), q=1-alpha and dq/dv=2(1-alpha).
            alpha_value = 1.0 - (float(cv[row]) - float(c0[row]) - float((J0.getrow(row) @ test)[0]))
            alpha_gradient = 1.0 - 0.5 * (float(Jv[row, production[t][0]]) - float(J0[row, production[t][0]]))
            if (not math.isfinite(alpha_value) or abs(alpha_value - alpha_gradient) > 5e-13
                    or abs(alpha_value - 0.1) > 1e-7):
                raise RuntimeError(f"PRODPL1 smoothing coefficient fails value/gradient agreement at {cname}")
            alpha = alpha_value
            v, u = production[t], production[t + 1]
            rows, cols, values = [], [], []
            for block_i, block_j, value in [(u, u, 2.0), (v, v, 2.0 * (1.0 - alpha)),
                                          (u, v, -2.0), (v, u, -2.0)]:
                for i in block_i:
                    for j in block_j:
                        rows.append(i); cols.append(j); values.append(value)
            H = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
            coefficients.append({"constraint_index": row, "constraint_name": cname,
                                 "coefficient_from_values": alpha_value, "coefficient_from_gradients": alpha_gradient})
        constraints.append((cname, float(c0[row]), J0.getrow(row).copy(), H))
    if sorted(smooth_rows) != [1, 2, 3, 4]:
        raise RuntimeError("PRODPL1 expects exactly four smoothing constraints")
    points = _audit_points(x0, bl, bu, helpers)
    worst_value = worst_gradient = 0.0
    for x in points:
        of, og = _oracle_objective(prob, x)
        pf, pg = _eval(objective, x)
        worst_value = max(worst_value, abs(of - pf) / max(1.0, abs(of), abs(pf)))
        worst_gradient = max(worst_gradient, float(np.linalg.norm(og - pg, np.inf)) /
                             max(1.0, float(np.linalg.norm(og, np.inf)), float(np.linalg.norm(pg, np.inf))))
        cv, Jv = _oracle_constraints(prob, x)
        for i, q in enumerate(constraints):
            pv, pg = _eval(q, x)
            actual_g = Jv.getrow(i).toarray().ravel()
            worst_value = max(worst_value, abs(cv[i] - pv) / max(1.0, abs(float(cv[i])), abs(pv)))
            worst_gradient = max(worst_gradient, float(np.linalg.norm(actual_g - pg, np.inf)) /
                                 max(1.0, float(np.linalg.norm(actual_g, np.inf)), float(np.linalg.norm(pg, np.inf))))
    if worst_value > VALUE_TOL or worst_gradient > GRADIENT_TOL:
        raise RuntimeError(f"PRODPL1 exact reconstruction fails value/gradient audit "
                           f"(value={worst_value:.3e}, gradient={worst_gradient:.3e})")
    # Constant-Hessian audit is separate from value/gradient agreement.  The
    # diagnosed native Hessian inconsistency is recorded explicitly, not hidden.
    native_disagreement = 0.0
    native_variation = 0.0
    for i in [None] + list(range(int(prob.m))):
        def native(x):
            result = prob.isphess(np.ascontiguousarray(x), cons_index=i) if i is not None else prob.isphess(np.ascontiguousarray(x))
            result = sp.csr_matrix(result, dtype=float, copy=True)
            if result.shape != (n, n) or not np.all(np.isfinite(result.data)):
                raise RuntimeError("PRODPL1 native Hessian has invalid shape or values")
            return result
        native_anchor = native(anchor)
        recovered = objective[3] if i is None else constraints[i][3]
        delta = (native_anchor - recovered).tocsr()
        native_disagreement = max(native_disagreement, float(np.max(np.abs(delta.data), initial=0.0)))
        for xp in (points[-1], points[-2]):
            delta = (native(xp) - native_anchor).tocsr()
            native_variation = max(native_variation, float(np.max(np.abs(delta.data), initial=0.0)))
    if native_variation > 1e-12:
        raise RuntimeError(f"PRODPL1 native Hessian is not constant: {native_variation:.3e}")
    # Reject unrelated source changes rather than silently overriding them.
    if native_disagreement > 1e-6:
        raise RuntimeError(f"PRODPL1 native Hessian discrepancy exceeds the known literal-rounding defect: {native_disagreement:.3e}")
    eq, ineq = _canonicalize_original(prob, constraints, n, helpers)
    audit = {"probes": len(points), "value_error": worst_value, "gradient_error": worst_gradient,
             "native_hessian_variation_max_entry": native_variation,
             "native_hessian_inconsistency_max_entry": native_disagreement,
             "reconstruction_anchor": "zero", "original_x0_preserved": True,
             "smoothing_coefficients": coefficients}
    return {"n": n, "n_original": n, "x0": x0, "lb": bl.copy(), "ub": bu.copy(),
            "obj_const": f0, "obj_linear": objective[2], "H0": objective[3],
            "equalities": eq, "inequalities": ineq, "reformulation": "prodpl1_exact_qcqp",
            "abs_lifts": 0, "lift_variables": [], "lift_validation": audit,
            "lift_validation_value": worst_value, "lift_validation_gradient": worst_gradient,
            "lift_mapping": {"kind": "direct_quadratic", "original_n": n,
                "original_variable_indices": list(range(n)), "auxiliary_indices": [],
                "original_constraint_count": int(prob.m), "original_x0_preserved": True}}
