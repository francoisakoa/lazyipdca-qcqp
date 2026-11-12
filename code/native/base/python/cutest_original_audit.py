"""Independent original-CUTEst KKT audit for explicitly mapped QCQP exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import scipy.sparse as sp

SCHEMA = "R9_CUTEST_ORIGINAL_AUDIT_1"
BOUND_INFINITY = 1.0e19


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _array(value, size, name):
    a = np.asarray(value, dtype=float).reshape(-1)
    if a.size != size or not np.all(np.isfinite(a)):
        raise ValueError(f"invalid {name}: expected {size} finite numbers")
    return a


def _names(prob, attribute):
    if getattr(prob, "_module", None) is not None:
        from safe_cutest_names import safe_names
        return safe_names(prob, attribute)
    value = getattr(prob, attribute, None)
    if callable(value):
        value = value()
    return None if value is None else [str(x).strip() for x in value]


def oracle_signature(prob):
    """Bind order, dimensions, bounds and native starting point to the export."""
    content = {"name": str(prob.name).strip().upper(), "n": int(prob.n), "m": int(prob.m),
               "varnames": _names(prob, "varnames"),
               "connames": _names(prob, "connames") if int(prob.m) else []}
    for key in ("x0", "bl", "bu", "cl", "cu", "is_eq_cons"):
        source = getattr(prob, key, None)
        values = np.asarray([] if source is None else source, dtype=float).reshape(-1)
        if np.any(np.isnan(values)):
            raise ValueError("NaN in native oracle metadata: " + key)
        content[key] = [float(v).hex() for v in values]
    raw = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def original_audit_metadata(prob, model, sif_params):
    maps = {"equalities": [], "inequalities": []}
    for kind in maps:
        for index, quadratic in enumerate(model[kind]):
            match = re.fullmatch(r"c(\d+)_(eq|upper|lower)", quadratic[0])
            if match:
                maps[kind].append({"index": index, "original_index": int(match[1]),
                                   "side": match[2]})
    metadata = {"schema": SCHEMA, "problem": str(prob.name).strip().upper(), "sif_params": sif_params,
                "n_original": int(prob.n), "m_original": int(prob.m),
                "n_exported": int(model["n"]),
                "equalities_exported": len(model["equalities"]),
                "inequalities_exported": len(model["inequalities"]),
                "oracle_signature_sha256": oracle_signature(prob), **maps}
    _validate_map(prob, metadata)
    return metadata


def _validate_map(prob, metadata):
    expected = set()
    cl, cu = np.asarray(prob.cl), np.asarray(prob.cu)
    eq = np.asarray(prob.is_eq_cons, dtype=bool)
    for i in range(int(prob.m)):
        lower = math.isfinite(float(cl[i])) and cl[i] > -BOUND_INFINITY
        upper = math.isfinite(float(cu[i])) and cu[i] < BOUND_INFINITY
        equality = lower and upper and abs(cu[i] - cl[i]) <= 1e-12 * max(1., abs(cl[i]), abs(cu[i]))
        if eq[i] or equality:
            if not (lower or upper):
                raise ValueError("unbounded equality in original oracle")
            expected.add((i, "eq"))
        else:
            if lower:
                expected.add((i, "lower"))
            if upper:
                expected.add((i, "upper"))
    observed = set()
    for kind in ("equalities", "inequalities"):
        used = set()
        for row in metadata[kind]:
            k, i, side = row["index"], row["original_index"], row["side"]
            if not isinstance(k, int) or not 0 <= k < metadata[kind + "_exported"] or k in used:
                raise ValueError("invalid or repeated exported multiplier index")
            if not isinstance(i, int) or (i, side) not in expected or (i, side) in observed:
                raise ValueError("invalid or repeated original constraint mapping")
            if (kind == "equalities") != (side == "eq"):
                raise ValueError("equality/inequality mapping mismatch")
            used.add(k)
            observed.add((i, side))
    if observed != expected:
        raise ValueError("original constraint mapping is incomplete")


def audit_original(prob, endpoint, metadata, tolerance=1e-6):
    if metadata.get("schema") != SCHEMA or str(prob.name).strip().upper() != metadata.get("problem"):
        raise ValueError("unsupported original-audit contract")
    n, m = int(prob.n), int(prob.m)
    if (n, m) != (metadata["n_original"], metadata["m_original"]):
        raise ValueError("original dimensions differ from export")
    if oracle_signature(prob) != metadata["oracle_signature_sha256"]:
        raise ValueError("original CUTEst order/bounds/starting point changed")
    _validate_map(prob, metadata)
    ne = metadata["n_exported"]
    x = _array(endpoint.get("x", []), ne, "primal endpoint")[:n]
    if endpoint.get("solver") == "Ipopt" or endpoint.get("schema") == "ipopt-direct-qcqp-result-v1":
        if (endpoint.get("solver"), endpoint.get("schema")) != ("Ipopt", "ipopt-direct-qcqp-result-v1"):
            raise ValueError("inconsistent Ipopt endpoint schema")
        keys = ("equality_multipliers", "inequality_multipliers",
                "lower_bound_multipliers", "upper_bound_multipliers")
    else:
        keys = ("certificate_equality_multipliers", "certificate_inequality_multipliers",
                "certificate_lower_multipliers", "certificate_upper_multipliers")
    le = _array(endpoint.get(keys[0], []),
                metadata["equalities_exported"], "equality multipliers")
    li = _array(endpoint.get(keys[1], []),
                metadata["inequalities_exported"], "inequality multipliers")
    zl = _array(endpoint.get(keys[2], []), ne, "lower multipliers")[:n]
    zu = _array(endpoint.get(keys[3], []), ne, "upper multipliers")[:n]
    value, grad = prob.obj(x, gradient=True)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite native objective")
    grad = _array(grad, n, "native objective gradient")
    if m:
        if hasattr(prob, "scons"):
            cv, jac = prob.scons(x, gradient=True)
        else:
            cv, jac = prob.cons(x, gradient=True)
        cv = _array(cv, m, "native constraints")
        jac = sp.csr_matrix(jac)
        if jac.shape != (m, n) or not np.all(np.isfinite(jac.data)):
            raise ValueError("invalid native constraint Jacobian")
    else:
        cv, jac = np.empty(0), sp.csr_matrix((0, n))
    lagrange = np.zeros(m)
    feasibility, complementarity = 0.0, 0.0
    dual_sign = max(0.0, float(np.max(-zl, initial=0)), float(np.max(-zu, initial=0)))
    cl, cu = np.asarray(prob.cl), np.asarray(prob.cu)
    for row in metadata["equalities"]:
        i = row["original_index"]
        lagrange[i] += le[row["index"]]
        if math.isfinite(float(cl[i])) and cl[i] > -BOUND_INFINITY:
            feasibility = max(feasibility, float(cl[i] - cv[i]))
        if math.isfinite(float(cu[i])) and cu[i] < BOUND_INFINITY:
            feasibility = max(feasibility, float(cv[i] - cu[i]))
    for row in metadata["inequalities"]:
        i, multiplier = row["original_index"], float(li[row["index"]])
        side = row["side"]
        slack = float(cu[i] - cv[i]) if side == "upper" else float(cv[i] - cl[i])
        lagrange[i] += multiplier if side == "upper" else -multiplier
        feasibility = max(feasibility, -slack)
        complementarity = max(complementarity, abs(multiplier * slack))
        dual_sign = max(dual_sign, -multiplier)
    for j, xx in enumerate(x):
        lo, up = float(prob.bl[j]), float(prob.bu[j])
        if math.isfinite(lo) and lo > -BOUND_INFINITY:
            feasibility = max(feasibility, lo - xx)
            complementarity = max(complementarity, abs(zl[j] * (xx - lo)))
        elif zl[j] != 0:
            raise ValueError("nonzero multiplier for an absent original lower bound")
        if math.isfinite(up) and up < BOUND_INFINITY:
            feasibility = max(feasibility, xx - up)
            complementarity = max(complementarity, abs(zu[j] * (up - xx)))
        elif zu[j] != 0:
            raise ValueError("nonzero multiplier for an absent original upper bound")
    stationarity = grad + np.asarray(jac.T @ lagrange).reshape(n) - zl + zu
    dual = max(float(np.max(np.abs(stationarity), initial=0)), dual_sign)
    qcq_value = float(endpoint.get("objective", float("nan")))
    if not math.isfinite(qcq_value):
        raise ValueError("non-finite QCQP objective")
    objective_gap = abs(value - qcq_value) / max(1.0, abs(value), abs(qcq_value))
    numbers = (feasibility, dual, complementarity, objective_gap)
    passed = all(math.isfinite(z) and 0 <= z <= tolerance for z in numbers)
    return {"schema": SCHEMA, "problem": str(prob.name).strip().upper(), "strict_pass": passed,
            "objective": value, "feasibility": float(feasibility),
            "dual_infeasibility_inf": dual, "complementarity_inf": float(complementarity),
            "dual_sign_violation": float(dual_sign), "objective_relative_gap": objective_gap,
            "tolerance": tolerance, "n_original": n, "m_original": m,
            "source": "unmodified-native-PyCUTEst-value-gradient-jacobian",
            "endpoint_multiplier_fields": list(keys),
            "multiplier_policy": "mapped-original-rows-and-original-variable-bounds"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--endpoint", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--qcq", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        data = json.loads(args.metadata.read_text())
        if data.get("input_sha256") != digest(args.qcq):
            raise ValueError("QCQ hash differs from original-audit metadata")
        metadata = data["original_audit"]
        if metadata["problem"] != args.problem:
            raise ValueError("requested CUTEst problem differs from export")
        if args.cache:
            os.environ["PYCUTEST_CACHE"] = str(args.cache.resolve())
        import pycutest
        prob = pycutest.import_problem(args.problem, sifParams=metadata["sif_params"] or None,
                                      drop_fixed_variables=False)
        result = audit_original(prob, json.loads(args.endpoint.read_text()), metadata)
        result["input_sha256"] = data["input_sha256"]
        result["metadata_sha256"] = digest(args.metadata)
        result["endpoint_sha256"] = digest(args.endpoint)
        rc = 0 if result["strict_pass"] else 1
    except Exception as exc:
        result = {"schema": SCHEMA, "problem": args.problem, "strict_pass": False,
                  "status": "original_audit_error", "error": f"{type(exc).__name__}: {exc}"}
        rc = 2
    result["audit_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
