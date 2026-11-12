"""Bounded dual-only certificate recovery for exact free-variable graph lifts.

This does not resume optimization or change a primal point. The original native
CUTEst oracle and an independent sparse QCQ reader must both certify the result
at 1e-6 before a separate corrected endpoint can be written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

_PROCESS_STARTED = time.perf_counter()

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsmr

from cutest_original_audit import audit_original, digest

SCHEMA = "R9_LIFT_DUAL_RECOVERY_1"
TOL = 1.0e-6
INF = 1.0e19


class RecoveryDeadline(TimeoutError):
    pass


def _check(deadline):
    if time.perf_counter() >= deadline:
        raise RecoveryDeadline("dual recovery time budget exhausted")


def _finite_vector(values, size, label):
    result = np.asarray(values, dtype=float).reshape(-1)
    if result.size != size or not np.all(np.isfinite(result)):
        raise ValueError("invalid " + label)
    return result


def _fields(endpoint):
    if endpoint.get("solver") == "Ipopt" or endpoint.get("schema") == "ipopt-direct-qcqp-result-v1":
        if (endpoint.get("solver"), endpoint.get("schema")) != ("Ipopt", "ipopt-direct-qcqp-result-v1"):
            raise ValueError("inconsistent Ipopt schema")
        return ("equality_multipliers", "inequality_multipliers", "lower_bound_multipliers", "upper_bound_multipliers")
    return ("certificate_equality_multipliers", "certificate_inequality_multipliers", "certificate_lower_multipliers", "certificate_upper_multipliers")


class _Quadratic:
    def __init__(self, name, constant, linear, hessian):
        self.name, self.constant = name, constant
        self.li = np.array([t[0] for t in linear], dtype=np.int64)
        self.lv = np.array([t[1] for t in linear], dtype=float)
        self.hr = np.array([t[0] for t in hessian], dtype=np.int64)
        self.hc = np.array([t[1] for t in hessian], dtype=np.int64)
        self.hv = np.array([t[2] for t in hessian], dtype=float)
        self.diagonal = self.hr == self.hc

    def value(self, x):
        terms = self.hv * x[self.hr] * x[self.hc]
        terms = np.where(self.diagonal, 0.5 * terms, terms)
        return float(self.constant + np.dot(self.lv, x[self.li]) + np.sum(terms))

    def add_gradient(self, x, dest, multiplier=1.0):
        if multiplier == 0.0:
            return
        np.add.at(dest, self.li, multiplier * self.lv)
        np.add.at(dest, self.hr, multiplier * self.hv * x[self.hc])
        off = ~self.diagonal
        np.add.at(dest, self.hc[off], multiplier * self.hv[off] * x[self.hr[off]])


class _QCQReader:
    """Strict one-pass V1/V2 reader with O(n + total nonzeros) storage."""
    def __init__(self, path, deadline):
        self.deadline = deadline
        self.stream = Path(path).open("r", encoding="utf8")
        self.line = 0

    def get(self, keyword=None):
        _check(self.deadline)
        text = self.stream.readline()
        self.line += 1
        if not text:
            raise ValueError("unexpected QCQ EOF")
        parts = text.strip().split()
        if not parts or (keyword is not None and parts[0] != keyword):
            raise ValueError(f"unexpected QCQ line {self.line}: expected {keyword}")
        return parts

    def scalar(self, keyword, cast=float):
        parts = self.get(keyword)
        if len(parts) != 2:
            raise ValueError("invalid QCQ scalar " + keyword)
        value = cast(parts[1])
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite QCQ " + keyword)
        return value

    def vector(self, keyword, n, finite=True):
        parts = self.get(keyword)
        values = np.asarray([float(v) for v in parts[1:]], dtype=float)
        if values.size != n or np.any(np.isnan(values)) or (finite and not np.all(np.isfinite(values))):
            raise ValueError("invalid QCQ vector " + keyword)
        return values

    def quadratic(self, n):
        self.get("quadratic")
        name = self.scalar("name", str)
        constant = self.scalar("constant")
        row = self.get()
        if row[0] == "linear":
            linear = [(j, float(value)) for j, value in enumerate(row[1:]) if float(value) != 0.0]
            if len(row) != n + 1:
                raise ValueError("invalid dense QCQ linear row")
        elif row[0] in {"linear_nnz", "lin_nnz"} and len(row) == 2:
            count = int(row[1])
            if not 0 <= count <= n:
                raise ValueError("invalid QCQ linear count")
            linear = []
            for _ in range(count):
                item = self.get()
                if len(item) != 2:
                    raise ValueError("invalid QCQ linear entry")
                linear.append((int(item[0]), float(item[1])))
        else:
            raise ValueError("unsupported QCQ linear representation")
        if any(not 0 <= j < n or not math.isfinite(v) for j, v in linear):
            raise ValueError("invalid QCQ linear index/coefficient")
        if len({j for j, _ in linear}) != len(linear):
            raise ValueError("duplicate QCQ linear indices")
        count = self.scalar("hessian_nnz", int)
        if not 0 <= count <= n * (n + 1) // 2:
            raise ValueError("invalid QCQ Hessian count")
        hessian = []
        for _ in range(count):
            item = self.get()
            if len(item) != 3:
                raise ValueError("invalid QCQ Hessian entry")
            i, j, value = int(item[0]), int(item[1]), float(item[2])
            if not 0 <= i <= j < n or not math.isfinite(value):
                raise ValueError("invalid QCQ upper Hessian entry")
            hessian.append((i, j, value))
        self.get("end_quadratic")
        return _Quadratic(name, constant, linear, hessian)

    def read(self):
        try:
            magic = self.get()
            if magic not in (["LAZYIPDCA_QCQP_V1"], ["LAZYIPDCA_QCQP_V2"]):
                raise ValueError("unsupported QCQ schema")
            name = self.scalar("name", str)
            n = self.scalar("n", int)
            if n <= 0:
                raise ValueError("invalid QCQ dimension")
            x0 = self.vector("x0", n)
            lb, ub = self.vector("lb", n, False), self.vector("ub", n, False)
            if np.any(lb > ub):
                raise ValueError("inconsistent QCQ bounds")
            self.get("objective")
            objective = self.quadratic(n)
            me = self.scalar("equalities", int)
            if me < 0:
                raise ValueError("negative equality count")
            equalities = [self.quadratic(n) for _ in range(me)]
            mi = self.scalar("inequalities", int)
            if mi < 0:
                raise ValueError("negative inequality count")
            inequalities = [self.quadratic(n) for _ in range(mi)]
            self.get("end_problem")
            if self.stream.read().strip():
                raise ValueError("trailing QCQ data")
            return {"name": name, "n": n, "x0": x0, "lb": lb, "ub": ub,
                    "objective": objective, "equalities": equalities, "inequalities": inequalities}
        finally:
            self.stream.close()


def _free_bounds(lb, ub):
    return bool(np.all(np.asarray(lb) <= -INF) and np.all(np.asarray(ub) >= INF))


def _qcq_free_bounds(lb, ub):
    # The native QCQ reader/acceptable.hpp defines presence by isfinite.
    # Historic exporter _fmt emitted +INF for both signs of infinity. Such
    # frozen files represent absent bounds in both Lazy and direct Ipopt.
    return bool(np.all(~np.isfinite(lb)) and np.all(~np.isfinite(ub)))


def _audit_qcq(qcq, endpoint, deadline):
    n = qcq["n"]
    fields = _fields(endpoint)
    x = _finite_vector(endpoint.get("x", []), n, "endpoint primal vector")
    lam = _finite_vector(endpoint.get(fields[0], []), len(qcq["equalities"]), "equality multipliers")
    li = _finite_vector(endpoint.get(fields[1], []), len(qcq["inequalities"]), "inequality multipliers")
    zl = _finite_vector(endpoint.get(fields[2], []), n, "lower multipliers")
    zu = _finite_vector(endpoint.get(fields[3], []), n, "upper multipliers")
    if li.size or np.any(zl != 0.0) or np.any(zu != 0.0) or not _qcq_free_bounds(qcq["lb"], qcq["ub"]):
        raise ValueError("QCQ is outside the free equality certificate contract")
    g = np.zeros(n)
    qcq["objective"].add_gradient(x, g)
    f = qcq["objective"].value(x)
    primal = 0.0
    for row, multiplier in zip(qcq["equalities"], lam):
        _check(deadline)
        value = row.value(x)
        if not math.isfinite(value):
            raise ValueError("nonfinite QCQ equality value")
        primal = max(primal, abs(value))
        row.add_gradient(x, g, multiplier)
    if not np.all(np.isfinite(g)):
        raise ValueError("nonfinite QCQ Lagrangian gradient")
    dual = float(np.max(np.abs(g), initial=0))
    reported = float(endpoint.get("objective", float("nan")))
    gap = abs(reported - f) / max(1.0, abs(reported), abs(f))
    numbers = (primal, dual, f, reported, gap)
    valid = all(math.isfinite(v) for v in numbers)
    if not valid:
        raise ValueError("nonfinite QCQ certificate values")
    return {"strict_pass": bool(valid and max(primal, dual, gap) <= TOL),
            "objective": f, "feasibility": primal, "dual_infeasibility_inf": dual,
            "complementarity_inf": 0.0, "dual_sign_violation": 0.0,
            "objective_relative_gap": gap, "tolerance": TOL,
            "source": "independent-sparse-original-QCQ-parser"}


def _graph(qcq, metadata, n0):
    n = qcq["n"]
    mapping = metadata["equalities"]
    original = {row["index"] for row in mapping}
    if len(original) != len(mapping) or any(not 0 <= i < len(qcq["equalities"]) for i in original):
        raise ValueError("invalid original equality mapping")
    graph = []
    assigned = set()
    for i, quad in enumerate(qcq["equalities"]):
        if i in original:
            continue
        if np.any(quad.hr >= n0) or np.any(quad.hc >= n0):
            raise ValueError("graph definition Hessian involves auxiliary variables")
        aux = np.flatnonzero((quad.li >= n0) & (quad.lv != 0.0))
        if aux.size != 1:
            raise ValueError("graph row must define exactly one affine auxiliary")
        j, a = int(quad.li[aux[0]]), float(quad.lv[aux[0]])
        if j in assigned:
            raise ValueError("repeated auxiliary graph definition")
        assigned.add(j); graph.append((i, j, a))
    if assigned != set(range(n0, n)) or len(graph) != n - n0 or not graph:
        raise ValueError("incomplete or empty auxiliary graph")
    return graph


def recover_certificate(prob, endpoint, full_export_metadata, qcq_path, time_limit):
    """Return ``(new_endpoint_or_None, report)``; never modify caller objects."""
    started = time.perf_counter()
    report = {"schema": SCHEMA, "accepted": False, "reason": "not_attempted",
              "fit_seconds": 0.0, "total_seconds": 0.0, "point_unchanged": True,
              "objective_unchanged": True}
    corrected = None
    try:
        if not math.isfinite(float(time_limit)) or time_limit <= 0:
            raise RecoveryDeadline("no dual recovery budget remaining")
        deadline = started + float(time_limit)
        metadata = full_export_metadata["original_audit"]
        report["qcq_sha256"] = digest(qcq_path)
        _check(deadline)
        if full_export_metadata.get("input_sha256") != report["qcq_sha256"]:
            raise ValueError("QCQ digest differs from export metadata")
        n0, m = int(prob.n), int(prob.m)
        if m <= 0 or metadata["inequalities"] or not np.all(np.asarray(prob.is_eq_cons, dtype=bool)):
            report["reason"] = "requires_original_equalities_only"
            return None, report
        if not _free_bounds(prob.bl, prob.bu):
            report["reason"] = "requires_original_free_variables"
            return None, report
        qcq = _QCQReader(qcq_path, deadline).read()
        if qcq["n"] != metadata["n_exported"] or len(qcq["equalities"]) != metadata["equalities_exported"] or len(qcq["inequalities"]) != metadata["inequalities_exported"]:
            raise ValueError("QCQ dimensions differ from export metadata")
        if qcq["inequalities"] or not _qcq_free_bounds(qcq["lb"], qcq["ub"]):
            report["reason"] = "requires_lifted_free_equalities_only"
            return None, report
        if qcq["name"].strip().upper() != metadata["problem"] or qcq["x0"][:n0].tobytes() != np.asarray(prob.x0,dtype=float).tobytes():
            raise ValueError("QCQ identity/original initial point differs from native oracle")
        report["original_before"] = audit_original(prob, endpoint, metadata, TOL)
        _check(deadline)
        report["qcqp_before"] = _audit_qcq(qcq, endpoint, deadline)
        if not report["qcqp_before"]["strict_pass"]:
            report["reason"] = "raw_QCQP_certificate_not_strict"
            return None, report
        before = report["original_before"]
        if before["strict_pass"]:
            report["reason"] = "already_original_strict"
            return None, report
        if before["feasibility"] > TOL or before["complementarity_inf"] > TOL or before["dual_sign_violation"] != 0.0 or before["objective_relative_gap"] > TOL:
            report["reason"] = "original_failure_is_not_dual_only"
            return None, report
        graph = _graph(qcq, metadata, n0)
        fields = _fields(endpoint)
        xfull = _finite_vector(endpoint["x"], qcq["n"], "primal endpoint")
        x = xfull[:n0]
        old = _finite_vector(endpoint[fields[0]], len(qcq["equalities"]), "multipliers")
        lam = np.zeros(m)
        for row in metadata["equalities"]:
            if row["side"] != "eq":
                raise ValueError("non-equality original multiplier mapping")
            lam[row["original_index"]] = old[row["index"]]
        _, g = prob.obj(x, gradient=True)
        g = _finite_vector(g, n0, "original gradient")
        _, J = prob.scons(x, gradient=True) if hasattr(prob, "scons") else prob.cons(x, gradient=True)
        J = sp.csr_matrix(J)
        if J.shape != (m, n0) or not np.all(np.isfinite(J.data)):
            raise ValueError("invalid original Jacobian")
        residual = g + J.T @ lam
        if not np.all(np.isfinite(residual)):
            raise ValueError("nonfinite original Lagrangian gradient")
        _check(deadline)
        def matvec(v):
            _check(deadline)
            return J.T @ v
        def rmatvec(v):
            _check(deadline)
            return J @ v
        operator = LinearOperator((n0, m), matvec=matvec, rmatvec=rmatvec, dtype=float)
        fit_start = time.perf_counter()
        fit = lsmr(operator, -residual, atol=1e-14, btol=1e-14, conlim=1e12, maxiter=min(2000, 2*m+50))
        if not np.all(np.isfinite(fit[0])):
            raise ValueError("nonfinite LSMR correction")
        report.update(fit_seconds=time.perf_counter()-fit_start, lsmr_iterations=int(fit[2]), lsmr_status=int(fit[1]),
                      lsmr_condition_estimate=float(fit[6]) if math.isfinite(float(fit[6])) else None,
                      multiplier_delta_inf=float(np.max(np.abs(fit[0]), initial=0)), jacobian_nnz=int(J.nnz))
        newlam = lam + fit[0]
        trial = copy.deepcopy(endpoint)
        multipliers = np.zeros(len(qcq["equalities"]))
        aux_lagrangian_gradient = np.zeros(qcq["n"])
        qcq["objective"].add_gradient(xfull, aux_lagrangian_gradient)
        for row in metadata["equalities"]:
            value = newlam[row["original_index"]]
            multipliers[row["index"]] = value
            qcq["equalities"][row["index"]].add_gradient(xfull, aux_lagrangian_gradient, value)
        for i, j, a in graph:
            multipliers[i] = -aux_lagrangian_gradient[j] / a
        if not np.all(np.isfinite(multipliers)):
            raise ValueError("nonfinite recovered multipliers")
        trial[fields[0]] = multipliers.tolist()
        report["original_after"] = audit_original(prob, trial, metadata, TOL)
        _check(deadline)
        report["qcqp_after"] = _audit_qcq(qcq, trial, deadline)
        _check(deadline)
        report["point_unchanged"] = np.asarray(trial["x"], dtype=float).tobytes() == xfull.tobytes()
        report["objective_unchanged"] = struct.pack("d",float(trial["objective"])) == struct.pack("d",float(endpoint["objective"]))
        if not report["point_unchanged"] or not report["objective_unchanged"]:
            raise ValueError("primal point or objective changed during dual recovery")
        if not (report["original_after"]["strict_pass"] and report["qcqp_after"]["strict_pass"]):
            report["reason"] = "double_certificate_not_strict"
            return None, report
        if digest(qcq_path) != report["qcq_sha256"]:
            raise ValueError("QCQ changed during recovery")
        _check(deadline)
        report["accepted"] = True
        report["reason"] = "strict_original_and_QCQP_certificates"
        # Update only final certificate summaries. The internal solver trace
        # and all raw termination statistics remain available unchanged.
        trial["dual_infeasibility_inf"] = report["qcqp_after"]["dual_infeasibility_inf"]
        trial["feasibility"] = report["qcqp_after"]["feasibility"]
        trial["complementarity_inf"] = 0.0
        trial["raw_kkt_error"] = max(trial["dual_infeasibility_inf"],trial["feasibility"])
        recomputed = ["dual_infeasibility_inf", "feasibility", "complementarity_inf", "raw_kkt_error"]
        for key, value in {"model_stationarity_inf":trial["dual_infeasibility_inf"],
                           "termination_dual_inf":trial["dual_infeasibility_inf"],
                           "termination_compl_inf":0.0, "equality_violation":trial["feasibility"],
                           "inequality_violation":0.0, "bound_violation":0.0,
                           "termination_audit_valid":True}.items():
            if key in trial:
                trial[key] = value
                recomputed.append(key)
        trial["certificate_source"] = "exact-lift-dual-recovery:independent-native-original-and-QCQ-audits"
        trial["lift_dual_recovery"] = {"schema": SCHEMA, "accepted": True,
            "point_unchanged": True, "objective_unchanged": True,
            "recomputed_certificate_fields":recomputed,
            "other_termination_statistics":"historical-from-immutable-raw-endpoint",
            "original_dual_before": before["dual_infeasibility_inf"],
            "original_dual_after": report["original_after"]["dual_infeasibility_inf"]}
        corrected = trial
    except Exception as exc:
        report["reason"] = "deadline_exhausted" if isinstance(exc, RecoveryDeadline) else "recovery_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        corrected = None
        report["accepted"] = False
    finally:
        report["total_seconds"] = time.perf_counter() - started
    return corrected, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("problem", "endpoint", "metadata", "qcq", "output", "report"):
        parser.add_argument("--"+key, required=True, type=str if key=="problem" else Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args()
    started = _PROCESS_STARTED
    report = {"schema": SCHEMA, "accepted": False, "reason": "recovery_error"}
    file_created = False
    try:
        sources = {args.endpoint.resolve(),args.metadata.resolve(),args.qcq.resolve()}
        if args.output.resolve() in sources or args.report.resolve() in sources or args.output.resolve()==args.report.resolve():
            raise ValueError("recovery output/report must be distinct from every input")
        if args.output.exists():
            raise FileExistsError("corrected endpoint already exists")
        report["raw_endpoint_sha256"] = digest(args.endpoint)
        report["metadata_sha256"] = digest(args.metadata)
        hashes = dict(report)
        meta = json.loads(args.metadata.read_text())
        endpoint = json.loads(args.endpoint.read_text())
        if str(args.problem).strip().upper() != meta["original_audit"]["problem"]:
            raise ValueError("requested problem differs from export metadata")
        if args.cache:
            os.environ["PYCUTEST_CACHE"] = str(args.cache.resolve())
        if not math.isfinite(args.time_limit) or args.time_limit <= 0:
            raise RecoveryDeadline("invalid or exhausted recovery budget")
        import pycutest
        prob = pycutest.import_problem(args.problem,sifParams=meta["original_audit"]["sif_params"] or None,drop_fixed_variables=False)
        corrected, report = recover_certificate(prob,endpoint,meta,args.qcq,args.time_limit-(time.perf_counter()-started))
        report.update({k:v for k,v in hashes.items() if k.endswith("sha256")})
        if corrected is not None:
            _check(started+args.time_limit)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            report["total_seconds"] = time.perf_counter()-started
            corrected["lift_dual_recovery"]["seconds"] = report["total_seconds"]
            corrected["lift_dual_recovery"]["raw_endpoint_sha256"] = report["raw_endpoint_sha256"]
            corrected["lift_dual_recovery"]["qcq_sha256"] = report["qcq_sha256"]
            raw = json.dumps(corrected,indent=2,allow_nan=False)+"\n"
            _check(started+args.time_limit)
            with args.output.open("x",encoding="utf8") as stream:
                file_created = True
                stream.write(raw)
            report["corrected_endpoint_sha256"] = digest(args.output)
            _check(started+args.time_limit)
        rc = 0 if report["accepted"] or report["reason"]=="already_original_strict" else (2 if "error" in report else 1)
    except Exception as exc:
        if file_created:
            args.output.unlink(missing_ok=True)
            report.pop("corrected_endpoint_sha256",None)
        report.update(accepted=False,reason="deadline_exhausted" if isinstance(exc,RecoveryDeadline) else "recovery_error",
                      error=f"{type(exc).__name__}: {exc}")
        rc = 2
    report["total_seconds"] = time.perf_counter()-started
    # Never overwrite an input even when argument validation itself failed.
    if args.report.resolve() not in {args.endpoint.resolve(),args.metadata.resolve(),args.qcq.resolve(),args.output.resolve()}:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,allow_nan=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
