#!/usr/bin/env python3
"""Export a MATPOWER / PGLib-OPF case as an exact QCQP in LAZYIPDCA_QCQP_V1.

The AC optimal power flow written in rectangular voltage coordinates is a
QCQP with no approximation whatsoever: every expression below is at most
quadratic in the decision vector.

    variables   e_i, f_i        (V_i = e_i + j f_i)          2 per bus
                Pg_g, Qg_g      (per-unit generator output)  2 per generator
                Pf,Qf,Pt,Qt     (rate-limited branch flows,
                                  --flow-limit apparent only)

    objective   sum_g  c2 (S Pg)^2 + c1 (S Pg) + c0          convex, diagonal
    equalities  P_i(e,f) - sum_{g in i} Pg + Pd_i = 0        quadratic
                Q_i(e,f) - sum_{g in i} Qg + Qd_i = 0        quadratic
    inequality  e_i^2 + f_i^2 - Vmax_i^2 <= 0                CONVEX
                Vmin_i^2 - e_i^2 - f_i^2 <= 0                CONCAVE  <- the DC part
                branch flow limit                            see --flow-limit
                angle difference limits                      indefinite

The lower-voltage inequalities are concave and the angle-difference
inequalities are generally indefinite.  Their sparse, rank-two Hessian blocks
are the nonconvex inequality structure handled by the lazy DC majorants.

Reference-bus rotation
----------------------
Rotating every voltage by a common phase leaves P, Q, |V| and |I| unchanged,
so the model has an exact one-dimensional invariance.  Left in place it puts a
direction in the null space of the constraint Jacobian that no multiplier can
ever cancel, and the dual infeasibility never converges.  The exporter fixes
f = 0 at the reference bus (lb = ub = 0) and keeps e >= Vmin there, which
removes both the rotation and the V -> -V reflection.

Usage
-----
    python export_pglib_qcqp_r3p20.py CASE.m --output CASE.qcq
    python export_pglib_qcqp_r3p20.py CASE.m --output CASE.qcq \
        --flow-limit apparent --power-flow-start \
        --solve-with build/lazyipdca_qcqp_cli --linear-backend ldl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

BOUND_INFINITY = 1.0e19
DEG = math.pi / 180.0
R3P20_VERSION = "0.4.28r3p20-certified-block-single-normal"
R3P20_API = 4303
R3P20_BUILD_ID = "v0.4.28r3p20-certified-block-single-normal-api-4303"
R3P20_BANNER_RE = re.compile(
    rf"^LazyIPDCA-QCQP {re.escape(R3P20_VERSION)} "
    rf"\(API {R3P20_API}, {re.escape(R3P20_BUILD_ID)}\)"
    rf"(?: \[[^\]\r\n]+\])?$")


# --------------------------------------------------------------------------
# MATPOWER .m parsing (self-contained: no matpowercaseframes dependency)
# --------------------------------------------------------------------------

def parse_matpower(path: Path) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"%.*", "", text)          # strip comments

    out: Dict[str, object] = {}
    m = re.search(r"mpc\.baseMVA\s*=\s*([0-9.eE+-]+)\s*;", text)
    if not m:
        raise ValueError("mpc.baseMVA not found")
    out["baseMVA"] = float(m.group(1))

    for field in ("bus", "gen", "branch", "gencost"):
        m = re.search(r"mpc\." + field + r"\s*=\s*\[(.*?)\]\s*;", text, re.S)
        if not m:
            out[field] = np.zeros((0, 0))
            continue
        rows = []
        for line in m.group(1).split(";"):
            toks = line.split()
            if not toks:
                continue
            rows.append([float(t) for t in toks])
        width = max(len(r) for r in rows)
        table = np.zeros((len(rows), width))
        for k, r in enumerate(rows):
            table[k, :len(r)] = r
        out[field] = table
    return out


# --------------------------------------------------------------------------
# Admittance matrices (MATPOWER makeYbus, branch-wise)
# --------------------------------------------------------------------------

def branch_admittances(branch: np.ndarray, nb: int, bus_index: Dict[int, int]):
    """Per-branch (Yff, Yft, Ytf, Ytt, f, t) for in-service branches."""
    out = []
    for row in branch:
        if len(row) > 10 and row[10] == 0:      # BR_STATUS
            continue
        f = bus_index[int(row[0])]
        t = bus_index[int(row[1])]
        r, x, b = row[2], row[3], row[4]
        ys = 1.0 / complex(r, x)
        tap_mag = row[8] if len(row) > 8 and row[8] != 0.0 else 1.0
        shift = row[9] * DEG if len(row) > 9 else 0.0
        tap = tap_mag * complex(math.cos(shift), math.sin(shift))
        ytt = ys + 1j * b / 2.0
        yff = ytt / (tap * np.conj(tap))
        yft = -ys / np.conj(tap)
        ytf = -ys / tap
        out.append((yff, yft, ytf, ytt, f, t, row))
    return out


def build_ybus(bus: np.ndarray, branch: np.ndarray, baseMVA: float,
               bus_index: Dict[int, int]):
    nb = bus.shape[0]
    rows, cols, vals = [], [], []
    for yff, yft, ytf, ytt, f, t, _row in branch_admittances(branch, nb,
                                                             bus_index):
        rows += [f, f, t, t]
        cols += [f, t, f, t]
        vals += [yff, yft, ytf, ytt]
    for i in range(nb):
        rows.append(i)
        cols.append(i)
        vals.append(complex(bus[i, 4], bus[i, 5]) / baseMVA)   # Gs, Bs
    return sp.coo_matrix((vals, (rows, cols)), shape=(nb, nb),
                         dtype=complex).tocsr()


# --------------------------------------------------------------------------
# Newton-Raphson AC power flow, used to produce a feasible starting point
# --------------------------------------------------------------------------

def newton_power_flow(Y, bus, gen, online, bus_index, baseMVA, ref,
                      max_iter: int = 40, tol: float = 1e-12,
                      q_limit_passes: int = 8):
    """Slack / PV / PQ load flow in polar coordinates, with Q-limit switching.

    Pg is held at the generator table values except at the slack bus, and |V|
    at the generator setpoints on PV buses.  After each solve, any generator
    bus whose reactive output leaves [Qmin, Qmax] is converted to PQ with Qg
    pinned at the violated limit and |V| freed -- the standard MATPOWER
    enforcement, without which the load-flow point is bound-infeasible and the
    exported x0 still carries an equality residual.

    Raises RuntimeError if Newton does not converge; the caller falls back to
    the table starting point rather than shipping a silently bad x0.
    """
    nb = bus.shape[0]
    vm = np.where(bus[:, 7] > 0.0, bus[:, 7], 1.0).astype(float)
    va = bus[:, 8] * DEG

    pg = np.zeros(nb)
    qg = np.zeros(nb)
    qmin = np.zeros(nb)
    qmax = np.zeros(nb)
    is_gen = np.zeros(nb, dtype=bool)
    for g in online:
        i = bus_index[int(gen[g, 0])]
        pg[i] += gen[g, 1] / baseMVA
        qg[i] += gen[g, 2] / baseMVA
        qmin[i] += gen[g, 4] / baseMVA
        qmax[i] += gen[g, 3] / baseMVA
        is_gen[i] = True
        if bus[i, 1] != 3 and gen[g, 5] > 0.0:
            vm[i] = gen[g, 5]

    # The PGLib generator table is not a balanced dispatch: on case118 the
    # tabulated output falls 591 MW short of the load, and a load flow would
    # dump the whole shortfall on the slack bus, past its Pmax.  Re-dispatch
    # every unit on a common fraction of its range to cover load plus a 3 %
    # loss allowance, so the slack only picks up the residual losses.
    pmin_bus = np.zeros(nb)
    pmax_bus = np.zeros(nb)
    for g in online:
        i = bus_index[int(gen[g, 0])]
        pmin_bus[i] += gen[g, 9] / baseMVA
        pmax_bus[i] += gen[g, 8] / baseMVA
    target = bus[:, 2].sum() / baseMVA * 1.03
    span = pmax_bus.sum() - pmin_bus.sum()
    if span > 1e-12:
        frac = (target - pmin_bus.sum()) / span
        frac = min(max(frac, 0.0), 1.0)
        pg = pmin_bus + frac * (pmax_bus - pmin_bus)

    psp = pg - bus[:, 2] / baseMVA
    qsp = qg - bus[:, 3] / baseMVA
    Yd = np.asarray(Y.todense()) if sp.issparse(Y) else np.asarray(Y)

    voltage_controlled = np.array(
        [is_gen[i] and i != ref and bus[i, 1] == 2 for i in range(nb)])

    def solve_once(pv, pq):
        nonlocal va, vm
        idx_p = np.array(pv + pq, dtype=int)
        idx_q = np.array(pq, dtype=int)
        for _ in range(max_iter):
            V = vm * np.exp(1j * va)
            S = V * np.conj(Yd @ V)
            dp = psp[idx_p] - S.real[idx_p]
            dq = qsp[idx_q] - S.imag[idx_q] if idx_q.size else np.zeros(0)
            mismatch = np.concatenate([dp, dq])
            if mismatch.size == 0 or np.abs(mismatch).max() <= tol:
                return
            nvar = idx_p.size + idx_q.size
            J = np.zeros((mismatch.size, nvar))
            h = 1e-7
            for col in range(nvar):
                va2, vm2 = va.copy(), vm.copy()
                if col < idx_p.size:
                    va2[idx_p[col]] += h
                else:
                    vm2[idx_q[col - idx_p.size]] += h
                V2 = vm2 * np.exp(1j * va2)
                S2 = V2 * np.conj(Yd @ V2)
                dp2 = psp[idx_p] - S2.real[idx_p]
                dq2 = qsp[idx_q] - S2.imag[idx_q] if idx_q.size else np.zeros(0)
                J[:, col] = (np.concatenate([dp2, dq2]) - mismatch) / h
            try:
                step = np.linalg.solve(J, -mismatch)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(f"power flow Jacobian is singular: {exc}")
            # Damped Newton: a flat start diverges on some cases (case300),
            # and backtracking on the mismatch norm recovers them.
            base = float(np.linalg.norm(mismatch))
            alpha = 1.0
            va_k, vm_k = va.copy(), vm.copy()
            for _ in range(20):
                va[idx_p] = va_k[idx_p] + alpha * step[:idx_p.size]
                if idx_q.size:
                    vm[idx_q] = vm_k[idx_q] + alpha * step[idx_p.size:]
                Vt = vm * np.exp(1j * va)
                St = Vt * np.conj(Yd @ Vt)
                trial = np.concatenate([
                    psp[idx_p] - St.real[idx_p],
                    (qsp[idx_q] - St.imag[idx_q]) if idx_q.size
                    else np.zeros(0)])
                if float(np.linalg.norm(trial)) < base:
                    break
                alpha *= 0.5
            else:
                va, vm = va_k, vm_k
                raise RuntimeError("power flow line search failed")
        raise RuntimeError("power flow did not converge")

    for _ in range(max(1, q_limit_passes)):
        pv = [i for i in range(nb) if voltage_controlled[i]]
        pq = [i for i in range(nb) if i != ref and not voltage_controlled[i]]
        solve_once(pv, pq)
        V = vm * np.exp(1j * va)
        S = V * np.conj(Yd @ V)
        switched = False
        for i in range(nb):
            if not is_gen[i] or not (voltage_controlled[i] or i == ref):
                continue
            q_needed = S.imag[i] + bus[i, 3] / baseMVA
            limit = None
            if q_needed > qmax[i] + 1e-9:
                limit = qmax[i]
            elif q_needed < qmin[i] - 1e-9:
                limit = qmin[i]
            if limit is None or i == ref:
                continue
            qsp[i] = limit - bus[i, 3] / baseMVA
            voltage_controlled[i] = False
            switched = True
        if not switched:
            break

    V = vm * np.exp(1j * va)
    return V


# --------------------------------------------------------------------------
# Quadratic-form assembly
# --------------------------------------------------------------------------

class Quad:
    """constant + linear.x + 0.5 x^T H x, with H symmetric.

    Terms are accumulated as coeff * x_p * x_q into a one-sided matrix M;
    H = M + M^T reproduces  x^T M x = 0.5 x^T H x.
    """

    def __init__(self, n: int, name: str):
        self.n = n
        self.name = name
        self.constant = 0.0
        self.linear = np.zeros(n)
        self._rows: List[int] = []
        self._cols: List[int] = []
        self._vals: List[float] = []

    def add_bilinear(self, p: int, q: int, coeff: float) -> None:
        if coeff == 0.0:
            return
        self._rows.append(p)
        self._cols.append(q)
        self._vals.append(coeff)

    def hessian(self) -> sp.csr_matrix:
        if not self._vals:
            return sp.csr_matrix((self.n, self.n))
        M = sp.coo_matrix((self._vals, (self._rows, self._cols)),
                          shape=(self.n, self.n)).tocsr()
        return (M + M.T).tocsr()

    def scale(self, s: float) -> "Quad":
        self.constant *= s
        self.linear *= s
        self._vals = [s * v for v in self._vals]
        return self

    def value(self, x: np.ndarray) -> float:
        H = self.hessian()
        return float(self.constant + self.linear @ x + 0.5 * x @ (H @ x))


def power_injection_quads(Y: np.ndarray, nb: int, ie, jf, n: int):
    """P_i and Q_i as quadratic forms in (e, f).

    P_i = sum_k  G_ik (e_i e_k + f_i f_k) + B_ik (f_i e_k - e_i f_k)
    Q_i = sum_k  G_ik (f_i e_k - e_i f_k) - B_ik (e_i e_k + f_i f_k)
    """
    P, Q = [], []
    Y = Y.tocsr()
    for i in range(nb):
        qp = Quad(n, f"P{i}")
        qq = Quad(n, f"Q{i}")
        lo, hi = Y.indptr[i], Y.indptr[i + 1]
        for k, yik in zip(Y.indices[lo:hi], Y.data[lo:hi]):
            k = int(k)
            g, b = float(yik.real), float(yik.imag)
            qp.add_bilinear(ie(i), ie(k), g)
            qp.add_bilinear(jf(i), jf(k), g)
            qp.add_bilinear(jf(i), ie(k), b)
            qp.add_bilinear(ie(i), jf(k), -b)
            qq.add_bilinear(jf(i), ie(k), g)
            qq.add_bilinear(ie(i), jf(k), -g)
            qq.add_bilinear(ie(i), ie(k), -b)
            qq.add_bilinear(jf(i), jf(k), -b)
        P.append(qp)
        Q.append(qq)
    return P, Q


# --------------------------------------------------------------------------
# .qcq writer (LAZYIPDCA_QCQP_V1)
# --------------------------------------------------------------------------

def _fmt(v: float) -> str:
    v = float(v)
    if math.isnan(v):
        raise ValueError("NaN is not a representable coefficient")
    if v >= BOUND_INFINITY:
        return "INF"
    if v <= -BOUND_INFINITY:
        return "-INF"
    return f"{v:.17g}"


def _write_quadratic(fh, q: Quad) -> None:
    H = sp.triu(q.hessian().tocoo(), format="coo")
    order = np.lexsort((H.col, H.row))
    trips = [(int(H.row[k]), int(H.col[k]), float(H.data[k]))
             for k in order if H.data[k] != 0.0]
    fh.write("quadratic\n")
    fh.write(f"name {q.name}\n")
    fh.write(f"constant {_fmt(q.constant)}\n")
    fh.write("linear " + " ".join(_fmt(v) for v in q.linear) + "\n")
    fh.write(f"hessian_nnz {len(trips)}\n")
    for i, j, v in trips:
        fh.write(f"{i} {j} {_fmt(v)}\n")
    fh.write("end_quadratic\n")


def write_qcq(path: Path, name: str, n: int, x0, lb, ub,
              objective: Quad, equalities: List[Quad],
              inequalities: List[Quad]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("LAZYIPDCA_QCQP_V1\n")
        fh.write(f"name {name}\n")
        fh.write(f"n {n}\n")
        fh.write("x0 " + " ".join(_fmt(v) for v in x0) + "\n")
        fh.write("lb " + " ".join(_fmt(v) for v in lb) + "\n")
        fh.write("ub " + " ".join(_fmt(v) for v in ub) + "\n")
        fh.write("objective\n")
        _write_quadratic(fh, objective)
        fh.write(f"equalities {len(equalities)}\n")
        for q in equalities:
            _write_quadratic(fh, q)
        fh.write(f"inequalities {len(inequalities)}\n")
        for q in inequalities:
            _write_quadratic(fh, q)
        fh.write("end_problem\n")


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------

def build_model(case: Dict[str, object], flow_limit: str = "current",
                angle_limits: bool = True, power_flow_start: bool = False,
                power_flow_voltage_continuation: bool = False,
                power_flow_start_tolerance: float = 1e-9):
    if flow_limit not in {"current", "apparent", "none"}:
        raise ValueError(
            "flow_limit must be current, apparent, or none; got "
            f"{flow_limit!r}")
    if (not math.isfinite(power_flow_start_tolerance)
            or power_flow_start_tolerance <= 0.0):
        raise ValueError("power_flow_start_tolerance must be positive and finite")
    bus = np.asarray(case["bus"], dtype=float)
    gen = np.asarray(case["gen"], dtype=float)
    branch = np.asarray(case["branch"], dtype=float)
    gencost = np.asarray(case["gencost"], dtype=float)
    baseMVA = float(case["baseMVA"])

    nb = bus.shape[0]
    bus_index = {int(bus[i, 0]): i for i in range(nb)}
    online = [g for g in range(gen.shape[0])
              if gen.shape[1] <= 7 or gen[g, 7] != 0]
    ng = len(online)

    Y = build_ybus(bus, branch, baseMVA, bus_index)
    branches = branch_admittances(branch, nb, bus_index)
    nbr = len(branches)

    lift = (flow_limit == "apparent")
    lifted_branch_index = {
        b: k for k, b in enumerate(
            b for b, item in enumerate(branches)
            if len(item[6]) > 5 and item[6][5] > 0.0)
    } if lift else {}
    n = 2 * nb + 2 * ng + 4 * len(lifted_branch_index)

    def ie(i): return 2 * i
    def jf(i): return 2 * i + 1
    def ipg(g): return 2 * nb + 2 * g
    def iqg(g): return 2 * nb + 2 * g + 1
    def iflow(b, k):
        return 2 * nb + 2 * ng + 4 * lifted_branch_index[b] + k

    # ---- bounds and starting point -------------------------------------
    lb = np.full(n, -BOUND_INFINITY)
    ub = np.full(n, BOUND_INFINITY)
    x0 = np.zeros(n)
    vmax = bus[:, 11]
    vmin = bus[:, 12]
    for i in range(nb):
        lb[ie(i)], ub[ie(i)] = -vmax[i], vmax[i]
        lb[jf(i)], ub[jf(i)] = -vmax[i], vmax[i]
        vm, va = bus[i, 7], bus[i, 8] * DEG
        if vm <= 0.0:
            vm = 1.0
        x0[ie(i)] = vm * math.cos(va)
        x0[jf(i)] = vm * math.sin(va)

    ref = [i for i in range(nb) if bus[i, 1] == 3]
    if not ref:
        ref = [0]
    r = ref[0]
    lb[jf(r)] = ub[jf(r)] = 0.0      # kills the global rotation invariance
    lb[ie(r)] = vmin[r]              # kills the V -> -V reflection
    x0[jf(r)] = 0.0
    x0[ie(r)] = max(x0[ie(r)], vmin[r])

    start_note = "table"
    power_flow_raw_seconds = 0.0
    if power_flow_start:
        # The generator table of a PGLib case is not a balanced dispatch: on
        # case118 the whole 591 MW shortfall lands on the slack bus and the
        # solver's feasibility phase needs ~700 iterations to absorb it.  A
        # load flow removes it before the solver ever sees the problem.
        try:
            raw_clock = time.perf_counter()
            V = newton_power_flow(Y, bus, gen, online, bus_index, baseMVA, r)
            power_flow_raw_seconds = time.perf_counter() - raw_clock
            phase = np.angle(V[r])
            V = V * np.exp(-1j * phase)      # put the reference on the real axis
            for i in range(nb):
                x0[ie(i)] = float(V[i].real)
                x0[jf(i)] = float(V[i].imag)
            x0[jf(r)] = 0.0
            start_note = "power-flow"
        except RuntimeError as exc:
            power_flow_raw_seconds = time.perf_counter() - raw_clock
            start_note = f"table (power flow failed: {exc})"

    gen_at_bus: Dict[int, List[int]] = {i: [] for i in range(nb)}
    for k, g in enumerate(online):
        i = bus_index[int(gen[g, 0])]
        gen_at_bus[i].append(k)
        lb[ipg(k)] = gen[g, 9] / baseMVA        # PMIN
        ub[ipg(k)] = gen[g, 8] / baseMVA        # PMAX
        lb[iqg(k)] = gen[g, 4] / baseMVA        # QMIN
        ub[iqg(k)] = gen[g, 3] / baseMVA        # QMAX
        x0[ipg(k)] = min(max(gen[g, 1] / baseMVA, lb[ipg(k)]), ub[ipg(k)])
        x0[iqg(k)] = min(max(gen[g, 2] / baseMVA, lb[iqg(k)]), ub[iqg(k)])

    # ---- objective ------------------------------------------------------
    obj = Quad(n, "objective")
    for k, g in enumerate(online):
        if gencost.size == 0:
            continue
        row = gencost[g]
        if row[0] != 2:
            raise ValueError("only polynomial cost (model 2) is supported; "
                             "this case uses a piecewise-linear cost")
        ncost = int(row[3])
        coeffs = row[4:4 + ncost]            # highest degree first
        if ncost > 3:
            raise ValueError("cost polynomial of degree > 2 is not a QCQP")
        c2 = coeffs[0] if ncost == 3 else 0.0
        c1 = coeffs[-2] if ncost >= 2 else 0.0
        c0 = coeffs[-1]
        obj.constant += float(c0)
        obj.linear[ipg(k)] += float(c1) * baseMVA
        obj.add_bilinear(ipg(k), ipg(k), float(c2) * baseMVA * baseMVA)

    # ---- power balance equalities --------------------------------------
    Pq, Qq = power_injection_quads(Y, nb, ie, jf, n)
    equalities: List[Quad] = []
    for i in range(nb):
        hp, hq = Pq[i], Qq[i]
        hp.name = f"balance_P_{int(bus[i,0])}"
        hq.name = f"balance_Q_{int(bus[i,0])}"
        hp.constant += bus[i, 2] / baseMVA      # PD
        hq.constant += bus[i, 3] / baseMVA      # QD
        for k in gen_at_bus[i]:
            hp.linear[ipg(k)] -= 1.0
            hq.linear[iqg(k)] -= 1.0
        equalities.append(hp)
        equalities.append(hq)

    # ---- voltage magnitude inequalities --------------------------------
    inequalities: List[Quad] = []
    for i in range(nb):
        hi = Quad(n, f"vmax_{int(bus[i,0])}")
        hi.constant = -float(vmax[i]) ** 2
        hi.add_bilinear(ie(i), ie(i), 1.0)
        hi.add_bilinear(jf(i), jf(i), 1.0)
        inequalities.append(hi)
        if vmin[i] > 0.0 and i != r:
            lo = Quad(n, f"vmin_{int(bus[i,0])}")
            lo.constant = float(vmin[i]) ** 2
            lo.add_bilinear(ie(i), ie(i), -1.0)
            lo.add_bilinear(jf(i), jf(i), -1.0)
            inequalities.append(lo)

    # ---- branch limits --------------------------------------------------
    def current_terms(yff, yft, f, t):
        """Real and imaginary parts of I = yff V_f + yft V_t, as (index, coef)."""
        re = [(ie(f), yff.real), (jf(f), -yff.imag),
              (ie(t), yft.real), (jf(t), -yft.imag)]
        im = [(ie(f), yff.imag), (jf(f), yff.real),
              (ie(t), yft.imag), (jf(t), yft.real)]
        return re, im

    def add_squared_linear(q: Quad, terms, sign: float) -> None:
        # value = 0.5 x^T (M + M^T) x = x^T M x, so the coefficient of
        # x_p x_s in M must be a_p a_s, not half of it.
        for p, a in terms:
            for s, b in terms:
                q.add_bilinear(p, s, sign * a * b)

    for b, (yff, yft, ytf, ytt, f, t, row) in enumerate(branches):
        rate = row[5] if len(row) > 5 else 0.0          # RATE_A, MVA
        if rate > 0.0 and flow_limit != "none":
            rate_pu = rate / baseMVA
            for tag, (ya, yb, u, v) in (("f", (yff, yft, f, t)),
                                        ("t", (ytf, ytt, t, f))):
                # NB: for the "to" end the pair is (Ytf, Ytt) applied to
                # (V_f, V_t); current_terms takes (coef of first, coef of
                # second) so we pass the endpoints in the matching order.
                if tag == "f":
                    re, im = current_terms(yff, yft, f, t)
                else:
                    re, im = current_terms(ytt, ytf, t, f)
                if flow_limit == "current":
                    lim = rate_pu / float(vmax[u])
                    q = Quad(n, f"ilim_{tag}_{b}")
                    q.constant = -lim * lim
                    add_squared_linear(q, re, 1.0)
                    add_squared_linear(q, im, 1.0)
                    inequalities.append(q)
                else:
                    kp = 0 if tag == "f" else 2
                    # P = e_u * Ire + f_u * Iim ,  Q = f_u * Ire - e_u * Iim
                    hp = Quad(n, f"flowP_{tag}_{b}")
                    hq = Quad(n, f"flowQ_{tag}_{b}")
                    for p, a in re:
                        hp.add_bilinear(ie(u), p, -a)
                        hq.add_bilinear(jf(u), p, -a)
                    for p, a in im:
                        hp.add_bilinear(jf(u), p, -a)
                        hq.add_bilinear(ie(u), p, a)
                    hp.linear[iflow(b, kp)] += 1.0
                    hq.linear[iflow(b, kp + 1)] += 1.0
                    equalities.append(hp)
                    equalities.append(hq)
                    lim = Quad(n, f"slim_{tag}_{b}")
                    lim.constant = -rate_pu * rate_pu
                    lim.add_bilinear(iflow(b, kp), iflow(b, kp), 1.0)
                    lim.add_bilinear(iflow(b, kp + 1), iflow(b, kp + 1), 1.0)
                    inequalities.append(lim)

        if angle_limits and len(row) > 12:
            amin, amax = row[11], row[12]
            # u = Re(V_f conj(V_t)) = e_f e_t + f_f f_t
            # w = Im(V_f conj(V_t)) = f_f e_t - e_f f_t
            for bound, sense in ((amax, +1.0), (amin, -1.0)):
                if abs(bound) >= 89.9 or bound == 0.0:
                    continue
                tn = math.tan(bound * DEG)
                q = Quad(n, f"angle_{'max' if sense > 0 else 'min'}_{b}")
                # sense * (w - tan(bound) * u) <= 0
                q.add_bilinear(jf(f), ie(t), 0.5 * sense)
                q.add_bilinear(ie(t), jf(f), 0.5 * sense)
                q.add_bilinear(ie(f), jf(t), -0.5 * sense)
                q.add_bilinear(jf(t), ie(f), -0.5 * sense)
                q.add_bilinear(ie(f), ie(t), -0.5 * sense * tn)
                q.add_bilinear(ie(t), ie(f), -0.5 * sense * tn)
                q.add_bilinear(jf(f), jf(t), -0.5 * sense * tn)
                q.add_bilinear(jf(t), jf(f), -0.5 * sense * tn)
                inequalities.append(q)

    start_residual = None
    start_inequality_violation = None
    start_bound_violation = None
    raw_start_equality_residual = None
    raw_start_inequality_violation = None
    raw_start_bound_violation = None
    power_flow_continuation_trials = 0
    power_flow_continuation_seconds = 0.0
    power_flow_raw_postvalidation_seconds = 0.0
    power_flow_voltage_lift = 0.0
    power_flow_continuation_status = (
        "power-flow-failed" if power_flow_start and start_note != "power-flow"
        else "not-requested")

    def recover_power_flow_variables(candidate) -> float:
        """Complete Pg/Qg and apparent-flow lifts for one voltage vector."""
        Vc = candidate[0:2 * nb:2] + 1j * candidate[1:2 * nb:2]
        S = Vc * np.conj(Y @ Vc)
        clipped = 0.0
        for i, ks in gen_at_bus.items():
            if not ks:
                continue
            for need, index in ((S.real[i] + bus[i, 2] / baseMVA, ipg),
                                (S.imag[i] + bus[i, 3] / baseMVA, iqg)):
                lows = np.array([lb[index(k)] for k in ks])
                highs = np.array([ub[index(k)] for k in ks])
                spans = np.maximum(highs - lows, 0.0)
                bounded_need = min(max(float(need), float(lows.sum())),
                                   float(highs.sum()))
                clipped = max(clipped, abs(float(need) - bounded_need))
                total_span = float(spans.sum())
                if total_span > 0.0:
                    values = lows + ((bounded_need - float(lows.sum()))
                                     / total_span) * spans
                else:
                    values = lows
                for k, value in zip(ks, values):
                    candidate[index(k)] = float(value)
        if lift:
            for b, (yff, yft, ytf, ytt, f, t, _row) in enumerate(branches):
                if b not in lifted_branch_index:
                    continue
                If = yff * Vc[f] + yft * Vc[t]
                It = ytf * Vc[f] + ytt * Vc[t]
                Sf = Vc[f] * np.conj(If)
                St = Vc[t] * np.conj(It)
                candidate[iflow(b, 0)] = float(Sf.real)
                candidate[iflow(b, 1)] = float(Sf.imag)
                candidate[iflow(b, 2)] = float(St.real)
                candidate[iflow(b, 3)] = float(St.imag)
        return clipped

    def start_metrics(candidate):
        eq_inf = max((abs(q.value(candidate)) for q in equalities),
                     default=0.0)
        ineq_inf = max(0.0, max((q.value(candidate) for q in inequalities),
                                default=0.0))
        bound_inf = max(0.0, float(np.max(lb - candidate)),
                        float(np.max(candidate - ub)))
        return float(eq_inf), float(ineq_inf), float(bound_inf)

    def start_is_feasible(metrics) -> bool:
        return max(metrics) <= power_flow_start_tolerance

    if start_note == "power-flow":
        # A conventional load flow certifies the balance equations, not all
        # OPF inequalities.  Complete the lifted QCQP variables, measure the
        # raw point, and only then call it a feasible QCQP start.
        raw_postvalidation_clock = time.perf_counter()
        raw_clipped = recover_power_flow_variables(x0)
        raw_x0 = x0.copy()
        raw_metrics = start_metrics(raw_x0)
        power_flow_raw_postvalidation_seconds = (
            time.perf_counter() - raw_postvalidation_clock)
        (raw_start_equality_residual,
         raw_start_inequality_violation,
         raw_start_bound_violation) = raw_metrics
        accepted_metrics = raw_metrics
        power_flow_continuation_status = "raw-feasible"

        violated = [q.name for q in inequalities
                    if q.value(raw_x0) > power_flow_start_tolerance]
        continuation_eligible = (
            power_flow_voltage_continuation
            and raw_metrics[0] <= power_flow_start_tolerance
            and raw_metrics[2] <= power_flow_start_tolerance
            and bool(violated)
            and all(name.startswith("vmin_") for name in violated))

        if not start_is_feasible(raw_metrics) and continuation_eligible:
            # PGLib tables commonly give every PV bus Vg=1.0.  A scalar,
            # bounded lift of those voltage controls is a cheap continuation
            # parameter.  The name of the case and the required lift never
            # enter this logic: every candidate is accepted solely through
            # exact evaluation of the assembled QCQP.
            control_base: Dict[int, float] = {}
            for g in online:
                i = bus_index[int(gen[g, 0])]
                if i != r and int(bus[i, 1]) == 2 and gen[g, 5] > 0.0:
                    # Match newton_power_flow when several units share a bus:
                    # its existing loop leaves the last positive Vg in vm[i].
                    control_base[i] = float(gen[g, 5])
            max_lift = max((max(0.0, float(vmax[i]) - value)
                            for i, value in control_base.items()),
                           default=0.0)
            continuation_clock = time.perf_counter()
            best = None
            best_metrics = None
            best_clipped = None
            low = 0.0
            high = None

            def voltage_trial(delta: float):
                nonlocal power_flow_continuation_trials
                trial_gen = gen.copy()
                for g in online:
                    i = bus_index[int(gen[g, 0])]
                    if i in control_base:
                        trial_gen[g, 5] = min(
                            float(vmax[i]),
                            max(float(vmin[i]), control_base[i] + delta))
                power_flow_continuation_trials += 1
                try:
                    Vt = newton_power_flow(
                        Y, bus, trial_gen, online, bus_index, baseMVA, r)
                except RuntimeError:
                    return None
                phase = np.angle(Vt[r])
                Vt = Vt * np.exp(-1j * phase)
                candidate = raw_x0.copy()
                for i in range(nb):
                    candidate[ie(i)] = float(Vt[i].real)
                    candidate[jf(i)] = float(Vt[i].imag)
                candidate[jf(r)] = 0.0
                clipped = recover_power_flow_variables(candidate)
                return candidate, start_metrics(candidate), clipped

            if max_lift > 0.0:
                delta = min(max_lift, max(max_lift / 16.0, 1e-4))
                for _ in range(8):
                    trial = voltage_trial(delta)
                    if trial is not None and start_is_feasible(trial[1]):
                        high = delta
                        best, best_metrics, best_clipped = trial
                        break
                    low = delta
                    if delta >= max_lift:
                        break
                    delta = min(max_lift, 2.0 * delta)

            # Retain a feasible upper endpoint while bisecting toward the
            # smallest observed lift.  Failed/nonfeasible midpoints only move
            # the lower endpoint and can never replace a certified candidate.
            if high is not None:
                width_tolerance = max(1e-5, max_lift * 1e-3)
                for _ in range(10):
                    if high - low <= width_tolerance:
                        break
                    mid = 0.5 * (low + high)
                    trial = voltage_trial(mid)
                    if trial is not None and start_is_feasible(trial[1]):
                        high = mid
                        best, best_metrics, best_clipped = trial
                    else:
                        low = mid

            power_flow_continuation_seconds = (
                time.perf_counter() - continuation_clock)
            if best is not None:
                x0 = best
                accepted_metrics = best_metrics
                raw_clipped = float(best_clipped)
                power_flow_voltage_lift = float(high)
                power_flow_continuation_status = "accepted"
                start_note = ("power-flow voltage-continuation "
                              f"(lift={high:.6g})")
            else:
                # Explicitly retain the exactly measured raw PF point.  The
                # solver may run its visible Phase I, and the metadata below
                # makes both that work and this failed preprocessing visible.
                x0 = raw_x0
                accepted_metrics = raw_metrics
                power_flow_continuation_status = "unavailable-raw-retained"
                start_note = ("power-flow (voltage continuation unavailable; "
                              "raw retained)")
        elif not start_is_feasible(raw_metrics):
            power_flow_continuation_status = (
                "disabled-raw-retained" if not power_flow_voltage_continuation
                else "ineligible-raw-retained")
            start_note = (
                "power-flow (voltage continuation disabled; raw retained)"
                if not power_flow_voltage_continuation
                else "power-flow (voltage continuation ineligible; raw retained)")

        (start_residual, start_inequality_violation,
         start_bound_violation) = accepted_metrics
        if raw_clipped > power_flow_start_tolerance:
            start_note += f" (dispatch clipped by {raw_clipped:.3e} p.u.)"

    meta = {
        "start": start_note,
        "start_equality_residual": start_residual,
        "start_inequality_violation": start_inequality_violation,
        "start_bound_violation": start_bound_violation,
        "raw_start_equality_residual": raw_start_equality_residual,
        "raw_start_inequality_violation": raw_start_inequality_violation,
        "raw_start_bound_violation": raw_start_bound_violation,
        "power_flow_voltage_continuation_status":
            power_flow_continuation_status,
        "power_flow_voltage_continuation_trials":
            power_flow_continuation_trials,
        "power_flow_voltage_lift": power_flow_voltage_lift,
        "power_flow_raw_seconds": power_flow_raw_seconds,
        "power_flow_raw_postvalidation_seconds":
            power_flow_raw_postvalidation_seconds,
        "power_flow_voltage_continuation_seconds":
            power_flow_continuation_seconds,
        "power_flow_start_preprocess_seconds":
            (power_flow_raw_seconds + power_flow_raw_postvalidation_seconds
             + power_flow_continuation_seconds),
        "buses": nb, "generators": ng, "branches": nbr,
        "lifted_flow_branches": len(lifted_branch_index),
        "n": n, "equalities": len(equalities),
        "inequalities": len(inequalities),
        "reference_bus": int(bus[r, 0]),
        "flow_limit": flow_limit,
        "baseMVA": baseMVA,
    }
    return (n, x0, lb, ub, obj, equalities, inequalities, Y, bus, gen, branch,
            online, baseMVA, bus_index, meta, gencost, branches, lift)


# --------------------------------------------------------------------------
# Independent validation: complex arithmetic vs the assembled quadratics
# --------------------------------------------------------------------------

def validate(model, seed: int = 0) -> Dict[str, float]:
    """Re-evaluate every assembled quadratic against complex arithmetic.

    Nothing here reuses the assembly code: injections come from
    V .* conj(Y V), branch flows from V_u .* conj(I), the cost from the raw
    gencost polynomial.  A wrong sign or a transposed index shows up here.
    """
    (n, x0, lb, ub, obj, equalities, inequalities, Y, bus, gen, branch,
     online, baseMVA, bus_index, meta, gencost, branches, lift) = model
    nb = bus.shape[0]
    ng = len(online)
    rng = np.random.default_rng(seed)
    x = x0 + rng.normal(0.0, 0.05, n)
    lo = np.where(lb > -BOUND_INFINITY, lb, -1e3)
    hi = np.where(ub < BOUND_INFINITY, ub, 1e3)
    x = np.clip(x, lo, hi)

    def ie(i): return 2 * i
    def jf(i): return 2 * i + 1
    lifted_branch_index = {
        b: k for k, b in enumerate(
            b for b, item in enumerate(branches)
            if len(item[6]) > 5 and item[6][5] > 0.0)
    } if lift else {}

    def iflow(b, k):
        return 2 * nb + 2 * ng + 4 * lifted_branch_index[b] + k

    V = x[0:2 * nb:2] + 1j * x[1:2 * nb:2]
    S = V * np.conj(Y @ V)
    Pg = x[2 * nb:2 * nb + 2 * ng:2]
    Qg = x[2 * nb + 1:2 * nb + 2 * ng:2]
    inj_p = np.zeros(nb)
    inj_q = np.zeros(nb)
    for k, g in enumerate(online):
        i = bus_index[int(gen[g, 0])]
        inj_p[i] += Pg[k]
        inj_q[i] += Qg[k]

    errors = {}
    worst = 0.0
    for i in range(nb):
        worst = max(worst,
                    abs(equalities[2 * i].value(x)
                        - (S.real[i] - inj_p[i] + bus[i, 2] / baseMVA)),
                    abs(equalities[2 * i + 1].value(x)
                        - (S.imag[i] - inj_q[i] + bus[i, 3] / baseMVA)))
    errors["balance_error"] = float(worst)

    cost = 0.0
    for k, g in enumerate(online):
        row = gencost[g]
        ncost = int(row[3])
        c = row[4:4 + ncost]
        mw = Pg[k] * baseMVA
        cost += float(np.polyval(c, mw))
    errors["objective_error"] = float(abs(obj.value(x) - cost))

    by_name = {q.name: q for q in inequalities}
    worst_v = 0.0
    for i in range(nb):
        q = by_name.get(f"vmax_{int(bus[i,0])}")
        if q is not None:
            worst_v = max(worst_v,
                          abs(q.value(x) - (abs(V[i]) ** 2 - bus[i, 11] ** 2)))
        q = by_name.get(f"vmin_{int(bus[i,0])}")
        if q is not None:
            worst_v = max(worst_v,
                          abs(q.value(x) - (bus[i, 12] ** 2 - abs(V[i]) ** 2)))
    errors["voltage_error"] = float(worst_v)

    worst_flow = 0.0
    worst_angle = 0.0
    eq_by_name = {q.name: q for q in equalities}
    for b, (yff, yft, ytf, ytt, f, t, row) in enumerate(branches):
        If = yff * V[f] + yft * V[t]
        It = ytf * V[f] + ytt * V[t]
        Sf = V[f] * np.conj(If)
        St = V[t] * np.conj(It)
        rate = row[5] if len(row) > 5 else 0.0
        if rate > 0.0:
            rate_pu = rate / baseMVA
            for tag, cur, flow, u, kp in (("f", If, Sf, f, 0),
                                          ("t", It, St, t, 2)):
                if meta["flow_limit"] == "current":
                    q = by_name.get(f"ilim_{tag}_{b}")
                    if q is not None:
                        lim = rate_pu / float(bus[u, 11])
                        worst_flow = max(worst_flow, abs(
                            q.value(x) - (abs(cur) ** 2 - lim ** 2)))
                elif lift:
                    hp = eq_by_name.get(f"flowP_{tag}_{b}")
                    hq = eq_by_name.get(f"flowQ_{tag}_{b}")
                    if hp is not None:
                        worst_flow = max(worst_flow, abs(
                            hp.value(x) - (x[iflow(b, kp)] - flow.real)))
                        worst_flow = max(worst_flow, abs(
                            hq.value(x) - (x[iflow(b, kp + 1)] - flow.imag)))
                    q = by_name.get(f"slim_{tag}_{b}")
                    if q is not None:
                        worst_flow = max(worst_flow, abs(
                            q.value(x)
                            - (x[iflow(b, kp)] ** 2 + x[iflow(b, kp + 1)] ** 2
                               - rate_pu ** 2)))
        if len(row) > 12:
            prod = V[f] * np.conj(V[t])
            for bound, sense, tag in ((row[12], +1.0, "max"),
                                      (row[11], -1.0, "min")):
                q = by_name.get(f"angle_{tag}_{b}")
                if q is None:
                    continue
                tn = math.tan(bound * DEG)
                ref = sense * (prod.imag - tn * prod.real)
                worst_angle = max(worst_angle, abs(q.value(x) - ref))
    errors["flow_error"] = float(worst_flow)
    errors["angle_error"] = float(worst_angle)
    errors["probe_norm"] = float(np.linalg.norm(x - x0))
    return errors


def build_r3p20_solver_command(
        binary: Path, problem: Path, result_json: Path,
        linear_backend: str = "ldl",
        equality_model: str = "condensed-retraction",
        phase1_threads: int = 1, max_outer: int = 1000,
        max_inner: int = 1000, time_limit: float = 1800.0,
        live_every: int = 1, live_iterations: bool = True,
        condensed_curvature: bool = True,
        working_set_drop_patience: int = 8,
        lazyqp_kkt_form: str = "reduced",
        activation: float = 5.0e-4,
        adaptive_normal_refinement: bool = True,
        normal_intermediate_berr_tolerance: float = 1.0e-6,
        normal_intermediate_berr_reject: float = 1.0e-4,
        normal_strict_stationarity_factor: float = 1.0e3,
        alpha_predictor: bool = True,
        alpha_growth: float = 2.0,
        alpha_full_step_frequency: int = 8,
        flow_lift_schur: bool = True,
        post_retraction_ws_expansion: bool = True,
        post_retraction_ws_max: int = 8,
        block_exact_curvature: bool = True,
        projected_multi_ritz_inertia: Optional[bool] = None,
        lazyqp_flow_lift_elimination: bool = True) -> List[str]:
    """Build the validated r3p20 Lazy-DCA command for an exported PGLib QCQP."""
    if linear_backend not in {"native", "ldl", "qdldl", "mumps"}:
        raise ValueError(f"unsupported linear backend: {linear_backend}")
    if equality_model not in {
            "linearized", "condensed-retraction", "elastic-split"}:
        raise ValueError(f"unsupported equality model: {equality_model}")
    if lazyqp_kkt_form not in {"auto", "full", "reduced"}:
        raise ValueError(
            f"unsupported LazyQP KKT form: {lazyqp_kkt_form}")
    for label, value in (("phase1_threads", phase1_threads),
                         ("max_outer", max_outer), ("max_inner", max_inner),
                         ("live_every", live_every),
                         ("working_set_drop_patience",
                          working_set_drop_patience)):
        if int(value) <= 0:
            raise ValueError(f"{label} must be positive")
    if time_limit <= 0.0:
        raise ValueError("time_limit must be positive")
    if not math.isfinite(activation) or activation < 0.0:
        raise ValueError("activation must be a finite nonnegative number")
    if (not math.isfinite(normal_intermediate_berr_tolerance) or
            normal_intermediate_berr_tolerance <= 0.0):
        raise ValueError(
            "normal_intermediate_berr_tolerance must be finite and positive")
    if (not math.isfinite(normal_intermediate_berr_reject) or
            normal_intermediate_berr_reject <
            normal_intermediate_berr_tolerance):
        raise ValueError(
            "normal_intermediate_berr_reject must be finite and at least "
            "normal_intermediate_berr_tolerance")
    if (not math.isfinite(normal_strict_stationarity_factor) or
            normal_strict_stationarity_factor <= 0.0):
        raise ValueError(
            "normal_strict_stationarity_factor must be finite and positive")
    if not math.isfinite(alpha_growth) or alpha_growth <= 1.0:
        raise ValueError("alpha_growth must be finite and greater than one")
    if int(alpha_full_step_frequency) < 1:
        raise ValueError("alpha_full_step_frequency must be positive")
    if int(post_retraction_ws_max) < 0:
        raise ValueError("post_retraction_ws_max must be nonnegative")
    if (equality_model == "condensed-retraction" and
            block_exact_curvature and
            projected_multi_ritz_inertia is True):
        raise ValueError(
            "block_exact_curvature and an explicitly enabled "
            "projected_multi_ritz_inertia are mutually exclusive")
    multi_ritz_auto_policy = (
        projected_multi_ritz_inertia
        if projected_multi_ritz_inertia is not None
        else linear_backend == "native")
    # The certified block-exact route computes the curvature correction
    # directly.  Multi-Ritz is an alternative proposal path and must never be
    # enabled in the same command, including under the native-backend policy.
    use_projected_multi_ritz = (
        multi_ritz_auto_policy and not block_exact_curvature)

    command = [
        str(binary), "--problem", str(problem), "--json", str(result_json),
        "--algorithm", "lazy-dca", "--inner-backend", "lazyqp",
        "--linear-backend", linear_backend, "--qp-backend", linear_backend,
        "--linear-metis", "1",
        "--qp-option", "C", "--qp-amrs", "--qp-init", "2",
        "--qp-mehrotra", "--qp-bp-adaptive",
        "--kkt-adaptive-reg", "--kkt-persistent-symbolic-cache",
        "--equality-qp-original-kkt-gmres",
        "--equality-qp-original-kkt-gmres-max-iterations", "128",
        "--lazyqp-kkt-form", lazyqp_kkt_form,
        "--inner-full-dual-warm-start", "--lazyqp-warm-guard", "2",
        "--max-outer", str(max_outer), "--max-inner", str(max_inner),
        "--time-limit", str(float(time_limit)),
        "--dca-affine-box-bounded-lazy",
        "--feas-tol", "1e-6", "--opt-tol", "1e-6",
        "--dual-inf-tol", "1e-6", "--compl-inf-tol", "1e-6",
        # The final/original KKT systems retain the strict r3p17 contract.
        # Only explicitly identified intermediate condensed-normal solves may
        # use the separate relaxed threshold below.
        "--kkt-berr-tol", "1e-9", "--kkt-berr-reject", "1e-7",
        "--kkt-refine", "3",
        "--inner-tol", "1e-8", "--inner-mu-fallback-tol", "1e-7",
        "--restoration-iters", "600", "--restoration-restarts", "16",
        "--restoration-restart-iters", "400", "--dca-inner-retries", "3",
        "--restoration-exact-workspace-reuse",
        "--restoration-exact-sparse-row-dots",
        "--restoration-exact-parallel-min-rows", "128",
        "--dca-inner-retry-prox", "1e-6", "--dca-step-control",
        "--working-set-drop-patience", str(working_set_drop_patience),
        "--activation", str(float(activation)),
        "--dca-equality-funnel-auto",
    ]
    command.append("--restoration-exact-parallel-row-dots"
                   if phase1_threads > 1
                   else "--no-restoration-exact-parallel-row-dots")

    if equality_model == "condensed-retraction":
        command += [
            "--dca-corrector", "--dca-iterated-soc",
            "--dca-equality-retraction",
            "--dca-equality-model", "condensed-retraction",
            ("--dca-condensed-curvature" if condensed_curvature
             else "--no-dca-condensed-curvature"),
            "--lazyqp-low-rank-form", "bordered",
            "--dca-condensed-max-rank", "8",
            "--dca-condensed-scalar-safeguard-updates", "8",
            "--dca-condensed-newton-iters", "12",
            "--dca-condensed-krylov-iters", "32",
            "--dca-condensed-krylov-rtol", "1e-6",
            "--dca-condensed-continuation", "32",
            "--dca-condensed-tol", "1e-8",
            ("--dca-condensed-adaptive-normal-refinement"
             if adaptive_normal_refinement else
             "--no-dca-condensed-adaptive-normal-refinement"),
            "--dca-condensed-normal-intermediate-berr-tol",
            str(float(normal_intermediate_berr_tolerance)),
            "--dca-condensed-normal-intermediate-berr-reject",
            str(float(normal_intermediate_berr_reject)),
            "--dca-condensed-normal-strict-stationarity-factor",
            str(float(normal_strict_stationarity_factor)),
            ("--dca-condensed-alpha-predictor" if alpha_predictor else
             "--no-dca-condensed-alpha-predictor"),
            "--dca-condensed-alpha-growth", str(float(alpha_growth)),
            "--dca-condensed-alpha-full-step-frequency",
            str(int(alpha_full_step_frequency)),
            ("--dca-condensed-flow-lift-schur" if flow_lift_schur else
             "--no-dca-condensed-flow-lift-schur"),
            ("--dca-condensed-post-retraction-ws-expansion"
             if post_retraction_ws_expansion else
             "--no-dca-condensed-post-retraction-ws-expansion"),
            "--dca-condensed-post-retraction-ws-max",
            str(int(post_retraction_ws_max)),
            ("--dca-condensed-block-exact-curvature"
             if block_exact_curvature else
             "--no-dca-condensed-block-exact-curvature"),
            ("--dca-condensed-projected-multi-ritz-inertia"
             if use_projected_multi_ritz else
             "--no-dca-condensed-projected-multi-ritz-inertia"),
            ("--lazyqp-flow-lift-elimination"
             if lazyqp_flow_lift_elimination else
             "--no-lazyqp-flow-lift-elimination"),
        ]
    elif equality_model == "linearized":
        command += [
            "--dca-corrector", "--dca-iterated-soc",
            "--dca-equality-retraction",
            "--dca-equality-model", "linearized",
            "--lazyqp-low-rank-form", "auto",
        ]
    else:
        command += [
            "--no-dca-corrector", "--no-dca-iterated-soc",
            "--no-dca-equality-retraction", "--no-soc",
            "--dca-equality-model", "elastic-split",
            "--no-dca-equality-elastic-post-retraction",
            "--lazyqp-low-rank-form", "auto",
        ]

    if live_iterations:
        command += ["--live-iterations", "--live-iterations-every",
                    str(live_every), "--live-header-every", "20"]
    else:
        command.append("--no-live-iterations")
    command.append("--quiet")
    return command


def solve_with_r3p20(
        binary: Path, problem: Path, result_json: Path,
        solver_log: Optional[Path] = None, linear_backend: str = "ldl",
        equality_model: str = "condensed-retraction",
        phase1_threads: int = 1, max_outer: int = 1000,
        max_inner: int = 1000, time_limit: float = 1800.0,
        live_every: int = 1, live_output: bool = True,
        condensed_curvature: str = "auto",
        working_set_drop_patience: int = 8,
        lazyqp_kkt_form: str = "reduced",
        activation: float = 5.0e-4,
        adaptive_normal_refinement: bool = True,
        normal_intermediate_berr_tolerance: float = 1.0e-6,
        normal_intermediate_berr_reject: float = 1.0e-4,
        normal_strict_stationarity_factor: float = 1.0e3,
        alpha_predictor: bool = True,
        alpha_growth: float = 2.0,
        alpha_full_step_frequency: int = 8,
        flow_lift_schur: bool = True,
        post_retraction_ws_expansion: bool = True,
        post_retraction_ws_max: int = 8,
        block_exact_curvature: bool = True,
        projected_multi_ritz_inertia: Optional[bool] = None,
        lazyqp_flow_lift_elimination: bool = True
        ) -> Tuple[int, Dict[str, object], str, Path, str]:
    """Run r3p20, stream its output, and return its machine-readable result."""
    binary = binary.resolve()
    problem = problem.resolve()
    result_json = result_json.resolve()
    solver_log = (solver_log or result_json.with_suffix(".log")).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"solver binary is missing or not executable: {binary}")
    if not problem.is_file():
        raise RuntimeError(f"exported QCQP is missing: {problem}")
    if condensed_curvature not in {"auto", "on", "off"}:
        raise ValueError("condensed_curvature must be auto, on, or off")

    try:
        version = subprocess.check_output(
            [str(binary), "--version"], text=True,
            stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"cannot query solver version: {exc.output}") from exc
    if R3P20_BANNER_RE.fullmatch(version) is None:
        raise RuntimeError(
            f"wrong solver binary: {version!r}; expected {R3P20_VERSION}, "
            f"API {R3P20_API}, build {R3P20_BUILD_ID}")

    available = [item.strip() for item in subprocess.check_output(
        [str(binary), "--linear-backends"], text=True,
        stderr=subprocess.STDOUT).strip().split(",") if item.strip()]
    if linear_backend not in available:
        raise RuntimeError(
            f"backend {linear_backend!r} is not compiled; available: "
            + ",".join(available))

    if result_json == solver_log:
        raise ValueError("solver JSON and solver log must use different paths")
    result_json.parent.mkdir(parents=True, exist_ok=True)
    solver_log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(phase1_threads),
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    })
    def run_once(curvature_enabled: bool):
        result_json.unlink(missing_ok=True)
        command = build_r3p20_solver_command(
            binary, problem, result_json, linear_backend, equality_model,
            phase1_threads, max_outer, max_inner, time_limit, live_every,
            live_iterations=live_output,
            condensed_curvature=curvature_enabled,
            working_set_drop_patience=working_set_drop_patience,
            lazyqp_kkt_form=lazyqp_kkt_form,
            activation=activation,
            adaptive_normal_refinement=adaptive_normal_refinement,
            normal_intermediate_berr_tolerance=
                normal_intermediate_berr_tolerance,
            normal_intermediate_berr_reject=
                normal_intermediate_berr_reject,
            normal_strict_stationarity_factor=
                normal_strict_stationarity_factor,
            alpha_predictor=alpha_predictor,
            alpha_growth=alpha_growth,
            alpha_full_step_frequency=alpha_full_step_frequency,
            flow_lift_schur=flow_lift_schur,
            post_retraction_ws_expansion=post_retraction_ws_expansion,
            post_retraction_ws_max=post_retraction_ws_max,
            block_exact_curvature=block_exact_curvature,
            projected_multi_ritz_inertia=projected_multi_ritz_inertia,
            lazyqp_flow_lift_elimination=lazyqp_flow_lift_elimination)
        with solver_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                if live_output:
                    print(line, end="", flush=True)
            return_code = process.wait()
        if not result_json.is_file():
            raise RuntimeError(
                f"solver exited with rc={return_code} without writing "
                f"{result_json}; see {solver_log}")
        result = json.loads(result_json.read_text(encoding="utf-8"))
        if return_code == 0 and result.get("status") != "solved":
            return_code = 1
        return return_code, result

    curvature_enabled = condensed_curvature != "off"
    return_code, result = run_once(curvature_enabled)
    curvature_used = "on" if curvature_enabled else "off"
    lanczos_failure = (
        equality_model == "condensed-retraction"
        and condensed_curvature == "auto"
        and result.get("status") == "numerical_error"
        and "projected Lanczos failed to return a finite Ritz value"
        in str(result.get("message", "")))
    if lanczos_failure:
        first_json = result_json.with_name(
            result_json.stem + ".curvature-on" + result_json.suffix)
        first_log = solver_log.with_name(
            solver_log.stem + ".curvature-on" + solver_log.suffix)
        shutil.copy2(result_json, first_json)
        shutil.copy2(solver_log, first_log)
        if live_output:
            print("[PGLIB] projected condensed Lanczos unavailable; "
                  "retrying the same exact retraction without its optional "
                  "tangent-curvature correction", flush=True)
        return_code, result = run_once(False)
        curvature_used = "off-fallback"
    return return_code, result, version, solver_log, curvature_used


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=Path, help="MATPOWER / PGLib .m file")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--flow-limit", choices=("current", "apparent", "none"),
                    default="current",
                    help="current: |I|^2 <= (rate/Vmax)^2, conservative and "
                         "quadratic in V.  apparent: exact |S|^2 <= rate^2 "
                         "with four lifted flow variables per branch.  "
                         "none: omit every RATE_A branch-flow limit.")
    ap.add_argument("--no-angle-limits", action="store_true")
    start = ap.add_mutually_exclusive_group()
    start.add_argument("--power-flow-start", dest="power_flow_start",
                       action="store_true",
                       help="use a Newton-Raphson load flow as x0")
    start.add_argument("--no-power-flow-start", dest="power_flow_start",
                       action="store_false",
                       help="keep the MATPOWER table start")
    ap.set_defaults(power_flow_start=None)
    voltage_continuation = ap.add_mutually_exclusive_group()
    voltage_continuation.add_argument(
        "--power-flow-voltage-continuation",
        dest="power_flow_voltage_continuation", action="store_true",
        help="opt in to bounded PV-voltage continuation when the raw power "
             "flow violates only lower-voltage constraints")
    voltage_continuation.add_argument(
        "--no-power-flow-voltage-continuation",
        dest="power_flow_voltage_continuation", action="store_false",
        help="retain the raw power-flow start (default)")
    ap.set_defaults(power_flow_voltage_continuation=False)
    ap.add_argument("--json", type=Path, default=None,
                    help="write export and optional solve metadata")
    ap.add_argument("--solve-with", type=Path, default=None,
                    help="run an already compiled r3p20 lazyipdca_qcqp_cli")
    ap.add_argument("--solver-json", type=Path, default=None,
                    help="solver result JSON (default: OUTPUT.result.json)")
    ap.add_argument("--solver-log", type=Path, default=None,
                    help="complete solver log (default: OUTPUT.solve.log)")
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
                    help="LazyQP KKT route; reduced is the r3p20 PGLib "
                         "default and retains the solver's guarded full "
                         "fallback")
    ap.add_argument("--condensed-curvature", choices=("auto", "on", "off"),
                    default="auto",
                    help="auto retries without the optional tangent-curvature "
                         "correction only after the known projected-Lanczos "
                         "failure")
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
        help="relax only intermediate condensed-normal solves")
    normal_refinement.add_argument(
        "--no-dca-condensed-adaptive-normal-refinement",
        dest="adaptive_normal_refinement", action="store_false",
        help="use the strict KKT backward-error tolerance for every normal solve")
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
        help="restore strict normal solves below this multiple of dual_inf_tol")
    alpha_predictor = ap.add_mutually_exclusive_group()
    alpha_predictor.add_argument(
        "--dca-condensed-alpha-predictor", dest="alpha_predictor",
        action="store_true", help="warm-start the condensed outer step length")
    alpha_predictor.add_argument(
        "--no-dca-condensed-alpha-predictor", dest="alpha_predictor",
        action="store_false", help="restart every condensed line search at one")
    ap.set_defaults(alpha_predictor=True)
    ap.add_argument("--dca-condensed-alpha-growth", type=float, default=2.0)
    ap.add_argument("--dca-condensed-alpha-full-step-frequency", type=int,
                    default=8)
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
    block_exact = ap.add_mutually_exclusive_group()
    block_exact.add_argument(
        "--dca-condensed-block-exact-curvature",
        dest="block_exact_curvature", action="store_true",
        help="use the certified block-exact curvature path (default)")
    block_exact.add_argument(
        "--no-dca-condensed-block-exact-curvature",
        dest="block_exact_curvature", action="store_false",
        help="disable block-exact curvature for a comparison run and restore "
             "the r3p19 multi-Ritz backend policy")
    projected_multi_ritz = ap.add_mutually_exclusive_group()
    projected_multi_ritz.add_argument(
        "--dca-condensed-projected-multi-ritz-inertia",
        dest="projected_multi_ritz_inertia", action="store_true",
        help="force batched projected Ritz proposals with an exact inertia "
             "gate (requires --no-dca-condensed-block-exact-curvature)")
    projected_multi_ritz.add_argument(
        "--no-dca-condensed-projected-multi-ritz-inertia",
        dest="projected_multi_ritz_inertia", action="store_false",
        help="disable projected multi-Ritz (default with block-exact and for "
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
    ap.set_defaults(block_exact_curvature=True,
                    projected_multi_ritz_inertia=None,
                    lazyqp_flow_lift_elimination=True)
    ap.add_argument("--no-live-iterations", action="store_true")
    args = ap.parse_args()
    if (args.dca_equality_model == "condensed-retraction" and
            args.block_exact_curvature and
            args.projected_multi_ritz_inertia is True):
        ap.error("--dca-condensed-projected-multi-ritz-inertia requires "
                 "--no-dca-condensed-block-exact-curvature")

    use_power_flow_start = (args.solve_with is not None
                            if args.power_flow_start is None
                            else args.power_flow_start)
    case = parse_matpower(args.case)
    model = build_model(case, flow_limit=args.flow_limit,
                        angle_limits=not args.no_angle_limits,
                        power_flow_start=use_power_flow_start,
                        power_flow_voltage_continuation=
                            args.power_flow_voltage_continuation)
    (n, x0, lb, ub, obj, eqs, ineqs, Y, bus, gen, branch, online, baseMVA,
     bus_index, meta, gencost, branches, lift) = model

    check = validate(model)
    meta.update(check)

    name = args.case.stem
    write_qcq(args.output, name, n, x0, lb, ub, obj, eqs, ineqs)
    meta["output"] = str(args.output)
    meta["bytes"] = args.output.stat().st_size
    return_code = 0
    if args.solve_with is not None:
        solver_json = args.solver_json or args.output.with_suffix(".result.json")
        solver_log = args.solver_log or args.output.with_suffix(".solve.log")
        projected_multi_ritz_policy = (
            args.projected_multi_ritz_inertia
            if args.projected_multi_ritz_inertia is not None
            else args.linear_backend == "native")
        projected_multi_ritz_effective = (
            projected_multi_ritz_policy and
            not args.block_exact_curvature)
        try:
            return_code, result, version, solver_log, curvature_used = \
                solve_with_r3p20(
                args.solve_with, args.output, solver_json, solver_log,
                args.linear_backend, args.dca_equality_model,
                args.phase1_threads, args.max_outer, args.max_inner,
                args.time_limit, args.live_every,
                live_output=not args.no_live_iterations,
                condensed_curvature=args.condensed_curvature,
                working_set_drop_patience=args.working_set_drop_patience,
                lazyqp_kkt_form=args.lazyqp_kkt_form,
                activation=args.activation,
                adaptive_normal_refinement=
                    args.adaptive_normal_refinement,
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
                block_exact_curvature=args.block_exact_curvature,
                projected_multi_ritz_inertia=
                    args.projected_multi_ritz_inertia,
                lazyqp_flow_lift_elimination=
                    args.lazyqp_flow_lift_elimination)
            condensed_profile = (
                args.dca_equality_model == "condensed-retraction")
            meta["solver"] = {
                "version": version,
                "return_code": return_code,
                "status": result.get("status"),
                "message": result.get("message"),
                "objective": result.get("objective"),
                "equality_violation": result.get("equality_violation"),
                "inequality_violation": result.get("inequality_violation"),
                "dual_infeasibility_inf": result.get("dual_infeasibility_inf"),
                "outer_iterations": result.get("outer_iterations"),
                "inner_iterations": result.get("inner_iterations"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "condensed_curvature": curvature_used,
                "flow_lift_schur_requested":
                    condensed_profile and args.flow_lift_schur,
                "post_retraction_ws_expansion_requested":
                    condensed_profile and args.post_retraction_ws_expansion,
                "post_retraction_ws_max":
                    (args.dca_condensed_post_retraction_ws_max
                     if condensed_profile else 0),
                "block_exact_curvature_requested":
                    condensed_profile and args.block_exact_curvature,
                "projected_multi_ritz_inertia_requested":
                    condensed_profile and projected_multi_ritz_effective,
                "lazyqp_flow_lift_elimination_requested":
                    condensed_profile and args.lazyqp_flow_lift_elimination,
                "flow_lift_schur_detected_variables": result.get(
                    "lazy_dca_condensed_flow_lift_schur_detected_variables"),
                "flow_lift_schur_factor_dimension": result.get(
                    "lazy_dca_condensed_flow_lift_schur_factor_dimension"),
                "flow_lift_schur_factorizations": result.get(
                    "lazy_dca_condensed_flow_lift_schur_factorizations"),
                "flow_lift_schur_fallbacks": result.get(
                    "lazy_dca_condensed_flow_lift_schur_fallbacks"),
                "post_retraction_ws_expansions": result.get(
                    "lazy_dca_condensed_post_retraction_working_set_expansions"),
                "post_retraction_ws_fallbacks": result.get(
                    "lazy_dca_condensed_post_retraction_expansion_fallbacks"),
                "condensed_factorizations": result.get(
                    "lazy_dca_condensed_factorizations"),
                "geometry_normal_solves": result.get(
                    "lazy_dca_condensed_geometry_normal_solves"),
                "block_exact_attempts": result.get(
                    "lazy_dca_condensed_block_exact_attempts"),
                "block_exact_acceptances": result.get(
                    "lazy_dca_condensed_block_exact_acceptances"),
                "block_exact_blocks_max": result.get(
                    "lazy_dca_condensed_block_exact_blocks_max"),
                "block_exact_active_dimension_max": result.get(
                    "lazy_dca_condensed_block_exact_active_dimension_max"),
                "block_exact_max_block_dimension": result.get(
                    "lazy_dca_condensed_block_exact_max_block_dimension"),
                "projected_multi_ritz_batches": result.get(
                    "lazy_dca_condensed_projected_multi_ritz_batches"),
                "projected_multi_ritz_fallbacks": result.get(
                    "lazy_dca_condensed_projected_multi_ritz_fallbacks"),
                "tangent_inertia_attempts": result.get(
                    "lazy_dca_condensed_tangent_inertia_attempts"),
                "tangent_inertia_acceptances": result.get(
                    "lazy_dca_condensed_tangent_inertia_acceptances"),
                "lazyqp_flow_lift_elimination_attempts": result.get(
                    "lazyqp_affine_elimination_attempts"),
                "lazyqp_flow_lift_elimination_acceptances": result.get(
                    "lazyqp_affine_elimination_acceptances"),
                "lazyqp_flow_lift_elimination_fallbacks": result.get(
                    "lazyqp_affine_elimination_fallbacks"),
                "lazyqp_flow_lift_eliminated_variables": result.get(
                    "lazyqp_affine_eliminated_variables"),
                "lazyqp_flow_lift_eliminated_equalities": result.get(
                    "lazyqp_affine_eliminated_equalities"),
                "lazyqp_reduced_variables": result.get(
                    "lazyqp_affine_reduced_variables"),
                "lazyqp_reduced_equalities": result.get(
                    "lazyqp_affine_reduced_equalities"),
                "result_json": str(solver_json),
                "log": str(solver_log),
            }
        except (OSError, RuntimeError, ValueError,
                subprocess.SubprocessError) as exc:
            meta["solver"] = {"status": "runner_failure", "message": str(exc)}
            return_code = 2
            print(f"[FAIL] {exc}", file=sys.stderr)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    sys.exit(main())
