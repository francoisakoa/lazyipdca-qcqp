#!/usr/bin/env python3
"""Strict reader and dependency-free evaluator for LAZYIPDCA_QCQP_V1.

The on-disk format stores the upper triangle of every symmetric Hessian and
uses the convention

    q(x) = constant + linear' x + 0.5 x' Hessian x.

Dense linear vectors in the file are compressed while parsing.  This keeps
the in-memory representation, CasADi expression graph, and residual checks
sparse without changing a coefficient of the exported QCQP.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence, TextIO


@dataclass(frozen=True)
class HessianEntry:
    i: int
    j: int
    value: float


@dataclass(frozen=True)
class QuadraticFunction:
    name: str
    constant: float
    linear: tuple[tuple[int, float], ...]
    hessian: tuple[HessianEntry, ...]

    def value(self, x: Sequence[float]) -> float:
        result = self.constant
        for index, coefficient in self.linear:
            result += coefficient * x[index]
        for entry in self.hessian:
            if entry.i == entry.j:
                result += 0.5 * entry.value * x[entry.i] * x[entry.i]
            else:
                result += entry.value * x[entry.i] * x[entry.j]
        return result

    def add_gradient(
        self, x: Sequence[float], destination: list[float], scale: float = 1.0
    ) -> None:
        if scale == 0.0:
            return
        for index, coefficient in self.linear:
            destination[index] += scale * coefficient
        for entry in self.hessian:
            coefficient = scale * entry.value
            if entry.i == entry.j:
                destination[entry.i] += coefficient * x[entry.i]
            else:
                destination[entry.i] += coefficient * x[entry.j]
                destination[entry.j] += coefficient * x[entry.i]

    def gradient(self, x: Sequence[float], dimension: int) -> list[float]:
        result = [0.0] * dimension
        self.add_gradient(x, result)
        return result

    def gradient_structure(self) -> tuple[int, ...]:
        columns = {index for index, _ in self.linear}
        for entry in self.hessian:
            if entry.value != 0.0:
                columns.add(entry.i)
                columns.add(entry.j)
        return tuple(sorted(columns))


@dataclass(frozen=True)
class QCQPModel:
    name: str
    n: int
    x0: tuple[float, ...]
    lb: tuple[float, ...]
    ub: tuple[float, ...]
    objective: QuadraticFunction
    equalities: tuple[QuadraticFunction, ...]
    inequalities: tuple[QuadraticFunction, ...]

    @property
    def constraints(self) -> tuple[QuadraticFunction, ...]:
        return self.equalities + self.inequalities

    def objective_value(self, x: Sequence[float]) -> float:
        self._check_point(x)
        return self.objective.value(x)

    def constraint_values(self, x: Sequence[float]) -> list[float]:
        self._check_point(x)
        return [quadratic.value(x) for quadratic in self.constraints]

    def residuals(self, x: Sequence[float]) -> dict[str, float]:
        self._check_point(x)
        equality_inf = max(
            (abs(quadratic.value(x)) for quadratic in self.equalities),
            default=0.0,
        )
        inequality_inf = max(
            (max(0.0, quadratic.value(x)) for quadratic in self.inequalities),
            default=0.0,
        )
        bound_inf = 0.0
        for index, value in enumerate(x):
            if math.isfinite(self.lb[index]):
                bound_inf = max(bound_inf, self.lb[index] - value)
            if math.isfinite(self.ub[index]):
                bound_inf = max(bound_inf, value - self.ub[index])
        bound_inf = max(0.0, bound_inf)
        return {
            "equality_inf": equality_inf,
            "inequality_inf": inequality_inf,
            "bound_inf": bound_inf,
            "primal_inf": max(equality_inf, inequality_inf, bound_inf),
        }

    def jacobian_structural_nnz(self) -> int:
        return sum(
            len(quadratic.gradient_structure()) for quadratic in self.constraints
        )

    def lagrangian_hessian_structural_nnz(self) -> int:
        structure: set[tuple[int, int]] = set()
        for quadratic in (self.objective,) + self.constraints:
            for entry in quadratic.hessian:
                if entry.value != 0.0:
                    structure.add((entry.i, entry.j))
        return len(structure)

    def kkt_residuals(
        self,
        x: Sequence[float],
        constraint_multipliers: Sequence[float],
        bound_multipliers: Sequence[float],
    ) -> dict[str, float]:
        """Return unscaled KKT diagnostics in CasADi/Ipopt convention.

        ``constraint_multipliers`` uses one free multiplier per equality and
        one nonnegative multiplier per upper-bounded inequality ``g(x)<=0``.
        ``bound_multipliers`` is Ipopt's combined ``lam_x``: negative at a
        lower bound and positive at an upper bound.
        """

        self._check_point(x)
        if len(constraint_multipliers) != len(self.constraints):
            raise ValueError("constraint multiplier dimension mismatch")
        if len(bound_multipliers) != self.n:
            raise ValueError("bound multiplier dimension mismatch")

        stationarity = self.objective.gradient(x, self.n)
        for multiplier, quadratic in zip(
            constraint_multipliers, self.constraints
        ):
            quadratic.add_gradient(x, stationarity, multiplier)
        for index, multiplier in enumerate(bound_multipliers):
            stationarity[index] += multiplier

        complementarity_inf = 0.0
        dual_sign_inf = 0.0
        equality_count = len(self.equalities)
        for offset, quadratic in enumerate(self.inequalities):
            multiplier = constraint_multipliers[equality_count + offset]
            dual_sign_inf = max(dual_sign_inf, max(0.0, -multiplier))
            complementarity_inf = max(
                complementarity_inf,
                abs(max(0.0, multiplier) * quadratic.value(x)),
            )

        for index, multiplier in enumerate(bound_multipliers):
            lower_finite = math.isfinite(self.lb[index])
            upper_finite = math.isfinite(self.ub[index])
            fixed = (
                lower_finite
                and upper_finite
                and self.lb[index] == self.ub[index]
            )
            if fixed:
                # Ipopt removes fixed variables under make_parameter.  The
                # common LazyIPDCA termination metric also excludes them.
                stationarity[index] = 0.0
                continue
            if multiplier < 0.0:
                if lower_finite:
                    complementarity_inf = max(
                        complementarity_inf,
                        abs((-multiplier) * (x[index] - self.lb[index])),
                    )
                else:
                    dual_sign_inf = max(dual_sign_inf, -multiplier)
            elif multiplier > 0.0:
                if upper_finite:
                    complementarity_inf = max(
                        complementarity_inf,
                        abs(multiplier * (self.ub[index] - x[index])),
                    )
                else:
                    dual_sign_inf = max(dual_sign_inf, multiplier)

        return {
            "dual_inf": max((abs(value) for value in stationarity), default=0.0),
            "complementarity_inf": complementarity_inf,
            "dual_sign_inf": dual_sign_inf,
        }

    def _check_point(self, x: Sequence[float]) -> None:
        if len(x) != self.n:
            raise ValueError(f"point dimension {len(x)} != model dimension {self.n}")


def _parse_number(token: str, *, allow_infinity: bool) -> float:
    normalized = token.upper()
    if normalized in {"INF", "+INF"}:
        if allow_infinity:
            return math.inf
        raise ValueError("infinite value outside a bound vector")
    if normalized == "-INF":
        if allow_infinity:
            return -math.inf
        raise ValueError("infinite value outside a bound vector")
    value = float(token)
    if math.isnan(value):
        raise ValueError("NaN model value")
    if not allow_infinity and not math.isfinite(value):
        raise ValueError("non-finite model coefficient")
    return value


class _LineReader:
    def __init__(self, stream: TextIO, source: Path):
        self.stream = stream
        self.source = source
        self.line_number = 0

    def line(self) -> list[str]:
        text = self.stream.readline()
        self.line_number += 1
        if text == "":
            self.error("unexpected end of file")
        fields = text.split()
        if not fields:
            self.error("unexpected blank line")
        return fields

    def expect(self, token: str) -> list[str]:
        fields = self.line()
        if fields[0] != token:
            self.error(f"expected {token!r}, got {fields[0]!r}")
        return fields[1:]

    def error(self, message: str) -> None:
        raise ValueError(f"{self.source}:{self.line_number}: {message}")


def _read_dense_vector(
    reader: _LineReader, key: str, n: int, *, allow_infinity: bool
) -> tuple[float, ...]:
    fields = reader.expect(key)
    if len(fields) != n:
        reader.error(f"{key} has {len(fields)} entries; expected {n}")
    try:
        return tuple(
            _parse_number(token, allow_infinity=allow_infinity) for token in fields
        )
    except ValueError as exc:
        reader.error(f"invalid {key}: {exc}")
    raise AssertionError("unreachable")


def _read_sparse_linear(reader: _LineReader, n: int) -> tuple[tuple[int, float], ...]:
    fields = reader.expect("linear")
    if len(fields) != n:
        reader.error(f"linear has {len(fields)} entries; expected {n}")
    entries = []
    try:
        for index, token in enumerate(fields):
            # Exact lexical zero only. Every other token retains the strict
            # independent Python numeric validation, including tiny values.
            if token == "0":
                continue
            value = _parse_number(token, allow_infinity=False)
            if value != 0.0:
                entries.append((index, value))
    except ValueError as exc:
        reader.error(f"invalid linear: {exc}")
    return tuple(entries)


def _single_field(reader: _LineReader, key: str) -> str:
    fields = reader.expect(key)
    if len(fields) != 1:
        reader.error(f"{key} requires exactly one value")
    return fields[0]


def _read_quadratic(reader: _LineReader, n: int) -> QuadraticFunction:
    if reader.expect("quadratic"):
        reader.error("quadratic marker must not have a value")
    name = _single_field(reader, "name")
    try:
        constant = _parse_number(
            _single_field(reader, "constant"), allow_infinity=False
        )
    except ValueError as exc:
        reader.error(f"invalid constant: {exc}")

    linear = _read_sparse_linear(reader, n)
    try:
        hessian_nnz = int(_single_field(reader, "hessian_nnz"))
    except ValueError:
        reader.error("invalid hessian_nnz")
    if hessian_nnz < 0:
        reader.error("negative hessian_nnz")

    hessian: list[HessianEntry] = []
    for _ in range(hessian_nnz):
        fields = reader.line()
        if len(fields) != 3:
            reader.error("Hessian entry requires i, j, value")
        try:
            i = int(fields[0])
            j = int(fields[1])
            value = _parse_number(fields[2], allow_infinity=False)
        except ValueError as exc:
            reader.error(f"invalid Hessian entry: {exc}")
        if not (0 <= i < n and 0 <= j < n):
            reader.error(f"Hessian index ({i}, {j}) outside [0, {n})")
        if i > j:
            i, j = j, i
        if value != 0.0:
            hessian.append(HessianEntry(i, j, value))

    if reader.expect("end_quadratic"):
        reader.error("end_quadratic marker must not have a value")
    return QuadraticFunction(name, constant, linear, tuple(hessian))


def read_qcq(path: str | Path) -> QCQPModel:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        reader = _LineReader(stream, source)
        if reader.expect("LAZYIPDCA_QCQP_V1"):
            reader.error("format marker must not have a value")
        name = _single_field(reader, "name")
        try:
            n = int(_single_field(reader, "n"))
        except ValueError:
            reader.error("invalid model dimension")
        if n <= 0:
            reader.error("model dimension must be positive")

        x0 = _read_dense_vector(reader, "x0", n, allow_infinity=False)
        lb = _read_dense_vector(reader, "lb", n, allow_infinity=True)
        ub = _read_dense_vector(reader, "ub", n, allow_infinity=True)
        for index, (lower, upper) in enumerate(zip(lb, ub)):
            if lower > upper:
                reader.error(f"inconsistent bounds for variable {index}")

        if reader.expect("objective"):
            reader.error("objective marker must not have a value")
        objective = _read_quadratic(reader, n)

        try:
            equality_count = int(_single_field(reader, "equalities"))
        except ValueError:
            reader.error("invalid equality count")
        if equality_count < 0:
            reader.error("negative equality count")
        equalities = tuple(
            _read_quadratic(reader, n) for _ in range(equality_count)
        )

        try:
            inequality_count = int(_single_field(reader, "inequalities"))
        except ValueError:
            reader.error("invalid inequality count")
        if inequality_count < 0:
            reader.error("negative inequality count")
        inequalities = tuple(
            _read_quadratic(reader, n) for _ in range(inequality_count)
        )

        if reader.expect("end_problem"):
            reader.error("end_problem marker must not have a value")
        trailing = stream.read()
        if trailing.strip():
            reader.error("unexpected content after end_problem")

    return QCQPModel(
        name=name,
        n=n,
        x0=x0,
        lb=lb,
        ub=ub,
        objective=objective,
        equalities=equalities,
        inequalities=inequalities,
    )


def total_hessian_triplets(model: QCQPModel) -> int:
    quadratics: Iterable[QuadraticFunction] = (
        (model.objective,) + model.constraints
    )
    return sum(len(quadratic.hessian) for quadratic in quadratics)
