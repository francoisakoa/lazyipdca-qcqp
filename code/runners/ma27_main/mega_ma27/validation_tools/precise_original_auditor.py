#!/usr/bin/env python3
"""Independent full-original QCQP audit with outward Decimal150 intervals.

Every coefficient, endpoint coordinate and multiplier is interpreted as its
binary64 value using Decimal.from_float. Directed rounding encloses every
product and accumulation. This is a second audit, never a replacement selected
because its result is more favourable than the historical scalar audit.
Neither audit establishes task budgets, provenance or a solver termination.
"""
from __future__ import annotations

import argparse
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

PRECISION = 150
TOLERANCE = Decimal('0.000001')
ZERO = Decimal(0)
ONE = Decimal(1)
HALF = Decimal('0.5')


def binary64_decimal(value, label='value'):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(label + ': a finite binary64 number is required')
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise ValueError(label + ': not representable as finite binary64') from None
    if not math.isfinite(number):
        raise ValueError(label + ': nonfinite binary64 value')
    if isinstance(value, int) and int(number) != value:
        raise ValueError(label + ': integer does not represent a binary64 value exactly')
    return Decimal.from_float(number)


class Arithmetic:
    def __init__(self):
        self.lower = Context(prec=PRECISION, rounding=ROUND_FLOOR,
                             Emin=-999999, Emax=999999)
        self.upper = Context(prec=PRECISION, rounding=ROUND_CEILING,
                             Emin=-999999, Emax=999999)
        self.nearest = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN,
                               Emin=-999999, Emax=999999)

    @staticmethod
    def point(value):
        return value, value

    def add(self, left, right):
        return (self.lower.add(left[0], right[0]),
                self.upper.add(left[1], right[1]))

    def subtract_point(self, interval, value):
        return (self.lower.subtract(interval[0], value),
                self.upper.subtract(interval[1], value))

    def multiply_point(self, interval, value):
        if value >= ZERO:
            return (self.lower.multiply(interval[0], value),
                    self.upper.multiply(interval[1], value))
        return (self.lower.multiply(interval[1], value),
                self.upper.multiply(interval[0], value))

    def product(self, *values):
        result = self.point(ONE)
        for value in values:
            result = self.multiply_point(result, value)
        return result

    @staticmethod
    def absolute(interval):
        lo, hi = interval
        if lo >= ZERO:
            return interval
        if hi <= ZERO:
            return hi.copy_negate(), lo.copy_negate()
        return ZERO, max(lo.copy_abs(), hi.copy_abs())

    @staticmethod
    def maximum(left, right):
        return max(left[0], right[0]), max(left[1], right[1])

    def report(self, interval):
        lo, hi = interval
        mid = self.nearest.divide(self.nearest.add(lo, hi), Decimal(2))
        try:
            approximate = float(mid)
        except (ValueError, OverflowError):
            approximate = None
        if approximate is not None and not math.isfinite(approximate):
            approximate = None
        return {'lower': str(lo), 'upper': str(hi),
                'approximate_binary64': approximate}


def primary_auditor_path():
    # Both independent audits are shipped together; no prior workspace needed.
    path = Path(__file__).resolve().with_name('original_auditor.py')
    if not path.is_file():
        raise FileNotFoundError('Bundled scalar original_auditor.py is required')
    return path


def load_primary():
    path = primary_auditor_path()
    spec = importlib.util.spec_from_file_location('frozen_original_scalar_auditor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def _certificate(model, endpoint):
    if not isinstance(endpoint, dict):
        raise ValueError('Endpoint absent')
    data = dict(endpoint)
    if data.get('solver') == 'Ipopt':
        for source, target in (
            ('equality_multipliers', 'certificate_equality_multipliers'),
            ('inequality_multipliers', 'certificate_inequality_multipliers'),
            ('lower_bound_multipliers', 'certificate_lower_multipliers'),
            ('upper_bound_multipliers', 'certificate_upper_multipliers')):
            data[target] = data[source]
    n = model.get('n')
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError('Invalid model dimension')
    arrays = [data[k] for k in ('x', 'certificate_equality_multipliers',
              'certificate_inequality_multipliers', 'certificate_lower_multipliers',
              'certificate_upper_multipliers')]
    expected = [n, len(model['eq']), len(model['ineq']), n, n]
    if any(not isinstance(a, (tuple, list)) for a in arrays) or [len(a) for a in arrays] != expected:
        raise ValueError('Certificate dimensions differ')
    converted = [[binary64_decimal(v, 'certificate') for v in a] for a in arrays]
    if len(model['lb']) != n or len(model['ub']) != n:
        raise ValueError('Bound dimensions differ')
    bounds = []
    for lo, hi in zip(model['lb'], model['ub']):
        if isinstance(lo, bool) or isinstance(hi, bool):
            raise ValueError('Invalid bound')
        lo, hi = float(lo), float(hi)
        # Frozen V1/V2 uses either infinity sign for an absent bound side.
        # Match the native readers and scalar original audit: finiteness alone
        # determines whether a side exists. Do not change finite coefficients,
        # starts, or serialized input; NaNs and reversed finite pairs remain
        # invalid. Comparing raw infinity sentinels would reject valid exports.
        if (math.isnan(lo) or math.isnan(hi)
                or (math.isfinite(lo) and math.isfinite(hi) and lo > hi)):
            raise ValueError('Invalid bound orientation or ordering')
        bounds.append((Decimal.from_float(lo) if math.isfinite(lo) else None,
                       Decimal.from_float(hi) if math.isfinite(hi) else None))
    return converted, bounds


def audit_decimal(model, endpoint):
    """Enclose all full-original residuals; ignore status/residual metadata."""
    started = time.monotonic()
    (x, y, z, zl, zu), bounds = _certificate(model, endpoint)
    n = model['n']
    arithmetic = Arithmetic()
    zero = arithmetic.point(ZERO)
    gradient = [arithmetic.add(arithmetic.point(l.copy_negate()), arithmetic.point(u))
                for l, u in zip(zl, zu)]

    def validate_quad(quad):
        if not isinstance(quad, (tuple, list)) or len(quad) != 3:
            raise ValueError('Invalid quadratic model')
        constant, linear, hessian = quad
        c = binary64_decimal(constant, 'quadratic constant')
        lin = []
        for j, coefficient in linear:
            if isinstance(j, bool) or not isinstance(j, int) or not 0 <= j < n:
                raise ValueError('Invalid linear index')
            lin.append((j, binary64_decimal(coefficient, 'linear coefficient')))
        hes = []
        for i, j, coefficient in hessian:
            if any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < n for k in (i, j)):
                raise ValueError('Invalid Hessian index')
            hes.append((i, j, binary64_decimal(coefficient, 'Hessian coefficient')))
        return c, lin, hes

    def value(quad):
        c, lin, hes = quad
        total = arithmetic.point(c)
        for j, coefficient in lin:
            total = arithmetic.add(total, arithmetic.product(coefficient, x[j]))
        for i, j, coefficient in hes:
            term = arithmetic.product(coefficient, x[i], x[j])
            if i == j:
                term = arithmetic.multiply_point(term, HALF)
            total = arithmetic.add(total, term)
        return total

    def add_gradient(quad, weight):
        if weight == ZERO:
            return
        _, lin, hes = quad
        for j, coefficient in lin:
            gradient[j] = arithmetic.add(gradient[j], arithmetic.product(weight, coefficient))
        for i, j, coefficient in hes:
            gradient[i] = arithmetic.add(gradient[i], arithmetic.product(weight, coefficient, x[j]))
            if i != j:
                gradient[j] = arithmetic.add(gradient[j], arithmetic.product(weight, coefficient, x[i]))

    objective_quad = validate_quad(model['obj'])
    objective = value(objective_quad)
    add_gradient(objective_quad, ONE)
    primal, dual, complementarity = zero, zero, zero
    for raw, multiplier in zip(model['eq'], y):
        quad = validate_quad(raw)
        primal = arithmetic.maximum(primal, arithmetic.absolute(value(quad)))
        add_gradient(quad, multiplier)
    for raw, multiplier in zip(model['ineq'], z):
        quad = validate_quad(raw)
        g = value(quad)
        primal = arithmetic.maximum(primal, g)
        complementarity = arithmetic.maximum(complementarity,
            arithmetic.absolute(arithmetic.multiply_point(g, multiplier)))
        dual = arithmetic.maximum(dual, arithmetic.point(multiplier.copy_negate()))
        add_gradient(quad, multiplier)
    for j, (lo, hi) in enumerate(bounds):
        dual = arithmetic.maximum(dual, arithmetic.point(zl[j].copy_negate()))
        dual = arithmetic.maximum(dual, arithmetic.point(zu[j].copy_negate()))
        if lo is not None:
            difference = arithmetic.subtract_point(arithmetic.point(x[j]), lo)
            primal = arithmetic.maximum(primal, (difference[1].copy_negate(), difference[0].copy_negate()))
            complementarity = arithmetic.maximum(complementarity,
                arithmetic.absolute(arithmetic.multiply_point(difference, zl[j])))
        else:
            dual = arithmetic.maximum(dual, arithmetic.point(zl[j].copy_abs()))
        if hi is not None:
            difference = arithmetic.subtract_point(arithmetic.point(hi), x[j])
            primal = arithmetic.maximum(primal, (difference[1].copy_negate(), difference[0].copy_negate()))
            complementarity = arithmetic.maximum(complementarity,
                arithmetic.absolute(arithmetic.multiply_point(difference, zu[j])))
        else:
            dual = arithmetic.maximum(dual, arithmetic.point(zu[j].copy_abs()))
        dual = arithmetic.maximum(dual, arithmetic.absolute(gradient[j]))
    residuals = (primal, dual, complementarity)
    if all(r[1] <= TOLERANCE for r in residuals):
        strict, classification = True, 'within_original_tolerance'
    elif any(r[0] > TOLERANCE for r in residuals):
        strict, classification = False, 'outside_original_tolerance'
    else:
        strict, classification = None, 'arithmetic_undetermined'
    return {
        'scope': 'full original frozen V1/V2 binary64 QCQP, original function units',
        'arithmetic': 'Decimal.from_float exact inputs; outward-rounded interval operations',
        'precision_decimal_digits': PRECISION,
        'objective': arithmetic.report(objective), 'feasibility': arithmetic.report(primal),
        'dual': arithmetic.report(dual), 'complementarity': arithmetic.report(complementarity),
        'strict': strict, 'classification': classification, 'tolerance': str(TOLERANCE),
        'endpoint_status_ignored': True, 'qualification_claimed': False,
        'audit_seconds': time.monotonic() - started,
    }


def audit_pair(model, endpoint):
    primary, path = load_primary()
    output = {'schema': 'independent-original-scalar-and-decimal150-audits-v1',
              'policy': 'report both audits; no favourable-audit selection; budgets and provenance remain separate',
              'primary_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'qualification_claimed': False}
    for name, operation in (('primary_scalar', primary.audit), ('high_precision', audit_decimal)):
        try:
            output[name] = {'state': 'computed', **operation(model, endpoint)}
        except (ValueError, KeyError, TypeError, OverflowError, ArithmeticError) as exc:
            output[name] = {'state': 'unavailable', 'strict': None,
                            'error': f'{type(exc).__name__}: {exc}'}
    p, h = output['primary_scalar']['strict'], output['high_precision']['strict']
    if isinstance(p, bool) and isinstance(h, bool):
        output['conservative_mathematical_strict'] = p and h
        output['strict_agreement'] = p == h
        output['comparison'] = 'agree' if p == h else 'disagree_requires_adjudication'
    else:
        output.update(conservative_mathematical_strict=None, strict_agreement=None,
                      comparison='one_or_both_audits_indeterminate')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('endpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    primary, _ = load_primary()
    model = primary.read_model(args.input)
    endpoint = json.loads(args.endpoint.read_text())
    result = audit_pair(model, endpoint)
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['endpoint_sha256'] = hashlib.sha256(args.endpoint.read_bytes()).hexdigest()
    data = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(data)
    print(data, end='')
    return 0 if result['conservative_mathematical_strict'] is True else 1


if __name__ == '__main__':
    raise SystemExit(main())
