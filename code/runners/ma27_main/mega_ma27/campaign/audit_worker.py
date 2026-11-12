#!/usr/bin/env python3
"""Audit every frozen original QCQP row/bound, independently of solver status."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
TOLERANCE = 1e-6


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def origin(endpoint, case):
    if case.get('method') in ('ipopt_qcqp','ipopt_mumps') or endpoint.get('solver') == 'Ipopt':
        return 'ipopt'
    route = endpoint.get('native_al_fallback')
    if isinstance(route, dict) and route.get('accepted') is False and route.get('endpoint_origin') == 'lazy':
        return 'lazy'
    return endpoint.get('solution_origin', 'unknown') if endpoint.get('solution_origin') == 'lazy' else 'unknown'


def normalized_ipopt(endpoint):
    result = dict(endpoint)
    if endpoint.get('solver') == 'Ipopt':
        for old, new in [('equality_multipliers', 'certificate_equality_multipliers'),
                         ('inequality_multipliers', 'certificate_inequality_multipliers'),
                         ('lower_bound_multipliers', 'certificate_lower_multipliers'),
                         ('upper_bound_multipliers', 'certificate_upper_multipliers')]:
            result[new] = endpoint[old]
    return result


def timing_and_budget(endpoint, case):
    """Report elapsed and inclusive native work without adding overlapping clocks."""
    ipopt = endpoint.get('solver') == 'Ipopt' or case.get('method') in ('ipopt_qcqp','ipopt_mumps')
    time_keys = ('core_s', 'solve_seconds') if ipopt else ('elapsed_seconds',)
    # The first present documented clock is authoritative. An invalid core_s
    # cannot be hidden by selecting a more favourable secondary clock.
    numerical_field = next((k for k in time_keys if k in endpoint), None)
    numerical_seconds = endpoint.get(numerical_field) if numerical_field else None
    if not finite_number(numerical_seconds): numerical_seconds = None
    budget, outer_limit = case.get('seconds'), case.get('max_outer')
    restoration_limit = case.get('shared_restoration_limit', case.get('restoration_max_iterations'))
    reasons = []
    if not finite_number(budget) or budget != 7200:
        reasons.append('configured_seconds_must_be_7200')
    if type(outer_limit) is not int or outer_limit != 5000:
        reasons.append('configured_outer_limit_must_be_5000')
    if type(restoration_limit) is not int or restoration_limit != 600:
        reasons.append('configured_shared_restoration_limit_must_be_600')
    if numerical_seconds is None:
        reasons.append('raw_numerical_time_missing_or_invalid')
    elif finite_number(budget) and numerical_seconds > budget:
        reasons.append('raw_numerical_time_exceeds_budget')
    process_seconds = case.get('solve_process_seconds')
    if not finite_number(process_seconds):
        reasons.append('solve_process_envelope_missing_or_invalid')
    elif finite_number(budget) and process_seconds > budget:
        reasons.append('solve_process_envelope_exceeds_budget')
    if ipopt:
        if endpoint.get('max_iterations') != outer_limit:
            reasons.append('Ipopt_maximum_iteration_contract_differs')
        if endpoint.get('optimization_wall_limit_seconds') != budget:
            reasons.append('Ipopt_time_limit_contract_differs')
        iteration_field = next((k for k in ('iterations', 'iteration_count', 'iter_count') if k in endpoint), None)
        iterations = endpoint.get(iteration_field) if iteration_field else None
        if iteration_field is None:
            reasons.append('Ipopt_actual_iteration_count_missing')
        elif type(iterations) is not int or not 0 <= iterations <= 5000:
            reasons.append('Ipopt_reported_iteration_budget_invalid')
        restoration = None
    else:
        iteration_field = 'outer_iterations'; iterations = endpoint.get(iteration_field)
        restoration = endpoint.get('restoration_iterations')
        if type(iterations) is not int or not 0 <= iterations <= 5000:
            reasons.append('native_total_outer_budget_invalid_or_exceeded')
        if type(restoration) is not int or not 0 <= restoration <= 600:
            reasons.append('native_shared_restoration_budget_invalid_or_exceeded')
    explicit_failures = []
    # Inclusive route values are checks/measurements, never additive timers.
    for name in ('native_al_fallback', 'adaptive_equality_probe', 'native_barrier_retry'):
        route = endpoint.get(name)
        if isinstance(route, dict) and route.get('within_global_budget') is False:
            explicit_failures.append(name)
        if isinstance(route, dict) and 'total_seconds' in route:
            elapsed = route.get('total_seconds')
            if not finite_number(elapsed) or elapsed > 7200:
                reasons.append('native_inclusive_route_time_invalid_or_exceeded:' + name)
    if explicit_failures:
        reasons.append('explicit_native_global_budget_failure')
    status = str(endpoint.get('status', endpoint.get('ipopt_application_status', '')))
    status_folded = status.casefold()
    exhausted = (numerical_seconds is not None and finite_number(budget) and numerical_seconds > budget)
    exhausted = exhausted or (finite_number(process_seconds) and finite_number(budget) and process_seconds > budget)
    exhausted = exhausted or bool(explicit_failures) or status_folded in (
        'time_limit', 'maximum_cputime_exceeded', 'maximum_walltime_exceeded')
    return dict(budget_seconds=budget, max_outer=outer_limit, shared_restoration_limit=restoration_limit,
                numerical_seconds=numerical_seconds, numerical_seconds_field=numerical_field,
                outer_iterations=iterations, iteration_field=iteration_field,
                iteration_limit_enforcement='qualified Ipopt max_iterations contract' if ipopt else 'native inclusive counters',
                restoration_iterations=restoration,
                budget_integrity_pass=not reasons, budget_integrity_reasons=reasons,
                budget_exhausted=bool(exhausted), native_status=status,
                explicit_source_budget_failure=bool(explicit_failures),
                explicit_budget_failure_routes=explicit_failures,
                solve_process_seconds=process_seconds if finite_number(process_seconds) else None,
                solve_external_timeout=case.get('solve_external_timeout'),
                scope='Inclusive solver process envelope plus native/core wall/outer/shared-restoration budgets; independent audit outside numerical7200s, separately measured; overlapping timers never added')


def _resolve(path, root):
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def run(case_path, endpoint_path, output_path):
    began = time.monotonic()
    config_path = ROOT / 'INSTALLATION.json'
    if not config_path.is_file(): config_path = HERE / 'INSTALLATION.json'
    configuration = json.loads(config_path.read_text())
    root = config_path.parent
    case = json.loads(Path(case_path).read_text()); case = case.get('case', case)
    inp = _resolve(case['input'], root)
    if inp.is_symlink() or sha(inp) != case['input_sha256']:
        raise ValueError('Frozen original QCQP differs from its contract')
    auditor_path = _resolve(configuration['native_auditor'], root)
    if sha(auditor_path) != configuration['native_auditor_sha256']:
        raise ValueError('Original coefficient auditor differs from installation')
    auditor = module(auditor_path, '_fresh_mega_original_auditor')
    endpoint = json.loads(Path(endpoint_path).read_text())
    timing = timing_and_budget(endpoint, case)
    native = case.get('method') not in ('ipopt_qcqp','ipopt_mumps') and endpoint.get('solver') != 'Ipopt'
    route = endpoint.get('native_al_fallback') or {}
    route_valid = (not native or (origin(endpoint, case) == 'lazy'
                   and route.get('policy') == 'off'
                   and all(route.get(k) is False for k in ('enabled', 'attempted', 'accepted'))
                   and (endpoint.get('native_augmented_repair') or {}).get('enabled') is not True))
    answer = dict(schema='fresh-original-qcqp-audit-v2', audit_available=False,
                  strict_qcqp=None, mathematical_strict_qcqp=None,
                  budget_and_timing=timing, native_status=timing['native_status'],
                  solution_origin=origin(endpoint, case), primal=None, dual=None,
                  native_route_integrity_pass=route_valid,
                  complementarity=None, objective=None, tolerance=TOLERANCE,
                  criterion='all original primal, dual, complementarity <= 1e-6',
                  input_sha256=case['input_sha256'], endpoint_sha256=sha(endpoint_path),
                  native_auditor_sha256=sha(auditor_path), multipliers_refitted=False,
                  native_status_does_not_determine_success=True)
    try:
        model = auditor.read_model(inp)
        # The scalar original auditor already supports both native and Ipopt
        # arrays; keep the original endpoint in its independently checked form.
        result = auditor.audit(model, endpoint, tolerance=TOLERANCE)
        metrics = [result[k] for k in ('feasibility', 'dual', 'complementarity')]
        if not all(finite_number(v) for v in metrics) or type(result.get('strict')) is not bool:
            raise ValueError('Original auditor returned invalid residuals')
        decision = all(v <= TOLERANCE for v in metrics)
        if result['strict'] != decision:
            raise ValueError('Original auditor decision disagrees with residuals')
        answer.update(audit_available=True, qcqp=result,
                      mathematical_strict_qcqp=decision,
                      primal=result['feasibility'], dual=result['dual'],
                      complementarity=result['complementarity'], objective=result['objective'])
        if configuration.get('precise_auditor'):
            precise_path = _resolve(configuration['precise_auditor'], root)
            if sha(precise_path) != configuration.get('precise_auditor_sha256'):
                raise ValueError('High precision original auditor differs from installation')
            precise = module(precise_path, '_fresh_mega_precise_original_auditor')
            interval = precise.audit_decimal(model, endpoint)
            answer['high_precision_original_audit'] = interval
            agreement = type(interval.get('strict')) is bool and interval['strict'] == decision
            answer['precision_agreement'] = agreement
            answer['precise_auditor_sha256'] = sha(precise_path)
            if not agreement:
                answer.update(mathematical_strict_qcqp=None,
                              audit_indeterminate_reason='scalar_and_interval_audits_disagree_or_interval_undetermined')
        if (answer['mathematical_strict_qcqp'] is not None and timing['budget_integrity_pass'] and route_valid
                and case.get('solve_external_timeout') is not True):
            answer['strict_qcqp'] = answer['mathematical_strict_qcqp']
    except (ValueError, KeyError, TypeError, OverflowError, ArithmeticError) as error:
        answer.update(audit_available=False, strict_qcqp=None, mathematical_strict_qcqp=None,
                      audit_error=f'{type(error).__name__}: {error}')
    answer['audit_seconds'] = time.monotonic() - began
    write(output_path, answer)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--endpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    answer = run(args.case_json, args.endpoint, args.out)
    print(json.dumps({k: answer[k] for k in ('strict_qcqp', 'solution_origin', 'primal', 'dual', 'complementarity')}, allow_nan=False))
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (OSError, ValueError, KeyError, ImportError) as error:
        print('AUDIT_QCQP_UNAVAILABLE: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
