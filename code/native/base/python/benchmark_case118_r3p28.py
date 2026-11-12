#!/usr/bin/env python3
"""Frozen-QCQP comparison for r3p28: Ipopt, r3p27 sparse control, the grafted
sparse single-loop kernel (control envelope + --inner-backend lazyqp-sparse)
and the lazy-newton route.

Every measured endpoint is independently checked with the file coefficients
and the exported certificate multipliers (unscaled KKT residuals <= TOL).
No tolerance is relaxed for timing.  The script also exposes every
``--ln-*`` control of the C++ binary so that the three-way CLI contract
(binary parser, this wrapper, shell runner) stays in sync; an option that is
not given is not forwarded, so the binary defaults remain authoritative.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from qcq_model_r3p23 import read_qcq

VERSION = '0.4.28r3p28g1-atomic-graft'
API_VERSION = 4327
BUILD_ID = 'v' + VERSION + '-api-' + str(API_VERSION)
TOL = 1e-6
ROOT = Path(__file__).resolve().parents[1]
ALL_MODES = ('ipopt', 'control', 'graft', 'lazy-newton')
GRAFT_INNER_BACKEND = 'lazyqp-sparse'
CONTROL_PROFILE = ROOT / 'python' / 'case118_control_profile_r3p28.json'
LAZY_NEWTON_PROFILE = ROOT / 'python' / 'case118_lazy_newton_profile_r3p28.json'

# (wrapper option, binary option, argparse kwargs).  Kept in one table so the
# contract test can derive the set from this file.
LN_OPTIONS = [
    ('--ln-mu-init', 'float'),
    ('--ln-mu-kappa', 'float'),
    ('--ln-mu-theta', 'float'),
    ('--ln-kappa-epsilon', 'float'),
    ('--ln-activation', 'float'),
    ('--ln-envelope', 'float'),
    ('--ln-drop-patience', 'int'),
    ('--ln-drop-distance', 'float'),
    ('--ln-delta-c', 'float'),
    ('--ln-delta-x-init', 'float'),
    ('--ln-delta-x-min', 'float'),
    ('--ln-kkt-refine', 'int'),
    ('--ln-kkt-berr-tol', 'float'),
    ('--ln-max-soc', 'int'),
    ('--ln-bound-push', 'float'),
    ('--ln-tau-min', 'float'),
    ('--ln-max-line-search', 'int'),
    ('--ln-prox-alpha-min', 'float'),
    ('--ln-mult-init-max', 'float'),
    ('--ln-scaling-max-gradient', 'float'),
    ('--ln-inequality-curvature', 'str'),
    ('--ln-ordering', 'str'),
    ('--ln-adaptive-curvature-delta', 'float'),
    ('--ln-condense-max-support', 'int'),
    ('--ln-globalization', 'str'),
    ('--ln-transport-complementarity-floor', 'float'),
]
LN_FLAG_OPTIONS = [('--ln-quadratic-correction', '--no-ln-quadratic-correction')]


def finite(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_kkt(model, result):
    names = ('certificate_equality_multipliers', 'certificate_inequality_multipliers',
             'certificate_lower_multipliers', 'certificate_upper_multipliers')
    vectors = [result.get(k) for k in names]
    sizes = (len(model.equalities), len(model.inequalities), model.n, model.n)
    for k, v, n in zip(names, vectors, sizes):
        if not isinstance(v, list) or len(v) != n or not all(finite(x) for x in v):
            raise ValueError('invalid certificate vector ' + k)
    equality, inequality, lower, upper = vectors
    if any(x < 0 for v in (inequality, lower, upper) for x in v):
        raise ValueError('negative inequality/bound multiplier')
    x = result.get('x')
    if not isinstance(x, list) or len(x) != model.n or not all(finite(v) for v in x):
        raise ValueError('invalid primal vector')
    residuals = model.residuals(x)
    residuals.update(model.kkt_residuals(x, equality + inequality, [u - l for u, l in zip(upper, lower)]))
    for j in range(model.n):
        if model.lb[j] == model.ub[j]:
            continue
        for b, z in ((model.lb[j], lower[j]), (model.ub[j], upper[j])):
            if math.isfinite(b):
                residuals['complementarity_inf'] = max(residuals['complementarity_inf'], abs(z * (x[j] - b)))
            elif z != 0:
                raise ValueError('multiplier on an absent bound')
    for k, v in residuals.items():
        if not finite(v) or not 0 <= v <= TOL:
            raise ValueError(f'independent original KKT failed: {k}={v}')
    objective = model.objective_value(x)
    if not finite(result.get('objective')) or abs(objective - result['objective']) > 1e-7 * max(1, abs(objective)):
        raise ValueError('objective mismatch with original file')
    return objective, residuals


def check_lazy(model, result, mode):
    if result.get('status') != 'solved':
        raise ValueError(str(result.get('message', 'not solved')))
    if result.get('version') != VERSION or result.get('build_id') != BUILD_ID or result.get('api_version') != API_VERSION:
        raise ValueError('incorrect solver identity')
    objective, residuals = independent_kkt(model, result)
    for k in ('dual_infeasibility_inf', 'complementarity_inf', 'scaled_nlp_error'):
        if not finite(result.get(k)) or not 0 <= result[k] <= TOL:
            raise ValueError('invalid reported KKT ' + k)
    if result.get('ipopt_desired_convergence') is not True:
        raise ValueError('Ipopt-style desired convergence not reported')
    if mode in ('control', 'graft'):
        if result.get('effective_outer_algorithm') != 'lazy-dca':
            raise ValueError(mode + ' did not run the Lazy-DCA route')
        backend = str(result.get('linear_solver_backend', ''))
        if mode == 'graft' and not backend.startswith(GRAFT_INNER_BACKEND + '/'):
            raise ValueError('graft did not run the sparse single-loop kernel: ' + backend)
        if mode == 'control' and not backend.startswith('lazyqp/'):
            raise ValueError('control did not run the LazyQP kernel: ' + backend)
        if result.get('dca_single_loop_attempts', 0) < 1 or result.get('dca_single_loop_acceptances', 0) < 1:
            raise ValueError('single-loop transactions not recorded')
        if result.get('dca_single_loop_full_refit_calls') != 1 or result.get('multiplier_refit_attempts') != 0:
            raise ValueError('single-loop final-only multiplier-fit contract violated')
        if result.get('lazy_dca_equality_retraction_attempts') != 0:
            raise ValueError('unexpected equality retraction')
        if result.get('lazy_dca_equality_model') != 'condensed-retraction':
            raise ValueError('wrong equality model')
    else:
        if result.get('effective_outer_algorithm') != 'lazy-newton':
            raise ValueError('lazy-newton route not taken')
        if result.get('termination_multiplier_source') != 'lazy-newton-primal-dual':
            raise ValueError('certificate multipliers do not come from the primal-dual iterate')
        if result.get('lazy_dca_equality_model') != 'curvature-newton':
            raise ValueError('wrong equality model label')
        if not isinstance(result.get('lazy_newton_iterations'), int) or result['lazy_newton_iterations'] < 1:
            raise ValueError('no lazy-newton iteration recorded')
        if result.get('lazy_newton_certificate_evaluations', 0) < 1:
            raise ValueError('original-QCQP certificate never evaluated')
        ws = result.get('lazy_newton_working_set_final')
        if not isinstance(ws, int) or not 0 <= ws <= len(model.inequalities):
            raise ValueError('invalid final working set size')
        if ws >= len(model.inequalities) and len(model.inequalities) > 0:
            raise ValueError('working set is not lazy: every inequality was modeled')
        if result.get('lazy_newton_factorizations', 0) < result['lazy_newton_iterations']:
            raise ValueError('factorization counter inconsistent')
    return objective, residuals


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--case', type=Path, help='PGLib/MATPOWER .m file exported once with export_pglib_qcqp_r3p23.py')
    source.add_argument('--qcq', type=Path, help='already exported immutable .qcq file')
    parser.add_argument('--binary', type=Path, required=True, help='r3p28 lazyipdca_qcqp_cli')
    parser.add_argument('--output', type=Path, help='campaign directory (must not exist)')
    parser.add_argument('--base', type=Path, default=Path('/data/projects/IPDCA_QCQP'))
    parser.add_argument('--modes', default=','.join(ALL_MODES), help='comma list among ipopt,control,graft (control envelope + --inner-backend lazyqp-sparse),lazy-newton')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1, help='OMP threads for the control profile (lazy-newton is serial)')
    parser.add_argument('--time-limit', type=float, default=1800)
    parser.add_argument('--target-seconds', type=float, default=.2)
    parser.add_argument('--dimensions', default='1088,980,979', help='expected n,eq,ineq or "any"')
    parser.add_argument('--max-outer', type=int, default=300, help='lazy-newton outer iteration cap')
    for opt, kind in LN_OPTIONS:
        parser.add_argument(opt, type={'float': float, 'int': int, 'str': str}[kind], default=None,
                            help='forwarded to the binary when given')
    for on, off in LN_FLAG_OPTIONS:
        dest = on[2:].replace('-', '_')
        parser.add_argument(on, dest=dest, action='store_true', default=None, help='forwarded to the binary when given')
        parser.add_argument(off, dest=dest, action='store_false', default=None, help='forwarded to the binary when given')
    return parser


def lazy_newton_overrides(args):
    extra = []
    for opt, _kind in LN_OPTIONS:
        value = getattr(args, opt[2:].replace('-', '_'))
        if value is not None:
            extra += [opt, str(value)]
    for on, off in LN_FLAG_OPTIONS:
        value = getattr(args, on[2:].replace('-', '_'))
        if value is True:
            extra.append(on)
        elif value is False:
            extra.append(off)
    return extra


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = tuple(m.strip() for m in args.modes.split(',') if m.strip())
    if not modes or any(m not in ALL_MODES for m in modes) or len(set(modes)) != len(modes):
        parser.error('--modes must be a non-empty subset of ' + ','.join(ALL_MODES))
    if args.repetitions < 1 or args.warmups < 0 or args.threads < 1 or not finite(args.time_limit) or args.time_limit <= 0 \
            or not finite(args.target_seconds) or args.target_seconds <= 0 or args.max_outer < 1:
        parser.error('positive repetitions/threads/time limits/max-outer required; warmups>=0')
    binary = args.binary.resolve(strict=True)
    banner = subprocess.check_output([str(binary), '--version'], text=True)
    if BUILD_ID not in banner:
        parser.error('the r3p28 binary is required (got: ' + banner.strip() + ')')
    out = (args.output or args.base / 'results' / ('R3P28_CASE118_' + time.strftime('%Y%m%d_%H%M%S') + '_' + str(time.time_ns()))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('CAMPAIGN=' + str(out), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    qcq = out / 'case.shared.qcq'
    if args.qcq:
        shutil.copyfile(args.qcq.resolve(strict=True), qcq)
        provenance = {'input_qcq': str(args.qcq.resolve()), 'input_sha256': digest(args.qcq)}
    else:
        case = args.case.resolve(strict=True)
        command = [sys.executable, str(ROOT / 'python/export_pglib_qcqp_r3p23.py'), str(case),
                   '--output', str(qcq), '--json', str(out / 'export.json'), '--flow-limit', 'apparent',
                   '--power-flow-start', '--no-power-flow-voltage-continuation']
        with (out / 'export.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
        provenance = {'case': str(case), 'case_sha256': digest(case)}
    model = read_qcq(qcq)
    dims = (model.n, len(model.equalities), len(model.inequalities))
    if args.dimensions != 'any':
        expected = tuple(int(v) for v in args.dimensions.split(','))
        if dims != expected:
            raise ValueError(f'expected dimensions {expected}, got {dims}')
    sha = digest(qcq)
    qcq.chmod(0o444)
    control_profile = json.loads(CONTROL_PROFILE.read_text())
    lazy_newton_profile = json.loads(LAZY_NEWTON_PROFILE.read_text()) + ['--max-outer', str(args.max_outer)] + lazy_newton_overrides(args)
    graft_profile = control_profile + ['--inner-backend', GRAFT_INNER_BACKEND]
    for opt in ('--ln-condense-max-support', '--ln-ordering', '--ln-kkt-refine', '--ln-kkt-berr-tol', '--ln-delta-c',
                '--ln-delta-x-init', '--ln-transport-complementarity-floor'):
        value = getattr(args, opt[2:].replace('-', '_'))
        if value is not None:
            graft_profile += [opt, str(value)]
    records = []

    def run(mode, index, warmup):
        stem = ('warmup_' if warmup else 'run_') + str(index) + '_' + mode
        result_path = out / (stem + '.json')
        before = digest(qcq)
        if before != sha:
            raise ValueError('model changed before ' + stem)
        if mode == 'ipopt':
            command = [sys.executable, str(ROOT / 'python/solve_qcq_ipopt_casadi_r3p23.py'), str(qcq),
                       '--json', str(result_path), '--tol', '1e-6', '--constr-viol-tol', '1e-6',
                       '--dual-inf-tol', '1e-6', '--compl-inf-tol', '1e-6', '--max-iter', '3000',
                       '--time-limit', str(args.time_limit), '--linear-solver', 'mumps', '--print-level', '0']
        elif mode == 'control':
            command = [str(binary), '--problem', str(qcq), '--json', str(result_path)] + control_profile + [
                '--time-limit', str(args.time_limit), '--quiet']
        elif mode == 'graft':
            command = [str(binary), '--problem', str(qcq), '--json', str(result_path)] + graft_profile + [
                '--time-limit', str(args.time_limit), '--quiet']
        else:
            command = [str(binary), '--problem', str(qcq), '--json', str(result_path)] + lazy_newton_profile + [
                '--time-limit', str(args.time_limit), '--quiet']
        record = {'mode': mode, 'index': index, 'warmup': warmup, 'command': command, 'sha256_before': before}
        start = time.perf_counter()
        with (out / (stem + '.log')).open('w') as log:
            try:
                process = subprocess.run(command, env=env, stdout=log, stderr=log, timeout=args.time_limit + 60)
                record['returncode'] = process.returncode
            except subprocess.TimeoutExpired:
                record['returncode'] = -1
        record['process_seconds'] = time.perf_counter() - start
        record['sha256_after'] = digest(qcq)
        if record['sha256_after'] != sha:
            raise ValueError('model changed after ' + stem)
        try:
            result = json.loads(result_path.read_text())
            if record['returncode'] != 0:
                raise ValueError('optimizer returned ' + str(record['returncode']) + ': ' + str(result.get('message')))
            if mode == 'ipopt':
                if not result.get('success') or not result.get('kkt_passed') or result.get('model_sha256') != sha:
                    raise ValueError('Ipopt reference failed')
                residuals = result['residuals']
                if any(not finite(v) or v < 0 or v > TOL for v in residuals.values()):
                    raise ValueError('invalid Ipopt KKT')
                objective = result['objective']; seconds = result['solver_seconds']; iterations = result['iterations']
            else:
                objective, residuals = check_lazy(model, result, mode)
                seconds = result['elapsed_seconds']; iterations = result['outer_iterations']
                keys = ('inner_iterations', 'dca_single_loop_factorizations', 'primary_inner_seconds',
                        'equality_model_seconds', 'restoration_total_seconds', 'dca_single_loop_attempts',
                        'dca_single_loop_acceptances', 'linear_solver_backend') if mode in ('control', 'graft') else (
                    'lazy_newton_iterations', 'lazy_newton_factorizations', 'lazy_newton_symbolic_analyses',
                    'lazy_newton_inertia_corrections', 'lazy_newton_working_set_final', 'lazy_newton_working_set_max',
                    'lazy_newton_lazy_cuts', 'lazy_newton_quadratic_correction_acceptances',
                    'lazy_newton_curvature_fallbacks', 'lazy_newton_kkt_dimension', 'lazy_newton_factor_nnz',
                    'lazy_newton_setup_seconds', 'lazy_newton_factorization_seconds', 'lazy_newton_certificate_seconds',
                    'dual_infeasibility_inf', 'complementarity_inf', 'linear_solver_backend')
                for k in keys:
                    record[k] = result.get(k)
            if not finite(objective):
                raise ValueError('nonfinite objective')
            if not finite(seconds) or seconds < 0:
                raise ValueError('invalid solve time')
            record.update(valid=True, objective=objective, residuals=residuals, solver_seconds=seconds, iterations=iterations)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record.update(valid=False, error=str(exc))
        records.append(record)
        (out / 'records.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
        print(stem, 'PASS' if record['valid'] else 'FAIL', record.get('solver_seconds', record.get('error')), flush=True)

    for i in range(args.warmups):
        for mode in modes:
            run(mode, i + 1, True)
    for i in range(args.repetitions):
        k = i % len(modes)
        for mode in modes[k:] + modes[:k]:
            run(mode, i + 1, False)
    reference = [r['objective'] for r in records if r['mode'] == 'ipopt' and r['valid']]
    if 'ipopt' in modes:
        if reference:
            fref = statistics.median(reference)
            for r in records:
                if r['valid'] and abs(r['objective'] - fref) > 1e-5 + 1e-8 * abs(fref):
                    r.update(valid=False, error='objective differs from Ipopt reference')
        else:
            for r in records:
                r.update(valid=False, error='no valid Ipopt reference')
    summary = {'schema': 'lazyipdca-case118-r3p28-v1', 'model_sha256': sha, 'provenance': provenance,
               'dimensions': list(dims), 'tolerance': TOL, 'target_seconds': args.target_seconds,
               'lazy_newton_profile': lazy_newton_profile, 'control_profile': control_profile,
               'graft_profile': graft_profile,
               'environment': {k: env[k] for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS']},
               'all_correct': all(r['valid'] for r in records), 'modes': {}}
    for mode in modes:
        selected = [r for r in records if r['mode'] == mode and not r['warmup']]
        valid = [r for r in selected if r['valid']]
        data = {'valid_runs': len(valid), 'measured_runs': len(selected)}
        if len(valid) == len(selected):
            data['median_solver_seconds'] = statistics.median(r['solver_seconds'] for r in valid)
            data['min_solver_seconds'] = min(r['solver_seconds'] for r in valid)
            data['max_solver_seconds'] = max(r['solver_seconds'] for r in valid)
            data['median_process_seconds'] = statistics.median(r['process_seconds'] for r in valid)
            data['iterations'] = [r['iterations'] for r in valid]
            data['objectives'] = [r['objective'] for r in valid]
            data['target_met'] = data['median_solver_seconds'] <= args.target_seconds
        summary['modes'][mode] = data
    ref_time = summary['modes'].get('ipopt', {}).get('median_solver_seconds')
    ctl_time = summary['modes'].get('control', {}).get('median_solver_seconds')
    ln_time = summary['modes'].get('lazy-newton', {}).get('median_solver_seconds')
    graft_time = summary['modes'].get('graft', {}).get('median_solver_seconds')
    if ln_time:
        summary['speedup_vs_ipopt'] = (ref_time / ln_time) if ref_time else None
        summary['speedup_vs_control'] = (ctl_time / ln_time) if ctl_time else None
    if graft_time:
        summary['graft_speedup_vs_control'] = (ctl_time / graft_time) if ctl_time else None
        summary['graft_speedup_vs_ipopt'] = (ref_time / graft_time) if ref_time else None
    (out / 'records.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    (out / 'benchmark_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k in ('modes', 'speedup_vs_ipopt', 'speedup_vs_control', 'graft_speedup_vs_control', 'graft_speedup_vs_ipopt')}, indent=2), flush=True)
    print('CORRECTNESS=' + ('PASS' if summary['all_correct'] else 'FAIL'), flush=True)
    print('SUMMARY=' + str(out / 'benchmark_summary.json'), flush=True)
    return 0 if summary['all_correct'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
