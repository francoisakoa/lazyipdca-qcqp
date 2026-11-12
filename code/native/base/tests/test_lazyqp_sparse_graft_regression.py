#!/usr/bin/env python3
"""Grafted sparse single-loop kernel regression on the bundled PGLib case14.

tests/regression/pglib_opf_case14_ieee_r3p28.qcq was exported once from
pglib_opf_case14_ieee.m (PGLib-OPF v23.07, sha256
bd5c568621de65e4b0922317010868bc7fa94173807faa10ea8fdbbe77c28106) with
export_pglib_qcqp_r3p23.py --flow-limit apparent --power-flow-start
--no-power-flow-voltage-continuation.

The r3p28 control profile (LazyQP single-loop kernel), the same envelope
with --inner-backend lazyqp-sparse (the graft) and --algorithm lazy-newton
must all solve it to the same objective, pass an independent unscaled KKT
check with the file coefficients, and report their own kernel labels.
Outside the single-loop route the graft must delegate to LazyQP exactly
(identical iteration counters on HS116).

Usage: test_lazyqp_sparse_graft_regression.py <lazyipdca_qcqp_cli>
"""
from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from qcq_model_r3p23 import read_qcq  # noqa: E402
from benchmark_case118_r3p28 import independent_kkt, TOL, CONTROL_PROFILE, LAZY_NEWTON_PROFILE, GRAFT_INNER_BACKEND  # noqa: E402

CASE14 = ROOT / 'tests' / 'regression' / 'pglib_opf_case14_ieee_r3p28.qcq'
HS116 = ROOT / 'tests' / 'regression' / 'HS116.qcq'
CASE14_REFERENCE = 2178.080428  # PGLib-OPF v23.07 case14 optimum (control and Ipopt agree)


def run(binary, qcq, options):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'r.json'
        command = [str(binary), '--problem', str(qcq), '--json', str(out), '--quiet', *options]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return proc.returncode, proc.stdout, json.loads(out.read_text())


def main(argv):
    if len(argv) != 2:
        print(__doc__); return 2
    binary = Path(argv[1]).resolve(strict=True)
    backend_query = subprocess.run(
        [str(binary), '--linear-backends'], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert backend_query.returncode == 0, backend_query.stdout
    if 'ldl' not in {item.strip() for item in
                     backend_query.stdout.strip().split(',')}:
        print('[SKIP] lazyqp-sparse graft requires the historical '
              'SuiteSparse LDL backend')
        return 77
    model = read_qcq(CASE14)
    assert (model.n, len(model.equalities), len(model.inequalities)) == (118, 108, 107), 'unexpected case14 dimensions'
    control = json.loads(CONTROL_PROFILE.read_text())
    lazy_newton = json.loads(LAZY_NEWTON_PROFILE.read_text())
    results = {}
    for name, options in (('control', control), ('graft', control + ['--inner-backend', GRAFT_INNER_BACKEND]),
                          ('lazy-newton', lazy_newton)):
        rc, log, result = run(binary, CASE14, options)
        assert rc == 0 and result['status'] == 'solved', (name, rc, result.get('message'), log[-600:])
        objective, residuals = independent_kkt(model, result)
        assert all(0 <= v <= TOL for v in residuals.values()), (name, residuals)
        assert abs(objective - CASE14_REFERENCE) <= 1e-5 * max(1.0, abs(CASE14_REFERENCE)), (name, objective)
        assert result['ipopt_desired_convergence'] is True, name
        results[name] = result
    ctl, graft, ln = results['control'], results['graft'], results['lazy-newton']
    assert ctl['linear_solver_backend'].startswith('lazyqp/'), ctl['linear_solver_backend']
    assert graft['linear_solver_backend'].startswith(GRAFT_INNER_BACKEND + '/'), graft['linear_solver_backend']
    assert ln['effective_outer_algorithm'] == 'lazy-newton'
    for r in (ctl, graft):
        assert r['effective_outer_algorithm'] == 'lazy-dca'
        assert r['dca_single_loop_attempts'] >= 1 and r['dca_single_loop_acceptances'] >= 1, r['dca_single_loop_attempts']
        assert r['dca_single_loop_full_refit_calls'] == 1 and r['multiplier_refit_attempts'] == 0
        assert r['lazy_dca_equality_model'] == 'condensed-retraction'
    assert graft['outer_iterations'] <= 3 * max(1, ctl['outer_iterations']), (graft['outer_iterations'], ctl['outer_iterations'])
    # outside the single-loop route the graft delegates exactly
    base = ['--linear-backend', 'ldl', '--qp-backend', 'ldl', '--feas-tol', '1e-6', '--opt-tol', '1e-6',
            '--dual-inf-tol', '1e-6', '--compl-inf-tol', '1e-6']
    rc_a, _log_a, a = run(binary, HS116, base + ['--inner-backend', 'lazyqp'])
    rc_b, _log_b, b = run(binary, HS116, base + ['--inner-backend', GRAFT_INNER_BACKEND])
    assert rc_a == 0 and rc_b == 0 and a['status'] == b['status'] == 'solved'
    for k in ('outer_iterations', 'inner_iterations', 'kkt_factorizations'):
        assert a[k] == b[k], (k, a[k], b[k])
    assert abs(a['objective'] - b['objective']) <= 1e-9 * max(1.0, abs(a['objective']))
    print('lazyqp-sparse graft regression: OK (control %d / graft %d / lazy-newton %d iterations)' % (
        ctl['outer_iterations'], graft['outer_iterations'], ln['outer_iterations']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
