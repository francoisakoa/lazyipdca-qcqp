#!/usr/bin/env python3
"""Lazy-newton regression on the bundled CUTEst QCQP files.

BT1 and HS116 must be solved by ``--algorithm lazy-newton`` with the same
objective as the Lazy-DCA route, the certificate must pass an independent
unscaled KKT check (file coefficients, exported multipliers), the working
set must be lazy (not every inequality modeled), and the route must report
its own counters.  The two files that need a restoration phase (DISCS,
HIMMELBD) are documented as out of scope for this route and are only
required to fail cleanly (non-zero exit, no crash, valid JSON).

Usage: test_lazy_newton_regression.py <lazyipdca_qcqp_cli>
"""
from __future__ import annotations
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from qcq_model_r3p23 import read_qcq  # noqa: E402
from benchmark_case118_r3p28 import independent_kkt, TOL  # noqa: E402
IDENTITY = json.loads((ROOT / 'release_identity_g10.json').read_text())
VERSION, API_VERSION = IDENTITY['version'], IDENTITY['api_version']

SOLVED = {'BT1': (-1.0, 1e-6), 'HS116': (97.5875096, 1e-5)}
OUT_OF_SCOPE = ('DISCS', 'HIMMELBD')


def run(binary, name, extra=()):
    qcq = ROOT / 'tests' / 'regression' / (name + '.qcq')
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / (name + '.json')
        command = [str(binary), '--problem', str(qcq), '--algorithm', 'lazy-newton',
                   '--linear-backend', 'ldl', '--qp-backend', 'ldl', '--quiet',
                   '--feas-tol', '1e-6', '--opt-tol', '1e-6', '--dual-inf-tol', '1e-6',
                   '--compl-inf-tol', '1e-6', '--max-outer', '300', '--json', str(out), *extra]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        result = json.loads(out.read_text())
    return proc.returncode, proc.stdout, result, read_qcq(qcq)


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
        print('[SKIP] lazy-newton regression requires the historical '
              'SuiteSparse LDL backend')
        return 77
    for name, (ref, tol) in SOLVED.items():
        variants = [(), ('--ln-globalization', 'merit'), ('--ln-ordering', 'metis'), ('--ln-condense-max-support', '0')]
        # Pure majorant curvature converges only linearly on HS116 (active
        # nonconvex inequalities); the adaptive default is the supported mode.
        if name == 'BT1':
            variants.append(('--ln-inequality-curvature', 'majorant'))
        for extra in variants:
            rc, log, result, model = run(binary, name, extra)
            assert rc == 0 and result['status'] == 'solved', (name, extra, rc, result.get('message'), log[-800:])
            assert result['version'] == VERSION and result['api_version'] == API_VERSION, result['version']
            assert result['effective_outer_algorithm'] == 'lazy-newton'
            assert result['termination_multiplier_source'] == 'lazy-newton-primal-dual'
            assert result['ipopt_desired_convergence'] is True
            objective, residuals = independent_kkt(model, result)
            assert abs(objective - ref) <= tol, (name, extra, objective, ref)
            assert all(0 <= v <= TOL for v in residuals.values()), (name, extra, residuals)
            assert result['lazy_newton_iterations'] >= 1
            assert result['lazy_newton_certificate_evaluations'] >= 1
            assert result['lazy_newton_factorizations'] >= result['lazy_newton_iterations']
            ws = result['lazy_newton_working_set_final']
            assert 0 <= ws <= len(model.inequalities), ws
            if extra == ('--ln-condense-max-support', '0'):
                assert result['lazy_newton_condensed_rows_max'] == 0, result['lazy_newton_condensed_rows_max']
            if extra == ('--ln-ordering', 'metis'):
                assert result['linear_solver_backend'].endswith('_metis'), result['linear_solver_backend']
            if extra == ('--ln-inequality-curvature', 'majorant'):
                assert result['lazy_newton_exact_curvature_steps'] == 0
            if name == 'HS116':
                assert ws < len(model.inequalities), 'working set is not lazy on HS116'
                assert result['lazy_newton_iterations'] <= 120, result['lazy_newton_iterations']
    for name in OUT_OF_SCOPE:
        rc, log, result, _model = run(binary, name)
        assert rc == 1, (name, rc, log[-500:])  # clean failure, no crash
        assert result['status'] != 'solved', (name, result['status'])
        assert isinstance(result.get('message'), str) and result['message'], name
        assert math.isfinite(result['elapsed_seconds'])
    print('lazy-newton regression: OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
