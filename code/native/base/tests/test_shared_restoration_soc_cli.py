#!/usr/bin/env python3
"""Real CLI contract checks; optional tiny native accounting integration."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', required=True, type=Path)
parser.add_argument('--native', action='store_true', help='also execute tiny Lazy/MUMPS QCQPs')
ARGS = parser.parse_args()
BINARY = ARGS.binary.resolve(strict=True)
FLAGS = ('--shared-restoration-budget', '--original-gr-kkt-soc')


class CliBooleanContracts(unittest.TestCase):
    def invoke(self, arguments):
        return subprocess.run([str(BINARY)] + list(arguments), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)

    def test_help_documents_both_opt_in_flags(self):
        result = self.invoke(['--help'])
        self.assertEqual(result.returncode, 0, result.stdout)
        for flag in FLAGS:
            self.assertIn(flag + ' 0|1', result.stdout)

    def test_exact_zero_and_one_are_accepted(self):
        for flag in FLAGS:
            for value in ('0', '1'):
                with self.subTest(flag=flag, value=value):
                    result = self.invoke([flag, value, '--help'])
                    self.assertEqual(result.returncode, 0, result.stdout)

    def test_invalid_or_partially_numeric_tokens_are_rejected(self):
        for flag in FLAGS:
            for value in ('', '2', '-1', 'true', 'false', '00', '1.0', '1x', '0x1', '--help'):
                with self.subTest(flag=flag, value=value):
                    result = self.invoke([flag, value])
                    self.assertEqual(result.returncode, 2, result.stdout)
                    self.assertIn(flag[2:] + ' expects 0|1', result.stdout)

    def test_missing_value_is_rejected(self):
        for flag in FLAGS:
            with self.subTest(flag=flag):
                result = self.invoke([flag])
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn('missing value after ' + flag, result.stdout)

    def test_flags_can_be_combined_in_either_order(self):
        for left, right in (FLAGS, FLAGS[::-1]):
            for a, b in (('0', '1'), ('1', '0'), ('1', '1')):
                with self.subTest(left=left, a=a, b=b):
                    result = self.invoke([left, a, right, b, '--help'])
                    self.assertEqual(result.returncode, 0, result.stdout)


@unittest.skipUnless(ARGS.native, 'native integration is opt-in')
class CliNativeBudgetWiring(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='lazy-cli-shared-budget-')
        self.directory = Path(self.temporary.name)
        self.input = self.directory / 'input.qcq'
        # Affine x-1=0 is far from x0; y^2+1=0 has no real feasible point.
        self.input.write_text(
            'LAZYIPDCA_QCQP_V1\nname generic_cli_budget_fixture\nn 2\nx0 256 0\n'
            'lb -INF -INF\nub INF INF\nobjective\nquadratic\nname f\nconstant 0\n'
            'linear 1 0\nhessian_nnz 0\nend_quadratic\nequalities 2\n'
            'quadratic\nname affine\nconstant -1\nlinear 1 0\nhessian_nnz 0\nend_quadratic\n'
            'quadratic\nname impossible\nconstant 1\nlinear 0 0\nhessian_nnz 1\n1 1 2\n'
            'end_quadratic\ninequalities 1\nquadratic\nname bound\nconstant -512\n'
            'linear 1 0\nhessian_nnz 0\nend_quadratic\nend_problem\n')
        self.original = self.input.read_bytes()
        self.serial = 0

    def tearDown(self):
        self.temporary.cleanup()

    def solve(self, budget, flags, outer=12):
        self.serial += 1
        endpoint = self.directory / ('endpoint_%d.json' % self.serial)
        command = [str(BINARY), '--problem', str(self.input), '--json', str(endpoint),
                   '--algorithm', 'lazy-dca', '--dca-engine', 'majorants', '--linear-backend', 'mumps',
                   '--linear-metis', '1', '--standard-r10-policy', 'off', '--native-al-fallback', 'off',
                   '--dca-rank-presolve', '0', '--free-epigraph-presolve', '0',
                   '--majorant-startup-policy', 'auto', '--majorant-adaptive-globalization', '1',
                   '--restoration-restarts', '0', '--restoration-minimax-iters', '0',
                   '--restoration-iters', str(budget), '--max-outer', str(outer), '--time-limit', '5',
                   '--display-level', '1'] + flags
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(('LAZY_', 'ELASTIC_', 'KERNEL_', 'IPDCA_'))}
        env.update(OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 env=env, cwd=self.directory, timeout=15)
        self.assertIn(process.returncode, (0, 1), process.stdout)
        self.assertTrue(endpoint.is_file(), process.stdout)
        raw = json.loads(endpoint.read_text())
        self.assertEqual(self.input.read_bytes(), self.original, 'frozen original fixture changed')
        self.assertFalse(raw['status'].startswith('solved'), 'infeasible equality acquired a false native success')
        self.assertFalse(raw['native_al_fallback']['attempted'])
        self.assertFalse(raw['native_al_fallback']['accepted'])
        return raw, process.stdout

    def test_shared_flag_bounds_actual_native_work(self):
        affine, terminal = 0, 0
        for budget in (0, 1, 2, 3):
            with self.subTest(budget=budget):
                raw, trace = self.solve(budget, ['--shared-restoration-budget', '1'])
                self.assertLessEqual(raw['restoration_iterations'], budget)
                self.assertLessEqual(raw['outer_iterations'], 12)
                affine += '[AFFINE-START-RESTORATION]' in trace
                terminal += '[RESTORATION-BUDGET] phase=bounded-origin action=skip' in trace
                if budget == 0:
                    self.assertNotIn('[AFFINE-START-RESTORATION]', trace)
        self.assertGreater(affine, 0, 'CLI integration did not reach affine startup')
        self.assertGreater(terminal, 0, 'CLI integration did not reach terminal budget guard')

    def test_explicit_zero_preserves_the_unchanged_default(self):
        default, _ = self.solve(1, [])
        disabled, _ = self.solve(1, ['--shared-restoration-budget', '0', '--original-gr-kkt-soc', '0'])
        for field in ('status', 'outer_iterations', 'restoration_iterations', 'x',
                      'certificate_equality_multipliers', 'certificate_inequality_multipliers',
                      'certificate_lower_multipliers', 'certificate_upper_multipliers'):
            self.assertEqual(default[field], disabled[field], field)

    def test_independent_outer_and_restoration_limits_remain_valid(self):
        raw, _ = self.solve(600, ['--shared-restoration-budget', '1'], outer=1)
        self.assertLessEqual(raw['restoration_iterations'], 600)
        self.assertLessEqual(raw['outer_iterations'], 1)


if __name__ == '__main__':
    print('CLI_SHA256=' + hashlib.sha256(BINARY.read_bytes()).hexdigest(), flush=True)
    unittest.main(argv=[__file__], verbosity=2)
