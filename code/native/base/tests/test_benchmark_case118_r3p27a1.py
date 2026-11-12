#!/usr/bin/env python3
"""Ensure timing cannot bypass independent original-QCQP certification."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from benchmark_case118_r3p27a1 import BUILD_ID, VERSION, check_lazy
from qcq_model_r3p23 import HessianEntry, QuadraticFunction, QCQPModel


class CertificateGate(unittest.TestCase):
    def setUp(self):
        self.model = QCQPModel('gate', 1, (0.,), (-2.,), (2.,),
            QuadraticFunction('f', 0., (), (HessianEntry(0, 0, 2.),)), (),
            (QuadraticFunction('g', -1., ((0, 1.),), ()),))
        self.result = dict(status='solved', version=VERSION, build_id=BUILD_ID,
            api_version=4311, certificate_equality_multipliers=[],
            certificate_inequality_multipliers=[0.], certificate_lower_multipliers=[0.],
            certificate_upper_multipliers=[0.], x=[0.], objective=0.,
            dual_infeasibility_inf=0., complementarity_inf=0., scaled_nlp_error=0.,
            dca_single_loop_full_refit_calls=1, multiplier_refit_attempts=0,
            lazy_dca_equality_retraction_attempts=0,
            lazy_dca_equality_model='condensed-retraction')

    def test_valid_certificate(self):
        objective, residuals = check_lazy(self.model, self.result, 'sparse')
        self.assertEqual(objective, 0.)
        self.assertEqual(max(residuals.values()), 0.)

    def test_reject_forged_success(self):
        mutations = [dict(x=[float('nan')]), dict(x=[3.]), dict(x=[.1], objective=.01),
            dict(certificate_upper_multipliers=[.5]),
            dict(certificate_inequality_multipliers=[-1.]),
            dict(certificate_equality_multipliers=None), dict(objective=1.),
            dict(dca_single_loop_full_refit_calls=2), dict(multiplier_refit_attempts=1),
            dict(lazy_dca_equality_retraction_attempts=1), dict(build_id='stale'),
            dict(lazy_dca_equality_model='linearized'), dict(scaled_nlp_error=float('nan'))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                result = copy.deepcopy(self.result)
                result.update(mutation)
                with self.assertRaises(ValueError):
                    check_lazy(self.model, result, 'sparse')

    def test_tangent_must_be_exercised(self):
        self.result['lazy_dca_equality_model'] = 'tangent-inertia'
        with self.assertRaises(ValueError):
            check_lazy(self.model, self.result, 'tangent')
        self.result['equality_tangent_inertia_acceptances'] = 1
        check_lazy(self.model, self.result, 'tangent')


if __name__ == '__main__':
    unittest.main()
