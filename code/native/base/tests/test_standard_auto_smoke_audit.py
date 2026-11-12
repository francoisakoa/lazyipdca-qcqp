#!/usr/bin/env python3
"""Regression for the independent smoke auditor's project/MAX criterion."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'tools/smoke_standard_auto.py'
spec=importlib.util.spec_from_file_location('standard_auto_smoke',path)
smoke=importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeAudit(unittest.TestCase):
    def test_max_and_sum_are_distinct_certificate_scopes(self):
        case=dict(n=3,lb=[0,0,0],ub=[1,1,1],objective=(0,[],[]),eq=[],iq=[],expected=[.5,.5,.5])
        endpoint=dict(x=[.5,.5,.5],certificate_equality_multipliers=[],
            certificate_inequality_multipliers=[],certificate_lower_multipliers=[5.12e-7]*3,
            certificate_upper_multipliers=[5.12e-7]*3)
        result=smoke.audit(case,endpoint,1e-6)
        self.assertEqual(result['primal'],0)
        self.assertEqual(result['stationarity'],0)
        self.assertAlmostEqual(result['complementarity_max'],2.56e-7)
        self.assertAlmostEqual(result['complementarity_sum'],1.536e-6)
        self.assertTrue(result['strict'])
        self.assertFalse(result['stronger_sum_pass'])
        endpoint['certificate_lower_multipliers']=[3e-6]*3
        endpoint['certificate_upper_multipliers']=[3e-6]*3
        result=smoke.audit(case,endpoint,1e-6)
        self.assertFalse(result['strict'])
        self.assertFalse(result['stronger_sum_pass'])


if __name__=='__main__':unittest.main()
