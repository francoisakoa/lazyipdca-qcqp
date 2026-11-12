#!/usr/bin/env python3
"""Small adversarial checks for the frozen historical timing cohort."""
from pathlib import Path
import json
import unittest

import BUILD_COHORT as c


class CohortTests(unittest.TestCase):
    def test_one_second_is_excluded(self):
        row=dict(classification='strict',strict=True,seconds_process=1.,primal=0.,dual=0.,complementarity=0.)
        self.assertFalse(c.selected(row))
        row['seconds_process']=1.0000000001
        self.assertTrue(c.selected(row))

    def test_nonfinite_and_unqualified_are_excluded(self):
        row=dict(classification='strict',strict=True,seconds_process=float('inf'),primal=0.,dual=0.,complementarity=0.)
        self.assertFalse(c.selected(row))
        row.update(classification='indeterminate',seconds_process=10.)
        self.assertFalse(c.selected(row))

    def test_false_strict_residual_is_refused(self):
        row=dict(case_id='synthetic',classification='strict',strict=True,seconds_process=10.,primal=0.,dual=1.01e-6,complementarity=0.)
        with self.assertRaises(ValueError):c.selected(row)

    def test_frozen_matrix_and_hash(self):
        data=json.loads(Path(__file__).with_name('COHORTE_CHRONO_FIGEE.json').read_text())
        recorded=data.pop('cohort_content_sha256')
        self.assertEqual(c.digest(data),recorded)
        self.assertEqual(data['counts']['cohorts'],c.EXPECTED)
        self.assertEqual(len({j['job_id']for j in data['jobs']}),550)
        self.assertEqual(len({p['pair_id']for p in data['pairs']}),275)
        self.assertEqual(len({p['case_id']for p in data['pairs']}),179)
        jobs={j['job_id']:j for j in data['jobs']}
        for pair in data['pairs']:
            native,reference=jobs[pair['native_job_id']],jobs[pair['reference_job_id']]
            self.assertEqual(native['input_sha256'],reference['input_sha256'])
            self.assertEqual(native['start_sha256'],reference['start_sha256'])
            self.assertGreater(native['historical_seconds_process'],1.)
            self.assertGreater(reference['historical_seconds_process'],1.)


if __name__=='__main__':unittest.main()
