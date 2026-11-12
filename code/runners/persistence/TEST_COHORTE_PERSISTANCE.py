#!/usr/bin/env python3
from pathlib import Path
import json
import unittest
import BUILD_COHORTE_PERSISTANCE as cohort


class FrozenSelectionTests(unittest.TestCase):
    def test_frozen_selection_reproduces(self):
        d=json.loads(Path(__file__).with_name('COHORTE_PERSISTANCE.json').read_text())
        self.assertEqual(cohort.freeze(),d)
        checksum=d.pop('cohort_content_sha256');self.assertEqual(cohort.digest(d),checksum)

    def test_pairs_and_size_strata(self):
        d=cohort.freeze();self.assertEqual(len(d['pairs']),24)
        self.assertEqual(len({p['case_id'] for p in d['pairs']}),12)
        for case in d['cases']:
            rows=[p for p in d['pairs'] if p['case_id']==case['id']]
            self.assertEqual({p['backend'] for p in rows},{'ma27','mumps'})
            self.assertEqual(rows[0]['input_sha256'],rows[1]['input_sha256'])
            self.assertEqual(rows[0]['start_sha256'],rows[1]['start_sha256'])
            self.assertGreater(case['dimensions']['quadratic_inequalities'],0)
        self.assertEqual(len(d['strata']),6)
        self.assertTrue(all(len(s['selected'])==2 for s in d['strata']))


if __name__=='__main__':unittest.main()
