#!/usr/bin/env python3
"""A worker awaiting post-run checks must stay pending in all campaign totals."""
import concurrent.futures
import copy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import MEGA
from dataset_schedule import prepare_datasets


class PublicationTests(unittest.TestCase):
    def test_raw_false_becomes_indeterminate_without_transient_failure(self):
        case=dict(id='CUTEst__publication',family='CUTEst',source_kind='historical',
            regime='I',input_sha256='0'*64,dimensions=dict(n=1,quadratic_inequalities=1,
            quadratic_equalities=0,linear_equalities=0,linear_inequalities=0))
        cases,datasets=prepare_datasets([case],['R3']);case=cases[0]
        task=dict(id=case['id']+'__R3',case=case,method='R3')
        plan=dict(cases=cases,datasets=datasets,methods=['R3'],tasks=[task],
            cpus=[0],jobs=1,execution_binding_sha256='binding')
        raw=dict(case_id=case['id'],method='R3',strict_qcqp=False,
            status='endpoint_absent',evidence_sha256={},
            work=dict(returncode=1,interrupted=False,external_timeout=False))
        entered=threading.Event();release=threading.Event()

        class PostRunGuard:
            calls=0
            def check(self):
                self.calls+=1
                if self.calls==2:
                    entered.set()
                    if not release.wait(5):raise TimeoutError('Test post-run check not released')

        def legacy_executor(out,plan,task,cpu):
            # Force the former publication window even after the executor fix.
            # The consumer must also reject provisional records defensively.
            with MEGA.executor.RESULT_LOCK:
                MEGA.executor.RUN_RESULTS.setdefault(str(out),{})[task['id']]=raw
            return raw

        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            with patch.object(MEGA.executor,'run_task',legacy_executor), \
                    patch.object(MEGA.executor,'verify_case'), \
                    concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(MEGA.guarded_run_task,out,plan,task,0,PostRunGuard())
                try:
                    self.assertTrue(entered.wait(5),'Worker did not reach post-run integrity check')
                    before=MEGA.summary(out,plan,'running',1)
                    self.assertEqual((before['done'],before['nonstrict'],before['indeterminate']),(0,0,0))
                    self.assertEqual(before['by_dataset']['CUTEst']['done'],0)
                    self.assertEqual(before['by_regime']['I']['methods']['R3']['done'],0)
                    self.assertEqual(MEGA.finalized_records(out),[])
                finally:
                    release.set()
                result=future.result(timeout=5)
            after=MEGA.summary(out,plan,'completed')
            self.assertIsNone(result['strict_qcqp'])
            self.assertEqual(result['observation_classification'],'indeterminate_observation')
            self.assertFalse(result['numerical_failure_classified'])
            self.assertEqual((after['done'],after['nonstrict'],after['indeterminate']),(1,0,1))
            self.assertEqual(after['by_dataset']['CUTEst']['by_method']['R3']['failed'],0)
            self.assertEqual(after['by_dataset']['CUTEst']['by_method']['R3']['indeterminate'],1)
            self.assertEqual(MEGA.finalized_records(out),[result])
            # Integrity flag and normalized classification alone are insufficient:
            # the final publication must be bound to its pre/post proof.
            for missing in ('integrity_verified','observation_classification','integrity_receipt'):
                incomplete=copy.deepcopy(result);incomplete.pop(missing)
                self.assertFalse(MEGA.is_finalized_observation(incomplete,task['id']))
            incomplete=copy.deepcopy(result);incomplete['evidence_sha256'].clear()
            self.assertFalse(MEGA.is_finalized_observation(incomplete,task['id']))


if __name__=='__main__':unittest.main(verbosity=2)
