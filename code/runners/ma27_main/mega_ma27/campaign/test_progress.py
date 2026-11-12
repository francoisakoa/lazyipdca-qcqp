#!/usr/bin/env python3
"""Read-only monitoring emits immediately and displays all scientific groupings."""
import concurrent.futures
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import MEGA
import progress
from dataset_schedule import prepare_datasets


class ProgressTests(unittest.TestCase):
    def test_first_message_precedes_blocked_status_read(self):
        entered=threading.Event();release=threading.Event();messages=[]
        def slow_read(out):
            entered.set()
            if not release.wait(5):raise TimeoutError('Test reader not released')
            return dict(state='completed',done=0,total=0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(progress.follow_status,Path('/unused'),once=True,
                emit=messages.append,loader=slow_read)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(messages,['SUIVI ACTIF — /unused'])
                self.assertFalse(future.done())
            finally:release.set()
            future.result(timeout=5)
        self.assertIn('terminée',messages[-1])

    def test_missing_status_and_json_mode_are_visible(self):
        messages=[]
        progress.follow_status(Path('/unused'),once=True,emit=messages.append,loader=lambda out:None)
        self.assertIn('EN ATTENTE DU PREMIER STATUT',messages[1])
        messages=[]
        progress.follow_status(Path('/unused'),once=True,json_output=True,
            emit=messages.append,loader=lambda out:dict(state='completed',done=2,total=2))
        self.assertEqual(json.loads(messages[0])['event'],'monitor_started')
        self.assertEqual(json.loads(messages[1])['done'],2)

    def test_status_read_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'STATUS.json').write_bytes(b' '*33)
            messages=[]
            with patch.object(progress,'MAX_STATUS_BYTES',32):
                progress.follow_status(out,once=True,emit=messages.append)
            self.assertIn('taille de lecture',messages[-1])

    def test_global_dataset_and_regime_tables_preserve_unknowns_and_pending(self):
        cases=[]
        for dataset,regime in [('CUTEst','I'),('PGLIB','Q')]:
            cases.append(dict(id=dataset+'__sample',family=dataset,source_kind='historical',
                regime=regime,input_sha256='0'*64,dimensions=dict(n=1,quadratic_inequalities=1,
                quadratic_equalities=0,linear_equalities=0,linear_inequalities=0)))
        cases,datasets=prepare_datasets(cases,MEGA.METHODS)
        plan=dict(cases=cases,datasets=datasets,methods=list(MEGA.METHODS),jobs=22,
            cpus=list(range(22)),tasks=[dict(id=c['id']+'__'+m,case=c,method=m)
                for c in cases for m in MEGA.METHODS])
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);records={}
            for method,strict in [('R3',True),('HH',None)]:
                task_id=cases[0]['id']+'__'+method
                receipt=str(Path('tasks')/task_id/'PREPOST_INTEGRITY.json')
                records[task_id]=dict(case_id=cases[0]['id'],method=method,strict_qcqp=strict,
                    integrity_verified=True,integrity_receipt=receipt,evidence_sha256={receipt:'fake'},
                    observation_classification='strict_original_qcqp' if strict else 'indeterminate_observation')
            with MEGA.executor.RESULT_LOCK:MEGA.executor.RUN_RESULTS[str(out)]=records
            value=MEGA.summary(out,plan,'running',22)
            value['running_datasets']=['CUTEst'];value['active_solver']=20;value['active_audit']=2
            text=progress.format_status(value)
        self.assertIn('terminées 2/16 | KKT strict 1 | non strict 0 | indéterminées 1',text)
        self.assertIn('Créneaux occupés 22/22 | résolutions 20 | audits 2',text)
        self.assertIn('GLOBAL — toutes les instances',text)
        self.assertIn('CUTEst',text);self.assertIn('PGLIB',text)
        self.assertIn('Q : 0/1',text)
        self.assertIn('I (1)',text);self.assertIn('A (0)',text)
        self.assertGreaterEqual(text.count('0/0/1 ; 1/1'),2)
        self.assertIn('indéterminées 1 | en attente 14',text)
        self.assertIn('en attente = prévues − terminées',text)
        self.assertGreaterEqual(text.count('En attente'),2)
        global_r3=next(line for line in text.splitlines() if line.startswith('R3 '))
        self.assertEqual(global_r3.split()[-1],'1')
        for method in progress.METHODS:
            self.assertIn(progress.LABELS.get(method,method),text)
            self.assertIn(progress.SHORT_LABELS.get(method,method),text)

    def test_root_entry_point_displays_monitor_without_installation_or_audits(self):
        root=Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,str(root/'MEGA.py'),'suivre','--once',
                '--out',directory],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue(result.stdout.startswith('SUIVI ACTIF — '))
        self.assertIn('EN ATTENTE DU PREMIER STATUT',result.stdout)

    def test_controller_announces_before_long_validation(self):
        # Bypass only signal registration (this test runs in a thread).
        entered=threading.Event();release=threading.Event();output=io.StringIO()
        def long_validation(*args,**kwargs):
            entered.set()
            if not release.wait(5):raise TimeoutError('Test validation not released')
            return object()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);MEGA.write(out/'PLAN.json',dict(tasks=[],jobs=22))
            with patch.object(MEGA.signal,'signal'),patch.object(MEGA,'validate_plan',long_validation), \
                    patch.object(MEGA,'continuous_loop',return_value=dict(state='completed',done=0,total=0)), \
                    contextlib.redirect_stdout(output), \
                    concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(MEGA.run_controller,out,out,'test-token')
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertIn('CONTRÔLE DE L’INSTALLATION ET DES DONNÉES',output.getvalue())
                    self.assertEqual(progress.read_status(out)['state'],'initializing')
                finally:release.set()
                self.assertEqual(future.result(timeout=5),0)


if __name__=='__main__':unittest.main(verbosity=2)
