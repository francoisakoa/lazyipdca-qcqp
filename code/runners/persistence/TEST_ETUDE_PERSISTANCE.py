#!/usr/bin/env python3
"""Failure/recovery tests of the study envelope; no optimizer is started."""
from pathlib import Path
from types import SimpleNamespace
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest

SOURCE=Path(__file__).with_name('ETUDE_PERSISTANCE.py')


class StudyEnvelopeTests(unittest.TestCase):
    def setup_study(self,base,complete=True):
        spec=importlib.util.spec_from_file_location('study_under_test',SOURCE)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        kit=base/'kit';kit.mkdir();(kit/'COHORTE_PERSISTANCE.json').write_text('{}')
        (kit/'controller.py').write_text('stable source')
        mod.HERE=kit;calls=[]
        def invoke(script,*args):
            calls.append((script,tuple(map(str,args))));arguments=list(map(str,args));out=Path(arguments[arguments.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
            if script=='BUILD_PERSISTANCE.py':mod.write(out/'BUILD_PERSISTANCE.json',{'synthetic':True})
            elif arguments[0]=='preparer':mod.write(out/'INSTALLATION.json',{'synthetic':True})
            elif arguments[0]=='synthese':mod.write(out/'SYNTHESE_PERSISTANCE.json',dict(complete=complete,qualification_passed=True,planned=144,timed_repetitions_completed=144 if complete else 12))
        mod.invoke=invoke
        return mod,SimpleNamespace(out=base/'results',build=None,ipopt_run=base/'historical'),calls

    def test_partial_build_preserved_and_stable_resume(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(tmp);mod,args,calls=self.setup_study(base)
            (args.out/'build').mkdir(parents=True);(args.out/'build/interrupted.txt').write_text('preserve')
            mod.run(args)
            self.assertEqual((args.out/'build/interrupted.txt').read_text(),'preserve')
            self.assertTrue((args.out/'build_retry_002/BUILD_PERSISTANCE.json').is_file())
            before=len([c for c in calls if c[0]=='BUILD_PERSISTANCE.py'])
            preparations=len([c for c in calls if c[0]=='PERSISTANCE.py' and c[1][0]=='preparer'])
            mod.run(args)
            self.assertEqual(len([c for c in calls if c[0]=='BUILD_PERSISTANCE.py']),before)
            self.assertEqual(len([c for c in calls if c[0]=='PERSISTANCE.py' and c[1][0]=='preparer']),preparations)
            self.assertEqual(mod.read(args.out/'ETUDE_STATUS.json')['stage'],'completed')

    def test_changed_source_refused_before_next_action(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(tmp);mod,args,calls=self.setup_study(base);mod.run(args)
            (mod.HERE/'controller.py').write_text('changed source');count=len(calls)
            with self.assertRaises(ValueError):mod.run(args)
            self.assertEqual(len(calls),count)

    def test_incomplete_summary_cannot_complete_study(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            base=Path(tmp);mod,args,calls=self.setup_study(base,False)
            with self.assertRaises(ValueError):mod.run(args)
            self.assertEqual(mod.read(args.out/'ETUDE_STATUS.json')['stage'],'blocked')


if __name__=='__main__':unittest.main()
