#!/usr/bin/env python3
"""Orchestration tests using harmless local Python commands; no optimizer."""
from pathlib import Path
from unittest.mock import patch
import contextlib,io,json,os,signal,subprocess,sys,tempfile,threading,time,unittest
import PILOTER_IPOPT as p

class PilotTests(unittest.TestCase):
 def test_background_phase_has_heartbeat_without_child_output(self):
  with tempfile.TemporaryDirectory()as d:
   x=p.Pipeline(d,heartbeat=.02);output=io.StringIO()
   with contextlib.redirect_stdout(output):x.phase('MOCK SILENCIEUX',[sys.executable,'-c','import time;time.sleep(.12)'])
   self.assertIn('en cours',output.getvalue());self.assertEqual(x.state['active_child'],None);self.assertEqual(x.owned.remaining(),[])
 def test_failed_child_is_visible_and_blocks_the_following_stage(self):
  with tempfile.TemporaryDirectory()as d:
   x=p.Pipeline(d,heartbeat=.02)
   with self.assertRaisesRegex(RuntimeError,'code 7'):x.phase('MOCK ÉCHEC',[sys.executable,'-c','raise SystemExit(7)'])
   self.assertEqual(p.read(Path(d)/'IPOPT_PIPELINE_STATUS.json')['phase'],'MOCK ÉCHEC')
 def test_stop_only_terminates_owned_processes(self):
  other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],start_new_session=True)
  try:
   with tempfile.TemporaryDirectory()as d:
    event=threading.Event();x=p.Pipeline(d,event,heartbeat=.02,terminate_deadline=.2);timer=threading.Timer(.08,event.set);timer.start()
    try:
     with self.assertRaises(InterruptedError):x.phase('MOCK ATTENTE',[sys.executable,'-c','import time;time.sleep(20)'])
    finally:timer.cancel();x.cleanup()
    self.assertEqual(x.owned.remaining(),[]);self.assertIsNone(other.poll())
  finally:other.terminate();other.wait(timeout=3)
 def test_pid_start_time_prevents_reused_pid_signalling(self):
  self.assertTrue(p.live(os.getpid(),p.ticks(os.getpid())))
  self.assertFalse(p.live(os.getpid(),p.ticks(os.getpid())+1));self.assertFalse(p.live(None,None))
 def test_nonblocking_pipeline_lock_is_dedicated_and_exclusive(self):
  with tempfile.TemporaryDirectory()as d:
   lock=p.pipeline_lock(Path(d)/'ipopt.lock')
   try:
    with self.assertRaisesRegex(ValueError,'déjà'):p.pipeline_lock(Path(d)/'ipopt.lock')
    unrelated=p.pipeline_lock(Path(d)/'refinement.lock');unrelated.close()
   finally:lock.close()
   p.pipeline_lock(Path(d)/'ipopt.lock').close()
 def test_resume_requires_all_four_qualified_immutable_receipts(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);paths=[root/x for x in('build/BUILD.json','qualification/QUALIFICATION.json','preparation/PREPARATION.json','preparation/INSTALLATION_IPOPT_ALIGNE.json')]
   for a in paths:a.parent.mkdir(parents=True,exist_ok=True);a.write_text(json.dumps({'passed':True}))
   p.atomic(root/'IPOPT_PIPELINE_STATUS.json',dict(qualified_bindings={str(a):p.sha(a)for a in paths}))
   self.assertEqual(p.bound_installation(root),paths[-1]);paths[0].write_text('{}')
   with self.assertRaisesRegex(ValueError,'modifiée'):p.bound_installation(root)
 def test_manifest_checks_immutable_files_and_ignores_mutable_launcher(self):
  with tempfile.TemporaryDirectory()as d,patch.object(p,'KIT',Path(d)):
   root=Path(d);(root/'code.py').write_text('x=1\n');(root/'LAST_IPOPT_RUN.json').write_text('{}')
   p.atomic(root/'KIT_SHA256.json',{'code.py':p.sha(root/'code.py'),'LAST_IPOPT_RUN.json':'old mutable value'})
   self.assertTrue(p.verify_kit());(root/'code.py').write_text('x=2\n')
   with self.assertRaisesRegex(ValueError,'modifié'):p.verify_kit()
 def test_manifest_cannot_escape_its_kit(self):
  with tempfile.TemporaryDirectory()as d,patch.object(p,'KIT',Path(d)):
   p.atomic(Path(d)/'KIT_SHA256.json',{'../elsewhere':'a'*64})
   with self.assertRaisesRegex(ValueError,'invalide'):p.verify_kit()
 def test_analysis_uses_new_four_configuration_observations(self):
  with tempfile.TemporaryDirectory()as d:
   x=p.Pipeline(d);calls=[]
   x.phase=lambda name,command:calls.append((name,command))
   with patch.object(p,'export',return_value=Path(d)/'export.zip'):p.analyze(x)
   command=calls[0][1];self.assertIn(str(Path(d)/'campaign/OBSERVATIONS.jsonl'),command);self.assertIn(str(Path(d)/'campaign/PLAN.json'),command)
   self.assertIn('ipopt_aligned/ANALYSER_IPOPT_ALIGNE.py',' '.join(command))
 def test_wrapper_paths_are_quoted_and_shell_syntax_is_valid(self):
  for name in('LANCER','SUIVRE','ARRETER','REPRENDRE','DEPOUILLER'):
   path=p.KIT/(name+'_IPOPT_ALIGNE.sh');self.assertEqual(subprocess.run(['bash','-n',str(path)]).returncode,0);self.assertIn('"$@"',path.read_text())

if __name__=='__main__':unittest.main(verbosity=2)
