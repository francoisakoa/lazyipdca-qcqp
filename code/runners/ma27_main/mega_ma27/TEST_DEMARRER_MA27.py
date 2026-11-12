#!/usr/bin/env python3
"""Native-mega launch routing and failure tests with Python-only fixtures."""
from pathlib import Path
import argparse,contextlib,fcntl,hashlib,importlib.util,io,json,os,shutil,subprocess,sys,tempfile,time,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('test_ma27_launcher',HERE/'DEMARRER.py');launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
CAUSE='PRÉPARATION MÉGA REFUSÉE — entrée gelée différente ; attendu abc, observé def'
LOG='PREPARATION_20261108T120000_000000Z.log'
FAKE_PREPARER=r'''from pathlib import Path
import argparse,json,os,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--out',type=Path);p.add_argument('--with-factorial',action='store_true');a=p.parse_args()
if os.environ.get('FAKE_PREP_FAILURE'):
 print(os.environ['FAKE_PREP_FAILURE'],flush=True);raise SystemExit(2)
a.out.mkdir();device=a.out/'device';(device/'campaign').mkdir(parents=True)
source=Path(__file__).parent/'FAKE_MEGA.py';(device/'campaign/MEGA.py').write_bytes(source.read_bytes())
ready=dict(passed=True,root=str(device.resolve()),python=sys.executable,tasks=39960 if a.with_factorial else32624,native_methods=['R3','HH','HS','SH','SS','SS_legacy'],native_backend='ma27',ipopt_backends=['ma27','mumps'],ipopt_protocol='library-defaults',source_root=str(a.root.resolve()),source_unchanged=True,with_factorial=a.with_factorial)
if os.environ.get('FAKE_BAD_READY')=='tasks':ready['tasks']+=1
if os.environ.get('FAKE_BAD_READY')=='root':ready['root']=str(a.root)
(a.out/'READY.json').write_text(json.dumps(ready))
(a.out/'PREPARER_CALL.json').write_text(json.dumps(sys.argv[1:]))
print('READY : '+str(device),flush=True)
'''.replace('else32624','else 32624')
FAKE_MEGA=r'''from pathlib import Path
import argparse,json,os,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path);s=p.add_subparsers(dest='action',required=True)
a=s.add_parser('lancer');a.add_argument('--cpus');a.add_argument('--jobs');a.add_argument('--out',type=Path)
a=s.add_parser('suivre');a.add_argument('--out',type=Path);a.add_argument('--interval');a.add_argument('--once',action='store_true');a.add_argument('--json',action='store_true')
a=p.parse_args()
with(a.root/'CALLS.jsonl').open('a')as f:f.write(json.dumps(sys.argv[1:])+'\n')
if a.action=='lancer':
 if os.environ.get('FAKE_LAUNCH_FAILURE'):print('LANCEMENT REFUSÉ — CPU réservé',flush=True);raise SystemExit(2)
 a.out.mkdir();(a.out/'PLAN.json').write_text(json.dumps({'test_stub_only':True}));(a.out/'STATUS.json').write_text(json.dumps({'state':'completed','test_stub_only':True}))
 (a.root/'LAST_RUN.txt').write_text(str(a.out))
 print('FILE UNIQUE 22 — stub without optimizers',flush=True)
else:print('REGIMES I M E Q — previous device suivre stub',flush=True)
'''

class LaunchTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='mega MA27 épreuve ');self.root=Path(self.tmp.name);self.kit=self.root/'kit';self.kit.mkdir()
  for name in('DEMARRER.py','SUIVRE.py'):shutil.copy2(HERE/name,self.kit/name)
  (self.kit/'PREPARER_MEGA.py').write_text(FAKE_PREPARER);(self.kit/'FAKE_MEGA.py').write_text(FAKE_MEGA)
  self.old=self.root/'old';self.old.mkdir();(self.old/'INSTALLATION.json').write_text('{"protected":true}');(self.old/'PREPARATION.json').write_text('{"frozen":true}')
  self.prep=self.root/'new_preparation';self.run=self.root/'new_results';self.log=self.kit/LOG
  self.before={p.name:p.read_bytes()for p in self.old.iterdir()}
 def tearDown(self):self.tmp.cleanup()
 def worker(self,extra=(),env=None):
  command=[sys.executable,str(self.kit/'DEMARRER.py'),'--worker','--root',str(self.old),'--out',str(self.prep),'--results',str(self.run),'--log',str(self.log),'--launch-token','test-only-token',*extra]
  e=dict(os.environ);e.update(env or{})
  with self.log.open('w')as f:done=subprocess.run(command,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=10)
  return done.returncode,json.loads((self.kit/'LAUNCH_STATE.json').read_text())
 def calls(self):
  p=self.prep/'device/CALLS.jsonl'
  return [json.loads(x)for x in p.read_text().splitlines()]if p.exists()else[]
 def test_principal_routes_all_six_native_and_two_defaults_to_one_previous_controller(self):
  code,state=self.worker();self.assertEqual(code,0,self.log.read_text());self.assertEqual(state['tasks'],32624);self.assertFalse(state['with_factorial'])
  self.assertEqual(state['state'],'launched');self.assertEqual(len(self.calls()),1)
  argv=self.calls()[0];self.assertEqual(argv,['--root',str(self.prep/'device'),'lancer','--cpus','0-21','--jobs','22','--out',str(self.run)])
  self.assertEqual({p.name:p.read_bytes()for p in self.old.iterdir()},self.before)
  self.assertEqual(Path((self.kit/'LAST_RUN.txt').read_text().strip()),self.run)
  self.assertEqual(state['controller_script'],str(self.prep/'device/campaign/MEGA.py'))
 def test_optional_factorial_adds_7336_in_same_controller(self):
  code,state=self.worker(['--with-factorial']);self.assertEqual(code,0,self.log.read_text());self.assertEqual(state['tasks'],39960)
  args=json.loads((self.prep/'PREPARER_CALL.json').read_text());self.assertIn('--with-factorial',args)
  self.assertEqual(len(self.calls()),1);self.assertTrue(state['with_factorial'])
 def test_preparation_root_cause_visible_and_no_campaign_on_failure(self):
  code,state=self.worker(env={'FAKE_PREP_FAILURE':CAUSE});self.assertEqual(code,2)
  self.assertEqual(state['child_returncode'],2);self.assertIn(CAUSE,state['diagnostic']);self.assertFalse(state['benchmark_launched'])
  self.assertFalse(self.run.exists());self.assertEqual(self.calls(),[])
  monitored=subprocess.run([sys.executable,str(self.kit/'SUIVRE.py'),'--once'],capture_output=True,text=True)
  self.assertEqual(monitored.returncode,2);self.assertIn(CAUSE,monitored.stdout);self.assertIn(str(self.log),monitored.stdout)
 def test_wrong_ready_task_scope_blocks_controller(self):
  code,state=self.worker(env={'FAKE_BAD_READY':'tasks'});self.assertEqual(code,2);self.assertIn('32624',state['error']);self.assertEqual(self.calls(),[])
 def test_ready_cannot_redirect_into_previous_installation(self):
  code,state=self.worker(env={'FAKE_BAD_READY':'root'});self.assertEqual(code,2);self.assertIn('nouveau dispositif isolé',state['error']);self.assertFalse(self.run.exists())
  self.assertEqual({p.name:p.read_bytes()for p in self.old.iterdir()},self.before)
 def test_launch_failure_is_not_reported_as_success(self):
  code,state=self.worker(env={'FAKE_LAUNCH_FAILURE':'1'});self.assertEqual(code,2);self.assertEqual(state['state'],'blocked');self.assertIn('CPU réservé',state['diagnostic'])
  self.assertIsNone(state['benchmark_launched']);self.assertFalse((self.kit/'LAST_RUN.txt').exists())
 def test_monitor_uses_original_mega_regime_follow_cli(self):
  code,state=self.worker();self.assertEqual(code,0,self.log.read_text())
  done=subprocess.run([sys.executable,str(self.kit/'SUIVRE.py'),'--once','--interval','10'],capture_output=True,text=True)
  self.assertEqual(done.returncode,0,done.stderr);self.assertIn('REGIMES I M E Q',done.stdout)
  calls=self.calls();self.assertEqual(len(calls),2)
  self.assertEqual(calls[1],['--root',str(self.prep/'device'),'suivre','--out',str(self.run),'--interval','10.0','--once'])
 def test_stale_preparation_pid_does_not_wait_or_mutate_state(self):
  state=dict(state='preparing',phase='qualifying',preparation_log=str(self.log),worker_pid=999999999,worker_start_ticks='missing')
  p=self.kit/'LAUNCH_STATE.json';p.write_text(json.dumps(state));self.log.write_text('last progress\n');before=p.read_bytes();start=time.monotonic()
  done=subprocess.run([sys.executable,str(self.kit/'SUIVRE.py')],capture_output=True,text=True,timeout=2)
  self.assertEqual(done.returncode,2);self.assertIn('contrôleur identifié absent',done.stdout);self.assertEqual(before,p.read_bytes());self.assertLess(time.monotonic()-start,1)
 def test_missing_first_status_and_dead_controller_never_waits_indefinitely(self):
  code,state=self.worker();self.assertEqual(code,0,self.log.read_text())
  (self.run/'STATUS.json').unlink()
  start=time.monotonic();done=subprocess.run([sys.executable,str(self.kit/'SUIVRE.py')],capture_output=True,text=True,timeout=2)
  self.assertEqual(done.returncode,2);self.assertIn('CONTRÔLEUR NON CONFIRMÉ',done.stdout);self.assertLess(time.monotonic()-start,1)
 def test_outer_launcher_publishes_fresh_start_before_background_spawn(self):
  statepath=self.kit/'LAUNCH_STATE.json';statepath.write_text(json.dumps(dict(state='blocked',error='OLD')))
  seen=[]
  class P:pid=os.getpid()
  def spawn(command,**_):
   seen.append(json.loads(statepath.read_text()));self.assertEqual(seen[-1]['state'],'starting');self.assertNotIn('OLD',str(seen[-1]));return P()
  with patch.object(launch,'HERE',self.kit),patch.object(launch,'STATE',statepath),patch.object(sys,'argv',['DEMARRER.py','--root',str(self.old),'--out',str(self.prep),'--results',str(self.run)]),patch.object(launch.subprocess,'Popen',side_effect=spawn),contextlib.redirect_stdout(io.StringIO()):
   self.assertEqual(launch.main(),0)
  self.assertEqual(len(seen),1);self.assertEqual(json.loads(statepath.read_text())['worker_start_ticks'],launch.identity(os.getpid()))
 def test_live_campaign_prevents_second_controller(self):
  self.run.mkdir();(self.run/'STATUS.json').write_text(json.dumps(dict(state='running',controller_pid=os.getpid(),controller_start_ticks=launch.identity(os.getpid()))))
  statepath=self.kit/'LAUNCH_STATE.json';statepath.write_text(json.dumps(dict(state='launched',run=str(self.run))))
  with patch.object(launch,'HERE',self.kit),patch.object(launch,'STATE',statepath),patch.object(sys,'argv',['DEMARRER.py','--root',str(self.old)]),patch.object(launch.subprocess,'Popen',side_effect=AssertionError('second launch')):
   with self.assertRaisesRegex(ValueError,'déjà active'):launch.main()
 def test_existing_preparation_lock_prevents_second_preparer(self):
  with(self.kit/'preparation.lock').open('a')as f:
   fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   command=[sys.executable,str(self.kit/'DEMARRER.py'),'--worker','--root',str(self.old),'--out',str(self.prep),'--results',str(self.run),'--log',str(self.log)]
   done=subprocess.run(command,capture_output=True,text=True,timeout=2)
   self.assertEqual(done.returncode,2);self.assertIn('déjà active',done.stderr);self.assertFalse(self.prep.exists());self.assertFalse(self.run.exists())

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(LaunchTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if result.wasSuccessful():
  receipt=dict(schema='previous-mega-ma27-launcher-tests-v1',passed=True,tests=result.testsRun,optimizer_invocations=0,remote_actions=0,
   principal_tasks=32624,optional_with_factorial_tasks=39960,single_controller=True,
   source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()for name in('DEMARRER.py','SUIVRE.py')})
  (HERE/'LAUNCHER_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful()else 1)
