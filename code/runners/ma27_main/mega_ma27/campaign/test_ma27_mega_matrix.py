#!/usr/bin/env python3
"""Controller and command tests; no solver or real MA27 qualification."""
from pathlib import Path
from collections import Counter,defaultdict
import concurrent.futures,copy,hashlib,importlib.util,json,os,sys,tempfile,threading,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import MEGA,executor,dataset_schedule as ds,prepare_inventory,progress

class MatrixTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=Path(__file__).resolve().parent.parent
  cls.cases,_=prepare_inventory.select_metadata(cls.root)
 def test_full_principal_and_crossed_matrices(self):
  before=json.dumps(self.cases,sort_keys=True);methods=list(MEGA.METHODS)
  cases,groups,_=MEGA.ordered_cases(self.cases,methods=methods)
  pairs=list(ds.task_pairs(cases,methods))
  self.assertEqual(len(pairs),32624);self.assertEqual(Counter(m for _,m in pairs),{m:4078 for m in methods})
  self.assertEqual([g['id']for g in groups[:3]],['CUTEst','multicast_beamforming','PGLIB'])
  self.assertEqual(pairs,[(c,m)for c in cases for m in methods])
  combined=methods+list(ds.RESEARCH_METHODS);cases,groups,_=MEGA.ordered_cases(self.cases,methods=combined)
  pairs=list(ds.task_pairs(cases,combined));self.assertEqual(len(pairs),39960)
  self.assertEqual(sum(g['total_cells']for g in groups),39960)
  self.assertEqual(len({c['id']+'__'+m for c,m in pairs}),39960)
  self.assertEqual(Counter(m for _,m in pairs),{**{m:4078 for m in methods},**{m:1834 for m in ds.RESEARCH_METHODS}})
  self.assertTrue(all(c['regime']in('I','M')for c,m in pairs if m in ds.RESEARCH_METHODS))
  self.assertEqual(pairs,list(ds.task_pairs(cases,combined)))
  self.assertEqual(before,json.dumps(self.cases,sort_keys=True))
 def test_balanced_order_and_regime_specific_denominators(self):
  methods=list(MEGA.METHODS)+list(ds.RESEARCH_METHODS)
  cases,groups,_=MEGA.ordered_cases(self.cases,methods=methods);pairs=list(ds.task_pairs(cases,methods))
  case_methods=defaultdict(list)
  for case,m in pairs:case_methods[case['id']].append(m)
  first=defaultdict(Counter)
  for case in cases:
   n=12 if case['regime']in('I','M')else 8
   self.assertEqual(len(case_methods[case['id']]),n);first[n][case_methods[case['id']][0]]+=1
  for counts in first.values():self.assertLessEqual(max(counts.values())-min(counts.values()),1)
  plan=dict(cases=cases,datasets=groups,methods=methods,tasks=[{'case':c,'method':m}for c,m in pairs],dataset_schedule_schema=ds.SCHEMA)
  self.assertTrue(ds.verify_dataset_plan(plan))
  summary=ds.dataset_summaries(plan,[])
  for group in summary.values():
   n=group['by_regime']['I']+group['by_regime']['M']
   self.assertEqual(group['by_method']['F_R3']['planned'],n)
   for regime in('E','Q'):
    if regime in group['regimes']:self.assertEqual(group['regimes'][regime]['methods']['F_R3']['planned'],0)
 def test_crossed_cells_share_actual_continuous22_queue(self):
  methods=list(MEGA.METHODS)+list(ds.RESEARCH_METHODS)
  raw=[]
  for ident,regime,family in [('CUTEst__generic0','I','CUTEst'),('CUTEst__generic1','E','CUTEst'),('PGLIB__generic0','M','PGLIB')]:
   raw.append(dict(id=ident,regime=regime,family=family,source_kind='historical',input_sha256='0'*64,
    dimensions=dict(n=2,quadratic_inequalities=regime in('I','M'),quadratic_equalities=regime in('M','E'),linear_equalities=0,linear_inequalities=0)))
  cases,groups=ds.prepare_datasets(raw,methods)
  tasks=[dict(id=c['id']+'__'+m,case=c,method=m,contract_sha256='fixture')for c,m in ds.task_pairs(cases,methods)]
  plan=dict(cases=cases,datasets=groups,methods=methods,tasks=tasks,cpus=list(range(22)),jobs=22)
  began=time.monotonic();events=[];lock=threading.Lock();active=set();peak=0;started=0;barrier=threading.Barrier(22);MEGA.executor.STOP.clear()
  slow='CUTEst__generic0__R3'
  def worker(out,plan,task,cpu,guard):
   nonlocal peak,started
   with lock:
    self.assertNotIn(cpu,active);active.add(cpu);peak=max(peak,len(active));events.append(('start',task['id'],time.monotonic()-began));started+=1;initial=started<=22
   if initial:barrier.wait(timeout=5)
   time.sleep(.2 if task['id']==slow else .002)
   rel=str(Path('tasks')/task['id']/'PREPOST_INTEGRITY.json')
   r=dict(case_id=task['case']['id'],method=task['method'],strict_qcqp=None,integrity_verified=True,
    observation_classification='indeterminate_observation',integrity_receipt=rel,evidence_sha256={rel:'fixture'})
   with MEGA.executor.RESULT_LOCK:MEGA.executor.RUN_RESULTS[str(out)][task['id']]=r
   with lock:active.remove(cpu);events.append(('end',task['id'],time.monotonic()-began))
   return r
  with tempfile.TemporaryDirectory()as d:
   result=MEGA.continuous_loop(Path(d),plan,None,worker=worker,restore=lambda *_:None,report=lambda *_:None)
  self.assertEqual(result['done'],32);self.assertEqual(result['indeterminate'],32);self.assertEqual(peak,22)
  self.assertTrue(all(result['by_method'][m]['planned']==2 for m in ds.RESEARCH_METHODS))
  self.assertEqual(result['by_regime']['E']['methods']['F_R3']['planned'],0)
  end=next(t for e,i,t in events if e=='end'and i==slow)
  self.assertTrue(any(e=='start'and i.startswith('PGLIB')and t<end for e,i,t in events))
  self.assertFalse(active)

 def test_native_six_commands_use_ma27_and_same_original_case_path(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);canonical=root/'canonical_inputs'/'generic'/'input.qcq';canonical.parent.mkdir(parents=True);canonical.write_bytes(b'exact serialized original coefficients and rawx0\n')
   wrong=root/'data/frozen/generic/input.qcq';wrong.parent.mkdir(parents=True);wrong.write_bytes(b'different raw serialization must not be substituted')
   case=dict(id='generic',input=str(canonical),input_sha256=executor.sha(canonical))
   executor.verify_case(case)
   cfg=dict(binary=str(root/'native'),native_backend='ma27',hsl_library=str(root/'licensed-HSL.so'))
   profiles=executor.read(self.root/'campaign/profiles.json')
   for method in MEGA.METHODS[:6]:
    command=executor.native_command(cfg,profiles,method,case,root/'endpoint.json',7200,5000,'off')
    self.assertEqual(command[command.index('--problem')+1],str(canonical))
    self.assertEqual(command[command.index('--linear-backend')+1],'ma27')
    self.assertEqual(command[command.index('--hsl-library')+1],str(root/'licensed-HSL.so'))
    self.assertEqual(command[command.index('--restoration-iters')+1],'600')
    self.assertEqual(command[command.index('--dca-curvature-backend')+1],'same')
 def test_observed_backend_does_not_claim_a_factor_on_early_exit(self):
  cfg={'native_backend':'ma27'}
  a=executor.native_backend_evidence({'linear_solver_backend':'native_dense_lapack','lazy_newton_factor_attempts':0},cfg)
  self.assertTrue(a['pass']);self.assertFalse(a['backend_observed']);self.assertTrue(a['early_exit_without_primary_factorization'])
  self.assertFalse(executor.native_backend_evidence({'linear_solver_backend':'majorants/mumps','lazy_newton_factor_attempts':4},cfg)['pass'])
  self.assertFalse(executor.native_backend_evidence({'linear_solver_backend':'native_dense_lapack','lazy_newton_factor_attempts':4},cfg)['pass'])
  self.assertFalse(executor.native_backend_evidence({'linear_solver_backend':'majorants/ma27','lazy_newton_factor_attempts':4,'lazy_newton_native_factor_successes':1},cfg)['pass'])
 def test_dead_controller_monitor_is_finite_and_read_only(self):
  old={'state':'running','controller_pid':os.getpid(),'controller_start_ticks':'wrong-old-identity','active_slots':6,'active_solver':6,'active_audit':0,'done':810,'total':816}
  before=copy.deepcopy(old);observed=progress.observed_controller_state(old)
  self.assertEqual(observed['state'],'controller_dead_stale_state');self.assertEqual(observed['active_slots'],0);self.assertEqual(old,before)
  rendered=progress.compact_status(dict(observed,by_regime={'I':{'N':52,'methods':{}}},by_method={'R3':{}},datasets_total=1,datasets_completed=0))
  self.assertIn('I 52',rendered);self.assertNotIn('A (',rendered)

if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(MatrixTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
 out=Path(__file__).resolve().parent.parent/'validation/MEGA_MA27_MATRIX_TESTS.json';out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(dict(schema='previous-device-ma27-full-matrix-controller-tests-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,optimizer_invocations=0,principal_tasks=32624,crossed_tasks=39960,source=__file__),indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful() else 1)
