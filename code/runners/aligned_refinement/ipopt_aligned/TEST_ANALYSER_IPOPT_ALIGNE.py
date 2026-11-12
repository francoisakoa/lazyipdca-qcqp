#!/usr/bin/env python3
"""Synthetic analysis checks; no optimizer invocation."""
import copy,gzip,importlib.util,json,tempfile,unittest
from pathlib import Path
P=Path(__file__).with_name('ANALYSER_IPOPT_ALIGNE.py')
spec=importlib.util.spec_from_file_location('analysis',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def cases():
 return {k:dict(case_id=k,regime=r,dataset=d,family=d,input_sha256='input-'+k,start_sha256='start-'+k)for k,r,d in [('a','I','first'),('b','M','beam_structural'),('c','M','second')]}
def row(cid,cfg,strict=True,objective=1.,process=2.,native=1.5):
 c=cases()[cid];cell=m.CONFIG[cfg];opts=dict(m.COMMON,mu_strategy=cell['strategy'],linear_solver=cell['backend']);res=1e-8 if strict else 2e-5
 proof=dict(aligned_options_verified=True,native_route_verified=True,backend_verified=True,budget_verified=True,option_file_disabled=True,backend_banner_verified=True,endpoint_sha256='endpoint',solver_log_sha256='log',mu_strategy=cell['strategy'],linear_solver=cell['backend'],effective_options_sha256=m.json_sha(opts),native_iterations=10,native_elapsed_seconds=native)
 return dict(c,method=cfg,run_id='fresh',binary_sha256='binary',classification='strict'if strict else'nonstrict',strict_qcqp=strict,primal=res,dual=1e-8,complementarity=1e-8,objective=objective,budget_verified=True,provenance=proof,effective_options=opts,original_audit=dict(strict_qcqp=strict,primal=res,dual=1e-8,complementarity=1e-8),solver_resources=dict(elapsed_seconds=process,user_cpu_seconds=1.,system_cpu_seconds=.1,returncode=0,censored=False),seconds_process=process,seconds_process_scope='solver process elapsed; independent audit excluded',ipopt_application_status_code=0,ipopt_application_status='Solve_Succeeded')
def full():return [row(cid,cfg)for cid in cases()for cfg in m.CELLS]
def analyse(rows=None,**kw):return m.analyse(cases(),full()if rows is None else rows,dict(run_id='fresh',binary_sha256='binary'),**kw)

class AnalysisTests(unittest.TestCase):
 def test_plan_actual_shape(self):
  p=dict(cases=3,case_bindings=list(cases().values()),methods=list(m.CELLS));self.assertEqual(len(m.plan_cases(p,False)),3)
 def test_full_counts(self):
  report,cov,pairs,*_=analyse();self.assertTrue(report['complete_matrix']);self.assertEqual(report['planned_Ipopt_tasks'],12);self.assertEqual(len([p for p in pairs if p['scope_type']=='all']),6)
 def test_all_four_ids(self):
  self.assertEqual(len(m.CELLS),4);self.assertEqual({m.identify(row('a',c))for c in m.CELLS},set(m.CELLS))
 def test_old_ipopt_never_relabelled(self):
  self.assertIsNone(m.identify(dict(method='ipopt_ma27_default',backend='ma27',strategy='monotone')))
 def test_bad_option_unqualified(self):
  r=row('a',m.CELLS[0]);r['effective_options']['bound_relax_factor']=1e-8;r['provenance']['effective_options_sha256']=m.json_sha(r['effective_options']);v=m.validate_row(r,m.CELLS[0],cases()['a'],'fresh','binary');self.assertIsNone(v['strict']);self.assertIn('effective_option:bound_relax_factor',v['issues'])
 def test_bad_original_decision_unqualified(self):
  r=row('a',m.CELLS[0]);r['primal']=.1;self.assertIsNone(m.validate_row(r,m.CELLS[0],cases()['a'])['strict'])
 def test_proven_nonstrict(self):
  v=m.validate_row(row('a',m.CELLS[0],False),m.CELLS[0],cases()['a']);self.assertIs(v['strict'],False);self.assertEqual(v['failure_category'],'above_original_tolerance:primal')
 def test_internal_success_kept_distinct(self):
  rows=full();rows[0]=row('a',m.CELLS[0],False);rows[4]['provenance']['backend_verified']=False
  _,cov,*_=analyse(rows);c=next(x for x in cov if x['scope_type']=='all'and x['configuration']==m.CELLS[0]);self.assertEqual(c['positive_return_audited_nonstrict'],1);self.assertEqual(c['positive_return_unqualified'],1);self.assertEqual(c['success'],1)
 def test_missing_not_numerical(self):
  with self.assertRaises(ValueError):analyse(full()[:-1])
  report,*_=analyse(full()[:-1],require_full=False);self.assertTrue(report['provisional']);self.assertEqual(report['missing_Ipopt_tasks'],1)
 def test_duplicate_rejected(self):
  with self.assertRaises(ValueError):analyse(full()+[full()[0]])
 def test_multiple_run_ids_rejected(self):
  r=full();r[0]['run_id']='other'
  with self.assertRaises(ValueError):analyse(r)
 def test_different_binary_rejected(self):
  r=full();r[0]['binary_sha256']='other'
  with self.assertRaises(ValueError):analyse(r)
 def test_unregistered_extra_setting_rejected(self):
  r=full();r[0]['effective_options']['new_option']='changed';r[0]['provenance']['effective_options_sha256']=m.json_sha(r[0]['effective_options'])
  with self.assertRaises(ValueError):analyse(r)
 def test_bound_budget_is_not_iteration_inference(self):
  r=row('a',m.CELLS[0]);r['solver_resources']['user_cpu_seconds']=7200;r['solver_resources']['system_cpu_seconds']=1;self.assertIsNone(m.validate_row(r,m.CELLS[0],cases()['a'])['strict'])
 def test_quantile_and_gap(self):
  self.assertEqual(m.quantile([0,2,4,6],.25),1.5)
  left,right=m.CELLS[:2];idx={}
  for cid in cases():
   for cfg in (left,right):idx[cid,cfg]=m.validate_row(row(cid,cfg,objective=2. if cfg==left else 1.),cfg,cases()[cid])
  pair,_=m.pair_summary(list(cases().values()),idx,left,right);self.assertEqual(pair['signed_gap_median'],.5);self.assertEqual(pair['higher_tol_1em3'],3)
 def test_timing_strict_greater_than_one(self):
  left,right=m.CELLS[:2];idx={}
  for cid in cases():
   for cfg in (left,right):idx[cid,cfg]=m.validate_row(row(cid,cfg,process=1. if cfg==left else 2.,native=.5),cfg,cases()[cid])
  _,times=m.pair_summary(list(cases().values()),idx,left,right);gt=next(t for t in times if t['clock']=='seconds_process'and t['cohort']=='both_gt1s');self.assertEqual(gt['N'],0)
  native=next(t for t in times if t['clock']=='seconds_numerical'and t['cohort']=='all_common_strict');self.assertEqual(native['median'],1.)
 def test_partial_line_only_last(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'x.jsonl';p.write_text('{"x":1}\n{"x":');self.assertEqual(list(m.stream(p,True)),[{'x':1}])
   with self.assertRaises(ValueError):list(m.stream(p,False))
   p.write_text('{bad}\n{"x":1}\n')
   with self.assertRaises(ValueError):list(m.stream(p,True))
 def test_native_projection_excludes_old_ipopt(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'x.jsonl.gz';rr=[]
   for c in cases().values():
    for config in m.NATIVE:rr.append(dict(c,config=config,classification='strict',primal=0.,dual=0.,complementarity=0.,objective=1.,seconds_process=2.))
   rr.append(dict(case_id='bad-old-id',config='ipopt_ma27_default'))
   with gzip.open(p,'wt')as f:f.write(''.join(json.dumps(r)+'\n'for r in rr))
   idx,_=m.native_reference(p,cases());self.assertEqual(len(idx),36)
   rr[0]['input_sha256']='bad'
   with gzip.open(p,'wt')as f:f.write(''.join(json.dumps(r)+'\n'for r in rr))
   with self.assertRaises(ValueError):m.native_reference(p,cases())
 def test_own_success_codes_not_python_bool(self):
  self.assertIsNone(m.own_success({'ipopt_application_status_code':True}));self.assertTrue(m.own_success({'ipopt_application_status_code':1}));self.assertFalse(m.own_success({'ipopt_application_status_code':2}))

if __name__=='__main__':unittest.main(verbosity=2)
