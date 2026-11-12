#!/usr/bin/env python3
from pathlib import Path
import copy,gzip,hashlib,importlib.util,json,tempfile,unittest,sys
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('export_mega_ma27',HERE/'DEPOUILLER.py');e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
DEFAULT=HERE.parent/'default_relaunch'
if not DEFAULT.is_dir():DEFAULT=HERE.parent/'article_revision_R4/default_relaunch'
FACTOR=HERE.parent/'factorial'
if not FACTOR.is_dir():FACTOR=HERE.parent/'article_revision_R4/factorial'
NATIVE=('R3','HH','HS','SH','SS','SS_legacy')
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True)+'\n')
def case(ident,regime):
 start=hashlib.sha256(('start-'+ident).encode()).hexdigest();inp=hashlib.sha256(('input-'+ident).encode()).hexdigest()
 return dict(case_id=ident,dataset='collection',family='collection',regime=regime,input_sha256=inp,start_binding=dict(start_sha256=start,raw_x0_line_sha256=start,external_start_required=False))
class Fixture:
 def __init__(self,folder,factorial=False,full=False):
  self.root=Path(folder);self.run=self.root/'run';self.run.mkdir();self.install=self.root/'installation';self.install.mkdir()
  if full:
   self.corpus=json.load(gzip.open(DEFAULT/'CONTRAT_4078.json.gz','rt'))['cases']
  else:self.corpus=[case('case_one','I'),case('case_two','M'),case('case_three','E')]
  contract=self.install/'contract.json.gz';contract.write_bytes(gzip.compress(json.dumps(dict(cases=self.corpus)).encode()))
  self.default=dict(contract=str(contract),contract_sha256=e.sha(contract),default_binary_sha256='a'*64,hsl_library_sha256='h'*64)
  defaultpath=self.install/'default.json';write(defaultpath,self.default)
  installation=dict(native_backend='ma27',backend='ma27',binary_sha256='b'*64,ipopt_binary_sha256='a'*64,hsl_library_sha256='h'*64,
   ipopt_defaults_installation=str(defaultpath),ipopt_defaults_installation_sha256=e.sha(defaultpath))
  native=self.install/'native_qualification.json'
  write(native,dict(schema='native-ma27-six-profiles-qualification-v1',passed=True,native_backend='ma27',binary_sha256='b'*64,hsl_library_sha256='h'*64,six_profiles_qualified=True,mumps_compiled=False,profiles=list(NATIVE),observations=[dict(regime=g,method=m,passed=True,ma27_primary_successful_factors=1,original_audit=dict(strict=True,feasibility=1e-8,dual=1e-8,complementarity=1e-8))for g in e.REGIMES for m in NATIVE]))
  installation.update(native_backend_qualification=str(native),native_backend_qualification_sha256=e.sha(native))
  if factorial:
   reg=FACTOR/'PRE_ENREGISTREMENT_MA27_20261108.json'
   if not reg.is_file():reg=FACTOR/'PRE_ENREGISTREMENT.json'
   pre=json.loads(reg.read_text());pre['inputs']['contract_file_sha256']=e.sha(contract)
   prereg=self.install/'registration.json';write(prereg,pre)
   gate=self.install/'GATE.json';write(gate,dict(passed=True,production_authorized=True,coupling_separation_verified=True,native_backend='ma27',factor_binary_sha256='f'*64,r3_diagonal_cases=50,hs_diagonal_cases=50,registration_sha256=e.sha(prereg)))
   factpath=self.install/'factorial.json';self.fact=dict(binary_sha256='f'*64,native_backend='ma27',preregistration=str(prereg),preregistration_sha256=e.sha(prereg),qualified_50gate=str(gate),qualified_50gate_sha256=e.sha(gate))
   write(factpath,self.fact);installation.update(factorial_installation=str(factpath),factorial_installation_sha256=e.sha(factpath))
  cases=[dict(id=c['case_id'],dataset_id=c['dataset'],regime=c['regime'],family=c['family'],input_sha256=c['input_sha256'],start_sha256=c['start_binding']['start_sha256'])for c in self.corpus]
  methods=list(e.BASE)+(list(e.RESEARCH)if factorial else[])
  p=dict(schema='mega-fresh-continuous-no-dey-v1',root=str(self.install),output_directory=str(self.run),run_id='MEGA_MA27_test',installation=installation,cases=cases,profiles={},runtime_sha256={},source_sha256={},seconds=7200,max_outer=5000,shared_restoration=600,tolerance=1e-6,threads=1,al_mode='off',methods=methods,datasets=[])
  p['execution_binding_sha256']=e.value_sha({**{k:p[k]for k in('installation','runtime_sha256','source_sha256','profiles','seconds','max_outer','shared_restoration','tolerance','threads','al_mode','methods','datasets')},'run_id':p['run_id']})
  p['tasks']=[dict(id=c['id']+'__'+m,case=c,method=m,contract_sha256=e.value_sha(dict(case=c,method=m,execution_binding_sha256=p['execution_binding_sha256'])))for c in cases for m in methods if m not in e.RESEARCH or c['regime']in('I','M')]
  self.plan=p;write(self.run/'PLAN.json',p);write(self.run/'PLAN_BINDING.json',dict(plan_sha256=e.sha(self.run/'PLAN.json')))
 def task(self,ident,method):return next(t for t in self.plan['tasks']if t['case']['id']==ident and t['method']==method)
 def result(self,ident,method,strict=True,code=None,final=True,seconds=2.0):
  task=self.task(ident,method);folder=self.run/'tasks'/task['id'];folder.mkdir(parents=True,exist_ok=True)
  known=strict is not None;metric=1e-8 if strict is True else 2e-6 if strict is False else None
  ipopt=method in e.IPOPT;origin='ipopt'if ipopt else'lazy'
  resources=dict(elapsed_seconds=seconds-.01,user_cpu_seconds=.3,system_cpu_seconds=.01,peak_rss_kib=1234,returncode=0)
  work=dict(seconds=seconds,returncode=0,interrupted=False,external_timeout=False,resources=resources)
  audit=dict(audit_available=known,primal=metric,dual=metric,complementarity=metric,objective=1.,mathematical_strict_qcqp=strict,strict_qcqp=strict,solution_origin=origin,native_status='solved',budget_and_timing=dict(budget_integrity_pass=True,budget_exhausted=False))
  result=dict(case_id=ident,method=method,run_id=self.plan['run_id'],contract_sha256=task['contract_sha256'],input_sha256=task['case']['input_sha256'],start_sha256=task['case']['start_sha256'],dataset=task['case']['dataset_id'],regime=task['case']['regime'],family=task['case']['family'],
   work=work,audit_work=dict(work,returncode=0),audit=audit,solution_origin=origin,route_policy_check=dict(passed=True),
   backend='mumps'if method=='ipopt_mumps'else'ma27',binary_sha256='a'*64 if ipopt else'f'*64 if method in e.RESEARCH else'b'*64,installation_sha256=e.value_sha(self.plan['installation']),evidence_sha256={},status='audited',native_status='solved',ipopt_application_status_code=code)
  endpoint=dict(status='solved',outer_iterations=4,linear_solver_backend='inequality-majorants/ma27',lazy_newton_factorizations=4,ipopt_application_status_code=code)
  if ipopt:
   helper,_=e.stats_module();endpoint.update(adapter_version='ipopt-library-defaults-original-qcqp-v1',effective_options={**helper.campaign.EXPECTED_DEFAULTS,'linear_solver':result['backend'],'max_iter':5000,'max_cpu_time':7200.,'max_wall_time':7200.})
   if method=='ipopt_mumps':endpoint['effective_options']['mumps_pivot_order']=7
  if method in e.RESEARCH:
   result['factorial_proof']=dict(cell=e.RESEARCH[method],objective_exact=True,trace_success=True,budget_verified=True,preregistration_sha256=self.fact['preregistration_sha256'],accepted_direction_origins={'majorant':1},accepted_events=1)
  write(folder/'endpoint.json',endpoint);write(folder/'ENDPOINT_METADATA.json',dict(raw_endpoint_sha256=e.sha(folder/'endpoint.json'),endpoint=endpoint))
  write(folder/'AUDIT.json',audit)
  proof=folder/'PREPOST_INTEGRITY.json';write(proof,dict(before_pass=True,after_pass=True,input_sha256=result['input_sha256'],execution_binding_sha256=self.plan['execution_binding_sha256']))
  for path in folder.iterdir():result['evidence_sha256'][str(path.relative_to(self.run))]=e.sha(path)
  if final:result['integrity_receipt']=str(proof.relative_to(self.run))
  normalizer=e.module(HERE/'campaign/observation_contract.py','test_normalizer')
  result=normalizer.normalize_result(result,integrity_verified=final)
  write(folder/'RESULT.json',result);return folder,result
class Tests(unittest.TestCase):
 def test_eight_methods_and_incomplete_states_remain_separate(self):
  with tempfile.TemporaryDirectory()as d:
   f=Fixture(d);f.result('case_one','R3',True);f.result('case_one','ipopt_qcqp',False,1);f.result('case_two','HH',None);f.result('case_three','SS',False,final=False)
   m=e.export(f.run,Path(d)/'out',archive=True,require_full=False)
   self.assertEqual((m['planned_tasks'],m['recorded_tasks'],m['missing_tasks']),(24,3,21))
   rows=[json.loads(x)for x in(Path(d)/'out/OBSERVATIONS.jsonl').read_text().splitlines()]
   self.assertEqual(next(x for x in rows if x['method']=='HH')['strict_qcqp'],None)
   ip=next(x for x in rows if x['method']=='ipopt_qcqp');self.assertTrue(ip['ipopt_internal_acceptable']);self.assertFalse(ip['strict_qcqp'])
   self.assertTrue((Path(d)/'out/DEPOUILLEMENT_A_TRANSMETTRE.zip').is_file())
   compact=json.loads((Path(d)/'out/evidence/PLAN_COMPACT.json').read_text());restored=compact['plan_header'].copy();bycase={c['id']:c for c in restored['cases']}
   restored['tasks']=[{**{k:v for k,v in t.items()if k!='case_reference_id'},'case':bycase[t['case_reference_id']]}for t in compact['tasks']]
   self.assertEqual(e.value_sha(restored),compact['source_plan_value_sha256']);self.assertEqual(compact['source_plan_sha256'],e.sha(f.run/'PLAN.json'))
 def test_mixed_run_is_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   f=Fixture(d);folder,r=f.result('case_one','R3');r['run_id']='historical';write(folder/'RESULT.json',r)
   with self.assertRaisesRegex(ValueError,'Historical'):e.export(f.run,Path(d)/'out',require_full=False)
 def test_changed_compact_audit_is_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   f=Fixture(d);folder,r=f.result('case_one','R3');write(folder/'AUDIT.json',{'bad':True})
   with self.assertRaisesRegex(ValueError,'evidence'):e.export(f.run,Path(d)/'out',require_full=False)
 def test_plan_contract_and_backend_changes_are_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   f=Fixture(d);folder,r=f.result('case_one','R3');r['backend']='mumps';write(folder/'RESULT.json',r)
   with self.assertRaisesRegex(ValueError,'backend'):e.export(f.run,Path(d)/'out',require_full=False)
 def test_factorial_full_contract_is_ingested_by_real_analyzer(self):
  with tempfile.TemporaryDirectory()as d:
   f=Fixture(d,factorial=True,full=True);ident=next(c['case_id']for c in f.corpus if c['regime']=='I')
   for method in e.RESEARCH:f.result(ident,method,strict=None)
   m=e.export(f.run,Path(d)/'out',require_full=True)
   self.assertEqual(m['planned_tasks'],39960);self.assertEqual(m['factorial_analysis']['rows_received'],4)
   self.assertEqual(m['factorial_analysis']['verification_issues'],[])
   self.assertEqual(m['recorded_tasks'],4)
 def test_objectives_and_time_cohorts_compare_all_historical_variants(self):
  with tempfile.TemporaryDirectory()as d:
   f=Fixture(d);f.result('case_one','R3',True,seconds=3.);folder,r=f.result('case_one','HH',True,seconds=2.)
   r['audit']['objective']=2.;write(folder/'AUDIT.json',r['audit']);r['evidence_sha256'][str((folder/'AUDIT.json').relative_to(f.run))]=e.sha(folder/'AUDIT.json');write(folder/'RESULT.json',r)
   e.export(f.run,Path(d)/'out',require_full=False)
   import csv
   with(Path(d)/'out/OBJECTIVES_PAIRED.csv').open()as stream:values=list(csv.DictReader(stream))
   pair=[r for r in values if r['left']=='R3'and r['right']=='HH'and r['stratum_type']=='regime'and r['stratum']=='I']
   self.assertEqual(len(pair),3);self.assertTrue(all(r['left_lower']=='1'for r in pair))
   with(Path(d)/'out/TIMINGS_PAIRED.csv').open()as stream:times=list(csv.DictReader(stream))
   pair=[r for r in times if r['left']=='R3'and r['right']=='HH'and r['stratum_type']=='regime'and r['stratum']=='I'and r['cohort']=='common_strict_both_process_times_over_1s']
   self.assertEqual(float(pair[0]['median_paired_process_time_ratio']),1.5)
if __name__=='__main__':unittest.main(verbosity=2)
