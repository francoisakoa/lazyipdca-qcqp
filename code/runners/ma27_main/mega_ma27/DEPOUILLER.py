#!/usr/bin/env python3
"""Readonly extraction of the recovered mega controller's finalized observations.
No solver, endpoint replay, source-data import, or historical result is used.
"""
from __future__ import annotations
import argparse,collections,datetime,functools,gzip,hashlib,importlib.util,json,math,shutil,subprocess,sys,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
BASE=('R3','HH','HS','SH','SS','SS_legacy','ipopt_qcqp','ipopt_mumps')
IPOPT=('ipopt_qcqp','ipopt_mumps')
RESEARCH={'F_R3':'R3','F_X1':'X1','F_X2':'X2','F_H0':'H0'}
REGIMES=('I','M','E','Q')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def value_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def reject(value):raise ValueError('Nonfinite JSON value '+str(value))
def load(path):
 path=Path(path)
 with(gzip.open(path,'rt')if path.suffix=='.gz'else path.open())as f:return json.load(f,parse_constant=reject)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def jsonl(path,rows):
 with Path(path).open('w')as f:
  for row in rows:f.write(json.dumps(row,allow_nan=False,ensure_ascii=False)+'\n')
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
@functools.lru_cache(maxsize=1)
def stats_module():
 candidates=(HERE.parent/'default_relaunch',HERE.parent/'article_revision_R4/default_relaunch')
 for folder in candidates:
  path=folder/'DEPOUILLER.py'
  if path.is_file():
   sys.path.insert(0,str(folder));m=module(path,'mega_shared_descriptive_analysis');m.DEFAULTS=IPOPT;m.CELLS=tuple(RESEARCH)
   return m,path
 raise ValueError('Bundled descriptive analysis helper is absent')
def checked(path,expected,label):
 p=Path(path).resolve()
 if not p.is_file()or not expected or sha(p)!=expected:raise ValueError(label+' absent or changed: '+str(p))
 return p

def configuration(run,require_full=True):
 run=Path(run).resolve();plan_path=run/'PLAN.json';plan=load(plan_path);binding=load(run/'PLAN_BINDING.json')
 if binding.get('plan_sha256')!=sha(plan_path):raise ValueError('Frozen campaign plan changed')
 if plan.get('schema')!='mega-fresh-continuous-no-dey-v1' or not plan.get('run_id'):
  raise ValueError('A contemporary MA27 mega plan with run_id is required')
 if Path(plan.get('output_directory','')).resolve()!=run:raise ValueError('Result directory and frozen plan disagree')
 installation=plan['installation']
 if installation.get('backend')!='ma27' or installation.get('native_backend')!='ma27':raise ValueError('All native policies must use the declared MA27 installation')
 default_path=checked(installation['ipopt_defaults_installation'],installation.get('ipopt_defaults_installation_sha256'),'Qualified default Ipopt installation')
 default=load(default_path)
 if default.get('default_binary_sha256')!=installation.get('ipopt_binary_sha256') or default.get('hsl_library_sha256')!=installation.get('hsl_library_sha256'):raise ValueError('Merged and default Ipopt binary/HSL identities differ')
 corpus_path=checked(default['contract'],default.get('contract_sha256'),'Published frozen corpus contract')
 contract=load(corpus_path);corpus={c['case_id']:c for c in contract['cases']};cases=plan['cases']
 if len(corpus)!=len(contract['cases'])or len(cases)!=len(corpus)or {c['id']for c in cases}!=set(corpus):raise ValueError('Mega and published frozen corpus differ')
 if require_full:
  if len(cases)!=4078 or len({c['dataset_id']for c in cases})!=28:raise ValueError('Full4078-case28-dataset corpus required')
  if collections.Counter(c['regime']for c in cases)!=collections.Counter({'I':1037,'M':797,'E':1704,'Q':540}):raise ValueError('Complete I/M/E/Q population differs')
 if any(c['regime']not in REGIMES or 'dey'in c['dataset_id'].lower()for c in cases):raise ValueError('Out-of-scope regime or excluded Dey dataset')
 for case in cases:
  c=corpus[case['id']]
  for key,wanted in [('input_sha256',c['input_sha256']),('regime',c['regime']),('dataset_id',c['dataset'])]:
   if case.get(key)!=wanted:raise ValueError('Case and published corpus differ: '+case['id']+' '+key)
  if(case.get('start_sha256')or case.get('embedded_start_sha256'))!=c['start_binding']['start_sha256']:raise ValueError('Raw start binding differs: '+case['id'])
 methods=tuple(plan['methods']);factorial_enabled=bool(installation.get('factorial_installation'))
 if set(methods)!=set(BASE+(tuple(RESEARCH)if factorial_enabled else())):raise ValueError('Declared historical/native research method set differs')
 expected={(c['id'],m):c for c in cases for m in methods if m not in RESEARCH or c['regime']in('I','M')}
 tasks={}
 for task in plan['tasks']:
  key=(task['case']['id'],task['method'])
  if key not in expected or key in tasks or task['case']!=expected[key]or task['id']!=key[0]+'__'+key[1]:raise ValueError('Unplanned or duplicated task in mega plan')
  if task.get('contract_sha256')!=value_sha(dict(case=task['case'],method=task['method'],execution_binding_sha256=plan['execution_binding_sha256'])):raise ValueError('Task contract changed: '+task['id'])
  tasks[key]=task
 if set(tasks)!=set(expected):raise ValueError('Mega task matrix incomplete')
 execution={**{k:plan[k]for k in('installation','runtime_sha256','source_sha256','profiles','seconds','max_outer','shared_restoration','tolerance','threads','al_mode','methods','datasets')},'run_id':plan.get('run_id')}
 if value_sha(execution)!=plan.get('execution_binding_sha256'):raise ValueError('Mega execution binding changed')
 if(plan['seconds'],plan['max_outer'],plan['shared_restoration'],plan['tolerance'],plan['threads'])!=(7200,5000,600,1e-6,1):raise ValueError('Scientific task budgets differ')
 paths={'PLAN':plan_path,'PLAN_BINDING':run/'PLAN_BINDING.json','default_installation':default_path,'corpus_contract':corpus_path}
 native_path=checked(installation['native_backend_qualification'],installation.get('native_backend_qualification_sha256'),'Native MA27 qualification')
 native=load(native_path)
 required={'schema':'native-ma27-six-profiles-qualification-v1','passed':True,'native_backend':'ma27','binary_sha256':installation['binary_sha256'],'hsl_library_sha256':installation['hsl_library_sha256'],'six_profiles_qualified':True,'mumps_compiled':False}
 if any(native.get(k)!=v for k,v in required.items())or set(native.get('profiles',[]))!=set(BASE[:6]):raise ValueError('Fresh six-profile MA27 qualification differs')
 observations=native.get('observations',[])
 if len(observations)!=24 or {(r.get('regime'),r.get('method'))for r in observations}!={(g,m)for g in REGIMES for m in BASE[:6]}:raise ValueError('All24 native MA27 controls required')
 for row in observations:
  audit=row.get('original_audit')or{}
  if row.get('passed')is not True or row.get('ma27_primary_successful_factors',0)<1 or audit.get('strict')is not True or any(type(audit.get(k))not in(int,float)or not math.isfinite(audit[k])or not 0<=audit[k]<=1e-6 for k in('feasibility','dual','complementarity')):raise ValueError('Native MA27 qualification is incomplete')
 paths['native_backend_qualification']=native_path
 for key in ('qualification','build','frozen_resolution'):
  if default.get(key):paths['default_'+key]=checked(default[key],default.get(key+'_sha256'),'Default '+key)
 for key in ('native_backend_qualification','native_ma27_adaptation','native_build_manifest'):
  if installation.get(key):paths[key]=checked(installation[key],installation.get(key+'_sha256'),key)
 fact=None
 if factorial_enabled:
  fp=checked(installation['factorial_installation'],installation.get('factorial_installation_sha256'),'Factorial installation');fact=load(fp);paths['factorial_installation']=fp
  if fact.get('native_backend')!='ma27':raise ValueError('Historical MUMPS factorial configuration cannot qualify MA27')
  for key in ('preregistration','qualified_50gate','coupling_proof','build'):
   if fact.get(key):paths['factorial_'+key]=checked(fact[key],fact.get(key+'_sha256'),'Factorial '+key)
  gate=load(paths['factorial_qualified_50gate'])
  if not all(gate.get(k)is True for k in('passed','production_authorized','coupling_separation_verified'))or gate.get('native_backend')!='ma27' or gate.get('factor_binary_sha256')!=fact.get('binary_sha256')or gate.get('r3_diagonal_cases')!=50 or gate.get('hs_diagonal_cases')!=50:raise ValueError('MA27 factorial qualification is absent or differs')
  prereg=load(paths['factorial_preregistration'])
  if prereg['inputs']['contract_file_sha256']!=sha(corpus_path)or gate.get('registration_sha256')!=sha(paths['factorial_preregistration']):raise ValueError('Preregistered corpus or experiment differs')
 return plan,corpus,methods,tasks,paths,default,fact

def relative_path(run,value):
 p=Path(value)
 if p.is_absolute()or'..'in p.parts:raise ValueError('Invalid observation evidence path: '+value)
 q=(run/p).resolve()
 if not q.is_relative_to(run):raise ValueError('Observation evidence escapes its run')
 return q

def compact_evidence(run,result,verify_all,verified):
 skipped=[];entries=result.get('evidence_sha256',{})
 if not isinstance(entries,dict):raise ValueError('Observation evidence hashes absent')
 for name,wanted in entries.items():
  path=relative_path(run,name);large=(path.name=='endpoint.json'or path.name.startswith('solver.log')or path.name=='accepted_trace.jsonl')
  if large and not verify_all:skipped.append(dict(path=name,sha256=wanted,policy='archived hash retained; bulk bytes not reread'));continue
  key=(str(path),wanted)
  if key not in verified:checked(path,wanted,'Finalized observation evidence');verified.add(key)
 return skipped

def scalar_work(endpoint,prefix=''):
 result={};markers=('factor','solve','iteration','restoration','eval','seconds','time_','backtrack','newton','work','restart','accepted_steps')
 for key,value in endpoint.items():
  name=prefix+key
  if isinstance(value,dict):result.update(scalar_work(value,name+'.'))
  elif any(marker in name.lower()for marker in markers)and(value is None or type(value)in(str,int,bool)or type(value)is float and math.isfinite(value)):result[name]=value
 return result

def normalize_record(run,plan,task,case,result,data_sha,verify_all,verified,normalizer):
 ident=task['id'];method=task['method'];folder=run/'tasks'/ident
 if result.get('run_id')!=plan['run_id']or result.get('case_id')!=case['case_id']or result.get('method')!=method or result.get('contract_sha256')!=task['contract_sha256']:raise ValueError('Historical or incorrectly paired observation: '+ident)
 for key,wanted in [('input_sha256',case['input_sha256']),('start_sha256',case['start_binding']['start_sha256']),('dataset',case['dataset']),('regime',case['regime'])]:
  if result.get(key)!=wanted:raise ValueError('Observation binding differs '+ident+' '+key)
 # Preliminary worker files are not yet campaign observations.
 receipt='tasks/'+ident+'/PREPOST_INTEGRITY.json'
 if result.get('integrity_verified')is not True or result.get('integrity_receipt')is None:return None,[]
 if result.get('integrity_receipt')!=receipt or receipt not in result.get('evidence_sha256',{}):raise ValueError('Final integrity receipt is absent or misbound: '+ident)
 skipped=compact_evidence(run,result,verify_all,verified);proof=load(run/receipt)
 if proof.get('before_pass')is not True or proof.get('after_pass')is not True or proof.get('input_sha256')!=case['input_sha256']or proof.get('execution_binding_sha256')!=plan['execution_binding_sha256']:raise ValueError('Final integrity evidence differs: '+ident)
 normalized=normalizer.normalize_result(result,integrity_verified=True)
 for key in ('strict_qcqp','observation_classification','observation_indeterminate_reasons'):
  if normalized.get(key)!=result.get(key):raise ValueError('Published three-valued observation decision differs: '+ident+' '+key)
 strict=result['strict_qcqp'];audit=result.get('audit')or{};work=result.get('work')or{};resources=work.get('resources')or{}
 endpoint={};metadata_name='tasks/'+ident+'/ENDPOINT_METADATA.json'
 if metadata_name in result.get('evidence_sha256',{}):
  compact=load(run/metadata_name);endpoint=compact.get('endpoint')or{}
  original_name='tasks/'+ident+'/endpoint.json'
  if compact.get('raw_endpoint_sha256')!=result['evidence_sha256'].get(original_name):raise ValueError('Compact and original endpoint identity differ: '+ident)
 classification='strict'if strict is True else'nonstrict'if strict is False else'indeterminate'
 backend='ma27'if method not in IPOPT else('ma27'if method=='ipopt_qcqp'else'mumps')
 if result.get('backend')!=backend:raise ValueError('Observed backend differs: '+ident)
 if method not in IPOPT and 'mumps'in str(endpoint.get('linear_solver_backend','')).lower():raise ValueError('Native endpoint reports MUMPS instead of requested MA27: '+ident)
 binary=plan['installation']['ipopt_binary_sha256']if method in IPOPT else plan['installation']['binary_sha256']
 if method in RESEARCH:
  fact=load(plan['installation']['factorial_installation']);binary=fact['binary_sha256']
 if result.get('binary_sha256')!=binary:raise ValueError('Observed binary differs: '+ident)
 if result.get('installation_sha256')!=value_sha(plan['installation']):raise ValueError('Observation installation differs: '+ident)
 code=result.get('ipopt_application_status_code',endpoint.get('ipopt_application_status_code'))
 if code is not None and type(code)is not int:raise ValueError('Ipopt application code is not integer')
 if method not in IPOPT and code is not None:raise ValueError('Native observation contains Ipopt application code')
 censored=any(work.get(k)is True for k in('external_timeout','interrupted'))or(type(work.get('returncode'))is int and work['returncode']<0)
 row=dict(run_id=plan['run_id'],case_id=case['case_id'],dataset=case['dataset'],regime=case['regime'],family=case.get('family'),reporting_family=case.get('reporting_family'),
  method=method,cell=RESEARCH.get(method),backend=backend,classification=classification,strict_qcqp=strict,
  status=result.get('native_status',audit.get('native_status',endpoint.get('status'))),worker_status=result.get('status'),
  original_observation_classification=result['observation_classification'],original_indeterminate_reasons=result['observation_indeterminate_reasons'],
  primal=audit.get('primal'),dual=audit.get('dual'),complementarity=audit.get('complementarity'),objective=audit.get('objective'),
  seconds_process=work.get('seconds'),seconds_process_scope='solver process envelope including resource wrapper startup; independent audit excluded',
  solver_child_wall_seconds=resources.get('elapsed_seconds'),audit_seconds=(result.get('audit_work')or{}).get('seconds'),
  solver_resources=resources,audit_resources=(result.get('audit_work')or{}).get('resources'),censored=censored,
  budget_verified=(audit.get('budget_and_timing')or{}).get('budget_integrity_pass')is True,
  objective_exact=None,trace_success=None,preregistration_sha256=None,accepted_direction_origins=None,
  input_sha256=case['input_sha256'],start_binding=case['start_binding'],start_sha256=case['start_binding']['start_sha256'],
  binary_sha256=binary,installation_sha256=value_sha(plan['installation']),task_contract_sha256=task['contract_sha256'],
  execution_binding_sha256=plan['execution_binding_sha256'],result_sha256=data_sha,
  ipopt_application_status_code=code,ipopt_application_status=result.get('ipopt_application_status',endpoint.get('ipopt_application_status')),
  effective_options=result.get('effective_options',endpoint.get('effective_options')),
  ipopt_internal_success=code==0 if method in IPOPT and code is not None else None,
  ipopt_internal_acceptable=code==1 if method in IPOPT and code is not None else None,
  ipopt_internal_success_or_acceptable=code in(0,1)if method in IPOPT and code is not None else None,
  iterations=result.get('iterations',endpoint.get('outer_iterations',endpoint.get('iterations'))),
  internal_work={**scalar_work(endpoint),**(result.get('internal_work')or{})},original_audit=audit,
  backend_proof=result.get('backend_check'),native_route_proof=result.get('route_policy_check'),factorial_proof=result.get('factorial_proof'),
  endpoint_linear_solver_backend=endpoint.get('linear_solver_backend'),backend_requested=backend)
 for target,key in [('solver_user_cpu_seconds','user_cpu_seconds'),('solver_system_cpu_seconds','system_cpu_seconds'),('solver_peak_rss_kib','peak_rss_kib'),('solver_returncode','returncode')]:row[target]=resources.get(key)
 if method in IPOPT and strict is not None:
  defaults_check=stats_module()[0].campaign.verify_effective_defaults
  defaults_check(endpoint,backend)
 if method in RESEARCH:
  p=result.get('factorial_proof')or{}
  for key in ('objective_exact','trace_success','preregistration_sha256','accepted_direction_origins','accepted_base_direction_origins','accepted_events'):row[key]=p.get(key)
  if p.get('cell')not in(None,RESEARCH[method]):raise ValueError('Research cell differs: '+ident)
  if strict is not None:
   if any(p.get(k)is not True for k in('objective_exact','trace_success','budget_verified')):raise ValueError('Known factorial observation lacks complete exact-objective/trace/budget evidence: '+ident)
   fact=load(plan['installation']['factorial_installation'])
   if p.get('preregistration_sha256')!=fact['preregistration_sha256']:raise ValueError('Known factorial observation uses a different preregistration: '+ident)
 return row,skipped

def balanced(cases,methods,planned,records):
 output=[]
 for regime in('all',)+REGIMES:
  pool=[c for c in cases if regime=='all'or c['regime']==regime]
  for method in methods:
   pool_m=[c for c in pool if(c['case_id'],method)in planned]
   if not pool_m:continue
   for unit in('instance','family','dataset'):
    groups=collections.defaultdict(list)
    for c in pool_m:groups[c['case_id']if unit=='instance'else c.get(unit)or'(unassigned)'].append(c)
    low=[];high=[];unknown=0;known=0
    for group in groups.values():
     yes=sum(records.get((c['case_id'],method),{}).get('strict_qcqp')is True for c in group)
     unk=sum(records.get((c['case_id'],method),{}).get('strict_qcqp')is None for c in group)
     low.append(yes/len(group));high.append((yes+unk)/len(group));unknown+=unk;known+=len(group)-unk
    output.append(dict(regime=regime,method=method,weighting=unit,groups=len(groups),planned=len(pool_m),known=known,unknown_or_missing=unknown,
     coverage_lower_bound=sum(low)/len(low),coverage_upper_bound=sum(high)/len(high),point_rate_identified=unknown==0))
 return output

def export(run,out,archive=False,verify_all=False,require_full=True):
 run=Path(run).resolve();out=Path(out).resolve()
 if out.exists():raise ValueError('Choose a new extraction output folder')
 plan,corpus,methods,tasks,paths,default,fact=configuration(run,require_full)
 helper,helper_path=stats_module();normalizer=module(HERE/'campaign/observation_contract.py','mega_observation_publication_contract')
 records={};sources=[];warnings=[];verified=set();skipped_bulk=[]
 case_list=[corpus[c['id']]for c in plan['cases']]
 planned={(c['case_id'],m):c for c in case_list for m in methods if m not in RESEARCH or c['regime']in('I','M')}
 expected_paths={task['id']:key for key,task in tasks.items()}
 for path in sorted((run/'tasks').glob('*/RESULT.json')):
  key=expected_paths.get(path.parent.name)
  if key is None:raise ValueError('Unplanned historical result folder: '+str(path))
  raw=path.read_bytes();result=json.loads(raw,parse_constant=reject)
  row,bulk=normalize_record(run,plan,tasks[key],corpus[key[0]],result,hashlib.sha256(raw).hexdigest(),verify_all,verified,normalizer)
  if row is None:warnings.append(dict(task=path.parent.name,issue='worker_receipt_not_yet_finalized'));continue
  records[key]=row;sources.append(dict(path=str(path.relative_to(run)),sha256=row['result_sha256']));skipped_bulk.extend(bulk)
 out.mkdir(parents=True);rows=[records[k]for k in sorted(records)]
 jsonl(out/'OBSERVATIONS.jsonl',rows)
 fields=list(dict.fromkeys(list(helper.CSV_FIELDS)+['solver_child_wall_seconds','task_contract_sha256','execution_binding_sha256','original_observation_classification']))
 helper.write_csv(out/'OBSERVATIONS.csv',rows,fields)
 jsonl(out/'ORIGINAL_AUDIT_PROOFS.jsonl',[{k:r.get(k)for k in('run_id','case_id','method','result_sha256','original_audit','backend_proof','native_route_proof','factorial_proof','internal_work','solver_resources','audit_resources')}for r in rows])
 groups={'GLOBAL':(),'BY_METHOD':('method',),'BY_REGIME':('regime','method'),'BY_DATASET':('dataset','method'),'BY_DATASET_REGIME':('dataset','regime','method')}
 summaries={}
 for name,group in groups.items():
  values=helper.summaries(planned,records,group)
  for summary in values:
   members=[r for r in rows if all(r.get(key)==summary.get(key)for key in group)]
   summary['ipopt_success_or_acceptable_original_nonstrict']=sum(r.get('ipopt_internal_success_or_acceptable')is True and r['strict_qcqp']is False for r in members)
   summary['ipopt_success_or_acceptable_original_indeterminate']=sum(r.get('ipopt_internal_success_or_acceptable')is True and r['strict_qcqp']is None for r in members)
  summaries[name]=values;helper.write_csv(out/(name+'.csv'),values)
 _,cross,objectives,timings=helper.pairs(case_list,methods,records)
 for name,values in [('COVERAGE_PAIRED',cross),('OBJECTIVES_PAIRED',objectives),('TIMINGS_PAIRED',timings),('COVERAGE_EQUAL_WEIGHTS',balanced(case_list,methods,planned,records))]:helper.write_csv(out/(name+'.csv'),values)
 missing=[dict(case_id=c['case_id'],method=method,dataset=c['dataset'],regime=c['regime'],strict_qcqp=None,classification='missing_planned_observation',input_sha256=c['input_sha256'],start_sha256=c['start_binding']['start_sha256'])for(caseid,method),c in sorted(planned.items())if(caseid,method)not in records]
 jsonl(out/'MISSING_PLANNED.jsonl',missing)
 evidence=out/'evidence';evidence.mkdir();copied=[]
 for name,path in paths.items():
  if name=='PLAN':
   target=evidence/'PLAN_COMPACT.json'
   compact=dict(schema='mega-compact-plan-evidence-v1',source_plan_sha256=sha(path),source_plan_value_sha256=value_sha(plan),
    plan_header={key:value for key,value in plan.items()if key!='tasks'},
    tasks=[{**{k:v for k,v in task.items()if k!='case'},'case_reference_id':task['case']['id']}for task in plan['tasks']])
   write(target,compact);copied.append(dict(label=name,kind='lossless task-to-case reference compaction',file=str(target.relative_to(out)),sha256=sha(target),source_sha256=sha(path),source_value_sha256=value_sha(plan)))
  else:
   target=evidence/(name+(''.join(path.suffixes)or'.json'));shutil.copyfile(path,target);copied.append(dict(label=name,file=str(target.relative_to(out)),sha256=sha(target)))
 factor_summary=None
 if fact:
  factorial_rows=[r for r in rows if r['method']in RESEARCH];jsonl(out/'OBSERVATIONS_FACTORIELLES.jsonl',factorial_rows)
  analyzer=HERE.parent/'factorial/analysis/ANALYSER_FACTORIEL.py'
  if not analyzer.is_file():analyzer=HERE.parent/'article_revision_R4/factorial/analysis/ANALYSER_FACTORIEL.py'
  if not analyzer.is_file():raise ValueError('Bundled preregistered factorial analyzer missing')
  result=subprocess.run([sys.executable,str(analyzer),'--contract',str(paths['corpus_contract']),'--preregistration',str(paths['factorial_preregistration']),'--observations',str(out/'OBSERVATIONS_FACTORIELLES.jsonl'),'--out',str(out/'factorial')],capture_output=True,text=True)
  (out/'FACTORIAL_ANALYSIS.log').write_text(result.stdout+result.stderr)
  if result.returncode:raise ValueError('Factorial analysis failed; inspect FACTORIAL_ANALYSIS.log')
  factor_summary=load(out/'factorial/FACTORIAL_REPORT.json')
 # No numeric worker index is authoritative: only final atomic receipts above.
 manifest=dict(schema='recovered-mega-ma27-defaults-extraction-v1',run_id=plan['run_id'],run=str(run),
  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),planned_tasks=len(tasks),recorded_tasks=len(rows),missing_tasks=len(missing),
  cases=len(case_list),methods=list(methods),regimes=list(REGIMES),optimizer_invocations=0,
  protocol=dict(native_backend='ma27',ipopt_algorithmic_settings='library-defaults',ipopt_backends=['ma27','mumps']),
  accepted_direction_provenance_scope='Final accepted-direction origins are instrumented only for the four crossed I/M research cells. Fractions for the six historical profiles remain unidentified and are not inferred from candidate counters.',
  result_source='Finalized atomic RESULT.json with execution/task/input/start bindings and original publication normalizer',
  evidence_policy='All referenced bytes rehashed'if verify_all else'Compact original-audit/metadata/command/route/integrity files rehashed; raw endpoint/log/accepted-trace hashes retained without rereading bulk bytes',
  compact_evidence_rehashed=len(verified),bulk_hashes_retained=skipped_bulk,
  ipopt_discordance_policy='Confirmed internal-success/original-nonstrict pairs are separate from internal-success/original-indeterminate observations; without_original_strict includes both and is not a confirmed contradiction count.',
  null_policy='Indeterminate finalized observations and missing planned observations remain distinct; neither is reclassified as a numerical failure.',
  timing_scope='Per-task solver process envelope including resource wrapper startup; independent audit excluded; child wait4 wall/CPU/RSS separate; no startup subtraction.',
  objective_gap='(f_left-f_right)/max(1,abs(f_left),abs(f_right)); common strict original certificates; minimization',
  plan_sha256=sha(run/'PLAN.json'),execution_binding_sha256=plan['execution_binding_sha256'],
  source_bindings=copied,result_files=sources,warnings=warnings,analysis_helper_sha256=sha(helper_path),
  factorial_analysis=None if factor_summary is None else{key:factor_summary[key]for key in('rows_received','missing_cells','verification_issues')})
 write(out/'SUMMARY.json',summaries);write(out/'MANIFEST.json',manifest)
 hashes={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()};write(out/'SHA256.json',hashes)
 if archive:
  with zipfile.ZipFile(out/'DEPOUILLEMENT_A_TRANSMETTRE.zip','x',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
   for p in sorted(out.rglob('*')):
    if p.is_file()and p.name!='DEPOUILLEMENT_A_TRANSMETTRE.zip':z.write(p,p.relative_to(out))
 return manifest

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--archive',action='store_true');p.add_argument('--verify-all-evidence',action='store_true',help='Rehash also large raw endpoint/log/accepted-trace artifacts')
 a=p.parse_args();m=export(a.run,a.out,a.archive,a.verify_all_evidence);print(json.dumps({k:m[k]for k in('run_id','planned_tasks','recorded_tasks','missing_tasks','optimizer_invocations')},indent=2));print('DÉPOUILLEMENT : '+str(a.out.resolve()))
if __name__=='__main__':
 try:main()
 except(OSError,ValueError,KeyError,TypeError)as error:print('DÉPOUILLEMENT REFUSÉ — '+str(error),file=sys.stderr);raise SystemExit(2)
