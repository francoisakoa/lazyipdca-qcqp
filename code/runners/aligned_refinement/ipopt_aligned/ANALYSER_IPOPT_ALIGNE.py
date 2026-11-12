#!/usr/bin/env python3
"""Read-only analysis of four fresh, aligned Ipopt configurations.

All planned cases stay in certificate-coverage denominators. Raw numerical
failure and missing/invalid proof stay distinct. The optional Lazy reference
imports only the twelve native configurations, never the older Ipopt runs.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,itertools,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

CELLS=tuple(f'ipopt_{b}_{s}_aligned'for s in('monotone','adaptive')for b in('ma27','mumps'))
CONFIG={c:dict(backend=c.split('_')[1],strategy=c.split('_')[2])for c in CELLS}
REGIMES=dict(I=1037,M=797,E=1704,Q=540)
COMMON=dict(tol=1e-8,constr_viol_tol=1e-6,dual_inf_tol=1e-6,compl_inf_tol=1e-6,acceptable_iter=0,bound_relax_factor=0.,max_iter=5000,max_cpu_time=7200.,max_wall_time=7200.,option_file_name='',print_user_options='yes',nlp_scaling_method='gradient-based',hessian_approximation='exact',fixed_variable_treatment='make_parameter')
NATIVE=tuple(f'{m}_{b}'for b in('ma27','mumps')for m in('R3','HH','HS','SH','SS','SS_legacy'))

def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def json_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path):
 p=Path(path)
 with(gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:return json.load(f)
def finite(value):
 if isinstance(value,bool):return None
 try:x=float(value)
 except(TypeError,ValueError):return None
 return x if math.isfinite(x)else None
def truth(value):
 if value is True or value in('True','true'):return True
 if value is False or value in('False','false'):return False
 if value is None or value in('','None','null','NA'):return None
 raise ValueError('Invalid explicit certificate flag '+str(value))
def quantile(values,p):
 x=sorted(values)
 if not x:return None
 k=(len(x)-1)*p;i=int(k);f=k-i
 return x[i]if i+1==len(x)else x[i]*(1-f)+x[i+1]*f
def distribution(values):
 v=[x for x in values if finite(x)is not None]
 return dict(N=len(v),minimum=min(v)if v else None,q25=quantile(v,.25),median=quantile(v,.5),q75=quantile(v,.75),maximum=max(v)if v else None)
def csvwrite(path,rows):
 if not rows:return
 with Path(path).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stream(path,allow_partial=False):
 p=Path(path)
 with(gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:
  if '.csv'in p.name:yield from csv.DictReader(f);return
  for line_no,line in enumerate(f,1):
   if not line.strip():continue
   try:yield json.loads(line)
   except json.JSONDecodeError:
    if allow_partial and not line.endswith('\n')and not f.read(1):return
    raise ValueError('Malformed observation JSON at line '+str(line_no))
def case_id(r):return r.get('case_id',r.get('id'))
def start_hash(r):return r.get('start_sha256')or(r.get('start_binding')or{}).get('start_sha256')
def plan_cases(plan,require_full=True):
 cs=plan.get('case_bindings')or plan.get('cases')or plan.get('all_cases')
 if isinstance(cs,dict):cs=list(cs.values())
 if not isinstance(cs,list)or not cs:raise ValueError('Complete case plan absent')
 result={}
 for c in cs:
  cid=case_id(c)
  if not cid or cid in result:raise ValueError('Absent/duplicate planned case')
  if c.get('regime')not in REGIMES:raise ValueError('Unsupported structural regime '+str(c.get('regime')))
  if not c.get('input_sha256')or not start_hash(c):raise ValueError('Frozen input/start proof absent: '+cid)
  result[cid]=dict(c,case_id=cid,start_sha256=start_hash(c),dataset=c.get('dataset',c.get('dataset_id')),family=c.get('family')or c.get('reporting_family')or c.get('dataset')or'unclassified')
 if require_full and(len(result)!=4078 or Counter(c['regime']for c in result.values())!=Counter(REGIMES)):raise ValueError('The full4078-case I/M/E/Q plan is required')
 methods=plan.get('methods',plan.get('configurations'))
 if methods is not None and set(methods)!=set(CELLS):raise ValueError('Plan must contain exactly the four aligned Ipopt configurations')
 return result

def identify(row):
 c=row.get('configuration',row.get('cell',row.get('method',row.get('config'))))
 if c in CELLS:return c
 if c is not None:return None
 b=str(row.get('backend','')).lower();s=str(row.get('strategy',row.get('mu_strategy',''))).lower()
 k=f'ipopt_{b}_{s}_aligned';return k if k in CELLS else None

def own_success(row):
 code=row.get('ipopt_application_status_code')
 if code is not None:
  if isinstance(code,bool):return None
  try:return int(code)in(0,1)
  except(TypeError,ValueError):return None
 value=row.get('ipopt_internal_success_or_acceptable')
 return truth(value)if value is not None else None

def validate_row(row,cfg,case,run_id=None,binary_sha256=None):
 issues=[];p=row.get('provenance')or{};opts=row.get('effective_options')or p.get('effective_options')or{};strict=truth(row.get('strict_original',row.get('strict_qcqp')))
 if row.get('classification')=='indeterminate':strict=None
 expected=CONFIG[cfg]
 if row.get('input_sha256')!=case['input_sha256']:issues.append('frozen_input_hash_mismatch')
 if start_hash(row)!=case['start_sha256']:issues.append('raw_start_hash_mismatch')
 if run_id and row.get('run_id')!=run_id:issues.append('run_identity_mismatch')
 if not row.get('run_id'):issues.append('run_identity_absent')
 if not row.get('binary_sha256'):issues.append('adapter_binary_identity_absent')
 if binary_sha256 and row.get('binary_sha256')!=binary_sha256:issues.append('adapter_binary_identity_mismatch')
 for k,v in dict(COMMON,mu_strategy=expected['strategy'],linear_solver=expected['backend']).items():
  if opts.get(k)!=v:issues.append('effective_option:'+k)
 for k in('aligned_options_verified','native_route_verified','backend_verified','option_file_disabled','backend_banner_verified'):
  if p.get(k)is not True:issues.append('proof:'+k)
 for key in('endpoint_sha256','solver_log_sha256'):
  if not p.get(key):issues.append('proof:'+key)
 if p.get('mu_strategy')!=expected['strategy']or p.get('linear_solver')!=expected['backend']:issues.append('strategy_or_backend_proof_mismatch')
 if p.get('effective_options_sha256')!=json_sha(opts):issues.append('effective_options_hash_mismatch')
 if truth(row.get('budget_verified'))is not True or p.get('budget_verified')is not True:issues.append('budget_unverified_or_exceeded')
 resources=row.get('solver_resources')or{}
 for k in('elapsed_seconds','user_cpu_seconds','system_cpu_seconds'):
  x=finite(resources.get(k))
  if x is None or x<0:issues.append('process_resource:'+k)
 elapsed=finite(resources.get('elapsed_seconds'));u=finite(resources.get('user_cpu_seconds'));system=finite(resources.get('system_cpu_seconds'))
 if elapsed is not None and elapsed>7200:issues.append('process_wall_budget_exceeded')
 if u is not None and system is not None and u+system>7200:issues.append('process_CPU_budget_exceeded')
 if resources.get('censored')is not False or row.get('censored')is True or resources.get('timed_out')is True or resources.get('interrupted')is True:issues.append('censored_or_interrupted')
 rc=resources.get('returncode')
 if type(rc)is not int or rc<0:issues.append('process_exit_unverified_or_signal')
 ni=p.get('native_iterations');nt=finite(p.get('native_elapsed_seconds'))
 if type(ni)is not int or not 0<=ni<=5000 or nt is None or not 0<=nt<=7200:issues.append('native_time_or_iteration_budget_unverified')
 vals=[finite(row.get(k))for k in('primal','dual','complementarity')]
 if any(x is None or x<0 for x in vals):issues.append('original_residuals_absent_or_nonfinite')
 elif strict is not None and strict!=all(x<=1e-6 for x in vals):issues.append('certificate_residual_disagreement')
 audit=row.get('original_audit')
 if not isinstance(audit,dict):issues.append('independent_original_audit_absent')
 else:
  af=truth(audit.get('strict_original',audit.get('strict_qcqp')))
  if af!=strict or any(finite(audit.get(k))!=finite(row.get(k))for k in('primal','dual','complementarity')):issues.append('independent_original_audit_disagreement')
 if strict is None:issues.append('original_certificate_unqualified')
 if issues:strict=None
 mask='+'.join(k for k,x in zip(('primal','dual','complementarity'),vals)if x is not None and x>1e-6)or'none'
 category='strict_certificate'if strict is True else'above_original_tolerance:'+mask if strict is False else'execution_or_audit_unqualified'
 compact=dict(case_id=case['case_id'],config=cfg,backend=expected['backend'],strategy=expected['strategy'],regime=case['regime'],dataset=case['dataset'],family=case['family'],strict=strict,classification='strict'if strict is True else'nonstrict'if strict is False else'indeterminate',
  source_classification=row.get('classification'),issues=sorted(set(issues)),failure_category=category,original_violation_mask=mask,
  input_sha256=case['input_sha256'],start_sha256=case['start_sha256'],run_id=row.get('run_id'),objective=finite(row.get('objective')),primal=vals[0],dual=vals[1],complementarity=vals[2],
  own_success_or_acceptable=own_success(row),ipopt_application_status_code=row.get('ipopt_application_status_code'),ipopt_application_status=row.get('ipopt_application_status'),native_status=row.get('status'),
  seconds_process=finite(row.get('seconds_process')),seconds_process_scope=row.get('seconds_process_scope'),seconds_numerical=nt,audit_seconds=finite(row.get('audit_seconds')),
  iterations=ni,internal_work=row.get('internal_work'),input_representation_overrides=row.get('input_representation_overrides',p.get('input_representation_overrides')),effective_options=opts,binary_sha256=row.get('binary_sha256'),proof_level='supplied independent original audit plus aligned effective options/backend/budget provenance; no endpoint reevaluation')
 return compact

def scopes(cases):
 cs=list(cases.values());out={('all','ALL'):cs}
 for key in REGIMES:out['regime',key]=[c for c in cs if c['regime']==key]
 for partition in('dataset','family'):
  for name in sorted({c[partition]for c in cs}):out[partition,name]=[c for c in cs if c[partition]==name]
 m=[c for c in cs if c['regime']=='M'];out['M_collection','ASU']= [c for c in m if c['dataset']=='beam_structural'];out['M_collection','outside_ASU']=[c for c in m if c['dataset']!='beam_structural']
 return out

def coverage(group,idx,configs):
 rows=[]
 for config in configs:
  rr=[idx.get((c['case_id'],config),{})for c in group];n=len(group);success=sum(r.get('strict')is True for r in rr);known=sum(r.get('strict')is False for r in rr);unknown=n-success-known
  own=sum(r.get('own_success_or_acceptable')is True for r in rr);own_only=sum(r.get('own_success_or_acceptable')is True and r.get('strict')is not True for r in rr);orig_only=sum(r.get('strict')is True and r.get('own_success_or_acceptable')is False for r in rr)
  rows.append(dict(configuration=config,planned=n,success=success,failure_to_certify=n-success,strict_rate=success/n if n else None,audited_nonstrict=known,technical_or_missing=unknown,own_success_or_acceptable=own,own_only=own_only,original_only=orig_only,positive_return_original_strict=sum(r.get('own_success_or_acceptable')is True and r.get('strict')is True for r in rr),positive_return_audited_nonstrict=sum(r.get('own_success_or_acceptable')is True and r.get('strict')is False for r in rr),positive_return_unqualified=sum(r.get('own_success_or_acceptable')is True and r.get('strict')is None for r in rr)))
 return rows

def pair_summary(group,idx,left,right):
 pp=[];both=lo=ro=neither=unknown=0
 for c in group:
  a=idx.get((c['case_id'],left),{});b=idx.get((c['case_id'],right),{});sa=a.get('strict')is True;sb=b.get('strict')is True
  both+=sa and sb;lo+=sa and not sb;ro+=sb and not sa;neither+=not sa and not sb;unknown+=a.get('strict')is None or b.get('strict')is None
  if sa and sb:pp.append((a,b))
 gaps=[(a['objective']-b['objective'])/max(1,abs(a['objective']),abs(b['objective']))for a,b in pp if finite(a.get('objective'))is not None and finite(b.get('objective'))is not None]
 base=dict(left=left,right=right,N_planned=len(group),both_success=both,left_only=lo,right_only=ro,both_failure_to_certify=neither,technical_or_missing_pairs=unknown,
  objective_common_N=len(gaps),signed_gap_q25=quantile(gaps,.25),signed_gap_median=quantile(gaps,.5),signed_gap_q75=quantile(gaps,.75),absolute_gap_q25=quantile([abs(v)for v in gaps],.25),absolute_gap_median=quantile([abs(v)for v in gaps],.5),absolute_gap_q75=quantile([abs(v)for v in gaps],.75))
 for t,tag in[(1e-3,'tol_1em3'),(1e-2,'tol_1em2')]:base.update({f'lower_{tag}':sum(v< -t for v in gaps),f'equivalent_{tag}':sum(abs(v)<=t for v in gaps),f'higher_{tag}':sum(v>t for v in gaps)})
 times=[]
 for clock in('seconds_process','seconds_numerical'):
  for cohort in('all_common_strict','both_gt1s','both_gt1s_objective_equivalent_1e3'):
   vv=[]
   for a,b in pp:
    x,y=a.get(clock),b.get(clock)
    if x is None or y is None or x<=0 or y<=0:continue
    if cohort!='all_common_strict'and(x<=1 or y<=1):continue
    if 'objective_equivalent'in cohort:
     fa,fb=a.get('objective'),b.get('objective')
     if fa is None or fb is None or abs(fa-fb)/max(1,abs(fa),abs(fb))>1e-3:continue
    vv.append(x/y)
   times.append(dict(left=left,right=right,clock=clock,cohort=cohort,N=len(vv),q25=quantile(vv,.25),median=quantile(vv,.5),q75=quantile(vv,.75)))
 return base,times

def native_reference(path,cases):
 idx={};levels=Counter()
 for r in stream(path):
  config=r.get('config')
  if config not in NATIVE:continue
  cid=r.get('case_id')
  if cid not in cases:raise ValueError('Native reference outside frozen plan')
  c=cases[cid]
  if r.get('input_sha256')!=c['input_sha256']or start_hash(r)!=c['start_sha256']:raise ValueError('Native reference input/start mismatch: '+cid)
  key=cid,config
  if key in idx:raise ValueError('Duplicate native reference')
  strict=True if r.get('classification')=='strict'else False if r.get('classification')=='nonstrict'else None
  vals=[finite(r.get(k))for k in('primal','dual','complementarity')]
  if strict is not None and(any(x is None or x<0 for x in vals)or strict!=all(x<=1e-6 for x in vals)):raise ValueError('Native supplied certificate and original residuals disagree')
  idx[key]=dict(r,config=config,strict=strict,objective=finite(r.get('objective')),seconds_process=finite(r.get('seconds_process')),seconds_numerical=None);levels[r.get('proof_level','unspecified')]+=1
 if set(idx)!={(cid,cfg)for cid in cases for cfg in NATIVE}:raise ValueError('Full twelve-native reference matrix required')
 return idx,dict(levels)

def analyse(cases,observations,plan=None,lazy=None,require_full=True):
 plan=plan or{};idx={};runs=set();binaries=set();option_sets=defaultdict(set);issues=[]
 for row in observations:
  config=identify(row)
  if config is None:raise ValueError('Observation is not one of the four aligned Ipopt cells')
  cid=row.get('case_id');key=cid,config
  if cid not in cases:raise ValueError('Unplanned case '+str(cid))
  if key in idx:raise ValueError('Duplicate case/configuration '+str(key))
  r=validate_row(row,config,cases[cid],plan.get('run_id'),plan.get('binary_sha256'));idx[key]=r
  if r['run_id']:runs.add(r['run_id'])
  if r['binary_sha256']:binaries.add(r['binary_sha256'])
  if r['strict']is not None:
   o=dict(r['effective_options']);o.pop('mu_strategy',None);option_sets[cid,r['backend']].add(json_sha(o))
  if r['issues']:issues.append(dict(case_id=cid,configuration=config,reasons=r['issues']))
 if len(runs)>1:raise ValueError('Different fresh Ipopt experiments cannot be pooled')
 if len(binaries)>1:raise ValueError('Four aligned cells must use the same admitted adapter binary')
 if any(len(v)>1 for v in option_sets.values()):raise ValueError('Unregistered options vary beyond mu_strategy within a case/backend pair')
 expected={(c,cfg)for c in cases for cfg in CELLS};missing=expected-set(idx)
 if require_full and missing:raise ValueError('Incomplete four-cell matrix; use --allow-partial only for provisional monitoring')
 levels={}
 if lazy is not None:
  native,levels=native_reference(lazy,cases);idx.update(native)
 cfgs=CELLS+(NATIVE if lazy is not None else())
 cov=[];pairs=[];time=[];fail=[];residuals=[]
 ip_pairs=list(itertools.combinations(CELLS,2));all_pairs=ip_pairs+[(n,i)for n in NATIVE for i in CELLS]if lazy is not None else ip_pairs
 for(scope_type,scope),group in scopes(cases).items():
  prefix=dict(scope_type=scope_type,scope=scope)
  cov.extend(dict(prefix,**r)for r in coverage(group,idx,cfgs))
  for left,right in all_pairs:
   r,t=pair_summary(group,idx,left,right);pairs.append(dict(prefix,**r));time.extend(dict(prefix,**x)for x in t)
  for config in CELLS:
   rr=[idx.get((c['case_id'],config),{})for c in group];counts=Counter((r.get('failure_category','missing_observation'),str(r.get('ipopt_application_status')),r.get('original_violation_mask','unavailable'))for r in rr if r.get('strict')is not True)
   fail.extend(dict(prefix,configuration=config,category=k[0],internal_return=k[1],original_violation_mask=k[2],count=v)for k,v in sorted(counts.items()))
   for population in('all_available','strict_only'):
    selected=[r for r in rr if population=='all_available'or r.get('strict')is True]
    for metric in('primal','dual','complementarity'):
     residuals.append(dict(prefix,configuration=config,population=population,metric=metric,**distribution([r.get(metric)for r in selected if finite(r.get(metric))is not None])))
 report=dict(schema='ipopt-aligned-four-cell-analysis-v1',planned_cases=len(cases),planned_Ipopt_tasks=4*len(cases),recorded_Ipopt_tasks=4*len(cases)-len(missing),missing_Ipopt_tasks=len(missing),complete_matrix=not missing,provisional=bool(missing),run_ids=sorted(runs),verification_issues=issues,
  options_scope=dict(common=COMMON,mu_strategies=['monotone','adaptive'],backends=['ma27','mumps'],restoration_limit='No separate600-iteration Ipopt restoration cap is claimed from max_iter; native Lazy has its own declared600 shared allowance'),
  original_certificate='Complete original QCQP: primal,dual,complementarity<=1e-6, plus data/start/route/backend/budget proof',article_outcomes=['success','failure to certify'],technical_states_retained=True,
  quantile_definition='Linear interpolation at (N-1)*p (type7), N explicit',objective_gap='(f_left-f_right)/max(1,abs(f_left),abs(f_right)); negative favors left for minimization',native_reference_included=lazy is not None,native_reference_proof_levels=levels,
  timings='Descriptive single-execution paired ratios, with explicit process/numerical clock and cohorts; no startup subtraction or summing overlapping clocks',
  causal_scope='Only the four freshly aligned Ipopt configurations isolate strategy/backend under common settings. Archived Lazy comparisons are descriptive configurations; old Ipopt references are excluded.',optimizer_invocations=0,new_endpoint_evaluations=0)
 return report,cov,pairs,time,fail,residuals,idx

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--lazy-reference',type=Path);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
 plan=load(a.plan);cases=plan_cases(plan);report,cov,pairs,times,fail,res,idx=analyse(cases,stream(a.observations,a.allow_partial),plan,a.lazy_reference,not a.allow_partial)
 report.update(plan_sha256=sha(a.plan),observations_sha256=sha(a.observations),native_reference_sha256=sha(a.lazy_reference)if a.lazy_reference else None);a.out.mkdir(parents=True,exist_ok=True)
 for name,rows in [('COUVERTURE.csv',cov),('PAIRES_KKT_OBJECTIFS.csv',pairs),('TEMPS_APPARIES.csv',times),('CAUSES_NON_CERTIFICATION.csv',fail),('DISTRIBUTIONS_RESIDUS.csv',res)]:csvwrite(a.out/name,rows)
 (a.out/'ANALYSE_IPOPT_ALIGNE.json').write_text(json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
 with gzip.open(a.out/'OBSERVATIONS_ANALYSE_COMPACTES.jsonl.gz','wt')as f:
  for key in sorted(idx):f.write(json.dumps(idx[key],sort_keys=True,allow_nan=False)+'\n')
 lines=['IPOPT ALIGNÉ — QUATRE CONFIGURATIONS','Cas prévus : '+str(len(cases))+' ; tâches : '+str(report['recorded_Ipopt_tasks'])+'/'+str(report['planned_Ipopt_tasks']), 'État : '+('provisoire'if report['provisional']else'matrice complète'),'Chaque ligne : certificats stricts / prévues ; échecs à produire le certificat ; dossiers techniques inclus dans ces échecs.','']
 for r in cov:
  if r['scope_type']=='regime'and r['configuration']in CELLS:lines.append(f"{r['scope']} {r['configuration']}: {r['success']}/{r['planned']} ; échecs {r['failure_to_certify']} ; techniques/absents {r['technical_or_missing']}")
 lines+=['','La réussite repose sur le QCQP original et les preuves, indépendamment du statut interne.','Les défauts de certificat ne sont pas tous des échecs numériques. Objectifs et temps utilisent des intersections de certificats propres à chaque paire.','Aucun ancien résultat Ipopt adaptatif ou monotone n’est substitué aux quatre cellules neuves.']
 (a.out/'LIRE_IPOPT_ALIGNE.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
