#!/usr/bin/env python3
"""Paired blocked 2x2 analysis of adaptive refinement; no solver is launched.

The primary estimand compares a shared hybrid initialization kept frozen or
adaptively refined, under each of two globalizations. All other settings,
including safeguards and the exact objective, remain fixed. Backends form two
separate blocks, not independent replications of the 1,834 cases.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

CELLS=('FROZEN_GR3','FROZEN_GHS','REFINED_GR3','REFINED_GHS')
BACKENDS=('ma27','mumps')
FACTORS={c:dict(evolution='frozen' if c.startswith('FROZEN') else 'adaptive',globalization='r3' if c.endswith('GR3') else 'hs')for c in CELLS}
CONTRASTS={
 'refinement_average_over_globalizations':dict(FROZEN_GR3=-.5,FROZEN_GHS=-.5,REFINED_GR3=.5,REFINED_GHS=.5),
 'globalization_average_over_evolutions':dict(FROZEN_GR3=-.5,FROZEN_GHS=.5,REFINED_GR3=-.5,REFINED_GHS=.5),
 'refinement_under_r3_globalization':dict(FROZEN_GR3=-1,REFINED_GR3=1),
 'refinement_under_hs_globalization':dict(FROZEN_GHS=-1,REFINED_GHS=1),
 'interaction_difference_of_differences':dict(FROZEN_GR3=1,FROZEN_GHS=-1,REFINED_GR3=-1,REFINED_GHS=1),
}

def load(path):
 p=Path(path)
 with(gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:return json.load(f)
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def finite(value):
 if isinstance(value,bool):return None
 try:x=float(value)
 except(TypeError,ValueError):return None
 return x if math.isfinite(x)else None
def boolean(value):
 if value is True or value in('true','True'):return True
 if value is False or value in('false','False'):return False
 if value is None or value in('','null','None','NA'):return None
 raise ValueError('Invalid explicit outcome '+str(value))
def bounds(x):return(0,1)if x is None else(int(x),int(x))
def pair_delta(a,b):
 al,au=bounds(a);bl,bu=bounds(b);return al-bu,au-bl

def records(path):
 p=Path(path)
 with(gzip.open(p,'rt')if p.suffix=='.gz'else p.open())as f:
  if '.csv'in p.name:yield from csv.DictReader(f)
  else:
   for line in f:
    if line.strip():yield json.loads(line)
def csvwrite(path,rows):
 if not rows:return
 with Path(path).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def states(cases,idx,backend,cell,technical_as_failure=False):
 return[False if technical_as_failure and idx.get((c['case_id'],backend,cell),{}).get('strict')is None else idx.get((c['case_id'],backend,cell),{}).get('strict')for c in cases]

def contrast(cases,idx,backend,coeff,technical_as_failure=False):
 lo=hi=0.
 for c in cases:
  for cell,a in coeff.items():
   x=idx.get((c['case_id'],backend,cell),{}).get('strict')
   if technical_as_failure and x is None:x=False
   l,u=bounds(x);lo+=a*(l if a>=0 else u);hi+=a*(u if a>=0 else l)
 return dict(lower_certificate_equivalents=lo,upper_certificate_equivalents=hi,
  point_identified=lo==hi,lower_percentage_points=100*lo/len(cases)if cases else None,
  upper_percentage_points=100*hi/len(cases)if cases else None)

def primary(cases,idx,backend,minimum=20):
 # Confirmatory comparison is REFINED_GHS - FROZEN_GHS, per backend.
 groups=defaultdict(lambda:[0,0,0]);families=defaultdict(lambda:[0,0,0]);wins=losses=possible_wins=possible_losses=unknown=0
 for c in cases:
  a=idx.get((c['case_id'],backend,'REFINED_GHS'),{}).get('strict');b=idx.get((c['case_id'],backend,'FROZEN_GHS'),{}).get('strict')
  lo,hi=pair_delta(a,b)
  for dest,key in((groups,c['dataset']),(families,c['family'])):dest[key][0]+=lo;dest[key][1]+=hi;dest[key][2]+=1
  wins+=a is True and b is False;losses+=a is False and b is True
  possible_wins+=a is not False and b is not True;possible_losses+=a is not True and b is not False;unknown+=a is None or b is None
 lo=sum(v[0]for v in groups.values());hi=sum(v[1]for v in groups.values())
 fl=sum(v[0]/v[2]for v in families.values())/len(families)if families else 0.
 fu=sum(v[1]/v[2]for v in families.values())/len(families)if families else 0.
 # For each group d, 2*Delta_d - Delta_total = Delta_d - Delta_outside.
 # This preserves dependence between a group's gain and the total.
 concentration={k:dict(net_lower=v[0],net_upper=v[1],planned=v[2],
  twice_group_minus_total_lower=v[0]-(hi-v[1]),twice_group_minus_total_upper=v[1]-(lo-v[0]))for k,v in groups.items()}
 conditions=dict(net_certificates_at_least_minimum_robust=lo>=minimum,
  paired_discordances_at_least_two_to_one_robust=wins>0 and wins>=2*possible_losses,
  no_collection_above_half_net_gain_robust=lo>0 and all(v['twice_group_minus_total_upper']<=0 for v in concentration.values()),
  family_macro_sign_positive_robust=fl>0)
 if all(conditions.values()):decision='positive_under_preregistered_rule'
 elif unknown==0:decision='criterion_not_met'
 elif hi<minimum or possible_wins<2*losses or fu<=0 or(lo>0 and any(v['twice_group_minus_total_lower']>0 for v in concentration.values())):decision='criterion_not_met_for_every_possible_completion'
 else:decision='not_identified_due_to_technical_or_missing_outcomes'
 return dict(backend=backend,scope='I+M confirmatory; I and M descriptive only',
  contrast='REFINED_GHS minus FROZEN_GHS',minimum_net_certificates=minimum,
  net_gain_bounds=[lo,hi],known_wins=wins,known_losses=losses,
  possible_wins_upper=possible_wins,possible_losses_upper=possible_losses,
  technical_or_missing_pairs=unknown,family_macro_delta_bounds=[fl,fu],
  collection_contributions=concentration,conditions=conditions,decision=decision)

def secondary(cases,idx,backend):
 rows=[]
 for i,a in enumerate(CELLS):
  for b in CELLS[i+1:]:
   pp=[(idx.get((c['case_id'],backend,a),{}),idx.get((c['case_id'],backend,b),{}))for c in cases]
   pp=[(x,y)for x,y in pp if x.get('strict')is True and y.get('strict')is True]
   gg=[(x['objective']-y['objective'])/max(1,abs(x['objective']),abs(y['objective']))for x,y in pp if finite(x.get('objective'))is not None and finite(y.get('objective'))is not None]
   tt=[x['seconds_process']/y['seconds_process']for x,y in pp if finite(x.get('seconds_process'))is not None and finite(y.get('seconds_process'))is not None and x['seconds_process']>1 and y['seconds_process']>1]
   for threshold in(1e-3,1e-2):rows.append(dict(backend=backend,left=a,right=b,common_strict_N=len(pp),objective_N=len(gg),threshold=threshold,
    left_lower=sum(g< -threshold for g in gg),equivalent=sum(abs(g)<=threshold for g in gg),left_higher=sum(g>threshold for g in gg),
    both_process_times_gt_1s_N=len(tt),median_process_ratio=statistics.median(tt)if tt else None))
 return rows

def origins(cases,idx,backend):
 out=[]
 for cell in CELLS:
  counts=Counter();logged=missing=0
  for c in cases:
   r=idx.get((c['case_id'],backend,cell),{});v=r.get('accepted_direction_origins')
   if not isinstance(v,dict):missing+=1;continue
   if any(type(n)is not int or n<0 for n in v.values()):raise ValueError('Invalid accepted-origin counts')
   total=r.get('accepted_primary_updates')
   if type(total)is not int or total!=sum(v.values()):missing+=1;continue
   counts.update(v);logged+=1
  total=sum(counts.values());pure=counts['majorant_primary']+counts['majorant_refined'];unknown=counts['unknown']+counts['unidentified']
  out.append(dict(backend=backend,cell=cell,tasks_with_complete_origin_counts=logged,tasks_missing_origin_proof=missing,
   accepted_updates_identified=total,final_origin_counts=dict(counts),pure_majorant_updates=pure,
   pure_majorant_fraction_on_identified_updates=pure/(total-unknown)if total>unknown else None,pure_majorant_fraction_bounds=[pure/total,(pure+unknown)/total]if total else None,unknown_final_origin_updates=unknown,
   full_scope_fraction_identified=missing==0 and unknown==0))
 return out

def summarize(cases,idx,backends=BACKENDS,minimum=20):
 out={};sec=[]
 for scope,group in [('I+M',cases)]+[(g,[c for c in cases if c['regime']==g])for g in('I','M')]:
  out[scope]={}
  for b in backends:
   totals={}
   for cell in CELLS:
    z=Counter(states(group,idx,b,cell));totals[cell]=dict(planned=len(group),success=z[True],failure_to_certify=len(group)-z[True],
     audited_nonstrict=z[False],technical_or_missing=z[None])
   effects={name:dict(sensitivity_bounds=contrast(group,idx,b,v),binary_certificate_effect=contrast(group,idx,b,v,True))for name,v in CONTRASTS.items()}
   result=dict(planned=len(group),binary_outcomes=totals,effects=effects,primary_rule=primary(group,idx,b,minimum),accepted_direction_origins=origins(group,idx,b))
   result['composition']={}
   for partition in('dataset','family'):
    strata=defaultdict(list)
    for c in group:strata[c[partition]].append(c)
    result['composition'][partition]={}
    for name,members in sorted(strata.items()):
     result['composition'][partition][name]=dict(planned=len(members),effects={k:contrast(members,idx,b,v)for k,v in CONTRASTS.items()})
   out[scope][b]=result
   sec.extend(dict(scope=scope,**r)for r in secondary(group,idx,b))
 # Only one overarching affirmative claim: the a priori rule succeeds in BOTH blocks.
 decisions={b:out['I+M'][b]['primary_rule']['decision']for b in backends}
 positive=all(v=='positive_under_preregistered_rule'for v in decisions.values())
 incomplete=any(v=='not_identified_due_to_technical_or_missing_outcomes'for v in decisions.values())
 overall='positive_on_both_backends_under_preregistered_rule'if positive else'general_two_backend_claim_not_established'if any(v.startswith('criterion_not_met')for v in decisions.values())else'not_identified_due_to_technical_or_missing_outcomes'if incomplete else'general_two_backend_claim_not_established'
 averages={}
 for effect in CONTRASTS:
  lo=sum(out['I+M'][b]['effects'][effect]['sensitivity_bounds']['lower_certificate_equivalents']for b in backends)/len(backends)
  hi=sum(out['I+M'][b]['effects'][effect]['sensitivity_bounds']['upper_certificate_equivalents']for b in backends)/len(backends)
  averages[effect]=dict(lower_mean_certificate_equivalents=lo,upper_mean_certificate_equivalents=hi,backend_weights_equal=True)
 return dict(results=out,backend_primary_decisions=decisions,overall_decision=overall,balanced_backend_effect_summaries=averages,
  globalization_scope='Declared eight-control package, including initial dual balancing, retraction and SOC; not an isolated filter-only intervention.',inference='Descriptive paired contrasts on fixed corpus; backend blocks are repeated measurements of the same instances, not independent samples.'),sec

def invariant_command_digest(row):
 r=row.get('solver_resources')or{};cmd=r.get('command')
 if not isinstance(cmd,list)or not cmd:return None
 expected=r.get('command_sha256')
 if expected and hashlib.sha256(json.dumps(cmd).encode()).hexdigest()!=expected:return None
 if Path(str(cmd[0])).name=='taskset':
  if len(cmd)<4 or cmd[1]!='-c':return None
  cmd=cmd[3:]
 cmd=cmd[1:] # The executable is separately bound by its recorded binary SHA.
 exclude={'--problem','--json','--factorial-trace','--dca-rank-presolve-report','--linear-backend','--hsl-library','--factorial-refinement','--factorial-globalization'}
 kept=[];i=0
 while i<len(cmd):
  if cmd[i]in exclude:
   if i+1>=len(cmd):return None
   i+=2
  else:kept.append(cmd[i]);i+=1
 return hashlib.sha256(json.dumps(kept,separators=(',',':')).encode()).hexdigest()

def canonical_observation(raw):
 r=dict(raw);cell=str(r.get('cell')or r.get('configuration')or r.get('method')or'')
 b=str(r.get('native_backend',r.get('backend',''))).lower()
 for backend in BACKENDS:
  if cell.lower().endswith('_'+backend):b=backend;cell=cell[:-(len(backend)+1)];break
 r['cell']=cell;r['backend']={'mumps_metis':'mumps','mumps+metis':'mumps','hsl_ma27':'ma27'}.get(b,b)
 p=r.get('provenance')or{}
 if p.get('intervention_verified')is True:
  requested='ma27'if r['backend']=='ma27'else'mumps_metis';wanted='inequality-majorants/'+r['backend']
  early=(p.get('native_pre_model_return')is True or p.get('backend_invoked')is False)and p.get('initial_model_events')==0 and p.get('accepted_events')==0
  r['backend_verified']=p.get('requested_linear_backend')==requested and(p.get('actual_linear_backend')==wanted or early)
  r['native_route_verified']=True
 audit=r.get('original_audit')
 if isinstance(audit,dict):
  expected=boolean(r.get('strict_original',r.get('strict_qcqp')));actual=boolean(audit.get('strict_original',audit.get('strict_qcqp')))
  r['original_audit_verified']=actual is not None and actual==expected and all(finite(audit.get(k))==finite(r.get(k))for k in('primal','dual','complementarity'))
 if not isinstance(r.get('factorial_proof'),dict)and p.get('intervention_verified')is True:
  count=p.get('initial_model_events');digest=p.get('initial_inequality_model_sha256')
  state='reached'if type(count)is int and count>0 else'not_reached'if count==0 and digest in(None,'')else None
  r['factorial_proof']=dict(correction_evolution=p.get('refinement_policy',r.get('refinement_policy')),globalization=p.get('globalization_policy',r.get('globalization_policy')),
   common_representation=p.get('common_representation',r.get('common_representation')),
   guard_policy='common_r3_guards_v1'if p.get('common_guards')=='r3'else None,
   correction_initialization='shared_hybrid_initialization_v1'if p.get('inequality_policy')=='hybrid'else None,
   initial_model_state=state,initialization_calls=count,initial_inequality_model_sha256=digest or None,
   invariant_options_sha256=r.get('invariant_options_sha256')or invariant_command_digest(r),
   guard_config_sha256=p.get('guard_config_sha256',r.get('guard_config_sha256')),accepted_refinements=p.get('accepted_refinements'),refinement_attempt_events=p.get('refinement_attempt_events'),refinement_update_events=p.get('refinement_update_events'),refinement_rollback_events=p.get('refinement_rollback_events'),frozen_metric_invariant_verified=p.get('frozen_metric_invariant_verified'))
  r['accepted_primary_updates']=p.get('accepted_events')
  if r.get('accepted_direction_origins')is None:r['accepted_direction_origins']=p.get('accepted_direction_origins')
 return r

def validate(rows,pre,pre_sha,study_sha=None,expected_binaries=None,expected_guards=None):
 expected={c['case_id']:c for c in pre['inputs']['case_bindings']};idx={};issues=[];run_ids=set();initial=defaultdict(dict);binaries=defaultdict(set)
 aliases={c.lower():c for c in CELLS}
 for number,raw in enumerate(rows,1):
  row=canonical_observation(raw)
  cell=row.get('cell');backend=str(row.get('backend','')).lower();backend={'mumps_metis':'mumps','mumps+metis':'mumps','hsl_ma27':'ma27'}.get(backend,backend)
  if cell is None:
   raw=str(row.get('configuration',row.get('method','')))
   for b in pre['backends']:
    if raw.lower().endswith('_'+b):backend=b;raw=raw[:-(len(b)+1)];break
   cell=aliases.get(raw.lower())
  if cell not in CELLS:continue # Public R3/HH or Ipopt controls are outside this estimand.
  if backend not in pre['backends']:raise ValueError('Unregistered backend '+backend)
  cid=row.get('case_id');key=cid,backend,cell
  if cid not in expected:raise ValueError('Unregistered case '+str(cid))
  if key in idx:raise ValueError('Duplicate factorial cell '+str(key))
  c=expected[cid];p=row.get('factorial_proof')or{};f=FACTORS[cell]
  tests=[(row.get('preregistration_sha256')==pre_sha,'preregistration binding'),(row.get('input_sha256')==c['input_sha256'],'input binding'),
   (row.get('start_sha256')==c['start_sha256'],'raw start binding'),(row.get('run_id')not in(None,''),'same factorial experiment identity'),
   (boolean(row.get('budget_verified'))is True,'shared budget proof'),(boolean(row.get('trace_success'))is True,'accepted trace proof'),
   (boolean(row.get('objective_exact'))is True,'exact objective proof'),(boolean(row.get('backend_verified'))is True,'actual backend proof'),(boolean(row.get('native_route_verified'))is True,'native-only route proof'),(boolean(row.get('original_audit_verified'))is True,'independent original audit proof'),(p.get('correction_evolution')==f['evolution'],'correction evolution'),
   (p.get('globalization')==f['globalization'],'globalization'),(p.get('guard_policy')==pre['invariants']['guard_policy'],'common safeguard policy'),
   (p.get('correction_initialization')==pre['invariants']['correction_initialization'],'shared initialization policy'),
   ((p.get('initial_model_state')=='reached'and type(p.get('initialization_calls'))is int and p['initialization_calls']>0 and isinstance(p.get('initial_inequality_model_sha256'),str)and len(p['initial_inequality_model_sha256'])==64)or(p.get('initial_model_state')=='not_reached'and p.get('initialization_calls')==0 and p.get('initial_inequality_model_sha256')is None and p.get('accepted_refinements')==0),'initial matrix or native pre-model route proof'),
   (isinstance(p.get('invariant_options_sha256'),str)and len(p['invariant_options_sha256'])==64,'invariant option digest'),
   (isinstance(row.get('binary_sha256'),str)and len(row['binary_sha256'])==64,'binary identity'),
   (isinstance(p.get('guard_config_sha256'),str)and len(p['guard_config_sha256'])==64,'native common guard digest'),
   (type(p.get('accepted_refinements'))is int and p['accepted_refinements']>=0,'refinement counter')]
  if pre['invariants'].get('common_representation') is not None:tests.append((p.get('common_representation')==pre['invariants']['common_representation'],'common original-QCQP representation'))
  if study_sha is not None:tests.append((row.get('study_contract_sha256')==study_sha,'static study contract binding'))
  if expected_binaries is not None:tests.append((row.get('binary_sha256')==expected_binaries.get(backend),'registered binary binding'))
  if expected_guards is not None:tests.append((p.get('guard_config_sha256')==expected_guards.get(backend),'qualified native common guard digest'))
  for name in('refinement_attempt_events','refinement_update_events','refinement_rollback_events'):tests.append((type(p.get(name))is int and p[name]>=0,'native '+name))
  tests.append((p.get('accepted_refinements')==p.get('refinement_update_events'),'accepted refinement counter reconciliation'))
  if f['evolution']=='frozen':tests.append((all(p.get(name)==0 for name in('accepted_refinements','refinement_attempt_events','refinement_update_events','refinement_rollback_events'))and p.get('frozen_metric_invariant_verified')is True,'frozen corrections never updated or refined'))
  bad=[name for ok,name in tests if not ok]
  strict=boolean(row.get('strict_original',row.get('strict_qcqp')))
  rr=[finite(row.get(k))for k in('primal','dual','complementarity')]
  if strict is not None and(any(x is None or x<0 for x in rr)or strict!=(max(rr)<=1e-6)):bad.append('original residual decision')
  if boolean(row.get('censored'))is True:bad.append('censored execution')
  if bad:strict=None
  for item in bad:issues.append(dict(line=number,case_id=cid,backend=backend,cell=cell,issue=item))
  idx[key]=dict(row,strict=strict)
  if row.get('run_id'):run_ids.add(row['run_id'])
  if p.get('initial_model_state')=='not_reached':initial[cid,backend][cell]='not_reached'
  elif p.get('initial_inequality_model_sha256'):initial[cid,backend][cell]=p['initial_inequality_model_sha256']
  if row.get('binary_sha256'):binaries[backend].add(row['binary_sha256'])
 if len(run_ids)>1:raise ValueError('Factorial cells from different experiments cannot be pooled')
 if any(len(v)>1 for v in binaries.values()):raise ValueError('Multiple binaries within a backend block')
 if pre.get('production',{}).get('single_binary_for_both_backends') and len(set().union(*binaries.values()))>1:raise ValueError('Both backend blocks must use the same preregistered binary')
 # Hashes are checked within backend blocks. Compiler/backend float effects
 # are never disguised as a proof that the two builds are bit-identical.
 for(cid,b),values in initial.items():
  for g in ('r3','hs'):
   paired=[v for cell,v in values.items()if FACTORS[cell]['globalization']==g]
   if len(set(paired))>1:raise ValueError('Initial corrections differ within a fixed-globalization refinement pair: '+cid+' '+b+' '+g)
 # Invariant option digest must match every cell of a backend block; factors
 # and observation-only output paths are excluded by the native digest schema.
 for b in pre['backends']:
  values={r.get('factorial_proof',{}).get('invariant_options_sha256')for(c,bb,cell),r in idx.items()if bb==b}
  values.discard(None)
  if len(values)>1:raise ValueError('Unregistered settings differ within backend block '+b)
  guards={r.get('factorial_proof',{}).get('guard_config_sha256')for(c,bb,cell),r in idx.items()if bb==b}
  guards.discard(None)
  if len(guards)>1:raise ValueError('Common safeguard settings differ within backend block '+b)
 return idx,issues,sorted(run_ids)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preregistration',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 seal=load(a.preregistration)
 if seal.get('registration_status')!='sealed_before_production':raise ValueError('A draft protocol cannot authorize confirmatory analysis; seal after source/qualification gates and before production')
 study_path=a.preregistration.parent/seal['study_contract_file'];study_sha=sha(study_path)
 if study_sha!=seal.get('study_contract_sha256'):raise ValueError('Static study contract changed after qualification')
 pre=load(study_path);bindings=seal.get('bindings')or{}
 for name in('source_tree_sha256','source_separation_audit_sha256','qualification50_receipt_sha256','direction_trace_schema_sha256','common_guard_vector_sha256'):
  if not isinstance(bindings.get(name),str)or len(bindings[name])!=64:raise ValueError('Unsealed or absent production binding: '+name)
 binary_bindings=bindings.get('binary_sha256_by_backend')or{}
 if any(not isinstance(binary_bindings.get(b),str)or len(binary_bindings[b])!=64 for b in pre['backends']):raise ValueError('Absent registered backend binary')
 cs=pre['inputs']['case_bindings'];assert len(cs)==1834 and Counter(c['regime']for c in cs)==Counter(dict(I=1037,M=797))
 if tuple(pre['backends'])!=BACKENDS:raise ValueError('Both preregistered backend blocks are required')
 idx,issues,run_ids=validate(records(a.observations),pre,sha(a.preregistration),study_sha,binary_bindings,seal.get('qualified_guard_config_sha256_by_backend'));result,secondary_rows=summarize(cs,idx,pre['backends'],pre['primary']['minimum_net_certificates_per_backend'])
 out=a.out;out.mkdir(parents=True,exist_ok=True)
 report=dict(schema='adaptive-refinement-blocked-factorial-analysis-v1',preregistration_sha256=sha(a.preregistration),observations_sha256=sha(a.observations),study_contract_sha256=study_sha,run_ids=run_ids,
  records_received=len(idx),planned_tasks=len(cs)*4*len(pre['backends']),missing_tasks=len(cs)*4*len(pre['backends'])-len(idx),verification_issues=issues,optimizer_invocations=0,new_endpoint_audits=0,**result)
 (out/'ANALYSE_RAFFINEMENT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');csvwrite(out/'COMPARAISONS_SECONDAIRES.csv',secondary_rows)
 print(json.dumps(dict(overall_decision=report['overall_decision'],backends=report['backend_primary_decisions'],received=report['records_received'],missing=report['missing_tasks'],verification_issues=len(issues)),indent=2))
if __name__=='__main__':main()
