#!/usr/bin/env python3
"""Join completed original-certified objectives to published reference values.
No solver invocation, endpoint audit, or classification change. References are
published feasible-point values, not certified global optima. Source comparability
is based on archived model identity/conversion contracts, as recorded below.
"""
from pathlib import Path
import argparse,csv,gzip,json,hashlib,math
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(p,rows):
 if not rows:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 here=Path(__file__).resolve().parent
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--compact',type=Path,default=here.parent/'OBSERVATIONS_16_CONFIGS.jsonl.gz',help='classified compact observations; no solver is invoked')
 a.add_argument('--metadata',type=Path,default=here/'data/metadata',help='portable archived conversion subsets')
 a.add_argument('--reference-json',type=Path,default=here/'data/OFFICIAL_QPLIB_SOLUTION_VALUES_COMPLETE.json',help='official QPLIB solution-value snapshot')
 a.add_argument('--facility-reference-json',type=Path,default=here/'data/MINLPLIB_REFERENCE_SNAPSHOT.json',help='official MINLPLib facility-point snapshot')
 a.add_argument('--out',type=Path,default=here/'output',help='output directory')
 args=a.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
 for p in [args.compact,args.metadata/'CANDIDATES.json',args.metadata/'PLAN_3103.json',args.reference_json,args.facility_reference_json]:
  if not p.is_file():a.error('Required input missing: '+str(p))
 obs=[json.loads(s) for s in gzip.open(args.compact,'rt')];byconfig={}
 for r in obs:byconfig.setdefault(r['config'],{})[r['case_id']]=r
 cat=json.loads((args.metadata/'CANDIDATES.json').read_text())['cases'];qps=[r for r in cat if r['id'].startswith('QPLIB__')]
 plan=json.loads((args.metadata/'PLAN_3103.json').read_text())['cases'];official={r['id']:r for r in json.loads(args.reference_json.read_text())['values']}
 refs=[]
 for r in qps:
  i=r['id'].split('__')[1];cv=r['conversion'];s=r['source_metadata'];v=official[i]
  assert v['status']=='=best=' and v['value'] is not None
  assert abs(v['value']-float(s['solobjvalue']))<1e-8
  assert not cv['integer_relaxation'] and not cv['quadratic_equality_rewritten'] and r['source_is_continuous']
  assert cv['triangular_QPLIB_offdiagonal_to_symmetric_Hessian_factor']==.5
  sign=cv['minimization_sign'];assert sign in [-1,1]
  refs.append({'case_id':r['id'],'dataset':'QPLIB_reference','regime':'I','input_sha256':r['input_sha256'],'reference_kind':'published_best_point_not_proven_global_optimum','reference_value_source':v['value'],'source_objective_sense':s['objsense'],'minimization_sign':sign,'objective_scale':sign,'objective_offset':0,'reference_value_frozen':sign*v['value'],'reference_high_precision_frozen':sign*v['value'],'published_primal_feasibility':s.get('solinfeasibility'),'published_source':s.get('solsource'),'reference_url':'https://qplib.zib.de/qplib.solu','model_source_url':r['source_url'],'model_source_sha256':r['source_sha256'],'identity_scope':'archived continuous direct QPLIB conversion; original objective sign retained; no new source re-export audit','global_optimum_certified':False})
 # Official MINLPLib pages, retrieved 2026-11-09; values with page-reported
 # infeasibility <=1e-8 are the library's primal-bound group. Keep the most
 # accurate <=1e-9 subgroup separately to expose endpoint-precision sensitivity.
 emfl=json.loads(args.facility_reference_json.read_text())['instances']
 for i,e in emfl.items():
  cid='MINLPLIB__'+i;r=plan[cid];sr=r['source_record'];p=sr['export_parity']
  assert sr['mathematical_scope']=='public_continuous_original' and sr['direct_qcqp'] and not sr['n_integer'] and not sr['n_binary'] and p['performed']
  assert p['source_value_relative_error_max']<=p['threshold'] and p['source_lagrangian_gradient_relative_error_max']<=p['threshold']
  v,feas=min(e['points']);hp=min(x for x,tol in e['points'] if tol<=1e-9)
  refs.append({'case_id':cid,'dataset':'MINLPLIB','regime':'I','input_sha256':r['input_sha256'],'reference_kind':'published_best_primal_point_reported_infeasibility_le_1e-8','reference_value_source':v,'source_objective_sense':'min','minimization_sign':1,'objective_scale':1,'objective_offset':0,'reference_value_frozen':v,'reference_high_precision_frozen':hp,'published_primal_feasibility':feas,'published_source':'MINLPLib primal-bound group; numeric dual values not treated as rigorous proof','reference_url':sr['primary_instance_url'],'model_source_url':sr['primary_instance_url'],'model_source_sha256':sr['source_sha256']['osil'],'identity_scope':'archived direct continuous public model; matching input hash; recorded independent OSiL/QCQP value/gradient parity','global_optimum_certified':False})
 refs.sort(key=lambda r:(r['dataset'],r['case_id']));write_csv(out/'REFERENCE_VALUES_ADMISSIBLE.csv',refs)
 lookup={r['case_id']:r for r in refs};joined=[];inventory=[]
 methods=['R3','HH','HS','SH','SS','SS_legacy'];configs=[m+'_'+b for b in ['ma27','mumps'] for m in methods]+['ipopt_'+b+'_'+s+'_aligned'for s in ['monotone','adaptive']for b in ['ma27','mumps']]
 assert set(configs)==set(byconfig) and len(configs)==16
 for ref in refs:
  starts={byconfig[c][ref['case_id']]['start_sha256'] for c in configs}
  assert len(starts)==1,('Unmatched frozen start',ref['case_id'])
 for c in configs:
  for ref in refs:
   r=byconfig[c][ref['case_id']];assert r['input_sha256']==ref['input_sha256']
   valid=r['classification']=='strict' and isinstance(r['objective'],(int,float)) and math.isfinite(r['objective'])
   f=r['objective'] if valid else None;v=ref['reference_value_frozen'];hp=ref['reference_high_precision_frozen']
   joined.append({'case_id':r['case_id'],'dataset':r['dataset'],'regime':r['regime'],'config':c,'classification':r['classification'],'original_objective':f,'reference_value':v,'signed_relative_reference_gap':(f-v)/max(1,abs(v)) if valid else None,'high_precision_reference_value':hp,'signed_relative_high_precision_gap':(f-hp)/max(1,abs(hp)) if valid else None,'reference_kind':ref['reference_kind'],'input_sha256':r['input_sha256'],'start_sha256':r['start_sha256'],'reference_url':ref['reference_url']})
 summaries=[]
 for scope in ['QPLIB_reference','MINLPLIB','all_53']:
  for c in configs:
   rs=[r for r in joined if r['config']==c and (scope=='all_53' or r['dataset']==scope)];good=[r for r in rs if r['signed_relative_reference_gap'] is not None];g=np.array([r['signed_relative_reference_gap'] for r in good]);h=np.array([r['signed_relative_high_precision_gap'] for r in good])
   q=np.quantile(g,[.25,.5,.75],method='linear') if len(g) else [None]*3
   summaries.append({'scope':scope,'config':c,'planned_reference_cases':len(rs),'certified_cases':len(g),'q25_percent':100*q[0] if len(g) else None,'median_percent':100*q[1] if len(g) else None,'q75_percent':100*q[2] if len(g) else None,'maximum_percent':100*max(g) if len(g) else None,'within_relative_1e-3':int(np.count_nonzero(abs(g)<=1e-3)),'within_relative_1e-2':int(np.count_nonzero(abs(g)<=1e-2)),'below_reference_by_more_than_1e-3':int(np.count_nonzero(g< -1e-3)),'high_precision_reference_median_percent':float(100*np.median(h)) if len(h) else None})
 write_csv(out/'REFERENCE_OBJECTIVE_GAPS_ALL_CONFIGS.csv',joined);write_csv(out/'REFERENCE_OBJECTIVE_SUMMARY_ALL_CONFIGS.csv',summaries)
 # Common cohort supplies an equal-population descriptive comparison.
 allfour=['R3_ma27','HH_ma27','ipopt_ma27_monotone_aligned','ipopt_mumps_monotone_aligned']
 common=[ref['case_id'] for ref in refs if all(byconfig[c][ref['case_id']]['classification']=='strict' for c in allfour)]
 matched=[]
 for c in configs:
  rs=[r for r in joined if r['config']==c and r['case_id'] in common and r['signed_relative_reference_gap'] is not None]
  vals=[r['signed_relative_reference_gap'] for r in rs]
  matched.append({'config':c,'shared_reference_cohort_size':len(common),'certified_in_shared_cohort':len(vals),'median_percent':float(np.median(vals)*100) if vals else None,'within_relative_1e-3':sum(abs(v)<=1e-3 for v in vals),'within_relative_1e-2':sum(abs(v)<=1e-2 for v in vals)})
 write_csv(out/'REFERENCE_OBJECTIVE_COMMON_COHORT.csv',matched)
 # Every I/M case in the requested source libraries is assigned a reason.
 baseline=byconfig['R3_ma27']
 for cid,r in baseline.items():
  if r['regime'] not in ['I','M'] or r['dataset'] not in ['QPLIB_reference','MINLPLIB','pooling_MINLPLIB','PGLIB','beam_structural']:continue
  if cid in lookup:status='admitted_by_archived_model_identity_contract';reason='published primal reference value, source convention/identity verified against archived conversion metadata'
  elif r['dataset']=='PGLIB':status='not_admitted';reason='source network hash retained, but exact source-version baseline bound and frozen AC-QCQP objective/formulation mapping not available; no identifier-only comparison'
  elif r['dataset']=='pooling_MINLPLIB':status='not_admitted';reason='published model/normalization/reformulation identity not established from retained reduced metadata'
  else:status='not_available';reason='constructed ASU QCQP collection; no published best-known global value or certified lower bound retrieved for its exact frozen formulation'
  inventory.append({'case_id':cid,'dataset':r['dataset'],'regime':r['regime'],'input_sha256':r['input_sha256'],'reference_status':status,'reason':reason})
 write_csv(out/'REFERENCE_COVERAGE_AND_BLOCKERS.csv',inventory)
 receipt={'schema':'published-reference-objective-r14-v1','retrieved_date':'2026-11-09','admitted_I_reference_cases':53,'admitted_M_reference_cases':0,'source_libraries':{'QPLIB':49,'MINLPLib_EMFL':4},'published_reference_not_global_optimum':True,'new_solvers':0,'new_endpoint_audits':0,'new_classifications':0,'source_contracts_not_new_full_source_reexport_audit':True,'all_four_principal_comparators_common_cases':len(common),'common_configs':allfour,'aligned_Ipopt_results':'all four newly supplied aligned configurations are analyzed; primary common cohort uses the two monotone configurations','inputs':{str(p):sha(p) for p in [args.compact,args.metadata/'CANDIDATES.json',args.metadata/'PLAN_3103.json',args.reference_json,args.facility_reference_json]},'summaries':summaries}
 (out/'REFERENCE_VALUE_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'admitted':53,'common':len(common),'configs':configs,'summaries_first':summaries[:4]},indent=2))
if __name__=='__main__':main()
