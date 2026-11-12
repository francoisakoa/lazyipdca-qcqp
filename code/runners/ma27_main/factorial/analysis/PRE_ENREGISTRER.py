#!/usr/bin/env python3
"""Fix corpus, four cells, decisions and a 50-case parity sample before new results."""
import argparse,datetime,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from ANALYSER_FACTORIEL import CELLS,FACTORS,CONTRASTS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise SystemExit('Preregistration already exists; it is never overwritten.')
 with (gzip.open(a.contract,'rt') if a.contract.suffix=='.gz' else a.contract.open()) as f:d=json.load(f)
 cases=[c for c in d['cases'] if c['regime'] in ('I','M')]
 if Counter(c['regime'] for c in cases)!=Counter({'I':1037,'M':797}):raise ValueError('Expected complete1037+797 frozen cases')
 seed=20261108
 # Selection uses only structural metadata and fixed hashes, never outcomes.
 sample=[]
 for regime in ('I','M'):
  groups=defaultdict(list)
  for c in cases:
   if c['regime']==regime:groups[c['dataset']].append(c)
  ordered={}
  for key,rows in groups.items():
   rows.sort(key=lambda c:(int(c.get('dimensions',{}).get('n',0)),c['case_id']))
   # Interleave low/median/high dimensions, with deterministic hash tie breaks.
   bins=[rows[:len(rows)//3],rows[len(rows)//3:2*len(rows)//3],rows[2*len(rows)//3:]]
   for b in bins:b.sort(key=lambda c:hashlib.sha256((str(seed)+'|'+c['case_id']).encode()).hexdigest())
   merged=[]
   while any(bins):
    for b in bins:
     if b:merged.append(b.pop(0))
   ordered[key]=merged
  keys=sorted(ordered,key=lambda k:hashlib.sha256((str(seed)+'|'+regime+'|'+k).encode()).hexdigest())
  chosen=[]
  while len(chosen)<25:
   for key in keys:
    if ordered[key] and len(chosen)<25:chosen.append(ordered[key].pop(0))
  sample.extend(chosen)
 obj={'schema':'native-factorial-preregistration-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'inputs':{'contract_file_sha256':sha(a.contract),'cases':1834,'regimes':{'I':1037,'M':797},
   'case_bindings':[{'case_id':c['case_id'],'input_sha256':c['input_sha256'],'start_sha256':c['start_binding']['start_sha256']} for c in sorted(cases,key=lambda c:c['case_id'])]},
  'cells':{k:{'inequality_corrections':FACTORS[k][0],'controller_package':FACTORS[k][1],
     'objective':'exact_original_objective','signed_original_equalities':True} for k in CELLS},
  'controller_scope':'Published R3 versus H/S step-acceptance and safeguard package; both use study_controller=r3. Certified inner completion is excluded.',
  'correction_scope':'Inequality correction construction, reuse/refinement and necessary model representation/symbolic assembly. No objective correction in any cell.',
  'resources':{'seconds_per_solve':7200,'max_outer':5000,'shared_native_restoration_iterations':600,
    'numerical_threads':1,'concurrent_slots':22,'cpus':list(range(22)),'raw_frozen_start':True,'external_endpoint_start':False},
  'planned_native_runs':7336,'all_four_cells_contemporary':True,
  'separate_Ipopt_default_reference':{'cases':4078,'backends':['ma27','mumps'],'runs':8156,
    'all_four_regimes':True,'same_22_slot_resource_pool_or_sequential_execution':True},
  'primary':{'scope':'I+M pooled1834 cases','comparison':'H0 minus X1',
    'minimum_net_certificates':20,'known_discordance_wins_to_losses_minimum':2,
    'largest_positive_dataset_net_contribution_over_total_net_gain_maximum':.5,
    'declared_family_macro_delta_strictly_positive':True,
    'all_four_conditions_required':True,'indeterminate':'unknown, never numerical failure',
    'robust_positive_rule':'Every condition must hold for every binary completion consistent with observed endpoints.',
    'otherwise':'Negative if completely observed or an explicit bound rules out the rule; inconclusive if unknown outcomes can change the decision.'},
  'effects':CONTRASTS,'effect_unit':'Average certificate-count difference across the other factor, plus percentage points',
  'secondary':{'timing':'Common strict pairs with BOTH process clocks>1s; no startup subtraction',
    'objective_normalization':'(fA-fB)/max(1,abs(fA),abs(fB))','objective_thresholds':[.001,.01],
    'accepted_origins':'Final commit provenance. Pure majorant, original curvature, mixed SOC/retraction and other origins reported separately.',
    'composition':'Dataset and declared family partitions remain separate. All planned denominators include unknowns.'},
  'parity_gate':{'cases':50,'I':25,'M':25,'selection_seed':seed,'selection_uses_outcomes':False,
    'sample':[{'case_id':c['case_id'],'regime':c['regime'],'dataset':c['dataset'],'n':c['dimensions']['n'],
               'input_sha256':c['input_sha256'],'start_sha256':c['start_binding']['start_sha256']} for c in sample],
    'required':'Original R3 and original complete HH each reproduced against unchanged reference trajectories. HH objective hybrid is validation-only, distinct from H0 exact-objective cell.',
    'censored':'A censoring-dependent mismatch is unresolved, never called a numerical regression or silently accepted.'},
  'not_a_conclusion':'No factorial outcomes have been observed by this preregistration generator.'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'path':str(a.out),'sha256':sha(a.out),'native_runs':7336,'Ipopt_default_runs':8156,'parity_sample':50}))
if __name__=='__main__':main()
