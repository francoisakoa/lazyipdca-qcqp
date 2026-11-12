#!/usr/bin/env python3
"""Paired certificate coverage and nominal-budget/profile binding for H/S edges.

This consumes recorded strict certificates, not new solves. The two edges of
interest are HS->HH and SH->HH, with source-code isolation audited separately.
"""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_quartet_objectives import METHODS,match,strict,stats

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compact',required=True,type=Path);p.add_argument('--ma27-plan',required=True,type=Path);p.add_argument('--mumps-context',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 contexts={};budget_evidence={}
 for b,path in [('ma27',a.ma27_plan),('mumps',a.mumps_context)]:
  d=json.loads(path.read_text());h=d.get('plan_header',d);prof=h['profiles']['profiles'];norm=[]
  budget={k:h[k] for k in ['seconds','max_outer','shared_restoration','tolerance','threads']}
  assert budget=={'seconds':7200,'max_outer':5000,'shared_restoration':600,'tolerance':1e-6,'threads':1}
  for m in METHODS:
   command=prof[m];n=command.copy();i=n.index('--convexification-study');assert n[i+1]==m;n[i+1]='MODE';norm.append(n)
   assert command[command.index('--time-limit')+1]=='{seconds}'
   assert command[command.index('--max-outer')+1]=='{max_outer}'
   assert command[command.index('--restoration-iters')+1]=='600'
   assert command[command.index('--shared-restoration-budget')+1]=='1'
  assert all(x==norm[0] for x in norm)
  contexts[b]=prof;budget_evidence[b]={'path':str(path),'sha256':sha(path),'nominal_limits':budget,'normalized_vectors_equal_except_mode':True,'profile_sha256':{m:hashlib.sha256(json.dumps(prof[m],sort_keys=True,separators=(',',':')).encode()).hexdigest() for m in METHODS}}
 with gzip.open(a.compact,'rt') as f:rows=[json.loads(l) for l in f]
 lookup={}
 for r in rows:
  if r.get('method') not in METHODS:continue
  b=r['config'].split('_')[-1]
  assert r['profile_sha256']==budget_evidence[b]['profile_sha256'][r['method']]
  lookup[(b,r['method'],r['case_id'])]=r
 summary=[];by_collection=[];discordant=[];objective_edges=[]
 edges=[('HS','HH'),('SH','HH')]
 scopes={'ASU':lambda r:r['dataset']=='beam_structural','M_other':lambda r:r['regime']=='M' and r['dataset']!='beam_structural','M':lambda r:r['regime']=='M','I':lambda r:r['regime']=='I','I_union_M':lambda r:r['regime'] in ['I','M'],'E':lambda r:r['regime']=='E','Q':lambda r:r['regime']=='Q','All':lambda r:True}
 for b in ['ma27','mumps']:
  ids=sorted(cid for bb,m,cid in lookup if bb==b and m=='HH')
  for ma,mb in edges:
   casepairs=[]
   for cid in ids:
    ra,rb=lookup[(b,ma,cid)],lookup[(b,mb,cid)];match([ra,rb]);assert ra['binary_sha256']==rb['binary_sha256'];assert ra.get('source_original_observations_sha256')==rb.get('source_original_observations_sha256')
    sa,sb=strict(ra),strict(rb);cat='both' if sa and sb else 'A_only' if sa else 'B_only' if sb else 'neither';casepairs.append((ra,rb,cat))
    if cat in ['A_only','B_only']:discordant.append({'backend':b,'method_A':ma,'method_B':mb,'case_id':cid,'dataset':ra['dataset'],'regime':ra['regime'],'certificate_category':cat,'input_sha256':ra['input_sha256'],'start_sha256':ra['start_sha256'],'binary_sha256':ra['binary_sha256']})
   for scope,pred in scopes.items():
    selected=[x for x in casepairs if pred(x[0])];counts=collections.Counter(x[2] for x in selected)
    summary.append({'backend':b,'method_A':ma,'method_B':mb,'scope':scope,'N':len(selected),'both':counts['both'],'A_only':counts['A_only'],'B_only':counts['B_only'],'neither':counts['neither'],'B_minus_A_certificates':counts['B_only']-counts['A_only']})
    common=[x for x in selected if x[2]=='both'];vals=[(ra['objective']-rb['objective'])/max(1,abs(ra['objective']),abs(rb['objective'])) for ra,rb,cat in common]
    objective_edges.append({'backend':b,'method_A':ma,'method_B':mb,'scope':scope,**stats(vals)})
   grouped=collections.defaultdict(list)
   for x in casepairs:grouped[(x[0]['regime'],x[0]['dataset'])].append(x)
   for (reg,ds),selected in sorted(grouped.items()):
    co=collections.Counter(x[2] for x in selected)
    by_collection.append({'backend':b,'method_A':ma,'method_B':mb,'regime':reg,'dataset':ds,'N':len(selected),'both':co['both'],'A_only':co['A_only'],'B_only':co['B_only'],'neither':co['neither'],'B_minus_A_certificates':co['B_only']-co['A_only']})
 csvout(a.out/'COVERAGE_EDGE_COUNTS.csv',summary);csvout(a.out/'COVERAGE_EDGE_BY_COLLECTION.csv',by_collection);csvout(a.out/'COVERAGE_EDGE_DISCORDANT_CASES.csv',discordant)
 (a.out/'OBJECTIVE_CLEAN_EDGE_SUMMARIES.json').write_text(json.dumps(objective_edges,indent=2)+'\n')
 (a.out/'BUDGET_AND_PROFILE_BINDING.json').write_text(json.dumps({'source_compact_sha256':sha(a.compact),'profile_sources':budget_evidence,'projection_rows_bound_to_profile':len(lookup),'limits_scope':'Nominal published limits and profiles are verified; no new runtime endpoint audit is performed.','source_code_scope':'The separate source audit is required to establish which edges also share preprocessing.'},indent=2)+'\n')
 lines=[r'\begin{tabular}{@{}lllrrrrr@{}}',r'\toprule',r'Linear solver & $A/B$ & Scope & $N$ & Both & $A$ only & $B$ only & Neither\\',r'\midrule']
 for row in summary:
  if row['scope'] not in ['I','M','ASU']:continue
  lines.append(f"{row['backend'].upper()} & {row['method_A']}/{row['method_B']} & {row['scope']} & {row['N']} & {row['both']} & {row['A_only']} & {row['B_only']} & {row['neither']}"+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}'])
 (a.out/'quartet_clean_edge_coverage_SI.tex').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'summaries':len(summary),'collection_strata':len(by_collection),'discordant_records':len(discordant),'bound_rows':len(lookup)},indent=2))
 for r in summary:
  if r['scope'] in ['ASU','M_other','I','M','I_union_M']:print(r)
if __name__=='__main__':main()
