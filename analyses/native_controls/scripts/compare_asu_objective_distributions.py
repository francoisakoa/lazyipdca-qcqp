#!/usr/bin/env python3
"""Re-pair archived original-certified ASU objectives, without optimizer calls.

Reports pair-specific and five-way common cohorts; preserves recorded strict
classifications and checks their three exported residuals and identity hashes.
Statistics concern attained objectives, not full endpoints, trajectories or a
causal decomposition of coupled policy mechanisms.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,io
METHODS=('R3','HH','HS','SH','SS')
PAIRS=(('R3','SS'),('R3','HH'),('R3','SH'),('R3','HS'),('SS','HH'),('SS','SH'),('SS','HS'),('HS','HH'),('SH','HH'),('HS','SH'))
THRESHOLDS=(1e-12,1e-10,1e-8,1e-6,1e-4,1e-3,1e-2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x,p):
 if not x:return None
 a=sorted(x);t=(len(a)-1)*p;i=int(t);j=min(i+1,len(a)-1);return a[i]+(t-i)*(a[j]-a[i])
def summarize(x):
 n=len(x)
 return {'N':n,'q1':q(x,.25),'median':q(x,.5),'q3':q(x,.75),'mean':math.fsum(x)/n if n else None,'minimum':min(x) if n else None,'maximum':max(x) if n else None,'absolute_q1':q([abs(v) for v in x],.25),'absolute_median':q([abs(v) for v in x],.5),'absolute_q3':q([abs(v) for v in x],.75),'exact_zero':sum(v==0 for v in x),'thresholds':{str(t):{'lower':sum(v < -t for v in x),'equivalent':sum(-t<=v<=t for v in x),'higher':sum(v>t for v in x),'not_equivalent':sum(abs(v)>t for v in x)} for t in THRESHOLDS}}
def strict(r):
 if r['classification']!='strict':return False
 for k in ['primal','dual','complementarity']:
  v=r.get(k)
  if not isinstance(v,(int,float)) or not math.isfinite(v) or v>1e-6:raise ValueError(('Recorded certificate/export inconsistency',r['config'],r['case_id'],k,v))
 if not isinstance(r.get('objective'),(int,float)) or not math.isfinite(r['objective']):raise ValueError('Nonfinite certified objective')
 return True
def identity(rows):
 for key in ['case_id','dataset','regime','input_sha256','start_sha256']:
  v=[r.get(key) for r in rows]
  if any(x in [None,''] for x in v) or len(set(v))!=1:raise ValueError(('Identity mismatch',key,v))
 for key in ['raw_x0_line_sha256','policy']:
  v=[r.get('start_binding',{}).get(key) for r in rows]
  if any(x in [None,''] for x in v) or len(set(v))!=1:raise ValueError(('Raw start mismatch',key,v))
 if any(r['start_binding']['policy']!='raw_frozen_x0' or not r['start_binding'].get('input_includes_raw_x0') for r in rows):raise ValueError('Raw-start contract not verified')
def scopes(r):
 out=['All',r['regime']]
 if r['regime']=='M':out.append('ASU' if r['dataset']=='beam_structural' else 'M_other')
 return out
def csvout(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def pct(v):return '--' if v is None else f'{100*v:.8g}'
def display(v,median=False):
 if v is None:return '--'
 x=100*v
 if median and abs(x)<0.005:return '0.00'
 if x==0:return '0.00'
 if abs(x)>=0.1:return f'{x:.3f}'
 z=f'{x:.6f}'.rstrip('0').rstrip('.')
 return '0.00' if z in ['0','-0'] else z

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compact',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 with gzip.open(a.compact,'rt') as f:allrows=[json.loads(l) for l in f]
 data=[r for r in allrows if r.get('method') in METHODS];lookup={}
 for r in data:
  key=(r['config'],r['case_id'])
  if key in lookup:raise ValueError(('Duplicate observation',key))
  lookup[key]=r
 summaries=[];details=[];coverage=[];identity_checks=0
 for backend in ['ma27','mumps']:
  groups={m:{cid:r for (conf,cid),r in lookup.items() if conf==m+'_'+backend} for m in METHODS};ids=sorted(groups['R3'])
  if len(ids)!=4078 or any(set(g)!=set(ids) for g in groups.values()):raise ValueError('Incomplete five-policy plan')
  five=[]
  for cid in ids:
   rows=[groups[m][cid] for m in METHODS];identity(rows);identity_checks+=1
   if all(strict(r) for r in rows):five.append(cid)
  five=set(five)
  for sc in ['ASU','M_other','M','I','E','Q','All']:
   planned=[cid for cid in ids if sc in scopes(groups['R3'][cid])]
   for ma,mb in PAIRS:
    cat={'both':0,'A_only':0,'B_only':0,'neither':0}
    for cid in planned:
     sa,sb=strict(groups[ma][cid]),strict(groups[mb][cid]);cat['both' if sa and sb else 'A_only' if sa else 'B_only' if sb else 'neither']+=1
    coverage.append({'backend':backend,'scope':sc,'method_A':ma,'method_B':mb,'N_planned':len(planned),**cat})
    for cohort in ['pair_specific','five_way_common']:
     use=[cid for cid in planned if strict(groups[ma][cid]) and strict(groups[mb][cid]) and (cohort=='pair_specific' or cid in five)]
     vals=[];absdiff=[];equal=[];largest=None
     for cid in use:
      ra,rb=groups[ma][cid],groups[mb][cid];identity([ra,rb]);fa,fb=ra['objective'],rb['objective'];den=max(1.,abs(fa),abs(fb));delta=(fa-fb)/den;vals.append(delta);absdiff.append(abs(fa-fb));equal.append(fa==fb)
      row={'backend':backend,'scope':sc,'method_A':ma,'method_B':mb,'cohort':cohort,'case_id':cid,'dataset':ra['dataset'],'regime':ra['regime'],'f_A':fa,'f_B':fb,'normalizer':den,'signed_relative_gap':delta,'absolute_objective_difference':abs(fa-fb),'input_sha256':ra['input_sha256'],'start_sha256':ra['start_sha256'],'raw_x0_line_sha256':ra['start_binding']['raw_x0_line_sha256']}
      if largest is None or abs(delta)>abs(largest['signed_relative_gap']):largest=row
      if sc=='ASU' and cohort=='pair_specific':details.append(row)
     summaries.append({'backend':backend,'scope':sc,'method_A':ma,'method_B':mb,'cohort':cohort,'exact_objective_equal':sum(equal),'maximum_absolute_objective_difference':max(absdiff) if absdiff else None,'largest_relative_difference_case':largest['case_id'] if largest else None,**summarize(vals)})
 for r in summaries:
  assert r['exact_zero']==r['exact_objective_equal']
  assert all(t['lower']+t['equivalent']+t['higher']==r['N'] for t in r['thresholds'].values())
 (a.out/'DISTRIBUTION_SUMMARIES.json').write_text(json.dumps(summaries,indent=2)+'\n')
 flat=[]
 for r in summaries:
  d={k:v for k,v in r.items() if k!='thresholds'}
  for t,count in r['thresholds'].items():
   for k,v in count.items():d[t+'_'+k]=v
  flat.append(d)
 csvout(a.out/'DISTRIBUTION_SUMMARIES.csv',flat);csvout(a.out/'COVERAGE_PAIRED.csv',coverage);csvout(a.out/'ASU_PAIR_CASES.csv',details)
 (a.out/'SOURCE_AND_METHODS.json').write_text(json.dumps({'source_compact':str(a.compact),'source_sha256':sha(a.compact),'total_source_rows':len(allrows),'selected_native_rows':len(data),'full_plan_identity_checks':identity_checks,'methods':METHODS,'gap_definition':'(f_A-f_B)/max(1,abs(f_A),abs(f_B)); positive means B attains a lower objective','quartiles':'type7','near_zero_thresholds_fraction':THRESHOLDS,'scope':'Uses archived strict classification and exported residuals; no new endpoint certification. Pairing verifies original input/raw-start hashes. Five-way common cohort is distinct from pair-specific cohorts.','no_new_solver_invocations':True,'no_new_endpoint_audits':True,'no_classification_changes':True,'causal_limit':'R3 or SS comparisons mix several policy choices. Source audit is required to isolate a correction policy; equality of f does not establish equality of x, trajectory or model corrections.'},indent=2)+'\n')
 lines=[r'\begin{tabular}{@{}llrccccc@{}}',r'\toprule',r'Solver & $A/B$ & $N$ & Med. $\delta_f$ & Med. $|\delta_f|$ & $[Q_1,Q_3]$ & L/E/H $10^{-3}$ & L/E/H $10^{-2}$\\',r'\midrule']
 for backend in ['ma27','mumps']:
  for pair in [('R3','SS'),('SS','HH'),('HS','HH'),('SH','HH')]:
   r=next(r for r in summaries if r['backend']==backend and r['scope']=='ASU' and (r['method_A'],r['method_B'])==pair and r['cohort']=='pair_specific')
   tc=[]
   for t in ['0.001','0.01']:
    z=r['thresholds'][t];tc.append(f"{z['lower']}/{z['equivalent']}/{z['higher']}")
   lines.append(f"{backend.upper()} & {pair[0]}/{pair[1]} & {r['N']} & {display(r['median'],True)} & {display(r['absolute_median'])} & [{display(r['q1'])},{display(r['q3'])}] & {tc[0]} & {tc[1]}"+r'\\')
  if backend=='ma27':lines.append(r'\addlinespace')
 lines += [r'\bottomrule',r'\end{tabular}'];(a.out/'asu_distribution_table_R13.tex').write_text('\n'.join(lines)+'\n')
 # Publication variant: quartiles remain in the full table and source data.
 compact=[]
 for line in lines:
  if line.startswith(r'\begin{tabular}'):
   compact.append(r'\begin{tabular}{@{}llrcccc@{}}')
  elif ' & ' in line:
   cells=line.split(' & ')
   compact.append(' & '.join(cells[:5]+cells[6:]))
  else:compact.append(line)
 (a.out/'asu_distribution_compact_R13.tex').write_text('\n'.join(compact)+'\n')
 print(json.dumps({'summaries':len(summaries),'identity_checks':identity_checks,'source_rows':len(allrows),'new_solver_invocations':0},indent=2))
 for r in summaries:
  if r['scope']=='ASU' and r['cohort']=='pair_specific' and (r['method_A'],r['method_B']) in [('R3','SS'),('HS','HH'),('SH','HH')]:print(json.dumps(r))
if __name__=='__main__':main()
