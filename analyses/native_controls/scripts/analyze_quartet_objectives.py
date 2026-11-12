#!/usr/bin/env python3
"""Direct paired H/S objective analysis from already audited campaign projections.

No optimizer or endpoint evaluator is invoked. The campaign's recorded strict
classification is retained, checked against its three exported residuals, and
pairing verifies frozen-QCQP and raw-start hashes. Four-way factorial contrasts
use the same per-instance scale across all four policies. Quartiles are type7.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,math,io
METHODS=('HH','HS','SH','SS')
PAIRS=(('SS','SH'),('SS','HS'),('SS','HH'),('HH','SH'),('HH','HS'),('SH','HS'))
THRESHOLDS=(1e-3,1e-2)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def quantile(x,p):
 if not x:return None
 y=sorted(x);h=(len(y)-1)*p;i=int(h);j=min(i+1,len(y)-1)
 return y[i]+(h-i)*(y[j]-y[i])
def stats(x):
 return {'N':len(x),'mean':sum(x)/len(x) if x else None,'q1':quantile(x,.25),'median':quantile(x,.5),'q3':quantile(x,.75),'min':min(x) if x else None,'max':max(x) if x else None,'exact_zero_count':sum(v==0 for v in x),'negative_count':sum(v<0 for v in x),'positive_count':sum(v>0 for v in x),'thresholds':{str(t):{'lower':sum(v < -t for v in x),'equivalent':sum(-t<=v<=t for v in x),'higher':sum(v>t for v in x)} for t in THRESHOLDS}}
def scope(r):
 out=['All',r['regime']]
 if r['regime']=='M':out.append('ASU' if r['dataset']=='beam_structural' else 'M_other')
 return out
def match(rows):
 for field in ['case_id','dataset','regime','input_sha256','start_sha256']:
  values=[r.get(field) for r in rows]
  if any(v in [None,''] for v in values) or len(set(values))!=1:raise ValueError(('Pair identity mismatch',field,values))
 for field in ['raw_x0_line_sha256','policy']:
  values=[r.get('start_binding',{}).get(field) for r in rows]
  if any(v in [None,''] for v in values) or len(set(values))!=1:raise ValueError(('Raw start mismatch',field,values))
 values=[r.get('start_binding',{}).get('embedded_start_sha256') for r in rows]
 if len(set(values))!=1:raise ValueError(('Embedded start mismatch',values))
 if any(r['start_binding'].get('policy')!='raw_frozen_x0' for r in rows):raise ValueError('Nonraw start')
 if any(not r['start_binding'].get('input_includes_raw_x0') for r in rows):raise ValueError('Raw x0 not included')
def strict(r):
 if r['classification']!='strict':return False
 for k in ['primal','dual','complementarity']:
  v=r.get(k)
  if not isinstance(v,(float,int)) or not math.isfinite(v) or v>1e-6:raise ValueError(('Recorded strict residual contradiction',k,r['config'],r['case_id'],v))
 f=r.get('objective')
 if not isinstance(f,(float,int)) or not math.isfinite(f):raise ValueError(('Nonfinite objective',r['config'],r['case_id']))
 return True
def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);w.writeheader();w.writerows(rows)
def compact(x):
 if x is None:return '--'
 return f'{100*x:.2f}'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 records=[]
 with gzip.open(a.compact,'rt') as f:
  for line in f:
   r=json.loads(line)
   if r.get('method') in METHODS:records.append(r)
 lookup={}
 for r in records:
  key=(r['config'],r['case_id'])
  if key in lookup:raise ValueError(('Duplicate',key))
  lookup[key]=r
 pair_summaries=[];factorial_summaries=[];pair_rows=[];factorial_rows=[];cohort_counts=[];identity_checks=0
 for backend in ['ma27','mumps']:
  groups={m:{cid:r for (config,cid),r in lookup.items() if config==m+'_'+backend} for m in METHODS}
  ids=set(groups['SS'])
  if len(ids)!=4078 or any(set(g)!=ids for g in groups.values()):raise ValueError('Incomplete quartet plan')
  four_common={}
  for cid in sorted(ids):
   rs=[groups[m][cid] for m in METHODS];match(rs);identity_checks+=1
   if all(strict(r) for r in rs):
    fs={m:groups[m][cid]['objective'] for m in METHODS};den=max(1.,*(abs(v) for v in fs.values()))
    four_common[cid]=(fs,den)
    row={'backend':backend,'case_id':cid,'dataset':rs[0]['dataset'],'regime':rs[0]['regime'],'input_sha256':rs[0]['input_sha256'],'start_sha256':rs[0]['start_sha256'],'raw_x0_line_sha256':rs[0]['start_binding']['raw_x0_line_sha256'],'denominator':den,**{'f_'+m:fs[m] for m in METHODS}}
    row.update({
     'objective_hybrid_gain_at_scalar_inequalities':(fs['SS']-fs['HS'])/den,
     'objective_hybrid_gain_at_hybrid_inequalities':(fs['SH']-fs['HH'])/den,
     'inequality_hybrid_gain_at_scalar_objective':(fs['SS']-fs['SH'])/den,
     'inequality_hybrid_gain_at_hybrid_objective':(fs['HS']-fs['HH'])/den,
     'mean_objective_hybrid_gain':((fs['SS']-fs['HS'])+(fs['SH']-fs['HH']))/(2*den),
     'mean_inequality_hybrid_gain':((fs['SS']-fs['SH'])+(fs['HS']-fs['HH']))/(2*den),
     'joint_gain_SS_to_HH':(fs['SS']-fs['HH'])/den,
     'sum_separate_gains_from_SS':((fs['SS']-fs['HS'])+(fs['SS']-fs['SH']))/den,
     'objective_interaction_HH_minus_HS_minus_SH_plus_SS':(fs['HH']-fs['HS']-fs['SH']+fs['SS'])/den,
     'HH_SH_objective_exact_equal':fs['HH']==fs['SH']})
    factorial_rows.append(row)
  for sc in ['ASU','M_other','M','I','E','Q','All']:
   planned=[cid for cid in sorted(ids) if sc in scope(groups['SS'][cid])]
   common=[cid for cid in planned if cid in four_common]
   cohort_counts.append({'backend':backend,'scope':sc,'planned':len(planned),'HH_strict':sum(strict(groups['HH'][cid]) for cid in planned),'HS_strict':sum(strict(groups['HS'][cid]) for cid in planned),'SH_strict':sum(strict(groups['SH'][cid]) for cid in planned),'SS_strict':sum(strict(groups['SS'][cid]) for cid in planned),'four_way_common':len(common)})
   commonset=set(common)
   rows=[r for r in factorial_rows if r['backend']==backend and r['case_id'] in commonset]
   fields=[k for k in rows[0] if k.startswith(('objective_hybrid','inequality_hybrid','mean_','joint_','sum_','objective_interaction'))] if rows else []
   factorial_summaries.append({'backend':backend,'scope':sc,'N':len(rows),'HH_SH_objective_exact_equal_count':sum(r['HH_SH_objective_exact_equal'] for r in rows),'contrasts':{k:stats([r[k] for r in rows]) for k in fields}})
   for ma,mb in PAIRS:
    for cohort in ['pair_specific','four_way_common']:
     use=[cid for cid in planned if strict(groups[ma][cid]) and strict(groups[mb][cid]) and (cohort=='pair_specific' or cid in four_common)]
     vals=[];equal=0
     for cid in use:
      ra,rb=groups[ma][cid],groups[mb][cid];match([ra,rb])
      fa,fb=ra['objective'],rb['objective'];gap=(fa-fb)/max(1.,abs(fa),abs(fb));vals.append(gap);equal+=fa==fb
      if cohort=='pair_specific':pair_rows.append({'backend':backend,'scope':sc,'method_A':ma,'method_B':mb,'case_id':cid,'dataset':ra['dataset'],'regime':ra['regime'],'f_A':fa,'f_B':fb,'gap':gap,'input_sha256':ra['input_sha256'],'start_sha256':ra['start_sha256'],'raw_x0_line_sha256':ra['start_binding']['raw_x0_line_sha256']})
     pair_summaries.append({'backend':backend,'scope':sc,'method_A':ma,'method_B':mb,'cohort':cohort,'exact_objective_equal_count':equal,**stats(vals)})
 source={'path':str(a.compact),'sha256':sha(a.compact),'native_quartet_projection_rows':len(records),'full_plan_identity_checks':identity_checks,'same_input_raw_start':True,'new_solver_invocations':0,'new_endpoint_audits':0,'classification_changes':0,'certificate_scope':'Uses recorded campaign strict certificates and verifies exported finite primal/dual/complementarity <=1e-6; full endpoints are not present in this projection.','endpoint_identity_scope':'Objective equality is directly testable; identical primal endpoints, iterates or work counters cannot be inferred from this projection.','pair_gap':'(f_A-f_B)/max(1,abs(f_A),abs(f_B)); positive means B lower','quartile_definition':'Hyndman-Fan type 7','factorial_scale':'D_i=max(1,abs(f_HH),abs(f_HS),abs(f_SH),abs(f_SS)) shared across all four policies in a case','factorial_interaction':'(f_HH-f_HS-f_SH+f_SS)/D_i. Positive means joint objective reduction below sum of separate reductions relative to SS.','selection_scope':'All objective summaries condition on recorded strict certificates; four-way common cohorts differ from pair-specific cohorts.','causal_scope':'Descriptive contrasts of complete recorded policies. Source audit identifies an implied-bound presolve difference between SS and HH/HS/SH, so the quartet is not a curvature-only factorial experiment.'}
 unequal=[]
 for backend in ['ma27','mumps']:
  for (config,cid),ra in lookup.items():
   if config!='HH_'+backend or ra['dataset']!='beam_structural' or not strict(ra):continue
   rb=lookup[('SH_'+backend,cid)]
   if not strict(rb) or ra['objective']==rb['objective']:continue
   fa,fb=ra['objective'],rb['objective']
   unequal.append({'backend':backend,'case_id':cid,'f_HH':fa,'f_SH':fb,'absolute_difference':abs(fa-fb),'relative_gap':(fa-fb)/max(1.,abs(fa),abs(fb)),'ulps_around_HH':abs(fa-fb)/math.ulp(fa)})
 (a.out/'HH_SH_OBJECTIVE_EQUALITY.json').write_text(json.dumps({'unequal_cases':unequal,'meaning':'Exported original objective equality only; no primal endpoint or trajectory equality inferred.'},indent=2)+'\n')
 (a.out/'SOURCE_AND_METHODS.json').write_text(json.dumps(source,indent=2)+'\n')
 (a.out/'PAIR_SUMMARIES.json').write_text(json.dumps(pair_summaries,indent=2)+'\n');(a.out/'FACTORIAL_SUMMARIES.json').write_text(json.dumps(factorial_summaries,indent=2)+'\n')
 write_csv(a.out/'COHORT_COUNTS.csv',cohort_counts)
 flat=[]
 for r in pair_summaries:
  z={k:v for k,v in r.items() if k!='thresholds'}
  for th,t in r['thresholds'].items():
   for k,v in t.items():z[f'{th}_{k}']=v
  flat.append(z)
 write_csv(a.out/'PAIR_SUMMARIES.csv',flat);write_csv(a.out/'FACTORIAL_COMMON_CASES.csv',factorial_rows)
 with (a.out/'PAIR_CASES.csv.gz').open('wb') as binary, gzip.GzipFile(filename='',mode='wb',fileobj=binary,mtime=0) as compressed, io.TextIOWrapper(compressed,encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
 # Compact manuscript table, ASU pair-specific; HH/SH relationship plus selection reported separately.
 lines=[r'\begin{tabular}{@{}llrcc@{}}',r'\toprule',r'Linear solver & Direct pair & $N$ & Median $[Q_1,Q_3]$ (\%) & L/E/H\\',r'\midrule']
 for backend in ['ma27','mumps']:
  for ma,mb in [('SS','SH'),('SS','HS'),('SS','HH'),('HH','SH')]:
   r=next(r for r in pair_summaries if r['backend']==backend and r['scope']=='ASU' and r['method_A']==ma and r['method_B']==mb and r['cohort']=='pair_specific');tr=r['thresholds']['0.001']
   lines.append(f"{backend.upper()} & {ma}/{mb} & {r['N']} & {compact(r['median'])} [{compact(r['q1'])},{compact(r['q3'])}] & {tr['lower']}/{tr['equivalent']}/{tr['higher']}"+r'\\')
  if backend=='ma27':lines.append(r'\addlinespace')
 lines.extend([r'\bottomrule',r'\end{tabular}']);(a.out/'quartet_ASU_direct_main.tex').write_text('\n'.join(lines)+'\n')
 lines=[r'\begin{tabular}{@{}llrcr@{}}',r'\toprule',r'Linear solver & Pair & $N$ & Median $[Q_1,Q_3]$ (\%) & Exact equal\\',r'\midrule']
 for backend in ['ma27','mumps']:
  for ma,mb in PAIRS:
   r=next(r for r in pair_summaries if r['backend']==backend and r['scope']=='ASU' and r['method_A']==ma and r['method_B']==mb and r['cohort']=='four_way_common')
   lines.append(f"{backend.upper()} & {ma}/{mb} & {r['N']} & {compact(r['median'])} [{compact(r['q1'])},{compact(r['q3'])}] & {r['exact_objective_equal_count']}"+r'\\')
  if backend=='ma27':lines.append(r'\addlinespace')
 lines.extend([r'\bottomrule',r'\end{tabular}']);(a.out/'quartet_ASU_common_SI.tex').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'pair_summaries':len(pair_summaries),'factorial_summaries':len(factorial_summaries),'identity_checks':identity_checks,'native_quartet_rows':len(records)},indent=2))
if __name__=='__main__':main()
