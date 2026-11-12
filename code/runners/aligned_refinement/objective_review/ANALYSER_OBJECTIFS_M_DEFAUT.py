#!/usr/bin/env python3
"""Descriptive objective diagnostics on the completed default-setting archive."""
import argparse,csv,gzip,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path
LAZY=tuple(f'{p}_{b}'for b in('ma27','mumps')for p in('R3','HH','HS','SH','SS','SS_legacy'))
IPOPT=('ipopt_ma27_default','ipopt_mumps_default')
CONFIGS=LAZY+IPOPT
SCOPES=('M_all','M_ASU','M_outside_ASU')
def finite(x):return type(x)in(int,float)and math.isfinite(x)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def quantile(x,p):
 x=sorted(x)
 if not x:return None
 at=(len(x)-1)*p;i=int(at)
 return x[i]if i+1==len(x)else x[i]*(1-(at-i))+x[i+1]*(at-i)
def dist(x):return dict(N=len(x),q25=quantile(x,.25),median=quantile(x,.5),q75=quantile(x,.75))
def start(r):return r.get('start_sha256')or(r.get('start_binding')or{}).get('start_sha256')
def records(path):
 with(gzip.open(path,'rt')if str(path).endswith('.gz')else Path(path).open())as f:
  for line in f:
   if line.strip():yield json.loads(line)
def parse(rows,require_full=True):
 idx={};cases={};proofs=Counter()
 for r in rows:
  cfg=r.get('config');cid=r.get('case_id')
  if cfg not in CONFIGS:raise ValueError('Unexpected archived configuration '+str(cfg))
  if not cid or(cid,cfg)in idx:raise ValueError('Absent case or duplicate archived observation')
  identity=dict(regime=r.get('regime'),dataset=r.get('dataset'),family=r.get('family'),input_sha256=r.get('input_sha256'),start_sha256=start(r))
  if not identity['input_sha256']or not identity['start_sha256']:raise ValueError('Frozen data/start hashes absent: '+cid)
  if cid in cases and cases[cid]!=identity:raise ValueError('Unmatched frozen data/start or structural metadata: '+cid)
  cases[cid]=identity
  vals=[r.get(k)for k in('primal','dual','complementarity')]
  if r.get('classification')in('strict','nonstrict'):
   if not all(finite(v)and v>=0 for v in vals)or(r['classification']=='strict')!=all(v<=1e-6 for v in vals):raise ValueError('Inconsistent original certificate: '+cid+'/'+cfg)
  idx[cid,cfg]=r;proofs[r.get('proof_level','unspecified')]+=1
 if require_full:
  if len(cases)!=4078 or len(idx)!=4078*14:raise ValueError('Full 4078 by14 archived projection required')
  if set(idx)!={(c,k)for c in cases for k in CONFIGS}:raise ValueError('Incomplete archived matrix')
  if Counter(v['regime']for v in cases.values())!=Counter(I=1037,M=797,E=1704,Q=540):raise ValueError('Unexpected regime coverage')
 return idx,cases,dict(proofs)
def analyse(idx,cases):
 cohorts={scope:[]for scope in SCOPES}
 for cid,c in cases.items():
  if c['regime']!='M':continue
  cohorts['M_all'].append(cid);cohorts['M_ASU'if c['dataset']=='beam_structural'else'M_outside_ASU'].append(cid)
 summaries=[];pairrows=[];residuals=[]
 for scope,ids in cohorts.items():
  for left in LAZY:
   for right in IPOPT:
    common=[]
    for cid in ids:
     a,b=idx[cid,left],idx[cid,right]
     if a['classification']!='strict'or b['classification']!='strict':continue
     if not finite(a.get('objective'))or not finite(b.get('objective')):raise ValueError('Objective missing on common strict pair '+cid)
     f,g=a['objective'],b['objective'];gap=(f-g)/max(1,abs(f),abs(g));common.append((cid,a,b,gap))
     pairrows.append(dict(scope=scope,case_id=cid,dataset=cases[cid]['dataset'],left=left,right=right,objective_left=f,objective_right=g,signed_relative_gap=gap,absolute_relative_gap=abs(gap),input_sha256=cases[cid]['input_sha256'],start_sha256=cases[cid]['start_sha256']))
    gaps=[p[3]for p in common];r=dict(scope=scope,left=left,right=right,planned_cases=len(ids),left_strict=sum(idx[c,left]['classification']=='strict'for c in ids),right_strict=sum(idx[c,right]['classification']=='strict'for c in ids))
    for name,x in [('signed_gap',gaps),('absolute_gap',[abs(x)for x in gaps])]:r.update({name+'_'+k:v for k,v in dist(x).items()})
    for tol,tag in[(1e-3,'1em3'),(1e-2,'1em2')]:
     lower=[-v for v in gaps if v< -tol];higher=[v for v in gaps if v>tol]
     r.update({f'lower_{tag}':len(lower),f'equivalent_{tag}':sum(abs(v)<=tol for v in gaps),f'higher_{tag}':len(higher)})
     for label,vs in [('conditional_lower_magnitude',lower),('conditional_higher_magnitude',higher)]:r.update({label+'_'+tag+'_'+k:v for k,v in dist(vs).items()})
    summaries.append(r)
    for side,pos in [('left',1),('right',2)]:
     for metric in('primal','dual','complementarity'):
      residuals.append(dict(scope=scope,left=left,right=right,side=side,metric=metric,**dist([p[pos][metric]for p in common])))
 return summaries,pairrows,residuals,cohorts
def writecsv(path,rows):
 if not rows:return
 with Path(path).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(x):return '—'if x is None else f'{100*x:.4g}%'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();idx,cases,proofs=parse(records(a.observations));summary,pairs,residuals,cohorts=analyse(idx,cases)
 if[len(cohorts[k])for k in SCOPES]!=[797,503,294]:raise ValueError('Unexpected ASU/rest coverage')
 a.out.mkdir(parents=True,exist_ok=True);writecsv(a.out/'OBJECTIFS_M_TOUTES_VARIANTES.csv',summary);writecsv(a.out/'RESIDUS_M_PAIRES_COMMUNES.csv',residuals)
 with gzip.open(a.out/'OBJECTIFS_M_PAIRES.csv.gz','wt',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
 facts=dict(schema='archived-default-objective-M-review-v1',source=str(a.observations.resolve()),source_sha256=sha(a.observations),cases=4078,archived_configurations=14,all_variants=12,comparators=list(IPOPT),M_cases=797,ASU_dataset='beam_structural',ASU_cases=503,outside_ASU_cases=294,summary_rows=len(summary),quartiles='type7 interpolation at (N-1)*p',relative_gap='(f_left-f_right)/max(1,abs(f_left),abs(f_right))',minimization=True,thresholds=[1e-3,1e-2],proof_levels=proofs,source_frozen_identity_checked=True,original_residual_consistency_checked=True,causal_attribution=False,optimizer_invocations=0,new_endpoint_evaluations=0,summaries=summary)
 (a.out/'OBJECTIFS_M_FAITS.json').write_text(json.dumps(facts,indent=2,allow_nan=False,ensure_ascii=False)+'\n')
 lines=['QUALITÉ DES OBJECTIFS — RÉGIME M, ARCHIVE IPOPT PAR DÉFAUT','797 cas : 503 ASU (beam_structural) et 294 autres. Chaque comparaison utilise sa propre intersection de certificats originaux stricts, mêmes QCQP et départs par empreinte.','Les 12 variantes Lazy sont toutes incluses. Les archives ne permettent aucune attribution causale à la stratégie de barrière ni au backend seul.','Écart signé relatif = (f_Lazy-f_Ipopt)/max(1,|f_Lazy|,|f_Ipopt|), négatif en faveur de Lazy. Quartiles calculés sur toutes les paires communes ; les médianes conditionnelles ne concernent que les objectifs effectivement plus élevés au seuil.','']
 for scope in SCOPES:
  lines.append(scope+' : '+str(len(cohorts[scope]))+' cas')
  lines.append('Lazy | Ipopt | N | quartiles signés q25/médiane/q75 | inférieur/équivalent/supérieur à 1e-3 | à 1e-2 | médiane conditionnelle supérieure, 1e-3')
  for r in summary:
   if r['scope']!=scope:continue
   lines.append(f"{r['left']} | {r['right']} | {r['signed_gap_N']} | {pct(r['signed_gap_q25'])}/{pct(r['signed_gap_median'])}/{pct(r['signed_gap_q75'])} | {r['lower_1em3']}/{r['equivalent_1em3']}/{r['higher_1em3']} | {r['lower_1em2']}/{r['equivalent_1em2']}/{r['higher_1em2']} | {pct(r['conditional_higher_magnitude_1em3_median'])}")
  lines.append('')
 lines+=['Les petits résidus KKT ne garantissent pas l’identité des minima locaux. Les résidus finaux sont livrés pour chaque cohorte appariée ; aucune correction de précision n’est imputée sans mesure. Les preuves archivées ont deux niveaux : projection authentifiée pour une campagne et audits complets pour l’autre ; aucun audit vectoriel original n’est recréé à partir de ce fichier compact.','Les échecs de certificat, le nombre de paires et les collections doivent être lus ensemble : une faible différence sur les seuls succès communs ne mesure pas la couverture. La nouvelle expérience Ipopt alignée est distincte et nécessite de nouveaux endpoints pour ses quatre cellules.']
 (a.out/'LIRE_OBJECTIFS_M.txt').write_text('\n'.join(lines)+'\n');print('OBJECTIFS_M',len(summary),'résumés ;',len(pairs),'paires stratifiées ; 0 solveur')
if __name__=='__main__':main()
