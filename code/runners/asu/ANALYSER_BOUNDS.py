#!/usr/bin/env python3
"""Paired original-certified on/off objectives, including absolute gaps and common-six cohorts."""
from pathlib import Path
import argparse,collections,csv,json,math,statistics
import PROFILES_BOUNDS as profiles
def quantile(v,p):
    if not v:return None
    a=sorted(v);z=(len(a)-1)*p;i=int(z);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(z-i)
def summary(rows,A,B,scope,backend):
    gaps=[];counts=collections.Counter();records=[]
    for cid,by in rows.items():
        a=by.get(A);b=by.get(B)
        sa=isinstance(a,dict)and a.get('classification')=='strict';sb=isinstance(b,dict)and b.get('classification')=='strict'
        counts['both'if sa and sb else'A_only'if sa else'B_only'if sb else'neither']+=1
        if not(sa and sb):continue
        if a['input_sha256']!=b['input_sha256']or a['start_binding']!=b['start_binding']:raise ValueError('Appariement gelé incohérent : '+cid)
        for r in [a,b]:
            if r.get('budget_verified')is not True or r.get('intervention_verified')is not True:raise ValueError('Observation stricte sans preuve : '+cid)
            if any(type(r.get(k))not in(int,float)or not math.isfinite(r[k])or r[k]>1e-6 for k in ['primal','dual','complementarity']):raise ValueError('Résidu strict non admissible')
        fa=float(a['objective']);fb=float(b['objective'])
        if not(math.isfinite(fa)and math.isfinite(fb)):raise ValueError('Objectif non fini sur paire certifiée : '+cid)
        gap=(fa-fb)/max(1.,abs(fa),abs(fb));gaps.append(gap)
        records.append(dict(case_id=cid,scope=scope,backend=backend,A=A,B=B,gap=gap,absolute_gap=abs(gap),objective_A=fa,objective_B=fb,
          equality_implied_bound_sides_A=a.get('equality_implied_bound_sides'),equality_implied_bound_sides_B=b.get('equality_implied_bound_sides')))
    absolute=list(map(abs,gaps))
    d=dict(scope=scope,backend=backend,A=A,B=B,N=len(rows),**dict(counts),signed_median=quantile(gaps,.5),signed_Q1=quantile(gaps,.25),signed_Q3=quantile(gaps,.75),absolute_median=quantile(absolute,.5),absolute_Q1=quantile(absolute,.25),absolute_Q3=quantile(absolute,.75))
    for key in ['A','B']:
        vals=[r['equality_implied_bound_sides_'+key]for r in records]
        d['bound_sides_'+key+'_counts']=dict(collections.Counter(map(str,vals)))
    for tol in [1e-3,1e-2]:d['threshold_'+str(tol)]=dict(A_lower=sum(g< -tol for g in gaps),equivalent=sum(abs(g)<=tol for g in gaps),B_lower=sum(g>tol for g in gaps))
    return d,records
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise ValueError('Sortie d’analyse neuve requise')
    plan=json.loads((a.run/'RUN_CONTRACT.json').read_text());installation=json.loads(Path(plan['installation']).read_text())
    cases=set(plan['case_ids']);bound={c['case_id']for c in installation['cases']}
    if cases!=bound or len(cases)!=plan['cases']or plan['tasks']!=len(cases)*12:raise ValueError('Périmètre du run incohérent avec la préparation')
    rows={cid:{}for cid in cases}
    for p in (a.run/'tasks').glob('*/RESULT.json'):
        r=json.loads(p.read_text());cid=r['case_id'];m=r['method']
        if cid not in rows or m not in profiles.CELLS:raise ValueError('Observation hors plan')
        if m in rows[cid]:raise ValueError('Observation dupliquée')
        if r.get('installation_sha256')!=plan['installation_sha256']:raise ValueError('Observation liée à une autre installation')
        rows[cid][m]=r
    missing=sum(len(profiles.CELLS)-len(v)for v in rows.values());summaries=[];pairs=[]
    for backend in profiles.BACKENDS:
        labels=[f'{p}_{s}_{backend}'for p in profiles.POLICIES for s in profiles.STATES]
        joint={cid:v for cid,v in rows.items()if all(v.get(m,{}).get('classification')=='strict'for m in labels)}
        for scope,data in [('all_planned',rows),('six_common_certificates',joint)]:
            for policy in profiles.POLICIES:
                d,p=summary(data,f'{policy}_keep_{backend}',f'{policy}_remove_{backend}',scope,backend);summaries.append(d);pairs+=p
            for state in profiles.STATES:
                for p,q in [('R3','SS'),('R3','HH'),('SS','HH')]:
                    d,pairs_new=summary(data,f'{p}_{state}_{backend}',f'{q}_{state}_{backend}',scope,backend);summaries.append(d);pairs+=pairs_new
    a.out.mkdir(parents=True)
    planned=plan['tasks']
    report=dict(schema='implied-bound-paired-analysis-v1',planned=planned,recorded=planned-missing,missing=missing,complete=missing==0,
      pair_gap='(f_A-f_B)/max(1,abs(f_A),abs(f_B)); positive favors B',summaries=summaries,
      interpretation='Each same-policy keep/remove pair isolates only the declared bound rule within its backend implementation. Absolute gaps and six-cell common cohorts prevent signed-median cancellation. No global optimality follows.')
    (a.out/'PAIRED_SUMMARY.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if pairs:
        with(a.out/'PAIRED_OBJECTIVES.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps({'complete':missing==0,'recorded':planned-missing,'missing':missing,'summaries':len(summaries),'optimizer_invocations':0},indent=2))
if __name__=='__main__':main()
