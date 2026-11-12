#!/usr/bin/env python3
"""Freeze 12 size-stratified cases before new persistence measurements."""
from pathlib import Path
import argparse
import hashlib
import json

HERE=Path(__file__).resolve().parent
SEED='20261110-persistence-r1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def freeze():
    source=HERE/'support_chronos/cohort/COHORTE_CHRONO_FIGEE.json'
    context=HERE/'support_chronos/evidence/CONTEXTE_HISTORIQUE.json'
    historical=json.loads(source.read_text());metadata=json.loads(context.read_text())
    declared=historical.pop('cohort_content_sha256')
    if digest(historical)!=declared:raise ValueError('Historical cohort changed')
    jobs={j['job_id']:j for j in historical['jobs']}
    index={(p['case_id'],p['backend']):p for p in historical['pairs']}
    common={p['case_id'] for p in historical['pairs'] if p['backend']=='ma27'} & {p['case_id'] for p in historical['pairs'] if p['backend']=='mumps'}
    cases={c['id']:c for c in metadata['cases']}
    chosen=[];strata=[]
    for regime in ('I','M'):
        candidates=sorted((cases[c] for c in common if cases[c]['regime']==regime and cases[c]['dimensions'].get('quadratic_inequalities',0)>0),key=lambda c:(c['dimensions']['n'],c['id']))
        if len(candidates)<6:raise ValueError('Insufficient predeclared candidate pool')
        for band in range(3):
            pool=candidates[len(candidates)*band//3:len(candidates)*(band+1)//3]
            selected=sorted(pool,key=lambda c:digest([SEED,regime,band,c['id']]))[:2]
            if len(selected)!=2:raise ValueError('Size stratum incomplete')
            chosen+=selected
            strata.append(dict(regime=regime,size_band=band+1,candidates=len(pool),n_min=pool[0]['dimensions']['n'],n_max=pool[-1]['dimensions']['n'],selected=[c['id'] for c in selected]))
    pairs=[]
    for case in chosen:
        for backend in ('ma27','mumps'):
            old=index[case['id'],backend];job=jobs[old['native_job_id']]
            pairs.append(dict(pair_id='persist_'+digest([case['id'],backend])[:24],case_id=case['id'],backend=backend,regime=case['regime'],
                dimensions=case['dimensions'],dataset_id=case['dataset_id'],family=case['family'],
                input_sha256=job['input_sha256'],start_sha256=job['start_sha256'],start_binding=job['start_binding'],
                frozen_input=case['input'],case_metadata=case,historical_native_job=job))
    result=dict(schema='persistence-cohort-v1',seed=SEED,selection=dict(
        scope='R3 fixed deterministic inequality curvature; both historical backends strict with process times >1 s',
        rule='Intersection of backend cohorts, quadratic_inequalities >0; six I and six M cases; sort by dimension n then identifier, divide each regime into three equal-count size bands, select two per band by seeded SHA256.',
        future_timings_used=False,favorable_effect_selection=False,positive_correction_not_presumed=True),
        source_cohort_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_context_sha256=hashlib.sha256(context.read_bytes()).hexdigest(),
        strata=strata,cases=chosen,pairs=pairs,counts=dict(cases=12,backend_pairs=24,qualification_runs=72,timed_runs=144,repetitions=3,total_solver_runs=216),
        interpretation='Targeted mechanism and cost control. Eligibility uses historical successes; no general-corpus speed claim. Actual corrected rows, repeated model installations and all numerical attrition must be reported.')
    result['cohort_content_sha256']=digest(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=HERE/'COHORTE_PERSISTANCE.json');a=p.parse_args()
    value=freeze()
    if a.out.exists():
        if json.loads(a.out.read_text())!=value:raise ValueError('Frozen cohort differs; do not overwrite it')
    else:a.out.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(value['counts'],indent=2))


if __name__=='__main__':main()
