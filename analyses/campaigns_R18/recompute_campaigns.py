#!/usr/bin/env python3
"""Recompute the two R18 ablations from compact scalar records (standard library).

No optimizer is invoked. Existing original-QCQP audit classifications are retained;
this script does not recertify endpoint vectors. Use --out DIR to save the summary
and the complete ASU keep/remove pairing table.
"""
from pathlib import Path
from collections import Counter, defaultdict
from statistics import median, quantiles
import argparse, csv, gzip, json, math

BASE = Path(__file__).resolve().parent


def read_rows(name):
    with gzip.open(BASE / name, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def strict(row):
    return row['classification'] == 'strict'


def gap(a, b):
    return (a-b) / max(1, abs(a), abs(b))


def describe(values):
    q = quantiles(values, n=4, method='inclusive') if len(values)>1 else values*3
    return dict(N=len(values), median=median(values), q1=q[0], q3=q[2],
                minimum=min(values), maximum=max(values))


def assert_close(a, b):
    assert math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15), (a, b)


def asu():
    rows = read_rows('ASU_OBSERVATIONS.jsonl.gz')
    cases = sorted({r['case_id'] for r in rows})
    lookup = {(r['case_id'],r['backend'],r['policy'],r['bounds']):r for r in rows}
    assert len(rows) == len(lookup) == 6036 and len(cases) == 503
    for case in cases:
        chosen = [r for r in rows if r['case_id']==case]
        assert len(chosen)==12
        assert len({(r['input_sha256'],r['start_sha256']) for r in chosen})==1
    counts = dict(Counter(r['classification'] for r in rows))
    assert counts == dict(strict=4285, nonstrict=1729, indeterminate=22)
    for r in rows:
        if r['classification'] != 'indeterminate':
            assert r['budget_verified'] and r['intervention_verified']
            assert r['audit_available'] and r['original_bounds_used']
        if strict(r):
            assert r['strict_qcqp']
            assert all(r[k] <= 1e-6 for k in ('primal','dual','complementarity'))
    cells=[]
    for b in ('ma27','mumps'):
        for p in ('R3','SS','HH'):
            for mode in ('keep','remove'):
                selected=[lookup[c,b,p,mode] for c in cases]
                cells.append(dict(backend=b,policy=p,bounds=mode,planned=503,
                    classifications=dict(Counter(r['classification'] for r in selected)),
                    statuses=dict(Counter(r['status'] for r in selected))))
    comparisons=[]; pairs=[]
    for b in ('ma27','mumps'):
        common=[c for c in cases if all(strict(lookup[c,b,p,m])
                    for p in ('R3','SS','HH') for m in ('keep','remove'))]
        assert len(common)==dict(ma27=305,mumps=304)[b]
        for p in ('R3','SS','HH'):
            both=[c for c in cases if all(strict(lookup[c,b,p,m]) for m in ('keep','remove'))]
            lost=[c for c in cases if strict(lookup[c,b,p,'keep']) and not strict(lookup[c,b,p,'remove'])]
            gained=[c for c in cases if not strict(lookup[c,b,p,'keep']) and strict(lookup[c,b,p,'remove'])]
            assert not gained
            for scope,cohort in (('all_planned',both),('six_common_certificates',common)):
                values=[gap(lookup[c,b,p,'keep']['objective'],lookup[c,b,p,'remove']['objective']) for c in cohort]
                d=dict(backend=b,policy=p,scope=scope,normalized_gap=describe(values),
                    threshold_0_001=dict(remove_lower=sum(v>.001 for v in values),
                        equivalent=sum(abs(v)<=.001 for v in values),keep_lower=sum(v<-.001 for v in values)),
                    lost_strict=len(lost) if scope=='all_planned' else 0,
                    gained_strict=len(gained) if scope=='all_planned' else 0,
                    lost_remove_classifications=dict(Counter(lookup[c,b,p,'remove']['classification'] for c in lost)) if scope=='all_planned' else {},
                    lost_remove_statuses=dict(Counter(lookup[c,b,p,'remove']['status'] for c in lost)) if scope=='all_planned' else {})
                comparisons.append(d)
            for c in cases:
                a,z=lookup[c,b,p,'keep'],lookup[c,b,p,'remove']
                ok=strict(a) and strict(z)
                pairs.append(dict(case_id=c,backend=b,policy=p,keep_classification=a['classification'],
                    remove_classification=z['classification'],keep_status=a['status'],remove_status=z['status'],
                    keep_objective=a['objective'],remove_objective=z['objective'],
                    normalized_gap=gap(a['objective'],z['objective']) if ok else None,
                    both_strict=ok,in_six_cell_strict_cohort=c in common))
    expected=json.loads((BASE/'receipts/ASU_EXPECTED_STATISTICS.json').read_text())
    for d in comparisons:
        scope=d['scope']
        e=next(c for c in expected['comparisons'] if (c['backend'],c['policy'],c['scope'])==(d['backend'],d['policy'],scope))
        assert d['normalized_gap']['N']==e['both_strict']
        for k in ('median','q1','q3'): assert_close(d['normalized_gap'][k], e['normalized_gap'][k])
        assert d['threshold_0_001']==e['thresholds']['0.001']
        assert d['lost_strict']==e['lost_strict_on_removal']
    return dict(observations=len(rows),cases=len(cases),classification_counts=counts,
        cells=cells,comparisons=comparisons,
        gap_definition='(f_keep-f_remove)/max(1,abs(f_keep),abs(f_remove)); positive favors removal'),pairs


def persistence():
    rows=read_rows('PERSISTENCE_OBSERVATIONS.jsonl.gz')
    assert len(rows)==144
    grouped=defaultdict(list)
    for r in rows:
        assert strict(r) and r['strict_qcqp'] and r['budget_verified'] and r['quiet_valid'] and r['finalized']
        assert not r['quiet_contaminated']
        assert r['resource_checks']['cpu']==0 and r['resource_checks']['threads']==1 and r['resource_checks']['returncode']==0
        assert r['resource_checks']['elapsed_seconds']==r['seconds_process']
        assert all(v<=1e-6 for v in r['residuals'].values())
        grouped[r['pair_id']].append(r)
    assert len(grouped)==24 and len({r['case_id'] for r in rows})==12
    pairs=[]
    for pair_id,chosen in sorted(grouped.items()):
        assert len(chosen)==6
        for key in ('numerical_core_sha256','objective','factor_attempts'):
            assert len({r[key] for r in chosen})==1
        assert len({(r['input_sha256'],r['start_sha256']) for r in chosen})==1
        assert len({json.dumps(r['residuals'],sort_keys=True) for r in chosen})==1
        assert len({r['cache']['reanchors'] for r in chosen})==1
        modes={}
        for mode in ('reuse','rebuild'):
            selected=sorted((r for r in chosen if r['mode']==mode),key=lambda r:r['repetition'])
            assert [r['repetition'] for r in selected]==[1,2,3]
            modes[mode]=dict(seconds=[r['seconds_process'] for r in selected],median_seconds=median(r['seconds_process'] for r in selected),
                preparations=selected[0]['cache']['preparations'],reanchors=selected[0]['cache']['reanchors'],
                corrected_rows=selected[0]['cache']['corrected_rows'])
        ratio=modes['reuse']['median_seconds']/modes['rebuild']['median_seconds']
        pairs.append(dict(pair_id=pair_id,case_id=chosen[0]['case_id'],backend=chosen[0]['backend'],family=chosen[0]['family'],
            ratio_reuse_over_rebuild=ratio,gain_percent=100*(1-ratio),modes=modes,
            repetition_wins=sum(a<b for a,b in zip(modes['reuse']['seconds'],modes['rebuild']['seconds']))))
    backends=[];families=[]
    for b in ('ma27','mumps'):
        selected=[p for p in pairs if p['backend']==b]
        d=describe([p['ratio_reuse_over_rebuild'] for p in selected])
        d.update(backend=b,median_gain_percent=100*(1-d['median']),reuse_faster_pairs=sum(p['ratio_reuse_over_rebuild']<1 for p in selected))
        backends.append(d)
        for family in sorted({p['family'] for p in selected}):
            chosen=[p for p in selected if p['family']==family]
            d=describe([p['ratio_reuse_over_rebuild'] for p in chosen])
            d.update(backend=b,family=family,median_gain_percent=100*(1-d['median']))
            families.append(d)
    expected=json.loads((BASE/'receipts/PERSISTENCE_EXPECTED_STATISTICS.json').read_text())
    for d in backends:
        e=next(b for b in expected['backends'] if b['backend']==d['backend'])
        for k in ('median','q1','q3'): assert_close(d[k],e[k])
    q=json.loads((BASE/'PERSISTENCE_QUALIFICATION.json').read_text())
    assert q['passed'] and q['observations']==q['expected_observations']==72
    assert len(q['pairs'])==24
    for p in q['pairs']:
        assert all(p[k] for k in ('passed','reference_reuse_endpoint_and_work_exact','reuse_rebuild_numerical_endpoint_and_work_exact','original_audit_exact','ordered_correction_and_state_trace_exact','cache_exercised'))
    totals={m:sum(p['preparations_'+m] for p in q['pairs']) for m in ('reuse','rebuild')}
    assert totals==dict(reuse=25,rebuild=4638)
    integrity=json.loads((BASE/'receipts/PERSISTENCE_INTEGRITY.json').read_text())
    nominal=sum(v['count'] for v in integrity['nominal_results'].values())
    attempts=sum(v['count'] for v in integrity['attempts'].values())
    assert nominal==216 and attempts==220 and len(integrity['retries'])==4
    assert integrity['quiet_recomputed_from_raw_samples']['checked']==220
    assert not integrity['quiet_recomputed_from_raw_samples']['mismatches']
    return dict(timed_observations=144,qualification_observations=72,nominal_observations=216,
        extra_contaminated_attempts=attempts-nominal,actual_attempts=attempts,all_timed_strict=True,
        all_pairs_numerically_identical=True,qualification_preparations=totals,
        backends=backends,families=families,pairs=pairs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    a,pairs=asu();p=persistence()
    result=dict(status='PASS',asu=a,persistence=p)
    if args.out:
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'RECOMPUTED_SUMMARY.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        with (args.out/'ASU_ALL_KEEP_REMOVE_PAIRS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
        flat=[dict(pair_id=p['pair_id'],case_id=p['case_id'],backend=p['backend'],family=p['family'],
             reuse_median_seconds=p['modes']['reuse']['median_seconds'],
             rebuild_median_seconds=p['modes']['rebuild']['median_seconds'],
             ratio_reuse_over_rebuild=p['ratio_reuse_over_rebuild'],gain_percent=p['gain_percent']) for p in p['pairs']]
        with (args.out/'PERSISTENCE_ALL_PAIRS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print('PASS: ASU 6036 observations, 3018 pairs; persistence 144 timed observations, 24 pairs, 72 qualification observations.')
    print('ASU six-cell strict cohorts: MA27 305; MUMPS 304. Persistence median ratios: '+', '.join(f"{b['backend']} {b['median']:.9f}" for b in p['backends']))


if __name__=='__main__': main()
