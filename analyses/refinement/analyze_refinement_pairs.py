#!/usr/bin/env python3
"""Descriptive matched frozen/adaptive comparisons; no solver or reclassification.

FROZEN means the common initial hybrid correction is retained. It does not
denote the historical complete R3 policy. The archived original classification
and intervention admission are preserved, including all indeterminate records.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import gzip
import hashlib
import io
import json
import math

BACKENDS = ('ma27', 'mumps')
GLOBALIZATIONS = ('GR3', 'GHS')
METHODS = tuple(f'{r}_{g}_{b}' for b in BACKENDS for r in ('FROZEN', 'REFINED') for g in GLOBALIZATIONS)
KEYS = ('case_id', 'method', 'backend', 'dataset', 'family', 'regime',
        'classification', 'strict_qcqp', 'status', 'experiment_proof_error',
        'input_sha256', 'start_sha256', 'primal', 'dual', 'complementarity',
        'objective', 'seconds_process', 'audit_seconds', 'iterations',
        'budget_verified', 'intervention_verified', 'objective_exact',
        'common_guards', 'common_representation', 'refinement_policy',
        'globalization_policy', 'inequality_policy', 'guard_config_sha256',
        'initial_model_state', 'initial_inequality_model_sha256',
        'refinement_attempt_events', 'accepted_refinements',
        'refinement_update_events', 'refinement_rollback_events',
        'frozen_metric_invariant_verified', 'accepted_direction_origins',
        'accepted_base_direction_origins', 'binary_sha256',
        'installation_sha256', 'study_contract_sha256', 'preregistration_sha256')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    z = (len(values) - 1) * probability
    lo = int(z)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (z - lo)


def distribution(values):
    return dict(n=len(values), Q1=quantile(values, .25), median=quantile(values, .5), Q3=quantile(values, .75),
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def ratios(values):
    return dict(distribution(values), geometric_mean=math.exp(sum(map(math.log, values)) / len(values)) if values else None)


def strict(row):
    if row['classification'] != 'strict':
        return False
    assert row.get('strict_qcqp') is True
    assert row.get('budget_verified') is True and row.get('intervention_verified') is True
    assert all(finite(row[k]) and 0 <= row[k] <= 1e-6 for k in ('primal', 'dual', 'complementarity'))
    return True


def objective_gap(adaptive, frozen):
    a, f = adaptive.get('objective'), frozen.get('objective')
    if not (finite(a) and finite(f)):
        return None
    # Same symmetric denominator as the R16 original-objective analysis.
    return (a - f) / max(1., abs(a), abs(f))


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def compact(raw, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(raw).open() as src, target.open('wb') as binary:
        with gzip.GzipFile(filename='', fileobj=binary, mode='wb', mtime=0) as zipped:
            for line in src:
                if not line.strip():
                    continue
                r = json.loads(line)
                value = {k: r.get(k) for k in KEYS}
                value['start_sha256'] = r.get('start_sha256') or r.get('start_binding', {}).get('start_sha256')
                zipped.write((json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode())


def pair_summary(index, ids, backend, globalization, scope, case_rows):
    A = f'REFINED_{globalization}_{backend}'
    F = f'FROZEN_{globalization}_{backend}'
    counts = Counter()
    gaps, times, long_times, same_objective_long, iterations = [], [], [], [], []
    classes = Counter()
    net_lower = net_upper = 0
    for cid in ids:
        a, f = index[A][cid], index[F][cid]
        assert a['input_sha256'] == f['input_sha256'] and a['start_sha256'] == f['start_sha256']
        assert all(a[k] == f[k] for k in ('dataset', 'regime', 'family', 'common_guards', 'common_representation',
                                        'objective_exact', 'inequality_policy', 'globalization_policy'))
        assert a['objective_exact'] is True and a['inequality_policy'] == 'hybrid'
        if a.get('guard_config_sha256') and f.get('guard_config_sha256'):
            assert a['guard_config_sha256'] == f['guard_config_sha256']
        if a.get('initial_model_state') == f.get('initial_model_state') == 'reached':
            assert a['initial_inequality_model_sha256'] == f['initial_inequality_model_sha256']
            counts['initial_metric_identity_verified'] += 1
        sa, sf = strict(a), strict(f)
        cat = 'both_strict' if sa and sf else 'adaptive_only' if sa else 'frozen_only' if sf else 'neither_strict'
        counts[cat] += 1
        classes[a['classification'], f['classification']] += 1
        counts['adaptive_strict'] += sa; counts['frozen_strict'] += sf
        counts['adaptive_indeterminate'] += a['classification'] == 'indeterminate'
        counts['frozen_indeterminate'] += f['classification'] == 'indeterminate'
        au = a['classification'] == 'indeterminate'
        fu = f['classification'] == 'indeterminate'
        net_lower += (0 if au else int(sa)) - (1 if fu else int(sf))
        net_upper += (1 if au else int(sa)) - (0 if fu else int(sf))
        counts['technical_or_missing_pairs'] += au or fu
        if a['classification'] != 'indeterminate' and f['classification'] != 'indeterminate':
            counts['both_qualified'] += 1
            counts['qualified_adaptive_only'] += sa and not sf
            counts['qualified_frozen_only'] += sf and not sa
        if not (sa and sf):
            continue
        gap = objective_gap(a, f)
        if gap is not None:
            gaps.append(gap)
        ta, tf = a.get('seconds_process'), f.get('seconds_process')
        time_ratio = ta / tf if finite(ta) and finite(tf) and ta > 0 and tf > 0 else None
        if time_ratio is not None:
            times.append(time_ratio)
            if ta > 1 and tf > 1:
                long_times.append(time_ratio)
                if gap is not None and abs(gap) <= 1e-3:
                    same_objective_long.append(time_ratio)
        ia, iff = a.get('iterations'), f.get('iterations')
        delta_iterations = ia - iff if finite(ia) and finite(iff) else None
        if delta_iterations is not None:
            iterations.append(delta_iterations)
            counts['adaptive_fewer_iterations'] += delta_iterations < 0
            counts['same_iterations'] += delta_iterations == 0
            counts['adaptive_more_iterations'] += delta_iterations > 0
        if scope in ('I', 'M'):
            case_rows.append(dict(case_id=cid, dataset=a['dataset'], regime=a['regime'], backend=backend,
                                  globalization=globalization, A=A, F=F, objective_A=a.get('objective'),
                                  objective_F=f.get('objective'), signed_objective_gap=gap,
                                  seconds_A=ta, seconds_F=tf, ratio_A_over_F=time_ratio,
                                  iterations_A=ia, iterations_F=iff,
                                  accepted_refinements_A=a.get('accepted_refinements'),
                                  input_sha256=a['input_sha256'], start_sha256=a['start_sha256']))
    assert sum(counts[k] for k in ('both_strict', 'adaptive_only', 'frozen_only', 'neither_strict')) == len(ids)
    result = dict(backend=backend, globalization=globalization, scope=scope, planned=len(ids),
                  **{k: counts[k] for k in ('both_strict', 'adaptive_only', 'frozen_only', 'neither_strict',
                                           'adaptive_strict', 'frozen_strict', 'adaptive_indeterminate', 'frozen_indeterminate',
                                           'both_qualified', 'qualified_adaptive_only', 'qualified_frozen_only',
                                           'initial_metric_identity_verified', 'adaptive_fewer_iterations',
                                           'same_iterations', 'adaptive_more_iterations')},
                  net_certificates=counts['adaptive_strict']-counts['frozen_strict'],
                  unresolved_completion_net_bounds=[net_lower, net_upper],
                  technical_or_missing_pairs=counts['technical_or_missing_pairs'],
                  objective_signed_gap=distribution(gaps), objective_absolute_gap=distribution([abs(x) for x in gaps]),
                  process_ratio_A_over_F=ratios(times), process_ratio_both_over_1s=ratios(long_times),
                  process_ratio_both_over_1s_same_objective_0p1pct=ratios(same_objective_long),
                  iterations_A_minus_F=distribution(iterations),
                  classification_cross=[dict(adaptive=a, frozen=f, count=n) for (a, f), n in sorted(classes.items())])
    for tol, tag in [(1e-3, '0p1pct'), (1e-2, '1pct')]:
        result['objective_' + tag] = dict(adaptive_lower=sum(x < -tol for x in gaps),
                                         equivalent=sum(abs(x) <= tol for x in gaps),
                                         adaptive_higher=sum(x > tol for x in gaps))
    return result


def registered_primary(cases, index, backend, minimum=20):
    """Same conservative completion rule as the preregistered primary().

    A technical outcome is unknown, not silently a numerical failure. The
    concentration comparison preserves dependence of a collection and total.
    """
    groups=defaultdict(lambda:[0,0,0]); families=defaultdict(lambda:[0,0,0])
    wins=losses=possible_wins=possible_losses=unknown=0
    for c in cases:
        ar=index[f'REFINED_GHS_{backend}'][c['case_id']]
        fr=index[f'FROZEN_GHS_{backend}'][c['case_id']]
        a=None if ar['classification']=='indeterminate' else strict(ar)
        f=None if fr['classification']=='indeterminate' else strict(fr)
        al,au=(0,1)if a is None else(int(a),int(a))
        fl,fu=(0,1)if f is None else(int(f),int(f))
        lo,hi=al-fu,au-fl
        for dest,key in ((groups,c['dataset']),(families,c['family'])):
            dest[key][0]+=lo;dest[key][1]+=hi;dest[key][2]+=1
        wins+=a is True and f is False;losses+=a is False and f is True
        possible_wins+=a is not False and f is not True
        possible_losses+=a is not True and f is not False
        unknown+=a is None or f is None
    lo=sum(v[0]for v in groups.values());hi=sum(v[1]for v in groups.values())
    fl=sum(v[0]/v[2]for v in families.values())/len(families)
    fu=sum(v[1]/v[2]for v in families.values())/len(families)
    concentration={k:dict(net_lower=v[0],net_upper=v[1],planned=v[2],
                          twice_group_minus_total_lower=v[0]-(hi-v[1]),
                          twice_group_minus_total_upper=v[1]-(lo-v[0]))for k,v in groups.items()}
    conditions=dict(net_certificates_at_least_minimum_robust=lo>=minimum,
                    paired_discordances_at_least_two_to_one_robust=wins>0 and wins>=2*possible_losses,
                    no_collection_above_half_net_gain_robust=lo>0 and all(v['twice_group_minus_total_upper']<=0 for v in concentration.values()),
                    family_macro_sign_positive_robust=fl>0)
    if all(conditions.values()):
        decision='positive_under_preregistered_rule'
    elif unknown==0:
        decision='criterion_not_met'
    elif hi<minimum or possible_wins<2*losses or fu<=0 or (lo>0 and any(v['twice_group_minus_total_lower']>0 for v in concentration.values())):
        decision='criterion_not_met_for_every_possible_completion'
    else:
        decision='not_identified_due_to_technical_or_missing_outcomes'
    return dict(backend=backend,scope='I+M confirmatory; I and M descriptive only',
                contrast='REFINED_GHS minus FROZEN_GHS',minimum_net_certificates=minimum,
                net_gain_bounds=[lo,hi],known_wins=wins,known_losses=losses,
                possible_wins_upper=possible_wins,possible_losses_upper=possible_losses,
                technical_or_missing_pairs=unknown,family_macro_delta_bounds=[fl,fu],
                collection_contributions=concentration,conditions=conditions,decision=decision)


def fmt(value, digits=2):
    return '--' if value is None else f'{value:.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--raw', type=Path); group.add_argument('--compact', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--study', type=Path, help='Optional sealed study contract, preserved as preregistration evidence.')
    a = p.parse_args(); out=a.out; out.mkdir(parents=True, exist_ok=True)
    source = a.compact or out / 'REFINEMENT_OBSERVATIONS_8_CONFIGS.jsonl.gz'
    if a.raw:
        compact(a.raw, source)
    with gzip.open(source, 'rt') as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    assert len(rows)==14672
    index={m:{} for m in METHODS}
    for r in rows:
        assert r['method'] in index and r['case_id'] not in index[r['method']]
        index[r['method']][r['case_id']]=r
    ids=sorted(index[METHODS[0]])
    assert len(ids)==1834 and all(set(index[m])==set(ids) for m in METHODS)
    regimes={g:[cid for cid in ids if index[METHODS[0]][cid]['regime']==g] for g in ('I','M')}
    assert {g:len(v)for g,v in regimes.items()}=={'I':1037,'M':797}
    asu=[cid for cid in regimes['M']if index[METHODS[0]][cid]['dataset']=='beam_structural']
    other=[cid for cid in regimes['M']if cid not in set(asu)]
    scopes=dict(I=regimes['I'],M=regimes['M'],IM=ids,ASU=asu,other_M=other)
    comparisons=[];case_rows=[]
    for b in BACKENDS:
        for g in GLOBALIZATIONS:
            for scope,cohort in scopes.items():
                comparisons.append(pair_summary(index,cohort,b,g,scope,case_rows))
    activation=[]
    for m in METHODS:
        for scope,cohort in scopes.items():
            group=[index[m][cid] for cid in cohort]
            known=[r for r in group if type(r.get('accepted_refinements')) is int]
            active=[r for r in known if r['accepted_refinements']>0]
            st=[r for r in group if strict(r)]
            if m.startswith('FROZEN'):
                assert not active
            activation.append(dict(method=m,scope=scope,planned=len(group),
                                   known_refinement_counters=len(known),unknown_counters=len(group)-len(known),
                                   activated=len(active),strict=len(st),strict_activated=sum(r['accepted_refinements']>0 for r in st),
                                   attempts_positive=sum((r.get('refinement_attempt_events')or 0)>0 for r in known),
                                   accepted_updates=sum(r['accepted_refinements'] for r in known),
                                   accepted_update_distribution=distribution([r['accepted_refinements']for r in active]),
                                   rollbacks=sum(r.get('refinement_rollback_events')or 0 for r in known)))
    interactions=[]
    for b in BACKENDS:
        for scope,cohort in scopes.items():
            four_common=[cid for cid in cohort if all(strict(index[f'{r}_{g}_{b}'][cid]) for r in ('FROZEN','REFINED')for g in GLOBALIZATIONS)]
            log_interactions=[]; gap_interactions=[]
            for cid in four_common:
                f3,a3,fh,ah=[index[m][cid]for m in [f'FROZEN_GR3_{b}',f'REFINED_GR3_{b}',f'FROZEN_GHS_{b}',f'REFINED_GHS_{b}']]
                if all(finite(r.get('seconds_process'))and r['seconds_process']>1 for r in [f3,a3,fh,ah]):
                    log_interactions.append(math.log(ah['seconds_process']/fh['seconds_process'])-math.log(a3['seconds_process']/f3['seconds_process']))
                d3,dh=objective_gap(a3,f3),objective_gap(ah,fh)
                if d3 is not None and dh is not None:gap_interactions.append(dh-d3)
            x3=next(r for r in comparisons if r['backend']==b and r['globalization']=='GR3'and r['scope']==scope)
            xh=next(r for r in comparisons if r['backend']==b and r['globalization']=='GHS'and r['scope']==scope)
            interactions.append(dict(backend=b,scope=scope,planned=len(cohort),four_cell_common_strict=len(four_common),
                                     coverage_interaction=xh['net_certificates']-x3['net_certificates'],
                                     refinement_average_net_certificates=.5*(xh['net_certificates']+x3['net_certificates']),
                                     log_time_interaction_both_over_1s_all4=distribution(log_interactions),
                                     objective_gap_interaction=distribution(gap_interactions)))
    study=json.loads(a.study.read_text())if a.study else None
    if study:
        assert study['primary']['comparison']=='REFINED_GHS minus FROZEN_GHS'
        assert study['primary']['minimum_net_certificates_per_backend']==20
        assert study['primary']['discordance_ratio_at_least']==2
        assert study['primary']['maximum_single_collection_fraction_of_net_gain']==.5
        assert study['primary']['family_macro_effect_strictly_positive']is True
    cases=[{k:index[METHODS[0]][cid][k]for k in ('case_id','regime','dataset','family')}for cid in ids]
    primary=[registered_primary(cases,index,b)for b in BACKENDS]
    report=dict(schema='controlled-refinement-pairs-v1',input_sha256=sha(source),records=len(rows),cases=len(ids),
                classifications=dict(Counter(r['classification']for r in rows)), comparisons=comparisons,
                activation=activation,interactions=interactions,registered_primary=primary,
                study_sha256=sha(a.study)if a.study else None,
                optimizer_invocations=0,classification_changes=0,
                definitions=dict(time_ratio='adaptive/frozen solver-process elapsed time, independent audit excluded; positive finite times; selected >1s cohort requires both >1s',
                                 objective_gap='(f_adaptive-f_frozen)/max(1,abs(f_adaptive),abs(f_frozen)); original objective includes its constant',
                                 frozen='Common initial hybrid correction retained; not the historical complete R3 policy',
                                 validity='Descriptive archived qualified common-certificate pairs; indeterminate records retained separately; no fresh residual audit',
                                 activation='Known admitted counters only; missing counters are unknown, not zero',
                                 sampling='One run per cell under concurrent load; descriptive corpus summaries, not independent random samples'))
    write_json(out/'REFINEMENT_PAIR_SUMMARY.json',report)
    csv_write(out/'REFINEMENT_PAIR_SUMMARY.csv',comparisons)
    csv_write(out/'REFINEMENT_COMMON_CERTIFICATE_PAIRS.csv',case_rows)
    csv_write(out/'REFINEMENT_ACTIVATION.csv',activation)
    csv_write(out/'REFINEMENT_INTERACTIONS.csv',interactions)
    write_json(out/'REGISTERED_PRIMARY.json',dict(registered_primary=primary,study_sha256=report['study_sha256']))
    main=[r'\begin{tabular}{@{}lllrrrrr@{}}',r'\toprule',r'Linear solver & Glob. & Regime & F & A & A-only & F-only & Time A/F ($n$)\\',r'\midrule']
    detail=[r'\begin{tabular}{@{}lllrrrrrr@{}}',r'\toprule',r'Linear solver & Glob. & Regime & Both & A-only & F-only & Neither & Unqual. F/A & Obj. L/E/H\\',r'\midrule']
    times=[r'\begin{tabular}{@{}lllrrrrr@{}}',r'\toprule',r'Linear solver & Glob. & Regime & $n$ & $Q_1$ & Median & $Q_3$ & Iter. $A-F$\\',r'\midrule']
    for r in comparisons:
        if r['scope'] not in ('I','M'):continue
        q=r['process_ratio_both_over_1s'];o=r['objective_0p1pct']
        main.append(f"{r['backend'].upper()} & {r['globalization']} & {r['scope']} & {r['frozen_strict']} & {r['adaptive_strict']} & {r['adaptive_only']} & {r['frozen_only']} & {fmt(q['median'])} ({q['n']})"+r'\\')
        detail.append(f"{r['backend'].upper()} & {r['globalization']} & {r['scope']} & {r['both_strict']} & {r['adaptive_only']} & {r['frozen_only']} & {r['neither_strict']} & {r['frozen_indeterminate']}/{r['adaptive_indeterminate']} & {o['adaptive_lower']}/{o['equivalent']}/{o['adaptive_higher']}"+r'\\')
        times.append(f"{r['backend'].upper()} & {r['globalization']} & {r['scope']} & {q['n']} & {fmt(q['Q1'])} & {fmt(q['median'])} & {fmt(q['Q3'])} & {fmt(r['iterations_A_minus_F']['median'],0)}"+r'\\')
    for name,table in [('refinement_main.tex',main),('refinement_detail.tex',detail),('refinement_times.tex',times)]:
        (out/name).write_text('\n'.join(table+[r'\bottomrule',r'\end{tabular}'])+'\n')
    asutable=[r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',
              r'Linear solver & Glob. & Common & Obj. L/E/H & $|g|$ med. (\%) & Iter. $A-F$ & Time A/F & $n_{>1s}$\\',r'\midrule']
    for r in comparisons:
        if r['scope']!='ASU':continue
        o=r['objective_0p1pct'];q=r['process_ratio_both_over_1s']
        asutable.append(f"{r['backend'].upper()} & {r['globalization']} & {r['both_strict']} & {o['adaptive_lower']}/{o['equivalent']}/{o['adaptive_higher']} & {fmt(100*r['objective_absolute_gap']['median'])} & {fmt(r['iterations_A_minus_F']['median'],0)} & {fmt(q['median'])} & {q['n']}"+r'\\')
    (out/'refinement_asu.tex').write_text('\n'.join(asutable+[r'\bottomrule',r'\end{tabular}'])+'\n')
    print(json.dumps([{k:r[k]for k in ['backend','globalization','scope','planned','both_strict','adaptive_only','frozen_only','net_certificates','process_ratio_both_over_1s','objective_0p1pct']}for r in comparisons if r['scope']in ('I','M')],indent=2))


if __name__=='__main__':
    main()
