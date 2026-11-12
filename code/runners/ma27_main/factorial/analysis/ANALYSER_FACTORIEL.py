#!/usr/bin/env python3
"""Preregistered paired 2x2 analysis; no optimizer, no imputation of unknown outcomes."""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math
from collections import Counter,defaultdict
from pathlib import Path

CELLS=('R3','X1','X2','H0')
ALIASES={'R3':'R3','X1':'X1','X2':'X2','H0':'H0','factorial_R3':'R3',
         'factorial_X1':'X1','factorial_X2':'X2','factorial_H0':'H0'}
FACTORS={'R3':('cached_support','r3'),'X1':('cached_support','hs'),
         'X2':('hybrid','r3'),'H0':('hybrid','hs')}
CONTRASTS={'correction_average':{'R3':-.5,'X1':-.5,'X2':.5,'H0':.5},
           'controller_average':{'R3':-.5,'X1':.5,'X2':-.5,'H0':.5},
           'interaction':{'R3':1,'X1':-1,'X2':-1,'H0':1},
           'correction_under_common_controller':{'X1':-1,'H0':1},
           'correction_under_r3_controller':{'R3':-1,'X2':1}}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)
def flag(value):
    if value is True or value=='true' or value=='True':return True
    if value is False or value=='false' or value=='False':return False
    if value is None or value in ('','null','None','NA'):return None
    raise ValueError('Invalid explicit three-valued outcome: '+str(value))
def observations(path):
    p=Path(path)
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
        if '.csv' in p.name:yield from csv.DictReader(f)
        else:
            for line in f:
                if line.strip():yield json.loads(line)
def finite(value):
    try:v=float(value)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None
def median(values):
    v=sorted(values);n=len(v)
    return None if not n else v[n//2] if n%2 else (v[n//2-1]+v[n//2])/2
def writecsv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def bounds(value):return (0,1) if value is None else (int(value),int(value))
def delta(a,b):
    # a - b, preserving the shared pair rather than marginal success subtraction.
    al,au=bounds(a);bl,bu=bounds(b);return al-bu,au-bl
def totals(cases,records):
    out={}
    for cell in CELLS:
        count=Counter(records.get((c['case_id'],cell),{}).get('strict') for c in cases)
        out[cell]={'strict':count[True],'nonstrict':count[False],'indeterminate':count[None],'planned':len(cases)}
    return out
def contrast(cases,records,coeff):
    lo=hi=0.
    for c in cases:
        for cell,a in coeff.items():
            l,u=bounds(records.get((c['case_id'],cell),{}).get('strict'))
            lo+=a*(l if a>=0 else u);hi+=a*(u if a>=0 else l)
    return {'lower_certificate_equivalents':lo,'upper_certificate_equivalents':hi,
            'point_identified':lo==hi,'lower_percentage_points':100*lo/len(cases) if cases else None,
            'upper_percentage_points':100*hi/len(cases) if cases else None}

def primary(cases,records,net_threshold=20):
    # The one primary criterion compares H0 against X1 under the common controller.
    grouped=defaultdict(lambda:[0,0,0]);families=defaultdict(lambda:[0,0,0]);wins=losses=possible_wins=possible_losses=unknown=0
    for c in cases:
        a=records.get((c['case_id'],'H0'),{}).get('strict');b=records.get((c['case_id'],'X1'),{}).get('strict')
        lo,hi=delta(a,b)
        for target,key in ((grouped,c['dataset']),(families,c['family'])):
            target[key][0]+=lo;target[key][1]+=hi;target[key][2]+=1
        wins+=a is True and b is False;losses+=a is False and b is True
        possible_wins+=(a is not False and b is not True)
        possible_losses+=(a is not True and b is not False)
        unknown+=a is None or b is None
    lower=sum(v[0] for v in grouped.values());upper=sum(v[1] for v in grouped.values())
    fl=sum(v[0]/v[2] for v in families.values())/len(families) if families else 0
    fu=sum(v[1]/v[2] for v in families.values())/len(families) if families else 0
    # 2*Delta_dataset <= Delta_total is equivalent to the <=50% criterion
    # whenever Delta_total>0. Bounds preserve covariance with the total:
    # inside deltas have coefficient +1, outside coefficient -1.
    concentration={k:{'net_lower':v[0],'net_upper':v[1],'planned':v[2],
        'twice_dataset_minus_total_lower':v[0]-(upper-v[1]),
        'twice_dataset_minus_total_upper':v[1]-(lower-v[0])} for k,v in grouped.items()}
    concentration_pass=lower>0 and all(v['twice_dataset_minus_total_upper']<=0 for v in concentration.values())
    ratio_pass=wins>=2*possible_losses and wins>0
    conditions={'net_gain_at_least_20_robust':lower>=net_threshold,
        'paired_discordances_at_least_two_to_one_robust':ratio_pass,
        'no_dataset_above_half_total_net_gain_robust':concentration_pass,
        'declared_family_macro_sign_positive_robust':fl>0}
    if all(conditions.values()):decision='positive_under_preregistered_rule'
    elif unknown==0:decision='negative_under_preregistered_rule'
    elif upper<net_threshold or possible_wins<2*losses or fu<=0 or (
         lower>0 and any(v['twice_dataset_minus_total_lower']>0 for v in concentration.values())):
        decision='negative_under_preregistered_rule_for_every_possible_completion'
    else:decision='inconclusive_due_to_indeterminate_or_missing_observations'
    return {'comparison':'H0 minus X1: hybrid minus cached inequalities, common controller',
        'threshold_net_certificates':net_threshold,'net_gain_bounds':[lower,upper],
        'known_wins':wins,'known_losses':losses,'possible_wins_upper':possible_wins,
        'possible_losses_upper':possible_losses,'unknown_pairs':unknown,
        'known_discordance_ratio':None if losses==0 else wins/losses,
        'known_discordance_ratio_kind':'infinite' if wins and not losses else 'undefined' if not losses else 'finite',
        'family_macro_delta_bounds':[fl,fu],'dataset_contributions':concentration,
        'conditions':conditions,'decision':decision}

def secondary(cases,records):
    rows=[];origins=[]
    for i,a in enumerate(CELLS):
        for b in CELLS[i+1:]:
            gaps=[];times=[];n=0
            for c in cases:
                x=records.get((c['case_id'],a),{});y=records.get((c['case_id'],b),{})
                if x.get('strict') is not True or y.get('strict') is not True:continue
                n+=1
                fa,fb=finite(x.get('objective')),finite(y.get('objective'))
                if fa is not None and fb is not None:gaps.append((fa-fb)/max(1,abs(fa),abs(fb)))
                ta,tb=finite(x.get('seconds_process')),finite(y.get('seconds_process'))
                if ta is not None and tb is not None and ta>1 and tb>1:times.append(ta/tb)
            for tol in (1e-3,1e-2):
                rows.append({'left':a,'right':b,'common_strict':n,'objective_pairs':len(gaps),
                    'symmetric_objective_threshold':tol,'left_lower':sum(v < -tol for v in gaps),
                    'equivalent':sum(abs(v)<=tol for v in gaps),'left_higher':sum(v>tol for v in gaps),
                    'both_process_times_over_1s_pairs':len(times),'median_process_time_ratio':median(times)})
    for cell in CELLS:
        counts=Counter();available=0;missing=0
        for c in cases:
            row=records.get((c['case_id'],cell),{})
            events=row.get('accepted_direction_origins')
            if not isinstance(events,dict):missing+=1;continue
            if any(type(v) is not int or v<0 for v in events.values()):raise ValueError('Invalid origin counts')
            available+=1;counts.update(events)
        denominator=sum(counts.values())
        # A SOC/retraction descendant is deliberately not folded into pure majorant.
        numerator=counts.get('majorant_primary',0)+counts.get('majorant',0)
        origins.append({'cell':cell,'logged_tasks':available,'missing_task_provenance':missing,
            'accepted_updates_identified':denominator,'all_final_origin_counts':dict(counts),
            'pure_majorant_fraction_on_logged_updates':numerator/denominator if denominator else None,
            'complete_campaign_fraction_identified':missing==0 and denominator>0})
    return rows,origins

def analyse(cases,records,threshold=20):
    groups={'I+M':cases,**{g:[c for c in cases if c['regime']==g] for g in ('I','M')}}
    result={};secondary_rows=[]
    for name,group in groups.items():
        sec,orig=secondary(group,records)
        for row in sec:secondary_rows.append({'regime':name,'stratum_type':'all','stratum':'all',**row})
        result[name]={'planned':len(group),'outcomes':totals(group,records),
          'effects':{k:contrast(group,records,v) for k,v in CONTRASTS.items()},
          'primary_rule':primary(group,records,threshold),'accepted_direction_origins':orig}
        composition={}
        for partition in ('dataset','family'):
            strata=defaultdict(list)
            for c in group:strata[c[partition]].append(c)
            output={}
            for key,members in sorted(strata.items()):
                pairwise,origin_counts=secondary(members,records)
                for row in pairwise:secondary_rows.append({'regime':name,'stratum_type':partition,'stratum':key,**row})
                output[key]={'planned':len(members),'outcomes':totals(members,records),
                  'effects':{k:contrast(members,records,v) for k,v in CONTRASTS.items()},
                  'accepted_direction_origins':origin_counts}
            macro={}
            for effect in CONTRASTS:
                lower=[v['effects'][effect]['lower_percentage_points'] for v in output.values()]
                upper=[v['effects'][effect]['upper_percentage_points'] for v in output.values()]
                macro[effect]={'lower_percentage_points':sum(lower)/len(lower) if lower else None,
                    'upper_percentage_points':sum(upper)/len(upper) if upper else None}
            composition[partition]={'strata':output,'equal_weight_macro_effects':macro}
        result[name]['composition']=composition
    return result,secondary_rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--preregistration',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();contract=load(a.contract);pre=load(a.preregistration)
    cases=[c for c in contract['cases'] if c['regime'] in ('I','M')]
    if len(cases)!=1834 or Counter(c['regime'] for c in cases)!=Counter({'I':1037,'M':797}):raise ValueError('Full1834 I/M contract required')
    if sha(a.contract)!=pre['inputs']['contract_file_sha256']:raise ValueError('Preregistered corpus changed')
    expected={c['case_id']:c for c in cases};records={};issues=[];run_ids=set();pre_sha=sha(a.preregistration)
    for line,row in enumerate(observations(a.observations),1):
        cell=ALIASES.get(row.get('cell',row.get('method',row.get('configuration'))))
        if cell is None:continue # Independently rerun Ipopt observations are not factorial cells.
        case=row.get('case_id',row.get('instance'));key=(case,cell)
        if case not in expected:raise ValueError('Out-of-scope case '+str(case))
        if key in records:raise ValueError('Duplicate cell '+str(key))
        c=expected[case];r=dict(row);valid=True
        tests=[(row.get('preregistration_sha256')==pre_sha,'preregistration hash'),
          (row.get('input_sha256')==c['input_sha256'],'input hash'),
          (row.get('start_sha256')==c['start_binding']['start_sha256'],'raw-start binding'),
          (flag(row.get('objective_exact')) is True,'actual exact-objective evidence'),
          (flag(row.get('budget_verified')) is True,'shared-budget evidence'),
          (flag(row.get('trace_success')) is True,'accepted-trace invariant evidence'),
          (bool(row.get('run_id')),'contemporary run identity')]
        for ok,label in tests:
            if not ok:issues.append({'line':line,'case_id':case,'cell':cell,'issue':label+' unavailable or different'});valid=False
        if row.get('run_id'):run_ids.add(row['run_id'])
        r['strict']=flag(row.get('strict_qcqp')) if valid else None
        if r['strict'] is not None:
            residuals=[finite(row.get(k)) for k in ('primal','dual','complementarity')]
            if any(v is None or v<0 for v in residuals) or (
                r['strict'] is True and max(residuals)>1e-6) or (
                r['strict'] is False and max(residuals)<=1e-6):
                issues.append({'line':line,'case_id':case,'cell':cell,
                               'issue':'Original residuals absent or inconsistent with certification'})
                r['strict']=None
            if flag(row.get('censored')) is True:
                issues.append({'line':line,'case_id':case,'cell':cell,
                               'issue':'Censored task'})
                r['strict']=None
        records[key]=r
    if len(run_ids)>1:raise ValueError('Historical and new runs cannot be mixed in the factorial analysis')
    result,sec=analyse(cases,records,pre['primary']['minimum_net_certificates'])
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    report={'schema':'native-factorial-preregistered-analysis-v1','preregistration_sha256':pre_sha,
      'observations_sha256':sha(a.observations),'run_ids':sorted(run_ids),'rows_received':len(records),
      'missing_cells':len(cases)*4-len(records),'verification_issues':issues,'results':result,
      'decision_scope':'Only I+M is confirmatory; I/M separate results are descriptive. Unknown outcomes are never imputed as failures.',
      'optimizer_invocations':0}
    (out/'FACTORIAL_REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    writecsv(out/'SECONDARY_PAIRED.csv',sec)
    print(json.dumps({'primary_I_M':result['I+M']['primary_rule']['decision'],
      'net_gain_bounds':result['I+M']['primary_rule']['net_gain_bounds'],'received':len(records),
      'missing':report['missing_cells'],'binding_issues':len(issues)},indent=2))
if __name__=='__main__':main()
