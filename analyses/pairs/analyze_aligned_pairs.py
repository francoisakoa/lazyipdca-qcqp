#!/usr/bin/env python3
"""Descriptive pairings of archived original-certified observations; no optimizer."""
from pathlib import Path
import argparse, collections, csv, gzip, hashlib, itertools, json, math, shutil, zipfile

REGIMES=('I','M','E','Q')
POLICIES=('R3','HH','HS','SH','SS','SS_legacy')
BACKENDS=('ma27','mumps')
STRATEGIES=('monotone','adaptive')
MEMBER='DEPOUILLEMENT/DEPOUILLEMENT/OBSERVATIONS_ANALYSE_COMPACTES.jsonl.gz'
NATIVES=tuple(p+'_'+b for b in BACKENDS for p in POLICIES)
IPOPTS=tuple('ipopt_'+b+'_'+s+'_aligned' for s in STRATEGIES for b in BACKENDS)
CONFIGS=NATIVES+IPOPTS

def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read_rows(path):
    with gzip.open(path,'rt')as f:return [json.loads(s)for s in f if s.strip()]
def q(v,p):
    if not v:return None
    a=sorted(v);z=(len(a)-1)*p;i=int(z)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(z-i)
def finite(x):return type(x)in(int,float)and math.isfinite(x)
def strict(r):
    if r['classification']!='strict':return False
    assert all(finite(r.get(k))and 0<=r[k]<=1e-6 for k in ['primal','dual','complementarity']),(r['case_id'],r['config'])
    return True
def signed(r,s):
    a,b=r.get('objective'),s.get('objective')
    if not(finite(a)and finite(b)):return None
    return (a-b)/max(1.,abs(a),abs(b))
def valid_time(r):
    x=r.get('seconds_process');return x if finite(x)and x>0 else None
def start(r):return r.get('start_sha256')or r.get('start_binding',{}).get('start_sha256')
def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='')as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True)if isinstance(v,(dict,list))else v for k,v in r.items()})
def metrics(gaps):
    a=[abs(x)for x in gaps]
    d=dict(objective_pairs=len(gaps),median_gap=q(gaps,.5),Q1_gap=q(gaps,.25),Q3_gap=q(gaps,.75),median_absolute_gap=q(a,.5),Q1_absolute_gap=q(a,.25),Q3_absolute_gap=q(a,.75),minimum_gap=min(gaps)if gaps else None,maximum_gap=max(gaps)if gaps else None)
    for tol,tag in [(1e-3,'1e_3'),(1e-2,'1e_2')]:
        d.update({tag+'_A_lower':sum(x< -tol for x in gaps),tag+'_equivalent':sum(abs(x)<=tol for x in gaps),tag+'_B_lower':sum(x>tol for x in gaps)})
    return d
def ratios_metrics(ratios,prefix):
    return {prefix+'_N':len(ratios),prefix+'_median':q(ratios,.5),prefix+'_Q1':q(ratios,.25),prefix+'_Q3':q(ratios,.75),prefix+'_geometric_mean':math.exp(sum(map(math.log,ratios))/len(ratios))if ratios else None}
def pair_summary(index,ids,A,B,scope,write_cases=None):
    counts=collections.Counter();gaps=[];times=[];long=[];equiv_long={1e-3:[],1e-2:[]};resids={x+'_'+k:[]for x in ['A','B']for k in ['primal','dual','complementarity']}
    for cid in ids:
        a,b=index[A][cid],index[B][cid];sa,sb=strict(a),strict(b)
        assert a['input_sha256']==b['input_sha256']and start(a)==start(b),(cid,A,B,'input/start')
        assert a['regime']==b['regime']and a['family']==b['family']and a['dataset']==b['dataset'],(cid,A,B,'metadata')
        counts['both'if sa and sb else'A_only'if sa else'B_only'if sb else'neither']+=1
        counts['A_indeterminate']+=a['classification']=='indeterminate';counts['B_indeterminate']+=b['classification']=='indeterminate'
        if not(sa and sb):continue
        for x,row in [('A',a),('B',b)]:
            for k in ['primal','dual','complementarity']:resids[x+'_'+k].append(row[k])
        delta=signed(a,b)
        if delta is not None:gaps.append(delta)
        ta,tb=valid_time(a),valid_time(b)
        if ta is not None and tb is not None:
            ratio=ta/tb;times.append(ratio)
            if ta>1 and tb>1:
                long.append(ratio)
                if delta is not None:
                    for tol in equiv_long:
                        if abs(delta)<=tol:equiv_long[tol].append(ratio)
        if write_cases is not None:
            write_cases.append(dict(case_id=cid,regime=a['regime'],dataset=a['dataset'],family=a['family'],A=A,B=B,input_sha256=a['input_sha256'],start_sha256=start(a),objective_A=a.get('objective'),objective_B=b.get('objective'),signed_gap=delta,absolute_gap=abs(delta)if delta is not None else None,seconds_A=ta,seconds_B=tb))
    r=dict(A=A,B=B,scope=scope,planned=len(ids),both=counts['both'],A_only=counts['A_only'],B_only=counts['B_only'],neither=counts['neither'],A_indeterminate=counts['A_indeterminate'],B_indeterminate=counts['B_indeterminate'],**metrics(gaps))
    r.update(ratios_metrics(times,'process_ratio_A_over_B'));r.update(ratios_metrics(long,'process_ratio_both_over_1s'))
    for tol,ratios in equiv_long.items():r.update(ratios_metrics(ratios,'process_ratio_both_over_1s_objective_equivalent_'+('1e_3'if tol==1e-3 else'1e_2')))
    for key,vals in resids.items():
        for tag,p in [('Q1',.25),('median',.5),('Q3',.75)]:r['residual_'+key+'_'+tag]=q(vals,p)
    assert sum(r[k]for k in ['both','A_only','B_only','neither'])==len(ids)
    return r

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path);ap.add_argument('--compact',type=Path);ap.add_argument('--legacy-compact',type=Path);ap.add_argument('--native-details',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if (a.archive is None)==(a.compact is None):raise ValueError('Choose exactly one --archive or --compact')
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    if a.archive:
        with zipfile.ZipFile(a.archive)as z:data=z.read(MEMBER)
        source=out/'OBSERVATIONS_16_CONFIGS.jsonl.gz';source.write_bytes(data)
    else:
        source=out/'OBSERVATIONS_16_CONFIGS.jsonl.gz'
        if source.resolve()!=a.compact.resolve():shutil.copy2(a.compact,source)
    rows=read_rows(source);index={c:{}for c in CONFIGS}
    for r in rows:
        assert r['config']in index,r['config'];assert r['case_id']not in index[r['config']]
        assert r['classification']in ['strict','nonstrict','indeterminate']
        assert isinstance(start(r),str)and len(start(r))==64
        assert isinstance(r['input_sha256'],str)and len(r['input_sha256'])==64
        index[r['config']][r['case_id']]=r
    assert len(rows)==16*4078
    ids=sorted(index[CONFIGS[0]])
    for c in CONFIGS:assert sorted(index[c])==ids
    source_validation={}
    if a.legacy_compact:
        old={}
        for r in read_rows(a.legacy_compact):
            if r['config']in NATIVES:old[r['config'],r['case_id']]=r
        changes=[]
        for c in NATIVES:
            for cid,r in index[c].items():
                s=old[c,cid]
                for k in ['classification','input_sha256','start_sha256','native_status','objective','primal','dual','complementarity','seconds_process','profile_sha256','binary_sha256']:
                    if s.get(k)!=r.get(k):changes.append(dict(config=c,case_id=cid,field=k,old=s.get(k),new=r.get(k)))
        source_validation=dict(legacy_compact_sha256=sha(a.legacy_compact),native_compared=12*4078,changed_fields=changes)
        assert not changes,changes[:5]
    metadata=index[NATIVES[0]]
    scopes={'all':ids,**{s:[i for i in ids if metadata[i]['regime']==s]for s in REGIMES},
        'M_ASU':[i for i in ids if metadata[i]['regime']=='M'and metadata[i]['dataset']=='beam_structural'],
        'M_outside_ASU':[i for i in ids if metadata[i]['regime']=='M'and metadata[i]['dataset']!='beam_structural']}
    coverage=[];collections_rows=[];composition=[]
    for c in CONFIGS:
        for regime in REGIMES:
            subset=scopes[regime];d=dict(config=c,regime=regime,planned=len(subset));counts=collections.Counter(index[c][i]['classification']for i in subset)
            d.update(strict=counts['strict'],nonstrict=counts['nonstrict'],indeterminate=counts['indeterminate'],failed_to_certify=len(subset)-counts['strict'],coverage=counts['strict']/len(subset));coverage.append(d)
            partitions={}
            for key in ['family','dataset']:
                groups=collections.defaultdict(list)
                for i in subset:groups[metadata[i][key]].append(i)
                partitions[key]=groups
                macro=sum(sum(strict(index[c][i])for i in members)/len(members)for members in groups.values())/len(groups)
                largest=sorted(groups,key=lambda x:(-len(groups[x]),x))[0];rest=[i for i in subset if metadata[i][key]!=largest]
                composition.append(dict(config=c,regime=regime,partition=key,groups=len(groups),instance_rate=d['coverage'],equal_group_rate=macro,largest_group=largest,largest_group_size=len(groups[largest]),outside_largest_N=len(rest),outside_largest_strict=sum(strict(index[c][i])for i in rest),outside_largest_rate=sum(strict(index[c][i])for i in rest)/len(rest)if rest else None))
            for name,members in partitions['dataset'].items():
                cc=collections.Counter(index[c][i]['classification']for i in members)
                collections_rows.append(dict(config=c,regime=regime,dataset=name,planned=len(members),strict=cc['strict'],nonstrict=cc['nonstrict'],indeterminate=cc['indeterminate']))
    pairings=list(itertools.product(NATIVES,IPOPTS))
    pairings+=[(p+'_'+b,q+'_'+b)for b in BACKENDS for p,q in itertools.combinations(POLICIES,2)]
    pairings+=[(p+'_ma27',p+'_mumps')for p in POLICIES]
    pairings+=list(itertools.combinations(IPOPTS,2))
    summaries=[];paired=[]
    for A,B in pairings:
        for scope,subset in scopes.items():summaries.append(pair_summary(index,subset,A,B,scope,paired if A in NATIVES and B in IPOPTS and scope in REGIMES else None))
    write_json(out/'PAIR_SUMMARIES.json',summaries);csv_write(out/'PAIR_SUMMARIES.csv',summaries)
    csv_write(out/'COVERAGE_BY_REGIME.csv',coverage);csv_write(out/'COVERAGE_BY_COLLECTION.csv',collections_rows);csv_write(out/'COMPOSITION.csv',composition)
    with gzip.open(out/'PAIRED_OBJECTIVES_AND_TIMES.csv.gz','wt',newline='')as f:
        w=csv.DictWriter(f,list(paired[0]));w.writeheader();w.writerows(paired)
    confirmed=[];undetermined=[];native_details={}
    if a.native_details:
        detail_source=json.loads(a.native_details.read_text())
        native_details={r['case_id']:r for r in detail_source['records']}
        source_validation['native_details_sha256']=sha(a.native_details)
        source_validation['native_details_original_observations_sha256']=detail_source['source_sha256']
    for b in BACKENDS:
        A='R3_'+b
        for ref_backend,strategy in itertools.product(BACKENDS,STRATEGIES):
            B='ipopt_'+ref_backend+'_'+strategy+'_aligned'
            for cid in scopes['I']:
                x,y=index[A][cid],index[B][cid]
                if strict(x)or not strict(y):continue
                detail=native_details.get(cid)if b=='ma27'else None
                if detail is not None:
                    assert detail['input_sha256']==x['input_sha256']and detail['start_sha256']==start(x)
                    assert detail['result_sha256']==x['result_sha256']and detail['classification']==x['classification']
                    assert detail['budget_verified']is True
                work=detail.get('internal_work',{})if detail else{}
                r=dict(case_id=cid,dataset=x['dataset'],family=x['family'],regime='I',native_config=A,ipopt_config=B,
                    native_backend=b,ipopt_backend=ref_backend,same_backend=b==ref_backend,
                    native_classification=x['classification'],native_status=x['native_status'],ipopt_status=y['native_status'],
                    native_primal=x.get('primal'),native_dual=x.get('dual'),native_complementarity=x.get('complementarity'),
                    ipopt_primal=y.get('primal'),ipopt_dual=y.get('dual'),ipopt_complementarity=y.get('complementarity'),
                    native_seconds=x.get('seconds_process'),ipopt_seconds=y.get('seconds_process'),
                    native_iterations=detail.get('iterations')if detail else x.get('iterations'),ipopt_iterations=y.get('iterations'),
                    native_restoration_iterations=work.get('restoration_iterations'),native_factorizations=work.get('kkt_factorizations'),native_linear_solves=work.get('lazy_newton_linear_solves'),
                    native_budget_verified_detail=detail.get('budget_verified')if detail else None,
                    input_sha256=x['input_sha256'],start_sha256=start(x),
                    native_proof_level=x.get('proof_level'),ipopt_proof_level=y.get('proof_level'),
                    cause_of_native_failure='Source terminal status and available exported work counters; native endpoint error message/direction trace not supplied',
                    scientific_scope='Archived original-certificate comparison, identical frozen input/start and declared nominal budgets; no new endpoint audit')
                (confirmed if x['classification']=='nonstrict'else undetermined).append(r)
    csv_write(out/'R3_IPOPT_REGIME_I_CONFIRMED_PAIRS.csv',confirmed);csv_write(out/'R3_IPOPT_REGIME_I_INDETERMINATE_NATIVE.csv',undetermined)
    csv_write(out/'R3_IPOPT_REGIME_I_CONFIRMED_SAME_BACKEND.csv',[r for r in confirmed if r['same_backend']])
    ix={(r['A'],r['B'],r['scope']):r for r in summaries}
    def fmt(x,n=2):
        if x is None:return '--'
        if abs(x)<.5*10**(-n):x=0.
        return f'{x:.{n}f}'
    def trip(r,tag='1e_3'):return '/'.join(str(r[tag+'_'+k])for k in ['A_lower','equivalent','B_lower'])
    def tabular(headers,body,spec):return '\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,row))+'\\\\'for row in body)+'\n\\bottomrule\n\\end{tabular}\n'
    by_cov={(r['config'],r['regime']):r for r in coverage}
    body=[]
    for c in CONFIGS:
        if c in NATIVES:p=c.rsplit('_',1)[0];b=c.rsplit('_',1)[1].upper();label='$SS_0$'if p=='SS_legacy'else p
        else:p=c.split('_');label='Ipopt '+p[2];b=p[1].upper()
        nums=[by_cov[c,r]['strict']for r in REGIMES];body.append([label,b,*nums,sum(nums)])
    (out/'coverage_all_16.tex').write_text(tabular(['Policy','Linear solver','I','M','E','Q','All'],body,'llrrrrr'))
    body=[];cross=[];time_body=[];dist=[]
    for b in BACKENDS:
        B='ipopt_'+b+'_monotone_aligned'
        for p in POLICIES:
            A=p+'_'+b;ri,rm=ix[A,B,'I'],ix[A,B,'M'];label='$SS_0$'if p=='SS_legacy'else p
            body.append([b.upper(),label,ri['objective_pairs'],trip(ri),rm['objective_pairs'],trip(rm)])
            time_body.append([b.upper(),label,ri['process_ratio_both_over_1s_N'],fmt(ri['process_ratio_both_over_1s_median']),rm['process_ratio_both_over_1s_N'],fmt(rm['process_ratio_both_over_1s_median'])])
            da,dr=ix[A,B,'M_ASU'],ix[A,B,'M_outside_ASU']
            med=lambda d:fmt(100*d['median_gap'])+' ['+fmt(100*d['Q1_gap'])+','+fmt(100*d['Q3_gap'])+']'if d['median_gap']is not None else'--'
            dist.append([b.upper(),label,da['objective_pairs'],med(da),dr['objective_pairs'],med(dr)])
        for strategy in STRATEGIES:
            r=ix['R3_'+b,'ipopt_'+b+'_'+strategy+'_aligned','I'];cross.append([b.upper(),strategy,r['both'],r['A_only'],r['B_only'],r['neither']])
    (out/'objectives_native_vs_monotone.tex').write_text(tabular(['Linear solver','Native','N$_I$','I: L/E/H','N$_M$','M: L/E/H'],body,'llrlrl'))
    (out/'timing_native_vs_monotone_over1s.tex').write_text(tabular(['Linear solver','Native','N$_I$','I median ratio','N$_M$','M median ratio'],time_body,'llrrrr'))
    (out/'objectives_M_ASU_rest_monotone.tex').write_text(tabular(['Linear solver','Native','N$_{ASU}$','ASU gap (\\%)','N$_{rest}$','Rest gap (\\%)'],dist,'llrcrc'))
    (out/'crossed_R3_I_references.tex').write_text(tabular(['Linear solver','Barrier','Both','R3 only','Ipopt only','Neither'],cross,'llrrrr'))
    body=[]
    for regime in REGIMES:
        for policy in POLICIES:
            r=ix[policy+'_ma27',policy+'_mumps',regime]
            body.append([regime,'$SS_0$'if policy=='SS_legacy'else policy,r['both'],r['A_only'],r['B_only'],r['neither']])
    (out/'native_backend_crossed.tex').write_text(tabular(['Regime','Policy','Both','MA27 only','MUMPS only','Neither'],body,'llrrrr'))
    macro={(r['config'],r['regime'],r['partition']):r for r in composition}
    for b in BACKENDS:
        body=[]
        for regime in REGIMES:
            for config in [p+'_'+b for p in POLICIES]+['ipopt_'+b+'_'+s+'_aligned'for s in STRATEGIES]:
                label=config[:-len('_'+b)]if config in NATIVES else'Ipopt '+config.split('_')[2]
                if label=='SS_legacy':label='$SS_0$'
                x,y=macro[config,regime,'family'],macro[config,regime,'dataset']
                body.append([regime,label,fmt(100*x['instance_rate'],1),fmt(100*x['equal_group_rate'],1),fmt(100*y['equal_group_rate'],1),str(y['outside_largest_strict'])+'/'+str(y['outside_largest_N'])])
        (out/('composition_'+b+'_aligned.tex')).write_text(tabular(['Regime','Configuration','Instance (\\%)','Family (\\%)','Collection (\\%)','Outside largest'],body,'llrrrl'))
    result=dict(schema='native-versus-aligned-pair-analysis-v1',rows=len(rows),cases=len(ids),configs=list(CONFIGS),scope_sizes={k:len(v)for k,v in scopes.items()},
        source_compact_sha256=sha(source),archive_sha256=sha(a.archive)if a.archive else None,source_validation=source_validation,
        aligned_internal_status_not_used=True,endpoint_reaudits=0,optimizer_invocations=0,
        strict_definition='Archived strict classification plus finite nonnegative original primal/dual/complementarity residuals <=1e-6. The independent audit agent checks archived RESULT/AUDIT/provenance separately.',
        pairs_hash_checked=True,case_metadata_checked=True,objective_definition='(f_A-f_B)/max(1,abs(f_A),abs(f_B)); positive favors B',
        timing_definition='Declared solver-process elapsed A/B; audit excluded; no startup subtraction; >1s cohort requires both declared process times >1s',
        R3_I_archived_confirmed_pairs=len(confirmed),R3_I_unknown_native_with_Ipopt_strict=len(undetermined),
        native_iterations_missing_in_compact=True,full_native_endpoint_traces_not_included=True)
    write_json(out/'VALIDATION.json',result)
    print(json.dumps({k:result[k]for k in ['rows','cases','pairs_hash_checked','R3_I_archived_confirmed_pairs','R3_I_unknown_native_with_Ipopt_strict']},indent=2))
if __name__=='__main__':main()
