#!/usr/bin/env python3
"""Export a hash-bound snapshot of this campaign, without invoking an optimizer."""
from __future__ import annotations
import argparse,collections,csv,datetime,gzip,hashlib,itertools,json,math,shutil,statistics,subprocess,sys,zipfile
from pathlib import Path
import CAMPAGNE as campaign

HERE=Path(__file__).resolve().parent
CELLS=('R3','X1','X2','H0')
DEFAULTS=tuple(m for m,_ in campaign.METHODS)
RESIDUALS=('primal','dual','complementarity')
CSV_FIELDS=('run_id','case_id','dataset','regime','family','method','cell','backend','classification',
    'strict_qcqp','status','ipopt_application_status','ipopt_application_status_code',
    'ipopt_internal_success','ipopt_internal_acceptable','ipopt_internal_success_or_acceptable',
    'primal','dual','complementarity','objective','seconds_process','audit_seconds','iterations',
    'solver_user_cpu_seconds','solver_system_cpu_seconds','solver_peak_rss_kib','solver_returncode',
    'censored','budget_verified','objective_exact','trace_success','input_sha256','start_sha256',
    'binary_sha256','installation_sha256','preregistration_sha256','result_sha256')

def digest(path):return campaign.sha(path)
def reject_constant(value):raise ValueError('Non-finite JSON number: '+value)
def load(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream,parse_constant=reject_constant)
def load_contract(path):
    path=Path(path)
    with (gzip.open(path,'rt',encoding='utf-8') if path.suffix=='.gz' else path.open(encoding='utf-8')) as stream:
        return json.load(stream,parse_constant=reject_constant)
def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf-8')
def write_jsonl(path,rows):
    with Path(path).open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False,ensure_ascii=False)+'\n')
def write_csv(path,rows,fields=None):
    fields=list(fields or (list(rows[0]) if rows else []))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in rows:
            # Explicit null distinguishes unavailable measurements from zero.
            writer.writerow({key:'null' if row.get(key) is None else row[key] for key in fields})
def finite(value):
    if type(value) not in (float,int):return None
    return value if math.isfinite(value) else None
def med(values):return statistics.median(values) if values else None
def pick_path(value,run):
    if not value:return None
    path=Path(value);return path.resolve() if path.is_absolute() else (run/path).resolve()
def checked(path,expected,label):
    if path is None or not path.is_file():raise ValueError(label+' absent: '+str(path))
    if not expected or digest(path)!=expected:raise ValueError(label+' hash differs')
    return path

def configuration(run,contract_override=None,require_full=True):
    receipt=load(run/'RUN_CONTRACT.json')
    combined=receipt.get('schema')=='combined-factorial-default-full-run-contract-v1'
    if not combined and receipt.get('schema')!='ipopt-default-full-run-contract-v1':
        raise ValueError('Unsupported campaign contract schema')
    if not receipt.get('run_id'):raise ValueError('Contemporary run identity is absent')
    path=pick_path(contract_override or receipt.get('contract'),run)
    if path is None and (HERE/'CONTRAT_4078.json.gz').is_file():path=HERE/'CONTRAT_4078.json.gz'
    path=checked(path,receipt.get('contract_sha256'),'Frozen corpus contract')
    cases=campaign.plan_cases(load_contract(path),require_full=require_full)
    if receipt.get('cases')!=len(cases):raise ValueError('Plan and corpus case counts differ')
    methods=list(CELLS)+list(DEFAULTS) if combined else list(DEFAULTS)
    if set(receipt.get('methods',[]))!=set(methods):raise ValueError('Plan methods differ from the declared current campaign')
    planned={(c['case_id'],m):c for c in cases for m in methods if m not in CELLS or c['regime'] in ('I','M')}
    if receipt.get('tasks')!=len(planned):raise ValueError('Plan task count differs from the declared corpus')
    paths={'RUN_CONTRACT':run/'RUN_CONTRACT.json','corpus_contract':path}
    default_path=checked(pick_path(receipt.get('installation'),run),
        receipt.get('default_installation_sha256') if combined else receipt.get('installation_sha256'),'Default installation')
    default=load(default_path);paths['default_installation']=default_path
    for key in ('qualification','build','frozen_resolution'):
        if default.get(key):paths['default_'+key]=checked(pick_path(default[key],run),default.get(key+'_sha256'),'Default '+key)
    if default.get('contract') and digest(Path(default['contract']))!=receipt['contract_sha256']:
        raise ValueError('Installed and run corpus differ')
    fact=None
    if combined:
        fact_path=checked(pick_path(receipt.get('factorial_installation'),run),receipt.get('factorial_installation_sha256'),'Factorial installation')
        fact=load(fact_path);paths['factorial_installation']=fact_path
        prereg=checked(pick_path(receipt.get('preregistration') or fact.get('preregistration'),run),
                      receipt.get('preregistration_sha256') or fact.get('preregistration_sha256'),'Preregistration')
        if fact.get('preregistration_sha256')!=digest(prereg):raise ValueError('Preregistration differs between installations')
        registration=load(prereg)
        if registration['inputs']['contract_file_sha256']!=digest(path):raise ValueError('Preregistered corpus differs')
        gate=checked(pick_path(fact.get('qualified_50gate'),run),fact.get('qualified_50gate_sha256'),'Native qualification gate')
        if receipt.get('qualification_sha256')!=digest(gate):raise ValueError('Native qualification differs from run contract')
        paths.update(preregistration=prereg,native_qualification=gate)
        for key in ('coupling_proof','build'):
            if fact.get(key):paths['native_'+key]=checked(pick_path(fact[key],run),fact.get(key+'_sha256'),'Native '+key)
    return receipt,cases,methods,planned,paths,default,fact

def verify_record(row,case,method,receipt,default,fact):
    for key,expected in (('case_id',case['case_id']),('method',method),('dataset',case['dataset']),
                         ('regime',case['regime']),('run_id',receipt['run_id']),('input_sha256',case['input_sha256']),
                         ('start_sha256',case['start_binding'].get('start_sha256'))):
        if row.get(key)!=expected:raise ValueError('Historical or incorrectly paired record: '+key+' '+case['case_id']+' '+method)
    if row.get('start_binding')!=case['start_binding']:raise ValueError('Raw start binding differs: '+case['case_id'])
    if row.get('family')!=case.get('family'):raise ValueError('Family differs from the declared corpus')
    installation_sha=receipt.get('default_installation_sha256',receipt.get('installation_sha256'))
    if row.get('installation_sha256') not in (None,installation_sha):raise ValueError('Record installation differs')
    expected_binary=fact['binary_sha256'] if method in CELLS else default['default_binary_sha256']
    if row.get('binary_sha256') not in (None,expected_binary):raise ValueError('Record binary differs')
    if method in CELLS:
        if row.get('cell')!=method or row.get('preregistration_sha256')!=fact['preregistration_sha256']:
            raise ValueError('Factorial cell or preregistration differs')
    elif row.get('cell') is not None or row.get('preregistration_sha256') is not None:
        raise ValueError('Ipopt record incorrectly labeled as native factorial cell')
    strict=row.get('strict_qcqp');expected_class={True:'strict',False:'nonstrict',None:'indeterminate'}
    if strict is not None and type(strict) is not bool:raise ValueError('Outcome must be boolean or null')
    if row.get('classification')!=expected_class[strict]:raise ValueError('KKT class and three-valued outcome differ')
    if strict is not None:
        residuals=[finite(row.get(key)) for key in RESIDUALS]
        if any(v is None or v<0 for v in residuals):raise ValueError('Certified outcome without finite original residuals')
        if strict!=(max(residuals)<=1e-6):raise ValueError('Original residuals contradict the reported certificate')
        if row.get('censored') is not False or row.get('budget_verified') is not True:
            raise ValueError('Known certificate on censored or unverified-budget observation')
        audit=row.get('original_audit')
        if not isinstance(audit,dict) or audit.get('strict_qcqp') is not strict or audit.get('audit_available') is not True:
            raise ValueError('Known outcome lacks matching independent original audit')
        if audit.get('input_sha256')!=case['input_sha256']:raise ValueError('Original audit used a different input')
        if not row.get('binary_sha256') or not row.get('installation_sha256'):
            raise ValueError('Known certificate lacks binary or installation binding')
        if method in CELLS and (row.get('trace_success') is not True or row.get('objective_exact') is not True):
            raise ValueError('Known factorial outcome lacks actual trace and objective evidence')
    code=row.get('ipopt_application_status_code')
    if code is not None and type(code) is not int:raise ValueError('Ipopt application status code must be integer or null')
    if method in CELLS and code is not None:raise ValueError('Native observation contains an Ipopt application status')

def snapshot(run,receipt,planned,default,fact):
    records={};sources=[];warnings=[]
    expected_folders={campaign.safe_id(caseid+'__'+method):(caseid,method) for caseid,method in planned}
    if len(expected_folders)!=len(planned):raise ValueError('Task path collision')
    taskroot=run/'tasks'
    for path in sorted(taskroot.glob('*/RESULT.json')):
        key=expected_folders.get(path.parent.name)
        if key is None:raise ValueError('Unplanned result: '+str(path))
        row=load(path)
        if not isinstance(row,dict):raise ValueError('Invalid result object: '+str(path))
        verify_record(row,planned[key],key[1],receipt,default,fact)
        row=dict(row);row['result_sha256']=digest(path)
        records[key]=row;sources.append({'path':str(path.relative_to(run)),'sha256':row['result_sha256']})
    # Atomic RESULT receipts are authoritative. The append-only index can be
    # incomplete after a crash; a live producer may also append during export.
    index=run/'RESULTS.jsonl';index_count=0;index_keys=set();index_incomplete=False
    if index.is_file():
        with index.open('rb') as stream:
            for number,raw in enumerate(stream,1):
                if not raw.strip():continue
                try:row=json.loads(raw,parse_constant=reject_constant)
                except (ValueError,UnicodeError):
                    if not raw.endswith(b'\n') and stream.read(1)==b'':
                        warnings.append({'issue':'truncated_final_index_line','line':number});index_incomplete=True;break
                    raise ValueError('Invalid results index line '+str(number))
                key=(row.get('case_id'),row.get('method'));index_count+=1
                if key not in planned or row.get('run_id')!=receipt['run_id']:raise ValueError('Historical or unplanned results index row')
                if key in index_keys:warnings.append({'issue':'duplicate_index_row','case_id':key[0],'method':key[1]})
                index_keys.add(key)
                if key not in records:
                    warnings.append({'issue':'index_row_after_snapshot_or_without_atomic_result','case_id':key[0],'method':key[1]});continue
                authoritative=dict(records[key]);authoritative.pop('result_sha256',None)
                if row!=authoritative:raise ValueError('Append-only index disagrees with atomic RESULT: '+str(key))
        missing_index=set(records)-index_keys
        if missing_index:warnings.append({'issue':'atomic_results_missing_from_index','count':len(missing_index)})
    else:warnings.append({'issue':'append_only_index_absent','atomic_results_used':len(records)})
    return records,sources,warnings,dict(index_rows=index_count,index_distinct_rows=len(index_keys),truncated_last_line=index_incomplete)

def enrich(row):
    row=dict(row);code=row.get('ipopt_application_status_code');is_ipopt=row['method'] in DEFAULTS
    for key,value in (('ipopt_internal_success',code==0),('ipopt_internal_acceptable',code==1),
                      ('ipopt_internal_success_or_acceptable',code in (0,1))):
        row[key]=value if is_ipopt and code is not None else None
    resources=row.get('solver_resources') or {}
    for output,key in (('solver_user_cpu_seconds','user_cpu_seconds'),('solver_system_cpu_seconds','system_cpu_seconds'),
                       ('solver_peak_rss_kib','peak_rss_kib'),('solver_returncode','returncode')):row[output]=resources.get(key)
    return row

def summaries(planned,records,groupby):
    groups=collections.defaultdict(list)
    for key,case in planned.items():
        meta=dict(case,method=key[1]);group=tuple(meta[k] for k in groupby);groups[group].append(key)
    output=[]
    for group,keys in sorted(groups.items()):
        rows=[records[k] for k in keys if k in records];count=collections.Counter(r['classification'] for r in rows)
        known_ipopt=[r for r in rows if r.get('ipopt_internal_success_or_acceptable') is not None]
        strict=[r for r in rows if r['strict_qcqp'] is True]
        row={key:value for key,value in zip(groupby,group)}
        row.update(planned=len(keys),recorded=len(rows),strict=count['strict'],nonstrict=count['nonstrict'],
            indeterminate=count['indeterminate'],missing=len(keys)-len(rows),
            certificate_rate_over_planned=count['strict']/len(keys) if keys else None,
            ipopt_internal_status_available=len(known_ipopt),
            ipopt_success=sum(r['ipopt_internal_success'] is True for r in known_ipopt),
            ipopt_acceptable=sum(r['ipopt_internal_acceptable'] is True for r in known_ipopt),
            ipopt_success_or_acceptable_without_original_strict=sum(r['ipopt_internal_success_or_acceptable'] is True and r['strict_qcqp'] is not True for r in known_ipopt),
            original_strict_without_ipopt_success_or_acceptable=sum(r['strict_qcqp'] is True and r['ipopt_internal_success_or_acceptable'] is False for r in known_ipopt),
            median_strict_primal=med([r['primal'] for r in strict]),median_strict_dual=med([r['dual'] for r in strict]),
            median_strict_complementarity=med([r['complementarity'] for r in strict]))
        output.append(row)
    return output

def pairs(cases,methods,records):
    output=[];coverage=[];objectives=[];timings=[]
    for left,right in itertools.combinations(methods,2):
        rows=[];strata=collections.defaultdict(list);planned_counts=collections.Counter()
        for case in cases:
            if (left in CELLS or right in CELLS) and case['regime'] not in ('I','M'):continue
            for key in (('all','all'),('regime',case['regime']),('dataset',case['dataset'])):planned_counts[key]+=1
            a=records.get((case['case_id'],left));b=records.get((case['case_id'],right))
            sa=a.get('strict_qcqp') if a else None;sb=b.get('strict_qcqp') if b else None
            common=sa is True and sb is True
            fa=finite(a.get('objective')) if a else None;fb=finite(b.get('objective')) if b else None
            ta=finite(a.get('seconds_process')) if a else None;tb=finite(b.get('seconds_process')) if b else None
            gap=(fa-fb)/max(1,abs(fa),abs(fb)) if common and fa is not None and fb is not None else None
            ratio=ta/tb if common and ta is not None and tb is not None and ta>0 and tb>0 else None
            row=dict(case_id=case['case_id'],dataset=case['dataset'],regime=case['regime'],left=left,right=right,
                left_recorded=a is not None,right_recorded=b is not None,left_strict=sa,right_strict=sb,
                common_strict=common,left_objective=fa,right_objective=fb,symmetric_objective_gap=gap,
                left_seconds_process=ta,right_seconds_process=tb,process_time_ratio_left_over_right=ratio,
                both_times_over_1s=ratio is not None and ta>1 and tb>1)
            rows.append(row);output.append(row)
            for key in (('all','all'),('regime',case['regime']),('dataset',case['dataset'])):strata[key].append(row)
        for (kind,name),planned in sorted(planned_counts.items()):
            group=strata[kind,name];common=[r for r in group if r['common_strict']]
            base=dict(left=left,right=right,stratum_type=kind,stratum=name,planned_pairs=planned)
            coverage.append(dict(base,both_recorded=sum(r['left_recorded'] and r['right_recorded'] for r in group),
                both_known=sum(r['left_strict'] is not None and r['right_strict'] is not None for r in group),
                both_strict=len(common),left_only_confirmed=sum(r['left_strict'] is True and r['right_strict'] is False for r in group),
                right_only_confirmed=sum(r['left_strict'] is False and r['right_strict'] is True for r in group),
                both_nonstrict=sum(r['left_strict'] is False and r['right_strict'] is False for r in group),
                unknown_pair=sum(r['left_strict'] is None or r['right_strict'] is None for r in group)))
            gaps=[r['symmetric_objective_gap'] for r in common if r['symmetric_objective_gap'] is not None]
            for threshold in (1e-4,1e-3,1e-2):
                objectives.append(dict(base,common_strict=len(common),objective_pairs=len(gaps),threshold=threshold,
                    left_lower=sum(v < -threshold for v in gaps),equivalent=sum(abs(v)<=threshold for v in gaps),
                    left_higher=sum(v>threshold for v in gaps),median_signed_gap=med(gaps),median_absolute_gap=med([abs(v) for v in gaps]),
                    median_positive_gap_conditional=med([v for v in gaps if v>threshold]),
                    median_negative_gap_magnitude_conditional=med([-v for v in gaps if v < -threshold])))
            for cohort in ('common_strict_positive_process_times','common_strict_both_process_times_over_1s','common_strict_objective_equivalent_1e-3_both_times_over_1s'):
                eligible=[r for r in common if r['process_time_ratio_left_over_right'] is not None]
                if 'over_1s' in cohort:eligible=[r for r in eligible if r['both_times_over_1s']]
                if 'equivalent' in cohort:eligible=[r for r in eligible if r['symmetric_objective_gap'] is not None and abs(r['symmetric_objective_gap'])<=1e-3]
                ratios=[r['process_time_ratio_left_over_right'] for r in eligible]
                timings.append(dict(base,cohort=cohort,common_strict=len(common),paired_times=len(ratios),median_paired_process_time_ratio=med(ratios),
                    geometric_mean_paired_process_time_ratio=math.exp(sum(math.log(v) for v in ratios)/len(ratios)) if ratios else None,
                    median_left_seconds_process=med([r['left_seconds_process'] for r in eligible]),
                    median_right_seconds_process=med([r['right_seconds_process'] for r in eligible]),
                    clock='solver process wall elapsed; independent audit excluded; no startup subtraction'))
    return output,coverage,objectives,timings

def export(run,out,archive=False,contract_override=None,require_full=True):
    run=Path(run).resolve();out=Path(out).resolve()
    if out.exists():raise ValueError('Output already exists: choose a new directory')
    receipt,cases,methods,planned,paths,default,fact=configuration(run,contract_override,require_full)
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw,sources,warnings,index=snapshot(run,receipt,planned,default,fact)
    records={key:enrich(row) for key,row in raw.items()}
    rows=[records[key] for key in sorted(records)]
    out.mkdir(parents=True)
    write_jsonl(out/'OBSERVATIONS.jsonl',rows);write_csv(out/'OBSERVATIONS.csv',rows,CSV_FIELDS)
    write_jsonl(out/'ORIGINAL_AUDIT_PROOFS.jsonl',[
        dict(run_id=r['run_id'],case_id=r['case_id'],method=r['method'],result_sha256=r['result_sha256'],
             original_audit=r.get('original_audit'),provenance=r.get('provenance'),effective_options=r.get('effective_options'),
             internal_work=r.get('internal_work'),solver_resources=r.get('solver_resources'),audit_resources=r.get('audit_resources')) for r in rows])
    groups={'GLOBAL':(), 'BY_METHOD':('method',),'BY_REGIME':('regime','method'),
            'BY_DATASET':('dataset','method'),'BY_DATASET_REGIME':('dataset','regime','method')}
    summary={}
    for name,fields in groups.items():
        values=summaries(planned,records,fields);summary[name]=values;write_csv(out/(name+'.csv'),values)
    paired,coverage,objectives,timings=pairs(cases,methods,records)
    write_csv(out/'PAIRED_OBSERVATIONS.csv',paired)
    write_csv(out/'COVERAGE_PAIRED.csv',coverage)
    write_csv(out/'OBJECTIVES_PAIRED.csv',objectives)
    write_csv(out/'TIMINGS_PAIRED.csv',timings)
    write_jsonl(out/'MISSING_PLANNED.jsonl',[dict(case_id=k[0],method=k[1],dataset=c['dataset'],regime=c['regime'],
        input_sha256=c['input_sha256'],start_sha256=c['start_binding'].get('start_sha256'),strict_qcqp=None,
        classification='missing_planned_observation') for k,c in sorted(planned.items()) if k not in records])
    source_dir=out/'evidence';source_dir.mkdir()
    copied=[]
    for label,path in paths.items():
        target=source_dir/(label+(''.join(path.suffixes) or '.json'))
        shutil.copyfile(path,target);copied.append(dict(label=label,source=str(path),file=str(target.relative_to(out)),sha256=digest(target)))
    factor_report=None
    if fact:
        analyzer=HERE.parent/'factorial/analysis/ANALYSER_FACTORIEL.py'
        if not analyzer.is_file():raise ValueError('Bundled preregistered factorial analyzer is absent')
        result=subprocess.run([sys.executable,str(analyzer),'--contract',str(paths['corpus_contract']),
            '--preregistration',str(paths['preregistration']),'--observations',str(out/'OBSERVATIONS.jsonl'),
            '--out',str(out/'factorial')],capture_output=True,text=True)
        (out/'FACTORIAL_ANALYSIS.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        if result.returncode:raise ValueError('Preregistered factorial analysis failed; see FACTORIAL_ANALYSIS.log')
        factor_report=load(out/'factorial/FACTORIAL_REPORT.json')
        # Incomplete task evidence remains reported as unknown by the analyzer.
        # Its binding warnings are retained verbatim, never silently dropped.
        copied.append(dict(label='factorial_analyzer',source=str(analyzer),sha256=digest(analyzer)))
    manifest=dict(schema='campaign-current-observations-extraction-v1',run_id=receipt['run_id'],run=str(run),
        snapshot_started_utc=start,snapshot_finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        corpus_cases=len(cases),planned_tasks=len(planned),recorded_tasks=len(rows),missing_tasks=len(planned)-len(rows),
        methods=methods,regimes=['I','M','E','Q'],optimizer_invocations=0,source_policy='Atomic per-task RESULT.json receipts; append-only index checked separately',
        null_policy='Missing planned observations and indeterminate completed tasks remain distinct; neither is a numerical failure.',
        objective_gap='(f_left-f_right)/max(1,abs(f_left),abs(f_right)); minimization; only common original-KKT-strict endpoints',
        timing_scope='Solver process wall elapsed; independent audit excluded. Paired ratios, no startup subtraction. Concurrent campaign timings are descriptive.',
        source_bindings=copied,result_files=sources,index=index,warnings=warnings,
        factorial_analysis=None if factor_report is None else dict(report='factorial/FACTORIAL_REPORT.json',
            verification_issues=factor_report['verification_issues'],rows_received=factor_report['rows_received'],missing_cells=factor_report['missing_cells']))
    write_json(out/'SUMMARY.json',summary);write_json(out/'MANIFEST.json',manifest)
    files={str(path.relative_to(out)):digest(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write_json(out/'SHA256.json',files)
    if archive:
        # Contains no frozen QCQP, primal/dual endpoint vectors, logs, binaries,
        # source tree, or previous campaign observations.
        with zipfile.ZipFile(out/'DEPOUILLEMENT_A_TRANSMETTRE.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as bundle:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.name!='DEPOUILLEMENT_A_TRANSMETTRE.zip':bundle.write(path,path.relative_to(out))
    return manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--contract',type=Path,help='Relocated corpus contract, with the identical run-bound SHA256')
    parser.add_argument('--archive',action='store_true',help='Also create the compact evidence ZIP inside --out')
    args=parser.parse_args();manifest=export(args.run,args.out,args.archive,args.contract)
    print(json.dumps({key:manifest[key] for key in ('run_id','planned_tasks','recorded_tasks','missing_tasks','optimizer_invocations')},indent=2))
    print('DÉPOUILLEMENT : '+str(args.out.resolve()),flush=True)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,TypeError) as error:
        print('DÉPOUILLEMENT REFUSÉ — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
