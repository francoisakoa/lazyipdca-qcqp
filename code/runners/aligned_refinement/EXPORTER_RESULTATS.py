#!/usr/bin/env python3
"""Export complete scalar observations and reproducibility receipts, read-only.

Large solver logs, original inputs and endpoint vectors stay on the machine.
Their identities are retained; --vectors explicitly includes all endpoint vectors.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,zipfile

KIT=Path(__file__).resolve().parent

def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def export(run,out,vectors=False):
    run=Path(run).resolve(strict=True);out=Path(out).resolve()
    if out.exists():raise ValueError('Archive existante : '+str(out))
    status=json.loads((run/'campaign/STATUS.json').read_text())
    candidates=[run/'campaign'/n for n in ('OBSERVATIONS.jsonl','RESULTS.jsonl')]
    observation=next((p for p in candidates if p.is_file()),None)
    pairs=set();by_method={};observation_errors=[]
    allowed=set(status.get('by_method',{}))
    if observation is None:observation_errors.append('scalar_observations_absent')
    else:
        with observation.open()as stream:
            for line_number,line in enumerate(stream,1):
                try:
                    row=json.loads(line);key=(row['case_id'],row['method'])
                    if not all(isinstance(v,str)and v for v in key):raise ValueError('invalid observation identity')
                    if key in pairs:raise ValueError('duplicate observation identity')
                    if allowed and key[1]not in allowed:raise ValueError('undeclared method')
                    pairs.add(key);by_method[key[1]]=by_method.get(key[1],0)+1
                except (ValueError,KeyError,TypeError)as error:observation_errors.append('line '+str(line_number)+': '+str(error))
    if len(pairs)!=status.get('total'):observation_errors.append('observation_count_differs_from_planned_total')
    for method,counts in status.get('by_method',{}).items():
        if by_method.get(method,0)!=counts.get('planned'):observation_errors.append('method_count_differs: '+method)
    files={}
    for p in run.rglob('*'):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(run)
        if any(x in rel.parts for x in ('__pycache__','observer_source','reference','observer','experiment','compat','smoke')):continue
        if len(rel.parts)>1 and rel.parts[0]=='build' and p.suffix not in ('.json','.txt','.log'):continue
        if 'tasks'in rel.parts:
            if p.name not in ('RESULT.json','AUDIT.json','solver_resources.json','audit_resources.json','COMMAND.json','case.json','factorial_endpoint.json','endpoint.json.factorial.json'):
                if not(vectors and p.name in ('endpoint.json','accepted_trace.jsonl')):continue
        elif p.suffix not in ('.json','.jsonl','.gz','.csv','.txt','.log','.tex'):continue
        if p.name in ('EXPERIENCE.log','IPOPT_ALIGNE.log'):continue
        files[str(rel)]=p
    result=dict(schema='lazyqcqp-research-export-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_run=str(run),state=status.get('state'),done=status.get('done'),total=status.get('total'),
        complete=status.get('state')=='completed'and status.get('done')==status.get('total')and not observation_errors,
        unique_scalar_observations=len(pairs),observation_integrity_errors=observation_errors,
        endpoint_vectors_included=bool(vectors),accepted_iteration_traces_included=bool(vectors),large_solver_logs_included=False,input_models_included=False,
        source_installations_modified=False,scope='All final scalar observations, audits and receipts; no inference that missing evidence is a numerical failure',files={n:dict(sha256=sha(p),bytes=p.stat().st_size)for n,p in sorted(files.items())})
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True)as z:
        for n,p in sorted(files.items()):z.write(p,'DEPOUILLEMENT/'+n)
        z.writestr('DEPOUILLEMENT/EXPORT.json',json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    return dict(archive=str(out),sha256=sha(out),bytes=out.stat().st_size,files=len(files),complete=result['complete'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('ipopt','raffinement'),required=True);p.add_argument('--run',type=Path);p.add_argument('--out',type=Path);p.add_argument('--vectors',action='store_true');a=p.parse_args()
    last=KIT/('LAST_IPOPT_RUN.json'if a.kind=='ipopt'else'LAST_RUN.json')
    run=a.run or Path(json.loads(last.read_text())['out'])
    out=a.out or run/('DEPOUILLEMENT_'+a.kind.upper()+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.zip')
    print(json.dumps(export(run,out,a.vectors),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
