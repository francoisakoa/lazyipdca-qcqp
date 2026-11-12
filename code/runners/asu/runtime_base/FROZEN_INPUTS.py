#!/usr/bin/env python3
"""Resolve frozen QCQPs by published bytes and raw starts, never by a case label.
The scan is read-only, confined to the supplied frozen-data roots. It neither
regenerates a model nor rewrites a contractual hash or a primal start.
"""
from pathlib import Path
import datetime,hashlib,json,os,re,time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def fingerprint(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def verify_raw_start(path,case):
    binding=case.get('start_binding',{})
    if binding.get('external_start_required'):
        raise ValueError('Un départ externe exige une liaison explicite distincte : '+case['case_id'])
    expected=binding.get('raw_x0_line_sha256')
    if not expected:raise ValueError('Empreinte de la ligne x0 brute absente : '+case['case_id'])
    with Path(path).open('rb')as f:line=next((raw for raw in f if raw.startswith(b'x0 ')),None)
    if line is None:raise ValueError('Ligne x0 brute absente : '+str(path))
    observed=hashlib.sha256(line).hexdigest()
    if observed!=expected:raise ValueError('Empreinte x0 brute différente : '+case['case_id'])
    return observed


def walk_files(roots,issues):
    seen_dirs=set()
    def onerror(error):issues.append(dict(path=str(error.filename),error=str(error)))
    for root in roots:
        if not root.is_dir():
            issues.append(dict(path=str(root),error='frozen_root_absent'));continue
        for base,dirs,files in os.walk(root,topdown=True,followlinks=True,onerror=onerror):
            directory=Path(base)
            try:
                s=directory.stat();key=(s.st_dev,s.st_ino)
                if key in seen_dirs:dirs[:]=[];continue
                seen_dirs.add(key)
            except OSError as error:
                issues.append(dict(path=str(directory),error=str(error)));dirs[:]=[];continue
            dirs[:]=sorted(dirs)
            for name in sorted(files):
                p=directory/name
                try:
                    if p.is_file():yield p
                except OSError as error:issues.append(dict(path=str(p),error=str(error)))


class FrozenInputsError(ValueError):
    def __init__(self,message,report):super().__init__(message);self.report=report


def resolve_all(root,cases,report_path,extra_roots=(),progress=None):
    root=Path(root).expanduser().resolve();start=time.monotonic()
    roots=list(dict.fromkeys([root/'data/frozen',root/'canonical_inputs',*[Path(p).expanduser().resolve()for p in extra_roots]]))
    expected={c['case_id']:c for c in cases}
    if len(expected)!=len(cases):raise ValueError('Identifiants contractuels dupliqués')
    for case in cases:
        if not re.fullmatch('[0-9a-f]{64}',case.get('input_sha256','')):
            raise ValueError('Empreinte contractuelle invalide : '+case['case_id'])
    wanted={c['input_sha256']for c in cases};hits={h:[]for h in wanted}
    declared={}
    preparation=root/'PREPARATION.json'
    if preparation.is_file():
        previous=json.loads(preparation.read_text())
        for row in previous.get('cases',[]):
            cid=row.get('id',row.get('case_id'))
            if cid in expected and row.get('input'):
                path=Path(row['input'])
                if not path.is_absolute():path=root/path
                if row.get('input_sha256')!=expected[cid]['input_sha256']:
                    raise ValueError('Le chemin déclaré dans PREPARATION est lié à une autre empreinte : '+cid)
                declared[cid]=path
    nominal={cid:declared.get(cid,root/'data/frozen'/cid/'input.qcq')for cid in expected}
    nominal_names={str(p)for p in nominal.values()}
    observed={};inode_cache={};issues=[];scanned=0;hashed=0;last=0.
    for p in walk_files(roots,issues):
        scanned+=1
        try:
            before=fingerprint(p)
            if before in inode_cache:value=inode_cache[before]
            else:
                with p.open('rb')as f:prefix=f.read(32)
                # Non-QCQP metadata is not a substitute for a model. A nominal
                # path is hashed even with an invalid header for diagnosis.
                if not prefix.startswith(b'LAZYIPDCA_QCQP_') and str(p)not in nominal_names:continue
                value=sha(p);hashed+=1
                if fingerprint(p)!=before:raise ValueError('Fichier modifié pendant la lecture')
                inode_cache[before]=value
            observed[str(p)]=value
            if value in wanted:hits[value].append(p)
        except (OSError,ValueError)as error:issues.append(dict(path=str(p),error=str(error)))
        now=time.monotonic()
        if progress and (hashed%100==0 or now-last>=10):
            progress('index',dict(scanned=scanned,hashed=hashed,matched_hashes=sum(bool(v)for v in hits.values())))
            last=now
    for cid,path in declared.items():
        try:
            before=fingerprint(path)
            value=inode_cache.get(before)
            if value is None:
                value=sha(path);hashed+=1
                if fingerprint(path)!=before:raise ValueError('Entrée déclarée modifiée pendant son contrôle')
                inode_cache[before]=value
            if value in wanted and path not in hits[value]:hits[value].append(path)
        except (OSError,ValueError)as error:issues.append(dict(path=str(path),error=str(error)))
    bindings={};rows=[]
    for i,case in enumerate(cases,1):
        cid=case['case_id'];path=nominal[cid];state='absent';actual=None
        try:
            if path.is_file():
                key=fingerprint(path);actual=inode_cache.get(key)
                if actual is None:
                    actual=sha(path)
                    if fingerprint(path)!=key:raise ValueError('Chemin nominal modifié pendant la lecture')
                state='verified'if actual==case['input_sha256']else'different'
            elif path.is_symlink():state='broken_link'
        except (OSError,ValueError)as error:state='unreadable';issues.append(dict(path=str(path),error=str(error)))
        choices=hits[case['input_sha256']]
        selected=path if state=='verified' else (min(choices,key=lambda p:str(p))if choices else None)
        row=dict(case_id=cid,dataset=case.get('dataset'),regime=case.get('regime'),
            expected_sha256=case['input_sha256'],nominal_path=str(path),nominal_state=state,
            nominal_observed_sha256=actual,status='not_found',selected_input=None,
            raw_x0_verified=False,matching_copies=len(choices))
        if selected is not None:
            try:
                before=fingerprint(selected)
                if inode_cache.get(before)!=case['input_sha256']:
                    if sha(selected)!=case['input_sha256']:raise ValueError('Entrée sélectionnée modifiée après son indexation')
                start_sha=verify_raw_start(selected,case)
                if fingerprint(selected)!=before:raise ValueError('Entrée sélectionnée modifiée pendant le contrôle x0')
                resolved=selected.resolve(strict=True)
                row.update(status='verified',selected_input=str(resolved),raw_x0_verified=True,
                    raw_x0_line_sha256=start_sha,relocated=str(path.resolve())!=str(resolved))
                bindings[cid]=dict(input=str(resolved),input_sha256=case['input_sha256'],
                    raw_x0_line_sha256=start_sha,start_sha256=case['start_binding'].get('start_sha256'))
            except (OSError,ValueError)as error:row.update(status='start_or_input_invalid',error=str(error))
        rows.append(row)
        if progress and(i%100==0 or i==len(cases)):
            progress('cases',dict(done=i,total=len(cases),verified=len(bindings)))
    failures=[r for r in rows if r['status']!='verified']
    report=dict(schema='lazyqcqp-frozen-content-resolution-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        root=str(root),frozen_roots=list(map(str,roots)),prior_preparation=str(preparation)if preparation.is_file()else None,declared_paths=len(declared),cases=len(cases),verified=len(bindings),missing_or_invalid=len(failures),
        relocated=sum(r.get('relocated')is True for r in rows),scanned_files=scanned,hashed_files=hashed,
        seconds=time.monotonic()-start,all_inputs_and_raw_starts_verified=not failures,
        input_files_modified=False,optimizer_invocations=0,scan_issues=issues,rows=rows)
    atomic(report_path,report)
    if failures:
        sample=' ; '.join(r['case_id']+' : '+r['nominal_state']+', SHA attendu '+r['expected_sha256']+
            (', SHA observé '+r['nominal_observed_sha256']if r['nominal_observed_sha256']else'')for r in failures[:3])
        raise FrozenInputsError(str(len(failures))+' entrée(s) sans copie gelée et départ vérifiés. '+sample+
            '. Diagnostic complet : '+str(report_path),report)
    return bindings,report
