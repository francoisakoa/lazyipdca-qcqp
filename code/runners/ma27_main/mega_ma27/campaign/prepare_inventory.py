#!/usr/bin/env python3
"""Import unchanged frozen models and starts into a new self-contained installation.

Old preparation JSONs are optional file locators only. No result, endpoint,
optimizer, predecessor runtime, or previous numerical status is imported.
"""
from __future__ import annotations
import collections,copy,gzip,hashlib,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from executor import sha,read,write,THREAD_ENV,METHODS
from dataset_schedule import excluded_dey,prepare_datasets


def gzread(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def valid_id(ident):
    return isinstance(ident,str) and ident not in ('','.','..') and '/' not in ident and '\\' not in ident


def path_values(value,field):
    """Only source file path fields, never endpoint/result fields."""
    answer=[]
    if isinstance(value,dict):
        if isinstance(value.get(field),str):answer.append(value[field])
        if field=='input' and isinstance(value.get('original_input'),str):answer.append(value['original_input'])
        for key in ('aliases','source_record','r3_prepared_row','candidate_provenance','serialization_provenance'):
            if key in value:answer.extend(path_values(value[key],field))
    elif isinstance(value,list):
        for item in value:answer.extend(path_values(item,field))
    return answer


class FrozenLocator:
    def __init__(self,root,frozen_roots):
        self.root=Path(root);self.roots=[Path(p).expanduser().resolve() for p in frozen_roots]
        self.rows=collections.defaultdict(list);self.hashed={};self.bundled_index=None
        for base in self.roots:
            # This is deliberately not a recursive search of previous results.
            for name in ('PREPARATION.json','FROZEN_INVENTORY.json'):
                p=base/name
                if not p.is_file():continue
                value=read(p);rows=value.get('cases',[])
                if isinstance(rows,dict):rows=[dict(v,id=v.get('id',k)) for k,v in rows.items()]
                for row in rows:
                    if isinstance(row,dict) and valid_id(row.get('id')):self.rows[row['id']].append(row)

    def matches(self,p,expected):
        try:
            p=p.expanduser().resolve(strict=True)
            if not p.is_file():return False
            stat=p.stat();signature=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            record=self.hashed.get(str(p))
            if record is None or record[0]!=signature:self.hashed[str(p)]=(signature,sha(p))
            return self.hashed[str(p)][1]==expected
        except (OSError,RuntimeError):return False

    def locate(self,ident,field,expected,source_row):
        paths=[]
        if field=='input':
            paths.extend([self.root/'data/frozen'/ident/'input.qcq',
                self.root/'data/bundled_frozen'/ident/'input.qcq',self.root/'data/reference_inputs'/ident/'input.qcq',
                self.root/'test_data/protected81'/ident/'input.qcq'])
        for row in [source_row]+self.rows.get(ident,[]):
            for value in path_values(row,field):
                p=Path(value)
                if p.is_absolute():
                    paths.append(p)
                    project=Path('/data/projects/IPDCA_QCQP')
                    if p.is_relative_to(project):
                        paths.extend(base/p.relative_to(project) for base in self.roots)
                else:
                    paths.extend([self.root/p,self.root/'data/catalogue'/p])
                    if p.parts and p.parts[0]=='data':paths.append(self.root/'data/catalogue'/Path(*p.parts[1:]))
                    for base in self.roots:paths.extend([base/p,base/'data'/p])
        for base in self.roots:
            if field=='input':paths.extend([base/ident/'input.qcq',base/'data/frozen'/ident/'input.qcq',base/'inputs'/ident/'input.qcq',base/'canonical_inputs'/ident/'input.qcq'])
        seen=set()
        for p in paths:
            if str(p) not in seen and self.matches(p,expected):return p.resolve()
            seen.add(str(p))
        # Only the bundled input area is indexed by hashes as a last resort.
        # User project/result trees are never scanned recursively.
        bundled=self.root/'data/bundled_frozen'
        if self.bundled_index is None:
            self.bundled_index={}
            if bundled.is_dir():
                for p in sorted(bundled.rglob('*')):
                    if p.is_file() and not p.is_symlink():self.bundled_index.setdefault(sha(p),p)
        if expected in self.bundled_index:return self.bundled_index[expected].resolve()
        raise FileNotFoundError('Frozen '+field+' missing or different for '+ident+' (SHA256 '+expected+'). Keep old frozen inputs; do not delete data roots.')


def materialize(root,ident,field,expected,source,imports):
    root=Path(root).resolve(strict=True)
    if not valid_id(ident):raise ValueError('Invalid frozen destination identifier')
    folder=root/'data/frozen'/ident
    cursor=root
    for component in ('data','frozen',ident):
        cursor=cursor/component
        if cursor.is_symlink():raise ValueError('Symlink in frozen import destination: '+str(cursor))
        if not cursor.resolve().is_relative_to(root):raise ValueError('Frozen import destination escaped new root')
    folder.mkdir(parents=True,exist_ok=True)
    name='input.qcq' if field=='input' else 'start'+source.suffix
    target=folder/name
    if target.exists():
        if target.is_symlink() or sha(target)!=expected:raise ValueError('Destination frozen file differs: '+str(target))
    else:
        temporary=target.with_name(target.name+'.importing')
        if temporary.exists():
            if temporary.is_symlink():raise ValueError('Unexpected symlink in owned import temporary')
            if sha(temporary)==expected:
                temporary.replace(target)
                imports.append(dict(id=ident,field=field,source=str(source),copied=str(target),sha256=expected))
                return str(target.resolve())
            temporary.unlink()
        with source.open('rb') as src,temporary.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
        if sha(temporary)!=expected or sha(source)!=expected:temporary.unlink();raise ValueError('Frozen source changed during copy: '+ident)
        temporary.replace(target)
    imports.append(dict(id=ident,field=field,source=str(source),copied=str(target),sha256=expected))
    return str(target.resolve())


def historical_reporting(ident,family):
    t=ident.lower()
    if t.startswith(('beam__','beam_safety__')):return 'legacy_beam_structural'
    if t.startswith('robot__'):return 'legacy_robot_arm'
    if 'camshape' in t:return 'COPS_camshape'
    if 'multicast' in t:return 'multicast_beamforming'
    return 'legacy_'+str(family)


def select_metadata(root):
    metadata=Path(root)/'campaign/metadata'
    old_path=metadata/'PLAN_3103.json';old=read(old_path)
    structure=gzread(metadata/'OLD_STRUCTURE_INDEX.json.gz')
    cache=gzread(metadata/'QUALIFIED_COEFFICIENTS.json.gz')
    candidates=read(metadata/'CANDIDATES.json')
    if len(old.get('cases',{}))!=3103 or structure.get('source_plan_sha256')!=sha(old_path):
        raise ValueError('Frozen3103 source metadata mismatch')
    if cache.get('candidate_manifest_sha256')!=sha(metadata/'CANDIDATES.json'):
        raise ValueError('Extension candidate manifest changed')
    if cache.get('reader_sha256')!=sha(Path(root)/'campaign/common/original_auditor.py') or cache.get('classifier_sha256')!=sha(Path(root)/'campaign/classify_qcqp.py'):
        raise ValueError('Coefficient qualification tools differ from source receipts')
    rows=[];removed=[]
    for ident,row in old['cases'].items():
        if not valid_id(ident):raise ValueError('Invalid frozen identifier')
        info=structure['by_id'][ident]
        if info['input_sha256']!=row['input_sha256']:raise ValueError('Historical frozen coefficient identity differs: '+ident)
        for key,value in info['dimensions'].items():
            if key in row['dimensions'] and row['dimensions'][key]!=value:raise ValueError('Historical dimension differs: '+ident)
        case=dict(id=ident,family=row['family'],reporting_family=historical_reporting(ident,row['family']),
            regime=info['regime'],dimensions=row['dimensions'],source_kind='historical',
            input=row['input'],input_sha256=row['input_sha256'],external_start_required=row.get('external_start_required',False),
            start_sha256=row.get('start_sha256'),embedded_start_sha256=row.get('embedded_start_sha256'),
            frozen_start_policy='Raw frozen x0, no external endpoint',_locator=row)
        if case['external_start_required']:
            case.update(start_file=row['start_file'],start_file_sha256=row['start_file_sha256'])
        if excluded_dey(case):removed.append(ident)
        else:rows.append(case)
    if len(removed)!=50:raise ValueError('Expected exact removal of50Dey cases')
    old_hashes={r['input_sha256']:k for k,r in old['cases'].items()}
    poly={};equivalent={};rejected=[];selected=[]
    for row in sorted(candidates['cases'],key=lambda r:r['id']):
        ident=row['id'];q=cache['by_id'][ident]
        if not valid_id(ident) or q['input_sha256']!=row['input_sha256']:raise ValueError('Extension input identity differs: '+ident)
        equivalence=row.get('source_equivalence_key');reason=None
        if equivalence and equivalence in equivalent:reason='same_declared_source_model:'+equivalent[equivalence]
        elif row['input_sha256'] in old_hashes:reason='identical_old_frozen_file:'+old_hashes[row['input_sha256']]
        elif q['canonical_model_sha256'] in poly:reason='duplicate_new_polynomial:'+poly[q['canonical_model_sha256']]
        if reason:rejected.append(dict(id=ident,reason=reason));continue
        poly[q['canonical_model_sha256']]=ident
        if equivalence:equivalent[equivalence]=ident
        selected.append(dict(id=ident,family=row['family'],reporting_family=row['family'],source_kind='extension',
            regime=q['regime'],dimensions=q['dimensions'],input=row['input'],input_sha256=row['input_sha256'],
            external_start_required=False,start_sha256=q['embedded_start_sha256'],embedded_start_sha256=q['embedded_start_sha256'],
            start_hash_encoding=q['start_hash_encoding'],canonical_model_sha256=q['canonical_model_sha256'],
            frozen_start_policy='Raw frozen x0, no external endpoint',candidate_provenance=row,_locator=row))
    if len(selected)!=1025 or {r['id'] for r in rejected}!={'CBLIB__chainsing-1000-2','CBLIB__chainsing-1000-3'}:
        raise ValueError('Frozen1025 extension selection differs from R6')
    rows+=selected
    if len(rows)!=4078 or len({c['id'] for c in rows})!=4078:raise ValueError('Non-Dey frozen inventory incomplete')
    return rows,dict(source_cases=3103,extension_candidates=1027,extensions_selected=1025,
        excluded_dataset='pooling_dey',excluded_case_ids=removed,rejected_duplicate_extensions=rejected,
        source_plan_sha256=sha(old_path),selection_uses_solver_outcomes=False,
        copied_source_metadata='cases only, no previous runtime or numerical result imported')


def import_inventory(root,frozen_roots=()):
    root=Path(root).resolve();rows,selection=select_metadata(root);locator=FrozenLocator(root,frozen_roots)
    imports=[];started=time.monotonic()
    resolved={};missing=[]
    for number,case in enumerate(rows,1):
        for field,key in [('input','input_sha256')]+([('start_file','start_file_sha256')] if case.get('external_start_required') else []):
            try:resolved[(case['id'],field)]=locator.locate(case['id'],field,case[key],case['_locator'])
            except FileNotFoundError as exc:missing.append(dict(id=case['id'],field=field,expected_sha256=case[key],reason=str(exc)))
        if number%500==0:print('VERIFY_FROZEN_AVAILABILITY',number,'/',len(rows),'missing=',len(missing),flush=True)
    write(root/'INPUT_AVAILABILITY.json',dict(complete=not missing,expected_cases=len(rows),
        files_available=len(resolved),files_missing=len(missing),missing=missing,
        previous_results_read=False,source_metadata_only=True))
    if missing:raise FileNotFoundError(str(len(missing))+' frozen files missing; exact list: '+str(root/'INPUT_AVAILABILITY.json')+'. No previous campaign can be purged until import is complete.')
    for number,case in enumerate(rows,1):
        source=resolved[(case['id'],'input')]
        case['input']=materialize(root,case['id'],'input',case['input_sha256'],source,imports)
        with Path(case['input']).open('rb') as f:
            header=[f.readline() for _ in range(4)]
        if not header[0].startswith(b'LAZYIPDCA_QCQP_V') or not header[3].startswith(b'x0 '):raise ValueError('Raw frozen QCQP header/start missing: '+case['id'])
        case['raw_x0_line_sha256']=hashlib.sha256(header[3]).hexdigest()
        if case.get('external_start_required'):
            source=resolved[(case['id'],'start_file')]
            case['start_file']=materialize(root,case['id'],'start_file',case['start_file_sha256'],source,imports)
        case.pop('_locator')
        if number%250==0:print('IMPORT_FROZEN',number,'/',len(rows),flush=True)
    # Reproduce the source coefficient dedup checks for the six shared shapes.
    from classify_qcqp import classify,DIMENSION_KEYS
    old_shapes=collections.defaultdict(list)
    for c in rows:
        if c['source_kind']=='historical':old_shapes[tuple(c['dimensions'][k] for k in DIMENSION_KEYS)].append(c)
    old_poly={};checks=[]
    for c in rows:
        if c['source_kind']!='extension':continue
        for previous in old_shapes.get(tuple(c['dimensions'][k] for k in DIMENSION_KEYS),[]):
            if previous['id'] not in old_poly:old_poly[previous['id']]=classify(previous['input'])['canonical_model_sha256']
            if old_poly[previous['id']]==c['canonical_model_sha256']:raise ValueError('Unexpected historical/extension polynomial duplicate: '+c['id'])
            checks.append(dict(extension=c['id'],historical=previous['id'],same_polynomial=False))
    from canonical_inputs import build_parser,prepare_cases
    parser=build_parser(root/'source',root/'parser_probe')
    canonical=root/'canonical_inputs'
    if canonical.exists() and any(canonical.iterdir()) and not (canonical/'CANONICAL_INPUTS.json').is_file():
        canonical=root/('canonical_inputs_'+str(time.time_ns()))
    rows,beam=prepare_cases(rows,root/'data',canonical,parser,expected_extension_count=120)
    from MEGA import ordered_cases
    rows,datasets,_=ordered_cases(rows)
    inventory=dict(schema='mega-fresh-frozen-no-dey-inventory-v1',qualified=True,cases=rows,datasets=datasets,
        methods=list(__import__('executor').METHODS),selection=selection,coefficient_crosschecks=checks,
        source_metadata_sha256={p.name:sha(p) for p in (root/'campaign/metadata').iterdir() if p.is_file()},
        beamforming_reader_report=str(canonical/'CANONICAL_INPUTS.json'),
        counts=dict(cases=len(rows),datasets=len(datasets),tasks=len(rows)*len(METHODS),excluded_Dey=50),
        historical_numeric_reuse_allowed=False,import_seconds=time.monotonic()-started)
    write(root/'FROZEN_INVENTORY.json',inventory)
    write(root/'INPUT_IMPORT_COMPLETE.json',dict(complete=True,cases=len(rows),tasks=len(rows)*len(METHODS),
        datasets=len(datasets),Dey_cases=0,files_copied=len(imports),imports=imports,
        input_inventory_sha256=sha(root/'FROZEN_INVENTORY.json'),
        raw_frozen_input_and_start_bytes_preserved=True,beamforming_exact_serialization_only=True,
        previous_results_or_endpoints_imported=False))
    return inventory


def shared_profiles(root):
    path=Path(root)/'campaign/profiles.json';value=read(path)
    if set(value['profiles'])!={'R3','HH','HS','SH','SS','SS_legacy'}:raise ValueError('Six native identities required')
    for command in value['profiles'].values():
        for flag,setting in [('--shared-restoration-budget','1'),('--restoration-iters','600')]:
            positions=[i for i,x in enumerate(command[:-1]) if x==flag]
            if positions:
                for i in positions:command[i+1]=setting
            else:command.extend([flag,setting])
    # The source payload freezes these shared limit changes before installation.
    if read(path)!=value:raise ValueError('Bundled profiles must already declare shared restoration600')
    return value


def bind_preparation(root,installation,inventory):
    root=Path(root).resolve()
    if installation.get('success') is not True:raise ValueError('Complete native/comparator qualification required')
    from MEGA import verify_installation
    verify_installation(root,installation)
    import importlib.util
    gate_path=root/'validation_tools/audit_certificate_gate.py'
    gate_spec=importlib.util.spec_from_file_location('_preparation_audit_gate',gate_path)
    gate=importlib.util.module_from_spec(gate_spec);gate_spec.loader.exec_module(gate)
    gate_receipt=gate.verify_receipt(root,prepared=inventory)
    source=read(root/'SOURCE_MANIFEST.json')['source_sha256']
    for rel,expected in source.items():
        if sha(root/'source'/rel)!=expected:raise ValueError('Compiled source manifest differs: '+rel)
    runtime={}
    for p in (root/'campaign').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and 'upstream' not in p.parts:
            runtime[str(p)]=sha(p)
    for p in [root/'INSTALLATION.json',root/'SOURCE_MANIFEST.json',root/'INPUT_IMPORT_COMPLETE.json',root/'FROZEN_INVENTORY.json',root/'native/BUILD.json',root/'setup_runtime.py']:
        runtime[str(p)]=sha(p)
    for key in ('binary','native_auditor','ipopt_binary','hsl_library','ma27_qualification','mumps_qualification'):
        runtime[installation[key]]=installation[key+'_sha256']
    for key in ('ipopt_build_manifest','precise_auditor'):
        if installation.get(key):runtime[installation[key]]=sha(installation[key])
    for p in (gate_path,root/'validation/AUDIT_CERTIFICATE_GATE.json'):
        runtime[str(p)]=sha(p)
    for relative,expected in gate_receipt['auditor_source_sha256'].items():
        p=root/relative
        if sha(p)!=expected:raise ValueError('Qualified audit dependency changed: '+relative)
        runtime[str(p)]=expected
    for value,expected in installation.get('runtime_dependency_sha256',{}).items():
        if sha(value)!=expected:raise ValueError('Qualified numerical dependency changed: '+value)
        runtime[value]=expected
    if not Path('/usr/bin/taskset').is_file():raise ValueError('Linux util-linux taskset is required')
    runtime['/usr/bin/taskset']=sha('/usr/bin/taskset')
    value=dict(schema='mega-fresh-preparation-no-dey-v1',root=str(root),cases=inventory['cases'],datasets=inventory['datasets'],
        methods=list(__import__('executor').METHODS),profiles=shared_profiles(root),installation=installation,
        runtime_sha256=runtime,source_sha256={str(root/'source'/rel):h for rel,h in source.items()},
        excluded_datasets=['pooling_dey'],historical_numeric_reuse_allowed=False,historical_results_reused=0,
        counts=inventory['counts'],beamforming_file=inventory['beamforming_reader_report'])
    write(root/'PREPARATION.json',value);return value
