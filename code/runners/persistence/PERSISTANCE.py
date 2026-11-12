#!/usr/bin/env python3
"""Ablation contrôlée du cache de courbure R3 : préparation, qualification et temps."""
from pathlib import Path
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import struct
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'support_chronos'))
import CHRONO_CALME as quiet
sys.path.insert(0,str(HERE))
import BUILD_PERSISTANCE as builder

MODES=('reuse','rebuild')
CONTROL_MODES=('reference','reuse','rebuild')
PROTOCOL=dict(quiet.PROTOCOL,schema='curvature-persistence-protocol-v1',
    pairs=24,cases=12,timed_repetitions=3,timed_observations=144,qualification_observations=72,
    intervention='same deterministic R3 correction; rebuild at each model reanchor event, including repeated centers; reuse constructs once per model instance',
    controls='unpatched reference, patched reuse and patched rebuild from identical raw starts; exact original endpoint/work, ordered correction/state traces and original audits',
    trace_policy='qualification only; report sidecar in all patched runs; no fingerprint or trajectory trace in timed repetitions',
    ordering='pairs hashed with seed 20261110; alternating modes per pair; repetition 2 reverses modes; repetition 3 rotates fixed pair order',
    numerical_failure_policy='commit first finalized uncontaminated attempt, including numerical failure or exhausted budget; never retry to obtain success')
quiet.PROTOCOL=PROTOCOL

# These counters measure the intervention itself. Every other non-clock
# endpoint field, including barrier choices and numerical work, must match.
CACHE_COUNTER_FIELDS=frozenset({'dca_filter_curvature_reuses','dca_filter_majorant_curvature_reuses'})
AUDIT_CORE=('classification','strict_qcqp','primal','dual','complementarity','objective','audit_available','original_KKT_endpoint_strict')


def load_cohort():
    c=quiet.read(HERE/'COHORTE_PERSISTANCE.json')
    if c.get('cohort_content_sha256')!=quiet.digest({k:v for k,v in c.items() if k!='cohort_content_sha256'}):
        raise ValueError('Cohorte de persistance modifiée')
    pairs=c.get('pairs',[])
    if len(pairs)!=24 or len({p['pair_id'] for p in pairs})!=24 or len({p['case_id'] for p in pairs})!=12:
        raise ValueError('Cohorte attendue : 12 cas, 24 couples instance/backend')
    if any({p['backend'] for p in pairs if p['case_id']==cid}!={'ma27','mumps'} for cid in {p['case_id'] for p in pairs}):
        raise ValueError('Les deux backends doivent être présents pour chaque cas')
    return c


def files_snapshot():
    paths=[HERE/'PERSISTANCE.py',HERE/'BUILD_PERSISTANCE.py',HERE/'COHORTE_PERSISTANCE.json']
    paths += sorted((HERE/'support_chronos').rglob('*.py'))
    paths += [HERE/'support_chronos/evidence/CONTEXTE_HISTORIQUE.json']
    return {str(p.relative_to(HERE)):quiet.sha(p) for p in paths if '__pycache__' not in p.parts}


def build_manifest(path):
    path=Path(path).expanduser().resolve(strict=True);b=quiet.read(path)
    if b.get('schema')!='lazy-curvature-persistence-build-v1':raise ValueError('Construction de persistance requise')
    for key in ('production_dependencies_verified','same_binary_for_both_backends','homogeneous_compilation','original_installation_unchanged'):
        if b.get(key)is not True:raise ValueError('Preuve de construction absente : '+key)
    if b.get('compiled_native_backends')!=['ma27','mumps']:raise ValueError('Deux backends compilés requis')
    pins={str(path):quiet.sha(path)}
    for key in ('binary','reference_binary','reference_build','compiled_native_build','patch_receipt','dependencies','hsl_library'):
        pins[b[key]]=quiet.artifact(b[key],b[key+'_sha256'])['sha256']
    builder.validate_build_pair(quiet.read(b['reference_build']),quiet.read(b['compiled_native_build']),quiet.read(b['patch_receipt']))
    for root_key,map_key in [('source_root','source_sha256'),('reference_source_root','base_source_sha256')]:
        for name,digest in b[map_key].items():
            item=quiet.artifact(Path(b[root_key])/name,digest);pins[item['path']]=item['sha256']
    if not b.get('runtime_sha256'):raise ValueError('Dépendances dynamiques non liées')
    for p,h in b['runtime_sha256'].items():pins[p]=quiet.artifact(p,h)['sha256']
    return b,pins


def schedules(cohort):
    pairs=sorted(cohort['pairs'],key=lambda p:hashlib.sha256(('20261110|'+p['pair_id']).encode()).hexdigest())
    timed=[];controls=[]
    for pair in pairs:
        for mode in CONTROL_MODES:
            controls.append(dict(pair_id=pair['pair_id'],mode=mode,job_id=pair['pair_id']+'_'+mode,
                nominal_id=pair['pair_id']+'_'+mode+'_qualification',repetition=0,phase='qualification'))
    for rep in range(1,4):
        ordering=pairs if rep!=3 else pairs[len(pairs)//2:]+pairs[:len(pairs)//2]
        for index,pair in enumerate(ordering):
            modes=MODES if (index+rep)%2 else tuple(reversed(MODES))
            for mode in modes:
                job=pair['pair_id']+'_'+mode
                timed.append(dict(pair_id=pair['pair_id'],mode=mode,job_id=job,
                    nominal_id=job+'_rep'+str(rep),repetition=rep,phase='timing'))
    if len(timed)!=144 or len(controls)!=72 or len({t['nominal_id'] for t in timed+controls})!=216:
        raise ValueError('Plan des répétitions incohérent')
    return controls,timed


def resolve_inputs(cohort,ipopt_run):
    inputs={};pins={};fallback=None
    for pair in cohort['pairs']:
        cid=pair['case_id']
        if cid in inputs:continue
        path=Path(pair['frozen_input']);external=pair['start_binding'].get('start_file')
        if not path.is_file():
            if fallback is None:
                fallback,extra=quiet.verified_ipopt(Path(ipopt_run)/'preparation/INSTALLATION_IPOPT_ALIGNE.json');pins.update(extra)
            binding=fallback['frozen_bindings'].get(cid)
            if not binding:raise ValueError('Entrée gelée absente : '+cid+' ; '+str(path))
            path=Path(binding['input']);external=binding.get('start') or external
        if not pair['start_binding'].get('external_start_required'):external=None
        quiet.verify_start(path,pair,external)
        inputs[cid]=dict(input=str(path.resolve()),external_start=external)
        pins[str(path.resolve())]=pair['input_sha256']
        if external:pins[str(Path(external).resolve())]=pair['start_binding']['start_file_sha256']
    return inputs,pins


def prepare(build,out,ipopt_run=quiet.DEFAULT_IPOPT_RUN):
    out=Path(out).expanduser().resolve()
    if (out/'INSTALLATION.json').is_file():
        d=load_installation(out)
        if d['build']['path']!=str(Path(build).expanduser().resolve()):raise ValueError('Autre construction demandée')
        print('PRÉPARATION EXISTANTE VÉRIFIÉE : '+str(out),flush=True);return d
    quiet.empty_preparation_output(out)
    if out==HERE or HERE in out.parents or out in HERE.parents:raise ValueError('Résultats séparés du kit requis')
    c=load_cohort();b,pins=build_manifest(build);inputs,extra=resolve_inputs(c,ipopt_run);pins.update(extra)
    context=quiet.read(HERE/'support_chronos/evidence/CONTEXTE_HISTORIQUE.json')
    if quiet.digest(context['R3'])!=context['R3_profile_sha256']:raise ValueError('Profil R3 différent')
    controls,timed=schedules(c)
    if 0 not in os.sched_getaffinity(0):raise ValueError('CPU 0 indisponible')
    d=dict(schema='curvature-persistence-installation-v1',created_utc=quiet.utc(),protocol=PROTOCOL,
        cohort_sha256=quiet.sha(HERE/'COHORTE_PERSISTANCE.json'),source_sha256=files_snapshot(),
        build=quiet.artifact(build),pinned_files=pins,inputs=inputs,R3=context['R3'],
        binary=b['binary'],binary_sha256=b['binary_sha256'],reference_binary=b['reference_binary'],
        reference_binary_sha256=b['reference_binary_sha256'],hsl_library=b['hsl_library'],
        runtime_environment=b['runtime_environment'],qualification_schedule=controls,schedule=timed,
        pairs=c['pairs'],machine=quiet.machine_info(),output_directory=str(out),optimizer_invocations=0)
    d['installation_content_sha256']=quiet.digest(d)
    quiet.empty_preparation_output(out);out.mkdir(parents=True,exist_ok=True);quiet.write(out/'INSTALLATION.json',d)
    print('PERSISTANCE PRÉPARÉE — 72 contrôles puis 144 mesures ; aucun solveur lancé',flush=True)
    return d


def load_installation(out,check_runtime=True):
    out=Path(out).resolve();d=quiet.read(out/'INSTALLATION.json')
    if d.get('schema')!='curvature-persistence-installation-v1' or d.get('protocol')!=PROTOCOL:
        raise ValueError('Installation/protocole de persistance incompatible')
    if d.get('installation_content_sha256')!=quiet.digest({k:v for k,v in d.items() if k!='installation_content_sha256'}):
        raise ValueError('Installation modifiée')
    if d['output_directory']!=str(out):raise ValueError('Dossier déplacé')
    if d['cohort_sha256']!=quiet.sha(HERE/'COHORTE_PERSISTANCE.json'):raise ValueError('Cohorte modifiée')
    for p,h in d['source_sha256'].items():quiet.artifact(HERE/p,h)
    if check_runtime:
        for p,h in d['pinned_files'].items():quiet.artifact(p,h)
        if d['machine']['hostname']!=os.uname().nodename:raise ValueError('Autre machine')
        if 0 not in os.sched_getaffinity(0):raise ValueError('CPU 0 indisponible')
    return d


class JsonNumber(str):
    """Preserve native JSON number tokens, including the integer token -0."""


def exact_core(value,cache_counts=False):
    if isinstance(value,dict):
        return ['object',[[k,exact_core(value[k],cache_counts)] for k in sorted(value)
                if not k.endswith('_seconds') and not (cache_counts and k in CACHE_COUNTER_FIELDS)]]
    if isinstance(value,list):return ['array',[exact_core(v,cache_counts) for v in value]]
    if isinstance(value,JsonNumber):return ['json_number_token',str(value)]
    if type(value)is float:return ['binary64',struct.pack('>d',value).hex()]
    if type(value)is bool:return ['boolean',value]
    if type(value)is int:return ['integer',str(value)]
    if value is None:return ['null']
    if type(value)is str:return ['string',value]
    raise TypeError('Type de donnée non JSON dans la comparaison exacte')


def endpoint_core(path,cache_counts=False):
    value=json.loads(Path(path).read_text(),parse_int=JsonNumber,parse_float=JsonNumber,parse_constant=JsonNumber)
    if not isinstance(value,dict):raise ValueError('Endpoint JSON non objet')
    return exact_core(value,cache_counts)


def json_lines(path):
    if not Path(path).is_file():raise ValueError('Rapport de courbure absent : '+str(path))
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if any(not isinstance(r,dict) for r in rows):raise ValueError('Rapport JSONL invalide')
    return rows


def cache_report(path,mode,traced,endpoint_models=None):
    records=json_lines(path);seen=set();totals=collections.Counter()
    if not records:raise ValueError('Rapport de courbure sans clôture de session')
    session=records[-1];rows=records[:-1]
    if session.get('schema')!='research-curvature-cache-session-v1' or session.get('mode')!=mode or session.get('solver_returned')is not True:
        raise ValueError('Clôture de session de courbure absente')
    if type(session.get('model_instances'))is not int or session['model_instances']!=len(rows):
        raise ValueError('Nombre de modèles incomplet dans le rapport')
    if not rows and not (type(endpoint_models)is int and endpoint_models==0):
        raise ValueError('Zéro instance sans preuve native de zéro modèle')
    for row in rows:
        if row.get('schema')!='research-curvature-cache-summary-v1' or row.get('mode')!=mode:
            raise ValueError('Mode/schéma de courbure différent')
        if row.get('complete')is not True:raise ValueError('Résumé de modèle incomplet')
        instance=row.get('model_instance')
        if type(instance)is not int or instance<1 or instance in seen:raise ValueError('Instance de modèle absente/dupliquée')
        seen.add(instance)
        for key in ('preparations','reanchors','quadratic_row_builds','corrected_rows','trace_records','reuse_hits'):
            if type(row.get(key))is not int or row[key]<0:raise ValueError('Compteur de courbure invalide : '+key)
            totals[key]+=row[key]
        for key in ('construction_seconds','rebuild_reset_seconds'):
            if type(row.get(key))not in (int,float) or not math.isfinite(row[key]) or row[key]<0:
                raise ValueError('Horloge de courbure invalide : '+key)
            totals[key]+=row[key]
        anchors=row['reanchors'];preps=row['preparations']
        if anchors and preps!=(1 if mode=='reuse' else anchors):raise ValueError('Construction/reconstruction non prouvée')
        if not anchors and preps not in (0,1):raise ValueError('Constructions inattendues sans recalage')
        if preps and (row['quadratic_row_builds']%preps or row['corrected_rows']>row['quadratic_row_builds']//preps):
            raise ValueError('Lignes corrigées/construites incohérentes')
        if not preps and (row['corrected_rows'] or row['quadratic_row_builds']):raise ValueError('Courbure déclarée sans préparation')
        if row['reuse_hits']!=(max(0,anchors-1) if mode=='reuse' else 0):raise ValueError('Réutilisations non prouvées')
        if not traced and row['trace_records']!=0:raise ValueError('Trace activée dans une mesure de temps')
    if seen and seen!=set(range(1,len(seen)+1)):raise ValueError('Instances de modèle non contiguës')
    return dict(rows=rows,session=session,totals=dict(totals),model_instances=len(rows),mode=mode)


def command_for(d,pair,mode,folder,traced):
    binary=d['reference_binary'] if mode=='reference' else d['binary'];binding=d['inputs'][pair['case_id']]
    values=dict(binary=binary,input=binding['input'],endpoint=str(folder/'endpoint.json'),
        backend=pair['backend'],max_outer=5000,seconds=7200)
    cmd=[s.format(**values) for s in d['R3']]
    cmd+=['--native-al-fallback','off','--standard-r10-policy','auto',
          '--dca-rank-presolve-report',str(folder/'rank_presolve.json')]
    if pair['backend']=='ma27':cmd+=['--hsl-library',d['hsl_library']]
    if binding.get('external_start'):cmd+=['--frozen-start',binding['external_start']]
    if mode!='reference':
        cmd+=['--research-curvature-cache',mode,'--research-curvature-report',str(folder/'curvature_report.jsonl')]
        if traced:cmd+=['--research-curvature-trace',str(folder/'curvature_trace.jsonl')]
    return cmd


def endpoint_proof(d,pair,path):
    ep=quiet.read(path);native_job=dict(pair,role='native')
    proof=quiet.verify_endpoint(d,native_job,path)
    for key,limit in [('outer_iterations',5000),('restoration_iterations',600)]:
        if type(ep.get(key))is not int or not 0<=ep[key]<=limit:raise ValueError('Budget natif absent/dépassé : '+key)
    seconds=ep.get('elapsed_seconds')
    if type(seconds)not in(int,float) or not math.isfinite(seconds) or not 0<=seconds<=7200:
        raise ValueError('Horloge native absente/dépassée')
    return proof


def attempt_result(d,pair,task,folder,stop):
    mode=task['mode'];traced=task['phase']=='qualification'
    binding=d['inputs'][pair['case_id']];quiet.verify_start(binding['input'],pair,binding.get('external_start'))
    quiet.artifact(d['reference_binary'] if mode=='reference' else d['binary'],
        d['reference_binary_sha256'] if mode=='reference' else d['binary_sha256'])
    monitor=quiet.QuietMonitor(folder/'QUIET.jsonl');monitor.wait_quiet(stop)
    cmd=command_for(d,pair,mode,folder,traced);config=dict(environment=d['runtime_environment'])
    env=quiet.environment(config)
    runtime_env={k:v for k,v in env.items() if k.startswith(('OMP_','OPENBLAS_','MKL_','BLIS_','NUMEXPR_')) or k in ('LD_LIBRARY_PATH','HSL_LIBRARY')}
    quiet.write(folder/'COMMAND.json',dict(command=cmd,numerical_runtime_environment=runtime_env,pair=pair,task=task))
    resources=quiet.run_process(cmd,folder,env,stop,monitor);quiet.write(folder/'solver_resources.json',resources)
    errors=[];proof=None;report=None;audit=None;auditres=None;endpoint=None
    if (folder/'endpoint.json').is_file():
        try:
            endpoint=quiet.read(folder/'endpoint.json')
            if not isinstance(endpoint,dict):raise ValueError('Endpoint JSON non objet')
            proof=endpoint_proof(d,pair,folder/'endpoint.json')
        except (ValueError,TypeError,KeyError,OSError) as e:
            if not isinstance(endpoint,dict):endpoint=None
            errors.append('endpoint_contract: '+str(e))
        if mode!='reference':
            try:report=cache_report(folder/'curvature_report.jsonl',mode,traced,(endpoint or {}).get('dca_filter_models'))
            except (ValueError,TypeError,KeyError,OSError) as e:errors.append('curvature_contract: '+str(e))
        case=dict(pair,method='R3_'+mode,dataset=pair.get('dataset_id'))
        quiet.write(folder/'case.json',case);audit_dir=folder/'audit';audit_dir.mkdir()
        auditcmd=[sys.executable,str(HERE/'support_chronos/support/AUDIT_ORIGINAL.py'),
            '--input',binding['input'],'--endpoint',str(folder/'endpoint.json'),'--out',str(folder/'AUDIT.json'),
            '--resources',str(folder/'solver_resources.json'),'--case-json',str(folder/'case.json')]
        quiet.write(folder/'AUDIT_COMMAND.json',auditcmd)
        auditres=quiet.run_process(auditcmd,audit_dir,env,stop,timeout=600);quiet.write(folder/'audit_resources.json',auditres)
        if (folder/'AUDIT.json').is_file():
            try:
                audit=quiet.read(folder/'AUDIT.json')
                if not isinstance(audit,dict):raise ValueError('Audit JSON non objet')
                if audit.get('input_sha256')!=pair['input_sha256'] or audit.get('endpoint_sha256')!=quiet.sha(folder/'endpoint.json'):
                    raise ValueError('Audit original lié à d’autres données/endpoint')
            except (ValueError,TypeError,KeyError,OSError) as e:errors.append('original_audit_contract: '+str(e));audit=None
        else:errors.append('original_audit_unavailable')
    else:errors.append('endpoint_unavailable')
    classification=(audit or {}).get('classification','indeterminate')
    if errors or not quiet.budget_verified(resources) or (auditres and (auditres['returncode']!=0 or auditres['censored'])):
        classification='indeterminate'
    residuals={k:(audit or {}).get(k) for k in ('primal','dual','complementarity')}
    if classification=='strict' and not ((audit or {}).get('strict_qcqp')is True and
        all(type(v)in(int,float) and math.isfinite(v) and 0<=v<=1e-6 for v in residuals.values())):
        classification='indeterminate';errors.append('strict_certificate_inconsistent')
    cause='none' if classification=='strict' else ('budget_exhaustion' if resources['timed_out'] else
        ('endpoint_above_tolerance' if classification=='nonstrict' else 'unavailable_or_unverified_endpoint'))
    numerical_core_sha256=quiet.digest(endpoint_core(folder/'endpoint.json',True)) if isinstance(endpoint,dict) else None
    return dict(schema='curvature-persistence-attempt-v1',nominal_id=task['nominal_id'],job_id=task['job_id'],
        pair_id=pair['pair_id'],case_id=pair['case_id'],backend=pair['backend'],regime=pair['regime'],
        mode=mode,phase=task['phase'],repetition=task['repetition'],finished_utc=quiet.utc(),
        quiet_valid=not resources['quiet_observation']['contaminated'],solver_resources=resources,
        classification=classification,failure_cause=cause,strict_qcqp=classification=='strict',
        budget_verified=quiet.budget_verified(resources),residuals=residuals,objective=(audit or {}).get('objective'),
        proof=proof,cache_report=report,errors=errors,numerical_core_sha256=numerical_core_sha256,seconds_process=resources['elapsed_seconds'],
        seconds_process_scope=PROTOCOL['timed_scope'],audit_seconds=(audit or {}).get('audit_seconds'),all_run_costs_retained=True)


def phase_run(d,out,phase):
    root=out/'qualification' if phase=='qualification' else out
    root.mkdir(parents=True,exist_ok=True);quiet.ensure_no_owned_live(root)
    schedule=d['qualification_schedule'] if phase=='qualification' else d['schedule']
    pairs={p['pair_id']:p for p in d['pairs']};installation_sha=quiet.sha(out/'INSTALLATION.json')
    stop=[False]
    def stopping(signum,frame):stop[0]=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopping)
    completed=0
    for task in schedule:
        folder=root/'runs'/task['nominal_id'];final=folder/'RESULT.json'
        if quiet.result_complete(final,installation_sha,task['nominal_id']):completed+=1;continue
        if stop[0]:break
        folder.mkdir(parents=True,exist_ok=True)
        attempts=sorted(p for p in folder.glob('attempt_*') if p.is_dir());recovered=False
        for old in attempts:
            if (old/'ATTEMPT.json').is_file() and quiet.commit_attempt(old,final,installation_sha,task['nominal_id']):
                completed+=1;recovered=True;break
            if not (old/'ATTEMPT.json').exists() and not (old/'INTERRUPTED.json').exists():
                quiet.write(old/'INTERRUPTED.json',dict(discovered_utc=quiet.utc(),reason='controller stopped before finalization',raw_files_retained=True))
        if recovered:continue
        while not stop[0]:
            attempt=folder/('attempt_'+str(len(attempts)+1).zfill(4));attempt.mkdir();attempts.append(attempt)
            quiet.write(attempt/'STARTED.json',dict(utc=quiet.utc(),installation_sha256=installation_sha,task=task))
            quiet.write(root/'STATUS.json',dict(utc=quiet.utc(),state='waiting_for_quiet_then_solver_and_audit',phase=phase,
                active=task,attempt=str(attempt),quiet_log=str(attempt/'QUIET.jsonl'),completed=completed,planned=len(schedule)))
            print('PERSISTANCE '+phase+' — '+task['nominal_id']+' ; attente de calme',flush=True)
            try:r=attempt_result(d,pairs[task['pair_id']],task,attempt,stop)
            except BaseException as e:
                quiet.write(attempt/'INTERRUPTED.json',dict(utc=quiet.utc(),reason=type(e).__name__+': '+str(e),raw_files_retained=True));raise
            r.update(installation_sha256=installation_sha,finalized=True)
            r['raw_artifacts']={p.name:quiet.artifact(p) for p in sorted(attempt.iterdir()) if p.is_file()}
            r['attempt_content_sha256']=quiet.digest(r);quiet.write(attempt/'ATTEMPT.json',r)
            if r['solver_resources']['interrupted'] or stop[0]:break
            if r['quiet_valid']:
                quiet.commit_attempt(attempt,final,installation_sha,task['nominal_id']);completed+=1
                print('PERSISTANCE '+phase+' — '+str(completed)+'/'+str(len(schedule))+' ; '+r['classification']+
                    ' ; '+format(r['seconds_process'],'.6f')+' s',flush=True);break
            print('CHARGE ÉTRANGÈRE — tentative conservée ; même répétition à reprendre après retour au calme',flush=True)
        quiet.write(root/'STATUS.json',dict(utc=quiet.utc(),phase=phase,state='interrupted' if stop[0] else 'running',completed=completed,planned=len(schedule)))
    quiet.write(root/'STATUS.json',dict(utc=quiet.utc(),phase=phase,state='completed' if completed==len(schedule) else 'interrupted',completed=completed,planned=len(schedule)))
    return completed


def result_folder(result):
    item=result.get('artifacts',{}).get('endpoint.json')
    if not item:raise ValueError('Endpoint non lié au résultat')
    return Path(item['path']).parent


def trace_core(path,mode):
    rows=json_lines(path);core=[]
    for row in rows:
        if row.get('schema')!='research-curvature-cache-trace-v1' or row.get('event')not in ('reanchor','controller','accepted') or row.get('mode')!=mode:
            raise ValueError('Événement de trace inconnu')
        for key in ('model_instance','reanchor','outer','accepted_steps'):
            if type(row.get(key))is not int:raise ValueError('Compteur de trace absent : '+key)
        if row['model_instance']<1 or row['reanchor']<1 or row['accepted_steps']<0 or not isinstance(row.get('origin'),str):
            raise ValueError('Instance/recalage/origine de trace invalide')
        for key in ('correction_identity','state_identity'):
            value=row.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Empreinte exacte de trace absente')
        core.append({k:v for k,v in row.items() if k!='mode'})
    return core


def compare_qualification(group,d,pair):
    folders={mode:result_folder(group[mode]) for mode in CONTROL_MODES}
    for mode,r in group.items():
        if r.get('classification')not in ('strict','nonstrict') or r.get('budget_verified')is not True or r.get('errors'):
            raise ValueError('Qualification sans preuve complète : '+mode)
        resources=quiet.read(folders[mode]/'solver_resources.json')
        if resources!=r.get('solver_resources') or not quiet.budget_verified(resources):raise ValueError('Ressources de contrôle incohérentes')
        command=command_for(d,pair,mode,folders[mode],True)
        recorded=quiet.read(folders[mode]/'COMMAND.json')
        if recorded.get('command')!=command or resources.get('command')!=command or resources.get('command_sha256')!=quiet.digest(command):
            raise ValueError('Commande de contrôle non liée aux ressources')
        env=recorded.get('numerical_runtime_environment',{})
        if any(env.get(k)!='1' for k in quiet.THREAD_ENV):raise ValueError('Nombre de threads de contrôle différent')
        endpoint_proof(d,pair,folders[mode]/'endpoint.json')
        audit=quiet.read(folders[mode]/'AUDIT.json')
        if audit.get('input_sha256')!=pair['input_sha256'] or audit.get('endpoint_sha256')!=quiet.sha(folders[mode]/'endpoint.json'):
            raise ValueError('Audit de contrôle non lié aux données/endpoint')
        if audit.get('classification')!=r.get('classification'):raise ValueError('Classification de contrôle différente de son audit')
        if exact_core(r.get('residuals'))!=exact_core({k:audit.get(k) for k in ('primal','dual','complementarity')}):
            raise ValueError('Résidus de contrôle différents de leur audit')
    endpoints={m:quiet.read(p/'endpoint.json') for m,p in folders.items()}
    if endpoint_core(folders['reference']/'endpoint.json')!=endpoint_core(folders['reuse']/'endpoint.json'):raise ValueError('Référence/réutilisation : endpoint ou travail différent')
    if endpoint_core(folders['reuse']/'endpoint.json',True)!=endpoint_core(folders['rebuild']/'endpoint.json',True):raise ValueError('Réutilisation/reconstruction : endpoint ou travail numérique différent')
    audits={m:{k:quiet.read(p/'AUDIT.json').get(k) for k in AUDIT_CORE} for m,p in folders.items()}
    if not exact_core(audits['reference'])==exact_core(audits['reuse'])==exact_core(audits['rebuild']):
        raise ValueError('Audit original différent entre contrôles')
    reports={m:cache_report(folders[m]/'curvature_report.jsonl',m,True,endpoints[m].get('dca_filter_models')) for m in MODES}
    if reports['reuse']['model_instances']!=reports['rebuild']['model_instances']:raise ValueError('Nombre d’instances de modèle différent')
    for a,b in zip(reports['reuse']['rows'],reports['rebuild']['rows']):
        for key in ('model_instance','reanchors','corrected_rows'):
            if a[key]!=b[key]:raise ValueError('Instances/recalages/corrections non appariés : '+key)
        if b['quadratic_row_builds']*a['preparations']!=a['quadratic_row_builds']*b['preparations']:
            raise ValueError('Nombre de lignes reconstruites par préparation différent')
    traces={m:trace_core(folders[m]/'curvature_trace.jsonl',m) for m in MODES}
    if traces['reuse']!=traces['rebuild']:raise ValueError('Correction ou trajectoire tracée différente')
    for mode in MODES:
        if len(traces[mode])!=reports[mode]['totals'].get('trace_records',0):raise ValueError('Trace incomplète')
        if sum(r['event']=='reanchor' for r in traces[mode])!=reports[mode]['totals'].get('reanchors',0):raise ValueError('Recalage sans trace')
        trace_instances={r['model_instance'] for r in traces[mode]}
        if not trace_instances.issubset({r['model_instance'] for r in reports[mode]['rows']}):raise ValueError('Trace sans résumé de modèle')
        for row in reports[mode]['rows']:
            events=[r for r in traces[mode] if r['model_instance']==row['model_instance']]
            anchors=[r for r in events if r['event']=='reanchor']
            if len(events)!=row['trace_records'] or len(anchors)!=row['reanchors']:
                raise ValueError('Trace incomplète pour une instance de modèle')
            if [r['reanchor'] for r in anchors]!=list(range(1,row['reanchors']+1)):
                raise ValueError('Ordre de recalage invalide')
    avoided=reports['rebuild']['totals'].get('preparations',0)-reports['reuse']['totals'].get('preparations',0)
    corrected=reports['reuse']['totals'].get('corrected_rows',0)
    return dict(passed=True,reference_reuse_endpoint_and_work_exact=True,reuse_rebuild_numerical_endpoint_and_work_exact=True,
        original_audit_exact=True,ordered_correction_and_state_trace_exact=True,trace_records=len(traces['reuse']),
        reanchors=reports['reuse']['totals'].get('reanchors',0),model_instances=reports['reuse']['model_instances'],
        preparations_reuse=reports['reuse']['totals'].get('preparations',0),preparations_rebuild=reports['rebuild']['totals'].get('preparations',0),
        preparations_avoided=avoided,corrected_rows=corrected,cache_exercised=avoided>0,
        nonzero_correction_observed=corrected>0,mechanism_informative=avoided>0 and corrected>0)


def qualification_gate(d,out):
    installation_sha=quiet.sha(out/'INSTALLATION.json');rows=[];artifacts={};passed=True
    by_pair=collections.defaultdict(dict)
    for task in d['qualification_schedule']:
        path=out/'qualification/runs'/task['nominal_id']/'RESULT.json'
        if quiet.result_complete(path,installation_sha,task['nominal_id']):by_pair[task['pair_id']][task['mode']]=quiet.read(path)
        else:passed=False
        if path.is_file():artifacts[str(path)]=quiet.sha(path)
    for pair in d['pairs']:
        record=dict(pair_id=pair['pair_id'],case_id=pair['case_id'],backend=pair['backend'],passed=False)
        try:
            group=by_pair[pair['pair_id']]
            if set(group)!=set(CONTROL_MODES):raise ValueError('Trois contrôles complets requis')
            record.update(compare_qualification(group,d,pair))
        except (ValueError,TypeError,KeyError,OSError) as e:record['reason']=str(e);passed=False
        rows.append(record)
    receipt=dict(schema='curvature-persistence-qualification-v1',created_utc=quiet.utc(),passed=passed,
        installation=quiet.artifact(out/'INSTALLATION.json'),controller=quiet.artifact(__file__),
        observations=sum(len(v) for v in by_pair.values()),expected_observations=72,pairs=rows,result_artifacts=artifacts,
        endpoint_clock_exclusion='dictionary keys ending _seconds, recursively',
        intervention_counter_exclusion=sorted(CACHE_COUNTER_FIELDS),
        trace_scope='ordered reanchor and accepted-step correction/state identities; no claim about unrecorded trial/rejected steps',
        timed_runs_authorized=passed)
    receipt['qualification_content_sha256']=quiet.digest(receipt)
    quiet.write(out/'QUALIFICATION.json',receipt)
    return receipt


def verify_qualification(d,out):
    path=out/'QUALIFICATION.json';r=quiet.read(path)
    if r.get('qualification_content_sha256')!=quiet.digest({k:v for k,v in r.items() if k!='qualification_content_sha256'}):
        raise ValueError('Reçu de qualification modifié')
    if r.get('schema')!='curvature-persistence-qualification-v1' or r.get('passed')is not True or r.get('timed_runs_authorized')is not True:
        raise ValueError('Qualification native réussie requise avant chronométrage')
    if r.get('observations')!=72 or r.get('expected_observations')!=72 or len(r.get('pairs',[]))!=24:
        raise ValueError('Qualification incomplète')
    if {p['pair_id'] for p in r['pairs']}!={p['pair_id'] for p in d['pairs']} or any(p.get('passed')is not True for p in r['pairs']):
        raise ValueError('Paires de qualification absentes ou refusées')
    if len(r.get('result_artifacts',{}))!=72:raise ValueError('Résultats de qualification non liés')
    quiet.check_artifact(r['installation']);quiet.check_artifact(r['controller'])
    if r['installation']!=quiet.artifact(out/'INSTALLATION.json') or r['controller']!=quiet.artifact(__file__):
        raise ValueError('Qualification d’une autre installation/source')
    for p,h in r['result_artifacts'].items():quiet.artifact(p,h)
    # Recheck raw evidence, not only the qualification's optimistic booleans.
    for task in d['qualification_schedule']:
        p=out/'qualification/runs'/task['nominal_id']/'RESULT.json'
        if not quiet.result_complete(p,quiet.sha(out/'INSTALLATION.json'),task['nominal_id']):raise ValueError('Contrôle manquant')
    return r


def execute(out,phase):
    out=Path(out).resolve();d=load_installation(out)
    if phase=='timing':verify_qualification(d,out)
    out.mkdir(parents=True,exist_ok=True)
    global_lock=Path('/tmp')/('lazyqcqp_quiet_timing_'+str(os.getuid())+'.lock')
    with (out/'CONTROLLER.lock').open('a') as local,global_lock.open('a') as machine:
        try:fcntl.flock(local,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(machine,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Un autre contrôleur utilise ces résultats ou le chronométrage')
        # Scan both control and timed children before every phase transition.
        quiet.ensure_no_owned_live(out);quiet.ensure_no_owned_live(out/'qualification')
        completed=phase_run(d,out,phase)
        if phase=='qualification':
            r=qualification_gate(d,out)
            if not r['passed']:raise ValueError('Qualification bloquée : '+str(out/'QUALIFICATION.json'))
            print('QUALIFICATION PERSISTANCE VALIDÉE — 72 contrôles',flush=True)
        return completed


def summarize(out):
    out=Path(out).resolve();d=load_installation(out,False);installation_sha=quiet.sha(out/'INSTALLATION.json')
    qualification=verify_qualification(d,out) if (out/'QUALIFICATION.json').is_file() else None
    qualified_pairs={r['pair_id']:r for r in (qualification or {}).get('pairs',[])}
    rows=[];byjob=collections.defaultdict(list)
    for task in d['schedule']:
        path=out/'runs'/task['nominal_id']/'RESULT.json'
        if quiet.result_complete(path,installation_sha,task['nominal_id']):
            r=quiet.read(path)
            if r.get('numerical_core_sha256'):
                actual=quiet.digest(endpoint_core(result_folder(r)/'endpoint.json',True))
                if actual!=r['numerical_core_sha256']:raise ValueError('Identité numérique finale incohérente')
            rows.append(r);byjob[r['job_id']].append(r)
    if rows and not qualification:raise ValueError('Mesures présentes sans qualification vérifiable')
    stats={}
    for pair in d['pairs']:
        for mode in MODES:
            job=pair['pair_id']+'_'+mode;rr=sorted(byjob[job],key=lambda r:r['repetition'])
            complete=len(rr)==3 and {r['repetition'] for r in rr}=={1,2,3};times=[r['seconds_process'] for r in rr]
            counters=[r.get('cache_report',{}).get('totals',{}) for r in rr if r.get('cache_report')]
            stats[job]=dict(complete_three=complete,strict_three=complete and all(r['classification']=='strict' for r in rr),
                repetitions=len(rr),classifications=dict(collections.Counter(r['classification'] for r in rr)),times_seconds=times,
                median_seconds=statistics.median(times) if complete else None,min_seconds=min(times) if complete else None,
                max_seconds=max(times) if complete else None,preparations=[c.get('preparations') for c in counters],
                construction_seconds=[c.get('construction_seconds') for c in counters],rebuild_reset_seconds=[c.get('rebuild_reset_seconds') for c in counters],
                median_construction_seconds=statistics.median([c['construction_seconds'] for c in counters]) if complete and len(counters)==3 else None,
                median_rebuild_reset_seconds=statistics.median([c['rebuild_reset_seconds'] for c in counters]) if complete and len(counters)==3 else None,
                numerical_core_sha256=[r.get('numerical_core_sha256') for r in rr],
                numerical_work_repeat_stable=complete and len({r.get('numerical_core_sha256') for r in rr})==1 and rr[0].get('numerical_core_sha256')is not None)
    pairs=[]
    for p in d['pairs']:
        a=stats[p['pair_id']+'_reuse'];b=stats[p['pair_id']+'_rebuild'];both_strict=a['strict_three'] and b['strict_three']
        numerical_parity=(a['complete_three'] and b['complete_three'] and
            all(x is not None and x==y for x,y in zip(a['numerical_core_sha256'],b['numerical_core_sha256'])))
        repeat_stable=a['numerical_work_repeat_stable'] and b['numerical_work_repeat_stable']
        cache_exercised=qualified_pairs.get(p['pair_id'],{}).get('cache_exercised')is True
        descriptive_eligible=both_strict and numerical_parity and repeat_stable
        eligible=descriptive_eligible and cache_exercised
        pairs.append(dict(pair_id=p['pair_id'],case_id=p['case_id'],backend=p['backend'],regime=p['regime'],
            both_complete_three=a['complete_three'] and b['complete_three'],both_strict_three=both_strict,
            timed_numerical_and_work_parity_three=numerical_parity,timed_repeat_stable=repeat_stable,pure_cache_cost_eligible=eligible,
            cache_exercised=cache_exercised,nonzero_correction_observed=qualified_pairs.get(p['pair_id'],{}).get('nonzero_correction_observed',False),
            reuse_median_seconds=a['median_seconds'],rebuild_median_seconds=b['median_seconds'],
            reuse_median_construction_seconds=a['median_construction_seconds'],rebuild_median_construction_seconds=b['median_construction_seconds'],
            rebuild_median_reset_seconds=b['median_rebuild_reset_seconds'],
            descriptive_ratio_reuse_over_rebuild=a['median_seconds']/b['median_seconds'] if descriptive_eligible and b['median_seconds']>0 else None,
            speedup_rebuild_over_reuse=b['median_seconds']/a['median_seconds'] if eligible and a['median_seconds']>0 else None,
            ratio_reuse_over_rebuild=a['median_seconds']/b['median_seconds'] if eligible and b['median_seconds']>0 else None))
    by_backend=[]
    for backend in ('ma27','mumps'):
        pp=[p for p in pairs if p['backend']==backend];values=[p['ratio_reuse_over_rebuild'] for p in pp if p['pure_cache_cost_eligible']]
        q1,q3=quiet.quartiles(values)
        by_backend.append(dict(backend=backend,planned_pairs=len(pp),complete_pairs=sum(p['both_complete_three'] for p in pp),
            both_strict_three=sum(p['both_strict_three'] for p in pp),pure_cache_cost_eligible=len(values),
            cache_exercised_pairs=sum(p['cache_exercised'] for p in pp),inactive_cache_pairs=sum(not p['cache_exercised'] for p in pp),
            numerical_or_work_mismatch_pairs=sum(p['both_complete_three'] and not (p['timed_numerical_and_work_parity_three'] and p['timed_repeat_stable']) for p in pp),
            excluded_from_speed_summary=len(pp)-len(values),
            median_ratio_reuse_over_rebuild=statistics.median(values) if values else None,ratio_q1=q1,ratio_q3=q3,
            reuse_faster=sum(v<1 for v in values),rebuild_faster=sum(v>1 for v in values)))
    resources=[quiet.read(p) for p in out.glob('**/attempt_*/solver_resources.json')]
    audit_resources=[quiet.read(p) for p in out.glob('**/attempt_*/audit_resources.json')]
    attempts=[quiet.read(p) for p in out.glob('**/attempt_*/ATTEMPT.json')]
    missing=[str(p.parent) for p in out.glob('**/attempt_*/STARTED.json') if not (p.parent/'solver_resources.json').exists()]
    report=dict(schema='curvature-persistence-summary-v1',created_utc=quiet.utc(),protocol=PROTOCOL,
        planned=144,timed_repetitions_completed=len(rows),complete=len(rows)==144,
        qualification_passed=(qualification or {}).get('passed',False),
        cache_exercised_pairs=sum(p.get('cache_exercised',False) for p in (qualification or {}).get('pairs',[])),
        mechanism_informative_pairs=sum(p.get('mechanism_informative',False) for p in (qualification or {}).get('pairs',[])),
        qualification_pairs=(qualification or {}).get('pairs',[]),
        failure_breakdown=dict(collections.Counter(r['classification'] for r in rows)),
        failure_causes=dict(collections.Counter(r['failure_cause'] for r in rows)),
        contaminated_attempts=sum(not r['quiet_valid'] for r in attempts),
        all_attempt_solver_seconds=sum(r['elapsed_seconds'] for r in resources),
        all_attempt_audit_seconds=sum(r['elapsed_seconds'] for r in audit_resources),
        incomplete_attempts_without_resource_record=missing,backends=by_backend,job_statistics=stats,pairs=pairs,
        interpretation='Fixed selected cases; exact shared-model qualification precedes timing. Three quiet repetitions, medians and attrition; all failures and attempts retained. Primary pure-cache ratios require constructions actually avoided in the prior qualification, plus exact timed numerical/work endpoint parity and repetition stability. Non-exercised pairs remain in coverage and receive only a separately labelled descriptive ratio when valid. No selection uses a measured time gain. Divergences remain outcomes and are never rerun for success. Same-center reanchors count as construction events. No whole-corpus speed claim or learned-memory ablation follows.')
    quiet.write(out/'SYNTHESE_PERSISTANCE.json',report)
    with (out/'PAIRES_PERSISTANCE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    with (out/'OBSERVATIONS_PERSISTANCE.csv').open('w',newline='') as f:
        keys=['nominal_id','case_id','backend','regime','mode','repetition','classification','failure_cause','budget_verified','seconds_process']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps({k:report[k] for k in ('complete','timed_repetitions_completed','planned','qualification_passed','mechanism_informative_pairs','backends')},ensure_ascii=False,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    q=s.add_parser('preparer');q.add_argument('--build',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q.add_argument('--ipopt-run',type=Path,default=Path(quiet.DEFAULT_IPOPT_RUN))
    for name in ('qualifier','lancer','synthese'):
        q=s.add_parser(name);q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='preparer':prepare(a.build,a.out,a.ipopt_run)
    elif a.action=='qualifier':execute(a.out,'qualification')
    elif a.action=='lancer':execute(a.out,'timing')
    else:summarize(a.out)


if __name__=='__main__':
    try:main()
    except Exception as e:print('PERSISTANCE BLOQUÉE — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
