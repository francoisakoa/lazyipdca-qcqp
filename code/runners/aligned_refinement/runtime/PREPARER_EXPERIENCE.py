#!/usr/bin/env python3
"""Bind the exact frozen corpus and isolated native build, then promote only
against the completed trajectory and intervention qualification. No QCQP is
copied, generated, optimized or used as an external warm start here.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gzip,hashlib,importlib.util,json,math,os,sys,time
HERE=Path(__file__).resolve().parent;KIT=HERE.parent;BASE=KIT/'runtime_base'
sys.path.insert(0,str(BASE))
import CAMPAGNE as common
import FROZEN_INPUTS
import PROFILES_EXPERIENCE as profiles

def sha(path):return common.sha(path)
def atomic(path,obj):return common.atomic(path,obj)
def read(path):return json.loads(Path(path).read_text())
def exact_json_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def artifact(path):
    p=Path(path).expanduser().resolve(strict=True)
    if not p.is_file():raise ValueError('Fichier attendu : '+str(p))
    return {'path':str(p),'sha256':sha(p)}
def assert_artifact(item):
    if not isinstance(item,dict)or not item.get('path')or not item.get('sha256')or sha(item['path'])!=item['sha256']:raise ValueError('Pièce modifiée ou absente : '+str(item))
def fresh(path,protected):
    path=Path(path).expanduser().resolve()
    if path.exists():raise ValueError('Dossier neuf requis : '+str(path))
    for p in protected:
        p=Path(p).resolve()
        if path==p or p in path.parents or path in p.parents:raise ValueError('La sortie doit être séparée des sources et données : '+str(path))
    return path

def load_build(path):
    path=Path(path).resolve(strict=True);b=read(path)
    if b.get('schema')!='lazy-refinement-build-v1':raise ValueError('Schéma de construction expérimental inconnu')
    if b.get('same_source_and_binary_for_both_backends')is not True:raise ValueError('Les deux backends doivent partager les mêmes exécutables')
    if b.get('homogeneous_compilation')is not True:raise ValueError('Compilation homogène non prouvée')
    if b.get('original_installation_unchanged')is not True:raise ValueError('Préservation de l’installation non prouvée')
    backends=b.get('backends')
    if not isinstance(backends,dict)or set(backends)!=set(profiles.BACKENDS):raise ValueError('Les deux backends MA27/MUMPS sont requis')
    for backend,cfg in backends.items():
        for kind in ('reference','observer','experiment'):
            key=kind+'_binary';expected=cfg.get(key+'_sha256')
            if not cfg.get(key)or not expected or sha(cfg[key])!=expected:raise ValueError('Identité binaire non vérifiée : '+backend+'/'+kind)
        if not isinstance(cfg.get('environment',{}),dict):raise ValueError('Environnement de backend invalide')
    for role in ('reference','observer','experiment'):
        if len({cfg[role+'_binary_sha256']for cfg in backends.values()})!=1:raise ValueError('Exécutables différents entre backends : '+role)
    for key in ('reference_source_sha256','observer_source_sha256','experiment_source_sha256'):
        if not isinstance(b.get(key),dict)or not b[key]:raise ValueError('Empreintes de sources absentes : '+key)
    return b

def verified_registration(path,selected):
    path=Path(path).resolve(strict=True);d=read(path)
    if d.get('production_authorized')is True:raise ValueError('Un préenregistrement ne peut pas autoriser la production')
    # The definition may carry the list under inputs.cases or cases. If present,
    # it must match every frozen identifier; the current manifest separately
    # binds the complete list and the study-contract bytes in all cases.
    listed=d.get('cases') or d.get('inputs',{}).get('cases')
    if isinstance(listed,list)and listed:
        ids={x.get('case_id',x.get('id'))if isinstance(x,dict)else x for x in listed}
        if ids!={x['case_id']for x in selected}:raise ValueError('Préenregistrement lié à un autre périmètre')
    return d

def declared_resolution(path):
    if path is None:return [],None
    path=Path(path).resolve(strict=True);d=read(path)
    if d.get('all_inputs_and_raw_starts_verified')is not True:raise ValueError('Reçu d’entrées antérieur incomplet')
    roots=[Path(x).expanduser().resolve()for x in d.get('frozen_roots',[])]
    return roots,artifact(path)

def prepare(root,run,build_path,out,study_contract,registration=None,contract_path=None,resolution_source=None,progress=print):
    root=Path(root).expanduser().resolve(strict=True);run=Path(run).expanduser().resolve(strict=True)
    installation=root/'INSTALLATION.json'
    if not installation.is_file():raise ValueError('Installation de référence absente : '+str(installation))
    plan_path=run/'PLAN.json'
    if not plan_path.is_file():raise ValueError('PLAN contemporain absent : '+str(plan_path))
    expected_plan=KIT/'evidence/CURRENT_PLAN_COMPACT.json'
    if not expected_plan.is_file():raise ValueError('Contexte de plan fourni absent : '+str(expected_plan))
    evidence=read(expected_plan)
    if sha(plan_path)!=evidence.get('source_plan_sha256'):raise ValueError('Le run source ne correspond pas au plan lié aux sources')
    cfg=read(installation);build=load_build(build_path)
    study_contract=Path(study_contract).resolve(strict=True);registration=Path(registration or study_contract).resolve(strict=True)
    contract_path=Path(contract_path or BASE/'CONTRAT_4078.json.gz').resolve(strict=True)
    c=common.contract(contract_path);cases=common.plan_cases(c)
    from collections import Counter
    if Counter(x['regime']for x in cases)!={'I':1037,'M':797,'E':1704,'Q':540}:raise ValueError('Périmètre original des quatre régimes modifié')
    published={x.get('id',x.get('case_id')):x for x in evidence.get('plan_header',{}).get('cases',[])}
    if set(published)!={x['case_id']for x in cases}:raise ValueError('Le contrat diffère du plan de source lié')
    for case in cases:
        source=published[case['case_id']];binding=case['start_binding']
        for key,actual,wanted in [('input',case['input_sha256'],source.get('input_sha256')),('raw_x0',binding.get('raw_x0_line_sha256'),source.get('raw_x0_line_sha256')),('start',binding.get('start_sha256'),source.get('start_sha256')),('regime',case['regime'],source.get('regime')),('dataset',case['dataset'],source.get('dataset_id',source.get('dataset')))]:
            if actual!=wanted:raise ValueError('Contrat et plan divergent : '+case['case_id']+'/'+key)
    selected=[x for x in cases if x['regime']in('I','M')]
    if len(selected)!=1834:raise ValueError('Les 1 834 cas I/M sont requis')
    verified_registration(registration,selected);study=read(study_contract)
    for key,value in [('common_guards','r3'),('common_representation','original_qcqp'),('objective_exact',True),('inequality_policy','hybrid')]:
        if study.get(key)!=value:raise ValueError('Contrat scientifique différent : '+key)
    if study.get('configurations')!={label:dict(c,cell=label.rsplit('_',1)[0])for label,c in profiles.CELLS.items()}:raise ValueError('Cellules scientifiques différentes')
    expected_bindings={c['case_id']:(c['input_sha256'],c['start_binding']['start_sha256'],c['regime'],c['dataset'],c.get('family'))for c in selected}
    declared_bindings={c['case_id']:(c['input_sha256'],c['start_sha256'],c['regime'],c['dataset'],c.get('family'))for c in study.get('inputs',{}).get('case_bindings',[])}
    if declared_bindings!=expected_bindings:raise ValueError('Empreintes ou groupes des cas scientifiques différents')
    for key,value in [('seconds_per_task',7200),('max_outer',5000),('shared_restoration',600),('threads_per_task',1),('jobs',1),('cpus',[23]),('tasks',14672),('new_Ipopt_runs',0)]:
        if study.get('production',{}).get(key)!=value:raise ValueError('Ressource scientifique différente : '+key)
    if build.get('study_contract_sha256')!=sha(study_contract):raise ValueError('Construction liée à une autre étude')
    extras,resolution_receipt=declared_resolution(resolution_source)
    out=fresh(out,[root,run,KIT,*extras]);out.mkdir(parents=True)
    bound={str(installation):sha(installation),str(plan_path):sha(plan_path),str(Path(build_path).resolve()):sha(build_path),str(study_contract):sha(study_contract),str(registration):sha(registration),str(contract_path):sha(contract_path)}
    for source in [HERE/'PROFILES_EXPERIENCE.py',HERE/'REFERENCE_PROFILES.json',HERE/'CONTROLEUR_EXPERIENCE.py',Path(__file__).resolve(),BASE/'CAMPAGNE.py',BASE/'AUDIT_ORIGINAL.py',BASE/'frozen_qcqp_reader.py',BASE/'FROZEN_INPUTS.py',KIT/'qualification/QUALIFIER_EXPERIENCE.py']:
        bound[str(source)]=sha(source)
    seal_module=KIT/'analysis/SCELLER_ENREGISTREMENT.py'
    for source in (seal_module,KIT/'SOURCE_SEPARATION_AUDIT.json',KIT/'DIRECTION_TRACE_SCHEMA.json'):
        if not source.is_file():raise ValueError('Pièce de scellement absente : '+str(source))
        bound[str(source)]=sha(source)
    try:
        def report(phase,counts):progress('ENTRÉES GELÉES — '+phase+' '+json.dumps(counts,ensure_ascii=False))
        progress('RÉSOLUTION EXACTE DES 4 078 ENTRÉES ET DÉPARTS — démarrage')
        bindings,resolution=FROZEN_INPUTS.resolve_all(root,cases,out/'FROZEN_RESOLUTION.json',extra_roots=extras,progress=report)
        if set(bindings)!={x['case_id']for x in cases}:raise ValueError('Résolution gelée incomplète')
        # Recheck source receipts after the read-only scan and retain those
        # bytes as promotion requirements, never overwrite their hashes.
        for p,h in bound.items():
            if sha(p)!=h:raise ValueError('Pièce modifiée pendant la préparation : '+p)
        hsl=cfg.get('hsl_library')or evidence.get('plan_header',{}).get('installation',{}).get('hsl_library')
        if not hsl or not Path(hsl).is_file():raise ValueError('Bibliothèque MA27/HSL déclarée absente')
        expected_hsl=cfg.get('hsl_library_sha256')or evidence.get('plan_header',{}).get('installation',{}).get('hsl_library_sha256')
        if not expected_hsl or sha(hsl)!=expected_hsl:raise ValueError('Bibliothèque HSL différente de la référence déclarée')
        record=dict(schema='lazy-refinement-preparation-v1',created_utc=datetime.now(timezone.utc).isoformat(),root=str(root),source_run=str(run),source_plan_sha256=sha(plan_path),contract=str(contract_path),contract_sha256=sha(contract_path),all_cases=cases,cases=selected,frozen_bindings=bindings,frozen_resolution=artifact(out/'FROZEN_RESOLUTION.json'),prior_resolution_receipt=resolution_receipt,build_manifest=str(Path(build_path).resolve()),build_manifest_sha256=sha(build_path),build=build,profiles=profiles.command_templates(),profile_factory=artifact(HERE/'PROFILES_EXPERIENCE.py'),profile_evidence=profiles.evidence(),study_contract=str(study_contract),study_contract_sha256=sha(study_contract),registration=str(registration),registration_sha256=sha(registration),hsl_library=str(Path(hsl).resolve()),hsl_library_sha256=sha(hsl),python=sys.executable,budgets=dict(seconds=7200,max_outer=5000,shared_restoration=600,tolerance=1e-6),cpus=[23],jobs=1,threads=1,source_snapshot=bound,production_authorized=False,case_matrix_sha256=exact_json_hash(selected),selected_regimes=['I','M'],planned_native_tasks=14672,optimizer_invocations=0,input_files_modified=False)
        atomic(out/'PREPARATION.json',record)
        progress('PRÉPARATION LIÉE — 1 834 cas × 8 ; qualification avant production requise')
        return out/'PREPARATION.json'
    except BaseException as error:
        atomic(out/'PREPARATION_ERROR.json',dict(error=type(error).__name__+': '+str(error),production_authorized=False,input_files_modified=False,optimizer_invocations=0));raise

def promote(preparation,qualification_gate,study_contract=None):
    preparation=Path(preparation).resolve(strict=True);d=read(preparation)
    if d.get('schema')!='lazy-refinement-preparation-v1'or d.get('production_authorized')is not False:raise ValueError('Préparation non admissible')
    for p,h in d['source_snapshot'].items():
        if sha(p)!=h:raise ValueError('Source ou reçu de préparation modifié : '+p)
    for key in ('frozen_resolution','profile_factory'):assert_artifact(d[key])
    if profiles.evidence()!=d['profile_evidence']:raise ValueError('Fabrique ou profils modifiés')
    b=load_build(d['build_manifest'])
    if sha(d['build_manifest'])!=d['build_manifest_sha256']or b!=d['build']:raise ValueError('Construction différente')
    study=Path(study_contract or d['study_contract']).resolve(strict=True)
    if str(study)!=d['study_contract']or sha(study)!=d['study_contract_sha256']:raise ValueError('Contrat d’étude modifié')
    gate_path=Path(qualification_gate).resolve(strict=True);gate=read(gate_path)
    for key in ('passed','production_authorized','reference_reproduction_passed','study_intervention_verified'):
        if gate.get(key)is not True:raise ValueError('Qualification insuffisante : '+key)
    if gate.get('cases')!=50:raise ValueError('Qualification requise sur 50 cas fixés')
    for k,value in [('build_manifest_sha256',d['build_manifest_sha256']),('study_contract_sha256',d['study_contract_sha256']),('registration_sha256',d['registration_sha256'])]:
        if gate.get(k)!=value:raise ValueError('Qualification liée à un autre objet : '+k)
    # Qualification must also bind this exact input table, not a nominal path
    # reconstruction. Gate producers may use either explicit spelling.
    prep_hash=gate.get('preparation_sha256',gate.get('preparation_manifest_sha256'))
    if prep_hash!=sha(preparation):raise ValueError('Qualification non liée à cette préparation')
    target=preparation.parent/'EXPERIMENT_INSTALLATION.json'
    if target.exists():
        existing=read(target)
        if existing.get('preparation')!=artifact(preparation)or existing.get('qualification_gate')!=artifact(gate_path):raise ValueError('Installation existante liée à une autre qualification')
        import CONTROLEUR_EXPERIENCE as controller
        controller.load_installation(target)
        return target
    seal_path=KIT/'analysis/SCELLER_ENREGISTREMENT.py'
    spec=importlib.util.spec_from_file_location('refinement_production_seal',seal_path);sealer=importlib.util.module_from_spec(spec);spec.loader.exec_module(sealer)
    sealed=preparation.parent/'PRODUCTION_REGISTRATION.json'
    if sealed.exists():
        value=read(sealed)
        for key,item in [('study_contract',artifact(study)),('preparation',artifact(preparation)),('build',artifact(d['build_manifest'])),('gate',artifact(gate_path))]:
            if value.get('evidence',{}).get(key)!=item:raise ValueError('Sceau existant lié à une autre preuve : '+key)
        for item in value.get('evidence',{}).values():assert_artifact(item)
        if value.get('schema')!='adaptive-refinement-production-registration-v2'or value.get('registration_status')!='sealed_before_production':raise ValueError('Sceau existant non admissible')
    else:
        sealed=sealer.seal_registration(study,preparation,d['build_manifest'],gate_path,sealed,KIT/'SOURCE_SEPARATION_AUDIT.json',KIT/'DIRECTION_TRACE_SCHEMA.json')
    sealed_value=read(sealed)
    guards=gate.get('intervention_checks',{}).get('guard_config_sha256_by_backend')
    if not isinstance(guards,dict)or set(guards)!=set(profiles.BACKENDS):raise ValueError('Garde-fous qualifiés absents pour les deux backends')
    result=dict(d,schema='lazy-refinement-qualified-installation-v1',production_authorized=True,preparation=artifact(preparation),qualification_gate=artifact(gate_path),qualification_summary=gate,qualification_registration=d['registration'],qualification_registration_sha256=d['registration_sha256'],registration=str(sealed),registration_sha256=sha(sealed),registration_status='sealed_before_production',registration_evidence=sealed_value['evidence'],qualified_guard_config_sha256_by_backend=guards,promoted_utc=datetime.now(timezone.utc).isoformat())
    atomic(target,result);return target

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path);ap.add_argument('--run',type=Path);ap.add_argument('--build',type=Path);ap.add_argument('--out',type=Path)
    ap.add_argument('--study-contract',type=Path,default=KIT/'STUDY_CONTRACT.json');ap.add_argument('--registration',type=Path);ap.add_argument('--contract',type=Path,default=BASE/'CONTRAT_4078.json.gz')
    ap.add_argument('--resolution-source',type=Path,default=KIT/'evidence/FROZEN_RESOLUTION_SOURCE.json')
    ap.add_argument('--promouvoir',type=Path);ap.add_argument('--qualification-gate',type=Path)
    a=ap.parse_args()
    if a.promouvoir:
        if not a.qualification_gate:raise ValueError('--qualification-gate requis')
        target=promote(a.promouvoir,a.qualification_gate,a.study_contract)
        print('EXPÉRIENCE QUALIFIÉE — '+str(target),flush=True)
    else:
        for key in ('root','run','build','out'):
            if getattr(a,key)is None:raise ValueError('--'+key+' requis')
        target=prepare(a.root,a.run,a.build,a.out,a.study_contract,a.registration,a.contract,a.resolution_source,lambda s:print(s,flush=True))
        print('PREPARATION_JSON '+str(target),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:print('PRÉPARATION REFUSÉE — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
