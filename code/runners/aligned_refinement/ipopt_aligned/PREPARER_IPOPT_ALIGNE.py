#!/usr/bin/env python3
"""Bind 4,078 unchanged QCQPs and the isolated, qualified Ipopt adapter.
This preparation is independent of the native refinement qualification.
"""
from pathlib import Path
from collections import Counter
import argparse,json,sys
HERE=Path(__file__).resolve().parent;KIT=HERE.parent;BASE=KIT/'runtime_base';sys.path.insert(0,str(BASE))
import CAMPAGNE as base
import FROZEN_INPUTS
import IPOPT_ALIGNE_CONTRACT as contract

def artifact(path):
    p=Path(path).expanduser().resolve(strict=True)
    if not p.is_file():raise ValueError('Fichier requis : '+str(p))
    return dict(path=str(p),sha256=base.sha(p))

def assert_artifact(item):
    if not isinstance(item,dict)or not item.get('path')or base.sha(item['path'])!=item.get('sha256'):raise ValueError('Pièce absente ou modifiée : '+str(item))

def fresh(path,protected):
    p=Path(path).expanduser().resolve()
    if p.exists():raise ValueError('Dossier neuf requis : '+str(p))
    for item in protected:
        q=Path(item).resolve()
        if p==q or p in q.parents or q in p.parents:raise ValueError('La sortie doit être séparée des sources et données : '+str(p))
    return p

def load_build(path):
    path=Path(path).resolve(strict=True);b=base.read(path)
    if b.get('schema')!='isolated-ipopt-aligned-options-build-v1':raise ValueError('Construction Ipopt alignée requise')
    if b.get('original_installation_unchanged')is not True:raise ValueError('Préservation de l’installation non démontrée')
    if b.get('production_dependencies_verified')is not True or b.get('dependencies_rehashed_after_build')is not True:raise ValueError('Dépendances TRX non qualifiées pour la production')
    for key in ('binary','ipopt_library','hsl_library'):
        if not b.get(key)or base.sha(b[key])!=b.get(key+'_sha256'):raise ValueError('Identité non vérifiée : '+key)
    if not isinstance(b.get('source_sha256'),dict)or not b['source_sha256']:raise ValueError('Sources de l’adaptateur non liées')
    source_root=Path(b.get('source_directory',HERE/'adapter')).resolve(strict=True)
    for source,digest in b['source_sha256'].items():
        p=Path(source);p=p if p.is_absolute()else source_root/p
        if base.sha(p)!=digest:raise ValueError('Source adaptateur modifiée : '+source)
    if set(b.get('backends',[]))!={'ma27','mumps'}or set(b.get('strategies',[]))!={'adaptive','monotone'}:raise ValueError('Construction sans les quatre cellules')
    if base.sha(HERE/'PATCH_CONTRACT.json')!=b.get('patch_contract_sha256'):raise ValueError('Contrat de compilation modifié')
    if not b.get('dependency_receipt')or base.sha(b['dependency_receipt'])!=b.get('dependency_receipt_sha256'):raise ValueError('Reçu de dépendances modifié')
    for name,value in b.get('ipopt_headers_sha256',{}).items():
        if base.sha(Path(b['ipopt_headers'])/name)!=value:raise ValueError('En-tête Ipopt modifié : '+name)
    if not b.get('ipopt_headers_sha256'):raise ValueError('En-têtes de la compilation non liés')
    for key in ('linked_libraries','hsl_linked_libraries'):
        links=b.get(key)
        if not isinstance(links,dict)or not links:raise ValueError('Bibliothèques effectivement liées non prouvées : '+key)
        for name,item in links.items():
            if not isinstance(item,dict)or not item.get('path')or base.sha(item['path'])!=item.get('sha256'):raise ValueError('Bibliothèque transitive modifiée : '+name)
    if not isinstance(b.get('runtime_environment',{}),dict)or not b.get('ipopt_version'):raise ValueError('Contexte de compilation absent')
    return b

def check_qualification(path,build_path,build):
    q=base.read(path)
    if not isinstance(q,dict)or q.get('schema')!='ipopt-aligned-qualification-v1':raise ValueError('Qualification Ipopt alignée absente')
    for key in ('passed','options_verified','original_audit_verified'):
        if q.get(key)is not True:raise ValueError('Qualification Ipopt insuffisante : '+key)
    if q.get('runs')!=4 or q.get('cases')!=4:raise ValueError('Les quatre contrôles backend/stratégie sont requis')
    for key,value in [('build_sha256',base.sha(build_path)),('binary_sha256',build['binary_sha256']),('ipopt_version',build['ipopt_version'])]:
        if q.get(key)!=value:raise ValueError('Qualification liée à un autre objet : '+key)
    if set(q.get('backends',[]))!={'ma27','mumps'}or set(q.get('strategies',[]))!={'adaptive','monotone'}:raise ValueError('Les quatre cellules doivent être qualifiées')
    if q.get('production_dependencies_verified')is not True or q.get('ambient_hostile_option_file_ignored')is not True or q.get('dependencies_rehashed_after_smokes')is not True:raise ValueError('Qualification des dépendances ou du fichier d’options insuffisante')
    smokes=q.get('smokes',[])
    if len(smokes)!=4 or {(r.get('backend'),r.get('strategy'))for r in smokes}!={(v['backend'],v['strategy'])for v in contract.CELLS.values()}:raise ValueError('Contrôles backend/stratégie incomplets ou dupliqués')
    for smoke in smokes:
        if smoke.get('passed')is not True or smoke.get('options_verified')is not True or smoke.get('returncode')!=0:raise ValueError('Contrôle Ipopt non réussi')
        for key in ('endpoint','log'):
            if base.sha(smoke[key])!=smoke.get(key+'_sha256'):raise ValueError('Preuve de qualification modifiée : '+key)
        label=f"ipopt_{smoke['backend']}_{smoke['strategy']}_aligned"
        command=smoke.get('command',[])
        if '--problem'not in command:raise ValueError('Entrée de contrôle non liée')
        problem=Path(command[command.index('--problem')+1])
        if base.sha(problem)!=q.get('input_sha256'):raise ValueError('QCQP de qualification modifié')
        contract.verify_endpoint(base.read(smoke['endpoint']),label,build['ipopt_version'],build['hsl_library'],contract.representation_for_input(problem))
        audit=smoke.get('original_audit',{})
        if audit.get('strict')is not True or any(not contract.finite_number(audit.get(k))or not 0<=audit[k]<=1e-6 for k in('primal','dual','complementarity')):raise ValueError('Certificat analytique de contrôle absent')
    return q

def verify_corpus(cases,evidence):
    if Counter(c['regime']for c in cases)!=contract.REGIMES:raise ValueError('Les quatre régimes du corpus sont requis')
    published={x.get('id',x.get('case_id')):x for x in evidence.get('plan_header',{}).get('cases',[])}
    if set(published)!={c['case_id']for c in cases}:raise ValueError('Le contrat ne correspond pas au plan source')
    for c in cases:
        p=published[c['case_id']];b=c['start_binding']
        for name,actual,wanted in [('input',c['input_sha256'],p.get('input_sha256')),('start',b.get('start_sha256'),p.get('start_sha256')),('raw_x0',b.get('raw_x0_line_sha256'),p.get('raw_x0_line_sha256')),('regime',c['regime'],p.get('regime')),('dataset',c['dataset'],p.get('dataset_id',p.get('dataset')))]:
            if actual!=wanted:raise ValueError('Entrée ou départ divergent : '+c['case_id']+'/'+name)

def prepare(root,run,build_path,out,qualification=None,contract_path=None,resolution_source=None,progress=print):
    root=Path(root).expanduser().resolve(strict=True);run=Path(run).expanduser().resolve(strict=True)
    source_installation=root/'INSTALLATION.json';plan=run/'PLAN.json';context=KIT/'evidence/CURRENT_PLAN_COMPACT.json'
    source=base.read(source_installation)
    if not isinstance(source,dict):raise ValueError('Installation source absente')
    evidence=base.read(context)
    if not plan.is_file()or base.sha(plan)!=evidence.get('source_plan_sha256'):raise ValueError('Plan contemporain différent de celui du corpus lié')
    build_path=Path(build_path).resolve(strict=True);build=load_build(build_path)
    qualification=Path(qualification or build_path.parent/'QUALIFICATION.json').resolve(strict=True)
    q=check_qualification(qualification,build_path,build)
    corpus_path=Path(contract_path or BASE/'CONTRAT_4078.json.gz').resolve(strict=True)
    cases=base.plan_cases(base.contract(corpus_path));verify_corpus(cases,evidence)
    resolution_source=Path(resolution_source or KIT/'evidence/FROZEN_RESOLUTION_SOURCE.json').resolve(strict=True)
    old_resolution=base.read(resolution_source)
    if old_resolution.get('all_inputs_and_raw_starts_verified')is not True:raise ValueError('Reçu de résolution gelée incomplet')
    extras=[Path(x).resolve()for x in old_resolution.get('frozen_roots',[])]
    out=fresh(out,[root,run,KIT,*extras]);out.mkdir(parents=True)
    paths=[source_installation,plan,context,build_path,qualification,corpus_path,resolution_source,
        Path(__file__).resolve(),HERE/'IPOPT_ALIGNE_CONTRACT.py',HERE/'CONTROLEUR_IPOPT_ALIGNE.py',HERE/'BUILD_IPOPT_ALIGNED.py',HERE/'QUALIFIER_IPOPT_ALIGNE.py',
        BASE/'CAMPAGNE.py',BASE/'AUDIT_ORIGINAL.py',BASE/'FROZEN_INPUTS.py',BASE/'frozen_qcqp_reader.py']
    sources={str(p.resolve()):base.sha(p)for p in paths}
    try:
        progress('RÉSOLUTION ET VÉRIFICATION DES 4 078 QCQP ET DÉPARTS GELÉS — démarrage')
        bindings,_=FROZEN_INPUTS.resolve_all(root,cases,out/'FROZEN_RESOLUTION.json',extra_roots=extras,progress=lambda phase,counts:progress('ENTRÉES GELÉES — '+phase+' '+json.dumps(counts,ensure_ascii=False)))
        if set(bindings)!={c['case_id']for c in cases}:raise ValueError('Résolution gelée incomplète')
        representations={}
        for index,case in enumerate(cases,1):
            cid=case['case_id'];representations[cid]=contract.representation_for_input(bindings[cid]['input'])
            if index%100==0 or index==len(cases):progress(f'REPRÉSENTATION EXACTE DES BORNES {index} / {len(cases)}')
        for p,h in sources.items():
            if base.sha(p)!=h:raise ValueError('Source ou reçu modifié pendant la préparation : '+p)
        d=dict(schema='ipopt-aligned-qualified-installation-v1',production_authorized=True,created_utc=base.utc(),
            root=str(root),source_run=str(run),source_plan_sha256=base.sha(plan),cases=cases,cells=contract.CELLS,
            case_matrix_sha256=contract.json_sha(cases),binary=build['binary'],binary_sha256=build['binary_sha256'],
            ipopt_library=build['ipopt_library'],ipopt_library_sha256=build['ipopt_library_sha256'],
            hsl_library=build['hsl_library'],hsl_library_sha256=build['hsl_library_sha256'],ipopt_version=build['ipopt_version'],
            runtime_environment=build.get('runtime_environment',{}),python=sys.executable,build=artifact(build_path),
            qualification=artifact(qualification),qualification_summary=q,frozen_resolution=artifact(out/'FROZEN_RESOLUTION.json'),
            frozen_bindings=bindings,input_representation_expectations=representations,representation_matrix_sha256=contract.json_sha(representations),contract=artifact(corpus_path),source_snapshot=sources,
            expected_options=contract.ALIGNED_OPTIONS,budgets=dict(seconds=7200,max_iter=5000,tolerance=1e-6,threads=1),
            cpus=list(contract.CPUS),tasks=contract.TASKS,optimizer_invocations=0,input_files_modified=False,
            independent_of_refinement_qualification=True)
        base.atomic(out/'PREPARATION.json',dict(d,schema='ipopt-aligned-preparation-v1'))
        d['preparation']=artifact(out/'PREPARATION.json')
        base.atomic(out/'INSTALLATION_IPOPT_ALIGNE.json',d)
        progress('IPOPT ALIGNÉ PRÊT — 4 078 cas × 4 = 16 312 tâches ; aucun départ modifié')
        return out/'INSTALLATION_IPOPT_ALIGNE.json'
    except BaseException as error:
        base.atomic(out/'PREPARATION_ERROR.json',dict(error=type(error).__name__+': '+str(error),production_authorized=False,input_files_modified=False,optimizer_invocations=0));raise

def load_installation(path):
    path=Path(path).resolve(strict=True);d=base.read(path)
    if d.get('schema')!='ipopt-aligned-qualified-installation-v1'or d.get('production_authorized')is not True:raise ValueError('Installation Ipopt non qualifiée')
    for key in ('build','qualification','frozen_resolution','contract','preparation'):assert_artifact(d[key])
    prep=base.read(d['preparation']['path'])
    if prep.get('schema')!='ipopt-aligned-preparation-v1':raise ValueError('Préparation contemporaine absente')
    for key,value in prep.items():
        if key!='schema'and d.get(key)!=value:raise ValueError('Installation différente de sa préparation : '+key)
    build=load_build(d['build']['path']);check_qualification(d['qualification']['path'],d['build']['path'],build)
    for key in ('binary','binary_sha256','ipopt_library','ipopt_library_sha256','hsl_library','hsl_library_sha256','ipopt_version','runtime_environment'):
        if d.get(key)!=build.get(key,{}if key=='runtime_environment'else None):raise ValueError('Installation différente du build : '+key)
    for p,h in d['source_snapshot'].items():
        if base.sha(p)!=h:raise ValueError('Source ou reçu modifié : '+p)
    cases=base.plan_cases(base.contract(d['contract']['path']));verify_corpus(cases,base.read(KIT/'evidence/CURRENT_PLAN_COMPACT.json'))
    if d.get('cases')!=cases or d.get('case_matrix_sha256')!=contract.json_sha(cases):raise ValueError('Corpus modifié')
    if d.get('cells')!=contract.CELLS or d.get('expected_options')!=contract.ALIGNED_OPTIONS:raise ValueError('Cellules ou options modifiées')
    if d.get('budgets')!=dict(seconds=7200,max_iter=5000,tolerance=1e-6,threads=1)or d.get('cpus')!=list(contract.CPUS)or d.get('tasks')!=contract.TASKS:raise ValueError('Budgets ou plan modifiés')
    resolved=base.read(d['frozen_resolution']['path'])
    if resolved.get('all_inputs_and_raw_starts_verified')is not True:raise ValueError('Résolution gelée non vérifiée')
    # Re-hash each selected file on actual use; the binding table is immutable.
    if set(d['frozen_bindings'])!={c['case_id']for c in cases}:raise ValueError('Table de résolution incomplète')
    if set(d.get('input_representation_expectations',{}))!=set(d['frozen_bindings'])or d.get('representation_matrix_sha256')!=contract.json_sha(d['input_representation_expectations']):raise ValueError('Table de représentation des bornes modifiée ou incomplète')
    rows={r['case_id']:r for r in resolved.get('rows',[])}
    if set(rows)!=set(d['frozen_bindings']):raise ValueError('Reçu de résolution incomplet')
    for c in cases:
        cid=c['case_id'];row=rows[cid]
        binding=dict(input=row.get('selected_input'),input_sha256=row.get('expected_sha256'),raw_x0_line_sha256=row.get('raw_x0_line_sha256'),start_sha256=c['start_binding'].get('start_sha256'))
        if row.get('status')!='verified'or row.get('raw_x0_verified')is not True or d['frozen_bindings'][cid]!=binding:raise ValueError('Chemin gelé différent du reçu : '+cid)
    d['_path']=str(path);d['_sha256']=base.sha(path);return d

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','run','build','out'):p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--qualification',type=Path);p.add_argument('--contract',type=Path);p.add_argument('--resolution-source',type=Path)
    a=p.parse_args();path=prepare(a.root,a.run,a.build,a.out,a.qualification,a.contract,a.resolution_source)
    print('INSTALLATION_IPOPT_ALIGNE '+str(path),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:print('PRÉPARATION IPOPT REFUSÉE — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
