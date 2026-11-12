#!/usr/bin/env python3
"""One fixed-CPU continuous queue for the eight native refinement cells.
The completed default Ipopt runs are not substituted for any factorial cell.
"""
from __future__ import annotations
from pathlib import Path
import argparse,collections,fcntl,hashlib,importlib.util,json,math,os,random,re,signal,struct,sys
HERE=Path(__file__).resolve().parent;KIT=HERE.parent;BASE=KIT/'runtime_base';sys.path.insert(0,str(BASE))
import CAMPAGNE as base
import PROFILES_EXPERIENCE as profiles
import PREPARER_EXPERIENCE as prepare
CELLS=tuple(profiles.CELLS);SEED=20261109;CPUS=(23,);JOBS=1
HEX=re.compile(r'[0-9a-f]{64}')
STATES={'startup_restoration_commit','state_recovery_commit','terminal_central_restart_commit','post_multiplier_update','central_restart_commit'}

def cpu_pool_lock():
    path=Path('/tmp/LAZYQCQP_CPU_23.lock')
    try:fd=os.open(path,os.O_RDWR|os.O_CREAT,0o666)
    except PermissionError:fd=os.open(path,os.O_RDONLY)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise ValueError('Le cœur 23 est déjà réservé par une ablation de ce kit')
    return os.fdopen(fd,'r')

def hex64(value):return isinstance(value,str)and HEX.fullmatch(value)is not None

def initial_model_identity(result):
    state=result.get('initial_model_state')
    return (state,result.get('initial_inequality_model_sha256'))if state in('reached','not_reached')else None

def process_budget_verified(resources):
    elapsed=resources.get('elapsed_seconds');code=resources.get('returncode')
    return (type(elapsed)in(int,float)and math.isfinite(elapsed)and 0<=elapsed<=7200 and type(code)is int and code>=0 and resources.get('censored')is False and resources.get('timed_out')is not True and resources.get('interrupted')is not True)

def load_installation(path):
    path=Path(path).resolve(strict=True);d=base.read(path)
    if d.get('schema')!='lazy-refinement-qualified-installation-v1'or d.get('production_authorized')is not True:raise ValueError('Installation non qualifiée pour la production')
    for key in ('preparation','qualification_gate','frozen_resolution','profile_factory'):prepare.assert_artifact(d[key])
    original_preparation=base.read(d['preparation']['path'])
    for key,value in original_preparation.items():
        if key not in ('schema','production_authorized','registration','registration_sha256')and d.get(key)!=value:raise ValueError('Installation différente de la préparation qualifiée : '+key)
    gate=base.read(d['qualification_gate']['path'])
    for key in ('passed','production_authorized','reference_reproduction_passed','study_intervention_verified'):
        if gate.get(key)is not True:raise ValueError('Qualification insuffisante : '+key)
    if gate.get('cases')!=50:raise ValueError('Les 50 cas fixés sont requis')
    for key in ('build_manifest_sha256','study_contract_sha256'):
        if gate.get(key)!=d[key]:raise ValueError('Qualification liée à une autre étude : '+key)
    if gate.get('registration_sha256')!=d.get('qualification_registration_sha256')or original_preparation.get('registration_sha256')!=d.get('qualification_registration_sha256'):raise ValueError('Protocole de qualification différent')
    sealed=base.read(d['registration'])
    if sealed.get('schema')!='adaptive-refinement-production-registration-v2'or sealed.get('registration_status')!='sealed_before_production':raise ValueError('Sceau de production absent')
    if sealed.get('study_contract_sha256')!=d['study_contract_sha256']or sealed.get('qualification_registration_sha256')!=d['qualification_registration_sha256']:raise ValueError('Sceau lié à une autre étude')
    for item in d['registration_evidence'].values():prepare.assert_artifact(item)
    if sealed.get('evidence')!=d['registration_evidence']:raise ValueError('Preuves du sceau modifiées')
    guards=gate.get('intervention_checks',{}).get('guard_config_sha256_by_backend')
    if d.get('qualified_guard_config_sha256_by_backend')!=guards:raise ValueError('Garde-fous qualifiés du sceau différents')
    evidence=sealed['evidence'];bindings=sealed.get('bindings',{})
    for key,item in [('study_contract',prepare.artifact(d['study_contract'])),('preparation',d['preparation']),('build',prepare.artifact(d['build_manifest'])),('gate',d['qualification_gate'])]:
        if evidence.get(key)!=item:raise ValueError('Sceau lié à une autre preuve : '+key)
    expected_bindings=dict(source_tree_sha256=prepare.exact_json_hash(d['build']['experiment_source_sha256']),binary_sha256_by_backend={b:d['build']['backends'][b]['experiment_binary_sha256']for b in profiles.BACKENDS},common_guard_vector_sha256=prepare.exact_json_hash(guards),qualification50_receipt_sha256=d['qualification_gate']['sha256'],source_separation_audit_sha256=evidence['source_audit']['sha256'],direction_trace_schema_sha256=evidence['direction_schema']['sha256'])
    for key,value in expected_bindings.items():
        if bindings.get(key)!=value:raise ValueError('Empreinte du sceau différente : '+key)
    source_audit=base.read(evidence['source_audit']['path'])
    if source_audit.get('passed')is not True or source_audit.get('source_tree_sha256')!=expected_bindings['source_tree_sha256']:raise ValueError('Séparation des sources non vérifiée')
    if gate.get('preparation_sha256',gate.get('preparation_manifest_sha256'))!=d['preparation']['sha256']:raise ValueError('Préparation de qualification différente')
    for key in ('study_contract','registration','contract','build_manifest'):
        if base.sha(d[key])!=d[key+'_sha256']:raise ValueError('Pièce modifiée : '+key)
    if profiles.evidence()!=d['profile_evidence']or d['profiles']!=profiles.command_templates():raise ValueError('Les profils qualifiés ont changé')
    if set(d['profiles'])!=set(CELLS):raise ValueError('Les huit cellules sont requises')
    if prepare.load_build(d['build_manifest'])!=d['build']:raise ValueError('La construction qualifiée a changé')
    if base.sha(d['hsl_library'])!=d['hsl_library_sha256']:raise ValueError('Bibliothèque HSL modifiée')
    for p,h in d['source_snapshot'].items():
        if base.sha(p)!=h:raise ValueError('Source ou reçu historique modifié : '+p)
    cases=d.get('cases',[])
    if len(cases)!=1834 or collections.Counter(c['regime']for c in cases)!={'I':1037,'M':797}:raise ValueError('Périmètre I/M incomplet')
    if prepare.exact_json_hash(cases)!=d['case_matrix_sha256']:raise ValueError('Matrice des cas modifiée')
    if d.get('budgets')!=dict(seconds=7200,max_outer=5000,shared_restoration=600,tolerance=1e-6):raise ValueError('Budgets modifiés')
    if d.get('cpus')!=list(CPUS)or d.get('jobs')!=JOBS:raise ValueError('L’ablation doit utiliser uniquement le cœur 23')
    d['_path']=str(path);d['_sha256']=base.sha(path);return d

def make_plan(cases,seed=SEED):
    order=list(CELLS);random.Random(seed).shuffle(order);jobs=[];rotations=[]
    for index,case in enumerate(sorted(cases,key=base.priority)):
        shift=index%len(order);treatments=order[shift:]+order[:shift]
        rotations.append(dict(case_id=case['case_id'],rotation=shift,treatments=treatments))
        for label in treatments:jobs.append((case,label,profiles.CELLS[label]['backend']))
    return jobs,dict(seed=seed,scope='treatment order only; raw starts unchanged',initial_order=order,rotations=rotations)

def check_native_proof(endpoint,folder,label):
    """Verify actual commits, original objective, no hidden frozen updates,
    initial-model digests and declared native backend, independently of KKT.
    """
    c=profiles.CELLS[label];folder=Path(folder);side_path=folder/'endpoint.json.factorial.json';trace=folder/'accepted_trace.jsonl'
    spec=importlib.util.spec_from_file_location('refinement_shared_trace_parser',KIT/'qualification/QUALIFIER_EXPERIENCE.py')
    qualified=importlib.util.module_from_spec(spec);spec.loader.exec_module(qualified)
    qualified.read_trace(trace,folder/'endpoint.json',True,label.rsplit('_',1)[0])
    side=base.read(side_path)
    if not isinstance(side,dict):raise ValueError('Preuve native factorielle absente')
    expected=dict(objective_exact=True,inequality_policy='hybrid',globalization_policy=c['globalization_policy'],refinement_policy=c['refinement_policy'],common_guards='r3',common_representation='original_qcqp')
    for key,value in expected.items():
        if side.get(key)!=value:raise ValueError('Preuve native différente : '+key)
    if side.get('trace_success')is not True or side.get('legacy_objective_validation')is not False:raise ValueError('Trace de recherche exacte non confirmée')
    if side.get('budget_verified')is not True:raise ValueError('Budget natif non vérifié')
    for key,bound in [('outer_iterations',5000),('restoration_iterations',600)]:
        v=side.get(key)
        if type(v)is not int or not 0<=v<=bound:raise ValueError('Compteur de budget invalide : '+key)
    seconds=side.get('seconds')
    if type(seconds)not in(int,float)or not math.isfinite(seconds)or not 0<=seconds<=7200:raise ValueError('Temps natif invalide')
    requested='ma27'if c['backend']=='ma27'else'mumps_metis'
    actual=side.get('actual_linear_backend');wanted='inequality-majorants/'+c['backend']
    if side.get('native_solver_completed')is not True or side.get('guard_policy_version')!='common_r3_guards_v1':raise ValueError('Fin native ou politique des garde-fous non prouvée')
    if side.get('requested_linear_backend')!=requested or not isinstance(actual,str)or endpoint.get('linear_solver_backend','')!=actual:raise ValueError('Backend natif déclaré incohérent')
    attempts=side.get('primary_backend_factor_attempts')
    if type(attempts)is not int or attempts<0 or side.get('primary_newton_backend_used')is not(attempts>0):raise ValueError('Travail de backend primaire non prouvé')
    pre_model=side.get('initial_model_events')==0
    if pre_model:
        if side.get('initial_model_state')!='not_reached'or side.get('initialization_calls')!=0 or side.get('native_pre_model_return')is not True or attempts!=0 or side.get('primary_newton_backend_used')is not False:raise ValueError('Retour natif avant modèle non prouvé')
        if side.get('accepted_events')!=0 or side.get('endpoint_accepted_steps')!=0 or endpoint.get('lazy_newton_accepted_steps')!=0:raise ValueError('Pas accepté avant une initialisation absente')
        if any(side.get(k)!=0 for k in('refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events')):raise ValueError('Raffinement avant modèle absent')
        if 'ipopt'in actual.lower()or'augmented'in actual.lower():raise ValueError('Route externe interdite avant modèle')
    elif actual!=wanted or side.get('initial_model_state')!='reached'or side.get('initialization_calls')!=side.get('initial_model_events'):
        raise ValueError('Backend natif effectif non prouvé : '+str(actual))
    guard=side.get('guard_config_sha256')
    if not hex64(guard):raise ValueError('Empreinte des garde-fous absente')
    counters=('refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events','initial_model_events')
    for key in counters:
        if type(side.get(key))is not int or side[key]<0:raise ValueError('Compteur natif absent ou invalide : '+key)
    if side['accepted_refinements']!=side['refinement_update_events']:raise ValueError('Compteurs de raffinement contradictoires')
    if c['refinement_policy']=='frozen':
        if any(side[key]!=0 for key in counters[:4])or side.get('frozen_metric_invariant_verified')is not True:raise ValueError('Métrique gelée modifiée ou invariance non prouvée')
    count=0;header=False;origins=collections.Counter();bases=collections.Counter();states=collections.Counter();metrics=[];sequence=hashlib.sha256()
    with trace.open()as stream:
        for line in stream:
            r=json.loads(line);event=r.get('event')
            if event in STATES:
                if not header or r.get('after_accepted_index')!=count or not hex64(r.get('state_sha256')):raise ValueError('Événement d’état incohérent')
                states[event]+=1;continue
            for key,value in expected.items():
                if r.get(key)!=value:raise ValueError('Contrat de trace différent : '+key)
            if r.get('guard_config_sha256')!=guard:raise ValueError('Garde-fous différents dans la trace')
            if event=='header':
                if header:raise ValueError('Entête de trace dupliqué')
                header=True;continue
            if not header:raise ValueError('Trace sans entête')
            if event=='metric_initialized':
                digest=r.get('initial_inequality_model_sha256');idx=r.get('phase_metric_index')
                if idx!=len(metrics)+1 or not hex64(digest)or r.get('initialization')!='common_certified_hybrid':raise ValueError('Initialisation commune non prouvée')
                metrics.append(digest);payload=digest.encode();sequence.update(struct.pack('<Q',len(payload)));sequence.update(payload);continue
            if event!='accepted':raise ValueError('Événement de trace inconnu : '+str(event))
            count+=1
            if r.get('accepted_index')!=count or r.get('objective_E0_zero')is not True:raise ValueError('Objectif exact ou indice accepté invalide')
            origin=r.get('final_origin');direction=r.get('direction_origin');post=r.get('postprocess')
            if not isinstance(origin,str)or not origin or not direction or not isinstance(post,str):raise ValueError('Origine finale absente')
            if origin!=(direction if post=='none'else post):raise ValueError('Origine finale incohérente')
            origins[origin]+=1;bases[direction]+=1
    if not header or count!=side.get('accepted_events')or count!=side.get('endpoint_accepted_steps')or count!=endpoint.get('lazy_newton_accepted_steps'):raise ValueError('Acceptations non réconciliées')
    if len(metrics)!=side['initial_model_events']:raise ValueError('Initialisations non réconciliées')
    if side.get('initial_inequality_model_sha256')!=(metrics[0]if metrics else''):raise ValueError('Première métrique initiale différente')
    if side.get('initial_hybrid_models_sha256')!=sequence.hexdigest():raise ValueError('Séquence des métriques initiales différente')
    return dict(expected,trace_success=True,budget_verified=True,guard_config_sha256=guard,initial_inequality_model_sha256=metrics[0]if metrics else'',initial_hybrid_models_sha256=sequence.hexdigest(),initial_model_events=len(metrics),refinement_attempt_events=side['refinement_attempt_events'],accepted_refinements=side['accepted_refinements'],refinement_update_events=side['refinement_update_events'],refinement_rollback_events=side['refinement_rollback_events'],frozen_metric_invariant_verified=side.get('frozen_metric_invariant_verified'),accepted_direction_origins=dict(origins),accepted_base_direction_origins=dict(bases),state_events=dict(states),accepted_events=count,sidecar_sha256=base.sha(side_path),trace_sha256=base.sha(trace),actual_linear_backend=actual,requested_linear_backend=requested,primary_newton_backend_used=side['primary_newton_backend_used'],primary_backend_factor_attempts=attempts,native_pre_model_return=pre_model,native_solver_completed=True,initial_model_state=side['initial_model_state'],initialization_calls=side['initialization_calls'],guard_policy_version=side['guard_policy_version'],intervention_verified=True)

class ExperimentCampaign(base.Campaign):
    def __init__(self,config,out,cases,**kwargs):
        kwargs.setdefault('cpus',CPUS)
        super().__init__(config,out,cases,**kwargs);self.planned,self.order_plan=make_plan(cases);self.methods=CELLS
        prior=base.read(self.out/'INTERVENTION_VIOLATION.json')
        self.protocol_failures=(list(prior.get('failures')or[dict(reason='persisted_intervention_violation')])if isinstance(prior,dict)else[dict(reason='invalid_persisted_intervention_violation')])if prior is not None else[]
        if prior is not None:self.stop.set()
    def prepare(self,resume=False):
        if self.protocol_failures:raise ValueError('Une violation d’intervention persistée interdit la reprise')
        super().prepare(resume)
        if not resume:return
        cases={c['case_id']:c for c in self.cases}
        for(cid,method),r in self.results.items():
            if r['classification']not in('strict','nonstrict'):continue
            f=self.folder(cases[cid],method);endpoint=base.read(f/'endpoint.json');audit=base.read(f/'AUDIT.json');resources=base.read(f/'solver_resources.json')
            if not isinstance(endpoint,dict)or not isinstance(audit,dict)or not isinstance(resources,dict):raise ValueError('Preuves de reprise absentes : '+cid+'/'+method)
            if audit!=r.get('original_audit')or resources!=r.get('solver_resources'):raise ValueError('Preuves de reprise modifiées : '+cid+'/'+method)
            if not process_budget_verified(resources)or r.get('budget_verified')is not True:raise ValueError('Budget de processus de reprise non vérifié')
            if audit.get('endpoint_sha256')!=base.sha(f/'endpoint.json')or audit.get('input_sha256')!=cases[cid]['input_sha256']:raise ValueError('Audit de reprise lié à d’autres données')
            proof=check_native_proof(endpoint,f,method);saved=r.get('provenance',{})
            for key in ('sidecar_sha256','trace_sha256','guard_config_sha256','initial_inequality_model_sha256'):
                if proof.get(key)!=saved.get(key):raise ValueError('Trace de reprise modifiée : '+key)
        grouped=collections.defaultdict(list)
        for r in self.results.values():grouped[r['case_id'],profiles.CELLS[r['method']]['backend']].append(r)
        violations=[]
        for records in grouped.values():
            guards={r['guard_config_sha256']for r in records if r.get('guard_config_sha256')}
            if len(guards)>1:violations.append(dict(case_id=records[0]['case_id'],reason='persisted_common_guard_mismatch'))
            for glob in('r3','hs'):
                initial={initial_model_identity(r)for r in records if r.get('globalization_policy')==glob and initial_model_identity(r)is not None}
                if len(initial)>1:violations.append(dict(case_id=records[0]['case_id'],reason='persisted_initial_metric_mismatch'))
        if violations:
            base.atomic(self.out/'INTERVENTION_VIOLATION.json',dict(passed=False,production_authorized=False,failures=violations));self.protocol_failures=violations;self.stop.set();raise ValueError('Intervention incohérente dans les résultats persistés')
    def binary_identity(self,method,backend):
        b=self.config['build']['backends'][backend];return b['experiment_binary'],b['experiment_binary_sha256']
    def runtime_environment(self,method,backend):return self.config['build']['backends'][backend].get('environment',{})
    def make_command(self,case,method,backend,endpoint,folder):
        return profiles.make_command(method,self.binary_identity(method,backend)[0],base.input_path(self.config,case),endpoint,folder/'accepted_trace.jsonl',self.config['hsl_library'])
    def check_endpoint(self,endpoint,method,backend,folder):
        proof=check_native_proof(endpoint,folder,method)
        if proof['guard_config_sha256']!=self.config['qualified_guard_config_sha256_by_backend'][backend]:raise ValueError('Garde-fous différents de la qualification scellée')
        return proof
    def observation_metadata(self,case,method):
        c=profiles.CELLS[method]
        return dict(run_id=self.run_id,cell=method,start_sha256=case['start_binding'].get('start_sha256'),preregistration_sha256=self.config['registration_sha256'],study_contract_sha256=self.config['study_contract_sha256'],objective_exact=True,refinement_policy=c['refinement_policy'],inequality_policy='hybrid',globalization_policy=c['globalization_policy'],common_guards='r3',common_representation='original_qcqp',native_backend=c['backend'])
    def decorate_result(self,result,case,method,resources,proof):
        super().decorate_result(result,case,method,resources,proof)
        result['budget_verified']=process_budget_verified(resources)and proof.get('budget_verified')is True
        if result['budget_verified']is not True:result.update(classification='indeterminate',strict_qcqp=None)
        result['experiment_proof_error']=result.pop('defaults_proof_error',None)
        result['native_binary_sha256']=result.pop('default_binary_sha256',None)
        result.update({k:proof[k]for k in ('guard_config_sha256','initial_inequality_model_sha256','initial_hybrid_models_sha256','initial_model_events','refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events','frozen_metric_invariant_verified','actual_linear_backend','requested_linear_backend','primary_newton_backend_used','primary_backend_factor_attempts','native_pre_model_return','native_solver_completed','initial_model_state','initialization_calls','guard_policy_version','intervention_verified')if k in proof})
        return result
    def finish(self,case,method,result):
        # Contradictory common guards or initial metric are protocol failures,
        # never silently averaged into a refinement effect. Preserve raw files.
        with self.lock:
            peers=[x for(c,m),x in self.results.items()if c==case['case_id']and profiles.CELLS[m]['backend']==profiles.CELLS[method]['backend']]
            guard=result.get('guard_config_sha256')
            for other in peers:
                if guard and other.get('guard_config_sha256')and guard!=other['guard_config_sha256']:
                    self.protocol_failures.append(dict(case_id=case['case_id'],methods=[method,other['method']],reason='common_guard_mismatch'))
                if result.get('globalization_policy')==other.get('globalization_policy'):
                    a=initial_model_identity(result);b=initial_model_identity(other)
                    if a is not None and b is not None and a!=b:self.protocol_failures.append(dict(case_id=case['case_id'],methods=[method,other['method']],reason='initial_metric_mismatch'))
            if self.protocol_failures:
                self.stop.set();base.atomic(self.out/'INTERVENTION_VIOLATION.json',dict(passed=False,production_authorized=False,failures=self.protocol_failures))
            super().finish(case,method,result)
    def snapshot(self):
        s=super().snapshot()
        with self.lock:values=list(self.results.values())
        s.update(schema='controlled-refinement-campaign-status-v1',by_regime_method={g:{m:dict(planned=sum(c['regime']==g for c in self.cases),done=sum(r['regime']==g and r['method']==m for r in values),strict=sum(r['regime']==g and r['method']==m and r['classification']=='strict'for r in values))for m in CELLS}for g in ('I','M')},intervention_failures=len(self.protocol_failures))
        if self.protocol_failures:s['state']='failed_intervention'
        return s

def intervention_summary(results,failures,expected_tasks):
    verified=sum(r.get('intervention_verified')is True and r.get('budget_verified')is True for r in results)
    return dict(passed=not failures and len(results)==expected_tasks and verified==expected_tasks,intervention_integrity_verified=not failures,expected_tasks=expected_tasks,finalized_observations=len(results),verified_native_interventions=verified,incomplete_native_interventions=expected_tasks-verified,failures=failures,finished=True)

def show(s):
    print(f"[{s['timestamp']}] {s['state']} | terminées {s['done']}/{s['total']} | KKT {s['strict']} | preuves incomplètes {s['indeterminate']} | créneaux {s['active_slots']}/1",flush=True)
    print('Cellules : KKT / terminées / prévues ; départs et budgets inchangés',flush=True)
    for b in profiles.BACKENDS:
        print(b.upper()+' : régime | F-GR3 | A-GR3 | F-GHS | A-GHS',flush=True)
        for g in ('I','M'):
            vals=[]
            for r,glob in [('FROZEN','GR3'),('REFINED','GR3'),('FROZEN','GHS'),('REFINED','GHS')]:
                q=s.get('by_regime_method',{}).get(g,{}).get(f'{r}_{glob}_{b}',{})
                vals.append(f"{q.get('strict',0)}/{q.get('done',0)}/{q.get('planned',0)}")
            print(g+' | '+' | '.join(vals),flush=True)
    active=sorted({x['dataset']for x in s.get('active',[])})
    print('Datasets actifs : '+(', '.join(active)if active else'aucun'),flush=True)
base.show=show

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('lancer','reprendre'):
        q=sub.add_parser(action);q.add_argument('--installation',required=True,type=Path);q.add_argument('--out',required=True,type=Path);q.add_argument('--interval',type=float,default=10)
    q=sub.add_parser('suivre');q.add_argument('--out',required=True,type=Path);q.add_argument('--interval',type=float,default=10);q.add_argument('--once',action='store_true')
    q=sub.add_parser('arret');q.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.action=='suivre':base.follow(a.out,a.interval,a.once);return
    if a.action=='arret':
        c=base.read(a.out/'CONTROLLER.json',{})
        if base.owned_live(c.get('pid'),c.get('pid_start_ticks')):os.kill(c['pid'],signal.SIGTERM);print('Arrêt demandé au contrôleur identifié ; résultats conservés.',flush=True)
        else:print('Aucun contrôleur possédé vivant.',flush=True)
        return
    d=load_installation(a.installation);out=a.out.resolve();resume=a.action=='reprendre'
    if resume and(out/'INTERVENTION_VIOLATION.json').exists():raise ValueError('Une violation d’intervention interdit la reprise')
    if not resume:prepare.fresh(out,[Path(d['root']),Path(d['source_run']),KIT])
    if not set(CPUS).issubset(os.sched_getaffinity(0)):raise ValueError('Cœur 23 indisponible')
    out.mkdir(parents=True,exist_ok=True);campaign=ExperimentCampaign(d,out,d['cases'])
    if len(campaign.planned)!=14672:raise ValueError('Matrice attendue : 1 834 cas × 8')
    if len({str(campaign.folder(c,m))for c,m,b in campaign.planned})!=14672:raise ValueError('Collision de noms de dossiers de tâches')
    settings=dict(schema='controlled-refinement-run-contract-v1',run_id=campaign.run_id,installation_sha256=d['_sha256'],study_contract_sha256=d['study_contract_sha256'],registration_sha256=d['registration_sha256'],build_manifest_sha256=d['build_manifest_sha256'],case_matrix_sha256=d['case_matrix_sha256'],methods=list(CELLS),cells=profiles.CELLS,seconds=7200,max_outer=5000,shared_restoration=600,tolerance=1e-6,cpus=list(CPUS),jobs=JOBS,threads=1,seed=SEED,order_plan_sha256=prepare.exact_json_hash(campaign.order_plan),cases=1834,tasks=14672)
    prior=base.read(out/'RUN_CONTRACT.json')
    if resume and not prior:raise ValueError('Reprise sans contrat contemporain interdite')
    if prior is not None and prior!=settings:raise ValueError('Contrat de reprise différent')
    with cpu_pool_lock(),(out/'controller.lock').open('a')as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Un contrôleur possède déjà cette sortie')
        base.atomic(out/'RUN_CONTRACT.json',settings);base.atomic(out/'ORDER_PLAN.json',campaign.order_plan)
        print('EXPÉRIENCE DÉMARRÉE — 1 834 cas × 8 ; cœur 23, un créneau continu, audit original indépendant.',flush=True)
        result=campaign.execute(resume,max(1,a.interval))
        if result['state']=='completed':base.atomic(out/'INTERVENTION_CHECK.json',dict(intervention_summary(list(campaign.results.values()),campaign.protocol_failures,14672),case_matrix_sha256=d['case_matrix_sha256'],study_contract_sha256=d['study_contract_sha256']))
if __name__=='__main__':
    try:main()
    except Exception as e:print('CAMPAGNE REFUSÉE — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
