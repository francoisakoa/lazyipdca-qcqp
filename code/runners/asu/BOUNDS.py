#!/usr/bin/env python3
"""Prepare and run a separate 503 x 3 x 2 x 2 implied-bound experiment."""
from pathlib import Path
import argparse,collections,fcntl,hashlib,json,math,os,random,signal,sys,uuid
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'runtime_base'))
import CAMPAGNE as base
import FROZEN_INPUTS
import PROFILES_BOUNDS as profiles
class ProtocolViolation(ValueError):pass
def show(s):
    jobs=s.get('jobs',22)
    print(f"[{s['timestamp']}] {s['state']} | {s['done']}/{s['total']} | KKT {s['strict']} | non strict {s['nonstrict']} | indéterminé {s['indeterminate']} | créneaux {s['active_slots']}/{jobs}",flush=True)
    print('Cellules : strict/non strict/indéterminé ; terminées/prévues',flush=True)
    for b in profiles.BACKENDS:
        print(b+' : '+' | '.join(m.replace('_'+b,'')+': '+f"{x['strict']}/{x['nonstrict']}/{x['indeterminate']} ; {x['done']}/{x['planned']}"for m,x in s['by_method'].items()if m.endswith('_'+b)),flush=True)
base.show=show
def read(p):return json.loads(Path(p).read_text())
def matrix_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_artifact(path,wanted):
    if base.sha(path)!=wanted:raise ValueError('Pièce absente ou modifiée : '+str(path))
def cases():
    data=read(HERE/'CONTRAT_ASU_503.json')['cases']
    if len(data)!=503 or len({c['case_id']for c in data})!=503 or any(c['regime']!='M'for c in data):raise ValueError('Le périmètre ASU doit rester à 503 cas M')
    return data
def selected_cases(n=None):
    data=cases()
    if n is None:return data
    if n not in(12,24):raise ValueError('Pilote de 12 ou 24 cas requis')
    # The two smallest dimensions are retained, then a deterministic hash order.
    return sorted(data,key=lambda c:(c['dimensions']['n'],hashlib.sha256(c['case_id'].encode()).hexdigest()))[:n]
def build_manifest(path):
    d=read(path)
    if d.get('schema')!='lazy-bound-removal-build-v2'or d.get('same_binary_for_both_backends')is not True:raise ValueError('Construction de l’ablation inconnue')
    for p,h in [(d['binary'],d['binary_sha256']),(d['reference_binary'],d['reference_binary_sha256']),(d['reference_build'],d['reference_build_sha256']),(d['hsl_library'],d['hsl_library_sha256']),(d['patch_receipt'],d['patch_receipt_sha256']),(d['dependencies'],d['dependencies_sha256'])]:verify_artifact(p,h)
    for folder,key in [('source_root','source_sha256'),('reference_source_root','base_source_sha256')]:
        for relative,wanted in d[key].items():verify_artifact(Path(d[folder])/relative,wanted)
    dependencies=read(d['dependencies'])
    for p,h in {**dependencies['all_loaded_library_sha256'],**dependencies['hsl_runtime']['dependency_sha256']}.items():verify_artifact(p,h)
    if d.get('homogeneous_compilation')is not True:raise ValueError('Compilation homogène requise')
    return d
def prepare(a):
    root=a.root.expanduser().resolve(strict=True);out=a.out.expanduser().resolve()
    if out.exists():raise ValueError('Répertoire de préparation neuf requis')
    if out==root or root in out.parents or out in root.parents or out==HERE or HERE in out.parents:raise ValueError('Préparation séparée requise')
    b=build_manifest(a.build);selected=selected_cases(a.pilot_cases);out.mkdir(parents=True)
    bindings,res=FROZEN_INPUTS.resolve_all(root,selected,out/'FROZEN_RESOLUTION.json',extra_roots=a.frozen_root,progress=lambda phase,x:print('ENTRÉES '+phase+' '+json.dumps(x),flush=True))
    config=dict(schema='lazy-bound-removal-installation-v1',root=str(root),cases=selected,frozen_bindings=bindings,
      build_manifest=str(a.build.resolve()),build_manifest_sha256=base.sha(a.build),build=b,
      python=sys.executable,hsl_library=b['hsl_library'],hsl_library_sha256=b['hsl_library_sha256'],
      contract_sha256=base.sha(HERE/'CONTRAT_ASU_503.json'),profiles=profiles.templates(),
      profile_factory_sha256=base.sha(HERE/'PROFILES_BOUNDS.py'),controller_sha256=base.sha(Path(__file__)),
      auditor_sha256=base.sha(HERE/'runtime_base/AUDIT_ORIGINAL.py'),reader_sha256=base.sha(HERE/'runtime_base/frozen_qcqp_reader.py'),
      source_snapshot={str(p.resolve()):base.sha(p)for p in [HERE/'BOUNDS.py',HERE/'PROFILES_BOUNDS.py',HERE/'REFERENCE_PROFILES.json',HERE/'NATIVE_PROOF.py',HERE/'REUSE_BOUNDS.py',HERE/'QUALIFIER_BOUNDS.py',
        HERE/'runtime_base/CAMPAGNE.py',HERE/'runtime_base/FROZEN_INPUTS.py',HERE/'runtime_base/AUDIT_ORIGINAL.py',HERE/'runtime_base/frozen_qcqp_reader.py']},
      case_matrix_sha256=matrix_hash(selected),budgets=dict(seconds=7200,outer=5000,restoration_shared=600,tolerance=1e-6),
      native_profiles_preserved=True,intervention='quadratic-equality-implied-bound rule only',tasks=len(selected)*12,pilot_cases=a.pilot_cases,
      optimizer_invocations=0,prior_results_reused=False,inputs_modified=False)
    base.atomic(out/'INSTALLATION.json',config);print('PRÉPARATION TERMINÉE : '+str(out/'INSTALLATION.json'))
def installation(path):
    d=read(path)
    if d.get('schema')!='lazy-bound-removal-installation-v1':raise ValueError('Installation inconnue')
    verify_artifact(d['build_manifest'],d['build_manifest_sha256'])
    if build_manifest(d['build_manifest'])!=d['build']:raise ValueError('Construction modifiée')
    for p,h in d['source_snapshot'].items():verify_artifact(p,h)
    for p,h in [(HERE/'CONTRAT_ASU_503.json',d['contract_sha256']),(HERE/'PROFILES_BOUNDS.py',d['profile_factory_sha256']),
      (Path(__file__),d['controller_sha256']),(HERE/'runtime_base/AUDIT_ORIGINAL.py',d['auditor_sha256']),(HERE/'runtime_base/frozen_qcqp_reader.py',d['reader_sha256'])]:verify_artifact(p,h)
    if d['profiles']!=profiles.templates()or d['cases']!=selected_cases(d.get('pilot_cases'))or matrix_hash(d['cases'])!=d['case_matrix_sha256']:raise ValueError('Profils ou cas modifiés')
    d['_path']=str(path.resolve());d['_sha256']=base.sha(path);return d
def locks(cpus):
    name='0_21'if cpus==tuple(range(22))else'23'
    p=Path('/tmp/LAZYQCQP_CPU_'+name+'.lock')
    try:fd=os.open(p,os.O_RDWR|os.O_CREAT,0o666)
    except PermissionError:fd=os.open(p,os.O_RDONLY)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:os.close(fd);raise ValueError('Les cœurs demandés sont réservés par un autre contrôleur ; attendre sa fin')
    return os.fdopen(fd,'r')
class BoundCampaign(base.Campaign):
    def __init__(self,config,out,cpus):
        super().__init__(config,out,config['cases'],cpus=cpus)
        order=list(profiles.CELLS);random.Random(20261110).shuffle(order)
        self.planned=[]
        for i,c in enumerate(sorted(config['cases'],key=lambda x:x['case_id'])):
            self.planned.extend((c,label,profiles.CELLS[label]['backend'])for label in order[i%12:]+order[:i%12])
        self.methods=tuple(profiles.CELLS)
    def indeterminate(self,case,method,reason,resources=None,audit=None):
        result=super().indeterminate(case,method,reason,resources,audit)
        result.update(installation_sha256=self.config['_sha256'],binary_sha256=self.binary_identity(method,profiles.CELLS[method]['backend'])[1])
        return result
    def prepare(self,resume=False):
        super().prepare(resume)
        if resume:
            from REUSE_BOUNDS import validate_existing
            for (cid,method),result in self.results.items():
                case=next(c for c in self.cases if c['case_id']==cid)
                validate_existing(self,case,method,result)
    def binary_identity(self,method,backend):return self.config['build']['binary'],self.config['build']['binary_sha256']
    def runtime_environment(self,method,backend):return self.config['build']['runtime_environment']
    def make_command(self,case,method,backend,endpoint,folder):
        args=profiles.command(method,self.binary_identity(method,backend)[0],base.input_path(self.config,case),endpoint,self.config['hsl_library'])
        if base.external_start(self.config,case) is not None:raise ValueError('Ce périmètre exige le départ x0 intégré brut')
        return args
    def observation_metadata(self,case,method):
        return dict(run_id=self.run_id,start_sha256=case['start_binding'].get('start_sha256'),**profiles.CELLS[method],research_variant=True,intervention='quadratic-equality-implied-bound rule only')
    def check_endpoint(self,ep,method,backend,folder):
        try:return self._check_endpoint(ep,method,backend,folder)
        except ProtocolViolation as e:
            self.stop.set();base.atomic(self.out/'INTERVENTION_VIOLATION.json',dict(method=method,folder=str(folder),reason=str(e),passed=False))
            raise ValueError('Intervention ou backend non vérifié : '+str(e))from e
        except(ValueError,KeyError,FileNotFoundError,TypeError)as e:
            # Missing execution evidence invalidates this task, not its peers.
            return dict(intervention_verified=False,budget_verified=False,native_proof_error=str(e))
    def _check_endpoint(self,ep,method,backend,folder):
        c=profiles.CELLS[method];rp=folder/'endpoint.json.rank_presolve.json';r=read(rp)
        for key,value in [('research_equality_implied_bounds',c['bounds']),('quadratic_equality_rule_enabled',c['bounds']=='remove'),('radial_inequality_master_enabled',True),('affine_equality_rule_enabled',c['policy']=='R3')]:
            if key not in r:raise ValueError('Preuve native absente : '+key)
            if r[key]!=value:raise ProtocolViolation('Intervention native contradictoire : '+key)
        from NATIVE_PROOF import validate_side_report,validate_case_report
        validate_case_report(r,read(folder/'case.json'))
        side_proof=validate_side_report(r,keep=(c['bounds']=='keep'))
        budget_reasons=[]
        for k,bound in [('outer_iterations',5000),('restoration_iterations',600)]:
            value=ep.get(k,ep.get('iterations')if k=='outer_iterations'else None)
            if type(value)is not int or not 0<=value<=bound:budget_reasons.append('Budget natif non vérifié : '+k)
        elapsed=ep.get('elapsed_seconds')
        if type(elapsed)not in(int,float)or not math.isfinite(elapsed)or not 0<=elapsed<=7200:budget_reasons.append('Temps natif non vérifié')
        al=ep.get('native_al_fallback',{})
        if al.get('enabled')is True or al.get('attempted')is True or al.get('accepted')is True:raise ProtocolViolation('Route native modifiée')
        actual=ep.get('linear_solver_backend','')
        if not actual:raise ValueError('Preuve de backend absente')
        if actual!='inequality-majorants/'+backend:raise ProtocolViolation('Backend natif inattendu : '+str(actual))
        if ep.get('lazy_newton_pivoted_fallbacks')is None:raise ValueError('Compteur de fallback primaire absent')
        if ep['lazy_newton_pivoted_fallbacks']!=0:raise ProtocolViolation('Fallback primaire non autorisé pour cette comparaison de backend')
        return dict(intervention_verified=True,budget_verified=not budget_reasons,budget_reasons=budget_reasons,rank_presolve_report_sha256=base.sha(rp),
          quadratic_equality_rule_enabled=r['quadratic_equality_rule_enabled'],
          equality_implied_bound_sides=r['equality_implied_bound_sides'],implied_bound_sides=r['implied_bound_sides'],actual_linear_backend=actual,removed_sides=side_proof)
    def snapshot(self):
        s=super().snapshot();s.update(jobs=len(self.cpus),cpus=list(self.cpus));return s
    def decorate_result(self,result,case,method,resources,proof):
        super().decorate_result(result,case,method,resources,proof)
        seconds=resources.get('elapsed_seconds')
        result['budget_verified']=result.get('budget_verified')is True and type(seconds)in(int,float)and math.isfinite(seconds)and 0<=seconds<=7200
        if result['budget_verified']is not True or proof.get('intervention_verified')is not True:
            result.update(classification='indeterminate',strict_qcqp=None)
        result['experiment_proof_error']=result.pop('defaults_proof_error',None)
        result['native_binary_sha256']=result.pop('default_binary_sha256',None)
        result.update({k:proof[k]for k in ['removed_sides','intervention_verified','rank_presolve_report_sha256','quadratic_equality_rule_enabled','equality_implied_bound_sides','implied_bound_sides','actual_linear_backend']if k in proof})
def run_contract(d,out,cpus,old=None):
    return dict(schema='native-bound-removal-run-v1',run_id=old['run_id']if old else out.name+'_'+uuid.uuid4().hex,
      installation_sha256=d['_sha256'],installation=d['_path'],cpus=list(cpus),jobs=len(cpus),tasks=d['tasks'],cases=len(d['cases']),
      case_ids=[c['case_id']for c in d['cases']],cells=profiles.CELLS,budgets=d['budgets'],case_matrix_sha256=d['case_matrix_sha256'],qualification=d['_qualification'])
def run(a):
    d=installation(a.installation);cpus=(23,)if a.cpus=='23'else tuple(range(22))
    from QUALIFIER_BOUNDS import admission
    d['_qualification']=admission(a.qualification,d)
    if not set(cpus).issubset(os.sched_getaffinity(0)):raise ValueError('Cœurs demandés indisponibles')
    out=a.out.expanduser().resolve();resume=a.action=='reprendre'
    if not resume and out.exists():raise ValueError('Répertoire de résultats neuf requis')
    out.mkdir(parents=True,exist_ok=True)
    with locks(cpus):
        old=base.read(out/'RUN_CONTRACT.json')
        if resume and not old:raise ValueError('Reprise sans contrat')
        contract=run_contract(d,out,cpus,old)
        if old and old!=contract:raise ValueError('Contrat de reprise modifié')
        base.atomic(out/'RUN_CONTRACT.json',contract)
        for i,c in enumerate(d['cases'],1):
            base.verify_input(d,c)
            if i%100==0 or i==len(d['cases']):print(f"GELÉ VÉRIFIÉ {i}/{len(d['cases'])}",flush=True)
        BoundCampaign(d,out,cpus).execute(resume,10.)
def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='action',required=True)
    p=sp.add_parser('plan');p.add_argument('--out',type=Path,required=True);p.add_argument('--pilot-cases',type=int,choices=[12,24])
    p=sp.add_parser('preparer');p.add_argument('--root',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--frozen-root',type=Path,action='append',default=[]);p.add_argument('--pilot-cases',type=int,choices=[12,24])
    for name in ['lancer','reprendre']:
        p=sp.add_parser(name);p.add_argument('--installation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--cpus',choices=['0-21','23'],default='0-21')
    p=sp.add_parser('suivre');p.add_argument('--out',type=Path,required=True);p.add_argument('--once',action='store_true')
    a=ap.parse_args()
    if a.action=='plan':
        if a.out.exists():raise ValueError('Fichier de plan neuf requis')
        selected=selected_cases(a.pilot_cases);base.atomic(a.out,dict(schema='bound-removal-plan-v1',cases=selected,profiles=profiles.templates(),tasks=len(selected)*12,optimizer_invocations=0));print('PLAN ÉCRIT : '+str(a.out));return
    if a.action=='preparer':prepare(a);return
    if a.action=='suivre':base.follow(a.out,10,a.once);return
    run(a)
if __name__=='__main__':
    try:main()
    except Exception as e:print('ABLATION REFUSÉE — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
