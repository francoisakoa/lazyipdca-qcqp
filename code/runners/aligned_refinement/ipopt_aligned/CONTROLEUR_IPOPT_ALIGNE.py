#!/usr/bin/env python3
"""Continuous 22-slot comparison of aligned Ipopt monotone/adaptive policies."""
from pathlib import Path
import argparse,collections,fcntl,json,os,random,signal,sys
HERE=Path(__file__).resolve().parent;KIT=HERE.parent;BASE=KIT/'runtime_base';sys.path.insert(0,str(BASE))
import CAMPAGNE as base
import IPOPT_ALIGNE_CONTRACT as contract
import PREPARER_IPOPT_ALIGNE as preparation

def make_plan(cases,seed=contract.SEED):
    order=list(contract.CELLS);random.Random(seed).shuffle(order);jobs=[];rotations=[]
    for i,case in enumerate(sorted(cases,key=base.priority)):
        shift=i%len(order);treatments=order[shift:]+order[:shift]
        rotations.append(dict(case_id=case['case_id'],rotation=shift,treatments=treatments))
        jobs.extend((case,label,contract.CELLS[label]['backend'])for label in treatments)
    return jobs,dict(seed=seed,scope='treatment order only; unchanged raw starts',initial_order=order,rotations=rotations)

class IpoptCampaign(base.Campaign):
    def __init__(self,config,out,cases,**kwargs):
        kwargs.setdefault('cpus',contract.CPUS);super().__init__(config,out,cases,**kwargs)
        self.planned,self.order_plan=make_plan(cases);self.methods=tuple(contract.CELLS)
    def binary_identity(self,method,backend):return self.config['binary'],self.config['binary_sha256']
    def runtime_environment(self,method,backend):return self.config.get('runtime_environment',{})
    def make_command(self,case,method,backend,endpoint,folder):
        return contract.make_command(self.config,case,method,endpoint,base.input_path(self.config,case),base.external_start(self.config,case))
    def check_endpoint(self,endpoint,method,backend,folder):
        folder=Path(folder)
        cid=base.read(folder/'case.json',{}).get('case_id')
        if cid not in self.config.get('input_representation_expectations',{}):raise ValueError('Représentation QCQP du cas non liée')
        proof=contract.verify_endpoint(endpoint,method,self.config['ipopt_version'],self.config['hsl_library'],self.config['input_representation_expectations'][cid])
        with (folder/'solver.log').open('r',errors='replace')as stream:banner=stream.read(131072)
        if 'running with linear solver '+backend not in banner.lower():raise ValueError('Backend effectivement initialisé non prouvé dans le journal Ipopt')
        proof.update(endpoint_sha256=base.sha(folder/'endpoint.json'),solver_log_sha256=base.sha(folder/'solver.log'),backend_banner_verified=True)
        return proof
    def observation_metadata(self,case,method):
        c=contract.CELLS[method]
        return dict(run_id=self.run_id,cell=method,backend=c['backend'],mu_strategy=c['strategy'],ipopt_strategy=c['strategy'],strategy=c['strategy'],
            start_sha256=case['start_binding'].get('start_sha256'),options_policy='aligned_original_kkt_tolerances_v1',
            installation_sha256=self.config['_sha256'],binary_sha256=self.config['binary_sha256'])
    def indeterminate(self,case,method,reason,resources=None,audit=None):
        r=super().indeterminate(case,method,reason,resources,audit)
        r.update(aligned_options_verified=False,native_route_verified=False,backend_verified=False)
        return r
    def decorate_result(self,result,case,method,resources,proof):
        super().decorate_result(result,case,method,resources,proof)
        result['budget_verified']=contract.process_budget_verified(resources)and proof.get('budget_verified')is True
        if result['budget_verified']is not True:result.update(classification='indeterminate',strict_qcqp=None)
        result['aligned_proof_error']=result.pop('defaults_proof_error',None)
        result.pop('default_binary_sha256',None)
        result.update({k:proof[k]for k in ('aligned_options_verified','native_route_verified','backend_verified','ipopt_version','mu_strategy','linear_solver','effective_options_sha256','native_elapsed_seconds','native_iterations','option_file_disabled','input_representation_overrides','effective_bound_infinity')if k in proof})
        return result
    def finish(self,case,method,result):
        with self.lock:
            super().finish(case,method,result)
            with (self.out/'OBSERVATIONS.jsonl').open('a')as stream:
                stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
    def export_observations(self):
        # Rebuild from task receipts on resume to recover a crash between the
        # atomic task result and its append-only observation line.
        tmp=self.out/'OBSERVATIONS.jsonl.tmp'
        with self.lock,tmp.open('w')as stream:
            for key in sorted(self.results):stream.write(json.dumps(self.results[key],allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,self.out/'OBSERVATIONS.jsonl')
    def prepare(self,resume=False):
        super().prepare(resume)
        if resume:
            cases={c['case_id']:c for c in self.cases}
            for(cid,method),r in self.results.items():
                if r['classification']not in('strict','nonstrict'):continue
                folder=self.folder(cases[cid],method);ep=base.read(folder/'endpoint.json');audit=base.read(folder/'AUDIT.json');resources=base.read(folder/'solver_resources.json')
                if not isinstance(ep,dict)or not isinstance(audit,dict)or not isinstance(resources,dict):raise ValueError('Preuves de reprise absentes : '+cid+'/'+method)
                if audit!=r.get('original_audit')or resources!=r.get('solver_resources'):raise ValueError('Audit ou ressources modifiés : '+cid+'/'+method)
                if not contract.process_budget_verified(resources)or r.get('budget_verified')is not True:raise ValueError('Budget de reprise non vérifié')
                if audit.get('endpoint_sha256')!=base.sha(folder/'endpoint.json')or audit.get('input_sha256')!=cases[cid]['input_sha256']:raise ValueError('Audit lié à un autre endpoint ou QCQP')
                if audit.get('classification')!=r['classification']or audit.get('strict_qcqp')is not r.get('strict_qcqp'):raise ValueError('Qualification mathématique de reprise modifiée')
                proof=self.check_endpoint(ep,method,contract.CELLS[method]['backend'],folder)
                if proof!=r.get('provenance'):raise ValueError('Preuve effective de reprise modifiée')
        self.export_observations()
    def snapshot(self):
        s=super().snapshot()
        with self.lock:rows=list(self.results.values())
        def stats(g,m):
            r=[x for x in rows if x['regime']==g and x['method']==m];c=collections.Counter(x['classification']for x in r)
            return dict(planned=sum(x['regime']==g for x in self.cases),done=len(r),strict=c['strict'],nonstrict=c['nonstrict'],indeterminate=c['indeterminate'])
        s.update(schema='ipopt-aligned-campaign-status-v1',by_regime_method={g:{m:stats(g,m)for m in self.methods}for g in contract.REGIMES})
        return s

def show(s,detail=False):
    print(f"[{s['timestamp']}] {s['state']} | terminées {s['done']}/{s['total']} | KKT {s['strict']} | non certifiées {s['nonstrict']} | preuves incomplètes {s['indeterminate']} | créneaux {s['active_slots']}/22 (solveurs {s['active_solver']}, audits {s['active_audit']})",flush=True)
    print('Régime | monotone MA27 | monotone MUMPS | adaptive MA27 | adaptive MUMPS — KKT/terminées/prévues',flush=True)
    for g in contract.REGIMES:
        values=[]
        for m in contract.CELLS:
            r=s.get('by_regime_method',{}).get(g,{}).get(m,{})
            values.append(f"{r.get('strict',0)}/{r.get('done',0)}/{r.get('planned',0)}")
        print(g+' | '+' | '.join(values),flush=True)
    active=collections.Counter(x['dataset']for x in s.get('active',[]))
    print('Datasets actifs : '+(', '.join(f'{name} ({n})'for name,n in sorted(active.items()))if active else'aucun'),flush=True)
    if detail:
        for name,row in sorted(s.get('by_dataset',{}).items()):print(f"{name}: KKT {row['strict']}, terminées {row['done']}/{row['planned']}, preuves incomplètes {row['indeterminate']}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in('lancer','reprendre'):
        q=sub.add_parser(action);q.add_argument('--installation',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--interval',type=float,default=10)
    q=sub.add_parser('suivre');q.add_argument('--out',type=Path,required=True);q.add_argument('--interval',type=float,default=10);q.add_argument('--once',action='store_true');q.add_argument('--detail',action='store_true')
    q=sub.add_parser('arret');q.add_argument('--out',type=Path,required=True)
    a=p.parse_args();base.show=lambda s:show(s,getattr(a,'detail',False))
    if a.action=='suivre':base.follow(a.out,a.interval,a.once);return
    if a.action=='arret':
        saved=base.read(a.out/'CONTROLLER.json',{})
        if base.owned_live(saved.get('pid'),saved.get('pid_start_ticks')):os.kill(saved['pid'],signal.SIGTERM);print('Arrêt demandé au contrôleur Ipopt identifié.',flush=True)
        else:print('Aucun contrôleur Ipopt possédé vivant.',flush=True)
        return
    d=preparation.load_installation(a.installation);out=a.out.resolve();resume=a.action=='reprendre'
    if not resume:preparation.fresh(out,[Path(d['root']),Path(d['source_run']),KIT,*map(Path,base.read(d['frozen_resolution']['path']).get('frozen_roots',[]))])
    if not set(contract.CPUS).issubset(os.sched_getaffinity(0)):raise ValueError('Cœurs 0 à 21 indisponibles')
    out.mkdir(parents=True,exist_ok=True);campaign=IpoptCampaign(d,out,d['cases'])
    if len(campaign.planned)!=contract.TASKS or len({str(campaign.folder(c,m))for c,m,b in campaign.planned})!=contract.TASKS:raise ValueError('Plan incomplet ou dossiers de tâches confondus')
    settings=dict(schema='ipopt-aligned-run-contract-v1',run_id=campaign.run_id,installation_sha256=d['_sha256'],binary_sha256=d['binary_sha256'],case_matrix_sha256=d['case_matrix_sha256'],cells=contract.CELLS,seconds=7200,max_iter=5000,tolerance=1e-6,cpus=list(contract.CPUS),threads=1,seed=contract.SEED,order_plan_sha256=contract.json_sha(campaign.order_plan),cases=4078,tasks=contract.TASKS)
    prior=base.read(out/'RUN_CONTRACT.json')
    if resume and prior is None:raise ValueError('Reprise sans contrat contemporain interdite')
    if prior is not None and prior!=settings:raise ValueError('Contrat de reprise modifié')
    with base.machine_lock(),(out/'controller.lock').open('a')as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Cette sortie appartient déjà à un contrôleur')
        base.atomic(out/'RUN_CONTRACT.json',settings);base.atomic(out/'ORDER_PLAN.json',campaign.order_plan)
        base.atomic(out/'PLAN.json',dict(settings,methods=list(contract.CELLS),configurations=contract.CELLS,regimes=contract.REGIMES,case_bindings=d['cases'],installation_path=d['_path'],expected_options=contract.ALIGNED_OPTIONS))
        print('IPOPT ALIGNÉ — 4 078 cas × 4 = 16 312 tâches ; 22 créneaux continus sur 0–21.',flush=True)
        result=campaign.execute(resume,max(1,a.interval));campaign.export_observations()
        if result['state']=='completed':base.atomic(out/'COMPLETION.json',dict(completed=True,tasks=result['done'],expected_tasks=contract.TASKS,installation_sha256=d['_sha256'],strict=result['strict'],nonstrict=result['nonstrict'],technical_incomplete=result['indeterminate'],at_utc=base.utc()))
if __name__=='__main__':
    try:main()
    except Exception as error:print('CAMPAGNE IPOPT REFUSÉE — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
