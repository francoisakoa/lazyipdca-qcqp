#!/usr/bin/env python3
"""Fresh native mega campaign: continuous CPU slots, original QCQP audits, no Dey."""
from __future__ import annotations
import argparse,concurrent.futures,copy,fcntl,hashlib,json,math,os,signal,subprocess,sys,threading,time,uuid
from collections import deque
from pathlib import Path
from datetime import datetime,timezone

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import executor
from dataset_schedule import (REGIMES,SCHEMA,PRIORITIES,excluded_dey,prepare_datasets,
                              dataset_summaries,write_dataset_report,report_text,case_methods,task_pairs,RESEARCH_METHODS)
METHODS=executor.METHODS
FINAL_STATES={'completed','stopped','failed'}
COMPLETED_INPUT_SIGNATURES={}
read,write,sha,stamp,digest=executor.read,executor.write,executor.sha,executor.stamp,executor.digest


class RuntimeGuard:
    """Hash once, then rehash changed filesystem identities before and after tasks."""
    def __init__(self,hashes):
        self.hashes=dict(hashes);self.stats={};self.lock=threading.RLock();self.check(force=True)
    def check(self,force=False):
        with self.lock:
            for value,expected in self.hashes.items():
                p=Path(value);s=p.stat();signature=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                if force or self.stats.get(value)!=signature:
                    if sha(p)!=expected:raise ValueError('Frozen runtime changed: '+value)
                    self.stats[value]=signature


def ordered_cases(cases,require_full=True,methods=None):
    source=list(cases);removed=[c['id'] for c in source if excluded_dey(c)]
    kept=[c for c in source if not excluded_dey(c)]
    ordered,datasets=prepare_datasets(kept,METHODS if methods is None else methods)
    if any(excluded_dey(c) for c in ordered):raise ValueError('Dey remained in selected corpus')
    if require_full:
        counts={d['id']:d['N'] for d in datasets}
        expected={'CUTEst':136,'multicast_beamforming':129,'PGLIB':198,'beam_safety':3}
        if len(ordered)!=4078 or counts.get('pooling_dey',0)!=0 or any(counts.get(k)!=v for k,v in expected.items()):
            raise ValueError('Complete non-Dey frozen corpus required: 4078 cases, CUTEst136, multicast129, PGLIB198, beam_safety3')
        if [d['id'] for d in datasets[:3]]!=list(PRIORITIES):raise ValueError('Dataset priority changed')
        if sum(c.get('source_kind')=='extension' for c in ordered if c['dataset_id']=='multicast_beamforming')!=120:
            raise ValueError('The 120 multicast extensions are missing')
    return ordered,datasets,removed


def execution_binding(plan):
    return digest({**{key:plan[key]for key in ('installation','runtime_sha256','source_sha256','profiles',
        'seconds','max_outer','shared_restoration','tolerance','threads','al_mode','methods','datasets')},
        'run_id':plan.get('run_id')})


def task_digest(case,method,binding):
    return digest(dict(case=case,method=method,execution_binding_sha256=binding))


def prepare_inventory(root,frozen_roots=()):
    from prepare_inventory import import_inventory
    return import_inventory(root,frozen_roots)


def finalize_installation(root,native,comparator,qualification,imported):
    """Commit the exact runtime first, then bind PREPARATION to those bytes."""
    from prepare_inventory import bind_preparation
    root=Path(root).resolve()
    if qualification.get('pass') is not True and qualification.get('success') is not True:
        raise ValueError('Native protected qualification must pass before installation success')
    if native.get('API')!=4390 or native.get('source_unchanged') is not True:
        raise ValueError('Homogeneous native API4390 build required')
    installation=copy.deepcopy(comparator)
    library_roots=list(dict.fromkeys(str(Path(p).resolve()) for p in native.get('library_roots',[])+comparator.get('library_roots',[])+[str(Path(comparator['hsl_library']).parent)]))
    installation.update(success=True,API=4390,native_solver_api=4390,
        native_solver_release='mega-fresh-no-dey-api4390',native_al_fallback_policy='off',primary_standard_policy='auto',
        backend='ma27',native_backend='ma27',binary=native['binary'],binary_sha256=native['binary_sha256'],
        native_auditor=str(root/'campaign/common/original_auditor.py'),
        native_auditor_sha256=sha(root/'campaign/common/original_auditor.py'),
        python=sys.executable,ipopt_linear_solver='ma27',ipopt_mumps_linear_solver='mumps',ipopt_mumps_pivot_order=7,ipopt_mumps_ordering_policy='library-default',
        ipopt_methods={'ipopt_qcqp':'ma27','ipopt_mumps':'mumps'},library_roots=library_roots,
        environment={**THREAD_ENV_COPY(),**comparator.get('environment',{}),'LD_LIBRARY_PATH':os.pathsep.join(library_roots)},
        historical_numeric_reuse_allowed=False,excluded_datasets=['pooling_dey'],
        native_qualification=qualification,task_budgets=dict(seconds=7200,max_outer=5000,shared_restoration=600,tolerance=1e-6))
    installation['runtime_dependency_sha256']={**native.get('runtime_library_sha256',{}),
        **comparator.get('runtime_library_sha256',{}),**comparator.get('runtime_dependency_sha256',{})}
    for key in ('ipopt_binary','hsl_library','ma27_qualification','mumps_qualification'):
        if not installation.get(key):raise ValueError('Missing comparator runtime field: '+key)
        installation[key+'_sha256']=sha(installation[key])
    precise=root/'validation_tools/precise_original_auditor.py'
    if precise.is_file():
        installation.update(precise_auditor=str(precise),precise_auditor_sha256=sha(precise))
    installation['ipopt_command']=['{python}',str(root/'campaign/ipopt_worker.py'),'--case-json','{case_json}',
        '--input','{input}','--endpoint','{endpoint}','--seconds','{seconds}','--max-outer','{max_outer}']
    # Match INSTALLER.write exactly; its final rewrite of this returned object
    # keeps the same hash and creates no circular preparation/config binding.
    (root/'INSTALLATION.json').write_text(json.dumps(installation,indent=2,allow_nan=False)+'\n')
    bind_preparation(root,installation,imported)
    return installation


def THREAD_ENV_COPY():
    return dict(executor.THREAD_ENV,PYTHONDONTWRITEBYTECODE='1')


def verify_installation(root,installation):
    if installation.get('native_solver_api')!=4390 or installation.get('native_al_fallback_policy')!='off':
        raise ValueError('Native API4390 installation required')
    if installation.get('native_backend')!='ma27' or installation.get('backend')!='ma27':
        raise ValueError('All six historical Lazy profiles require the requested fresh MA27 backend')
    for key in ('binary','native_auditor','ipopt_binary','hsl_library','ma27_qualification','mumps_qualification'):
        path=Path(installation[key]).resolve(strict=True)
        if not installation.get(key+'_sha256') or sha(path)!=installation[key+'_sha256']:
            raise ValueError('Installed binary/auditor/qualification changed: '+key)
    native_path=Path(installation['native_backend_qualification']).resolve(strict=True)
    if sha(native_path)!=installation.get('native_backend_qualification_sha256'):
        raise ValueError('Fresh native MA27 qualification differs')
    native=read(native_path)
    expected={'schema':'native-ma27-six-profiles-qualification-v1','passed':True,'native_backend':'ma27',
              'binary_sha256':installation['binary_sha256'],'hsl_library_sha256':installation['hsl_library_sha256'],
              'six_profiles_qualified':True,'mumps_compiled':False}
    if any(native.get(k)!=v for k,v in expected.items()) or set(native.get('profiles',[]))!=set(METHODS[:6]):
        raise ValueError('Actual six-profile MA27 qualification is absent or does not bind this installation')
    observations=native.get('observations',[])
    if len(observations)!=24 or {(r.get('regime'),r.get('method'))for r in observations}!={(g,m)for g in('I','M','E','Q')for m in METHODS[:6]}:
        raise ValueError('All24 actual native MA27 controls are required')
    for row in observations:
        audit=row.get('original_audit')or{}
        residuals=[audit.get(k)for k in('feasibility','dual','complementarity')]
        if row.get('passed')is not True or row.get('ma27_primary_successful_factors',0)<1 or audit.get('strict')is not True or any(type(v)not in(int,float)or not math.isfinite(v)or not 0<=v<=1e-6 for v in residuals):
            raise ValueError('Actual native MA27 factor/original-KKT qualification is incomplete')
        task=native_path.parent/'tasks'/(row['regime']+'__'+row['method'])
        if sha(task/'endpoint.json')!=row.get('endpoint_sha256')or sha(task/'COMMAND.json')!=row.get('command_sha256'):
            raise ValueError('Actual native MA27 qualification endpoint/command differs')
    sys.path.insert(0,str(root))
    from setup_runtime import validate_ipopt_runtime
    validate_ipopt_runtime(installation)
    version=subprocess.check_output([installation['binary'],'--version'],env=executor.environment(installation),text=True,timeout=30)
    if 'API 4390'not in version:raise ValueError('Effective native binary API differs')
    if installation.get('factorial_installation'):
        from factorial_runtime import validate
        validate(installation)


def validate_plan(root,out,plan,verify_inputs=False):
    root=Path(root).resolve();out=Path(out).resolve()
    if plan.get('schema')!='mega-fresh-continuous-no-dey-v1' or plan.get('root')!=str(root) or plan.get('output_directory')!=str(out):
        raise ValueError('Plan/root/output identity mismatch')
    if read(out/'PLAN_BINDING.json').get('plan_sha256')!=sha(out/'PLAN.json'):raise ValueError('Frozen plan changed')
    declared_methods=METHODS+(RESEARCH_METHODS if plan['installation'].get('factorial_installation')else())
    if tuple(plan['methods'])!=declared_methods or any(excluded_dey(c) for c in plan['cases']) or any(excluded_dey(t['case']) for t in plan['tasks']):
        raise ValueError('Methods changed or forbidden Dey task found')
    if (plan['seconds'],plan['max_outer'],plan['shared_restoration'],plan['tolerance'],plan['threads'],plan['al_mode'])!=(7200,5000,600,1e-6,1,'off'):
        raise ValueError('Frozen global task budgets changed')
    if plan['jobs']!=22 or plan['cpus']!=list(range(22)):
        raise ValueError('This mega campaign requires exactly22CPUslots0through21')
    if plan.get('historical_numeric_reuse_allowed') is not False or plan.get('historical_results_reused')!=0:
        raise ValueError('Previous campaign results cannot be reused')
    ordered,datasets,_=ordered_cases(plan['cases'],methods=plan['methods'])
    if ordered!=plan['cases'] or datasets!=plan['datasets']:raise ValueError('Dataset order or membership changed')
    expected=[(c['id'],m) for c,m in task_pairs(ordered,plan['methods'])]
    actual=[(t['case']['id'],t['method']) for t in plan['tasks']]
    wanted=32624+(7336 if RESEARCH_METHODS[0]in declared_methods else 0)
    if actual!=expected or len(actual)!=wanted or len({t['id'] for t in plan['tasks']})!=wanted:raise ValueError('Complete contemporaneous mega method matrix changed')
    binding=execution_binding(plan)
    if binding!=plan['execution_binding_sha256']:raise ValueError('Execution contract changed')
    by_id={c['id']:c for c in ordered}
    for task in plan['tasks']:
        if task['id']!=task['case']['id']+'__'+task['method'] or task['case']!=by_id[task['case']['id']] or task['contract_sha256']!=task_digest(task['case'],task['method'],binding):
            raise ValueError('Task contract changed: '+task['id'])
    verify_installation(root,plan['installation'])
    guard=RuntimeGuard({**plan['runtime_sha256'],**plan['source_sha256']})
    if verify_inputs:
        for case in ordered:executor.verify_case(case)
    return guard


def guarded_run_task(out,plan,task,cpu,guard):
    if excluded_dey(task['case']):raise ValueError('Dey task refused at execution boundary')
    guard.check();executor.verify_case(task['case'])
    result=executor.run_task(out,plan,task,cpu)
    if result is None:return None
    try:
        guard.check();executor.verify_case(task['case'])
        from observation_contract import normalize_result
        result=normalize_result(result,integrity_verified=True)
        proof=out/'tasks'/task['id']/'PREPOST_INTEGRITY.json'
        write(proof,dict(before_pass=True,after_pass=True,input_sha256=task['case']['input_sha256'],
              execution_binding_sha256=plan['execution_binding_sha256'],created_utc=stamp()))
        result['integrity_receipt']=str(proof.relative_to(out))
        result['evidence_sha256'][str(proof.relative_to(out))]=sha(proof)
        write(executor.result_path(out,task),result)
        with executor.RESULT_LOCK:executor.RUN_RESULTS.setdefault(str(out),{})[task['id']]=result
        return result
    except BaseException:
        with executor.RESULT_LOCK:executor.RUN_RESULTS.get(str(out),{}).pop(task['id'],None)
        raise


def finalized_result(out,plan,task):
    result=executor.completed_result(out,task)
    if result is None:return None
    if not is_finalized_observation(result,task['id']):return None
    rel=str(Path('tasks')/task['id']/'PREPOST_INTEGRITY.json')
    if result.get('integrity_receipt')!=rel or rel not in result.get('evidence_sha256',{}):return None
    proof=read(out/rel)
    if proof.get('before_pass') is not True or proof.get('after_pass') is not True or proof.get('input_sha256')!=task['case']['input_sha256'] or proof.get('execution_binding_sha256')!=plan['execution_binding_sha256']:
        raise ValueError('Final integrity receipt differs: '+task['id'])
    for value,expected in executor.frozen_files(task['case']):
        p=executor.resolve_file(value);s=p.stat();signature=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,expected)
        if COMPLETED_INPUT_SIGNATURES.get(str(p))!=signature:
            if sha(p)!=expected:raise ValueError('Completed case input/start changed before resume: '+str(p))
            COMPLETED_INPUT_SIGNATURES[str(p)]=signature
    return result


def is_finalized_observation(result,task_id):
    """Only a normalized observation with its post-run proof is terminal."""
    receipt=str(Path('tasks')/task_id/'PREPOST_INTEGRITY.json')
    return (result.get('integrity_verified') is True
        and result.get('observation_classification') in (
            'strict_original_qcqp','audited_original_qcqp_nonstrict',
            'censored_indeterminate','budget_exhausted_indeterminate',
            'terminal_time_limit_indeterminate','indeterminate_observation')
        and result.get('integrity_receipt')==receipt
        and receipt in result.get('evidence_sha256',{}))


def finalized_records(out):
    with executor.RESULT_LOCK:
        return [result for task_id,result in executor.RUN_RESULTS.get(str(out),{}).items()
                if is_finalized_observation(result,task_id)]


def summary(out,plan,state,occupied_slots=0,current_dataset=None):
    results=finalized_records(out)
    with executor.ACTIVE_LOCK:active=copy.deepcopy(executor.ACTIVE)
    counts={m:dict(planned=sum(m in case_methods(c,plan['methods'])for c in plan['cases']),done=0,strict=0,nonstrict=0,indeterminate=0) for m in plan['methods']}
    by_regime={r:dict(N=sum(c['regime']==r for c in plan['cases']),methods={m:dict(planned=sum(c['regime']==r and m in case_methods(c,plan['methods'])for c in plan['cases']),done=0,strict=0,nonstrict=0,indeterminate=0) for m in plan['methods']}) for r in REGIMES}
    cases={c['id']:c for c in plan['cases']}
    for result in results:
        method=result['method'];c=cases[result['case_id']]
        group=by_regime[c['regime']]['methods'][method]
        for owner in (counts[method],group):
            owner['done']+=1
            owner['strict' if result.get('strict_qcqp') is True else 'nonstrict' if result.get('strict_qcqp') is False else 'indeterminate']+=1
    by_dataset=dataset_summaries(plan,results)
    running_datasets=sorted({cases[k.rsplit('__',1)[0]]['dataset_id'] for k in active if k.rsplit('__',1)[0] in cases},key=lambda k:next(d['position'] for d in plan['datasets'] if d['id']==k))
    value=dict(schema='mega-fresh-continuous-no-dey-status-v1',timestamp=stamp(),state=state,
        done=len(results),total=len(plan['tasks']),strict_qcqp=sum(x['strict'] for x in counts.values()),
        nonstrict=sum(x['nonstrict'] for x in counts.values()),indeterminate=sum(x['indeterminate'] for x in counts.values()),
        by_method=counts,by_regime=by_regime,by_dataset=by_dataset,datasets_total=len(plan['datasets']),
        datasets_completed=sum(d['state']=='completed' for d in by_dataset.values()),active=active,
        active_slots=occupied_slots,active_solver=sum(x.get('phase')=='solver' for x in active.values()),
        active_audit=sum(x.get('phase')=='audit' for x in active.values()),running_datasets=running_datasets,
        cpus=plan['cpus'],jobs=plan['jobs'],cpu_batches=plan.get('cpu_batches',[]),
        current_dataset=current_dataset,excluded_datasets=['pooling_dey'],historical_results_reused=0,
        scheduling='continuous-no-dataset-barrier',native_backend='ma27',controller_pid=os.getpid(),controller_start_ticks=executor.identity(os.getpid()),AL='off')
    write(out/'STATUS.json',value);return value


def continuous_loop(out,plan,guard,*,worker=None,restore=None,report=None,status=None):
    """One global queue; completed dataset reports never block dispatch."""
    worker=worker or guarded_run_task;restore=restore or finalized_result;status=status or summary
    if report is None:
        def report(dataset_id,records):
            item,directory,_=write_dataset_report(out,plan,dataset_id,records,write,stamp())
            print(report_text(item,plan['methods']),end='',flush=True)
            return directory
    with executor.RESULT_LOCK:executor.RUN_RESULTS[str(out)]={}
    waiting=deque()
    done_by={d['id']:0 for d in plan['datasets']}
    for task in plan['tasks']:
        if excluded_dey(task['case']):raise ValueError('Dey task in continuous queue')
        record=restore(out,plan,task)
        if record is None:waiting.append(task)
        else:
            executor.RUN_RESULTS[str(out)][task['id']]=record
            done_by[task['case']['dataset_id']]+=1
    declared={d['id']:d['total_cells'] for d in plan['datasets']}
    emitted=set();report_futures={};pending={};next_status=0.;current=None
    def stopping():
        p=out/'STOP.json'
        if p.exists() and read(p).get('token')==plan.get('_controller_token'):executor.STOP.set()
        return executor.STOP.is_set()
    state='running'
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan['jobs']) as pool, concurrent.futures.ThreadPoolExecutor(max_workers=1) as reporting:
        while waiting or pending:
            stopping()
            # Refill every idle CPU before report publication or polling.
            if not executor.STOP.is_set():
                used={item[0] for item in pending.values()}
                for cpu in plan['cpus'][:plan['jobs']]:
                    if waiting and cpu not in used:
                        task=waiting.popleft();current=task['case']['dataset_id']
                        pending[pool.submit(worker,out,plan,task,cpu,guard)]=(cpu,task)
            if pending:
                settled,_=concurrent.futures.wait(pending,timeout=.1,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in settled:
                    _,task=pending.pop(future)
                    try:result=future.result()
                    except BaseException:executor.STOP.set();raise
                    if result is not None:done_by[task['case']['dataset_id']]+=1
            # Reports run independently; they do not create dataset barriers.
            records=finalized_records(out)
            for dataset_id,number in done_by.items():
                if number==declared[dataset_id] and dataset_id not in emitted:
                    emitted.add(dataset_id);report_futures[reporting.submit(report,dataset_id,records)]=dataset_id
            for future in list(report_futures):
                if future.done():
                    try:future.result()
                    except BaseException:executor.STOP.set();raise
                    report_futures.pop(future)
            if time.monotonic()>=next_status:
                status(out,plan,'stopping' if executor.STOP.is_set() else 'running',len(pending),current);next_status=time.monotonic()+10
            if executor.STOP.is_set() and not pending:state='stopped';break
        for future in report_futures:future.result()
    if executor.STOP.is_set():state='stopped'
    elif not waiting and not pending:state='completed'
    return status(out,plan,state,0,current)


def run_controller(root,out,token):
    with executor.cpu_slots_lock(),(out/'RUNNING.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(out/'CONTROLLER_OWNER.json',dict(pid=os.getpid(),token=token,acquired=stamp()))
        signal.signal(signal.SIGTERM,lambda *_:executor.STOP.set())
        signal.signal(signal.SIGINT,lambda *_:executor.STOP.set())
        print('PRÉPARATION DU CONTRÔLEUR — lecture du plan : '+str(out),flush=True)
        plan=read(out/'PLAN.json')
        from progress import read_status
        initial=read_status(out) or dict(done=0,total=len(plan.get('tasks',[])))
        initial.pop('error',None)
        initial.update(state='initializing',timestamp=stamp(),controller_pid=os.getpid(),controller_start_ticks=executor.identity(os.getpid()),active_slots=0,active_solver=0,active_audit=0,
            jobs=plan['jobs'],phase='Contrôle de l’installation et des données avant lancement')
        write(out/'STATUS.json',initial)
        try:
            print('CONTRÔLE DE L’INSTALLATION ET DES DONNÉES',flush=True)
            guard=validate_plan(root,out,plan,verify_inputs=False)
            print('CONTRÔLES VALIDÉS — alimentation continue des 22 créneaux',flush=True)
            plan['_controller_token']=token
            result=continuous_loop(out,plan,guard)
            print(result['state'],result['done'],'/',result['total'],flush=True)
            return 0
        except BaseException as exc:
            executor.STOP.set()
            write(out/'STATUS.json',dict(state='failed',error=repr(exc),timestamp=stamp(),total=len(plan.get('tasks',[]))))
            raise


def selected_out(root,value):
    return value.resolve() if value else Path((root/'LAST_RUN.txt').read_text().strip()).resolve()


def start_controller(root,out):
    from stop_campaign import assert_stopped,stopped_run_lock
    previous=root/'LAST_RUN.txt'
    if previous.exists() and Path(previous.read_text().strip()).resolve()!=out:
        assert_stopped(root,Path(previous.read_text().strip()))
    with stopped_run_lock(root,out):
        (out/'STOP.json').unlink(missing_ok=True);token=uuid.uuid4().hex
        command=[sys.executable,str(HERE/'MEGA.py'),'--root',str(root),'_run','--out',str(out),'--token',token]
        with (out/'controller.log').open('ab') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(root))
        write(out/'CONTROLLER.json',dict(pid=process.pid,token=token,script=str(HERE/'MEGA.py'),out=str(out),created=stamp()))
        end=time.monotonic()+15
        while time.monotonic()<end:
            owner=out/'CONTROLLER_OWNER.json'
            if owner.exists() and read(owner).get('token')==token:break
            if process.poll() is not None:
                tail='\n'.join((out/'controller.log').read_text(errors='replace').splitlines()[-30:])
                raise ValueError('Controller failed (code '+str(process.returncode)+'); journal '+str(out/'controller.log')+'\n'+tail)
            time.sleep(.05)
        else:executor.kill_child(process);raise ValueError('Controller did not acknowledge start')
    previous.write_text(str(out)+'\n')
    print('MEGA DÉMARRÉE — '+str(read(out/'PLAN.json')['total_cells'])+' tâches ; six Lazy MA27 et deux Ipopt par défaut ; 22 créneaux continus',flush=True)
    print('Ordre CUTEst → multicast Beamforming → PGLIB → autres ; sortie '+str(out),flush=True)


def make_plan(root,args):
    from stop_campaign import assert_stopped
    print('PRÉPARATION DU PLAN — inventaire des instances et des départs gelés',flush=True)
    prepared=read(root/'PREPARATION.json')
    methods=METHODS+(RESEARCH_METHODS if prepared['installation'].get('factorial_installation')else())
    ordered,datasets,_=ordered_cases(prepared['cases'],methods=methods)
    cpus=executor.parse_cpus(args.cpus)
    if args.jobs!=22 or cpus!=list(range(22)) or not set(cpus)<=set(os.sched_getaffinity(0)):raise ValueError('Exactly22 available CPUslots0through21 required')
    if not Path('/usr/bin/taskset').is_file():raise ValueError('Linux taskset required')
    previous=root/'LAST_RUN.txt'
    if previous.exists():assert_stopped(root,Path(previous.read_text().strip()))
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=args.out.resolve() if args.out else Path('/extra/LAZYQCQP_MEGA_SANS_DEY_RESULTS')/('CUTEST_BEAM_PGLIB_'+suffix)
    if out.exists() or out==root or root in out.parents:raise ValueError('New result directory outside installation required')
    out.mkdir(parents=True)
    plan=copy.deepcopy(prepared)
    plan.update(schema='mega-fresh-continuous-no-dey-v1',created=stamp(),root=str(root),output_directory=str(out),
        cases=ordered,datasets=datasets,methods=list(methods),run_id='MEGA_MA27_'+uuid.uuid4().hex,seconds=7200,max_outer=5000,shared_restoration=600,
        tolerance=1e-6,threads=1,al_mode='off',cpus=cpus,jobs=args.jobs,cpu_batches=[cpus[:11],cpus[11:]],
        historical_numeric_reuse_allowed=False,historical_results_reused=0,excluded_datasets=['pooling_dey'],
        dataset_schedule_schema=SCHEMA,task_ordering=dict(rule='continuous-one-global-queue',dataset_order=[d['id'] for d in datasets],
            boundary='No dataset barrier: next dataset starts as soon as a CPU slot becomes idle',within_dataset='I/M/E/Q then frozen identifier',within_case=list(methods),treatment_order=('balanced cyclic rotations of all12 onI/M and all8 onE/Q; seed20261108; rawstartsunchanged'if RESEARCH_METHODS[0]in methods else'published eight-method order'),order_seed=20261108 if RESEARCH_METHODS[0]in methods else None))
    plan['runtime_sha256'][str(root/'PREPARATION.json')]=sha(root/'PREPARATION.json')
    plan['execution_binding_sha256']=execution_binding(plan)
    plan['tasks']=[dict(id=c['id']+'__'+m,case=c,method=m,contract_sha256=task_digest(c,m,plan['execution_binding_sha256'])) for c,m in task_pairs(ordered,methods)]
    plan['total_cells']=len(plan['tasks'])
    write(out/'PLAN.json',plan);write(out/'PLAN_BINDING.json',dict(plan_sha256=sha(out/'PLAN.json'),created_utc=stamp()))
    write(out/'DATASETS.json',dict(schema=SCHEMA,datasets=datasets,excluded_datasets=['pooling_dey']))
    print('PLAN PRÊT — contrôle de l’installation et des données avant démarrage',flush=True)
    validate_plan(root,out,plan,verify_inputs=False);start_controller(root,out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=HERE.parent)
    sub=p.add_subparsers(dest='action',required=True)
    launch=sub.add_parser('lancer');launch.add_argument('--cpus',default='0-21');launch.add_argument('--jobs',type=int,default=22);launch.add_argument('--out',type=Path)
    for action in ('suivre','arreter','reprendre','verifier'):
        q=sub.add_parser(action);q.add_argument('--out',type=Path)
        if action=='suivre':
            q.add_argument('--once',action='store_true');q.add_argument('--interval',type=float,default=10)
            q.add_argument('--json',action='store_true',help='Sortie JSON pour traitement automatique')
            q.add_argument('--detail',action='store_true',help='Ancien affichage complet global et par dataset')
        if action=='arreter':q.add_argument('--timeout',type=float,default=90)
    q=sub.add_parser('_run');q.add_argument('--out',type=Path,required=True);q.add_argument('--token',required=True)
    a=p.parse_args();root=a.root.resolve(strict=True)
    if root!=HERE.parent:raise ValueError('This controller must run from its own installed ROOT/campaign directory')
    if a.action=='lancer':
        print('LANCEMENT DEMANDÉ — 22 cœurs (0 à 21), CUTEst → Beamforming → PGLIB → autres, sans Dey',flush=True)
        with (root/'OPERATION.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);make_plan(root,a)
    elif a.action=='_run':return run_controller(root,a.out.resolve(),a.token)
    else:
        out=selected_out(root,a.out)
        if a.action=='arreter':
            from stop_campaign import stop_reference
            print(json.dumps(stop_reference(root,out,timeout=a.timeout),ensure_ascii=False))
        elif a.action in ('reprendre','verifier'):
            print(('REPRISE' if a.action=='reprendre' else 'VÉRIFICATION')+' — contrôle du plan, de l’installation et des données : '+str(out),flush=True)
            with (root/'OPERATION.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);validate_plan(root,out,read(out/'PLAN.json'),verify_inputs=a.action=='verifier')
                if a.action=='reprendre':start_controller(root,out)
                else:print('PLAN INTÈGRE — 4 078 instances, '+str(len(read(out/'PLAN.json')['tasks']))+' tâches, Dey exclu',flush=True)
        else:
            from progress import follow_status
            follow_status(out,once=a.once,interval=a.interval,json_output=a.json,detail=a.detail)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
        print('MEGA_REFUSEE : '+str(exc),file=sys.stderr,flush=True);raise SystemExit(2)
