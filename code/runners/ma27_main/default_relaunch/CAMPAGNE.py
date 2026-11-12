#!/usr/bin/env python3
"""Full frozen-corpus Ipopt-default replay, with continuous fixed-CPU slots."""
from __future__ import annotations
import argparse,collections,concurrent.futures,csv,datetime as dt,fcntl,gzip,hashlib,json,os
from pathlib import Path
import queue,re,signal,subprocess,sys,threading,time,uuid

HERE=Path(__file__).resolve().parent
METHODS=(('ipopt_ma27_default','ma27'),('ipopt_mumps_default','mumps'))
CPUS=tuple(range(22)); SECONDS=7200.0; MAX_ITER=5000
THREAD_ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}
THREAD_ENV.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE')
EXPECTED_DEFAULTS=dict(mu_strategy='monotone',tol=1e-8,constr_viol_tol=1e-4,dual_inf_tol=1.0,
 compl_inf_tol=1e-4,acceptable_iter=15,acceptable_tol=1e-6,acceptable_constr_viol_tol=1e-2,
 acceptable_dual_inf_tol=1e10,acceptable_compl_inf_tol=1e-2,acceptable_obj_change_tol=1e20,
 s_max=100.0,nlp_scaling_method='gradient-based',hessian_approximation='exact',
 fixed_variable_treatment='make_parameter',bound_relax_factor=1e-8)

def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with tmp.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def internal_work(endpoint):
    """Preserve numerical work scalars without copying primal/dual vectors."""
    markers=('iteration','factorization','factorisation','linear_solve','linear_solves','newton','restoration','internal_work','seconds','time_','solve_count','backtrack')
    return {k:v for k,v in endpoint.items() if any(m in k.lower() for m in markers)
            and (v is None or type(v) in (str,int,bool) or type(v) is float and __import__('math').isfinite(v))}
def contract(path):
    p=Path(path);data=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    return json.loads(data)
def start_ticks(pid):
    try:return Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None
def owned_live(pid,ticks):return ticks is not None and start_ticks(pid)==str(ticks)
def machine_lock():
    """One machine-wide lock shared by the standalone and combined launchers."""
    path=Path('/tmp/LAZYQCQP_CPU_0_21.lock')
    try:fd=os.open(path,os.O_RDWR|os.O_CREAT,0o666)
    except PermissionError:fd=os.open(path,os.O_RDONLY)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise ValueError('Les 22 cœurs sont déjà réservés par un contrôleur de ce kit')
    return os.fdopen(fd,'r')
def safe_id(s):
    value=re.sub(r'[^A-Za-z0-9_.-]','_',s)
    return value if value==s and len(s)<180 else value[:150]+'_'+hashlib.sha256(s.encode()).hexdigest()[:16]
def priority(case):
    dataset=case['dataset']; order=0 if dataset=='CUTEst' else 1 if dataset=='multicast_beamforming' else 2 if dataset=='PGLIB' else 3
    return order,dataset,case['case_id']
def plan_cases(data,require_full=True):
    cases=data.get('cases',[])
    if require_full and len(cases)!=4078:raise ValueError(f'Corpus complet requis : {len(cases)}/4078')
    if len({x['case_id'] for x in cases})!=len(cases):raise ValueError('Identifiants dupliqués')
    if any(x['regime'] not in ('I','M','E','Q') for x in cases):raise ValueError('Régime inattendu')
    if any('dey' in x['dataset'].lower() for x in cases):raise ValueError('Dey exclu du contrat')
    if require_full and len({x['dataset'] for x in cases})!=28:raise ValueError('Les 28 datasets sont requis')
    return sorted(cases,key=priority)
def installation(path):
    path=Path(path).resolve();d=read(path)
    if not isinstance(d,dict):raise ValueError('INSTALLATION.json absent')
    for key in ('default_binary','default_binary_sha256','root','contract','hsl_library','hsl_library_sha256'):
        if not d.get(key):raise ValueError('Installation sans '+key)
    for key,hkey in (('default_binary','default_binary_sha256'),('hsl_library','hsl_library_sha256'),('ipopt_library','ipopt_library_sha256')):
        if d.get(key) and (not d.get(hkey) or sha(d[key])!=d[hkey]):raise ValueError('Identité modifiée : '+key)
    d['_path']=str(path);d['_sha256']=sha(path)
    return d
def input_path(config,case):
    bindings=config.get('frozen_bindings',{}).get(case['case_id'],{})
    return Path(bindings.get('input') or Path(config['root'])/'data/frozen'/case['case_id']/'input.qcq').resolve()
def external_start(config,case):
    binding=case['start_binding']
    if not binding.get('external_start_required'):return None
    paths=config.get('frozen_bindings',{}).get(case['case_id'],{})
    value=paths.get('start') or binding.get('start_file') or case.get('start_file')
    if not value:raise ValueError('Départ externe non résolu : '+case['case_id'])
    p=Path(value)
    if not p.is_absolute():p=Path(config['root'])/p
    return p.resolve()
def verify_input(config,case):
    p=input_path(config,case)
    if sha(p)!=case['input_sha256']:raise ValueError('QCQP gelé modifié : '+case['case_id'])
    binding=case['start_binding'];ext=external_start(config,case)
    if ext:
        if sha(ext)!=binding.get('start_file_sha256'):raise ValueError('Départ externe modifié : '+case['case_id'])
    elif binding.get('raw_x0_line_sha256'):
        line=None
        with p.open('rb') as f:
            for raw in f:
                if raw.startswith(b'x0 '):line=raw;break
        if line is None or hashlib.sha256(line).hexdigest()!=binding['raw_x0_line_sha256']:
            raise ValueError('Ligne x0 brute modifiée : '+case['case_id'])
    return p,ext
def command(config,case,backend,endpoint):
    cmd=[config['default_binary'],'--problem',str(input_path(config,case)),'--json',str(endpoint),
         '--time-limit',str(int(SECONDS)),'--max-iter',str(MAX_ITER),'--linear-solver',backend]
    if backend=='ma27':cmd+=['--hsl-library',config['hsl_library']]
    start=external_start(config,case)
    if start:cmd+=['--frozen-start',str(start)]
    return cmd

def verify_effective_defaults(endpoint,backend):
    if endpoint.get('adapter_version')!='ipopt-library-defaults-original-qcqp-v1':
        raise ValueError('Identité de l’adaptateur par défaut absente')
    options=endpoint.get('effective_options',{})
    for key,value in EXPECTED_DEFAULTS.items():
        if options.get(key)!=value:raise ValueError('Paramètre non défaut : '+key)
    for key,value in [('linear_solver',backend),('max_iter',MAX_ITER),('max_cpu_time',SECONDS),('max_wall_time',SECONDS)]:
        if options.get(key)!=value:raise ValueError('Contrat ressource/backend différent : '+key)
    if backend=='mumps' and options.get('mumps_pivot_order')!=7:
        raise ValueError('Ordre MUMPS différent du défaut de cette version')
    return True

def run_process(cmd,cwd,log,timeout,stop,cpu=None,on_start=None,env_extra=None):
    """One owned process group; wait4 preserves per-child CPU/RSS even with threads."""
    environment=dict(os.environ);environment.update(env_extra or {});environment.update(THREAD_ENV)
    for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT'):environment.pop(key,None)
    actual=(['taskset','-c',str(cpu)]+list(cmd)) if cpu is not None else list(cmd)
    started=time.monotonic();started_utc=utc();reason=None;term_at=None;rusage=None;rc=None
    with Path(log).open('wb') as output:
        proc=subprocess.Popen(actual,cwd=cwd,env=environment,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=start_ticks(proc.pid)
        if on_start:on_start(proc.pid,ticks)
        while True:
            waited,status,usage=os.wait4(proc.pid,os.WNOHANG)
            if waited:
                rc=os.waitstatus_to_exitcode(status);rusage=usage;proc.returncode=rc;break
            now=time.monotonic()
            if reason is None and (stop.is_set() or now-started>=timeout):
                reason='interrupted' if stop.is_set() else 'wall_time_limit';term_at=now
                if owned_live(proc.pid,ticks):
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            if term_at is not None and now-term_at>=1.0 and owned_live(proc.pid,ticks):
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                term_at=now+86400
            time.sleep(.05)
    elapsed=time.monotonic()-started
    if reason is None and elapsed>timeout:reason='wall_time_limit'
    return dict(schema='ipopt-default-process-resources-v1',started_utc=started_utc,finished_utc=utc(),
      command=actual,command_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest(),pid=proc.pid,pid_start_ticks=ticks,
      returncode=rc,elapsed_seconds=elapsed,user_cpu_seconds=rusage.ru_utime,system_cpu_seconds=rusage.ru_stime,
      peak_rss_kib=rusage.ru_maxrss,minor_faults=rusage.ru_minflt,major_faults=rusage.ru_majflt,
      voluntary_context_switches=rusage.ru_nvcsw,involuntary_context_switches=rusage.ru_nivcsw,
      timed_out=reason=='wall_time_limit',interrupted=reason=='interrupted',censored=reason is not None,
      termination_reason=reason,cpu=cpu,threads=1)

class Campaign:
    def __init__(self,config,out,cases,audit_timeout=600.0,auditor=None,cpus=CPUS):
        self.config=config;self.out=Path(out).resolve();self.cases=cases;self.cpus=tuple(cpus)
        self.audit_timeout=audit_timeout;self.auditor=Path(auditor or HERE/'AUDIT_ORIGINAL.py').resolve()
        self.stop=threading.Event();self.lock=threading.RLock();self.active={};self.results={};self.jobs=queue.Queue()
        self.planned=[(case,m,b) for case in cases for m,b in METHODS]
        self.started_utc=utc();self.state='running';self.errors=[]
        self.methods=tuple(m for m,_ in METHODS)
        self.run_id=read(self.out/'RUN_CONTRACT.json',{}).get('run_id') or self.out.name+'_'+uuid.uuid4().hex
    def folder(self,case,method):return self.out/'tasks'/safe_id(case['case_id']+'__'+method)
    def set_active(self,cpu,value):
        with self.lock:
            if value is None:self.active.pop(cpu,None)
            else:self.active[cpu]=value
    def phase(self,cpu,case,method,phase):
        self.set_active(cpu,dict(case_id=case['case_id'],dataset=case['dataset'],regime=case['regime'],method=method,
                                 phase=phase,started_utc=utc(),cpu=cpu,pid=None,pid_start_ticks=None))
        def started(pid,ticks):
            with self.lock:
                self.active[cpu].update(pid=pid,pid_start_ticks=ticks)
                atomic(self.folder(case,method)/'ACTIVE.json',self.active[cpu])
        return started
    def indeterminate(self,case,method,reason,resources=None,audit=None):
        result=dict(case_id=case['case_id'],dataset=case['dataset'],regime=case['regime'],family=case.get('family'),
         method=method,classification='indeterminate',strict_qcqp=None,status=reason,primal=None,dual=None,
         complementarity=None,objective=None,input_sha256=case['input_sha256'],start_binding=case['start_binding'],
         solver_resources=resources,original_audit=audit,finalized_utc=utc())
        result.update(self.observation_metadata(case,method))
        result.update(censored=bool(resources.get('censored') or resources.get('returncode',0)<0) if resources else None,budget_verified=None,trace_success=None)
        return result
    def observation_metadata(self,case,method):
        return dict(run_id=self.run_id,start_sha256=case['start_binding'].get('start_sha256'),
                    cell=None,preregistration_sha256=None,objective_exact=None)
    def decorate_result(self,result,case,method,resources,proof):
        result.update(self.observation_metadata(case,method))
        interrupted=bool(resources.get('censored') or resources.get('returncode',-1)<0)
        result.update(censored=interrupted,
            budget_verified=not resources.get('censored') and resources.get('returncode',-1)>=0 and 0<=resources.get('elapsed_seconds',float('inf'))<=SECONDS,
            trace_success=None,accepted_direction_origins=None)
        for key in ('objective_exact','budget_verified','trace_success','accepted_direction_origins','accepted_base_direction_origins'):
            if key in proof:result[key]=proof[key]
        if interrupted:result['budget_verified']=False
        if result['budget_verified'] is not True:
            result['classification']='indeterminate';result['strict_qcqp']=None
        return result
    def binary_identity(self,method,backend):return self.config['default_binary'],self.config['default_binary_sha256']
    def runtime_environment(self,method,backend):return self.config.get('runtime_environment')
    def make_command(self,case,method,backend,endpoint,folder):return command(self.config,case,backend,endpoint)
    def check_endpoint(self,endpoint,method,backend,folder):
        verify_effective_defaults(endpoint,backend)
        return dict(defaults_verified=True)
    def finish(self,case,method,result):
        f=self.folder(case,method)
        with self.lock:
            atomic(f/'RESULT.json',result);self.results[case['case_id'],method]=result
            with (self.out/'RESULTS.jsonl').open('a') as stream:stream.write(json.dumps(result,allow_nan=False)+'\n');stream.flush()
            try:(f/'ACTIVE.json').unlink()
            except FileNotFoundError:pass
    def task(self,cpu,case,method,backend,recover=False):
        f=self.folder(case,method);f.mkdir(parents=True,exist_ok=True);resources=None
        try:
            verify_input(self.config,case)
            if recover:
                stored_case=read(f/'case.json',{})
                if stored_case.get('run_id')!=self.run_id or stored_case.get('installation_sha256')!=self.config['_sha256']:
                    raise ValueError('Reçu de résolution historique ou lié à une autre installation')
                resources=read(f/'solver_resources.json')
                if not isinstance(resources,dict) or resources.get('returncode') is None:
                    self.finish(case,method,self.indeterminate(case,method,'controller_interrupted_resources_unavailable'));return
            else:
                # No optimization is repeated for an existing endpoint/receipt.
                if any((f/name).exists() for name in ('endpoint.json','solver_resources.json')):
                    raise ValueError('Sortie préexistante sans reprise explicite')
                casefile=dict(case,method=method,backend=backend,budget_seconds=SECONDS,max_iterations=MAX_ITER,
                    installation_sha256=self.config['_sha256'],created_utc=utc(),run_id=self.run_id)
                casefile.update(self.observation_metadata(case,method))
                atomic(f/'case.json',casefile)
                binary,binary_sha=self.binary_identity(method,backend)
                if sha(binary)!=binary_sha:
                    raise ValueError('Binaire par défaut modifié avant résolution')
                cmd=self.make_command(case,method,backend,f/'endpoint.json',f);atomic(f/'COMMAND.json',cmd)
                resources=run_process(cmd,f,f/'solver.log',SECONDS,self.stop,cpu,self.phase(cpu,case,method,'solver'),self.runtime_environment(method,backend))
                atomic(f/'solver_resources.json',resources)
            if not (f/'endpoint.json').is_file():
                self.finish(case,method,self.indeterminate(case,method,'endpoint_unavailable',resources));return
            auditcmd=[self.config.get('python',sys.executable),str(self.auditor),'--input',str(input_path(self.config,case)),
                '--endpoint',str(f/'endpoint.json'),'--out',str(f/'AUDIT.json'),'--resources',str(f/'solver_resources.json'),
                '--case-json',str(f/'case.json')]
            # An interrupted supervisor does not start a new audit; resume can recover it.
            if self.stop.is_set():
                self.finish(case,method,self.indeterminate(case,method,'supervisor_interrupted',resources));return
            auditres=run_process(auditcmd,f,f/'audit.log',self.audit_timeout,self.stop,cpu,self.phase(cpu,case,method,'audit'),self.runtime_environment(method,backend))
            atomic(f/'audit_resources.json',auditres);audit=read(f/'AUDIT.json')
            if auditres.get('censored') or not isinstance(audit,dict):
                self.finish(case,method,self.indeterminate(case,method,'original_audit_unavailable',resources,audit));return
            classification=audit.get('classification','indeterminate');strict=audit.get('strict_qcqp')
            if classification not in ('strict','nonstrict','indeterminate'):classification='indeterminate';strict=None
            if resources.get('censored') or resources.get('returncode',0)<0:classification='indeterminate';strict=None
            ep=read(f/'endpoint.json',{})
            proof_error=None;proof={}
            try:proof=self.check_endpoint(ep,method,backend,f)
            except ValueError as error:classification='indeterminate';strict=None;proof_error=str(error)
            binary,binary_sha=self.binary_identity(method,backend)
            result=dict(case_id=case['case_id'],dataset=case['dataset'],regime=case['regime'],family=case.get('family'),
                method=method,backend=backend,classification=classification,strict_qcqp=strict,
                status=ep.get('status'),ipopt_application_status=ep.get('ipopt_application_status'),
                ipopt_application_status_code=ep.get('ipopt_application_status_code'),
                ipopt_own_success=ep.get('ipopt_application_status_code')==0 if ep.get('ipopt_application_status_code') is not None else None,
                defaults_proof_error=proof_error,effective_options=ep.get('effective_options'),
                input_sha256=case['input_sha256'],start_binding=case['start_binding'],
                primal=audit.get('primal'),dual=audit.get('dual'),complementarity=audit.get('complementarity'),objective=audit.get('objective'),
                seconds_process=resources.get('elapsed_seconds'),seconds_process_scope='solver process elapsed; independent audit excluded',
                audit_seconds=auditres['elapsed_seconds'],iterations=ep.get('iterations',ep.get('outer_iterations')),internal_work=internal_work(ep),
                original_audit=audit,solver_resources=resources,audit_resources=auditres,
                binary_sha256=binary_sha,default_binary_sha256=binary_sha if backend in ('ma27','mumps') else None,
                installation_sha256=self.config['_sha256'],provenance=proof,
                recovered_without_solver=bool(recover),finalized_utc=utc())
            self.decorate_result(result,case,method,resources,proof)
            self.finish(case,method,result)
        except BaseException as error:
            # Every queued unit receives a final record, even on worker exceptions.
            self.finish(case,method,self.indeterminate(case,method,type(error).__name__+': '+str(error),resources))
        finally:self.set_active(cpu,None)
    def worker(self,cpu):
        while not self.stop.is_set():
            try:case,method,backend,recover=self.jobs.get_nowait()
            except queue.Empty:return
            try:self.task(cpu,case,method,backend,recover)
            finally:self.jobs.task_done()
    def prepare(self,resume=False):
        for case,method,backend in self.planned:
            f=self.folder(case,method);prior=read(f/'RESULT.json')
            if prior is not None:
                if not resume:raise ValueError('Résultat préexistant : utiliser reprendre')
                if prior.get('input_sha256')!=case['input_sha256']:raise ValueError('Résultat lié à une autre entrée')
                for key,expected in self.observation_metadata(case,method).items():
                    if key in ('objective_exact','requested_objective_exact'):continue
                    if prior.get(key)!=expected:raise ValueError('Résultat historique ou mal apparié : '+key)
                if prior.get('installation_sha256')!=self.config['_sha256'] or prior.get('binary_sha256')!=self.binary_identity(method,backend)[1]:
                    raise ValueError('Résultat lié à une autre installation ou un autre binaire')
                self.results[case['case_id'],method]=prior;continue
            active=read(f/'ACTIVE.json',{})
            if active and owned_live(active.get('pid'),active.get('pid_start_ticks')):raise ValueError('Processus encore actif : '+str(f))
            started=(f/'case.json').exists()
            self.jobs.put((case,method,backend,bool(started)))
    def snapshot(self):
        with self.lock:
            values=list(self.results.values());active=list(self.active.values());state=self.state
        def count(rows):
            c=collections.Counter(x['classification'] for x in rows)
            return dict(done=len(rows),strict=c['strict'],nonstrict=c['nonstrict'],indeterminate=c['indeterminate'],
                        ipopt_own_success=sum(x.get('ipopt_own_success') is True for x in rows))
        datasets=collections.Counter(c['dataset'] for c in self.cases);regimes=collections.Counter(c['regime'] for c in self.cases)
        return dict(schema='ipopt-default-campaign-status-v1',timestamp=utc(),state=state,started_utc=self.started_utc,
          controller_pid=os.getpid(),controller_start_ticks=start_ticks(os.getpid()),total=len(self.planned),**count(values),
          active_slots=len(active),active_solver=sum(x['phase']=='solver' for x in active),active_audit=sum(x['phase']=='audit' for x in active),
          active=active,queued=self.jobs.qsize(),
          by_method={m:dict(planned=sum(method==m for _,method,_ in self.planned),**count([x for x in values if x['method']==m])) for m in self.methods},
          by_regime={g:dict(planned=sum(case['regime']==g for case,_,_ in self.planned),**count([x for x in values if x['regime']==g])) for g in ('I','M','E','Q') if regimes[g]},
          by_dataset={g:dict(planned=sum(case['dataset']==g for case,_,_ in self.planned),**count([x for x in values if x['dataset']==g])) for g,n in datasets.items()})
    def execute(self,resume=False,interval=10.0):
        self.prepare(resume);self.started_utc=utc();atomic(self.out/'CONTROLLER.json',dict(pid=os.getpid(),pid_start_ticks=start_ticks(os.getpid()),started_utc=self.started_utc))
        previous={s:signal.signal(s,lambda *_:self.stop.set()) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.cpus)) as pool:
                futures=[pool.submit(self.worker,cpu) for cpu in self.cpus]
                try:
                    while True:
                        failures=[f.exception() for f in futures if f.done() and f.exception() is not None]
                        if failures:self.stop.set();self.errors=[str(e) for e in failures]
                        snapshot=self.snapshot();atomic(self.out/'STATUS.json',snapshot);show(snapshot)
                        if all(f.done() for f in futures):break
                        self.stop.wait(interval) if not self.stop.is_set() else time.sleep(.2)
                    for future in futures:future.result()
                except BaseException:
                    self.stop.set();raise
            self.state='stopped' if self.stop.is_set() else 'completed'
            snapshot=self.snapshot();atomic(self.out/'STATUS.json',snapshot);show(snapshot)
            return snapshot
        except BaseException:
            self.state='failed'
            try:atomic(self.out/'STATUS.json',self.snapshot())
            except OSError:pass
            raise
        finally:
            for s,handler in previous.items():signal.signal(s,handler)

def show(s):
    print(f"[{s['timestamp']}] {s['state']} : {s['done']}/{s['total']} | KKT {s['strict']} | non strict {s['nonstrict']} | indéterminé {s['indeterminate']} | créneaux {s['active_slots']}/22",flush=True)
    print('RÉGIMES  strict/non strict/indéterminé ; terminées/prévues',flush=True)
    print(' | '.join(f"{g} {x['strict']}/{x['nonstrict']}/{x['indeterminate']} ; {x['done']}/{x['planned']}" for g,x in s['by_regime'].items()),flush=True)
    print('MÉTHODES  '+' | '.join(f"{m}: {x['strict']}/{x['nonstrict']}/{x['indeterminate']} ; {x['done']}/{x['planned']}" for m,x in s['by_method'].items()),flush=True)
    active_datasets={a['dataset'] for a in s['active']}
    selected=[(g,x) for g,x in s['by_dataset'].items() if g in active_datasets]
    closed=sum(x['done']==x['planned'] for x in s['by_dataset'].values())
    print('DATASETS ACTIFS  '+' | '.join(f"{g}: {x['done']}/{x['planned']} KKT={x['strict']}" for g,x in selected[:5])+
          (f" | +{len(selected)-5} autres" if len(selected)>5 else '')+f" | terminés {closed}/{len(s['by_dataset'])}",flush=True)

def observed_status(status):
    """Read-only monitor view; stale saved activity is never called live work."""
    s=dict(status)
    if s.get('state')!='running':return s
    alive=owned_live(s.get('controller_pid'),s.get('controller_start_ticks'))
    if not alive:
        active=[x for x in s.get('active',[]) if owned_live(x.get('pid'),x.get('pid_start_ticks'))]
        s.update(state='controller_dead_children_present' if active else 'controller_dead_stale_state',active=active,
                 active_slots=len(active),active_solver=sum(x.get('phase')=='solver' for x in active),
                 active_audit=sum(x.get('phase')=='audit' for x in active),controller_alive=False)
    else:s['controller_alive']=True
    return s

def follow(out,interval=10.0,once=False):
    while True:
        saved=read(Path(out)/'STATUS.json')
        s=observed_status(saved) if saved else None
        if s:
            show(s)
            if s.get('controller_alive') is False:
                print('CONTRÔLEUR ARRÊTÉ : état sauvegardé périmé ; reprendre après vérification des processus identifiés.',flush=True)
            elif s.get('state')=='running':
                age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(s['timestamp'])).total_seconds()
                if age>max(60,3*interval):print(f'ÉTAT ANCIEN : dernière mise à jour il y a {age:.0f} s ; contrôleur présent.',flush=True)
        else:print('En attente du premier état : '+str(out),flush=True)
        if once or s and s['state']!='running':return s
        time.sleep(max(1,interval))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    for name in ('lancer','reprendre'):
        p=sub.add_parser(name);p.add_argument('--installation',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--interval',type=float,default=10)
    p=sub.add_parser('suivre');p.add_argument('--out',type=Path,required=True);p.add_argument('--interval',type=float,default=10);p.add_argument('--once',action='store_true')
    p=sub.add_parser('arret');p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='suivre':
        follow(args.out,args.interval,args.once);return
    if args.action=='arret':
        c=read(args.out/'CONTROLLER.json',{})
        if owned_live(c.get('pid'),c.get('pid_start_ticks')):os.kill(c['pid'],signal.SIGTERM);print('Arrêt demandé au contrôleur identifié ; aucun résultat supprimé.')
        else:print('Aucun contrôleur correspondant actif.')
        return
    config=installation(args.installation);data=contract(config['contract']);cases=plan_cases(data)
    if not set(CPUS).issubset(os.sched_getaffinity(0)):raise ValueError('Les cœurs 0 à 21 ne sont pas tous disponibles dans cette session')
    out=args.out.resolve();resume=args.action=='reprendre'
    if not resume and out.exists():raise ValueError('Sortie existante : choisir un nouveau dossier')
    out.mkdir(parents=True,exist_ok=True)
    with machine_lock(),(out/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Un contrôleur possède déjà ce dossier')
        old=read(out/'RUN_CONTRACT.json')
        if resume and not old:raise ValueError('Reprise sans contrat de campagne : refus de mélanger des résultats historiques')
        run_contract=dict(schema='ipopt-default-full-run-contract-v1',run_id=old.get('run_id') if old else out.name+'_'+uuid.uuid4().hex,installation_sha256=config['_sha256'],
            installation=str(Path(config['_path']).resolve()),contract=str(Path(config['contract']).resolve()),contract_sha256=sha(config['contract']),cases=4078,datasets=28,regimes=['I','M','E','Q'],tasks=8156,
            methods=[m for m,_ in METHODS],cpus=list(CPUS),seconds=SECONDS,max_iterations=MAX_ITER,original_kkt_tolerance=1e-6)
        if old and old!=run_contract:raise ValueError('Contrat de reprise différent')
        atomic(out/'RUN_CONTRACT.json',run_contract)
        print('PRÉCONTRÔLE : 4078 QCQP et départs bruts, deux Ipopt par défaut.',flush=True)
        for i,case in enumerate(cases,1):
            verify_input(config,case)
            if i%100==0 or i==len(cases):print(f'FROZEN_VERIFIED {i}/{len(cases)}',flush=True)
        campaign=Campaign(config,out,cases)
        print('DÉMARRAGE : 8156 tâches ; 22 créneaux continus ; CUTEst → Beamforming → PGLIB → autres.',flush=True)
        campaign.execute(resume,max(1,args.interval))

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as error:print('CAMPAGNE REFUSÉE — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
