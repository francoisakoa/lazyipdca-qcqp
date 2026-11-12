#!/usr/bin/env python3
"""Chronométrage confirmatoire séquentiel, données et binaires historiques inchangés.

Standard library only. No optimizer is started by `preparer` or `synthese`.
`lancer` is the sole solver-launching action. Linux /proc and CPU 0 are required.
"""
from pathlib import Path
import argparse, collections, csv, datetime, fcntl, hashlib, json, math, os, re
import resource, signal, statistics, subprocess, sys, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'support'))
import IPOPT_ALIGNE_CONTRACT as IPOPT

THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',
    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
DEFAULT_IPOPT_RUN = '/extra/LAZYQCQP_IPOPT_ALIGNE_RESULTS/IPOPT_4CONFIG_20261109T140336_fa503d1c'
PROTOCOL = dict(schema='quiet-targeted-timing-protocol-v1', repetitions=3, cpu=0,
    threads=1, solver_wall_limit_seconds=7200, solver_cpu_limit_seconds=7200,
    max_iterations=5000, audit_timeout_seconds=600, tolerance=1e-6,
    quiet_window_seconds=10., sample_seconds=1., foreign_cpu_rate_limit=.05,
    foreign_cpu_tick_allowance_seconds=.03, audit_excluded_from_solver_time=True,
    exit_poll_seconds=.005,
    timed_scope='fresh direct process envelope: Popen through wait4; includes loader/startup and shutdown; excludes original audit and quiet waits',
    historical_time_scope_note='Historical envelopes differ by producer and include their respective startup wrappers; no historical startup subtraction is applied.',
    ordering='fixed case order hashed with seed 20261110; four configuration slots; repetition 2 reverses repetition 1, repetition 3 repeats repetition 1 with case-order rotation; no endpoint warm start',
    invalid_environment_policy='keep raw attempt; do not commit nominal repetition; wait for quiet, then retry that same nominal repetition',
    numerical_failure_policy='commit first uncontaminated complete attempt, including numerical failure or budget exhaustion; do not retry for a better result',
    foreign_observation_scope='all /proc processes and aggregate CPU counters on every online CPU, including cores 22 and 23; no foreign process is stopped')

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
def artifact(path, expected=None):
    path = Path(path).expanduser().resolve(strict=True); actual = sha(path)
    if expected and actual != expected: raise ValueError('Empreinte différente : ' + str(path))
    return dict(path=str(path), sha256=actual)
def check_artifact(a):
    if not isinstance(a, dict) or not a.get('path') or not a.get('sha256'): raise ValueError('Pièce non liée')
    artifact(a['path'], a['sha256'])
def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise ValueError('Fichier figé différent : ' + str(path))
    else: write(path, value)
def files_snapshot():
    exact={HERE/'cohort/COHORTE_CHRONO_FIGEE.json',HERE/'evidence/CONTEXTE_HISTORIQUE.json'}
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and (p.suffix=='.py' or p in exact)}

def load_cohort():
    c = read(HERE / 'cohort/COHORTE_CHRONO_FIGEE.json')
    if c.get('cohort_content_sha256') != digest({k:v for k,v in c.items() if k != 'cohort_content_sha256'}):
        raise ValueError('Cohorte figée modifiée')
    if len(c['jobs']) != 550 or len(c['pairs']) != 275 or len({j['case_id'] for j in c['jobs']}) != 179:
        raise ValueError('Cohorte attendue : 275 paires / 179 cas / 550 exécutions uniques')
    if len({j['job_id'] for j in c['jobs']}) != 550: raise ValueError('Tâche dupliquée')
    return c

def verify_start(path, job, external=None):
    artifact(path, job['input_sha256']); binding = job['start_binding']
    if binding.get('external_start_required'):
        if not external or not binding.get('start_file_sha256'): raise ValueError('Départ externe non lié')
        artifact(external, binding['start_file_sha256'])
    else:
        with Path(path).open('rb') as f: raw = next((line for line in f if line.startswith(b'x0 ')), None)
        if raw is None or hashlib.sha256(raw).hexdigest() != binding.get('raw_x0_line_sha256'):
            raise ValueError('Ligne x0 brute différente : ' + job['case_id'])
    if binding.get('start_sha256') != job['start_sha256']: raise ValueError('Identité du départ incohérente')

def verified_ipopt(path):
    """Read the already qualified installation; never construct another binary."""
    item = artifact(path); d = read(path)
    if d.get('schema') != 'ipopt-aligned-qualified-installation-v1' or d.get('production_authorized') is not True:
        raise ValueError('Installation Ipopt alignée qualifiée requise')
    for key in ('build', 'qualification', 'preparation', 'contract', 'frozen_resolution'): check_artifact(d[key])
    prep = read(d['preparation']['path'])
    if prep.get('schema') != 'ipopt-aligned-preparation-v1': raise ValueError('Préparation Ipopt absente')
    for k,v in prep.items():
        if k != 'schema' and d.get(k) != v: raise ValueError('Installation différente de sa préparation : ' + k)
    q = read(d['qualification']['path']); b = read(d['build']['path'])
    for key in ('passed', 'options_verified', 'original_audit_verified', 'production_dependencies_verified',
                'ambient_hostile_option_file_ignored', 'dependencies_rehashed_after_smokes'):
        if q.get(key) is not True: raise ValueError('Qualification Ipopt insuffisante : ' + key)
    if q.get('runs') != 4 or q.get('cases') != 4 or q.get('build_sha256') != d['build']['sha256']:
        raise ValueError('Qualification non liée aux quatre cellules')
    if b.get('schema') != 'isolated-ipopt-aligned-options-build-v1': raise ValueError('Build Ipopt aligné requis')
    for key in ('production_dependencies_verified', 'dependencies_rehashed_after_build', 'original_installation_unchanged'):
        if b.get(key) is not True: raise ValueError('Build Ipopt insuffisant : ' + key)
    pins = {item['path']: item['sha256']}
    for key in ('build', 'qualification', 'preparation', 'contract', 'frozen_resolution'):
        pins[d[key]['path']] = d[key]['sha256']
    for key in ('binary', 'ipopt_library', 'hsl_library'):
        if d.get(key) != b.get(key) or d.get(key+'_sha256') != b.get(key+'_sha256'):
            raise ValueError('Build/installation différents : ' + key)
        pins[d[key]] = artifact(d[key], d[key+'_sha256'])['sha256']
    if q.get('binary_sha256') != d['binary_sha256'] or q.get('ipopt_version') != d['ipopt_version']:
        raise ValueError('Qualification d’un autre binaire ou version')
    for group in ('linked_libraries', 'hsl_linked_libraries'):
        if not b.get(group): raise ValueError('Bibliothèques effectivement liées absentes : ' + group)
        for a in b[group].values(): check_artifact(a); pins[a['path']] = a['sha256']
    for p,h in d.get('source_snapshot', {}).items(): artifact(p,h); pins[p] = h
    if d.get('expected_options') != IPOPT.ALIGNED_OPTIONS: raise ValueError('Options Ipopt modifiées')
    if d.get('budgets') != dict(seconds=7200,max_iter=5000,tolerance=1e-6,threads=1): raise ValueError('Budgets Ipopt modifiés')
    return d, pins

def schedule(cohort):
    by_case = collections.defaultdict(dict)
    for j in cohort['jobs']: by_case[j['case_id']][j['config']] = j['job_id']
    cases = sorted(by_case, key=lambda x: hashlib.sha256(('20261110|' + x).encode()).hexdigest())
    slots = ['R3_ma27', 'ipopt_ma27_monotone_aligned', 'R3_mumps', 'ipopt_mumps_monotone_aligned']
    out = []
    for rep in range(1,4):
        ordering = cases if rep != 3 else cases[len(cases)//2:] + cases[:len(cases)//2]
        for index, case in enumerate(ordering):
            order = slots if index % 2 == 0 else list(reversed(slots))
            if rep == 2: order = list(reversed(order))
            for config in order:
                if config in by_case[case]:
                    job = by_case[case][config]
                    out.append(dict(repetition=rep, job_id=job, nominal_id=job + '_rep' + str(rep)))
    if len(out) != len(cohort['jobs']) * 3 or len({x['nominal_id'] for x in out}) != len(out): raise ValueError('Ordonnancement dupliqué')
    return out

def empty_preparation_output(out):
    """An interrupted mkdir is resumable; every nonempty directory is preserved."""
    out=Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Dossier de préparation absent ou strictement vide requis ; contenu conservé : '+str(out))
    return out

def prepare(out, ipopt_run):
    out = Path(out).expanduser().resolve()
    empty_preparation_output(out)
    if out == HERE or HERE in out.parents or out in HERE.parents: raise ValueError('Sortie séparée du kit requise')
    c = load_cohort(); context = read(HERE/'evidence/CONTEXTE_HISTORIQUE.json'); native = context['installation']
    old = read(native['source_installation']); artifact(native['source_installation'], native['source_installation_sha256'])
    ip, pins = verified_ipopt(Path(ipopt_run)/'preparation/INSTALLATION_IPOPT_ALIGNE.json')
    pins[native['source_installation']] = native['source_installation_sha256']
    pins[native['binary']] = artifact(native['binary'], c['configs']['R3_ma27']['binary_sha256'])['sha256']
    pins[old['binary']] = artifact(old['binary'], c['configs']['R3_mumps']['binary_sha256'])['sha256']
    for config in ('ipopt_ma27_monotone_aligned', 'ipopt_mumps_monotone_aligned'):
        artifact(ip['binary'],c['configs'][config]['binary_sha256'])
    # All archived runtime libraries are held at their historical bytes. No new compilation.
    for p,h in (context.get('runtime_sha256') or {}).items(): artifact(p,h); pins[p]=h
    for key in ('native_build','hsl_library','native_backend_qualification','previous_device_source_proof'):
        if native.get(key) and native.get(key+'_sha256'):
            pins[native[key]] = artifact(native[key],native[key+'_sha256'])['sha256']
    if digest(context['R3']) != context['R3_profile_sha256']: raise ValueError('Profil R3 différent de l’historique')
    cases = {x['id']:x for x in context['cases']}; bindings={}; representatives={}
    for job in c['jobs']:
        cid=job['case_id']; row=ip['frozen_bindings'].get(cid)
        if not row: raise ValueError('Entrée gelée non résolue : ' + cid)
        path=row['input']; start=row.get('start') or job['start_binding'].get('start_file')
        verify_start(path,job,start)
        if cases[cid]['input_sha256'] != job['input_sha256'] or cases[cid]['start_sha256'] != job['start_sha256']:
            raise ValueError('Entrée/départ historique différent : ' + cid)
        rep=IPOPT.representation_for_input(path)
        if rep != ip['input_representation_expectations'][cid]: raise ValueError('Représentation des bornes différente')
        bindings[cid]=dict(input=str(Path(path).resolve()), external_start=start)
        representatives[cid]=rep
    native_env=dict(old.get('environment',{})); native_env.update(native.get('environment',{}))
    configs={
        'R3_ma27':dict(binary=native['binary'], environment=native_env, backend='ma27',native_al_fallback_policy=native['native_al_fallback_policy']),
        'R3_mumps':dict(binary=old['binary'], environment=old.get('environment',{}), backend='mumps', expected_effective_backend='mumps_metis',
            native_al_fallback_policy=old.get('native_al_fallback_policy',native['native_al_fallback_policy'])),
        'ipopt_ma27_monotone_aligned':dict(binary=ip['binary'],environment=ip.get('runtime_environment',{}),backend='ma27'),
        'ipopt_mumps_monotone_aligned':dict(binary=ip['binary'],environment=ip.get('runtime_environment',{}),backend='mumps')}
    if any(configs[k]['native_al_fallback_policy']!='off' for k in ('R3_ma27','R3_mumps')):
        raise ValueError('Politique native différente du protocole historique AL off')
    if 0 not in os.sched_getaffinity(0): raise ValueError('CPU 0 indisponible dans cette session')
    installed=dict(schema='quiet-targeted-timing-installation-v1', created_utc=utc(),
        cohort_sha256=sha(HERE/'cohort/COHORTE_CHRONO_FIGEE.json'), cohort_content_sha256=c['cohort_content_sha256'],
        protocol=PROTOCOL, source_sha256=files_snapshot(), pinned_files=pins, configs=configs,
        inputs=bindings, representations=representatives, R3=context['R3'], ipopt_version=ip['ipopt_version'],
        hsl_library=ip['hsl_library'], schedule=schedule(c), counts=c['counts'], optimizer_invocations=0,
        machine=machine_info(), output_directory=str(out))
    installed['installation_content_sha256']=digest(installed)
    # Recheck after verification: never overwrite a concurrent preparation/result.
    empty_preparation_output(out);out.mkdir(parents=True,exist_ok=True);write(out/'INSTALLATION.json',installed)
    print('PRÊT — 179 instances ; 550 couples instance/configuration ; 1 650 exécutions ; aucun solveur lancé',flush=True)
    return installed

def machine_info():
    cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    model=next((x.split(':',1)[1].strip() for x in cpuinfo.splitlines() if x.startswith('model name')),None)
    return dict(hostname=os.uname().nodename,kernel=os.uname().release,cpu_model=model,
        logical_cpus=os.cpu_count(),available_affinity=sorted(os.sched_getaffinity(0)),python=sys.version,
        governors={p.parent.name:p.read_text().strip() for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_governor')},
        boost={str(p):p.read_text().strip() for p in (Path('/sys/devices/system/cpu/cpufreq/boost'),Path('/sys/devices/system/cpu/intel_pstate/no_turbo')) if p.exists()})

def load_installation(out):
    path=Path(out)/'INSTALLATION.json'; d=read(path)
    if d.get('schema')!='quiet-targeted-timing-installation-v1' or d.get('protocol')!=PROTOCOL: raise ValueError('Installation/protocole incompatible')
    if d.get('installation_content_sha256')!=digest({k:v for k,v in d.items() if k!='installation_content_sha256'}): raise ValueError('Installation modifiée')
    if str(Path(out).resolve())!=d['output_directory']: raise ValueError('Dossier de résultats déplacé')
    if d['cohort_sha256']!=sha(HERE/'cohort/COHORTE_CHRONO_FIGEE.json'): raise ValueError('Cohorte changée')
    for p,h in d['source_sha256'].items(): artifact(HERE/p,h)
    for p,h in d['pinned_files'].items(): artifact(p,h)
    if d['machine']['hostname']!=os.uname().nodename: raise ValueError('Autre machine que celle de la préparation')
    if 0 not in os.sched_getaffinity(0): raise ValueError('CPU 0 indisponible')
    return d

def proc_snapshot():
    """Thread-group utime+stime only; cutime/cstime would double-count children."""
    hz=os.sysconf('SC_CLK_TCK'); processes={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            raw=(path/'stat').read_text(); close=raw.rfind(')'); fields=raw[close+2:].split()
            pid=int(path.name); start=int(fields[19]); group=int(fields[2]); ticks=int(fields[11])+int(fields[12])
            processes[(pid,start)]=dict(cpu=ticks/hz,group=group,comm=raw[raw.find('(')+1:close])
        except (OSError,ValueError,IndexError): pass
    parts=Path('/proc/stat').read_text().splitlines()[0].split()[1:]
    nums=list(map(int,parts)); busy=sum(nums[:8])-nums[3]-(nums[4] if len(nums)>4 else 0)
    return dict(time=time.monotonic(),busy=busy/hz,processes=processes)

def process_start_ticks(pid):
    try:
        raw=Path('/proc',str(pid),'stat').read_text();return int(raw[raw.rfind(')')+2:].split()[19])
    except (OSError,ValueError,IndexError):return None

def ensure_no_owned_live(out):
    for p in (Path(out)/'runs').glob('*/attempt_*/**/OWNED_PROCESS.json'):
        record=read(p)
        if record.get('finished')is not True and record.get('start_ticks')is not None and process_start_ticks(record['pid'])==record['start_ticks']:
            raise ValueError('Processus d’une tentative précédente encore présent : PID '+str(record['pid'])+' ; attendre sa fin avant reprise ; '+str(p))

def quiet_interval(previous,current,owned_groups=()):
    elapsed=current['time']-previous['time']; foreign=[]; own=0.
    for ident,row in current['processes'].items():
        old=previous['processes'].get(ident); delta=max(0.,row['cpu']-(old['cpu'] if old else 0.))
        if ident[0]==os.getpid() or row['group'] in owned_groups: own+=delta; continue
        if delta>0: foreign.append(dict(pid=ident[0],start_ticks=ident[1],comm=row['comm'],cpu_seconds=delta))
    process_cpu=sum(x['cpu_seconds'] for x in foreign)
    allowance=PROTOCOL['foreign_cpu_tick_allowance_seconds']+PROTOCOL['foreign_cpu_rate_limit']*elapsed
    aggregate=max(0.,current['busy']-previous['busy']-own)
    return dict(elapsed_seconds=elapsed,foreign_cpu_seconds=process_cpu,aggregate_unassigned_cpu_seconds=aggregate,
        foreign_processes=sorted(foreign,key=lambda x:-x['cpu_seconds']),process_contamination=process_cpu>allowance,
        aggregate_contamination=aggregate>allowance)

def contaminated_measurement(samples,aggregate_foreign,elapsed,tail_aggregate,tail_elapsed):
    """Keep burst detection; do not dilute a one-second burst over a long solve."""
    def allowance(seconds):return PROTOCOL['foreign_cpu_tick_allowance_seconds']+PROTOCOL['foreign_cpu_rate_limit']*seconds
    return (any(x['process_contamination'] or (x.get('solver_observed') and x['aggregate_contamination']) for x in samples)
        or aggregate_foreign>allowance(elapsed) or tail_aggregate>allowance(tail_elapsed))

class QuietMonitor:
    def __init__(self,log): self.log=Path(log)
    def append(self,row):
        with self.log.open('a') as f: f.write(json.dumps(dict(utc=utc(),**row),ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    def wait_quiet(self,stop):
        previous=proc_snapshot(); quiet=0.
        self.append(dict(phase='preflight',event='waiting_for_systemwide_quiet',machine=machine_info()))
        while quiet<PROTOCOL['quiet_window_seconds']:
            if stop[0]: raise InterruptedError('Interruption pendant attente de calme')
            time.sleep(PROTOCOL['sample_seconds']); current=proc_snapshot(); row=quiet_interval(previous,current)
            quiet=0. if row['process_contamination'] or row['aggregate_contamination'] else quiet+row['elapsed_seconds']
            self.append(dict(phase='preflight',quiet_accumulated_seconds=quiet,**row)); previous=current
        return previous

def environment(config):
    env={k:v for k,v in os.environ.items() if not k.startswith(('LAZY_','ELASTIC_','KERNEL_','IPDCA_'))}
    env.update(config.get('environment',{}));env.update(THREAD_ENV)
    for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT'):env.pop(key,None)
    return env

def run_process(command,folder,env,stop,monitor=None,timeout=7200,cpu=0):
    """One new session, owned-process-only termination, wait4 accounting."""
    folder=Path(folder); before=proc_snapshot() if monitor else None
    self_before=time.process_time(); samples=[]; reason=None;term_at=None
    last_solver_snapshot=before;last_solver_cpu=0.;last_controller_cpu=self_before
    def affinity():
        os.sched_setaffinity(0,{cpu})
        resource.setrlimit(resource.RLIMIT_CPU,(math.ceil(timeout),math.ceil(timeout)+1))
    with (folder/'process.log').open('wb') as log:
        started=time.monotonic(); started_utc=utc()
        proc=subprocess.Popen(command,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=affinity)
        owned=dict(pid=proc.pid,start_ticks=process_start_ticks(proc.pid),command_sha256=digest(command),started_utc=started_utc,finished=False)
        write(folder/'OWNED_PROCESS.json',owned)
        next_sample=started+PROTOCOL['sample_seconds']; previous=before
        while True:
            waited,status,usage=os.wait4(proc.pid,os.WNOHANG)
            if waited:
                rc=os.waitstatus_to_exitcode(status);proc.returncode=rc;break
            now=time.monotonic()
            if monitor and now>=next_sample:
                current=proc_snapshot(); row=quiet_interval(previous,current,(proc.pid,));samples.append(row)
                solver_rows=[r for ident,r in current['processes'].items() if ident[0]==proc.pid]
                row['solver_observed']=bool(solver_rows)
                if solver_rows:
                    last_solver_snapshot=current;last_solver_cpu=solver_rows[0]['cpu'];last_controller_cpu=time.process_time()
                monitor.append(dict(phase='solver',pid=proc.pid,**row));previous=current;next_sample=now+PROTOCOL['sample_seconds']
            if reason is None and (stop[0] or now-started>=timeout):
                reason='interrupted' if stop[0] else 'wall_time_limit';term_at=now
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if term_at is not None and now-term_at>=1.:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                term_at=now+86400.
            time.sleep(PROTOCOL['exit_poll_seconds'])
    elapsed=time.monotonic()-started
    write(folder/'OWNED_PROCESS.json',dict(owned,finished=True,returncode=rc,finished_utc=utc()))
    if reason is None and elapsed>timeout: reason='wall_time_limit'
    if rc<0 and -rc==signal.SIGXCPU: reason='cpu_time_limit'
    result=dict(schema='quiet-solver-resources-v1',started_utc=started_utc,finished_utc=utc(),command=command,
        command_sha256=digest(command),pid=proc.pid,returncode=rc,elapsed_seconds=elapsed,
        user_cpu_seconds=usage.ru_utime,system_cpu_seconds=usage.ru_stime,peak_rss_kib=usage.ru_maxrss,
        voluntary_context_switches=usage.ru_nvcsw,involuntary_context_switches=usage.ru_nivcsw,
        censored=reason is not None,timed_out=reason in ('wall_time_limit','cpu_time_limit'),
        interrupted=reason=='interrupted',termination_reason=reason,cpu=cpu,threads=1)
    if monitor:
        after=proc_snapshot();tail=quiet_interval(previous,after,(proc.pid,));samples.append(tail)
        # wait4 supplies all direct solver CPU even when its /proc entry has disappeared.
        own_cpu=usage.ru_utime+usage.ru_stime+(time.process_time()-self_before)
        aggregate_foreign=max(0.,after['busy']-before['busy']-own_cpu)
        allowance=PROTOCOL['foreign_cpu_tick_allowance_seconds']+PROTOCOL['foreign_cpu_rate_limit']*elapsed
        # The last interval includes a reaped process. Its /proc entry vanished;
        # wait4 corrects the unobserved tail instead of falsely treating it as foreign.
        tail_elapsed=after['time']-last_solver_snapshot['time']
        tail_aggregate=max(0.,after['busy']-last_solver_snapshot['busy']
            -max(0.,usage.ru_utime+usage.ru_stime-last_solver_cpu)
            -(time.process_time()-last_controller_cpu))
        contaminated=contaminated_measurement(samples,aggregate_foreign,elapsed,tail_aggregate,tail_elapsed)
        result['quiet_observation']=dict(contaminated=contaminated,scope=PROTOCOL['foreign_observation_scope'],
            samples=len(samples),process_foreign_cpu_seconds=sum(x['foreign_cpu_seconds'] for x in samples),
            aggregate_unassigned_cpu_seconds=aggregate_foreign,aggregate_allowance_seconds=allowance,
            tail_aggregate_unassigned_cpu_seconds=tail_aggregate,tail_elapsed_seconds=tail_elapsed,
            measurement_limit='CPU-active foreign processes are sampled once per second; aggregate CPU counters also detect short-lived work. Idle processes, I/O wait and non-CPU contention are not excluded by this test.')
        monitor.append(dict(phase='solver',event='final',resources=result,tail=tail))
    return result

def budget_verified(r):
    return IPOPT.process_budget_verified(r)

def quartiles(values):
    """Linear interpolation at (n-1)*p, matching the usual type-7 quantile."""
    if not values:return (None,None)
    values=sorted(values)
    def q(p):
        at=(len(values)-1)*p;lo=math.floor(at);hi=math.ceil(at)
        return values[lo]+(values[hi]-values[lo])*(at-lo)
    return q(.25),q(.75)

def command_for(d,job,folder):
    config=d['configs'][job['config']];binding=d['inputs'][job['case_id']]
    inp=binding['input'];external=binding.get('external_start');endpoint=str(Path(folder)/'endpoint.json')
    if job['role']=='native':
        values=dict(binary=config['binary'],input=inp,endpoint=endpoint,backend=config['backend'],max_outer=5000,seconds=7200)
        cmd=[s.format(**values) for s in d['R3']]
        # The historical campaign applies this installation policy outside the
        # profile template. Omitting it would silently restore the binary's auto default.
        cmd+=['--native-al-fallback',config['native_al_fallback_policy'],'--standard-r10-policy','auto']
        if job['backend']=='ma27':
            cmd+=['--dca-rank-presolve-report',str(Path(folder)/'rank_presolve.json'),'--hsl-library',d['hsl_library']]
        if external: cmd+=['--frozen-start',external]
    else:
        ic=dict(binary=config['binary'],hsl_library=d['hsl_library'])
        cmd=IPOPT.make_command(ic,job,job['config'],endpoint,inp,external)
    return cmd

def verify_endpoint(d,job,path):
    ep=read(path)
    if job['role']!='native':
        return IPOPT.verify_endpoint(ep,job['config'],d['ipopt_version'],d['hsl_library'],d['representations'][job['case_id']])
    route=ep.get('native_al_fallback')
    if not isinstance(route,dict) or route.get('policy')!='off':raise ValueError('Route native AL off non prouvée')
    if any(route.get(k)is not False for k in ('attempted','accepted','enabled')):
        raise ValueError('Une route AL a été activée')
    if any(route.get(k,0)!=0 for k in ('fallback_accepted_newton_steps','fallback_factorizations','fallback_linear_solves')):
        raise ValueError('Travail AL caché')
    if route.get('primary_standard_auto_reason')=='explicit-standard-policy-off':raise ValueError('Politique standard auto désactivée')
    label=str(ep.get('linear_solver_backend','')).lower();attempts=ep.get('lazy_newton_factor_attempts')
    observed=job['backend'] in label
    wrong=any(k in label for k in (('mumps','cholesk','qdldl','suitesparse_ldl') if job['backend']=='ma27' else ('ma27','cholesk','qdldl','suitesparse_ldl')))
    if wrong or not (observed or attempts==0) or ep.get('dca_filter_pivoted_backend'):
        raise ValueError('Backend natif historique non prouvé : '+label)
    # The independent original audit, not the native status flag, decides success.
    return dict(native_binary_and_command_bound=True,solver_success_is_not_certificate=True,
        native_AL_off_verified=True,primary_backend_observed=observed,primary_factor_attempts=attempts,
        endpoint_primary_backend=label,native_status=ep.get('status'),native_endpoint_sha256=sha(path))

def result_complete(path,installation_sha,nominal):
    if not path.exists(): return False
    r=read(path)
    if r.get('installation_sha256')!=installation_sha or r.get('nominal_id')!=nominal: raise ValueError('Résultat lié à un autre plan')
    if r.get('result_content_sha256')!=digest({k:v for k,v in r.items() if k!='result_content_sha256'}): raise ValueError('Résultat final modifié')
    for name,a in r.get('artifacts',{}).items(): check_artifact(a)
    if r.get('quiet_valid')is not True or r.get('finalized')is not True: raise ValueError('Résultat engagé non valide')
    return True

def commit_attempt(folder,final,installation_sha,nominal):
    """Recover the completed clean attempt after a crash before RESULT.json."""
    p=Path(folder)/'ATTEMPT.json'; r=read(p)
    if r.get('installation_sha256')!=installation_sha or r.get('nominal_id')!=nominal:
        raise ValueError('Tentative liée à un autre plan')
    if r.get('attempt_content_sha256')!=digest({k:v for k,v in r.items() if k!='attempt_content_sha256'}):
        raise ValueError('Tentative finale modifiée')
    if not isinstance(r.get('raw_artifacts'),dict):raise ValueError('Pièces brutes de tentative non liées')
    for a in r['raw_artifacts'].values():check_artifact(a)
    if r.get('finalized')is not True:raise ValueError('Tentative non finalisée')
    if r.get('quiet_valid')is not True or r['solver_resources'].get('interrupted')is True:
        return False
    r['artifacts']={p.name:artifact(p) for p in sorted(Path(folder).iterdir()) if p.is_file()}
    r['result_content_sha256']=digest(r);write(final,r)
    return True

def attempt_result(d,job,task,folder,stop):
    binding=d['inputs'][job['case_id']];verify_start(binding['input'],job,binding.get('external_start'))
    conf=d['configs'][job['config']]; artifact(conf['binary'],job['binary_sha256'])
    monitor=QuietMonitor(folder/'QUIET.jsonl');monitor.wait_quiet(stop)
    cmd=command_for(d,job,folder);env=environment(conf)
    numerical_env={k:v for k,v in env.items() if k.startswith(('OMP_','OPENBLAS_','MKL_','BLIS_','NUMEXPR_','VECLIB_','GOTO_')) or k in ('LD_LIBRARY_PATH','HSL_LIBRARY')}
    write(folder/'COMMAND.json',dict(command=cmd,numerical_runtime_environment=numerical_env,job=job,task=task))
    resources=run_process(cmd,folder,environment(conf),stop,monitor)
    write(folder/'solver_resources.json',resources)
    errors=[]; proof=None; audit=None; auditres=None
    if (folder/'endpoint.json').exists():
        try:proof=verify_endpoint(d,job,folder/'endpoint.json')
        except (OSError,ValueError,KeyError,TypeError) as e:errors.append('endpoint_contract: '+str(e))
        # Audit is a separate process and never inside the measured solver envelope.
        write(folder/'case.json',job);audit_dir=folder/'audit';audit_dir.mkdir()
        auditcmd=[sys.executable,str(HERE/'support/AUDIT_ORIGINAL.py'),'--input',binding['input'],
            '--endpoint',str(folder/'endpoint.json'),'--out',str(folder/'AUDIT.json'),
            '--resources',str(folder/'solver_resources.json'),'--case-json',str(folder/'case.json')]
        auditres=run_process(auditcmd,audit_dir,environment(conf),stop,timeout=PROTOCOL['audit_timeout_seconds'])
        write(folder/'audit_resources.json',auditres)
        if (folder/'AUDIT.json').exists():audit=read(folder/'AUDIT.json')
        else:errors.append('original_audit_unavailable')
    else: errors.append('endpoint_unavailable')
    classification=(audit or {}).get('classification','indeterminate')
    if errors or not budget_verified(resources) or (auditres and (auditres['returncode']!=0 or auditres['censored'])):
        classification='indeterminate'
    residuals={k:(audit or {}).get(k) for k in ('primal','dual','complementarity')}
    if classification=='strict' and not (audit.get('strict_qcqp')is True and all(type(v)in(int,float) and math.isfinite(v) and 0<=v<=1e-6 for v in residuals.values())):
        classification='indeterminate';errors.append('strict_certificate_inconsistent')
    failure='none' if classification=='strict' else ('budget_exhaustion' if resources['timed_out'] else
        ('endpoint_above_tolerance' if classification=='nonstrict' else 'unavailable_or_unverified_endpoint'))
    result=dict(schema='quiet-timing-attempt-v1',nominal_id=task['nominal_id'],job_id=job['job_id'],case_id=job['case_id'],
        config=job['config'],backend=job['backend'],regime=job['regime'],repetition=task['repetition'],finished_utc=utc(),
        quiet_valid=not resources['quiet_observation']['contaminated'],solver_resources=resources,
        classification=classification,failure_cause=failure,strict_qcqp=classification=='strict',budget_verified=budget_verified(resources),
        residuals=residuals,objective=(audit or {}).get('objective'),proof=proof,errors=errors,
        seconds_process=resources['elapsed_seconds'],seconds_process_scope=PROTOCOL['timed_scope'],
        audit_seconds=(audit or {}).get('audit_seconds'),all_run_costs_retained=True)
    return result

def run(out):
    out=Path(out).resolve();d=load_installation(out);c=load_cohort();jobs={j['job_id']:j for j in c['jobs']}
    # Prevent another timing controller for this user, even with a different output directory.
    global_lock=Path('/tmp')/('lazyqcqp_quiet_timing_'+str(os.getuid())+'.lock')
    with (out/'CONTROLLER.lock').open('a') as lock,global_lock.open('a') as machine_lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Contrôleur déjà actif dans ce dossier')
        try:fcntl.flock(machine_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Autre campagne de chronométrage active')
        return run_locked(out,d,jobs)

def run_locked(out,d,jobs):
    ensure_no_owned_live(out)
    stop=[False]
    def stopping(signum,frame):stop[0]=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopping)
    install_sha=sha(out/'INSTALLATION.json');completed=0
    for task in d['schedule']:
        nominal=out/'runs'/task['nominal_id'];final=nominal/'RESULT.json'
        if result_complete(final,install_sha,task['nominal_id']):completed+=1;continue
        if stop[0]:break
        nominal.mkdir(parents=True,exist_ok=True)
        # A crash may leave raw evidence; never overwrite or silently reuse it.
        attempts=sorted(p for p in nominal.glob('attempt_*') if p.is_dir())
        recovered=False
        for old in attempts:
            if (old/'ATTEMPT.json').exists() and commit_attempt(old,final,install_sha,task['nominal_id']):
                recovered=True;completed+=1;break
            if not (old/'ATTEMPT.json').exists() and not (old/'INTERRUPTED.json').exists():
                write(old/'INTERRUPTED.json',dict(discovered_utc=utc(),reason='controller stopped before attempt finalization',raw_files_retained=True))
        if recovered:continue
        while not stop[0]:
            folder=nominal/('attempt_'+str(len(attempts)+1).zfill(4));folder.mkdir();attempts.append(folder)
            write(folder/'STARTED.json',dict(utc=utc(),installation_sha256=install_sha,task=task))
            write(out/'STATUS.json',dict(utc=utc(),state='waiting_for_quiet_then_solver_and_audit',active=task,
                attempt=str(folder),quiet_log=str(folder/'QUIET.jsonl'),completed=completed,planned=len(d['schedule'])))
            print('ATTENTE DE CALME — '+task['nominal_id']+' ; journal '+str(folder/'QUIET.jsonl'),flush=True)
            try:r=attempt_result(d,jobs[task['job_id']],task,folder,stop)
            except BaseException as e:
                write(folder/'INTERRUPTED.json',dict(utc=utc(),reason=type(e).__name__+': '+str(e),raw_files_retained=True))
                raise
            r.update(installation_sha256=install_sha,finalized=True)
            r['raw_artifacts']={p.name:artifact(p) for p in sorted(folder.iterdir()) if p.is_file()}
            r['attempt_content_sha256']=digest(r)
            write(folder/'ATTEMPT.json',r)
            if r['solver_resources']['interrupted'] or stop[0]:break
            if r['quiet_valid']:
                commit_attempt(folder,final,install_sha,task['nominal_id']);completed+=1
                print(f"CALME — {completed}/{len(d['schedule'])} ; {task['nominal_id']} ; {r['classification']} ; {r['seconds_process']:.6f} s",flush=True)
                break
            print('CHARGE ÉTRANGÈRE — tentative conservée, répétition à refaire après retour au calme : '+task['nominal_id'],flush=True)
        write(out/'STATUS.json',dict(utc=utc(),state='interrupted' if stop[0] else 'running',completed=completed,planned=len(d['schedule']),optimizer_invocations=sum(1 for p in (out/'runs').glob('*/attempt_*/solver_resources.json'))))
    write(out/'STATUS.json',dict(utc=utc(),state='completed' if completed==len(d['schedule']) else 'interrupted',completed=completed,planned=len(d['schedule'])))
    return completed

def summarize(out):
    out=Path(out).resolve();d=read(out/'INSTALLATION.json');c=load_cohort();ish=sha(out/'INSTALLATION.json')
    # Summary works without the remote binaries; it revalidates immutable result evidence.
    if d.get('installation_content_sha256')!=digest({k:v for k,v in d.items() if k!='installation_content_sha256'}):raise ValueError('Installation modifiée')
    rows=[];byjob=collections.defaultdict(list)
    for task in d['schedule']:
        p=out/'runs'/task['nominal_id']/'RESULT.json'
        if result_complete(p,ish,task['nominal_id']):r=read(p);rows.append(r);byjob[r['job_id']].append(r)
    attempts=[read(p) for p in (out/'runs').glob('*/attempt_*/ATTEMPT.json')]
    resource_records=[read(p) for p in (out/'runs').glob('*/attempt_*/solver_resources.json')]
    audit_resources=[read(p) for p in (out/'runs').glob('*/attempt_*/audit_resources.json')]
    missing_resources=[str(p.parent) for p in (out/'runs').glob('*/attempt_*/STARTED.json') if not (p.parent/'solver_resources.json').exists()]
    jobstats={}
    for j in c['jobs']:
        rr=sorted(byjob[j['job_id']],key=lambda r:r['repetition'])
        complete=len(rr)==3 and {r['repetition'] for r in rr}=={1,2,3}
        allstrict=complete and all(r['classification']=='strict' for r in rr)
        jobstats[j['job_id']]=dict(case_id=j['case_id'],config=j['config'],complete_three=complete,
            strict_three=allstrict,repetitions=len(rr),times_seconds=[r['seconds_process'] for r in rr],
            median_seconds=statistics.median([r['seconds_process'] for r in rr]) if complete else None,
            minimum_seconds=min([r['seconds_process'] for r in rr]) if complete else None,
            maximum_seconds=max([r['seconds_process'] for r in rr]) if complete else None,
            classifications=collections.Counter(r['classification'] for r in rr))
    pairrows=[]
    for pair in c['pairs']:
        a=jobstats[pair['native_job_id']];b=jobstats[pair['reference_job_id']]
        complete=a['complete_three'] and b['complete_three'];strict=complete and a['strict_three'] and b['strict_three']
        pairrows.append(dict(**pair,three_repetitions_complete=complete,both_strict_all_three=strict,
            native_median_seconds=a['median_seconds'],reference_median_seconds=b['median_seconds'],
            speedup_reference_over_R3=b['median_seconds']/a['median_seconds'] if strict and a['median_seconds']>0 else None,
            ratio_R3_over_reference=a['median_seconds']/b['median_seconds'] if strict and b['median_seconds']>0 else None,
            native_classifications=a['classifications'],reference_classifications=b['classifications']))
    cohorts=[]
    for cohort in c['cohorts']:
        name=cohort['cohort_id'];pp=[p for p in pairrows if p['cohort_id']==name];eligible=[p for p in pp if p['both_strict_all_three']]
        values=[p['speedup_reference_over_R3'] for p in eligible]
        ratios=[p['ratio_R3_over_reference'] for p in eligible]
        q=quartiles(ratios)
        cohorts.append(dict(cohort_id=name,frozen_pairs=len(pp),complete_pairs=sum(p['three_repetitions_complete'] for p in pp),
            both_strict_all_three=len(eligible),pairs_excluded_from_speed_summary=len(pp)-len(eligible),
            median_paired_speedup=statistics.median(values) if values else None,
            geometric_mean_paired_speedup=math.exp(sum(math.log(v) for v in values)/len(values)) if values else None,
            median_ratio_R3_over_reference=statistics.median(ratios) if ratios else None,
            ratio_R3_over_reference_q1=q[0],ratio_R3_over_reference_q3=q[1],
            R3_faster_pairs=sum(v>1 for v in values),reference_faster_pairs=sum(v<1 for v in values)))
    report=dict(schema='quiet-targeted-timing-summary-v1',created_utc=utc(),protocol=PROTOCOL,
        frozen_counts=c['counts'],nominal_repetitions_completed=len(rows),planned=1650,complete=len(rows)==1650,
        committed_solver_seconds=sum(r['seconds_process'] for r in rows),
        all_attempt_solver_seconds=sum(r['elapsed_seconds'] for r in resource_records),
        all_attempt_audit_seconds=sum(r['elapsed_seconds'] for r in audit_resources),
        attempted_solver_processes_with_resources=len(resource_records),
        incomplete_attempts_without_resource_record=missing_resources,
        contaminated_attempts=sum(not r['quiet_valid'] for r in attempts),
        incomplete_attempts=len(list((out/'runs').glob('*/attempt_*/INTERRUPTED.json'))),
        failure_breakdown=dict(collections.Counter(r['classification'] for r in rows)),
        failure_causes=dict(collections.Counter(r.get('failure_cause','unknown') for r in rows)),
        cohorts=cohorts,job_statistics=jobstats,pairs=pairrows,
        interpretation='Cohorts were selected using historical strict success and times >1 s, never new timings. Speed summaries use medians of exactly three valid quiet repetitions and report numerical attrition; all failures and invalid attempts remain in raw data. No global-corpus speed claim is implied.')
    write(out/'SYNTHESE_CHRONO.json',report)
    with (out/'PAIRES_CHRONO.csv').open('w',newline='') as f:
        keys=['cohort_id','case_id','three_repetitions_complete','both_strict_all_three','native_median_seconds','reference_median_seconds','speedup_reference_over_R3','ratio_R3_over_reference']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(pairrows)
    with (out/'OBSERVATIONS_CHRONO.csv').open('w',newline='') as f:
        keys=['nominal_id','case_id','config','backend','regime','repetition','classification','failure_cause','budget_verified','seconds_process','audit_seconds']
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps({k:report[k] for k in ('complete','nominal_repetitions_completed','planned','contaminated_attempts','cohorts')},ensure_ascii=False,indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('preparer');a.add_argument('--out',type=Path,required=True);a.add_argument('--ipopt-run',type=Path,default=Path(DEFAULT_IPOPT_RUN))
    for action in ('lancer','synthese'):
        a=s.add_parser(action);a.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='preparer':prepare(a.out,a.ipopt_run)
    elif a.action=='lancer':run(a.out)
    else:summarize(a.out)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,InterruptedError) as e:
        print('CHRONOMÉTRAGE ARRÊTÉ — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
