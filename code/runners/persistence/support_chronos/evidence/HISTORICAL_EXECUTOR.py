#!/usr/bin/env python3
"""Native frozen task executor; source-derived from the R6 executor, without predecessor dependencies."""
from __future__ import annotations
import fcntl,gzip,hashlib,json,math,os,shutil,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
METHODS=('R3','HH','HS','SH','SS','SS_legacy','ipopt_qcqp','ipopt_mumps')
RESEARCH_METHODS=('F_R3','F_X1','F_X2','F_H0')
THREAD_ENV=dict(OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
STOP=threading.Event()
ACTIVE={}
ACTIVE_LOCK=threading.RLock()
RESULT_CACHE={}
RUN_RESULTS={}
RESULT_LOCK=threading.RLock()


def cpu_slots_lock():
    """Lifetime lock shared with preparation/default controllers onCPUs0-21."""
    path=Path('/tmp/LAZYQCQP_CPU_0_21.lock')
    try:fd=os.open(path,os.O_RDWR|os.O_CREAT,0o666)
    except PermissionError:fd=os.open(path,os.O_RDONLY)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise ValueError('Les22cœurs0-21 sont déjà réservés par un contrôleur ou une qualification du kit')
    return os.fdopen(fd,'r')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(str(tmp), str(path))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def identity(pid):
    try:
        # Fields after the final ')' start at the process state, field 3.
        fields = Path('/proc/%d/stat' % int(pid)).read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]
    except (OSError, ValueError, IndexError, TypeError):
        return None


def matching_process(pid, started):
    return started is not None and identity(pid) == str(started)


def child_identity(pid, command):
    """Only report a /proc identity when its full command matches our child."""
    try:
        actual = Path('/proc/%d/cmdline' % pid).read_bytes().rstrip(b'\0').split(b'\0')
        expected = [os.fsencode(token) for token in command]
        if actual != expected:
            return None
        return identity(pid)
    except (OSError, ValueError):
        return None


def resolve_file(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def parse_cpus(text):
    values = []
    for part in text.split(','):
        if '-' in part:
            lo, hi = map(int, part.split('-', 1))
            if lo < 0 or hi < lo:
                raise ValueError('Invalid CPU range: ' + part)
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(part))
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise ValueError('CPU list must contain distinct nonnegative indices')
    return values


def environment(installation):
    env = dict(os.environ)
    # Do not inherit solver tuning from an unrelated research shell.
    for name in list(env):
        if name.startswith(('LAZY_', 'ELASTIC_', 'KERNEL_', 'IPDCA_')):
            env.pop(name, None)
    env.update({str(k): str(v) for k, v in installation.get('environment', {}).items()})
    env.update(THREAD_ENV)
    for key in ('LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG', 'LD_DEBUG_OUTPUT'):
        env.pop(key, None)
    return env


def frozen_files(case):
    paths = [(case['input'], case['input_sha256'])]
    for name, key in [('start_file', 'start_file_sha256'),
                      ('problem_file', 'problem_sha256')]:
        if case.get(name):
            # A start hash without a physical file is not silently invented.
            if not case.get(key):
                raise ValueError('Missing %s for %s' % (key, case['id']))
            paths.append((case[name], case[key]))
    return paths


def al_options(mode):
    if mode != 'off':
        raise ValueError('This unified campaign requires AL off')
    return ['--native-al-fallback', mode, '--standard-r10-policy', 'auto']


def al_route_check(endpoint, mode, method):
    if method in ('ipopt_qcqp','ipopt_mumps'):
        return dict(mode=mode, applicable=False, passed=True, violations=[])
    route = endpoint.get('native_al_fallback')
    violations = []
    if not isinstance(route, dict):
        violations.append('native_AL_route_metadata_missing')
    else:
        if route.get('policy') != mode:
            violations.append('native_al_fallback.policy_differs_from_plan')
        if route.get('primary_standard_auto_reason') == 'explicit-standard-policy-off':
            violations.append('primary_standard_policy_was_explicitly_disabled')
        if mode == 'off':
            for field in ('attempted', 'accepted', 'enabled'):
                if route.get(field) is not False:
                    violations.append('native_al_fallback.%s_is_not_false' % field)
            for field in ('fallback_accepted_newton_steps', 'fallback_factorizations', 'fallback_linear_solves'):
                if route.get(field, 0) != 0:
                    violations.append('native_al_fallback.%s_is_nonzero' % field)
    return dict(mode=mode, applicable=True, passed=not violations, violations=violations,
                route_policy=route.get('policy') if isinstance(route, dict) else None,
                attempted=route.get('attempted') if isinstance(route, dict) else None,
                accepted=route.get('accepted') if isinstance(route, dict) else None,
                primary_standard_auto_enabled=route.get('primary_standard_auto_enabled') if isinstance(route, dict) else None)


def native_backend_evidence(endpoint,installation):
    """Report actual primary backend separately from early structural exits."""
    label=str(endpoint.get('linear_solver_backend',''))
    attempts=endpoint.get('lazy_newton_factor_attempts')
    attempts=attempts if type(attempts)is int and attempts>=0 else None
    observed='ma27'in label.casefold()
    wrong=any(name in label.casefold()for name in('mumps','cholesk','qdldl','suitesparse_ldl'))
    no_primary_work=attempts==0
    passed=(installation.get('native_backend')=='ma27'and not wrong and
            (observed or no_primary_work)and not endpoint.get('dca_filter_pivoted_backend')and
            endpoint.get('lazy_newton_native_factor_successes',0)==0)
    return dict(requested='ma27',endpoint_primary_backend=label,backend_observed=observed,
                primary_factor_attempts=attempts,early_exit_without_primary_factorization=no_primary_work,
                **{'pass':passed},KKT_certificate=False)


def verify_case(case):
    for value, expected in frozen_files(case):
        path = resolve_file(value)
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Frozen input hash mismatch: ' + str(path))


def native_command(installation, profiles, method, case, endpoint, seconds, max_outer, al_mode='off'):
    if method in RESEARCH_METHODS:
        from factorial_runtime import command
        return command(installation,method,case,endpoint)
    binary = resolve_file(installation['binary'])
    if installation.get('native_backend')!='ma27':raise ValueError('All six native profiles require declared MA27')
    mapping = dict(binary=str(binary), input=str(resolve_file(case['input'])),
                   endpoint=str(endpoint), backend='ma27', seconds=str(seconds),
                   max_outer=str(max_outer))
    command = [str(token).format(**mapping) for token in profiles['profiles'][method]]
    command += ['--dca-rank-presolve-report', str(Path(endpoint).parent / 'rank_presolve.json')]
    command += al_options(al_mode)
    command += ['--hsl-library',str(resolve_file(installation['hsl_library']))]
    for flag,value in [('--shared-restoration-budget','1'),('--restoration-iters','600')]:
        positions=[j for j,token in enumerate(command[:-1]) if token==flag]
        if positions:
            for j in positions:command[j+1]=value
        else:command.extend([flag,value])
    if case.get('external_start_required'):
        if not case.get('start_file'):
            raise ValueError('External start missing: ' + case['id'])
        command += ['--frozen-start', str(resolve_file(case['start_file']))]
    return command


def ipopt_command(installation, case, case_file, endpoint, seconds, max_outer):
    template = installation.get('ipopt_command')
    if not template:
        raise ValueError('Ipopt command was not qualified at installation')
    mapping = dict(python=installation.get('python', sys.executable),
                   input=str(resolve_file(case['input'])), endpoint=str(endpoint),
                   seconds=str(seconds), max_outer=str(max_outer),
                   case_json=str(case_file), root=str(HERE))
    return [str(token).format(**mapping) for token in template]


def result_path(out, task):
    return out / 'tasks' / task['id'] / 'RESULT.json'


def completed_result(out, task):
    path = result_path(out, task)
    cache_key = (str(path), task['contract_sha256'])
    if cache_key in RESULT_CACHE:
        return RESULT_CACHE[cache_key]
    if not path.is_file():
        return None
    result = read(path)
    if result.get('contract_sha256') != task['contract_sha256']:
        raise ValueError('Completed result contract mismatch: ' + str(path))
    # Evidence itself is guarded, not only the result's success flag.
    for value, expected in result.get('evidence_sha256', {}).items():
        evidence = out / value
        if not evidence.is_file() or sha(evidence) != expected:
            raise ValueError('Completed evidence changed: ' + str(evidence))
    for value, expected in result.get('external_evidence_sha256', {}).items():
        evidence = Path(value)
        if not evidence.is_file() or sha(evidence) != expected:
            raise ValueError('Reused source evidence changed: ' + str(evidence))
    RESULT_CACHE[cache_key] = result
    return result


def kill_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute(command, log, env, cpu, timeout, task_id, phase):
    resources = log.with_name(phase + '_resources.json')
    wrapped = [sys.executable, str(HERE / 'resource_worker.py'), '--resource-json', str(resources), '--'] + command
    if not Path('/usr/bin/taskset').is_file():raise ValueError('Per-task CPU affinity requires /usr/bin/taskset')
    full = ['/usr/bin/taskset', '-c', str(cpu)] + wrapped
    began = time.monotonic()
    with log.open('wb') as stream:
        process = subprocess.Popen(full, stdin=subprocess.DEVNULL, stdout=stream,
                                   stderr=subprocess.STDOUT, env=env,
                                   cwd=str(HERE), start_new_session=True)
        with ACTIVE_LOCK:
            ACTIVE[task_id] = dict(pid=process.pid, starttime=child_identity(process.pid, command),
                                   cpu=cpu, phase=phase, started=stamp(), command=command)
        stopped = timed_out = False
        while process.poll() is None:
            if STOP.is_set():
                stopped = True
                kill_child(process)
                break
            if time.monotonic() - began > timeout:
                timed_out = True
                kill_child(process)
                break
            STOP.wait(.1)
        code = process.wait()
    with ACTIVE_LOCK:
        ACTIVE.pop(task_id, None)
    measurements = read(resources) if resources.is_file() else None
    return dict(returncode=code, seconds=time.monotonic() - began,
                interrupted=stopped, external_timeout=timed_out, resources=measurements,
                resource_measurement_available=measurements is not None,
                resource_scope='Linux wait4 child rusage; process wall includes wrapper startup; numerical endpoint timers remain separate')


ENDPOINT_VECTOR_FIELDS = frozenset((
    'x', 'primal', 'primal_variables', 'primal_solution', 'slacks',
    'certificate_equality_multipliers', 'certificate_inequality_multipliers',
    'certificate_lower_multipliers', 'certificate_upper_multipliers',
    'equality_multipliers', 'inequality_multipliers', 'lower_bound_multipliers', 'upper_bound_multipliers',
    'termination_inequality_slacks'))


def compact_endpoint(endpoint, endpoint_sha256):
    """Keep scalar counters, statuses, timers, policies and diagnostic histories.

    Only known primal/dual/slack vector arrays are omitted, at any nesting level.
    The original raw endpoint remains untouched and is separately hashed.
    """
    omitted=[]
    def visit(value, path=''):
        if isinstance(value, dict):
            result={}
            for key,item in value.items():
                current=path+'.'+key if path else key
                if key in ENDPOINT_VECTOR_FIELDS and isinstance(item,list):
                    omitted.append(dict(path=current, length=len(item)))
                else:
                    result[key]=visit(item,current)
            return result
        if isinstance(value,list):
            return [visit(item,path+'[%d]'%j) for j,item in enumerate(value)]
        return value
    data=visit(endpoint)
    return dict(schema='compact-endpoint-metadata-v1', raw_endpoint_sha256=endpoint_sha256,
                omitted_vector_arrays=omitted, raw_endpoint_preserved=True,
                scope='Only named primal, multiplier and slack vector arrays omitted; all scalar fields and other nested diagnostics retained',
                endpoint=data)


def run_task(out, plan, task, cpu):
    directory = out / 'tasks' / task['id']
    if directory.exists() and any(directory.iterdir()):
        archive = directory / 'interrupted_attempts' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
        archive.mkdir(parents=True)
        for file in list(directory.iterdir()):
            if file.name != 'interrupted_attempts':
                file.rename(archive / file.name)
    directory.mkdir(parents=True, exist_ok=True)
    endpoint = directory / 'endpoint.json'
    case_file = directory / 'case.json'
    case = dict(task['case'])
    # Workers receive absolute, hash-checked paths independent of their cwd.
    for key in ('input', 'start_file', 'problem_file'):
        if case.get(key):
            case[key] = str(resolve_file(case[key]))
    case['method'] = task['method']
    case['al_mode'] = plan.get('al_mode', 'auto')
    case['seconds'] = plan['seconds']
    case['max_outer'] = plan['max_outer']
    case['restoration_max_iterations'] = plan['shared_restoration']
    write(case_file, case)
    installation = plan['installation']
    env = environment(installation)
    if task['method'] in ('ipopt_qcqp','ipopt_mumps'):
        command = ipopt_command(installation, case, case_file, endpoint,
                                plan['seconds'], plan['max_outer'])
    else:
        command = native_command(installation, plan['profiles'], task['method'], case,
                                 endpoint, plan['seconds'], plan['max_outer'], plan.get('al_mode', 'auto'))
    write(directory / 'COMMAND.json', dict(command=command, cpu=cpu,
          environment=THREAD_ENV, al_mode=plan.get('al_mode', 'auto'), contract_sha256=task['contract_sha256'], created=stamp()))
    work = execute(command, directory / 'solver.log', env, cpu,
                   plan['seconds'] + 30, task['id'], 'solver')
    case['solve_process_seconds'] = work['seconds']
    case['solve_external_timeout'] = work['external_timeout']
    write(case_file, case)
    if work['interrupted']:
        write(directory / 'INTERRUPTED.json', dict(work=work, created=stamp()))
        return None
    result = dict(schema='mega-fresh-two-ipopt-result-v1', case_id=case['id'],
                  method=task['method'], al_mode=plan.get('al_mode', 'auto'), contract_sha256=task['contract_sha256'],
                  work=work, cpu=cpu, strict_qcqp=False, solution_origin='unknown',
                  created=stamp(), evidence_sha256={},run_id=plan.get('run_id'),
                  dataset=case.get('dataset_id'),regime=case.get('regime'),family=case.get('family'),
                  input_sha256=case['input_sha256'],start_sha256=case.get('start_sha256')or case.get('embedded_start_sha256'),
                  native_backend='ma27'if task['method']not in('ipopt_qcqp','ipopt_mumps')else None)
    if endpoint.is_file():
        raw_endpoint = read(endpoint)
        result['native_status']=raw_endpoint.get('status')
        result['iterations']=raw_endpoint.get('outer_iterations',raw_endpoint.get('iterations'))
        for key in ('ipopt_application_status','ipopt_application_status_code','effective_options'):
            result[key]=raw_endpoint.get(key)
        markers=('iteration','factorization','factorisation','linear_solve','newton','restoration','seconds','time_','backtrack')
        result['internal_work']={k:v for k,v in raw_endpoint.items()if any(m in k.lower()for m in markers)and(v is None or type(v)in(str,int,bool)or type(v)is float and math.isfinite(v))}
        write(directory / 'ENDPOINT_METADATA.json', compact_endpoint(raw_endpoint, sha(endpoint)))
        route_check = al_route_check(raw_endpoint, plan.get('al_mode', 'auto'), task['method'])
        if task['method'] in ('ipopt_qcqp','ipopt_mumps'):
            sys.path.insert(0,str(HERE.parent))
            from setup_runtime import backend_evidence
            backend_check = backend_evidence(raw_endpoint, (directory / 'solver.log').read_text(errors='replace'), installation, method=task['method'], require_banner=False)
            result['backend_check'] = backend_check
            write(directory / 'BACKEND_CHECK.json', backend_check)
            route_check['passed'] = bool(route_check['passed'] and backend_check['pass'])
        if task['method']not in('ipopt_qcqp','ipopt_mumps'):
            backend_check=native_backend_evidence(raw_endpoint,installation)
            result['backend_check']=backend_check
            write(directory/'BACKEND_CHECK.json',backend_check)
            route_check['passed']=bool(route_check['passed']and backend_check['pass'])
        if task['method']in RESEARCH_METHODS:
            from factorial_runtime import check
            try:result['factorial_proof']=check(raw_endpoint,directory,task['method'],installation)
            except (OSError,ValueError,KeyError,TypeError)as error:
                route_check['passed']=False
                result['factorial_proof_error']=str(error)
        result['route_policy_check'] = route_check
        write(directory / 'ROUTE_POLICY_CHECK.json', route_check)
        audit_command = [installation.get('python', sys.executable),
                         str(HERE / 'audit_worker.py'), '--case-json', str(case_file),
                         '--endpoint', str(endpoint), '--out', str(directory / 'AUDIT.json')]
        write(directory / 'AUDIT_COMMAND.json', {'command': audit_command})
        audit_work = execute(audit_command, directory / 'audit.log', env, cpu,
                             installation.get('audit_timeout', 1800), task['id'], 'audit')
        if audit_work['interrupted']:
            write(directory / 'INTERRUPTED.json', dict(work=work, audit_work=audit_work, created=stamp()))
            return None
        result['audit_work'] = audit_work
        if not audit_work['returncode'] and (directory / 'AUDIT.json').is_file():
            audit = read(directory / 'AUDIT.json')
            result['audit'] = audit
            result['strict_qcqp'] = audit.get('strict_qcqp') is True and not work['external_timeout'] and route_check['passed']
            result['solution_origin'] = audit.get('solution_origin', 'unknown')
            result['status'] = 'audited' if route_check['passed'] else 'route_policy_violation'
        else:
            result['status'] = 'audit_error'
    else:
        result['status'] = 'endpoint_absent'
    if task['method']in RESEARCH_METHODS:
        from factorial_runtime import validate
        cfg=validate(installation);result['binary_sha256']=cfg['binary_sha256'];result['backend']='ma27'
    elif task['method']in('ipopt_qcqp','ipopt_mumps'):
        result['binary_sha256']=installation['ipopt_binary_sha256'];result['backend']='ma27'if task['method']=='ipopt_qcqp'else'mumps'
    else:result['binary_sha256']=installation['binary_sha256'];result['backend']='ma27'
    result['installation_sha256']=digest(installation)
    audit=result.get('audit')or{}
    for key in ('primal','dual','complementarity','objective'):result[key]=audit.get(key)
    result['seconds_process']=work['seconds'];result['audit_seconds']=(result.get('audit_work')or{}).get('seconds')
    result['budget_verified']=(audit.get('budget_and_timing')or{}).get('budget_integrity_pass')is True
    if task['method']in RESEARCH_METHODS:
        proof=result.get('factorial_proof')or{}
        result.update(proof)
        result.setdefault('cell',{'F_R3':'R3','F_X1':'X1','F_X2':'X2','F_H0':'H0'}[task['method']])
        for key in ('objective_exact','trace_success'):result.setdefault(key,None)
    # Full log is retained losslessly, including all curvature/controller records.
    # Compression occurs after numerical solve/audit, outside the solve timer.
    full_log = directory / 'solver.log'
    if full_log.is_file():
        compressed_log = directory / 'solver.log.gz'
        with full_log.open('rb') as source, compressed_log.open('wb') as target:
            with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as stream:
                shutil.copyfileobj(source, stream, 1048576)
        full_log.unlink()
        result['solver_log'] = 'solver.log.gz'
        result['solver_log_lossless_compression'] = True
    for file in directory.iterdir():
        if file.is_file() and file.name not in ('RESULT.json',):
            result['evidence_sha256'][str(file.relative_to(out))] = sha(file)
    write(directory / 'RESULT.json', result)
    # This is the executor receipt, not a finalized campaign observation.
    # guarded_run_task publishes only after normalization and post-run integrity.
    return result

