#!/usr/bin/env python3
"""Run the immutable 50-case, three-way native diagonal parity gate.

This script never substitutes missing trajectories or resamples a failed case.
Fresh MA27 references using R3 and HH controls are compared with a passive observer,
then with the research diagonals. H0's exact objective is not historical HH.
"""
from __future__ import annotations
import argparse,collections,concurrent.futures,datetime,hashlib,importlib.util,json,math,os
from pathlib import Path
import queue,re,signal,struct,sys,threading,time,subprocess
import PROFILES

HERE=Path(__file__).resolve().parent
STATE_KEYS=('accepted_index','state_sha256','outer_iteration','phase_n','phase_mi','phase_me','alpha','mu')
STATE_EVENTS={'startup_restoration_commit','state_recovery_commit','terminal_central_restart_commit','post_multiplier_update','central_restart_commit'}
VECTOR_KEYS=('x','certificate_equality_multipliers','certificate_inequality_multipliers','certificate_lower_multipliers','certificate_upper_multipliers')
CELLS={'R3':('cache','r3'),'X1':('cache','hs'),'X2':('hybrid','r3'),'H0':('hybrid','hs')}
OBSERVER_PATHS={'src/factorial_trace.hpp','src/atomic_newton.cpp','src/main.cpp','src/dca_inequality_majorant_engine.hpp','include/lazyipdca_qcqp/settings.hpp'}
OBSERVATION_COUNTERS=('free_epigraph_checkpoint_lifts',)
OBSERVATION_METADATA=(('standard_auto_checkpoint_requested',),('standard_auto_checkpoint_effective',),
    ('native_al_fallback','primary_standard_auto_checkpoint_requested'),('native_al_fallback','primary_standard_auto_checkpoint_effective'))

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(),parse_int=lambda s:-0.0 if s=='-0' else int(s))
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def common_runner():
    for p in (HERE.parent/'CAMPAGNE.py',HERE.parent/'default_relaunch/CAMPAGNE.py'):
        if p.is_file():return load_module('factorial_common_runner',p)
    raise ValueError('Shared continuous-slot process controller CAMPAGNE.py unavailable')
def cpus(text):
    out=[]
    for part in text.split(','):
        if '-' in part:a,b=map(int,part.split('-',1));out.extend(range(a,b+1))
        else:out.append(int(part))
    if len(set(out))!=len(out) or not out:raise ValueError('Invalid or duplicate CPUs')
    return out
def clock_key(key):
    return key.endswith('_seconds') or key in ('seconds','cost_seconds','elapsed','wall_s','cpu_s')
def nonclock(value):
    if isinstance(value,dict):return {k:nonclock(v) for k,v in value.items() if not clock_key(k)}
    if isinstance(value,list):return [nonclock(v) for v in value]
    return value
def exact_tree(value):
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Nonfinite comparison value')
        return ['binary64',struct.pack('<d',value).hex()]
    if isinstance(value,dict):return {k:exact_tree(v) for k,v in sorted(value.items())}
    if isinstance(value,list):return [exact_tree(v) for v in value]
    return value
def digest(value):return hashlib.sha256(json.dumps(exact_tree(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def differences(a,b,prefix='',out=None):
    out=[] if out is None else out
    if len(out)>=80:return out
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            path=prefix+'.'+k
            if k not in a or k not in b:out.append(path+': missing key')
            else:differences(a[k],b[k],path,out)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):out.append(prefix+': different length')
        else:
            for i,(x,y) in enumerate(zip(a,b)):differences(x,y,prefix+'['+str(i)+']',out)
    elif exact_tree(a)!=exact_tree(b):out.append(prefix+': different value')
    return out
def endpoint_parity(a,b,allow_observation_delta=False):
    aa,bb=nonclock(a),nonclock(b)
    deltas={};metadata={}
    if allow_observation_delta:
        for key in OBSERVATION_COUNTERS:
            if key in aa or key in bb:
                left,right=aa.pop(key,None),bb.pop(key,None)
                if not isinstance(left,int) or not isinstance(right,int) or isinstance(left,bool) or isinstance(right,bool):raise ValueError('Observation counter absent or invalid')
                deltas[key]={'reference':left,'research':right,'research_minus_reference':right-left}
        for keys in OBSERVATION_METADATA:
            left,right=aa,bb
            for key in keys[:-1]:
                left=left.get(key,{});right=right.get(key,{})
            key=keys[-1]
            if key in left or key in right:
                x,y=left.pop(key,None),right.pop(key,None)
                if type(x) is not bool or type(y) is not bool:raise ValueError('Checkpoint I/O metadata absent or invalid')
                metadata['.'.join(keys)]={'reference':x,'research':y}
    return {'passed':digest(aa)==digest(bb),'nonclock_sha256':[digest(aa),digest(bb)],'differences':differences(aa,bb),
            'observation_work_deltas':deltas,'observation_configuration_values':metadata,
            'observation_only_allowlist':list(OBSERVATION_COUNTERS) if allow_observation_delta else [],
            'observation_metadata_allowlist':['.'.join(k)for k in OBSERVATION_METADATA] if allow_observation_delta else []}
def trace_summary(observation):return {k:v for k,v in observation.items() if k!='states'}
def trace(path,endpoint_path,expect_exact=None):
    rows=[]
    with Path(path).open() as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if not rows or rows[0].get('event')!='header' or rows[0].get('schema')!='lazy-factorial-accepted-trace-v1':raise ValueError('Accepted trace header absent')
    header=rows[0]
    if header.get('state_encoding')!='sha256-length-prefixed-binary64-le-x-s-z-lambda' or header.get('state_coordinates')!='native-phase-internal':raise ValueError('Unsupported state trace encoding')
    policies={k:header.get(k) for k in ('inequality_policy','globalization_policy','objective_exact')}
    if (policies['inequality_policy'],policies['globalization_policy']) not in CELLS.values() or type(policies['objective_exact']) is not bool:raise ValueError('Invalid trace policy header')
    accepted=[];states=[];current_index=0
    for row in rows[1:]:
        event=row.get('event')
        if event=='accepted':
            current_index+=1;accepted.append(row)
            if row.get('accepted_index')!=current_index:raise ValueError('Accepted trace index invalid')
            keys=STATE_KEYS
            if expect_exact is True and row.get('objective_E0_zero') is not True:raise ValueError('Nonzero objective correction in exact-objective cell')
        elif event in STATE_EVENTS:
            if row.get('after_accepted_index')!=current_index:raise ValueError('State event refers to wrong accepted generation')
            keys=('after_accepted_index','state_sha256','outer_iteration','phase_n','phase_mi','phase_me','mu')
        else:raise ValueError('Unrecognized trace event: '+str(event))
        if any(row.get(k)!=v for k,v in policies.items()):raise ValueError('Trace event policy differs from its header')
        if not re.fullmatch('[0-9a-f]{64}',row.get('state_sha256','')) or any(k not in row for k in keys):raise ValueError('Incomplete state hash/event')
        states.append({'event':event,**{k:row[k] for k in keys}})
    side=read(str(endpoint_path)+'.factorial.json')
    if side.get('schema')!='lazy-factorial-endpoint-v1' or side.get('trace_success') is not True:raise ValueError('Trace finalization failed')
    if any(side.get(k)!=v for k,v in policies.items()):raise ValueError('Trace endpoint policy differs from its header')
    if side.get('accepted_events')!=len(accepted):raise ValueError('Trace/sidecar accepted count differs')
    if expect_exact is not None and side.get('objective_exact') is not expect_exact:raise ValueError('Wrong exact/legacy objective mode')
    if side.get('endpoint_accepted_steps')!=len(accepted):raise ValueError('Endpoint accepted count differs or is missing')
    if side.get('budget_verified') is not True:raise ValueError('Trace budget proof missing/invalid')
    if any(type(side.get(k)) is not int or not 0<=side[k]<=bound for k,bound in [('outer_iterations',5000),('restoration_iterations',600)]):raise ValueError('Trace actual iteration budget invalid')
    seconds=side.get('seconds')
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=7200:raise ValueError('Trace actual time budget invalid')
    return dict(header=rows[0],sidecar=side,accepted_events=len(accepted),states=states,
                state_sequence_sha256=digest(states),trace_sha256=sha(path),objective_zero_at_all_commits=all(r.get('objective_E0_zero') is True for r in accepted))
def trace_parity(a,b):
    return dict(passed=digest(a['states'])==digest(b['states']),accepted_events=[a['accepted_events'],b['accepted_events']],
                state_sequence_sha256=[a['state_sequence_sha256'],b['state_sequence_sha256']],
                differences=differences(a['states'],b['states']))
def censored(resources,endpoint):
    reasons=[]
    if resources.get('censored') or resources.get('returncode',0)<0:reasons.append('external_censor')
    status=str(endpoint.get('status','')).lower()
    if status in ('time_limit','max_outer_iterations','max_iterations','interrupted','external_timeout'):reasons.append('terminal_resource_limit')
    if float(endpoint.get('elapsed_seconds',0) or 0)>7200:reasons.append('native_time_budget_exceeded')
    if int(endpoint.get('outer_iterations',0) or 0)>5000:reasons.append('native_outer_budget_exceeded')
    return reasons
def input_binding(root,case,input_config=None):
    ident=case['case_id']
    if Path(ident).name!=ident or ident in ('.','..'):raise ValueError('Invalid case identifier')
    if input_config is not None:
        helper=load_module('factorial_bound_inputs',HERE/'VERIFIER_COUPLAGE.py')
        p=helper.bound_input(input_config,case)
    else:p=root/'data/frozen'/ident/'input.qcq'
    if sha(p)!=case['input_sha256']:raise ValueError('Frozen coefficients differ: '+ident)
    with p.open('rb') as f:
        line=next((line for line in f if line.startswith(b'x0 ')),None)
    if not line:raise ValueError('Raw start line absent')
    x=list(map(float,line.split()[1:]))
    if not all(math.isfinite(v) for v in x):raise ValueError('Nonfinite raw start')
    h=hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
    if h!=case['start_sha256']:raise ValueError('Frozen primal start differs: '+ident)
    return p
def objective_indefinite(path):
    # Exact binary64 coefficient witnesses on one/two coordinate supports.
    from fractions import Fraction
    diagonal={};off={}
    with Path(path).open() as f:
        for line in f:
            if line.strip()=='objective':break
        for line in f:
            if line.startswith('hessian_nnz '):count=int(line.split()[1]);break
        else:return False
        # A bounded structural witness search. Selection falls back to the
        # preregistered hash order if this exact sparse scan is too large.
        if count>1000000:return False
        for _ in range(count):
            i,j,a=f.readline().split();i=int(i);j=int(j);v=Fraction(float(a))
            if i==j:diagonal[i]=diagonal.get(i,Fraction(0))+v
            else:
                key=tuple(sorted((i,j)));off[key]=off.get(key,Fraction(0))+v
    if any(v<0 for v in diagonal.values()) and any(v>0 for v in diagonal.values()):return True
    return any(diagonal.get(i,0)*diagonal.get(j,0)-v*v<0 for (i,j),v in off.items())
def select_exact_probes(sample,paths):
    selected=[]
    for regime in ('I','M'):
        group=[c for c in sample if c['regime']==regime]
        group.sort(key=lambda c:hashlib.sha256(c['case_id'].encode()).hexdigest())
        choices=[c for c in group if objective_indefinite(paths[c['case_id']])]
        c=(choices or group)[0]
        selected.append(dict(case=c,selection='exact_indefinite_coordinate_witness' if choices else 'first_structural_hash_no_indefinite_witness'))
    return selected

class Gate:
    def __init__(self,args,runner,sample,paths,profiles,environment):
        self.a=args;self.runner=runner;self.sample=sample;self.paths=paths;self.profiles=profiles;self.environment=environment
        self.stop=threading.Event();self.monitor_stop=threading.Event();self.lock=threading.RLock();self.queue=queue.Queue();self.results=[];self.active={};self.last_print=0
        for c in sample:
            for method in ('R3','HH'):self.queue.put(('diagonal',c,method))
        self.probes=select_exact_probes(sample,paths)
        for rec in self.probes:
            for cell in ('X1','X2','H0'):self.queue.put(('exact',rec['case'],cell))
    def status(self):
        with self.lock:
            atomic(self.a.out/'STATUS.json',dict(schema='factorial50-progress-v1',timestamp=utc(),
                state='interrupted' if self.stop.is_set() else 'running',planned_diagonals=100,planned_exact_probes=6,
                done=len(self.results),active=dict(self.active),remaining=self.queue.qsize()))
            if time.monotonic()-self.last_print>=10:
                print('QUALIFICATION 50 — comparaisons '+str(len(self.results))+'/106 ; créneaux '+str(len(self.active))+'/22 ; validées '+str(sum(r['passed']for r in self.results)),flush=True);self.last_print=time.monotonic()
    def monitor(self):
        while not self.monitor_stop.wait(10):self.status()
    def execute(self,command,folder,cpu):
        folder.mkdir(parents=True,exist_ok=False);atomic(folder/'COMMAND.json',dict(command=command,command_sha256=digest(command)))
        res=self.runner.run_process(command,folder,folder/'solver.log',7200,self.stop,cpu=cpu,env_extra=self.environment)
        res['schema']='factorial-diagonal-process-resources-v1';atomic(folder/'solver_resources.json',res)
        path=folder/'endpoint.json'
        if not path.is_file():raise ValueError('Native endpoint absent')
        endpoint=read(path)
        reasons=censored(res,endpoint)
        if reasons:raise ValueError('Censored qualification observation: '+','.join(reasons))
        return endpoint,res,path
    def unit(self,kind,case,method,cpu):
        ident=case['case_id'];folder=self.a.out/'tasks'/(ident+'__'+method+('__exact' if kind=='exact' else '__diagonal'))
        result=dict(case_id=ident,regime=case['regime'],dataset=case['dataset'],kind=kind,method=method,passed=False,
                    input_sha256=case['input_sha256'],start_sha256=case['start_sha256'],native_executions=0,completed_native_executions=0,resources={},reason=None)
        try:
            inp=self.paths[ident]
            result['input_path']=str(Path(inp).resolve())
            result['input_installation_sha256']=getattr(self.a,'input_installation_sha256',None)
            if kind=='exact':
                dest=folder/'research';endpoint=dest/'endpoint.json';tracepath=dest/'trace.jsonl'
                command=PROFILES.make_command(self.profiles,method,self.a.binary,inp,endpoint,tracepath,backend='ma27',hsl_library=self.a.hsl_library)
                result['native_executions']+=1;data,res,path=self.execute(command,dest,cpu)
                result['completed_native_executions']+=1;result['resources']['research']=res
                obs=trace(tracepath,path,True)
                if obs['accepted_events']<1:raise ValueError('Exact-objective runtime probe has no accepted model direction')
                expected=CELLS[method]
                if (obs['sidecar'].get('inequality_policy'),obs['sidecar'].get('globalization_policy'))!=expected:raise ValueError('Exact-objective runtime probe used the wrong factorial cell')
                result.update(passed=True,trace=trace_summary(obs),endpoint_sha256=sha(path),runtime_objective_E0_zero=True)
            else:
                olddir=folder/'original';path=olddir/'endpoint.json'
                command=PROFILES.make_legacy_command(self.profiles,method,self.a.baseline_binary,inp,path,backend='ma27',hsl_library=self.a.hsl_library)
                result['native_executions']+=1;old,ores,oldpath=self.execute(command,olddir,cpu)
                result['completed_native_executions']+=1;result['resources']['original']=ores
                obsdir=folder/'observer';path=obsdir/'endpoint.json';tr=obsdir/'trace.jsonl'
                command=PROFILES.make_legacy_command(self.profiles,method,self.a.baseline_trace_binary,inp,path,trace=tr,backend='ma27',hsl_library=self.a.hsl_library)
                result['native_executions']+=1;observed,sres,obspath=self.execute(command,obsdir,cpu)
                result['completed_native_executions']+=1;result['resources']['observer']=sres
                obs=trace(tr,obspath,method=='R3');result['baseline_observer_endpoint_parity']=endpoint_parity(old,observed)
                candir=folder/'research';path=candir/'endpoint.json';tr=candir/'trace.jsonl'
                cell='R3' if method=='R3' else 'H0'
                command=PROFILES.make_command(self.profiles,cell,self.a.binary,inp,path,tr,legacy=method=='HH',backend='ma27',hsl_library=self.a.hsl_library)
                result['native_executions']+=1;candidate,cres,canpath=self.execute(command,candir,cpu)
                result['completed_native_executions']+=1;result['resources']['research']=cres
                can=trace(tr,canpath,method=='R3')
                result['observer_research_endpoint_parity']=endpoint_parity(observed,candidate,allow_observation_delta=method=='HH')
                result['trajectory_parity']=trace_parity(obs,can)
                result['endpoint_sha256']={k:sha(p) for k,p in [('original',oldpath),('observer',obspath),('research',canpath)]}
                result['traces']={'observer':trace_summary(obs),'research':trace_summary(can)}
                result['passed']=all(result[k]['passed'] for k in ('baseline_observer_endpoint_parity','observer_research_endpoint_parity','trajectory_parity'))
                if method=='HH' and can['sidecar'].get('legacy_objective_validation') is not True:raise ValueError('Historical HH validation flag absent')
                if not result['passed']:result['reason']='actual_endpoint_counter_or_trajectory_mismatch'
        except Exception as error:result['reason']=type(error).__name__+': '+str(error)
        atomic(folder/'COMPARISON.json',result)
        return result
    def worker(self,cpu):
        while not self.stop.is_set():
            try:kind,case,method=self.queue.get_nowait()
            except queue.Empty:return
            with self.lock:self.active[str(cpu)]=dict(case_id=case['case_id'],method=method,kind=kind)
            self.status();result=self.unit(kind,case,method,cpu)
            with self.lock:self.results.append(result);self.active.pop(str(cpu),None)
            self.queue.task_done();self.status()
    def run(self,cpulist):
        old={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
        for s in old:signal.signal(s,lambda *_:self.stop.set())
        monitor=threading.Thread(target=self.monitor,daemon=True);monitor.start()
        try:
            pool=concurrent.futures.ThreadPoolExecutor(max_workers=len(cpulist))
            try:
                for future in concurrent.futures.as_completed([pool.submit(self.worker,c) for c in cpulist]):future.result()
            except BaseException:
                self.stop.set();raise
            finally:pool.shutdown(wait=True,cancel_futures=True)
        finally:
            self.stop.set()
            self.monitor_stop.set();monitor.join()
            for s,handler in old.items():signal.signal(s,handler)
        return self.results

def verify_admission(receipt,original,adapted,adapter_sha):
    if receipt.get('schema')!='lazy-ma27-backend-support-adaptation-v1' or receipt.get('primary_backend')!='ma27':raise ValueError('MA27 source admission proof absent')
    if receipt.get('adapter_sha256')!=adapter_sha or receipt.get('original_tree_unchanged')is not True or receipt.get('hidden_mumps_substitution')is not False:raise ValueError('MA27 source admission identity/routing differs')
    if receipt.get('models_inputs_starts_changed')is not False or receipt.get('original_certificate_tests_changed')is not False or receipt.get('pivot_decisions_changed')is not False:raise ValueError('MA27 admission changed a protected numerical mechanism')
    if receipt.get('original_source_sha256')!=original or receipt.get('adapted_source_sha256')!=adapted or set(original)!=set(adapted):raise ValueError('MA27 admission source maps differ')
    changes={c['path']:c for c in receipt.get('changed_files',[])}
    actual={name for name in original if original[name]!=adapted[name]}
    if set(changes)!=actual or not actual:raise ValueError('MA27 admission source delta is incomplete')
    for name in actual:
        if changes[name].get('before_sha256')!=original[name] or changes[name].get('after_sha256')!=adapted[name]:raise ValueError('MA27 admission changed-file hash differs')
    if sha(receipt['patch'])!=receipt.get('patch_sha256'):raise ValueError('MA27 admission patch changed')
    return True

def verify_native_ma27_qualification(proof,path,binary,hsl_sha):
    if proof.get('schema')!='native-ma27-six-profiles-qualification-v1' or proof.get('passed')is not True or proof.get('six_profiles_qualified')is not True or proof.get('native_backend')!='ma27':raise ValueError('Actual six-profile MA27 qualification is absent or failed')
    if proof.get('binary_sha256')!=sha(binary) or proof.get('hsl_library_sha256')!=hsl_sha or proof.get('mumps_compiled')is not False:raise ValueError('MA27 qualification backend/binary/HSL binding differs')
    methods={'R3','HH','HS','SH','SS','SS_legacy'}
    if set(proof.get('profiles',[]))!=methods or proof.get('native_executions')!=24 or proof.get('completed_native_runs')!=24:raise ValueError('Twenty-four actual MA27 qualification observations are required')
    observations=proof.get('observations',[])
    if len(observations)!=24 or {(r.get('regime'),r.get('method'))for r in observations}!={(r,m)for r in 'IMEQ'for m in methods}:raise ValueError('MA27 six-profile regime controls differ')
    if sha(proof['auditor'])!=proof.get('auditor_sha256') or sha(HERE/'QUALIFIER_MA27.py')!=proof.get('qualifier_sha256') or sha(HERE/'PROFILES.py')!=proof.get('profiles_factory_sha256'):raise ValueError('MA27 qualification evaluator/factory changed')
    auditor=load_module('factorial_ma27_actual_auditor',Path(proof['auditor']))
    for row in observations:
        if row.get('passed')is not True or row.get('ma27_primary_successful_factors',0)<1 or'ma27'not in str(row.get('native_backend','')).lower():raise ValueError('MA27 primitive/runtime route is unqualified')
        fixture=proof['fixtures'][row['regime']];inp=Path(fixture['path']);endpoint=path.parent/'tasks'/(row['regime']+'__'+row['method'])/'endpoint.json'
        if sha(inp)!=fixture['sha256'] or sha(endpoint)!=row['endpoint_sha256']:raise ValueError('Actual MA27 qualification input/endpoint changed')
        data=read(endpoint);audit=auditor.audit(auditor.read_model(inp),data)
        if not audit.get('strict') or'ma27'not in str(data.get('linear_solver_backend','')).lower():raise ValueError('Actual MA27 endpoint lacks a strict original audit/backend')
    return dict(passed=True,native_backend='ma27',binary_sha256=sha(binary),hsl_library_sha256=hsl_sha,six_profiles_qualified=True,actual_runs=24,proof_sha256=sha(path))

def verify_build(proof,args,published):
    required=['research_binary','research_binary_sha256','baseline_trace_binary','baseline_trace_binary_sha256',
              'research_source_sha256','baseline_trace_source_sha256','baseline_source_sha256','observer_source_proof',
              'baseline_binary','baseline_binary_sha256','native_dependency_proof','native_dependency_proof_sha256',
              'profiles_factory_sha256','ma27_admissions','ma27_baseline_source_sha256','original_research_source_sha256',
              'source_directories','ma27_adapter','ma27_adapter_sha256','hsl_runtime_proof','hsl_runtime_proof_sha256',
              'hsl_library','hsl_library_sha256','historical_installed_binary','historical_installed_binary_sha256',
              'native_ma27_qualification','native_ma27_qualification_sha256']
    if any(k not in proof for k in required):raise ValueError('Native build proof lacks required MA27 source/observer/runtime bindings')
    if proof.get('schema')!='lazy-factorial-three-native-builds-ma27-v2' or proof.get('native_backend')!='ma27' or proof.get('mumps_compiled')is not False or proof.get('all_binaries_homogeneous')is not True:raise ValueError('Three fresh homogeneous MA27-only builds are required; historical MUMPS gates do not qualify')
    for name,binary in [('baseline',args.baseline_binary),('research',args.binary),('baseline_trace',args.baseline_trace_binary)]:
        if Path(proof[name+'_binary']).resolve()!=binary or sha(binary)!=proof[name+'_binary_sha256']:raise ValueError('Native '+name+' MA27 binary binding differs')
        backends=subprocess.check_output([str(binary),'--linear-backends'],text=True).strip().split(',')
        if'ma27'not in backends or'mumps'in backends:raise ValueError('Native binary does not enforce the requested MA27-only backend')
    historical=Path(proof['historical_installed_binary']).resolve()
    if sha(historical)!=proof['historical_installed_binary_sha256'] or proof['historical_installed_binary_sha256']!=published['installation']['binary_sha256']:raise ValueError('Protected historical installed binary changed')
    for key in ('native_dependency_proof','hsl_runtime_proof','native_ma27_qualification'):
        if sha(Path(proof[key]))!=proof[key+'_sha256']:raise ValueError('Native MA27 dependency/runtime qualification receipt differs')
    if sha(HERE/'PROFILES.py')!=proof['profiles_factory_sha256'] or sha(proof['ma27_adapter'])!=proof['ma27_adapter_sha256']:raise ValueError('Canonical profiles factory or MA27 adapter changed')
    runtime=load_module('factorial_actual_hsl_pin',HERE/'MA27_RUNTIME.py');hsl_pin=runtime.verify_pin(read(proof['hsl_runtime_proof']))
    if hsl_pin['hsl_library_sha256']!=proof['hsl_library_sha256']or Path(hsl_pin['hsl_library']).resolve()!=Path(proof['hsl_library']).resolve():raise ValueError('Build and actual HSL pin refer to different libraries')
    dependencies=read(proof['native_dependency_proof'])
    if dependencies.get('native_backend')!='ma27'or dependencies.get('mumps_compiled')is not False or dependencies.get('runtime_environment')!=proof.get('runtime_environment')or proof.get('runtime_environment',{}).get('HSL_LIBRARY')!=proof['hsl_library']:raise ValueError('MA27 runtime environment differs from its pinned dependency receipt')
    if sha(proof['hsl_library'])!=proof['hsl_library_sha256']:raise ValueError('Actual HSL library changed')
    source=published['source_sha256'];expected={k.split('/source/',1)[-1]:v for k,v in source.items()}
    if proof['baseline_source_sha256']!=expected or len(expected)!=823:raise ValueError('Original823 source differs from the published source provenance')
    original_research={str(p.relative_to(HERE/'source')):sha(p)for p in sorted((HERE/'source').rglob('*'))if p.is_file()}
    if original_research!=proof['original_research_source_sha256']:raise ValueError('Original factorial source changed')
    for name,mapkey in [('baseline','ma27_baseline_source_sha256'),('baseline_trace','baseline_trace_source_sha256'),('research','research_source_sha256')]:
        directory=Path(proof['source_directories'][name]);actual={str(p.relative_to(directory)):sha(p)for p in sorted(directory.rglob('*'))if p.is_file()}
        if actual!=proof[mapkey]:raise ValueError('Actual adapted '+name+' source differs from its homogeneous build')
    verify_admission(proof['ma27_admissions']['baseline'],expected,proof['ma27_baseline_source_sha256'],proof['ma27_adapter_sha256'])
    verify_admission(proof['ma27_admissions']['research'],original_research,proof['research_source_sha256'],proof['ma27_adapter_sha256'])
    observer=proof['observer_source_proof'];observer=read(observer)if isinstance(observer,str)else observer
    if observer.get('schema')!='lazy-factorial-baseline-observer-source-v1'or observer.get('qualified_files')!=823 or observer.get('numerical_changes')is not False:raise ValueError('Passive MA27 reference observer proof unavailable')
    if not{c['path']for c in observer['changes']}<=OBSERVER_PATHS:raise ValueError('Observer changes extend beyond the passive whitelist')
    reference=proof['ma27_baseline_source_sha256'];observed=proof['baseline_trace_source_sha256'];changes={c['path']:c for c in observer['changes']}
    if set(observed)!=set(reference)|{'src/factorial_trace.hpp'}:raise ValueError('Unexpected passive observer source additions/removals')
    for name,value in observed.items():
        if value!=reference.get(name):
            entry=changes.get(name,{})
            if entry.get('qualified_sha256')!=reference.get(name)or entry.get('observer_sha256')!=value:raise ValueError('Passive observer source hash differs: '+name)
    qualified=verify_native_ma27_qualification(read(proof['native_ma27_qualification']),Path(proof['native_ma27_qualification']),args.baseline_binary,proof['hsl_library_sha256'])
    return dict(source_proof_passed=True,native_backend='ma27',ma27_reference_scope='fresh same-backend backend-support reference; historical MUMPS binary is provenance only',observer_source_proof=observer,build_proof_sha256=sha(args.build_proof),native_ma27_qualification=qualified,baseline_source_files=823,research_source_files=len(proof['research_source_sha256']),baseline_trace_source_files=len(observed))

def verify_coupling(proof,args,registration,build):
    if proof.get('schema')!='factorial-corpus-coupling-gate-v1':raise ValueError('Full-corpus coefficient coupling preflight schema absent')
    if proof.get('passed') is not True or proof.get('coupling_separation_verified') is not True:raise ValueError('Full-corpus coefficient coupling separation is unverified')
    if proof.get('cases')!=1834 or proof.get('inspected_cases')!=1834 or proof.get('changed_cases')!=0 or proof.get('regimes')!={'I':1037,'M':797}:raise ValueError('Every one of the 1834 I/M cases must have an unchanged signed-square representation')
    if proof.get('unresolved_cases')!=0 or proof.get('missing_cases')!=0:raise ValueError('Unresolved or missing full-corpus coefficient inspections')
    if proof.get('native_backend')!='ma27' or proof.get('hsl_library_sha256')!=build.get('hsl_library_sha256'):raise ValueError('Coefficient inspection MA27/HSL binding differs')
    if getattr(args,'build_proof',None)is not None and proof.get('native_build_proof_sha256')!=sha(args.build_proof):raise ValueError('Coefficient inspection MA27 build proof differs')
    expected={'factor_binary_sha256':sha(args.binary),'registration_sha256':sha(args.registration),
              'contract_sha256':registration['inputs']['contract_file_sha256'],'research_source_sha256':build['research_source_sha256']}
    for key,value in expected.items():
        if proof.get(key)!=value:raise ValueError('Coefficient coupling preflight binding differs: '+key)
    input_installation=getattr(args,'input_installation',None)
    if input_installation is not None:
        installation_sha=sha(input_installation)
        if proof.get('input_installation_sha256')!=installation_sha:raise ValueError('Coefficient inspection input installation differs')
        input_config=read(input_installation)
    else:input_config=None
    relative=Path(proof.get('rows_file',''))
    rows_file=(args.coupling_proof.parent/relative).resolve()
    if not str(relative) or relative.is_absolute() or not rows_file.is_relative_to(args.coupling_proof.parent):raise ValueError('Invalid coefficient inspection evidence path')
    if sha(rows_file)!=proof.get('rows_sha256'):raise ValueError('Actual coefficient inspection rows differ')
    bindings=registration['inputs']['case_bindings'];case_map={r['case_id']:r for r in bindings}
    if len(bindings)!=1834 or len(case_map)!=1834:raise ValueError('Malformed complete preregistered input bindings')
    rows=[json.loads(line) for line in rows_file.read_text().splitlines() if line.strip()]
    if len(rows)!=1834 or {r.get('case_id')for r in rows}!=set(case_map):raise ValueError('Coefficient inspection evidence lacks the exact 1834 unique cases')
    if collections.Counter(r.get('regime')for r in rows)!=collections.Counter({'I':1037,'M':797}):raise ValueError('Coefficient inspection row regime counts differ')
    for row in rows:
        binding=case_map[row['case_id']]
        if any(row.get(k)!=binding[k] for k in ('input_sha256','start_sha256')):raise ValueError('Coefficient inspection frozen input/start binding differs: '+row['case_id'])
        if row.get('native_backend')!='ma27'or row.get('hsl_library_sha256')!=build.get('hsl_library_sha256'):raise ValueError('Coefficient row MA27/HSL binding differs')
        if row.get('passed') is not True or row.get('signed_square_changed') is not False or row.get('unresolved_structural_phase') is not False:raise ValueError('Unresolved or changed coefficient representation: '+row['case_id'])
        if input_config is not None:
            supplied=input_config.get('frozen_bindings',{}).get(row['case_id'],{}).get('input')
            if not supplied or row.get('input_path')!=str(Path(supplied).resolve()) or row.get('input_installation_sha256')!=installation_sha:
                raise ValueError('Coefficient inspection resolved input path differs: '+row['case_id'])
        if not row.get('dataset') or not re.fullmatch('[0-9a-f]{64}',row.get('inspection_sha256','')):raise ValueError('Actual native coefficient inspection receipt is absent: '+row['case_id'])
    return {'coupling_separation_verified':True,'proof_sha256':sha(args.coupling_proof),'cases':1834,'regimes':proof['regimes'],'changed_cases':0,'rows_file':str(rows_file),'rows_sha256':sha(rows_file)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--baseline-trace-binary',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build-proof',type=Path,default=HERE/'BUILD_NATIVE.json')
    p.add_argument('--coupling-proof',type=Path,required=True)
    p.add_argument('--input-installation',type=Path)
    p.add_argument('--jobs',type=int,default=22);p.add_argument('--cpus',default='0-21')
    a=p.parse_args();a.root=a.root.resolve();a.binary=a.binary.resolve();a.baseline_trace_binary=a.baseline_trace_binary.resolve();a.registration=a.registration.resolve();a.build_proof=a.build_proof.resolve();a.coupling_proof=a.coupling_proof.resolve();a.out=a.out.resolve()
    if a.out.exists():raise SystemExit('Qualification output must be fresh; no result or sample is overwritten')
    a.out.mkdir(parents=True)
    registration_sha=sha(a.registration);results=[];gate=dict(schema='factorial-50-diagonal-gate-v1',native_backend='ma27',passed=False,created_utc=utc(),registration_sha256=registration_sha,
        fixed_sample_cases=50,required_trajectory_comparisons=100,required_diagonal_executions=300,required_exact_probes=6,
        all_four_production_cells_have_exact_objective=True,production_authorized=False,coupling_separation_verified=False,observations=[],failure=None)
    try:
        reg=read(a.registration);sample=reg['parity_gate']['sample']
        if a.input_installation is None:raise ValueError('Explicit --input-installation is required; frozen paths are not inferred')
        a.input_installation=a.input_installation.resolve()
        input_helper=load_module('factorial_bound_input_setup',HERE/'VERIFIER_COUPLAGE.py')
        setup=read(a.input_installation)
        input_config=input_helper.input_installation(a.input_installation,a.root,Path(setup['contract']))
        a.input_installation_sha256=input_config['_input_installation_sha256']
        if setup.get('contract_sha256')!=reg['inputs']['contract_file_sha256']:raise ValueError('Input installation differs from preregistered corpus')
        if len(sample)!=50 or collections.Counter(c['regime'] for c in sample)!=collections.Counter({'I':25,'M':25}):raise ValueError('Exactly the fixed25I+25M sample is required')
        if len({c['case_id'] for c in sample})!=50:raise ValueError('Duplicate qualification case')
        if reg['parity_gate'].get('selection_uses_outcomes') is not False:raise ValueError('Qualification sample is not outcome independent')
        cpu_list=cpus(a.cpus)
        if a.jobs!=22 or cpu_list!=list(range(22)):raise ValueError('This qualification uses exactly22slots onCPUs0-21')
        if not set(cpu_list)<=set(os.sched_getaffinity(0)):raise ValueError('Requested22CPUs unavailable in this session')
        published=read(HERE/'profile_evidence/PUBLISHED_CONTEXT.json');profiles=published['profiles']['profiles']
        build_receipt=read(a.build_proof)
        a.baseline_binary=Path(build_receipt['baseline_binary']).resolve();a.hsl_library=Path(build_receipt['hsl_library']).resolve()
        build=verify_build(build_receipt,a,published)
        coupling=verify_coupling(read(a.coupling_proof),a,reg,build_receipt)
        gate.update(coupling_separation_verified=True,coupling_preflight=coupling,coupling_proof_sha256=sha(a.coupling_proof))
        for rel,expected in build_receipt['baseline_source_sha256'].items():
            if sha(a.root/'source'/rel)!=expected:raise ValueError('Installed original source differs: '+rel)
        paths={c['case_id']:input_binding(a.root,c,input_config) for c in sample}
        atomic(a.out/'REGISTRATION.json',reg);atomic(a.out/'PUBLISHED_PROFILES.json',profiles)
        runner=common_runner();env=build_receipt['runtime_environment'].copy()
        # Only relocate a historical prefix if the installation itself moved.
        oldroot=published['installation']['binary'].split('/install/bin/')[0]
        env={k:(v.replace(oldroot,str(a.root)) if isinstance(v,str) else v) for k,v in env.items()}
        bound_files=[Path(build_receipt['historical_installed_binary']),a.hsl_library,Path(build_receipt['hsl_runtime_proof']),Path(build_receipt['native_ma27_qualification']),a.baseline_binary,a.baseline_trace_binary,a.binary,a.registration,a.build_proof,a.coupling_proof,a.input_installation,Path(coupling['rows_file']),HERE/'PROFILES.py']+list((HERE/'profile_evidence').glob('*.json'))
        before={str(x):sha(x) for x in bound_files}
        controller=Gate(a,runner,sample,paths,profiles,env)
        with runner.machine_lock():results=controller.run(cpu_list)
        unchanged=all(sha(Path(path))==value for path,value in before.items())
        # The delivered sources must still be the ones attached to the tested
        # executables; changes during a long qualification invalidate the gate.
        verify_build(read(a.build_proof),a,published)
        for rel,expected in build_receipt['baseline_source_sha256'].items():
            if sha(a.root/'source'/rel)!=expected:raise ValueError('Original source changed during qualification: '+rel)
        diag=[r for r in results if r['kind']=='diagonal'];exact=[r for r in results if r['kind']=='exact']
        passed=len(diag)==100 and len(exact)==6 and all(r['passed'] for r in results) and unchanged and gate['coupling_separation_verified']
        gate.update(passed=passed,production_authorized=passed,source_bindings=build,
            r3_diagonal_cases=sum(r['passed'] and r['method']=='R3' for r in diag),
            hs_diagonal_cases=sum(r['passed'] and r['method']=='HH' for r in diag),
            factor_binary_sha256=before[str(a.binary)],baseline_binary_sha256=before[str(a.baseline_binary)],
            native_backend='ma27',hsl_library_sha256=before[str(a.hsl_library)],historical_mumps_binary_sha256=build_receipt['historical_installed_binary_sha256'],
            baseline_scope='fresh MA27 reference, not a historical MUMPS trajectory',
            contract_sha256=reg['inputs']['contract_file_sha256'],
            input_installation=str(a.input_installation),input_installation_sha256=a.input_installation_sha256,
            binaries={'baseline':before[str(a.baseline_binary)],'baseline_trace':before[str(a.baseline_trace_binary)],'research':before[str(a.binary)]},
            profiles_factory_sha256=sha(HERE/'PROFILES.py'),profile_context_sha256=sha(HERE/'profile_evidence/PUBLISHED_CONTEXT.json'),
            sample=sample,completed_diagonal_comparisons=len(diag),passed_trajectory_comparisons=sum(r.get('trajectory_parity',{}).get('passed') is True for r in diag),
            completed_exact_probes=len(exact),native_executions=sum(r['native_executions'] for r in results),
            completed_native_executions=sum(r['completed_native_executions'] for r in results),
            immutable_bindings_unchanged=unchanged,exact_probe_selection=controller.probes,observations=results)
        gate['io_protocol']={'historical_profile_controls_preserved_except_declared_ma27_backend':True,'all_four_research_cells_accepted_checkpoint':0,
            'accepted_sha256_trace':'enabled in all research cells and passive observers',
            'nonclock_observation_counter_allowlist':list(OBSERVATION_COUNTERS),
            'configuration_metadata_allowlist':['.'.join(k)for k in OBSERVATION_METADATA],
            'scope':'Only documented checkpoint lift counts are separately reported on the HH legacy diagonal; all numerical work and every trajectory hash remain exact.'}
        if not passed:gate['failure']='Incomplete or failed actual qualification; production remains blocked. No automatic resampling.'
    except Exception as error:gate['failure']=type(error).__name__+': '+str(error)
    gate['finished_utc']=utc();atomic(a.out/'GATE.json',gate)
    (a.out/'GATE.sha256').write_text(sha(a.out/'GATE.json')+'  GATE.json\n')
    atomic(a.out/'STATUS.json',dict(state='qualified' if gate['passed'] else 'blocked',passed=gate['passed'],timestamp=utc(),failure=gate['failure'],native_executions=gate.get('native_executions',0)))
    print(json.dumps({k:gate[k] for k in ('schema','passed','production_authorized','failure')}),flush=True)
    return 0 if gate['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
