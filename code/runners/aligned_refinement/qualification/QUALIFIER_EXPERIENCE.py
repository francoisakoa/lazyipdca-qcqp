#!/usr/bin/env python3
"""Fresh, outcome-blind native trajectory and intervention qualification.

This gate makes no claim before real native executions complete on the target.
The 50 original cases are fixed by the static study contract before execution.
A deterministic iteration-cap endpoint may qualify identity; a wall-time stop,
external censor, absent endpoint, incomplete trace or changed binding may not.
"""
from __future__ import annotations
import argparse, collections, concurrent.futures, datetime, fcntl, hashlib, importlib.util
import json, math, os, queue, re, signal, struct, subprocess, sys, threading, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
KIT=HERE.parent
CPUS=(23,)
BACKENDS=('ma27','mumps')
CELLS=('FROZEN_GR3','FROZEN_GHS','REFINED_GR3','REFINED_GHS')
STATE_KEYS=('accepted_index','state_sha256','outer_iteration','phase_n','phase_mi','phase_me','alpha','mu')
STATE_EVENTS={'startup_restoration_commit','state_recovery_commit','terminal_central_restart_commit','post_multiplier_update','central_restart_commit'}
DIRECTION_ORIGINS={'frozen_kkt_soc','analytic_soc','square_feasibility_normal','original_curvature_recovery','majorant_refined','majorant_primary'}
POSTPROCESSES={'none','source_curve','consistent_bound_curve','soc_replacement'}
HASH_RE=re.compile(r'[0-9a-f]{64}')
LEGACY_RUNS=700
INTERVENTION_RUNS=48
PLANNED_NATIVE_RUNS=LEGACY_RUNS+INTERVENTION_RUNS


def qualification_cpu_lock():
    path='/tmp/LAZYQCQP_CPU_23.lock'
    try:fd=os.open(path,os.O_RDWR|os.O_CREAT,0o666)
    except PermissionError:fd=os.open(path,os.O_RDONLY)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd);raise ValueError('CPU23 is already reserved by a native refinement controller')
    return os.fdopen(fd,'r')

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(),parse_int=lambda s:-0.0 if s=='-0' else int(s))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(threading.get_ident()))
    with tmp.open('w')as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def modules():
    sys.path.insert(0,str(KIT/'runtime_base'))
    runner=load_module('refinement_qualification_runner',KIT/'runtime_base/CAMPAGNE.py')
    auditor=load_module('refinement_qualification_auditor',KIT/'runtime_base/AUDIT_ORIGINAL.py')
    profiles=load_module('refinement_qualification_profiles',KIT/'runtime/PROFILES_EXPERIENCE.py')
    return runner,auditor,profiles

def exact_tree(v):
    if isinstance(v,float):
        if not math.isfinite(v):raise ValueError('Nonfinite numerical comparison value')
        return ['binary64',struct.pack('<d',v).hex()]
    if isinstance(v,dict):return {k:exact_tree(w)for k,w in sorted(v.items())}
    if isinstance(v,list):return [exact_tree(w)for w in v]
    return v

def digest(v):return hashlib.sha256(json.dumps(exact_tree(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def clock_key(k):return k.endswith('_seconds')or k in('seconds','cost_seconds','elapsed','wall_s','cpu_s')
def nonclock(v):
    if isinstance(v,dict):return {k:nonclock(w)for k,w in v.items()if not clock_key(k)}
    if isinstance(v,list):return [nonclock(w)for w in v]
    return v

def differences(a,b,p='',out=None):
    out=[]if out is None else out
    if len(out)>=50:return out
    if isinstance(a,dict)and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            path=p+'.'+k
            if k not in a or k not in b:out.append(path+': missing key')
            else:differences(a[k],b[k],path,out)
    elif isinstance(a,list)and isinstance(b,list):
        if len(a)!=len(b):out.append(p+': different length')
        else:
            for i,(x,y)in enumerate(zip(a,b)):differences(x,y,p+'['+str(i)+']',out)
    elif exact_tree(a)!=exact_tree(b):out.append(p+': different value')
    return out

def endpoint_parity(left,right):
    a,b=nonclock(left),nonclock(right)
    return dict(passed=digest(a)==digest(b),nonclock_sha256=[digest(a),digest(b)],differences=differences(a,b),ignored_fields='Only elapsed/timing fields; numerical work, statuses and vectors retained')

def censor_reasons(resources,endpoint):
    reasons=[]
    if any(resources.get(k)is True for k in('censored','timed_out','external_timeout','interrupted')):reasons.append('external_censor')
    process_seconds=resources.get('elapsed_seconds')
    if type(process_seconds)not in(int,float)or not math.isfinite(process_seconds)or not 0<=process_seconds<=7200:reasons.append('invalid_process_elapsed_seconds')
    code=resources.get('returncode')
    if type(code)is not int:reasons.append('returncode_unavailable')
    elif code<0:reasons.append('process_signal')
    status=str(endpoint.get('status','')).lower()
    if status in('time_limit','wall_time_limit','interrupted','external_timeout'):reasons.append('time_or_interruption_terminal')
    for key,bound in [('elapsed_seconds',7200),('outer_iterations',5000),('restoration_iterations',600)]:
        x=endpoint.get(key)
        if type(x)not in(int,float)or not math.isfinite(x)or not 0<=x<=bound:reasons.append('invalid_native_'+key)
    # max_outer_iterations/max_iterations is a complete deterministic endpoint,
    # never an external censor when its actual count is within the fixed cap.
    return sorted(set(reasons))

def check_command(command,backend,input_path=None,endpoint_path=None):
    values={'--linear-backend':backend,'--time-limit':'7200','--max-outer':'5000','--restoration-iters':'600','--shared-restoration-budget':'1','--native-al-fallback':'off'}
    for flag,value in values.items():
        if command.count(flag)!=1 or command.index(flag)+1>=len(command)or str(command[command.index(flag)+1])!=value:raise ValueError('Qualification command changed the declared budget/route: '+flag)
    if any(flag in command for flag in('--dca-endpoint-start','--endpoint-start','--warm-start-file')):raise ValueError('External endpoint start is forbidden')
    for flag,path in [('--problem',input_path),('--json',endpoint_path)]:
        if path is not None and(command.count(flag)!=1 or command.index(flag)+1>=len(command)or Path(command[command.index(flag)+1]).resolve()!=Path(path).resolve()):raise ValueError('Qualification command differs from its frozen input or endpoint binding: '+flag)
    return True

def policy_tuple(cell):
    if cell not in CELLS:raise ValueError('Unknown intervention cell')
    return ('frozen'if cell.startswith('FROZEN')else'adaptive','r3'if cell.endswith('GR3')else'hs')

def read_trace(path,endpoint_path,expect_exact=None,cell=None):
    rows=[]
    with Path(path).open()as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    if not rows or rows[0].get('event')!='header'or rows[0].get('schema')!='lazy-factorial-accepted-trace-v1':raise ValueError('Accepted-trace header absent')
    header=rows[0]
    if header.get('state_encoding')!='sha256-length-prefixed-binary64-le-x-s-z-lambda'or header.get('state_coordinates')!='native-phase-internal':raise ValueError('Unsupported committed-state encoding')
    if header.get('inequality_policy')not in('cache','hybrid')or header.get('globalization_policy')not in('r3','hs')or type(header.get('objective_exact'))is not bool:raise ValueError('Invalid trace policies')
    policies={k:header[k]for k in('inequality_policy','globalization_policy','objective_exact')}
    if expect_exact is not None and policies['objective_exact']is not expect_exact:raise ValueError('Wrong exact-objective mode')
    primary=cell is not None
    if primary:
        refinement,globalization=policy_tuple(cell)
        if policies!=dict(inequality_policy='hybrid',globalization_policy=globalization,objective_exact=True):raise ValueError('Wrong intervention policy')
        for k,v in [('refinement_policy',refinement),('common_guards','r3'),('common_representation','original_qcqp')]:
            if header.get(k)!=v:raise ValueError('Intervention header differs: '+k)
        if not HASH_RE.fullmatch(header.get('guard_config_sha256','')):raise ValueError('Common safeguard fingerprint absent')
    states=[];accepted=[];initial=[];index=0
    work_keys=('refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events')
    previous_work={k:0 for k in work_keys}
    for row in rows[1:]:
        event=row.get('event')
        if event=='metric_initialized':
            if primary and (any(row.get(k)!=v for k,v in policies.items())or row.get('refinement_policy')!=refinement or row.get('common_guards')!='r3'or row.get('common_representation')!='original_qcqp'or row.get('guard_config_sha256')!=header['guard_config_sha256']):raise ValueError('Initial metric event has different intervention policies')
            if not primary:raise ValueError('Research metric event in a legacy identity trace')
            if not HASH_RE.fullmatch(row.get('initial_inequality_model_sha256','')):raise ValueError('Initial metric fingerprint absent')
            if row.get('initialization')!='common_certified_hybrid':raise ValueError('Unequal correction initialization')
            phase=row.get('phase_metric_index')
            if type(phase)is not int or phase!=len(initial)+1:raise ValueError('Metric phase index is incomplete')
            initial.append(row);continue
        if event=='accepted':
            index+=1
            if row.get('accepted_index')!=index:raise ValueError('Accepted-state index is incomplete')
            keys=STATE_KEYS;accepted.append(row)
            if expect_exact is True and row.get('objective_E0_zero')is not True:raise ValueError('Objective correction appeared at an accepted commit')
            if row.get('direction_origin')not in DIRECTION_ORIGINS or row.get('postprocess')not in POSTPROCESSES:raise ValueError('Final direction provenance unavailable')
            final=row['direction_origin']if row['postprocess']=='none'else row['postprocess']
            if row.get('final_origin')!=final:raise ValueError('Final origin does not describe the committed direction')
        elif event in STATE_EVENTS:
            if row.get('after_accepted_index')!=index:raise ValueError('State event refers to the wrong commit')
            keys=('after_accepted_index','state_sha256','outer_iteration','phase_n','phase_mi','phase_me','mu')
        else:raise ValueError('Unsupported trace event: '+str(event))
        if any(row.get(k)!=v for k,v in policies.items()):raise ValueError('Trace policy changed after header')
        if not HASH_RE.fullmatch(row.get('state_sha256',''))or any(k not in row for k in keys):raise ValueError('Incomplete committed-state trace')
        if primary:
            if row.get('guard_config_sha256')!=header['guard_config_sha256']:raise ValueError('Guard policy changed within trace')
            if row.get('refinement_policy')!=refinement or row.get('common_guards')!='r3'or row.get('common_representation')!='original_qcqp':raise ValueError('Intervention policy changed within trace')
        if primary and event=='accepted':
            phase=row.get('phase_metric_index');initial_hash=row.get('initial_inequality_model_sha256');current_hash=row.get('current_inequality_model_sha256')
            if type(phase)is not int or not 1<=phase<=len(initial):raise ValueError('Accepted metric phase is unavailable')
            expected=initial[phase-1]['initial_inequality_model_sha256']if phase else''
            if initial_hash!=expected or (phase and not HASH_RE.fullmatch(current_hash or'')):raise ValueError('Accepted metric fingerprint is inconsistent')
            if phase and any(row.get(k)!=initial[phase-1].get(k)for k in('phase_n','phase_mi','phase_me')):raise ValueError('Metric dimensions differ from committed phase')
            if refinement=='frozen'and current_hash!=expected:raise ValueError('Frozen correction changed at an actual commit')
            for k in('refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events'):
                if type(row.get(k))is not int or row[k]<previous_work[k] or(refinement=='frozen'and row[k]!=0):raise ValueError('Invalid or decreasing committed refinement counter')
                previous_work[k]=row[k]
            if row['accepted_refinements']!=row['refinement_update_events']:raise ValueError('Committed metric-update counters disagree')
        states.append(dict(event=event,**{k:row[k]for k in keys}))
    side=read(str(endpoint_path)+'.factorial.json')
    if side.get('schema')!='lazy-factorial-endpoint-v1'or side.get('trace_success')is not True:raise ValueError('Committed trace did not finalize')
    if any(side.get(k)!=v for k,v in policies.items()):raise ValueError('Endpoint sidecar differs from trace policy')
    if side.get('accepted_events')!=len(accepted)or side.get('endpoint_accepted_steps')!=len(accepted):raise ValueError('Committed trace count differs from actual endpoint counter')
    if side.get('budget_verified')is not True:raise ValueError('Shared native-budget proof missing')
    for k,b in [('outer_iterations',5000),('restoration_iterations',600)]:
        if type(side.get(k))is not int or not 0<=side[k]<=b:raise ValueError('Actual native iteration budget invalid')
    secs=side.get('seconds')
    if type(secs)not in(int,float)or not math.isfinite(secs)or not 0<=secs<=7200:raise ValueError('Actual native time budget invalid')
    if primary:
        if side.get('refinement_policy')!=refinement or side.get('common_guards')!='r3'or side.get('common_representation')!='original_qcqp'or side.get('guard_config_sha256')!=header['guard_config_sha256']:raise ValueError('Intervention endpoint fingerprint differs')
        if side.get('initial_model_events')!=len(initial):raise ValueError('Initial-metric event count differs')
        first=initial[0]['initial_inequality_model_sha256']if initial else''
        if side.get('initial_inequality_model_sha256')!=first:raise ValueError('First correction initialization differs')
        sequence=hashlib.sha256()
        for entry in initial:
            payload=entry['initial_inequality_model_sha256'].encode('utf8');sequence.update(struct.pack('<Q',len(payload)));sequence.update(payload)
        if side.get('initial_hybrid_models_sha256')!=sequence.hexdigest():raise ValueError('Initial metric sequence digest differs')
        if side.get('accepted_refinements')!=side.get('refinement_update_events'):raise ValueError('Accepted metric-update counters disagree')
        if not initial:
            if side.get('native_solver_completed')is not True or side.get('initial_model_state')!='not_reached' or side.get('initialization_calls')!=0 or side.get('native_pre_model_return')is not True or side.get('primary_newton_backend_used')is not False or side.get('primary_backend_factor_attempts')!=0 or accepted:raise ValueError('Pre-model completion is not proven')
            if any(side[k]!=0 for k in('refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events')):raise ValueError('Unreached model performed refinement work')
        for k in ('refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events'):
            if type(side.get(k))is not int or side[k]<previous_work[k]:raise ValueError('Adaptive-work counter absent or less than a committed value: '+k)
        if refinement=='frozen':
            if any(side[k]!=0 for k in ('refinement_attempt_events','accepted_refinements','refinement_update_events','refinement_rollback_events'))or side.get('frozen_metric_invariant_verified')is not True:raise ValueError('Frozen correction changed or performed refinement work')
    return dict(header=header,sidecar=side,accepted_events=len(accepted),states=states,state_sequence_sha256=digest(states),initial_events=initial,trace_sha256=sha(path),model_reached=bool(initial),objective_zero_at_all_commits=all(x.get('objective_E0_zero')is True for x in accepted),direction_origins=dict(collections.Counter(x['final_origin']for x in accepted)))

def trace_parity(left,right):
    return dict(passed=left['state_sequence_sha256']==right['state_sequence_sha256'],accepted_events=[left['accepted_events'],right['accepted_events']],state_sequence_sha256=[left['state_sequence_sha256'],right['state_sequence_sha256']],differences=differences(left['states'],right['states']))

def legacy_roles(backend):
    if backend not in BACKENDS:raise ValueError('Unknown legacy backend')
    return ('installed_reference','reference','observer','experiment')if backend=='ma27'else('reference','observer','experiment')

def legacy_comparisons(data,observed,audits,backend):
    roles=legacy_roles(backend)
    if set(data)!=set(roles)or set(audits)!=set(roles)or set(observed)!={'observer','experiment'}:raise ValueError('Required installed/fresh legacy observation is absent')
    checks=dict(reference_observer=endpoint_parity(data['reference'],data['observer']),observer_experiment=endpoint_parity(data['observer'],data['experiment']),actual_trajectory=trace_parity(observed['observer'],observed['experiment']))
    if backend=='ma27':checks['installed_reference']=endpoint_parity(data['installed_reference'],data['reference'])
    original=[dict(strict=audits[r]['original_KKT_endpoint_strict'],intervals=audits[r]['intervals'])for r in roles]
    checks['original_certification']=dict(passed=all(digest(x)==digest(original[0])for x in original),strict_classifications=[audits[r]['original_KKT_endpoint_strict']for r in roles],roles=list(roles))
    return checks

def verify_qualification_spec(contract):
    rule=contract.get('parity_gate',{})
    expected=dict(legacy_runs=LEGACY_RUNS,synthetic_controls=6,synthetic_runs=INTERVENTION_RUNS,total_runs=PLANNED_NATIVE_RUNS,installed_ma27_reference_runs=100,installed_ma27_reference_required=True)
    if any(rule.get(k)!=v for k,v in expected.items())or rule.get('legacy_runs_by_backend')!=dict(ma27=400,mumps=300):raise ValueError('Static qualification contract does not declare all748 required native observations')
    if any(contract.get(section,{}).get('cpus')!=[23]or contract.get(section,{}).get('jobs')!=1 for section in('qualification','production')):raise ValueError('Native refinement qualification and production must use CPU23 and one slot')
    return True

def sample_cases(contract,cases):
    lookup={c['case_id']:c for c in cases}
    if len(lookup)!=1834 or collections.Counter(c['regime']for c in cases)!=collections.Counter(dict(I=1037,M=797)):raise ValueError('Complete I/M corpus required')
    rule=contract.get('parity_gate',contract.get('qualification',{}))
    values=rule.get('sample',rule.get('selected_case_ids',[]))
    ids=[x['case_id']if isinstance(x,dict)else x for x in values]
    if len(ids)!=50 or len(set(ids))!=50 or any(x not in lookup for x in ids):raise ValueError('Static study contract must identify the same fifty unique frozen cases')
    if rule.get('selection_uses_outcomes')is not False:raise ValueError('Qualification selection must be outcome blind')
    selected=[lookup[x]for x in ids]
    if collections.Counter(c['regime']for c in selected)!=collections.Counter(dict(I=25,M=25)):raise ValueError('Fixed qualification sample must contain25I and25M cases')
    bindings=rule.get('sample_bindings',[])
    if bindings:
        b={c['case_id']:c for c in bindings}
        if set(b)!=set(ids):raise ValueError('Qualification sample binding identities differ')
        for c in selected:
            for k,v in [('input_sha256',c['input_sha256']),('start_sha256',start_sha(c))]:
                if b[c['case_id']].get(k)!=v:raise ValueError('Qualification sample has a different '+k)
    return selected

def start_sha(case):return case.get('start_sha256')or case.get('start_binding',{}).get('start_sha256')
def bound_input(case,preparation):
    cid=case['case_id'];binding=preparation.get('frozen_bindings',{}).get(cid)
    if case.get('start_binding',{}).get('external_start_required'):raise ValueError('External start is not admitted by this raw-start gate')
    if not binding or not binding.get('input'):raise ValueError('Explicit resolved frozen input absent: '+cid)
    p=Path(binding['input']).expanduser().resolve(strict=True)
    if sha(p)!=case['input_sha256']or binding.get('input_sha256')!=case['input_sha256']:raise ValueError('Frozen coefficients differ: '+cid)
    if binding.get('start_sha256')!=start_sha(case):raise ValueError('Resolved frozen start differs: '+cid)
    with p.open('rb')as f:line=next((x for x in f if x.startswith(b'x0 ')),None)
    expected=case.get('start_binding',{}).get('raw_x0_line_sha256')or binding.get('raw_x0_line_sha256')
    if line is None or not expected or hashlib.sha256(line).hexdigest()!=expected:raise ValueError('Frozen raw x0 line differs: '+cid)
    return p

def verify_build(path,preparation):
    proof=read(path)
    if proof.get('schema')!='lazy-refinement-build-v1'or proof.get('homogeneous_compilation')is not True or proof.get('original_installation_unchanged')is not True:raise ValueError('Fresh homogeneous native build proof absent')
    if preparation.get('build_manifest_sha256')!=sha(path):raise ValueError('Prepared native build binding differs')
    if set(proof.get('backends',{}))!=set(BACKENDS):raise ValueError('Both native backends must be declared')
    for key in ('reference_source_sha256','observer_source_sha256','experiment_source_sha256'):
        if not isinstance(proof.get(key),dict)or not proof[key]or any(not HASH_RE.fullmatch(v)for v in proof[key].values()):raise ValueError('Native source map missing: '+key)
    source_files={}
    for folder,key in [('reference_source','reference_source_sha256'),('experiment_source','experiment_source_sha256')]:
        tree=KIT/'native'/folder
        actual={str(f.relative_to(tree)):sha(f)for f in sorted(tree.rglob('*'))if f.is_file()}
        if actual!=proof[key]:raise ValueError('Delivered source differs from the compiled native snapshot: '+folder)
        source_files.update({str(tree/relative):value for relative,value in actual.items()})
    binaries={};all_files=source_files
    for b in BACKENDS:
        rec=proof['backends'][b]
        for role in legacy_roles(b):
            p=Path(rec.get(role+'_binary','')).resolve(strict=True)
            if not p.is_file()or not os.access(p,os.X_OK)or p.open('rb').read(4)!=b'\x7fELF':raise ValueError('Actual native ELF executable required: '+role)
            if sha(p)!=rec.get(role+'_binary_sha256'):raise ValueError('Native executable binding changed: '+role)
            all_files[str(p)]=rec[role+'_binary_sha256'];binaries[b,role]=p
        if not isinstance(rec.get('environment'),dict):raise ValueError('Pinned backend runtime environment absent')
    # The two backend blocks select one of the same three built executables;
    # this experiment does not compare distinct numerical source versions.
    for role in ('reference','observer','experiment'):
        if proof['backends']['ma27'][role+'_binary_sha256']!=proof['backends']['mumps'][role+'_binary_sha256']:raise ValueError('Backends do not share the same '+role+' executable')
    for p in {p for (b,r),p in binaries.items()if r!='installed_reference'}:
        cap=subprocess.check_output([str(p),'--linear-backends'],text=True,timeout=15).strip()
        if any(not re.search(r'(^|[^A-Za-z0-9_])'+b+r'($|[^A-Za-z0-9_])',cap)for b in BACKENDS):raise ValueError('Native executable does not expose both requested backends')
    cap=subprocess.check_output([str(binaries['ma27','installed_reference']),'--linear-backends'],text=True,timeout=15).strip()
    if not re.search(r'(^|[^A-Za-z0-9_])ma27($|[^A-Za-z0-9_])',cap):raise ValueError('Installed campaign reference does not expose MA27')
    return proof,binaries,all_files

def fixtures(out):
    """Six fixed, generic QCQPs; no corpus-result-dependent fixture choice."""
    out.mkdir(parents=True,exist_ok=False)
    def quad(name,c,lin,h):return ['quadratic','name '+name,'constant '+str(c),'linear '+' '.join(map(str,lin)),'hessian_nnz '+str(len(h))]+[f'{i} {j} {v}'for i,j,v in h]+['end_quadratic']
    def model(k,n,x,lb,ub,f,eq,iq):
        lines=['LAZYIPDCA_QCQP_V1','name structural_control_'+str(k),'n '+str(n),'x0 '+' '.join(map(str,x)),'lb '+' '.join(map(str,lb)),'ub '+' '.join(map(str,ub)),'objective']+quad('f',*f)+['equalities '+str(len(eq))]
        for i,q in enumerate(eq):lines+=quad('h'+str(i),*q)
        lines+=['inequalities '+str(len(iq))]
        for i,q in enumerate(iq):lines+=quad('g'+str(i),*q)
        lines+=['end_problem'];p=out/f'control_{k:02d}.qcq';p.write_text('\n'.join(lines)+'\n');return p
    paths=[]
    paths.append(model(1,2,[.3,.4],['-inf']*2,['inf']*2,(0,[-1,-2],[(0,0,2),(1,1,2)]),[],[(-4,[0,0],[(0,0,2),(1,1,2)])]))
    for k,beta in [(2,.1),(3,1.)]:paths.append(model(k,2,[2,1],[0,-4],[8,4],(0,[1,.1],[]),[],[(1,[0,0],[(0,0,-2),(1,1,2*beta)])]))
    paths.append(model(4,3,[.5,.5,.7071067811865476],[-2]*3,[2]*3,(0,[-.2,0,0],[(0,0,-2),(1,1,4),(2,2,1)]),[(-1,[0,0,0],[(0,0,2),(1,1,2),(2,2,2)])],[(-.3,[0,0,0],[(0,1,1)]),(-2,[0,0,0],[(0,0,2),(1,1,2),(2,2,2)])]))
    paths.append(model(5,3,[2,1,3],[0,-4,-8],[8,4,12],(0,[1,.1,.03],[]),[(0,[-1,-1,1],[])],[(1,[0,0,0],[(0,0,-2),(1,1,.5)])]))
    paths.append(model(6,4,[.5,.8660254037844386,2,1],[-2,-2,0,-4],[2,2,8,4],(0,[-.2,0,1,.1],[(1,1,.4)]),[(-1,[0,0,0,0],[(0,0,2),(1,1,2)])],[(1,[0,0,0,0],[(2,2,-2),(3,3,.2)])]))
    return [dict(case_id=f'structural_control_{i+1:02d}',dataset='native_intervention_controls',regime='M'if i in(3,5)else'I',input=str(p),input_sha256=sha(p))for i,p in enumerate(paths)]

def summarize_trace(obs):return {k:v for k,v in obs.items()if k not in('states','initial_events')}
def intervention_comparison(records):
    if len(records)!=48 or any(r.get('passed')is not True for r in records):return dict(passed=False,reason='Forty-eight complete native intervention observations required')
    idx={(r['case_id'],r['backend'],r['cell']):r for r in records};checks=[]
    for cid in sorted({r['case_id']for r in records}):
        for b in BACKENDS:
            rr=[idx[cid,b,c]for c in CELLS]
            side=[r['trace']['sidecar']for r in rr]
            guards={r['guard_config_sha256']for r in side}
            if len(guards)!=1:checks.append(dict(case_id=cid,backend=b,check='common_guards',passed=False))
            for g in ('GR3','GHS'):
                a=idx[cid,b,'FROZEN_'+g]['trace']['sidecar'];z=idx[cid,b,'REFINED_'+g]['trace']['sidecar']
                checks.append(dict(case_id=cid,backend=b,globalization=g,check='shared_initial_correction',first_metric_sha256_frozen=a['initial_inequality_model_sha256'],first_metric_sha256_adaptive=z['initial_inequality_model_sha256'],model_reached=bool(a['initial_inequality_model_sha256']),passed=a['initial_inequality_model_sha256']==z['initial_inequality_model_sha256']))
    guard_map={}
    for b in BACKENDS:
        values={r['trace']['sidecar']['guard_config_sha256']for r in records if r['backend']==b}
        if len(values)!=1:return dict(passed=False,reason='Safeguard fingerprint varies between structural controls',backend=b)
        guard_map[b]=next(iter(values))
    active=[]
    for b in BACKENDS:
        for g in ('GR3','GHS'):
            rr=[r for r in records if r['backend']==b and r['cell']=='REFINED_'+g]
            attempts=sum(r['trace']['sidecar']['refinement_attempt_events']for r in rr)
            accepted=sum(r['trace']['sidecar']['accepted_refinements']for r in rr)
            meaningful=sum(r['trace']['model_reached']and r['trace']['accepted_events']>0 for r in rr)
            active.append(dict(backend=b,globalization=g,actual_refinement_attempts=attempts,accepted_refinements=accepted,meaningful_model_controls=meaningful,passed=attempts>0 and meaningful>0))
    return dict(passed=all(r['passed']for r in checks+active),guard_config_sha256_by_backend=guard_map,common_guard_vector_sha256=digest(guard_map),pair_checks=checks,realized_adaptation=active,accepted_refinement_is_not_a_success_gate=True,scope='Mechanism qualification on six fixed structural controls; benefit is decided only by preregistered full-corpus outcomes')

class Qualification:
    def __init__(self,args,prep,contract,build,binaries,sample,paths,runner,auditor,profiles):
        self.a=args;self.prep=prep;self.contract=contract;self.build=build;self.binaries=binaries;self.sample=sample;self.paths=paths;self.runner=runner;self.auditor=auditor;self.profiles=profiles
        self.stop=threading.Event();self.monitor_stop=threading.Event();self.lock=threading.RLock();self.todo=queue.Queue();self.results=[];self.active={};self.native_started=0;self.native_done=0;self.last_print=0
        for c in sample:
            for b in BACKENDS:
                for method in('R3','HH'):self.todo.put(('legacy',c,b,method))
        for c in fixtures(args.out/'fixtures'):
            for b in BACKENDS:
                for cell in CELLS:self.todo.put(('intervention',c,b,cell))
        self.planned_groups=self.todo.qsize()
    def status(self,state='running'):
        with self.lock:
            rec=dict(schema='refinement-qualification-status-v1',timestamp=utc(),state=state,cpu_affinity=list(CPUS),jobs=1,planned_groups=self.planned_groups,done=len(self.results),native_started=self.native_started,native_completed=self.native_done,planned_native_runs=PLANNED_NATIVE_RUNS,active=dict(self.active),remaining=self.todo.qsize())
            atomic(self.a.out/'STATUS.json',rec)
            if time.monotonic()-self.last_print>=10 or state!='running':
                print('QUALIFICATION — groupes '+str(len(self.results))+'/'+str(self.planned_groups)+' ; résolutions '+str(self.native_done)+'/'+str(PLANNED_NATIVE_RUNS)+' ; créneaux '+str(len(self.active))+'/'+str(len(CPUS)),flush=True);self.last_print=time.monotonic()
    def monitor(self):
        while not self.monitor_stop.wait(10):self.status()
    def execute(self,command,folder,cpu,case,backend,environment,legacy=False):
        folder.mkdir(parents=True,exist_ok=False);atomic(folder/'COMMAND.json',dict(command=command,command_sha256=digest(command)))
        check_command(command,backend,self.paths.get(case['case_id'],case.get('input')),folder/'endpoint.json')
        with self.lock:self.native_started+=1
        def started(pid,ticks):
            with self.lock:
                self.active[str(cpu)].update(pid=pid,pid_start_ticks=ticks,folder=str(folder),phase='solver')
        resources=self.runner.run_process(command,folder,folder/'solver.log',7200,self.stop,cpu=cpu,on_start=started,env_extra=environment)
        with self.lock:self.native_done+=1;self.active[str(cpu)].update(phase='audit',pid=None)
        atomic(folder/'solver_resources.json',resources)
        path=folder/'endpoint.json'
        if not path.is_file():raise ValueError('Native endpoint absent')
        endpoint=read(path);reasons=censor_reasons(resources,endpoint)
        if reasons:raise ValueError('Qualification observation incomplete: '+','.join(reasons))
        actual=str(endpoint.get('linear_solver_backend','')).lower()
        no_primary=legacy and endpoint.get('lazy_newton_factor_attempts')==0 and endpoint.get('lazy_newton_accepted_steps')==0
        if backend not in actual and not no_primary:raise ValueError('Endpoint backend differs from declared native backend: '+actual)
        if 'ipopt'in actual:raise ValueError('External optimizer route in a native qualification')
        if endpoint.get('dca_filter_pivoted_backend')or endpoint.get('lazy_newton_native_factor_successes',0):raise ValueError('Undeclared primary factorization fallback')
        al=endpoint.get('native_al_fallback',{})
        if isinstance(al,dict)and(al.get('enabled')is True or al.get('active')is True):raise ValueError('AL phase was enabled or activated')
        audit=self.auditor.audit_files(self.paths.get(case['case_id'],case.get('input')),path,resources,case)
        atomic(folder/'AUDIT.json',audit)
        if audit.get('audit_available')is not True or type(audit.get('original_KKT_endpoint_strict'))is not bool:raise ValueError('Independent original endpoint qualification unavailable')
        return endpoint,resources,path,audit
    def unit(self,kind,case,backend,method,cpu):
        cid=case['case_id'];folder=self.a.out/'tasks'/(self.runner.safe_id(cid)+'__'+backend+'__'+method)
        result=dict(kind=kind,case_id=cid,regime=case['regime'],dataset=case['dataset'],backend=backend,method=method,passed=False,failure=None,input_sha256=case['input_sha256'])
        environment=self.build['backends'][backend]['environment'];hsl=environment.get('HSL_LIBRARY')
        inp=self.paths.get(cid,Path(case.get('input','')))
        try:
            if sha(inp)!=case['input_sha256']:raise ValueError('Input changed during qualification')
            if kind=='legacy':
                data={};observed={};audits={};endpoints={}
                for role in legacy_roles(backend):
                    d=folder/role;p=d/'endpoint.json';tr=d/'trace.jsonl'if role in('observer','experiment')else None
                    cmd=self.profiles.make_legacy_command(method,backend,self.binaries[backend,role],inp,p,trace=tr,hsl_library=hsl)
                    endpoint,res,path,audit=self.execute(cmd,d,cpu,case,backend,environment,legacy=True);data[role]=endpoint;audits[role]=audit;endpoints[role]=dict(endpoint_sha256=sha(path),resources=res,audit_sha256=sha(d/'AUDIT.json'))
                    if tr is not None:observed[role]=read_trace(tr,path,method=='R3')
                checks=legacy_comparisons(data,observed,audits,backend)
                result.update(passed=all(c['passed']for c in checks.values()),comparisons=checks,endpoints=endpoints,traces={r:summarize_trace(o)for r,o in observed.items()})
                if not result['passed']:result['failure']='Actual legacy vectors, work, original audit or committed trajectories differ'
            else:
                result['cell']=method;d=folder/'experiment';p=d/'endpoint.json';tr=d/'trace.jsonl'
                cmd=self.profiles.make_command(method+'_'+backend,self.binaries[backend,'experiment'],inp,p,tr,hsl_library=hsl)
                endpoint,res,path,audit=self.execute(cmd,d,cpu,case,backend,environment);obs=read_trace(tr,path,True,method)
                if not obs['model_reached']or obs['accepted_events']<1:raise ValueError('Structural intervention control must build its shared hybrid model and commit a native update')
                side=obs['sidecar'];requested='ma27'if backend=='ma27'else'mumps_metis'
                if side.get('primary_newton_backend_used')is not True or type(side.get('primary_backend_factor_attempts'))is not int or side['primary_backend_factor_attempts']<1:raise ValueError('No real primary native factorization in the intervention control')
                if side.get('requested_linear_backend')!=requested or side.get('actual_linear_backend')!=endpoint.get('linear_solver_backend'):raise ValueError('Intervention backend sidecar differs from actual endpoint')
                result.update(passed=True,model_reached=obs['model_reached'],trace=summarize_trace(obs),endpoint_sha256=sha(path),resources=res,original_audit_strict=audit['original_KKT_endpoint_strict'],original_audit_sha256=sha(d/'AUDIT.json'))
        except Exception as e:result['failure']=type(e).__name__+': '+str(e)
        atomic(folder/'COMPARISON.json',result)
        return result
    def worker(self,cpu):
        os.sched_setaffinity(0,{cpu})  # The audit and its native child share one fixed slot.
        while not self.stop.is_set():
            try:kind,case,backend,method=self.todo.get_nowait()
            except queue.Empty:return
            with self.lock:self.active[str(cpu)]=dict(case_id=case['case_id'],backend=backend,method=method,kind=kind,phase='preflight',started_utc=utc())
            self.status();result=self.unit(kind,case,backend,method,cpu)
            with self.lock:self.results.append(result);self.active.pop(str(cpu),None)
            self.todo.task_done();self.status()
    def run(self):
        old={s:signal.getsignal(s)for s in(signal.SIGINT,signal.SIGTERM)}
        for s in old:signal.signal(s,lambda *_:self.stop.set())
        monitor=threading.Thread(target=self.monitor,daemon=True);monitor.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(CPUS))as pool:
                for f in concurrent.futures.as_completed([pool.submit(self.worker,c)for c in CPUS]):f.result()
        finally:
            self.stop.set();self.monitor_stop.set();monitor.join()
            for s,h in old.items():signal.signal(s,h)
        return self.results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparation',required=True,type=Path);p.add_argument('--build',type=Path);p.add_argument('--study-contract',type=Path);p.add_argument('--registration',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--cpus',default='23');p.add_argument('--jobs',type=int,default=1);a=p.parse_args(argv)
    a.out=a.out.expanduser().resolve()
    if a.out.exists():raise ValueError('Qualification output must be fresh; no old result or sample is overwritten')
    a.out.mkdir(parents=True)
    gate=dict(schema='lazy-refinement-qualification50-v1',created_utc=utc(),cpu_affinity=list(CPUS),jobs=1,passed=False,production_authorized=False,cases=50,reference_reproduction_passed=False,installed_ma27_reference_passed=False,installed_ma27_reference_runs=0,study_intervention_verified=False,required_legacy_comparisons=200,required_legacy_runs=LEGACY_RUNS,required_intervention_runs=INTERVENTION_RUNS,planned_native_runs=PLANNED_NATIVE_RUNS,completed_native_runs=0,qualification_script_sha256=sha(Path(__file__)),failure=None,observations=[])
    controller=None
    try:
        if a.jobs!=1 or a.cpus!='23':raise ValueError('Qualification uses exactly one native slot on CPU23')
        if not set(CPUS)<=set(os.sched_getaffinity(0)):raise ValueError('Requested CPU23 is unavailable in this session')
        os.sched_setaffinity(0,set(CPUS))  # Spawned workers inherit the declared machine envelope.
        a.preparation=a.preparation.expanduser().resolve(strict=True);prep=read(a.preparation)
        if prep.get('schema')!='lazy-refinement-preparation-v1'or prep.get('production_authorized')is not False:raise ValueError('A fresh, nonpromoted experiment preparation is required')
        a.build=(a.build or Path(prep['build_manifest'])).expanduser().resolve(strict=True);a.study_contract=(a.study_contract or Path(prep['study_contract'])).expanduser().resolve(strict=True);a.registration=(a.registration or a.study_contract).expanduser().resolve(strict=True)
        if prep.get('study_contract_sha256')!=sha(a.study_contract):raise ValueError('Static intervention contract changed after preparation')
        contract=read(a.study_contract);verify_qualification_spec(contract);sample=sample_cases(contract,prep['cases'])
        paths={c['case_id']:bound_input(c,prep)for c in sample};build,binaries,bound_files=verify_build(a.build,prep)
        gate.update(preparation_sha256=sha(a.preparation),build_manifest_sha256=sha(a.build),study_contract_sha256=sha(a.study_contract),registration_sha256=sha(a.registration),registration_role='Static draft scientific protocol for qualification; final production registration is sealed afterward',sample_case_ids=[c['case_id']for c in sample],selection_uses_outcomes=False,sample_regimes=dict(collections.Counter(c['regime']for c in sample)),installed_ma27_reference_binary=str(binaries['ma27','installed_reference']),installed_ma27_reference_binary_sha256=build['backends']['ma27']['installed_reference_binary_sha256'])
        sample_file=a.out/'SAMPLE50.json';atomic(sample_file,dict(selection='Exactly the static contract sample; no resampling',selection_uses_outcomes=False,cases=sample));gate['sample_sha256']=sha(sample_file)
        for path in (Path(__file__).resolve(),a.preparation,a.build,a.study_contract,a.registration,KIT/'runtime/PROFILES_EXPERIENCE.py',KIT/'runtime_base/AUDIT_ORIGINAL.py',KIT/'runtime_base/frozen_qcqp_reader.py',KIT/'runtime_base/CAMPAGNE.py'):bound_files[str(path)]=sha(path)
        for cid,path in paths.items():bound_files[str(path)]=sha(path)
        runner,auditor,profiles=modules();controller=Qualification(a,prep,contract,build,binaries,sample,paths,runner,auditor,profiles)
        with qualification_cpu_lock():results=controller.run()
        legacy=[r for r in results if r['kind']=='legacy'];studies=[r for r in results if r['kind']=='intervention'];intervention=intervention_comparison(studies)
        unchanged=all(sha(Path(path))==h for path,h in bound_files.items())
        installed=[r for r in legacy if r['backend']=='ma27'and 'installed_reference'in r.get('endpoints',{})]
        installed_passed=len(installed)==100 and all(r.get('comparisons',{}).get('installed_reference',{}).get('passed')is True and r.get('comparisons',{}).get('original_certification',{}).get('passed')is True for r in installed)
        gate.update(observations=results,completed_native_runs=controller.native_done,reference_reproduction_passed=len(legacy)==200 and all(r['passed']for r in legacy),installed_ma27_reference_passed=installed_passed,installed_ma27_reference_runs=len(installed),study_intervention_verified=intervention['passed'],intervention_checks=intervention,bound_files_unchanged=unchanged,all_bindings_sha256=bound_files,legacy_identity_observations=len(legacy),intervention_observations=len(studies))
        gate['passed']=gate['reference_reproduction_passed']and installed_passed and gate['study_intervention_verified']and unchanged and controller.native_done==PLANNED_NATIVE_RUNS
        gate['production_authorized']=gate['passed']
        gate['guard_config_sha256_by_backend']=intervention.get('guard_config_sha256_by_backend',{})
        gate['common_guard_vector_sha256']=intervention.get('common_guard_vector_sha256')
        gate['first_metric_pair_digest_proof']=intervention.get('pair_checks',[])
        gate['direction_trace_schema_sha256']=sha(KIT/'native/experiment_source/src/factorial_trace.hpp')
        gate['frozen_metric_invariant_verified']=len(studies)==48 and all(r.get('passed')is True and (not r['cell'].startswith('FROZEN')or r['trace']['sidecar'].get('frozen_metric_invariant_verified')is True)for r in studies)
        if not gate['passed']:gate['failure']='Qualification incomplete or a required trajectory/intervention/binding check failed'
    except Exception as e:gate['failure']=type(e).__name__+': '+str(e)
    finally:
        gate['finished_utc']=utc();atomic(a.out/'GATE50.json',gate);(a.out/'GATE50.sha256').write_text(sha(a.out/'GATE50.json')+'  GATE50.json\n')
        if controller:controller.status('completed'if gate['passed']else'blocked')
    print('QUALIFICATION50 '+('VALIDÉE'if gate['passed']else'BLOQUÉE')+' — '+(gate['failure']or'700 exécutions de fidélité et 48 contrôles réels vérifiés'),flush=True)
    return 0 if gate['passed']else 2

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError)as e:print('QUALIFICATION REFUSÉE — '+str(e),file=sys.stderr);raise SystemExit(2)
