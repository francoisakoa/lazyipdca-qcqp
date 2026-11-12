#!/usr/bin/env python3
"""Four preregistered Ipopt comparisons; no Lazy profile is changed."""
from pathlib import Path
import hashlib,json,math

CELLS={f'ipopt_{backend}_{strategy}_aligned':dict(backend=backend,strategy=strategy)
       for strategy in ('monotone','adaptive') for backend in ('ma27','mumps')}
REGIMES={'I':1037,'M':797,'E':1704,'Q':540}
JOBS=22;CPUS=tuple(range(JOBS));SECONDS=7200;MAX_ITER=5000;TASKS=16312;SEED=20261109
ALIGNED_OPTIONS=dict(tol=1e-8,constr_viol_tol=1e-6,dual_inf_tol=1e-6,compl_inf_tol=1e-6,
    acceptable_iter=0,bound_relax_factor=0.,max_iter=MAX_ITER,max_cpu_time=float(SECONDS),
    max_wall_time=float(SECONDS),option_file_name='',print_user_options='yes')

def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def finite_number(value):return type(value)in(int,float)and math.isfinite(value)

def representation_for_input(path):
    """Read only the frozen V1/V2 header; mirror the exact infinity encoding."""
    with Path(path).open()as stream:
        if stream.readline().strip()not in('LAZYIPDCA_QCQP_V1','LAZYIPDCA_QCQP_V2'):raise ValueError('QCQP V1/V2 requis')
        if not stream.readline().startswith('name '):raise ValueError('Nom QCQP absent')
        nrow=stream.readline().split()
        if len(nrow)!=2 or nrow[0]!='n':raise ValueError('Dimension QCQP absente')
        n=int(nrow[1]);start=stream.readline().split()
        if not start or start[0]!='x0'or len(start)!=n+1:raise ValueError('Départ QCQP incomplet')
        lower=stream.readline().split();upper=stream.readline().split()
        if not lower or lower[0]!='lb'or not upper or upper[0]!='ub'or len(lower)!=n+1 or len(upper)!=n+1:raise ValueError('Bornes QCQP incomplètes')
        bounds=list(map(float,lower[1:]+upper[1:]))
        if n<1 or any(math.isnan(x)for x in bounds):raise ValueError('Bornes QCQP invalides')
    largest=max((abs(x)for x in bounds if math.isfinite(x)),default=0.)
    infinity=max(2e19,2*largest)
    if not math.isfinite(infinity):raise ValueError('Borne finie trop grande pour la représentation Ipopt exacte')
    overrides=[];effective=dict(nlp_lower_bound_inf=-1e19,nlp_upper_bound_inf=1e19)
    if largest>=1e19:
        effective=dict(nlp_lower_bound_inf=-infinity,nlp_upper_bound_inf=infinity)
        overrides=[dict(reason='preserve finite original bounds beyond the library default infinity cutoff',**effective)]
    return dict(input_representation_overrides=overrides,effective_bound_infinity=effective)

def process_budget_verified(resources):
    if not isinstance(resources,dict):return False
    return (all(finite_number(resources.get(k))and resources[k]>=0 for k in ('elapsed_seconds','user_cpu_seconds','system_cpu_seconds'))
        and resources['elapsed_seconds']<=SECONDS and resources['user_cpu_seconds']+resources['system_cpu_seconds']<=SECONDS
        and type(resources.get('returncode'))is int and resources['returncode']>=0
        and resources.get('censored')is False and resources.get('timed_out')is not True and resources.get('interrupted')is not True)

def verify_endpoint(endpoint,label,ipopt_version,hsl_library=None,representation=None):
    if label not in CELLS:raise ValueError('Cellule Ipopt inconnue')
    cell=CELLS[label]
    for key,value in dict(adapter_version='ipopt-aligned-original-qcqp-v1',reference_policy='aligned_original_kkt_tolerances_v1',aligned_contract_verified=True,option_file_disabled=True,solver='Ipopt',mode='direct-frozen-qcqp',available=True,ipopt_version=ipopt_version,linear_solver=cell['backend'],mu_strategy=cell['strategy'],optimization_wall_limit_seconds=SECONDS,max_iterations=MAX_ITER,requested_mumps_ordering=0).items():
        if endpoint.get(key)!=value:raise ValueError('Preuve Ipopt absente ou différente : '+key)
    options=endpoint.get('effective_options')
    if not isinstance(options,dict):raise ValueError('Options effectives absentes')
    for key,value in dict(ALIGNED_OPTIONS,mu_strategy=cell['strategy'],linear_solver=cell['backend'],nlp_scaling_method='gradient-based',hessian_approximation='exact',fixed_variable_treatment='make_parameter').items():
        if options.get(key)!=value:raise ValueError('Option Ipopt non conforme : '+key)
    if cell['backend']=='ma27'and(not hsl_library or options.get('hsllib')!=str(hsl_library)):raise ValueError('Bibliothèque HSL effective différente de l’installation')
    if cell['backend']=='mumps'and options.get('mumps_pivot_order')!=7:raise ValueError('Ordre MUMPS effectif différent du défaut lié')
    expected=representation or dict(input_representation_overrides=[],effective_bound_infinity=dict(nlp_lower_bound_inf=-1e19,nlp_upper_bound_inf=1e19))
    if endpoint.get('input_representation_overrides')!=expected['input_representation_overrides']:raise ValueError('Déclaration de représentation différente du QCQP gelé')
    for key,value in expected['effective_bound_infinity'].items():
        if options.get(key)!=value:raise ValueError('Seuil d’infini effectif non conforme au QCQP gelé : '+key)
    seconds=endpoint.get('elapsed_seconds');iterations=endpoint.get('iterations')
    if not finite_number(seconds)or not 0<=seconds<=SECONDS:raise ValueError('Temps natif Ipopt non vérifié')
    if type(iterations)is not int or not 0<=iterations<=MAX_ITER:raise ValueError('Budget itérations Ipopt non vérifié')
    if type(endpoint.get('ipopt_application_status_code'))is not int:raise ValueError('Terminaison Ipopt non prouvée')
    # The independent original evaluator checks vector dimensions and values.
    for key in ('x','equality_multipliers','inequality_multipliers','lower_bound_multipliers','upper_bound_multipliers'):
        if not isinstance(endpoint.get(key),list)or any(not finite_number(v)for v in endpoint[key]):raise ValueError('Vecteur final absent ou non fini : '+key)
    return dict(aligned_options_verified=True,native_route_verified=True,backend_verified=True,budget_verified=True,
        ipopt_version=ipopt_version,mu_strategy=cell['strategy'],linear_solver=cell['backend'],
        effective_options_sha256=json_sha(options),native_elapsed_seconds=seconds,native_iterations=iterations,
        option_file_disabled=True,input_representation_overrides=expected['input_representation_overrides'],effective_bound_infinity=expected['effective_bound_infinity'])

def make_command(config,case,label,endpoint,input_path,external_start=None):
    c=CELLS[label]
    cmd=[config['binary'],'--problem',str(input_path),'--json',str(endpoint),'--mu-strategy',c['strategy'],
        '--linear-solver',c['backend'],'--max-iter',str(MAX_ITER),'--time-limit',str(SECONDS),'--print-level','5']
    if c['backend']=='ma27':cmd+=['--hsl-library',config['hsl_library']]
    if external_start is not None:cmd+=['--frozen-start',str(external_start)]
    return cmd
