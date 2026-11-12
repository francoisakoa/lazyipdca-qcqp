"""Qualified research cells within the recovered continuous mega controller.

The six historical method names remain separate from the four crossed cells.
No numerical result is inferred from an option or a previous campaign.
"""
from pathlib import Path
import hashlib,importlib.util,json,re

CELLS={'F_R3':'R3','F_X1':'X1','F_X2':'X2','F_H0':'H0'}
FACTORS={'R3':('cache','r3'),'X1':('cache','hs'),'X2':('hybrid','r3'),'H0':('hybrid','hs')}
CACHE={}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def module(path):
 s=importlib.util.spec_from_file_location('qualified_mega_factorial_factory',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def signature(path):
 st=Path(path).stat();return(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)

def validate(installation):
 path=Path(installation['factorial_installation']).resolve();expected=installation.get('factorial_installation_sha256')
 cache_key=(str(path),expected,installation.get('hsl_library_sha256'),installation.get('ipopt_defaults_installation_sha256'))
 cached=CACHE.get(cache_key)
 if cached is not None and all(signature(p)==sig for p,sig in cached['_dependency_signatures'].items()):return cached
 value=sha(path)
 if value!=installation.get('factorial_installation_sha256'):raise ValueError('Factorial installation changed')
 d=read(path)
 default_path=Path(installation['ipopt_defaults_installation'])
 if sha(default_path)!=installation.get('ipopt_defaults_installation_sha256'):raise ValueError('Default input installation changed')
 if d.get('default_installation_sha256')!=installation['ipopt_defaults_installation_sha256']:raise ValueError('Research and primary input installations differ')
 defaults=read(default_path)
 contract_sha=sha(d['contract'])
 if contract_sha!=defaults.get('contract_sha256'):raise ValueError('Research and primary frozen corpus differ')
 for key,hkey in [('binary','binary_sha256'),('hsl_library','hsl_library_sha256'),('qualified_50gate','qualified_50gate_sha256'),('preregistration','preregistration_sha256')]:
  if not d.get(key)or not d.get(hkey)or sha(d[key])!=d[hkey]:raise ValueError('Unverified factorial dependency: '+key)
 if d.get('native_backend')!='ma27':raise ValueError('Research cells must use their freshly qualified MA27 backend')
 if d['hsl_library_sha256']!=installation.get('hsl_library_sha256'):raise ValueError('Research and historical native HSL differ')
 gate=read(d['qualified_50gate'])
 for key in ('passed','production_authorized','coupling_separation_verified'):
  if gate.get(key)is not True:raise ValueError('Actual factorial gate not passed: '+key)
 for key,hkey in [('coupling_proof','coupling_proof_sha256'),('build','build_sha256')]:
  if not d.get(key)or sha(d[key])!=d.get(hkey):raise ValueError('Actual research gate dependency differs: '+key)
 if gate.get('contract_sha256')!=contract_sha or gate.get('hsl_library_sha256')!=d['hsl_library_sha256']:raise ValueError('Actual crossed gate corpus/HSL binding differs')
 if gate.get('native_backend')!='ma27':raise ValueError('Historical MUMPS qualification cannot qualify MA27 cells')
 if gate.get('factor_binary_sha256')!=d['binary_sha256']or gate.get('r3_diagonal_cases')!=50 or gate.get('hs_diagonal_cases')!=50:
  raise ValueError('Actual MA27 50-case diagonal gate differs')
 if gate.get('registration_sha256')!=d['preregistration_sha256']:raise ValueError('Preregistered experiment differs')
 factory=module(Path(d['profile_factory']).resolve())
 if factory.evidence(backend='ma27',hsl_library=d['hsl_library'])['source_sha256']!=d.get('profile_factory_source_sha256'):
  raise ValueError('Qualified factorial command factory differs')
 if set(d.get('profiles',{}))!=set(FACTORS):raise ValueError('The four exact-objective cells are required')
 for cell,profile in d['profiles'].items():
  actual=factory.make_command(None,cell,d['binary'],'{input}','{endpoint}','{trace}',backend='ma27',hsl_library=d['hsl_library'])
  def normalized(command):return [x.replace('{binary}',d['binary']).replace('{seconds}','7200').replace('{max_outer}','5000')for x in command]
  if normalized(profile['command_template'])!=normalized(actual):raise ValueError('Qualified factorial profile differs: '+cell)
 d['_factory']=factory;d['_path']=str(path);d['_sha256']=value
 deps=[path,default_path,Path(d['contract']),Path(d['coupling_proof']),Path(d['build']),Path(d['binary']),Path(d['hsl_library']),Path(d['qualified_50gate']),Path(d['preregistration']),Path(d['profile_factory'])]
 deps.extend(Path(d['profile_factory']).parent/rel for rel in d['profile_factory_source_sha256'])
 d['_dependency_signatures']={str(p):signature(p)for p in deps}
 CACHE[cache_key]=d
 return d

def command(installation,method,case,endpoint):
 d=validate(installation);cell=CELLS[method]
 if case.get('regime')not in('I','M'):raise ValueError('Research cell outside preregistered I/M corpus')
 return d['_factory'].make_command(None,cell,d['binary'],str(Path(case['input']).resolve()),str(endpoint),
   str(Path(endpoint).parent/'accepted_trace.jsonl'),backend='ma27',hsl_library=d['hsl_library'])

def check(endpoint,folder,method,installation):
 d=validate(installation);cell=CELLS[method];iq,glob=FACTORS[cell]
 side=read(str(Path(folder)/'endpoint.json')+'.factorial.json');trace=Path(folder)/'accepted_trace.jsonl'
 for key,value in [('objective_exact',True),('inequality_policy',iq),('globalization_policy',glob),('trace_success',True),('budget_verified',True),('legacy_objective_validation',False)]:
  if side.get(key)!=value:raise ValueError('Actual research endpoint proof differs: '+key)
 accepted=0;origins={};base_origins={};states={};header=False
 with trace.open()as stream:
  for line in stream:
   r=json.loads(line);event=r.get('event')
   if event=='header':
    if header:raise ValueError('Duplicate accepted trace header')
    header=True
   elif event=='accepted':
    accepted+=1
    if not header or r.get('accepted_index')!=accepted or r.get('objective_E0_zero')is not True:raise ValueError('Invalid accepted exact-objective trace')
    origin=r.get('final_origin');base=r.get('direction_origin');post=r.get('postprocess')
    if not origin or not base or post is None or origin!=(base if post=='none'else post):raise ValueError('Actual accepted direction origin is missing or inconsistent')
    origins[origin]=origins.get(origin,0)+1;base_origins[base]=base_origins.get(base,0)+1
   elif event in ('startup_restoration_commit','state_recovery_commit','terminal_central_restart_commit','post_multiplier_update','central_restart_commit'):
    if not header or r.get('after_accepted_index')!=accepted:raise ValueError('State event detached from accepted sequence')
    states[event]=states.get(event,0)+1
   else:raise ValueError('Unknown actual research trace event')
   for key,value in [('objective_exact',True),('inequality_policy',iq),('globalization_policy',glob)]:
    if r.get(key)!=value:raise ValueError('Actual research trace policy differs: '+key)
   if event!='header'and not re.fullmatch('[0-9a-f]{64}',r.get('state_sha256','')):raise ValueError('Actual research state digest absent')
 if not header or accepted!=side.get('accepted_events')or accepted!=side.get('endpoint_accepted_steps')or accepted!=endpoint.get('lazy_newton_accepted_steps'):
  raise ValueError('Actual accepted steps do not reconcile')
 return dict(cell=cell,requested_research_method=method,objective_exact=True,trace_success=True,budget_verified=True,
   preregistration_sha256=d['preregistration_sha256'],accepted_direction_origins=origins,
   accepted_base_direction_origins=base_origins,state_events=states,accepted_events=accepted,
   trace_sha256=sha(trace),sidecar_sha256=sha(str(Path(folder)/'endpoint.json')+'.factorial.json'),
   factorial_installation_sha256=d['_sha256'])
