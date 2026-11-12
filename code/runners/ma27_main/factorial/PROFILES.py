#!/usr/bin/env python3
"""Single canonical command factory for qualification and production.
Every factorial cell starts from the full published HH command. Research flags
replace only the two factors and suppress objective corrections. Historical R3
and HH controls are retained for same-backend fresh diagonal qualification.
The requested campaign backend is MA27; MUMPS evidence remains historical.
"""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
CELLS={'R3':('cache','r3'),'X1':('cache','hs'),'X2':('hybrid','r3'),'H0':('hybrid','hs')}
ALIASES={'cache_r3':'R3','cache_hs':'X1','hybrid_r3':'X2','hybrid_hs':'H0'}
METHOD_TAGS={k:ALIASES.get(k,k)for k in list(CELLS)+list(ALIASES)}
def canonical_cell(cell):
    cell=ALIASES.get(cell,cell)
    if cell not in CELLS:raise ValueError('Unknown factorial cell: '+str(cell))
    return cell

def _published(method):
    return json.loads((HERE/'profile_evidence'/('PUBLISHED_'+method+'_COMMAND.json')).read_text())['command']

def _replace(command,binary,input,endpoint,trace=None,backend='ma27',hsl_library=None):
    if backend not in ('ma27','mumps'):raise ValueError('Native backend must be ma27 or explicit historical mumps')
    args=list(command);args[0]=str(binary)
    if args.count('--linear-backend')!=1:raise ValueError('Canonical command must declare one primary backend')
    args[args.index('--linear-backend')+1]=backend
    if hsl_library is not None and backend=='ma27':args+=['--hsl-library',str(hsl_library)]
    replacements={'--problem':str(input),'--json':str(endpoint),'--dca-rank-presolve-report':str(endpoint)+'.rank_presolve.json'}
    for option,value in replacements.items():
        if args.count(option)!=1:raise ValueError('Canonical command has ambiguous option '+option)
        args[args.index(option)+1]=value
    if trace is not None:args+=['--factorial-trace',str(trace)]
    return args

def make_command(profiles,cell,binary,input,endpoint,trace,legacy=False,backend='ma27',hsl_library=None):
    """The optional profiles argument is compatibility-only; frozen command
    evidence bundled beside this module is authoritative and hash-bound.
    """
    cell=canonical_cell(cell)
    if legacy and cell!='H0':raise ValueError('Legacy-objective validation is allowed only for H0')
    if trace is None:raise ValueError('Factorial production/qualification requires a trace path')
    c,g=CELLS[cell]
    args=_replace(_published('HH'),binary,input,endpoint,trace,backend,hsl_library)
    # Uniform observation-only I/O override; algorithmic HH options unchanged.
    args[args.index('--accepted-checkpoint')+1]='0'
    args+=['--factorial-inequalities',c,'--factorial-globalization',g]
    if legacy:args+=['--factorial-validation-legacy-objective']
    return args

def make_legacy_command(profiles,method,binary,input,endpoint,trace=None,backend='ma27',hsl_library=None):
    if method not in ('R3','HH'):raise ValueError('Historical diagonal must be R3 or HH')
    return _replace(_published(method),binary,input,endpoint,trace,backend,hsl_library)

def make_published_command(method,binary,input,endpoint,backend='ma27',hsl_library=None):
    templates=json.loads((HERE/'profile_evidence/PUBLISHED_SIX_PROFILES.json').read_text())['profiles']
    if method not in templates:raise ValueError('Unknown published method: '+str(method))
    if backend not in ('ma27','mumps'):raise ValueError('Native backend must be ma27 or explicit historical mumps')
    fields=dict(binary=str(binary),input=str(input),endpoint=str(endpoint),backend=backend,seconds='7200',max_outer='5000')
    args=[x.format(**fields)for x in templates[method]]+['--dca-rank-presolve-report',str(endpoint)+'.rank_presolve.json','--native-al-fallback','off','--standard-r10-policy','auto']
    if backend=='ma27'and hsl_library is not None:args+=['--hsl-library',str(hsl_library)]
    return args

def command_templates(binary='{binary}',backend='ma27',hsl_library=None):
    result={}
    for name,(c,g)in CELLS.items():
        cmd=make_command(None,name,binary,'{input}','{endpoint}','{trace}',backend=backend,hsl_library=hsl_library)
        cmd[cmd.index('--time-limit')+1]='{seconds}'
        cmd[cmd.index('--max-outer')+1]='{max_outer}'
        result[name]={'label':name,'native_backend':backend,'inequality_policy':c,'globalization_policy':g,
                      'objective_exact':True,'command_template':cmd,
                      'accepted_steps_endpoint_key':'lazy_newton_accepted_steps'}
    return result

def evidence(backend='ma27',hsl_library=None):
    files=[Path(__file__).resolve()]+sorted((HERE/'profile_evidence').glob('*.json'))
    return {'schema':'lazy-factorial-profile-factory-v2-ma27','base_profile':'full published HH algorithmic options for all four cells',
            'observation_only_output_override':{'--accepted-checkpoint':'0 for all production cells; per-commit hashed trace supplies trajectories without full-vector disk output'},
            'aliases':ALIASES,'cells':command_templates(backend=backend,hsl_library=hsl_library),
            'native_backend':backend,'hsl_library':str(hsl_library)if hsl_library is not None else'HSL_LIBRARY from pinned runtime environment',
            'backend_change':'MA27 is an explicitly requested fresh experiment; historical MUMPS trajectories/certificates do not qualify these commands',
            'ma27_qualification_pending':backend=='ma27',
            'source_sha256':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},
            'budgets':{'seconds':7200,'outer_iterations':5000,'shared_restoration_iterations':600},
            'legacy_objective_validation':'H0 only; excluded from the 7336 production tasks'}
if __name__=='__main__':print(json.dumps(evidence(),indent=2))
