#!/usr/bin/env python3
"""Canonical commands for one controlled refinement intervention.
All production cells use hybrid inequality initialization and an exact objective;
only continued refinement and the declared globalization switches vary.
"""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
BACKENDS=('ma27','mumps')
CELLS={f'{r}_{g}_{b}':dict(refinement_policy='frozen'if r=='FROZEN'else'adaptive',globalization_policy='r3'if g=='GR3'else'hs',backend=b,objective_exact=True,inequality_policy='hybrid',common_guards='r3',common_representation='original_qcqp')for b in BACKENDS for r in ('FROZEN','REFINED')for g in ('GR3','GHS')}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def data():return json.loads((HERE/'REFERENCE_PROFILES.json').read_text())
def set_value(args,flag,value):
    if args.count(flag)>1:raise ValueError('Option ambiguë : '+flag)
    if flag in args:
        i=args.index(flag)
        if i+1==len(args):raise ValueError('Valeur absente : '+flag)
        args[i+1]=str(value)
    else:args.extend((flag,str(value)))

def _base(method,backend,binary,input,endpoint,hsl_library=None):
    if backend not in BACKENDS:raise ValueError('Backend non déclaré : '+str(backend))
    templates=data()['profiles']
    if method not in templates:raise ValueError('Profil historique inconnu : '+str(method))
    fields=dict(binary=str(binary),input=str(input),endpoint=str(endpoint),backend=backend,seconds='7200',max_outer='5000')
    args=[v.format(**fields)for v in templates[method]]
    set_value(args,'--native-al-fallback','off');set_value(args,'--standard-r10-policy','auto')
    set_value(args,'--dca-rank-presolve-report',str(endpoint)+'.rank_presolve.json')
    if backend=='ma27':
        if not hsl_library:raise ValueError('Bibliothèque HSL explicite requise pour MA27')
        set_value(args,'--hsl-library',hsl_library)
    return args

def make_command(cell,binary,input,endpoint,trace,hsl_library=None):
    if cell not in CELLS:raise ValueError('Cellule non déclarée : '+str(cell))
    if trace is None:raise ValueError('Trace des acceptations requise')
    c=CELLS[cell];args=_base('HH',c['backend'],binary,input,endpoint,hsl_library)
    # Identical observation work for all cells. These counters are not tests
    # for algorithmic acceptance; final accepted-state hashes remain recorded.
    set_value(args,'--accepted-checkpoint','0')
    for flag,value in (('--factorial-inequalities','hybrid'),('--factorial-globalization',c['globalization_policy']),('--factorial-refinement',c['refinement_policy']),('--factorial-common-guards','r3'),('--factorial-trace',trace)):
        set_value(args,flag,value)
    return args

def make_legacy_command(method,backend,binary,input,endpoint,trace=None,hsl_library=None):
    if method not in ('R3','HH'):raise ValueError('Qualification historique R3/HH seulement')
    args=_base(method,backend,binary,input,endpoint,hsl_library)
    if trace is not None:set_value(args,'--factorial-trace',trace)
    return args

def command_templates():
    return {cell:dict(c,command_template=make_command(cell,'{binary}','{input}','{endpoint}','{trace}','{hsl_library}'if c['backend']=='ma27'else None))for cell,c in CELLS.items()}

def evidence():
    return dict(schema='controlled-refinement-profile-factory-v1',cells=CELLS,source_sha256={p.name:sha(p)for p in [Path(__file__).resolve(),HERE/'REFERENCE_PROFILES.json']},objective_exact=True,inequality_initialization='common_certified_hybrid',common_guards='r3',budgets=dict(seconds=7200,max_outer=5000,shared_restoration=600),baseline_names_preserved=True)
if __name__=='__main__':print(json.dumps(evidence(),indent=2))
