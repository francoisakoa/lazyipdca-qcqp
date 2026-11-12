#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
POLICIES=('R3','SS','HH');BACKENDS=('ma27','mumps');STATES=('keep','remove')
CELLS={f'{p}_{s}_{b}':dict(policy=p,bounds=s,backend=b)for b in BACKENDS for p in POLICIES for s in STATES}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def set_value(args,key,value):
    if args.count(key)>1:raise ValueError('Option ambiguë : '+key)
    if key in args:args[args.index(key)+1]=str(value)
    else:args.extend((key,str(value)))
def command(cell,binary,input,endpoint,hsl):
    return declared_command(CELLS[cell],binary,input,endpoint,hsl)
def profile_command(policy,backend,binary,input,endpoint,hsl,reference=False):
    if policy not in POLICIES or backend not in BACKENDS:raise ValueError('Unknown policy/backend')
    args=declared_command(dict(policy=policy,bounds='profile',backend=backend),binary,input,endpoint,hsl)
    if reference:
        i=args.index('--research-equality-implied-bounds');del args[i:i+2]
    return args
def declared_command(c,binary,input,endpoint,hsl):
    data=json.loads((HERE/'REFERENCE_PROFILES.json').read_text())
    fields=dict(binary=str(binary),input=str(input),endpoint=str(endpoint),backend=c['backend'],seconds='7200',max_outer='5000')
    args=[s.format(**fields)for s in data['profiles'][c['policy']]]
    for k,v in [('--native-al-fallback','off'),('--standard-r10-policy','auto'),
      ('--dca-rank-presolve-report',str(endpoint)+'.rank_presolve.json'),
      ('--research-equality-implied-bounds',c['bounds']),
      ('--restoration-iters',600),('--shared-restoration-budget',1)]:set_value(args,k,v)
    if c['backend']=='ma27':
        if not hsl:raise ValueError('Bibliothèque HSL réelle requise')
        set_value(args,'--hsl-library',hsl)
    return args
def templates():return {cell:dict(v,command=command(cell,'{binary}','{input}','{endpoint}','{hsl_library}'))for cell,v in CELLS.items()}
if __name__=='__main__':print(json.dumps(templates(),indent=2))
