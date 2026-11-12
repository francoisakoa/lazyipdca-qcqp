#!/usr/bin/env python3
"""Prepare an isolated, qualified all-corpus Ipopt-defaults replay."""
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse, collections, gzip, hashlib, json, os, platform, re
import shutil, subprocess, sys, threading, time
from FROZEN_INPUTS import resolve_all

HERE=Path(__file__).resolve().parent
EXPECTED={'mu_strategy':'monotone','tol':1e-8,'constr_viol_tol':1e-4,
          'dual_inf_tol':1.0,'compl_inf_tol':1e-4,'acceptable_iter':15,
          'bound_relax_factor':1e-8,'nlp_scaling_method':'gradient-based',
          'hessian_approximation':'exact'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)

@contextmanager
def phase(name):
    done=threading.Event();started=time.monotonic()
    print(name+' — démarrage',flush=True)
    def heartbeat():
        while not done.wait(10):print(name+' — en cours, '+str(int(time.monotonic()-started))+' s',flush=True)
    worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
    try:yield
    finally:done.set();worker.join()
    print(name+' — terminé, '+format(time.monotonic()-started,'.1f')+' s',flush=True)

def contract(path):
    with gzip.open(path,'rt') as f:d=json.load(f)
    cases=d['cases'];counts=collections.Counter(c['regime'] for c in cases)
    if len(cases)!=4078 or len({c['case_id'] for c in cases})!=4078:
        raise ValueError('Le contrat doit contenir exactement 4 078 identifiants distincts')
    if counts!={'I':1037,'M':797,'E':1704,'Q':540}:
        raise ValueError('Le périmètre complet I/M/E/Q a changé : '+str(counts))
    if len({c['dataset'] for c in cases})!=28 or any(c['dataset']=='pooling_dey' for c in cases):
        raise ValueError('Le contrat doit contenir les 28 datasets sans Dey')
    if any(c.get('start_binding',{}).get('external_start_required') for c in cases):
        raise ValueError('Ce contrat contient un départ externe non prévu')
    return d

def protected(root,cfg):
    files=[root/'INSTALLATION.json']
    for key in ('ipopt_binary','ma27_qualification','mumps_qualification','hsl_library'):
        if cfg.get(key):files.append(Path(cfg[key]))
    folder=root/'install/bin'
    if folder.is_dir():files.extend(p for p in folder.iterdir() if p.is_file())
    return {str(p.resolve()):sha(p) for p in files if p.is_file()}

def resolve_hsl(root,cfg,override=None):
    if override:
        p=Path(override).expanduser().resolve()
        if not p.is_file():raise ValueError('Bibliothèque HSL absente : '+str(p))
        if cfg.get('hsl_library') and p!=Path(cfg['hsl_library']).resolve():
            raise ValueError('La bibliothèque HSL diffère de la référence de campagne')
        return p
    if cfg.get('hsl_library'):
        p=Path(cfg['hsl_library']).expanduser().resolve()
        if p.is_file():return p
    raise ValueError('Bibliothèque HSL qualifiée absente du reçu INSTALLATION.json ; utiliser --hsl-library avec la même bibliothèque')

def defaults_proof(endpoint,log,backend):
    ep=json.loads(Path(endpoint).read_text())
    effective=ep.get('effective_options')
    text=Path(log).read_text(errors='replace')
    markers=[line.split('IPOPT_DEFAULTS_EFFECTIVE ',1)[1] for line in text.splitlines()
             if 'IPOPT_DEFAULTS_EFFECTIVE ' in line]
    if not markers:raise ValueError('Preuve des options effectives absente avant la résolution')
    marker=json.loads(markers[-1]);observed=marker.get('effective_options',{})
    if effective!=observed:raise ValueError('Options du journal et du point final différentes')
    for key,wanted in EXPECTED.items():
        value=observed.get(key)
        if isinstance(wanted,float):
            if not isinstance(value,(float,int)) or abs(value-wanted)>max(abs(wanted)*1e-12,1e-300):
                raise ValueError('Option non par défaut '+key+' : '+str(value))
        elif value!=wanted:raise ValueError('Option non par défaut '+key+' : '+str(value))
    if observed.get('linear_solver')!=backend:raise ValueError('Backend différent du backend demandé')
    if backend=='mumps' and observed.get('mumps_pivot_order')!=7:
        raise ValueError('Ordre MUMPS différent du défaut automatique 7')
    versions=re.findall(r'This is Ipopt version\s+([^,\s]+)',text)
    if not versions or versions[-1]!='3.14.11':
        raise ValueError('La bibliothèque exécutée ne confirme pas Ipopt 3.14.11 : '+str(versions))
    return dict(effective_options=observed,pre_solve_proof=marker,runtime_version=versions[-1])

def write_fixture(path,n,x0,lb,ub,obj,eq=(),ineq=()):
    def row(q):
        c,linear,hessian=q
        return ['quadratic','name q','constant '+str(c),'linear_nnz '+str(len(linear)),
                *[str(i)+' '+str(v) for i,v in linear], 'hessian_nnz '+str(len(hessian)),
                *[str(i)+' '+str(j)+' '+str(v) for i,j,v in hessian],'end_quadratic']
    lines=['LAZYIPDCA_QCQP_V2','name default_qualification','n '+str(n),
           'x0 '+' '.join(map(str,x0)),'lb '+' '.join(map(str,lb)),
           'ub '+' '.join(map(str,ub)),'objective',*row(obj),'equalities '+str(len(eq))]
    for q in eq:lines+=row(q)
    lines+=['inequalities '+str(len(ineq))]
    for q in ineq:lines+=row(q)
    path.write_text('\n'.join(lines+['end_problem'])+'\n')

def qualify(binary,hsl,out,env,backends=('mumps','ma27')):
    folder=out/'qualification';folder.mkdir()
    fixtures={
      'active_bound':(1,[.25],[0],[1],(0,[(0,-2)],[(0,0,1)]),(),()),
      'quadratic_inequality':(1,[.3],['-inf'],['inf'],(0,[(0,-1)],[]),(),[(-1,[],[(0,0,2)])]),
      'quadratic_equality':(2,[.8,.6],['-inf','-inf'],['inf','inf'],(0,[(0,-2)],[(0,0,1),(1,1,1)]),[(-1,[],[(0,0,2),(1,1,2)])],()),
      'large_finite_bound':(1,[.2],[-1e20],[1e20],(0,[],[(0,0,1)]),(),()),
    }
    records=[]
    for name,data in fixtures.items():write_fixture(folder/(name+'.qcq'),*data)
    for backend in backends:
        for name in fixtures:
            task=folder/(name+'__'+backend);task.mkdir()
            inp=folder/(name+'.qcq');endpoint=task/'endpoint.json';log=task/'solver.log'
            cmd=[str(binary),'--problem',str(inp),'--json',str(endpoint),'--linear-solver',backend,
                 '--time-limit','60','--max-iter','300']
            if backend=='ma27':cmd+=['--hsl-library',str(hsl)]
            with log.open('w') as f:
                result=subprocess.run(cmd,cwd=task,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=70)
            if not endpoint.is_file():raise ValueError('Point final absent ; consulter '+str(log))
            proof=defaults_proof(endpoint,log,backend)
            audit=task/'AUDIT.json'
            subprocess.run([sys.executable,str(HERE/'AUDIT_ORIGINAL.py'),'--input',str(inp),
                            '--endpoint',str(endpoint),'--out',str(audit)],check=True,env=env,
                           stdout=subprocess.DEVNULL,timeout=120)
            checked=json.loads(audit.read_text())
            if checked.get('strict_qcqp') is not True:
                raise ValueError('Qualification KKT originale refusée : '+str(audit))
            records.append(dict(fixture=name,backend=backend,returncode=result.returncode,
                input_sha256=sha(inp),endpoint_sha256=sha(endpoint),audit_sha256=sha(audit),proof=proof))
            print('QUALIFICATION',backend,name,'KKT original PASS',flush=True)
    receipt=dict(schema='ipopt-defaults-target-qualification-v1',records=records,
                 qualified_backends=list(backends),all_original_strict=True)
    atomic(out/'QUALIFICATION.json',receipt)
    return receipt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--hsl-library',type=Path)
    ap.add_argument('--ipopt-include',type=Path)
    ap.add_argument('--frozen-root',type=Path,action='append',default=[],help='Racine supplémentaire explicitement autorisée pour des copies gelées ; vérification SHA256 obligatoire')
    a=ap.parse_args();root=a.root.expanduser().resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    out=(a.out or Path('/extra/LAZYQCQP_IPOPT_DEFAUT_INSTALL')/stamp).expanduser().resolve()
    if out.exists():raise ValueError('Le dossier de préparation doit être neuf : '+str(out))
    if not(root/'INSTALLATION.json').is_file():raise ValueError('Installation de campagne absente : '+str(root))
    cfg=json.loads((root/'INSTALLATION.json').read_text());hsl=resolve_hsl(root,cfg,a.hsl_library)
    if cfg.get('hsl_library_sha256') and sha(hsl)!=cfg['hsl_library_sha256']:
        raise ValueError('La bibliothèque HSL qualifiée a changé')
    old=protected(root,cfg);plan=contract(HERE/'CONTRAT_4078.json.gz')
    out.mkdir(parents=True);frozen={}
    try:
        with phase('VÉRIFICATION DES 4 078 ENTRÉES ET DÉPARTS GELÉS'):
            def progress(stage,value):
                if stage=='index':print('INDEX GELÉ',value['hashed'],'QCQP ; empreintes retrouvées',value['matched_hashes'],'/ 4078',flush=True)
                else:print('ENTRÉES ET x0 VÉRIFIÉS',value['done'],'/',value['total'],'; valides',value['verified'],flush=True)
            frozen,resolution=resolve_all(root,plan['cases'],out/'FROZEN_INPUTS_REPORT.json',a.frozen_root,progress)
            print('CHEMINS GELÉS VÉRIFIÉS :',len(frozen),'entrées ;',resolution['relocated'],'chemins résolus par contenu',flush=True)
        with phase('COMPILATION ISOLÉE IPOPT PAR DÉFAUT'):
            cmd=[sys.executable,str(HERE/'BUILD_IPOPT_DEFAULTS.py'),'--root',str(root),'--out',str(out/'build')]
            if a.ipopt_include:cmd+=['--ipopt-include',str(a.ipopt_include.resolve())]
            subprocess.run(cmd,check=True)
        build=json.loads((out/'build/BUILD.json').read_text());binary=Path(build['binary']).resolve()
        library=Path(build['ipopt_library']).resolve()
        env=dict(os.environ)
        for k in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT'):env.pop(k,None)
        libpath=':'.join(dict.fromkeys([str(library.parent),str(hsl.parent),*filter(None,env.get('LD_LIBRARY_PATH','').split(':'))]))
        env.update(LD_LIBRARY_PATH=libpath,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',BLIS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        with phase('QUALIFICATION DES DEUX BACKENDS ET DES OPTIONS EFFECTIVES'):
            q=qualify(binary,hsl,out,env)
        after=protected(root,cfg)
        if old!=after:raise ValueError('Un fichier de l’installation précédente a changé pendant la préparation')
        shutil.copy2(HERE/'CONTRAT_4078.json.gz',out/'CONTRAT_4078.json.gz')
        machine=dict(platform=platform.platform(),python=sys.version,
                     cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_model=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),None),
                     memory_total=next((x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')),None))
        installation=dict(schema='ipopt-defaults-full-corpus-installation-v1',prepared_utc=datetime.now(timezone.utc).isoformat(),
            root=str(root),contract=str(out/'CONTRAT_4078.json.gz'),contract_sha256=sha(out/'CONTRAT_4078.json.gz'),
            default_binary=str(binary),default_binary_sha256=sha(binary),ipopt_library=str(library),ipopt_library_sha256=sha(library),
            hsl_library=str(hsl),hsl_library_sha256=sha(hsl),python=sys.executable,frozen_bindings=frozen,
            runtime_environment={'LD_LIBRARY_PATH':libpath},qualified_backends=q['qualified_backends'],
            qualification=str(out/'QUALIFICATION.json'),qualification_sha256=sha(out/'QUALIFICATION.json'),
            build=str(out/'build/BUILD.json'),build_sha256=sha(out/'build/BUILD.json'),
            frozen_resolution=str(out/'FROZEN_INPUTS_REPORT.json'),frozen_resolution_sha256=sha(out/'FROZEN_INPUTS_REPORT.json'),
            cases=4078,tasks=8156,regimes=dict(collections.Counter(c['regime'] for c in plan['cases'])),
            datasets=28,budgets={'seconds':7200,'max_iterations':5000},cpus=list(range(22)),
            numerical_defaults=EXPECTED,baseline_files_unchanged=old,machine=machine)
        atomic(out/'INSTALLATION.json',installation)
        (HERE/'LAST_INSTALLATION.txt').write_text(str(out/'INSTALLATION.json')+'\n')
        print('PRÉPARATION VALIDÉE : 4 078 instances × 2 = 8 156 tâches ; tous les régimes, sans Dey',flush=True)
        print('INSTALLATION_PRÊTE',out/'INSTALLATION.json',flush=True)
    except BaseException as error:
        atomic(out/'PREPARATION_ERROR.json',dict(error=type(error).__name__+': '+str(error),baseline_still_unchanged=(old==protected(root,cfg))))
        raise

if __name__=='__main__':
    try:main()
    except Exception as e:print('PRÉPARATION REFUSÉE — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
