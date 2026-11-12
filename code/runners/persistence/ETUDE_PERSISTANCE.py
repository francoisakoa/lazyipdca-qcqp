#!/usr/bin/env python3
"""Build, qualify and measure the persistence ablation without touching old runs."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    if not path.is_file():return None
    return json.loads(path.read_text())


def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def stage(out,name,**extra):
    write(out/'ETUDE_STATUS.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=name,**extra))
    print('PERSISTANCE — '+name,flush=True)


def invoke(script,*args):
    subprocess.run([sys.executable,'-u',str(HERE/script),*map(str,args)],check=True)


def run(args):
    out=args.out.expanduser().resolve()
    if out==HERE or HERE in out.parents or out in HERE.parents:raise ValueError('Sorties séparées du kit requises')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'ETUDE.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Une étude de persistance utilise déjà ce dossier')
        plan=dict(schema='persistence-study-plan-v1',output=str(out),source_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
                  cohort_sha256=sha(HERE/'COHORTE_PERSISTANCE.json'),supplied_build=str(args.build.expanduser().resolve()) if args.build else None,
                  ipopt_run=str(args.ipopt_run.expanduser().resolve()),qualification_runs=72,timed_runs=144,
                  total_solver_runs=216,repetitions=3,backend_pairs=24,cpu=0,threads=1)
        previous=read(out/'PLAN_ETUDE.json')
        if previous and previous!=plan:raise ValueError('Plan ou code différent : conserver cette étude et utiliser un nouveau dossier')
        write(out/'PLAN_ETUDE.json',plan)
        try:
            stage(out,'construction')
            if args.build:
                build=args.build.expanduser().resolve(strict=True)
            else:
                selected=read(out/'BUILD_SELECTION.json')
                if selected:
                    build=Path(selected['path'])
                    if not build.is_file() or sha(build)!=selected['sha256']:raise ValueError('Reçu de construction modifié ou absent')
                else:
                    folder=out/'build';attempt=1
                    while folder.exists() and not (folder/'BUILD_PERSISTANCE.json').is_file():
                        attempt+=1;folder=out/f'build_retry_{attempt:03d}'
                    build=folder/'BUILD_PERSISTANCE.json'
                    if not build.is_file():invoke('BUILD_PERSISTANCE.py','--out',folder,'--cpu','0')
                    write(out/'BUILD_SELECTION.json',dict(path=str(build),sha256=sha(build)))
            experience=out/'experience'
            stage(out,'préparation',build=str(build))
            if not (experience/'INSTALLATION.json').is_file():
                invoke('PERSISTANCE.py','preparer','--build',build,'--out',experience,'--ipopt-run',args.ipopt_run)
            stage(out,'qualification',planned=72)
            invoke('PERSISTANCE.py','qualifier','--out',experience)
            stage(out,'chronométrage',planned=144)
            invoke('PERSISTANCE.py','lancer','--out',experience)
            stage(out,'analyse')
            invoke('PERSISTANCE.py','synthese','--out',experience)
            summary=read(experience/'SYNTHESE_PERSISTANCE.json')
            if not summary or summary.get('complete')is not True or summary.get('qualification_passed')is not True or summary.get('planned')!=144 or summary.get('timed_repetitions_completed')!=144:
                raise ValueError('Mesures ou qualification incomplètes : les preuves sont conservées ; reprendre la même commande')
            stage(out,'completed',qualification_runs=72,timed_runs=144,summary=str(experience/'SYNTHESE_PERSISTANCE.json'))
        except BaseException as error:
            prior=read(out/'ETUDE_STATUS.json') or {}
            stage(out,'blocked',blocked_stage=prior.get('stage'),reason=str(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('/extra/LAZYQCQP_PERSISTANCE_RESULTS/ETUDE_PERSISTANCE_20261110_R1'))
    p.add_argument('--build',type=Path)
    p.add_argument('--ipopt-run',type=Path,default=Path('/extra/LAZYQCQP_IPOPT_ALIGNE_RESULTS/IPOPT_4CONFIG_20261109T140336_fa503d1c'))
    run(p.parse_args())


if __name__=='__main__':
    try:main()
    except Exception as error:print('ÉTUDE BLOQUÉE — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
