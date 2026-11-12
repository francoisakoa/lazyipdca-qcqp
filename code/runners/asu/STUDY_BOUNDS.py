#!/usr/bin/env python3
"""Build, qualify, pilot, promote and complete the ASU503 ablation, resumably."""
from pathlib import Path
from types import SimpleNamespace
import argparse,fcntl,json,os,subprocess,sys
import BOUNDS as study
import QUALIFIER_BOUNDS as qualification
import REUSE_BOUNDS as reuse

HERE=Path(__file__).resolve().parent;base=study.base


def default_root():return Path(base.read(HERE/'BASE_CONTEXT.json')['plan_header']['root'])


def state(out,stage,**extra):
    value=dict(schema='asu-bound-study-status-v2',timestamp=base.utc(),stage=stage,**extra)
    base.atomic(out/'ETUDE_STATUS.json',value)
    print('ÉTUDE ASU — '+stage,flush=True)


def prepare_once(root,build,out,frozen,pilot):
    installation=out/'INSTALLATION.json'
    if installation.is_file():
        config=study.installation(installation)
        if config['root']!=str(root)or config['build_manifest']!=str(build)or config.get('pilot_cases')!=pilot:
            raise ValueError('Existing preparation differs from requested study')
        return installation
    if out.exists():
        # Preparation is read-only with respect to the frozen inputs. Preserve
        # its own incomplete diagnostic, then retry into a fresh directory.
        if any(p.name!='FROZEN_RESOLUTION.json'or not p.is_file()for p in out.iterdir()):
            raise ValueError('Unrecognized incomplete preparation retained: '+str(out))
        attempt=1;archived=out.with_name(out.name+f'_incomplete_{attempt:03d}')
        while archived.exists():attempt+=1;archived=out.with_name(out.name+f'_incomplete_{attempt:03d}')
        out.rename(archived)
        print('PRÉPARATION INCOMPLÈTE CONSERVÉE : '+str(archived),flush=True)
    study.prepare(SimpleNamespace(root=root,build=build,out=out,frozen_root=frozen,pilot_cases=pilot))
    return installation


def campaign_once(installation,out,qualified):
    action='reprendre'if(out/'RUN_CONTRACT.json').is_file()else'lancer'
    study.run(SimpleNamespace(installation=installation,out=out,qualification=qualified,cpus='0-21',action=action))
    result=base.read(out/'STATUS.json',{})
    config=study.installation(installation)
    if result.get('state')!='completed'or result.get('done')!=config['tasks']or result.get('active_slots')!=0:
        raise ValueError('Campaign incomplete: '+str(out))


def run(a):
    root=a.root.expanduser().resolve(strict=True);out=a.out.expanduser().resolve()
    if out==root or root in out.parents or out in root.parents or out==HERE or HERE in out.parents:
        raise ValueError('Study results must be separate from the kit and original installation')
    out.mkdir(parents=True,exist_ok=True)
    with(out/'ETUDE.lock').open('a')as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another controller already owns this ASU study')
        plan=dict(schema='asu-bound-study-plan-v2',root=str(root),out=str(out),
          supplied_build=str(a.build.expanduser().resolve())if a.build else None,
          frozen_roots=[str(p.expanduser().resolve())for p in a.frozen_root],
          source_snapshot={str(p.relative_to(HERE)):base.sha(p)for p in sorted(HERE.glob('*.py'))},
          pilot_cases=12,pilot_ablation_observations=144,full_ablation_observations=6036,
          qualification_controls=144,total_native_executions=6180,
          pilot_reused_in_full=True,cpus=list(range(22)),jobs=22,threads=1,
          budgets=dict(seconds=7200,outer=5000,restoration_shared=600,tolerance=1e-6))
        old=base.read(out/'PLAN_ETUDE.json')
        if old and old!=plan:raise ValueError('Study plan/source changed; a separate output folder is required')
        base.atomic(out/'PLAN_ETUDE.json',plan)
        try:
            state(out,'construction')
            if a.build:build=a.build.expanduser().resolve(strict=True);study.build_manifest(build)
            else:
                selected=base.read(out/'BUILD_SELECTION.json')
                if selected:
                    build=Path(selected['path']);study.verify_artifact(build,selected['sha256']);study.build_manifest(build)
                else:
                    build_dir=out/'build';attempt=1
                    while build_dir.exists()and not(build_dir/'BUILD_BOUNDS.json').is_file():
                        attempt+=1;build_dir=out/f'build_retry_{attempt:03d}'
                    build=build_dir/'BUILD_BOUNDS.json'
                    if not build.is_file():
                        subprocess.run([sys.executable,str(HERE/'BUILD_BOUNDS.py'),'--out',str(build_dir),'--cpu','0'],check=True)
                    study.build_manifest(build);base.atomic(out/'BUILD_SELECTION.json',reuse.artifact(build))
            state(out,'préparation pilote',build_manifest=str(build))
            pilot=prepare_once(root,build,out/'preparation_pilot',a.frozen_root,12)
            state(out,'qualification profil',controls=144)
            qualification.run(pilot,out/'qualification');qualified=out/'qualification/QUALIFICATION.json'
            state(out,'pilote ablation',observations=144)
            campaign_once(pilot,out/'pilote',qualified);reuse.pilot_gate(pilot,out/'pilote')
            state(out,'préparation complète',cases=503)
            full=prepare_once(root,build,out/'preparation_full',a.frozen_root,None)
            state(out,'réutilisation pilote',new_native_processes=0)
            reuse.promote(pilot,out/'pilote',full,out/'campagne',qualified)
            state(out,'campagne complète',planned=6036,pilot_already_counted=144)
            campaign_once(full,out/'campagne',qualified)
            state(out,'analyse appariée')
            if not(out/'analyse/PAIRED_SUMMARY.json').is_file():
                subprocess.run([sys.executable,str(HERE/'ANALYSER_BOUNDS.py'),'--run',str(out/'campagne'),'--out',str(out/'analyse')],check=True)
            analysis=base.read(out/'analyse/PAIRED_SUMMARY.json')
            if analysis.get('complete')is not True or analysis.get('recorded')!=6036:raise ValueError('Full analysis is incomplete')
            state(out,'completed',planned=6036,recorded=6036,qualification_controls=144,
                  total_native_executions=6180,qualification=str(qualified),analysis=str(out/'analyse/PAIRED_SUMMARY.json'))
        except BaseException as error:
            previous=base.read(out/'ETUDE_STATUS.json',{})
            state(out,'blocked',blocked_stage=previous.get('stage'),reason=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=default_root())
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--build',type=Path,help='Optional verified BUILD_BOUNDS.json, built with this kit')
    parser.add_argument('--frozen-root',type=Path,action='append',default=[])
    a=parser.parse_args();run(a)


if __name__=='__main__':
    try:main()
    except Exception as error:print('ÉTUDE ASU BLOQUÉE — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
