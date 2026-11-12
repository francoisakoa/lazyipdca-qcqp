#!/usr/bin/env python3
"""Qualify the optional four-cell MA27 experiment using the already built sources."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys
HERE=Path(__file__).resolve().parent;KIT=HERE.parent;F=KIT/'factorial';D=KIT/'default_relaunch'
if not F.is_dir():F=KIT/'article_revision_R4/factorial'
if not D.is_dir():D=KIT/'article_revision_R4/default_relaunch'
sys.path.insert(0,str(D));import PREPARER as common

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--installation',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();out=a.out.resolve();cfg=json.loads(a.installation.read_text());build=json.loads(a.build.read_text());root=Path(cfg['root']);reg=F/'PRE_ENREGISTREMENT_MA27_20261108.json'
 if out.exists():raise ValueError('Sortie factorielle neuve requise')
 out.mkdir(parents=True)
 if build.get('native_backend')!='ma27'or build.get('native_ma27_qualified')is not True or build.get('mumps_compiled')is not False:raise ValueError('Compilation et qualification MA27 réelle requises')
 env=dict(__import__('os').environ);env.update(build['runtime_environment'])
 for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT'):env.pop(key,None)
 runner=module('factorial_MA27_cpu_lock',D/'CAMPAGNE.py')
 with runner.machine_lock(),common.phase('SÉPARATION DES FACTEURS SUR LES 1 834 CAS I/M'):
  subprocess.run([sys.executable,'-u',str(F/'VERIFIER_COUPLAGE.py'),'--root',str(root),'--binary',build['research_binary'],
   '--registration',str(reg),'--contract',cfg['contract'],'--input-installation',str(a.installation.resolve()),
   '--build-proof',str(a.build.resolve()),'--out',str(out/'coupling')],check=True,env=env)
 with common.phase('REPRODUCTION DES 50 CAS SOUS MA27 — MÊME BACKEND DES DEUX CÔTÉS'):
  subprocess.run([sys.executable,'-u',str(F/'QUALIFIER_50.py'),'--root',str(root),'--binary',build['research_binary'],
   '--baseline-trace-binary',build['baseline_trace_binary'],'--build-proof',str(a.build.resolve()),'--registration',str(reg),
   '--coupling-proof',str(out/'coupling/COUPLING_GATE.json'),'--input-installation',str(a.installation.resolve()),
   '--out',str(out/'qualification_50'),'--jobs','22','--cpus','0-21'],check=True,env=env)
 gate=out/'qualification_50/GATE.json';value=json.loads(gate.read_text())
 if not all(value.get(k)is True for k in ('passed','production_authorized','coupling_separation_verified')):raise ValueError('Qualification factorielle MA27 incomplète')
 factory=module('factorial_MA27_factory',F/'PROFILES.py')
 result=dict(schema='native-factorial-ma27-qualified-installation-v1',native_backend='ma27',binary=build['research_binary'],binary_sha256=build['research_binary_sha256'],
  hsl_library=cfg['hsl_library'],hsl_library_sha256=cfg['hsl_library_sha256'],runtime_environment=build['runtime_environment'],contract=cfg['contract'],
  qualified_50gate=str(gate),qualified_50gate_sha256=common.sha(gate),profile_factory=str(F/'PROFILES.py'),profile_factory_source_sha256=factory.evidence()['source_sha256'],
  preregistration=str(reg),preregistration_sha256=common.sha(reg),profiles=factory.command_templates(hsl_library=cfg['hsl_library']),
  coupling_proof=str(out/'coupling/COUPLING_GATE.json'),coupling_proof_sha256=common.sha(out/'coupling/COUPLING_GATE.json'),
  build=str(a.build.resolve()),build_sha256=common.sha(a.build),default_installation=str(a.installation.resolve()),default_installation_sha256=common.sha(a.installation),
  cases=1834,tasks=7336,objective_exact=True)
 common.atomic(out/'FACTORIAL_INSTALLATION.json',result)
 print('FACTORIEL MA27 QUALIFIÉ : 7 336 tâches natives supplémentaires',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:print('FACTORIEL MA27 BLOQUÉ — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
