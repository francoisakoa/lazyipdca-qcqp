#!/usr/bin/env python3
"""Prepare the previous mega device with six MA27 policies and two default Ipopt references.
All existing scientific inputs are reused at the paths declared by the previous
PREPARATION.json. No frozen file, start or old installation is rewritten.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,importlib.util,json,os,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
KIT=HERE.parent
DEFAULT=KIT/'default_relaunch'
FACTOR=KIT/'factorial'
if not DEFAULT.is_dir():DEFAULT=KIT/'article_revision_R4/default_relaunch'
if not FACTOR.is_dir():FACTOR=KIT/'article_revision_R4/factorial'
sys.path.insert(0,str(DEFAULT))
import PREPARER as common

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def run(command,phase,env):
 with common.phase(phase):subprocess.run(list(map(str,command)),check=True,env=env)

def stage_previous_device(root,oldroot,oldcfg,prepared,defaults,build,include_factorial,factorial):
 root.mkdir(parents=True,exist_ok=False)
 shutil.copytree(HERE/'campaign',root/'campaign',ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.log'))
 shutil.copy2(HERE/'setup_runtime.py',root/'setup_runtime.py')
 shutil.copytree(HERE/'validation_tools',root/'validation_tools',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (root/'native').mkdir();(root/'validation').mkdir()
 # The code source, not 22 GB of input data, is shared with the isolated build.
 reference_source=Path(build['ma27_reference_source']).resolve(strict=True)
 (root/'source').symlink_to(reference_source,target_is_directory=True)
 (root/'install/bin').mkdir(parents=True)
 ref=Path(build['ma27_reference_binary']).resolve(strict=True)
 (root/'install/bin/lazyipdca_qcqp_cli').symlink_to(ref)
 qpath=Path(build['native_ma27_qualification']).resolve(strict=True)
 qualification=json.loads(qpath.read_text())
 if qualification.get('passed')is not True or qualification.get('six_profiles_qualified')is not True:
  raise ValueError('Qualification MA27 réelle des six profils absente')
 if qualification.get('binary_sha256')!=common.sha(ref)or qualification.get('hsl_library_sha256')!=defaults['hsl_library_sha256']:
  raise ValueError('Qualification native liée à un autre binaire ou HSL')
 profiles=root/'campaign/profiles.json';catalog=json.loads(profiles.read_text());old_profiles=copy.deepcopy(catalog['profiles'])
 (root/'validation/REFERENCE_PROFILES_PUBLISHED.json').write_text(profiles.read_text())
 catalog.update(schema='six-historical-policies-ma27-backend-variant-v1',backend='ma27',
  historical_profile_commands_preserved=True,backend_change='All six command lists are preserved. Their backend slot is MA27 with explicit HSL; backend-support source changes and measured inertia are declared separately.')
 common.atomic(profiles,catalog)
 if json.loads(profiles.read_text())['profiles']!=old_profiles:raise ValueError('Un contrôle algorithmique de profil a changé')
 runtime_manifest=root/'campaign/CAMPAIGN_RUNTIME_MANIFEST.json'
 common.atomic(runtime_manifest,dict(schema='previous-mega-device-ma27-default-ipopt-runtime-v1',
  historical_device_proof=str(HERE/'SOURCE_AUDIT.json'),files={str(p.relative_to(root/'campaign')):common.sha(p)
   for p in sorted((root/'campaign').rglob('*'))if p.is_file()and p!=runtime_manifest and '__pycache__'not in p.parts}))
 merged={}
 native_dependencies=json.loads(Path(build['native_dependency_proof']).read_text())
 hsl_runtime=json.loads(Path(build['hsl_runtime_proof']).read_text())
 libraries={**native_dependencies.get('all_loaded_library_sha256',{}),**hsl_runtime.get('dependency_sha256',{}),
  defaults['ipopt_library']:defaults['ipopt_library_sha256'],defaults['hsl_library']:defaults['hsl_library_sha256']}
 env=dict(build['runtime_environment']);env['HSL_LIBRARY']=defaults['hsl_library']
 ldpaths=list(dict.fromkeys([*env.get('LD_LIBRARY_PATH','').split(':'),str(Path(defaults['ipopt_library']).parent),str(Path(defaults['hsl_library']).parent)]))
 env['LD_LIBRARY_PATH']=':'.join(p for p in ldpaths if p)
 merged.update(success=True,API=4390,native_solver_api=4390,native_solver_release='qualified-source-MA27-backend-variant-20261108',
  native_al_fallback_policy='off',primary_standard_policy='auto',native_backend='ma27',backend='ma27',
  binary=str(ref),binary_sha256=common.sha(ref),native_backend_qualification=str(qpath),native_backend_qualification_sha256=common.sha(qpath),
  native_qualification=qualification,native_build=str(Path(build['_proof']).resolve()),native_build_sha256=common.sha(build['_proof']),
  native_auditor=str(root/'campaign/common/original_auditor.py'),native_auditor_sha256=common.sha(root/'campaign/common/original_auditor.py'),
  precise_auditor=str(root/'validation_tools/precise_original_auditor.py'),precise_auditor_sha256=common.sha(root/'validation_tools/precise_original_auditor.py'),
  python=sys.executable,ipopt_binary=defaults['default_binary'],ipopt_binary_sha256=defaults['default_binary_sha256'],
  ipopt_version='3.14.11',ipopt_linear_solver='ma27',ipopt_mumps_linear_solver='mumps',ipopt_mumps_pivot_order=7,
  ipopt_mumps_ordering_policy='library-default',ipopt_methods={'ipopt_qcqp':'ma27','ipopt_mumps':'mumps'},
  ipopt_defaults_installation=defaults['_path'],ipopt_defaults_installation_sha256=common.sha(defaults['_path']),
  ipopt_build_manifest=defaults['build'],ipopt_build_manifest_sha256=defaults['build_sha256'],
  ma27_qualification=defaults['qualification'],ma27_qualification_sha256=defaults['qualification_sha256'],
  mumps_qualification=defaults['qualification'],mumps_qualification_sha256=defaults['qualification_sha256'],
  hsl_library=defaults['hsl_library'],hsl_library_sha256=defaults['hsl_library_sha256'],
  environment=env,library_roots=ldpaths,runtime_dependency_sha256=libraries,
  historical_numeric_reuse_allowed=False,excluded_datasets=['pooling_dey'],
  task_budgets=dict(seconds=7200,max_outer=5000,shared_restoration=600,tolerance=1e-6),
  native_profile_backend_override='ma27',native_compilation_reused_byte_exact=False,
  backend_variant='MA27 backend support; existing guards receive its measured inertia; no historical MUMPS result is reused',
  source_installation=str(oldroot/'INSTALLATION.json'),source_installation_sha256=common.sha(oldroot/'INSTALLATION.json'),
  previous_device_source_proof=str(HERE/'SOURCE_AUDIT.json'),
  previous_device_source_proof_sha256=common.sha(HERE/'SOURCE_AUDIT.json'),
  frozen_input_resolution=defaults['frozen_resolution'],frozen_input_resolution_sha256=defaults['frozen_resolution_sha256'])
 merged['ipopt_command']=['{python}',str(root/'campaign/ipopt_worker.py'),'--case-json','{case_json}',
  '--input','{input}','--endpoint','{endpoint}','--seconds','{seconds}','--max-outer','{max_outer}']
 if include_factorial:
  if factorial is None:raise ValueError('Contrôles factoriels requis pour ce mode')
  merged.update(factorial_installation=str(factorial),factorial_installation_sha256=common.sha(factorial))
 else:
  merged.pop('factorial_installation',None);merged.pop('factorial_installation_sha256',None)
 common.atomic(root/'INSTALLATION.json',merged)
 common.atomic(root/'SOURCE_MANIFEST.json',dict(schema='native-ma27-reference-source-manifest-v1',source_sha256=build['ma27_reference_source_sha256'],historical_source_sha256=build['baseline_source_sha256'],ma27_admission=build['ma27_admissions']['baseline']))
 common.atomic(root/'native/BUILD.json',build)
 old_cases=prepared['cases'];by_id={c['case_id']:c for c in common.contract(DEFAULT/'CONTRAT_4078.json.gz')['cases']}
 bindings=defaults['frozen_bindings']
 if len(old_cases)!=4078 or {c['id']for c in old_cases}!=set(by_id):raise ValueError('Le corpus précédent diffère des 4 078 cas publiés')
 for c in old_cases:
  published=by_id[c['id']]
  if c['input_sha256']!=published['input_sha256'] or Path(c['input']).resolve()!=Path(bindings[c['id']]['input']).resolve():
   raise ValueError('Chemin gelé ou empreinte du dispositif précédent différent : '+c['id'])
 inventory=copy.deepcopy(prepared)
 inventory.update(cases=old_cases,beamforming_reader_report=prepared.get('beamforming_file',prepared.get('beamforming_reader_report')))
 common.atomic(root/'FROZEN_INVENTORY.json',inventory)
 common.atomic(root/'INPUT_IMPORT_COMPLETE.json',dict(schema='frozen-input-reuse-no-copy-v1',source_root=str(oldroot),source_preparation_sha256=common.sha(oldroot/'PREPARATION.json'),cases=4078,input_files_copied=0,input_files_modified=False,raw_starts_preserved=True,resolution_receipt=defaults['frozen_resolution'],resolution_sha256=defaults['frozen_resolution_sha256']))
 # Qualify exactly the previous independent scalar/interval certificate device
 # and all published input versions, then use its original binding routine.
 gate=load('mega_new_audit_gate',root/'validation_tools/audit_certificate_gate.py')
 with common.phase('LECTEURS ET CERTIFICATS ORIGINAUX — 4 078 CAS'):
  gate.qualify(root,prepared=inventory,require_complete=True)
 sys.path.insert(0,str(root));sys.path.insert(0,str(root/'campaign'))
 # Ensure these names refer to this new device, not an earlier module cache.
 for name in ('MEGA','executor','setup_runtime','prepare_inventory','dataset_schedule','progress'):
  sys.modules.pop(name,None)
 import MEGA,prepare_inventory
 with common.phase('LIAISON DU DISPOSITIF MÉGA PRÉCÉDENT ET DES NOUVELLES PREUVES'):
  prepare_inventory.bind_preparation(root,merged,inventory)
 return merged


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--with-factorial',action='store_true');p.add_argument('--build-jobs',type=int,default=3)
 a=p.parse_args();oldroot=a.root.resolve();out=a.out.resolve()
 if out.exists():raise ValueError('Répertoire de préparation neuf requis : '+str(out))
 if not(oldroot/'PREPARATION.json').is_file()or not(oldroot/'INSTALLATION.json').is_file():raise ValueError('Dispositif précédent incomplet : '+str(oldroot))
 oldcfg=json.loads((oldroot/'INSTALLATION.json').read_text());prepared=json.loads((oldroot/'PREPARATION.json').read_text())
 before=common.protected(oldroot,oldcfg);before[str(oldroot/'PREPARATION.json')]=common.sha(oldroot/'PREPARATION.json')
 hsl=common.resolve_hsl(oldroot,oldcfg)
 env=dict(os.environ);env.update(oldcfg.get('environment',{}));env['HSL_LIBRARY']=str(hsl)
 for key in ('LD_PRELOAD','LD_AUDIT','LD_DEBUG','LD_DEBUG_OUTPUT'):env.pop(key,None)
 out.mkdir(parents=True)
 try:
  run([sys.executable,'-u',DEFAULT/'PREPARER.py','--root',oldroot,'--out',out/'defaults'],
   'IPOPT PAR DÉFAUT — MA27 ET MUMPS SUR TOUT LE CORPUS',env)
  defaults=json.loads((out/'defaults/INSTALLATION.json').read_text());defaults['_path']=str(out/'defaults/INSTALLATION.json')
  guard=load('mega_prepare_cpu_lock',DEFAULT/'CAMPAGNE.py')
  with guard.machine_lock():
   run([sys.executable,'-u',FACTOR/'BUILD_NATIVE.py','--root',oldroot,
    '--qualified-source',FACTOR/'qualified_source','--qualified-manifest',FACTOR/'QUALIFIED_BASELINE.json',
    '--ma27-adapter',HERE/'ADAPTER_MA27.py','--hsl-library',hsl,'--out',out/'native_build','--jobs',a.build_jobs,
    '--qualify-ma27-auditor',HERE/'campaign/common/original_auditor.py','--qualification-cpu','0'],
    'COMPILATION MA27 ET QUALIFICATION RÉELLE DES SIX VARIANTES LAZY',env)
  build=json.loads((out/'native_build/BUILD_NATIVE.json').read_text());build['_proof']=str(out/'native_build/BUILD_NATIVE.json')
  factorial=None
  if a.with_factorial:
   run([sys.executable,'-u',HERE/'PREPARER_FACTORIEL_MA27.py','--installation',out/'defaults/INSTALLATION.json',
    '--build',out/'native_build/BUILD_NATIVE.json','--out',out/'factorial'],
    'EXPÉRIENCE CROISÉE — CONTRÔLE STRUCTUREL ET 50 DIAGONALES MA27',env)
   factorial=out/'factorial/FACTORIAL_INSTALLATION.json'
  with common.phase('INSTALLATION DU DISPOSITIF MÉGA PRÉCÉDENT — SANS RECOPIE DES QCQP'):
   cfg=stage_previous_device(out/'device',oldroot,oldcfg,prepared,defaults,build,a.with_factorial,factorial)
  if any(common.sha(path)!=value for path,value in before.items()):raise ValueError('L’installation précédente a changé pendant la préparation')
  common.atomic(out/'READY.json',dict(passed=True,root=str(out/'device'),python=sys.executable,tasks=39960 if a.with_factorial else 32624,
   native_methods=['R3','HH','HS','SH','SS','SS_legacy'],native_backend='ma27',ipopt_backends=['ma27','mumps'],
   ipopt_protocol='library-defaults',source_root=str(oldroot),source_unchanged=True,with_factorial=a.with_factorial))
  print('DISPOSITIF MÉGA PRÊT :',out/'device',';',39960 if a.with_factorial else 32624,'tâches, 22 créneaux continus',flush=True)
 except BaseException as error:
  common.atomic(out/'PREPARATION_ERROR.json',dict(error=type(error).__name__+': '+str(error),old_installation_unchanged=all(common.sha(p)==v for p,v in before.items()),benchmark_launched=False))
  raise
if __name__=='__main__':
 try:main()
 except Exception as error:print('PRÉPARATION MÉGA REFUSÉE — '+str(error),file=sys.stderr,flush=True);raise SystemExit(2)
