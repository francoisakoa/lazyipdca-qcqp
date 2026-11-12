#!/usr/bin/env python3
"""Isolated homogeneous build for the refinement experiment; no solve or input copy."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,time
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def read_bound(path,expected=None):
 path=Path(path).expanduser().resolve(strict=True)
 if expected and sha(path)!=expected:raise ValueError('Empreinte différente : '+str(path))
 return json.loads(path.read_text())

def source_map(root):return {str(p.relative_to(root)):sha(p)for p in sorted(root.rglob('*'))if p.is_file()}

def ldd(binary,environment=None):
 p=subprocess.run(['ldd',str(binary)],capture_output=True,text=True,env=environment)
 if p.returncode or 'not found'in p.stdout:raise ValueError('Dépendance native absente : '+str(binary)+'\n'+p.stdout+p.stderr)
 libs={m.group(1):Path(m.group(2)).resolve()for m in re.finditer(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(',p.stdout,re.M)}
 if not libs:raise ValueError('Aucune dépendance native dynamique identifiée : '+str(binary))
 return libs,p.stdout

def select_sparse_libraries(primary,mumps):
 ml={p for n,p in mumps.items()if re.search(r'(coinmumps|dmumps)',n)}
 if len(ml)!=1:raise ValueError('La référence native MUMPS doit déclarer une seule bibliothèque MUMPS')
 metis={p for libs in(primary,mumps)for n,p in libs.items()if re.search(r'(coinmetis|libmetis)',n)}
 if len(metis)!=1:raise ValueError('Les deux références natives doivent partager une seule bibliothèque METIS explicite')
 selected={p for libs in(primary,mumps)for n,p in libs.items()if re.search(r'(coinmumps|dmumps|coinmetis|libmetis|openblas|libblas|lapack|pord|mpiseq)',n)}
 if not selected:raise ValueError('Bibliothèques natives de factorisation absentes')
 return sorted(selected)

def strict_flags(flags):
 banned={'-ffast-math','-Ofast','-funsafe-math-optimizations','-fassociative-math','-ffinite-math-only'}
 if any(x in banned for x in flags):raise ValueError('Drapeau de calcul flottant incompatible avec la qualification')
 out=[x for x in flags if not any(k in x for k in('LAZYIPDCA_HAVE_MUMPS','LAZYIPDCA_HAVE_HSL','LAZYIPDCA_HAVE_METIS','LAZYIPDCA_HAVE_OPENMP'))]
 if not any(x.startswith('-std=')for x in out):out.insert(0,'-std=c++17')
 if '-fopenmp'not in out:out.append('-fopenmp')
 return out+['-DLAZYIPDCA_HAVE_MUMPS=1','-DLAZYIPDCA_HAVE_HSL=1','-DLAZYIPDCA_HAVE_METIS=1','-DLAZYIPDCA_HAVE_OPENMP=1']

def pin_dependencies(header,mumps_override=None,hsl_override=None,boost_override=None):
 inst=header['installation'];original=Path(inst['binary']).resolve(strict=True)
 if sha(original)!=inst['binary_sha256']:raise ValueError('Le binaire source ne correspond pas au plan transmis')
 # The old source installation is used only to identify its native MUMPS
 # dependencies. No old numerical endpoint is reused by this experiment.
 old=read_bound(inst['source_installation'],inst['source_installation_sha256'])
 mb=mumps_override or old.get('binary')
 if not mb:raise ValueError('Binaire natif MUMPS absent du reçu source ; utiliser --mumps-reference-binary')
 mb=Path(mb).expanduser().resolve(strict=True)
 if not old.get('binary_sha256')or sha(mb)!=old['binary_sha256']:raise ValueError('La référence de dépendances MUMPS a changé')
 build=read_bound(inst['native_build'],inst['native_build_sha256'])
 deps=read_bound(build['native_dependency_proof'],build.get('native_dependency_proof_sha256'))
 declared_environment=dict(os.environ)
 declared_environment.update(old.get('environment',{}));declared_environment.update(inst.get('environment',{}))
 primary,primary_ldd=ldd(original,declared_environment);mumps,mumps_ldd=ldd(mb,declared_environment);selected=select_sparse_libraries(primary,mumps)
 roots=[Path(p).resolve()for p in deps.get('include_roots',[])if Path(p).is_dir()]
 for lib in selected:
  for parent in (lib.parent,lib.parent.parent):
   for suffix in ('include','include/coin-or/mumps','include/coin-or/metis','include/mumps','include/metis'):
    p=parent/suffix
    if(p/'dmumps_c.h').is_file()or(p/'metis.h').is_file():roots.append(p.resolve())
 for p in (Path('/usr/include'),Path('/usr/include/mumps'),Path('/usr/include/metis')):
  if(p/'dmumps_c.h').is_file()or(p/'metis.h').is_file():roots.append(p)
 roots=list(dict.fromkeys(roots))
 if not any((p/'dmumps_c.h').is_file()for p in roots)or not any((p/'metis.h').is_file()for p in roots):raise ValueError('En-têtes natifs MUMPS/METIS absents des dépendances déclarées')
 boost=Path(boost_override or deps.get('boost_include','/usr/include')).expanduser().resolve()
 if not(boost/'boost/multiprecision/cpp_int.hpp').is_file():raise ValueError('En-têtes Boost.Multiprecision absents ; fournir --boost-include (ou installer libboost-dev)')
 hsl=Path(hsl_override or inst['hsl_library']).expanduser().resolve(strict=True)
 hp=load('real_hsl_runtime',HERE/'native/MA27_RUNTIME.py').pin_hsl(hsl,inst['hsl_library_sha256'])
 env={k:'1'for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')}
 runtime_dirs=list(dict.fromkeys([str(p.parent)for p in selected]+[str(hsl.parent)]+[str(Path(p).parent)for p in hp['dependency_sha256']]))
 env.update(HSL_LIBRARY=str(hsl),LD_LIBRARY_PATH=':'.join(runtime_dirs))
 return dict(schema='refinement-dual-native-dependencies-v1',native_backend='ma27_and_mumps',original_binary=str(original),original_binary_sha256=sha(original),mumps_reference_binary=str(mb),mumps_reference_binary_sha256=sha(mb),original_native_build=str(inst['native_build']),original_native_build_sha256=sha(inst['native_build']),original_source_installation=str(inst['source_installation']),original_source_installation_sha256=sha(inst['source_installation']),all_loaded_library_sha256={str(p):sha(p)for p in set(primary.values())|set(mumps.values())},primary_ldd=primary_ldd,mumps_ldd=mumps_ldd,compile_flags=strict_flags(deps.get('compile_flags',['-std=c++17','-O3','-DNDEBUG'])),include_roots=[str(p)for p in roots],link_libraries=[str(p)for p in selected],boost_include=str(boost),compiler=deps.get('compiler','g++'),runtime_environment=env,hsl_runtime=hp)

def declared_runtime_environment(header,hsl_override=None):
 inst=header['installation'];old=read_bound(inst['source_installation'],inst['source_installation_sha256'])
 env=dict(os.environ);env.update(old.get('environment',{}));env.update(inst.get('environment',{}))
 for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
 hsl=str(Path(hsl_override or inst['hsl_library']).expanduser().resolve())
 dirs=list(dict.fromkeys([str(Path(hsl).parent)]+[p for e in (inst.get('environment',{}),old.get('environment',{}))for p in e.get('LD_LIBRARY_PATH','').split(':')if p]))
 env.update(HSL_LIBRARY=hsl,LD_LIBRARY_PATH=':'.join(dirs))
 return env

def require_isolated_output(out,protected):
 out=Path(out).resolve()
 for value in protected:
  p=Path(value).resolve()
  if out==p or p in out.parents or out in p.parents:raise ValueError('La sortie de compilation doit être séparée des sources et installations : '+str(out))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--plan',type=Path,default=HERE/'evidence/CURRENT_PLAN_COMPACT.json');p.add_argument('--study-contract',type=Path,default=HERE/'STUDY_CONTRACT.json');p.add_argument('--mumps-reference-binary',type=Path);p.add_argument('--hsl-library',type=Path);p.add_argument('--boost-include',type=Path);p.add_argument('--jobs',type=int,default=1);p.add_argument('--compiler');a=p.parse_args()
 if a.jobs!=1:raise ValueError('Le raffinement est réservé au cœur23 : --jobs 1 requis')
 if 23 not in os.sched_getaffinity(0):raise ValueError('Le cœur23 n’est pas disponible')
 os.sched_setaffinity(0,{23})
 out=a.out.expanduser().resolve()
 if out.exists()and any(out.iterdir()):raise ValueError('Le dossier de compilation doit être neuf : '+str(out))
 plan=read_bound(a.plan);header=plan['plan_header'];read_bound(a.study_contract)
 ref=HERE/'native/reference_source';research=HERE/'native/experiment_source';expected={k.split('/source/',1)[-1]:v for k,v in header['source_sha256'].items()}
 if source_map(ref)!=expected or len(expected)!=823:raise ValueError('Les 823 fichiers de référence ne correspondent pas au plan')
 require_isolated_output(out,[ref,research,Path(header['root']),Path(header['installation']['source_installation']).parent,Path(header['installation']['native_build']).parent])
 compiler=load('experiment_compiler',HERE/'native/COMPILER_NATIVE.py');observer=load('experiment_observer',HERE/'native/PREPARER_OBSERVATEUR.py')
 wanted_environment=declared_runtime_environment(header,a.hsl_library)
 if any(os.environ.get(k)!=wanted_environment.get(k)for k in ('LD_LIBRARY_PATH','HSL_LIBRARY','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')):
  print('ENVIRONNEMENT NATIF — activation des dépendances déclarées',flush=True)
  os.execve(sys.executable,[sys.executable,*sys.argv],wanted_environment)
 print('DÉPENDANCES NATIVES — vérification MA27, MUMPS, METIS, Boost et compilateur',flush=True)
 native=pin_dependencies(header,a.mumps_reference_binary,a.hsl_library,a.boost_include);cc=a.compiler or native['compiler']
 if not shutil.which(cc):raise ValueError('Compilateur C++ absent : '+str(cc))
 require_isolated_output(out,[Path(native['original_binary']).parent,Path(native['mumps_reference_binary']).parent,Path(native['hsl_runtime']['hsl_library']).parent])
 initial_reference_sources=source_map(ref);initial_experiment_sources=source_map(research)
 native['compiler_version']=subprocess.check_output([cc,'--version'],text=True).splitlines()[0]
 out.mkdir(parents=True,exist_ok=True);write(out/'DEPENDENCIES.json',native)
 observed=out/'observer_source';op=observer.prepare(ref,research,observed)
 bins={};start=time.monotonic()
 for label,source in [('reference',ref),('observer',observed),('experiment',research)]:
  print('COMPILATION HOMOGÈNE',label,'— démarrage',flush=True)
  bins[label]=compiler.build(source,out/label,Path(native['link_libraries'][0]).parent,Path(native['boost_include']),a.jobs,cc,native)
  for backend in ('ma27','mumps'):
   if not re.search(r'\b'+backend+r'\b',bins[label]['backends']):raise ValueError('Backend absent du binaire '+label+' : '+backend)
  print('COMPILATION HOMOGÈNE',label,'— terminée',flush=True)
 if sha(native['original_binary'])!=native['original_binary_sha256']or sha(native['original_source_installation'])!=native['original_source_installation_sha256']:raise ValueError('Installation source modifiée pendant la compilation')
 if source_map(ref)!=initial_reference_sources or source_map(research)!=initial_experiment_sources:raise ValueError('Sources modifiées pendant la compilation')
 if source_map(observed)!=bins['observer']['source_sha256']:raise ValueError('Sources observateur modifiées pendant la compilation')
 for lib,h in {**native['all_loaded_library_sha256'],**native['hsl_runtime']['dependency_sha256'],native['hsl_runtime']['hsl_library']:native['hsl_runtime']['hsl_library_sha256']}.items():
  if sha(lib)!=h:raise ValueError('Dépendance modifiée pendant la compilation : '+lib)
 entries={}
 for backend in ('ma27','mumps'):
  row={'environment':native['runtime_environment']}
  for key,label in [('reference','reference'),('observer','observer'),('experiment','experiment')]:row[key+'_binary']=bins[label]['binary'];row[key+'_binary_sha256']=bins[label]['binary_sha256']
  if backend=='ma27':row.update(installed_reference_binary=native['original_binary'],installed_reference_binary_sha256=native['original_binary_sha256'])
  entries[backend]=row
 result=dict(schema='lazy-refinement-build-v1',created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),backends=entries,reference_source_sha256=source_map(ref),experiment_source_sha256=source_map(research),observer_source_sha256=source_map(observed),homogeneous_compilation=True,same_source_and_binary_for_both_backends=True,original_installation_unchanged=True,study_contract_sha256=sha(a.study_contract),plan_sha256=sha(a.plan),compiler=cc,compiler_version=native['compiler_version'],dependency_proof=str(out/'DEPENDENCIES.json'),dependency_proof_sha256=sha(out/'DEPENDENCIES.json'),observer_proof=op,builds=bins,seconds=time.monotonic()-start,qualification_50_passed=False,production_authorized=False,optimizer_invocations=0)
 write(out/'BUILD_EXPERIMENT.json',result);print('COMPILATION PRÊTE — qualification requise :',out/'BUILD_EXPERIMENT.json',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:print('COMPILATION REFUSÉE — '+str(e),file=sys.stderr,flush=True);raise SystemExit(2)
