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

