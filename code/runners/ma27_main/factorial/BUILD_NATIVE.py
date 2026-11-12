#!/usr/bin/env python3
"""Build three isolated native binaries using the installed native dependencies.
No installation or old binary is modified. An unresolved original dependency
stops preparation; this script never substitutes the Ipopt comparator's libs.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,shlex,subprocess,sys
HERE=Path(__file__).resolve().parent

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
compiler=load('native_compiler',HERE/'COMPILER_NATIVE.py');observer=load('native_observer',HERE/'PREPARER_OBSERVATEUR.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources(path):return {str(p.relative_to(path)):sha(p)for p in sorted(path.rglob('*'))if p.is_file()}
def cache_read(path):
 out={}
 for line in path.read_text(errors='replace').splitlines():
  if line and not line.startswith(('#','//'))and'='in line:
   k,v=line.split('=',1);out[k.split(':',1)[0]]=v
 return out

def pin(root,binary,boost_override=None,retain_auxiliary_mumps=False):
 version=subprocess.check_output([str(binary),'--version'],text=True);backends=subprocess.check_output([str(binary),'--linear-backends'],text=True)
 if 'mumps'not in backends and 'ma27'not in backends:raise RuntimeError('Original native binary has no supported sparse backend')
 p=subprocess.run(['ldd',str(binary)],capture_output=True,text=True)
 if p.returncode or'not found'in p.stdout:raise RuntimeError('Original native dependency resolution failed: '+p.stdout+p.stderr)
 libs={m.group(1):Path(m.group(2)).resolve()for m in re.finditer(r'^\s*(\S+)\s+=>\s+(/\S+)\s+\(',p.stdout,re.M)}
 mumps=[v for k,v in libs.items()if re.search(r'(coinmumps|dmumps)',k)]
 metis=[v for k,v in libs.items()if re.search(r'(coinmetis|libmetis)',k)]
 if len(metis)!=1 or len(mumps)>1:raise RuntimeError('Cannot uniquely pin original sparse linear algebra from ldd')
 caches=[]
 for cp in sorted(root.rglob('CMakeCache.txt')):
  if any(x in cp.parts for x in ('data','runtime','test_data')):continue
  c=cache_read(cp)
  ml=c.get('MUMPS_LIBRARY','')
  if ml and Path(ml).exists()and mumps and Path(ml).resolve()==mumps[0]:caches.append((cp,c))
 if len(caches)>1:
  signatures={tuple(sorted((k,v)for k,v in c.items()if k.startswith(('CMAKE_CXX','MUMPS_','METIS_','LAZYIPDCA_WITH_','LAZYIPDCA_NATIVE_ARCH'))))for _,c in caches}
  if len(signatures)>1:raise RuntimeError('Ambiguous original native CMake configurations')
 cp,c=caches[0]if caches else(None,{})
 include=[]
 for key in ('MUMPS_INCLUDE_DIR','METIS_INCLUDE_DIR'):
  if c.get(key)and Path(c[key]).is_dir():include.append(Path(c[key]).resolve())
 candidate_roots={v.parent for v in mumps+metis}|{v.parent.parent for v in mumps+metis}
 for r in candidate_roots:
  for suf in ('include/coin-or/mumps','include/coin-or/metis','include','include/mumps','include/metis','coin-or/mumps','coin-or/metis'):
   d=r/suf
   if(d/'dmumps_c.h').is_file()or(d/'metis.h').is_file():include.append(d.resolve())
 for d in (Path('/usr/include'),Path('/usr/include/mumps'),Path('/usr/include/metis')):
  if(d/'dmumps_c.h').is_file()or(d/'metis.h').is_file():include.append(d)
 include=list(dict.fromkeys(include))
 if not any((r/'metis.h').is_file()for r in include):raise RuntimeError('Pinned METIS headers absent')
 if retain_auxiliary_mumps and(not mumps or not any((r/'dmumps_c.h').is_file()for r in include)):raise RuntimeError('Explicit auxiliary MUMPS requested but pinned library/header unavailable')
 boost=boost_override or Path(c.get('LAZYIPDCA_BOOST_INCLUDE_DIR','/usr/include'))
 if not(boost/'boost/multiprecision/cpp_int.hpp').is_file():raise RuntimeError('Boost headers absent; install libboost-dev or pass --boost-include')
 flags=['-std=c++17']+shlex.split(c.get('CMAKE_CXX_FLAGS',''))+shlex.split(c.get('CMAKE_CXX_FLAGS_RELEASE','-O3 -DNDEBUG'))
 if any(x in flags for x in ('-ffast-math','-Ofast','-funsafe-math-optimizations','-fassociative-math')):raise RuntimeError('Reassociating floating-point build flags are incompatible with this qualification')
 if c.get('LAZYIPDCA_NATIVE_ARCH')=='ON':flags+=['-march=native']
 flags=[f for f in flags if 'LAZYIPDCA_HAVE_MUMPS'not in f and 'LAZYIPDCA_HAVE_HSL'not in f]
 flags+=['-fopenmp','-DLAZYIPDCA_HAVE_HSL=1','-DLAZYIPDCA_HAVE_METIS=1','-DLAZYIPDCA_HAVE_OPENMP=1']
 if retain_auxiliary_mumps:flags+=['-DLAZYIPDCA_HAVE_MUMPS=1']
 # Explicit dependencies of the original executable, never those of Ipopt.
 selected={v for k,v in libs.items()if re.search(r'(metis|lapack|blas)',k)or(retain_auxiliary_mumps and re.search(r'(mumps|pord|mpiseq)',k))}
 if not selected:raise RuntimeError('No pinned native linear algebra libraries')
 return {'native_backend':'ma27','mumps_compiled':retain_auxiliary_mumps,'original_binary':str(binary),'original_binary_sha256':sha(binary),'version':version,'backends':backends,'ldd':p.stdout,'ldd_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'all_loaded_library_sha256':{str(v):sha(v)for v in libs.values()},'cmake_cache':str(cp)if cp else None,'cmake_cache_sha256':sha(cp)if cp else None,'compile_flags':flags,'include_roots':[str(x)for x in include],'link_libraries':[str(v)for v in sorted(selected)],'boost_include':str(boost),'compiler':c.get('CMAKE_CXX_COMPILER','g++'),'flags_provenance':'original native CMake cache'if cp else'C++17 release strict-FP defaults; exact parity gate mandatory'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--qualified-source',required=True,type=Path);p.add_argument('--qualified-manifest',required=True,type=Path);p.add_argument('--baseline-binary',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--boost-include',type=Path);p.add_argument('--jobs',type=int,default=3);p.add_argument('--hsl-library',type=Path,default=os.environ.get('HSL_LIBRARY'));p.add_argument('--ma27-adapter',type=Path);p.add_argument('--qualify-ma27-auditor',type=Path);p.add_argument('--qualification-cpu',type=int);a=p.parse_args()
 root=a.root.resolve();base=a.qualified_source.resolve();out=a.out.resolve();historical=(a.baseline_binary or root/'install/bin/lazyipdca_qcqp_cli').resolve()
 if not a.hsl_library:raise RuntimeError('The real licensed HSL library must be supplied with --hsl-library')
 if a.ma27_adapter is None:
  a.ma27_adapter=next((x for x in (HERE.parent/'ADAPTER_MA27.py',HERE.parent.parent/'mega_ma27/ADAPTER_MA27.py')if x.is_file()),None)
 if a.ma27_adapter is None:raise RuntimeError('Explicit MA27 source-admission adapter is absent')
 if out.exists()and any(out.iterdir()):raise RuntimeError('Refusing an existing nonempty isolated build target')
 out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads(a.qualified_manifest.read_text());expected=manifest.get('source_sha256',manifest);got=sources(base)
 if got!=expected or len(got)!=823:raise RuntimeError('Qualified original 823-source manifest mismatch')
 runtime=load('native_ma27_runtime',HERE/'MA27_RUNTIME.py');hsl=runtime.pin_hsl(a.hsl_library)
 (out/'HSL_RUNTIME.json').write_text(json.dumps(hsl,indent=2)+'\n')
 historical_sha=sha(historical);native=pin(root,historical,a.boost_include,retain_auxiliary_mumps=False)
 native.update(hsl_library=hsl['hsl_library'],hsl_library_sha256=hsl['hsl_library_sha256'],hsl_runtime_proof=str(out/'HSL_RUNTIME.json'),hsl_runtime_proof_sha256=sha(out/'HSL_RUNTIME.json'),mumps_compiled=False)
 runtime_dirs=list(dict.fromkeys([str(Path(x).parent)for x in list(native['all_loaded_library_sha256'])+list(hsl['dependency_sha256'])]+[str(Path(hsl['hsl_library']).parent)]))
 native['runtime_environment']={'LD_LIBRARY_PATH':':'.join(runtime_dirs),'HSL_LIBRARY':hsl['hsl_library'],**{k:'1'for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
 (out/'DEPENDANCES_NATIVES.json').write_text(json.dumps(native,indent=2)+'\n')
 adapter=load('native_ma27_admission',a.ma27_adapter)
 print('MA27_PREPARATION reference823 et recherche825 — copies isolees',flush=True)
 reference_source=out/'ma27_reference_source';research_source=out/'ma27_research_source'
 reference_admission=adapter.prepare(base,reference_source);research_admission=adapter.prepare(HERE/'source',research_source)
 obs=out/'baseline_trace_source';obsproof=observer.prepare(reference_source,research_source,obs)
 blds={}
 for name,src in [('reference',reference_source),('baseline_trace',obs),('research',research_source)]:
  print('MA27_COMPILATION',name,'— demarrage',flush=True)
  blds[name]=compiler.build(src,out/name,Path(native['link_libraries'][0]).parent,Path(native['boost_include']),a.jobs,native['compiler'],native)
  if 'ma27'not in blds[name]['backends']or'mumps'in blds[name]['backends']:raise RuntimeError('Fresh native build must expose MA27 and omit MUMPS')
  if sha(historical)!=historical_sha:raise RuntimeError('Historical installed binary changed during isolated build')
 result={'schema':'lazy-factorial-three-native-builds-ma27-v2','native_backend':'ma27','historical_installed_binary':str(historical),'historical_installed_binary_sha256':historical_sha,'baseline_binary':blds['reference']['binary'],'baseline_binary_sha256':blds['reference']['binary_sha256'],'qualified_rebuilt_binary':blds['reference']['binary'],'qualified_rebuilt_binary_sha256':blds['reference']['binary_sha256'],'baseline_trace_binary':blds['baseline_trace']['binary'],'baseline_trace_binary_sha256':blds['baseline_trace']['binary_sha256'],'research_binary':blds['research']['binary'],'research_binary_sha256':blds['research']['binary_sha256'],'baseline_source_sha256':got,'ma27_baseline_source_sha256':blds['reference']['source_sha256'],'original_research_source_sha256':sources(HERE/'source'),'baseline_trace_source_sha256':blds['baseline_trace']['source_sha256'],'research_source_sha256':blds['research']['source_sha256'],'source_directories':{'baseline':str(reference_source),'baseline_trace':str(obs),'research':str(research_source)},'ma27_admissions':{'baseline':reference_admission,'research':research_admission},'ma27_adapter':str(a.ma27_adapter.resolve()),'ma27_adapter_sha256':sha(a.ma27_adapter),'observer_source_proof':obsproof,'native_dependency_proof':str(out/'DEPENDANCES_NATIVES.json'),'native_dependency_proof_sha256':sha(out/'DEPENDANCES_NATIVES.json'),'hsl_library':hsl['hsl_library'],'hsl_library_sha256':hsl['hsl_library_sha256'],'hsl_runtime_proof':str(out/'HSL_RUNTIME.json'),'hsl_runtime_proof_sha256':sha(out/'HSL_RUNTIME.json'),'runtime_environment':native['runtime_environment'],'mumps_compiled':False,'all_binaries_homogeneous':True,'profiles_factory':str(HERE/'PROFILES.py'),'profiles_factory_sha256':sha(HERE/'PROFILES.py'),'qualification_50_passed':False,'native_ma27_qualified':False,'baseline_scope':'Fresh same-backend MA27 backend-support reference; existing guards now receive actual computed inertia; historical MUMPS trajectory is not a numerical baseline'}
 result.update(ma27_reference_binary=result['baseline_binary'],ma27_reference_binary_sha256=result['baseline_binary_sha256'],ma27_reference_source=str(reference_source),ma27_reference_source_sha256=result['ma27_baseline_source_sha256'],baseline_trace_source=str(obs),research_source=str(research_source),original_qualified_source=str(base),original_qualified_source_sha256=got)
 if a.qualify_ma27_auditor is not None:
  qualification=load('native_ma27_qualification',HERE/'QUALIFIER_MA27.py').qualify(Path(result['baseline_binary']),Path(result['hsl_library']),a.qualify_ma27_auditor,out/'native_ma27_qualification',a.qualification_cpu,native['runtime_environment'])
  result.update(native_ma27_qualification=str(out/'native_ma27_qualification/QUALIFICATION_MA27.json'),native_ma27_qualification_sha256=sha(out/'native_ma27_qualification/QUALIFICATION_MA27.json'),native_ma27_qualified=qualification['passed'])
 (out/'BUILD_NATIVE.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items()if k.endswith('binary')or k in ('qualification_50_passed','native_ma27_qualified')},indent=2))
 if a.qualify_ma27_auditor is not None and not result['native_ma27_qualified']:raise RuntimeError('Real MA27 six-profile qualification did not pass; production remains blocked')
if __name__=='__main__':
 try:main()
 except Exception as e:print('BUILD_NATIVE REFUSÉ — '+str(e),file=sys.stderr);raise SystemExit(2)
