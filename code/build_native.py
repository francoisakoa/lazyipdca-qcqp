#!/usr/bin/env python3
"""Build a selected frozen profile using external native dependencies."""
from pathlib import Path
import argparse,importlib.util,json,os,shutil,subprocess,sys
from prepare_profile import prepare

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--profile',choices=['mumps','ma27','asu','persistence','refinement'],default='mumps')
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--boost-include',type=Path)
 p.add_argument('--jobs',type=int,default=2)
 p.add_argument('--dependency-root',type=Path,help='Directory containing libcoinmumps.so and libcoinmetis.so (e.g. CasADi)')
 a=p.parse_args();out=a.out.resolve()
 if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty')
 out.mkdir(parents=True,exist_ok=True)
 boost=a.boost_include or next((x for x in [Path('/usr/include'),Path('/usr/local/include'),Path(os.environ.get('CONDA_PREFIX','/nonexistent'))/'include'] if (x/'boost/multiprecision/cpp_int.hpp').is_file()),None)
 dep=a.dependency_root
 if dep is None:
  spec=importlib.util.find_spec('casadi')
  if spec:dep=Path(spec.origin).parent
 required={'compiler':shutil.which('g++'),'cmake':shutil.which('cmake'),'boost_headers':str(boost)if boost else None,'dependency_root':str(dep)if dep else None}
 checks={'dependencies':required,'profile':a.profile,'optimizer_invocations':0,'compiled':False}
 if not all(required.values()):
  checks['error']='Missing build dependencies: '+', '.join(k for k,v in required.items() if not v)
  (out/'BUILD_CHECK.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks));return 2
 dep=dep.resolve()
 mumps_include=dep/'include/coin-or/mumps';metis_include=dep/'include/coin-or/metis'
 for rel in ['libcoinmumps.so','libcoinmetis.so','libcasadi-tp-openblas.so','include/coin-or/mumps/dmumps_c.h','include/coin-or/metis/metis.h']:
  if not(dep/rel).is_file():raise ValueError('Missing external dependency: '+str(dep/rel))
 source=prepare(a.profile,out/'source');build=out/'build'
 command=['cmake','-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DLAZYIPDCA_BUILD_TESTS=OFF','-DLAZYIPDCA_WITH_IPOPT=OFF','-DLAZYIPDCA_WITH_LDL=OFF','-DLAZYIPDCA_WITH_QDLDL=OFF','-DLAZYIPDCA_WITH_MUMPS=ON','-DLAZYIPDCA_WITH_METIS=ON','-DLAZYIPDCA_WITH_HSL=ON','-DLAZYIPDCA_NATIVE_ARCH=OFF','-DLAZYIPDCA_BOOST_INCLUDE_DIR='+str(boost),'-DMUMPS_INCLUDE_DIR='+str(mumps_include),'-DMUMPS_DMUMPS_LIBRARY='+str(dep/'libcoinmumps.so'),'-DMETIS_INCLUDE_DIR='+str(metis_include),'-DMETIS_LIBRARY='+str(dep/'libcoinmetis.so'),'-DBLAS_LIBRARIES='+str(dep/'libcasadi-tp-openblas.so'),'-DLAPACK_LIBRARIES='+str(dep/'libcasadi-tp-openblas.so')]
 env=dict(os.environ,LD_LIBRARY_PATH=str(dep)+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''))
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
 checks['configure_command']=command
 with (out/'configure.log').open('w')as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError('Configuration failed; inspect '+str(out/'configure.log'))
 command=['cmake','--build',str(build),'--target','lazyipdca_qcqp_cli','-j',str(max(1,a.jobs))];checks['build_command']=command
 with (out/'compile.log').open('w')as log:r=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
 checks['compiled']=r.returncode==0
 if r.returncode:checks['error']='Compilation failed; inspect compile.log'
 else:
  binary=build/'lazyipdca_qcqp_cli';checks['binary']=str(binary);checks['version']=subprocess.check_output([str(binary),'--version'],env=env,text=True);checks['backends']=subprocess.check_output([str(binary),'--linear-backends'],env=env,text=True)
 (out/'BUILD_CHECK.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks));return r.returncode
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as error:print(type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(2)
