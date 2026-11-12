#!/usr/bin/env python3
"""Compile all native translation units in an isolated directory.
The same compiler, flags and dependency roots must build baseline, observer and
research binaries. No stale object reuse, numerical shortcut or Ipopt linkage.
"""
from pathlib import Path
import argparse, concurrent.futures, hashlib, importlib.util, json, os, subprocess, sys, time
FILES='condensed_retraction curvature direct_ipopt inner_ipm inner_lazyqp inner_socp inner_ipopt inner_sqp linear_solver lazy_newton atomic_newton io_compact inner_sparse_single_loop original_kkt_gmres io restoration rank_presolve free_epigraph_presolve solver main'.split()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def build(source:Path,out:Path,dependency_root:Path,boost_include:Path,jobs:int=3,compiler='g++',native_config=None):
    source=source.resolve();out=out.resolve();dependency_root=dependency_root.resolve();boost_include=boost_include.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing stale build directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    roots=[Path(p)for p in native_config['include_roots']]if native_config else [dependency_root/'include/coin-or/mumps',dependency_root/'include/coin-or/metis']
    wanted=native_config['compile_flags']if native_config else []
    need_mumps=not native_config or any('LAZYIPDCA_HAVE_MUMPS' in x for x in wanted)
    if need_mumps and not any((r/'dmumps_c.h').is_file()for r in roots):raise RuntimeError('Required MUMPS development headers absent')
    if not any((r/'metis.h').is_file()for r in roots):raise RuntimeError('Required METIS development headers absent')
    if not(boost_include/'boost/multiprecision/cpp_int.hpp').is_file():raise RuntimeError('Boost.Multiprecision headers absent: set --boost-include')
    flags=[compiler]+(native_config['compile_flags']if native_config else ['-std=c++17','-O3','-DNDEBUG','-fopenmp','-DLAZYIPDCA_HAVE_MUMPS=1','-DLAZYIPDCA_HAVE_METIS=1','-DLAZYIPDCA_HAVE_OPENMP=1'])+['-I'+str(p)for p in [source/'include',boost_include]+roots]
    start=time.monotonic();commands=[]
    def one(n):
        cmd=flags+['-c',str(source/'src'/f'{n}.cpp'),'-o',str(out/f'{n}.o')]
        p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/f'{n}.compile.log').write_text(p.stdout);print('NATIVE_COMPILE',out.name,n,p.returncode,flush=True)
        return {'name':n,'command':cmd,'returncode':p.returncode}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs)as ex:commands=list(ex.map(one,FILES))
    if any(c['returncode']for c in commands):raise RuntimeError('Native compilation failed; inspect *.compile.log in '+str(out))
    binary=out/'lazyipdca_qcqp_cli';link=[compiler,'-fopenmp']+[str(out/f'{f}.o')for f in FILES]+['-L'+str(dependency_root),'-Wl,-rpath,'+str(dependency_root),'-lcoinmumps','-lcoinmetis','-lcasadi-tp-openblas','-ldl','-o',str(binary)]
    if native_config:link=[compiler,'-fopenmp']+[str(out/f'{f}.o')for f in FILES]+['-Wl,-rpath,'+str(p)for p in sorted({Path(v).parent for v in native_config['link_libraries']})]+native_config['link_libraries']+['-ldl','-o',str(binary)]
    p=subprocess.run(link,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/'link.log').write_text(p.stdout)
    if p.returncode:raise RuntimeError('Native link failed: '+str(out/'link.log'))
    dependencies={str(p):sha(p)for p in map(Path,native_config['link_libraries'])}if native_config else {str(p):sha(p)for name in ('libcoinmumps.so','libcoinmetis.so','libcasadi-tp-openblas.so')for p in [dependency_root/name]}
    source_files={str(p.relative_to(source)):sha(p)for p in sorted(source.rglob('*'))if p.is_file()}
    receipt={'schema':'lazy-factorial-native-build-v1','source':str(source),'source_sha256':source_files,'binary':str(binary),'binary_sha256':sha(binary),'homogeneous_compilation':True,'native_backend':native_config.get('native_backend')if native_config else'mumps','hsl_adapter_compiled':any('LAZYIPDCA_HAVE_HSL=1'in x for x in flags),'translation_units':commands,'link_command':link,'dependencies':dependencies,'compiler_version':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],'seconds':time.monotonic()-start,'version':subprocess.check_output([str(binary),'--version'],text=True),'backends':subprocess.check_output([str(binary),'--linear-backends'],text=True)}
    (out/'BUILD_NATIVE.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--dependency-root',type=Path);p.add_argument('--boost-include',type=Path,default=Path('/usr/include'));p.add_argument('--jobs',type=int,default=3);p.add_argument('--compiler',default='g++');a=p.parse_args()
    if a.dependency_root is None:
        spec=importlib.util.find_spec('casadi')
        if spec is None:raise SystemExit('CasADi native libraries absent in this Python; use the qualified runtime or --dependency-root')
        a.dependency_root=Path(spec.origin).parent
    result=build(a.source,a.out,a.dependency_root,a.boost_include,a.jobs,a.compiler);print(json.dumps({k:result[k]for k in ('binary','binary_sha256','homogeneous_compilation','seconds')},indent=2))
