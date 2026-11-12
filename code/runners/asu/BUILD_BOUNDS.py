#!/usr/bin/env python3
"""Isolated MA27+MUMPS build. No model copy, optimizer invocation or campaign restart."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,re,sys,time
HERE=Path(__file__).resolve().parent
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cpu',type=int,default=0);ap.add_argument('--hsl-library',type=Path);ap.add_argument('--boost-include',type=Path)
    ap.add_argument('--mumps-reference-binary',type=Path);a=ap.parse_args()
    if a.cpu not in os.sched_getaffinity(0):raise ValueError('Cœur indisponible : '+str(a.cpu))
    os.sched_setaffinity(0,{a.cpu})
    out=a.out.expanduser().resolve()
    if out.exists():raise ValueError('Répertoire de construction neuf requis')
    if out==HERE or HERE in out.parents or out in HERE.parents:raise ValueError('Construction séparée du kit requise')
    dep=load('bounds_dependency_tools',HERE/'DEPENDENCIES.py')
    ctx=json.loads((HERE/'BASE_CONTEXT.json').read_text());header=ctx['plan_header']
    env=dep.declared_runtime_environment(header,a.hsl_library)
    if any(os.environ.get(k)!=env.get(k)for k in ['LD_LIBRARY_PATH','HSL_LIBRARY','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']):
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    lock_name='0_21'if a.cpu in range(22)else str(a.cpu)
    lock_path=Path('/tmp/LAZYQCQP_CPU_'+lock_name+'.lock')
    try:fd=os.open(lock_path,os.O_RDWR|os.O_CREAT,0o666)
    except PermissionError:fd=os.open(lock_path,os.O_RDONLY)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:os.close(fd);raise ValueError('Cœur de compilation réservé par un contrôleur ; attendre sa fin')
    native=dep.pin_dependencies(header,a.mumps_reference_binary,a.hsl_library,a.boost_include)
    out.mkdir(parents=True);dep.write(out/'DEPENDENCIES.json',native)
    patch=load('bounds_source_patch',HERE/'PATCH_SOURCE.py');receipt=patch.patch(out/'research_source')
    compiler=load('bounds_compiler',HERE/'native/COMPILER_NATIVE.py');start=time.monotonic()
    reference=compiler.build(HERE/'native/reference_source',out/'build_reference',Path(native['link_libraries'][0]).parent,Path(native['boost_include']),1,native['compiler'],native)
    build=compiler.build(out/'research_source',out/'build',Path(native['link_libraries'][0]).parent,Path(native['boost_include']),1,native['compiler'],native)
    if not all(re.search(r'\b'+b+r'\b',reference['backends'])for b in ['ma27','mumps']):raise ValueError('Deux backends de référence requis')
    if not all(re.search(r'\b'+b+r'\b',build['backends'])for b in ['ma27','mumps']):raise ValueError('Deux backends natifs requis')
    result=dict(schema='lazy-bound-removal-build-v2',reference_binary=reference['binary'],reference_binary_sha256=reference['binary_sha256'],reference_build=str(out/'build_reference/BUILD_NATIVE.json'),reference_build_sha256=sha(out/'build_reference/BUILD_NATIVE.json'),binary=build['binary'],binary_sha256=build['binary_sha256'],
      source_root=str(out/'research_source'),reference_source_root=str(HERE/'native/reference_source'),
      source_sha256=receipt['source_research_sha256'],base_source_sha256=receipt['source_reference_sha256'],
      patch_receipt=str(out/'PATCH_RECEIPT.json'),patch_receipt_sha256=sha(out/'PATCH_RECEIPT.json'),
      dependencies=str(out/'DEPENDENCIES.json'),dependencies_sha256=sha(out/'DEPENDENCIES.json'),
      runtime_environment=native['runtime_environment'],hsl_library=native['hsl_runtime']['hsl_library'],hsl_library_sha256=native['hsl_runtime']['hsl_library_sha256'],
      same_binary_for_both_backends=True,homogeneous_compilation=True,research_variant=True,
      original_installation_unchanged=True,optimizer_invocations=0,seconds=time.monotonic()-start)
    dep.write(out/'BUILD_BOUNDS.json',result)
    print('CONSTRUCTION ISOLÉE TERMINÉE : '+str(out/'BUILD_BOUNDS.json'),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:print('CONSTRUCTION REFUSÉE — '+str(e),file=sys.stderr);raise SystemExit(2)
