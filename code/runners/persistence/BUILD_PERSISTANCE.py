#!/usr/bin/env python3
"""Build isolated reference/research binaries with identical MA27+MUMPS dependencies.

No QCQP is solved by this command. The historical installations and all frozen
problem/start files remain untouched. Actual numerical qualification is separate.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import sys
import time

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require_isolated(out,protected):
    out=Path(out).resolve()
    for value in protected:
        path=Path(value).resolve()
        if out==path or path in out.parents or out in path.parents:
            raise ValueError('Construction séparée des sources et installations requise : '+str(out))


def normalized_command(command,source,output):
    return [str(item).replace(str(source),'<SOURCE>').replace(str(output),'<BUILD>') for item in command]


def validate_build_pair(reference,research,receipt):
    """Refuse stale source maps, missing backends or asymmetric compiler flags."""
    for build in (reference,research):
        if build.get('homogeneous_compilation') is not True or build.get('hsl_adapter_compiled') is not True:
            raise ValueError('Compilation MA27/MUMPS homogène requise')
        if not all(re.search(r'\b'+backend+r'\b',build.get('backends','')) for backend in ('ma27','mumps')):
            raise ValueError('Les deux backends natifs doivent être compilés')
        if any(row.get('returncode')!=0 for row in build.get('translation_units',[])):
            raise ValueError('Une unité native a échoué')
    if reference.get('source_sha256')!=receipt['source_reference_sha256']:
        raise ValueError('La source de référence compilée diffère du reçu de patch')
    if research.get('source_sha256')!=receipt['source_research_sha256']:
        raise ValueError('La source de recherche compilée diffère du reçu de patch')
    if reference.get('compiler_version')!=research.get('compiler_version'):
        raise ValueError('Deux compilateurs différents')
    if reference.get('dependencies')!=research.get('dependencies'):
        raise ValueError('Deux ensembles de bibliothèques différents')
    left=reference.get('translation_units',[]);right=research.get('translation_units',[])
    if not left or len(left)!=len(right) or [r['name'] for r in left]!=[r['name'] for r in right]:
        raise ValueError('Liste des unités de compilation différente')
    for old,new in zip(left,right):
        a=normalized_command(old['command'],reference['source'],Path(reference['binary']).parent)
        b=normalized_command(new['command'],research['source'],Path(research['binary']).parent)
        if a!=b:raise ValueError('Options de compilation différentes : '+old['name'])
    a=normalized_command(reference['link_command'],reference['source'],Path(reference['binary']).parent)
    b=normalized_command(research['link_command'],research['source'],Path(research['binary']).parent)
    if a!=b:raise ValueError('Options de liaison différentes')


def runtime_pins(native):
    result=dict(native['all_loaded_library_sha256'])
    result.update(native['hsl_runtime']['dependency_sha256'])
    result[native['hsl_runtime']['hsl_library']]=native['hsl_runtime']['hsl_library_sha256']
    for key in ('original_binary','mumps_reference_binary','original_native_build','original_source_installation'):
        result[native[key]]=native[key+'_sha256']
    return result


def verify_pins(pins):
    for path,expected in pins.items():
        if sha(path)!=expected:raise ValueError('Dépendance modifiée pendant la construction : '+path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=0)
    parser.add_argument('--hsl-library',type=Path)
    parser.add_argument('--boost-include',type=Path)
    parser.add_argument('--mumps-reference-binary',type=Path)
    args=parser.parse_args()
    if args.cpu not in os.sched_getaffinity(0):raise ValueError('Cœur indisponible : '+str(args.cpu))
    out=args.out.expanduser().resolve()
    if out.exists():raise ValueError('Répertoire de construction neuf requis : '+str(out))
    require_isolated(out,[HERE])
    dependency=load('persistence_dependencies',HERE/'DEPENDENCIES.py')
    header=json.loads((HERE/'BASE_CONTEXT.json').read_text())['plan_header']
    # Bootstrap the exact declared loader environment before loading MA27.
    environment=dependency.declared_runtime_environment(header,args.hsl_library)
    if any(os.environ.get(key)!=environment.get(key) for key in ('LD_LIBRARY_PATH','HSL_LIBRARY','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        os.execve(sys.executable,[sys.executable,*sys.argv],environment)
    os.sched_setaffinity(0,{args.cpu})
    lock_name='0_21' if args.cpu in range(22) else str(args.cpu)
    lock_path=Path('/tmp/LAZYQCQP_CPU_'+lock_name+'.lock')
    try:fd=os.open(lock_path,os.O_RDWR|os.O_CREAT,0o666)
    except PermissionError:fd=os.open(lock_path,os.O_RDONLY)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Cœur de compilation réservé ; attendre la fin du calcul existant')
        native=dependency.pin_dependencies(header,args.mumps_reference_binary,args.hsl_library,args.boost_include)
        require_isolated(out,[header['root'],native['original_binary'],native['mumps_reference_binary'],
            Path(native['original_source_installation']).parent,Path(native['original_native_build']).parent,
            HERE/'native/reference_source'])
        pins=runtime_pins(native);verify_pins(pins)
        out.mkdir(parents=True);dependency.write(out/'DEPENDENCIES.json',native)
        patch=load('persistence_source_patch',HERE/'PATCH_PERSISTENCE.py')
        receipt=patch.patch(out/'research_source')
        # patch() normally writes this receipt beside research_source. Write the
        # exact returned record here as well, rather than accepting an old file.
        dependency.write(out/'PATCH_RECEIPT.json',receipt)
        compiler=load('persistence_native_compiler',HERE/'native/COMPILER_NATIVE.py')
        started=time.monotonic()
        reference=compiler.build(HERE/'native/reference_source',out/'build_reference',
            Path(native['link_libraries'][0]).parent,Path(native['boost_include']),1,native['compiler'],native)
        research=compiler.build(out/'research_source',out/'build',
            Path(native['link_libraries'][0]).parent,Path(native['boost_include']),1,native['compiler'],native)
        validate_build_pair(reference,research,receipt);verify_pins(pins)
        result=dict(schema='lazy-curvature-persistence-build-v1',
            reference_binary=reference['binary'],reference_binary_sha256=reference['binary_sha256'],
            reference_build=str(out/'build_reference/BUILD_NATIVE.json'),reference_build_sha256=sha(out/'build_reference/BUILD_NATIVE.json'),
            binary=research['binary'],binary_sha256=research['binary_sha256'],
            compiled_native_build=str(out/'build/BUILD_NATIVE.json'),compiled_native_build_sha256=sha(out/'build/BUILD_NATIVE.json'),
            source_root=str(out/'research_source'),reference_source_root=str(HERE/'native/reference_source'),
            source_sha256=receipt['source_research_sha256'],base_source_sha256=receipt['source_reference_sha256'],
            patch_receipt=str(out/'PATCH_RECEIPT.json'),patch_receipt_sha256=sha(out/'PATCH_RECEIPT.json'),
            dependencies=str(out/'DEPENDENCIES.json'),dependencies_sha256=sha(out/'DEPENDENCIES.json'),
            runtime_environment=native['runtime_environment'],environment=native['runtime_environment'],runtime_sha256=pins,
            hsl_library=native['hsl_runtime']['hsl_library'],hsl_library_sha256=native['hsl_runtime']['hsl_library_sha256'],
            compiled_native_backends=['ma27','mumps'],same_binary_for_both_backends=True,homogeneous_compilation=True,
            production_dependencies_verified=True,research_variant=True,original_installation_unchanged=True,
            optimizer_invocations=0,seconds=time.monotonic()-started)
        dependency.write(out/'BUILD_PERSISTANCE.json',result)
        print('CONSTRUCTION PERSISTANCE TERMINÉE : '+str(out/'BUILD_PERSISTANCE.json'),flush=True)
    finally:os.close(fd)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print('CONSTRUCTION PERSISTANCE REFUSÉE — '+str(error),file=sys.stderr,flush=True)
        raise SystemExit(2)
