#!/usr/bin/env python3
"""Plan, or explicitly execute, a cold standard-auto build with native R10 backends.

The default only writes a reviewable build plan. --run compiles every solver
translation unit from this source tree and selected contract tests; it never
launches a problem solve. No object or executable from a previous build is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(source):
    return {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*'))
            if p.is_file() and p.suffix in ('.hpp', '.h', '.cpp', '.py')
            and '__pycache__' not in p.parts}


def dependencies(explicit, prefix):
    roots = [Path(explicit).resolve()] if explicit else []
    roots += sorted(Path('/workspace/scratch').glob('*/R9_DEPS'))
    if prefix:roots.insert(0,Path(prefix).resolve())
    roots += [Path('/usr'), Path('/usr/local')]
    cmake = shutil.which('cmake')
    boost = None
    for root in roots:
        if not cmake:
            candidate = root/'python_tools/cmake/data/bin/cmake'
            if candidate.is_file():
                cmake = str(candidate)
        for suffix in ('boost_headers/include', 'sysroot/usr/include', 'include'):
            candidate = root/suffix
            if boost is None and (candidate/'boost/multiprecision/cpp_int.hpp').is_file():
                boost = candidate
    if not cmake or boost is None:
        raise RuntimeError('CMake or Boost.Multiprecision headers unavailable; pass --deps with an installed dependency root')
    # A supplied/active prefix is the portable TRX path: use its already
    # installed native numerical libraries through ordinary CMake discovery.
    # CasADi is only a local fallback for this qualification environment.
    casadi=None;identity={};library_roots=[]
    extra=[]
    if prefix:
        prefix=Path(prefix).resolve()
        extra=['-DCMAKE_PREFIX_PATH:PATH='+str(prefix),'-DMUMPS_ROOT:PATH='+str(prefix),
               '-DMETIS_ROOT:PATH='+str(prefix),'-DCMAKE_BUILD_RPATH:STRING='+str(prefix/'lib')]
        library_roots=[str(prefix/'lib')]
    else:
        spec = importlib.util.find_spec('casadi')
        if spec is not None:
            casadi=Path(spec.origin).parent
            required=['include/coin-or/mumps/dmumps_c.h','include/coin-or/metis/metis.h',
                      'libcoinmumps.so','libcoinmetis.so','libcasadi-tp-openblas.so']
            if all((casadi/name).is_file() for name in required):
                identity={str(casadi/name):sha(casadi/name) for name in required}
                library_roots=[str(casadi)]
                extra=['-DCMAKE_BUILD_RPATH:STRING='+str(casadi),
                    '-DMUMPS_INCLUDE_DIR:PATH='+str(casadi/'include/coin-or/mumps'),
                    '-DMUMPS_DMUMPS_LIBRARY:FILEPATH='+str(casadi/'libcoinmumps.so'),
                    '-DMUMPS_COMMON_LIBRARY:FILEPATH=','-DMUMPS_PORD_LIBRARY:FILEPATH=',
                    '-DMUMPS_MPISEQ_LIBRARY:FILEPATH=',
                    '-DMETIS_INCLUDE_DIR:PATH='+str(casadi/'include/coin-or/metis'),
                    '-DMETIS_LIBRARY:FILEPATH='+str(casadi/'libcoinmetis.so'),
                    '-DBLA_VENDOR:STRING=OpenBLAS','-DBLA_SIZEOF_INTEGER:STRING=4',
                    '-DBLAS_openblas_LIBRARY:FILEPATH='+str(casadi/'libcasadi-tp-openblas.so'),
                    '-DLAPACK_openblas_LIBRARY:FILEPATH='+str(casadi/'libcasadi-tp-openblas.so')]
    compiler = shutil.which('g++-13') or shutil.which('g++')
    if not compiler:
        raise RuntimeError('C++17 compiler unavailable')
    identity[str(Path(cmake).resolve())] = sha(Path(cmake).resolve())
    identity[str(boost/'boost/multiprecision/cpp_int.hpp')] = sha(boost/'boost/multiprecision/cpp_int.hpp')
    return cmake, boost, compiler, identity, extra, library_roots


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--deps', type=Path)
    ap.add_argument('--prefix',type=Path,default=os.environ.get('CONDA_PREFIX'),
                    help='Existing native dependency prefix; defaults to CONDA_PREFIX')
    ap.add_argument('--cmake-arg',action='append',default=[],help='Additional CMake option; use --cmake-arg=-DKEY=VALUE')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.jobs < 1:
        ap.error('--jobs must be positive')
    if (output/'native/CMakeCache.txt').exists():
        raise SystemExit('Cold build refuses an existing CMake cache: '+str(output/'native'))
    cmake, boost, compiler, dep_identity, runtime_options, library_roots = dependencies(args.deps,args.prefix)
    native = output/'native'
    configure = [cmake, '-S', str(source), '-B', str(native), '-G', 'Unix Makefiles',
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG',
        '-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DLAZYIPDCA_BUILD_TESTS=ON', '-DLAZYIPDCA_WITH_HSL=OFF',
        '-DLAZYIPDCA_WITH_IPOPT=OFF', '-DLAZYIPDCA_WITH_LDL=OFF',
        '-DLAZYIPDCA_WITH_QDLDL=OFF', '-DLAZYIPDCA_WITH_MUMPS=ON',
        '-DLAZYIPDCA_WITH_METIS=ON', '-DLAZYIPDCA_WITH_OPENMP=ON',
        '-DLAZYIPDCA_NATIVE_ARCH=OFF', '-DLAZYIPDCA_REQUIRE_SPARSE_BACKENDS=OFF',
        '-DCMAKE_CXX_COMPILER:FILEPATH='+compiler,
        '-DLAZYIPDCA_BOOST_INCLUDE_DIR:PATH='+str(boost),*runtime_options,*args.cmake_arg]
    targets = ['lazyipdca_qcqp_cli', 'test_signed_quadratic_equality_domain', 'test_standard_auto_policy', 'test_standard_exact_presolve',
               'test_native_signed_square_reduction', 'test_native_signed_square_path',
               'test_original_filter_commit','test_coefficient_constraint_increment',
               'test_gr_proximal_policy','test_gr_fixed_centering','test_native_al_fallback_policy',
               'test_zero_barrier_merit','test_zero_barrier_epoch','test_auto_startup',
               'test_restoration_negative_curvature','test_failed_linearization_restoration',
               'test_initial_original_fit_guard','test_initial_original_fit_certificate',
               'test_psd_equality_route','test_square_continuation','test_zero_barrier_forcing','test_constant_square_policy','test_historical_equality_watch', 'test_positive_square_epigraph_restoration', 'test_coefficient_graph_certificate','test_positive_square_epigraph_substitution','test_native_closed_bound_restoration','test_common_factor_substitution','test_affine_cycle_certificate','test_gr_bound_stagnation_watch','test_convex_aggregate_certificate']
    build = [cmake, '--build', str(native), '--target', *targets, '-j', str(args.jobs)]
    ctest = str(Path(cmake).with_name('ctest'))
    test = [ctest, '--test-dir', str(native), '--output-on-failure', '-R',
            '^(homogeneous_spectral_envelope|convex_aggregate_certificate|gr_bound_stagnation_watch|affine_cycle_certificate|common_factor_substitution|native_closed_bound_restoration|positive_square_epigraph_substitution|coefficient_graph_certificate|historical_equality_watch|positive_square_epigraph_restoration|signed_quadratic_equality_domain|standard_exact_presolve|standard_auto_policy|native_signed_square_reduction|native_signed_square_path|original_filter_commit|r10_full_step_controller|coefficient_constraint_increment|gr_coefficient_actual_wiring|gr_proximal_policy|gr_proximal_controller|gr_fixed_centering|native_al_fallback_policy|zero_barrier_merit|zero_barrier_epoch|auto_startup_policy|restoration_negative_curvature|failed_linearization_restoration|initial_original_fit_guard|initial_original_fit_certificate|constant_objective_guard|constant_feasibility_cli|initial_original_fit_provenance|diagonal_psd_equality_route|square_continuation|zero_barrier_forcing|constant_square_policy)$']
    commands = [configure, build, test,
                [str(native/'lazyipdca_qcqp_cli'), '--version'],
                [str(native/'lazyipdca_qcqp_cli'), '--linear-backends']]
    receipt = dict(schema='standard-auto-cold-build-v1', executed=args.run,
        source=str(source), source_sha256=source_identity(source),
        dependency_sha256=dep_identity,dependency_prefix=str(args.prefix) if args.prefix else None,
        output=str(output), commands=commands,
        reused_solver_objects=False, reused_executables=False,
        binary=str(native/'lazyipdca_qcqp_cli'), command_records=[])
    output.mkdir(parents=True, exist_ok=True)
    plan = output/'BUILD_PLAN.json'
    plan.write_text(json.dumps(receipt, indent=2)+'\n')
    print('BUILD_PLAN', plan, flush=True)
    if not args.run:
        return
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1', OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1')
    if library_roots:
        environment['LD_LIBRARY_PATH']=':'.join(library_roots)+(':'+environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
    started = time.monotonic()
    success = True
    for index, command in enumerate(commands):
        log = output/f'{index:02d}_{("configure", "build", "tests", "version", "backends")[index]}.log'
        began = time.monotonic()
        if index==2:
            # Some workspace mounts retain a non-executable placeholder mode
            # after linking. Restore owner execute permission on verified ELF
            # outputs only; every qualification test still runs normally.
            repairs=[]
            for target in targets:
                executable=native/target
                if executable.is_file() and not os.access(executable,os.X_OK):
                    with executable.open('rb') as elf: magic=elf.read(4)
                    if magic!=b'\x7fELF':raise RuntimeError('Built target is not ELF: '+str(executable))
                    previous=executable.stat().st_mode
                    executable.chmod(previous|0o100)
                    repairs.append(dict(path=str(executable),previous_mode=oct(previous&0o777)))
            receipt['executable_permission_repairs']=repairs
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        record = dict(command=command, returncode=result.returncode,
                      seconds=time.monotonic()-began, log=str(log), log_sha256=sha(log))
        receipt['command_records'].append(record)
        print(log.name, result.returncode, round(record['seconds'], 3), flush=True)
        if result.returncode:
            success = False
            break
        if index==0:
            cache=(native/'CMakeCache.txt').read_text()
            resolved={}
            for line in cache.splitlines():
                if line.startswith(('MUMPS_INCLUDE_DIR:','MUMPS_DMUMPS_LIBRARY:','METIS_INCLUDE_DIR:','METIS_LIBRARY:')):
                    key,value=line.split('=',1);resolved[key.split(':',1)[0]]=value
            receipt['resolved_native_dependencies']=resolved
            if len(resolved)!=4 or any(not v or v.endswith('-NOTFOUND') for v in resolved.values()):
                success=False;receipt['error']='Native MUMPS+METIS dependencies were not found in the configured environment'
                break
            for path in resolved.values():
                if Path(path).is_file():receipt['dependency_sha256'][path]=sha(path)
    receipt['source_unchanged'] = receipt['source_sha256'] == source_identity(source)
    receipt['success'] = success and receipt['source_unchanged']
    receipt['seconds'] = time.monotonic()-started
    binary = native/'lazyipdca_qcqp_cli'
    receipt['binary_sha256'] = sha(binary) if binary.is_file() else None
    (output/'BUILD.json').write_text(json.dumps(receipt, indent=2)+'\n')
    raise SystemExit(0 if receipt['success'] else 1)


if __name__ == '__main__':
    main()
