#!/usr/bin/env python3
"""Package G9 and verify completeness separately from manifest integrity."""
from pathlib import Path
import argparse
import hashlib
import os
import tarfile

TOP = 'lazyipdca_qcqp_v0.4.28r3p28g9_krylov_source'
MANIFEST = 'RELEASE_MANIFEST_SHA256_G9.txt'
REQUIRED = {
    'CMakeLists.txt', 'tools/build_g9.sh', 'tools/run_g9.sh', 'tools/install_and_test_g9.sh',
    'python/benchmark_filter_dca_g9.py', 'src/dca_local_active.hpp', 'src/dca_krylov.hpp', 'tests/test_g9_krylov.cpp', 'tests/test_g9_krylov_rollback.cpp',
    'src/sparse_pair_elimination.hpp', 'include/lazyipdca_qcqp/version.hpp',
    'release_identity_g9.json', 'INSTALL_AND_TEST_G9.md',
}

def source_files(root):
    # Only root BUILD DIRECTORIES are excluded. Never apply this to filenames:
    # tools/build_g9.sh and validation_g6/build_provenance.json are deliverables.
    build_dirs = {p.name for p in root.iterdir()
                  if p.is_dir() and (p.name == 'build' or p.name.startswith('build_')
                                     or p.name.startswith('build-')
                                     or (p / 'CMakeCache.txt').is_file())}
    files = []
    for p in root.rglob('*'):
        rel = p.relative_to(root)
        if rel.parts[0] in build_dirs or any(c in {'.git', '__pycache__'} for c in rel.parts[:-1]):
            continue
        if p.is_file() and p.suffix not in {'.pyc', '.pyo'}:
            files.append(p)
    return sorted(files)

def stream_digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()

def digest(path):
    with path.open('rb') as stream:
        return stream_digest(stream)

def package(root, output):
    root = root.resolve()
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError('archive destination must be outside the source tree')
    files = source_files(root)
    names = {p.relative_to(root).as_posix() for p in files}
    if REQUIRED - names:
        raise ValueError('missing required source files: ' + str(sorted(REQUIRED - names)))
    expected = {p.relative_to(root).as_posix(): digest(p) for p in files
                if p.relative_to(root).as_posix() != MANIFEST}
    (root / MANIFEST).write_text(''.join(f'{h}  {name}\n' for name, h in sorted(expected.items())))
    expected[MANIFEST] = digest(root / MANIFEST)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    with tarfile.open(temporary, 'w:gz', compresslevel=6) as archive:
        for name in sorted(expected):
            archive.add(root / name, arcname=f'{TOP}/{name}', recursive=False)
    seen = set()
    with tarfile.open(temporary, 'r:gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(TOP + '/'):
                raise ValueError('unexpected archive member: ' + member.name)
            name = member.name[len(TOP) + 1:]
            if name in seen or name not in expected:
                raise ValueError('duplicate or unexpected file: ' + name)
            seen.add(name)
            if stream_digest(archive.extractfile(member)) != expected[name]:
                raise ValueError('archive checksum mismatch: ' + name)
    if seen != set(expected) or REQUIRED - seen:
        raise ValueError('archive is incomplete')
    os.replace(temporary, output)
    print(f'PACKAGE=PASS files={len(seen)} bytes={output.stat().st_size} sha256={digest(output)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.source, args.output)
