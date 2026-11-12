#!/usr/bin/env python3
"""Canonical sparse-linear serialization for a fresh campaign input layer.

The numerical solver source is unchanged. Both Lazy and Ipopt must receive
the same returned case objects. A changed byte hash is always accompanied by
the original hash and a constructive equality proof: each changed block is
only a permutation of its original unique index/value lines.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path

from CANONICALISER_QCQ_V2 import canonicalize

HERE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _bytes_sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_sha(value):
    return _bytes_sha(json.dumps(value, sort_keys=True, separators=(',', ':'),
                               allow_nan=False).encode())


def _write(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def build_parser(source, destination):
    """Compile only both public QCQ reader entry points, without any optimizer."""
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('g++')
    if not compiler:
        raise ValueError('g++ est nécessaire pour vérifier les lecteurs QCQ réels')
    inputs = [HERE / 'PARSE_QCQ_ONLY.cpp', source / 'src/io.cpp',
              source / 'src/io_compact.cpp', source / 'src/qcq_bound_sentinels.hpp']
    inputs.extend(sorted((source / 'include').rglob('*.hpp')))
    identities = {str(path): sha(path) for path in inputs}
    binary = destination / 'parse_qcq_only'
    receipt = destination / 'PARSER_BUILD.json'
    if receipt.is_file():
        previous = json.loads(receipt.read_text())
        if previous.get('inputs_sha256') != identities or not binary.is_file() or \
                previous.get('binary_sha256') != sha(binary):
            raise ValueError('La sonde de lecture ou ses sources ont changé')
        return binary
    flags = ['-std=c++17', '-O2', '-ffunction-sections', '-fdata-sections']
    command = [compiler] + flags + ['-I', str(source / 'include'),
               str(HERE / 'PARSE_QCQ_ONLY.cpp'), str(source / 'src/io_compact.cpp'),
               str(source / 'src/io.cpp'), '-Wl,--gc-sections', '-o', str(binary)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (destination / 'parser_build.log').write_text(result.stdout)
    if result.returncode:
        raise RuntimeError('Compilation de la sonde QCQ refusée ; voir parser_build.log')
    if identities != {str(path): sha(path) for path in inputs}:
        raise ValueError('Les sources des lecteurs ont changé pendant la compilation')
    _write(receipt, dict(schema='two-entry-point-qcq-parser-probe-v1',
                        binary=str(binary), binary_sha256=sha(binary),
                        inputs_sha256=identities, command=command,
                        compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                        optimizer_called=False, numerical_solver_modified=False,
                        reader_entry_points=['read_qcq_compact', 'read_qcq']))
    return binary


def run_parser(paths, parser):
    """Return checked parser rows; a process error or any rejected input fails."""
    paths = [str(Path(path).resolve(strict=True)) for path in paths]
    if not paths:
        raise ValueError('Aucune entrée sélectionnée pour le contrôle des lecteurs')
    if any('\n' in path or '\t' in path for path in paths):
        raise ValueError('Nom de fichier incompatible avec la preuve de lecture TSV')
    result = subprocess.run([str(Path(parser).resolve(strict=True))] + paths,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=300, check=False)
    records = []
    for line in result.stdout.splitlines():
        fields = line.split('\t')
        if len(fields) != 6 or fields[0] != 'PASS' or fields[5] != 'public_and_compact':
            raise ValueError('Lecture QCQ native refusée : ' + line[:500])
        records.append(dict(path=fields[1], n=int(fields[2]), equalities=int(fields[3]),
                            inequalities=int(fields[4]), readers='public_and_compact'))
    if result.returncode or [row['path'] for row in records] != paths:
        raise ValueError('Contrôle des lecteurs incomplet : ' + result.stderr[:500])
    return records


def _header_identity(data):
    lines = data.splitlines(keepends=True)
    if len(lines) < 6:
        raise ValueError('En-tête QCQ incomplet')
    expected = (b'n', b'x0', b'lb', b'ub')
    for line, key in zip(lines[2:6], expected):
        if line.split()[:1] != [key]:
            raise ValueError('En-tête QCQ non conforme : ' + key.decode())
    n = int(lines[2].split()[1])
    x0 = [float(token) for token in lines[3].split()[1:]]
    if len(x0) != n or not all(math.isfinite(value) for value in x0):
        raise ValueError('Départ QCQ non fini ou de mauvaise dimension')
    return dict(n=n, x0_line_sha256=_bytes_sha(lines[3]),
                x0_binary64_sha256=_bytes_sha(b''.join(struct.pack('>d', value) for value in x0)),
                lower_bounds_line_sha256=_bytes_sha(lines[4]),
                upper_bounds_line_sha256=_bytes_sha(lines[5]))


def _is_beamforming(case):
    return any(case.get(key) == 'multicast_beamforming'
               for key in ('dataset_id', 'family', 'reporting_family'))


def _remap(cases, report):
    changes = {row['id']: row for row in report['entries'] if row['changed']}
    answer = []
    for original in cases:
        row = changes.get(original['id'])
        case = copy.deepcopy(original)
        if row:
            case['input'] = row['input']
            case['input_sha256'] = row['input_sha256']
            case['serialization_provenance'] = {
                key: row[key] for key in ('original_input', 'original_input_sha256',
                    'input_sha256', 'original_case_sha256', 'header_identity',
                    'normalization')}
            case['serialization_provenance'].update(
                rule='permutation_of_unique_sparse_linear_entry_lines_only',
                numerical_model_changed=False, frozen_start_changed=False,
                applies_to_all_methods=True)
            if case.get('dimensions_provenance'):
                case['dimensions_provenance'] = dict(
                    kind='exact_serialization_permutation_of_frozen_model',
                    input_sha256=row['input_sha256'],
                    original=case['dimensions_provenance'])
        answer.append(case)
    return answer


def prepare_cases(cases, source_root, destination, parser, expected_extension_count=120):
    """Prepare copies for the whole beamforming dataset and verify both readers.

    Existing canonical inputs are reused byte-for-byte. Only changed files are
    written to the new destination. Existing campaigns, plans and source input
    files are never altered. The complete input list is returned in its original
    order, with explicit provenance only on cases whose serialization changed.
    """
    source_root = Path(source_root).resolve(strict=True)
    destination = Path(destination).resolve()
    parser = Path(parser).resolve(strict=True)
    if destination == source_root or source_root in destination.parents:
        raise ValueError('Les copies doivent être placées hors de l’installation R6')
    if len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Identifiants de cas dupliqués')
    selected = [case for case in cases if _is_beamforming(case)]
    extensions = [case for case in selected if case.get('source_kind') == 'extension']
    if not selected:
        raise ValueError('Le jeu multicast_beamforming manque dans le plan')
    if expected_extension_count is not None and len(extensions) != expected_extension_count:
        raise ValueError('Panel beamforming incomplet : %d extensions, %d attendues' %
                         (len(extensions), expected_extension_count))
    source_binding = _json_sha(selected)
    receipt = destination / 'CANONICAL_INPUTS.json'
    if receipt.is_file():
        report = json.loads(receipt.read_text())
        if not report.get('pass') or report.get('source_cases_sha256') != source_binding or \
                report.get('parser_sha256') != sha(parser):
            raise ValueError('Preuve de canonicalisation absente, périmée ou différente')
        for row in report['entries']:
            if sha(row['original_input']) != row['original_input_sha256'] or \
                    sha(row['input']) != row['input_sha256']:
                raise ValueError('Une entrée beamforming liée à la preuve a changé : ' + row['id'])
        return _remap(cases, report), report
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Destination de canonicalisation déjà utilisée sans preuve complète')
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for case in selected:
        ident = case['id']
        if not ident or ident in ('.', '..') or '/' in ident or '\\' in ident:
            raise ValueError('Identifiant de cas invalide : ' + ident)
        source = Path(case['input']).expanduser()
        source = (source if source.is_absolute() else source_root / source).resolve(strict=True)
        if destination == source or destination in source.parents:
            raise ValueError('La destination recouvre une entrée source')
        data = source.read_bytes()
        if _bytes_sha(data) != case['input_sha256']:
            raise ValueError('Entrée d’origine différente du plan : ' + ident)
        normalized, evidence = canonicalize(data)
        identity = _header_identity(data)
        if _header_identity(normalized) != identity:
            raise AssertionError('La canonicalisation a modifié les bornes ou le départ')
        target = source
        if evidence['changed']:
            target = destination / ident / 'input.qcq'
            target.parent.mkdir()
            with target.open('xb') as stream:
                stream.write(normalized)
        if sha(source) != case['input_sha256']:
            raise ValueError('Entrée d’origine modifiée pendant le contrôle : ' + ident)
        entries.append(dict(id=ident, source_kind=case.get('source_kind'),
                            original_case_sha256=_json_sha(case), original_input=str(source),
                            original_input_sha256=case['input_sha256'], input=str(target),
                            input_sha256=evidence['output_sha256'], changed=evidence['changed'],
                            header_identity=identity, normalization=evidence))
    parsed = run_parser([row['input'] for row in entries], parser)
    for case, row, checked in zip(selected, entries, parsed):
        dimensions = case.get('dimensions', {})
        expected = (dimensions.get('n'), dimensions.get('equalities_total'),
                    dimensions.get('inequalities_total'))
        actual = (checked['n'], checked['equalities'], checked['inequalities'])
        if any(value is not None and value != actual[index] for index, value in enumerate(expected)):
            raise ValueError('Dimensions lues différentes du plan : ' + case['id'])
        row['native_parser'] = checked
    report = dict(schema='beamforming-canonical-input-layer-v1', pass_=True,
                  source_cases_sha256=source_binding, parser=str(parser), parser_sha256=sha(parser),
                  inspected=len(entries), extensions=len(extensions),
                  changed=sum(row['changed'] for row in entries),
                  changed_extensions=sum(row['changed'] and row['source_kind'] == 'extension' for row in entries),
                  native_read_pass=len(parsed), native_extension_read_pass=len(extensions),
                  original_inputs_modified=False, numerical_models_modified=False,
                  objective_constraints_bounds_starts_preserved=True,
                  coefficients_reserialized=False, optimizer_called=False,
                  KKT_qualification=False,
                  validation_scope='Both original C++ reader entry points; no optimizer execution',
                  entries=entries)
    report['pass'] = report.pop('pass_')
    _write(receipt, report)
    return _remap(cases, report), report
