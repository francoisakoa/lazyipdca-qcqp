#!/usr/bin/env python3
"""Copy frozen QCQP inputs by recorded SHA256; never run an optimizer."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def source_paths(value, field):
    """Only input/start locators; endpoint/result fields are never followed."""
    result = []
    if isinstance(value, dict):
        for key in ([field, 'original_input'] if field == 'input' else [field]):
            if isinstance(value.get(key), str):
                result.append(value[key])
        for key in ('aliases', 'source_record', 'r3_prepared_row',
                    'candidate_provenance', 'serialization_provenance'):
            if key in value:
                result.extend(source_paths(value[key], field))
    elif isinstance(value, list):
        for row in value:
            result.extend(source_paths(row, field))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).with_name('BENCHMARK_CASES.json.gz'))
    parser.add_argument('--source-root', action='append', type=Path, default=[],
                        help='Existing installation or frozen-input root; repeat if needed.')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--case', action='append', default=[],
                        help='Optional exact case ID; repeat to collect a subset.')
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    rows = manifest['cases']
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate case ID in manifest')
    unknown = set(args.case) - set(by_id)
    if unknown:
        raise ValueError('Unknown case IDs: ' + ', '.join(sorted(unknown)))
    if args.case:
        wanted = set(args.case)
        rows = [r for r in rows if r['id'] in wanted]
    roots = [p.expanduser().resolve(strict=True) for p in args.source_root]
    locators = {}
    for root in roots:
        for name in ('PREPARATION.json', 'FROZEN_INVENTORY.json'):
            p = root / name
            if not p.is_file():
                continue
            records = read_json(p).get('cases', [])
            if isinstance(records, dict):
                records = [dict(v, id=v.get('id', k)) for k, v in records.items()]
            for row in records:
                if isinstance(row, dict) and isinstance(row.get('id'), str):
                    locators.setdefault(row['id'], []).append(row)
    cache = {}

    def locate(case, field, expected):
        paths = []
        for row in [case] + locators.get(case['id'], []):
            for raw in source_paths(row, field):
                p = Path(raw).expanduser()
                paths.extend([p] if p.is_absolute() else [root / p for root in roots])
        if field == 'input':
            for root in roots:
                paths.extend(root / tail for tail in (
                    f"{case['id']}/input.qcq", f"data/frozen/{case['id']}/input.qcq",
                    f"inputs/{case['id']}/input.qcq", f"canonical_inputs/{case['id']}/input.qcq",
                    f"data/bundled_frozen/{case['id']}/input.qcq",
                    f"data/reference_inputs/{case['id']}/input.qcq"))
        seen = set()
        for path in paths:
            try:
                p = path.resolve(strict=True)
                if p in seen or not p.is_file():
                    continue
                seen.add(p)
                stat = p.stat()
                key = (str(p), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if key not in cache:
                    cache[key] = digest(p)
                if cache[key] == expected:
                    return p
            except (OSError, RuntimeError):
                continue
        return None

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    for row in rows:
        ident = row['id']
        if not isinstance(ident, str) or ident in ('', '.', '..') or '/' in ident or '\\' in ident:
            raise ValueError('Unsafe case ID')
        fields = [('input', row['input_sha256'])]
        if row.get('external_start_required'):
            expected = row.get('start_file_sha256')
            if not expected:
                raise ValueError('External start hash missing: ' + ident)
            fields.append(('start_file', expected))
        sources = [(field, expected, locate(row, field, expected)) for field, expected in fields]
        absent = [field for field, _, path in sources if path is None]
        if absent:
            missing.append({'case_id': ident, 'missing': absent})
            continue
        for field, expected, source in sources:
            name = 'input.qcq' if field == 'input' else 'start' + source.suffix
            folder = out / 'frozen' / ident
            current = out
            for part in ('frozen', ident):
                current = current / part
                if current.is_symlink():
                    raise ValueError('Symlink in destination: ' + str(current))
                current.mkdir(exist_ok=True)
            target = folder / name
            if target.is_symlink():
                raise ValueError('Symlink destination: ' + str(target))
            if target.exists():
                if not target.is_file() or digest(target) != expected:
                    raise ValueError('Existing destination differs: ' + str(target))
            else:
                temp = target.with_name(target.name + '.copying')
                with temp.open('xb') as dest, source.open('rb') as src:
                    shutil.copyfileobj(src, dest)
                if digest(temp) != expected:
                    temp.unlink()
                    raise ValueError('Source changed during copy: ' + ident)
                temp.rename(target)
            records.append({'case_id': ident, 'field': field,
                            'path': str(target.relative_to(out)),
                            'sha256': expected, 'bytes': target.stat().st_size})
    result = {'schema': 'lazyqcqp-frozen-input-collection-v1',
              'planned_cases': len(rows), 'collected_cases': len(rows) - len(missing),
              'complete': not missing, 'files': records, 'missing': missing,
              'optimizer_invocations': 0, 'endpoint_files_copied': 0}
    (out / 'COLLECTION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Collected {result['collected_cases']}/{len(rows)} cases; "
          f"report: {out / 'COLLECTION.json'}")
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
