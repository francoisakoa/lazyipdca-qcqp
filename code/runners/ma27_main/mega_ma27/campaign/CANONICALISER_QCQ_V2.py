#!/usr/bin/env python3
"""Sort unique V2 sparse-linear entries on a new copy; never change a source.

This is a serialization candidate, not a numerical solver correction. Only
the order of complete entry lines changes. Number tokens, starts, bounds,
Hessians and row order are preserved exactly. No optimizer is called.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonicalize(data):
    lines = data.splitlines(keepends=True)
    if not lines or lines[0].strip() not in (b'LAZYIPDCA_QCQP_V1', b'LAZYIPDCA_QCQP_V2'):
        raise ValueError('Expected frozen QCQ V1/V2 format')
    if lines[0].strip() == b'LAZYIPDCA_QCQP_V1':
        return data, dict(format='V1', changed=False, reordered_blocks=0, entry_lines_rewritten=0,
                          reason='Dense linear format unchanged', source_sha256=digest(data), output_sha256=digest(data))
    if len(lines) < 6 or len(lines[2].split()) != 2 or lines[2].split()[0] != b'n':
        raise ValueError('Expected scalar dimension on header line 3')
    n = int(lines[2].split()[1])
    if n < 1:
        raise ValueError('Positive dimension required')
    changed, total, reordered, i = [], 0, 0, 0
    output = list(lines)
    while i < len(lines):
        parts = lines[i].split()
        if parts and parts[0] == b'linear_nnz':
            if len(parts) != 2:
                raise ValueError('Invalid sparse linear count line')
            count = int(parts[1])
            if count < 0 or count > n or i + count >= len(lines):
                raise ValueError('Invalid sparse linear count')
            entries = []
            for offset, line in enumerate(lines[i+1:i+1+count]):
                pair = line.split()
                if len(pair) != 2:
                    raise ValueError('Sparse entries must occupy separate lines')
                j, value = int(pair[0]), float(pair[1])
                if j < 0 or j >= n or not math.isfinite(value):
                    raise ValueError('Invalid sparse linear entry')
                entries.append((j, line))
            if len({j for j, _ in entries}) != len(entries):
                raise ValueError('Duplicate indices refused; no coefficient merge allowed')
            normalized = [line for _, line in sorted(entries, key=lambda e: e[0])]
            if normalized != lines[i+1:i+1+count]:
                output[i+1:i+1+count] = normalized
                changed.append(dict(count_line=i+1, entries=count))
                reordered += 1
            total += count
            i += count
        i += 1
    result = b''.join(output)
    # Constructive equivalence: unchanged bytes outside entry blocks; each
    # changed block is exactly a permutation of the same unique index/value lines.
    for block in changed:
        j, k = block['count_line'], block['entries']
        if sorted(lines[j:j+k]) != sorted(output[j:j+k]):
            raise AssertionError('Entry line multiset changed')
    return result, dict(format='V2', changed=result != data, reordered_blocks=reordered,
                        entry_lines_examined=total, entry_lines_rewritten=0,
                        numbers_reserialized=False, coefficient_lines_preserved=True,
                        other_bytes_preserved=True, source_sha256=digest(data), output_sha256=digest(result),
                        blocks=changed)


def prepare(source, destination):
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve()
    if source == destination:
        raise ValueError('In-place normalization forbidden')
    for parent in destination.parents:
        if (parent / 'MANIFEST.json').is_file() or (parent / 'PLAN.json').is_file():
            raise ValueError('Destination must be outside frozen packages and existing campaigns')
    data = source.read_bytes()
    normalized, report = canonicalize(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(normalized)
    if source.read_bytes() != data:
        raise ValueError('Source changed concurrently')
    return dict(source=str(source), destination=str(destination), **report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))
