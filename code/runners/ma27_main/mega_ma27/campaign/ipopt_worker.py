#!/usr/bin/env python3
"""Run the separate library-default Ipopt MA27 or MUMPS reference on frozen input."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def command_for(configuration,case,inp,endpoint,seconds,max_outer):
    expected={'ipopt_qcqp':'ma27','ipopt_mumps':'mumps'}
    method=case.get('method')
    if method not in expected:raise ValueError('Unknown independent Ipopt method: '+str(method))
    backend=expected[method]
    # Use the same hash-bound receipt as controller startup. This is a reader
    # only; real-solve qualification has already completed during preparation.
    sys.path.insert(0,str(HERE.parent))
    from setup_runtime import default_installation
    defaults=default_installation(configuration)
    if seconds!=7200 or max_outer!=5000:
        raise ValueError('The full-corpus campaign requires the common7200s/5000iteration budget')
    binary=Path(defaults['default_binary'])
    command=[str(binary),'--problem',str(Path(inp).resolve()),'--json',str(Path(endpoint).resolve()),
        '--time-limit',str(seconds),'--max-iter',str(max_outer),'--linear-solver',backend]
    if backend=='ma27':command+=['--hsl-library',str(Path(defaults['hsl_library']).resolve())]
    # No numerical tuning flags: defaults, including monotone barrier and
    # automatic MUMPS ordering7, are obtained from the qualified library.
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-json', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--endpoint', type=Path, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--max-outer', type=int, required=True)
    args = parser.parse_args()
    configuration = json.loads((HERE.parent / 'INSTALLATION.json').read_text())
    case = json.loads(args.case_json.read_text())
    case = case.get('case', case)
    if sha(args.input) != case['input_sha256']:
        raise ValueError('QCQP gelé différent du contrat')
    binary=Path(configuration['ipopt_binary'])
    command=command_for(configuration,case,args.input,args.endpoint,args.seconds,args.max_outer)
    if case.get('external_start_required'):
        start = Path(case['start_file'])
        if not start.is_absolute():
            start = HERE / start
        if sha(start) != case['start_file_sha256']:
            raise ValueError('Départ primal externe différent du contrat')
        command += ['--frozen-start', str(start.resolve())]
    # Replacing the worker keeps the controller's process identity and affinity.
    # There is no optimizer substitution or extra child process to orphan.
    print('IPOPT_QCQP_COMMAND ' + json.dumps(command), flush=True)
    for variable in ('LD_PRELOAD', 'LD_AUDIT', 'LD_DEBUG', 'LD_DEBUG_OUTPUT'):
        os.environ.pop(variable, None)
    # The package cwd contains no ipopt.opt; reject a later accidental override.
    if (HERE / 'ipopt.opt').exists():
        raise ValueError('ipopt.opt non autorisé dans le dossier de campagne')
    os.execv(str(binary), command)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print('IPOPT_QCQP_REFUSE : ' + str(error), file=sys.stderr)
        raise SystemExit(2)
