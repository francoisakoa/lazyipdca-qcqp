#!/usr/bin/env python3
"""Measure a single owned child using Linux wait4 resource usage.

No solver options or numerical stopping rule are changed. The wrapper and its
child remain in the process group created by the campaign controller.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def run(command, output):
    if not hasattr(os, 'wait4'):
        raise RuntimeError('Linux wait4 is required for exact per-process resource usage')
    began = time.monotonic()
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
    while True:
        try:
            pid, status, usage = os.wait4(child.pid, 0)
            break
        except InterruptedError:
            continue
    if os.WIFEXITED(status):
        code = os.WEXITSTATUS(status)
    elif os.WIFSIGNALED(status):
        code = -os.WTERMSIG(status)
    else:
        raise RuntimeError('Unexpected wait4 child status')
    child.returncode = code
    result = dict(schema='linux-wait4-child-resources-v1', measurement='os.wait4 exact child rusage',
                  command=command, child_pid=pid, returncode=code,
                  elapsed_seconds=time.monotonic()-began,
                  user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime,
                  peak_rss_kib=usage.ru_maxrss,
                  peak_rss_units='KiB on Linux',
                  scope='Invoked solver/auditor process including its startup and accounted descendants; excludes this wrapper startup and controller; separate from numerical core timers')
    output=Path(output);temporary=output.with_name(output.name+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(output)
    return code if code>=0 else 128-code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-json',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        parser.error('A child command is required after --')
    return run(command,args.resource_json)


if __name__=='__main__':
    raise SystemExit(main())
