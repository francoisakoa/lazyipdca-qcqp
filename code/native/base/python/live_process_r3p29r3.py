"""Stream a child's output to the terminal and its log, enforcing wall timeout.

The selector deadline also applies when the child prints nothing. A private
process group makes timeout and Ctrl-C cleanup include that child's workers.
Linux / WSL runner, matching the distributed shell tooling.
"""
from __future__ import annotations
import codecs
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


def run_logged(command, log_path: Path, *, env, timeout=None, live=False):
    if not live:
        with log_path.open('w') as log:
            return subprocess.run(command, env=env, stdout=log, stderr=log, timeout=timeout)
    print('[RUN] ' + log_path.stem + '  LOG=' + str(log_path), flush=True)
    deadline=time.monotonic()+timeout if timeout is not None else None
    decoder=codecs.getincrementaldecoder('utf-8')('replace')
    process=None
    def kill_group():
        if process is not None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    with log_path.open('wb') as log, selectors.DefaultSelector() as selector:
        try:
            process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,start_new_session=True,bufsize=0)
            selector.register(process.stdout,selectors.EVENT_READ)
            while selector.get_map():
                remaining=None if deadline is None else deadline-time.monotonic()
                if remaining is not None and remaining<=0:
                    raise subprocess.TimeoutExpired(command,timeout)
                for key,_ in selector.select(None if remaining is None else remaining):
                    data=os.read(key.fileobj.fileno(),65536)
                    if not data:
                        selector.unregister(key.fileobj)
                    else:
                        log.write(data);log.flush()
                        sys.stdout.write(decoder.decode(data));sys.stdout.flush()
            remaining=None if deadline is None else max(0,deadline-time.monotonic())
            returncode=process.wait(timeout=remaining)
            sys.stdout.write(decoder.decode(b'',final=True));sys.stdout.flush()
            return subprocess.CompletedProcess(command,returncode)
        except BaseException:
            kill_group()
            if process is not None:process.wait()
            raise
        finally:
            if process is not None and process.stdout is not None:process.stdout.close()
