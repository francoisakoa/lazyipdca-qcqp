#!/usr/bin/env python3
import json, subprocess, sys, tempfile
from pathlib import Path
binary, problem = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    for cache, stop in ((0,0),(1,0),(1,1)):
        output=Path(tmp)/f'{cache}_{stop}.json'
        p=subprocess.run([binary,'--problem',problem,'--algorithm','lazy-dca','--dca-engine','filter',
            '--linear-backend','native','--dca-aggregate','--dca-aggregate-backend','native',
            '--dca-aggregate-preparation-cache',str(cache),'--dca-aggregate-strict-probe-stop',str(stop),
            '--max-outer','100','--display-level','0','--json',str(output)],capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        d=json.loads(output.read_text())
        assert d['dca_filter_aggregate_preparation_cache_enabled']==bool(cache)
        assert d['dca_filter_aggregate_strict_probe_stop_enabled']==bool(stop)
        assert d['dca_filter_aggregate_preparation_requests']==d['dca_filter_aggregate_preparation_hits']+d['dca_filter_aggregate_preparation_builds']
    for flag in ('--dca-aggregate-preparation-cache','--dca-aggregate-strict-probe-stop'):
        for bad in ('2','-1','yes','1oops'):
            p=subprocess.run([binary,'--problem',problem,flag,bad],capture_output=True,text=True)
            assert p.returncode!=0,(flag,bad)
print('PASS CLI, JSON controls, cache accounting, invalid option rejection')
