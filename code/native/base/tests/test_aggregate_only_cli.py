import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary, problem = sys.argv[1:]
base = [binary, '--problem', problem, '--algorithm', 'lazy-dca', '--dca-engine', 'filter',
        '--linear-backend', 'native', '--dca-aggregate', '--dca-aggregate-backend', 'native',
        '--no-lazy-newton-lazy-constraints', '--dca-aggregate-only', '1', '--display-level', '0']
for opt,value in [('--dca-aggregate-only','2'),('--dca-aggregate-retry-max','0'),
                  ('--dca-aggregate-rho-initial','nan'),('--dca-aggregate-rho-growth','1'),
                  ('--dca-aggregate-rho-decay','1.1'),('--dca-aggregate-rho-max','inf')]:
    r=subprocess.run(base+[opt,value],capture_output=True,text=True,timeout=20)
    assert r.returncode != 0, (opt,value)
with tempfile.TemporaryDirectory() as d:
    endpoint=Path(d)/'endpoint.json'
    r=subprocess.run(base+['--json',str(endpoint),'--max-outer','250','--time-limit','15',
          '--dca-aggregate-rho-max','1e8','--dca-rho-complete-search','0'],
          capture_output=True,text=True,timeout=25)
    assert endpoint.is_file(), r.stderr+r.stdout
    data=json.loads(endpoint.read_text())
    assert data['dca_filter_aggregate_only_enabled'] is True
    assert data['dca_filter_aggregate_only_rho_cap'] == 1e8
    assert data['dca_filter_models'] == 0, data['dca_filter_models']
    assert data['dca_filter_inner_steps'] == 0
    assert '[DCA-BUDGET]' not in r.stdout
    assert r.returncode == 0, r.stdout+r.stderr
    print('PASS aggregate-only CLI: invalid controls rejected, toy solve, no private model, JSON effective values')
