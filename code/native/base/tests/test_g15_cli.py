#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path
binary, problem = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    for enabled in (0, 1):
        path = Path(tmp) / f'{enabled}.json'
        args = [binary, '--problem', problem, '--algorithm', 'lazy-dca', '--dca-engine', 'filter',
                '--linear-backend', 'native', '--dca-aggregate', '--dca-aggregate-backend', 'native',
                '--dca-rho-complete-search', str(enabled), '--dca-strict-finish', str(enabled),
                '--dca-strict-finish-max', '3', '--dca-strict-finish-refinements', '5',
                '--dca-strict-finish-eta', '0.03', '--dca-strict-finish-mu-fraction', '0.007', '--display-level', '0', '--json', str(path)]
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stdout + p.stderr
        d = json.loads(path.read_text())
        assert d['dca_filter_rho_complete_search_enabled'] == bool(enabled)
        assert d['dca_filter_strict_finish_enabled'] == bool(enabled)
        assert d['dca_filter_strict_finish_budget'] == 3
        assert d['dca_filter_strict_finish_refinement_budget'] == 5
        assert d['dca_filter_strict_finish_eta'] == .03
        assert d['dca_filter_strict_finish_mu_fraction'] == .007
    for flag, values in {
        '--dca-rho-complete-search': ['yes', '2', '-1'],
        '--dca-strict-finish': ['yes', '2', '-1'],
        '--dca-strict-finish-max': ['-1', '101'],
        '--dca-strict-finish-refinements': ['-1', '101'],
        '--dca-strict-finish-eta': ['0', '-1', '1', 'nan', 'inf'],
        '--dca-strict-finish-mu-fraction': ['0', '-1', '1', 'nan', 'inf'],
    }.items():
        for value in values:
            p = subprocess.run([binary, '--problem', problem, '--algorithm', 'lazy-dca',
                                '--dca-engine', 'filter', flag, value], capture_output=True, text=True, timeout=30)
            assert p.returncode != 0, (flag, value)
print('PASS settings/CLI/JSON round-trip and invalid budgets/forcing values')
