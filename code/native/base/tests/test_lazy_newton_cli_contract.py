#!/usr/bin/env python3
"""Three-way CLI contract for the r3p28 lazy-newton route.

The option set is derived from the sources, so a ``--ln-*`` option added to
one leg and forgotten in another fails this test (mutation detection):

  1. C++ parser   : every ``a == "--ln-..."`` / ``"--no-ln-..."`` branch of
                    src/main.cpp must be documented in its usage text and
                    must be accepted by the built binary;
  2. Python wrapper: python/benchmark_case118_r3p28.py must declare the same
                    option set (LN_OPTIONS / LN_FLAG_OPTIONS);
  3. Shell runner  : tools/run_case118_r3p28.sh --help must expose them all.

Usage: test_lazy_newton_cli_contract.py <lazyipdca_qcqp_cli>
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'src' / 'main.cpp'
WRAPPER = ROOT / 'python' / 'benchmark_case118_r3p28.py'
RUNNER = ROOT / 'tools' / 'run_case118_r3p28.sh'
REGRESSION = ROOT / 'tests' / 'regression' / 'BT1.qcq'


def parser_options(source: str) -> set[str]:
    return set(re.findall(r'a == "(--(?:no-)?ln-[a-z0-9-]+)"', source))


def usage_options(source: str) -> set[str]:
    usage_start = source.index('void usage(')
    usage = source[usage_start:source.index('\n}\n', usage_start)]
    return set(re.findall(r'"  (--(?:no-)?ln-[a-z0-9-]+)', usage))


def wrapper_options(source: str) -> set[str]:
    found = set(re.findall(r"\('(--ln-[a-z0-9-]+)', '(?:float|int|str)'\)", source))
    for on, off in re.findall(r"\('(--ln-[a-z0-9-]+)', '(--no-ln-[a-z0-9-]+)'\)", source):
        found.add(on); found.add(off)
    return found


def main(argv):
    if len(argv) != 2:
        print(__doc__); return 2
    binary = Path(argv[1]).resolve(strict=True)
    backend_query = subprocess.run(
        [str(binary), '--linear-backends'], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert backend_query.returncode == 0, backend_query.stdout
    compiled_backends = {
        item.strip() for item in backend_query.stdout.strip().split(',')
        if item.strip()
    }
    historical_ldl = 'ldl' in compiled_backends
    main_source = MAIN.read_text()
    parsed = parser_options(main_source)
    assert parsed, 'no --ln-* option found in the C++ parser'
    documented = usage_options(main_source)
    assert parsed == documented, ('C++ usage text out of sync', sorted(parsed ^ documented))
    wrapped = wrapper_options(WRAPPER.read_text())
    assert parsed == wrapped, ('Python wrapper out of sync', sorted(parsed ^ wrapped))
    # settings fields: every parsed option must map onto an ln_ field
    settings = (ROOT / 'include' / 'lazyipdca_qcqp' / 'settings.hpp').read_text()
    fields = set(re.findall(r'\b(ln_[a-z0-9_]+)\s*=', settings))
    for opt in parsed:
        field = opt.replace('--no-', '--')[2:].replace('-', '_')
        assert field in fields, ('settings.hpp lacks a field for', opt)
    # shell runner exposes the wrapper help verbatim
    help_result = subprocess.run(['bash', str(RUNNER), '--help'], text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert help_result.returncode == 0, help_result.stdout
    for opt in sorted(parsed):
        assert opt in help_result.stdout, ('shell runner help lacks', opt)
    # binary help
    binary_help = subprocess.run([str(binary), '--help'], text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, check=False).stdout
    for opt in sorted(parsed):
        assert opt in binary_help, ('binary --help lacks', opt)
    # binary accepts every value option and rejects invalid ones
    base = [str(binary), '--problem', str(REGRESSION), '--algorithm', 'lazy-newton',
            '--linear-backend', 'ldl', '--qp-backend', 'ldl', '--quiet', '--max-outer', '1']
    ok = subprocess.run(base, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if historical_ldl:
        # 1 = not solved within one iteration is acceptable.
        assert ok.returncode in (0, 1), ok.stdout
    else:
        assert ok.returncode == 2 and "linear backend 'ldl' not compiled" in ok.stdout, ok.stdout
    rejected = [
        ('--ln-mu-init', '0'), ('--ln-mu-kappa', '1'), ('--ln-mu-theta', '1'), ('--ln-drop-patience', '0'),
        ('--ln-tau-min', '1'), ('--ln-inequality-curvature', 'magic'), ('--ln-ordering', 'magic'),
        ('--ln-globalization', 'magic'), ('--ln-condense-max-support', '-1'), ('--ln-max-soc', '0'),
    ]
    for opt, value in rejected:
        bad = subprocess.run(base + [opt, value], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        assert bad.returncode == 2, (opt, value, bad.stdout)
        assert '[fatal]' in bad.stdout, (opt, value, bad.stdout)
    missing = subprocess.run(base + ['--ln-mu-init'], text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
    assert missing.returncode == 2 and 'missing value' in missing.stdout, missing.stdout
    wrong_backend = subprocess.run([str(binary), '--problem', str(REGRESSION), '--algorithm', 'lazy-newton',
                                    '--linear-backend', 'native', '--quiet'], text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert wrong_backend.returncode == 2 and 'requires --linear-backend ldl' in wrong_backend.stdout, wrong_backend.stdout
    # the wrapper forwards only explicitly given options (binary defaults stay authoritative)
    sys.path.insert(0, str(ROOT / 'python'))
    import benchmark_case118_r3p28 as bench  # noqa: E402
    parser = bench.build_parser()
    args = parser.parse_args(['--qcq', str(REGRESSION), '--binary', str(binary)])
    assert bench.lazy_newton_overrides(args) == [], bench.lazy_newton_overrides(args)
    args = parser.parse_args(['--qcq', str(REGRESSION), '--binary', str(binary),
                              '--ln-mu-init', '0.5', '--no-ln-quadratic-correction', '--ln-ordering', 'metis'])
    forwarded = bench.lazy_newton_overrides(args)
    assert forwarded == ['--ln-mu-init', '0.5', '--ln-ordering', 'metis', '--no-ln-quadratic-correction'], forwarded
    # the grafted kernel value of --inner-backend must be exposed on the three legs
    assert 'lazyqp-sparse' in binary_help, 'binary --help lacks --inner-backend lazyqp-sparse'
    assert bench.GRAFT_INNER_BACKEND == 'lazyqp-sparse' and 'graft' in bench.ALL_MODES
    assert 'graft' in help_result.stdout and 'lazyqp-sparse' in help_result.stdout, 'runner help lacks the graft mode'
    graft_ok = subprocess.run([str(binary), '--problem', str(REGRESSION), '--inner-backend', 'lazyqp-sparse',
                               '--linear-backend', 'ldl', '--quiet'], text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=False)
    if historical_ldl:
        assert graft_ok.returncode in (0, 1), graft_ok.stdout
    else:
        assert (graft_ok.returncode == 2 and
                "linear backend 'ldl' not compiled" in graft_ok.stdout), graft_ok.stdout
    graft_bad = subprocess.run([str(binary), '--problem', str(REGRESSION), '--inner-backend', 'lazyqp-sparse',
                                '--linear-backend', 'native', '--quiet'], text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    assert graft_bad.returncode == 2 and 'requires --linear-backend ldl' in graft_bad.stdout, graft_bad.stdout
    if not historical_ldl:
        print('[SKIP] lazy-newton runtime requires the uncompiled historical '
              'SuiteSparse LDL backend; static CLI/wrapper/runner contracts passed')
        return 77
    print('lazy-newton three-way CLI contract: OK (%d options)' % len(parsed))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
