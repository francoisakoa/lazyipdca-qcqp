#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "python" / "select_cutest_qcqp_manifest.py"
RUNNER = ROOT / "tools" / "run_cutest100_v0428r3p16_phase1_runtime_latch_tol1e6.sh"


with tempfile.TemporaryDirectory(prefix="lazyipdca_cutest100_selector_") as raw_tmp:
    tmp = Path(raw_tmp)
    fake_module = tmp / "pycutest.py"
    fake_module.write_text(r'''
class Problem:
    def __init__(self, n, m=0, equalities=0, two_sided=False):
        self.n = n
        self.m = m
        if two_sided:
            self.cl = [-1.0] * m
            self.cu = [1.0] * m
            self.is_eq_cons = [False] * m
        else:
            self.cl = ([0.0] * equalities) + ([-float("inf")] * (m-equalities))
            self.cu = ([0.0] * equalities) + ([1.0] * (m-equalities))
            self.is_eq_cons = ([True] * equalities) + ([False] * (m-equalities))

def import_problem(name, sifParams=None, drop_fixed_variables=False):
    if name == "PROBEFAIL":
        raise RuntimeError("synthetic CUTEst import failure")
    specs = {
        "OK1": Problem(9999),
        "LARGE_EQ": Problem(20, m=5000, equalities=5000),
        "LARGE_INEQ": Problem(20, m=5000),
        "LARGE_RANGE": Problem(20, m=2500, two_sided=True),
        "LARGE_N": Problem(10000),
        "CONT6-QQ": Problem(12000, m=10),
        "OK2": Problem(80, m=2, equalities=1),
        "OK3": Problem(50, m=1),
    }
    return specs[name]
''')
    manifest = tmp / "candidates.txt"
    manifest.write_text(
        "BT4\nDRCAVTY2\nOK1\nLARGE_EQ\nLARGE_INEQ\nLARGE_RANGE\n"
        "LARGE_N\nCONT6-QQ\nPROBEFAIL\nOK2\nOK3\n"
    )
    selected = tmp / "selected.txt"
    audit = tmp / "audit.csv"
    summary = tmp / "summary.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp)
    result = subprocess.run(
        [
            sys.executable, str(SELECTOR),
            "--manifest", str(manifest),
            "--selected", str(selected),
            "--audit", str(audit),
            "--summary-json", str(summary),
            "--target", "3",
            "--n-limit", "10000",
            "--me-limit", "5000",
            "--mi-limit", "5000",
            "--exclude", "BT4",
            "--exclude", "DRCAVTY2",
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, env=env,
    )
    assert result.returncode == 0, result.stdout
    assert selected.read_text().splitlines() == ["OK1", "OK2", "OK3"], result.stdout
    payload = json.loads(summary.read_text())
    assert payload["selected_count"] == 3, payload
    assert payload["excluded_size"] == 5, payload
    assert payload["excluded_name"] == 2, payload
    assert payload["probe_failure"] == 1, payload
    rows = {row["problem"]: row for row in csv.DictReader(audit.open())}
    assert rows["BT4"]["status"] == "excluded_name", rows["BT4"]
    assert rows["DRCAVTY2"]["status"] == "excluded_name", rows["DRCAVTY2"]
    assert rows["LARGE_N"]["reason"] == "n=10000>=10000", rows["LARGE_N"]
    assert rows["LARGE_EQ"]["reason"] == "m_e=5000>=5000", rows["LARGE_EQ"]
    assert rows["LARGE_INEQ"]["reason"] == "m_I=5000>=5000", rows["LARGE_INEQ"]
    assert rows["LARGE_RANGE"]["reason"] == "m_I=5000>=5000", rows["LARGE_RANGE"]
    assert rows["CONT6-QQ"]["status"] == "excluded_size", rows["CONT6-QQ"]

help_result = subprocess.run(
    ["bash", str(RUNNER), "--help"], text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
assert help_result.returncode == 0, help_result.stdout
for token in ("exactly 100", "n >= 10000", "m_e >= 5000", "m_I >= 5000",
              "BT4", "DRCAVTY2", "--source-manifest", "--target-count",
              "--phase1-threads", "--working-set-drop-patience"):
    assert token in help_result.stdout, (token, help_result.stdout)

bad_limit = subprocess.run(
    ["bash", str(RUNNER), "--n-limit", "0"], text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
assert bad_limit.returncode == 2, bad_limit.stdout
assert "positive integer" in bad_limit.stdout, bad_limit.stdout

runner_source = RUNNER.read_text()
for exclusion in ("--exclude BT4", "--exclude DRCAVTY2"):
    assert exclusion in runner_source, (exclusion, runner_source)
assert "cmake " not in runner_source.lower(), "CUTEst100 runner must not compile"

print("[PASS] r3p16 CUTEst100 selector enforces size caps, BT4/DRCAVTY2 exclusions, and exact target count")
