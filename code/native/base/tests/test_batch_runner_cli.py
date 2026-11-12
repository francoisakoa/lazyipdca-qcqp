#!/usr/bin/env python3
from pathlib import Path
import os
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_cutest_batch_v0428r3p16_phase1_runtime_latch_tol1e6.sh"


def run(*args):
    return subprocess.run(
        ["bash", str(RUNNER), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


help_result = run("--help")
assert help_result.returncode == 0, help_result.stdout
for token in ("--linear-backend", "--dca-equality-model",
              "condensed-retraction", "elastic-split", "--problems", "--manifest",
              "--phase1-threads", "--working-set-drop-patience"):
    assert token in help_result.stdout, (token, help_result.stdout)

bad_backend = run("--linear-backend", "dense-magic")
assert bad_backend.returncode == 2, bad_backend.stdout
assert "invalid --linear-backend" in bad_backend.stdout, bad_backend.stdout

bad_model = run("--dca-equality-model", "adaptive-undocumented")
assert bad_model.returncode == 2, bad_model.stdout
assert "invalid --dca-equality-model" in bad_model.stdout, bad_model.stdout

bad_threads = run("--phase1-threads", "0")
assert bad_threads.returncode == 2, bad_threads.stdout
assert "--phase1-threads must be a positive integer" in bad_threads.stdout, bad_threads.stdout

bad_drop_patience = run("--working-set-drop-patience", "0")
assert bad_drop_patience.returncode == 2, bad_drop_patience.stdout
assert "--working-set-drop-patience must be a positive integer" in bad_drop_patience.stdout, bad_drop_patience.stdout

missing = run("--linear-backend")
assert missing.returncode == 2, missing.stdout
assert "missing value" in missing.stdout, missing.stdout

missing_manifest = run("--manifest", "/definitely/not/a/manifest.txt")
assert missing_manifest.returncode == 2, missing_manifest.stdout
assert "manifest not found" in missing_manifest.stdout, missing_manifest.stdout

mixed_selection = run("--manifest", __file__, "--problems", "BT1")
assert mixed_selection.returncode == 2, mixed_selection.stdout
assert "mutually exclusive" in mixed_selection.stdout, mixed_selection.stdout

# Regression exposed by HS108 and CATENA.  The condensed selector is only an
# equality model: it must not globally disable the corrector/SOC machinery
# used when a problem has no quadratic equality.  CATENA additionally needs
# the already validated rank-8 border plus repeated scalar safeguarding.
source = RUNNER.read_text()
condensed_start = source.index("\n    condensed-retraction)\n")
condensed_end = source.index("\n    elastic-split)\n", condensed_start)
condensed = source[condensed_start:condensed_end]
for forbidden in ("--no-soc", "--no-dca-corrector",
                  "--no-dca-iterated-soc", "--no-dca-equality-retraction"):
    assert forbidden not in condensed, (forbidden, condensed)
for required in ("--dca-corrector", "--dca-iterated-soc",
                 "--dca-equality-retraction", "--dca-condensed-max-rank 8",
                 "--dca-condensed-scalar-safeguard-updates 8",
                 "--lazyqp-low-rank-form bordered", "--dca-condensed-tol 1e-8"):
    assert required in condensed, (required, condensed)

# The r3p10 rescue is explicit in the reproducibility profile.  It must not be
# coupled to the equality-model selector or silently inherit the stale rank-64
# CATENA settings from the external patch kit.
for required in ("--equality-qp-original-kkt-gmres",
                 "--equality-qp-original-kkt-gmres-max-iterations 128"):
    assert required in source, (required, source)
assert "--dca-condensed-max-rank 64" not in source, source
for required in ("--restoration-exact-workspace-reuse",
                 "--restoration-exact-sparse-row-dots",
                 "--restoration-exact-parallel-min-rows 128",
                 'export OMP_NUM_THREADS="$PHASE1_THREADS" OMP_DYNAMIC=FALSE',
                 '--working-set-drop-patience "$WORKING_SET_DROP_PATIENCE"'):
    assert required in source, (required, source)
assert "first mismatch latches Phase I dense" in source, source
assert "cmake " not in source.lower(), "batch runner must not compile"

python_runner = (ROOT / "python" / "run_cutest_qcqp.py").read_text()
for required in ('ap.add_argument("--equality-qp-original-kkt-gmres"',
                 'ap.add_argument("--no-equality-qp-original-kkt-gmres"',
                 '"--equality-qp-original-kkt-gmres-max-iterations"',
                 'cmd.append("--equality-qp-original-kkt-gmres"'):
    assert required in python_runner, (required, python_runner)
for required in ('ap.add_argument("--restoration-exact-workspace-reuse"',
                 'ap.add_argument("--restoration-exact-parallel-row-dots"',
                 '"--restoration-exact-parallel-min-rows"'):
    assert required in python_runner, (required, python_runner)

# A missing solver JSON used to collapse every early Python/export crash into
# the opaque label "missing_json".  The runner must expose the actual log tail
# and common Python exception lines on the console.
for diagnostic in ("[FAIL-DETAIL]", 'tail -n 80 "$LOG"', "^Traceback",
                   "Error|Exception", "'status': 'runner_failure'", "'log_tail': tail"):
    assert diagnostic in source, (diagnostic, source)

# Exercise the failure path, rather than merely checking its source text.  A
# fake exporter process dies before writing JSON; subsequent Python calls are
# delegated to the real interpreter so the runner can still build its summary.
with tempfile.TemporaryDirectory(prefix="lazyipdca_runner_diag_") as tmp:
    tmp = Path(tmp)
    fake_cli = tmp / "fake_cli"
    fake_cli.write_text("""#!/usr/bin/env bash
if [[ "${1:-}" == "--version" ]]; then
  echo "LazyIPDCA-QCQP 0.4.28r3p16-phase1-runtime-latch"
elif [[ "${1:-}" == "--linear-backends" ]]; then
  echo "native,ldl"
else
  exit 0
fi
""")
    fake_cli.chmod(0o755)
    fake_python = tmp / "python"
    fake_python.write_text("""#!/usr/bin/env bash
if [[ "${1:-}" == "-u" ]]; then
  echo "A4X12_DIAGNOSTIC_SENTINEL: exporter terminated before JSON"
  exit 139
fi
exec %s "$@"
""" % shlex.quote(sys.executable))
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": str(tmp) + os.pathsep + env.get("PATH", ""),
        "BASE": str(tmp / "base"),
        "BIN": str(fake_cli),
        "RUN_ID": "MISSING_JSON_DIAGNOSTIC",
        "PROBLEMS": "A4X12",
    })
    diagnostic_run = subprocess.run(
        ["bash", str(RUNNER), "--linear-backend", "ldl"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, env=env)
    assert diagnostic_run.returncode == 0, diagnostic_run.stdout
    assert "status=runner_failure" in diagnostic_run.stdout, diagnostic_run.stdout
    assert "[FAIL-DETAIL]" in diagnostic_run.stdout, diagnostic_run.stdout
    assert "A4X12_DIAGNOSTIC_SENTINEL" in diagnostic_run.stdout, diagnostic_run.stdout
    failure_json = tmp / "base" / "results" / "MISSING_JSON_DIAGNOSTIC" / "A4X12" / "A4X12.json"
    failure_payload = __import__("json").loads(failure_json.read_text())
    assert failure_payload["status"] == "runner_failure", failure_payload
    assert failure_payload["runner_return_code"] == 139, failure_payload
    assert any("A4X12_DIAGNOSTIC_SENTINEL" in line for line in failure_payload["log_tail"]), failure_payload

print("[PASS] r3p16 batch runner exposes independent selectors, guarded condensed tolerance, working-set patience, and missing-JSON diagnostics")
