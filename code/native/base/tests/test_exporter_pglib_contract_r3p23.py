#!/usr/bin/env python3
"""The r3p23 exporter is export-only, fingerprinted, and non-overwriting."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from export_pglib_qcqp_r3p23 import (  # noqa: E402
    export_case,
    sha256_file,
    write_json_atomic,
)

TWO_BUS = """
function mpc = tiny_r3p23
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
  1 3 0 0 0 0 1 1.0 0 230 1 1.10 0.90;
  2 1 20 5 0 0 1 1.0 0 230 1 1.10 0.90;
];
mpc.gen = [
  1 0 0 100 -100 1.0 100 1 100 0;
];
mpc.branch = [
  1 2 0.01 0.10 0.02 0 0 0 0 0 1 -30 30;
];
mpc.gencost = [
  2 0 0 3 0.1 2.0 3.0;
];
"""

with tempfile.TemporaryDirectory() as temporary_name:
    temporary = Path(temporary_name)
    case = temporary / "tiny_r3p23.m"
    model = temporary / "tiny_r3p23.qcq"
    case.write_text(TWO_BUS)
    metadata = export_case(
        case, model, flow_limit="apparent", power_flow_start=False)
    assert metadata["model"] == str(model.resolve()), metadata
    assert metadata["model_sha256"] == sha256_file(model), metadata
    assert metadata["source_case_sha256"] == sha256_file(case), metadata
    assert metadata["n"] == 6, metadata
    assert metadata["fixed_variables"] >= 1, metadata
    assert model.read_text().startswith("LAZYIPDCA_QCQP_V1\n"), metadata
    try:
        export_case(case, model, power_flow_start=False)
    except FileExistsError:
        pass
    else:
        raise AssertionError("r3p23 exporter silently overwrote its QCQP")

    # Exercise the user-facing --json path.  Built-in sum() over the NumPy
    # bound masks above returns np.int64, which used to write the .qcq and then
    # fail while serializing the manifest.
    cli_model = temporary / "tiny_r3p23_cli.qcq"
    cli_manifest = temporary / "tiny_r3p23_cli.export.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "python" / "export_pglib_qcqp_r3p23.py"),
            str(case),
            "--output", str(cli_model),
            "--json", str(cli_manifest),
            "--flow-limit", "apparent",
            "--no-power-flow-start",
            "--no-power-flow-voltage-continuation",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    assert cli_model.is_file(), completed.stdout
    assert cli_manifest.is_file(), completed.stdout
    cli_payload = json.loads(cli_manifest.read_text(encoding="utf-8"))
    printed_payload = json.loads(completed.stdout)
    assert cli_payload["model_sha256"] == sha256_file(cli_model), cli_payload
    assert printed_payload["model_sha256"] == cli_payload["model_sha256"]
    assert printed_payload["manifest"] == str(cli_manifest.resolve())
    for field in (
        "finite_lower_bounds", "finite_upper_bounds", "fixed_variables"
    ):
        assert type(cli_payload[field]) is int, (field, cli_payload[field])

    # The serializer is deliberately recursive, not a three-field workaround.
    nested_manifest = temporary / "nested_numpy.json"
    write_json_atomic(
        nested_manifest,
        {
            "scalar": np.float64(1.25),
            "nested": {
                "vector": np.array([np.int64(2), np.int64(3)]),
                "tuple": (np.bool_(True), np.float32(4.5)),
            },
        },
    )
    nested = json.loads(nested_manifest.read_text(encoding="utf-8"))
    assert nested == {
        "nested": {"tuple": [True, 4.5], "vector": [2, 3]},
        "scalar": 1.25,
    }, nested

print("[PASS] r3p23 exporter writes one fingerprinted, non-overwritten QCQP")
