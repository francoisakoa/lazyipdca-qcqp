#!/usr/bin/env python3
"""Focused contract for the bounded PF voltage-setpoint continuation."""

import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from export_pglib_qcqp_r3p21 import build_model, parse_matpower  # noqa: E402
from run_pglib_qcqp_r3p21 import parser as runner_parser  # noqa: E402


CASE = """
function mpc = voltage_start
mpc.version = '2';
mpc.baseMVA = 100.0;
mpc.bus = [
  1  3   0.0   0.0  0.00  0.00  1  1.00  0.0  230.0  1  1.10  0.90;
  2  2  30.0  10.0  0.00  0.05  1  1.00  0.0  230.0  1  1.10  0.90;
  3  1  50.0  20.0  0.02  0.00  1  1.00  0.0  230.0  1  1.10  0.90;
];
mpc.gen = [
  1  5.0  0.0  100.0  -100.0  1.0  100.0  1  30.0  2.0;
  2  5.0  0.0  100.0  -100.0  1.0  100.0  1  60.0  3.0;
];
mpc.branch = [
  1  2  0.01  0.08  0.020  60.0  0  0  0.00  0.0  1  -25.0  25.0;
  1  3  0.02  0.10  0.030  70.0  0  0  0.98  3.0  1  -25.0  25.0;
  2  3  0.03  0.12  0.010   0.0  0  0  0.00  0.0  1  -25.0  25.0;
];
mpc.gencost = [
  2  0.0  0.0  3  0.110  5.0  20.0;
  2  0.0  0.0  3  0.085  1.2  15.0;
];
"""


def metrics(model):
    x, lb, ub, equalities, inequalities = (
        model[1], model[2], model[3], model[5], model[6])
    eq_inf = max(abs(q.value(x)) for q in equalities)
    ineq_inf = max(0.0, max(q.value(x) for q in inequalities))
    bound_inf = max(0.0, float(np.max(lb - x)), float(np.max(x - ub)))
    return eq_inf, ineq_inf, bound_inf


def make_model(case, continuation):
    return build_model(
        case, flow_limit="apparent", angle_limits=True,
        power_flow_start=True,
        power_flow_voltage_continuation=continuation)


def main() -> int:
    case_path = pathlib.Path(tempfile.mkdtemp()) / "voltage_start.m"
    case_path.write_text(CASE, encoding="utf-8")

    # A raw-feasible PF point must not take the continuation path or move by
    # even one bit merely because the feature is enabled.
    ordinary = parse_matpower(case_path)
    raw_ordinary = make_model(ordinary, False)
    default_ordinary = build_model(
        ordinary, flow_limit="apparent", angle_limits=True,
        power_flow_start=True)
    auto_ordinary = make_model(ordinary, True)
    assert np.array_equal(raw_ordinary[1], default_ordinary[1])
    assert np.array_equal(raw_ordinary[1], auto_ordinary[1])
    assert auto_ordinary[14]["power_flow_voltage_continuation_status"] == \
        "raw-feasible"
    assert auto_ordinary[14]["power_flow_voltage_continuation_trials"] == 0

    # Tighten only the PQ-bus lower voltage bound.  The raw load flow still
    # satisfies every equality, but violates vmin_3.  A bounded lift of the PV
    # voltage control makes the complete QCQP start feasible.
    low_voltage = parse_matpower(case_path)
    low_voltage["bus"][2, 12] = 0.995
    raw_low = make_model(low_voltage, False)
    default_low = build_model(
        low_voltage, flow_limit="apparent", angle_limits=True,
        power_flow_start=True)
    auto_low = make_model(low_voltage, True)
    assert np.array_equal(raw_low[1], default_low[1])
    assert default_low[14]["power_flow_voltage_continuation_status"] == \
        "disabled-raw-retained"
    raw_meta = raw_low[14]
    meta = auto_low[14]
    assert raw_meta["raw_start_equality_residual"] <= 1e-9
    assert raw_meta["raw_start_inequality_violation"] > 1e-4
    assert meta["power_flow_voltage_continuation_status"] == "accepted"
    assert meta["power_flow_voltage_continuation_trials"] > 0
    assert meta["power_flow_voltage_lift"] > 0.0
    assert max(metrics(auto_low)) <= 1e-9
    assert meta["start_equality_residual"] == metrics(auto_low)[0]
    assert meta["start_inequality_violation"] == metrics(auto_low)[1]
    assert meta["start_bound_violation"] == metrics(auto_low)[2]
    assert meta["power_flow_raw_seconds"] >= 0.0
    assert meta["power_flow_raw_postvalidation_seconds"] >= 0.0
    assert meta["power_flow_voltage_continuation_seconds"] >= 0.0
    assert meta["power_flow_start_preprocess_seconds"] == (
        meta["power_flow_raw_seconds"]
        + meta["power_flow_raw_postvalidation_seconds"]
        + meta["power_flow_voltage_continuation_seconds"])

    # If the PV bus has insufficient voltage headroom, the search must retain
    # the measured raw start exactly and advertise the fallback.
    no_headroom = parse_matpower(case_path)
    no_headroom["bus"][2, 12] = 0.995
    no_headroom["bus"][1, 11] = 1.001
    raw_no_headroom = make_model(no_headroom, False)
    auto_no_headroom = make_model(no_headroom, True)
    assert np.array_equal(raw_no_headroom[1], auto_no_headroom[1])
    assert auto_no_headroom[14][
        "power_flow_voltage_continuation_status"] == \
        "unavailable-raw-retained"
    assert auto_no_headroom[14]["start_inequality_violation"] > 1e-4

    # Both command-line surfaces are opt-in and expose symmetrical flags.
    runner_default = runner_parser().parse_args([str(case_path)])
    runner_on = runner_parser().parse_args(
        [str(case_path), "--power-flow-voltage-continuation"])
    runner_off = runner_parser().parse_args(
        [str(case_path), "--no-power-flow-voltage-continuation"])
    assert runner_default.power_flow_voltage_continuation is False
    assert runner_on.power_flow_voltage_continuation is True
    assert runner_off.power_flow_voltage_continuation is False

    low_case_path = case_path.with_name("low_voltage.m")
    low_case_path.write_text(
        CASE.replace(
            "3  1  50.0  20.0  0.02  0.00  1  1.00  0.0  230.0  1  1.10  0.90;",
            "3  1  50.0  20.0  0.02  0.00  1  1.00  0.0  230.0  1  1.10  0.995;"),
        encoding="utf-8")
    exporter = ROOT / "python" / "export_pglib_qcqp_r3p21.py"
    default_json = case_path.with_name("default.json")
    enabled_json = case_path.with_name("enabled.json")
    base_command = [
        sys.executable, str(exporter), str(low_case_path),
        "--flow-limit", "apparent", "--power-flow-start"]
    subprocess.run(
        base_command + ["--output", str(case_path.with_name("default.qcq")),
                        "--json", str(default_json)],
        check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        base_command + ["--power-flow-voltage-continuation",
                        "--output", str(case_path.with_name("enabled.qcq")),
                        "--json", str(enabled_json)],
        check=True, stdout=subprocess.DEVNULL)
    default_cli = json.loads(default_json.read_text(encoding="utf-8"))
    enabled_cli = json.loads(enabled_json.read_text(encoding="utf-8"))
    assert default_cli["power_flow_voltage_continuation_status"] == \
        "disabled-raw-retained"
    assert default_cli["start_inequality_violation"] > 1e-4
    assert enabled_cli["power_flow_voltage_continuation_status"] == "accepted"
    assert enabled_cli["start_inequality_violation"] <= 1e-9

    print("[PASS] bounded PF voltage continuation: raw parity, exact "
          "postvalidation, transparent fallback, CLI opt-in")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
