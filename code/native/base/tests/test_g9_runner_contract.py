#!/usr/bin/env python3
"""Guard the G9/G7 comparison and the independent correctness/time exit contracts."""
from __future__ import annotations
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import benchmark_filter_dca_g9 as runner


def arguments():
    return argparse.Namespace(max_iter=1000, time_limit=1800, display_level=0,
        live_every=1, live_header_every=20, inner_max=8, max_atoms=500,
        atom_batch_size=2, atom_event_every=1, atom_events=True,
        research_profile="local", local_trigger=0.02, local_primal_trigger=1e-4,
        local_backend="mumps", local_weak_affine=True,
        rho_refinements=5, initial_restoration=False, normal_weighted_atoms=False,
        no_model_reuse=False, no_extrapolation=False, krylov_max=24, krylov_cooldown=2, no_krylov_pc=False)


class RunnerContract(unittest.TestCase):
    def test_g7_baseline_remains_fixed_when_g9_triggers_change(self):
        args = arguments()
        baseline = runner.lazy_command(args, Path("g7"), Path("same.qcq"), Path("old.json"), "baseline")
        current = runner.lazy_command(args, Path("g9"), Path("same.qcq"), Path("new.json"), "filter-dca")
        for flag in runner.RESEARCH_PROFILES["local"]:
            self.assertIn(flag, baseline)
            self.assertIn(flag, current)
        self.assertEqual(baseline[baseline.index("--dca-local-trigger") + 1], "1.0")
        self.assertEqual(baseline[baseline.index("--dca-rho-refinements") + 1], "3")
        self.assertNotIn("--dca-local-primal-trigger", baseline)
        self.assertNotIn("--dca-local-backend", baseline)
        self.assertNotIn("--dca-local-weak-affine", baseline)
        self.assertEqual(current[current.index("--dca-local-backend") + 1], "mumps")
        self.assertIn("--dca-local-weak-affine", current)
        self.assertEqual(current[current.index("--dca-local-trigger") + 1], "0.02")
        self.assertEqual(current[current.index("--dca-local-primal-trigger") + 1], "0.0001")
        for flag in ("--problem", "--feas-tol", "--opt-tol", "--dual-inf-tol",
                     "--compl-inf-tol", "--kkt-berr-reject", "--dca-inner-max"):
            self.assertEqual(baseline[baseline.index(flag) + 1], current[current.index(flag) + 1])

    def test_g9_local_options_do_not_change_nonlocal_profiles(self):
        args = arguments()
        args.research_profile = "block"
        command = runner.lazy_command(args, Path("g9"), Path("same.qcq"), Path("new.json"), "filter-dca")
        for flag in ("--dca-local-trigger", "--dca-local-primal-trigger",
                     "--dca-local-backend", "--dca-local-weak-affine"):
            self.assertNotIn(flag, command)

    def test_ablation_changes_only_acceleration_switch(self):
        args = arguments()
        args.research_profile = "krylov-local"
        on = runner.lazy_command(args, Path("g9"), Path("same.qcq"), Path("same.json"), "filter-dca")
        off = runner.lazy_command(args, Path("g9"), Path("same.qcq"), Path("same.json"), "unaccelerated")
        self.assertEqual(off, on + ["--no-dca-krylov"])
        baseline = runner.lazy_command(args, Path("g7"), Path("same.qcq"), Path("same.json"), "baseline")
        self.assertNotIn("--dca-krylov", baseline)
        self.assertNotIn("--dca-sparse-fallback", baseline)

    def test_performance_does_not_masquerade_as_correctness(self):
        for correct in (False, True):
            for target in (False, True):
                summary = {"all_correct": correct, "target_met": target}
                self.assertEqual(runner.campaign_exit_code(summary), 0 if correct else 1)
                self.assertEqual(runner.campaign_exit_code(summary, True),
                                 1 if not correct else (0 if target else 2))

    def test_cli_defaults_and_explicit_performance_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["runner", "--frozen", "case300", "--binary", "g9",
                    "--output", str(Path(tmp) / "campaign"), "--require-target"]
            with patch.object(sys, "argv", argv), patch.object(runner, "campaign", return_value=0) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(), 0)
            args = run.call_args.args[0]
            self.assertEqual(args.research_profile, "krylov-local")
            self.assertEqual(args.local_trigger, 1.0)
            self.assertEqual(args.local_primal_trigger, 1e-3)
            self.assertEqual(args.local_backend, "same")
            self.assertFalse(args.local_weak_affine)
            self.assertTrue(args.require_target)

    def test_identity_and_packaging_required_entry_points(self):
        identity = json.loads((ROOT / "release_identity_g9.json").read_text())
        self.assertEqual(identity["api_version"], 4335)
        current = json.loads((ROOT / "release_identity_g10.json").read_text())
        for token in (current["version"], current["build_id"], str(current["api_version"])):
            self.assertIn(token, (ROOT / "include/lazyipdca_qcqp/version.hpp").read_text())
        spec = importlib.util.spec_from_file_location("package_g9", ROOT / "tools/package_g9.py")
        packager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packager)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            for name in ("tools/build_g9.sh", "tools/run_g9.sh", "validation/build_provenance.json",
                         "build_pf_gpu/omit", "build-custom/omit", "__pycache__/omit.pyc"):
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            names = {p.relative_to(source).as_posix() for p in packager.source_files(source)}
            self.assertEqual(names, {"tools/build_g9.sh", "tools/run_g9.sh", "validation/build_provenance.json"})
        self.assertIn("tools/build_g9.sh", packager.REQUIRED)
        self.assertIn("tools/run_g9.sh", packager.REQUIRED)


if __name__ == "__main__":
    unittest.main()
