#!/usr/bin/env python3
"""Reproduce archived analyses and figures without running an optimizer."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "analyses")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "reproduced")
    args = parser.parse_args()
    base, out = args.analysis_dir.resolve(), args.out.resolve()
    if not (base / "run_reproduction.py").is_file():
        parser.error("--analysis-dir must contain the archived analyses/run_reproduction.py")
    # Keep frozen inputs and their expected outputs separate from generated files.
    if out == base or base in out.parents:
        parser.error("--out must be outside the archived analyses directory")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(base / "run_reproduction.py"),
                    "--out", str(out)], check=True)
    compact = base / "pairs/OBSERVATIONS_16_CONFIGS.jsonl.gz"
    figures = out / "figures"
    plotter = base / "manuscript_tools/plot_aligned_profiles.py"
    for regimes, stem in [("I M", "aligned_core_performance_profiles"),
                          ("E Q", "aligned_equality_performance_profiles")]:
        subprocess.run([sys.executable, str(plotter), "--compact", str(compact),
                        "--out", str(figures), "--regimes", *regimes.split(),
                        "--stem", stem], check=True)
    # The archived geometry producer writes beside its own source. Execute a
    # copy so its original trajectories and validated figure remain unchanged.
    controlled = out / "controlled_geometry"
    controlled.mkdir(parents=True, exist_ok=True)
    source = base / "figures_R19"
    for item in ["plot_favorable_geometry.py", "lazy_adaptive_trace.json",
                 "lazy_scalar_trace.json", "slsqp_trace.json",
                 "ipopt_monotone_trace.json"]:
        shutil.copy2(source / item, controlled / item)
    subprocess.run([sys.executable, str(controlled / "plot_favorable_geometry.py")],
                   check=True)
    validation = json.loads((out / "REPRODUCTION_VALIDATION.json").read_text())
    figures_exist = all((figures / (stem + extension)).is_file()
                        for stem in ["aligned_core_performance_profiles",
                                     "aligned_equality_performance_profiles"]
                        for extension in [".pdf", ".png", "_validation.json"])
    geometry_exists = (controlled / "figures/favorable_geometry.pdf").is_file()
    assert figures_exist and geometry_exists
    report = {
        "all_tables_matched": validation["all_tables_matched"],
        "table_count": len(validation["tables"]),
        "pair_summary_rows": validation["pair_summary_rows"],
        "pair_summaries_exactly_equal": validation["pair_summaries_exactly_equal"],
        "controlled_refinement_summary_exact": validation["controlled_refinement_summary_exact"],
        "asu_persistence_summaries_exact": validation["asu_persistence_summaries_exact"],
        "performance_profile_outputs_present": figures_exist,
        "controlled_geometry_output_present": geometry_exists,
        "optimizer_invocations": 0,
        "new_endpoint_audits": 0,
        "network_requests": 0,
    }
    (out / "GITHUB_REPRODUCTION_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
