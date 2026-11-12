#!/usr/bin/env python3
"""Sequential CUTEst QCQP campaign runner with one JSON result per problem."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    root = here.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--results", type=Path, default=root / "campaign" / "results")
    ap.add_argument("--logs", type=Path, default=root / "campaign" / "logs")
    ap.add_argument("--cache", type=Path, default=root / "work_qcqp" / "pycutest_cache")
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--linear-backend", choices=("native", "ldl", "qdldl", "mumps"), default="native")
    ap.add_argument("--max-outer", type=int, default=300)
    ap.add_argument("--max-inner", type=int, default=500)
    ap.add_argument("--feas-tol", type=float, default=1e-7)
    ap.add_argument("--inner-tol", type=float, default=1e-7)
    ap.add_argument("--inner-mu-fallback-tol", type=float, default=1e-6)
    ap.add_argument("--restoration-iters", type=int, default=600)
    ap.add_argument("--restoration-restarts", type=int, default=16)
    ap.add_argument("--restoration-restart-iters", type=int, default=400)
    ap.add_argument("--prox", type=float, default=1e-6)
    args = ap.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    problems = []
    for raw in args.manifest.read_text().splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            problems.append(s)

    rows = []
    for idx, problem in enumerate(problems, 1):
        print(f"===== [{idx}/{len(problems)}] {problem} =====", flush=True)
        log = args.logs / f"{problem}.log"
        work = args.results / problem
        work.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(here / "run_cutest_qcqp.py"), problem,
            "--work-dir", str(work), "--cache", str(args.cache), "--keep-model", "--quiet",
            "--linear-backend", str(args.linear_backend),
            "--time-limit", str(args.time_limit), "--max-outer", str(args.max_outer),
            "--max-inner", str(args.max_inner), "--feas-tol", str(args.feas_tol),
            "--inner-tol", str(args.inner_tol),
            "--inner-mu-fallback-tol", str(args.inner_mu_fallback_tol),
            "--restoration-iters", str(args.restoration_iters),
            "--restoration-restarts", str(args.restoration_restarts),
            "--restoration-restart-iters", str(args.restoration_restart_iters),
            "--prox", str(args.prox),
        ]
        with log.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
        json_files = sorted(work.glob("*.json"))
        data = json.loads(json_files[-1].read_text()) if json_files else {}
        rows.append({
            "problem": problem,
            "rc": proc.returncode,
            "status": data.get("status", "missing"),
            "linear_backend": data.get("linear_solver_backend", args.linear_backend),
            "objective": data.get("cutest_objective", data.get("objective")),
            "feasibility": data.get("cutest_feasibility", data.get("feasibility")),
            "lift_abs_consistency": data.get("lift_abs_consistency"),
            "outer_iterations": data.get("outer_iterations"),
            "inner_iterations": data.get("inner_iterations"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "stationarity": data.get("model_stationarity_inf"),
            "lagrangian_correction_rank": data.get("lagrangian_correction_rank", data.get("objective_rank")),
            "inequality_total_rank": data.get("inequality_total_rank"),
            "restoration_iterations": data.get("restoration_iterations"),
            "restoration_restarts": data.get("restoration_restarts"),
            "inner_fallback_acceptances": data.get("inner_iteration_limit_fallback_acceptances"),
        })
        print(f"[run] {problem:16s} rc={proc.returncode} status={rows[-1]['status']} feas={rows[-1]['feasibility']}")

    summary = args.results.parent / "summary.csv"
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["problem"])
        w.writeheader(); w.writerows(rows)
    print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
