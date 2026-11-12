#!/usr/bin/env python3
"""Select a fixed-size CUTEst QCQP campaign under explicit dimension caps."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys


BOUND_INFINITY = 1.0e19
EQUALITY_TOLERANCE = 1.0e-12


def finite_lower(value: object) -> bool:
    number = float(value)
    return math.isfinite(number) and number > -BOUND_INFINITY


def finite_upper(value: object) -> bool:
    number = float(value)
    return math.isfinite(number) and number < BOUND_INFINITY


def constraint_dimensions(problem: object) -> tuple[int, int, int]:
    """Return original n and canonical equality/inequality counts."""
    n = int(getattr(problem, "n"))
    m = int(getattr(problem, "m"))
    if m == 0:
        return n, 0, 0

    lower = list(getattr(problem, "cl"))
    upper = list(getattr(problem, "cu"))
    flagged_equalities = list(getattr(problem, "is_eq_cons"))
    if len(lower) != m or len(upper) != m or len(flagged_equalities) != m:
        raise ValueError("CUTEst constraint metadata has inconsistent dimensions")

    equalities = 0
    inequalities = 0
    for lo_raw, up_raw, flagged_raw in zip(lower, upper, flagged_equalities):
        lo = float(lo_raw)
        up = float(up_raw)
        lo_finite = finite_lower(lo)
        up_finite = finite_upper(up)
        numerically_equal = (
            lo_finite
            and up_finite
            and abs(up - lo)
            <= EQUALITY_TOLERANCE * max(1.0, abs(lo), abs(up))
        )
        if bool(flagged_raw) or numerically_equal:
            equalities += 1
        else:
            # This is the number of scalar inequalities in the canonical QCQP:
            # a finite two-sided CUTEst constraint contributes two inequalities.
            inequalities += int(lo_finite) + int(up_finite)
    return n, equalities, inequalities


def read_manifest(path: Path) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text().splitlines():
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        if name not in seen:
            seen.add(name)
            problems.append(name)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--n-limit", type=int, default=10_000,
                        help="exclude n greater than or equal to this value")
    parser.add_argument("--me-limit", type=int, default=5_000,
                        help="exclude equality counts greater than or equal to this value")
    parser.add_argument("--mi-limit", type=int, default=5_000,
                        help="exclude canonical inequality counts greater than or equal to this value")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    for label, value in (("target", args.target), ("n-limit", args.n_limit),
                         ("me-limit", args.me_limit), ("mi-limit", args.mi_limit)):
        if value <= 0:
            parser.error(f"--{label} must be positive")
    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
        os.environ["PYCUTEST_CACHE"] = str(args.cache)

    try:
        import pycutest
    except Exception as exc:
        print(f"[SELECT-FAIL] cannot import PyCUTEst: {exc}", file=sys.stderr)
        return 3

    problems = read_manifest(args.manifest)
    excluded_names = {item.strip().upper() for item in args.exclude if item.strip()}
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    counters = {
        "selected": 0,
        "excluded_size": 0,
        "excluded_name": 0,
        "probe_failure": 0,
    }

    for source_index, name in enumerate(problems, 1):
        row: dict[str, object] = {
            "source_index": source_index,
            "problem": name,
            "n": None,
            "m_e": None,
            "m_I": None,
            "status": None,
            "reason": None,
            "selection_index": None,
        }
        if name.upper() in excluded_names:
            row.update(status="excluded_name", reason="explicit benchmark exclusion")
            counters["excluded_name"] += 1
            rows.append(row)
            print(f"[EXCLUDE-NAME] {name}", flush=True)
            continue

        try:
            problem = pycutest.import_problem(
                name, sifParams=None, drop_fixed_variables=False)
            n, m_e, m_i = constraint_dimensions(problem)
            del problem
            row.update(n=n, m_e=m_e, m_I=m_i)
        except Exception as exc:
            row.update(status="probe_failure", reason=f"{type(exc).__name__}: {exc}")
            counters["probe_failure"] += 1
            rows.append(row)
            print(f"[PROBE-FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        reasons: list[str] = []
        if n >= args.n_limit:
            reasons.append(f"n={n}>={args.n_limit}")
        if m_e >= args.me_limit:
            reasons.append(f"m_e={m_e}>={args.me_limit}")
        if m_i >= args.mi_limit:
            reasons.append(f"m_I={m_i}>={args.mi_limit}")
        if reasons:
            row.update(status="excluded_size", reason="; ".join(reasons))
            counters["excluded_size"] += 1
            rows.append(row)
            print(f"[EXCLUDE-SIZE] {name} n={n} m_e={m_e} m_I={m_i} reason={row['reason']}", flush=True)
            continue

        selected.append(name)
        row.update(status="selected", reason="within all strict caps",
                   selection_index=len(selected))
        counters["selected"] += 1
        rows.append(row)
        print(f"[SELECT] {len(selected)}/{args.target} {name} n={n} m_e={m_e} m_I={m_i}", flush=True)
        if len(selected) == args.target:
            break

    args.selected.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.selected.write_text("".join(f"{name}\n" for name in selected))
    fieldnames = ["source_index", "problem", "n", "m_e", "m_I",
                  "status", "reason", "selection_index"]
    with args.audit.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_manifest": str(args.manifest),
        "source_count": len(problems),
        "scanned_count": len(rows),
        "target_count": args.target,
        "selected_count": len(selected),
        "n_limit_exclusive": args.n_limit,
        "m_e_limit_exclusive": args.me_limit,
        "m_I_limit_exclusive": args.mi_limit,
        "explicit_exclusions": sorted(excluded_names),
        **counters,
        "selected_manifest": str(args.selected),
        "audit_csv": str(args.audit),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "[SELECTION-SUMMARY] "
        f"selected={len(selected)}/{args.target} scanned={len(rows)}/{len(problems)} "
        f"excluded_size={counters['excluded_size']} "
        f"excluded_name={counters['excluded_name']} "
        f"probe_failure={counters['probe_failure']}",
        flush=True,
    )
    print(f"SELECTED_MANIFEST={args.selected}", flush=True)
    print(f"SELECTION_AUDIT={args.audit}", flush=True)
    print(f"SELECTION_JSON={args.summary_json}", flush=True)
    if len(selected) != args.target:
        print(
            f"[SELECT-FAIL] only {len(selected)} eligible instances were found; "
            f"{args.target} required",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
