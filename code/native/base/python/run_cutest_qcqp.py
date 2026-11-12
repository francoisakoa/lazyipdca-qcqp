#!/usr/bin/env python3
"""Export one CUTEst QCQP, run LazyIPDCA-QCQP, and validate on CUTEst."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from export_cutest_qcqp import (
    export_problem, evaluate_cutest_solution, _parse_sif_param,
)


def main():
    here = Path(__file__).resolve().parent
    root = here.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("problem")
    ap.add_argument("--binary", type=Path, default=root / "build" / "lazyipdca_qcqp_cli")
    ap.add_argument("--work-dir", type=Path, default=root / "work_qcqp")
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--sif-param", action="append", default=[], type=_parse_sif_param)
    ap.add_argument("--keep-model", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--reformulation", choices=("auto", "none", "core-xabs", "bt2-exact", "eigen2-residual"), default="auto")
    ap.add_argument("--algorithm", choices=("lazy-dca","auto","legacy-sqp"), default="lazy-dca",
                    help="lazy-dca is the single scientific route; auto is a compatibility alias")
    ap.add_argument("--auto-sqp-max-n", type=int, default=0,
                    help="deprecated compatibility option; ignored by the single-route solver")
    ap.add_argument("--inner-backend", choices=("lazyqp", "ipm", "native", "legacy-ipm", "filter-sqp", "tr-sqp", "ipopt"), default="lazyqp")
    # LazyIPDCA-QP compatibility profile.  These names deliberately retain
    # the qp_ namespace in the settings JSON while their CLI spelling uses
    # conventional hyphens.  The nonconvex atom/AMRS controls are recorded but
    # bypassed because LazyQP receives an already-convex QCQP.
    ap.add_argument("--qp-option", default="C")
    ap.add_argument("--qp-amrs", dest="qp_amrs", action="store_true", default=True)
    ap.add_argument("--no-qp-amrs", dest="qp_amrs", action="store_false")
    ap.add_argument("--qp-init", type=int, default=2)
    ap.add_argument("--qp-backend", choices=("native", "ldl", "qdldl", "mumps"), default="ldl")
    ap.add_argument("--qp-rho-trial-every", type=int, default=2)
    ap.add_argument("--qp-max-atoms", type=int, default=500)
    ap.add_argument("--qp-seed-coverage", type=float, default=0.25)
    ap.add_argument("--qp-seed-stall-tol", type=float, default=0.0)
    ap.add_argument("--qp-seed-patience", type=int, default=999)
    ap.add_argument("--qp-atom-events", dest="qp_atom_events", action="store_true", default=True)
    ap.add_argument("--no-qp-atom-events", dest="qp_atom_events", action="store_false")
    ap.add_argument("--qp-atom-event-every", type=int, default=50)
    ap.add_argument("--qp-bound-push", type=float, default=0.0)
    ap.add_argument("--qp-bp-push-kappa", type=float, default=1e-4)
    ap.add_argument("--qp-bp-margin-tol", type=float, default=1e-4)
    ap.add_argument("--qp-escapes", type=int, default=3)
    ap.add_argument("--qp-mehrotra", dest="qp_mehrotra", action="store_true", default=True)
    ap.add_argument("--no-qp-mehrotra", dest="qp_mehrotra", action="store_false")
    ap.add_argument("--qp-sigma-max", type=float, default=1e14)
    ap.add_argument("--qp-model-solved-tol", type=float, default=1e-12)
    ap.add_argument("--qp-dca-stop", dest="qp_dca_stop", action="store_true", default=True)
    ap.add_argument("--no-qp-dca-stop", dest="qp_dca_stop", action="store_false")
    ap.add_argument("--qp-dca-stall-tol-rel", type=float, default=1e-10)
    ap.add_argument("--qp-dca-stall-iters", type=int, default=5)
    ap.add_argument("--qp-dca-monotone-tol-abs", type=float, default=1e-6)
    ap.add_argument("--qp-bp-adaptive", dest="qp_bp_adaptive", action="store_true", default=True)
    ap.add_argument("--no-qp-bp-adaptive", dest="qp_bp_adaptive", action="store_false")
    ap.add_argument("--linear-backend", choices=("native", "ldl", "qdldl", "mumps"), default="native",
                    help="compatibility/global backend; --qp-backend controls LazyQP")
    ap.add_argument("--linear-metis", type=int, choices=(0, 1), default=1)
    ap.add_argument("--kkt-persistent-symbolic-cache", dest="kkt_persistent_symbolic_cache", action="store_true", default=True)
    ap.add_argument("--no-kkt-persistent-symbolic-cache", dest="kkt_persistent_symbolic_cache", action="store_false")
    ap.add_argument("--kkt-primal-reg", type=float, default=1e-12)
    ap.add_argument("--kkt-dual-reg", type=float, default=1e-12)
    ap.add_argument("--kkt-adaptive-reg", action="store_true")
    ap.add_argument("--rho-x-floor", type=float, default=1e-12)
    ap.add_argument("--rho-x-growth", type=float, default=10.0)
    ap.add_argument("--rho-x-decay", type=float, default=0.5)
    ap.add_argument("--rho-x-max", type=float, default=1e-2)
    ap.add_argument("--rho-x-clean", type=int, default=3)
    ap.add_argument("--delta-dual-base", type=float, default=1e-8)
    ap.add_argument("--delta-dual-floor", type=float, default=1e-12)
    ap.add_argument("--delta-dual-max", type=float, default=1e-4)
    ap.add_argument("--delta-dual-mode", choices=("inverse","fixed"), default="inverse")
    ap.add_argument("--kkt-berr-tol", type=float, default=1e-9)
    ap.add_argument("--kkt-berr-reject", type=float, default=1e-7)
    ap.add_argument("--kkt-refine", type=int, default=3)
    ap.add_argument("--equality-qp-original-kkt-gmres",
                    dest="equality_qp_original_kkt_gmres",
                    action="store_true", default=True)
    ap.add_argument("--no-equality-qp-original-kkt-gmres",
                    dest="equality_qp_original_kkt_gmres",
                    action="store_false")
    ap.add_argument("--equality-qp-original-kkt-gmres-max-iterations",
                    type=int, default=128)
    ap.add_argument("--max-outer", type=int, default=300)
    ap.add_argument("--max-inner", type=int, default=500)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--feas-tol", type=float, default=1e-6)
    ap.add_argument("--opt-tol", type=float, default=1e-5)
    ap.add_argument("--dual-inf-tol", type=float, default=1e-5)
    ap.add_argument("--compl-inf-tol", type=float, default=1e-5)
    ap.add_argument("--s-max", type=float, default=100.0)
    ap.add_argument("--inner-tol", type=float, default=1e-7)
    ap.add_argument("--inner-mu-fallback-tol", type=float, default=1e-6)
    ap.add_argument("--lazyqp-mu0", type=float, default=1e-4)
    ap.add_argument("--lazyqp-mu-r", type=float, default=0.1)
    ap.add_argument("--lazyqp-mu-theta", type=float, default=0.1)
    ap.add_argument("--lazyqp-mu-inner-max", type=int, default=0)
    ap.add_argument("--lazyqp-kkt-form", choices=("auto", "full", "reduced"), default="auto")
    ap.add_argument("--lazyqp-reduced-kkt-min-ineq", type=int, default=32)
    ap.add_argument("--lazyqp-low-rank-form", choices=("auto", "condensed", "bordered"), default="auto")
    ap.add_argument("--lazyqp-condense-threshold", type=int, default=256)
    ap.add_argument("--lazyqp-warm-guard", type=float, default=2.0)
    ap.add_argument("--inner-warm-start", "--inner-equality-warm-start", dest="inner_warm_start_mode", action="store_const", const="equality", default="off")
    ap.add_argument("--inner-full-dual-warm-start", dest="inner_warm_start_mode", action="store_const", const="full")
    ap.add_argument("--no-inner-warm-start", dest="inner_warm_start_mode", action="store_const", const="off")
    ap.add_argument("--inner-bound-push", type=float, default=0.0)
    ap.add_argument("--inner-bound-frac", type=float, default=1e-2)
    ap.add_argument("--inner-bound-push-ref", type=float, default=1e-6)
    ap.add_argument("--inner-bound-mult-init", type=float, default=1.0)
    ap.add_argument("--restoration-iters", type=int, default=600)
    ap.add_argument("--restoration-restarts", type=int, default=16)
    ap.add_argument("--restoration-restart-iters", type=int, default=400)
    ap.add_argument("--restoration-exact-workspace-reuse",
                    dest="restoration_exact_workspace_reuse",
                    action="store_true", default=True)
    ap.add_argument("--no-restoration-exact-workspace-reuse",
                    dest="restoration_exact_workspace_reuse",
                    action="store_false")
    ap.add_argument("--restoration-exact-sparse-row-dots",
                    dest="restoration_exact_sparse_row_dots",
                    action="store_true", default=True)
    ap.add_argument("--no-restoration-exact-sparse-row-dots",
                    dest="restoration_exact_sparse_row_dots",
                    action="store_false")
    ap.add_argument("--restoration-exact-parallel-row-dots",
                    dest="restoration_exact_parallel_row_dots",
                    action="store_true", default=False)
    ap.add_argument("--no-restoration-exact-parallel-row-dots",
                    dest="restoration_exact_parallel_row_dots",
                    action="store_false")
    ap.add_argument("--restoration-exact-parallel-min-rows", type=int,
                    default=128)
    ap.add_argument("--restoration-tangent-activation", dest="restoration_tangent_activation", action="store_true", default=True)
    ap.add_argument("--no-restoration-tangent-activation", dest="restoration_tangent_activation", action="store_false")
    ap.add_argument("--restoration-minimax-iters", type=int, default=80)
    ap.add_argument("--restoration-minimax-prox", type=float, default=1e-6)
    ap.add_argument("--prox", type=float, default=1e-6)
    ap.add_argument("--dca-extra-prox", type=float, default=0.0)
    ap.add_argument("--dca-step-control", dest="dca_step_control", action="store_true", default=True)
    ap.add_argument("--no-dca-step-control", dest="dca_step_control", action="store_false")
    ap.add_argument("--dca-step-target-abs", type=float, default=1.0)
    ap.add_argument("--dca-step-target-rel", type=float, default=1e-2)
    ap.add_argument("--dca-step-target-max", type=float, default=10.0)
    ap.add_argument("--dca-step-control-retries", type=int, default=8)
    ap.add_argument("--dca-affine-full-limit", type=int, default=4096)
    ap.add_argument("--no-dca-affine-full", action="store_true")
    ap.add_argument("--dca-corrector", dest="dca_corrector", action="store_true", default=True)
    ap.add_argument("--no-dca-corrector", dest="dca_corrector", action="store_false")
    ap.add_argument("--dca-corrector-start", type=int, default=2)
    ap.add_argument("--dca-corrector-frequency", type=int, default=1)
    ap.add_argument("--dca-iterated-soc", dest="dca_iterated_soc", action="store_true", default=True)
    ap.add_argument("--no-dca-iterated-soc", dest="dca_iterated_soc", action="store_false")
    ap.add_argument("--dca-iterated-soc-iters", type=int, default=12)
    ap.add_argument("--dca-iterated-soc-max-ratio", type=float, default=2.0)
    ap.add_argument("--dca-iterated-soc-trigger", type=float, default=1e6)
    ap.add_argument("--dca-warm-failure-limit", type=int, default=3)
    ap.add_argument("--dca-warm-cooldown", type=int, default=32)
    ap.add_argument("--dca-inner-retries", type=int, default=3)
    ap.add_argument("--dca-inner-retry-prox", type=float, default=1e-6)
    ap.add_argument("--dca-equality-funnel", type=float, default=0.0)
    ap.add_argument("--dca-equality-funnel-auto", dest="dca_equality_funnel_auto", action="store_true", default=True)
    ap.add_argument("--no-dca-equality-funnel-auto", dest="dca_equality_funnel_auto", action="store_false")
    ap.add_argument("--dca-equality-funnel-ceiling", type=float, default=1e-2)
    ap.add_argument("--dca-equality-retraction", dest="dca_equality_retraction", action="store_true", default=True)
    ap.add_argument("--no-dca-equality-retraction", dest="dca_equality_retraction", action="store_false")
    ap.add_argument("--dca-equality-retraction-max-ratio", type=float, default=2.0)
    ap.add_argument("--dca-equality-model", choices=("linearized", "condensed-retraction", "elastic-split"), default="linearized")
    ap.add_argument("--dca-condensed-newton-iters", type=int, default=8)
    ap.add_argument("--dca-condensed-krylov-iters", type=int, default=16)
    ap.add_argument("--dca-condensed-krylov-rtol", type=float, default=1e-4)
    ap.add_argument("--dca-condensed-continuation", type=int, default=16)
    ap.add_argument("--dca-condensed-tol", type=float, default=1e-10)
    ap.add_argument("--dca-condensed-regularization", type=float, default=1e-12)
    ap.add_argument("--dca-condensed-reg-growth", type=float, default=10.0)
    ap.add_argument("--dca-condensed-reg-max", type=float, default=1e-4)
    ap.add_argument("--dca-condensed-max-ratio", type=float, default=2.0)
    ap.add_argument("--dca-condensed-curvature", dest="dca_condensed_curvature", action="store_true", default=True)
    ap.add_argument("--no-dca-condensed-curvature", dest="dca_condensed_curvature", action="store_false")
    ap.add_argument("--dca-condensed-lanczos-steps", type=int, default=24)
    ap.add_argument("--dca-condensed-lanczos-starts", type=int, default=2)
    ap.add_argument("--dca-condensed-max-rank", type=int, default=64)
    ap.add_argument("--dca-condensed-scalar-safeguard-updates", type=int, default=8)
    ap.add_argument("--dca-condensed-affine-normal-reuse", dest="dca_condensed_affine_normal_reuse", action="store_true", default=True)
    ap.add_argument("--no-dca-condensed-affine-normal-reuse", dest="dca_condensed_affine_normal_reuse", action="store_false")
    ap.add_argument("--dca-condensed-quadratic-linesearch-reuse", dest="dca_condensed_quadratic_linesearch_reuse", action="store_true", default=True)
    ap.add_argument("--no-dca-condensed-quadratic-linesearch-reuse", dest="dca_condensed_quadratic_linesearch_reuse", action="store_false")
    ap.add_argument("--dca-condensed-sparse-equalities", dest="dca_condensed_sparse_equality_evaluation", action="store_true", default=True)
    ap.add_argument("--no-dca-condensed-sparse-equalities", dest="dca_condensed_sparse_equality_evaluation", action="store_false")
    ap.add_argument("--dca-condensed-cached-regularization", dest="dca_condensed_cached_regularization", action="store_true", default=True)
    ap.add_argument("--no-dca-condensed-cached-regularization", dest="dca_condensed_cached_regularization", action="store_false")
    ap.add_argument("--dca-condensed-parallel-equalities", dest="dca_condensed_parallel_equalities", action="store_true", default=False)
    ap.add_argument("--no-dca-condensed-parallel-equalities", dest="dca_condensed_parallel_equalities", action="store_false")
    ap.add_argument("--dca-condensed-parallel-min-equalities", type=int, default=128)
    ap.add_argument("--dca-equality-elastic-penalty", type=float, default=10.0)
    ap.add_argument("--dca-equality-elastic-quadratic", type=float, default=1e-8)
    ap.add_argument("--dca-equality-elastic-cap-factor", type=float, default=1.0)
    ap.add_argument("--dca-equality-elastic-post-retraction", dest="dca_equality_elastic_post_retraction", action="store_true", default=False)
    ap.add_argument("--no-dca-equality-elastic-post-retraction", dest="dca_equality_elastic_post_retraction", action="store_false")
    ap.add_argument("--no-restoration", action="store_true")
    ap.add_argument("--activation", type=float, default=1e-4)
    ap.add_argument("--max-rank", type=int, default=None,
                    help="compatibility override: set both constraint and Lagrangian rank caps")
    ap.add_argument("--max-constraint-rank", type=int, default=24)
    ap.add_argument("--max-objective-rank", type=int, default=64)
    ap.add_argument("--max-lagrangian-rank", type=int, default=64)
    ap.add_argument("--constraint-model", choices=("auto", "linearized", "majorant"), default="auto",
                    help="phase-II nonlinear-inequality model; auto linearizes after a feasible QP-dual state")
    ap.add_argument("--line-search-armijo", type=float, default=1e-4)
    ap.add_argument("--no-tangential", action="store_true")
    ap.add_argument("--trust-region", action="store_true")
    ap.add_argument("--tr-initial-radius", type=float, default=10.0)
    ap.add_argument("--tr-min-radius", type=float, default=1e-10)
    ap.add_argument("--tr-max-radius", type=float, default=10.0)
    ap.add_argument("--tr-eta-accept", type=float, default=0.10)
    ap.add_argument("--tr-eta-shrink", type=float, default=0.25)
    ap.add_argument("--tr-eta-expand", type=float, default=0.75)
    ap.add_argument("--tr-shrink", type=float, default=0.25)
    ap.add_argument("--tr-expand", type=float, default=2.0)
    ap.add_argument("--tr-boundary-fraction", type=float, default=0.80)
    ap.add_argument("--tr-boundary-rel-tol", type=float, default=1e-4)
    ap.add_argument("--tr-boundary-abs-tol", type=float, default=1e-9)
    ap.add_argument("--tr-max-rejects", type=int, default=18)
    ap.add_argument("--tr-feas-floor", type=float, default=1e-10)
    ap.add_argument("--tr-feas-growth", type=float, default=2.0)
    ap.add_argument("--tr-feas-reduction", type=float, default=0.10)
    ap.add_argument("--tr-penalty-min", type=float, default=10.0,
                    help="minimum exact-penalty coefficient for trust-region merit acceptance")
    ap.add_argument("--tr-penalty-safety", type=float, default=1.25,
                    help="safety factor applied above the current multiplier l1 norm")
    ap.add_argument("--no-soc", action="store_true")
    ap.add_argument("--soc-max-ratio", type=float, default=0.5)
    ap.add_argument("--soc-trigger-ratio", type=float, default=1000.0)
    ap.add_argument("--working-set-drop-slack", type=float, default=2e-4)
    ap.add_argument("--working-set-drop-mu", type=float, default=1e-9)
    ap.add_argument("--working-set-drop-patience", type=int, default=3)
    ap.add_argument("--no-working-set-drop", action="store_true")
    ap.add_argument("--terminal-kkt-trigger", type=float, default=1e-2)
    ap.add_argument("--terminal-kkt-iters", type=int, default=16,
                    help="local KKT steps on small systems; n>64 is hard-capped to one attempt per fresh QP dual")
    ap.add_argument("--terminal-kkt-mu-abs", type=float, default=1e-9)
    ap.add_argument("--terminal-kkt-mu-rel", type=float, default=1e-8)
    ap.add_argument("--terminal-kkt-stability", type=int, default=3)
    ap.add_argument("--terminal-kkt", action="store_true",
                    help="enable guarded KKT accelerator also for n>64 (small systems use it by default)")
    ap.add_argument("--no-terminal-kkt", action="store_true")
    ap.add_argument("--lanczos-steps", type=int, default=40)
    ap.add_argument("--lanczos-starts", type=int, default=2)
    ap.add_argument("--exact-eigen-threshold", type=int, default=256)
    ap.add_argument("--strict-scalar-certificate", action="store_true")
    ap.add_argument("--live-iterations", action="store_true",
                    help="print a compact flushed Lazy-DCA outer-iteration table")
    ap.add_argument("--live-iterations-every", type=int, default=1)
    ap.add_argument("--live-header-every", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", type=int, choices=(0, 1), default=1)
    argv = [
        ("--qp-" + token[5:].replace("_", "-"))
        if token.startswith("--qp_") else token
        for token in sys.argv[1:]
    ]
    args = ap.parse_args(argv)
    if args.terminal_kkt and args.no_terminal_kkt:
        ap.error("--terminal-kkt and --no-terminal-kkt are mutually exclusive")
    if args.live_iterations_every < 1 or args.live_header_every < 1:
        ap.error("live-iteration periods must be positive integers")
    if args.equality_qp_original_kkt_gmres_max_iterations < 0:
        ap.error("--equality-qp-original-kkt-gmres-max-iterations must be nonnegative")
    if args.restoration_exact_parallel_min_rows < 1:
        ap.error("--restoration-exact-parallel-min-rows must be positive")
    if args.working_set_drop_patience < 1:
        ap.error("--working-set-drop-patience must be positive")
    if args.verbose == 0:
        args.quiet = True
    print("[RUNNER] settings:", json.dumps({
        "problem": args.problem, "reformulation": args.reformulation,
        "algorithm": args.algorithm, "auto_sqp_max_n": args.auto_sqp_max_n,
        "inner_backend": args.inner_backend, "linear_backend": args.linear_backend,
        "equality_qp_original_kkt_gmres": args.equality_qp_original_kkt_gmres,
        "equality_qp_original_kkt_gmres_max_iterations":
            args.equality_qp_original_kkt_gmres_max_iterations,
        "qp_option": args.qp_option, "qp_amrs": args.qp_amrs,
        "qp_init": args.qp_init, "qp_backend": args.qp_backend,
        "qp_rho_trial_every": args.qp_rho_trial_every,
        "qp_max_atoms": args.qp_max_atoms,
        "qp_seed_coverage": args.qp_seed_coverage,
        "qp_seed_stall_tol": args.qp_seed_stall_tol,
        "qp_seed_patience": args.qp_seed_patience,
        "qp_atom_events": args.qp_atom_events,
        "qp_atom_event_every": args.qp_atom_event_every,
        "qp_bound_push": args.qp_bound_push,
        "qp_bp_push_kappa": args.qp_bp_push_kappa,
        "qp_bp_margin_tol": args.qp_bp_margin_tol,
        "qp_escapes": args.qp_escapes,
        "qp_mehrotra": args.qp_mehrotra,
        "qp_sigma_max": args.qp_sigma_max,
        "qp_model_solved_tol": args.qp_model_solved_tol,
        "qp_dca_stop": args.qp_dca_stop,
        "qp_dca_stall_tol_rel": args.qp_dca_stall_tol_rel,
        "qp_dca_stall_iters": args.qp_dca_stall_iters,
        "qp_dca_monotone_tol_abs": args.qp_dca_monotone_tol_abs,
        "qp_bp_adaptive": args.qp_bp_adaptive,
        "linear_metis": args.linear_metis,
        "work_dir": str(args.work_dir), "time_limit": args.time_limit,
        "max_outer": args.max_outer, "max_inner": args.max_inner,
        "feas_tol": args.feas_tol, "opt_tol": args.opt_tol,
        "dual_inf_tol": args.dual_inf_tol, "compl_inf_tol": args.compl_inf_tol,
        "s_max": args.s_max,
        "inner_tol": args.inner_tol, "inner_mu_fallback_tol": args.inner_mu_fallback_tol,
        "lazyqp_mu0": args.lazyqp_mu0, "lazyqp_mu_r": args.lazyqp_mu_r,
        "lazyqp_mu_theta": args.lazyqp_mu_theta, "lazyqp_mu_inner_max": args.lazyqp_mu_inner_max,
        "lazyqp_kkt_form": args.lazyqp_kkt_form,
        "lazyqp_low_rank_form": args.lazyqp_low_rank_form,
        "lazyqp_condense_threshold": args.lazyqp_condense_threshold,
        "lazyqp_reduced_kkt_min_ineq": args.lazyqp_reduced_kkt_min_ineq,
        "lazyqp_warm_guard": args.lazyqp_warm_guard,
        "inner_warm_start_mode": args.inner_warm_start_mode, "inner_bound_push": args.inner_bound_push,
        "inner_bound_frac": args.inner_bound_frac, "inner_bound_push_ref": args.inner_bound_push_ref,
        "inner_bound_mult_init": args.inner_bound_mult_init,
        "restoration_iters": args.restoration_iters,
        "restoration_restarts": args.restoration_restarts,
        "restoration_restart_iters": args.restoration_restart_iters,
        "restoration_exact_workspace_reuse":
            args.restoration_exact_workspace_reuse,
        "restoration_exact_sparse_row_dots":
            args.restoration_exact_sparse_row_dots,
        "restoration_exact_parallel_row_dots":
            args.restoration_exact_parallel_row_dots,
        "restoration_exact_parallel_min_rows":
            args.restoration_exact_parallel_min_rows,
        "restoration_tangent_activation": args.restoration_tangent_activation,
        "restoration_minimax_iters": args.restoration_minimax_iters,
        "restoration_minimax_prox": args.restoration_minimax_prox,
        "prox": args.prox, "dca_extra_prox": args.dca_extra_prox,
        "dca_equality_model": args.dca_equality_model,
        "dca_condensed_newton_iters": args.dca_condensed_newton_iters,
        "dca_condensed_krylov_iters": args.dca_condensed_krylov_iters,
        "dca_condensed_krylov_rtol": args.dca_condensed_krylov_rtol,
        "dca_condensed_continuation": args.dca_condensed_continuation,
        "dca_condensed_tol": args.dca_condensed_tol,
        "dca_condensed_regularization": args.dca_condensed_regularization,
        "dca_condensed_reg_growth": args.dca_condensed_reg_growth,
        "dca_condensed_reg_max": args.dca_condensed_reg_max,
        "dca_condensed_max_ratio": args.dca_condensed_max_ratio,
        "dca_condensed_curvature": args.dca_condensed_curvature,
        "dca_condensed_lanczos_steps": args.dca_condensed_lanczos_steps,
        "dca_condensed_lanczos_starts": args.dca_condensed_lanczos_starts,
        "dca_condensed_max_rank": args.dca_condensed_max_rank,
        "dca_condensed_affine_normal_reuse": args.dca_condensed_affine_normal_reuse,
        "dca_condensed_quadratic_linesearch_reuse": args.dca_condensed_quadratic_linesearch_reuse,
        "dca_condensed_sparse_equality_evaluation": args.dca_condensed_sparse_equality_evaluation,
        "dca_condensed_cached_regularization": args.dca_condensed_cached_regularization,
        "dca_condensed_parallel_equalities": args.dca_condensed_parallel_equalities,
        "dca_condensed_parallel_min_equalities": args.dca_condensed_parallel_min_equalities,
        "dca_equality_elastic_penalty": args.dca_equality_elastic_penalty,
        "dca_equality_elastic_quadratic": args.dca_equality_elastic_quadratic,
        "dca_equality_elastic_cap_factor": args.dca_equality_elastic_cap_factor,
        "dca_equality_elastic_post_retraction": args.dca_equality_elastic_post_retraction,
        "dca_step_control": args.dca_step_control,
        "dca_step_target_abs": args.dca_step_target_abs,
        "dca_step_target_rel": args.dca_step_target_rel,
        "dca_step_target_max": args.dca_step_target_max,
        "dca_step_control_retries": args.dca_step_control_retries,
        "dca_affine_full_limit": args.dca_affine_full_limit,
        "dca_affine_full": not args.no_dca_affine_full,
        "dca_equality_funnel": args.dca_equality_funnel,
        "dca_equality_funnel_auto": args.dca_equality_funnel_auto,
        "dca_equality_funnel_ceiling": args.dca_equality_funnel_ceiling,
        "dca_equality_retraction": args.dca_equality_retraction,
        "dca_equality_retraction_max_ratio": args.dca_equality_retraction_max_ratio,
        "live_iterations": args.live_iterations,
        "live_iterations_every": args.live_iterations_every,
        "live_header_every": args.live_header_every,
        "verbose": args.verbose,
        "max_constraint_rank": args.max_constraint_rank,
        "max_objective_rank": args.max_objective_rank,
        "max_lagrangian_rank": args.max_lagrangian_rank,
        "constraint_model": args.constraint_model,
        "line_search_armijo": args.line_search_armijo,
        "trust_region": args.trust_region,
        "tr_initial_radius": args.tr_initial_radius,
        "tr_min_radius": args.tr_min_radius,
        "tr_max_radius": args.tr_max_radius,
        "tr_eta_accept": args.tr_eta_accept,
        "tr_eta_shrink": args.tr_eta_shrink,
        "tr_eta_expand": args.tr_eta_expand,
        "tr_shrink": args.tr_shrink,
        "tr_expand": args.tr_expand,
        "tr_boundary_fraction": args.tr_boundary_fraction,
        "tr_boundary_rel_tol": args.tr_boundary_rel_tol,
        "tr_boundary_abs_tol": args.tr_boundary_abs_tol,
        "tr_max_rejects": args.tr_max_rejects,
        "tr_feas_floor": args.tr_feas_floor,
        "tr_feas_growth": args.tr_feas_growth,
        "tr_feas_reduction": args.tr_feas_reduction,
        "tr_penalty_min": args.tr_penalty_min,
        "tr_penalty_safety": args.tr_penalty_safety,
        "soc_enabled": not args.no_soc,
        "soc_max_ratio": args.soc_max_ratio, "soc_trigger_ratio": args.soc_trigger_ratio,
        "working_set_drop_slack": args.working_set_drop_slack,
        "working_set_drop_mu": args.working_set_drop_mu,
        "working_set_drop_patience": args.working_set_drop_patience,
        "terminal_kkt_trigger": args.terminal_kkt_trigger,
        "terminal_kkt_iters": args.terminal_kkt_iters,
        "terminal_kkt_mu_abs": args.terminal_kkt_mu_abs,
        "terminal_kkt_mu_rel": args.terminal_kkt_mu_rel,
        "terminal_kkt_stability": args.terminal_kkt_stability,
    }, sort_keys=True))

    if args.cache:
        args.cache.mkdir(parents=True, exist_ok=True)
        os.environ["PYCUTEST_CACHE"] = str(args.cache.resolve())

    args.work_dir.mkdir(parents=True, exist_ok=True)
    tag = args.problem
    if args.sif_param:
        tag += "_" + "_".join(f"{k}{v}" for k, v in args.sif_param)
    safe_tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
    model = args.work_dir / f"{safe_tag}.qcq"
    result = args.work_dir / f"{safe_tag}.json"
    sif_params = dict(args.sif_param)

    try:
        info = export_problem(
            args.problem, model, sif_params, check=not args.no_check,
            reformulation=args.reformulation,
        )
    except Exception as exc:
        message = str(exc)
        status = "not_qcqp" if "not numerically quadratic" in message else "export_failure"
        data = {
            "problem": args.problem,
            "status": status,
            "message": message,
            "effective_outer_algorithm": None,
            "linear_solver_backend": None,
            "objective": None,
            "cutest_objective": None,
            "feasibility": None,
            "cutest_feasibility": None,
            "model_stationarity_inf": None,
            "dual_infeasibility_inf": None,
            "complementarity_inf": None,
            "scaled_nlp_error": None,
            "ipopt_desired_convergence": False,
            "quadratic_validation_passed": False,
            "reformulation": args.reformulation,
            "outer_iterations": 0,
            "inner_iterations": 0,
            "elapsed_seconds": None,
        }
        result.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"[EXPORT-REJECTED] problem={args.problem} status={status} reason={message}")
        print("[RESULT]", json.dumps(data, sort_keys=True))
        if not args.keep_model:
            try:
                model.unlink()
            except FileNotFoundError:
                pass
        raise SystemExit(2)
    print("[1/2] CUTEst export:", json.dumps(info, sort_keys=True))

    cmd = [
        str(args.binary), "--problem", str(model), "--json", str(result),
        "--algorithm", str(args.algorithm),
        "--auto-sqp-max-n", str(args.auto_sqp_max_n),
        "--inner-backend", str(args.inner_backend),
        "--linear-backend", str(args.linear_backend),
        "--qp-option", str(args.qp_option),
        "--qp-init", str(args.qp_init),
        "--qp-backend", str(args.qp_backend),
        "--qp-rho-trial-every", str(args.qp_rho_trial_every),
        "--qp-max-atoms", str(args.qp_max_atoms),
        "--qp-seed-coverage", str(args.qp_seed_coverage),
        "--qp-seed-stall-tol", str(args.qp_seed_stall_tol),
        "--qp-seed-patience", str(args.qp_seed_patience),
        "--qp-atom-event-every", str(args.qp_atom_event_every),
        "--qp-bound-push", str(args.qp_bound_push),
        "--qp-bp-push-kappa", str(args.qp_bp_push_kappa),
        "--qp-bp-margin-tol", str(args.qp_bp_margin_tol),
        "--qp-escapes", str(args.qp_escapes),
        "--qp-sigma-max", str(args.qp_sigma_max),
        "--qp-model-solved-tol", str(args.qp_model_solved_tol),
        "--qp-dca-stall-tol-rel", str(args.qp_dca_stall_tol_rel),
        "--qp-dca-stall-iters", str(args.qp_dca_stall_iters),
        "--qp-dca-monotone-tol-abs", str(args.qp_dca_monotone_tol_abs),
        "--linear-metis", str(args.linear_metis),
        "--kkt-primal-reg", str(args.kkt_primal_reg),
        "--kkt-dual-reg", str(args.kkt_dual_reg),
        "--rho-x-floor", str(args.rho_x_floor),
        "--rho-x-growth", str(args.rho_x_growth),
        "--rho-x-decay", str(args.rho_x_decay),
        "--rho-x-max", str(args.rho_x_max),
        "--rho-x-clean", str(args.rho_x_clean),
        "--delta-dual-base", str(args.delta_dual_base),
        "--delta-dual-floor", str(args.delta_dual_floor),
        "--delta-dual-max", str(args.delta_dual_max),
        "--delta-dual-mode", str(args.delta_dual_mode),
        "--kkt-berr-tol", str(args.kkt_berr_tol),
        "--kkt-berr-reject", str(args.kkt_berr_reject),
        "--kkt-refine", str(args.kkt_refine),
        "--equality-qp-original-kkt-gmres-max-iterations",
        str(args.equality_qp_original_kkt_gmres_max_iterations),
        "--max-outer", str(args.max_outer), "--max-inner", str(args.max_inner),
        "--time-limit", str(args.time_limit), "--feas-tol", str(args.feas_tol),
        "--opt-tol", str(args.opt_tol),
        "--dual-inf-tol", str(args.dual_inf_tol),
        "--compl-inf-tol", str(args.compl_inf_tol),
        "--s-max", str(args.s_max),
        "--inner-tol", str(args.inner_tol),
        "--inner-mu-fallback-tol", str(args.inner_mu_fallback_tol),
        "--lazyqp-mu0", str(args.lazyqp_mu0),
        "--lazyqp-mu-r", str(args.lazyqp_mu_r),
        "--lazyqp-mu-theta", str(args.lazyqp_mu_theta),
        "--lazyqp-mu-inner-max", str(args.lazyqp_mu_inner_max),
        "--lazyqp-kkt-form", str(args.lazyqp_kkt_form),
        "--lazyqp-reduced-kkt-min-ineq", str(args.lazyqp_reduced_kkt_min_ineq),
        "--lazyqp-low-rank-form", str(args.lazyqp_low_rank_form),
        "--lazyqp-condense-threshold", str(args.lazyqp_condense_threshold),
        "--lazyqp-warm-guard", str(args.lazyqp_warm_guard),
        "--inner-bound-push", str(args.inner_bound_push),
        "--inner-bound-frac", str(args.inner_bound_frac),
        "--inner-bound-push-ref", str(args.inner_bound_push_ref),
        "--inner-bound-mult-init", str(args.inner_bound_mult_init),
        "--restoration-iters", str(args.restoration_iters),
        "--restoration-restarts", str(args.restoration_restarts),
        "--restoration-restart-iters", str(args.restoration_restart_iters),
        "--restoration-exact-parallel-min-rows",
        str(args.restoration_exact_parallel_min_rows),
        "--restoration-minimax-iters", str(args.restoration_minimax_iters),
        "--restoration-minimax-prox", str(args.restoration_minimax_prox),
        "--prox", str(args.prox),
        "--dca-extra-prox", str(args.dca_extra_prox),
        "--dca-step-target-abs", str(args.dca_step_target_abs),
        "--dca-step-target-rel", str(args.dca_step_target_rel),
        "--dca-step-target-max", str(args.dca_step_target_max),
        "--dca-step-control-retries", str(args.dca_step_control_retries),
        "--dca-affine-full-limit", str(args.dca_affine_full_limit),
        "--dca-corrector-start", str(args.dca_corrector_start),
        "--dca-corrector-frequency", str(args.dca_corrector_frequency),
        "--dca-iterated-soc-iters", str(args.dca_iterated_soc_iters),
        "--dca-iterated-soc-max-ratio", str(args.dca_iterated_soc_max_ratio),
        "--dca-iterated-soc-trigger", str(args.dca_iterated_soc_trigger),
        "--dca-warm-failure-limit", str(args.dca_warm_failure_limit),
        "--dca-warm-cooldown", str(args.dca_warm_cooldown),
        "--dca-inner-retries", str(args.dca_inner_retries),
        "--dca-inner-retry-prox", str(args.dca_inner_retry_prox),
        "--dca-equality-funnel", str(args.dca_equality_funnel),
        "--dca-equality-funnel-ceiling", str(args.dca_equality_funnel_ceiling),
        "--dca-equality-retraction-max-ratio", str(args.dca_equality_retraction_max_ratio),
        "--dca-equality-model", str(args.dca_equality_model),
        "--dca-condensed-newton-iters", str(args.dca_condensed_newton_iters),
        "--dca-condensed-krylov-iters", str(args.dca_condensed_krylov_iters),
        "--dca-condensed-krylov-rtol", str(args.dca_condensed_krylov_rtol),
        "--dca-condensed-continuation", str(args.dca_condensed_continuation),
        "--dca-condensed-tol", str(args.dca_condensed_tol),
        "--dca-condensed-regularization", str(args.dca_condensed_regularization),
        "--dca-condensed-reg-growth", str(args.dca_condensed_reg_growth),
        "--dca-condensed-reg-max", str(args.dca_condensed_reg_max),
        "--dca-condensed-max-ratio", str(args.dca_condensed_max_ratio),
        "--dca-condensed-lanczos-steps", str(args.dca_condensed_lanczos_steps),
        "--dca-condensed-lanczos-starts", str(args.dca_condensed_lanczos_starts),
        "--dca-condensed-max-rank", str(args.dca_condensed_max_rank),
        "--dca-condensed-scalar-safeguard-updates", str(args.dca_condensed_scalar_safeguard_updates),
        "--dca-equality-elastic-penalty", str(args.dca_equality_elastic_penalty),
        "--dca-equality-elastic-quadratic", str(args.dca_equality_elastic_quadratic),
        "--dca-equality-elastic-cap-factor", str(args.dca_equality_elastic_cap_factor),
        "--activation", str(args.activation),
        "--max-constraint-rank", str(args.max_constraint_rank),
        "--max-objective-rank", str(args.max_objective_rank),
        "--max-lagrangian-rank", str(args.max_lagrangian_rank),
        "--constraint-model", str(args.constraint_model),
        "--line-search-armijo", str(args.line_search_armijo),
        "--tr-initial-radius", str(args.tr_initial_radius),
        "--tr-min-radius", str(args.tr_min_radius),
        "--tr-max-radius", str(args.tr_max_radius),
        "--tr-eta-accept", str(args.tr_eta_accept),
        "--tr-eta-shrink", str(args.tr_eta_shrink),
        "--tr-eta-expand", str(args.tr_eta_expand),
        "--tr-shrink", str(args.tr_shrink),
        "--tr-expand", str(args.tr_expand),
        "--tr-boundary-fraction", str(args.tr_boundary_fraction),
        "--tr-boundary-rel-tol", str(args.tr_boundary_rel_tol),
        "--tr-boundary-abs-tol", str(args.tr_boundary_abs_tol),
        "--tr-max-rejects", str(args.tr_max_rejects),
        "--tr-feas-floor", str(args.tr_feas_floor),
        "--tr-feas-growth", str(args.tr_feas_growth),
        "--tr-feas-reduction", str(args.tr_feas_reduction),
        "--tr-penalty-min", str(args.tr_penalty_min),
        "--tr-penalty-safety", str(args.tr_penalty_safety),
        "--soc-max-ratio", str(args.soc_max_ratio),
        "--soc-trigger-ratio", str(args.soc_trigger_ratio),
        "--working-set-drop-slack", str(args.working_set_drop_slack),
        "--working-set-drop-mu", str(args.working_set_drop_mu),
        "--working-set-drop-patience", str(args.working_set_drop_patience),
        "--terminal-kkt-trigger", str(args.terminal_kkt_trigger),
        "--terminal-kkt-iters", str(args.terminal_kkt_iters),
        "--terminal-kkt-mu-abs", str(args.terminal_kkt_mu_abs),
        "--terminal-kkt-mu-rel", str(args.terminal_kkt_mu_rel),
        "--terminal-kkt-stability", str(args.terminal_kkt_stability),
        "--lanczos-steps", str(args.lanczos_steps), "--lanczos-starts", str(args.lanczos_starts),
        "--exact-eigen-threshold", str(args.exact_eigen_threshold),
    ]
    if args.qp_amrs:
        cmd.append("--qp-amrs")
    else:
        cmd.append("--no-qp-amrs")
    if args.qp_atom_events:
        cmd.append("--qp-atom-events")
    else:
        cmd.append("--no-qp-atom-events")
    if args.qp_mehrotra:
        cmd.append("--qp-mehrotra")
    else:
        cmd.append("--no-qp-mehrotra")
    if args.qp_dca_stop:
        cmd.append("--qp-dca-stop")
    else:
        cmd.append("--no-qp-dca-stop")
    if args.qp_bp_adaptive:
        cmd.append("--qp-bp-adaptive")
    else:
        cmd.append("--no-qp-bp-adaptive")
    if args.max_rank is not None:
        cmd.extend(["--max-rank", str(args.max_rank)])
    if args.trust_region:
        cmd.append("--trust-region")
    if args.no_tangential:
        cmd.append("--no-tangential")
    if args.no_soc:
        cmd.append("--no-soc")
    if args.no_working_set_drop:
        cmd.append("--no-working-set-drop")
    if args.terminal_kkt:
        cmd.append("--terminal-kkt")
    if args.no_terminal_kkt:
        cmd.append("--no-terminal-kkt")
    if args.strict_scalar_certificate:
        cmd.append("--strict-scalar-certificate")
    if args.kkt_adaptive_reg:
        cmd.append("--kkt-adaptive-reg")
    cmd.append("--equality-qp-original-kkt-gmres" if
               args.equality_qp_original_kkt_gmres else
               "--no-equality-qp-original-kkt-gmres")
    cmd.append("--kkt-persistent-symbolic-cache" if args.kkt_persistent_symbolic_cache else "--no-kkt-persistent-symbolic-cache")
    cmd.append("--restoration-tangent-activation" if args.restoration_tangent_activation else "--no-restoration-tangent-activation")
    cmd.append("--restoration-exact-workspace-reuse" if
               args.restoration_exact_workspace_reuse else
               "--no-restoration-exact-workspace-reuse")
    cmd.append("--restoration-exact-sparse-row-dots" if
               args.restoration_exact_sparse_row_dots else
               "--no-restoration-exact-sparse-row-dots")
    cmd.append("--restoration-exact-parallel-row-dots" if
               args.restoration_exact_parallel_row_dots else
               "--no-restoration-exact-parallel-row-dots")
    cmd.append("--dca-step-control" if args.dca_step_control else "--no-dca-step-control")
    if args.no_dca_affine_full:
        cmd.append("--no-dca-affine-full")
    cmd.append("--dca-corrector" if args.dca_corrector else "--no-dca-corrector")
    cmd.append("--dca-iterated-soc" if args.dca_iterated_soc else "--no-dca-iterated-soc")
    cmd.append("--dca-equality-funnel-auto" if args.dca_equality_funnel_auto else "--no-dca-equality-funnel-auto")
    cmd.append("--dca-equality-retraction" if args.dca_equality_retraction else "--no-dca-equality-retraction")
    cmd.append("--dca-condensed-curvature" if args.dca_condensed_curvature else "--no-dca-condensed-curvature")
    cmd.append("--dca-condensed-affine-normal-reuse" if args.dca_condensed_affine_normal_reuse else "--no-dca-condensed-affine-normal-reuse")
    cmd.append("--dca-condensed-quadratic-linesearch-reuse" if args.dca_condensed_quadratic_linesearch_reuse else "--no-dca-condensed-quadratic-linesearch-reuse")
    cmd.append("--dca-condensed-sparse-equalities" if args.dca_condensed_sparse_equality_evaluation else "--no-dca-condensed-sparse-equalities")
    cmd.append("--dca-condensed-cached-regularization" if args.dca_condensed_cached_regularization else "--no-dca-condensed-cached-regularization")
    cmd.append("--dca-condensed-parallel-equalities" if args.dca_condensed_parallel_equalities else "--no-dca-condensed-parallel-equalities")
    cmd.extend(["--dca-condensed-parallel-min-equalities", str(args.dca_condensed_parallel_min_equalities)])
    cmd.append("--dca-equality-elastic-post-retraction" if args.dca_equality_elastic_post_retraction else "--no-dca-equality-elastic-post-retraction")
    cmd.append({"equality":"--inner-equality-warm-start", "full":"--inner-full-dual-warm-start", "off":"--no-inner-warm-start"}[args.inner_warm_start_mode])
    if args.no_restoration:
        cmd.append("--no-restoration")
    if args.live_iterations:
        cmd.extend(["--live-iterations",
                    "--live-iterations-every", str(args.live_iterations_every),
                    "--live-header-every", str(args.live_header_every)])
    if args.quiet:
        cmd.append("--quiet")

    print("[2/2] Solver:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if result.exists():
        data = json.loads(result.read_text())
        x = data.get("x") or []
        n_original = int(info["n_original"])
        if len(x) >= n_original:
            original_x = x[:n_original]
            data.update(evaluate_cutest_solution(args.problem, original_x, sif_params))
            data["reformulation"] = info["reformulation"]
            data["quadratic_validation_passed"] = True
            for key in (
                "quadratic_validation_probes",
                "quadratic_validation_objective_error",
                "quadratic_validation_constraint_error",
                "quadratic_validation_worst_constraint",
            ):
                data[key] = info.get(key)
            data["n_original"] = n_original
            data["n_lifted"] = int(info["n"])
            data["abs_lifts"] = int(info["abs_lifts"])
            data["lift_variables"] = info["lift_variables"]
            if info["abs_lifts"]:
                errs = []
                for k, j in enumerate(info["lift_variables"]):
                    t = float(x[n_original+k])
                    q = float(x[j])
                    errs.append(abs(t - abs(q)))
                data["lift_abs_consistency"] = max(errs, default=0.0)
            else:
                data["lift_abs_consistency"] = 0.0
            if data.get("objective") is not None:
                data["objective_vs_cutest_abs_error"] = abs(
                    float(data["objective"]) - float(data["cutest_objective"])
                )
            result.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

        print("[RESULT]", json.dumps({
            k: data.get(k) for k in (
                "problem", "status", "linear_solver_backend", "effective_outer_algorithm",
                "objective", "cutest_objective",
                "feasibility", "cutest_feasibility", "lift_abs_consistency",
                "outer_iterations", "inner_iterations", "elapsed_seconds",
                "model_stationarity_inf", "dual_infeasibility_inf",
                "complementarity_inf", "scaled_nlp_error",
                "dual_error_scaling", "complementarity_error_scaling",
                "ipopt_desired_convergence",
                "termination_multiplier_source",
                "termination_multiplier_fit_converged",
                "termination_multiplier_fit_sweeps",
                "termination_multiplier_fit_projected_gradient_inf",
                "lagrangian_correction_rank",
                "inequality_total_rank", "working_set_additions", "working_set_drops",
                "active_jacobian_rank", "tangent_dimension",
                "reduced_negative_eigenvalues", "reduced_min_eigenvalue",
                "terminal_kkt_iterations", "terminal_kkt_accepted_steps",
                "terminal_kkt_attempts", "terminal_kkt_rejected_steps",
                "terminal_kkt_skipped_unstable",
                "terminal_kkt_active_stability",
                "terminal_kkt_multiplier_cutoff",
                "restoration_iterations", "restoration_restarts",
                "restoration_structured_abs_links", "restoration_effective_dimension",
                "restoration_initial_feasibility", "restoration_final_feasibility",
                "restoration_row_space_solves",
                "restoration_harmonic_attempts", "restoration_harmonic_acceptances",
                "restoration_harmonic_factorizations", "restoration_harmonic_feasibility",
                "restoration_tangent_activation_attempts",
                "restoration_tangent_activation_acceptances",
                "restoration_tangent_activation_components",
                "restoration_tangent_activation_feasibility",
                "restoration_tangent_activation_step_inf",
                "lazy_dca_step_control_predictions",
                "lazy_dca_step_control_retries",
                "lazy_dca_max_raw_direction_inf",
                "lazy_dca_last_raw_direction_inf",
                "lazy_dca_last_step_target_inf",
                "lazy_dca_max_model_proximal",
                "final_proximal_regularization",
                "last_inner_dual_inf", "last_inner_equality_inf",
                "last_inner_inequality_inf", "last_inner_primal_residual_inf",
                "last_inner_complementarity", "last_inner_used_iteration_limit_fallback",
                "inner_iteration_limit_fallback_acceptances",
                "inner_dual_warm_start_acceptances",
                "inner_full_dual_warm_start_acceptances",
                "inner_dual_warm_start_rejections",
                "lazyqp_reduced_kkt_factorizations",
                "lazyqp_full_kkt_factorizations",
                "lazyqp_reduced_kkt_fallbacks",
                "lazyqp_max_kkt_dimension", "lazyqp_max_full_kkt_dimension",
                "kkt_factorizations", "kkt_symbolic_analyses",
                "line_search_backtracks", "armijo_rejections",
                "lazy_dca_equality_model",
                "lazy_dca_condensed_factorizations",
                "lazy_dca_condensed_normal_solves",
                "lazy_dca_condensed_symbolic_analyses",
                "lazy_dca_condensed_symbolic_reuses",
                "lazy_dca_condensed_retraction_attempts",
                "lazy_dca_condensed_retraction_acceptances",
                "lazy_dca_condensed_accepted_steps",
                "lazy_dca_condensed_retraction_rejections",
                "lazy_dca_condensed_continuation_attempts",
                "lazy_dca_condensed_continuation_acceptances",
                "lazy_dca_condensed_newton_iterations",
                "lazy_dca_condensed_krylov_iterations",
                "lazy_dca_condensed_backtracks",
                "lazy_dca_condensed_tangent_curvature_calls",
                "lazy_dca_condensed_tangent_rank_max",
                "lazy_dca_condensed_tangent_shift_max",
                "lazy_dca_condensed_initial_tangent_min_eigenvalue",
                "lazy_dca_condensed_final_tangent_min_eigenvalue",
                "lazy_dca_condensed_normal_regularization",
                "lazy_dca_condensed_last_residual_inf",
                "lazy_dca_condensed_max_correction_inf",
                "lazy_dca_condensed_max_correction_ratio",
                "soc_attempts", "soc_accepted_steps", "soc_rejected_steps",
                "soc_inner_iterations", "soc_factorizations",
                "reformulation", "abs_lifts",
            )
        }, sort_keys=True))
    if not args.keep_model:
        try:
            model.unlink()
        except FileNotFoundError:
            pass
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
