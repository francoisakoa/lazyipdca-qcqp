#include <set>
#include "factorial_trace.hpp"
#include "native_signed_square_reduction.hpp"
#include "factorial_structure_inspection.hpp"
#include "lazyipdca_qcqp/io.hpp"
#include "lazyipdca_qcqp/rank_presolve.hpp"
#include "accepted_checkpoint_writer.hpp"
#include "frozen_start_json.hpp"
#include "lazyipdca_qcqp/solver.hpp"
#include "lazyipdca_qcqp/version.hpp"
#include "lazyipdca_qcqp/linear_solver.hpp"
#include "lazyipdca_qcqp/restoration.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <chrono>
#include "newton_cli_report.hpp"
#include <stdexcept>
#include <string>

using namespace lazyipdca::qcqp;

namespace {
void usage(const char* exe) {
  std::cerr
    << "Usage: " << exe << " --problem FILE.qcq [options]\n"
    << "Options:\n"
    << "  --hsl-library FILE           MA27 shared library; defaults to HSL_LIBRARY/environment search\n"
    << "  --hsl-pivot-tol X            MA27 threshold in (0,0.5] (default 1e-8)\n"
    << "  --hsl-memory-factor X        MA27 workspace growth in (1,64] (default 1.5)\n"
    << "  --accepted-checkpoint 0|1      save accepted diagnostic state to --json+.accepted.json (majorants only, default off)\n"
    << "  --json FILE                    write machine-readable result\n"
    << "  --factorial-structure-preflight       inspect exact representation coupling; no solver\n"
    << "  --factorial-inequalities cache|hybrid  declared research factor; objective exact\n"
    << "  --factorial-globalization r3|hs       independent published control package\n"
    << "  --factorial-trace FILE                accepted-state digest/origin JSONL\n"
    << "  --factorial-validation-legacy-objective  HH diagonal validation only\n"
    << "  --frozen-start FILE            frozen JSON x0 only; no dual/slack/history import\n"
    << "  --algorithm NAME               lazy-dca (default); lazy-newton (v0.4.28r3p28 curvature route); auto is an alias; legacy-sqp is comparison only\n"
    << "  --inner-backend lazyqp-sparse  v0.4.28r3p28: Lazy-DCA envelope with the sparse single-loop kernel (lazy-newton machinery)\n"
    << "  --newton-engine legacy|atomic      choose original filter engine or compact atomic engine\n"
    << "  --dca-rho-refinements N       certified scalar refinement trials (3, range 0..12)\n"
    << "  --dca-initial-restoration     enable the legacy initial feasibility phase\n"
    << "  --dca-normal-weighted-atoms   legacy atom weights including normal completion\n"
    << "  G6 research controls (off by default; each accepts --no-...):\n"
    << "  --dca-active-evaluation      skip derivatives of inactive inequality rows\n"
    << "  --dca-packed-atoms           apply atoms on retained coordinates only\n"
    << "  --dca-lazy-merit             defer private merit work until original-filter rejection\n"
    << "  --dca-block-ritz             admit independent directions from a shared Ritz block\n"
    << "  --dca-anderson               enable filtered multi-step acceleration\n"
    << "  --dca-krylov                 G9: original KKT acceleration using the DCA factor\n"
    << "  --no-dca-krylov              disable G9 acceleration (default)\n"
    << "  --dca-krylov-max N           GMRES budget per right-hand side (24; 0 disables trials)\n"
    << "  --dca-krylov-cooldown N      outer iterations skipped after rejection (2)\n"
    << "  --no-dca-krylov-pc           use one centered original RHS instead of predictor-corrector\n"
    << "  --dca-rank-audit-mode M      Original audit: full|mapped|check (default mapped); check is diagnostic\n"
    << "  --dca-rank-presolve 0|1     exact coefficient reductions before iteration 0 (0)\n"
    << "  --power2-scaling 0|1 experimental exact dyadic coordinate/row scaling (0)\n"
    << "  --product-definition-presolve 0|1 experimental exact unit-product substitution (0)\n"
    << "  --native-augmented-repair 0|1 native PHR/free-bound-face recovery, original QCQP KKT (0)\n"
    << "  --free-epigraph-presolve 0|1  exact objective-free negative-square leaves; adaptive study only (1)\n"
    << "  --dca-rank-implied-bounds 0|1  remove box sides proved by a retained quadratic (1)\n"
    << "  --dca-rank-presolve-report FILE  JSON of verified reductions\n"
    << "  --dca-aggregate              certified aggregate barrier/Lagrangian quadratic DCA\n"
    << "  --dca-aggregate-kkt-layout M persistent|dynamic border slots (persistent)\n"
    << "  --dca-aggregate-border-reserve X capacity multiplier in [1,4] (1.25)\n"
    << "  --dca-aggregate-fast-assembly 0|1 cached MUMPS assembly (1)\n"
    << "  --dca-aggregate-rho-memory 0|1 retain positive curvature scale (1)\n"
    << "  --dca-aggregate-rejection-policy M separated|legacy (separated)\n"
    << "  --dca-aggregate-lean 0|1 reuse values and defer majorant layout (1)\n"
    << "  --dca-aggregate-only 0|1     no nested majorants/restoration; bounded original aggregate search (0)\n"
    << "  --dca-aggregate-retry-max N  total regularization trials per aggregate-only outer (24)\n"
    << "  --dca-aggregate-rho-initial X first positive shift, aggregate-only (1e-6)\n"
    << "  --dca-aggregate-rho-growth X shift escalation factor, aggregate-only (10)\n"
    << "  --dca-aggregate-rho-decay X  remembered shift decay, aggregate-only (0.2)\n"
    << "  --dca-aggregate-affine-retry 0|1 also test already computed predictor (1, aggregate-only)\n"
    << "  --dca-rho-complete-search 0|1  test remembered shift/cap before inertia abandonment (1)\n"
    << "  --dca-strict-finish 0|1  bounded original-precise KKT finish of acceptable points (1)\n"
    << "  --dca-strict-finish-max N  total finish attempts per solve, 0..100 (6)\n"
    << "  --dca-strict-finish-refinements N  same-factor residual corrections, 0..100 (8)\n"
    << "  --dca-strict-finish-mu-fraction X  central target fraction of strict error caps (0.01)\n"
    << "  --dca-strict-finish-eta X  terminal linear forcing factor, 0<X<1 (0.1)\n"
    << "  --dca-aggregate-preparation-cache 0|1  exact same-anchor reuse between rho probes (1)\n"
    << "  --dca-aggregate-strict-probe-stop 0|1  stop rho contest only after full strict certificate (1)\n"
    << "  --dca-aggregate-rho-max X    scalar correction cap (default 100; zero requires unshifted certificate)\n"
    << "  --dca-aggregate-backend NAME auto, cholesk, native, ldl, qdldl, mumps or ma27\n"
    << "  --dca-dual-step MODE         coupled/ipopt for filter; common for majorants (one primal-dual step)\n"
    << "  --dca-common-step-primal-trigger X compare a common feasible step below this original primal residual (0.1)\n"
    << "  --dca-common-step-dual-growth X compare only when the accepted split step grows original inf_du by X (1.25)\n"
    << "  --dca-aggregate-progress-window N accepted points used to detect aggregate/dual stalls (0 disables; default 0)\n"
    << "  --dca-kkt-growth-max X       global original-KKT growth ceiling on every accepted path (10)\n"
    << "  --dca-local-active           try certified local strongly-active DCA transactions\n"
    << "  --dca-pair-elimination       bounded sparse elimination in global/local curvature and Newton systems\n"
    << "  --dca-local-trigger X        original KKT gate for local transactions (default 1)\n"
    << "  --dca-local-primal-trigger X primal feasibility gate for local transactions (default 1e-3)\n"
    << "  --dca-local-backend NAME     local linear solver: same, cholesk, ldl, qdldl, mumps, ma27 or native\n"
    << "  --dca-local-weak-affine      use guarded affine local models for weakly active inequalities\n"
    << "  --dca-endpoint-start FILE    primal-dual endpoint vectors; reconstruct slacks, reset history\n"
    << "  --dca-tangent-curvature      use the experimental tangent curvature certificate\n"
    << "  --dca-certificate-transport  try a checked certificate bound between models\n"
    << "  --dca-sparse-fallback        use MUMPS for pivoted recovery when available\n"
    << "  --ldl-equilibration          symmetrically equilibrate LDL; keep original residual checks\n"
    << "  --[no-]cholesk-equilibration symmetric Ruiz scaling inside Cholesk (default on)\n"
    << "  --dca-curvature-backend same|cholesk|ldl|qdldl|mumps|ma27|native  curvature factor backend\n"
    << "  --no-dca-extrapolation     disable guarded local DCA extrapolation\n"
    << "  --no-dca-model-reuse       G4 control: rebuild at each accepted point\n"
    << "  --majorant-dual-diagnostics 0|1  QCQP stationarity decomposition (default 0)\n"
    << "  --majorant-curvature-corrections N  late Lazy factor corrections, 0..16 (default 0)\n"
    << "  --majorant-lambda-interpolation 0|1  minimax dual segment (default 0)\n"
    << "  --majorant-lambda-rho-retry 0|1  next auxiliary factor with reduced rho seed (default 0)\n"
    << "  --majorant-lambda-trigger consecutive|window  stagnation trigger (default consecutive)\n"
    << "  --majorant-recalc-lambda 0|1  conditional equality multiplier projection (default 0)\n"
    << "  --majorant-recalc-lambda-max-iter N  projection iteration budget (default 40)\n"
    << "  --majorant-startup-policy legacy|auto  structural central start; recovery remains available (default legacy)\n"
    << "  --majorant-filter-recenter 0|1  one primal-preserving centrality reset after exhausted filter (default 0)\n"
    << "  --majorant-filter-history-fix 0|1 paper switching/Armijo filter augmentation (default 0)\n"
    << "  --majorant-equality-stabilization 0|1 V2 dual stabilization of h and source g+s, original R3 affine f/g only (default 0)\n"
    << "  --original-gr-kkt-soc 0|1      factor-fixed general KKT SOC after GR rejection (default 0)\n"
    << "  env LAZY_ORIGINAL_SOURCE_PRIMAL_BARRIER=0|1 exact source barrier/free duals (default GR:1, otherwise:0)\n"
    << "  --majorant-restoration 0|1  permit conditional feasibility recovery (default 0)\n"
    << "  --majorant-conditional-retraction 0|1  conditional equality SOC, reused factor (default 0)\n"
    << "  --majorant-initial-dual-balance 0|1 verified affine-objective cold dual/barrier initialization (default0)\n"
    << "  --majorant-rho-feasibility-guard 0|1 hold proximal decay after physical feasibility loss (default0)\n"
    << "  --majorant-trial-soc-max-corrections N  accepted-trial SOC correction budget in [1,3] (default 1)\n"
    << "  --majorant-trial-soc-full-primal 0|1  also correct active original g+s residual (default 0)\n"
    << "  --majorant-trial-soc-arc 0|1  private SOC arc after a barrier decrease and size/domain refusal (default 0)\n"
    << "  --majorant-soc-defect-control 0|1  coefficient-verified repair after insufficient selected post-barrier SOC (default 0)\n"
    << "  --majorant-nonlinear-trial-soc 0|1 propose frozen-factor SOC on accepted nonlinear equality drift (default 0)\n"
    << "  --majorant-adaptive-globalization 0|1 complete barrier residual gate and nonlinear stalled-window SOC (default 0)\n"
    << "  --majorant-restoration-bound-stall-fix 0|1 bounded native recovery after original residual stall (default 1)\n"
    << "  --majorant-recovery-gap-fix 0|1 bounded recovery for every infeasible adaptive anchor (default 1)\n"
    << "  --majorant-soc-small-step-patience N  latch R3-study SOC after N accepted small steps (0: historical immediate)\n"
    << "  --majorant-soc-small-step-threshold X accepted-step threshold in (0,1] (default 0.125)\n"
    << "  --majorant-restoration-iterations N  conditional restoration budget (default 20)\n"
    << "  --convexification-study off|U|F|SS|SA|AS|AA|HH|HS|SH  explicit globally convex research models\n"
    << "  --study-hybrid-max-atoms N --study-hybrid-event-interval N --study-hybrid-atom-support N\n"
    << "  --study-hybrid-downweight-patience N --study-hybrid-downweight-factor V --study-hybrid-dominance V\n"
    << "  --study-hybrid-scalar-decay V --study-hybrid-margin V\n"
    << "  --study-inner-relative V --study-inner-absolute V --study-inner-min N --study-inner-max N\n"
    << "  --study-controller r3|certified (default r3: original equalities, single Newton, original filter)\n"
    << "  --study-inner-rule model-relative|barrier-relative|native-relative|native-direction-relative|native-reduced-relative|native-reduced-direction-relative|direction-relative|native-direction-soc|native-reduced-direction-soc --study-inner-margin V --study-inner-kappa-fraction V\n"
    << "  --study-inner-activity V --study-inner-feasibility V --study-inner-complementarity V\n"
    << "  --study-inner-barrier-decay V --study-inner-predictor-corrector 0|1\n"
    << "  --study-hybrid-candidates N --study-hybrid-lanczos-steps N --study-hybrid-lanczos-starts N\n"
    << "  --study-hybrid-additions-per-event N --study-hybrid-support-energy V --study-hybrid-strength V\n"
    << "  --study-hybrid-beta-offset V --study-hybrid-negative-tolerance V --study-hybrid-scalar-safety V\n"
    << "  --study-hybrid-scalar-floor V --study-hybrid-score-{activity,scalar,support,density,weight} V\n"
    << "  --study-hybrid-min-{lift,activity,score} V --study-hybrid-objective-floor V\n"
    << "  --convexification-schedule lazy|eager  first construction timing, same later refinements\n"
    << "  --objective-convexification off|lazy  optional objective majorant (default off)\n"
    << "  --objective-max-atoms N      objective atom cap (default 32)\n"
    << "  --objective-initial-atoms N  initial objective atoms (default 2)\n"
    << "  --objective-atom-batch N     atoms admitted per accepted outer step (default 1)\n"
    << "  --objective-rho-decay X       excess-rho decay after acceptance, 0<X<1 (default .5)\n"
    << "  --objective-rho-excess X      initial scalar fraction of curvature deficit (default .1)\n"
    << "  --dca-engine auto|legacy|filter|majorants  auto (default): exact-structure STANDARD selection\n"
    << "  --native-al-fallback auto|off  native AL fallback after non-KKT Lazy, shared wall budget (auto)\n"
    << "  --standard-r10-policy auto|off  native R10 on its verified coefficient domain (default auto)\n"
    << "  --gr-constraint-increments coefficients|rounded  GR source increments (default coefficients)\n"
    << "  --gr-proximal-policy historical|frozen  isolate globalization-driven proximal growth\n"
    << "  --gr-fixed-centering-steps N   bounded fixed-parameter centering experiment, 0..8 (default 0)\n"
    << "  --gr-residual-components 0|1   separate original/stabilized barrier blocks (default 1)\n"
    << "  --dca-inner-max N             inner step budget (filter: 8; ignored by majorants R8 outer-only)\n"
    << "  --dca-model-retries N         additional majorant rebuilds after first attempt (0..100; default 3)\n"
    << "  --dca-aggregate-full-refinement  Refine full reconstructed equations on precision failure (off)\n  --dca-aggregate-adaptive-dual    Activate dual perturbation after precision failure (off)\n  --acceptable-iter N          Consecutive acceptable iterates (15; 0 disables)\n  --acceptable-tol T           Normalized original KKT error (1e-5)\n  --acceptable-constr-viol-tol T  Original primal ceiling (1e-6)\n  --acceptable-dual-inf-tol T   Original dual ceiling (1e-5)\n  --acceptable-compl-inf-tol T  Original complementarity ceiling (1e-6)\n  --dca-aggregate-adaptive-pivot  retry inaccurate aggregate directions with MUMPS pivot 0.01 -> 0.1 (default off)\n"
    << "  --termination-test MODE    project (legacy original units) or ipopt (bundled 3.14.11)\n"
    << "  --dca-control-tol T        DCA numerical control tolerance, independent of Ipopt stop (1e-6)\n"
    << "  --acceptable-obj-change-tol T  Relative scaled objective change limit (1e20)\n"
    << "  --acceptable-on-failure     recover one audited acceptable point after terminal failure (default off)\n"
    << "  --no-acceptable-on-failure  disable acceptable terminal recovery\n"
    << "  --no-dca-aggregate-adaptive-pivot disable aggregate pivot recovery\n"
    << "  --mumps-memory-recovery      recover MUMPS -8/-9 on the same matrix (default on)\n"
    << "  --no-mumps-memory-recovery   historical allocation behavior, for controlled comparisons\n"
    << "  --dca-restoration-compare    compare feasible-filter candidates after aggregate certificate failure (default on)\n"
    << "  --no-dca-restoration-compare historical majorant selection, for controlled comparisons\n"
    << "  --dca-filter-diagnostics      explain each original-filter/private-merit trial (default off)\n"
    << "  --no-dca-filter-diagnostics   disable detailed filter trial diagnostics\n"
    << "  --dca-dual-decomposition      decompose original inf_du growth into primal, dual, step-coupling and direction/model terms\n"
    << "  --dca-dual-decomposition-growth X  emit decomposition when inf_du grows by at least X (1.25; 0 emits every trial)\n"
    << "  --max-atoms N                 total DCA atom storage cap, including inequalities (500)\n"
    << "  --atom-event-every N          admission period in Hessian updates (1)\n"
    << "  --atom-basis-keep N            compaction threshold of the atomic search basis (32)\n"
    << "  --atom-basis-tail N            most recent basis vectors kept by a compaction (8)\n"
    << "  --atom-probe-depth N           inverse Krylov/Ritz probe depth (18)\n"
    << "  --atom-probe-count-projection yes|no  ker(N) projection solve consumes one unit of depth (yes)\n"
    << "  --display-level 0..4           atomic engine: quiet, outer, work, atoms, trace\n"
    << "  --ln-mu-init X                 lazy-newton initial barrier parameter (default 0.1)\n"
    << "  --ln-mu-kappa X                lazy-newton linear barrier decrease factor (default 0.2)\n"
    << "  --ln-mu-theta X                lazy-newton superlinear barrier exponent (default 1.5)\n"
    << "  --ln-kappa-epsilon X           lazy-newton barrier subproblem accuracy factor (default 10)\n"
    << "  --ln-activation X              lazy-newton working-set activation, scaled distance (default 0.05)\n"
    << "  --ln-envelope X                lazy-newton structural reservation distance at the start (default 0.2)\n"
    << "  --ln-drop-patience N           lazy-newton iterations before dropping an inactive row (default 8)\n"
    << "  --ln-drop-distance X           lazy-newton scaled distance beyond which a row may be dropped (default 1)\n"
    << "  --ln-delta-c X                 lazy-newton equality-row regularization (default 1e-8)\n"
    << "  --ln-delta-x-init X            lazy-newton first inertia shift trial (default 1e-4)\n"
    << "  --ln-delta-x-min X             lazy-newton floor of the warm-started inertia shift (default 1e-10)\n"
    << "  --ln-kkt-refine N              lazy-newton iterative refinement sweeps (default 3)\n"
    << "  --ln-kkt-berr-tol X            lazy-newton refinement stop, relative backward error (default 1e-10)\n"
    << "  --ln-quadratic-correction      lazy-newton exact second-order equality correction (default on)\n"
    << "  --no-ln-quadratic-correction   disable the lazy-newton exact second-order correction\n"
    << "  --ln-max-soc N                 lazy-newton correction sweeps per line search (default 4)\n"
    << "  --ln-bound-push X              lazy-newton initial relative bound push (default 1e-2)\n"
    << "  --ln-tau-min X                 lazy-newton fraction-to-boundary floor (default 0.99)\n"
    << "  --ln-max-line-search N         lazy-newton Armijo backtracks (default 30)\n"
    << "  --ln-globalization M           lazy-newton step acceptance: filter (default) or merit\n"
    << "  --ln-transport-complementarity-floor K  lazyqp-sparse: keep s*z >= mu/K on the returned state, 0 disables (default 1e3)\n"
    << "  --ln-condense-max-support N    lazy-newton working-set rows with at most N free variables are condensed (default 16)\n"
    << "  --ln-prox-alpha-min X          lazy-newton blocked-step threshold for the adaptive proximal shift, 0 disables (default 0)\n"
    << "  --ln-mult-init-max X           lazy-newton least-squares multiplier start cap (default 1e3)\n"
    << "  --ln-scaling-max-gradient X    lazy-newton gradient-based scaling target, 0 disables (default 100)\n"
    << "  --ln-inequality-curvature M    lazy-newton working-set curvature: adaptive (default), majorant, or exact\n"
    << "  --ln-adaptive-curvature-delta X lazy-newton shift above which adaptive reverts to the majorant (default 1)\n"
    << "  --ln-ordering M                lazy-newton fill-reducing ordering: amd (default when compiled) or metis\n"
    << "  --auto-sqp-max-n N             deprecated compatibility option; ignored\n"
    << "  --inner-backend NAME           lazyqp (default), lazyqp-sparse, socp, filter-sqp, tr-sqp, ipopt, or legacy-ipm\n"
    << "                                   socp = exact rotated cones with analytically condensed auxiliaries\n"
    << "  --qp-option NAME               LazyIPDCA-QP profile (default C)\n"
    << "  --qp-amrs / --no-qp-amrs      enable/disable QP AMRS profile flag (default on)\n"
    << "  --qp-init N                    LazyQP initialization mode (default 2)\n"
    << "  --qp-backend NAME              LazyQP linear backend (default ldl; native fallback if unavailable)\n"
    << "  --qp-rho-trial-every N         QP curvature-trial period metadata (default 2)\n"
    << "  --qp-max-atoms N               QP rank-one correction cap metadata (default 500)\n"
    << "  --qp-seed-coverage X           QP seed coverage metadata (default 0.25)\n"
    << "  --qp-seed-stall-tol X          QP seed stall tolerance metadata (default 0)\n"
    << "  --qp-seed-patience N           QP seed patience metadata (default 999)\n"
    << "  --qp-atom-events / --no-qp-atom-events  QP event selector flag (default on)\n"
    << "  --qp-atom-event-every N        QP event period metadata (default 50)\n"
    << "  --qp-bound-push X              LazyQP absolute interior push (default 0)\n"
    << "  --qp-bp-push-kappa X           adaptive bound-push coefficient (default 1e-4)\n"
    << "  --qp-bp-margin-tol X           adaptive bound-push margin (default 1e-4)\n"
    << "  --qp-escapes N                 QP escape budget metadata (default 3)\n"
    << "  --qp-mehrotra / --no-qp-mehrotra  predictor-corrector control (default on)\n"
    << "  --qp-sigma-max X               centering parameter cap (default 1e14)\n"
    << "  --qp-model-solved-tol X        convex-model solved floor (default 1e-12)\n"
    << "  --qp-dca-stop / --no-qp-dca-stop QP DCA-stop profile flag (default on)\n"
    << "  --qp-dca-stall-tol-rel X       QP DCA stall metadata (default 1e-10)\n"
    << "  --qp-dca-stall-iters N         QP DCA stall patience (default 5)\n"
    << "  --qp-dca-monotone-tol-abs X    QP monotonicity metadata (default 1e-6)\n"
    << "  --qp-bp-adaptive / --no-qp-bp-adaptive adaptive bound-push flag (default on)\n"
    << "  --qp-defaults                  print the qp_ profile as JSON and exit\n"
    << "  --linear-backend NAME          compatibility alias for --qp-backend\n"
    << "  --linear-metis 0|1             enable METIS ordering for sparse backends (default 1)\n"
    << "  --kkt-persistent-symbolic-cache keep sparse symbolic analysis across Lazy-DCA majorants (default)\n"
    << "  --equality-qp-original-kkt-gmres / --no-equality-qp-original-kkt-gmres  guarded exact-KKT rescue\n"
    << "  --equality-qp-original-kkt-gmres-max-iterations N  Krylov cap (default 128)\n"
    << "  --no-kkt-persistent-symbolic-cache recreate the sparse factor object for every majorant\n"
    << "  --kkt-primal-reg X             rho_x initial numerical primal KKT shift (default 1e-12)\n"
    << "  --kkt-dual-reg X               fixed delta_d when adaptive KKT regularization is off\n"
    << "  --kkt-adaptive-reg             enable historical rho_x / delta_d controller\n"
    << "  --no-kkt-adaptive-reg          disable adaptive KKT regularization\n"
    << "  --rho-x-floor X                minimum numerical primal shift (default 1e-12)\n"
    << "  --rho-x-growth X               rho_x growth factor (default 10)\n"
    << "  --rho-x-decay X                rho_x clean-step decay factor (default 0.5)\n"
    << "  --rho-x-max X                  maximum numerical primal shift (default 1e-2)\n"
    << "  --rho-x-clean N                clean outer accepts before rho_x decay (default 3)\n"
    << "  --delta-dual-base X            inverse-mode delta_d numerator (default 1e-8)\n"
    << "  --delta-dual-floor X           minimum delta_d (default 1e-12)\n"
    << "  --delta-dual-growth X          independent delta_d growth factor (default 10)\n"
    << "  --delta-dual-decay X           stable-factor warm-start decay (default 0.5)\n"
    << "  --delta-dual-max X             maximum delta_d (default 1e-4)\n"
    << "  --kkt-reg-retries N            Cholesk primal/dual retry budget (default 8)\n"
    << "  --delta-dual-mode MODE         inverse (default) or fixed\n"
    << "  --kkt-berr-tol X               iterative-refinement trigger (default 1e-9)\n"
    << "  --kkt-berr-reject X            reject solve above this berr (default 1e-7)\n"
    << "  --kkt-refine N                 max iterative-refinement steps (default 3)\n"
    << "  --cholesk-adaptive-reg         sign-aware primal/dual pivot repair in fixed KKT mode (default)\n"
    << "  --no-cholesk-adaptive-reg      use exactly --kkt-primal-reg/--kkt-dual-reg\n"
    << "  --[no-]cholesk-reg-warm-start reuse a decayed last stable primal/dual shift after exact failure\n"
    << "  --cholesk-pivot-rel-tol X      minimum local LDL pivot ratio accepted by cholesk (default 1e-14)\n"
    << "  --max-outer N                  outer iteration limit\n"
    << "  --max-inner N                  convex-subproblem kernel iteration limit\n"
    << "  --time-limit SEC               wall-clock limit\n"
    << "  --feas-tol X                   original QCQP feasibility tolerance\n"
    << "  --opt-tol X                    Ipopt-style scaled overall NLP tolerance\n"
    << "  --dual-inf-tol X               unscaled dual-infeasibility guard (default 1)\n"
    << "  --compl-inf-tol X              unscaled complementarity guard (default 1e-4)\n"
    << "  --s-max X                      Ipopt multiplier scaling threshold (default 100)\n"
    << "  --inner-tol X                  convex-subproblem KKT tolerance\n"
    << "  --inner-mu-fallback-tol X      guarded complementarity fallback at max-inner\n"
    << "  --lazyqp-mu0 X                 initial barrier target for LazyQP kernel (default 1e-4)\n"
    << "  --lazyqp-mu-r X                barrier reduction factor (default 0.1)\n"
    << "  --lazyqp-mu-theta X            relative residual gate for barrier reduction (default 0.1)\n"
    << "  --lazyqp-mu-inner-max N        optional max Newton steps per barrier level (0=off)\n"
    << "  --lazyqp-numeric-kkt-reuse / --no-lazyqp-numeric-kkt-reuse  residual-certified numerical-factor reuse (default off)\n"
    << "  --lazyqp-numeric-kkt-reuse-age N  maximum accepted steps per reused factor (default 3)\n"
    << "  --lazyqp-numeric-kkt-reuse-iters N  FGMRES cap per KKT right-hand side (default 4)\n"
    << "  --lazyqp-inexact-forcing / --no-lazyqp-inexact-forcing  guarded outer-residual forcing sequence (default off)\n"
    << "  --lazyqp-forcing-eta X         forcing ratio in (0,1), default 0.1\n"
    << "  --lazyqp-forcing-max-tol X     maximum early-return KKT tolerance (default 1e-2)\n"
    << "  --lazyqp-forcing-fast-iters N  early-return iteration budget before strict fallback (default 4)\n"
    << "  --lazyqp-forcing-min-iters N   minimum Newton steps before early return (default 1)\n"
    << "  --dca-single-loop-primal-dual / --no-dca-single-loop-primal-dual  one guarded predictor-corrector step per majorant (default off)\n"
    << "  --dca-single-loop-multiplier-transport / --no-dca-single-loop-multiplier-transport  transport guarded mapped multipliers after an accepted provisional step (default on under master)\n"
    << "  --dca-single-loop-retraction-fallback-only / --no-dca-single-loop-retraction-fallback-only  deprecated auxiliary retraction switch (default off; rejected under master)\n"
    << "  --dca-single-loop-inner-steps N  must be 1 while single-loop mode is active\n"
    << "  --dca-single-loop-forcing-eta X  required residual contraction in (0,1), default 0.9\n"
    << "  --dca-single-loop-baseline-prox X deterministic convex curvature for single-loop majorants, default 0.05\n"
    << "  --dca-single-loop-direct-start / --no-dca-single-loop-direct-start  guarded Phase-I bypass for equality-feasible side starts (default off)\n"
    << "  --dca-single-loop-direct-start-theta-cap X  maximum finite original violation (default 1e-2)\n"
    << "  --dca-single-loop-direct-start-gamma-h X  h-step contraction fraction in (0,1), default 0.05\n"
    << "  --lazyqp-kkt-form MODE         auto (default), full, reduced, or hybrid inequality KKT\n"
    << "  --lazyqp-reduced-kkt-min-ineq N minimum inequality count for auto reduction (default 32)\n"
    << "  --lazyqp-hybrid-retained-ineq N maximum dense inequality rows retained in hybrid KKT (default 2)\n"
    << "  --lazyqp-hybrid-min-row-nnz N minimum Jacobian row nnz eligible for retention (default 64)\n"
    << "  --lazyqp-flow-lift-elimination exact PGLib apparent-flow lift elimination (default off)\n"
    << "  --no-lazyqp-flow-lift-elimination disable apparent-flow lift elimination\n"
    << "  --lazyqp-model-support-fastpath restrict sparse inequality kernels to validated supports (default off)\n"
    << "  --no-lazyqp-model-support-fastpath use the generic dense inequality kernels\n"
    << "  --lazyqp-low-rank-form MODE    auto (default), condensed, or bordered KKT corrections\n"
    << "  --lazyqp-condense-threshold N  primal dimension limit used only by low-rank auto mode\n"
    << "  --lazyqp-warm-guard X          residual guard factor for multiplier continuation (default 2)\n"
    << "  --lazyqp-zero-dual-continuation use a capped barrier for transported numerical-zero inequality multipliers (default off)\n"
    << "  --no-lazyqp-zero-dual-continuation preserve the r3p23 centered replacement\n"
    << "  --lazyqp-zero-dual-mu-cap X   componentwise s*z cap for transported numerical zeros (default 1e-4)\n"
    << "  --inner-warm-start             reuse previous convex-subproblem equality multipliers (opt-in)\n"
    << "  --inner-full-dual-warm-start   also reuse inequality multipliers (diagnostic; default off)\n"
    << "  --inner-equality-warm-start    equality-only warm start (opt-in)\n"
    << "  --no-inner-warm-start          disable inner dual warm start (default)\n"
    << "  --inner-bound-push X           absolute interior push factor (default 1e-2)\n"
    << "  --inner-bound-frac X           relative interior push factor (default 1e-2)\n"
    << "  --inner-bound-push-ref X       reference slack for active constraints (default 1e-6)\n"
    << "  --inner-bound-mult-init X      fallback positive dual initialization (default 1)\n"
    << "  --restoration-iters N          local Phase-I exact-QCQP iterations\n"
    << "  --shared-restoration-budget 0|1 share restoration-iters across native phases (default 0; automatic routes may enable it)\n"
    << "  --restoration-exact-workspace-reuse / --no-restoration-exact-workspace-reuse\n"
    << "  --restoration-exact-parallel-row-dots / --no-restoration-exact-parallel-row-dots\n"
    << "  --restoration-exact-sparse-row-dots / --no-restoration-exact-sparse-row-dots\n"
    << "  --restoration-exact-parallel-min-rows N  Phase-I OpenMP threshold (default 128)\n"
    << "  --restoration-restarts N       deterministic Phase-I restart budget\n"
    << "  --restoration-restart-iters N  Phase-I iterations per restart\n"
    << "  --restoration-tangent-activation activate flat one-anchor quadratic components (default)\n"
    << "  --no-restoration-tangent-activation disable objective-driven activation\n"
    << "  --restoration-minimax-iters N  terminal L-infinity QP passes (default 80)\n"
    << "  --restoration-minimax-prox X   terminal minimax proximal weight (default 1e-6)\n"
    << "  --prox X                       safeguarded-SQP comparison proximal regularization\n"
    << "  --dca-extra-prox X             optional extra convex term in F_{0,k}; default 0\n"
    << "  --dca-affine-full-limit N      full-majorant size limit for affine objectives\n"
    << "  --no-dca-affine-full           disable full-majorant affine-objective safeguard\n"
    << "  --dca-affine-box-bounded-lazy  lazy WS for box-bounded affine cost in condensed/linearized modes (default)\n"
    << "  --no-dca-affine-box-bounded-lazy force historical full WS for A/B tests\n"
    << "  --dca-corrector                enable multiplier-aware local corrector (default)\n"
    << "  --no-dca-corrector             disable the local corrector for pure-majorant A/B tests\n"
    << "  --dca-corrector-start N        first outer iteration eligible for correction (default 2)\n"
    << "  --dca-corrector-frequency N    correction frequency in outer iterations (default 1)\n"
    << "  --dca-iterated-soc             refine failed one-shot equality SOCs (default)\n"
    << "  --no-dca-iterated-soc          disable iterated SOC for A/B tests\n"
    << "  --dca-iterated-soc-iters N     normal-correction iterations (default 12)\n"
    << "  --dca-iterated-soc-max-ratio X max total correction/step ratio (default 2)\n"
    << "  --dca-iterated-soc-trigger X   trial feasibility trigger ratio (default 1e6)\n"
    << "  --dca-warm-failure-limit N     failures before continuation cooldown (default 3)\n"
    << "  --dca-warm-cooldown N          cold outer iterations after suppression (default 32)\n"
    << "  --dca-inner-retries N          cold/proximal retries after an inner failure (default 3)\n"
    << "  --dca-inner-retry-prox X       first retry model proximal term (default 1e-8)\n"
    << "  --dca-step-control             scale-aware proximal control of oversized DCA directions (default)\n"
    << "  --no-dca-step-control          disable scale-aware DCA direction control\n"
    << "  --dca-step-target-abs X        absolute part of the DCA direction envelope (default 1)\n"
    << "  --dca-step-target-rel X        state-relative part of the envelope (default 1e-2)\n"
    << "  --dca-step-target-max X        maximum DCA direction envelope (default 10)\n"
    << "  --dca-step-control-retries N   maximum proximal re-solves per majorant (default 8)\n"
    << "  --dca-equality-funnel X        explicit initial funnel for nonlinear equality residuals\n"
    << "  --dca-equality-funnel-auto     auto-initialize funnel at sqrt(feas_tol) (default)\n"
    << "  --no-dca-equality-funnel-auto  disable automatic funnel initialization\n"
    << "  --dca-equality-funnel-ceiling X cap for automatic funnel (default 1e-2)\n"
    << "  --dca-equality-retraction      guarded SQP equality retraction before SOC (default)\n"
    << "  --no-dca-equality-retraction   disable equality retraction for A/B tests\n"
    << "  --dca-equality-retraction-max-ratio X correction/step guard (default 2)\n"
    << "  --dca-equality-model MODE      linearized|condensed-retraction|tangent-inertia|elastic-split\n"
    << "  --dca-tangent-inertia-trigger X  original projected stationarity gate (default 1)\n"
    << "  --lazyqp-affine-support-fastpath / --no-lazyqp-affine-support-fastpath  sparse affine audits\n"
    << "  --dca-condensed-newton-iters N condensed Newton iterations (default 8)\n"
    << "  --dca-condensed-krylov-iters N GMRES iterations in equality space (default 16)\n"
    << "  --dca-condensed-krylov-rtol X  relative GMRES tolerance (default 1e-4)\n"
    << "  --dca-condensed-continuation N branch-continuation attempts (default 16)\n"
    << "  --dca-condensed-tol X          exact equality target tolerance (default 1e-10)\n"
    << "  --dca-condensed-regularization X first rank-deficiency safeguard (default 1e-12)\n"
    << "  --dca-condensed-reg-growth X   regularization growth factor (default 10)\n"
    << "  --dca-condensed-reg-max X      maximum normal regularization (default 1e-4)\n"
    << "  --dca-condensed-max-ratio X    normal-correction/step guard (default 2)\n"
    << "  --dca-condensed-bound-aware-normal keep normal corrections on active bound faces (default)\n"
    << "  --no-dca-condensed-bound-aware-normal disable active-face normal map for A/B tests\n"
    << "  --dca-condensed-bound-activity-tol X relative active-face tolerance (default 1e-10)\n"
    << "  --dca-condensed-curvature      condensed curvature correction (ambient on large models, projected fallback; default)\n"
    << "  --no-dca-condensed-curvature   disable condensed curvature correction\n"
    << "  --dca-condensed-projected-multi-ritz-inertia batched projected Ritz proposals with exact inertia gate\n"
    << "  --no-dca-condensed-projected-multi-ritz-inertia preserve historical ambient/projected route (default)\n"
    << "  --dca-condensed-block-exact-curvature exact sparse-block ambient certificate and one normal map\n"
    << "  --no-dca-condensed-block-exact-curvature preserve the r3p19 two-map route (default)\n"
    << "  --dca-condensed-lanczos-steps N projected Lanczos dimension (default 24)\n"
    << "  --dca-condensed-lanczos-starts N projected Lanczos starts (default 2)\n"
    << "  --dca-condensed-max-rank N     tangent rank-one correction budget (default 64)\n"
    << "  --dca-condensed-scalar-safeguard-updates N verified scalar updates after rank cap (default 8)\n"
    << "  --dca-condensed-affine-normal-reuse reuse Y dz in Newton line search (default)\n"
    << "  --no-dca-condensed-affine-normal-reuse disable algebraic Y reuse\n"
    << "  --dca-condensed-quadratic-linesearch-reuse exact quadratic residual screening (default)\n"
    << "  --no-dca-condensed-quadratic-linesearch-reuse disable quadratic screening\n"
    << "  --dca-condensed-sparse-equalities sparse exact equality evaluation (default)\n"
    << "  --no-dca-condensed-sparse-equalities use historical dense equality evaluation\n"
    << "  --dca-condensed-cached-regularization reuse stable positive normal regularization (default)\n"
    << "  --no-dca-condensed-cached-regularization always restart normal regularization at zero\n"
    << "  --dca-condensed-parallel-equalities parallelize independent equality rows (runtime off by default)\n"
    << "  --no-dca-condensed-parallel-equalities force serial equality-row kernels\n"
    << "  --dca-condensed-parallel-min-equalities N minimum equality count for OpenMP (default 128)\n"
    << "  --dca-condensed-adaptive-normal-refinement relaxed intermediate retraction solves with strict final map (default off)\n"
    << "  --no-dca-condensed-adaptive-normal-refinement preserve strict refinement for every normal solve\n"
    << "  --dca-condensed-normal-intermediate-berr-tol X intermediate backward-error target (default 1e-6)\n"
    << "  --dca-condensed-normal-intermediate-berr-reject X intermediate rejection threshold (default 1e-4)\n"
    << "  --dca-condensed-normal-strict-stationarity-factor X strict gate times requested dual tolerance (default 1e3)\n"
    << "  --dca-condensed-alpha-predictor seed the next outer line search from the previous accepted alpha (default off)\n"
    << "  --no-dca-condensed-alpha-predictor always start the outer line search at alpha=1\n"
    << "  --dca-condensed-alpha-growth X growth from the preceding accepted alpha (default 2)\n"
    << "  --dca-condensed-alpha-full-step-frequency N force an alpha=1 probe every N outer iterations (default 8)\n"
    << "  --dca-condensed-outer-extrapolation probe one alpha>1 after stable full steps (default off)\n"
    << "  --no-dca-condensed-outer-extrapolation retain the historical alpha<=1 schedule\n"
    << "  --dca-condensed-outer-extrapolation-alpha X guarded probe alpha in (1,2] (default 1.5)\n"
    << "  --dca-condensed-outer-extrapolation-min-full-steps N required consecutive alpha=1 steps (default 3)\n"
    << "  --dca-condensed-post-retraction-ws-expansion add omitted side offenders exposed by retraction (default off)\n"
    << "  --no-dca-condensed-post-retraction-ws-expansion retain historical alpha backtracking\n"
    << "  --dca-condensed-modeled-side-filter allow bounded modeled-only Maratos side violations (default off)\n"
    << "  --no-dca-condensed-modeled-side-filter retain the strict final side tube at every iterate\n"
    << "  --dca-condensed-modeled-side-filter-limit X transient tube, capped by --activation (default 1e-3)\n"
    << "  --dca-condensed-modeled-side-recovery-margin X inward majorant margin factor (default 1)\n"
    << "  --dca-condensed-flow-lift-schur exact Schur elimination of recognized PGLib flow lifts (default off)\n"
    << "  --no-dca-condensed-flow-lift-schur retain the full bordered normal operator\n"
    << "  --dca-condensed-post-retraction-ws-max N cap same-outer re-solves before historical fallback (default 8)\n"
    << "  --dca-equality-elastic-penalty X linear penalty rho*eta (default 10)\n"
    << "  --dca-equality-elastic-quadratic X quadratic eta penalty (default 1e-8)\n"
    << "  --dca-equality-elastic-cap-factor X elastic tube factor (default 1)\n"
    << "  --dca-equality-elastic-post-retraction enable guarded post-step equality retraction\n"
    << "  --no-dca-equality-elastic-post-retraction disable post-step retraction (default)\n"
    << "  --activation X                 lazy inequality activation threshold\n"
    << "  --max-rank N                   compatibility: set both rank caps\n"
    << "  --max-constraint-rank N        max retained directions per inequality Qi\n"
    << "  --max-objective-rank N         max retained directions for Q0 (Lazy-DCA)\n"
    << "  --max-lagrangian-rank N        legacy-SQP Lagrangian rank cap\n"
    << "  --constraint-model NAME        phase-II inequalities: auto (default) or majorant\n"
    << "  --line-search-armijo X         Armijo sufficient-decrease constant (default 1e-4)\n"
    << "  --no-tangential               disable tangential globalization fallback\n"
    << "  --no-soc                       disable one-shot second-order correction\n"
    << "  --soc-max-ratio X              max ||SOC||inf / ||alpha*d||inf (default 0.5)\n"
    << "  --soc-trigger-ratio X          attempt SOC when theta <= X*feas_tol (default 1000)\n"
    << "  --working-set-drop-slack X     hysteresis slack for dropping constraints\n"
    << "  --working-set-drop-mu X        multiplier threshold for dropping constraints\n"
    << "  --working-set-drop-patience N  consecutive accepted points before drop (default 3)\n"
    << "  --no-working-set-drop          disable working-set dropping\n"
    << "  --terminal-kkt-trigger X       stationarity threshold for guarded reduced Newton\n"
    << "  --terminal-kkt-iters N         guarded local KKT step limit (large systems require explicit opt-in)\n"
    << "  --terminal-kkt-mu-abs X        absolute active-dual cutoff (default 1e-9)\n"
    << "  --terminal-kkt-mu-rel X        relative active-dual cutoff (default 1e-8)\n"
    << "  --terminal-kkt-stability N     stable active signatures before Newton (default 3)\n"
    << "  --terminal-kkt                 enable guarded KKT also for n>64 (small systems use it by default)\n"
    << "  --no-terminal-kkt              explicitly disable guarded KKT acceleration\n"
    << "  --tr-boundary-rel-tol X       numerical relative tolerance on ||d||<=Delta (default 1e-4)\n"
    << "  --tr-boundary-abs-tol X       numerical absolute tolerance on ||d||<=Delta (default 1e-9)\n"
    << "  --lanczos-steps N              Lanczos dimension\n"
    << "  --lanczos-starts N             independent Lanczos starts\n"
    << "  --exact-eigen-threshold N      exact LAPACK eigensolve below N\n"
    << "  --strict-scalar-certificate    Gershgorin-certified scalar convexification\n"
    << "  --no-restoration               disable Phase I (diagnostic only)\n"
    << "  --live-iterations              flushed Ipopt-style Lazy-DCA outer table\n"
    << "  --live-iterations-every N      print one live row every N accepted outer iterations\n"
    << "  --live-header-every N          repeat the live-table header every N rows\n"
    << "  --no-live-iterations           disable the live table (default)\n"
    << "  --quiet                        suppress iteration log\n"
    << "  --linear-backends              list compiled linear backends\n"
    << "  --version                      print version\n";
}

template<class T> T parse_num(const char* s);
template<> int parse_num<int>(const char* s) { return std::stoi(s); }
template<> double parse_num<double>(const char* s) { return std::stod(s); }
}

int main(int argc, char** argv) {
  std::string problem_path, json_path, frozen_start_path;
  const auto process_start=std::chrono::steady_clock::now();
  bool compact_qcq_input=true;
  bool accepted_checkpoint=false;
  bool factorial_structure_preflight=false;
  Settings st;
  st.outer_algorithm = "lazy-dca";
  // The command-line profile defaults to production QP option C + AMRS.  A
  // portable build without SuiteSparse LDL falls back to native LAPACK unless
  // the user explicitly requests an unavailable backend.
  st.linear_backend = linear_backend_compiled("mumps") ? "mumps" :
      (linear_backend_compiled(st.qp_backend) ? st.qp_backend : "native");
  try {
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      // Accept the exact qp_* spelling used in experiment settings as well as
      // the conventional CLI qp-* spelling.
      if (a.rfind("--qp_", 0) == 0)
        std::replace(a.begin()+2, a.end(), '_', '-');
      auto need = [&](const char* flag) -> const char* {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + flag);
        return argv[++i];
      };
      if (a == "--problem") problem_path = need("--problem");
      else if (a == "--algorithm") st.outer_algorithm = need("--algorithm");
      else if(a=="--dca-rho-refinements") st.dca_filter_rho_refinements=parse_num<int>(need("--dca-rho-refinements"));
      else if(a=="--dca-initial-restoration") st.dca_filter_initial_restoration=true;
      else if(a=="--dca-normal-weighted-atoms") st.dca_filter_normal_atom_weighting=true;
      else if(a=="--dca-active-evaluation") st.dca_filter_active_evaluation=true;
      else if(a=="--no-dca-active-evaluation") st.dca_filter_active_evaluation=false;
      else if(a=="--dca-packed-atoms") st.dca_filter_packed_atoms=true;
      else if(a=="--no-dca-packed-atoms") st.dca_filter_packed_atoms=false;
      else if(a=="--dca-lazy-merit") st.dca_filter_lazy_merit=true;
      else if(a=="--no-dca-lazy-merit") st.dca_filter_lazy_merit=false;
      else if(a=="--dca-block-ritz") st.dca_filter_block_ritz=true;
      else if(a=="--no-dca-block-ritz") st.dca_filter_block_ritz=false;
      else if(a=="--dca-anderson") st.dca_filter_anderson=true;
      else if(a=="--no-dca-anderson") st.dca_filter_anderson=false;
      else if(a=="--dca-rank-audit-mode") {
        st.dca_rank_audit_mode=need(a.c_str());
        if(st.dca_rank_audit_mode!="full"&&st.dca_rank_audit_mode!="mapped"&&st.dca_rank_audit_mode!="check")
          throw std::invalid_argument(a+" must be full, mapped or check");
      }
      else if(a=="--dca-rank-presolve") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::invalid_argument(a+" must be 0 or 1");
        st.dca_rank_presolve=v=="1";
      }
      else if(a=="--native-augmented-repair") {
        const std::string v=need("--native-augmented-repair");
        if(v!="0"&&v!="1")throw std::runtime_error("native-augmented-repair expects 0|1");
        st.native_augmented_repair=v=="1";
      }
      else if(a=="--power2-scaling") {
        const std::string v=need("--power2-scaling");
        if(v!="0"&&v!="1")throw std::runtime_error("power2-scaling expects 0|1");
        st.power2_scaling=v=="1";
      }
      else if(a=="--product-definition-presolve") {
        const std::string v=need("--product-definition-presolve");
        if(v!="0"&&v!="1")throw std::runtime_error("product-definition-presolve expects 0|1");
        st.product_definition_presolve=v=="1";
      }
      else if(a=="--free-epigraph-presolve") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::invalid_argument(a+" must be 0 or 1");
        st.free_epigraph_presolve=v=="1";
      }
      else if(a=="--dca-rank-implied-bounds") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::invalid_argument(a+" must be 0 or 1");
        st.dca_rank_implied_bounds=v=="1";
      }
      else if(a=="--dca-rank-presolve-report")st.dca_rank_presolve_report_file=need(a.c_str());
      else if(a=="--dca-aggregate-backend") st.dca_aggregate_backend=need(a.c_str());
      else if(a=="--dca-dual-step") st.dca_dual_step=need(a.c_str());
      else if(a=="--dca-common-step-primal-trigger") st.dca_common_step_primal_trigger=parse_num<double>(need(a.c_str()));
      else if(a=="--dca-common-step-dual-growth") st.dca_common_step_dual_growth=parse_num<double>(need(a.c_str()));
      else if(a=="--dca-aggregate-only" || a=="--dca-aggregate-affine-retry") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::invalid_argument(a+" must be 0 or 1");
        if(a=="--dca-aggregate-only")st.dca_aggregate_only=v=="1";
        else st.dca_aggregate_affine_retry=v=="1";
      }
      else if(a=="--dca-aggregate-retry-max")st.dca_aggregate_retry_max=parse_num<int>(need(a.c_str()));
      else if(a=="--dca-aggregate-rho-initial")st.dca_aggregate_rho_initial=parse_num<double>(need(a.c_str()));
      else if(a=="--dca-aggregate-rho-growth")st.dca_aggregate_rho_growth=parse_num<double>(need(a.c_str()));
      else if(a=="--dca-aggregate-rho-decay")st.dca_aggregate_rho_decay=parse_num<double>(need(a.c_str()));
      else if(a=="--dca-aggregate-kkt-layout") {
        st.dca_aggregate_kkt_layout=need(a.c_str());
        if(st.dca_aggregate_kkt_layout!="persistent"&&st.dca_aggregate_kkt_layout!="dynamic")
          throw std::invalid_argument(a+" must be persistent or dynamic");
      }
      else if(a=="--dca-aggregate-border-reserve") {
        st.dca_aggregate_border_reserve=parse_num<double>(need(a.c_str()));
        if(!std::isfinite(st.dca_aggregate_border_reserve)||st.dca_aggregate_border_reserve<1||st.dca_aggregate_border_reserve>4)
          throw std::invalid_argument(a+" must be finite and in [1,4]");
      }
      else if(a=="--dca-aggregate-rejection-policy") {
        st.dca_aggregate_rejection_policy=need(a.c_str());
        if(st.dca_aggregate_rejection_policy!="separated"&&st.dca_aggregate_rejection_policy!="legacy")
          throw std::invalid_argument(a+" must be separated or legacy");
      }
      else if(a=="--dca-aggregate-fast-assembly"||a=="--dca-aggregate-rho-memory"||a=="--dca-aggregate-lean") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::invalid_argument(a+" must be 0 or 1");
        if(a=="--dca-aggregate-fast-assembly")st.dca_aggregate_fast_assembly=v=="1";
        else if(a=="--dca-aggregate-rho-memory")st.dca_aggregate_rho_memory=v=="1";
        else st.dca_aggregate_lean=v=="1";
      }
      else if(a=="--dca-aggregate-rho-max") st.dca_aggregate_rho_max=std::stod(need(a.c_str()));
      else if(a=="--dca-aggregate-primal-trigger") st.dca_aggregate_primal_trigger=std::stod(need(a.c_str()));
      else if(a=="--dca-aggregate-progress-window") st.dca_aggregate_progress_window=std::stoi(need(a.c_str()));
      else if(a=="--dca-kkt-growth-max") st.dca_kkt_growth_max=std::stod(need(a.c_str()));
      else if(a=="--dca-aggregate-preparation-cache" || a=="--dca-aggregate-strict-probe-stop") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::invalid_argument(a+" must be 0 or 1");
        if(a=="--dca-aggregate-preparation-cache")st.dca_aggregate_preparation_cache=v=="1";
        else st.dca_aggregate_strict_probe_stop=v=="1";
      }
      else if(a=="--dca-rho-complete-search" || a=="--dca-strict-finish") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::invalid_argument(a+" must be 0 or 1");
        if(a=="--dca-rho-complete-search")st.dca_rho_complete_search=v=="1";
        else st.dca_strict_finish=v=="1";
      }
      else if(a=="--dca-strict-finish-max")st.dca_strict_finish_max=parse_num<int>(need(a.c_str()));
      else if(a=="--dca-strict-finish-refinements")st.dca_strict_finish_refinements=parse_num<int>(need(a.c_str()));
      else if(a=="--dca-strict-finish-mu-fraction")st.dca_strict_finish_mu_fraction=parse_num<double>(need(a.c_str()));
      else if(a=="--dca-strict-finish-eta")st.dca_strict_finish_eta=parse_num<double>(need(a.c_str()));
      else if(a=="--dca-aggregate") st.dca_aggregate=true;
      else if(a=="--no-dca-aggregate") st.dca_aggregate=false;
      else if(a=="--dca-krylov") st.dca_krylov=true;
      else if(a=="--no-dca-krylov") st.dca_krylov=false;
      else if(a=="--dca-krylov-max") st.dca_krylov_max_iterations=std::stoi(need(a.c_str()));
      else if(a=="--dca-krylov-cooldown") st.dca_krylov_failure_cooldown=std::stoi(need(a.c_str()));
      else if(a=="--no-dca-krylov-pc") st.dca_krylov_predictor_corrector=false;
      else if(a=="--dca-local-active") st.dca_local_active=true;
      else if(a=="--no-dca-local-active") st.dca_local_active=false;
      else if(a=="--dca-local-weak-affine") st.dca_local_weak_affine=true;
      else if(a=="--dca-pair-elimination") st.dca_pair_elimination=true;
      else if(a=="--dca-local-trigger") { st.dca_local_trigger=std::stod(need("--dca-local-trigger")); if(!std::isfinite(st.dca_local_trigger)||st.dca_local_trigger<=0)throw std::runtime_error("positive finite local trigger required"); }
      else if(a=="--dca-local-primal-trigger") { st.dca_local_primal_trigger=std::stod(need("--dca-local-primal-trigger")); if(!std::isfinite(st.dca_local_primal_trigger)||st.dca_local_primal_trigger<=0)throw std::runtime_error("positive finite local primal trigger required"); }
      else if(a=="--dca-local-backend") { st.dca_local_linear_backend=need("--dca-local-backend"); if(st.dca_local_linear_backend!="same"&&st.dca_local_linear_backend!="cholesk"&&st.dca_local_linear_backend!="ldl"&&st.dca_local_linear_backend!="qdldl"&&st.dca_local_linear_backend!="mumps"&&st.dca_local_linear_backend!="native"&&st.dca_local_linear_backend!="ma27")throw std::runtime_error("local backend must be same, cholesk, ldl, qdldl, mumps, ma27 or native"); }
      else if(a=="--dca-tangent-curvature") st.dca_filter_tangent_curvature=true;
      else if(a=="--no-dca-tangent-curvature") st.dca_filter_tangent_curvature=false;
      else if(a=="--dca-certificate-transport") st.dca_filter_certificate_transport=true;
      else if(a=="--no-dca-certificate-transport") st.dca_filter_certificate_transport=false;
      else if(a=="--dca-sparse-fallback") st.dca_filter_sparse_fallback=true;
      else if(a=="--no-dca-sparse-fallback") st.dca_filter_sparse_fallback=false;
      else if(a=="--ldl-equilibration") st.kkt_equilibration=true;
      else if(a=="--no-ldl-equilibration") st.kkt_equilibration=false;
      else if(a=="--dca-curvature-backend") st.dca_filter_curvature_backend=need("--dca-curvature-backend");
      else if(a=="--no-dca-extrapolation") st.dca_filter_extrapolation=false;
      else if(a=="--no-dca-model-reuse") st.dca_filter_model_reuse=false;
      else if(a=="--majorant-curvature-corrections") {
        st.majorant_curvature_corrections=std::stoi(need(a.c_str()));
        if(st.majorant_curvature_corrections<0||st.majorant_curvature_corrections>16)throw std::runtime_error("curvature corrections must be 0..16");
      }
      else if(a=="--majorant-dual-diagnostics") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::runtime_error("diagnostics expects 0|1");
        st.majorant_dual_diagnostics=v=="1";
      }
      else if(a=="--majorant-lambda-trigger") {
        st.majorant_lambda_trigger=need("--majorant-lambda-trigger");
        if(st.majorant_lambda_trigger!="consecutive"&&st.majorant_lambda_trigger!="window")throw std::runtime_error("invalid lambda trigger");
      }
      else if(a=="--majorant-lambda-interpolation"||a=="--majorant-lambda-rho-retry") {
        const std::string v=need(a.c_str());
        if(v!="0"&&v!="1")throw std::runtime_error("lambda option expects 0|1");
        if(a=="--majorant-lambda-interpolation")st.majorant_lambda_interpolation=v=="1";
        else st.majorant_lambda_rho_retry=v=="1";
      }
      else if(a=="--majorant-recalc-lambda") {
        const std::string v=need("--majorant-recalc-lambda");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-recalc-lambda must be 0 or 1");
        st.majorant_recalc_lambda=v=="1";
      }
      else if(a=="--majorant-recalc-lambda-max-iter") {
        st.majorant_recalc_lambda_max_iterations=std::stoi(need("--majorant-recalc-lambda-max-iter"));
        if(st.majorant_recalc_lambda_max_iterations<1)throw std::runtime_error("majorant-recalc-lambda-max-iter must be positive");
      }
      else if(a=="--majorant-startup-policy") {
        const std::string v=need("--majorant-startup-policy");
        if(v!="legacy"&&v!="auto")throw std::runtime_error("majorant-startup-policy expects legacy|auto");
        st.majorant_startup_policy=v;
      }
      else if(a=="--majorant-filter-recenter") {
        const std::string v=need("--majorant-filter-recenter");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-filter-recenter expects 0|1");
        st.majorant_filter_recenter=v=="1";
      }
      else if(a=="--majorant-restoration" || a=="--majorant-initial-restoration") {
        const std::string v=need("--majorant-initial-restoration");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-initial-restoration expects 0|1");
        st.majorant_initial_restoration=v=="1";
      }
      else if(a=="--majorant-filter-history-fix") {
        const std::string v=need("--majorant-filter-history-fix");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-filter-history-fix expects 0|1");
        st.majorant_filter_history_fix=v=="1";
      }
      else if(a=="--majorant-equality-stabilization") {
        const std::string v=need("--majorant-equality-stabilization");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-equality-stabilization expects 0|1");
        st.majorant_equality_stabilization=v=="1";
        st.majorant_equality_stabilization_explicit=true;
      }
      else if(a=="--original-gr-kkt-soc") {
        const std::string v=need("--original-gr-kkt-soc");
        if(v!="0"&&v!="1")throw std::runtime_error("original-gr-kkt-soc expects 0|1");
        st.original_gr_kkt_soc=v=="1";
      }
      else if(a=="--majorant-initial-dual-balance") {
        const std::string v=need("--majorant-initial-dual-balance");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-initial-dual-balance expects 0|1");
        st.majorant_initial_dual_balance=v=="1";
      }
      else if(a=="--majorant-rho-feasibility-guard") {
        const std::string v=need("--majorant-rho-feasibility-guard");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-rho-feasibility-guard expects 0|1");
        st.majorant_rho_feasibility_guard=v=="1";
      }
      else if(a=="--majorant-trial-soc-max-corrections") {
        st.majorant_trial_soc_max_corrections=std::stoi(need("--majorant-trial-soc-max-corrections"));
        if(st.majorant_trial_soc_max_corrections<1||st.majorant_trial_soc_max_corrections>3)
          throw std::runtime_error("majorant-trial-soc-max-corrections expects an integer in [1,3]");
      }
      else if(a=="--majorant-trial-soc-full-primal") {
        const std::string v=need("--majorant-trial-soc-full-primal");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-trial-soc-full-primal expects 0|1");
        st.majorant_trial_soc_full_primal=v=="1";
      }
      else if(a=="--majorant-trial-soc-arc") {
        const std::string v=need("--majorant-trial-soc-arc");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-trial-soc-arc expects 0|1");
        st.majorant_trial_soc_arc=v=="1";
      }
      else if(a=="--majorant-restoration-bound-stall-fix") {
        const std::string v=need("--majorant-restoration-bound-stall-fix");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-restoration-bound-stall-fix expects 0|1");
        st.majorant_restoration_bound_stall_fix=v=="1";
      }
      else if(a=="--majorant-recovery-gap-fix") {
        const std::string v=need("--majorant-recovery-gap-fix");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-recovery-gap-fix expects 0|1");
        st.majorant_recovery_gap_fix=v=="1";
      }
      else if(a=="--majorant-soc-defect-control") {
        const std::string v=need("--majorant-soc-defect-control");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-soc-defect-control expects 0|1");
        st.majorant_soc_defect_control=v=="1";
      }
      else if(a=="--majorant-nonlinear-trial-soc") {
        const std::string v=need("--majorant-nonlinear-trial-soc");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-nonlinear-trial-soc expects 0|1");
        st.majorant_nonlinear_trial_soc=v=="1";
      }
      else if(a=="--majorant-adaptive-globalization") {
        const std::string v=need("--majorant-adaptive-globalization");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-adaptive-globalization expects 0|1");
        st.majorant_adaptive_globalization=v=="1";
      }
      else if(a=="--majorant-soc-small-step-patience") {
        st.majorant_soc_small_step_patience=std::stoi(need("--majorant-soc-small-step-patience"));
        if(st.majorant_soc_small_step_patience<0||st.majorant_soc_small_step_patience>1000000)
          throw std::runtime_error("majorant-soc-small-step-patience expects an integer in [0,1000000]");
      }
      else if(a=="--majorant-soc-small-step-threshold") {
        st.majorant_soc_small_step_threshold=std::stod(need("--majorant-soc-small-step-threshold"));
        if(!std::isfinite(st.majorant_soc_small_step_threshold)||
            !(st.majorant_soc_small_step_threshold>0&&st.majorant_soc_small_step_threshold<=1))
          throw std::runtime_error("majorant-soc-small-step-threshold expects a finite value in (0,1]");
      }
      else if(a=="--majorant-conditional-retraction") {
        const std::string v=need("--majorant-conditional-retraction");
        if(v!="0"&&v!="1")throw std::runtime_error("majorant-conditional-retraction expects 0|1");
        st.majorant_conditional_retraction=v=="1";
      }
      else if(a=="--majorant-restoration-iterations") {
        st.majorant_restoration_iterations=std::stoi(need("--majorant-restoration-iterations"));
        if(st.majorant_restoration_iterations<1)throw std::runtime_error("majorant-restoration-iterations must be positive");
      }
      else if(a=="--factorial-structure-check"||a=="--factorial-structure-preflight") factorial_structure_preflight=true;
      else if(a=="--factorial-inequalities") st.factorial_inequalities=need("--factorial-inequalities");
      else if(a=="--factorial-globalization") st.factorial_globalization=need("--factorial-globalization");
      else if(a=="--factorial-trace") st.factorial_trace=need("--factorial-trace");
      else if(a=="--factorial-validation-legacy-objective") st.factorial_validation_legacy_objective=true;
      else if(a=="--convexification-study") st.convexification_study=need("--convexification-study");
      else if(a=="--convexification-schedule") st.convexification_schedule=need("--convexification-schedule");
      else if(a=="--study-hybrid-total-atoms") st.study_hybrid_total_atoms=std::stoi(need("--study-hybrid-total-atoms"));
      else if(a=="--study-hybrid-max-atoms") st.study_hybrid_max_atoms=std::stoi(need("--study-hybrid-max-atoms"));
      else if(a=="--study-hybrid-event-interval") st.study_hybrid_event_interval=std::stoi(need("--study-hybrid-event-interval"));
      else if(a=="--study-hybrid-downweight-patience") st.study_hybrid_downweight_patience=std::stoi(need("--study-hybrid-downweight-patience"));
      else if(a=="--study-hybrid-atom-support") st.study_hybrid_atom_support=std::stoi(need("--study-hybrid-atom-support"));
      else if(a=="--study-hybrid-downweight-factor") st.study_hybrid_downweight_factor=std::stod(need("--study-hybrid-downweight-factor"));
      else if(a=="--study-hybrid-dominance") st.study_hybrid_dominance=std::stod(need("--study-hybrid-dominance"));
      else if(a=="--study-hybrid-scalar-decay") st.study_hybrid_scalar_decay=std::stod(need("--study-hybrid-scalar-decay"));
      else if(a=="--study-hybrid-margin") st.study_hybrid_margin=std::stod(need("--study-hybrid-margin"));
      else if(a=="--study-inner-relative") st.study_inner_relative=std::stod(need("--study-inner-relative"));
      else if(a=="--study-inner-absolute") st.study_inner_absolute=std::stod(need("--study-inner-absolute"));
      else if(a=="--study-inner-max") st.study_inner_max=std::stoi(need("--study-inner-max"));
      else if(a=="--study-inner-min") st.study_inner_min=std::stoi(need("--study-inner-min"));
      else if(a=="--study-hybrid-candidates") st.study_hybrid_candidates=std::stoi(need("--study-hybrid-candidates"));
      else if(a=="--study-hybrid-lanczos-steps") st.study_hybrid_lanczos_steps=std::stoi(need("--study-hybrid-lanczos-steps"));
      else if(a=="--study-hybrid-lanczos-starts") st.study_hybrid_lanczos_starts=std::stoi(need("--study-hybrid-lanczos-starts"));
      else if(a=="--study-hybrid-additions-per-event") st.study_hybrid_additions_per_event=std::stoi(need("--study-hybrid-additions-per-event"));
      else if(a=="--study-hybrid-support-energy") st.study_hybrid_support_energy=std::stod(need("--study-hybrid-support-energy"));
      else if(a=="--study-hybrid-strength") st.study_hybrid_strength=std::stod(need("--study-hybrid-strength"));
      else if(a=="--study-hybrid-beta-offset") st.study_hybrid_beta_offset=std::stod(need("--study-hybrid-beta-offset"));
      else if(a=="--study-hybrid-negative-tolerance") st.study_hybrid_negative_tolerance=std::stod(need("--study-hybrid-negative-tolerance"));
      else if(a=="--study-hybrid-scalar-safety") st.study_hybrid_scalar_safety=std::stod(need("--study-hybrid-scalar-safety"));
      else if(a=="--study-hybrid-scalar-floor") st.study_hybrid_scalar_floor=std::stod(need("--study-hybrid-scalar-floor"));
      else if(a=="--study-hybrid-score-activity") st.study_hybrid_score_activity=std::stod(need("--study-hybrid-score-activity"));
      else if(a=="--study-hybrid-score-scalar") st.study_hybrid_score_scalar=std::stod(need("--study-hybrid-score-scalar"));
      else if(a=="--study-hybrid-score-support") st.study_hybrid_score_support=std::stod(need("--study-hybrid-score-support"));
      else if(a=="--study-hybrid-score-density") st.study_hybrid_score_density=std::stod(need("--study-hybrid-score-density"));
      else if(a=="--study-hybrid-score-weight") st.study_hybrid_score_weight=std::stod(need("--study-hybrid-score-weight"));
      else if(a=="--study-hybrid-min-lift") st.study_hybrid_min_lift=std::stod(need("--study-hybrid-min-lift"));
      else if(a=="--study-hybrid-min-activity") st.study_hybrid_min_activity=std::stod(need("--study-hybrid-min-activity"));
      else if(a=="--study-hybrid-min-score") st.study_hybrid_min_score=std::stod(need("--study-hybrid-min-score"));
      else if(a=="--study-hybrid-objective-floor") st.study_hybrid_objective_floor=std::stod(need("--study-hybrid-objective-floor"));
      else if(a=="--study-inner-kappa-fraction") st.study_inner_kappa_fraction=std::stod(need("--study-inner-kappa-fraction"));
      else if(a=="--study-inner-margin") st.study_inner_margin=std::stod(need("--study-inner-margin"));
      else if(a=="--study-inner-activity") st.study_inner_activity=std::stod(need("--study-inner-activity"));
      else if(a=="--study-inner-feasibility") st.study_inner_feasibility=std::stod(need("--study-inner-feasibility"));
      else if(a=="--study-inner-complementarity") st.study_inner_complementarity=std::stod(need("--study-inner-complementarity"));
      else if(a=="--study-inner-barrier-decay") st.study_inner_barrier_decay=std::stod(need("--study-inner-barrier-decay"));
      else if(a=="--study-inner-merit-recovery") st.study_inner_merit_recovery=std::stoi(need("--study-inner-merit-recovery"))!=0;
      else if(a=="--study-inner-separate-steps") st.study_inner_separate_steps=std::stoi(need("--study-inner-separate-steps"))!=0;
      else if(a=="--study-controller") st.study_controller=need("--study-controller");
      else if(a=="--study-inner-rule") st.study_inner_rule=need("--study-inner-rule");
      else if(a=="--study-inner-predictor-corrector") { const std::string v=need(a.c_str()); if(v!="0"&&v!="1")throw std::runtime_error("study-inner-predictor-corrector expects 0|1");st.study_inner_predictor_corrector=v=="1"; }
      else if(a=="--objective-convexification") {
        const std::string mode=need("--objective-convexification");
        if(mode!="off"&&mode!="lazy")throw std::runtime_error("objective-convexification must be off or lazy");
        st.objective_lazy_convexification=mode=="lazy";
      }
      else if(a=="--objective-max-atoms") st.objective_max_atoms=parse_num<int>(need("--objective-max-atoms"));
      else if(a=="--objective-initial-atoms") st.objective_initial_atoms=parse_num<int>(need("--objective-initial-atoms"));
      else if(a=="--objective-atom-batch") st.objective_atom_batch=parse_num<int>(need("--objective-atom-batch"));
      else if(a=="--objective-rho-decay") st.objective_rho_decay=parse_num<double>(need("--objective-rho-decay"));
      else if(a=="--objective-rho-excess") st.objective_rho_excess=parse_num<double>(need("--objective-rho-excess"));
      else if(a=="--dca-engine") st.dca_engine=need("--dca-engine");
      else if(a=="--native-al-fallback") {
        const std::string v=need("--native-al-fallback");
        if(v!="auto"&&v!="off")throw std::runtime_error("native-al-fallback expects auto|off");
        st.native_al_fallback=v;
      }
      else if(a=="--standard-r10-policy") st.standard_r10_policy=need("--standard-r10-policy");
      else if(a=="--gr-constraint-increments") {
        const std::string v=need("--gr-constraint-increments");
        if(v!="coefficients"&&v!="rounded")throw std::runtime_error("gr-constraint-increments expects coefficients|rounded");
        st.gr_coefficient_constraint_increments=v=="coefficients";
      }
      else if(a=="--gr-proximal-policy") {
        st.gr_proximal_policy=need("--gr-proximal-policy");
        if(st.gr_proximal_policy!="historical"&&st.gr_proximal_policy!="frozen")
          throw std::runtime_error("gr-proximal-policy expects historical|frozen");
      }
      else if(a=="--gr-fixed-centering-steps") {
        st.gr_fixed_centering_steps=parse_num<int>(need("--gr-fixed-centering-steps"));
        if(st.gr_fixed_centering_steps<0||st.gr_fixed_centering_steps>8)
          throw std::runtime_error("gr-fixed-centering-steps expects 0..8");
      }
      else if(a=="--gr-residual-components") {
        const std::string v=need("--gr-residual-components");
        if(v!="0"&&v!="1")throw std::runtime_error("gr-residual-components expects 0|1");
        st.gr_residual_components=v=="1";
      }
      else if(a=="--dca-inner-max") st.dca_filter_inner_max=parse_num<int>(need("--dca-inner-max"));
      else if(a=="--dca-aggregate-full-refinement") st.dca_aggregate_full_refinement=true;
      else if(a=="--no-dca-aggregate-full-refinement") st.dca_aggregate_full_refinement=false;
      else if(a=="--dca-aggregate-adaptive-dual") st.dca_aggregate_adaptive_dual=true;
      else if(a=="--no-dca-aggregate-adaptive-dual") st.dca_aggregate_adaptive_dual=false;
      else if(a=="--acceptable-iter") {const std::string v=need("--acceptable-iter");std::size_t n=0;st.acceptable_iterations=std::stoi(v,&n);if(n!=v.size()||st.acceptable_iterations<0)throw std::runtime_error("acceptable-iter must be a nonnegative integer");}
      else if(a=="--acceptable-on-failure") st.acceptable_on_failure=true;
      else if(a=="--no-acceptable-on-failure") st.acceptable_on_failure=false;
      else if(a=="--dca-endpoint-start") st.dca_endpoint_start_file=need("--dca-endpoint-start");
      else if(a=="--dca-control-tol") st.dca_control_tolerance=parse_num<double>(need("--dca-control-tol"));
      else if(a=="--termination-test") st.termination_test=need("--termination-test");
      else if(a=="--acceptable-obj-change-tol") st.acceptable_objective_change_tolerance=parse_num<double>(need("--acceptable-obj-change-tol"));
      else if(a=="--acceptable-tol") st.acceptable_tolerance=parse_num<double>(need("--acceptable-tol"));
      else if(a=="--acceptable-constr-viol-tol") st.acceptable_constraint_tolerance=parse_num<double>(need("--acceptable-constr-viol-tol"));
      else if(a=="--acceptable-dual-inf-tol") st.acceptable_dual_tolerance=parse_num<double>(need("--acceptable-dual-inf-tol"));
      else if(a=="--acceptable-compl-inf-tol") st.acceptable_complementarity_tolerance=parse_num<double>(need("--acceptable-compl-inf-tol"));
      else if(a=="--dca-aggregate-adaptive-pivot") st.dca_aggregate_adaptive_pivot=true;
      else if(a=="--no-dca-aggregate-adaptive-pivot") st.dca_aggregate_adaptive_pivot=false;
      else if(a=="--mumps-memory-recovery") st.mumps_memory_recovery=true;
      else if(a=="--no-mumps-memory-recovery") st.mumps_memory_recovery=false;
      else if(a=="--dca-restoration-compare") st.dca_restoration_compare=true;
      else if(a=="--no-dca-restoration-compare") st.dca_restoration_compare=false;
      else if(a=="--dca-filter-diagnostics") st.dca_filter_diagnostics=true;
      else if(a=="--no-dca-filter-diagnostics") st.dca_filter_diagnostics=false;
      else if(a=="--dca-dual-decomposition") st.dca_dual_decomposition=true;
      else if(a=="--no-dca-dual-decomposition") st.dca_dual_decomposition=false;
      else if(a=="--dca-dual-decomposition-growth") st.dca_dual_decomposition_growth=parse_num<double>(need("--dca-dual-decomposition-growth"));
      else if(a=="--dca-model-retries") {
        const std::string value=need("--dca-model-retries");
        std::size_t consumed=0;
        st.dca_filter_model_retries=std::stoi(value,&consumed);
        if(consumed!=value.size() || st.dca_filter_model_retries<0 || st.dca_filter_model_retries>100)
          throw std::runtime_error("--dca-model-retries must be an integer in [0,100]");
      }
      else if(a=="--max-atoms") st.dca_filter_max_atoms=parse_num<int>(need("--max-atoms"));
      else if(a=="--atom-events") st.dca_filter_atom_events=true;
      else if(a=="--no-atom-events") st.dca_filter_atom_events=false;
      else if(a=="--atom-event-every") st.dca_filter_atom_every=parse_num<int>(need("--atom-event-every"));
      else if(a=="--atom-batch-size") st.dca_filter_atom_batch=parse_num<int>(need("--atom-batch-size"));
      else if(a=="--atom-basis-keep") {
        st.dca_atom_basis_keep=parse_num<int>(need(a.c_str()));
        if(st.dca_atom_basis_keep<1)throw std::invalid_argument("--atom-basis-keep must be >= 1");
      } else if(a=="--atom-basis-tail") {
        st.dca_atom_basis_tail=parse_num<int>(need(a.c_str()));
        if(st.dca_atom_basis_tail<0)throw std::invalid_argument("--atom-basis-tail must be >= 0");
      } else if(a=="--atom-probe-depth") {
        st.dca_atom_probe_depth=parse_num<int>(need(a.c_str()));
        if(st.dca_atom_probe_depth<1)throw std::invalid_argument("--atom-probe-depth must be >= 1");
      }
      else if(a=="--atom-probe-count-projection") {
        const std::string v=need(a.c_str());
        if(v!="yes"&&v!="no")throw std::invalid_argument("atom-probe-count-projection must be yes or no");
        st.dca_atom_probe_count_projection=v=="yes";
      }
      else if(a=="--newton-engine") st.ln_engine=need("--newton-engine");
      else if (a == "--lazy-newton-reduction") st.lazy_newton_reduction = true;
      else if (a == "--no-lazy-newton-reduction") st.lazy_newton_reduction = false;
      else if (a == "--lazy-newton-lazy-curvature") st.lazy_newton_lazy_curvature = true;
      else if (a == "--no-lazy-newton-lazy-curvature") st.lazy_newton_lazy_curvature = false;
      else if (a == "--lazy-newton-lazy-constraints") st.lazy_newton_lazy_constraints = true;
      else if (a == "--no-lazy-newton-lazy-constraints") st.lazy_newton_lazy_constraints = false;
      else if (a == "--no-lazy-newton-packed-gradients") st.lazy_newton_packed_gradients = false;
      else if (a == "--lazy-newton-packed-gradients") st.lazy_newton_packed_gradients = true;
      else if (a == "--lazy-newton-predictor-corrector") st.lazy_newton_predictor_corrector = true;
      else if (a == "--no-lazy-newton-predictor-corrector") st.lazy_newton_predictor_corrector = false;
      else if (a == "--lazy-newton-activation-margin") st.lazy_newton_activation_margin=std::stod(need("--lazy-newton-activation-margin"));
      else if (a == "--lazy-newton-dense-threshold") st.lazy_newton_dense_threshold=std::stoi(need("--lazy-newton-dense-threshold"));
      else if (a == "--lazy-newton-adaptive-barrier") st.lazy_newton_adaptive_barrier=true;
      else if (a == "--no-lazy-newton-adaptive-barrier") st.lazy_newton_adaptive_barrier=false;
      else if (a == "--lazy-newton-interleaved-ordering") st.lazy_newton_interleaved_ordering=true;
      else if (a == "--no-lazy-newton-interleaved-ordering") st.lazy_newton_interleaved_ordering=false;
      else if (a == "--lazy-newton-sparse-recovery") st.lazy_newton_sparse_recovery=true;
      else if (a == "--no-lazy-newton-sparse-recovery") st.lazy_newton_sparse_recovery=false;
      else if (a == "--lazy-newton-initial-mu") {
        st.lazy_newton_initial_mu=std::stod(need("--lazy-newton-initial-mu"));
        st.lazy_newton_initial_mu_explicit=true;
      }
      else if (a == "--lazy-newton-regularization-initial") st.lazy_newton_regularization_initial=std::stod(need("--lazy-newton-regularization-initial"));
      else if (a == "--lazy-newton-regularization-decay") st.lazy_newton_regularization_decay=std::stod(need("--lazy-newton-regularization-decay"));
      else if (a == "--lazy-newton-objective-gradient-target") st.lazy_newton_objective_gradient_target=std::stod(need("--lazy-newton-objective-gradient-target"));
      else if (a == "--lazy-newton-constraint-gradient-target") st.lazy_newton_constraint_gradient_target=std::stod(need("--lazy-newton-constraint-gradient-target"));
      else if (a == "--display-level") {st.live_display_level=parse_num<int>(need("--display-level"));if(st.live_display_level<0||st.live_display_level>4)throw std::runtime_error("display-level must be between 0 and 4");}
      else if (a == "--lazy-newton-sparse-certificate") st.lazy_newton_sparse_certificate=true;
      else if (a == "--no-lazy-newton-sparse-certificate") st.lazy_newton_sparse_certificate=false;
      else if (a == "--lazy-newton-atom-guided-after" || a == "--lazy-newton-atom-plateau-batches") st.lazy_newton_atom_guided_after=parse_num<int>(need("--lazy-newton-atom-guided-after"));
      else if (a == "--lazy-newton-atom-guided") st.lazy_newton_atom_guided=true;
      else if (a == "--no-lazy-newton-atom-guided") st.lazy_newton_atom_guided=false;
      else if (a == "--lazy-newton-atom-step-decay") st.lazy_newton_atom_step_decay=true;
      else if (a == "--no-lazy-newton-atom-step-decay") st.lazy_newton_atom_step_decay=false;
      else if (a == "--lazy-newton-atom-transport") st.lazy_newton_atom_transport=true;
      else if (a == "--no-lazy-newton-atom-transport") st.lazy_newton_atom_transport=false;
      else if (a == "--lazy-newton-atomic") st.lazy_newton_atomic_regularization = true;
      else if (a == "--no-lazy-newton-atom-energy") st.lazy_newton_atom_energy_sparsify=false;
      else if (a == "--lazy-newton-atom-energy") st.lazy_newton_atom_energy_sparsify=true;
      else if (a == "--no-lazy-newton-atom-rayleigh") st.lazy_newton_atom_rayleigh_gate=false;
      else if (a == "--lazy-newton-atom-rayleigh") st.lazy_newton_atom_rayleigh_gate=true;
      else if (a == "--lazy-newton-atom-decay") st.lazy_newton_atom_decay = parse_num<double>(need("--lazy-newton-atom-decay"));
      else if (a == "--live-curvature") st.live_curvature = true;
      else if (a == "--lazy-newton-condensation-guard") st.lazy_newton_condensation_guard = true;
      else if (a == "--no-lazy-newton-condensation-guard") st.lazy_newton_condensation_guard = false;
      else if (a == "--lazy-newton-condensation-ratio") st.lazy_newton_condensation_ratio = parse_num<double>(need("--lazy-newton-condensation-ratio"));
      else if(a=="--lazy-newton-uniform") st.lazy_newton_atomic_regularization=false;
      else if(a=="--no-compact-qcq-input") compact_qcq_input=false;
      else if (a == "--ln-mu-init") st.ln_mu_init = std::stod(need(a.c_str()));
      else if (a == "--ln-mu-kappa") st.ln_mu_kappa = std::stod(need(a.c_str()));
      else if (a == "--ln-mu-theta") st.ln_mu_theta = std::stod(need(a.c_str()));
      else if (a == "--ln-kappa-epsilon") st.ln_kappa_epsilon = std::stod(need(a.c_str()));
      else if (a == "--ln-activation") st.ln_activation = std::stod(need(a.c_str()));
      else if (a == "--ln-envelope") st.ln_envelope = std::stod(need(a.c_str()));
      else if (a == "--ln-drop-patience") st.ln_drop_patience = std::stoi(need(a.c_str()));
      else if (a == "--ln-drop-distance") st.ln_drop_distance = std::stod(need(a.c_str()));
      else if (a == "--ln-delta-c") st.ln_delta_c = std::stod(need(a.c_str()));
      else if (a == "--ln-delta-x-init") st.ln_delta_x_init = std::stod(need(a.c_str()));
      else if (a == "--ln-delta-x-min") st.ln_delta_x_min = std::stod(need(a.c_str()));
      else if (a == "--ln-kkt-refine") st.ln_kkt_refine = std::stoi(need(a.c_str()));
      else if (a == "--ln-kkt-berr-tol") st.ln_kkt_berr_tol = std::stod(need(a.c_str()));
      else if (a == "--ln-quadratic-correction") st.ln_quadratic_correction = true;
      else if (a == "--no-ln-quadratic-correction") st.ln_quadratic_correction = false;
      else if (a == "--ln-max-soc") st.ln_max_soc = std::stoi(need(a.c_str()));
      else if (a == "--ln-bound-push") st.ln_bound_push = std::stod(need(a.c_str()));
      else if (a == "--ln-tau-min") st.ln_tau_min = std::stod(need(a.c_str()));
      else if (a == "--ln-max-line-search") st.ln_max_line_search = std::stoi(need(a.c_str()));
      else if (a == "--ln-prox-alpha-min") st.ln_prox_alpha_min = std::stod(need(a.c_str()));
      else if (a == "--ln-mult-init-max") st.ln_mult_init_max = std::stod(need(a.c_str()));
      else if (a == "--ln-scaling-max-gradient") st.ln_scaling_max_gradient = std::stod(need(a.c_str()));
      else if (a == "--ln-inequality-curvature") st.ln_inequality_curvature = need(a.c_str());
      else if (a == "--ln-ordering") st.ln_ordering = need(a.c_str());
      else if (a == "--ln-condense-max-support") st.ln_condense_max_support = std::stoi(need(a.c_str()));
      else if (a == "--ln-globalization") st.ln_globalization = need(a.c_str());
      else if (a == "--ln-transport-complementarity-floor") st.ln_transport_complementarity_floor = std::stod(need(a.c_str()));
      else if (a == "--ln-adaptive-curvature-delta") st.ln_adaptive_curvature_delta = std::stod(need(a.c_str()));
      else if (a == "--auto-sqp-max-n") st.auto_sqp_max_dimension = parse_num<int>(need("--auto-sqp-max-n"));
      else if (a == "--inner-backend") st.inner_backend = need("--inner-backend");
      else if (a == "--qp-option") st.qp_option = need("--qp-option");
      else if (a == "--qp-amrs") st.qp_amrs = true;
      else if (a == "--no-qp-amrs") st.qp_amrs = false;
      else if (a == "--qp-init") st.qp_init = parse_num<int>(need("--qp-init"));
      else if (a == "--qp-backend") {
        st.qp_backend = need("--qp-backend");
        st.linear_backend = st.qp_backend;
      }
      else if (a == "--qp-rho-trial-every") st.qp_rho_trial_every = parse_num<int>(need("--qp-rho-trial-every"));
      else if (a == "--qp-max-atoms") st.qp_max_atoms = parse_num<int>(need("--qp-max-atoms"));
      else if (a == "--qp-seed-coverage") st.qp_seed_coverage = parse_num<double>(need("--qp-seed-coverage"));
      else if (a == "--qp-seed-stall-tol") st.qp_seed_stall_tol = parse_num<double>(need("--qp-seed-stall-tol"));
      else if (a == "--qp-seed-patience") st.qp_seed_patience = parse_num<int>(need("--qp-seed-patience"));
      else if (a == "--qp-atom-events") st.qp_atom_events = true;
      else if (a == "--no-qp-atom-events") st.qp_atom_events = false;
      else if (a == "--qp-atom-event-every") st.qp_atom_event_every = parse_num<int>(need("--qp-atom-event-every"));
      else if (a == "--qp-bound-push") {
        st.qp_bound_push = parse_num<double>(need("--qp-bound-push"));
        st.inner_bound_push = st.qp_bound_push;
      }
      else if (a == "--qp-bp-push-kappa") st.qp_bp_push_kappa = parse_num<double>(need("--qp-bp-push-kappa"));
      else if (a == "--qp-bp-margin-tol") st.qp_bp_margin_tol = parse_num<double>(need("--qp-bp-margin-tol"));
      else if (a == "--qp-escapes") st.qp_escapes = parse_num<int>(need("--qp-escapes"));
      else if (a == "--qp-mehrotra") st.qp_mehrotra = true;
      else if (a == "--no-qp-mehrotra") st.qp_mehrotra = false;
      else if (a == "--qp-sigma-max") st.qp_sigma_max = parse_num<double>(need("--qp-sigma-max"));
      else if (a == "--qp-model-solved-tol") st.qp_model_solved_tol = parse_num<double>(need("--qp-model-solved-tol"));
      else if (a == "--qp-dca-stop") st.qp_dca_stop = true;
      else if (a == "--no-qp-dca-stop") st.qp_dca_stop = false;
      else if (a == "--qp-dca-stall-tol-rel") st.qp_dca_stall_tol_rel = parse_num<double>(need("--qp-dca-stall-tol-rel"));
      else if (a == "--qp-dca-stall-iters") st.qp_dca_stall_iters = parse_num<int>(need("--qp-dca-stall-iters"));
      else if (a == "--qp-dca-monotone-tol-abs") st.qp_dca_monotone_tol_abs = parse_num<double>(need("--qp-dca-monotone-tol-abs"));
      else if (a == "--qp-bp-adaptive") st.qp_bp_adaptive = true;
      else if (a == "--no-qp-bp-adaptive") st.qp_bp_adaptive = false;
      else if (a == "--hsl-library") st.hsl_library=need(a.c_str());
      else if (a == "--hsl-pivot-tol") st.hsl_pivot_tolerance=parse_num<double>(need(a.c_str()));
      else if (a == "--hsl-memory-factor") st.hsl_memory_factor=parse_num<double>(need(a.c_str()));
      else if (a == "--linear-backend") {
        st.linear_backend = need("--linear-backend");
        st.qp_backend = st.linear_backend;
      }
      else if (a == "--linear-metis") st.linear_metis = parse_num<int>(need("--linear-metis")) != 0;
      else if (a == "--kkt-persistent-symbolic-cache") st.kkt_persistent_symbolic_cache = true;
      else if (a == "--no-kkt-persistent-symbolic-cache") st.kkt_persistent_symbolic_cache = false;
      else if (a == "--kkt-primal-reg") st.kkt_primal_regularization = parse_num<double>(need("--kkt-primal-reg"));
      else if (a == "--kkt-dual-reg") st.kkt_dual_regularization = parse_num<double>(need("--kkt-dual-reg"));
      else if (a == "--kkt-adaptive-reg") st.kkt_adaptive_regularization = true;
      else if (a == "--no-kkt-adaptive-reg") st.kkt_adaptive_regularization = false;
      else if (a == "--rho-x-floor") st.kkt_rho_x_floor = parse_num<double>(need("--rho-x-floor"));
      else if (a == "--rho-x-growth") st.kkt_rho_x_growth = parse_num<double>(need("--rho-x-growth"));
      else if (a == "--rho-x-decay") st.kkt_rho_x_decay = parse_num<double>(need("--rho-x-decay"));
      else if (a == "--rho-x-max") st.kkt_rho_x_max = parse_num<double>(need("--rho-x-max"));
      else if (a == "--rho-x-clean") st.kkt_rho_x_clean_outer = parse_num<int>(need("--rho-x-clean"));
      else if (a == "--delta-dual-base") st.kkt_delta_dual_base = parse_num<double>(need("--delta-dual-base"));
      else if (a == "--delta-dual-floor") st.kkt_delta_dual_floor = parse_num<double>(need("--delta-dual-floor"));
      else if (a == "--delta-dual-growth") st.kkt_delta_dual_growth = parse_num<double>(need("--delta-dual-growth"));
      else if (a == "--delta-dual-decay") st.kkt_delta_dual_decay = parse_num<double>(need("--delta-dual-decay"));
      else if (a == "--delta-dual-max") st.kkt_delta_dual_max = parse_num<double>(need("--delta-dual-max"));
      else if (a == "--kkt-reg-retries") st.kkt_regularization_retry_max = parse_num<int>(need("--kkt-reg-retries"));
      else if (a == "--delta-dual-mode") {
        const std::string mode=need("--delta-dual-mode");
        if(mode=="inverse") st.kkt_delta_dual_inverse=true;
        else if(mode=="fixed") st.kkt_delta_dual_inverse=false;
        else throw std::runtime_error("--delta-dual-mode must be inverse or fixed");
      }
      else if (a == "--kkt-berr-tol") st.kkt_backward_error_tolerance = parse_num<double>(need("--kkt-berr-tol"));
      else if (a == "--kkt-berr-reject") st.kkt_backward_error_reject = parse_num<double>(need("--kkt-berr-reject"));
      else if (a == "--kkt-refine") st.kkt_iterative_refinement_steps = parse_num<int>(need("--kkt-refine"));
      else if (a == "--cholesk-adaptive-reg") st.cholesk_adaptive_regularization = true;
      else if (a == "--no-cholesk-adaptive-reg") st.cholesk_adaptive_regularization = false;
      else if (a == "--cholesk-reg-warm-start") st.cholesk_regularization_warm_start = true;
      else if (a == "--no-cholesk-reg-warm-start") st.cholesk_regularization_warm_start = false;
      else if (a == "--cholesk-equilibration") st.cholesk_equilibration = true;
      else if (a == "--no-cholesk-equilibration") st.cholesk_equilibration = false;
      else if (a == "--cholesk-pivot-rel-tol") st.cholesk_relative_pivot_tolerance = parse_num<double>(need("--cholesk-pivot-rel-tol"));
      else if (a == "--equality-qp-original-kkt-gmres") st.equality_qp_original_kkt_gmres = true;
      else if (a == "--no-equality-qp-original-kkt-gmres") st.equality_qp_original_kkt_gmres = false;
      else if (a == "--equality-qp-original-kkt-gmres-max-iterations")
        st.equality_qp_original_kkt_gmres_max_iterations =
            parse_num<int>(need("--equality-qp-original-kkt-gmres-max-iterations"));
      else if (a == "--accepted-checkpoint") {
        const std::string v=need("--accepted-checkpoint");
        if(v!="0"&&v!="1")throw std::runtime_error("accepted-checkpoint expects 0|1");
        accepted_checkpoint=v=="1";
      }
      else if (a == "--json") json_path = need("--json");
      else if (a == "--frozen-start") frozen_start_path = need("--frozen-start");
      else if (a == "--max-outer") st.max_outer_iterations = parse_num<int>(need("--max-outer"));
      else if (a == "--max-inner") st.max_inner_iterations = parse_num<int>(need("--max-inner"));
      else if (a == "--time-limit") st.time_limit_seconds = parse_num<double>(need("--time-limit"));
      else if (a == "--feas-tol") st.feasibility_tolerance = parse_num<double>(need("--feas-tol"));
      else if (a == "--opt-tol") st.stationarity_tolerance = parse_num<double>(need("--opt-tol"));
      else if (a == "--dual-inf-tol") st.dual_infeasibility_tolerance = parse_num<double>(need("--dual-inf-tol"));
      else if (a == "--compl-inf-tol") st.complementarity_tolerance = parse_num<double>(need("--compl-inf-tol"));
      else if (a == "--s-max") st.nlp_error_scaling_max = parse_num<double>(need("--s-max"));
      else if (a == "--inner-tol") st.inner_tolerance = parse_num<double>(need("--inner-tol"));
      else if (a == "--inner-mu-fallback-tol") st.inner_iteration_limit_mu_fallback = parse_num<double>(need("--inner-mu-fallback-tol"));
      else if (a == "--lazyqp-mu0") st.lazyqp_barrier_mu0 = parse_num<double>(need("--lazyqp-mu0"));
      else if (a == "--lazyqp-mu-r") st.lazyqp_barrier_r = parse_num<double>(need("--lazyqp-mu-r"));
      else if (a == "--lazyqp-mu-theta") st.lazyqp_barrier_theta = parse_num<double>(need("--lazyqp-mu-theta"));
      else if (a == "--lazyqp-mu-inner-max") st.lazyqp_mu_inner_max = parse_num<int>(need("--lazyqp-mu-inner-max"));
      else if (a == "--lazyqp-numeric-kkt-reuse") st.lazyqp_numeric_kkt_reuse = true;
      else if (a == "--no-lazyqp-numeric-kkt-reuse") st.lazyqp_numeric_kkt_reuse = false;
      else if (a == "--lazyqp-numeric-kkt-reuse-age") st.lazyqp_numeric_kkt_reuse_max_age = parse_num<int>(need("--lazyqp-numeric-kkt-reuse-age"));
      else if (a == "--lazyqp-numeric-kkt-reuse-iters") st.lazyqp_numeric_kkt_reuse_max_iterations = parse_num<int>(need("--lazyqp-numeric-kkt-reuse-iters"));
      else if (a == "--lazyqp-inexact-forcing") st.lazyqp_inexact_forcing = true;
      else if (a == "--no-lazyqp-inexact-forcing") st.lazyqp_inexact_forcing = false;
      else if (a == "--lazyqp-forcing-eta") st.lazyqp_forcing_eta = parse_num<double>(need("--lazyqp-forcing-eta"));
      else if (a == "--lazyqp-forcing-max-tol") st.lazyqp_forcing_max_tolerance = parse_num<double>(need("--lazyqp-forcing-max-tol"));
      else if (a == "--lazyqp-forcing-fast-iters") st.lazyqp_forcing_fast_iterations = parse_num<int>(need("--lazyqp-forcing-fast-iters"));
      else if (a == "--lazyqp-forcing-min-iters") st.lazyqp_forcing_min_iterations = parse_num<int>(need("--lazyqp-forcing-min-iters"));
      else if (a == "--dca-single-loop-primal-dual") st.dca_single_loop_primal_dual = true;
      else if (a == "--no-dca-single-loop-primal-dual") st.dca_single_loop_primal_dual = false;
      else if (a == "--dca-single-loop-multiplier-transport") st.dca_single_loop_multiplier_transport = true;
      else if (a == "--no-dca-single-loop-multiplier-transport") st.dca_single_loop_multiplier_transport = false;
      else if (a == "--dca-single-loop-retraction-fallback-only") st.dca_single_loop_retraction_fallback_only = true;
      else if (a == "--no-dca-single-loop-retraction-fallback-only") st.dca_single_loop_retraction_fallback_only = false;
      else if (a == "--dca-single-loop-inner-steps") st.dca_single_loop_inner_steps = parse_num<int>(need("--dca-single-loop-inner-steps"));
      else if (a == "--dca-single-loop-forcing-eta") st.dca_single_loop_forcing_eta = parse_num<double>(need("--dca-single-loop-forcing-eta"));
      else if (a == "--dca-single-loop-baseline-prox") st.dca_single_loop_baseline_proximal_regularization = parse_num<double>(need("--dca-single-loop-baseline-prox"));
      else if (a == "--dca-single-loop-direct-start") st.dca_single_loop_direct_start = true;
      else if (a == "--no-dca-single-loop-direct-start") st.dca_single_loop_direct_start = false;
      else if (a == "--dca-single-loop-direct-start-theta-cap") st.dca_single_loop_direct_start_theta_cap = parse_num<double>(need("--dca-single-loop-direct-start-theta-cap"));
      else if (a == "--dca-single-loop-direct-start-gamma-h") st.dca_single_loop_direct_start_gamma_h = parse_num<double>(need("--dca-single-loop-direct-start-gamma-h"));
      else if (a == "--lazyqp-kkt-form") st.lazyqp_kkt_form = need("--lazyqp-kkt-form");
      else if (a == "--lazyqp-reduced-kkt-min-ineq") st.lazyqp_reduced_kkt_min_inequalities = parse_num<int>(need("--lazyqp-reduced-kkt-min-ineq"));
      else if (a == "--lazyqp-hybrid-retained-ineq") st.lazyqp_hybrid_kkt_max_retained_inequalities = parse_num<int>(need("--lazyqp-hybrid-retained-ineq"));
      else if (a == "--lazyqp-hybrid-min-row-nnz") st.lazyqp_hybrid_kkt_min_row_nnz = parse_num<int>(need("--lazyqp-hybrid-min-row-nnz"));
      else if (a == "--lazyqp-flow-lift-elimination") st.lazyqp_flow_lift_elimination = true;
      else if (a == "--no-lazyqp-flow-lift-elimination") st.lazyqp_flow_lift_elimination = false;
      else if (a == "--lazyqp-model-support-fastpath") st.lazyqp_model_support_fastpath = true;
      else if (a == "--no-lazyqp-model-support-fastpath") st.lazyqp_model_support_fastpath = false;
      else if (a == "--lazyqp-affine-support-fastpath") st.lazyqp_affine_support_fastpath = true;
      else if (a == "--no-lazyqp-affine-support-fastpath") st.lazyqp_affine_support_fastpath = false;
      else if (a == "--lazyqp-low-rank-form") st.lazyqp_low_rank_form = need("--lazyqp-low-rank-form");
      else if (a == "--lazyqp-condense-threshold") st.lazyqp_low_rank_condense_threshold = parse_num<int>(need("--lazyqp-condense-threshold"));
      else if (a == "--lazyqp-warm-guard") st.lazyqp_warm_start_guard_factor = parse_num<double>(need("--lazyqp-warm-guard"));
      else if (a == "--lazyqp-zero-dual-continuation") st.lazyqp_zero_dual_continuation = true;
      else if (a == "--no-lazyqp-zero-dual-continuation") st.lazyqp_zero_dual_continuation = false;
      else if (a == "--lazyqp-zero-dual-mu-cap") st.lazyqp_zero_dual_mu_cap = parse_num<double>(need("--lazyqp-zero-dual-mu-cap"));
      else if (a == "--inner-warm-start" || a == "--inner-equality-warm-start") { st.inner_dual_warm_start = true; st.inner_inequality_dual_warm_start = false; }
      else if (a == "--inner-full-dual-warm-start") { st.inner_dual_warm_start = true; st.inner_inequality_dual_warm_start = true; }
      else if (a == "--no-inner-warm-start") { st.inner_dual_warm_start = false; st.inner_inequality_dual_warm_start = false; }
      else if (a == "--inner-bound-push") {
        st.inner_bound_push = parse_num<double>(need("--inner-bound-push"));
        st.qp_bound_push = st.inner_bound_push;
      }
      else if (a == "--inner-bound-frac") st.inner_bound_frac = parse_num<double>(need("--inner-bound-frac"));
      else if (a == "--inner-bound-push-ref") st.inner_bound_push_reference = parse_num<double>(need("--inner-bound-push-ref"));
      else if (a == "--inner-bound-mult-init") st.inner_bound_mult_init_val = parse_num<double>(need("--inner-bound-mult-init"));
      else if (a == "--restoration-iters") st.restoration_max_iterations = parse_num<int>(need("--restoration-iters"));
      else if (a == "--shared-restoration-budget") {
        const std::string v=need("--shared-restoration-budget");
        if(v!="0"&&v!="1")throw std::runtime_error("shared-restoration-budget expects 0|1");
        st.restoration_shared_iteration_budget=v=="1";
      }
      else if (a == "--restoration-restarts") st.restoration_max_restarts = parse_num<int>(need("--restoration-restarts"));
      else if (a == "--restoration-restart-iters") st.restoration_restart_iterations = parse_num<int>(need("--restoration-restart-iters"));
      else if (a == "--restoration-exact-workspace-reuse") st.restoration_exact_workspace_reuse = true;
      else if (a == "--no-restoration-exact-workspace-reuse") st.restoration_exact_workspace_reuse = false;
      else if (a == "--restoration-exact-parallel-row-dots") st.restoration_exact_parallel_row_dots = true;
      else if (a == "--no-restoration-exact-parallel-row-dots") st.restoration_exact_parallel_row_dots = false;
      else if (a == "--restoration-exact-sparse-row-dots") st.restoration_exact_sparse_row_dots = true;
      else if (a == "--no-restoration-exact-sparse-row-dots") st.restoration_exact_sparse_row_dots = false;
      else if (a == "--restoration-exact-parallel-min-rows")
        st.restoration_exact_parallel_min_rows =
            parse_num<int>(need("--restoration-exact-parallel-min-rows"));
      else if (a == "--restoration-tangent-activation") st.restoration_tangent_activation = true;
      else if (a == "--no-restoration-tangent-activation") st.restoration_tangent_activation = false;
      else if (a == "--restoration-minimax-iters") st.restoration_minimax_max_iterations = parse_num<int>(need("--restoration-minimax-iters"));
      else if (a == "--restoration-minimax-prox") st.restoration_minimax_proximal = parse_num<double>(need("--restoration-minimax-prox"));
      else if (a == "--prox") st.proximal_regularization = parse_num<double>(need("--prox"));
      else if (a == "--dca-extra-prox") st.dca_extra_proximal_regularization = parse_num<double>(need("--dca-extra-prox"));
      else if (a == "--dca-affine-full-limit") st.dca_affine_full_constraint_limit = parse_num<int>(need("--dca-affine-full-limit"));
      else if (a == "--dca-affine-full") st.dca_full_constraints_for_affine_objective = true;
      else if (a == "--no-dca-affine-full") st.dca_full_constraints_for_affine_objective = false;
      else if (a == "--dca-affine-box-bounded-lazy") st.dca_affine_box_bounded_lazy = true;
      else if (a == "--no-dca-affine-box-bounded-lazy") st.dca_affine_box_bounded_lazy = false;
      else if (a == "--dca-corrector") st.dca_local_corrector = true;
      else if (a == "--no-dca-corrector") st.dca_local_corrector = false;
      else if (a == "--dca-corrector-start") st.dca_local_corrector_start_outer = parse_num<int>(need("--dca-corrector-start"));
      else if (a == "--dca-corrector-frequency") st.dca_local_corrector_frequency = parse_num<int>(need("--dca-corrector-frequency"));
      else if (a == "--dca-iterated-soc") st.dca_iterated_soc = true;
      else if (a == "--no-dca-iterated-soc") st.dca_iterated_soc = false;
      else if (a == "--dca-iterated-soc-iters") st.dca_iterated_soc_max_iterations = parse_num<int>(need("--dca-iterated-soc-iters"));
      else if (a == "--dca-iterated-soc-max-ratio") st.dca_iterated_soc_max_ratio = parse_num<double>(need("--dca-iterated-soc-max-ratio"));
      else if (a == "--dca-iterated-soc-trigger") st.dca_iterated_soc_trigger_ratio = parse_num<double>(need("--dca-iterated-soc-trigger"));
      else if (a == "--dca-warm-failure-limit") st.dca_warm_failure_limit = parse_num<int>(need("--dca-warm-failure-limit"));
      else if (a == "--dca-warm-cooldown") st.dca_warm_cooldown = parse_num<int>(need("--dca-warm-cooldown"));
      else if (a == "--dca-inner-retries") st.dca_inner_retry_max = parse_num<int>(need("--dca-inner-retries"));
      else if (a == "--dca-inner-retry-prox") st.dca_inner_retry_initial_prox = parse_num<double>(need("--dca-inner-retry-prox"));
      else if (a == "--dca-step-control") st.dca_step_control = true;
      else if (a == "--no-dca-step-control") st.dca_step_control = false;
      else if (a == "--dca-step-target-abs") st.dca_step_target_absolute = parse_num<double>(need("--dca-step-target-abs"));
      else if (a == "--dca-step-target-rel") st.dca_step_target_relative = parse_num<double>(need("--dca-step-target-rel"));
      else if (a == "--dca-step-target-max") st.dca_step_target_maximum = parse_num<double>(need("--dca-step-target-max"));
      else if (a == "--dca-step-control-retries") st.dca_step_control_max_retries = parse_num<int>(need("--dca-step-control-retries"));
      else if (a == "--dca-equality-funnel") st.dca_equality_funnel_initial = parse_num<double>(need("--dca-equality-funnel"));
      else if (a == "--dca-equality-funnel-auto") st.dca_equality_funnel_auto = true;
      else if (a == "--no-dca-equality-funnel-auto") st.dca_equality_funnel_auto = false;
      else if (a == "--dca-equality-funnel-ceiling") st.dca_equality_funnel_auto_ceiling = parse_num<double>(need("--dca-equality-funnel-ceiling"));
      else if (a == "--dca-equality-retraction") st.dca_equality_retraction = true;
      else if (a == "--no-dca-equality-retraction") st.dca_equality_retraction = false;
      else if (a == "--dca-equality-retraction-max-ratio") st.dca_equality_retraction_max_ratio = parse_num<double>(need("--dca-equality-retraction-max-ratio"));
      else if (a == "--dca-equality-model") st.dca_equality_model = need("--dca-equality-model");
      else if (a == "--dca-tangent-inertia-trigger") st.dca_tangent_inertia_trigger=std::stod(need("--dca-tangent-inertia-trigger"));
      else if (a == "--dca-condensed-newton-iters") st.dca_condensed_newton_iterations = parse_num<int>(need("--dca-condensed-newton-iters"));
      else if (a == "--dca-condensed-krylov-iters") st.dca_condensed_krylov_iterations = parse_num<int>(need("--dca-condensed-krylov-iters"));
      else if (a == "--dca-condensed-krylov-rtol") st.dca_condensed_krylov_relative_tolerance = parse_num<double>(need("--dca-condensed-krylov-rtol"));
      else if (a == "--dca-condensed-continuation") st.dca_condensed_continuation_attempts = parse_num<int>(need("--dca-condensed-continuation"));
      else if (a == "--dca-condensed-tol") st.dca_condensed_retraction_tolerance = parse_num<double>(need("--dca-condensed-tol"));
      else if (a == "--dca-condensed-regularization") st.dca_condensed_regularization = parse_num<double>(need("--dca-condensed-regularization"));
      else if (a == "--dca-condensed-reg-growth") st.dca_condensed_regularization_growth = parse_num<double>(need("--dca-condensed-reg-growth"));
      else if (a == "--dca-condensed-reg-max") st.dca_condensed_regularization_max = parse_num<double>(need("--dca-condensed-reg-max"));
      else if (a == "--dca-condensed-max-ratio") st.dca_condensed_retraction_max_ratio = parse_num<double>(need("--dca-condensed-max-ratio"));
      else if (a == "--dca-condensed-bound-aware-normal") st.dca_condensed_bound_aware_normal = true;
      else if (a == "--no-dca-condensed-bound-aware-normal") st.dca_condensed_bound_aware_normal = false;
      else if (a == "--dca-condensed-bound-activity-tol") st.dca_condensed_bound_activity_tolerance = parse_num<double>(need("--dca-condensed-bound-activity-tol"));
      else if (a == "--dca-condensed-curvature") st.dca_condensed_tangent_curvature = true;
      else if (a == "--no-dca-condensed-curvature") st.dca_condensed_tangent_curvature = false;
      else if (a == "--dca-condensed-projected-multi-ritz-inertia") st.dca_condensed_projected_multi_ritz_inertia = true;
      else if (a == "--no-dca-condensed-projected-multi-ritz-inertia") st.dca_condensed_projected_multi_ritz_inertia = false;
      else if (a == "--dca-condensed-block-exact-curvature") st.dca_condensed_block_exact_curvature = true;
      else if (a == "--no-dca-condensed-block-exact-curvature") st.dca_condensed_block_exact_curvature = false;
      else if (a == "--dca-condensed-lanczos-steps") st.dca_condensed_lanczos_steps = parse_num<int>(need("--dca-condensed-lanczos-steps"));
      else if (a == "--dca-condensed-lanczos-starts") st.dca_condensed_lanczos_starts = parse_num<int>(need("--dca-condensed-lanczos-starts"));
      else if (a == "--dca-condensed-max-rank") st.dca_condensed_max_tangent_rank = parse_num<int>(need("--dca-condensed-max-rank"));
      else if (a == "--dca-condensed-scalar-safeguard-updates") st.dca_condensed_scalar_safeguard_updates = parse_num<int>(need("--dca-condensed-scalar-safeguard-updates"));
      else if (a == "--dca-condensed-affine-normal-reuse") st.dca_condensed_affine_normal_reuse = true;
      else if (a == "--no-dca-condensed-affine-normal-reuse") st.dca_condensed_affine_normal_reuse = false;
      else if (a == "--dca-condensed-quadratic-linesearch-reuse") st.dca_condensed_quadratic_linesearch_reuse = true;
      else if (a == "--no-dca-condensed-quadratic-linesearch-reuse") st.dca_condensed_quadratic_linesearch_reuse = false;
      else if (a == "--dca-condensed-sparse-equalities") st.dca_condensed_sparse_equality_evaluation = true;
      else if (a == "--no-dca-condensed-sparse-equalities") st.dca_condensed_sparse_equality_evaluation = false;
      else if (a == "--dca-condensed-cached-regularization") st.dca_condensed_cached_regularization = true;
      else if (a == "--no-dca-condensed-cached-regularization") st.dca_condensed_cached_regularization = false;
      else if (a == "--dca-condensed-parallel-equalities") st.dca_condensed_parallel_equalities = true;
      else if (a == "--no-dca-condensed-parallel-equalities") st.dca_condensed_parallel_equalities = false;
      else if (a == "--dca-condensed-parallel-min-equalities") st.dca_condensed_parallel_min_equalities = parse_num<int>(need("--dca-condensed-parallel-min-equalities"));
      else if (a == "--dca-condensed-adaptive-normal-refinement") st.dca_condensed_adaptive_normal_refinement = true;
      else if (a == "--no-dca-condensed-adaptive-normal-refinement") st.dca_condensed_adaptive_normal_refinement = false;
      else if (a == "--dca-condensed-normal-intermediate-berr-tol") st.dca_condensed_normal_intermediate_backward_error_tolerance = parse_num<double>(need("--dca-condensed-normal-intermediate-berr-tol"));
      else if (a == "--dca-condensed-normal-intermediate-berr-reject") st.dca_condensed_normal_intermediate_backward_error_reject = parse_num<double>(need("--dca-condensed-normal-intermediate-berr-reject"));
      else if (a == "--dca-condensed-normal-strict-stationarity-factor") st.dca_condensed_normal_strict_stationarity_factor = parse_num<double>(need("--dca-condensed-normal-strict-stationarity-factor"));
      else if (a == "--dca-condensed-alpha-predictor") st.dca_condensed_alpha_predictor = true;
      else if (a == "--no-dca-condensed-alpha-predictor") st.dca_condensed_alpha_predictor = false;
      else if (a == "--dca-condensed-alpha-growth") st.dca_condensed_alpha_growth = parse_num<double>(need("--dca-condensed-alpha-growth"));
      else if (a == "--dca-condensed-alpha-full-step-frequency") st.dca_condensed_alpha_full_step_frequency = parse_num<int>(need("--dca-condensed-alpha-full-step-frequency"));
      else if (a == "--dca-condensed-outer-extrapolation") st.dca_condensed_outer_extrapolation = true;
      else if (a == "--no-dca-condensed-outer-extrapolation") st.dca_condensed_outer_extrapolation = false;
      else if (a == "--dca-condensed-outer-extrapolation-alpha") st.dca_condensed_outer_extrapolation_alpha = parse_num<double>(need("--dca-condensed-outer-extrapolation-alpha"));
      else if (a == "--dca-condensed-outer-extrapolation-min-full-steps") st.dca_condensed_outer_extrapolation_min_full_steps = parse_num<int>(need("--dca-condensed-outer-extrapolation-min-full-steps"));
      else if (a == "--dca-condensed-post-retraction-ws-expansion") st.dca_condensed_post_retraction_working_set_expansion = true;
      else if (a == "--no-dca-condensed-post-retraction-ws-expansion") st.dca_condensed_post_retraction_working_set_expansion = false;
      else if (a == "--dca-condensed-modeled-side-filter") st.dca_condensed_modeled_side_filter = true;
      else if (a == "--no-dca-condensed-modeled-side-filter") st.dca_condensed_modeled_side_filter = false;
      else if (a == "--dca-condensed-modeled-side-filter-limit") st.dca_condensed_modeled_side_filter_limit = parse_num<double>(need("--dca-condensed-modeled-side-filter-limit"));
      else if (a == "--dca-condensed-modeled-side-recovery-margin") st.dca_condensed_modeled_side_recovery_margin = parse_num<double>(need("--dca-condensed-modeled-side-recovery-margin"));
      else if (a == "--dca-condensed-flow-lift-schur") st.dca_condensed_flow_lift_schur = true;
      else if (a == "--no-dca-condensed-flow-lift-schur") st.dca_condensed_flow_lift_schur = false;
      else if (a == "--dca-condensed-post-retraction-ws-max") st.dca_condensed_post_retraction_working_set_max_expansions = parse_num<int>(need("--dca-condensed-post-retraction-ws-max"));
      else if (a == "--dca-equality-elastic-penalty") st.dca_equality_elastic_penalty = parse_num<double>(need("--dca-equality-elastic-penalty"));
      else if (a == "--dca-equality-elastic-quadratic") st.dca_equality_elastic_quadratic = parse_num<double>(need("--dca-equality-elastic-quadratic"));
      else if (a == "--dca-equality-elastic-cap-factor") st.dca_equality_elastic_cap_factor = parse_num<double>(need("--dca-equality-elastic-cap-factor"));
      else if (a == "--dca-equality-elastic-post-retraction") st.dca_equality_elastic_post_retraction = true;
      else if (a == "--no-dca-equality-elastic-post-retraction") st.dca_equality_elastic_post_retraction = false;
      else if (a == "--activation") st.working_set_activation = parse_num<double>(need("--activation"));
      else if (a == "--max-rank") {
        const int r=parse_num<int>(need("--max-rank"));
        st.max_rank_per_quadratic=r; st.max_objective_rank=r; st.max_lagrangian_rank=r;
      }
      else if (a == "--max-constraint-rank") st.max_rank_per_quadratic = parse_num<int>(need("--max-constraint-rank"));
      else if (a == "--max-objective-rank") st.max_objective_rank = parse_num<int>(need("--max-objective-rank"));
      else if (a == "--max-lagrangian-rank") st.max_lagrangian_rank = parse_num<int>(need("--max-lagrangian-rank"));
      else if (a == "--constraint-model") {
        const std::string mode=need("--constraint-model");
        if (mode=="auto" || mode=="linearized") st.phase2_linearized_inequalities=true;
        else if (mode=="majorant") st.phase2_linearized_inequalities=false;
        else throw std::runtime_error("--constraint-model must be auto, linearized, or majorant");
      }
      else if (a == "--line-search-armijo") st.line_search_armijo = parse_num<double>(need("--line-search-armijo"));
      else if (a == "--no-tangential") st.tangential_globalization_enabled = false;
      else if (a == "--tangential") st.tangential_globalization_enabled = true;
      else if (a == "--trust-region") st.trust_region_enabled = true;
      else if (a == "--no-trust-region") st.trust_region_enabled = false;
      else if (a == "--tr-initial-radius") st.trust_region_initial_radius = parse_num<double>(need("--tr-initial-radius"));
      else if (a == "--tr-min-radius") st.trust_region_min_radius = parse_num<double>(need("--tr-min-radius"));
      else if (a == "--tr-max-radius") st.trust_region_max_radius = parse_num<double>(need("--tr-max-radius"));
      else if (a == "--tr-eta-accept") st.trust_region_eta_accept = parse_num<double>(need("--tr-eta-accept"));
      else if (a == "--tr-eta-shrink") st.trust_region_eta_shrink = parse_num<double>(need("--tr-eta-shrink"));
      else if (a == "--tr-eta-expand") st.trust_region_eta_expand = parse_num<double>(need("--tr-eta-expand"));
      else if (a == "--tr-shrink") st.trust_region_shrink_factor = parse_num<double>(need("--tr-shrink"));
      else if (a == "--tr-expand") st.trust_region_expand_factor = parse_num<double>(need("--tr-expand"));
      else if (a == "--tr-boundary-fraction") st.trust_region_boundary_fraction = parse_num<double>(need("--tr-boundary-fraction"));
      else if (a == "--tr-boundary-rel-tol") st.trust_region_boundary_relative_tolerance = parse_num<double>(need("--tr-boundary-rel-tol"));
      else if (a == "--tr-boundary-abs-tol") st.trust_region_boundary_absolute_tolerance = parse_num<double>(need("--tr-boundary-abs-tol"));
      else if (a == "--tr-max-rejects") st.trust_region_max_rejections = parse_num<int>(need("--tr-max-rejects"));
      else if (a == "--tr-feas-floor") st.trust_region_feasibility_floor = parse_num<double>(need("--tr-feas-floor"));
      else if (a == "--tr-feas-growth") st.trust_region_feasibility_growth = parse_num<double>(need("--tr-feas-growth"));
      else if (a == "--tr-feas-reduction") st.trust_region_feasibility_reduction = parse_num<double>(need("--tr-feas-reduction"));
      else if (a == "--tr-penalty-min") st.trust_region_penalty_min = parse_num<double>(need("--tr-penalty-min"));
      else if (a == "--tr-penalty-safety") st.trust_region_penalty_safety = parse_num<double>(need("--tr-penalty-safety"));
      else if (a == "--no-soc") st.second_order_correction = false;
      else if (a == "--soc-max-ratio") st.second_order_correction_max_ratio = parse_num<double>(need("--soc-max-ratio"));
      else if (a == "--soc-trigger-ratio") st.second_order_correction_trigger_ratio = parse_num<double>(need("--soc-trigger-ratio"));
      else if (a == "--working-set-drop-slack") st.working_set_drop_slack = parse_num<double>(need("--working-set-drop-slack"));
      else if (a == "--working-set-drop-mu") st.working_set_drop_multiplier = parse_num<double>(need("--working-set-drop-mu"));
      else if (a == "--working-set-drop-patience") st.working_set_drop_patience = parse_num<int>(need("--working-set-drop-patience"));
      else if (a == "--no-working-set-drop") st.working_set_dropping = false;
      else if (a == "--terminal-kkt-trigger") st.terminal_kkt_trigger_stationarity = parse_num<double>(need("--terminal-kkt-trigger"));
      else if (a == "--terminal-kkt-iters") st.terminal_kkt_max_iterations = parse_num<int>(need("--terminal-kkt-iters"));
      else if (a == "--terminal-kkt-mu-abs") st.terminal_kkt_multiplier_tolerance = parse_num<double>(need("--terminal-kkt-mu-abs"));
      else if (a == "--terminal-kkt-mu-rel") st.terminal_kkt_multiplier_relative_tolerance = parse_num<double>(need("--terminal-kkt-mu-rel"));
      else if (a == "--terminal-kkt-stability") st.terminal_kkt_active_stability_iterations = parse_num<int>(need("--terminal-kkt-stability"));
      else if (a == "--terminal-kkt") { st.terminal_kkt_enabled = true; st.terminal_kkt_large_scale_enabled = true; }
      else if (a == "--no-terminal-kkt") st.terminal_kkt_enabled = false;
      else if (a == "--lanczos-steps") st.lanczos_steps = parse_num<int>(need("--lanczos-steps"));
      else if (a == "--lanczos-starts") st.lanczos_starts = parse_num<int>(need("--lanczos-starts"));
      else if (a == "--exact-eigen-threshold") st.exact_eigen_threshold = parse_num<int>(need("--exact-eigen-threshold"));
      else if (a == "--strict-scalar-certificate") st.strict_scalar_certificate = true;
      else if (a == "--no-restoration") st.restoration_enabled = false;
      else if (a == "--live-iterations") st.live_iterations = true;
      else if (a == "--no-live-iterations") st.live_iterations = false;
      else if (a == "--live-iterations-every") st.live_iteration_every = parse_num<int>(need("--live-iterations-every"));
      else if (a == "--live-header-every") st.live_iteration_header_every = parse_num<int>(need("--live-header-every"));
      else if (a == "--quiet") st.verbose = false;
      else if (a == "--qp-defaults") {
        std::cout << "{"
                  << "\"qp_option\":\"" << st.qp_option << "\","
                  << "\"qp_amrs\":" << (st.qp_amrs ? "true" : "false") << ","
                  << "\"qp_init\":" << st.qp_init << ","
                  << "\"qp_backend\":\"" << st.qp_backend << "\","
                  << "\"qp_rho_trial_every\":" << st.qp_rho_trial_every << ","
                  << "\"qp_max_atoms\":" << st.qp_max_atoms << ","
                  << "\"qp_seed_coverage\":" << st.qp_seed_coverage << ","
                  << "\"qp_seed_stall_tol\":" << st.qp_seed_stall_tol << ","
                  << "\"qp_seed_patience\":" << st.qp_seed_patience << ","
                  << "\"qp_atom_events\":" << (st.qp_atom_events ? "true" : "false") << ","
                  << "\"qp_atom_event_every\":" << st.qp_atom_event_every << ","
                  << "\"qp_bound_push\":" << st.qp_bound_push << ","
                  << "\"qp_bp_push_kappa\":" << st.qp_bp_push_kappa << ","
                  << "\"qp_bp_margin_tol\":" << st.qp_bp_margin_tol << ","
                  << "\"qp_escapes\":" << st.qp_escapes << ","
                  << "\"qp_mehrotra\":" << (st.qp_mehrotra ? "true" : "false") << ","
                  << "\"qp_sigma_max\":" << st.qp_sigma_max << ","
                  << "\"qp_model_solved_tol\":" << st.qp_model_solved_tol << ","
                  << "\"qp_dca_stop\":" << (st.qp_dca_stop ? "true" : "false") << ","
                  << "\"qp_dca_stall_tol_rel\":" << st.qp_dca_stall_tol_rel << ","
                  << "\"qp_dca_stall_iters\":" << st.qp_dca_stall_iters << ","
                  << "\"qp_dca_monotone_tol_abs\":" << st.qp_dca_monotone_tol_abs << ","
                  << "\"qp_bp_adaptive\":" << (st.qp_bp_adaptive ? "true" : "false")
                  << "}\n";
        return 0;
      }
      else if (a == "--linear-backends") {
        std::cout << compiled_linear_backends() << "\n";
        return 0;
      }
      else if (a == "--version") {
        std::cout << kSolverName << " " << kVersion << " (API "
                  << kApiVersion << ", " << kBuildId << ") [backend-patch=KKT-PIVOT-DELAY-20261026-R1] [repair=R12 exact-free-epigraph-preserve-start bounded-closed-box-recovery accurate-callback-values] [candidate=HH-R3-R1 hybrid-corrections R3-controller-default] [water24-recovery=20261023_R2] [pooling-recovery=20261023_R4 postsolve-multiplier-fit native-barrier-retry] [beam-hybrid-recovery=20261024_R5 exact-curvature-cache affine-fill-guard]"
                  << " [phase1-sparse-jv="
                  << (restoration_exact_sparse_row_dots_build_compatible()
                          ? "runtime-guarded"
                          : "disabled-fast-fp")
                  << "] [repair=R9-THREE-PRIORITIES-R4 recovery=column-lm-and-small-step continuation=separate] [repair=R9-CORE2-REGRESSION-R2 model-release=1 feasible-proximal=1] [repair=R9-PGLIB-FILTER-RECENTER-R1 opt-in=1] [repair=R9-AUTO-STARTUP-R1 opt-in=1] [repair=R9-AUTO-STARTUP-R2 low-mu-stall-retry=1] [repair=R9-AUTO-STARTUP-R3 terminal-central-retry=1]\n";
        return 0;
      } else if (a == "--help" || a == "-h") {
        usage(argv[0]); return 0;
      } else throw std::runtime_error("unknown option: " + a);
    }
    if (problem_path.empty()) { usage(argv[0]); return 2; }
    if (!(st.stationarity_tolerance > 0.0) || !(st.feasibility_tolerance > 0.0) ||
        !(st.dual_infeasibility_tolerance > 0.0) || !(st.complementarity_tolerance > 0.0) ||
        !(st.nlp_error_scaling_max > 0.0))
      throw std::runtime_error("KKT-stop tolerances and --s-max must be positive");
    if (st.outer_algorithm != "auto" && st.outer_algorithm != "lazy-dca" &&
        st.outer_algorithm != "legacy-sqp" && st.outer_algorithm != "lazy-newton")
      throw std::runtime_error("--algorithm must be lazy-dca, lazy-newton, auto (alias), or legacy-sqp (comparison)");
    if (st.outer_algorithm == "lazy-newton") {
      if (!(st.ln_mu_init > 0.0) || !(st.ln_mu_kappa > 0.0 && st.ln_mu_kappa < 1.0) ||
          !(st.ln_mu_theta > 1.0) || !(st.ln_kappa_epsilon > 0.0) || !(st.ln_activation >= 0.0) ||
          st.ln_drop_patience < 1 || !(st.ln_delta_c >= 0.0) || !(st.ln_delta_x_init > 0.0) ||
          st.ln_kkt_refine < 0 || !(st.ln_kkt_berr_tol > 0.0) || !(st.ln_bound_push > 0.0) ||
          !(st.ln_tau_min > 0.0 && st.ln_tau_min < 1.0) || st.ln_max_line_search < 1 || st.ln_max_soc < 1)
        throw std::runtime_error("--ln-* controls are outside their admissible range");
      if (st.ln_engine=="legacy" && st.linear_backend != "ldl")
        throw std::runtime_error("--algorithm lazy-newton requires --linear-backend ldl");
      if (st.ln_inequality_curvature != "majorant" && st.ln_inequality_curvature != "exact" &&
          st.ln_inequality_curvature != "adaptive")
        throw std::runtime_error("--ln-inequality-curvature must be adaptive, majorant, or exact");
      if (st.ln_ordering != "amd" && st.ln_ordering != "metis")
        throw std::runtime_error("--ln-ordering must be amd or metis");
      if (st.ln_globalization != "filter" && st.ln_globalization != "merit")
        throw std::runtime_error("--ln-globalization must be filter or merit");
      if (st.ln_condense_max_support < 0)
        throw std::runtime_error("--ln-condense-max-support must be nonnegative");
    }
    if (st.inner_backend != "lazyqp" && st.inner_backend != "socp" &&
        st.inner_backend != "ipm" && st.inner_backend != "native" &&
        st.inner_backend != "legacy-ipm" && st.inner_backend != "filter-sqp" &&
        st.inner_backend != "tr-sqp" && st.inner_backend != "ipopt" &&
        st.inner_backend != "lazyqp-sparse")
      throw std::runtime_error("--inner-backend must be lazyqp, lazyqp-sparse, socp, filter-sqp, tr-sqp, ipopt, or legacy-ipm");
    if (st.inner_backend == "lazyqp-sparse" && st.linear_backend != "ldl")
      throw std::runtime_error("--inner-backend lazyqp-sparse requires --linear-backend ldl");
    if (st.inner_backend == "socp" && st.outer_algorithm == "lazy-dca" &&
        st.dca_equality_model != "linearized" &&
        st.dca_equality_model != "condensed-retraction")
      throw std::runtime_error(
          "--inner-backend socp supports --dca-equality-model linearized or condensed-retraction");
    if (st.linear_backend != "native" && st.linear_backend != "ldl" && st.linear_backend != "qdldl" && st.linear_backend != "cholesk" && st.linear_backend != "mumps" && st.linear_backend != "ma27")
      throw std::runtime_error("--qp-backend/--linear-backend must be native, cholesk, ldl, qdldl, mumps, or ma27");
    if (!std::isfinite(st.cholesk_relative_pivot_tolerance) ||
        st.cholesk_relative_pivot_tolerance < 0.0)
      throw std::runtime_error("--cholesk-pivot-rel-tol must be finite and nonnegative");
    const auto finite_nonnegative=[](double value) {
      return std::isfinite(value) && value>=0.0;
    };
    if (!finite_nonnegative(st.dca_dual_decomposition_growth))
      throw std::runtime_error("--dca-dual-decomposition-growth must be finite and nonnegative");
    if (!finite_nonnegative(st.dca_common_step_primal_trigger) ||
        !std::isfinite(st.dca_common_step_dual_growth) ||
        st.dca_common_step_dual_growth<1.0)
      throw std::runtime_error("common-step primal trigger must be nonnegative and dual growth must be finite and at least one");
    if (!finite_nonnegative(st.kkt_primal_regularization) ||
        !finite_nonnegative(st.kkt_dual_regularization) ||
        !finite_nonnegative(st.kkt_rho_x_floor) ||
        !finite_nonnegative(st.kkt_rho_x_max) ||
        st.kkt_rho_x_max<st.kkt_rho_x_floor ||
        !std::isfinite(st.kkt_rho_x_growth) || st.kkt_rho_x_growth<=1.0 ||
        !std::isfinite(st.kkt_rho_x_decay) || st.kkt_rho_x_decay<=0.0 ||
        st.kkt_rho_x_decay>1.0 || st.kkt_rho_x_clean_outer<1 ||
        !finite_nonnegative(st.kkt_delta_dual_floor) ||
        !finite_nonnegative(st.kkt_delta_dual_max) ||
        st.kkt_delta_dual_max<st.kkt_delta_dual_floor ||
        !std::isfinite(st.kkt_delta_dual_growth) ||
        st.kkt_delta_dual_growth<=1.0 ||
        !std::isfinite(st.kkt_delta_dual_decay) ||
        st.kkt_delta_dual_decay<=0.0 || st.kkt_delta_dual_decay>1.0 ||
        st.kkt_regularization_retry_max<0)
      throw std::runtime_error(
          "invalid Cholesk regularization floors, caps, growth, decay, clean window, or retry budget");
    if (st.qp_option.empty()) throw std::runtime_error("--qp-option must not be empty");
    if (st.qp_init < 0 || st.qp_init > 2) throw std::runtime_error("--qp-init must be 0, 1, or 2");
    if (st.equality_qp_original_kkt_gmres_max_iterations < 0)
      throw std::runtime_error("--equality-qp-original-kkt-gmres-max-iterations must be nonnegative");
    if (st.restoration_exact_parallel_min_rows < 1)
      throw std::runtime_error("--restoration-exact-parallel-min-rows must be positive");
    if (st.working_set_drop_patience < 1)
      throw std::runtime_error("--working-set-drop-patience must be positive");
    if (st.dca_condensed_post_retraction_working_set_max_expansions < 0)
      throw std::runtime_error(
          "--dca-condensed-post-retraction-ws-max must be nonnegative");
    if (!std::isfinite(st.dca_condensed_modeled_side_filter_limit) ||
        st.dca_condensed_modeled_side_filter_limit <= 0.0)
      throw std::runtime_error(
          "--dca-condensed-modeled-side-filter-limit must be finite and positive");
    if (!std::isfinite(st.dca_condensed_modeled_side_recovery_margin) ||
        st.dca_condensed_modeled_side_recovery_margin < 0.0)
      throw std::runtime_error(
          "--dca-condensed-modeled-side-recovery-margin must be finite and nonnegative");
    if (st.qp_rho_trial_every < 1 || st.qp_max_atoms < 0 || st.qp_seed_patience < 0 ||
        st.qp_atom_event_every < 1 || st.qp_escapes < 0 || st.qp_dca_stall_iters < 0)
      throw std::runtime_error("integer qp_ controls are outside their admissible range");
    if (!(st.qp_seed_coverage >= 0.0 && st.qp_seed_coverage <= 1.0) ||
        !(st.qp_bound_push >= 0.0) || !(st.qp_bp_push_kappa >= 0.0) ||
        !(st.qp_bp_margin_tol >= 0.0) || !(st.qp_sigma_max > 0.0) ||
        !(st.qp_model_solved_tol > 0.0) || !(st.qp_dca_stall_tol_rel >= 0.0) ||
        !(st.qp_dca_monotone_tol_abs >= 0.0))
      throw std::runtime_error("real-valued qp_ controls are outside their admissible range");
    if (st.lazyqp_kkt_form == "augmented") st.lazyqp_kkt_form = "hybrid";
    if (st.lazyqp_kkt_form != "auto" && st.lazyqp_kkt_form != "full" &&
        st.lazyqp_kkt_form != "reduced" && st.lazyqp_kkt_form != "hybrid")
      throw std::runtime_error("--lazyqp-kkt-form must be auto, full, reduced, or hybrid");
    if (st.lazyqp_reduced_kkt_min_inequalities < 0)
      throw std::runtime_error("--lazyqp-reduced-kkt-min-ineq must be nonnegative");
    if (st.lazyqp_hybrid_kkt_max_retained_inequalities < 0)
      throw std::runtime_error("--lazyqp-hybrid-retained-ineq must be nonnegative");
    if (st.lazyqp_hybrid_kkt_min_row_nnz < 1)
      throw std::runtime_error("--lazyqp-hybrid-min-row-nnz must be positive");
    if (st.lazyqp_low_rank_form != "auto" && st.lazyqp_low_rank_form != "condensed" &&
        st.lazyqp_low_rank_form != "bordered")
      throw std::runtime_error("--lazyqp-low-rank-form must be auto, condensed, or bordered");
    if (st.lazyqp_low_rank_condense_threshold < 0)
      throw std::runtime_error("--lazyqp-condense-threshold must be nonnegative");
    if (!(st.lazyqp_warm_start_guard_factor >= 1.0))
      throw std::runtime_error("--lazyqp-warm-guard must be at least 1");
    if (!std::isfinite(st.lazyqp_zero_dual_mu_cap) ||
        !(st.lazyqp_zero_dual_mu_cap > 0.0))
      throw std::runtime_error("--lazyqp-zero-dual-mu-cap must be finite and positive");
    if (st.lazyqp_numeric_kkt_reuse_max_age < 0 ||
        st.lazyqp_numeric_kkt_reuse_max_iterations < 0)
      throw std::runtime_error("LazyQP numerical KKT reuse limits must be nonnegative");
    if (st.lazyqp_inexact_forcing &&
        (!(st.lazyqp_forcing_eta > 0.0 && st.lazyqp_forcing_eta < 1.0) ||
         !(st.lazyqp_forcing_max_tolerance > st.inner_tolerance) ||
         st.lazyqp_forcing_fast_iterations < 1 ||
         st.lazyqp_forcing_min_iterations < 1 ||
         st.lazyqp_forcing_min_iterations >
             st.lazyqp_forcing_fast_iterations))
      throw std::runtime_error("LazyQP forcing controls require 0<eta<1, max-tol>inner-tol, and 1<=min-iters<=fast-iters");
    if (st.dca_single_loop_inner_steps < 1 ||
        !std::isfinite(st.dca_single_loop_forcing_eta) ||
        !(st.dca_single_loop_forcing_eta > 0.0 &&
          st.dca_single_loop_forcing_eta < 1.0))
      throw std::runtime_error(
          "single-loop primal-dual controls require inner-steps>=1 and 0<forcing-eta<1");
    if (st.dca_single_loop_primal_dual &&
        st.dca_single_loop_inner_steps != 1)
      throw std::runtime_error(
          "--dca-single-loop-primal-dual requires --dca-single-loop-inner-steps 1");
    if (!std::isfinite(st.dca_single_loop_baseline_proximal_regularization) ||
        st.dca_single_loop_baseline_proximal_regularization < 0.0)
      throw std::runtime_error(
          "--dca-single-loop-baseline-prox must be finite and nonnegative");
    if (st.dca_single_loop_primal_dual &&
        st.outer_algorithm != "lazy-dca")
      throw std::runtime_error(
          "--dca-single-loop-primal-dual requires --algorithm lazy-dca");
    if (st.dca_single_loop_primal_dual &&
        st.inner_backend != "lazyqp" && st.inner_backend != "socp" &&
        st.inner_backend != "lazyqp-sparse")
      throw std::runtime_error(
          "--dca-single-loop-primal-dual requires --inner-backend lazyqp, lazyqp-sparse, or socp");
    if (st.dca_single_loop_primal_dual &&
        st.dca_single_loop_retraction_fallback_only)
      throw std::runtime_error(
          "--dca-single-loop-primal-dual forbids --dca-single-loop-retraction-fallback-only");
    if (st.dca_single_loop_primal_dual && st.lazyqp_numeric_kkt_reuse)
      throw std::runtime_error(
          "--dca-single-loop-primal-dual forbids --lazyqp-numeric-kkt-reuse");
    if (st.dca_single_loop_primal_dual && !st.qp_mehrotra)
      throw std::runtime_error(
          "--dca-single-loop-primal-dual requires --qp-mehrotra");
    if (!std::isfinite(st.dca_single_loop_direct_start_theta_cap) ||
        !(st.dca_single_loop_direct_start_theta_cap > 0.0))
      throw std::runtime_error(
          "--dca-single-loop-direct-start-theta-cap must be finite and positive");
    if (!std::isfinite(st.dca_single_loop_direct_start_gamma_h) ||
        !(st.dca_single_loop_direct_start_gamma_h > 0.0 &&
          st.dca_single_loop_direct_start_gamma_h < 1.0))
      throw std::runtime_error(
          "--dca-single-loop-direct-start-gamma-h must lie in (0,1)");
    if (st.dca_single_loop_direct_start &&
        !st.dca_single_loop_primal_dual)
      throw std::runtime_error(
          "--dca-single-loop-direct-start requires --dca-single-loop-primal-dual");
    if (!std::isfinite(st.terminal_kkt_trigger_stationarity) ||
        !(st.terminal_kkt_trigger_stationarity > 0.0) ||
        !std::isfinite(st.terminal_kkt_active_slack) ||
        !(st.terminal_kkt_active_slack >= 0.0) ||
        !std::isfinite(st.terminal_kkt_multiplier_tolerance) ||
        !(st.terminal_kkt_multiplier_tolerance >= 0.0) ||
        !std::isfinite(st.terminal_kkt_multiplier_relative_tolerance) ||
        !(st.terminal_kkt_multiplier_relative_tolerance >= 0.0) ||
        st.terminal_kkt_active_stability_iterations < 1 ||
        !std::isfinite(st.terminal_kkt_primal_regularization) ||
        !(st.terminal_kkt_primal_regularization >= 0.0) ||
        !std::isfinite(st.terminal_kkt_dual_regularization) ||
        !(st.terminal_kkt_dual_regularization >= 0.0) ||
        st.terminal_kkt_max_iterations < 1)
      throw std::runtime_error("terminal KKT controls require a positive finite trigger, nonnegative finite cutoffs/regularizations, stability>=1, and iters>=1");
    if (st.dca_equality_model != "linearized" &&
        st.dca_equality_model != "condensed-retraction" &&
        st.dca_equality_model != "tangent-inertia" &&
        st.dca_equality_model != "elastic-split")
      throw std::runtime_error("--dca-equality-model must be linearized, condensed-retraction, tangent-inertia, or elastic-split");
    if (st.dca_condensed_newton_iterations < 1 ||
        st.dca_condensed_krylov_iterations < 1 ||
        st.dca_condensed_continuation_attempts < 1 ||
        st.dca_condensed_lanczos_steps < 2 ||
        st.dca_condensed_lanczos_starts < 1 ||
        st.dca_condensed_max_tangent_rank < 0 ||
        st.dca_condensed_scalar_safeguard_updates < 1 ||
        !(st.dca_condensed_krylov_relative_tolerance > 0.0) ||
        !(st.dca_condensed_retraction_tolerance > 0.0) ||
        !(st.dca_condensed_regularization >= 0.0) ||
        !(st.dca_condensed_regularization_growth > 1.0) ||
        !(st.dca_condensed_regularization_max >= st.dca_condensed_regularization) ||
        !(st.dca_condensed_bound_activity_tolerance >= 0.0) ||
        !(st.dca_condensed_retraction_max_ratio > 0.0) ||
        !(st.dca_condensed_normal_intermediate_backward_error_tolerance > 0.0) ||
        !(st.dca_condensed_normal_intermediate_backward_error_reject >=
          st.dca_condensed_normal_intermediate_backward_error_tolerance) ||
        !(st.dca_condensed_normal_strict_stationarity_factor > 0.0) ||
        !(st.dca_condensed_alpha_growth > 1.0) ||
        st.dca_condensed_alpha_full_step_frequency < 1 ||
        !(st.dca_condensed_outer_extrapolation_alpha > 1.0) ||
        !(st.dca_condensed_outer_extrapolation_alpha <= 2.0) ||
        st.dca_condensed_outer_extrapolation_min_full_steps < 1)
      throw std::runtime_error("invalid condensed quadratic retraction parameters");
    if (!(st.dca_equality_elastic_penalty >= 0.0) ||
        !(st.dca_equality_elastic_quadratic >= 0.0) ||
        !(st.dca_equality_elastic_cap_factor > 0.0))
      throw std::runtime_error("invalid elastic equality parameters");

    if (!(st.dca_step_target_absolute >= 0.0) ||
        !(st.dca_step_target_relative >= 0.0) ||
        !(st.dca_step_target_maximum > 0.0) ||
        st.dca_step_control_max_retries < 0)
      throw std::runtime_error("DCA step-control targets must be nonnegative, the maximum positive, and retries nonnegative");
    if (st.live_iteration_every < 1 || st.live_iteration_header_every < 1)
      throw std::runtime_error("live-iteration periods must be positive integers");
    if (st.inner_backend != "ipopt" && !linear_backend_compiled(st.linear_backend))
      throw std::runtime_error("linear backend '" + st.linear_backend + "' not compiled; available: " + compiled_linear_backends());

    if(st.study_controller!="r3"&&st.study_controller!="certified")throw std::runtime_error("study-controller must be r3 or certified");
    detail::factorial_configure(st);
    if(st.convexification_study!="off") {
      const std::set<std::string> modes={"U","F","SS","SS_legacy","SA","AS","AA","HH","HS","SH"};
      if(!modes.count(st.convexification_study))throw std::runtime_error("unknown convexification-study");
      if(st.convexification_schedule!="lazy"&&st.convexification_schedule!="eager")throw std::runtime_error("convexification-schedule must be lazy or eager");
      if((st.convexification_study=="U"||st.convexification_study=="F")&&st.convexification_schedule!="eager")throw std::runtime_error("fixed U/F require --convexification-schedule eager");
      if((st.dca_engine!="majorants"&&st.dca_engine!="auto")||st.linear_backend!="mumps"||!st.linear_metis||st.inner_backend=="ipopt")throw std::runtime_error("convexification study requires native dca-engine majorants/auto and MUMPS+METIS");
      const bool hybrid_study=st.convexification_study=="HH"||st.convexification_study=="HS"||st.convexification_study=="SH"||st.convexification_study=="SS";
      if(hybrid_study &&
          (st.study_hybrid_max_atoms<0 || st.study_hybrid_max_atoms>256 || st.study_hybrid_total_atoms<0 || st.study_hybrid_total_atoms>512 ||
           st.study_hybrid_event_interval<1 || st.study_hybrid_downweight_patience<1 || st.study_hybrid_atom_support<1 ||
           !std::isfinite(st.study_hybrid_downweight_factor) || st.study_hybrid_downweight_factor<=0 || st.study_hybrid_downweight_factor>=1 ||
           !std::isfinite(st.study_hybrid_dominance) || st.study_hybrid_dominance<=0 ||
           !std::isfinite(st.study_hybrid_scalar_decay) || st.study_hybrid_scalar_decay<=0 || st.study_hybrid_scalar_decay>=1 ||
           !std::isfinite(st.study_hybrid_margin) || st.study_hybrid_margin<0 ||
           !std::isfinite(st.study_inner_relative) || st.study_inner_relative<=0 || st.study_inner_relative>=1 ||
           !std::isfinite(st.study_inner_absolute) || st.study_inner_absolute<0 ||
           st.study_inner_min<1 || st.study_inner_max<st.study_inner_min))
        throw std::runtime_error("invalid hybrid controller or inexact-solve parameters");
      if(hybrid_study) {
        if(st.dca_pair_elimination)
          throw std::runtime_error("HH/HS/SH/SS require direct bordered MUMPS; optional pair elimination is unsupported");
        if(st.study_hybrid_candidates<1||st.study_hybrid_candidates>64||
           st.study_hybrid_lanczos_steps<2||st.study_hybrid_lanczos_steps>200||
           st.study_hybrid_lanczos_starts<1||st.study_hybrid_lanczos_starts>8||
           st.study_hybrid_additions_per_event<1||st.study_hybrid_additions_per_event>32||
           !std::isfinite(st.study_hybrid_support_energy)||st.study_hybrid_support_energy<=0||st.study_hybrid_support_energy>1||
           !std::isfinite(st.study_hybrid_strength)||st.study_hybrid_strength<=0||
           !std::isfinite(st.study_hybrid_beta_offset)||st.study_hybrid_beta_offset<=0||
           !std::isfinite(st.study_hybrid_negative_tolerance)||st.study_hybrid_negative_tolerance<=0||
           !std::isfinite(st.study_hybrid_scalar_safety)||st.study_hybrid_scalar_safety<=1||
           !std::isfinite(st.study_hybrid_scalar_floor)||st.study_hybrid_scalar_floor<0||st.study_hybrid_scalar_floor>1||
           !std::isfinite(st.study_hybrid_objective_floor)||st.study_hybrid_objective_floor<=0)
          throw std::runtime_error("invalid manuscript selector parameters");
        for(double v:{st.study_hybrid_score_activity,st.study_hybrid_score_scalar,
            st.study_hybrid_score_support,st.study_hybrid_score_density,st.study_hybrid_score_weight,
            st.study_hybrid_min_lift,st.study_hybrid_min_activity,st.study_hybrid_min_score})
          if(!std::isfinite(v)||v<0)throw std::runtime_error("selector scores and thresholds must be finite and nonnegative");
        if(st.study_controller=="certified"&&((st.study_inner_rule!="model-relative"&&st.study_inner_rule!="barrier-relative"&&st.study_inner_rule!="native-relative"&&st.study_inner_rule!="native-direction-relative"&&st.study_inner_rule!="native-reduced-relative"&&st.study_inner_rule!="native-reduced-direction-relative"&&st.study_inner_rule!="direction-relative"&&st.study_inner_rule!="native-direction-soc"&&st.study_inner_rule!="native-reduced-direction-soc")||
           !std::isfinite(st.study_inner_kappa_fraction)||st.study_inner_kappa_fraction<=0||st.study_inner_kappa_fraction>=.5||
           !std::isfinite(st.study_inner_margin)||st.study_inner_margin<=0||
           !std::isfinite(st.study_inner_activity)||st.study_inner_activity<0||
           !std::isfinite(st.study_inner_feasibility)||st.study_inner_feasibility<0||
           !std::isfinite(st.study_inner_complementarity)||st.study_inner_complementarity<0||
           !std::isfinite(st.study_inner_barrier_decay)||st.study_inner_barrier_decay<=0||st.study_inner_barrier_decay>=1))
          throw std::runtime_error("invalid accepted-model relative parameters");
        if(st.study_controller=="certified"&&st.study_inner_rule!="barrier-relative"&&st.study_inner_kappa_fraction+st.study_inner_relative>=.5)
          throw std::runtime_error("model-relative requires kappa-fraction + relative < 0.5");
      }
      st.objective_lazy_convexification=true; // disables historical unconvexified probes
      st.majorant_curvature_corrections=0;
    }
    detail::factorial_control_semantics(st);
    if(st.objective_lazy_convexification&&st.dca_engine!="majorants"&&st.dca_engine!="auto")
      throw std::runtime_error("objective-convexification lazy requires dca-engine majorants");
    if(st.objective_max_atoms<0||st.objective_initial_atoms<0||st.objective_atom_batch<1||
       !std::isfinite(st.objective_rho_decay)||st.objective_rho_decay<=0||st.objective_rho_decay>=1||
       !std::isfinite(st.objective_rho_excess)||st.objective_rho_excess<0)
      throw std::runtime_error("invalid objective rho controls");
    if(st.dca_engine!="auto"&&st.dca_engine!="legacy"&&st.dca_engine!="filter"&&st.dca_engine!="majorants")
      throw std::runtime_error("dca-engine must be auto, legacy, filter, or majorants");
    if(st.dca_engine=="majorants") {
      const auto supplied=[&](const char* option){for(int i=1;i<argc;++i)if(std::string(argv[i])==option)return true;return false;};
      if(!supplied("--dca-dual-step"))st.dca_dual_step="common";
      if(!supplied("--dca-strict-finish"))st.dca_strict_finish=false;
      if(!supplied("--lazy-newton-lazy-constraints"))st.lazy_newton_lazy_constraints=false;
      if(st.dca_dual_step!="common")
        throw std::runtime_error("dca-engine majorants requires --dca-dual-step common");
      if(st.dca_aggregate_only || st.dca_aggregate || st.dca_krylov || st.dca_local_active || st.dca_strict_finish)
        throw std::runtime_error("dca-engine majorants forbids original-problem acceleration: use --no-dca-aggregate --dca-aggregate-only 0 --no-dca-krylov --no-dca-local-active --dca-strict-finish 0");
      if(st.lazy_newton_lazy_constraints)
        throw std::runtime_error("dca-engine majorants requires --no-lazy-newton-lazy-constraints (all inequality majorants are retained)");
      if(st.dca_filter_anderson)
        throw std::runtime_error("dca-engine majorants requires --no-dca-anderson");
    }
    if(st.dca_dual_step!="ipopt"&&st.dca_dual_step!="coupled"&&!((st.dca_engine=="majorants"||st.dca_engine=="auto")&&st.dca_dual_step=="common"))
      throw std::runtime_error("dca-dual-step must be ipopt, coupled, or common (majorants only)");
    if(st.dca_filter_curvature_backend!="same"&&st.dca_filter_curvature_backend!="cholesk"&&
       st.dca_filter_curvature_backend!="ldl"&&st.dca_filter_curvature_backend!="qdldl"&&
       st.dca_filter_curvature_backend!="mumps"&&st.dca_filter_curvature_backend!="native"&&st.dca_filter_curvature_backend!="ma27")
      throw std::runtime_error("dca-curvature-backend must be same, cholesk, ldl, qdldl, mumps, ma27, or native");
    if(st.dca_filter_curvature_backend!="same"&&!linear_backend_compiled(st.dca_filter_curvature_backend))
      throw std::runtime_error("DCA curvature backend '"+st.dca_filter_curvature_backend+
                               "' not compiled; available: "+compiled_linear_backends());
    if((st.dca_engine=="filter"||st.dca_engine=="majorants")&&st.outer_algorithm!="lazy-dca"&&st.outer_algorithm!="auto") throw std::runtime_error("dca-engine filter/majorants requires outer-algorithm lazy-dca");
    if(st.ln_engine!="legacy"&&st.ln_engine!="atomic") throw std::runtime_error("ln-engine must be legacy or atomic");
    if(st.live_display_level < -1 || st.live_display_level>4) throw std::runtime_error("display-level must be 0..4");
    if(st.majorant_equality_stabilization&&((st.dca_engine!="majorants"&&st.dca_engine!="auto")||
        st.native_augmented_repair||st.product_definition_presolve))
      throw std::runtime_error("equality stabilization requires original majorants, native-augmented-repair=0 and product-definition-presolve=0");
    const char* gr_merit_flag=std::getenv("LAZY_ORIGINAL_STABILIZED_MERIT");
    if(gr_merit_flag&&std::string(gr_merit_flag)!="0"&&std::string(gr_merit_flag)!="1")
      throw std::runtime_error("LAZY_ORIGINAL_STABILIZED_MERIT expects 0 or 1");
    const bool uses_ma27=st.linear_backend=="ma27" ||
        ((st.dca_engine=="filter"||st.dca_engine=="majorants") && (st.dca_filter_curvature_backend=="ma27" ||
          (st.dca_aggregate && st.dca_aggregate_backend=="ma27") ||
          (st.dca_local_active && st.dca_local_linear_backend=="ma27")));
    if(uses_ma27) {
      Settings probe_settings=st;probe_settings.linear_backend="ma27";
      std::string message;auto probe=make_symmetric_factorization(probe_settings,message);
      if(!probe)throw std::runtime_error(message);
      std::cout<<"[LINEAR-CONFIG] main="<<st.linear_backend
               <<" aggregate="<<st.dca_aggregate_backend
               <<" curvature="<<st.dca_filter_curvature_backend
               <<" hsl_library="<<(st.hsl_library.empty()?"environment/search":st.hsl_library)
               <<" pivot_tol="<<st.hsl_pivot_tolerance<<std::endl;
    }
    const auto read_start=std::chrono::steady_clock::now();
    QCQPProblem p = (st.outer_algorithm=="lazy-newton"||st.dca_engine=="auto"||st.dca_engine=="filter"||st.dca_engine=="majorants")&&compact_qcq_input?read_qcq_compact(problem_path):read_qcq(problem_path);
    if(!frozen_start_path.empty()) {
      if(!st.dca_endpoint_start_file.empty())throw std::runtime_error("frozen-start cannot be combined with primal-dual endpoint import");
      p.x0=detail::read_frozen_start(frozen_start_path,static_cast<std::size_t>(p.n));
      p.validate();
      std::cout<<"[FROZEN-START] primal_only=1 n="<<p.n<<" source="<<frozen_start_path<<std::endl;
    }
    const auto read_end=std::chrono::steady_clock::now();
    if(factorial_structure_preflight) {
      const detail::FactorialStructureInspection inspection(p);
      const auto shape_count=inspection.potential_shapes;
      std::ostringstream proof;proof<<std::setprecision(17)
          <<"{\"schema\":\"lazy-factorial-structure-preflight-v1\",\"inspection_only\":true,\"solver_invocations\":0"
          <<",\"n\":"<<p.n<<",\"equalities\":"<<p.equalities.size()<<",\"inequalities\":"<<p.inequalities.size()
          <<",\"native_signed_square_changed\":"<<(inspection.reduction_changed?"true":"false")
          <<",\"has_signed_square\":"<<(inspection.has_signed_square?"true":"false")
          <<",\"inspected_models\":"<<inspection.models<<",\"inspected_rank_models\":"<<inspection.rank_models<<",\"inspected_exact_phase_models\":"<<inspection.exact_phase_models
          <<",\"potential_signed_square_shape_count\":"<<shape_count
          <<",\"unresolved_structural_phase\":"<<(inspection.unresolved?"true":"false")
          <<",\"post_presolve_absence_proved\":"<<(inspection.eligible()?"true":"false")
          <<",\"eligible\":"<<(inspection.eligible()?"true":"false")
          <<",\"reason\":\""<<detail::factorial_json_string(inspection.reason)<<"\",\"links\":[";
      for(std::size_t i=0;i<inspection.original_links.size();++i){if(i)proof<<',';const auto& l=inspection.original_links[i];
        proof<<"{\"x\":"<<l.x<<",\"t\":"<<l.t<<",\"equality\":"<<l.e<<",\"scale\":"<<l.scale<<'}';}
      proof<<"]}";
      if(!json_path.empty()){std::ofstream file(json_path);file<<proof.str()<<'\n';if(!file)throw std::runtime_error("cannot write structure preflight proof");}
      std::cout<<proof.str()<<std::endl;return 0;
    }
    if(accepted_checkpoint) {
      if(st.outer_algorithm!="lazy-dca"||(st.dca_engine!="majorants"&&st.dca_engine!="auto")||json_path.empty())
        throw std::runtime_error("accepted-checkpoint requires --algorithm lazy-dca --dca-engine majorants and --json");
      const std::string checkpoint_path=json_path+".accepted.json";
      st.accepted_checkpoint_builtin=true;
      st.accepted_checkpoint_callback=[&p,checkpoint_path,warned=false](const QCQPProblem& model,Result& snapshot,
          const ConstraintPresolve* context,int iteration,double elapsed) mutable {
        try {
          if(&model!=&p&&!(context&&&context->original()==&p&&&context->reduced()==&model))
            throw std::runtime_error("checkpoint original-coordinate owner unavailable; private or unsupported transformed model");
          if(context)context->lift_checkpoint(snapshot);
          const auto bytes=detail::write_accepted_checkpoint(snapshot,checkpoint_path,iteration,elapsed);
          std::cout<<"[ACCEPTED-CHECKPOINT] written=1 iteration="<<iteration<<" bytes="<<bytes
              <<" audit_performed=0 success=0 full_original_coordinates=1"<<std::endl;
        }catch(const std::exception& error) {
          if(!warned)std::cerr<<"[ACCEPTED-CHECKPOINT] written=0 previous_file_preserved=1 reason="<<error.what()<<std::endl;
          warned=true;
        }
      };
    }
    detail::factorial_trace_begin(st);
    Solver solver(st);
    Result r = solver.solve(p);
    const auto solver_end=std::chrono::steady_clock::now();
    detail::factorial_trace_finish(st,json_path,r.outer_iterations,r.restoration_iterations,r.elapsed_seconds,r.lazy_newton.accepted_steps);
    r.cli_read_seconds=std::chrono::duration<double>(read_end-read_start).count();
    r.cli_arguments_seconds=std::chrono::duration<double>(read_start-process_start).count();
    r.cli_solver_call_seconds=std::chrono::duration<double>(solver_end-read_end).count();
    if(r.atomic_engine_used||r.dca_filter.used) {
      if(r.atomic_engine_used)print_newton_report(r);
      else if(st.live_display_level>=2){
        std::cout<<"\nTiming (seconds, exclusive)\n";
        for(std::size_t i=0;i<newton_cost_names.size();++i)std::cout<<"  "<<newton_cost_names[i]<<": "<<r.lazy_newton.cost_seconds[i]<<"\n";
        std::cout<<"  core_solve: "<<r.lazy_newton.core_solve_seconds<<"\n  solver_wall: "<<r.elapsed_seconds<<"\n  input_read: "<<r.cli_read_seconds<<"\n";
        std::cout<<"\nCore evaluations (individual functions)\n"
                 <<"  family             original values / gradients     majorant values / gradients\n";
        const char* names[]={"objective","equalities","inequalities","bound rows"};
        for(int i=0;i<4;++i){const auto& c=r.lazy_newton.calls[1][i];std::cout<<"  "<<std::left<<std::setw(17)<<names[i]<<std::right
          <<std::setw(12)<<c.values-c.model_values<<" / "<<std::setw(10)<<c.gradients-c.model_gradients
          <<std::setw(17)<<c.model_values<<" / "<<std::setw(10)<<c.model_gradients<<"\n";}

      }
      const auto summary_end=std::chrono::steady_clock::now();
      if(!json_path.empty()) {
        const auto write_start=std::chrono::steady_clock::now();write_result_json(r,p.name,json_path);
        const auto write_end=std::chrono::steady_clock::now();
        write_newton_cli_timing(json_path+".timing.json",r,
          std::chrono::duration<double>(summary_end-solver_end).count(),
          std::chrono::duration<double>(write_end-write_start).count(),
          std::chrono::duration<double>(write_end-process_start).count());
      }
      return (r.status==Status::Solved || r.status==Status::SolvedAcceptable ||
              r.status==Status::SolvedAcceptableOnFailure)?0:1;
    }


    std::cout << "====================================================================\n";
    std::cout << kSolverName << " " << kVersion << "\n";
    std::cout << "build                : " << kBuildId << "\n";
    std::cout << "Phase-I sparse Jv    : "
              << (restoration_exact_sparse_row_dots_build_compatible()
                      ? "runtime-guarded candidate"
                      : "disabled (fast-FP build)")
              << "\n";
    std::cout << "linear solver        : " << r.linear_solver_backend << "\n";
    std::cout << "compiled linear      : " << compiled_linear_backends() << "\n";
    std::cout << "METIS ordering       : " << (st.linear_metis ? "on" : "off") << "\n";
    std::cout << "persistent symbolic  : " << (st.kkt_persistent_symbolic_cache ? "on" : "off") << "\n";
    std::cout << "problem              : " << p.name << "\n";
    std::cout << "outer algorithm      : " << st.outer_algorithm;
    if (r.effective_outer_algorithm != st.outer_algorithm)
      std::cout << " -> " << r.effective_outer_algorithm;
    std::cout << "\n";
    std::cout << "convex subproblem    : " << st.inner_backend << "\n";
    std::cout << "LazyQP qp profile    : " << st.qp_option
              << " + AMRS=" << (st.qp_amrs ? "on" : "off")
              << " init=" << st.qp_init
              << " backend=" << st.qp_backend
              << " effective=" << st.linear_backend << "\n";
    std::cout << "LazyQP qp IPM        : Mehrotra=" << (st.qp_mehrotra ? "on" : "off")
              << " bound_push=" << st.qp_bound_push
              << " bp_adaptive=" << (st.qp_bp_adaptive ? "on" : "off")
              << " sigma_max=" << st.qp_sigma_max
              << " model_tol=" << st.qp_model_solved_tol << "\n";
    std::cout << "LazyQP qp DC controls: recorded; inactive for already-convex subproblems\n";
    std::cout << "LazyQP KKT policy    : " << st.lazyqp_kkt_form
              << " (auto threshold " << st.lazyqp_reduced_kkt_min_inequalities
              << ", hybrid retained/min-nnz "
              << st.lazyqp_hybrid_kkt_max_retained_inequalities << "/"
              << st.lazyqp_hybrid_kkt_min_row_nnz << ")\n";
    std::cout << "LazyQP low-rank form : " << st.lazyqp_low_rank_form
              << " (auto threshold " << st.lazyqp_low_rank_condense_threshold << ")\n";
    std::cout << "inner warm start     : "
              << (!st.inner_dual_warm_start ? "off" :
                  (st.inner_inequality_dual_warm_start ? "guarded-full-dual" : "equality-only"))
              << " (guard " << st.lazyqp_warm_start_guard_factor << ")\n";
    std::cout << "zero-dual continuation: "
              << (st.lazyqp_zero_dual_continuation ? "on" : "off")
              << " (mu cap " << st.lazyqp_zero_dual_mu_cap << ")\n";
    std::cout << "n / eq / ineq        : " << p.n << " / " << p.equalities.size() << " / " << p.inequalities.size() << "\n";
    std::cout << "status               : " << status_string(r.status) << "\n";
    std::cout << "message              : " << r.message << "\n";
    std::cout << std::setprecision(12);
    std::cout << "objective            : " << r.objective << "\n";
    std::cout << std::scientific;
    std::cout << "feasibility          : " << r.feasibility.total_inf << "\n";
    std::cout << "  equality           : " << r.feasibility.equality_inf << "\n";
    std::cout << "  inequality         : " << r.feasibility.inequality_inf << "\n";
    std::cout << "  bounds             : " << r.feasibility.bound_inf << "\n";
    std::cout << "model stationarity   : " << r.model_stationarity_inf << "\n";
    std::cout << "KKT-stop dual infeas.: " << r.dual_infeasibility_inf << "\n";
    std::cout << "KKT-stop complement. : " << r.complementarity_inf << "\n";
    std::cout << "KKT-stop scaled NLP  : " << r.scaled_nlp_error << "\n";
    std::cout << "KKT-stop s_d / s_c   : " << r.dual_error_scaling << " / " << r.complementarity_error_scaling << "\n";
    std::cout << "KKT-stop ||y||1/||z||1: " << r.constraint_multiplier_l1 << " / " << r.bound_slack_multiplier_l1 << "\n";
    std::cout << "KKT-stop dims dual/comp: " << r.ipopt_dual_multiplier_dimension << " / " << r.ipopt_complementarity_multiplier_dimension << "\n";
    std::cout << "KKT multiplier source: " << r.termination_multiplier_source << "\n";
    if (r.termination_multiplier_source == "fit" ||
        r.termination_multiplier_source == "fit-refined") {
      std::cout << "KKT-stop fit         : " << (r.termination_multiplier_fit_converged ? "converged" : "FAILED")
                << " sweeps=" << r.termination_multiplier_fit_sweeps
                << " pg=" << r.termination_multiplier_fit_projected_gradient_inf
                << " stat=" << r.termination_multiplier_fit_stationarity_inf << "\n";
      std::cout << "KKT-stop fit reg/tries: " << r.termination_multiplier_regularization
                << " / " << r.termination_multiplier_refinement_attempts << "\n";
    }
    std::cout << "KKT-stop terminal rescue: "
              << (r.termination_certificate_rescued ? "yes" : "no") << "\n";
    std::cout << "multiplier refit att/acc: "
              << r.multiplier_refit_attempts << " / "
              << r.multiplier_refit_successes << "\n";
    std::cout << "multiplier refit fallback: "
              << r.multiplier_refit_fallback_reuses << "\n";
    std::cout << "multiplier rowspace fact/solve: "
              << r.multiplier_row_space_factorizations << " / "
              << r.multiplier_row_space_solves << "\n";
    std::cout << "multiplier rowspace sym/reuse/fail: "
              << r.multiplier_row_space_symbolic_analyses << " / "
              << r.multiplier_row_space_symbolic_reuses << " / "
              << r.multiplier_row_space_failures << "\n";
    std::cout << "last multiplier refit stat: "
              << r.last_multiplier_refit_stationarity_inf << "\n";
    std::cout << "single-loop refit skip/fallback: "
              << r.dca_single_loop_refits_skipped << " / "
              << r.dca_single_loop_refit_fallbacks << "\n";
    std::cout << "single-loop bootstrap full fits: "
              << r.dca_single_loop_bootstrap_full_fits << "\n";
    std::cout << "single-loop full refit calls: "
              << r.dca_single_loop_full_refit_calls << "\n";
    std::cout << "single-loop retract skip/fallback: "
              << r.dca_single_loop_retractions_skipped << " / "
              << r.dca_single_loop_retraction_fallbacks << "\n";
    std::cout << "KKT-stop desired test: " << (r.ipopt_desired_convergence ? "PASS" : "FAIL") << "\n";
    std::cout << "KKT-stop tol/dual/cv/comp: " << st.stationarity_tolerance << " / " << st.dual_infeasibility_tolerance << " / " << st.feasibility_tolerance << " / " << st.complementarity_tolerance << "\n";
    std::cout << "KKT-stop s_max       : " << st.nlp_error_scaling_max << "\n";
    std::cout << "last upper step      : " << r.step_inf << "\n";
    std::cout << "last inner dual      : " << r.last_inner_dual_inf << "\n";
    std::cout << "last inner equality  : " << r.last_inner_equality_inf << "\n";
    std::cout << "last inner inequality: " << r.last_inner_inequality_inf << "\n";
    std::cout << "last inner primal res: " << r.last_inner_primal_residual_inf << "\n";
    std::cout << "last inner mu        : " << r.last_inner_complementarity << "\n";
    std::cout << "last inner scaled res: " << r.last_inner_scaled_model_residual_inf << "\n";
    std::cout << "inner mu fallback tol: " << st.inner_iteration_limit_mu_fallback << "\n";
    std::cout << "last inner fallback  : " << (r.last_inner_used_iteration_limit_fallback ? "yes" : "no") << "\n";
    std::cout << "KKT rho_x final/max  : " << r.final_kkt_rho_x << " / " << r.max_kkt_rho_x << "\n";
    std::cout << "KKT delta_d final/max: " << r.final_kkt_delta_d << " / " << r.max_kkt_delta_d << "\n";
    std::cout << "KKT max residual/berr: " << r.max_kkt_residual_inf << " / " << r.max_kkt_backward_error << "\n";
    std::cout << "KKT reg retry/refine : " << r.kkt_regularization_retries << " / " << r.kkt_refinement_steps << "\n";
    std::cout << "original-KKT GMRES att/acc/it: "
              << r.equality_qp_original_kkt_gmres_attempts << " / "
              << r.equality_qp_original_kkt_gmres_acceptances << " / "
              << r.equality_qp_original_kkt_gmres_iterations << "\n";
    std::cout << "prox regularization  : " << r.final_proximal_regularization << "\n";
    std::cout << "restoration it/restart: " << r.restoration_iterations << " / " << r.restoration_restarts << "\n";
    std::cout << "restoration solves   : " << r.restoration_linear_solves << "\n";
    std::cout << "restoration exact H  : " << r.restoration_exact_hessian_solves << "\n";
    std::cout << "restoration mf solves: " << r.restoration_matrix_free_solves << "\n";
    std::cout << "restoration row-space: " << r.restoration_row_space_solves << "\n";
    std::cout << "restoration harmonic att/acc/fact: "
              << r.restoration_harmonic_attempts << " / "
              << r.restoration_harmonic_acceptances << " / "
              << r.restoration_harmonic_factorizations << "\n";
    std::cout << "restoration tangent att/acc/components: "
              << r.restoration_tangent_activation_attempts << " / "
              << r.restoration_tangent_activation_acceptances << " / "
              << r.restoration_tangent_activation_components << "\n";
    std::cout << "restoration tangent step: "
              << r.restoration_tangent_activation_step_inf << "\n";
    std::cout << "restoration profile s: harmonic=" << r.restoration_harmonic_seconds
              << " tangent=" << r.restoration_tangent_activation_seconds
              << " legacy=" << r.restoration_legacy_gate_seconds
              << " exact=" << r.restoration_exact_seconds
              << " minimax=" << r.restoration_minimax_seconds
              << " restart=" << r.restoration_restart_seconds
              << " total=" << r.restoration_total_seconds << "\n";
    std::cout << "restoration harmonic graph/unknown: "
              << r.restoration_harmonic_graph_edges << " / "
              << r.restoration_harmonic_unknowns << "\n";
    std::cout << "restoration tangent rows/quad/cand: "
              << r.restoration_tangent_activation_gradient_rows << " / "
              << r.restoration_tangent_activation_line_quadratics << " / "
              << r.restoration_tangent_activation_candidates << "\n";
    std::cout << "restoration CG iters : " << r.restoration_cg_iterations << "\n";
    std::cout << "restoration exact ws/matvec: "
              << r.restoration_exact_workspace_growths << " / "
              << r.restoration_exact_matvecs << "\n";
    std::cout << "restoration OpenMP batches/teams: "
              << r.restoration_exact_parallel_row_dot_batches << " / "
              << r.restoration_exact_parallel_teams << "\n";
    std::cout << "restoration sparse Jv batches/check/fallback: "
              << r.restoration_exact_sparse_row_dot_batches << " / "
              << r.restoration_exact_sparse_row_dot_verifications << " / "
              << r.restoration_exact_sparse_row_dot_fallbacks << "\n";
    std::cout << "restoration LS backtr: " << r.restoration_line_search_backtracks << "\n";
    std::cout << "restoration stagnation: " << r.restoration_stagnation_events << "\n";
    std::cout << "restoration minimax QP: " << r.restoration_minimax_qp_solves << "\n";
    std::cout << "restoration minimax inner: " << r.restoration_minimax_inner_iterations << "\n";
    std::cout << "restoration minimax fact: " << r.restoration_minimax_factorizations << "\n";
    std::cout << "restoration minimax accept: " << r.restoration_minimax_accepted_steps << "\n";
    std::cout << "restoration minimax backtr: " << r.restoration_minimax_line_search_backtracks << "\n";
    std::cout << "restoration abs links: " << r.restoration_structured_abs_links << "\n";
    std::cout << "restoration eff. dim : " << r.restoration_effective_dimension << "\n";
    std::cout << "restoration theta    : " << r.restoration_initial_feasibility
              << " -> " << r.restoration_harmonic_feasibility
              << " -> " << r.restoration_tangent_activation_feasibility
              << " -> " << r.restoration_final_feasibility << "\n";
    std::cout << std::defaultfloat;
    std::cout << "outer / inner        : " << r.outer_iterations << " / " << r.inner_iterations << "\n";
    std::cout << "KKT factorizations   : " << r.kkt_factorizations << "\n";
    std::cout << "KKT symbolic analyses: " << r.kkt_symbolic_analyses << "\n";
    std::cout << "KKT symbolic reuse/inval: "
              << r.kkt_symbolic_reuses << " / "
              << r.kkt_symbolic_invalidations << "\n";
    std::cout << "KKT numeric scatter build/reuse/inval: "
              << r.kkt_numeric_scatter_builds << " / "
              << r.kkt_numeric_scatter_reuses << " / "
              << r.kkt_numeric_scatter_invalidations << "\n";
    std::cout << "LazyQP forcing try/ok/strict/outer-retry: "
              << r.lazyqp_inexact_forcing_attempts << " / "
              << r.lazyqp_inexact_forcing_acceptances << " / "
              << r.lazyqp_inexact_forcing_strict_fallbacks << " / "
              << r.lazyqp_inexact_forcing_outer_retries
              << " model-eval="
              << r.lazyqp_inexact_forcing_model_evaluations
              << " max-target=" << r.lazyqp_inexact_forcing_max_target << "\n";
    std::cout << "LazyQP model support request/effective/build: "
              << r.lazyqp_model_support_fastpath_requests << " / "
              << r.lazyqp_model_support_fastpath_effective_subproblems << " / "
              << r.lazyqp_model_support_builds
              << " eligible/fallback="
              << r.lazyqp_model_support_eligible_models << " / "
              << r.lazyqp_model_support_generic_fallback_models
              << " value/gradient-eval="
              << r.lazyqp_model_support_value_evaluations << " / "
              << r.lazyqp_model_support_gradient_evaluations << "\n";
    std::cout << "single-loop reduced bootstrap try/ok: "
              << r.dca_single_loop_reduced_bootstrap_attempts << " / "
              << r.dca_single_loop_reduced_bootstrap_acceptances << "\n";
    std::cout << std::scientific
              << "single-loop effective baseline prox: "
              << r.dca_single_loop_baseline_proximal_regularization
              << "\n" << std::defaultfloat;
    std::cout << "single-loop direct start try/ok/nonfinite/bound/equality/theta: "
              << r.dca_single_loop_direct_start_attempts << " / "
              << r.dca_single_loop_direct_start_acceptances << " / "
              << r.dca_single_loop_direct_start_rejections_nonfinite << " / "
              << r.dca_single_loop_direct_start_rejections_bounds << " / "
              << r.dca_single_loop_direct_start_rejections_equalities << " / "
              << r.dca_single_loop_direct_start_rejections_theta
              << " theta0="
              << r.dca_single_loop_direct_start_initial_theta
              << " violated/ws/filter="
              << r.dca_single_loop_direct_start_violated_inequalities << "/"
              << r.dca_single_loop_direct_start_working_set_rows << "/"
              << r.dca_single_loop_direct_start_filter_initializations << "\n";
    std::cout << "single-loop direct recovery try/ok/nonfinite/bound/cap/h/unmodeled/latch: "
              << r.dca_single_loop_direct_start_recovery_attempts << " / "
              << r.dca_single_loop_direct_start_recovery_acceptances << " / "
              << r.dca_single_loop_direct_start_recovery_rejections_nonfinite << " / "
              << r.dca_single_loop_direct_start_recovery_rejections_bounds << " / "
              << r.dca_single_loop_direct_start_recovery_rejections_cap << " / "
              << r.dca_single_loop_direct_start_recovery_rejections_contraction << " / "
              << r.dca_single_loop_direct_start_recovery_rejections_unmodeled << " / "
              << r.dca_single_loop_direct_start_strict_latches
              << " ws-add="
              << r.dca_single_loop_direct_start_recovery_working_set_additions
              << " max-theta="
              << r.dca_single_loop_direct_start_recovery_max_theta << "\n";
    std::cout << "inner warm accept/full/reject: "
              << r.inner_dual_warm_start_acceptances << " / "
              << r.inner_full_dual_warm_start_acceptances << " / "
              << r.inner_dual_warm_start_rejections << "\n";
    std::cout << "LazyQP zero-dual continuation try/ok/fallback/repairs: "
              << r.lazyqp_zero_dual_continuation_attempts << " / "
              << r.lazyqp_zero_dual_continuation_acceptances << " / "
              << r.lazyqp_zero_dual_continuation_fallbacks << " / "
              << r.lazyqp_zero_dual_continuation_repairs << "\n";
    std::cout << "LazyQP zero-dual provenance mapped/direct: "
              << r.lazyqp_zero_dual_continuation_mapped_attempts << " / "
              << r.lazyqp_zero_dual_continuation_direct_attempts << "\n";
    std::cout << "working-set primal transports: "
              << r.working_set_primal_transports << "\n";
    std::cout << "LazyQP reduced/full KKT fact: "
              << r.lazyqp_reduced_kkt_factorizations << " / "
              << r.lazyqp_full_kkt_factorizations << "\n";
    std::cout << "LazyQP reduced KKT fallback: " << r.lazyqp_reduced_kkt_fallbacks << "\n";
    std::cout << "LazyQP hybrid KKT try/fact/fallback: "
              << r.lazyqp_hybrid_kkt_attempts << " / "
              << r.lazyqp_hybrid_kkt_factorizations << " / "
              << r.lazyqp_hybrid_kkt_fallbacks << "\n";
    std::cout << "LazyQP hybrid retained total/max: "
              << r.lazyqp_hybrid_kkt_retained_rows_total << " / "
              << r.lazyqp_hybrid_kkt_retained_rows_max << "\n";
    std::cout << "LazyQP hybrid Schur avoided/border triplets: "
              << r.lazyqp_hybrid_kkt_outer_product_triplets_avoided << " / "
              << r.lazyqp_hybrid_kkt_border_triplets_emitted << "\n";
    std::cout << "SOCP formulation try/build/reject/audit-fail: "
              << r.socp_formulation_attempts << " / "
              << r.socp_formulation_builds << " / "
              << r.socp_formulation_rejections << " / "
              << r.socp_formulation_audit_failures << "\n";
    std::cout << "SOCP cones objective/quadratic/linear, max affine eq: "
              << r.socp_objective_rotated_cones << " / "
              << r.socp_inequality_rotated_cones << " / "
              << r.socp_linear_nonnegative_cones << " / "
              << r.socp_max_affine_equalities << "\n";
    std::cout << "SOCP max explicit lift dim/condensed Newton/max cone violation: "
              << r.socp_max_explicit_lift_cone_dimension << " / "
              << r.socp_condensed_newton_iterations << " / "
              << r.socp_max_rotated_cone_violation << "\n";
    std::cout << "LazyQP KKT triplet layout build/reuse/inval: "
              << r.lazyqp_kkt_triplet_layout_builds << " / "
              << r.lazyqp_kkt_triplet_layout_reuses << " / "
              << r.lazyqp_kkt_triplet_layout_invalidations << "\n";
    std::cout << "LazyQP numeric reuse try/ok/fallback/krylov: "
              << r.lazyqp_numeric_kkt_reuse_attempts << " / "
              << r.lazyqp_numeric_kkt_reuse_acceptances << " / "
              << r.lazyqp_numeric_kkt_reuse_fallbacks << " / "
              << r.lazyqp_numeric_kkt_reuse_iterations << "\n";
    std::cout << "LazyQP max KKT dim/full dim: "
              << r.lazyqp_max_kkt_dimension << " / "
              << r.lazyqp_max_full_kkt_dimension << "\n";
    std::cout << "LazyQP affine elim attempt/accept/fallback: "
              << r.lazyqp_affine_elimination_attempts << " / "
              << r.lazyqp_affine_elimination_acceptances << " / "
              << r.lazyqp_affine_elimination_fallbacks << "\n";
    std::cout << "LazyQP affine elim vars/eq, reduced n/eq: "
              << r.lazyqp_affine_eliminated_variables << " / "
              << r.lazyqp_affine_eliminated_equalities << " / "
              << r.lazyqp_affine_reduced_variables << " / "
              << r.lazyqp_affine_reduced_equalities << "\n";
    std::cout << "inner fallback accepts: " << r.inner_iteration_limit_fallback_acceptances << "\n";
    std::cout << "line-search backtracks: " << r.line_search_backtracks << "\n";
    std::cout << "Armijo rejections   : " << r.armijo_rejections << "\n";
    std::cout << "Lazy-DCA equality model : " << r.lazy_dca_equality_model << "\n";
    std::cout << "condensed curvature    : "
              << (st.dca_condensed_tangent_curvature
                  ? (st.dca_condensed_block_exact_curvature
                         ? "certified-block-exact/single-normal"
                     : st.dca_condensed_projected_multi_ritz_inertia
                         ? "projected-multi-Ritz+inertia/historical-fallback"
                         : "ambient-large/projected-fallback")
                  : "off")
              << " rank-cap=" << st.dca_condensed_max_tangent_rank << "\n";
    std::cout << "condensed adaptive normal / alpha predictor: "
              << (st.dca_condensed_adaptive_normal_refinement ? "on" : "off")
              << " / "
              << (st.dca_condensed_alpha_predictor ? "on" : "off") << "\n";
    std::cout << "condensed post-retract WS expansion/cap: "
              << (st.dca_condensed_post_retraction_working_set_expansion
                    ? "on" : "off")
              << " / "
              << st.dca_condensed_post_retraction_working_set_max_expansions
              << "\n";
    std::cout << "condensed modeled-side filter/limit: "
              << (st.dca_condensed_modeled_side_filter ? "on" : "off")
              << " / " << std::scientific
              << st.dca_condensed_modeled_side_filter_limit
              << " recovery-margin="
              << st.dca_condensed_modeled_side_recovery_margin
              << "\n" << std::defaultfloat;
    std::cout << "Lazy-DCA elastic sub/acc: " << r.lazy_dca_elastic_subproblems << " / " << r.lazy_dca_elastic_accepted_steps << "\n";
    std::cout << "Lazy-DCA elastic last eta: " << r.lazy_dca_elastic_last_eta << "\n";
    std::cout << "Lazy-DCA linearized equality steps: " << r.lazy_dca_linearized_equality_steps << "\n";
    std::cout << "Lazy-DCA equality f/h steps: "
              << r.lazy_dca_equality_objective_steps << " / "
              << r.lazy_dca_equality_feasibility_steps << "\n";
    std::cout << std::scientific << "Lazy-DCA last equality alpha/funnel: "
              << r.lazy_dca_last_equality_step_alpha << " / "
              << r.lazy_dca_final_equality_funnel << "\n" << std::defaultfloat;
    std::cout << "Lazy-DCA eq retract att/acc/rej: "
              << r.lazy_dca_equality_retraction_attempts << " / "
              << r.lazy_dca_equality_retraction_acceptances << " / "
              << r.lazy_dca_equality_retraction_rejections << "\n";
    std::cout << std::scientific << "Lazy-DCA max eq retract: "
              << r.lazy_dca_max_equality_retraction_inf << "\n" << std::defaultfloat;
    std::cout << "condensed retract att/acc/rej: "
              << r.lazy_dca_condensed_retraction_attempts << " / "
              << r.lazy_dca_condensed_retraction_acceptances << " / "
              << r.lazy_dca_condensed_retraction_rejections << "\n";
    std::cout << "condensed globally accepted steps: "
              << r.lazy_dca_condensed_accepted_steps << "\n";
    std::cout << "condensed continuation acc/att: "
              << r.lazy_dca_condensed_continuation_acceptances << " / "
              << r.lazy_dca_condensed_continuation_attempts << "\n";
    std::cout << "condensed Newton/Krylov/backtr: "
              << r.lazy_dca_condensed_newton_iterations << " / "
              << r.lazy_dca_condensed_krylov_iterations << " / "
              << r.lazy_dca_condensed_backtracks << "\n";
    std::cout << "condensed affine-normal reuses: "
              << r.lazy_dca_condensed_affine_normal_map_reuses << "\n";
    std::cout << "condensed Newton endpoint reuses: "
              << r.lazy_dca_condensed_newton_endpoint_reuses << "\n";
    std::cout << "condensed quadratic LS reuse/build/check: "
              << r.lazy_dca_condensed_quadratic_linesearch_reuses << " / "
              << r.lazy_dca_condensed_quadratic_linesearch_predictor_builds << " / "
              << r.lazy_dca_condensed_quadratic_linesearch_exact_checks << "\n";
    std::cout << "condensed quadratic LS early/rows-skipped: "
              << r.lazy_dca_condensed_quadratic_linesearch_early_rejections << " / "
              << r.lazy_dca_condensed_quadratic_linesearch_rows_skipped << "\n";
    std::cout << "condensed normal fact/solve: "
              << r.lazy_dca_condensed_factorizations << " / "
              << r.lazy_dca_condensed_normal_solves << "\n";
    std::cout << "condensed normal solves geometry/retraction: "
              << r.lazy_dca_condensed_geometry_normal_solves << " / "
              << r.lazy_dca_condensed_retraction_normal_solves << "\n";
    std::cout << "condensed bound-face fact/max-active: "
              << r.lazy_dca_condensed_bound_face_factorizations << " / "
              << r.lazy_dca_condensed_max_active_bound_coordinates << "\n";
    std::cout << "condensed symbolic ana/reuse/inval: "
              << r.lazy_dca_condensed_symbolic_analyses << " / "
              << r.lazy_dca_condensed_symbolic_reuses << " / "
              << r.lazy_dca_condensed_symbolic_invalidations << "\n";
    std::cout << "condensed cached reg att/hit: "
              << r.lazy_dca_condensed_cached_regularization_attempts << " / "
              << r.lazy_dca_condensed_cached_regularization_hits << "\n";
    std::cout << "condensed sparse eq evals / parallel batches: "
              << r.lazy_dca_condensed_sparse_equality_evaluations << " / "
              << r.lazy_dca_condensed_parallel_equality_batches << "\n";
    std::cout << "condensed normal refinement steps: "
              << r.lazy_dca_condensed_normal_refinement_steps << "\n";
    std::cout << "condensed normal refinements geometry/retraction: "
              << r.lazy_dca_condensed_geometry_normal_refinement_steps << " / "
              << r.lazy_dca_condensed_retraction_normal_refinement_steps << "\n";
    std::cout << "condensed normal refine solves 0/1/2/3+: "
              << r.lazy_dca_condensed_normal_refinement_zero_solves << " / "
              << r.lazy_dca_condensed_normal_refinement_one_solves << " / "
              << r.lazy_dca_condensed_normal_refinement_two_solves << " / "
              << r.lazy_dca_condensed_normal_refinement_three_plus_solves << "\n";
    std::cout << "condensed normal relaxed/strict solves: "
              << r.lazy_dca_condensed_relaxed_normal_solves << " / "
              << r.lazy_dca_condensed_strict_normal_solves << "\n";
    std::cout << "condensed relaxed above-strict / strict retries: "
              << r.lazy_dca_condensed_relaxed_accepts_above_strict_tolerance << " / "
              << r.lazy_dca_condensed_strict_retry_retractions << "\n";
    std::cout << std::scientific
              << "condensed normal berr initial max/mean: "
              << r.lazy_dca_condensed_max_normal_initial_backward_error << " / "
              << r.lazy_dca_condensed_mean_normal_initial_backward_error << "\n"
              << "condensed normal berr final max/mean: "
              << r.lazy_dca_condensed_max_normal_backward_error << " / "
              << r.lazy_dca_condensed_mean_normal_final_backward_error << "\n"
              << std::defaultfloat;
    std::cout << "condensed GMRES workspace growths: "
              << r.lazy_dca_condensed_gmres_workspace_growths << "\n";
    std::cout << "condensed zero-rhs solve skips: "
              << r.lazy_dca_condensed_zero_rhs_solve_skips << "\n";
    std::cout << "condensed alpha predictor uses/resets/skipped/retries: "
              << r.lazy_dca_condensed_alpha_predictor_uses << " / "
              << r.lazy_dca_condensed_alpha_predictor_full_step_resets << " / "
              << r.lazy_dca_condensed_alpha_predictor_leading_retractions_skipped << " / "
              << r.lazy_dca_condensed_alpha_predictor_historical_retries << "\n";
    std::cout << "condensed outer extrapolation try/accept/fallback/streak/max-alpha: "
              << r.lazy_dca_condensed_outer_extrapolation_attempts << " / "
              << r.lazy_dca_condensed_outer_extrapolation_acceptances << " / "
              << r.lazy_dca_condensed_outer_extrapolation_historical_fallbacks << " / "
              << r.lazy_dca_condensed_outer_extrapolation_full_step_streak_max << " / "
              << r.lazy_dca_condensed_outer_extrapolation_alpha_max << "\n";
    std::cout << "condensed outer rejects bound/side/funnel/filter/objective: "
              << r.lazy_dca_condensed_outer_reject_bound << " / "
              << r.lazy_dca_condensed_outer_reject_side << " / "
              << r.lazy_dca_condensed_outer_reject_funnel << " / "
              << r.lazy_dca_condensed_outer_reject_filter << " / "
              << r.lazy_dca_condensed_outer_reject_objective << "\n";
    std::cout << "condensed post-retract offenders modeled/omitted, WS expand/fallback: "
              << r.lazy_dca_condensed_post_retraction_modeled_offenders << " / "
              << r.lazy_dca_condensed_post_retraction_omitted_offenders << " / "
              << r.lazy_dca_condensed_post_retraction_working_set_expansions << " / "
              << r.lazy_dca_condensed_post_retraction_expansion_fallbacks << "\n";
    std::cout << "condensed modeled-side filter try/accept/recovery/fallback/margin-rows/max: "
              << r.lazy_dca_condensed_modeled_side_filter_attempts << " / "
              << r.lazy_dca_condensed_modeled_side_filter_acceptances << " / "
              << r.lazy_dca_condensed_modeled_side_filter_recovery_acceptances << " / "
              << r.lazy_dca_condensed_modeled_side_filter_fallbacks << " / "
              << r.lazy_dca_condensed_modeled_side_filter_recovery_margin_rows << " / "
              << std::scientific
              << r.lazy_dca_condensed_modeled_side_filter_max_violation
              << "\n" << std::defaultfloat;
    std::cout << "condensed flow-lift Schur detected/dim/fact/fallback: "
              << r.lazy_dca_condensed_flow_lift_schur_detected_variables << " / "
              << r.lazy_dca_condensed_flow_lift_schur_factor_dimension << " / "
              << r.lazy_dca_condensed_flow_lift_schur_factorizations << " / "
              << r.lazy_dca_condensed_flow_lift_schur_fallbacks << "\n";
    std::cout << "condensed tangent calls/rank: "
              << r.lazy_dca_condensed_tangent_curvature_calls << " / "
              << r.lazy_dca_condensed_tangent_rank_max << "\n";
    std::cout << "condensed block-exact att/acc, blocks/active/max-block: "
              << r.lazy_dca_condensed_block_exact_attempts << " / "
              << r.lazy_dca_condensed_block_exact_acceptances << " / "
              << r.lazy_dca_condensed_block_exact_blocks_max << " / "
              << r.lazy_dca_condensed_block_exact_active_dimension_max << " / "
              << r.lazy_dca_condensed_block_exact_max_block_dimension << "\n";
    std::cout << "condensed block partial spectrum att/acc/fallback: "
              << r.lazy_dca_condensed_block_partial_spectrum_attempts << " / "
              << r.lazy_dca_condensed_block_partial_spectrum_acceptances << " / "
              << r.lazy_dca_condensed_block_partial_spectrum_fallbacks << "\n";
    std::cout << "condensed multi-Ritz batches/fallbacks: "
              << r.lazy_dca_condensed_projected_multi_ritz_batches << " / "
              << r.lazy_dca_condensed_projected_multi_ritz_fallbacks << "\n";
    std::cout << "condensed tangent inertia att/acc/shift: "
              << r.lazy_dca_condensed_tangent_inertia_attempts << " / "
              << r.lazy_dca_condensed_tangent_inertia_acceptances << " / "
              << r.lazy_dca_condensed_tangent_inertia_shift_updates << "\n";
    std::cout << "condensed scalar safeguard updates max: "
              << r.lazy_dca_condensed_scalar_safeguard_updates_max << "\n";
    std::cout << std::scientific
              << "condensed tangent lambda: "
              << r.lazy_dca_condensed_initial_tangent_min_eigenvalue << " -> "
              << r.lazy_dca_condensed_final_tangent_min_eigenvalue << "\n"
              << "condensed shift/normal-reg: "
              << r.lazy_dca_condensed_tangent_shift_max << " / "
              << r.lazy_dca_condensed_normal_regularization << "\n"
              << "condensed inertia max added shift: "
              << r.lazy_dca_condensed_tangent_inertia_shift_max << "\n"
              << "condensed residual/max-corr/ratio: "
              << r.lazy_dca_condensed_last_residual_inf << " / "
              << r.lazy_dca_condensed_max_correction_inf << " / "
              << r.lazy_dca_condensed_max_correction_ratio << "\n"
              << std::defaultfloat;
    std::cout << "Lazy-DCA corrector attempt/accept/reject: "
              << r.lazy_dca_corrector_attempts << " / "
              << r.lazy_dca_corrector_acceptances << " / "
              << r.lazy_dca_corrector_rejections << "\n";
    std::cout << "Lazy-DCA corrector inner/cold retry: "
              << r.lazy_dca_corrector_inner_iterations << " / "
              << r.lazy_dca_corrector_cold_retries << "\n";
    std::cout << "Lazy-DCA iterated SOC attempt/restore/reject: "
              << r.lazy_dca_iterated_soc_attempts << " / "
              << r.lazy_dca_iterated_soc_acceptances << " / "
              << r.lazy_dca_iterated_soc_rejections << "\n";
    std::cout << "Lazy-DCA warm suppress/skip: "
              << r.lazy_dca_warm_start_suppressions << " / "
              << r.lazy_dca_warm_start_skips << "\n";
    std::cout << "Lazy-DCA inner retry/cold/prox: "
              << r.lazy_dca_inner_retries << " / "
              << r.lazy_dca_cold_retries << " / "
              << r.lazy_dca_proximal_retries << "\n";
    std::cout << "Lazy-DCA step-control predict/retry: "
              << r.lazy_dca_step_control_predictions << " / "
              << r.lazy_dca_step_control_retries << "\n";
    std::cout << std::scientific << "Lazy-DCA raw direction/target: "
              << r.lazy_dca_last_raw_direction_inf << " (max "
              << r.lazy_dca_max_raw_direction_inf << ") / "
              << r.lazy_dca_last_step_target_inf << "\n" << std::defaultfloat;
    std::cout << std::scientific << "Lazy-DCA max model proximal: "
              << r.lazy_dca_max_model_proximal << "\n" << std::defaultfloat;
    std::cout << "trust-region subproblems: " << r.trust_region_subproblems << "\n";
    std::cout << "trust-region accept/reject: " << r.trust_region_accepted_steps << " / " << r.trust_region_rejected_steps << "\n";
    std::cout << "trust-region contract/expand: " << r.trust_region_contractions << " / " << r.trust_region_expansions << "\n";
    std::cout << "trust-region feasibility steps: " << r.trust_region_feasibility_steps << "\n";
    std::cout << std::scientific << "trust-region final radius: " << r.final_trust_region_radius << "\n";
    std::cout << "trust-region last rho/pred/ared: " << r.last_trust_region_ratio << " / " << r.last_trust_region_predicted_reduction << " / " << r.last_trust_region_actual_reduction << "\n";
    std::cout << "trust-region penalty       : " << r.last_trust_region_penalty << "\n";
    std::cout << "trust-region merit pred/ared: " << r.last_trust_region_merit_predicted_reduction << " / " << r.last_trust_region_merit_actual_reduction << "\n" << std::defaultfloat;
    std::cout << "SOC attempt/accept/reject: " << r.soc_attempts << " / " << r.soc_accepted_steps << " / " << r.soc_rejected_steps << "\n";
    std::cout << "SOC inner/factorizations: " << r.soc_inner_iterations << " / " << r.soc_factorizations << "\n";
    std::cout << "curvature calls      : " << r.curvature_calls << "\n";
    std::cout << "rank objective/ineq  : " << r.objective_rank << " / " << r.inequality_total_rank << "\n";
    std::cout << "rank caps obj/constraint: " << st.max_objective_rank << " / " << st.max_rank_per_quadratic << "\n";
    std::cout << "DCA extra proximal   : " << st.dca_extra_proximal_regularization << "\n";
    std::cout << "constraint model     : "
              << (r.effective_outer_algorithm=="lazy-dca"
                  ? (r.lazy_dca_equality_model=="original-nonlinear-equalities"
                      ? "DC function models; original nonlinear equalities"
                      : (r.lazy_dca_equality_model=="elastic-split"
                      ? "DC majorants; elastic-split nonlinear equalities"
                      : (r.lazy_dca_equality_model=="condensed-retraction"
                          ? "DC inequality majorants; signed tangent Hessian and exact condensed equality retraction"
                          : "DC majorants for inequalities; SQP-linearized equalities")))
                  : (st.phase2_linearized_inequalities ? "auto-linearized" : "quadratic-majorant"))
              << "\n";
    std::cout << "active Jacobian rank : " << r.active_jacobian_rank << "\n";
    std::cout << "tangent dimension    : " << r.tangent_dimension << "\n";
    std::cout << "reduced negative eigs: " << r.reduced_negative_eigenvalues << "\n";
    std::cout << std::scientific << "reduced min eigenvalue: " << r.reduced_min_eigenvalue << "\n" << std::defaultfloat;
    std::cout << "terminal reduced attempts/accept/reject: " << r.terminal_kkt_attempts << " / " << r.terminal_kkt_accepted_steps << " / " << r.terminal_kkt_rejected_steps << "\n";
    std::cout << "terminal active stability/cutoff: " << r.terminal_kkt_active_stability << " / " << r.terminal_kkt_multiplier_cutoff << "\n";
    std::cout << "terminal legacy pivot drops: " << r.terminal_kkt_active_set_drops << "\n";
    std::cout << "active inequalities  : " << r.active_inequalities << "\n";
    std::cout << "working-set add/drop : " << r.working_set_additions << " / " << r.working_set_drops << "\n";
    std::cout << "affine box lazy WS   : "
              << (st.dca_affine_box_bounded_lazy ? "on" : "off") << "\n";
    std::cout << "working-set drop patience: " << st.working_set_drop_patience << "\n";
    if (r.effective_outer_algorithm == "lazy-newton") {
      std::cout << "lazy-newton iterations/factorizations/symbolic: " << r.lazy_newton_iterations << " / "
                << r.lazy_newton_factorizations << " / " << r.lazy_newton_symbolic_analyses << "\n";
      std::cout << "lazy-newton inertia/dual-reg/prox shifts: " << r.lazy_newton_inertia_corrections << " / "
                << r.lazy_newton_dual_regularizations << " / " << r.lazy_newton_proximal_shifts << "\n";
      std::cout << "lazy-newton curvature exact/majorant/fallback: " << r.lazy_newton_exact_curvature_steps << " / "
                << r.lazy_newton_majorant_curvature_steps << " / " << r.lazy_newton_curvature_fallbacks << "\n";
      std::cout << "lazy-newton quadratic corr att/acc: " << r.lazy_newton_quadratic_corrections << " / "
                << r.lazy_newton_quadratic_correction_acceptances << "\n";
      std::cout << "lazy-newton lazy cuts/backtracks/barrier updates: " << r.lazy_newton_lazy_cuts << " / "
                << r.lazy_newton_line_search_backtracks << " / " << r.lazy_newton_barrier_updates << "\n";
      std::cout << "lazy-newton working set final/max: " << r.lazy_newton_working_set_final << " / "
                << r.lazy_newton_working_set_max << " (condensed/augmented rows max "
                << r.lazy_newton_condensed_rows_max << " / " << r.lazy_newton_augmented_rows_max << ")\n";
      std::cout << "lazy-newton KKT dim/nnz/factor nnz: " << r.lazy_newton_kkt_dimension << " / "
                << r.lazy_newton_kkt_nnz << " / " << r.lazy_newton_factor_nnz << "\n";
      std::cout << std::scientific << "lazy-newton final mu/objective scale/max shift: " << r.lazy_newton_final_mu
                << " / " << r.lazy_newton_objective_scale << " / " << r.lazy_newton_max_delta_x << "\n";
      std::cout << "lazy-newton seconds setup/structure/assembly/factor/eval/ls/cert: "
                << r.lazy_newton_setup_seconds << " / " << r.lazy_newton_structure_seconds << " / "
                << r.lazy_newton_assembly_seconds << " / " << r.lazy_newton_factorization_seconds << " / "
                << r.lazy_newton_evaluation_seconds << " / " << r.lazy_newton_line_search_seconds << " / "
                << r.lazy_newton_certificate_seconds << "\n" << std::defaultfloat;
    }
    std::cout << "elapsed (s)          : " << r.elapsed_seconds << "\n";
    std::cout << "====================================================================\n";

    if (!json_path.empty()) write_result_json(r, p.name, json_path);
    return (r.status == Status::Solved || r.status == Status::SolvedAcceptable ||
            r.status == Status::SolvedAcceptableOnFailure) ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "[fatal] " << e.what() << "\n";
    return 2;
  }
}
