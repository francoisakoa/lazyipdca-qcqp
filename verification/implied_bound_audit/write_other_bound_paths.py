#!/usr/bin/env python3
"""Read-only source/counter audit: no optimizer or kit mutation."""
from pathlib import Path
import collections
import hashlib
import json

BASE = Path(__file__).resolve().parents[2]
ROOTS = {
    'mumps': BASE / 'campaign_20261109/recovered/article/reproduction/qualified_native_source',
    'ma27': BASE / 'factorial_20261109/staging/LAZYQCQP_IPOPT_ALIGNE_RAFFINEMENT_20261109_R1/native/reference_source',
}
OBS = BASE / 'campaign_20261109/input/OBSERVATIONS.jsonl'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def evidence(path, start, end):
    return {'path': path, 'lines': [start, end], 'sha256': {k:sha(v/path) for k,v in ROOTS.items()}}

entries = [
 {'name':'Frozen-reader absent-side sentinel canonicalization',
  'effect':'Maps either infinite sign on a lower side to -infinity and either infinite sign on an upper side to +infinity. Finiteness determines bound-side presence; finite bounds and raw start coordinates are untouched.',
  'gate':'QCQ reader, before model validation and all presolve comparisons; applies independently of policy.',
  'evidence':[evidence('src/io.cpp',107,111),evidence('src/qcq_bound_sentinels.hpp',8,19)],
  'asu_limit':'This cannot remove a finite ASU bound and is distinct from implied-bound suppression.'},
 {'name':'R3 exact equality/closed-zero-face phase before standard selection',
  'effect':'ClosedZeroFace may set an existing lower-zero variable upper bound to zero and discard exactly redundant rows. It tightens/fixes a proved face; it does not drop variable bound sides.',
  'gate':'standard_r10_policy=auto, convexification_study=off, study_controller=r3, majorants/lazy-dca, no rank context/private phase/checkpoint/product-definition/scaling, shared restoration budget; coefficient proof and time/source caps must pass.',
  'evidence':[evidence('src/solver.cpp',4072,4098), evidence('src/closed_zero_face.hpp',70,111)],
  'asu_limit':'Original quadratic inequalities exclude standard-R10 selection, but do not by themselves exclude this preceding exact-face inspection. No retained per-case face flag in the projected observations establishes whether inspection applied on ASU.'},
 {'name':'Global exact affine-face variable elimination',
  'effect':'Removes coordinates proved zero by exact affine-combination/singleton/product closure certificates. Copies every bound of each retained coordinate. This is not general implied-bound-side suppression.',
  'gate':'original_exact_affine_faces or LAZY_EXACT_AFFINE_FACE_RESEARCH, not consumed; majorants/lazy-dca; rank context/checkpoints/global product substitution/augmented repair prohibited.',
  'evidence':[evidence('src/solver.cpp',4692,4750), evidence('src/exact_affine_face_presolve.hpp',62,140)],
  'asu_limit':'Absent frozen coefficients and full native flags/logs prevent a generic proof that no zero-face certificate exists. standard_auto applies only if original inequalities all affine; ASU has quadratic inequalities so that automatic route cannot enable this branch.'},
 {'name':'Native signed-square auxiliary elimination before rank presolve',
  'effect':'Eliminates t from t^2-x^2=0 with t>=0, introducing signed-square functions, and removes its variable/bound coordinates. Bounds of retained variables are copied.',
  'gate':'No equality stabilization, input has no prior signed-square terms, lazy-dca/majorants, convexification_study=off. Eligible auxiliary needs original lb=0, ub=+infinity, exact raw t0=abs(x0), plus compatible component incidence.',
  'evidence':[evidence('src/solver.cpp',4860,4913),evidence('src/native_signed_square_reduction.hpp',18,73)],
  'asu_limit':'The observation projection has no signed-square/auxiliary-elimination trace. Zero reported eliminated variables and unchanged rank dimensions alone are not a proof that this preceding transformation never ran.'},
 {'name':'Common-factor exact coordinate substitution',
  'effect':'For u_j=t*x_j, sets non-pivot lb/ub to infinite but introduces the corresponding two affine inequalities L_j*t<=u_j<=U_j*t. This replaces a box representation; it does not remove its feasible-set restriction.',
  'gate':'R3 auto/off/r3, majorants or auto, no rank context/private phase/product/scaling/custom callback/signed-square; positive-start nonnegative-domain pivot; all other bounds finite; every nonzero Hessian term must be off-diagonal and incident to the pivot.',
  'evidence':[evidence('src/solver.cpp',4269,4312), evidence('src/common_factor_substitution.hpp',17,70)],
  'asu_limit':'All 503 retained MA27 ASU R3 observations report common_factor_substitution=false. In addition, a nonzero diagonal equality term would reject this substitution, but coefficients are absent: do not assert the detailed ASU equality structure from names alone.'},
 {'name':'Positive-square epigraph coordinate substitution',
  'effect':'Substitution u=t^2 changes lb_t to lb_t^2 and maps the raw start. No bound side is removed. Other coordinates must already have no finite bounds.',
  'gate':'R3 auto/off/r3, allowed engines, affine objective with one positive linear term, epigraph variable lb>=0 and ub=+infinity, all equalities affine and independent of t, specified diagonal quadratic inequality structure.',
  'evidence':[evidence('src/solver.cpp',4320,4346), evidence('src/positive_square_epigraph_substitution.hpp',17,56)],
  'asu_limit':'ASU has quadratic equalities (from metadata), violating the all-affine equality requirement; no fresh coefficient audit performed.'},
 {'name':'Explicit product-definition auxiliary elimination',
  'effect':'Eliminates variables defined by unit products. Rewrites each eliminated finite bound as a quadratic inequality, except when exact rational factor-interval proof establishes it redundant. All retained-factor bounds remain.',
  'gate':'product_definition_presolve=true; top-level original project termination; majorants/lazy-dca or permitted atomic route. Default is false.',
  'evidence':[evidence('src/solver.cpp',4780,4787), evidence('src/product_definition_presolve.hpp',30,93), evidence('include/lazyipdca_qcqp/settings.hpp',1051,1052)],
  'asu_limit':'No explicit enabling option found in retained ASU R3 commands; default false. This is independent of rank implied-bound removal if enabled in a separately declared profile.'},
 {'name':'Free-epigraph leaf elimination',
  'effect':'Eliminates an isolated negative-diagonal inequality leaf only if its original box is [0,+infinity), retaining all remaining variable bounds.',
  'gate':'free_epigraph_presolve && majorant_adaptive_globalization && convexification_study!=off && lazy-dca/majorants && no rank context/signed-square.',
  'evidence':[evidence('src/solver.cpp',4793,4796), evidence('src/free_epigraph_presolve.cpp',64,92)],
  'asu_limit':'Published R3 uses convexification_study=off, so this entry is gated off regardless of source coefficients. Zero preparation/postsolve counters are consistent, but not the sole proof.'},
 {'name':'Exact dyadic power-of-two scaling',
  'effect':'Maps finite box endpoints by exact powers of two, preserving whether each side is finite. No side is suppressed.',
  'gate':'power2_scaling=true, or internally invoked after common-factor/positive-square substitution; no rank/checkpoint context for explicit use.',
  'evidence':[evidence('src/solver.cpp',4788,4791), evidence('src/power2_scaling.hpp',35,49)],
  'asu_limit':'Explicit scaling option defaults false; internal invocations are limited to the preceding separately gated substitutions.'},
 {'name':'Fixed-bound representation inside auxiliary model',
  'effect':'A finite interval whose width is within fixed_bound_tolerance is represented as an equality; its two bound inequality rows are not additionally constructed. This retains a fixing condition rather than deleting feasible-set control.',
  'gate':'Both sides finite and abs(ub-lb)<=fixed_bound_tolerance.',
  'evidence':[evidence('src/solver.cpp',1238,1268)],
  'asu_limit':'All 503 MA27 ASU R3 observations report lazy_newton_fixed_variables=0. No evidence of this case on the retained ASU runs.'},
 {'name':'Condensed-bound promotion for linear-system precision',
  'effect':'Moves an already active inequality/bound from condensed Gram representation to an explicit KKT border. Appends algebraic matrix entries; preserves the QCQP, inequalities, active flag, slacks and barrier equations. This is not bound removal or working-set deactivation.',
  'gate':'Condensation/direction precision recovery after factor or expanded-residual difficulty; ratio/roundoff selection. Stats distinguish promoted bounds by inequality_kind!=0.',
  'evidence':[evidence('src/atomic_newton.cpp',1430,1448), evidence('src/atomic_newton.cpp',1450,1504)],
  'asu_limit':'The nonzero lazy_newton_promoted_bounds counters are evidence of equivalent numerical representation changes only; they cannot be counted as removed original bound sides.'},
 {'name':'Affine quadratic rewrite and equality-row redundancy',
  'effect':'Exact polynomial rewrite keeps equality constraints and the original variables/bounds. Exact redundant-equality removal may discard proved equalities, but preserves all variable bounds. Neither is a second implied-bound-removal implementation.',
  'gate':'R3 standard auto/off/r3 majorants/lazy-dca top-level with stated no-context/research restrictions.',
  'evidence':[evidence('src/solver.cpp',4210,4267), evidence('src/affine_face_quadratic_rewrite.hpp',42,85), evidence('src/exact_redundant_equalities.hpp',20,145)],
  'asu_limit':'No coefficient-level rewrite incidence or equality-redundancy audit repeated; unchanged variable dimensions would not alone exclude these row operations.'},
]

counts={}; n=0
for line in OBS.open():
 r=json.loads(line)
 if r.get('method')!='R3' or 'asu' not in str(r.get('case_id','')):continue
 n+=1
 for k,v in r['internal_work'].items():
  if k in ['common_factor_substitution','common_factor_variable','lazy_newton_fixed_variables','lazy_newton_promoted_bounds','lazy_newton_eliminated_variables','free_epigraph_preparation_seconds','free_epigraph_postsolve_seconds','free_epigraph_postsolve_valid','standard_auto_rank_presolve_effective','standard_auto_rank_presolve_requested','standard_auto_free_epigraph_presolve_skipped']:
   counts.setdefault(k,collections.Counter())[json.dumps(v)]+=1

out={'schema':'other-native-bound-representations-static-audit-v1','optimizer_invocations':0,
     'sources':{k:str(v.relative_to(BASE)) for k,v in ROOTS.items()},'paths':entries,
     'observed_ma27_asu_r3':{'records':n,'counter_values':{k:dict(v) for k,v in counts.items()},'observations_sha256':sha(OBS)},
     'limitations':['No full frozen coefficient model, endpoint, or per-iteration solver log was available for this audit. Metadata and exported counters are not substituted for coefficient-level exclusion proofs.',
       'Source-level transformed-model gates and recorded transformation flags can exclude specific paths. Absence of an exported counter cannot prove absence of the corresponding native phase.',
       'MUMPS source mechanisms are inspected, but the available MUMPS publication projections do not contain the detailed ASU counters measured here for MA27.']}
(Path(__file__).parent/'OTHER_BOUND_PATHS.json').write_text(json.dumps(out,indent=2)+'\n')
print('written',len(entries),'paths;',n,'MA27 R3 ASU observations')
