"""Read-only local evidence audit; does not invoke any optimizer."""
from pathlib import Path
import collections, csv, hashlib, importlib.util, json

OUT = Path(__file__).resolve().parent
MUMPS = Path('campaign_20261109/recovered/article/reproduction/qualified_native_source')
MA27 = Path('factorial_20261109/staging/LAZYQCQP_IPOPT_ALIGNE_RAFFINEMENT_20261109_R1/native/reference_source')
KIT = Path('factorial_20261109/kit/LAZYQCQP_REFINEMENT_ABLATION_20261109_R1')
EXP = KIT/'native/experiment_source'
PLAN = Path('campaign_20261109/input/evidence/PLAN_COMPACT.json')
CONTEXT = KIT/'native/profile_evidence/PUBLISHED_CONTEXT.json'
OBS = Path('campaign_20261109/input/OBSERVATIONS.jsonl')
OLD_PLAN = KIT/'ma27_base/campaign/metadata/PLAN_3103.json'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def validate(root, manifest):
    missing, mismatch, identical = [], [], 0
    for n, h in manifest.items():
        rel = n.split('/source/', 1)[1] if '/source/' in n else n
        p = root/rel
        if not p.is_file(): missing.append(rel)
        elif sha(p) != h: mismatch.append(rel)
        else: identical += 1
    return dict(manifest_files=len(manifest), identical=identical,
                missing=missing, mismatch=mismatch)

def option(args, name):
    return args[args.index(name)+1] if name in args else None

ctx = json.loads(CONTEXT.read_text())
plan = json.loads(PLAN.read_text())['plan_header']
historical_profiles = {}
for backend, profiles in [('ma27',plan['profiles']['profiles']),
                          ('mumps',ctx['profiles']['profiles'])]:
    historical_profiles[backend] = {
        method: {flag: option(profiles[method], flag) for flag in
                 ['--convexification-study','--dca-engine','--dca-rank-presolve',
                  '--dca-rank-implied-bounds','--standard-r10-policy',
                  '--product-definition-presolve','--free-epigraph-presolve']}
        for method in ['R3','HH','HS','SH','SS']}

rows, observed_other = [], collections.defaultdict(collections.Counter)
actual_profile_flags = collections.Counter()
for line in OBS.open():
    r = json.loads(line)
    if r['dataset'] != 'beam_structural' or r['method'] != 'R3': continue
    w = r['internal_work']
    actual_profile_flags[tuple(option(r['solver_resources']['command'], flag) for flag in
        ['--convexification-study','--dca-engine','--dca-rank-presolve',
         '--dca-rank-implied-bounds','--product-definition-presolve'])] += 1
    rows.append(dict(case_id=r['case_id'], backend=r['backend'],
        classification=r['classification'], input_sha256=r['input_sha256'],
        start_sha256=r['start_sha256'],
        rank_presolve_enabled=w.get('rank_presolve_enabled'),
        rank_presolve_applied=w.get('rank_presolve_applied'),
        implied_bound_sides=w.get('rank_presolve_implied_bound_sides'),
        original_variables=w.get('rank_presolve_original_variables'),
        reduced_variables=w.get('rank_presolve_reduced_variables'),
        original_equalities=w.get('rank_presolve_original_equalities'),
        reduced_equalities=w.get('rank_presolve_reduced_equalities'),
        original_inequalities=w.get('rank_presolve_original_inequalities'),
        reduced_inequalities=w.get('rank_presolve_reduced_inequalities'),
        start_preserved=w.get('rank_presolve_initial_point_preserved'),
        common_factor_substitution=w.get('common_factor_substitution'),
        eliminated_variables=w.get('lazy_newton_eliminated_variables'),
        linear_bound_representation_promotions=w.get('lazy_newton_promoted_bounds'),
        source_record_sha256=hashlib.sha256(line.encode()).hexdigest()))
    for k in ['common_factor_substitution', 'common_factor_variable',
              'lazy_newton_eliminated_variables','lazy_newton_promoted_bounds',
              'free_epigraph_preparation_seconds','free_epigraph_postsolve_seconds']:
        observed_other[k][str(w.get(k))] += 1

csv_path = OUT/'ASU_R3_PREPROCESSING_MA27.csv'
with csv_path.open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)

spec = importlib.util.spec_from_file_location('bound_audit_profiles',KIT/'runtime/PROFILES_EXPERIENCE.py')
factory = importlib.util.module_from_spec(spec); spec.loader.exec_module(factory)
cells = []
for name, cell in factory.CELLS.items():
    args = factory.make_command(name,'{binary}','{input}','{endpoint}','{trace}',
                               '{hsl_library}' if cell['backend']=='ma27' else None)
    # Exact evaluation of the inspected Settings predicates for these profiles.
    active = bool(option(args,'--factorial-inequalities'))
    guards = option(args,'--factorial-common-guards')
    glob = option(args,'--factorial-globalization')
    master = option(args,'--dca-rank-implied-bounds') == '1'
    engine = option(args,'--dca-engine')
    hs_guards = (guards=='hs') if guards else (glob=='hs')
    r3_controls = (guards=='r3') if guards else (glob=='r3')
    cells.append(dict(cell=name, **cell,
        source_evaluation_only=True, runtime_observations_audited=False,
        rank_presolve_requested=option(args,'--dca-rank-presolve'),
        implied_bounds_master=master, dca_engine=engine,
        convexification_study=option(args,'--convexification-study'),
        factorial_active=active, factorial_hs_guards=hs_guards,
        factorial_r3_controls=r3_controls,
        quadratic_equality_implied_bounds_gate=bool(master and engine=='majorants' and hs_guards),
        affine_singleton_equality_implied_bounds_gate=bool(master and engine=='majorants' and r3_controls),
        radial_inequality_implied_bounds_gate=master,
        command_template=args))

meta = json.loads(OLD_PLAN.read_text())['cases']
if isinstance(meta, dict): meta = list(meta.values())
asu_ids = {r['case_id'] for r in rows}
meta_rows = [c for c in meta if c.get('id') in asu_ids]
dimension_summary = collections.Counter(tuple(c['dimensions'].get(k) for k in
    ['n','quadratic_equalities','linear_equalities','quadratic_inequalities',
     'fixed_variables','lower_bounds','upper_bounds']) for c in meta_rows)

refs = [
    (MUMPS/'src/rank_presolve.cpp', [925,931], 'Historical gates: quadratic equalities only HH/H0/HS/SH; affine singleton equalities only off/R3.'),
    (MUMPS/'src/rank_presolve.cpp', [57,65], 'Exact implication sign test a*b*b+c >= 0.'),
    (MUMPS/'src/rank_presolve.cpp', [370,385], 'All modes: bounds implied by a retained diagonal radial inequality.'),
    (MUMPS/'src/rank_presolve.cpp', [393,415], 'Affine singleton equality bound omission, separate gate.'),
    (MUMPS/'src/rank_presolve.cpp', [418,445], 'Quadratic radial equality bound omission; retained equation unchanged.'),
    (MUMPS/'src/main.cpp', [536,540], 'Existing CLI option is only master 0|1, not a force-on quadratic-equality switch.'),
    (MUMPS/'include/lazyipdca_qcqp/settings.hpp', [16,21], 'Existing master bool default and rank flag.'),
    (EXP/'src/rank_presolve.cpp', [925,932], 'Factorial quadratic-equality omission depends on factorial_hs_guards; affine omission on factorial_r3_controls.'),
    (EXP/'include/lazyipdca_qcqp/settings.hpp', [33,49], 'Explicit common guards override globalization in the two Settings predicates.'),
    (KIT/'runtime/PROFILES_EXPERIENCE.py', [11,12], 'Eight production cells all declare common_guards r3.'),
    (KIT/'runtime/PROFILES_EXPERIENCE.py', [34,44], 'Production command factory sets exact objective factors and common R3 guards on HH base.'),
]

report = dict(schema='bound-removal-r3-factorial-audit-v1',
    optimizer_invocations=0, remote_accesses=0, running_kit_modified=False,
    conclusion='Historical R3 and all eight configured refinement cells keep the quadratic-equality-implied bound sides. The archived MA27 R3 observations show no rank-presolve bound removal on all 503 ASU cases. A new independent switch is required to force only that removal in R3.',
    historical_sources={'mumps':str(MUMPS),'ma27':str(MA27)},
    historical_source_validation={'mumps':validate(MUMPS,ctx['source_sha256']),
                                  'ma27':validate(MA27,plan['source_sha256'])},
    historical_profiles=historical_profiles,
    historical_gate={'quadratic_equality':'master && majorants && study in {HH,H0,HS,SH}',
        'affine_singleton_equality':'master && majorants && study == off',
        'radial_quadratic_inequality':'master',
        'H0_note':'Present in rank gate but not admitted by the archived public CLI mode enumeration.',
        'outer_gate':'rank presolve must itself be enabled/effective; standard-auto and exact native dispatch run before this function.'},
    observed_r3_ma27_asu={'records':len(rows),'cases':len(asu_ids),
        'actual_command_flag_fields':['convexification_study','dca_engine','dca_rank_presolve','dca_rank_implied_bounds','product_definition_presolve'],
        'actual_command_flags':[dict(values=list(k),cases=v) for k,v in actual_profile_flags.items()],
        'rank_presolve_applied':dict(collections.Counter(str(r['rank_presolve_applied']) for r in rows)),
        'implied_bound_sides':dict(collections.Counter(str(r['implied_bound_sides']) for r in rows)),
        'original_and_reduced_variables_equal':all(r['original_variables']==r['reduced_variables'] for r in rows),
        'initial_start_preserved':all(r['start_preserved'] is True for r in rows),
        'cases_with_32_variables':[r['case_id'] for r in rows if r['original_variables']==32],
        'other_exported_counters':{k:dict(v) for k,v in observed_other.items()},
        'promotion_semantics':'lazy_newton_promoted_bounds changes condensed-vs-bordered linear representation of existing rows, not the bound set or barrier equations.'},
    mumps_runtime_limit='The retained MUMPS projection does not export detailed rank-presolve counters; no separately observed 503-case MUMPS bound-removal count is claimed.',
    asu_metadata={'records':len(meta_rows),'dimension_fields':['n','quadratic_equalities','linear_equalities','quadratic_inequalities','fixed_variables','lower_bounds','upper_bounds'],
        'dimension_counts':[dict(values=list(k),cases=v) for k,v in sorted(dimension_summary.items())],
        'metadata_bound_to_input_sha256':all(c.get('dimensions_provenance',{}).get('input_sha256')==c.get('input_sha256') for c in meta_rows),
        'metadata_inputs_match_observed_r3':all(c['input_sha256']=={r['case_id']:r['input_sha256'] for r in rows}[c['id']] for c in meta_rows)},
    geometry_limit={'frozen_coefficients_present_locally':False,
        'proved':'For each omitted side, a retained sign-oriented diagonal quadratic equality globally implies the side at the exact binary coefficients.',
        'not_proved':'The counts and metadata do not show whether all sides are strict at any feasible point, active at a particular endpoint, or forced active by the equalities.',
        'distinction':'A two-coordinate circle can have strict box sides simultaneously; a singleton equality at a box radius can force a bound side active. Redundancy alone does not decide strict interior.',
        'causal_limit':'No quantitative attribution of objective differences to the barrier terms follows from this audit.'},
    factorial_cells=cells,
    factorial_scope='These eight rows evaluate local source/profile predicates, not the state or results of a remote running campaign.',
    independent_control_proposal={'not_implemented_in_this_audit':True,
        'setting':'std::string dca_rank_quadratic_equality_implied_bounds = "profile";',
        'cli':'--dca-rank-quadratic-equality-implied-bounds profile|keep|remove',
        'minimal_logic':'equality_bounds = master && majorants && (choice==remove || (choice==profile && historical_or_factorial_gate));',
        'preserved':'keep disables only radial quadratic-equality omission; remove enables it independently of study/controller/guards. The master bool, affine-equality omission, inequality omission and all other exact reductions retain their existing meanings.',
        'required_evidence':'Emit requested/effective mode and per-side original variable/equality proof, input hash, start binding and complete original postsolve certificate; test profile mode reproduces historical behavior before production.'},
    source_evidence=[dict(path=str(p),lines=lines,finding=finding,sha256=sha(p)) for p,lines,finding in refs],
    other_exact_native_bound_paths={'path':str(OUT/'OTHER_BOUND_PATHS.json'),
        'sha256':sha(OUT/'OTHER_BOUND_PATHS.json'),
        'note':'This companion enumerates twelve related transformations and their evidence limits, including non-removing KKT representation promotions.'},
    inputs={str(p):sha(p) for p in [OBS,PLAN,CONTEXT,OLD_PLAN,KIT/'runtime/PROFILES_EXPERIENCE.py',KIT/'runtime/REFERENCE_PROFILES.json']},
    output_csv={'path':str(csv_path),'rows':len(rows),'sha256':sha(csv_path)})
report['manuscript_scope_review'] = {
    'correction_required':'Replace affine singleton elimination with bound removal implied by singleton affine equalities; variables and equalities are retained.',
    'avoid_broad_claim':'The common affine presolve is not universally enabled. original_exact_affine_faces defaults false and standard-auto selects it only in a coefficient-restricted domain; the specifically proved common enabled gate is affine_equality_bounds.',
    'bound_retention_scope':'Only retention under the radial quadratic-equality implication rule is fixed across all eight cells; do not equate it with absence of all other exact transformations.'}
note = OUT/'BOUND_REMOVAL_R3_FACTORIAL_NOTE.md'
if note.is_file(): report['note'] = dict(path=str(note),sha256=sha(note))
(OUT/'BOUND_REMOVAL_R3_FACTORIAL.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(source_validation=report['historical_source_validation'],r3_cases=len(rows),
    quadratic_equality_gate_by_cell={c['cell']:c['quadratic_equality_implied_bounds_gate'] for c in cells},
    dimension_counts=report['asu_metadata']['dimension_counts']),indent=2))
