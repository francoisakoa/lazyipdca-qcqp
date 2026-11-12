#!/usr/bin/env python3
"""Create an outcome-blind draft protocol and immutable corpus/50-case bindings."""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,datetime
P=argparse.ArgumentParser();P.add_argument('--corpus',type=Path,required=True);P.add_argument('--out',type=Path,required=True);a=P.parse_args();a.out.mkdir(parents=True,exist_ok=True)
with gzip.open(a.corpus,'rt')as f:corpus=json.load(f)
cs=sorted((c for c in corpus['cases']if c['regime']in('I','M')),key=lambda c:c['case_id']);assert len(cs)==1834 and collections.Counter(c['regime']for c in cs)==collections.Counter(dict(I=1037,M=797))
seed='LazyQCQP-refinement-globalization-common-guards-qualification50-v2-20261109'
h=lambda t:hashlib.sha256(t.encode()).hexdigest()
# Select one per nonempty regime/dataset stratum, then outcome-blind round robin.
cap=10000
all_strata=collections.defaultdict(list)
strata=collections.defaultdict(list)
for c in cs:
 all_strata[c['regime'],c['dataset']].append(c)
 if c['dimensions']['n']<=cap:strata[c['regime'],c['dataset']].append(c)
for key in strata:strata[key].sort(key=lambda c:(c['dimensions']['n'],c['dimensions'].get('constraints_total',0),h(seed+'|'+c['case_id'])))
selected=[]
for regime in ('I','M'):
 keys=sorted((k for k in strata if k[0]==regime),key=lambda k:h(seed+'|'+str(k)));picked=[];depth=0
 while len(picked)<25:
  for key in keys:
   if depth<len(strata[key]):picked.append(strata[key][depth])
   if len(picked)==25:break
  depth+=1
 selected.extend(picked)
bindings=[dict(case_id=c['case_id'],regime=c['regime'],dataset=c['dataset'],family=c['family'],input_sha256=c['input_sha256'],start_sha256=c['start_binding']['start_sha256'])for c in cs]
pre=dict(schema='adaptive-refinement-globalization-common-guards-preregistration-v2',study_status='specified_before_qualification',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scientific_question='Does adaptive updating of otherwise identically initialized inequality corrections improve original-KKT certification, independently of globalization and safeguards?',
 labels=dict(FROZEN_GR3=dict(correction_evolution='frozen',globalization='r3'),FROZEN_GHS=dict(correction_evolution='frozen',globalization='hs'),REFINED_GR3=dict(correction_evolution='adaptive',globalization='r3'),REFINED_GHS=dict(correction_evolution='adaptive',globalization='hs')),
 backends=['ma27','mumps'],
 inputs=dict(corpus_file_sha256=hashlib.sha256(a.corpus.read_bytes()).hexdigest(),cases=1834,regimes=dict(I=1037,M=797),case_bindings=bindings,selection='All frozen I/M cases; no exclusion according to solver outcome, family, feasibility, or objective'),
 production=dict(cells_per_backend=4,backends=2,tasks=14672,new_Ipopt_runs=0,rerun_every_cell=True,legacy_results_not_substituted=True,objective_exact=True,seconds_per_task=7200,max_outer=5000,shared_restoration=600,threads_per_task=1,single_binary_for_both_backends=True,jobs=22,cpus=list(range(22)),continuous_shared_queue=True,backend_is_block_not_independent_replicate=True),
 invariants=dict(common_representation='original_qcqp',representation_rule='SignedSquareReduction is disabled uniformly in all production cells; exact affine/rank maps that remain QCQP are declared and fully counted; legacy controls are unchanged',correction_initialization='shared_hybrid_initialization_v1',guard_policy='common_r3_guards_v1',objective_correction_E0='identically_zero',
  signed_quadratic_equality_curvature='identical_exact_treatment',expanded_Newton_accuracy_audit='same_required_test',inertia_and_factorization_safeguards='same_required_rules',
  exact_original_KKT_evaluator='same_complete_original_problem_primal_dual_complementarity_at_most_1e-6',external_optimizer='none',external_endpoint_warm_start=False,
  numerical_recovery_and_curvature_probes='same_policy_in_every_cell',
  structural_guard_policy_explanation='The R3 structural/recovery safeguard eligibility vector is fixed in every production cell, independently of globalization. Mandatory accuracy, inertia and original-acceptance safeguards remain enabled. The common vector is frozen and hash-bound before production.',
  frozen_semantics='Build exactly the shared initial inequality corrections and never change them; update their anchoring only.',
  refined_semantics='Use exactly the same initial corrections; permit only the preregistered adaptive updates to inequality corrections.'),
 primary=dict(scope='I+M pooled instances within each backend, not pooled backend trials',comparison='REFINED_GHS minus FROZEN_GHS',minimum_net_certificates_per_backend=20,
  discordance_ratio_at_least=2,maximum_single_collection_fraction_of_net_gain=.5,family_macro_effect_strictly_positive=True,
  general_positive_claim='All four criteria must hold separately on both native backends, robust to unresolved technical outcomes. Per-backend findings are also reported.',
  negative_interpretation='Failure of the stated rule is not a proof that adaptation is useless; report observed effects and interactions.',
  significance_claim='No statistical significance is inferred from these engineering decision thresholds; the corpus is fixed and backend records are paired repeats.'),
 contrasts=dict(refinement_average='0.5*((REFINED_GR3-FROZEN_GR3)+(REFINED_GHS-FROZEN_GHS))',globalization_average='0.5*((FROZEN_GHS-FROZEN_GR3)+(REFINED_GHS-REFINED_GR3))',interaction='(REFINED_GHS-FROZEN_GHS)-(REFINED_GR3-FROZEN_GR3)'),
 secondary=dict(regimes=['I','M'],objectives='Common strict certificates; symmetric gap (fA-fB)/max(1,abs(fA),abs(fB)); thresholds1e-3 and1e-2',time='Median of casewise process ratios on common strict certificates with both process times>1s; N always shown; no startup subtraction',
  composition='By collection and family, equal-family and equal-collection effects, largest-collection omission, fixed partitions from frozen corpus',
  native_work='Outer/restoration counts, accepted updates, refinement attempts/accepted updates, factorization attempts/failures/solves, numerical CPU/wall/process/audit clocks separately',
  direction_provenance='Final accepted direction origin after SOC/retraction/arc/other processing; base model and descendant origin remain distinct',
  realised_adaptation='Fraction of refined runs with at least one accepted correction update, number of changed components, correction ranks/norms and initial/final digests'),
 qualification=dict(required_before_production=True,case_count=50,legacy_runs=700,legacy_runs_by_backend=dict(ma27=400,mumps=300),installed_ma27_reference_runs=100,installed_ma27_reference_required=True,synthetic_controls=6,synthetic_cell_backend_runs=48,total_gate_runs=748,selection_seed=seed,strata='25 I and25 M; round-robin regime x dataset, n then constraints then seeded identifier hash',source_strata_count=len(all_strata),eligible_strata_count=len(strata),qualification_dimension_cap=cap,selection_uses_outcomes=False,selected_case_ids=[c['case_id']for c in selected],
  legacy_controls='R3 and HH remain their original complete profiles, including HH objective treatment, and are run only as controls. Exact-objective production cells are not relabelled public R3/HH.',
  checks=['The currently installed MA27 baseline, the fresh reference, passive observer and experiment-with-research-flags-off agree in final vectors, work counters and original certification; all observed fresh trajectories agree. MUMPS uses the three fresh binaries from the same current source, never the older numerically different MUMPS implementation.',
   'On six deterministic structural controls, all four factorial cells have the same exact objective, shared initial component correction digest and invariant safeguard digest. Production rechecks these invariants for every paired case.',
   'Frozen cells have zero accepted correction updates; refined cells differ only by admissible adaptive updating.',
   'All native phases share one7200s/5000outer/600restoration budget; direction traces record only committed final directions as accepted.',
   'No production task launches if any required gate is absent, fails, or remains unresolved.']),
 outcome_definition=dict(article_binary='success if full original KKT and data/start/backend/budget proofs qualify; otherwise failure to certify, all planned cases retained',
  raw_states='Keep audited nonstrict, censored/interrupted, missing or invalid proofs separate internally; never call an absent proof a numerical failure.',
  sensitivity='For unresolved technical/missing outcomes, paired contrast bounds allow either binary numerical outcome; a confirmatory positive claim must survive every completion.'),
 trace_record_required=['case_id','cell','backend','run_id','preregistration_sha256','binary_sha256','input_sha256','start_sha256','objective_exact','budget_verified','trace_success','strict_original','primal','dual','complementarity','factorial_proof'],
 factorial_proof_required=['correction_evolution','globalization','guard_policy','correction_initialization','initial_model_state','initial_inequality_model_sha256','initialization_calls','invariant_options_sha256','accepted_refinements'],
 sealing=dict(required_before_production=True,required_bindings=['source_tree_sha256','binary_sha256_by_backend','source_separation_audit_sha256','qualification50_receipt_sha256','direction_trace_schema_sha256','common_guard_vector_sha256'],
  procedure='Keep this study contract unchanged. Fill concrete source/build/qualification bindings in the separate production registration, and set its registration_status=sealed_before_production only after all gates pass and before any production outcome. Qualification binds this static study SHA; production observations bind the final registration SHA.'),
 optimizer_invocations=0)
coverage=[dict(regime=k[0],dataset=k[1],source_cases=len(v),eligible_cases=len(strata.get(k,[])),selected_cases=sum(c['regime']==k[0]and c['dataset']==k[1]for c in selected),eligible_fraction=len(strata.get(k,[]))/len(v),selected_fraction=sum(c['regime']==k[0]and c['dataset']==k[1]for c in selected)/len(v))for k,v in sorted(all_strata.items())]
pre['common_guards']='r3';pre['common_representation']='original_qcqp';pre['objective_exact']=True;pre['inequality_policy']='hybrid'
pre['configurations']={f'{cell}_{backend}':dict(cell=cell,backend=backend,refinement_policy=v['correction_evolution'],globalization_policy=v['globalization'],objective_exact=True,inequality_policy='hybrid',common_guards='r3',common_representation='original_qcqp')for backend in pre['backends']for cell,v in pre['labels'].items()}
pre['parity_gate']=dict(qualification_only_dimension_cap=cap,production_cases_excluded=0,stratum_coverage=coverage,sample=[c['case_id']for c in selected],sample_bindings=[dict(case_id=c['case_id'],regime=c['regime'],dataset=c['dataset'],input_sha256=c['input_sha256'],start_sha256=c['start_binding']['start_sha256'])for c in selected],legacy_runs=700,legacy_runs_by_backend=dict(ma27=400,mumps=300),installed_ma27_reference_runs=100,installed_ma27_reference_required=True,synthetic_controls=6,synthetic_runs=48,total_runs=748,selection_uses_outcomes=False)
(a.out/'STUDY_CONTRACT.json').write_text(json.dumps(pre,indent=2,ensure_ascii=False)+'\n')
study_sha=hashlib.sha256((a.out/'STUDY_CONTRACT.json').read_bytes()).hexdigest()
seal=dict(schema='adaptive-refinement-production-registration-v2',registration_status='draft_not_sealed',study_contract_file='STUDY_CONTRACT.json',study_contract_sha256=study_sha,bindings=dict(source_tree_sha256=None,binary_sha256_by_backend=dict(ma27=None,mumps=None),source_separation_audit_sha256=None,qualification50_receipt_sha256=None,direction_trace_schema_sha256=None,common_guard_vector_sha256=None),note='Seal only after the source/build/parity/synthetic gates pass. Gate receipts bind the static study SHA; production observations bind this final seal SHA, avoiding a hash cycle.')
(a.out/'PRE_ENREGISTREMENT_RAFFINEMENT.json').write_text(json.dumps(seal,indent=2,ensure_ascii=False)+'\n')
(a.out/'QUALIFICATION50_CASES.json').write_text(json.dumps(dict(schema='outcome-blind-qualification50-v2',selection_seed=seed,study_contract_sha256=study_sha,corpus_file_sha256=pre['inputs']['corpus_file_sha256'],cases=[dict(case_id=c['case_id'],regime=c['regime'],dataset=c['dataset'],input_sha256=c['input_sha256'],start_sha256=c['start_binding']['start_sha256'],n=c['dimensions']['n'])for c in selected],selection_uses_outcomes=False,qualification_dimension_cap=cap,stratum_coverage=coverage),indent=2)+'\n')
print(json.dumps(dict(draft_seal=True,study_contract_sha256=study_sha,cases=len(cs),production_tasks=14672,qualification_cases=len(selected),qualification_regimes=dict(collections.Counter(c['regime']for c in selected)),qualification_strata=len(strata),source_strata=len(all_strata),qualification_n_max=max(c['dimensions']['n']for c in selected)),indent=2))
