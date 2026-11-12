#!/usr/bin/env python3
"""Controller fixtures only: these checks are not a native corpus preflight."""
import importlib.util,json,sys,tempfile,types,gzip
from pathlib import Path
import VERIFIER_COUPLAGE as gate

tests=[]
with tempfile.TemporaryDirectory(prefix='coupling-gate-controller-') as directory:
    root=Path(directory);inp=root/'fixture.qcq';inp.write_bytes(b'controller fixture, never passed to an optimizer\n')
    source=root/'source';source.mkdir();(source/'fixed.hpp').write_text('int passive_fixture;\n')
    binary=root/'native_fixture'
    binary.write_text('#!'+sys.executable+'\nimport json,sys\nfrom pathlib import Path\n'
      'p=Path(sys.argv[sys.argv.index("--json")+1]);p.write_text((Path(__file__).parent/"native_payload.json").read_text())\n')
    binary.chmod(0o755)
    payload={'schema':'lazy-factorial-structure-preflight-v1','inspection_only':True,'solver_invocations':0,
      'n':3,'equalities':1,'inequalities':1,'native_signed_square_changed':False,'has_signed_square':False,
      'inspected_models':3,'inspected_rank_models':2,'inspected_exact_phase_models':0,
      'potential_signed_square_shape_count':0,'unresolved_structural_phase':False,
      'post_presolve_absence_proved':True,'eligible':True,'reason':'controller fixture only'}
    control=root/'native_payload.json'
    case={'case_id':'generic_fixture','dataset':'fixture','regime':'M','input_sha256':gate.sha(inp),
      'start_binding':{'start_sha256':'fixture-start'},
      'dimensions':{'n':3,'equalities_total':1,'inequalities_total':1}}
    runner=types.SimpleNamespace(verify_input=lambda cfg,c:(inp,None))
    runtime={'native_backend':'ma27','hsl_library':'/fixture/never_loaded/libhsl.so','hsl_library_sha256':'b'*64,'runtime_environment':{'HSL_LIBRARY':'/fixture/never_loaded/libhsl.so'},'native_build_proof_sha256':'c'*64}
    def inspect(name,changes):
        control.write_text(json.dumps({**payload,**changes}))
        out=root/name;out.mkdir()
        return gate.inspect(case,root,binary,out,runner,5,runtime_config=runtime)
    assert inspect('clean',{})['passed'] is True
    bound={'root':str(root),'frozen_bindings':{case['case_id']:{'input':str(inp),'input_sha256':case['input_sha256']}},'_input_installation_sha256':'a'*64}
    seen=[]
    exact_runner=types.SimpleNamespace(verify_input=lambda cfg,c:(seen.append(cfg)or inp,None))
    dest=root/'explicit_bound';dest.mkdir()
    row=gate.inspect(case,root,binary,dest,exact_runner,5,bound,runtime)
    assert row['passed'] is True and seen==[bound] and row['input_path']==str(inp.resolve())
    assert row['input_installation_sha256']=='a'*64
    tests.append('Inspector receives exact installation bindings and records the resolved path')
    tests.append('Complete native inspection-only evidence accepted by the controller')
    for name,changes in (
      ('changed',{'native_signed_square_changed':True}),
      ('unknown_phase',{'post_presolve_absence_proved':False}),
      ('numerical_work',{'solver_invocations':1}),
      ('potential_shape',{'potential_signed_square_shape_count':1}),
      ('different_model',{'n':4})):
        assert inspect(name,changes)['passed'] is False
        tests.append(name+' blocks the factorial gate')
    proof={'binary_sha256':gate.sha(binary),'source':str(source),
           'source_sha256':{'fixed.hpp':gate.sha(source/'fixed.hpp')},'homogeneous_compilation':True}
    (root/'BUILD_NATIVE.json').write_text(json.dumps(proof))
    assert gate.build_identity(binary)['research_source_sha256']==proof['source_sha256']
    (source/'fixed.hpp').write_text('int changed_after_build;\n')
    try:gate.build_identity(binary)
    except ValueError:pass
    else:raise AssertionError('Source mutation after homogeneous build accepted')
    tests.append('Actual source tree mutation after build is rejected')
    try:gate.ma27_runtime_binding(binary)
    except ValueError:pass
    else:raise AssertionError('Historical non-MA27 build supplied an inspector runtime binding')
    tests.append('Old/MUMPS build cannot supply a new MA27 inspector runtime')
# A complete path map can point to exact files under arbitrary filenames;
# missing explicit bindings cannot silently fall back to canonical locations.
with tempfile.TemporaryDirectory(prefix='coupling-bound-inputs-') as directory:
    root=Path(directory);contract=root/'contract.json.gz'
    supplied=Path(__file__).parent.parent/'default_relaunch/CONTRAT_4078.json.gz'
    contract.write_bytes(supplied.read_bytes());cases=gate.load(contract)['cases']
    bindings={c['case_id']:{'input':str(root/'read_only_aliases'/(str(i)+'.qcq')),
        'input_sha256':c['input_sha256']}for i,c in enumerate(cases)}
    installation=root/'INSTALLATION.json';cfg={'schema':'ipopt-defaults-full-corpus-installation-v1',
        'root':str(root),'contract':str(contract),'contract_sha256':gate.sha(contract),'frozen_bindings':bindings}
    installation.write_text(json.dumps(cfg));resolved=gate.input_installation(installation,root,contract)
    assert gate.bound_input(resolved,cases[0])==Path(bindings[cases[0]['case_id']]['input'])
    tests.append('All4078 hash-bound paths are retained independently of case-directory names')
    removed=bindings.pop(cases[0]['case_id']);installation.write_text(json.dumps(cfg))
    try:gate.input_installation(installation,root,contract)
    except ValueError as e:assert 'exactly all 4078' in str(e)
    else:raise AssertionError('Incomplete explicit installation accepted')
    try:gate.bound_input(cfg,cases[0])
    except ValueError:pass
    else:raise AssertionError('Missing explicit bound path fell back to an inferred filename')
    tests.append('Missing explicit path blocks rather than reconstructing a canonical path')
    bindings[cases[0]['case_id']]=removed;removed['input_sha256']='0'*64
    installation.write_text(json.dumps(cfg))
    try:gate.input_installation(installation,root,contract)
    except ValueError as e:assert 'hash differs' in str(e)
    else:raise AssertionError('Different input SHA accepted in explicit path map')
    tests.append('All installed input hashes are bound to the same frozen full corpus')

receipt={'schema':'factorial-coupling-driver-controller-tests-v1','passed':True,'tests':tests,
         'actual_native_corpus_inspections':0,'actual_optimizer_invocations':0,
         'scope':'Synthetic controller fixtures; no scientific applicability claim.'}
Path(__file__).with_name('COUPLING_DRIVER_VALIDATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS',len(tests),'coupling-driver controller checks; native corpus uninspected')
