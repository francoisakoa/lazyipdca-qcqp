#!/usr/bin/env python3
"""Synthetic protocol tests. No optimizer or actual backend qualification."""
from pathlib import Path
import copy,gzip,hashlib,importlib.util,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import setup_runtime as runtime
import ipopt_worker as worker

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return path

def endpoint(backend,**changes):
    options=dict(runtime.DEFAULT_OPTIONS,linear_solver=backend,max_iter=300,max_cpu_time=60,max_wall_time=60)
    if backend=='mumps':options['mumps_pivot_order']=7
    data=dict(adapter_version='ipopt-library-defaults-original-qcqp-v1',solver='Ipopt',
        ipopt_version='3.14.11',linear_solver=backend,requested_mumps_ordering=0,
        effective_options=options,input_representation_overrides=[])
    data.update(changes);return data

def log_for(data,version='3.14.11'):
    marker=dict(ipopt_version=data['ipopt_version'],effective_options=data['effective_options'],
        input_representation_overrides=data['input_representation_overrides'],
        explicit_nonrepresentation_overrides=sorted(runtime.DEFAULT_EXPLICIT_OVERRIDES))
    return 'This is Ipopt version '+version+', running with linear solver '+data['linear_solver']+'\nIPOPT_DEFAULTS_EFFECTIVE '+json.dumps(marker)+'\n'

def fixture(root,alter=None):
    root=Path(root);out=root/'defaults';out.mkdir()
    source=root/'adapter';(source/'src').mkdir(parents=True)
    src=[]
    for name in ('direct_ipopt.cpp','ipopt_qcqp_main.cpp','io.cpp','io_compact.cpp'):
        p=source/'src'/name;p.write_bytes(b'synthetic test source; not compiled\n');src.append(p)
    patch=root/'PATCH_CONTRACT.json';patch.write_text('{}\n')
    binary=out/'adapter_cli';binary.write_bytes(b'synthetic checksum fixture; never executed\n')
    library=out/'libipopt_fixture';library.write_bytes(b'library checksum fixture\n')
    hsl=out/'libhsl_fixture';hsl.write_bytes(b'HSL checksum fixture\n')
    corpus=out/'CONTRAT.json.gz';corpus.write_bytes(gzip.compress(json.dumps({'cases':[{'case_id':'case'+str(i)}for i in range(4078)]}).encode()))
    records=[]
    for name in runtime.DEFAULT_FIXTURES:
        inp=out/'qualification'/(name+'.qcq');inp.parent.mkdir(exist_ok=True);inp.write_bytes(b'synthetic input protocol fixture\n')
        for backend in ('ma27','mumps'):
            task=inp.parent/(name+'__'+backend);task.mkdir()
            ep=endpoint(backend)
            if alter:alter(ep,name,backend)
            point=write(task/'endpoint.json',ep)
            audit=dict(strict_qcqp=True,original_bounds_used=True,input_sha256=sha(inp),endpoint_sha256=sha(point),
                primal=0.,dual=0.,complementarity=0.)
            checked=write(task/'AUDIT.json',audit);(task/'solver.log').write_text(log_for(ep))
            records.append(dict(fixture=name,backend=backend,returncode=0,input_sha256=sha(inp),endpoint_sha256=sha(point),audit_sha256=sha(checked),
                proof=dict(runtime_version='3.14.11',effective_options=ep['effective_options'])))
    qualification=write(out/'QUALIFICATION.json',dict(schema='ipopt-defaults-target-qualification-v1',records=records,
        qualified_backends=['ma27','mumps'],all_original_strict=True))
    build=write(out/'BUILD.json',dict(schema='isolated-ipopt-library-defaults-build-v1',original_installation_modified=False,
        binary=str(binary),binary_sha256=sha(binary),ipopt_library=str(library),ipopt_library_sha256=sha(library),
        ipopt_version='3.14.11',source_sha256={str(p.relative_to(source)):sha(p)for p in src},
        compile_command=['g++']+[str(p)for p in src],patch_contract_sha256=sha(patch)))
    defaults=dict(schema='ipopt-defaults-full-corpus-installation-v1',cases=4078,tasks=8156,datasets=28,
        qualified_backends=['ma27','mumps'],budgets={'seconds':7200,'max_iterations':5000},
        default_binary=str(binary),default_binary_sha256=sha(binary),ipopt_library=str(library),ipopt_library_sha256=sha(library),
        hsl_library=str(hsl),hsl_library_sha256=sha(hsl),qualification=str(qualification),qualification_sha256=sha(qualification),
        build=str(build),build_sha256=sha(build),contract=str(corpus),contract_sha256=sha(corpus))
    installation=write(out/'INSTALLATION.json',defaults)
    cfg=dict(ipopt_defaults_installation=str(installation),ipopt_defaults_installation_sha256=sha(installation),
        ipopt_binary=str(binary),ipopt_binary_sha256=sha(binary),hsl_library=str(hsl),hsl_library_sha256=sha(hsl),
        ipopt_version='3.14.11',ipopt_linear_solver='ma27',ipopt_mumps_linear_solver='mumps',
        ipopt_mumps_pivot_order=7,ipopt_mumps_ordering_policy='library-default',
        ma27_qualification=str(qualification),ma27_qualification_sha256=sha(qualification),
        mumps_qualification=str(qualification),mumps_qualification_sha256=sha(qualification))
    return cfg,defaults,qualification

class BridgeTests(unittest.TestCase):
    def test_full_eight_fixture_receipts_are_verified_without_optimizer(self):
        with tempfile.TemporaryDirectory()as root:
            cfg,_,_=fixture(root);proof=runtime.validate_ipopt_runtime(cfg)
            self.assertEqual(proof['original_KKT_fixture_count'],8);self.assertEqual(proof['optimizer_invocations'],0)
            self.assertEqual(len(proof['backend_evidence']['ma27']),4);self.assertEqual(len(proof['backend_evidence']['mumps']),4)
    def test_command_has_only_resource_backend_io_options(self):
        with tempfile.TemporaryDirectory()as root:
            cfg,_,_=fixture(root)
            for method,backend in [('ipopt_qcqp','ma27'),('ipopt_mumps','mumps')]:
                cmd=worker.command_for(cfg,{'method':method},Path(root)/'input',Path(root)/'output',7200,5000)
                self.assertEqual(cmd[cmd.index('--linear-solver')+1],backend)
                flags={v for v in cmd if v.startswith('--')}
                self.assertEqual(flags,{'--problem','--json','--time-limit','--max-iter','--linear-solver'}|({'--hsl-library'}if backend=='ma27'else set()))
    def test_worker_rejects_other_task_budget(self):
        with tempfile.TemporaryDirectory()as root:
            cfg,_,_=fixture(root)
            with self.assertRaisesRegex(ValueError,'common7200'):worker.command_for(cfg,{'method':'ipopt_qcqp'},'in','out',600,5000)
    def test_changed_binary_and_library_rejected(self):
        for key in ('ipopt_binary','hsl_library'):
            with tempfile.TemporaryDirectory()as root:
                cfg,_,_=fixture(root);Path(cfg[key]).write_bytes(b'mutated')
                with self.assertRaisesRegex(RuntimeError,'changed'):runtime.validate_ipopt_runtime(cfg)
    def test_forged_rehashed_adaptive_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory()as root:
            def adaptive(ep,name,backend):
                if name=='active_bound'and backend=='mumps':ep['effective_options']['mu_strategy']='adaptive'
            cfg,_,_=fixture(root,adaptive)
            with self.assertRaisesRegex(RuntimeError,'fixture proof failed'):runtime.validate_ipopt_runtime(cfg)
    def test_original_audit_is_required_even_when_hashes_agree(self):
        with tempfile.TemporaryDirectory()as root:
            cfg,defaults,qpath=fixture(root);q=json.loads(qpath.read_text());r=q['records'][0]
            audit_path=qpath.parent/'qualification'/(r['fixture']+'__'+r['backend'])/'AUDIT.json'
            audit=json.loads(audit_path.read_text());audit['strict_qcqp']=False;write(audit_path,audit)
            r['audit_sha256']=sha(audit_path);write(qpath,q)
            defaults['qualification_sha256']=sha(qpath);write(cfg['ipopt_defaults_installation'],defaults)
            cfg['ipopt_defaults_installation_sha256']=sha(cfg['ipopt_defaults_installation'])
            cfg['ma27_qualification_sha256']=cfg['mumps_qualification_sha256']=sha(qpath)
            with self.assertRaisesRegex(RuntimeError,'original-KKT'):runtime.validate_ipopt_runtime(cfg)
    def test_missing_backend_fixture_rejected(self):
        with tempfile.TemporaryDirectory()as root:
            cfg,defaults,qpath=fixture(root);q=json.loads(qpath.read_text());q['records'].pop();write(qpath,q)
            defaults['qualification_sha256']=sha(qpath);write(cfg['ipopt_defaults_installation'],defaults)
            cfg['ipopt_defaults_installation_sha256']=sha(cfg['ipopt_defaults_installation'])
            cfg['ma27_qualification_sha256']=cfg['mumps_qualification_sha256']=sha(qpath)
            with self.assertRaisesRegex(RuntimeError,'four independently'):runtime.validate_ipopt_runtime(cfg)
    def test_forced_ordering_and_nondefault_actual_order_rejected(self):
        cfg={'ipopt_version':'3.14.11','task_budgets':{'seconds':60,'max_outer':300}}
        for requested,actual in ((5,7),(0,5)):
            ep=endpoint('mumps',requested_mumps_ordering=requested);ep['effective_options']['mumps_pivot_order']=actual
            checked=runtime.backend_evidence(ep,log_for(ep),cfg,method='ipopt_mumps',require_banner=True)
            self.assertFalse(checked['pass']);self.assertFalse(checked['KKT_certificate'])
    def test_wrong_runtime_banner_version_rejected(self):
        ep=endpoint('ma27');cfg={'ipopt_version':'3.14.11','task_budgets':{'seconds':60,'max_outer':300}}
        proof=runtime.backend_evidence(ep,log_for(ep,'3.14.10'),cfg,require_banner=True)
        self.assertFalse(proof['pass']);self.assertIn('runtime_version_banner_differs',proof['violations'])
    def test_missing_marker_or_disagreeing_marker_rejected(self):
        ep=endpoint('ma27');cfg={'ipopt_version':'3.14.11','task_budgets':{'seconds':60,'max_outer':300}}
        self.assertFalse(runtime.backend_evidence(ep,'',cfg)['pass'])
        altered=copy.deepcopy(ep);altered['effective_options']['tol']=1e-6
        self.assertFalse(runtime.backend_evidence(ep,log_for(altered),cfg)['pass'])
    def test_unapproved_option_override_is_rejected(self):
        ep=endpoint('ma27');cfg={'ipopt_version':'3.14.11','task_budgets':{'seconds':60,'max_outer':300}}
        text=log_for(ep).replace('"print_level",','"tol",')
        self.assertFalse(runtime.backend_evidence(ep,text,cfg)['pass'])
    def test_incomplete_option_metadata_returns_unverified_evidence(self):
        ep=endpoint('mumps',effective_options=None)
        cfg={'ipopt_version':'3.14.11','task_budgets':{'seconds':60,'max_outer':300}}
        proof=runtime.backend_evidence(ep,'',cfg,method='ipopt_mumps')
        self.assertFalse(proof['pass']);self.assertIsNone(proof['effective_mumps_pivot_order'])

    def test_no_silent_legacy_adaptive_installation_fallback(self):
        with self.assertRaises(KeyError):runtime.validate_ipopt_runtime({'ipopt_binary':'old'})

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BridgeTests))
    destination=Path(__file__).resolve().parent.parent/'validation/IPOPT_DEFAULTS_BRIDGE_TESTS.json'
    write(destination,dict(schema='previous-mega-default-ipopt-bridge-tests-v1',passed=result.wasSuccessful(),
        tests_run=result.testsRun,optimizer_invocations=0,real_backend_qualification_performed=False,
        source_sha256={str(p.relative_to(destination.parent.parent)):sha(p)for p in (Path(runtime.__file__),Path(worker.__file__))}))
    raise SystemExit(0 if result.wasSuccessful()else 1)
