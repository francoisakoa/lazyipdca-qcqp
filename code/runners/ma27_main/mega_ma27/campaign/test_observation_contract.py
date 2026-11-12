#!/usr/bin/env python3
"""Counterexamples for original audits, native budgets and tri-state reporting."""
import copy
import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import audit_worker as worker
from observation_contract import normalize_result

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDITOR_PATHS = [ROOT/'runtime/original_auditor.py', ROOT/'common/original_auditor.py',
                 ROOT/'campaign/original_auditor.py', ROOT/'campaign/common/original_auditor.py',
                 ROOT.parent/'resume_dey/DEY_NATIVE_RECHERCHE_20261104/reprise/r8_2_pooling_candidate/runtime/original_auditor.py']
AUDITOR = next(path for path in AUDITOR_PATHS if path.is_file())
MODEL = '''LAZYIPDCA_QCQP_V1
name unit_original
n 1
x0 1
lb 0
ub inf
objective
quadratic
name f
constant 0
linear 0
hessian_nnz 1
0 0 1
end_quadratic
equalities 0
inequalities 0
end_problem
'''

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = pathlib.Path(self.tmp.name)
        self.inp = self.root/'input.qcq';self.inp.write_text(MODEL)
        self.case = dict(id='unit_original',input=str(self.inp),input_sha256=worker.sha(self.inp),
                         method='R3',seconds=7200,max_outer=5000,shared_restoration_limit=600,
                         solve_external_timeout=False,solve_process_seconds=.02)
        self.point = dict(x=[0.],certificate_equality_multipliers=[],certificate_inequality_multipliers=[],
                          certificate_lower_multipliers=[0.],certificate_upper_multipliers=[0.],
                          elapsed_seconds=.01,outer_iterations=2,restoration_iterations=1,
                          solution_origin='lazy',status='inner_failure',
                          native_al_fallback=dict(policy='off',enabled=False,attempted=False,accepted=False,
                                                  endpoint_origin='lazy',within_global_budget=True))
        config = dict(native_auditor=str(AUDITOR),native_auditor_sha256=worker.sha(AUDITOR))
        (self.root/'INSTALLATION.json').write_text(json.dumps(config))
    def tearDown(self):self.tmp.cleanup()
    def audit(self, point=None, case=None):
        (self.root/'case.json').write_text(json.dumps(case or self.case))
        (self.root/'endpoint.json').write_text(json.dumps(point or self.point))
        with patch.object(worker,'ROOT',self.root),patch.object(worker,'HERE',self.root/'campaign'):
            return worker.run(self.root/'case.json',self.root/'endpoint.json',self.root/'AUDIT.json')
    def result(self,audit):
        return dict(method=self.case['method'],work=dict(returncode=1,external_timeout=False,interrupted=False),
                    audit_work=dict(returncode=0,external_timeout=False,interrupted=False),
                    route_policy_check=dict(passed=True),audit=audit,solution_origin=audit['solution_origin'])
    def test_actual_original_KKT_ignores_native_status(self):
        a=self.audit();self.assertTrue(a['strict_qcqp']);self.assertEqual(a['dual'],0)
        r=normalize_result(self.result(a),integrity_verified=True)
        self.assertIs(r['strict_qcqp'],True);self.assertFalse(r['numerical_failure_classified'])
        for status in ['solved','time_limit','factorization_failure']:
            p=copy.deepcopy(self.point);p['status']=status
            self.assertIs(self.audit(p)['strict_qcqp'],True)
    def test_stationarity_and_complementarity_are_not_feasibility(self):
        for x,zl,want_dual,want_comp in [(1.,0.,1.,0.),(1.,1.,0.,1.)]:
            p=copy.deepcopy(self.point);p.update(x=[x],certificate_lower_multipliers=[zl],status='solved')
            a=self.audit(p);self.assertEqual(a['primal'],0);self.assertEqual(a['dual'],want_dual);self.assertEqual(a['complementarity'],want_comp)
            r=normalize_result(self.result(a),integrity_verified=True)
            self.assertIs(r['strict_qcqp'],False);self.assertTrue(r['numerical_failure_classified'])
    def test_shared_inclusive_budgets_enforced(self):
        modifications=[dict(elapsed_seconds=7200.0001),dict(outer_iterations=5001),dict(restoration_iterations=601),
                       dict(restoration_iterations=True),dict(outer_iterations=None)]
        for modification in modifications:
            p=copy.deepcopy(self.point);p.update(modification)
            a=self.audit(p);self.assertTrue(a['mathematical_strict_qcqp']);self.assertIsNone(a['strict_qcqp'])
            self.assertIsNone(normalize_result(self.result(a),integrity_verified=True)['strict_qcqp'])
        p=copy.deepcopy(self.point);p['status']='time_limit';p['elapsed_seconds']=7200.1
        p['native_al_fallback']['within_global_budget']=False
        a=self.audit(p);r=normalize_result(self.result(a),integrity_verified=True)
        self.assertIsNone(r['strict_qcqp']);self.assertEqual(r['observation_classification'],'budget_exhausted_indeterminate')
        self.assertFalse(r['numerical_failure_classified'])
        p=copy.deepcopy(self.point);p.update(status='time_limit',x=[1.],elapsed_seconds=7199.99)
        a=self.audit(p);r=normalize_result(self.result(a),integrity_verified=True)
        self.assertIsNone(r['strict_qcqp']);self.assertFalse(r['numerical_failure_classified'])
        for process in [None,True,-1.,float('nan'),7200.1]:
            case=copy.deepcopy(self.case);case['solve_process_seconds']=process
            p=copy.deepcopy(self.point);p['elapsed_seconds']=7199.
            a=self.audit(p,case);self.assertIsNone(a['strict_qcqp'])
            self.assertIsNone(normalize_result(self.result(a),integrity_verified=True)['strict_qcqp'])
    def test_missing_arrays_audits_and_censures_stay_indeterminate(self):
        p=copy.deepcopy(self.point);p.pop('certificate_equality_multipliers')
        a=self.audit(p);self.assertIsNone(a['strict_qcqp']);self.assertFalse(a['audit_available'])
        a=self.audit()
        for name,change in [('audit',None),('work',dict(returncode=-9,external_timeout=True)),
                            ('audit_work',dict(returncode=-9,interrupted=True)),('work',{}),('audit_work',{}),
                            ('work',dict(returncode=True,external_timeout=False,interrupted=False)),
                            ('audit_work',dict(returncode=False,external_timeout=False,interrupted=False))]:
            r=self.result(a);r[name]=change
            self.assertIsNone(normalize_result(r,integrity_verified=True)['strict_qcqp'])
        self.assertIsNone(normalize_result(self.result(a))['strict_qcqp'])
    def test_wrong_optimizer_route_cannot_qualify(self):
        p=copy.deepcopy(self.point);p['native_augmented_repair']=dict(enabled=True)
        a=self.audit(p);self.assertFalse(a['native_route_integrity_pass'])
        self.assertIsNone(normalize_result(self.result(a),integrity_verified=True)['strict_qcqp'])
    def test_Ipopt_arrays_status_and_limits(self):
        p=copy.deepcopy(self.point);p.update(solver='Ipopt',linear_solver='ma27',
                    ipopt_application_status='Restoration_Failed',core_s=.01,max_iterations=5000,
                    optimization_wall_limit_seconds=7200,iterations=2,equality_multipliers=[],inequality_multipliers=[],
                    lower_bound_multipliers=[0.],upper_bound_multipliers=[0.])
        for name in list(p):
            if name.startswith('certificate_'):p.pop(name)
        case=copy.deepcopy(self.case);case['method']='ipopt_qcqp'
        a=self.audit(p,case);self.assertIs(a['strict_qcqp'],True);self.assertEqual(a['solution_origin'],'ipopt')
        r=self.result(a);r['method']='ipopt_qcqp'
        self.assertIs(normalize_result(r,integrity_verified=True)['strict_qcqp'],True)
        for values in [dict(core_s=7201),dict(max_iterations=6000),dict(iterations=5001)]:
            q=copy.deepcopy(p);q.update(values);a=self.audit(q,case)
            self.assertIsNone(a['strict_qcqp'])
        q=copy.deepcopy(p);q.pop('iterations');a=self.audit(q,case)
        self.assertIsNone(a['strict_qcqp'])
        for process in [None,7200.1]:
            late=copy.deepcopy(case);late['solve_process_seconds']=process
            q=copy.deepcopy(p);q['core_s']=7199.
            a=self.audit(q,late);self.assertIsNone(a['strict_qcqp'])
        q=copy.deepcopy(p);q['x']=[1.];q['ipopt_application_status']='Solve_Succeeded'
        a=self.audit(q,case);self.assertIs(a['strict_qcqp'],False);self.assertEqual(a['dual'],1.)
    def test_Ipopt_MUMPS_is_separate_reference_not_native_or_MA27(self):
        p=copy.deepcopy(self.point);p.update(solver='Ipopt',linear_solver='mumps',requested_mumps_ordering=5,
                    core_s=.01,max_iterations=5000,optimization_wall_limit_seconds=7200,iterations=2,
                    equality_multipliers=[],inequality_multipliers=[],lower_bound_multipliers=[0.],upper_bound_multipliers=[0.])
        for name in list(p):
            if name.startswith('certificate_'):p.pop(name)
        case=copy.deepcopy(self.case);case['method']='ipopt_mumps'
        a=self.audit(p,case);self.assertIs(a['strict_qcqp'],True)
        r=self.result(a);r['method']='ipopt_mumps'
        self.assertIs(normalize_result(r,integrity_verified=True)['strict_qcqp'],True)
        r['route_policy_check']['passed']=False
        self.assertIsNone(normalize_result(r,integrity_verified=True)['strict_qcqp'])
    def test_invalid_configuration_or_nonfinite_cannot_qualify(self):
        for values in [dict(max_outer=None),dict(seconds=600),dict(shared_restoration_limit=0)]:
            case=copy.deepcopy(self.case);case.update(values)
            self.assertIsNone(self.audit(case=case)['strict_qcqp'])
        p=copy.deepcopy(self.point);p['x']=[float('nan')]
        a=self.audit(p);self.assertFalse(a['audit_available']);self.assertIsNone(a['strict_qcqp'])

if __name__=='__main__':unittest.main(verbosity=2)
