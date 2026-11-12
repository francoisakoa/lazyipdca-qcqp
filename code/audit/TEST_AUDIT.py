#!/usr/bin/env python3
"""Small mathematical qualification tests; no optimizer is executed."""
from pathlib import Path
from decimal import Decimal
import json,tempfile,tracemalloc
import AUDIT_ORIGINAL as checker

ZERO=(0.,[],[])

def model(objective,eq=None,ineq=None,lb=None,ub=None):
    return dict(n=1,x0=[0.],lb=lb if lb is not None else [-float('inf')],
                ub=ub if ub is not None else [float('inf')],obj=objective,eq=eq or [],ineq=ineq or [])

def endpoint(x,y=None,z=None,zl=0.,zu=0.):
    return dict(solver='Ipopt',status='solved',x=[x],equality_multipliers=y or [],
                inequality_multipliers=z or [],lower_bound_multipliers=[zl],upper_bound_multipliers=[zu])

def run():
    checks=[]
    def check(name,ok,**extra):
        assert ok,name
        checks.append(dict(test=name,passed=True,**extra))
    iv=checker.Intervals(checker.DecimalArithmetic())
    exact=Decimal('1.000000000000000000000000000000000001')
    check('directed_decimal_sign_operations_preserve_more_than28_digits',
          iv.neg((exact,exact))[0].copy_negate()==exact and iv.abs(iv.neg((exact,exact)))[0]==exact)
    # Active lower bound: grad f - z_L = 0 on the original bound x>=1.
    m=model((0.,[(0,1.)],[]),lb=[1.],ub=[-float('inf')])
    a=checker.endpoint_audit(m,endpoint(1.,zl=1.))
    check('active_original_lower_bound_and_noncanonical_absent_upper',a['original_KKT_endpoint_strict'] is True)
    e=endpoint(1.,zl=0.);e['solver']='native';e.update(certificate_equality_multipliers=[],
        certificate_inequality_multipliers=[],certificate_lower_multipliers=[1.],certificate_upper_multipliers=[0.])
    a=checker.endpoint_audit(m,e)
    check('native_point_uses_physical_certificate_vectors',a['original_KKT_endpoint_strict'] is True)
    # Relaxation is an Ipopt internal operation; the checker retains the original
    # bound and must detect violation rather than projecting the endpoint.
    m=model((0.,[(0,1.)],[]),lb=[0.])
    a=checker.endpoint_audit(m,endpoint(-2e-6,zl=1.))
    check('relaxed_bound_endpoint_audited_against_original_bound',a['original_KKT_endpoint_strict'] is False and a['residual_upper']['primal']>1e-6)
    # Cancellation makes a binary64 enclosure undecided. The exact physical
    # coefficient is just above the decimal threshold, so Decimal must reject it.
    c=1.0000000000000002e-6
    m=model((0.,[(0,1e16)],[]),eq=[(0.,[(0,-1e16)],[])],ineq=[(0.,[(0,c)],[])])
    a=checker.endpoint_audit(m,endpoint(0.,y=[1.],z=[1.]))
    check('cancellation_and_exact_decimal_threshold',a['original_KKT_endpoint_strict'] is False and a['precision']=='directed_decimal150')
    # The certificate is first order. An indefinite unconstrained stationary
    # point is certified without asserting local or global minimization.
    m=model((0.,[],[(0,0,-2.)]))
    a=checker.endpoint_audit(m,endpoint(0.))
    check('indefinite_stationary_point_is_only_a_first_order_certificate',a['original_KKT_endpoint_strict'] is True)
    fixture='''LAZYIPDCA_QCQP_V2
name audit_fixture
n 1
x0 1
lb 1
ub -inf
objective
quadratic
name objective
constant 0
linear_nnz 1
0 1
hessian_nnz 0
end_quadratic
equalities 0
inequalities 0
end_problem
'''
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/'problem.qcq').write_text(fixture);(p/'endpoint.json').write_text(json.dumps(endpoint(1.,zl=1.)))
        a=checker.audit_files(p/'problem.qcq',p/'endpoint.json',resources={'returncode':1})
        check('nonzero_solver_return_does_not_override_original_certificate',a['strict_qcqp'] is True and a['classification']=='strict')
        a=checker.audit_files(p/'problem.qcq',p/'endpoint.json',resources={'timed_out':True,'returncode':-15})
        check('censored_observation_keeps_audited_point_separate',a['strict_qcqp'] is None and a['classification']=='indeterminate' and a['original_KKT_endpoint_strict'] is True)
        a=checker.audit_files(p/'problem.qcq',p/'missing.json')
        check('missing_endpoint_is_indeterminate',a['strict_qcqp'] is None and not a['audit_available'])
        bad=endpoint(1.,zl=1.);bad['x'][0]=float('nan');(p/'endpoint.json').write_text(json.dumps(bad))
        a=checker.audit_files(p/'problem.qcq',p/'endpoint.json')
        check('nonfinite_endpoint_is_indeterminate',a['strict_qcqp'] is None and not a['audit_available'])
        # Force the streaming path on the same physically active bound.
        a=checker.endpoint_audit_stream(p/'problem.qcq',endpoint(1.,zl=1.))
        check('streaming_active_original_bound',a['original_KKT_endpoint_strict'] is True)
        cancellation=fixture.replace('lb 1','lb -inf').replace('linear_nnz 1\n0 1','linear_nnz 1\n0 1e16')
        q='quadratic\nname q\nconstant 0\nlinear_nnz 1\n0 {a}\nhessian_nnz 0\nend_quadratic\n'
        cancellation=cancellation.replace('equalities 0\ninequalities 0\n',
            'equalities 1\n'+q.format(a='-1e16')+'inequalities 1\n'+q.format(a='1.0000000000000002e-6'))
        (p/'cancellation.qcq').write_text(cancellation)
        a=checker.endpoint_audit_stream(p/'cancellation.qcq',endpoint(0.,y=[1.],z=[1.]))
        check('streaming_directed_decimal_threshold',a['original_KKT_endpoint_strict'] is False and a['precision']=='streaming_directed_decimal150')
        # 160,000 retained nonzero linear coefficients are evaluated rather than
        # retained as a model. The endpoint arrays and gradient expansions fit
        # in a small fixed space; no list grows with all these coefficients.
        n=32;m=5000
        with (p/'many_rows.qcq').open('w') as f:
            f.write('LAZYIPDCA_QCQP_V1\nname streaming_memory\nn '+str(n)+'\nx0 '+' '.join(['0']*n)+'\nlb '+' '.join(['-inf']*n)+'\nub '+' '.join(['inf']*n)+'\nobjective\nquadratic\nname f\nconstant 0\nlinear '+' '.join(['0']*n)+'\nhessian_nnz 0\nend_quadratic\nequalities '+str(m)+'\n')
            row='quadratic\nname q\nconstant 0\nlinear '+' '.join(['1']*n)+'\nhessian_nnz 0\nend_quadratic\n'
            for _ in range(m):f.write(row)
            f.write('inequalities 0\nend_problem\n')
        e=dict(solver='Ipopt',x=[0.]*n,equality_multipliers=[0.]*m,inequality_multipliers=[],lower_bound_multipliers=[0.]*n,upper_bound_multipliers=[0.]*n)
        tracemalloc.start();a=checker.endpoint_audit_stream(p/'many_rows.qcq',e);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        check('streaming160000_nonzero_coefficients_bounded_memory',a['original_KKT_endpoint_strict'] is True and peak<8*1024*1024,peak_traced_bytes=peak,coefficients=160000)
    report=dict(schema='default-ipopt-original-audit-qualification-v1',status='PASS',tests=checks,
                optimizer_invocations=0,qualified_first_order_threshold='1e-6',new_solver_runs=0)
    (Path(__file__).parent/'AUDIT_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),optimizer_invocations=0)))

if __name__=='__main__':run()
