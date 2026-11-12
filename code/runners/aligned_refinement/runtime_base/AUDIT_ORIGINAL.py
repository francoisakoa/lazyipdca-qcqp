#!/usr/bin/env python3
"""Independent interval KKT assessment on an unchanged frozen V1/V2 QCQP.

The solver's status and residuals never determine the mathematical certificate.
Original finite bounds are retained, including when Ipopt internally relaxes
them. A censored execution can have a separately audited endpoint, but its
observation remains indeterminate. No optimizer is invoked by this file.
"""
from __future__ import annotations
import argparse
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import time

from frozen_qcqp_reader import read_model

TOLERANCE=Decimal('0.000001')

class PrecisionNeeded(Exception):
    pass

class FloatArithmetic:
    name='outward_binary64_fsum'
    @staticmethod
    def number(x): return float(x)
    @staticmethod
    def down(x):
        if math.isnan(x): raise PrecisionNeeded('Indeterminate binary64 operation')
        return math.nextafter(x,-math.inf)
    @staticmethod
    def up(x):
        if math.isnan(x): raise PrecisionNeeded('Indeterminate binary64 operation')
        return math.nextafter(x,math.inf)
    def product(self,a,b,lower):
        return self.down(a*b) if lower else self.up(a*b)
    def total(self,values,lower):
        try: value=math.fsum(values)
        except (ValueError,OverflowError) as e: raise PrecisionNeeded(str(e))
        # A second outward ulp also encloses the documented occasional
        # last-bit double rounding of fsum on extended-precision builds.
        fn=self.down if lower else self.up
        return fn(fn(value))

class DecimalArithmetic:
    name='directed_decimal150'
    def __init__(self):
        self.low=Context(prec=150,rounding=ROUND_FLOOR)
        self.high=Context(prec=150,rounding=ROUND_CEILING)
    @staticmethod
    def number(x):
        if isinstance(x,float):return Decimal.from_float(x)
        return Decimal(x)
    def product(self,a,b,lower):
        return (self.low if lower else self.high).multiply(a,b)
    def total(self,values,lower):
        context=self.low if lower else self.high
        value=Decimal(0)
        for x in values:value=context.add(value,x)
        return value

class Intervals:
    def __init__(self,arithmetic):self.a=arithmetic;self.zero=self.point(0)
    def point(self,x):v=self.a.number(x);return v,v
    def sum(self,terms):
        if not terms:return self.zero
        return self.a.total((v[0] for v in terms),True),self.a.total((v[1] for v in terms),False)
    @staticmethod
    def neg(x):
        # Decimal unary minus/abs obey the ambient context, which may be
        # only 28 digits. Exact sign operations preserve our directed bounds.
        negate=lambda v:v.copy_negate() if isinstance(v,Decimal) else -v
        return negate(x[1]),negate(x[0])
    def mul(self,x,y):
        if x==self.zero or y==self.zero:return self.zero
        low=[self.a.product(a,b,True) for a in x for b in y]
        high=[self.a.product(a,b,False) for a in x for b in y]
        return min(low),max(high)
    def abs(self,x):
        absolute=lambda v:v.copy_abs() if isinstance(v,Decimal) else abs(v)
        a,b=absolute(x[0]),absolute(x[1])
        return (self.zero[0] if x[0]<=0<=x[1] else min(a,b),max(a,b))
    @staticmethod
    def maximum(*values):return max(x[0] for x in values),max(x[1] for x in values)

def get_vectors(endpoint):
    groups=[('x',),('equality_multipliers','certificate_equality_multipliers'),
            ('inequality_multipliers','certificate_inequality_multipliers'),
            ('lower_bound_multipliers','certificate_lower_multipliers'),
            ('upper_bound_multipliers','certificate_upper_multipliers')]
    vectors=[]
    for names in groups:
        if endpoint.get('solver')!='Ipopt' and len(names)>1:
            names=tuple(reversed(names))
        vector=next((endpoint[n] for n in names if n in endpoint),None)
        if not isinstance(vector,list):raise ValueError('Missing physical vector: '+names[0])
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in vector):
            raise ValueError('Nonfinite physical vector: '+names[0])
        vectors.append(vector)
    return vectors

def physical_vectors(model,endpoint):
    vectors=get_vectors(endpoint)
    expected=[model['n'],len(model['eq']),len(model['ineq']),model['n'],model['n']]
    if [len(v) for v in vectors]!=expected:raise ValueError('Physical vector dimensions differ')
    return vectors

class FloatExpansion:
    """Compact nonoverlapping expansion; storage does not grow with term count."""
    def __init__(self):self.parts=[]
    def add(self,x):
        if x==0:return
        if not math.isfinite(x):raise PrecisionNeeded('Expansion term is nonfinite')
        i=0
        for y in self.parts:
            if abs(x)<abs(y):x,y=y,x
            hi=x+y
            if not math.isfinite(hi):raise PrecisionNeeded('Expansion overflow')
            lo=y-(hi-x)
            if not math.isfinite(lo):raise PrecisionNeeded('Expansion remainder overflow')
            if lo:self.parts[i]=lo;i+=1
            x=hi
        self.parts[i:]=[x]

class StreamingSum:
    def __init__(self,iv):
        self.iv=iv;self.decimal=isinstance(iv.a,DecimalArithmetic)
        if self.decimal:self.low=self.high=Decimal(0)
        else:self.low=FloatExpansion();self.high=FloatExpansion()
    def add(self,value):
        if value==self.iv.zero:return
        if self.decimal:
            self.low=self.iv.a.low.add(self.low,value[0]);self.high=self.iv.a.high.add(self.high,value[1])
        else:self.low.add(value[0]);self.high.add(value[1])
    def finish(self):
        if self.decimal:return self.low,self.high
        if not self.low.parts and not self.high.parts:return self.iv.zero
        return self.iv.a.total(self.low.parts,True),self.iv.a.total(self.high.parts,False)

def enclose_stream(input_path,endpoint,arithmetic):
    """One coefficient pass, O(n) gradient expansions; never stores the model."""
    iv=Intervals(arithmetic);xf,yf,zf,zlf,zuf=get_vectors(endpoint)
    x,y,z,zl,zu=[[iv.point(v) for v in vector] for vector in (xf,yf,zf,zlf,zuf)]
    with Path(input_path).open(encoding='utf-8') as stream:
        def row(key):
            parts=stream.readline().split()
            if not parts or parts[0]!=key:raise ValueError('Expected '+key)
            return parts[1:]
        if stream.readline().strip() not in ('LAZYIPDCA_QCQP_V1','LAZYIPDCA_QCQP_V2'):raise ValueError('V1/V2 model required')
        row('name');n=int(row('n')[0]);start=list(map(float,row('x0')))
        lower=list(map(float,row('lb')));upper=list(map(float,row('ub')))
        if n<1 or [len(v) for v in (x,start,lower,upper,zl,zu)]!=[n]*6:raise ValueError('Model/endpoint dimensions differ')
        if not all(math.isfinite(v) for v in start):raise ValueError('Raw start is nonfinite')
        if any(math.isnan(v) for v in lower+upper):raise ValueError('NaN bound')
        if any(math.isfinite(l) and math.isfinite(u) and l>u for l,u in zip(lower,upper)):raise ValueError('Finite bound order invalid')
        gradient=[StreamingSum(iv) for _ in range(n)]
        for acc,l,u in zip(gradient,zl,zu):acc.add(iv.neg(l));acc.add(u)
        def coefficient(token):
            v=float(token)
            if not math.isfinite(v):raise ValueError('Nonfinite coefficient')
            return v
        def function(weight):
            row('quadratic');row('name');value=StreamingSum(iv);value.add(iv.point(coefficient(row('constant')[0])))
            def linear_term(j,a):
                if not 0<=j<n:raise ValueError('Invalid linear coordinate')
                if not a:return
                value.add(iv.mul(iv.point(a),x[j]))
                if weight!=iv.zero:gradient[j].add(iv.mul(weight,iv.point(a)))
            parts=stream.readline().split()
            if not parts:raise ValueError('Linear row absent')
            if parts[0]=='linear':
                if len(parts)!=n+1:raise ValueError('Dense linear dimension differs')
                for j,token in enumerate(parts[1:]):linear_term(j,coefficient(token))
            elif parts[0]=='linear_nnz':
                count=int(parts[1])
                if count<0:raise ValueError('Negative linear count')
                for _ in range(count):
                    j,a=stream.readline().split();linear_term(int(j),coefficient(a))
            else:raise ValueError('Unknown linear format')
            count=int(row('hessian_nnz')[0])
            if count<0:raise ValueError('Negative Hessian count')
            for _ in range(count):
                i,j,a=stream.readline().split();i=int(i);j=int(j);a=coefficient(a)
                if not(0<=i<n and 0<=j<n):raise ValueError('Invalid Hessian coordinate')
                if not a:continue
                c=iv.point(a)
                value.add(iv.mul(iv.mul(c,iv.mul(x[i],x[j])),iv.point(.5 if i==j else 1)))
                if weight!=iv.zero:
                    weighted=iv.mul(weight,c);gradient[i].add(iv.mul(weighted,x[j]))
                    if i!=j:gradient[j].add(iv.mul(weighted,x[i]))
            row('end_quadratic');return value.finish()
        row('objective');objective=function(iv.point(1));primal=dual=compl=iv.zero
        count=int(row('equalities')[0])
        if count<0 or len(y)!=count:raise ValueError('Equality multiplier dimension differs')
        for mult in y:primal=iv.maximum(primal,iv.abs(function(mult)))
        count=int(row('inequalities')[0])
        if count<0 or len(z)!=count:raise ValueError('Inequality multiplier dimension differs')
        for mult in z:
            g=function(mult);primal=iv.maximum(primal,g);compl=iv.maximum(compl,iv.abs(iv.mul(g,mult)));dual=iv.maximum(dual,iv.neg(mult))
        row('end_problem')
        while True:
            trailing=stream.read(65536)
            if not trailing:break
            if trailing.strip():raise ValueError('Trailing model content')
        for j,(l,u) in enumerate(zip(lower,upper)):
            dual=iv.maximum(dual,iv.neg(zl[j]),iv.neg(zu[j]),iv.abs(gradient[j].finish()))
            if math.isfinite(l):
                gap=iv.sum([x[j],iv.neg(iv.point(l))]);primal=iv.maximum(primal,iv.neg(gap));compl=iv.maximum(compl,iv.abs(iv.mul(zl[j],gap)))
            else:dual=iv.maximum(dual,iv.abs(zl[j]))
            if math.isfinite(u):
                gap=iv.sum([iv.point(u),iv.neg(x[j])]);primal=iv.maximum(primal,iv.neg(gap));compl=iv.maximum(compl,iv.abs(iv.mul(zu[j],gap)))
            else:dual=iv.maximum(dual,iv.abs(zu[j]))
    return dict(objective=objective,primal=primal,dual=dual,complementarity=compl)

def enclose(model,endpoint,arithmetic):
    iv=Intervals(arithmetic)
    xf,yf,zf,zlf,zuf=physical_vectors(model,endpoint)
    x,y,z,zl,zu=[[iv.point(t) for t in v] for v in (xf,yf,zf,zlf,zuf)]
    terms=[[iv.neg(l),u] for l,u in zip(zl,zu)]
    primal=dual=compl=iv.zero
    def value(q):
        constant,linear,hessian=q
        values=[iv.point(constant)]
        values.extend(iv.mul(iv.point(a),x[j]) for j,a in linear)
        for i,j,a in hessian:
            values.append(iv.mul(iv.mul(iv.point(a),iv.mul(x[i],x[j])),iv.point(.5 if i==j else 1)))
        return iv.sum(values)
    def gradient(q,weight):
        if weight==iv.zero:return
        _,linear,hessian=q
        for j,a in linear:terms[j].append(iv.mul(weight,iv.point(a)))
        for i,j,a in hessian:
            c=iv.mul(weight,iv.point(a))
            terms[i].append(iv.mul(c,x[j]))
            if i!=j:terms[j].append(iv.mul(c,x[i]))
    objective=value(model['obj']);gradient(model['obj'],iv.point(1))
    for q,mult in zip(model['eq'],y):
        primal=iv.maximum(primal,iv.abs(value(q)));gradient(q,mult)
    for q,mult in zip(model['ineq'],z):
        g=value(q);primal=iv.maximum(primal,g)
        compl=iv.maximum(compl,iv.abs(iv.mul(g,mult)))
        dual=iv.maximum(dual,iv.neg(mult));gradient(q,mult)
    for j,(lower,upper) in enumerate(zip(model['lb'],model['ub'])):
        dual=iv.maximum(dual,iv.neg(zl[j]),iv.neg(zu[j]))
        if math.isfinite(lower):
            gap=iv.sum([x[j],iv.neg(iv.point(lower))])
            primal=iv.maximum(primal,iv.neg(gap));compl=iv.maximum(compl,iv.abs(iv.mul(zl[j],gap)))
        else:dual=iv.maximum(dual,iv.abs(zl[j]))
        if math.isfinite(upper):
            gap=iv.sum([iv.point(upper),iv.neg(x[j])])
            primal=iv.maximum(primal,iv.neg(gap));compl=iv.maximum(compl,iv.abs(iv.mul(zu[j],gap)))
        else:dual=iv.maximum(dual,iv.abs(zu[j]))
        dual=iv.maximum(dual,iv.abs(iv.sum(terms[j])))
    return dict(objective=objective,primal=primal,dual=dual,complementarity=compl)

def assess_intervals(intervals,decimal=False):
    threshold=TOLERANCE if decimal else float(TOLERANCE)
    lo_threshold=threshold if decimal else math.nextafter(threshold,-math.inf)
    hi_threshold=threshold if decimal else math.nextafter(threshold,math.inf)
    residuals=[intervals[n] for n in ('primal','dual','complementarity')]
    if all(v[1]<=lo_threshold for v in residuals):return True
    if any(v[0]>hi_threshold for v in residuals):return False
    return None

def endpoint_audit(model,endpoint):
    try:
        bounds=enclose(model,endpoint,FloatArithmetic())
        strict=assess_intervals(bounds)
    except (PrecisionNeeded,OverflowError):
        bounds=None;strict=None
    method=FloatArithmetic.name
    if strict is None:
        bounds=enclose(model,endpoint,DecimalArithmetic())
        strict=assess_intervals(bounds,True);method=DecimalArithmetic.name
    return format_assessment(bounds,strict,method,'sparse_model')

def endpoint_audit_stream(input_path,endpoint):
    try:
        bounds=enclose_stream(input_path,endpoint,FloatArithmetic())
        strict=assess_intervals(bounds)
    except (PrecisionNeeded,OverflowError):bounds=None;strict=None
    method='streaming_'+FloatArithmetic.name
    if strict is None:
        bounds=enclose_stream(input_path,endpoint,DecimalArithmetic())
        strict=assess_intervals(bounds,True);method='streaming_'+DecimalArithmetic.name
    return format_assessment(bounds,strict,method,'streamed_coefficients_O_n_gradient_expansions')

def format_assessment(bounds,strict,method,storage):
    return dict(precision=method,storage_mode=storage,original_KKT_endpoint_strict=strict,
                intervals={k:dict(lower=str(v[0]),upper=str(v[1])) for k,v in bounds.items()},
                residual_upper={k:finite_float(v[1]) for k,v in bounds.items() if k!='objective'},
                tolerance='1e-6',original_bounds_used=True,
                absent_bound_semantics='A bound side is absent iff its frozen value is infinite, regardless of sign; finite bounds unchanged')

def finite_float(value):
    try:v=float(value)
    except (ValueError,OverflowError):return None
    return v if math.isfinite(v) else None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def censor_reasons(resources):
    reasons=[]
    for key in ('external_timeout','timed_out','censored','interrupted','killed','terminated_by_controller'):
        if resources.get(key) is True:reasons.append(key)
    code=resources.get('returncode')
    if isinstance(code,int) and code<0:reasons.append('process_signal_'+str(-code))
    reasons.extend(str(s) for s in resources.get('observation_indeterminate_reasons',[]) if s)
    return sorted(set(reasons))

def audit_files(input_path,endpoint_path,resources=None,case=None):
    started=time.monotonic();resources=resources or {};case=case or {}
    case=case.get('case',case)
    result=dict(schema='ipopt-default-original-interval-audit-v1',input_sha256=sha(input_path),
                endpoint_sha256=None,audit_available=False,strict_qcqp=None,
                observation_classification='indeterminate',indeterminate_reasons=censor_reasons(resources),
                original_KKT_endpoint_strict=None,classification='indeterminate',
                primal=None,dual=None,complementarity=None,objective=None,audit_precision=None,
                case_id=case.get('case_id',case.get('id')),method=case.get('method'),
                dataset=case.get('dataset'),regime=case.get('regime'),
                optimizer_invocations=0,tolerance='1e-6')
    try:
        if case.get('input_sha256') and result['input_sha256']!=case['input_sha256']:
            raise ValueError('Frozen input hash differs from case contract')
        if not Path(endpoint_path).is_file():raise ValueError('Endpoint absent')
        result['endpoint_sha256']=sha(endpoint_path)
        endpoint=json.loads(Path(endpoint_path).read_text())
        if Path(input_path).stat().st_size>32*1024*1024:
            point=endpoint_audit_stream(input_path,endpoint)
        else:
            model=read_model(input_path)
            for lower,upper in zip(model['lb'],model['ub']):
                if math.isfinite(lower) and math.isfinite(upper) and lower>upper:raise ValueError('Finite bound order invalid')
            point=endpoint_audit(model,endpoint)
        result.update(point);result['audit_available']=True
        result.update(point['residual_upper']);result['audit_precision']=point['precision']
        obj=point['intervals']['objective']
        result['objective']=finite_float((Decimal(obj['lower'])+Decimal(obj['upper']))/2)
        if point['original_KKT_endpoint_strict'] is None:result['indeterminate_reasons'].append('Original interval straddles threshold at Decimal150')
        if not result['indeterminate_reasons']:
            result['strict_qcqp']=point['original_KKT_endpoint_strict']
            result['observation_classification']='strict_original_qcqp' if result['strict_qcqp'] else 'audited_original_qcqp_nonstrict'
            result['classification']='strict' if result['strict_qcqp'] else 'nonstrict'
        result['solver_status']=endpoint.get('status')
        result['solver_success_is_not_certificate']=True
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError) as error:
        result['indeterminate_reasons'].append(type(error).__name__+': '+str(error))
    result['indeterminate_reasons']=sorted(set(result['indeterminate_reasons']))
    result['reason_list']=result['indeterminate_reasons']
    result['audit_seconds']=time.monotonic()-started
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--endpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resources',type=Path)
    parser.add_argument('--case-json',type=Path)
    args=parser.parse_args()
    resources=json.loads(args.resources.read_text()) if args.resources else {}
    case=json.loads(args.case_json.read_text()) if args.case_json else {}
    result=audit_files(args.input,args.endpoint,resources,case)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(args.out)
    print(json.dumps({k:result[k] for k in ('audit_available','strict_qcqp','observation_classification','indeterminate_reasons')}),flush=True)
    # A completed mathematical audit is a successful process even for a
    # noncertified endpoint. Missing/indeterminate observations are separate.
    return 0 if result['audit_available'] else 2

if __name__=='__main__':raise SystemExit(main())
