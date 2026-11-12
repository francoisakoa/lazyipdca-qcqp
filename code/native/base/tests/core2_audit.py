#!/usr/bin/env python3
"""Independent sparse scalar audit of a direct V1 QCQP, not a CUTEst call.

Dense V1 linear rows are streamed and zero coefficients are discarded. Neither
solver callbacks nor status/residual fields are used to compute the certificate.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

def read_model(path):
    with Path(path).open(encoding='utf-8') as stream:
        def row(key):
            parts=stream.readline().split()
            if not parts or parts[0]!=key:raise ValueError('Expected '+key)
            return parts[1:]
        if stream.readline().strip() not in ('LAZYIPDCA_QCQP_V1','LAZYIPDCA_QCQP_V2'):raise ValueError('V1/V2 model required')
        name=' '.join(row('name'));n=int(row('n')[0]);x0=list(map(float,row('x0')))
        lb=list(map(float,row('lb')));ub=list(map(float,row('ub')))
        if n<1 or any(len(a)!=n for a in (x0,lb,ub)):raise ValueError('Model dimensions differ')
        if not all(math.isfinite(x) for x in x0) or any(math.isnan(v) for a in (lb,ub) for v in a):raise ValueError('Invalid coordinates or bounds')
        def quad():
            row('quadratic');nm=' '.join(row('name'));constant=float(row('constant')[0])
            parts=stream.readline().split();linear=[]
            if parts[0]=='linear':
                if len(parts)!=n+1:raise ValueError('Linear dimension differs')
                linear=[(j,float(v)) for j,v in enumerate(parts[1:]) if float(v)]
            elif parts[0]=='linear_nnz':
                nnz=int(parts[1])
                if nnz<0:raise ValueError('Negative linear length')
                for _ in range(nnz):
                    j,v=stream.readline().split();linear.append((int(j),float(v)))
            else:raise ValueError('Unknown linear format')
            if any(not (0<=j<n) or not math.isfinite(v) for j,v in linear):raise ValueError('Invalid linear coefficient')
            nnz=int(row('hessian_nnz')[0]);hessian=[]
            if nnz<0:raise ValueError('Negative Hessian length')
            for _ in range(nnz):
                a,b,c=stream.readline().split();i=int(a);j=int(b);v=float(c)
                if not(0<=i<n and 0<=j<n) or not math.isfinite(v):raise ValueError('Invalid Hessian')
                if v:hessian.append((i,j,v))
            row('end_quadratic')
            if not math.isfinite(constant):raise ValueError('Nonfinite constant')
            return constant,linear,hessian
        row('objective');obj=quad();ne=int(row('equalities')[0])
        if ne<0:raise ValueError('Negative equality count')
        eq=[quad() for _ in range(ne)];ni=int(row('inequalities')[0])
        if ni<0:raise ValueError('Negative inequality count')
        ineq=[quad() for _ in range(ni)];row('end_problem')
        if stream.read().strip():raise ValueError('Trailing model content')
    return dict(name=name,n=n,x0=x0,lb=lb,ub=ub,obj=obj,eq=eq,ineq=ineq)

def value(q,x):
    c,lin,h=q
    v=math.fsum([c]+[a*x[i] for i,a in lin]+[(.5 if i==j else 1)*a*x[i]*x[j] for i,j,a in h])
    if not math.isfinite(v):raise ValueError('Nonfinite function value')
    return v

def add_gradient(q,x,weight,terms):
    if weight==0:return
    _,lin,h=q
    for i,a in lin:terms[i].append(weight*a)
    for i,j,a in h:
        terms[i].append(weight*a*x[j])
        if i!=j:terms[j].append(weight*a*x[i])

def audit(model,endpoint,tolerance=1e-6):
    if not isinstance(endpoint,dict):raise ValueError('Endpoint absent')
    if endpoint.get('solver')=='Ipopt':
        endpoint=dict(endpoint)
        for source,target in [('equality_multipliers','certificate_equality_multipliers'),('inequality_multipliers','certificate_inequality_multipliers'),('lower_bound_multipliers','certificate_lower_multipliers'),('upper_bound_multipliers','certificate_upper_multipliers')]:
            endpoint[target]=endpoint[source]
    d=endpoint;n=model['n'];x=d['x'];y=d['certificate_equality_multipliers'];z=d['certificate_inequality_multipliers'];zl=d['certificate_lower_multipliers'];zu=d['certificate_upper_multipliers']
    if [len(a) for a in (x,y,z,zl,zu)]!=[n,len(model['eq']),len(model['ineq']),n,n]:raise ValueError('Certificate dimensions differ')
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for a in (x,y,z,zl,zu) for v in a):raise ValueError('Nonfinite certificate')
    terms=[[-l,u] for l,u in zip(zl,zu)];f=value(model['obj'],x);add_gradient(model['obj'],x,1,terms)
    primal=0.;compl=0.;dual=0.
    for q,mult in zip(model['eq'],y):
        primal=max(primal,abs(value(q,x)));add_gradient(q,x,mult,terms)
    for q,mult in zip(model['ineq'],z):
        g=value(q,x);primal=max(primal,g);compl=max(compl,abs(g*mult));dual=max(dual,-mult);add_gradient(q,x,mult,terms)
    for j,(l,u) in enumerate(zip(model['lb'],model['ub'])):
        dual=max(dual,-zl[j],-zu[j])
        if math.isfinite(l):primal=max(primal,l-x[j]);compl=max(compl,abs(zl[j]*(x[j]-l)))
        else:dual=max(dual,abs(zl[j]))
        if math.isfinite(u):primal=max(primal,x[j]-u);compl=max(compl,abs(zu[j]*(u-x[j])))
        else:dual=max(dual,abs(zu[j]))
        residual=math.fsum(terms[j])
        if not math.isfinite(residual):raise ValueError('Nonfinite stationarity')
        dual=max(dual,abs(residual))
    ok=all(math.isfinite(v) and 0<=v<=tolerance for v in (primal,dual,compl))
    return dict(scope='independent frozen V1/V2 QCQP; no native CUTEst audit',objective=f,feasibility=primal,dual=dual,complementarity=compl,strict=ok,tolerance=tolerance)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('endpoint',type=Path);a=p.parse_args()
    out=audit(read_model(a.input),json.loads(a.endpoint.read_text()));print(json.dumps(out,indent=2,allow_nan=False));return 0 if out['strict'] else 1
if __name__=='__main__':raise SystemExit(main())
