#!/usr/bin/env python3
"""Exact dyadic complex factor-lift fixtures for allocation regression tests.

These synthetic test models exercise the same affine-lift algebra; they are
not additional public benchmark instances and do not replace author data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def build(N, target):
    if N<4 or N&(N-1) or math.isqrt(N)**2!=N:
        raise ValueError('N must be a power of four')
    F=2; wdim=2*N; u0=wdim; t=u0+2*N*F; n=t+1
    rows=[]
    def quad(name,constant,linear,hessian):
        rows.extend(('quadratic','name '+name,'constant '+str(constant),'linear_nnz '+str(len(linear))))
        rows.extend(f'{j} {value:.17g}' for j,value in linear)
        rows.append('hessian_nnz '+str(len(hessian)))
        rows.extend(f'{i} {j} {value:.17g}' for i,j,value in hessian)
        rows.append('end_quadratic')
    # w=1+0i, u=D*w. The first Hadamard row contains all ones; all
    # remaining rows sum to zero. Each complex row has both real and
    # imaginary nonzero factors, so all 2N source coordinates occur.
    values=[1.0]*N+[0.0]*N
    for m in range(N):
        for k in range(F):
            factor=(1.0 if k==0 else .5)/math.isqrt(N)
            real=N*factor if m==0 else 0.0
            values.extend((real,.25*real))
    port_power=N*(1.0+.25*.25)*(1+.5*.5)
    values.append(port_power+1)
    rows.extend(('LAZYIPDCA_QCQP_V2','name exact_dyadic_complex_factor_lift_'+str(N),'n '+str(n),
        'x0 '+' '.join(format(v,'.17g') for v in values),'lb '+' '.join(['-inf']*t+['0']),
        'ub '+' '.join(['inf']*n),'objective'))
    quad('worst_port_power',0,[(t,1)],[])
    rows.append('equalities '+str(N+2*N*F))
    for j in range(N):quad('modulus_'+str(j),-1,[],[(j,j,2),(N+j,N+j,2)])
    for m in range(N):
        for k in range(F):
            factor=(1.0 if k==0 else .5)/math.isqrt(N)
            for part in range(2):
                coeff=[]
                for j in range(N):
                    h=-1 if (m&j).bit_count()%2 else 1
                    a=h*factor;b=.25*a
                    coeff.append((j,-a if part==0 else -b))
                    coeff.append((N+j,b if part==0 else -a))
                coeff.append((u0+2*(m*F+k)+part,1))
                quad('factor_'+str(m)+'_'+str(k)+'_'+str(part),0,sorted(coeff),[])
    rows.append('inequalities '+str(N+1))
    for m in range(N):
        quad('port_'+str(m),0,[(t,-1)],[(u0+2*(m*F+k)+part,u0+2*(m*F+k)+part,2)
            for k in range(F) for part in range(2)])
    quad('gain',N/1.5,[(j,-1) for j in range(N)],[])
    rows.append('end_problem')
    target.write_text('\n'.join(rows)+'\n')
    info=dict(N=N,F=F,n=n,equalities=N+2*N*F,inequalities=N+1,
        affine_factor_rows=2*N*F,proposed_factor_dependency_width=2*N,
        projected_diagonal_map_entries=(2*N*F)*(2*N)*(2*N+1)//2+2*N+1,
        projected_diagonal_map_bytes_24=((2*N*F)*(2*N)*(2*N+1)//2+2*N+1)*24,
        x0_objective=port_power+1,exact_dyadic_D=True,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),test_fixture_only=True)
    target.with_suffix('.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=64);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True);build(a.N,a.out)
