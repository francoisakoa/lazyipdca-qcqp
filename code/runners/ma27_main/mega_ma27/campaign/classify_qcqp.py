#!/usr/bin/env python3
"""Classify actual frozen V1/V2 coefficients with the common campaign reader.

No optimizer, native status, sampled derivative, or solver residual is consulted.
Duplicate polynomial terms are combined exactly as binary64 rational numbers.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
AUDITOR_SHA='d500ff62371d7409180245e6f946c011f5b973537c600789d1c3b4671be3df8a'
DIMENSION_KEYS=('n','quadratic_inequalities','quadratic_equalities','linear_inequalities','linear_equalities')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()

def load_reader():
 p=HERE/'common/original_auditor.py'
 if sha(p)!=AUDITOR_SHA:raise ValueError('The common coefficient reader changed')
 spec=importlib.util.spec_from_file_location('_balanced_original_coefficient_reader',p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def canonical(q):
 c,linear,hessian=q;lin={};quad={}
 for i,a in linear:lin[i]=lin.get(i,Fraction(0))+Fraction(a)
 for i,j,a in hessian:
  ij=(min(i,j),max(i,j));v=Fraction(a)/(2 if i==j else 1)
  quad[ij]=quad.get(ij,Fraction(0))+v
 return Fraction(c),{i:a for i,a in lin.items() if a},{ij:a for ij,a in quad.items() if a}

def degree(cq):return 2 if cq[2] else 1 if cq[1] else 0

def polynomial_hash(cq):
 c,l,q=cq
 data=[str(c),[[i,str(a)]for i,a in sorted(l.items())],[[i,j,str(a)]for(i,j),a in sorted(q.items())]]
 return hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()

def classify(path, include_semantic=True):
 path=Path(path).expanduser().resolve(strict=True);reader=load_reader();m=reader.read_model(path)
 n=m['n'];fixed={i:a for i,(a,b) in enumerate(zip(m['lb'],m['ub'])) if a==b and math.isfinite(a)}
 # The frozen V1 reader/auditor treats any infinite bound token as absent.
 lower=[a if math.isfinite(a) else -math.inf for a in m['lb']]
 upper=[a if math.isfinite(a) else math.inf for a in m['ub']]
 bounds_valid=all(a<=b for a,b in zip(lower,upper))
 iq=eq=li=le=0;eqhash=[];iqhash=[];effective_iq=effective_eq=0;nonzeros=0
 for group in ('eq','ineq'):
  for q in m[group]:
   cq=canonical(q);dg=degree(cq);nonzeros+=len(cq[1])+len(cq[2])
   if group=='eq':eq+=dg==2;le+=dg<2
   else:iq+=dg==2;li+=dg<2
   effective=any(i not in fixed and j not in fixed for i,j in cq[2])
   if group=='eq':effective_eq+=effective
   else:effective_iq+=effective
   if include_semantic:(eqhash if group=='eq' else iqhash).append(polynomial_hash(cq))
 oq=canonical(m['obj']);od=degree(oq)
 regime='M' if iq and eq else 'I' if iq else 'E' if eq and od<2 else 'Q' if eq else 'A'
 dims=dict(n=n,quadratic_inequalities=iq,quadratic_equalities=eq,linear_inequalities=li,linear_equalities=le,
           objective_degree=od,lower_bounds=sum(math.isfinite(x)for x in m['lb']),upper_bounds=sum(math.isfinite(x)for x in m['ub']),fixed_variables=len(fixed),linear_and_quadratic_terms=nonzeros+len(oq[1])+len(oq[2]))
 result=dict(schema='coefficient-qcqp-classification-v1',input=str(path),input_sha256=sha(path),name=m['name'],regime=regime,dimensions=dims,
             reader_sha256=AUDITOR_SHA,bounds_valid=bounds_valid,
             bound_infinity_semantics='Any infinite V1/V2 bound token denotes an absent bound, matching the unchanged campaign auditor; the side supplies the infinity sign',
             effective_quadratic_inequalities_after_fixed_bounds=effective_iq,effective_quadratic_equalities_after_fixed_bounds=effective_eq,
             structural_rule='I iff at least one nonzero quadratic inequality and zero quadratic equalities; exact coefficient cancellation, no sampled derivatives',
             embedded_start_sha256=hashlib.sha256(json.dumps(m['x0'],separators=(',',':'),allow_nan=False).encode()).hexdigest(),
             start_hash_encoding='SHA256 compact JSON native binary64-number array, json separators comma/colon',
             start_primal_residual=max([0.]+[abs(reader.value(q,m['x0']))for q in m['eq']]+[reader.value(q,m['x0'])for q in m['ineq']]+[a-x for a,x in zip(m['lb'],m['x0']) if math.isfinite(a)]+[x-b for b,x in zip(m['ub'],m['x0'])if math.isfinite(b)]))
 if include_semantic:
  def bound(v):return float(v).hex() if math.isfinite(v) else '-inf' if v<0 else '+inf'
  data=dict(n=n,lb=[bound(x)for x in lower],ub=[bound(x)for x in upper],objective=polynomial_hash(oq),equalities=sorted(eqhash),inequalities=sorted(iqhash))
  result['canonical_model_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  result['canonical_model_scope']='Exact polynomial coefficients, bounds and variable order; names/starts/row order omitted. Constraint multiplicity retained. No claim to identify all algebraic reformulations.'
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='*',type=Path);p.add_argument('--root',action='append',type=Path,default=[]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 paths=list(a.paths)
 for root in a.root:paths.extend(root.rglob('*.qcq'))
 paths=sorted(set(x.resolve()for x in paths));a.out.mkdir(parents=True,exist_ok=True)
 if not paths:p.error('No .qcq input files selected')
 records=[]
 for x in paths:
  try:records.append(classify(x))
  except (ValueError,OSError,KeyError,OverflowError)as e:records.append(dict(input=str(x),error=str(e),qualified=False))
 (a.out/'CLASSIFICATION.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
 cols=['input','input_sha256','regime',*DIMENSION_KEYS,'objective_degree','bounds_valid','error']
 with (a.out/'CLASSIFICATION.csv').open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
  for r in records:w.writerow({k:(r.get('dimensions',{}).get(k)if k in DIMENSION_KEYS or k=='objective_degree'else r.get(k))for k in cols})
 print(json.dumps(dict(files=len(records),errors=sum('error'in r for r in records),by_regime={c:sum(r.get('regime')==c for r in records)for c in ['E','I','M','Q','A']})))
 return int(any('error'in r for r in records))
if __name__=='__main__':raise SystemExit(main())
