#!/usr/bin/env python3
"""Extract the production GR delta lambda and compare coefficients to Fraction.

No optimizer is imported or invoked. Mock objects provide storage only; the
actual Quadratic method, merit overload and engine lambda execute unchanged.
"""
import argparse, hashlib, json, math, random, subprocess
from fractions import Fraction
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cxx',default='g++')
a=ap.parse_args();src=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
atomic=(src/'src/atomic_newton.cpp').read_text();engine=(src/'src/dca_inequality_majorant_engine.hpp').read_text()
def block(text, marker):
    start=text.index(marker);brace=text.index('{',start);depth=1;end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]
method=block(atomic,'  long double original_difference(')
helper=block(engine,'            auto gr_actual_change=')+';'
assert 'gr_change=gr_actual_change(trial,actual_primal,trial_slacks' in engine
assert 'soc_change=gr_actual_change(soc_state,soc_primal,soc_slacks' in engine
code=r'''
#include "lazyipdca_qcqp/primal_dual_merit.hpp"
#include "coefficient_quadratic_increment.hpp"
#include <vector>
#include <utility>
#include <iostream>
#include <iomanip>
#include <stdexcept>
namespace detail=lazyipdca::qcqp::detail;
using Vec=std::vector<double>;
struct Entry{int i,j;double v;};
struct Quadratic{
 double scale=1.;std::vector<std::pair<int,double>> linear,signed_square;
 std::vector<Entry> hessian,base_hessian;
''' +method+r'''
};
struct Workspace{Vec x,s;int me=0,mi=0;std::vector<Quadratic> equalities,inequalities;std::vector<int> active,inequality_kind;};
static int checks=0;
static void require(bool p){++checks;if(!p)throw std::runtime_error("actual wiring mismatch");}
static void wiring(){
 Workspace w;w.x={1.,2.,2.};w.s={1.,2.,3.};w.me=1;w.mi=3;
 w.equalities.resize(1);w.equalities[0].linear={{2,1.}};w.equalities[0].hessian={{0,1,-1.}};
 w.inequalities.resize(3);w.inequalities[0].linear={{0,1.}};w.active={1,1,0};w.inequality_kind={0,-1,0};
 struct{bool gr_coefficient_constraint_increments=true;}st;
 detail::gr::State gr_anchor;gr_anchor.c={0,1};gr_anchor.y={7,11};gr_anchor.s={1,2};gr_anchor.b={.25};
 detail::gr::Parameters gr_parameters;gr_parameters.mu=.1;gr_parameters.rho=.5;gr_parameters.center={7,11};gr_parameters.kappa=2;
''' +helper+r'''
 Vec point={std::nextafter(1.,2.),2.,2.},slacks={std::nextafter(1.,2.),2.,99.};
 auto trial=gr_anchor;trial.s[0]=slacks[0];trial.b[0]=.5;
 const long double dx=static_cast<long double>(point[0])-1.;
 auto expected=detail::gr::change_with_constraint_delta(gr_anchor,trial,gr_parameters,0,{-2*dx,2*dx});
 auto actual=gr_actual_change(trial,point,slacks,0);
 require(actual.valid&&actual.value==expected.value);
 require(actual.center==expected.center&&actual.constraint==expected.constraint&&actual.shifted==expected.shifted);
 require(actual.hard_bounds==expected.hard_bounds);
 st.gr_coefficient_constraint_increments=false;
 require(gr_actual_change(trial,point,slacks,0).value==detail::gr::change(gr_anchor,trial,gr_parameters,0).value);
 st.gr_coefficient_constraint_increments=true;
 bool rejected=false;try{gr_actual_change(trial,point,Vec{},0);}catch(const std::invalid_argument&){rejected=true;}
 require(rejected);
 // SOC-like arbitrary staged point/slacks: no alpha or nominal ds exists.
 point={1.5,2.5,2.125};slacks={1.125,4.,99.};trial.s={1.125,4.};
 expected=detail::gr::change_with_constraint_delta(gr_anchor,trial,gr_parameters,0,{-1.625L,.625L});
 actual=gr_actual_change(trial,point,slacks,0);require(actual.value==expected.value);
}
int main()try{
 wiring();int count;std::cin>>count;std::cout<<"wiring_checks "<<checks<<"\n";
 for(int k=0;k<count;++k){
  int n,nl,nh,ns;double scale;std::cin>>n>>nl>>nh>>ns>>scale;
  Vec x(n),t(n);for(double&v:x)std::cin>>v;for(double&v:t)std::cin>>v;
  Quadratic q;q.scale=scale;
  for(int i=0;i<nl;++i){int j;double c;std::cin>>j>>c;q.linear.push_back({j,c});}
  for(int i=0;i<nh;++i){int j,l;double c;std::cin>>j>>l>>c;q.hessian.push_back({j,l,c});}
  for(int i=0;i<ns;++i){int j;double c;std::cin>>j>>c;q.signed_square.push_back({j,c});}
  q.base_hessian=q.hessian;
  std::cout<<std::setprecision(25)<<q.original_difference(x,t)<<"\n";
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
'''
(out/'actual_wiring.cpp').write_text(code)
argv=[a.cxx,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(src/'include'),'-I'+str(src/'src'),str(out/'actual_wiring.cpp'),'-o',str(out/'actual_wiring')]
compiled=subprocess.run(argv,text=True,capture_output=True)
if compiled.returncode:print(compiled.stdout+compiled.stderr);raise SystemExit(compiled.returncode)
rng=random.Random(20261029);cases=[]
def add(x,t,l,h,ss=(),scale=1.):cases.append((x,t,l,h,list(ss),scale))
add([1.],[math.nextafter(1.,2.)],[(0,2.**-20)],[])
add([1e16],[1e16+.1],[(0,1.)],[])
add([0.,0.,0.],[1.,1.,1.],[(0,1e30),(1,1.),(2,-1e30)],[])
for unused in range(125):
    x=[math.ldexp(rng.uniform(-2,2),rng.randrange(-40,40)) for _ in range(3)]
    t=[math.nextafter(v,math.inf if rng.randrange(2) else -math.inf) if rng.randrange(2) else v*rng.uniform(-1.5,1.5) for v in x]
    l=[(j,math.ldexp(rng.uniform(-2,2),rng.randrange(-40,40))) for j in range(3)]
    h=[(i,j,math.ldexp(rng.uniform(-2,2),rng.randrange(-40,40))) for i in range(3) for j in range(i,3)]
    ss=[(j,math.ldexp(rng.uniform(-2,2),rng.randrange(-40,40))) for j in range(3)]
    add(x,t,l,h,ss,math.ldexp(1.,rng.randrange(-20,20)))
lines=[str(len(cases))];exact=[];magnitudes=[]
F=Fraction.from_float
for x,t,l,h,ss,scale in cases:
    lines.append(f'{len(x)} {len(l)} {len(h)} {len(ss)} {scale!r}')
    lines.append(' '.join(map(repr,x)));lines.append(' '.join(map(repr,t)))
    terms=[]
    for j,c in l:lines.append(f'{j} {c!r}');terms.append(F(c)*(F(t[j])-F(x[j])))
    for i,j,c in h:
        lines.append(f'{i} {j} {c!r}');terms.append(F(c)*(F(t[i])*F(t[j])-F(x[i])*F(x[j]))/(2 if i==j else 1))
    for j,c in ss:lines.append(f'{j} {c!r}');terms.append(F(c)*(F(t[j])*abs(F(t[j]))-F(x[j])*abs(F(x[j]))))
    exact.append(sum(terms,F(0))*F(scale));magnitudes.append(sum(map(abs,terms),F(0))*abs(F(scale)))
run=subprocess.run([str(out/'actual_wiring')],input='\n'.join(lines)+'\n',capture_output=True,text=True)
if run.returncode:print(run.stdout+run.stderr);raise SystemExit(run.returncode)
output=run.stdout.splitlines();assert len(output)==len(cases)+1
worst=0.;errors=[]
for i,(text,ref,mag) in enumerate(zip(output[1:],exact,magnitudes)):
    actual=Fraction(text);error=abs(actual-ref)
    # Extended precision bound uses actual coefficient-increment terms;
    # the targeted sub-1e-20 example additionally requires relative accuracy.
    allowed=max(abs(ref),mag)*Fraction(1,2**56)
    if i==0:allowed=abs(ref)*Fraction(1,2**60)
    if error>allowed:errors.append({'case':i,'actual':text,'reference':str(ref),'error':str(error),'allowed':str(allowed)})
    if mag:worst=max(worst,float(error/mag))
receipt={'schema':'R12-actual-coefficient-GR-increment-wiring-v1','success':not errors,'cases':len(cases),'wiring':output[0],'arithmetic_reference':'exact Fraction(binary64 stored coefficients and actual rounded points)','maximum_absolute_error_over_term_magnitude':worst,'failed_cases':errors,'compile_returncode':compiled.returncode,'run_returncode':run.returncode,'optimizer_calls':0,'actual_engine_lambda':helper,'actual_quadratic_method':method,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src/'src/atomic_newton.cpp',src/'src/dca_inequality_majorant_engine.hpp',src/'include/lazyipdca_qcqp/primal_dual_merit.hpp',src/'src/coefficient_quadratic_increment.hpp']}}
(out/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('actual_engine_lambda','actual_quadratic_method','source_sha256')},indent=2))
raise SystemExit(0 if receipt['success'] else 1)
