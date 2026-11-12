#!/usr/bin/env python3
"""No optimization: check exact source anchors, generic policy, profiles and plan."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tempfile,sys,threading
import PROFILES_BOUNDS as p
import PATCH_SOURCE
HERE=Path(__file__).resolve().parent
CPP=r'''#include "research_equality_implied_bounds.hpp"
#include <cassert>
#include <iostream>
int main(){using lazyipdca::qcqp::research::quadratic_equality_implied_bounds_enabled;
int checks=0;
for(bool master:{false,true})for(const char* engine:{"majorants","filter"})
for(const char* mode:{"off","HH","H0","HS","SH","SS"}){
bool historical=std::string(mode)=="HH"||std::string(mode)=="H0"||std::string(mode)=="HS"||std::string(mode)=="SH";
for(const char* state:{"profile","keep","remove"}){
bool expected=master&&std::string(engine)=="majorants"&&(std::string(state)=="remove"||(std::string(state)=="profile"&&historical));
assert(quadratic_equality_implied_bounds_enabled(master,engine,mode,state)==expected);++checks;}}
bool rejected=false;try{quadratic_equality_implied_bounds_enabled(true,"majorants","off","wrong");}catch(const std::invalid_argument&){rejected=true;}assert(rejected);++checks;
std::cout<<checks<<" policy checks passed\n";
}
'''
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--syntax',action='store_true');a=ap.parse_args()
    if a.out.exists():raise ValueError('Répertoire de validation neuf requis')
    a.out.mkdir(parents=True);src=a.out/'research_source';receipt=PATCH_SOURCE.patch(src)
    (a.out/'test_policy.cpp').write_text(CPP)
    cc=['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(src/'src'),str(a.out/'test_policy.cpp'),'-o',str(a.out/'test_policy')]
    subprocess.run(cc,check=True);unit=subprocess.check_output([str(a.out/'test_policy')],text=True)
    checks=0
    for backend in p.BACKENDS:
        for policy in p.POLICIES:
            keep=p.command(f'{policy}_keep_{backend}','binary','input','endpoint','hsl');remove=p.command(f'{policy}_remove_{backend}','binary','input','endpoint','hsl')
            differences=[(x,y)for x,y in zip(keep,remove)if x!=y]
            assert differences==[('keep','remove')]
            for key,value in [('--time-limit','7200'),('--max-outer','5000'),('--restoration-iters','600'),('--shared-restoration-budget','1'),('--native-al-fallback','off'),('--dca-rank-presolve','1'),('--dca-rank-implied-bounds','1')]:assert keep[keep.index(key)+1]==value
            checks+=1
    contract=json.loads((HERE/'CONTRAT_ASU_503.json').read_text());assert len(contract['cases'])==503;assert len(p.CELLS)==12
    import BOUNDS
    pilot=BOUNDS.selected_cases(12);assert len(pilot)==12;assert sum(c['dimensions']['n']==32 for c in pilot)==2
    fixtures=a.out/'SYNTHETIC_CONTROLLER_FIXTURES';fixtures.mkdir()
    native=dict(schema='exact-rank-presolve-v1',original_variables=64,reduced_variables=64,original_equalities=32,original_inequalities=4,reduced_inequalities=4,
      equality_implied_bounds=[[0,-1,0],[0,1,0]],implied_bounds=[],
      research_equality_implied_bounds='remove',quadratic_equality_rule_enabled=True,
      radial_inequality_master_enabled=True,affine_equality_rule_enabled=False,
      equality_implied_bound_sides=2,implied_bound_sides=2)
    report=fixtures/'endpoint.json.rank_presolve.json';report.write_text(json.dumps(native))
    (fixtures/'case.json').write_text(json.dumps(pilot[0]))
    # The synthetic proof dimensions intentionally follow a64-variable case.
    (fixtures/'case.json').write_text(json.dumps(next(x for x in pilot if x['dimensions']['n']==64)))
    ep=dict(outer_iterations=1,restoration_iterations=0,elapsed_seconds=7200.677,
      native_al_fallback=dict(enabled=False,attempted=False,accepted=False),
      linear_solver_backend='inequality-majorants/mumps',lazy_newton_pivoted_fallbacks=0)
    c=BOUNDS.BoundCampaign.__new__(BOUNDS.BoundCampaign);c.stop=threading.Event();c.out=fixtures
    proof=c.check_endpoint(ep,'SS_remove_mumps','mumps',fixtures)
    assert proof['budget_verified']is False and not c.stop.is_set()
    ep['elapsed_seconds']=.1;proof=c.check_endpoint(ep,'SS_remove_mumps','mumps',fixtures)
    assert proof['budget_verified']is True and not c.stop.is_set()
    report.unlink();proof=c.check_endpoint(ep,'SS_remove_mumps','mumps',fixtures)
    assert proof['budget_verified']is False and not c.stop.is_set()
    native['quadratic_equality_rule_enabled']=False;report.write_text(json.dumps(native))
    rejected=False
    try:c.check_endpoint(ep,'SS_remove_mumps','mumps',fixtures)
    except ValueError:rejected=True
    assert rejected and c.stop.is_set()
    from NATIVE_PROOF import endpoint_core,validate_side_report
    import copy
    native['quadratic_equality_rule_enabled']=True
    assert validate_side_report(native)['equality_sides']==2
    proof_rejections=0
    for mutation in ['missing_array','wrong_count','duplicate','bad_origin','keep_removal']:
        broken=copy.deepcopy(native)
        if mutation=='missing_array':del broken['equality_implied_bounds']
        elif mutation=='wrong_count':broken['equality_implied_bound_sides']=64
        elif mutation=='duplicate':broken['equality_implied_bounds'][1]=broken['equality_implied_bounds'][0]
        elif mutation=='bad_origin':broken['equality_implied_bounds'][0][2]=999
        try:validate_side_report(broken,keep=mutation=='keep_removal')
        except ValueError:proof_rejections+=1
        else:raise AssertionError('Invalid native proof admitted: '+mutation)
    a_core={'x':[0.0,-0.0,1.0],'outer_iterations':8,'status':'solved','elapsed_seconds':1.}
    b_core=copy.deepcopy(a_core);b_core['elapsed_seconds']=3.;assert endpoint_core(a_core)==endpoint_core(b_core)
    exact_checks=0
    for field in ['x','outer_iterations','status']:
        b_core=copy.deepcopy(a_core)
        if field=='x':b_core['x'][1]=0.0
        elif field=='outer_iterations':b_core[field]=9
        else:b_core[field]='failed'
        assert endpoint_core(a_core)!=endpoint_core(b_core);exact_checks+=1
    syntax=[]
    if a.syntax:
        for name in ['rank_presolve.cpp','main.cpp']:
            cmd=['g++','-std=c++17','-fsyntax-only','-I'+str(src/'include'),'-I'+str(src/'src'),str(src/'src'/name)]
            r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(a.out/(name+'.syntax.log')).write_text(r.stdout)
            assert r.returncode==0,r.stdout[-2000:];syntax.append(name)
    result=dict(schema='bound-removal-kit-local-validation-v2',passed=True,generic_cpp_policy_checks=int(unit.split()[0]),policy_output=unit.strip(),same_policy_on_off_profiles_checked=checks,
      changed_source_files=receipt['changed_files'],syntax_checked_translation_units=syntax,plan_cases=503,cells=12,tasks=6036,
      synthetic_controller_checks=4,native_proof_rejections=proof_rejections,exact_nonclock_mutation_checks=exact_checks,pilot_selection_checked=True,
      original_source_preserved=True,optimizer_invocations=0,ASU_solver_runs=0,full_native_link_tested=False,TRX_runtime_qualified=False,
      scope='Generic gate compiled; unchanged same-policy commands and exact patch anchors checked. Native translation units/link and endpoint/campaign qualification are separate.')
    (a.out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
