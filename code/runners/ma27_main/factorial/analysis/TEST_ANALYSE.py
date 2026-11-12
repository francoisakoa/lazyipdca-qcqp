#!/usr/bin/env python3
"""Meaningful synthetic checks of preregistered effects and missingness rules."""
from ANALYSER_FACTORIEL import CELLS,analyse,primary,contrast,CONTRASTS
import json,hashlib,subprocess,sys,tempfile
from pathlib import Path

def make(n=60):
    cases=[{'case_id':str(i),'dataset':'d'+str(i%3),'family':'f'+str(i%2),
            'regime':'I' if i%2==0 else 'M'} for i in range(n)]
    records={(c['case_id'],cell):{'strict':False} for c in cases for cell in CELLS}
    return cases,records

tests=[]
def test(name,callback):callback();tests.append(name)

def positive():
    c,r=make()
    for x in c:
        for cell in ('X2','H0'):r[x['case_id'],cell]['strict']=True
    p=primary(c,r);assert p['decision']=='positive_under_preregistered_rule'
    assert p['net_gain_bounds']==[60,60]
    assert all(p['conditions'].values())
    assert contrast(c,r,CONTRASTS['correction_average'])['lower_certificate_equivalents']==60
    assert contrast(c,r,CONTRASTS['controller_average'])['lower_certificate_equivalents']==0
    assert contrast(c,r,CONTRASTS['interaction'])['lower_certificate_equivalents']==0
test('Positive correction effect and zero controller/interaction with 3 dispersed collections',positive)

def concentration():
    c,r=make()
    for x in c:
        if x['dataset']=='d0':r[x['case_id'],'H0']['strict']=True
    p=primary(c,r);assert p['net_gain_bounds']==[20,20]
    assert not p['conditions']['no_dataset_above_half_total_net_gain_robust']
    assert p['decision']=='negative_under_preregistered_rule'
test('20 certificates concentrated in one collection fail concentration criterion',concentration)

def unknown():
    c,r=make()
    for x in c:
        r[x['case_id'],'H0']['strict']=True;r[x['case_id'],'X1']['strict']=None
    p=primary(c,r);assert p['net_gain_bounds']==[0,60]
    assert p['unknown_pairs']==60 and p['known_losses']==0
    assert p['decision'].startswith('inconclusive')
test('Unknown counterparts remain intervals; they are not classified as failures',unknown)

def impossible():
    c,r=make(18)
    for x in c:r[x['case_id'],'H0']['strict']=None
    p=primary(c,r);assert p['net_gain_bounds']==[0,18]
    assert p['decision']=='negative_under_preregistered_rule_for_every_possible_completion'
test('Unknown outcomes cannot rescue a net gain below 20',impossible)

def interaction():
    c,r=make(6)
    for x in c:r[x['case_id'],'H0']['strict']=True
    assert contrast(c,r,CONTRASTS['interaction'])['lower_certificate_equivalents']==6
    assert contrast(c,r,CONTRASTS['correction_average'])['lower_certificate_equivalents']==3
    assert contrast(c,r,CONTRASTS['controller_average'])['lower_certificate_equivalents']==3
test('Interaction and half-weighted main effects use the specified 2x2 signs',interaction)

def origins_and_timing():
    c,r=make(3)
    for x in c:
        r[x['case_id'],'R3'].update(strict=True,objective=1,seconds_process=.5,
          accepted_direction_origins={'majorant_primary':2,'majorant_with_soc':1})
        r[x['case_id'],'X1'].update(strict=True,objective=1,seconds_process=2)
    v,s=analyse(c,r)
    a=next(x for x in s if x['regime']=='I+M' and x['left']=='R3' and x['right']=='X1')
    assert a['common_strict']==3 and a['both_process_times_over_1s_pairs']==0
    o=v['I+M']['accepted_direction_origins'][0]
    assert o['pure_majorant_fraction_on_logged_updates']==2/3
    assert o['all_final_origin_counts']['majorant_with_soc']==3
test('Mixed SOC is not counted as pure majorant; BOTH times must exceed 1 second',origins_and_timing)

def ingestion_evidence():
    with tempfile.TemporaryDirectory(prefix='factorial-analysis-test-') as temporary:
        folder=Path(temporary)
        cases=[{'case_id':'c'+str(i),'dataset':'d'+str(i%3),'family':'f'+str(i%2),
                'regime':'I' if i<1037 else 'M','input_sha256':'input'+str(i),
                'start_binding':{'start_sha256':'start'+str(i)}} for i in range(1834)]
        contract=folder/'contract.json';contract.write_text(json.dumps({'cases':cases}))
        prereg=folder/'prereg.json';prereg.write_text(json.dumps({'inputs':{
          'contract_file_sha256':hashlib.sha256(contract.read_bytes()).hexdigest()},
          'primary':{'minimum_net_certificates':20}}))
        def row(c):return {'case_id':c['case_id'],'cell':'R3','run_id':'one-contemporary-run',
          'input_sha256':c['input_sha256'],'start_sha256':c['start_binding']['start_sha256'],
          'preregistration_sha256':hashlib.sha256(prereg.read_bytes()).hexdigest(),
          'objective_exact':True,'budget_verified':True,'trace_success':True,
          'strict_qcqp':True,'primal':0,'dual':1e-8,'complementarity':1e-8}
        rows=[row(c) for c in cases[:3]]
        rows[1].pop('budget_verified')
        rows[2]['dual']=1e-3
        source=folder/'observations.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        subprocess.run([sys.executable,str(Path(__file__).with_name('ANALYSER_FACTORIEL.py')),
            '--contract',str(contract),'--preregistration',str(prereg),'--observations',str(source),
            '--out',str(folder/'analysis')],check=True,stdout=subprocess.PIPE,text=True)
        report=json.loads((folder/'analysis'/'FACTORIAL_REPORT.json').read_text())
        result=report['results']['I+M']['outcomes']['R3']
        assert result=={'strict':1,'nonstrict':0,'indeterminate':1833,'planned':1834}
        assert len(report['verification_issues'])==2
test('End-to-end evidence gate: missing budget and inconsistent strict residuals remain unknown',ingestion_evidence)

Path(__file__).with_name('TEST_VALIDATION.json').write_text(json.dumps({'status':'PASS',
    'tests':tests,'actual_optimizer_calls':0,'synthetic_decisions_are_not_experimental_results':True},indent=2)+'\n')
print('PASS',len(tests),'preregistered-analysis checks')
