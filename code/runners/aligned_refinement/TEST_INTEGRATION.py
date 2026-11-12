#!/usr/bin/env python3
"""Actual CLI-boundary and temporary orchestration checks; no optimizer runs.

Stage computations are replaced by clearly synthetic receipts in temporary
directories. Actual distributed scripts parse every orchestrator command, so
path, command-line and hand-off mistakes remain observable.
"""
from pathlib import Path
import argparse,contextlib,importlib.util,io,json,os,runpy,sys,tempfile,unittest,zipfile
from types import SimpleNamespace
from unittest.mock import patch

KIT=Path(__file__).resolve().parent

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

class ParserCaptured(BaseException):pass

def parse_actual(command):
    """Execute actual imports/parser, stopping immediately after parse_args."""
    script_index=next(i for i,x in enumerate(command)if str(x).endswith('.py'))
    script=Path(command[script_index]);args=list(map(str,command[script_index+1:]))
    if not script.is_file():raise AssertionError('Distributed stage path absent: '+str(script))
    original=argparse.ArgumentParser.parse_args
    def capture(parser,*unused,**ignored):raise ParserCaptured(original(parser,args))
    old=list(sys.path);sys.path.insert(0,str(script.parent))
    try:
        with patch.object(argparse.ArgumentParser,'parse_args',capture),contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script),run_name='__main__')
    except ParserCaptured as result:return script,result.args[0]
    finally:sys.path[:]=old
    raise AssertionError('No command-line parser reached: '+str(script))

def atom(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n')

class Integration(unittest.TestCase):
    def test_actual_4078_bindings_and_four_cell_queue(self):
        old=list(sys.path);sys.path.insert(0,str(KIT/'ipopt_aligned'))
        try:
            preparation=load('integration_actual_corpus',KIT/'ipopt_aligned/PREPARER_IPOPT_ALIGNE.py')
            controller=load('integration_actual_queue',KIT/'ipopt_aligned/CONTROLEUR_IPOPT_ALIGNE.py')
            cases=preparation.base.plan_cases(preparation.base.contract(KIT/'runtime_base/CONTRAT_4078.json.gz'))
            preparation.verify_corpus(cases,preparation.base.read(KIT/'evidence/CURRENT_PLAN_COMPACT.json'))
            jobs,proof=controller.make_plan(cases)
            self.assertEqual(len(cases),4078);self.assertEqual(len(jobs),16312)
            self.assertEqual(len({(c['case_id'],m)for c,m,b in jobs}),16312)
            self.assertEqual({m for c,m,b in jobs},set(controller.contract.CELLS))
            self.assertEqual({c['dataset']for c in cases}.__len__(),28)
            self.assertFalse(any('dey'in c['dataset'].lower()for c in cases))
            for i in range(0,len(jobs),4):
                block=jobs[i:i+4];self.assertEqual(len({c['case_id']for c,m,b in block}),1)
                self.assertEqual({m for c,m,b in block},set(controller.contract.CELLS))
        finally:sys.path[:]=old

    def test_export_cannot_promote_truncated_or_duplicate_observations(self):
        e=load('integration_export_integrity',KIT/'EXPORTER_RESULTATS.py')
        for name,rows,expected in [('complete',[('a','m1'),('a','m2')],True),('truncated',[('a','m1')],False),('duplicate',[('a','m1'),('a','m1')],False),('wrong_method',[('a','m1'),('a','undeclared')],False)]:
            with self.subTest(name=name),tempfile.TemporaryDirectory(prefix='integration-export-no-solver-')as td:
                root=Path(td);run=root/'run';campaign=run/'campaign';campaign.mkdir(parents=True)
                atom(campaign/'STATUS.json',dict(state='completed',done=2,total=2,by_method=dict(m1=dict(planned=1),m2=dict(planned=1))))
                original=''.join(json.dumps(dict(case_id=c,method=m))+'\n'for c,m in rows)
                (campaign/'OBSERVATIONS.jsonl').write_text(original)
                task=campaign/'tasks/a__m1';task.mkdir(parents=True);atom(task/'AUDIT.json',dict(synthetic_test_only=True));atom(task/'endpoint.json',dict(x=[1,2,3]));(task/'solver.log').write_text('synthetic-only-large-log')
                result=e.export(run,root/'out.zip');self.assertIs(result['complete'],expected)
                self.assertEqual((campaign/'OBSERVATIONS.jsonl').read_text(),original)
                with zipfile.ZipFile(root/'out.zip')as z:
                    receipt=json.loads(z.read('DEPOUILLEMENT/EXPORT.json'));self.assertIs(receipt['complete'],expected)
                    self.assertIn('DEPOUILLEMENT/campaign/tasks/a__m1/AUDIT.json',z.namelist())
                    self.assertNotIn('DEPOUILLEMENT/campaign/tasks/a__m1/endpoint.json',z.namelist())
                    self.assertNotIn('DEPOUILLEMENT/campaign/tasks/a__m1/solver.log',z.namelist())

    def test_allocations_and_scopes(self):
        p=load('integration_ipopt_pilot',KIT/'PILOTER_IPOPT.py');n=load('integration_native_pilot',KIT/'PILOTER.py')
        c=load('integration_ipopt_contract',KIT/'ipopt_aligned/IPOPT_ALIGNE_CONTRACT.py')
        self.assertEqual(p.CPUS,set(range(22)));self.assertEqual(n.CPUS,{23});self.assertFalse(p.CPUS&n.CPUS)
        self.assertEqual(p.TOTAL,16312);self.assertEqual(c.TASKS,4078*4);self.assertEqual(len(c.CELLS),4)
        self.assertEqual({(x['strategy'],x['backend'])for x in c.CELLS.values()},{(s,b)for s in ('adaptive','monotone')for b in ('ma27','mumps')})
        self.assertEqual(c.REGIMES,dict(I=1037,M=797,E=1704,Q=540))

    def test_ipopt_full_command_chain(self):
        p=load('integration_ipopt_chain',KIT/'PILOTER_IPOPT.py');calls=[]
        with tempfile.TemporaryDirectory(prefix='integration-only-no-optimizer-')as td:
            root=Path(td);out=root/'out';out.mkdir();dummy_lock=(root/'lock').open('w')
            def stage(self,name,command):
                script,a=parse_actual(command);calls.append((script.name,a));self.save(state='running',phase=name)
                if script.name=='BUILD_IPOPT_ALIGNED.py':atom(a.out/'BUILD.json',dict(synthetic_test_only=True))
                elif script.name=='QUALIFIER_IPOPT_ALIGNE.py':
                    selftest=Path(a.build);assert selftest==out/'build/BUILD.json'
                    atom(a.out/'QUALIFICATION.json',dict(passed=True,synthetic_test_only=True))
                elif script.name=='PREPARER_IPOPT_ALIGNE.py':
                    assert a.qualification==out/'qualification/QUALIFICATION.json'
                    atom(a.out/'PREPARATION.json',dict(synthetic_test_only=True));atom(a.out/'INSTALLATION_IPOPT_ALIGNE.json',dict(synthetic_test_only=True))
                elif script.name=='CONTROLEUR_IPOPT_ALIGNE.py':
                    assert a.installation==out/'preparation/INSTALLATION_IPOPT_ALIGNE.json'
                    atom(a.out/'STATUS.json',dict(state='completed',done=4,total=4))
                    atom(a.out/'PLAN.json',dict(synthetic_test_only=True))
                    (a.out/'OBSERVATIONS.jsonl').write_text(''.join(json.dumps(dict(case_id='SYNTHETIC',method=m))+'\n'for m in ('ipopt_ma27_monotone_aligned','ipopt_mumps_monotone_aligned','ipopt_ma27_adaptive_aligned','ipopt_mumps_adaptive_aligned')))
                elif script.name=='ANALYSER_IPOPT_ALIGNE.py':
                    assert a.plan==out/'campaign/PLAN.json';assert a.observations==out/'campaign/OBSERVATIONS.jsonl'
                    atom(a.out/'SYNTHETIC_ONLY.json',dict(no_optimizer_runs=True))
                elif script.name=='EXPORTER_RESULTATS.py':
                    exporter=load('integration_exporter',script);exporter.export(a.run,a.out,a.vectors)
                else:raise AssertionError('Unexpected stage: '+str(script))
            a=SimpleNamespace(out=out,root=root/'source',run=root/'previous',reprise_production=False,ipopt_library=None,ipopt_include=None,ipopt_prefix=None,hsl_library=None)
            with patch.object(p.Pipeline,'phase',stage),patch.object(p,'verify_kit',return_value='test-only'),patch.object(p,'pipeline_lock',return_value=dummy_lock),patch.object(p.os,'sched_getaffinity',return_value=set(range(24))),patch.object(p.os,'sched_setaffinity'),patch.object(p.signal,'signal'),patch.object(p,'TOTAL',4),contextlib.redirect_stdout(io.StringIO()):
                code=p.execute(a)
            self.assertEqual(code,0,json.loads((out/'IPOPT_PIPELINE_STATUS.json').read_text()).get('error'))
            self.assertEqual([n for n,a in calls],['BUILD_IPOPT_ALIGNED.py','QUALIFIER_IPOPT_ALIGNE.py','PREPARER_IPOPT_ALIGNE.py','CONTROLEUR_IPOPT_ALIGNE.py','ANALYSER_IPOPT_ALIGNE.py','EXPORTER_RESULTATS.py'])
            self.assertEqual(json.loads((out/'IPOPT_PIPELINE_STATUS.json').read_text())['state'],'completed')

    def test_native_full_command_chain(self):
        n=load('integration_native_chain',KIT/'PILOTER.py');calls=[]
        with tempfile.TemporaryDirectory(prefix='integration-only-no-native-')as td:
            out=Path(td);seal=out/'synthetic-registration.json';atom(seal,dict(synthetic_test_only=True))
            class Child:
                pid=999999999
                def __init__(self,command,**unused):
                    script,a=parse_actual(command);calls.append((script.name,a))
                    if script.name=='BUILD_EXPERIMENT.py':
                        assert a.jobs==1;atom(a.out/'BUILD_EXPERIMENT.json',dict(synthetic_test_only=True))
                    elif script.name=='PREPARER_EXPERIENCE.py':
                        if a.promouvoir:atom(out/'preparation/EXPERIMENT_INSTALLATION.json',dict(registration=str(seal),synthetic_test_only=True))
                        else:atom(a.out/'PREPARATION.json',dict(synthetic_test_only=True))
                    elif script.name=='QUALIFIER_EXPERIENCE.py':
                        assert a.jobs==1 and a.cpus=='23';atom(a.out/'GATE50.json',dict(passed=True,synthetic_test_only=True))
                    elif script.name=='CONTROLEUR_EXPERIENCE.py':atom(a.out/'STATUS.json',dict(state='completed',done=14672,total=14672));(a.out/'RESULTS.jsonl').write_text('')
                    elif script.name=='ANALYSER_RAFFINEMENT.py':
                        assert a.preregistration==seal and a.observations==out/'campaign/RESULTS.jsonl'
                        atom(a.out/'SYNTHETIC_ONLY.json',dict(no_optimizer_runs=True))
                    else:raise AssertionError('Unexpected native stage '+str(script))
                def poll(self):return 0
                def wait(self):return 0
            a=SimpleNamespace(out=out,root=out/'source',run=out/'previous',jobs_build=1,reprise_production=False,boost_include=None,hsl_library=None,mumps_reference_binary=None)
            with patch.object(n,'verify_kit',return_value='test-only'),patch.object(n.subprocess,'Popen',Child),patch.object(n.os,'sched_getaffinity',return_value=set(range(24))),patch.object(n.os,'sched_setaffinity'),patch.object(n.signal,'signal'),contextlib.redirect_stdout(io.StringIO()):
                code=n.executer(a)
            self.assertEqual(code,0)
            self.assertEqual([x[0]for x in calls],['BUILD_EXPERIMENT.py','PREPARER_EXPERIENCE.py','QUALIFIER_EXPERIENCE.py','PREPARER_EXPERIENCE.py','CONTROLEUR_EXPERIENCE.py','ANALYSER_RAFFINEMENT.py'])
            self.assertEqual(json.loads((out/'PIPELINE_STATUS.json').read_text())['state'],'completed')

if __name__=='__main__':unittest.main(verbosity=2)
