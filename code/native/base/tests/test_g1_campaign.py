import json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
backend_query=subprocess.run([sys.argv[1],'--linear-backends'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
assert backend_query.returncode==0,backend_query.stdout
if 'ldl' not in {item.strip() for item in backend_query.stdout.strip().split(',')}:
    print('[SKIP] historical G1 campaign requires the uncompiled SuiteSparse LDL backend')
    raise SystemExit(77)
def no_duplicates(items):
    d={}
    for k,v in items:
        assert k not in d,('duplicate JSON key',k)
        d[k]=v
    return d
with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)/'campaign'
    cmd=[sys.executable,str(ROOT/'python/benchmark_g1.py'),'--qcq',str(ROOT/'tests/regression/BT1.qcq'),'--binary',sys.argv[1],'--modes','legacy,atomic','--warmups','0','--repetitions','1','--time-limit','10','--output',str(out)]
    r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
    assert r.returncode==0,r.stdout
    d=json.loads((out/'benchmark_summary.json').read_text());assert d['correctness_pass']
    for name in ('legacy','atomic'):
        assert d['modes'][name]['all_certificates_independent']
    for p in out.glob('run_*.json'):
        if p.name.endswith('.timing.json'):continue
        d=json.loads(p.read_text(),object_pairs_hook=no_duplicates)
        if 'legacy' in p.name:assert d['atomic_engine_used'] is False
        else:assert d['atomic_engine_used'] is True
    for excluded in ['--initial-atoms','--quadratic-model']:
        r=subprocess.run([sys.argv[1],excluded,'2'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        assert r.returncode==2,r.stdout
print('g1 campaign, independent certificates, unique JSON keys, r11 exclusion PASS')
