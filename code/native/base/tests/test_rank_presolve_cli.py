import json, pathlib, subprocess, sys, tempfile
binary=pathlib.Path(sys.argv[1]);toy=pathlib.Path(sys.argv[2])
help_result=subprocess.run([str(binary),'--help'],capture_output=True,text=True)
assert '--dca-rank-presolve 0|1' in help_result.stdout+help_result.stderr
assert '--dca-rank-implied-bounds 0|1' in help_result.stdout+help_result.stderr
bad_bound=subprocess.run([str(binary),'--problem',str(toy),'--dca-rank-implied-bounds','2'],capture_output=True,text=True)
assert bad_bound.returncode!=0 and 'must be 0 or 1' in bad_bound.stdout+bad_bound.stderr
bad=subprocess.run([str(binary),'--problem',str(toy),'--dca-rank-presolve','2'],capture_output=True,text=True)
assert bad.returncode!=0 and 'must be 0 or 1' in bad.stdout+bad.stderr
with tempfile.TemporaryDirectory() as td:
    td=pathlib.Path(td);endpoint=td/'endpoint.json';report=td/'reduction.json'
    args=[str(binary),'--problem',str(toy),'--algorithm','lazy-dca','--dca-engine','filter',
          '--dca-aggregate','--dca-aggregate-backend','native','--linear-backend','native',
          '--dca-rank-presolve','1','--dca-rank-presolve-report',str(report),
          '--display-level','0','--json',str(endpoint)]
    run=subprocess.run(args,capture_output=True,text=True)
    assert run.returncode==0,(run.stdout,run.stderr)
    data=json.loads(endpoint.read_text());proof=json.loads(report.read_text())
    assert data['rank_presolve_enabled'] is True
    assert proof['schema']=='exact-rank-presolve-v1'
    assert data['rank_presolve_original_variables']==len(data['x'])
    assert data['rank_presolve_initial_point_preserved'] is True
    bad=subprocess.run(args+['--dca-endpoint-start',str(td/'missing.bin')],capture_output=True,text=True)
    assert bad.returncode!=0 and 'cold start' in bad.stdout+bad.stderr
print('PASS rank CLI, flag validation, original-size endpoint, report, cold-only guard')
