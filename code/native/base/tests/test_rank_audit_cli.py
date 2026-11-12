import json
import pathlib
import subprocess
import sys
import tempfile
binary=pathlib.Path(sys.argv[1]); toy=pathlib.Path(sys.argv[2])
help_result=subprocess.run([str(binary),'--help'],text=True,capture_output=True,check=True)
assert '--dca-rank-audit-mode' in help_result.stdout+help_result.stderr
bad=subprocess.run([str(binary),'--problem',str(toy),'--dca-rank-audit-mode','incorrect'],text=True,capture_output=True)
assert bad.returncode and 'full, mapped or check' in bad.stdout+bad.stderr
with tempfile.TemporaryDirectory() as tmp:
    tmp=pathlib.Path(tmp); p=tmp/'duplicate.qcq'
    text=toy.read_text(); old=text[text.index('inequalities 1\n')+len('inequalities 1\n'):text.index('end_problem')]
    text=text.replace('inequalities 1\n'+old,'inequalities 2\n'+old+old);p.write_text(text)
    records=[]
    for mode in ('full','mapped','check'):
        out=tmp/(mode+'.json')
        args=[str(binary),'--problem',str(p),'--algorithm','lazy-dca','--dca-engine','filter',
              '--dca-aggregate','--dca-aggregate-backend','native','--linear-backend','native',
              '--dca-rank-presolve','1','--dca-rank-audit-mode',mode,
              '--display-level','0','--json',str(out)]
        proc=subprocess.run(args,text=True,capture_output=True)
        assert proc.returncode==0,(mode,proc.stdout,proc.stderr)
        r=json.loads(out.read_text());records.append(r)
        assert r['rank_presolve_duplicate_inequalities']==1
        assert r['rank_audit_mode']==mode
        assert r['rank_audit_calls']>0 and r['rank_audit_final_full_calls']==1
        if mode=='mapped':
            assert r['rank_audit_result_copies']==0 and r['rank_audit_lift_calls']==1
            assert r['rank_audit_value_reuses']>0 and r['rank_audit_full_calls']==0
    for r in records[1:]:
        for key in ('x','certificate_equality_multipliers','certificate_inequality_multipliers',
                    'certificate_lower_multipliers','certificate_upper_multipliers',
                    'status','objective','feasibility','dual_infeasibility_inf','complementarity_inf','scaled_nlp_error'):
            assert r[key]==records[0][key],key
print('PASS mapped/full/check CLI, invalid option, original certificate/export, no internal copies, identical solver endpoint')
