"""Real native solve from a singular cone apex, then independent original audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from core2_audit import read_model, audit

binary=Path(sys.argv[1]).resolve()
fixture=Path(__file__).parent/'regression/homogeneous_spectral_cone.qcq'
source=fixture.read_bytes()
with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    for number,label in enumerate(['anonymous_cone','unrelated_label']):
        model=tmp/(str(number)+'.qcq')
        model.write_bytes(source.replace(b'anonymous_cone',label.encode()))
        before=hashlib.sha256(model.read_bytes()).hexdigest()
        endpoint=tmp/(str(number)+'.json')
        command=[str(binary),'--problem',str(model),'--json',str(endpoint),
            '--algorithm','lazy-dca','--dca-engine','majorants',
            '--linear-backend','mumps','--linear-metis','1',
            '--native-al-fallback','off','--standard-r10-policy','auto',
            '--termination-test','project','--max-outer','3000','--time-limit','20',
            '--feas-tol','1e-6','--opt-tol','1e-6','--dual-inf-tol','1e-6',
            '--compl-inf-tol','1e-6','--acceptable-iter','0','--display-level','2']
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                           text=True,env=env,timeout=30)
        if run.returncode:raise AssertionError(run.stdout)
        result=json.loads(endpoint.read_text())
        check=audit(read_model(model),result)
        assert check['strict'],check
        assert abs(check['objective']+1)<2e-6,check
        assert 'HOMOGENEOUS-SPECTRAL-ENVELOPE' in run.stdout,run.stdout
        assert hashlib.sha256(model.read_bytes()).hexdigest()==before
        assert result['native_al_fallback']['attempted'] is False
    model=Path(__file__).parent/'regression/homogeneous_near_psd40.qcq'
    endpoint=tmp/'near_psd40.json'
    # Same frozen native settings; this full 40-coordinate block needs no
    # fill, but its Gershgorin deficit is far larger than its negative part.
    command[command.index('--problem')+1]=str(model)
    command[command.index('--json')+1]=str(endpoint)
    run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                       text=True,env=env,timeout=30)
    if run.returncode:raise AssertionError(run.stdout)
    result=json.loads(endpoint.read_text());check=audit(read_model(model),result)
    assert check['strict'],check
    assert 19.8<check['objective']<20.1,check
    assert 'HOMOGENEOUS-SMALL-SPECTRUM' in run.stdout,run.stdout
    assert result['dca_filter_rho_inequalities_max']<0.01,result['dca_filter_rho_inequalities_max']
print('homogeneous_spectral_envelope PASS; two names, independent KKT, original input unchanged')
