#!/usr/bin/env python3
"""Validate archived result/audit/provenance bindings; never optimize or re-audit endpoints."""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,json,math,zipfile

def sha(data):return hashlib.sha256(data).hexdigest()
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':')).encode())
def finite(v):return type(v)in(int,float)and math.isfinite(v)
def options(cmd):
 out={}
 for i,s in enumerate(cmd):
  if s.startswith('--'):
   if s in out:raise ValueError('Duplicate command option '+s)
   out[s]=cmd[i+1]if i+1<len(cmd)else None
 return out
def csvwrite(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--native',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 native={}
 with gzip.open(a.native,'rt')as f:
  for line in f:
   r=json.loads(line)
   if r['method']in('ipopt_mumps','ipopt_qcqp'):continue
   native.setdefault(r['case_id'],[]).append(r)
 issues=[];strict_issues=[];compact=[];filecounts=collections.Counter();counts=collections.Counter();own=collections.Counter();statuses=collections.Counter();technical=collections.Counter();own_disagreements=[];binding_disagreements=[]
 with zipfile.ZipFile(a.archive)as z:
  names=set(z.namelist());plan=json.loads(z.read('DEPOUILLEMENT/campaign/PLAN.json'));status=json.loads(z.read('DEPOUILLEMENT/campaign/STATUS.json'));completion=json.loads(z.read('DEPOUILLEMENT/campaign/COMPLETION.json'))
  cases={r['case_id']:r for r in plan['case_bindings']};methods=plan['configurations'];seen=set();archivefileinfo=[]
  expected=dict(plan['expected_options'])
  fixed=dict(hessian_approximation='exact',fixed_variable_treatment='make_parameter',nlp_scaling_method='gradient-based')
  tasknames=sorted(n for n in names if '/campaign/tasks/'in n and n.endswith('/RESULT.json'))
  for k,name in enumerate(tasknames,1):
   r=json.loads(z.read(name));folder=name.rsplit('/',1)[0];cid=r['case_id'];method=r['method'];c=cases[cid];conf=methods[method];classification=r['classification'];reasons=[]
   key=(cid,method)
   if key in seen:raise ValueError('Duplicate '+str(key))
   seen.add(key)
   if classification not in ('strict','nonstrict','indeterminate'):raise ValueError(classification)
   for field in ['dataset','regime','family','input_sha256','start_binding']:
    if r.get(field)!=c.get(field):reasons.append('plan_binding:'+field)
   for field in ['installation_sha256','binary_sha256','run_id']:
    if r.get(field)!=plan[field]:reasons.append('plan_identity:'+field)
   if r.get('backend')!=conf['backend']or r.get('strategy')!=conf['strategy']:reasons.append('strategy_backend')
   inputhash=r['input_sha256'];start=r['start_binding']['start_sha256']
   for n in native.get(cid,[]):
    ns=n.get('start_sha256')or n.get('start_binding',{}).get('start_sha256')
    if n.get('input_sha256')!=inputhash or ns!=start:binding_disagreements.append(dict(case_id=cid,native_config=n['config'],aligned_config=method,input_same=n.get('input_sha256')==inputhash,start_same=ns==start))
   local={}
   for filename in ['COMMAND.json','case.json','solver_resources.json','AUDIT.json','audit_resources.json']:
    path=folder+'/'+filename
    if path in names:
     raw=z.read(path);local[filename]=(json.loads(raw),sha(raw));filecounts[filename]+=1
    else:filecounts['missing:'+filename]+=1
   cmd=local.get('COMMAND.json',([],''))[0];opts=options(cmd)
   for field,value in {'--mu-strategy':conf['strategy'],'--linear-solver':conf['backend'],'--max-iter':'5000','--time-limit':'7200'}.items():
    if opts.get(field)!=value:reasons.append('command:'+field)
   for field in ['--problem','--json']:
    if field not in opts:reasons.append('command_absent:'+field)
   if any(f in opts for f in ['--endpoint-start','--warm-start-file']):reasons.append('external_start_override')
   rc=local.get('solver_resources.json',({},''))[0]
   if rc!=r.get('solver_resources'):reasons.append('resources_copy')
   resourcecmd=rc.get('command',[])
   if resourcecmd[:1]==['taskset']and len(resourcecmd)>3:
    if resourcecmd[3:]!=cmd:reasons.append('resource_command_binding')
   else:reasons.append('resource_affinity_command')
   if rc.get('command_sha256')!=sha(json.dumps(resourcecmd).encode()):reasons.append('resource_command_hash')
   if rc.get('threads')!=1 or rc.get('cpu')not in plan['cpus']:reasons.append('resource_threads_or_affinity')
   case=local.get('case.json',({},''))[0]
   for field in ['case_id','input_sha256','start_binding']:
    if case.get(field)!=r.get(field):reasons.append('case_binding:'+field)
   effective=r.get('effective_options')or{};provenance=r.get('provenance')or{};audit=local.get('AUDIT.json',({},''))[0]
   proof_reasons=[]
   for field,value in {**expected,**fixed,'mu_strategy':conf['strategy'],'linear_solver':conf['backend']}.items():
    if effective.get(field)!=value:proof_reasons.append('effective:'+field)
   if effective and digest(effective)!=r.get('effective_options_sha256'):proof_reasons.append('effective_options_hash')
   for field in ['aligned_options_verified','native_route_verified','backend_verified','budget_verified','option_file_disabled']:
    if r.get(field)is not True:proof_reasons.append('proof:'+field)
   if r.get('ipopt_version')!='3.14.11':proof_reasons.append('ipopt_version')
   if r.get('input_representation_overrides')!=[]:proof_reasons.append('input_representation_overrides')
   if not finite(rc.get('elapsed_seconds'))or not 0<=rc['elapsed_seconds']<=7200 or rc.get('censored')or rc.get('timed_out')or rc.get('interrupted'):proof_reasons.append('process_budget_or_censor')
   if type(rc.get('returncode'))is not int or rc['returncode']<0:proof_reasons.append('process_returncode')
   if not finite(r.get('native_elapsed_seconds'))or not 0<=r['native_elapsed_seconds']<=7200:proof_reasons.append('native_time')
   if type(r.get('native_iterations'))is not int or not 0<=r['native_iterations']<=5000:proof_reasons.append('native_iterations')
   if not audit:proof_reasons.append('original_audit_absent')
   else:
    if audit!=r.get('original_audit'):reasons.append('original_audit_copy')
    for field in ['case_id','method','dataset','regime','input_sha256']:
     if audit.get(field)!=r.get(field):reasons.append('audit_binding:'+field)
    if audit.get('endpoint_sha256')!=provenance.get('endpoint_sha256')and provenance:reasons.append('endpoint_hash_binding')
    if audit.get('audit_available')is not True:proof_reasons.append('audit_unavailable')
    if audit.get('original_bounds_used')is not True:proof_reasons.append('original_bounds_unverified')
    if audit.get('classification')!=classification:proof_reasons.append('audit_result_classification_difference')
    for field in ['primal','dual','complementarity']:
     if audit.get(field)!=r.get(field):reasons.append('residual_copy:'+field)
     if finite(audit.get(field))and audit.get('residual_upper',{}).get(field)!=audit[field]:reasons.append('interval_upper_copy:'+field)
   if classification=='strict':
    if r.get('strict_qcqp')is not True or audit.get('strict_qcqp')is not True or audit.get('original_KKT_endpoint_strict')is not True:reasons.append('strict_boolean')
    for field in ['primal','dual','complementarity']:
     if not finite(r.get(field))or not 0<=r[field]<=1e-6:reasons.append('strict_residual:'+field)
    if proof_reasons:reasons+=proof_reasons
    if reasons:strict_issues.append(dict(case_id=cid,method=method,reasons=reasons))
   else:
    technical.update(proof_reasons if classification=='indeterminate'else [])
   code=r.get('ipopt_application_status_code');named=r.get('ipopt_application_status');ownsuccess=type(code)is int and code in(0,1);positive=type(code)is int and code>=0
   if r.get('ipopt_own_success',False)!=ownsuccess:own_disagreements.append(dict(case_id=cid,method=method,recorded=r.get('ipopt_own_success'),derived=ownsuccess,code=code,status=named))
   statuses[method,str(code),str(named),classification]+=1
   for scope_type,scope in [('all','ALL'),('regime',r['regime']),('dataset',r['dataset']),('family',r['family'])]:
    counts[scope_type,scope,method,classification]+=1
    for label,b in [('own_success',ownsuccess),('positive_return',positive),('own_only',ownsuccess and classification!='strict'),('original_only',not ownsuccess and classification=='strict'),('positive_return_nonstrict',positive and classification=='nonstrict'),('positive_return_technical',positive and classification=='indeterminate')]:own[scope_type,scope,method,label]+=int(b)
   if reasons:issues.append(dict(case_id=cid,method=method,classification=classification,reasons=reasons))
   compact.append(dict(config=method,method=method,backend=conf['backend'],strategy=conf['strategy'],case_id=cid,dataset=r['dataset'],regime=r['regime'],family=r['family'],classification=classification,strict_qcqp=r.get('strict_qcqp'),input_sha256=inputhash,start_sha256=start,start_binding=r['start_binding'],primal=r.get('primal'),dual=r.get('dual'),complementarity=r.get('complementarity'),objective=r.get('objective'),seconds_process=r.get('seconds_process'),seconds_numerical=(r.get('internal_work')or{}).get('solve_seconds'),audit_seconds=r.get('audit_seconds'),iterations=r.get('iterations'),ipopt_application_status=named,ipopt_application_status_code=code,own_success=ownsuccess,positive_application_return=positive,options_verified=not any(x.startswith('effective:')or x.startswith('proof:')for x in proof_reasons),budget_verified=r.get('budget_verified'),native_status=r.get('status'),binary_sha256=r['binary_sha256'],installation_sha256=r['installation_sha256'],original_audit_sha256=local.get('AUDIT.json',({},None))[1],archived_proof_issues=proof_reasons,observation_proof_level='archived_result_audit_and_execution_receipts; raw_endpoint_log_and_coefficients_absent'))
   if k%4000==0:print('AUDIT',k,'/',len(tasknames),flush=True)
  expected_keys={(cid,m)for cid in cases for m in methods};missing=expected_keys-seen;extra=seen-expected_keys
  assert not missing and not extra
  grouped=[]
  for st,sc,m in sorted({key[:3]for key in counts}):
   strict=counts[st,sc,m,'strict'];non=counts[st,sc,m,'nonstrict'];tech=counts[st,sc,m,'indeterminate'];N=strict+non+tech
   grouped.append(dict(scope_type=st,scope=sc,configuration=m,planned=N,strict=strict,failure_to_certify=N-strict,audited_nonstrict=non,technical=tech,strict_rate=strict/N,**{lab:own[st,sc,m,lab]for lab in ['own_success','positive_return','own_only','original_only','positive_return_nonstrict','positive_return_technical']}))
  bymethod={m:{cl:counts['all','ALL',m,cl]for cl in ['strict','nonstrict','indeterminate']}for m in methods}
  comparison_to_status={m:all(bymethod[m][cl]==status['by_method'][m][cl]for cl in bymethod[m])for m in methods}
  report=dict(schema='ipopt-aligned-independent-archive-validation-v1',archive_sha256=sha(a.archive.read_bytes()),planned=16312,results=len(tasknames),matrix_complete=not missing and not extra,missing=len(missing),extra=len(extra),case_count=len(cases),classification_counts=bymethod,strict_validation_issues=strict_issues,result_binding_issues=issues,native_input_start_binding_issues=binding_disagreements,comparison_to_status=comparison_to_status,file_counts=dict(filecounts),raw_endpoints_present=sum(n.endswith('/endpoint.json')for n in names),raw_solver_logs_present=sum(n.endswith('/solver.log')for n in names),qcqp_inputs_present=sum(n.endswith('.qcq')for n in names),own_success_field_disagreements=own_disagreements,technical_proof_reason_counts=dict(technical),limitations=['No raw campaign endpoints, solver logs or frozen QCQP coefficients in supplied archive; no new numerical residual evaluation was possible.','Audited certificates are validated as archived original-KKT audits with their input/start/endpoint/provenance bindings; this is not a fresh endpoint audit.','Internal success means application status code 0 or 1. Nonnegative application exit codes 2..6 are separately counted and never called success.'],optimizer_invocations=0,new_endpoint_audits=0,classification_changes=0)
  (a.out/'ARCHIVE_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');csvwrite(a.out/'COVERAGE_RECOMPUTED.csv',grouped);csvwrite(a.out/'APPLICATION_STATUS_COUNTS.csv',[dict(configuration=m,code=code,status=n,classification=cl,count=v)for(m,code,n,cl),v in sorted(statuses.items())]);csvwrite(a.out/'OBSERVATION_BINDING_ISSUES.csv',issues);csvwrite(a.out/'STRICT_VALIDATION_ISSUES.csv',strict_issues)
  with gzip.open(a.out/'ALIGNED_4_CONFIGS_AUDITED_COMPACT.jsonl.gz','wt')as f:
   for r in compact:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
  print(json.dumps({'results':len(tasknames),'strict_issues':len(strict_issues),'binding_issues':len(issues),'native_binding_issues':len(binding_disagreements),'own_field_disagreements':len(own_disagreements),'file_counts':dict(filecounts),'counts':bymethod},indent=2))
if __name__=='__main__':main()
