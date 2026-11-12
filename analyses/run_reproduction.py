#!/usr/bin/env python3
"""Reproduce R18 descriptive analyses offline; never invoke an optimizer."""
from pathlib import Path
import argparse,gzip,hashlib,json,subprocess,sys,time

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,default=None)
 p.add_argument('--archive',type=Path,help='optional original Ipopt ZIP for archived-report audit')
 a=p.parse_args();base=Path(__file__).resolve().parent;out=(a.out or base/'reproduced').resolve();out.mkdir(parents=True,exist_ok=True)
 started=time.monotonic();compact=base/'pairs/OBSERVATIONS_16_CONFIGS.jsonl.gz';q=out/'pairs';ref=out/'reference';nat=out/'native_controls';tab=out/'manuscript_tables'
 commands=[
  [sys.executable,str(base/'pairs/analyze_aligned_pairs.py'),'--compact',str(compact),'--native-details',str(base/'pairs/NATIVE_R3_MA27_DETAILS.json'),'--out',str(q)],
  [sys.executable,str(base/'pairs/build_full_tables.py'),'--input-directory',str(q),'--out',str(q)],
  [sys.executable,str(base/'pairs/reference/analyse_reference_values.py'),'--compact',str(compact),'--out',str(ref)],
  [sys.executable,str(base/'pairs/reference/build_reference_tables.py'),'--input-directory',str(ref),'--out',str(ref)],
  [sys.executable,str(base/'native_controls/scripts/analyze_quartet_objectives.py'),'--compact',str(compact),'--out',str(nat)],
  [sys.executable,str(base/'native_controls/scripts/audit_quartet_coverage.py'),'--compact',str(compact),'--ma27-plan',str(base/'native_controls/metadata/profiles_ma27.json'),'--mumps-context',str(base/'native_controls/metadata/profiles_mumps.json'),'--out',str(nat)],
  [sys.executable,str(base/'native_controls/scripts/compare_asu_objective_distributions.py'),'--compact',str(compact),'--out',str(nat)],
  [sys.executable,str(base/'manuscript_tools/complete_tables.py'),'--compact',str(compact),'--out',str(tab)],
  [sys.executable,str(base/'refinement/analyze_refinement_pairs.py'),'--compact',str(base/'refinement/REFINEMENT_OBSERVATIONS_8_CONFIGS.jsonl.gz'),'--study',str(base/'refinement/STUDY.json'),'--out',str(out/'refinement')],
  [sys.executable,str(base/'refinement/render_compact_tables.py'),'--summary',str(out/'refinement/REFINEMENT_PAIR_SUMMARY.json'),'--out',str(out/'refinement')],
  [sys.executable,str(base/'campaigns_R18/recompute_campaigns.py'),'--out',str(out/'campaigns_R18')],
 ]
 if a.archive:
  native=out/'NATIVE_12_CONFIGS_AUDIT_INPUT.jsonl.gz'
  with gzip.open(compact,'rt')as src,gzip.open(native,'wt')as dst:
   for line in src:
    if not json.loads(line)['config'].startswith('ipopt_'):dst.write(line)
  commands.append([sys.executable,str(base/'audit/audit_aligned_archive.py'),'--archive',str(a.archive.resolve()),'--native',str(native),'--out',str(out/'archive_audit')])
 for i,cmd in enumerate(commands,1):
  with (out/f'step_{i:02}.log').open('w')as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
  print(f'Analyse {i}/{len(commands)} terminée',flush=True)
 checks=[]
 scopes=[(base/'pairs',q),(base/'pairs/reference/output',ref),(base/'native_controls/expected',nat),(base/'manuscript_tools/expected',tab),(base/'refinement/expected',out/'refinement')]
 for expected_dir,reproduced_dir in scopes:
  for expected in sorted(expected_dir.glob('*.tex')):
   got=reproduced_dir/expected.name
   checks.append({'table':str(expected.relative_to(base)),'same_bytes':got.is_file()and got.read_bytes()==expected.read_bytes(),'expected_sha256':sha(expected),'reproduced_sha256':sha(got)if got.exists()else None})
 old=json.loads((base/'pairs/PAIR_SUMMARIES.json').read_text());new=json.loads((q/'PAIR_SUMMARIES.json').read_text())
 refinement_exact=json.loads((base/'refinement/REFINEMENT_PAIR_SUMMARY.json').read_text())==json.loads((out/'refinement/REFINEMENT_PAIR_SUMMARY.json').read_text())
 assert refinement_exact,'Controlled refinement summaries differ'
 campaigns_exact=json.loads((base/'campaigns_R18/derived/RECOMPUTED_SUMMARY.json').read_text())==json.loads((out/'campaigns_R18/RECOMPUTED_SUMMARY.json').read_text())
 assert campaigns_exact,'ASU/persistence summaries differ'
 result={'schema':'r18-offline-reproduction-v1','all_tables_matched':all(c['same_bytes']for c in checks),'tables':checks,'pair_summaries_exactly_equal':new==old,'pair_summary_rows':len(new),'controlled_refinement_summary_exact':refinement_exact,'asu_persistence_summaries_exact':campaigns_exact,'optimizer_invocations':0,'new_endpoint_audits':0,'network_requests':0,'optional_archived_report_audit':bool(a.archive),'elapsed_seconds':time.monotonic()-started}
 (out/'REPRODUCTION_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
 if not result['all_tables_matched']or not result['pair_summaries_exactly_equal']:raise RuntimeError('Une reproduction diffère : consulter REPRODUCTION_VALIDATION.json')
 print(json.dumps({'all_tables_matched':True,'table_count':len(checks),'pair_summary_rows':len(new),'controlled_refinement_summary_exact':refinement_exact,'asu_persistence_summaries_exact':campaigns_exact,'optimizer_invocations':0},indent=2))
if __name__=='__main__':main()
