#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json
p=argparse.ArgumentParser(description='Build the three LaTeX tables from the reproduced reference CSVs.')
p.add_argument('--input-directory',type=Path,default=Path(__file__).resolve().parent/'output')
p.add_argument('--out',type=Path,default=None)
a=p.parse_args();R=a.input_directory;O=a.out or R;O.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((R/'REFERENCE_OBJECTIVE_SUMMARY_ALL_CONFIGS.csv').open()))
D={(r['scope'],r['config']):r for r in rows}
configs=[m+'_'+b for b in ['ma27','mumps'] for m in ['R3','HH','HS','SH','SS','SS_legacy']]+['ipopt_'+b+'_'+s+'_aligned'for s in ['monotone','adaptive']for b in ['ma27','mumps']]
def fmt(x):return '--' if x=='' else f'{float(x):.5g}'
lines=[r'\begin{tabular}{@{}llrrrrrr@{}}',r'\toprule',r'Policy & Linear solver & $N_Q$ & Q median & Q $Q_3$ & Q max & $N_F$ & F max\\',r'\midrule']
for c in configs:
 q=D['QPLIB_reference',c];f=D['MINLPLIB',c]
 if c.startswith('ipopt'):
  label='Ipopt, '+c.split('_')[2];backend='MA27' if 'ma27' in c else 'MUMPS'
 else:
  policy,backend=c.rsplit('_',1);label='$SS_0$' if policy=='SS_legacy' else policy;backend=backend.upper()
 lines.append(' & '.join([label,backend,q['certified_cases'],fmt(q['median_percent']),fmt(q['q75_percent']),fmt(q['maximum_percent']),f['certified_cases'],fmt(f['maximum_percent'])])+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(O/'reference_values_summary.tex').write_text('\n'.join(lines)+'\n')
rr=list(csv.DictReader((R/'REFERENCE_VALUES_ADMISSIBLE.csv').open()));lines=[r'\begin{tabular}{@{}lrrrr@{}}',r'\toprule',r'Facility model & Published reference & Reported infeas. & Ref. at $10^{-9}$ & Difference (\%)\\',r'\midrule']
for r in rr:
 if r['dataset']!='MINLPLIB':continue
 v=float(r['reference_value_frozen']);hp=float(r['reference_high_precision_frozen']);gap=100*(hp-v)/max(1,abs(v));n=r['case_id'].split('__')[1].replace('_',r'\_')
 lines.append(f'{n} & {v:.8f} & ${float(r["published_primal_feasibility"]):.0e}$ & {hp:.8f} & {gap:.5g}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(O/'reference_facility_precision.tex').write_text('\n'.join(lines)+'\n')
# The primary common cohort uses R3/HH MA27 and both monotone references.
rs=list(csv.DictReader((R/'REFERENCE_OBJECTIVE_COMMON_COHORT.csv').open()));N=rs[0]['shared_reference_cohort_size'];lines=[r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',r'Policy & Linear solver & Certified/$'+N+r'$ & Median gap (\%) & Within $0.1\%$\\',r'\midrule']
for r in rs:
 c=r['config'];
 if c.startswith('ipopt'):label='Ipopt, '+c.split('_')[2];backend='MA27' if 'ma27' in c else 'MUMPS'
 else:policy,backend=c.rsplit('_',1);label='$SS_0$' if policy=='SS_legacy' else policy;backend=backend.upper()
 lines.append(' & '.join([label,backend,r['certified_in_shared_cohort']+'/'+N,fmt(r['median_percent']),r['within_relative_1e-3']])+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(O/'reference_values_common_monotone.tex').write_text('\n'.join(lines)+'\n')
