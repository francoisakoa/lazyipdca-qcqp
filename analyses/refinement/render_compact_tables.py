"""Render the printed compact tables from the archived paired summaries."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    d=json.loads(a.summary.read_text());a.out.mkdir(parents=True,exist_ok=True)
    table=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',
           r'Backend/glob. & Regime & F & A & A-only/F-only & \shortstack{Time A/F\\$Q_1$/median/$Q_3$} & $n_t$\\',r'\midrule']
    for r in d['comparisons']:
        if r['scope']not in('I','M'):continue
        q=r['process_ratio_both_over_1s']
        table.append(f"{r['backend'].upper()}/{r['globalization']} & {r['scope']} & {r['frozen_strict']} & {r['adaptive_strict']} & {r['adaptive_only']}/{r['frozen_only']} & {q['Q1']:.2f}/{q['median']:.2f}/{q['Q3']:.2f} & {q['n']}"+r'\\')
    (a.out/'refinement_coverage_times.tex').write_text('\n'.join(table+[r'\bottomrule',r'\end{tabular}'])+'\n')
    table=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',
           r'Backend/glob. & Common & Obj. L/E/H & \shortstack{Median\\$|g|$ (\%)} & \shortstack{Median\\iter. A$-$F} & \shortstack{Median\\time A/F} & $n_t$\\',r'\midrule']
    for r in d['comparisons']:
        if r['scope']!='ASU':continue
        q=r['process_ratio_both_over_1s'];o=r['objective_0p1pct']
        table.append(f"{r['backend'].upper()}/{r['globalization']} & {r['both_strict']} & {o['adaptive_lower']}/{o['equivalent']}/{o['adaptive_higher']} & {100*r['objective_absolute_gap']['median']:.2f} & {r['iterations_A_minus_F']['median']:.0f} & {q['median']:.2f} & {q['n']}"+r'\\')
    (a.out/'refinement_asu_compact.tex').write_text('\n'.join(table+[r'\bottomrule',r'\end{tabular}'])+'\n')
    print('Two compact tables rendered from paired summary.')


if __name__=='__main__':main()
