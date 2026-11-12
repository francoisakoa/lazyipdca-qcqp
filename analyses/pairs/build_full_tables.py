#!/usr/bin/env python3
"""All eight within-backend configurations: coverage and 28 timing pairings."""
from pathlib import Path
import argparse,csv,gzip,itertools,json
POLS=('R3','HH','HS','SH','SS','SS_legacy');REG=('I','M','E','Q')
LABELS={'CBLIB_SOC_reference':'CBLIB SOC','COPS_camshape':'COPS camshape','COPS_minimal_surface':'COPS minimal surface','CUTEst':'CUTEst','GasLib134':'GasLib134','GasLib_other':'GasLib other','IJOC_partition_QCQP':'IJOC partition QCQP','MINLPLIB':'MINLPLib','Mittelmann_boundary_cont6':r'\shortstack[l]{Mittelmann\\boundary control}','Mittelmann_distributed_dist2':r'\shortstack[l]{Mittelmann\\distributed control}','PGLIB':'PGLib','QPLIB_reference':'QPLIB reference','beam_safety':'Beam safety','beam_structural':'ASU radio-beamforming','controlled_kappa':'Controlled curvature','gas_power':'Gas--power','gas_transport':'Gas transport','lennard_jones':'Lennard--Jones','multicast_beamforming':'Multicast beamforming','obstacle_path_smoothing':'Obstacle path smoothing','packing_equal_disks':'Equal disk packing','pooling_MINLPLIB':'MINLPLib pooling','pooling_luedtke':'Luedtke pooling','pooling_reduced':'Reduced pooling','pose_graph_2d':'Pose graph','robot_arm':'Robot arm','two_ellipsoid_qcqp':'Two ellipsoids','water_networks':'Water networks'}
def table(headers,rows,spec):return '\\begin{tabular}{@{}'+spec+'@{}}\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(map(str,r))+'\\\\'for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
def label(c,b):
    if c.startswith('ipopt'):return 'Ipopt '+c.split('_')[2]
    p=c[:-len('_'+b)];return '$SS_0$'if p=='SS_legacy'else p
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-directory',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--out',type=Path);a=p.parse_args();R=a.input_directory;O=a.out or R;O.mkdir(parents=True,exist_ok=True)
    coverage=list(csv.DictReader((R/'COVERAGE_BY_REGIME.csv').open()));C={(r['config'],r['regime']):r for r in coverage}
    pairs=json.loads((R/'PAIR_SUMMARIES.json').read_text());P={(r['A'],r['B'],r['scope']):r for r in pairs}
    with gzip.open(R/'OBSERVATIONS_16_CONFIGS.jsonl.gz','rt')as f:obs=[json.loads(s)for s in f]
    idx={}
    for r in obs:idx.setdefault(r['config'],{})[r['case_id']]=r
    meta=idx['R3_ma27'];groups={}
    for cid,r in meta.items():groups.setdefault(r['dataset'],[]).append(cid)
    assert len(groups)==28 and sum(map(len,groups.values()))==4078
    for b in ['ma27','mumps']:
        configs=[p+'_'+b for p in POLS]+['ipopt_'+b+'_'+s+'_aligned'for s in ['monotone','adaptive']]
        rows=[]
        for config in configs:
            ns=[int(C[config,g]['strict'])for g in REG];tot=sum(ns)
            cells=[str(n)+' ('+f'{100*n/int(C[config,g]["planned"]):.2f}'+r'\%)'for n,g in zip(ns,REG)]
            rows.append([label(config,b),*cells,str(tot)+' ('+f'{100*tot/4078:.2f}'+r'\%)'])
        (O/f'complete_benchmark_{b}_aligned.tex').write_text(table(['Configuration','I ($1037$)','M ($797$)','E ($1704$)','Q ($540$)','All ($4078$)'],rows,'lrrrrr'))
        rows=[]
        for name,ids in sorted(groups.items()):rows.append([LABELS[name],len(ids),*[sum(idx[c][i]['classification']=='strict'for i in ids)for c in configs]])
        (O/f'dataset_coverage_{b}_aligned.tex').write_text(table(['Collection','$N$',*[label(c,b)for c in configs]],rows,'l'+'r'*9))
        for long in [False,True]:
            rows=[];prefix='process_ratio_both_over_1s'if long else'process_ratio_A_over_B'
            for A,B in itertools.combinations(configs,2):
                vals=[]
                for regime in [*REG,'all']:
                    r=P[A,B,regime];n=r[prefix+'_N'];v=r[prefix+'_median'];vals.append('--'if v is None else f'{v:.3g} [{n}]')
                rows.append([label(A,b)+' / '+label(B,b),*vals])
            name=('paired_over1s_'if long else'paired_all_')+b+'_aligned.tex'
            (O/name).write_text(table(['Numerator / denominator','I','M','E','Q','All'],rows,'lrrrrr'))
    print(json.dumps({'backend_panels':2,'configs_per_panel':8,'datasets':28,'pairs_per_panel':28,'tables_written':8,'optimizer_invocations':0}))
if __name__=='__main__':main()
