from pathlib import Path
import gzip,json,statistics,itertools,math,collections,argparse

def generate(compact,out):
 rows=[json.loads(l)for l in gzip.open(compact,'rt')if l.strip()]
 by=collections.defaultdict(dict)
 for r in rows:by[r['config']][r['case_id']]=r
 order=[]
 for b in ['ma27','mumps']:
  order +=[(f'{m}_{b}',r'$SS_0$'if m=='SS_legacy'else m,b.upper())for m in ['R3','HH','HS','SH','SS','SS_legacy']]
 order +=[(f'ipopt_{b}_{s}_aligned','Ipopt '+s,b.upper())for s in ['monotone','adaptive']for b in ['ma27','mumps']]
 lines=[r'\begin{tabular}{@{}llrrrrr@{}}',r'\toprule',r'Policy & Linear solver & I & M & E & Q & All \\',r'\midrule']
 for cfg,label,b in order:
  rr=list(by[cfg].values());cells=[]
  for reg in ['I','M','E','Q','ALL']:
   rs=[r for r in rr if reg=='ALL'or r['regime']==reg];strict=sum(r['classification']=='strict'for r in rs);cells.append(f'{strict}/{len(rs)-strict}')
  lines.append(' & '.join([label,b,*cells])+r' \\')
 lines +=[r'\bottomrule',r'\end{tabular}'];(out/'classification_totals_backends.tex').write_text('\n'.join(lines)+'\n')
 names={'CBLIB_SOC_reference':'CBLIB','COPS_camshape':'COPS camshape','COPS_minimal_surface':'COPS minimal surface','CUTEst':'CUTEst','GasLib134':'GasLib134','GasLib_other':'Other GasLib','IJOC_partition_QCQP':'IJOC partition QCQP','MINLPLIB':'MINLPLib','Mittelmann_boundary_cont6':'Boundary control','Mittelmann_distributed_dist2':'Distributed control','PGLIB':'PGLib','QPLIB_reference':'QPLIB','beam_safety':'Array safety','beam_structural':'ASU radio-beamforming','controlled_kappa':'Controlled curvature','gas_power':'Coupled gas/power','gas_transport':'Gas transport','lennard_jones':'Particle potential','multicast_beamforming':'Multicast beamforming','obstacle_path_smoothing':'Path smoothing','packing_equal_disks':'Disk packing','pooling_MINLPLIB':'MINLPLib pooling','pooling_luedtke':'Luedtke pooling','pooling_reduced':'Reduced pooling','pose_graph_2d':r'2D pose graph$^\dagger$','robot_arm':r'Robot arm$^\dagger$','two_ellipsoid_qcqp':'Two ellipsoids','water_networks':'Water networks'}
 for backend in ['ma27','mumps']:
  configs=[(f'{m}_{backend}',r'$SS_0$'if m=='SS_legacy'else m)for m in ['R3','HH','HS','SH','SS','SS_legacy']]+[(f'ipopt_{backend}_{s}_aligned','Ipopt '+('mono'if s=='monotone'else'adapt'))for s in ['monotone','adaptive']]
  first=list(by[configs[0][0]].values());datasets=sorted({r['dataset']for r in first})
  lines=[r'\begin{tabular}{@{}lrrrrrrrrr@{}}',r'\toprule',' & '.join(['Collection',r'$N$']+[v for _,v in configs])+r' \\',r'\midrule']
  for ds in datasets:
   ids={r['case_id']for r in first if r['dataset']==ds};vals=[sum(by[c][cid]['classification']=='strict'for cid in ids)for c,_ in configs]
   lines.append(' & '.join([names.get(ds,ds.replace('_',r'\_')),str(len(ids)),*map(str,vals)])+r' \\')
  lines +=[r'\bottomrule',r'\end{tabular}'];(out/f'dataset_coverage_native_{backend}.tex').write_text('\n'.join(lines)+'\n')
  lines=[r'\begin{tabular}{@{}lrrrrr@{}}',r'\toprule',r'Numerator / denominator & I & M & E & Q & All \\',r'\midrule']
  for (ca,la),(cb,lb)in itertools.combinations(configs,2):
   vals=[]
   for reg in ['I','M','E','Q','ALL']:
    ratios=[]
    for cid,a in by[ca].items():
     b=by[cb][cid]
     if reg!='ALL'and a['regime']!=reg:continue
     ta,tb=a.get('seconds_process'),b.get('seconds_process')
     if a['classification']==b['classification']=='strict'and isinstance(ta,(int,float))and isinstance(tb,(int,float))and math.isfinite(ta)and math.isfinite(tb)and ta>0 and tb>0:ratios.append(ta/tb)
    vals.append(f'{statistics.median(ratios):.3g} [{len(ratios)}]'if ratios else'-- [0]')
   lines.append(' & '.join([la+' / '+lb,*vals])+r' \\')
  lines +=[r'\bottomrule',r'\end{tabular}'];(out/('paired_all.tex'if backend=='ma27'else'paired_all_native_mumps.tex')).write_text('\n'.join(lines)+'\n')
 return {'rows':len(rows),'configurations':len(by),'collections':len(datasets),'unordered_comparisons_per_backend':28}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--compact',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);print(json.dumps(generate(a.compact,a.out),indent=2))
