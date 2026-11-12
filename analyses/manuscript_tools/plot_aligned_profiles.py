from pathlib import Path
import argparse,gzip,io,json,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(path):
    with zipfile.ZipFile(path) as z:
        raw=z.read('DEPOUILLEMENT/DEPOUILLEMENT/OBSERVATIONS_ANALYSE_COMPACTES.jsonl.gz')
    return [json.loads(s)for s in gzip.decompress(raw).decode().splitlines()if s]

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--compact',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--regimes',nargs=2,default=['I','M']);p.add_argument('--stem',default='aligned_core_performance_profiles');a=p.parse_args()
    if (a.archive is None)==(a.compact is None):raise ValueError('Choose exactly one --archive or --compact')
    rows=load(a.archive)if a.archive else[json.loads(s)for s in gzip.open(a.compact,'rt')if s.strip()];by={(r['case_id'],r['config']):r for r in rows}
    a.out.mkdir(parents=True,exist_ok=True)
    native=['R3','HH','HS','SH','SS','SS_legacy']
    colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#8B6F47','#151515']
    labels=['R3','HH','HS','SH','SS',r'$SS_0$','Ipopt monotone']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(8.9,5.9),sharex=True,sharey=True)
    report=[]
    for i,backend in enumerate(['ma27','mumps']):
      configs=[f'{m}_{backend}'for m in native]+[f'ipopt_{backend}_monotone_aligned']
      for j,reg in enumerate(a.regimes):
        ids=sorted({r['case_id']for r in rows if r['regime']==reg});assert len(ids)==({'I':1037,'M':797,'E':1704,'Q':540}[reg])
        times=np.full((len(ids),len(configs)),np.inf)
        for k,cid in enumerate(ids):
          for l,cfg in enumerate(configs):
            r=by[cid,cfg]
            t=r.get('seconds_process')
            if r['classification']=='strict'and isinstance(t,(float,int))and np.isfinite(t)and t>0:times[k,l]=t
        best=np.min(times,axis=1);ratios=np.full_like(times,np.inf)
        good=np.isfinite(best)
        ratios[good,:]=times[good,:]/best[good,None]
        ax=axs[i,j]
        for l,cfg in enumerate(configs):
          finite=np.sort(ratios[np.isfinite(ratios[:,l]),l]);x=np.r_[1.,finite,1000.];y=np.r_[0.,np.arange(1,len(finite)+1)/len(ids),len(finite)/len(ids)]
          ax.step(x,y,where='post',color=colors[l],linestyle='--'if l==6 else'-',linewidth=1.3 if l in[0,6]else 1.,label=labels[l])
          report.append({'backend':backend,'regime':reg,'config':cfg,'planned':len(ids),'certificates_with_positive_process_time':len(finite)})
        ax.set_xscale('log');ax.set_xlim(1,1000);ax.set_ylim(0,1.02);ax.grid(True,which='major',alpha=.22)
        ax.set_title(f'{backend.upper()} — regime {reg}');ax.set_yticks([0,.2,.4,.6,.8,1])
        if j==0:ax.set_ylabel('Fraction of planned instances')
        if i==1:ax.set_xlabel('Elapsed-process time / fastest certified time')
    hs,ls=axs[0,0].get_legend_handles_labels();fig.legend(hs,ls,ncol=4,loc='upper center',bbox_to_anchor=(.5,1.005),frameon=False,columnspacing=1.5,handlelength=2.5)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(a.out/(a.stem+'.pdf'),bbox_inches='tight');fig.savefig(a.out/(a.stem+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    (a.out/(a.stem+'_validation.json')).write_text(json.dumps({'normalization':'fastest certified process time among all six native policies and the monotone certificate-matched Ipopt reference in each backend/regime panel; all planned cases in denominator','counts':report},indent=2)+'\n')
if __name__=='__main__':main()
