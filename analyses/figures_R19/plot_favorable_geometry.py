"""Render current-version controlled trajectories and analytic local geometry."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':11,'axes.labelsize':11.5,
    'axes.titlesize':11.5,'legend.fontsize':10,'mathtext.fontset':'dejavuserif',
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
BETA=.1;X0=np.array([10.,15.732132722552274]);TOL=1e-8
COLORS={'lazy_adaptive':'#1768ac','lazy_scalar':'#60a5d3','slsqp':'#2b8a5b','ipopt_monotone':'#bd5a18'}
LABELS={'lazy_adaptive':'LazyQCQP: directional','lazy_scalar':'LazyQCQP: scalar',
    'slsqp':'SLSQP','ipopt_monotone':'Ipopt: monotone'}
STYLES={'lazy_adaptive':('-',None),'lazy_scalar':(':',None),'slsqp':('-','s'),
    'ipopt_monotone':('-','^')}
TRACES={method:json.loads((ROOT/(method+'_trace.json')).read_text()) for method in COLORS}

def save(fig,name):
    for ext in ('pdf','png'):
        fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',dpi=180)
    plt.close(fig)

def geometry():
    fig,axs=plt.subplots(1,2,figsize=(8,4.1),constrained_layout=True)
    ax=axs[0];a,b=2.,3.;v=np.linspace(-1.1,4.6,600)
    original=np.sqrt(1+BETA*v*v);direction=(1+a*a+BETA*v*v)/(2*a)
    scalar=(1+a*a+b*b-2*b*v+(BETA+1)*v*v)/(2*a)
    ax.fill_betweenx(v,.7,original,color='#ededed')
    ax.fill_betweenx(v,direction,4.,color='#dcecf9')
    ax.fill_betweenx(v,scalar,4.,color='#d9e5c9')
    ax.plot(original,v,color='#333',lw=1.5,label='Original boundary')
    ax.plot(direction,v,color=COLORS['lazy_adaptive'],lw=1.6,label='Directional majorant')
    ax.plot(scalar,v,color='#9a7025',lw=1.5,ls='--',label='Scalar majorant')
    ax.scatter([a],[b],s=28,color='black',zorder=5)
    ax.annotate(r'$x^k$',(a,b),xytext=(7,5),textcoords='offset points')
    td=np.array([(a+1/a)/2,0]);ts=np.array([(1+a*a+BETA/(BETA+1)*b*b)/(2*a),b/(1+BETA)])
    for dest,color,label in [(td,COLORS['lazy_adaptive'],r'$T_D(x^k)$'),(ts,'#9a7025',r'$T_S(x^k)$')]:
        ax.annotate('',xy=dest,xytext=(a,b),arrowprops={'arrowstyle':'->','color':color,'lw':1.2})
        ax.scatter(*dest,color=color,s=26,marker='s',zorder=5)
        ax.annotate(label,dest,xytext=(5,-16 if dest[1]<1 else 8),textcoords='offset points',color=color,fontsize=10)
    ax.scatter([1],[0],marker='*',color='black',s=90,zorder=5)
    ax.annotate('infeasible',(.88,2.7),rotation=72,color='#777',fontsize=10)
    ax.set(xlim=(.72,3.9),ylim=(-1.05,4.5),xlabel=r'$y_1$',ylabel=r'$y_2$',title='(a) Quadratic majorants at $x^k$')
    ax=axs[1];v=np.linspace(-1,17,600);original=np.sqrt(1+BETA*v*v)
    ax.fill_betweenx(v,0,original,color='#ededed');ax.plot(original,v,color='#333',lw=1.5,label='Original boundary')
    xy=np.vstack([X0,np.array([r['x'] for r in TRACES['lazy_adaptive']])])
    ax.plot(xy[:,0],xy[:,1],'-o',ms=3,lw=1.5,color=COLORS['lazy_adaptive'],label='LazyQCQP trajectory')
    ref=[X0.copy()];p=X0.copy()
    for _ in range(7):
        p=np.array([.5*(p[0]+1/p[0]),0.]);ref.append(p.copy())
    ref=np.array(ref)
    ax.plot(ref[:,0],ref[:,1],':s',ms=3,lw=1.4,color='#8b4ba3',label='Exact-majorant map')
    ax.scatter(*X0,color='black',s=28,zorder=5)
    ax.annotate(r'$x^0$',X0,xytext=(-20,5),textcoords='offset points')
    ax.scatter([1],[0],s=90,marker='*',color='black',zorder=5)
    ax.set(xlim=(0,10.6),ylim=(-1,17),xlabel=r'$x_1$',ylabel=r'$x_2$',title='(b) Computed accepted iterates')
    handles=[];labels=[]
    for ax in axs:
        hh,ll=ax.get_legend_handles_labels()
        for h,l in zip(hh,ll):
            if l not in labels:handles.append(h);labels.append(l)
    fig.legend(handles,labels,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.21),ncol=3,fontsize=10.5)
    save(fig,'favorable_geometry')

def convergence():
    fig,axs=plt.subplots(1,2,figsize=(8,4.1),constrained_layout=True)
    for method,rows in TRACES.items():
        iterations=[r['iteration'] for r in rows];linestyle,marker=STYLES[method]
        for ax,field in zip(axs,('box_kkt','primal_error_inf')):
            ax.semilogy(iterations,np.maximum([r[field] for r in rows],1e-17),linestyle,
                color=COLORS[method],lw=1.7 if method=='lazy_adaptive' else 1.3,
                marker=marker,markersize=3,markevery=3,label=LABELS[method])
    axs[0].axhline(TOL,color='black',lw=.8,ls='--')
    axs[0].set(ylabel=r'Original box-KKT residual $\mathcal{R}_B$',ylim=(1e-14,10**(np.ceil(np.log10(max(r['box_kkt'] for rows in TRACES.values() for r in rows)))+1)),title='(a) First-order residual')
    axs[1].set(ylabel=r'Primal error $\|x^k-x_*\|_\infty$',ylim=(1e-16,2e2),title='(b) Distance to the solution')
    for ax in axs:
        ax.set(xlabel='Iteration',xlim=(0,max(r['iteration'] for rows in TRACES.values() for r in rows)+4));ax.grid(axis='y',color='#eee',lw=.6)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.21),ncol=3,fontsize=10.5)
    save(fig,'favorable_convergence')

if __name__=='__main__':
    geometry()
