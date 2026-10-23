"""Analyze stored probe matrices: principal-angle affinity and rank inversion.
Usage: python analyze.py --run runs/pilot --subspace-rank 16
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from scipy.stats import kendalltau, spearmanr

TASKS=['sst2','qqp','qnli','mnli','ag_news']

def affinity(stage_dir,layer,rank,seed):
    bases=[]
    for task in TASKS:
        x=np.load(stage_dir/f'{task}_layer{layer}.npy').astype('float64')
        x=x-x.mean(axis=0,keepdims=True)
        k=min(rank,x.shape[0]-1,x.shape[1])
        if k<1:raise ValueError('Too few probes')
        _,_,vt=randomized_svd(x,n_components=k,n_iter=5,random_state=seed)
        bases.append(vt.T)
    A=np.eye(len(TASKS))
    for i in range(len(TASKS)):
        for j in range(i+1,len(TASKS)):
            c=bases[i].T@bases[j]
            # Squared cosines of principal angles averaged over r.
            A[i,j]=A[j,i]=np.sum(c*c)/rank
    return A

def order_stats(A0,At,tol=1e-3):
    idx=np.triu_indices(len(TASKS),1)
    a=A0[idx];b=At[idx]
    concord=discord=ties=0; examples=[]
    pairs=[(TASKS[i],TASKS[j]) for i,j in zip(*idx)]
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            d0=a[i]-a[j];dt=b[i]-b[j]
            if abs(d0)<tol or abs(dt)<tol:ties+=1;continue
            if d0*dt<0:
                discord+=1
                examples.append({'pair_1':pairs[i],'pair_2':pairs[j],'initial':[float(a[i]),float(a[j])],'current':[float(b[i]),float(b[j])]})
            else:concord+=1
    denom=concord+discord
    return {'inversion_rate':discord/denom if denom else None,'valid_comparisons':denom,'excluded_near_ties':ties,'kendall_tau':float(kendalltau(a,b).statistic),'spearman_rho':float(spearmanr(a,b).statistic),'inversions':examples}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--subspace-rank',type=int,default=16);p.add_argument('--tolerance',type=float,default=0.001)
    args=p.parse_args();root=Path(args.run);config=json.loads((root/'config.json').read_text());layers=[int(i) for i in config['layers'].split(',')]
    stages=sorted((root/'features').glob('stage_*'),key=lambda p:int(p.name.split('_')[1]));out=root/'analysis';out.mkdir(exist_ok=True)
    if not stages:raise ValueError('No feature matrices. Run train_and_probe.py first.')
    report={}
    for layer in layers:
        matrices=[affinity(stage,layer,args.subspace_rank,config['seed']) for stage in stages]
        np.save(out/f'affinity_layer{layer}.npy',np.stack(matrices))
        stats=[order_stats(matrices[0],a,args.tolerance) for a in matrices]
        report[f'layer{layer}']=[{'stage':int(stages[i].name.split('_')[1]),**stat} for i,stat in enumerate(stats)]
        fig,ax=plt.subplots(figsize=(9,5))
        for i in range(len(TASKS)):
            for j in range(i+1,len(TASKS)):
                ax.plot(range(len(matrices)),[a[i,j] for a in matrices],label=f'{TASKS[i]}/{TASKS[j]}')
        ax.set(title=f'Principal-angle affinity trajectories · layer {layer}',xlabel='Completed tasks',ylabel='Mean squared principal-angle cosine')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=7);fig.tight_layout();fig.savefig(out/f'trajectories_layer{layer}.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(range(len(stats)),[s['inversion_rate'] for s in stats],marker='o')
        ax.set(title=f'H2 inversion rate · layer {layer}',xlabel='Completed tasks',ylabel='Inversion rate',ylim=(0,1));fig.tight_layout();fig.savefig(out/f'inversions_layer{layer}.png',dpi=180);plt.close(fig)
        print('Layer',layer,[(i,round(s['inversion_rate'],3) if s['inversion_rate'] is not None else None) for i,s in enumerate(stats)])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('Saved:',out)

if __name__=='__main__':main()
