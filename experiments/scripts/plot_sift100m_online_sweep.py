"""Plot the online sweep and analyze actual final dynamic page membership.

The lower two figures use the N=75 MiB, prefetch=25 MiB final snapshot.
Heat ranks use the same preexisting per-workload expand profile as the original
figures, ONLY during offline analysis; it is not loaded by online search.
"""
import csv
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/mnt/nvme_single/sift100m/runs/n_prefetch_online_probe_20261026')
COVERAGE_ROOT=Path('/mnt/nvme_single/sift100m/runs/n_prefetch_online_20261026/coverage_runs')
OLD=Path('/mnt/nvme_single/sift100m/runs/n_prefetch_sweep_20261026')
WORK=[('pub10k','10K','#E66100'),('zipf','10K+Zipf','#0072B2'),('x5','10K×5','#009E73')]
PF=[0,5,10,25,50,100,200]
NS=[4,16,32,75,100,200,300]
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def load(work,n,p):
    name=f'pf{p}_n75' if n==75 else f'n{n}_pf25'
    return json.loads((ROOT/work/name/'measurement.json').read_text())

def save(fig,stem):
    for ext in ('png','pdf'):
        fig.savefig(ROOT/(stem+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    print('wrote',ROOT/(stem+'.png'),flush=True)

def coverage(work):
    raw=np.fromfile(COVERAGE_ROOT/work/'pf25_n75/dynamic_pages.bin',dtype='<u4')
    assert raw[0]==0x4e435047 and raw[1]==1
    slots=int(raw[2]); assert slots==10
    rows=raw[3:].reshape(-1,slots+3)
    # Online default has one primary page per seed. Do not silently union extra pages.
    assert not np.any(rows[:,1]), 'Secondary pages require an explicit extent coverage definition'
    assert np.all((rows[:,2]>0)&(rows[:,2]<=slots))
    seeds=rows[:,3].copy()
    assert len(np.unique(seeds))==len(seeds)
    order=np.argsort(seeds)
    rows=rows[order]; seeds=seeds[order]
    words=np.memmap('/mnt/nvme_single/sift100m/sift100m_index_disk.index',mode='r',dtype='<u4')
    degrees=np.empty(len(rows),dtype=np.uint32)
    covered=np.empty(len(rows),dtype=np.uint32)
    for start in range(0,len(rows),4096):
        r=rows[start:start+4096]; s=r[:,3].astype(np.int64)
        offsets=(s//10+1)*1024+(s%10)*97+32
        deg=words[offsets]
        assert np.all((deg>0)&(deg<=64))
        neighbors=words[offsets[:,None]+1+np.arange(64)[None,:]]
        members=r[:,3:]
        matches=(neighbors[:,:,None]==members[:,None,:])
        matches &= np.arange(slots)[None,None,:]<r[:,2,None,None]
        matches &= np.arange(64)[None,:,None]<deg[:,None,None]
        count=np.any(matches,axis=2).sum(axis=1)
        degrees[start:start+len(r)]=deg
        covered[start:start+len(r)]=count
    del words
    profile=OLD/'profile'/f'{work}_node_expand.bin'
    shape=np.fromfile(profile,dtype='<u4',count=2)
    assert tuple(shape)==(100000000,1)
    heat_map=np.memmap(profile,mode='r',dtype='<u8',offset=8,shape=(int(shape[0]),))
    heat=np.array(heat_map[seeds]); del heat_map
    percent=100*covered/degrees
    np.savez_compressed(ROOT/f'{work}_page_coverage.npz',seed=seeds,degree=degrees,covered=covered,heat=heat,
                        page_id=rows[:,0],member_count=rows[:,2],members=rows[:,3:])
    edges=[0,1,5,10,25,50,100]
    ranked=np.lexsort((seeds,-heat.astype(np.int64)))
    bins=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        start=int(len(seeds)*lo/100); end=int(len(seeds)*hi/100)
        ids=ranked[start:end]
        bins.append(dict(lo=lo,hi=hi,seeds=len(ids),mean_pct=float(percent[ids].mean()) if len(ids) else None))
    summary=dict(workload=work,seeds=len(seeds),mean_graph_neighbors=float(degrees.mean()),
                 mean_neighbors_on_page=float(covered.mean()),mean_coverage_pct=float(percent.mean()),
                 heat_weighted_coverage_pct=float(np.average(percent,weights=heat)) if heat.sum() else None,
                 bins=bins,coverage_source='Final live D directory snapshot from separate diagnostic run, N=75 MiB / prefetch=25 MiB',
                 heat_source=str(profile),note='Snapshot distribution, not a hit-event-weighted distribution')
    (ROOT/f'{work}_coverage_summary.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)
    return percent,bins

def main():
    global ROOT, COVERAGE_ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('--results-dir',type=Path,default=ROOT)
    parser.add_argument('--coverage-dir',type=Path,default=COVERAGE_ROOT)
    args=parser.parse_args()
    ROOT=args.results_dir;COVERAGE_ROOT=args.coverage_dir
    assert len(list(ROOT.glob('*/*/measurement.json')))==39
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.2))
    records=[]
    for work,label,color in WORK:
        p=[load(work,75,x) for x in PF]
        n=[load(work,x,25) for x in NS]
        records.extend(p+[x for x in n if x['n_mb']!=75])
        for ax,x,values,metric in [(axs[0],PF[1:],p[1:],'d_probe_hit_pct'),
                                   (axs[1],NS,n,'d_probe_hit_pct')]:
            ax.plot(x,[v[metric] for v in values],'-o',color=color,label=label,markersize=3,linewidth=1.3)
    axs[0].set_xlabel('Prefetch (MB), N = 75MB')
    axs[0].set_xticks([25,50,100,150,200])
    axs[1].set_xlabel('N cache (MB), prefetch = 25MB')
    rate_limit=max(5,math.ceil(max(v['d_probe_hit_pct'] for v in records)/5)*5)
    for ax in axs:
        ax.set_ylabel('D-cache hit rate (%)')
        ax.set_ylim(0,rate_limit)
    for ax in axs.flat: ax.grid(axis='y',alpha=.3,linewidth=.5)
    axs[0].legend(ncol=3,fontsize=8,frameon=False,loc='best')
    fig.suptitle('sift100M',fontsize=11)
    fig.tight_layout()
    save(fig,'sift100m_n_prefetch_sweep_online')
    keys=['workload','n_mb','prefetch_mb','qps','recall','d_probe_hit_pct','d_demand_probes','d_demand_hits','d_demand_misses']
    with (ROOT/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    datasets=[(label,color,*coverage(work)) for work,label,color in WORK]
    fig,ax=plt.subplots(figsize=(5.4,4.2))
    for label,color,percent,bins in datasets:
        x,counts=np.unique(percent,return_counts=True)
        ax.step(np.r_[0,x,40],np.r_[0,np.cumsum(counts)/len(percent),1],where='post',label=label,color=color)
    ax.set(xlabel='Graph neighbors on the D page (%)',ylabel='Fraction of seeds',xlim=(0,40),ylim=(0,1.02))
    ax.grid(alpha=.3);ax.legend(frameon=False,loc='lower right')
    ax.set_title('Online D-cache: final page snapshot',fontsize=10)
    fig.tight_layout();save(fig,'sift100m_nbr_coverage_cdf_online')
    fig,ax=plt.subplots(figsize=(5.4,4.2))
    for label,color,percent,bins in datasets:
        ax.plot([(b['lo']+b['hi'])/2 for b in bins],[b['mean_pct'] for b in bins],'-o',color=color,label=label)
    ax.set(xlabel='Seed heat rank (percentile, 0 = hottest)',ylabel='Neighbors on the D page (%)',xlim=(0,100),ylim=(0,40))
    ax.grid(axis='y',alpha=.3);ax.legend(frameon=False)
    ax.set_title('Online D-cache: final page snapshot',fontsize=10)
    fig.tight_layout();save(fig,'sift100m_nbr_coverage_by_heat_online')
    metadata=dict(mode='Online D-cache, cold start, no prefill, no extra warmup',threads=15,L=128,beam=4,k=10,
                  dcache_capacity_pages=1500000,query_workloads=['10K','10K+Zipf','10Kx5'],
                  hit_rate='n_neighborcache_dcache_demand_hits / n_neighborcache_dcache_demand_probes',
                  hit_rate_note='Only actual demand-node dynamic directory lookups; once per beam node request; skips, maintenance and speculative lookups excluded; SSD reads not required',
                  results_dir=str(ROOT),coverage_runs=str(COVERAGE_ROOT),
                  lower_figures='Separate diagnostic runs: end-of-search selected primary-page snapshot at N=75MB, Prefetch=25MB; not weighted by runtime D-hit events',
                  heat='Original per-workload expansion profile used offline for comparable heat ranking, never to seed online cache')
    (ROOT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
