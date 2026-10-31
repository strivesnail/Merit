"""Plot measured NMD hit/QPS sweep and the N-only QPS/memory crossing."""
from pathlib import Path
import argparse
import csv
import json
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path('/mnt/nvme2/nc_n_capacity_match_20261031'))
args=parser.parse_args(); root=args.root
raw=json.loads((root/'results.json').read_text())
groups={}
for row in raw:groups.setdefault((row['mode'],row['nodes']),[]).append(row)
points=[]
for (mode,nodes),rows in groups.items():
    p=dict(mode=mode,nodes=nodes,runs=len(rows))
    for key in ('qps','extra_rss_mib','n_hit_pct','p_hit_pct','p_after_n_miss_pct','disk_reads_per_query','d_reads_per_query'):
        p[key]=st.mean(row[key] for row in rows)
    p['qps_min']=min(row['qps'] for row in rows);p['qps_max']=max(row['qps'] for row in rows)
    points.append(p)
points.sort(key=lambda p:(p['mode'],p['nodes']))
baseline=next((p['extra_rss_mib'] for p in points if p['mode']=='nonly' and p['nodes']==0),0)
for p in points:p['cache_rss_mib']=p['extra_rss_mib']-baseline
(root/'summary.json').write_text(json.dumps(points,indent=2)+'\n')
with (root/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(points[0]),lineterminator='\n');w.writeheader();w.writerows(points)
plt.rcParams.update({'font.family':'serif','font.size':12,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42})
fmt=FuncFormatter(lambda x,_:f'{x/1000:g}K' if abs(x)>=1000 else f'{x:g}')
nmd=[p for p in points if p['mode']=='nmd']
fig,axes=plt.subplots(2,1,figsize=(8,6.4),sharex=True)
x=[p['nodes'] for p in nmd]
for key,label,color,style in [('n_hit_pct','N cache','#cc79a7','o-'),('p_hit_pct','Prefetch','#0072b2','s-')]:
    axes[0].plot(x,[p[key] for p in nmd],style,color=color,label=label,lw=2,ms=5)
axes[0].set_ylabel('Hits / all accesses (%)');axes[0].set_ylim(bottom=0)
axes[0].legend(frameon=False,ncol=2);axes[0].grid(alpha=.2)
axes[1].errorbar(x,[p['qps'] for p in nmd],
                 yerr=[[p['qps']-p['qps_min'] for p in nmd],[p['qps_max']-p['qps'] for p in nmd]],
                 fmt='o-',color='#009e73',capsize=3,lw=2,label='NMD')
axes[1].set_ylabel('QPS');axes[1].set_xlabel('N cache capacity (nodes)')
axes[1].set_ylim(0,max(p['qps_max'] for p in nmd)*1.15)
axes[1].xaxis.set_major_formatter(fmt);axes[1].grid(alpha=.2)
selection=json.loads((root/'selection.json').read_text()) if (root/'selection.json').exists() else None
if selection and selection.get('plateau_found'):
    for ax in axes:ax.axvline(selection['selected']['nodes'],color='.5',ls=':',lw=1.2)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(root/f'nmd_n_capacity_hit_qps.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
if (root/'match.json').exists():
    match=json.loads((root/'match.json').read_text());selected=match['selected_nmd'];target=match['target_qps']
    nonly=sorted([p for p in points if p['mode']=='nonly'],key=lambda p:p['cache_rss_mib'])
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.errorbar([p['cache_rss_mib'] for p in nonly],[p['qps'] for p in nonly],
                yerr=[[p['qps']-p['qps_min'] for p in nonly],[p['qps_max']-p['qps'] for p in nonly]],
                fmt='o-',color='#0072b2',lw=2,capsize=3,label='N only')
    ax.errorbar([match['nmd_cache_rss_mib']],[target],
                yerr=[[target-min(selected['qps_runs'])],[max(selected['qps_runs'])-target]],
                fmt='D',color='#009e73',ms=8,capsize=4,label='NMD at prefetch plateau')
    ax.axhline(target,color='#009e73',ls='--',lw=1.2,alpha=.6)
    ax.set_xlabel('Additional resident memory (MiB)');ax.set_ylabel('QPS')
    ax.set_ylim(bottom=0);ax.set_xlim(left=0);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/f'nmd_vs_nonly_memory_qps.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
print(root/'nmd_n_capacity_hit_qps.png')
