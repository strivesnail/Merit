"""Reduce only private prefetch; keep single-worker online N/M/D fixed."""
from pathlib import Path
import argparse
import csv
import json
import os
import statistics as st
import run as bench
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def report(root):
    rows=[json.loads(p.read_text()) for p in root.glob('*/result.json')]
    groups={}
    for row in rows:groups.setdefault(row['prefetch_pages'],[]).append(row)
    points=[]
    for pages,runs in sorted(groups.items()):
        p=dict(prefetch_pages=pages,prefetch_mib=pages/256,record_slots=pages*10,
               n_nodes=2048,runs=len(runs))
        for key in ('qps','n_hit_pct','p_hit_pct','p_after_n_miss_pct',
                    'disk_reads_per_query','d_reads_per_query','extra_rss_mib'):
            p[key]=st.mean(x[key] for x in runs)
        points.append(p)
    bench.write(root/'summary.json',points)
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(points[0]),lineterminator='\n');w.writeheader();w.writerows(points)
    plt.rcParams.update({'font.family':'serif','font.size':12,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    ax.plot([p['record_slots'] for p in points],[p['p_hit_pct'] for p in points],
            'o-',color='#0072b2',lw=2,ms=6)
    ax.set_xlabel('Prefetch capacity (node records)');ax.set_ylabel('Prefetch hit rate (%)')
    ax.set_ylim(0,45);ax.grid(alpha=.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:g}K' if x>=1000 else f'{x:g}'))
    ax.set_xticks([p['record_slots'] for p in points])
    ax.tick_params(axis='x',labelrotation=30)
    secondary=ax.secondary_xaxis('top',functions=(lambda x:x/2560,lambda x:x*2560))
    secondary.set_xlabel('Prefetch capacity (MiB)')
    secondary.set_xticks([p['prefetch_mib'] for p in points])
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/f'prefetch_capacity_hit_rate.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    print('SUMMARY',json.dumps(points),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/mnt/nvme2/nc_prefetch_reduction_single_20261031'))
    p.add_argument('--plot-only',action='store_true')
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    os.sched_setaffinity(0,{23})
    if args.plot_only:report(root);return
    bench.write(root/'protocol.json',dict(threads=1,n_nodes=2048,m_entries=131072,d_pages=65536,
        stash_capacity_nodes=424742,prefetch_pages=[2560,1280,512,256,128],
        prefetch_mib=[10,5,2,1,.5],warmup_queries=20000,measured_queries=20000,
        main_denominator='all node accesses',baseline_reused_from='nc_n_capacity_match_20261031',
        no_other_experiment_running=True))
    for rep in (0,1):
        source=bench.DEFAULT_ROOT/f'nmd_2048_{rep}'
        dest=root/f'baseline10_{rep}';dest.mkdir(exist_ok=True)
        row=json.loads((source/'result.json').read_text());row['prefetch_pages']=2560
        row['source_artifact']=str(source);bench.write(dest/'result.json',row)
        bench.write(dest/'command.json',json.loads((source/'command.json').read_text()))
    for pages in (1280,512,256,128):
        bench.run(root,'nmd',2048,0,prefetch_pages=pages)
        report(root)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
