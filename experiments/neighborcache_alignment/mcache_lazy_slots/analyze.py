"""Compare allocated descriptor storage and observed RSS at equal capacities."""
from pathlib import Path
import argparse,csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,default=HERE/'results.json')
parser.add_argument('--output-dir',type=Path,default=HERE/'plots')
args=parser.parse_args();R=args.output_dir;R.mkdir(parents=True,exist_ok=True)
rows=json.loads(args.input.read_text());assert len(rows)==2
out=[]
for a in rows:
 assert a['ids_and_distances_bit_identical'] and a['recall_at_10']==94.94
 assert a['mcache_storage']['capacity']==4000000 and a['mcache_storage']['live']==1403632
 rss=a['rss_kib'];b=dict(mode=a['mode'],qps=a['qps'],N_hit_percent=a['N_hit_percent'],P_hit_all_access_percent=a['P_hit_all_access_percent'],
  disk_reads_per_query=a['disk_reads_per_query'],M_live=a['mcache_storage']['live'],
  M_descriptor_bytes=a['mcache_storage']['edge_descriptor_capacity_bytes'],
  M_descriptor_MiB=a['mcache_storage']['edge_descriptor_capacity_bytes']/1048576,
  additional_RSS_init_MiB=(rss['after_dyn_init']-rss['after_load'])/1024,
  additional_RSS_search_MiB=(rss['after_search']-rss['after_load'])/1024)
 out.append(b)
before,after=out
report=dict(rows=out,descriptor_saved_MiB=before['M_descriptor_MiB']-after['M_descriptor_MiB'],
 additional_RSS_saved_init_MiB=before['additional_RSS_init_MiB']-after['additional_RSS_init_MiB'],
 additional_RSS_saved_search_MiB=before['additional_RSS_search_MiB']-after['additional_RSS_search_MiB'],
 qps_change_percent=(after['qps']/before['qps']-1)*100,passes_each=1,
 RSS_definition='Per-process RSS after cache initialization/search minus RSS after common index load; includes N/M/P and cache metadata, excludes common loaded index RSS.',
 limits='One cold run per allocation mode; descriptor bytes are exact, RSS and QPS are observed values. Savings depend on how many M slots become live; full M occupancy approaches eager storage.')
(R/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
with (R/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(before));w.writeheader();w.writerows(out)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7,4.8));x=[0,1];width=.3
for offset,row,color,label in [(-width/2,before,'#0072b2','Eager allocation'),(width/2,after,'#009e73','Lazy allocation')]:
 values=[row['additional_RSS_init_MiB'],row['additional_RSS_search_MiB']]
 bars=ax.bar([i+offset for i in x],values,width=width,color=color,label=label)
 for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+10,f'{v:.1f}',ha='center',fontsize=11)
ax.set_xticks(x,['Cache initialization','After 100K queries']);ax.set_ylabel('Additional RSS (MiB)')
ax.set_ylim(0,max(a['additional_RSS_search_MiB'] for a in out)*1.18);ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(R/f'sift1b_mcache_lazy_memory.{ext}',dpi=200,bbox_inches='tight')
print(json.dumps(report,indent=2))
