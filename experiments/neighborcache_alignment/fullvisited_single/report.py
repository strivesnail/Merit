from pathlib import Path
import csv,json,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path('/mnt/nvme2/nc_fullvisited_seed_single_20261031')
raw=json.loads((R/'results.json').read_text());visited=json.loads((R/'visited_summary.json').read_text())
online=json.loads(Path('/mnt/nvme2/nc_n_capacity_match_20261031/selection.json').read_text())['selected']
base=json.loads(Path('/mnt/nvme2/nc_n_capacity_match_20261031/nonly_0_0/result.json').read_text())
mean={key:st.mean(x[key] for x in raw) for key in ('qps','n_hit_pct','p_hit_pct','p_after_n_miss_pct','d_reads_per_query','extra_rss_mib')}
mean['cache_rss_mib']=mean['extra_rss_mib']-base['extra_rss_mib']
summary=dict(full_visited=mean,online_dynamic=online,visited=visited,
             qps_change_vs_online_pct=(mean['qps']/online['qps']-1)*100,
             prefetch_hit_change_pp=mean['p_hit_pct']-online['p_hit_pct'],
             disk_request_change_pct=(mean['d_reads_per_query']/online['disk_reads_per_query']-1)*100,
             n_nodes=2048,n_payload_bytes=794624,prefetch_mib=10,m_entries=0,
             D_static_immutable=True,D_writes=0,base_reads=0,
             scope='same-workload oracle: all full-vector visited nodes become offline seeds; M/dynamic maintenance disabled; only target nodes expanded',
             ids_and_distances_equal_reference=all(x['ids_and_distances_equal_reference'] for x in raw))
if (R/'nonly_match.json').exists():summary['nonly_match']=json.loads((R/'nonly_match.json').read_text())
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'serif','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(9.2,4.2))
labels=['Online D\nM on','Full visited D\nM off']; colors=['#999999','#009e73']
for ax,values,ylabel,fmt in [(axes[0],[online['qps'],mean['qps']],'QPS','{:.0f}'),
                            (axes[1],[online['p_hit_pct'],mean['p_hit_pct']],'Prefetch hits / all accesses (%)','{:.2f}%')]:
 ax.bar(range(2),values,color=colors,width=.55)
 ax.set_xticks(range(2),labels);ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.18)
 ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 for i,y in enumerate(values):ax.text(i,y+max(values)*.025,fmt.format(y),ha='center',va='bottom')
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(R/f'full_visited_vs_online.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
if 'nonly_match' in summary:
 match=summary['nonly_match']; nmd_mib=match['full_d_extra_rss_mib']
 points=[]
 old=Path('/mnt/nvme2/nc_n_capacity_match_20261031')
 for d in list(old.glob('nonly_*/result.json'))+list((R/'nonly_match').glob('nonly_*/result.json')):
  p=json.loads(d.read_text());points.append(p)
 groups={}
 for p in points:groups.setdefault(p['nodes'],[]).append(p)
 a=[]
 for nodes,ps in sorted(groups.items()):a.append(dict(nodes=nodes,qps=st.mean(p['qps'] for p in ps),memory=st.mean(p['extra_rss_mib'] for p in ps)-base['extra_rss_mib']))
 fig,ax=plt.subplots(figsize=(8.4,5.0))
 ax.plot([p['memory'] for p in a],[p['qps'] for p in a],'o-',color='#0072b2',label='N cache only',lw=2,ms=5)
 ax.scatter([nmd_mib],[mean['qps']],marker='D',s=65,color='#009e73',zorder=5)
 ax.axhline(mean['qps'],color='#009e73',ls='--',lw=1.5,label='NMD')
 ax.axvline(nmd_mib,color='#009e73',ls='--',lw=1.5)
 ymax=max(p['qps'] for p in a)*1.13
 xmax=max(p['memory'] for p in a)*1.06
 ax.text(xmax*.985,mean['qps']+ymax*.018,f"{mean['qps']:.2f} QPS",ha='right',va='bottom',color='#007d5b')
 ax.text(nmd_mib+xmax*.012,ymax*.58,f'{nmd_mib:.2f} MiB',rotation=90,ha='left',va='center',color='#007d5b')
 ax.set_xlabel('N cache only memory (MiB)');ax.set_ylabel('QPS')
 ax.set_xlim(0,xmax);ax.set_ylim(0,ymax);ax.grid(alpha=.2);ax.legend(frameon=False,loc='lower right')
 fig.tight_layout()
 for ext in ('png','pdf'):fig.savefig(R/f'nonly_memory_qps.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
 with (R/'nonly_memory_qps.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(a[0]),lineterminator='\n');w.writeheader();w.writerows(a)
print(json.dumps(summary,indent=2))
