"""Current native DiskANN versus the exact historical screenshot logs."""
from pathlib import Path
import csv,hashlib,json,re,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/nvme2/sift100m_figure_repro_20261027')
rows=json.loads((R/'results.json').read_text());assert len(rows)==27
cfgs=json.loads((R/'configurations.json').read_text());summary=[]
for w in ['pub','x5','zipf']:
 for m in ['diskann','neighbor','bfs']:
  group=[r for r in rows if (r['workload'],r['mode'])==(w,m)];assert len(group)==3
  cfg=next(c for c in cfgs if (c['workload'],c['mode'])==(w,m));log=Path(cfg['source_log']).read_text()
  oldfields=re.findall(r'^\s*128\s+4\s+(.+)$',log,re.M)[-1].split()
  q=[r['qps'] for r in group];med=statistics.median(q)
  current_fields=[re.findall(r'^\s*128\s+4\s+(.+)$',(R/f'{w}_{m}_{r["rep"]}/run.out').read_text(),re.M)[-1].split() for r in group]
  summary.append(dict(workload=w,mode=m,historical_qps=cfg['historical_qps'],current_qps=med,change_pct=100*(med/cfg['historical_qps']-1),min_qps=min(q),max_qps=max(q),historical_recall=cfg['historical_recall'],current_recall_min=min(r['recall'] for r in group),current_recall_max=max(r['recall'] for r in group),historical_io=float(oldfields[3]),current_io=statistics.median(r['ios_per_query'] for r in group),historical_io_us=float(oldfields[-3]),current_io_us=statistics.median(float(f[-3]) for f in current_fields),historical_cpu_us=float(oldfields[-2]),current_cpu_us=statistics.median(float(f[-2]) for f in current_fields)))
for r in summary:
 base=next(x['current_qps'] for x in summary if x['workload']==r['workload'] and x['mode']=='diskann')
 r['vs_current_diskann_pct']=100*(r['current_qps']/base-1)
with (R/'comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
(R/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
labels={'pub':'10K','x5':'10K×5','zipf':'10K+Zipf'};names={'diskann':'DiskANN','neighbor':'DiskANN+NC','bfs':'DiskANN-BFS'}
plt.rcParams.update({'font.family':'serif','font.size':11})
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,field,title in zip(axs,['historical_qps','current_qps'],['Historical figure','Current code (median of 3)']):
 for i,(m,color,hatch) in enumerate([('diskann','#bfbfbf',''),('neighbor','#bfbfbf','xx'),('bfs','#7f7f7f','')]):
  s=[next(r for r in summary if r['workload']==w and r['mode']==m) for w in labels]
  v=np.array([r[field] for r in s]);xx=np.arange(3)+(i-1)*.25
  err=np.array([v-[r['min_qps'] for r in s],[r['max_qps'] for r in s]-v]) if field=='current_qps' else None
  ax.bar(xx,v,.23,color=color,edgecolor='#555555',linewidth=.6,hatch=hatch,label=names[m],yerr=err,capsize=2)
  for x,y in zip(xx,v):ax.text(x,y+100,f'{y/1000:.2f}K',ha='center',fontsize=9)
 ax.set_xticks(range(3),labels.values());ax.set_title(title);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.3);ax.set_axisbelow(True)
axs[0].set_ylabel('QPS');axs[0].set_ylim(0,max(r['historical_qps'] for r in summary)*1.2)
fig.legend(*axs[0].get_legend_handles_labels(),ncol=3,loc='upper center',frameon=False)
fig.suptitle('SIFT100M',y=.89);fig.tight_layout(rect=(0,0,1,.83))
for ext in ['png','pdf']:fig.savefig(R/f'diskann_three_configs.{ext}',dpi=180)
lines=['# SIFT100M：原图配置在当前代码上的复测','',
'27 次测试：9 组配置各独立启动三轮，轮换配置顺序；当前值为原生查询计时 QPS 的中位数。原图数值从对应历史日志提取，不是本次测量。',
'','| workload | 配置 | 原图 QPS | 当前 QPS | 相对原图 | 当前最小–最大 |','|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {labels[r['workload']]} | {names[r['mode']]} | {r['historical_qps']:.2f} | {r['current_qps']:.2f} | {r['change_pct']:+.2f}% | {r['min_qps']:.2f}–{r['max_qps']:.2f} |")
lines+=['','## 参数核对','',
'- 与截图一致：K=10、L=128、W=4、T=20、CPU 0–19，同一个SIFT100M uint8/L2索引和原查询/真值文件。',
'- DiskANN无节点缓存；BFS预加载1,081,006节点。NC的N冷启动容量212,371节点，prefetch请求10,059页、实际10,112页；复用每个workload原来的318,557页D缓存，DYNAMIC_3CACHE=0。',
'- 原图的预算是N=20%、prefetch=10%、D seed=30%，visited来自Starling profile。本轮保留这些历史设置，未替换成hot20、5% prefetch或新的公共D页。',
'- 运行当前原生diskann/build/apps/search_disk_index，没有调用四个适配器的共享后端。逐次command.json保存实际环境和二进制SHA-256。',
'- 进程环境清除了继承的NEIGHBORCACHE/PIPEANN/AISAQ/MERIT配置；无完整记录VERIFY或阶段插桩。未清理操作系统页缓存，也未使用sudo/drop_caches。',
'- 旧图二进制未有可确认哈希；当前代码还包含此前其他修改。因此本轮验证的是相同历史配置在当前完整代码上的结果，不能把差值全部归因于最近一次父关系改动。',
'','## 召回率与I/O','', '| workload | 配置 | 原Recall@10 | 当前Recall范围 | 原IO/query | 当前IO/query |','|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {labels[r['workload']]} | {names[r['mode']]} | {r['historical_recall']:.2f} | {r['current_recall_min']:.2f}–{r['current_recall_max']:.2f} | {r['historical_io']:.2f} | {r['current_io']:.2f} |")
lines+=['','## NC 原生计时诊断','',
'下表为每查询累计阶段耗时；程序CPU列虽然标题写作CPU(s)，源码实际打印cpu_us均值，因此这里使用微秒。I/O计时包含该路径等待，不等同于纯SSD硬件服务时间，不能据此单独定位原因。',
'','| workload | 原IO耗时 µs/query | 当前IO耗时 µs/query | 原CPU统计 µs/query | 当前CPU统计 µs/query |','|---|---:|---:|---:|---:|']
for r in summary:
 if r['mode']=='neighbor':lines.append(f"| {labels[r['workload']]} | {r['historical_io_us']:.2f} | {r['current_io_us']:.2f} | {r['historical_cpu_us']:.2f} | {r['current_cpu_us']:.2f} |")
lines+=['','查询结果ID逐行对比见 VALIDATION.json；逐次原始日志、命令、输出都在各 workload_mode_rep 子目录。','']
(R/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(summary,indent=2))
