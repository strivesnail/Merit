"""Report same-session old/new static timings; profiles are diagnostic only."""
from pathlib import Path
import csv,json,statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/nvme2/nc_read_parent_20261027')
engines=['pipeann','starling','margo','aisaq']; names=['PipeANN','Starling','MARGO','AiSAQ']
rows=[json.loads(p.read_text()) for p in R.glob('*_*/measurement.json') if not p.parent.name.startswith(('verify','verified'))]
rows=[r for r in rows if r['mode'] in ['before','after','off']]
assert len(rows)==72,len(rows)
groups={(e,w,m):[r for r in rows if (r['engine'],r['workload'],r['mode'])==(e,w,m)] for e in engines for w in ['public','x5'] for m in ['off','before','after']}
summary=[]
for e in engines:
 for w in ['public','x5']:
  g={m:groups[e,w,m] for m in ['off','before','after']}
  assert all(len(v)==3 for v in g.values())
  q={m:st.median(r['qps'] for r in rr) for m,rr in g.items()}
  summary.append(dict(engine=e,workload=w,**{m+'_qps':v for m,v in q.items()},change_pct=(q['after']/q['before']-1)*100,vs_off_pct=(q['after']/q['off']-1)*100,
   before_io=st.median(r['reported_ios_per_query'] for r in g['before']),after_io=st.median(r['reported_ios_per_query'] for r in g['after']),
   before_recall_min=min(r['recall'] for r in g['before']),before_recall_max=max(r['recall'] for r in g['before']),
   after_recall_min=min(r['recall'] for r in g['after']),after_recall_max=max(r['recall'] for r in g['after'])))
with (R/'summary.csv').open('w') as f:
 d=csv.DictWriter(f,fieldnames=list(summary[0]));d.writeheader();d.writerows(summary)
(R/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,w in zip(axs,['public','x5']):
 for i,(m,label,c) in enumerate([('off','No NC','#9da3aa'),('before','Previous static NC','#d98c3f'),('after','Static NC: no M / parent table','#009e73')]):
  vals=np.array([st.median(r['qps'] for r in groups[e,w,m]) for e in engines]);lo=[min(r['qps'] for r in groups[e,w,m]) for e in engines];hi=[max(r['qps'] for r in groups[e,w,m]) for e in engines]
  ax.bar(np.arange(4)+(i-1)*.25,vals,.24,label=label,color=c,yerr=np.array([vals-lo,hi-vals]),capsize=2)
 ax.set_xticks(range(4),names);ax.set_title('Public queries' if w=='public' else 'Public queries ×5');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axs[0].set_ylabel('QPS (median of 3 runs)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
fig.text(.5,.01,'SIFT10M: PipeANN / Starling / MARGO; SIFT100K: AiSAQ. Error bars: min–max.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.90))
for ext in ['png','pdf']:fig.savefig(R/f'static_no_m_qps.{ext}',dpi=180)
lines=['# 静态路径去掉 M 和父关系记录：同场重跑','',
'固定 hot top 20% visited seed 和 D 页。N/prefetch 的 LRU 保留；新静态模式不分配 M 热度条目/seed 替换堆，不记录父关系。D 的节点→页位置索引保留。',
'','每组独立启动三次、轮换 before/after/off 顺序，报告 QPS 中位数；错误条为三次最小值到最大值。计时关闭完整记录校验及阶段插桩。',
'','| 引擎 | 查询 | 无 NC | 修改前 NC | 修改后 NC | 修改后/修改前 | 修改后/无 NC |','|---|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['engine']} | {r['workload']} | {r['off_qps']:.2f} | {r['before_qps']:.2f} | {r['after_qps']:.2f} | {r['change_pct']:+.2f}% | {r['vs_off_pct']:+.2f}% |")
lines+=['','## 可比性和范围','',
'- PipeANN、Starling、MARGO 为 SIFT10M public10K/重复50K；AiSAQ 为 SIFT100K public1K/重复5K。未将 AiSAQ 标为 SIFT10M。',
'- before 使用备份旧可执行文件和 ABI1 库；after/off 使用新可执行文件，after 使用 ABI2 库。逐次 command.json 记录参数和二进制 SHA-256。',
'- 两侧复用同一组不可变预填充页和 seed 文件；seed 数是各自 profile 非零 visited 节点数的 ceil(20%)。这是同工作负载 profile，不是 held-out。',
'- 新候选携带父节点 ID，可能改变结构体大小；新 off 基线也采用相同新可执行文件。收益包含移除发现候选时的父关系哈希、静态 M 数据和多余维护调用，不仅是移动一次回调。',
'- 静态 after 全部断言 M=0、recorded_parents=0、opens/appends/replaces/writes=0。缓存记录取得计数是诊断计数，缓存命中也算取得记录，不等于物理 I/O。',
'- 动态路径仍维护 M；只有消费完整记录后才登记父关系，候选携带来源 ID 不算登记。',
'- NeighborCache 本体已编译、1K 静态输出验证；四个适配器完整记录校验通过。动态记录语义有单元测试和三种布局并发轨迹测试。',
'','## 召回率与 I/O','', '| 引擎 | 查询 | 原 NC recall 范围 | 新 NC recall 范围 | 原 IO/query | 新 IO/query |','|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['engine']} | {r['workload']} | {r['before_recall_min']:.2f}–{r['before_recall_max']:.2f} | {r['after_recall_min']:.2f}–{r['after_recall_max']:.2f} | {r['before_io']:.2f} | {r['after_io']:.2f} |")
lines+=['','PipeANN 并发搜索本身存在小幅输出波动；不声称所有引擎逐字节搜索等价。Starling 新旧输出集合最多相差 1/10000、3/50000 条查询；旧版重复运行也有差异。MARGO public IDs 一致，×5 最多 2 条仅顺序不同；AiSAQ IDs 全部一致。详细比较见 output_comparison.json。各引擎原生 IO 列按其自身统计口径展示，不跨引擎比较。','']
(R/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
