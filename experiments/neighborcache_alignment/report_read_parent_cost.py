from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/nvme2/nc_read_parent_20261027')
rows=json.loads((R/'stage_profile/summary.json').read_text());assert len(rows)==8
engines=['pipeann','starling','margo','aisaq'];names=['PipeANN','Starling','MARGO','AiSAQ']
stages=[('visit','Source / previous parent table','#e69f00'),('acquired_parent','Acquired-record callback','#f0cf77'),('resolve','N / prefetch / D lookup','#0072b2'),('complete','D record completion','#56b4e9'),('admit_N_and_seed','N admission / copying / LRU','#009e73'),('selected_M','Selected callback','#cc79a7'),('note_base_stash','Base-load maintenance','#d55e00'),('begin','Query begin','#888888'),('query_end','Query end','#bbbbbb'),('contains','Contains','#444444')]
vals={}
for r in rows:
 n=1000 if r['engine']=='aisaq' else 10000
 vals[r['engine'],r['mode']]={s['stage']:s['estimated_thread_seconds']*1000/n for s in r['stages']}
flat=[]
for (e,m),v in vals.items():
 for stage,_,_ in stages:flat.append(dict(engine=e,mode=m,stage=stage,ms_per_query=v.get(stage,0)))
with (R/'cost.csv').open('w') as f:
 d=csv.DictWriter(f,fieldnames=list(flat[0]));d.writeheader();d.writerows(flat)
fig,ax=plt.subplots(figsize=(11,4.6));x=np.arange(4)
for mi,mode in enumerate(['before','after']):
 bottom=np.zeros(4)
 for stage,label,color in stages:
  y=np.array([vals[e,mode].get(stage,0) for e in engines])
  ax.bar(x+(mi-.5)*.34,y,.32,bottom=bottom,label=label if mi==0 else None,color=color)
  bottom+=y
 for xx,yy in zip(x+(mi-.5)*.34,bottom):ax.text(xx,yy+.01,'Old' if mi==0 else 'New',ha='center',va='bottom',fontsize=9)
ax.set_xticks(x,names);ax.set_ylabel('Sampled cache-stage thread time (ms/query)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,fontsize=9)
fig.text(.01,.02,'Fixed visited-hot20 D; public queries. Includes lock waiting; excludes native search and async I/O wait.',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1))
for ext in ['png','pdf']:fig.savefig(R/f'static_no_m_cost.{ext}',dpi=180,bbox_inches='tight')
lines=['# 去掉静态 M / 父关系表后的开销','',
'单独插桩运行，未用于 QPS 结果。节点阶段每 128 次采样一次，查询阶段每 8 次一次。估计累计线程耗时除以查询数；包含阶段内锁等待，不是 CPU 时间，也不是墙钟时间百分比。原生图搜索、候选/PQ 计算及宿主异步 I/O 等待不包含在图中。',
'','| 引擎 | 原父关系记录 ms/query | 新来源/取得记录回调 ms/query | 原缓存阶段合计 | 新缓存阶段合计 | 新实现最大已测阶段 |','|---|---:|---:|---:|---:|---|']
for e,name in zip(engines,names):
 old,new=vals[e,'before'],vals[e,'after'];big=max(stages,key=lambda t:new.get(t[0],0))
 lines.append(f"| {name} | {old.get('visit',0):.4f} | {new.get('visit',0)+new.get('acquired_parent',0):.4f} | {sum(old.get(s,0) for s,_,_ in stages):.4f} | {sum(new.get(s,0) for s,_,_ in stages):.4f} | {big[1]} ({new.get(big[0],0):.4f}) |")
lines+=['',
'新静态运行父关系表写入次数为零。表中的新回调时间仍包含取得记录计数和检查维护开关，不表示还在维护父关系表。候选携带的来源字段及宿主侧调用开销没有全部纳入这些后端阶段。',
'',
'静态 N 插入/复制/LRU 和 N→prefetch→D 查询仍是必要操作。对比收益包含取消候选发现时的父关系哈希、取消 M 初始化/seed 替换堆、跳过无用动态维护入口；不能把总 QPS 变化全部归因于一个哈希调用。',
'',
'图中排除了嵌套的 seed_note、n_evict 和 global_dyn_write_wait，避免与其外层阶段重复相加。每组仅一次阶段采样，毫秒数用于定位，性能结论使用三轮未插桩 QPS。','']
(R/'COST.md').write_text('\n'.join(lines))
print('\n'.join(lines))
