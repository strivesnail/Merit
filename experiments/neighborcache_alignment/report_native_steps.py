"""Plot exclusive per-query elapsed stage time; do not use diagnostic QPS."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path('/mnt/nvme2/sift100m_access_steps_20261027');rows=json.loads((R/'measurements.json').read_text());assert len(rows)==3
font_manager.fontManager.addfont('/home/jianz/.fonts/wqy-microhei.ttc')
plt.rcParams.update({'font.family':'WenQuanYi Micro Hei','font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42})
steps=[('query_setup','1  查询初始化 / PQ 距离表','other'),('select_candidate','2  选择待展开候选','search'),('n_lookup','3  查 N / 更新最近访问','cache'),('d_directory','4  查只读 D 索引 / 页路由','cache'),('prefetch_lookup','5  查共享 prefetch','cache'),('d_request_plan','6  D 请求准备 / 同页合并','cache'),('io_prepare_dispatch','7a I/O 请求包装 / 调度','other'),('io_submit','7b 提交 I/O','submit'),('io_wait','7c 等待 D / 原始索引 I/O','wait'),('d_completion','8a 完成后分发页缓冲区','cache'),('page_copy','8b 复制 4 KiB 页到 prefetch','cache'),('prefetch_put','8c prefetch 分区锁 / 插入管理','cache'),('prefetch_eviction','9  prefetch LRU 淘汰','cache'),('decode_record','10a 定位页内节点记录','cache'),('n_admit','10b N 准入 / 复制 / LRU','cache'),('exact_distance','10c 精确向量距离','search'),('pq_distance','10d 邻居 PQ 距离','search'),('discover_candidates','10e 遍历邻居 / visited / 候选更新','search'),('query_finish','12 查询结束统计 / 排序 / 结果','other'),('other','其他循环调度 / 对象生命周期','other')]
colors={'cache':'#0072B2','search':'#009E73','submit':'#E69F00','wait':'#D55E00','other':'#999999'}
worknames={'pub':'10K','x5':'10K×5','zipf':'10K+Zipf'}
data={};flat=[]
for row in rows:
 n=row['queries'];s=row['stages'];total=s['query']['inclusive_ns']/n/1000
 values={name:v['exclusive_ns']/n/1000 for name,v in s.items()};values['other']=values['query']+values['loop_other']
 assert abs(sum(values[k] for k,_,_ in steps)-total)<1e-6
 for key,label,group in steps:
  calls=s[key]['calls']/n if key in s else None
  flat.append(dict(workload=row['workload'],stage=key,label=label,microseconds_per_query=values[key],fraction_pct=100*values[key]/total,calls_per_query=calls))
 data[row['workload']]=(values,total)
with (R/'steps.csv').open('w') as f:
 d=csv.DictWriter(f,fieldnames=list(flat[0]));d.writeheader();d.writerows(flat)
fig=plt.figure(figsize=(15.2,12.5));gs=fig.add_gridspec(2,3,height_ratios=[1.7,8],hspace=.42,wspace=.15)
ax=fig.add_subplot(gs[0,:]);left=np.zeros(3)
for g,label in [('wait','I/O 等待'),('submit','I/O 提交'),('cache','缓存 / D 路由 / 请求管理'),('search','距离 / 候选处理'),('other','初始化 / 调度 / 收尾')]:
 vals=np.array([sum(v for k,v in data[w][0].items() if k in {key for key,_,group in steps if group==g})/1000 for w in worknames])
 ax.barh(range(3),vals,left=left,height=.52,color=colors[g],label=label);left+=vals
for i,w in enumerate(worknames):
 v,total=data[w];ax.text(total/1000+.04,i,f'{total/1000:.3f} ms；等待占 {100*v["io_wait"]/total:.1f}%',va='center',fontsize=10)
ax.set_yticks(range(3),worknames.values());ax.invert_yaxis();ax.set_xlim(0,max(left)*1.43);ax.set_xlabel('每查询累计线程耗时（ms），各项互不重复');ax.legend(loc='upper left',bbox_to_anchor=(0,1.39),ncol=3,frameon=False,fontsize=10)
ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for wi,w in enumerate(worknames):
 ax=fig.add_subplot(gs[1,wi]);v,total=data[w];y=np.arange(len(steps));values=[v[k] for k,_,_ in steps]
 ax.barh(y,values,height=.62,color=[colors[g] for _,_,g in steps])
 ax.set_xscale('log');ax.set_xlim(.1,15000);ax.set_xticks([.1,1,10,100,1000,10000],['0.1','1','10','100','1000','10000']);ax.set_ylim(len(steps)-.5,-.6)
 ax.set_yticks(y,[label for _,label,_ in steps] if wi==0 else ['']*len(steps));ax.tick_params(axis='y',length=0,labelsize=10)
 for yi,value in zip(y,values):ax.text(max(value,.1)*1.12,yi,f'{value:.2f}',va='center',fontsize=9)
 ax.set_title(worknames[w]);ax.set_xlabel('µs / query（对数坐标）');ax.grid(axis='x',which='major',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False)
fig.suptitle('SIFT100M · 静态 DiskANN+NC 访问步骤开销',fontsize=17,y=.98)
fig.text(.5,.947,'同截图配置：T=20，L=128，W=4；N 冷启动，prefetch 10,112 页，D 预填 318,557 页',ha='center',fontsize=10)
fig.text(.025,.055,'第 11 步的原始索引读取与 D 读取共同计入 7a–7c；同批并发等待不按文件强行拆分。',fontsize=10)
fig.text(.025,.034,'离线 D 建页未重新执行；静态 M / 父关系表维护、D 脏页写回均不执行。节点来源 ID 仍用于静态页路由。',fontsize=10)
fig.text(.025,.013,'独立插桩运行；耗时包含等待与计时扰动，不是纯 CPU 时间。下图扣除子步骤，上图为线性总览。',fontsize=10)
fig.subplots_adjust(left=.26,right=.98,top=.86,bottom=.11)
for ext in ['png','pdf']:fig.savefig(R/f'access_steps.{ext}',dpi=180)
# Validate IDs against the current uninstrumented runs, not the historical figure.
def read_ids(p):
 with p.open('rb') as f:
  n,k=np.fromfile(f,np.uint32,2);return np.fromfile(f,np.uint32).reshape(int(n),int(k))
validation=[]
for row in rows:
 w=row['workload'];a=read_ids(R/w/'result_128_idx_uint32.bin');b=read_ids(Path('/mnt/nvme2/sift100m_figure_repro_20261027')/f'{w}_neighbor_0/result_128_idx_uint32.bin')
 validation.append(dict(workload=w,queries=len(a),different_order_rows=int(np.any(a!=b,axis=1).sum()),different_set_rows=int(np.any(np.sort(a,axis=1)!=np.sort(b,axis=1),axis=1).sum()),recall=row['recall'],exclusive_sum_equals_query_total=True))
prod=Path('/home/jianz/workload/code/Merit/diskann/build/apps/search_disk_index');prodsha=hashlib.sha256(prod.read_bytes()).hexdigest();assert prodsha==json.loads((R/'source_hashes.json').read_text())['production_binary_sha256']
(R/'VALIDATION.json').write_text(json.dumps(dict(results=validation,production_binary_unchanged=True,production_binary_sha256=prodsha),indent=2)+'\n')
lines=['# 静态 DiskANN+NC：逐步骤开销','',
'使用当前原生 DiskANN 的独立插桩副本，正式可执行文件和生产源码未改。保持截图参数：SIFT100M，K=10/L=128/W=4/T=20，CPU0–19；N=212371节点、prefetch实际10112页、静态D=318557页。',
'','图：access_steps.png / access_steps.pdf；数据：steps.csv；原始计时：各workload/run.out中的NC_STEP。',
'','## 计时口径','',
'- 每组一次诊断，测完整10K、50K、20K查询；记录每个范围的累计线程elapsed time。不是采样推算，也不是纯CPU时间。',
'- 每个阶段保留inclusive和exclusive；图用exclusive（扣除嵌套子阶段），所有分项之和严格等于Query根计时。I/O等待是各查询线程自己的阻塞时间；多线程累加结果不等于整场墙钟时间。',
'- root从进入cached_beam_search到退出，包含查询初始化和局部对象清理；与原生total_us计时边界不同。其他项明确保留循环调度、容器管理和生命周期开销。',
'- N lookup含选择和处理阶段的重复查找；D目录含准备请求时的重复解析。调用数不是唯一节点数，也不是命中率分母。',
'- 7c计时wait_for_events包含io_getevents等待及完成事件收集；7b是io_submit调用。D与原始索引可能同批并发，无法把联合等待直接相加拆成两份。',
'- 8c是prefetch put扣除整页memcpy和LRU淘汰后的时间，含分区锁等待、槽位申请/回收、映射发布和访问时间更新；不是单纯加锁时间。',
'- 9只是静态干净页的LRU移除，已发布旧槽位的延迟回收也可能发生在后续put；不含D写回。',
'- 10d是PQ距离批处理，10e是邻居visited检查/筛选/候选队列插入；没有把每个邻居都计为完整节点读取。',
'- 离线D构建、本次启动时加载索引/sidecar均不在查询计时内。静态M、父关系热度表、seed替换、D写回均不执行，不能为其绘制非零查询成本。',
'- 全量计时有扰动；这些诊断QPS不加入性能对比。正式QPS沿用未插桩运行。','',
'| 步骤（µs/query） | 10K | 10K×5 | 10K+Zipf |','|---|---:|---:|---:|']
for key,label,_ in steps:lines.append('| '+label+' | '+' | '.join(f'{data[w][0][key]:.3f}' for w in worknames)+' |')
lines+=['| 合计 | '+' | '.join(f'{data[w][1]:.3f}' for w in worknames)+' |','',
'## 与未插桩运行的输出检查','']
for v in validation:lines.append(f"- {worknames[v['workload']]}：Recall@10={v['recall']:.2f}%，{v['queries']}条查询中结果ID顺序不同{v['different_order_rows']}条，top10集合不同{v['different_set_rows']}条。")
lines+=['','## 重现','',
'依次运行 experiments/neighborcache_alignment/prepare_native_steps.py、build_native_steps.py、run_native_steps.py、report_native_steps.py。准备脚本复制源码到本结果目录后插桩；构建脚本复用现有静态库并在副本中替换插桩对象，正式二进制SHA-256前后核对一致。','']
(R/'REPORT.md').write_text('\n'.join(lines))
for w in worknames:
 v,total=data[w];print(w,'total_us',total,'wait_pct',100*v['io_wait']/total,'prefetch_fill_us',v['page_copy']+v['prefetch_put']+v['prefetch_eviction'])
print(json.dumps(validation,indent=2))
