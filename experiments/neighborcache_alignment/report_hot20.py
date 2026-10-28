"""Plot measured visited-hot20 runs; verification/profile runs are excluded."""
from pathlib import Path
import csv
import json
import re
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/mnt/nvme2/nc_hot20_20261027')
ENGINES = ['pipeann', 'starling', 'margo', 'aisaq']
NAMES = dict(zip(ENGINES, ['PipeANN', 'Starling', 'MARGO', 'AiSAQ']))
rows = [json.loads(p.read_text()) for p in ROOT.glob('*/measurement.json')]
rows = [r for r in rows if not r['verification']]
groups = {}
for row in rows:
    groups.setdefault((row['engine'], row['workload'], row['mode']), []).append(row)
def values(engine, work, mode, field='qps'):
    return [r[field] for r in groups[(engine, work, mode)]]
def median(engine, work, mode, field='qps'):
    return st.median(values(engine, work, mode, field))
summary = []
for e in ENGINES:
    for w in ['public', 'x5']:
        for mode in ['off', 'hot_online', 'hot_fixed', 'cold']:
            if (e, w, mode) not in groups:
                continue
            v = values(e, w, mode)
            assert len(v) == 3, (e, w, mode, len(v))
            summary.append(dict(engine=e, workload=w, mode=mode, qps=st.median(v),
                qps_min=min(v), qps_max=max(v), qps_runs=v,
                recall=median(e,w,mode,'recall'),
                change_vs_off_pct=(st.median(v)/median(e,w,'off')-1)*100))
(ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
with (ROOT/'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader(); writer.writerows(summary)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig, axes = plt.subplots(2,2,figsize=(10,6.8),layout='constrained')
for e, ax in zip(ENGINES, axes.flat):
    for mode, label, color, shift in [('off','Baseline','#888888',-.18),('hot_fixed','NC: static visited hot 20%','#009E73',.18)]:
        vs=[values(e,w,mode) for w in ['public','x5']]
        med=np.array([st.median(x) for x in vs])
        err=np.array([med-np.array([min(x) for x in vs]),np.array([max(x) for x in vs])-med])
        ax.bar(np.arange(2)+shift,med,width=.34,label=label,color=color,yerr=err,capsize=3)
    ax.set_xticks([0,1],['Public 1K','Public 1K ×5'] if e=='aisaq' else ['Public 10K','Public 10K ×5'])
    ax.set_title(NAMES[e]+(' · SIFT100K' if e=='aisaq' else ' · SIFT10M'))
    ax.set_ylabel('QPS');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.22)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncols=2,frameon=False)
for ext in ['png','pdf']:fig.savefig(ROOT/f'visited_hot20_static_qps.{ext}',dpi=180)
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for e,ax in zip(ENGINES,axes.flat):
    modes=['off','cold','hot_online','hot_fixed']
    vs=[values(e,'public',m) for m in modes]
    med=np.array([st.median(x) for x in vs])
    err=np.array([med-np.array([min(x) for x in vs]),np.array([max(x) for x in vs])-med])
    ax.bar(np.arange(4),med,color=['#888888','#D55E00','#009E73','#0072B2'],yerr=err,capsize=3)
    ax.set_xticks(range(4),['Baseline','Empty D\n+ online','Hot20 D\n+ online','Hot20 D\nfixed'])
    ax.set_title(NAMES[e]+(' · SIFT100K / 1K' if e=='aisaq' else ' · SIFT10M / 10K'))
    ax.set_ylabel('QPS');ax.set_axisbelow(True);ax.grid(axis='y',alpha=.22)
for ext in ['png','pdf']:fig.savefig(ROOT/f'visited_hot20_maintenance.{ext}',dpi=180)
plt.close(fig)

lines=['# Visited hot top 20% 静态 D-cache 实测','',
       '正式结果固定预填 D 页和 seed，测量期间关闭 M/seed 补页/替换维护，保留 N/prefetch LRU。hot_online 是额外的动态维护诊断，不是本轮要求的静态结果。','',
       '所有 QPS 为三次独立进程的中位数，误差线为最小/最大值。正式运行不启用逐条记录校验或阶段 profile。',
       '每个引擎用自己的 profile 中 count>0 的节点作为 visited nodes，按 count 降序、ID 升序打破并列，取 ceil(visited×0.2) 个 seed。每个 seed 一页。','',
       '| 引擎 | visited nodes | hot20 seeds / D 页容量 |','|---|---:|---:|']
for e in ENGINES:
    m=json.loads((ROOT/'prefill'/f'{e}.json').read_text())
    assert m['seeds']==(m['visited_nodes']+4)//5
    lines.append(f"| {NAMES[e]} | {m['visited_nodes']:,} | {m['seeds']:,} |")
lines += ['', '## QPS：静态 hot20 D 页','',
          '| 引擎 | 查询 | 基线 QPS | 静态 hot20 QPS | 相对基线 | Recall@10 基线 → hot20 |',
          '|---|---|---:|---:|---:|---:|']
for e in ENGINES:
    for w in ['public','x5']:
        off=median(e,w,'off');on=median(e,w,'hot_fixed')
        lines.append(f"| {NAMES[e]} | {w} | {off:,.2f} | {on:,.2f} | {(on/off-1)*100:+.2f}% | {median(e,w,'off','recall'):.2f}% → {median(e,w,'hot_fixed','recall'):.2f}% |")
lines += ['', '## 维护开关对照（公共查询，三轮中位数）','',
          'cold 与 hot_online、hot_fixed 的 D/M/N/P 容量一致。hot_fixed 关闭 M/pair/heap、seed 补页和替换维护，保留 N/P LRU 和 D 读取。',
          '参考实现的 freeze 同时启用成员 bitset 的快速负查；因此是整条维护路径的对照，不能把增益全部归因于某一把锁。','',
          '| 引擎 | 无 NC | 空 D + online | hot20 + online | hot20 fixed | fixed / online |',
          '|---|---:|---:|---:|---:|---:|']
for e in ENGINES:
    v=[median(e,'public',m) for m in ['off','cold','hot_online','hot_fixed']]
    lines.append('| '+NAMES[e]+' | '+' | '.join(f'{x:,.2f}' for x in v)+f' | {v[3]/v[2]:.2f}× |')
lines += ['', '## 实验边界与重现','',
          '- PipeANN/Starling/MARGO 为 SIFT10M，public=10K、×5=50K；AiSAQ 为 SIFT100K，public=1K、×5=5K。AiSAQ 不能与其余三者直接横比绝对 QPS。',
          '- 使用相同 public 工作负载的访问 profile 为 public 和 ×5 选择同一组 seed；这是已知工作负载的预填充实验，不是独立训练集泛化结果。',
          '- 离线页为 seed 加按原图邻接顺序选出的邻居，直到页满，离线读取原索引 payload。在线建页仍从 N/stash 取记录。N 和 prefetch 在测量前均为空。',
          '- PipeANN：K10/L128/W4/T20，N=31 MiB（125,022 节点），P=16 MiB。Starling/MARGO：K10/L128/W4/T20，N=91,842 节点，P 请求4,350页，实际分片取整4,352页。AiSAQ：K10/L64/W4/V1/T8，N/P 各1 MiB。',
          '- 每个引擎的 D 容量显式设为本轮 visited hot20 seed 数；M 容量为该数两倍。与旧“20%原索引页数”的默认 D 容量不同，旧结果没有混入新图。',
          '- CPU 绑定、查询、索引、主程序在每个对照内一致，所有计时运行串行。未清理系统 page cache；缓存索引/动态 D 采用现有 I/O 路径。',
          '- QPS 取原程序搜索计时，不包含离线建页、导入、退出清理。完整记录校验的八次运行只用于正确性，不参与 QPS。',
          '- 普通动态命中统计只累计实际 D lookup；N/P 命中和 fast-negative-filter 跳过不计入分母。未用该命中率替代物理 I/O 统计。',
          '- 预填充固定页对照要求 opens/appends/replaces/writes 为零；脚本自动校验。后台 writer 统计可能含导入阶段，不能直接解释为查询期写盘耗时。',
          '- 每次 command.json 保存参数、二进制/后端 SHA-256 和返回码，run.out 保存完整参考缓存/锁统计。summary.csv/json 保存三轮数据。','',
          '```sh',
          'python3 experiments/neighborcache_alignment/run_hot20.py --phase verify',
          'python3 experiments/neighborcache_alignment/run_hot20.py --phase timing',
          'python3 experiments/neighborcache_alignment/run_hot20.py --phase diagnose --workloads public --modes cold hot_online',
          '/home/jianz/miniconda3/bin/python experiments/neighborcache_alignment/report_hot20.py',
          '```','',
          '脚本拒绝覆盖已有运行目录；重复执行需换 ROOT 或归档旧目录。','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(ROOT/'REPORT.md')
