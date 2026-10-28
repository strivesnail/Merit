"""Render the timing records produced by compare_qps.py, without changing measurements."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
r=a.root;rows=json.loads((r/'summary.json').read_text())
names={'pipeann':'PipeANN','starling':'Starling','margo':'MARGO','aisaq':'AiSAQ'}
lines=['# 共用 NeighborCache 后的 QPS 对比（2026-10-27）','',
       '每个配置三轮，关闭完整记录校验和 profile，分别启动新进程，NC 从空缓存在线填充。',
       '相同引擎的基线和 NC 使用同一个已验证的二进制、索引、查询、线程数和搜索参数；',
       '执行顺序为 off/on、on/off、off/on，所有运行串行。下表 QPS、召回率取各自三轮中位数。',
       '变化比例 = (开启 NC 的 QPS / 关闭 NC 的 QPS − 1) × 100%。','',
       '| 引擎 | 数据集 / 查询 | 关闭 NC QPS | 开启 NC QPS | QPS 变化 | Recall@10（关闭 → 开启） |',
       '|---|---|---:|---:|---:|---:|']
for v in rows:
    n='10K' if v['dataset']=='SIFT10M' else '1K'
    work=n if v['workload']=='public' else n+' ×5'
    lines.append(f"| {names[v['engine']]} | {v['dataset']} / {work} | {v['off_qps']:,.2f} | {v['online_qps']:,.2f} | {v['change_pct']:+.2f}% | {v['off_recall_median']:.2f}% → {v['online_recall_median']:.2f}% |")
lines += ['', '## 参数与范围','',
          '- PipeANN：SIFT10M，K=10/L=128/W=4，20 线程，N=31 MiB，prefetch=16 MiB。',
          '- Starling/MARGO：SIFT10M，K=10/L=128/W=4，20 线程，N=91,842 节点，prefetch=4,350 个 4 KiB 页；mem_L=10，use_ratio=1.0。',
          '- AiSAQ：SIFT100K，K=10/L=64/W=4/V=1，8 线程，N=1 MiB，prefetch=1 MiB；full-inline PQ、无 ID 重排。',
          '- D 页容量采用当前参考实现的容量规则。运行日志中的上限：PipeANN 133,334 页；Starling/MARGO 200,001 页；AiSAQ 4,001 页。',
          '- 前三个引擎绑定 CPU 0–19，AiSAQ 绑定 CPU 0–7；每个基线/NC 对使用相同绑定。',
          '- ×5 是公共查询顺序完整重复五遍；单次进程内缓存延续，下一轮重新为空。没有额外预热，也没有清理系统 page cache。',
          '- QPS 使用各引擎原有搜索计时区间；不包括加载索引和退出清理，不是端到端进程吞吐。',
          '- AiSAQ 使用不同规模和参数，只能比较它自身关闭/开启 NC 的差异，不能与另外三个引擎的绝对 QPS 横比。',
          '- 此次没有跑 SIFT100M、DEEP10M 或 Zipf，也未改写此前的图。','',
          '## 三轮明细','', '| 引擎 | 查询 | 关闭 NC 三轮 QPS | 开启 NC 三轮 QPS |','|---|---|---|---|']
for v in rows:
    lines.append('| '+names[v['engine']]+' | '+v['workload']+' | '+', '.join(f'{x:.2f}' for x in v['off_qps_runs'])+' | '+', '.join(f'{x:.2f}' for x in v['online_qps_runs'])+' |')
lines += ['', '## 原始记录','',
          f'目录：`{r}`。`summary.csv`/`summary.json` 为汇总，`repeats.json` 为逐轮结果；',
          '每次运行目录包含 command.json（含文件 SHA-256 和返回码）、run.out、measurement.json。',
          '报告不从上轮开启校验的日志中取 QPS。AiSAQ 的 VERIFY 变量按存在性判断，正式计时必须将其删除，设置为 0 不足以关闭。',
          '若目录存在 excluded_verification_enabled/，其中是发现该开关语义后排除的 AiSAQ 初次运行，不参与本报告。',
          'reported_ios_per_query 仅保留引擎输出的 I/O 统计，其含义随引擎不同；cache_stats 是参考缓存计数，均不等同物理 SSD I/O。','']
(r/'REPORT.md').write_text('\n'.join(lines))
print(r/'REPORT.md')
