"""Summarize separately instrumented runs, without using their QPS in plots."""
from pathlib import Path
import json
import statistics as st

ROOT=Path('/mnt/nvme2/nc_hot20_20261027')
rows=json.loads((ROOT/'stage_profile/summary.json').read_text())
names={'pipeann':'PipeANN','starling':'Starling','margo':'MARGO','aisaq':'AiSAQ'}
lines=['# 静态 hot20 的开销定位','',
       '正式配置的 D 页和 seed 固定；N/prefetch 仍执行 LRU。所有静态运行的 opens/appends/replaces/writes 均为零。',
       '主 QPS 图只采用未开启阶段插桩的三轮结果。以下数据来自独立诊断库，每线程对节点阶段每128次采样一次、查询阶段每8次采样一次。',
       '表内为采样估算的累计线程耗时除以查询数，单位 ms/query，包含等待，不是 CPU 时间或墙钟百分比；不含主程序异步 I/O 等待、距离计算和接口调用前的 TLS 查找。','',
       '| 引擎 | 父边记录 visit | N 插入/淘汰 | 查缓存 resolve | D 完成处理 | query begin | query end |',
       '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    if row['mode']!='hot_fixed':continue
    n=1000 if row['engine']=='aisaq' else 10000
    s={x['stage']:x['estimated_thread_seconds']*1000/n for x in row['stages']}
    lines.append('| '+names[row['engine']]+' | '+' | '.join(f'{s[k]:.3f}' for k in ['visit','admit_N_and_seed','resolve','complete','begin','query_end'])+' |')
lines += ['', '## 直接观察与代码解释','',
          '- Starling/MARGO 的主要缓存内部开销是 `NeighborCacheAdapter::visit`：即便冻结动态维护，仍先执行 `profile_parent.insert({child,parent})`，再判断是否维护 M。这在静态运行中保留了大量父边哈希表操作。',
          '- 采样估算每条查询的 visit 调用约为 Starling 23,797 次、MARGO 26,048 次。核查宿主 page_search.cpp 后确认：observe_edge 位于 visited 未命中分支，已见过的邻居不会重复进入该回调；主要成本来自大量首次发现邻居的父关系插入。这些邻居随后可能被候选距离剪枝，不一定展开或读盘。此前“重复边仍进入函数”的解释不适用于本轮 Starling/MARGO 路径。N 插入为第二个大项，包含记录复制和 LRU 更新。',
          '- 静态 seed_note、动态 N 淘汰回调和全局动态目录写锁等待的采样均为零；不能把动态实验的写锁竞争套到静态结果上。',
          '- `begin` 包含 scratch 重置、准备查询和查询对象池的操作；目前未进一步分拆其内部成本。',
          '- 两个页搜索引擎的请求数减少较少：Starling 原程序报告约99.59→90.51次/query，MARGO约99.81→91.63次/query。这是需求读请求统计，已含D读取，不能再加一次D计数，也不等于物理SSD读。',
          '- 因而“少量I/O减少 + 每条查询数万次额外父边记录 + N复制/LRU等开销”与静态QPS仍低约8%–10%的结果一致。阶段采样定位了耗时项，但还未通过逐项删除消融证明每项的独立QPS贡献。','',
          '## 动态维护诊断（不属于静态结果）','',
          '额外动态实验保留同一初始 hot20 页，但允许 M/seed 页更新。它用于解释此前约80%的下降。',
          '这些子阶段嵌套在 N admission 等阶段内，不能相加作为独立占比；全局写锁还包含后台维护线程。','',
          '| 引擎 | N admission 总耗时 ms/query | 其中 seed 补页 ms/query | 全局动态目录写锁累计等待 ms/query |',
          '|---|---:|---:|---:|']
for row in rows:
    if row['mode']!='hot_online':continue
    n=1000 if row['engine']=='aisaq' else 10000
    s={x['stage']:x['estimated_thread_seconds']*1000/n for x in row['stages']}
    lines.append('| '+names[row['engine']]+' | '+' | '.join(f'{s[k]:.3f}' for k in ['admit_N_and_seed','seed_note','global_dyn_write_wait'])+' |')
lines += ['', '动态 seed 补页会持有 seed 锁、检查页成员、读取驻留 prefetch 页，再通过全局 `_neighborcache_dyn_mu` 排他锁发布 payload/成员和位置索引。prefetch 分片不消除这层全局目录锁。',
          '普通日志中的 dyn_unique 将全局锁和位置分片锁合并统计；本轮诊断库另行只计 `_neighborcache_dyn_mu` 的排他锁获取等待，避免把两者混为一谈。','',
          '原始阶段数据：`stage_profile/*/stages.json`、`run.out`；插桩源码及原始源码SHA-256：`stage_profile_source/`。',
          '后续优先针对静态路径跳过无用途的父边/M维护事件和清理工作，再单独重测；本报告没有把这项尚未实施的优化当作已完成结果。','']
(ROOT/'COST.md').write_text('\n'.join(lines))
print(ROOT/'COST.md')
