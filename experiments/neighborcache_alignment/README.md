# NeighborCache 共用实现与验证

2026-10-27：已完成当前默认在线 N/M/prefetch/D 路径的共用实现迁移。
PipeANN、AiSAQ、Starling、MARGO 的适配层现在调用同一个
[`neighborcache_backend`](../../neighborcache_backend/README.md)，该库直接编译
`diskann/src/pq_flash_index.cpp` 中的真实 NeighborCache 缓存实现。

这次不再分别仿写 seed 热度门限、M 候选堆、prefetch 管理和 D 替换规则。
适配层负责记录布局、候选来源 ID、实际取得记录后的登记、查询结束和读请求完成事件。
N 的默认 LRU、M 状态机、按图邻接顺序建页、N/stash 取邻居、已有 seed
增量更新、prefetch LRU、脏页淘汰写回、D 读回和有界替换都走参考代码。
新邻居是否能补入 seed 页由图关系和 owner 关系决定，不要求实际遍历父节点就是该 seed。

四个引擎仍产生各自的访问序列，因此“共用缓存实现”不意味着缓存状态、
命中率、ANN 输出或 QPS 必然相同。此次范围是默认在线 LRU 配置；可选
hop/shadow 控制器、更新/过滤索引等未验证模式见后端 README。

## 验证结果

结果目录：`/mnt/nvme2/nc_shared_backend_20261027/final`。
每次运行保留 command.json、二进制和后端 SHA-256、日志和输出；
`summary.json` 是机器可读比较。全部为正确性运行，开启逐条完整记录校验，
不能把日志中的 QPS 当作性能结果。

| 项目 | 结果 |
|---|---|
| NeighborCache、PipeANN、AiSAQ、Starling、MARGO 编译 | 全部通过 |
| 三种记录布局的并发轨迹测试 | 全部通过，各 38,400 次展开，N/P/D、写回、满容量均触发 |
| AIO EINTR/分批完成故障注入 | 普通模式、延迟统计模式均通过 |
| PipeANN 极小 prefetch 压力测试 | 最终版本连续 20 次通过，每次触发 D 读回且完整记录校验通过 |
| PipeANN 访问 profile | 保留原二进制格式并验证写出 |
| AiSAQ public 1K / 重复 5K | IDs 和 float distances 均与重新运行的无 NC 基线逐字节一致 |
| 三个外部适配补丁 | 均能应用于记录的基准版本文件 |

SIFT10M 前 1K 公共查询、K=10/L=128/W=4/T=20 的比较：

| 引擎 | 两次无 NC：相同查询结果行 | 无 NC 对在线 NC：相同结果行 | top-10 集合重合率 | Recall@10：无 NC → 在线 NC |
|---|---:|---:|---:|---:|
| PipeANN | 986/1000 | 973/1000 | 99.66% | 97.00% → 96.91% |
| Starling | 1000/1000 | 998/1000 | 99.99% | 99.82% → 99.82% |
| MARGO | 1000/1000 | 999/1000 | 100% | 99.91% → 99.91% |

PipeANN 无 NC 重复运行本身有差异；这不能证明启用 NC 后的所有差异都只由
调度引起。Starling 有一条查询的结果集合不同，MARGO 的集合相同但有一条
顺序不同。缓存命中记录均与原索引一致，但不声称这三个引擎搜索输出逐字节
等价，也没有完成独立的全内部状态逐事件等价证明。

## 本轮修复的真实问题

- N 和 prefetch 的复制读与并发改写竞争；现有分片锁保护复制。
- 动态页成员与 payload/dirty 发布不同步；发布时统一保护，已有读 pin
  阻止覆盖仍在使用的替换槽。
- 后台维护直接读 D 旧页，遗漏尚在 prefetch 中的最新脏页；现在先读驻留页。
- `io_getevents` 收到 EINTR 时退出进程，导致其他查询线程继续访问已经析构的
  PipeANN 全局 page-cache。修复为等待剩余完成项，不重新提交已完成请求。
  原回溯并非“未分区页锁”本身导致的崩溃；临时 libcuckoo 调试修改已撤回。

## 重现

先按后端 README 编译共享库，重新编译各引擎，再运行：

```sh
python3 experiments/neighborcache_alignment/validate_shared.py
```

脚本默认使用本机已有索引路径。支持 `--root`、`--backend`、`--engines`、
`--pressure-runs`。会裁出一致的 1K 查询和真值，分别运行两次无 NC、一次
在线 NC，再执行 PipeANN 压力测试及 AiSAQ 1K/5K 比较。不会沿用历史
command.json 中过时的 seed/N/prefetch 百分比标签。

## 外部适配补丁

版本和补丁 SHA-256 记录在 `adapters.json`。

- PipeANN 修改直接在本仓库 `pipeann/`，已同步并编译 `/tmp/PipeANN`。
- AiSAQ：见 `../aisaq_nmd/README.md` 和 `aisaq.patch`。
- Starling：`starling_shared.patch`，基准 `68aec13e854219bb2baf39b94d56d2bbda689696`。
- MARGO：`margo_shared.patch`，基准 `485a9c25facea68bb3b6c44eb280b95c14ee5076`。

Starling/MARGO 在对应基准 checkout 中应用补丁，再把下面三个文件复制到
其 include/ 后重新编译：

```text
neighborcache_backend/include/page_neighbor_cache.h
neighborcache_backend/include/nc_client.h
neighborcache_backend/include/nc_backend.h
```

补丁只包含本轮缓存接口/事件/在线启动迁移，不收录外部仓库已有的离线预填装
多页修改或其他构建/测试修改。活动 checkout 中的既有工作仍保留。

`historical/` 保存改造前审计和上轮局部修补，不能作为当前状态或当前测试入口。
旧三数据集 QPS 图和 SIFT100M 扫描图未改动。后续已完成 SIFT10M 和 AiSAQ SIFT100K 的关闭/开启 NC 三轮 QPS 对比，见 [QPS 报告](qps_20261027.md)；SIFT100M、DEEP10M 和 Zipf 尚未重跑。

## Visited hot20 静态预填充（本轮）

结果目录：`/mnt/nvme2/nc_hot20_20261027`。正式结果是固定 D 页和 seed、
关闭 M/seed 补页和替换维护，保留 N/prefetch LRU。每个引擎从自己访问
profile 的非零节点中，取访问次数最高的 `ceil(visited_nodes × 20%)` 个
seed，每个 seed 一页；public 和 ×5 共用同一组 seed。

`build_hot_prefill.py` 生成 canonical 页和成员 metadata；`run_hot20.py`
默认运行基线和静态 hot20；`report_hot20.py` 生成 PNG/PDF 和 CSV/报告。
`hot_online` 仅是额外诊断，不能当成本轮的静态结果。此前误启用动态维护
的运行保留为对照，未混入静态图。

- `visited_hot20_static_qps.pdf` / `.png`：四个引擎 public、×5 的静态结果。
- `REPORT.md` / `summary.csv`：参数、三轮明细、召回率和维护开关对照。
- `COST.md`：独立阶段采样定位；静态 Starling/MARGO 的父边哈希记录为
  已测缓存阶段中的主要开销，其次是 N 插入/复制/LRU。动态 seed 补页的
  锁等待是另一组结果。
- `VALIDATION.json`：按 visited 排序的 seed 列表复核、静态零更新、计时
  不含 VERIFY/PROFILE；`ctest_final.out`：五项回归测试全部通过。

AiSAQ 为 SIFT100K，其余为 SIFT10M。本轮使用同工作负载 profile 预填，
不是 held-out 训练。N/P 测量前为空，离线 D 页直接从原索引读取 seed
及图邻居。完整记录校验与阶段插桩运行均不参加正式 QPS。

## 实际读取后登记父关系；静态无 M（后续修正）

结果目录：`/mnt/nvme2/nc_read_parent_20261027`。上面的历史静态结果虽然
冻结了 seed 更新，仍维护候选发现时的父关系表，并导入 M seed 状态。
后续版本显式 `NEIGHBORCACHE_FREEZE_MAINTENANCE=1` 不再建立 M/seed 替换堆，
静态父关系登记数为零；只读 D 位置索引、N/prefetch LRU 保留。

动态路径使用候选来源字段，仅在处理完整记录时建立父关系；缓存命中也
属于取得记录，不能把这个计数当作物理 I/O。参考 NeighborCache、PipeANN、
AiSAQ、Starling、MARGO 均已修改。共享接口升到 ABI 2，须同时重编译宿主和库。

`run_read_parent.py --phase timing` 用备份的旧二进制/旧库对新静态实现和
新关闭 NC 基线各跑三次；`report_read_parent.py` 生成表格和柱状图。
`prepare_read_parent_profile.py`、`run_read_parent_profile.py` 使用独立插桩库。
新的外部补丁已合入本次变动，仍可应用到 `adapters.json` 中的基准提交。
