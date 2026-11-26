# NeighborCache 适配和实验

当前版本和验证结果见[实验台账](EXPERIMENTS.md)。DiskANN 和 PipeANN 在仓库内构建。AiSAQ、Starling 和 MARGO 的上游提交及补丁校验值见 [adapters.json](adapters.json)。

Starling 和 MARGO 在对应基准提交上应用本目录中的补丁，再把 `neighborcache_backend/include` 中的 `nc_backend.h`、`nc_client.h` 和 `page_neighbor_cache.h` 复制到宿主的 `include` 目录后重新编译。AiSAQ 使用[自己的记录适配器](../aisaq_nmd/README.md)。

以下为旧版本实验记录。其运行脚本可从清理前的 Git 提交 `4efb25d` 获取。

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
