# MERIT 缓存与 I/O 实验台账

这份台账收录了本机上已核实的 SpaceV100M 缓存实验。按实验问题查下面的表，再打开原始结果。全量 `500k×10` 共 500 万次查询，冷前缀只取其中前 50 万次。`10k×10` 是另一份共 10 万次查询的 trace。三者分别比较。

开跑前核对 trace 指纹和截取范围、代码或二进制版本、缓存参数、查询和写线程数、文件落盘位置、清缓存与 TRIM 条件。已有逐查询日志能回答的问题，直接重算指标。新实验先记运行中，完成后补数据。取消或失败的运行保留状态。

从 2026-11-26 起，每次新测量都按[仓库实验规则](../../AGENTS.md)在计时前自动 TRIM 所有参与 I/O 的实验 SSD。每次重复测量单独执行，将运行目录的 `trim.log` 随结果登记。下表的 TRIM 条件记录各历史轮次的实际操作。

## 第三跳 PathSeed 的全量结果

SpaceV100M，`500k×10`，10 个查询线程，L=32，beam=8。从第三跳开始当 seed。N-cache 为 765,308 个节点、64 分片 LRU，共享 prefetch 容纳 3,200 页，D-cache 无容量上限，前台队列阈值为 1,024 页。完整 trace 的 SHA-256 为 `a1ca9d95c82dba62083d6785cb9517e1b7b456a1166345504cf26c2b5c4b54f1`。下表 I/O 是每查询前台读路径经过时间，单位 µs。AIO 诊断表分别报告全量查询的 I/O 均值和 1/32 查询抽样的系统调用计时。

| 写线程 | 原始图 / D 页布局 | 开跑前 TRIM | QPS | 读 I/O µs/查询 | 读页/查询 | 结果或状态 |
|---:|---|---|---:|---:|---:|---|
| 1 | 单盘，原始开销图 | nvme2，1.0 GiB | 9,497.87 | 554.66 | 26.93645 | [第 2/3 跳对比](/mnt/nvme2/merit-third-hop-500k.Ir5hPU/comparison.csv) · [开销图](/mnt/nvme2/merit-third-hop-500k.Ir5hPU/overhead/merit_pre_ebr_500kx10.png) |
| 1 | 单盘，细分前台计时 | nvme2，1.1 GiB | 9,450.64 | 554.58 | 26.93645 | [结果](/mnt/nvme2/merit-residual-500k/shared-1/result.json) · [368.8 µs 拆分](/mnt/nvme2/merit-residual-500k/detail.csv) |
| 4 | 单盘，前台计时 | nvme2，24.8 GiB | 9,328.68 | 707.74 | 26.93644 | [结果](/mnt/nvme2/merit-4writer-full500k/shared-1/result.json) · [开销排名](/mnt/nvme2/merit-4writer-full500k/ranking.csv) |
| 4 | 单盘，复测并细分 AIO | nvme2，35.5 GiB | 9,331.07 | 708.67 | 26.93644 | [逐阶段及设备指标](/mnt/nvme2/merit-io-diagnosis-full/writer4-1/phases.csv) |
| 4 | 原始图全在 nvme0，D 页全在 nvme2 | nvme2，25.7 GiB | 9,519.40 | 686.04 | 26.93644 | [结果](/mnt/nvme2/merit-io-diagnosis-full/dual4-1/result.json) · [两盘计数器](/mnt/nvme2/merit-io-diagnosis-full/dual4-1/devices.csv) |
| 4 | 单盘 nvme2 | 本轮跳过 | 7,050.35 | 1,045.58 | 26.93641 | [逐阶段及设备指标](/mnt/nvme2/merit-io-diagnosis-full/single4notrim-1/phases.csv) |
| 4 | 原始图和 D 页各自按 4 KiB 页号奇偶均分到 nvme2、nvme0 | 两盘本轮均跳过 | 8,918.15 | 774.30 | 26.93642 | [校验、页数与写次数](/mnt/nvme2/merit-io-diagnosis-full/stripe4-2/result.json) · [配置](/mnt/nvme2/merit-io-diagnosis-full/stripe4-2/command.json) |
| 4 | 原始图和 D 页按 4 KiB 页号均分 | nvme0 406.8 GiB，nvme2 67.1 GiB | 11,320.61 | 536.43 | 26.93643 | 已完成并通过 Top-10 ID、距离与写队列核验。[结果](/mnt/nvme2/merit-io-diagnosis-full/stripe4-bothtrim1/result.json) · [TRIM 回执](/mnt/nvme2/merit-io-diagnosis-full/stripe4-bothtrim1/trim.log) |

表中五种已完成的 4 写 AIO 配置都先对索引和查询文件做 `POSIX_FADV_DONTNEED`，用 `mincore` 确认驻留页数为零，并等待 20 秒。“本轮跳过”仅指开跑前没有执行 TRIM，SSD 仍有此前运行留下的状态。单盘有 TRIM 的两轮结果接近。单盘跳过 TRIM 的这轮读 I/O 高出约 337 µs/查询。原始图单独放在第二盘时，前 50 万次查询 QPS 提高约 5%，最后 50 万次基本持平。[清理与运行日志](/mnt/nvme2/merit-io-diagnosis-full/placement.log)

### 按页均分的测量结果

2026-11-26 完成。原始图和 D 页各自用逻辑页号的最低位选盘。偶数页在 nvme2，奇数页在 nvme0，盘内页号为逻辑页号右移一位。四个后台线程按此映射写页，同一页的后续更新继续落在同一块盘。全量 Top-10 ID 和距离的 SHA-256 与单盘基线一致，后台已排空并核对全部 9,269,112 个 D 页。[图分片校验](/mnt/nvme2/merit-io-diagnosis-full/stripe-build/graph-verified.json) · [D 页读写实现](/mnt/nvme2/merit-io-diagnosis-full/stripe-build/paired_cache.h) · [查询校验](/mnt/nvme2/merit-io-diagnosis-full/stripe4-2/result.json)

| 全量统计 | nvme2 偶数页 | nvme0 奇数页 |
|---|---:|---:|
| 原始图存放页数 | 4,545,456 | 4,545,455 |
| D-cache 存放页数 | 4,634,556 | 4,634,556 |
| 后台实际写 D 页次数（含更新） | 6,095,801 | 6,092,628 |
| 设备读取 MB/s | 496.32 | 495.96 |
| 设备写入 MB/s | 44.54 | 44.49 |

开跑前均跳过 TRIM 的条件下，相比单盘，均分双盘全量 QPS 提高 26.5%，前台读 I/O 降低 25.9%。前 50 万次 QPS 从 3,043 升至 4,793，最后 50 万次从 9,737 升至 11,219。两组的 N-cache 和 prefetch 命中率分别约为全部节点访问的 33.823% 和 28.3645%。抽样每查询提交读请求由 279.9 降到 131.2 µs，等待完成由 726.5 降到 601.8 µs。每个布局各有一次可校验的无 TRIM 全量结果，各 SSD 此前写入历史不同。[全量与分阶段对照](/mnt/nvme2/merit-io-diagnosis-full/stripe-comparison.csv)

双盘均分、两盘均 TRIM 的一次全量测量为 11,320.61 QPS、536.43 µs 读 I/O/查询。相较同布局本轮跳过 TRIM，QPS 提高 26.9%，读 I/O 降低 30.7%，抽样每查询的 AIO 完成等待从 601.80 缩短到 380.15 µs。相较已 TRIM 的单盘，QPS 提高 21.3%。前 50 万次查询为 6,358 对 3,837 QPS，最后 50 万次为 13,263 对 12,676 QPS。查询期间 nvme2、nvme0 平均分别读取 629.70、629.89 MB/s，写入 56.51、56.52 MB/s。nvme0 由操作者执行 TRIM，块设备 discard 字节增量与回执一致。完整 trace 和双盘程序指纹均与双盘无 TRIM 组相同。[分阶段与设备对照](/mnt/nvme2/merit-io-diagnosis-full/stripe4-bothtrim1/placement-comparison.csv)

双盘均分且两盘 TRIM 的前台开销为 852.57 µs/查询。全量按搜索读 I/O、已计时搜索 CPU、其他前台耗时分为 536.43、100.39、215.74 µs。156,250 个查询的细分样本中，其他前台耗时平均为 240.27 µs，CPU 分段取整计入该项。[开销条形图](../../figures/spacev100m_500kx10_dual_trim_latency_breakdown.png) · [分项数据](../../figures/spacev100m_500kx10_dual_trim_latency_breakdown.csv)

单盘与双盘均分的 TRIM 对照曲线按连续 50 万次查询分为 10 段，纵轴为当段查询数除以当段墙钟耗时。[QPS 曲线](../../figures/spacev100m_500kx10_trim_single_dual_qps_per_500k.png) · [逐段数据](../../figures/spacev100m_500kx10_trim_single_dual_qps_per_500k.csv)

另有一轮 1 写线程复测于查询期间按要求停止，不计入性能比较。需要 1 写数据时使用上表两份已完成的记录。[停止记录](/mnt/nvme2/merit-io-diagnosis-full/writer1-1/stopped.txt)

首次按页均分的实验在最终统计处异常退出。最后的查询输出未落盘，无法校验，首轮不计 QPS。统一编译统计入口和 D-cache 后，先用前 2,048 次查询校验 Top-10 ID、距离和写队列排空，再启动全量复测。[失败记录](/mnt/nvme2/merit-io-diagnosis-full/stripe4-1/failed.json) · [修复验证](/mnt/nvme2/merit-io-diagnosis-full/stripe-smoke/passed.txt)

## 第三跳 PathSeed 的 D 页填充率

SpaceV100M 的每条节点记录占 360 字节，PathSeed 每个 4 KiB 页最多使用 10 个槽，其中 1 个固定给 seed。用已保存的 50 万条不同查询的扩展顺序和原图邻接表重建首次发现的父子边，对同一 seed 的孩子去重并按每页 9 个孩子打包。得到 8,989,555 个 seed、22,840,547 条不同父子边，重建页数 9,269,112，与双盘均分的 500k×10 运行结束时清点的页数一致。按最终页数等权，每页平均 3.464 个节点（含 seed），中位数 3 个，2 个节点的页占 45.23%，满 10 个的页占 3.84%。[填充分布](/mnt/nvme2/merit-io-diagnosis-full/occupancy/histogram.csv) · [离线复算代码](/mnt/nvme2/merit-io-diagnosis-full/occupancy/replay.cpp)

查询 39930，复用双盘均分可执行文件从空缓存跑一次。82 个展开节点中 61 个出现在排空后的 D 文件中。第 4 跳及以后的 57 个都在对应父页，另有 4 个第 3 跳父节点作为 seed。27 张实际页中 13 张只含 1 个孩子。全 trace 离线统计为 8,765,579 个单页 seed、184,774 个双页 seed 和 39,202 个超过两页的 seed（最多 7 页）。以 trace 首次出现顺序模拟补页顺序，在最终恰好两页的 seed 上，970,924 个（不同查询，父 seed）组合中有 231,313 个所用孩子分布在两页；不计缓存路由或 I/O。[逐节点实测](/mnt/nvme2/merit-io-diagnosis-full/query39930-pages/node-coverage.csv) · [实盘页成员](/mnt/nvme2/merit-io-diagnosis-full/query39930-pages/coverage.json) · [离线跨页计数](/mnt/nvme2/merit-io-diagnosis-full/query39930-pages/multipage.txt)

按需相邻双页已完成一次全量验证。500k×10、4 写线程、两盘均 TRIM（本轮各约 18.1 GiB），复用双盘基线。新布局在同一 seed 扩页时使用另一盘的配对页，用一位记录有无第二页，保留溢出，仅对当前批次有需求的页提交读取。基线与新布局的 QPS 分别为 11,320.61 和 11,291.53（−0.26%），P99 为 2,268 和 2,403 µs，搜索读 I/O 为 536.43 和 529.32 µs/查询。写队列等待抽样均值从 11.13 增至 21.51 µs/查询。新布局抽样的 993,694 个含实际 D 页读取的批次中，3,864 个同时读取同一对中的两页（0.389%）。实际写入的 D 页共有 9,269,112 张，236,826 对相邻页全部通过核验，39,202 个超过两页的 seed 均保留，Top-10 ID 和距离校验通过。两盘 D 文件的逻辑长度合计 68.91 GiB，文件系统实际分配 35.54 GiB。[逐阶段对照](/mnt/nvme2/merit-io-diagnosis-full/adaptive-1/comparison.csv) · [计数与校验](/mnt/nvme2/merit-io-diagnosis-full/adaptive-1/result.json) · [布局改动](/mnt/nvme2/merit-io-diagnosis-full/adaptive-build/layout.patch)

查询 39930 首次冷访问后的 27 个 seed 页离线试装。每页最多 10 个不同节点，每次只配两个 seed，按访问顺序贪心配对。要求两个 seed 在完整展开流中紧邻时得到 17 页；仅要求在 seed 序列中相邻时得到 14 页。两个口径都保留这次组页的全部 61 个不同节点。结果仅描述这次查询的装页数。[配对明细](/mnt/nvme2/merit-io-diagnosis-full/query39930-pages/adjacent-seed-packing.json)

## 写队列、写线程与 SSD

| 要回答的问题 | 已有数据 |
|---|---|
| 写线程从 1 到 2、4，前台等空位是否缩短 | 前 50 万次查询。随线程数增加，排队时间约为 1,525、282、45 µs/查询。可查 QPS、读 I/O、后台写页数和分位数。[线程数对照](/mnt/nvme2/merit-writeworkers-prefix500k/comparison.csv) |
| 1 个写线程，队列扩大 5、10、20 倍 | 前 50 万次查询。阈值为 1,024、5,120、10,240、20,480 页，QPS 依次为 3,507、3,523、3,518、3,531。[队列容量对照](/mnt/nvme2/merit-queue-capacity-prefix500k/comparison.csv) |
| 4 个写线程，队列有上限或无上限 | 前 50 万次查询，各两轮。有限队列 3,831–3,838 QPS，无上限 3,817–3,825 QPS。含积压峰值、内存和排空时间。[队列对照](/mnt/nvme2/merit-unlimited-queue-4w/comparison.csv) |
| 设备延迟、吞吐，与独立压测的关系 | 前缀有 1 写和 4 写的块设备记录。独立 fio 使用同一块盘、4 KiB 混合随机 I/O、73% 读，覆盖不同队列深度，可查带宽和完成延迟。[应用设备计数](/mnt/nvme2/merit-writeworkers-prefix500k/device-check/device-rates.csv) · [fio](/mnt/nvme2/merit-writeworkers-prefix500k/device-check/fio-rates.csv) |

## 此前的缓存与路径实验

| 主题 | 已有配置与数据入口 |
|---|---|
| N-cache 的 LRU、CLOCK、分片 | 同一访问流的[策略回放校验](/mnt/nvme2/spacev500kx10_ncache_replay.GoY3C2/validation.json)、[真实查询 LRU/CLOCK](/mnt/nvme2/spacev500kx10_native_policy.TXsOVh/results.json)、[CLOCK 单分片和 64 分片](/mnt/nvme2/spacev500kx10_clock_shards.lG3b59/summary.json) |
| M-cache 替换、边更新锁、堆分片 | [LRU 的 no_victim 计数](/mnt/nvme2/merit-mcache-replacement.Jcb7dh/lru-verified.csv)、[CLOCK 回放](/mnt/nvme2/merit-mcache-replacement.Jcb7dh/clock-8388608.csv)、[分片锁和记录锁](/mnt/nvme2/merit-medge.F8QxeQ/edge-bench/results.csv)、[4/8/16 个堆分片](/mnt/nvme2/merit-heap-shards.OWYper/results.csv) |
| 跨线程 prefetch | [只探测同伴缓存](/mnt/nvme2/merit-peer-prefetch.T8PZDM/shadow/summary.json)、[跨线程索引](/mnt/nvme2/merit-peer-prefetch.T8PZDM/index/summary.json) |
| D-cache 20% 上限与无上限，反向索引及其余差异 | [全量容量对照](/mnt/nvme2/merit-uncapped-500kx10.1KZK1V/comparison.csv)、[全量逐项恢复](/mnt/nvme2/merit-nc-ablation.JLSP4V/results.csv)、[10k×10 和全量趋势](/mnt/nvme2/merit-nc-10kx10.uRdciG/comparison.csv) |
| 查询 39930 的路径、父子、叶子和组页归属 | [逐节点/逐次访问](/mnt/nvme2/merit-profile-query.w746am/node_matrix.csv)、[父页覆盖](/mnt/nvme2/merit-profile-query.w746am/parent_coverage.csv)、[seed 优先前后](/mnt/nvme2/merit-self-seed.n4MuKV/query_39930_comparison.csv)、[页来源](/mnt/nvme2/merit-self-seed.n4MuKV/page_origins.csv)、[N 命中与 medoid](/mnt/nvme2/merit-self-seed.n4MuKV/query_39930_ncache_medoid.csv)、[父节点查表开关](/mnt/nvme2/merit-39930-parent-lookup.qtYpJv/comparison.csv) |
| PathSeed 组页，私有或共享 prefetch | [10k×10 各方案](/mnt/nvme2/merit-pathseed-10k.bHWGJb/comparison.csv)、[500k×10 各方案](/mnt/nvme2/merit-pathseed-500k.Svv8wP/comparison.csv) |
| 第几跳开始当 seed，以及 N 与 prefetch 重叠 | 10k×10 的[第 1/2 跳](/mnt/nvme2/merit-first-hop-10k.twmplfev/seed_depth_comparison.csv)、[第 2/3 跳交叉复测](/mnt/nvme2/merit-depth-cross-10k.K2blUJ/comparison.csv)、[第 3/4 跳](/mnt/nvme2/merit-fourth-hop-10k.cKJrN8/comparison.csv)、[按深度双命中](/mnt/nvme2/merit-overlap-10k.tAhpD3/overlap_by_depth.csv)。全量的[第 2/3 跳](/mnt/nvme2/merit-third-hop-500k.Ir5hPU/comparison.csv) |
| EBR、锁与 seed 争用 | 10k×10 的[锁/shared_ptr/EBR](/mnt/nvme2/merit-seed-ebr-10k/results.csv)、[不同并发和访问顺序](/mnt/nvme2/merit-seed-contention-10k/summary.csv)。500k×10 的[锁冲突实测](/mnt/nvme2/merit-seed-lock-500k/comparison.csv) |

## 追加记录

每组保留日期、状态、trace 和查询数、代码版本或二进制指纹、变量与对照组、执行命令、磁盘与 TRIM 条件、结果文件和校验状态。重点比较 QPS、延迟、读页数或命中率时一并写明单位和分母。原始命令文件中的环境变量与 trace/binary SHA-256 可用于确认两组能否直接比较。

### 2026-11-26 · 同一查询相邻的两个 seed 共用一张 4 KiB 页

已完成双盘 500k×10，10 查询线程、4 写线程、第 3 跳起建 seed、每页最多 10 个节点、共享 prefetch。按实际展开顺序只看要建页的 seed，两个相邻 seed 都没有页且成员去重后放得下时共页。两个 seed ID 指向同一物理页和缓存项，先补共享页，满后由需要更多槽的 seed 单独开溢出页。两组本轮均未手动 TRIM，复用 stripe4-2 基线，单次比较。运行命令为 `python3 coseed_run.py coseed-notrim`。[代码补丁](/mnt/nvme2/merit-io-diagnosis-full/coseed-build/coseed.patch) · [运行配置、二进制和 trace 指纹](/mnt/nvme2/merit-io-diagnosis-full/coseed-notrim/command.json)

| 指标 | 原版 | 相邻 seed 合页 |
|---|---:|---:|
| QPS | 8,918.15 | 12,834.24 |
| 平均 / P99 延迟（µs） | 1,090.64 / 4,473 | 748.81 / 1,788 |
| prefetch 命中 / 全部节点访问 | 28.3645% | 31.2620% |
| N-cache 命中 / 全部节点访问 | 33.8227% | 34.1316% |
| 前台每查询读取 D 页 / 原图页 | 22.6136 / 4.3228 | 17.7488 / 4.3162 |
| D-cache 最终物理页 / 后台写页次数 | 9,269,112 / 12,188,429 | 5,714,754 / 8,436,179 |
| 补页时读旧页次数 | 1,148,297 | 1,217,830 |

全量两组都有 378,754,240 次节点访问，500 万组 Top-10 ID 和距离文件 SHA-256 一致。新方案的 3,828,238 个物理页各有两个 seed，全量写完后 5,714,754 个物理页均通过排空校验。第 10 段 50 万查询中两组后台写页次数均为 0，原版与合页每查询分别读 D 页 24.19 和 18.86。首次冷访问 39930，经实际文件逐槽比对，27 个 seed 从占 27 页降到 14 页，均覆盖 61 个不同节点。在完整 trace 中，39930 的第 10 次访问均访问 82 个节点，N 均命中 19 个，prefetch 从 31 升至 35，D 页读取从 30 降至 20。[完整指标](/mnt/nvme2/merit-io-diagnosis-full/coseed-notrim/comparison.csv) · [分阶段结果](/mnt/nvme2/merit-io-diagnosis-full/coseed-notrim/phases.csv) · [39930 十次访问](/mnt/nvme2/merit-io-diagnosis-full/coseed-notrim/39930.csv) · [39930 实际页内容](/mnt/nvme2/merit-io-diagnosis-full/coseed-39930/coverage.json)

双盘自动 TRIM 合页复测已完成。500k×10，10 个查询线程、4 个写线程，运行命令为 `python3 coseed_run.py coseed-bothtrim1`。本轮 nvme0 返回 0 B、nvme2 返回 47.3 GiB，两者退出码均为 0。复用同布局且两盘均 TRIM 的 stripe4-bothtrim1 原版基线。QPS 从 11,320.61 升到 12,744.07（+12.57%），平均延迟从 852.57 降到 753.92 µs，P99 从 2,268 降到 1,803 µs。Prefetch 命中占全部访问的比例从 28.3645% 升到 31.2619%，每查询读取 D 页从 22.6136 降到 17.7488。查询 39930 第 10 次访问，prefetch 从 31/82 升至 35/82，D 页从 30 降至 20。500 万组 Top-10 ID、距离的 SHA-256 通过，5,714,754 个物理页写队列排空校验通过。[配置与指纹](/mnt/nvme2/merit-io-diagnosis-full/coseed-bothtrim1/command.json) · [TRIM 回执](/mnt/nvme2/merit-io-diagnosis-full/coseed-bothtrim1/trim.log) · [同条件对照](/mnt/nvme2/merit-io-diagnosis-full/coseed-bothtrim1/comparison.csv)

### 2026-11-26 · 最多三个相邻新 seed 共用一页

已完成 500k×10，复用双 seed 自动 TRIM 组 coseed-bothtrim1。新组按相同 seed 顺序尝试最多三个，去重后超过 10 个节点或第三个 seed 已有页时回退到两个；前两个不能共页时从第一个单独组页。4 KiB 页、双盘均分、10 个查询线程、4 个写线程。运行命令为 `python3 triseed_run.py triseed-bothtrim1`。39930 单查询验证前两盘各 TRIM 11.2 GiB，全量前再次执行，两盘均返回 0 B、退出码 0。[实现差异](/mnt/nvme2/merit-io-diagnosis-full/triseed-build/triseed.patch) · [配置与指纹](/mnt/nvme2/merit-io-diagnosis-full/triseed-bothtrim1/command.json) · [全量 TRIM 回执](/mnt/nvme2/merit-io-diagnosis-full/triseed-bothtrim1/trim.log)

相对于双 seed，本轮 QPS 从 12,744.07 到 13,281.24（+4.22%），平均/P99 延迟从 753.92/1,803 到 722.48/1,788 µs。prefetch 命中占全部节点访问的比例从 31.2619% 到 32.3236%，N 命中从 34.1316% 到 34.2672%。每查询读 D 页从 17.7488 到 16.2167，读原图从 4.3162 到 4.3134。后台写页次数从 8,436,037 到 7,414,370，补页时读旧页次数从 1,217,837 到 1,274,099。最终 4,936,086 页，其中 1,967,564 张有三个 seed，填入 16,185,518 个节点槽。500 万组 Top-10 ID、距离校验一致，写队列已排空。39930 冷访问，27 个 seed 从 14 页降至 10 页（7 页三个，3 页两个），逐槽核对全覆盖原 61 个节点；第 10 次访问 N 命中均为 19，prefetch 从 35/82 到 39/82，读 D 页从 20 到 15。[全量对照](/mnt/nvme2/merit-io-diagnosis-full/triseed-bothtrim1/comparison.csv) · [39930 页内容](/mnt/nvme2/merit-io-diagnosis-full/triseed-39930/coverage.json) · [逐次访问](/mnt/nvme2/merit-io-diagnosis-full/triseed-bothtrim1/39930.csv)

2026-11-26，DiskANN 和 PipeANN+NC 三 seed 功能验证已完成，各次磁盘实测预先对参与的 SSD 自动 TRIM。DiskANN 查询 39930 的 Top-10 ID、距离、两份物理 D 页文件与三 seed 实验版逐字节一致。PipeANN 使用 SPACEV 128 条不同查询重复 3 次（384 次），L=32、2 查询线程，带缓存时开启原索引逐记录核验。beam=8 的新模式共写出 1,103 张三 seed 页，实际读取 2,884 张 D 页，5,362 次 prefetch 命中。一次样本中不启用 NC、旧 NC、新 NC 的 Recall@10 分别为 90.68%、89.22%、89.56%，旧 NC 与新 NC 有 341/384 条 Top-10 列表完全一致。[验证结果](/mnt/nvme2/merit-pathseed-port-validation/recall.json) · [原始日志](/mnt/nvme2/merit-pathseed-port-validation/beam8on/run.log)

## 全仓 NC 清理验证

DiskANN 查询 39930 的 Top-10 ID、距离及两个 D 文件与三 seed 实验版逐字节一致。PipeANN 使用 128 条不同查询重复 3 次，L=32、beam=8、两查询线程，逐条读取原图核对缓存返回的数据。最终构建的 Recall@10 为 89.48%，与清理前三 seed 的上一轮相比有 349/384 条 Top-10 列表完全一致。每轮先对 nvme0 和 nvme2 自动 TRIM，两盘回执均为 exit=0。原始记录在 `/mnt/nvme2/merit-pathseed-cleanup-validation`，最终构建的[核验结果](/mnt/nvme2/merit-pathseed-cleanup-validation/release/result.json)和[读回日志](/mnt/nvme2/merit-pathseed-cleanup-validation/release/beam8on/run.log)单独保存。
