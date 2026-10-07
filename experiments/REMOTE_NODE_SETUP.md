# 在另一台机器（10.64.1.42）上运行 MERIT 实验指南

目标机器示例：`jianz@10.64.1.42`  
数据与代码根目录：`/home/jianz/workload`  
本机（源）搬迁脚本：`experiments/scripts/migrate_essential_to_42.sh`

---

## 0. 搬完后远端目录应长什么样

```text
/home/jianz/workload/
  code/Merit/                 # 代码（无 data 实体、无 diskann/build）
  code/Merit/data -> ../../data
  data/
    sift1m/                   # 索引 + workloads + profiles（无 runs/）
    sift10m/
    sift100m/
```

粗体积（无 `runs/`）：1M ≈ 7GB，10M ≈ 37GB，100M ≈ 129GB，合计约 **173GB**（400GB 盘够用）。

---

## 1. 从本机完成搬迁（在 a15 / 源机器上做）

### 1.1 建议先做 SSH 免密（避免每步输密码、连接被掐）

```bash
ssh-copy-id jianz@10.64.1.42
ssh jianz@10.64.1.42 'hostname; df -h /home/jianz/workload'
```

### 1.2 正式搬迁

```bash
# 三个数据集都搬
/home/jianz/Merit/experiments/scripts/migrate_essential_to_42.sh

# 或只要 1M（先验证环境）
DATASETS="sift1m" /home/jianz/Merit/experiments/scripts/migrate_essential_to_42.sh
```

脚本会：同步 `code/Merit`、同步各 `data/<ds>`（排除 `runs/`）、并创建 `code/Merit/data -> ../../data`。

中断后可再跑同一命令，rsync 会续传/跳过已有文件。

### 1.3 在远端确认

```bash
ssh jianz@10.64.1.42
df -h /home/jianz/workload
du -sh /home/jianz/workload/code /home/jianz/workload/data/*
ls -ld /home/jianz/workload/code/Merit/data
ls /home/jianz/workload/data/sift1m/sift1m_index_disk.index
ls /home/jianz/workload/data/sift1m/workloads/uniform_10k.fbin
```

---

## 2. 路径对齐（重要）

很多脚本写死了：

- `SEARCH=/home/jianz/Merit/diskann/build/apps/search_disk_index`
- `DATA=/home/jianz/Merit/data/sift1m`

远端实际代码在 `/home/jianz/workload/code/Merit`。最省事的做法是在 **42 上**建软链：

```bash
ln -sfn /home/jianz/workload/code/Merit /home/jianz/Merit
ls -ld /home/jianz/Merit /home/jianz/Merit/data
# 期望：
#   /home/jianz/Merit -> .../workload/code/Merit
#   /home/jianz/Merit/data -> ../../data  （指向 /home/jianz/workload/data）
```

之后下面所有命令都可以沿用本机路径习惯。

---

## 3. 安装编译依赖（Ubuntu/Debian 示例）

在 **42** 上：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake g++ git \
  libaio-dev libgoogle-perftools-dev libmkl-full-dev \
  libboost-dev libboost-program-options-dev \
  python3 python3-pip python3-venv

# 若没有 MKL 包，可先试不加 MKL（视你们 CMake 配置而定）；
# OpenMP 一般随 g++ 提供。
```

Python 画图/脚本：

```bash
pip3 install --user matplotlib numpy
# 或 conda：conda install matplotlib numpy
```

---

## 4. 编译 DiskANN / MERIT 搜索程序

```bash
cd /home/jianz/Merit/diskann
mkdir -p build && cd build
cmake ..
make -j$(nproc) search_disk_index
```

成功标志：

```bash
ls -lh /home/jianz/Merit/diskann/build/apps/search_disk_index
```

若 cmake 报缺库，把报错里的包名补装后再 `cmake .. && make -j$(nproc) search_disk_index`。

可选：把常用环境写进 `~/.bashrc`：

```bash
export OMP_NUM_THREADS=8
export MERIT_USE_RAMFS=0
# 若用 conda 的 lib：
# export LD_LIBRARY_PATH="$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"
```

```bash
source ~/.bashrc
```

---

## 5. 冒烟测试：SIFT1M disk-only 搜索

```bash
export LD_LIBRARY_PATH="${HOME}/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8
export MERIT_USE_RAMFS=0

SEARCH=/home/jianz/Merit/diskann/build/apps/search_disk_index
DATA=/home/jianz/Merit/data/sift1m

"$SEARCH" --data_type float --dist_fn l2 \
  --index_path_prefix "$DATA/sift1m_index" \
  --query_file "$DATA/workloads/uniform_10k.fbin" \
  --gt_file "$DATA/workloads/uniform_10k_gt.bin" \
  --result_path /tmp/merit_smoke_run \
  --recall_at 1 --search_list 50 --beamwidth 4 --num_threads 8 \
  --num_nodes_to_cache 0
```

能打印 QPS / Recall 即基本 OK。

带 access profile（会写 visit / out-heat 等）：

```bash
PROF=/home/jianz/Merit/data/sift1m/workloads/profiles/smoke_profile
"$SEARCH" --data_type float --dist_fn l2 \
  --index_path_prefix "$DATA/sift1m_index" \
  --query_file "$DATA/workloads/uniform_10k.fbin" \
  --gt_file "$DATA/workloads/uniform_10k_gt.bin" \
  --result_path /tmp/merit_smoke_prof \
  --recall_at 1 --search_list 50 --beamwidth 4 --num_threads 8 \
  --num_nodes_to_cache 0 \
  --enable_access_profile \
  --access_profile_prefix "$PROF"
ls -lh ${PROF}_node_expand.bin ${PROF}_node_visit.bin ${PROF}_node_out_heat.bin 2>/dev/null
```

（若 profile 是旧的、没有 visit/out_heat，重跑带 `--enable_access_profile` 的新二进制才会生成新文件。）

---

## 6. 跑 seed / nbrTop sweep（示例）

结果会写到 `data/.../runs/...`（搬迁时没带旧 runs，这里会新建，注意磁盘空间）。

### 6.1 SIFT1M disk-only Top5–90 扩展

```bash
cd /home/jianz/Merit
python3 -u experiments/scripts/sift1m_disk_seed_nbrtop_extend.py \
  2>&1 | tee /tmp/sift1m_disk_seed_nbrtop_extend.log
```

图输出大致在：

- `data/sift1m/runs/adaptive_reduction_sweep/seed_nbrtop_disk_reads.png`
- `data/sift1m/runs/adaptive_reduction_sweep/seed_nbrtop_vs_base.png`

### 6.2 SIFT1M mem 0.01GB（noexclude）

```bash
python3 -u experiments/scripts/sift1m_mem001_noexclude_seed_nbrtop_extend.py \
  2>&1 | tee /tmp/sift1m_mem001_extend.log
```

### 6.3 SIFT10M（需已搬 10M 数据）

```bash
python3 -u experiments/scripts/sift10m_uniform100k_seed_nbrtop_sweep.py \
  2>&1 | tee /tmp/sift10m_seed_nbrtop.log
```

10M 查询是 uint8 + `uniform_100k`，脚本内路径已写死；确认：

```bash
ls /home/jianz/Merit/data/sift10m/sift10m_index_disk.index
ls /home/jianz/Merit/data/sift10m/workloads/uniform_100k.u8bin
ls /home/jianz/Merit/data/sift10m/workloads/profiles/uniform_100k_node_expand.bin
```

### 6.4 SIFT100M

```bash
python3 -u experiments/scripts/sift100m_uniform10k_seed_nbrtop_sweep.py \
  2>&1 | tee /tmp/sift100m_seed_nbrtop.log
```

100M 更慢、更吃盘；跑前再看 `df -h`。

---

## 7. 常用环境变量（与本机一致）

多数 sweep / 搜索会用到：

```bash
export OMP_NUM_THREADS=8
export MERIT_USE_RAMFS=0
export MERIT_RECORD_DRIVEN_SEED_ACCESS=1
export MERIT_ADAPTIVE_PARENT_OR_SELF=1
unset MERIT_ADAPTIVE_EXTENT
unset MERIT_DISABLE_MULTIREAD
unset MERIT_PARENT_SINGLE_PAGE
unset MERIT_CACHE_REPLICA_FALLBACK
```

Python 脚本里通常已设置一部分；若手跑 `search_disk_index`，请自己 export。

---

## 8. 磁盘与数据注意点

1. **不要**把源机的 `runs/`（数百 GB）拷过来；需要结果图可单独 scp 几个 png/json。  
2. 新跑 sweep 会在远端生成新的 `runs/`，占空间；定期 `du -sh data/*/runs` 并清理不需要的 case。  
3. 测 IO 时尽量让 `data/` 落在你要测的那块盘上（确认 `df` 挂载点）。  
4. Profile 策略（现状约定）：  
   - **Seed Top%**：`out_heat`（孩子从该点发现后被 expand 则 +1）  
   - **nbrTop%**：优先 `node_visit`（有 `_node_visit.bin` 时）  
   - 旧 profile 没有 visit/out_heat 时，脚本会回退到 expand / 边汇总。

---

## 9. 常见问题

| 现象 | 处理 |
|------|------|
| rsync 要密码后 `Connection closed` / code 255 | `ssh-copy-id` 免密后重跑脚本 |
| `No such file: .../search_disk_index` | 先完成第 4 节编译；检查 `/home/jianz/Merit` 软链 |
| `libxxx.so` 找不到 | 设 `LD_LIBRARY_PATH`（conda/MKL） |
| cmake 找不到 Boost/AIO/MKL | `apt install` 对应 `-dev` 包 |
| 脚本找不到 data | 检查 `Merit/data` 软链是否指向 `/home/jianz/workload/data` |
| 盘满 | `df -h`；删远端 `runs/` 下旧 case；先只跑 1M |
| 10M/100M recall/类型不对 | 10M 用 uint8 + u8bin；1M 用 float + fbin |

---

## 10. 建议上手顺序

1. 搬迁 + `/home/jianz/Merit` 软链  
2. 装依赖 + 编译 `search_disk_index`  
3. SIFT1M smoke 搜索  
4. 跑 `sift1m_disk_seed_nbrtop_extend.py`  
5. 再按需搬/跑 10M、100M  

---

## 11. 从本机只补拷几张图（可选）

```bash
scp /home/jianz/Merit/data/sift1m/runs/adaptive_reduction_sweep/seed_nbrtop_*.png \
  jianz@10.64.1.42:/home/jianz/workload/data/sift1m/runs/adaptive_reduction_sweep/
# 远端需先 mkdir -p 对应目录
```

---

文档路径（源仓库）：`experiments/REMOTE_NODE_SETUP.md`  
搬迁后远端也可直接看：`/home/jianz/workload/code/Merit/experiments/REMOTE_NODE_SETUP.md`
