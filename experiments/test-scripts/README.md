# Benchmark and test scripts

Repeatable **SIFT1M / SIFT100M / MERIT** benchmarks live here.  
Full Run1/Run2/Run3 relayout pipelines are under [`../relayout/`](../relayout/).

## Environment variables

| Variable | Default |
|----------|---------|
| `REPO_ROOT` | Repo root (auto) |
| `DISKANN_BUILD` | `$REPO_ROOT/diskann/build` |
| `DATA_DIR` | `$REPO_ROOT/data/sift1m` (or `sift100m` in 100M scripts) |
| `THREADS` | `16` |
| `L` / `K` / `W` | `100` / `10` / `2` |

## MERIT

| Script | Description |
|--------|-------------|
| `run_merit_five_way.sh` | Core five-way: baseline / BFS / mem / disk / mem+disk |
| `run_sift1m_merit_five_way.sh` | SIFT1M wrapper for five-way |
| `run_sift1m_merit_same_trace.sh` | Profile then search on the same query trace |
| `run_sift1m_merit_combined.sh` | baseline / disk-only / memory-only / combined |
| `run_sift1m_merit_eviction.sh` | Disjoint vs overlapping tiers; batch eviction |
| `run_sift1m_merit_full_verify.sh` | Feature check + summary table |
| `run_sift1m_merit_vs_diskann_cache.sh` | ~0.01 GB: baseline vs BFS vs MERIT pool |
| `run_sift1m_merit_mem001_disk01.sh` | Mem 0.01 GB + disk ratio 0.1 |
| `run_sift1m_layout_ab_disk50.sh` | Layout A (kh0) vs B (kh2) at 50% disk |
| `run_sift100m_merit_background.sh` | 100M five-way (mem 2 GB, disk 0.1) |
| `build_sift100m_index_and_profile.sh` | Build uint8 index + same-trace profile |

```bash
bash experiments/test-scripts/run_sift1m_merit_five_way.sh
bash experiments/test-scripts/run_merit_five_way.sh
```

## Relayout / IO comparisons

| Script | Description |
|--------|-------------|
| `run_sift1m_cold_compare.sh` | Drop page cache each run; baseline / edge / node |
| `run_sift1m_sector_cache_compare.sh` | Cold-ish + per-query sector cache |
| `run_sift1m_k5_compare.sh` | Offline relayout with k_hops=5 |

## Prerequisites

- Built `search_disk_index` (and `build_disk_index` for index construction)
- Dataset under `data/` (not shipped in this repo)
- Optional: `scripts/drop_system_caches.sh` for cold runs (sudo)
