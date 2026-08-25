# MERIT

**MERIT** (profile-guided **M**emory + disk cach**E** for **R**epeated **I**ndex **T**raversal) extends [DiskANN](https://github.com/microsoft/DiskANN) disk search with:

1. **Same-trace access profiling** (`--enable_access_profile`) — node expand and directed-edge counts on a fixed query workload.
2. **MERIT memory pool** (`--merit_memory_gb`) — dynamic DRAM cache sized by budget, filled from profile hotness (not DiskANN static BFS `_nhood_cache` alone).
3. **MERIT disk cache** (`--merit_disk_cache_ratio`, `--merit_disk_cache_k_hops`, `--merit_disk_cache_layout`) — extra on-disk layout for hot nodes:
   - **Layout A**: `k_hops=0`, flat top-N by `node_expand`.
   - **Layout B** (`node`): same (a)–(e) page packing as full relayout, but seed = hottest **node** and path avg uses **node_expand**; capped at disk-cache ratio (~10%).
   - **Layout C** (`edge`): same Jiang edge (a)–(e) flow as offline relayout; capped at ~10% in the disk cache.
   - **Layout D** (`frontier`, alias `d`): hop-frontier template co-location from profile.
   - **Layout E** (`directed_beam`, alias `e`): **parent outgoing-heat** seed (profile parent rank) + optional top directed children (`k_hops` = beam width) + **undirected profile star** page fill; default disk cache layout (`k_hops=1` recommended).
   - **Layout P** (`parent`): parent seed + all directed profile children per page (no star fill).
   - **Seed replica** (`directed_seed_replica_pct100`, aliases `seed_replica_pct100` / `seed_pct100`): parent-seed extents with replica packing for coverage; reuse with `--merit_disk_cache_reuse_prefix`. Variants `pct30` / `pct50` / `pct80` / `pct90` / `pct100` control replica threshold.
4. **Per-query sector cache** (`--enable_query_sector_cache`) — avoids re-reading the same sector within a query (used with disk cache).
5. **Seed access policy** (env, with seed-replica layout + query sector cache):
   - **Full Extent** (default): unset `MERIT_RECORD_DRIVEN_SEED_ACCESS` — always fetch the parent seed extent when expanding children from that seed.
   - **Adaptive**: `MERIT_RECORD_DRIVEN_SEED_ACCESS=1` and `MERIT_ADAPTIVE_PARENT_OR_SELF=1` — leaf layer always uses parent extent; intermediate nodes switch to parent only when it needs fewer physical reads than self-seed access; skip I/O when the record is already in the query sector cache.

Offline **k-hop relayout** tools (`relayout_disk_index`, `apply_disk_permutation`) are included for index rewrite experiments; runtime MERIT disk cache uses a separate disk cache file without rewriting the base index.

This tree is a **DiskANN fork** with MERIT integrated in `diskann/` plus reproducible scripts under `experiments/`.

## Build

```bash
cd diskann
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j --target search_disk_index build_disk_index relayout_disk_index apply_disk_permutation
```

Dependencies match upstream DiskANN (C++17, Boost, OpenMP, aio on Linux).

## Quick start (SIFT1M)

1. Place SIFT1M under `data/sift1m/` (`sift_base.fbin`, query, ground truth, built `sift1m_index_*`).
2. Record a profile on the same queries you will benchmark:

```bash
./diskann/build/apps/search_disk_index \
  --data_type float --dist_fn l2 \
  --index_path_prefix data/sift1m/sift1m_index \
  --query_file data/sift1m/sift_query.fbin \
  --gt_file data/sift1m/sift_groundtruth.bin \
  --result_path data/sift1m/profile_results \
  --recall_at 10 --search_list 100 --beamwidth 2 --num_threads 16 \
  --num_nodes_to_cache 0 \
  --enable_access_profile \
  --access_profile_prefix data/sift1m/run2_profile_same_trace
```

3. Run the five-way comparison:

```bash
bash experiments/test-scripts/run_sift1m_merit_five_way.sh
```

See `experiments/test-scripts/README.md` for more scripts (eviction, Layout A/B, 100M pipeline).

## Key CLI flags

| Flag / env | Role |
|------------|------|
| `--enable_access_profile` / `--access_profile_prefix` | Run2 profiling |
| `--merit_profile_prefix` | Profile prefix for MERIT planning (required with memory/disk tiers) |
| `--merit_memory_gb` | MERIT DRAM pool budget |
| `--merit_disk_cache_ratio` | Fraction of index nodes for disk cache |
| `--merit_disk_cache_k_hops` | `0` = flat layout; `2` = k-hop packing |
| `--merit_disk_cache_layout` | Packing policy (`directed_beam`, `directed_seed_replica_pct100`, …) |
| `--merit_disk_cache_reuse_prefix` | Load an existing disk-cache file instead of rebuilding |
| `--merit_disk_cache_exclude_memory` | When true, disk tier skips nodes already in memory pool |
| `--enable_query_sector_cache` | Intra-query sector reuse |
| `MERIT_RECORD_DRIVEN_SEED_ACCESS=1` | Record-driven seed routing (required for Adaptive) |
| `MERIT_ADAPTIVE_PARENT_OR_SELF=1` | Adaptive parent-or-self policy (with record-driven) |

Example Adaptive search (reuse a built seed-replica cache):

```bash
export MERIT_RECORD_DRIVEN_SEED_ACCESS=1
export MERIT_ADAPTIVE_PARENT_OR_SELF=1
./diskann/build/apps/search_disk_index \
  --data_type float --dist_fn l2 \
  --index_path_prefix data/sift1m/sift1m_index \
  --query_file data/sift1m/sift_query.fbin \
  --gt_file data/sift1m/sift_groundtruth.bin \
  --result_path /tmp/adaptive_out \
  --recall_at 1 --search_list 50 --beamwidth 4 --num_threads 16 \
  --num_nodes_to_cache 0 \
  --merit_profile_prefix data/sift1m/workloads/profiles/uniform_10k \
  --merit_disk_cache_ratio 0.906 \
  --merit_disk_cache_layout directed_seed_replica_pct100 \
  --merit_disk_cache_reuse_prefix data/sift1m/runs/seed_replica_fullcover_workloads/uniform/seed_replica \
  --enable_query_sector_cache
```

Unset both env vars for Full Extent on the same CLI.

Use **`uint8`** / **`float`** data types; MERIT flags are wired for unfiltered search on all supported scalar types.

## Repository layout

```
diskann/          DiskANN + MERIT implementation
experiments/      Benchmarks, relayout pipelines, analysis
scripts/          Helpers (e.g. drop page cache for cold runs)
data/             Local datasets and indices (gitignored)
```