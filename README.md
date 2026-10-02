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

## Online adaptive three-cache mode

The current runtime can learn directly from the live query stream; it does not need to know whether the workload is uniform or contains moving hotspots.

- **N-cache** stores complete nodes in DRAM. It uses 128 lock shards by default, an atomic membership bitmap for lock-free misses, a reader-writer lock, and CLOCK replacement. Adaptive mode samples nodes by ID (1/64 by default) and continuously simulates three independent shadow caches: normal MRU admission, Hop-12 reject-first, and Hop-8 reject-first. The shadow caches never serve queries. Every 25K queries, MERIT estimates each policy's cost from its predicted misses and admissions, the measured device I/O time, and sampled real admission time. It switches immediately when the predicted gain is at least 2%, or after the same policy wins three consecutive windows by at least 0.5%. Evaluation starts only after the active shadow cache is full, avoiding cold-start decisions. No workload label or hit-rate range is used.
- **M-cache** stores node and edge hotness metadata. Every 1,000 queries it measures the current-window hit rate. Each run of 25 consecutive windows below 20% reduces updates one step from 100% to 10%, then from 10% to 1%; six windows above 40% restore one level at a time.
- **D-cache** stores runtime-generated 4 KiB pages. Page membership is learned from nodes actually served by base storage or D-cache in the same query, rather than graph neighbors merely discovered during traversal. One in every 64 queries records a four-node neighborhood in the real storage-access order; each metadata entry retains at most 64 partners (`MERIT_REAL_IO_MAX_PARTNERS`). With deferred writes enabled, an optional bounded write-back buffer retains complete payloads only for nodes already selected into pending seed pages. D-cache maintenance is controlled by net physical-I/O benefit, not hit rate alone. MERIT counts base-index pages avoided and D-cache pages read. Twenty-five consecutive 1,000-query windows saving less than 0.1 page/query disable M/D-cache maintenance. While disabled, 1% of queries remain full probes. Two consecutive 10,000-query probe windows with at least 0.25 net page saved/query and at least 1.25 avoided pages per D-cache page read re-enable maintenance.

The D-cache net-benefit gate is enabled automatically when `MERIT_MCACHE_ADAPTIVE_UPDATE=1`; set `MERIT_DCACHE_NET_GATE=0` to disable it.

All reported dynamic D-cache benchmarks must use a full-cache warmup. Set
`MERIT_REQUIRE_FULL_DCACHE_WARMUP=1`; the search application repeatedly runs
the supplied query workload until the configured D-cache reaches 100%
capacity, then starts the timed query pass with fresh per-query statistics.
Thus a reported 2M-query experiment means 2M measured queries after the
D-cache is full; warmup queries are not included in QPS, latency, hit-rate, or
I/O results. If the cache cannot fill within
`MERIT_FULL_WARMUP_MAX_QUERIES` (default 100M), the run fails instead of
publishing a cold-cache result.

Large D-caches may not fill from a stable skewed stream. For a reproducible
full initial state, use `--merit_disk_cache_layout bfs` together with
`MERIT_DYNAMIC_PREFILL_STATIC=1`. MERIT first packs the BFS-selected nodes
into full 10-node pages, imports those pages into the dynamic D-cache, registers
one seed per page in M-cache, and only then enables normal dynamic replacement.
`MERIT_NCACHE_CAP_NODES` and `MERIT_DCACHE_NODE_CAP` provide exact node caps
for percentage-based experiments; `MERIT_DCACHE_CAP` remains the exact dynamic
page cap.

Set `MERIT_NCACHE_GHOST_STATS=1` only for diagnostics to attribute later N-cache hits to second-encounter admissions. It is disabled by default because exact per-hit accounting adds atomic operations to the search path.

The N-cache shadow controller is enabled automatically by `MERIT_NCACHE_ADMISSION=adaptive_hop_reject`. `MERIT_NCACHE_SHADOW_SAMPLE_SHIFT=6` selects 1/64 node sampling, `MERIT_NCACHE_SHADOW_MIN_GAIN=0.02` sets the immediate-switch margin, and `MERIT_NCACHE_SHADOW_CONTROL=0` disables shadow-based switching.

Example:

```bash
export MERIT_DYNAMIC_3CACHE=1
export MERIT_DEFERRED_NCACHE_PAGE_WRITE=1
export MERIT_NCACHE_SHARDS=128
export MERIT_NCACHE_FAST_MISS=1
export MERIT_NCACHE_CLOCK=1
export MERIT_NCACHE_SPIN=0
export MERIT_NCACHE_ADMISSION=adaptive_hop_reject
export MERIT_NCACHE_HOP_THRESHOLD=12
export MERIT_NCACHE_ADAPT_MIN_HOP=8
export MERIT_NCACHE_ADAPT_HALF_LIFE_QUERIES=25000
export MERIT_NCACHE_GHOST_PERCENT=10
export MERIT_MCACHE_ADAPTIVE_UPDATE=1

./diskann/build/apps/search_disk_index \
  --data_type uint8 --dist_fn l2 \
  --index_path_prefix /path/to/index \
  --query_file /path/to/queries.u8bin \
  --gt_file /path/to/ground_truth.bin \
  --result_path /path/to/results \
  --recall_at 1 --search_list 100 --beamwidth 4 --num_threads 24 \
  --num_nodes_to_cache 0 \
  --merit_memory_gb 2 --merit_memory_runtime_admit true
```

### SIFT100M moving-hotspot result

Reproduced on SIFT100M with 2 million queries: ten cycles of 100K global-uniform queries followed by 100K Zipfian queries (`alpha=1.2`). Every Zipfian segment uses a different, non-overlapping 100K-node region. Configuration: 24 threads, `L=100`, beam width 4, 2 GiB N-cache, mdadm RAID5.

- **Plain full MERIT:** 6,257.90 QPS, 50.90 reads/query, P99 8,580 us.
- **DiskANN BFS:** 7,682.13 QPS, 99.23 reads/query, P99 5,289 us.
- **Adaptive N-cache only:** 11,256.89 QPS, 51.05 reads/query, P99 6,637 us.
- **General shadow-adaptive full MERIT:** 11,261.18 QPS, 51.20 reads/query, P99 5,851 us.
- Recall@1 is 99.99% for all four runs. The general controller changes policy 25K queries after each uniform/hotspot boundary, with no workload-specific rule.

With `MERIT_NCACHE_GHOST_STATS=1`, 979,234 nodes were admitted after a second encounter. Of these, 598,719 (61.14%) produced a later cache hit, contributing 70,637,091 of 133,952,593 total N-cache hits (52.73%). A same-build back-to-back comparison of 128 versus 32 shards measured 11,235.81 versus 11,124.77 QPS, 296.54 versus 314.00 CPU us/query, and 5,828 versus 6,644 us P99.

### SIFT100M moderate-skew result

The moderate workload contains rounds of one million queries. Each round samples a different one-million-node global support with finite Zipf `alpha=0.8`; about 391K nodes occur per round, the ten hottest nodes account for 4.75% of queries, and the hottest 1% account for 36.8%.

- **DiskANN BFS:** 7,143.60 QPS, 99.03 reads/query, P99 5,395 us.
- **Plain MRU MERIT, first 1M round:** 9,704 QPS.
- **General shadow-adaptive MERIT, same 1M round:** 9,971 QPS, 59.68 reads/query, P99 5,819 us.
- The controller selects Hop-12 at 225K queries and Hop-8 at 775K from measured costs. Recall@1 is 99.97%.

A full-capacity 5M-query run used exactly 1M N-cache nodes and 20M
D-cache nodes (2M full BFS-prefilled pages). Original dynamic MERIT achieved
4,558 QPS and 75.81 reads/query. N-cache removed 39.60 read candidates/query
(33.77% of estimated no-cache demand); D-cache avoided 16.99 base reads while
issuing 15.11 D-cache reads, for a net 1.87 reads/query (1.60%). Requiring the
4 MiB pending buffer to preserve 10/10-node pages increased occupancy from
8.44 to 10 nodes/page, but fell to 4,085 QPS and produced effectively zero
net D-cache savings. Full pages alone are therefore not a sufficient write
criterion; replacement locality must also be preserved.

Repeating the same full-warmup run without the pending buffer and halving
N-cache to 500K nodes raised D-cache's gross hit rate from 21.87% to 25.98%
and its net contribution from 1.60% to 3.99%. However, N-cache contribution
fell from 33.77% to 27.97%; D-cache recovered only 41.2% of the lost N-cache
savings. Overall QPS fell from 4,558 to 4,225 and reads/query rose from 75.81
to 79.77.

An experimental exclusive-tier run set
`MERIT_DCACHE_EVICT_NCACHE_ON_COMMIT=1`, removing every newly committed
D-cache page member from N-cache. It performed 1,077,050 removals, but
D-cache hits stayed flat (17.13 to 17.10/query), net D-cache savings fell from
1.87 to 1.73 reads/query, and QPS fell from 4,558 to 4,438. This workload
therefore shows little useful N/D masking: most removed overlaps were not
reused enough to benefit from forced D-cache routing.

On the stronger moving Zipfian workload (`alpha=1.2`, top-ten share about
48%), N-cache supplied 95.76% of estimated no-cache demand. Exclusive-tier
mode performed 247,335 N-cache removals: D-cache hits rose only from 1.135 to
1.186/query and net D-cache savings from 0.001 to 0.009/query, while physical
reads rose from 5.01 to 5.05/query and QPS fell from 13,806 to 13,474. Strong
skew makes N-cache more important, not forced N/D exclusivity more profitable.

Replacing graph-discovery edges with sampled real storage co-access improved
the 5M-query moderate-skew run from 4,558 to 4,930 QPS. D-cache net savings
rose from 1.87 to 2.39 reads/query, useful nodes per D-cache read rose from
1.13 to 1.17, CPU time fell from 904 to 591 us/query, and metadata edges fell
from 50.4M to 2.27M. Recall@1 remained 99.97%.

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
| `MERIT_DYNAMIC_3CACHE=1` | Enable online N/M/D three-cache mode |
| `MERIT_NCACHE_ADMISSION=adaptive_hop_reject` | Enable adaptive normal/Hop-12/Hop-8 admission |
| `MERIT_NCACHE_FAST_MISS=1` / `MERIT_NCACHE_CLOCK=1` | Enable lock-free miss checks and CLOCK replacement |
| `MERIT_MCACHE_ADAPTIVE_UPDATE=1` | Enable adaptive M-cache update frequency and D-cache net-I/O gate |
| `MERIT_DCACHE_NET_GATE=0` | Explicitly disable the net-I/O gate |
| `MERIT_DCACHE_PROBE_PERIOD` | Probe period while D-cache is disabled; default `100` |
| `MERIT_PENDING_BUFFER_BYTES` | Pending seed-page payload buffer; default 4 MiB with deferred writes, `0` restores immediate first-eviction writes |
| `MERIT_PENDING_BUFFER_READY_PERCENT` | Optional early-write fullness threshold; default `0` (pressure-only) |
| `MERIT_PENDING_REQUIRE_FULL_PAGE` | Require every dynamic page to fill all physical slots before writing; default `1` with the pending buffer |
| `MERIT_REQUIRE_FULL_DCACHE_WARMUP=1` | Require D-cache to reach 100% before timed queries |
| `MERIT_FULL_WARMUP_MAX_QUERIES` | Fail if full warmup exceeds this many queries; default 100M |
| `MERIT_DYNAMIC_PREFILL_STATIC=1` | Import a full static BFS layout into the dynamic D-cache before measurement |
| `MERIT_REAL_IO_COACCESS` | Build dynamic pages from real per-query storage access; default `1` |
| `MERIT_REAL_IO_WINDOW` / `MERIT_REAL_IO_SAMPLE_QUERIES` | Access-order neighborhood (default `4`) and query sampling denominator (default `64`) |
| `MERIT_REAL_IO_MAX_PARTNERS` | Maximum retained real-I/O partners per metadata node; default `32` |
| `MERIT_DCACHE_EVICT_NCACHE_ON_COMMIT=1` | Experimental exclusive-tier mode: erase committed D-cache page members from N-cache |
| `MERIT_NCACHE_CAP_NODES` | Exact N-cache node cap for controlled experiments |
| `MERIT_DCACHE_NODE_CAP` / `MERIT_DCACHE_CAP` | Exact D-cache node/page caps |

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