# DiskANN Re-layout Experiment (SIFT100K)

Edge-count-guided page relayout experiment for Jiang's immediate results track.

## Build

```bash
cd diskann
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

New tools: `search_disk_index` (with profiling flags), `relayout_disk_index`, `apply_disk_permutation`.

## Three-run design

| Run | Index | Profiling | Purpose |
|-----|-------|-----------|---------|
| Run1 | Original | OFF | Baseline IO / QPS / recall |
| Run2 | Original | ON | Node/edge counts + CDF |
| offline | — | — | Jiang k-hop relayout → rewrite index |
| Run3 | Re-layout | OFF | Compare IO vs Run1 |

Fair comparison: **Run1 vs Run3 only**. Use `num_nodes_to_cache=0`.

## Profiling (Run2)

```bash
search_disk_index ... \
  --enable_access_profile \
  --access_profile_prefix /path/to/run2_profile
```

Outputs:
- `run2_profile_node_expand.bin`
- `run2_profile_edge_{u,v,count}.bin`

## Offline relayout

```bash
relayout_disk_index \
  --mem_index sift100k_index_mem.index \
  --profile_prefix /path/to/run2_profile \
  --output_order /path/to/relayout_order \
  --k_hops 2

apply_disk_permutation \
  --data_type float \
  --base_file sift_base.bin \
  --mem_index sift100k_index_mem.index \
  --index_prefix sift100k_index \
  --order_file /path/to/relayout_order \
  --output_prefix sift100k_relayout_index
```

## Full pipeline script

Edit paths in `run_sift100k.sh`, then:

```bash
chmod +x experiments/relayout/run_sift100k.sh
DATA_DIR=/your/sift100k experiments/relayout/run_sift100k.sh
```

## SIFT1M scripts (MERIT / IO)

Benchmarks and MERIT tests are in [`../test-scripts/`](../test-scripts/) (cold start, sector cache, memory/disk tiers).  
`experiments/relayout/run_sift1m_*_compare.sh` are thin forwarders to that directory.

```bash
bash experiments/test-scripts/run_sift1m_merit_combined.sh
bash experiments/test-scripts/run_sift1m_cold_compare.sh
```

## Edge counting semantics

- **Node expand**: counted when a node is selected for expansion in beam search.
- **Directed edge** `u→v`: counted when `v` is expanded and `u` was the first discoverer (`parent[v]=u`) in that query.

Undirected weight for relayout: `I(u,v) = C(u→v) + C(v→u)`.

## Not in this README

- MERIT is documented in the repo root README (runtime memory + disk sidecar).
- EdgeAccessSet / medoid partitioning
- Search algorithm changes beyond DiskANN baseline beam search
