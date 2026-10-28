# AiSAQ with the shared NeighborCache backend

Pinned upstream: `https://github.com/kioxia-jp/aisaq-diskann`, commit
`32881764fad3d2a56b337f17d26cbdd2dca4b554`.

`aisaq.patch` contains the host search/reader integration. `nmd_cache.h` is only a
record/configuration adapter. The N/M/prefetch/D implementation now comes from
[`neighborcache_backend`](../../neighborcache_backend/README.md), which compiles
the actual NeighborCache code. The old standalone PipeANN-derived policy is gone.

Install into a clean checkout of the pinned revision:

```sh
git -C /path/to/aisaq apply /path/to/Merit/experiments/aisaq_nmd/aisaq.patch
cp /path/to/Merit/experiments/aisaq_nmd/nmd_cache.h /path/to/aisaq/include/
cp /path/to/Merit/neighborcache_backend/include/nc_backend.h /path/to/aisaq/include/
cp /path/to/Merit/neighborcache_backend/include/nc_client.h /path/to/aisaq/include/
```

Build the shared backend, rebuild AiSAQ's `search_disk_index`, and set
`NEIGHBORCACHE_BACKEND_LIBRARY` to the resulting library. Use `AISAQ_NMD=1`,
`AISAQ_NMD_PREFILL=0`, `AISAQ_NMD_NMB`, `AISAQ_NMD_PMB`, `AISAQ_NMD_PATH` and
`NEIGHBORCACHE_THREADS`. Full inline PQ and no ID rearrangement are required;
current validation disables AiSAQ's native node cache and uses unlimited search
I/O budget. The record adapter preserves inline PQ bytes and the 32-bit degree.

Current verification: 100K SIFT index; public 1K and repeated 5K queries;
K=10, L=64, W=4, V=1, 8 threads, N=1 MiB/P=1 MiB, cold online startup.
Both returned IDs and float distances match newly rerun NMD-off baselines
byte-for-byte, and every cache hit passes full-record verification.

Use `../neighborcache_alignment/validate_shared.py --engines aisaq` for the
current local test. Results are in
`/mnt/nvme2/nc_shared_backend_20261027/final`.

`historical/README.md` and the old trial/compare scripts describe earlier
standalone-cache experiments. Their QPS values, counter parsers and private
header tests do not describe this backend. `test_batch_reader.cpp` still targets
the host mixed-file reader, which retains completion checking and D-read fallback.
