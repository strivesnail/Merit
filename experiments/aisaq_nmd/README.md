# AiSAQ with the shared NeighborCache backend

Pinned upstream: `https://github.com/kioxia-jp/aisaq-diskann`, commit
`32881764fad3d2a56b337f17d26cbdd2dca4b554`.

`aisaq.patch` contains the host search/reader integration. `nmd_cache.h` is only a
record/configuration adapter. Path pages and the node cache are provided by the
[shared backend](../../neighborcache_backend/README.md).

Install into a clean checkout of the pinned revision:

```sh
git -C /path/to/aisaq apply /path/to/Merit/experiments/aisaq_nmd/aisaq.patch
cp /path/to/Merit/experiments/aisaq_nmd/nmd_cache.h /path/to/aisaq/include/
cp /path/to/Merit/neighborcache_backend/include/nc_backend.h /path/to/aisaq/include/
cp /path/to/Merit/neighborcache_backend/include/nc_client.h /path/to/aisaq/include/
```

Build the shared backend, rebuild AiSAQ's `search_disk_index`, and set
`NEIGHBORCACHE_BACKEND_LIBRARY` to the resulting library. Use `AISAQ_NMD=1`,
`AISAQ_NMD_NMB`, `AISAQ_NMD_PMB`, `AISAQ_NMD_PATH` and
`NEIGHBORCACHE_THREADS`. Full inline PQ and no ID rearrangement are required;
current validation disables AiSAQ's native node cache and uses unlimited search
I/O budget. The record adapter preserves inline PQ bytes and the 32-bit degree.
