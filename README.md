# MERIT NeighborCache

MERIT builds 4 KiB cache pages from the nodes actually expanded by a query. DiskANN and the shared backend use the same node cache and path-page implementation. PipeANN, AiSAQ, Starling and MARGO keep their own search algorithms.

- The node cache uses 64 LRU shards and a counting Bloom filter.
- Starting at depth three (the medoid is depth zero), an expanded parent and its expanded children form a seed group. Up to three consecutive new groups can share one physical page when their distinct records fit. Later queries can add children to an existing seed.
- A node-cache miss probes the node's own seed and the discovery parent's seed. Resident page images are shared across workers. Demand reads fetch a D page or fall back to the original index. Four background workers write new and updated pages by default.

## Build on Linux

```sh
cmake -S diskann -B diskann/build -DCMAKE_BUILD_TYPE=Release
cmake --build diskann/build -j --target search_disk_index build_disk_index
cmake -S neighborcache_backend -B neighborcache_backend/build -DCMAKE_BUILD_TYPE=Release
cmake --build neighborcache_backend/build -j
ctest --test-dir neighborcache_backend/build --output-on-failure
```

DiskANN supports a baseline search with zero NC capacity. For a cached search, set `NEIGHBORCACHE_NCACHE_CAP_NODES` to a positive node count and use a fresh result prefix. PipeANN loads the shared library when `PIPEANN_NMD=1`.

See the [DiskANN dependencies](diskann/README.md), [cache configuration](diskann/README-path-seed.md), [engine integration](neighborcache_backend/README.md) and [experiment ledger](experiments/neighborcache_alignment/EXPERIMENTS.md). Historical comparisons in the ledger identify the implementation measured in each run.

The [pre-cleanup SIFT100M measurements](https://github.com/strivesnail/Merit/blob/4efb25d/README.md#sift100m-moving-hotspot-result) remain available with the code used for those runs.
