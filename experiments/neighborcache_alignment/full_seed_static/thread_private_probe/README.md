# Thread-private prefetch experiment

Artifact root: `/mnt/nvme2/nc_thread_private_20261028`.

This isolated snapshot derives from `/mnt/nvme2/nc_query_reset_20261028`. The included header and driver patch reproduce the isolated implementation; `pq_flash_index.cpp` is unchanged from that recorded snapshot. `python build.py --base /mnt/nvme2/nc_query_reset_20261028 --output /tmp/nc_fullseed_thread_private_build` stages those sources, applies the driver patch, compiles both translation units and links the snapshot library. The base snapshot, its dependency libraries and data are local experiment artifacts, not bundled in Git. This isolated experiment does not modify production cache files; the later general backend integration is documented in `../../thread_private/README.md`.

Environment flags:
- `NC_FULLSEED_THREAD_PRIVATE=1`: divide total `NC_FULLSEED_PREFETCH_PAGES` exactly among the OpenMP query workers, each with its own page payloads, replica directory and single LRU. No locks on the private cache path.
- `NC_FULLSEED_RESET_EACH_QUERY=1`: clear only the calling worker's cache before its next query. For shared mode, resetting still requires a single query thread.

Private mode assumes one non-nested OpenMP query team, as used by this driver; unrelated concurrent teams are not supported. D data and the immutable seed map remain shared. On-demand reads are performed on the query worker before its private-cache insertion.

Compile tests with `g++ -std=c++17 -O2 -pthread -fopenmp`; run `test_private` modes 0,1,2 and `test_thread_shards` modes 0,1,2,3.

See REPORT.md for measured results, checks and comparison limits.
