# Path-seed cache

The Linux disk index search supports an online path cache with `MERIT_PATH_SEED=shared`. Enable the NeighborCache N-cache, runtime admission, and dynamic cache options in the search driver. Use a fresh result prefix for each run. The writer creates `<prefix>_pathseed.data` exclusively.

At query completion, expanded parents at depth 3 or greater are grouped with their expanded children. Up to three consecutive new seeds in expansion order share a 4 KiB page if the union fits (at most ten records, including the seeds). Later queries can append members; a full page spills into an additional page owned by the seed receiving the new children. Lookup checks N-cache, then checks the seed and parent page directories. Reading a page makes its other members available in the prefetch cache.

Set `NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES` for prefetch capacity, `NEIGHBORCACHE_DCACHE_WRITER_THREADS` for the number of background writers, and `MERIT_PATH_QUEUE_PAGES` for the producer queue limit. To stripe D pages, set `MERIT_D_ODD` to the second output file. For a pre-striped base graph, set `MERIT_GRAPH_LOGICAL` to the logical graph filename and `MERIT_GRAPH_EVEN` and `MERIT_GRAPH_ODD` to the compact even- and odd-sector files. Both use 4 KiB sectors.
