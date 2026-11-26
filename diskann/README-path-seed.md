# Path-seed cache

On Linux, enable the path cache with a positive `NEIGHBORCACHE_NCACHE_CAP_NODES` or a positive `--neighborcache_memory_gb` budget. Use a fresh result prefix for each run. The writer creates `<prefix>_pathseed.data` exclusively.

At query completion, expanded parents at depth 3 or greater are grouped with their expanded children. Up to three consecutive new seeds in expansion order share a 4 KiB page if the union fits (at most ten records, including the seeds). Later queries can append members; a full page spills into an additional page owned by the seed receiving the new children. Lookup checks N-cache, then checks the seed and parent page directories. Reading a page makes its other members available in the prefetch cache.

Set `NEIGHBORCACHE_SHARED_PAGE_CACHE_PAGES` for prefetch capacity (at least 64 pages), `NEIGHBORCACHE_DCACHE_WRITER_THREADS` for the number of background writers (default 4), and `MERIT_PATH_QUEUE_PAGES` for the producer queue limit. To stripe D pages, set `MERIT_D_ODD` to the second output file. For a pre-striped base graph, set `MERIT_GRAPH_LOGICAL` to the logical graph filename and `MERIT_GRAPH_EVEN` and `MERIT_GRAPH_ODD` to the compact even- and odd-sector files. Both use 4 KiB sectors.
