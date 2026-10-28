# Thread-private prefetch capacity sweep

20 query workers, CPUs 0–19. Each worker owns one LRU, page payloads and resident-node replica directory. Pages are retained across queries and the A→B boundary; there are zero query resets. Immutable D and seed map are shared. Each run starts with empty prefetch; sync + file-scoped POSIX_FADV_DONTNEED and mincore verification precede every run. One run per capacity.

SIFT100M A→B 40K queries; each phase is 10K uniform + corresponding 10K Zipf0.8, shuffled. K=10 L=128 beam=4, N=0 M=0, full profiled visited-seed immutable D (2,632,133 seeds, one 4KiB page each). Same executable and query order at every capacity. OpenMP dynamic scheduling may assign queries to different workers across runs.

Memory capacities below are page payload, excluding metadata and other index memory. Total payload = 20 × per-thread payload. The x-axis shows nominal node-record capacity per thread (pages × 10), using actual numeric spacing and K = 1000. This is capacity, not measured distinct-node occupancy; partial pages and replicated records can reduce occupancy.

| Per-thread MiB | Total MiB | QPS | Prefetch hit % | D reads/query |
|---:|---:|---:|---:|---:|
| 0.402344 | 8.04688 | 8520.33 | 58.29 | 60.65 |
| 1 | 20 | 8590.80 | 58.97 | 59.66 |
| 2 | 40 | 8625.36 | 59.66 | 58.67 |
| 4 | 80 | 8569.28 | 60.51 | 57.43 |
| 8 | 160 | 8764.71 | 61.44 | 56.07 |
| 16 | 320 | 8823.39 | 62.54 | 54.48 |
| 24 | 480 | 8856.50 | 63.33 | 53.32 |
| 32 | 640 | 8911.46 | 63.87 | 52.54 |
| 48 | 960 | 9046.36 | 64.82 | 51.16 |
| 64 | 1280 | 9123.80 | 65.46 | 50.23 |

First to last capacity: QPS +7.08%; hit ratio +7.16 percentage points.

Number of full target-node accesses served from this worker prefetch / all full target-node accesses; insertion-causing demand is a miss. N=0; D disk reads are not memory hits.

All 40K×10 returned IDs match the existing reference for every run. Per-query full target counts are unchanged, and hit + D-read counts equal target accesses. These are reference-ID checks, not a new ground-truth recall measurement. Clear logs confirm zero file residency before each run. Private-LRU count and exact capacity per worker are asserted from the run log.

Curves use raw measurements without smoothing, interpolation-generated points or timing adjustment. One run per point provides no uncertainty interval; small local QPS variations cannot establish a causal effect. A larger retained cache can provide cross-query reuse even after the within-query working set fits.

Source executable: /mnt/nvme2/nc_thread_private_20261028/search_disk_index. See method.json and runs/*/command.json for full settings, hashes and affinity observations. Raw query stats, timeline and returned IDs are retained for audit.
